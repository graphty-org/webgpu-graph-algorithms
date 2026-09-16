/**
 * The reduce primitive (spec 6 row 1, 11.3; contract 3.11, 5.5): sum / min / max x f32 / u32 / vec4f over sizes
 * 0, 1, 255, 256, 257, 4097 and 65,536 x 256 + 1 (three levels on hardware; scaled by gpuScale) vs the f64 oracle
 * (u32 exact; f32 within count x 2^-24 relative), both twins in-process within 1e-6 (u32 bitwise), two runs bitwise
 * identical, lastDispatches 2 or 3, the FINAL level touching exactly one output element, E_INVALID_ARGUMENT for a
 * src / out too small, and the f32 sum over random1k as a cross-adapter noise fixture with the twin and cross rows
 * (under GRAPHTY_GPU_NO_SUBGROUPS=1 the run is the class's twin: it compares within the floor and writes nothing).
 */

import { existsSync } from "node:fs";

import { MAX_WORKGROUPS_PER_DIM } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { hasErrorCode } from "../../src/errors.js";
import { prepareReduce, type ReduceDtype, type ReduceOp } from "../../src/primitives/reduce.js";
import { bindingOf, uploadBuffer } from "../helpers/device.js";
import { expectAllClose, expectBitwiseEqual, maxRelError } from "../helpers/matchers.js";
import {
    adapterClass,
    noiseFixturePath,
    noiseFloorFor,
    type NoiseRow,
    readNoiseFixtures,
    recordNoiseRow,
    writeNoiseFixture,
} from "../helpers/noise-floor.js";
import {
    caseLabel,
    OUT_POISON,
    paddedInput,
    type ReduceCase,
    reduceCaseReport,
    type ReduceRun,
    reduceScopeOf,
    runReduce,
    type TestReduceScope,
} from "../helpers/reduce-check.js";
import {
    INPUT_SEED,
    lanesOf,
    RANDOM1K_COUNT,
    RANDOM1K_SEED,
    REDUCE_NOISE_COUNTS,
    reduceInput,
} from "../helpers/reduce-input.js";
import { assertCheckPasses } from "../helpers/sabotage.js";
import { reduceIdentity, reduceOracle } from "../oracle/reduce.js";
import { acquire, adapterSummary, gpuScale, requireGpu } from "../setup/gpu.js";

const OPS: readonly ReduceOp[] = ["sum", "min", "max"];
const DTYPES: readonly ReduceDtype[] = ["f32", "u32", "vec4f"];
/** Each adapter is within count x 2^-24 of the exact sum, so two adapters are within twice that. */
const REDUCE_CROSS_ADAPTER_BOUND = 2 * RANDOM1K_COUNT * 2 ** -24;

/** The 11.3 sizes; only the largest is scaled (the boundary sizes must stay exact). */
function reduceSizes(): number[] {
    return [0, 1, 255, 256, 257, 4097, Math.ceil((65_536 * 256 + 1) * gpuScale())];
}

function expectedDispatches(ctx: GpuContext, count: number): number {
    return Math.ceil(count / ctx.workgroupSize) > MAX_WORKGROUPS_PER_DIM ? 3 : 2;
}

function asList(v: number | readonly [number, number, number, number]): readonly number[] {
    return typeof v === "number" ? [v] : v;
}

function scalarOf(v: number | readonly [number, number, number, number]): number {
    if (typeof v !== "number") {
        throw new Error("expected a scalar result");
    }
    return v;
}

function expectTwinAgrees(
    a: number | readonly [number, number, number, number],
    b: number | readonly [number, number, number, number],
    op: ReduceOp,
    dtype: ReduceDtype,
    label: string,
): void {
    if (dtype === "u32" || op !== "sum") {
        // u32 sums are exact; min / max return an input element
        expect(a, label).toEqual(b);
        return;
    }
    expectAllClose(asList(a), asList(b), { rel: 1e-6, abs: 0 }, label);
}

function expectOnlyTheElementWritten(run: ReduceRun, dtype: ReduceDtype, outOffset: number, label: string): void {
    const lanes = lanesOf(dtype);
    for (let i = 0; i < run.words.length; i++) {
        const inside = i >= outOffset * lanes && i < (outOffset + 1) * lanes;
        if (!inside) {
            expect(run.words[i], `${label}: word ${i} outside out[${outOffset}] must keep the poison`).toBe(OUT_POISON);
        }
    }
}

describe("reduce (spec 6 row 1): equals the f64 oracle, twins agree, twice bitwise", () => {
    for (const op of OPS) {
        for (const dtype of DTYPES) {
            it(`${op} / ${dtype} over sizes 0, 1, 255, 256, 257, 4097 and the scaled 65536 x 256 + 1`, async (t) => {
                requireGpu(t);
                const ctx = await acquire();
                const twin = await acquire({ subgroups: false });
                try {
                    let twinWorstRel = 0;
                    for (const count of reduceSizes()) {
                        const c: ReduceCase = { op, dtype, count, outOffset: count % 4 };
                        const label = caseLabel(c);
                        const values = reduceInput(dtype, count, INPUT_SEED + count);
                        const words = paddedInput(values, dtype, op);
                        const src = uploadBuffer(ctx, words, `reduce/src/${label}`);
                        const srcTwin = uploadBuffer(twin, words, `reduce/src-twin/${label}`);
                        try {
                            const first = await reduceCaseReport(ctx, c, src, values);
                            assertCheckPasses(first.report);
                            expect(first.run.dispatches, `${label} dispatches`).toBe(expectedDispatches(ctx, count));
                            expectOnlyTheElementWritten(first.run, dtype, c.outOffset, label);
                            if (count === 0) {
                                const identity = reduceIdentity(op, dtype);
                                expect(first.run.value).toEqual(
                                    dtype === "vec4f" ? [identity, identity, identity, identity] : identity,
                                );
                            }
                            if (dtype === "u32" || op !== "sum") {
                                expect(first.run.value, `${label} exact`).toEqual(reduceOracle(values, op, dtype));
                            }
                            const second = await runReduce(ctx, op, dtype, src, count, c.outOffset);
                            expectBitwiseEqual(first.run.words, second.words, `${label} twice`);
                            const other = await runReduce(twin, op, dtype, srcTwin, count, c.outOffset);
                            expectTwinAgrees(first.run.value, other.value, op, dtype, `${label} twin`);
                            if (op === "sum" && dtype !== "u32") {
                                twinWorstRel = Math.max(
                                    twinWorstRel,
                                    maxRelError(asList(first.run.value), asList(other.value), 1),
                                );
                            }
                        } finally {
                            src.destroy();
                            srcTwin.destroy();
                        }
                    }
                    if (op === "sum" && dtype === "f32") {
                        expect(twinWorstRel).toBeLessThanOrEqual(1e-6);
                    }
                } finally {
                    twin.dispose();
                    ctx.dispose();
                }
            });
        }
    }

    it("rejects a src smaller than count x element size, an out that cannot hold out[outOffset], and a bad count / outOffset", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const held: TestReduceScope = reduceScopeOf(ctx);
        const small = uploadBuffer(ctx, new Uint32Array(8), "reduce/small");
        const out = uploadBuffer(ctx, new Uint32Array(4), "reduce/out");
        try {
            const u32 = await prepareReduce(held.scope, "sum", "u32");
            const vec = await prepareReduce(held.scope, "sum", "vec4f");
            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            const codeOf = (fn: () => void): string | null => {
                try {
                    fn();
                    return null;
                } catch (e) {
                    return hasErrorCode(e, "E_INVALID_ARGUMENT") ? "E_INVALID_ARGUMENT" : String(e);
                }
            };
            expect(codeOf(() => u32.record(pass, bindingOf(small), 9, bindingOf(out), 0))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => vec.record(pass, bindingOf(small), 3, bindingOf(out), 0))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => u32.record(pass, bindingOf(small), 8, bindingOf(out), 4))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => vec.record(pass, bindingOf(small), 2, bindingOf(out), 1))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => u32.record(pass, bindingOf(small), -1, bindingOf(out), 0))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => u32.record(pass, bindingOf(small), 1.5, bindingOf(out), 0))).toBe("E_INVALID_ARGUMENT");
            expect(codeOf(() => u32.record(pass, bindingOf(small), 8, bindingOf(out), -1))).toBe("E_INVALID_ARGUMENT");
            // the boundary cases are accepted (8 elements in 32 bytes; out[3] is the last of 4 words)
            expect(codeOf(() => u32.record(pass, bindingOf(small), 8, bindingOf(out), 3))).toBeNull();
            expect(u32.lastDispatches).toBe(2);
            pass.end();
        } finally {
            held.release();
            small.destroy();
            out.destroy();
            ctx.dispose();
        }
    });

    it("writes the f32 sum over random1k as this adapter's noise fixture, records the twin and cross rows, and agrees with every committed adapter within the derived bound", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const twin = await acquire({ subgroups: false });
        const values = reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED);
        if (!(values instanceof Float32Array)) {
            throw new Error('reduceInput("f32") must return a Float32Array');
        }
        const words = paddedInput(values, "f32", "sum");
        const src = uploadBuffer(ctx, words, "reduce-noise/src");
        const srcTwin = uploadBuffer(twin, words, "reduce-noise/src-twin");
        try {
            const sums = new Float32Array(REDUCE_NOISE_COUNTS.length);
            const twinSums = new Float32Array(REDUCE_NOISE_COUNTS.length);
            for (const [j, count] of REDUCE_NOISE_COUNTS.entries()) {
                const run = await runReduce(ctx, "sum", "f32", src, count, 0);
                sums[j] = scalarOf(run.value);
                const exact = scalarOf(reduceOracle(values.subarray(0, count), "sum", "f32"));
                expect(Math.abs(sums[j] - exact) / exact, `sum of ${count}`).toBeLessThanOrEqual(count * 2 ** -24);
                twinSums[j] = scalarOf((await runReduce(twin, "sum", "f32", srcTwin, count, 0)).value);
            }
            const cls = adapterClass(ctx.caps);
            // GRAPHTY_GPU_NO_SUBGROUPS=1 on a subgroup-capable adapter (the default lane's second pass): this run IS
            // the workgroup-memory twin of the class, so it neither writes the class's fixture / rows nor matches the
            // committed subgroup-path output bitwise -- it agrees with it within the derived floor (contract 5.5),
            // exactly as another adapter would.
            const suppressedTwin =
                (adapterSummary()?.features.includes("subgroups") ?? false) && !ctx.caps.features.has("subgroups");
            const twinRel = maxRelError(sums, twinSums, 1);
            expect(twinRel).toBeLessThanOrEqual(1e-6);
            expect(sums.length).toBe(11);
            const twinRow: NoiseRow = {
                id: "reduce.sum.twin",
                kernel: "reduce",
                fixture: "random1k",
                comparison: "twin",
                a: cls,
                b: `${cls}/no-subgroups`,
                maxRelError: twinRel,
                maxAbsError: Math.max(...Array.from(sums, (v, j) => Math.abs(v - twinSums[j]))),
                samples: sums.length,
            };
            if (!suppressedTwin) {
                recordNoiseRow(twinRow);
                writeNoiseFixture("reduce", "random1k", cls, sums, "f32");
                if (process.env.GRAPHTY_NOISE_FLOOR_WRITE === "1") {
                    expect(existsSync(noiseFixturePath("reduce", "random1k", cls))).toBe(true);
                }
            }
            let worstRel = 0;
            let worstAbs = 0;
            let worstClass: string | null = null;
            for (const other of readNoiseFixtures("reduce", "random1k")) {
                expect(other.dtype, other.adapterClass).toBe("f32");
                expect(other.values.length, other.adapterClass).toBe(sums.length);
                if (other.adapterClass === cls && !suppressedTwin) {
                    expectBitwiseEqual(Float32Array.from(other.values), sums, `committed ${cls} vs this run`);
                    continue;
                }
                const rel = maxRelError(sums, other.values, 1);
                expect(rel, `reduce random1k: ${cls} vs ${other.adapterClass}`).toBeLessThanOrEqual(
                    REDUCE_CROSS_ADAPTER_BOUND,
                );
                if (rel >= worstRel) {
                    worstRel = rel;
                    worstAbs = Math.max(...Array.from(sums, (v, j) => Math.abs(v - other.values[j])));
                    worstClass = other.adapterClass;
                }
            }
            if (worstClass !== null && !suppressedTwin) {
                const row: NoiseRow = {
                    id: "reduce.sum.cross",
                    kernel: "reduce",
                    fixture: "random1k",
                    comparison: "cross-adapter",
                    a: cls,
                    b: worstClass,
                    maxRelError: worstRel,
                    maxAbsError: worstAbs,
                    samples: sums.length,
                };
                recordNoiseRow(row);
            }
        } finally {
            src.destroy();
            srcTwin.destroy();
            twin.dispose();
            ctx.dispose();
        }
    });

    it("noiseFloorFor rejects an id without a committed floor (a tolerance without a floor is a finding)", () => {
        expect(() => noiseFloorFor("reduce.no-such-tolerance")).toThrow(/noise floor/);
    });
});
