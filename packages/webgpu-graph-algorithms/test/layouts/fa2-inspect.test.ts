/**
 * Per-kernel inspection parity (spec 11.9 item 2; contract 5.5 fa2-inspect.test.ts): debugRunStages("K2") then
 * inspect("force") equals the oracle's attraction stage; after "K3" the force stage and the epilogue's swing /
 * traction partials; after "K4" the state's controller fields; after "K5" the positions and the partials A / C;
 * after "toScene" the scene positions; and the K1 fold of iteration 2 -- each within the tolerance traced to
 * benchmarks/results/noise-floor.json. The first block pins the helper's pure arithmetic on hand-computed numbers.
 */

import type { GpuContext } from "../../src/context.js";
import type { ForceAtlas2TraceRecord } from "../../src/types/layout.js";
import {
    BASE_OPTIONS,
    captureAllStages,
    DISTRIBUTIONAL_FLOOR,
    distributionalError,
    distributionalValuesError,
    forceParityCases,
    maxAbsDiff,
    NETWORKX,
    NOISE_FIXTURES,
    noiseInputs,
    ORACLE_F64_CLASS,
    P3_TOLERANCE_CAPS,
    PAPER,
    type ParityGraph,
    paritySnapshot,
    pinIndex,
    pinMask,
    rel,
    STAGE_KERNEL,
    STAGE_KEYS,
    STAGE_TOLERANCE,
    stageError,
    stageReport,
    startPositions,
    traceError,
    traceValues,
} from "../helpers/fa2-parity.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { adapterClass, writeNoiseFixture } from "../helpers/noise-floor.js";
import { assertCheckPasses } from "../helpers/sabotage.js";
import type { OracleTraceRecord } from "../oracle/forceatlas2.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

// ---------------------------------------------------------------- the helper's pure arithmetic

function record(
    swing: number,
    traction: number,
    speed: number,
    speedEfficiency: number,
    meanDisplacement: number,
    settledCount: number,
): ForceAtlas2TraceRecord {
    return { swing, traction, speed, speedEfficiency, meanDisplacement, settledCount };
}

function oracleRecord(base: ForceAtlas2TraceRecord): OracleTraceRecord {
    return { ...base, rmsRadius: 1, layoutRadius: 2, centroid: [0, 0, 0] };
}

describe("fa2-parity helper (pure)", () => {
    it("rel: |a - e| / max(|e|, floor)", () => {
        expect(rel(1.5, 1, 1e-9)).toBeCloseTo(0.5, 12);
        expect(rel(0.001, 0, 1)).toBeCloseTo(0.001, 12);
        expect(rel(2, 2, 1e-9)).toBe(0);
        expect(Number.isNaN(rel(Number.NaN, 1, 1e-9))).toBe(true);
    });

    it("maxAbsDiff: the largest |a_i - b_i|", () => {
        expect(maxAbsDiff([1, 2, 3], [1, 2.5, 2])).toBe(1);
        expect(maxAbsDiff([], [])).toBe(0);
    });

    it("traceError: the worst relative error over the six fields of records [from, to); a missing record is Infinity", () => {
        const gpu = [record(100, 50, 1.5, 1.3, 0, 0), record(80, 40, 2, 1, 0.5, 0)];
        const oracle = [oracleRecord(record(100, 50, 1.5, 1.3, 0, 0)), oracleRecord(record(80, 40, 2.02, 1, 0.5, 0))];
        expect(traceError(gpu, oracle, 0, 1)).toBe(0);
        // speed 2 vs 2.02 -> 0.02 / 2.02
        expect(traceError(gpu, oracle, 0, 2)).toBeCloseTo(0.02 / 2.02, 12);
        // settledCount 1 vs 0 under the floor of 1 -> 1
        const settled = [record(100, 50, 1.5, 1.3, 0, 1)];
        expect(traceError(settled, oracle, 0, 1)).toBe(1);
        expect(traceError(gpu, oracle, 0, 3)).toBe(Number.POSITIVE_INFINITY);
    });

    it("traceValues: six values per record in field order", () => {
        const v = traceValues([record(1, 2, 3, 4, 5, 6), record(7, 8, 9, 10, 11, 12)], 0, 2);
        expect(Array.from(v)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        expect(Array.from(traceValues([record(1, 2, 3, 4, 5, 6)], 1, 1))).toEqual([]);
    });

    it("stageError: vector stages use the floored per-node metric (floor 1e-3 x max |e_j|), scalar stages the 1e-6-floored elementwise metric", () => {
        // two nodes: e = (1,0,0), (0,0,0); a differs on the second node by 1e-4 -> floored by 1e-3 x 1 -> 0.1
        const vector = stageError(true, [1, 0, 0, 1e-4, 0, 0], [1, 0, 0, 0, 0, 0]);
        expect(vector.rel).toBeCloseTo(0.1, 9);
        expect(vector.abs).toBeCloseTo(1e-4, 15);
        const scalar = stageError(false, [10, 2e-6], [10, 1e-6]);
        // the second entry: 1e-6 / max(1e-6, 1e-6) = 1
        expect(scalar.rel).toBeCloseTo(1, 9);
        expect(scalar.abs).toBeCloseTo(1e-6, 15);
    });

    it("distributionalError: |a - b| / max(|b|, 0.05) per non-histogram key, the nnBin* keys as one total-variation distance; a key on one side only is Infinity", () => {
        // stress 0.1 / 1 = 0.1; the one-bin histogram difference 0.01 counts 0.005 (half the L1 distance)
        expect(distributionalError({ stress: 1.1, nnBin0: 0.01 }, { stress: 1, nnBin0: 0.02 })).toBeCloseTo(0.1, 12);
        // a 0.2 node-mass shift between two bins is 0.2, whatever the bins' own fractions are
        expect(distributionalError({ nnBin0: 0.5, nnBin1: 0.5 }, { nnBin0: 0.3, nnBin1: 0.7 })).toBeCloseTo(0.2, 12);
        // a small metric under the floor: |0.01 - 0.02| / DISTRIBUTIONAL_FLOOR (0.05) = 0.2
        expect(DISTRIBUTIONAL_FLOOR).toBe(0.05);
        expect(distributionalError({ separation: 0.01 }, { separation: 0.02 })).toBeCloseTo(
            0.01 / DISTRIBUTIONAL_FLOOR,
            12,
        );
        expect(distributionalError({ stress: 1 }, { stress: 1, spread: 2 })).toBe(Number.POSITIVE_INFINITY);
        // the flat form needs one key per value
        expect(distributionalValuesError(["stress"], [1, 2], [1, 2])).toBe(Number.POSITIVE_INFINITY);
        expect(distributionalValuesError(["nnBin0", "nnBin1", "stress"], [0.4, 0.6, 3], [0.5, 0.5, 3])).toBeCloseTo(
            0.1,
            12,
        );
    });

    it("forceParityCases: 175 named cases (44 per-fixture variations, 128 karate law combinations, 3 karate specials), names unique", () => {
        const cases = forceParityCases();
        expect(cases).toHaveLength(175);
        expect(new Set(cases.map((c) => c.name)).size).toBe(175);
        expect(cases.filter((c) => c.pinned)).toHaveLength(8);
        expect(cases.filter((c) => c.unseeded)).toHaveLength(1);
        expect(cases.filter((c) => c.graph === "isolated34")).toHaveLength(1);
        expect(cases.filter((c) => c.tuning.compat === "networkx")).toHaveLength(72);
        expect(cases.filter((c) => c.options.dim === 3)).toHaveLength(72);
    });

    it("the stage tables agree: every stage has an up-to name, a kernel, a tolerance id with a cap and a noise fixture", () => {
        expect(STAGE_KEYS).toEqual([
            "attraction",
            "force",
            "epilogue",
            "state",
            "positions",
            "partials",
            "scene",
            "k1",
        ]);
        for (const key of STAGE_KEYS) {
            expect(P3_TOLERANCE_CAPS[STAGE_TOLERANCE[key]], `${key}: cap`).toBeDefined();
            expect(NOISE_FIXTURES[key].fixture.startsWith("random1k-"), `${key}: fixture name`).toBe(true);
        }
        expect(NOISE_FIXTURES.trace10.kernel).toBe("fa2-speed-finalize");
        expect(NOISE_FIXTURES.metrics100.kernel).toBe("fa2-integrate");
        for (const [id, spec] of Object.entries(P3_TOLERANCE_CAPS)) {
            expect(spec.cap, `${id}: cap > 0`).toBeGreaterThan(0);
            expect(spec.basis.length, `${id}: basis`).toBeGreaterThan(0);
        }
    });
});

// ---------------------------------------------------------------- stage by stage on the device (spec 11.9 item 2)

const CASE_TIMEOUT = 300_000;
const GRAPHS: readonly ParityGraph[] = ["karate", "random1k"];

describe("FA2 inspect(): every stage against the oracle's (spec 11.9 item 2; G3)", () => {
    let ctx: GpuContext;

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-inspect" });
    });

    for (const graph of GRAPHS) {
        for (const tuning of [PAPER, NETWORKX]) {
            const label = `${graph}/${tuning.compat ?? "paper"}`;
            it(
                `${label}: K2, K3, the epilogue, K4, K5, toScene and the K1 fold within their traced tolerances, twice bitwise`,
                async (t) => {
                    requireGpu(t);
                    const s = paritySnapshot(graph, gpuScale(), false);
                    try {
                        const start = startPositions(s, BASE_OPTIONS, false);
                        const a = await captureAllStages(ctx, s, start, BASE_OPTIONS, tuning, null);
                        const b = await captureAllStages(ctx, s, start, BASE_OPTIONS, tuning, null);
                        for (const key of STAGE_KEYS) {
                            expectBitwiseEqual(a[key].values, b[key].values, `${label}/${key}: run 1 vs run 2`);
                            const report = stageReport(a, key);
                            console.warn(
                                `[fa2-inspect] ${label}/${key} (${STAGE_KERNEL[key]}): error ${a[key].error.toExponential(3)}, ratio ${report.worst.toExponential(3)}`,
                            );
                            assertCheckPasses(report);
                        }
                        // structure that needs no tolerance: the iteration counter after one real step and the debug K1, the free count, a positive speed
                        expect(a.k1.values[7], `${label}: S.iteration after step(1) + K1`).toBe(2);
                        expect(a.partials.values[12], `${label}: free count`).toBe(s.nodeCount);
                        expect(a.state.values[0], `${label}: speed`).toBeGreaterThan(0);
                        // K2 writes attraction only: on a node with no arcs (none in these graphs) it would be 0; on every node here it is finite
                        expect(a.attraction.values.every((v) => Number.isFinite(v))).toBe(true);
                    } finally {
                        ctx.release(s);
                    }
                },
                CASE_TIMEOUT,
            );
        }
    }

    it(
        "a pinned node (paper mode, karate): its force is computed, the free count excludes it, K5 leaves it in place, every stage still within tolerance",
        async (t) => {
            requireGpu(t);
            const s = paritySnapshot("karate", 1, false);
            try {
                const pinned = pinIndex(s.nodeCount);
                const mask = pinMask(s.nodeCount, pinned);
                const start = startPositions(s, BASE_OPTIONS, false);
                const a = await captureAllStages(ctx, s, start, BASE_OPTIONS, PAPER, mask);
                for (const key of STAGE_KEYS) {
                    assertCheckPasses(stageReport(a, key));
                }
                expect(a.partials.values[12], "free count").toBe(s.nodeCount - 1);
                expect(
                    Math.hypot(
                        a.force.values[3 * pinned],
                        a.force.values[3 * pinned + 1],
                        a.force.values[3 * pinned + 2],
                    ),
                    "the pinned node's force",
                ).toBeGreaterThan(0);
                for (let k = 0; k < 3; k++) {
                    expect(a.positions.values[3 * pinned + k], `pinned position component ${k}`).toBe(
                        k === 2 ? 0 : start[3 * pinned + k],
                    );
                }
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );

    it(
        "writes this adapter's stage outputs of the UNSCALED random1k and the f64 reference as noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const { s, start, options, tuning } = noiseInputs();
            try {
                const capture = await captureAllStages(ctx, s, start, options, tuning, null);
                const cls = adapterClass(ctx.caps);
                // the raw outputs are written BEFORE the checks (never hand-written; a floor above the cap surfaces
                // through the noise-floor validation, not through missing fixtures)
                for (const key of STAGE_KEYS) {
                    const name = NOISE_FIXTURES[key];
                    writeNoiseFixture(name.kernel, name.fixture, cls, capture[key].values, "f32");
                    writeNoiseFixture(name.kernel, name.fixture, ORACLE_F64_CLASS, capture[key].expected, "f32");
                }
                for (const key of STAGE_KEYS) {
                    const report = stageReport(capture, key);
                    console.warn(
                        `[fa2-inspect] noise/random1k/${key} (${STAGE_KERNEL[key]}): error ${capture[key].error.toExponential(3)}, ratio ${report.worst.toExponential(3)}`,
                    );
                    assertCheckPasses(report);
                }
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );
});
