/**
 * The thread-per-row segmentedReduce differential test (contract 5.5 P2; spec 6 row 3, 11.3, 11.9 item 4): the
 * oracle's hand-computed pins, every named fixture and a row-count ladder vs segmentedReduceOracle for sum / min /
 * max with `v = weight;` and `v = 1.0;`, empty rows and arcCount 0 writing the identity element, the accumulate
 * mode, both twins in-process, two runs bitwise, the P2 rejections (tiers, windowed cores, bad snippets), and the
 * weighted f32 sum over random1k as the cross-adapter noise fixture compared within the derived floor.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { type TestContext } from "vitest";

import { MAX_1D_ITEMS } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import { type CoreBinding } from "../../src/memory/residency.js";
import { type ReduceOp } from "../../src/primitives/reduce.js";
import {
    type DegreeTiers,
    prepareSegmentedReduce,
    type SegmentedReduceOptions,
    type SegmentedReducePlanner,
} from "../../src/primitives/segmented-reduce.js";
import { withContext } from "../helpers/device.js";
import { fixture, pathEdges, randomEdges, snapshotOf, starEdges } from "../helpers/graphs.js";
import { expectAllClose, expectBitwiseEqual, maxRelError } from "../helpers/matchers.js";
import {
    adapterClass,
    noiseFloorFor,
    readNoiseFixtures,
    recordNoiseRow,
    writeNoiseFixture,
} from "../helpers/noise-floor.js";
import { SEGMENTED_REDUCE_SNIPPETS } from "../helpers/override-matrix.js";
import {
    maxAbsError,
    oracleValueOf,
    relTolerance,
    runSegmentedReduce,
    sabotageChecks,
    SR_ABS_FLOOR,
    testReduceScope,
    weightedRandom,
    worstFactor,
} from "../helpers/segmented-reduce.js";
import { F32_MAX, segmentedReduceOracle } from "../oracle/segmented-reduce.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

const OPS: readonly ReduceOp[] = ["sum", "min", "max"];
const SNIPPETS: readonly string[] = [SEGMENTED_REDUCE_SNIPPETS.weight, SEGMENTED_REDUCE_SNIPPETS.one];
const WEIGHT = SEGMENTED_REDUCE_SNIPPETS.weight;
const ONE = SEGMENTED_REDUCE_SNIPPETS.one;
const NAMED_FIXTURES = [
    "one",
    "self-loop",
    "karate",
    "grid10",
    "path1k",
    "star200",
    "complete6",
    "random1k",
    "hub10k",
    "isolated",
    "parallel",
];

/** Awaits a rejection and asserts its code; returns the error for detail assertions. */
async function expectRejection(promise: Promise<unknown>, code: string): Promise<WebGpuGraphError> {
    let caught: unknown = null;
    try {
        await promise;
    } catch (err) {
        caught = err;
    }
    expect(isWebGpuGraphError(caught)).toBe(true);
    const error = caught as WebGpuGraphError;
    expect(error.code).toBe(code);
    return error;
}

/** Runs fn and returns the error it throws, asserting the code. */
function expectThrow(fn: () => unknown, code: string): WebGpuGraphError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(isWebGpuGraphError(caught)).toBe(true);
    const error = caught as WebGpuGraphError;
    expect(error.code).toBe(code);
    return error;
}

describe("segmentedReduceOracle (pure)", () => {
    // path 0-1-2-3 with weights 1.5, 2.5, 4.0 and a fifth node without arcs: rows 0 -> [1], 1 -> [0, 2], 2 -> [1, 3],
    // 3 -> [2], 4 -> []; arcs 0..5 in that order
    const s = snapshotOf(
        [
            [0, 1, 1.5],
            [1, 2, 2.5],
            [2, 3, 4.0],
        ],
        { nodeCount: 5, weighted: true },
    );

    it("sum / min / max of the weights per row; the empty row gets the kernel's identity element", () => {
        expect(Array.from(segmentedReduceOracle(s, (_row, _arc, _target, weight) => weight, "sum"))).toEqual([
            1.5, 4.0, 6.5, 4.0, 0,
        ]);
        expect(Array.from(segmentedReduceOracle(s, (_row, _arc, _target, weight) => weight, "min"))).toEqual([
            1.5,
            1.5,
            2.5,
            4.0,
            F32_MAX,
        ]);
        expect(Array.from(segmentedReduceOracle(s, (_row, _arc, _target, weight) => weight, "max"))).toEqual([
            1.5,
            2.5,
            4.0,
            4.0,
            -F32_MAX,
        ]);
    });

    it("v = 1 is the out-degree; row, arc and target are the CSR entries in arc order", () => {
        expect(Array.from(segmentedReduceOracle(s, () => 1, "sum"))).toEqual([1, 2, 2, 1, 0]);
        expect(Array.from(segmentedReduceOracle(s, (_row, _arc, target) => target, "sum"))).toEqual([1, 2, 4, 2, 0]);
        expect(Array.from(segmentedReduceOracle(s, (_row, arc) => arc, "max"))).toEqual([0, 2, 4, 5, -F32_MAX]);
        expect(Array.from(segmentedReduceOracle(s, (row) => row, "min"))).toEqual([0, 1, 2, 3, F32_MAX]);
    });

    it("an unweighted snapshot passes weight 1; the fold is f64 (no f32 rounding of the sum)", () => {
        const unweighted = snapshotOf(pathEdges(4));
        expect(Array.from(segmentedReduceOracle(unweighted, (_row, _arc, _target, weight) => weight, "sum"))).toEqual([
            1, 2, 2, 1,
        ]);
        // 16777216 + 1 is not representable in f32 (it rounds to 16777216); the f64 oracle keeps 16777217
        const big = snapshotOf(
            [
                [0, 1, 16777216],
                [0, 2, 1],
            ],
            { weighted: true },
        );
        expect(segmentedReduceOracle(big, (_row, _arc, _target, weight) => weight, "sum")[0]).toBe(16777217);
    });

    it("F32_MAX is the largest finite f32 (0x1.fffffep+127)", () => {
        expect(Math.fround(F32_MAX)).toBe(F32_MAX);
        expect(Number.isFinite(Math.fround(F32_MAX * 1.0000001))).toBe(false);
    });
});

describe("segmentedReduce thread-per-row (GPU)", () => {
    let shared: GpuContext | null = null;

    async function context(t: TestContext): Promise<GpuContext> {
        requireGpu(t);
        const ctx = shared ?? (await acquire({ label: "segmented-reduce" }));
        shared = ctx;
        return ctx;
    }

    /** sum / min / max x weight / one: twice bitwise, then against the f64 oracle within the analytic tolerance. */
    async function checkAgainstOracle(ctx: GpuContext, s: GraphSnapshot, label: string): Promise<void> {
        for (const op of OPS) {
            for (const snippet of SNIPPETS) {
                const first = await runSegmentedReduce(ctx, s, op, snippet);
                const second = await runSegmentedReduce(ctx, s, op, snippet);
                expectBitwiseEqual(first, second, `${label} ${op} ${snippet} twice`);
                const expected = segmentedReduceOracle(s, oracleValueOf(snippet), op);
                expect(first.length).toBe(expected.length);
                expectAllClose(first, expected, { rel: relTolerance(s, op), abs: 0 }, `${label} ${op} ${snippet}`);
            }
        }
    }

    it("n = 0: prepare succeeds, record records nothing, the result is empty", async (t) => {
        const ctx = await context(t);
        const { snapshot } = fixture("empty", gpuScale());
        expect(snapshot.nodeCount).toBe(0);
        for (const op of OPS) {
            const result = await runSegmentedReduce(ctx, snapshot, op, ONE);
            expect(result.length).toBe(0);
        }
        ctx.release(snapshot);
    });

    for (const name of NAMED_FIXTURES) {
        it(`matches the f64 oracle on ${name} (sum / min / max x weight / one), twice bitwise`, async (t) => {
            const ctx = await context(t);
            const { snapshot } = fixture(name, gpuScale());
            await checkAgainstOracle(ctx, snapshot, name);
            ctx.release(snapshot);
        });
    }

    it("a directed snapshot walks the out-rows", async (t) => {
        const ctx = await context(t);
        const s = snapshotOf(randomEdges(300, 900, 11), { directed: true, label: "directed-300" });
        await checkAgainstOracle(ctx, s, "directed");
        ctx.release(s);
    });

    it("all-equal weights (0.75 on every arc): the sum is d x 0.75 within the bound, min = max = 0.75 exactly", async (t) => {
        const ctx = await context(t);
        const edges = randomEdges(2000, 12000, 13).map(([u, v]): [number, number, number] => [u, v, 0.75]);
        const s = snapshotOf(edges, { weighted: true, label: "all-equal" });
        await checkAgainstOracle(ctx, s, "all-equal");
        const mins = await runSegmentedReduce(ctx, s, "min", WEIGHT);
        const maxs = await runSegmentedReduce(ctx, s, "max", WEIGHT);
        for (let row = 0; row < s.nodeCount; row++) {
            if (s.rowPtr[row + 1] > s.rowPtr[row]) {
                expect(mins[row]).toBe(0.75);
                expect(maxs[row]).toBe(0.75);
            }
        }
        ctx.release(s);
    });

    it("one hub row: a weighted star with 10k leaves (scaled)", async (t) => {
        const ctx = await context(t);
        const leaves = Math.max(2, Math.round(10_000 * gpuScale()));
        const star = snapshotOf(
            starEdges(leaves).map(([u, v], k): [number, number, number] => [u, v, Math.fround(0.5 + (k % 7) * 0.25)]),
            { weighted: true, label: `star-${leaves}` },
        );
        await checkAgainstOracle(ctx, star, "star-hub");
        let dmax = 0;
        for (let row = 0; row < star.nodeCount; row++) {
            dmax = Math.max(dmax, star.rowPtr[row + 1] - star.rowPtr[row]);
        }
        expect(dmax).toBe(leaves); // one row holds every leaf (the hub10k fixture runs in the named loop above)
        ctx.release(star);
    });

    it("empty rows and arcCount 0 get the identity element: F32_MAX under min, -F32_MAX under max, 0 under sum", async (t) => {
        const ctx = await context(t);
        const holes = weightedRandom(200, 600, 5, 300);
        const mins = await runSegmentedReduce(ctx, holes, "min", WEIGHT);
        const maxs = await runSegmentedReduce(ctx, holes, "max", ONE);
        const sums = await runSegmentedReduce(ctx, holes, "sum", WEIGHT);
        for (let row = 200; row < 300; row++) {
            expect(mins[row]).toBe(F32_MAX);
            expect(maxs[row]).toBe(-F32_MAX);
            expect(sums[row]).toBe(0);
        }
        ctx.release(holes);
        const bare = snapshotOf([], { nodeCount: 4097, label: "bare-4097" });
        expect(bare.arcCount).toBe(0);
        const bareMin = await runSegmentedReduce(ctx, bare, "min", ONE);
        const bareMax = await runSegmentedReduce(ctx, bare, "max", ONE);
        const bareSum = await runSegmentedReduce(ctx, bare, "sum", ONE);
        for (let row = 0; row < 4097; row++) {
            expect(bareMin[row]).toBe(F32_MAX);
            expect(bareMax[row]).toBe(-F32_MAX);
            expect(bareSum[row]).toBe(0);
        }
        ctx.release(bare);
    });

    it("row-count ladder 0, 1, 255, 256, 257, 4097, 65537 and 2^20 (scaled): weighted randoms vs the oracle", async (t) => {
        const ctx = await context(t);
        const ladder = [0, 1, 255, 256, 257, 4097, 65_537, Math.max(2, Math.round(2 ** 20 * gpuScale()))];
        for (const n of ladder) {
            const m = n >= 2 ? Math.min(2 * n, Math.floor((n * (n - 1)) / 2)) : 0;
            const s = weightedRandom(n, m, 100 + n);
            const result = await runSegmentedReduce(ctx, s, "sum", WEIGHT);
            const again = await runSegmentedReduce(ctx, s, "sum", WEIGHT);
            expectBitwiseEqual(result, again, `ladder ${n} twice`);
            expectAllClose(
                result,
                segmentedReduceOracle(s, oracleValueOf(WEIGHT), "sum"),
                { rel: relTolerance(s, "sum"), abs: 0 },
                `ladder ${n}`,
            );
            ctx.release(s);
        }
    }, 120_000);

    it("the 1D / 2D dispatch boundary: MAX_1D_ITEMS + 1 rows (scaled) with no arcs -> every row F32_MAX under min", async (t) => {
        const ctx = await context(t);
        const n = Math.max(1, Math.round((MAX_1D_ITEMS + 1) * gpuScale()));
        const s = snapshotOf([], { nodeCount: n, label: `rows-only-${n}` });
        const result = await runSegmentedReduce(ctx, s, "min", ONE);
        expect(result.length).toBe(n);
        let wrong = 0;
        for (let row = 0; row < n; row++) {
            if (result[row] !== F32_MAX) {
                wrong++;
            }
        }
        expect(wrong).toBe(0);
        for (const sample of [0, 1, 255, 256, MAX_1D_ITEMS - 1, MAX_1D_ITEMS, n - 1]) {
            if (sample < n) {
                expect(result[sample]).toBe(F32_MAX);
            }
        }
        ctx.release(s);
    }, 120_000);

    it("accumulate = true combines into out: the sum doubles, min / max are unchanged", async (t) => {
        const ctx = await context(t);
        const s = weightedRandom(500, 2000, 3);
        const base = await runSegmentedReduce(ctx, s, "sum", WEIGHT);
        const twice = await runSegmentedReduce(ctx, s, "sum", WEIGHT, { accumulate: true, initial: base });
        const doubled = segmentedReduceOracle(s, oracleValueOf(WEIGHT), "sum").map((x) => 2 * x);
        expectAllClose(twice, doubled, { rel: relTolerance(s, "sum") + 2 ** -23, abs: 0 }, "accumulated sum");
        for (const op of ["min", "max"] as const) {
            const once = await runSegmentedReduce(ctx, s, op, WEIGHT);
            const again = await runSegmentedReduce(ctx, s, op, WEIGHT, { accumulate: true, initial: once });
            expectBitwiseEqual(once, again, `accumulated ${op}`);
        }
        ctx.release(s);
    });

    it("USE_PERM is false in every P2 pipeline key; tiers !== null -> E_UNSUPPORTED { feature: segmentedReduce.tiers }", async (t) => {
        const ctx = await context(t);
        const { snapshot } = fixture("karate", gpuScale());
        await runSegmentedReduce(ctx, snapshot, "sum", ONE);
        const keys = ctx.pipelines.keys().filter((key) => key.startsWith("segmented-reduce|"));
        expect(keys.length).toBeGreaterThan(0);
        for (const key of keys) {
            const overrides = JSON.parse(key.split("|")[1]) as { USE_PERM?: boolean; TIER?: number };
            expect(overrides.USE_PERM).toBe(false);
            expect(overrides.TIER).toBe(0);
        }
        const core = ctx.residency.core(snapshot);
        const view = ctx.residency.view(snapshot, "degreeOrder");
        const so = view.scalars.segmentOffsets;
        expect(so.length).toBe(5);
        expect(so[4]).toBe(snapshot.nodeCount);
        const tiers: DegreeTiers = { perm: view.bindings.perm, segmentOffsets: [so[0], so[1], so[2], so[3], so[4]] };
        const scope = testReduceScope(ctx);
        const error = await expectRejection(
            prepareSegmentedReduce(scope, core, { op: "sum", valueSnippet: ONE, tiers }),
            "E_UNSUPPORTED",
        );
        expect(error.details.feature).toBe("segmentedReduce.tiers");
        scope.dispose();
        ctx.release(snapshot);
    });

    it("a windowed core -> E_UNSUPPORTED; a core of the other weights pattern, a malformed rowPtr or a short out -> E_INVALID_ARGUMENT at record", async (t) => {
        const ctx = await context(t);
        const weighted = weightedRandom(50, 100, 2);
        const unweighted = snapshotOf(randomEdges(50, 100, 2), { label: "unweighted-50" });
        const core = ctx.residency.core(weighted);
        const scope = testReduceScope(ctx);
        const windowed: CoreBinding = { ...core, plan: "windowed", windows: [] };
        const unsupported = await expectRejection(
            prepareSegmentedReduce(scope, windowed, { op: "sum", valueSnippet: ONE, tiers: null }),
            "E_UNSUPPORTED",
        );
        expect(unsupported.details.feature).toBe("segmentedReduce.windowed");
        const options: SegmentedReduceOptions = { op: "sum", valueSnippet: WEIGHT, tiers: null };
        const planner: SegmentedReducePlanner = await prepareSegmentedReduce(scope, core, options);
        const out = scope.scratch(4 * 50, "short-out");
        const encoder = ctx.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const other = ctx.residency.core(unweighted);
        const pattern = expectThrow(
            () => planner.record(pass, other, { buffer: out, offset: 0, size: 4 * 50, window: null }),
            "E_INVALID_ARGUMENT",
        );
        expect(pattern.details.argument).toBe("core");
        // a rowPtr binding whose byte size is not 4(n + 1): the row count cannot be derived, nothing is recorded
        const malformed: CoreBinding = { ...core, rowPtr: { ...core.rowPtr, size: 6 } };
        const rows = expectThrow(
            () => planner.record(pass, malformed, { buffer: out, offset: 0, size: 4 * 50, window: null }),
            "E_INVALID_ARGUMENT",
        );
        expect(rows.details.argument).toBe("core.rowPtr");
        expect(rows.details.value).toBe(6);
        const short = expectThrow(
            () => planner.record(pass, core, { buffer: out, offset: 0, size: 4, window: null }),
            "E_INVALID_ARGUMENT",
        );
        expect(short.details.argument).toBe("out");
        pass.end();
        scope.dispose();
        ctx.release(weighted);
        ctx.release(unweighted);
    });

    it("bad snippets -> E_SHADER_COMPILE { stage: compose, slot: VALUE }: target, a binding, a uniform, a foreign identifier, a flow keyword, no assignment to v", async (t) => {
        const ctx = await context(t);
        const { snapshot } = fixture("karate", gpuScale());
        const core = ctx.residency.core(snapshot);
        const scope = testReduceScope(ctx);
        // every case is rejected textually, before any shader is created: no pipeline key is logged for them
        const bad: readonly [string, string][] = [
            ["v = target;", "target"],
            ["v = weights[arc];", "weights"],
            ["v = out[row];", "out"],
            ["v = P.n;", "P"],
            ["v = xNorm[nbr];", "xNorm"],
            ["v = weight; return;", "return"],
            ["let w = weight; v = w;", "w"],
        ];
        for (const [snippet, identifier] of bad) {
            const error = await expectRejection(
                prepareSegmentedReduce(scope, core, { op: "sum", valueSnippet: snippet, tiers: null }),
                "E_SHADER_COMPILE",
            );
            expect(error.details.stage, snippet).toBe("compose");
            expect(error.details.slot, snippet).toBe("VALUE");
            expect(error.details.identifier, snippet).toBe(identifier);
        }
        for (const snippet of ["v == weight;", "// v = weight;", "weight;"]) {
            const error = await expectRejection(
                prepareSegmentedReduce(scope, core, { op: "sum", valueSnippet: snippet, tiers: null }),
                "E_SHADER_COMPILE",
            );
            expect(error.details.stage, snippet).toBe("compose");
            expect(error.details.slot, snippet).toBe("VALUE");
            expect(error.details.identifier, snippet).toBeUndefined();
        }
        scope.dispose();
        ctx.release(snapshot);
    });

    it("good snippets beyond the two plain ones: a multi-statement conditional (keywords are not identifiers) and a comment naming forbidden words both run and match the oracle", async (t) => {
        const ctx = await context(t);
        const s = weightedRandom(400, 1600, 17);
        for (const snippet of [SEGMENTED_REDUCE_SNIPPETS.conditional, SEGMENTED_REDUCE_SNIPPETS.commented]) {
            for (const op of OPS) {
                const actual = await runSegmentedReduce(ctx, s, op, snippet);
                const expected = segmentedReduceOracle(s, oracleValueOf(snippet), op);
                expectAllClose(actual, expected, { rel: relTolerance(s, op), abs: 0 }, `${op} ${snippet}`);
            }
        }
        // the conditional keeps only arcs into lower-numbered rows: row 0 has none, so its sum is exactly 0
        const lower = await runSegmentedReduce(ctx, s, "sum", SEGMENTED_REDUCE_SNIPPETS.conditional);
        expect(lower[0]).toBe(0);
        const explicit = segmentedReduceOracle(s, (row, _arc, target, weight) => (target < row ? weight : 0), "sum");
        expectAllClose(
            lower,
            explicit,
            { rel: relTolerance(s, "sum"), abs: 0 },
            "conditional vs the explicit callback",
        );
        ctx.release(s);
    });

    it("the sabotage check set passes on the real kernel with an error factor below 1", async (t) => {
        const ctx = await context(t);
        expect(await worstFactor(ctx, sabotageChecks())).toBeLessThan(1);
    });

    it("twins in-process: the no-subgroup context gives bitwise the same result (TIER 0 calls no reduction helper)", async (t) => {
        const ctx = await context(t);
        const s = weightedRandom(1000, 5000, 7);
        const withFeature = await runSegmentedReduce(ctx, s, "sum", WEIGHT);
        const withoutFeature = await withContext({ subgroups: false, label: "segmented-reduce-twin" }, async (twin) => {
            expect(twin.caps.features.has("subgroups")).toBe(false);
            const result = await runSegmentedReduce(twin, s, "sum", WEIGHT);
            twin.release(s);
            return result;
        });
        expectBitwiseEqual(withFeature, withoutFeature, "twins");
        ctx.release(s);
    });

    it("the weighted f32 sum over random1k is the cross-adapter noise fixture: within the derived floor of every committed adapter, rows recorded", async (t) => {
        const ctx = await context(t);
        const s = weightedRandom(1000, 5000, 7);
        const result = await runSegmentedReduce(ctx, s, "sum", WEIGHT);
        const expected = segmentedReduceOracle(s, oracleValueOf(WEIGHT), "sum");
        const oracleErr = maxRelError(result, expected, SR_ABS_FLOOR);
        expect(oracleErr).toBeLessThanOrEqual(relTolerance(s, "sum"));
        const mine = adapterClass(ctx.caps);
        console.warn(`[segmented-reduce] ${mine}: oracle-f64 maxRelError ${oracleErr.toExponential(3)}`);
        writeNoiseFixture("segmented-reduce", "random1k", mine, result, "f32");
        recordNoiseRow({
            id: `segmented-reduce.sum.oracle-f64.${mine}`,
            kernel: "segmented-reduce",
            fixture: "random1k",
            comparison: "oracle-f64",
            a: mine,
            b: "oracle-f64",
            maxRelError: oracleErr,
            maxAbsError: maxAbsError(result, expected),
            samples: s.nodeCount,
        });
        const others = readNoiseFixtures("segmented-reduce", "random1k").filter((f) => f.adapterClass !== mine);
        if (others.length > 0) {
            const floor = noiseFloorFor("segmented-reduce.cross");
            for (const other of others) {
                expect(other.dtype).toBe("f32");
                expect(other.values.length).toBe(result.length);
                const err = maxRelError(result, other.values, SR_ABS_FLOOR);
                console.warn(
                    `[segmented-reduce] cross-adapter ${mine} vs ${other.adapterClass}: maxRelError ${err.toExponential(3)} (floor ${floor.value.toExponential(3)}, basis ${floor.basis})`,
                );
                recordNoiseRow({
                    id: `segmented-reduce.sum.cross.${mine}--${other.adapterClass}`,
                    kernel: "segmented-reduce",
                    fixture: "random1k",
                    comparison: "cross-adapter",
                    a: mine,
                    b: other.adapterClass,
                    maxRelError: err,
                    maxAbsError: maxAbsError(result, other.values),
                    samples: s.nodeCount,
                });
                expect(err).toBeLessThanOrEqual(floor.value);
            }
        }
        ctx.release(s);
    });
});
