/**
 * Distributional parity (spec 11.4; contract 5.5 fa2-distributional.test.ts): same seed, 100 iterations, the
 * layout-quality metrics of the GPU layout and of the f64 oracle's layout agree within 10%; coordinates are never
 * compared (spec 7.16: chaotic divergence beyond the trace horizon is expected). The first block pins the metrics
 * themselves (test/helpers/metrics.ts) on hand-computed values.
 *
 * The cases (P3-T5 PLAN DECISION 18; docs/decisions/G3.md finding G3-F4): after 100 iterations any two runs from
 * starts one f32 ulp apart have decorrelated coordinates, so the metrics of two such runs differ by the run-to-run
 * spread of the layout's OWN basin structure, whatever the implementation. A case is admitted here only where the
 * f64 oracle reproduces its own metrics within a third of the 10% cap under eight one-ulp start perturbations
 * (measured on the dev box, section 10 of G3.md: karate/paper/2d 1.7e-2, star200/paper/2d 1.4e-5, random1k/paper/2d
 * 7.4e-2 over 32 perturbations -- the flagship shape and the noise member, kept with its measured 1.8x margin --,
 * karate/paper/3d 1.3e-4, random1k/paper/3d 2.5e-2, star200/networkx/2d 5.9e-6, karate/networkx/3d 4.1e-6,
 * isolated/networkx/3d 3.1e-5). The 10 x 10 grid in 2D (2.0e-1: the grid folds into different basins), karate in
 * networkx 2D (1.77e-1: the nearest-neighbour histogram of 34 nodes moves by whole nodes) and the isolated fixture in
 * paper mode (2.2e-1 in 2D, 1.56e-1 in 3D: the `separation` extreme value and the strays' nearest neighbours) are
 * unmeetable by ANY implementation at the 10% cap and are not cases; their force, trace and twin parity is covered
 * by the other P3 tests.
 */

import type { F32, GraphSnapshot } from "@graphty/graph-format";

import type { GpuContext } from "../../src/context.js";
import type { GpuLayoutTuning } from "../../src/types/layout.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import {
    BASE_OPTIONS,
    distributionalError,
    metricsValues,
    NETWORKX,
    NOISE_FIXTURES,
    noiseInputs,
    ORACLE_F64_CLASS,
    oracleOptionsFor,
    PAPER,
    type ParityGraph,
    paritySnapshot,
    startPositions,
    toleranceOf,
    withSim,
} from "../helpers/fa2-parity.js";
import { fixture, snapshotOf } from "../helpers/graphs.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import {
    componentSeparation,
    edgeLengthQuantiles,
    layoutMetrics,
    nearestNeighbourHistogram,
    spread,
    stress,
} from "../helpers/metrics.js";
import { adapterClass, writeNoiseFixture } from "../helpers/noise-floor.js";
import { assertCheckPasses, ratioOf } from "../helpers/sabotage.js";
import { forceAtlas2Oracle } from "../oracle/forceatlas2.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

// ---------------------------------------------------------------- the metrics on a hand-computed layout

/** Path 0 - 1 - 2 laid out on the x axis at 0, 1, 3 (stride 3). */
const LINE = Float32Array.from([0, 0, 0, 1, 0, 0, 3, 0, 0]);
const PATH3 = snapshotOf(
    [
        [0, 1],
        [1, 2],
    ],
    { nodeCount: 3, label: "path3" },
);
/** Nodes 0 - 1 joined, node 2 alone. */
const SPLIT3 = snapshotOf([[0, 1]], { nodeCount: 3, label: "split3" });

describe("metrics (test/helpers/metrics.ts) on a hand-computed layout", () => {
    it("stress: mean over connected pairs of ((euclid - hops) / hops)^2 = (0 + 0.25 + 1) / 3", () => {
        // pairs: (0,1) euclid 1 hops 1 -> 0; (0,2) euclid 3 hops 2 -> 0.25; (1,2) euclid 2 hops 1 -> 1
        expect(stress(PATH3, LINE, 2)).toBeCloseTo(1.25 / 3, 12);
        // the same numbers in 3D (z is 0 everywhere)
        expect(stress(PATH3, LINE, 3)).toBeCloseTo(1.25 / 3, 12);
        // one pair only in the split graph: (0,1) -> 0
        expect(stress(SPLIT3, LINE, 2)).toBe(0);
    });

    it("edgeLengthQuantiles: lengths [1, 2] -> q0 1, q0.5 1.5, q1 2 (linear interpolation), each undirected edge once", () => {
        expect(edgeLengthQuantiles(PATH3, LINE, 2, [0, 0.5, 1])).toEqual([1, 1.5, 2]);
        expect(edgeLengthQuantiles(PATH3, LINE, 2, [0.1, 0.9])).toEqual([1.1, 1.9]);
        expect(edgeLengthQuantiles(SPLIT3, LINE, 2, [0.5])).toEqual([1]);
        expect(edgeLengthQuantiles(snapshotOf([], { nodeCount: 2 }), LINE, 2, [0.5])).toEqual([0]);
    });

    it("nearestNeighbourHistogram: nn = [1, 1, 2], mean 4/3, ratios [0.75, 0.75, 1.5] over [0, 2) in 4 bins -> [0, 2/3, 0, 1/3]", () => {
        const h = nearestNeighbourHistogram(LINE, 3, 2, 4);
        expect(h).toHaveLength(4);
        expect(h[0]).toBe(0);
        expect(h[1]).toBeCloseTo(2 / 3, 12);
        expect(h[2]).toBe(0);
        expect(h[3]).toBeCloseTo(1 / 3, 12);
        // fractions sum to 1
        expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
        // a single node has no neighbour: all zeros
        expect(nearestNeighbourHistogram(LINE, 1, 2, 4)).toEqual([0, 0, 0, 0]);
        // every node coincident (dim 2 of a z-only spread): mean 0 -> everything in bin 0
        const zOnly = Float32Array.from([0, 0, 0, 0, 0, 1, 0, 0, 3]);
        expect(nearestNeighbourHistogram(zOnly, 3, 2, 4)).toEqual([1, 0, 0, 0]);
    });

    it("componentSeparation: Infinity when connected; the closest cross-component pair otherwise", () => {
        expect(componentSeparation(PATH3, LINE, 2)).toBe(Number.POSITIVE_INFINITY);
        // split: node 2 at x = 3 against nodes 0 (x = 0) and 1 (x = 1) -> 2
        expect(componentSeparation(SPLIT3, LINE, 2)).toBe(2);
    });

    it("spread: the largest axis extent (max - min) over the dim axes", () => {
        // x extent 3, y extent 0
        expect(spread(LINE, 3, 2)).toBe(3);
        expect(spread(LINE, 0, 2)).toBe(0);
        // z spread invisible in 2D, visible in 3D
        const zOnly = Float32Array.from([0, 0, 0, 0, 0, 1, 0, 0, 3]);
        expect(spread(zOnly, 3, 2)).toBe(0);
        expect(spread(zOnly, 3, 3)).toBe(3);
    });

    it("layoutMetrics: the record of every metric (separation only when finite; 8 self-normalised nn bins)", () => {
        const m = layoutMetrics(PATH3, LINE, 2);
        expect(m.stress).toBeCloseTo(1.25 / 3, 12);
        expect(m.edgeQ10).toBeCloseTo(1.1, 12);
        expect(m.edgeQ50).toBeCloseTo(1.5, 12);
        expect(m.edgeQ90).toBeCloseTo(1.9, 12);
        // nn sorted [1, 1, 2]: q0.1 at position 0.2 -> 1; q0.5 -> 1; q0.9 at position 1.8 -> 1.8
        expect(m.nnQ10).toBeCloseTo(1, 12);
        expect(m.nnQ50).toBeCloseTo(1, 12);
        expect(m.nnQ90).toBeCloseTo(1.8, 12);
        expect(m.spread).toBe(3);
        expect(m.separation).toBeUndefined();
        // ratios 0.75, 0.75, 1.5 over 8 bins of width 0.25: bin 3 twice, bin 6 once
        expect(m.nnBin3).toBeCloseTo(2 / 3, 12);
        expect(m.nnBin6).toBeCloseTo(1 / 3, 12);
        expect(Object.keys(m).filter((k) => k.startsWith("nnBin"))).toHaveLength(8);
        const split = layoutMetrics(SPLIT3, LINE, 2);
        expect(split.separation).toBe(2);
    });
});

// ---------------------------------------------------------------- 100 iterations on the device vs the f64 oracle

const ITERATIONS = 100;
const CASE_TIMEOUT = 300_000;

interface DistributionalCase {
    /** A parity graph, or the contract 5.2 named fixture "isolated" (the only multi-component case: it carries `separation`). */
    readonly graph: ParityGraph | "isolated";
    readonly dim: 2 | 3;
    readonly tuning: GpuLayoutTuning;
}

const CASES: readonly DistributionalCase[] = [
    { graph: "karate", dim: 2, tuning: PAPER },
    { graph: "star200", dim: 2, tuning: PAPER },
    { graph: "random1k", dim: 2, tuning: PAPER },
    { graph: "karate", dim: 3, tuning: PAPER },
    { graph: "random1k", dim: 3, tuning: PAPER },
    { graph: "star200", dim: 2, tuning: NETWORKX },
    { graph: "karate", dim: 3, tuning: NETWORKX },
    // the multi-component case of spec 11.4 ("inter-component separation within 10%"): every parity graph is
    // connected, so without it `separation` is never in the compared record; networkx 3D is the stable instance
    // (file header)
    { graph: "isolated", dim: 3, tuning: NETWORKX },
];

/**
 * The snapshot of a distributional case at gpuScale(): the named fixture "isolated" (giant component + 1% isolated
 * nodes + 100 triangles, P1-T2's fixture()) or a parity graph.
 */
function distributionalSnapshot(graph: ParityGraph | "isolated"): GraphSnapshot {
    return graph === "isolated" ? fixture("isolated", gpuScale()).snapshot : paritySnapshot(graph, gpuScale(), false);
}

/**
 * run() to exactly ITERATIONS iterations (maxIter = ITERATIONS, settleThreshold 0) on a fresh simulation; returns the
 * owner's array (scene units, scale 1) after the last batch.
 */
async function runLayout(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
): Promise<F32> {
    return await withSim(ctx, options, tuning, async (sim) => {
        const positions = Float32Array.from(start);
        sim.load(s, positions);
        const stats = await sim.run({ batch: 10 });
        expect(sim.iterationsDone, "iterationsDone").toBe(ITERATIONS);
        expect(stats.iteration, "stats.iteration").toBe(ITERATIONS);
        expect(sim.settled, "settled at maxIter").toBe(true);
        expect(sim.inFlight, "nothing in flight after run()").toBe(0);
        return positions;
    });
}

describe("FA2 distributional parity: 100 iterations, metrics within the traced 10% (spec 11.4)", () => {
    let ctx: GpuContext;

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-distributional" });
    });

    for (const c of CASES) {
        const label = `${c.graph}/${c.tuning.compat ?? "paper"}/${c.dim}d`;
        it(
            `${label}: layoutMetrics of the GPU layout vs the f64 oracle's, coordinates never compared, twice bitwise`,
            async (t) => {
                requireGpu(t);
                const s = distributionalSnapshot(c.graph);
                try {
                    const options: ForceAtlas2Options = { ...BASE_OPTIONS, dim: c.dim, maxIter: ITERATIONS };
                    const start = startPositions(s, options, false);
                    const a = await runLayout(ctx, s, start, options, c.tuning);
                    const b = await runLayout(ctx, s, start, options, c.tuning);
                    expectBitwiseEqual(a, b, `${label}: run 1 vs run 2`);
                    const reference = forceAtlas2Oracle(
                        s,
                        Float32Array.from(start),
                        oracleOptionsFor(s, options, c.tuning, null, "f64"),
                        ITERATIONS,
                    ).positions;
                    const gpuMetrics = layoutMetrics(s, a, c.dim);
                    const oracleMetrics = layoutMetrics(s, reference, c.dim);
                    expect(Object.keys(gpuMetrics).sort()).toEqual(Object.keys(oracleMetrics).sort());
                    // `separation` is present iff the graph has two or more components (a property of the snapshot, not of
                    // the positions): the "isolated" fixture by construction, and random1k (seed 1234) through the few nodes
                    // its G(n, 3n) draw left without an edge; the connected graphs must not carry it
                    expect("separation" in gpuMetrics, `${label}: separation present`).toBe(
                        Number.isFinite(componentSeparation(s, start, c.dim)),
                    );
                    const err = distributionalError(gpuMetrics, oracleMetrics);
                    console.warn(
                        `[fa2-distributional] ${label}: worst metric difference ${err.toExponential(3)} (spread gpu ${gpuMetrics.spread.toFixed(4)} oracle ${oracleMetrics.spread.toFixed(4)})`,
                    );
                    assertCheckPasses({
                        worst: ratioOf(err, toleranceOf("fa2-distributional")),
                        worstLabel: label,
                        samples: Object.keys(oracleMetrics).length,
                    });
                    // the layout expanded from the unit square: FA2 has no unit-ball normalisation (spec 7.18)
                    expect(gpuMetrics.spread).toBeGreaterThan(2);
                } finally {
                    ctx.release(s);
                }
            },
            CASE_TIMEOUT,
        );
    }

    it(
        "writes the UNSCALED random1k metrics after 100 iterations and the f64 reference's as noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const { s, start, tuning } = noiseInputs();
            try {
                const options: ForceAtlas2Options = { ...BASE_OPTIONS, maxIter: ITERATIONS };
                const gpu = metricsValues(layoutMetrics(s, await runLayout(ctx, s, start, options, tuning), 2));
                const reference = forceAtlas2Oracle(
                    s,
                    Float32Array.from(start),
                    oracleOptionsFor(s, options, tuning, null, "f64"),
                    ITERATIONS,
                ).positions;
                const oracle = metricsValues(layoutMetrics(s, reference, 2));
                expect(gpu.keys).toEqual(oracle.keys);
                writeNoiseFixture(
                    NOISE_FIXTURES.metrics100.kernel,
                    NOISE_FIXTURES.metrics100.fixture,
                    adapterClass(ctx.caps),
                    gpu.values,
                    "f32",
                );
                writeNoiseFixture(
                    NOISE_FIXTURES.metrics100.kernel,
                    NOISE_FIXTURES.metrics100.fixture,
                    ORACLE_F64_CLASS,
                    oracle.values,
                    "f32",
                );
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );
});
