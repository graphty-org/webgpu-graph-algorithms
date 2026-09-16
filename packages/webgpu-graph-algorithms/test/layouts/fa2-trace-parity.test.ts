/**
 * Trace parity (spec 11.4; contract 5.5 fa2-trace-parity.test.ts): 50 x step(1) on 10 .. 1,000-node graphs in both
 * compat modes, twice bitwise (spec 11.9 item 4), then two kinds of comparison with the f32 / f64 oracles:
 *
 * 1. The RE-SYNCHRONISED leg (P3-T5 PLAN DECISION 17; docs/decisions/G3.md finding G3-F3), in BOTH modes: before every
 *    iteration a fresh oracle is seeded with the GPU's iteration-start state (positions, controller, K1 statistics,
 *    its own previous force as F(t-1)) and stepped once, so one GPU iteration is compared with one oracle iteration.
 *    The record (K4's swing, traction, speed, speedEfficiency; K1's meanDisplacement, settledCount, centroid and
 *    rmsRadius of the same iteration, i.e. the fold of the GPU's previous integrate against the oracle's fold of ITS
 *    step from the same start) must agree within fa2-trace-parity.resync.f32 / .resync.f64 (cap 1e-4 each; floors
 *    measured by the random1k and karate resync members). This is the leg that carries the sabotage sensitivity of
 *    K3 / K4 / K5 over a real trajectory, in both modes.
 * 2. The FREE-RUNNING legs of spec 11.4. The first 10 records vs the f32 oracle within fa2-trace-parity.f32 are
 *    asserted in compat: "networkx" only. The 50-record leg vs the f64 oracle (fa2-trace-parity.f64, cap 5e-2) and the
 *    K1 fold fields over the same horizon are PRINTED in both modes, never asserted (G3-F3, the owner decision of
 *    section 10 and its item (2) as re-fixed by the review of 2026-09-16): in compat: "paper" the trajectory diverges
 *    exponentially (x1.1 .. x1.5 per iteration) from f32 rounding and the f64 oracle misses BOTH spec caps against
 *    ITSELF under a one-f32-ulp start perturbation (0.86 .. 1.71 relative through 50 on karate / grid10 / random1k,
 *    8e-4 within 10 on grid10); in compat: "networkx" the controller grows ~x1.25 per iteration and the 50-iteration
 *    cap holds at seed 7 with a 1% margin on karate but at NO other seed of 1 .. 8 (the f32 oracle vs the f64 oracle:
 *    karate 1.5e-1 .. 4.7e-1 at seeds 1-6 and 8, grid10 9.4e-2 / 6.3e-2 / 5.0e-2 at seeds 1 / 4 / 6), so an assertion
 *    of that leg would pin one seed, not bound the kernel. The seed sweep is measured by the last case of this file on
 *    every run (the GPU's own free-running legs at seeds 1 .. 8 printed beside the oracle pair) and its oracle-only
 *    rows are recorded in the noise-floor file under GRAPHTY_NOISE_FLOOR_WRITE=1 (`fa2-trace-parity.seeds.<graph>`
 *    through 50, `fa2-trace-parity.seeds10.<graph>` over the first 10; a: oracle-f32, b: oracle-f64 -- the basis of
 *    the decision, adapter-independent, never a tolerance's basis). The same sweep shows the asserted first-10 leg
 *    holding the SPEC cap 1e-4 at every seed measured (GPU vs f32 at most 3.9e-5, karate seed 2, NVIDIA) but not the
 *    DERIVED value fa2-trace-parity.f32 = 2.05e-5 (10x the random1k member's floor at seed 7): karate seed 2 reads
 *    1.88x and grid10 seed 8 1.06x of it on the GPU. It stays asserted as the spec's leg at the pinned seed; the
 *    owner item of G3-F3 (2) carries the re-derivation option (the seeds10 rows as the basis -> the cap).
 *
 * A recording run writes the unscaled random1k networkx traces (10 and 50 iterations) and the karate networkx trace
 * (50) with their f32 / f64 references, and the paper-mode re-synchronised traces of random1k and karate with their
 * per-adapter oracle references, as noise fixtures (PLAN DECISIONS 6, 7, 17).
 */

import type { GraphSnapshot } from "@graphty/graph-format";

import type { GpuContext } from "../../src/context.js";
import type { ForceAtlas2TraceRecord, GpuLayoutTuning } from "../../src/types/layout.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import {
    BASE_OPTIONS,
    maxAbsDiff,
    NETWORKX,
    NOISE_FIXTURES,
    noiseInputs,
    ORACLE_F32_CLASS,
    ORACLE_F64_CLASS,
    oracleOptionsFor,
    PAPER,
    type ParityGraph,
    paritySnapshot,
    resyncOracleClass,
    resyncReports,
    resyncTrace,
    resyncValues,
    startPositions,
    statsFoldError,
    toleranceOf,
    TRACE_ITERATIONS,
    TRACE_TIGHT,
    traceError,
    traceReports,
    traceValues,
    withSim,
} from "../helpers/fa2-parity.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { adapterClass, type NoiseRow, recordNoiseRow, writeNoiseFixture } from "../helpers/noise-floor.js";
import { assertCheckPasses, ratioOf } from "../helpers/sabotage.js";
import { forceAtlas2Oracle, type OracleTraceRecord } from "../oracle/forceatlas2.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

const ITERATIONS = TRACE_ITERATIONS;
/** 10 .. 1,000 nodes (spec 11.4): the path of 10, karate, the grid and random1k (scaled on software adapters). */
const GRAPHS: readonly ParityGraph[] = ["path10", "karate", "grid10", "random1k"];
const MODES = [PAPER, NETWORKX];
const CASE_TIMEOUT = 300_000;
/** The seeds of the free-running sweep (G3-F3 item 2): the pinned parity seed 7 and its seven neighbours. */
const SWEEP_SEEDS: readonly number[] = [1, 2, 3, 4, 5, 6, 7, 8];
/** The sweep's graphs: the three small fixtures of GRAPHS (random1k's oracle at 8 seeds would cost the lavapipe budget). */
const SWEEP_GRAPHS: readonly ParityGraph[] = ["path10", "karate", "grid10"];

/** The two free-running oracle trajectories of one start. */
function oracleTraces(
    s: GraphSnapshot,
    start: ArrayLike<number>,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
): { readonly f32: readonly OracleTraceRecord[]; readonly f64: readonly OracleTraceRecord[] } {
    const run = (precision: "f32" | "f64"): readonly OracleTraceRecord[] =>
        forceAtlas2Oracle(
            s,
            Float32Array.from(start),
            oracleOptionsFor(s, options, tuning, null, precision),
            ITERATIONS,
        ).trace;
    return { f32: run("f32"), f64: run("f64") };
}

/** The GPU's free-running trace of `iterations` x step(1) from `start` (no oracle, no re-synchronisation). */
async function freeRunningTrace(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: ArrayLike<number>,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    iterations: number,
): Promise<ForceAtlas2TraceRecord[]> {
    return await withSim(ctx, options, tuning, async (sim) => {
        sim.load(s, Float32Array.from(start));
        const trace: ForceAtlas2TraceRecord[] = [];
        for (let t = 0; t < iterations; t++) {
            await sim.step(1);
            trace.push(...sim.stats.trace);
        }
        return trace;
    });
}

describe("FA2 trace parity: 50 x step(1) vs the f32 / f64 oracles (spec 11.4)", () => {
    let ctx: GpuContext;

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-trace-parity" });
    });

    it("every tolerance is traced to the noise-floor file and sits under its spec cap", () => {
        expect(toleranceOf("fa2-trace-parity.f32")).toBeLessThanOrEqual(1e-4);
        expect(toleranceOf("fa2-trace-parity.f64")).toBeLessThanOrEqual(5e-2);
        expect(toleranceOf("fa2-trace-parity.resync.f32")).toBeLessThanOrEqual(1e-4);
        expect(toleranceOf("fa2-trace-parity.resync.f64")).toBeLessThanOrEqual(1e-4);
    });

    for (const graph of GRAPHS) {
        for (const tuning of MODES) {
            const paper = tuning.compat === "paper";
            const label = `${graph}/${tuning.compat ?? "paper"}`;
            it(
                `${label}: twice bitwise; re-synchronised within the traced tolerances${paper ? "; the free-running legs printed (chaotic, G3-F3)" : "; free-running first 10 vs f32 within the spec cap, 50 vs f64 printed (seed-bound, G3-F3 item 2)"}`,
                async (t) => {
                    requireGpu(t);
                    const s = paritySnapshot(graph, gpuScale(), false);
                    try {
                        const start = startPositions(s, BASE_OPTIONS, false);
                        const a = await resyncTrace(ctx, s, start, BASE_OPTIONS, tuning, ITERATIONS);
                        const b = await resyncTrace(ctx, s, start, BASE_OPTIONS, tuning, ITERATIONS);
                        expectBitwiseEqual(
                            traceValues(a.trace, 0, ITERATIONS),
                            traceValues(b.trace, 0, ITERATIONS),
                            `${label}: trace run 1 vs run 2`,
                        );
                        expectBitwiseEqual(a.positions, b.positions, `${label}: positions run 1 vs run 2`);
                        // the re-synchronised leg (both modes)
                        const resync = resyncReports(a, label);
                        console.warn(
                            `[fa2-trace-parity] ${label}: re-synchronised ${ITERATIONS} iterations: ratio vs f32 ${resync.f32.worst.toExponential(3)}, vs f64 ${resync.f64.worst.toExponential(3)}; per-node positions after one step differ by at most ${Math.max(...a.positionError.f32).toExponential(2)} (f32) / ${Math.max(...a.positionError.f64).toExponential(2)} (f64)`,
                        );
                        assertCheckPasses(resync.f32);
                        assertCheckPasses(resync.f64);
                        // the free-running legs of spec 11.4
                        const { f32, f64 } = oracleTraces(s, start, BASE_OPTIONS, tuning);
                        expect(f32).toHaveLength(ITERATIONS);
                        expect(f64).toHaveLength(ITERATIONS);
                        const reports = traceReports(a.trace, f32, f64, label);
                        const fold = {
                            worst: ratioOf(
                                statsFoldError(a.stats, f64, 0, ITERATIONS),
                                toleranceOf("fa2-trace-parity.f64"),
                            ),
                            worstLabel: `${label}: K1 fold fields`,
                            samples: 5 * ITERATIONS,
                        };
                        console.warn(
                            `[fa2-trace-parity] ${label}: free-running first ${TRACE_TIGHT} vs f32 ratio ${reports.first10.worst.toExponential(3)}${paper ? " (paper mode: informational, G3-F3)" : ""}; ${ITERATIONS} vs f64 ratio ${reports.through50.worst.toExponential(3)}; K1 fold ratio ${fold.worst.toExponential(3)} (the 50-iteration legs: informational in both modes, G3-F3 item 2)`,
                        );
                        if (!paper) {
                            assertCheckPasses(reports.first10);
                        }
                        // settleThreshold 0: K1 increments settledCount only when meanDisplacement is exactly 0 (every node fixed)
                        expect(
                            a.trace.every((r) => r.settledCount === 0),
                            `${label}: never settles`,
                        ).toBe(true);
                        // iteration 1 carries FA2_FLAG_FIRST: K1 folds nothing and reports the host-written meanDisplacement 0
                        expect(a.trace[0].meanDisplacement).toBe(0);
                        // every later record reports a positive mean displacement (the layout moves)
                        expect(
                            a.trace.slice(1).every((r) => r.meanDisplacement > 0),
                            `${label}: moving`,
                        ).toBe(true);
                        // every value of every record finite
                        expect(
                            traceValues(a.trace, 0, ITERATIONS).every((v) => Number.isFinite(v)),
                            `${label}: finite trace`,
                        ).toBe(true);
                    } finally {
                        ctx.release(s);
                    }
                },
                CASE_TIMEOUT,
            );
        }
    }

    it(
        "writes the karate NETWORKX trace (50 iterations) and its f64 reference as noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const { s, start, options, tuning } = noiseInputs(NETWORKX, "karate");
            try {
                const run = await resyncTrace(ctx, s, start, options, tuning, ITERATIONS);
                const { f64 } = oracleTraces(s, start, options, tuning);
                const through50 = {
                    worst: ratioOf(traceError(run.trace, f64, 0, ITERATIONS), toleranceOf("fa2-trace-parity.f64")),
                    worstLabel: `noise/karate/networkx: iterations 1-${ITERATIONS} vs f64`,
                    samples: 6 * ITERATIONS,
                };
                const cls = adapterClass(ctx.caps);
                writeNoiseFixture(
                    NOISE_FIXTURES.trace50Karate.kernel,
                    NOISE_FIXTURES.trace50Karate.fixture,
                    cls,
                    traceValues(run.trace, 0, ITERATIONS),
                    "f32",
                );
                writeNoiseFixture(
                    NOISE_FIXTURES.trace50Karate.kernel,
                    NOISE_FIXTURES.trace50Karate.fixture,
                    ORACLE_F64_CLASS,
                    traceValues(f64, 0, ITERATIONS),
                    "f32",
                );
                // the 50-iteration leg is recorded (the floor of fa2-trace-parity.f64), never asserted: G3-F3 item 2
                console.warn(
                    `[fa2-trace-parity] noise/karate/networkx: ${ITERATIONS} vs f64 ratio ${through50.worst.toExponential(3)} (informational, G3-F3 item 2)`,
                );
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );

    it(
        "writes the unscaled random1k NETWORKX traces (10 and 50 iterations) and the f32 / f64 references as noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const { s, start, options, tuning } = noiseInputs(NETWORKX);
            try {
                const run = await resyncTrace(ctx, s, start, options, tuning, ITERATIONS);
                const { f32, f64 } = oracleTraces(s, start, options, tuning);
                const reports = traceReports(run.trace, f32, f64, "noise/random1k/networkx");
                // the fixtures are the RAW outputs (never hand-written, spec 11.9 item 3) and are written BEFORE the
                // check so that test/noise-floor.test.ts can measure the basis rows even when the check fails: a
                // floor above the spec cap then surfaces through the validation of the noise-floor file (the finding
                // path of the plan's Step 17d item 1) instead of through missing fixtures
                const cls = adapterClass(ctx.caps);
                writeNoiseFixture(
                    NOISE_FIXTURES.trace10.kernel,
                    NOISE_FIXTURES.trace10.fixture,
                    cls,
                    traceValues(run.trace, 0, TRACE_TIGHT),
                    "f32",
                );
                writeNoiseFixture(
                    NOISE_FIXTURES.trace10.kernel,
                    NOISE_FIXTURES.trace10.fixture,
                    ORACLE_F32_CLASS,
                    traceValues(f32, 0, TRACE_TIGHT),
                    "f32",
                );
                writeNoiseFixture(
                    NOISE_FIXTURES.trace50.kernel,
                    NOISE_FIXTURES.trace50.fixture,
                    cls,
                    traceValues(run.trace, 0, ITERATIONS),
                    "f32",
                );
                writeNoiseFixture(
                    NOISE_FIXTURES.trace50.kernel,
                    NOISE_FIXTURES.trace50.fixture,
                    ORACLE_F64_CLASS,
                    traceValues(f64, 0, ITERATIONS),
                    "f32",
                );
                console.warn(
                    `[fa2-trace-parity] noise/random1k/networkx: first ${TRACE_TIGHT} vs f32 ratio ${reports.first10.worst.toExponential(3)}; ${ITERATIONS} vs f64 ratio ${reports.through50.worst.toExponential(3)} (informational, G3-F3 item 2)`,
                );
                assertCheckPasses(reports.first10);
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );

    for (const member of ["resync50", "resyncKarate50"] as const) {
        const graph: ParityGraph = member === "resync50" ? "random1k" : "karate";
        it(
            `writes the unscaled ${graph} PAPER re-synchronised trace and its per-adapter f32 / f64 references as noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)`,
            async (t) => {
                requireGpu(t);
                const { s, start, options, tuning } = noiseInputs(PAPER, graph);
                try {
                    const run = await resyncTrace(ctx, s, start, options, tuning, ITERATIONS);
                    const reports = resyncReports(run, `noise/${graph}/paper`);
                    const cls = adapterClass(ctx.caps);
                    const { kernel, fixture } = NOISE_FIXTURES[member];
                    writeNoiseFixture(kernel, fixture, cls, resyncValues(run.gpu), "f32");
                    writeNoiseFixture(
                        kernel,
                        fixture,
                        resyncOracleClass(ORACLE_F32_CLASS, cls),
                        resyncValues(run.f32),
                        "f32",
                    );
                    writeNoiseFixture(
                        kernel,
                        fixture,
                        resyncOracleClass(ORACLE_F64_CLASS, cls),
                        resyncValues(run.f64),
                        "f32",
                    );
                    console.warn(
                        `[fa2-trace-parity] noise/${graph}/paper: re-synchronised ratio vs f32 ${reports.f32.worst.toExponential(3)}, vs f64 ${reports.f64.worst.toExponential(3)}`,
                    );
                    assertCheckPasses(reports.f32);
                    assertCheckPasses(reports.f64);
                } finally {
                    ctx.release(s);
                }
            },
            CASE_TIMEOUT,
        );
    }
    it(
        "the networkx free-running 50-vs-f64 leg across seeds 1 .. 8 is seed-bound, not kernel-bound (G3-F3 item 2): printed on every run, the oracle pair recorded (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const f32Tol = toleranceOf("fa2-trace-parity.f32");
            const f64Tol = toleranceOf("fa2-trace-parity.f64");
            const capRatio = 5e-2 / f64Tol;
            for (const graph of SWEEP_GRAPHS) {
                const s = paritySnapshot(graph, 1, false);
                try {
                    const lines: string[] = [];
                    let oracleWorst10 = 0;
                    let oracleWorst50 = 0;
                    let gpuWorst10 = 0;
                    let gpuWorst50 = 0;
                    let oracleAbs10 = 0;
                    let oracleAbs50 = 0;
                    let seedsOverCap = 0;
                    for (const seed of SWEEP_SEEDS) {
                        const options: ForceAtlas2Options = { ...BASE_OPTIONS, seed };
                        const start = startPositions(s, options, false);
                        const { f32, f64 } = oracleTraces(s, start, options, NETWORKX);
                        const gpu = await freeRunningTrace(ctx, s, start, options, NETWORKX, ITERATIONS);
                        expect(gpu, `${graph} seed ${seed}: one record per step`).toHaveLength(ITERATIONS);
                        // the oracle pair: the f32 oracle's own free-running divergence from the f64 oracle
                        const o10 = traceError(f32, f64, 0, TRACE_TIGHT) / f32Tol;
                        const o50 = traceError(f32, f64, 0, ITERATIONS) / f64Tol;
                        // the GPU's legs of this file, at this seed
                        const g10 = traceError(gpu, f32, 0, TRACE_TIGHT) / f32Tol;
                        const g50 = traceError(gpu, f64, 0, ITERATIONS) / f64Tol;
                        oracleWorst10 = Math.max(oracleWorst10, o10);
                        oracleWorst50 = Math.max(oracleWorst50, o50);
                        oracleAbs10 = Math.max(
                            oracleAbs10,
                            maxAbsDiff(traceValues(f32, 0, TRACE_TIGHT), traceValues(f64, 0, TRACE_TIGHT)),
                        );
                        oracleAbs50 = Math.max(
                            oracleAbs50,
                            maxAbsDiff(traceValues(f32, 0, ITERATIONS), traceValues(f64, 0, ITERATIONS)),
                        );
                        gpuWorst10 = Math.max(gpuWorst10, g10);
                        gpuWorst50 = Math.max(gpuWorst50, g50);
                        if (g50 > capRatio) {
                            seedsOverCap += 1;
                        }
                        lines.push(
                            `s${seed}: oracle f32-vs-f64 ${o10.toExponential(2)} / ${o50.toExponential(2)}, gpu ${g10.toExponential(2)} / ${g50.toExponential(2)}`,
                        );
                    }
                    console.warn(
                        `[fa2-trace-parity] seeds/${graph}/networkx (ratios first ${TRACE_TIGHT} vs f32 / ${ITERATIONS} vs f64, tolerances ${f32Tol.toExponential(3)} / ${f64Tol.toExponential(3)}): ${lines.join("; ")}; worst oracle ${oracleWorst10.toExponential(3)} / ${oracleWorst50.toExponential(3)}, worst gpu ${gpuWorst10.toExponential(3)} / ${gpuWorst50.toExponential(3)}; ${seedsOverCap} of ${SWEEP_SEEDS.length} seeds over the 5e-2 cap on the GPU`,
                    );
                    // every value finite at every seed (a NaN trajectory would be a kernel defect, whatever the seed)
                    expect(Number.isFinite(gpuWorst10) && Number.isFinite(gpuWorst50), `${graph}: finite`).toBe(true);
                    // the record of the decision: the oracle pair at both horizons, adapter-independent, never a
                    // tolerance's basis (the first-10 row is the basis the owner would re-derive fa2-trace-parity.f32
                    // from, G3-F3 item 2)
                    const rowOf = (
                        suffix: string,
                        horizon: number,
                        relRatio: number,
                        tol: number,
                        abs: number,
                    ): NoiseRow => ({
                        id: `fa2-trace-parity.${suffix}.${graph}`,
                        kernel: NOISE_FIXTURES.trace50Karate.kernel,
                        fixture: `${graph}-trace${horizon}-seeds1-8`,
                        comparison: "oracle-f64",
                        a: ORACLE_F32_CLASS,
                        b: ORACLE_F64_CLASS,
                        maxRelError: relRatio * tol,
                        maxAbsError: abs,
                        samples: 6 * horizon * SWEEP_SEEDS.length,
                    });
                    recordNoiseRow(rowOf("seeds10", TRACE_TIGHT, oracleWorst10, f32Tol, oracleAbs10));
                    recordNoiseRow(rowOf("seeds", ITERATIONS, oracleWorst50, f64Tol, oracleAbs50));
                } finally {
                    ctx.release(s);
                }
            }
        },
        CASE_TIMEOUT,
    );
});
