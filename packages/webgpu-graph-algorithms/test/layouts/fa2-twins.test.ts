/**
 * The subgroup twins (spec 11.3 "Subgroup variants" row; contract 5.5 fa2-twins.test.ts): K1 / K3 / K4 / K5 with and
 * without subgroups IN-PROCESS -- the feature context and a second context from acquire({ subgroups: false }) -- must
 * agree (the subgroup tree and the workgroup tree sum in different orders; K2's attraction, upstream of every
 * reduction, must agree bitwise) on:
 *
 * 1. every stage's output of one iteration from the seeded start, within the traced fa2-twins.force / fa2-twins.trace
 *    / fa2-twins.positions tolerances (the stage twin);
 * 2. one iteration from EACH of ten iteration-start states of the f64 oracle's free-running trajectory (positions
 *    after t = 0 .. 9 iterations, oracleStates(), loaded into both twins: the RE-SYNCHRONISED per-iteration twin,
 *    P3-T5 PLAN DECISION 17), the trace record within fa2-twins.trace and the positions within fa2-twins.positions
 *    -- ten different geometries (the layout expands from the unit square to a radius of ~1e2) through the same
 *    reductions; the states are the oracle's, not a twin's, so the states10 noise member that measures this floor
 *    holds the same inputs on every adapter;
 * 3. the workgroup twin's 50-iteration trajectory against the f32 / f64 oracles re-synchronised to ITS state before
 *    every iteration (resyncTrace), within fa2-trace-parity.resync.f32 / .f64 -- the full-state per-iteration
 *    correctness of the twin that the feature context's run of fa2-trace-parity.test.ts establishes for the other;
 * 4. the free-running 10-iteration trace, asserted in compat: "networkx" within fa2-twins.trace (the trace10 twin
 *    member measures its floor) and printed in compat: "paper", where the reduction-order difference is amplified
 *    chaotically (1.57e-4 after 10 iterations on random1k, 0.9 after 50; docs/decisions/G3.md G3-F3) and no
 *    tolerance derived from one fixture's floor bounds another's.
 *
 * A recording run writes the `<class>-no-subgroups` fixtures of the P3 noise members that carry a twin row, the
 * networkx trace10 twin, the states10 member (both twins and the f64 reference, ten states of each mode), and the
 * paper-mode re-synchronised members of the twin with its per-adapter oracle references.
 */

import type { F32, GraphSnapshot } from "@graphty/graph-format";

import type { GpuContext } from "../../src/context.js";
import type { GpuLayoutTuning } from "../../src/types/layout.js";
import {
    BASE_OPTIONS,
    captureAllStages,
    NETWORKX,
    NOISE_FIXTURES,
    noiseInputs,
    ORACLE_F32_CLASS,
    ORACLE_F64_CLASS,
    oracleRecordFrom,
    oracleStates,
    PAPER,
    type ParityGraph,
    paritySnapshot,
    resyncOracleClass,
    resyncReports,
    resyncTrace,
    resyncValues,
    STAGE_KEYS,
    stageError,
    type StageKey,
    startPositions,
    toleranceOf,
    TRACE_ITERATIONS,
    traceValues,
    TWIN_SUFFIX,
    withSim,
} from "../helpers/fa2-parity.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { adapterClass, writeNoiseFixture } from "../helpers/noise-floor.js";
import { assertCheckPasses, type CheckReport, mergeReports, ratioOf } from "../helpers/sabotage.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

const CASE_TIMEOUT = 300_000;
/** The free-running twin trace and the number of re-synchronised states. */
const TRACE_TEN = 10;
const GRAPHS: readonly ParityGraph[] = ["karate", "random1k"];
/** Spec 12.2's second pass: acquire() then defaults to subgroups: false, so both contexts are the workgroup twin and the twin axis is vacuous here (a policy skip, not a wrong-result skip). */
const NO_SUBGROUPS_PASS = process.env.GRAPHTY_GPU_NO_SUBGROUPS === "1";
/**
 * Stages that see a reduction before they are read (K3's epilogue, K4, K5's partials, K1; the positions and the scene
 * carry K4's speed, itself a reduction result); K2's attraction is the one stage with no reduction anywhere upstream
 * and must agree bitwise. K3's force is computed before the epilogue and is expected bitwise too, but its kernel has a
 * twin, so it is only held to fa2-twins.force.
 */
const TWINNED: readonly StageKey[] = ["force", "epilogue", "state", "positions", "partials", "scene", "k1"];
const TWIN_TOLERANCE: Readonly<Record<StageKey, string>> = {
    attraction: "fa2-twins.force",
    force: "fa2-twins.force",
    epilogue: "fa2-twins.trace",
    state: "fa2-twins.trace",
    positions: "fa2-twins.positions",
    partials: "fa2-twins.trace",
    scene: "fa2-twins.positions",
    k1: "fa2-twins.trace",
};
/**
 * The noise members that carry a twin row (contract 5.6; test/noise-floor.test.ts): the scalar members feed
 * fa2-twins.trace.twin, the force member fa2-twins.force.twin, the two per-node vector members (positions, scene)
 * fa2-twins.positions.twin -- every tolerance above has a measured twin floor of the SAME quantity and metric.
 */
const TWIN_MEMBERS: readonly StageKey[] = ["force", "epilogue", "state", "positions", "partials", "scene", "k1"];

/**
 * One iteration from a given layout state on a fresh simulation (load() then step(1)): the trace record's six
 * values and the positions after (scene = layout units here).
 * @param ctx - the context
 * @param graph - the snapshot
 * @param state - the iteration-start positions (copied)
 * @param tuning - the GPU tuning
 * @returns the record values and the positions after the step
 */
async function oneIterationFrom(
    ctx: GpuContext,
    graph: GraphSnapshot,
    state: F32,
    tuning: GpuLayoutTuning,
): Promise<{ readonly record: Float64Array; readonly positions: F32 }> {
    return await withSim(ctx, BASE_OPTIONS, tuning, async (sim) => {
        const positions = Float32Array.from(state);
        sim.load(graph, positions);
        await sim.step(1);
        return { record: traceValues(sim.stats.trace, 0, 1), positions };
    });
}

describe("FA2 subgroup twins in-process (spec 11.3)", () => {
    let withSubgroups: GpuContext;
    let withoutSubgroups: GpuContext;

    beforeAll(async () => {
        withSubgroups = await acquire({ label: "fa2-twins/subgroups" });
        withoutSubgroups = await acquire({ label: "fa2-twins/no-subgroups", subgroups: false });
    });

    it("the two contexts are the two twins: the first exposes subgroups, the second does not", () => {
        expect(
            withSubgroups.caps.features.has("subgroups"),
            "the feature context has subgroups (lavapipe 8, NVIDIA 32)",
        ).toBe(!NO_SUBGROUPS_PASS);
        expect(withoutSubgroups.caps.features.has("subgroups")).toBe(false);
        expect(adapterClass(withSubgroups.caps)).toBe(adapterClass(withoutSubgroups.caps));
    });

    for (const graph of GRAPHS) {
        for (const tuning of [PAPER, NETWORKX]) {
            const paper = tuning.compat === "paper";
            const label = `${graph}/${tuning.compat ?? "paper"}`;
            it(
                `${label}: every stage agrees between the twins within the traced tolerances; K2's attraction bitwise`,
                async (t) => {
                    requireGpu(t);
                    if (NO_SUBGROUPS_PASS) {
                        t.skip("GRAPHTY_GPU_NO_SUBGROUPS=1: both contexts are the workgroup twin");
                        return;
                    }
                    const s = paritySnapshot(graph, gpuScale(), false);
                    try {
                        const start = startPositions(s, BASE_OPTIONS, false);
                        const a = await captureAllStages(withSubgroups, s, start, BASE_OPTIONS, tuning, null);
                        const b = await captureAllStages(withoutSubgroups, s, start, BASE_OPTIONS, tuning, null);
                        for (const key of STAGE_KEYS) {
                            const err = stageError(a[key].vector, a[key].values, b[key].values);
                            const report: CheckReport = {
                                worst: ratioOf(err.rel, toleranceOf(TWIN_TOLERANCE[key])),
                                worstLabel: `${label}/${key}`,
                                samples: a[key].values.length,
                            };
                            console.warn(
                                `[fa2-twins] ${label}/${key}: rel ${err.rel.toExponential(3)} abs ${err.abs.toExponential(3)} ratio ${report.worst.toExponential(3)}`,
                            );
                            assertCheckPasses(report);
                            if (!TWINNED.includes(key)) {
                                expectBitwiseEqual(a[key].values, b[key].values, `${label}/${key}: no twin, bitwise`);
                            }
                        }
                    } finally {
                        withSubgroups.release(s);
                        withoutSubgroups.release(s);
                    }
                },
                CASE_TIMEOUT,
            );

            it(
                `${label}: one iteration from each of ${TRACE_TEN} trajectory states agrees between the twins (re-synchronised per iteration)`,
                async (t) => {
                    requireGpu(t);
                    if (NO_SUBGROUPS_PASS) {
                        t.skip("GRAPHTY_GPU_NO_SUBGROUPS=1: both contexts are the workgroup twin");
                        return;
                    }
                    const s = paritySnapshot(graph, gpuScale(), false);
                    try {
                        const start = startPositions(s, BASE_OPTIONS, false);
                        // the states: the f64 oracle's positions after t = 0 .. TRACE_TEN - 1 iterations
                        const states = oracleStates(s, start, BASE_OPTIONS, tuning, TRACE_TEN);
                        const records: CheckReport[] = [];
                        const positions: CheckReport[] = [];
                        for (let k = 0; k < states.length; k++) {
                            const a = await oneIterationFrom(withSubgroups, s, states[k], tuning);
                            const b = await oneIterationFrom(withoutSubgroups, s, states[k], tuning);
                            records.push({
                                worst: ratioOf(
                                    stageError(false, a.record, b.record).rel,
                                    toleranceOf("fa2-twins.trace"),
                                ),
                                worstLabel: `${label}: record from state ${k}`,
                                samples: a.record.length,
                            });
                            positions.push({
                                worst: ratioOf(
                                    stageError(true, a.positions, b.positions).rel,
                                    toleranceOf("fa2-twins.positions"),
                                ),
                                worstLabel: `${label}: positions from state ${k}`,
                                samples: s.nodeCount,
                            });
                        }
                        const record = mergeReports(records);
                        const position = mergeReports(positions);
                        console.warn(
                            `[fa2-twins] ${label}: re-synchronised twin over ${states.length} states: record ratio ${record.worst.toExponential(3)} (${record.worstLabel}), positions ratio ${position.worst.toExponential(3)}`,
                        );
                        assertCheckPasses(record);
                        assertCheckPasses(position);
                    } finally {
                        withSubgroups.release(s);
                        withoutSubgroups.release(s);
                    }
                },
                CASE_TIMEOUT,
            );

            it(
                `${label}: the workgroup twin's ${TRACE_ITERATIONS}-iteration trajectory agrees with the re-synchronised f32 / f64 oracles; the free-running ${TRACE_TEN}-iteration twin trace ${paper ? "printed (chaotic)" : "within fa2-twins.trace"}`,
                async (t) => {
                    requireGpu(t);
                    if (NO_SUBGROUPS_PASS) {
                        t.skip("GRAPHTY_GPU_NO_SUBGROUPS=1: both contexts are the workgroup twin");
                        return;
                    }
                    const s = paritySnapshot(graph, gpuScale(), false);
                    try {
                        const start = startPositions(s, BASE_OPTIONS, false);
                        const a = await resyncTrace(withSubgroups, s, start, BASE_OPTIONS, tuning, TRACE_ITERATIONS);
                        const b = await resyncTrace(withoutSubgroups, s, start, BASE_OPTIONS, tuning, TRACE_ITERATIONS);
                        const twin = resyncReports(b, `${label}/no-subgroups`);
                        console.warn(
                            `[fa2-twins] ${label}: workgroup twin re-synchronised ratio vs f32 ${twin.f32.worst.toExponential(3)}, vs f64 ${twin.f64.worst.toExponential(3)}`,
                        );
                        assertCheckPasses(twin.f32);
                        assertCheckPasses(twin.f64);
                        const err = stageError(
                            false,
                            traceValues(a.trace, 0, TRACE_TEN),
                            traceValues(b.trace, 0, TRACE_TEN),
                        );
                        const report: CheckReport = {
                            worst: ratioOf(err.rel, toleranceOf("fa2-twins.trace")),
                            worstLabel: `${label}: free-running trace`,
                            samples: 6 * TRACE_TEN,
                        };
                        // the positions after TRACE_ITERATIONS and the paper-mode trace amplify the controller's
                        // reduction noise chaotically (spec 7.16; G3-F3), so they are printed, not asserted
                        const pos = stageError(true, a.positions, b.positions);
                        console.warn(
                            `[fa2-twins] ${label}: free-running trace over ${TRACE_TEN} iterations rel ${err.rel.toExponential(3)} ratio ${report.worst.toExponential(3)}${paper ? " (paper mode: informational)" : ""}; positions after ${TRACE_ITERATIONS} iterations differ by rel ${pos.rel.toExponential(3)}`,
                        );
                        if (!paper) {
                            assertCheckPasses(report);
                        }
                    } finally {
                        withSubgroups.release(s);
                        withoutSubgroups.release(s);
                    }
                },
                CASE_TIMEOUT,
            );
        }
    }

    it(
        "writes the workgroup twin's outputs of the UNSCALED random1k / karate as `<class>-no-subgroups` noise fixtures (GRAPHTY_NOISE_FLOOR_WRITE=1 only)",
        async (t) => {
            requireGpu(t);
            const twinClass = `${adapterClass(withoutSubgroups.caps)}${TWIN_SUFFIX}`;
            // the stage members (paper, random1k)
            const paperInputs = noiseInputs();
            try {
                const capture = await captureAllStages(
                    withoutSubgroups,
                    paperInputs.s,
                    paperInputs.start,
                    paperInputs.options,
                    paperInputs.tuning,
                    null,
                );
                for (const key of TWIN_MEMBERS) {
                    writeNoiseFixture(
                        NOISE_FIXTURES[key].kernel,
                        NOISE_FIXTURES[key].fixture,
                        twinClass,
                        capture[key].values,
                        "f32",
                    );
                }
            } finally {
                withoutSubgroups.release(paperInputs.s);
            }
            // the states10 member: both twins and the f64 reference from the same ten oracle-trajectory states of
            // each mode (twenty geometries)
            const st = noiseInputs();
            try {
                const feature: number[] = [];
                const workgroup: number[] = [];
                const reference: number[] = [];
                for (const tuning of [PAPER, NETWORKX]) {
                    for (const state of oracleStates(st.s, st.start, st.options, tuning, TRACE_TEN)) {
                        feature.push(...(await oneIterationFrom(withSubgroups, st.s, state, tuning)).record);
                        workgroup.push(...(await oneIterationFrom(withoutSubgroups, st.s, state, tuning)).record);
                        reference.push(...oracleRecordFrom(st.s, state, st.options, tuning));
                    }
                }
                const { kernel, fixture } = NOISE_FIXTURES.states10;
                if (!NO_SUBGROUPS_PASS) {
                    writeNoiseFixture(kernel, fixture, adapterClass(withSubgroups.caps), feature, "f32");
                }
                writeNoiseFixture(kernel, fixture, twinClass, workgroup, "f32");
                writeNoiseFixture(kernel, fixture, ORACLE_F64_CLASS, reference, "f32");
            } finally {
                withSubgroups.release(st.s);
                withoutSubgroups.release(st.s);
            }
            // the free-running networkx trace10 twin member
            const nx = noiseInputs(NETWORKX);
            try {
                const run = await resyncTrace(withoutSubgroups, nx.s, nx.start, nx.options, nx.tuning, TRACE_TEN);
                writeNoiseFixture(
                    NOISE_FIXTURES.trace10.kernel,
                    NOISE_FIXTURES.trace10.fixture,
                    twinClass,
                    traceValues(run.trace, 0, TRACE_TEN),
                    "f32",
                );
            } finally {
                withoutSubgroups.release(nx.s);
            }
            // the paper-mode re-synchronised members of the twin, with the oracles that followed ITS states
            for (const member of ["resync50", "resyncKarate50"] as const) {
                const inputs = noiseInputs(PAPER, member === "resync50" ? "random1k" : "karate");
                try {
                    const run = await resyncTrace(
                        withoutSubgroups,
                        inputs.s,
                        inputs.start,
                        inputs.options,
                        inputs.tuning,
                        TRACE_ITERATIONS,
                    );
                    const { kernel, fixture } = NOISE_FIXTURES[member];
                    writeNoiseFixture(kernel, fixture, twinClass, resyncValues(run.gpu), "f32");
                    writeNoiseFixture(
                        kernel,
                        fixture,
                        resyncOracleClass(ORACLE_F32_CLASS, twinClass),
                        resyncValues(run.f32),
                        "f32",
                    );
                    writeNoiseFixture(
                        kernel,
                        fixture,
                        resyncOracleClass(ORACLE_F64_CLASS, twinClass),
                        resyncValues(run.f64),
                        "f32",
                    );
                } finally {
                    withoutSubgroups.release(inputs.s);
                }
            }
        },
        CASE_TIMEOUT,
    );
});
