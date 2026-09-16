/**
 * The sabotage matrix of the P3 kernels (spec 11.9 item 1; spec 13 rule f; contract 5.5 fa2.test.ts): every row of
 * SABOTAGE["fa2-stats-finalize"], ["fa2-attraction"] and ["fa2-integrate"] is spliced into its kernel body through
 * setKernelBodyOverride (a FRESH context per mutant: the pipeline key does not include the body) and the SAME check the
 * row's `test` field names -- the stage capture of test/layouts/fa2-inspect.test.ts or the re-synchronised trace
 * reports of test/layouts/fa2-trace-parity.test.ts (resyncReports over TRACE_ITERATIONS in paper mode, the leg that
 * test asserts in that mode; P3-T5 PLAN DECISION 17), with the same traced tolerances -- must fail by at least
 * minFactor (10) x the tolerance. The P1 K3 / K4 rows are additionally run against the P3 force report and the P3
 * trace reports (contract 5.5), and so are the SABOTAGE_P3_ADDENDUM rows on those kernels (the K4 knife-edge row of
 * CONTRACT DECISION K4-1, G3-F6; the review fix of 2026-09-16). A mutant that survives is a bug in the test suite and
 * blocks G3 exactly as a failing test does. Unscaled karate on every adapter: sabotage is about the test, not the
 * hardware (lavapipe is enough).
 */

import type { GpuContext } from "../../src/context.js";
import { type KernelId, KERNELS } from "../../src/kernels.js";
import {
    BASE_OPTIONS,
    captureAllStages,
    forceReport,
    PAPER,
    paritySnapshot,
    resyncReports,
    resyncTrace,
    STAGE_KERNEL,
    STAGE_KEYS,
    stageReport,
    startPositions,
    TRACE_ITERATIONS,
} from "../helpers/fa2-parity.js";
import {
    assertCheckPasses,
    type CheckReport,
    mergeReports,
    type Mutation,
    SABOTAGE,
    SABOTAGE_EXEMPT,
    SABOTAGE_P3_ADDENDUM,
    SABOTAGE_PHASES,
    withSabotage,
} from "../helpers/sabotage.js";
import { acquire, requireGpu } from "../setup/gpu.js";

const CASE_TIMEOUT = 300_000;
const P3_KERNELS: readonly KernelId[] = ["fa2-stats-finalize", "fa2-attraction", "fa2-integrate"];
const P1_FA2_KERNELS: readonly KernelId[] = ["fa2-repulsion-exact", "fa2-speed-finalize"];

/** The rows of a kernel this file runs: SABOTAGE's and, on the P1 FA2 kernels, the P3 addendum's. */
function rowsOf(id: KernelId): readonly Mutation[] {
    return [...(SABOTAGE[id] ?? []), ...(P1_FA2_KERNELS.includes(id) ? (SABOTAGE_P3_ADDENDUM[id] ?? []) : [])];
}

const s = paritySnapshot("karate", 1, false);
const start = startPositions(s, BASE_OPTIONS, false);

/** The stage-parity check of fa2-inspect.test.ts restricted to the stages of one kernel (the worst over them). */
async function stageCheck(ctx: GpuContext, id: KernelId): Promise<CheckReport> {
    const capture = await captureAllStages(ctx, s, start, BASE_OPTIONS, PAPER, null);
    const keys = STAGE_KEYS.filter((key) => STAGE_KERNEL[key] === id);
    if (keys.length === 0) {
        throw new Error(`${id}: no inspect stage`);
    }
    return mergeReports(keys.map((key) => stageReport(capture, key)));
}

/**
 * The trace-parity check of fa2-trace-parity.test.ts in paper mode: the re-synchronised legs against the f32 and the
 * f64 oracle over TRACE_ITERATIONS iterations (the same helper, horizon and tolerances that test asserts). Every
 * trace-sensitive row shows within the first iterations: the K4 halving row at iteration 2, where karate's swing /
 * traction is already 2.15 > 2 (the overshoot after the first expansion), the rise and swap rows at iteration 1,
 * K5's store-old row at iteration 2 -- and, the oracle being re-seeded from the GPU's state before every iteration,
 * the mutant's error is measured against the same-state oracle at every one of the 50, never washed out or blown up
 * by the chaotic divergence of the trajectory (G3-F3).
 */
async function traceCheck(ctx: GpuContext): Promise<CheckReport> {
    const run = await resyncTrace(ctx, s, start, BASE_OPTIONS, PAPER, TRACE_ITERATIONS);
    const r = resyncReports(run, "sabotage/karate");
    return mergeReports([r.f32, r.f64]);
}

/** The force-parity check of fa2-force-parity.test.ts. */
async function forceCheck(ctx: GpuContext): Promise<CheckReport> {
    return (await forceReport(ctx, s, start, BASE_OPTIONS, PAPER, null, "sabotage/karate")).report;
}

/** The check a row is measured with: by its `test` field for the P3 rows, by kernel for the P1 K3 / K4 rows. */
function checkFor(id: KernelId, row: Mutation): (ctx: GpuContext) => Promise<CheckReport> {
    if (row.test.endsWith("fa2-trace-parity.test.ts")) {
        return traceCheck;
    }
    if (row.test.endsWith("fa2-inspect.test.ts")) {
        return (ctx: GpuContext): Promise<CheckReport> => stageCheck(ctx, id);
    }
    if (id === "fa2-repulsion-exact") {
        return forceCheck;
    }
    if (id === "fa2-speed-finalize") {
        return traceCheck;
    }
    throw new Error(`${id}/${row.name}: no P3 check for test ${row.test}`);
}

describe("sabotage: the ForceAtlas2 kernels against the P3 parity checks (spec 11.9 item 1)", () => {
    it("SABOTAGE_PHASES lists P3 and every non-exempt P3 kernel carries at least three rows naming a P3 test", () => {
        expect(SABOTAGE_PHASES).toContain("P3");
        for (const id of P3_KERNELS) {
            const rows = SABOTAGE[id] ?? [];
            expect(rows.length, `${id}: rows`).toBeGreaterThanOrEqual(3);
            for (const row of rows) {
                expect(row.test.startsWith("test/layouts/fa2-"), `${id}/${row.name}: names a P3 layout test`).toBe(
                    true,
                );
                expect(row.minFactor).toBeGreaterThanOrEqual(10);
                expect(KERNELS[id].body.split(row.find).length, `${id}/${row.name}: find occurs once`).toBe(2);
            }
        }
        expect(SABOTAGE_EXEMPT).toContain("fa2-to-scene");
        expect(SABOTAGE["fa2-to-scene"]).toBeUndefined();
    });

    it("the P3 addendum names only P1 FA2 kernels, K4's knife-edge row by name, each row a P3 layout test, a unique find and a name unused by SABOTAGE", () => {
        expect(Object.keys(SABOTAGE_P3_ADDENDUM).every((id) => P1_FA2_KERNELS.includes(id as KernelId))).toBe(true);
        expect((SABOTAGE_P3_ADDENDUM["fa2-speed-finalize"] ?? []).map((row) => row.name)).toEqual([
            "halving-at-exact-equality",
        ]);
        for (const id of P1_FA2_KERNELS) {
            const names = new Set((SABOTAGE[id] ?? []).map((row) => row.name));
            for (const row of SABOTAGE_P3_ADDENDUM[id] ?? []) {
                expect(names.has(row.name), `${id}/${row.name}: a name SABOTAGE does not use`).toBe(false);
                expect(row.test.startsWith("test/layouts/fa2-"), `${id}/${row.name}: names a P3 layout test`).toBe(
                    true,
                );
                expect(row.minFactor).toBeGreaterThanOrEqual(10);
                expect(row.replace, `${id}/${row.name}: replace differs from find`).not.toBe(row.find);
                expect(KERNELS[id].body.split(row.find).length, `${id}/${row.name}: find occurs once`).toBe(2);
            }
        }
    });

    it(
        "the pristine kernels pass every check (the baseline the mutants are measured against)",
        async (t) => {
            requireGpu(t);
            const ctx = await acquire({ label: "sabotage/fa2/baseline" });
            for (const id of P3_KERNELS) {
                const report = await stageCheck(ctx, id);
                console.warn(
                    `[sabotage] baseline ${id}: ratio ${report.worst.toExponential(3)} at ${report.worstLabel}`,
                );
                assertCheckPasses(report);
            }
            const force = await forceCheck(ctx);
            const trace = await traceCheck(ctx);
            console.warn(
                `[sabotage] baseline force ratio ${force.worst.toExponential(3)}, trace ratio ${trace.worst.toExponential(3)}`,
            );
            assertCheckPasses(force);
            assertCheckPasses(trace);
        },
        CASE_TIMEOUT,
    );

    for (const id of [...P3_KERNELS, ...P1_FA2_KERNELS]) {
        for (const row of rowsOf(id)) {
            it(
                `${id}/${row.name}: fails its check by >= ${row.minFactor}x the tolerance`,
                async (t) => {
                    requireGpu(t);
                    const check = checkFor(id, row);
                    const report = await withSabotage(id, row, check);
                    console.warn(
                        `[sabotage] ${id}/${row.name}: ratio ${report.worst.toExponential(3)} at ${report.worstLabel}`,
                    );
                    expect(report.worst, `${id}/${row.name}: detection factor`).toBeGreaterThanOrEqual(row.minFactor);
                },
                CASE_TIMEOUT,
            );
        }
    }
});
