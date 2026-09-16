/**
 * Force parity (spec 11.4; contract 5.5 fa2-force-parity.test.ts): one iteration K2 + K3, `force` read through
 * inspect() after K3, against the f64 oracle's force stage with the floored denominator
 * max(|F_cpu(i)|, 1e-3 x max_j |F_cpu(j)|) (DEPARTURE-6) on karate, grid10, star200 and random1k x every option
 * variation (PLAN DECISION 8: 175 cases), each run twice and bitwise-equal first (spec 11.9 item 4). The tolerance
 * comes from benchmarks/results/noise-floor.json (fa2-force-parity), never from a literal; a recording run writes
 * this adapter's force on the unscaled random1k as a noise fixture together with the f64 reference.
 */

import type { GpuContext } from "../../src/context.js";
import {
    caseOptions,
    forceParityCases,
    forceReport,
    noiseInputs,
    paritySnapshot,
    pinIndex,
    pinMask,
    startPositions,
    toleranceOf,
} from "../helpers/fa2-parity.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { assertCheckPasses, type CheckReport, mergeReports } from "../helpers/sabotage.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

/** PLAN DECISION 14: 175 cases on lavapipe need more than the node project's 30 s default. */
const CASE_TIMEOUT = 300_000;

describe("FA2 force parity: K2 + K3 vs the f64 oracle (spec 11.4)", () => {
    let ctx: GpuContext;
    const reports: CheckReport[] = [];

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-force-parity" });
    });

    it("the tolerance is traced to the noise-floor file", () => {
        expect(toleranceOf("fa2-force-parity")).toBeGreaterThan(0);
        expect(toleranceOf("fa2-force-parity")).toBeLessThanOrEqual(1e-4);
    });

    for (const c of forceParityCases()) {
        it(
            `${c.name}: force within the traced tolerance, twice bitwise`,
            async (t) => {
                requireGpu(t);
                const s = paritySnapshot(c.graph, gpuScale(), c.weighted);
                try {
                    const options = caseOptions(c, s);
                    const mask = c.pinned ? pinMask(s.nodeCount, pinIndex(s.nodeCount)) : null;
                    const start = startPositions(s, options, c.unseeded);
                    const first = await forceReport(ctx, s, start, options, c.tuning, mask, c.name);
                    // the second run starts from the array the first run's load() seeded (identical for a seeded case)
                    const second = await forceReport(ctx, s, first.scene, options, c.tuning, mask, `${c.name} (run 2)`);
                    expectBitwiseEqual(first.force, second.force, `${c.name}: run 1 vs run 2`);
                    if (c.pinned) {
                        // the pinned node's force is computed (its swing is excluded, not its force)
                        const i = pinIndex(s.nodeCount);
                        const magnitude = Math.hypot(
                            first.force[3 * i],
                            first.force[3 * i + 1],
                            first.force[3 * i + 2],
                        );
                        expect(magnitude, `${c.name}: the pinned node has a force`).toBeGreaterThan(0);
                    }
                    if (c.graph === "isolated34") {
                        // arcCount 0: repulsion + gravity only, every force finite and non-zero
                        expect(first.force.every((v) => Number.isFinite(v))).toBe(true);
                        expect(first.force.some((v) => v !== 0)).toBe(true);
                    }
                    reports.push(first.report);
                    assertCheckPasses(first.report);
                } finally {
                    ctx.release(s);
                }
            },
            CASE_TIMEOUT,
        );
    }

    it(
        "the UNSCALED random1k (the noise member's inputs) passes too, and the matrix's worst ratio is printed",
        async (t) => {
            requireGpu(t);
            const { s, start, options, tuning } = noiseInputs();
            try {
                const { report } = await forceReport(ctx, s, start, options, tuning, null, "noise/random1k");
                reports.push(report);
                assertCheckPasses(report);
            } finally {
                ctx.release(s);
            }
            const worst = mergeReports(reports);
            console.warn(
                `[fa2-force-parity] ${reports.length} cases; worst error / tolerance ratio ${worst.worst.toExponential(3)} at ${worst.worstLabel}`,
            );
        },
        CASE_TIMEOUT,
    );
});
