/**
 * Spec 11.9 item 1 for the segmented-reduce kernel (contract 5.5 P2): every SABOTAGE["segmented-reduce"] mutation
 * is spliced into the normative body, and the SAME check set that passes on the real kernel (worstFactor over
 * sabotageChecks, the primitive test's set) fails on the mutant by at least minFactor (10x the tolerance).
 */

import { type TestContext } from "vitest";

import { type GpuContext } from "../../src/context.js";
import { SABOTAGE, SABOTAGE_PHASES, sabotagedBody, withSabotage } from "../helpers/sabotage.js";
import { sabotageChecks, worstFactor } from "../helpers/segmented-reduce.js";
import { acquire, requireGpu } from "../setup/gpu.js";

const ROWS = SABOTAGE["segmented-reduce"] ?? [];

describe("sabotage: segmented-reduce", () => {
    let shared: GpuContext | null = null;

    async function context(t: TestContext): Promise<GpuContext> {
        requireGpu(t);
        const ctx = shared ?? (await acquire({ label: "sabotage-segmented-reduce" }));
        shared = ctx;
        return ctx;
    }

    it("has at least three rows, each find unique in the normative body, each naming this task's primitive test, and P2 is a gated phase", () => {
        expect(ROWS.length).toBeGreaterThanOrEqual(3);
        for (const mutation of ROWS) {
            expect(() => sabotagedBody("segmented-reduce", mutation)).not.toThrow();
            expect(sabotagedBody("segmented-reduce", mutation)).toContain(mutation.replace);
            expect(mutation.minFactor).toBeGreaterThanOrEqual(10);
            expect(mutation.test).toContain("segmented-reduce.test.ts");
        }
        expect(new Set(ROWS.map((m) => m.name)).size).toBe(ROWS.length);
        expect(SABOTAGE_PHASES).toContain("P2");
    });

    it("the real kernel passes the check set (factor < 1)", async (t) => {
        const ctx = await context(t);
        expect(await worstFactor(ctx, sabotageChecks())).toBeLessThan(1);
    });

    for (const mutation of ROWS) {
        it(`${mutation.name}: fails the segmented-reduce check set by >= ${mutation.minFactor}x`, async (t) => {
            requireGpu(t);
            const factor = await withSabotage("segmented-reduce", mutation, (ctx) =>
                worstFactor(ctx, sabotageChecks()),
            );
            console.warn(`[sabotage] segmented-reduce/${mutation.name}: factor ${factor.toExponential(2)}`);
            expect(factor).toBeGreaterThanOrEqual(mutation.minFactor);
        });
    }
});
