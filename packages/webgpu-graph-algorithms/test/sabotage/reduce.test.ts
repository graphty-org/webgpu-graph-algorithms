/**
 * The tests are tested (spec 11.9 item 1): each SABOTAGE.reduce mutation, spliced into the normative body and compiled
 * on a fresh context, makes the SAME reduce check that passes on the real kernel fail by >= minFactor: the FINAL level
 * writing out[0] (visible through the non-zero outOffset of every sabotage case), the swapped u32 / f32 min
 * identities (every input is >= 1), and the inclusive level-1 bound (the poison element past the range).
 */

import { KERNELS, kernelSpec } from "../../src/kernels.js";
import { reduceSabotageReport, SABOTAGE_REDUCE_CASES } from "../helpers/reduce-check.js";
import { assertCheckPasses, SABOTAGE, withSabotage } from "../helpers/sabotage.js";
import { acquire, requireGpu } from "../setup/gpu.js";

const rows = SABOTAGE.reduce ?? [];

describe("sabotage: reduce (spec 11.9 item 1)", () => {
    it("carries the four P1 rows, each naming the reduce test; the case set touches every mutated line", () => {
        expect(rows.map((m) => m.name)).toEqual([
            "final-writes-out-zero",
            "u32-min-identity-zero",
            "f32-min-identity-zero",
            "level-bound-inclusive",
        ]);
        for (const m of rows) {
            expect(m.test).toBe("test/primitives/reduce.test.ts");
            expect(m.minFactor).toBe(10);
        }
        expect(SABOTAGE_REDUCE_CASES.every((c) => c.outOffset > 0)).toBe(true);
        expect(SABOTAGE_REDUCE_CASES.every((c) => c.count % 256 !== 0)).toBe(true);
        expect(SABOTAGE_REDUCE_CASES.some((c) => c.op === "min" && c.dtype === "u32")).toBe(true);
        expect(SABOTAGE_REDUCE_CASES.some((c) => c.op === "min" && c.dtype === "f32")).toBe(true);
        expect(SABOTAGE_REDUCE_CASES.some((c) => c.dtype === "vec4f")).toBe(true);
    });

    it("the check passes on the normative body (the control)", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        try {
            const report = await reduceSabotageReport(ctx);
            assertCheckPasses(report);
            expect(report.samples).toBe(SABOTAGE_REDUCE_CASES.length);
        } finally {
            ctx.dispose();
        }
    });

    for (const m of rows) {
        it(`${m.name}: the same check fails by >= ${m.minFactor}x`, async (t) => {
            requireGpu(t);
            const report = await withSabotage("reduce", m, (ctx) => reduceSabotageReport(ctx));
            expect(() => {
                assertCheckPasses(report);
            }).toThrow(/check failed/);
            expect(report.worst, report.worstLabel).toBeGreaterThanOrEqual(m.minFactor);
        });
    }

    it("the normative body is restored after every mutation", () => {
        expect(kernelSpec("reduce").body).toBe(KERNELS.reduce.body);
    });
});
