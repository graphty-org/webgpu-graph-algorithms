/**
 * The tests are tested (spec 11.9 item 1, 11.5): each SABOTAGE.degree mutation, spliced into the normative body and
 * compiled on a fresh context, makes the SAME degree check that passes on the real kernel fail by >= minFactor
 * (Infinity for a bitwise test: any mismatch). The row-window leg of the check is what catches the ignored rebase
 * (`degree()` itself always dispatches with arcBase = 0).
 */

import { KERNELS, kernelSpec } from "../../src/kernels.js";
import { degreeSabotageReport } from "../helpers/degree-check.js";
import { assertCheckPasses, SABOTAGE, withSabotage } from "../helpers/sabotage.js";
import { acquire, requireGpu } from "../setup/gpu.js";

const rows = SABOTAGE.degree ?? [];

describe("sabotage: degree (spec 11.9 item 1)", () => {
    it("carries the four P1 rows, each naming the degree test", () => {
        expect(rows.map((m) => m.name)).toEqual([
            "last-row-skipped",
            "use-perm-select-swapped",
            "rebase-ignored",
            "target-counted-twice",
        ]);
        for (const m of rows) {
            expect(m.test).toBe("test/algorithms/degree.test.ts");
            expect(m.minFactor).toBe(10);
        }
    });

    it("the check passes on the normative body (the control)", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        try {
            const report = await degreeSabotageReport(ctx);
            assertCheckPasses(report);
            expect(report.worst).toBe(0);
            expect(report.samples).toBe(68);
        } finally {
            ctx.dispose();
        }
    });

    for (const m of rows) {
        it(`${m.name}: the same check fails by >= ${m.minFactor}x`, async (t) => {
            requireGpu(t);
            const report = await withSabotage("degree", m, (ctx) => degreeSabotageReport(ctx));
            expect(() => {
                assertCheckPasses(report);
            }).toThrow(/check failed/);
            expect(report.worst, report.worstLabel).toBeGreaterThanOrEqual(m.minFactor);
        });
    }

    it("the normative body is restored after every mutation", () => {
        expect(kernelSpec("degree").body).toBe(KERNELS.degree.body);
    });
});
