/**
 * The browser twin of test/kernel/wgsl-compile.test.ts (spec 5.1, 11.3, 11.6 item 6; contract 5.5 P2): every
 * OVERRIDE_MATRIX case compiles on Chromium (SwiftShader on the default lane, NVIDIA on the GPU lane), twin cases
 * also on a second context created without the subgroups feature (the workgroup twin's browser leg, spec 11.6 item
 * 5), which catches uniform-layout bugs Dawn-node's uniform_buffer_standard_layout masks. PLAN DECISION (P2-T2): the
 * file also runs the thread-per-row segmented-reduce on the weighted random1k graph so the browser adapters' noise
 * fixtures and oracle rows exist (G2: "noise-floor rows recorded from the three adapters"); the cross-adapter
 * comparison itself runs in the node primitive test against every committed fixture.
 */

import { type TestContext } from "vitest";

import { type GpuContext } from "../../src/context.js";
import { type KernelId, KERNELS, kernelSpec } from "../../src/kernels.js";
import { expectAllClose, expectBitwiseEqual, maxRelError } from "../helpers/matchers.js";
import {
    adapterClassBrowser,
    recordNoiseRowBrowser,
    writeNoiseFixtureBrowser,
} from "../helpers/noise-floor-browser.js";
import { entryOf, OVERRIDE_MATRIX, type OverrideCase, SEGMENTED_REDUCE_SNIPPETS } from "../helpers/override-matrix.js";
import {
    maxAbsError,
    oracleValueOf,
    relTolerance,
    runSegmentedReduce,
    SR_ABS_FLOOR,
    weightedRandom,
} from "../helpers/segmented-reduce.js";
import { segmentedReduceOracle } from "../oracle/segmented-reduce.js";
import { acquireBrowser, requireBrowserGpu } from "../setup/browser.js";

describe("compile matrix on Chromium", () => {
    let ctx: GpuContext | null = null;
    let twin: GpuContext | null = null;

    async function contexts(t: TestContext): Promise<{ ctx: GpuContext; twin: GpuContext }> {
        await requireBrowserGpu(t);
        const a = ctx ?? (await acquireBrowser({ label: "compile-matrix" }));
        ctx = a;
        const b = twin ?? (await acquireBrowser({ optionalFeatures: [], label: "compile-matrix-twin" }));
        twin = b;
        return { ctx: a, twin: b };
    }

    async function compileAll(context: GpuContext, cases: readonly OverrideCase[]): Promise<void> {
        for (const c of cases) {
            const kernel = await context.pipelines.kernel(kernelSpec(c.id, c.overrides, c.snippets));
            expect(kernel.entryPoint).toBe(entryOf(c.id).entryPoint);
        }
    }

    it("the feature context has subgroups (both CI browser adapters expose it) and the twin does not", async (t) => {
        const { ctx: a, twin: b } = await contexts(t);
        expect(a.caps.features.has("subgroups")).toBe(true);
        expect(b.caps.features.has("subgroups")).toBe(false);
    });

    for (const id of Object.keys(KERNELS) as KernelId[]) {
        it(`compiles every case of ${id} on Chromium (twin cases on both feature settings)`, async (t) => {
            const { ctx: a, twin: b } = await contexts(t);
            const cases = OVERRIDE_MATRIX.filter((c) => c.id === id);
            expect(cases.length).toBeGreaterThan(0);
            await compileAll(a, cases);
            const twins = cases.filter((c) => c.twin);
            if (twins.length > 0) {
                await compileAll(b, twins);
            }
        });
    }

    it("compiled the whole matrix once (bounded: pipelines.size equals the case count)", async (t) => {
        const { ctx: a } = await contexts(t);
        expect(a.pipelines.size).toBe(OVERRIDE_MATRIX.length);
    });

    it("segmented-reduce thread-per-row on random1k (weighted sum): oracle within the analytic bound, twice bitwise, this adapter's noise fixture and oracle row written", async (t) => {
        const { ctx: a } = await contexts(t);
        const s = weightedRandom(1000, 5000, 7);
        const snippet = SEGMENTED_REDUCE_SNIPPETS.weight;
        const first = await runSegmentedReduce(a, s, "sum", snippet);
        const second = await runSegmentedReduce(a, s, "sum", snippet);
        expectBitwiseEqual(first, second, "twice");
        const expected = segmentedReduceOracle(s, oracleValueOf(snippet), "sum");
        expectAllClose(first, expected, { rel: relTolerance(s, "sum"), abs: 0 }, "random1k weighted sum");
        const mine = adapterClassBrowser(a.caps);
        const oracleErr = maxRelError(first, expected, SR_ABS_FLOOR);
        console.warn(`[segmented-reduce] ${mine}: oracle-f64 maxRelError ${oracleErr.toExponential(3)}`);
        const written = await writeNoiseFixtureBrowser("segmented-reduce", "random1k", mine, Array.from(first), "f32");
        expect(typeof written).toBe("string");
        await recordNoiseRowBrowser({
            id: `segmented-reduce.sum.oracle-f64.${mine}`,
            kernel: "segmented-reduce",
            fixture: "random1k",
            comparison: "oracle-f64",
            a: mine,
            b: "oracle-f64",
            maxRelError: oracleErr,
            maxAbsError: maxAbsError(first, expected),
            samples: s.nodeCount,
        });
        a.release(s);
    });
});
