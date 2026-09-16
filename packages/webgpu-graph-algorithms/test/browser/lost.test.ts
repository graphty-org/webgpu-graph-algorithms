/**
 * The browser leg of spec 5.7 / 11.3 "both runtimes": device.destroy() while a submitted CommandBatch's readback is
 * pending rejects that readback with E_DEVICE_LOST, ctx.state is "lost", the residency is cleared without uncaptured
 * errors, every registered onLost listener ran; a new context from requestGpuContext() (a fresh adapter) uploads karate
 * and runs degree afterwards; nothing is left in the pending-error slot. Own contexts (no onError) so the drain of the
 * afterEach hook has nothing to see; both are disposed here.
 */

import { degree } from "../../src/algorithms/degree.js";
import { requestGpuContext } from "../../src/browser/index.js";
import type { GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode, isWebGpuGraphError } from "../../src/errors.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import type { Binding } from "../../src/types/memory.js";
import { KARATE_EDGES, snapshotOf } from "../helpers/graphs.js";
import { outDegreeOracle } from "../oracle/degree.js";
import { browserGpu, requireBrowserGpu } from "../setup/browser.js";

function scratch(ctx: GpuContext, byteLength: number, label: string): GPUBuffer {
    return ctx.device.createBuffer({
        label,
        size: byteLength,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
    });
}

function whole(buffer: GPUBuffer): Binding {
    return { buffer, offset: 0, size: buffer.size, window: null };
}

function caught(fn: () => unknown): unknown {
    try {
        fn();
    } catch (error) {
        return error;
    }
    return null;
}

const ITEMS = 4_194_304;

describe("device loss mid-batch in the browser (spec 5.7, 2.2 step 5)", () => {
    it("device.destroy() mid-batch rejects the readback with E_DEVICE_LOST; lost state; residency cleared; listeners ran; recovery from a fresh adapter", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await requestGpuContext({ label: "browser-lost" });
        const s = snapshotOf(KARATE_EDGES, { label: "karate" });
        ctx.residency.core(s);
        expect(ctx.residency.stats().snapshots).toBe(1);
        const ran: string[] = [];
        ctx.onLost((info) => {
            ran.push(`A:${info.reason}`);
        });
        ctx.onLost((info) => {
            ran.push(`B:${info.reason}`);
        });

        const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
        const ring = new UniformRing(ctx.device, ctx.allocator, 1, "browser/lost/ring");
        const dst = scratch(ctx, ITEMS * 4, "browser/lost/dst");
        ring.write(0, FILL_PARAMS, { count: ITEMS, value: 1, mode: 1 });
        ring.flush();
        const batch = new CommandBatch(ctx, "lost-mid-batch");
        const pass = batch.pass("fill");
        const bound = kernel.bind({ dst: whole(dst), P: ring.binding(FILL_PARAMS) });
        for (let k = 0; k < 4; k++) {
            kernel.dispatch(pass, bound, plan1d(ITEMS, ctx.workgroupSize, ctx.caps), [0]);
        }
        batch.readback(dst, 0, 64);
        const submitted = batch.submit();
        expect(ctx.readback.borrowed).toBe(1);

        ctx.device.destroy();
        const outcome = await submitted.readback.then(
            () => null,
            (error: unknown) => error,
        );
        expect(isWebGpuGraphError(outcome)).toBe(true);
        expect(hasErrorCode(outcome, "E_DEVICE_LOST")).toBe(true);
        const info = await ctx.lost;
        expect(info.reason).toBe("destroyed");
        console.warn(
            `[browser/lost] mid-batch readback rejected E_DEVICE_LOST on ${browserGpu()} (reason=${info.reason})`,
        );
        expect(ctx.state).toBe("lost");
        expect(ran).toEqual(["A:destroyed", "B:destroyed"]);
        expect(ctx.residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
        expect(ctx.residency.residentBytes).toBe(0);
        expect(ctx.readback.borrowed).toBe(0);
        expect(
            hasErrorCode(
                caught(() => ctx.assertReady()),
                "E_DEVICE_LOST",
            ),
        ).toBe(true);
        expect(ctx.takePendingError()).toBeNull();
        const late = new CommandBatch(ctx, "after-loss");
        expect(
            hasErrorCode(
                caught(() => late.submit()),
                "E_DEVICE_LOST",
            ),
        ).toBe(true);
        ctx.dispose();
        expect(ctx.state).toBe("disposed");

        // recovery: requestGpuContext() requests a FRESH adapter (the old one is consumed, spec 2.2 step 1)
        const fresh = await requestGpuContext({ label: "browser-recovered" });
        try {
            expect(fresh.state).toBe("ready");
            const result = await degree(fresh, s);
            expect(result.length).toBe(34);
            expect(Array.from(result)).toEqual(Array.from(outDegreeOracle(s)));
            fresh.release(s);
            expect(fresh.residency.residentBytes).toBe(0);
            await fresh.device.queue.onSubmittedWorkDone();
            expect(fresh.takePendingError()).toBeNull();
        } finally {
            fresh.dispose();
        }
    });
});
