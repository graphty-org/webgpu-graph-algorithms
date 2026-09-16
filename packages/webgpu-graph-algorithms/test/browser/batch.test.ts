/**
 * The browser leg of CommandBatch (contract 5.5 P2 row, spec 5.7): the 8-dispatch batch submits once and its two
 * readbacks land at their offsets; discard() returns the slot; and a deliberately bad bind group is delivered
 * ASYNCHRONOUSLY in Chromium -- either the batch takes it after its map (its readback rejects E_VALIDATION with the
 * batchId) or the context's next assertReady() throws it after onSubmittedWorkDone(); exactly one of the two happens,
 * and the error carries the bind group's `fill/2` label. The bad-bind-group test creates its own context through
 * requestGpuContext() (no onError, so the error goes to the pending slot) and disposes it.
 */

import { requestGpuContext } from "../../src/browser/index.js";
import type { GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode, isWebGpuGraphError } from "../../src/errors.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import type { Binding } from "../../src/types/memory.js";
import { acquireBrowser, browserGpu, requireBrowserGpu } from "../setup/browser.js";

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

function countSubmits(queue: GPUQueue): { count(): number; restore(): void } {
    let n = 0;
    const original = queue.submit.bind(queue);
    queue.submit = (commandBuffers: Iterable<GPUCommandBuffer>): undefined => {
        n += 1;
        original(commandBuffers);
        return undefined;
    };
    return {
        count: (): number => n,
        restore(): void {
            queue.submit = original;
        },
    };
}

describe("CommandBatch in the browser (spec 5.8, 5.7)", () => {
    it("a batch of 8 dispatches submits once; two readbacks land at their offsets; discard returns the slot", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser({ label: "browser-batch" });
        const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
        const ring = new UniformRing(ctx.device, ctx.allocator, 8, "browser/batch/ring");
        const params = ring.binding(FILL_PARAMS);
        const a = scratch(ctx, 64, "browser/batch/a");
        const b = scratch(ctx, 128, "browser/batch/b");
        const submits = countSubmits(ctx.device.queue);
        try {
            for (let k = 0; k < 8; k++) {
                ring.write(k, FILL_PARAMS, { count: k % 2 === 0 ? 16 : 32, value: 100 + k, mode: 0 });
            }
            ring.flush();
            const batch = new CommandBatch(ctx, "eight");
            const boundA = kernel.bind({ dst: whole(a), P: params });
            const boundB = kernel.bind({ dst: whole(b), P: params });
            const pass = batch.pass("fill");
            for (let k = 0; k < 8; k++) {
                const even = k % 2 === 0;
                kernel.dispatch(pass, even ? boundA : boundB, plan1d(even ? 16 : 32, ctx.workgroupSize, ctx.caps), [
                    ring.offsetOf(k),
                ]);
            }
            expect(batch.dispatches).toBe(8);
            const requestA = batch.readback(a, 0, 64);
            const requestB = batch.readback(b, 0, 128);
            expect(requestA.offset).toBe(0);
            expect(requestB.offset).toBe(64);
            expect(ctx.readback.borrowed).toBe(1);
            const submitted = batch.submit();
            expect(submits.count()).toBe(1);
            const bytes = await submitted.readback;
            expect(bytes.byteLength).toBe(192);
            expect(Array.from(new Uint32Array(bytes, requestA.offset, 16))).toEqual(new Array<number>(16).fill(106));
            expect(Array.from(new Uint32Array(bytes, requestB.offset, 32))).toEqual(new Array<number>(32).fill(107));
            expect(ctx.readback.borrowed).toBe(0);
            expect(submits.count()).toBe(1);
            expect(
                hasErrorCode(
                    caught(() => batch.submit()),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);

            const discarded = new CommandBatch(ctx, "discarded", 3);
            kernel.dispatch(discarded.pass("fill"), boundA, plan1d(16, ctx.workgroupSize, ctx.caps), [
                ring.offsetOf(0),
            ]);
            discarded.readback(a, 0, 64);
            const handle = discarded.submit();
            expect(handle.generation).toBe(3);
            handle.discard();
            expect((await handle.readback).byteLength).toBe(0);
            expect(ctx.readback.borrowed).toBe(0);
            // the discarded batch ran: a now holds 100 (slot 0)
            const words = new Uint32Array(await ctx.readback.read(a, 64));
            expect(Array.from(words)).toEqual(new Array<number>(16).fill(100));
        } finally {
            submits.restore();
            ring.destroy();
            a.destroy();
            b.destroy();
        }
    });

    it("a deliberately bad bind group: rejected by its own batch or thrown from the next assertReady() after onSubmittedWorkDone()", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await requestGpuContext({ label: "browser-batch-slot" });
        try {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const dst = scratch(ctx, 64, "browser/batch/bad-dst");
            const wrongUsage = ctx.allocator.createBuffer({
                label: "browser/batch/wrong-usage",
                size: 16,
                usage: BufferUsage.STORAGE | BufferUsage.COPY_DST,
            });
            try {
                const batch = new CommandBatch(ctx, "bad");
                const bound = kernel.bind({
                    dst: whole(dst),
                    P: { buffer: wrongUsage, offset: 0, size: 16, window: null },
                });
                kernel.dispatch(batch.pass("fill"), bound, plan1d(16, ctx.workgroupSize, ctx.caps), [0]);
                batch.readback(dst, 0, 64);
                const submitted = batch.submit();
                const outcome = await submitted.readback.then(
                    () => null,
                    (error: unknown) => error,
                );
                await ctx.device.queue.onSubmittedWorkDone();
                await new Promise<void>((resolve) => {
                    setTimeout(resolve, 0);
                });
                let delivered: unknown = outcome;
                let path = "readback";
                if (delivered === null) {
                    delivered = caught(() => ctx.assertReady());
                    path = "assertReady";
                }
                expect(isWebGpuGraphError(delivered)).toBe(true);
                if (!isWebGpuGraphError(delivered)) {
                    throw new Error("unreachable");
                }
                expect(delivered.code).toBe("E_VALIDATION");
                // Chromium quotes the bind group's label ("fill/2"); WebKit names the failing call (createBindGroup)
                expect(`${String(delivered.details.label)} ${String(delivered.details.message)}`).toMatch(
                    /fill\/2|createBindGroup/,
                );
                if (path === "readback") {
                    expect(delivered.details.batchId).toBe(submitted.id);
                }
                console.warn(
                    `[browser/batch] bad bind group on ${browserGpu()} delivered through ${path}: ${delivered.message}`,
                );
                // Chromium reports the bad bind group twice: the usage mismatch at createBindGroup and, as a separate
                // event, the command buffer it invalidated at Queue.submit. Both usually arrive before the map resolves
                // and are chained behind the first (PendingErrorSlot); on the Metal runner the second sometimes lands
                // after the batch took the slot, and then it is what the context's next public call throws (spec 5.7,
                // browsers) -- so nothing OR that consequence may be pending here, never an unrelated error
                const late = ctx.takePendingError();
                if (late !== null) {
                    expect(late.code).toBe("E_VALIDATION");
                    expect(String(late.details.message)).toContain(`batch/bad#${submitted.id}`);
                }
                expect(ctx.takePendingError()).toBeNull();
                expect(() => {
                    ctx.assertReady();
                }).not.toThrow();
                expect(ctx.readback.borrowed).toBe(0);
                expect(ctx.state).toBe("ready");
            } finally {
                ctx.allocator.destroy(wrongUsage);
                dst.destroy();
            }
        } finally {
            ctx.dispose();
        }
    });
});
