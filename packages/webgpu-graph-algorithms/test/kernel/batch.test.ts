/**
 * CommandBatch (contract 3.9, spec 5.8, 5.7, 4.4): one encoder, one compute pass per phase, the staging copies at the
 * end, ONE queue.submit; the borrowed staging slot goes back to the ring when the readback settles; the pending-error
 * slot is checked right after submit so a bad bind group rejects THIS batch's readback under Dawn-node (synchronous
 * delivery, uncaptured-order-probe.mjs); discard() still awaits the map and resolves empty; ids start at 1 and grow.
 *
 * The bad-bind-group test needs a context WITHOUT an onError sink (acquire() installs one that collects into the
 * afterEach list), so it creates its own through GpuContext.create({ gpu }) on the worker's Dawn handle and disposes it.
 */

import { GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode, isWebGpuGraphError } from "../../src/errors.js";
import { type BatchHost, CommandBatch, type SubmittedBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { bindingOf, readU32, scratchBuffer, withContext } from "../helpers/device.js";
import { acquireRaw, requireGpu } from "../setup/gpu.js";

function caught(fn: () => unknown): unknown {
    try {
        fn();
    } catch (error) {
        return error;
    }
    return null;
}

/** Counts queue.submit calls until restore(). */
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

/** A context whose uncaptured errors land in the pending slot (no onError): the delivery path of spec 5.7. Disposed by the caller. */
async function slotContext(label: string): Promise<GpuContext> {
    const raw = await acquireRaw();
    return GpuContext.create({ gpu: raw.gpu, runtime: "node", label });
}

describe("CommandBatch (spec 5.8)", () => {
    it("a batch of 8 dispatches submits once; readback resolves the bytes of two requests at their offsets; dispatches count", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const ring = new UniformRing(ctx.device, ctx.allocator, 8, "test/batch/ring");
            const params = ring.binding(FILL_PARAMS);
            const a = scratchBuffer(ctx, 64, "test/batch/a");
            const b = scratchBuffer(ctx, 128, "test/batch/b");
            const submits = countSubmits(ctx.device.queue);
            try {
                // dispatch k fills a (even k, 16 words) or b (odd k, 32 words) with 100 + k; the last writer wins: a = 106, b = 107
                for (let k = 0; k < 8; k++) {
                    ring.write(k, FILL_PARAMS, { count: k % 2 === 0 ? 16 : 32, value: 100 + k, mode: 0 });
                }
                ring.flush();
                // the context satisfies BatchHost structurally (contract 3.9); the annotation keeps the interface referenced
                const host: BatchHost = ctx;
                const batch = new CommandBatch(host, "eight");
                expect(batch.id).toBe(1);
                expect(batch.generation).toBe(0);
                expect(batch.label).toBe("eight");
                expect(batch.dispatches).toBe(0);
                const boundA = kernel.bind({ dst: bindingOf(a), P: params });
                const boundB = kernel.bind({ dst: bindingOf(b), P: params });
                const pass = batch.pass("fill");
                for (let k = 0; k < 8; k++) {
                    const even = k % 2 === 0;
                    kernel.dispatch(pass, even ? boundA : boundB, plan1d(even ? 16 : 32, ctx.workgroupSize, ctx.caps), [
                        ring.offsetOf(k),
                    ]);
                }
                expect(batch.dispatches).toBe(8);
                // an empty plan records nothing (spec 5.6) and is not counted
                kernel.dispatch(pass, boundA, plan1d(0, ctx.workgroupSize, ctx.caps), [ring.offsetOf(0)]);
                expect(batch.dispatches).toBe(8);

                expect(ctx.readback.borrowed).toBe(0);
                const requestA = batch.readback(a, 0, 64);
                expect(ctx.readback.borrowed).toBe(1);
                const requestB = batch.readback(b, 0, 128);
                expect(ctx.readback.borrowed).toBe(1);
                expect(requestA).toEqual({ src: a, srcOffset: 0, byteLength: 64, offset: 0 });
                expect(requestB).toEqual({ src: b, srcOffset: 0, byteLength: 128, offset: 64 });
                expect(submits.count()).toBe(0);

                const submitted: SubmittedBatch = batch.submit();
                expect(submits.count()).toBe(1);
                expect(submitted.id).toBe(1);
                expect(submitted.generation).toBe(0);
                const bytes = await submitted.readback;
                expect(submits.count()).toBe(1);
                expect(bytes.byteLength).toBe(192);
                expect(Array.from(new Uint32Array(bytes, requestA.offset, 16))).toEqual(
                    new Array<number>(16).fill(106),
                );
                expect(Array.from(new Uint32Array(bytes, requestB.offset, 32))).toEqual(
                    new Array<number>(32).fill(107),
                );
                expect(ctx.readback.borrowed).toBe(0);

                // the device holds the same words (the batch copied, it did not move them)
                expect(Array.from(await readU32(ctx, a, 16))).toEqual(new Array<number>(16).fill(106));
                expect(Array.from(await readU32(ctx, b, 32))).toEqual(new Array<number>(32).fill(107));

                // ids are monotonic, generation is carried
                const second = new CommandBatch(ctx, "second", 7);
                expect(second.id).toBe(2);
                expect(second.generation).toBe(7);
                const handle = second.submit();
                expect(handle.id).toBe(2);
                expect(handle.generation).toBe(7);
                // a batch without readback requests resolves an empty buffer
                expect((await handle.readback).byteLength).toBe(0);
                expect(new CommandBatch(ctx, "third").id).toBe(3);
            } finally {
                submits.restore();
                ring.destroy();
                a.destroy();
                b.destroy();
            }
        });
    });

    it("copy() moves bytes between bindings after ending the pass; bad lengths are E_INVALID_ARGUMENT", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const ring = new UniformRing(ctx.device, ctx.allocator, 1, "test/batch/ring");
            const src = scratchBuffer(ctx, 64, "test/batch/src");
            const dst = scratchBuffer(ctx, 64, "test/batch/dst");
            try {
                ring.write(0, FILL_PARAMS, { count: 16, value: 5, mode: 1 });
                ring.flush();
                const batch = new CommandBatch(ctx, "copy");
                kernel.dispatch(
                    batch.pass("fill"),
                    kernel.bind({ dst: bindingOf(src), P: ring.binding(FILL_PARAMS) }),
                    plan1d(16, ctx.workgroupSize, ctx.caps),
                    [0],
                );
                batch.copy(bindingOf(src), bindingOf(dst), 64);
                const request = batch.readback(dst, 0, 64);
                const bytes = await batch.submit().readback;
                expect(Array.from(new Uint32Array(bytes, request.offset, 16))).toEqual([
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                ]);

                const other = new CommandBatch(ctx, "copy-bad");
                expect(
                    hasErrorCode(
                        caught(() => other.copy(bindingOf(src), bindingOf(dst), 6)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.copy(bindingOf(src), bindingOf(dst), 0)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.copy(bindingOf(src), bindingOf(dst), 68)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.readback(dst, 0, 6)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.readback(dst, 0, 0)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.readback(dst, 4, 64)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => other.readback(dst, 2, 8)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(ctx.readback.borrowed).toBe(0);
                await other.submit().readback;
            } finally {
                ring.destroy();
                src.destroy();
                dst.destroy();
            }
        });
    });

    it("submit twice / pass after submit / readback after submit -> E_INVALID_ARGUMENT", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const buffer = scratchBuffer(ctx, 16, "test/batch/once");
            try {
                const batch = new CommandBatch(ctx, "once");
                batch.pass("empty");
                batch.endPass();
                batch.endPass();
                const handle = batch.submit();
                expect(
                    hasErrorCode(
                        caught(() => batch.submit()),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => batch.pass("late")),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => batch.readback(buffer, 0, 16)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect(
                    hasErrorCode(
                        caught(() => batch.copy(bindingOf(buffer), bindingOf(buffer), 16)),
                        "E_INVALID_ARGUMENT",
                    ),
                ).toBe(true);
                expect((await handle.readback).byteLength).toBe(0);
            } finally {
                buffer.destroy();
            }
        });
    });

    it("discard() returns the slot after mapAsync and resolves an empty ArrayBuffer; two batches in flight hold two slots", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const ring = new UniformRing(ctx.device, ctx.allocator, 2, "test/batch/ring");
            const params = ring.binding(FILL_PARAMS);
            const a = scratchBuffer(ctx, 64, "test/batch/a");
            const b = scratchBuffer(ctx, 64, "test/batch/b");
            try {
                ring.write(0, FILL_PARAMS, { count: 16, value: 11, mode: 0 });
                ring.write(1, FILL_PARAMS, { count: 16, value: 22, mode: 0 });
                ring.flush();
                const plan = plan1d(16, ctx.workgroupSize, ctx.caps);

                const first = new CommandBatch(ctx, "first");
                kernel.dispatch(first.pass("fill"), kernel.bind({ dst: bindingOf(a), P: params }), plan, [
                    ring.offsetOf(0),
                ]);
                first.readback(a, 0, 64);
                const second = new CommandBatch(ctx, "second");
                kernel.dispatch(second.pass("fill"), kernel.bind({ dst: bindingOf(b), P: params }), plan, [
                    ring.offsetOf(1),
                ]);
                const requestB = second.readback(b, 0, 64);
                expect(ctx.readback.borrowed).toBe(2);

                const handleA = first.submit();
                const handleB = second.submit();
                handleA.discard();
                expect(ctx.readback.borrowed).toBe(2);
                const [bytesA, bytesB] = await Promise.all([handleA.readback, handleB.readback]);
                expect(bytesA.byteLength).toBe(0);
                expect(Array.from(new Uint32Array(bytesB, requestB.offset, 16))).toEqual(
                    new Array<number>(16).fill(22),
                );
                expect(ctx.readback.borrowed).toBe(0);
                // the discarded batch still ran on the device
                expect(Array.from(await readU32(ctx, a, 16))).toEqual(new Array<number>(16).fill(11));
                // discard after settlement is a no-op
                handleB.discard();
                expect(ctx.readback.borrowed).toBe(0);
            } finally {
                ring.destroy();
                a.destroy();
                b.destroy();
            }
        });
    });

    it("warm(): a batch after PipelineCache.warm dispatches without compiling", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            await ctx.pipelines.warm([kernelSpec("fill")]);
            const { size } = ctx.pipelines;
            expect(size).toBeGreaterThanOrEqual(1);
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            expect(ctx.pipelines.size).toBe(size);
            await ctx.pipelines.warm([kernelSpec("fill")]);
            expect(ctx.pipelines.size).toBe(size);
            const ring = new UniformRing(ctx.device, ctx.allocator, 1, "test/batch/ring");
            const dst = scratchBuffer(ctx, 16, "test/batch/warm");
            try {
                ring.write(0, FILL_PARAMS, { count: 4, value: 9, mode: 0 });
                ring.flush();
                const batch = new CommandBatch(ctx, "warm");
                kernel.dispatch(
                    batch.pass("fill"),
                    kernel.bind({ dst: bindingOf(dst), P: ring.binding(FILL_PARAMS) }),
                    plan1d(4, ctx.workgroupSize, ctx.caps),
                    [0],
                );
                const request = batch.readback(dst, 0, 16);
                const bytes = await batch.submit().readback;
                expect(Array.from(new Uint32Array(bytes, request.offset, 4))).toEqual([9, 9, 9, 9]);
                expect(ctx.pipelines.size).toBe(size);
            } finally {
                ring.destroy();
                dst.destroy();
            }
        });
    });

    it("a deliberately bad bind group rejects THE SAME batch's readback with E_VALIDATION { batchId, label } under Dawn-node", async (t) => {
        requireGpu(t);
        const ctx = await slotContext("batch-slot");
        try {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const dst = scratchBuffer(ctx, 64, "test/batch/bad-dst");
            // a STORAGE-only buffer in the UNIFORM slot: createBindGroup rejects the usage, Dawn-node delivers the
            // uncapturederror synchronously inside bind(), the context stores it in the pending slot
            const wrongUsage = ctx.allocator.createBuffer({
                label: "test/batch/wrong-usage",
                size: 16,
                usage: BufferUsage.STORAGE | BufferUsage.COPY_DST,
            });
            try {
                const batch = new CommandBatch(ctx, "bad");
                const bound = kernel.bind({
                    dst: bindingOf(dst),
                    P: { buffer: wrongUsage, offset: 0, size: 16, window: null },
                });
                kernel.dispatch(batch.pass("fill"), bound, plan1d(16, ctx.workgroupSize, ctx.caps), [0]);
                batch.readback(dst, 0, 64);
                const submitted = batch.submit();
                const outcome = await submitted.readback.then(
                    () => null,
                    (error: unknown) => error,
                );
                expect(isWebGpuGraphError(outcome)).toBe(true);
                if (!isWebGpuGraphError(outcome)) {
                    throw new Error("unreachable");
                }
                expect(outcome.code).toBe("E_VALIDATION");
                expect(outcome.details.batchId).toBe(submitted.id);
                expect(outcome.details.batchLabel).toBe("bad");
                // the error carries the bind group's `<id>/<group>` label (contract 3.9 Kernel.bind), in `label` or in the message
                expect(`${String(outcome.details.label)} ${String(outcome.details.message)}`).toContain("fill/2");
                expect(typeof outcome.details.message).toBe("string");
                console.warn(
                    `[batch] bad bind group -> E_VALIDATION batchId=${submitted.id} label=${String(outcome.details.label)} message=${String(outcome.details.message).split("\n")[0]} (Dawn-node synchronous delivery)`,
                );
                // delivered exactly once: the slot is empty, the ring slot is back, the context is still usable
                expect(ctx.takePendingError()).toBeNull();
                expect(() => {
                    ctx.assertReady();
                }).not.toThrow();
                expect(ctx.readback.borrowed).toBe(0);
                expect(ctx.state).toBe("ready");

                // the same context runs a correct batch afterwards
                const ring = new UniformRing(ctx.device, ctx.allocator, 1, "test/batch/ring");
                ring.write(0, FILL_PARAMS, { count: 16, value: 3, mode: 0 });
                ring.flush();
                const good = new CommandBatch(ctx, "good");
                kernel.dispatch(
                    good.pass("fill"),
                    kernel.bind({ dst: bindingOf(dst), P: ring.binding(FILL_PARAMS) }),
                    plan1d(16, ctx.workgroupSize, ctx.caps),
                    [0],
                );
                const request = good.readback(dst, 0, 64);
                const bytes = await good.submit().readback;
                expect(Array.from(new Uint32Array(bytes, request.offset, 16))).toEqual(new Array<number>(16).fill(3));
                ring.destroy();
            } finally {
                ctx.allocator.destroy(wrongUsage);
                dst.destroy();
            }
        } finally {
            ctx.dispose();
        }
    });
});
