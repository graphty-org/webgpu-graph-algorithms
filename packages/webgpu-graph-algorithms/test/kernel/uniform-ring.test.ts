/**
 * UniformRing (contract 3.9, spec 5.3): one UNIFORM buffer of UNIFORM_SLOT_BYTES-stride slots. offsetOf(k) is 256 k,
 * reserve() hands out contiguous slots and wraps to 0 when the tail is too short, write() fills the host shadow and
 * flush() sends the dirty slots with ONE queue.writeBuffer, and a kernel bound once to the whole-buffer binding reads
 * slot k through the dynamic offset offsetOf(k). The fill kernel of contract 3.10 is the reader: FillParams
 * { count, value, mode } with mode 0 = constant `value`, mode 1 = iota `i + value`.
 */

import { UNIFORM_SLOT_BYTES } from "../../src/constants.js";
import { hasErrorCode } from "../../src/errors.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { UniformBlock } from "../../src/kernel/struct-block.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { bindingOf, scratchBuffer, withContext } from "../helpers/device.js";
import { requireGpu } from "../setup/gpu.js";

function caught(fn: () => unknown): unknown {
    try {
        fn();
    } catch (error) {
        return error;
    }
    return null;
}

interface WriteCall {
    readonly offset: number;
    readonly size: number;
}

/** Records every queue.writeBuffer aimed at `target` (offset, size) until `restore()`; the ring's flush rule is asserted through it. */
function recordWrites(queue: GPUQueue, target: GPUBuffer): { readonly calls: WriteCall[]; restore(): void } {
    const calls: WriteCall[] = [];
    const original = queue.writeBuffer.bind(queue);
    queue.writeBuffer = (
        buffer: GPUBuffer,
        bufferOffset: number,
        data: GPUAllowSharedBufferSource,
        dataOffset?: number,
        size?: number,
    ): undefined => {
        if (buffer === target) {
            calls.push({ offset: bufferOffset, size: size ?? data.byteLength });
        }
        original(buffer, bufferOffset, data, dataOffset, size);
        return undefined;
    };
    return {
        calls,
        restore(): void {
            queue.writeBuffer = original;
        },
    };
}

describe("UniformRing (spec 5.3)", () => {
    it("offsetOf(k) === 256 k; reserve wraps; count > slots -> E_INVALID_ARGUMENT", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new UniformRing(ctx.device, ctx.allocator, 8, "test/ring");
            expect(ring.slots).toBe(8);
            expect(UNIFORM_SLOT_BYTES).toBe(256);
            for (let k = 0; k < 8; k++) {
                expect(ring.offsetOf(k)).toBe(256 * k);
            }
            expect(ring.offsetOf(7)).toBe(1792);
            expect(
                hasErrorCode(
                    caught(() => ring.offsetOf(8)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.offsetOf(-1)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.offsetOf(1.5)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);

            // 8 slots: 3 + 3 fit, the third 3 would end at 9 -> wraps to 0
            expect(ring.reserve(3)).toBe(0);
            expect(ring.reserve(3)).toBe(3);
            expect(ring.reserve(3)).toBe(0);
            expect(ring.reserve(2)).toBe(3);
            // 5 + 8 > 8 -> wraps; a whole-ring reservation always starts at 0
            expect(ring.reserve(8)).toBe(0);
            expect(ring.reserve(1)).toBe(0);
            expect(
                hasErrorCode(
                    caught(() => ring.reserve(9)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.reserve(0)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.reserve(2.5)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            await Promise.resolve();
            ring.destroy();
        });
    });

    it("binding() is the whole-buffer binding sized by the block; a block wider than a slot is E_INVALID_ARGUMENT", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const before = ctx.allocator.liveBuffers;
            const ring = new UniformRing(ctx.device, ctx.allocator, 8, "test/ring");
            expect(ctx.allocator.liveBuffers).toBe(before + 1);
            const binding = ring.binding(FILL_PARAMS);
            expect(binding.offset).toBe(0);
            expect(binding.size).toBe(16);
            expect(binding.window).toBeNull();
            expect(binding.buffer.size).toBe(8 * 256);
            expect(binding.buffer.size).toBe(2048);
            expect(ring.binding(FILL_PARAMS).buffer).toBe(binding.buffer);

            const wide = UniformBlock.define("Wide", [
                ["a0", "vec4f"],
                ["a1", "vec4f"],
                ["a2", "vec4f"],
                ["a3", "vec4f"],
                ["a4", "vec4f"],
                ["a5", "vec4f"],
                ["a6", "vec4f"],
                ["a7", "vec4f"],
                ["a8", "vec4f"],
                ["a9", "vec4f"],
                ["a10", "vec4f"],
                ["a11", "vec4f"],
                ["a12", "vec4f"],
                ["a13", "vec4f"],
                ["a14", "vec4f"],
                ["a15", "vec4f"],
                ["a16", "vec4f"],
            ]);
            expect(wide.byteLength).toBe(272);
            expect(
                hasErrorCode(
                    caught(() => ring.binding(wide)),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.write(0, wide, {})),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.write(8, FILL_PARAMS, {})),
                    "E_INVALID_ARGUMENT",
                ),
            ).toBe(true);

            ring.destroy();
            expect(ctx.allocator.liveBuffers).toBe(before);
            expect(
                hasErrorCode(
                    caught(() => ring.binding(FILL_PARAMS)),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.reserve(1)),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.write(0, FILL_PARAMS, {})),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.flush()),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => ring.offsetOf(0)),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            ring.destroy();
            expect(ctx.allocator.liveBuffers).toBe(before);
            await Promise.resolve();
        });
    });

    it("write + flush + a kernel reading slot k sees the right values through the dynamic offset; flush sends one writeBuffer of the dirty range", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const ring = new UniformRing(ctx.device, ctx.allocator, 8, "test/ring");
            const params = ring.binding(FILL_PARAMS);
            const writes = recordWrites(ctx.device.queue, params.buffer);
            try {
                const slots = [0, 3, 7];
                const targets = new Map<number, GPUBuffer>();
                for (const slot of slots) {
                    ring.write(slot, FILL_PARAMS, { count: 8, value: 1000 + slot, mode: 0 });
                    targets.set(slot, scratchBuffer(ctx, 32, `test/ring/dst${slot}`));
                }
                expect(writes.calls).toEqual([]);
                ring.flush();
                // slots 0 and 7 are dirty: one writeBuffer covering slots 0..7 = the whole 2048-byte buffer
                expect(writes.calls).toEqual([{ offset: 0, size: 2048 }]);
                ring.flush();
                expect(writes.calls.length).toBe(1);

                const plan = plan1d(8, ctx.workgroupSize, ctx.caps);
                const batch = new CommandBatch(ctx, "ring");
                const pass = batch.pass("fill");
                const requests = new Map<number, number>();
                for (const [slot, dst] of targets) {
                    const bound = kernel.bind({ dst: bindingOf(dst), P: params });
                    kernel.dispatch(pass, bound, plan, [ring.offsetOf(slot)]);
                }
                for (const [slot, dst] of targets) {
                    requests.set(slot, batch.readback(dst, 0, 32).offset);
                }
                const bytes = await batch.submit().readback;
                expect(bytes.byteLength).toBe(96);
                for (const slot of slots) {
                    const words = Array.from(new Uint32Array(bytes, requests.get(slot), 8));
                    expect(words).toEqual([
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                        1000 + slot,
                    ]);
                }
                expect(Array.from(new Uint32Array(bytes, requests.get(3), 8))).toEqual([
                    1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003,
                ]);

                // a second write to slot 3 only: iota from 5; the flush sends exactly that slot
                ring.write(3, FILL_PARAMS, { count: 8, value: 5, mode: 1 });
                ring.flush();
                expect(writes.calls).toEqual([
                    { offset: 0, size: 2048 },
                    { offset: 768, size: 256 },
                ]);
                const again = new CommandBatch(ctx, "ring-iota");
                const pass2 = again.pass("fill");
                const dst3 = targets.get(3);
                if (dst3 === undefined) {
                    throw new Error("slot 3 target missing");
                }
                kernel.dispatch(pass2, kernel.bind({ dst: bindingOf(dst3), P: params }), plan, [ring.offsetOf(3)]);
                const request3 = again.readback(dst3, 0, 32);
                const bytes2 = await again.submit().readback;
                expect(Array.from(new Uint32Array(bytes2, request3.offset, 8))).toEqual([5, 6, 7, 8, 9, 10, 11, 12]);

                // twice: the same batch shape gives bitwise the same words (spec 11.9 item 4)
                const third = new CommandBatch(ctx, "ring-iota-2");
                kernel.dispatch(third.pass("fill"), kernel.bind({ dst: bindingOf(dst3), P: params }), plan, [
                    ring.offsetOf(3),
                ]);
                const request3b = third.readback(dst3, 0, 32);
                const bytes3 = await third.submit().readback;
                expect(Array.from(new Uint32Array(bytes3, request3b.offset, 8))).toEqual(
                    Array.from(new Uint32Array(bytes2, request3.offset, 8)),
                );

                for (const dst of targets.values()) {
                    dst.destroy();
                }
            } finally {
                writes.restore();
                ring.destroy();
            }
        });
    });
});
