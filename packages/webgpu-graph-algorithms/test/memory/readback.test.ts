/**
 * Readback (spec 4.4; contract 3.8, 5.5): 100 back-to-back reads reuse the three slots without a validation
 * error, chunking above slotBytes, readU32, a caller-supplied dest (its buffer is returned), concurrent reads and
 * borrowSlot growing the ring, returnSlot (unmapping), the allocator path for slots at or above the OOM threshold,
 * argument validation, E_DISPOSED after destroyAll (including a read that is pending when destroyAll runs, whose
 * finally-returned slot must not mask the E_DISPOSED). The setup's afterEach hook fails the test on any uncaptured
 * error, which is how "without validation errors" is asserted.
 */

import { type TestContext } from "vitest";

import { DEFAULT_STAGING_SLOTS, OOM_SCOPE_THRESHOLD_BYTES } from "../../src/constants.js";
import { BufferUsage, MapMode } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError } from "../../src/errors.js";
import { Readback } from "../../src/memory/readback.js";
import { uploadBuffer, withContext } from "../helpers/device.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { requireGpu } from "../setup/gpu.js";

function pattern(words: number, salt: number): Uint32Array<ArrayBuffer> {
    const out = new Uint32Array(words);
    for (let i = 0; i < words; i++) {
        out[i] = i * 7 + salt;
    }
    return out;
}

async function rejectedCode(
    promise: Promise<unknown>,
): Promise<{ code: string; details: Readonly<Record<string, unknown>> }> {
    try {
        await promise;
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected a rejection");
}

function caught(fn: () => unknown): { code: string; details: Readonly<Record<string, unknown>> } {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

/**
 * Counts destroy() calls on one buffer through an own-property override (the Dawn-node object model allows it,
 * CLAUDE.md "Verified Platform Facts"); restore() deletes the override.
 * @param buffer - the buffer to watch
 * @returns the counter and its restore
 */
function spyDestroy(buffer: GPUBuffer): { count: () => number; restore: () => void } {
    let calls = 0;
    const original = buffer.destroy.bind(buffer);
    Object.defineProperty(buffer, "destroy", {
        configurable: true,
        writable: true,
        value: (): void => {
            calls += 1;
            original();
        },
    });
    return {
        count: (): number => calls,
        restore: (): void => {
            Reflect.deleteProperty(buffer, "destroy");
        },
    };
}

describe("Readback", () => {
    it("serves 100 back-to-back read() calls from the default three slots, each into a fresh ArrayBuffer", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const expected = pattern(256, 1);
            const buffer = uploadBuffer(ctx, expected, "pattern");
            expect(ctx.readback.slots).toBe(DEFAULT_STAGING_SLOTS);
            let previous: ArrayBuffer | null = null;
            for (let i = 0; i < 100; i++) {
                const bytes = await ctx.readback.read(buffer, 1024);
                expect(bytes.byteLength).toBe(1024);
                expect(bytes).not.toBe(previous);
                expectBitwiseEqual(new Uint32Array(bytes), expected, `read ${i}`);
                previous = bytes;
            }
            expect(ctx.readback.slots).toBe(DEFAULT_STAGING_SLOTS);
            expect(ctx.readback.borrowed).toBe(0);
            buffer.destroy();
        });
    });

    it("chunks a request above slotBytes and honours srcOffset", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 2, slotBytes: 1024 });
            expect(ring.slots).toBe(2);
            const expected = pattern(2500, 5);
            const buffer = uploadBuffer(ctx, expected, "chunked");
            const bytes = await ring.read(buffer, 10_000);
            expectBitwiseEqual(new Uint32Array(bytes), expected);
            expect(ring.slots).toBe(2);
            const tail = await ring.read(buffer, 1020, undefined, 4);
            expectBitwiseEqual(new Uint32Array(tail), expected.slice(1, 256));
            const middle = await ring.read(buffer, 4096, undefined, 2048);
            expectBitwiseEqual(new Uint32Array(middle), expected.slice(512, 1536));
            ring.destroyAll();
            expect(ring.slots).toBe(0);
            buffer.destroy();
        });
    });

    it("readU32 reads one counter through the ring", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const buffer = uploadBuffer(ctx, new Uint32Array([11, 22, 33]), "counters");
            expect(await ctx.readback.readU32(buffer, 0)).toBe(11);
            expect(await ctx.readback.readU32(buffer, 4)).toBe(22);
            expect(await ctx.readback.readU32(buffer, 8)).toBe(33);
            expect(ctx.readback.borrowed).toBe(0);
            buffer.destroy();
        });
    });

    it("writes into a caller-supplied dest (with a byteOffset) and resolves dest.buffer", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const expected = pattern(256, 3);
            const buffer = uploadBuffer(ctx, expected, "dest");
            const dest = new Uint32Array(256);
            const resolved = await ctx.readback.read(buffer, 1024, dest);
            expect(resolved).toBe(dest.buffer);
            expectBitwiseEqual(dest, expected);
            const big = new Uint32Array(300);
            const view = new Uint32Array(big.buffer, 16, 256);
            expect(await ctx.readback.read(buffer, 1024, view)).toBe(big.buffer);
            expectBitwiseEqual(big.subarray(4, 260), expected);
            expect(big[3]).toBe(0);
            expect(big[260]).toBe(0);
            buffer.destroy();
        });
    });

    it("rejects bad arguments before touching the device", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const buffer = uploadBuffer(ctx, pattern(256, 0), "args");
            expect((await rejectedCode(ctx.readback.read(buffer, 0))).details.argument).toBe("byteLength");
            expect((await rejectedCode(ctx.readback.read(buffer, 6))).details.argument).toBe("byteLength");
            expect((await rejectedCode(ctx.readback.read(buffer, 2048))).details.argument).toBe("byteLength");
            expect((await rejectedCode(ctx.readback.read(buffer, 4, undefined, 2))).details.argument).toBe("srcOffset");
            expect((await rejectedCode(ctx.readback.read(buffer, 1024, new Uint32Array(10)))).details.argument).toBe(
                "dest",
            );
            const shared = new Float32Array(new SharedArrayBuffer(1024));
            expect((await rejectedCode(ctx.readback.read(buffer, 1024, shared))).details.argument).toBe("dest");
            const noCopySrc = ctx.device.createBuffer({ label: "no-copy-src", size: 16, usage: BufferUsage.STORAGE });
            expect((await rejectedCode(ctx.readback.read(noCopySrc, 16))).details.argument).toBe("src");
            for (const call of [
                ctx.readback.read(buffer, 0),
                ctx.readback.read(buffer, 4, undefined, 2),
                ctx.readback.read(noCopySrc, 16),
            ]) {
                expect((await rejectedCode(call)).code).toBe("E_INVALID_ARGUMENT");
            }
            expect(ctx.readback.borrowed).toBe(0);
            noCopySrc.destroy();
            buffer.destroy();
        });
    });

    it("grows the ring when concurrent reads borrow every slot", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const expected = pattern(256, 9);
            const buffer = uploadBuffer(ctx, expected, "concurrent");
            const results = await Promise.all([0, 1, 2, 3, 4].map(() => ctx.readback.read(buffer, 1024)));
            for (const bytes of results) {
                expectBitwiseEqual(new Uint32Array(bytes), expected);
            }
            expect(ctx.readback.slots).toBe(5);
            expect(ctx.readback.borrowed).toBe(0);
            buffer.destroy();
        });
    });

    it("borrowSlot hands out unmapped slots of sufficient capacity, grows when none is free; returnSlot unmaps and frees", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 2, slotBytes: 4096 });
            const s0 = ring.borrowSlot(100);
            const s1 = ring.borrowSlot(100);
            expect([s0.index, s1.index]).toEqual([0, 1]);
            expect(s0.capacity).toBe(4096);
            expect(s0.buffer.usage).toBe(BufferUsage.MAP_READ | BufferUsage.COPY_DST);
            expect(ring.borrowed).toBe(2);
            expect(ring.slots).toBe(2);
            const s2 = ring.borrowSlot(100);
            expect(s2.index).toBe(2);
            expect(ring.slots).toBe(3);
            // 10,000 rounded up to 256 = 10,240
            const s3 = ring.borrowSlot(10_000);
            expect(s3.capacity).toBe(10_240);
            expect(ring.slots).toBe(4);
            expect(ring.borrowed).toBe(4);
            // use a borrowed slot the way CommandBatch will: copy, submit, map, read, return (which unmaps)
            const expected = pattern(16, 2);
            const src = uploadBuffer(ctx, expected, "slot-src");
            const encoder = ctx.device.createCommandEncoder();
            encoder.copyBufferToBuffer(src, 0, s0.buffer, 0, 64);
            ctx.device.queue.submit([encoder.finish()]);
            await s0.buffer.mapAsync(MapMode.READ, 0, 64);
            expectBitwiseEqual(new Uint32Array(s0.buffer.getMappedRange(0, 64).slice(0)), expected);
            expect(s0.buffer.mapState).toBe("mapped");
            ring.returnSlot(s0);
            expect(s0.buffer.mapState).toBe("unmapped");
            ring.returnSlot(s1);
            ring.returnSlot(s2);
            ring.returnSlot(s3);
            expect(ring.borrowed).toBe(0);
            expect(caught(() => ring.returnSlot(s0)).code).toBe("E_INVALID_ARGUMENT");
            const foreign = { index: 0, buffer: s0.buffer, capacity: 4096 };
            expect(caught(() => ring.returnSlot(foreign)).code).toBe("E_INVALID_ARGUMENT");
            const reused = ring.borrowSlot(5000);
            expect(reused).toBe(s3);
            expect(ring.slots).toBe(4);
            ring.returnSlot(reused);
            expect(caught(() => ring.borrowSlot(0)).code).toBe("E_INVALID_ARGUMENT");
            ring.destroyAll();
            src.destroy();
        });
    });

    it("creates a slot at or above the OOM threshold through the allocator and destroys it with destroyAll", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 1, slotBytes: 4096 });
            const before = ctx.allocator.liveBuffers;
            const big = ring.borrowSlot(OOM_SCOPE_THRESHOLD_BYTES);
            expect(big.capacity).toBe(OOM_SCOPE_THRESHOLD_BYTES);
            expect(ctx.allocator.liveBuffers).toBe(before + 1);
            ring.returnSlot(big);
            await ctx.allocator.check();
            ring.destroyAll();
            expect(ctx.allocator.liveBuffers).toBe(before);
        });
    });

    it("is inert after destroyAll", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator);
            const buffer = uploadBuffer(ctx, pattern(4, 0), "late");
            ring.destroyAll();
            expect(ring.slots).toBe(0);
            expect(ring.borrowed).toBe(0);
            const late = await rejectedCode(ring.read(buffer, 16));
            expect(late.code).toBe("E_DISPOSED");
            expect(late.details.label).toBe("readback");
            expect(caught(() => ring.borrowSlot(16)).code).toBe("E_DISPOSED");
            expect(() => {
                ring.destroyAll();
            }).not.toThrow();
            buffer.destroy();
        });
    });

    it("a read pending when destroyAll() runs rejects with E_DISPOSED, not with a returnSlot error", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 1, slotBytes: 1024 });
            const buffer = uploadBuffer(ctx, pattern(256, 4), "pending");
            // read() runs synchronously up to `await this.map(...)`: the copy is submitted and mapAsync is pending
            const pending = ring.read(buffer, 1024);
            expect(ring.borrowed).toBe(1);
            // destroying the slot rejects the pending mapAsync; map() turns that into E_DISPOSED and read()'s
            // finally returns the slot to a ring that no longer exists, which must not throw
            ring.destroyAll();
            const late = await rejectedCode(pending);
            expect(late.code).toBe("E_DISPOSED");
            expect(late.details.label).toBe("readback");
            expect(ring.slots).toBe(0);
            expect(ring.borrowed).toBe(0);
            expect(() => {
                ring.returnSlot({ index: 0, buffer, capacity: 1024 });
            }).not.toThrow();
            buffer.destroy();
            await ctx.allocator.check();
        });
    });

    it("mapSlot: a slot returned while its map is pending stays borrowed until the map settles, then is unmapped and free", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 1, slotBytes: 1024 });
            const src = uploadBuffer(ctx, pattern(256, 5), "map-slot");
            const slot = ring.borrowSlot(1024);
            const encoder = ctx.device.createCommandEncoder({ label: "map-slot" });
            encoder.copyBufferToBuffer(src, 0, slot.buffer, 0, 1024);
            ctx.device.queue.submit([encoder.finish()]);
            const mapped = ring.mapSlot(slot, 1024);
            expect(caught(() => ring.mapSlot(slot, 1024)).details.argument).toBe("slot"); // one map at a time
            // the return arrives while the map is pending (it cannot settle before this macrotask ends): deferred
            ring.returnSlot(slot);
            expect(ring.borrowed).toBe(1);
            expect(ring.borrowSlot(1024)).not.toBe(slot); // the ring grows rather than hand out a slot with a map pending
            expect(ring.slots).toBe(2);
            await mapped;
            // the settle unmapped and freed the slot; the second slot is still borrowed
            expect(ring.borrowed).toBe(1);
            expect(slot.buffer.mapState ?? "unmapped").toBe("unmapped");
            const again = ring.borrowSlot(1024);
            expect(again).toBe(slot);
            await ring.mapSlot(again, 1024); // a fresh map on the returned slot works
            expect(new Uint32Array(again.buffer.getMappedRange(0, 1024))[3]).toBe(pattern(256, 5)[3]);
            ring.returnSlot(again);
            expect(slot.buffer.mapState ?? "unmapped").toBe("unmapped");
            ring.destroyAll();
            src.destroy();
            await ctx.allocator.check();
        });
    });

    it("destroyAll() during a mapSlot map defers the destruction to the settle; the borrower's return is a no-op", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const ring = new Readback(ctx.device, ctx.allocator, { slots: 1, slotBytes: 1024 });
            const src = uploadBuffer(ctx, pattern(256, 6), "map-slot-dispose");
            const slot = ring.borrowSlot(1024);
            const encoder = ctx.device.createCommandEncoder({ label: "map-slot-dispose" });
            encoder.copyBufferToBuffer(src, 0, slot.buffer, 0, 1024);
            ctx.device.queue.submit([encoder.finish()]);
            const mapped = ring.mapSlot(slot, 1024);
            const destroyed = spyDestroy(slot.buffer);
            ring.destroyAll();
            expect(destroyed.count()).toBe(0); // not while the map is pending
            expect(ring.slots).toBe(0);
            await mapped.then(
                () => undefined,
                () => undefined,
            );
            expect(destroyed.count()).toBe(1); // the settle destroyed it (whichever way the runtime settled the map)
            expect(() => {
                ring.returnSlot(slot);
            }).not.toThrow();
            destroyed.restore();
            src.destroy();
            await ctx.allocator.check();
        });
    });

    it("rejects a bad constructor option", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            expect(caught(() => new Readback(ctx.device, ctx.allocator, { slotBytes: 6 })).details.argument).toBe(
                "slotBytes",
            );
            expect(caught(() => new Readback(ctx.device, ctx.allocator, { slots: -1 })).details.argument).toBe("slots");
            await ctx.allocator.check();
        });
    });
});
