/**
 * BufferPool (spec 4.4; contract 3.8, 5.5): the size-class table, acquire / release reuse per (class, usage), the
 * maxIdlePerClass eviction, trim, liveBytes / idleBytes accounting, the class cap at maxBufferSize (PLAN DECISION
 * 14), E_TOO_LARGE above it, E_INVALID_ARGUMENT for a foreign or double release, E_DISPOSED after destroyAll.
 * The pool allocates through the context's AllocationTracker, so every test but the table needs a device.
 */

import { type TestContext } from "vitest";

import { POOL_MAX_IDLE_PER_CLASS } from "../../src/constants.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError } from "../../src/errors.js";
import { BufferPool } from "../../src/memory/buffer-pool.js";
import { withContext } from "../helpers/device.js";
import { requireGpu } from "../setup/gpu.js";

const MIB = 1024 * 1024;
const STORAGE = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;
const UNIFORM = BufferUsage.UNIFORM | BufferUsage.COPY_DST;

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

describe("BufferPool.sizeClass", () => {
    it("rounds up to powers of two from 4 KiB to 64 MiB, then to 16 MiB steps", () => {
        const table: [number, number][] = [
            [1, 4096],
            [4095, 4096],
            [4096, 4096],
            [4097, 8192],
            [8192, 8192],
            [1_000_000, 1_048_576],
            [33_554_433, 67_108_864],
            [64 * MIB, 64 * MIB],
            // 67,108,865 / 16,777,216 = 4.00000006 -> 5 steps = 80 MiB
            [64 * MIB + 1, 83_886_080],
            // 100,000,000 / 16,777,216 = 5.96 -> 6 steps
            [100_000_000, 100_663_296],
            [256 * MIB, 256 * MIB],
            [1024 * MIB, 1024 * MIB],
            [1024 * MIB + 1, 1024 * MIB + 16 * MIB],
        ];
        for (const [bytes, expected] of table) {
            expect(BufferPool.sizeClass(bytes), `sizeClass(${bytes})`).toBe(expected);
        }
    });

    it("rejects a non-positive or fractional byte length", () => {
        expect(caught(() => BufferPool.sizeClass(0)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => BufferPool.sizeClass(-4)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => BufferPool.sizeClass(1.5)).code).toBe("E_INVALID_ARGUMENT");
    });
});

describe("BufferPool on a device", () => {
    it("acquire rounds up to the class, relabels a reused buffer and keeps usage classes apart", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const { pool } = ctx;
            const a = pool.acquire(1000, STORAGE, "a");
            expect(a.size).toBe(4096);
            expect(a.label).toBe("a");
            expect(a.usage).toBe(STORAGE);
            expect(pool.liveBytes).toBe(4096);
            expect(pool.idleBytes).toBe(0);
            pool.release(a);
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(4096);
            const b = pool.acquire(4000, STORAGE, "b");
            expect(b).toBe(a);
            expect(b.label).toBe("b");
            expect(pool.idleBytes).toBe(0);
            expect(pool.liveBytes).toBe(4096);
            const c = pool.acquire(4000, UNIFORM, "c");
            expect(c).not.toBe(a);
            expect(c.usage).toBe(UNIFORM);
            expect(pool.liveBytes).toBe(8192);
            pool.release(b);
            pool.release(c);
            expect(pool.idleBytes).toBe(8192);
            const d = pool.acquire(4096, STORAGE, "d");
            expect(d).toBe(a);
            pool.release(d);
            await ctx.allocator.check();
        });
    });

    it("keeps at most maxIdlePerClass idle buffers per (class, usage) and destroys the rest; trim destroys all idle", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const pool = new BufferPool(ctx.device, ctx.allocator, ctx.caps.limits.maxBufferSize, {
                maxIdlePerClass: 2,
            });
            const before = ctx.allocator.liveBuffers;
            const buffers = [
                pool.acquire(1000, STORAGE, "x0"),
                pool.acquire(1000, STORAGE, "x1"),
                pool.acquire(1000, STORAGE, "x2"),
            ];
            expect(new Set(buffers).size).toBe(3);
            expect(ctx.allocator.liveBuffers).toBe(before + 3);
            expect(pool.liveBytes).toBe(3 * 4096);
            for (const buffer of buffers) {
                pool.release(buffer);
            }
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(2 * 4096);
            expect(ctx.allocator.liveBuffers).toBe(before + 2);
            const again = pool.acquire(2000, STORAGE, "y");
            expect(buffers).toContain(again);
            expect(pool.idleBytes).toBe(4096);
            pool.release(again);
            pool.trim();
            expect(pool.idleBytes).toBe(0);
            expect(ctx.allocator.liveBuffers).toBe(before);
            pool.destroyAll();
            // the context's own pool uses the POOL_MAX_IDLE_PER_CLASS default
            const five = [0, 1, 2, 3, 4].map((i) => ctx.pool.acquire(1000, STORAGE, `z${i}`));
            for (const buffer of five) {
                ctx.pool.release(buffer);
            }
            expect(ctx.pool.idleBytes).toBe(POOL_MAX_IDLE_PER_CLASS * 4096);
            ctx.pool.trim();
            await ctx.allocator.check();
        });
    });

    it("throws E_TOO_LARGE above maxBufferSize and caps the class at maxBufferSize otherwise", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const small = new BufferPool(ctx.device, ctx.allocator, 1 * MIB);
            const err = caught(() => small.acquire(2 * MIB, STORAGE, "big"));
            expect(err.code).toBe("E_TOO_LARGE");
            expect(err.details).toEqual({ needed: 2 * MIB, limit: 1 * MIB, path: "pool", algorithm: null });
            const exact = small.acquire(1 * MIB, STORAGE, "exact");
            expect(exact.size).toBe(1 * MIB);
            small.release(exact);
            small.destroyAll();
            // a request that fits the device but whose class overshoots gets a buffer of the capped class
            const capped = new BufferPool(ctx.device, ctx.allocator, 6000);
            const c = capped.acquire(4097, STORAGE, "capped");
            expect(c.size).toBe(6000);
            capped.release(c);
            expect(capped.acquire(6000, STORAGE, "fits")).toBe(c);
            capped.release(c);
            expect(caught(() => capped.acquire(6001, STORAGE, "over")).code).toBe("E_TOO_LARGE");
            // an odd maxBufferSize is rounded down to a multiple of 4 (a storage binding size must be one)
            const odd = new BufferPool(ctx.device, ctx.allocator, 4_294_967_295);
            expect(odd.acquire(1000, STORAGE, "odd").size).toBe(4096);
            odd.destroyAll();
            capped.destroyAll();
            await ctx.allocator.check();
        });
    });

    it("rejects a release of a foreign buffer or a second release with E_INVALID_ARGUMENT", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const raw = ctx.device.createBuffer({ label: "raw", size: 16, usage: STORAGE });
            const foreign = caught(() => ctx.pool.release(raw));
            expect(foreign.code).toBe("E_INVALID_ARGUMENT");
            expect(foreign.details.argument).toBe("buffer");
            raw.destroy();
            const a = ctx.pool.acquire(16, STORAGE, "a");
            ctx.pool.release(a);
            expect(caught(() => ctx.pool.release(a)).code).toBe("E_INVALID_ARGUMENT");
            expect(caught(() => ctx.pool.acquire(0, STORAGE, "zero")).code).toBe("E_INVALID_ARGUMENT");
            await ctx.allocator.check();
        });
    });

    it("accounts liveBytes / idleBytes over a mixed sequence and is inert after destroyAll", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const pool = new BufferPool(ctx.device, ctx.allocator, ctx.caps.limits.maxBufferSize);
            const a = pool.acquire(1000, STORAGE, "a");
            const b = pool.acquire(5000, STORAGE, "b");
            const c = pool.acquire(3 * MIB, STORAGE, "c");
            expect(pool.liveBytes).toBe(4096 + 8192 + 4 * MIB);
            pool.release(b);
            expect(pool.liveBytes).toBe(4096 + 4 * MIB);
            expect(pool.idleBytes).toBe(8192);
            pool.trim();
            expect(pool.idleBytes).toBe(0);
            pool.release(a);
            pool.release(c);
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(4096 + 4 * MIB);
            const d = pool.acquire(4096, STORAGE, "d");
            pool.destroyAll();
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(0);
            const disposed = caught(() => pool.acquire(16, STORAGE, "late"));
            expect(disposed.code).toBe("E_DISPOSED");
            expect(disposed.details.label).toBe("pool");
            expect(() => {
                pool.release(d);
            }).not.toThrow();
            expect(() => {
                pool.trim();
                pool.destroyAll();
            }).not.toThrow();
            await ctx.allocator.check();
        });
    });
});
