/**
 * Lease (contract 3.8, spec 4.4): a scope object over ctx.pool -- storage / uniform / acquire go back to the pool on
 * release (also when the algorithm throws, through the try / finally pattern spec 4.4 prescribes), `count` tracks the
 * live buffers, and a released lease refuses further acquisitions with E_DISPOSED. The byte accounting below is
 * hand-computed from the size classes of spec 4.4: 256 B and 4096 B both round to the 4 KiB minimum class, 70,000 B
 * rounds to 128 KiB (131,072 B).
 */

import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode } from "../../src/errors.js";
import { BufferPool } from "../../src/memory/buffer-pool.js";
import { Lease } from "../../src/memory/lease.js";
import { withContext } from "../helpers/device.js";
import { acquire, requireGpu } from "../setup/gpu.js";

function caught(fn: () => unknown): unknown {
    try {
        fn();
    } catch (error) {
        return error;
    }
    return null;
}

describe("Lease (spec 4.4)", () => {
    it("the size classes the accounting below relies on", () => {
        expect(BufferPool.sizeClass(256)).toBe(4096);
        expect(BufferPool.sizeClass(4096)).toBe(4096);
        expect(BufferPool.sizeClass(70_000)).toBe(131_072);
        expect(BufferPool.sizeClass(1024) + BufferPool.sizeClass(2048)).toBe(8192);
    });

    it("storage / uniform / acquire go back to the pool on release; count tracks the live buffers", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const { pool } = ctx;
            expect(pool.liveBytes).toBe(0);
            const lease = pool.lease();
            expect(lease).toBeInstanceOf(Lease);
            expect(lease.count).toBe(0);

            const a = lease.storage(4096, "lease/a");
            const b = lease.uniform(256, "lease/b");
            const c = lease.acquire(70_000, BufferUsage.STORAGE | BufferUsage.COPY_SRC, "lease/c");
            expect(lease.count).toBe(3);
            // the pool hands out whole size classes: 4096 -> 4 KiB, 256 -> the 4 KiB minimum, 70,000 -> 128 KiB
            expect(a.size).toBe(4096);
            expect(b.size).toBe(4096);
            expect(c.size).toBe(131_072);
            expect(pool.liveBytes).toBe(4096 + 4096 + 131_072);
            expect(pool.liveBytes).toBe(139_264);
            expect(pool.idleBytes).toBe(0);

            lease.release();
            expect(lease.count).toBe(0);
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(139_264);

            // the idle buffers are handed out again by (class, usage): the next lease gets the same objects back
            const again = pool.lease();
            expect(again.storage(4000, "lease/a2")).toBe(a);
            expect(again.uniform(16, "lease/b2")).toBe(b);
            expect(again.acquire(100_000, BufferUsage.STORAGE | BufferUsage.COPY_SRC, "lease/c2")).toBe(c);
            expect(again.count).toBe(3);
            expect(pool.idleBytes).toBe(0);
            expect(pool.liveBytes).toBe(139_264);
            again.release();
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(139_264);

            // idempotent: a second release changes nothing
            lease.release();
            again.release();
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(139_264);
            await Promise.resolve();
        });
    });

    it("releases in the try / finally pattern when the algorithm throws", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const { pool } = ctx;
            const lease = pool.lease();
            const algorithm = async (): Promise<void> => {
                try {
                    lease.storage(1024, "lease/sigma");
                    lease.storage(2048, "lease/delta");
                    expect(lease.count).toBe(2);
                    expect(pool.liveBytes).toBe(8192);
                    await Promise.resolve();
                    throw new Error("boom");
                } finally {
                    lease.release();
                }
            };
            await expect(algorithm()).rejects.toThrow("boom");
            expect(lease.count).toBe(0);
            expect(pool.liveBytes).toBe(0);
            expect(pool.idleBytes).toBe(8192);
        });
    });

    it("E_DISPOSED after release; the pool's own errors pass through unchanged", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const lease = ctx.pool.lease();
            lease.release();
            const late = caught(() => lease.storage(16, "lease/late"));
            expect(hasErrorCode(late, "E_DISPOSED")).toBe(true);
            expect((late as { details: { label: string } }).details.label).toBe("lease/late");
            expect(
                hasErrorCode(
                    caught(() => lease.uniform(16, "lease/late-u")),
                    "E_DISPOSED",
                ),
            ).toBe(true);
            expect(
                hasErrorCode(
                    caught(() => lease.acquire(16, BufferUsage.STORAGE, "lease/late-a")),
                    "E_DISPOSED",
                ),
            ).toBe(true);

            const fresh = ctx.pool.lease();
            const tooBig = caught(() => fresh.storage(ctx.caps.limits.maxBufferSize + 4, "lease/huge"));
            expect(hasErrorCode(tooBig, "E_TOO_LARGE")).toBe(true);
            expect(fresh.count).toBe(0);
            fresh.release();
            await Promise.resolve();
        });
    });

    it("release() after ctx.dispose() is silent: the pool already destroyed everything", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "lease-dispose" });
        const lease = ctx.pool.lease();
        lease.storage(4096, "lease/orphan");
        expect(lease.count).toBe(1);
        ctx.dispose();
        expect(() => {
            lease.release();
        }).not.toThrow();
        expect(lease.count).toBe(0);
    });
});
