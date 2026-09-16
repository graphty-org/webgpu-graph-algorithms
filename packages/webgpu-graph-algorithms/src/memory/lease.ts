/**
 * The Lease scope object of spec 4.4: every buffer an algorithm acquires through it goes back to the
 * BufferPool in ONE `release()` call placed in the algorithm's `finally` block, so scratch never
 * outlives the call that took it (a `try / finally`, not `Symbol.dispose`, until the monorepo's
 * TypeScript target supports `using`). Persistent buffers (layout state) are owned by the simulation
 * object instead and freed by its `dispose()`.
 */

import { BufferUsage } from "../device/webgpu-constants.js";
import { hasErrorCode, WebGpuGraphError } from "../errors.js";
import type { BufferPool } from "./buffer-pool.js";

/** Scope object algorithms use: every buffer acquired through it is released by `release()` in a finally block (spec 4.4). */
export class Lease {
    private readonly pool: BufferPool;
    private readonly buffers = new Set<GPUBuffer>();
    private released = false;

    /**
     * Creates a lease over a pool; `pool.lease()` is the usual way to get one.
     * @param pool - the pool every acquisition of this lease goes through
     */
    constructor(pool: BufferPool) {
        this.pool = pool;
    }

    /**
     * pool.acquire(byteLength, STORAGE | COPY_SRC | COPY_DST, label).
     * @param byteLength - bytes needed; the pool rounds up to its size class
     * @param label - the buffer label
     * @returns the buffer, returned to the pool by `release()`
     */
    storage(byteLength: number, label: string): GPUBuffer {
        return this.acquire(byteLength, BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST, label);
    }

    /**
     * pool.acquire(byteLength, UNIFORM | COPY_DST, label).
     * @param byteLength - bytes needed; the pool rounds up to its size class
     * @param label - the buffer label
     * @returns the buffer, returned to the pool by `release()`
     */
    uniform(byteLength: number, label: string): GPUBuffer {
        return this.acquire(byteLength, BufferUsage.UNIFORM | BufferUsage.COPY_DST, label);
    }

    /**
     * pool.acquire with explicit usage.
     * @param byteLength - bytes needed; the pool rounds up to its size class
     * @param usage - the GPUBufferUsage bits
     * @param label - the buffer label
     * @returns the buffer, returned to the pool by `release()`
     */
    acquire(byteLength: number, usage: number, label: string): GPUBuffer {
        if (this.released) {
            throw new WebGpuGraphError("E_DISPOSED", `lease already released: cannot acquire "${label}"`, { label });
        }
        const buffer = this.pool.acquire(byteLength, usage, label);
        this.buffers.add(buffer);
        return buffer;
    }

    /**
     * Releases every buffer acquired through this lease; idempotent. A pool that `ctx.dispose()` already
     * destroyed has nothing left to take back, so its E_DISPOSED is swallowed here: the lease sits in a
     * `finally` block and must never mask the error that unwound the algorithm.
     */
    release(): void {
        if (this.released) {
            return;
        }
        this.released = true;
        const owned = [...this.buffers];
        this.buffers.clear();
        for (const buffer of owned) {
            try {
                this.pool.release(buffer);
            } catch (error) {
                if (!hasErrorCode(error, "E_DISPOSED")) {
                    throw error;
                }
            }
        }
    }

    /**
     * Live buffers of this lease.
     * @returns the number of buffers acquired through the lease and not yet released
     */
    get count(): number {
        return this.buffers.size;
    }
}
