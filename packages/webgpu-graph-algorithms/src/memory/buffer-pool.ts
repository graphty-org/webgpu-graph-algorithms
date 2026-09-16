/**
 * Size-class pool of GPUBuffers by usage (spec 4.4): a request rounds up to a class (powers of two from 4 KiB to
 * 64 MiB, then 16 MiB steps), an idle buffer of that (class, usage) is reused, otherwise one is created through the
 * context's AllocationTracker (so the OOM scope of spec 5.7 and the resident accounting are shared with the
 * residency); `release` keeps at most maxIdlePerClass idle buffers per (class, usage); `trim` destroys the idle
 * ones (ctx.release and layouts on dispose); `destroyAll` everything (ctx.dispose). The Lease scope object arrives
 * with P2-T1 (`lease()`, PLAN DECISION 1). Classes above maxBufferSize are never created: a request that fits the
 * device but whose class overshoots gets the capped class (PLAN DECISION 14).
 */

import {
    POOL_LINEAR_STEP_BYTES,
    POOL_MAX_IDLE_PER_CLASS,
    POOL_MAX_POW2_CLASS_BYTES,
    POOL_MIN_CLASS_BYTES,
} from "../constants.js";
import { type AllocationTracker } from "../device/error-scope.js";
import { WebGpuGraphError } from "../errors.js";
import { Lease } from "./lease.js";

/** What the pool remembers about a live buffer. */
interface LiveEntry {
    readonly sizeClass: number;
    readonly usage: number;
}

/**
 * The idle-list key of a (class, usage) pair.
 * @param sizeClass - the class in bytes
 * @param usage - the GPUBufferUsage flags
 * @returns the map key
 */
function classKey(sizeClass: number, usage: number): string {
    return `${sizeClass}:${usage}`;
}

/** Size-class pool of GPUBuffers by usage (spec 4.4). */
export class BufferPool {
    private readonly allocator: AllocationTracker;
    private readonly maxBufferSize: number;
    private readonly maxIdlePerClass: number;
    private readonly idle = new Map<string, GPUBuffer[]>();
    private readonly live = new Map<GPUBuffer, LiveEntry>();
    private liveBytesValue = 0;
    private idleBytesValue = 0;
    private disposed = false;

    /**
     * Creates an empty pool.
     * @param _device - the device (buffers are created through the allocator, which owns it; the parameter keeps
     *   the contract's constructor shape)
     * @param allocator - the context's OOM-scoped allocator
     * @param maxBufferSize - the device's maxBufferSize (classes above it are never created)
     * @param options - pool options
     * @param options.maxIdlePerClass - idle buffers kept per (class, usage); default POOL_MAX_IDLE_PER_CLASS
     */
    constructor(
        _device: GPUDevice,
        allocator: AllocationTracker,
        maxBufferSize: number,
        options?: { readonly maxIdlePerClass?: number | undefined },
    ) {
        this.allocator = allocator;
        this.maxBufferSize = maxBufferSize - (maxBufferSize % 4);
        this.maxIdlePerClass = options?.maxIdlePerClass ?? POOL_MAX_IDLE_PER_CLASS;
    }

    /**
     * The size class a byte length rounds up to: powers of two from 4 KiB to 64 MiB, then 16 MiB steps (pure).
     * @param byteLength - a positive integer
     * @returns the class in bytes
     */
    static sizeClass(byteLength: number): number {
        if (!Number.isInteger(byteLength) || byteLength <= 0) {
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", "byteLength: expected a positive integer", {
                argument: "byteLength",
                value: byteLength,
                expected: "a positive integer",
            });
        }
        if (byteLength > POOL_MAX_POW2_CLASS_BYTES) {
            return Math.ceil(byteLength / POOL_LINEAR_STEP_BYTES) * POOL_LINEAR_STEP_BYTES;
        }
        let sizeClass = POOL_MIN_CLASS_BYTES;
        while (sizeClass < byteLength) {
            sizeClass *= 2;
        }
        return sizeClass;
    }

    /**
     * Acquires a buffer of at least byteLength (an idle one of the class and usage, else a new one through the
     * allocator); E_TOO_LARGE above maxBufferSize.
     * @param byteLength - the bytes needed
     * @param usage - the GPUBufferUsage flags
     * @param label - the buffer label (a reused buffer is relabelled)
     * @returns a buffer of the class size
     */
    acquire(byteLength: number, usage: number, label: string): GPUBuffer {
        if (this.disposed) {
            throw new WebGpuGraphError("E_DISPOSED", "the buffer pool was disposed", { label: "pool" });
        }
        const rounded = BufferPool.sizeClass(byteLength);
        if (byteLength > this.maxBufferSize) {
            throw new WebGpuGraphError(
                "E_TOO_LARGE",
                `pool: ${byteLength} bytes exceed the device maxBufferSize of ${this.maxBufferSize} bytes`,
                { needed: byteLength, limit: this.maxBufferSize, path: "pool", algorithm: null },
            );
        }
        const sizeClass = Math.min(rounded, this.maxBufferSize);
        const key = classKey(sizeClass, usage);
        const idle = this.idle.get(key);
        let buffer = idle?.pop();
        if (buffer === undefined) {
            buffer = this.allocator.createBuffer({ label, size: sizeClass, usage });
        } else {
            this.idleBytesValue -= sizeClass;
            buffer.label = label;
        }
        this.live.set(buffer, { sizeClass, usage });
        this.liveBytesValue += sizeClass;
        return buffer;
    }

    /**
     * Returns a buffer to its class; destroys it when the class already holds maxIdlePerClass idle buffers. A no-op
     * after destroyAll() (so a `finally` never masks the original error).
     * @param buffer - a buffer acquired from this pool and not yet released
     */
    release(buffer: GPUBuffer): void {
        if (this.disposed) {
            return;
        }
        const entry = this.live.get(buffer);
        if (entry === undefined) {
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", "buffer: not a live buffer of this pool", {
                argument: "buffer",
                value: buffer.label,
                expected: "a buffer acquired from this pool and not yet released",
            });
        }
        this.live.delete(buffer);
        this.liveBytesValue -= entry.sizeClass;
        const key = classKey(entry.sizeClass, entry.usage);
        let idle = this.idle.get(key);
        if (idle === undefined) {
            idle = [];
            this.idle.set(key, idle);
        }
        if (idle.length >= this.maxIdlePerClass) {
            this.allocator.destroy(buffer);
            return;
        }
        idle.push(buffer);
        this.idleBytesValue += entry.sizeClass;
    }

    /** Destroys every idle buffer. */
    trim(): void {
        for (const buffers of this.idle.values()) {
            for (const buffer of buffers) {
                this.allocator.destroy(buffer);
            }
        }
        this.idle.clear();
        this.idleBytesValue = 0;
    }

    /** Destroys everything, idle and live (ctx.dispose()); idempotent. @internal */
    destroyAll(): void {
        this.trim();
        for (const buffer of this.live.keys()) {
            this.allocator.destroy(buffer);
        }
        this.live.clear();
        this.liveBytesValue = 0;
        this.disposed = true;
    }

    /**
     * A scope object that releases everything acquired through it (spec 4.4, P2).
     * @returns a fresh lease over this pool
     */
    lease(): Lease {
        return new Lease(this);
    }

    /**
     * Bytes of buffers acquired and not yet released (class sizes).
     * @returns the live byte count
     */
    get liveBytes(): number {
        return this.liveBytesValue;
    }

    /**
     * Bytes of idle buffers (class sizes).
     * @returns the idle byte count
     */
    get idleBytes(): number {
        return this.idleBytesValue;
    }
}
