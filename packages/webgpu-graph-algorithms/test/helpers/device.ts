/**
 * Raw-buffer helpers around a context (contract 5.2): STORAGE | COPY_SRC | COPY_DST buffers created DIRECTLY on the
 * device (never through the allocator, so allocator.liveBuffers keeps counting residency and pool buffers only),
 * whole-buffer bindings, U32 / F32 readbacks through ctx.readback, and a context scope that disposes on the way out.
 * The copy-before-unmap idiom lives in src/memory/readback.ts (gpu-upload.test.ts lines 122-131 are its origin).
 */

import { type F32, type U32 } from "@graphty/graph-format";

import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { type Binding } from "../../src/types/memory.js";
import { acquire, type AcquireOptions } from "../setup/gpu.js";

/** Every raw buffer is bindable as storage, readable back and writable by writeBuffer. */
const RAW_USAGE = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;

/**
 * A STORAGE | COPY_SRC | COPY_DST buffer holding `data` (one writeBuffer, the array as given: no copy, no cast).
 * @param ctx - the context
 * @param data - the bytes (byteLength a multiple of 4)
 * @param label - the buffer label
 * @param extraUsage - further GPUBufferUsage bits
 * @returns the buffer
 */
export function uploadBuffer(ctx: GpuContext, data: ArrayBufferView, label: string, extraUsage?: number): GPUBuffer {
    const buffer = ctx.device.createBuffer({ label, size: data.byteLength, usage: RAW_USAGE | (extraUsage ?? 0) });
    // writeBuffer takes ArrayBufferView<ArrayBuffer> under @webgpu/types 0.1.72; test data is never SharedArrayBuffer-backed
    ctx.device.queue.writeBuffer(buffer, 0, data as ArrayBufferView<ArrayBuffer>);
    return buffer;
}

/**
 * A zeroed STORAGE | COPY_SRC | COPY_DST buffer of byteLength (WebGPU zero-initialises new buffers).
 * @param ctx - the context
 * @param byteLength - the size (a multiple of 4)
 * @param label - the buffer label
 * @returns the buffer
 */
export function scratchBuffer(ctx: GpuContext, byteLength: number, label: string): GPUBuffer {
    return ctx.device.createBuffer({ label, size: byteLength, usage: RAW_USAGE });
}

/**
 * Whole-buffer Binding.
 * @param buffer - the buffer
 * @returns the binding over [0, buffer.size)
 */
export function bindingOf(buffer: GPUBuffer): Binding {
    return { buffer, offset: 0, size: buffer.size, window: null };
}

/**
 * Reads `count` u32 words back through ctx.readback.
 * @param ctx - the context
 * @param buffer - a COPY_SRC buffer
 * @param count - words to read
 * @param byteOffset - where to start (default 0)
 * @returns a fresh Uint32Array<ArrayBuffer>
 */
export async function readU32(ctx: GpuContext, buffer: GPUBuffer, count: number, byteOffset?: number): Promise<U32> {
    return new Uint32Array(await ctx.readback.read(buffer, 4 * count, undefined, byteOffset));
}

/**
 * Reads `count` f32 values back through ctx.readback.
 * @param ctx - the context
 * @param buffer - a COPY_SRC buffer
 * @param count - values to read
 * @param byteOffset - where to start (default 0)
 * @returns a fresh Float32Array<ArrayBuffer>
 */
export async function readF32(ctx: GpuContext, buffer: GPUBuffer, count: number, byteOffset?: number): Promise<F32> {
    return new Float32Array(await ctx.readback.read(buffer, 4 * count, undefined, byteOffset));
}

/**
 * Runs `fn` with a context from acquire(options) and disposes it afterwards (also on throw).
 * @param options - the acquire options
 * @param fn - the body
 * @returns what `fn` resolves
 */
export async function withContext<T>(
    options: AcquireOptions | undefined,
    fn: (ctx: GpuContext) => Promise<T>,
): Promise<T> {
    const ctx = await acquire(options);
    try {
        return await fn(ctx);
    } finally {
        ctx.dispose();
    }
}
