/**
 * `degree(ctx, s)` -- the walking-skeleton algorithm, kept public as a diagnostic (spec 3.3, 11.5): the out-degree of
 * every node computed on the device by the row-walking `degree` kernel (4.5) over the resident core of the snapshot,
 * with the group-0 dummy-binding pattern of spec 3.5 (`colIdx` in the `weights` slot when the snapshot is unweighted,
 * `rowPtr` in the `perm` slot because P1-P3 pass no row permutation, so `USE_PERM` is false). The result equals
 * `snapshot.outDegree()`; the point of the kernel is to prove the upload, bind, dispatch and readback path end to end
 * on every adapter (bitwise identical across adapters, spec 11.5), not to be faster than the CPU (the README says
 * so).
 *
 * Contract (3.12): `ctx.assertReady()` first; `nodeCount === 0` returns an empty `Uint32Array` (or `dest`) with no
 * GPU work (spec 5.6: there is no work, which is not a fallback); an already-aborted `signal` is `E_ABORTED` before
 * any work; `dest` must be a `Uint32Array` of length n over an `ArrayBuffer`; a windowed core is `E_TOO_LARGE
 * { path: "windowed", algorithm: "degree" }` until P4 executes windows -- `residency.core()` (3.8) throws that error
 * itself with `algorithm: null` before it would ever return a windowed `CoreBinding`, so `degree` catches it and
 * re-throws the same error with `algorithm: "degree"` (the 3.12 detail) instead of testing `core.plan`, which never
 * observes "windowed" in P1-P3; `arcCount === 0` skips the dispatch (only `rowPtr` is resident)
 * and the result is zeros; otherwise ONE dispatch over rows [0, n) with `accumulate = 0` (every row is written, so no
 * fill precedes it), a readback into the result, `onProgress(1, 1)`, and the scratch returned in a finally. Two runs
 * are bitwise identical (spec 11.9 item 4).
 */

import { type GraphSnapshot, type U32 } from "@graphty/graph-format";

import { type GpuContext } from "../context.js";
import { BufferUsage } from "../device/webgpu-constants.js";
import { isWebGpuGraphError, WebGpuGraphError } from "../errors.js";
import { plan1d } from "../kernel/dispatch.js";
import { graphBindings, graphOverrides, kernelSpec, RANGE_PARAMS } from "../kernels.js";
import { type CoreBinding } from "../memory/residency.js";
import { type Binding } from "../types/memory.js";
import { type GpuRunOptions } from "../types/run.js";

/**
 * Validates `options.dest` for a result of `n` elements.
 * @param dest - the caller's destination array, if any
 * @param n - the node count
 * @returns the destination as a U32, or null when none was given
 */
function checkDest(dest: Float32Array | Uint32Array | undefined, n: number): U32 | null {
    if (dest === undefined) {
        return null;
    }
    if (dest instanceof Uint32Array && dest.length === n && dest.buffer instanceof ArrayBuffer) {
        return dest as U32;
    }
    throw new WebGpuGraphError(
        "E_INVALID_ARGUMENT",
        `degree: dest must be a Uint32Array of length ${n} over an ArrayBuffer`,
        {
            argument: "dest",
            value: `${dest.constructor.name}(${dest.length})`,
            expected: `Uint32Array(${n}) over an ArrayBuffer`,
        },
    );
}

/**
 * Uploads (or finds) the core through the residency; a windowed plan, which the residency reports as `E_TOO_LARGE
 * { path: "windowed", algorithm: null }` (3.8), is re-thrown with `algorithm: "degree"` (3.12) and its other details
 * (`needed`, `limit`, `path`) kept. Every other error passes through untouched.
 * @param ctx - the context whose residency holds the core
 * @param s - the snapshot
 * @returns the resident core binding (arena or perArray)
 */
function coreOf(ctx: GpuContext, s: GraphSnapshot): CoreBinding {
    try {
        return ctx.residency.core(s);
    } catch (error: unknown) {
        if (isWebGpuGraphError(error) && error.code === "E_TOO_LARGE" && error.details.path === "windowed") {
            throw new WebGpuGraphError(
                "E_TOO_LARGE",
                "degree: the arc arrays need a windowed upload, which P1-P3 plan but do not execute",
                { ...error.details, algorithm: "degree" },
            );
        }
        throw error;
    }
}

/**
 * The walking-skeleton kernel, kept public as a diagnostic (spec 3.3): out-degree per node through the row-walking
 * gather with the USE_PERM dummy pattern; equals snapshot.outDegree().
 * @param ctx - the context whose device runs the kernel
 * @param s - the snapshot (uploaded through ctx.residency, or found there)
 * @param options - dest / signal / onProgress (spec 3.3). PLAN DECISION: spelled `?: GpuRunOptions` rather than the
 *   contract's `?: GpuRunOptions | undefined` because the root ESLint rule no-duplicate-type-constituents rejects the
 *   explicit undefined on an optional parameter (the call signature is identical; P1-T3 made the same choice).
 * @returns the out-degree of every node, index-aligned (`dest` itself when given)
 */
export async function degree(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<U32> {
    ctx.assertReady();
    const n = s.nodeCount;
    const dest = checkDest(options?.dest, n);
    if (options?.signal?.aborted) {
        throw new WebGpuGraphError("E_ABORTED", "degree: the signal was aborted before any work started", {});
    }
    if (n === 0) {
        options?.onProgress?.(1, 1);
        return dest ?? new Uint32Array(0);
    }
    const core = coreOf(ctx, s);
    if (s.arcCount === 0) {
        const zeros = dest ?? new Uint32Array(n);
        zeros.fill(0);
        options?.onProgress?.(1, 1);
        return zeros;
    }
    const byteLength = n * 4;
    const out = ctx.pool.acquire(
        byteLength,
        BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
        "degree/out",
    );
    const params = ctx.pool.acquire(
        RANGE_PARAMS.byteLength,
        BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        "degree/params",
    );
    try {
        await ctx.allocator.check();
        const kernel = await ctx.pipelines.kernel(kernelSpec("degree", graphOverrides(core, null)));
        const bytes = new ArrayBuffer(RANGE_PARAMS.byteLength);
        RANGE_PARAMS.write(new DataView(bytes), { start: 0, end: n, arcBase: 0, arcEnd: s.arcCount, accumulate: 0, n });
        ctx.device.queue.writeBuffer(params, 0, bytes);
        const outBinding: Binding = { buffer: out, offset: 0, size: byteLength, window: null };
        const paramsBinding: Binding = { buffer: params, offset: 0, size: RANGE_PARAMS.byteLength, window: null };
        const bound = kernel.bind({ ...graphBindings(core, null), out: outBinding, P: paramsBinding });
        const encoder = ctx.device.createCommandEncoder({ label: "degree" });
        const pass = encoder.beginComputePass({ label: "degree" });
        kernel.dispatch(pass, bound, plan1d(n, ctx.workgroupSize, ctx.caps), [0]);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        ctx.assertReady();
        const result = await ctx.readback.read(out, byteLength, dest ?? undefined);
        ctx.assertReady();
        options?.onProgress?.(1, 1);
        return dest ?? new Uint32Array(result);
    } finally {
        ctx.pool.release(params);
        ctx.pool.release(out);
    }
}
