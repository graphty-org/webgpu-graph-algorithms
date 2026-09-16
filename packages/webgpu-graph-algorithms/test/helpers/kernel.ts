/**
 * runKernel: P1's pre-CommandBatch dispatch driver (contract 5.2). Records ONE dispatch of a kernel with the given
 * bindings and params values into a fresh encoder and submits; the params slot is a fresh UNIFORM | COPY_DST buffer
 * of the block's byteLength, destroyed once the queue reports the submitted work done. Because that buffer is new on
 * every call, the bind groups of a params run can never be reused: the helper calls `kernel.invalidate()` right after
 * the submit so `Kernel`'s bind-group cache does not grow by one entry (three GPUBindGroups over a destroyed buffer)
 * per call across the loops of the P1-T4 17M-item test and the P1-T5 / P1-T6 suites. The pre-CommandBatch driver
 * therefore never reuses bind groups; BoundKernels other code already holds stay valid (invalidate() only forgets).
 * Imports only src/kernel/** and src/context.js, so P1-T2's helpers and P1-T3's stay independent.
 *
 * PLAN DECISION: `params?` is spelled `?: T` rather than the contract's `?: T | undefined` because the root ESLint
 * rule no-duplicate-type-constituents rejects the explicit undefined on an optional parameter (the call signature is
 * identical).
 *
 * PLAN DECISION (test-only module ids): every spec a test builds carries an id prefixed `test-` (`test-add`,
 * `test-copy`, `test-total`, `test-broken`, `test-other`, and every ad hoc spec of later tests). They compile
 * through `ctx.pipelines` of `acquire()` contexts, so their keys reach the GRAPHTY_PIPELINE_KEY_LOG directory in
 * afterAll; P2-T2's coverage check (`matrixCovers` and the test/setup/global.ts teardown) MUST ignore every key
 * whose id segment (the text before the first `|`) is not a KernelId of src/kernels.ts, so no `test-*` key ever
 * counts as uncovered.
 */

import { type GpuContext } from "../../src/context.js";
import { type DispatchPlan } from "../../src/kernel/dispatch.js";
import { type KernelBindings } from "../../src/kernel/kernel.js";
import { type UniformBlock, type UniformValues } from "../../src/kernel/struct-block.js";
import { type WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { type Binding } from "../../src/types/memory.js";

/** Records one dispatch of a kernel with the given bindings and params values into a fresh encoder and submits (P1's pre-CommandBatch driver); the params slot is a fresh UNIFORM | COPY_DST buffer of the block's byteLength. */
export async function runKernel(
    ctx: GpuContext,
    spec: WgslModuleSpec,
    bindings: KernelBindings,
    plan: DispatchPlan,
    params?: { readonly block: UniformBlock; readonly values: UniformValues },
): Promise<void> {
    const kernel = await ctx.pipelines.kernel(spec);
    const resources: Record<string, Binding> = { ...bindings };
    let paramsBuffer: GPUBuffer | null = null;
    if (params !== undefined) {
        const decl = spec.bindings.find((b) => b.kind === "uniform" && b.wgslType === params.block.name);
        if (decl === undefined) {
            throw new Error(`runKernel: ${spec.id} has no uniform binding of type ${params.block.name}`);
        }
        const bytes = new ArrayBuffer(params.block.byteLength);
        params.block.write(new DataView(bytes), params.values);
        paramsBuffer = ctx.device.createBuffer({
            label: `${spec.id}/params`,
            size: params.block.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        ctx.device.queue.writeBuffer(paramsBuffer, 0, bytes);
        resources[decl.name] = { buffer: paramsBuffer, offset: 0, size: params.block.byteLength, window: null };
    }
    const bound = kernel.bind(resources);
    const encoder = ctx.device.createCommandEncoder({ label: `${spec.id}/runKernel` });
    const pass = encoder.beginComputePass({ label: spec.id });
    kernel.dispatch(pass, bound, plan);
    pass.end();
    ctx.device.queue.submit([encoder.finish()]);
    if (paramsBuffer !== null) {
        // the params buffer is unique to this call: drop its bind groups from the kernel's cache (no eviction otherwise)
        kernel.invalidate();
        await ctx.device.queue.onSubmittedWorkDone();
        paramsBuffer.destroy();
    }
}
