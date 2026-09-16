/**
 * The exact-tier repulsion stage of ForceAtlas2 (spec 7.6, 7.9, 7.10; contract 3.13): K3, the tiled all-pairs
 * repulsion with the gravity term and the swing / traction epilogue, followed by K4, the one-workgroup speed finalize
 * that folds the per-workgroup partials and runs estimateFactor on the device (D15). P1 ships the stage driven over a
 * pass the caller opens (the tests); P3-T2 records it from ForceAtlas2Model.recordIteration without changing these
 * declarations. Both kernels come from the registry (src/kernels.ts) through the pipeline cache, so a distinct
 * override set is a distinct pipeline and the subgroup twin is selected by the device's features (spec 5.1, D16).
 */

import { WebGpuGraphError } from "../errors.js";
import { type DispatchPlan, plan1d } from "../kernel/dispatch.js";
import { type BoundKernel, type Kernel } from "../kernel/kernel.js";
import { type PipelineCache } from "../kernel/pipeline-cache.js";
import { type WgslModuleSpec } from "../kernel/wgsl.js";
import { kernelSpec } from "../kernels.js";
import { type PlanCaps } from "../types/context.js";
import { type Binding } from "../types/memory.js";

/** The buffers the exact-tier repulsion stage binds (all in layout units; spec 7.3). */
export interface RepulsionExactResources {
    readonly pos: Binding;
    readonly state: Binding;
    readonly trace: Binding;
    readonly force: Binding;
    readonly oldForce: Binding;
    readonly fixedMask: Binding;
    readonly partials: Binding;
    readonly params: Binding;
}

/** The overrides K3 / K4 compile with. */
export interface RepulsionExactOverrides {
    readonly SWING_MODE: 0 | 1;
    readonly STRONG_GRAVITY: boolean;
    readonly GRAVITY_CENTER: 0 | 1;
}

/** K3 (tiled all-pairs repulsion + gravity + the swing / traction epilogue) followed by K4 (the one-workgroup speed finalize) (spec 7.6, 7.10). */
export class RepulsionExact {
    /** The overrides both kernels were compiled with (a frozen copy of the argument of create()). */
    readonly overrides: RepulsionExactOverrides;

    private readonly caps: PlanCaps;
    private readonly repulsion: Kernel;
    private readonly speedFinalize: Kernel;
    /** Exactly one workgroup of K4 (spec 7.4: "1 workgroup"). */
    private readonly finalizePlan: DispatchPlan;
    private boundRepulsion: BoundKernel | null = null;
    private boundSpeedFinalize: BoundKernel | null = null;

    /**
     * Holds the two compiled kernels; create() is the only caller.
     * @param repulsion - the K3 kernel
     * @param speedFinalize - the K4 kernel
     * @param caps - the device caps the dispatch planner reads
     * @param overrides - the override set both kernels were compiled with
     */
    private constructor(repulsion: Kernel, speedFinalize: Kernel, caps: PlanCaps, overrides: RepulsionExactOverrides) {
        this.repulsion = repulsion;
        this.speedFinalize = speedFinalize;
        this.caps = caps;
        this.overrides = Object.freeze({
            SWING_MODE: overrides.SWING_MODE,
            STRONG_GRAVITY: overrides.STRONG_GRAVITY,
            GRAVITY_CENTER: overrides.GRAVITY_CENTER,
        });
        this.finalizePlan = plan1d(speedFinalize.workgroupSize, speedFinalize.workgroupSize, caps);
    }

    /**
     * Compiles both kernels through the cache (the twin is selected by caps.features).
     * @param pipelines - the context's pipeline cache
     * @param caps - the device caps the dispatch planner reads
     * @param overrides - the SWING_MODE / STRONG_GRAVITY / GRAVITY_CENTER set of this stage
     * @returns the stage, ready for bind()
     */
    static async create(
        pipelines: PipelineCache,
        caps: PlanCaps,
        overrides: RepulsionExactOverrides,
    ): Promise<RepulsionExact> {
        const [repulsionSpec, finalizeSpec] = RepulsionExact.specs(overrides);
        // Sequential on purpose: PipelineCache.get runs createComputePipelineAsync inside a validation scope, and error
        // scopes form one stack per device, so two interleaved compilations would pop each other's scope.
        const repulsion = await pipelines.kernel(repulsionSpec);
        const speedFinalize = await pipelines.kernel(finalizeSpec);
        return new RepulsionExact(repulsion, speedFinalize, caps, overrides);
    }

    /**
     * The two specs (for warm() and the compile matrix).
     * @param overrides - the override set of the stage
     * @returns the K3 spec then the K4 spec
     */
    static specs(overrides: RepulsionExactOverrides): readonly [WgslModuleSpec, WgslModuleSpec] {
        return [
            kernelSpec("fa2-repulsion-exact", {
                SWING_MODE: overrides.SWING_MODE,
                STRONG_GRAVITY: overrides.STRONG_GRAVITY,
                GRAVITY_CENTER: overrides.GRAVITY_CENTER,
            }),
            kernelSpec("fa2-speed-finalize", { SWING_MODE: overrides.SWING_MODE }),
        ];
    }

    /**
     * Creates the bind groups once per load(): K3 takes pos, state (S), force, oldForce, fixedMask, partials and the
     * params slot; K4 takes partials, state (S), trace (T) and the params slot (the 3.10.1 binding names). The state
     * header and the trace region are two disjoint ranges of one buffer, which Kernel.bind() accepts (no writable alias).
     * @param resources - the buffers of spec 7.3
     */
    bind(resources: RepulsionExactResources): void {
        this.boundRepulsion = this.repulsion.bind({
            pos: resources.pos,
            S: resources.state,
            force: resources.force,
            oldForce: resources.oldForce,
            fixedMask: resources.fixedMask,
            partials: resources.partials,
            P: resources.params,
        });
        this.boundSpeedFinalize = this.speedFinalize.bind({
            partials: resources.partials,
            S: resources.state,
            T: resources.trace,
            P: resources.params,
        });
    }

    /**
     * Records K3 (plan1d(n)) then K4 (1 workgroup) with the params slot's dynamic offset.
     * @param pass - the open compute pass of the batch
     * @param n - the node count (rows [0, n) of pos / force)
     * @param paramsOffset - the dynamic offset of this iteration's Fa2Params slot in the uniform ring
     */
    record(pass: GPUComputePassEncoder, n: number, paramsOffset: number): void {
        this.recordRepulsion(pass, n, paramsOffset);
        this.recordSpeedFinalize(pass, paramsOffset);
    }

    /**
     * Records K3 only (the inspect() stage split, spec 11.9 item 2).
     * @param pass - the open compute pass
     * @param n - the node count
     * @param paramsOffset - the dynamic offset of the Fa2Params slot
     * @internal
     */
    recordRepulsion(pass: GPUComputePassEncoder, n: number, paramsOffset: number): void {
        const bound = this.bound(this.boundRepulsion, "recordRepulsion");
        const plan = plan1d(n, this.repulsion.workgroupSize, this.caps);
        this.repulsion.dispatch(pass, bound, plan, [paramsOffset]);
    }

    /**
     * Records K4 only.
     * @param pass - the open compute pass
     * @param paramsOffset - the dynamic offset of the Fa2Params slot
     * @internal
     */
    recordSpeedFinalize(pass: GPUComputePassEncoder, paramsOffset: number): void {
        const bound = this.bound(this.boundSpeedFinalize, "recordSpeedFinalize");
        this.speedFinalize.dispatch(pass, bound, this.finalizePlan, [paramsOffset]);
    }

    /**
     * The bound kernel of a method, or E_NOT_LOADED when bind() has not run.
     * @param bound - the cached BoundKernel, null before bind()
     * @param method - the caller's name for the message
     * @returns the bound kernel
     */
    private bound(bound: BoundKernel | null, method: string): BoundKernel {
        if (bound === null) {
            throw new WebGpuGraphError(
                "E_NOT_LOADED",
                `RepulsionExact.${method}(): bind() has not been called for this stage`,
                {
                    state: "unbound",
                },
            );
        }
        return bound;
    }
}
