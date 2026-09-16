/**
 * Kernel (spec 5.1; contract 3.9): a compiled pipeline bound to the binding list of its spec. `bind()` creates the
 * bind groups from a `Record<name, Binding>` keyed by the declared names and caches them by the identity of the
 * buffers, offsets and sizes (a layout's bind groups are created once per load(), not per iteration); `dispatch()`
 * records setPipeline / setBindGroup / dispatchWorkgroups and skips an empty plan (spec 5.6).
 */

import { STORAGE_ALIGN } from "../constants.js";
import { WebGpuGraphError } from "../errors.js";
import { type Binding } from "../types/memory.js";
import { type DispatchPlan } from "./dispatch.js";
import { type BindingDecl, type ComposedModule, type WgslModuleSpec } from "./wgsl.js";

/** The resources of one bind(): one Binding per BindingDecl name; a uniform decl takes the ring's whole-buffer binding (the slot is chosen by the dynamic offset at dispatch). */
export type KernelBindings = Readonly<Record<string, Binding>>;
/** A kernel with its bind groups created (spec 5.1: cached per set of buffers and offsets). */
export interface BoundKernel {
    readonly kernel: Kernel;
    readonly bindGroups: readonly GPUBindGroup[];
    /** Group indices whose bind group takes a dynamic offset (the uniform groups), in group order. */
    readonly dynamicGroups: readonly number[];
}

/** Buffer identities for the bind-group cache key: every GPUBuffer seen by any Kernel gets one number, once. */
const bufferIds = new WeakMap<GPUBuffer, number>();
let nextBufferId = 1;

/**
 * The identity number of a buffer.
 * @param buffer - the buffer
 * @returns its number
 */
function bufferId(buffer: GPUBuffer): number {
    let id = bufferIds.get(buffer);
    if (id === undefined) {
        id = nextBufferId;
        nextBufferId += 1;
        bufferIds.set(buffer, id);
    }
    return id;
}

/**
 * True when two byte ranges intersect.
 * @param a - one binding
 * @param b - the other binding
 * @returns whether [a.offset, a.offset + a.size) and [b.offset, b.offset + b.size) overlap
 */
function rangesIntersect(a: Binding, b: Binding): boolean {
    return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

/** A compiled pipeline plus the binding list of its spec (spec 5.1). */
export class Kernel {
    /** The module spec the pipeline was compiled from (its bindings drive bind()). */
    readonly spec: WgslModuleSpec;
    /** The compiled compute pipeline. */
    readonly pipeline: GPUComputePipeline;
    /** The explicit bind group layouts, one per group index (an empty layout for an unused index). */
    readonly layouts: readonly GPUBindGroupLayout[];
    /** The workgroup size the pipeline was compiled with (the effective WG override). */
    readonly workgroupSize: number;
    /** The entry point name the pipeline was compiled with. */
    readonly entryPoint: string;
    private readonly device: GPUDevice;
    /** The declarations of each group in binding order (index = group). */
    private readonly groups: readonly (readonly BindingDecl[])[];
    /** The uniform-binding count of each group (the dynamic offsets a setBindGroup takes). */
    private readonly dynamicCounts: readonly number[];
    /** Cached bind groups by buffer identity + offset + size. */
    private readonly cache = new Map<string, BoundKernel>();

    /**
     * Wraps a compiled pipeline.
     * @param device - the device the bind groups are created on
     * @param spec - the module
     * @param composed - the composed text (workgroup size and entry point)
     * @param pipeline - the compiled pipeline
     * @param layouts - the explicit bind-group layouts, one per group 0..maxGroup
     */
    constructor(
        device: GPUDevice,
        spec: WgslModuleSpec,
        composed: ComposedModule,
        pipeline: GPUComputePipeline,
        layouts: readonly GPUBindGroupLayout[],
    ) {
        this.device = device;
        this.spec = spec;
        this.pipeline = pipeline;
        this.layouts = layouts;
        this.workgroupSize = composed.overrides.WG as number;
        this.entryPoint = composed.entryPoint;
        const groups: BindingDecl[][] = layouts.map(() => []);
        for (const decl of spec.bindings) {
            groups[decl.group].push(decl);
        }
        for (const group of groups) {
            group.sort((a, b) => a.binding - b.binding);
        }
        this.groups = groups;
        this.dynamicCounts = groups.map((group) => group.filter((decl) => decl.kind === "uniform").length);
    }

    /**
     * Creates (or reuses, keyed by every buffer identity + offset + size) the bind groups, each labelled
     * `<spec.id>/<group>`; a missing or extra name is E_INVALID_ARGUMENT; two bindings of one call whose ranges
     * intersect on one buffer while either slot is `storage` (read_write) is E_INVALID_ARGUMENT
     * { argument: "aliasing" } (the host-side mirror of WebGPU's writable buffer-binding-aliasing rule, 3.10.1, so
     * the failure is synchronous and labelled). CONTRACT DECISION: bind() is synchronous, so a createBindGroup
     * validation error (a wrong-size uniform binding, a usage mismatch) is NOT thrown here -- it reaches the
     * pending-error slot as E_VALIDATION { label: "<spec.id>/<group>" } and is thrown by the batch's readback
     * (Dawn-node) or the next assertReady() (browser), spec 5.7. An empty bind group is created for every empty
     * layout index so setBindGroup is called for 0..maxGroup.
     * PLAN DECISION (verified on Dawn 0.4.0 / lavapipe): one buffer bound with two DIFFERENT access modes when one is
     * `storage` is also E_INVALID_ARGUMENT { argument: "aliasing", reason: "usage" } even with disjoint ranges, because
     * WebGPU's usage-scope rule rejects "writable usage and another usage in the same synchronization scope" per
     * BUFFER; two `storage` slots with disjoint ranges (the state header / trace pattern) are accepted; a zero-size
     * binding (never bind a zero-length buffer, spec 3.6) and an offset that is not a multiple of STORAGE_ALIGN = 256
     * (spec 2.6; Dawn: "does not satisfy the minimum ... alignment (256)") are E_INVALID_ARGUMENT.
     * @param resources - one Binding per declared name
     * @returns the bound kernel (the same object for the same buffers, offsets and sizes until invalidate())
     */
    bind(resources: KernelBindings): BoundKernel {
        const { id } = this.spec;
        const declared = new Map(this.spec.bindings.map((decl) => [decl.name, decl] as const));
        for (const name of Object.keys(resources)) {
            if (!declared.has(name)) {
                throw new WebGpuGraphError("E_INVALID_ARGUMENT", `${id}: no binding named "${name}"`, {
                    argument: "bindings",
                    value: name,
                    expected: [...declared.keys()].join(", "),
                });
            }
        }
        const resolved: { readonly decl: BindingDecl; readonly binding: Binding }[] = [];
        for (const decl of this.spec.bindings) {
            const binding = resources[decl.name] as Binding | undefined;
            if (binding === undefined) {
                throw new WebGpuGraphError("E_INVALID_ARGUMENT", `${id}: binding "${decl.name}" is missing`, {
                    argument: "bindings",
                    value: decl.name,
                    expected: "a Binding for every declared name",
                });
            }
            if (!(Number.isInteger(binding.size) && binding.size > 0)) {
                throw new WebGpuGraphError(
                    "E_INVALID_ARGUMENT",
                    `${id}: binding "${decl.name}" has size ${binding.size}`,
                    {
                        argument: decl.name,
                        value: binding.size,
                        expected: "size > 0 (a zero-length buffer is never bound)",
                    },
                );
            }
            if (!(Number.isInteger(binding.offset) && binding.offset >= 0 && binding.offset % STORAGE_ALIGN === 0)) {
                throw new WebGpuGraphError(
                    "E_INVALID_ARGUMENT",
                    `${id}: binding "${decl.name}" has offset ${binding.offset}`,
                    {
                        argument: decl.name,
                        value: binding.offset,
                        expected: `a multiple of ${STORAGE_ALIGN} (spec 2.6: the offset alignment the package always honours)`,
                    },
                );
            }
            resolved.push({ decl, binding });
        }
        for (let i = 0; i < resolved.length; i++) {
            for (let j = i + 1; j < resolved.length; j++) {
                const a = resolved[i];
                const b = resolved[j];
                if (a.binding.buffer !== b.binding.buffer) {
                    continue;
                }
                const writable = a.decl.kind === "storage" || b.decl.kind === "storage";
                if (!writable) {
                    continue;
                }
                if (a.decl.kind !== b.decl.kind) {
                    throw new WebGpuGraphError(
                        "E_INVALID_ARGUMENT",
                        `${id}: "${a.decl.name}" and "${b.decl.name}" bind one buffer with different access modes (writable usage next to another usage)`,
                        {
                            argument: "aliasing",
                            value: [a.decl.name, b.decl.name],
                            expected: "one access mode per buffer when a slot is read_write",
                            reason: "usage",
                        },
                    );
                }
                if (rangesIntersect(a.binding, b.binding)) {
                    throw new WebGpuGraphError(
                        "E_INVALID_ARGUMENT",
                        `${id}: "${a.decl.name}" and "${b.decl.name}" bind intersecting ranges of one read_write buffer`,
                        {
                            argument: "aliasing",
                            value: [a.decl.name, b.decl.name],
                            expected: "disjoint ranges",
                            reason: "range",
                        },
                    );
                }
            }
        }
        const key = resolved
            .map(({ binding }) => `${bufferId(binding.buffer)}:${binding.offset}:${binding.size}`)
            .join("|");
        const cached = this.cache.get(key);
        if (cached !== undefined) {
            return cached;
        }
        const byName = new Map(resolved.map(({ decl, binding }) => [decl.name, binding] as const));
        const bindGroups = this.layouts.map((layout, group) =>
            this.device.createBindGroup({
                label: `${id}/${group}`,
                layout,
                entries: this.groups[group].map((decl) => {
                    const binding = byName.get(decl.name) as Binding;
                    return {
                        binding: decl.binding,
                        resource: { buffer: binding.buffer, offset: binding.offset, size: binding.size },
                    };
                }),
            }),
        );
        const dynamicGroups: number[] = [];
        this.dynamicCounts.forEach((count, group) => {
            if (count > 0) {
                dynamicGroups.push(group);
            }
        });
        const bound: BoundKernel = Object.freeze({
            kernel: this,
            bindGroups: Object.freeze(bindGroups),
            dynamicGroups: Object.freeze(dynamicGroups),
        });
        this.cache.set(key, bound);
        return bound;
    }

    /**
     * setPipeline + setBindGroup for every group (dynamic offsets in dynamicGroups order) + dispatchWorkgroups(plan.x, plan.y, 1); a plan with x === 0 records nothing (spec 5.6).
     * PLAN DECISION: one dynamic offset per dynamic GROUP, replicated over every uniform binding of that group (every
     * P1-P3 kernel has exactly one params uniform per group); absent offsets mean 0; a BoundKernel of another kernel
     * or an offset list of the wrong length is E_INVALID_ARGUMENT.
     * PLAN DECISION: `dynamicOffsets?` is spelled `?: readonly number[]` rather than the contract's
     * `?: readonly number[] | undefined` because the root ESLint rule no-duplicate-type-constituents rejects the
     * explicit undefined on an optional parameter (the call signature is identical).
     * @param pass - the open compute pass
     * @param bound - a BoundKernel of THIS kernel
     * @param plan - the dispatch shape
     * @param dynamicOffsets - one byte offset per entry of bound.dynamicGroups
     */
    dispatch(
        pass: GPUComputePassEncoder,
        bound: BoundKernel,
        plan: DispatchPlan,
        dynamicOffsets?: readonly number[],
    ): void {
        const { id } = this.spec;
        if (bound.kernel !== this) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `${id}: the bound kernel belongs to "${bound.kernel.spec.id}"`,
                {
                    argument: "bound",
                    value: bound.kernel.spec.id,
                    expected: id,
                },
            );
        }
        if (dynamicOffsets !== undefined && dynamicOffsets.length !== bound.dynamicGroups.length) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `${id}: ${dynamicOffsets.length} dynamic offsets for ${bound.dynamicGroups.length} dynamic groups`,
                {
                    argument: "dynamicOffsets",
                    value: dynamicOffsets.length,
                    expected: bound.dynamicGroups.length,
                },
            );
        }
        if (plan.x === 0) {
            return;
        }
        pass.setPipeline(this.pipeline);
        let next = 0;
        bound.bindGroups.forEach((bindGroup, group) => {
            const count = this.dynamicCounts[group];
            if (count > 0) {
                const offset = dynamicOffsets?.[next] ?? 0;
                next += 1;
                const offsets: number[] = [];
                for (let k = 0; k < count; k++) {
                    offsets.push(offset);
                }
                pass.setBindGroup(group, bindGroup, offsets);
            } else {
                pass.setBindGroup(group, bindGroup);
            }
        });
        pass.dispatchWorkgroups(plan.x, plan.y, 1);
    }

    /** Drops cached bind groups (a layout's buffers changed). */
    invalidate(): void {
        this.cache.clear();
    }
}
