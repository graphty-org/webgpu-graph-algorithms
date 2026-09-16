/**
 * The `reduce` primitive driver (spec 6 row 1): sum / min / max over f32, u32 or vec4f elements in two or three
 * dispatches of the `reduce` module (3.10.1). Level 1 folds WG elements per workgroup into per-workgroup partials
 * (the prelude's tree helpers, 4.3); a third level folds the partials again when the level-1 grid needed a 2D
 * dispatch (more than MAX_WORKGROUPS_PER_DIM workgroups, i.e. more than MAX_1D_ITEMS elements); the FINAL level --
 * ONE workgroup whose lanes walk the remaining partials sequentially in index order -- writes one element
 * (4 or 16 bytes) at out[outOffset]. Every level reduces in a fixed order, so two runs are bitwise identical on one
 * device (spec 11.9 item 4); the subgroup twin (4.3) agrees to summation-order noise on f32 and bitwise on u32.
 * `count === 0` records level 1 over an empty plan (Kernel.dispatch records nothing, spec 5.6) and the FINAL level
 * over 0 partials, which writes the identity element.
 *
 * The driver owns no device objects: the caller supplies a ReduceScope (P1: the tests build one over a GpuContext and
 * release its scratch in a finally; P2: CommandBatch supplies the same record with a Lease) and the compute pass to
 * record into. `src/primitives/**` never imports `src/context.ts` (the eslint zone of 2.4).
 */

import { MAX_WORKGROUPS_PER_DIM } from "../constants.js";
import { WebGpuGraphError } from "../errors.js";
import { type DispatchPlan, groupsOf, plan1d } from "../kernel/dispatch.js";
import { type Kernel } from "../kernel/kernel.js";
import { type PipelineCache } from "../kernel/pipeline-cache.js";
import { type UniformBlock, type UniformValues } from "../kernel/struct-block.js";
import { kernelSpec, REDUCE_PARAMS } from "../kernels.js";
import { type BufferPool } from "../memory/buffer-pool.js";
import { type PlanCaps } from "../types/context.js";
import { type Binding } from "../types/memory.js";

/** The reduction operator (4.5 OP override: sum, min, max). */
export type ReduceOp = "sum" | "min" | "max";
/** The element type (4.5 DTYPE override: f32, u32, or vec4f lanes of f32). */
export type ReduceDtype = "f32" | "u32" | "vec4f";

/** What reduce() needs of its caller: a pass to record into, scratch, the ring and the cache (P1 has no CommandBatch yet; P2's batch supplies the same record). */
export interface ReduceScope {
    readonly device: GPUDevice;
    readonly caps: PlanCaps;
    readonly pipelines: PipelineCache;
    readonly pool: BufferPool;
    readonly workgroupSize: number;
    /** Acquires scratch released by the caller's scope (a Lease from P2; P1 releases in a finally). */
    scratch(byteLength: number, label: string): GPUBuffer;
    /** The uniform-slot writer: returns the binding and dynamic offset for a params record. */
    params(block: UniformBlock, values: UniformValues): { readonly binding: Binding; readonly offset: number };
}

/** A prepared reduce: records the 2-3 dispatches of spec 6 row 1 into a pass. */
export interface ReducePlanner {
    readonly op: ReduceOp;
    readonly dtype: ReduceDtype;
    /** Records: level 1 over `count` elements of `src` into partials; a third level when groups > MAX_WORKGROUPS_PER_DIM; the FINAL one-workgroup level writing one element (4 or 16 bytes) at out[outOffset] (element index). Deterministic order. count 0 writes the identity element. */
    record(pass: GPUComputePassEncoder, src: Binding, count: number, out: Binding, outOffset: number): void;
    /** Dispatches the last record() issued (tests bound it: 2 or 3). */
    readonly lastDispatches: number;
}

/** The OP override values of the reduce module (4.5: 0 = sum, 1 = min, 2 = max). */
const OP_CODE: Readonly<Record<ReduceOp, number>> = Object.freeze({ sum: 0, min: 1, max: 2 });
/** The DTYPE override values (4.5: 0 = f32, 1 = u32, 2 = vec4f). */
const DTYPE_CODE: Readonly<Record<ReduceDtype, number>> = Object.freeze({ f32: 0, u32: 1, vec4f: 2 });
/** Bytes of one element per dtype. */
const ELEMENT_BYTES: Readonly<Record<ReduceDtype, number>> = Object.freeze({ f32: 4, u32: 4, vec4f: 16 });

/**
 * Prepares the reduce pipelines of a scope (compiles once) so record() is synchronous.
 * @param scope - the caller's scope (device, caps, cache, scratch, params)
 * @param op - the operator
 * @param dtype - the element type
 * @returns the planner, with the level and FINAL pipelines resolved
 */
export async function prepareReduce(scope: ReduceScope, op: ReduceOp, dtype: ReduceDtype): Promise<ReducePlanner> {
    const overrides = { OP: OP_CODE[op], DTYPE: DTYPE_CODE[dtype] };
    const level = await scope.pipelines.kernel(kernelSpec("reduce", { ...overrides, FINAL: false }));
    const final = await scope.pipelines.kernel(kernelSpec("reduce", { ...overrides, FINAL: true }));
    return new ReducePlannerImpl(scope, op, dtype, level, final);
}

/**
 * The argument checks of record() (E_INVALID_ARGUMENT before anything is recorded).
 * @param src - the input binding
 * @param count - the element count
 * @param out - the output binding
 * @param outOffset - the output element index
 * @param elementBytes - bytes per element of the planner's dtype
 */
function checkRecordArguments(
    src: Binding,
    count: number,
    out: Binding,
    outOffset: number,
    elementBytes: number,
): void {
    if (!Number.isSafeInteger(count) || count < 0) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", "reduce: count must be a non-negative integer", {
            argument: "count",
            value: count,
        });
    }
    if (!Number.isSafeInteger(outOffset) || outOffset < 0) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", "reduce: outOffset must be a non-negative integer", {
            argument: "outOffset",
            value: outOffset,
        });
    }
    if (src.size < count * elementBytes) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", "reduce: src is smaller than count x element size", {
            argument: "src",
            value: src.size,
            expected: count * elementBytes,
        });
    }
    if (out.size < (outOffset + 1) * elementBytes) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", "reduce: out cannot hold the element at outOffset", {
            argument: "out",
            value: out.size,
            expected: (outOffset + 1) * elementBytes,
        });
    }
}

/** The planner: two resolved kernels (FINAL false / true) over one scope. */
class ReducePlannerImpl implements ReducePlanner {
    readonly op: ReduceOp;
    readonly dtype: ReduceDtype;
    private readonly scope: ReduceScope;
    private readonly level: Kernel;
    private readonly final: Kernel;
    private dispatches = 0;

    /**
     * Wraps the resolved kernels; use prepareReduce().
     * @param scope - the caller's scope
     * @param op - the operator
     * @param dtype - the element type
     * @param level - the per-workgroup level pipeline (FINAL = false)
     * @param final - the one-workgroup pipeline (FINAL = true)
     */
    constructor(scope: ReduceScope, op: ReduceOp, dtype: ReduceDtype, level: Kernel, final: Kernel) {
        this.scope = scope;
        this.op = op;
        this.dtype = dtype;
        this.level = level;
        this.final = final;
    }

    /**
     * Dispatches the last record() issued (2, or 3 above MAX_1D_ITEMS elements).
     * @returns the count
     */
    get lastDispatches(): number {
        return this.dispatches;
    }

    /**
     * Records the levels into the pass (see the interface).
     * @param pass - the compute pass
     * @param src - the elements (count x element bytes at least)
     * @param count - the element count
     * @param out - the output range
     * @param outOffset - the element index written
     */
    record(pass: GPUComputePassEncoder, src: Binding, count: number, out: Binding, outOffset: number): void {
        const elementBytes = ELEMENT_BYTES[this.dtype];
        checkRecordArguments(src, count, out, outOffset, elementBytes);
        const { scope } = this;
        const wg = scope.workgroupSize;
        let dispatches = 0;
        const plan1 = plan1d(count, wg, scope.caps);
        const groups1 = groupsOf(plan1);
        let partials = this.partialsBinding(Math.max(groups1, 1), "reduce/partials-1");
        this.dispatchLevel(pass, this.level, src, count, partials, 0, plan1, 1);
        dispatches += 1;
        let partialCount = groups1;
        if (groups1 > MAX_WORKGROUPS_PER_DIM) {
            const plan2 = plan1d(groups1, wg, scope.caps);
            const groups2 = groupsOf(plan2);
            const partials2 = this.partialsBinding(groups2, "reduce/partials-2");
            this.dispatchLevel(pass, this.level, partials, groups1, partials2, 0, plan2, 2);
            dispatches += 1;
            partials = partials2;
            partialCount = groups2;
        }
        this.dispatchLevel(
            pass,
            this.final,
            partials,
            partialCount,
            out,
            outOffset,
            plan1d(1, wg, scope.caps),
            dispatches + 1,
        );
        dispatches += 1;
        this.dispatches = dispatches;
    }

    /**
     * A partials scratch of `elements` elements from the scope, bound whole (never zero-length).
     * @param elements - the partial count (>= 1)
     * @param label - the scratch label
     * @returns the binding
     */
    private partialsBinding(elements: number, label: string): Binding {
        const size = elements * ELEMENT_BYTES[this.dtype];
        const buffer = this.scope.scratch(size, label);
        return { buffer, offset: 0, size, window: null };
    }

    /**
     * One level: a params record, the bind groups, one dispatch.
     * @param pass - the compute pass
     * @param kernel - the level or FINAL kernel
     * @param src - the elements of this level
     * @param count - how many
     * @param out - where the partials (or the result) go
     * @param outOffset - the element index (0 for partials)
     * @param plan - the dispatch plan
     * @param level - the level number written into the params (informational)
     */
    private dispatchLevel(
        pass: GPUComputePassEncoder,
        kernel: Kernel,
        src: Binding,
        count: number,
        out: Binding,
        outOffset: number,
        plan: DispatchPlan,
        level: number,
    ): void {
        const params = this.scope.params(REDUCE_PARAMS, { count, outOffset, level });
        const bound = kernel.bind({ src, out, P: params.binding });
        kernel.dispatch(pass, bound, plan, [params.offset]);
    }
}
