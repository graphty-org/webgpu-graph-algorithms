/**
 * The reduce check shared by test/primitives/reduce.test.ts and test/sabotage/reduce.test.ts (spec 11.9 item 1), and
 * the P1 ReduceScope over a GpuContext (P2's CommandBatch supplies the same record with a Lease; until then the tests
 * hold the scratch and release it in a finally).
 *
 * Inputs come from test/helpers/reduce-input.ts (the pure, import-free recipe the browser skeleton and the noise-floor
 * test share): f32 / vec4f values in [1, 2), u32 values in [1, 127]. Every uploaded input carries PAD_ELEMENTS poison elements past `count`
 * (sum / max: +2^20 f32 / 2^30 u32; min: -2^20 f32 / 0 u32), so a read past the range changes the result -- the
 * "level-bound-inclusive" mutation is observable, and the real kernel never touches the tail. The output region is
 * pre-filled with OUT_POISON so a missing write (the "final-writes-out-zero" mutation with outOffset != 0) is visible
 * and every word the FINAL level must not touch can be checked.
 */

import { type U32 } from "@graphty/graph-format";

import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { type UniformBlock, type UniformValues } from "../../src/kernel/struct-block.js";
import {
    prepareReduce,
    type ReduceDtype,
    type ReduceOp,
    type ReducePlanner,
    type ReduceScope,
} from "../../src/primitives/reduce.js";
import { type Binding } from "../../src/types/memory.js";
import { reduceOracle } from "../oracle/reduce.js";
import { bindingOf, readU32, scratchBuffer, uploadBuffer } from "./device.js";
import { INPUT_SEED, lanesOf, reduceInput } from "./reduce-input.js";
import { type CheckReport, mergeReports, ratioOf } from "./sabotage.js";

/** One reduce case of the check. */
export interface ReduceCase {
    readonly op: ReduceOp;
    readonly dtype: ReduceDtype;
    readonly count: number;
    readonly outOffset: number;
}

/** The poison word the output region is pre-filled with. */
export const OUT_POISON = 0xdeadbeef;
/** Poison elements appended past `count` of every uploaded input. */
const PAD_ELEMENTS = 64;

/** The cases the sabotage test runs: together they touch every mutated line (PLAN DECISION 10). */
export const SABOTAGE_REDUCE_CASES: readonly ReduceCase[] = Object.freeze([
    { op: "sum", dtype: "f32", count: 257, outOffset: 3 },
    { op: "sum", dtype: "u32", count: 255, outOffset: 1 },
    { op: "min", dtype: "u32", count: 257, outOffset: 3 },
    { op: "min", dtype: "f32", count: 257, outOffset: 3 },
    { op: "max", dtype: "vec4f", count: 257, outOffset: 3 },
    { op: "min", dtype: "vec4f", count: 257, outOffset: 3 },
]);

/**
 * The poison element past the range for an op: above every input for sum / max, below it for min.
 * @param op - the operator
 * @param dtype - the element type
 * @returns the poison value
 */
function poisonFor(op: ReduceOp, dtype: ReduceDtype): number {
    if (dtype === "u32") {
        return op === "min" ? 0 : 1073741824;
    }
    return op === "min" ? -1048576 : 1048576;
}

/**
 * The words to upload for an input: the values followed by PAD_ELEMENTS poison elements.
 * @param values - reduceInput()
 * @param dtype - the element type
 * @param op - the operator (selects the poison)
 * @returns the u32 words (f32 bit patterns for f32 / vec4f)
 */
export function paddedInput(
    values: Float32Array | Uint32Array,
    dtype: ReduceDtype,
    op: ReduceOp,
): Uint32Array<ArrayBuffer> {
    const poison = poisonFor(op, dtype);
    if (dtype === "u32") {
        const words = new Uint32Array(values.length + PAD_ELEMENTS);
        words.set(values);
        words.fill(poison, values.length);
        return words;
    }
    const floats = new Float32Array(values.length + PAD_ELEMENTS * lanesOf(dtype));
    floats.set(values);
    floats.fill(poison, values.length);
    return new Uint32Array(floats.buffer);
}

/** A ReduceScope over a context plus the release of everything it handed out. */
export interface TestReduceScope {
    readonly scope: ReduceScope;
    release(): void;
}

/**
 * The P1 ReduceScope: scratch from ctx.pool, params written into a fresh pool uniform buffer per record (dynamic
 * offset 0), everything released by release().
 * @param ctx - the context
 * @returns the scope and its release
 */
export function reduceScopeOf(ctx: GpuContext): TestReduceScope {
    const held: GPUBuffer[] = [];
    const scope: ReduceScope = {
        device: ctx.device,
        caps: ctx.caps,
        pipelines: ctx.pipelines,
        pool: ctx.pool,
        workgroupSize: ctx.workgroupSize,
        scratch(byteLength: number, label: string): GPUBuffer {
            const buffer = ctx.pool.acquire(
                byteLength,
                BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
                label,
            );
            held.push(buffer);
            return buffer;
        },
        params(block: UniformBlock, values: UniformValues): { readonly binding: Binding; readonly offset: number } {
            const buffer = ctx.pool.acquire(
                block.byteLength,
                BufferUsage.UNIFORM | BufferUsage.COPY_DST,
                `${block.name}/params`,
            );
            held.push(buffer);
            const bytes = new ArrayBuffer(block.byteLength);
            block.write(new DataView(bytes), values);
            ctx.device.queue.writeBuffer(buffer, 0, bytes);
            return { binding: { buffer, offset: 0, size: block.byteLength, window: null }, offset: 0 };
        },
    };
    return {
        scope,
        release(): void {
            for (const buffer of held.splice(0, held.length)) {
                ctx.pool.release(buffer);
            }
        },
    };
}

/** One recorded, submitted and read-back reduce (`words` is a U32 over a plain ArrayBuffer so it can go straight into expectBitwiseEqual). */
export interface ReduceRun {
    readonly value: number | readonly [number, number, number, number];
    readonly words: U32;
    readonly dispatches: number;
}

/**
 * The result element at outOffset of the read-back words.
 * @param words - the output region
 * @param dtype - the element type
 * @param outOffset - the element index
 * @returns the scalar, or the four lanes
 */
function reduceValueOf(
    words: Uint32Array,
    dtype: ReduceDtype,
    outOffset: number,
): number | readonly [number, number, number, number] {
    if (dtype === "u32") {
        return words[outOffset];
    }
    const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    if (dtype === "f32") {
        return floats[outOffset];
    }
    return [floats[4 * outOffset], floats[4 * outOffset + 1], floats[4 * outOffset + 2], floats[4 * outOffset + 3]];
}

/**
 * Records one reduce into a fresh pass over an uploaded src, submits, reads the poisoned output region back.
 * @param ctx - the context
 * @param op - the operator
 * @param dtype - the element type
 * @param src - the uploaded input (paddedInput)
 * @param count - the element count
 * @param outOffset - the output element index
 * @returns the value, every word of the output region, and the planner's lastDispatches
 */
export async function runReduce(
    ctx: GpuContext,
    op: ReduceOp,
    dtype: ReduceDtype,
    src: GPUBuffer,
    count: number,
    outOffset: number,
): Promise<ReduceRun> {
    const outWords = lanesOf(dtype) * (outOffset + 2);
    const out = scratchBuffer(ctx, outWords * 4, "reduce-check/out");
    ctx.device.queue.writeBuffer(out, 0, new Uint32Array(outWords).fill(OUT_POISON));
    const held = reduceScopeOf(ctx);
    try {
        const planner: ReducePlanner = await prepareReduce(held.scope, op, dtype);
        const encoder = ctx.device.createCommandEncoder({ label: "reduce-check" });
        const pass = encoder.beginComputePass({ label: "reduce-check" });
        planner.record(pass, bindingOf(src), count, bindingOf(out), outOffset);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        ctx.assertReady();
        const words = await readU32(ctx, out, outWords);
        return { value: reduceValueOf(words, dtype, outOffset), words, dispatches: planner.lastDispatches };
    } finally {
        held.release();
        out.destroy();
    }
}

/**
 * The tolerance of a case (relative): count x 2^-24 for an f32 / vec4f sum, 0 (exact) otherwise.
 * @param op - the operator
 * @param dtype - the element type
 * @param count - the element count
 * @returns the relative tolerance
 */
function reduceTolerance(op: ReduceOp, dtype: ReduceDtype, count: number): number {
    if (op === "sum" && dtype !== "u32") {
        return count * 2 ** -24;
    }
    return 0;
}

/**
 * The relative error of a result against the oracle (max over lanes), |e| floored at 1 so identities compare
 * exactly and sums (>= 1 for these inputs) compare relatively.
 * @param actual - the GPU value
 * @param expected - the oracle value
 * @returns the error
 */
function reduceError(
    actual: number | readonly [number, number, number, number],
    expected: number | readonly [number, number, number, number],
): number {
    const a: readonly number[] = typeof actual === "number" ? [actual] : actual;
    const e: readonly number[] = typeof expected === "number" ? [expected] : expected;
    if (a.length !== e.length) {
        return Infinity;
    }
    let worst = 0;
    for (let i = 0; i < e.length; i++) {
        const err = Math.abs(a[i] - e[i]) / Math.max(Math.abs(e[i]), 1);
        worst = Number.isNaN(err) ? Infinity : Math.max(worst, err);
    }
    return worst;
}

/**
 * A readable case label.
 * @param c - the case
 * @returns "op/dtype/count@outOffset"
 */
export function caseLabel(c: ReduceCase): string {
    return `${c.op}/${c.dtype}/${c.count}@${c.outOffset}`;
}

/**
 * Runs one case against an uploaded src and reports its error / tolerance ratio.
 * @param ctx - the context
 * @param c - the case
 * @param src - the uploaded paddedInput of the case
 * @param values - the case's reduceInput
 * @returns the report and the run
 */
export async function reduceCaseReport(
    ctx: GpuContext,
    c: ReduceCase,
    src: GPUBuffer,
    values: Float32Array | Uint32Array,
): Promise<{ readonly report: CheckReport; readonly run: ReduceRun }> {
    const run = await runReduce(ctx, c.op, c.dtype, src, c.count, c.outOffset);
    const expected = reduceOracle(values, c.op, c.dtype);
    const worst = ratioOf(reduceError(run.value, expected), reduceTolerance(c.op, c.dtype, c.count));
    return {
        report: { worst, worstLabel: `${caseLabel(c)}: ${String(run.value)} vs ${String(expected)}`, samples: 1 },
        run,
    };
}

/**
 * The check the sabotage rows of `reduce` must break: SABOTAGE_REDUCE_CASES on a fresh context.
 * @param ctx - the context (mutant or not)
 * @returns the merged report
 */
export async function reduceSabotageReport(ctx: GpuContext): Promise<CheckReport> {
    const reports: CheckReport[] = [];
    for (const c of SABOTAGE_REDUCE_CASES) {
        const values = reduceInput(c.dtype, c.count, INPUT_SEED + c.count);
        const src = uploadBuffer(ctx, paddedInput(values, c.dtype, c.op), `reduce-sabotage/${caseLabel(c)}`);
        try {
            reports.push((await reduceCaseReport(ctx, c, src, values)).report);
        } finally {
            src.destroy();
        }
    }
    return mergeReports(reports);
}
