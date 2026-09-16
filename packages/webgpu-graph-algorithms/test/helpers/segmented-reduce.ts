/**
 * The run driver and the check set of the thread-per-row segmentedReduce tests, shared by
 * test/primitives/segmented-reduce.test.ts, test/sabotage/segmented-reduce.test.ts, test/kernel/determinism.test.ts
 * and the browser leg of test/browser/compile-matrix.test.ts. IMPORT RULE: Vite bundles this file for Chromium, so
 * it imports only @graphty/graph-format types, src/** modules outside src/node/**, test/oracle/**, and the pure
 * helpers graphs.ts / matchers.ts / override-matrix.ts -- never test/helpers/device.ts (its withContext pulls in
 * test/setup/gpu.ts: scripts/gpu-policy.js, src/node/index.js = the native `webgpu` module, node:fs, process.env),
 * never test/setup/**, never node:*. The two device.ts conveniences it needs (a whole-buffer Binding, an f32
 * readback) are written here on ctx.readback (contract 3.8) instead. The ReduceScope is built here the way P1-T5's
 * reduce test built it (a fresh pool-acquired uniform buffer per params record): this task is parallel with P2-T1
 * and never uses CommandBatch / UniformRing / Lease.
 */

import { type F32, type GraphSnapshot } from "@graphty/graph-format";

import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { type UniformBlock, type UniformValues } from "../../src/kernel/struct-block.js";
import { type ReduceOp, type ReduceScope } from "../../src/primitives/reduce.js";
import { prepareSegmentedReduce } from "../../src/primitives/segmented-reduce.js";
import { type Binding } from "../../src/types/memory.js";
import { segmentedReduceOracle } from "../oracle/segmented-reduce.js";
import { type EdgeSpec, randomEdges, snapshotOf } from "./graphs.js";
import { maxRelError } from "./matchers.js";
import { SEGMENTED_REDUCE_SNIPPETS } from "./override-matrix.js";

/**
 * Whole-buffer Binding (the device.ts helper of the same name, repeated here under the import rule above).
 * @param buffer - the buffer
 * @returns the binding over [0, buffer.size)
 */
function bindingOf(buffer: GPUBuffer): Binding {
    return { buffer, offset: 0, size: buffer.size, window: null };
}

/**
 * Reads `count` f32 values from the start of a COPY_SRC buffer through the context's staging ring.
 * @param ctx - the context
 * @param buffer - the buffer
 * @param count - values to read (> 0)
 * @returns a fresh Float32Array over the copied bytes
 */
async function readF32(ctx: GpuContext, buffer: GPUBuffer, count: number): Promise<F32> {
    const bytes = await ctx.readback.read(buffer, 4 * count);
    return new Float32Array(bytes);
}

/** The value every out slot holds before a dispatch: a row the kernel never writes keeps it and misses the oracle by ~1e30. */
const SR_SENTINEL = -1e30;
/** The absolute floor of maxRelError in these tests (rows whose expected value is 0 -- empty rows under sum -- divide by it). */
export const SR_ABS_FLOOR = 1e-12;
/** The smallest tolerance the sabotage factor divides by (min / max are exact, so their analytic tolerance is 0). */
const SR_FACTOR_FLOOR = 2 ** -24;

/**
 * A ReduceScope over a context whose params slots and scratch come from the pool and go back on dispose(). The
 * return type of testReduceScope (knip: exported for the signature, not imported by name).
 * @public
 */
export interface TestReduceScope extends ReduceScope {
    dispose(): void;
}

/**
 * The scope the primitives take (contract 3.11 ReduceScope) over a context.
 * @param ctx - the context
 * @returns a scope; dispose() releases every buffer it acquired
 */
export function testReduceScope(ctx: GpuContext): TestReduceScope {
    const owned: GPUBuffer[] = [];
    return {
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
            owned.push(buffer);
            return buffer;
        },
        params(block: UniformBlock, values: UniformValues): { readonly binding: Binding; readonly offset: number } {
            const buffer = ctx.pool.acquire(
                block.byteLength,
                BufferUsage.UNIFORM | BufferUsage.COPY_DST,
                `${block.name}/params`,
            );
            owned.push(buffer);
            const bytes = new ArrayBuffer(block.byteLength);
            block.write(new DataView(bytes), values);
            ctx.device.queue.writeBuffer(buffer, 0, bytes);
            return { binding: { buffer, offset: 0, size: block.byteLength, window: null }, offset: 0 };
        },
        dispose(): void {
            for (const buffer of owned.splice(0)) {
                ctx.pool.release(buffer);
            }
        },
    };
}

/**
 * The oracle callback of a snippet of SEGMENTED_REDUCE_SNIPPETS.
 * @param snippet - the WGSL snippet text
 * @returns the TypeScript value callback (which keeps `target`, contract 5.3)
 */
export function oracleValueOf(snippet: string): (row: number, arc: number, target: number, weight: number) => number {
    if (snippet === SEGMENTED_REDUCE_SNIPPETS.weight || snippet === SEGMENTED_REDUCE_SNIPPETS.commented) {
        return (_row: number, _arc: number, _target: number, weight: number): number => weight;
    }
    if (snippet === SEGMENTED_REDUCE_SNIPPETS.one) {
        return (): number => 1;
    }
    if (snippet === SEGMENTED_REDUCE_SNIPPETS.conditional) {
        // `if (nbr < row) { v = weight; } else { v = 0.0; }`: the weight of every arc into a lower-numbered row
        return (row: number, _arc: number, target: number, weight: number): number => (target < row ? weight : 0);
    }
    throw new Error(`no oracle callback for the snippet ${JSON.stringify(snippet)}`);
}

/**
 * The analytic tolerance of the f32 sequential fold against the f64 oracle: for sum, each of the at most dmax - 1
 * additions of non-negative terms rounds once (2^-24 relative), so 2 x dmax x 2^-24 covers the row with the largest
 * degree with a 2x margin; min / max are exact (0).
 * @param s - the snapshot
 * @param op - the operator
 * @returns the relative tolerance
 */
export function relTolerance(s: GraphSnapshot, op: ReduceOp): number {
    if (op !== "sum") {
        return 0;
    }
    const { rowPtr } = s;
    let dmax = 1;
    for (let row = 0; row < s.nodeCount; row++) {
        dmax = Math.max(dmax, rowPtr[row + 1] - rowPtr[row]);
    }
    return 2 * dmax * 2 ** -24;
}

/**
 * max_i |a_i - b_i|.
 * @param a - one array
 * @param b - the other
 * @returns the largest absolute difference (NaN if any pair is NaN)
 */
export function maxAbsError(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let worst = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        if (Number.isNaN(d)) {
            return Number.NaN;
        }
        worst = Math.max(worst, d);
    }
    return worst;
}

/**
 * maxRelError / tolerance: the "factor" of spec 11.9 (the real kernel is below 1, a mutant must reach minFactor).
 * NaN counts as infinite.
 * @param actual - the GPU result
 * @param expected - the oracle
 * @param s - the snapshot (for the tolerance)
 * @param op - the operator
 * @returns the factor
 */
function errorFactor(actual: ArrayLike<number>, expected: ArrayLike<number>, s: GraphSnapshot, op: ReduceOp): number {
    const err = maxRelError(actual, expected, SR_ABS_FLOOR);
    const tolerance = Math.max(relTolerance(s, op), SR_FACTOR_FLOOR);
    return Number.isNaN(err) ? Number.POSITIVE_INFINITY : err / tolerance;
}

/**
 * Options of runSegmentedReduce: the parameter type (knip: exported for the signature, not imported by name).
 * @public
 */
export interface RunOptions {
    readonly accumulate?: boolean | undefined;
    /** The out contents before the dispatch (default: every row SR_SENTINEL). */
    readonly initial?: F32 | undefined;
}

/**
 * Uploads the core, prepares the thread-per-row planner, records ONE dispatch into a fresh encoder, submits and reads
 * out[0..n) back. The out buffer starts as SR_SENTINEL so an unwritten row is visible. The snapshot stays resident
 * (the caller releases it).
 * @param ctx - the context
 * @param s - the snapshot
 * @param op - the operator
 * @param snippet - the VALUE snippet
 * @param options - accumulate / initial
 * @returns the n results
 */
export async function runSegmentedReduce(
    ctx: GpuContext,
    s: GraphSnapshot,
    op: ReduceOp,
    snippet: string,
    options?: RunOptions,
): Promise<F32> {
    const n = s.nodeCount;
    const core = ctx.residency.core(s);
    const scope = testReduceScope(ctx);
    try {
        const out = scope.scratch(Math.max(4, 4 * n), "segmented-reduce/out");
        if (n > 0) {
            const initial = options?.initial ?? new Float32Array(n).fill(SR_SENTINEL);
            ctx.device.queue.writeBuffer(out, 0, initial);
        }
        const planner = await prepareSegmentedReduce(scope, core, {
            op,
            valueSnippet: snippet,
            tiers: null,
            accumulate: options?.accumulate,
        });
        const encoder = ctx.device.createCommandEncoder({ label: "segmented-reduce/test" });
        const pass = encoder.beginComputePass({ label: "segmented-reduce/test" });
        planner.record(pass, core, bindingOf(out));
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        if (n === 0) {
            return new Float32Array(0);
        }
        return await readF32(ctx, out, n);
    } finally {
        scope.dispose();
    }
}

/**
 * A seeded G(n, m) without self-loops or parallels carrying f32 weights in [0.25, 4) from an LCG over the edge
 * index, over `nodeCount` rows (rows >= n have no arcs).
 * @param n - nodes with edges
 * @param m - edges (0 when n < 2)
 * @param seed - the generator seed
 * @param nodeCount - the row count (default n)
 * @returns an undirected weighted snapshot
 */
export function weightedRandom(n: number, m: number, seed: number, nodeCount?: number): GraphSnapshot {
    const edges: EdgeSpec[] = n >= 2 ? randomEdges(n, m, seed) : [];
    let state = seed % 4294967296;
    const weighted: EdgeSpec[] = edges.map(([u, v]): EdgeSpec => {
        state = (state * 1664525 + 1013904223) % 4294967296;
        return [u, v, Math.fround(0.25 + 3.75 * (state / 4294967296))];
    });
    return snapshotOf(weighted, {
        weighted: true,
        nodeCount: nodeCount ?? n,
        label: `weighted-random-${n}-${m}-${seed}`,
    });
}

/**
 * One check of the sabotage set: the element type of sabotageChecks() and the parameter type of worstFactor (knip:
 * exported for the signatures, not imported by name).
 * @public
 */
export interface SrCheck {
    readonly name: string;
    readonly snapshot: GraphSnapshot;
    readonly op: ReduceOp;
    readonly snippet: string;
}

/**
 * The check set both the primitive test and the sabotage test run: the weighted random1k sum and min (weights are
 * not 1, so a dropped weight read is a ~50% error and a zeroed min identity a 100% one), and a 300-row graph whose
 * last 100 rows are empty under sum and max (a skipped empty row keeps the sentinel; the max identity is -F32_MAX).
 * @returns the checks
 */
export function sabotageChecks(): readonly SrCheck[] {
    const random1k = weightedRandom(1000, 5000, 7);
    const holes = weightedRandom(200, 600, 5, 300);
    return [
        { name: "random1k weighted sum", snapshot: random1k, op: "sum", snippet: SEGMENTED_REDUCE_SNIPPETS.weight },
        { name: "random1k weighted min", snapshot: random1k, op: "min", snippet: SEGMENTED_REDUCE_SNIPPETS.weight },
        {
            name: "300 rows / 100 empty, weighted sum",
            snapshot: holes,
            op: "sum",
            snippet: SEGMENTED_REDUCE_SNIPPETS.weight,
        },
        {
            name: "300 rows / 100 empty, weighted max",
            snapshot: holes,
            op: "max",
            snippet: SEGMENTED_REDUCE_SNIPPETS.weight,
        },
    ];
}

/**
 * The worst error factor over the checks (each run once; the snapshots are released afterwards).
 * @param ctx - the context (a fresh one per mutation under withSabotage)
 * @param checks - the checks
 * @returns max over checks of errorFactor
 */
export async function worstFactor(ctx: GpuContext, checks: readonly SrCheck[]): Promise<number> {
    let worst = 0;
    for (const check of checks) {
        const actual = await runSegmentedReduce(ctx, check.snapshot, check.op, check.snippet);
        const expected = segmentedReduceOracle(check.snapshot, oracleValueOf(check.snippet), check.op);
        worst = Math.max(worst, errorFactor(actual, expected, check.snapshot, check.op));
    }
    for (const check of checks) {
        ctx.release(check.snapshot);
    }
    return worst;
}
