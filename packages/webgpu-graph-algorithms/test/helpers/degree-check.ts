/**
 * The degree check shared by test/algorithms/degree.test.ts and test/sabotage/degree.test.ts (spec 11.9 item 1: the
 * sabotage test asserts the SAME check fails on a mutant). Two legs:
 *
 * 1. `degreeRun`: the public `degree(ctx, s)` against `outDegreeOracle(s)`, bitwise (tolerance 0).
 * 2. `degreeWindowedRun`: the `degree` kernel dispatched by the test over arc windows of `arcsPerWindow` arcs with
 *    `arcBase != 0` and `accumulate = 1` -- the P4 windowed pattern executed early. Each window binds a COPY of its
 *    colIdx slice followed by POISON_TAIL words of INVALID_INDEX, so a read past the window (the "rebase-ignored"
 *    mutation reads `colIdx[arc]` instead of `colIdx[arc - P.arcBase]`) hits a target >= n that the bounds check
 *    refuses to count; on the real kernel the tail is never read. Rows split across windows accumulate.
 */

import { type GraphSnapshot, INVALID_INDEX, type U32 } from "@graphty/graph-format";

import { degree } from "../../src/algorithms/degree.js";
import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { kernelSpec, RANGE_PARAMS } from "../../src/kernels.js";
import { type Binding } from "../../src/types/memory.js";
import { outDegreeOracle } from "../oracle/degree.js";
import { bindingOf, readU32, scratchBuffer, uploadBuffer } from "./device.js";
import { fixture } from "./graphs.js";
import { type CheckReport, mergeReports, ratioOf } from "./sabotage.js";

/** Poison words appended to every window copy (INVALID_INDEX >= n for every snapshot). */
const POISON_TAIL = 64;

/** One arc window of the row-window leg: arcs [start, end), rows rowFirst..rowLast inclusive. */
export interface ArcWindowSpec {
    readonly start: number;
    readonly end: number;
    readonly rowFirst: number;
    readonly rowLast: number;
}

/** The result of one degree leg. */
export interface DegreeRun {
    readonly result: U32;
    readonly expected: U32;
    readonly report: CheckReport;
}

/**
 * Windows of at most `arcsPerWindow` arcs over [0, arcCount) with the rows whose arc ranges intersect each window
 * (a row longer than a window appears in several). Pure; no bitwise arithmetic on arc indices.
 * @param rowPtr - the CSR row pointers (nodeCount + 1 entries)
 * @param nodeCount - the node count
 * @param arcCount - the arc count
 * @param arcsPerWindow - the window length
 * @returns the windows in arc order (none when arcCount is 0)
 */
export function windowsOf(
    rowPtr: Uint32Array,
    nodeCount: number,
    arcCount: number,
    arcsPerWindow: number,
): ArcWindowSpec[] {
    const windows: ArcWindowSpec[] = [];
    for (let start = 0; start < arcCount; start += arcsPerWindow) {
        const end = Math.min(start + arcsPerWindow, arcCount);
        let rowFirst = 0;
        while (rowFirst < nodeCount && rowPtr[rowFirst + 1] <= start) {
            rowFirst += 1;
        }
        let rowLast = nodeCount - 1;
        while (rowLast >= 0 && rowPtr[rowLast] >= end) {
            rowLast -= 1;
        }
        windows.push({ start, end, rowFirst, rowLast });
    }
    return windows;
}

/**
 * The bitwise report of a result against its expectation: worst is Infinity on any mismatch, 0 otherwise.
 * @param actual - the GPU result
 * @param expected - the oracle
 * @param label - the report label
 * @returns the report
 */
function bitwiseReport(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): CheckReport {
    if (actual.length !== expected.length) {
        return {
            worst: Infinity,
            worstLabel: `${label}: length ${actual.length} vs ${expected.length}`,
            samples: expected.length,
        };
    }
    let worst = 0;
    let worstLabel = label;
    for (let i = 0; i < expected.length; i++) {
        const ratio = ratioOf(Math.abs(actual[i] - expected[i]), 0);
        if (ratio > worst) {
            worst = ratio;
            worstLabel = `${label}[${i}]: ${actual[i]} vs ${expected[i]}`;
        }
    }
    return { worst, worstLabel, samples: expected.length };
}

/**
 * The public path: degree(ctx, s) vs outDegreeOracle(s).
 * @param ctx - the context
 * @param s - the snapshot
 * @param label - the report label
 * @returns the result, the oracle and the report
 */
export async function degreeRun(ctx: GpuContext, s: GraphSnapshot, label: string): Promise<DegreeRun> {
    const result = await degree(ctx, s);
    const expected = outDegreeOracle(s);
    return { result, expected, report: bitwiseReport(result, expected, label) };
}

/**
 * The row-window leg: one `degree` dispatch per window with arcBase = start, arcEnd = end, accumulate = 1, over the
 * rows rowFirst..rowLast, all recorded into one pass; the accumulated out equals the oracle.
 * @param ctx - the context
 * @param s - the snapshot
 * @param arcsPerWindow - arcs per window (64 in the tests)
 * @param label - the report label
 * @returns the result, the oracle and the report
 */
export async function degreeWindowedRun(
    ctx: GpuContext,
    s: GraphSnapshot,
    arcsPerWindow: number,
    label: string,
): Promise<DegreeRun> {
    const n = s.nodeCount;
    const expected = outDegreeOracle(s);
    if (n === 0) {
        return { result: new Uint32Array(0), expected, report: { worst: 0, worstLabel: label, samples: 0 } };
    }
    const kernel = await ctx.pipelines.kernel(kernelSpec("degree", { USE_PERM: false, HAS_WEIGHTS: false }));
    const rowPtrBuf = uploadBuffer(ctx, s.rowPtr, `${label}/rowPtr`);
    const out = scratchBuffer(ctx, n * 4, `${label}/out`);
    const owned: GPUBuffer[] = [];
    const pooled: GPUBuffer[] = [];
    try {
        const encoder = ctx.device.createCommandEncoder({ label });
        const pass = encoder.beginComputePass({ label });
        for (const w of windowsOf(s.rowPtr, n, s.arcCount, arcsPerWindow)) {
            const len = w.end - w.start;
            const words = new Uint32Array(len + POISON_TAIL);
            words.set(s.colIdx.subarray(w.start, w.end));
            words.fill(INVALID_INDEX, len);
            const win = uploadBuffer(ctx, words, `${label}/colIdx[${w.start},${w.end})`);
            owned.push(win);
            const params = ctx.pool.acquire(
                RANGE_PARAMS.byteLength,
                BufferUsage.UNIFORM | BufferUsage.COPY_DST,
                `${label}/params`,
            );
            pooled.push(params);
            const bytes = new ArrayBuffer(RANGE_PARAMS.byteLength);
            RANGE_PARAMS.write(new DataView(bytes), {
                start: w.rowFirst,
                end: w.rowLast + 1,
                arcBase: w.start,
                arcEnd: w.end,
                accumulate: 1,
                n,
            });
            ctx.device.queue.writeBuffer(params, 0, bytes);
            const paramsBinding: Binding = { buffer: params, offset: 0, size: RANGE_PARAMS.byteLength, window: null };
            const bound = kernel.bind({
                rowPtr: bindingOf(rowPtrBuf),
                colIdx: bindingOf(win),
                weights: bindingOf(win),
                perm: bindingOf(rowPtrBuf),
                out: bindingOf(out),
                P: paramsBinding,
            });
            kernel.dispatch(pass, bound, plan1d(w.rowLast - w.rowFirst + 1, ctx.workgroupSize, ctx.caps), [0]);
        }
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        ctx.assertReady();
        const result = await readU32(ctx, out, n);
        return { result, expected, report: bitwiseReport(result, expected, label) };
    } finally {
        for (const b of pooled) {
            ctx.pool.release(b);
        }
        for (const b of owned) {
            b.destroy();
        }
        rowPtrBuf.destroy();
        out.destroy();
    }
}

/**
 * The check the sabotage rows of `degree` must break: both legs over the undirected karate club (node 33 has
 * degree 17, so a skipped last row is visible; 156 arcs make three windows of 64 with a row split at 63 / 64).
 * @param ctx - a fresh context (mutant or not)
 * @returns the merged report
 */
export async function degreeSabotageReport(ctx: GpuContext): Promise<CheckReport> {
    const { snapshot } = fixture("karate", 1);
    const expected = outDegreeOracle(snapshot);
    if (expected[snapshot.nodeCount - 1] === 0) {
        throw new Error("degree sabotage precondition: the last node must have a non-zero degree");
    }
    const direct = await degreeRun(ctx, snapshot, "degree/karate");
    const windowed = await degreeWindowedRun(ctx, snapshot, 64, "degree-windowed/karate");
    return mergeReports([direct.report, windowed.report]);
}
