/**
 * The CPU reference of segmentedReduce (contract 5.3; spec 6 row 3): a per-row sequential fold, in f64, of
 * value(row, arc, target, weight) over the CSR rows in arc order. The TypeScript callback keeps `target`; only the
 * WGSL snippet vocabulary says `nbr` (contract 4.5, `target` being a WGSL reserved word). An empty row yields the
 * kernel's identity element (0, F32_MAX, -F32_MAX), never +-Infinity, so a GPU result compares equal to the oracle
 * on it.
 */

import { type GraphSnapshot } from "@graphty/graph-format";

import { type ReduceOp } from "../../src/primitives/reduce.js";

/** The largest finite f32, 0x1.fffffep+127: the prelude's F32_MAX (contract 4.1) and the kernel's `min` identity. */
export const F32_MAX = 3.4028234663852886e38;

/**
 * The identity element of an operator, as the kernel's `identity()` defines it (contract 4.5).
 * @param op - the operator
 * @returns 0 for sum, F32_MAX for min, -F32_MAX for max
 */
function identityOf(op: ReduceOp): number {
    switch (op) {
        case "sum":
            return 0;
        case "min":
            return F32_MAX;
        case "max":
            return -F32_MAX;
        default:
            throw new Error(`segmentedReduceOracle: unknown op ${String(op)}`);
    }
}

/**
 * One fold step in f64.
 * @param acc - the accumulator
 * @param v - the next value
 * @param op - the operator
 * @returns the combined value
 */
function combine(acc: number, v: number, op: ReduceOp): number {
    switch (op) {
        case "sum":
            return acc + v;
        case "min":
            return Math.min(acc, v);
        case "max":
            return Math.max(acc, v);
        default:
            throw new Error(`segmentedReduceOracle: unknown op ${String(op)}`);
    }
}

/**
 * Per-row f64 reduction of value(row, arc, target, weight) over the CSR rows (the TypeScript callback keeps
 * `target`; only the WGSL snippet vocabulary says `nbr`, 4.5).
 * @param s - the snapshot whose rowPtr / colIdx / weights are walked (weight is 1 when the snapshot is unweighted)
 * @param value - the per-arc value
 * @param op - the operator
 * @returns one f64 per row; an empty row holds the identity element
 */
export function segmentedReduceOracle(
    s: GraphSnapshot,
    value: (row: number, arc: number, target: number, weight: number) => number,
    op: ReduceOp,
): Float64Array {
    const n = s.nodeCount;
    const { rowPtr, colIdx, weights } = s; // prefer-destructuring (object: true) is on for test/** too
    const out = new Float64Array(n);
    for (let row = 0; row < n; row++) {
        let acc = identityOf(op);
        const a0 = rowPtr[row];
        const a1 = rowPtr[row + 1];
        for (let arc = a0; arc < a1; arc++) {
            const weight = weights === null ? 1 : weights[arc];
            acc = combine(acc, value(row, arc, colIdx[arc], weight), op);
        }
        out[row] = acc;
    }
    return out;
}
