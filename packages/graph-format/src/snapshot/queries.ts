/**
 * The row-ordering queries of design section 3.9 as pure functions over the core arrays: binary
 * searches over sorted rows (invariant I4) for `findArc` / `hasArc` / `arcsBetween` /
 * `multiplicity` / `selfLoopsAt`, and the binary search over `rowPtr` behind `arcSource`. Every
 * function is total for in-range arguments and unchecked for out-of-range ones (design section
 * 11.1: typed-array semantics, no bounds checks in hot paths). GraphSnapshot's methods of the same
 * names delegate here; the views and the derived graphs reuse the searches directly.
 *
 * Multigraph semantics (design section 3.5): parallel arcs are adjacent and ordered by logical edge
 * index, so the first arc of `[lo, hi)` is the lowest logical edge index among the parallels.
 */

import { INVALID_INDEX } from "../constants.js";
import { type U32 } from "../types/index.js";

/**
 * The first index in [lo, hi) whose value is >= v, or hi when none is (the sorted-row lower bound).
 * @param colIdx - the sorted targets
 * @param lo - the start of the row
 * @param hi - one past the end of the row
 * @param v - the target to search for
 * @returns the lower bound
 */
export function lowerBound(colIdx: U32, lo: number, hi: number, v: number): number {
    let low = lo;
    let high = hi;
    while (low < high) {
        const mid = low + Math.floor((high - low) / 2);
        if (colIdx[mid] < v) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

/**
 * The first index in [lo, hi) whose value is > v, or hi when none is (the sorted-row upper bound).
 * @param colIdx - the sorted targets
 * @param lo - the start of the row
 * @param hi - one past the end of the row
 * @param v - the target to search for
 * @returns the upper bound
 */
export function upperBound(colIdx: U32, lo: number, hi: number, v: number): number {
    let low = lo;
    let high = hi;
    while (low < high) {
        const mid = low + Math.floor((high - low) / 2);
        if (colIdx[mid] <= v) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

/**
 * The first arc u -> v (the lowest logical edge index among parallels), or INVALID_INDEX.
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param u - the source node index
 * @param v - the target node index
 * @returns the arc index or INVALID_INDEX
 */
export function findArcIn(rowPtr: U32, colIdx: U32, u: number, v: number): number {
    const hi = rowPtr[u + 1];
    const at = lowerBound(colIdx, rowPtr[u], hi, v);
    return at < hi && colIdx[at] === v ? at : INVALID_INDEX;
}

/**
 * The half-open arc range [lo, hi) of every arc u -> v; empty (lo === hi) when there is none.
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param u - the source node index
 * @param v - the target node index
 * @returns the range as a fresh two-element tuple
 */
export function arcRangeIn(rowPtr: U32, colIdx: U32, u: number, v: number): [lo: number, hi: number] {
    const start = rowPtr[u];
    const end = rowPtr[u + 1];
    const lo = lowerBound(colIdx, start, end, v);
    const hi = upperBound(colIdx, lo, end, v);
    return [lo, hi];
}

/**
 * The number of parallel arcs u -> v.
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param u - the source node index
 * @param v - the target node index
 * @returns hi - lo of the arc range
 */
export function multiplicityIn(rowPtr: U32, colIdx: U32, u: number, v: number): number {
    const start = rowPtr[u];
    const end = rowPtr[u + 1];
    const lo = lowerBound(colIdx, start, end, v);
    return upperBound(colIdx, lo, end, v) - lo;
}

/**
 * The row containing an arc: the largest u with rowPtr[u] <= a < rowPtr[u + 1], found by binary
 * search over rowPtr (empty rows are skipped because their two offsets are equal).
 * @param rowPtr - the row offsets, nodeCount + 1 entries
 * @param a - the arc index, below rowPtr[nodeCount]
 * @returns the source node index
 */
export function arcSourceIn(rowPtr: U32, a: number): number {
    // first index i with rowPtr[i] > a; the row is i - 1
    let low = 0;
    let high = rowPtr.length;
    while (low < high) {
        const mid = low + Math.floor((high - low) / 2);
        if (rowPtr[mid] <= a) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}

/**
 * The number of self-loop arcs at a node: the multiplicity of u -> u.
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param u - the node index
 * @returns the loop count
 */
export function selfLoopsAtIn(rowPtr: U32, colIdx: U32, u: number): number {
    return multiplicityIn(rowPtr, colIdx, u, u);
}
