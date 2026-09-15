/**
 * The two stable counting-sort passes of the freeze pipeline (design section 6.3 steps 3-6, 8): arc
 * materialisation (doubled storage for undirected graphs, self-loops once, invariant I7), the
 * identity check for directed input that is already grouped by source and sorted by target (step 4),
 * pass 1 by target, pass 2 by source into rowPtr with the scatter writing colIdx / arcToEdge /
 * edgeToArc straight into the target arrays (the arena, for a "keep" freeze), and the flag
 * predicates of design section 3.8 computed on the way (I9). There is no comparator sort anywhere;
 * the cost is O(n + A).
 *
 * Because pass 2 is stable and consumes arcs in pass-1 order, every row ends sorted by target with
 * parallel arcs in ascending logical-edge order (I4), and the k-th `u -> v` arc in row u and the
 * k-th `v -> u` arc in row v come from the same edge (design section 6.4).
 *
 * Layout of the passes, chosen from the memory profile of the 100k-node / 1M-edge benchmark (the
 * scatters are cache-miss bound, so the rule is: as few random streams per loop as possible, and no
 * random LOAD whose value a branch depends on inside a scatter):
 *
 * - One sequential pass over the edges counts arcs per target and per source (both prefix sums come
 *   out of it), counts the self-loops, and scans the weights (NaN refusal, I8; the min / max of the
 *   stored f32 values, from which the three weight flags follow, I9). Every edge lands on at least
 *   one arc, so predicates over the edges equal predicates over the arcs.
 * - Pass 1 scatters (source, edge) PAIRS into target order, one 8-byte write per arc into a single
 *   `byTarget` transient of 2A words; for an undirected graph a bitmap over the pass-1 positions
 *   marks the arc whose stored orientation is the declared one (the mirror arc of an edge is
 *   emitted right after it, so arcs stay in edge order within a bucket).
 * - Pass 2 reads `byTarget` sequentially bucket by bucket and scatters ONLY colIdx and arcToEdge (and
 *   edgeToArc when undirected, where the declared bit is at hand).
 * - Sequential post-passes derive edgeToArc from arcToEdge (directed), gather the arc weights
 *   through arcToEdge, and test adjacent targets within each row for the multigraph flag.
 *
 * Every loop lives in its own small function: V8 optimises a hot loop by on-stack replacement of the
 * function that contains it, and a function invoked once per freeze with a dozen loops in it is
 * re-entered through stale OSR code that deoptimises at every loop exit.
 *
 * Transients of the non-identity path: 2 x U32(n + 1), U32(2A), and for an undirected graph a bitmap
 * of A bits.
 */

import { GraphFormatError } from "../errors.js";
import { type F32, type F64, type SnapshotFlags, type U32 } from "../types/index.js";
import { allocateCoreArrays, type CoreArrays } from "./arena.js";

/** The per-logical-edge input of the sort: views over the (compacted) staging arrays. */
export interface SortInput {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** The node count n; every endpoint is below it. */
    readonly nodeCount: number;
    /** The logical edge count E; only the first E entries of src / dst / weights are read. */
    readonly edgeCount: number;
    /** Declared source of every edge. */
    readonly src: U32;
    /** Declared target of every edge. */
    readonly dst: U32;
    /** Per-edge staging weights (f32 or f64), or null when unweighted. */
    readonly weights: F32 | F64 | null;
}

/** What the sort produces: the core arrays, the counts and the truthful flags. */
export interface SortResult {
    /** The core arrays (in the arena when requested). */
    readonly core: CoreArrays;
    /** colIdx.length. */
    readonly arcCount: number;
    /** Logical edges with source === target. */
    readonly selfLoopCount: number;
    /** The flags of design section 3.8, computed from the arrays (invariant I9). */
    readonly flags: SnapshotFlags;
}

/**
 * The number of self-loop edges among the first `edgeCount` entries.
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - how many edges to inspect
 * @returns the loop count
 */
export function countSelfLoops(src: U32, dst: U32, edgeCount: number): number {
    let loops = 0;
    for (let e = 0; e < edgeCount; e++) {
        if (src[e] === dst[e]) {
            loops++;
        }
    }
    return loops;
}

/**
 * The identity check of design section 6.3 step 4: whether (src[e], dst[e]) is non-decreasing
 * lexicographically, so that a stable sort by (source, target) moves nothing and the CSR permutation is
 * the identity.
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - how many edges to inspect
 * @returns true when the edges are already in CSR order
 */
export function isSortedEdgeList(src: U32, dst: U32, edgeCount: number): boolean {
    for (let e = 1; e < edgeCount; e++) {
        const u = src[e];
        const prev = src[e - 1];
        if (u < prev || (u === prev && dst[e] < dst[e - 1])) {
            return false;
        }
    }
    return true;
}

/**
 * The E_INVALID_WEIGHT error for a NaN staging weight (invariant I8).
 * @param e - the logical edge
 * @param w - the value
 * @returns the error
 */
function nanWeight(e: number, w: number): GraphFormatError {
    return new GraphFormatError("E_INVALID_WEIGHT", `edge ${e} has a NaN weight`, { edge: e, weight: w });
}

// ============================================================ the sequential edge pass

/**
 * The weight scan of design section 6.3 step 8 over the first `edgeCount` staging weights: NaN is
 * refused (invariant I8, the bulk-input re-check, naming the lowest such edge), and the smallest and
 * largest STORED values -- the values as the f32 arc array holds them (invariant I9: an f64 that
 * rounds to 1, to -0 or to Infinity counts as its rounded value) -- are left in `bounds[0]` and
 * `bounds[1]`. They are written from inside the loop whenever they change, so the loop's optimised
 * code has no exit path without type feedback.
 * @param edgeWeights - the staging weights
 * @param edgeCount - how many edges to inspect; at least 1
 * @param bounds - two slots initialised to +Infinity / -Infinity; receive the minimum and the maximum
 */
function scanStoredWeights(edgeWeights: F32 | F64, edgeCount: number, bounds: F64): void {
    let min = Infinity;
    let max = -Infinity;
    for (let e = 0; e < edgeCount; e++) {
        const w = Math.fround(edgeWeights[e]);
        if (w !== w) {
            throw nanWeight(e, edgeWeights[e]);
        }
        if (w < min) {
            min = w;
            bounds[0] = w;
        }
        if (w > max) {
            max = w;
            bounds[1] = w;
        }
    }
}

/**
 * The counting half of the directed sort (design section 6.3 steps 5 and 6, the `cnt` arrays):
 * arcs per target into `targetCount[v + 1]` and arcs per source into `rowCount[u + 1]`, ready for the
 * exclusive prefix sums.
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - the logical edge count
 * @param targetCount - n + 1 zeroed counters, by target
 * @param rowCount - n + 1 zeroed counters, by source
 * @returns the self-loop count
 */
function countDirectedArcs(src: U32, dst: U32, edgeCount: number, targetCount: U32, rowCount: U32): number {
    let loops = 0;
    for (let e = 0; e < edgeCount; e++) {
        const u = src[e];
        const v = dst[e];
        targetCount[v + 1]++;
        rowCount[u + 1]++;
        if (u === v) {
            loops++;
        }
    }
    return loops;
}

/**
 * The counting half of the undirected sort (design section 6.3 steps 3, 5 and 6): every non-loop
 * edge contributes an arc in each direction, a self-loop one arc (invariant I7).
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - the logical edge count
 * @param targetCount - n + 1 zeroed counters, by target
 * @param rowCount - n + 1 zeroed counters, by source
 * @returns the self-loop count
 */
function countUndirectedArcs(src: U32, dst: U32, edgeCount: number, targetCount: U32, rowCount: U32): number {
    let loops = 0;
    for (let e = 0; e < edgeCount; e++) {
        const u = src[e];
        const v = dst[e];
        targetCount[v + 1]++;
        rowCount[u + 1]++;
        if (u === v) {
            loops++;
        } else {
            targetCount[u + 1]++;
            rowCount[v + 1]++;
        }
    }
    return loops;
}

/**
 * Turn per-node counts stored at index + 1 into exclusive prefix sums in place: afterwards
 * `counts[v]` is the start of v's bucket and `counts[nodeCount]` the total.
 * @param counts - n + 1 counters with `counts[0] === 0`
 * @param nodeCount - n
 */
function prefixSum(counts: U32, nodeCount: number): void {
    for (let v = 0; v < nodeCount; v++) {
        counts[v + 1] += counts[v];
    }
}

// ============================================================ the identity path

/**
 * The identity path of design section 6.3 step 4 (directed input already in CSR order): colIdx is a
 * copy of dst, and the multigraph flag is one comparison of adjacent edges.
 * @param src - sources, grouped
 * @param dst - targets, sorted within each group
 * @param edgeCount - the logical edge count
 * @param colIdx - receives dst
 * @returns true when two adjacent edges connect the same pair
 */
function copySortedTargets(src: U32, dst: U32, edgeCount: number, colIdx: U32): boolean {
    let multigraph = false;
    for (let e = 0; e < edgeCount; e++) {
        const v = dst[e];
        colIdx[e] = v;
        if (e > 0 && src[e] === src[e - 1] && dst[e - 1] === v) {
            multigraph = true;
        }
    }
    return multigraph;
}

// ============================================================ pass 1: by target

/**
 * Pass 1 of a directed sort (design section 6.3 step 5): scatter every edge into target order as a
 * (source, edge) pair, `byTarget[2p] = src[e]`, `byTarget[2p + 1] = e`.
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - the logical edge count
 * @param cursor - the target bucket starts; advanced in place
 * @param byTarget - 2A words receiving the pairs
 */
function scatterDirectedByTarget(src: U32, dst: U32, edgeCount: number, cursor: U32, byTarget: U32): void {
    for (let e = 0; e < edgeCount; e++) {
        const p = 2 * cursor[dst[e]]++;
        byTarget[p] = src[e];
        byTarget[p + 1] = e;
    }
}

/**
 * Pass 1 of an undirected sort (design section 6.3 steps 3 and 5, fused): the declared arc
 * `(src, dst, e)` goes into bucket dst and, for a non-loop, the mirror `(dst, src, e)` into bucket
 * src right after it, so arcs stay in edge order within every bucket. The pass-1 position of every
 * declared arc is marked in `declared` (bit p).
 * @param src - sources
 * @param dst - targets
 * @param edgeCount - the logical edge count
 * @param cursor - the target bucket starts; advanced in place
 * @param byTarget - 2A words receiving the (source, edge) pairs
 * @param declared - a zeroed bitmap of A bits
 */
function scatterUndirectedByTarget(
    src: U32,
    dst: U32,
    edgeCount: number,
    cursor: U32,
    byTarget: U32,
    declared: U32,
): void {
    for (let e = 0; e < edgeCount; e++) {
        const u = src[e];
        const v = dst[e];
        const p = cursor[v]++;
        byTarget[2 * p] = u;
        byTarget[2 * p + 1] = e;
        declared[p >>> 5] |= 1 << (p & 31);
        if (u !== v) {
            const q = 2 * cursor[u]++;
            byTarget[q] = v;
            byTarget[q + 1] = e;
        }
    }
}

// ============================================================ pass 2: by source

/**
 * Pass 2 of a directed sort (design section 6.3 step 6): walk the target buckets in order and
 * scatter every arc into its row, writing colIdx and arcToEdge. Stable, so rows end sorted by target
 * with parallels in edge order (I4).
 * @param nodeCount - n
 * @param targetStart - the target bucket starts (pass-1 layout)
 * @param byTarget - the pass-1 (source, edge) pairs
 * @param cursor - the row starts (a copy of rowPtr); advanced in place
 * @param colIdx - receives the targets
 * @param arcToEdge - receives the edges
 */
function scatterDirectedBySource(
    nodeCount: number,
    targetStart: U32,
    byTarget: U32,
    cursor: U32,
    colIdx: U32,
    arcToEdge: U32,
): void {
    for (let v = 0; v < nodeCount; v++) {
        const end = 2 * targetStart[v + 1];
        for (let p = 2 * targetStart[v]; p < end; p += 2) {
            const a = cursor[byTarget[p]]++;
            colIdx[a] = v;
            arcToEdge[a] = byTarget[p + 1];
        }
    }
}

/**
 * Pass 2 of an undirected sort (design section 6.3 step 6): as the directed pass, plus
 * `edgeToArc[e] = a` for the arc whose stored orientation is the declared one (for a self-loop the
 * single arc), which the pass-1 bitmap identifies.
 * @param nodeCount - n
 * @param targetStart - the target bucket starts (pass-1 layout)
 * @param byTarget - the pass-1 (source, edge) pairs
 * @param declared - bit p set when pass-1 position p holds a declared arc
 * @param cursor - the row starts (a copy of rowPtr); advanced in place
 * @param colIdx - receives the targets
 * @param arcToEdge - receives the edges
 * @param edgeToArc - receives the declared arc of every edge
 */
function scatterUndirectedBySource(
    nodeCount: number,
    targetStart: U32,
    byTarget: U32,
    declared: U32,
    cursor: U32,
    colIdx: U32,
    arcToEdge: U32,
    edgeToArc: U32,
): void {
    for (let v = 0; v < nodeCount; v++) {
        const end = targetStart[v + 1];
        for (let p = targetStart[v]; p < end; p++) {
            const e = byTarget[2 * p + 1];
            const a = cursor[byTarget[2 * p]]++;
            colIdx[a] = v;
            arcToEdge[a] = e;
            if (((declared[p >>> 5] >>> (p & 31)) & 1) === 1) {
                edgeToArc[e] = a;
            }
        }
    }
}

// ============================================================ sequential post-passes

/**
 * The inverse of a directed permutation: `edgeToArc[arcToEdge[a]] = a` (every arc is its edge's only
 * arc).
 * @param arcToEdge - the arc permutation
 * @param edgeToArc - receives the inverse
 */
function invertPermutation(arcToEdge: U32, edgeToArc: U32): void {
    const arcCount = arcToEdge.length;
    for (let a = 0; a < arcCount; a++) {
        edgeToArc[arcToEdge[a]] = a;
    }
}

/**
 * The arc weights (design section 3.7): `weights[a] = f32(edgeWeights[arcToEdge[a]])`, a sequential
 * write with one independent random read per arc.
 * @param arcToEdge - the arc permutation
 * @param edgeWeights - the staging weights
 * @param weights - receives the f32 arc weights
 */
function gatherWeights(arcToEdge: U32, edgeWeights: F32 | F64, weights: F32): void {
    const arcCount = arcToEdge.length;
    for (let a = 0; a < arcCount; a++) {
        weights[a] = edgeWeights[arcToEdge[a]];
    }
}

/**
 * The multigraph flag (design section 3.8): whether some row holds two adjacent arcs with the same
 * target; rows are sorted by target, so parallels are adjacent (I4).
 * @param rowPtr - the row starts
 * @param colIdx - the sorted targets
 * @param nodeCount - n
 * @returns true when a parallel pair exists
 */
function hasAdjacentParallels(rowPtr: U32, colIdx: U32, nodeCount: number): boolean {
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u] + 1; a < end; a++) {
            if (colIdx[a] === colIdx[a - 1]) {
                return true;
            }
        }
    }
    return false;
}

// ============================================================ the sort

/**
 * Sort a per-edge list into a CSR core (design section 6.3 steps 3-6 and 8). Directed input that is
 * already in CSR order takes the identity path (one count + scan + copy, no permutation arrays,
 * `flags.arcToEdgeIsIdentity`); everything else takes the two counting-sort passes. Undirected input
 * emits the mirror arc of every non-loop edge (I7). Weights are stored as f32 (design section 3.7) and
 * the flags are computed from the stored values (I9). A NaN weight is E_INVALID_WEIGHT (I8) naming
 * the lowest such edge, thrown before anything is allocated.
 * @param input - the per-edge arrays
 * @param useArena - allocate the core inside one 256-aligned arena (true) or as separate buffers
 * @returns the core, the counts and the flags
 */
export function sortIntoCore(input: SortInput, useArena: boolean): SortResult {
    const { directed, nodeCount, edgeCount, src, dst } = input;
    const edgeWeights = input.weights;

    // step 8's weight predicates, over the edges: every edge lands on at least one arc, so the
    // per-value predicates over the stored arc weights equal those over the f32-rounded edge weights
    let allOne = true;
    let nonNegative = true;
    let finite = true;
    if (edgeWeights !== null && edgeCount > 0) {
        const bounds = new Float64Array([Infinity, -Infinity]);
        scanStoredWeights(edgeWeights, edgeCount, bounds);
        const min = bounds[0];
        const max = bounds[1];
        allOne = min === 1 && max === 1;
        nonNegative = min >= 0;
        finite = min > -Infinity && max < Infinity;
    }

    // steps 3-6 counts: one sequential pass over the edges yields both prefix sums and the loop count
    const targetStart = new Uint32Array(nodeCount + 1);
    const rowCount = new Uint32Array(nodeCount + 1);
    const selfLoopCount = directed
        ? countDirectedArcs(src, dst, edgeCount, targetStart, rowCount)
        : countUndirectedArcs(src, dst, edgeCount, targetStart, rowCount);
    const arcCount = directed ? edgeCount : 2 * edgeCount - selfLoopCount;
    const identity = directed && isSortedEdgeList(src, dst, edgeCount);
    const core = allocateCoreArrays(
        { nodeCount, arcCount, edgeCount, weighted: edgeWeights !== null, identity },
        useArena,
    );
    const { rowPtr, colIdx, weights, arcToEdge, edgeToArc } = core;
    prefixSum(rowCount, nodeCount);
    rowPtr.set(rowCount);
    let multigraph: boolean;

    if (identity) {
        // step 4: the edge arrays are the arc arrays -- copy, and round the weights to f32
        multigraph = copySortedTargets(src, dst, edgeCount, colIdx);
        if (weights !== null && edgeWeights !== null) {
            weights.set(edgeWeights.subarray(0, edgeCount));
        }
    } else if (arcToEdge !== null && edgeToArc !== null) {
        // step 5: pass 1 by target; `rowCount` is reused as the pass-1 cursor
        prefixSum(targetStart, nodeCount);
        const cursor = rowCount;
        cursor.set(targetStart);
        const byTarget = new Uint32Array(2 * arcCount);
        let declared: U32 | null = null;
        if (directed) {
            scatterDirectedByTarget(src, dst, edgeCount, cursor, byTarget);
        } else {
            declared = new Uint32Array((arcCount + 31) >>> 5);
            scatterUndirectedByTarget(src, dst, edgeCount, cursor, byTarget, declared);
        }
        // step 6: pass 2 by source into the rows; the cursor now walks rowPtr
        cursor.set(rowPtr);
        if (declared === null) {
            scatterDirectedBySource(nodeCount, targetStart, byTarget, cursor, colIdx, arcToEdge);
            invertPermutation(arcToEdge, edgeToArc);
        } else {
            scatterUndirectedBySource(nodeCount, targetStart, byTarget, declared, cursor, colIdx, arcToEdge, edgeToArc);
        }
        if (weights !== null && edgeWeights !== null) {
            gatherWeights(arcToEdge, edgeWeights, weights);
        }
        multigraph = hasAdjacentParallels(rowPtr, colIdx, nodeCount);
    } else {
        // allocateCoreArrays materialises both permutations whenever the sort is not the identity
        throw new GraphFormatError("E_UNSUPPORTED", "unreachable: a non-identity core without permutation arrays", {
            directed,
            arcCount,
        });
    }

    const flags: SnapshotFlags = Object.freeze({
        multigraph,
        hasSelfLoops: selfLoopCount > 0,
        arcToEdgeIsIdentity: identity,
        weighted: weights !== null,
        allWeightsOne: allOne,
        nonNegativeWeights: nonNegative,
        finiteWeights: finite,
    });
    return { core, arcCount, selfLoopCount, flags };
}
