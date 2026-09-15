/**
 * The cached views of design section 7.2 as pure functions of a snapshot's core arrays, plus the
 * per-arc / per-edge boundary helpers `foldArcs` and `expandEdges` of design section 7.5.
 *
 * Every function here computes one view from the public core (rowPtr, colIdx, weights, arcToEdge,
 * edgeToArc, flags, counts) in O(n + m) or O(n log d) and returns a fresh object; GraphSnapshot owns
 * the per-instance cache (invariant I17: computed once, never invalidated, SHARED -- a caller that
 * needs scratch calls `.slice()`). Aliasing is deliberate and documented per view: `coo().dst` is
 * `colIdx`, `reverse()` of an undirected snapshot is the forward arrays, an identity permutation
 * lets `edgeList().weights` alias `weights`, and so on, so `viewByteLength()` counts only the bytes a
 * view owns.
 *
 * Nothing here allocates an identity permutation that the snapshot has not materialised: view objects
 * reach `arcToEdge` / `edgeToArc` through getters, so touching `coo()` on a directed identity
 * snapshot costs one `src` array and nothing else until `coo().arcToEdge` is read.
 */

import { GraphFormatError } from "../errors.js";
import {
    type CooView,
    type DegreeOrderView,
    type EdgeListView,
    type F32,
    type F64,
    type GraphSnapshot,
    type NumericVector,
    type ReverseView,
    type U32,
    type ViewName,
} from "../types/index.js";
import { assertOneOf } from "../util/options.js";
import { arcRangeIn } from "./queries.js";

// ============================================================ reverse

/**
 * The cuGraph degree tiers of `degreeOrder()`: a node with at least DEGREE_TIER_HIGH arcs is scheduled
 * per workgroup, at least DEGREE_TIER_MID per subgroup, at least DEGREE_TIER_LOW per thread; degree 0
 * nodes form the trailing segment (design section 7.2).
 */
export const DEGREE_TIER_HIGH = 1024;
/** Lower bound of the middle degree tier. */
export const DEGREE_TIER_MID = 32;
/** Lower bound of the low degree tier. */
const DEGREE_TIER_LOW = 1;

/**
 * The identity permutation of `length` entries as a fresh 4-byte-aligned array (invariant I10).
 * @param length - the entry count
 * @returns 0..length-1
 */
export function identityPermutation(length: number): U32 {
    const out = new Uint32Array(length);
    for (let i = 0; i < length; i++) {
        out[i] = i;
    }
    return out;
}

/**
 * The in-adjacency of a snapshot (design section 7.2): `ReverseView` with `fwdArc` (reverse arc k ->
 * forward arc) and `arcToEdge` (`arcToEdge[fwdArc[k]]`) materialised lazily. For an undirected
 * snapshot the forward arrays are shared (invariant I7), `fwdArc` is the identity (allocated on first
 * read) and `arcToEdge` is the snapshot's own array. For a directed snapshot whose permutation is the
 * identity, the reverse `arcToEdge` IS `fwdArc` (aliased, zero bytes).
 */
export class ReverseAdjacency implements ReverseView {
    /** Whether the source snapshot is directed. */
    readonly directed: boolean;
    /** Number of nodes. */
    readonly nodeCount: number;
    /** Number of arcs. */
    readonly arcCount: number;
    /** Row offsets of the reverse adjacency (the forward rowPtr when undirected). */
    readonly rowPtr: U32;
    /** Source node of every forward arc, sorted within each reverse row (the forward colIdx when undirected). */
    readonly colIdx: U32;
    /** Per-reverse-arc weights, gathered from the forward weights (the forward array when undirected). */
    readonly weights: F32 | null;

    private readonly source: GraphSnapshot;
    private fwdArcCache: U32 | null;
    private arcToEdgeCache: U32 | null;
    private arcToEdgeOwned: boolean;
    private readonly fwdArcIsIdentity: boolean;

    /**
     * Wrap the reverse arrays. Used by `computeReverse()`; not meant to be constructed elsewhere.
     * @param source - the snapshot the view belongs to
     * @param rowPtr - the reverse row offsets
     * @param colIdx - the reverse targets (forward sources)
     * @param weights - the gathered weights, or null when unweighted
     * @param fwdArc - the reverse arc -> forward arc map, or null when it is the identity (undirected)
     */
    constructor(source: GraphSnapshot, rowPtr: U32, colIdx: U32, weights: F32 | null, fwdArc: U32 | null) {
        this.source = source;
        this.directed = source.directed;
        this.nodeCount = source.nodeCount;
        this.arcCount = source.arcCount;
        this.rowPtr = rowPtr;
        this.colIdx = colIdx;
        this.weights = weights;
        this.fwdArcCache = fwdArc;
        this.fwdArcIsIdentity = fwdArc === null;
        this.arcToEdgeCache = null;
        this.arcToEdgeOwned = false;
    }

    /**
     * Reverse arc k -> forward arc index; the identity for an undirected snapshot, materialised on first
     * read outside any arena.
     * @returns the map, arcCount entries
     */
    get fwdArc(): U32 {
        this.fwdArcCache ??= identityPermutation(this.arcCount);
        return this.fwdArcCache;
    }

    /**
     * Logical edge of every reverse arc: `source.arcToEdge[fwdArc[k]]`, materialised on first read.
     * Aliases `source.arcToEdge` when undirected and `fwdArc` when the forward permutation is the
     * identity.
     * @returns the map, arcCount entries
     */
    get arcToEdge(): U32 {
        if (this.arcToEdgeCache === null) {
            if (this.fwdArcIsIdentity) {
                this.arcToEdgeCache = this.source.arcToEdge;
            } else if (this.source.flags.arcToEdgeIsIdentity) {
                this.arcToEdgeCache = this.fwdArc;
            } else {
                const { fwdArc } = this;
                const forward = this.source.arcToEdge;
                const out = new Uint32Array(this.arcCount);
                for (let k = 0; k < out.length; k++) {
                    out[k] = forward[fwdArc[k]];
                }
                this.arcToEdgeCache = out;
                this.arcToEdgeOwned = true;
            }
        }
        return this.arcToEdgeCache;
    }

    /**
     * Bytes owned by this view (not shared with the snapshot's core): the reverse arrays when directed,
     * a materialised identity `fwdArc`, and a gathered `arcToEdge`.
     * @returns the byte count
     */
    get ownedByteLength(): number {
        let bytes = 0;
        if (this.directed) {
            bytes += this.rowPtr.byteLength + this.colIdx.byteLength;
            if (this.weights !== null) {
                bytes += this.weights.byteLength;
            }
        }
        if (this.fwdArcCache !== null) {
            bytes += this.fwdArcCache.byteLength;
        }
        if (this.arcToEdgeOwned && this.arcToEdgeCache !== null) {
            bytes += this.arcToEdgeCache.byteLength;
        }
        return bytes;
    }

    /**
     * The arrays this view has materialised so far, for checksums and byte accounting.
     * @returns the arrays, keyed by member name
     */
    materialised(): Readonly<Record<string, Uint32Array | Float32Array>> {
        const out: Record<string, Uint32Array | Float32Array> = {
            rowPtr: this.rowPtr,
            colIdx: this.colIdx,
        };
        if (this.weights !== null) {
            out.weights = this.weights;
        }
        if (this.fwdArcCache !== null) {
            out.fwdArc = this.fwdArcCache;
        }
        if (this.arcToEdgeCache !== null) {
            out.arcToEdge = this.arcToEdgeCache;
        }
        return out;
    }
}

/**
 * Compute the reverse view (design section 7.2): a stable counting sort of the forward arcs by target
 * (O(n + m)) so that every reverse row is sorted by source with parallels in ascending edge order
 * (invariant I4 for the reverse), weights gathered when weighted (Q37). Undirected: the forward
 * arrays themselves (invariant I7).
 * @param snapshot - the snapshot
 * @returns the reverse view
 */
export function computeReverse(snapshot: GraphSnapshot): ReverseAdjacency {
    const { nodeCount, arcCount, rowPtr, colIdx, weights } = snapshot;
    if (!snapshot.directed) {
        return new ReverseAdjacency(snapshot, rowPtr, colIdx, weights, null);
    }
    const revPtr = new Uint32Array(nodeCount + 1);
    for (let a = 0; a < arcCount; a++) {
        revPtr[colIdx[a] + 1]++;
    }
    for (let v = 0; v < nodeCount; v++) {
        revPtr[v + 1] += revPtr[v];
    }
    const revIdx = new Uint32Array(arcCount);
    const fwdArc = new Uint32Array(arcCount);
    const revWeights = weights === null ? null : new Float32Array(arcCount);
    const cursor = revPtr.slice(0, nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            const v = colIdx[a];
            const k = cursor[v]++;
            revIdx[k] = u;
            fwdArc[k] = a;
            if (revWeights !== null && weights !== null) {
                revWeights[k] = weights[a];
            }
        }
    }
    return new ReverseAdjacency(snapshot, revPtr, revIdx, revWeights, fwdArc);
}

// ============================================================ coo and edge list

/**
 * The source node of every arc: `rowPtr` expanded to arc length.
 * @param rowPtr - the row offsets
 * @param nodeCount - the node count
 * @param arcCount - the arc count
 * @returns a fresh Uint32Array(arcCount)
 */
function expandRowPtr(rowPtr: U32, nodeCount: number, arcCount: number): U32 {
    const src = new Uint32Array(arcCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            src[a] = u;
        }
    }
    return src;
}

/**
 * Compute the per-arc COO view (design section 7.2): `src` is the only new array; `dst` aliases
 * `colIdx`, `weights` aliases `weights`, and `arcToEdge` reaches the snapshot's array through a
 * getter (so an identity permutation is not materialised by building the view).
 * @param snapshot - the snapshot
 * @returns the COO view
 */
export function computeCoo(snapshot: GraphSnapshot): CooView {
    return cooViewOf(snapshot, expandRowPtr(snapshot.rowPtr, snapshot.nodeCount, snapshot.arcCount));
}

/**
 * Wrap a per-arc source array as the COO view of a snapshot (the shape `computeCoo()` returns; also
 * used to install a carried `src` from the wire).
 * @param snapshot - the snapshot
 * @param src - source node of every arc, arcCount entries
 * @returns the COO view
 */
export function cooViewOf(snapshot: GraphSnapshot, src: U32): CooView {
    return Object.freeze({
        src,
        dst: snapshot.colIdx,
        weights: snapshot.weights,
        get arcToEdge(): U32 {
            return snapshot.arcToEdge;
        },
    });
}

/**
 * Compute the edge list view (design section 7.2): every logical edge once in declared orientation.
 * `src[e]` is the row holding `edgeToArc[e]`, `dst[e] === colIdx[edgeToArc[e]]`, `arc` aliases
 * `edgeToArc` through a getter, `weights` is gathered through `edgeToArc` (aliased when the
 * permutation is the identity, in which case `dst` aliases `colIdx` too).
 * @param snapshot - the snapshot
 * @returns the edge list view
 */
export function computeEdgeList(snapshot: GraphSnapshot): EdgeListView {
    const { nodeCount, edgeCount, arcCount, rowPtr, colIdx, weights } = snapshot;
    if (snapshot.flags.arcToEdgeIsIdentity) {
        return edgeListViewOf(snapshot, expandRowPtr(rowPtr, nodeCount, arcCount), colIdx, weights);
    }
    const { arcToEdge, edgeToArc } = snapshot;
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            const e = arcToEdge[a];
            if (edgeToArc[e] === a) {
                src[e] = u;
                dst[e] = colIdx[a];
            }
        }
    }
    let edgeWeights: F32 | null = null;
    if (weights !== null) {
        edgeWeights = new Float32Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            edgeWeights[e] = weights[edgeToArc[e]];
        }
    }
    return edgeListViewOf(snapshot, src, dst, edgeWeights);
}

/**
 * Wrap per-edge arrays as the edge list view of a snapshot (the shape `computeEdgeList()` returns;
 * also used to install carried arrays from the wire). `arc` reaches `edgeToArc` through a getter so
 * an identity permutation is not materialised by building the view.
 * @param snapshot - the snapshot
 * @param src - declared source of every edge, edgeCount entries
 * @param dst - declared target of every edge, edgeCount entries (colIdx itself when the permutation is the identity)
 * @param weights - per-edge weights, or null when unweighted
 * @returns the edge list view
 */
export function edgeListViewOf(snapshot: GraphSnapshot, src: U32, dst: U32, weights: F32 | null): EdgeListView {
    return Object.freeze({
        src,
        dst,
        weights,
        get arc(): U32 {
            return snapshot.edgeToArc;
        },
    });
}

// ============================================================ degrees

/**
 * Out-arc counts: `rowPtr[u + 1] - rowPtr[u]` (a self-loop counted once, design section 3.4).
 * @param rowPtr - the row offsets
 * @param nodeCount - the node count
 * @returns a fresh Uint32Array(nodeCount)
 */
export function rowLengths(rowPtr: U32, nodeCount: number): U32 {
    const out = new Uint32Array(nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        out[u] = rowPtr[u + 1] - rowPtr[u];
    }
    return out;
}

/**
 * The graph-theoretic degree (design section 3.4): in + out when directed, out + self-loops when
 * undirected (a loop counts twice, the NetworkX convention).
 * @param outDegree - the out-degree view
 * @param inOrLoops - inDegree() when directed, selfLoopsPerNode() when undirected
 * @returns a fresh Uint32Array(n)
 */
export function sumDegrees(outDegree: U32, inOrLoops: U32): U32 {
    const out = new Uint32Array(outDegree.length);
    for (let u = 0; u < out.length; u++) {
        out[u] = outDegree[u] + inOrLoops[u];
    }
    return out;
}

/**
 * Row sums of an arc-aligned weight array as f64 (design section 7.2): a self-loop arc counted once;
 * a row may sum to 0 because zero weights are legal.
 * @param rowPtr - the row offsets of the adjacency being summed (forward or reverse)
 * @param weights - the arc-aligned weights of that adjacency
 * @param nodeCount - the node count
 * @returns a fresh Float64Array(nodeCount)
 */
export function rowWeightSums(rowPtr: U32, weights: F32, nodeCount: number): F64 {
    const out = new Float64Array(nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let sum = 0;
        for (let a = rowPtr[u]; a < end; a++) {
            sum += weights[a];
        }
        out[u] = sum;
    }
    return out;
}

/**
 * A u32 count vector widened to f64 (the unweighted case of the weighted-degree views).
 * @param counts - the counts
 * @returns a fresh Float64Array of the same length
 */
export function widenToF64(counts: U32): F64 {
    return new Float64Array(counts);
}

/**
 * Element-wise sum of two f64 vectors of equal length.
 * @param a - the first vector
 * @param b - the second vector
 * @returns a fresh Float64Array
 */
export function sumF64(a: F64, b: F64): F64 {
    const out = new Float64Array(a.length);
    for (let i = 0; i < out.length; i++) {
        out[i] = a[i] + b[i];
    }
    return out;
}

/**
 * The sum of weights over logical edges, each undirected edge once (design section 7.2), accumulated
 * in f64. `edgeCount` when unweighted.
 * @param snapshot - the snapshot
 * @returns the total weight
 */
export function computeTotalWeight(snapshot: GraphSnapshot): number {
    const { weights, edgeCount } = snapshot;
    if (weights === null) {
        return edgeCount;
    }
    let sum = 0;
    if (snapshot.flags.arcToEdgeIsIdentity) {
        for (let a = 0; a < weights.length; a++) {
            sum += weights[a];
        }
        return sum;
    }
    const { edgeToArc } = snapshot;
    for (let e = 0; e < edgeCount; e++) {
        sum += weights[edgeToArc[e]];
    }
    return sum;
}

// ============================================================ self-loops

/** The two self-loop views, computed together in one O(n log d) pass. */
interface SelfLoopViews {
    /** Arcs a with colIdx[a] === row(a), ascending. */
    readonly selfLoopArcs: U32;
    /** Loop arcs per node. */
    readonly selfLoopsPerNode: U32;
}

/**
 * Find every self-loop arc by a binary search per row (design section 7.2).
 * @param snapshot - the snapshot
 * @returns the loop arcs (length selfLoopCount) and the per-node counts
 */
export function computeSelfLoops(snapshot: GraphSnapshot): SelfLoopViews {
    const { nodeCount, rowPtr, colIdx, selfLoopCount } = snapshot;
    const perNode = new Uint32Array(nodeCount);
    const arcs = new Uint32Array(selfLoopCount);
    let at = 0;
    for (let u = 0; u < nodeCount; u++) {
        const [lo, hi] = arcRangeIn(rowPtr, colIdx, u, u);
        perNode[u] = hi - lo;
        for (let a = lo; a < hi && at < selfLoopCount; a++) {
            arcs[at++] = a;
        }
    }
    return { selfLoopArcs: arcs, selfLoopsPerNode: perNode };
}

/**
 * Sum of weights over each node's self-loop arcs as f64; the loop counts widened when unweighted.
 * @param snapshot - the snapshot
 * @param loops - the self-loop views
 * @returns a fresh Float64Array(n)
 */
export function computeSelfLoopWeight(snapshot: GraphSnapshot, loops: SelfLoopViews): F64 {
    const { weights, nodeCount, rowPtr, colIdx } = snapshot;
    if (weights === null) {
        return widenToF64(loops.selfLoopsPerNode);
    }
    const out = new Float64Array(nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        if (loops.selfLoopsPerNode[u] === 0) {
            continue;
        }
        const [lo, hi] = arcRangeIn(rowPtr, colIdx, u, u);
        let sum = 0;
        for (let a = lo; a < hi; a++) {
            sum += weights[a];
        }
        out[u] = sum;
    }
    return out;
}

// ============================================================ mate

/**
 * The E_INVALID_SNAPSHOT error the mate walk raises when the doubled storage of invariant I7 does not
 * hold (the row of the mate does not hold the expected source at the cursor).
 * @param u - the row being walked
 * @param v - the target whose mate group was expected
 * @param arc - the arc where the walk failed
 * @returns the error
 */
function pairingError(u: number, v: number, arc: number): GraphFormatError {
    return new GraphFormatError(
        "E_INVALID_SNAPSHOT",
        `invariant I7 violated: arc ${arc} in row ${u} targeting ${v} has no mate in row ${v}`,
        { invariant: "I7", row: u, arc, target: v },
    );
}

/**
 * For every arc of an undirected snapshot, the arc storing the opposite orientation of the same edge
 * (a self-loop maps to itself), by the O(m) lockstep walk of design section 6.4: rows are visited in
 * ascending order, every group of k parallel arcs u -> v (v > u) is paired k-th to k-th with the group
 * v -> u at row v's cursor, and the cursor advances so that by the time row v is visited every
 * target below v has been consumed. Throws E_DIRECTED on a directed snapshot.
 * @param snapshot - an undirected snapshot
 * @returns a fresh Uint32Array(arcCount)
 */
export function computeMate(snapshot: GraphSnapshot): U32 {
    if (snapshot.directed) {
        throw new GraphFormatError("E_DIRECTED", "mate() is defined for undirected snapshots only", {
            directed: true,
        });
    }
    const { nodeCount, arcCount, rowPtr, colIdx } = snapshot;
    const mate = new Uint32Array(arcCount);
    const cursor = rowPtr.slice(0, nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let a = cursor[u];
        while (a < end) {
            const v = colIdx[a];
            let g = a + 1;
            while (g < end && colIdx[g] === v) {
                g++;
            }
            const k = g - a;
            if (v === u) {
                for (let i = 0; i < k; i++) {
                    mate[a + i] = a + i;
                }
            } else {
                const b = cursor[v];
                if (v < u || b + k > rowPtr[v + 1] || colIdx[b] !== u || colIdx[b + k - 1] !== u) {
                    throw pairingError(u, v, a);
                }
                for (let i = 0; i < k; i++) {
                    mate[a + i] = b + i;
                    mate[b + i] = a + i;
                }
                cursor[v] = b + k;
            }
            a = g;
        }
        cursor[u] = end;
    }
    return mate;
}

// ============================================================ degree order

/**
 * Nodes permuted by descending out-degree of an adjacency (counting sort, ties in ascending node
 * index) with the cuGraph tier boundaries (design section 7.2): `segmentOffsets = [0, hiEnd,
 * midEnd, lowEnd, n]` for the thresholds 1024 / 32 / 1.
 * @param rowPtr - the row offsets of the adjacency to order by (forward, or reverse for the in-degree)
 * @param nodeCount - the node count
 * @returns the permutation and its tier offsets
 */
export function computeDegreeOrder(rowPtr: U32, nodeCount: number): DegreeOrderView {
    const degree = rowLengths(rowPtr, nodeCount);
    let maxDegree = 0;
    for (let u = 0; u < nodeCount; u++) {
        if (degree[u] > maxDegree) {
            maxDegree = degree[u];
        }
    }
    const start = new Uint32Array(maxDegree + 1);
    for (let u = 0; u < nodeCount; u++) {
        start[degree[u]]++;
    }
    // descending: the start of degree d is the number of nodes with a larger degree
    let run = 0;
    for (let d = maxDegree; d >= 0; d--) {
        const count = start[d];
        start[d] = run;
        run += count;
    }
    const perm = new Uint32Array(nodeCount);
    let hi = 0;
    let mid = 0;
    let low = 0;
    for (let u = 0; u < nodeCount; u++) {
        const d = degree[u];
        perm[start[d]++] = u;
        if (d >= DEGREE_TIER_HIGH) {
            hi++;
        } else if (d >= DEGREE_TIER_MID) {
            mid++;
        } else if (d >= DEGREE_TIER_LOW) {
            low++;
        }
    }
    const segmentOffsets = new Uint32Array([0, hi, hi + mid, hi + mid + low, nodeCount]);
    return Object.freeze({ perm, segmentOffsets });
}

// ============================================================ symmetry

/**
 * Whether a directed snapshot's arc set is closed under reversal with equal weights (design sections
 * 3.6 and 7.2): forward row v and reverse row v hold the same targets (versus sources; both sorted,
 * so one lockstep walk), and within each run of parallel arcs to one target the same multiset of
 * weights. Parallel arcs are ordered by edge index in both rows, so their weights can arrive in a
 * different order (an expanded undirected multigraph with unequal parallel weights, design section
 * 6.6); each run is compared as a sorted multiset, which is positional for simple graphs. Undirected
 * snapshots are symmetric by construction.
 * @param snapshot - the snapshot
 * @param reverse - its reverse view
 * @returns true when symmetric
 */
export function computeSymmetric(snapshot: GraphSnapshot, reverse: ReverseView): boolean {
    if (!snapshot.directed) {
        return true;
    }
    const { nodeCount, arcCount, rowPtr, colIdx, weights } = snapshot;
    for (let v = 0; v <= nodeCount; v++) {
        if (rowPtr[v] !== reverse.rowPtr[v]) {
            return false;
        }
    }
    const revIdx = reverse.colIdx;
    for (let a = 0; a < arcCount; a++) {
        if (colIdx[a] !== revIdx[a]) {
            return false;
        }
    }
    const revWeights = reverse.weights;
    if (weights === null || revWeights === null) {
        return true;
    }
    if (!snapshot.flags.multigraph) {
        for (let a = 0; a < arcCount; a++) {
            if (weights[a] !== revWeights[a]) {
                return false;
            }
        }
        return true;
    }
    // multigraph: compare each parallel run (adjacent by I4, equal targets in both rows) as a multiset
    let a = 0;
    while (a < arcCount) {
        let b = a + 1;
        while (b < arcCount && colIdx[b] === colIdx[a] && b < rowPtr[rowOf(rowPtr, nodeCount, a) + 1]) {
            b++;
        }
        if (b - a === 1) {
            if (weights[a] !== revWeights[a]) {
                return false;
            }
        } else if (!sameWeightMultiset(weights, revWeights, a, b)) {
            return false;
        }
        a = b;
    }
    return true;
}

/**
 * The row holding arc `a` (binary search on rowPtr).
 * @param rowPtr - the row offsets
 * @param nodeCount - the node count
 * @param a - the arc
 * @returns the row
 */
function rowOf(rowPtr: U32, nodeCount: number, a: number): number {
    let lo = 0;
    let hi = nodeCount;
    while (hi - lo > 1) {
        const mid = (lo + hi) >>> 1;
        if (rowPtr[mid] <= a) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * Whether two weight runs hold the same multiset (sorted scratch copies compared position by position).
 * @param x - one weight array
 * @param y - the other
 * @param start - the first arc of the run
 * @param end - one past the last arc of the run
 * @returns true when the multisets agree
 */
function sameWeightMultiset(x: F32, y: F32, start: number, end: number): boolean {
    const p = x.slice(start, end).sort();
    const q = y.slice(start, end).sort();
    for (let i = 0; i < p.length; i++) {
        if (p[i] !== q[i] && !(Number.isNaN(p[i]) && Number.isNaN(q[i]))) {
            return false;
        }
    }
    return true;
}

// ============================================================ byte accounting

/**
 * The bytes a cached view owns beyond the snapshot's core (aliased arrays count zero), for
 * `byteLength({ views: true })` (design section 7.2).
 * @param snapshot - the snapshot
 * @param name - the view name
 * @param value - the cached value
 * @returns the byte count
 */
export function viewByteLength(snapshot: GraphSnapshot, name: ViewName, value: unknown): number {
    switch (name) {
        case "reverse":
            return value instanceof ReverseAdjacency ? value.ownedByteLength : 0;
        case "coo": {
            const coo = value as CooView;
            return coo.src.byteLength;
        }
        case "edgeList": {
            const list = value as EdgeListView;
            let bytes = list.src.byteLength;
            if (list.dst !== snapshot.colIdx) {
                bytes += list.dst.byteLength;
            }
            if (list.weights !== null && list.weights !== snapshot.weights) {
                bytes += list.weights.byteLength;
            }
            return bytes;
        }
        case "outDegree":
        case "inDegree":
        case "degree":
        case "weightedOutDegree":
        case "weightedInDegree":
        case "weightedDegree":
        case "selfLoopWeight":
        case "selfLoopArcs":
        case "selfLoopsPerNode":
        case "mate":
            return (value as ArrayBufferView).byteLength;
        case "degreeOrder":
        case "reverseDegreeOrder": {
            const order = value as DegreeOrderView;
            return order.perm.byteLength + order.segmentOffsets.byteLength;
        }
        case "totalWeight":
        case "symmetric":
            return 0;
        default: {
            const unknown: never = name;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown view ${String(unknown)}`, { view: unknown });
        }
    }
}

/**
 * The typed arrays a cached view currently holds, keyed by member name, for the checksum records of
 * design section 5.8 (aliased core arrays are included: a write through the alias is a violation
 * too). Scalar views hold nothing.
 * @param snapshot - the snapshot
 * @param name - the view name
 * @param value - the cached value
 * @returns the arrays by member name
 */
export function viewArrays(
    snapshot: GraphSnapshot,
    name: ViewName,
    value: unknown,
): Readonly<Record<string, ArrayBufferView>> {
    switch (name) {
        case "reverse":
            return value instanceof ReverseAdjacency ? value.materialised() : {};
        case "coo": {
            const coo = value as CooView;
            return { src: coo.src };
        }
        case "edgeList": {
            const list = value as EdgeListView;
            const out: Record<string, ArrayBufferView> = { src: list.src };
            if (list.dst !== snapshot.colIdx) {
                out.dst = list.dst;
            }
            if (list.weights !== null && list.weights !== snapshot.weights) {
                out.weights = list.weights;
            }
            return out;
        }
        case "outDegree":
        case "inDegree":
        case "degree":
        case "weightedOutDegree":
        case "weightedInDegree":
        case "weightedDegree":
        case "selfLoopWeight":
        case "selfLoopArcs":
        case "selfLoopsPerNode":
        case "mate":
            return { data: value as ArrayBufferView };
        case "degreeOrder":
        case "reverseDegreeOrder": {
            const order = value as DegreeOrderView;
            return { perm: order.perm, segmentOffsets: order.segmentOffsets };
        }
        case "totalWeight":
        case "symmetric":
            return {};
        default: {
            const unknown: never = name;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown view ${String(unknown)}`, { view: unknown });
        }
    }
}

// ============================================================ boundary helpers (7.5)

/**
 * A zeroed vector of the same class as `like`.
 * @param like - the vector whose class to match
 * @param length - the element count
 * @returns the new vector
 */
function allocVector<T extends NumericVector>(like: T, length: number): T {
    if (like instanceof Float64Array) {
        return new Float64Array(length) as T;
    }
    if (like instanceof Float32Array) {
        return new Float32Array(length) as T;
    }
    if (like instanceof Int32Array) {
        return new Int32Array(length) as T;
    }
    return new Uint32Array(length) as T;
}

/**
 * Reduce a per-arc vector (arcCount entries) to a per-edge vector (edgeCount entries) through
 * `arcToEdge` (design section 7.5): `"first"` takes the value of the arc holding the declared
 * orientation (`edgeToArc[e]`), `"sum"` / `"max"` / `"min"` combine both arcs of an undirected edge.
 * Returns `perArc` itself when the permutation is the identity and no `out` is given.
 * @param snapshot - the snapshot the vector is aligned to
 * @param perArc - the per-arc values, arcCount entries
 * @param reducer - how the arcs of one edge combine
 * @param out - an optional destination of edgeCount entries
 * @returns the per-edge vector
 */
export function foldArcs<T extends NumericVector>(
    snapshot: GraphSnapshot,
    perArc: T,
    reducer: "first" | "sum" | "max" | "min",
    out?: T,
): T {
    const { edgeCount, arcCount } = snapshot;
    assertOneOf("reducer", reducer, ["first", "sum", "max", "min"] as const);
    if (perArc.length !== arcCount) {
        throw new GraphFormatError("E_COLUMN_LENGTH", `perArc has ${perArc.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: perArc.length,
        });
    }
    if (snapshot.flags.arcToEdgeIsIdentity) {
        if (out === undefined) {
            return perArc;
        }
        out.set(perArc);
        return out;
    }
    const result = out ?? allocVector(perArc, edgeCount);
    const { edgeToArc, arcToEdge } = snapshot;
    for (let e = 0; e < edgeCount; e++) {
        result[e] = perArc[edgeToArc[e]];
    }
    switch (reducer) {
        case "first":
            break;
        case "sum":
            for (let a = 0; a < arcCount; a++) {
                const e = arcToEdge[a];
                if (edgeToArc[e] !== a) {
                    result[e] += perArc[a];
                }
            }
            break;
        case "max":
            for (let a = 0; a < arcCount; a++) {
                const e = arcToEdge[a];
                if (perArc[a] > result[e]) {
                    result[e] = perArc[a];
                }
            }
            break;
        case "min":
            for (let a = 0; a < arcCount; a++) {
                const e = arcToEdge[a];
                if (perArc[a] < result[e]) {
                    result[e] = perArc[a];
                }
            }
            break;
        default: {
            const unknown: never = reducer;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown reducer ${String(unknown)}`, { reducer: unknown });
        }
    }
    return result;
}

/**
 * Expand a per-edge vector (edgeCount entries) to a per-arc vector (arcCount entries) through
 * `arcToEdge` (design section 7.5). Returns `perEdge` itself when the permutation is the identity and
 * no `out` is given.
 * @param snapshot - the snapshot the vector is aligned to
 * @param perEdge - the per-edge values, edgeCount entries
 * @param out - an optional destination of arcCount entries
 * @returns the per-arc vector
 */
export function expandEdges<T extends NumericVector>(snapshot: GraphSnapshot, perEdge: T, out?: T): T {
    const { edgeCount, arcCount } = snapshot;
    if (perEdge.length !== edgeCount) {
        throw new GraphFormatError("E_COLUMN_LENGTH", `perEdge has ${perEdge.length} entries, expected ${edgeCount}`, {
            expected: edgeCount,
            found: perEdge.length,
        });
    }
    if (snapshot.flags.arcToEdgeIsIdentity) {
        if (out === undefined) {
            return perEdge;
        }
        out.set(perEdge);
        return out;
    }
    const result = out ?? allocVector(perEdge, arcCount);
    const { arcToEdge } = snapshot;
    for (let a = 0; a < arcCount; a++) {
        result[a] = perEdge[arcToEdge[a]];
    }
    return result;
}
