/**
 * Content checks of the views a wire manifest carries (`includeViews`, design sections 9.1 and 9.5),
 * run at the "structure" level before a carried view is installed on the receiver. Every view is a
 * pure function of the core (design section 7.2), so each check is exact: the index views are
 * compared with what the core says arc by arc in O(n + m) without allocating a second copy, and the
 * four f64 views are compared with the snapshot's own computation. A view that disagrees is
 * E_INVALID_SNAPSHOT with `details.invariant` "I2" (a corrupt index array) or "I17" (a corrupt
 * value array) and `details.view` naming it. At "none" nothing is checked (the manifest is trusted,
 * design section 9.5); at "full" carried views are never installed.
 */

import { type GraphFormatError } from "../errors.js";
import { type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import { invariantViolation } from "../snapshot/validate.js";
import { DEGREE_TIER_HIGH, DEGREE_TIER_MID, rowLengths } from "../snapshot/views.js";
import { type F32, type F64, type ReverseView, type U32 } from "../types/index.js";

/** The per-arc facts every index-view check reads, computed once per manifest. */
export class CoreFacts {
    /** The row (source) of every arc. */
    readonly arcRow: U32;
    /** Loop arcs per node. */
    readonly loops: U32;
    /** In-arcs per node (directed only; the out-degree when undirected). */
    readonly inDegree: U32;

    /**
     * Expand the core once.
     * @param snapshot - the snapshot
     */
    constructor(snapshot: GraphSnapshot) {
        const { nodeCount, arcCount, rowPtr, colIdx, directed } = snapshot;
        const arcRow = new Uint32Array(arcCount);
        const loops = new Uint32Array(nodeCount);
        const inDegree = directed ? new Uint32Array(nodeCount) : rowLengths(rowPtr, nodeCount);
        for (let u = 0; u < nodeCount; u++) {
            const end = rowPtr[u + 1];
            for (let a = rowPtr[u]; a < end; a++) {
                arcRow[a] = u;
                const v = colIdx[a];
                if (v === u) {
                    loops[u]++;
                }
                if (directed) {
                    inDegree[v]++;
                }
            }
        }
        this.arcRow = arcRow;
        this.loops = loops;
        this.inDegree = inDegree;
    }

    /**
     * The exclusive prefix sum of the in-degrees (the reverse rowPtr).
     * @returns nodeCount + 1 offsets
     */
    inDegreePrefix(): U32 {
        const n = this.inDegree.length;
        const out = new Uint32Array(n + 1);
        for (let u = 0; u < n; u++) {
            out[u + 1] = out[u] + this.inDegree[u];
        }
        return out;
    }
}

/**
 * The error for a carried view that disagrees with the core.
 * @param view - the view name
 * @param member - the member array
 * @param index - the offending index
 * @param invariant - "I2" for an index array, "I17" for a value array
 * @returns the error
 */
function viewError(view: string, member: string, index: number, invariant: "I2" | "I17"): GraphFormatError {
    return invariantViolation(invariant, `carried view ${view}.${member} disagrees with the core at ${index}`, {
        view,
        member,
        index,
        reason: "view",
    });
}

/**
 * Compare two arrays element by element (NaN equal to NaN).
 * @param view - the view name
 * @param member - the member name
 * @param found - the carried array
 * @param expected - the truth
 * @param invariant - the invariant to report
 */
function expectSame(
    view: string,
    member: string,
    found: ArrayLike<number>,
    expected: ArrayLike<number>,
    invariant: "I2" | "I17",
): void {
    for (let i = 0; i < expected.length; i++) {
        if (!Object.is(found[i], expected[i]) && found[i] !== expected[i]) {
            throw viewError(view, member, i, invariant);
        }
    }
}

/**
 * Check that an array is a permutation of 0..length-1.
 * @param view - the view name
 * @param member - the member name
 * @param perm - the array
 */
function expectPermutation(view: string, member: string, perm: U32): void {
    const seen = new Uint8Array(perm.length);
    for (let i = 0; i < perm.length; i++) {
        const p = perm[i];
        if (p >= perm.length || seen[p] === 1) {
            throw viewError(view, member, i, "I2");
        }
        seen[p] = 1;
    }
}

/**
 * outDegree: the row lengths.
 * @param snapshot - the snapshot
 * @param data - the carried array
 */
export function checkOutDegree(snapshot: GraphSnapshot, data: U32): void {
    expectSame("outDegree", "data", data, rowLengths(snapshot.rowPtr, snapshot.nodeCount), "I2");
}

/**
 * inDegree (directed): the in-arc counts.
 * @param facts - the core facts
 * @param data - the carried array
 */
export function checkInDegree(facts: CoreFacts, data: U32): void {
    expectSame("inDegree", "data", data, facts.inDegree, "I2");
}

/**
 * degree: outDegree + inDegree (directed) or outDegree + loops (undirected), design section 3.4.
 * @param snapshot - the snapshot
 * @param facts - the core facts
 * @param data - the carried array
 */
export function checkDegree(snapshot: GraphSnapshot, facts: CoreFacts, data: U32): void {
    const out = rowLengths(snapshot.rowPtr, snapshot.nodeCount);
    const add = snapshot.directed ? facts.inDegree : facts.loops;
    for (let u = 0; u < out.length; u++) {
        if (data[u] !== out[u] + add[u]) {
            throw viewError("degree", "data", u, "I2");
        }
    }
}

/**
 * selfLoopsPerNode: the loop arcs per row.
 * @param facts - the core facts
 * @param data - the carried array
 */
export function checkSelfLoopsPerNode(facts: CoreFacts, data: U32): void {
    expectSame("selfLoopsPerNode", "data", data, facts.loops, "I2");
}

/**
 * selfLoopArcs: ascending arcs, each a loop; with the length fixed to selfLoopCount that is exact.
 * @param snapshot - the snapshot
 * @param facts - the core facts
 * @param data - the carried array
 */
export function checkSelfLoopArcs(snapshot: GraphSnapshot, facts: CoreFacts, data: U32): void {
    const { colIdx, arcCount } = snapshot;
    for (let i = 0; i < data.length; i++) {
        const a = data[i];
        if (a >= arcCount || colIdx[a] !== facts.arcRow[a] || (i > 0 && a <= data[i - 1])) {
            throw viewError("selfLoopArcs", "data", i, "I2");
        }
    }
}

/**
 * coo.src: the row of every arc.
 * @param facts - the core facts
 * @param src - the carried array
 */
export function checkCooSrc(facts: CoreFacts, src: U32): void {
    expectSame("coo", "src", src, facts.arcRow, "I2");
}

/**
 * edgeList: `src[e]` is the row of `edgeToArc[e]`, `dst[e]` its target, `weights[e]` its weight.
 * @param snapshot - the snapshot
 * @param facts - the core facts
 * @param src - the carried sources
 * @param dst - the carried targets, or null when colIdx is aliased
 * @param weights - the carried weights, or null when aliased or unweighted
 */
export function checkEdgeList(
    snapshot: GraphSnapshot,
    facts: CoreFacts,
    src: U32,
    dst: U32 | null,
    weights: F32 | null,
): void {
    const { edgeCount, edgeToArc, colIdx, arcCount } = snapshot;
    const forward = snapshot.weights;
    for (let e = 0; e < edgeCount; e++) {
        const a = edgeToArc[e];
        if (a >= arcCount || src[e] !== facts.arcRow[a]) {
            throw viewError("edgeList", "src", e, "I2");
        }
        if (dst !== null && dst[e] !== colIdx[a]) {
            throw viewError("edgeList", "dst", e, "I2");
        }
        if (weights !== null && forward !== null && !Object.is(weights[e], forward[a])) {
            throw viewError("edgeList", "weights", e, "I17");
        }
    }
}

/**
 * mate (undirected): an involution pairing the two arcs of every non-loop edge (invariant I7) and
 * fixing every loop arc.
 * @param snapshot - the snapshot
 * @param facts - the core facts
 * @param mate - the carried array
 */
export function checkMate(snapshot: GraphSnapshot, facts: CoreFacts, mate: U32): void {
    const { arcCount, arcToEdge, colIdx } = snapshot;
    for (let a = 0; a < arcCount; a++) {
        const m = mate[a];
        const loop = colIdx[a] === facts.arcRow[a];
        if (m >= arcCount || mate[m] !== a || arcToEdge[m] !== arcToEdge[a] || (m === a) !== loop) {
            throw viewError("mate", "data", a, "I2");
        }
    }
}

/**
 * degreeOrder / reverseDegreeOrder: a permutation ordered by non-increasing degree with ties in
 * ascending node index, and the cuGraph tier offsets of design section 7.2.
 * @param name - the view name
 * @param degree - the degree of every node the order sorts by
 * @param perm - the carried permutation
 * @param segmentOffsets - the carried tier offsets
 */
export function checkDegreeOrder(
    name: "degreeOrder" | "reverseDegreeOrder",
    degree: U32,
    perm: U32,
    segmentOffsets: U32,
): void {
    expectPermutation(name, "perm", perm);
    let hi = 0;
    let mid = 0;
    let low = 0;
    for (let i = 0; i < perm.length; i++) {
        const d = degree[perm[i]];
        if (i > 0) {
            const previous = degree[perm[i - 1]];
            if (d > previous || (d === previous && perm[i] < perm[i - 1])) {
                throw viewError(name, "perm", i, "I2");
            }
        }
        if (d >= DEGREE_TIER_HIGH) {
            hi++;
        } else if (d >= DEGREE_TIER_MID) {
            mid++;
        } else if (d >= 1) {
            low++;
        }
    }
    const expected = [0, hi, hi + mid, hi + mid + low, perm.length];
    expectSame(name, "segmentOffsets", segmentOffsets, expected, "I2");
}

/**
 * reverse (directed): the in-adjacency with rows sorted by source and ties by forward arc, `fwdArc`
 * a permutation of the arcs, `colIdx` the sources, `weights` the forward weights gathered.
 * @param snapshot - the snapshot
 * @param facts - the core facts
 * @param reverse - the carried view
 */
export function checkReverse(snapshot: GraphSnapshot, facts: CoreFacts, reverse: ReverseView): void {
    const { nodeCount, arcCount, colIdx } = snapshot;
    const forward = snapshot.weights;
    const { rowPtr, fwdArc } = reverse;
    const revIdx = reverse.colIdx;
    const revWeights = reverse.weights;
    if (rowPtr[0] !== 0 || rowPtr[nodeCount] !== arcCount) {
        throw viewError("reverse", "rowPtr", 0, "I2");
    }
    expectSame("reverse", "rowPtr", rowPtr, facts.inDegreePrefix(), "I2");
    expectPermutation("reverse", "fwdArc", fwdArc);
    for (let v = 0; v < nodeCount; v++) {
        const end = rowPtr[v + 1];
        for (let k = rowPtr[v]; k < end; k++) {
            const a = fwdArc[k];
            const source = facts.arcRow[a];
            if (colIdx[a] !== v || revIdx[k] !== source) {
                throw viewError("reverse", "colIdx", k, "I2");
            }
            if (k > rowPtr[v]) {
                const previous = fwdArc[k - 1];
                const previousSource = facts.arcRow[previous];
                if (source < previousSource || (source === previousSource && a <= previous)) {
                    throw viewError("reverse", "fwdArc", k, "I2");
                }
            }
            if (revWeights !== null && forward !== null && !Object.is(revWeights[k], forward[a])) {
                throw viewError("reverse", "weights", k, "I17");
            }
        }
    }
}

/**
 * An f64 view: compared with the snapshot's own computation.
 * @param name - the view name
 * @param found - the carried array
 * @param expected - the computed truth
 */
export function checkF64View(name: string, found: F64, expected: F64): void {
    expectSame(name, "data", found, expected, "I17");
}
