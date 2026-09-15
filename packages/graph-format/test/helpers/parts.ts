/**
 * Hand-built SnapshotParts for the snapshot tests (the builder does not exist yet). The CSR is
 * constructed NAIVELY here -- arcs sorted with a comparator by (source, target, edge) -- so the
 * counting-sort core builder of src/snapshot/derived.ts is checked against an independent
 * construction, and the flags are recomputed independently of src/snapshot/validate.ts.
 */

import { createTable } from "../../src/columns/table.js";
import { ALIGNMENT } from "../../src/constants.js";
import { identityNodeIdMap, nodeIdMapFromIds } from "../../src/ids/node-id-map.js";
import { createSnapshot, EMPTY_GRAPH_META, GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import {
    type ArenaLayout,
    type AttributeTable,
    type ColumnDeclPatch,
    type ColumnInput,
    type F32,
    type NodeId,
    type SnapshotFlags,
    type TypedArrayData,
    type U32,
} from "../../src/types/index.js";
import { type SnapshotParts } from "../../src/types/internal.js";

/** One edge: [source, target] or [source, target, weight]. */
export type EdgeSpec = readonly [number, number] | readonly [number, number, number];

/** A column given to the helper: a typed array, a ColumnInput, or a JS array with a declaration. */
type ColumnSpec =
    TypedArrayData | ColumnInput | { readonly values: readonly unknown[]; readonly decl?: ColumnDeclPatch };

/** The description of a test graph. */
export interface GraphSpec {
    readonly directed: boolean;
    /** Default: max endpoint + 1 (0 for no edges), or ids.length. */
    readonly nodeCount?: number;
    /** External ids in index order; default identity. */
    readonly ids?: readonly NodeId[];
    readonly edges: readonly EdgeSpec[];
    /** Force the weights array (all ones when no edge carries a weight); default: weighted iff some edge has a weight. */
    readonly weighted?: boolean;
    readonly nodeColumns?: Readonly<Record<string, ColumnSpec>>;
    readonly edgeColumns?: Readonly<Record<string, ColumnSpec>>;
    readonly graphColumns?: Readonly<Record<string, ColumnSpec>>;
    /** Extension tables: name -> row count and columns. */
    readonly extensions?: Readonly<
        Record<string, { readonly rowCount: number; readonly columns: Readonly<Record<string, ColumnSpec>> }>
    >;
    /** Allocate the core in one arena (default true). */
    readonly arena?: boolean;
    /** Record checksums (default false). */
    readonly checksum?: boolean;
    readonly label?: string | null;
}

/** A mutable SnapshotParts so tests can corrupt fields before createSnapshot(). */
export type MutableParts = { -readonly [K in keyof SnapshotParts]: SnapshotParts[K] };

/** The naive CSR of a spec, before it is wrapped. */
interface NaiveCsr {
    readonly nodeCount: number;
    readonly edgeCount: number;
    readonly arcCount: number;
    readonly selfLoopCount: number;
    readonly rowPtr: U32;
    readonly colIdx: U32;
    readonly weights: F32 | null;
    readonly arcToEdge: U32;
    readonly edgeToArc: U32;
    readonly src: U32;
    readonly dst: U32;
    readonly edgeWeights: F32 | null;
}

function roundUp(value: number, multiple: number): number {
    const r = value % multiple;
    return r === 0 ? value : value + multiple - r;
}

/**
 * The node count of a spec: explicit, or ids.length, or max endpoint + 1.
 */
function nodeCountOf(spec: GraphSpec): number {
    if (spec.nodeCount !== undefined) {
        return spec.nodeCount;
    }
    if (spec.ids !== undefined) {
        return spec.ids.length;
    }
    let max = -1;
    for (const [u, v] of spec.edges) {
        max = Math.max(max, u, v);
    }
    return max + 1;
}

/**
 * Build the CSR of a spec with a comparator sort over (source, target, edge).
 */
export function naiveCsr(spec: GraphSpec): NaiveCsr {
    const nodeCount = nodeCountOf(spec);
    const edgeCount = spec.edges.length;
    const weighted = spec.weighted ?? spec.edges.some((e) => e.length === 3);
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    const edgeWeights = weighted ? new Float32Array(edgeCount).fill(1) : null;
    let selfLoopCount = 0;
    spec.edges.forEach((edge, e) => {
        src[e] = edge[0];
        dst[e] = edge[1];
        if (edgeWeights !== null && edge.length === 3) {
            edgeWeights[e] = edge[2];
        }
        if (edge[0] === edge[1]) {
            selfLoopCount++;
        }
    });
    const arcs: { source: number; target: number; edge: number; declared: boolean }[] = [];
    for (let e = 0; e < edgeCount; e++) {
        arcs.push({ source: src[e], target: dst[e], edge: e, declared: true });
        if (!spec.directed && src[e] !== dst[e]) {
            arcs.push({ source: dst[e], target: src[e], edge: e, declared: false });
        }
    }
    arcs.sort((a, b) => a.source - b.source || a.target - b.target || a.edge - b.edge);
    const arcCount = arcs.length;
    const rowPtr = new Uint32Array(nodeCount + 1);
    const colIdx = new Uint32Array(arcCount);
    const arcToEdge = new Uint32Array(arcCount);
    const edgeToArc = new Uint32Array(edgeCount);
    const weights = weighted ? new Float32Array(arcCount) : null;
    arcs.forEach((arc, a) => {
        rowPtr[arc.source + 1]++;
        colIdx[a] = arc.target;
        arcToEdge[a] = arc.edge;
        if (arc.declared) {
            edgeToArc[arc.edge] = a;
        }
        if (weights !== null && edgeWeights !== null) {
            weights[a] = edgeWeights[arc.edge];
        }
    });
    for (let u = 0; u < nodeCount; u++) {
        rowPtr[u + 1] += rowPtr[u];
    }
    return {
        nodeCount,
        edgeCount,
        arcCount,
        selfLoopCount,
        rowPtr,
        colIdx,
        weights,
        arcToEdge,
        edgeToArc,
        src,
        dst,
        edgeWeights,
    };
}

/**
 * The flags of a naive CSR, computed here independently of the package.
 */
export function naiveFlags(directed: boolean, csr: NaiveCsr): SnapshotFlags {
    let multigraph = false;
    for (let u = 0; u < csr.nodeCount; u++) {
        for (let a = csr.rowPtr[u] + 1; a < csr.rowPtr[u + 1]; a++) {
            if (csr.colIdx[a] === csr.colIdx[a - 1]) {
                multigraph = true;
            }
        }
    }
    const identity = directed && csr.arcToEdge.every((e, a) => e === a);
    const w = csr.weights;
    return {
        multigraph,
        hasSelfLoops: csr.selfLoopCount > 0,
        arcToEdgeIsIdentity: identity,
        weighted: w !== null,
        allWeightsOne: w === null || w.every((x) => x === 1),
        nonNegativeWeights: w === null || w.every((x) => x >= 0),
        finiteWeights: w === null || w.every((x) => Number.isFinite(x)),
    };
}

/**
 * Copy the core arrays into one 256-aligned arena (hot to cold), returning the views and the layout.
 */
function intoArena(
    rowPtr: U32,
    colIdx: U32,
    weights: F32 | null,
    arcToEdge: U32 | null,
    edgeToArc: U32 | null,
): { rowPtr: U32; colIdx: U32; weights: F32 | null; arcToEdge: U32 | null; edgeToArc: U32 | null; arena: ArenaLayout } {
    const lengths = [
        rowPtr.byteLength,
        colIdx.byteLength,
        weights === null ? 0 : weights.byteLength,
        arcToEdge === null ? 0 : arcToEdge.byteLength,
        edgeToArc === null ? 0 : edgeToArc.byteLength,
    ];
    const offsets: (number | null)[] = [];
    let cursor = 0;
    for (const length of lengths) {
        if (length === 0) {
            offsets.push(null);
            continue;
        }
        cursor = roundUp(cursor, ALIGNMENT);
        offsets.push(cursor);
        cursor += length;
    }
    const buffer = new ArrayBuffer(cursor);
    const seg = (i: number): { byteOffset: number; byteLength: number } | null =>
        offsets[i] === null ? null : { byteOffset: offsets[i], byteLength: lengths[i] };
    const placeU32 = (i: number, source: U32): U32 => {
        const offset = offsets[i];
        if (offset === null) {
            return new Uint32Array(0);
        }
        const view = new Uint32Array(buffer, offset, source.length);
        view.set(source);
        return view;
    };
    let hot = 0;
    for (const i of [0, 1, 2]) {
        const s = seg(i);
        if (s !== null) {
            hot = s.byteOffset + s.byteLength;
        }
    }
    let arenaWeights: F32 | null = null;
    if (weights !== null) {
        const offset = offsets[2];
        arenaWeights = offset === null ? new Float32Array(0) : new Float32Array(buffer, offset, weights.length);
        arenaWeights.set(weights);
    }
    const arena: ArenaLayout = {
        buffer,
        byteOffset: 0,
        byteLength: cursor,
        alignment: 256,
        segments: { rowPtr: seg(0), colIdx: seg(1), weights: seg(2), arcToEdge: seg(3), edgeToArc: seg(4) },
        hotByteLength: hot,
    };
    return {
        rowPtr: placeU32(0, rowPtr),
        colIdx: placeU32(1, colIdx),
        weights: arenaWeights,
        arcToEdge: arcToEdge === null ? null : placeU32(3, arcToEdge),
        edgeToArc: edgeToArc === null ? null : placeU32(4, edgeToArc),
        arena,
    };
}

function isTypedArrayData(x: unknown): x is TypedArrayData {
    return (
        x instanceof Uint32Array ||
        x instanceof Int32Array ||
        x instanceof Float32Array ||
        x instanceof Float64Array ||
        x instanceof Uint8Array
    );
}

/**
 * A table with the given columns attached through set().
 */
function tableOf(
    domain: "node" | "edge" | "graph" | "extension",
    rowCount: number,
    columns: Readonly<Record<string, ColumnSpec>> = {},
): AttributeTable {
    const table = createTable(domain, rowCount);
    for (const [name, spec] of Object.entries(columns)) {
        if (isTypedArrayData(spec)) {
            table.set(name, spec);
        } else if ("values" in spec) {
            table.set(name, spec.values, spec.decl);
        } else {
            table.set(name, spec.data, spec.decl);
        }
    }
    return table;
}

/**
 * Build mutable SnapshotParts for a spec: identity permutations are passed as null with the flag set
 * (the rule of SnapshotParts), everything else materialised.
 */
export function makeParts(spec: GraphSpec): MutableParts {
    const csr = naiveCsr(spec);
    const flags = naiveFlags(spec.directed, csr);
    const identity = flags.arcToEdgeIsIdentity;
    let core = {
        rowPtr: csr.rowPtr,
        colIdx: csr.colIdx,
        weights: csr.weights,
        arcToEdge: identity ? null : csr.arcToEdge,
        edgeToArc: identity ? null : csr.edgeToArc,
        arena: null as ArenaLayout | null,
    };
    if (spec.arena !== false) {
        core = intoArena(core.rowPtr, core.colIdx, core.weights, core.arcToEdge, core.edgeToArc);
    }
    const extensions = new Map<string, AttributeTable>();
    for (const [name, ext] of Object.entries(spec.extensions ?? {})) {
        extensions.set(name, tableOf("extension", ext.rowCount, ext.columns));
    }
    return {
        label: spec.label ?? null,
        serial: null,
        directed: spec.directed,
        nodeCount: csr.nodeCount,
        edgeCount: csr.edgeCount,
        arcCount: csr.arcCount,
        selfLoopCount: csr.selfLoopCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags,
        ids: spec.ids === undefined ? identityNodeIdMap(csr.nodeCount) : nodeIdMapFromIds(spec.ids),
        nodes: tableOf("node", csr.nodeCount, spec.nodeColumns),
        edges: tableOf("edge", csr.edgeCount, spec.edgeColumns),
        graph: tableOf("graph", 1, spec.graphColumns),
        extensions,
        meta: EMPTY_GRAPH_META,
        arena: core.arena,
        checksum: spec.checksum ?? false,
    };
}

/**
 * Build a snapshot for a spec.
 */
export function makeSnapshot(spec: GraphSpec): GraphSnapshot {
    return createSnapshot(makeParts(spec));
}

/**
 * The per-edge weight of a spec edge (1 when omitted).
 */
export function weightOf(edge: EdgeSpec): number {
    return edge.length === 3 ? edge[2] : 1;
}

/**
 * The neighbour multiset of a node as a sorted array of [target, edge] pairs, from the spec.
 */
export function naiveOutArcs(spec: GraphSpec, u: number): [number, number][] {
    const out: [number, number][] = [];
    spec.edges.forEach((edge, e) => {
        if (edge[0] === u) {
            out.push([edge[1], e]);
        }
        if (!spec.directed && edge[1] === u && edge[0] !== u) {
            out.push([edge[0], e]);
        }
    });
    out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return out;
}

/** Zachary's karate club (34 nodes, 78 edges), the classic fixture. */
export const KARATE_EDGES: readonly EdgeSpec[] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [0, 17],
    [0, 19],
    [0, 21],
    [0, 31],
    [1, 2],
    [1, 3],
    [1, 7],
    [1, 13],
    [1, 17],
    [1, 19],
    [1, 21],
    [1, 30],
    [2, 3],
    [2, 7],
    [2, 8],
    [2, 9],
    [2, 13],
    [2, 27],
    [2, 28],
    [2, 32],
    [3, 7],
    [3, 12],
    [3, 13],
    [4, 6],
    [4, 10],
    [5, 6],
    [5, 10],
    [5, 16],
    [6, 16],
    [8, 30],
    [8, 32],
    [8, 33],
    [9, 33],
    [13, 33],
    [14, 32],
    [14, 33],
    [15, 32],
    [15, 33],
    [18, 32],
    [18, 33],
    [19, 33],
    [20, 32],
    [20, 33],
    [22, 32],
    [22, 33],
    [23, 25],
    [23, 27],
    [23, 29],
    [23, 32],
    [23, 33],
    [24, 25],
    [24, 27],
    [24, 31],
    [25, 31],
    [26, 29],
    [26, 33],
    [27, 33],
    [28, 31],
    [28, 33],
    [29, 32],
    [29, 33],
    [30, 32],
    [30, 33],
    [31, 32],
    [31, 33],
    [32, 33],
];

/**
 * A w x h grid (4-neighbour), directed or undirected, row-major node indices.
 */
export function gridEdges(w: number, h: number): EdgeSpec[] {
    const edges: EdgeSpec[] = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const u = y * w + x;
            if (x + 1 < w) {
                edges.push([u, u + 1]);
            }
            if (y + 1 < h) {
                edges.push([u, u + w]);
            }
        }
    }
    return edges;
}
