/**
 * The frozen snapshot and its side structures (design sections 3, 4, 7, 8.1, 9, 10 and 12.2): the id
 * map, the flags, the arena layout, the views, the derived graphs, the GraphSnapshot contract itself
 * and the inputs of the snapshot factories.
 *
 * GraphSnapshotContract and NodeIdMapContract are the instance contracts of the GraphSnapshot and
 * NodeIdMap classes implemented in src/snapshot/graph-snapshot.ts and src/ids/node-id-map.ts (design
 * section 12.1 makes them classes so the JSDoc-on-method lint applies). The public names
 * GraphSnapshot and NodeIdMap are the classes themselves, re-exported type-only from here so every
 * type on this surface (DerivedGraph.snapshot, SnapshotParts.ids, ...) names the class the barrel
 * exports. The imports are type-only, so the import graph stays free of runtime cycles.
 */

import { type SNAPSHOT_BRAND } from "../constants.js";
import { type NodeIdMap } from "../ids/node-id-map.js";
import { type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import {
    type AttributeTable,
    type ColumnDecl,
    type ColumnInput,
    type ColumnReducer,
    type EdgeId,
    type EdgeMask,
    type F32,
    type F64,
    type GraphMeta,
    type GraphMetaPatch,
    type IdCoercion,
    type Loose,
    type NodeId,
    type NodeMask,
    type TypedArrayData,
    type U8,
    type U32,
    type ValidationLevel,
    type WeightReducer,
} from "./columns.js";
import { type ToBytesOptions, type ToWireOptions, type WireSnapshot } from "./wire.js";

export type { GraphSnapshot, NodeIdMap };

// ============================================================ id map

/**
 * Storage kind of a NodeIdMap, chosen at freeze by inspecting the ids once (design section 4.2):
 * "identity" (id === index + offset, zero bytes), "dense" (distinct integers with maxId + 1 <= 2n),
 * "numeric" (any numbers), "string" (all strings) or "mixed".
 */
export type NodeIdMapKind = "identity" | "dense" | "numeric" | "string" | "mixed";

/**
 * Bijection between node ids and node indices for one snapshot, kept outside the CSR (invariant
 * I11: ids.indexOf(ids.idOf(i)) === i for every i, size === nodeCount, SameValueZero equality, no
 * NaN). Immutable. Lookups by id return INVALID_INDEX on a miss rather than -1 or undefined so the
 * sentinel is the same one used in every u32 result vector (decision C9).
 *
 * Instance contract of the NodeIdMap class in src/ids/node-id-map.ts, which implements it; the
 * public barrel exports the class and every public type names the class.
 */
export interface NodeIdMapContract extends Iterable<NodeId> {
    /** The storage kind (design section 4.2). */
    readonly kind: NodeIdMapKind;
    /** Number of ids; equals nodeCount. */
    readonly size: number;
    /** identity only: id === index + offset (0 or 1 in practice); 0 for every other kind. */
    readonly offset: number;
    /**
     * The id of a node index.
     * @param index - the node index
     * @returns the id; E_INDEX_RANGE when index >= size
     */
    idOf(index: number): NodeId;
    /**
     * Total lookup by id.
     * @param id - the node id
     * @returns the node index, or INVALID_INDEX when absent
     */
    indexOf(id: NodeId): number;
    /**
     * Whether an id is present.
     * @param id - the node id
     * @returns true when present
     */
    has(id: NodeId): boolean;
    /**
     * Checked lookup by id, for algorithms that today throw "node not found".
     * @param id - the node id
     * @returns the node index; E_UNKNOWN_NODE when absent
     */
    requireIndex(id: NodeId): number;
    /**
     * Bulk lookup.
     * @param ids - the ids to resolve
     * @param onMissing - "invalid" (default) writes INVALID_INDEX for a miss; "throw" raises E_UNKNOWN_NODE
     * @returns a fresh U32 of indices in input order
     */
    indicesOf(ids: Iterable<NodeId>, onMissing?: "invalid" | "throw"): U32;
    /**
     * Bulk decode of a range of ids in one pass.
     * @param start - first index (default 0)
     * @param end - one past the last index (default size)
     * @returns the ids in index order
     */
    idsSlice(start?: number, end?: number): NodeId[];
    /**
     * Every id in index order.
     * @returns a fresh array
     */
    toArray(): NodeId[];
    /**
     * Iterate the ids in index order.
     * @returns an iterator over the ids
     */
    [Symbol.iterator](): IterableIterator<NodeId>;
    /**
     * Key an index-aligned result vector by id (boundary helper, decision C11).
     * @param values - one value per node index
     * @returns a Map from id to value
     */
    toMap<T>(values: ArrayLike<T>): Map<NodeId, T>;
    /**
     * Key an index-aligned result vector by String(id), for legacy Map<string, T> result shapes.
     * @param values - one value per node index
     * @returns a Map from String(id) to value
     */
    toStringMap<T>(values: ArrayLike<T>): Map<string, T>;
    /**
     * Key an index-aligned result vector by String(id), for legacy record result shapes only.
     * @param values - one value per node index
     * @returns a record from String(id) to value
     */
    toRecord<T>(values: ArrayLike<T>): Record<string, T>;
    /**
     * Iterate [id, value] pairs of an index-aligned result vector.
     * @param values - one value per node index
     * @returns an iterator over the pairs in index order
     */
    entries<T>(values: ArrayLike<T>): IterableIterator<[NodeId, T]>;
    /**
     * String(idOf(i)) -> i, built lazily once, for legacy string-typed id parameters.
     * @returns the read-only string index
     */
    stringIndex(): ReadonlyMap<string, number>;
    /**
     * Bytes of typed storage; excludes the reverse Map and JS strings.
     * @returns the byte count
     */
    byteLength(): number;
}

// ============================================================ flags, arena, views

/**
 * Flags kernels branch on (design section 3.8). Each is defined by a predicate over the arrays,
 * computed at freeze or by validation, never guessed; there is no flag whose value may be "unknown"
 * (invariant I9). The three weight flags describe the f32 ARC ARRAY (`weights`): when an f64
 * role-"weight" shadow column is kept (design section 3.7) its exact values can differ from the
 * rounded arc values (1 + 2^-30 rounds to 1), so a consumer that substitutes the shadow for the arc
 * weights must not branch on these flags.
 */
export interface SnapshotFlags {
    /** Some row contains two arcs with equal colIdx (parallel edges). */
    readonly multigraph: boolean;
    /** selfLoopCount > 0. */
    readonly hasSelfLoops: boolean;
    /**
     * directed && arcToEdge[a] === a for all a (and edgeToArc is the identity). Always false when !directed, even when
     * arcCount === edgeCount.
     */
    readonly arcToEdgeIsIdentity: boolean;
    /** weights !== null. */
    readonly weighted: boolean;
    /** weights === null, or every value === 1. Lets SSSP degrade to BFS. */
    readonly allWeightsOne: boolean;
    /** weights === null, or every value >= 0. Dijkstra / delta-stepping legal. */
    readonly nonNegativeWeights: boolean;
    /** weights === null, or every value is finite. */
    readonly finiteWeights: boolean;
}

/** Flag claims a caller passes to fromCsr(); verified unless validate is "none" (design section 8.1). */
export type FlagClaims = Loose<SnapshotFlags>;

/** The five core arrays, in the hot-to-cold order they occupy the arena (design section 10.3). */
export type CoreArrayName = "rowPtr" | "colIdx" | "weights" | "arcToEdge" | "edgeToArc";

/** One core array's location inside the arena buffer, in absolute bytes (design section 10.3). */
export interface ArenaSegment {
    /** Absolute byte offset in ArenaLayout.buffer; a multiple of 256 relative to ArenaLayout.byteOffset. */
    readonly byteOffset: number;
    /** Byte length of the array (unpadded). */
    readonly byteLength: number;
}

/**
 * One ArrayBuffer holding the core arrays at 256-byte-aligned offsets, hot to cold: rowPtr, colIdx,
 * weights, arcToEdge, edgeToArc (design section 10.3, invariant I10). A GPU package uploads the first
 * hotByteLength bytes in one writeBuffer call and binds each segment as a storage-buffer window. A
 * zero-length array, an absent weights array and an identity permutation have a null segment and
 * are never in the arena.
 */
export interface ArenaLayout {
    /** The backing buffer. */
    readonly buffer: ArrayBuffer;
    /** Start of the arena inside buffer: 0 for builder output; bytes.byteOffset + B for a container. */
    readonly byteOffset: number;
    /** Total padded length of the arena. */
    readonly byteLength: number;
    /** Segment alignment in bytes. */
    readonly alignment: 256;
    /** Absolute offsets in buffer per core array; null = absent, zero-length, or identity (never in the arena). */
    readonly segments: Readonly<Record<CoreArrayName, ArenaSegment | null>>;
    /**
     * End of the weights segment (or colIdx when unweighted) relative to byteOffset: the prefix a traversal kernel
     * needs.
     */
    readonly hotByteLength: number;
}

/**
 * Names of the lazily computed, cached views (design section 7.2), as accepted by prepare(),
 * cachedViews() and the wire includeViews option. "degreeOrder" and "reverseDegreeOrder" are the two
 * cached results of degreeOrder({ of }); "symmetric" is the result of isSymmetric(). The scalar
 * views "totalWeight" and "symmetric" are ignored by includeViews and always recomputed.
 */
export type ViewName =
    | "reverse"
    | "coo"
    | "edgeList"
    | "outDegree"
    | "inDegree"
    | "degree"
    | "weightedOutDegree"
    | "weightedInDegree"
    | "weightedDegree"
    | "selfLoopWeight"
    | "totalWeight"
    | "selfLoopArcs"
    | "selfLoopsPerNode"
    | "mate"
    | "degreeOrder"
    | "reverseDegreeOrder"
    | "symmetric";

/** Options of degreeOrder(): which adjacency's out-degree orders the nodes (design section 7.2). */
export interface DegreeOrderOptions {
    /** "forward" (default) orders by rowPtr; "reverse" by reverse().rowPtr, i.e. the in-degree, for pull kernels. */
    readonly of?: "forward" | "reverse" | undefined;
}

/**
 * The structural row-walking interface implemented by GraphSnapshot and by ReverseView (design
 * section 7.2), so an algorithm that only walks rows takes either without a wrapper: an
 * in-neighbour BFS is the out-neighbour BFS over reverse(). Rows are sorted by target with ties in
 * ascending arcToEdge order (invariant I4); no INVALID_INDEX appears in any array (I2).
 */
export interface AdjacencyView {
    /** Whether the graph is directed (design section 3.6). */
    readonly directed: boolean;
    /** Number of nodes n. */
    readonly nodeCount: number;
    /** Number of arcs: colIdx.length === rowPtr[nodeCount] (invariant I1). */
    readonly arcCount: number;
    /** nodeCount + 1 row offsets; rowPtr[0] === 0, non-decreasing (invariant I1). */
    readonly rowPtr: U32;
    /** Target node index of every arc, sorted within each row (invariants I2, I4); length 0 when arcCount === 0. */
    readonly colIdx: U32;
    /** Logical edge of every arc (invariant I5). */
    readonly arcToEdge: U32;
    /** Per-arc f32 weights, or null when every weight is 1 (design section 3.7, invariant I8). */
    readonly weights: F32 | null;
}

/**
 * The in-adjacency of a snapshot (design section 7.2): rows sorted by source, weights gathered when
 * weighted. For an undirected snapshot the forward arrays themselves are returned (invariant I7)
 * with an identity fwdArc, and arcToEdge is the very same array as the snapshot's.
 */
export interface ReverseView extends AdjacencyView {
    /** Reverse arc k -> forward arc index; the identity for undirected snapshots (materialised lazily). */
    readonly fwdArc: U32;
}

/** Per-arc COO form: src is the only new array; dst aliases colIdx (design section 7.2). */
export interface CooView {
    /** Source node index of every arc (the row containing it). */
    readonly src: U32;
    /** Target node index of every arc; aliases colIdx. */
    readonly dst: U32;
    /** Logical edge of every arc; aliases the snapshot's arcToEdge. */
    readonly arcToEdge: U32;
    /** Per-arc weights; aliases the snapshot's weights. */
    readonly weights: F32 | null;
}

/**
 * Every logical edge once, in declared orientation (design section 7.2): the binding an
 * each-edge-once edge-parallel kernel uses on directed and undirected snapshots alike.
 */
export interface EdgeListView {
    /** Declared source of every logical edge. */
    readonly src: U32;
    /** Declared target of every logical edge. */
    readonly dst: U32;
    /** The arc holding the declared orientation; aliases edgeToArc. */
    readonly arc: U32;
    /**
     * Per-edge weights gathered through edgeToArc (aliased when the permutation is the identity); null when unweighted.
     */
    readonly weights: F32 | null;
}

/**
 * Nodes permuted by descending out-degree of the chosen adjacency with cuGraph's tier thresholds
 * 1024 / 32 / 1 (design section 7.2): perm is the GPU load-balancing binding, segmentOffsets =
 * [0, hiEnd, midEnd, lowEnd, n] is read on the CPU to size the three dispatches.
 */
export interface DegreeOrderView {
    /** Node indices in descending degree order; length n. */
    readonly perm: U32;
    /** The five tier boundaries [0, hiEnd, midEnd, lowEnd, n]. */
    readonly segmentOffsets: U32;
}

// ============================================================ derived graphs

/**
 * A new snapshot produced from an existing one by a structural mapping the caller specifies, plus
 * index maps back to the source (design section 7.3). Never cached by the format. Every map is null
 * when the corresponding index space is unchanged; edgeRemap maps every merged edge to its survivor
 * so a per-edge result of the derived graph writes back onto source edges with
 * `out[e] = vec[edgeRemap[e]]`, and edgeRemap[edgeOrigin[d]] === d for every derived edge d.
 */
export interface DerivedGraph {
    /** The derived snapshot; `this` for the identity cases (transpose / toUndirected of an undirected snapshot). */
    readonly snapshot: GraphSnapshot;
    /**
     * New node index -> source node index (contract: the LOWEST source index of the block); null when the node space is
     * unchanged.
     */
    readonly nodeOrigin: U32 | null;
    /**
     * New edge index -> source edge index (a merged edge: its SURVIVOR's source index, never INVALID_INDEX); null when
     * unchanged.
     */
    readonly edgeOrigin: U32 | null;
    /** Source node index -> new index (contract: the block) or INVALID_INDEX (dropped); null when unchanged. */
    readonly nodeRemap: U32 | null;
    /**
     * Source edge index -> new index; a merged / collapsed edge maps to its SURVIVOR, a dropped edge to INVALID_INDEX;
     * null when unchanged.
     */
    readonly edgeRemap: U32 | null;
    /** contract only: source nodes per new node; null otherwise. */
    readonly blockSizes: U32 | null;
    /** How many source edges were dropped and how many merged. */
    readonly report: { readonly droppedEdges: number; readonly mergedEdges: number };
}

/** Options of toUndirected() (design section 7.3). */
export interface ToUndirectedOptions {
    /** Keep only pairs present in both directions (cuGraph symmetrize(reciprocal)). */
    readonly reciprocal?: boolean | undefined;
    /** How the weights of a collapsed reciprocal pair combine; default "first" (the lower index's weight). */
    readonly weights?: WeightReducer | undefined;
}

/** Options of simplified() (design section 7.3): one edge per (u, v) group, survivor = lowest index. */
export interface SimplifyOptions {
    /** How the weights of a parallel group combine; default "first". */
    readonly weights?: WeightReducer | undefined;
    /** Whether self-loops survive; default "keep". */
    readonly selfLoops?: "keep" | "drop" | undefined;
    /** Per-column reducers for the other edge columns; default: the survivor's row. */
    readonly edgeReducers?: Readonly<Record<string, ColumnReducer>> | undefined;
}

/**
 * Options of contract(partition) (design section 7.3): the Leiden / Louvain aggregation and
 * condensationGraph primitive. Semantics are at the logical-edge level: an intra-block edge becomes
 * one self-loop with weight w, an inter-block edge one edge between the blocks.
 */
export interface ContractOptions {
    /** How merged edge weights combine; default "sum" (on an unweighted source "sum" yields multiplicities). */
    readonly weights?: WeightReducer | undefined;
    /** Whether intra-block edges become self-loops; default "keep". */
    readonly selfLoops?: "keep" | "drop" | undefined;
    /** Whether parallel inter-block edges merge; default "merge". */
    readonly parallel?: "merge" | "keep" | undefined;
    /** Node columns to keep and how to reduce them; default: drop node columns. */
    readonly nodeReducers?: Readonly<Record<string, ColumnReducer>> | undefined;
    /** Edge columns to keep and how to reduce them; default: drop edge columns. */
    readonly edgeReducers?: Readonly<Record<string, ColumnReducer>> | undefined;
}

// ============================================================ checks and memory

/** Options of validate() (design section 11.4). */
export interface ValidateOptions {
    /** "structure" or "full" per the level table of design section 9.5; default "full". */
    readonly level?: "structure" | "full" | undefined;
    /**
     * Compare the FNV-1a checksums recorded by freeze({ checksum: true }); E_INVALID_SNAPSHOT (details.reason
     * "no-checksum") when none were recorded.
     */
    readonly checksum?: boolean | undefined;
}

/** Options of byteLength(): which side structures to include beyond the core (design section 7.2). */
export interface ByteLengthOptions {
    /** Include cached views. */
    readonly views?: boolean | undefined;
    /** Include attribute columns. */
    readonly columns?: boolean | undefined;
    /** Include the id map's typed storage. */
    readonly ids?: boolean | undefined;
}

// ============================================================ snapshot

/**
 * The frozen graph (design section 3): a CSR core over 4-byte typed arrays, the id map, the
 * attribute tables, the flags and a per-instance view cache. Topology, counts, flags, id map and
 * immutable columns never change after freeze() returns (invariant I17); the column SET of nodes /
 * edges / graph and the CONTENTS of columns declared mutable are mutable side tables (design section
 * 5.8). Every view is a pure function of the core, memoised once and SHARED: writing into a view is
 * a contract violation; call .slice() for scratch. Index-taking queries are total for in-range
 * arguments and unchecked otherwise (design section 11.1); every core accessor throws E_DETACHED
 * after a consuming transfer.
 *
 * Instance contract of the GraphSnapshot class in src/snapshot/graph-snapshot.ts, which implements
 * it; the public barrel exports the class and every public type names the class. isGraphSnapshot()
 * recognises instances structurally through SNAPSHOT_BRAND and formatVersion, never instanceof.
 */
export interface GraphSnapshotContract extends AdjacencyView {
    /** Symbol.for brand read by isGraphSnapshot() (design section 7.5). */
    readonly [SNAPSHOT_BRAND]: true;
    /** Process-unique identity of the CORE; shared by withColumns() snapshots (design section 5.8). */
    readonly serial: number;
    /** Debugging aid supplied at freeze. */
    readonly label: string | null;
    /** The data-model major (design section 13.5). */
    readonly formatVersion: 1;
    /** Whether the graph is directed; no tri-state (decision C3). */
    readonly directed: boolean;
    /** n, <= MAX_COUNT (invariant I3). */
    readonly nodeCount: number;
    /** Number of logical edges; every edge column has edgeCount rows (invariant I13). */
    readonly edgeCount: number;
    /** colIdx.length: edgeCount when directed, 2 * edgeCount - selfLoopCount when undirected (invariants I6, I7). */
    readonly arcCount: number;
    /** Logical edges with source === target. */
    readonly selfLoopCount: number;
    /** nodeCount + 1 row offsets (invariant I1). */
    readonly rowPtr: U32;
    /** Target node index of each arc, sorted within each row (invariants I2, I4); length 0 when arcCount === 0. */
    readonly colIdx: U32;
    /**
     * arcCount f32 weights; null when unweighted (every weight is 1). Both arcs of an undirected edge carry the same
     * value (I7).
     */
    readonly weights: F32 | null;
    /**
     * arcCount entries: logical edge of each arc (invariant I5); materialised on first access when
     * flags.arcToEdgeIsIdentity.
     */
    readonly arcToEdge: U32;
    /**
     * edgeCount entries: the arc holding the declared orientation (invariant I5); materialised on first access when
     * identity.
     */
    readonly edgeToArc: U32;
    /** Flags kernels branch on (design section 3.8, invariant I9). */
    readonly flags: SnapshotFlags;
    /** The id map (design section 4, invariant I11). */
    readonly ids: NodeIdMap;
    /** Node attribute table, rowCount === nodeCount (invariant I12). */
    readonly nodes: AttributeTable;
    /** Edge attribute table, rowCount === edgeCount, indexed by logical edge (invariants I12, I13). */
    readonly edges: AttributeTable;
    /** Graph attribute table, rowCount === 1 (design section 5.9). */
    readonly graph: AttributeTable;
    /** Extension tables such as GEXF temporal tables, keyed by name (design section 5.10). */
    readonly extensions: ReadonlyMap<string, AttributeTable>;
    /** Graph-level metadata (design section 5.9). */
    readonly meta: GraphMeta;
    /**
     * The arena holding the core arrays (design section 10.3); null when the arrays were adopted from separate buffers.
     */
    readonly arena: ArenaLayout | null;
    /** rowPtr.length === 0: the core was transferred away (design section 9.4); derived, never a stored bit. */
    readonly detached: boolean;

    // queries (3.9)

    /**
     * The out-arc range of a node: [rowPtr[u], rowPtr[u + 1]]. Allocates a tuple; hot loops read
     * rowPtr directly.
     * @param u - the node index
     * @returns the half-open arc range as [start, end]
     */
    outArcs(u: number): readonly [start: number, end: number];
    /**
     * rowPtr[u + 1] - rowPtr[u]: the out-arc count (a self-loop counted once, design section 3.4).
     * @param u - the node index
     * @returns the out-degree
     */
    outDegreeOf(u: number): number;
    /**
     * Binary search for the first arc u -> v (the lowest logical edge index among parallels).
     * @param u - the source node index
     * @param v - the target node index
     * @returns the arc index, or INVALID_INDEX when absent
     */
    findArc(u: number, v: number): number;
    /**
     * Whether an arc u -> v exists.
     * @param u - the source node index
     * @param v - the target node index
     * @returns findArc(u, v) !== INVALID_INDEX
     */
    hasArc(u: number, v: number): boolean;
    /**
     * The half-open arc range [lo, hi) of every arc u -> v; empty when lo === hi. Allocates a tuple.
     * @param u - the source node index
     * @param v - the target node index
     * @returns the range as [lo, hi]
     */
    arcsBetween(u: number, v: number): readonly [lo: number, hi: number];
    /**
     * Number of parallel arcs u -> v.
     * @param u - the source node index
     * @param v - the target node index
     * @returns hi - lo of arcsBetween(u, v)
     */
    multiplicity(u: number, v: number): number;
    /**
     * The row containing an arc: O(1) after coo(), else a binary search on rowPtr.
     * @param a - the arc index
     * @returns the source node index
     */
    arcSource(a: number): number;
    /**
     * Declared source of a logical edge: arcSource(edgeToArc[e]).
     * @param e - the logical edge index
     * @returns the source node index
     */
    edgeSource(e: number): number;
    /**
     * Declared target of a logical edge: colIdx[edgeToArc[e]].
     * @param e - the logical edge index
     * @returns the target node index
     */
    edgeTarget(e: number): number;
    /**
     * Lookup through the role "id" edge column, backed by a Map built on first call (design section 4.6).
     * @param id - the edge id
     * @returns the logical edge index, or INVALID_INDEX on a miss or when no id column exists
     */
    edgeIndexOf(id: EdgeId): number;

    // views (7.2)

    /**
     * The in-adjacency, rows sorted by source; the forward arrays themselves when undirected. Cached.
     * @returns the reverse view
     */
    reverse(): ReverseView;
    /**
     * Per-arc COO form; src is the only new array. Cached.
     * @returns the COO view
     */
    coo(): CooView;
    /**
     * Every logical edge once in declared orientation. Cached.
     * @returns the edge list view
     */
    edgeList(): EdgeListView;
    /**
     * rowPtr differences materialised; a self-loop counted once. Cached and shared.
     * @returns Uint32Array(n)
     */
    outDegree(): U32;
    /**
     * Arcs targeting each node, via reverse().rowPtr; the same object as outDegree() when undirected. Cached and
     * shared.
     * @returns Uint32Array(n)
     */
    inDegree(): U32;
    /**
     * Graph-theoretic degree (NetworkX convention, design section 3.4): in + out when directed; out plus self-loops
     * when undirected. Cached and shared.
     * @returns Uint32Array(n)
     */
    degree(): U32;
    /**
     * Row sums of weights (a self-loop arc counted once); outDegree widened when unweighted. F64 for CPU precision; may
     * be 0 for a node with out-arcs. Cached and shared.
     * @returns Float64Array(n)
     */
    weightedOutDegree(): F64;
    /**
     * Weight sums over incoming arcs; the same object as weightedOutDegree() when undirected. Cached and shared.
     * @returns Float64Array(n)
     */
    weightedInDegree(): F64;
    /**
     * NetworkX weighted degree: weightedOutDegree + (directed ? weightedInDegree : selfLoopWeight), so sum === 2 *
     * totalWeight() when undirected. Cached and shared.
     * @returns Float64Array(n)
     */
    weightedDegree(): F64;
    /**
     * Sum of weights over each node's self-loop arcs (selfLoopsPerNode widened when unweighted). Cached and shared.
     * @returns Float64Array(n)
     */
    selfLoopWeight(): F64;
    /**
     * Sum of weights over logical edges (each undirected edge once). Cached.
     * @returns the total weight
     */
    totalWeight(): number;
    /**
     * The arcs a with colIdx[a] === row(a), found by binary search per row. Cached and shared.
     * @returns Uint32Array(selfLoopCount)
     */
    selfLoopArcs(): U32;
    /**
     * Self-loop arcs per node. Cached and shared.
     * @returns Uint32Array(n)
     */
    selfLoopsPerNode(): U32;
    /**
     * Point query: self-loop arcs at one node, O(log d).
     * @param u - the node index
     * @returns the count
     */
    selfLoopsAt(u: number): number;
    /**
     * For every arc, the arc storing the opposite orientation of the same edge (a self-loop maps to itself); a lockstep
     * walk (design section 6.4). Undirected only. Cached and shared.
     * @returns Uint32Array(arcCount); E_DIRECTED on a directed snapshot
     */
    mate(): U32;
    /**
     * Nodes ordered by descending out-degree of the forward or reverse adjacency with the cuGraph tier boundaries. Both
     * variants cached.
     * @param options - which adjacency's degree to order by
     * @returns the permutation and its tier offsets
     */
    degreeOrder(options?: DegreeOrderOptions): DegreeOrderView;
    /**
     * Whether the arc set is closed under reversal with equal weights (forward row v equals reverse row v for every v);
     * true without work when undirected. Cached.
     * @returns true when symmetric
     */
    isSymmetric(): boolean;
    /**
     * Compute a set of views eagerly (inside freeze() through FreezeOptions.prepare, or off the critical path).
     * @param views - the views to materialise
     * @returns this snapshot
     */
    prepare(views: readonly ViewName[]): this;
    /** Release every cached view and every cached gpuView copy; they are recomputed on demand. */
    dropCaches(): void;
    /**
     * Which views are resident.
     * @returns the names of the cached views
     */
    cachedViews(): readonly ViewName[];

    // derived graphs (7.3)

    /**
     * Every directed edge becomes undirected; reciprocal pairs collapse to one edge keeping the lower index's row
     * (keep-first). Returns `{ snapshot: this, null maps }` on an undirected snapshot.
     * @param options - reciprocal filtering and the weight reducer
     * @returns the derived graph
     */
    toUndirected(options?: ToUndirectedOptions): DerivedGraph;
    /**
     * Orientation of every edge swapped: the reverse view's arrays become the core. Returns this on an undirected
     * snapshot.
     * @returns the derived graph (same node and edge spaces)
     */
    transpose(): DerivedGraph;
    /**
     * One edge per (u, v) group (parallels are adjacent by invariant I4), survivor = lowest index; flags.multigraph is
     * false afterwards.
     * @param options - reducers and self-loop policy
     * @returns the derived graph
     */
    simplified(options?: SimplifyOptions): DerivedGraph;
    /**
     * Every edge with source !== target.
     * @returns the derived graph
     */
    withoutSelfLoops(): DerivedGraph;
    /**
     * Keep the logical edges whose mask bit is set (a rebuild; iterative algorithms keep their own alive bitmap
     * instead).
     * @param keep - packed bitmap over logical edges; E_MASK_LENGTH when shorter than ceil(edgeCount / 32) words
     * @returns the derived graph
     */
    filterEdges(keep: EdgeMask): DerivedGraph;
    /**
     * The subgraph induced by a node selection: an index list (order = new index order; E_INDEX_RANGE for an
     * out-of-range or repeated index) or a packed mask (ascending order; E_MASK_LENGTH when short). Edges with both
     * endpoints kept.
     * @param selection - the node indices or a mask
     * @returns the derived graph (compact new node space)
     */
    inducedSubgraph(selection: U32 | { readonly mask: NodeMask }): DerivedGraph;
    /**
     * Contract the nodes of each partition block into one node (design section 7.3): labels already forming 0..k-1 are
     * kept as block indices, any other labelling is renumbered in first-seen order; E_PARTITION for a wrong length or
     * an INVALID_INDEX label.
     * @param partition - one label per node
     * @param options - weight reducer, self-loop and parallel policies, column reducers
     * @returns the derived graph with blockSizes and an identity id map
     */
    contract(partition: U32, options?: ContractOptions): DerivedGraph;
    /**
     * Permute the node space: perm[newIndex] = oldIndex; the id map follows; edge order is preserved. Never implicit
     * (decision C18).
     * @param perm - a permutation of 0..n-1; E_INVALID_PERMUTATION otherwise
     * @returns the derived graph
     */
    relabel(perm: U32): DerivedGraph;
    /**
     * A new snapshot object sharing the core, the id map and the serial with a CLONED column set plus the given columns
     * (the only operation that clones the column set).
     * @param nodes - node columns to add, keyed by name
     * @param edges - edge columns to add, keyed by name
     * @returns the new snapshot
     */
    withColumns(
        nodes?: Readonly<Record<string, TypedArrayData | ColumnInput>>,
        edges?: Readonly<Record<string, TypedArrayData | ColumnInput>>,
    ): GraphSnapshot;

    // memory, transfer, checks (9, 11)

    /**
     * Resident bytes of the core, plus the side structures selected by the options.
     * @param options - which side structures to include
     * @returns the byte count
     */
    byteLength(options?: ByteLengthOptions): number;
    /**
     * A 64-bit content hash of the core arrays (two 32-bit FNV-1a lanes) as 16 hex characters, computed lazily and
     * cached (design section 9.3).
     * @returns the hash
     */
    contentHash(): string;
    /**
     * The distinct, exclusively owned backing buffers of the core, id map, typed columns, string stores and extensions:
     * the postMessage transfer list (design section 9.1).
     * @returns the buffers
     */
    transferables(): ArrayBuffer[];
    /**
     * The plain-object wire form: a JSON-serialisable manifest plus ArrayBuffers (design section 9.1). Allocates
     * nothing for typed data except lazily kept representations on first use.
     * @param options - transfer mode, views and columns to include
     * @returns the wire snapshot; E_UNSUPPORTED on a big-endian host
     */
    toWire(options?: ToWireOptions): WireSnapshot;
    /**
     * The GSNP byte container as one contiguous buffer (design section 9.2).
     * @param options - views to include
     * @returns the container bytes; E_UNSUPPORTED on a big-endian host
     */
    toBytes(options?: ToBytesOptions): U8;
    /**
     * The GSNP container as a sequence: header plus manifest, then each 256-padded segment in manifest order, for
     * streaming writers.
     * @param options - views to include
     * @returns the chunks
     */
    toByteChunks(options?: ToBytesOptions): Iterable<U8>;
    /**
     * Check the invariants (design section 11.4); throws E_INVALID_SNAPSHOT with details.invariant and the location on
     * the first violation.
     * @param options - level and checksum comparison
     */
    validate(options?: ValidateOptions): void;
}

// ============================================================ factory inputs (8.1)

/**
 * COO typed arrays with dense indices: the 20 ms path of fromEdgeArrays() (design section 8.1). No
 * id Map is built unless `ids` is given.
 */
export interface EdgeArraysInput {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** Required unless `ids` is given; isolates are preserved. */
    readonly nodeCount?: number | undefined;
    /** Optional external ids in index order (length = nodeCount). */
    readonly ids?: readonly NodeId[] | F64 | undefined;
    /** Source node index of every edge. */
    readonly src: U32;
    /** Target node index of every edge. */
    readonly dst: U32;
    /** Per-edge weights; F64 is downcast to f32 and an f64 shadow column kept only when not f32-exact. */
    readonly weights?: F32 | F64 | undefined;
    /** Node columns, keyed by name. */
    readonly nodeColumns?: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined;
    /** Edge columns, keyed by name. */
    readonly edgeColumns?: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined;
    /** Graph metadata. */
    readonly meta?: GraphMetaPatch | undefined;
}

/**
 * Prebuilt CSR arrays adopted by fromCsr() without copying by default (design section 8.1): the
 * caller transfers ownership and must not mutate them afterwards.
 */
export interface CsrInput {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** Number of nodes. */
    readonly nodeCount: number;
    /** nodeCount + 1 row offsets. */
    readonly rowPtr: U32;
    /** Target node index of every arc. */
    readonly colIdx: U32;
    /** Per-arc weights, or null / absent when unweighted. */
    readonly weights?: F32 | null | undefined;
    /** Absent => identity (directed only; undirected input must supply it). */
    readonly arcToEdge?: U32 | undefined;
    /** Absent => derived in one O(m) pass (or identity). */
    readonly edgeToArc?: U32 | undefined;
    /** Defaults to arcCount (directed) or is derived from arcToEdge. */
    readonly edgeCount?: number | undefined;
    /** Optional external ids in index order. */
    readonly ids?: readonly NodeId[] | F64 | undefined;
    /** Node columns, keyed by name. */
    readonly nodeColumns?: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined;
    /** Edge columns, keyed by name. */
    readonly edgeColumns?: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined;
    /** Graph metadata. */
    readonly meta?: GraphMetaPatch | undefined;
    /** Flag claims; verified unless validate === "none". */
    readonly flags?: FlagClaims | undefined;
}

/** Options of fromCsr() (design section 8.1). */
export interface FromCsrOptions {
    /** Validation level; default "full" because adopted arrays typically come from a file or the network. */
    readonly validate?: ValidationLevel | undefined;
    /** Copy the arrays instead of adopting them; default false. */
    readonly copy?: boolean | undefined;
    /**
     * Default true: check invariant I4 and rebuild through the freeze pipeline when rows are unsorted; false asserts
     * sorted rows.
     */
    readonly sortRows?: boolean | undefined;
}

/**
 * Plain records, the node-link shape every JSON dialect parses into and what graphty-element's data
 * sources emit (design section 8.1).
 */
export interface RecordsInput {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** Node records; optional when every node appears as an edge endpoint. */
    readonly nodes?: Iterable<Readonly<Record<string, unknown>>> | undefined;
    /** Edge records. */
    readonly edges: Iterable<Readonly<Record<string, unknown>>>;
    /** Node id key; default "id"; null = node index is array position and endpoints are indices (d3 v3). */
    readonly nodeId?: string | null | undefined;
    /** Edge source key; default "source", falling back to "src" / "from". */
    readonly edgeSource?: string | undefined;
    /** Edge target key; default "target", falling back to "dst" / "to". */
    readonly edgeTarget?: string | undefined;
    /** Edge weight key; default "weight"; null = unweighted. */
    readonly edgeWeight?: string | null | undefined;
    /** Column handling; default "infer" (design section 5.1 widening). */
    readonly columns?: "infer" | "json" | "none" | readonly ColumnDecl[] | undefined;
    /** Id coercion; default "keep" (values are already typed). */
    readonly ids?: IdCoercion | undefined;
}
