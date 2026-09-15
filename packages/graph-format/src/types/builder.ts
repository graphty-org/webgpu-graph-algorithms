/**
 * The builder surface (design sections 6, 8.3 and 12.2): options, freeze options and report, the
 * branded handles, the GraphSink contract importers program against, and the GraphBuilder contract.
 *
 * GraphBuilderContract is the instance contract of the GraphBuilder class implemented in
 * src/builder/graph-builder.ts, which adds `constructor(options)` and `static from(snapshot,
 * options?)` (design section 12.2). The public name GraphBuilder is the class itself, exported by
 * the barrel; nothing on this surface needs to name it (the factories return GraphSnapshot).
 */

import {
    type ColumnDecl,
    type ColumnDeclPatch,
    type ColumnDomain,
    type Dtype,
    type DuplicatePolicy,
    type F32,
    type F64,
    type GraphMetaPatch,
    type Loose,
    type NodeId,
    type TypedArrayData,
    type U32,
} from "./columns.js";
import { type GraphSnapshot, type ViewName } from "./snapshot.js";

/** Constructor options of GraphBuilder (design sections 3.6, 3.7, 6.5 and 12.2). */
export interface GraphBuilderOptions {
    /** REQUIRED; no default; changeable later via setDirected() (design section 6.6). */
    directed: boolean;
    /**
     * Default "auto": weighted if any addEdge supplied a weight; true allocates the array even when every value is 1.
     */
    weighted?: boolean | "auto" | undefined;
    /** Staging precision; default "f32" (design section 3.7); importers pass "f64". */
    weightDtype?: "f32" | "f64" | undefined;
    /** Default "keep". */
    duplicateEdges?: DuplicatePolicy | undefined;
    /** Default "keep". */
    selfLoops?: "keep" | "drop" | "error" | undefined;
    /** Default true: addEdge creates unknown endpoints; false throws E_UNKNOWN_NODE. */
    addMissingNodes?: boolean | undefined;
    /** Staging capacity hint. */
    expectedNodes?: number | undefined;
    /** Staging capacity hint. */
    expectedEdges?: number | undefined;
}

/** Patch form of GraphBuilderOptions, accepted by the factories and by GraphBuilder.from(). */
export type BuilderOptionsPatch = Loose<GraphBuilderOptions>;

/** The builder's options after defaults were applied; the same shape under every compiler flag. */
export interface ResolvedBuilderOptions {
    /** The CURRENT value (setDirected() updates it). */
    readonly directed: boolean;
    /** The weighted policy. */
    readonly weighted: boolean | "auto";
    /** The staging weight precision. */
    readonly weightDtype: "f32" | "f64";
    /** The default duplicate policy of freeze(). */
    readonly duplicateEdges: DuplicatePolicy;
    /** The self-loop policy. */
    readonly selfLoops: "keep" | "drop" | "error";
    /** Whether addEdge creates unknown endpoints. */
    readonly addMissingNodes: boolean;
    /** Capacity hint, or null. */
    readonly expectedNodes: number | null;
    /** Capacity hint, or null. */
    readonly expectedEdges: number | null;
}

/** Options of freeze() and freezeWithReport() (design section 6.7). */
export interface FreezeOptions {
    /** Debugging label carried by the snapshot. */
    label?: string | undefined;
    /** Views computed eagerly inside freeze. */
    prepare?: readonly ViewName[] | undefined;
    /** Default true: core arrays share one 256-aligned arena (design section 10.3). */
    arena?: boolean | undefined;
    /** Default false; true empties staging after freezing (Arrow flush semantics). */
    release?: boolean | undefined;
    /** Overrides the builder default for this freeze; a merge policy REWRITES THE BUILDER (design section 6.5). */
    duplicateEdges?: DuplicatePolicy | undefined;
    /** Fills FreezeReport.timings. */
    profile?: boolean | undefined;
    /** Records FNV-1a checksums for validate({ checksum: true }) (design section 5.8). */
    checksum?: boolean | undefined;
}

/**
 * What freezeWithReport() returns next to the snapshot (design sections 4.4 and 6.6). The remaps
 * are relative to the previous freeze of the same builder, or to the builder's own index space on
 * the first freeze (invariant I16); each is null exactly when nothing was renumbered, and the two
 * are independent.
 */
export interface FreezeReport {
    /** Previous freeze's node index space -> new index or INVALID_INDEX; null when nodes were not renumbered (I16). */
    readonly nodeRemap: U32 | null;
    /** Same for edges (a merged edge: its survivor); null when edges were not renumbered; independent of nodeRemap. */
    readonly edgeRemap: U32 | null;
    /** Whether step 1 of the freeze pipeline compacted tombstones. */
    readonly compacted: boolean;
    /** Self-loops removed under selfLoops "drop". */
    readonly droppedSelfLoops: number;
    /** Parallel edges collapsed by duplicateEdges. */
    readonly mergedEdges: number;
    /** Tombstoned + dropped loops + merged. */
    readonly droppedEdges: number;
    /** Columns whose inferred dtype widened during staging (design section 5.1), so importers can fix headers. */
    readonly widened: readonly {
        readonly column: string;
        readonly domain: ColumnDomain;
        readonly from: Dtype;
        readonly to: Dtype;
    }[];
    /** ms per phase; empty unless profile. */
    readonly timings: Readonly<Record<string, number>>;
}

/**
 * Branded index into the builder's column list for its domain; INVALID_INDEX when absent (design
 * section 12.1). Handles never come out of a typed array, and the brand stops
 * setNodeValue(index, handle, v) with the arguments swapped from type-checking.
 */
export type ColumnHandle = number & { readonly __brand: "ColumnHandle" };

/** Branded index of an extension table in the builder (design section 5.10). */
export type ExtensionHandle = number & { readonly __brand: "ExtensionHandle" };

/** Options of setDirected() (design section 6.6). */
export interface SetDirectedOptions {
    /**
     * Convert a non-empty undirected builder to directed by appending a mirror edge per existing edge and writing the
     * graphty.directed / graphty.pair columns (design section 3.6).
     */
    readonly expand?: boolean | undefined;
}

/**
 * The subset of GraphBuilder that importers may call (design section 8.3). Importers push scalars,
 * never objects; the record methods exist for the transitional element path and JSON. GraphBuilder
 * implements it, and tests can pass a recording sink.
 */
export interface GraphSink {
    /** The resolved options; `directed` is the current value. */
    readonly options: ResolvedBuilderOptions;
    /** The current direction. */
    readonly directed: boolean;
    /** Whether lockDirected() was called. */
    readonly directedLocked: boolean;
    /** Live edge count. */
    readonly edgeCount: number;
    /**
     * Change the direction (design section 6.6): free while the builder holds no edges; with edges,
     * only undirected -> directed with expand. E_DIRECTED when refused or locked.
     * @param directed - the new value
     * @param options - expansion of an undirected builder
     */
    setDirected(directed: boolean, options?: SetDirectedOptions): void;
    /**
     * Grow staging capacity ahead of a bulk push.
     * @param nodes - expected node count
     * @param edges - expected edge count
     */
    reserve(nodes?: number, edges?: number): void;
    /**
     * Add a node or return the index of an existing live one (idempotent); revives a tombstoned id before the next
     * freeze.
     * @param id - the node id; E_INVALID_ID when not a legal id
     * @returns the node index
     */
    addNode(id: NodeId): number;
    /**
     * Add many nodes.
     * @param ids - the node ids
     * @param out - receives the index of every id (allocated when omitted)
     * @returns `out`
     */
    addNodes(ids: Iterable<NodeId>, out?: U32): U32;
    /**
     * Add a logical edge; unknown endpoints are created when addMissingNodes is true, else E_UNKNOWN_NODE. NaN weight
     * is E_INVALID_WEIGHT.
     * @param source - source node id
     * @param target - target node id
     * @param weight - the weight; 1 when omitted
     * @returns the logical edge index
     */
    addEdge(source: NodeId, target: NodeId, weight?: number): number;
    /**
     * Bulk add edges by node index (the index-space path used with addAnonymousNodes).
     * @param src - source node indices
     * @param dst - target node indices
     * @param weights - per-edge weights; 1 when omitted
     * @returns the first new logical edge index
     */
    addEdges(src: U32, dst: U32, weights?: F32 | F64): number;
    /**
     * Set the weight of a live edge (allocates the weight array on first use).
     * @param edge - the logical edge index
     * @param weight - the weight; E_INVALID_WEIGHT for NaN
     */
    setEdgeWeight(edge: number, weight: number): void;
    /**
     * Read a live edge's endpoints back.
     * @param edge - the logical edge index
     * @returns [source index, target index] in declared orientation
     */
    edgeEndpoints(edge: number): readonly [source: number, target: number];
    /**
     * Read a live edge's weight back.
     * @param edge - the logical edge index
     * @returns the weight (1 when unweighted)
     */
    edgeWeight(edge: number): number;
    /**
     * Total lookup of a live node by id.
     * @param id - the node id
     * @returns the node index, or INVALID_INDEX
     */
    indexOf(id: NodeId): number;
    /**
     * Declare a node column; the same declaration again returns the existing handle, a different one is
     * E_COLUMN_EXISTS.
     * @param decl - the declaration
     * @returns the column handle
     */
    declareNodeColumn(decl: ColumnDecl): ColumnHandle;
    /**
     * Declare an edge column; the same declaration again returns the existing handle, a different one is
     * E_COLUMN_EXISTS.
     * @param decl - the declaration
     * @returns the column handle
     */
    declareEdgeColumn(decl: ColumnDecl): ColumnHandle;
    /**
     * Look up a node column handle.
     * @param name - the column name
     * @returns the handle, or INVALID_INDEX when absent
     */
    nodeColumn(name: string): ColumnHandle;
    /**
     * Look up an edge column handle.
     * @param name - the column name
     * @returns the handle, or INVALID_INDEX when absent
     */
    edgeColumn(name: string): ColumnHandle;
    /**
     * Widen an inferred node column to a wider dtype of the design section 5.1 order without
     * changing any value (a text importer that knows from the lexical grammar that `2.0` cells
     * are f64 although every value so far was integral). Optional: a sink without it makes such an
     * importer report the dtype it could not widen.
     * @param column - the handle or name
     * @param dtype - the dtype to widen to
     */
    widenNodeColumn?(column: ColumnHandle | string, dtype: Dtype): void;
    /**
     * Widen an inferred edge column; see widenNodeColumn().
     * @param column - the handle or name
     * @param dtype - the dtype to widen to
     */
    widenEdgeColumn?(column: ColumnHandle | string, dtype: Dtype): void;
    /**
     * Set one node cell; a string column name auto-declares with inference (design section 5.1).
     * @param column - the handle or name
     * @param index - the node index
     * @param value - the value
     */
    setNodeValue(column: ColumnHandle | string, index: number, value: unknown): void;
    /**
     * Set one edge cell; a string column name auto-declares with inference.
     * @param column - the handle or name
     * @param edge - the logical edge index
     * @param value - the value
     */
    setEdgeValue(column: ColumnHandle | string, edge: number, value: unknown): void;
    /**
     * Set a graph-level attribute (the graph table's single row).
     * @param name - the column name
     * @param value - the value
     * @param decl - declaration fields for a new column
     */
    setGraphValue(name: string, value: unknown, decl?: ColumnDeclPatch): void;
    /**
     * Merge fields into the graph metadata; non-JSON extra values are E_COLUMN_TYPE.
     * @param meta - the fields to set
     */
    setMeta(meta: GraphMetaPatch): void;
    /**
     * Create an extension table (design section 5.10).
     * @param name - the table name, e.g. "temporal:node:price"
     * @param decls - its columns
     * @returns the table handle
     */
    addExtensionTable(name: string, decls: readonly ColumnDecl[]): ExtensionHandle;
    /**
     * Append a row to an extension table.
     * @param table - the table handle
     * @param values - one value per declared column
     * @returns the new row index
     */
    addExtensionRow(table: ExtensionHandle, values: readonly unknown[]): number;
    /**
     * Add a node from a record; on an existing live id every key present overwrites that row (last-write-wins per
     * attribute).
     * @param id - the node id
     * @param attrs - attribute values keyed by column name
     * @returns the node index
     */
    addNodeRecord(id: NodeId, attrs: Readonly<Record<string, unknown>>): number;
    /**
     * Add an edge from a record; always creates a new edge (parallels are kept).
     * @param source - source node id
     * @param target - target node id
     * @param attrs - attribute values keyed by column name
     * @param weightKey - the key holding the weight; default "weight"; null = no weight
     * @returns the logical edge index
     */
    addEdgeRecord(
        source: NodeId,
        target: NodeId,
        attrs: Readonly<Record<string, unknown>>,
        weightKey?: string | null,
    ): number;
}

/**
 * The only mutable object in the package and the producer of snapshots (design section 6): a
 * long-lived, structure-of-arrays accumulator that can be frozen repeatedly. freeze() never shares
 * core arrays or columns with the snapshot it returns (invariant I18; the id Map and ids array are
 * shared by design), is deterministic (I15), and preserves index prefixes between freezes with only
 * appends in between (I16). Indices are never reused while the builder lives except through a
 * compacting freeze, after which builder indices equal the new snapshot's (decision C7).
 *
 * Instance contract of the GraphBuilder class in src/builder/graph-builder.ts, which implements it
 * and adds `constructor(options: GraphBuilderOptions)` and `static from(snapshot, options?)`; the
 * public barrel exports the class and every public type names the class.
 */
export interface GraphBuilderContract extends GraphSink {
    /** Live node count. */
    readonly nodeCount: number;
    /** Next node index to be assigned. */
    readonly nodeBound: number;
    /** Next logical edge index to be assigned. */
    readonly edgeBound: number;
    /** Increments on every topology or weight mutation; column writes and freeze() do not count. */
    readonly mutationCount: number;
    /** Mutated since the last freeze(). */
    readonly dirty: boolean;
    /** Fix the direction so importers cannot change it; every later changing setDirected() is E_DIRECTED. */
    lockDirected(): void;
    // nodes
    /**
     * Append nodes whose ids are their own indices, never touching the id Map while every node is anonymous.
     * @param count - how many
     * @returns the first new index
     */
    addAnonymousNodes(count: number): number;
    /**
     * Whether a live node has this id.
     * @param id - the node id
     * @returns true when present and alive
     */
    hasNode(id: NodeId): boolean;
    /**
     * The id of a live node index.
     * @param index - the node index
     * @returns the id; E_INDEX_RANGE when out of range or dead
     */
    idOf(index: number): NodeId;
    /**
     * Tombstone a node and every live incident edge (both directions); O(degree); bumps mutationCount.
     * @param id - the node id
     * @returns the removed live incident edge indices
     */
    removeNode(id: NodeId): U32;
    /**
     * Tombstone a node by index and every live incident edge.
     * @param index - the node index
     * @returns the removed live incident edge indices
     */
    removeNodeByIndex(index: number): U32;
    // edges
    /**
     * Add a logical edge by node index; E_UNKNOWN_NODE (details.index) for a dead or out-of-range index.
     * @param u - source node index
     * @param v - target node index
     * @param weight - the weight; 1 when omitted
     * @returns the logical edge index
     */
    addEdgeByIndex(u: number, v: number, weight?: number): number;
    /**
     * Bulk add edges by id.
     * @param src - source node ids
     * @param dst - target node ids
     * @param weights - per-edge weights; 1 when omitted
     * @returns the first new logical edge index
     */
    addEdgesByIds(src: ArrayLike<NodeId>, dst: ArrayLike<NodeId>, weights?: ArrayLike<number>): number;
    /**
     * Tombstone an edge; O(1); bumps mutationCount.
     * @param edge - the logical edge index
     * @returns true when a live edge was removed
     */
    removeEdge(edge: number): boolean;
    /**
     * Whether an edge index is live.
     * @param edge - the logical edge index
     * @returns true when live
     */
    hasEdge(edge: number): boolean;
    /**
     * Live edges leaving a node, read from the incidence lists (O(degree)). On an undirected
     * builder every incident edge leaves the node (invariant I7: the snapshot's row holds both
     * orientations), so the declared orientation is ignored and a self-loop is listed once.
     * @param index - the node index
     * @returns a fresh ascending array of live edge indices
     */
    outEdgesOf(index: number): U32;
    /**
     * Live edges entering a node (O(degree)); on an undirected builder the same set as outEdgesOf
     * (the alias rule of inDegree === outDegree).
     * @param index - the node index
     * @returns a fresh ascending array of live edge indices
     */
    inEdgesOf(index: number): U32;
    /**
     * Live edges u -> v (undirected: either orientation), for answering "is there an edge" before a freeze.
     * @param u - source node index
     * @param v - target node index
     * @returns a fresh array of live edge indices
     */
    findEdges(u: number, v: number): U32;
    // attributes
    /**
     * Bulk-set a node column; length must equal nodeBound.
     * @param name - the column name
     * @param data - the values
     * @param decl - declaration fields
     */
    setNodeColumn(name: string, data: TypedArrayData, decl?: ColumnDeclPatch): void;
    /**
     * Bulk-set an edge column; length must equal edgeBound.
     * @param name - the column name
     * @param data - the values
     * @param decl - declaration fields
     */
    setEdgeColumn(name: string, data: TypedArrayData, decl?: ColumnDeclPatch): void;
    // composition
    /**
     * Append another snapshot (disjoint union, or merge by id), re-interning dictionaries (design section 6.6).
     * @param snapshot - the snapshot to append
     * @param options - duplicate-id handling
     * @param options.onDuplicateNode - "merge" (default) overwrites set rows of existing ids; "error" throws
     *   E_DUPLICATE_ID
     */
    addGraph(snapshot: GraphSnapshot, options?: { readonly onDuplicateNode?: "merge" | "error" | undefined }): void;
    // output and lifecycle
    /**
     * Run the freeze pipeline of design section 6.3 and return a snapshot; the builder keeps its staging unless release
     * is set.
     * @param options - freeze options
     * @returns the frozen snapshot
     */
    freeze(options?: FreezeOptions): GraphSnapshot;
    /**
     * freeze() plus the report of what was renumbered, merged, dropped and widened.
     * @param options - freeze options
     * @returns the snapshot and its report
     */
    freezeWithReport(options?: FreezeOptions): { snapshot: GraphSnapshot; report: FreezeReport };
    /** Empty the builder (nodes, edges, columns, meta) while keeping its options. */
    clear(): void;
    /** Release everything; every further call throws E_BUILDER_DISPOSED. */
    dispose(): void;
    /**
     * Bytes of staging currently held.
     * @returns the byte count
     */
    byteLength(): number;
}
