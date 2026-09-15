/**
 * Construction contracts shared by the builder, columns, snapshot, populate and wire modules
 * (derived from design sections 3, 5, 6.3 and 7). NOT part of the public surface: src/types/index.ts
 * does not re-export this file, and nothing here is named in design section 12.2.
 *
 * The file exists so that the modules can be written in sequence without renegotiating names:
 * freeze() (design section 6.3 step 12), fromCsr() and fromWire() / fromBytes() build a
 * SnapshotParts and hand it to the snapshot constructor; freeze() step 10, AttributeTable.set() and
 * the wire reader build one MutableColumnParts per column and hand it to the column factory of
 * src/columns/column.ts; the snapshot keeps one ViewCache per instance.
 */

import {
    type AttributeTable,
    type Column,
    type ColumnDomain,
    type ColumnMeta,
    type F32,
    type F64,
    type GraphMeta,
    type TypedArrayData,
    type U8,
    type U32,
} from "./columns.js";
import {
    type ArenaLayout,
    type CooView,
    type DegreeOrderView,
    type EdgeListView,
    type NodeIdMap,
    type ReverseView,
    type SnapshotFlags,
    type ViewName,
} from "./snapshot.js";

/**
 * Everything the snapshot constructor needs (design sections 3.1 and 6.3 step 12). The producer has
 * already established invariants I1-I13 (the builder by construction, fromCsr / fromWire by
 * validation); the constructor freezes the object, records checksums when asked and allocates the
 * per-instance view cache. No array here may alias memory a builder can still write (I18).
 */
export interface SnapshotParts {
    /** Debugging label supplied at freeze, or null. */
    readonly label: string | null;
    /**
     * null: allocate a fresh process-unique serial; a number: share the core identity (withColumns(), design section
     * 5.8).
     */
    readonly serial: number | null;
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** n (invariant I3). */
    readonly nodeCount: number;
    /** Logical edge count. */
    readonly edgeCount: number;
    /** colIdx.length (invariants I1, I6, I7). */
    readonly arcCount: number;
    /** Logical edges with source === target. */
    readonly selfLoopCount: number;
    /** nodeCount + 1 row offsets. */
    readonly rowPtr: U32;
    /** arcCount targets, sorted within rows (I4); length 0 when arcCount === 0. */
    readonly colIdx: U32;
    /** arcCount f32 weights, or null when unweighted. */
    readonly weights: F32 | null;
    /**
     * arcCount entries, or null exactly when flags.arcToEdgeIsIdentity (the snapshot materialises the identity on first
     * access, outside the arena).
     */
    readonly arcToEdge: U32 | null;
    /** edgeCount entries, or null exactly when flags.arcToEdgeIsIdentity. */
    readonly edgeToArc: U32 | null;
    /** Truthful flags computed by the producer (invariant I9). */
    readonly flags: SnapshotFlags;
    /** The id map, size === nodeCount (invariant I11). */
    readonly ids: NodeIdMap;
    /** Node table, rowCount === nodeCount (invariant I12). */
    readonly nodes: AttributeTable;
    /** Edge table, rowCount === edgeCount. */
    readonly edges: AttributeTable;
    /** Graph table, rowCount === 1. */
    readonly graph: AttributeTable;
    /** Extension tables keyed by name; may be empty. */
    readonly extensions: ReadonlyMap<string, AttributeTable>;
    /** Graph metadata, every field present. */
    readonly meta: GraphMeta;
    /** The arena the core arrays are views into, or null when they are separate buffers (design section 10.3). */
    readonly arena: ArenaLayout | null;
    /**
     * Record FNV-1a checksums of the core arrays, immutable columns and views for validate({ checksum: true }) (design
     * section 5.8).
     */
    readonly checksum: boolean;
}

/**
 * The raw storage of one column before it is wrapped in a Column object (design sections 5.1, 5.3
 * and 5.7). Every slot a dtype does not use is null. The column factory switches on meta.dtype and
 * checks the length rules of design section 5.7; the producer (freeze step 10, AttributeTable.set,
 * the wire reader) fills the slots and may keep mutating them until the wrap.
 *
 * Slot use per dtype: f32 / f64 / i32 / u32 / u8 -> data (rows * components values); bool -> data
 * (ceil(rows / 32) packed U32 words); dict -> data (U32 codes) + dictionary; string -> offsets
 * (rows + 1) + utf8, optionally strings (decoded cache, sparse); list -> offsets (rows + 1) + child;
 * json -> values (rows entries).
 */
export interface MutableColumnParts {
    /** Resolved metadata; every field present. */
    meta: ColumnMeta;
    /** Number of rows. */
    length: number;
    /** Numeric values, packed bool words or dict codes; null for string / list / json. */
    data: TypedArrayData | null;
    /** Validity words, ceil(length / 32); null when every row is set. */
    validity: U32 | null;
    /** Number of unset rows; must agree with validity. */
    nullCount: number;
    /** dict only: the dictionary in code order. */
    dictionary: string[] | null;
    /** string / list only: rows + 1 non-decreasing offsets. */
    offsets: U32 | null;
    /** string only: the concatenated UTF-8 bytes. */
    utf8: U8 | null;
    /** string only: decoded rows, sparse (undefined = not yet decoded); null when none decoded. */
    strings: (string | undefined)[] | null;
    /** list only: the already wrapped, non-nullable, non-list child column. */
    child: Column | null;
    /** json only: one value per row, undefined in unset rows. */
    values: unknown[] | null;
}

/**
 * What the AttributeTable constructor takes: the domain, the fixed row count and the wrapped columns in declaration
 * order.
 */
export interface TableParts {
    /** The table's domain. */
    readonly domain: ColumnDomain;
    /** Row count of every column; fixed for the life of the table. */
    readonly rowCount: number;
    /** The columns in declaration order; names unique, at most one column per role. */
    readonly columns: readonly Column[];
}

/**
 * The value each view name resolves to (design section 7.2). Keyed exactly by ViewName so that
 * ViewCache, prepare(), cachedViews() and the wire includeViews option agree on the vocabulary
 * (test/types/internal.test-d.ts asserts the key set).
 */
export interface ViewValues {
    /** In-adjacency (the forward arrays themselves when undirected). */
    readonly reverse: ReverseView;
    /** Per-arc COO. */
    readonly coo: CooView;
    /** Every logical edge once. */
    readonly edgeList: EdgeListView;
    /** Out-arc counts. */
    readonly outDegree: U32;
    /** In-arc counts (the outDegree object when undirected). */
    readonly inDegree: U32;
    /** Graph-theoretic degree. */
    readonly degree: U32;
    /** Row weight sums. */
    readonly weightedOutDegree: F64;
    /** Incoming weight sums. */
    readonly weightedInDegree: F64;
    /** NetworkX weighted degree. */
    readonly weightedDegree: F64;
    /** Self-loop weight per node. */
    readonly selfLoopWeight: F64;
    /** Sum of weights over logical edges. */
    readonly totalWeight: number;
    /** Arcs a with colIdx[a] === row(a). */
    readonly selfLoopArcs: U32;
    /** Self-loop arcs per node. */
    readonly selfLoopsPerNode: U32;
    /** Opposite-orientation arc per arc (undirected only). */
    readonly mate: U32;
    /** degreeOrder({ of: "forward" }). */
    readonly degreeOrder: DegreeOrderView;
    /** degreeOrder({ of: "reverse" }). */
    readonly reverseDegreeOrder: DegreeOrderView;
    /** isSymmetric(). */
    readonly symmetric: boolean;
}

/**
 * The per-instance, per-realm cache of lazily computed views (design section 7.2, invariant I17):
 * one writable slot per ViewName, null until first materialisation. Every slot holds the object the
 * public method returns (SHARED, never copied); dropCaches() nulls every slot; cachedViews() lists
 * the non-null ones. Views are pure functions of the core, so a slot is never invalidated, only
 * dropped. The lazily materialised identity permutations of design section 3.1 are not views and
 * are kept by the snapshot separately.
 */
export type ViewCache = { -readonly [K in ViewName]: ViewValues[K] | null };
