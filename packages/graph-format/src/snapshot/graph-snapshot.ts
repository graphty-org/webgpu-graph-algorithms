/**
 * The concrete GraphSnapshot class (design sections 3, 5.8, 7, 9.3 and 11.4) and its factory
 * `createSnapshot(parts)`, plus the boundary helpers `isGraphSnapshot` / `equalsTopology` (design
 * section 7.5) and the arena helpers a GPU consumer binds segments with (section 10.3).
 *
 * The object is frozen with `Object.freeze` so `snapshot.rowPtr = x` throws in strict mode; every
 * lazily populated member (the identity permutations of design section 3.1, the view cache of
 * section 7.2, the checksum records of section 5.8, the content hash, the edge id index) lives in
 * one mutable state record reachable only through a private field, which `Object.freeze` does not
 * reach. `arcToEdge` / `edgeToArc` are prototype getters that materialise an identity permutation on
 * first access outside the arena (never on the wire, never in `byteLength()`); `rowPtr`, `colIdx`
 * and `weights` are plain data properties so hot loops pay nothing. Every method that reads the core
 * throws `E_DETACHED` once the core buffer was transferred away (`detached` is derived:
 * `rowPtr.length === 0`).
 *
 * Views are computed by views.ts, queries by queries.ts, derived graphs by derived.ts (which returns
 * SnapshotParts that this module wraps), invariant checks by validate.ts and hashes by hash.ts. The
 * wire methods (`toWire`, `toBytes`, `toByteChunks`, `transferables`) call the wire module directly;
 * the wire module imports this one too, and the cycle is safe because each side reaches the other
 * only through hoisted function declarations invoked at call time, never at module load.
 */

import { dropGpuViewCache, isColumnDetached } from "../columns/column.js";
import { columnSetVersion } from "../columns/table.js";
import { FORMAT_VERSION, INVALID_INDEX, SNAPSHOT_BRAND } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { EdgeIdIndex } from "../ids/edge-id-index.js";
import { idMapBuffers, idMapDetached } from "../ids/node-id-map.js";
import {
    type ArenaLayout,
    type AttributeTable,
    type ByteLengthOptions,
    type Column,
    type ColumnInput,
    type ContractOptions,
    type CooView,
    type CoreArrayName,
    type DegreeOrderOptions,
    type DegreeOrderView,
    type DerivedGraph,
    type EdgeId,
    type EdgeListView,
    type EdgeMask,
    type F32,
    type F64,
    type GraphMeta,
    type GraphSnapshotContract,
    type NodeIdMap,
    type NodeMask,
    type ReverseView,
    type SimplifyOptions,
    type SnapshotFlags,
    type ToBytesOptions,
    type ToUndirectedOptions,
    type ToWireOptions,
    type TypedArrayData,
    type U8,
    type U32,
    type ValidateOptions,
    type ViewName,
    type WireSnapshot,
} from "../types/index.js";
import { type SnapshotParts, type ViewCache } from "../types/internal.js";
import { assertOneOf } from "../util/options.js";
import { claimHolder } from "../util/shared-buffers.js";
import { toByteChunks, toBytes } from "../wire/bytes.js";
import { toWire, transferables } from "../wire/to-wire.js";
import {
    deriveContract,
    type DerivedParts,
    deriveFilterEdges,
    deriveInducedSubgraph,
    deriveRelabel,
    deriveSimplified,
    deriveToUndirected,
    deriveTranspose,
    deriveWithoutSelfLoops,
    identityDerived,
    withColumnsParts,
} from "./derived.js";
import { contentHashOf, hashColumn, hashTypedArray } from "./hash.js";
import { arcRangeIn, arcSourceIn, findArcIn, multiplicityIn, selfLoopsAtIn } from "./queries.js";
import { validateFull, validateStructure } from "./validate.js";
import {
    computeCoo,
    computeDegreeOrder,
    computeEdgeList,
    computeMate,
    computeReverse,
    computeSelfLoops,
    computeSelfLoopWeight,
    computeSymmetric,
    computeTotalWeight,
    identityPermutation,
    rowLengths,
    rowWeightSums,
    sumDegrees,
    sumF64,
    viewArrays,
    viewByteLength,
    widenToF64,
} from "./views.js";

// ============================================================ module state

/** The next process-unique serial (design section 5.8). */
let nextSerial = 1;

/** Every view name in the order `cachedViews()` reports them. */
const VIEW_NAMES: readonly ViewName[] = [
    "reverse",
    "coo",
    "edgeList",
    "outDegree",
    "inDegree",
    "degree",
    "weightedOutDegree",
    "weightedInDegree",
    "weightedDegree",
    "selfLoopWeight",
    "totalWeight",
    "selfLoopArcs",
    "selfLoopsPerNode",
    "mate",
    "degreeOrder",
    "reverseDegreeOrder",
    "symmetric",
];

/**
 * The own enumerable property every snapshot carries whose value is a function, so that
 * `structuredClone(snapshot)` and `postMessage(snapshot)` throw `DataCloneError` immediately (design
 * section 7.5; the structured clone algorithm refuses functions and visits enumerable own
 * properties only).
 */
export const CLONE_GUARD_KEY = "__graphtyNoStructuredClone";

export { EMPTY_GRAPH_META } from "./graph-meta.js";

/**
 * The value of the clone guard property.
 * @returns nothing
 */
function cloneGuard(): void {
    return undefined;
}

/** The FNV-1a records of design section 5.8, kept when the snapshot was built with `checksum: true`. */
interface ChecksumRecords {
    /** Core array name -> digest (identity permutations recorded when materialised). */
    readonly core: Map<string, string>;
    /** Immutable column -> digest, at construction. */
    readonly columns: WeakMap<Column, string>;
    /** "<view>.<member>" -> digest, at first materialisation. */
    readonly views: Map<string, string>;
}

/** The mutable state behind a frozen snapshot object. */
interface SnapshotState {
    /** The arc -> edge permutation, or null until an identity permutation is materialised. */
    arcToEdge: U32 | null;
    /** The edge -> arc permutation, or null until an identity permutation is materialised. */
    edgeToArc: U32 | null;
    /** The view cache (one slot per ViewName). */
    readonly views: ViewCache;
    /** Checksum records, or null when the snapshot was built without `checksum: true`. */
    readonly checksums: ChecksumRecords | null;
    /** The cached content hash. */
    contentHash: string | null;
    /** The lazily built edge id index and the column it was built over. */
    edgeIds: { readonly version: number; readonly column: Column | null; readonly index: EdgeIdIndex | null } | null;
}

/** Module-private access to the state of a snapshot for the helpers below. */
const STATES = new WeakMap<GraphSnapshot, SnapshotState>();

/**
 * A fresh view cache with every slot empty.
 * @returns the cache
 */
function emptyViewCache(): ViewCache {
    return {
        reverse: null,
        coo: null,
        edgeList: null,
        outDegree: null,
        inDegree: null,
        degree: null,
        weightedOutDegree: null,
        weightedInDegree: null,
        weightedDegree: null,
        selfLoopWeight: null,
        totalWeight: null,
        selfLoopArcs: null,
        selfLoopsPerNode: null,
        mate: null,
        degreeOrder: null,
        reverseDegreeOrder: null,
        symmetric: null,
    };
}

/**
 * The E_INVALID_SNAPSHOT error of a SnapshotParts inconsistency caught at construction.
 * @param invariant - the invariant number
 * @param message - the description
 * @param details - the location
 * @returns the error
 */
function partsError(invariant: string, message: string, details: Readonly<Record<string, unknown>>): GraphFormatError {
    return new GraphFormatError("E_INVALID_SNAPSHOT", `invariant ${invariant} violated: ${message}`, {
        invariant,
        ...details,
    });
}

/**
 * The O(1) consistency checks the constructor runs on the parts a producer hands it (lengths and
 * the identity-flag rule); everything deeper is `validate()`.
 * @param parts - the parts
 */
function checkParts(parts: SnapshotParts): void {
    const { nodeCount, edgeCount, arcCount, flags } = parts;
    if (parts.rowPtr.length !== nodeCount + 1) {
        throw partsError("I1", `rowPtr has ${parts.rowPtr.length} entries, expected ${nodeCount + 1}`, {
            expected: nodeCount + 1,
            found: parts.rowPtr.length,
        });
    }
    if (parts.colIdx.length !== arcCount) {
        throw partsError("I1", `colIdx has ${parts.colIdx.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: parts.colIdx.length,
        });
    }
    if (parts.weights !== null && parts.weights.length !== arcCount) {
        throw partsError("I8", `weights has ${parts.weights.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: parts.weights.length,
        });
    }
    const identity = flags.arcToEdgeIsIdentity;
    if ((parts.arcToEdge === null) !== identity || (parts.edgeToArc === null) !== identity) {
        throw partsError("I9", "arcToEdge / edgeToArc must be null exactly when flags.arcToEdgeIsIdentity", {
            flag: "arcToEdgeIsIdentity",
            found: identity,
        });
    }
    if (identity && (!parts.directed || arcCount !== edgeCount)) {
        throw partsError("I9", "flags.arcToEdgeIsIdentity requires a directed snapshot with arcCount === edgeCount", {
            flag: "arcToEdgeIsIdentity",
            directed: parts.directed,
            arcCount,
            edgeCount,
        });
    }
    if (parts.arcToEdge !== null && parts.arcToEdge.length !== arcCount) {
        throw partsError("I5", `arcToEdge has ${parts.arcToEdge.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: parts.arcToEdge.length,
        });
    }
    if (parts.edgeToArc !== null && parts.edgeToArc.length !== edgeCount) {
        throw partsError("I5", `edgeToArc has ${parts.edgeToArc.length} entries, expected ${edgeCount}`, {
            expected: edgeCount,
            found: parts.edgeToArc.length,
        });
    }
    if (parts.ids.size !== nodeCount) {
        throw partsError("I11", `ids.size is ${parts.ids.size}, expected ${nodeCount}`, {
            expected: nodeCount,
            found: parts.ids.size,
        });
    }
    if (parts.nodes.rowCount !== nodeCount || parts.edges.rowCount !== edgeCount || parts.graph.rowCount !== 1) {
        throw partsError("I12", "table row counts do not match nodeCount / edgeCount / 1", {
            nodes: parts.nodes.rowCount,
            edges: parts.edges.rowCount,
            graph: parts.graph.rowCount,
        });
    }
}

/**
 * Record the digests of every immutable column of a table.
 * @param records - the records
 * @param table - the table
 */
function recordColumns(records: ChecksumRecords, table: AttributeTable): void {
    for (const column of table) {
        if (!column.meta.mutable) {
            records.columns.set(column, hashColumn(column));
        }
    }
}

// ============================================================ the class

/**
 * The frozen graph (design section 3): a CSR core over 4-byte typed arrays, the id map, the
 * attribute tables, the flags and a per-instance view cache. Topology, counts, flags, id map and
 * immutable columns never change after construction (invariant I17); the column SET of `nodes` /
 * `edges` / `graph` and the CONTENTS of columns declared mutable are mutable side tables (design
 * section 5.8). Every view is a pure function of the core, memoised once and SHARED: a call returns
 * the cached array itself, so writing into a view is a contract violation; call `.slice()` for
 * scratch. Index-taking queries are total for in-range arguments and unchecked otherwise (design
 * section 11.1). Instances come from `createSnapshot()` (the builder, `fromCsr`, `fromWire` and the
 * derived-graph methods); `isGraphSnapshot()` recognises them structurally by brand and
 * formatVersion.
 */
export class GraphSnapshot implements GraphSnapshotContract {
    /** Symbol.for brand read by isGraphSnapshot() (design section 7.5). */
    readonly [SNAPSHOT_BRAND]: true;
    /** Process-unique identity of the CORE; shared by withColumns() snapshots (design section 5.8). */
    readonly serial: number;
    /** Debugging aid supplied at freeze. */
    readonly label: string | null;
    /** The data-model major (design section 13.5). */
    readonly formatVersion: 1;
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** n (invariant I3). */
    readonly nodeCount: number;
    /** Number of logical edges (invariant I13). */
    readonly edgeCount: number;
    /** colIdx.length (invariants I6, I7). */
    readonly arcCount: number;
    /** Logical edges with source === target. */
    readonly selfLoopCount: number;
    /** nodeCount + 1 row offsets (invariant I1). */
    readonly rowPtr: U32;
    /** Target node index of each arc, sorted within each row (invariants I2, I4). */
    readonly colIdx: U32;
    /** arcCount f32 weights; null when unweighted. */
    readonly weights: F32 | null;
    /** Flags kernels branch on (design section 3.8). */
    readonly flags: SnapshotFlags;
    /** The id map (design section 4). */
    readonly ids: NodeIdMap;
    /** Node attribute table, rowCount === nodeCount. */
    readonly nodes: AttributeTable;
    /** Edge attribute table, rowCount === edgeCount, indexed by logical edge. */
    readonly edges: AttributeTable;
    /** Graph attribute table, rowCount === 1. */
    readonly graph: AttributeTable;
    /** Extension tables keyed by name (design section 5.10). */
    readonly extensions: ReadonlyMap<string, AttributeTable>;
    /** Graph-level metadata (design section 5.9). */
    readonly meta: GraphMeta;
    /** The arena holding the core arrays, or null (design section 10.3). */
    readonly arena: ArenaLayout | null;

    /**
     * The lazily populated members (design section 5.8): defined as a NON-enumerable own property in
     * the constructor (`declare` keeps tsc from emitting a field), so `Object.keys(snapshot)` lists
     * exactly the public fields plus the documented clone guard, and JSON / spread / structured
     * clone never walk the view cache.
     */
    declare private readonly state: SnapshotState;

    /**
     * Wrap prepared parts. Producers call `createSnapshot()`; the constructor is not part of the public
     * surface. The producer has established I1-I13; the constructor checks the O(1) length rules,
     * records checksums when asked, freezes the object and installs the clone guard.
     * @param parts - the parts (no array may alias memory a builder can still write, invariant I18)
     * @internal
     */
    constructor(parts: SnapshotParts) {
        checkParts(parts);
        this[SNAPSHOT_BRAND] = true;
        this.serial = parts.serial ?? nextSerial++;
        this.label = parts.label;
        this.formatVersion = FORMAT_VERSION;
        this.directed = parts.directed;
        this.nodeCount = parts.nodeCount;
        this.edgeCount = parts.edgeCount;
        this.arcCount = parts.arcCount;
        this.selfLoopCount = parts.selfLoopCount;
        this.rowPtr = parts.rowPtr;
        this.colIdx = parts.colIdx;
        this.weights = parts.weights;
        this.flags = Object.freeze({ ...parts.flags });
        this.ids = parts.ids;
        this.nodes = parts.nodes;
        this.edges = parts.edges;
        this.graph = parts.graph;
        this.extensions = parts.extensions;
        this.meta = parts.meta;
        this.arena = parts.arena;
        let checksums: ChecksumRecords | null = null;
        if (parts.checksum) {
            checksums = { core: new Map(), columns: new WeakMap(), views: new Map() };
            checksums.core.set("rowPtr", hashTypedArray(parts.rowPtr));
            checksums.core.set("colIdx", hashTypedArray(parts.colIdx));
            if (parts.weights !== null) {
                checksums.core.set("weights", hashTypedArray(parts.weights));
            }
            if (parts.arcToEdge !== null) {
                checksums.core.set("arcToEdge", hashTypedArray(parts.arcToEdge));
            }
            if (parts.edgeToArc !== null) {
                checksums.core.set("edgeToArc", hashTypedArray(parts.edgeToArc));
            }
            recordColumns(checksums, parts.nodes);
            recordColumns(checksums, parts.edges);
            recordColumns(checksums, parts.graph);
            for (const table of parts.extensions.values()) {
                recordColumns(checksums, table);
            }
        }
        const state: SnapshotState = {
            arcToEdge: parts.arcToEdge,
            edgeToArc: parts.edgeToArc,
            views: emptyViewCache(),
            checksums,
            contentHash: null,
            edgeIds: null,
        };
        Object.defineProperty(this, "state", { value: state, enumerable: false, writable: false, configurable: false });
        STATES.set(this, state);
        claimStorage(parts);
        Object.defineProperty(this, CLONE_GUARD_KEY, {
            value: cloneGuard,
            enumerable: true,
            writable: false,
            configurable: false,
        });
        Object.freeze(this);
    }

    // ---------------------------------------------------------------- core accessors

    /**
     * rowPtr.length === 0: the core buffer was transferred away (design section 9.4). Derived from the
     * array state, never a stored bit, so every holder of the same core agrees.
     * @returns true when detached
     */
    get detached(): boolean {
        return this.rowPtr.length === 0;
    }

    /**
     * Logical edge of every arc (invariant I5); an identity permutation is materialised on first
     * access as a separate 4-byte-aligned array outside the arena. GPU code tests
     * `flags.arcToEdgeIsIdentity` before reading it.
     * @returns arcCount entries; E_DETACHED after a consuming transfer
     */
    get arcToEdge(): U32 {
        this.assertAttached();
        const { state } = this;
        if (state.arcToEdge === null) {
            state.arcToEdge = identityPermutation(this.arcCount);
            state.checksums?.core.set("arcToEdge", hashTypedArray(state.arcToEdge));
        }
        return state.arcToEdge;
    }

    /**
     * The arc holding the declared orientation of every logical edge (invariant I5); an identity
     * permutation is materialised on first access outside the arena.
     * @returns edgeCount entries; E_DETACHED after a consuming transfer
     */
    get edgeToArc(): U32 {
        this.assertAttached();
        const { state } = this;
        if (state.edgeToArc === null) {
            state.edgeToArc = identityPermutation(this.edgeCount);
            state.checksums?.core.set("edgeToArc", hashTypedArray(state.edgeToArc));
        }
        return state.edgeToArc;
    }

    // ---------------------------------------------------------------- queries (3.9)

    /**
     * The out-arc range of a node: [rowPtr[u], rowPtr[u + 1]]. Allocates a tuple; hot loops read
     * rowPtr directly.
     * @param u - the node index
     * @returns the half-open arc range as [start, end]
     */
    outArcs(u: number): readonly [start: number, end: number] {
        this.assertAttached();
        return [this.rowPtr[u], this.rowPtr[u + 1]];
    }

    /**
     * rowPtr[u + 1] - rowPtr[u]: the out-arc count (a self-loop counted once).
     * @param u - the node index
     * @returns the out-degree
     */
    outDegreeOf(u: number): number {
        this.assertAttached();
        return this.rowPtr[u + 1] - this.rowPtr[u];
    }

    /**
     * Binary search for the first arc u -> v (the lowest logical edge index among parallels).
     * @param u - the source node index
     * @param v - the target node index
     * @returns the arc index, or INVALID_INDEX when absent
     */
    findArc(u: number, v: number): number {
        this.assertAttached();
        return findArcIn(this.rowPtr, this.colIdx, u, v);
    }

    /**
     * Whether an arc u -> v exists.
     * @param u - the source node index
     * @param v - the target node index
     * @returns findArc(u, v) !== INVALID_INDEX
     */
    hasArc(u: number, v: number): boolean {
        return this.findArc(u, v) !== INVALID_INDEX;
    }

    /**
     * The half-open arc range [lo, hi) of every arc u -> v; empty when lo === hi. Allocates a tuple.
     * @param u - the source node index
     * @param v - the target node index
     * @returns the range as [lo, hi]
     */
    arcsBetween(u: number, v: number): readonly [lo: number, hi: number] {
        this.assertAttached();
        return arcRangeIn(this.rowPtr, this.colIdx, u, v);
    }

    /**
     * Number of parallel arcs u -> v.
     * @param u - the source node index
     * @param v - the target node index
     * @returns hi - lo of arcsBetween(u, v)
     */
    multiplicity(u: number, v: number): number {
        this.assertAttached();
        return multiplicityIn(this.rowPtr, this.colIdx, u, v);
    }

    /**
     * The row containing an arc: O(1) through the cached coo() view, else a binary search on rowPtr.
     * @param a - the arc index
     * @returns the source node index
     */
    arcSource(a: number): number {
        this.assertAttached();
        const { coo } = this.state.views;
        if (coo !== null) {
            return coo.src[a];
        }
        return arcSourceIn(this.rowPtr, a);
    }

    /**
     * Declared source of a logical edge: arcSource(edgeToArc[e]).
     * @param e - the logical edge index
     * @returns the source node index
     */
    edgeSource(e: number): number {
        this.assertAttached();
        const list = this.state.views.edgeList;
        if (list !== null) {
            return list.src[e];
        }
        return this.arcSource(this.edgeToArc[e]);
    }

    /**
     * Declared target of a logical edge: colIdx[edgeToArc[e]].
     * @param e - the logical edge index
     * @returns the target node index
     */
    edgeTarget(e: number): number {
        return this.colIdx[this.edgeToArc[e]];
    }

    /**
     * Lookup through the role "id" edge column, backed by a Map built on first call and rebuilt when the
     * column is replaced or a mutable id column's version changes (design section 4.6).
     * @param id - the edge id
     * @returns the logical edge index, or INVALID_INDEX on a miss or when no id column exists
     */
    edgeIndexOf(id: EdgeId): number {
        const { state } = this;
        const version = columnSetVersion(this.edges);
        if (state.edgeIds === null || state.edgeIds.version !== version) {
            // the role lookup is repeated only when the column set changed
            const column = this.edges.byRole("id");
            state.edgeIds = { version, column, index: column === null ? null : new EdgeIdIndex(column) };
        }
        return state.edgeIds.index === null ? INVALID_INDEX : state.edgeIds.index.indexOf(id);
    }

    // ---------------------------------------------------------------- views (7.2)

    /**
     * The in-adjacency, rows sorted by source; the forward arrays themselves when undirected. Cached
     * and shared.
     * @returns the reverse view
     */
    reverse(): ReverseView {
        return this.cached("reverse", () => computeReverse(this));
    }

    /**
     * Per-arc COO form; src is the only new array. Cached and shared.
     * @returns the COO view
     */
    coo(): CooView {
        return this.cached("coo", () => computeCoo(this));
    }

    /**
     * Every logical edge once in declared orientation. Cached and shared.
     * @returns the edge list view
     */
    edgeList(): EdgeListView {
        return this.cached("edgeList", () => computeEdgeList(this));
    }

    /**
     * rowPtr differences materialised; a self-loop counted once. Cached and shared: call `.slice()`
     * before writing.
     * @returns Uint32Array(n)
     */
    outDegree(): U32 {
        return this.cached("outDegree", () => rowLengths(this.rowPtr, this.nodeCount));
    }

    /**
     * Arcs targeting each node, via reverse().rowPtr; the same object as outDegree() when undirected.
     * Cached and shared: call `.slice()` before writing.
     * @returns Uint32Array(n)
     */
    inDegree(): U32 {
        return this.cached("inDegree", () =>
            this.directed ? rowLengths(this.reverse().rowPtr, this.nodeCount) : this.outDegree(),
        );
    }

    /**
     * Graph-theoretic degree (NetworkX convention, design section 3.4): in + out when directed; out
     * plus self-loops when undirected. Cached and shared: call `.slice()` before writing.
     * @returns Uint32Array(n)
     */
    degree(): U32 {
        return this.cached("degree", () =>
            sumDegrees(this.outDegree(), this.directed ? this.inDegree() : this.selfLoopsPerNode()),
        );
    }

    /**
     * Row sums of weights (a self-loop arc counted once); outDegree widened when unweighted. F64 for
     * CPU precision; may be 0 for a node with out-arcs. Cached and shared: call `.slice()` before
     * writing.
     * @returns Float64Array(n)
     */
    weightedOutDegree(): F64 {
        return this.cached("weightedOutDegree", () =>
            this.weights === null
                ? widenToF64(this.outDegree())
                : rowWeightSums(this.rowPtr, this.weights, this.nodeCount),
        );
    }

    /**
     * Weight sums over incoming arcs; the same object as weightedOutDegree() when undirected. Cached
     * and shared: call `.slice()` before writing.
     * @returns Float64Array(n)
     */
    weightedInDegree(): F64 {
        return this.cached("weightedInDegree", () => {
            if (!this.directed) {
                return this.weightedOutDegree();
            }
            const reverse = this.reverse();
            return reverse.weights === null
                ? widenToF64(this.inDegree())
                : rowWeightSums(reverse.rowPtr, reverse.weights, this.nodeCount);
        });
    }

    /**
     * NetworkX weighted degree: weightedOutDegree + (directed ? weightedInDegree : selfLoopWeight), so
     * the sum is 2 * totalWeight() when undirected. Cached and shared: call `.slice()` before writing.
     * @returns Float64Array(n)
     */
    weightedDegree(): F64 {
        return this.cached("weightedDegree", () =>
            sumF64(this.weightedOutDegree(), this.directed ? this.weightedInDegree() : this.selfLoopWeight()),
        );
    }

    /**
     * Sum of weights over each node's self-loop arcs (selfLoopsPerNode widened when unweighted).
     * Cached and shared: call `.slice()` before writing.
     * @returns Float64Array(n)
     */
    selfLoopWeight(): F64 {
        return this.cached("selfLoopWeight", () =>
            computeSelfLoopWeight(this, {
                selfLoopArcs: this.selfLoopArcs(),
                selfLoopsPerNode: this.selfLoopsPerNode(),
            }),
        );
    }

    /**
     * Sum of weights over logical edges (each undirected edge once). Cached.
     * @returns the total weight
     */
    totalWeight(): number {
        return this.cached("totalWeight", () => computeTotalWeight(this));
    }

    /**
     * The arcs a with colIdx[a] === row(a), found by binary search per row. Cached and shared: call
     * `.slice()` before writing.
     * @returns Uint32Array(selfLoopCount)
     */
    selfLoopArcs(): U32 {
        return this.cached("selfLoopArcs", () => this.materialiseSelfLoops().selfLoopArcs);
    }

    /**
     * Self-loop arcs per node. Cached and shared: call `.slice()` before writing.
     * @returns Uint32Array(n)
     */
    selfLoopsPerNode(): U32 {
        return this.cached("selfLoopsPerNode", () => this.materialiseSelfLoops().selfLoopsPerNode);
    }

    /**
     * Point query: self-loop arcs at one node, O(log d).
     * @param u - the node index
     * @returns the count
     */
    selfLoopsAt(u: number): number {
        this.assertAttached();
        return selfLoopsAtIn(this.rowPtr, this.colIdx, u);
    }

    /**
     * For every arc, the arc storing the opposite orientation of the same edge (a self-loop maps to
     * itself); the lockstep walk of design section 6.4. Undirected only. Cached and shared: call
     * `.slice()` before writing.
     * @returns Uint32Array(arcCount); E_DIRECTED on a directed snapshot
     */
    mate(): U32 {
        return this.cached("mate", () => computeMate(this));
    }

    /**
     * Nodes ordered by descending out-degree of the forward or reverse adjacency with the cuGraph tier
     * boundaries. Both variants cached; on an undirected snapshot they are the same object.
     * @param options - which adjacency's degree to order by
     * @returns the permutation and its tier offsets
     */
    degreeOrder(options?: DegreeOrderOptions): DegreeOrderView {
        const of = assertOneOf("of", options?.of, ["forward", "reverse"] as const) ?? "forward";
        if (of === "reverse" && this.directed) {
            return this.cached("reverseDegreeOrder", () => computeDegreeOrder(this.reverse().rowPtr, this.nodeCount));
        }
        const forward = this.cached("degreeOrder", () => computeDegreeOrder(this.rowPtr, this.nodeCount));
        if (!this.directed) {
            this.state.views.reverseDegreeOrder = forward;
        }
        return forward;
    }

    /**
     * Whether the arc set is closed under reversal with equal weights (forward row v equals reverse row
     * v for every v); true without work when undirected. Cached.
     * @returns true when symmetric
     */
    isSymmetric(): boolean {
        return this.cached("symmetric", () => (this.directed ? computeSymmetric(this, this.reverse()) : true));
    }

    /**
     * Compute a set of views eagerly (inside freeze() through FreezeOptions.prepare, or off the critical
     * path).
     * @param views - the views to materialise
     * @returns this snapshot
     */
    prepare(views: readonly ViewName[]): this {
        for (const name of views) {
            this.materialiseView(name);
        }
        return this;
    }

    /**
     * Release every cached view (and every identity permutation materialised on demand) and every
     * cached `gpuView()` f32 copy of an f64 column (design section 7.2); they are recomputed on
     * demand. The checksum records of the dropped views are dropped with them.
     */
    dropCaches(): void {
        const { state } = this;
        for (const name of VIEW_NAMES) {
            state.views[name] = null;
        }
        state.checksums?.views.clear();
        if (this.flags.arcToEdgeIsIdentity) {
            state.arcToEdge = null;
            state.edgeToArc = null;
            state.checksums?.core.delete("arcToEdge");
            state.checksums?.core.delete("edgeToArc");
        }
        for (const table of [this.nodes, this.edges, this.graph, ...this.extensions.values()]) {
            for (const column of table) {
                dropGpuViewCache(column);
            }
        }
    }

    /**
     * Which views are resident.
     * @returns the names of the cached views, in the fixed ViewName order
     */
    cachedViews(): readonly ViewName[] {
        if (this.detached) {
            // design section 9.1: the view caches of a detached snapshot are dropped
            this.dropCaches();
            return [];
        }
        const { views } = this.state;
        return VIEW_NAMES.filter((name) => views[name] !== null);
    }

    // ---------------------------------------------------------------- derived graphs (7.3)

    /**
     * Every directed edge becomes undirected; reciprocal pairs collapse to one edge keeping the lower
     * index's row (keep-first). Returns `{ snapshot: this, null maps }` on an undirected snapshot.
     * @param options - reciprocal filtering and the weight reducer
     * @returns the derived graph
     */
    toUndirected(options?: ToUndirectedOptions): DerivedGraph {
        this.assertAttached();
        if (!this.directed) {
            return this.wrapDerived(identityDerived());
        }
        return this.wrapDerived(deriveToUndirected(this, this.edgeEndpoints(), options));
    }

    /**
     * Orientation of every edge swapped: the reverse view's arrays become the core. Returns this on an
     * undirected snapshot.
     * @returns the derived graph (same node and edge spaces)
     */
    transpose(): DerivedGraph {
        this.assertAttached();
        if (!this.directed) {
            return this.wrapDerived(identityDerived());
        }
        return this.wrapDerived(deriveTranspose(this, this.reverse()));
    }

    /**
     * One edge per (u, v) group (parallels are adjacent by invariant I4), survivor = lowest index;
     * flags.multigraph is false afterwards.
     * @param options - reducers and self-loop policy
     * @returns the derived graph
     */
    simplified(options?: SimplifyOptions): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveSimplified(this, this.edgeEndpoints(), options));
    }

    /**
     * Every edge with source !== target.
     * @returns the derived graph
     */
    withoutSelfLoops(): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveWithoutSelfLoops(this, this.edgeEndpoints()));
    }

    /**
     * Keep the logical edges whose mask bit is set.
     * @param keep - packed bitmap over logical edges; E_MASK_LENGTH when shorter than ceil(edgeCount / 32) words
     * @returns the derived graph
     */
    filterEdges(keep: EdgeMask): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveFilterEdges(this, this.edgeEndpoints(), keep));
    }

    /**
     * The subgraph induced by a node selection: an index list (order = new index order; E_INDEX_RANGE
     * for an out-of-range or repeated index) or a packed mask (ascending order; E_MASK_LENGTH when
     * short). Edges with both endpoints kept.
     * @param selection - the node indices or a mask
     * @returns the derived graph
     */
    inducedSubgraph(selection: U32 | { readonly mask: NodeMask }): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveInducedSubgraph(this, this.edgeEndpoints(), selection));
    }

    /**
     * Contract the nodes of each partition block into one node (design section 7.3).
     * @param partition - one label per node; E_PARTITION for a wrong length or an INVALID_INDEX label
     * @param options - weight reducer, self-loop and parallel policies, column reducers
     * @returns the derived graph with blockSizes and an identity id map
     */
    contract(partition: U32, options?: ContractOptions): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveContract(this, this.edgeEndpoints(), partition, options));
    }

    /**
     * Permute the node space: perm[newIndex] = oldIndex; the id map follows; edge order is preserved.
     * @param perm - a permutation of 0..n-1; E_INVALID_PERMUTATION otherwise
     * @returns the derived graph
     */
    relabel(perm: U32): DerivedGraph {
        this.assertAttached();
        return this.wrapDerived(deriveRelabel(this, this.edgeEndpoints(), perm));
    }

    /**
     * A new snapshot object sharing the core, the id map and the serial with a CLONED column set plus
     * the given columns (the only operation that clones the column set).
     * @param nodes - node columns to add, keyed by name
     * @param edges - edge columns to add, keyed by name
     * @returns the new snapshot
     */
    withColumns(
        nodes?: Readonly<Record<string, TypedArrayData | ColumnInput>>,
        edges?: Readonly<Record<string, TypedArrayData | ColumnInput>>,
    ): GraphSnapshot {
        this.assertAttached();
        const identity = this.flags.arcToEdgeIsIdentity;
        const parts = withColumnsParts(
            this,
            nodes,
            edges,
            identity ? null : this.arcToEdge,
            identity ? null : this.edgeToArc,
        );
        return new GraphSnapshot({ ...parts, checksum: this.state.checksums !== null });
    }

    // ---------------------------------------------------------------- memory, transfer, checks (9, 11)

    /**
     * Resident bytes of the core (an identity permutation counts zero even when materialised), plus
     * the side structures selected by the options.
     * @param options - which side structures to include
     * @returns the byte count
     */
    byteLength(options: ByteLengthOptions = {}): number {
        let bytes = this.rowPtr.byteLength + this.colIdx.byteLength;
        if (this.weights !== null) {
            bytes += this.weights.byteLength;
        }
        if (!this.flags.arcToEdgeIsIdentity) {
            bytes += this.arcToEdge.byteLength + this.edgeToArc.byteLength;
        }
        if (options.views === true) {
            const { views } = this.state;
            for (const name of VIEW_NAMES) {
                const value = views[name];
                if (value !== null && !(name === "reverseDegreeOrder" && value === views.degreeOrder)) {
                    bytes += viewByteLength(this, name, value);
                }
            }
        }
        if (options.columns === true) {
            for (const table of [this.nodes, this.edges, this.graph, ...this.extensions.values()]) {
                for (const column of table) {
                    bytes += column.byteLength;
                }
            }
        }
        if (options.ids === true) {
            bytes += this.ids.byteLength();
        }
        return bytes;
    }

    /**
     * A 64-bit content hash of the core arrays (two 32-bit FNV-1a lanes) as 16 hex characters,
     * computed lazily and cached (design section 9.3).
     * @returns the hash
     */
    contentHash(): string {
        this.assertAttached();
        const { state } = this;
        state.contentHash ??= contentHashOf(this);
        return state.contentHash;
    }

    /**
     * The distinct, exclusively owned backing buffers: the postMessage transfer list (design section
     * 9.1). Buffers shared with another holder (a withColumns() sibling, a derived graph sharing the
     * node table) are excluded.
     * @returns the buffers
     */
    transferables(): ArrayBuffer[] {
        this.assertAttached();
        return transferables(this);
    }

    /**
     * The plain-object wire form (design section 9.1): a JSON-serialisable manifest plus the backing
     * buffers. With `transfer: true` only exclusively owned buffers are listed as transferables and
     * shared ones are copied.
     * @param options - transfer mode, views and columns to include
     * @returns the wire snapshot
     */
    toWire(options?: ToWireOptions): WireSnapshot {
        this.assertAttached();
        return toWire(this, options);
    }

    /**
     * The GSNP byte container as one contiguous buffer (design section 9.2).
     * @param options - views to include
     * @returns the container bytes
     */
    toBytes(options?: ToBytesOptions): U8 {
        this.assertAttached();
        return toBytes(this, options);
    }

    /**
     * The GSNP container as a sequence of chunks (design section 9.2): the header plus manifest,
     * then one chunk per non-empty segment.
     * @param options - views to include
     * @returns the chunks
     */
    toByteChunks(options?: ToBytesOptions): Iterable<U8> {
        this.assertAttached();
        return toByteChunks(this, options);
    }

    /**
     * Check the invariants (design section 11.4): the recorded checksums first when `checksum` is set
     * (a mutated frozen array is the most useful diagnosis, details.reason "checksum"; "no-checksum"
     * when none were recorded), then "structure" or "full" (default) per the level table of design
     * section 9.5. Throws E_INVALID_SNAPSHOT with details.invariant and the location on the first
     * violation.
     * @param options - level and checksum comparison
     */
    validate(options: ValidateOptions = {}): void {
        this.assertAttached();
        const level = assertOneOf("level", options.level, ["structure", "full"] as const) ?? "full";
        if (typeof options.checksum !== "boolean" && options.checksum !== undefined) {
            throw new GraphFormatError("E_UNSUPPORTED", "validate option checksum must be a boolean", {
                field: "checksum",
                found: options.checksum,
                reason: "unsupported option",
            });
        }
        // a column or id store transferred away underneath the snapshot (a holder outside the owner
        // count, design section 9.1) is E_DETACHED, not a length-rule violation
        if (idMapDetached(this.ids)) {
            throw new GraphFormatError("E_DETACHED", "the id map's storage was transferred away", {
                serial: this.serial,
            });
        }
        for (const table of [this.nodes, this.edges, this.graph, ...this.extensions.values()]) {
            for (const column of table) {
                if (isColumnDetached(column)) {
                    throw new GraphFormatError("E_DETACHED", `column "${column.meta.name}" was transferred away`, {
                        column: column.meta.name,
                        serial: this.serial,
                    });
                }
            }
        }
        if (options.checksum === true) {
            this.verifyChecksums();
        }
        if (level === "structure") {
            validateStructure(this);
        } else {
            validateFull(this);
        }
    }

    // ---------------------------------------------------------------- private helpers

    /**
     * Throw E_DETACHED when the core was transferred away.
     */
    private assertAttached(): void {
        if (this.rowPtr.length === 0) {
            throw new GraphFormatError("E_DETACHED", "the snapshot's core buffer was transferred away", {
                serial: this.serial,
            });
        }
    }

    /**
     * Return the cached value of a view, computing and recording it on first use.
     * @param name - the view name
     * @param compute - the computation
     * @returns the cached value
     */
    private cached<K extends ViewName>(name: K, compute: () => NonNullable<ViewCache[K]>): NonNullable<ViewCache[K]> {
        this.assertAttached();
        const { views } = this.state;
        const existing = views[name];
        if (existing !== null) {
            return existing;
        }
        const value = compute();
        views[name] = value;
        this.recordView(name, value);
        return value;
    }

    /**
     * Compute both self-loop views in one pass and seed the slot the caller did not ask for.
     * @returns the two views
     */
    private materialiseSelfLoops(): { selfLoopArcs: U32; selfLoopsPerNode: U32 } {
        const loops = computeSelfLoops(this);
        const { views } = this.state;
        if (views.selfLoopArcs === null) {
            views.selfLoopArcs = loops.selfLoopArcs;
            this.recordView("selfLoopArcs", loops.selfLoopArcs);
        }
        if (views.selfLoopsPerNode === null) {
            views.selfLoopsPerNode = loops.selfLoopsPerNode;
            this.recordView("selfLoopsPerNode", loops.selfLoopsPerNode);
        }
        return { selfLoopArcs: views.selfLoopArcs, selfLoopsPerNode: views.selfLoopsPerNode };
    }

    /**
     * Record the digests of a freshly materialised view when checksums are on.
     * @param name - the view name
     * @param value - the view
     */
    private recordView(name: ViewName, value: unknown): void {
        recordViewDigests(this, this.state, name, value);
    }

    /**
     * Compare every recorded checksum with the current bytes (design section 5.8).
     */
    private verifyChecksums(): void {
        const records = this.state.checksums;
        if (records === null) {
            throw new GraphFormatError("E_INVALID_SNAPSHOT", "no checksums were recorded for this snapshot", {
                reason: "no-checksum",
            });
        }
        const mismatch = (what: string, details: Readonly<Record<string, unknown>>): GraphFormatError =>
            new GraphFormatError("E_INVALID_SNAPSHOT", `checksum mismatch: ${what} was modified after freeze`, {
                reason: "checksum",
                ...details,
            });
        const { state } = this;
        const core: readonly [CoreArrayName, ArrayBufferView | null][] = [
            ["rowPtr", this.rowPtr],
            ["colIdx", this.colIdx],
            ["weights", this.weights],
            ["arcToEdge", state.arcToEdge],
            ["edgeToArc", state.edgeToArc],
        ];
        for (const [name, array] of core) {
            const recorded = records.core.get(name);
            if (recorded !== undefined && array !== null && hashTypedArray(array) !== recorded) {
                throw mismatch(`core array ${name}`, { array: name });
            }
        }
        const tables: readonly [string, AttributeTable][] = [
            ["nodes", this.nodes],
            ["edges", this.edges],
            ["graph", this.graph],
            ...[...this.extensions].map(([name, table]): [string, AttributeTable] => [`extensions[${name}]`, table]),
        ];
        for (const [tableName, table] of tables) {
            for (const column of table) {
                const recorded = records.columns.get(column);
                if (recorded !== undefined && hashColumn(column) !== recorded) {
                    throw mismatch(`column "${column.meta.name}" of ${tableName}`, {
                        table: tableName,
                        column: column.meta.name,
                    });
                }
            }
        }
        for (const name of VIEW_NAMES) {
            const value = state.views[name];
            if (value === null) {
                continue;
            }
            for (const [member, array] of Object.entries(viewArrays(this, name, value))) {
                const recorded = records.views.get(`${name}.${member}`);
                if (recorded !== undefined && hashTypedArray(array) !== recorded) {
                    throw mismatch(`view ${name}.${member}`, { view: name, member });
                }
            }
        }
    }

    /**
     * The edge endpoints in declared orientation for the derived-graph functions: the cached edge list
     * when resident, else a fresh one that is NOT cached (derived graphs must not grow the source's
     * resident size, design section 7.3).
     * @returns an edge list view
     */
    private edgeEndpoints(): EdgeListView {
        return this.state.views.edgeList ?? computeEdgeList(this);
    }

    /**
     * Wrap derived parts into a DerivedGraph, creating the new snapshot (with this snapshot's checksum
     * setting) or returning this for the identity cases.
     * @param derived - the derived parts
     * @returns the derived graph
     */
    private wrapDerived(derived: DerivedParts): DerivedGraph {
        const snapshot =
            derived.parts === null
                ? this
                : new GraphSnapshot({ ...derived.parts, checksum: this.state.checksums !== null });
        return {
            snapshot,
            nodeOrigin: derived.nodeOrigin,
            edgeOrigin: derived.edgeOrigin,
            nodeRemap: derived.nodeRemap,
            edgeRemap: derived.edgeRemap,
            blockSizes: derived.blockSizes,
            report: derived.report,
        };
    }

    /**
     * Materialise one named view (the prepare() dispatch).
     * @param name - the view name
     */
    private materialiseView(name: ViewName): void {
        switch (name) {
            case "reverse":
                this.reverse();
                break;
            case "coo":
                this.coo();
                break;
            case "edgeList":
                this.edgeList();
                break;
            case "outDegree":
                this.outDegree();
                break;
            case "inDegree":
                this.inDegree();
                break;
            case "degree":
                this.degree();
                break;
            case "weightedOutDegree":
                this.weightedOutDegree();
                break;
            case "weightedInDegree":
                this.weightedInDegree();
                break;
            case "weightedDegree":
                this.weightedDegree();
                break;
            case "selfLoopWeight":
                this.selfLoopWeight();
                break;
            case "totalWeight":
                this.totalWeight();
                break;
            case "selfLoopArcs":
                this.selfLoopArcs();
                break;
            case "selfLoopsPerNode":
                this.selfLoopsPerNode();
                break;
            case "mate":
                this.mate();
                break;
            case "degreeOrder":
                this.degreeOrder();
                break;
            case "reverseDegreeOrder":
                this.degreeOrder({ of: "reverse" });
                break;
            case "symmetric":
                this.isSymmetric();
                break;
            default: {
                const unknown: never = name;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown view ${String(unknown)}`, { view: unknown });
            }
        }
    }
}

/**
 * Record this snapshot as a holder of every buffer, table and id map it keeps (design section 9.1):
 * a second snapshot claiming the same storage (a withColumns() sibling, a derived graph sharing the
 * node table) marks it shared, so a transfer copies it instead of emptying the sibling. Each distinct
 * buffer is claimed once per snapshot (the arena backs several core arrays).
 * @param parts - the parts the snapshot was built from
 */
function claimStorage(parts: SnapshotParts): void {
    const buffers = new Set<ArrayBuffer>([parts.rowPtr.buffer, parts.colIdx.buffer]);
    if (parts.weights !== null) {
        buffers.add(parts.weights.buffer);
    }
    if (parts.arcToEdge !== null) {
        buffers.add(parts.arcToEdge.buffer);
    }
    if (parts.edgeToArc !== null) {
        buffers.add(parts.edgeToArc.buffer);
    }
    for (const buffer of idMapBuffers(parts.ids)) {
        buffers.add(buffer);
    }
    for (const buffer of buffers) {
        claimHolder(buffer);
    }
    claimHolder(parts.ids);
    claimHolder(parts.nodes);
    claimHolder(parts.edges);
    claimHolder(parts.graph);
    for (const table of parts.extensions.values()) {
        claimHolder(table);
    }
}

// ============================================================ factory and helpers

/**
 * Create a snapshot from prepared parts (design section 6.3 step 12): the constructor freezes the
 * object, records checksums when `parts.checksum` is set and allocates the per-instance view cache.
 * @param parts - the parts; I1-I13 established by the producer, O(1) length rules checked here
 * @returns the frozen snapshot
 */
export function createSnapshot(parts: SnapshotParts): GraphSnapshot {
    return new GraphSnapshot(parts);
}

/**
 * Structural check for a snapshot (design section 7.5): the SNAPSHOT_BRAND property plus the
 * formatVersion, never instanceof, so a duplicated package copy still interoperates.
 * @param x - any value
 * @returns true when x is a GraphSnapshot of this format version
 */
export function isGraphSnapshot(x: unknown): x is GraphSnapshot {
    if (typeof x !== "object" || x === null) {
        return false;
    }
    const candidate = x as { [SNAPSHOT_BRAND]?: unknown; formatVersion?: unknown };
    return candidate[SNAPSHOT_BRAND] === true && candidate.formatVersion === FORMAT_VERSION;
}

/**
 * Whether two typed arrays hold the same elements.
 * @param a - the first array
 * @param b - the second array
 * @returns true when equal element by element
 */
function sameElements(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

/**
 * Compare the core arrays and id maps of two snapshots element by element (design section 7.5):
 * directedness, counts, rowPtr, colIdx, weights (both absent or equal), arcToEdge / edgeToArc
 * (identity compared without materialising) and every id (SameValueZero).
 * @param a - the first snapshot
 * @param b - the second snapshot
 * @returns true when the topologies and id maps are equal
 */
export function equalsTopology(a: GraphSnapshot, b: GraphSnapshot): boolean {
    if (
        a.directed !== b.directed ||
        a.nodeCount !== b.nodeCount ||
        a.edgeCount !== b.edgeCount ||
        a.arcCount !== b.arcCount ||
        a.selfLoopCount !== b.selfLoopCount
    ) {
        return false;
    }
    if (!sameElements(a.rowPtr, b.rowPtr) || !sameElements(a.colIdx, b.colIdx)) {
        return false;
    }
    if ((a.weights === null) !== (b.weights === null)) {
        return false;
    }
    if (a.weights !== null && b.weights !== null && !sameElements(a.weights, b.weights)) {
        return false;
    }
    const identityA = a.flags.arcToEdgeIsIdentity;
    const identityB = b.flags.arcToEdgeIsIdentity;
    if (!identityA || !identityB) {
        if (!sameElements(a.arcToEdge, b.arcToEdge) || !sameElements(a.edgeToArc, b.edgeToArc)) {
            return false;
        }
    }
    if (a.ids.size !== b.ids.size) {
        return false;
    }
    for (let i = 0; i < a.nodeCount; i++) {
        const idA = a.ids.idOf(i);
        const idB = b.ids.idOf(i);
        if (
            idA !== idB &&
            !(typeof idA === "number" && typeof idB === "number" && Number.isNaN(idA) && Number.isNaN(idB))
        ) {
            return false;
        }
    }
    return true;
}

/**
 * Record the digests of a materialised view in the checksum records, when the snapshot keeps any.
 * @param snapshot - the snapshot
 * @param state - its state record
 * @param name - the view name
 * @param value - the view
 */
function recordViewDigests(snapshot: GraphSnapshot, state: SnapshotState, name: ViewName, value: unknown): void {
    const records = state.checksums;
    if (records === null) {
        return;
    }
    for (const [member, array] of Object.entries(viewArrays(snapshot, name, value))) {
        records.views.set(`${name}.${member}`, hashTypedArray(array));
    }
}

/**
 * Install a view value the wire carried (`includeViews`, design section 9.1) into the snapshot's
 * cache, so the receiver does not recompute it. The caller has already checked that the value has
 * the shape of the view; the slot is only filled when it is still empty.
 * @param snapshot - the snapshot
 * @param name - the view name
 * @param value - the view value
 * @returns true when the view was installed, false when the slot was already resident
 */
export function seedView<K extends ViewName>(
    snapshot: GraphSnapshot,
    name: K,
    value: NonNullable<ViewCache[K]>,
): boolean {
    const state = STATES.get(snapshot);
    if (state === undefined || state.views[name] !== null) {
        return false;
    }
    state.views[name] = value;
    recordViewDigests(snapshot, state, name, value);
    return true;
}

/**
 * The cached value of a view without computing it (for tests and for the wire module's
 * `includeViews`).
 * @param snapshot - the snapshot
 * @param name - the view name
 * @returns the cached value, or null when not resident
 */
export function peekView<K extends ViewName>(snapshot: GraphSnapshot, name: K): ViewCache[K] {
    const state = STATES.get(snapshot);
    if (state === undefined) {
        return null;
    }
    return state.views[name];
}

/**
 * Whether the snapshot's permutation arrays are held (materialised) rather than lazy, for tests of
 * the identity rule and for the wire module (which never serialises an identity permutation).
 * @param snapshot - the snapshot
 * @returns true when arcToEdge has been materialised or was supplied
 */
export function permutationMaterialised(snapshot: GraphSnapshot): boolean {
    const state = STATES.get(snapshot);
    return state !== undefined && state.arcToEdge !== null;
}

/**
 * Whether checksums were recorded at construction (design section 5.8).
 * @param snapshot - the snapshot
 * @returns true when validate({ checksum: true }) can compare
 */
export function hasChecksums(snapshot: GraphSnapshot): boolean {
    const state = STATES.get(snapshot);
    return state !== undefined && state.checksums !== null;
}

// ============================================================ arena helpers (10.3)
