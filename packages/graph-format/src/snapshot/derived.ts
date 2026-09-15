/**
 * Derived graphs (design section 7.3): `toUndirected`, `transpose`, `simplified`, `withoutSelfLoops`,
 * `filterEdges`, `inducedSubgraph`, `contract`, `relabel` and `withColumns`, each returning the
 * `SnapshotParts` of a NEW snapshot plus the index maps back to the source (`DerivedParts`), and the
 * `renumberPartition` helper of design section 7.5. GraphSnapshot wraps the parts with
 * `createSnapshot()`; this module imports the class type only, so there is no runtime cycle.
 *
 * Every derived core is built by `buildCore()`, the two stable counting-sort passes of design section
 * 6.3 steps 3-6 over a per-logical-edge list (src, dst, weight) in declared orientation, into a fresh
 * 256-aligned arena (section 10.3), so every derived snapshot inherits invariants I1-I10 by
 * construction and `flags` are computed by the one predicate set in validate.ts (I9).
 *
 * Index-map conventions (design section 7.3): `nodeOrigin` / `edgeOrigin` are NEW -> SOURCE,
 * `nodeRemap` / `edgeRemap` SOURCE -> NEW with INVALID_INDEX for a dropped row and the SURVIVOR's new
 * index for a merged edge; each map is null exactly when its index space is unchanged (same length,
 * identity). New edges are numbered in ascending order of their survivor's source index, so relative
 * edge order is preserved (I14). Attribute tables follow the propagation table of design section
 * 5.11: same node space -> the SAME `AttributeTable` instance; gathered otherwise; `refersTo` values
 * rewritten through the matching remap.
 */

import { sortIntoCore } from "../builder/counting-sort.js";
import { bitmapClear, makeBitmap } from "../columns/bitmap.js";
import { createColumn, emptyParts, resolveColumnMeta } from "../columns/column.js";
import { gatherColumn, gatherTable, remapTable } from "../columns/remap.js";
import { AttributeTable as AttributeTableClass, tableWithColumns } from "../columns/table.js";
import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { gatherNodeIdMap, identityNodeIdMap } from "../ids/node-id-map.js";
import {
    type ArenaLayout,
    type AttributeTable,
    type Column,
    type ColumnInput,
    type ColumnReducer,
    type ContractOptions,
    type EdgeListView,
    type EdgeMask,
    type F32,
    type GraphSnapshot,
    type NodeIdMap,
    type NodeMask,
    type ReverseView,
    type SimplifyOptions,
    type SnapshotFlags,
    type ToUndirectedOptions,
    type TypedArrayData,
    type U32,
    type WeightReducer,
} from "../types/index.js";
import { type SnapshotParts } from "../types/internal.js";
import { checkMaskLength, maskTest, maskToIndices } from "../util/mask.js";
import { assertOneOf, COLUMN_REDUCERS, WEIGHT_REDUCERS } from "../util/options.js";
import { noteShared } from "../util/shared-buffers.js";
import { arcRangeIn } from "./queries.js";
import { computeFlags, isIdentity } from "./validate.js";

// ============================================================ result shape

/** The report of a derived graph: how many source edges were dropped and how many merged into a survivor. */
interface DerivedReport {
    /** Source edges with no counterpart in the derived graph. */
    readonly droppedEdges: number;
    /** Source edges collapsed into another edge (the survivor). */
    readonly mergedEdges: number;
}

/**
 * What a derive function returns: the parts of the new snapshot (null when the operation is the
 * identity and the caller returns the source itself) and the maps of `DerivedGraph`.
 */
export interface DerivedParts {
    /** The parts of the new snapshot, or null for "return this". */
    readonly parts: SnapshotParts | null;
    /** New node index -> source node index, or null when the node space is unchanged. */
    readonly nodeOrigin: U32 | null;
    /** New edge index -> source edge index (survivor), or null when the edge space is unchanged. */
    readonly edgeOrigin: U32 | null;
    /** Source node index -> new node index or INVALID_INDEX, or null when unchanged. */
    readonly nodeRemap: U32 | null;
    /** Source edge index -> new edge index (survivor) or INVALID_INDEX, or null when unchanged. */
    readonly edgeRemap: U32 | null;
    /** contract only: source nodes per block. */
    readonly blockSizes: U32 | null;
    /** Dropped and merged edge counts. */
    readonly report: DerivedReport;
}

const NO_REPORT: DerivedReport = Object.freeze({ droppedEdges: 0, mergedEdges: 0 });

/**
 * The DerivedParts of an identity operation: the caller returns the source snapshot itself.
 * @returns parts null, every map null, an empty report
 */
export function identityDerived(): DerivedParts {
    return {
        parts: null,
        nodeOrigin: null,
        edgeOrigin: null,
        nodeRemap: null,
        edgeRemap: null,
        blockSizes: null,
        report: NO_REPORT,
    };
}

// ============================================================ core builder (6.3 steps 3-6, 10.3)

/** A per-logical-edge description of a graph in declared orientation: the input of `buildCore()`. */
interface EdgeArrays {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** The node count. */
    readonly nodeCount: number;
    /** The logical edge count. */
    readonly edgeCount: number;
    /** Declared source of every edge. */
    readonly src: U32;
    /** Declared target of every edge. */
    readonly dst: U32;
    /** Per-edge f32 weights, or null when unweighted. */
    readonly weights: F32 | null;
}

/** The core a `buildCore()` call produces: exactly the core fields of SnapshotParts. */
interface BuiltCore {
    /** colIdx.length. */
    readonly arcCount: number;
    /** Logical edges with source === target. */
    readonly selfLoopCount: number;
    /** nodeCount + 1 row offsets. */
    readonly rowPtr: U32;
    /** Sorted targets. */
    readonly colIdx: U32;
    /** Per-arc weights, or null. */
    readonly weights: F32 | null;
    /** arcCount entries, or null when the permutation is the identity. */
    readonly arcToEdge: U32 | null;
    /** edgeCount entries, or null when the permutation is the identity. */
    readonly edgeToArc: U32 | null;
    /** The truthful flags (I9). */
    readonly flags: SnapshotFlags;
    /** The arena the arrays are views into, or null when `arena: false`. */
    readonly arena: ArenaLayout | null;
}

/** Options of `buildCore()`. */
interface BuildCoreOptions {
    /** Allocate the arrays inside one 256-aligned arena (default true) or as separate buffers. */
    readonly arena?: boolean | undefined;
}

/**
 * Build the core of a derived graph from a per-logical-edge list in declared orientation: the two
 * stable counting-sort passes of design section 6.3 steps 3-6 (`sortIntoCore` of the builder module,
 * so derived graphs and freezes share one implementation), into a fresh 256-aligned arena unless
 * `arena: false`. Invariants I1-I10 hold by construction and the flags are computed from the stored
 * values (I9). A NaN weight (a reducer summing +Infinity and -Infinity) is E_INVALID_WEIGHT (I8).
 * @param edges - the per-edge arrays; only the first `edgeCount` entries are read
 * @param options - arena or separate buffers
 * @returns the core, the counts and the flags
 */
export function buildCore(edges: EdgeArrays, options: BuildCoreOptions = {}): BuiltCore {
    const result = sortIntoCore(edges, options.arena !== false);
    const { core } = result;
    return {
        arcCount: result.arcCount,
        selfLoopCount: result.selfLoopCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags: result.flags,
        arena: core.arena,
    };
}

// ============================================================ assembling parts

/** The side structures of a derived snapshot. */
interface DerivedSides {
    readonly directed: boolean;
    readonly nodeCount: number;
    readonly edgeCount: number;
    readonly ids: NodeIdMap;
    readonly nodes: AttributeTable;
    readonly edges: AttributeTable;
    readonly extensions: ReadonlyMap<string, AttributeTable>;
}

/**
 * Combine a built core with the side structures into SnapshotParts (serial null: a new core identity;
 * checksum false: the caller substitutes its own setting).
 * @param source - the source snapshot (label, meta and graph table are carried over)
 * @param core - the built core
 * @param sides - the side structures
 * @returns the parts
 */
function assembleParts(source: GraphSnapshot, core: BuiltCore, sides: DerivedSides): SnapshotParts {
    return {
        label: source.label,
        serial: null,
        directed: sides.directed,
        nodeCount: sides.nodeCount,
        edgeCount: sides.edgeCount,
        arcCount: core.arcCount,
        selfLoopCount: core.selfLoopCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags: core.flags,
        ids: sides.ids,
        nodes: sides.nodes,
        edges: sides.edges,
        graph: source.graph,
        extensions: sides.extensions,
        meta: source.meta,
        arena: core.arena,
        checksum: false,
    };
}

/**
 * Extension tables with refersTo values rewritten through the remaps; the source map itself when
 * neither space changed.
 * @param source - the source snapshot
 * @param nodeRemap - the node remap or null
 * @param edgeRemap - the edge remap or null
 * @returns the extension map to use
 */
function remapExtensions(
    source: GraphSnapshot,
    nodeRemap: U32 | null,
    edgeRemap: U32 | null,
): ReadonlyMap<string, AttributeTable> {
    if ((nodeRemap === null && edgeRemap === null) || source.extensions.size === 0) {
        return source.extensions;
    }
    const out = new Map<string, AttributeTable>();
    for (const [name, table] of source.extensions) {
        out.set(name, remapTable(table, null, table.rowCount, { node: nodeRemap, edge: edgeRemap }));
    }
    return out;
}

// ============================================================ edge selection and grouping

/**
 * A grouping of source edges into new edges: `remap` (source -> new or INVALID_INDEX), `origin` (new
 * -> survivor = lowest source index of the group) and the new count. Null maps mean "unchanged".
 */
interface EdgeGrouping {
    readonly count: number;
    readonly remap: U32 | null;
    readonly origin: U32 | null;
    readonly dropped: number;
    readonly merged: number;
}

/**
 * Build the new -> survivor map and count from a source -> new remap whose survivors are numbered in
 * ascending source order.
 * @param remap - source edge -> new edge or INVALID_INDEX, survivors ascending
 * @param count - the new edge count
 * @returns the origin map
 */
function originOf(remap: U32, count: number): U32 {
    const origin = new Uint32Array(count).fill(INVALID_INDEX);
    for (let e = 0; e < remap.length; e++) {
        const d = remap[e];
        if (d !== INVALID_INDEX && origin[d] === INVALID_INDEX) {
            origin[d] = e;
        }
    }
    return origin;
}

/**
 * Turn a per-edge "survivor of e" array (`INVALID_INDEX` = dropped, `e` itself = survivor, another
 * index = merged into that survivor) into an EdgeGrouping with new indices in ascending survivor order.
 * @param survivorOf - source edge -> its survivor, or INVALID_INDEX
 * @returns the grouping; maps null when every edge survives on its own
 */
function groupingFromSurvivors(survivorOf: U32): EdgeGrouping {
    const edgeCount = survivorOf.length;
    const newIndex = new Uint32Array(edgeCount).fill(INVALID_INDEX);
    let count = 0;
    let dropped = 0;
    let merged = 0;
    for (let e = 0; e < edgeCount; e++) {
        if (survivorOf[e] === e) {
            newIndex[e] = count++;
        } else if (survivorOf[e] === INVALID_INDEX) {
            dropped++;
        } else {
            merged++;
        }
    }
    if (dropped === 0 && merged === 0) {
        return { count, remap: null, origin: null, dropped, merged };
    }
    const remap = new Uint32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        const s = survivorOf[e];
        remap[e] = s === INVALID_INDEX ? INVALID_INDEX : newIndex[s];
    }
    return { count, remap, origin: originOf(remap, count), dropped, merged };
}

/**
 * Whether two arrays of node indices are the identity selection.
 * @param origin - new -> source
 * @param nodeCount - the source node count
 * @returns true when origin is 0..nodeCount-1
 */
function isIdentitySelection(origin: U32, nodeCount: number): boolean {
    return origin.length === nodeCount && isIdentity(origin);
}

/**
 * The declared endpoints and weights of the surviving edges, endpoints mapped through `nodeRemap`
 * when given, in new-index order.
 * @param list - the source edge list
 * @param grouping - the grouping (origin null = all edges in order)
 * @param nodeRemap - source node -> new node, or null
 * @returns the edge arrays without weights (weights are reduced separately)
 */
function gatherEndpoints(list: EdgeListView, grouping: EdgeGrouping, nodeRemap: U32 | null): { src: U32; dst: U32 } {
    const { count, origin } = grouping;
    const src = new Uint32Array(count);
    const dst = new Uint32Array(count);
    for (let d = 0; d < count; d++) {
        const e = origin === null ? d : origin[d];
        src[d] = nodeRemap === null ? list.src[e] : nodeRemap[list.src[e]];
        dst[d] = nodeRemap === null ? list.dst[e] : nodeRemap[list.dst[e]];
    }
    return { src, dst };
}

/**
 * Check every reducer of a per-column reducer record (design section 7.3) before anything is built.
 * @param field - the option name
 * @param reducers - the record, or undefined
 */
function checkReducers(field: string, reducers: Readonly<Record<string, unknown>> | undefined): void {
    if (reducers === undefined) {
        return;
    }
    for (const name of Object.keys(reducers)) {
        assertOneOf(`${field}.${name}`, reducers[name], COLUMN_REDUCERS);
    }
}

/**
 * Reduce per-edge weights over a grouping (design section 7.3): `"first"` keeps the survivor's
 * (lowest index) weight, `"last"` the highest index's, `"sum"` / `"min"` / `"max"` combine the group.
 * On an unweighted source `"sum"` yields the multiplicities as an F32 and every other reducer null.
 * @param weights - the per-edge source weights, or null
 * @param grouping - the grouping
 * @param reducer - the weight reducer
 * @returns the per-new-edge weights, or null
 */
function reduceWeights(weights: F32 | null, grouping: EdgeGrouping, reducer: WeightReducer): F32 | null {
    const { count, remap, origin } = grouping;
    if (weights === null) {
        if (reducer !== "sum") {
            return null;
        }
        const out = new Float32Array(count);
        if (remap === null) {
            out.fill(1);
            return out;
        }
        for (let e = 0; e < remap.length; e++) {
            if (remap[e] !== INVALID_INDEX) {
                out[remap[e]]++;
            }
        }
        return out;
    }
    if (remap === null || origin === null) {
        return weights;
    }
    const out = new Float32Array(count);
    for (let d = 0; d < count; d++) {
        out[d] = weights[origin[d]];
    }
    if (reducer === "first") {
        return out;
    }
    for (let e = 0; e < remap.length; e++) {
        const d = remap[e];
        if (d === INVALID_INDEX || origin[d] === e) {
            continue;
        }
        const w = weights[e];
        switch (reducer) {
            case "last":
                out[d] = w;
                break;
            case "sum":
                out[d] += w;
                break;
            case "min":
                if (w < out[d]) {
                    out[d] = w;
                }
                break;
            case "max":
                if (w > out[d]) {
                    out[d] = w;
                }
                break;
            default: {
                const unknown: never = reducer;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown weight reducer ${String(unknown)}`, {
                    reducer: unknown,
                });
            }
        }
    }
    return out;
}

/**
 * The grouping of a per-edge keep predicate: kept edges survive on their own in order, others are
 * dropped.
 * @param edgeCount - the source edge count
 * @param keep - whether edge e is kept
 * @returns the grouping
 */
function groupingFromPredicate(edgeCount: number, keep: (e: number) => boolean): EdgeGrouping {
    const survivorOf = new Uint32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        survivorOf[e] = keep(e) ? e : INVALID_INDEX;
    }
    return groupingFromSurvivors(survivorOf);
}

// ============================================================ column reduction

/** The five flat numeric column shapes the group reducers accumulate. */
type NumericColumn = Extract<Column, { dtype: "f32" | "f64" | "i32" | "u32" | "u8" }>;

/**
 * A group reducer's numeric accumulation for one column into a new row space.
 * @param column - a numeric column
 * @param grouping - the grouping (remap non-null)
 * @param reducer - sum / min / max / mean
 * @returns the reduced column
 */
function reduceNumericColumn(
    column: NumericColumn,
    grouping: EdgeGrouping,
    reducer: "sum" | "min" | "max" | "mean",
): Column {
    const { count, remap } = grouping;
    const { components } = column.meta;
    const width = components;
    const acc = new Float64Array(count * width);
    const members = new Uint32Array(count);
    const seen = new Uint8Array(count);
    const source = column.data;
    const rows = remap === null ? column.length : remap.length;
    for (let r = 0; r < rows; r++) {
        const d = remap === null ? r : remap[r];
        if (d === INVALID_INDEX || !column.isSet(r)) {
            continue;
        }
        members[d]++;
        for (let k = 0; k < width; k++) {
            const value = source[r * width + k];
            const at = d * width + k;
            if (seen[d] === 0) {
                acc[at] = value;
            } else {
                switch (reducer) {
                    case "sum":
                    case "mean":
                        acc[at] += value;
                        break;
                    case "min":
                        if (value < acc[at]) {
                            acc[at] = value;
                        }
                        break;
                    case "max":
                        if (value > acc[at]) {
                            acc[at] = value;
                        }
                        break;
                    default: {
                        const unknown: never = reducer;
                        throw new GraphFormatError("E_UNSUPPORTED", `unknown reducer ${String(unknown)}`, {
                            reducer: unknown,
                        });
                    }
                }
            }
        }
        seen[d] = 1;
    }
    if (reducer === "mean") {
        for (let d = 0; d < count; d++) {
            if (members[d] > 1) {
                for (let k = 0; k < width; k++) {
                    acc[d * width + k] /= members[d];
                }
            }
        }
    }
    // "mean" is f64; a "sum" over an integer dtype widens to f64 too, so a total can never wrap
    // silently in a typed array that every write path would have refused (design section 5.1)
    const widen = reducer === "mean" || (reducer === "sum" && column.dtype !== "f32" && column.dtype !== "f64");
    const dtype = widen ? "f64" : column.dtype;
    let data: TypedArrayData;
    switch (dtype) {
        case "f64":
            data = acc;
            break;
        case "f32":
            data = new Float32Array(acc);
            break;
        case "i32":
            data = new Int32Array(acc);
            break;
        case "u32":
            data = new Uint32Array(acc);
            break;
        case "u8":
            data = new Uint8Array(new ArrayBuffer(Math.ceil((count * width) / 4) * 4), 0, count * width);
            data.set(acc);
            break;
        default: {
            const unknown: never = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `cannot reduce a ${String(unknown)} column`, {
                dtype: unknown,
            });
        }
    }
    let validity: U32 | null = null;
    let nullCount = 0;
    for (let d = 0; d < count; d++) {
        if (seen[d] === 0) {
            validity ??= makeBitmap(count, true);
            bitmapClear(validity, d);
            nullCount++;
        }
    }
    const meta = resolveColumnMeta(column.meta.name, column.meta.domain, {
        dtype,
        components,
        nullable: column.meta.nullable || nullCount > 0,
        role: column.meta.role ?? undefined,
        extra: column.meta.extra,
    });
    return createColumn({ ...emptyParts(meta, count, validity), data, nullCount });
}

/**
 * The u32 column of group sizes ("count").
 * @param column - the source column (name, domain, role and extra are kept)
 * @param grouping - the grouping
 * @returns the count column
 */
function countColumn(column: Column, grouping: EdgeGrouping): Column {
    const { count, remap } = grouping;
    const sizes = new Uint32Array(count);
    if (remap === null) {
        sizes.fill(1);
    } else {
        for (let r = 0; r < remap.length; r++) {
            if (remap[r] !== INVALID_INDEX) {
                sizes[remap[r]]++;
            }
        }
    }
    const meta = resolveColumnMeta(column.meta.name, column.meta.domain, {
        dtype: "u32",
        nullable: false,
        role: column.meta.role ?? undefined,
        extra: column.meta.extra,
    });
    return createColumn({ ...emptyParts(meta, count, null), data: sizes });
}

/**
 * Reduce one column over a grouping with a ColumnReducer (design section 7.3): `"first"` / `"last"`
 * gather the lowest / highest member's row (any dtype), `"count"` yields the group sizes, `"sum"` /
 * `"min"` / `"max"` / `"mean"` combine the set members of a numeric column (`"mean"` as f64, and so
 * is a `"sum"` over an i32 / u32 / u8 column, which would otherwise wrap; `"min"` / `"max"` keep
 * the dtype since the result is a member), `"drop"` omits the column. Index-valued (refersTo)
 * columns are always dropped: their values would refer to the source space.
 * @param column - the source column
 * @param grouping - the grouping
 * @param reducer - the reducer
 * @returns the reduced column, or null to drop it; E_COLUMN_TYPE for a numeric reducer on a non-numeric column
 */
function reduceColumn(column: Column, grouping: EdgeGrouping, reducer: ColumnReducer): Column | null {
    if (reducer === "drop" || column.meta.refersTo !== null) {
        return null;
    }
    const { count, remap, origin } = grouping;
    switch (reducer) {
        case "first":
            return origin === null ? column : gatherColumn(column, origin);
        case "last": {
            if (remap === null) {
                return column;
            }
            const last = new Uint32Array(count);
            for (let r = 0; r < remap.length; r++) {
                if (remap[r] !== INVALID_INDEX) {
                    last[remap[r]] = r;
                }
            }
            return gatherColumn(column, last);
        }
        case "count":
            return countColumn(column, grouping);
        case "sum":
        case "min":
        case "max":
        case "mean": {
            const { dtype } = column;
            switch (dtype) {
                case "f32":
                case "f64":
                case "i32":
                case "u32":
                case "u8":
                    return reduceNumericColumn(column, grouping, reducer);
                case "bool":
                case "dict":
                case "string":
                case "list":
                case "json":
                    throw new GraphFormatError(
                        "E_COLUMN_TYPE",
                        `reducer "${reducer}" needs a numeric column; "${column.meta.name}" is ${dtype}`,
                        { column: column.meta.name, dtype, reducer },
                    );
                default: {
                    const unknown: never = dtype;
                    throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${String(unknown)}`, { dtype: unknown });
                }
            }
        }
        default: {
            const unknown: never = reducer;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown column reducer ${String(unknown)}`, {
                reducer: unknown,
            });
        }
    }
}

/**
 * The edge table of a merging derived graph: named columns reduced per `reducers`, every other column
 * gathered from the survivor's row (the default of design section 5.11), refersTo values rewritten.
 * @param source - the source snapshot
 * @param grouping - the edge grouping
 * @param reducers - per-column reducers, or undefined for survivor rows throughout
 * @param nodeRemap - the node remap for refersTo node columns, or null
 * @returns the new edge table (the source table when the edge space is unchanged and no reducer applies)
 */
function reduceEdgeTable(
    source: GraphSnapshot,
    grouping: EdgeGrouping,
    reducers: Readonly<Record<string, ColumnReducer>> | undefined,
    nodeRemap: U32 | null,
): AttributeTable {
    const refs = { node: nodeRemap, edge: grouping.remap };
    if (reducers === undefined || Object.keys(reducers).length === 0) {
        if (grouping.origin === null) {
            return nodeRemap === null ? source.edges : remapTable(source.edges, null, source.edgeCount, refs);
        }
        return gatherTable(source.edges, grouping.origin, refs);
    }
    const columns: Column[] = [];
    for (const column of source.edges) {
        const reducer = reducers[column.meta.name];
        if (reducer === undefined) {
            const gathered = grouping.origin === null ? column : gatherColumn(column, grouping.origin);
            columns.push(gathered);
            continue;
        }
        const reduced = reduceColumn(column, grouping, reducer);
        if (reduced !== null) {
            columns.push(reduced);
        }
    }
    const table = new AttributeTableClass({ domain: "edge", rowCount: grouping.count, columns });
    return remapTable(table, null, grouping.count, refs);
}

// ============================================================ filterEdges, withoutSelfLoops

/**
 * The derived graph keeping the edges a predicate accepts, in the same node space (design section
 * 7.3: `filterEdges`, `withoutSelfLoops`). Always a new snapshot; the maps are null when every edge
 * is kept.
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param keep - whether edge e is kept
 * @returns the derived parts
 */
function deriveKeptEdges(source: GraphSnapshot, list: EdgeListView, keep: (e: number) => boolean): DerivedParts {
    const grouping = groupingFromPredicate(source.edgeCount, keep);
    const { src, dst } = gatherEndpoints(list, grouping, null);
    const weights = reduceWeights(list.weights, grouping, "first");
    const core = buildCore({
        directed: source.directed,
        nodeCount: source.nodeCount,
        edgeCount: grouping.count,
        src,
        dst,
        weights,
    });
    const edges = reduceEdgeTable(source, grouping, undefined, null);
    const parts = assembleParts(source, core, {
        directed: source.directed,
        nodeCount: source.nodeCount,
        edgeCount: grouping.count,
        ids: source.ids,
        nodes: source.nodes,
        edges,
        extensions: remapExtensions(source, null, grouping.remap),
    });
    return {
        parts,
        nodeOrigin: null,
        edgeOrigin: grouping.origin,
        nodeRemap: null,
        edgeRemap: grouping.remap,
        blockSizes: null,
        report: { droppedEdges: grouping.dropped, mergedEdges: 0 },
    };
}

/**
 * `filterEdges(keep)`: the logical edges whose mask bit is set (design section 7.3).
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param keep - packed bitmap over logical edges; E_MASK_LENGTH when short
 * @returns the derived parts
 */
export function deriveFilterEdges(source: GraphSnapshot, list: EdgeListView, keep: EdgeMask): DerivedParts {
    checkMaskLength(keep, source.edgeCount, "edges");
    return deriveKeptEdges(source, list, (e) => maskTest(keep, e));
}

/**
 * `withoutSelfLoops()`: every edge with source !== target (design section 7.3).
 * @param source - the source snapshot
 * @param list - the source edge list
 * @returns the derived parts
 */
export function deriveWithoutSelfLoops(source: GraphSnapshot, list: EdgeListView): DerivedParts {
    return deriveKeptEdges(source, list, (e) => list.src[e] !== list.dst[e]);
}

// ============================================================ inducedSubgraph, relabel

/**
 * The E_INDEX_RANGE error of a bad inducedSubgraph index list entry.
 * @param position - the position in the list
 * @param value - the offending value
 * @param nodeCount - the node count
 * @param reason - "out of range" or "repeated"
 * @returns the error
 */
function selectionError(position: number, value: number, nodeCount: number, reason: string): GraphFormatError {
    return new GraphFormatError(
        "E_INDEX_RANGE",
        `inducedSubgraph selection[${position}] = ${value} is ${reason} (nodeCount ${nodeCount})`,
        { index: position, found: value, nodeCount, reason },
    );
}

/**
 * Resolve an inducedSubgraph selection to the new -> source node list (design section 7.3): an
 * index list in the given order (E_INDEX_RANGE for an out-of-range or repeated index) or a packed
 * mask in ascending order (E_MASK_LENGTH when short).
 * @param source - the source snapshot
 * @param selection - the index list or mask
 * @returns the node origin list (a fresh array)
 */
function resolveSelection(source: GraphSnapshot, selection: U32 | { readonly mask: NodeMask }): U32 {
    const { nodeCount } = source;
    if (selection instanceof Uint32Array) {
        const seen = new Uint8Array(nodeCount);
        const origin = new Uint32Array(selection.length);
        for (let i = 0; i < selection.length; i++) {
            const u = selection[i];
            if (u >= nodeCount) {
                throw selectionError(i, u, nodeCount, "out of range");
            }
            if (seen[u] === 1) {
                throw selectionError(i, u, nodeCount, "repeated");
            }
            seen[u] = 1;
            origin[i] = u;
        }
        return origin;
    }
    checkMaskLength(selection.mask, nodeCount, "nodes");
    return maskToIndices(selection.mask, nodeCount);
}

/**
 * The inverse of a new -> source node list: source -> new or INVALID_INDEX.
 * @param origin - new -> source
 * @param nodeCount - the source node count
 * @returns the remap
 */
function invertOrigin(origin: U32, nodeCount: number): U32 {
    const remap = new Uint32Array(nodeCount).fill(INVALID_INDEX);
    for (let i = 0; i < origin.length; i++) {
        remap[origin[i]] = i;
    }
    return remap;
}

/**
 * Assemble a derived graph over a gathered node space (inducedSubgraph, relabel): the node table and
 * id map gathered through `nodeOrigin`, the edge table gathered through the edge grouping (or the
 * source table with refersTo node values rewritten when the edge space is unchanged).
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param nodeOrigin - new -> source node, or null for the identity
 * @param grouping - the edge grouping
 * @returns the derived parts
 */
function deriveOverNodeSpace(
    source: GraphSnapshot,
    list: EdgeListView,
    nodeOrigin: U32 | null,
    grouping: EdgeGrouping,
): DerivedParts {
    const nodeCount = nodeOrigin === null ? source.nodeCount : nodeOrigin.length;
    const nodeRemap = nodeOrigin === null ? null : invertOrigin(nodeOrigin, source.nodeCount);
    const { src, dst } = gatherEndpoints(list, grouping, nodeRemap);
    const weights = reduceWeights(list.weights, grouping, "first");
    const core = buildCore({ directed: source.directed, nodeCount, edgeCount: grouping.count, src, dst, weights });
    const refs = { node: nodeRemap, edge: grouping.remap };
    const nodes = nodeOrigin === null ? source.nodes : gatherTable(source.nodes, nodeOrigin, refs);
    const ids = nodeOrigin === null ? source.ids : gatherNodeIdMap(source.ids, nodeOrigin);
    const edges = reduceEdgeTable(source, grouping, undefined, nodeRemap);
    const parts = assembleParts(source, core, {
        directed: source.directed,
        nodeCount,
        edgeCount: grouping.count,
        ids,
        nodes,
        edges,
        extensions: remapExtensions(source, nodeRemap, grouping.remap),
    });
    return {
        parts,
        nodeOrigin,
        edgeOrigin: grouping.origin,
        nodeRemap,
        edgeRemap: grouping.remap,
        blockSizes: null,
        report: { droppedEdges: grouping.dropped, mergedEdges: grouping.merged },
    };
}

/**
 * `inducedSubgraph(selection)`: the subgraph on a node selection with every edge whose endpoints are
 * both kept (design section 7.3); a compact new node space unless the selection is the identity.
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param selection - an index list (new index order) or a packed mask (ascending)
 * @returns the derived parts
 */
export function deriveInducedSubgraph(
    source: GraphSnapshot,
    list: EdgeListView,
    selection: U32 | { readonly mask: NodeMask },
): DerivedParts {
    const origin = resolveSelection(source, selection);
    const identity = isIdentitySelection(origin, source.nodeCount);
    const nodeOrigin = identity ? null : origin;
    const nodeRemap = identity ? null : invertOrigin(origin, source.nodeCount);
    const grouping =
        nodeRemap === null
            ? groupingFromPredicate(source.edgeCount, () => true)
            : groupingFromPredicate(
                  source.edgeCount,
                  (e) => nodeRemap[list.src[e]] !== INVALID_INDEX && nodeRemap[list.dst[e]] !== INVALID_INDEX,
              );
    return deriveOverNodeSpace(source, list, nodeOrigin, grouping);
}

/**
 * `relabel(perm)`: the node space permuted with `perm[newIndex] = oldIndex`, edge order preserved
 * (design section 7.3); E_INVALID_PERMUTATION unless perm is a permutation of 0..n-1.
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param perm - the permutation
 * @returns the derived parts
 */
export function deriveRelabel(source: GraphSnapshot, list: EdgeListView, perm: U32): DerivedParts {
    const { nodeCount } = source;
    if (perm.length !== nodeCount) {
        throw new GraphFormatError("E_INVALID_PERMUTATION", `perm has ${perm.length} entries, expected ${nodeCount}`, {
            expected: nodeCount,
            found: perm.length,
        });
    }
    const seen = new Uint8Array(nodeCount);
    for (let i = 0; i < nodeCount; i++) {
        const u = perm[i];
        if (u >= nodeCount || seen[u] === 1) {
            throw new GraphFormatError(
                "E_INVALID_PERMUTATION",
                `perm[${i}] = ${u} is out of range or repeated (nodeCount ${nodeCount})`,
                { index: i, found: u, nodeCount },
            );
        }
        seen[u] = 1;
    }
    const nodeOrigin = isIdentity(perm) ? null : perm.slice();
    const grouping = groupingFromPredicate(source.edgeCount, () => true);
    return deriveOverNodeSpace(source, list, nodeOrigin, grouping);
}

// ============================================================ toUndirected, transpose

/**
 * `toUndirected(options)` on a directed snapshot (design section 7.3): every edge becomes undirected;
 * reciprocal pairs u -> v / v -> u (paired k-th to k-th among parallels, in edge order) collapse to
 * one edge keeping the lower index's row and the reduced weight (default "first"); with
 * `reciprocal: true` only paired edges survive. Self-loops are kept.
 * @param source - a directed snapshot
 * @param list - the source edge list
 * @param options - reciprocal filtering and the weight reducer
 * @returns the derived parts
 */
export function deriveToUndirected(
    source: GraphSnapshot,
    list: EdgeListView,
    options: ToUndirectedOptions = {},
): DerivedParts {
    const reciprocal = options.reciprocal === true;
    const reducer = assertOneOf("weights", options.weights, WEIGHT_REDUCERS) ?? "first";
    const { nodeCount, edgeCount, rowPtr, colIdx, arcToEdge } = source;
    const survivorOf = new Uint32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        survivorOf[e] = reciprocal && list.src[e] !== list.dst[e] ? INVALID_INDEX : e;
    }
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let a = rowPtr[u];
        while (a < end) {
            const v = colIdx[a];
            let g = a + 1;
            while (g < end && colIdx[g] === v) {
                g++;
            }
            if (v > u) {
                const [lo, hi] = arcRangeIn(rowPtr, colIdx, v, u);
                const pairs = Math.min(g - a, hi - lo);
                for (let i = 0; i < pairs; i++) {
                    const e1 = arcToEdge[a + i];
                    const e2 = arcToEdge[lo + i];
                    const low = Math.min(e1, e2);
                    const high = Math.max(e1, e2);
                    survivorOf[low] = low;
                    survivorOf[high] = low;
                }
            }
            a = g;
        }
    }
    const grouping = groupingFromSurvivors(survivorOf);
    const { src, dst } = gatherEndpoints(list, grouping, null);
    const weights = reduceWeights(list.weights, grouping, reducer);
    const core = buildCore({ directed: false, nodeCount, edgeCount: grouping.count, src, dst, weights });
    const edges = reduceEdgeTable(source, grouping, undefined, null);
    const parts = assembleParts(source, core, {
        directed: false,
        nodeCount,
        edgeCount: grouping.count,
        ids: source.ids,
        nodes: source.nodes,
        edges,
        extensions: remapExtensions(source, null, grouping.remap),
    });
    return {
        parts,
        nodeOrigin: null,
        edgeOrigin: grouping.origin,
        nodeRemap: null,
        edgeRemap: grouping.remap,
        blockSizes: null,
        report: { droppedEdges: grouping.dropped, mergedEdges: grouping.merged },
    };
}

/**
 * `transpose()` on a directed snapshot (design section 7.3): the orientation of every edge swapped by
 * adopting the reverse view's arrays as the core (zero copy, so `arena` is null); node and edge
 * spaces unchanged, tables shared.
 * @param source - a directed snapshot
 * @param reverse - its reverse view
 * @returns the derived parts
 */
export function deriveTranspose(source: GraphSnapshot, reverse: ReverseView): DerivedParts {
    const { nodeCount, edgeCount, arcCount, selfLoopCount } = source;
    const revArcToEdge = reverse.arcToEdge;
    const identity = isIdentity(revArcToEdge);
    let arcToEdge: U32 | null = null;
    let edgeToArc: U32 | null = null;
    if (!identity) {
        arcToEdge = revArcToEdge;
        edgeToArc = new Uint32Array(edgeCount);
        for (let k = 0; k < arcCount; k++) {
            edgeToArc[revArcToEdge[k]] = k;
        }
    }
    const flags = computeFlags({
        directed: true,
        nodeCount,
        rowPtr: reverse.rowPtr,
        colIdx: reverse.colIdx,
        weights: reverse.weights,
        arcToEdge,
        selfLoopCount,
    });
    // the adopted arrays stay held by the source's cached reverse view (design section 9.1)
    noteShared(reverse.rowPtr.buffer);
    noteShared(reverse.colIdx.buffer);
    if (reverse.weights !== null) {
        noteShared(reverse.weights.buffer);
    }
    if (arcToEdge !== null) {
        noteShared(arcToEdge.buffer);
    }
    const parts: SnapshotParts = {
        label: source.label,
        serial: null,
        directed: true,
        nodeCount,
        edgeCount,
        arcCount,
        selfLoopCount,
        rowPtr: reverse.rowPtr,
        colIdx: reverse.colIdx,
        weights: reverse.weights,
        arcToEdge,
        edgeToArc,
        flags,
        ids: source.ids,
        nodes: source.nodes,
        edges: source.edges,
        graph: source.graph,
        extensions: source.extensions,
        meta: source.meta,
        arena: null,
        checksum: false,
    };
    return {
        parts,
        nodeOrigin: null,
        edgeOrigin: null,
        nodeRemap: null,
        edgeRemap: null,
        blockSizes: null,
        report: NO_REPORT,
    };
}

// ============================================================ simplified

/**
 * `simplified(options)` (design section 7.3): one edge per (u, v) group (parallels are adjacent by
 * I4; undirected groups are unordered pairs), survivor = lowest index, weights per `weights`
 * (default "first"), self-loops kept or dropped, other edge columns per `edgeReducers` (default: the
 * survivor's row). `flags.multigraph` is false afterwards.
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param options - reducers and self-loop policy
 * @returns the derived parts
 */
export function deriveSimplified(
    source: GraphSnapshot,
    list: EdgeListView,
    options: SimplifyOptions = {},
): DerivedParts {
    const reducer = assertOneOf("weights", options.weights, WEIGHT_REDUCERS) ?? "first";
    const dropLoops = assertOneOf("selfLoops", options.selfLoops, ["keep", "drop"] as const) === "drop";
    checkReducers("edgeReducers", options.edgeReducers);
    const { nodeCount, edgeCount, rowPtr, colIdx, arcToEdge } = source;
    const survivorOf = new Uint32Array(edgeCount).fill(INVALID_INDEX);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let a = rowPtr[u];
        while (a < end) {
            const v = colIdx[a];
            let g = a + 1;
            while (g < end && colIdx[g] === v) {
                g++;
            }
            if (source.directed || v >= u) {
                if (!(v === u && dropLoops)) {
                    const survivor = arcToEdge[a];
                    for (let i = a; i < g; i++) {
                        survivorOf[arcToEdge[i]] = survivor;
                    }
                }
            }
            a = g;
        }
    }
    const grouping = groupingFromSurvivors(survivorOf);
    const { src, dst } = gatherEndpoints(list, grouping, null);
    const weights = reduceWeights(list.weights, grouping, reducer);
    const core = buildCore({ directed: source.directed, nodeCount, edgeCount: grouping.count, src, dst, weights });
    const edges = reduceEdgeTable(source, grouping, options.edgeReducers, null);
    const parts = assembleParts(source, core, {
        directed: source.directed,
        nodeCount,
        edgeCount: grouping.count,
        ids: source.ids,
        nodes: source.nodes,
        edges,
        extensions: remapExtensions(source, null, grouping.remap),
    });
    return {
        parts,
        nodeOrigin: null,
        edgeOrigin: grouping.origin,
        nodeRemap: null,
        edgeRemap: grouping.remap,
        blockSizes: null,
        report: { droppedEdges: grouping.dropped, mergedEdges: grouping.merged },
    };
}

// ============================================================ contract

/**
 * Renumber an arbitrary u32 labelling to dense 0..k-1 in first-seen order (design section 7.5; what
 * `contract()` does with non-dense labels). INVALID_INDEX is not a legal label (E_PARTITION).
 * @param labels - the labels
 * @param out - an optional destination of the same length (may be `labels` itself)
 * @returns the dense labels and the block count
 */
export function renumberPartition(labels: U32, out?: U32): { readonly labels: U32; readonly count: number } {
    const n = labels.length;
    const result = out ?? new Uint32Array(n);
    if (result.length !== n) {
        throw new GraphFormatError("E_COLUMN_LENGTH", `out has ${result.length} entries, expected ${n}`, {
            expected: n,
            found: result.length,
        });
    }
    let maxLabel = 0;
    for (let i = 0; i < n; i++) {
        const label = labels[i];
        if (label === INVALID_INDEX) {
            throw new GraphFormatError("E_PARTITION", `labels[${i}] is INVALID_INDEX`, { index: i });
        }
        if (label > maxLabel) {
            maxLabel = label;
        }
    }
    let count = 0;
    if (maxLabel < 4 * n + 1024) {
        const lookup = new Uint32Array(maxLabel + 1).fill(INVALID_INDEX);
        for (let i = 0; i < n; i++) {
            const label = labels[i];
            let dense = lookup[label];
            if (dense === INVALID_INDEX) {
                dense = count++;
                lookup[label] = dense;
            }
            result[i] = dense;
        }
    } else {
        const lookup = new Map<number, number>();
        for (let i = 0; i < n; i++) {
            const label = labels[i];
            let dense = lookup.get(label);
            if (dense === undefined) {
                dense = count++;
                lookup.set(label, dense);
            }
            result[i] = dense;
        }
    }
    return { labels: result, count };
}

/**
 * The dense block labels of a contract partition (design section 7.3): labels already forming the
 * set 0..k-1 are kept, any other labelling is renumbered in first-seen order. E_PARTITION for a wrong
 * length or an INVALID_INDEX label.
 * @param partition - the caller's labels
 * @param nodeCount - the node count
 * @returns the dense labels (a fresh array) and the block count
 */
function denseBlocks(partition: U32, nodeCount: number): { readonly labels: U32; readonly count: number } {
    if (partition.length !== nodeCount) {
        throw new GraphFormatError(
            "E_PARTITION",
            `partition has ${partition.length} entries, expected nodeCount ${nodeCount}`,
            { expected: nodeCount, found: partition.length },
        );
    }
    let maxLabel = -1;
    for (let i = 0; i < nodeCount; i++) {
        const label = partition[i];
        if (label === INVALID_INDEX) {
            throw new GraphFormatError("E_PARTITION", `partition[${i}] is INVALID_INDEX`, { index: i });
        }
        if (label > maxLabel) {
            maxLabel = label;
        }
    }
    if (maxLabel < nodeCount) {
        const seen = new Uint8Array(maxLabel + 1);
        let distinct = 0;
        for (let i = 0; i < nodeCount; i++) {
            if (seen[partition[i]] === 0) {
                seen[partition[i]] = 1;
                distinct++;
            }
        }
        if (distinct === maxLabel + 1) {
            return { labels: partition.slice(), count: maxLabel + 1 };
        }
    }
    return renumberPartition(partition);
}

/**
 * Reduce the node columns of a contracted graph per `nodeReducers` over the blocks; columns not named
 * are dropped (design section 7.3).
 * @param source - the source snapshot
 * @param blocks - the block labels (source node -> block)
 * @param blockCount - the number of blocks
 * @param blockOrigin - block -> lowest source node
 * @param reducers - the named reducers
 * @returns the new node table
 */
function contractNodeTable(
    source: GraphSnapshot,
    blocks: U32,
    blockCount: number,
    blockOrigin: U32,
    reducers: Readonly<Record<string, ColumnReducer>> | undefined,
): AttributeTable {
    const columns: Column[] = [];
    if (reducers !== undefined) {
        const grouping: EdgeGrouping = { count: blockCount, remap: blocks, origin: blockOrigin, dropped: 0, merged: 0 };
        for (const column of source.nodes) {
            const reducer = reducers[column.meta.name];
            if (reducer === undefined) {
                continue;
            }
            const reduced = reduceColumn(column, grouping, reducer);
            if (reduced !== null) {
                columns.push(reduced);
            }
        }
    }
    return new AttributeTableClass({ domain: "node", rowCount: blockCount, columns });
}

/**
 * Group the kept contracted edges by block pair with two stable counting-sort passes (by second key,
 * then by first key), so each run of equal pairs is one group and the first member of a run is the
 * lowest source edge index.
 * @param first - the first key of every source edge (INVALID_INDEX = dropped)
 * @param second - the second key of every source edge
 * @param blockCount - the key range
 * @returns source edge -> survivor (INVALID_INDEX when dropped)
 */
function groupByBlockPair(first: U32, second: U32, blockCount: number): U32 {
    const edgeCount = first.length;
    const kept: number[] = [];
    for (let e = 0; e < edgeCount; e++) {
        if (first[e] !== INVALID_INDEX) {
            kept.push(e);
        }
    }
    const start = new Uint32Array(blockCount + 1);
    for (const e of kept) {
        start[second[e] + 1]++;
    }
    for (let b = 0; b < blockCount; b++) {
        start[b + 1] += start[b];
    }
    const bySecond = new Uint32Array(kept.length);
    for (const e of kept) {
        bySecond[start[second[e]]++] = e;
    }
    start.fill(0);
    for (const e of kept) {
        start[first[e] + 1]++;
    }
    for (let b = 0; b < blockCount; b++) {
        start[b + 1] += start[b];
    }
    const sorted = new Uint32Array(kept.length);
    for (let k = 0; k < bySecond.length; k++) {
        const e = bySecond[k];
        sorted[start[first[e]]++] = e;
    }
    const survivorOf = new Uint32Array(edgeCount).fill(INVALID_INDEX);
    let runStart = 0;
    while (runStart < sorted.length) {
        const leader = sorted[runStart];
        let runEnd = runStart + 1;
        while (
            runEnd < sorted.length &&
            first[sorted[runEnd]] === first[leader] &&
            second[sorted[runEnd]] === second[leader]
        ) {
            runEnd++;
        }
        for (let k = runStart; k < runEnd; k++) {
            survivorOf[sorted[k]] = leader;
        }
        runStart = runEnd;
    }
    return survivorOf;
}

/**
 * `contract(partition, options)` (design section 7.3): the nodes of each block become one node (block
 * indices as labels, ids 0..k-1), every source edge contributes once: an inter-block edge becomes an
 * edge block(u) -> block(v), an intra-block edge one self-loop with weight w, kept or dropped per
 * `selfLoops`; parallels merge per `parallel` with the `weights` reducer (default "sum"; on an
 * unweighted source "sum" materialises multiplicities). Node and edge columns are kept only through
 * `nodeReducers` / `edgeReducers`.
 * @param source - the source snapshot
 * @param list - the source edge list
 * @param partition - one label per node; E_PARTITION for a wrong length or an INVALID_INDEX label
 * @param options - the contraction options
 * @returns the derived parts with blockSizes
 */
export function deriveContract(
    source: GraphSnapshot,
    list: EdgeListView,
    partition: U32,
    options: ContractOptions = {},
): DerivedParts {
    const reducer = assertOneOf("weights", options.weights, WEIGHT_REDUCERS) ?? "sum";
    const dropLoops = assertOneOf("selfLoops", options.selfLoops, ["keep", "drop"] as const) === "drop";
    const merge = (assertOneOf("parallel", options.parallel, ["merge", "keep"] as const) ?? "merge") === "merge";
    checkReducers("nodeReducers", options.nodeReducers);
    checkReducers("edgeReducers", options.edgeReducers);
    const { nodeCount, edgeCount } = source;
    const { labels: blocks, count: blockCount } = denseBlocks(partition, nodeCount);
    const blockSizes = new Uint32Array(blockCount);
    const blockOrigin = new Uint32Array(blockCount).fill(INVALID_INDEX);
    for (let u = 0; u < nodeCount; u++) {
        const b = blocks[u];
        blockSizes[b]++;
        if (blockOrigin[b] === INVALID_INDEX) {
            blockOrigin[b] = u;
        }
    }
    const first = new Uint32Array(edgeCount);
    const second = new Uint32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        const bu = blocks[list.src[e]];
        const bv = blocks[list.dst[e]];
        if (bu === bv && dropLoops) {
            first[e] = INVALID_INDEX;
            second[e] = INVALID_INDEX;
        } else if (source.directed || bu <= bv) {
            first[e] = bu;
            second[e] = bv;
        } else {
            first[e] = bv;
            second[e] = bu;
        }
    }
    let survivorOf: U32;
    if (merge) {
        survivorOf = groupByBlockPair(first, second, blockCount);
    } else {
        survivorOf = new Uint32Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            survivorOf[e] = first[e] === INVALID_INDEX ? INVALID_INDEX : e;
        }
    }
    const grouping = groupingFromSurvivors(survivorOf);
    const { src, dst } = gatherEndpoints(list, grouping, blocks);
    const weights = reduceWeights(list.weights, grouping, reducer);
    const core = buildCore({
        directed: source.directed,
        nodeCount: blockCount,
        edgeCount: grouping.count,
        src,
        dst,
        weights,
    });
    const nodes = contractNodeTable(source, blocks, blockCount, blockOrigin, options.nodeReducers);
    const edgeColumns: Column[] = [];
    if (options.edgeReducers !== undefined) {
        for (const column of source.edges) {
            const columnReducer = options.edgeReducers[column.meta.name];
            if (columnReducer === undefined) {
                continue;
            }
            const reduced = reduceColumn(column, grouping, columnReducer);
            if (reduced !== null) {
                edgeColumns.push(reduced);
            }
        }
    }
    const edges = new AttributeTableClass({ domain: "edge", rowCount: grouping.count, columns: edgeColumns });
    const parts = assembleParts(source, core, {
        directed: source.directed,
        nodeCount: blockCount,
        edgeCount: grouping.count,
        ids: identityNodeIdMap(blockCount),
        nodes,
        edges,
        extensions: remapExtensions(source, blocks, grouping.remap),
    });
    return {
        parts,
        nodeOrigin: blockOrigin,
        edgeOrigin: grouping.origin,
        nodeRemap: blocks,
        edgeRemap: grouping.remap,
        blockSizes,
        report: { droppedEdges: grouping.dropped, mergedEdges: grouping.merged },
    };
}

// ============================================================ withColumns

/**
 * `withColumns(nodes, edges)` (design section 7.3): the parts of a snapshot sharing the core, the id
 * map and the SERIAL of the source with a CLONED column set plus the given columns (the graph table is
 * cloned too; extension tables are shared).
 * @param source - the source snapshot
 * @param nodes - node columns to add, keyed by name
 * @param edges - edge columns to add, keyed by name
 * @param arcToEdge - the source's permutation arrays as the constructor holds them (null when identity)
 * @param edgeToArc - see arcToEdge
 * @returns the parts
 */
export function withColumnsParts(
    source: GraphSnapshot,
    nodes: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined,
    edges: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined,
    arcToEdge: U32 | null,
    edgeToArc: U32 | null,
): SnapshotParts {
    return {
        label: source.label,
        serial: source.serial,
        directed: source.directed,
        nodeCount: source.nodeCount,
        edgeCount: source.edgeCount,
        arcCount: source.arcCount,
        selfLoopCount: source.selfLoopCount,
        rowPtr: source.rowPtr,
        colIdx: source.colIdx,
        weights: source.weights,
        arcToEdge,
        edgeToArc,
        flags: source.flags,
        ids: source.ids,
        nodes: tableWithColumns(source.nodes, nodes ?? {}),
        edges: tableWithColumns(source.edges, edges ?? {}),
        graph: source.graph.clone(),
        extensions: source.extensions,
        meta: source.meta,
        arena: source.arena,
        checksum: false,
    };
}
