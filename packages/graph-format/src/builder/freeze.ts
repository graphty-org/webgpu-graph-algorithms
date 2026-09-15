/**
 * The freeze pipeline of design section 6.3, exactly: the pre-checks that throw without touching the
 * builder (self-loop "error", arc-count bound), compaction of tombstones and dropped loops (steps 1-2,
 * into a fresh staging the builder commits only on success), the two counting-sort passes into the
 * arena (steps 3-6, counting-sort.ts), the duplicate-policy walk with its one repeat after a merge
 * (step 7), the weight flags and the f64 / explicit-weight shadow column (step 8), the id map (step
 * 9), the attribute tables (step 10), the arena and prepared views (step 11) and the snapshot itself
 * with its FreezeReport (step 12).
 *
 * Nothing here mutates the builder: `runFreeze` returns the staging the builder must adopt (the
 * source staging when nothing was renumbered, a compacted one otherwise) so that a throw anywhere in
 * the pipeline leaves the builder as it was (design section 11.1).
 */

import { bitmapClear, bitmapGet, bitmapSet, bitmapWordCount } from "../columns/bitmap.js";
import { columnFromValues, createColumn, emptyParts, resolveColumnMeta } from "../columns/column.js";
import { AttributeTable, verifyUniqueColumns } from "../columns/table.js";
import { INVALID_INDEX, MAX_COUNT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { identityNodeIdMap, nodeIdMapFromIds } from "../ids/node-id-map.js";
import { createSnapshot, type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import { isIdentity } from "../snapshot/validate.js";
import {
    type Column,
    type ColumnDeclPatch,
    type ColumnDomain,
    type Dtype,
    type DuplicatePolicy,
    type F32,
    type F64,
    type FreezeOptions,
    type FreezeReport,
    type GraphMeta,
    type NodeIdMap,
    type ResolvedBuilderOptions,
    type U32,
} from "../types/index.js";
import { type SnapshotParts } from "../types/internal.js";
import { copyCoreIntoArena } from "./arena.js";
import {
    compactStaging,
    type IndexRemap,
    remapKeepingBits,
    remapSurvivors,
    type Staging,
    type StagingColumn,
} from "./compact.js";
import { type SortInput, sortIntoCore, type SortResult } from "./counting-sort.js";
import { assertDuplicatePolicy } from "./options.js";

// ============================================================ context and outcome

/** One graph-level attribute the builder holds until freeze (design section 5.9). */
export interface GraphValue {
    /** The declaration patch given to setGraphValue, or undefined. */
    readonly decl: ColumnDeclPatch | undefined;
    /** The value; undefined = unset. */
    readonly value: unknown;
}

/** A widening recorded by the builder since the previous freeze (design section 5.1). */
export interface Widening {
    /** The column name. */
    readonly column: string;
    /** The column's table. */
    readonly domain: ColumnDomain;
    /** The dtype before. */
    readonly from: Dtype;
    /** The dtype after. */
    readonly to: Dtype;
}

/** Everything the pipeline reads from the builder. */
export interface FreezeContext {
    /** The staging to freeze; never modified. */
    readonly staging: Staging;
    /** The builder's current direction. */
    readonly directed: boolean;
    /** The resolved builder options (policies, precision). */
    readonly options: ResolvedBuilderOptions;
    /** The graph-level attributes in declaration order. */
    readonly graphValues: ReadonlyMap<string, GraphValue>;
    /** The graph metadata. */
    readonly meta: GraphMeta;
    /** The widenings since the previous freeze. */
    readonly widened: readonly Widening[];
}

/** What a freeze produces: the snapshot, its report, and the staging the builder must adopt. */
interface FreezeOutcome {
    /** The frozen snapshot. */
    readonly snapshot: GraphSnapshot;
    /** The report. */
    readonly report: FreezeReport;
    /** The staging the builder holds afterwards: the source staging, or the compacted one. */
    readonly staging: Staging;
}

/** The name of the shadow weight column the freeze writes (design sections 3.7 and 5.6). */
export const WEIGHT_COLUMN_NAME = "graphty.weight";

/**
 * A millisecond clock for the profile timings.
 * @returns the current time in milliseconds
 */
function now(): number {
    return performance.now();
}

/** A per-edge scratch: which survivor an edge folds into, and the reduced weights. */
interface DuplicateWalk {
    /** old edge -> its survivor (itself for a survivor). */
    readonly survivorOf: U32;
    /** Per old edge: the reduced weight to store on a survivor, meaningful where `stored` is set; null when none. */
    readonly reduced: Float64Array | null;
    /** Per old edge: bit set on a survivor whose group produced a reduced weight to store; null when none. */
    readonly stored: U32 | null;
    /** Per old edge: bit set on a survivor whose weight is explicit after the merge although it was not before; null when none. */
    readonly explicit: U32 | null;
    /** How many edges folded into a survivor. */
    readonly merged: number;
}

// ============================================================ steps

/**
 * The E_SELF_LOOP pre-check of design section 6.3 step 2 (policy "error"): the first live loop is
 * named. Runs before anything is renumbered so the builder is untouched.
 * @param staging - the source staging
 */
function checkSelfLoops(staging: Staging): void {
    if (staging.selfLoopCount === 0) {
        return;
    }
    const bound = staging.edgeBound;
    for (let e = 0; e < bound; e++) {
        if (staging.edgeAlive.get(e) && staging.src.get(e) === staging.dst.get(e)) {
            const node = staging.src.get(e);
            throw new GraphFormatError("E_SELF_LOOP", `edge ${e} is a self-loop at node ${node}`, { edge: e, node });
        }
    }
}

/**
 * The live edges that are not self-loops, as a bitmap over the edge index space (the edges a
 * `selfLoops: "drop"` freeze keeps, design section 6.3 step 2).
 * @param staging - the source staging
 * @returns a fresh bitmap: bit e set = edge e is live and src !== dst
 */
function liveNonLoopBits(staging: Staging): U32 {
    const keep = staging.edgeAlive.view().slice();
    const src = staging.src.view();
    const dst = staging.dst.view();
    const bound = staging.edgeBound;
    for (let e = 0; e < bound; e++) {
        if (src[e] === dst[e] && bitmapGet(keep, e)) {
            bitmapClear(keep, e);
        }
    }
    return keep;
}

/**
 * The sort input over a staging's live (compacted) edges.
 * @param staging - a staging without tombstones
 * @param directed - the direction
 * @returns the per-edge views
 */
function sortInputOf(staging: Staging, directed: boolean): SortInput {
    const edgeCount = staging.edgeBound;
    return {
        directed,
        nodeCount: staging.nodeBound,
        edgeCount,
        src: staging.src.view(),
        dst: staging.dst.view(),
        weights: staging.weight === null ? null : (staging.weight.view() as F32 | F64),
    };
}

/**
 * The duplicate-policy walk of design section 6.3 step 7 over sorted rows: adjacent arcs with equal
 * colIdx form a group; "error" throws E_DUPLICATE_EDGE; the merge policies choose a survivor (the
 * lowest edge index, or the highest for "last"), reduce the weights over the group from the staging
 * values (f64 precision), and fold every other member into the survivor. A reducer that produces NaN
 * (a "sum" over +Infinity and -Infinity) is the E_INVALID_WEIGHT re-check of step 8 and throws before
 * anything is renumbered. A reduced value makes the survivor's weight explicit (the merge stored it,
 * design section 6.5), as does an explicit weight anywhere in the group. For an undirected graph the
 * `(u, v)` and `(v, u)` groups are one group, walked once from the row of the lower endpoint.
 * @param sorted - the sorted core
 * @param staging - the (compacted) staging the core was built from
 * @param directed - the direction
 * @param policy - a policy other than "keep"
 * @param countMultiplicity - whether "sum" on an unweighted staging materialises multiplicities (false on a `weighted: false` builder, which never gains a weight array)
 * @param builderIndex - compacted edge index -> the builder's own index, for error details
 * @returns the walk, or null when no group had more than one member
 */
function walkDuplicates(
    sorted: SortResult,
    staging: Staging,
    directed: boolean,
    policy: Exclude<DuplicatePolicy, "keep">,
    countMultiplicity: boolean,
    builderIndex: (e: number) => number,
): DuplicateWalk | null {
    const { rowPtr, colIdx, arcToEdge } = sorted.core;
    const nodeCount = staging.nodeBound;
    const edgeCount = staging.edgeBound;
    const weights = staging.weight === null ? null : staging.weight.view();
    const survivorOf = new Uint32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        survivorOf[e] = e;
    }
    let reduced: Float64Array | null = null;
    let stored: U32 | null = null;
    let explicit: U32 | null = null;
    let merged = 0;
    const group: number[] = [];
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let a = rowPtr[u];
        while (a < end) {
            const v = colIdx[a];
            let b = a + 1;
            while (b < end && colIdx[b] === v) {
                b++;
            }
            if (b - a > 1 && (directed || v >= u)) {
                group.length = 0;
                for (let k = a; k < b; k++) {
                    group.push(arcToEdge === null ? k : arcToEdge[k]);
                }
                if (policy === "error") {
                    const edges = [builderIndex(group[0]), builderIndex(group[1])];
                    throw new GraphFormatError(
                        "E_DUPLICATE_EDGE",
                        `edges ${edges[0]} and ${edges[1]} both connect ${u} -> ${v}`,
                        { source: u, target: v, edges },
                    );
                }
                const survivor = policy === "last" ? group[group.length - 1] : group[0];
                let value: number | null = null;
                if (policy === "sum" || policy === "min" || policy === "max") {
                    value = reduceGroup(weights, group, policy, countMultiplicity);
                    if (value !== null && Number.isNaN(value)) {
                        const edges = group.map(builderIndex);
                        throw new GraphFormatError(
                            "E_INVALID_WEIGHT",
                            `duplicateEdges "${policy}" over edges ${edges.join(", ")} (${u} -> ${v}) yields NaN`,
                            { source: u, target: v, edges, policy, reason: "reducer" },
                        );
                    }
                }
                if (value !== null) {
                    reduced ??= new Float64Array(edgeCount);
                    stored ??= new Uint32Array(bitmapWordCount(edgeCount));
                    reduced[survivor] = value;
                    bitmapSet(stored, survivor);
                }
                let anyExplicit = value !== null;
                for (let i = 0; i < group.length && !anyExplicit; i++) {
                    anyExplicit = staging.weightExplicit(group[i]);
                }
                if (anyExplicit && !staging.weightExplicit(survivor)) {
                    explicit ??= new Uint32Array(bitmapWordCount(edgeCount));
                    bitmapSet(explicit, survivor);
                }
                for (const e of group) {
                    if (e !== survivor) {
                        survivorOf[e] = survivor;
                        merged++;
                    }
                }
            }
            a = b;
        }
    }
    return merged === 0 ? null : { survivorOf, reduced, stored, explicit, merged };
}

/**
 * Reduce the weights of a parallel group (design section 6.5): sum / min / max over the staging
 * values; on an unweighted staging every weight is 1, so "sum" yields the group size (the
 * multiplicity, when the builder may gain a weight array) and min / max leave 1 (nothing to store).
 * @param weights - the staging weights, or null
 * @param group - the edge indices of the group
 * @param policy - the reducer
 * @param countMultiplicity - whether "sum" on an unweighted staging yields the group size
 * @returns the reduced weight (NaN when the operands cancel), or null when the survivor's own value already holds it
 */
function reduceGroup(
    weights: ArrayLike<number> | null,
    group: readonly number[],
    policy: "sum" | "min" | "max",
    countMultiplicity: boolean,
): number | null {
    if (weights === null) {
        return policy === "sum" && countMultiplicity ? group.length : null;
    }
    let value = weights[group[0]];
    for (let i = 1; i < group.length; i++) {
        const w = weights[group[i]];
        switch (policy) {
            case "sum":
                value += w;
                break;
            case "min":
                value = Math.min(value, w);
                break;
            case "max":
                value = Math.max(value, w);
                break;
            default: {
                const name: string = policy;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown reducer ${name}`, { policy: name });
            }
        }
    }
    return value;
}

/**
 * The role-"weight" edge column of design section 3.7, when one is needed: the f64 shadow when the
 * staging precision is f64 and some value is not f32-exact, and / or the explicit-weight record (the
 * `weightSet` bitmap as validity) when some edge omitted its weight while another supplied it, or when
 * a `weighted: true` builder's edges all omitted it.
 * @param staging - the final staging
 * @param edgeCount - the logical edge count
 * @param weightDtype - the staging precision
 * @returns the column, or null when the arc array already says everything
 */
function weightShadowColumn(staging: Staging, edgeCount: number, weightDtype: "f32" | "f64"): Column | null {
    const { weight } = staging;
    if (weight === null || edgeCount === 0) {
        return null;
    }
    const values = weight.view();
    let inexact = false;
    if (weightDtype === "f64") {
        for (let e = 0; e < edgeCount; e++) {
            const w = values[e];
            if (Math.fround(w) !== w) {
                inexact = true;
                break;
            }
        }
    }
    const needValidity = staging.weightSet !== null || staging.weightMode === "omitted";
    if (!inexact && !needValidity) {
        return null;
    }
    const dtype = inexact ? "f64" : "f32";
    const meta = resolveColumnMeta(WEIGHT_COLUMN_NAME, "edge", { dtype, role: "weight", nullable: needValidity });
    const data = dtype === "f64" ? new Float64Array(edgeCount) : new Float32Array(edgeCount);
    data.set(values.subarray(0, edgeCount));
    let validity: U32 | null = null;
    if (needValidity) {
        validity = new Uint32Array(bitmapWordCount(edgeCount));
        for (let e = 0; e < edgeCount; e++) {
            if (staging.weightExplicit(e)) {
                bitmapSet(validity, e);
            }
        }
    }
    return createColumn({ ...emptyParts(meta, edgeCount, validity), data });
}

/**
 * The id map of design section 6.3 step 9: identity with no storage while every node is anonymous,
 * otherwise the kind detected from the ids with the builder's `Map` and `ids` array shared by reference
 * (decision C17).
 * @param staging - the final staging
 * @param nodeCount - the node count
 * @returns the id map
 */
function idMapOf(staging: Staging, nodeCount: number): NodeIdMap {
    if (staging.ids === null || staging.idToIndex === null) {
        return identityNodeIdMap(nodeCount);
    }
    return nodeIdMapFromIds(staging.ids, nodeCount, { map: staging.idToIndex });
}

/**
 * Freeze a list of staging columns into a table (design section 6.3 step 10) and enforce its unique
 * columns.
 * @param domain - the table domain
 * @param rowCount - the row count
 * @param columns - the staging columns
 * @param extra - columns to append (the weight shadow)
 * @returns the table
 */
function tableOf(
    domain: ColumnDomain,
    rowCount: number,
    columns: readonly StagingColumn[],
    extra: readonly Column[] = [],
): AttributeTable {
    const frozen: Column[] = columns.map((column) => column.toColumn(rowCount));
    const table = new AttributeTable({ domain, rowCount, columns: [...frozen, ...extra] });
    verifyUniqueColumns(table);
    return table;
}

/**
 * The graph table (one row) from the builder's graph values.
 * @param values - the graph values in declaration order
 * @returns the table
 */
function graphTableOf(values: ReadonlyMap<string, GraphValue>): AttributeTable {
    const columns: Column[] = [];
    for (const [name, entry] of values) {
        columns.push(columnFromValues("graph", 1, name, [entry.value], entry.decl ?? {}));
    }
    return new AttributeTable({ domain: "graph", rowCount: 1, columns });
}

/**
 * Compose the report's edge remap out of the compaction remap (old -> compacted) and the merge walk
 * (compacted -> survivor -> final): every old edge maps to its final index, or to INVALID_INDEX when
 * tombstoned or dropped; a merged edge maps to its survivor (design section 6.5).
 * @param oldBound - the old edge index space
 * @param first - the compaction remap, or null when none ran
 * @param walk - the merge walk, or null
 * @param second - the post-merge remap, or null
 * @returns the composed remap, or null when it is the identity
 */
function composeEdgeRemap(
    oldBound: number,
    first: IndexRemap | null,
    walk: DuplicateWalk | null,
    second: IndexRemap | null,
): U32 | null {
    if (first === null && walk === null) {
        return null;
    }
    const out = new Uint32Array(oldBound);
    for (let e = 0; e < oldBound; e++) {
        let index = first === null ? e : first.remap[e];
        if (index !== INVALID_INDEX && walk !== null && second !== null) {
            index = second.remap[walk.survivorOf[index]];
        }
        out[e] = index;
    }
    return isIdentity(out) ? null : out;
}

/**
 * The identity remap of a space of `count` indices (a compaction that only renumbers edges).
 * @param count - the space size
 * @returns the identity
 */
function identityRemap(count: number): IndexRemap {
    const remap = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
        remap[i] = i;
    }
    return { remap, origin: remap, count };
}

/**
 * The E_UNSUPPORTED error for a freeze option outside its documented set (design section 12.2
 * FreezeOptions), thrown before any staging work so the builder is untouched.
 * @param field - the option name
 * @param found - the value
 * @returns the error
 */
function freezeOptionError(field: string, found: unknown): GraphFormatError {
    return new GraphFormatError("E_UNSUPPORTED", `freeze option ${field} has an unsupported value ${String(found)}`, {
        field,
        found,
        reason: "unsupported option",
    });
}

/**
 * Validate every FreezeOptions field (design section 12.2): the per-freeze duplicateEdges override
 * against the DuplicatePolicy set (a merge policy rewrites the builder, so an unknown value must
 * never act as one), the booleans, the label and the prepare list's shape. View names are checked
 * by `prepare()` itself.
 * @param options - the freeze options
 */
function checkFreezeOptions(options: FreezeOptions): void {
    if (options.duplicateEdges !== undefined) {
        assertDuplicatePolicy("duplicateEdges", options.duplicateEdges, freezeOptionError);
    }
    for (const field of ["arena", "release", "profile", "checksum"] as const) {
        const value = options[field];
        if (value !== undefined && typeof value !== "boolean") {
            throw freezeOptionError(field, value);
        }
    }
    if (options.label !== undefined && typeof options.label !== "string") {
        throw freezeOptionError("label", options.label);
    }
    if (options.prepare !== undefined && !Array.isArray(options.prepare)) {
        throw freezeOptionError("prepare", options.prepare);
    }
}

// ============================================================ the pipeline

/**
 * Run the freeze pipeline of design section 6.3 over a builder's state and produce the snapshot, the
 * FreezeReport and the staging the builder should hold afterwards. The context is never mutated: a
 * throw (E_SELF_LOOP, E_DUPLICATE_EDGE, E_DUPLICATE_EDGE_ID, E_TOO_LARGE, E_INVALID_WEIGHT,
 * E_COLUMN_TYPE) leaves the builder exactly as it was.
 * @param ctx - the builder's state
 * @param options - the freeze options
 * @returns the snapshot, the report and the staging to adopt
 */
export function runFreeze(ctx: FreezeContext, options: FreezeOptions): FreezeOutcome {
    const profile = options.profile === true;
    const timings: Record<string, number> = {};
    const start = profile ? now() : 0;
    let mark = start;
    const lap = (phase: string): void => {
        if (profile) {
            const t = now();
            timings[phase] = (timings[phase] ?? 0) + (t - mark);
            mark = t;
        }
    };
    const { directed } = ctx;
    const source = ctx.staging;
    checkFreezeOptions(options);
    const useArena = options.arena !== false;
    const policy = options.duplicateEdges ?? ctx.options.duplicateEdges;
    const loopPolicy = ctx.options.selfLoops;
    const oldNodeBound = source.nodeBound;
    const oldEdgeBound = source.edgeBound;
    const tombstonedEdges = oldEdgeBound - source.liveEdgeCount;

    // pre-checks that must not leave a half-applied freeze behind
    if (loopPolicy === "error") {
        checkSelfLoops(source);
    }
    const liveArcs = directed ? source.liveEdgeCount : 2 * source.liveEdgeCount - source.selfLoopCount;
    if (liveArcs > MAX_COUNT) {
        throw new GraphFormatError("E_TOO_LARGE", `arc count ${liveArcs} exceeds MAX_COUNT`, {
            count: liveArcs,
            max: MAX_COUNT,
        });
    }

    // steps 1-2: compaction of tombstones and dropped loops into a fresh staging
    let staging = source;
    let nodeRemap: IndexRemap | null = null;
    let edgeRemap: IndexRemap | null = null;
    let droppedSelfLoops = 0;
    const dropLoops = loopPolicy === "drop" && source.selfLoopCount > 0;
    if (source.hasTombstones || dropLoops) {
        const nodes = remapKeepingBits(oldNodeBound, source.nodeAlive.view());
        const edges = remapKeepingBits(oldEdgeBound, dropLoops ? liveNonLoopBits(source) : source.edgeAlive.view());
        droppedSelfLoops = dropLoops ? source.selfLoopCount : 0;
        staging = compactStaging(source, {
            nodes,
            edges,
            edgeValues: null,
            mergedWeights: null,
            mergedWeightStored: null,
            mergedWeightSet: null,
        });
        nodeRemap = nodes;
        edgeRemap = edges;
    }
    lap("compact");

    // steps 3-6: arcs and the two counting sorts
    const nodeCount = staging.nodeBound;
    let sorted = sortIntoCore(sortInputOf(staging, directed), useArena && policy === "keep");
    lap("sort");

    // step 7: duplicate policy
    let walk: DuplicateWalk | null = null;
    let mergeRemap: IndexRemap | null = null;
    if (policy !== "keep") {
        const compaction = edgeRemap;
        walk = walkDuplicates(sorted, staging, directed, policy, ctx.options.weighted !== false, (e) =>
            compaction === null ? e : compaction.origin[e],
        );
        if (walk !== null) {
            const { survivorOf } = walk;
            const edges = remapSurvivors(survivorOf);
            let mergedWeights: Float64Array | null = null;
            let mergedWeightStored: U32 | null = null;
            if (walk.reduced !== null && walk.stored !== null) {
                mergedWeights = new Float64Array(edges.count);
                mergedWeightStored = new Uint32Array(bitmapWordCount(edges.count));
                for (let e = 0; e < survivorOf.length; e++) {
                    if (survivorOf[e] === e && bitmapGet(walk.stored, e)) {
                        mergedWeights[edges.remap[e]] = walk.reduced[e];
                        bitmapSet(mergedWeightStored, edges.remap[e]);
                    }
                }
            }
            let mergedWeightSet: U32 | null = null;
            if (walk.explicit !== null) {
                mergedWeightSet = new Uint32Array(bitmapWordCount(edges.count));
                for (let e = 0; e < survivorOf.length; e++) {
                    if (survivorOf[e] === e && bitmapGet(walk.explicit, e)) {
                        bitmapSet(mergedWeightSet, edges.remap[e]);
                    }
                }
            }
            // refersTo edge values follow the survivor (design sections 5.11 and 7.3), not the row remap
            const edgeValues = new Uint32Array(survivorOf.length);
            for (let e = 0; e < survivorOf.length; e++) {
                edgeValues[e] = edges.remap[survivorOf[e]];
            }
            staging = compactStaging(staging, {
                nodes: identityRemap(nodeCount),
                edges,
                edgeValues,
                mergedWeights,
                mergedWeightStored,
                mergedWeightSet,
            });
            mergeRemap = edges;
            sorted = sortIntoCore(sortInputOf(staging, directed), useArena);
        } else if (useArena) {
            sorted = { ...sorted, core: copyCoreIntoArena(sorted.core) };
        }
        lap("duplicates");
    }
    const edgeCount = staging.edgeBound;
    const { core, arcCount, selfLoopCount, flags } = sorted;

    // steps 8-10: weights, id map, columns
    const shadow = weightShadowColumn(staging, edgeCount, ctx.options.weightDtype);
    lap("weights");
    const ids = idMapOf(staging, nodeCount);
    lap("ids");
    const nodes = tableOf("node", nodeCount, staging.nodeColumns);
    const edges = tableOf("edge", edgeCount, staging.edgeColumns, shadow === null ? [] : [shadow]);
    const graph = graphTableOf(ctx.graphValues);
    const extensions = new Map<string, AttributeTable>();
    for (const table of staging.extensions) {
        extensions.set(table.name, tableOf("extension", table.rowCount, table.columns));
    }
    lap("columns");

    // step 12: the snapshot
    const parts: SnapshotParts = {
        label: options.label ?? null,
        serial: null,
        directed,
        nodeCount,
        edgeCount,
        arcCount,
        selfLoopCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags,
        ids,
        nodes,
        edges,
        graph,
        extensions,
        meta: ctx.meta,
        arena: core.arena,
        checksum: options.checksum === true,
    };
    const snapshot = createSnapshot(parts);
    if (options.prepare !== undefined && options.prepare.length > 0) {
        snapshot.prepare(options.prepare);
    }
    lap("snapshot");
    if (profile) {
        timings.total = now() - start;
    }

    const nodeRemapOut = nodeRemap !== null && !isIdentity(nodeRemap.remap) ? nodeRemap.remap : null;
    const edgeRemapOut = composeEdgeRemap(oldEdgeBound, edgeRemap, walk, mergeRemap);
    const mergedEdges = walk === null ? 0 : walk.merged;
    const report: FreezeReport = Object.freeze({
        nodeRemap: nodeRemapOut,
        edgeRemap: edgeRemapOut,
        compacted: nodeRemapOut !== null || edgeRemapOut !== null,
        droppedSelfLoops,
        mergedEdges,
        droppedEdges: tombstonedEdges + droppedSelfLoops + mergedEdges,
        widened: Object.freeze(ctx.widened.map((w) => Object.freeze({ ...w }))),
        timings: Object.freeze({ ...timings }),
    });
    return { snapshot, report, staging };
}
