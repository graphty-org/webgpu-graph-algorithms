/**
 * GraphBuilder: the only mutable object in the package and the producer of snapshots (design
 * section 6). A long-lived, structure-of-arrays accumulator (compact.ts) that can be frozen
 * repeatedly through the pipeline of freeze.ts. Node indices are assigned in first-seen order and
 * edge indices in addEdge order (invariant I14); removal tombstones in O(degree) through the two
 * incidence lists; a compacting freeze renumbers the builder's own indices to equal the new
 * snapshot's (decision C7) and reports the remaps; freeze() never shares core arrays or columns with
 * the snapshot (I18) while the id `Map` and `ids` array are shared by design (section 4.2).
 *
 * Every public method and getter throws E_BUILDER_DISPOSED after dispose(); every throw leaves the
 * builder in a consistent state (design section 11.1): the failing operation is not applied (the
 * record, composition and expansion methods validate everything they will write before the first
 * mutation), and a freeze that fails commits nothing.
 */

import { assertJsonValue, dtypeOfArray, metaToDecl, resolveColumnMeta } from "../columns/column.js";
import { assertWellFormedString } from "../columns/dictionary.js";
import { type InferredDtype, wideningRank } from "../columns/infer.js";
import { INVALID_INDEX, MAX_COUNT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { describeId, validateNodeId } from "../ids/node-id-map.js";
import { detachString } from "../ids/string-store.js";
import { EMPTY_GRAPH_META, resolveGraphMeta } from "../snapshot/graph-meta.js";
import {
    type BuilderOptionsPatch,
    type Column,
    type ColumnDecl,
    type ColumnDeclPatch,
    type ColumnDomain,
    type ColumnHandle,
    type ColumnMeta,
    type Dtype,
    type DuplicatePolicy,
    type ExtensionHandle,
    type F32,
    type F64,
    type FreezeOptions,
    type FreezeReport,
    type GraphBuilderContract,
    type GraphBuilderOptions,
    type GraphMeta,
    type GraphMetaPatch,
    type GraphSnapshot,
    type NodeId,
    type ResolvedBuilderOptions,
    type SetDirectedOptions,
    type TypedArrayData,
    type U32,
} from "../types/index.js";
import {
    createStagingColumn,
    type ExtensionStaging,
    type IndexMaps,
    inferInitialDtype,
    inferredEquivalent,
    Staging,
    StagingColumn,
} from "./compact.js";
import { type FreezeContext, type GraphValue, runFreeze, type Widening } from "./freeze.js";
import { assertDuplicatePolicy } from "./options.js";

// ============================================================ option handling

const SELF_LOOP_POLICIES: ReadonlySet<string> = new Set(["keep", "drop", "error"]);
const WEIGHT_DTYPES: ReadonlySet<string> = new Set(["f32", "f64"]);
/** The name of the bool edge column written by an in-place expansion (design section 3.6). */
const DIRECTED_COLUMN = "graphty.directed";

/** The name of the u32 edge column pairing the halves of an expanded edge (design section 3.6). */
const PAIR_COLUMN = "graphty.pair";

/** The builder's option fields other than the current direction, resolved once at construction. */
interface FixedOptions {
    readonly weighted: boolean | "auto";
    readonly weightDtype: "f32" | "f64";
    readonly duplicateEdges: DuplicatePolicy;
    readonly selfLoops: "keep" | "drop" | "error";
    readonly addMissingNodes: boolean;
    readonly expectedNodes: number | null;
    readonly expectedEdges: number | null;
}

/**
 * The E_UNSUPPORTED error for an option value outside its documented set.
 * @param field - the option name
 * @param found - the value
 * @returns the error
 */
function optionError(field: string, found: unknown): GraphFormatError {
    return new GraphFormatError("E_UNSUPPORTED", `builder option ${field} has an unsupported value ${String(found)}`, {
        field,
        found,
        reason: "unsupported option",
    });
}

/**
 * Resolve the constructor options (design section 12.2) with their defaults, checking every enum.
 * @param options - the constructor options
 * @returns the fixed options and the initial direction
 */
function resolveOptions(options: GraphBuilderOptions): { readonly fixed: FixedOptions; readonly directed: boolean } {
    if (typeof options !== "object" || options === null) {
        throw optionError("options", options);
    }
    if (typeof options.directed !== "boolean") {
        throw optionError("directed", options.directed);
    }
    const weighted = options.weighted ?? "auto";
    if (weighted !== "auto" && typeof weighted !== "boolean") {
        throw optionError("weighted", weighted);
    }
    const weightDtype = options.weightDtype ?? "f32";
    if (!WEIGHT_DTYPES.has(weightDtype)) {
        throw optionError("weightDtype", weightDtype);
    }
    const duplicateEdges = assertDuplicatePolicy("duplicateEdges", options.duplicateEdges ?? "keep", optionError);
    const selfLoops = options.selfLoops ?? "keep";
    if (!SELF_LOOP_POLICIES.has(selfLoops)) {
        throw optionError("selfLoops", selfLoops);
    }
    const addMissingNodes = options.addMissingNodes ?? true;
    if (typeof addMissingNodes !== "boolean") {
        throw optionError("addMissingNodes", addMissingNodes);
    }
    const hint = (field: "expectedNodes" | "expectedEdges"): number | null => {
        const value = options[field];
        if (value === undefined) {
            return null;
        }
        if (!Number.isInteger(value) || value < 0) {
            throw optionError(field, value);
        }
        return value;
    };
    return {
        fixed: {
            weighted,
            weightDtype,
            duplicateEdges,
            selfLoops,
            addMissingNodes,
            expectedNodes: hint("expectedNodes"),
            expectedEdges: hint("expectedEdges"),
        },
        directed: options.directed,
    };
}

/**
 * Drop the explicitly undefined fields of a patch so it can be spread over defaults.
 * @param patch - the patch
 * @returns the defined fields
 */
function definedFields(patch: BuilderOptionsPatch): Partial<GraphBuilderOptions> {
    const out: Partial<GraphBuilderOptions> = {};
    for (const key of Object.keys(patch) as (keyof GraphBuilderOptions)[]) {
        const value = patch[key];
        if (value !== undefined) {
            (out as Record<string, unknown>)[key] = value;
        }
    }
    return out;
}

/**
 * Whether two column declarations describe the same storage (the "same declaration" rule of design
 * section 11.3 for a repeated declare*Column).
 * @param a - one resolved metadata
 * @param b - the other
 * @returns true when dtype, components and the list child agree
 */
function sameShape(a: ColumnMeta, b: ColumnMeta): boolean {
    return (
        a.dtype === b.dtype &&
        a.components === b.components &&
        a.itemDtype === b.itemDtype &&
        a.itemComponents === b.itemComponents
    );
}

/** The name index of one staging column array (see GraphBuilder.columnIndex). */
interface ColumnNameIndex {
    /** The array the map describes. */
    readonly columns: readonly StagingColumn[];
    /** Column name -> index in `columns`. */
    readonly map: Map<string, number>;
}

// ============================================================ the class

/**
 * The mutable accumulator that produces frozen snapshots (design sections 6 and 12.2). Construct
 * with `{ directed }` (required) and the policies of GraphBuilderOptions; push nodes and edges by id
 * or by index; declare and write attribute columns; call `freeze()` as often as needed. Implements
 * GraphSink, the subset importers program against (design section 8.3).
 */
export class GraphBuilder implements GraphBuilderContract {
    private readonly fixed: FixedOptions;

    private directedValue: boolean;

    private locked = false;

    private optionsCache: ResolvedBuilderOptions | null = null;

    private staging: Staging;

    /** Column name -> index of the node staging columns; rebuilt when the column array is replaced. */
    private nodeNames: ColumnNameIndex | null = null;

    /** Column name -> index of the edge staging columns; rebuilt when the column array is replaced. */
    private edgeNames: ColumnNameIndex | null = null;

    private graphValues = new Map<string, GraphValue>();

    private metaValue: GraphMeta = EMPTY_GRAPH_META;

    private widenings: Widening[] = [];

    private mutations = 0;

    private dirtyFlag = true;

    private disposed = false;

    /**
     * Create an empty builder.
     * @param options - the options; `directed` is required
     */
    constructor(options: GraphBuilderOptions) {
        const resolved = resolveOptions(options);
        this.fixed = resolved.fixed;
        this.directedValue = resolved.directed;
        this.staging = this.freshStaging();
    }

    /**
     * Seed a builder from a snapshot in O(n + m) (design section 6.6): ids, edges in logical order
     * with their declared orientation, weights (through the role-"weight" column when present, so an
     * f64 shadow and the explicit / omitted record survive), columns, extension tables and meta;
     * indices are preserved. The staging precision follows an f64 shadow unless the patch says
     * otherwise.
     * @param snapshot - the snapshot to continue
     * @param options - option overrides; `directed` defaults to the snapshot's
     * @returns the builder
     */
    static from(snapshot: GraphSnapshot, options?: BuilderOptionsPatch): GraphBuilder {
        const patch = definedFields(options ?? {});
        const shadow = snapshot.edges.byRole("weight");
        const weightDtype = patch.weightDtype ?? (shadow !== null && shadow.dtype === "f64" ? "f64" : undefined);
        // a declared weight array is never dropped (design section 3.7): a weighted snapshot whose
        // edges all omitted the weight seeds a `weighted: true` builder unless the patch says otherwise
        const weighted = patch.weighted ?? (snapshot.flags.weighted ? true : undefined);
        const builder = new GraphBuilder({
            directed: snapshot.directed,
            expectedNodes: snapshot.nodeCount,
            expectedEdges: snapshot.edgeCount,
            ...patch,
            ...(weightDtype === undefined ? {} : { weightDtype }),
            ...(weighted === undefined ? {} : { weighted }),
        });
        builder.addGraph(snapshot, { onDuplicateNode: "error" });
        builder.setMeta(snapshot.meta);
        return builder;
    }

    // ---------------------------------------------------------------- options and state

    /**
     * The resolved options; `directed` is the current value.
     * @returns the options
     */
    get options(): ResolvedBuilderOptions {
        this.check();
        this.optionsCache ??= Object.freeze({ directed: this.directedValue, ...this.fixed });
        return this.optionsCache;
    }

    /**
     * The current direction.
     * @returns true when directed
     */
    get directed(): boolean {
        this.check();
        return this.directedValue;
    }

    /**
     * Whether lockDirected() was called.
     * @returns true when locked
     */
    get directedLocked(): boolean {
        this.check();
        return this.locked;
    }

    /**
     * Live node count.
     * @returns the count
     */
    get nodeCount(): number {
        this.check();
        return this.staging.liveNodeCount;
    }

    /**
     * Live edge count (exact, from the incidence lists).
     * @returns the count
     */
    get edgeCount(): number {
        this.check();
        return this.staging.liveEdgeCount;
    }

    /**
     * Next node index to be assigned.
     * @returns the bound
     */
    get nodeBound(): number {
        this.check();
        return this.staging.nodeBound;
    }

    /**
     * Next logical edge index to be assigned.
     * @returns the bound
     */
    get edgeBound(): number {
        this.check();
        return this.staging.edgeBound;
    }

    /**
     * Increments on every topology or weight mutation; column writes and freeze() do not count.
     * @returns the count
     */
    get mutationCount(): number {
        this.check();
        return this.mutations;
    }

    /**
     * Whether the builder was mutated since the last freeze() (true for a builder never frozen).
     * @returns true when a fresh freeze would differ from the last one
     */
    get dirty(): boolean {
        this.check();
        return this.dirtyFlag;
    }

    /**
     * Change the direction (design section 6.6): a no-op when unchanged; free while no live edge
     * exists; with live edges only undirected -> directed with `expand`, which appends a mirror edge
     * for every live edge and writes the graphty.directed / graphty.pair columns. E_DIRECTED when
     * locked or refused.
     * @param directed - the new direction
     * @param options - `expand` for the in-place expansion
     */
    setDirected(directed: boolean, options?: SetDirectedOptions): void {
        this.check();
        if (directed === this.directedValue) {
            return;
        }
        if (this.locked) {
            throw new GraphFormatError("E_DIRECTED", "the builder's direction is locked", {
                reason: "locked",
                directed: this.directedValue,
            });
        }
        if (this.staging.liveEdgeCount === 0) {
            this.directedValue = directed;
            this.optionsCache = null;
            this.mutated();
            return;
        }
        if (!directed) {
            throw new GraphFormatError("E_DIRECTED", "a builder with edges cannot become undirected", {
                reason: "edges present",
                edgeCount: this.staging.liveEdgeCount,
            });
        }
        if (options?.expand !== true) {
            throw new GraphFormatError(
                "E_DIRECTED",
                "an undirected builder with edges becomes directed only with { expand: true }",
                { reason: "expand required", edgeCount: this.staging.liveEdgeCount },
            );
        }
        this.expandToDirected();
    }

    /** Fix the direction: every later changing setDirected() throws E_DIRECTED. */
    lockDirected(): void {
        this.check();
        this.locked = true;
    }

    // ---------------------------------------------------------------- nodes

    /**
     * Add a node, or return the index of the live node with this id; a tombstoned id is revived at
     * its old index (design section 6.6).
     * @param id - the node id; E_INVALID_ID when illegal
     * @returns the node index
     */
    addNode(id: NodeId): number {
        this.check();
        return this.addValidatedNode(validateNodeId(id));
    }

    /**
     * Add many nodes, writing every id's index (new or existing) into `out`.
     * @param ids - the ids
     * @param out - receives the indices; allocated when omitted; E_COLUMN_LENGTH when too short
     * @returns `out`
     */
    addNodes(ids: Iterable<NodeId>, out?: U32): U32 {
        this.check();
        const list = Array.isArray(ids) ? (ids as readonly NodeId[]) : [...ids];
        const validated = list.map((id) => validateNodeId(id));
        if (out !== undefined && out.length < validated.length) {
            throw new GraphFormatError("E_COLUMN_LENGTH", `out has ${out.length} entries for ${validated.length} ids`, {
                expected: validated.length,
                found: out.length,
            });
        }
        const target = out ?? new Uint32Array(validated.length);
        for (let i = 0; i < validated.length; i++) {
            target[i] = this.addValidatedNode(validated[i]);
        }
        return target;
    }

    /**
     * Append `count` nodes whose ids are their own indices, never touching the id Map while every
     * node is anonymous (design section 6.6).
     * @param count - how many; E_TOO_LARGE beyond MAX_COUNT; E_DUPLICATE_ID when an index is already an id
     * @returns the first new index
     */
    addAnonymousNodes(count: number): number {
        this.check();
        if (!Number.isInteger(count) || count < 0) {
            throw new GraphFormatError("E_INDEX_RANGE", `node count ${count} is not a non-negative integer`, {
                count,
            });
        }
        const { staging } = this;
        const first = staging.nodeBound;
        this.checkNodeLimit(count);
        if (staging.idToIndex !== null) {
            for (let i = first; i < first + count; i++) {
                if (staging.idToIndex.has(i)) {
                    throw new GraphFormatError(
                        "E_DUPLICATE_ID",
                        `node id ${i} exists; anonymous node ${i} cannot use it`,
                        {
                            id: i,
                            index: staging.idToIndex.get(i),
                        },
                    );
                }
            }
        }
        for (let i = 0; i < count; i++) {
            staging.pushNode(null);
        }
        if (count > 0) {
            this.mutated();
        }
        return first;
    }

    /**
     * Whether a live node has this id (total: an illegal id is simply absent).
     * @param id - the id
     * @returns true when present and alive
     */
    hasNode(id: NodeId): boolean {
        this.check();
        return this.lookup(id) !== INVALID_INDEX;
    }

    /**
     * Total lookup of a live node.
     * @param id - the id
     * @returns the index, or INVALID_INDEX
     */
    indexOf(id: NodeId): number {
        this.check();
        return this.lookup(id);
    }

    /**
     * The id of a live node index.
     * @param index - the index; E_INDEX_RANGE when out of range or tombstoned
     * @returns the id
     */
    idOf(index: number): NodeId {
        this.check();
        const { staging } = this;
        if (!this.isLiveNode(index)) {
            throw new GraphFormatError("E_INDEX_RANGE", `node index ${index} is out of range or removed`, {
                index,
                bound: staging.nodeBound,
            });
        }
        return staging.ids === null ? index : staging.ids[index];
    }

    /**
     * Tombstone a node and every live incident edge (O(degree)); bumps mutationCount.
     * @param id - the id; E_UNKNOWN_NODE when absent or already removed
     * @returns the removed live incident edge indices, ascending
     */
    removeNode(id: NodeId): U32 {
        this.check();
        const index = this.lookup(id);
        if (index === INVALID_INDEX) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `unknown node id ${describeId(id)}`, { id });
        }
        return this.removeLiveNode(index);
    }

    /**
     * Tombstone a node by index and every live incident edge.
     * @param index - the index; E_UNKNOWN_NODE (details.index) when out of range or removed
     * @returns the removed live incident edge indices, ascending
     */
    removeNodeByIndex(index: number): U32 {
        this.check();
        if (!this.isLiveNode(index)) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `node index ${index} is out of range or removed`, { index });
        }
        return this.removeLiveNode(index);
    }

    /**
     * Grow staging capacity ahead of a bulk push.
     * @param nodes - the node count to fit
     * @param edges - the edge count to fit
     */
    reserve(nodes?: number, edges?: number): void {
        this.check();
        const check = (field: string, value: number | undefined): number => {
            if (value === undefined) {
                return 0;
            }
            if (!Number.isInteger(value) || value < 0 || value > MAX_COUNT) {
                throw new GraphFormatError(
                    "E_TOO_LARGE",
                    `reserve(${field}) ${value} is not an integer in [0, MAX_COUNT]`,
                    {
                        field,
                        count: value,
                        max: MAX_COUNT,
                    },
                );
            }
            return value;
        };
        this.staging.reserve(check("nodes", nodes), check("edges", edges));
    }

    // ---------------------------------------------------------------- edges

    /**
     * Add a logical edge by id; unknown endpoints are created when addMissingNodes is set (a
     * tombstoned endpoint is revived), else E_UNKNOWN_NODE. A NaN weight is E_INVALID_WEIGHT.
     * @param source - source id
     * @param target - target id
     * @param weight - the weight; 1 when omitted
     * @returns the logical edge index
     */
    addEdge(source: NodeId, target: NodeId, weight?: number): number {
        this.check();
        const w = this.checkWeight(weight, undefined);
        const u = this.resolveEndpoint(validateNodeId(source));
        const v = this.resolveEndpoint(validateNodeId(target));
        return this.pushEdge(u, v, w);
    }

    /**
     * Add a logical edge by node index.
     * @param u - source index; E_UNKNOWN_NODE (details.index) when out of range or removed
     * @param v - target index
     * @param weight - the weight; 1 when omitted
     * @returns the logical edge index
     */
    addEdgeByIndex(u: number, v: number, weight?: number): number {
        this.check();
        const w = this.checkWeight(weight, undefined);
        this.checkLiveNode(u);
        this.checkLiveNode(v);
        return this.pushEdge(u, v, w);
    }

    /**
     * Bulk add edges by node index (the index-space path used with addAnonymousNodes). Everything is
     * validated before the first edge is pushed.
     * @param src - source indices
     * @param dst - target indices; E_COLUMN_LENGTH when the lengths differ
     * @param weights - per-edge weights; 1 when omitted; E_INVALID_WEIGHT for NaN
     * @returns the first new logical edge index
     */
    addEdges(src: U32, dst: U32, weights?: F32 | F64): number {
        this.check();
        const count = src.length;
        if (dst.length !== count || (weights !== undefined && weights.length !== count)) {
            throw new GraphFormatError(
                "E_COLUMN_LENGTH",
                `addEdges: src has ${count} entries, dst ${dst.length}, weights ${weights === undefined ? "none" : weights.length}`,
                { src: count, dst: dst.length, weights: weights?.length ?? null },
            );
        }
        for (let i = 0; i < count; i++) {
            this.checkLiveNode(src[i]);
            this.checkLiveNode(dst[i]);
        }
        let explicit: F32 | F64 | null = null;
        if (weights !== undefined) {
            for (let i = 0; i < count; i++) {
                this.checkWeight(weights[i], i);
            }
            explicit = this.fixed.weighted === false ? null : weights;
        }
        this.checkEdgeLimit(count, this.arcsOf(src, dst, count));
        const { staging } = this;
        staging.reserve(0, staging.edgeBound + count);
        const first = staging.pushEdges(src, dst, explicit);
        if (count > 0) {
            this.mutated();
        }
        return first;
    }

    /**
     * Bulk add edges by id. Every id and weight is validated before anything is added.
     * @param src - source ids
     * @param dst - target ids; E_COLUMN_LENGTH when the lengths differ
     * @param weights - per-edge weights; 1 when omitted
     * @returns the first new logical edge index
     */
    addEdgesByIds(src: ArrayLike<NodeId>, dst: ArrayLike<NodeId>, weights?: ArrayLike<number>): number {
        this.check();
        const count = src.length;
        if (dst.length !== count || (weights !== undefined && weights.length !== count)) {
            throw new GraphFormatError(
                "E_COLUMN_LENGTH",
                `addEdgesByIds: src has ${count} entries, dst ${dst.length}, weights ${weights === undefined ? "none" : weights.length}`,
                { src: count, dst: dst.length, weights: weights?.length ?? null },
            );
        }
        const sources = new Array<NodeId>(count);
        const targets = new Array<NodeId>(count);
        const checked = new Array<number | undefined>(count);
        for (let i = 0; i < count; i++) {
            sources[i] = validateNodeId(src[i]);
            targets[i] = validateNodeId(dst[i]);
            checked[i] = this.checkWeight(weights === undefined ? undefined : weights[i], i);
        }
        if (!this.fixed.addMissingNodes) {
            for (let i = 0; i < count; i++) {
                this.requireLive(sources[i]);
                this.requireLive(targets[i]);
            }
        }
        this.checkEdgeLimit(count, this.directedValue ? count : 2 * count);
        const first = this.staging.edgeBound;
        for (let i = 0; i < count; i++) {
            const u = this.resolveEndpoint(sources[i]);
            const v = this.resolveEndpoint(targets[i]);
            this.staging.pushEdge(u, v, checked[i]);
        }
        if (count > 0) {
            this.mutated();
        }
        return first;
    }

    /**
     * Tombstone an edge (O(1)); bumps mutationCount when something was removed.
     * @param edge - the logical edge index
     * @returns true when a live edge was removed
     */
    removeEdge(edge: number): boolean {
        this.check();
        if (!this.isLiveEdge(edge)) {
            return false;
        }
        this.staging.killEdge(edge);
        this.mutated();
        return true;
    }

    /**
     * Whether an edge index is live.
     * @param edge - the logical edge index
     * @returns true when live
     */
    hasEdge(edge: number): boolean {
        this.check();
        return this.isLiveEdge(edge);
    }

    /**
     * Read a live edge's endpoints back.
     * @param edge - the logical edge index; E_INDEX_RANGE when out of range or removed
     * @returns [source index, target index] in declared orientation
     */
    edgeEndpoints(edge: number): readonly [source: number, target: number] {
        this.check();
        this.checkEdge(edge);
        return [this.staging.src.get(edge), this.staging.dst.get(edge)];
    }

    /**
     * Read a live edge's weight back.
     * @param edge - the logical edge index; E_INDEX_RANGE when out of range or removed
     * @returns the weight (1 when unweighted)
     */
    edgeWeight(edge: number): number {
        this.check();
        this.checkEdge(edge);
        const { weight } = this.staging;
        return weight === null ? 1 : weight.get(edge);
    }

    /**
     * Set the weight of a live edge, allocating the weight array on first use and marking the weight
     * explicit; bumps mutationCount.
     * @param edge - the logical edge index; E_INDEX_RANGE when out of range or removed
     * @param weight - the weight; E_INVALID_WEIGHT for NaN
     */
    setEdgeWeight(edge: number, weight: number): void {
        this.check();
        this.checkEdge(edge);
        const w = this.checkWeight(weight, undefined);
        const { staging } = this;
        if (w !== undefined) {
            staging.ensureWeights().set(edge, w);
            staging.trackWeight(edge, true);
        }
        this.mutated();
    }

    /**
     * Live edges leaving a node, from the out-list (O(degree)). On an undirected builder every
     * incident edge leaves the node (invariant I7, the snapshot's row holds both orientations), so
     * both lists are walked and a self-loop is listed once.
     * @param index - the node index; E_INDEX_RANGE when out of range
     * @returns a fresh ascending array of live edge indices
     */
    outEdgesOf(index: number): U32 {
        this.check();
        this.checkNodeIndex(index);
        return this.directedValue ? this.staging.outEdges(index) : this.staging.incidentEdges(index);
    }

    /**
     * Live edges entering a node, from the in-list (O(degree)); on an undirected builder the same
     * set as outEdgesOf (the alias rule of inDegree === outDegree).
     * @param index - the node index; E_INDEX_RANGE when out of range
     * @returns a fresh ascending array of live edge indices
     */
    inEdgesOf(index: number): U32 {
        this.check();
        this.checkNodeIndex(index);
        return this.directedValue ? this.staging.inEdges(index) : this.staging.incidentEdges(index);
    }

    /**
     * Live edges u -> v (undirected: either orientation), from the incidence lists.
     * @param u - source index; E_INDEX_RANGE when out of range
     * @param v - target index; E_INDEX_RANGE when out of range
     * @returns a fresh ascending array of live edge indices
     */
    findEdges(u: number, v: number): U32 {
        this.check();
        this.checkNodeIndex(u);
        this.checkNodeIndex(v);
        const { staging } = this;
        const found: number[] = [];
        for (let e = staging.firstOut.get(u); e !== INVALID_INDEX; e = staging.nextOut.get(e)) {
            if (staging.edgeAlive.get(e) && staging.dst.get(e) === v) {
                found.push(e);
            }
        }
        if (!this.directedValue && u !== v) {
            for (let e = staging.firstIn.get(u); e !== INVALID_INDEX; e = staging.nextIn.get(e)) {
                if (staging.edgeAlive.get(e) && staging.src.get(e) === v) {
                    found.push(e);
                }
            }
        }
        return Uint32Array.from(found).sort();
    }

    // ---------------------------------------------------------------- attributes

    /**
     * Declare a node column; the same shape again returns the existing handle, a different shape is
     * E_COLUMN_EXISTS, a taken role E_DUPLICATE_ROLE.
     * @param decl - the declaration
     * @returns the column handle
     */
    declareNodeColumn(decl: ColumnDecl): ColumnHandle {
        this.check();
        return this.declare("node", decl);
    }

    /**
     * Declare an edge column; the same shape again returns the existing handle, a different shape is
     * E_COLUMN_EXISTS, a taken role E_DUPLICATE_ROLE.
     * @param decl - the declaration
     * @returns the column handle
     */
    declareEdgeColumn(decl: ColumnDecl): ColumnHandle {
        this.check();
        return this.declare("edge", decl);
    }

    /**
     * Look up a node column handle.
     * @param name - the column name
     * @returns the handle, or INVALID_INDEX when absent
     */
    nodeColumn(name: string): ColumnHandle {
        this.check();
        return this.handleOf(this.staging.nodeColumns, name);
    }

    /**
     * Look up an edge column handle.
     * @param name - the column name
     * @returns the handle, or INVALID_INDEX when absent
     */
    edgeColumn(name: string): ColumnHandle {
        this.check();
        return this.handleOf(this.staging.edgeColumns, name);
    }

    /**
     * Set one node cell; a string column name auto-declares an inferred column (design section 5.1)
     * on the first set value, and an unset value for an undeclared name is a no-op.
     * @param column - the handle (E_UNKNOWN_COLUMN when stale) or name
     * @param index - the node index; E_INDEX_RANGE when out of range
     * @param value - the value; undefined / null unset the row
     */
    setNodeValue(column: ColumnHandle | string, index: number, value: unknown): void {
        this.check();
        this.checkNodeIndex(index);
        this.writeCell("node", column, index, value);
    }

    /**
     * Set one edge cell; a string column name auto-declares an inferred column.
     * @param column - the handle (E_UNKNOWN_COLUMN when stale) or name
     * @param edge - the logical edge index; E_INDEX_RANGE when out of range
     * @param value - the value; undefined / null unset the row
     */
    setEdgeValue(column: ColumnHandle | string, edge: number, value: unknown): void {
        this.check();
        this.checkEdgeIndex(edge);
        this.writeCell("edge", column, edge, value);
    }

    /**
     * Widen an inferred node column to a wider dtype of the design section 5.1 order without
     * changing any value (an importer that knows from the lexical grammar that a column of `2.0`
     * cells is f64 although every value so far was integral). A no-op when the column is already
     * as wide; E_COLUMN_TYPE for a declared (non-inferred) column, a dtype outside the inference
     * order or a narrower one; the widening is reported in FreezeReport.widened.
     * @param column - the handle (E_UNKNOWN_COLUMN when stale) or name (E_UNKNOWN_COLUMN when absent)
     * @param dtype - the dtype to widen to
     */
    widenNodeColumn(column: ColumnHandle | string, dtype: Dtype): void {
        this.check();
        this.widenColumn("node", column, dtype);
    }

    /**
     * Widen an inferred edge column to a wider dtype of the design section 5.1 order without
     * changing any value; see widenNodeColumn().
     * @param column - the handle (E_UNKNOWN_COLUMN when stale) or name (E_UNKNOWN_COLUMN when absent)
     * @param dtype - the dtype to widen to
     */
    widenEdgeColumn(column: ColumnHandle | string, dtype: Dtype): void {
        this.check();
        this.widenColumn("edge", column, dtype);
    }

    /**
     * Bulk-set a node column from a typed array (copied into staging); replaces a column of the same
     * name. The length must be nodeBound * components (bool: ceil(nodeBound / 32) words).
     * @param name - the column name
     * @param data - the values
     * @param decl - declaration fields (dtype defaults to the array's)
     */
    setNodeColumn(name: string, data: TypedArrayData, decl?: ColumnDeclPatch): void {
        this.check();
        this.setBulkColumn("node", this.staging.nodeBound, name, data, decl ?? {});
    }

    /**
     * Bulk-set an edge column from a typed array (copied into staging); replaces a column of the same
     * name. The length must be edgeBound * components.
     * @param name - the column name
     * @param data - the values
     * @param decl - declaration fields (dtype defaults to the array's)
     */
    setEdgeColumn(name: string, data: TypedArrayData, decl?: ColumnDeclPatch): void {
        this.check();
        this.setBulkColumn("edge", this.staging.edgeBound, name, data, decl ?? {});
    }

    /**
     * Set a graph-level attribute (the graph table's single row); the value and declaration are
     * checked now, the column is built at freeze.
     * @param name - the column name
     * @param value - the value; undefined unsets it
     * @param decl - declaration fields for the column
     */
    setGraphValue(name: string, value: unknown, decl?: ColumnDeclPatch): void {
        this.check();
        const patch = decl ?? {};
        // build once now so a bad value or declaration fails at the call, not at freeze
        resolveColumnMeta(name, "graph", { dtype: "json", ...patch });
        if (value !== undefined) {
            buildGraphColumn(name, value, patch);
        }
        this.graphValues.set(name, { decl: patch, value });
    }

    /**
     * Merge fields into the graph metadata (design section 5.9); undefined fields are left alone,
     * `extra` must be a JSON value (E_COLUMN_TYPE with details.field otherwise).
     * @param meta - the fields to set
     */
    setMeta(meta: GraphMetaPatch): void {
        this.check();
        this.metaValue = resolveGraphMeta(this.metaValue, meta);
    }

    /**
     * Create an extension table (design section 5.10).
     * @param name - the table name; E_COLUMN_EXISTS when taken
     * @param decls - its columns; E_COLUMN_EXISTS for a repeated name, E_DUPLICATE_ROLE for a repeated role
     * @returns the table handle
     */
    addExtensionTable(name: string, decls: readonly ColumnDecl[]): ExtensionHandle {
        this.check();
        const { staging } = this;
        if (staging.extensions.some((table) => table.name === name)) {
            throw new GraphFormatError("E_COLUMN_EXISTS", `extension table "${name}" already exists`, { table: name });
        }
        const columns: StagingColumn[] = [];
        for (const decl of decls) {
            const column = createStagingColumn(decl.name, "extension", decl, false);
            if (columns.some((other) => other.meta.name === column.meta.name)) {
                throw new GraphFormatError("E_COLUMN_EXISTS", `column "${decl.name}" is declared twice in "${name}"`, {
                    table: name,
                    column: decl.name,
                });
            }
            checkRole(columns, column.meta, name);
            columns.push(column);
        }
        staging.extensions.push({ name, columns, rowCount: 0 });
        return (staging.extensions.length - 1) as ExtensionHandle;
    }

    /**
     * Append a row to an extension table.
     * @param table - the table handle; E_INDEX_RANGE when unknown
     * @param values - one value per declared column; E_COLUMN_LENGTH otherwise
     * @returns the new row index
     */
    addExtensionRow(table: ExtensionHandle, values: readonly unknown[]): number {
        this.check();
        const target = this.extensionOf(table);
        if (values.length !== target.columns.length) {
            throw new GraphFormatError(
                "E_COLUMN_LENGTH",
                `extension table "${target.name}" has ${target.columns.length} columns; ${values.length} values given`,
                { table: target.name, expected: target.columns.length, found: values.length },
            );
        }
        const row = target.rowCount;
        const checked = target.columns.map((column, i) => {
            const value = this.checkReference(column.meta, values[i]);
            column.checkValue(row, value);
            return value;
        });
        for (let i = 0; i < checked.length; i++) {
            target.columns[i].write(row, checked[i]);
        }
        target.rowCount = row + 1;
        return row;
    }

    /**
     * Add a node from a record: every key present overwrites that row (last-write-wins per attribute,
     * design section 6.6), auto-declaring inferred columns.
     * @param id - the node id
     * @param attrs - attribute values keyed by column name
     * @returns the node index
     */
    addNodeRecord(id: NodeId, attrs: Readonly<Record<string, unknown>>): number {
        this.check();
        const validated = validateNodeId(id);
        const keys = Object.keys(attrs);
        // every attribute is validated before the node is added (design section 11.1)
        this.checkCells("node", this.prospectiveNodeIndex(validated), keys, attrs);
        const index = this.addValidatedNode(validated);
        for (const key of keys) {
            this.writeCell("node", key, index, attrs[key]);
        }
        return index;
    }

    /**
     * Add an edge from a record; always a new edge (parallels are kept). The weight comes from
     * `weightKey` ("weight" by default; null = none) and must be a number (E_INVALID_WEIGHT).
     * @param source - source id
     * @param target - target id
     * @param attrs - attribute values keyed by column name
     * @param weightKey - the key holding the weight; default "weight"; null = no weight
     * @returns the logical edge index
     */
    addEdgeRecord(
        source: NodeId,
        target: NodeId,
        attrs: Readonly<Record<string, unknown>>,
        weightKey?: string | null,
    ): number {
        this.check();
        const key = weightKey === undefined ? "weight" : weightKey;
        let weight: number | undefined;
        if (key !== null && Object.prototype.hasOwnProperty.call(attrs, key)) {
            const raw = attrs[key];
            if (raw !== undefined && raw !== null) {
                if (typeof raw !== "number") {
                    throw new GraphFormatError("E_INVALID_WEIGHT", `edge weight "${key}" is not a number`, {
                        key,
                        found: typeof raw,
                    });
                }
                weight = raw;
            }
        }
        const keys = Object.keys(attrs).filter((name) => name !== key);
        // every attribute is validated before the edge (and any missing endpoint) is added (design 11.1)
        this.checkWeight(weight, undefined);
        this.checkCells("edge", this.staging.edgeBound, keys, attrs);
        const edge = this.addEdge(source, target, weight);
        for (const name of keys) {
            this.writeCell("edge", name, edge, attrs[name]);
        }
        return edge;
    }

    // ---------------------------------------------------------------- composition

    /**
     * Append another snapshot (design section 6.6): nodes are merged by id ("merge": the incoming set
     * rows overwrite; "error": E_DUPLICATE_ID), edges are appended in logical order with their
     * declared orientation and weights, columns declared in both graphs widen to the union dtype of
     * design section 5.1, dictionaries are re-interned, extension tables are appended with their
     * references rewritten, and graph attributes overwrite by name.
     * @param snapshot - the snapshot to append
     * @param options - duplicate-id handling
     * @param options.onDuplicateNode - "merge" (default) or "error"
     */
    addGraph(snapshot: GraphSnapshot, options?: { readonly onDuplicateNode?: "merge" | "error" | undefined }): void {
        this.check();
        const onDuplicate = options?.onDuplicateNode ?? "merge";
        if (onDuplicate !== "merge" && onDuplicate !== "error") {
            throw new GraphFormatError(
                "E_UNSUPPORTED",
                `addGraph option onDuplicateNode has an unsupported value ${String(onDuplicate)}`,
                { field: "onDuplicateNode", found: onDuplicate, reason: "unsupported option" },
            );
        }
        if (snapshot.directed !== this.directedValue) {
            throw new GraphFormatError(
                "E_DIRECTED",
                `a ${snapshot.directed ? "directed" : "undirected"} snapshot cannot be appended to a ${this.directedValue ? "directed" : "undirected"} builder`,
                { reason: "direction mismatch", directed: this.directedValue, found: snapshot.directed },
            );
        }
        const { staging } = this;
        const { nodeCount, edgeCount } = snapshot;
        // everything that can be refused is checked before the first mutation (design section 11.1):
        // duplicate ids, the count limits, weights, roles and extension column names
        const anonymous =
            staging.ids === null &&
            staging.nodeBound === 0 &&
            snapshot.ids.kind === "identity" &&
            snapshot.ids.offset === 0;
        const nodeMap = new Uint32Array(nodeCount);
        let newNodes = nodeCount;
        if (!anonymous) {
            newNodes = 0;
            for (let i = 0; i < nodeCount; i++) {
                const id = snapshot.ids.idOf(i);
                const existing = this.lookup(id);
                if (existing === INVALID_INDEX) {
                    newNodes++;
                } else if (onDuplicate === "error") {
                    throw new GraphFormatError("E_DUPLICATE_ID", `node id ${describeId(id)} already exists`, {
                        id,
                        index: existing,
                    });
                }
            }
        }
        this.checkNodeLimit(newNodes);
        const list = snapshot.edgeList();
        this.checkEdgeLimit(edgeCount, this.arcsOf(list.src, list.dst, edgeCount));
        const shadow = snapshot.edges.byRole("weight");
        const weights = new Array<number | undefined>(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            let weight: number | undefined;
            if (shadow !== null) {
                weight = shadow.isSet(e) ? (shadow.value(e) as number) : undefined;
            } else if (list.weights !== null) {
                weight = list.weights[e];
            }
            weights[e] = this.checkWeight(weight, e);
        }
        for (const column of snapshot.nodes) {
            this.checkIncomingColumn("node", column);
        }
        for (const column of snapshot.edges) {
            if (column !== shadow) {
                this.checkIncomingColumn("edge", column);
            }
        }
        for (const [name, table] of snapshot.extensions) {
            this.checkIncomingExtension(name, table);
        }
        // nodes
        if (anonymous) {
            const first = this.addAnonymousNodes(nodeCount);
            for (let i = 0; i < nodeCount; i++) {
                nodeMap[i] = first + i;
            }
        } else {
            for (let i = 0; i < nodeCount; i++) {
                const id = snapshot.ids.idOf(i);
                const existing = this.lookup(id);
                nodeMap[i] = existing === INVALID_INDEX ? this.addValidatedNode(id) : existing;
            }
        }
        // edges: declared orientation, weights through the role column when present; a weighted
        // snapshot keeps its declared weight array (design section 3.7) unless the builder is unweighted
        if (snapshot.flags.weighted && this.fixed.weighted !== false) {
            staging.ensureWeights();
        }
        const edgeMap = new Uint32Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            edgeMap[e] = this.pushEdge(nodeMap[list.src[e]], nodeMap[list.dst[e]], weights[e]);
        }
        // columns
        const refs: IndexMaps = { node: nodeMap, edge: edgeMap };
        for (const column of snapshot.nodes) {
            this.copyColumn("node", column, nodeMap, refs);
        }
        for (const column of snapshot.edges) {
            if (column !== shadow) {
                this.copyColumn("edge", column, edgeMap, refs);
            }
        }
        for (const column of snapshot.graph) {
            if (column.isSet(0)) {
                this.setGraphValue(column.meta.name, column.value(0), metaToDecl(column.meta));
            }
        }
        for (const [name, table] of snapshot.extensions) {
            this.appendExtension(name, table, refs);
        }
    }

    // ---------------------------------------------------------------- output and lifecycle

    /**
     * Run the freeze pipeline of design section 6.3 and return a snapshot; the builder keeps its
     * staging (compacted and renumbered when tombstones existed, rewritten by a merge policy) unless
     * `release` is set.
     * @param options - the freeze options
     * @returns the snapshot
     */
    freeze(options?: FreezeOptions): GraphSnapshot {
        return this.freezeWithReport(options).snapshot;
    }

    /**
     * freeze() plus the report of what was renumbered, merged, dropped and widened (design section
     * 6.6): the remaps are relative to the previous freeze (the builder's own index space on the
     * first) and null exactly when nothing was renumbered.
     * @param options - the freeze options
     * @returns the snapshot and its report
     */
    freezeWithReport(options?: FreezeOptions): { snapshot: GraphSnapshot; report: FreezeReport } {
        this.check();
        const opts = options ?? {};
        const context: FreezeContext = {
            staging: this.staging,
            directed: this.directedValue,
            options: this.options,
            graphValues: this.graphValues,
            meta: this.metaValue,
            widened: this.widenings,
        };
        const outcome = runFreeze(context, opts);
        this.staging = outcome.staging;
        this.widenings = [];
        this.dirtyFlag = false;
        if (opts.release === true) {
            this.staging = this.freshStaging();
        }
        return { snapshot: outcome.snapshot, report: outcome.report };
    }

    /** Empty the builder (nodes, edges, columns, extension tables, graph attributes, meta), keeping its options and lock. */
    clear(): void {
        this.check();
        this.staging = this.freshStaging();
        this.graphValues = new Map();
        this.metaValue = EMPTY_GRAPH_META;
        this.widenings = [];
        this.mutated();
    }

    /** Release everything; every further call throws E_BUILDER_DISPOSED. */
    dispose(): void {
        if (this.disposed) {
            return;
        }
        this.staging = new Staging({
            weightDtype: this.fixed.weightDtype,
            weighted: false,
            expectedNodes: null,
            expectedEdges: null,
            resizable: undefined,
        });
        this.graphValues = new Map();
        this.metaValue = EMPTY_GRAPH_META;
        this.widenings = [];
        this.disposed = true;
    }

    /**
     * Bytes of typed staging currently held (capacity, not length; JS arrays and the id Map excluded).
     * @returns the byte count
     */
    byteLength(): number {
        this.check();
        return this.staging.byteLength();
    }

    // ---------------------------------------------------------------- private: state

    private check(): void {
        if (this.disposed) {
            throw new GraphFormatError("E_BUILDER_DISPOSED", "the builder was disposed", {});
        }
    }

    private mutated(): void {
        this.mutations++;
        this.dirtyFlag = true;
    }

    private freshStaging(): Staging {
        return new Staging({
            weightDtype: this.fixed.weightDtype,
            weighted: this.fixed.weighted === true,
            expectedNodes: this.fixed.expectedNodes,
            expectedEdges: this.fixed.expectedEdges,
            resizable: undefined,
        });
    }

    // ---------------------------------------------------------------- private: nodes

    /**
     * The index a validated id would get from addValidatedNode: its existing (live or tombstoned)
     * index, or the next one.
     * @param id - a validated id
     * @returns the prospective index
     */
    private prospectiveNodeIndex(id: NodeId): number {
        const { staging } = this;
        if (staging.idToIndex === null) {
            return typeof id === "number" && Number.isInteger(id) && id >= 0 && id < staging.nodeBound
                ? id
                : staging.nodeBound;
        }
        return staging.idToIndex.get(typeof id === "number" && id === 0 ? 0 : id) ?? staging.nodeBound;
    }

    /**
     * The live index of an id, total.
     * @param id - any value
     * @returns the index, or INVALID_INDEX when absent, illegal or tombstoned
     */
    private lookup(id: NodeId): number {
        const { staging } = this;
        if (staging.idToIndex === null) {
            if (typeof id !== "number" || !Number.isInteger(id) || id < 0 || id >= staging.nodeBound) {
                return INVALID_INDEX;
            }
            return staging.nodeAlive.get(id) ? id : INVALID_INDEX;
        }
        const index = staging.idToIndex.get(typeof id === "number" && id === 0 ? 0 : id);
        if (index === undefined || !staging.nodeAlive.get(index)) {
            return INVALID_INDEX;
        }
        return index;
    }

    private isLiveNode(index: number): boolean {
        const { staging } = this;
        return Number.isInteger(index) && index >= 0 && index < staging.nodeBound && staging.nodeAlive.get(index);
    }

    private isLiveEdge(edge: number): boolean {
        const { staging } = this;
        return Number.isInteger(edge) && edge >= 0 && edge < staging.edgeBound && staging.edgeAlive.get(edge);
    }

    private checkNodeIndex(index: number): void {
        if (!Number.isInteger(index) || index < 0 || index >= this.staging.nodeBound) {
            throw new GraphFormatError("E_INDEX_RANGE", `node index ${index} is out of range`, {
                index,
                bound: this.staging.nodeBound,
            });
        }
    }

    private checkEdgeIndex(edge: number): void {
        if (!Number.isInteger(edge) || edge < 0 || edge >= this.staging.edgeBound) {
            throw new GraphFormatError("E_INDEX_RANGE", `edge index ${edge} is out of range`, {
                edge,
                bound: this.staging.edgeBound,
            });
        }
    }

    private checkLiveNode(index: number): void {
        if (!this.isLiveNode(index)) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `node index ${index} is out of range or removed`, { index });
        }
    }

    private checkEdge(edge: number): void {
        if (!this.isLiveEdge(edge)) {
            throw new GraphFormatError("E_INDEX_RANGE", `edge index ${edge} is out of range or removed`, {
                edge,
                bound: this.staging.edgeBound,
            });
        }
    }

    private checkNodeLimit(adding: number): void {
        if (this.staging.nodeBound + adding > MAX_COUNT) {
            throw new GraphFormatError(
                "E_TOO_LARGE",
                `node count ${this.staging.nodeBound + adding} exceeds MAX_COUNT`,
                {
                    count: this.staging.nodeBound + adding,
                    max: MAX_COUNT,
                },
            );
        }
    }

    /**
     * The E_TOO_LARGE check of invariant I3 at the add* call that would cross the limit: the edge
     * index space and the live arc count (doubled for an undirected graph, loops once) must both
     * stay at or below MAX_COUNT.
     * @param addingEdges - edges about to be added
     * @param addingArcs - arcs they contribute (1 per directed edge or loop, 2 per undirected non-loop)
     */
    private checkEdgeLimit(addingEdges: number, addingArcs: number): void {
        const { staging } = this;
        const edges = staging.edgeBound + addingEdges;
        const liveArcs = this.directedValue ? staging.liveEdgeCount : 2 * staging.liveEdgeCount - staging.selfLoopCount;
        const arcs = liveArcs + addingArcs;
        if (edges > MAX_COUNT || arcs > MAX_COUNT) {
            throw new GraphFormatError("E_TOO_LARGE", `edge count ${edges} (${arcs} arcs) exceeds MAX_COUNT`, {
                count: edges,
                arcs,
                max: MAX_COUNT,
            });
        }
    }

    /**
     * The arcs a batch of edges contributes.
     * @param src - sources
     * @param dst - targets
     * @param count - how many
     * @returns the arc count
     */
    private arcsOf(src: ArrayLike<number>, dst: ArrayLike<number>, count: number): number {
        if (this.directedValue) {
            return count;
        }
        let arcs = 0;
        for (let i = 0; i < count; i++) {
            arcs += src[i] === dst[i] ? 1 : 2;
        }
        return arcs;
    }

    /**
     * Add or revive a validated id.
     * @param id - a validated id
     * @returns the index
     */
    private addValidatedNode(id: NodeId): number {
        const { staging } = this;
        staging.materialiseIds();
        const map = staging.idToIndex as Map<NodeId, number>;
        const existing = map.get(id);
        if (existing !== undefined) {
            if (!staging.nodeAlive.get(existing)) {
                staging.nodeAlive.set(existing, true);
                staging.liveNodeCount++;
                this.mutated();
            }
            return existing;
        }
        this.checkNodeLimit(1);
        const index = staging.pushNode(typeof id === "string" ? detachString(id) : id);
        this.mutated();
        return index;
    }

    /**
     * Resolve an edge endpoint id: the live index, a revived or new node under addMissingNodes, or
     * E_UNKNOWN_NODE.
     * @param id - a validated id
     * @returns the index
     */
    private resolveEndpoint(id: NodeId): number {
        const index = this.lookup(id);
        if (index !== INVALID_INDEX) {
            return index;
        }
        if (this.fixed.addMissingNodes) {
            return this.addValidatedNode(id);
        }
        throw new GraphFormatError("E_UNKNOWN_NODE", `unknown node id ${describeId(id)}`, { id });
    }

    private requireLive(id: NodeId): void {
        if (this.lookup(id) === INVALID_INDEX) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `unknown node id ${describeId(id)}`, { id });
        }
    }

    private removeLiveNode(index: number): U32 {
        const { staging } = this;
        const removed: number[] = [];
        for (let e = staging.firstOut.get(index); e !== INVALID_INDEX; e = staging.nextOut.get(e)) {
            if (staging.edgeAlive.get(e)) {
                staging.killEdge(e);
                removed.push(e);
            }
        }
        for (let e = staging.firstIn.get(index); e !== INVALID_INDEX; e = staging.nextIn.get(e)) {
            if (staging.edgeAlive.get(e)) {
                staging.killEdge(e);
                removed.push(e);
            }
        }
        staging.nodeAlive.set(index, false);
        staging.liveNodeCount--;
        this.mutated();
        return Uint32Array.from(removed).sort();
    }

    // ---------------------------------------------------------------- private: edges

    /**
     * Validate a weight argument: NaN is E_INVALID_WEIGHT; on an unweighted (`weighted: false`) builder
     * only 1 is accepted and it is treated as omitted.
     * @param weight - the argument
     * @param edge - the bulk position, for the error details
     * @returns the weight to store, or undefined when omitted
     */
    private checkWeight(weight: number | undefined, edge: number | undefined): number | undefined {
        if (weight === undefined) {
            return undefined;
        }
        if (typeof weight !== "number" || Number.isNaN(weight)) {
            throw new GraphFormatError("E_INVALID_WEIGHT", `weight ${String(weight)} is not a number`, {
                weight,
                ...(edge === undefined ? {} : { edge }),
            });
        }
        if (this.fixed.weighted === false) {
            if (weight !== 1) {
                throw new GraphFormatError("E_INVALID_WEIGHT", `weight ${weight} on a builder declared unweighted`, {
                    weight,
                    reason: "unweighted builder",
                    ...(edge === undefined ? {} : { edge }),
                });
            }
            return undefined;
        }
        return weight;
    }

    private pushEdge(u: number, v: number, weight: number | undefined): number {
        this.checkEdgeLimit(1, this.directedValue || u === v ? 1 : 2);
        const e = this.staging.pushEdge(u, v, weight);
        this.mutated();
        return e;
    }

    /**
     * The in-place expansion of design section 6.6: every live edge u -> v gets a mirror v -> u at a
     * new index (loops are not mirrored), graphty.directed is 0 for both halves and graphty.pair links
     * them; every other column's mirror rows stay unset.
     */
    private expandToDirected(): void {
        const { staging } = this;
        const live = staging.liveEdgeCount;
        this.checkEdgeLimit(live, live);
        const directedDecl: ColumnDecl = { name: DIRECTED_COLUMN, dtype: "bool", role: "directed" };
        const pairDecl: ColumnDecl = { name: PAIR_COLUMN, dtype: "u32", role: "pair", refersTo: "edge" };
        // both declarations are checked before either is applied (design section 11.1)
        this.checkDeclaration("edge", directedDecl);
        this.checkDeclaration("edge", pairDecl);
        const directedHandle = this.declare("edge", directedDecl);
        const pairHandle = this.declare("edge", pairDecl);
        const directedColumn = staging.edgeColumns[directedHandle];
        const pairColumn = staging.edgeColumns[pairHandle];
        const bound = staging.edgeBound;
        for (let e = 0; e < bound; e++) {
            if (!staging.edgeAlive.get(e)) {
                continue;
            }
            directedColumn.write(e, false);
            const u = staging.src.get(e);
            const v = staging.dst.get(e);
            if (u === v) {
                continue;
            }
            const explicit = staging.weightExplicit(e);
            const weight = explicit && staging.weight !== null ? staging.weight.get(e) : undefined;
            const mirror = staging.pushEdge(v, u, weight);
            directedColumn.write(mirror, false);
            pairColumn.write(e, mirror);
            pairColumn.write(mirror, e);
        }
        this.directedValue = true;
        this.optionsCache = null;
        this.mutated();
    }

    // ---------------------------------------------------------------- private: columns

    private columnsOf(domain: "node" | "edge"): StagingColumn[] {
        return domain === "node" ? this.staging.nodeColumns : this.staging.edgeColumns;
    }

    private handleOf(columns: readonly StagingColumn[], name: string): ColumnHandle {
        const index = this.columnIndex(columns, name);
        return (index < 0 ? INVALID_INDEX : index) as ColumnHandle;
    }

    /**
     * The index of a staging column by name in O(1): a Map beside each column array, rebuilt when
     * the array is replaced (a fresh staging, a compaction) and kept in step by pushColumn().
     * Column names never change, so the map is invalidated only by identity or length.
     * @param columns - the node or edge staging columns
     * @param name - the column name
     * @returns the index, or -1 when absent
     */
    private columnIndex(columns: readonly StagingColumn[], name: string): number {
        const cached = columns === this.staging.nodeColumns ? this.nodeNames : this.edgeNames;
        let index = cached;
        if (index === null || index.columns !== columns || index.map.size !== columns.length) {
            const map = new Map<string, number>();
            for (let i = 0; i < columns.length; i++) {
                map.set(columns[i].meta.name, i);
            }
            index = { columns, map };
            if (columns === this.staging.nodeColumns) {
                this.nodeNames = index;
            } else if (columns === this.staging.edgeColumns) {
                this.edgeNames = index;
            }
        }
        return index.map.get(name) ?? -1;
    }

    /**
     * Append a staging column and keep the name index in step.
     * @param columns - the node or edge staging columns
     * @param column - the column to append
     * @returns the new column's index
     */
    private pushColumn(columns: StagingColumn[], column: StagingColumn): number {
        columns.push(column);
        const index = columns === this.staging.nodeColumns ? this.nodeNames : this.edgeNames;
        if (index !== null && index.columns === columns && index.map.size === columns.length - 1) {
            index.map.set(column.meta.name, columns.length - 1);
        }
        return columns.length - 1;
    }

    /**
     * Declare a column (design section 11.3): the same shape returns the existing handle, another
     * shape is E_COLUMN_EXISTS, a role held by another column E_DUPLICATE_ROLE.
     * @param domain - node or edge
     * @param decl - the declaration
     * @returns the handle
     */
    private declare(domain: "node" | "edge", decl: ColumnDecl): ColumnHandle {
        const { meta, existing } = this.checkDeclaration(domain, decl);
        if (existing !== INVALID_INDEX) {
            return existing as ColumnHandle;
        }
        const columns = this.columnsOf(domain);
        return this.pushColumn(columns, new StagingColumn(meta, false)) as ColumnHandle;
    }

    /**
     * The checks of declare() without applying anything: the resolved metadata and, when a column
     * of the same shape exists, its handle.
     * @param domain - node or edge
     * @param decl - the declaration
     * @returns the metadata and the existing handle (INVALID_INDEX when the column is new)
     */
    private checkDeclaration(domain: "node" | "edge", decl: ColumnDecl): { meta: ColumnMeta; existing: number } {
        if (typeof decl.name !== "string" || decl.name.length === 0) {
            throw new GraphFormatError("E_COLUMN_TYPE", "a column declaration needs a non-empty name", {
                field: "name",
                found: decl.name,
            });
        }
        const meta = resolveColumnMeta(decl.name, domain, decl);
        const columns = this.columnsOf(domain);
        const existing = this.columnIndex(columns, meta.name);
        if (existing >= 0) {
            if (sameShape(columns[existing].meta, meta)) {
                return { meta, existing };
            }
            throw new GraphFormatError(
                "E_COLUMN_EXISTS",
                `${domain} column "${meta.name}" is already declared as ${columns[existing].meta.dtype}`,
                { column: meta.name, domain, found: columns[existing].meta.dtype, expected: meta.dtype },
            );
        }
        checkRole(columns, meta, domain);
        return { meta, existing: INVALID_INDEX };
    }

    /**
     * The staging column a handle or name refers to, auto-declaring an inferred column for an unknown
     * name when the value is set.
     * @param domain - node or edge
     * @param column - the handle or name
     * @param value - the value about to be written (decides the inferred dtype)
     * @returns the column, or null when an unknown name receives an unset value
     */
    private resolveColumn(
        domain: "node" | "edge",
        column: ColumnHandle | string,
        value: unknown,
    ): StagingColumn | null {
        const columns = this.columnsOf(domain);
        if (typeof column === "number") {
            if (!Number.isInteger(column) || column < 0 || column >= columns.length) {
                throw new GraphFormatError("E_UNKNOWN_COLUMN", `no ${domain} column with handle ${column}`, {
                    domain,
                    handle: column,
                });
            }
            return columns[column];
        }
        const index = this.columnIndex(columns, column);
        if (index >= 0) {
            return columns[index];
        }
        if (value === undefined || value === null) {
            return null;
        }
        const dtype = inferInitialDtype(value);
        const created = createStagingColumn(column, domain, { dtype, nullable: true }, true);
        this.pushColumn(columns, created);
        return created;
    }

    /**
     * Validate every value of a record against the columns it would be written to, without writing
     * (design section 11.1: a record whose value is refused is not applied at all). An existing
     * column checks the value as its `write` would; a new name must be inferrable (a JSON value or a
     * well-formed string) unless the value is unset.
     * @param domain - node or edge
     * @param row - the row the record is meant for (error details only)
     * @param keys - the record keys
     * @param attrs - the record
     */
    private checkCells(
        domain: "node" | "edge",
        row: number,
        keys: readonly string[],
        attrs: Readonly<Record<string, unknown>>,
    ): void {
        const columns = this.columnsOf(domain);
        for (const key of keys) {
            const value = attrs[key];
            const at = this.columnIndex(columns, key);
            const column = at < 0 ? undefined : columns[at];
            if (column !== undefined) {
                column.checkValue(row, this.checkReference(column.meta, value));
            } else if (value !== undefined && value !== null) {
                inferInitialDtype(value);
                if (typeof value === "string") {
                    assertWellFormedString(value, { column: key, row });
                } else if (typeof value === "object") {
                    assertJsonValue(value, `${key}[${row}]`);
                }
            }
        }
    }

    /**
     * The refersTo range rule of invariant I12 at the write: an index-valued cell of a refersTo
     * column must be below the referenced space's current bound (a tombstoned target is a dangling
     * reference the next compaction resolves) or INVALID_INDEX; E_INDEX_RANGE otherwise. A scalar
     * INVALID_INDEX means "no reference" and is written as an unset row (E_COLUMN_TYPE on a
     * non-nullable column, which cannot hold it). Values of other shapes are left to the column's
     * own type check.
     * @param meta - the column
     * @param value - the value about to be written
     * @returns the value to write (undefined for a scalar INVALID_INDEX)
     */
    private checkReference(meta: ColumnMeta, value: unknown): unknown {
        if (meta.refersTo === null || value === undefined || value === null) {
            return value;
        }
        const bound = meta.refersTo === "node" ? this.staging.nodeBound : this.staging.edgeBound;
        const check = (index: unknown): void => {
            if (typeof index === "number" && index !== INVALID_INDEX && !(index >= 0 && index < bound)) {
                throw new GraphFormatError(
                    "E_INDEX_RANGE",
                    `column "${meta.name}" refers to ${String(meta.refersTo)} ${index}, which is not below ${bound}`,
                    { column: meta.name, refersTo: meta.refersTo, found: index, bound },
                );
            }
        };
        if (Array.isArray(value)) {
            for (const item of value as unknown[]) {
                check(item);
            }
            return value;
        }
        check(value);
        return value === INVALID_INDEX ? undefined : value;
    }

    /**
     * The checks copyColumn would fail on before it writes anything: a new column's role must be
     * free (E_DUPLICATE_ROLE).
     * @param domain - node or edge
     * @param column - the incoming column
     */
    private checkIncomingColumn(domain: "node" | "edge", column: Column): void {
        const columns = this.columnsOf(domain);
        if (this.columnIndex(columns, column.meta.name) < 0) {
            checkRole(columns, column.meta, domain);
        }
    }

    /**
     * The checks appendExtension would fail on before it writes anything: an existing table of the
     * same name must hold every incoming column (E_UNKNOWN_COLUMN); a new table's declarations must
     * resolve (E_COLUMN_EXISTS / E_DUPLICATE_ROLE).
     * @param name - the table name
     * @param table - the incoming table
     */
    private checkIncomingExtension(name: string, table: Iterable<Column>): void {
        const target = this.staging.extensions.find((t) => t.name === name);
        const columns = [...table];
        if (target === undefined) {
            const seen: StagingColumn[] = [];
            for (const column of columns) {
                const decl = { ...metaToDecl(column.meta), name: column.meta.name } as ColumnDecl;
                const created = createStagingColumn(decl.name, "extension", decl, false);
                if (seen.some((other) => other.meta.name === created.meta.name)) {
                    throw new GraphFormatError(
                        "E_COLUMN_EXISTS",
                        `column "${decl.name}" is declared twice in "${name}"`,
                        {
                            table: name,
                            column: decl.name,
                        },
                    );
                }
                checkRole(seen, created.meta, name);
                seen.push(created);
            }
            return;
        }
        for (const column of columns) {
            if (!target.columns.some((c) => c.meta.name === column.meta.name)) {
                throw new GraphFormatError(
                    "E_UNKNOWN_COLUMN",
                    `extension table "${name}" has no column "${column.meta.name}"`,
                    { table: name, column: column.meta.name },
                );
            }
        }
    }

    /**
     * The widening of widenNodeColumn / widenEdgeColumn.
     * @param domain - node or edge
     * @param column - the handle or name
     * @param dtype - the dtype to widen to
     */
    private widenColumn(domain: "node" | "edge", column: ColumnHandle | string, dtype: Dtype): void {
        const columns = this.columnsOf(domain);
        let target: StagingColumn | undefined;
        if (typeof column === "number") {
            target = Number.isInteger(column) && column >= 0 && column < columns.length ? columns[column] : undefined;
        } else {
            const index = this.columnIndex(columns, column);
            target = index < 0 ? undefined : columns[index];
        }
        if (target === undefined) {
            throw new GraphFormatError("E_UNKNOWN_COLUMN", `no ${domain} column ${String(column)}`, { domain, column });
        }
        const current = target.inferredDtype;
        if (current === null) {
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `${domain} column "${target.meta.name}" is declared, not inferred`,
                {
                    column: target.meta.name,
                    domain,
                },
            );
        }
        if (wideningRank(dtype as InferredDtype) < 0) {
            throw new GraphFormatError("E_COLUMN_TYPE", `${dtype} is not a dtype inference widens to`, {
                column: target.meta.name,
                dtype,
            });
        }
        const next = dtype as InferredDtype;
        if (wideningRank(next) < wideningRank(current)) {
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `${domain} column "${target.meta.name}" is ${current}; it cannot narrow to ${next}`,
                { column: target.meta.name, found: current, expected: next },
            );
        }
        if (next === current) {
            return;
        }
        target.widenTo(next);
        this.widenings.push({ column: target.meta.name, domain, from: current, to: next });
        this.mutated();
    }

    private writeCell(domain: "node" | "edge", column: ColumnHandle | string, row: number, value: unknown): void {
        const target = this.resolveColumn(domain, column, value);
        if (target === null) {
            return;
        }
        const widened = target.write(row, this.checkReference(target.meta, value));
        if (widened !== null) {
            this.widenings.push({ column: target.meta.name, domain, from: widened.from, to: widened.to });
        }
    }

    /**
     * Replace or create a column from a typed array (setNodeColumn / setEdgeColumn).
     * @param domain - node or edge
     * @param rows - the table's current bound
     * @param name - the column name
     * @param data - the values
     * @param decl - the declaration patch
     */
    private setBulkColumn(
        domain: "node" | "edge",
        rows: number,
        name: string,
        data: TypedArrayData,
        decl: ColumnDeclPatch,
    ): void {
        const column = stagingColumnFromTypedArray(domain, rows, name, data, decl, {
            node: this.staging.nodeBound,
            edge: this.staging.edgeBound,
        });
        const columns = this.columnsOf(domain);
        const existing = this.columnIndex(columns, name);
        checkRole(
            columns.filter((c) => c.meta.name !== name),
            column.meta,
            domain,
        );
        if (existing >= 0) {
            columns[existing] = column;
        } else {
            this.pushColumn(columns, column);
        }
    }

    private extensionOf(handle: ExtensionHandle): ExtensionStaging {
        const { extensions } = this.staging;
        if (!Number.isInteger(handle) || handle < 0 || handle >= extensions.length) {
            throw new GraphFormatError("E_INDEX_RANGE", `no extension table with handle ${handle}`, { handle });
        }
        return extensions[handle];
    }

    /**
     * Copy a snapshot column into the builder (addGraph): declared under the same metadata when the
     * name is new, else widened to the union dtype (design section 6.6); every set row is written
     * through `rowMap`, with refersTo values rewritten through `refs`.
     * @param domain - node or edge
     * @param column - the source column
     * @param rowMap - source row -> builder row
     * @param refs - source node / edge index -> builder index
     */
    private copyColumn(domain: "node" | "edge", column: Column, rowMap: U32, refs: IndexMaps): void {
        const columns = this.columnsOf(domain);
        const { name } = column.meta;
        let index = this.columnIndex(columns, name);
        if (index < 0) {
            const decl = { ...metaToDecl(column.meta), name };
            checkRole(columns, column.meta, domain);
            index = this.pushColumn(columns, createStagingColumn(name, domain, decl, false));
        } else {
            const target = columns[index];
            if (!sameShape(target.meta, column.meta)) {
                const union = wider(inferredEquivalent(target.meta.dtype), inferredEquivalent(column.meta.dtype));
                if (target.meta.dtype !== union) {
                    const from = target.meta.dtype;
                    target.widenTo(union);
                    this.widenings.push({ column: name, domain, from, to: union });
                }
            }
        }
        const target = columns[index];
        const valueMap = referenceMap(column.meta, refs);
        for (let row = 0; row < column.length; row++) {
            if (!column.isSet(row)) {
                continue;
            }
            let value = column.value(row);
            if (valueMap !== null) {
                value = remapReference(value, valueMap);
                if (value === undefined) {
                    continue;
                }
            }
            target.write(rowMap[row], value);
        }
    }

    /**
     * Append the rows of a snapshot extension table (addGraph): into the table of the same name when
     * the columns match, else a new table is declared from the source columns.
     * @param name - the table name
     * @param table - the source table
     * @param refs - source node / edge index -> builder index
     */
    private appendExtension(
        name: string,
        table: Iterable<Column> & { readonly rowCount: number },
        refs: IndexMaps,
    ): void {
        const { staging } = this;
        const sourceColumns = [...table];
        let target = staging.extensions.find((t) => t.name === name);
        if (target === undefined) {
            const decls = sourceColumns.map(
                (column) => ({ ...metaToDecl(column.meta), name: column.meta.name }) as ColumnDecl,
            );
            const handle = this.addExtensionTable(name, decls);
            target = staging.extensions[handle];
        }
        const targetColumns = target.columns;
        const first = target.rowCount;
        for (const column of sourceColumns) {
            const into = targetColumns.find((c) => c.meta.name === column.meta.name);
            if (into === undefined) {
                throw new GraphFormatError(
                    "E_UNKNOWN_COLUMN",
                    `extension table "${name}" has no column "${column.meta.name}"`,
                    { table: name, column: column.meta.name },
                );
            }
            const valueMap = referenceMap(column.meta, refs);
            for (let row = 0; row < column.length; row++) {
                if (!column.isSet(row)) {
                    into.ensureLength(first + row + 1);
                    continue;
                }
                let value = column.value(row);
                if (valueMap !== null) {
                    value = remapReference(value, valueMap);
                    if (value === undefined) {
                        into.ensureLength(first + row + 1);
                        continue;
                    }
                }
                into.write(first + row, value);
            }
        }
        for (const column of targetColumns) {
            column.ensureLength(first + table.rowCount);
        }
        target.rowCount = first + table.rowCount;
    }
}

// ============================================================ module helpers

/**
 * Enforce "at most one column per role" among staging columns (design section 5.5).
 * @param columns - the existing columns
 * @param meta - the column being added
 * @param where - the table, for the message
 */
function checkRole(columns: readonly StagingColumn[], meta: ColumnMeta, where: string): void {
    if (meta.role === null) {
        return;
    }
    const holder = columns.find((column) => column.meta.name !== meta.name && column.meta.role === meta.role);
    if (holder !== undefined) {
        throw new GraphFormatError(
            "E_DUPLICATE_ROLE",
            `role "${meta.role}" is already held by column "${holder.meta.name}" in ${where}`,
            { role: meta.role, column: meta.name, holder: holder.meta.name },
        );
    }
}

/**
 * The wider of two inferred dtypes in the order of design section 5.1.
 * @param a - one dtype
 * @param b - the other
 * @returns the wider
 */
function wider(a: InferredDtype, b: InferredDtype): InferredDtype {
    return wideningRank(a) >= wideningRank(b) ? a : b;
}

/**
 * The index map a refersTo column's values are rewritten through, if any.
 * @param meta - the column
 * @param refs - the node and edge maps
 * @returns the map of the referenced space, or null
 */
function referenceMap(meta: ColumnMeta, refs: IndexMaps): U32 | null {
    switch (meta.refersTo) {
        case "node":
            return refs.node;
        case "edge":
            return refs.edge;
        default:
            return null;
    }
}

/**
 * Rewrite an index-valued cell (a number, or an array of numbers for a list) through an index map;
 * dangling references are dropped, and a cell with nothing left is unset.
 * @param value - the cell
 * @param map - source index -> builder index
 * @returns the rewritten cell, or undefined to unset
 */
function remapReference(value: unknown, map: U32): unknown {
    if (typeof value === "number") {
        return value < map.length ? map[value] : undefined;
    }
    if (typeof value === "object" && value !== null) {
        const items = Array.from(value as ArrayLike<number>);
        const kept = items.filter((item) => item < map.length).map((item) => map[item]);
        return kept.length === 0 && items.length > 0 ? undefined : kept;
    }
    throw new GraphFormatError("E_COLUMN_TYPE", `a refersTo cell holds a ${typeof value}, not an index or a list`, {
        found: typeof value,
        reason: "refersTo",
    });
}

/**
 * Build a one-row graph column now to validate a setGraphValue call.
 * @param name - the column name
 * @param value - the value
 * @param decl - the declaration patch
 */
function buildGraphColumn(name: string, value: unknown, decl: ColumnDeclPatch): void {
    const column = createStagingColumn(
        name,
        "graph",
        decl.dtype === undefined ? { ...decl, dtype: inferInitialDtype(value) } : decl,
        decl.dtype === undefined,
    );
    column.write(0, value);
}

/**
 * A staging column from a typed array (setNodeColumn / setEdgeColumn, design section 12.2): the
 * dtype comes from the array class unless the patch names one (bool and dict over a Uint32Array), the
 * length must be rows * components (bool: ceil(rows / 32) words), and every row is set unless the
 * patch declares the column nullable.
 * @param domain - node or edge
 * @param rows - the row count
 * @param name - the column name
 * @param data - the values
 * @param decl - the declaration patch
 * @param bounds - the builder's node and edge bounds, for refersTo range checks (E_INDEX_RANGE)
 * @returns the staging column
 */
function stagingColumnFromTypedArray(
    domain: ColumnDomain,
    rows: number,
    name: string,
    data: TypedArrayData,
    decl: ColumnDeclPatch,
    bounds: IndexBounds,
): StagingColumn {
    const arrayDtype = dtypeOfArray(data);
    const dtype = decl.dtype ?? arrayDtype;
    const compatible = dtype === arrayDtype || (arrayDtype === "u32" && (dtype === "bool" || dtype === "dict"));
    if (!compatible) {
        throw new GraphFormatError(
            "E_COLUMN_TYPE",
            `column "${name}": a ${arrayDtype} array cannot back a ${dtype} column`,
            {
                column: name,
                field: "dtype",
                found: arrayDtype,
                expected: dtype,
            },
        );
    }
    const column = createStagingColumn(name, domain, { nullable: false, ...decl, dtype }, false);
    const { components } = column.meta;
    let expected: number;
    switch (dtype) {
        case "bool":
            expected = Math.ceil(rows / 32);
            break;
        case "dict":
            expected = rows;
            break;
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            expected = rows * components;
            break;
        default:
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `column "${name}": a ${String(dtype)} column cannot adopt a typed array`,
                {
                    column: name,
                    field: "dtype",
                },
            );
    }
    if (data.length !== expected) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `column "${name}": array length ${data.length}, expected ${expected}`,
            {
                column: name,
                expected,
                found: data.length,
            },
        );
    }
    column.ensureLength(rows);
    switch (dtype) {
        case "bool": {
            const words = data as U32;
            for (let row = 0; row < rows; row++) {
                column.write(row, ((words[row >>> 5] >>> (row & 31)) & 1) === 1);
            }
            break;
        }
        case "dict": {
            const { dictionary } = column;
            if (dictionary === null) {
                throw new GraphFormatError("E_COLUMN_TYPE", `column "${name}" has no dictionary`, { column: name });
            }
            for (let row = 0; row < rows; row++) {
                const code = data[row];
                if (code >= dictionary.size) {
                    throw new GraphFormatError(
                        "E_COLUMN_TYPE",
                        `column "${name}": code ${code} at row ${row} is outside the ${dictionary.size} declared options`,
                        {
                            column: name,
                            row,
                            code,
                        },
                    );
                }
                column.write(row, dictionary.values[code]);
            }
            break;
        }
        default: {
            const { typed } = column;
            if (typed === null) {
                throw new GraphFormatError("E_COLUMN_TYPE", `column "${name}" has no typed storage`, { column: name });
            }
            const { refersTo } = column.meta;
            const bound = refersTo === "node" ? bounds.node : bounds.edge;
            for (let i = 0; i < data.length; i++) {
                const value = data[i];
                if (refersTo !== null && value !== INVALID_INDEX && value >= bound) {
                    throw new GraphFormatError(
                        "E_INDEX_RANGE",
                        `column "${name}" refers to ${refersTo} ${value} at row ${i}, which is not below ${bound}`,
                        { column: name, row: i, refersTo, found: value, bound },
                    );
                }
                if (refersTo !== null && value === INVALID_INDEX && column.validity === null) {
                    throw new GraphFormatError(
                        "E_COLUMN_TYPE",
                        `column "${name}": row ${i} of a non-nullable refersTo column holds INVALID_INDEX`,
                        { column: name, row: i, field: "nullable" },
                    );
                }
                typed.set(i, value);
            }
            if (column.validity !== null) {
                for (let row = 0; row < rows; row++) {
                    column.validity.set(row, refersTo === null || data[row] !== INVALID_INDEX);
                }
            }
            break;
        }
    }
    return column;
}

/** The current node and edge bounds of a builder, for refersTo range checks. */
interface IndexBounds {
    /** The node index bound. */
    readonly node: number;
    /** The edge index bound. */
    readonly edge: number;
}
