/**
 * A Map-of-Maps reference model of the builder (design section 16.1): the same operations and the
 * same policies (compaction and renumbering, self-loop and duplicate policies, weight reducers,
 * explicit-weight tracking, inferred attribute columns) implemented naively with JS objects and a
 * comparator sort, so the property tests can compare every snapshot the builder freezes against an
 * independent construction. Nothing here touches typed arrays or the builder's own modules except
 * the inference grammar of src/columns/infer.ts, which is what defines the expected widened value.
 */

import { coerceValue, type InferredDtype, inferValueDtype, widenDtype } from "../../src/columns/infer.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type DuplicatePolicy, type NodeId } from "../../src/types/index.js";

/** The policies the model shares with the builder. */
interface ModelOptions {
    readonly directed: boolean;
    readonly weighted: boolean | "auto";
    readonly weightDtype: "f32" | "f64";
    readonly duplicateEdges: DuplicatePolicy;
    readonly selfLoops: "keep" | "drop" | "error";
    readonly addMissingNodes: boolean;
}

/** A thrown expectation: the error code the builder must throw for the same operation. */
export class ModelError extends Error {
    readonly code: string;

    constructor(code: string, message: string) {
        super(message);
        this.code = code;
    }
}

interface ModelNode {
    id: NodeId;
    alive: boolean;
    attrs: Map<string, unknown>;
}

interface ModelEdge {
    source: number;
    target: number;
    /** The staged weight (f32-rounded when the staging precision is f32). */
    weight: number;
    /** Whether the weight was supplied explicitly. */
    explicit: boolean;
    alive: boolean;
    attrs: Map<string, unknown>;
}

/** One inferred attribute column of the model: the widest dtype seen and the raw values. */
interface ModelColumn {
    dtype: InferredDtype;
}

/** What the model expects the snapshot of a freeze to look like. */
export interface ModelSnapshot {
    readonly directed: boolean;
    readonly nodeIds: readonly NodeId[];
    /** Live edges in final index order: declared orientation and f32 arc weight. */
    readonly edges: readonly { readonly source: number; readonly target: number; readonly weight: number }[];
    /** Whether the arc weights array exists. */
    readonly weighted: boolean;
    /** Per node: the sorted [target, edge] out-arcs (undirected: both orientations, loops once). */
    readonly rows: readonly (readonly (readonly [number, number])[])[];
    readonly selfLoopCount: number;
    readonly multigraph: boolean;
    readonly arcToEdgeIsIdentity: boolean;
    readonly nodeRemap: readonly number[] | null;
    readonly edgeRemap: readonly number[] | null;
    readonly compacted: boolean;
    readonly droppedSelfLoops: number;
    readonly mergedEdges: number;
    readonly droppedEdges: number;
    /** The role-"weight" shadow column: null when the arc array says everything. */
    readonly shadow: {
        readonly dtype: "f32" | "f64";
        readonly values: readonly number[];
        readonly validity: readonly boolean[] | null;
    } | null;
    /** Node columns: name -> dtype and per-row expected value (undefined = unset). */
    readonly nodeColumns: ReadonlyMap<string, { readonly dtype: InferredDtype; readonly values: readonly unknown[] }>;
    readonly edgeColumns: ReadonlyMap<string, { readonly dtype: InferredDtype; readonly values: readonly unknown[] }>;
}

/**
 * The reference model. Indices follow the builder's rules: first-seen node order, addEdge order,
 * tombstones until a freeze, renumbering at a freeze that dropped or merged anything.
 */
export class ModelGraph {
    readonly options: ModelOptions;
    directed: boolean;
    nodes: ModelNode[] = [];
    edges: ModelEdge[] = [];
    idIndex = new Map<NodeId, number>();
    /** Whether the weight array exists (sticky once allocated). */
    weightArray: boolean;
    /** The explicit-weight state machine of design section 3.7 ("mixed" is sticky). */
    weightMode: "none" | "explicit" | "omitted" | "mixed" = "none";
    nodeColumns = new Map<string, ModelColumn>();
    edgeColumns = new Map<string, ModelColumn>();

    constructor(options: ModelOptions) {
        this.options = options;
        this.directed = options.directed;
        this.weightArray = options.weighted === true;
    }

    get nodeCount(): number {
        return this.nodes.filter((n) => n.alive).length;
    }

    get edgeCount(): number {
        return this.edges.filter((e) => e.alive).length;
    }

    get nodeBound(): number {
        return this.nodes.length;
    }

    get edgeBound(): number {
        return this.edges.length;
    }

    private normalise(id: NodeId): NodeId {
        return typeof id === "number" && id === 0 ? 0 : id;
    }

    lookup(id: NodeId): number {
        const index = this.idIndex.get(this.normalise(id));
        if (index === undefined || !this.nodes[index].alive) {
            return INVALID_INDEX;
        }
        return index;
    }

    addNode(id: NodeId): number {
        const key = this.normalise(id);
        const existing = this.idIndex.get(key);
        if (existing !== undefined) {
            this.nodes[existing].alive = true;
            return existing;
        }
        const index = this.nodes.length;
        this.nodes.push({ id: key, alive: true, attrs: new Map() });
        this.idIndex.set(key, index);
        return index;
    }

    private endpoint(id: NodeId): number {
        const index = this.lookup(id);
        if (index !== INVALID_INDEX) {
            return index;
        }
        if (this.options.addMissingNodes) {
            return this.addNode(id);
        }
        throw new ModelError("E_UNKNOWN_NODE", `unknown node ${String(id)}`);
    }

    private staged(weight: number): number {
        return this.options.weightDtype === "f32" ? Math.fround(weight) : weight;
    }

    private trackWeight(explicit: boolean): void {
        switch (this.weightMode) {
            case "none":
                this.weightMode = explicit ? "explicit" : "omitted";
                break;
            case "explicit":
                if (!explicit) {
                    this.weightMode = "mixed";
                }
                break;
            case "omitted":
                if (explicit) {
                    this.weightMode = "mixed";
                }
                break;
            default:
                break;
        }
    }

    addEdge(source: NodeId, target: NodeId, weight?: number): number {
        if (weight !== undefined && Number.isNaN(weight)) {
            throw new ModelError("E_INVALID_WEIGHT", "NaN weight");
        }
        let explicit = weight !== undefined;
        if (this.options.weighted === false) {
            if (weight !== undefined && weight !== 1) {
                throw new ModelError("E_INVALID_WEIGHT", "unweighted builder");
            }
            explicit = false;
        }
        const u = this.endpoint(source);
        const v = this.endpoint(target);
        if (explicit) {
            this.weightArray = true;
        }
        this.trackWeight(explicit);
        const index = this.edges.length;
        this.edges.push({
            source: u,
            target: v,
            weight: explicit ? this.staged(weight as number) : 1,
            explicit,
            alive: true,
            attrs: new Map(),
        });
        return index;
    }

    removeNode(id: NodeId): number[] {
        const index = this.lookup(id);
        if (index === INVALID_INDEX) {
            throw new ModelError("E_UNKNOWN_NODE", `unknown node ${String(id)}`);
        }
        const removed: number[] = [];
        this.edges.forEach((edge, e) => {
            if (edge.alive && (edge.source === index || edge.target === index)) {
                edge.alive = false;
                removed.push(e);
            }
        });
        this.nodes[index].alive = false;
        return removed;
    }

    removeEdge(edge: number): boolean {
        const entry = this.edges[edge] as ModelEdge | undefined;
        if (entry === undefined || !entry.alive) {
            return false;
        }
        entry.alive = false;
        return true;
    }

    setEdgeWeight(edge: number, weight: number): void {
        const entry = this.edges[edge] as ModelEdge | undefined;
        if (entry === undefined || !entry.alive) {
            throw new ModelError("E_INDEX_RANGE", `edge ${edge}`);
        }
        if (Number.isNaN(weight)) {
            throw new ModelError("E_INVALID_WEIGHT", "NaN weight");
        }
        if (this.options.weighted === false) {
            if (weight !== 1) {
                throw new ModelError("E_INVALID_WEIGHT", "unweighted builder");
            }
            return;
        }
        this.weightArray = true;
        entry.weight = this.staged(weight);
        entry.explicit = true;
        if (this.weightMode === "omitted") {
            this.weightMode = "mixed";
        }
    }

    private setAttr(
        columns: Map<string, ModelColumn>,
        attrs: Map<string, unknown>,
        name: string,
        value: unknown,
    ): void {
        const observed = inferValueDtype(value);
        if (observed === null) {
            if (columns.has(name)) {
                attrs.delete(name);
            }
            return;
        }
        const column = columns.get(name);
        if (column === undefined) {
            columns.set(name, { dtype: observed });
        } else {
            column.dtype = widenDtype(column.dtype, observed) as InferredDtype;
        }
        attrs.set(name, value);
    }

    setNodeValue(index: number, name: string, value: unknown): void {
        this.setAttr(this.nodeColumns, this.nodes[index].attrs, name, value);
    }

    setEdgeValue(edge: number, name: string, value: unknown): void {
        this.setAttr(this.edgeColumns, this.edges[edge].attrs, name, value);
    }

    /**
     * Freeze: apply the policies, renumber the model itself the way a compacting freeze renumbers
     * the builder, and return the expected snapshot shape.
     */
    freeze(policyOverride?: DuplicatePolicy): ModelSnapshot {
        // A refused freeze leaves the builder untouched (design section 11.1), so it must leave the
        // model untouched too: the policies below mutate edges before a later policy can throw
        // (a self-loop is dropped before a duplicate is refused), so the mutable edge state is
        // restored when the freeze throws.
        const saved = this.edges.map((e) => ({ weight: e.weight, explicit: e.explicit, alive: e.alive }));
        const savedWeightArray = this.weightArray;
        try {
            return this.applyFreeze(policyOverride);
        } catch (err) {
            this.edges.forEach((edge, e) => {
                edge.weight = saved[e].weight;
                edge.explicit = saved[e].explicit;
                edge.alive = saved[e].alive;
            });
            this.weightArray = savedWeightArray;
            throw err;
        }
    }

    private applyFreeze(policyOverride?: DuplicatePolicy): ModelSnapshot {
        const policy = policyOverride ?? this.options.duplicateEdges;
        const oldNodeBound = this.nodes.length;
        const oldEdgeBound = this.edges.length;
        // pre-check: self-loop error
        if (this.options.selfLoops === "error" && this.edges.some((e) => e.alive && e.source === e.target)) {
            throw new ModelError("E_SELF_LOOP", "self-loop");
        }
        const tombstoned = this.edges.filter((e) => !e.alive).length;
        // dropped loops
        let droppedSelfLoops = 0;
        if (this.options.selfLoops === "drop") {
            for (const edge of this.edges) {
                if (edge.alive && edge.source === edge.target) {
                    edge.alive = false;
                    droppedSelfLoops++;
                }
            }
        }
        // duplicate policy over live edges (in the old index space; survivors keep their rows)
        let mergedEdges = 0;
        const groups = new Map<string, number[]>();
        this.edges.forEach((edge, e) => {
            if (!edge.alive) {
                return;
            }
            const key = this.groupKey(edge.source, edge.target);
            const group = groups.get(key);
            if (group === undefined) {
                groups.set(key, [e]);
            } else {
                group.push(e);
            }
        });
        const survivorOf = new Map<number, number>();
        if (policy !== "keep") {
            for (const group of groups.values()) {
                if (group.length < 2) {
                    continue;
                }
                if (policy === "error") {
                    throw new ModelError("E_DUPLICATE_EDGE", "duplicate edge");
                }
                const survivor = policy === "last" ? group[group.length - 1] : group[0];
                const target = this.edges[survivor];
                if (policy === "sum" || policy === "min" || policy === "max") {
                    // "sum" over an unweighted graph materialises multiplicities, unless the builder
                    // was declared unweighted (then no weight array ever exists)
                    if (!this.weightArray && policy === "sum" && this.options.weighted !== false) {
                        this.weightArray = true;
                    }
                    if (this.weightArray) {
                        let value = this.edges[group[0]].weight;
                        for (let i = 1; i < group.length; i++) {
                            const w = this.edges[group[i]].weight;
                            if (policy === "sum") {
                                value += w;
                            } else if (policy === "min") {
                                value = Math.min(value, w);
                            } else {
                                value = Math.max(value, w);
                            }
                        }
                        if (Number.isNaN(value)) {
                            throw new ModelError("E_INVALID_WEIGHT", "reducer produced NaN");
                        }
                        target.weight = this.staged(value);
                        // the merge stored the reduced weight: it is explicit from now on
                        target.explicit = true;
                    }
                }
                if (group.some((e) => this.edges[e].explicit)) {
                    target.explicit = true;
                }
                for (const e of group) {
                    if (e !== survivor) {
                        this.edges[e].alive = false;
                        survivorOf.set(e, survivor);
                        mergedEdges++;
                    }
                }
            }
        }
        // renumber
        const nodeRemap = new Array<number>(oldNodeBound).fill(INVALID_INDEX);
        const liveNodes: ModelNode[] = [];
        this.nodes.forEach((node, i) => {
            if (node.alive) {
                nodeRemap[i] = liveNodes.length;
                liveNodes.push(node);
            }
        });
        const edgeRemap = new Array<number>(oldEdgeBound).fill(INVALID_INDEX);
        const liveEdges: ModelEdge[] = [];
        this.edges.forEach((edge, e) => {
            if (edge.alive) {
                edgeRemap[e] = liveEdges.length;
                liveEdges.push(edge);
            }
        });
        for (const [e, survivor] of survivorOf) {
            edgeRemap[e] = edgeRemap[survivor];
        }
        for (const edge of liveEdges) {
            edge.source = nodeRemap[edge.source];
            edge.target = nodeRemap[edge.target];
        }
        this.nodes = liveNodes;
        this.edges = liveEdges;
        this.idIndex = new Map();
        liveNodes.forEach((node, i) => this.idIndex.set(node.id, i));
        const nodesRenumbered = nodeRemap.some((v, i) => v !== i);
        const edgesRenumbered = edgeRemap.some((v, i) => v !== i);
        // expected snapshot
        const rows: [number, number][][] = liveNodes.map(() => []);
        let selfLoopCount = 0;
        liveEdges.forEach((edge, e) => {
            rows[edge.source].push([edge.target, e]);
            if (edge.source === edge.target) {
                selfLoopCount++;
            } else if (!this.directed) {
                rows[edge.target].push([edge.source, e]);
            }
        });
        let multigraph = false;
        for (const row of rows) {
            row.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
            for (let i = 1; i < row.length; i++) {
                if (row[i][0] === row[i - 1][0]) {
                    multigraph = true;
                }
            }
        }
        let identity = this.directed;
        for (let e = 1; e < liveEdges.length && identity; e++) {
            const a = liveEdges[e - 1];
            const b = liveEdges[e];
            if (b.source < a.source || (b.source === a.source && b.target < a.target)) {
                identity = false;
            }
        }
        const needValidity = this.weightMode === "mixed" || this.weightMode === "omitted";
        const inexact = this.options.weightDtype === "f64" && liveEdges.some((e) => Math.fround(e.weight) !== e.weight);
        let shadow: ModelSnapshot["shadow"] = null;
        if (this.weightArray && liveEdges.length > 0 && (needValidity || inexact)) {
            shadow = {
                dtype: inexact ? "f64" : "f32",
                values: liveEdges.map((e) => (inexact ? e.weight : Math.fround(e.weight))),
                validity: needValidity ? liveEdges.map((e) => e.explicit) : null,
            };
        }
        return {
            directed: this.directed,
            nodeIds: liveNodes.map((n) => n.id),
            edges: liveEdges.map((e) => ({ source: e.source, target: e.target, weight: Math.fround(e.weight) })),
            weighted: this.weightArray,
            rows,
            selfLoopCount,
            multigraph,
            arcToEdgeIsIdentity: identity,
            nodeRemap: nodesRenumbered ? nodeRemap : null,
            edgeRemap: edgesRenumbered ? edgeRemap : null,
            compacted: nodesRenumbered || edgesRenumbered,
            droppedSelfLoops,
            mergedEdges,
            droppedEdges: tombstoned + droppedSelfLoops + mergedEdges,
            shadow,
            nodeColumns: this.expectedColumns(this.nodeColumns, liveNodes),
            edgeColumns: this.expectedColumns(this.edgeColumns, liveEdges),
        };
    }

    private groupKey(u: number, v: number): string {
        if (this.directed || u <= v) {
            return `${u}->${v}`;
        }
        return `${v}->${u}`;
    }

    private expectedColumns(
        columns: Map<string, ModelColumn>,
        rows: readonly { readonly attrs: Map<string, unknown> }[],
    ): Map<string, { readonly dtype: InferredDtype; readonly values: readonly unknown[] }> {
        const out = new Map<string, { readonly dtype: InferredDtype; readonly values: readonly unknown[] }>();
        for (const [name, column] of columns) {
            const values = rows.map((row) => {
                const raw = row.attrs.get(name);
                return raw === undefined || raw === null ? undefined : coerceValue(raw, column.dtype);
            });
            out.set(name, { dtype: column.dtype, values });
        }
        return out;
    }
}
