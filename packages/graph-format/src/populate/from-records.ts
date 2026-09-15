/**
 * `fromRecords()` (design section 8.1, entry point 3): plain node-link records, the shape every JSON
 * dialect parses into and what graphty-element's data sources emit. Every record is pushed into a
 * GraphBuilder scalar by scalar -- the id, the endpoints and the weight through `addNode` / `addEdge`,
 * every other key through `setNodeValue` / `setEdgeValue` -- so attribute columns are inferred and
 * widened per column by the builder exactly as `addNodeRecord` / `addEdgeRecord` would (design
 * section 5.1), and the FreezeReport carries the widenings. The reserved keys (the node id, the edge
 * source / target and the weight) never become attribute columns: the id map and the arc array
 * already hold them.
 *
 * Id coercion (design section 4.1) is applied here, before an id reaches the builder: "keep" (the
 * default; JSON values are already typed), "canonical" (canonical integer text becomes a number,
 * everything else stays a string), "string" and "number".
 */

import { GraphBuilder } from "../builder/graph-builder.js";
import { GraphFormatError } from "../errors.js";
import { type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import {
    type BuilderOptionsPatch,
    type ColumnDecl,
    type ColumnHandle,
    type FreezeOptions,
    type FreezeReport,
    type GraphBuilderOptions,
    type IdCoercion,
    type NodeId,
    type RecordsInput,
} from "../types/index.js";
import { assertOneOf } from "../util/options.js";
import { splitOptions } from "./from-edge-arrays.js";

/** A node or edge record. */
type RecordRow = Readonly<Record<string, unknown>>;

/** The default source keys, tried in order when `edgeSource` is not given. */
const SOURCE_KEYS: readonly string[] = ["source", "src", "from"];

/** The default target keys, tried in order when `edgeTarget` is not given. */
const TARGET_KEYS: readonly string[] = ["target", "dst", "to"];

/** The canonical integer text of design section 4.1. */
const CANONICAL_INTEGER = /^-?(0|[1-9][0-9]*)$/;

// ============================================================ ids

/**
 * Whether a record has an own property (records may come from `JSON.parse`, so prototype lookups are
 * never consulted).
 * @param record - the record
 * @param key - the key
 * @returns true when the key is an own property
 */
function hasKey(record: RecordRow, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(record, key);
}

/**
 * The E_INVALID_ID error for an id cell the coercion rule cannot turn into a NodeId.
 * @param field - which cell (id / source / target)
 * @param value - the value
 * @param rule - the coercion rule in force
 * @returns the error
 */
function invalidId(field: string, value: unknown, rule: IdCoercion): GraphFormatError {
    return new GraphFormatError(
        "E_INVALID_ID",
        `${field} value of type ${typeof value} is not a legal id under "${rule}"`,
        {
            field,
            found: typeof value,
            rule,
        },
    );
}

/**
 * Apply an id coercion rule to one id cell (design section 4.1). The result still goes through the
 * builder's own id validation (E_INVALID_ID for anything that is not a finite number or a well-formed
 * string), so this function only converts; it rejects the shapes a rule cannot convert.
 * @param value - the cell value
 * @param rule - the coercion rule
 * @param field - which cell, for the error message
 * @returns the id to hand to the builder
 */
export function coerceId(value: unknown, rule: IdCoercion, field: string): NodeId {
    switch (rule) {
        case "keep":
            if (typeof value === "string" || typeof value === "number") {
                return value;
            }
            throw invalidId(field, value, rule);
        case "canonical":
            if (typeof value === "number") {
                return value;
            }
            if (typeof value === "string") {
                // "-0" is excluded so the rule stays injective on text (-0 and 0 are one id)
                if (CANONICAL_INTEGER.test(value) && value !== "-0") {
                    const n = Number(value);
                    return Number.isSafeInteger(n) ? n : value;
                }
                return value;
            }
            throw invalidId(field, value, rule);
        case "string":
            if (typeof value === "string") {
                return value;
            }
            if (
                typeof value === "number" ||
                typeof value === "boolean" ||
                typeof value === "bigint" ||
                value === null
            ) {
                return String(value);
            }
            throw invalidId(field, value, rule);
        case "number":
            if (typeof value === "number") {
                return value;
            }
            if (typeof value === "string") {
                return Number(value);
            }
            throw invalidId(field, value, rule);
        default: {
            const name: string = rule;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id coercion "${name}"`, {
                field: "ids",
                found: name,
                reason: "unsupported option",
            });
        }
    }
}

/**
 * Read an endpoint cell: the explicit key when one was given, else the first of the default keys the
 * record has.
 * @param record - the edge record
 * @param explicit - the caller's key, or undefined
 * @param defaults - the default keys in order
 * @returns the key that was used and its value; the key is null when none is present
 */
function endpointOf(
    record: RecordRow,
    explicit: string | undefined,
    defaults: readonly string[],
): { readonly key: string | null; readonly value: unknown } {
    if (explicit !== undefined) {
        return { key: hasKey(record, explicit) ? explicit : null, value: record[explicit] };
    }
    for (const key of defaults) {
        if (hasKey(record, key)) {
            return { key, value: record[key] };
        }
    }
    return { key: null, value: undefined };
}

/**
 * The E_INVALID_ID error for an edge record without an endpoint.
 * @param field - "source" or "target"
 * @param keys - the keys that were tried
 * @param edge - the record's position in the edge iteration
 * @returns the error
 */
function missingEndpoint(field: string, keys: readonly string[], edge: number): GraphFormatError {
    return new GraphFormatError(
        "E_INVALID_ID",
        `edge record ${edge} has no ${field} (keys tried: ${keys.join(", ")})`,
        {
            field,
            keys: [...keys],
            edge,
            reason: "missing endpoint",
        },
    );
}

// ============================================================ attribute policy

/**
 * How non-reserved record keys become columns: the builder's inference, one json column per key, no
 * columns at all, or only the declared columns.
 */
class AttributePolicy {
    private readonly mode: "infer" | "json" | "none" | "declared";

    private readonly declared: ReadonlyMap<string, ColumnDecl>;

    private readonly handles: { readonly node: Map<string, ColumnHandle>; readonly edge: Map<string, ColumnHandle> };

    /**
     * Resolve the `columns` option.
     * @param columns - the option value; default "infer"
     */
    constructor(columns: RecordsInput["columns"]) {
        this.handles = { node: new Map(), edge: new Map() };
        if (columns === undefined || columns === "infer") {
            this.mode = "infer";
            this.declared = new Map();
            return;
        }
        if (columns === "json" || columns === "none") {
            this.mode = columns;
            this.declared = new Map();
            return;
        }
        if (typeof columns === "string") {
            // unreachable for typed callers; an unknown mode string from untyped JS lands here
            const found: string = columns;
            throw new GraphFormatError("E_UNSUPPORTED", `unsupported columns option "${found}"`, {
                field: "columns",
                found,
                reason: "unsupported option",
            });
        }
        const declared = new Map<string, ColumnDecl>();
        const list: readonly ColumnDecl[] = columns;
        for (const decl of list) {
            if (typeof decl.name !== "string" || decl.name.length === 0) {
                throw new GraphFormatError("E_COLUMN_TYPE", "a column declaration needs a non-empty name", {
                    field: "name",
                    found: decl.name,
                });
            }
            if (declared.has(decl.name)) {
                throw new GraphFormatError("E_COLUMN_EXISTS", `column "${decl.name}" is declared twice`, {
                    column: decl.name,
                });
            }
            declared.set(decl.name, decl);
        }
        this.mode = "declared";
        this.declared = declared;
    }

    /**
     * Whether any attribute key is read at all.
     * @returns false under "none"
     */
    get readsAttributes(): boolean {
        return this.mode !== "none";
    }

    /**
     * Write one attribute cell according to the policy.
     * @param builder - the builder
     * @param domain - "node" or "edge"
     * @param row - the node index or logical edge index
     * @param key - the record key
     * @param value - the cell value
     */
    write(builder: GraphBuilder, domain: "node" | "edge", row: number, key: string, value: unknown): void {
        switch (this.mode) {
            case "none":
                return;
            case "infer":
                if (domain === "node") {
                    builder.setNodeValue(key, row, value);
                } else {
                    builder.setEdgeValue(key, row, value);
                }
                return;
            case "json":
            case "declared": {
                const handle = this.handleOf(builder, domain, key);
                if (handle === null) {
                    return;
                }
                if (domain === "node") {
                    builder.setNodeValue(handle, row, value);
                } else {
                    builder.setEdgeValue(handle, row, value);
                }
                return;
            }
            default: {
                const name: string = this.mode;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown attribute mode ${name}`, { found: name });
            }
        }
    }

    /**
     * The handle of a key's column in one table, declaring it on first use: a json column under
     * "json", the caller's declaration under "declared" (an undeclared key has no column).
     * @param builder - the builder
     * @param domain - "node" or "edge"
     * @param key - the record key
     * @returns the handle, or null when the key is not stored
     */
    private handleOf(builder: GraphBuilder, domain: "node" | "edge", key: string): ColumnHandle | null {
        const cache = this.handles[domain];
        const cached = cache.get(key);
        if (cached !== undefined) {
            return cached;
        }
        let decl: ColumnDecl;
        if (this.mode === "json") {
            decl = { name: key, dtype: "json" };
        } else {
            const declared = this.declared.get(key);
            if (declared === undefined) {
                return null;
            }
            decl = declared;
        }
        const handle = domain === "node" ? builder.declareNodeColumn(decl) : builder.declareEdgeColumn(decl);
        cache.set(key, handle);
        return handle;
    }
}

// ============================================================ the entry point

/**
 * The resolved record keys and rules of one fromRecords call.
 */
interface RecordRules {
    /** The node id key, or null for positional nodes (d3 v3). */
    readonly nodeId: string | null;
    /** The explicit source key, or undefined for the default keys. */
    readonly edgeSource: string | undefined;
    /** The explicit target key, or undefined for the default keys. */
    readonly edgeTarget: string | undefined;
    /** The weight key, or null for unweighted. */
    readonly edgeWeight: string | null;
    /** The id coercion rule. */
    readonly ids: IdCoercion;
    /** The attribute policy. */
    readonly attributes: AttributePolicy;
}

/**
 * Resolve the keys and rules of a RecordsInput with their defaults.
 * @param input - the input
 * @returns the rules
 */
function rulesOf(input: RecordsInput): RecordRules {
    return {
        nodeId: input.nodeId === undefined ? "id" : input.nodeId,
        edgeSource: input.edgeSource,
        edgeTarget: input.edgeTarget,
        edgeWeight: input.edgeWeight === undefined ? "weight" : input.edgeWeight,
        ids: assertOneOf("ids", input.ids, ["keep", "canonical", "string", "number"] as const) ?? "keep",
        attributes: new AttributePolicy(input.columns),
    };
}

/**
 * Resolve a positional endpoint (nodeId null: endpoints are node indices, design section 8.1): a
 * non-negative integer below the builder's node bound, or -- under addMissingNodes -- one that
 * extends the anonymous node range.
 * @param builder - the builder
 * @param value - the cell value
 * @param field - "source" or "target"
 * @param edge - the record's position, for the error
 * @returns the node index
 */
function positionalEndpoint(builder: GraphBuilder, value: unknown, field: string, edge: number): number {
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
        throw new GraphFormatError(
            "E_INVALID_ID",
            `edge record ${edge}: ${field} ${String(value)} is not a node index`,
            {
                field,
                found: value,
                edge,
                reason: "not an index",
            },
        );
    }
    const bound = builder.nodeBound;
    if (value >= bound) {
        if (!builder.options.addMissingNodes) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `edge record ${edge}: node index ${value} does not exist`, {
                index: value,
                field,
                edge,
            });
        }
        builder.addAnonymousNodes(value - bound + 1);
    }
    return value;
}

/**
 * Read the weight cell of an edge record: absent, undefined and null mean "no weight" (1 in a
 * weighted graph, design section 3.7); anything else must be a number.
 * @param record - the edge record
 * @param key - the weight key
 * @param edge - the record's position, for the error
 * @returns the weight, or undefined when omitted
 */
function weightOf(record: RecordRow, key: string, edge: number): number | undefined {
    if (!hasKey(record, key)) {
        return undefined;
    }
    const raw = record[key];
    if (raw === undefined || raw === null) {
        return undefined;
    }
    if (typeof raw !== "number") {
        throw new GraphFormatError("E_INVALID_WEIGHT", `edge record ${edge}: weight "${key}" is not a number`, {
            key,
            found: typeof raw,
            edge,
        });
    }
    return raw;
}

/**
 * Push the node records.
 * @param builder - the builder
 * @param rules - the resolved rules
 * @param nodes - the node records
 */
function pushNodes(builder: GraphBuilder, rules: RecordRules, nodes: Iterable<RecordRow>): void {
    const { nodeId, attributes } = rules;
    let position = 0;
    for (const record of nodes) {
        let index: number;
        if (nodeId === null) {
            index = builder.addAnonymousNodes(1);
        } else {
            if (!hasKey(record, nodeId)) {
                throw new GraphFormatError("E_INVALID_ID", `node record ${position} has no "${nodeId}" key`, {
                    field: nodeId,
                    node: position,
                    reason: "missing id",
                });
            }
            index = builder.addNode(coerceId(record[nodeId], rules.ids, nodeId));
        }
        if (attributes.readsAttributes) {
            for (const key of Object.keys(record)) {
                if (key !== nodeId) {
                    attributes.write(builder, "node", index, key, record[key]);
                }
            }
        }
        position++;
    }
}

/**
 * Push the edge records.
 * @param builder - the builder
 * @param rules - the resolved rules
 * @param edges - the edge records
 */
function pushEdges(builder: GraphBuilder, rules: RecordRules, edges: Iterable<RecordRow>): void {
    const { nodeId, edgeWeight, attributes } = rules;
    let position = 0;
    for (const record of edges) {
        const source = endpointOf(record, rules.edgeSource, SOURCE_KEYS);
        const target = endpointOf(record, rules.edgeTarget, TARGET_KEYS);
        if (source.key === null) {
            throw missingEndpoint(
                "source",
                rules.edgeSource === undefined ? SOURCE_KEYS : [rules.edgeSource],
                position,
            );
        }
        if (target.key === null) {
            throw missingEndpoint(
                "target",
                rules.edgeTarget === undefined ? TARGET_KEYS : [rules.edgeTarget],
                position,
            );
        }
        const weight = edgeWeight === null ? undefined : weightOf(record, edgeWeight, position);
        let edge: number;
        if (nodeId === null) {
            const u = positionalEndpoint(builder, source.value, "source", position);
            const v = positionalEndpoint(builder, target.value, "target", position);
            edge = builder.addEdgeByIndex(u, v, weight);
        } else {
            edge = builder.addEdge(
                coerceId(source.value, rules.ids, "source"),
                coerceId(target.value, rules.ids, "target"),
                weight,
            );
        }
        if (attributes.readsAttributes) {
            for (const key of Object.keys(record)) {
                if (key !== source.key && key !== target.key && key !== edgeWeight) {
                    attributes.write(builder, "edge", edge, key, record[key]);
                }
            }
        }
        position++;
    }
}

/**
 * Build a snapshot from node-link records (design section 8.1): node records (optional; every node
 * an edge names is created under addMissingNodes, the default) and edge records, pushed into a
 * GraphBuilder one scalar at a time and frozen. The node id comes from `nodeId` ("id" by default;
 * null makes the node index the record position and the endpoints indices, the d3 v3 shape), the
 * endpoints from `edgeSource` / `edgeTarget` ("source" / "target" by default, falling back to "src"
 * / "from" and "dst" / "to"), the weight from `edgeWeight` ("weight" by default; null = unweighted).
 * Every other key becomes an attribute column per `columns`: "infer" (default) lets the builder
 * infer and widen per column (design section 5.1), "json" stores each key as a json column, "none"
 * loads only the structure, and a ColumnDecl list stores only the declared keys. Ids are coerced per
 * `ids` ("keep" by default). The staging weight precision defaults to f64 so a JSON weight such as
 * 0.1 is never corrupted (the shadow column costs nothing when every value is f32-exact); pass
 * `weightDtype: "f32"` to opt out.
 * @param input - the records and keys; a record without its id or an endpoint, or with an id the
 *   coercion rule cannot convert, is E_INVALID_ID; a non-numeric weight is E_INVALID_WEIGHT; an
 *   unknown endpoint under addMissingNodes false is E_UNKNOWN_NODE
 * @param options - builder policies and freeze options
 * @returns the snapshot and the freeze report (its `widened` entries name the inferred columns that
 *   widened while the records were read)
 */
export function fromRecords(
    input: RecordsInput,
    options: BuilderOptionsPatch & FreezeOptions = {},
): { snapshot: GraphSnapshot; report: FreezeReport } {
    const rules = rulesOf(input);
    const split = splitOptions(input.directed, options);
    const builderOptions: GraphBuilderOptions = {
        ...split.builder,
        weightDtype: options.weightDtype ?? "f64",
    };
    const builder = new GraphBuilder(builderOptions);
    if (input.nodes !== undefined) {
        pushNodes(builder, rules, input.nodes);
    }
    pushEdges(builder, rules, input.edges);
    return builder.freezeWithReport(split.freeze);
}
