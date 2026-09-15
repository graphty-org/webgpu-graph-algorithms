/**
 * The JSON exporter (design section 8.5): one GraphExporter with a `dialect` option writing
 * NetworkX node-link (the default), d3 (`links`, optional index endpoints), JSON Graph Format v2,
 * Cytoscape.js elements, graphology serialisation or vis.js. The dialect defaults to the one the
 * importer recorded under `meta.extra.json` so a JSON file re-exported as JSON keeps its shape,
 * and to node-link for a snapshot from any other source.
 *
 * check() runs the generic capability pre-flight against the dialect's table and appends the
 * dialect-specific notes (non-finite numbers written as null, JGF id stringification / collisions
 * / node order, direction dropped by Cytoscape and vis, mutual pairs expanded, reserved key
 * collisions, positions written as plain attributes, a Cytoscape `parents` column) before anything
 * is written; the same column plan drives write(), so the notes and the output agree.
 *
 * Every column without a structural slot in the dialect is written as a plain attribute under its
 * column name; an f32 value is written as the shortest decimal that round-trips through
 * Math.fround (design section 3.7); a multi-component value is an array; a list is an array; a json
 * value is its JSON text; an unset row writes no key (design section 5.3). Non-finite numbers,
 * which JSON cannot carry, become null and are reported.
 */

import { type Column, GraphFormatError, type GraphSnapshot, INVALID_INDEX, type NodeId } from "@graphty/graph-format";

import { type PairFolding, pairFolding } from "../../common/direction.js";
import { checkCapabilities, countMixedEdges, LOSS } from "../../common/export.js";
import { formatF32 } from "../../common/format.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import {
    CYTOSCAPE_ELEMENT_KEYS,
    DIALECT_DEFAULT_DIRECTED,
    dialectCapabilities,
    isJsonDialect,
    JSON_DIALECTS,
    type JsonDialect,
    type JsonShapeMeta,
    shapeMetaOf,
    SUFFIX,
} from "./dialect.js";

/** The format-specific options of the JSON exporter. */
export interface JsonExportOptions {
    /** The dialect to write; default: the dialect the importer recorded, else "node-link". */
    dialect?: JsonDialect | undefined;
    /** Spaces per indentation level; 0 (default) writes compact JSON. */
    indent?: number | undefined;
    /** node-link / d3: the key of the edge array; default: the recorded key, else "edges" (d3: "links"). */
    edgesKey?: string | undefined;
    /** node-link / d3 / vis: the node id key; default: the recorded key, else "id". */
    nodeIdKey?: string | undefined;
    /** node-link / d3: write endpoints as node array positions; default: the recorded flag, else false. */
    indexLinks?: boolean | undefined;
    /** node-link / d3 / vis: the source key; default: the recorded key, else "source" (vis: "from"). */
    sourceKey?: string | undefined;
    /** node-link / d3 / vis: the target key; default: the recorded key, else "target" (vis: "to"). */
    targetKey?: string | undefined;
    /** The key the weight is written under; default: the key the JSON importer read it from, else "weight". */
    weightKey?: string | undefined;
}

/**
 * The LossNote codes of the JSON exporter's check(): the dialect-specific ones and, aliased, the
 * shared ones it records (`LOSS` holds the rest of the generic pre-flight's codes). A key is the
 * code without its severity prefix.
 */
export const JSON_LOSS = Object.freeze({
    /** Non-finite numbers (columns, weights) are written as null. */
    NONFINITE_AS_NULL: "W_NONFINITE_AS_NULL",
    /** Cytoscape, vis and d3 carry no direction; the file re-imports with the dialect's default direction. */
    DIRECTION_DROPPED: "W_DIRECTION_DROPPED",
    /** GEXF mutual pairs are written as two directed edges. */
    MUTUAL_EXPANDED: LOSS.MUTUAL_EXPANDED,
    /** JGF keys its nodes by string; numeric ids re-import as text unless ids: "canonical". */
    NUMERIC_IDS_STRINGIFIED: "W_NUMERIC_IDS_STRINGIFIED",
    /** JGF: two ids have the same text; export() throws E_INVALID_ID. */
    ID_TEXT_COLLISION: LOSS.ID_TEXT_COLLISION,
    /** JGF: integer-like id keys are enumerated first and ascending by JSON parsers; node order changes on re-import. */
    NODE_ORDER: "W_NODE_ORDER",
    /** A column named like a reserved key of the dialect (id, source, target, ...) is skipped. */
    RESERVED_KEY: "W_RESERVED_KEY",
    /** A plain edge column named like the weight key reads back as THE weight (or is skipped when weights are written). */
    WEIGHT_KEY_CLASH: LOSS.WEIGHT_KEY_CLASH,
    /** A position column without a slot (every dialect but Cytoscape) is a plain array attribute; the role is lost. */
    POSITIONS_DROPPED: LOSS.POSITIONS,
    /** Cytoscape positions are 2D; non-zero z values are dropped. */
    POSITION_Z_DROPPED: "W_POSITION_Z_DROPPED",
    /** Cytoscape has a single parent; a `parents` list column cannot be written. */
    PARENTS_DROPPED: LOSS.PARENTS,
    /** node-link / d3 have no edge id slot; the id column is written as a plain attribute. */
    EDGE_IDS_DROPPED: LOSS.EDGE_IDS_DROPPED,
    /** An f64 column of integral values reads back as i32 (JSON declares no types). */
    INTEGRAL_F64_AS_I32: LOSS.INTEGRAL_F64,
    /** A role column the dialect has no slot for is a plain attribute; the role is lost. */
    ROLE_DROPPED: LOSS.ROLE,
    /** A column without a set cell is not written (JSON declares no columns). */
    EMPTY_COLUMN_DROPPED: LOSS.EMPTY_COLUMN,
});

/** The roles each dialect has a slot for (every other role column is a plain attribute, reported by checkCapabilities()). */
const SLOT_ROLES: Readonly<Record<JsonDialect, ReadonlySet<string>>> = Object.freeze({
    "node-link": new Set<string>(),
    d3: new Set<string>(),
    jgf: new Set(["label", "kind"]),
    cytoscape: new Set(["classes"]),
    graphology: new Set<string>(),
    vis: new Set<string>(),
});

/** The element domains. */
type Domain = "node" | "edge" | "graph";

/** A note-recording callback. */
type NoteFn = (code: string, message: string, column?: string | null, count?: number | null) => void;

/** The roles the pre-flight treats structurally; never written as attributes. */
const STRUCTURAL_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "originalId",
]);

/** The roles no JSON dialect can carry; skipped (checkCapabilities reports them). */
const TEMPORAL_ROLES: ReadonlySet<string> = new Set(["start", "end", "timestamp", "timestamps", "spells", "open"]);

/** The dialects whose attributes live in a nested dict (data / metadata / attributes). */
const NESTED_DIALECTS: ReadonlySet<JsonDialect> = new Set(["jgf", "cytoscape", "graphology"]);

/** A mutable counter of the non-finite numbers written as null. */
interface Counter {
    count: number;
}

/** How one column is written. */
type Slot = "attribute" | "element" | "position" | "cytoscapePosition" | "parent" | "classes" | "label" | "relation";

/** One column's place in the output. */
interface ColumnPlan {
    readonly column: Column;
    /** The output key (attribute or element-level key). */
    readonly key: string;
    readonly slot: Slot;
}

/** The resolved exporter options. */
interface Resolved {
    readonly common: ResolvedExportOptions;
    readonly dialect: JsonDialect;
    readonly indent: number;
    readonly edgesKey: string;
    readonly nodeIdKey: string | null;
    readonly indexLinks: boolean;
    readonly sourceKey: string;
    readonly targetKey: string;
    readonly weightKey: string;
    readonly shape: JsonShapeMeta;
}

/** Everything write() needs, computed once by plan(). */
interface Plan {
    readonly resolved: Resolved;
    readonly notes: LossNote[];
    readonly nodes: readonly ColumnPlan[];
    readonly edges: readonly ColumnPlan[];
    readonly graph: readonly ColumnPlan[];
    /** The id-role edge column, or null. */
    readonly edgeIds: Column | null;
    /** The weight of an edge as JSON text, or null when the edge has no explicit weight. */
    readonly weights: (e: number) => string | null;
    /** The pair folding (design section 3.6): mirrors skipped, source directions. */
    readonly folding: PairFolding;
    /** The file-level direction. */
    readonly directed: boolean;
    /** Whether the file declares a multigraph. */
    readonly multigraph: boolean;
}

/**
 * Resolve the options against the snapshot's recorded shape.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for a bad value
 */
function resolve(snapshot: GraphSnapshot, options: (JsonExportOptions & CommonExportOptions) | undefined): Resolved {
    const o = options ?? {};
    const common = resolveExportOptions(options);
    const shape = shapeMetaOf(snapshot.meta);
    let dialect: JsonDialect;
    if (o.dialect === undefined) {
        dialect = shape.dialect ?? "node-link";
    } else if (isJsonDialect(o.dialect)) {
        ({ dialect } = o);
    } else {
        throw unsupported("dialect", o.dialect, JSON_DIALECTS);
    }
    const indent = o.indent ?? 0;
    if (!Number.isInteger(indent) || indent < 0 || indent > 16) {
        throw unsupported("indent", indent, ["an integer 0..16"]);
    }
    const sameDialect = shape.dialect === dialect;
    const recorded = <T>(value: T | undefined, fallback: T): T =>
        sameDialect && value !== undefined ? value : fallback;
    const edgesKey =
        keyOption("edgesKey", o.edgesKey) ?? recorded(shape.edgesKey, dialect === "d3" ? "links" : "edges");
    let nodeIdKey: string | null;
    if (o.nodeIdKey !== undefined) {
        nodeIdKey = keyOption("nodeIdKey", o.nodeIdKey);
    } else if (sameDialect && shape.nodeIdKey !== undefined) {
        ({ nodeIdKey } = shape);
    } else {
        nodeIdKey = "id";
    }
    let indexLinks = o.indexLinks ?? recorded(shape.indexLinks, false);
    if (typeof indexLinks !== "boolean") {
        throw unsupported("indexLinks", indexLinks, ["true", "false"]);
    }
    if (nodeIdKey === null) {
        // positional nodes have no id to write; endpoints must be positions
        indexLinks = true;
    }
    const sourceKey =
        keyOption("sourceKey", o.sourceKey) ?? recorded(shape.sourceKey, dialect === "vis" ? "from" : "source");
    const targetKey =
        keyOption("targetKey", o.targetKey) ?? recorded(shape.targetKey, dialect === "vis" ? "to" : "target");
    const { weightOrigin } = snapshot.meta;
    const weightKey =
        keyOption("weightKey", o.weightKey) ??
        (weightOrigin !== null && weightOrigin.format === "json" && weightOrigin.id !== null
            ? weightOrigin.id
            : "weight");
    return { common, dialect, indent, edgesKey, nodeIdKey, indexLinks, sourceKey, targetKey, weightKey, shape };
}

/**
 * Check a key-valued option.
 * @param name - the option name
 * @param value - the caller's value
 * @returns the key, or null when absent
 */
function keyOption(name: string, value: unknown): string | null {
    if (value === undefined) {
        return null;
    }
    if (typeof value !== "string" || value.length === 0) {
        throw unsupported(name, value, ["a non-empty key"]);
    }
    return value;
}

/**
 * The E_UNSUPPORTED error of a bad option.
 * @param name - the option name
 * @param found - the value
 * @param supported - what is accepted
 * @returns the error
 */
function unsupported(name: string, found: unknown, supported: readonly string[]): GraphFormatError {
    const shown = typeof found === "string" ? JSON.stringify(found) : String(found);
    return new GraphFormatError("E_UNSUPPORTED", `option ${name}: ${shown} is not one of ${supported.join(", ")}`, {
        option: name,
        found: typeof found === "string" || typeof found === "number" ? found : typeof found,
        supported: [...supported],
    });
}

// ============================================================ value formatting

/**
 * The JSON text of a finite or non-finite number; non-finite values become null and are counted.
 * @param value - the number
 * @param f32 - whether the value came from an f32 store (shortest fround-round-trip text)
 * @param nonfinite - the counter of null-ed values
 * @returns the JSON text
 */
function numberText(value: number, f32: boolean, nonfinite: Counter): string {
    if (!Number.isFinite(value)) {
        nonfinite.count++;
        return "null";
    }
    if (f32) {
        return formatF32(value);
    }
    return Object.is(value, -0) ? "0" : String(value);
}

/**
 * The JSON text of any cell value: a number per its dtype, a boolean, a string, an array (a
 * multi-component subarray or a list row), or a json value.
 * @param value - the value
 * @param f32 - whether numbers come from an f32 store
 * @param nonfinite - the counter of null-ed values
 * @returns the JSON text
 */
function valueText(value: unknown, f32: boolean, nonfinite: Counter): string {
    switch (typeof value) {
        case "number":
            return numberText(value, f32, nonfinite);
        case "boolean":
            return value ? "true" : "false";
        case "string":
            return JSON.stringify(value);
        case "undefined":
            return "null";
        default:
            break;
    }
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
        const items = Array.from(value as ArrayLike<unknown>, (item) => valueText(item, f32, nonfinite));
        return `[${items.join(",")}]`;
    }
    const text = JSON.stringify(value);
    return text === undefined ? "null" : text;
}

/**
 * The JSON text of one set cell.
 * @param column - the column
 * @param row - the row
 * @param ids - the snapshot's id map (for refersTo node columns)
 * @param nonfinite - the counter of null-ed values
 * @returns the JSON text
 */
function cellText(column: Column, row: number, ids: GraphSnapshot["ids"], nonfinite: Counter): string {
    switch (column.dtype) {
        case "list":
            return valueText(column.sliceOf(row), column.child.dtype === "f32", nonfinite);
        case "json":
            return valueText(column.values[row], false, nonfinite);
        case "u32":
            if (column.meta.refersTo === "node" && column.meta.components === 1) {
                const index = column.data[row];
                return index === INVALID_INDEX ? "null" : JSON.stringify(ids.idOf(index));
            }
            return valueText(column.value(row), false, nonfinite);
        case "f32":
            return valueText(column.value(row), true, nonfinite);
        default:
            return valueText(column.value(row), false, nonfinite);
    }
}

/**
 * How many cells of a column hold a non-finite number (f32 / f64 columns, their components and
 * lists of them).
 * @param column - the column
 * @returns the count
 */
function countNonFinite(column: Column): number {
    let data: ArrayLike<number>;
    switch (column.dtype) {
        case "f32":
        case "f64":
            ({ data } = column);
            break;
        case "list":
            if (column.child.dtype !== "f32" && column.child.dtype !== "f64") {
                return 0;
            }
            ({ data } = column.child);
            break;
        default:
            return 0;
    }
    let count = 0;
    for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
            count++;
        }
    }
    return count;
}

// ============================================================ the writer

/**
 * Emits JSON tokens with optional indentation, accumulating text parts the generator yields one
 * element at a time.
 */
class JsonWriter {
    private parts: string[] = [];

    private depth = 0;

    private readonly firstAtDepth: boolean[] = [];

    private readonly indent: number;

    /**
     * Create a writer.
     * @param indent - spaces per level; 0 for compact output
     */
    constructor(indent: number) {
        this.indent = indent;
    }

    /**
     * Open an object or an array.
     * @param bracket - "{" or "["
     */
    open(bracket: "{" | "["): void {
        this.parts.push(bracket);
        this.depth++;
        this.firstAtDepth.push(true);
    }

    /**
     * Close the innermost object or array.
     * @param bracket - "}" or "]"
     */
    close(bracket: "}" | "]"): void {
        const wasEmpty = this.firstAtDepth.pop() ?? true;
        this.depth--;
        if (!wasEmpty) {
            this.parts.push(this.newline());
        }
        this.parts.push(bracket);
    }

    /**
     * Start an object member: the separator and the quoted key.
     * @param name - the key
     */
    key(name: string): void {
        this.separator();
        this.parts.push(JSON.stringify(name), this.indent > 0 ? ": " : ":");
    }

    /** Start an array element: the separator only. */
    item(): void {
        this.separator();
    }

    /**
     * Append raw JSON text (a value).
     * @param text - the text
     */
    raw(text: string): void {
        this.parts.push(text);
    }

    /**
     * Write a member with a value in one call.
     * @param name - the key
     * @param text - the value's JSON text
     */
    member(name: string, text: string): void {
        this.key(name);
        this.parts.push(text);
    }

    /**
     * Hand out the text accumulated since the last take.
     * @returns the text
     */
    take(): string {
        const text = this.parts.length === 1 ? this.parts[0] : this.parts.join("");
        this.parts = [];
        return text;
    }

    /** The comma and newline before a member or element. */
    private separator(): void {
        const top = this.firstAtDepth.length - 1;
        if (this.firstAtDepth[top]) {
            this.firstAtDepth[top] = false;
        } else {
            this.parts.push(",");
        }
        this.parts.push(this.newline());
    }

    /**
     * A newline plus the current indentation, or nothing in compact mode.
     * @returns the text
     */
    private newline(): string {
        return this.indent > 0 ? `\n${" ".repeat(this.depth * this.indent)}` : "";
    }
}

// ============================================================ planning

/** What plan() and its helpers share. */
interface PlanContext {
    readonly snapshot: GraphSnapshot;
    readonly resolved: Resolved;
    readonly dialect: JsonDialect;
    readonly caps: ExportCapabilities;
    readonly notes: LossNote[];
    readonly note: NoteFn;
    /** The reserved output keys per table. */
    readonly reserved: Readonly<Record<Domain, ReadonlySet<string>>>;
    readonly nonfinite: Counter;
}

/**
 * Build the column plan and the notes for one dialect.
 * @param snapshot - the snapshot
 * @param resolved - the resolved options
 * @returns the plan
 */
function plan(snapshot: GraphSnapshot, resolved: Resolved): Plan {
    const { dialect } = resolved;
    const caps = dialectCapabilities(dialect);
    // the generic position and edge-id notes are replaced by planTable()'s (both are written as
    // plain attributes and read back without their role)
    const notes = checkCapabilities(snapshot, caps, resolved.common, {
        roles: SLOT_ROLES[dialect],
        positionDtype: "f32",
    }).filter((n) => n.code !== LOSS.POSITIONS && n.code !== LOSS.EDGE_IDS_DROPPED);
    const note: NoteFn = (code, message, column = null, count = null): void => {
        notes.push(Object.freeze({ code, message, column, count }));
    };
    const ctx: PlanContext = {
        snapshot,
        resolved,
        dialect,
        caps,
        notes,
        note,
        reserved: reservedKeys(snapshot, resolved),
        nonfinite: { count: 0 },
    };
    const nodes = planTable(ctx, snapshot.nodes, "node");
    const edges = planTable(ctx, snapshot.edges, "edge");
    const graph = caps.graphAttributes ? planTable(ctx, snapshot.graph, "graph") : [];
    withdrawSlotNotes(notes, [...nodes, ...edges]);
    const weights = planWeights(ctx, edges);
    if (ctx.nonfinite.count > 0) {
        note(
            JSON_LOSS.NONFINITE_AS_NULL,
            `${ctx.nonfinite.count} non-finite number(s) are written as null`,
            null,
            ctx.nonfinite.count,
        );
    }
    const { folding, directed } = planDirection(ctx);
    const multigraph = snapshot.flags.multigraph || snapshot.meta.declaredMultigraph === true;
    if (dialect === "jgf") {
        jgfIdNotes(snapshot, note);
    }
    const edgeIds = snapshot.edges.byRole("id");
    if (dialect === "cytoscape" && edgeIds !== null && edgeIds.nullCount > 0) {
        note(
            LOSS.EDGE_IDS_GENERATED,
            `${edgeIds.nullCount} edge(s) have no id; canonical e<index> ids are generated for them`,
            edgeIds.meta.name,
            edgeIds.nullCount,
        );
    }
    return { resolved, notes, nodes, edges, graph, edgeIds, weights, folding, directed, multigraph };
}

/**
 * Withdraw the generic list note of a Cytoscape classes column: the slot writes the list as the
 * `classes` string and the importer reads it back as the list it was.
 * @param notes - the notes so far
 * @param plans - the node and edge plans
 */
function withdrawSlotNotes(notes: LossNote[], plans: readonly ColumnPlan[]): void {
    for (const plan of plans) {
        if (plan.slot !== "classes") {
            continue;
        }
        const at = notes.findIndex((n) => n.code === LOSS.LIST && n.column === plan.column.meta.name);
        if (at >= 0) {
            notes.splice(at, 1);
        }
    }
}

/**
 * The output keys each dialect reserves for its structure (a column of that name cannot be
 * written as an attribute): the id and endpoint keys, and the weight key of a weighted snapshot.
 * @param snapshot - the snapshot
 * @param resolved - the resolved options
 * @returns the reserved keys per table
 */
function reservedKeys(snapshot: GraphSnapshot, resolved: Resolved): Readonly<Record<Domain, ReadonlySet<string>>> {
    const { dialect } = resolved;
    const node = new Set<string>();
    const edge = new Set<string>(snapshot.flags.weighted ? [resolved.weightKey] : []);
    const graph = new Set<string>();
    switch (dialect) {
        case "node-link":
        case "d3":
            if (resolved.nodeIdKey !== null) {
                node.add(resolved.nodeIdKey);
            }
            edge.add(resolved.sourceKey).add(resolved.targetKey);
            break;
        case "vis":
            node.add(resolved.nodeIdKey ?? "id");
            edge.add(resolved.sourceKey).add(resolved.targetKey);
            break;
        case "cytoscape":
            node.add("id").add("parent");
            edge.add("id").add("source").add("target");
            break;
        case "jgf":
        case "graphology":
            break;
        default: {
            const name: string = dialect;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dialect ${name}`, { option: "dialect", found: name });
        }
    }
    return { node, edge, graph };
}

/**
 * Classify the columns of one table: the slot each is written into and what it loses on the way
 * (JSON declares no types, so the notes name what the importer's inference reads back).
 * @param ctx - the plan context
 * @param table - the table
 * @param domain - its domain
 * @returns the plans of the written columns, in declaration order
 */
function planTable(ctx: PlanContext, table: Iterable<Column>, domain: Domain): ColumnPlan[] {
    const { dialect, note } = ctx;
    const plans: ColumnPlan[] = [];
    const reserved = ctx.reserved[domain];
    const used = new Set<string>();
    for (const column of table) {
        const { meta } = column;
        const { role } = meta;
        const setRows = column.length - column.nullCount;
        if (role !== null && (STRUCTURAL_ROLES.has(role) || TEMPORAL_ROLES.has(role))) {
            continue;
        }
        if (domain !== "graph") {
            ctx.nonfinite.count += countNonFinite(column);
        }
        const slotted = planSlot(ctx, column, domain);
        if (slotted !== null) {
            if (slotted !== SKIPPED) {
                plans.push(slotted);
            }
            continue;
        }
        let key = meta.name;
        let slot: Slot = "attribute";
        if (NESTED_DIALECTS.has(dialect) && domain !== "graph") {
            if (key.endsWith(SUFFIX.element)) {
                key = key.slice(0, -SUFFIX.element.length);
                slot = "element";
            } else if (dialect === "cytoscape" && CYTOSCAPE_ELEMENT_KEYS.has(key)) {
                slot = "element";
            } else if (key.endsWith(SUFFIX.data)) {
                key = key.slice(0, -SUFFIX.data.length);
            }
        }
        if (slot === "attribute" && reserved.has(key)) {
            note(
                JSON_LOSS.RESERVED_KEY,
                `${domain} column "${meta.name}" cannot be written: "${key}" is a reserved key of ${dialect}`,
                meta.name,
                setRows,
            );
            continue;
        }
        const slotKey = `${slot}:${key}`;
        if (used.has(slotKey)) {
            note(
                JSON_LOSS.RESERVED_KEY,
                `${domain} column "${meta.name}" cannot be written: key "${key}" is already used by another column`,
                meta.name,
                setRows,
            );
            continue;
        }
        used.add(slotKey);
        if (setRows === 0 && domain !== "graph") {
            note(
                LOSS.EMPTY_COLUMN,
                `${domain} column "${meta.name}" has no set cell and is not written: JSON writes values, never declarations`,
                meta.name,
                0,
            );
        }
        inferenceNotes(ctx, column, domain, setRows);
        plans.push({ column, key, slot });
    }
    return plans;
}

/** planSlot()'s answer for a role column handled (or dropped) without a plan of its own. */
const SKIPPED = Symbol("skipped");

/**
 * The structural slot of a role column in the dialect, when it has one: Cytoscape's parent,
 * position and classes, JGF's label and relation; the id-role edge column is written
 * structurally (data.id / key / id) where the dialect has an edge id and as a plain attribute
 * elsewhere; a position column without a slot is a plain array attribute.
 * @param ctx - the plan context
 * @param column - the column
 * @param domain - its domain
 * @returns the plan, SKIPPED for a column written structurally or dropped, null for a plain attribute
 */
function planSlot(ctx: PlanContext, column: Column, domain: Domain): ColumnPlan | typeof SKIPPED | null {
    const { dialect, note } = ctx;
    const { meta } = column;
    const { role } = meta;
    const setRows = column.length - column.nullCount;
    if (role === "parent" || role === "parents") {
        if (dialect === "cytoscape" && domain === "node") {
            if (role === "parent" && column.dtype === "u32" && meta.refersTo === "node") {
                return { column, key: "parent", slot: "parent" };
            }
            note(
                LOSS.PARENTS,
                `node column "${meta.name}" (${role}) cannot be written: Cytoscape has a single parent`,
                meta.name,
                setRows,
            );
        }
        return SKIPPED;
    }
    if (role === "position" && domain === "node") {
        if (dialect === "cytoscape" && (column.dtype === "f32" || column.dtype === "f64") && meta.components >= 2) {
            const z = countNonZeroZ(column);
            if (z > 0) {
                note(
                    JSON_LOSS.POSITION_Z_DROPPED,
                    `${z} position(s) have a non-zero z; Cytoscape positions are 2D`,
                    meta.name,
                    z,
                );
            }
            return { column, key: "position", slot: "cytoscapePosition" };
        }
        note(
            LOSS.POSITIONS,
            `node column "${meta.name}" (position) is written as a plain array attribute; its role is lost on re-import`,
            meta.name,
            setRows,
        );
        return { column, key: meta.name, slot: "position" };
    }
    if (role === "id" && domain === "edge") {
        if (dialect === "node-link" || dialect === "d3") {
            note(
                LOSS.EDGE_IDS_DROPPED,
                `edge column "${meta.name}" (id) is written as a plain attribute; ${dialect} has no edge id`,
                meta.name,
                setRows,
            );
            return null;
        }
        return SKIPPED;
    }
    if (dialect === "cytoscape" && role === "classes" && column.dtype === "list" && column.child.dtype === "string") {
        return { column, key: "classes", slot: "classes" };
    }
    if (dialect === "jgf" && role === "label" && (column.dtype === "string" || column.dtype === "dict")) {
        return { column, key: "label", slot: "label" };
    }
    if (
        dialect === "jgf" &&
        role === "kind" &&
        domain === "edge" &&
        (column.dtype === "string" || column.dtype === "dict")
    ) {
        return { column, key: "relation", slot: "relation" };
    }
    return null;
}

/**
 * What the importer's inference (design section 5.1: JSON numbers are i32 when integral, else
 * f64; strings are strings; arrays and objects are json) changes about a written attribute
 * column beyond what checkCapabilities() already reported for its dtype: an f64 column whose set
 * values are all integers reads back as i32.
 * @param ctx - the plan context
 * @param column - the column
 * @param domain - its domain
 * @param setRows - the set rows
 */
function inferenceNotes(ctx: PlanContext, column: Column, domain: Domain, setRows: number): void {
    if (column.dtype !== "f64" || column.meta.components > 1 || setRows === 0) {
        return;
    }
    const { data } = column;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && !Number.isInteger(data[r])) {
            return;
        }
    }
    ctx.note(
        LOSS.INTEGRAL_F64,
        `${domain} column "${column.meta.name}" is f64 with integral values only; JSON declares no types and it reads back as i32`,
        column.meta.name,
        setRows,
    );
}

/**
 * The explicit weights and their JSON text (non-finite values become null and are counted).
 * @param ctx - the plan context
 * @param edges - the edge column plans (for the weight-key clash note)
 * @returns the weight text function
 */
function planWeights(ctx: PlanContext, edges: readonly ColumnPlan[]): (e: number) => string | null {
    const { snapshot, nonfinite } = ctx;
    const weights = explicitWeights(snapshot);
    if (!weights.weighted) {
        const clash = edges.find((p) => p.slot === "attribute" && p.key === ctx.resolved.weightKey);
        if (clash !== undefined) {
            ctx.note(
                LOSS.WEIGHT_KEY_CLASH,
                `edge column "${clash.column.meta.name}" is written under "${clash.key}", the key the importer reads THE weight from; it reads back as the weight, not as a column`,
                clash.column.meta.name,
                clash.column.length - clash.column.nullCount,
            );
        }
        return (): null => null;
    }
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (weights.isExplicit(e) && !Number.isFinite(weights.value(e))) {
            nonfinite.count++;
        }
    }
    return (e: number): string | null => {
        if (!weights.isExplicit(e)) {
            return null;
        }
        return Number.isFinite(weights.value(e)) ? weights.text(e) : "null";
    };
}

/**
 * The direction decisions: the pair folding (mutual pairs are written as two directed edges and
 * reported), the file-level direction under the onMixedDirection policy of a dialect without a
 * per-edge flag, and the note of the dialects that carry no direction at all.
 * @param ctx - the plan context
 * @returns the folding and the file-level direction
 */
function planDirection(ctx: PlanContext): { readonly folding: PairFolding; readonly directed: boolean } {
    const { snapshot, dialect, caps, note } = ctx;
    const folding = pairFolding(snapshot);
    if (folding.mutualCount > 0) {
        note(
            LOSS.MUTUAL_EXPANDED,
            `${folding.mutualCount} mutual pair(s) are written as two directed edges; the mutual mark is lost`,
            null,
            folding.mutualCount,
        );
    }
    let { directed } = snapshot;
    if (
        !caps.mixedDirection &&
        countMixedEdges(snapshot) > 0 &&
        ctx.resolved.common.onMixedDirection === "undirected"
    ) {
        directed = false;
    }
    if (dialect === "cytoscape" || dialect === "vis" || dialect === "d3") {
        const assumed = DIALECT_DEFAULT_DIRECTED[dialect];
        if (directed !== assumed) {
            note(
                JSON_LOSS.DIRECTION_DROPPED,
                `${dialect} has no direction flag; the ${directed ? "directed" : "undirected"} graph re-imports as ${assumed ? "directed" : "undirected"} unless defaultDirected is passed`,
            );
        }
    }
    return { folding, directed };
}

/**
 * How many rows of a position column have a non-zero third component.
 * @param column - an f32 / f64 column with at least two components
 * @returns the count
 */
function countNonZeroZ(column: Column): number {
    if ((column.dtype !== "f32" && column.dtype !== "f64") || column.meta.components < 3) {
        return 0;
    }
    const { components } = column.meta;
    let count = 0;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && column.data[r * components + 2] !== 0) {
            count++;
        }
    }
    return count;
}

/**
 * The JGF id notes: numeric ids become string keys, colliding texts are refused, and integer-like
 * keys are enumerated first and ascending by every JSON parser so the node order may change.
 * @param snapshot - the snapshot
 * @param note - the recorder
 */
function jgfIdNotes(
    snapshot: GraphSnapshot,
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): void {
    const { ids } = snapshot;
    const seen = new Set<string>();
    let numeric = 0;
    let collisions = 0;
    let indexLike = 0;
    let lastIndex = -1;
    let ordered = true;
    let sawOther = false;
    for (let i = 0; i < ids.size; i++) {
        const id = ids.idOf(i);
        const text = String(id);
        if (typeof id === "number") {
            numeric++;
        }
        if (seen.has(text)) {
            collisions++;
        }
        seen.add(text);
        if (isArrayIndexKey(text)) {
            indexLike++;
            const n = Number(text);
            if (sawOther || n <= lastIndex) {
                ordered = false;
            }
            lastIndex = n;
        } else {
            sawOther = true;
        }
    }
    if (numeric > 0) {
        note(
            JSON_LOSS.NUMERIC_IDS_STRINGIFIED,
            `${numeric} numeric node id(s) become JGF object keys (strings); pass ids: "canonical" on re-import`,
            null,
            numeric,
        );
    }
    if (collisions > 0) {
        note(
            JSON_LOSS.ID_TEXT_COLLISION,
            `${collisions} node id(s) share their text with another id; export() will throw`,
            null,
            collisions,
        );
    }
    if (!ordered) {
        note(
            JSON_LOSS.NODE_ORDER,
            `${indexLike} integer-like node id(s) are enumerated first and ascending by JSON parsers; node order changes on re-import`,
            null,
            indexLike,
        );
    }
}

/**
 * Whether a key is an array index in the JS sense (enumerated before other keys, ascending).
 * @param text - the key
 * @returns true for canonical integer text below 2^32 - 1
 */
function isArrayIndexKey(text: string): boolean {
    return /^(0|[1-9][0-9]*)$/.test(text) && Number(text) < 4294967295;
}

// ============================================================ writing

/**
 * The text parts of the document.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields one part per header, node, edge and footer
 * @returns nothing
 */
function* write(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { dialect } = p.resolved;
    if (p.notes.some((n) => n.code === LOSS.MIXED_DIRECTION_ERROR)) {
        throw new GraphFormatError(
            "E_DIRECTED",
            `${dialect} has no mixed direction; pass onMixedDirection "directed" or "undirected"`,
            { reason: "mixed direction", dialect },
        );
    }
    if (p.notes.some((n) => n.code === JSON_LOSS.ID_TEXT_COLLISION)) {
        throw new GraphFormatError("E_INVALID_ID", "two node ids have the same text; JGF keys nodes by text", {
            reason: "collision",
        });
    }
    switch (dialect) {
        case "node-link":
        case "d3":
            yield* writeNodeLink(snapshot, p);
            return;
        case "vis":
            yield* writeVis(snapshot, p);
            return;
        case "graphology":
            yield* writeGraphology(snapshot, p);
            return;
        case "jgf":
            yield* writeJgf(snapshot, p);
            return;
        case "cytoscape":
            yield* writeCytoscape(snapshot, p);
            return;
        default: {
            const name: string = dialect;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dialect ${name}`, { option: "dialect", found: name });
        }
    }
}

/** The shared per-row state of the writers. */
interface Cursor {
    readonly w: JsonWriter;
    readonly ids: GraphSnapshot["ids"];
    readonly nonfinite: Counter;
}

/**
 * Write the members of one row's planned columns into the current object, attributes only.
 * @param c - the cursor
 * @param plans - the table plan
 * @param row - the row
 * @param slots - which slots to write
 */
function writeSlots(c: Cursor, plans: readonly ColumnPlan[], row: number, slots: ReadonlySet<Slot>): void {
    for (const item of plans) {
        if (!slots.has(item.slot) || !item.column.isSet(row)) {
            continue;
        }
        let text: string;
        switch (item.slot) {
            case "position": {
                const value = item.column.value(row);
                const dims = dimsOf(item.column);
                const values = Array.from(value as ArrayLike<number>).slice(0, dims);
                text = valueText(values, item.column.dtype === "f32", c.nonfinite);
                break;
            }
            case "classes":
                text = JSON.stringify((item.column as Extract<Column, { dtype: "list" }>).sliceOf(row).join(" "));
                break;
            default:
                text = cellText(item.column, row, c.ids, c.nonfinite);
                break;
        }
        c.w.member(item.key, text);
    }
}

const ATTRIBUTE_SLOTS: ReadonlySet<Slot> = new Set(["attribute", "position"]);
const ELEMENT_SLOTS: ReadonlySet<Slot> = new Set(["element", "classes"]);

/**
 * The number of components a position column writes: `extra.sourceDims` when recorded, else all.
 * @param column - the position column
 * @returns 2 or 3 (or the component count)
 */
function dimsOf(column: Column): number {
    const dims: unknown = column.meta.extra.sourceDims;
    if (typeof dims === "number" && dims >= 1 && dims <= column.meta.components) {
        return dims;
    }
    return column.meta.components;
}

/**
 * Whether any planned column of a slot set has a value in a row (to omit an empty nested dict).
 * @param plans - the table plan
 * @param row - the row
 * @param slots - the slots
 * @returns true when something would be written
 */
function anySet(plans: readonly ColumnPlan[], row: number, slots: ReadonlySet<Slot>): boolean {
    return plans.some((item) => slots.has(item.slot) && item.column.isSet(row));
}

/**
 * The JSON text of a node id.
 * @param id - the id
 * @returns the text
 */
function idText(id: NodeId): string {
    return typeof id === "number" && Object.is(id, -0) ? "0" : JSON.stringify(id);
}

/**
 * The edge id text of an edge, or null when none is set.
 * @param p - the plan
 * @param e - the edge
 * @param c - the cursor
 * @returns the text, or null
 */
function edgeIdText(p: Plan, e: number, c: Cursor): string | null {
    const column = p.edgeIds;
    if (column === null || !column.isSet(e)) {
        return null;
    }
    return cellText(column, e, c.ids, c.nonfinite);
}

/**
 * Whether an edge is written, and as which kind, under the plan's direction policy.
 * @param p - the plan
 * @param e - the edge
 * @returns "skip" for a folded mirror, "undirected" for a pair primary, "directed" otherwise
 */
function edgeDisposition(p: Plan, e: number): "skip" | "undirected" | "directed" {
    const { folding } = p;
    if (folding.folded(e)) {
        return "skip";
    }
    if (!p.directed) {
        return "undirected";
    }
    if (folding.sourceDirected(e)) {
        return "directed";
    }
    // a source-undirected edge (pair primary or expanded self-loop): a dialect with a per-edge flag
    // keeps it; one without writes it per the policy (the pair folds to one directed edge)
    if (dialectCapabilities(p.resolved.dialect).mixedDirection) {
        return "undirected";
    }
    return p.resolved.common.onMixedDirection === "directed" ? "directed" : "undirected";
}

/**
 * The graph-level attribute object.
 * @param c - the cursor
 * @param p - the plan
 */
function writeGraphObject(c: Cursor, p: Plan): void {
    c.w.open("{");
    for (const item of p.graph) {
        if (item.column.isSet(0)) {
            c.w.member(item.key, cellText(item.column, 0, c.ids, c.nonfinite));
        }
    }
    c.w.close("}");
}

/**
 * The endpoint texts of an edge: ids, or array positions under index links.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @param e - the edge
 * @returns the source and target texts
 */
function endpoints(snapshot: GraphSnapshot, p: Plan, e: number): [string, string] {
    const list = snapshot.edgeList();
    const u = list.src[e];
    const v = list.dst[e];
    if (p.resolved.indexLinks) {
        return [String(u), String(v)];
    }
    return [idText(snapshot.ids.idOf(u)), idText(snapshot.ids.idOf(v))];
}

/**
 * Write a node-link / d3 document.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields the parts
 * @returns nothing
 */
function* writeNodeLink(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { resolved } = p;
    const c: Cursor = { w: new JsonWriter(resolved.indent), ids: snapshot.ids, nonfinite: { count: 0 } };
    const { w } = c;
    w.open("{");
    if (resolved.dialect !== "d3") {
        // d3 is the bare shape: no direction, multigraph or graph keys (the importer sniffs it by their absence)
        w.member("directed", p.directed ? "true" : "false");
        w.member("multigraph", p.multigraph ? "true" : "false");
        w.key("graph");
        writeGraphObject(c, p);
    }
    w.key("nodes");
    w.open("[");
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.item();
        w.open("{");
        if (resolved.nodeIdKey !== null) {
            w.member(resolved.nodeIdKey, idText(snapshot.ids.idOf(i)));
        }
        writeSlots(c, p.nodes, i, ATTRIBUTE_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.key(resolved.edgesKey);
    w.open("[");
    for (let e = 0; e < snapshot.edgeCount; e++) {
        const disposition = edgeDisposition(p, e);
        if (disposition === "skip") {
            continue;
        }
        const [s, t] = endpoints(snapshot, p, e);
        w.item();
        w.open("{");
        w.member(resolved.sourceKey, s);
        w.member(resolved.targetKey, t);
        const weight = p.weights(e);
        if (weight !== null) {
            w.member(resolved.weightKey, weight);
        }
        writeSlots(c, p.edges, e, ATTRIBUTE_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.close("}");
    yield w.take();
}

/**
 * Write a vis.js document.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields the parts
 * @returns nothing
 */
function* writeVis(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { resolved } = p;
    const c: Cursor = { w: new JsonWriter(resolved.indent), ids: snapshot.ids, nonfinite: { count: 0 } };
    const { w } = c;
    const nodeIdKey = resolved.nodeIdKey ?? "id";
    w.open("{");
    w.key("nodes");
    w.open("[");
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.item();
        w.open("{");
        w.member(nodeIdKey, idText(snapshot.ids.idOf(i)));
        writeSlots(c, p.nodes, i, ATTRIBUTE_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.key("edges");
    w.open("[");
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (edgeDisposition(p, e) === "skip") {
            continue;
        }
        const list = snapshot.edgeList();
        w.item();
        w.open("{");
        const id = edgeIdText(p, e, c);
        if (id !== null) {
            w.member("id", id);
        }
        w.member(resolved.sourceKey, idText(snapshot.ids.idOf(list.src[e])));
        w.member(resolved.targetKey, idText(snapshot.ids.idOf(list.dst[e])));
        const weight = p.weights(e);
        if (weight !== null) {
            w.member(resolved.weightKey, weight);
        }
        writeSlots(c, p.edges, e, ATTRIBUTE_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.close("}");
    yield w.take();
}

/**
 * Write a graphology serialisation.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields the parts
 * @returns nothing
 */
function* writeGraphology(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { resolved } = p;
    const c: Cursor = { w: new JsonWriter(resolved.indent), ids: snapshot.ids, nonfinite: { count: 0 } };
    const { w } = c;
    let type: string;
    if (!p.directed) {
        type = "undirected";
    } else {
        type = countMixedEdges(snapshot) > 0 ? "mixed" : "directed";
    }
    w.open("{");
    w.key("attributes");
    writeGraphObject(c, p);
    w.key("options");
    w.open("{");
    w.member("type", JSON.stringify(type));
    // the shape the importer read is written back as read: `multi` and `allowSelfLoops` only when
    // the source declared them (or the graph needs them)
    if (snapshot.meta.declaredMultigraph !== null || p.multigraph) {
        w.member("multi", p.multigraph ? "true" : "false");
    }
    if (resolved.shape.allowSelfLoops !== undefined) {
        w.member("allowSelfLoops", !resolved.shape.allowSelfLoops && snapshot.selfLoopCount === 0 ? "false" : "true");
    }
    w.close("}");
    w.key("nodes");
    w.open("[");
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.item();
        w.open("{");
        w.member("key", idText(snapshot.ids.idOf(i)));
        if (anySet(p.nodes, i, ATTRIBUTE_SLOTS)) {
            w.key("attributes");
            w.open("{");
            writeSlots(c, p.nodes, i, ATTRIBUTE_SLOTS);
            w.close("}");
        }
        writeSlots(c, p.nodes, i, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.key("edges");
    w.open("[");
    for (let e = 0; e < snapshot.edgeCount; e++) {
        const disposition = edgeDisposition(p, e);
        if (disposition === "skip") {
            continue;
        }
        const list = snapshot.edgeList();
        w.item();
        w.open("{");
        const id = edgeIdText(p, e, c);
        if (id !== null) {
            w.member("key", id);
        }
        w.member("source", idText(snapshot.ids.idOf(list.src[e])));
        w.member("target", idText(snapshot.ids.idOf(list.dst[e])));
        if (type === "mixed" && disposition === "undirected") {
            w.member("undirected", "true");
        }
        const weight = p.weights(e);
        if (weight !== null || anySet(p.edges, e, ATTRIBUTE_SLOTS)) {
            w.key("attributes");
            w.open("{");
            if (weight !== null) {
                w.member(resolved.weightKey, weight);
            }
            writeSlots(c, p.edges, e, ATTRIBUTE_SLOTS);
            w.close("}");
        }
        writeSlots(c, p.edges, e, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.close("}");
    yield w.take();
}

/**
 * Write a JSON Graph Format v2 document.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields the parts
 * @returns nothing
 */
function* writeJgf(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { resolved } = p;
    const c: Cursor = { w: new JsonWriter(resolved.indent), ids: snapshot.ids, nonfinite: { count: 0 } };
    const { w } = c;
    const labelSlots: ReadonlySet<Slot> = new Set(["label"]);
    const relationSlots: ReadonlySet<Slot> = new Set(["relation"]);
    w.open("{");
    w.key("graph");
    w.open("{");
    if (resolved.shape.id !== undefined) {
        w.member("id", JSON.stringify(resolved.shape.id));
    }
    if (snapshot.meta.name !== null) {
        w.member("label", JSON.stringify(snapshot.meta.name));
    }
    if (resolved.shape.type !== undefined) {
        w.member("type", JSON.stringify(resolved.shape.type));
    }
    w.member("directed", p.directed ? "true" : "false");
    if (p.graph.some((item) => item.column.isSet(0))) {
        w.key("metadata");
        writeGraphObject(c, p);
    }
    w.key("nodes");
    w.open("{");
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.key(String(snapshot.ids.idOf(i)));
        w.open("{");
        writeSlots(c, p.nodes, i, labelSlots);
        if (anySet(p.nodes, i, ATTRIBUTE_SLOTS)) {
            w.key("metadata");
            w.open("{");
            writeSlots(c, p.nodes, i, ATTRIBUTE_SLOTS);
            w.close("}");
        }
        writeSlots(c, p.nodes, i, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("}");
    w.key("edges");
    w.open("[");
    for (let e = 0; e < snapshot.edgeCount; e++) {
        const disposition = edgeDisposition(p, e);
        if (disposition === "skip") {
            continue;
        }
        const list = snapshot.edgeList();
        w.item();
        w.open("{");
        const id = edgeIdText(p, e, c);
        if (id !== null) {
            w.member("id", id);
        }
        w.member("source", JSON.stringify(String(snapshot.ids.idOf(list.src[e]))));
        w.member("target", JSON.stringify(String(snapshot.ids.idOf(list.dst[e]))));
        writeSlots(c, p.edges, e, relationSlots);
        if (p.directed && disposition === "undirected") {
            w.member("directed", "false");
        }
        writeSlots(c, p.edges, e, labelSlots);
        const weight = p.weights(e);
        if (weight !== null || anySet(p.edges, e, ATTRIBUTE_SLOTS)) {
            w.key("metadata");
            w.open("{");
            if (weight !== null) {
                w.member(resolved.weightKey, weight);
            }
            writeSlots(c, p.edges, e, ATTRIBUTE_SLOTS);
            w.close("}");
        }
        writeSlots(c, p.edges, e, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.close("}");
    w.close("}");
    yield w.take();
}

/**
 * Write Cytoscape.js elements.
 * @param snapshot - the snapshot
 * @param p - the plan
 * @yields the parts
 * @returns nothing
 */
function* writeCytoscape(snapshot: GraphSnapshot, p: Plan): Generator<string, void, undefined> {
    const { resolved } = p;
    const c: Cursor = { w: new JsonWriter(resolved.indent), ids: snapshot.ids, nonfinite: { count: 0 } };
    const { w } = c;
    const parentSlots: ReadonlySet<Slot> = new Set(["parent"]);
    const positionSlots: ReadonlySet<Slot> = new Set(["cytoscapePosition"]);
    w.open("{");
    w.key("elements");
    w.open("{");
    w.key("nodes");
    w.open("[");
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.item();
        w.open("{");
        w.key("data");
        w.open("{");
        w.member("id", idText(snapshot.ids.idOf(i)));
        writeSlots(c, p.nodes, i, parentSlots);
        writeSlots(c, p.nodes, i, ATTRIBUTE_SLOTS);
        w.close("}");
        for (const item of p.nodes) {
            if (positionSlots.has(item.slot) && item.column.isSet(i)) {
                const value = item.column.value(i) as ArrayLike<number>;
                const f32 = item.column.dtype === "f32";
                w.key("position");
                w.open("{");
                w.member("x", numberText(value[0], f32, c.nonfinite));
                w.member("y", numberText(value[1], f32, c.nonfinite));
                w.close("}");
            }
        }
        writeSlots(c, p.nodes, i, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.key("edges");
    w.open("[");
    const generated = edgeIdGenerator(snapshot, p);
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (edgeDisposition(p, e) === "skip") {
            continue;
        }
        const list = snapshot.edgeList();
        w.item();
        w.open("{");
        w.key("data");
        w.open("{");
        w.member("id", edgeIdText(p, e, c) ?? generated(e));
        w.member("source", idText(snapshot.ids.idOf(list.src[e])));
        w.member("target", idText(snapshot.ids.idOf(list.dst[e])));
        const weight = p.weights(e);
        if (weight !== null) {
            w.member(resolved.weightKey, weight);
        }
        writeSlots(c, p.edges, e, ATTRIBUTE_SLOTS);
        w.close("}");
        writeSlots(c, p.edges, e, ELEMENT_SLOTS);
        w.close("}");
        yield w.take();
    }
    w.close("]");
    w.close("}");
    if (p.graph.some((item) => item.column.isSet(0))) {
        w.key("data");
        writeGraphObject(c, p);
    }
    const extra = resolved.shape.cytoscape;
    if (extra !== undefined) {
        for (const key of Object.keys(extra)) {
            if (key !== "elements" && key !== "data") {
                w.member(key, valueText(extra[key], false, c.nonfinite));
            }
        }
    }
    w.close("}");
    yield w.take();
}

/**
 * A generator of canonical `e<index>` edge ids for the edges without one (design section 4.6),
 * skipping any text a node id or an explicit edge id already uses (Cytoscape ids share one
 * namespace).
 * @param snapshot - the snapshot
 * @param p - the plan
 * @returns the id text of an edge without an explicit id
 */
function edgeIdGenerator(snapshot: GraphSnapshot, p: Plan): (e: number) => string {
    const used = new Set<string>();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        used.add(String(snapshot.ids.idOf(i)));
    }
    const column = p.edgeIds;
    if (column !== null) {
        for (let e = 0; e < snapshot.edgeCount; e++) {
            if (column.isSet(e)) {
                used.add(String(column.value(e)));
            }
        }
    }
    return (e: number): string => {
        let candidate = `e${e}`;
        for (let k = 2; used.has(candidate); k++) {
            candidate = `e${e}_${k}`;
        }
        used.add(candidate);
        return JSON.stringify(candidate);
    };
}

// ============================================================ the plugin

/**
 * The JSON exporter plugin (design section 8.5). `capabilities` is the node-link table (the
 * default dialect); check() applies the table of the dialect actually selected.
 */
export const jsonExporter: GraphExporter<JsonExportOptions> = Object.freeze({
    format: "json",
    capabilities: dialectCapabilities("node-link"),

    /**
     * Pre-flight: every loss of the selected dialect, without writing anything.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the notes, empty when the export is exact
     */
    check(snapshot: GraphSnapshot, options?: JsonExportOptions & CommonExportOptions): readonly LossNote[] {
        return Object.freeze([...plan(snapshot, resolve(snapshot, options)).notes]);
    },

    /**
     * Write the document as UTF-8 chunks.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: JsonExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(write(snapshot, plan(snapshot, resolve(snapshot, options))));
    },

    /**
     * Write the document as one string.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the document
     */
    exportToString(snapshot: GraphSnapshot, options?: JsonExportOptions & CommonExportOptions): Promise<string> {
        return joinText(write(snapshot, plan(snapshot, resolve(snapshot, options))));
    },
});

/**
 * The capability table of one dialect, for callers that pick a dialect before check().
 * @param dialect - the dialect
 * @returns the frozen table
 */
export function jsonCapabilities(dialect: JsonDialect): ExportCapabilities {
    return dialectCapabilities(dialect);
}
