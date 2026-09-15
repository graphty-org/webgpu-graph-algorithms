/**
 * The DOT / Graphviz exporter (design section 8.5; research note 07 section 9). Writes one
 * `[strict] graph | digraph [name] { ... }` with the graph attributes first, every node in index
 * order (a node statement carrying its set cells as attributes, or a `subgraph` block for a cluster
 * container node with its members nested inside), then every logical edge in index order with its
 * explicit weight, `key` (the edge id role), ports and attributes.
 *
 * What the format keeps and what check() reports: any id and any text (DOT quotes everything
 * except a text ending in a backslash, which the lexer cannot read back); bool / i32 / f64 cells
 * (f64 written with a decimal point so the dtype survives re-import; other numeric dtypes are
 * written and read back inferred); containment through the parent role (clusters); graph
 * attributes; node positions as `pos="x,y"`. Mixed direction is folded per onMixedDirection
 * (DOT has none); lists, json, defaults, options, temporal and visual roles cannot be written.
 */

import { type Column, GraphFormatError, type GraphSnapshot, type NodeId } from "@graphty/graph-format";

import { type ChildrenCsr, childrenCsr } from "../../children.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { isWritableDotText, quoteDotId } from "../../common/escape.js";
import { capabilities, checkCapabilities, countMixedEdges, LOSS, sanitizeIds } from "../../common/export.js";
import { formatDecimal, formatF32, formatF64, formatInteger } from "../../common/format.js";
import { canonicalId } from "../../common/ids.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { inferTextDtype } from "../../common/text.js";
import { type ExplicitWeights, explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import {
    CLUSTER_COLUMN,
    DOT_FORMAT,
    DOT_META_KEY,
    KEY_ATTRIBUTE,
    LABEL_ATTRIBUTE,
    PARENT_COLUMN,
    POS_ATTRIBUTE,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
} from "./names.js";

/** The DOT exporter's format-specific options. */
export interface DotExportOptions {
    /** The indentation of one nesting level; four spaces by default. */
    indent?: string | undefined;
    /** The graph name to write; `meta.name` by default, null for an anonymous graph. */
    name?: string | null | undefined;
    /** Whether to write `strict`; by default when `meta.extra.dot.strict` is true. */
    strict?: boolean | undefined;
}

/** The LossNote codes of the DOT exporter; the shared ones are LOSS's. */
export const DOT_LOSS = Object.freeze({
    /** An id, name or text with a backslash before a quote or at its end cannot be written as a DOT quoted string; export() throws. */
    TRAILING_BACKSLASH: "E_DOT_TRAILING_BACKSLASH",
    /** A non-finite f32 / f64 cell has no numeric DOT spelling and reads back as text. */
    NON_FINITE: "W_DOT_NON_FINITE",
    /** Text cells that look like numbers or booleans read back as such (DOT attribute values are untyped). */
    TEXT_INFERRED: LOSS.TEXT_INFERRED,
    /** A plain column named like an attribute the exporter writes for a role (weight, key, pos) is not written. */
    ATTRIBUTE_CLASH: "W_DOT_ATTRIBUTE_CLASH",
    /** A mutual pair is written as two directed edges. */
    MUTUAL_EXPANDED: LOSS.MUTUAL_EXPANDED,
    /** A parents (multi-parent) column cannot be written; DOT clusters nest. */
    PARENTS_DROPPED: LOSS.PARENTS,
    /** A position column that is not a node column of 2 or 3 components is not written. */
    POSITION_SHAPE: "W_DOT_POSITION_SHAPE",
    /** An id whose text reads back as the other type under ids: "canonical" (1.5 as text, "1" as 1). */
    ID_TEXT_TYPE: LOSS.ID_TEXT_TYPE,
    /** A declared column whose every row is unset is not written (DOT writes cells, never declarations). */
    EMPTY_COLUMN_DROPPED: LOSS.EMPTY_COLUMN,
    /** A role-less column named `label` reads back with the label role. */
    ROLE_ASSUMED: LOSS.ROLE_ASSUMED,
});

/** The roles DOT has a slot for (the label attribute, key, ports, clusters); every other role is reported. */
const KEPT_ROLES: ReadonlySet<string> = new Set(["label", "id", "sourcePort", "targetPort", "parent"]);

/** The column name the importer gives each mapped role on re-import. */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({
    label: LABEL_ATTRIBUTE,
    id: KEY_ATTRIBUTE,
    sourcePort: SOURCE_PORT_COLUMN,
    targetPort: TARGET_PORT_COLUMN,
    parent: PARENT_COLUMN,
    position: POS_ATTRIBUTE,
});

const CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: false,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "any",
    dtypes: ["bool", "i32", "f64", "string"],
    components: false,
    lists: false,
    json: false,
    defaults: false,
    options: false,
    hierarchy: true,
    temporal: "none",
    graphAttributes: true,
    positions: true,
    viz: false,
});

const DEFAULT_INDENT = "    ";
const CLUSTER_PREFIX = "cluster";
const COMPASS_POINTS: ReadonlySet<string> = new Set(["n", "ne", "e", "se", "s", "sw", "w", "nw", "c", "_"]);

/** Roles never written as attributes, per domain. */
const SKIPPED_NODE_ROLES: ReadonlySet<string> = new Set([
    "parent",
    "parents",
    "position",
    "color",
    "size",
    "shape",
    "thickness",
    "start",
    "end",
    "timestamp",
    "timestamps",
    "spells",
    "open",
    "timeText",
    "directed",
    "pair",
    "mutual",
    "weight",
]);
const SKIPPED_EDGE_ROLES: ReadonlySet<string> = new Set([
    "id",
    "weight",
    "directed",
    "pair",
    "mutual",
    "sourcePort",
    "targetPort",
    "position",
    "color",
    "size",
    "shape",
    "thickness",
    "start",
    "end",
    "timestamp",
    "timestamps",
    "spells",
    "open",
    "timeText",
    "parent",
    "parents",
]);
const SKIPPED_GRAPH_ROLES: ReadonlySet<string> = new Set(["position", "color", "size", "shape", "thickness"]);
/** Text columns whose values are read back as text regardless of their spelling (no TEXT_INFERRED note). */
const TEXT_ROLES: ReadonlySet<string> = new Set(["label", "id", "sourcePort", "targetPort"]);

/**
 * The exporter plugin for DOT / Graphviz text (design section 12.4).
 */
export const dotExporter: GraphExporter<DotExportOptions> = Object.freeze({
    format: DOT_FORMAT,
    capabilities: CAPABILITIES,

    /**
     * Pre-flight: every loss the DOT text would incur, without writing anything.
     * @param snapshot - the snapshot to check
     * @param options - format-specific and common options
     * @returns the notes, empty when the export is exact
     */
    check(snapshot: GraphSnapshot, options?: DotExportOptions & CommonExportOptions): readonly LossNote[] {
        const resolved = resolveExportOptions(options);
        const notes = checkCapabilities(snapshot, CAPABILITIES, resolved, {
            positionDtype: "f32",
            roles: KEPT_ROLES,
            roleNames: ROLE_NAMES,
        });
        const plan = new ExportPlan(snapshot, resolved, options);
        return [...notes, ...plan.notes()];
    },

    /**
     * Write the snapshot as UTF-8 chunks.
     * @param snapshot - the snapshot to write
     * @param options - format-specific and common options
     * @returns the encoded chunks
     */
    export(snapshot: GraphSnapshot, options?: DotExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(writeDot(snapshot, options));
    },

    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot to write
     * @param options - format-specific and common options
     * @returns the whole document
     */
    exportToString(snapshot: GraphSnapshot, options?: DotExportOptions & CommonExportOptions): Promise<string> {
        return joinText(writeDot(snapshot, options));
    },
});

/**
 * The text parts of a DOT document.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @yields one statement (or header / footer) at a time
 * @returns nothing
 */
function* writeDot(snapshot: GraphSnapshot, options?: DotExportOptions & CommonExportOptions): Generator<string> {
    const resolved = resolveExportOptions(options);
    const plan = new ExportPlan(snapshot, resolved, options);
    plan.refuse();
    yield* plan.write();
}

/**
 * Whether a text ends in a backslash, which a DOT quoted string cannot carry (the lexer would read
 * the backslash and the closing quote as an escaped quote; common/escape.ts isWritableDotText).
 * @param text - the text
 * @returns true when unwritable
 */
function endsWithBackslash(text: string): boolean {
    return !isWritableDotText(text);
}

/**
 * Whether a text is an HTML string (balanced angle brackets around the whole text), which DOT
 * writes bare as `<...>`.
 * @param text - the text
 * @returns true for an HTML string
 */
function isHtmlString(text: string): boolean {
    if (text.length < 2 || !text.startsWith("<") || !text.endsWith(">")) {
        return false;
    }
    let depth = 0;
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        if (c === 0x3c) {
            depth++;
        } else if (c === 0x3e) {
            depth--;
            if (depth === 0 && i !== text.length - 1) {
                return false;
            }
        }
    }
    return depth === 0;
}

/**
 * Write a text as a DOT ID: an HTML string bare, everything else through quoteDotId.
 * @param text - the text
 * @returns the DOT ID
 */
function writeId(text: string): string {
    return isHtmlString(text) ? text : quoteDotId(text);
}

/**
 * Write a port text (`f0`, `f0:ne`, `ne`) as its DOT spelling after the endpoint.
 * @param port - the port text
 * @returns `:port[:compass]`
 */
function writePort(port: string): string {
    const colon = port.lastIndexOf(":");
    if (colon > 0 && COMPASS_POINTS.has(port.slice(colon + 1))) {
        return `:${quoteDotId(port.slice(0, colon))}:${port.slice(colon + 1)}`;
    }
    return `:${quoteDotId(port)}`;
}

/**
 * Whether a column can be written as attribute text: every scalar dtype; lists and json cannot.
 * @param column - the column
 * @returns true when writable
 */
function isTextualDtype(column: Column): boolean {
    return column.dtype !== "list" && column.dtype !== "json";
}

/**
 * Everything one export needs to know about a snapshot, computed once and shared by check() and
 * write(): the output direction, the columns written per table, the cluster structure, and the
 * format-specific loss notes.
 */
class ExportPlan {
    private readonly snapshot: GraphSnapshot;

    private readonly resolved: ResolvedExportOptions;

    private readonly indent: string;

    private readonly name: string | null;

    private readonly strict: boolean;

    private readonly mixed: number;

    private readonly directed: boolean;

    private readonly nodeColumns: Column[];

    private readonly edgeColumns: Column[];

    private readonly graphColumns: Column[];

    private readonly position: Column | null;

    private readonly positionDims: 2 | 3;

    private readonly parent: Column | null;

    private readonly cluster: Column | null;

    private readonly edgeId: Column | null;

    private readonly weights: ExplicitWeights;

    private readonly folding: PairFolding;

    private readonly sourcePort: Column | null;

    private readonly targetPort: Column | null;

    private readonly children: ChildrenCsr;

    private readonly extraNotes: LossNote[] = [];

    /**
     * Plan one export.
     * @param snapshot - the snapshot
     * @param resolved - the resolved common options
     * @param options - the caller's options (format-specific fields read here)
     */
    constructor(snapshot: GraphSnapshot, resolved: ResolvedExportOptions, options?: DotExportOptions) {
        this.snapshot = snapshot;
        this.resolved = resolved;
        this.indent = options?.indent ?? DEFAULT_INDENT;
        this.name = options?.name === undefined ? snapshot.meta.name : options.name;
        this.strict = options?.strict ?? isStrictMeta(snapshot);
        this.mixed = countMixedEdges(snapshot);
        if (this.mixed > 0 && resolved.onMixedDirection !== "error") {
            this.directed = resolved.onMixedDirection === "directed";
        } else {
            this.directed = snapshot.directed;
        }
        const { nodes, edges, graph } = snapshot;
        this.position = nodes.byRole("position");
        this.positionDims = positionDims(this.position);
        this.parent = nodes.byRole("parent");
        const cluster = nodes.get(CLUSTER_COLUMN);
        this.cluster = cluster !== null && cluster.dtype === "bool" ? cluster : null;
        this.edgeId = edges.byRole("id");
        this.weights = explicitWeights(snapshot);
        this.folding = pairFolding(snapshot);
        this.sourcePort = edges.byRole("sourcePort");
        this.targetPort = edges.byRole("targetPort");
        this.nodeColumns = [...nodes].filter((c) => this.writesNodeColumn(c));
        this.edgeColumns = [...edges].filter((c) => this.writesEdgeColumn(c));
        this.graphColumns = [...graph].filter(
            (c) => isTextualDtype(c) && (c.meta.role === null || !SKIPPED_GRAPH_ROLES.has(c.meta.role)),
        );
        this.children = childrenCsr(snapshot, {
            column: this.parent !== null && this.parent.dtype === "u32" ? this.parent : null,
        });
    }

    /**
     * Whether a node column is written as node attributes.
     * @param column - the column
     * @returns true when written
     */
    private writesNodeColumn(column: Column): boolean {
        const { name, role } = column.meta;
        if (!isTextualDtype(column) || name === CLUSTER_COLUMN) {
            return false;
        }
        if (role !== null && SKIPPED_NODE_ROLES.has(role)) {
            return false;
        }
        if (name === POS_ATTRIBUTE && role !== "position") {
            this.clash("node", name, "the position role is written as pos");
            return false;
        }
        return true;
    }

    /**
     * Whether an edge column is written as edge attributes.
     * @param column - the column
     * @returns true when written
     */
    private writesEdgeColumn(column: Column): boolean {
        const { name, role } = column.meta;
        if (!isTextualDtype(column)) {
            return false;
        }
        if (role !== null && SKIPPED_EDGE_ROLES.has(role)) {
            return false;
        }
        if (name === KEY_ATTRIBUTE && this.edgeId !== null) {
            this.clash("edge", name, "the edge id role is written as key");
            return false;
        }
        if (name === "weight") {
            this.clash("edge", name, "the weight is written as weight");
            return false;
        }
        return true;
    }

    /**
     * Record a column not written because its name is an attribute the exporter writes for a role.
     * @param domain - the table
     * @param name - the column name
     * @param reason - why
     */
    private clash(domain: string, name: string, reason: string): void {
        this.extraNotes.push(
            Object.freeze({
                code: DOT_LOSS.ATTRIBUTE_CLASH,
                message: `${domain} column "${name}" has no role and is not written: ${reason}`,
                column: name,
                count: null,
            }),
        );
    }

    /**
     * The format-specific loss notes (the generic ones come from checkCapabilities).
     * @returns the notes
     */
    notes(): LossNote[] {
        const notes: LossNote[] = [...this.extraNotes];
        const note = (
            code: string,
            message: string,
            column: string | null = null,
            count: number | null = null,
        ): void => {
            notes.push(Object.freeze({ code, message, column, count }));
        };
        const { snapshot } = this;
        const unwritable = this.countUnwritable();
        if (unwritable > 0) {
            note(
                DOT_LOSS.TRAILING_BACKSLASH,
                `${unwritable} id(s), name(s) or text value(s) hold a backslash before a quote or at the end, which a DOT quoted string cannot carry; export() will throw`,
                null,
                unwritable,
            );
        }
        for (const [label, columns] of [
            ["node", this.nodeColumns],
            ["edge", this.edgeColumns],
            ["graph", this.graphColumns],
        ] as const) {
            for (const column of columns) {
                if (label !== "graph" && column.nullCount === column.length) {
                    note(
                        DOT_LOSS.EMPTY_COLUMN_DROPPED,
                        `${label} column "${column.meta.name}" has no set cell and is not written: DOT writes cells, never declarations`,
                        column.meta.name,
                        0,
                    );
                    continue;
                }
                if (
                    label !== "graph" &&
                    column.meta.role === null &&
                    column.meta.name === LABEL_ATTRIBUTE &&
                    (column.dtype === "string" || column.dtype === "dict")
                ) {
                    note(
                        DOT_LOSS.ROLE_ASSUMED,
                        `${label} column "${column.meta.name}" reads back with the label role`,
                        column.meta.name,
                        column.length - column.nullCount,
                    );
                }
                if (column.dtype === "f32" || column.dtype === "f64") {
                    const bad = countNonFinite(column);
                    if (bad > 0) {
                        note(
                            DOT_LOSS.NON_FINITE,
                            `${label} column "${column.meta.name}" holds ${bad} non-finite value(s) with no numeric DOT spelling; they read back as text`,
                            column.meta.name,
                            bad,
                        );
                    }
                }
                if (
                    (column.dtype === "string" || column.dtype === "dict") &&
                    (column.meta.role === null || !TEXT_ROLES.has(column.meta.role))
                ) {
                    const typed = countTypedLookingText(column);
                    if (typed > 0) {
                        note(
                            DOT_LOSS.TEXT_INFERRED,
                            `${label} column "${column.meta.name}" holds ${typed} text value(s) that look like numbers or booleans; DOT values are untyped and they read back as such`,
                            column.meta.name,
                            typed,
                        );
                    }
                }
            }
        }
        const mutual = this.folding.mutualCount;
        if (mutual > 0) {
            note(
                DOT_LOSS.MUTUAL_EXPANDED,
                `${mutual} mutual pair(s) are written as two directed edges each; the mutual mark is lost`,
                null,
                mutual,
            );
        }
        const parents = snapshot.nodes.byRole("parents");
        if (parents !== null) {
            note(
                DOT_LOSS.PARENTS_DROPPED,
                `node column "${parents.meta.name}" (parents) cannot be written: a DOT node is in one cluster`,
                parents.meta.name,
                parents.length - parents.nullCount,
            );
        }
        if (this.position !== null && this.position.meta.components !== 2 && this.position.meta.components !== 3) {
            note(
                DOT_LOSS.POSITION_SHAPE,
                `node column "${this.position.meta.name}" (position) has ${this.position.meta.components} components and is not written; pos takes 2 or 3`,
                this.position.meta.name,
                this.position.length - this.position.nullCount,
            );
        }
        for (const [label, table] of [
            ["edge", snapshot.edges],
            ["graph", snapshot.graph],
        ] as const) {
            const column = table.byRole("position");
            if (column !== null) {
                note(
                    DOT_LOSS.POSITION_SHAPE,
                    `${label} column "${column.meta.name}" (position) is not written; only node positions map to pos`,
                    column.meta.name,
                    column.length - column.nullCount,
                );
            }
        }
        const textIds = countTextIds(snapshot);
        if (textIds > 0) {
            note(
                DOT_LOSS.ID_TEXT_TYPE,
                `${textIds} node id(s) read back as the other type under ids: "canonical" (a string "1" becomes 1, a number 1.5 becomes "1.5")`,
                null,
                textIds,
            );
        }
        return notes;
    }

    /**
     * Throw for the conditions check() reports as errors: mixed direction under "error", and an
     * unwritable id / name / text.
     */
    refuse(): void {
        if (this.mixed > 0 && this.resolved.onMixedDirection === "error") {
            throw new GraphFormatError(
                "E_DIRECTED",
                `${this.mixed} undirected edge(s) in a directed graph; DOT has no mixed direction and onMixedDirection is "error"`,
                { reason: LOSS.MIXED_DIRECTION_ERROR, count: this.mixed },
            );
        }
        const first = this.firstUnwritable();
        if (first !== null) {
            throw new GraphFormatError(
                first.kind === "id" ? "E_INVALID_ID" : "E_COLUMN_TYPE",
                `${first.kind} ${JSON.stringify(first.text)} ends in a backslash, which a DOT quoted string cannot carry`,
                { reason: "trailing backslash", kind: first.kind, value: first.text },
            );
        }
        sanitizeIds(this.snapshot, CAPABILITIES.idCharset, this.resolved.sanitizeIds);
    }

    /**
     * How many ids, names and text values end in a backslash.
     * @returns the count
     */
    private countUnwritable(): number {
        let count = 0;
        this.forEachText((text) => {
            if (endsWithBackslash(text)) {
                count++;
            }
        });
        return count;
    }

    /**
     * The first id, name or text value ending in a backslash.
     * @returns what it is and its text, or null
     */
    private firstUnwritable(): { kind: string; text: string } | null {
        let found: { kind: string; text: string } | null = null;
        this.forEachText((text, kind) => {
            if (found === null && endsWithBackslash(text)) {
                found = { kind, text };
            }
        });
        return found;
    }

    /**
     * Visit every text the document writes as a quoted string: the graph name, node ids, column
     * names and string / dict cells.
     * @param visit - the visitor
     */
    private forEachText(visit: (text: string, kind: string) => void): void {
        const { snapshot } = this;
        if (this.name !== null) {
            visit(this.name, "graph name");
        }
        for (let i = 0; i < snapshot.nodeCount; i++) {
            const id = snapshot.ids.idOf(i);
            if (typeof id === "string") {
                visit(id, "id");
            }
        }
        for (const columns of [this.nodeColumns, this.edgeColumns, this.graphColumns]) {
            for (const column of columns) {
                visit(column.meta.name, "attribute name");
                if (column.dtype === "string" || column.dtype === "dict") {
                    for (let r = 0; r < column.length; r++) {
                        if (column.isSet(r)) {
                            visit(column.value(r) as string, "value");
                        }
                    }
                }
            }
        }
        for (const column of [this.edgeId, this.sourcePort, this.targetPort]) {
            if (column !== null && (column.dtype === "string" || column.dtype === "dict")) {
                for (let r = 0; r < column.length; r++) {
                    if (column.isSet(r)) {
                        visit(column.value(r) as string, "value");
                    }
                }
            }
        }
    }

    /**
     * The document, one statement per part.
     * @yields the parts
     * @returns nothing
     */
    *write(): Generator<string> {
        const { snapshot, indent } = this;
        const keyword = this.directed ? "digraph" : "graph";
        const name = this.name === null ? "" : ` ${quoteDotId(this.name)}`;
        yield `${this.strict ? "strict " : ""}${keyword}${name} {\n`;
        const graphAttributes = this.attributesOf(this.graphColumns, 0);
        if (graphAttributes.length > 0) {
            yield `${indent}graph [${graphAttributes.join(", ")}];\n`;
        }
        const emitted = new Uint8Array(snapshot.nodeCount);
        for (let i = 0; i < snapshot.nodeCount; i++) {
            yield* this.writeNode(i, 1, emitted);
        }
        yield* this.writeEdges();
        yield "}\n";
    }

    /**
     * Write one node (and, for a container, its cluster block with the members nested).
     * @param i - the node index
     * @param depth - the nesting depth
     * @param emitted - which nodes were written already
     * @yields the statements
     * @returns nothing
     */
    private *writeNode(i: number, depth: number, emitted: Uint8Array): Generator<string> {
        if (emitted[i] === 1) {
            return;
        }
        emitted[i] = 1;
        const pad = this.indent.repeat(depth);
        const idText = this.idText(i);
        const kids = this.children.childrenOf(i);
        const marked = this.cluster !== null && this.cluster.isSet(i) && this.cluster.value(i) === true;
        if (kids.length === 0 && !marked) {
            yield `${pad}${this.nodeStatement(i, idText)};\n`;
            return;
        }
        if (!marked) {
            // a real node that is also a parent: its cells are node attributes, the cluster only groups
            yield `${pad}${this.nodeStatement(i, idText)};\n`;
        }
        const inner = `${pad}${this.indent}`;
        yield `${pad}subgraph ${writeId(idText)} {\n`;
        if (!idText.startsWith(CLUSTER_PREFIX)) {
            yield `${inner}cluster=true;\n`;
        }
        if (marked) {
            const attributes = this.nodeAttributes(i);
            if (attributes.length > 0) {
                yield `${inner}graph [${attributes.join(", ")}];\n`;
            }
        }
        for (const child of kids) {
            if (emitted[child] === 1) {
                yield `${inner}${writeId(this.idText(child))};\n`;
            } else {
                yield* this.writeNode(child, depth + 1, emitted);
            }
        }
        yield `${pad}}\n`;
    }

    /**
     * A node statement without its terminator.
     * @param i - the node index
     * @param idText - the id text
     * @returns `id` or `id [attrs]`
     */
    private nodeStatement(i: number, idText: string): string {
        const attributes = this.nodeAttributes(i);
        return attributes.length === 0 ? writeId(idText) : `${writeId(idText)} [${attributes.join(", ")}]`;
    }

    /**
     * The attributes of a node: its set cells plus `pos` from the position column.
     * @param i - the node index
     * @returns the attribute texts
     */
    private nodeAttributes(i: number): string[] {
        const attributes = this.attributesOf(this.nodeColumns, i);
        const pos = this.positionText(i);
        if (pos !== null) {
            attributes.push(`${POS_ATTRIBUTE}=${quoteDotId(pos)}`);
        }
        return attributes;
    }

    /**
     * The `pos` text of a node, or null when it has none.
     * @param i - the node index
     * @returns `x,y` or `x,y,z`
     */
    private positionText(i: number): string | null {
        const column = this.position;
        if (column === null || !column.isSet(i)) {
            return null;
        }
        if (column.dtype !== "f32" && column.dtype !== "f64" && column.dtype !== "i32") {
            return null;
        }
        const { components } = column.meta;
        if (components !== 2 && components !== 3) {
            return null;
        }
        const format = column.dtype === "f32" ? formatF32 : formatF64;
        const base = i * components;
        const { data } = column;
        const parts = [format(data[base]), format(data[base + 1])];
        if (this.positionDims === 3) {
            parts.push(format(components === 3 ? data[base + 2] : 0));
        }
        return parts.join(",");
    }

    /**
     * The `name=value` attributes of one row over a column list (set cells only).
     * @param columns - the columns
     * @param row - the row
     * @returns the attribute texts
     */
    private attributesOf(columns: readonly Column[], row: number): string[] {
        const out: string[] = [];
        for (const column of columns) {
            if (!column.isSet(row)) {
                continue;
            }
            const text = cellText(column, row);
            if (text !== null) {
                out.push(`${quoteDotId(column.meta.name)}=${writeId(text)}`);
            }
        }
        return out;
    }

    /**
     * Every logical edge in index order, mirror halves of expanded undirected pairs folded away.
     * @yields the edge statements
     * @returns nothing
     */
    private *writeEdges(): Generator<string> {
        const { snapshot, indent } = this;
        const list = snapshot.edgeList();
        const op = this.directed ? "->" : "--";
        for (let e = 0; e < snapshot.edgeCount; e++) {
            if (this.folding.folded(e)) {
                continue;
            }
            const attributes = this.attributesOf(this.edgeColumns, e);
            const weight = this.weights.text(e);
            if (weight !== null) {
                attributes.unshift(`weight=${quoteDotId(weight)}`);
            }
            const key = this.keyText(e);
            if (key !== null) {
                attributes.unshift(`${KEY_ATTRIBUTE}=${writeId(key)}`);
            }
            const source = `${writeId(this.idText(list.src[e]))}${this.portText(this.sourcePort, e)}`;
            const target = `${writeId(this.idText(list.dst[e]))}${this.portText(this.targetPort, e)}`;
            const tail = attributes.length === 0 ? "" : ` [${attributes.join(", ")}]`;
            yield `${indent}${source} ${op} ${target}${tail};\n`;
        }
    }

    /**
     * The edge id of an edge as `key` text, or null.
     * @param e - the edge index
     * @returns the text
     */
    private keyText(e: number): string | null {
        const column = this.edgeId;
        if (column === null || !column.isSet(e)) {
            return null;
        }
        if (column.dtype === "f32" || column.dtype === "f64") {
            return formatF64(column.data[e]);
        }
        return cellText(column, e);
    }

    /**
     * The port suffix of an endpoint.
     * @param column - the port column, or null
     * @param e - the edge index
     * @returns `:port` or ""
     */
    private portText(column: Column | null, e: number): string {
        if (column === null || !column.isSet(e)) {
            return "";
        }
        const text = cellText(column, e);
        return text === null || text.length === 0 ? "" : writePort(text);
    }

    /**
     * The id text of a node.
     * @param i - the node index
     * @returns String(id)
     */
    private idText(i: number): string {
        return idToText(this.snapshot.ids.idOf(i));
    }
}

/**
 * The text of a node id: the string itself, or the shortest decimal of a number.
 * @param id - the id
 * @returns the text
 */
function idToText(id: NodeId): string {
    return typeof id === "string" ? id : formatF64(id);
}

/**
 * Whether the snapshot's meta records a strict graph.
 * @param snapshot - the snapshot
 * @returns true when meta.extra.dot.strict is true
 */
function isStrictMeta(snapshot: GraphSnapshot): boolean {
    const dot: unknown = snapshot.meta.extra[DOT_META_KEY];
    return typeof dot === "object" && dot !== null && (dot as { strict?: unknown }).strict === true;
}

/**
 * The dimensions `pos` is written with: what the importer recorded in extra.sourceDims, else 2
 * for a 2-component column and 3 otherwise.
 * @param column - the position column, or null
 * @returns 2 or 3
 */
function positionDims(column: Column | null): 2 | 3 {
    if (column === null) {
        return 2;
    }
    const dims: unknown = column.meta.extra.sourceDims;
    if (dims === 2 || dims === 3) {
        return dims;
    }
    return column.meta.components === 2 ? 2 : 3;
}

/**
 * The attribute text of one set cell, or null for a dtype the format cannot write.
 * @param column - the column
 * @param row - the row
 * @returns the text
 */
function cellText(column: Column, row: number): string | null {
    switch (column.dtype) {
        case "bool":
            return column.value(row) === true ? "true" : "false";
        case "i32":
        case "u32":
        case "u8":
            return numericText(column, row, formatInteger);
        case "f32":
            return numericText(column, row, (v) => formatDecimal(v, "f32"));
        case "f64":
            return numericText(column, row, (v) => formatDecimal(v, "f64"));
        case "dict":
        case "string":
            return column.value(row) ?? "";
        case "list":
        case "json":
            return null;
        default: {
            const name: string = (column as Column).dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * The text of a numeric cell: one number, or the `components` numbers joined by commas.
 * @param column - a numeric column
 * @param row - the row
 * @param format - the per-number formatter
 * @returns the text
 */
function numericText(
    column: Column & { readonly data: ArrayLike<number> },
    row: number,
    format: (value: number) => string,
): string {
    const { components } = column.meta;
    if (components === 1) {
        return format(column.data[row]);
    }
    const parts: string[] = [];
    for (let k = 0; k < components; k++) {
        parts.push(format(column.data[row * components + k]));
    }
    return parts.join(",");
}

/**
 * How many set cells of an f32 / f64 column are non-finite.
 * @param column - the column
 * @returns the count
 */
function countNonFinite(column: Column): number {
    if (column.dtype !== "f32" && column.dtype !== "f64") {
        return 0;
    }
    const { components } = column.meta;
    let count = 0;
    for (let r = 0; r < column.length; r++) {
        if (!column.isSet(r)) {
            continue;
        }
        for (let k = 0; k < components; k++) {
            if (!Number.isFinite(column.data[r * components + k])) {
                count++;
                break;
            }
        }
    }
    return count;
}

/**
 * How many set cells of a text column read back as something other than text under the 5.1 grammar.
 * @param column - a string or dict column
 * @returns the count
 */
function countTypedLookingText(column: Column): number {
    if (column.dtype !== "string" && column.dtype !== "dict") {
        return 0;
    }
    let count = 0;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && inferTextDtype(column.value(r) ?? "") !== "string") {
            count++;
        }
    }
    return count;
}

/**
 * How many ids read back as the other type under the canonical rule: numeric ids that are not
 * canonical integer text (read back as strings) and string ids that are (read back as numbers).
 * @param snapshot - the snapshot
 * @returns the count
 */
function countTextIds(snapshot: GraphSnapshot): number {
    const { ids } = snapshot;
    if (ids.kind === "identity" || ids.kind === "dense") {
        return 0;
    }
    let count = 0;
    for (let i = 0; i < ids.size; i++) {
        const id = ids.idOf(i);
        if (canonicalId(idToText(id)) !== id) {
            count++;
        }
    }
    return count;
}
