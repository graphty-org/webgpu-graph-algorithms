/**
 * The CSV / TSV exporter (design section 8.5; research note 07 section 9): writes the edge table
 * (`Source,Target,Type,Id,Label,Weight,<attributes>` in the Gephi dialect, `source,target,weight,
 * <attributes>` in the generic one) or, with `table: "nodes"`, the node table (`Id,Label,
 * <attributes>`), RFC 4180 quoted, one row per logical edge with expanded pairs folded back
 * through the `pair` role column.
 *
 * What survives a re-import exactly: ids (as text under the canonical rule), topology and
 * orientation, explicit weights (blank cells for defaulted ones), the per-row direction of the
 * Gephi dialect, edge ids and labels (string), bool / i32 / f64 / string columns (an f64 value is
 * written with a decimal point so it reads back as f64), low-cardinality dicts, and set empty
 * strings (the quoted empty cell; an unset cell is written as nothing). check() reports everything
 * else: the generic dialect drops direction, mutual pairs become two directed rows, lists and json
 * values are written as text, non-finite numbers do not read back, attribute names that collide
 * with the reserved headers are not written, node attributes are written by the node table only,
 * and the edge table carries neither isolated nodes nor the node order.
 */

import { type Column, GraphFormatError, type GraphSnapshot, type NodeId } from "@graphty/graph-format";

import { DICT_SAMPLE_ROWS, DictHeuristic } from "../../common/attributes.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { quoteCsvCell } from "../../common/escape.js";
import { capabilities, checkCapabilities, countMixedEdges, LOSS } from "../../common/export.js";
import { formatDecimal, formatInteger } from "../../common/format.js";
import { canonicalId } from "../../common/ids.js";
import { joinListText } from "../../common/lists.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { inferTextDtype, type TextDtype } from "../../common/text.js";
import { type ExplicitWeights, explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import { EDGE_ID_NAMES, findColumn, LABEL_NAMES } from "./header.js";

/** The format-specific options of the CSV exporter. */
export interface CsvExportOptions {
    /**
     * The header spelling: "gephi" (default) writes `Source,Target,Type,...,Weight` with the per-row
     * direction; "generic" writes `source,target,...,weight` and no direction column.
     */
    dialect?: "gephi" | "generic" | undefined;
    /** Which table to write: the edge table (default) or the node table. */
    table?: "edges" | "nodes" | undefined;
    /** The field delimiter; "," by default. */
    delimiter?: string | undefined;
    /** The line terminator; "\n" by default. */
    newline?: "\n" | "\r\n" | undefined;
    /** Whether to write the header row; true by default. */
    header?: boolean | undefined;
}

/** The CSV loss-note codes of check(); the shared ones are LOSS's. */
export const CSV_LOSS = Object.freeze({
    /** Two node ids share one text (a number and a string); export() throws E_INVALID_ID. */
    ID_TEXT_COLLISION: LOSS.ID_TEXT_COLLISION,
    /** Ids whose text reads back as the other type under the canonical rule. */
    ID_TEXT_TYPE: LOSS.ID_TEXT_TYPE,
    /** The generic dialect has no direction column; an undirected or mixed graph reads back as directed. */
    DIRECTION_DROPPED: "W_CSV_DIRECTION_DROPPED",
    /** Mutual pairs are written as two directed rows. */
    MUTUAL_EXPANDED: LOSS.MUTUAL_EXPANDED,
    /** An attribute column named like a reserved header is not written. */
    RESERVED_NAME: "W_CSV_RESERVED_NAME",
    /** A column without a role that the importer gives one back by its name. */
    ROLE_ASSUMED: LOSS.ROLE_ASSUMED,
    /** A role column (id, label) whose name the importer does not recognise; the role is lost. */
    ROLE_NAME: "W_CSV_ROLE_NAME",
    /** A role column (id, label) that is not string / dict reads back as string. */
    TEXT_ROLE: "W_CSV_TEXT_ROLE",
    /** NaN / Infinity in a numeric column read back as text. */
    NONFINITE: "W_CSV_NONFINITE",
    /** A text column whose every value reads back as a number or boolean. */
    TEXT_INFERRED: LOSS.TEXT_INFERRED,
    /** A dict column whose cardinality makes the importer read it back as string, or the reverse. */
    STORAGE_CLASS_CHANGED: LOSS.STORAGE_CLASS,
    /** Node attributes are written by a `table: "nodes"` export only. */
    NODE_TABLE: "W_CSV_NODE_TABLE",
    /** The edge table carries no node without an edge: isolated nodes vanish on re-import. */
    ISOLATED_NODES: "W_CSV_ISOLATED_NODES",
    /** The edge table lists nodes by first appearance; the node order (and indices) change on re-import. */
    NODE_ORDER: "W_CSV_NODE_ORDER",
});

/** What the CSV format keeps as declared. */
export const CSV_CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: true,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "any",
    dtypes: ["bool", "i32", "f64", "string", "dict"],
});

/** The dialect header spellings. */
interface Dialect {
    readonly source: string;
    readonly target: string;
    readonly type: string | null;
    readonly id: string;
    readonly label: string;
    readonly weight: string;
}

const DIALECTS: Readonly<Record<"gephi" | "generic", Dialect>> = {
    gephi: { source: "Source", target: "Target", type: "Type", id: "Id", label: "Label", weight: "Weight" },
    generic: { source: "source", target: "target", type: null, id: "id", label: "label", weight: "weight" },
};

/** The roles the tables have a slot for (the id and label headers); every other role is reported. */
const KEPT_ROLES: ReadonlySet<string> = new Set(["id", "label"]);

/** Roles the edge and node tables never write as attributes. */
const SKIPPED_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "position",
    "color",
    "size",
    "shape",
    "thickness",
    "parent",
    "parents",
    "start",
    "end",
    "timestamp",
    "timestamps",
    "spells",
    "open",
]);

/** The CSV options with defaults applied. */
interface ResolvedCsvExportOptions {
    readonly dialect: Dialect;
    readonly table: "edges" | "nodes";
    readonly delimiter: string;
    readonly newline: string;
    readonly header: boolean;
}

/** A column written under a header. */
interface ColumnOut {
    readonly column: Column;
    readonly header: string;
}

/** The plan of one export call: what is written, and the notes. */
interface Plan {
    readonly csv: ResolvedCsvExportOptions;
    readonly common: ResolvedExportOptions;
    readonly notes: LossNote[];
    /** The rows of the edge table (logical edge indices; mirrors of undirected pairs left out). */
    readonly edgeRows: number[];
    readonly folding: PairFolding;
    readonly edgeId: Column | null;
    readonly edgeLabel: Column | null;
    readonly weights: ExplicitWeights;
    readonly edgeColumns: ColumnOut[];
    readonly nodeLabel: Column | null;
    readonly nodeColumns: ColumnOut[];
    /** The id text of every node. */
    readonly idText: (index: number) => string;
}

/**
 * Apply the defaults of the CSV export options and check every value.
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for a value outside its set
 */
function resolveCsvExportOptions(
    options: (CsvExportOptions & CommonExportOptions) | undefined,
): ResolvedCsvExportOptions {
    const o: CsvExportOptions = options ?? {};
    if (o.dialect !== undefined && o.dialect !== "gephi" && o.dialect !== "generic") {
        throw new GraphFormatError("E_UNSUPPORTED", 'option dialect: expected "gephi" or "generic"', {
            option: "dialect",
            found: o.dialect,
        });
    }
    if (o.table !== undefined && o.table !== "edges" && o.table !== "nodes") {
        throw new GraphFormatError("E_UNSUPPORTED", 'option table: expected "edges" or "nodes"', {
            option: "table",
            found: o.table,
        });
    }
    if (
        o.delimiter !== undefined &&
        (typeof o.delimiter !== "string" ||
            o.delimiter.length !== 1 ||
            o.delimiter === '"' ||
            o.delimiter === "\n" ||
            o.delimiter === "\r")
    ) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            "option delimiter: expected one character other than a quote or a line break",
            { option: "delimiter", found: o.delimiter },
        );
    }
    if (o.newline !== undefined && o.newline !== "\n" && o.newline !== "\r\n") {
        throw new GraphFormatError("E_UNSUPPORTED", "option newline: expected LF or CRLF", {
            option: "newline",
            found: o.newline,
        });
    }
    if (o.header !== undefined && typeof o.header !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", "option header: expected a boolean", {
            option: "header",
            found: o.header,
        });
    }
    return {
        dialect: DIALECTS[o.dialect ?? "gephi"],
        table: o.table ?? "edges",
        delimiter: o.delimiter ?? ",",
        newline: o.newline ?? "\n",
        header: o.header ?? true,
    };
}

/**
 * One scalar (or one list item) as text by its dtype.
 * @param dtype - the scalar dtype
 * @param value - the value
 * @returns the text
 */
function scalarText(dtype: string, value: unknown): string {
    switch (dtype) {
        case "bool":
            return value === true ? "true" : "false";
        case "f32":
        case "f64":
            return typeof value === "number" ? formatDecimal(value, dtype) : String(value);
        case "i32":
        case "u32":
        case "u8":
            return typeof value === "number" ? formatInteger(value) : String(value);
        case "string":
        case "dict":
            return typeof value === "string" ? value : String(value);
        case "json":
            return JSON.stringify(value) ?? "";
        default:
            return String(value);
    }
}

/**
 * The text of one cell: null for an unset row (written as nothing between the delimiters);
 * components and list items joined by `;`.
 * @param column - the column
 * @param row - the row
 * @returns the cell text, or null
 */
function cellText(column: Column, row: number): string | null {
    if (!column.isSet(row)) {
        return null;
    }
    switch (column.dtype) {
        case "list": {
            const items = column.sliceOf(row);
            const { dtype } = column.child;
            return joinListText(
                items.map((item) => scalarText(dtype, item)),
                "semicolon",
            );
        }
        case "json":
            return JSON.stringify(column.values[row]) ?? "";
        case "bool":
        case "string":
        case "dict":
            return scalarText(column.dtype, column.value(row));
        default: {
            const { components } = column.meta;
            const { data } = column;
            if (components === 1) {
                return scalarText(column.dtype, data[row]);
            }
            const parts: string[] = [];
            for (let k = 0; k < components; k++) {
                parts.push(scalarText(column.dtype, data[row * components + k]));
            }
            return parts.join(";");
        }
    }
}

/**
 * Build the plan of an export: resolve options, choose the rows and columns, and collect the
 * CSV-specific loss notes (check() adds the generic ones). The value-level notes need a pass over
 * every written cell and are collected for check() only; export() needs the plan and the fatal
 * id-collision note alone.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @param values - whether to scan the values for notes
 * @returns the plan
 */
/**
 * The id notes: text collisions are fatal (export() throws E_INVALID_ID); type changes under the
 * canonical re-read are reported.
 * @param snapshot - the snapshot
 * @param values - whether the values are inspected (false for a plan that writes nothing)
 * @param note - the note recorder
 */
function idNotes(
    snapshot: GraphSnapshot,
    values: boolean,
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): void {
    const { ids } = snapshot;
    if (ids.kind === "mixed") {
        const seen = new Set<string>();
        let collisions = 0;
        for (let i = 0; i < ids.size; i++) {
            const text = String(ids.idOf(i));
            if (seen.has(text)) {
                collisions++;
            } else {
                seen.add(text);
            }
        }
        if (collisions > 0) {
            note(
                CSV_LOSS.ID_TEXT_COLLISION,
                `${collisions} node id(s) share their text with another id (a number and a string); export() will throw E_INVALID_ID`,
                null,
                collisions,
            );
        }
    }
    if (values && ids.kind !== "identity" && ids.kind !== "dense") {
        let changed = 0;
        for (let i = 0; i < ids.size; i++) {
            const id: NodeId = ids.idOf(i);
            if (typeof canonicalId(String(id)) !== typeof id) {
                changed++;
            }
        }
        if (changed > 0) {
            note(
                CSV_LOSS.ID_TEXT_TYPE,
                `${changed} node id(s) read back as the other type under ids: "canonical" (a string "1" becomes 1, a number 1.5 becomes "1.5")`,
                null,
                changed,
            );
        }
    }
}

function planExport(
    snapshot: GraphSnapshot,
    options: (CsvExportOptions & CommonExportOptions) | undefined,
    values: boolean,
): Plan {
    const common = resolveExportOptions(options);
    const csv = resolveCsvExportOptions(options);
    const notes: LossNote[] = [];
    const note = (code: string, message: string, column: string | null = null, count: number | null = null): void => {
        notes.push(Object.freeze({ code, message, column, count }));
    };
    const { ids } = snapshot;
    const idText = (index: number): string => String(ids.idOf(index));
    idNotes(snapshot, values, note);

    // edge rows: fold the mirrors of undirected pairs; mutual pairs stay two directed rows
    const folding = pairFolding(snapshot);
    const edgeRows: number[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (!folding.folded(e)) {
            edgeRows.push(e);
        }
    }
    if (csv.dialect.type === null) {
        const mixed = countMixedEdges(snapshot);
        if (!snapshot.directed) {
            note(
                CSV_LOSS.DIRECTION_DROPPED,
                `the generic dialect has no direction column; ${snapshot.edgeCount} undirected edge(s) read back as directed unless the importer is told otherwise`,
                null,
                snapshot.edgeCount,
            );
        } else if (mixed > 0) {
            note(
                CSV_LOSS.DIRECTION_DROPPED,
                `the generic dialect has no direction column; ${mixed} undirected edge(s) of a mixed graph read back as one directed edge each`,
                null,
                mixed,
            );
        }
    }
    if (folding.mutualCount > 0) {
        note(
            CSV_LOSS.MUTUAL_EXPANDED,
            `${folding.mutualCount} mutual pair(s) are written as two directed rows; the mutual mark is lost`,
            null,
            folding.mutualCount,
        );
    }
    if (csv.table === "edges") {
        noteNodeCoverage(snapshot, edgeRows, note);
    }

    // weights
    const weights = explicitWeights(snapshot);

    // the edge table's columns
    const edgeId = snapshot.edges.byRole("id");
    const edgeLabel = snapshot.edges.byRole("label");
    const edgeReserved = new Set<string>([csv.dialect.source, csv.dialect.target, csv.dialect.weight].map(lower));
    if (csv.dialect.type !== null) {
        edgeReserved.add(lower(csv.dialect.type));
    }
    const edgeColumns = attributeColumns(
        snapshot.edges,
        "edge",
        edgeReserved,
        edgeId,
        edgeLabel,
        csv.table === "edges" ? note : null,
    );
    if (csv.table === "edges" && values) {
        checkRoleColumn(edgeId, "id", EDGE_ID_NAMES, "edge", edgeRows, note);
        checkRoleColumn(edgeLabel, "label", LABEL_NAMES, "edge", edgeRows, note);
        for (const { column } of edgeColumns) {
            checkValues(column, "edge", edgeRows, note);
        }
    }

    // the node table's columns
    const nodeLabel = snapshot.nodes.byRole("label");
    const nodeReserved = new Set<string>([lower(csv.dialect.id)]);
    const nodeRows: number[] = [];
    if (csv.table === "nodes" && values) {
        for (let i = 0; i < snapshot.nodeCount; i++) {
            nodeRows.push(i);
        }
    }
    const nodeColumns = attributeColumns(
        snapshot.nodes,
        "node",
        nodeReserved,
        null,
        nodeLabel,
        csv.table === "nodes" ? note : null,
    );
    if (csv.table === "nodes") {
        if (values) {
            checkRoleColumn(nodeLabel, "label", LABEL_NAMES, "node", nodeRows, note);
            for (const { column } of nodeColumns) {
                checkValues(column, "node", nodeRows, note);
            }
        }
    } else {
        const written = nodeColumns.length + (nodeLabel === null ? 0 : 1);
        if (written > 0) {
            note(
                CSV_LOSS.NODE_TABLE,
                `${written} node column(s) are written by a table: "nodes" export only`,
                null,
                written,
            );
        }
    }

    return {
        csv,
        common,
        notes,
        edgeRows,
        folding,
        edgeId,
        edgeLabel,
        weights,
        edgeColumns,
        nodeLabel,
        nodeColumns,
        idText,
    };
}

/**
 * The edge table is not a node carrier (research note 07 section 2.6: node attributes and node
 * presence come from a separate node table): a node without an edge is not written at all, and
 * the importer creates nodes in the order the edge rows first mention them. Both are reported so
 * check() predicts the ids and the order a re-import gives back (design section 8.5).
 * @param snapshot - the snapshot
 * @param edgeRows - the edge rows written
 * @param note - the recorder
 */
function noteNodeCoverage(
    snapshot: GraphSnapshot,
    edgeRows: readonly number[],
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): void {
    const { nodeCount } = snapshot;
    if (nodeCount === 0) {
        return;
    }
    const list = snapshot.edgeList();
    const seen = new Uint8Array(nodeCount);
    let next = 0;
    let reordered = 0;
    const mention = (index: number): void => {
        if (seen[index] === 1) {
            return;
        }
        seen[index] = 1;
        if (index !== next) {
            reordered++;
        }
        next++;
    };
    for (const e of edgeRows) {
        mention(list.src[e]);
        mention(list.dst[e]);
    }
    const isolated = nodeCount - next;
    if (isolated > 0) {
        note(
            CSV_LOSS.ISOLATED_NODES,
            `${isolated} node(s) have no edge and cannot be written by the edge table; write the node table (table: "nodes") to keep them`,
            null,
            isolated,
        );
    }
    if (reordered > 0) {
        note(
            CSV_LOSS.NODE_ORDER,
            `${reordered} node(s) are first mentioned by an edge row out of index order; a re-import numbers nodes by first appearance`,
            null,
            reordered,
        );
    }
}

/**
 * Lower-case a header name for reserved-name comparisons.
 * @param name - the name
 * @returns the lower-case name
 */
function lower(name: string): string {
    return name.toLowerCase();
}

/**
 * The attribute columns of a table that are written, in declaration order, skipping the role
 * columns handled elsewhere, the roles the format cannot hold, and names that collide with the
 * reserved headers (reported).
 * @param table - the node or edge table
 * @param domain - node or edge, for messages
 * @param reserved - lower-case header names the dialect writes for structure
 * @param idColumn - the table's id role column, or null
 * @param labelColumn - the table's label role column, or null
 * @param note - the recorder, or null when this table's notes are not wanted
 * @returns the columns and their headers
 */
function attributeColumns(
    table: Iterable<Column>,
    domain: "node" | "edge",
    reserved: ReadonlySet<string>,
    idColumn: Column | null,
    labelColumn: Column | null,
    note: ((code: string, message: string, column?: string | null, count?: number | null) => void) | null,
): ColumnOut[] {
    const out: ColumnOut[] = [];
    for (const column of table) {
        const { name, role } = column.meta;
        if (column === idColumn || column === labelColumn) {
            continue;
        }
        if (role !== null && SKIPPED_ROLES.has(role)) {
            continue;
        }
        const low = lower(name);
        const set = column.length - column.nullCount;
        const isIdName = domain === "edge" ? findColumn([name], EDGE_ID_NAMES) >= 0 : false;
        if (reserved.has(low) || isIdName || (idColumn !== null && lower(idColumn.meta.name) === low)) {
            note?.(
                CSV_LOSS.RESERVED_NAME,
                `${domain} column "${name}" is not written: the name is reserved for the ${isIdName ? "edge id" : low} column`,
                name,
                set,
            );
            continue;
        }
        if (findColumn([name], LABEL_NAMES) >= 0) {
            if (labelColumn !== null) {
                note?.(
                    CSV_LOSS.RESERVED_NAME,
                    `${domain} column "${name}" is not written: the name is reserved for the label column`,
                    name,
                    set,
                );
                continue;
            }
            note?.(
                CSV_LOSS.ROLE_ASSUMED,
                `${domain} column "${name}" reads back with the label role (string)`,
                name,
                set,
            );
        }
        out.push({ column, header: name });
    }
    return out;
}

/**
 * Notes about a role column (id or label): a name the importer does not map to the role, and a
 * dtype other than string / dict (read back as string).
 * @param column - the role column, or null
 * @param role - the role
 * @param names - the header names the importer maps to the role
 * @param domain - node or edge
 * @param rows - the rows written
 * @param note - the recorder
 */
function checkRoleColumn(
    column: Column | null,
    role: "id" | "label",
    names: readonly string[],
    domain: "node" | "edge",
    rows: readonly number[],
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): void {
    if (column === null) {
        return;
    }
    const { name } = column.meta;
    if (findColumn([name], names) < 0) {
        note(
            CSV_LOSS.ROLE_NAME,
            `${domain} column "${name}" (${role}) is written under its name, which the importer does not map to the ${role} role`,
            name,
            countSet(column, rows),
        );
    }
    if (column.dtype !== "string" && column.dtype !== "dict") {
        note(
            CSV_LOSS.TEXT_ROLE,
            `${domain} column "${name}" (${role}) is ${column.dtype}; it reads back as string`,
            name,
            countSet(column, rows),
        );
    }
    checkValues(column, domain, rows, note, true);
}

/**
 * Set rows among the rows written.
 * @param column - the column
 * @param rows - the rows written
 * @returns the count
 */
function countSet(column: Column, rows: readonly number[]): number {
    let count = 0;
    for (const row of rows) {
        if (column.isSet(row)) {
            count++;
        }
    }
    return count;
}

const WIDENING: Readonly<Record<TextDtype, number>> = { bool: 0, i32: 1, f64: 2, string: 3 };
const TEXT_DTYPES: readonly TextDtype[] = ["bool", "i32", "f64", "string"];

/**
 * Value-level notes of one written column over the rows written: non-finite numbers (text on
 * re-import), a text column whose values all read back as numbers or booleans, and the dict
 * heuristic's verdict when it differs from the dtype. A set empty string is written as the quoted
 * empty cell and reads back exactly.
 * @param column - the column
 * @param domain - node or edge
 * @param rows - the rows written
 * @param note - the recorder
 * @param roleColumn - whether the column is declared string on re-import (no inference)
 */
function checkValues(
    column: Column,
    domain: "node" | "edge",
    rows: readonly number[],
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
    roleColumn = false,
): void {
    const { name, dtype } = column.meta;
    const label = `${domain} column "${name}"`;
    if (column.dtype === "f32" || column.dtype === "f64") {
        let nonFinite = 0;
        const { components } = column.meta;
        const { data } = column;
        for (const row of rows) {
            if (!column.isSet(row)) {
                continue;
            }
            for (let k = 0; k < components; k++) {
                if (!Number.isFinite(data[row * components + k])) {
                    nonFinite++;
                    break;
                }
            }
        }
        if (nonFinite > 0) {
            note(CSV_LOSS.NONFINITE, `${label}: ${nonFinite} non-finite value(s) read back as text`, name, nonFinite);
        }
        return;
    }
    if (dtype !== "string" && dtype !== "dict") {
        return;
    }
    let widest = -1;
    let candidate = true;
    let set = 0;
    const heuristic = new DictHeuristic(DICT_SAMPLE_ROWS);
    for (const row of rows) {
        if (!column.isSet(row)) {
            continue;
        }
        const text = column.value(row);
        if (typeof text !== "string") {
            continue;
        }
        set++;
        const kind = inferTextDtype(text);
        widest = Math.max(widest, WIDENING[kind]);
        if (!heuristic.decided) {
            if (kind !== "string") {
                candidate = false;
            }
            heuristic.observe(text);
        }
    }
    if (roleColumn || widest < 0) {
        return;
    }
    const readsAsDict = candidate && heuristic.decide() === "dict";
    if (widest < WIDENING.string && !readsAsDict) {
        note(CSV_LOSS.TEXT_INFERRED, `${label}: every value reads back as ${TEXT_DTYPES[widest]}`, name, set);
        return;
    }
    if (dtype === "dict" && !readsAsDict) {
        note(
            CSV_LOSS.STORAGE_CLASS_CHANGED,
            `${label}: reads back as string (cardinality too high for a dict)`,
            name,
            null,
        );
    } else if (dtype === "string" && readsAsDict) {
        note(CSV_LOSS.STORAGE_CLASS_CHANGED, `${label}: reads back as dict (low cardinality)`, name, null);
    }
}

/**
 * The Type cell of an edge row.
 * @param snapshot - the snapshot
 * @param plan - the plan
 * @param e - the edge
 * @returns "Directed" or "Undirected"
 */
function typeText(snapshot: GraphSnapshot, plan: Plan, e: number): string {
    if (!snapshot.directed || !plan.folding.sourceDirected(e)) {
        return "Undirected";
    }
    return "Directed";
}

/**
 * The lines of the export, one string per row (terminator included).
 * @param snapshot - the snapshot
 * @param plan - the plan
 * @yields one line at a time
 * @returns nothing
 */
function* lines(snapshot: GraphSnapshot, plan: Plan): Generator<string, void, undefined> {
    const { csv } = plan;
    const { delimiter, newline } = csv;
    const quote = (text: string): string => quoteCsvCell(text, delimiter);
    const cell = (column: Column, row: number): string => {
        const text = cellText(column, row);
        return text === null ? "" : quote(text);
    };
    if (csv.table === "nodes") {
        const headers = [csv.dialect.id];
        if (plan.nodeLabel !== null) {
            headers.push(plan.nodeLabel.meta.name);
        }
        for (const { header } of plan.nodeColumns) {
            headers.push(header);
        }
        if (csv.header) {
            yield headers.map(quote).join(delimiter) + newline;
        }
        for (let i = 0; i < snapshot.nodeCount; i++) {
            const cells = [quote(plan.idText(i))];
            if (plan.nodeLabel !== null) {
                cells.push(cell(plan.nodeLabel, i));
            }
            for (const { column } of plan.nodeColumns) {
                cells.push(cell(column, i));
            }
            yield cells.join(delimiter) + newline;
        }
        return;
    }
    const headers = [csv.dialect.source, csv.dialect.target];
    if (csv.dialect.type !== null) {
        headers.push(csv.dialect.type);
    }
    if (plan.edgeId !== null) {
        headers.push(plan.edgeId.meta.name);
    }
    if (plan.edgeLabel !== null) {
        headers.push(plan.edgeLabel.meta.name);
    }
    if (plan.weights.weighted) {
        headers.push(csv.dialect.weight);
    }
    for (const { header } of plan.edgeColumns) {
        headers.push(header);
    }
    if (csv.header) {
        yield headers.map(quote).join(delimiter) + newline;
    }
    const list = snapshot.edgeList();
    for (const e of plan.edgeRows) {
        const cells = [quote(plan.idText(list.src[e])), quote(plan.idText(list.dst[e]))];
        if (csv.dialect.type !== null) {
            cells.push(typeText(snapshot, plan, e));
        }
        if (plan.edgeId !== null) {
            cells.push(cell(plan.edgeId, e));
        }
        if (plan.edgeLabel !== null) {
            cells.push(cell(plan.edgeLabel, e));
        }
        if (plan.weights.weighted) {
            cells.push(plan.weights.text(e) ?? "");
        }
        for (const { column } of plan.edgeColumns) {
            cells.push(cell(column, e));
        }
        yield cells.join(delimiter) + newline;
    }
}

/**
 * Plan an export and refuse it when a fatal note is present (E_INVALID_ID for id text collisions),
 * before anything is written.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @returns the plan
 */
function prepare(snapshot: GraphSnapshot, options: (CsvExportOptions & CommonExportOptions) | undefined): Plan {
    const plan = planExport(snapshot, options, false);
    const collision = plan.notes.find((n) => n.code === CSV_LOSS.ID_TEXT_COLLISION);
    if (collision !== undefined) {
        throw new GraphFormatError("E_INVALID_ID", collision.message, {
            reason: "collision",
            count: collision.count,
        });
    }
    return plan;
}

/**
 * Pre-flight: what a CSV export would lose. The generic notes describe the snapshot as a whole
 * against the format; the CSV notes concern the table selected by `table`.
 * @param snapshot - the snapshot
 * @param options - CSV and common options
 * @returns the notes, empty when the export is exact
 */
function check(snapshot: GraphSnapshot, options?: CsvExportOptions & CommonExportOptions): readonly LossNote[] {
    const plan = planExport(snapshot, options, true);
    return Object.freeze([
        ...checkCapabilities(snapshot, CSV_CAPABILITIES, plan.common, { roles: KEPT_ROLES }),
        ...plan.notes,
    ]);
}

/** The CSV / TSV exporter plugin (subpath `@graphty/graph-io/csv`). */
export const csvExporter: GraphExporter<CsvExportOptions> = Object.freeze({
    format: "csv",
    capabilities: CSV_CAPABILITIES,
    check,
    export(snapshot: GraphSnapshot, options?: CsvExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(lines(snapshot, prepare(snapshot, options)));
    },
    exportToString(snapshot: GraphSnapshot, options?: CsvExportOptions & CommonExportOptions): Promise<string> {
        return joinText(lines(snapshot, prepare(snapshot, options)));
    },
});
