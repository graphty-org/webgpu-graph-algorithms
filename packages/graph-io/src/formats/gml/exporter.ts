/**
 * The GML exporter (design section 8.5, research note 07 sections 2.2 and 9): writes a snapshot in
 * the NetworkX dialect of GML. Node ids must be integers (`idCharset: "integer"`; `sanitizeIds:
 * "mangle"` renumbers the others and keeps the original in a `graphty_originalId` key the importer
 * restores); columns are written per dtype (`int` for the integer dtypes, `real` with a decimal
 * point guaranteed for f32 / f64 so the dtype survives a re-import, quoted strings with `&#NN;`
 * references, repeated keys for lists with the `_networkx_list_start` marker for one-element
 * lists and `"[]"` for empty ones, nested records for json objects); the position role becomes
 * `graphics [ x y z ]` merged with the node's `graphics` json record; graph columns are written
 * inside `graph [ ]` (or at the top level when the importer found them there); `Creator` and
 * `Version` come from the metadata.
 *
 * check() reports, before anything is written, the capability gaps of the common check plus the
 * GML-specific ones: json columns holding numbers (JSON cannot keep the int / real distinction:
 * `W_GML_RECORD_NUMBER_TYPE`), booleans (written 1 / 0) or nulls (omitted), arrays nested in
 * arrays (no GML spelling; export() throws), column names and record keys that are not GML keys
 * or collide with the structural keys (`E_GML_INVALID_KEY` / `E_GML_RESERVED_KEY`, or
 * `W_GML_KEY_MANGLED` under `sanitizeKeys: "mangle"`), and graphics / position overlaps.
 */

import {
    type Column,
    type ColumnMeta,
    GraphFormatError,
    type GraphSnapshot,
    type JsonColumn,
    type NodeId,
} from "@graphty/graph-format";

import { DICT_SAMPLE_ROWS, DictHeuristic } from "../../common/attributes.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { quoteGmlString } from "../../common/escape.js";
import {
    capabilities,
    checkCapabilities,
    countMixedEdges,
    LOSS,
    type SanitizedIds,
    sanitizeIds,
} from "../../common/export.js";
import { formatGmlReal, formatInteger } from "../../common/format.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { type ExplicitWeights, explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import { EMPTY_LIST_TEXT, isGmlKey, LIST_START_MARKER, mangleGmlKey, ORIGINAL_ID_KEY } from "./syntax.js";

/** The format-specific options of the GML exporter. */
export interface GmlExportOptions {
    /**
     * The edge key explicit weights are written under; by default the key the GML importer read
     * them from (`meta.weightOrigin.id` when the snapshot came from GML), else `value`.
     */
    weightKey?: string | undefined;
    /**
     * "error" (default): a column name or record key that is not a GML key (`[A-Za-z][0-9A-Za-z_]*`)
     * or collides with a structural key makes export() throw; "mangle": such keys are rewritten
     * (`.` and other characters become `_`, collisions get a `_2` suffix) and check() reports them.
     */
    sanitizeKeys?: "error" | "mangle" | undefined;
}

/** Loss code: a json column holds numbers; JSON cannot keep GML's int / real distinction (design section 8.5). */
export const RECORD_NUMBER_TYPE_CODE = "W_GML_RECORD_NUMBER_TYPE";
/** Loss code: a json column holds booleans, written as 1 / 0. */
export const RECORD_BOOLEAN_CODE = "W_GML_RECORD_BOOLEAN";
/** Loss code: a json column holds nulls, which GML cannot write; the key (or the row) is omitted. */
export const RECORD_NULL_CODE = "W_GML_RECORD_NULL";
/** Loss code: a json column holds an array inside an array, which GML cannot write; export() throws. */
export const NESTED_ARRAY_CODE = "E_GML_NESTED_ARRAY";
/** Loss code: a json column holds arrays as row values; written as repeated keys, they re-import as a list column. */
export const JSON_ARRAY_CODE = "W_GML_JSON_ARRAY";
/** Loss code: a column name or record key is not a GML key; export() throws unless sanitizeKeys is "mangle". */
export const INVALID_KEY_CODE = "E_GML_INVALID_KEY";
/** Loss code: a column name collides with a structural GML key; export() throws unless sanitizeKeys is "mangle". */
export const RESERVED_KEY_CODE = "E_GML_RESERVED_KEY";
/** Loss code: keys rewritten under sanitizeKeys "mangle". */
export const KEY_MANGLED_CODE = "W_GML_KEY_MANGLED";
/** Loss code: a position column with more than three components; x, y and z are written. */
export const POSITION_COMPONENTS_CODE = "W_GML_POSITION_COMPONENTS";
/** Loss code: a node's graphics record has x / y / z keys the position column replaces. */
export const GRAPHICS_OVERRIDDEN_CODE = "W_GML_GRAPHICS_OVERRIDDEN";
/** Loss code: a node's graphics value is not a record and cannot hold the position; export() throws. */
export const GRAPHICS_CONFLICT_CODE = "E_GML_GRAPHICS_CONFLICT";

/** The default weight key (design section 8.4: GML weights are read from `value` by default). */
export const DEFAULT_WEIGHT_KEY = "value";

/** The roles GML has a key for (`label`, the edge `id` and `key`); every other role is reported. */
const KEPT_ROLES: ReadonlySet<string> = new Set(["label", "id", "key"]);

/** The key the importer maps each kept role back from (the column name after re-import). */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({
    label: "label",
    id: "id",
    key: "key",
});

const GML_CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: false,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "integer",
    dtypes: ["i32", "f64", "string", "dict", "json"],
    components: false,
    lists: true,
    json: true,
    defaults: false,
    options: false,
    hierarchy: false,
    temporal: "none",
    graphAttributes: true,
    positions: true,
    viz: false,
});

/** Roles whose columns are never written as attributes (structural, or dropped per the capabilities). */
const SKIPPED_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "originalId",
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

const NODE_RESERVED: ReadonlySet<string> = new Set(["id"]);
const EDGE_RESERVED: ReadonlySet<string> = new Set(["source", "target"]);
const GRAPH_RESERVED: ReadonlySet<string> = new Set(["node", "edge", "directed", "multigraph"]);
const TOP_RESERVED: ReadonlySet<string> = new Set(["graph"]);
const NUMBER_TEXT = /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/;

/** The resolved options of one export. */
interface GmlExportPlan {
    readonly common: ResolvedExportOptions;
    readonly weightKey: string;
    readonly mangle: boolean;
}

/** What one json column holds that GML cannot write exactly. */
interface JsonStats {
    numbers: number;
    booleans: number;
    nulls: number;
    nestedArrays: number;
    arrays: number;
    invalidKeys: number;
}

/** The kinds of value already counted for one row while inspecting a json value. */
interface SeenFlags {
    numbers: boolean;
    booleans: boolean;
    nulls: boolean;
    nested: boolean;
}

/** The columns of one table selected for writing, with their keys. */
interface WrittenColumn {
    readonly column: Column;
    readonly key: string;
}

/**
 * Resolve the options of one export call.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @returns the plan
 */
function planOf(snapshot: GraphSnapshot, options: (GmlExportOptions & CommonExportOptions) | undefined): GmlExportPlan {
    const common = resolveExportOptions(options);
    const mode = options?.sanitizeKeys ?? "error";
    if (mode !== "error" && mode !== "mangle") {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option sanitizeKeys: ${JSON.stringify(mode)} is not "error" or "mangle"`,
            {
                option: "sanitizeKeys",
                found: mode,
            },
        );
    }
    let weightKey = options?.weightKey;
    if (weightKey === undefined) {
        const origin = snapshot.meta.weightOrigin;
        weightKey = origin !== null && origin.format === "gml" && origin.id !== null ? origin.id : DEFAULT_WEIGHT_KEY;
    } else if (typeof weightKey !== "string" || !isGmlKey(weightKey)) {
        throw new GraphFormatError("E_UNSUPPORTED", `option weightKey: ${JSON.stringify(weightKey)} is not a GML key`, {
            option: "weightKey",
            found: weightKey,
        });
    }
    return { common, weightKey, mangle: mode === "mangle" };
}

/**
 * Whether a column is written as an attribute (its role is neither structural nor dropped).
 * @param column - the column
 * @returns true when written
 */
function isWritten(column: Column): boolean {
    const { role } = column.meta;
    return role === null || !SKIPPED_ROLES.has(role);
}

/**
 * The key a column is written under.
 * @param meta - the column metadata
 * @returns origin.id when the column came from GML (the key it was read from), else the name
 */
function preferredKey(meta: ColumnMeta): string {
    const { origin } = meta;
    return origin !== null && origin.format === "gml" && origin.id !== null ? origin.id : meta.name;
}

/**
 * Whether a snapshot has explicit weights to write.
 * @param snapshot - the snapshot
 * @returns true when weighted
 */
function hasWeights(snapshot: GraphSnapshot): boolean {
    return snapshot.flags.weighted;
}

/**
 * Whether the graph column is a top-level key (the importer's `extra.gmlTopLevel`).
 * @param column - the column
 * @returns true for a top-level key
 */
function isTopLevel(column: Column): boolean {
    return column.meta.extra.gmlTopLevel === true;
}

/**
 * The text of a `Version` value: bare when it is a number text, quoted otherwise.
 * @param text - the version text
 * @returns the GML value
 */
function versionText(text: string): string {
    return NUMBER_TEXT.test(text) ? text : quoteGmlString(text);
}

/**
 * Select the written columns of a table and assign their keys, recording invalid and reserved
 * keys as notes (or rewriting them under mangle).
 * @param table - the columns
 * @param reserved - the structural keys of the table
 * @param label - the table name for messages
 * @param plan - the export plan
 * @param notes - where to record
 * @param filter - an extra selection predicate
 * @returns the written columns with their keys
 */
function selectColumns(
    table: Iterable<Column>,
    reserved: ReadonlySet<string>,
    label: string,
    plan: GmlExportPlan,
    notes: LossNote[] | null,
    filter: (column: Column) => boolean = (): boolean => true,
): WrittenColumn[] {
    const used = new Set<string>(reserved);
    const out: WrittenColumn[] = [];
    for (const column of table) {
        if (!isWritten(column) || !filter(column)) {
            continue;
        }
        const preferred = preferredKey(column.meta);
        let key = preferred;
        const valid = isGmlKey(preferred);
        const taken = used.has(preferred);
        if (!valid || taken) {
            const code = valid ? RESERVED_KEY_CODE : INVALID_KEY_CODE;
            let why = "is not a GML key";
            if (valid) {
                why = reserved.has(preferred) ? "collides with a structural GML key" : "repeats another column's key";
            }
            if (!plan.mangle) {
                if (notes === null) {
                    throw new GraphFormatError(
                        "E_UNSUPPORTED",
                        `${label} column "${column.meta.name}" ${why}; pass sanitizeKeys: "mangle" to rewrite it`,
                        {
                            reason: "gml key",
                            column: column.meta.name,
                            key: preferred,
                        },
                    );
                }
                notes.push(
                    note(
                        code,
                        `${label} column "${column.meta.name}" ${why}; export() will throw unless sanitizeKeys is "mangle"`,
                        column.meta.name,
                    ),
                );
                continue;
            }
            key = uniqueKey(valid ? preferred : mangleGmlKey(preferred), used);
            notes?.push(
                note(
                    KEY_MANGLED_CODE,
                    `${label} column "${column.meta.name}" is written as "${key}"`,
                    column.meta.name,
                ),
            );
        }
        used.add(key);
        out.push({ column, key });
    }
    return out;
}

/**
 * A key not yet used: the base, else base_2, base_3...
 * @param base - a valid key
 * @param used - the keys taken
 * @returns a free key
 */
function uniqueKey(base: string, used: ReadonlySet<string>): string {
    let candidate = base;
    for (let k = 2; used.has(candidate); k++) {
        candidate = `${base}_${k}`;
    }
    return candidate;
}

/**
 * Build a frozen note.
 * @param code - the code
 * @param message - the message
 * @param column - the column, or null
 * @param count - the count, or null
 * @returns the note
 */
function note(code: string, message: string, column: string | null = null, count: number | null = null): LossNote {
    return Object.freeze({ code, message, column, count });
}

/**
 * Inspect a JSON value for what GML cannot write exactly.
 * @param value - the value
 * @param stats - the counters to update (each counted at most once per call for numbers / booleans / nulls)
 * @param inArray - whether the value is an array item
 * @param seen - the flags already counted for this row
 */
function inspectJson(value: unknown, stats: JsonStats, inArray: boolean, seen: SeenFlags): void {
    if (value === null) {
        if (!seen.nulls) {
            seen.nulls = true;
            stats.nulls++;
        }
        return;
    }
    switch (typeof value) {
        case "number":
            if (!seen.numbers) {
                seen.numbers = true;
                stats.numbers++;
            }
            return;
        case "boolean":
            if (!seen.booleans) {
                seen.booleans = true;
                stats.booleans++;
            }
            return;
        case "string":
            return;
        default:
            break;
    }
    if (Array.isArray(value)) {
        if (inArray && !seen.nested) {
            seen.nested = true;
            stats.nestedArrays++;
        }
        for (const item of value) {
            inspectJson(item, stats, true, seen);
        }
        return;
    }
    const record = value as Record<string, unknown>;
    for (const key of Object.keys(record)) {
        if (!isGmlKey(key)) {
            stats.invalidKeys++;
        }
        inspectJson(record[key], stats, false, seen);
    }
}

/**
 * The GML-specific notes of one table's json columns.
 * @param columns - the written columns
 * @param label - the table name
 * @param plan - the export plan
 * @param notes - where to record
 */
function jsonNotes(columns: readonly WrittenColumn[], label: string, plan: GmlExportPlan, notes: LossNote[]): void {
    for (const { column } of columns) {
        const stats = jsonStatsOf(column);
        if (stats === null) {
            continue;
        }
        const { name } = column.meta;
        const where = `${label} column "${name}"`;
        if (stats.numbers > 0) {
            notes.push(
                note(
                    RECORD_NUMBER_TYPE_CODE,
                    `${where} holds numbers in ${stats.numbers} row(s); GML records cannot keep the int / real distinction`,
                    name,
                    stats.numbers,
                ),
            );
        }
        if (stats.booleans > 0) {
            notes.push(
                note(
                    RECORD_BOOLEAN_CODE,
                    `${where} holds booleans in ${stats.booleans} row(s); written as 1 / 0`,
                    name,
                    stats.booleans,
                ),
            );
        }
        if (stats.nulls > 0) {
            notes.push(
                note(
                    RECORD_NULL_CODE,
                    `${where} holds nulls in ${stats.nulls} row(s); GML has no null, the key is omitted`,
                    name,
                    stats.nulls,
                ),
            );
        }
        if (stats.arrays > 0) {
            notes.push(
                note(
                    JSON_ARRAY_CODE,
                    `${where} holds arrays as values in ${stats.arrays} row(s); written as repeated keys, they re-import as a list`,
                    name,
                    stats.arrays,
                ),
            );
        }
        if (stats.nestedArrays > 0) {
            notes.push(
                note(
                    NESTED_ARRAY_CODE,
                    `${where} holds arrays nested in arrays in ${stats.nestedArrays} row(s); GML cannot write them and export() will throw`,
                    name,
                    stats.nestedArrays,
                ),
            );
        }
        if (stats.invalidKeys > 0) {
            notes.push(
                plan.mangle
                    ? note(
                          KEY_MANGLED_CODE,
                          `${where}: ${stats.invalidKeys} record key(s) are not GML keys and are rewritten`,
                          name,
                          stats.invalidKeys,
                      )
                    : note(
                          INVALID_KEY_CODE,
                          `${where}: ${stats.invalidKeys} record key(s) are not GML keys; export() will throw unless sanitizeKeys is "mangle"`,
                          name,
                          stats.invalidKeys,
                      ),
            );
        }
    }
}

/**
 * The json statistics of a json column, or of a list column with json items; null for other dtypes.
 * @param column - the column
 * @returns the stats, or null
 */
function jsonStatsOf(column: Column): JsonStats | null {
    const stats: JsonStats = { numbers: 0, booleans: 0, nulls: 0, nestedArrays: 0, arrays: 0, invalidKeys: 0 };
    if (column.dtype === "json") {
        for (let r = 0; r < column.length; r++) {
            if (!column.isSet(r)) {
                continue;
            }
            const value = column.values[r];
            const seen: SeenFlags = { numbers: false, booleans: false, nulls: false, nested: false };
            if (Array.isArray(value)) {
                stats.arrays++;
                for (const item of value) {
                    inspectJson(item, stats, true, seen);
                }
            } else {
                inspectJson(value, stats, false, seen);
            }
        }
        return stats;
    }
    if (column.dtype === "list" && column.meta.itemDtype === "json") {
        for (let r = 0; r < column.length; r++) {
            if (!column.isSet(r)) {
                continue;
            }
            const seen: SeenFlags = { numbers: false, booleans: false, nulls: false, nested: false };
            for (const item of column.sliceOf(r)) {
                inspectJson(item, stats, true, seen);
            }
        }
        return stats;
    }
    return null;
}

/**
 * The node position column when it can be mapped to graphics x / y / z: a numeric scalar column
 * with the position role.
 * @param snapshot - the snapshot
 * @returns the column, or null
 */
function positionColumn(snapshot: GraphSnapshot): Column | null {
    const column = snapshot.nodes.byRole("position");
    if (column === null) {
        return null;
    }
    switch (column.dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            return column;
        default:
            return null;
    }
}

/**
 * The node `graphics` json column, when present.
 * @param snapshot - the snapshot
 * @returns the column, or null
 */
function graphicsColumn(snapshot: GraphSnapshot): JsonColumn | null {
    return snapshot.nodes.typed("graphics", "json");
}

/**
 * The GML-specific notes about the position / graphics mapping.
 * @param snapshot - the snapshot
 * @param notes - where to record
 */
function graphicsNotes(snapshot: GraphSnapshot, notes: LossNote[]): void {
    const position = positionColumn(snapshot);
    if (position === null) {
        return;
    }
    if (position.meta.components > 3) {
        notes.push(
            note(
                POSITION_COMPONENTS_CODE,
                `node column "${position.meta.name}" has ${position.meta.components} components; only x, y and z are written`,
                position.meta.name,
                position.length - position.nullCount,
            ),
        );
    }
    const graphics = graphicsColumn(snapshot);
    if (graphics === null) {
        return;
    }
    let overridden = 0;
    let conflicts = 0;
    for (let r = 0; r < snapshot.nodeCount; r++) {
        if (!position.isSet(r) || !graphics.isSet(r)) {
            continue;
        }
        const value = graphics.values[r];
        if (typeof value !== "object" || value === null || Array.isArray(value)) {
            conflicts++;
        } else if ("x" in value || "y" in value || "z" in value) {
            overridden++;
        }
    }
    if (overridden > 0) {
        notes.push(
            note(
                GRAPHICS_OVERRIDDEN_CODE,
                `${overridden} node graphics record(s) have x / y / z keys the position column replaces`,
                "graphics",
                overridden,
            ),
        );
    }
    if (conflicts > 0) {
        notes.push(
            note(
                GRAPHICS_CONFLICT_CODE,
                `${conflicts} node graphics value(s) are not records and cannot hold the position; export() will throw`,
                "graphics",
                conflicts,
            ),
        );
    }
}

/**
 * What the importer's own rules change on re-import of the written columns: an all-unset column
 * vanishes (GML writes cells, never declarations), a role-less column named like a role key
 * gains the role, and the dictionary heuristic of design section 5.4 turns a low-cardinality
 * string column into a dict (or a wide dict into a string).
 * @param columns - the written columns
 * @param domain - node or edge
 * @param notes - where to record
 */
function reimportNotes(columns: readonly WrittenColumn[], domain: "node" | "edge", notes: LossNote[]): void {
    for (const { column, key } of columns) {
        const { meta } = column;
        const set = column.length - column.nullCount;
        if (set === 0) {
            notes.push(
                note(
                    LOSS.EMPTY_COLUMN,
                    `${domain} column "${meta.name}" has no set cell and is not written: GML writes cells, never declarations`,
                    meta.name,
                    0,
                ),
            );
            continue;
        }
        if (meta.role === null && (key === "label" || (domain === "edge" && (key === "id" || key === "key")))) {
            notes.push(
                note(
                    LOSS.ROLE_ASSUMED,
                    `${domain} column "${meta.name}" is written under "${key}" and reads back with the ${key === "label" ? "label" : key} role`,
                    meta.name,
                    set,
                ),
            );
            continue;
        }
        if (column.dtype === "list" || !(column.dtype === "string" || column.dtype === "dict")) {
            continue;
        }
        const heuristic = new DictHeuristic(DICT_SAMPLE_ROWS);
        for (let r = 0; r < column.length && !heuristic.decided; r++) {
            if (column.isSet(r)) {
                heuristic.observe(column.value(r) as string);
            }
        }
        const readsAsDict = heuristic.decide() === "dict";
        if (column.dtype === "string" && readsAsDict) {
            notes.push(
                note(
                    LOSS.STORAGE_CLASS,
                    `${domain} column "${meta.name}" reads back as dict (low cardinality)`,
                    meta.name,
                    null,
                ),
            );
        } else if (column.dtype === "dict" && !readsAsDict) {
            notes.push(
                note(
                    LOSS.STORAGE_CLASS,
                    `${domain} column "${meta.name}" reads back as string (cardinality too high for a dict)`,
                    meta.name,
                    null,
                ),
            );
        }
    }
}

/**
 * The importer names the graphics-derived position column `position`, or `position#graphics`
 * when a plain `position` key is written next to it (design section 5.6); a position column
 * named otherwise reads back under that name.
 * @param snapshot - the snapshot
 * @param nodeColumns - the written node columns
 * @param notes - where to record
 */
function positionNameNote(snapshot: GraphSnapshot, nodeColumns: readonly WrittenColumn[], notes: LossNote[]): void {
    const position = positionColumn(snapshot);
    if (position === null) {
        return;
    }
    const { name } = position.meta;
    const plainPosition = nodeColumns.some((c) => c.key === "position");
    const reimportName = plainPosition ? "position#graphics" : "position";
    if (name !== reimportName) {
        notes.push(
            note(
                LOSS.COLUMN_NAME_CHANGED,
                `node column "${name}" (position) is written as the graphics x / y / z keys and reads back as "${reimportName}"`,
                name,
                position.length - position.nullCount,
            ),
        );
    }
}

/**
 * Pre-flight: the common capability notes plus the GML-specific ones.
 * @param snapshot - the snapshot
 * @param options - the options
 * @returns the notes, empty when the export is exact
 */
function check(snapshot: GraphSnapshot, options?: GmlExportOptions & CommonExportOptions): readonly LossNote[] {
    const plan = planOf(snapshot, options);
    const notes = checkCapabilities(snapshot, GML_CAPABILITIES, plan.common, {
        positionDtype: "f64",
        roles: KEPT_ROLES,
        roleNames: ROLE_NAMES,
    });
    const nodeReserved = new Set(NODE_RESERVED);
    if (
        (plan.common.sanitizeIds === "mangle" && countMangled(snapshot) > 0) ||
        snapshot.nodes.byRole("originalId") !== null
    ) {
        nodeReserved.add(ORIGINAL_ID_KEY);
    }
    const edgeIds = snapshot.edges.byRole("id");
    const edgeReserved = edgeReservedKeys(snapshot, plan, edgeIds);
    const topReserved = topReservedKeys(snapshot);
    const nodeColumns = selectColumns(snapshot.nodes, nodeReserved, "node", plan, notes);
    const edgeColumns = selectColumns(snapshot.edges, edgeReserved, "edge", plan, notes, (c) => c !== edgeIds);
    const graphColumns = selectColumns(snapshot.graph, GRAPH_RESERVED, "graph", plan, notes, (c) => !isTopLevel(c));
    const topColumns = selectColumns(snapshot.graph, topReserved, "top-level", plan, notes, isTopLevel);
    jsonNotes(nodeColumns, "node", plan, notes);
    jsonNotes(edgeColumns, "edge", plan, notes);
    jsonNotes(graphColumns, "graph", plan, notes);
    jsonNotes(topColumns, "top-level", plan, notes);
    graphicsNotes(snapshot, notes);
    positionNameNote(snapshot, nodeColumns, notes);
    reimportNotes(nodeColumns, "node", notes);
    reimportNotes(edgeColumns, "edge", notes);
    if (!hasWeights(snapshot)) {
        const clash = edgeColumns.find((c) => c.key === plan.weightKey);
        if (clash !== undefined) {
            notes.push(
                note(
                    LOSS.WEIGHT_KEY_CLASH,
                    `edge column "${clash.column.meta.name}" is written under "${plan.weightKey}", the key the importer reads THE weight from; it reads back as the weight, not as a column`,
                    clash.column.meta.name,
                    clash.column.length - clash.column.nullCount,
                ),
            );
        }
    }
    const folding = pairFolding(snapshot);
    if (folding.mutualCount > 0) {
        notes.push(
            note(
                LOSS.MUTUAL_EXPANDED,
                `${folding.mutualCount} mutual pair(s) are written as two directed edges; the mutual mark is lost`,
                null,
                folding.mutualCount,
            ),
        );
    }
    return Object.freeze(notes);
}

/**
 * The structural keys of the edge table: source, target, the weight key when weights are written,
 * and `id` when a role-id column is written as the edge id.
 * @param snapshot - the snapshot
 * @param plan - the export plan
 * @param edgeIds - the role-id edge column, or null
 * @returns the reserved keys
 */
function edgeReservedKeys(snapshot: GraphSnapshot, plan: GmlExportPlan, edgeIds: Column | null): Set<string> {
    const reserved = new Set(EDGE_RESERVED);
    if (hasWeights(snapshot)) {
        reserved.add(plan.weightKey);
    }
    if (edgeIds !== null) {
        reserved.add("id");
    }
    return reserved;
}

/**
 * The structural keys of the top level: graph, plus Creator and Version when the metadata writes them.
 * @param snapshot - the snapshot
 * @returns the reserved keys
 */
function topReservedKeys(snapshot: GraphSnapshot): Set<string> {
    const reserved = new Set(TOP_RESERVED);
    if (snapshot.meta.creator !== null) {
        reserved.add("Creator");
    }
    if (snapshot.meta.sourceFormat === "gml" && snapshot.meta.sourceVersion !== null) {
        reserved.add("Version");
    }
    return reserved;
}

/**
 * How many node ids are not safe integers (what sanitizeIds "mangle" rewrites).
 * @param snapshot - the snapshot
 * @returns the count
 */
function countMangled(snapshot: GraphSnapshot): number {
    let count = 0;
    for (let i = 0; i < snapshot.nodeCount; i++) {
        const id = snapshot.ids.idOf(i);
        if (typeof id !== "number" || !Number.isSafeInteger(id)) {
            count++;
        }
    }
    return count;
}

// ============================================================ writing

/**
 * The GML text of a real: the shortest text of the dtype with a decimal point guaranteed
 * (`2.0`, `1.0e-7`), the NetworkX spellings `+INF` / `-INF` / `NAN` for the non-finite values.
 * @param value - the value
 * @param dtype - the dtype the value came from (f32 uses the fround-shortest text)
 * @returns the text
 */
export function gmlRealText(value: number, dtype: "f32" | "f64" | "i32" | "u32" | "u8" = "f64"): string {
    return formatGmlReal(value, dtype);
}

/**
 * The GML text of a number by the dtype and origin of its column: an integer text for the integer
 * dtypes and for an f64 column that came from GML `int` values, a real text otherwise.
 * @param value - the value
 * @param meta - the column metadata
 * @returns the text
 */
function numberText(value: number, meta: ColumnMeta): string {
    switch (meta.dtype) {
        case "i32":
        case "u32":
        case "u8":
            return formatInteger(value);
        case "f32":
            return gmlRealText(value, "f32");
        default:
            if (meta.origin?.type === "int" && Number.isInteger(value)) {
                return formatInteger(value);
            }
            return gmlRealText(value, "f64");
    }
}

/**
 * The GML text of a JSON scalar inside a record: integers as int, other numbers as real, strings
 * quoted, booleans as 1 / 0.
 * @param value - the scalar
 * @returns the text
 */
function jsonScalarText(value: number | string | boolean): string {
    switch (typeof value) {
        case "number":
            return Number.isInteger(value) ? formatInteger(value) : gmlRealText(value, "f64");
        case "boolean":
            return value ? "1" : "0";
        default:
            return quoteGmlString(value);
    }
}

/**
 * The GML text of a list item or components lane by the item dtype.
 * @param item - the item
 * @param itemDtype - the list's item dtype
 * @param meta - the list column metadata
 * @returns the text
 */
function itemText(item: unknown, itemDtype: string, meta: ColumnMeta): string {
    switch (typeof item) {
        case "number":
            switch (itemDtype) {
                case "i32":
                case "u32":
                case "u8":
                    return formatInteger(item);
                case "f32":
                    return gmlRealText(item, "f32");
                case "json":
                    return jsonScalarText(item);
                default:
                    return meta.origin?.type === "int" && Number.isInteger(item)
                        ? formatInteger(item)
                        : gmlRealText(item, "f64");
            }
        case "boolean":
            return item ? "1" : "0";
        case "string":
            return quoteGmlString(item);
        default:
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `column "${meta.name}": a ${typeof item} list item cannot be written as GML`,
                {
                    column: meta.name,
                    found: typeof item,
                },
            );
    }
}

/** Writes lines with indentation into an array of parts. */
class GmlWriter {
    private readonly lines: string[] = [];

    private readonly mangle: boolean;

    /**
     * Create a writer.
     * @param mangle - whether record keys are rewritten rather than refused
     */
    constructor(mangle: boolean) {
        this.mangle = mangle;
    }

    /**
     * Take the lines written so far as one string.
     * @returns the text; the writer is empty afterwards
     */
    take(): string {
        const text = this.lines.join("");
        this.lines.length = 0;
        return text;
    }

    /**
     * Write one `key value` line.
     * @param indent - the indentation
     * @param key - the key
     * @param value - the value text
     */
    line(indent: string, key: string, value: string): void {
        this.lines.push(`${indent}${key} ${value}\n`);
    }

    /**
     * Write a raw line.
     * @param text - the line without its terminator
     */
    raw(text: string): void {
        this.lines.push(`${text}\n`);
    }

    /**
     * Write one column cell of a row.
     * @param indent - the indentation
     * @param key - the key
     * @param column - the column
     * @param row - the row
     */
    cell(indent: string, key: string, column: Column, row: number): void {
        const { meta, dtype } = column;
        switch (dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
            case "u8": {
                const { components } = meta;
                if (components === 1) {
                    this.line(indent, key, numberText(column.data[row], meta));
                } else {
                    for (let k = 0; k < components; k++) {
                        this.line(indent, key, numberText(column.data[row * components + k], meta));
                    }
                }
                return;
            }
            case "bool":
                this.line(indent, key, column.value(row) === true ? "1" : "0");
                return;
            case "dict":
            case "string":
                this.line(indent, key, quoteGmlString(column.value(row) ?? ""));
                return;
            case "list":
                this.list(indent, key, column.sliceOf(row), meta.itemDtype ?? "string", meta);
                return;
            case "json":
                this.json(indent, key, column.values[row], meta.name);
                return;
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
    }

    /**
     * Write a list as repeated keys with the NetworkX conventions.
     * @param indent - the indentation
     * @param key - the key
     * @param items - the items
     * @param itemDtype - the item dtype
     * @param meta - the column metadata
     */
    private list(indent: string, key: string, items: readonly unknown[], itemDtype: string, meta: ColumnMeta): void {
        if (items.length === 0) {
            this.line(indent, key, quoteGmlString(EMPTY_LIST_TEXT));
            return;
        }
        if (items.length === 1) {
            this.line(indent, key, quoteGmlString(LIST_START_MARKER));
        }
        for (const item of items) {
            if (itemDtype === "json" && typeof item === "object") {
                this.json(indent, key, item, meta.name, true);
            } else {
                this.line(indent, key, itemText(item, itemDtype, meta));
            }
        }
    }

    /**
     * Write a JSON value: a record for an object, repeated keys for an array, a scalar otherwise;
     * null writes nothing.
     * @param indent - the indentation
     * @param key - the key
     * @param value - the value
     * @param column - the column name, for errors
     * @param inArray - whether the value is an array item (an array here has no GML spelling)
     */
    private json(indent: string, key: string, value: unknown, column: string, inArray = false): void {
        if (value === null || value === undefined) {
            return;
        }
        if (Array.isArray(value)) {
            if (inArray) {
                throw new GraphFormatError(
                    "E_UNSUPPORTED",
                    `column "${column}": an array nested in an array cannot be written as GML`,
                    {
                        reason: "nested array",
                        column,
                    },
                );
            }
            if (value.length === 0) {
                this.line(indent, key, quoteGmlString(EMPTY_LIST_TEXT));
                return;
            }
            if (value.length === 1) {
                this.line(indent, key, quoteGmlString(LIST_START_MARKER));
            }
            for (const item of value) {
                this.json(indent, key, item, column, true);
            }
            return;
        }
        if (typeof value === "object") {
            this.record(indent, key, value as Record<string, unknown>, column);
            return;
        }
        if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") {
            this.line(indent, key, jsonScalarText(value));
            return;
        }
        throw new GraphFormatError("E_COLUMN_TYPE", `column "${column}": a ${typeof value} cannot be written as GML`, {
            column,
            found: typeof value,
        });
    }

    /**
     * Write a record `key [ ... ]`.
     * @param indent - the indentation
     * @param key - the key
     * @param record - the object
     * @param column - the column name, for errors
     * @param extra - lines to write first (the position of a graphics record), or null
     */
    record(
        indent: string,
        key: string,
        record: Record<string, unknown>,
        column: string,
        extra: readonly [string, string][] | null = null,
    ): void {
        this.raw(`${indent}${key} [`);
        const inner = `${indent}  `;
        const used = new Set<string>();
        if (extra !== null) {
            for (const [k, v] of extra) {
                this.line(inner, k, v);
                used.add(k);
            }
        }
        for (const name of Object.keys(record)) {
            if (used.has(name) && extra !== null && (name === "x" || name === "y" || name === "z")) {
                continue;
            }
            const written = this.recordKey(name, used, column);
            used.add(written);
            this.json(inner, written, record[name], column);
        }
        this.raw(`${indent}]`);
    }

    /**
     * The key a record field is written under.
     * @param name - the field name
     * @param used - the keys already written in the record
     * @param column - the column name, for errors
     * @returns a valid, unused key; E_UNSUPPORTED for an invalid key unless mangling
     */
    private recordKey(name: string, used: ReadonlySet<string>, column: string): string {
        if (isGmlKey(name) && !used.has(name)) {
            return name;
        }
        if (!this.mangle) {
            throw new GraphFormatError(
                "E_UNSUPPORTED",
                `column "${column}": record key "${name}" ${isGmlKey(name) ? "repeats a key" : "is not a GML key"}; pass sanitizeKeys: "mangle" to rewrite it`,
                {
                    reason: "gml key",
                    column,
                    key: name,
                },
            );
        }
        return uniqueKey(isGmlKey(name) ? name : mangleGmlKey(name), used);
    }
}

/** Everything the writer needs, resolved once before the first line. */
interface WriteContext {
    readonly snapshot: GraphSnapshot;
    readonly plan: GmlExportPlan;
    readonly ids: SanitizedIds;
    readonly writeDirected: boolean;
    readonly foldPairs: boolean;
    readonly nodeColumns: readonly WrittenColumn[];
    readonly edgeColumns: readonly WrittenColumn[];
    readonly graphColumns: readonly WrittenColumn[];
    readonly topColumns: readonly WrittenColumn[];
    readonly position: Column | null;
    readonly graphics: JsonColumn | null;
    readonly originalIds: Column | null;
    readonly edgeIds: Column | null;
    readonly weights: ExplicitWeights;
    readonly folding: PairFolding;
}

/**
 * Resolve everything export() needs, throwing for what check() reported as an error.
 * @param snapshot - the snapshot
 * @param options - the options
 * @returns the context
 */
function contextOf(
    snapshot: GraphSnapshot,
    options: (GmlExportOptions & CommonExportOptions) | undefined,
): WriteContext {
    const plan = planOf(snapshot, options);
    const ids = sanitizeIds(snapshot, "integer", plan.common.sanitizeIds);
    const mixed = countMixedEdges(snapshot);
    if (mixed > 0 && plan.common.onMixedDirection === "error") {
        throw new GraphFormatError(
            "E_DIRECTED",
            `${mixed} undirected edge(s) in a directed graph; GML has no mixed direction (onMixedDirection: "error")`,
            {
                reason: "mixed direction",
                count: mixed,
            },
        );
    }
    // design 8.5: exporters fold expanded pairs back; under "directed" the folded edge is written
    // once, as one directed edge, under "undirected" the whole graph is written undirected
    const foldPairs = mixed > 0;
    const writeDirected = snapshot.directed && plan.common.onMixedDirection !== "undirected";
    const originalIds = snapshot.nodes.byRole("originalId");
    const nodeReserved = new Set(NODE_RESERVED);
    if (ids.changed > 0 || originalIds !== null) {
        nodeReserved.add(ORIGINAL_ID_KEY);
    }
    const edgeIds = snapshot.edges.byRole("id");
    const edgeReserved = edgeReservedKeys(snapshot, plan, edgeIds);
    const topReserved = topReservedKeys(snapshot);
    return {
        snapshot,
        plan,
        ids,
        writeDirected,
        foldPairs,
        nodeColumns: selectColumns(
            snapshot.nodes,
            nodeReserved,
            "node",
            plan,
            null,
            (c) => !(c.meta.name === "graphics" && c.dtype === "json"),
        ),
        edgeColumns: selectColumns(snapshot.edges, edgeReserved, "edge", plan, null, (c) => c !== edgeIds),
        graphColumns: selectColumns(snapshot.graph, GRAPH_RESERVED, "graph", plan, null, (c) => !isTopLevel(c)),
        topColumns: selectColumns(snapshot.graph, topReserved, "top-level", plan, null, isTopLevel),
        position: positionColumn(snapshot),
        graphics: graphicsColumn(snapshot),
        originalIds,
        edgeIds,
        weights: explicitWeights(snapshot),
        folding: pairFolding(snapshot),
    };
}

/**
 * The GML text of an id written into `graphty_originalId`: quoted for a string, a number text
 * otherwise (an integer as int, else real, so the importer restores the same typed value).
 * @param id - the original id
 * @returns the text
 */
function originalIdText(id: NodeId): string {
    if (typeof id === "string") {
        return quoteGmlString(id);
    }
    return Number.isSafeInteger(id) ? formatInteger(id) : gmlRealText(id, "f64");
}

/**
 * The parts of the document.
 * @param context - the resolved context
 * @yields one string per header line, node or edge
 * @returns nothing
 */
function* gmlParts(context: WriteContext): Generator<string, void, undefined> {
    const { snapshot, plan, ids } = context;
    const w = new GmlWriter(plan.mangle);
    const { meta } = snapshot;
    if (meta.creator !== null) {
        w.line("", "Creator", quoteGmlString(meta.creator));
    }
    if (meta.sourceFormat === "gml" && meta.sourceVersion !== null) {
        w.line("", "Version", versionText(meta.sourceVersion));
    }
    for (const { column, key } of context.topColumns) {
        if (column.isSet(0)) {
            w.cell("", key, column, 0);
        }
    }
    w.raw("graph [");
    w.line("  ", "directed", context.writeDirected ? "1" : "0");
    const multigraph = meta.declaredMultigraph ?? (snapshot.flags.multigraph ? true : null);
    if (multigraph !== null) {
        w.line("  ", "multigraph", multigraph ? "1" : "0");
    }
    for (const { column, key } of context.graphColumns) {
        if (column.isSet(0)) {
            w.cell("  ", key, column, 0);
        }
    }
    yield w.take();
    for (let i = 0; i < snapshot.nodeCount; i++) {
        w.raw("  node [");
        w.line("    ", "id", formatInteger(ids.idAt(i) as number));
        if (ids.isChanged(i)) {
            w.line("    ", ORIGINAL_ID_KEY, originalIdText(ids.originalAt(i)));
        } else if (context.originalIds !== null && context.originalIds.isSet(i)) {
            const original = context.originalIds.value(i);
            if (typeof original === "string" || typeof original === "number") {
                w.line("    ", ORIGINAL_ID_KEY, originalIdText(original));
            }
        }
        for (const { column, key } of context.nodeColumns) {
            if (column.isSet(i)) {
                w.cell("    ", key, column, i);
            }
        }
        writeGraphics(w, context, i);
        w.raw("  ]");
        yield w.take();
    }
    const el = snapshot.edgeList();
    const origin = meta.weightOrigin;
    const integerWeights = origin !== null && origin.type === "int";
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (context.foldPairs && context.folding.folded(e)) {
            continue;
        }
        w.raw("  edge [");
        w.line("    ", "source", formatInteger(ids.idAt(el.src[e]) as number));
        w.line("    ", "target", formatInteger(ids.idAt(el.dst[e]) as number));
        if (context.edgeIds !== null && context.edgeIds.isSet(e)) {
            w.cell("    ", "id", context.edgeIds, e);
        }
        if (context.weights.isExplicit(e)) {
            const weight = context.weights.value(e);
            const text =
                integerWeights && Number.isInteger(weight)
                    ? formatInteger(weight)
                    : gmlRealText(weight, context.weights.dtype);
            w.line("    ", plan.weightKey, text);
        }
        for (const { column, key } of context.edgeColumns) {
            if (column.isSet(e)) {
                w.cell("    ", key, column, e);
            }
        }
        w.raw("  ]");
        yield w.take();
    }
    w.raw("]");
    yield w.take();
}

/**
 * Write a node's `graphics [ ... ]` record from the position column and the graphics json column.
 * @param w - the writer
 * @param context - the context
 * @param i - the node
 */
function writeGraphics(w: GmlWriter, context: WriteContext, i: number): void {
    const { position, graphics } = context;
    const hasPosition = position !== null && position.isSet(i);
    const graphicsValue = graphics !== null && graphics.isSet(i) ? graphics.values[i] : undefined;
    if (!hasPosition) {
        if (graphics !== null && graphicsValue !== undefined) {
            w.cell("    ", "graphics", graphics, i);
        }
        return;
    }
    const numeric = position as Column & { readonly data: ArrayLike<number> };
    const { components } = numeric.meta;
    const lanes = Math.min(components, 3);
    const extra: [string, string][] = [];
    const names = ["x", "y", "z"];
    for (let k = 0; k < lanes; k++) {
        extra.push([names[k], numberText(numeric.data[i * components + k], numeric.meta)]);
    }
    let record: Record<string, unknown> = {};
    if (graphicsValue !== undefined) {
        if (typeof graphicsValue !== "object" || graphicsValue === null || Array.isArray(graphicsValue)) {
            throw new GraphFormatError(
                "E_UNSUPPORTED",
                `node ${i}: the graphics value is not a record and cannot hold the position`,
                {
                    reason: "graphics conflict",
                    node: i,
                },
            );
        }
        record = graphicsValue as Record<string, unknown>;
    }
    w.record("    ", "graphics", record, "graphics", extra);
}

/** The GML exporter plugin. */
export const gmlExporter: GraphExporter<GmlExportOptions> = Object.freeze({
    format: "gml",
    capabilities: GML_CAPABILITIES,
    check,
    /**
     * Write the snapshot as UTF-8 chunks; the context is resolved on the first pull, so the
     * errors check() announced surface from the iteration.
     * @param snapshot - the snapshot
     * @param options - the options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: GmlExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(lazyParts(snapshot, options));
    },
    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot
     * @param options - the options
     * @returns the document
     */
    exportToString(snapshot: GraphSnapshot, options?: GmlExportOptions & CommonExportOptions): Promise<string> {
        return joinText(lazyParts(snapshot, options));
    },
});

/**
 * The document parts, with the context resolved when iteration starts.
 * @param snapshot - the snapshot
 * @param options - the options
 * @yields the parts
 * @returns nothing
 */
function* lazyParts(
    snapshot: GraphSnapshot,
    options: (GmlExportOptions & CommonExportOptions) | undefined,
): Generator<string, void, undefined> {
    yield* gmlParts(contextOf(snapshot, options));
}
