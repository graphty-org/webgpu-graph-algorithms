/**
 * The exporter-side machinery of design section 8.5: the ExportCapabilities table helper, the
 * generic check() that turns capability gaps into LossNotes with shared codes, and id sanitising
 * (`sanitizeIds: "error"` by default; an exporter never silently renames a node; "mangle" rewrites
 * the ids the format cannot hold and hands the exporter the originals to write into its
 * `graphty:originalId` attribute).
 *
 * Every exporter's check() calls checkCapabilities() first and appends its format-specific notes
 * (design section 8.5 names W_GML_RECORD_NUMBER_TYPE and W_OPEN_INTERVAL); export() calls
 * sanitizeIds() before writing anything.
 */

import {
    type Column,
    type Dtype,
    GraphFormatError,
    type GraphSnapshot,
    INVALID_INDEX,
    type NodeId,
} from "@graphty/graph-format";

import { type ExportCapabilities, type LossNote } from "../types.js";
import {
    COLUMN_RENAMED_LOSS_CODE,
    EMPTY_COLUMN_DROPPED_CODE,
    ID_TEXT_COLLISION_CODE,
    ID_TEXT_TYPE_CODE,
    INTEGRAL_F64_CODE,
    MIXED_DIRECTION_CODE,
    MUTUAL_AS_UNDIRECTED_CODE,
    MUTUAL_EXPANDED_CODE,
    OPTIONS_GAINED_CODE,
    PARENTS_DROPPED_CODE,
    ROLE_ASSUMED_CODE,
    ROLE_DROPPED_CODE,
    STORAGE_CLASS_CODE,
    TEMPORAL_DROPPED_CODE,
    TEMPORAL_TEXT_DROPPED_CODE,
    TEXT_INFERRED_CODE,
    WEIGHT_KEY_CLASH_CODE,
    XML_ILLEGAL_CHAR_CODE,
} from "./codes.js";
import { type ResolvedExportOptions } from "./options.js";

/** The shared LossNote codes of checkCapabilities(). */
export const LOSS = Object.freeze({
    /** The format has no mixed direction; expanded pairs are written per onMixedDirection. */
    MIXED_DIRECTION: "W_MIXED_DIRECTION",
    /** The format has no mixed direction and onMixedDirection is "error": export() will throw E_DIRECTED. */
    MIXED_DIRECTION_ERROR: MIXED_DIRECTION_CODE,
    /** Parallel edges in a format without them. */
    MULTI_EDGES: "W_MULTI_EDGES",
    /** Self-loops in a format without them. */
    SELF_LOOPS: "W_SELF_LOOPS",
    /** The format requires edge ids and the snapshot has none; canonical e0..e{E-1} are generated. */
    EDGE_IDS_GENERATED: "W_EDGE_IDS_GENERATED",
    /** The snapshot has an edge id column and the format cannot write one. */
    EDGE_IDS_DROPPED: "W_EDGE_IDS_DROPPED",
    /** Ids outside the format's charset are rewritten (sanitizeIds "mangle"). */
    ID_MANGLED: "W_ID_MANGLED",
    /** Ids outside the format's charset and sanitizeIds "error": export() will throw E_INVALID_ID. */
    ID_CHARSET: "E_ID_CHARSET",
    /** The format numbers nodes 1..N; ids that are not their 1-based index are kept as labels only. */
    ID_RENUMBERED: "W_ID_RENUMBERED",
    /** A column dtype the format does not keep as declared. */
    DTYPE: "W_DTYPE_UNSUPPORTED",
    /** A multi-component column in a format without strides. */
    COMPONENTS: "W_COMPONENTS_FLATTENED",
    /** A list column in a format without lists. */
    LIST: "W_LIST_UNSUPPORTED",
    /** A json column in a format without nested values. */
    JSON: "W_JSON_UNSUPPORTED",
    /** A declared default in a format without defaults. */
    DEFAULT: "W_DEFAULT_DROPPED",
    /** Declared options in a format without enumerations. */
    OPTIONS: "W_OPTIONS_DROPPED",
    /** A parent / parents column in a format without containment. */
    HIERARCHY: "W_HIERARCHY_DROPPED",
    /** A start / end / timestamp column in a format without temporal support. */
    TEMPORAL: TEMPORAL_DROPPED_CODE,
    /** A `<column>.text` companion (design section 5.1) the format cannot carry. */
    TEMPORAL_TEXT: TEMPORAL_TEXT_DROPPED_CODE,
    /** A role column the format has no slot for; written as a plain attribute, the role lost. */
    ROLE: ROLE_DROPPED_CODE,
    /** A parents (multi-parent) column in a format with single containment only. */
    PARENTS: PARENTS_DROPPED_CODE,
    /** A mutual pair written as two directed edges without the mutual mark. */
    MUTUAL_EXPANDED: MUTUAL_EXPANDED_CODE,
    /** A mutual pair written as one undirected edge; the mark is lost and the pair reads back undirected. */
    MUTUAL_AS_UNDIRECTED: MUTUAL_AS_UNDIRECTED_CODE,
    /** Node ids whose written text reads back as the other type under the importer's id rule. */
    ID_TEXT_TYPE: ID_TEXT_TYPE_CODE,
    /** Two node ids share one written text; export() throws E_INVALID_ID. */
    ID_TEXT_COLLISION: ID_TEXT_COLLISION_CODE,
    /** A plain column named like the importer's weight key reads back as THE weight. */
    WEIGHT_KEY_CLASH: WEIGHT_KEY_CLASH_CODE,
    /** A role-less column whose written name the importer maps to a role. */
    ROLE_ASSUMED: ROLE_ASSUMED_CODE,
    /** A role column written into the format's slot reads back under the importer's fixed name. */
    COLUMN_NAME_CHANGED: COLUMN_RENAMED_LOSS_CODE,
    /** A declared column whose every row is unset vanishes through a format without declarations. */
    EMPTY_COLUMN: EMPTY_COLUMN_DROPPED_CODE,
    /** A string / dict column that reads back as the other storage class (the dictionary heuristic). */
    STORAGE_CLASS: STORAGE_CLASS_CODE,
    /** An f64 column whose set values are all integral reads back as i32 through an untyped format. */
    INTEGRAL_F64: INTEGRAL_F64_CODE,
    /** Text cells that read back as numbers or booleans under the 5.1 grammar. */
    TEXT_INFERRED: TEXT_INFERRED_CODE,
    /** A string cell holding a character XML 1.0 forbids; export() throws E_COLUMN_TYPE. */
    XML_ILLEGAL_CHAR: XML_ILLEGAL_CHAR_CODE,
    /** A dict column without declared options gains one on re-import. */
    OPTIONS_GAINED: OPTIONS_GAINED_CODE,
    /** A spells column in a format without spells. */
    SPELLS: "W_SPELLS_DROPPED",
    /** Dynamic attribute values (extension tables) in a format without them. */
    DYNAMIC_VALUES: "W_DYNAMIC_VALUES_DROPPED",
    /** An open-interval column in a format without open intervals (design section 5.1). */
    OPEN_INTERVAL: "W_OPEN_INTERVAL",
    /** Graph-level attributes in a format without them. */
    GRAPH_ATTRIBUTES: "W_GRAPH_ATTRIBUTES_DROPPED",
    /** A position column in a format without positions. */
    POSITIONS: "W_POSITIONS_DROPPED",
    /** Visual columns in a format without them. */
    VIZ: "W_VIZ_DROPPED",
    /** An extension table the format cannot carry. */
    EXTENSION_TABLE: "W_EXTENSION_TABLE_DROPPED",
});

/** The capabilities of a format that keeps nothing beyond plain topology; the base every exporter overrides. */
export const NO_CAPABILITIES: ExportCapabilities = Object.freeze({
    mixedDirection: false,
    multiEdges: false,
    selfLoops: false,
    edgeIds: "none",
    idCharset: "any",
    dtypes: Object.freeze([]),
    components: false,
    lists: false,
    json: false,
    defaults: false,
    options: false,
    hierarchy: false,
    temporal: "none",
    graphAttributes: false,
    positions: false,
    viz: false,
});

/**
 * Build a capabilities table from the fields a format supports; every field left out is the
 * conservative NO_CAPABILITIES value, so an exporter states what it keeps and nothing is assumed.
 * @param supported - the fields the format supports
 * @returns a frozen table
 */
export function capabilities(supported: Partial<ExportCapabilities>): ExportCapabilities {
    return Object.freeze({ ...NO_CAPABILITIES, ...supported, dtypes: Object.freeze([...(supported.dtypes ?? [])]) });
}

/**
 * Format facts checkCapabilities() needs that the 12.4 table does not carry.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface CheckExtras {
    /** Whether the format can write open intervals (GEXF 1.2 startopen / endopen); default false. */
    readonly openIntervals?: boolean | undefined;
    /**
     * Whether the format carries the `<column>.text` companions of design section 5.1 (the lexical
     * form of a temporal value); default false, and every companion is then reported as
     * W_TEMPORAL_TEXT_DROPPED.
     */
    readonly temporalText?: boolean | undefined;
    /**
     * The dtype the format's importer gives a position column; an f32 / f64 position of the other
     * dtype is reported as W_DTYPE_UNSUPPORTED because its values differ at the other precision.
     * Undefined skips the check.
     */
    readonly positionDtype?: Dtype | undefined;
    /**
     * The roles beyond the structural, position, viz, hierarchy and temporal ones that the format
     * has a slot for (`label`, `kind`, `labels`, `idSpace`, `sourcePort`, `targetPort`, ...); a
     * column with any other role is written as a plain attribute and reported as W_ROLE_DROPPED.
     * Defaults to none.
     */
    readonly roles?: ReadonlySet<string> | undefined;
    /**
     * Roles the format writes into a fixed slot whose importer reads them back under a fixed name;
     * a role column named otherwise is reported as W_COLUMN_NAME_CHANGED (role -> the name the
     * importer gives it). Defaults to none.
     */
    readonly roleNames?: Readonly<Record<string, string>> | undefined;
}

/** The roles checkCapabilities() treats structurally rather than as attribute columns. */
const STRUCTURAL_ROLES: ReadonlySet<string> = new Set(["directed", "pair", "mutual", "weight", "originalId"]);
const VIZ_ROLES: ReadonlySet<string> = new Set(["color", "size", "shape", "thickness"]);
const TEMPORAL_ROLES: ReadonlySet<string> = new Set(["start", "end", "timestamp", "timestamps"]);
const HIERARCHY_ROLES: ReadonlySet<string> = new Set(["parent", "parents"]);

/**
 * The generic pre-flight of design section 8.5: compare a snapshot with a format's capabilities
 * and return one LossNote per gap, with the shared codes of LOSS. Format-specific notes are the
 * exporter's to append.
 * @param snapshot - the snapshot about to be exported
 * @param caps - the exporter's capabilities
 * @param options - the resolved common export options
 * @param extras - format facts the capabilities table does not carry
 * @returns the notes, empty when the export is exact
 */
export function checkCapabilities(
    snapshot: GraphSnapshot,
    caps: ExportCapabilities,
    options: ResolvedExportOptions,
    extras: CheckExtras = {},
): LossNote[] {
    const notes: LossNote[] = [];
    const note = (code: string, message: string, column: string | null = null, count: number | null = null): void => {
        notes.push(Object.freeze({ code, message, column, count }));
    };

    // mixed direction (design section 3.6): an expanded snapshot carries the pair / directed roles
    const mixed = countMixedEdges(snapshot);
    if (mixed > 0 && !caps.mixedDirection) {
        if (options.onMixedDirection === "error") {
            note(
                LOSS.MIXED_DIRECTION_ERROR,
                `${mixed} undirected edge(s) in a directed graph; the format has no mixed direction and onMixedDirection is "error"`,
                null,
                mixed,
            );
        } else {
            note(
                LOSS.MIXED_DIRECTION,
                `${mixed} undirected edge(s) in a directed graph written as ${options.onMixedDirection}`,
                null,
                mixed,
            );
        }
    }

    if (snapshot.flags.multigraph && !caps.multiEdges) {
        const parallel = countParallelArcs(snapshot);
        note(LOSS.MULTI_EDGES, `${parallel} parallel edge(s); the format has no parallel edges`, null, parallel);
    }
    if (snapshot.selfLoopCount > 0 && !caps.selfLoops) {
        note(
            LOSS.SELF_LOOPS,
            `${snapshot.selfLoopCount} self-loop(s); the format has no self-loops`,
            null,
            snapshot.selfLoopCount,
        );
    }

    const edgeIds = snapshot.edges.byRole("id");
    if (caps.edgeIds === "required" && edgeIds === null && snapshot.edgeCount > 0) {
        note(
            LOSS.EDGE_IDS_GENERATED,
            "the format requires edge ids; canonical e0..e{E-1} are generated and not written back",
            null,
            snapshot.edgeCount,
        );
    } else if (caps.edgeIds === "none" && edgeIds !== null) {
        note(
            LOSS.EDGE_IDS_DROPPED,
            `edge id column "${edgeIds.meta.name}" cannot be written`,
            edgeIds.meta.name,
            snapshot.edgeCount,
        );
    }

    const unrepresentable = countUnrepresentableIds(snapshot, caps.idCharset);
    if (unrepresentable > 0) {
        if (caps.idCharset === "dense-1-based") {
            note(
                LOSS.ID_RENUMBERED,
                options.sanitizeIds === "mangle"
                    ? `${unrepresentable} node id(s) are not their 1-based index; nodes are numbered 1..N, the originals kept in the exporter's originalId attribute (restored by restoreMangledIds) and as labels of the nodes without a label value`
                    : `${unrepresentable} node id(s) are not their 1-based index; nodes are numbered 1..N and ids kept as labels of the nodes without a label value (a node with one loses its id)`,
                null,
                unrepresentable,
            );
        } else if (options.sanitizeIds === "mangle") {
            note(
                LOSS.ID_MANGLED,
                `${unrepresentable} node id(s) outside the ${caps.idCharset} charset are rewritten; originals kept in the originalId attribute`,
                null,
                unrepresentable,
            );
        } else {
            note(
                LOSS.ID_CHARSET,
                `${unrepresentable} node id(s) outside the ${caps.idCharset} charset; export() will throw unless sanitizeIds is "mangle"`,
                null,
                unrepresentable,
            );
        }
    }

    checkColumns(snapshot.nodes, "node", caps, extras, note);
    checkColumns(snapshot.edges, "edge", caps, extras, note);
    const graphColumns = snapshot.graph.names();
    if (graphColumns.length > 0) {
        if (caps.graphAttributes) {
            checkColumns(snapshot.graph, "graph", caps, extras, note);
        } else {
            note(
                LOSS.GRAPH_ATTRIBUTES,
                `${graphColumns.length} graph attribute(s) cannot be written`,
                null,
                graphColumns.length,
            );
        }
    }

    for (const [name, table] of snapshot.extensions) {
        if (name.startsWith("temporal:")) {
            if (caps.temporal !== "dynamic-values") {
                note(
                    LOSS.DYNAMIC_VALUES,
                    `dynamic values of "${name}" (${table.rowCount} row(s)) cannot be written`,
                    name,
                    table.rowCount,
                );
            }
        } else {
            note(
                LOSS.EXTENSION_TABLE,
                `extension table "${name}" (${table.rowCount} row(s)) cannot be written`,
                name,
                table.rowCount,
            );
        }
    }
    return notes;
}

/** A note-recording callback. */
type NoteFn = (code: string, message: string, column?: string | null, count?: number | null) => void;

/**
 * The per-column checks of checkCapabilities().
 * @param table - the table
 * @param domain - the domain, for messages
 * @param caps - the capabilities
 * @param extras - format extras
 * @param note - the recorder
 */
function checkColumns(
    table: Iterable<Column>,
    domain: string,
    caps: ExportCapabilities,
    extras: CheckExtras,
    note: NoteFn,
): void {
    for (const column of table) {
        const { meta } = column;
        const { name, role } = meta;
        if (role !== null && STRUCTURAL_ROLES.has(role)) {
            continue;
        }
        const label = `${domain} column "${name}"`;
        if (role !== null && extras.roleNames !== undefined) {
            const fixed = extras.roleNames[role];
            if (fixed !== undefined && fixed !== name) {
                note(
                    LOSS.COLUMN_NAME_CHANGED,
                    `${label} (${role}) is written into the format's ${role} slot and reads back as "${fixed}"`,
                    name,
                    column.length - column.nullCount,
                );
            }
        }
        if (domain === "edge" && role === "id") {
            continue;
        }
        if (role === "timeText") {
            if (extras.temporalText !== true) {
                note(
                    LOSS.TEMPORAL_TEXT,
                    `${label} (the lexical form of a temporal column, design section 5.1) cannot be written`,
                    name,
                    column.length - column.nullCount,
                );
            }
            continue;
        }
        if (role === "position") {
            if (!caps.positions) {
                note(LOSS.POSITIONS, `${label} (positions) cannot be written`, name, column.length - column.nullCount);
            } else if (extras.positionDtype !== undefined && meta.dtype !== extras.positionDtype) {
                note(
                    LOSS.DTYPE,
                    `${label} (positions) is ${meta.dtype}; the format reads positions back as ${extras.positionDtype}`,
                    name,
                    column.length - column.nullCount,
                );
            }
            continue;
        }
        if (role !== null && VIZ_ROLES.has(role)) {
            if (!caps.viz) {
                note(LOSS.VIZ, `${label} (${role}) cannot be written`, name, column.length - column.nullCount);
            }
            continue;
        }
        if (role !== null && HIERARCHY_ROLES.has(role)) {
            if (!caps.hierarchy) {
                note(LOSS.HIERARCHY, `${label} (${role}) cannot be written`, name, column.length - column.nullCount);
            }
            continue;
        }
        if (role !== null && TEMPORAL_ROLES.has(role)) {
            if (caps.temporal === "none") {
                note(LOSS.TEMPORAL, `${label} (${role}) cannot be written`, name, column.length - column.nullCount);
            }
            continue;
        }
        if (role === "spells") {
            if (caps.temporal !== "spells" && caps.temporal !== "dynamic-values") {
                note(LOSS.SPELLS, `${label} (spells) cannot be written`, name, column.length - column.nullCount);
            }
            continue;
        }
        if (role === "open") {
            // The open bits are written by the format's own temporal syntax when it has one; the
            // column's u8 storage is never an attribute dtype, so no dtype note either way.
            if (extras.openIntervals !== true) {
                note(
                    LOSS.OPEN_INTERVAL,
                    `${label} (open intervals) cannot be written`,
                    name,
                    column.length - column.nullCount,
                );
            }
            continue;
        }
        if (role !== null && !(extras.roles?.has(role) ?? false)) {
            note(
                LOSS.ROLE,
                `${label} (${role}) is written as a plain attribute; the format has no ${role} slot and the role is lost`,
                name,
                column.length - column.nullCount,
            );
        }
        if (meta.dynamic && caps.temporal !== "dynamic-values") {
            note(LOSS.DYNAMIC_VALUES, `${label} is dynamic; only its static value can be written`, name, null);
        }
        checkDtype(column, label, caps, note);
        if (meta.default !== undefined && !caps.defaults) {
            note(LOSS.DEFAULT, `${label} declares a default; the format has no defaults`, name, null);
        }
        if (meta.options !== null && !caps.options) {
            note(LOSS.OPTIONS, `${label} declares options; the format has no enumerations`, name, null);
        }
    }
}

/**
 * The dtype checks of one attribute column: lists, nested values, scalar dtypes and strides
 * against the capability table.
 * @param column - the column
 * @param label - the column's label for messages
 * @param caps - the capabilities
 * @param note - the recorder
 */
function checkDtype(column: Column, label: string, caps: ExportCapabilities, note: NoteFn): void {
    const { meta } = column;
    const { name } = meta;
    const set = column.length - column.nullCount;
    switch (meta.dtype) {
        case "list":
            if (!caps.lists) {
                note(LOSS.LIST, `${label} is a list; the format has no lists`, name, set);
            } else if (meta.itemDtype !== null && !caps.dtypes.includes(meta.itemDtype)) {
                note(
                    LOSS.DTYPE,
                    `${label} holds ${meta.itemDtype} items; the format cannot keep that dtype`,
                    name,
                    set,
                );
            } else if (meta.itemDtype === "dict") {
                // no format declares an enumeration for list items: they read back as strings
                note(LOSS.DTYPE, `${label} holds dict items; list items read back as string`, name, set);
            }
            break;
        case "json":
            if (!caps.json) {
                note(LOSS.JSON, `${label} holds nested values; the format has no nested values`, name, set);
            }
            break;
        default:
            if (!caps.dtypes.includes(meta.dtype)) {
                note(LOSS.DTYPE, `${label} is ${meta.dtype}; the format cannot keep that dtype`, name, set);
            }
            if (meta.components > 1 && !caps.components) {
                note(
                    LOSS.COMPONENTS,
                    `${label} has ${meta.components} components; the format has no strides`,
                    name,
                    set,
                );
            }
            break;
    }
}

/**
 * The number of source edges that were undirected in a directed snapshot (the halves of an
 * expanded pair count once; an expanded self-loop has no mirror and counts once).
 * @param snapshot - the snapshot
 * @returns the count; 0 for an undirected snapshot or one without the directed role
 */
export function countMixedEdges(snapshot: GraphSnapshot): number {
    if (!snapshot.directed) {
        return 0;
    }
    const directed = snapshot.edges.byRole("directed");
    if (directed === null || directed.dtype !== "bool") {
        return 0;
    }
    const pair = snapshot.edges.byRole("pair");
    let undirectedRows = 0;
    let paired = 0;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (directed.isSet(e) && directed.value(e) === false) {
            undirectedRows++;
            if (pair !== null && pair.dtype === "u32" && pair.isSet(e) && pair.data[e] !== INVALID_INDEX) {
                paired++;
            }
        }
    }
    return paired / 2 + (undirectedRows - paired);
}

/**
 * The number of arcs that repeat the previous arc's target within a row (parallel edges), counted
 * over the forward CSR; an undirected parallel edge is counted in both rows.
 * @param snapshot - the snapshot
 * @returns the count
 */
export function countParallelArcs(snapshot: GraphSnapshot): number {
    const { rowPtr, colIdx, nodeCount } = snapshot;
    let count = 0;
    for (let u = 0; u < nodeCount; u++) {
        for (let a = rowPtr[u] + 1; a < rowPtr[u + 1]; a++) {
            if (colIdx[a] === colIdx[a - 1]) {
                count++;
            }
        }
    }
    return snapshot.directed ? count : Math.ceil(count / 2);
}

// ============================================================ id sanitising

/**
 * The id charsets of ExportCapabilities.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export type IdCharset = ExportCapabilities["idCharset"];

/**
 * The ids to write for every node after sanitising (design section 8.5).
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface SanitizedIds {
    /** The charset applied. */
    readonly charset: IdCharset;
    /** How many ids differ from the snapshot's. */
    readonly changed: number;
    /**
     * The id to write for a node.
     * @param index - the node index
     * @returns the id as the format will read it back
     */
    idAt(index: number): NodeId;
    /**
     * Whether a node's written id differs from its original.
     * @param index - the node index
     * @returns true when the exporter must record the original
     */
    isChanged(index: number): boolean;
    /**
     * The original id of a node.
     * @param index - the node index
     * @returns the snapshot's id
     */
    originalAt(index: number): NodeId;
}

/** XML NameChar ranges (XML 1.0 fifth edition), as inclusive code point pairs. */
const NAME_CHAR_RANGES: readonly (readonly [number, number])[] = [
    [0x2d, 0x2e],
    [0x30, 0x3a],
    [0x41, 0x5a],
    [0x5f, 0x5f],
    [0x61, 0x7a],
    [0xb7, 0xb7],
    [0xc0, 0xd6],
    [0xd8, 0xf6],
    [0xf8, 0x37d],
    [0x37f, 0x1fff],
    [0x200c, 0x200d],
    [0x203f, 0x2040],
    [0x2070, 0x218f],
    [0x2c00, 0x2fef],
    [0x3001, 0xd7ff],
    [0xf900, 0xfdcf],
    [0xfdf0, 0xfffd],
    [0x10000, 0xeffff],
];

/**
 * Whether a code point is an XML NameChar.
 * @param cp - the code point
 * @returns true when it may appear in an NMTOKEN
 */
export function isNameChar(cp: number): boolean {
    for (const [lo, hi] of NAME_CHAR_RANGES) {
        if (cp < lo) {
            return false;
        }
        if (cp <= hi) {
            return true;
        }
    }
    return false;
}

/**
 * Whether a text is an XML NMTOKEN (one or more NameChars; GraphML node and edge ids).
 * @param text - the text
 * @returns true for a non-empty text of NameChars
 */
export function isNmtoken(text: string): boolean {
    if (text.length === 0) {
        return false;
    }
    for (const ch of text) {
        const cp = ch.codePointAt(0);
        if (cp === undefined || !isNameChar(cp)) {
            return false;
        }
    }
    return true;
}

/**
 * Whether an id can be written unchanged under a charset.
 * @param id - the id
 * @param charset - the charset
 * @returns true when the id needs no rewriting ("dense-1-based" is never true: see sanitizeIds)
 */
export function isRepresentableId(id: NodeId, charset: IdCharset): boolean {
    switch (charset) {
        case "any":
            return true;
        case "nmtoken":
            return isNmtoken(String(id));
        case "integer":
            return typeof id === "number" && Number.isSafeInteger(id);
        case "dense-1-based":
            return false;
        default: {
            const name: string = charset;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id charset ${name}`, { charset: name });
        }
    }
}

/**
 * How many node ids a charset cannot hold as they are (for check()); under "dense-1-based" the
 * ids that are not their own 1-based index.
 * @param snapshot - the snapshot
 * @param charset - the charset
 * @returns the count
 */
export function countUnrepresentableIds(snapshot: GraphSnapshot, charset: IdCharset): number {
    const { ids } = snapshot;
    if (charset === "any") {
        return 0;
    }
    if (charset === "dense-1-based") {
        if (ids.kind === "identity" && ids.offset === 1) {
            return 0;
        }
        let changed = 0;
        for (let i = 0; i < ids.size; i++) {
            if (ids.idOf(i) !== i + 1) {
                changed++;
            }
        }
        return changed;
    }
    let count = 0;
    for (let i = 0; i < ids.size; i++) {
        if (!isRepresentableId(ids.idOf(i), charset)) {
            count++;
        }
    }
    return count;
}

/**
 * Sanitise the node ids for a charset (design section 8.5). Under "error" any id the charset
 * cannot hold is E_INVALID_ID (details.reason "charset") and nothing is written; under "mangle"
 * such ids are rewritten deterministically and uniquely: NMTOKEN by replacing every other
 * character with `_` (an empty result becomes `_`) and suffixing `_2`, `_3`... on collision;
 * integer by the smallest unused non-negative integers in index order. "dense-1-based" always
 * writes the 1-based index (Pajek numbers vertices; the exporter writes the original id as the
 * label), which is a renumbering rather than a mangling, so it never throws under "error"; under
 * "mangle" the exporter also writes the original id under its originalId attribute.
 * @param snapshot - the snapshot
 * @param charset - the charset
 * @param mode - the resolved sanitizeIds option
 * @returns the ids to write
 */
export function sanitizeIds(snapshot: GraphSnapshot, charset: IdCharset, mode: "error" | "mangle"): SanitizedIds {
    const { ids } = snapshot;
    const n = ids.size;
    if (charset === "dense-1-based") {
        const changed = countUnrepresentableIds(snapshot, charset);
        return {
            charset,
            changed,
            idAt: (i: number): NodeId => i + 1,
            isChanged: (i: number): boolean => ids.idOf(i) !== i + 1,
            originalAt: (i: number): NodeId => ids.idOf(i),
        };
    }
    const unchanged: SanitizedIds = {
        charset,
        changed: 0,
        idAt: (i: number): NodeId => ids.idOf(i),
        isChanged: (): boolean => false,
        originalAt: (i: number): NodeId => ids.idOf(i),
    };
    if (charset === "any") {
        return unchanged;
    }
    const bad: number[] = [];
    for (let i = 0; i < n; i++) {
        if (!isRepresentableId(ids.idOf(i), charset)) {
            bad.push(i);
        }
    }
    if (bad.length === 0) {
        return unchanged;
    }
    if (mode === "error") {
        const first = ids.idOf(bad[0]);
        throw new GraphFormatError(
            "E_INVALID_ID",
            `${bad.length} node id(s) cannot be written as ${charset} (first: ${JSON.stringify(first)} at index ${bad[0]}); pass sanitizeIds: "mangle" to rewrite them`,
            { reason: "charset", charset, count: bad.length, id: first, index: bad[0] },
        );
    }
    const written = new Map<number, NodeId>();
    if (charset === "nmtoken") {
        const used = new Set<string>();
        for (let i = 0; i < n; i++) {
            const id = ids.idOf(i);
            if (isRepresentableId(id, charset)) {
                used.add(String(id));
            }
        }
        for (const i of bad) {
            const base = mangleNmtoken(String(ids.idOf(i)));
            let candidate = base;
            for (let k = 2; used.has(candidate); k++) {
                candidate = `${base}_${k}`;
            }
            used.add(candidate);
            written.set(i, candidate);
        }
    } else {
        const used = new Set<number>();
        for (let i = 0; i < n; i++) {
            const id = ids.idOf(i);
            if (typeof id === "number" && Number.isSafeInteger(id)) {
                used.add(id);
            }
        }
        let next = 0;
        for (const i of bad) {
            while (used.has(next)) {
                next++;
            }
            used.add(next);
            written.set(i, next);
            next++;
        }
    }
    return {
        charset,
        changed: bad.length,
        idAt: (i: number): NodeId => written.get(i) ?? ids.idOf(i),
        isChanged: (i: number): boolean => written.has(i),
        originalAt: (i: number): NodeId => ids.idOf(i),
    };
}

/**
 * Rewrite a text as an NMTOKEN: every character that is not a NameChar becomes `_`.
 * @param text - the text
 * @returns a non-empty NMTOKEN
 */
export function mangleNmtoken(text: string): string {
    let out = "";
    for (const ch of text) {
        const cp = ch.codePointAt(0);
        out += cp !== undefined && isNameChar(cp) ? ch : "_";
    }
    return out.length === 0 ? "_" : out;
}
