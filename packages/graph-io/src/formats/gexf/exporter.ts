/**
 * The GEXF exporter (design sections 8.5, 3.6, 3.7, 5.1, 5.10; research note 07 section 9): writes
 * GEXF 1.3 by default (1.2 on request), restoring declared attributes from `origin` (id, title,
 * type, defaults, options), the viz namespace from the position / color / size / shape /
 * thickness role columns, containment from the parent / parents columns, element lifetimes from
 * the temporal role columns (with the text companions of design section 5.1), and dynamic
 * attribute values from the temporal extension tables of design section 5.10. Expanded
 * mixed-direction pairs are folded back through the `pair` / `directed` / `mutual` roles into one
 * undirected or mutual edge; weights are written for explicit rows only (the role-weight column's
 * validity, design section 3.7). check() lists every loss before anything is written: the generic
 * capability gaps (W_OPEN_INTERVAL in 1.3, u32 / u8 dtypes, strides, nested json, graph
 * attributes, foreign extension tables) and the GEXF-specific ones (1.2 has no `kind`, no
 * timestamps and no typed lists; role columns of an unexpected shape; roles the format cannot
 * carry; attribute titles the importer would rename).
 */

import { type Column, GraphFormatError, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";

import { mapDeclaredType } from "../../common/declared-types.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { escapeXmlAttribute, escapeXmlText } from "../../common/escape.js";
import { capabilities, checkCapabilities, LOSS, sanitizeIds } from "../../common/export.js";
import { formatF32, formatF64, formatInteger } from "../../common/format.js";
import { isCanonicalIntegerText } from "../../common/ids.js";
import { joinListText } from "../../common/lists.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { formatTemporal, formatTimeValue, type TimeFormat } from "../../common/temporal.js";
import { explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { xmlIllegalTextNotes } from "../../common/xml.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import {
    canonicalScalarType,
    GEXF_FORMAT,
    GEXF_NAMESPACES,
    type GexfEdgeType,
    type GexfVersion,
    isListType,
    isScalarType,
    NODE_COLUMNS,
    OPEN_END,
    OPEN_START,
    parseTemporalTableName,
    RESERVED_EDGE_NAMES,
    RESERVED_NODE_NAMES,
    TEMPORAL_COLUMNS,
    VIZ_NAMESPACE,
} from "./schema.js";

/** The format-specific options of the GEXF exporter. */
export interface GexfExportOptions {
    /** The GEXF version to write: "1.3" (default) or "1.2". */
    version?: GexfVersion | undefined;
}

/** LossNote codes specific to the GEXF exporter, next to the shared LOSS codes. */
export const GEXF_LOSS = Object.freeze({
    /** GEXF 1.2 has no parallel-edge `kind`; the kind column is dropped. */
    KIND_DROPPED: "W_GEXF_KIND_DROPPED",
    /** GEXF 1.2 has no timestamps; a timestamp becomes a closed interval [t, t]. */
    TIMESTAMP_AS_INTERVAL: "W_TIMESTAMP_AS_INTERVAL",
    /** GEXF 1.2 liststring items are separated by `|`; an item containing one cannot be split back. */
    LIST_SEPARATOR: "W_LIST_SEPARATOR",
    /** A role column of a shape GEXF cannot map (a string `start`, a 2-component color); written as a plain attribute. */
    ROLE_SHAPE: "W_ROLE_SHAPE",
    /** A temporal extension table without the element / start / end / value columns of design section 5.10. */
    TEMPORAL_TABLE_SHAPE: "W_TEMPORAL_TABLE_SHAPE",
    /** An attribute whose title the importer would rename on re-import (a reserved name). */
    ATTRIBUTE_RENAMED: "W_ATTRIBUTE_RENAMED",
    /** A cell, default or option the declared type cannot express (skipped). */
    VALUE_UNWRITABLE: "W_VALUE_UNWRITABLE",
    /** A declared type the target version lacks (1.2: date, dateTime, typed lists...); the canonical type is written. */
    DECLARED_TYPE: "W_DECLARED_TYPE",
    /** A node id whose text reads back as the other type under the canonical rule (design section 4.1): a non-integer number, a string of integer text. */
    ID_TEXT_TYPE: LOSS.ID_TEXT_TYPE,
    /** A viz role column (position, color, size, thickness) that is not f32; the importer reads viz values as f32. */
    VIZ_DTYPE: "W_GEXF_VIZ_DTYPE",
    /** A plain `weight` edge column reads back as THE weight (the importer's weightFrom default). */
    WEIGHT_KEY_CLASH: LOSS.WEIGHT_KEY_CLASH,
    /** A dict column without declared options gains one from its dictionary on re-import. */
    OPTIONS_GAINED: LOSS.OPTIONS_GAINED,
    /** A string cell holding a character XML 1.0 forbids; export() throws E_COLUMN_TYPE. */
    XML_ILLEGAL_CHAR: LOSS.XML_ILLEGAL_CHAR,
});

const DTYPES_KEPT = ["f32", "f64", "i32", "bool", "dict", "string"] as const;

/** GEXF 1.3: everything the model has but nested json, strides, graph attributes and open intervals. */
const CAPABILITIES_1_3: ExportCapabilities = capabilities({
    mixedDirection: true,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "any",
    dtypes: DTYPES_KEPT,
    lists: true,
    defaults: true,
    options: true,
    hierarchy: true,
    temporal: "dynamic-values",
    positions: true,
    viz: true,
});

/** GEXF 1.2: no parallel edges, required edge ids, `liststring` only. */
const CAPABILITIES_1_2: ExportCapabilities = capabilities({
    ...CAPABILITIES_1_3,
    multiEdges: false,
    edgeIds: "required",
});

const STRUCTURAL_ROLES: ReadonlySet<string> = new Set(["directed", "pair", "mutual", "weight", "timeText"]);

/** The attribute title the importer reads THE weight from by default (its weightFrom default). */
const DEFAULT_WEIGHT_TITLE = "weight";
const NUMERIC_DTYPES: ReadonlySet<string> = new Set(["f32", "f64", "i32", "u32", "u8"]);
const TEXT_DTYPES: ReadonlySet<string> = new Set(["string", "dict"]);

/** Formats one cell value as GEXF text; null when the value cannot be expressed. */
type ValueFormatter = (value: unknown) => string | null;

/** The GEXF type of a column and how its values are written. */
interface TypedAttribute {
    /** The type text written in the declaration. */
    readonly type: string;
    /** Whether it is a list type. */
    readonly list: boolean;
    /** The item type of a list, the type itself otherwise. */
    readonly itemType: string;
    /** The value formatter. */
    readonly format: ValueFormatter;
    /** A declared type the target version cannot express (written as `type` instead), or null. */
    readonly dropped: string | null;
}

/** A temporal extension table resolved to its columns (design section 5.10). */
interface TemporalTable {
    readonly element: Column;
    readonly start: Column;
    readonly end: Column;
    readonly value: Column;
    readonly startText: Column | null;
    readonly endText: Column | null;
    readonly valueText: Column | null;
    readonly open: Column | null;
    /** Row indices per element index. */
    readonly rows: ReadonlyMap<number, readonly number[]>;
    readonly format: ValueFormatter;
}

/** One `<attribute>` declaration and how its values are written. */
interface AttributeSpec {
    readonly id: string;
    readonly title: string;
    readonly type: string;
    readonly column: Column | null;
    /** The text companion of a temporal-typed column, used only when the written type is temporal. */
    readonly companion: Column | null;
    readonly format: ValueFormatter;
    readonly defaultText: string | null;
    readonly optionsText: string | null;
    readonly dynamic: boolean;
    readonly table: TemporalTable | null;
}

/** The role columns of one domain, each null when absent or of an unusable shape. */
interface RoleColumns {
    label: Column | null;
    id: Column | null;
    kind: Column | null;
    start: Column | null;
    end: Column | null;
    timestamp: Column | null;
    spells: Column | null;
    timestamps: Column | null;
    open: Column | null;
    position: Column | null;
    color: Column | null;
    size: Column | null;
    shape: Column | null;
    shapeUri: Column | null;
    thickness: Column | null;
    parent: Column | null;
    parents: Column | null;
    /** Text companions keyed by the column they accompany. */
    readonly companions: ReadonlyMap<string, Column>;
}

/** Everything check() decides and export() writes from. */
interface ExportPlan {
    readonly version: GexfVersion;
    readonly options: ResolvedExportOptions;
    readonly notes: LossNote[];
    readonly nodeRoles: RoleColumns;
    readonly edgeRoles: RoleColumns;
    readonly nodeAttrs: readonly AttributeSpec[];
    readonly edgeAttrs: readonly AttributeSpec[];
    readonly timeFormat: TimeFormat | null;
    readonly temporal: boolean;
    readonly timeRepresentation: "interval" | "timestamp" | null;
}

/**
 * Resolve the format-specific options.
 * @param options - the caller's options
 * @returns the version to write
 */
function resolveVersion(options: (GexfExportOptions & CommonExportOptions) | undefined): GexfVersion {
    const version = options?.version;
    if (version === undefined) {
        return "1.3";
    }
    if (version !== "1.2" && version !== "1.3") {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option version: ${JSON.stringify(version)} is not "1.2" or "1.3"`,
            {
                option: "version",
                found: version,
                supported: ["1.2", "1.3"],
            },
        );
    }
    return version;
}

/**
 * Decide everything about an export: the capability notes, the role columns of each domain, the
 * attribute declarations and their formatters, the temporal tables and the graph header values.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @returns the plan
 */
function planExport(
    snapshot: GraphSnapshot,
    options: (GexfExportOptions & CommonExportOptions) | undefined,
): ExportPlan {
    const version = resolveVersion(options);
    const resolved = resolveExportOptions(options);
    const caps = version === "1.2" ? CAPABILITIES_1_2 : CAPABILITIES_1_3;
    const notes = [
        ...checkCapabilities(snapshot, caps, resolved, {
            openIntervals: version === "1.2",
            temporalText: true,
            positionDtype: "f32",
            roles: MAPPED_ROLES,
            roleNames: ROLE_NAMES,
        }),
    ];
    const note = (code: string, message: string, column: string | null = null, count: number | null = null): void => {
        notes.push(Object.freeze({ code, message, column, count }));
    };
    notes.push(...xmlIllegalTextNotes(snapshot));
    let typeChanges = 0;
    for (let i = 0; i < snapshot.nodeCount; i++) {
        const id = snapshot.ids.idOf(i);
        if (typeof id === "number" ? !Number.isSafeInteger(id) : isCanonicalIntegerText(id)) {
            typeChanges++;
        }
    }
    if (typeChanges > 0) {
        note(
            GEXF_LOSS.ID_TEXT_TYPE,
            `${typeChanges} node id(s) change type when read back under ids: "canonical" (string ids that are integer text, non-integer numbers); the file's idtype is not honoured by the importer`,
            null,
            typeChanges,
        );
    }
    const tables = collectTemporalTables(snapshot, version, note);
    const nodeRoles = collectRoles(snapshot.nodes, "node", note);
    const edgeRoles = collectRoles(snapshot.edges, "edge", note);
    const nodeAttrs = collectAttributes(snapshot, "node", nodeRoles, tables.get("node") ?? new Map(), version, note);
    const edgeAttrs = collectAttributes(snapshot, "edge", edgeRoles, tables.get("edge") ?? new Map(), version, note);
    const temporal =
        hasLifetime(nodeRoles) ||
        hasLifetime(edgeRoles) ||
        nodeAttrs.some((a) => a.dynamic) ||
        edgeAttrs.some((a) => a.dynamic);
    if (version === "1.2") {
        for (const [domain, roles] of [
            ["node", nodeRoles],
            ["edge", edgeRoles],
        ] as const) {
            if (roles.timestamp !== null || roles.timestamps !== null) {
                note(
                    GEXF_LOSS.TIMESTAMP_AS_INTERVAL,
                    `${domain} timestamps are written as closed intervals; GEXF 1.2 has no timestamp representation`,
                    (roles.timestamp ?? roles.timestamps)?.meta.name ?? null,
                    null,
                );
            }
        }
        if (edgeRoles.kind !== null) {
            note(
                GEXF_LOSS.KIND_DROPPED,
                `edge column "${edgeRoles.kind.meta.name}" (kind) cannot be written; GEXF 1.2 has no edge kind`,
                edgeRoles.kind.meta.name,
                edgeRoles.kind.length - edgeRoles.kind.nullCount,
            );
        }
    }
    const { meta } = snapshot;
    let { timeRepresentation } = meta;
    if (timeRepresentation === null && temporal) {
        const timestampOnly =
            (nodeRoles.timestamp !== null || edgeRoles.timestamp !== null) &&
            nodeRoles.start === null &&
            nodeRoles.end === null &&
            edgeRoles.start === null &&
            edgeRoles.end === null;
        timeRepresentation = timestampOnly ? "timestamp" : null;
    }
    return {
        version,
        options: resolved,
        notes,
        nodeRoles,
        edgeRoles,
        nodeAttrs,
        edgeAttrs,
        timeFormat: meta.timeFormat,
        temporal,
        timeRepresentation,
    };
}

/**
 * Whether a domain carries element lifetimes.
 * @param roles - the domain's role columns
 * @returns true when any temporal role column exists
 */
function hasLifetime(roles: RoleColumns): boolean {
    return (
        roles.start !== null ||
        roles.end !== null ||
        roles.timestamp !== null ||
        roles.spells !== null ||
        roles.timestamps !== null
    );
}

/**
 * Whether a column is a scalar numeric column.
 * @param column - the column
 * @returns true for f32 / f64 / i32 / u32 / u8 with one component
 */
function isNumericScalar(column: Column): boolean {
    return NUMERIC_DTYPES.has(column.dtype) && column.meta.components === 1;
}

/**
 * Whether a column is a numeric column with a given stride.
 * @param column - the column
 * @param components - the accepted strides
 * @returns true when numeric with one of the strides
 */
function isNumericVector(column: Column, components: readonly number[]): boolean {
    return NUMERIC_DTYPES.has(column.dtype) && components.includes(column.meta.components);
}

/**
 * Whether a column holds text (string or dict).
 * @param column - the column
 * @returns true for string / dict
 */
function isText(column: Column): boolean {
    return TEXT_DTYPES.has(column.dtype);
}

/**
 * Whether a column is a list of numbers with a given item stride.
 * @param column - the column
 * @param itemComponents - the required item stride
 * @returns true for a numeric list of that stride
 */
function isNumericList(column: Column, itemComponents: number): boolean {
    return (
        column.dtype === "list" &&
        column.meta.itemDtype !== null &&
        NUMERIC_DTYPES.has(column.meta.itemDtype) &&
        (column.meta.itemComponents ?? 1) === itemComponents
    );
}

/**
 * Whether a role column has the shape its GEXF field needs.
 * @param role - the role
 * @param column - the column
 * @param domain - node or edge
 * @returns true when writable
 */
function roleShapeOk(role: string, column: Column, domain: "node" | "edge"): boolean {
    switch (role) {
        case "label":
        case "shape":
            return isText(column);
        case "kind":
            return domain === "edge" && isText(column);
        case "id":
            return domain === "edge" && (isText(column) || isNumericScalar(column));
        case "start":
        case "end":
        case "timestamp":
        case "open":
            return isNumericScalar(column);
        case "size":
            return domain === "node" && isNumericScalar(column);
        case "thickness":
            return domain === "edge" && isNumericScalar(column);
        case "spells":
            return isNumericList(column, 2);
        case "timestamps":
            return isNumericList(column, 1);
        case "position":
            return domain === "node" && isNumericVector(column, [2, 3]);
        case "color":
            return isNumericVector(column, [3, 4]);
        case "parent":
            return domain === "node" && column.dtype === "u32" && column.meta.components === 1;
        case "parents":
            return domain === "node" && column.dtype === "list" && column.meta.itemDtype === "u32";
        default:
            return false;
    }
}

const VIZ_NUMERIC_ROLES: ReadonlySet<string> = new Set(["position", "color", "size", "thickness"]);

const MAPPED_ROLES: ReadonlySet<string> = new Set([
    "label",
    "id",
    "kind",
    "start",
    "end",
    "timestamp",
    "spells",
    "timestamps",
    "open",
    "position",
    "color",
    "size",
    "shape",
    "thickness",
    "parent",
    "parents",
]);

/** The column name the importer gives each mapped role on re-import (design section 5.6 fixed names). */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({
    label: NODE_COLUMNS.label,
    id: "id",
    kind: "kind",
    start: NODE_COLUMNS.start,
    end: NODE_COLUMNS.end,
    timestamp: NODE_COLUMNS.timestamp,
    spells: NODE_COLUMNS.spells,
    timestamps: NODE_COLUMNS.timestamps,
    open: NODE_COLUMNS.open,
    position: NODE_COLUMNS.position,
    color: NODE_COLUMNS.color,
    size: NODE_COLUMNS.size,
    shape: NODE_COLUMNS.shape,
    thickness: "thickness",
    parent: NODE_COLUMNS.parent,
    parents: NODE_COLUMNS.parents,
});

/**
 * Find the role columns of a table; a role column of the wrong shape is noted and left to the
 * attribute pass.
 * @param table - the node or edge table
 * @param domain - node or edge
 * @param note - the note recorder
 * @returns the role columns
 */
function collectRoles(
    table: Iterable<Column>,
    domain: "node" | "edge",
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): RoleColumns {
    const companions = new Map<string, Column>();
    const roles: RoleColumns = {
        label: null,
        id: null,
        kind: null,
        start: null,
        end: null,
        timestamp: null,
        spells: null,
        timestamps: null,
        open: null,
        position: null,
        color: null,
        size: null,
        shape: null,
        shapeUri: null,
        thickness: null,
        parent: null,
        parents: null,
        companions,
    };
    for (const column of table) {
        const { meta } = column;
        const forName = meta.extra.for;
        if ((meta.role === "timeText" || typeof forName === "string") && column.dtype === "string") {
            if (typeof forName === "string") {
                companions.set(forName, column);
            }
            continue;
        }
        if (domain === "node" && meta.name === NODE_COLUMNS.shapeUri && meta.origin?.namespace === VIZ_NAMESPACE) {
            roles.shapeUri = column.dtype === "string" ? column : null;
            continue;
        }
        const { role } = meta;
        if (role === null || !MAPPED_ROLES.has(role)) {
            continue;
        }
        if (!roleShapeOk(role, column, domain)) {
            note(
                GEXF_LOSS.ROLE_SHAPE,
                `${domain} column "${meta.name}" (${role}) has a shape GEXF cannot map (${describeShape(column)}); written as a plain attribute`,
                meta.name,
                null,
            );
            // the generic check skips role columns; report what the attribute path loses
            const set = column.length - column.nullCount;
            if (NUMERIC_DTYPES.has(column.dtype) && meta.components > 1) {
                note(
                    LOSS.COMPONENTS,
                    `${domain} column "${meta.name}" has ${meta.components} components; written as a list`,
                    meta.name,
                    set,
                );
            }
            if (column.dtype === "json") {
                note(
                    LOSS.JSON,
                    `${domain} column "${meta.name}" holds nested values; written as JSON text`,
                    meta.name,
                    set,
                );
            }
            if (column.dtype === "u32" || column.dtype === "u8") {
                note(
                    LOSS.DTYPE,
                    `${domain} column "${meta.name}" is ${column.dtype}; the format cannot keep that dtype`,
                    meta.name,
                    set,
                );
            }
            continue;
        }
        if (VIZ_NUMERIC_ROLES.has(role) && column.dtype !== "f32") {
            note(
                GEXF_LOSS.VIZ_DTYPE,
                `${domain} column "${meta.name}" (${role}) is ${column.dtype}; viz values read back as f32`,
                meta.name,
                column.length - column.nullCount,
            );
        }
        roles[role as keyof Omit<RoleColumns, "companions">] = column;
    }
    return roles;
}

/**
 * A short description of a column's shape for messages.
 * @param column - the column
 * @returns "dtype x components" or "list of item"
 */
function describeShape(column: Column): string {
    if (column.dtype === "list") {
        return `list of ${column.meta.itemDtype ?? "?"} x ${column.meta.itemComponents ?? 1}`;
    }
    return `${column.dtype} x ${column.meta.components}`;
}

/**
 * Resolve every temporal extension table (design section 5.10) into its columns, grouped by
 * domain and keyed by the static column's name; a table without the required columns is noted.
 * @param snapshot - the snapshot
 * @param version - the target version
 * @param note - the note recorder
 * @returns tables by domain, then by column name
 */
function collectTemporalTables(
    snapshot: GraphSnapshot,
    version: GexfVersion,
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): Map<"node" | "edge", Map<string, TemporalTable>> {
    const out = new Map<"node" | "edge", Map<string, TemporalTable>>();
    for (const [name, table] of snapshot.extensions) {
        const parsed = parseTemporalTableName(name);
        if (parsed === null) {
            continue;
        }
        const element = table.get(TEMPORAL_COLUMNS.element);
        const start = table.get(TEMPORAL_COLUMNS.start);
        const end = table.get(TEMPORAL_COLUMNS.end);
        const value = table.get(TEMPORAL_COLUMNS.value);
        if (
            element === null ||
            element.dtype !== "u32" ||
            start === null ||
            !isNumericScalar(start) ||
            end === null ||
            !isNumericScalar(end) ||
            value === null
        ) {
            note(
                GEXF_LOSS.TEMPORAL_TABLE_SHAPE,
                `extension table "${name}" lacks the element / start / end / value columns and cannot be written`,
                name,
                table.rowCount,
            );
            continue;
        }
        const rows = new Map<number, number[]>();
        for (let r = 0; r < table.rowCount; r++) {
            if (!element.isSet(r)) {
                continue;
            }
            const e = element.value(r) as number;
            const list = rows.get(e);
            if (list === undefined) {
                rows.set(e, [r]);
            } else {
                list.push(r);
            }
        }
        const typed = attributeType(value, version);
        const open = optionalNumeric(table.get(TEMPORAL_COLUMNS.open));
        if (version === "1.3" && open !== null && open.length - open.nullCount > 0) {
            note(
                LOSS.OPEN_INTERVAL,
                `dynamic values of "${name}" carry open intervals, which GEXF 1.3 cannot write`,
                name,
                open.length - open.nullCount,
            );
        }
        const resolved: TemporalTable = {
            element,
            start,
            end,
            value,
            startText: optionalText(table.get(TEMPORAL_COLUMNS.startText)),
            endText: optionalText(table.get(TEMPORAL_COLUMNS.endText)),
            valueText: isTemporalType(typed.type) ? optionalText(table.get(TEMPORAL_COLUMNS.valueText)) : null,
            open,
            rows,
            format: typed.format,
        };
        let byDomain = out.get(parsed.domain);
        if (byDomain === undefined) {
            byDomain = new Map();
            out.set(parsed.domain, byDomain);
        }
        byDomain.set(parsed.column, resolved);
    }
    return out;
}

/**
 * A string column, or null.
 * @param column - the column or null
 * @returns the column when it is a string column
 */
function optionalText(column: Column | null): Column | null {
    return column !== null && column.dtype === "string" ? column : null;
}

/**
 * A numeric scalar column, or null.
 * @param column - the column or null
 * @returns the column when numeric
 */
function optionalNumeric(column: Column | null): Column | null {
    return column !== null && isNumericScalar(column) ? column : null;
}

/**
 * The GEXF type of a column and the formatter of its values: the declared `origin.type` when the
 * version knows it and it agrees with the dtype, the canonical type of the dtype otherwise; lists
 * become `list<item>` (1.3) or `liststring` (1.2); a stride column becomes a list of its lanes;
 * json becomes a string of JSON text.
 * @param column - the column
 * @param version - the target version
 * @returns the type text, whether it is a list, and the formatter
 */
function attributeType(column: Column, version: GexfVersion): TypedAttribute {
    const { meta } = column;
    const originType = meta.origin?.type ?? null;
    if (meta.dtype === "list" || (NUMERIC_DTYPES.has(meta.dtype) && meta.components > 1)) {
        const itemDtype = meta.dtype === "list" ? (meta.itemDtype ?? "string") : meta.dtype;
        let itemType = canonicalScalarType(itemDtype);
        if (
            originType !== null &&
            meta.dtype === "list" &&
            isListType(originType, "1.3") &&
            agrees(originType, "list", itemDtype)
        ) {
            itemType = originType.slice(4);
        }
        if (version === "1.2") {
            const itemFormat = scalarFormatter(itemType, itemDtype);
            return {
                type: "liststring",
                list: true,
                itemType: "string",
                format: listFormatter(itemFormat, version),
                dropped: itemType === "string" ? null : `list${itemType}`,
            };
        }
        return {
            type: `list${itemType}`,
            list: true,
            itemType,
            format: listFormatter(scalarFormatter(itemType, itemDtype), version),
            dropped: null,
        };
    }
    let type = canonicalScalarType(meta.dtype);
    let dropped: string | null = null;
    const temporal = originType === null || meta.dtype !== "f64" ? null : gexfTemporalType(originType);
    if (temporal !== null) {
        // a date / dateTime column of any format (Neo4j spells them date / datetime /
        // localdatetime) keeps its temporal type, so the text companion is written back
        if (isScalarType(temporal, version)) {
            type = temporal;
        } else {
            dropped = temporal;
        }
    } else if (originType !== null && agrees(originType, meta.dtype, null)) {
        if (isScalarType(originType, version)) {
            type = originType;
        } else if (isScalarType(originType, "1.3")) {
            dropped = originType;
        }
    }
    return { type, list: false, itemType: type, format: scalarFormatter(type, meta.dtype), dropped };
}

/**
 * The GEXF temporal type a declared origin type of any format maps to: `date` for a date,
 * `dateTime` for a date-time with or without zone (GEXF's own spellings, Neo4j's `date` /
 * `datetime` / `localdatetime`), null for a non-temporal type or a time of day (GEXF has none).
 * @param originType - the declared type text
 * @returns "date", "dateTime" or null
 */
function gexfTemporalType(originType: string): "date" | "dateTime" | null {
    switch (originType.toLowerCase()) {
        case "date":
            return "date";
        case "datetime":
        case "localdatetime":
            return "dateTime";
        default:
            return null;
    }
}

/**
 * Whether a declared GEXF type maps back to a dtype (so the exporter may restore it verbatim).
 * @param type - the declared type text
 * @param dtype - the column dtype
 * @param itemDtype - the item dtype of a list column, or null
 * @returns true when mapDeclaredType agrees
 */
function agrees(type: string, dtype: string, itemDtype: string | null): boolean {
    const spec = mapDeclaredType("gexf", type, "f64");
    if (spec === null) {
        return false;
    }
    if (dtype === "list") {
        return spec.list && spec.itemDtype === itemDtype;
    }
    if (spec.list) {
        return false;
    }
    // a dict column is a string column with options
    return spec.dtype === dtype || (dtype === "dict" && spec.dtype === "string");
}

/**
 * The formatter of one scalar value under a GEXF type.
 * @param type - the GEXF scalar type text
 * @param dtype - the column (or item) dtype the values come from
 * @returns the formatter
 */
function scalarFormatter(type: string, dtype: string): ValueFormatter {
    switch (type) {
        case "boolean":
            return (value): string | null => {
                if (typeof value === "boolean") {
                    return value ? "true" : "false";
                }
                if (typeof value !== "number") {
                    return null;
                }
                return value !== 0 ? "true" : "false";
            };
        case "integer":
        case "long":
        case "byte":
        case "short":
        case "biginteger":
            return (value): string | null => {
                if (typeof value === "number") {
                    return Number.isInteger(value) ? formatInteger(value) : formatF64(value);
                }
                return typeof value === "string" ? value : null;
            };
        case "float":
            return (value): string | null => (typeof value === "number" ? formatF32(value) : null);
        case "double":
        case "bigdecimal":
            return (value): string | null => {
                if (typeof value === "number") {
                    return dtype === "f32" ? formatF32(value) : formatF64(value);
                }
                return typeof value === "string" ? value : null;
            };
        case "date":
        case "dateTime":
            return (value): string | null => {
                if (typeof value === "number") {
                    return formatTemporal(value, type);
                }
                return typeof value === "string" ? value : null;
            };
        default:
            return (value): string | null => {
                switch (typeof value) {
                    case "string":
                        return value;
                    case "number":
                        return dtype === "f32" ? formatF32(value) : formatF64(value);
                    case "boolean":
                        return value ? "true" : "false";
                    case "object":
                        return value === null ? null : JSON.stringify(value);
                    default:
                        return null;
                }
            };
    }
}

/**
 * The formatter of a list value: items through the item formatter, joined per version.
 * @param item - the item formatter
 * @param version - the target version (1.3 brackets, 1.2 pipes)
 * @returns the formatter; null when any item cannot be written
 */
function listFormatter(item: ValueFormatter, version: GexfVersion): ValueFormatter {
    return (value): string | null => {
        if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
            return null;
        }
        const items: string[] = [];
        for (const entry of Array.from(value as ArrayLike<unknown>)) {
            const text = item(entry);
            if (text === null) {
                return null;
            }
            items.push(text);
        }
        return joinListText(items, version === "1.2" ? "pipe" : "gexf");
    };
}

/**
 * Decide the `<attribute>` declarations of a domain: every column that is neither structural, a
 * companion nor a mapped role column, in declaration order, plus the temporal tables without a
 * static column (the dynamic weight) as dynamic attributes.
 * @param snapshot - the snapshot
 * @param domain - node or edge
 * @param roles - the domain's role columns
 * @param tables - the domain's temporal tables by column name
 * @param version - the target version
 * @param note - the note recorder
 * @returns the declarations
 */
function collectAttributes(
    snapshot: GraphSnapshot,
    domain: "node" | "edge",
    roles: RoleColumns,
    tables: ReadonlyMap<string, TemporalTable>,
    version: GexfVersion,
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): AttributeSpec[] {
    const table = domain === "node" ? snapshot.nodes : snapshot.edges;
    const reserved = domain === "node" ? RESERVED_NODE_NAMES : RESERVED_EDGE_NAMES;
    const roleColumns = new Set<Column>();
    for (const value of Object.values(roles)) {
        if (value !== null && !(value instanceof Map)) {
            roleColumns.add(value);
        }
    }
    const usedIds = new Set<string>();
    const specs: AttributeSpec[] = [];
    const usedTables = new Set<string>();
    const declare = (
        column: Column | null,
        name: string,
        origin: { id: string | null; title: string | null },
        typed: TypedAttribute,
        dynamic: boolean,
        temporal: TemporalTable | null,
    ): void => {
        const id = uniqueId(origin.id ?? name, name, usedIds);
        const title = origin.title ?? name;
        if (typed.dropped !== null) {
            note(
                GEXF_LOSS.DECLARED_TYPE,
                `${domain} column "${name}" is declared ${typed.dropped}, which GEXF ${version} lacks; written as ${typed.type}`,
                name,
                null,
            );
        }
        if (domain === "edge" && column !== null && column.meta.role === null && title === DEFAULT_WEIGHT_TITLE) {
            note(
                GEXF_LOSS.WEIGHT_KEY_CLASH,
                `edge column "${name}" is written as an attribute titled "${title}", which the importer reads as THE weight (weightFrom); it reads back as the weight, not as a column`,
                name,
                column.length - column.nullCount,
            );
        }
        const reimportName = reserved.has(title) ? `${title}#${id}` : title;
        if (reimportName !== name) {
            note(
                GEXF_LOSS.ATTRIBUTE_RENAMED,
                `${domain} column "${name}" is written with title "${title}" and reads back as "${reimportName}"`,
                name,
                null,
            );
        }
        const { defaultText, optionsText } =
            column === null
                ? { defaultText: null, optionsText: null }
                : declaredTexts(column, domain, typed, version, note);
        const companion = column === null ? null : (roles.companions.get(name) ?? null);
        if (companion !== null && !isTemporalType(typed.type)) {
            note(
                LOSS.TEMPORAL_TEXT,
                `${domain} column "${companion.meta.name}" (the lexical form of "${name}") cannot be written: "${name}" is written as ${typed.type}, not a GEXF temporal type`,
                companion.meta.name,
                companion.length - companion.nullCount,
            );
        }
        specs.push({
            id,
            title,
            type: typed.type,
            column,
            companion: isTemporalType(typed.type) ? companion : null,
            format: typed.format,
            defaultText,
            optionsText,
            dynamic,
            table: temporal,
        });
    };
    for (const column of table) {
        const { meta } = column;
        if (roleColumns.has(column) || (meta.role !== null && STRUCTURAL_ROLES.has(meta.role))) {
            continue;
        }
        if (typeof meta.extra.for === "string" && column.dtype === "string") {
            continue;
        }
        if (domain === "node" && column === roles.shapeUri) {
            continue;
        }
        const temporal = tables.get(meta.name) ?? null;
        if (temporal !== null) {
            usedTables.add(meta.name);
        }
        declare(
            column,
            meta.name,
            { id: meta.origin?.id ?? null, title: meta.origin?.title ?? null },
            attributeType(column, version),
            meta.dynamic || temporal !== null,
            temporal,
        );
    }
    for (const [name, temporal] of tables) {
        if (usedTables.has(name)) {
            continue;
        }
        const { origin } = temporal.value.meta;
        const { weightOrigin } = snapshot.meta;
        const weightId =
            domain === "edge" && weightOrigin !== null && (weightOrigin.title ?? weightOrigin.id) === name
                ? weightOrigin.id
                : null;
        declare(
            null,
            name,
            { id: origin?.id ?? weightId, title: null },
            attributeType(temporal.value, version),
            true,
            temporal,
        );
    }
    return specs;
}

/**
 * The `<default>` and `<options>` texts of a declared attribute, with the notes about what cannot
 * be written as the declared type, about a dictionary written as options, and about 1.2 list items
 * holding the `|` separator.
 * @param column - the column
 * @param domain - node or edge
 * @param typed - the attribute's GEXF type and formatter
 * @param version - the target version
 * @param note - the note recorder
 * @returns the default and options texts (null when absent or unwritable)
 */
function declaredTexts(
    column: Column,
    domain: "node" | "edge",
    typed: TypedAttribute,
    version: GexfVersion,
    note: (code: string, message: string, column?: string | null, count?: number | null) => void,
): { defaultText: string | null; optionsText: string | null } {
    const { meta } = column;
    const { name } = meta;
    let defaultText: string | null = null;
    let optionsText: string | null = null;
    if (meta.default !== undefined) {
        defaultText = typed.format(meta.default);
        if (defaultText === null) {
            note(
                GEXF_LOSS.VALUE_UNWRITABLE,
                `${domain} column "${name}": the default cannot be written as ${typed.type}`,
                name,
                null,
            );
        }
    }
    const options = meta.options ?? (column.dtype === "dict" ? column.dictionary : null);
    if (meta.options === null && options !== null && options.length > 0) {
        note(
            GEXF_LOSS.OPTIONS_GAINED,
            `${domain} column "${name}" declares no options; its dictionary is written as <options> and reads back as declared options`,
            name,
            null,
        );
    }
    if (options !== null) {
        const itemFormat = typed.list ? scalarFormatter(typed.itemType, meta.itemDtype ?? "string") : typed.format;
        const texts: string[] = [];
        let ok = true;
        for (const option of options) {
            const text = itemFormat(option);
            if (text === null) {
                ok = false;
                break;
            }
            texts.push(text);
        }
        if (ok) {
            optionsText = joinListText(texts, version === "1.2" ? "pipe" : "gexf");
        } else {
            note(
                GEXF_LOSS.VALUE_UNWRITABLE,
                `${domain} column "${name}": the options cannot be written as ${typed.type}`,
                name,
                null,
            );
        }
    }
    if (version === "1.2" && column.dtype === "list") {
        let count = 0;
        for (let r = 0; r < column.length; r++) {
            if (column.isSet(r) && listItems(column, r).some((item) => String(item).includes("|"))) {
                count++;
            }
        }
        if (count > 0) {
            note(
                GEXF_LOSS.LIST_SEPARATOR,
                `${domain} column "${name}": ${count} row(s) hold an item containing "|", the 1.2 list separator`,
                name,
                count,
            );
        }
    }
    return { defaultText, optionsText };
}

/**
 * Whether a GEXF type is a temporal one whose values have text companions.
 * @param type - the type text
 * @returns true for date / dateTime
 */
function isTemporalType(type: string): boolean {
    return type === "date" || type === "dateTime";
}

/**
 * A unique attribute id within a class.
 * @param preferred - the id to try first (the origin id or the name)
 * @param name - the column name, tried second
 * @param used - ids already taken
 * @returns a free id, recorded as used
 */
function uniqueId(preferred: string, name: string, used: Set<string>): string {
    const candidates = [preferred, name];
    for (const candidate of candidates) {
        if (candidate.length > 0 && !used.has(candidate)) {
            used.add(candidate);
            return candidate;
        }
    }
    for (let n = 2; ; n++) {
        const candidate = `${name}#${n}`;
        if (!used.has(candidate)) {
            used.add(candidate);
            return candidate;
        }
    }
}

// ============================================================ writing

/**
 * The text of a time bound: the companion text when present, the formatted number otherwise.
 * @param column - the numeric column
 * @param companion - its text companion, or null
 * @param row - the row
 * @param timeFormat - the graph's timeformat
 * @returns the text
 */
function timeText(column: Column, companion: Column | null, row: number, timeFormat: TimeFormat | null): string {
    if (companion !== null && companion.isSet(row)) {
        return companion.value(row) as string;
    }
    return formatTimeValue(column.value(row) as number, timeFormat);
}

/**
 * The `start` / `end` / `timestamp` / open attributes of one element.
 * @param roles - the domain's role columns
 * @param row - the element's row
 * @param plan - the plan
 * @returns the attribute text, starting with a space when non-empty
 */
function lifetimeAttrs(roles: RoleColumns, row: number, plan: ExportPlan): string {
    let out = "";
    const { timeFormat, version } = plan;
    let bits = 0;
    if (version === "1.2" && roles.open !== null && roles.open.isSet(row)) {
        bits = roles.open.value(row) as number;
    }
    const write = (name: string, column: Column | null, open: boolean): void => {
        if (column === null || !column.isSet(row)) {
            return;
        }
        const value = column.value(row) as number;
        if (!Number.isFinite(value)) {
            return;
        }
        const text = timeText(column, roles.companions.get(column.meta.name) ?? null, row, timeFormat);
        // GEXF 1.2: `startopen` / `endopen` hold the time of a non-inclusive bound (dynamics.xsd
        // time-type) and replace `start` / `end`
        out += ` ${open ? `${name}open` : name}="${escapeXmlAttribute(text)}"`;
    };
    if (roles.timestamp !== null && roles.timestamp.isSet(row)) {
        if (version === "1.3" && plan.timeRepresentation === "timestamp") {
            write("timestamp", roles.timestamp, false);
        } else {
            write("start", roles.timestamp, false);
            write("end", roles.timestamp, false);
        }
    } else {
        write("start", roles.start, (bits & OPEN_START) !== 0);
        write("end", roles.end, (bits & OPEN_END) !== 0);
    }
    if (version === "1.3" && roles.timestamps !== null && roles.timestamps.isSet(row)) {
        const items = listItems(roles.timestamps, row).map((t) => formatTimeValue(t as number, timeFormat));
        out += ` timestamps="${escapeXmlAttribute(`<[${items.join(", ")}]>`)}"`;
    }
    return out;
}

/**
 * The `<spells>` element of one element: its spells column, plus (1.2) its timestamps as [t, t].
 * @param roles - the domain's role columns
 * @param row - the element's row
 * @param plan - the plan
 * @param indent - the indentation of the element
 * @returns the lines, or an empty string
 */
function spellsElement(roles: RoleColumns, row: number, plan: ExportPlan, indent: string): string {
    const pairs: (readonly [number, number])[] = [];
    if (roles.spells !== null && roles.spells.isSet(row)) {
        for (const pair of listItems(roles.spells, row)) {
            const [s, e] = Array.from(pair as ArrayLike<number>);
            pairs.push([s, e]);
        }
    }
    if (plan.version === "1.2" && roles.timestamps !== null && roles.timestamps.isSet(row)) {
        for (const t of listItems(roles.timestamps, row)) {
            pairs.push([t as number, t as number]);
        }
    }
    if (pairs.length === 0) {
        return "";
    }
    let out = `${indent}<spells>\n`;
    for (const [s, e] of pairs) {
        let attrs = "";
        if (Number.isFinite(s)) {
            attrs += ` start="${escapeXmlAttribute(formatTimeValue(s, plan.timeFormat))}"`;
        }
        if (Number.isFinite(e)) {
            attrs += ` end="${escapeXmlAttribute(formatTimeValue(e, plan.timeFormat))}"`;
        }
        out += `${indent}  <spell${attrs}/>\n`;
    }
    return `${out}${indent}</spells>\n`;
}

/**
 * The `<attvalues>` element of one node or edge: static cells and dynamic rows.
 * @param attrs - the domain's declarations
 * @param row - the element's row
 * @param plan - the plan
 * @param indent - the indentation of the element
 * @returns the lines, or an empty string
 */
function attvaluesElement(attrs: readonly AttributeSpec[], row: number, plan: ExportPlan, indent: string): string {
    let out = "";
    for (const spec of attrs) {
        const { column } = spec;
        if (column !== null && column.isSet(row)) {
            let text: string | null;
            if (spec.companion !== null && spec.companion.isSet(row)) {
                text = spec.companion.value(row) as string;
            } else {
                text = spec.format(cellValue(column, row));
            }
            if (text !== null) {
                out += `${indent}  <attvalue for="${escapeXmlAttribute(spec.id)}" value="${escapeXmlAttribute(text)}"/>\n`;
            }
        }
        const { table } = spec;
        if (table === null) {
            continue;
        }
        const rows = table.rows.get(row);
        if (rows === undefined) {
            continue;
        }
        for (const r of rows) {
            if (!table.value.isSet(r)) {
                continue;
            }
            let text: string | null;
            if (table.valueText !== null && table.valueText.isSet(r)) {
                text = table.valueText.value(r) as string;
            } else {
                text = table.format(cellValue(table.value, r));
            }
            if (text === null) {
                continue;
            }
            out += `${indent}  <attvalue for="${escapeXmlAttribute(spec.id)}" value="${escapeXmlAttribute(text)}"${timedAttrs(table, r, plan)}/>\n`;
        }
    }
    return out.length === 0 ? "" : `${indent}<attvalues>\n${out}${indent}</attvalues>\n`;
}

/**
 * The time bounds of one temporal table row as attributes.
 * @param table - the table
 * @param r - the row
 * @param plan - the plan
 * @returns the attribute text, starting with a space when non-empty
 */
function timedAttrs(table: TemporalTable, r: number, plan: ExportPlan): string {
    const start = table.start.isSet(r) ? (table.start.value(r) as number) : -Infinity;
    const end = table.end.isSet(r) ? (table.end.value(r) as number) : Infinity;
    let out = "";
    if (plan.version === "1.3" && plan.timeRepresentation === "timestamp" && start === end && Number.isFinite(start)) {
        return ` timestamp="${escapeXmlAttribute(timeText(table.start, table.startText, r, plan.timeFormat))}"`;
    }
    let bits = 0;
    if (plan.version === "1.2" && table.open !== null && table.open.isSet(r)) {
        bits = table.open.value(r) as number;
    }
    if (Number.isFinite(start)) {
        const name = (bits & OPEN_START) !== 0 ? "startopen" : "start";
        out += ` ${name}="${escapeXmlAttribute(timeText(table.start, table.startText, r, plan.timeFormat))}"`;
    }
    if (Number.isFinite(end)) {
        const name = (bits & OPEN_END) !== 0 ? "endopen" : "end";
        out += ` ${name}="${escapeXmlAttribute(timeText(table.end, table.endText, r, plan.timeFormat))}"`;
    }
    return out;
}

/**
 * The value of a set cell as a plain JS value: lists through sliceOf, json through values, a
 * stride column as an array of lanes.
 * @param column - the column
 * @param row - the row
 * @returns the value
 */
function cellValue(column: Column, row: number): unknown {
    switch (column.dtype) {
        case "list":
            return column.sliceOf(row);
        case "json":
            return column.values[row];
        default: {
            const value = column.value(row);
            return ArrayBuffer.isView(value) ? Array.from(value as ArrayLike<number>) : value;
        }
    }
}

/**
 * Whether `viz:position` gets a z attribute: always for a 3-d source, and for a 2-d source
 * (`extra.sourceDims === 2`) only when some node has since been given a non-zero z.
 * @param position - the position role column, or null
 * @returns true when z is written
 */
function positionWritesZ(position: Column | null): boolean {
    if (position === null || position.meta.components < 3) {
        return false;
    }
    if (position.meta.extra.sourceDims !== 2) {
        return true;
    }
    const { components } = position.meta;
    let data: ArrayLike<number>;
    switch (position.dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            ({ data } = position);
            break;
        default:
            // A non-numeric position column has no z lane to inspect; write z so nothing is lost.
            return true;
    }
    for (let i = 0; i < position.length; i++) {
        if (position.isSet(i) && data[i * components + 2] !== 0) {
            return true;
        }
    }
    return false;
}

/**
 * The items of a set row of a list column.
 * @param column - a list column
 * @param row - the row
 * @returns the items; empty when the column is not a list
 */
function listItems(column: Column, row: number): readonly unknown[] {
    return column.dtype === "list" ? column.sliceOf(row) : [];
}

/**
 * The viz elements of one node or edge.
 * @param roles - the domain's role columns
 * @param row - the element's row
 * @param indent - the indentation
 * @param writeZ - whether viz:position gets its z attribute
 * @returns the lines, or an empty string
 */
function vizElements(roles: RoleColumns, row: number, indent: string, writeZ: boolean): string {
    let out = "";
    const { color } = roles;
    if (color !== null && color.isSet(row)) {
        const lanes = Array.from(color.value(row) as ArrayLike<number>);
        const scale = color.dtype === "u8" ? 1 : 255;
        const channel = (v: number): string => String(Math.max(0, Math.min(255, Math.round(v * scale))));
        let attrs = ` r="${channel(lanes[0])}" g="${channel(lanes[1])}" b="${channel(lanes[2])}"`;
        if (lanes.length > 3) {
            const a = color.dtype === "u8" ? lanes[3] / 255 : lanes[3];
            if (a !== 1) {
                attrs += ` a="${formatF32(Math.fround(a))}"`;
            }
        }
        out += `${indent}<viz:color${attrs}/>\n`;
    }
    const { position } = roles;
    if (position !== null && position.isSet(row)) {
        const lanes = Array.from(position.value(row) as ArrayLike<number>);
        const fmt = position.dtype === "f32" ? formatF32 : formatF64;
        let attrs = ` x="${fmt(lanes[0])}" y="${fmt(lanes[1])}"`;
        if (lanes.length > 2 && writeZ) {
            attrs += ` z="${fmt(lanes[2])}"`;
        }
        out += `${indent}<viz:position${attrs}/>\n`;
    }
    for (const [name, column] of [
        ["size", roles.size],
        ["thickness", roles.thickness],
    ] as const) {
        if (column !== null && column.isSet(row)) {
            const value = column.value(row) as number;
            const text = column.dtype === "f32" ? formatF32(value) : formatF64(value);
            out += `${indent}<viz:${name} value="${escapeXmlAttribute(text)}"/>\n`;
        }
    }
    const { shape } = roles;
    if (shape !== null && shape.isSet(row)) {
        let attrs = ` value="${escapeXmlAttribute(String(shape.value(row)))}"`;
        if (roles.shapeUri !== null && roles.shapeUri.isSet(row)) {
            attrs += ` uri="${escapeXmlAttribute(roles.shapeUri.value(row) as string)}"`;
        }
        out += `${indent}<viz:shape${attrs}/>\n`;
    }
    return out;
}

/**
 * The `<attributes>` groups of one class.
 * @param cls - node or edge
 * @param specs - the declarations
 * @param indent - the indentation
 * @returns the lines, or an empty string
 */
function attributesGroups(cls: "node" | "edge", specs: readonly AttributeSpec[], indent: string): string {
    let out = "";
    for (const mode of ["static", "dynamic"] as const) {
        const group = specs.filter((s) => s.dynamic === (mode === "dynamic"));
        if (group.length === 0) {
            continue;
        }
        out += `${indent}<attributes class="${cls}" mode="${mode}">\n`;
        for (const spec of group) {
            const head = `${indent}  <attribute id="${escapeXmlAttribute(spec.id)}" title="${escapeXmlAttribute(spec.title)}" type="${spec.type}"`;
            if (spec.defaultText === null && spec.optionsText === null) {
                out += `${head}/>\n`;
                continue;
            }
            out += `${head}>\n`;
            if (spec.defaultText !== null) {
                out += `${indent}    <default>${escapeXmlText(spec.defaultText)}</default>\n`;
            }
            if (spec.optionsText !== null) {
                out += `${indent}    <options>${escapeXmlText(spec.optionsText)}</options>\n`;
            }
            out += `${indent}  </attribute>\n`;
        }
        out += `${indent}</attributes>\n`;
    }
    return out;
}

/**
 * The `type` of a logical edge after folding an expanded pair (design section 3.6).
 * @param snapshot - the snapshot
 * @param e - the logical edge index
 * @param folding - the pair-folding view (mutual pairs fold: GEXF has the mutual type)
 * @returns the edge type, or null when the edge is the mirror half of a pair
 */
function edgeType(snapshot: GraphSnapshot, e: number, folding: PairFolding): GexfEdgeType | null {
    if (!snapshot.directed) {
        return "undirected";
    }
    if (folding.folded(e)) {
        return null;
    }
    if (!folding.sourceDirected(e)) {
        return "undirected";
    }
    if (folding.mateOf(e) !== INVALID_INDEX || folding.isMutual(e)) {
        return "mutual";
    }
    return "directed";
}

/**
 * Write the document as text parts.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @yields one part per element or line group
 * @returns nothing
 */
function* writeGexf(
    snapshot: GraphSnapshot,
    options: (GexfExportOptions & CommonExportOptions) | undefined,
): Generator<string, void, undefined> {
    const plan = planExport(snapshot, options);
    const { version } = plan;
    const ids = sanitizeIds(snapshot, "any", plan.options.sanitizeIds);
    const idText = (index: number): string => escapeXmlAttribute(String(ids.idAt(index)));
    const ns = GEXF_NAMESPACES[version];
    const { meta } = snapshot;

    yield '<?xml version="1.0" encoding="UTF-8"?>\n';
    yield `<gexf xmlns="${ns.gexf}" xmlns:viz="${ns.viz}" version="${version}">\n`;
    yield metaElement(meta);
    yield graphStart(snapshot, plan);
    yield attributesGroups("node", plan.nodeAttrs, "    ");
    yield attributesGroups("edge", plan.edgeAttrs, "    ");

    // nodes
    const { nodeRoles } = plan;
    const writeZ = positionWritesZ(nodeRoles.position);
    yield `    <nodes count="${snapshot.nodeCount}">\n`;
    for (let i = 0; i < snapshot.nodeCount; i++) {
        let attrs = ` id="${idText(i)}"`;
        if (nodeRoles.label !== null && nodeRoles.label.isSet(i)) {
            attrs += ` label="${escapeXmlAttribute(String(nodeRoles.label.value(i)))}"`;
        }
        if (nodeRoles.parent !== null && nodeRoles.parent.isSet(i)) {
            attrs += ` pid="${idText(nodeRoles.parent.value(i) as number)}"`;
        }
        attrs += lifetimeAttrs(nodeRoles, i, plan);
        let body = attvaluesElement(plan.nodeAttrs, i, plan, "        ");
        if (nodeRoles.parents !== null && nodeRoles.parents.isSet(i)) {
            const parents = listItems(nodeRoles.parents, i) as readonly number[];
            if (parents.length > 0) {
                body += "        <parents>\n";
                for (const p of parents) {
                    body += `          <parent for="${idText(p)}"/>\n`;
                }
                body += "        </parents>\n";
            }
        }
        body += spellsElement(nodeRoles, i, plan, "        ");
        body += vizElements(nodeRoles, i, "        ", writeZ);
        yield body.length === 0 ? `      <node${attrs}/>\n` : `      <node${attrs}>\n${body}      </node>\n`;
    }
    yield "    </nodes>\n";

    // edges: the count is decided in a cheap first pass (no strings built) so the section
    // streams element by element instead of buffering every edge
    const { edgeRoles } = plan;
    const edgeList = snapshot.edgeList();
    const folding = pairFolding(snapshot, { foldMutual: true });
    const weights = explicitWeights(snapshot);
    const defaultType: GexfEdgeType = snapshot.directed ? "directed" : "undirected";
    let written = 0;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (edgeType(snapshot, e, folding) !== null) {
            written++;
        }
    }
    yield `    <edges count="${written}">\n`;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        const type = edgeType(snapshot, e, folding);
        if (type === null) {
            continue;
        }
        let attrs = "";
        if (edgeRoles.id !== null && edgeRoles.id.isSet(e)) {
            attrs += ` id="${escapeXmlAttribute(String(edgeRoles.id.value(e)))}"`;
        } else if (version === "1.2") {
            attrs += ` id="e${e}"`;
        }
        attrs += ` source="${idText(edgeList.src[e])}" target="${idText(edgeList.dst[e])}"`;
        if (type !== defaultType) {
            attrs += ` type="${type}"`;
        }
        const weight = weights.text(e);
        if (weight !== null) {
            attrs += ` weight="${weight}"`;
        }
        if (edgeRoles.label !== null && edgeRoles.label.isSet(e)) {
            attrs += ` label="${escapeXmlAttribute(String(edgeRoles.label.value(e)))}"`;
        }
        if (version === "1.3" && edgeRoles.kind !== null && edgeRoles.kind.isSet(e)) {
            attrs += ` kind="${escapeXmlAttribute(String(edgeRoles.kind.value(e)))}"`;
        }
        attrs += lifetimeAttrs(edgeRoles, e, plan);
        let body = attvaluesElement(plan.edgeAttrs, e, plan, "        ");
        body += spellsElement(edgeRoles, e, plan, "        ");
        body += vizElements(edgeRoles, e, "        ", false);
        yield body.length === 0 ? `      <edge${attrs}/>\n` : `      <edge${attrs}>\n${body}      </edge>\n`;
    }
    yield "    </edges>\n";
    yield "  </graph>\n";
    yield "</gexf>\n";
}

/**
 * The `<meta>` element.
 * @param meta - the graph meta
 * @returns the lines, or an empty string when nothing is set
 */
function metaElement(meta: GraphSnapshot["meta"]): string {
    let body = "";
    if (meta.creator !== null) {
        body += `    <creator>${escapeXmlText(meta.creator)}</creator>\n`;
    }
    if (meta.description !== null) {
        body += `    <description>${escapeXmlText(meta.description)}</description>\n`;
    }
    if (meta.keywords.length > 0) {
        body += `    <keywords>${escapeXmlText(meta.keywords.join(", "))}</keywords>\n`;
    }
    const modified = meta.modified === null ? "" : ` lastmodifieddate="${escapeXmlAttribute(meta.modified)}"`;
    if (body.length === 0 && modified.length === 0) {
        return "";
    }
    return body.length === 0 ? `  <meta${modified}/>\n` : `  <meta${modified}>\n${body}  </meta>\n`;
}

/**
 * The `<graph>` start tag with its header attributes.
 * @param snapshot - the snapshot
 * @param plan - the plan
 * @returns the line
 */
function graphStart(snapshot: GraphSnapshot, plan: ExportPlan): string {
    const { meta } = snapshot;
    let attrs = ` defaultedgetype="${snapshot.directed ? "directed" : "undirected"}"`;
    let mode = meta.mode ?? "static";
    if (plan.temporal && mode === "static") {
        mode = "dynamic";
    }
    if (plan.version === "1.2" && mode === "slice") {
        mode = "dynamic";
    }
    attrs += ` mode="${mode}"`;
    let { idType } = meta;
    if (idType === null) {
        const { kind } = snapshot.ids;
        if (kind === "string") {
            idType = "string";
        } else if (kind !== "mixed") {
            idType = "integer";
        }
    }
    if (idType === "integer" || idType === "string") {
        attrs += ` idtype="${idType}"`;
    }
    if (plan.timeFormat !== null) {
        attrs += ` timeformat="${plan.timeFormat}"`;
    } else if (plan.temporal) {
        attrs += ' timeformat="double"';
    }
    if (plan.version === "1.3" && plan.timeRepresentation !== null) {
        attrs += ` timerepresentation="${plan.timeRepresentation}"`;
    }
    const gexfExtra = meta.extra.gexf;
    if (typeof gexfExtra === "object" && gexfExtra !== null) {
        for (const key of ["start", "end", "timestamp"]) {
            const value = (gexfExtra as Record<string, unknown>)[key];
            if (typeof value === "string") {
                attrs += ` ${key}="${escapeXmlAttribute(value)}"`;
            }
        }
    }
    return `  <graph${attrs}>\n`;
}

/** The GEXF exporter (design section 8.5); `capabilities` describes the default 1.3 output. */
export const gexfExporter: GraphExporter<GexfExportOptions> = Object.freeze({
    format: GEXF_FORMAT,
    capabilities: CAPABILITIES_1_3,
    /**
     * Pre-flight: what export() would lose.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the loss notes, empty when the export is exact
     */
    check(snapshot: GraphSnapshot, options?: GexfExportOptions & CommonExportOptions): readonly LossNote[] {
        return Object.freeze(planExport(snapshot, options).notes);
    },
    /**
     * Write the snapshot as UTF-8 chunks.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: GexfExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(writeGexf(snapshot, options));
    },
    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the document
     */
    async exportToString(snapshot: GraphSnapshot, options?: GexfExportOptions & CommonExportOptions): Promise<string> {
        return joinText(writeGexf(snapshot, options));
    },
});

/** The capabilities of a 1.2 export, for callers that pass `version: "1.2"`. */
export const GEXF_1_2_CAPABILITIES: ExportCapabilities = CAPABILITIES_1_2;
