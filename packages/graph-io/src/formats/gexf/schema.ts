/**
 * What the GEXF importer and exporter share (research note 07 section 2.1, design sections 5.1,
 * 5.5, 5.6 and 5.10): the namespaces of the two supported versions, the fixed names and
 * declarations of the columns derived from XML attributes and viz elements (design section 5.6:
 * XML-attribute-derived columns have fixed names and roles, and a declared attribute that would
 * take one of them is renamed `<name>#<id>`), the shape of a temporal extension table, and the
 * declared-type table of each version for the exporter.
 */

import { type ColumnDecl, type Dtype } from "@graphty/graph-format";

import { TIME_TEXT_ROLE, TIME_TEXT_SUFFIX } from "../../common/temporal.js";

/** The GEXF versions the exporter writes. */
export type GexfVersion = "1.2" | "1.3";

/** The format name. */
export const GEXF_FORMAT = "gexf";

/** The namespace URIs of each version. */
export const GEXF_NAMESPACES: Readonly<Record<GexfVersion, { readonly gexf: string; readonly viz: string }>> = {
    "1.2": { gexf: "http://www.gexf.net/1.2draft", viz: "http://www.gexf.net/1.2draft/viz" },
    "1.3": { gexf: "http://gexf.net/1.3", viz: "http://gexf.net/1.3/viz" },
};

/** The default edge type values of `<graph defaultedgetype>` and `<edge type>`. */
export type GexfEdgeType = "directed" | "undirected" | "mutual";

/** The GEXF `timeformat` values. */
export const TIME_FORMATS: ReadonlySet<string> = new Set(["integer", "double", "date", "dateTime"]);

/** The GEXF graph `mode` values. */
export const GRAPH_MODES: ReadonlySet<string> = new Set(["static", "dynamic", "slice"]);

/** The GEXF 1.3 `timerepresentation` values. */
export const TIME_REPRESENTATIONS: ReadonlySet<string> = new Set(["interval", "timestamp"]);

/** The `origin.namespace` of viz-derived columns. */
export const VIZ_NAMESPACE = "viz";

/** The origin every XML-derived column declares. */
const XML_ORIGIN = { format: GEXF_FORMAT };

/** The origin of every viz-derived column. */
const VIZ_ORIGIN = { format: GEXF_FORMAT, namespace: VIZ_NAMESPACE };

/**
 * The fixed node column names the importer owns (design section 5.6): a declared attribute with
 * one of these titles is renamed `<title>#<id>` so the XML-derived column keeps its name and role.
 */
export const NODE_COLUMNS = Object.freeze({
    label: "label",
    parent: "parent",
    parents: "parents",
    start: "start",
    end: "end",
    timestamp: "timestamp",
    spells: "spells",
    timestamps: "timestamps",
    open: "open",
    position: "position",
    color: "color",
    size: "size",
    shape: "shape",
    shapeUri: "shape.uri",
});

/** The fixed edge column names the importer owns. */
const EDGE_COLUMNS = Object.freeze({
    id: "id",
    label: "label",
    kind: "kind",
    start: "start",
    end: "end",
    timestamp: "timestamp",
    spells: "spells",
    timestamps: "timestamps",
    open: "open",
    color: "color",
    thickness: "thickness",
    shape: "shape",
});

/** Every reserved node column name, including the temporal companions. */
export const RESERVED_NODE_NAMES: ReadonlySet<string> = new Set([
    ...Object.values(NODE_COLUMNS),
    NODE_COLUMNS.start + TIME_TEXT_SUFFIX,
    NODE_COLUMNS.end + TIME_TEXT_SUFFIX,
    NODE_COLUMNS.timestamp + TIME_TEXT_SUFFIX,
]);

/** Every reserved edge column name, including the temporal companions. */
export const RESERVED_EDGE_NAMES: ReadonlySet<string> = new Set([
    ...Object.values(EDGE_COLUMNS),
    EDGE_COLUMNS.start + TIME_TEXT_SUFFIX,
    EDGE_COLUMNS.end + TIME_TEXT_SUFFIX,
    EDGE_COLUMNS.timestamp + TIME_TEXT_SUFFIX,
]);

/** The declarations of the XML-derived node columns, keyed as NODE_COLUMNS. */
export const NODE_DECLS: Readonly<Record<keyof typeof NODE_COLUMNS, ColumnDecl>> = {
    label: { name: NODE_COLUMNS.label, dtype: "string", role: "label", nullable: true, origin: XML_ORIGIN },
    parent: {
        name: NODE_COLUMNS.parent,
        dtype: "u32",
        role: "parent",
        refersTo: "node",
        nullable: true,
        origin: XML_ORIGIN,
    },
    parents: {
        name: NODE_COLUMNS.parents,
        dtype: "list",
        itemDtype: "u32",
        role: "parents",
        refersTo: "node",
        nullable: true,
        origin: XML_ORIGIN,
    },
    start: { name: NODE_COLUMNS.start, dtype: "f64", role: "start", nullable: true, origin: XML_ORIGIN },
    end: { name: NODE_COLUMNS.end, dtype: "f64", role: "end", nullable: true, origin: XML_ORIGIN },
    timestamp: { name: NODE_COLUMNS.timestamp, dtype: "f64", role: "timestamp", nullable: true, origin: XML_ORIGIN },
    spells: {
        name: NODE_COLUMNS.spells,
        dtype: "list",
        itemDtype: "f64",
        itemComponents: 2,
        role: "spells",
        nullable: true,
        origin: XML_ORIGIN,
    },
    timestamps: {
        name: NODE_COLUMNS.timestamps,
        dtype: "list",
        itemDtype: "f64",
        role: "timestamps",
        nullable: true,
        origin: XML_ORIGIN,
    },
    open: { name: NODE_COLUMNS.open, dtype: "u8", role: "open", nullable: true, origin: XML_ORIGIN },
    position: {
        name: NODE_COLUMNS.position,
        dtype: "f32",
        components: 3,
        role: "position",
        mutable: true,
        nullable: true,
        origin: VIZ_ORIGIN,
        extra: { units: "file" },
    },
    color: {
        name: NODE_COLUMNS.color,
        dtype: "f32",
        components: 4,
        role: "color",
        nullable: true,
        origin: VIZ_ORIGIN,
    },
    size: { name: NODE_COLUMNS.size, dtype: "f32", role: "size", nullable: true, origin: VIZ_ORIGIN },
    shape: { name: NODE_COLUMNS.shape, dtype: "dict", role: "shape", nullable: true, origin: VIZ_ORIGIN },
    shapeUri: { name: NODE_COLUMNS.shapeUri, dtype: "string", nullable: true, origin: VIZ_ORIGIN },
};

/** The declarations of the XML-derived edge columns, keyed as EDGE_COLUMNS. */
export const EDGE_DECLS: Readonly<Record<keyof typeof EDGE_COLUMNS, ColumnDecl>> = {
    id: { name: EDGE_COLUMNS.id, dtype: "string", role: "id", unique: true, nullable: true, origin: XML_ORIGIN },
    label: { name: EDGE_COLUMNS.label, dtype: "string", role: "label", nullable: true, origin: XML_ORIGIN },
    kind: { name: EDGE_COLUMNS.kind, dtype: "dict", role: "kind", nullable: true, origin: XML_ORIGIN },
    start: { name: EDGE_COLUMNS.start, dtype: "f64", role: "start", nullable: true, origin: XML_ORIGIN },
    end: { name: EDGE_COLUMNS.end, dtype: "f64", role: "end", nullable: true, origin: XML_ORIGIN },
    timestamp: { name: EDGE_COLUMNS.timestamp, dtype: "f64", role: "timestamp", nullable: true, origin: XML_ORIGIN },
    spells: {
        name: EDGE_COLUMNS.spells,
        dtype: "list",
        itemDtype: "f64",
        itemComponents: 2,
        role: "spells",
        nullable: true,
        origin: XML_ORIGIN,
    },
    timestamps: {
        name: EDGE_COLUMNS.timestamps,
        dtype: "list",
        itemDtype: "f64",
        role: "timestamps",
        nullable: true,
        origin: XML_ORIGIN,
    },
    open: { name: EDGE_COLUMNS.open, dtype: "u8", role: "open", nullable: true, origin: XML_ORIGIN },
    color: {
        name: EDGE_COLUMNS.color,
        dtype: "f32",
        components: 4,
        role: "color",
        nullable: true,
        origin: VIZ_ORIGIN,
    },
    thickness: { name: EDGE_COLUMNS.thickness, dtype: "f32", role: "thickness", nullable: true, origin: VIZ_ORIGIN },
    shape: { name: EDGE_COLUMNS.shape, dtype: "dict", role: "shape", nullable: true, origin: VIZ_ORIGIN },
};

/** The bit of the `open` column marking an open start (design section 5.1). */
export const OPEN_START = 1;

/** The bit of the `open` column marking an open end. */
export const OPEN_END = 2;

/** The prefix of a temporal extension table's name (design section 5.10). */
const TEMPORAL_TABLE_PREFIX = "temporal:";

/** The column names of a temporal extension table (design section 5.10). */
export const TEMPORAL_COLUMNS = Object.freeze({
    element: "element",
    start: "start",
    end: "end",
    value: "value",
    open: "open",
    startText: `start${TIME_TEXT_SUFFIX}`,
    endText: `end${TIME_TEXT_SUFFIX}`,
    valueText: `value${TIME_TEXT_SUFFIX}`,
});

/**
 * The name of the temporal extension table of a dynamic attribute.
 * @param domain - node or edge
 * @param column - the static column's name
 * @returns `temporal:<domain>:<column>`
 */
export function temporalTableName(domain: "node" | "edge", column: string): string {
    return `${TEMPORAL_TABLE_PREFIX}${domain}:${column}`;
}

/**
 * Split a temporal table name into its domain and column.
 * @param name - the table name
 * @returns the parts, or null when the name is not a temporal table's
 */
export function parseTemporalTableName(name: string): { domain: "node" | "edge"; column: string } | null {
    if (!name.startsWith(TEMPORAL_TABLE_PREFIX)) {
        return null;
    }
    const rest = name.slice(TEMPORAL_TABLE_PREFIX.length);
    const colon = rest.indexOf(":");
    if (colon < 0) {
        return null;
    }
    const domain = rest.slice(0, colon);
    if (domain !== "node" && domain !== "edge") {
        return null;
    }
    return { domain, column: rest.slice(colon + 1) };
}

/**
 * The column declarations of a temporal extension table (design section 5.10): the element index,
 * the interval, the value in the attribute's dtype, the interval's text companions, an open-bounds
 * byte and, for a temporal-typed attribute, the value's text companion.
 * @param domain - the domain the element column refers to
 * @param value - the value column's declaration fields (dtype, itemDtype, options...)
 * @param temporalValue - whether the value is itself a temporal type needing a text companion
 * @returns the declarations in table order
 */
export function temporalTableDecls(
    domain: "node" | "edge",
    value: Pick<ColumnDecl, "dtype" | "itemDtype" | "options" | "origin">,
    temporalValue: boolean,
): ColumnDecl[] {
    const decls: ColumnDecl[] = [
        { name: TEMPORAL_COLUMNS.element, dtype: "u32", refersTo: domain, nullable: false },
        { name: TEMPORAL_COLUMNS.start, dtype: "f64", role: "start", nullable: true },
        { name: TEMPORAL_COLUMNS.end, dtype: "f64", role: "end", nullable: true },
        { name: TEMPORAL_COLUMNS.value, ...value, nullable: true },
        {
            name: TEMPORAL_COLUMNS.startText,
            dtype: "string",
            role: TIME_TEXT_ROLE,
            nullable: true,
            extra: { for: TEMPORAL_COLUMNS.start },
        },
        { name: TEMPORAL_COLUMNS.endText, dtype: "string", nullable: true, extra: { for: TEMPORAL_COLUMNS.end } },
        { name: TEMPORAL_COLUMNS.open, dtype: "u8", role: "open", nullable: true },
    ];
    if (temporalValue) {
        decls.push({
            name: TEMPORAL_COLUMNS.valueText,
            dtype: "string",
            nullable: true,
            extra: { for: TEMPORAL_COLUMNS.value },
        });
    }
    return decls;
}

/** The scalar type names GEXF 1.2 declares. */
const TYPES_1_2: ReadonlySet<string> = new Set(["integer", "long", "double", "float", "boolean", "string", "anyURI"]);

/** The scalar type names GEXF 1.3 declares (the list variants are derived). */
const TYPES_1_3: ReadonlySet<string> = new Set([
    ...TYPES_1_2,
    "bigdecimal",
    "biginteger",
    "byte",
    "char",
    "short",
    "date",
    "dateTime",
]);

/**
 * Whether a declared type text is a scalar type of a version (case-sensitive, as the schema spells
 * them).
 * @param type - the type text
 * @param version - the target version
 * @returns true when the version declares it
 */
export function isScalarType(type: string, version: GexfVersion): boolean {
    return (version === "1.2" ? TYPES_1_2 : TYPES_1_3).has(type);
}

/**
 * Whether a declared type text is a list type of a version: `liststring` in 1.2, `list<scalar>` in 1.3.
 * @param type - the type text
 * @param version - the target version
 * @returns true when the version declares it
 */
export function isListType(type: string, version: GexfVersion): boolean {
    if (version === "1.2") {
        return type === "liststring";
    }
    return type.startsWith("list") && type.length > 4 && isScalarType(type.slice(4), "1.3");
}

/**
 * The canonical GEXF scalar type of a column dtype (design section 5.1 reversed).
 * @param dtype - a scalar dtype
 * @returns the type text; u32 becomes `long` (its range exceeds `integer`), u8 `integer`, json `string`
 */
export function canonicalScalarType(dtype: Dtype): string {
    switch (dtype) {
        case "bool":
            return "boolean";
        case "i32":
        case "u8":
            return "integer";
        case "u32":
            return "long";
        case "f32":
            return "float";
        case "f64":
            return "double";
        case "dict":
        case "string":
        case "json":
        case "list":
            return "string";
        default: {
            const name: string = dtype;
            return name;
        }
    }
}
