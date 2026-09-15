/**
 * What the GraphML importer and exporter share: the namespace, the format facts, the reserved
 * column names of the XML-attribute-derived columns (design section 5.6), the `meta.extra`
 * keys the importer records for the exporter, and the issue and loss codes.
 */

import {
    BAD_DEFAULT_CODE,
    COLUMN_RENAMED_CODE,
    COUNT_HINT_CODE,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_KEY_CODE,
    DUPLICATE_NODE_CODE,
    HYPEREDGE_CODE,
    ID_MERGED_CODE,
    ID_TEXT_TYPE_CODE,
    INVALID_UTF8_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MIXED_DIRECTION_CODE,
    MULTIPLE_GRAPHS_CODE,
    MUTUAL_AS_UNDIRECTED_CODE,
    NO_GRAPH_CODE,
    OPTION_IGNORED_CODE,
    PARENTS_DROPPED_CODE,
    PRECISION_CODE,
    ROLE_DROPPED_CODE,
    ROLE_TAKEN_CODE,
    SINK_OPTION_CODE,
    STRAY_TEXT_CODE,
    UNKNOWN_ATTR_TYPE_CODE,
    UNKNOWN_ELEMENT_CODE,
    XML_SYNTAX_CODE,
} from "../../common/codes.js";

/** The GraphML namespace. */
export const GRAPHML_NAMESPACE = "http://graphml.graphdrawing.org/xmlns";

/** The yFiles extension namespace, declared as `xmlns:y` when a yfiles column is written. */
export const YFILES_NAMESPACE = "http://www.yworks.com/xml/graphml";

/** The XML Schema instance namespace and the schema location the exporter writes. */
export const XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance";

/** The GraphML schema location written by the exporter. */
export const SCHEMA_LOCATION =
    "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd";

/** The format name. */
export const FORMAT = "graphml";

/** File extensions. */
export const EXTENSIONS: readonly string[] = Object.freeze([".graphml", ".xml"]);

/** MIME types. */
export const MIME_TYPES: readonly string[] = Object.freeze(["application/graphml+xml", "application/xml", "text/xml"]);

/** The `attr.name` of the node key that carries ids rewritten by `sanitizeIds: "mangle"` (design section 8.5). */
export const ORIGINAL_ID_ATTRIBUTE = "graphty:originalId";

/** The node column that keeps original ids when `restoreMangledIds` is off (design section 5.6). */
export const ORIGINAL_ID_COLUMN = "graphty.originalId";

/** The edge column of the `id` XML attribute (role id, unique). */
export const EDGE_ID_COLUMN = "id";

/** The edge columns of the `sourceport` / `targetport` XML attributes (roles sourcePort / targetPort). */
export const SOURCE_PORT_COLUMN = "sourceport";

/** The edge column of the `targetport` XML attribute. */
export const TARGET_PORT_COLUMN = "targetport";

/** The node column of the containing node of a nested graph (u32, role parent, refersTo node). */
export const PARENT_COLUMN = "parent";

/** The bool node column marking the hub nodes synthesised for hyperedges under `hyperedges: "star"`. */
export const HYPEREDGE_HUB_COLUMN = "graphty.hyperedge";

/** The column name a node key titled `label` receives the `label` role under. */
export const LABEL_COLUMN = "label";

/** Edge column names reserved for the XML-attribute-derived columns; a key titled like one is renamed `<name>#<id>`. */
export const RESERVED_EDGE_NAMES: ReadonlySet<string> = new Set([
    EDGE_ID_COLUMN,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
]);

/** Node column names reserved for the XML-derived columns. */
export const RESERVED_NODE_NAMES: ReadonlySet<string> = new Set([PARENT_COLUMN]);

/** The `meta.extra` key under which the importer records what the exporter needs. */
export const META_KEY = "graphml";

/** What the importer stores under `meta.extra.graphml`. */
export interface GraphmlMeta {
    /** The top-level `<graph id>`, or null. */
    readonly graphId: string | null;
    /** The top-level `edgedefault` as written, so a mixed file re-exports with the same layout. */
    readonly edgedefault: "directed" | "undirected" | null;
    /** The `xmlns:<prefix>` declarations of the root element (yFiles and the like). */
    readonly namespaces: Readonly<Record<string, string>>;
}

/**
 * The issue codes the GraphML importer records (design section 8.6), by name: the codes shared
 * with the other importers (src/common/codes.ts) and the GraphML-specific ones. A key is the code
 * without its severity and format prefixes.
 */
export const GRAPHML_ISSUE = Object.freeze({
    /** Fatal: the input is not well-formed XML. */
    XML_SYNTAX: XML_SYNTAX_CODE,
    /** Fatal: the input holds invalid UTF-8. */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** Fatal: the root element is not `<graphml>`. */
    NOT_GRAPHML: "E_NOT_GRAPHML",
    /** Fatal: the document has no `<graph>`. */
    NO_GRAPH: NO_GRAPH_CODE,
    /** A `<key>` without an id. */
    KEY_MISSING_ID: "E_GRAPHML_KEY_MISSING_ID",
    /** Two `<key>` elements with the same id. */
    DUPLICATE_KEY: DUPLICATE_KEY_CODE,
    /** A key declared for hyperedges, ports or endpoints (never used). */
    KEY_DOMAIN_UNSUPPORTED: "W_GRAPHML_KEY_DOMAIN_UNSUPPORTED",
    /** A `<node>` without an id. */
    MISSING_ID: MISSING_ID_CODE,
    /** An `<edge>` without a source or a target. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** A `<data>` whose key was never declared. */
    UNKNOWN_KEY: "E_GRAPHML_UNKNOWN_KEY",
    /** A `<data>` whose key is declared for another domain. */
    KEY_DOMAIN: "W_GRAPHML_KEY_DOMAIN",
    /** A `<data>` without a key attribute. */
    DATA_MISSING_KEY: "E_GRAPHML_DATA_MISSING_KEY",
    /** A `<key>` whose `for` is not a GraphML domain. */
    KEY_FOR_INVALID: "E_GRAPHML_KEY_FOR_INVALID",
    /** Data of a nested `<graph>`; only the top-level graph has attributes. */
    NESTED_GRAPH_DATA: "W_GRAPHML_NESTED_GRAPH_DATA",
    /** Data of a hyperedge expanded to a star or clique. */
    HYPEREDGE_DATA_DROPPED: "W_GRAPHML_HYPEREDGE_DATA_DROPPED",
    /** Nested elements in the `<data>` of a typed (non-yfiles) key. */
    DATA_NESTED: "E_GRAPHML_DATA_NESTED",
    /** A `<node>` declared twice. */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** An edge id already used by another edge. */
    DUPLICATE_EDGE_ID: DUPLICATE_EDGE_ID_CODE,
    /** A `directed` attribute that is neither true nor false. */
    INVALID_DIRECTED: "E_GRAPHML_INVALID_DIRECTED",
    /** An `edgedefault` that is neither directed nor undirected. */
    INVALID_EDGEDEFAULT: "E_GRAPHML_INVALID_EDGEDEFAULT",
    /** A `<graph>` without edgedefault; the `defaultDirected` option applies. */
    EDGEDEFAULT_MISSING: "W_GRAPHML_EDGEDEFAULT_MISSING",
    /** A second top-level `<graph>`; its nodes and edges are merged into the first. */
    MULTIPLE_GRAPHS: MULTIPLE_GRAPHS_CODE,
    /** A `parse.nodes` / `parse.edges` hint the sink cannot reserve (ignored). */
    COUNT_HINT: COUNT_HINT_CODE,
    /** A hyperedge under `hyperedges: "error"`. */
    HYPEREDGE: HYPEREDGE_CODE,
    /** Hyperedges skipped under `hyperedges: "skip"`. */
    HYPEREDGE_SKIPPED: "W_GRAPHML_HYPEREDGE_SKIPPED",
    /** An endpoint of a hyperedge without a node, or with an unknown type. */
    HYPEREDGE_ENDPOINT: "E_GRAPHML_HYPEREDGE_ENDPOINT",
    /** `<port>` declarations (and their data) are not kept; sourceport / targetport edge attributes are. */
    PORT_DECLARATION: "W_GRAPHML_PORT_DECLARATION",
    /** A `<locator>` element. */
    LOCATOR_DROPPED: "W_GRAPHML_LOCATOR_DROPPED",
    /** A `<desc>` of a node, an edge or a hyperedge. */
    DESC_DROPPED: "W_GRAPHML_DESC_DROPPED",
    /** An element the GraphML schema does not define at that place. */
    UNKNOWN_ELEMENT: UNKNOWN_ELEMENT_CODE,
    /** Non-whitespace text where the schema allows only elements. */
    STRAY_TEXT: STRAY_TEXT_CODE,
    /** Two distinct id texts merged into one number under `ids: "number"`. */
    ID_MERGED: ID_MERGED_CODE,
    /** An option GraphML has no use for (`nodeIdFrom`: nodes are identified by their id attribute). */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
    /** A yFiles key under `yfiles: "skip"`. */
    YFILES_SKIPPED: "W_GRAPHML_YFILES_SKIPPED",
    /** A key renamed `<name>#<id>` because the name was taken (design section 5.6). */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
    /** A key declared without its role because the table already holds it. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** A declared type the format does not define (kept as string). */
    UNKNOWN_ATTR_TYPE: UNKNOWN_ATTR_TYPE_CODE,
    /** A default that does not parse as the declared type. */
    BAD_DEFAULT: BAD_DEFAULT_CODE,
    /** A long value beyond 2^53 rounded. */
    PRECISION: PRECISION_CODE,
    /** A builder-policy option the sink does not honour. */
    SINK_OPTION: SINK_OPTION_CODE,
    /** The sink refused the file's direction. */
    DIRECTION_REFUSED: DIRECTION_REFUSED_CODE,
    /** Edges forced to the policy's direction. */
    DIRECTION_FORCED: DIRECTION_FORCED_CODE,
    /** A mixed file under onMixedDirection "error" (fatal). */
    MIXED_DIRECTION: MIXED_DIRECTION_CODE,
});

/**
 * Loss note codes of the GraphML importer (report.lossy) and exporter (check()): the GraphML
 * ones; the generic ones (dtypes, lists, json text, positions, visual and temporal roles,
 * extension tables, id charsets, the weight key clash, name changes) are those of `LOSS`.
 */
export const GRAPHML_LOSS = Object.freeze({
    /** yFiles nested XML kept as a JSON tree: structure preserved, not byte-exact. */
    YFILES_JSON: "W_GRAPHML_YFILES_JSON",
    /** A mutual pair written as one undirected edge; the mark is lost. */
    MUTUAL_AS_UNDIRECTED: MUTUAL_AS_UNDIRECTED_CODE,
    /** A `parents` (multi-parent) column cannot be written as nested graphs. */
    PARENTS_DROPPED: PARENTS_DROPPED_CODE,
    /** A role column GraphML has no slot for (kind, node ids, ...) written as a plain attribute. */
    ROLE_DROPPED: ROLE_DROPPED_CODE,
    /** Containment order differs from index order; node indices change after a round trip. */
    HIERARCHY_REORDERED: "W_GRAPHML_HIERARCHY_REORDERED",
    /** Nodes whose parent chain never reaches a root are written at the top level. */
    PARENT_CYCLE: "W_GRAPHML_PARENT_CYCLE",
    /** Node ids that change type after a round trip under the canonical rule (design section 4.1). */
    ID_TEXT_TYPE: ID_TEXT_TYPE_CODE,
    /** A numeric edge id column reads back as string. */
    EDGE_ID_TEXT: "W_GRAPHML_EDGE_ID_TEXT",
    /** A yfiles json value that is not a serialisable tree: export() will throw E_COLUMN_TYPE. */
    YFILES_TREE: "E_GRAPHML_YFILES_TREE",
});
