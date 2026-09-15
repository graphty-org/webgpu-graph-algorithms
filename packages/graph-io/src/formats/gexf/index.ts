/**
 * The GEXF subpath entry (`@graphty/graph-io/gexf`, design section 8.2): the importer and exporter
 * objects with their format-specific option types, the exporter's loss-note codes and the
 * importer's issue codes grouped in one table.
 */

import {
    BAD_DEFAULT_CODE,
    BAD_OPTIONS_CODE,
    COLUMN_RENAMED_CODE,
    COUNT_HINT_CODE,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_NODE_CODE,
    ID_MERGED_CODE,
    INVALID_UTF8_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MIXED_DIRECTION_CODE,
    NO_GRAPH_CODE,
    OPTION_IGNORED_CODE,
    PRECISION_CODE,
    ROLE_TAKEN_CODE,
    SINK_OPTION_CODE,
    STRAY_TEXT_CODE,
    UNKNOWN_ATTR_TYPE_CODE,
    UNKNOWN_ELEMENT_CODE,
    UNKNOWN_PARENT_CODE,
    XML_SYNTAX_CODE,
} from "../../common/codes.js";
import {
    ATTRIBUTE_ID_CODE,
    ATTRIBUTE_TYPE_CODE,
    ATTRIBUTES_CLASS_CODE,
    ATTVALUE_SHAPE_CODE,
    DUPLICATE_ATTRIBUTE_CODE,
    EDGE_TYPE_CODE,
    HEADER_VALUE_CODE,
    MISSING_NODES_CODE,
    NOT_GEXF_CODE,
    OPEN_BOUND_CONFLICT_CODE,
    SPELL_OPEN_CODE,
    TIMED_STATIC_CODE,
    UNKNOWN_ATTRIBUTE_CODE,
    VIZ_DYNAMIC_CODE,
    VIZ_SKIPPED_CODE,
    VIZ_VALUE_CODE,
    WEIGHT_IGNORED_CODE,
} from "./importer.js";

export { GEXF_1_2_CAPABILITIES, GEXF_LOSS, gexfExporter, type GexfExportOptions } from "./exporter.js";
export { gexfImporter, type GexfImportOptions } from "./importer.js";
export { type GexfVersion } from "./schema.js";

/**
 * The issue codes the GEXF importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts) and the GEXF-specific ones. A key is the code without
 * its severity and format prefixes.
 */
export const GEXF_ISSUE = Object.freeze({
    /** The XML is not well-formed (fatal). */
    XML_SYNTAX: XML_SYNTAX_CODE,
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** The root element is not `<gexf>` (fatal). */
    NOT_GEXF: NOT_GEXF_CODE,
    /** The document has no `<graph>` (fatal). */
    NO_GRAPH: NO_GRAPH_CODE,
    /** `<edges>` without `<nodes>`. */
    MISSING_NODES: MISSING_NODES_CODE,
    /** A node without an id. */
    MISSING_ID: MISSING_ID_CODE,
    /** An edge without a source or target. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** An edge `type` outside directed / undirected / mutual. */
    EDGE_TYPE: EDGE_TYPE_CODE,
    /** A `pid` / `<parent for>` naming an unknown node. */
    UNKNOWN_PARENT: UNKNOWN_PARENT_CODE,
    /** An `<attributes class>` outside node / edge. */
    ATTRIBUTES_CLASS: ATTRIBUTES_CLASS_CODE,
    /** An `<attribute>` without an id. */
    ATTRIBUTE_ID: ATTRIBUTE_ID_CODE,
    /** A `<graph>` header attribute with an unknown value. */
    HEADER_VALUE: HEADER_VALUE_CODE,
    /** A `count` hint the sink cannot reserve (ignored). */
    COUNT_HINT: COUNT_HINT_CODE,
    /** A node id declared twice. */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** An edge id declared twice (the second edge is skipped). */
    DUPLICATE_EDGE_ID: DUPLICATE_EDGE_ID_CODE,
    /** An attribute id declared twice in one class. */
    DUPLICATE_ATTRIBUTE: DUPLICATE_ATTRIBUTE_CODE,
    /** An attribute type the importer maps to string. */
    ATTRIBUTE_TYPE: ATTRIBUTE_TYPE_CODE,
    /** A declared type the format does not define (kept as string). */
    UNKNOWN_ATTR_TYPE: UNKNOWN_ATTR_TYPE_CODE,
    /** A default that does not parse as the declared type. */
    BAD_DEFAULT: BAD_DEFAULT_CODE,
    /** Options that do not parse as the declared type. */
    BAD_OPTIONS: BAD_OPTIONS_CODE,
    /** A column renamed `<name>#<id>` because the name was taken (design section 5.6). */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
    /** A column declared without its role because the table already holds it. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** An `<attvalue for>` naming an undeclared attribute. */
    UNKNOWN_ATTRIBUTE: UNKNOWN_ATTRIBUTE_CODE,
    /** An `<attvalue>` without a value. */
    ATTVALUE_SHAPE: ATTVALUE_SHAPE_CODE,
    /** A timed value on an attribute of a static group. */
    TIMED_VALUE_ON_STATIC: TIMED_STATIC_CODE,
    /** A long value beyond 2^53 rounded. */
    PRECISION: PRECISION_CODE,
    /** Two id texts merged into one number under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** The XML weight attribute ignored under weightFrom null. */
    WEIGHT_IGNORED: WEIGHT_IGNORED_CODE,
    /** viz elements skipped under viz: false. */
    VIZ_SKIPPED: VIZ_SKIPPED_CODE,
    /** A 1.2 dynamic viz element whose bounds were dropped. */
    VIZ_DYNAMIC_DROPPED: VIZ_DYNAMIC_CODE,
    /** A 1.2 startopen / endopen spell stored closed. */
    SPELL_OPEN_DROPPED: SPELL_OPEN_CODE,
    /** A viz value that could not be read. */
    VIZ_VALUE: VIZ_VALUE_CODE,
    /** Both `start` and `startopen` (or `end` and `endopen`) on one element. */
    OPEN_BOUND_CONFLICT: OPEN_BOUND_CONFLICT_CODE,
    /** An element or attribute GEXF does not define was skipped. */
    UNKNOWN_ELEMENT: UNKNOWN_ELEMENT_CODE,
    /** Text where GEXF allows only elements was ignored. */
    STRAY_TEXT: STRAY_TEXT_CODE,
    /** A common option the importer has no use for was given. */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
    /** A builder-policy option the sink does not honour. */
    SINK_OPTION: SINK_OPTION_CODE,
    /** The sink refused the file's direction. */
    DIRECTION_REFUSED: DIRECTION_REFUSED_CODE,
    /** Edges forced to the policy's direction. */
    DIRECTION_FORCED: DIRECTION_FORCED_CODE,
    /** A mixed file under onMixedDirection "error" (fatal). */
    MIXED_DIRECTION: MIXED_DIRECTION_CODE,
});
