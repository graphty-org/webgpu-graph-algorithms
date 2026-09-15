/**
 * The `@graphty/graph-io/gml` subpath (design section 8.2): the GML importer and exporter, their
 * option types, and their issue and loss-note codes grouped in two tables.
 */

import {
    COLUMN_RENAMED_CODE,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    ID_MERGED_CODE,
    INVALID_UTF8_CODE,
    MIXED_DIRECTION_CODE,
    OPTION_IGNORED_CODE,
    SINK_OPTION_CODE,
    SYNTAX_CODE,
} from "../../common/codes.js";
import {
    GRAPHICS_CONFLICT_CODE,
    GRAPHICS_OVERRIDDEN_CODE,
    INVALID_KEY_CODE,
    JSON_ARRAY_CODE,
    KEY_MANGLED_CODE,
    NESTED_ARRAY_CODE,
    POSITION_COMPONENTS_CODE,
    RECORD_BOOLEAN_CODE,
    RECORD_NULL_CODE,
    RECORD_NUMBER_TYPE_CODE,
    RESERVED_KEY_CODE,
} from "./exporter.js";
import {
    DUPLICATE_NODE_CODE,
    ELEMENT_TYPE_CODE,
    FLAG_TYPE_CODE,
    FLAG_VALUE_CODE,
    ID_DROPPED_CODE,
    ID_TYPE_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MISSING_LABEL_CODE,
    NO_GRAPH_CODE,
    PRECISION_CODE,
    REPEATED_KEY_CODE,
    ROLE_TAKEN_CODE,
    SECOND_GRAPH_CODE,
} from "./importer.js";

export { gmlExporter, type GmlExportOptions } from "./exporter.js";
export { gmlImporter, type GmlImportOptions } from "./importer.js";

/**
 * The issue codes the GML importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts) and the GML-specific ones. A key is the code without
 * its severity and format prefixes.
 */
export const GML_ISSUE = Object.freeze({
    /** A grammar violation: an untokenisable bare token, an unclosed string or `[`, a stray `]`, a key without a value (fatal). */
    SYNTAX: SYNTAX_CODE,
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** No `graph [` block (fatal). */
    NO_GRAPH: NO_GRAPH_CODE,
    /** More than one `graph` block (fatal). */
    SECOND_GRAPH: SECOND_GRAPH_CODE,
    /** A node without an `id`. */
    MISSING_ID: MISSING_ID_CODE,
    /** A node without a `label` under nodeIdFrom "label". */
    MISSING_LABEL: MISSING_LABEL_CODE,
    /** An edge without `source` or `target`. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** A node id, source or target that is not an integer. */
    ID_TYPE: ID_TYPE_CODE,
    /** A node id declared twice (later keys overwrite). */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** A structural key repeated in one element. */
    REPEATED_KEY: REPEATED_KEY_CODE,
    /** A `node` / `edge` key whose value is not a record. */
    ELEMENT_TYPE: ELEMENT_TYPE_CODE,
    /** A `directed` / `multigraph` flag that is not an integer. */
    FLAG_TYPE: FLAG_TYPE_CODE,
    /** A `directed` / `multigraph` flag outside 0 / 1, or repeated. */
    FLAG_VALUE: FLAG_VALUE_CODE,
    /** An integer beyond 2^53 rounded to f64. */
    PRECISION: PRECISION_CODE,
    /** A column renamed `<name>#<key>` because the sink held the name with another shape. */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
    /** A column declared without its role because the sink already holds it. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** Two id texts merged into one number under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
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
    /** Under nodeIdFrom "label" / "index" the integer ids are not kept (a loss note). */
    ID_DROPPED: ID_DROPPED_CODE,
});

/** The loss-note codes the GML exporter's check() reports beyond the shared LOSS table, by name. */
export const GML_LOSS = Object.freeze({
    /** A json column holds numbers; GML records cannot keep int versus real (design section 8.5). */
    RECORD_NUMBER_TYPE: RECORD_NUMBER_TYPE_CODE,
    /** A json column holds booleans, written 1 / 0. */
    RECORD_BOOLEAN: RECORD_BOOLEAN_CODE,
    /** A json column holds nulls, omitted. */
    RECORD_NULL: RECORD_NULL_CODE,
    /** An array inside an array; export() throws. */
    NESTED_ARRAY: NESTED_ARRAY_CODE,
    /** A json row that is an array, written as repeated keys. */
    JSON_ARRAY: JSON_ARRAY_CODE,
    /** A column name or record key outside the GML key grammar. */
    INVALID_KEY: INVALID_KEY_CODE,
    /** A column named like a structural key. */
    RESERVED_KEY: RESERVED_KEY_CODE,
    /** A key rewritten under sanitizeKeys "mangle". */
    KEY_MANGLED: KEY_MANGLED_CODE,
    /** A position column with more than three components. */
    POSITION_COMPONENTS: POSITION_COMPONENTS_CODE,
    /** A graphics record whose x / y / z the position column overrides. */
    GRAPHICS_OVERRIDDEN: GRAPHICS_OVERRIDDEN_CODE,
    /** A graphics record that cannot hold the position. */
    GRAPHICS_CONFLICT: GRAPHICS_CONFLICT_CODE,
});
