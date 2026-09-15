/**
 * The `@graphty/graph-io/neo4j` subpath: the Neo4j importer and exporter with their option types
 * (design section 8.2), the names of the reserved columns and the issue and loss-note codes
 * grouped in two tables.
 */

import {
    COLUMN_RENAMED_CODE,
    INVALID_UTF8_CODE,
    MUTUAL_EXPANDED_CODE,
    OPTION_IGNORED_CODE,
    PRECISION_CODE,
    ROLE_DROPPED_CODE,
    SINK_OPTION_CODE,
    UNKNOWN_ATTR_TYPE_CODE,
    WEIGHT_KEY_CLASH_CODE,
} from "../../common/codes.js";
import { BAD_QUOTE_CODE, UNCLOSED_QUOTE_CODE } from "../csv/records.js";
import {
    ARRAY_DELIMITER_LOSS,
    DECLARED_TYPE_CHANGED_LOSS,
    ID_COLUMN_TAKEN_LOSS,
    ID_TEXT_COLLISION_LOSS,
    ID_TEXT_TYPE_LOSS,
    MULTIPLE_ID_PROPERTIES_LOSS,
    UNDIRECTED_LOSS,
    WEIGHT_COLUMN_TAKEN_LOSS,
} from "./exporter.js";
import {
    COLUMN_COUNT_CODE,
    DUPLICATE_NODE_CODE,
    HEADER_CODE,
    HEADER_OPTION_CODE,
    ID_MERGED_CODE,
    ID_SPACE_COLLISION_CODE,
    IGNORED_COLUMNS_LOSS,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    ROLE_TAKEN_CODE,
} from "./importer.js";

export { NEO4J_CAPABILITIES, neo4jExporter, type Neo4jExportOptions } from "./exporter.js";
export { ID_SPACE_COLUMN, LABELS_COLUMN, neo4jImporter, type Neo4jImportOptions, TYPE_COLUMN } from "./importer.js";

/**
 * The issue codes the Neo4j importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts, the CSV record reader) and the Neo4j-specific ones.
 * A key is the code without its severity and format prefixes.
 */
export const NEO4J_ISSUE = Object.freeze({
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** An unterminated quoted field (fatal). */
    CSV_UNCLOSED_QUOTE: UNCLOSED_QUOTE_CODE,
    /** Text after a closing quote (fatal). */
    CSV_QUOTE: BAD_QUOTE_CODE,
    /** No header or a malformed header (fatal). */
    HEADER: HEADER_CODE,
    /** A row with a different field count than the header. */
    COLUMN_COUNT: COLUMN_COUNT_CODE,
    /** A node row without an id. */
    MISSING_ID: MISSING_ID_CODE,
    /** A relationship row without a start or end id. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** A node id repeated in one id space (last write wins). */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** A node id declared in two id spaces. */
    ID_SPACE_COLLISION: ID_SPACE_COLLISION_CODE,
    /** Two id texts merged into one number under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** A header brace option the importer does not apply. */
    HEADER_OPTION_IGNORED: HEADER_OPTION_CODE,
    /** A declared type the format does not define (kept as string). */
    UNKNOWN_ATTR_TYPE: UNKNOWN_ATTR_TYPE_CODE,
    /** A long value beyond 2^53 rounded. */
    PRECISION: PRECISION_CODE,
    /** A column renamed `<name>#<id>` because the name was taken (design section 5.6). */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
    /** A role the caller's sink already holds. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** A common option the importer has no use for (nodeIdFrom, defaultDirected, ...). */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
    /** A builder-policy option the sink does not honour. */
    SINK_OPTION: SINK_OPTION_CODE,
});

/**
 * The loss-note codes of the Neo4j importer (report.lossy) and exporter (check()), by name: the
 * Neo4j ones and, aliased, the shared ones it records (`LOSS` holds the generic pre-flight's).
 */
export const NEO4J_LOSS = Object.freeze({
    /** `:IGNORE` columns skipped on import. */
    IGNORED_COLUMNS: IGNORED_COLUMNS_LOSS,
    /** Undirected edges (an undirected snapshot, the folded pairs of a mixed one) written as directed relationships. */
    UNDIRECTED_AS_DIRECTED: UNDIRECTED_LOSS,
    /** A mutual pair written as two directed relationships without its mark. */
    MUTUAL_EXPANDED: MUTUAL_EXPANDED_CODE,
    /** A role column Neo4j has no slot for (label, ...) written as a plain property. */
    ROLE_DROPPED: ROLE_DROPPED_CODE,
    /** A plain `weight` edge column reads back as THE weight (the importer's weightFrom default). */
    WEIGHT_KEY_CLASH: WEIGHT_KEY_CLASH_CODE,
    /** A node id whose text re-imports as another type under the canonical rule. */
    ID_TEXT_TYPE: ID_TEXT_TYPE_LOSS,
    /** A number and a string id with the same text; export() throws. */
    ID_TEXT_COLLISION: ID_TEXT_COLLISION_LOSS,
    /** The weight column name is taken by an edge column; export() throws. */
    WEIGHT_COLUMN_TAKEN: WEIGHT_COLUMN_TAKEN_LOSS,
    /** The id column name is taken by a node column; export() throws. */
    ID_COLUMN_TAKEN: ID_COLUMN_TAKEN_LOSS,
    /** Several stored-id properties; one becomes the `:ID` column. */
    MULTIPLE_ID_PROPERTIES: MULTIPLE_ID_PROPERTIES_LOSS,
    /** An integer-typed column with non-integral values written as double. */
    DECLARED_TYPE_CHANGED: DECLARED_TYPE_CHANGED_LOSS,
    /** A list item containing the array delimiter, which Neo4j cannot escape. */
    ARRAY_DELIMITER: ARRAY_DELIMITER_LOSS,
});
