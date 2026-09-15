/**
 * The CSV / TSV importer and exporter of @graphty/graph-io: the subpath entry `@graphty/graph-io/csv`.
 */

import {
    COLUMN_RENAMED_CODE,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    INVALID_UTF8_CODE,
    MIXED_DIRECTION_CODE,
    OPTION_IGNORED_CODE,
    SINK_OPTION_CODE,
} from "../../common/codes.js";
import { WIDENING_UNSUPPORTED_CODE } from "../../common/text.js";
import {
    BAD_TYPE_CODE,
    COLUMN_MISSING_CODE,
    DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_NODE_CODE,
    EMPTY_INPUT_CODE,
    FIELD_COUNT_CODE,
    ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    NO_DATA_ROWS_CODE,
    NO_ENDPOINT_COLUMNS_CODE,
    NO_ID_COLUMN_CODE,
    ROLE_TAKEN_CODE,
} from "./importer.js";
import { BAD_QUOTE_CODE, UNCLOSED_QUOTE_CODE } from "./records.js";

export { CSV_CAPABILITIES, CSV_LOSS, csvExporter, type CsvExportOptions } from "./exporter.js";
export { type CsvColumnRef } from "./header.js";
export { csvImporter, type CsvImportOptions } from "./importer.js";

/**
 * The issue codes the CSV importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts) and the CSV-specific ones. A key is the code without
 * its severity and format prefixes.
 */
export const CSV_ISSUE = Object.freeze({
    /** The input is empty (fatal). */
    EMPTY_INPUT: EMPTY_INPUT_CODE,
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** The header names neither endpoint columns nor an id column (fatal). */
    NO_ENDPOINT_COLUMNS: NO_ENDPOINT_COLUMNS_CODE,
    /** A node table without an id column (fatal). */
    NO_ID_COLUMN: NO_ID_COLUMN_CODE,
    /** A row with a different field count than the header. */
    FIELD_COUNT: FIELD_COUNT_CODE,
    /** An edge row with a blank source or target. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** A node row with a blank id. */
    MISSING_ID: MISSING_ID_CODE,
    /** A Type cell outside Directed / Undirected / Mutual. */
    BAD_TYPE: BAD_TYPE_CODE,
    /** An unterminated quoted field (fatal). */
    UNCLOSED_QUOTE: UNCLOSED_QUOTE_CODE,
    /** Text after a closing quote (fatal). */
    QUOTE: BAD_QUOTE_CODE,
    /** A header and no data rows. */
    NO_DATA_ROWS: NO_DATA_ROWS_CODE,
    /** A node table row repeating an id. */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** An edge row repeating an edge id (skipped). */
    DUPLICATE_EDGE_ID: DUPLICATE_EDGE_ID_CODE,
    /** Two id cells merged into one number under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** An explicitly named weight column the file does not have. */
    COLUMN_MISSING: COLUMN_MISSING_CODE,
    /** A column whose role another column of the sink already holds. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** A column renamed `<name>#<position>` (a repeated header, or a name the sink holds with another shape). */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
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
    /** A text column the sink could not widen to the dtype its cells imply. */
    WIDENING_UNSUPPORTED: WIDENING_UNSUPPORTED_CODE,
});
