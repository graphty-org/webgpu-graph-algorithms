/**
 * The public barrel of @graphty/graph-io (design sections 8.2, 12.4 and 13.1): the io contract
 * types and ImportError, the registry with `importGraph()` / `exportGraph()` / `sniff()`, the
 * `children` CSR helper, every built-in importer and exporter (also reachable through the per-format
 * subpath exports `@graphty/graph-io/<format>`), and the shared helpers a third-party importer or
 * exporter builds on (report builder, input reader, option resolution, direction resolver, id
 * coercion, the capability check and the loss / issue codes callers branch on). Named exports
 * only; no default export.
 */

// ============================================================ io contract types (12.4)
export {
    type CommonExportOptions,
    type CommonImportOptions,
    type ExportCapabilities,
    type GraphExporter,
    type GraphImporter,
    ImportError,
    type ImportInput,
    type ImportIssue,
    type ImportReport,
    type IssueCategory,
    type LossNote,
} from "./types.js";

// ============================================================ registry, sniffing, children (8.2)
export {
    ChildrenCsr,
    childrenCsr,
    childrenFromColumn,
    type ChildrenOptions,
    type ContainmentRole,
    type DepthFirstOrder,
} from "./children.js";
export {
    type BuilderSeed,
    checkExport,
    createRegistry,
    exportGraph,
    type ExportGraphOptions,
    exportGraphToString,
    FormatRegistry,
    importGraph,
    type ImportGraphOptions,
    type ImportGraphResult,
    registry,
    sniff,
    UNKNOWN_FORMAT_CODE,
} from "./registry.js";
export {
    extensionOf,
    GRAPH_FORMATS,
    type GraphFormatName,
    headBytes,
    normalizeMimeType,
    rankFormats,
    SNIFF_HEAD_BYTES,
    sniffFormat,
    type SniffHints,
    sniffJsonDialectHead,
    type SniffResult,
} from "./sniff.js";

// ============================================================ formats (8.4, 8.5)
export {
    CSV_CAPABILITIES,
    CSV_ISSUE,
    CSV_LOSS,
    type CsvColumnRef,
    csvExporter,
    type CsvExportOptions,
    csvImporter,
    type CsvImportOptions,
} from "./formats/csv/index.js";
export {
    DOT_ISSUE,
    DOT_LOSS,
    dotExporter,
    type DotExportOptions,
    dotImporter,
    type DotImportOptions,
} from "./formats/dot/index.js";
export {
    GEXF_1_2_CAPABILITIES,
    GEXF_ISSUE,
    GEXF_LOSS,
    gexfExporter,
    type GexfExportOptions,
    gexfImporter,
    type GexfImportOptions,
    type GexfVersion,
} from "./formats/gexf/index.js";
export {
    GML_ISSUE,
    GML_LOSS,
    gmlExporter,
    type GmlExportOptions,
    gmlImporter,
    type GmlImportOptions,
} from "./formats/gml/index.js";
export {
    GRAPHML_ISSUE,
    GRAPHML_LOSS,
    graphmlExporter,
    type GraphmlExportOptions,
    graphmlImporter,
    type GraphmlImportOptions,
} from "./formats/graphml/index.js";
export {
    dialectCapabilities,
    JSON_DIALECTS,
    JSON_ISSUE,
    JSON_LOSS,
    jsonCapabilities,
    type JsonDialect,
    jsonExporter,
    type JsonExportOptions,
    jsonImporter,
    type JsonImportOptions,
    type JsonShapeMeta,
} from "./formats/json/index.js";
export {
    ID_SPACE_COLUMN,
    LABELS_COLUMN,
    NEO4J_CAPABILITIES,
    NEO4J_ISSUE,
    NEO4J_LOSS,
    neo4jExporter,
    type Neo4jExportOptions,
    neo4jImporter,
    type Neo4jImportOptions,
    TYPE_COLUMN,
} from "./formats/neo4j/index.js";
export {
    PAJEK_ISSUE,
    PAJEK_LOSS,
    pajekExporter,
    type PajekExportOptions,
    pajekImporter,
    type PajekImportOptions,
} from "./formats/pajek/index.js";

// ============================================================ shared helpers for plugin authors (8.4, 8.5, 8.6)
export {
    BAD_DEFAULT_CODE,
    BAD_OPTIONS_CODE,
    declareCompanion,
    declareResolved,
    PRECISION_CODE,
    RENAMED_CODE,
    type ResolvedDeclaration,
    ROLE_TAKEN_CODE,
    UNKNOWN_TYPE_CODE,
} from "./common/attributes.js";
export * from "./common/codes.js";
export {
    DIRECTED_COLUMN,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    DirectionResolver,
    type EdgeKind,
    type EdgeLocation,
    MIXED_DIRECTION_CODE,
    MUTUAL_COLUMN,
    PAIR_COLUMN,
    type PairFolding,
    pairFolding,
    type PairFoldingOptions,
} from "./common/direction.js";
export {
    capabilities,
    checkCapabilities,
    type CheckExtras,
    type IdCharset,
    isNmtoken,
    LOSS,
    mangleNmtoken,
    NO_CAPABILITIES,
    type SanitizedIds,
    sanitizeIds,
} from "./common/export.js";
export { formatDecimal, formatF32, formatF64, formatGmlReal, formatInteger } from "./common/format.js";
export {
    canonicalId,
    coerceId,
    coerceIdText,
    ID_MERGED_CODE,
    IdCoercer,
    isCanonicalIntegerText,
} from "./common/ids.js";
export {
    inputLength,
    INVALID_UTF8_CODE,
    isImportInput,
    LineReader,
    type ReadOptions,
    readText,
    textChunks,
    throwIfAborted,
} from "./common/input.js";
export {
    DEFAULT_ERROR_LIMIT,
    type ImportFormatDefaults,
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedExportOptions,
    type ResolvedImportOptions,
    resolveExportOptions,
    resolveImportOptions,
    SINK_OPTION_CODE,
} from "./common/options.js";
export {
    ImportReportBuilder,
    isAbortError,
    type IssueLocation,
    type MutableCounts,
    PARSE_ERROR_CODE,
} from "./common/report.js";
export {
    inferTextDtype,
    isNumericText,
    parseTextCell,
    TextCellWriter,
    type TextDtype,
    WIDENING_UNSUPPORTED_CODE,
} from "./common/text.js";
export { type ExplicitWeights, explicitWeights, isWeightField, parseWeightText } from "./common/weights.js";
export {
    collectBytes,
    decodeChunks,
    DEFAULT_CHUNK_BYTES,
    encodeChunks,
    joinText,
    type TextParts,
    toReadableStream,
} from "./common/writer.js";
export {
    hasIllegalXmlChar,
    tokenizeXml,
    type XmlHandler,
    xmlIllegalTextNotes,
    XmlSyntaxError,
    XmlTokenizer,
} from "./common/xml.js";
