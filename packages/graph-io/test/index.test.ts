import { describe, expect, it } from "vitest";

import * as graphIo from "../src/index.js";

/** The value exports of the barrel (classes, functions, constants); the types are checked by test/types. */
const VALUE_EXPORTS = [
    // io contract (12.4)
    "ImportError",
    // registry, sniffing, children (8.2)
    "FormatRegistry",
    "GRAPH_FORMATS",
    "SNIFF_HEAD_BYTES",
    "UNKNOWN_FORMAT_CODE",
    "ChildrenCsr",
    "checkExport",
    "childrenCsr",
    "childrenFromColumn",
    "createRegistry",
    "exportGraph",
    "exportGraphToString",
    "extensionOf",
    "headBytes",
    "importGraph",
    "normalizeMimeType",
    "rankFormats",
    "registry",
    "sniff",
    "sniffFormat",
    "sniffJsonDialectHead",
    // formats (8.4, 8.5)
    "CSV_CAPABILITIES",
    "CSV_ISSUE",
    "CSV_LOSS",
    "csvExporter",
    "csvImporter",
    "DOT_LOSS",
    "DOT_ISSUE",
    "dotExporter",
    "dotImporter",
    "GEXF_1_2_CAPABILITIES",
    "GEXF_ISSUE",
    "GEXF_LOSS",
    "gexfExporter",
    "gexfImporter",
    "GML_ISSUE",
    "GML_LOSS",
    "gmlExporter",
    "gmlImporter",
    "GRAPHML_ISSUE",
    "GRAPHML_LOSS",
    "graphmlExporter",
    "graphmlImporter",
    "JSON_DIALECTS",
    "JSON_ISSUE",
    "JSON_LOSS",
    "dialectCapabilities",
    "jsonCapabilities",
    "jsonExporter",
    "jsonImporter",
    "ID_SPACE_COLUMN",
    "LABELS_COLUMN",
    "NEO4J_CAPABILITIES",
    "NEO4J_ISSUE",
    "NEO4J_LOSS",
    "TYPE_COLUMN",
    "neo4jExporter",
    "neo4jImporter",
    "PAJEK_ISSUE",
    "PAJEK_LOSS",
    "pajekExporter",
    "pajekImporter",
    // shared helpers for plugin authors
    "BAD_DEFAULT_CODE",
    "BAD_OPTIONS_CODE",
    "PRECISION_CODE",
    "RENAMED_CODE",
    "UNKNOWN_TYPE_CODE",
    "DIRECTED_COLUMN",
    "DIRECTION_FORCED_CODE",
    "DIRECTION_REFUSED_CODE",
    "DirectionResolver",
    "MIXED_DIRECTION_CODE",
    "MUTUAL_COLUMN",
    "PAIR_COLUMN",
    "LOSS",
    "NO_CAPABILITIES",
    "capabilities",
    "checkCapabilities",
    "isNmtoken",
    "mangleNmtoken",
    "sanitizeIds",
    "ID_MERGED_CODE",
    "IdCoercer",
    "canonicalId",
    "coerceId",
    "coerceIdText",
    "isCanonicalIntegerText",
    "INVALID_UTF8_CODE",
    "LineReader",
    "inputLength",
    "isImportInput",
    "readText",
    "textChunks",
    "throwIfAborted",
    "DEFAULT_ERROR_LIMIT",
    "SINK_OPTION_CODE",
    "reportSinkOptions",
    "resolveExportOptions",
    "resolveImportOptions",
    "ImportReportBuilder",
    "PARSE_ERROR_CODE",
    "isAbortError",
    "inferTextDtype",
    "isNumericText",
    "parseTextCell",
    "DEFAULT_CHUNK_BYTES",
    "collectBytes",
    "decodeChunks",
    "encodeChunks",
    "joinText",
    "toReadableStream",
    // the shared issue and loss codes (src/common/codes.ts)
    "COLUMN_RENAMED_CODE",
    "COLUMN_RENAMED_LOSS_CODE",
    "COUNT_HINT_CODE",
    "DUPLICATE_EDGE_ID_CODE",
    "DUPLICATE_KEY_CODE",
    "DUPLICATE_NODE_CODE",
    "EMPTY_COLUMN_DROPPED_CODE",
    "EMPTY_INPUT_CODE",
    "HYPEREDGE_CODE",
    "ID_TEXT_COLLISION_CODE",
    "ID_TEXT_TYPE_CODE",
    "INTEGRAL_F64_CODE",
    "MISSING_ENDPOINT_CODE",
    "MISSING_ID_CODE",
    "MULTIPLE_GRAPHS_CODE",
    "MUTUAL_AS_UNDIRECTED_CODE",
    "MUTUAL_EXPANDED_CODE",
    "NO_GRAPH_CODE",
    "OPTIONS_GAINED_CODE",
    "OPTION_IGNORED_CODE",
    "PARENTS_DROPPED_CODE",
    "ROLE_ASSUMED_CODE",
    "ROLE_DROPPED_CODE",
    "ROLE_TAKEN_CODE",
    "STORAGE_CLASS_CODE",
    "STRAY_TEXT_CODE",
    "SYNTAX_CODE",
    "TEMPORAL_DROPPED_CODE",
    "TEMPORAL_TEXT_DROPPED_CODE",
    "TEXT_INFERRED_CODE",
    "UNKNOWN_ATTR_TYPE_CODE",
    "UNKNOWN_ELEMENT_CODE",
    "UNKNOWN_PARENT_CODE",
    "WEIGHT_KEY_CLASH_CODE",
    "WIDENING_UNSUPPORTED_CODE",
    "XML_ILLEGAL_CHAR_CODE",
    "XML_SYNTAX_CODE",
    // the shared importer and exporter machinery (8.4, 8.5)
    "TextCellWriter",
    "XmlSyntaxError",
    "XmlTokenizer",
    "declareCompanion",
    "declareResolved",
    "explicitWeights",
    "formatDecimal",
    "formatF32",
    "formatF64",
    "formatGmlReal",
    "formatInteger",
    "hasIllegalXmlChar",
    "isWeightField",
    "pairFolding",
    "parseWeightText",
    "reportUnusedOptions",
    "tokenizeXml",
    "xmlIllegalTextNotes",
];

describe("public barrel (design sections 8.2, 12.4, 13.1)", () => {
    it("exports exactly the documented value surface and no default export", () => {
        expect(Object.keys(graphIo).sort()).toEqual([...VALUE_EXPORTS].sort());
        expect((graphIo as Record<string, unknown>).default).toBeUndefined();
    });

    it("re-exports each format's importer and exporter under its format name", () => {
        const pairs: [string, graphIo.GraphImporter, graphIo.GraphExporter][] = [
            ["csv", graphIo.csvImporter, graphIo.csvExporter],
            ["dot", graphIo.dotImporter, graphIo.dotExporter],
            ["gexf", graphIo.gexfImporter, graphIo.gexfExporter],
            ["gml", graphIo.gmlImporter, graphIo.gmlExporter],
            ["graphml", graphIo.graphmlImporter, graphIo.graphmlExporter],
            ["json", graphIo.jsonImporter, graphIo.jsonExporter],
            ["neo4j", graphIo.neo4jImporter, graphIo.neo4jExporter],
            ["pajek", graphIo.pajekImporter, graphIo.pajekExporter],
        ];
        for (const [format, importer, exporter] of pairs) {
            expect(importer.format).toBe(format);
            expect(exporter.format).toBe(format);
            expect(graphIo.registry.importer(format)).toBe(importer);
            expect(graphIo.registry.exporter(format)).toBe(exporter);
            expect(importer.extensions.length).toBeGreaterThan(0);
            expect(typeof importer.sniff).toBe("function");
        }
        expect(new Set(graphIo.GRAPH_FORMATS)).toEqual(new Set(pairs.map(([format]) => format)));
    });

    it("keeps every issue and loss code table frozen with distinct string values", () => {
        const tables: Record<string, Readonly<Record<string, string>>> = {
            LOSS: graphIo.LOSS,
            CSV_ISSUE: graphIo.CSV_ISSUE,
            CSV_LOSS: graphIo.CSV_LOSS,
            DOT_ISSUE: graphIo.DOT_ISSUE,
            DOT_LOSS: graphIo.DOT_LOSS,
            GEXF_ISSUE: graphIo.GEXF_ISSUE,
            GEXF_LOSS: graphIo.GEXF_LOSS,
            GML_ISSUE: graphIo.GML_ISSUE,
            GML_LOSS: graphIo.GML_LOSS,
            GRAPHML_ISSUE: graphIo.GRAPHML_ISSUE,
            GRAPHML_LOSS: graphIo.GRAPHML_LOSS,
            JSON_ISSUE: graphIo.JSON_ISSUE,
            JSON_LOSS: graphIo.JSON_LOSS,
            NEO4J_ISSUE: graphIo.NEO4J_ISSUE,
            NEO4J_LOSS: graphIo.NEO4J_LOSS,
            PAJEK_ISSUE: graphIo.PAJEK_ISSUE,
            PAJEK_LOSS: graphIo.PAJEK_LOSS,
        };
        for (const [name, table] of Object.entries(tables)) {
            expect(Object.isFrozen(table), name).toBe(true);
            const values = Object.values(table);
            expect(values.length, name).toBeGreaterThan(0);
            expect(new Set(values).size, name).toBe(values.length);
            for (const value of values) {
                expect(value, `${name}: ${value}`).toMatch(/^[EW]_[A-Z0-9_]+$/);
            }
        }
    });

    it("round-trips a graph through importGraph and exportGraphToString", async () => {
        const { snapshot, report } = await graphIo.importGraph(
            "graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 value 2.5 ] ]",
        );
        expect(report.format).toBe("gml");
        expect(snapshot.edgeCount).toBe(1);
        const dot = await graphIo.exportGraphToString(snapshot, "dot");
        expect(dot).toContain("1 -- 2");
        const back = await graphIo.importGraph(dot, { format: "dot" });
        expect(back.snapshot.edgeList().weights?.[0]).toBe(2.5);
    });
});
