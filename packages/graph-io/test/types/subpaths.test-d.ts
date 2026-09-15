import { type GraphSink } from "@graphty/graph-format";
import { csvImporter as rootCsvImporter, importGraph, type ImportReport, type SniffResult } from "@graphty/graph-io";
import { CSV_ISSUE, csvExporter, csvImporter, type CsvImportOptions } from "@graphty/graph-io/csv";
import { dotExporter, dotImporter } from "@graphty/graph-io/dot";
import { GEXF_ISSUE, gexfExporter, gexfImporter, type GexfVersion } from "@graphty/graph-io/gexf";
import { GML_ISSUE, GML_LOSS, gmlExporter, gmlImporter } from "@graphty/graph-io/gml";
import { GRAPHML_ISSUE, graphmlExporter, graphmlImporter } from "@graphty/graph-io/graphml";
import { type JsonDialect, jsonExporter, jsonImporter } from "@graphty/graph-io/json";
import { NEO4J_ISSUE, neo4jExporter, neo4jImporter } from "@graphty/graph-io/neo4j";
import { pajekExporter, pajekImporter } from "@graphty/graph-io/pajek";
import { expectTypeOf } from "vitest";

// Every subpath exposes an importer / exporter pair typed by the 12.4 contract, and the root
// re-exports the same objects (the same type, the strict-consumer compile proves the shims agree).
expectTypeOf(csvImporter).toEqualTypeOf(rootCsvImporter);
expectTypeOf(csvImporter.format).toBeString();
expectTypeOf(csvExporter.capabilities.idCharset).toEqualTypeOf<"any" | "nmtoken" | "integer" | "dense-1-based">();
expectTypeOf(dotImporter.import).parameter(1).toEqualTypeOf<GraphSink>();
expectTypeOf(dotExporter.exportToString).returns.resolves.toBeString();
expectTypeOf(gexfImporter.import).returns.resolves.toEqualTypeOf<ImportReport>();
expectTypeOf(gexfExporter.format).toBeString();
expectTypeOf<GexfVersion>().toEqualTypeOf<"1.2" | "1.3">();
expectTypeOf(gmlImporter.format).toBeString();
expectTypeOf(gmlExporter.format).toBeString();
expectTypeOf(graphmlImporter.format).toBeString();
expectTypeOf(graphmlExporter.format).toBeString();
expectTypeOf(jsonImporter.format).toBeString();
expectTypeOf(jsonExporter.format).toBeString();
expectTypeOf<JsonDialect>().toEqualTypeOf<"node-link" | "d3" | "jgf" | "cytoscape" | "graphology" | "vis">();
expectTypeOf(neo4jImporter.format).toBeString();
expectTypeOf(neo4jExporter.format).toBeString();
expectTypeOf(pajekImporter.format).toBeString();
expectTypeOf(pajekExporter.format).toBeString();

// The grouped code tables are frozen string tables.
expectTypeOf(CSV_ISSUE.EMPTY_INPUT).toBeString();
expectTypeOf(GEXF_ISSUE.NOT_GEXF).toBeString();
expectTypeOf(GML_ISSUE.NO_GRAPH).toBeString();
expectTypeOf(GML_LOSS.RECORD_NUMBER_TYPE).toBeString();
expectTypeOf(GRAPHML_ISSUE.XML_SYNTAX).toBeString();
expectTypeOf(NEO4J_ISSUE.HEADER).toBeString();

// Format-specific options intersect with the common options under exactOptionalPropertyTypes.
const csvOptions: CsvImportOptions & { ids?: "canonical" | undefined } = { delimiter: ";", ids: "canonical" };
expectTypeOf(csvOptions.delimiter).toEqualTypeOf<string | undefined>();

// importGraph() accepts format-specific options next to the common ones and resolves to the result shape.
expectTypeOf(importGraph).parameter(1).toMatchTypeOf<{ format?: string | undefined } | undefined>();
expectTypeOf(importGraph("", { format: "csv", delimiter: ";", ids: "canonical" })).resolves.toHaveProperty("snapshot");
expectTypeOf(importGraph("")).resolves.toHaveProperty("sniff").toEqualTypeOf<SniffResult | null>();
