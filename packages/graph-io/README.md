# @graphty/graph-io

[![CI](https://github.com/graphty-org/graphty-monorepo/actions/workflows/ci.yml/badge.svg)](https://github.com/graphty-org/graphty-monorepo/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/graphty-org/graphty-monorepo/badge.svg?branch=master)](https://coveralls.io/github/graphty-org/graphty-monorepo?branch=master)
[![npm version](https://img.shields.io/npm/v/@graphty/graph-io.svg)](https://www.npmjs.com/package/@graphty/graph-io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Importers and exporters for the [@graphty/graph-format](https://www.npmjs.com/package/@graphty/graph-format)
snapshot: GEXF, GraphML, GML, DOT (Graphviz), Pajek NET, CSV / TSV, JSON (NetworkX node-link, d3,
JSON Graph Format, Cytoscape, graphology, vis.js) and Neo4j (`neo4j-admin import` CSV).

Every importer streams its input into a `GraphSink` (a `GraphBuilder` or your own sink) one scalar at
a time and reports what it could not represent instead of dropping it; every exporter says what it
would lose before it writes a byte. The core format package stays zero-dependency; this package owns
the parsers and the io contract types.

## Installation

```bash
npm install @graphty/graph-io @graphty/graph-format
```

ESM only. Node >= 18.19.0 or any browser with ES2020 support. `@graphty/graph-format` is a peer
dependency so an application that installs several consumers gets one copy of the format.

## Quick start

`importGraph()` sniffs the format from the file name, the MIME type and the first bytes, creates a
builder, imports and freezes:

```ts
import { importGraph, exportGraphToString, ImportError } from "@graphty/graph-io";

const bytes = await fetch("/data/lesmiserables.gexf").then((r) => r.arrayBuffer());
try {
    const { snapshot, report, freeze, format } = await importGraph(new Uint8Array(bytes), {
        filename: "lesmiserables.gexf", // a hint; the content decides
        ids: "canonical", // "1" becomes the number 1, "01" stays a string (the default for text formats)
        weightDtype: "f64", // the default: 0.1 and 16777217 survive the round trip
        errorLimit: 100, // recoverable errors tolerated before ImportError
    });
    console.log(format, snapshot.nodeCount, snapshot.edgeCount, report.warningCount, freeze.compacted);
    for (const issue of report.issues) {
        console.log(issue.severity, issue.code, issue.line, issue.message);
    }
    const gml = await exportGraphToString(snapshot, "gml");
} catch (err) {
    if (err instanceof ImportError) {
        // the input could not be read at all, or the error limit was reached
        console.log(err.report.issues);
    }
}
```

A caller who owns a builder (an application that appends several files into one graph) uses an
importer directly through its subpath, so a CSV-only bundle never loads the XML formats:

```ts
import { GraphBuilder } from "@graphty/graph-format";
import { csvImporter, csvExporter } from "@graphty/graph-io/csv";

const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
const report = await csvImporter.import(file.stream(), builder, {
    delimiter: ",",
    nodes: nodeTableText, // an optional paired node table (Gephi's nodes.csv + edges.csv)
    onProgress: (done, total) => console.log(done, total),
    signal: controller.signal,
});
const snapshot = builder.freeze();

const notes = csvExporter.check(snapshot, { dialect: "gephi" }); // what export() would lose; [] when exact
for await (const chunk of csvExporter.export(snapshot, { dialect: "gephi" })) {
    writable.write(chunk); // Uint8Array chunks
}
```

Inputs may be a `string`, a `Uint8Array`, a `ReadableStream<Uint8Array>` (a `File.stream()`, a fetch
body) or an async iterable of text or byte chunks. Bytes are decoded as UTF-8 with `fatal: true`, so
an invalid sequence is a `parse-error`, never a silent U+FFFD.

### Common import options

| Option                        | Default                                                               | Meaning                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ids`                         | `"canonical"` (text formats), `"keep"` (JSON)                         | How an id cell becomes a `NodeId`: canonical integer text becomes a number, everything else stays a string; `"string"`, `"number"` and `"keep"` too.                        |
| `nodeIdFrom`                  | `"id"`                                                                | `"label"` or `"index"` for GML / Pajek / d3 files whose ids are ambiguous.                                                                                                  |
| `weightFrom`                  | `"weight"` (GML `"value"`)                                            | The attribute that becomes THE edge weight; `null` = unweighted.                                                                                                            |
| `weightDtype`                 | `"f64"`                                                               | Weight staging precision; `"f32"` is an explicit opt-in.                                                                                                                    |
| `onMixedDirection`            | `"expand"`                                                            | A file whose edges disagree on direction: expand into a directed graph with `graphty.directed` / `graphty.pair` columns, force `"directed"` / `"undirected"`, or `"error"`. |
| `defaultDirected`             | per format (GEXF / GML / JSON undirected, DOT / CSV / Pajek directed) | The direction assumed when the file says nothing.                                                                                                                           |
| `addMissingNodes`             | `true` (GEXF `false`)                                                 | Whether an edge may name a node the file never declared.                                                                                                                    |
| `duplicateEdges`, `selfLoops` | `"keep"`                                                              | Builder policies, applied at freeze.                                                                                                                                        |
| `long`                        | `"f64"`                                                               | How a declared 64-bit integer column is stored (`"string"` keeps every digit).                                                                                              |
| `restoreMangledIds`           | `true`                                                                | Restore ids an exporter rewrote under `sanitizeIds: "mangle"` from the `graphty:originalId` attribute.                                                                      |
| `hyperedges`                  | `"skip"`                                                              | GraphML / JGF hyperedges: `"error"`, `"skip"` with a report entry, `"star"` or `"clique"`.                                                                                  |
| `errorLimit`                  | `100`                                                                 | Recoverable errors tolerated before the importer throws `ImportError` with the partial report.                                                                              |
| `signal`, `onProgress`        |                                                                       | Cancellation (rejects with the signal's reason) and byte progress (`bytesTotal` known for in-memory input).                                                                 |

On a caller's builder the builder-policy options (`addMissingNodes`, `duplicateEdges`, `selfLoops`,
`weightDtype`) are read from the sink; an explicit request the sink does not honour is reported once
as a `W_SINK_OPTION` warning. `importGraph()` seeds its own builder from them.

### Common export options

`sanitizeIds: "error" | "mangle"` (default `"error"`: an exporter never silently renames a node;
`"mangle"` rewrites ids the format cannot hold and keeps the original in `graphty:originalId`) and
`onMixedDirection: "error" | "directed" | "undirected"` for formats without mixed-direction support.
`check(snapshot, options)` returns the `LossNote[]` that `export()` would incur; the shared codes
are in `LOSS` (for example `LOSS.MIXED_DIRECTION`, `LOSS.DTYPE`, `LOSS.LIST`, `LOSS.ID_MANGLED`),
the per-format ones in `<FORMAT>_LOSS`.

## Formats

Subpath, extensions, and what each exporter keeps as declared (its `capabilities` table; `check()`
reports every column or feature outside it):

| Format  | Subpath                     | Extensions                         | Mixed dir.  | Multi-edges | Edge ids    | Id charset    | Dtypes kept                            | Lists | json | Defaults | Hierarchy | Temporal       | Graph attrs | Positions | Viz |
| ------- | --------------------------- | ---------------------------------- | ----------- | ----------- | ----------- | ------------- | -------------------------------------- | ----- | ---- | -------- | --------- | -------------- | ----------- | --------- | --- |
| GEXF    | `@graphty/graph-io/gexf`    | `.gexf`                            | yes         | yes (1.3)   | optional    | any           | f32 f64 i32 bool dict string           | yes   | no   | yes      | yes       | dynamic-values | no          | yes       | yes |
| GraphML | `@graphty/graph-io/graphml` | `.graphml` `.xml`                  | yes         | yes         | optional    | NMTOKEN       | bool i32 f32 f64 string (long as text) | no    | no   | yes      | yes       | none           | yes         | no        | no  |
| GML     | `@graphty/graph-io/gml`     | `.gml`                             | no          | yes         | optional    | integer       | i32 f64 string dict json               | yes   | yes  | no       | no        | none           | yes         | yes       | no  |
| DOT     | `@graphty/graph-io/dot`     | `.dot` `.gv`                       | no          | yes         | optional    | any           | bool i32 f64 string                    | no    | no   | no       | yes       | none           | yes         | yes       | no  |
| Pajek   | `@graphty/graph-io/pajek`   | `.net` `.paj`                      | yes         | yes         | none        | dense 1-based | f64 i32 bool string                    | no    | no   | no       | no        | spells         | no          | yes       | no  |
| CSV     | `@graphty/graph-io/csv`     | `.csv` `.tsv` `.edges` `.edgelist` | yes         | yes         | optional    | any           | bool i32 f64 string dict               | no    | no   | no       | no        | none           | no          | no        | no  |
| JSON    | `@graphty/graph-io/json`    | `.json`                            | per dialect | yes         | per dialect | any           | f64 i32 bool string (no declarations)  | no    | yes  | no       | Cytoscape | none           | per dialect | Cytoscape | no  |
| Neo4j   | `@graphty/graph-io/neo4j`   | `.csv` `.tsv`                      | no          | yes         | none        | any           | f32 f64 i32 bool string                | yes   | no   | no       | no        | none           | no          | no        | no  |

Every importer reads the whole corpus of research note 07 with the manifest counts and every
exporter round-trips it (import -> export -> import gives the same ids, topology, orientation,
weights and declared columns; the per-format caveats, such as Pajek's 0-based files or JSON's
untyped columns, are listed in the staging STATUS.md). `check()` predicts every difference the
format's own importer produces on re-import, including the importer's own rules, with one code per
concept across the formats (`W_ROLE_DROPPED` a role column written as a plain attribute,
`W_COLUMN_NAME_CHANGED` a role column read back under the importer's fixed name, `W_ROLE_ASSUMED` a
role-less column read back with a role, `W_WEIGHT_KEY_CLASH` a plain `weight` column read back as
THE weight, `W_ID_TEXT_TYPE` ids whose text reads back as the other type, `W_STORAGE_CLASS_CHANGED`
a string / dict column read back as the other, `W_INTEGRAL_F64_AS_I32`, `W_TEXT_INFERRED`,
`W_EMPTY_COLUMN_DROPPED`, `W_OPTIONS_GAINED`, `W_TEMPORAL_TEXT_DROPPED`, `W_MUTUAL_EXPANDED` /
`W_MUTUAL_AS_UNDIRECTED`, `E_XML_ILLEGAL_CHAR`, `E_ID_TEXT_COLLISION`). Every subpath exports its
`<FMT>_ISSUE` and `<FMT>_LOSS` tables; a key is the code without its `E_` / `W_` and format
prefixes. Under `onMixedDirection: "directed"` every exporter without mixed direction folds an
expanded pair back to one directed edge; `"undirected"` writes the whole graph undirected. Known
losses and format rules, in addition to the table:

- **GEXF**: 1.3 by default, 1.2 on request (`version: "1.2"`, no parallel edges, edge ids required).
  Dynamic attribute values become `temporal:<node|edge>:<name>` extension tables; `viz:color` is an
  f32 x4 rgba column in 0..1, `viz:position` an f32 x3 column; XML-derived column names (`label`,
  `parent`, `start`, ...) are reserved, a declared attribute with such a title is renamed
  `<title>#<id>`. Node ids that are non-integer numbers read back as text and string ids of integer
  text as numbers (`W_ID_TEXT_TYPE`; the importer reads ids by the canonical rule whatever `idtype`
  says, `ids: "string"` keeps the texts). A dict column without declared options gains one from
  its dictionary (`W_OPTIONS_GAINED`); text with a character XML 1.0 forbids is refused
  (`E_XML_ILLEGAL_CHAR`, `export()` throws).
- **GraphML**: parsed by the shared streaming XML tokenizer (no whole-document tree). `key for="all"`
  is declared in the node, edge and graph tables; yFiles trees are kept as `json` columns (structure
  preserved, not byte-exact); any other `json` column is written as JSON text and reads back as
  string (`W_JSON_UNSUPPORTED`). Ids outside NMTOKEN need `sanitizeIds: "mangle"` (restored on
  re-import). A label role column is written as the key titled `label` (the importer's label slot;
  `W_COLUMN_NAME_CHANGED` when it was named otherwise); edge ids and ports are the XML attributes
  (`id`, `sourceport`, `targetport`); a plain column titled like one of them reads back renamed
  `<name>#<key>`. A mutual pair is written as one undirected edge (`W_MUTUAL_AS_UNDIRECTED`).
  Lists, positions, viz and temporal columns are written as JSON text or reported.
- **GML**: NetworkX conventions (`_networkx_list_start`, `#` comments, `+INF` / `-INF` / `NAN`);
  `real` columns are written with a decimal point so the dtype survives; `graphics [ x y z ]` maps
  to the position role; records map to `json` and `check()` reports `W_GML_RECORD_NUMBER_TYPE` for
  numbers inside them (GML cannot keep int versus real inside a record). Node ids must be integers
  (`sanitizeIds: "mangle"` renumbers and keeps the original in `graphty_originalId`); column names
  outside `[A-Za-z][0-9A-Za-z_]*` are refused or mangled (`sanitizeKeys`).
- **DOT**: a Graphviz-faithful parser (grammar violations are fatal, as in Graphviz); clusters are
  container nodes with the `parent` role; ports are kept; HTML strings keep their brackets; `pos`
  maps to the position role. Mixed direction is folded per `onMixedDirection`; a text with a
  backslash before a quote or at its end cannot be written (`E_DOT_TRAILING_BACKSLASH`: Graphviz's
  scanner consumes backslash pairs, so such a text has no quoted spelling).
- **Pajek**: `*Vertices N` bounds the id space (ids 1..N; a 0-based file is detected and reported;
  a count the sink cannot reserve is fatal); `*Arcs` / `*Edges` sections give per-section
  direction; time intervals map to the spells role; vertex / line parameters are plain columns read
  through the 5.1 text grammar (`2.0` stays f64, lexical forms of a string column are kept); a
  `.paj` project file's `*Partition` / `*Vector` sections are skipped with an issue. Nodes are
  always written 1..N (`W_ID_RENUMBERED`: the id text is kept as the label of a node without a
  label value, a node with one loses its id; `W_PAJEK_LABEL_GAINED` when a line's parameters force
  a label); `sanitizeIds: "mangle"` also writes every renumbered vertex's original id as a
  `graphty_originalId` parameter, which the importer restores under `restoreMangledIds` (the
  default; `W_ID_TEXT_TYPE` when a string id of integer text reads back as a number); a `shape`
  column whose values are not all shape keywords is written as a parameter; labels holding a
  double quote or a line break cannot be written.
- **CSV**: header names resolve the endpoints (`source` / `target`, `from` / `to`, Gephi `Source` /
  `Target` / `Type` / `Id` / `Label` / `Weight`); a paired node table comes through the `nodes`
  option; the delimiter is sniffed. Leading `#` (SNAP) and `%` (KONECT) comment lines are skipped
  and read for the direction they declare (`# Directed graph`, `% sym` / `% asym`). A quoted empty
  cell is a set empty string, a bare one is unset; text after a closing quote is a fatal
  `E_CSV_QUOTE`. The Gephi dialect keeps per-row direction, the generic dialect drops it
  (`W_CSV_DIRECTION_DROPPED`). Untyped cells follow the 5.1 text grammar per column (`2.0` stays
  f64, `1e5` and `-0` keep their spelling in a string column). An edge table cannot carry an
  isolated node or the node order (`W_CSV_ISOLATED_NODES`, `W_CSV_NODE_ORDER`; write the node table).
- **JSON**: the dialect is sniffed from the document (`dialect` forces it); the importer records the
  shape under `meta.extra.json` so a re-export keeps it (a d3 document is written back bare, a
  graphology one with only the options it declared). JSON declares no types: the capability table
  lists the inferred dtypes only, and `check()` names every f32 / u32 / u8 / dict / list / vector
  column (they come back f64 / i32 / string / json), integral f64 columns (i32) and every role the
  dialect has no slot for. Edge ids exist in JGF, Cytoscape, graphology and vis only; positions in
  Cytoscape only. A repeated node id is merged with `W_DUPLICATE_NODE`, a repeated edge id skipped
  with `E_DUPLICATE_EDGE_ID`; an out-of-range d3 index link is `E_BAD_INDEX`. Non-finite numbers
  are written as `null` and reported.
- **Neo4j**: `neo4j-admin import` headers (`:ID`, `:LABEL`, `:START_ID`, `:END_ID`, `:TYPE`, typed
  properties, id spaces, arrays); one file may hold several sections; a `weight` property becomes
  THE weight; a quoted empty `:ID` is the id `""`. Everything is directed (an undirected snapshot,
  or the folded pairs of a mixed one under `onMixedDirection: "directed"` / `"undirected"`, is
  written with a `W_NEO4J_UNDIRECTED_AS_DIRECTED` note); `.text` companions keep the source text of
  temporal values whose canonical form differs; a dict column reads back as string and a position
  or visual column as a plain property.

## Format detection

`sniff({ filename?, mimeType?, head? })` ranks the registered importers: a content match scores
at least 0.5 (`0.5 + 0.35 * content + 0.1 * extension + 0.05 * MIME`), a hint alone at most 0.4,
so the content always beats a misleading extension (a `.csv` with a neo4j-admin header is Neo4j, a
`.xml` is GEXF or GraphML by its root element). `sniffJsonDialectHead()` reports the JSON dialect a
head suggests. `importGraph()` reads at most `SNIFF_HEAD_BYTES` (8 KiB) of a stream before deciding
and replays them to the importer.

## The import report

Every import returns an `ImportReport` (design section 8.6):

```ts
interface ImportReport {
    format: string; // "gexf", ...
    counts: { nodes; edges; skippedNodes; skippedEdges; expandedMixed }; // edges counts both halves of an expanded edge
    issues: ImportIssue[]; // in order: { category, severity, code, message, line, element }
    errorCount: number; // issues with severity "error"; they count toward errorLimit
    warningCount: number;
    truncated: boolean; // the error limit was reached and the import aborted
    lossy: LossNote[]; // what the importer could not represent: { code, message, column, count }
    durationMs: number; // the parse phase; the freeze reports separately
}
```

Issue categories are `parse-error`, `missing-value`, `validation-error`, `unsupported`, `precision`,
`coercion` and `merged`. Codes are stable strings (`E_*` errors, `W_*` warnings) exported per format
(`GEXF_ISSUE`, `GRAPHML_ISSUE`, `GML_ISSUE`, `DOT_ISSUE`, `PAJEK_ISSUE`, `CSV_ISSUE`,
`JSON_ISSUE`, `NEO4J_ISSUE`) and shared across formats (`SINK_OPTION_CODE`, `ID_MERGED_CODE`,
`DIRECTION_REFUSED_CODE`, `DIRECTION_FORCED_CODE`, `RENAMED_CODE`, `PRECISION_CODE`,
`INVALID_UTF8_CODE`, `PARSE_ERROR_CODE`). The builder throws on the first hard error; the importer
catches it per element, records an issue, skips the element and continues until `errorLimit`, then
throws `ImportError` (`code === "E_IMPORT"`) carrying the partial report. An input that cannot be
read at all (invalid UTF-8, malformed XML, no recognisable format) is an `ImportError` at once.

## Writing a plugin

`GraphImporter` and `GraphExporter` (design section 12.4) are plain objects; the helpers every
built-in format is built on are exported for third-party plugins: `ImportReportBuilder` (issues,
error limit, `warnOnce`, `ImportError`), `textChunks` / `readText` / `LineReader` (streaming UTF-8
input with cancellation and progress), `tokenizeXml` / `XmlTokenizer` (the streaming XML tokenizer
behind GEXF and GraphML), `resolveImportOptions` / `resolveExportOptions` / `reportSinkOptions` /
`reportUnusedOptions`, `DirectionResolver` (the mixed-direction rules of section 8.4) and
`pairFolding` (its inverse for exporters), `IdCoercer`, `parseTextCell` and `TextCellWriter` (the
text grammar of section 5.1, per column, with the lexical forms kept), `declareResolved` (the 5.6
collision rule), `explicitWeights` (the weight role column's validity, section 3.7),
`checkCapabilities` (with `CheckExtras` for the roles a format has a slot for) / `sanitizeIds` /
`LOSS` / `xmlIllegalTextNotes`, the shared issue and loss codes of `codes.ts`, `formatDecimal` and
`encodeChunks` / `joinText`. Register a plugin with `registry.registerImporter()` /
`registerExporter()` or build a registry of your own with `new FormatRegistry()`. The `children`
CSR helper (`childrenCsr`) inverts a `parent` / `parents` containment column for nested writers.

## License

MIT
