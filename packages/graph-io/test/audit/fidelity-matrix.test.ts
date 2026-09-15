/**
 * Fidelity audit, round-trip and cross-format lens (design sections 8.5 and 16.5).
 *
 * For EVERY corpus file (every format, the manifest entries plus the parseable files the manifests
 * omit) and the two synthetic GEXF documents that carry extension tables and temporal companions:
 *
 * - same format: import -> check() -> export -> re-import must be an equal snapshot on ids, arcs,
 *   orientation, weights (explicitness included), every declared column (dtype, role, origin type,
 *   defaults, options, every value and its validity), graph meta and extension tables, and check()
 *   must list exactly the documented losses (none, except the CSV node-table note and the Pajek
 *   renumbering of 0-based files);
 * - every ordered pair of formats: export to the other format after check(), re-import, and assert
 *   (a) every difference the re-import reveals is predicted by a loss note (no unreported loss),
 *   (b) every loss note corresponds to an observable difference (no phantom note), and
 *   (c) whatever check() did not flag survives byte-exact in value.
 *
 * The notes -> differences model is data (EXPLAINS below): a note explains the kinds of difference
 * its documented meaning covers, on the column it names (or its renamed counterpart) or globally.
 * A difference no note covers is a defect of the exporter's check() or of the pair; the failing
 * tests below pin those defects with the file, the column and the values.
 */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { countUnrepresentableIds } from "../../src/common/export.js";
import { csvExporter } from "../../src/formats/csv/exporter.js";
import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotExporter } from "../../src/formats/dot/exporter.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfExporter } from "../../src/formats/gexf/exporter.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { gmlExporter } from "../../src/formats/gml/exporter.js";
import { gmlImporter } from "../../src/formats/gml/importer.js";
import { graphmlExporter } from "../../src/formats/graphml/exporter.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { jsonExporter } from "../../src/formats/json/exporter.js";
import { jsonImporter } from "../../src/formats/json/importer.js";
import { neo4jExporter } from "../../src/formats/neo4j/exporter.js";
import { neo4jImporter } from "../../src/formats/neo4j/importer.js";
import { pajekExporter } from "../../src/formats/pajek/exporter.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import {
    type CommonExportOptions,
    type CommonImportOptions,
    type GraphExporter,
    type GraphImporter,
    type ImportReport,
    type LossNote,
} from "../../src/types.js";
import { DYNAMIC_1_3, OPEN_1_2 } from "../formats/gexf/fixtures.js";
import { CORPUS_FORMATS, CORPUS_ROOT, corpusFiles, type CorpusFormat } from "../helpers/corpus.js";
import { compareSnapshots, describeDiffs, type SnapshotDiff, valuesEqual } from "../helpers/roundtrip.js";

// ============================================================ the format pairs

type AnyExportOptions = Record<string, unknown> & CommonExportOptions;
type AnyImportOptions = Record<string, unknown> & CommonImportOptions;

interface Pair {
    readonly exporter: GraphExporter<AnyExportOptions>;
    readonly importer: GraphImporter<AnyImportOptions>;
}

const PAIRS: Readonly<Record<CorpusFormat, Pair>> = {
    csv: {
        exporter: csvExporter as GraphExporter<AnyExportOptions>,
        importer: csvImporter as GraphImporter<AnyImportOptions>,
    },
    dot: {
        exporter: dotExporter as GraphExporter<AnyExportOptions>,
        importer: dotImporter as GraphImporter<AnyImportOptions>,
    },
    gexf: {
        exporter: gexfExporter as GraphExporter<AnyExportOptions>,
        importer: gexfImporter as GraphImporter<AnyImportOptions>,
    },
    gml: {
        exporter: gmlExporter as GraphExporter<AnyExportOptions>,
        importer: gmlImporter as GraphImporter<AnyImportOptions>,
    },
    graphml: {
        exporter: graphmlExporter as GraphExporter<AnyExportOptions>,
        importer: graphmlImporter as GraphImporter<AnyImportOptions>,
    },
    json: {
        exporter: jsonExporter as GraphExporter<AnyExportOptions>,
        importer: jsonImporter as GraphImporter<AnyImportOptions>,
    },
    neo4j: {
        exporter: neo4jExporter as GraphExporter<AnyExportOptions>,
        importer: neo4jImporter as GraphImporter<AnyImportOptions>,
    },
    pajek: {
        exporter: pajekExporter as GraphExporter<AnyExportOptions>,
        importer: pajekImporter as GraphImporter<AnyImportOptions>,
    },
};

/** Formats without mixed direction: the export of an expanded snapshot needs a policy. */
const NO_MIXED_DIRECTION: ReadonlySet<CorpusFormat> = new Set(["dot", "gml", "json", "neo4j"]);

/** The export options a target needs so that export() can run on any corpus snapshot. */
function exportOptionsFor(target: CorpusFormat): AnyExportOptions {
    const options: AnyExportOptions = {};
    if (target === "graphml" || target === "gml") {
        options.sanitizeIds = "mangle";
    }
    if (target === "gml") {
        options.sanitizeKeys = "mangle";
    }
    if (NO_MIXED_DIRECTION.has(target)) {
        options.onMixedDirection = "directed";
    }
    return options;
}

// ============================================================ the inputs

interface Input {
    /** "gexf/minimal.gexf" or "synthetic/DYNAMIC_1_3". */
    readonly label: string;
    readonly format: CorpusFormat;
    readonly text: string;
    /** Importer options the file needs (a tab delimiter, a paired node or relationship file). */
    readonly importOptions: AnyImportOptions;
    /** Whether the file is a node table (CSV) with no edge rows. */
    readonly nodeTable: boolean;
}

/** Files under the corpus directories that are not graphs at all (a saved HTTP error page). */
const NOT_A_GRAPH: ReadonlySet<string> = new Set(["graphml/got-social-network.graphml"]);

function corpusText(format: CorpusFormat, name: string): string {
    return readFileSync(join(CORPUS_ROOT, format, name), "utf-8");
}

function importOptionsFor(format: CorpusFormat, name: string): AnyImportOptions {
    if (format === "neo4j" && name === "crlf-tabs.tsv") {
        return { delimiter: "\t" };
    }
    if (format === "neo4j" && name === "movies-nodes.csv") {
        return { relationships: [corpusText("neo4j", "movies-rels.csv")] };
    }
    if (format === "csv" && name === "got-edges.csv") {
        return { nodes: corpusText("csv", "got-nodes.csv") };
    }
    return {};
}

function allInputs(): Input[] {
    const inputs: Input[] = [];
    for (const format of CORPUS_FORMATS) {
        const listed = corpusFiles(format).map((f) => f.path);
        const names = [...listed];
        for (const name of readdirSync(join(CORPUS_ROOT, format)).sort()) {
            if (name !== "manifest.json" && !listed.includes(name)) {
                names.push(name);
            }
        }
        for (const name of names) {
            const label = `${format}/${name}`;
            if (NOT_A_GRAPH.has(label)) {
                continue;
            }
            inputs.push({
                label,
                format,
                text: corpusText(format, name),
                importOptions: importOptionsFor(format, name),
                nodeTable: format === "csv" && name === "got-nodes.csv",
            });
        }
    }
    inputs.push({
        label: "synthetic/DYNAMIC_1_3",
        format: "gexf",
        text: DYNAMIC_1_3,
        importOptions: {},
        nodeTable: false,
    });
    inputs.push({ label: "synthetic/OPEN_1_2", format: "gexf", text: OPEN_1_2, importOptions: {}, nodeTable: false });
    return inputs;
}

const INPUTS = allInputs();

interface Loaded {
    readonly snapshot: GraphSnapshot;
    readonly report: ImportReport;
}

async function load(format: CorpusFormat, text: string, options: AnyImportOptions, directed = true): Promise<Loaded> {
    const builder = new GraphBuilder({ directed, weightDtype: "f64" });
    const report = await PAIRS[format].importer.import(text, builder, options);
    return { snapshot: builder.freeze(), report };
}

const originals = new Map<string, Promise<Loaded>>();

function original(input: Input): Promise<Loaded> {
    let loaded = originals.get(input.label);
    if (loaded === undefined) {
        loaded = load(input.format, input.text, input.importOptions);
        originals.set(input.label, loaded);
    }
    return loaded;
}

// ============================================================ extra comparisons the helper lacks

const META_KEYS = [
    "name",
    "description",
    "creator",
    "created",
    "modified",
    "keywords",
    "timeFormat",
    "timeRepresentation",
    "declaredMultigraph",
    "weightOrigin",
    "extra",
] as const;

/**
 * Differences in graph meta (design 5.9). sourceVersion is exempt (an exporter writes its own
 * version); idType / mode / graphId may be added by an exporter that always writes a header value.
 */
function metaDiffs(expected: GraphSnapshot, actual: GraphSnapshot): SnapshotDiff[] {
    const diffs: SnapshotDiff[] = [];
    for (const key of META_KEYS) {
        const e: unknown = expected.meta[key];
        const a: unknown = actual.meta[key];
        if (key === "extra") {
            const ee = { ...(e as Record<string, unknown>) };
            const aa = { ...(a as Record<string, unknown>) };
            // graphml invents a graph id when the source had none
            const eg = ee.graphml;
            const ag = aa.graphml;
            if (typeof eg === "object" && eg !== null && typeof ag === "object" && ag !== null) {
                const egr = eg as Record<string, unknown>;
                const agr = ag as Record<string, unknown>;
                if (egr.graphId === null && agr.graphId === "G") {
                    aa.graphml = { ...agr, graphId: null };
                }
            }
            if (!valuesEqual(ee, aa, 0)) {
                diffs.push({
                    path: "meta.extra",
                    expected: ee,
                    actual: aa,
                    message: `meta.extra: expected ${JSON.stringify(ee)}, got ${JSON.stringify(aa)}`,
                });
            }
            continue;
        }
        if (!valuesEqual(e, a, 0)) {
            diffs.push({
                path: `meta.${key}`,
                expected: e,
                actual: a,
                message: `meta.${key}: expected ${JSON.stringify(e)}, got ${JSON.stringify(a)}`,
            });
        }
    }
    return diffs;
}

/** Declared defaults and options of matching columns, which compareSnapshots() does not compare. */
function declarationDiffs(expected: GraphSnapshot, actual: GraphSnapshot): SnapshotDiff[] {
    const diffs: SnapshotDiff[] = [];
    for (const [label, e, a] of [
        ["nodes", expected.nodes, actual.nodes],
        ["edges", expected.edges, actual.edges],
        ["graph", expected.graph, actual.graph],
    ] as const) {
        for (const column of e) {
            const other = a.get(column.meta.name);
            if (other === null) {
                continue;
            }
            if (!valuesEqual(column.meta.default, other.meta.default, 0)) {
                diffs.push({
                    path: `${label}.${column.meta.name}.default`,
                    expected: column.meta.default,
                    actual: other.meta.default,
                    message: `${label}.${column.meta.name}.default: expected ${JSON.stringify(column.meta.default)}, got ${JSON.stringify(other.meta.default)}`,
                });
            }
            if (!valuesEqual(column.meta.options, other.meta.options, 0)) {
                diffs.push({
                    path: `${label}.${column.meta.name}.options`,
                    expected: column.meta.options,
                    actual: other.meta.options,
                    message: `${label}.${column.meta.name}.options: expected ${JSON.stringify(column.meta.options)}, got ${JSON.stringify(other.meta.options)}`,
                });
            }
        }
    }
    return diffs;
}

// ============================================================ the notes -> differences model

/** What a difference is about. */
interface DiffKind {
    /** "nodes", "edges", "graph", "extension:<name>", or "" for the whole graph. */
    readonly table: string;
    /** The column name, or "" for the whole graph / a whole extension table. */
    readonly column: string;
    /** directed | nodeCount | edgeCount | ids | topology | weights | flags | value | dtype | itemDtype | components | role | missing | extra | length | default | options | rowCount | origin. */
    readonly what: string;
}

function kindOf(diff: SnapshotDiff): DiffKind {
    const { path } = diff;
    if (path === "directed" || path === "nodeCount" || path === "edgeCount") {
        return { table: "", column: "", what: path };
    }
    if (path.startsWith("ids[")) {
        return { table: "", column: "", what: "ids" };
    }
    if (path === "rowPtr" || path === "colIdx" || path.startsWith("edge[")) {
        return { table: "", column: "", what: "topology" };
    }
    if (path.startsWith("weight")) {
        return { table: "", column: "", what: "weights" };
    }
    if (path.startsWith("flags")) {
        return { table: "", column: "", what: "flags" };
    }
    if (path.startsWith("meta.")) {
        return { table: "meta", column: path.slice(5), what: "value" };
    }
    if (path.startsWith("extensions.")) {
        const rest = path.slice("extensions.".length);
        const dot = rest.indexOf(".");
        if (dot < 0) {
            return { table: `extension:${rest}`, column: "", what: diff.expected === "present" ? "missing" : "extra" };
        }
        const name = rest.slice(0, dot);
        const tail = rest.slice(dot + 1);
        if (tail === "rowCount") {
            return { table: `extension:${name}`, column: "", what: "rowCount" };
        }
        return columnKind(`extension:${name}`, tail, diff);
    }
    const dot = path.indexOf(".");
    return columnKind(path.slice(0, dot), path.slice(dot + 1), diff);
}

const SUFFIXES = ["dtype", "itemDtype", "components", "role", "origin.type", "length", "default", "options"];

function columnKind(table: string, tail: string, diff: SnapshotDiff): DiffKind {
    const bracket = /^(.*)\[(\d+)\]$/.exec(tail);
    if (bracket !== null) {
        return { table, column: bracket[1], what: "value" };
    }
    for (const suffix of SUFFIXES) {
        if (tail.endsWith(`.${suffix}`)) {
            return {
                table,
                column: tail.slice(0, -(suffix.length + 1)),
                what: suffix === "origin.type" ? "origin" : suffix,
            };
        }
    }
    if (diff.expected === "present" && diff.actual === "absent") {
        return { table, column: tail, what: "missing" };
    }
    if (diff.expected === "absent" && diff.actual === "present") {
        return { table, column: tail, what: "extra" };
    }
    return { table, column: tail, what: "value" };
}

/** The difference kinds a note explains: global kinds when column is null, column kinds otherwise. */
interface Explains {
    /** Whole-graph kinds and `table.column:what` patterns ("nodes.*:missing", "edges.graphty.*:*"). */
    readonly global?: readonly string[];
    /** Kinds on the note's own column (and its renamed counterpart). */
    readonly column?: readonly string[];
    /** The note is about the written file only; the importer restores the value (never a phantom). */
    readonly restored?: boolean;
    /** The note is not observable through the snapshot comparison (never a phantom). */
    readonly unobservable?: boolean;
}

const STRUCTURAL = ["edges.graphty.*:*", "topology", "flags", "edgeCount", "weights"];
const ID_TYPE = ["ids"];
const DIRECTION = ["directed", ...STRUCTURAL];
const DTYPE_CLASS = ["dtype", "itemDtype", "components", "value", "options", "origin"];
const MISSING_CLASS = ["missing", "rowCount", "extra", "role", ...DTYPE_CLASS];
const ROLE_CLASS = ["role", "origin"];
const RENAME_CLASS = ["missing", "extra", "value", "role", "origin", "dtype"];

const EXPLAINS: ReadonlyMap<string, Explains> = new Map<string, Explains>([
    // ids
    ["W_ID_RENUMBERED", { global: ["ids", "nodes.label:extra", "nodes.label:value"] }],
    ["W_ID_MANGLED", { restored: true }],
    ["W_ID_TYPE_CHANGED", { global: ID_TYPE }],
    ["W_ID_TEXT_TYPE", { global: ID_TYPE }],
    ["W_NUMERIC_IDS_STRINGIFIED", { global: ID_TYPE }],
    ["W_NODE_ORDER", { global: ["ids", "topology"] }],
    ["W_CSV_ISOLATED_NODES", { global: ["ids", "topology", "nodeCount", "nodes.*:*"] }],
    ["W_CSV_NODE_ORDER", { global: ["ids", "topology", "nodes.*:*"] }],
    ["W_GRAPHML_HIERARCHY_REORDERED", { global: ["ids", "topology", "nodes.*:*", "edges.*:*"] }],
    // direction and pairs
    ["W_NEO4J_UNDIRECTED_AS_DIRECTED", { global: DIRECTION }],
    ["W_CSV_DIRECTION_DROPPED", { global: DIRECTION }],
    ["W_DIRECTION_DROPPED", { global: DIRECTION }],
    ["W_MIXED_DIRECTION", { global: DIRECTION }],
    ["E_MIXED_DIRECTION", { global: DIRECTION }],
    ["W_MUTUAL_EXPANDED", { global: STRUCTURAL }],
    ["W_MUTUAL_DROPPED", { global: STRUCTURAL }],
    ["W_MUTUAL_AS_UNDIRECTED", { global: STRUCTURAL, column: STRUCTURAL }],
    ["W_MULTI_EDGES", { global: ["edgeCount", "topology", "flags"] }],
    ["W_SELF_LOOPS", { global: ["edgeCount", "topology", "flags"] }],
    // edge ids
    ["W_EDGE_IDS_GENERATED", { global: ["edges.id:extra", "edges.key:extra", "edges.Id:extra"] }],
    ["W_EDGE_IDS_DROPPED", { column: MISSING_CLASS }],
    ["W_GRAPHML_EDGE_ID_TEXT", { column: DTYPE_CLASS }],
    // whole tables
    ["W_GRAPH_ATTRIBUTES_DROPPED", { global: ["graph.*:missing"] }],
    ["W_CSV_NODE_TABLE", { global: ["nodes.*:missing"] }],
    ["W_NONFINITE_AS_NULL", { global: ["nodes.*:value", "edges.*:value", "graph.*:value", "weights"] }],
    // roles and names
    ["W_ROLE_DROPPED", { column: ROLE_CLASS }],
    ["W_CSV_ROLE_NAME", { column: ROLE_CLASS }],
    ["W_ROLE_ASSUMED", { column: ROLE_CLASS }],
    ["W_COLUMN_NAME_CHANGED", { column: RENAME_CLASS }],
    ["W_ROLE_SHAPE", { column: [...ROLE_CLASS, ...DTYPE_CLASS] }],
    ["W_POSITION_Z_DROPPED", { column: ["value"] }],
    ["W_ATTRIBUTE_RENAMED", { column: RENAME_CLASS }],
    ["W_GML_KEY_MANGLED", { column: RENAME_CLASS }],
    // the importer's own rules on re-import
    ["W_WEIGHT_KEY_CLASH", { global: ["weights", "flags"], column: MISSING_CLASS }],
    ["W_EMPTY_COLUMN_DROPPED", { column: MISSING_CLASS }],
    ["W_STORAGE_CLASS_CHANGED", { column: DTYPE_CLASS }],
    ["W_INTEGRAL_F64_AS_I32", { column: DTYPE_CLASS }],
    ["W_TEXT_INFERRED", { column: DTYPE_CLASS }],
    ["W_OPTIONS_GAINED", { column: ["options"] }],
    ["W_TEMPORAL_TEXT_DROPPED", { column: MISSING_CLASS }],
    // dtypes and values
    ["W_DTYPE_UNSUPPORTED", { column: DTYPE_CLASS }],
    ["W_CSV_TEXT_ROLE", { column: DTYPE_CLASS }],
    ["W_PAJEK_LABEL_AS_TEXT", { column: DTYPE_CLASS }],
    ["W_PAJEK_SHAPE_AS_PARAMETER", { column: DTYPE_CLASS }],
    ["W_PAJEK_LABEL_GAINED", { column: ["extra", "value", "missing"] }],
    ["W_NEO4J_DECLARED_TYPE_CHANGED", { column: DTYPE_CLASS }],
    ["W_GEXF_VIZ_DTYPE", { column: DTYPE_CLASS }],
    ["W_DECLARED_TYPE", { column: DTYPE_CLASS }],
    ["W_GML_RECORD_NUMBER_TYPE", { column: DTYPE_CLASS }],
    ["W_GML_RECORD_BOOLEAN", { column: DTYPE_CLASS }],
    ["W_GML_RECORD_NULL", { column: DTYPE_CLASS }],
    ["W_GML_JSON_ARRAY", { column: DTYPE_CLASS }],
    ["W_LIST_SEPARATOR", { column: DTYPE_CLASS }],
    ["W_TIMESTAMP_AS_INTERVAL", { column: DTYPE_CLASS }],
    ["W_CSV_EMPTY_VALUE", { column: ["value"] }],
    ["W_CSV_NONFINITE", { column: DTYPE_CLASS }],
    ["W_DOT_NON_FINITE", { column: DTYPE_CLASS }],
    ["W_PAJEK_NONFINITE_AS_TEXT", { column: DTYPE_CLASS }],
    ["W_NEO4J_ARRAY_DELIMITER", { column: DTYPE_CLASS }],
    // dropped columns
    ["W_HIERARCHY_DROPPED", { column: MISSING_CLASS }],
    ["W_VIZ_DROPPED", { column: MISSING_CLASS }],
    ["W_POSITIONS_DROPPED", { column: MISSING_CLASS }],
    ["W_TEMPORAL_DROPPED", { column: MISSING_CLASS }],
    ["W_SPELLS_DROPPED", { column: MISSING_CLASS }],
    ["W_OPEN_INTERVAL", { column: MISSING_CLASS }],
    ["W_DYNAMIC_VALUES_DROPPED", { column: MISSING_CLASS, unobservable: true }],
    ["W_EXTENSION_TABLE_DROPPED", { column: MISSING_CLASS }],
    ["W_CSV_RESERVED_NAME", { column: MISSING_CLASS }],
    ["W_RESERVED_KEY", { column: MISSING_CLASS }],
    ["W_PAJEK_KEY_DROPPED", { column: MISSING_CLASS }],
    ["W_DOT_ATTRIBUTE_CLASH", { column: MISSING_CLASS }],
    ["W_PARENTS_DROPPED", { column: MISSING_CLASS }],
    ["W_DOT_POSITION_SHAPE", { column: MISSING_CLASS }],
    ["W_PAJEK_POSITION_STRIDE", { column: MISSING_CLASS }],
    ["W_LIST_UNSUPPORTED", { column: MISSING_CLASS }],
    ["W_JSON_UNSUPPORTED", { column: MISSING_CLASS }],
    ["W_COMPONENTS_FLATTENED", { column: MISSING_CLASS }],
    ["W_GEXF_KIND_DROPPED", { column: MISSING_CLASS }],
    ["W_TEMPORAL_TABLE_SHAPE", { column: MISSING_CLASS }],
    ["W_VALUE_UNWRITABLE", { column: ["value"] }],
    // declarations
    ["W_DEFAULT_DROPPED", { column: ["default"] }],
    ["W_OPTIONS_DROPPED", { column: ["options", "dtype"] }],
    // export() throws: the error is the observable outcome
    ["E_ID_CHARSET", { unobservable: true }],
    ["E_ID_TEXT_COLLISION", { unobservable: true }],
    ["E_NEO4J_WEIGHT_COLUMN_TAKEN", { unobservable: true }],
    ["E_NEO4J_ID_COLUMN_TAKEN", { unobservable: true }],
    ["E_GML_INVALID_KEY", { unobservable: true }],
    ["E_GML_RESERVED_KEY", { unobservable: true }],
    ["E_GML_NESTED_ARRAY", { unobservable: true }],
    ["E_PAJEK_TEXT", { unobservable: true }],
    ["E_DOT_TRAILING_BACKSLASH", { unobservable: true }],
    ["E_XML_ILLEGAL_CHAR", { unobservable: true }],
    ["E_GRAPHML_YFILES_TREE", { unobservable: true }],
    // notes about the file the snapshot comparison cannot see
    ["W_PAJEK_FIRST_MODE_DROPPED", { unobservable: true }],
    ["W_NEO4J_MULTIPLE_ID_PROPERTIES", { unobservable: true }],
    ["W_GRAPHML_GRAPH_ID", { unobservable: true }],
    ["W_GRAPHML_PARENT_CYCLE", { unobservable: true }],
    ["W_GRAPHML_YFILES_JSON", { unobservable: true }],
]);

/** Whether a difference on `column` is on the note's column or a renamed form of it. */
function sameColumn(noteColumn: string, column: string): boolean {
    if (column === noteColumn) {
        return true;
    }
    if (column.startsWith(`${noteColumn}#`) || noteColumn.startsWith(`${column}#`)) {
        return true;
    }
    const mangled = noteColumn.replace(/[^A-Za-z0-9_]/g, "_");
    return column === mangled || column === `${mangled}_2`;
}

function matchesPattern(pattern: string, kind: DiffKind): boolean {
    if (!pattern.includes(":")) {
        return kind.table === "" && kind.what === pattern;
    }
    const colon = pattern.lastIndexOf(":");
    const what = pattern.slice(colon + 1);
    const where = pattern.slice(0, colon);
    if (what !== "*" && what !== kind.what) {
        return false;
    }
    const dot = where.indexOf(".");
    const table = where.slice(0, dot);
    const column = where.slice(dot + 1);
    if (table !== kind.table) {
        return false;
    }
    if (column === "*") {
        return true;
    }
    if (column.endsWith("*")) {
        return kind.column.startsWith(column.slice(0, -1));
    }
    return kind.column === column;
}

function explains(note: LossNote, kind: DiffKind): boolean {
    const model = EXPLAINS.get(note.code);
    if (model === undefined) {
        return false;
    }
    if (model.global !== undefined && model.global.some((pattern) => matchesPattern(pattern, kind))) {
        return true;
    }
    if (model.column !== undefined && note.column !== null) {
        if (kind.table.startsWith("extension:")) {
            const name = kind.table.slice("extension:".length);
            return name === note.column && model.column.includes(kind.what);
        }
        const onColumn = sameColumn(note.column, kind.column) || readsBackAs(note) === kind.column;
        return onColumn && (model.column.includes(kind.what) || model.column.includes("*"));
    }
    return false;
}

/** The name a rename note says the column reads back under (`... reads back as "<name>"`), or null. */
function readsBackAs(note: LossNote): string | null {
    const m = /reads back as "([^"]+)"/.exec(note.message);
    return m === null ? null : m[1];
}

// ============================================================ one trip

interface Trip {
    readonly notes: readonly LossNote[];
    readonly text: string | null;
    readonly exportError: { code: string; message: string } | null;
    readonly result: GraphSnapshot | null;
    readonly report: ImportReport | null;
    readonly importError: { code: string; message: string } | null;
    /** All differences, in helper order (the compare stops after a count mismatch). */
    readonly diffs: SnapshotDiff[];
    /** Whether the compare bailed early (a count mismatch), so later differences are unobservable. */
    readonly truncated: boolean;
}

function errorOf(err: unknown): { code: string; message: string } {
    const e = err as { code?: unknown; message?: unknown };
    return {
        code: typeof e.code === "string" ? e.code : "?",
        message: typeof e.message === "string" ? e.message : String(err),
    };
}

async function trip(
    snapshot: GraphSnapshot,
    target: CorpusFormat,
    exportOptions: AnyExportOptions,
    importOptions: AnyImportOptions,
): Promise<Trip> {
    const { exporter } = PAIRS[target];
    const notes = exporter.check(snapshot, exportOptions);
    let text: string;
    try {
        text = await exporter.exportToString(snapshot, exportOptions);
    } catch (err) {
        return {
            notes,
            text: null,
            exportError: errorOf(err),
            result: null,
            report: null,
            importError: null,
            diffs: [],
            truncated: true,
        };
    }
    let loaded: Loaded;
    try {
        loaded = await load(target, text, importOptions, snapshot.directed);
    } catch (err) {
        return {
            notes,
            text,
            exportError: null,
            result: null,
            report: null,
            importError: errorOf(err),
            diffs: [],
            truncated: true,
        };
    }
    const diffs = compareSnapshots(snapshot, loaded.snapshot, {
        allowExtraColumns: false,
        originType: true,
        limit: 100000,
    });
    const truncated = diffs.some((d) => d.path === "nodeCount" || d.path === "edgeCount" || d.path === "directed");
    if (!truncated) {
        diffs.push(...declarationDiffs(snapshot, loaded.snapshot));
    }
    return {
        notes,
        text,
        exportError: null,
        result: loaded.snapshot,
        report: loaded.report,
        importError: null,
        diffs,
        truncated,
    };
}

function noteText(notes: readonly LossNote[]): string {
    return notes.length === 0
        ? "(no notes)"
        : notes.map((n) => `  - ${n.code}${n.column === null ? "" : ` [${n.column}]`}: ${n.message}`).join("\n");
}

/** The difference kinds that change a value, a validity bit, a column's presence, an id or the topology. */
const VALUE_KINDS: ReadonlySet<string> = new Set([
    "directed",
    "nodeCount",
    "edgeCount",
    "ids",
    "topology",
    "weights",
    "flags",
    "value",
    "missing",
    "extra",
    "length",
    "default",
    "options",
    "rowCount",
]);

/** The difference kinds that change a declaration only (the values survive). */
const SCHEMA_KINDS: ReadonlySet<string> = new Set(["dtype", "itemDtype", "components", "role"]);

function unexplained(t: Trip, kinds: ReadonlySet<string>): SnapshotDiff[] {
    return t.diffs.filter((d) => {
        const kind = kindOf(d);
        if (!kinds.has(kind.what)) {
            // origin.type: a typed target format declares a type; the origin records where a column came from
            return false;
        }
        return !t.notes.some((n) => explains(n, kind));
    });
}

function phantoms(t: Trip): LossNote[] {
    if (t.truncated) {
        return [];
    }
    const kinds = t.diffs.map(kindOf);
    return t.notes.filter((n) => {
        const model = EXPLAINS.get(n.code);
        if (model === undefined) {
            return true;
        }
        if (model.restored === true || model.unobservable === true) {
            return false;
        }
        if (n.column !== null) {
            // a column note is exercised by any difference on that column (or its renamed form):
            // once a column is missing or renamed, its dtype or role cannot be observed
            const { column } = n;
            return !kinds.some(
                (k) =>
                    sameColumn(column, k.column) ||
                    (k.table.startsWith("extension:") && k.table.slice("extension:".length) === column),
            );
        }
        return !kinds.some((k) => explains(n, k));
    });
}

// ============================================================ same-format round trips

/**
 * The export options of an exact same-format trip: the CSV node table for a node-only file, id
 * mangling where GraphML cannot hold an id (restored on re-import), and the source's own GEXF
 * version (a 1.2 document with open intervals is exact as 1.2 only).
 */
function sameFormatExportOptions(input: Input, snapshot: GraphSnapshot): AnyExportOptions {
    const options: AnyExportOptions = {};
    if (input.nodeTable) {
        options.table = "nodes";
    }
    if (input.format === "graphml") {
        options.sanitizeIds = "mangle";
    }
    if (input.format === "gexf" && snapshot.meta.sourceVersion === "1.2") {
        options.version = "1.2";
    }
    return options;
}

describe("fidelity matrix: same-format round trips over every corpus file", () => {
    for (const input of INPUTS) {
        const { format } = input;
        it(`${input.label} -> ${format} -> ${format}: equal snapshot, meta and extension tables`, async () => {
            const { snapshot, report } = await original(input);
            expect(report.errorCount, `import of ${input.label} has errors`).toBe(0);
            const { exporter, importer } = PAIRS[format];

            const exportOptions = sameFormatExportOptions(input, snapshot);
            const notes = exporter.check(snapshot, exportOptions);
            const text = await exporter.exportToString(snapshot, exportOptions);
            const importOptions: AnyImportOptions = {};
            if (format === "pajek" && snapshot.ids.kind === "identity" && snapshot.ids.offset === 0) {
                importOptions.nodeIdFrom = "index";
            }
            if (format === "csv" && !input.nodeTable && snapshot.nodes.names().length > 0) {
                // the CSV edge table carries no node columns; the paired node table does
                importOptions.nodes = await exporter.exportToString(snapshot, { table: "nodes" });
            }
            if (input.nodeTable) {
                importOptions.table = "nodes";
            }
            const builder = new GraphBuilder({ directed: snapshot.directed, weightDtype: "f64" });
            const again = await importer.import(text, builder, importOptions);
            const result = builder.freeze();
            expect(
                again.errorCount,
                `re-import of ${input.label}:\n${JSON.stringify(again.issues.slice(0, 5), null, 1)}`,
            ).toBe(0);

            const documented: string[] = [];
            if (format === "pajek" && snapshot.ids.kind === "identity" && snapshot.ids.offset === 0) {
                documented.push("W_ID_RENUMBERED");
            }
            if (format === "graphml" && countUnrepresentableIds(snapshot, "nmtoken") > 0) {
                documented.push("W_ID_MANGLED");
            }
            if (format === "gexf" && exportOptions.version === "1.2" && snapshot.flags.multigraph) {
                // 1.2 has no parallel-edge kind: the expanded mirror of a mutual pair next to a directed edge counts
                documented.push("W_MULTI_EDGES");
            }
            if (format === "csv" && !input.nodeTable && snapshot.nodes.names().length > 0) {
                documented.push("W_CSV_NODE_TABLE");
            }
            expect(
                notes.map((n) => n.code),
                `check() notes of ${input.label}:\n${noteText(notes)}`,
            ).toEqual(documented);

            const diffs = [
                ...compareSnapshots(snapshot, result, { allowExtraColumns: false, originType: true, limit: 100000 }),
                ...declarationDiffs(snapshot, result),
            ];
            expect(diffs.length, `${input.label}: ${diffs.length} difference(s):\n${describeDiffs(diffs)}`).toBe(0);
            expect(result.extensions.size).toBe(snapshot.extensions.size);
        });

        it(`${input.label} -> ${format} -> ${format}: graph meta survives`, async () => {
            const { snapshot } = await original(input);
            const { exporter, importer } = PAIRS[format];
            const exportOptions = sameFormatExportOptions(input, snapshot);
            const text = await exporter.exportToString(snapshot, exportOptions);
            const builder = new GraphBuilder({ directed: snapshot.directed, weightDtype: "f64" });
            const importOptions: AnyImportOptions = input.nodeTable ? { table: "nodes" } : {};
            await importer.import(text, builder, importOptions);
            const result = builder.freeze();
            const diffs = metaDiffs(snapshot, result);
            expect(diffs.length, `${input.label}: meta differs:\n${describeDiffs(diffs)}`).toBe(0);
        });
    }
});

// ============================================================ cross-format round trips

describe("fidelity matrix: every ordered pair of formats", () => {
    for (const input of INPUTS) {
        for (const target of CORPUS_FORMATS) {
            if (target === input.format) {
                continue;
            }
            describe(`${input.label} -> ${target}`, () => {
                let cached: Promise<Trip> | null = null;
                const run = (): Promise<Trip> => {
                    if (cached === null) {
                        cached = original(input).then((o) => trip(o.snapshot, target, exportOptionsFor(target), {}));
                    }
                    return cached;
                };

                it("export() throws only when check() predicted an E_ note, and the target re-imports without errors", async () => {
                    const t = await run();
                    if (t.exportError !== null) {
                        const predicted = t.notes.filter((n) => n.code.startsWith("E_"));
                        expect(
                            predicted.length,
                            `export() threw ${t.exportError.code}: ${t.exportError.message}\nnotes:\n${noteText(t.notes)}`,
                        ).toBeGreaterThan(0);
                        return;
                    }
                    expect(
                        t.importError,
                        `the ${target} importer refused the ${target} exporter's own output:\n${JSON.stringify(t.importError)}\nnotes:\n${noteText(t.notes)}`,
                    ).toBeNull();
                    expect(
                        t.report?.errorCount,
                        `re-import errors:\n${JSON.stringify(t.report?.issues.slice(0, 5), null, 1)}`,
                    ).toBe(0);
                });

                it("(a) every lost or changed value, column, id or arc is predicted by a loss note", async () => {
                    const t = await run();
                    if (t.exportError !== null || t.importError !== null) {
                        return;
                    }
                    const missed = unexplained(t, VALUE_KINDS);
                    expect(
                        missed.length,
                        `${missed.length} unreported difference(s) after ${input.label} -> ${target}:\n${describeDiffs(missed.slice(0, 40))}\nnotes:\n${noteText(t.notes)}`,
                    ).toBe(0);
                });

                it("(b) every loss note corresponds to an observable difference (no phantom note)", async () => {
                    const t = await run();
                    if (t.exportError !== null || t.importError !== null) {
                        return;
                    }
                    const extra = phantoms(t);
                    expect(
                        extra.length,
                        `phantom note(s) after ${input.label} -> ${target}:\n${noteText(extra)}\ndifferences:\n${describeDiffs(t.diffs.slice(0, 40))}`,
                    ).toBe(0);
                });

                it("(c) every changed dtype, item dtype, stride or role is predicted by a loss note", async () => {
                    const t = await run();
                    if (t.exportError !== null || t.importError !== null) {
                        return;
                    }
                    const missed = unexplained(t, SCHEMA_KINDS);
                    expect(
                        missed.length,
                        `${missed.length} unreported declaration change(s) after ${input.label} -> ${target}:\n${describeDiffs(missed.slice(0, 40))}\nnotes:\n${noteText(t.notes)}`,
                    ).toBe(0);
                });
            });
        }
    }
});
