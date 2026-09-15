/**
 * The JSON importer (design sections 8.2, 8.4 and 8.5): one GraphImporter that sniffs the dialect
 * of a parsed document -- NetworkX node-link (old `links` and new `edges` key forms, graph-level
 * `directed` / `multigraph` / `graph`), the d3 lineage of the same shape (`name` ids, integer
 * index endpoints), JSON Graph Format v2 (nodes keyed by id, per-edge `directed`, hyperedges),
 * Cytoscape.js elements (`data.id` / `data.source` / `data.target`, `position`, `classes`,
 * `data.parent`), graphology serialisation (`key` / `attributes`, `undirected` edges, `options`)
 * and vis.js (`from` / `to`) -- and pushes it scalar by scalar into the sink.
 *
 * JSON awaits the whole text (design section 8.4: `JSON.parse` on 100 MB is fine; a streaming
 * tokeniser is a later improvement). The parsed records are iterated in place; the importer never
 * builds an intermediate array of node or edge objects. Ids are coerced per `ids` ("keep" by
 * default: JSON values are already typed); node ids that are JSON `true` / `false` / `null` are
 * reported as `unsupported` and coerced with `String(v)` only under `ids: "string"`. Attribute
 * columns are inferred per column by the sink (design section 5.1); the structural fields of a
 * dialect (Cytoscape `position` / `classes` / `parent`, JGF `label` / `relation`, edge ids) are
 * declared up front with their roles when the file uses them. Direction goes through the
 * DirectionResolver of design section 8.4; per-element errors are aggregated into the ImportReport
 * until the error limit (design section 8.6).
 *
 * Fatal errors (ImportError at once): empty input, invalid JSON, an unrecognised top-level shape, a
 * section that is not an array / object. Recoverable errors (an issue, the element skipped): a
 * missing nodes or edges array, a node without an id, an edge without an endpoint, a bad index
 * endpoint, a declared field of the wrong type, an unknown Cytoscape parent, an id the coercion rule
 * rejects.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphMetaPatch,
    type GraphSink,
    INVALID_INDEX,
    type NodeId,
} from "@graphty/graph-format";

import { uniqueColumnName } from "../../common/attributes.js";
import {
    DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_NODE_CODE,
    EMPTY_INPUT_CODE,
    HYPEREDGE_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MULTIPLE_GRAPHS_CODE,
    OPTION_IGNORED_CODE,
    SYNTAX_CODE,
    UNKNOWN_PARENT_CODE,
} from "../../common/codes.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { ID_MERGED_CODE, IdCoercer } from "../../common/ids.js";
import { readText, throwIfAborted } from "../../common/input.js";
import {
    type ImportFormatDefaults,
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
    SINK_OPTION_CODE,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { weightFromValue } from "../../common/weights.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    CLASSES_COLUMN,
    CYTOSCAPE_ELEMENT_KEYS,
    CYTOSCAPE_STRUCTURAL_KEYS,
    DIALECT_DEFAULT_DIRECTED,
    hasKey,
    isJsonDialect,
    isJsonObject,
    JSON_DIALECTS,
    type JsonDialect,
    type JsonShapeMeta,
    META_KEY,
    NODE_LINK_SOURCE_KEYS,
    NODE_LINK_TARGET_KEYS,
    PARENT_COLUMN,
    POSITION_COLUMN,
    sniffJsonDialect,
    SUFFIX,
} from "./dialect.js";

/** The format-specific options of the JSON importer. */
export interface JsonImportOptions {
    /** The dialect to read; "auto" (default) sniffs the parsed document. */
    dialect?: JsonDialect | "auto" | undefined;
    /** node-link / d3 / vis: the node key holding the id; auto: "id" when any node has it, else "name". */
    nodeIdKey?: string | undefined;
    /** node-link / d3: the top-level key holding the edges; auto: "edges" when present, else "links". */
    edgesKey?: string | undefined;
    /** node-link / d3 / vis: the edge key holding the source; auto: "source", "src" or "from" (vis: "from"). */
    sourceKey?: string | undefined;
    /** node-link / d3 / vis: the edge key holding the target; auto: "target", "dst" or "to" (vis: "to"). */
    targetKey?: string | undefined;
    /**
     * node-link / d3: whether edge endpoints are node array positions; "auto" (default) says yes when
     * every endpoint is an integer below the node count and no node id is a number.
     */
    indexLinks?: boolean | "auto" | undefined;
    /** jgf: which graph of a `graphs` array to read; 0 by default. */
    graphIndex?: number | undefined;
}

/**
 * The issue codes the JSON importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts) and the JSON-specific ones. A key is the code without
 * its severity and format prefixes.
 */
export const JSON_ISSUE = Object.freeze({
    /** The text is empty or whitespace (fatal). */
    EMPTY_INPUT: EMPTY_INPUT_CODE,
    /** JSON.parse refused the text (fatal). */
    SYNTAX: SYNTAX_CODE,
    /** No dialect matches the document's top-level shape. */
    DIALECT: "E_JSON_DIALECT",
    /** A section (nodes, edges, elements, graph) has the wrong JSON type. */
    SHAPE: "E_JSON_SHAPE",
    /** A node-link document lacks its nodes or its edges array, or a Cytoscape document its elements. */
    MISSING_SECTION: "E_MISSING_SECTION",
    /** A node record or an element is not an object. */
    BAD_ELEMENT: "E_BAD_ELEMENT",
    /** A node record has no id. */
    MISSING_ID: MISSING_ID_CODE,
    /** A node id is a JSON boolean or null (legal in NetworkX, not a NodeId); coerced only under ids "string". */
    UNSUPPORTED_ID: "E_UNSUPPORTED_ID",
    /** An edge record has no source or no target. */
    MISSING_ENDPOINT: MISSING_ENDPOINT_CODE,
    /** An index endpoint is not an integer below the node count, or names a skipped node. */
    BAD_INDEX: "E_BAD_INDEX",
    /** A declared field has the wrong JSON type (JGF label / relation / metadata, Cytoscape position / classes). */
    BAD_VALUE: "E_BAD_VALUE",
    /** A graph-level flag (`directed`, `multigraph`, graphology `options`) has the wrong type; the default is used. */
    BAD_FLAG: "W_BAD_FLAG",
    /** A Cytoscape `data.parent` names an unknown node. */
    UNKNOWN_PARENT: UNKNOWN_PARENT_CODE,
    /** A node id repeated by a later record; the records are merged (the later attributes win). */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** An edge id (Cytoscape data.id, graphology key, vis id) repeated by a later edge; the edge is skipped. */
    DUPLICATE_EDGE_ID: DUPLICATE_EDGE_ID_CODE,
    /** Two distinct id texts merged into one number under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** Edge ids of mixed JSON types were stored as text. */
    EDGE_ID_STRINGIFIED: "W_EDGE_ID_STRINGIFIED",
    /** A JGF `graphs` array holds more than one graph; only `graphIndex` is read. */
    MULTIPLE_GRAPHS: MULTIPLE_GRAPHS_CODE,
    /** JGF hyperedges under the "error" policy. */
    HYPEREDGE: HYPEREDGE_CODE,
    /** JGF hyperedges skipped under the default "skip" policy. */
    HYPEREDGES_SKIPPED: "W_HYPEREDGES_SKIPPED",
    /** A JGF hyperedge with neither a nodes array nor source / target arrays. */
    HYPEREDGE_SHAPE: "E_HYPEREDGE_SHAPE",
    /** The nodes have no id key at all; array positions became the ids. */
    POSITIONAL_NODES: "W_POSITIONAL_NODES",
    /** A builder-policy option (addMissingNodes, duplicateEdges, selfLoops, weightDtype) differs from the sink's (the shared W_SINK_OPTION). */
    SINK_OPTION: SINK_OPTION_CODE,
    /** A common option the dialect has no use for (nodeIdFrom outside node-link, long, restoreMangledIds). */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
});

/** The common options the JSON importer reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
    "nodeIdFrom",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "defaultDirected",
    "weightFrom",
    "weightDtype",
    "hyperedges",
    "errorLimit",
    "signal",
    "onProgress",
]);

const FORMAT_DEFAULTS: ImportFormatDefaults = { ids: "keep", defaultDirected: false, weightFrom: "weight" };

/** Bytes of the head sniff() inspects. */
const SNIFF_BYTES = 4096;

/** Elements pushed between two checks of the cancellation signal (the whole document is one chunk). */
const ABORT_CHECK_INTERVAL = 64;

/** The top-level keys whose presence in the head marks a graph document rather than arbitrary JSON. */
const SNIFF_KEYS: readonly string[] = ['"nodes"', '"links"', '"edges"', '"elements"', '"graph"', '"graphs"'];

const BOM = String.fromCharCode(0xfeff);

/** The JGF edge keys that are not metadata. */
const JGF_EDGE_KEYS: ReadonlySet<string> = new Set([
    "id",
    "source",
    "target",
    "relation",
    "directed",
    "label",
    "metadata",
]);

/** The JGF hyperedge keys that are not metadata. */
const JGF_HYPEREDGE_KEYS: ReadonlySet<string> = new Set([...JGF_EDGE_KEYS, "nodes"]);

/** The JGF node keys that are not metadata. */
const JGF_NODE_KEYS: ReadonlySet<string> = new Set(["label", "metadata"]);

/** The graphology node keys that are not attributes. */
const GRAPHOLOGY_NODE_KEYS: ReadonlySet<string> = new Set(["key", "attributes"]);

/** The graphology edge keys that are not attributes. */
const GRAPHOLOGY_EDGE_KEYS: ReadonlySet<string> = new Set(["key", "source", "target", "attributes", "undirected"]);

/** The vis.js endpoint keys. */
const VIS_SOURCE_KEYS: readonly string[] = Object.freeze(["from"]);
const VIS_TARGET_KEYS: readonly string[] = Object.freeze(["to"]);

type JsonRecord = Record<string, unknown>;

/** An edge id column declared from a scan of the file's edge ids. */
interface EdgeIdColumn {
    readonly handle: ColumnHandle;
    /** Whether numeric ids are stored as text (the file mixes numbers and strings). */
    readonly stringify: boolean;
    /** The id texts seen so far when the dialect requires unique ids, else null. */
    readonly seen: Set<string> | null;
}

/** The resolved format-specific options. */
interface ResolvedJsonOptions {
    readonly dialect: JsonDialect | "auto";
    readonly nodeIdKey: string | null;
    readonly edgesKey: string | null;
    readonly sourceKey: string | null;
    readonly targetKey: string | null;
    readonly indexLinks: boolean | "auto";
    readonly graphIndex: number;
}

/**
 * Check the format-specific options.
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for a bad value
 */
function resolveJsonOptions(options: (JsonImportOptions & CommonImportOptions) | undefined): ResolvedJsonOptions {
    const o = options ?? {};
    const dialect = o.dialect ?? "auto";
    if (dialect !== "auto" && !isJsonDialect(dialect)) {
        throw unsupportedOption("dialect", dialect, [...JSON_DIALECTS, "auto"]);
    }
    const indexLinks = o.indexLinks ?? "auto";
    if (indexLinks !== "auto" && typeof indexLinks !== "boolean") {
        throw unsupportedOption("indexLinks", indexLinks, ["true", "false", "auto"]);
    }
    const graphIndex = o.graphIndex ?? 0;
    if (!Number.isInteger(graphIndex) || graphIndex < 0) {
        throw unsupportedOption("graphIndex", graphIndex, ["a non-negative integer"]);
    }
    return {
        dialect,
        nodeIdKey: keyOption("nodeIdKey", o.nodeIdKey),
        edgesKey: keyOption("edgesKey", o.edgesKey),
        sourceKey: keyOption("sourceKey", o.sourceKey),
        targetKey: keyOption("targetKey", o.targetKey),
        indexLinks,
        graphIndex,
    };
}

/**
 * Check a key-valued option.
 * @param name - the option name
 * @param value - the caller's value
 * @returns the key, or null when absent
 */
function keyOption(name: string, value: unknown): string | null {
    if (value === undefined) {
        return null;
    }
    if (typeof value !== "string" || value.length === 0) {
        throw unsupportedOption(name, value, ["a non-empty key"]);
    }
    return value;
}

/**
 * The E_UNSUPPORTED error of a bad format-specific option (the core's convention, as in the common
 * option module).
 * @param name - the option name
 * @param found - the value
 * @param supported - what is accepted
 * @returns the error
 */
function unsupportedOption(name: string, found: unknown, supported: readonly string[]): GraphFormatError {
    return new GraphFormatError(
        "E_UNSUPPORTED",
        `option ${name}: ${describe(found)} is not one of ${supported.join(", ")}`,
        {
            option: name,
            found: typeof found === "string" ? found : typeof found,
            supported: [...supported],
        },
    );
}

/**
 * A short description of a value for messages.
 * @param value - the value
 * @returns JSON for primitives, "array" or the type name otherwise
 */
function describe(value: unknown): string {
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value)) {
        return "array";
    }
    switch (typeof value) {
        case "string":
        case "number":
        case "boolean":
            return JSON.stringify(value);
        default:
            return typeof value;
    }
}

// ============================================================ attribute writer

/**
 * Writes inferred attribute cells for one table, caching the handle of every column after its
 * first write so the hot loop never looks a column up by name twice. Names that collide with a
 * structural column declared up front are suffixed deterministically (design section 5.6).
 */
class AttributeWriter {
    private readonly sink: GraphSink;

    private readonly domain: "node" | "edge";

    private readonly handles = new Map<string, ColumnHandle>();

    private readonly reservedNames = new Set<string>();

    /**
     * Create a writer.
     * @param sink - the sink
     * @param domain - node or edge
     */
    constructor(sink: GraphSink, domain: "node" | "edge") {
        this.sink = sink;
        this.domain = domain;
    }

    /**
     * Declare a structural column up front; attribute keys with its name are suffixed from now on.
     * @param decl - the declaration
     * @returns the handle
     */
    declare(decl: ColumnDecl): ColumnHandle {
        this.reservedNames.add(decl.name);
        const handle = this.domain === "node" ? this.sink.declareNodeColumn(decl) : this.sink.declareEdgeColumn(decl);
        this.handles.set(decl.name, handle);
        return handle;
    }

    /**
     * Declare a structural column only when the file uses it.
     * @param decl - the declaration
     * @param present - whether any element carries the field
     * @returns the handle, or INVALID_INDEX when not declared
     */
    declareIf(decl: ColumnDecl, present: boolean): ColumnHandle {
        return present ? this.declare(decl) : (INVALID_INDEX as ColumnHandle);
    }

    /**
     * Whether a name is taken in the sink's table (for the deterministic rename rule).
     * @param name - the column name
     * @returns true when a column of that name exists
     */
    taken(name: string): boolean {
        return this.lookup(name) !== INVALID_INDEX;
    }

    /**
     * Write one attribute cell by its source key; null and undefined leave the row unset.
     * @param row - the node or edge index
     * @param key - the source key
     * @param value - the JSON value
     * @param suffix - the suffix applied when the key collides with a structural column
     */
    write(row: number, key: string, value: unknown, suffix: string): void {
        if (value === undefined || value === null) {
            return;
        }
        const name = this.reservedNames.has(key) ? `${key}${suffix}` : key;
        const cached = this.handles.get(name);
        if (cached !== undefined) {
            this.set(cached, row, value);
            return;
        }
        this.set(name, row, value);
        const handle = this.lookup(name);
        if (handle !== INVALID_INDEX) {
            this.handles.set(name, handle);
        }
    }

    /**
     * Write through a handle or a name.
     * @param column - the handle or name
     * @param row - the row
     * @param value - the value
     */
    set(column: ColumnHandle | string, row: number, value: unknown): void {
        if (this.domain === "node") {
            this.sink.setNodeValue(column, row, value);
        } else {
            this.sink.setEdgeValue(column, row, value);
        }
    }

    /**
     * Look a column up by name.
     * @param name - the column name
     * @returns the handle, or INVALID_INDEX
     */
    private lookup(name: string): ColumnHandle {
        return this.domain === "node" ? this.sink.nodeColumn(name) : this.sink.edgeColumn(name);
    }
}

// ============================================================ the import context

/**
 * Everything one import call shares between the dialect readers.
 */
class ImportContext {
    readonly sink: GraphSink;

    readonly report: ImportReportBuilder;

    readonly options: ResolvedImportOptions;

    readonly json: ResolvedJsonOptions;

    readonly ids: IdCoercer;

    readonly direction: DirectionResolver;

    readonly nodes: AttributeWriter;

    readonly edges: AttributeWriter;

    /** Whether the caller passed `defaultDirected` explicitly (then it beats the dialect's convention). */
    readonly explicitDefaultDirected: boolean;

    /** Nodes and edges pushed since the signal was last checked. */
    private elementsSinceCheck = 0;

    /**
     * Create the context.
     * @param sink - the sink
     * @param report - the report
     * @param options - the resolved common options
     * @param json - the resolved format options
     * @param explicitDefaultDirected - whether the caller passed defaultDirected
     */
    constructor(
        sink: GraphSink,
        report: ImportReportBuilder,
        options: ResolvedImportOptions,
        json: ResolvedJsonOptions,
        explicitDefaultDirected: boolean,
    ) {
        this.sink = sink;
        this.report = report;
        this.options = options;
        this.json = json;
        this.ids = new IdCoercer(options.ids);
        this.direction = new DirectionResolver(sink, report, options.onMixedDirection);
        this.nodes = new AttributeWriter(sink, "node");
        this.edges = new AttributeWriter(sink, "edge");
        this.explicitDefaultDirected = explicitDefaultDirected;
    }

    /**
     * Report a `nodeIdFrom` other than "id" for a dialect whose ids are unambiguous.
     * @param dialect - the dialect
     * @param idField - where the dialect's ids come from, for the message
     */
    reportNodeIdFrom(dialect: JsonDialect, idField: string): void {
        if (this.options.nodeIdFrom !== "id") {
            this.report.warning(
                "unsupported",
                JSON_ISSUE.OPTION_IGNORED,
                `nodeIdFrom "${this.options.nodeIdFrom}" does not apply to ${dialect}; ids are read from ${idField}`,
                { element: "nodeIdFrom" },
            );
        }
    }

    /**
     * The direction assumed when the file declares none: the caller's `defaultDirected` when given,
     * else the dialect's convention.
     * @param dialect - the dialect
     * @returns the direction
     */
    defaultDirected(dialect: JsonDialect): boolean {
        return this.explicitDefaultDirected ? this.options.defaultDirected : DIALECT_DEFAULT_DIRECTED[dialect];
    }

    /** The direction the file declares (or the default), set by setHeader(). */
    private fileDirected = false;

    /**
     * Rule 1 of design section 8.4: set the sink's direction from the file's header (or the
     * dialect's default) before the first edge, remembering the file's direction for uniformKind().
     * @param directed - the file's direction
     */
    setHeader(directed: boolean): void {
        this.fileDirected = directed;
        this.direction.setHeader(directed);
    }

    /**
     * The edge kind every edge of a dialect without per-edge direction has: the file's direction
     * (the resolver expands it when the sink's direction differs).
     * @returns the kind
     */
    uniformKind(): EdgeKind {
        return this.fileDirected ? "directed" : "undirected";
    }

    /**
     * Coerce a node id value per the `ids` option, reporting what cannot be an id. A JSON boolean or
     * null is `unsupported` unless `ids` is "string" (design section 8.5); anything the rule rejects
     * is recorded through the per-element catch.
     * @param raw - the JSON value; undefined when the record has no id key
     * @param element - the element name for the issue
     * @returns the id, or null when the value was reported and the element must be skipped
     */
    coerceId(raw: unknown, element: string): NodeId | null {
        if (raw === undefined) {
            this.report.error("missing-value", JSON_ISSUE.MISSING_ID, `${element} has no id`, { element });
            return null;
        }
        if ((typeof raw === "boolean" || raw === null) && this.options.ids !== "string") {
            const kind = raw === null ? "null" : "boolean";
            this.report.error(
                "unsupported",
                JSON_ISSUE.UNSUPPORTED_ID,
                `${element}: a JSON ${kind} is not a node id (pass ids: "string" to coerce it)`,
                { element },
            );
            return null;
        }
        let id: NodeId;
        try {
            id = this.ids.value(this.options.ids === "string" && typeof raw === "number" ? String(raw) : raw);
        } catch (err) {
            this.report.recordError(err, { element });
            return null;
        }
        const merge = this.ids.lastMerge;
        if (merge !== null) {
            this.report.warning(
                "coercion",
                JSON_ISSUE.ID_MERGED,
                `id text ${JSON.stringify(merge.text)} merged with ${JSON.stringify(merge.previousText)} as ${merge.id}`,
                { element },
            );
        }
        return id;
    }

    /**
     * Coerce an id that must be valid for the caller to proceed (hyperedge members): the rejections
     * of coerceId() are thrown instead of recorded.
     * @param raw - the JSON value
     * @param element - the element name
     * @returns the id
     */
    requireId(raw: unknown, element: string): NodeId {
        if ((typeof raw === "boolean" || raw === null) && this.options.ids !== "string") {
            const kind = raw === null ? "null" : "boolean";
            throw new GraphFormatError("E_INVALID_ID", `${element}: a JSON ${kind} is not a node id`, {
                reason: "unsupported id",
            });
        }
        return this.ids.value(this.options.ids === "string" && typeof raw === "number" ? String(raw) : raw);
    }

    /**
     * Add a node, counting it or recording the failure.
     * @param id - the node id
     * @param element - the element name
     * @returns the node index, or -1 when the sink refused the node
     */
    pushNode(id: NodeId, element: string): number {
        this.checkAbort();
        const existing = this.sink.indexOf(id);
        if (existing !== INVALID_INDEX) {
            this.report.warning(
                "merged",
                JSON_ISSUE.DUPLICATE_NODE,
                `${element}: node ${JSON.stringify(id)} already exists; its attributes are merged (the later values win)`,
                { element },
            );
            return existing;
        }
        let index: number;
        try {
            index = this.sink.addNode(id);
        } catch (err) {
            this.skip(err, "node", element);
            return -1;
        }
        this.report.counts.nodes++;
        return index;
    }

    /**
     * Push one edge through the direction resolver, counting every logical edge the sink gained
     * (both halves of an expanded edge, and the mirrors of an in-place expansion).
     * @param source - the source id
     * @param target - the target id
     * @param kind - the edge's direction in the file
     * @param weight - the weight, or undefined
     * @param element - the element name for issues
     * @returns the primary edge index
     */
    pushEdge(source: NodeId, target: NodeId, kind: EdgeKind, weight: number | undefined, element: string): number {
        this.checkAbort();
        const before = this.sink.edgeCount;
        // endpoints the sink creates (addMissingNodes) count as nodes too
        let created = this.sink.indexOf(source) === INVALID_INDEX ? 1 : 0;
        if (source !== target && this.sink.indexOf(target) === INVALID_INDEX) {
            created++;
        }
        const edge = this.direction.addEdge(source, target, kind, weight, { element });
        this.report.counts.edges += this.sink.edgeCount - before;
        this.report.counts.nodes += created;
        return edge;
    }

    /**
     * Check the cancellation signal every ABORT_CHECK_INTERVAL pushed elements, so an abort raised
     * while the whole in-memory document is being walked rejects promptly.
     */
    checkAbort(): void {
        if (++this.elementsSinceCheck >= ABORT_CHECK_INTERVAL) {
            this.elementsSinceCheck = 0;
            throwIfAborted(this.options.signal);
        }
    }

    /**
     * Record a per-element failure and count the skipped element.
     * @param err - the thrown value
     * @param domain - which counter to bump
     * @param element - the element name
     */
    skip(err: unknown, domain: "node" | "edge", element: string): void {
        this.report.recordError(err, { element });
        if (domain === "node") {
            this.report.counts.skippedNodes++;
        } else {
            this.report.counts.skippedEdges++;
        }
    }

    /**
     * Report an element that is not an object and count it as skipped.
     * @param domain - node or edge
     * @param element - the element name
     * @param what - what was expected
     */
    badElement(domain: "node" | "edge", element: string, what = "an object"): void {
        this.report.error("validation-error", JSON_ISSUE.BAD_ELEMENT, `${element} is not ${what}`, { element });
        this.countSkipped(domain);
    }

    /**
     * Count a skipped element whose issue was already recorded.
     * @param domain - node or edge
     */
    countSkipped(domain: "node" | "edge"): void {
        if (domain === "node") {
            this.report.counts.skippedNodes++;
        } else {
            this.report.counts.skippedEdges++;
        }
    }

    /**
     * Report an edge record without an endpoint and count it as skipped.
     * @param element - the element name
     * @param field - the missing field
     */
    missingEndpoint(element: string, field: string): void {
        this.report.error("missing-value", JSON_ISSUE.MISSING_ENDPOINT, `${element} has no ${field}`, { element });
        this.report.counts.skippedEdges++;
    }

    /**
     * Record the shape metadata, the source format and further metadata fields in one setMeta()
     * call (the sink replaces `extra` as a whole).
     * @param shape - the shape record
     * @param patch - further metadata fields
     */
    setMeta(shape: JsonShapeMeta, patch: GraphMetaPatch = {}): void {
        const extra: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(shape)) {
            if (value !== undefined) {
                extra[key] = value;
            }
        }
        this.sink.setMeta({ sourceFormat: "json", ...patch, extra: { [META_KEY]: extra } });
    }

    /**
     * Record which source field the weight came from, so the exporter writes it back under the same
     * key (design section 3.7, `meta.weightOrigin`).
     * @returns the metadata patch, empty for an unweighted import
     */
    weightOriginPatch(): GraphMetaPatch {
        const { weightFrom } = this.options;
        if (weightFrom === null) {
            return {};
        }
        return { weightOrigin: { format: "json", id: weightFrom, title: null, type: null, namespace: null } };
    }

    /**
     * Write the graph-level attributes of a dict (NetworkX `graph`, JGF `metadata`, graphology
     * `attributes`, Cytoscape `data`), one column per key.
     * @param dict - the dict, or anything else (then reported)
     * @param what - the dict's name for the issue
     */
    writeGraphDict(dict: unknown, what: string): void {
        if (dict === undefined || dict === null) {
            return;
        }
        if (!isJsonObject(dict)) {
            this.report.warning("validation-error", JSON_ISSUE.BAD_FLAG, `${what} is not an object; ignored`, {
                element: what,
            });
            return;
        }
        for (const key of Object.keys(dict)) {
            const value = dict[key];
            if (value === undefined || value === null) {
                continue;
            }
            try {
                this.sink.setGraphValue(key, value);
            } catch (err) {
                this.report.recordError(err, { element: `${what}.${key}` });
            }
        }
    }

    /**
     * Read the weight field of an edge record.
     * @param record - the record holding the attributes
     * @returns the weight, or undefined when absent or null; E_INVALID_WEIGHT otherwise
     */
    weightOf(record: JsonRecord): number | undefined {
        const { weightFrom } = this.options;
        if (weightFrom === null || !hasKey(record, weightFrom)) {
            return undefined;
        }
        return weightFromValue(record[weightFrom]);
    }

    /**
     * Write the attributes of a nested dict plus the element-level keys the dialect does not
     * define (kept with the `#element` suffix).
     * @param writer - the table writer
     * @param row - the row
     * @param record - the element record
     * @param dict - the nested attribute dict
     * @param structural - the element keys that are not attributes
     * @param weightFrom - the weight key to skip in the dict, or null
     */
    writeNested(
        writer: AttributeWriter,
        row: number,
        record: JsonRecord,
        dict: JsonRecord,
        structural: ReadonlySet<string>,
        weightFrom: string | null,
    ): void {
        for (const key of Object.keys(dict)) {
            if (key !== weightFrom) {
                writer.write(row, key, dict[key], SUFFIX.data);
            }
        }
        for (const key of Object.keys(record)) {
            if (!structural.has(key)) {
                writer.write(row, `${key}${SUFFIX.element}`, record[key], SUFFIX.data);
            }
        }
    }

    /**
     * Write a spec-typed string field (JGF label / relation), reporting a value of another type.
     * @param writer - the table writer
     * @param column - the declared column, or INVALID_INDEX when the file has no such field
     * @param row - the row
     * @param value - the value
     * @param field - the field name
     * @param element - the element name
     */
    writeStringField(
        writer: AttributeWriter,
        column: ColumnHandle,
        row: number,
        value: unknown,
        field: string,
        element: string,
    ): void {
        if (column === INVALID_INDEX || value === undefined || value === null) {
            return;
        }
        if (typeof value !== "string") {
            this.report.error(
                "validation-error",
                JSON_ISSUE.BAD_VALUE,
                `${element}: ${field} must be a string, found ${describe(value)}`,
                { element },
            );
            return;
        }
        writer.set(column, row, value);
    }

    /**
     * Declare an edge id column with role "id" from a scan of the file's edge ids: f64 when every
     * id is a number, string otherwise (numbers are then stored as their text and reported once).
     * @param edges - the edge records
     * @param read - how to read an edge's raw id
     * @param name - the column name
     * @param unique - whether uniqueness is enforced at freeze
     * @returns the column, or null when no edge has an id
     */
    declareEdgeIds(
        edges: readonly unknown[],
        read: (edge: JsonRecord) => unknown,
        name: string,
        unique: boolean,
    ): EdgeIdColumn | null {
        let numbers = 0;
        let strings = 0;
        for (const edge of edges) {
            if (!isJsonObject(edge)) {
                continue;
            }
            const raw = read(edge);
            if (typeof raw === "number") {
                numbers++;
            } else if (typeof raw === "string") {
                strings++;
            }
        }
        if (numbers + strings === 0) {
            return null;
        }
        const dtype = strings === 0 ? "f64" : "string";
        const columnName = uniqueColumnName(name, "id", (candidate) => this.edges.taken(candidate));
        const handle = this.edges.declare({ name: columnName, dtype, role: "id", nullable: true, unique });
        const stringify = strings > 0 && numbers > 0;
        const seen = unique ? new Set<string>() : null;
        if (stringify) {
            this.report.warning(
                "coercion",
                JSON_ISSUE.EDGE_ID_STRINGIFIED,
                `edge ids mix numbers and strings; ${numbers} numeric id(s) stored as text`,
                { element: columnName },
            );
        }
        return { handle, stringify, seen };
    }

    /**
     * The value an edge id column stores for a raw id, checked BEFORE the edge is pushed so a bad
     * id skips the edge without touching the sink (design section 11.1).
     * @param column - the column, or null when the file has no edge ids
     * @param raw - the raw id value
     * @returns the value to store, or null when there is nothing to store
     */
    edgeIdValue(column: EdgeIdColumn | null, raw: unknown): string | number | null {
        if (column === null || raw === undefined || raw === null) {
            return null;
        }
        let value: string | number;
        if (typeof raw === "number") {
            value = column.stringify ? String(raw) : raw;
        } else if (typeof raw === "string") {
            value = raw;
        } else {
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `an edge id must be a string or a number, found ${describe(raw)}`,
                { found: typeof raw },
            );
        }
        if (column.seen !== null) {
            const text = String(value);
            if (column.seen.has(text)) {
                throw new GraphFormatError(
                    JSON_ISSUE.DUPLICATE_EDGE_ID,
                    `edge id ${JSON.stringify(value)} is declared more than once; the edge is skipped`,
                    { id: value },
                );
            }
            column.seen.add(text);
        }
        return value;
    }

    /**
     * Write an edge id value from edgeIdValue().
     * @param column - the column, or null
     * @param edge - the edge index
     * @param value - the value, or null for none
     */
    setEdgeId(column: EdgeIdColumn | null, edge: number, value: string | number | null): void {
        if (column !== null && value !== null) {
            this.edges.set(column.handle, edge, value);
        }
    }
}

// ============================================================ document level

/**
 * Parse the whole text; syntax errors and empty input abort the import.
 * @param text - the decoded text
 * @param report - the report
 * @returns the parsed value
 */
function parseDocument(text: string, report: ImportReportBuilder): unknown {
    if (text.trim().length === 0) {
        report.fail(JSON_ISSUE.EMPTY_INPUT, "the input is empty");
    }
    try {
        return JSON.parse(text) as unknown;
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return report.fail(JSON_ISSUE.SYNTAX, `invalid JSON: ${message}`);
    }
}

/**
 * The dialect to read: the forced one, else the shape rule of sniffJsonDialect(); a document
 * that matches no dialect is a fatal E_JSON_DIALECT.
 * @param root - the parsed document
 * @param forced - the caller's dialect option
 * @param report - the report the failure is recorded in
 * @returns the dialect
 */
function detectDialect(root: unknown, forced: JsonDialect | "auto", report: ImportReportBuilder): JsonDialect {
    if (forced !== "auto") {
        return forced;
    }
    const dialect = sniffJsonDialect(root);
    if (dialect !== null) {
        return dialect;
    }
    if (Array.isArray(root)) {
        return report.fail(
            JSON_ISSUE.DIALECT,
            "a top-level array is only read as Cytoscape elements (objects with a data record)",
        );
    }
    if (!isJsonObject(root)) {
        return report.fail(JSON_ISSUE.DIALECT, `the document is a JSON ${describe(root)}, not a graph object`);
    }
    return report.fail(
        JSON_ISSUE.DIALECT,
        "no known dialect: expected nodes / links / edges (node-link), elements (Cytoscape) or graph (JGF)",
    );
}

/**
 * A section that must be an array: fail when it is something else.
 * @param value - the section
 * @param what - its name
 * @param report - the report
 * @returns the array, or null when absent
 */
function arraySection(value: unknown, what: string, report: ImportReportBuilder): readonly unknown[] | null {
    if (value === undefined || value === null) {
        return null;
    }
    if (!Array.isArray(value)) {
        return report.fail(JSON_ISSUE.SHAPE, `${what} must be an array, found ${describe(value)}`, { element: what });
    }
    return value;
}

/**
 * A boolean flag with a default and a warning when it has another type.
 * @param value - the flag value
 * @param what - its name
 * @param fallback - the default
 * @param report - the report
 * @returns the flag
 */
function flagOf(value: unknown, what: string, fallback: boolean, report: ImportReportBuilder): boolean {
    if (value === undefined || value === null) {
        return fallback;
    }
    if (typeof value === "boolean") {
        return value;
    }
    report.warning(
        "validation-error",
        JSON_ISSUE.BAD_FLAG,
        `${what} is ${describe(value)}, not a boolean; ${fallback} assumed`,
        { element: what },
    );
    return fallback;
}

/**
 * The endpoint key and value of an edge record: the explicit key when given, else the first of the
 * default keys the record has.
 * @param record - the edge record
 * @param explicit - the caller's key, or null
 * @param defaults - the default keys
 * @returns the key used (null when none) and its value
 */
function endpointOf(
    record: JsonRecord,
    explicit: string | null,
    defaults: readonly string[],
): { readonly key: string | null; readonly value: unknown } {
    if (explicit !== null) {
        return { key: hasKey(record, explicit) ? explicit : null, value: record[explicit] };
    }
    for (const key of defaults) {
        if (hasKey(record, key)) {
            return { key, value: record[key] };
        }
    }
    return { key: null, value: undefined };
}

/**
 * Whether any object of an array has a key with a non-null value.
 * @param items - the array
 * @param key - the key
 * @returns true when some element carries the field
 */
function anyHas(items: readonly unknown[], key: string): boolean {
    return items.some((item) => isJsonObject(item) && item[key] !== undefined && item[key] !== null);
}

/**
 * Whether a value is a non-negative integer below a bound.
 * @param value - the value
 * @param bound - the exclusive bound
 * @returns true for an index
 */
function isIndexBelow(value: unknown, bound: number): boolean {
    return typeof value === "number" && Number.isInteger(value) && value >= 0 && value < bound;
}

// ============================================================ node-link / d3

/**
 * Read a node-link or d3 document.
 * @param ctx - the context
 * @param root - the document
 * @param dialect - "node-link" or "d3"
 */
function importNodeLink(ctx: ImportContext, root: JsonRecord, dialect: JsonDialect): void {
    const { report, json } = ctx;
    let { edgesKey } = json;
    if (edgesKey === null) {
        edgesKey = hasKey(root, "edges") || !hasKey(root, "links") ? "edges" : "links";
    }
    const nodes = arraySection(root.nodes, "nodes", report);
    const edges = arraySection(root[edgesKey], edgesKey, report);
    if (nodes === null) {
        report.error(
            "missing-value",
            JSON_ISSUE.MISSING_SECTION,
            "the document has no nodes array; nodes come from the edges",
            { element: "nodes" },
        );
    }
    if (edges === null) {
        report.error(
            "missing-value",
            JSON_ISSUE.MISSING_SECTION,
            `the document has no ${edgesKey} array; the graph has no edges`,
            { element: edgesKey },
        );
    }
    const directed = flagOf(root.directed, "directed", ctx.defaultDirected(dialect), report);
    const multigraph = hasKey(root, "multigraph") ? flagOf(root.multigraph, "multigraph", false, report) : null;
    ctx.setHeader(directed);
    ctx.writeGraphDict(root.graph, "graph");

    const nodeList = nodes ?? [];
    const edgeList = edges ?? [];
    ctx.sink.reserve(nodeList.length, edgeList.length);

    // the id key: the caller's, else "id" when any node has it, else "name" (d3); nodes without any
    // id key are positional (nodeIdFrom "index", or a d3 v3 file whose nodes carry no id at all)
    let { nodeIdKey } = json;
    let positional = ctx.options.nodeIdFrom === "index";
    if (!positional && nodeIdKey === null) {
        const candidates = ctx.options.nodeIdFrom === "label" ? ["label", "name", "id"] : ["id", "name"];
        nodeIdKey = candidates.find((key) => anyHas(nodeList, key)) ?? null;
        if (ctx.options.nodeIdFrom === "label" && nodeIdKey === "id") {
            report.warning(
                "unsupported",
                JSON_ISSUE.OPTION_IGNORED,
                'nodeIdFrom "label": no node has a label or name key; ids are read from "id"',
                { element: "nodeIdFrom" },
            );
        }
        if (nodeIdKey === null) {
            if (nodeList.length > 0) {
                positional = true;
                report.warning(
                    "missing-value",
                    JSON_ISSUE.POSITIONAL_NODES,
                    "no node has an id or name key; array positions are the node ids",
                    { element: "nodes" },
                );
            } else {
                nodeIdKey = "id";
            }
        }
    }
    if (positional) {
        nodeIdKey = null;
    }
    let indexLinks: boolean;
    if (positional) {
        indexLinks = true;
    } else if (json.indexLinks === "auto") {
        indexLinks = nodeIdKey !== null && looksIndexLinked(nodeList, edgeList, nodeIdKey, json);
    } else {
        ({ indexLinks } = json);
    }
    const positionIds: (NodeId | null)[] | null = indexLinks ? [] : null;

    for (let i = 0; i < nodeList.length; i++) {
        const element = `nodes[${i}]`;
        const record = nodeList[i];
        let pushed: NodeId | null = null;
        if (!isJsonObject(record)) {
            ctx.badElement("node", element);
        } else {
            let id: NodeId | null;
            if (nodeIdKey === null) {
                id = i;
            } else {
                id = ctx.coerceId(hasKey(record, nodeIdKey) ? record[nodeIdKey] : undefined, element);
            }
            if (id === null) {
                ctx.countSkipped("node");
            } else {
                const index = ctx.pushNode(id, element);
                if (index >= 0) {
                    pushed = id;
                    writeFlat(ctx, ctx.nodes, index, record, id, (key) => key !== nodeIdKey);
                }
            }
        }
        positionIds?.push(pushed);
    }
    throwIfAborted(ctx.options.signal);
    const endpointKeys = importNodeLinkEdges(ctx, edgeList, edgesKey, positionIds);
    ctx.setMeta(
        {
            dialect,
            edgesKey,
            nodeIdKey,
            indexLinks,
            sourceKey: endpointKeys.source ?? undefined,
            targetKey: endpointKeys.target ?? undefined,
        },
        { declaredMultigraph: multigraph, ...ctx.weightOriginPatch() },
    );
}

/**
 * Read the node-link / d3 edge records: endpoints by id or array position, the weight, the other
 * keys as attributes.
 * @param ctx - the context
 * @param edgeList - the edge records
 * @param edgesKey - the top-level key they came from, for issues
 * @param positionIds - the ids by array position under index links, or null
 * @returns the source and target keys the first well-formed edge used (null when none did)
 */
function importNodeLinkEdges(
    ctx: ImportContext,
    edgeList: readonly unknown[],
    edgesKey: string,
    positionIds: readonly (NodeId | null)[] | null,
): { source: string | null; target: string | null } {
    const { json } = ctx;
    const kind = ctx.uniformKind();
    let sourceKey: string | null = null;
    let targetKey: string | null = null;
    for (let i = 0; i < edgeList.length; i++) {
        const element = `${edgesKey}[${i}]`;
        const record = edgeList[i];
        if (!isJsonObject(record)) {
            ctx.badElement("edge", element);
            continue;
        }
        const source = endpointOf(record, json.sourceKey, NODE_LINK_SOURCE_KEYS);
        const target = endpointOf(record, json.targetKey, NODE_LINK_TARGET_KEYS);
        if (source.key === null || target.key === null) {
            ctx.missingEndpoint(element, source.key === null ? "source" : "target");
            continue;
        }
        sourceKey ??= source.key;
        targetKey ??= target.key;
        try {
            const u = resolveEndpoint(ctx, source.value, positionIds, element, "source");
            const v = resolveEndpoint(ctx, target.value, positionIds, element, "target");
            if (u === null || v === null) {
                ctx.countSkipped("edge");
                continue;
            }
            const weight = ctx.weightOf(record);
            const edge = ctx.pushEdge(u, v, kind, weight, element);
            const { weightFrom } = ctx.options;
            for (const key of Object.keys(record)) {
                if (key !== source.key && key !== target.key && key !== weightFrom) {
                    ctx.edges.write(edge, key, record[key], SUFFIX.data);
                }
            }
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }
    return { source: sourceKey, target: targetKey };
}

/**
 * Write the flat attributes of a node record (every own key the filter keeps).
 * @param ctx - the context
 * @param writer - the node writer
 * @param index - the node index
 * @param record - the record
 * @param id - the node id, for issues
 * @param keep - which keys are attributes
 */
function writeFlat(
    ctx: ImportContext,
    writer: AttributeWriter,
    index: number,
    record: JsonRecord,
    id: NodeId,
    keep: (key: string) => boolean,
): void {
    try {
        for (const key of Object.keys(record)) {
            if (keep(key)) {
                writer.write(index, key, record[key], SUFFIX.data);
            }
        }
    } catch (err) {
        ctx.report.recordError(err, { element: String(id) });
    }
}

/**
 * The d3 index-link heuristic: endpoints are array positions when every endpoint is a
 * non-negative integer and no node id is a number (a numeric id would make the endpoints ids;
 * research note 07: d3 links reference nodes by array index and are never coerced to ids). An
 * index at or beyond the node count is then E_BAD_INDEX, never a new numeric node.
 * @param nodes - the node records
 * @param edges - the edge records
 * @param nodeIdKey - the node id key
 * @param json - the format options (endpoint keys)
 * @returns true when endpoints are indices
 */
function looksIndexLinked(
    nodes: readonly unknown[],
    edges: readonly unknown[],
    nodeIdKey: string,
    json: ResolvedJsonOptions,
): boolean {
    if (edges.length === 0 || nodes.length === 0) {
        return false;
    }
    for (const node of nodes) {
        if (isJsonObject(node) && typeof node[nodeIdKey] === "number") {
            return false;
        }
    }
    let seen = 0;
    for (const edge of edges) {
        if (!isJsonObject(edge)) {
            continue;
        }
        const s = endpointOf(edge, json.sourceKey, NODE_LINK_SOURCE_KEYS).value;
        const t = endpointOf(edge, json.targetKey, NODE_LINK_TARGET_KEYS).value;
        if (!isIndexBelow(s, Infinity) || !isIndexBelow(t, Infinity)) {
            return false;
        }
        seen++;
    }
    return seen > 0;
}

/**
 * Resolve one endpoint: a node id through the coercion rule, or an array position through the
 * ids pushed so far under index links.
 * @param ctx - the context
 * @param raw - the endpoint value
 * @param positionIds - the ids by array position under index links, or null
 * @param element - the edge element name
 * @param field - "source" or "target"
 * @returns the id, or null when reported
 */
function resolveEndpoint(
    ctx: ImportContext,
    raw: unknown,
    positionIds: readonly (NodeId | null)[] | null,
    element: string,
    field: string,
): NodeId | null {
    if (positionIds === null) {
        return ctx.coerceId(raw, `${element}.${field}`);
    }
    if (!isIndexBelow(raw, positionIds.length)) {
        ctx.report.error(
            "validation-error",
            JSON_ISSUE.BAD_INDEX,
            `${element}: ${field} ${describe(raw)} is not a node index below ${positionIds.length}`,
            { element },
        );
        return null;
    }
    const id = positionIds[raw as number];
    if (id === null) {
        ctx.report.error("missing-value", JSON_ISSUE.BAD_INDEX, `${element}: ${field} names a node that was skipped`, {
            element,
        });
    }
    return id;
}

// ============================================================ vis.js

/**
 * Read a vis.js document: `nodes` with `id`, `edges` with `from` / `to` and an optional `id`.
 * @param ctx - the context
 * @param root - the document
 */
function importVis(ctx: ImportContext, root: JsonRecord): void {
    const { report, json } = ctx;
    const nodes = arraySection(root.nodes, "nodes", report) ?? [];
    const edges = arraySection(root.edges, "edges", report) ?? [];
    ctx.setHeader(ctx.defaultDirected("vis"));
    ctx.sink.reserve(nodes.length, edges.length);
    const nodeIdKey = json.nodeIdKey ?? "id";
    ctx.reportNodeIdFrom("vis", `the ${JSON.stringify(nodeIdKey)} key`);
    for (let i = 0; i < nodes.length; i++) {
        const element = `nodes[${i}]`;
        const record = nodes[i];
        if (!isJsonObject(record)) {
            ctx.badElement("node", element);
            continue;
        }
        const id = ctx.coerceId(hasKey(record, nodeIdKey) ? record[nodeIdKey] : undefined, element);
        if (id === null) {
            ctx.countSkipped("node");
            continue;
        }
        const index = ctx.pushNode(id, element);
        if (index >= 0) {
            writeFlat(ctx, ctx.nodes, index, record, id, (key) => key !== nodeIdKey);
        }
    }
    throwIfAborted(ctx.options.signal);
    const ids = ctx.declareEdgeIds(edges, (edge) => edge.id, "id", true);
    const kind = ctx.uniformKind();
    let sourceKey: string | null = null;
    let targetKey: string | null = null;
    for (let i = 0; i < edges.length; i++) {
        const element = `edges[${i}]`;
        const record = edges[i];
        if (!isJsonObject(record)) {
            ctx.badElement("edge", element);
            continue;
        }
        const source = endpointOf(record, json.sourceKey, VIS_SOURCE_KEYS);
        const target = endpointOf(record, json.targetKey, VIS_TARGET_KEYS);
        if (source.key === null || target.key === null) {
            ctx.missingEndpoint(element, source.key === null ? "from" : "to");
            continue;
        }
        sourceKey ??= source.key;
        targetKey ??= target.key;
        try {
            const u = ctx.coerceId(source.value, `${element}.from`);
            const v = ctx.coerceId(target.value, `${element}.to`);
            if (u === null || v === null) {
                ctx.countSkipped("edge");
                continue;
            }
            const idValue = ctx.edgeIdValue(ids, record.id);
            const edge = ctx.pushEdge(u, v, kind, ctx.weightOf(record), element);
            ctx.setEdgeId(ids, edge, idValue);
            const { weightFrom } = ctx.options;
            for (const key of Object.keys(record)) {
                if (key !== source.key && key !== target.key && key !== "id" && key !== weightFrom) {
                    ctx.edges.write(edge, key, record[key], SUFFIX.data);
                }
            }
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }
    ctx.setMeta(
        { dialect: "vis", nodeIdKey, sourceKey: sourceKey ?? undefined, targetKey: targetKey ?? undefined },
        ctx.weightOriginPatch(),
    );
}

// ============================================================ graphology

/**
 * Read a graphology serialisation: `options.type` decides the header direction ("mixed" or absent:
 * from the edges' `undirected` flags), `options.multi` the declared multigraph flag, node `key`
 * the id, `attributes` the columns, edge `key` the edge id.
 * @param ctx - the context
 * @param root - the document
 */
function importGraphology(ctx: ImportContext, root: JsonRecord): void {
    const { report } = ctx;
    const nodes = arraySection(root.nodes, "nodes", report) ?? [];
    const edges = arraySection(root.edges, "edges", report) ?? [];
    const options = isJsonObject(root.options) ? root.options : {};
    if (hasKey(root, "options") && !isJsonObject(root.options)) {
        report.warning("validation-error", JSON_ISSUE.BAD_FLAG, "options is not an object; ignored", {
            element: "options",
        });
    }
    let type: "directed" | "undirected" | "mixed";
    if (options.type === "directed" || options.type === "undirected" || options.type === "mixed") {
        ({ type } = options);
    } else {
        if (options.type !== undefined && options.type !== null) {
            report.warning(
                "validation-error",
                JSON_ISSUE.BAD_FLAG,
                `options.type ${describe(options.type)} is not directed, undirected or mixed; mixed assumed`,
                { element: "options.type" },
            );
        }
        type = "mixed";
    }
    let directed: boolean;
    if (type === "mixed") {
        const objects = edges.filter((edge) => isJsonObject(edge)) as JsonRecord[];
        const undirectedEdges = objects.filter((edge) => edge.undirected === true).length;
        directed = objects.length === 0 ? ctx.defaultDirected("graphology") : undirectedEdges < objects.length;
    } else {
        directed = type === "directed";
    }
    ctx.setHeader(directed);
    const multi = hasKey(options, "multi") ? flagOf(options.multi, "options.multi", false, report) : null;
    const allowSelfLoops = hasKey(options, "allowSelfLoops")
        ? flagOf(options.allowSelfLoops, "options.allowSelfLoops", true, report)
        : undefined;
    ctx.writeGraphDict(root.attributes, "attributes");
    ctx.reportNodeIdFrom("graphology", "the node key");
    ctx.sink.reserve(nodes.length, edges.length);

    for (let i = 0; i < nodes.length; i++) {
        const element = `nodes[${i}]`;
        const record = nodes[i];
        if (!isJsonObject(record)) {
            ctx.badElement("node", element);
            continue;
        }
        const id = ctx.coerceId(hasKey(record, "key") ? record.key : undefined, element);
        if (id === null) {
            ctx.countSkipped("node");
            continue;
        }
        pushNestedNode(ctx, id, record, "attributes", GRAPHOLOGY_NODE_KEYS, element);
    }
    throwIfAborted(ctx.options.signal);

    const ids = ctx.declareEdgeIds(edges, (edge) => edge.key, "key", true);
    for (let i = 0; i < edges.length; i++) {
        const element = `edges[${i}]`;
        const record = edges[i];
        if (!isJsonObject(record)) {
            ctx.badElement("edge", element);
            continue;
        }
        if (!hasKey(record, "source") || !hasKey(record, "target")) {
            ctx.missingEndpoint(element, hasKey(record, "source") ? "target" : "source");
            continue;
        }
        try {
            const u = ctx.coerceId(record.source, `${element}.source`);
            const v = ctx.coerceId(record.target, `${element}.target`);
            if (u === null || v === null) {
                ctx.countSkipped("edge");
                continue;
            }
            let kind: EdgeKind;
            if (type === "mixed") {
                kind = flagOf(record.undirected, `${element}.undirected`, false, report) ? "undirected" : "directed";
            } else {
                kind = type;
            }
            const attributes = isJsonObject(record.attributes) ? record.attributes : {};
            const idValue = ctx.edgeIdValue(ids, record.key);
            const edge = ctx.pushEdge(u, v, kind, ctx.weightOf(attributes), element);
            ctx.setEdgeId(ids, edge, idValue);
            ctx.writeNested(ctx.edges, edge, record, attributes, GRAPHOLOGY_EDGE_KEYS, ctx.options.weightFrom);
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }
    ctx.setMeta({ dialect: "graphology", allowSelfLoops }, { declaredMultigraph: multi, ...ctx.weightOriginPatch() });
}

/**
 * Push a node whose attributes live in a nested dict (graphology `attributes`, JGF `metadata`);
 * element-level keys the dialect does not define are kept with the `#element` suffix.
 * @param ctx - the context
 * @param id - the node id
 * @param record - the element record
 * @param dictKey - the key of the nested dict
 * @param structural - the element keys that are not attributes
 * @param element - the element name
 * @returns the node index, or -1 when skipped
 */
function pushNestedNode(
    ctx: ImportContext,
    id: NodeId,
    record: JsonRecord,
    dictKey: string,
    structural: ReadonlySet<string>,
    element: string,
): number {
    const index = ctx.pushNode(id, element);
    if (index < 0) {
        return index;
    }
    try {
        const dict = record[dictKey];
        if (dict !== undefined && dict !== null && !isJsonObject(dict)) {
            ctx.report.error(
                "validation-error",
                JSON_ISSUE.BAD_VALUE,
                `${element}: ${dictKey} must be an object, found ${describe(dict)}`,
                { element },
            );
        }
        ctx.writeNested(ctx.nodes, index, record, isJsonObject(dict) ? dict : {}, structural, null);
    } catch (err) {
        ctx.report.recordError(err, { element: String(id) });
    }
    return index;
}

// ============================================================ JSON Graph Format

/**
 * Read a JGF v2 document (`graph` or `graphs[graphIndex]`): nodes keyed by id (or a v1 array with
 * `id`), `label` with role "label", `metadata` as columns, edges with `id` / `relation` /
 * `directed` / `label` / `metadata`, hyperedges per the `hyperedges` option.
 * @param ctx - the context
 * @param root - the document
 */
function importJgf(ctx: ImportContext, root: JsonRecord): void {
    const { report } = ctx;
    const graph = jgfGraphOf(ctx, root);
    const edges = arraySection(graph.edges, "graph.edges", report) ?? [];
    const hyperedges = arraySection(graph.hyperedges, "graph.hyperedges", report) ?? [];
    const nodesRaw: unknown = graph.nodes;
    if (nodesRaw !== undefined && nodesRaw !== null && !isJsonObject(nodesRaw) && !Array.isArray(nodesRaw)) {
        report.fail(
            JSON_ISSUE.SHAPE,
            `graph.nodes must be an object keyed by id or an array, found ${describe(nodesRaw)}`,
        );
    }
    const nodeRecords: readonly unknown[] = Array.isArray(nodesRaw) ? nodesRaw : Object.values(nodesRaw ?? {});

    let directed: boolean;
    if (typeof graph.directed === "boolean") {
        ({ directed } = graph);
    } else if (ctx.explicitDefaultDirected) {
        directed = ctx.options.defaultDirected;
    } else {
        if (graph.directed !== undefined && graph.directed !== null) {
            report.warning(
                "validation-error",
                JSON_ISSUE.BAD_FLAG,
                `graph.directed is ${describe(graph.directed)}, not a boolean`,
                { element: "graph.directed" },
            );
        }
        // the spec default is true; a file whose every edge says directed: false is read as undirected
        const objects = edges.filter((edge) => isJsonObject(edge)) as JsonRecord[];
        directed = objects.length === 0 || !objects.every((edge) => edge.directed === false);
    }
    ctx.setHeader(directed);
    ctx.writeGraphDict(graph.metadata, "graph.metadata");
    const shape: JsonShapeMeta = {
        dialect: "jgf",
        id: typeof graph.id === "string" ? graph.id : undefined,
        type: typeof graph.type === "string" ? graph.type : undefined,
    };
    const label = typeof graph.label === "string" ? graph.label : null;

    ctx.reportNodeIdFrom("jgf", "the node keys");
    ctx.sink.reserve(nodeRecords.length, edges.length);
    const labelColumn = ctx.nodes.declareIf(
        { name: "label", dtype: "string", role: "label", nullable: true },
        anyHas(nodeRecords, "label"),
    );
    if (isJsonObject(nodesRaw)) {
        for (const key of Object.keys(nodesRaw)) {
            const element = `nodes[${JSON.stringify(key)}]`;
            const record = nodesRaw[key];
            if (record !== null && !isJsonObject(record)) {
                ctx.badElement("node", element);
                continue;
            }
            const id = ctx.coerceId(key, element);
            if (id === null) {
                ctx.countSkipped("node");
                continue;
            }
            pushJgfNode(ctx, id, record ?? {}, labelColumn, element);
        }
    } else {
        for (let i = 0; i < nodeRecords.length; i++) {
            const element = `nodes[${i}]`;
            const record = nodeRecords[i];
            if (!isJsonObject(record)) {
                ctx.badElement("node", element);
                continue;
            }
            const id = ctx.coerceId(hasKey(record, "id") ? record.id : undefined, element);
            if (id === null) {
                ctx.countSkipped("node");
                continue;
            }
            pushJgfNode(ctx, id, record, labelColumn, element);
        }
    }
    throwIfAborted(ctx.options.signal);

    const ids = ctx.declareEdgeIds([...edges, ...hyperedges], (edge) => edge.id, "id", false);
    const relationColumn = ctx.edges.declareIf(
        { name: "relation", dtype: "string", role: "kind", nullable: true },
        anyHas(edges, "relation") || anyHas(hyperedges, "relation"),
    );
    const edgeLabelColumn = ctx.edges.declareIf(
        { name: "label", dtype: "string", role: "label", nullable: true },
        anyHas(edges, "label") || anyHas(hyperedges, "label"),
    );
    const writeJgfEdge = (
        record: JsonRecord,
        u: NodeId,
        v: NodeId,
        kind: EdgeKind,
        element: string,
        structural: ReadonlySet<string> = JGF_EDGE_KEYS,
    ): void => {
        const { metadata } = record;
        const dict = isJsonObject(metadata) ? metadata : {};
        const idValue = ctx.edgeIdValue(ids, record.id);
        const edge = ctx.pushEdge(u, v, kind, ctx.weightOf(dict), element);
        ctx.setEdgeId(ids, edge, idValue);
        ctx.writeStringField(ctx.edges, relationColumn, edge, record.relation, "relation", element);
        ctx.writeStringField(ctx.edges, edgeLabelColumn, edge, record.label, "label", element);
        if (metadata !== undefined && metadata !== null && !isJsonObject(metadata)) {
            report.error("validation-error", JSON_ISSUE.BAD_VALUE, `${element}: metadata must be an object`, {
                element,
            });
        }
        ctx.writeNested(ctx.edges, edge, record, dict, structural, ctx.options.weightFrom);
    };

    for (let i = 0; i < edges.length; i++) {
        const element = `edges[${i}]`;
        const record = edges[i];
        if (!isJsonObject(record)) {
            ctx.badElement("edge", element);
            continue;
        }
        if (!hasKey(record, "source") || !hasKey(record, "target")) {
            ctx.missingEndpoint(element, hasKey(record, "source") ? "target" : "source");
            continue;
        }
        try {
            const u = ctx.coerceId(record.source, `${element}.source`);
            const v = ctx.coerceId(record.target, `${element}.target`);
            if (u === null || v === null) {
                ctx.countSkipped("edge");
                continue;
            }
            const edgeDirected = flagOf(record.directed, `${element}.directed`, directed, report);
            writeJgfEdge(record, u, v, edgeDirected ? "directed" : "undirected", element);
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }

    importHyperedges(ctx, hyperedges, directed, writeJgfEdge);
    ctx.setMeta(shape, { name: label, ...ctx.weightOriginPatch() });
}

/**
 * The graph object of a JGF document: `graph`, or `graphs[graphIndex]`.
 * @param ctx - the context
 * @param root - the document
 * @returns the graph object; the import fails when there is none
 */
function jgfGraphOf(ctx: ImportContext, root: JsonRecord): JsonRecord {
    const { report } = ctx;
    if (isJsonObject(root.graph)) {
        return root.graph;
    }
    const graphs = arraySection(root.graphs, "graphs", report) ?? [];
    if (graphs.length === 0) {
        report.fail(JSON_ISSUE.SHAPE, "a JGF document needs a graph object or a non-empty graphs array");
    }
    if (graphs.length > 1) {
        report.warning(
            "unsupported",
            JSON_ISSUE.MULTIPLE_GRAPHS,
            `the document holds ${graphs.length} graphs; only graphs[${ctx.json.graphIndex}] is read`,
            { element: "graphs" },
        );
    }
    if (ctx.json.graphIndex >= graphs.length) {
        report.fail(JSON_ISSUE.SHAPE, `graphIndex ${ctx.json.graphIndex} is beyond the ${graphs.length} graph(s)`);
    }
    const graph = graphs[ctx.json.graphIndex];
    if (!isJsonObject(graph)) {
        return report.fail(JSON_ISSUE.SHAPE, `graphs[${ctx.json.graphIndex}] is not an object`);
    }
    return graph;
}

/**
 * Push a JGF node: `label` into the declared column, `metadata` as attributes.
 * @param ctx - the context
 * @param id - the node id
 * @param record - the node record
 * @param labelColumn - the label column handle, or INVALID_INDEX
 * @param element - the element name
 */
function pushJgfNode(
    ctx: ImportContext,
    id: NodeId,
    record: JsonRecord,
    labelColumn: ColumnHandle,
    element: string,
): void {
    const index = ctx.pushNode(id, element);
    if (index < 0) {
        return;
    }
    try {
        ctx.writeStringField(ctx.nodes, labelColumn, index, record.label, "label", element);
        const { metadata } = record;
        if (metadata !== undefined && metadata !== null && !isJsonObject(metadata)) {
            ctx.report.error("validation-error", JSON_ISSUE.BAD_VALUE, `${element}: metadata must be an object`, {
                element,
            });
        }
        ctx.writeNested(ctx.nodes, index, record, isJsonObject(metadata) ? metadata : {}, JGF_NODE_KEYS, null);
    } catch (err) {
        ctx.report.recordError(err, { element: String(id) });
    }
}

/**
 * JGF hyperedges per the `hyperedges` option: "error" aborts, "skip" (default) records a warning
 * and a loss note, "star" and "clique" expand an undirected `{ nodes }` hyperedge into edges from
 * its first node (star) or between every pair (clique); a directed `{ source, target }` hyperedge
 * becomes every source -> target edge under both policies. Every expanded edge carries the
 * hyperedge's id, relation, label and metadata.
 * @param ctx - the context
 * @param hyperedges - the hyperedge records
 * @param directed - the graph's direction
 * @param push - the edge writer of the JGF reader
 */
function importHyperedges(
    ctx: ImportContext,
    hyperedges: readonly unknown[],
    directed: boolean,
    push: (
        record: JsonRecord,
        u: NodeId,
        v: NodeId,
        kind: EdgeKind,
        element: string,
        structural: ReadonlySet<string>,
    ) => void,
): void {
    if (hyperedges.length === 0) {
        return;
    }
    const { report } = ctx;
    const policy = ctx.options.hyperedges;
    if (policy === "error") {
        report.error("unsupported", JSON_ISSUE.HYPEREDGE, `${hyperedges.length} hyperedge(s) (hyperedges: "error")`, {
            element: "hyperedges",
        });
        throw report.abort("hyperedges refused", { code: JSON_ISSUE.HYPEREDGE, count: hyperedges.length });
    }
    if (policy === "skip") {
        report.warning("unsupported", JSON_ISSUE.HYPEREDGES_SKIPPED, `${hyperedges.length} hyperedge(s) skipped`, {
            element: "hyperedges",
        });
        report.loss(
            JSON_ISSUE.HYPEREDGES_SKIPPED,
            `${hyperedges.length} hyperedge(s) were not imported`,
            null,
            hyperedges.length,
        );
        return;
    }
    for (let i = 0; i < hyperedges.length; i++) {
        const element = `hyperedges[${i}]`;
        const record = hyperedges[i];
        if (!isJsonObject(record)) {
            ctx.badElement("edge", element);
            continue;
        }
        try {
            if (Array.isArray(record.nodes)) {
                const members = record.nodes.map((raw) => ctx.requireId(raw, element));
                if (members.length < 2) {
                    throw new GraphFormatError(
                        "E_INVALID_ID",
                        `${element}: an undirected hyperedge needs two or more nodes`,
                        { reason: "hyperedge shape" },
                    );
                }
                const kind: EdgeKind = directed ? "directed" : "undirected";
                if (policy === "star") {
                    for (let k = 1; k < members.length; k++) {
                        push(record, members[0], members[k], kind, element, JGF_HYPEREDGE_KEYS);
                    }
                } else {
                    for (let a = 0; a < members.length; a++) {
                        for (let b = a + 1; b < members.length; b++) {
                            push(record, members[a], members[b], kind, element, JGF_HYPEREDGE_KEYS);
                        }
                    }
                }
            } else if (Array.isArray(record.source) && Array.isArray(record.target)) {
                const sources = record.source.map((raw) => ctx.requireId(raw, element));
                const targets = record.target.map((raw) => ctx.requireId(raw, element));
                if (sources.length === 0 || targets.length === 0) {
                    throw new GraphFormatError(
                        "E_INVALID_ID",
                        `${element}: a directed hyperedge needs sources and targets`,
                        { reason: "hyperedge shape" },
                    );
                }
                for (const s of sources) {
                    for (const t of targets) {
                        push(record, s, t, "directed", element, JGF_HYPEREDGE_KEYS);
                    }
                }
            } else {
                report.error(
                    "validation-error",
                    JSON_ISSUE.HYPEREDGE_SHAPE,
                    `${element} has neither a nodes array nor source / target arrays`,
                    { element },
                );
                ctx.countSkipped("edge");
            }
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }
}

// ============================================================ Cytoscape

/**
 * Read Cytoscape.js elements: `elements.nodes` / `elements.edges`, a flat `elements` array (group
 * from `group` or from the presence of source / target), or a top-level array. `data.id` is the id,
 * `data.parent` the parent (resolved after every node is known), `position` the position column,
 * `classes` the classes list; the other `data` keys are attributes and the element-level keys
 * (selected, locked, ...) are columns of the same name.
 * @param ctx - the context
 * @param root - the document
 */
function importCytoscape(ctx: ImportContext, root: unknown): void {
    const { report } = ctx;
    let elements: unknown;
    let extra: Record<string, unknown> | undefined;
    if (Array.isArray(root)) {
        elements = root;
    } else if (isJsonObject(root)) {
        ({ elements } = root);
        ctx.writeGraphDict(root.data, "data");
        const rest: Record<string, unknown> = {};
        for (const key of Object.keys(root)) {
            if (key !== "elements" && key !== "data") {
                rest[key] = root[key];
            }
        }
        if (Object.keys(rest).length > 0) {
            extra = rest;
        }
    } else {
        report.fail(JSON_ISSUE.SHAPE, `a Cytoscape document must be an object or an array, found ${describe(root)}`);
    }
    const { nodes, edges } = cytoscapeSections(ctx, elements);
    ctx.setHeader(ctx.defaultDirected("cytoscape"));
    ctx.reportNodeIdFrom("cytoscape", "data.id");
    ctx.sink.reserve(nodes.length, edges.length);

    const dataHas = (items: readonly unknown[], key: string): boolean =>
        items.some((item) => isJsonObject(item) && isJsonObject(item.data) && item.data[key] !== undefined);
    const positionColumn = ctx.nodes.declareIf(
        {
            name: POSITION_COLUMN,
            dtype: "f32",
            components: 3,
            role: "position",
            mutable: true,
            nullable: true,
            extra: { sourceDims: 2, units: "file" },
            origin: { format: "json", namespace: "cytoscape" },
        },
        anyHas(nodes, "position"),
    );
    const classesColumn = ctx.nodes.declareIf(
        { name: CLASSES_COLUMN, dtype: "list", itemDtype: "string", role: "classes", nullable: true },
        anyHas(nodes, "classes"),
    );
    const parentColumn = ctx.nodes.declareIf(
        { name: PARENT_COLUMN, dtype: "u32", role: "parent", refersTo: "node", nullable: true },
        dataHas(nodes, "parent"),
    );
    const edgeClassesColumn = ctx.edges.declareIf(
        { name: CLASSES_COLUMN, dtype: "list", itemDtype: "string", role: "classes", nullable: true },
        anyHas(edges, "classes"),
    );
    const parents: { index: number; parent: NodeId; element: string }[] = [];
    const point: [number, number, number] = [0, 0, 0];

    for (let i = 0; i < nodes.length; i++) {
        const element = `nodes[${i}]`;
        const record = nodes[i];
        if (!isJsonObject(record) || !isJsonObject(record.data)) {
            ctx.badElement("node", element, "an element with a data object");
            continue;
        }
        const { data } = record;
        const id = ctx.coerceId(hasKey(data, "id") ? data.id : undefined, element);
        if (id === null) {
            ctx.countSkipped("node");
            continue;
        }
        const index = ctx.pushNode(id, element);
        if (index < 0) {
            continue;
        }
        try {
            for (const key of Object.keys(data)) {
                if (key === "id") {
                    continue;
                }
                if (key === "parent") {
                    const raw = data.parent;
                    if (raw !== undefined && raw !== null) {
                        const parent = ctx.coerceId(raw, `${element}.data.parent`);
                        if (parent !== null) {
                            parents.push({ index, parent, element });
                        }
                    }
                    continue;
                }
                ctx.nodes.write(index, key, data[key], SUFFIX.data);
            }
            const { position } = record;
            if (position !== undefined && position !== null) {
                if (isJsonObject(position) && typeof position.x === "number" && typeof position.y === "number") {
                    point[0] = position.x;
                    point[1] = position.y;
                    ctx.nodes.set(positionColumn, index, point);
                } else {
                    report.error(
                        "validation-error",
                        JSON_ISSUE.BAD_VALUE,
                        `${element}: position must be an object with numeric x and y`,
                        { element },
                    );
                }
            }
            writeClasses(ctx, ctx.nodes, classesColumn, index, record.classes, element);
            writeElementKeys(ctx.nodes, index, record);
        } catch (err) {
            report.recordError(err, { element: String(id) });
        }
    }
    for (const { index, parent, element } of parents) {
        const parentIndex = ctx.sink.indexOf(parent);
        if (parentIndex === INVALID_INDEX) {
            report.error(
                "missing-value",
                JSON_ISSUE.UNKNOWN_PARENT,
                `${element}: parent ${JSON.stringify(parent)} is not a node`,
                { element },
            );
            continue;
        }
        ctx.nodes.set(parentColumn, index, parentIndex);
    }
    throwIfAborted(ctx.options.signal);

    importCytoscapeEdges(ctx, edges, edgeClassesColumn);
    ctx.setMeta({ dialect: "cytoscape", cytoscape: extra }, ctx.weightOriginPatch());
}

/**
 * Read the Cytoscape edge elements: `data.id`, `data.source` / `data.target`, the weight, the
 * other data keys as attributes, `classes` and the element-level keys.
 * @param ctx - the context
 * @param edges - the edge elements
 * @param edgeClassesColumn - the edge classes column, or INVALID_INDEX
 */
function importCytoscapeEdges(ctx: ImportContext, edges: readonly unknown[], edgeClassesColumn: ColumnHandle): void {
    const ids = ctx.declareEdgeIds(edges, (edge) => (isJsonObject(edge.data) ? edge.data.id : undefined), "id", true);
    const kind = ctx.uniformKind();
    for (let i = 0; i < edges.length; i++) {
        const element = `edges[${i}]`;
        const record = edges[i];
        if (!isJsonObject(record) || !isJsonObject(record.data)) {
            ctx.badElement("edge", element, "an element with a data object");
            continue;
        }
        const { data } = record;
        if (!hasKey(data, "source") || !hasKey(data, "target")) {
            ctx.missingEndpoint(element, `data.${hasKey(data, "source") ? "target" : "source"}`);
            continue;
        }
        try {
            const u = ctx.coerceId(data.source, `${element}.data.source`);
            const v = ctx.coerceId(data.target, `${element}.data.target`);
            if (u === null || v === null) {
                ctx.countSkipped("edge");
                continue;
            }
            const idValue = ctx.edgeIdValue(ids, data.id);
            const edge = ctx.pushEdge(u, v, kind, ctx.weightOf(data), element);
            ctx.setEdgeId(ids, edge, idValue);
            const { weightFrom } = ctx.options;
            for (const key of Object.keys(data)) {
                if (key !== "id" && key !== "source" && key !== "target" && key !== weightFrom) {
                    ctx.edges.write(edge, key, data[key], SUFFIX.data);
                }
            }
            writeClasses(ctx, ctx.edges, edgeClassesColumn, edge, record.classes, element);
            writeElementKeys(ctx.edges, edge, record);
        } catch (err) {
            ctx.skip(err, "edge", element);
        }
    }
}

/**
 * The node and edge element arrays of a Cytoscape `elements` value: an object with `nodes` /
 * `edges`, a flat array split by isNodeElement(), or nothing (reported).
 * @param ctx - the context
 * @param elements - the `elements` value
 * @returns the two arrays; the import fails when elements has another type
 */
function cytoscapeSections(
    ctx: ImportContext,
    elements: unknown,
): { readonly nodes: readonly unknown[]; readonly edges: readonly unknown[] } {
    const { report } = ctx;
    if (Array.isArray(elements)) {
        return {
            nodes: elements.filter((item) => isJsonObject(item) && isNodeElement(item)),
            edges: elements.filter((item) => isJsonObject(item) && !isNodeElement(item)),
        };
    }
    if (isJsonObject(elements)) {
        return {
            nodes: arraySection(elements.nodes, "elements.nodes", report) ?? [],
            edges: arraySection(elements.edges, "elements.edges", report) ?? [],
        };
    }
    if (elements === undefined || elements === null) {
        report.error("missing-value", JSON_ISSUE.MISSING_SECTION, "the document has no elements", {
            element: "elements",
        });
        return { nodes: [], edges: [] };
    }
    return report.fail(JSON_ISSUE.SHAPE, `elements must be an object or an array, found ${describe(elements)}`);
}

/**
 * Whether a flat Cytoscape element is a node: `group: "nodes"`, or no source / target in its data.
 * @param element - the element
 * @returns true for a node
 */
function isNodeElement(element: JsonRecord): boolean {
    if (element.group === "nodes") {
        return true;
    }
    if (element.group === "edges") {
        return false;
    }
    const { data } = element;
    return !(isJsonObject(data) && hasKey(data, "source") && hasKey(data, "target"));
}

/**
 * Write the classes of an element: a space-separated string or an array of strings.
 * @param ctx - the context
 * @param writer - the table writer
 * @param column - the classes column, or INVALID_INDEX when no element has classes
 * @param row - the row
 * @param raw - the `classes` value
 * @param element - the element name
 */
function writeClasses(
    ctx: ImportContext,
    writer: AttributeWriter,
    column: ColumnHandle,
    row: number,
    raw: unknown,
    element: string,
): void {
    if (column === INVALID_INDEX || raw === undefined || raw === null) {
        return;
    }
    let classes: string[];
    if (typeof raw === "string") {
        classes = raw.split(/\s+/).filter((c) => c.length > 0);
    } else if (Array.isArray(raw) && raw.every((c) => typeof c === "string")) {
        classes = raw;
    } else {
        ctx.report.error(
            "validation-error",
            JSON_ISSUE.BAD_VALUE,
            `${element}: classes must be a string or an array of strings`,
            { element },
        );
        return;
    }
    writer.set(column, row, classes);
}

/**
 * Write the element-level keys of a Cytoscape element other than the structural ones: the known
 * keys under their own name, unknown ones with the `#element` suffix.
 * @param writer - the table writer
 * @param row - the row
 * @param record - the element
 */
function writeElementKeys(writer: AttributeWriter, row: number, record: JsonRecord): void {
    for (const key of Object.keys(record)) {
        if (CYTOSCAPE_STRUCTURAL_KEYS.has(key)) {
            continue;
        }
        const name = CYTOSCAPE_ELEMENT_KEYS.has(key) ? key : `${key}${SUFFIX.element}`;
        writer.write(row, name, record[key], SUFFIX.data);
    }
}

// ============================================================ the plugin

/**
 * The JSON importer plugin (design section 8.4).
 */
export const jsonImporter: GraphImporter<JsonImportOptions> = Object.freeze({
    format: "json",
    extensions: Object.freeze([".json"]),
    mimeTypes: Object.freeze(["application/json"]),

    /**
     * Confidence that the head is a JSON graph document: 0 unless it starts with `{` or `[`, 0.5
     * for any JSON, 0.9 when a graph key (nodes, links, edges, elements, graph, graphs) appears in
     * the head.
     * @param head - the first bytes
     * @returns the confidence
     */
    sniff(head: Uint8Array): number {
        const text = new TextDecoder("utf-8").decode(head.subarray(0, SNIFF_BYTES));
        const trimmed = (text.startsWith(BOM) ? text.slice(1) : text).trimStart();
        if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
            return 0;
        }
        return SNIFF_KEYS.some((key) => trimmed.includes(key)) ? 0.9 : 0.5;
    },

    /**
     * Read a JSON graph document into the sink.
     * @param input - the text, bytes or stream
     * @param sink - the sink
     * @param options - format-specific and common options
     * @returns the import report; ImportError on a fatal error or beyond the error limit
     */
    async import(
        input: ImportInput,
        sink: GraphSink,
        options?: JsonImportOptions & CommonImportOptions,
    ): Promise<ImportReport> {
        const resolved = resolveImportOptions(options, FORMAT_DEFAULTS);
        const json = resolveJsonOptions(options);
        const report = new ImportReportBuilder("json", resolved.errorLimit);
        const text = await readText(input, report, resolved);
        const root = parseDocument(text, report);
        const dialect = detectDialect(root, json.dialect, report);
        const ctx = new ImportContext(sink, report, resolved, json, options?.defaultDirected !== undefined);
        reportSinkOptions(sink, options, report);
        reportUnusedOptions(options, report, USED_OPTIONS);
        if (dialect === "cytoscape") {
            importCytoscape(ctx, root);
            throwIfAborted(resolved.signal);
            return report.finish();
        }
        const doc = isJsonObject(root)
            ? root
            : report.fail(JSON_ISSUE.SHAPE, `a ${dialect} document must be a JSON object, found ${describe(root)}`);
        switch (dialect) {
            case "node-link":
            case "d3":
                importNodeLink(ctx, doc, dialect);
                break;
            case "jgf":
                importJgf(ctx, doc);
                break;
            case "graphology":
                importGraphology(ctx, doc);
                break;
            case "vis":
                importVis(ctx, doc);
                break;
            default: {
                const name: string = dialect;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown dialect ${name}`, {
                    option: "dialect",
                    found: name,
                });
            }
        }
        throwIfAborted(resolved.signal);
        return report.finish();
    },
});
