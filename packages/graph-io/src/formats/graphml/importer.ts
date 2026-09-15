/**
 * The GraphML importer (design sections 8.4 and 5.1; research note 07 section 2.3): a single-pass
 * SAX-style reader over the streaming XML tokenizer that pushes scalars into the sink as elements
 * close. `<key>` declarations become typed columns up front (boolean / int / long / float / double
 * / string, with `<default>`, for the node, edge and graph domains; `for="all"` declares in all
 * three with the same origin.id); `edgedefault` and per-edge `directed` go through the common
 * DirectionResolver; a nested `<graph>` inside a node becomes the `parent` column (containment,
 * never topology); `<port>` declarations and `<locator>` elements are reported and dropped while
 * `sourceport` / `targetport` edge attributes are kept as role columns; hyperedges follow the
 * `hyperedges` option (refuse, skip with a report entry, star, clique); yFiles `yfiles.type` keys
 * become `json` columns holding the nested XML as a tree (origin.namespace "yfiles"), reported as
 * a loss note because the structure, not the bytes, is preserved.
 *
 * Ids are coerced with the common rule (`ids: "canonical"` by default, design section 4.1); the
 * edge attribute whose `attr.name` is `weightFrom` ("weight" by default) is THE weight and is
 * passed to `addEdge` rather than stored as a column, with its declaration kept in
 * `meta.weightOrigin`; a node key named `graphty:originalId` restores ids the exporter mangled.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphSink,
    INVALID_INDEX,
    MAX_COUNT,
    type NodeId,
} from "@graphty/graph-format";

import {
    declareAttribute,
    type DeclaredAttribute,
    declareResolved,
    losesPrecision,
    parseDeclaredValue,
    PRECISION_CODE,
    takenIn,
    uniqueColumnName,
} from "../../common/attributes.js";
import { type DeclaredTypeSpec } from "../../common/declared-types.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { IdCoercer } from "../../common/ids.js";
import { textChunks, throwIfAborted } from "../../common/input.js";
import { type ListSyntax } from "../../common/lists.js";
import {
    type ImportFormatDefaults,
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { isWeightField, parseWeightText } from "../../common/weights.js";
import { isWhitespace, localName, tokenizeXml, type XmlHandler, XmlSyntaxError } from "../../common/xml.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    EDGE_ID_COLUMN,
    EXTENSIONS,
    FORMAT,
    GRAPHML_ISSUE,
    GRAPHML_LOSS,
    GRAPHML_NAMESPACE,
    type GraphmlMeta,
    HYPEREDGE_HUB_COLUMN,
    LABEL_COLUMN,
    META_KEY,
    MIME_TYPES,
    ORIGINAL_ID_ATTRIBUTE,
    ORIGINAL_ID_COLUMN,
    PARENT_COLUMN,
    RESERVED_EDGE_NAMES,
    RESERVED_NODE_NAMES,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
    XSI_NAMESPACE,
} from "./constants.js";
import { XmlTreeBuilder } from "./tree.js";

/** The format-specific options of the GraphML importer. */
export interface GraphmlImportOptions {
    /**
     * How keys with a `yfiles.type` (yEd / yFiles graphics) are read: "json" (default) keeps the
     * nested XML of each `<data>` as a json column with origin.namespace "yfiles"; "skip" reports
     * them once and declares nothing.
     */
    yfiles?: "json" | "skip" | undefined;
}

/** The per-format defaults of design section 8.4: text-cell ids, undirected when edgedefault is missing, weight key. */
const FORMAT_DEFAULTS: ImportFormatDefaults = { ids: "canonical", defaultDirected: false, weightFrom: "weight" };

/** Bytes inspected by sniff(). */
const SNIFF_BYTES = 4096;

const YFILES_MODES: ReadonlySet<string> = new Set(["json", "skip"]);

/** Elements between two checks of the cancellation signal. */
const ABORT_CHECK_INTERVAL = 64;

/** The common options the GraphML importer reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "defaultDirected",
    "weightFrom",
    "weightDtype",
    "long",
    "restoreMangledIds",
    "hyperedges",
    "errorLimit",
    "signal",
    "onProgress",
]);

/** Where the reader is, by the innermost open element. */
const enum Ctx {
    Graphml,
    Desc,
    Key,
    Default,
    Graph,
    Node,
    Edge,
    Hyperedge,
    Data,
    Capture,
    Skip,
}

/** The element domains a key can be declared for. */
type Domain = "graph" | "node" | "edge";

/** A declared column of one domain. */
interface ColumnTarget {
    /** The column name in its table. */
    readonly name: string;
    /** The handle (node / edge domains); INVALID_INDEX for the graph domain. */
    readonly handle: ColumnHandle;
    /** The declaration (graph domain, re-sent with every setGraphValue). */
    readonly decl: ColumnDecl;
    /** How text values are parsed; null for a json (yfiles) column. */
    readonly spec: DeclaredTypeSpec | null;
    /** The list syntax (unused by GraphML, which has no list type). */
    readonly listSyntax: ListSyntax;
}

/** One `<key>` after `</key>`. */
interface KeyEntry {
    readonly id: string;
    readonly name: string;
    readonly node: ColumnTarget | null;
    readonly edge: ColumnTarget | null;
    readonly graph: ColumnTarget | null;
    /** The edge domain of this key is THE weight. */
    readonly weight: boolean;
    /** The node domain of this key restores mangled ids. */
    readonly originalId: boolean;
    /** Whether the key holds yFiles nested XML. */
    readonly yfiles: boolean;
    /** Whether data of this key is silently ignored (yfiles: "skip"; already reported). */
    readonly skipped: boolean;
    /** The domains the key was declared for, for the mismatch warning. */
    readonly domains: readonly Domain[];
}

/** A `<key>` while its `<default>` and `<desc>` are being read. */
interface PendingKey {
    readonly id: string | null;
    readonly attrName: string | null;
    readonly attrType: string | null;
    readonly yfilesType: string | null;
    readonly domains: readonly Domain[];
    readonly line: number;
    desc: string;
    defaultText: string | null;
    defaultTree: unknown;
}

/** A `<node>` while open (nodes nest through nested graphs, so these stack). */
interface NodeState {
    /** The coerced id, or null when the element is unusable. */
    id: NodeId | null;
    /** The id attribute as written, for the mangled -> original map. */
    idText: string | null;
    /** The node index once added; INVALID_INDEX before. */
    index: number;
    /** The index of the containing node, or INVALID_INDEX at the top level. */
    readonly parent: number;
    /** Data values waiting for the node to be added: handle, value, handle, value... */
    readonly pending: unknown[];
    /** A `graphty:originalId` value, restored as the id. */
    restoredId: string | null;
    failed: boolean;
    readonly line: number;
}

/** The `<edge>` being read. */
interface EdgeState {
    source: NodeId | null;
    target: NodeId | null;
    id: string | null;
    kind: EdgeKind;
    sourcePort: string | null;
    targetPort: string | null;
    weight: number | undefined;
    readonly pending: unknown[];
    failed: boolean;
    line: number;
}

/** One `<endpoint>` of a hyperedge. */
interface Endpoint {
    readonly node: NodeId;
    readonly type: "in" | "out" | "undir";
}

/** The `<hyperedge>` being read. */
interface HyperedgeState {
    id: string | null;
    readonly endpoints: Endpoint[];
    failed: boolean;
    line: number;
}

/** The `<data>` or `<default>` being read. */
interface ContentState {
    key: KeyEntry | null;
    pendingKey: PendingKey | null;
    domain: Domain;
    text: string;
    tree: XmlTreeBuilder | null;
    line: number;
}

/** Where an issue was found. */
interface Where {
    /** The line. */
    readonly line: number;
    /** The element, or null. */
    readonly element: string | null;
}

/** One open `<graph>`. */
interface GraphState {
    readonly directed: boolean;
    readonly parent: number;
}

/**
 * A growable bit set over node indices, marking the indices declared by a `<node>` element so a
 * second declaration is reported and a node created by an edge reference is not.
 */
class IndexFlags {
    private words = new Uint32Array(64);

    /**
     * Whether an index is marked.
     * @param index - the node index
     * @returns true when set
     */
    has(index: number): boolean {
        const w = index >>> 5;
        return w < this.words.length && ((this.words[w] >>> (index & 31)) & 1) === 1;
    }

    /**
     * Mark an index.
     * @param index - the node index
     */
    add(index: number): void {
        const w = index >>> 5;
        if (w >= this.words.length) {
            let size = this.words.length * 2;
            while (size <= w) {
                size *= 2;
            }
            const grown = new Uint32Array(size);
            grown.set(this.words);
            this.words = grown;
        }
        this.words[w] |= 1 << (index & 31);
    }
}

/**
 * The event handler that turns the tokenizer's events into sink calls.
 */
class GraphmlReader implements XmlHandler {
    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private readonly options: ResolvedImportOptions;

    private readonly yfilesMode: "json" | "skip";

    private readonly coercer: IdCoercer;

    private readonly direction: DirectionResolver;

    private readonly ctx: Ctx[] = [];

    private readonly keys = new Map<string, KeyEntry>();

    private readonly graphs: GraphState[] = [];

    private readonly nodes: NodeState[] = [];

    private readonly declared = new IndexFlags();

    private readonly edgeIds = new Set<string>();

    private elementsSinceCheck = 0;

    /** Mangled id text -> the original id restored from `graphty:originalId`, for edge endpoints. */
    private readonly restored = new Map<string, NodeId>();

    private readonly graphColumnNames = new Set<string>();

    private pendingKey: PendingKey | null = null;

    private edge: EdgeState | null = null;

    private hyperedge: HyperedgeState | null = null;

    private readonly content: ContentState = {
        key: null,
        pendingKey: null,
        domain: "node",
        text: "",
        tree: null,
        line: 0,
    };

    private descTarget: "graph" | "key" | null = null;

    private descText = "";

    private description: string | null = null;

    private graphSeen = false;

    private graphId: string | null = null;

    private edgedefault: "directed" | "undirected" | null = null;

    private namespaces: Record<string, string> = {};

    private weightOrigin: { id: string; title: string | null; type: string | null } | null = null;

    private yfilesLossNoted = false;

    private hyperedgeCount = 0;

    private edgeIdHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private sourcePortHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private targetPortHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private parentHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private hubHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    /**
     * Create a reader for one import call.
     * @param sink - the sink
     * @param report - the report
     * @param options - the resolved common options
     * @param yfilesMode - the resolved yfiles option
     */
    constructor(
        sink: GraphSink,
        report: ImportReportBuilder,
        options: ResolvedImportOptions,
        yfilesMode: "json" | "skip",
    ) {
        this.sink = sink;
        this.report = report;
        this.options = options;
        this.yfilesMode = yfilesMode;
        this.coercer = new IdCoercer(options.ids);
        this.direction = new DirectionResolver(sink, report, options.onMixedDirection);
    }

    // ------------------------------------------------------------------ tokenizer events

    /**
     * An element starts.
     * @param name - the element name
     * @param attrs - its attributes
     * @param line - the line
     */
    start(name: string, attrs: ReadonlyMap<string, string>, line: number): void {
        const parent = this.ctx.length === 0 ? null : this.ctx[this.ctx.length - 1];
        switch (parent) {
            case Ctx.Capture:
                this.captureStart(name, attrs);
                return;
            case Ctx.Skip:
            case Ctx.Desc:
                this.ctx.push(Ctx.Skip);
                return;
            case Ctx.Data:
            case Ctx.Default:
                this.captureStart(name, attrs);
                return;
            default:
                break;
        }
        const local = localName(name);
        switch (parent) {
            case null:
                if (local !== "graphml") {
                    this.report.fail(GRAPHML_ISSUE.NOT_GRAPHML, `the root element is <${name}>, not <graphml>`, {
                        line,
                    });
                }
                this.readRoot(attrs);
                this.ctx.push(Ctx.Graphml);
                return;
            case Ctx.Graphml:
                this.startInGraphml(name, local, attrs, line);
                return;
            case Ctx.Key:
                this.startInKey(name, local, line);
                return;
            case Ctx.Graph:
                this.startInGraph(name, local, attrs, line);
                return;
            case Ctx.Node:
                this.startInNode(name, local, attrs, line);
                return;
            case Ctx.Edge:
                this.startInEdge(name, local, attrs, line);
                return;
            case Ctx.Hyperedge:
                this.startInHyperedge(name, local, attrs, line);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * An element ends.
     * @param name - the element name
     * @param _line - the line (unused: issues of an element cite its start line)
     */
    end(name: string, _line: number): void {
        const ctx = this.ctx.pop();
        switch (ctx) {
            case Ctx.Capture:
                this.captureEnd();
                return;
            case Ctx.Skip:
            case Ctx.Graphml:
            case undefined:
                return;
            case Ctx.Desc:
                this.finishDesc();
                return;
            case Ctx.Key:
                this.finishKey();
                return;
            case Ctx.Default:
                this.finishDefault();
                return;
            case Ctx.Data:
                this.finishData();
                return;
            case Ctx.Node:
                this.finishNode();
                return;
            case Ctx.Edge:
                this.finishEdge();
                return;
            case Ctx.Hyperedge:
                this.finishHyperedge();
                return;
            case Ctx.Graph:
                this.graphs.pop();
                return;
            default: {
                const value: never = ctx;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown reader state ${String(value)}`, { name });
            }
        }
    }

    /**
     * Character data.
     * @param text - the text
     * @param line - the line
     */
    text(text: string, line: number): void {
        const ctx = this.ctx.length === 0 ? null : this.ctx[this.ctx.length - 1];
        switch (ctx) {
            case Ctx.Capture:
                this.content.tree?.text(text);
                return;
            case Ctx.Data:
            case Ctx.Default:
                if (this.content.tree === null) {
                    this.content.text += text;
                } else {
                    this.content.tree.text(text);
                }
                return;
            case Ctx.Desc:
                this.descText += text;
                return;
            case Ctx.Skip:
                return;
            default:
                if (!isWhitespace(text)) {
                    this.report.warnOnce(
                        "validation-error",
                        GRAPHML_ISSUE.STRAY_TEXT,
                        "text where GraphML allows only elements was ignored",
                        { line },
                    );
                }
        }
    }

    /**
     * After the last event: the document must have held a graph; the metadata is written.
     */
    finish(): void {
        if (!this.graphSeen) {
            this.report.fail(GRAPHML_ISSUE.NO_GRAPH, "the document has no <graph> element");
        }
        const meta: GraphmlMeta = { graphId: this.graphId, edgedefault: this.edgedefault, namespaces: this.namespaces };
        this.sink.setMeta({
            sourceFormat: FORMAT,
            description: this.description ?? undefined,
            weightOrigin:
                this.weightOrigin === null
                    ? undefined
                    : {
                          format: FORMAT,
                          id: this.weightOrigin.id,
                          title: this.weightOrigin.title,
                          type: this.weightOrigin.type,
                          namespace: null,
                      },
            extra: { [META_KEY]: meta },
        });
    }

    // ------------------------------------------------------------------ structure

    /**
     * Record the root element's namespace declarations.
     * @param attrs - the root attributes
     */
    private readRoot(attrs: ReadonlyMap<string, string>): void {
        for (const [key, value] of attrs) {
            if (key.startsWith("xmlns:") && value !== GRAPHML_NAMESPACE && value !== XSI_NAMESPACE) {
                this.namespaces[key.slice(6)] = value;
            }
        }
    }

    /**
     * A child of `<graphml>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInGraphml(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "desc":
                this.beginDesc("graph");
                return;
            case "key":
                this.beginKey(attrs, line);
                return;
            case "graph":
                this.beginGraph(attrs, line, INVALID_INDEX);
                return;
            case "data":
                this.beginData(attrs, line, "graph");
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<key>`.
     * @param name - the element name
     * @param local - its local name
     * @param line - the line
     */
    private startInKey(name: string, local: string, line: number): void {
        switch (local) {
            case "desc":
                this.beginDesc("key");
                return;
            case "default":
                this.content.key = null;
                this.content.pendingKey = this.pendingKey;
                this.content.text = "";
                this.content.tree = null;
                this.content.line = line;
                this.ctx.push(Ctx.Default);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<graph>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInGraph(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "desc":
                if (this.graphs.length === 1) {
                    this.beginDesc("graph");
                } else {
                    this.ctx.push(Ctx.Skip);
                }
                return;
            case "data":
                if (this.graphs.length === 1) {
                    this.beginData(attrs, line, "graph");
                } else {
                    this.report.warnOnce(
                        "unsupported",
                        GRAPHML_ISSUE.NESTED_GRAPH_DATA,
                        "data of a nested <graph> cannot be kept; only the top-level graph has attributes",
                        { line },
                    );
                    this.ctx.push(Ctx.Skip);
                }
                return;
            case "node":
                this.beginNode(attrs, line);
                return;
            case "edge":
                this.beginEdge(attrs, line);
                return;
            case "hyperedge":
                this.beginHyperedge(attrs, line);
                return;
            case "locator":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.LOCATOR_DROPPED,
                    "<locator> references are not followed",
                    { line },
                );
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<node>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInNode(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "desc":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.DESC_DROPPED,
                    "node and edge <desc> texts are not kept",
                    {
                        line,
                    },
                );
                this.ctx.push(Ctx.Skip);
                return;
            case "data":
                this.beginData(attrs, line, "node");
                return;
            case "port":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.PORT_DECLARATION,
                    "<port> declarations are not kept; sourceport / targetport edge attributes are",
                    { line },
                );
                this.ctx.push(Ctx.Skip);
                return;
            case "graph": {
                const node = this.nodes[this.nodes.length - 1];
                this.materializeNode(node);
                this.beginGraph(attrs, line, node.index);
                return;
            }
            case "locator":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.LOCATOR_DROPPED,
                    "<locator> references are not followed",
                    { line },
                );
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<edge>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInEdge(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "desc":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.DESC_DROPPED,
                    "node and edge <desc> texts are not kept",
                    {
                        line,
                    },
                );
                this.ctx.push(Ctx.Skip);
                return;
            case "data":
                this.beginData(attrs, line, "edge");
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<hyperedge>` (under the star / clique policies).
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInHyperedge(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "endpoint":
                this.readEndpoint(attrs, line);
                this.ctx.push(Ctx.Skip);
                return;
            case "data":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.HYPEREDGE_DATA_DROPPED,
                    "hyperedge data are not kept by the star / clique expansions",
                    { line },
                );
                this.ctx.push(Ctx.Skip);
                return;
            case "desc":
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * Report an element the schema does not define at this place and skip its subtree.
     * @param name - the element name
     * @param line - the line
     */
    private unknownElement(name: string, line: number): void {
        this.report.warnOnce(
            "unsupported",
            GRAPHML_ISSUE.UNKNOWN_ELEMENT,
            `element <${name}> is not GraphML and was skipped`,
            {
                line,
                element: name,
            },
        );
        this.ctx.push(Ctx.Skip);
    }

    /**
     * Open a `<desc>`.
     * @param target - what the text describes
     */
    private beginDesc(target: "graph" | "key"): void {
        this.descTarget = target;
        this.descText = "";
        this.ctx.push(Ctx.Desc);
    }

    /** Close a `<desc>`. */
    private finishDesc(): void {
        const text = this.descText.trim();
        if (this.descTarget === "key" && this.pendingKey !== null) {
            this.pendingKey.desc = text;
        } else if (this.descTarget === "graph" && text.length > 0) {
            this.description = text;
        }
        this.descTarget = null;
        this.descText = "";
    }

    // ------------------------------------------------------------------ graphs

    /**
     * Open a `<graph>`: the top-level one sets the sink's direction (rule 1 of design section
     * 8.4); a nested one records its container as the parent of its nodes.
     * @param attrs - the graph attributes
     * @param line - the line
     * @param parent - the containing node's index, or INVALID_INDEX
     */
    private beginGraph(attrs: ReadonlyMap<string, string>, line: number, parent: number): void {
        const top = this.graphs.length === 0;
        const edgedefault = attrs.get("edgedefault");
        let directed: boolean;
        if (edgedefault === "directed" || edgedefault === "undirected") {
            directed = edgedefault === "directed";
        } else if (edgedefault === undefined) {
            directed = top ? this.options.defaultDirected : this.graphs[this.graphs.length - 1].directed;
            this.report.warning(
                "validation-error",
                GRAPHML_ISSUE.EDGEDEFAULT_MISSING,
                `<graph> has no edgedefault; read as ${directed ? "directed" : "undirected"}`,
                { line },
            );
        } else {
            directed = top ? this.options.defaultDirected : this.graphs[this.graphs.length - 1].directed;
            this.report.error(
                "validation-error",
                GRAPHML_ISSUE.INVALID_EDGEDEFAULT,
                `edgedefault "${edgedefault}" is neither directed nor undirected; read as ${directed ? "directed" : "undirected"}`,
                { line },
            );
        }
        if (top && this.graphSeen) {
            this.report.warnOnce(
                "unsupported",
                GRAPHML_ISSUE.MULTIPLE_GRAPHS,
                "the document holds more than one top-level <graph>; their nodes and edges are merged into one graph",
                { line, element: attrs.get("id") ?? null },
            );
        } else if (top) {
            this.graphSeen = true;
            this.graphId = attrs.get("id") ?? null;
            this.edgedefault = directed ? "directed" : "undirected";
            this.direction.setHeader(directed, { line });
            const nodes = this.hint(attrs.get("parse.nodes"), "parse.nodes", line);
            const edges = this.hint(attrs.get("parse.edges"), "parse.edges", line);
            if (nodes !== null || edges !== null) {
                try {
                    this.sink.reserve(nodes ?? undefined, edges ?? undefined);
                } catch (err) {
                    if (!(err instanceof GraphFormatError)) {
                        throw err;
                    }
                    this.report.warning(
                        "validation-error",
                        GRAPHML_ISSUE.COUNT_HINT,
                        `count hint refused: ${err.message}`,
                        {
                            line,
                        },
                    );
                }
            }
        }
        this.graphs.push({ directed, parent });
        this.ctx.push(Ctx.Graph);
    }

    /**
     * A `parse.nodes` / `parse.edges` hint the sink can reserve; anything else is reported and ignored.
     * @param text - the attribute text, or undefined
     * @param name - the attribute name
     * @param line - the line
     * @returns the count, or null
     */
    private hint(text: string | undefined, name: string, line: number): number | null {
        if (text === undefined) {
            return null;
        }
        const n = hint(text);
        if (n === null || n > MAX_COUNT) {
            this.report.warning(
                "validation-error",
                GRAPHML_ISSUE.COUNT_HINT,
                `<graph ${name}="${text}"> is not a count the sink can reserve; ignored`,
                { line, element: name },
            );
            return null;
        }
        return n;
    }

    // ------------------------------------------------------------------ keys

    /**
     * Open a `<key>`; the column is declared at `</key>` once its default is known.
     * @param attrs - the key attributes
     * @param line - the line
     */
    private beginKey(attrs: ReadonlyMap<string, string>, line: number): void {
        const id = attrs.get("id") ?? null;
        const forText = attrs.get("for") ?? "all";
        let domains: readonly Domain[];
        switch (forText) {
            case "all":
                domains = ["graph", "node", "edge"];
                break;
            case "graph":
            case "node":
            case "edge":
                domains = [forText];
                break;
            case "hyperedge":
            case "port":
            case "endpoint":
                this.report.warning(
                    "unsupported",
                    GRAPHML_ISSUE.KEY_DOMAIN_UNSUPPORTED,
                    `key "${id ?? ""}" is declared for ${forText} elements, which are not kept`,
                    { line, element: id },
                );
                domains = [];
                break;
            default:
                this.report.error(
                    "validation-error",
                    GRAPHML_ISSUE.KEY_FOR_INVALID,
                    `key "${id ?? ""}" has an unknown for="${forText}"`,
                    { line, element: id },
                );
                domains = [];
        }
        this.pendingKey = {
            id,
            attrName: attrs.get("attr.name") ?? null,
            attrType: attrs.get("attr.type") ?? null,
            yfilesType: attrs.get("yfiles.type") ?? null,
            domains,
            line,
            desc: "",
            defaultText: null,
            defaultTree: undefined,
        };
        this.ctx.push(Ctx.Key);
    }

    /** Close a `<key>`: declare its columns. */
    private finishKey(): void {
        const key = this.pendingKey;
        this.pendingKey = null;
        if (key === null) {
            return;
        }
        const where = { line: key.line, element: key.id };
        if (key.id === null || key.id.length === 0) {
            this.report.error("validation-error", GRAPHML_ISSUE.KEY_MISSING_ID, "<key> without an id", where);
            return;
        }
        if (this.keys.has(key.id)) {
            this.report.error(
                "validation-error",
                GRAPHML_ISSUE.DUPLICATE_KEY,
                `key "${key.id}" is declared twice`,
                where,
            );
            return;
        }
        const yfiles = key.yfilesType !== null;
        const name = key.attrName !== null && key.attrName.length > 0 ? key.attrName : key.id;
        const skipped = yfiles && this.yfilesMode === "skip";
        if (skipped) {
            this.report.warning(
                "unsupported",
                GRAPHML_ISSUE.YFILES_SKIPPED,
                `yFiles key "${key.id}" (${key.yfilesType ?? ""}) skipped per the yfiles option`,
                where,
            );
        }
        let weight = false;
        let originalId = false;
        let node: ColumnTarget | null = null;
        let edge: ColumnTarget | null = null;
        let graph: ColumnTarget | null = null;
        for (const domain of key.domains) {
            if (skipped) {
                continue;
            }
            if (domain === "edge" && !yfiles && isWeightField(name, this.options.weightFrom)) {
                weight = true;
                this.weightOrigin ??= { id: key.id, title: key.attrName, type: key.attrType };
                continue;
            }
            if (domain === "node" && !yfiles && name === ORIGINAL_ID_ATTRIBUTE) {
                if (this.options.restoreMangledIds) {
                    originalId = true;
                    continue;
                }
                node = this.declareOriginalIdColumn(key);
                continue;
            }
            try {
                const target = yfiles ? this.declareYfiles(key, name, domain) : this.declareTyped(key, name, domain);
                if (domain === "node") {
                    node = target;
                } else if (domain === "edge") {
                    edge = target;
                } else {
                    graph = target;
                }
            } catch (err) {
                this.report.recordError(err, where);
            }
        }
        this.keys.set(key.id, {
            id: key.id,
            name,
            node,
            edge,
            graph,
            weight,
            originalId,
            yfiles,
            skipped,
            domains: key.domains,
        });
    }

    /**
     * Declare the column of a typed (or untyped, then string) key in one domain.
     * @param key - the key
     * @param name - the resolved attribute name
     * @param domain - the domain
     * @returns the column target
     */
    private declareTyped(key: PendingKey, name: string, domain: Domain): ColumnTarget {
        const id = key.id ?? "";
        const declared: DeclaredAttribute = declareAttribute({
            format: FORMAT,
            id,
            title: key.attrName !== null && key.attrName.length > 0 ? key.attrName : null,
            type: key.attrType,
            defaultText: key.defaultTree === undefined ? key.defaultText : null,
            listSyntax: "comma",
            role: domain !== "graph" && name === LABEL_COLUMN ? "label" : null,
            long: this.options.long,
            taken: this.takenIn(domain),
        });
        for (const issue of declared.issues) {
            this.report.warning(issue.category, issue.code, issue.message, { line: key.line, element: id });
        }
        if (key.defaultTree !== undefined) {
            this.report.warning(
                "validation-error",
                GRAPHML_ISSUE.DATA_NESTED,
                `the <default> of key "${id}" holds elements; a ${key.attrType ?? "string"} default must be text`,
                { line: key.line, element: id },
            );
        }
        const { decl } = declared;
        if (key.desc.length > 0) {
            decl.extra = { desc: key.desc };
        }
        return this.declareTarget(domain, decl, declared.spec, declared.listSyntax);
    }

    /**
     * Declare the json column of a yFiles key in one domain.
     * @param key - the key
     * @param name - the resolved attribute name
     * @param domain - the domain
     * @returns the column target
     */
    private declareYfiles(key: PendingKey, name: string, domain: Domain): ColumnTarget {
        const id = key.id ?? "";
        const columnName = uniqueColumnName(name, id, this.takenIn(domain));
        const decl: ColumnDecl = {
            name: columnName,
            dtype: "json",
            nullable: true,
            origin: {
                format: FORMAT,
                id,
                title: key.attrName !== null && key.attrName !== columnName ? key.attrName : null,
                type: key.yfilesType,
                namespace: "yfiles",
            },
        };
        if (key.defaultTree !== undefined) {
            decl.default = key.defaultTree;
        } else if (key.defaultText !== null) {
            decl.default = key.defaultText;
        }
        if (key.desc.length > 0) {
            decl.extra = { desc: key.desc };
        }
        return this.declareTarget(domain, decl, null, "comma");
    }

    /**
     * Declare the `graphty.originalId` column that keeps the exporter's original ids when
     * restoreMangledIds is off.
     * @param key - the key
     * @returns the column target
     */
    private declareOriginalIdColumn(key: PendingKey): ColumnTarget {
        const decl: ColumnDecl = {
            name: uniqueColumnName(ORIGINAL_ID_COLUMN, key.id, this.takenIn("node")),
            dtype: "string",
            nullable: true,
            role: "originalId",
            origin: { format: FORMAT, id: key.id, title: key.attrName, type: key.attrType, namespace: null },
        };
        return this.declareTarget("node", decl, null, "comma");
    }

    /**
     * Push a declaration into the sink for one domain through the shared design section 5.6 rule:
     * a role another column already holds is dropped (reported), a name the sink holds with
     * another shape is renamed `<name>#<id>` (reported).
     * @param domain - the domain
     * @param decl - the declaration
     * @param spec - the value spec, or null for json
     * @param listSyntax - the list syntax
     * @returns the target
     */
    private declareTarget(
        domain: Domain,
        decl: ColumnDecl,
        spec: DeclaredTypeSpec | null,
        listSyntax: ListSyntax,
    ): ColumnTarget {
        if (domain === "graph") {
            this.sink.setGraphValue(decl.name, undefined, decl);
            this.graphColumnNames.add(decl.name);
            return { name: decl.name, handle: INVALID_INDEX as ColumnHandle, decl, spec, listSyntax };
        }
        const resolved = declareResolved(this.sink, domain, decl, this.report, { element: decl.name });
        return { name: resolved.decl.name, handle: resolved.handle, decl: resolved.decl, spec, listSyntax };
    }

    /**
     * The `taken` predicate of a domain: the reserved XML-derived names plus the sink's columns.
     * @param domain - the domain
     * @returns the predicate
     */
    private takenIn(domain: Domain): (name: string) => boolean {
        if (domain === "graph") {
            return (name: string): boolean => this.graphColumnNames.has(name);
        }
        const reserved = domain === "node" ? RESERVED_NODE_NAMES : RESERVED_EDGE_NAMES;
        const inSink = takenIn(this.sink, domain);
        return (name: string): boolean => reserved.has(name) || inSink(name);
    }

    // ------------------------------------------------------------------ data and defaults

    /**
     * Open a `<data>`.
     * @param attrs - its attributes
     * @param line - the line
     * @param domain - the domain of the element holding it
     */
    private beginData(attrs: ReadonlyMap<string, string>, line: number, domain: Domain): void {
        const keyId = attrs.get("key");
        const { content } = this;
        content.key = null;
        content.pendingKey = null;
        content.domain = domain;
        content.text = "";
        content.tree = null;
        content.line = line;
        if (keyId === undefined) {
            this.report.error("validation-error", GRAPHML_ISSUE.DATA_MISSING_KEY, "<data> without a key attribute", {
                line,
            });
        } else {
            const key = this.keys.get(keyId);
            if (key === undefined) {
                this.report.error(
                    "validation-error",
                    GRAPHML_ISSUE.UNKNOWN_KEY,
                    `<data> references undeclared key "${keyId}"`,
                    { line, element: keyId },
                );
            } else if (!key.skipped && !key.domains.includes(domain)) {
                this.report.warnOnce(
                    "validation-error",
                    GRAPHML_ISSUE.KEY_DOMAIN,
                    `key "${keyId}" is declared for ${key.domains.join(" / ")} elements but used on a ${domain}; ignored`,
                    { line, element: keyId },
                );
            } else {
                content.key = key;
            }
        }
        this.ctx.push(Ctx.Data);
    }

    /**
     * A child element inside `<data>` / `<default>`: the content becomes a tree.
     * @param name - the element name
     * @param attrs - its attributes
     */
    private captureStart(name: string, attrs: ReadonlyMap<string, string>): void {
        const { content } = this;
        if (content.tree === null) {
            content.tree = new XmlTreeBuilder();
            if (content.text.length > 0) {
                content.tree.text(content.text);
                content.text = "";
            }
        }
        content.tree.start(name, attrs);
        this.ctx.push(Ctx.Capture);
    }

    /** A captured element ends. */
    private captureEnd(): void {
        this.content.tree?.end();
    }

    /** Close a `<default>`: keep its text or tree for the key. */
    private finishDefault(): void {
        const { content } = this;
        const key = content.pendingKey;
        if (key === null) {
            return;
        }
        if (content.tree !== null) {
            key.defaultTree = content.tree.finish();
        } else {
            key.defaultText = content.text;
        }
        content.tree = null;
        content.text = "";
        content.pendingKey = null;
    }

    /** Close a `<data>`: parse the value and route it to its element. */
    private finishData(): void {
        const { content } = this;
        const { key } = content;
        const value: unknown = content.tree === null ? content.text : content.tree.finish();
        content.tree = null;
        content.text = "";
        content.key = null;
        if (key === null || key.skipped) {
            return;
        }
        const where = { line: content.line, element: key.id };
        switch (content.domain) {
            case "node": {
                const node = this.nodes[this.nodes.length - 1];
                if (key.originalId) {
                    if (typeof value === "string") {
                        node.restoredId = value;
                    }
                    return;
                }
                if (key.node === null) {
                    return;
                }
                const parsed = this.parseValue(key.node, value, where);
                if (parsed === undefined) {
                    return;
                }
                if (node.index === INVALID_INDEX) {
                    node.pending.push(key.node.handle, parsed);
                } else if (!node.failed) {
                    this.sink.setNodeValue(key.node.handle, node.index, parsed);
                }
                return;
            }
            case "edge": {
                const { edge } = this;
                if (edge === null) {
                    return;
                }
                if (key.weight) {
                    if (typeof value !== "string") {
                        this.report.error(
                            "validation-error",
                            GRAPHML_ISSUE.DATA_NESTED,
                            "a weight value must be text",
                            where,
                        );
                        return;
                    }
                    try {
                        edge.weight = parseWeightText(value);
                    } catch (err) {
                        this.skipEdge(edge);
                        this.report.recordError(err, where);
                    }
                    return;
                }
                if (key.edge === null) {
                    return;
                }
                const parsed = this.parseValue(key.edge, value, where);
                if (parsed !== undefined) {
                    edge.pending.push(key.edge.handle, parsed);
                }
                return;
            }
            case "graph": {
                if (key.graph === null) {
                    return;
                }
                const parsed = this.parseValue(key.graph, value, where);
                if (parsed !== undefined) {
                    try {
                        this.sink.setGraphValue(key.graph.name, parsed, key.graph.decl);
                    } catch (err) {
                        this.report.recordError(err, where);
                    }
                }
                return;
            }
            default: {
                const { domain } = content;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown data domain ${String(domain)}`, {});
            }
        }
    }

    /**
     * Parse the content of a `<data>` by its column's declaration.
     * @param target - the column
     * @param value - the text, or the tree of a nested subtree
     * @param where - the location for issues
     * @returns the value to write, or undefined when there is none (blank, or an error was recorded)
     */
    private parseValue(target: ColumnTarget, value: unknown, where: Where): unknown {
        if (target.spec === null) {
            if (!this.yfilesLossNoted) {
                this.yfilesLossNoted = true;
                this.report.loss(
                    GRAPHML_LOSS.YFILES_JSON,
                    "yFiles nested XML is kept as a JSON tree: structure preserved, not byte-exact",
                    target.name,
                );
            }
            return value;
        }
        if (typeof value !== "string") {
            this.report.error(
                "validation-error",
                GRAPHML_ISSUE.DATA_NESTED,
                `<data> of the ${target.spec.declared} attribute "${target.name}" holds elements, not text`,
                where,
            );
            return undefined;
        }
        const { spec } = target;
        if (spec.kind !== "string" && isWhitespace(value)) {
            return undefined;
        }
        try {
            const parsed = parseDeclaredValue(value, spec, target.listSyntax);
            if (spec.precision && losesPrecision(spec, value)) {
                this.report.warnOnce(
                    "precision",
                    PRECISION_CODE,
                    `long values of "${target.name}" beyond 2^53 are stored as the nearest double`,
                    { line: where.line, element: target.name },
                );
            }
            return parsed;
        } catch (err) {
            this.report.recordError(err, where);
            return undefined;
        }
    }

    // ------------------------------------------------------------------ nodes

    /**
     * Open a `<node>`; it is added to the sink when its data is complete (or a nested graph
     * starts), so a `graphty:originalId` value can replace its id.
     * @param attrs - the node attributes
     * @param line - the line
     */
    private beginNode(attrs: ReadonlyMap<string, string>, line: number): void {
        const graph = this.graphs[this.graphs.length - 1];
        const state: NodeState = {
            id: null,
            idText: null,
            index: INVALID_INDEX,
            parent: graph.parent,
            pending: [],
            restoredId: null,
            failed: false,
            line,
        };
        const idText = attrs.get("id");
        if (idText === undefined || idText.length === 0) {
            this.skipNode(state);
            this.report.error("missing-value", GRAPHML_ISSUE.MISSING_ID, "<node> without an id", { line });
        } else {
            try {
                state.id = this.coercer.text(idText);
                state.idText = idText;
                this.reportMerge(idText, line);
            } catch (err) {
                this.skipNode(state);
                this.report.recordError(err, { line, element: idText });
            }
        }
        this.nodes.push(state);
        this.ctx.push(Ctx.Node);
    }

    /**
     * Add a node to the sink and flush its pending values.
     * @param node - the node state
     */
    private materializeNode(node: NodeState): void {
        if (node.index !== INVALID_INDEX || node.failed) {
            return;
        }
        const id: NodeId | null = node.restoredId ?? node.id;
        if (id === null) {
            node.failed = true;
            return;
        }
        if (node.restoredId !== null && node.idText !== null) {
            this.restored.set(node.idText, node.restoredId);
        }
        try {
            const existing = this.sink.indexOf(id);
            const index = this.sink.addNode(id);
            if (existing !== INVALID_INDEX && this.declared.has(index)) {
                this.report.warning(
                    "validation-error",
                    GRAPHML_ISSUE.DUPLICATE_NODE,
                    `node "${String(id)}" is declared twice; the declarations are merged`,
                    { line: node.line, element: String(id) },
                );
            } else {
                this.report.counts.nodes++;
            }
            this.declared.add(index);
            node.index = index;
            this.checkAbort();
            if (node.parent !== INVALID_INDEX) {
                this.sink.setNodeValue(this.parentColumn(), index, node.parent);
            }
            const { pending } = node;
            for (let i = 0; i < pending.length; i += 2) {
                this.sink.setNodeValue(pending[i] as ColumnHandle, index, pending[i + 1]);
            }
            pending.length = 0;
        } catch (err) {
            this.skipNode(node);
            this.report.recordError(err, { line: node.line, element: String(id) });
        }
    }

    /**
     * Mark a node element skipped (once) before its error is recorded, so the report's counts
     * include it even when that error is the one that exceeds the limit.
     * @param node - the node state
     */
    private skipNode(node: NodeState): void {
        if (!node.failed) {
            node.failed = true;
            this.report.counts.skippedNodes++;
        }
    }

    /** Close a `<node>`. */
    private finishNode(): void {
        const node = this.nodes.pop();
        if (node !== undefined) {
            this.materializeNode(node);
        }
    }

    /**
     * The `parent` column, declared on first use.
     * @returns the handle
     */
    private parentColumn(): ColumnHandle {
        if (this.parentHandle === INVALID_INDEX) {
            this.parentHandle = this.sink.declareNodeColumn({
                name: PARENT_COLUMN,
                dtype: "u32",
                role: "parent",
                refersTo: "node",
                nullable: true,
                origin: { format: FORMAT, id: null, title: null, type: "graph", namespace: null },
            });
        }
        return this.parentHandle;
    }

    // ------------------------------------------------------------------ edges

    /**
     * Open an `<edge>`; it is pushed at `</edge>` once its weight and data are known.
     * @param attrs - the edge attributes
     * @param line - the line
     */
    private beginEdge(attrs: ReadonlyMap<string, string>, line: number): void {
        const graph = this.graphs[this.graphs.length - 1];
        const edge: EdgeState = {
            source: null,
            target: null,
            id: attrs.get("id") ?? null,
            kind: graph.directed ? "directed" : "undirected",
            sourcePort: attrs.get("sourceport") ?? null,
            targetPort: attrs.get("targetport") ?? null,
            weight: undefined,
            pending: [],
            failed: false,
            line,
        };
        const where = { line, element: edge.id };
        const sourceText = attrs.get("source");
        const targetText = attrs.get("target");
        if (sourceText === undefined || sourceText.length === 0) {
            this.skipEdge(edge);
            this.report.error("missing-value", GRAPHML_ISSUE.MISSING_ENDPOINT, "<edge> without a source", where);
        }
        if (targetText === undefined || targetText.length === 0) {
            this.skipEdge(edge);
            this.report.error("missing-value", GRAPHML_ISSUE.MISSING_ENDPOINT, "<edge> without a target", where);
        }
        if (!edge.failed && sourceText !== undefined && targetText !== undefined) {
            try {
                edge.source = this.endpointId(sourceText, line);
                edge.target = this.endpointId(targetText, line);
            } catch (err) {
                this.skipEdge(edge);
                this.report.recordError(err, where);
            }
        }
        const directedText = attrs.get("directed");
        if (directedText !== undefined) {
            if (directedText === "true") {
                edge.kind = "directed";
            } else if (directedText === "false") {
                edge.kind = "undirected";
            } else {
                this.skipEdge(edge);
                this.report.error(
                    "validation-error",
                    GRAPHML_ISSUE.INVALID_DIRECTED,
                    `directed="${directedText}" is neither true nor false`,
                    where,
                );
            }
        }
        this.edge = edge;
        this.ctx.push(Ctx.Edge);
    }

    /**
     * Close an `<edge>`: push it through the direction resolver and write its values.
     */
    private finishEdge(): void {
        const { edge } = this;
        this.edge = null;
        if (edge === null) {
            return;
        }
        if (edge.failed || edge.source === null || edge.target === null) {
            this.skipEdge(edge);
            return;
        }
        const element = edge.id ?? `${String(edge.source)} -> ${String(edge.target)}`;
        const where = { line: edge.line, element };
        try {
            if (!this.options.addMissingNodes) {
                this.requireNode(edge.source);
                this.requireNode(edge.target);
            }
            if (edge.id !== null && this.edgeIds.has(edge.id)) {
                throw new GraphFormatError("E_DUPLICATE_EDGE_ID", `edge id "${edge.id}" is used twice`, {
                    id: edge.id,
                });
            }
            const before = this.sink.edgeCount;
            this.countEndpoints(edge.source, edge.target);
            const e = this.direction.addEdge(edge.source, edge.target, edge.kind, edge.weight, where);
            this.report.counts.edges += this.sink.edgeCount - before;
            this.checkAbort();
            if (edge.id !== null) {
                this.edgeIds.add(edge.id);
                this.sink.setEdgeValue(this.edgeIdColumn(), e, edge.id);
            }
            if (edge.sourcePort !== null) {
                this.sink.setEdgeValue(this.portColumn("source"), e, edge.sourcePort);
            }
            if (edge.targetPort !== null) {
                this.sink.setEdgeValue(this.portColumn("target"), e, edge.targetPort);
            }
            const { pending } = edge;
            for (let i = 0; i < pending.length; i += 2) {
                this.sink.setEdgeValue(pending[i] as ColumnHandle, e, pending[i + 1]);
            }
        } catch (err) {
            this.skipEdge(edge);
            this.report.recordError(err, where);
        }
    }

    /**
     * Mark an edge (or hyperedge) element skipped once, before its error is recorded.
     * @param edge - the edge state
     */
    private skipEdge(edge: EdgeState | HyperedgeState): void {
        if (!edge.failed) {
            edge.failed = true;
            this.report.counts.skippedEdges++;
        }
    }

    /**
     * E_UNKNOWN_NODE when an endpoint was never declared (addMissingNodes false enforced by the
     * importer, whatever the sink allows).
     * @param id - the endpoint id
     */
    private requireNode(id: NodeId): void {
        if (this.sink.indexOf(id) === INVALID_INDEX) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `edge endpoint "${String(id)}" is not a declared node`, {
                id,
            });
        }
    }

    /**
     * The edge `id` column (role id, unique), declared on first use.
     * @returns the handle
     */
    private edgeIdColumn(): ColumnHandle {
        if (this.edgeIdHandle === INVALID_INDEX) {
            this.edgeIdHandle = this.sink.declareEdgeColumn({
                name: EDGE_ID_COLUMN,
                dtype: "string",
                role: "id",
                unique: true,
                nullable: true,
                origin: { format: FORMAT, id: null, title: null, type: "id", namespace: null },
            });
        }
        return this.edgeIdHandle;
    }

    /**
     * The sourceport / targetport column, declared on first use.
     * @param side - which end
     * @returns the handle
     */
    private portColumn(side: "source" | "target"): ColumnHandle {
        if (side === "source") {
            if (this.sourcePortHandle === INVALID_INDEX) {
                this.sourcePortHandle = this.sink.declareEdgeColumn({
                    name: SOURCE_PORT_COLUMN,
                    dtype: "string",
                    role: "sourcePort",
                    nullable: true,
                    origin: { format: FORMAT, id: null, title: null, type: "sourceport", namespace: null },
                });
            }
            return this.sourcePortHandle;
        }
        if (this.targetPortHandle === INVALID_INDEX) {
            this.targetPortHandle = this.sink.declareEdgeColumn({
                name: TARGET_PORT_COLUMN,
                dtype: "string",
                role: "targetPort",
                nullable: true,
                origin: { format: FORMAT, id: null, title: null, type: "targetport", namespace: null },
            });
        }
        return this.targetPortHandle;
    }

    // ------------------------------------------------------------------ hyperedges

    /**
     * Open a `<hyperedge>` per the hyperedges option.
     * @param attrs - its attributes
     * @param line - the line
     */
    private beginHyperedge(attrs: ReadonlyMap<string, string>, line: number): void {
        const id = attrs.get("id") ?? null;
        const where = { line, element: id };
        switch (this.options.hyperedges) {
            case "error":
                this.report.error(
                    "unsupported",
                    GRAPHML_ISSUE.HYPEREDGE,
                    'hyperedge refused (hyperedges: "error")',
                    where,
                );
                throw this.report.abort("hyperedge refused", { code: GRAPHML_ISSUE.HYPEREDGE });
            case "skip":
                this.report.warnOnce(
                    "unsupported",
                    GRAPHML_ISSUE.HYPEREDGE_SKIPPED,
                    'hyperedges are skipped (hyperedges: "skip"); use "star" or "clique" to expand them',
                    where,
                );
                this.report.counts.skippedEdges++;
                this.ctx.push(Ctx.Skip);
                return;
            case "star":
            case "clique":
                this.hyperedge = { id, endpoints: [], failed: false, line };
                this.ctx.push(Ctx.Hyperedge);
                return;
            default: {
                const policy: never = this.options.hyperedges;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown hyperedges policy ${String(policy)}`, {
                    option: "hyperedges",
                });
            }
        }
    }

    /**
     * An `<endpoint>` of the open hyperedge.
     * @param attrs - its attributes
     * @param line - the line
     */
    private readEndpoint(attrs: ReadonlyMap<string, string>, line: number): void {
        const { hyperedge } = this;
        if (hyperedge === null) {
            return;
        }
        const where = { line, element: hyperedge.id };
        const nodeText = attrs.get("node");
        const typeText = attrs.get("type") ?? "undir";
        if (nodeText === undefined || nodeText.length === 0) {
            this.skipEdge(hyperedge);
            this.report.error("missing-value", GRAPHML_ISSUE.HYPEREDGE_ENDPOINT, "<endpoint> without a node", where);
            return;
        }
        if (typeText !== "in" && typeText !== "out" && typeText !== "undir") {
            this.skipEdge(hyperedge);
            this.report.error(
                "validation-error",
                GRAPHML_ISSUE.HYPEREDGE_ENDPOINT,
                `endpoint type "${typeText}" is not in, out or undir`,
                where,
            );
            return;
        }
        try {
            const node = this.endpointId(nodeText, line);
            hyperedge.endpoints.push({ node, type: typeText });
        } catch (err) {
            this.skipEdge(hyperedge);
            this.report.recordError(err, where);
        }
    }

    /** Close a `<hyperedge>`: expand it to a star or a clique. */
    private finishHyperedge(): void {
        const { hyperedge } = this;
        this.hyperedge = null;
        if (hyperedge === null) {
            return;
        }
        if (hyperedge.failed) {
            return;
        }
        const where = { line: hyperedge.line, element: hyperedge.id };
        try {
            if (this.options.hyperedges === "star") {
                this.expandStar(hyperedge, where);
            } else {
                this.expandClique(hyperedge, where);
            }
        } catch (err) {
            this.skipEdge(hyperedge);
            this.report.recordError(err, where);
        }
    }

    /**
     * Star expansion: a synthetic hub node (marked in the `graphty.hyperedge` column) joined to
     * every endpoint; `in` endpoints point at the hub, `out` endpoints are pointed at by it,
     * `undir` endpoints are undirected.
     * @param hyperedge - the hyperedge
     * @param where - the location for issues
     */
    private expandStar(hyperedge: HyperedgeState, where: Where): void {
        this.hyperedgeCount++;
        let hubId = hyperedge.id ?? `hyperedge${this.hyperedgeCount}`;
        for (let n = 2; this.sink.indexOf(hubId) !== INVALID_INDEX; n++) {
            hubId = `${hyperedge.id ?? `hyperedge${this.hyperedgeCount}`}#${n}`;
        }
        const hub = this.sink.addNode(hubId);
        this.declared.add(hub);
        this.report.counts.nodes++;
        if (this.hubHandle === INVALID_INDEX) {
            this.hubHandle = this.sink.declareNodeColumn({
                name: HYPEREDGE_HUB_COLUMN,
                dtype: "bool",
                nullable: true,
                origin: { format: FORMAT, id: null, title: null, type: "hyperedge", namespace: null },
            });
        }
        this.sink.setNodeValue(this.hubHandle, hub, true);
        for (const endpoint of hyperedge.endpoints) {
            const before = this.sink.edgeCount;
            switch (endpoint.type) {
                case "in":
                    this.direction.addEdge(endpoint.node, hubId, "directed", undefined, where);
                    break;
                case "out":
                    this.direction.addEdge(hubId, endpoint.node, "directed", undefined, where);
                    break;
                case "undir":
                    this.direction.addEdge(endpoint.node, hubId, "undirected", undefined, where);
                    break;
                default: {
                    const { type } = endpoint;
                    throw new GraphFormatError("E_UNSUPPORTED", `unknown endpoint type ${String(type)}`, {});
                }
            }
            this.report.counts.edges += this.sink.edgeCount - before;
        }
    }

    /**
     * Clique expansion: every `in` endpoint points at every `out` endpoint; `undir` endpoints are
     * joined undirected to every other endpoint; two endpoints of the same direction are not joined.
     * @param hyperedge - the hyperedge
     * @param where - the location for issues
     */
    private expandClique(hyperedge: HyperedgeState, where: Where): void {
        const { endpoints } = hyperedge;
        for (let i = 0; i < endpoints.length; i++) {
            for (let j = i + 1; j < endpoints.length; j++) {
                const a = endpoints[i];
                const b = endpoints[j];
                const before = this.sink.edgeCount;
                if (a.type === "undir" || b.type === "undir") {
                    this.direction.addEdge(a.node, b.node, "undirected", undefined, where);
                } else if (a.type === "in" && b.type === "out") {
                    this.direction.addEdge(a.node, b.node, "directed", undefined, where);
                } else if (a.type === "out" && b.type === "in") {
                    this.direction.addEdge(b.node, a.node, "directed", undefined, where);
                } else {
                    continue;
                }
                this.report.counts.edges += this.sink.edgeCount - before;
            }
        }
    }

    // ------------------------------------------------------------------ options and ids

    /**
     * Count the endpoints of an edge the file never declared as nodes of the sink (created under
     * addMissingNodes), so the report describes what the sink holds (design section 8.6).
     * @param source - the source id
     * @param target - the target id
     */
    private countEndpoints(source: NodeId, target: NodeId): void {
        if (!this.options.addMissingNodes) {
            return;
        }
        let created = 0;
        if (this.sink.indexOf(source) === INVALID_INDEX) {
            created++;
        }
        if (source !== target && this.sink.indexOf(target) === INVALID_INDEX) {
            created++;
        }
        this.report.counts.nodes += created;
    }

    /** Check the cancellation signal every ABORT_CHECK_INTERVAL elements. */
    private checkAbort(): void {
        if (++this.elementsSinceCheck >= ABORT_CHECK_INTERVAL) {
            this.elementsSinceCheck = 0;
            throwIfAborted(this.options.signal);
        }
    }

    /**
     * The id an edge endpoint refers to: the original id when the text is a mangled id restored
     * from `graphty:originalId` (the exporter writes endpoints with the mangled id), else the
     * coerced text.
     * @param text - the source / target / endpoint node text
     * @param line - the line
     * @returns the node id
     */
    private endpointId(text: string, line: number): NodeId {
        const restored = this.restored.get(text);
        if (restored !== undefined) {
            return restored;
        }
        const id = this.coercer.text(text);
        this.reportMerge(text, line);
        return id;
    }

    /**
     * Report a merge the coercer detected under ids: "number".
     * @param text - the id text just coerced
     * @param line - the line
     */
    private reportMerge(text: string, line: number): void {
        const merge = this.coercer.lastMerge;
        if (merge !== null) {
            this.report.warning(
                "coercion",
                GRAPHML_ISSUE.ID_MERGED,
                `id "${merge.text}" merged with "${merge.previousText}" as ${merge.id} under ids: "number"`,
                { line, element: text },
            );
        }
    }
}

/**
 * A `parse.nodes` / `parse.edges` hint as a count.
 * @param text - the attribute text
 * @returns the count, or null when absent or not a non-negative integer
 */
function hint(text: string | undefined): number | null {
    if (text === undefined || !/^[0-9]+$/.test(text)) {
        return null;
    }
    const n = Number(text);
    return Number.isSafeInteger(n) ? n : null;
}

/**
 * Resolve the format-specific options.
 * @param options - the caller's options
 * @returns the yfiles mode
 */
function resolveYfiles(options: GraphmlImportOptions | undefined): "json" | "skip" {
    const value = options?.yfiles;
    if (value === undefined) {
        return "json";
    }
    if (typeof value !== "string" || !YFILES_MODES.has(value)) {
        throw new GraphFormatError("E_UNSUPPORTED", `option yfiles: ${JSON.stringify(value)} is not "json" or "skip"`, {
            option: "yfiles",
            found: value,
        });
    }
    return value;
}

/**
 * Confidence that a head of bytes is GraphML.
 * @param head - the first bytes
 * @returns 1 for a `<graphml` root in the GraphML namespace, 0.9 for a `<graphml` root, 0.05 for other XML, 0 otherwise
 */
function sniffGraphml(head: Uint8Array): number {
    const text = new TextDecoder("utf-8", { fatal: false }).decode(head.subarray(0, SNIFF_BYTES));
    const root = text.indexOf("<graphml");
    if (root >= 0) {
        return text.includes(GRAPHML_NAMESPACE, root) ? 1 : 0.9;
    }
    return /^\s*<(\?xml|!--|!DOCTYPE)/.test(text) ? 0.05 : 0;
}

/**
 * Import GraphML into a sink.
 * @param input - the document
 * @param sink - the sink
 * @param options - format-specific and common options
 * @returns the report; ImportError when the document cannot be read or the error limit is exceeded
 */
async function importGraphml(
    input: ImportInput,
    sink: GraphSink,
    options?: GraphmlImportOptions & CommonImportOptions,
): Promise<ImportReport> {
    const common = resolveImportOptions(options, FORMAT_DEFAULTS);
    const yfilesMode = resolveYfiles(options);
    const report = new ImportReportBuilder(FORMAT, common.errorLimit);
    reportSinkOptions(sink, options, report, true);
    reportUnusedOptions(options, report, USED_OPTIONS);
    const reader = new GraphmlReader(sink, report, common, yfilesMode);
    try {
        await tokenizeXml(textChunks(input, report, common), reader);
    } catch (err) {
        if (err instanceof XmlSyntaxError) {
            report.fail(GRAPHML_ISSUE.XML_SYNTAX, err.message, { line: err.line });
        }
        throw err;
    }
    reader.finish();
    throwIfAborted(common.signal);
    return report.finish();
}

/** The GraphML importer (design section 8.4). */
export const graphmlImporter: GraphImporter<GraphmlImportOptions> = Object.freeze({
    format: FORMAT,
    extensions: EXTENSIONS,
    mimeTypes: MIME_TYPES,
    sniff: sniffGraphml,
    import: importGraphml,
});
