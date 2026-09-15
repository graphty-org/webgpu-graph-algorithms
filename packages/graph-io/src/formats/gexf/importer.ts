/**
 * The GEXF importer (design sections 8.4, 8.6, 5.1, 5.6, 5.10; research note 07 section 2.1):
 * GEXF 1.1, 1.2 and 1.3 documents read in one SAX-style pass over the shared streaming XML
 * tokenizer (src/common/xml.ts) and pushed into the sink one scalar at a time, so a document of
 * any size is never held in memory. Declared attributes become typed columns as they are read
 * (defaults, options, list types, temporal types with their text companions declared on the first
 * value whose text is not canonical); the viz namespace becomes the position / color / size /
 * shape / thickness role columns with origin.namespace "viz"; `pid`, nested `<nodes>` and
 * `<parents>` become the parent / parents columns, resolved once every node is declared so a
 * forward reference is legal; element lifetimes (`start` / `end` / `timestamp`, `<spells>`, 1.3
 * `timestamps` / `intervals`, 1.2 `startopen` / `endopen` holding the open bound's time) become
 * the temporal role columns; dynamic attribute values go into the temporal extension tables of
 * design section 5.10; the graph header's `mode`, `timeformat`, `timerepresentation` and `idtype`
 * land in the graph meta (`idtype` is informational: Gephi writes `idtype="string"` for every
 * file, so ids follow the `ids` rule, canonical by default).
 *
 * Direction follows design section 8.4: `defaultedgetype` (undirected when absent, per the spec and
 * the `defaultDirected` option) sets the sink's direction once, and every per-edge `type` override,
 * `mutual` included, goes through the common DirectionResolver and the `onMixedDirection` policy.
 * The weight is the `weight` XML attribute, overridden by a static value of the edge attribute
 * titled `weightFrom` ("weight" by default), applied with `setEdgeWeight` when that value arrives;
 * timed values of that attribute form the `temporal:edge:weight` table (design section 5.10). Ids
 * are coerced with the "canonical" rule of design section 4.1 by default. Edges may not reference
 * undeclared nodes unless `addMissingNodes` is set (the GEXF default is false, design section 8.4).
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    type ExtensionHandle,
    GraphFormatError,
    type GraphMetaPatch,
    type GraphSink,
    INVALID_INDEX,
    MAX_COUNT,
    type NodeId,
} from "@graphty/graph-format";

import {
    type AttributeDeclarationInput,
    declareAttribute,
    declareCompanion,
    type DeclaredAttribute,
    declareResolved,
    losesPrecision,
    parseDeclaredTemporal,
    parseDeclaredValue,
    PRECISION_CODE,
} from "../../common/attributes.js";
import {
    COUNT_HINT_CODE,
    DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_NODE_CODE,
    ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    NO_GRAPH_CODE,
    STRAY_TEXT_CODE,
    UNKNOWN_ELEMENT_CODE,
    UNKNOWN_PARENT_CODE,
    XML_SYNTAX_CODE,
} from "../../common/codes.js";
import { type DeclaredTypeSpec, parseDecimalText } from "../../common/declared-types.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { IdCoercer } from "../../common/ids.js";
import { textChunks, throwIfAborted } from "../../common/input.js";
import { type ListSyntax, splitListText } from "../../common/lists.js";
import {
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
} from "../../common/options.js";
import { ImportReportBuilder, type IssueLocation } from "../../common/report.js";
import { parseTimeText, type TemporalValue, type TimeFormat, timeTextCompanion } from "../../common/temporal.js";
import { isWeightField, parseWeightText } from "../../common/weights.js";
import { isWhitespace, localName, tokenizeXml, type XmlHandler, XmlSyntaxError } from "../../common/xml.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    EDGE_DECLS,
    GEXF_FORMAT,
    type GexfEdgeType,
    GRAPH_MODES,
    NODE_DECLS,
    OPEN_END,
    OPEN_START,
    RESERVED_EDGE_NAMES,
    RESERVED_NODE_NAMES,
    temporalTableDecls,
    temporalTableName,
    TIME_FORMATS,
    TIME_REPRESENTATIONS,
} from "./schema.js";

/** The format-specific options of the GEXF importer. */
export interface GexfImportOptions {
    /**
     * Whether the viz namespace elements (color, position, size, shape, thickness) are imported as
     * role columns (default true); false ignores them and records one `unsupported` warning.
     */
    viz?: boolean | undefined;
}

/** Issue code: the document is not a GEXF document (no `<gexf>` root). */
export const NOT_GEXF_CODE = "E_NOT_GEXF";
/** Issue code: the graph declares edges but no `<nodes>` section. */
export const MISSING_NODES_CODE = "E_GEXF_MISSING_NODES";
/** Issue code: an edge `type` outside directed / undirected / mutual. */
export const EDGE_TYPE_CODE = "E_GEXF_EDGE_TYPE";
/** Issue code: an `<attributes>` group without a class, or with an unknown one. */
export const ATTRIBUTES_CLASS_CODE = "E_GEXF_ATTRIBUTES_CLASS";
/** Issue code: an `<attribute>` without an id. */
export const ATTRIBUTE_ID_CODE = "E_GEXF_ATTRIBUTE_ID";
/** Issue code: a graph header value (`defaultedgetype`, `mode`, `timeformat`, ...) outside its set. */
export const HEADER_VALUE_CODE = "W_GEXF_HEADER_VALUE";
/** Issue code: a second `<attribute>` with the same id in the same class (the first is kept). */
export const DUPLICATE_ATTRIBUTE_CODE = "W_GEXF_DUPLICATE_ATTRIBUTE";
/** Issue code: an `<attribute>` without a type (read as string). */
export const ATTRIBUTE_TYPE_CODE = "W_GEXF_ATTRIBUTE_TYPE";
/** Issue code: an `<attvalue>` naming an attribute the document never declares (once per id). */
export const UNKNOWN_ATTRIBUTE_CODE = "W_GEXF_UNKNOWN_ATTRIBUTE";
/** Issue code: an `<attvalue>` without a `for` or a `value`. */
export const ATTVALUE_SHAPE_CODE = "W_GEXF_ATTVALUE_SHAPE";
/** Issue code: a timed value on an attribute declared in a static group (stored as dynamic anyway). */
export const TIMED_STATIC_CODE = "W_GEXF_TIMED_VALUE_ON_STATIC";
/** Issue code: the `weight` XML attribute is present but `weightFrom` is null. */
export const WEIGHT_IGNORED_CODE = "W_GEXF_WEIGHT_IGNORED";
/** Issue code: viz elements skipped under `viz: false`. */
export const VIZ_SKIPPED_CODE = "W_GEXF_VIZ_SKIPPED";
/** Issue code: a 1.2 dynamic viz element (its bounds are dropped, the value kept). */
export const VIZ_DYNAMIC_CODE = "W_GEXF_VIZ_DYNAMIC_DROPPED";
/** Issue code: `startopen` / `endopen` on a `<spell>` (the spells column has no open bits). */
export const SPELL_OPEN_CODE = "W_GEXF_SPELL_OPEN_DROPPED";
/** Issue code: a viz element with a value that does not parse (the element is skipped). */
export const VIZ_VALUE_CODE = "W_GEXF_VIZ_VALUE";
/** Issue code: both `start` and `startopen` (or `end` and `endopen`) on one element; the closed bound wins. */
export const OPEN_BOUND_CONFLICT_CODE = "W_GEXF_OPEN_BOUND_CONFLICT";

export {
    DUPLICATE_NODE_CODE,
    ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    NO_GRAPH_CODE,
    UNKNOWN_PARENT_CODE,
};

const EDGE_TYPES: ReadonlySet<string> = new Set(["directed", "undirected", "mutual"]);
const ABORT_CHECK_INTERVAL = 64;

/** The XML attributes the importer reads on `<node>`; any other one is reported once. */
const NODE_ATTRIBUTES: ReadonlySet<string> = new Set([
    "id",
    "label",
    "pid",
    "start",
    "end",
    "timestamp",
    "startopen",
    "endopen",
    "timestamps",
    "intervals",
]);

/** The XML attributes the importer reads on `<edge>`; any other one is reported once. */
const EDGE_ATTRIBUTES: ReadonlySet<string> = new Set([
    "id",
    "source",
    "target",
    "type",
    "label",
    "weight",
    "kind",
    "start",
    "end",
    "timestamp",
    "startopen",
    "endopen",
    "timestamps",
    "intervals",
]);

/** The common options the GEXF importer reads (the rest is reported by reportUnusedOptions). */
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
    "errorLimit",
    "signal",
    "onProgress",
]);

/** A GraphMetaPatch under construction (its fields are readonly once handed to the sink). */
type MetaPatchDraft = { -readonly [K in keyof GraphMetaPatch]: GraphMetaPatch[K] };

/** A declared attribute as the importer tracks it: its column and how values are parsed and stored. */
interface TrackedAttribute {
    /** The attribute id as written. */
    readonly id: string;
    /** The column name. */
    readonly name: string;
    /** How values are parsed. */
    readonly spec: DeclaredTypeSpec;
    /** The list syntax of values. */
    readonly listSyntax: ListSyntax;
    /** The column handle. */
    readonly handle: ColumnHandle;
    /** The companion text column's declaration of a temporal attribute, or null. */
    readonly companionDecl: ColumnDecl | null;
    /** The companion's handle once a value needed it, or INVALID_INDEX. */
    companion: ColumnHandle;
    /** Whether the attribute was declared in a dynamic group. */
    readonly dynamic: boolean;
    /** The value column fields of the temporal table. */
    readonly valueDecl: Pick<ColumnDecl, "dtype" | "itemDtype" | "options" | "origin">;
    /** The temporal table, created on the first timed value. */
    table: ExtensionHandle;
}

/** A pending containment reference resolved after every node is declared. */
interface ParentRef {
    /** The child's node index. */
    readonly child: number;
    /** The parent's id (from `pid` / `<parent for>`), or null when the index is known. */
    readonly parentId: NodeId | null;
    /** The parent's index (nested `<nodes>`), or INVALID_INDEX. */
    readonly parentIndex: number;
    /** Where the reference was read. */
    readonly where: IssueLocation;
}

/** A column declared on first use, through the design section 5.6 rule. */
class LazyColumn {
    private handle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private readonly domain: "node" | "edge";

    private readonly decl: ColumnDecl;

    /**
     * Create a lazily declared column.
     * @param domain - node or edge
     * @param decl - the declaration
     */
    constructor(domain: "node" | "edge", decl: ColumnDecl) {
        this.domain = domain;
        this.decl = decl;
    }

    /**
     * The handle, declaring the column on the first call (renamed or without its role when the
     * sink already holds the name or the role, reported).
     * @param sink - the sink
     * @param report - the report
     * @param where - the location of the first use
     * @returns the handle
     */
    get(sink: GraphSink, report: ImportReportBuilder, where?: IssueLocation): ColumnHandle {
        if (this.handle === INVALID_INDEX) {
            this.handle = declareResolved(sink, this.domain, this.decl, report, where).handle;
        }
        return this.handle;
    }

    /**
     * Write one cell, declaring the column first when needed.
     * @param sink - the sink
     * @param report - the report
     * @param index - the row
     * @param value - the value
     * @param where - the location, for a declaration issue
     */
    set(sink: GraphSink, report: ImportReportBuilder, index: number, value: unknown, where?: IssueLocation): void {
        const handle = this.get(sink, report, where);
        if (this.domain === "node") {
            sink.setNodeValue(handle, index, value);
        } else {
            sink.setEdgeValue(handle, index, value);
        }
    }
}

/** A temporal role column with its lazily declared text companion. */
class TemporalColumn {
    private readonly column: LazyColumn;

    private companion: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private readonly domain: "node" | "edge";

    private readonly name: string;

    /**
     * Create the pair.
     * @param domain - node or edge
     * @param decl - the numeric column's declaration
     */
    constructor(domain: "node" | "edge", decl: ColumnDecl) {
        this.domain = domain;
        this.name = decl.name;
        this.column = new LazyColumn(domain, decl);
    }

    /**
     * Write a parsed time value and, when its text must be kept, the companion cell.
     * @param sink - the sink
     * @param report - the report
     * @param index - the row
     * @param value - the parsed value
     * @param where - the location, for a declaration issue
     */
    set(
        sink: GraphSink,
        report: ImportReportBuilder,
        index: number,
        value: TemporalValue,
        where?: IssueLocation,
    ): void {
        this.column.set(sink, report, index, value.value, where);
        if (value.text === null) {
            return;
        }
        if (this.companion === INVALID_INDEX) {
            this.companion = declareCompanion(sink, this.domain, timeTextCompanion(this.name), report);
        }
        if (this.domain === "node") {
            sink.setNodeValue(this.companion, index, value.text);
        } else {
            sink.setEdgeValue(this.companion, index, value.text);
        }
    }
}

/** The lazily declared columns of one domain. */
interface DomainColumns {
    readonly label: LazyColumn;
    readonly start: TemporalColumn;
    readonly end: TemporalColumn;
    readonly timestamp: TemporalColumn;
    readonly spells: LazyColumn;
    readonly timestamps: LazyColumn;
    readonly open: LazyColumn;
    readonly color: LazyColumn;
    readonly shape: LazyColumn;
}

/** Where the reader is in the document. */
const enum Ctx {
    Gexf,
    Meta,
    MetaText,
    Graph,
    Attributes,
    Attribute,
    AttributeText,
    Nodes,
    Node,
    Attvalues,
    Parents,
    Spells,
    Edges,
    Edge,
    Skip,
}

/** An open `<node>`: its index (INVALID_INDEX when the node was skipped) and location. */
interface NodeFrame {
    readonly index: number;
    readonly where: IssueLocation;
}

/**
 * The open `<edge>`: its primary index, the mirror of an expanded pair (or INVALID_INDEX) and
 * location. One frame is reused for every edge (the hot loop allocates nothing per edge but the
 * tokenizer's attribute map): the report copies a location's fields into each issue, and nothing
 * keeps a reference to an edge's frame or location past the edge's end tag.
 */
interface EdgeFrame {
    index: number;
    mirror: number;
    readonly where: MutableLocation;
}

/** A location whose fields are overwritten per element (see EdgeFrame). */
interface MutableLocation {
    line: number;
    element: string | null;
}

/** An `<attribute>` being read: its start-tag fields and the texts of `<default>` / `<options>`. */
interface PendingAttribute {
    readonly domain: "node" | "edge";
    readonly dynamic: boolean;
    readonly attrs: ReadonlyMap<string, string>;
    readonly line: number;
    defaultText: string | null;
    optionsText: string | null;
}

/**
 * The event handler that turns the tokenizer's events into sink calls: one import call.
 */
class GexfReader implements XmlHandler {
    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private readonly options: ResolvedImportOptions;

    private readonly viz: boolean;

    private readonly coercer: IdCoercer;

    private readonly resolver: DirectionResolver;

    private readonly ctx: Ctx[] = [];

    private timeFormat: TimeFormat | null = null;

    private defaultKind: GexfEdgeType = "undirected";

    private readonly meta: MetaPatchDraft = { sourceFormat: GEXF_FORMAT };

    private metaText = "";

    private metaField: "creator" | "description" | "keywords" | null = null;

    private readonly nodeAttrs = new Map<string, TrackedAttribute>();

    private readonly edgeAttrs = new Map<string, TrackedAttribute>();

    private readonly declaredNodeNames = new Set<string>();

    private readonly declaredEdgeNames = new Set<string>();

    /** The edge attribute that is THE weight (its title equals weightFrom), or null. */
    private weightAttr: { readonly id: string; readonly spec: DeclaredTypeSpec; table: ExtensionHandle } | null = null;

    private readonly nodeColumns: DomainColumns;

    private readonly edgeColumns: DomainColumns;

    private readonly parentColumn = new LazyColumn("node", NODE_DECLS.parent);

    private readonly parentsColumn = new LazyColumn("node", NODE_DECLS.parents);

    private positionColumn: LazyColumn | null = null;

    private readonly sizeColumn = new LazyColumn("node", NODE_DECLS.size);

    private readonly shapeUriColumn = new LazyColumn("node", NODE_DECLS.shapeUri);

    private readonly thicknessColumn = new LazyColumn("edge", EDGE_DECLS.thickness);

    private readonly edgeIdColumn = new LazyColumn("edge", EDGE_DECLS.id);

    private readonly edgeKindColumn = new LazyColumn("edge", EDGE_DECLS.kind);

    private readonly edgeIds = new Set<string>();

    private parentRefs: ParentRef[] = [];

    private parentsRefs: { readonly child: number; readonly ids: readonly NodeId[]; readonly where: IssueLocation }[] =
        [];

    private readonly unknownAttributes = new Set<string>();

    private readonly nodeStack: NodeFrame[] = [];

    private edge: EdgeFrame | null = null;

    /** The one edge frame, reused for every `<edge>` (see EdgeFrame). */
    private readonly edgeFrame: EdgeFrame = {
        index: INVALID_INDEX,
        mirror: INVALID_INDEX,
        where: { line: 0, element: null },
    };

    /** The one `<attvalue>` location, reused per value (its fields are copied into every issue). */
    private readonly valueWhere: MutableLocation = { line: 0, element: null };

    private attributeGroup: { readonly domain: "node" | "edge" | null; readonly dynamic: boolean } | null = null;

    private pendingAttribute: PendingAttribute | null = null;

    private attributeText: "default" | "options" | null = null;

    private attributeTextValue = "";

    private spellPairs: number[][] | null = null;

    private spellsWhere: IssueLocation | null = null;

    private parentIds: NodeId[] | null = null;

    private nodesSeen = false;

    private graphSeen = false;

    private elementsSinceCheck = 0;

    /**
     * Set up one import.
     * @param sink - the sink
     * @param report - the report
     * @param options - the resolved common options
     * @param viz - whether viz elements are imported
     */
    constructor(sink: GraphSink, report: ImportReportBuilder, options: ResolvedImportOptions, viz: boolean) {
        this.sink = sink;
        this.report = report;
        this.options = options;
        this.viz = viz;
        this.coercer = new IdCoercer(options.ids);
        this.resolver = new DirectionResolver(sink, report, options.onMixedDirection);
        this.nodeColumns = {
            label: new LazyColumn("node", NODE_DECLS.label),
            start: new TemporalColumn("node", NODE_DECLS.start),
            end: new TemporalColumn("node", NODE_DECLS.end),
            timestamp: new TemporalColumn("node", NODE_DECLS.timestamp),
            spells: new LazyColumn("node", NODE_DECLS.spells),
            timestamps: new LazyColumn("node", NODE_DECLS.timestamps),
            open: new LazyColumn("node", NODE_DECLS.open),
            color: new LazyColumn("node", NODE_DECLS.color),
            shape: new LazyColumn("node", NODE_DECLS.shape),
        };
        this.edgeColumns = {
            label: new LazyColumn("edge", EDGE_DECLS.label),
            start: new TemporalColumn("edge", EDGE_DECLS.start),
            end: new TemporalColumn("edge", EDGE_DECLS.end),
            timestamp: new TemporalColumn("edge", EDGE_DECLS.timestamp),
            spells: new LazyColumn("edge", EDGE_DECLS.spells),
            timestamps: new LazyColumn("edge", EDGE_DECLS.timestamps),
            open: new LazyColumn("edge", EDGE_DECLS.open),
            color: new LazyColumn("edge", EDGE_DECLS.color),
            shape: new LazyColumn("edge", EDGE_DECLS.shape),
        };
    }

    // ---------------------------------------------------------------- events

    start(name: string, attrs: ReadonlyMap<string, string>, line: number): void {
        const parent = this.ctx.length === 0 ? null : this.ctx[this.ctx.length - 1];
        const local = localName(name);
        switch (parent) {
            case null:
                if (local !== "gexf") {
                    this.report.fail(NOT_GEXF_CODE, `the root element is <${name}>, not <gexf>`, { line });
                }
                this.readRoot(attrs);
                this.ctx.push(Ctx.Gexf);
                return;
            case Ctx.Gexf:
                this.startInGexf(name, local, attrs, line);
                return;
            case Ctx.Meta:
                this.startInMeta(name, local, line);
                return;
            case Ctx.Graph:
                this.startInGraph(name, local, attrs, line);
                return;
            case Ctx.Attributes:
                this.startInAttributes(name, local, attrs, line);
                return;
            case Ctx.Attribute:
                this.startInAttribute(name, local, line);
                return;
            case Ctx.Nodes:
                this.startInNodes(name, local, attrs, line);
                return;
            case Ctx.Node:
                this.startInNode(name, local, attrs, line);
                return;
            case Ctx.Attvalues:
                this.startInAttvalues(name, local, attrs, line);
                return;
            case Ctx.Parents:
                this.startInParents(name, local, attrs, line);
                return;
            case Ctx.Spells:
                this.startInSpells(name, local, attrs, line);
                return;
            case Ctx.Edges:
                this.startInEdges(name, local, attrs, line);
                return;
            case Ctx.Edge:
                this.startInEdge(name, local, attrs, line);
                return;
            case Ctx.MetaText:
            case Ctx.AttributeText:
            case Ctx.Skip:
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * An element ends.
     * @param _name - the element name
     * @param _line - the line (unused: issues of an element cite its start line)
     */
    end(_name: string, _line: number): void {
        const ctx = this.ctx.pop();
        switch (ctx) {
            case Ctx.Meta:
                this.finishMeta();
                return;
            case Ctx.MetaText:
                this.finishMetaText();
                return;
            case Ctx.Graph:
                this.resolveParents();
                return;
            case Ctx.Attributes:
                this.attributeGroup = null;
                return;
            case Ctx.Attribute:
                this.finishAttribute();
                return;
            case Ctx.AttributeText:
                this.finishAttributeText();
                return;
            case Ctx.Node:
                this.nodeStack.pop();
                return;
            case Ctx.Edge:
                this.edge = null;
                return;
            case Ctx.Parents:
                this.finishParents();
                return;
            case Ctx.Spells:
                this.finishSpells();
                return;
            case Ctx.Gexf:
            case Ctx.Nodes:
            case Ctx.Edges:
            case Ctx.Attvalues:
            case Ctx.Skip:
            case undefined:
                return;
            default: {
                const value: never = ctx;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown reader state ${String(value)}`, {});
            }
        }
    }

    /**
     * Character data: kept inside `<creator>` / `<description>` / `<keywords>` and `<default>` /
     * `<options>`, ignored elsewhere (reported once when it is not whitespace).
     * @param text - the text
     * @param line - the line
     */
    text(text: string, line: number): void {
        const ctx = this.ctx.length === 0 ? null : this.ctx[this.ctx.length - 1];
        switch (ctx) {
            case Ctx.MetaText:
                this.metaText += text;
                return;
            case Ctx.AttributeText:
                this.attributeTextValue += text;
                return;
            case Ctx.Skip:
                return;
            default:
                if (!isWhitespace(text)) {
                    this.report.warnOnce(
                        "validation-error",
                        STRAY_TEXT_CODE,
                        "text where GEXF allows only elements was ignored",
                        { line },
                    );
                }
        }
    }

    /**
     * Apply what the root and `<meta>` gave to the sink; called once the graph's header is read.
     */
    finish(): void {
        if (!this.graphSeen) {
            this.report.fail(NO_GRAPH_CODE, "the document has no <graph> element");
        }
    }

    // ---------------------------------------------------------------- root, meta, graph

    /**
     * Record the root's version.
     * @param attrs - the `<gexf>` attributes
     */
    private readRoot(attrs: ReadonlyMap<string, string>): void {
        const version = attrs.get("version");
        if (version !== undefined && version.length > 0) {
            this.meta.sourceVersion = version;
        }
    }

    /**
     * A child of `<gexf>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInGexf(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        switch (local) {
            case "meta": {
                const modified = attrs.get("lastmodifieddate");
                if (modified !== undefined && modified.length > 0) {
                    this.meta.modified = modified;
                }
                this.ctx.push(Ctx.Meta);
                return;
            }
            case "graph":
                if (this.graphSeen) {
                    this.report.fail(NO_GRAPH_CODE, "a second <graph> element; a GEXF document has one graph", {
                        line,
                    });
                }
                this.graphSeen = true;
                this.readHeader(attrs, line);
                this.ctx.push(Ctx.Graph);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A child of `<meta>`.
     * @param name - the element name
     * @param local - its local name
     * @param line - the line
     */
    private startInMeta(name: string, local: string, line: number): void {
        if (local === "creator" || local === "description" || local === "keywords") {
            this.metaField = local;
            this.metaText = "";
            this.ctx.push(Ctx.MetaText);
            return;
        }
        this.unknownElement(name, line);
    }

    /** Close a `<creator>` / `<description>` / `<keywords>`. */
    private finishMetaText(): void {
        const text = this.metaText;
        switch (this.metaField) {
            case "creator":
                this.meta.creator = text;
                break;
            case "description":
                this.meta.description = text;
                break;
            case "keywords":
                this.meta.keywords = text
                    .split(",")
                    .map((k) => k.trim())
                    .filter((k) => k.length > 0);
                break;
            default:
                break;
        }
        this.metaField = null;
        this.metaText = "";
    }

    /** Close `<meta>`: nothing more to do (the patch is applied at `<graph>`). */
    private finishMeta(): void {
        // the meta patch is applied when the graph header is read
    }

    /**
     * Read `<graph defaultedgetype mode timeformat timerepresentation idtype start end timestamp>`
     * into the meta and the importer's own state, apply the meta patch and set the sink's
     * direction; an unknown value is a warning and the default applies.
     * @param attrs - the `<graph>` attributes
     * @param line - the line
     */
    private readHeader(attrs: ReadonlyMap<string, string>, line: number): void {
        const where = { line };
        const patch = this.meta;
        const edgeType = attrs.get("defaultedgetype");
        if (edgeType === undefined) {
            this.defaultKind = this.options.defaultDirected ? "directed" : "undirected";
        } else if (EDGE_TYPES.has(edgeType)) {
            this.defaultKind = edgeType as GexfEdgeType;
        } else {
            this.defaultKind = this.options.defaultDirected ? "directed" : "undirected";
            this.warnHeader("defaultedgetype", edgeType, where);
        }
        const mode = attrs.get("mode") ?? attrs.get("type");
        if (mode !== undefined) {
            if (GRAPH_MODES.has(mode)) {
                patch.mode = mode as "static" | "dynamic" | "slice";
            } else {
                this.warnHeader("mode", mode, where);
            }
        }
        const timeFormat = attrs.get("timeformat");
        if (timeFormat !== undefined) {
            if (TIME_FORMATS.has(timeFormat)) {
                this.timeFormat = timeFormat as TimeFormat;
                patch.timeFormat = this.timeFormat;
            } else {
                this.warnHeader("timeformat", timeFormat, where);
            }
        }
        const representation = attrs.get("timerepresentation");
        if (representation !== undefined) {
            if (TIME_REPRESENTATIONS.has(representation)) {
                patch.timeRepresentation = representation as "interval" | "timestamp";
            } else {
                this.warnHeader("timerepresentation", representation, where);
            }
        }
        const idType = attrs.get("idtype");
        if (idType !== undefined) {
            if (idType === "string") {
                // informational only: Gephi writes idtype="string" unconditionally, so honouring it
                // would turn every Gephi export's integer ids into text; the canonical rule of
                // design section 4.1 applies and the exporter's check() reports W_ID_TEXT_TYPE
                patch.idType = "string";
            } else if (idType === "integer" || idType === "long") {
                patch.idType = "integer";
            } else {
                this.warnHeader("idtype", idType, where);
            }
        }
        const extra: Record<string, unknown> = {};
        for (const key of ["start", "end", "timestamp"]) {
            const value = attrs.get(key);
            if (value !== undefined) {
                extra[key] = value;
            }
        }
        if (Object.keys(extra).length > 0) {
            patch.extra = { gexf: extra };
        }
        this.sink.setMeta(patch);
        this.resolver.setHeader(this.defaultKind !== "undirected", where);
    }

    /**
     * Record a header value outside its set.
     * @param name - the attribute name
     * @param value - the value found
     * @param where - the header line
     */
    private warnHeader(name: string, value: string, where: IssueLocation): void {
        this.report.warning(
            "validation-error",
            HEADER_VALUE_CODE,
            `<graph ${name}="${value}"> is not a known value; the default applies`,
            { ...where, element: name },
        );
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
            case "attributes":
                this.beginAttributes(attrs, line);
                return;
            case "nodes":
                this.nodesSeen = true;
                this.reserve(attrs, "node", line);
                this.ctx.push(Ctx.Nodes);
                return;
            case "edges":
                if (!this.nodesSeen) {
                    this.report.error(
                        "validation-error",
                        MISSING_NODES_CODE,
                        "the graph declares <edges> but no <nodes>",
                        { line },
                    );
                }
                this.resolveParents();
                this.reserve(attrs, "edge", line);
                this.ctx.push(Ctx.Edges);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * Pass a section's `count` hint to the sink; a hint the sink cannot reserve is a warning.
     * @param attrs - the `<nodes>` or `<edges>` attributes
     * @param domain - which count it is
     * @param line - the line
     */
    private reserve(attrs: ReadonlyMap<string, string>, domain: "node" | "edge", line: number): void {
        const text = attrs.get("count");
        if (text === undefined) {
            return;
        }
        const count = Number(text);
        if (!Number.isInteger(count) || count < 0 || count > MAX_COUNT) {
            this.report.warning(
                "validation-error",
                COUNT_HINT_CODE,
                `<${domain}s count="${text}"> is not a count the sink can reserve; ignored`,
                { line, element: "count" },
            );
            return;
        }
        if (count === 0) {
            return;
        }
        try {
            if (domain === "node") {
                this.sink.reserve(count, undefined);
            } else {
                this.sink.reserve(undefined, count);
            }
        } catch (err) {
            if (!(err instanceof GraphFormatError)) {
                throw err;
            }
            this.report.warning("validation-error", COUNT_HINT_CODE, `count hint ${text} refused: ${err.message}`, {
                line,
                element: "count",
            });
        }
    }

    // ---------------------------------------------------------------- attribute declarations

    /**
     * Open an `<attributes class mode>` group.
     * @param attrs - the group's attributes
     * @param line - the line
     */
    private beginAttributes(attrs: ReadonlyMap<string, string>, line: number): void {
        const cls = attrs.get("class");
        const where = { line };
        let domain: "node" | "edge" | null = null;
        if (cls === "node" || cls === "edge") {
            domain = cls;
        } else {
            this.report.error(
                "validation-error",
                ATTRIBUTES_CLASS_CODE,
                cls === undefined
                    ? "<attributes> without a class"
                    : `<attributes class="${cls}"> is neither node nor edge`,
                where,
            );
        }
        const mode = attrs.get("mode");
        if (mode !== undefined && mode !== "static" && mode !== "dynamic") {
            this.warnHeader("mode", mode, where);
        }
        this.attributeGroup = { domain, dynamic: mode === "dynamic" };
        this.ctx.push(Ctx.Attributes);
    }

    /**
     * A child of `<attributes>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInAttributes(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "attribute") {
            this.unknownElement(name, line);
            return;
        }
        const group = this.attributeGroup;
        if (group === null || group.domain === null) {
            // the group's class was refused; its attributes are dropped with it
            this.ctx.push(Ctx.Skip);
            return;
        }
        this.pendingAttribute = {
            domain: group.domain,
            dynamic: group.dynamic,
            attrs,
            line,
            defaultText: null,
            optionsText: null,
        };
        this.ctx.push(Ctx.Attribute);
    }

    /**
     * A child of `<attribute>`: `<default>` or `<options>`.
     * @param name - the element name
     * @param local - its local name
     * @param line - the line
     */
    private startInAttribute(name: string, local: string, line: number): void {
        if (local === "default" || local === "options") {
            this.attributeText = local;
            this.attributeTextValue = "";
            this.ctx.push(Ctx.AttributeText);
            return;
        }
        this.unknownElement(name, line);
    }

    /** Close a `<default>` / `<options>`. */
    private finishAttributeText(): void {
        const pending = this.pendingAttribute;
        if (pending !== null) {
            if (this.attributeText === "default") {
                pending.defaultText = this.attributeTextValue;
            } else if (this.attributeText === "options") {
                pending.optionsText = this.attributeTextValue;
            }
        }
        this.attributeText = null;
        this.attributeTextValue = "";
    }

    /** Close an `<attribute>`: declare it. */
    private finishAttribute(): void {
        const pending = this.pendingAttribute;
        this.pendingAttribute = null;
        if (pending === null) {
            return;
        }
        try {
            this.declareOne(pending);
        } catch (err) {
            this.report.recordError(err, { line: pending.line, element: pending.attrs.get("id") ?? null });
        }
    }

    /**
     * Declare one `<attribute id title type>` with its `<default>` and `<options>` (design
     * sections 5.1, 5.5, 5.6). The edge attribute titled `weightFrom` is the weight and gets no
     * column; its declaration is recorded in meta.weightOrigin.
     * @param pending - the attribute as read
     */
    private declareOne(pending: PendingAttribute): void {
        const { report, sink, options } = this;
        const { domain, dynamic, attrs } = pending;
        const id = attrs.get("id");
        const where = { line: pending.line, element: id ?? null };
        if (id === undefined) {
            report.error("validation-error", ATTRIBUTE_ID_CODE, "<attribute> without an id", where);
            return;
        }
        const tracked = domain === "node" ? this.nodeAttrs : this.edgeAttrs;
        if (tracked.has(id) || (domain === "edge" && this.weightAttr !== null && this.weightAttr.id === id)) {
            report.warning(
                "validation-error",
                DUPLICATE_ATTRIBUTE_CODE,
                `${domain} attribute "${id}" is declared twice; the first declaration is kept`,
                where,
            );
            return;
        }
        const title = attrs.get("title") ?? null;
        const type = attrs.get("type") ?? null;
        if (type === null) {
            report.warning(
                "validation-error",
                ATTRIBUTE_TYPE_CODE,
                `attribute "${id}" declares no type; read as string`,
                where,
            );
        }
        const name = title !== null && title.length > 0 ? title : id;
        if (domain === "edge" && isWeightField(name, options.weightFrom)) {
            const declared = declareAttribute({ format: GEXF_FORMAT, id, title, type, long: options.long });
            for (const issue of declared.issues) {
                report.warning(issue.category, issue.code, issue.message, where);
            }
            this.weightAttr = { id, spec: declared.spec, table: INVALID_INDEX as ExtensionHandle };
            sink.setMeta({ weightOrigin: { format: GEXF_FORMAT, id, title, type, namespace: null } });
            return;
        }
        const declaredNames = domain === "node" ? this.declaredNodeNames : this.declaredEdgeNames;
        const reserved = domain === "node" ? RESERVED_NODE_NAMES : RESERVED_EDGE_NAMES;
        const taken = (candidate: string): boolean => reserved.has(candidate) || declaredNames.has(candidate);
        const input: AttributeDeclarationInput = {
            format: GEXF_FORMAT,
            id,
            title,
            type,
            defaultText: pending.defaultText,
            optionsText: pending.optionsText,
            listSyntax: "gexf",
            dynamic,
            long: options.long,
            taken,
        };
        const declared: DeclaredAttribute = declareAttribute(input);
        for (const issue of declared.issues) {
            report.warning(issue.category, issue.code, issue.message, where);
        }
        const resolved = declareResolved(sink, domain, declared.decl, report, where);
        declaredNames.add(resolved.decl.name);
        let companionDecl: ColumnDecl | null = null;
        if (declared.companion !== null) {
            companionDecl = {
                ...declared.companion,
                name: `${resolved.decl.name}.text`,
                extra: { for: resolved.decl.name },
            };
            declaredNames.add(companionDecl.name);
        }
        const valueDecl: Pick<ColumnDecl, "dtype" | "itemDtype" | "options" | "origin"> = {
            dtype: resolved.decl.dtype,
            itemDtype: resolved.decl.itemDtype,
            options: resolved.decl.options,
            origin: resolved.decl.origin,
        };
        tracked.set(id, {
            id,
            name: resolved.decl.name,
            spec: declared.spec,
            listSyntax: declared.listSyntax,
            handle: resolved.handle,
            companionDecl,
            companion: INVALID_INDEX as ColumnHandle,
            dynamic,
            valueDecl,
            table: INVALID_INDEX as ExtensionHandle,
        });
    }

    // ---------------------------------------------------------------- nodes

    /**
     * A child of `<nodes>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInNodes(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "node") {
            this.unknownElement(name, line);
            return;
        }
        this.beginNode(attrs, line);
    }

    /**
     * Push one `<node>`.
     * @param attrs - its attributes
     * @param line - the line
     */
    private beginNode(attrs: ReadonlyMap<string, string>, line: number): void {
        const { report, sink } = this;
        this.checkAbort();
        const enclosing = this.nodeStack.length === 0 ? INVALID_INDEX : this.nodeStack[this.nodeStack.length - 1].index;
        const idText = attrs.get("id");
        const where: IssueLocation = { line, element: idText ?? attrs.get("label") ?? null };
        this.checkAttributeNames(attrs, NODE_ATTRIBUTES, "node", where);
        if (idText === undefined) {
            report.error("missing-value", MISSING_ID_CODE, "<node> without an id", where);
            report.counts.skippedNodes++;
            this.nodeStack.push({ index: INVALID_INDEX, where });
            this.ctx.push(Ctx.Node);
            return;
        }
        let index: number;
        try {
            const id = this.coerceId(idText, where);
            const existing = sink.indexOf(id);
            if (existing !== INVALID_INDEX) {
                report.warning(
                    "validation-error",
                    DUPLICATE_NODE_CODE,
                    `node "${idText}" is declared more than once; later attributes overwrite`,
                    where,
                );
                index = existing;
            } else {
                index = sink.addNode(id);
                report.counts.nodes++;
            }
        } catch (err) {
            report.recordError(err, where);
            report.counts.skippedNodes++;
            this.nodeStack.push({ index: INVALID_INDEX, where });
            this.ctx.push(Ctx.Node);
            return;
        }
        this.nodeStack.push({ index, where });
        this.ctx.push(Ctx.Node);
        try {
            const label = attrs.get("label");
            if (label !== undefined) {
                this.nodeColumns.label.set(sink, report, index, label, where);
            }
            const pid = attrs.get("pid");
            if (pid !== undefined) {
                this.parentRefs.push({
                    child: index,
                    parentId: this.coerceId(pid, where),
                    parentIndex: INVALID_INDEX,
                    where,
                });
            } else if (enclosing !== INVALID_INDEX) {
                this.parentRefs.push({ child: index, parentId: null, parentIndex: enclosing, where });
            }
            this.writeLifetime(attrs, index, this.nodeColumns, where);
        } catch (err) {
            report.recordError(err, where);
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
        const frame = this.nodeStack[this.nodeStack.length - 1];
        if (frame.index === INVALID_INDEX && local !== "nodes") {
            // the node was skipped; its children are too
            this.ctx.push(Ctx.Skip);
            return;
        }
        switch (local) {
            case "attvalues":
                this.ctx.push(Ctx.Attvalues);
                return;
            case "parents":
                this.parentIds = [];
                this.ctx.push(Ctx.Parents);
                return;
            case "spells":
                this.spellPairs = [];
                this.spellsWhere = frame.where;
                this.ctx.push(Ctx.Spells);
                return;
            case "nodes":
                this.reserve(attrs, "node", line);
                this.ctx.push(Ctx.Nodes);
                return;
            case "color":
            case "position":
            case "size":
            case "shape":
                this.writeViz(name, local, attrs, "node", frame.index, frame.where);
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    /**
     * A `<parent for>` inside `<parents>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInParents(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "parent") {
            this.unknownElement(name, line);
            return;
        }
        this.ctx.push(Ctx.Skip);
        const frame = this.nodeStack[this.nodeStack.length - 1];
        const text = attrs.get("for");
        if (text === undefined) {
            this.report.warning("missing-value", ATTVALUE_SHAPE_CODE, "<parent> without a for attribute", frame.where);
            return;
        }
        try {
            this.parentIds?.push(this.coerceId(text, frame.where));
        } catch (err) {
            this.report.recordError(err, frame.where);
        }
    }

    /** Close `<parents>`: queue the list for resolution. */
    private finishParents(): void {
        const frame = this.nodeStack[this.nodeStack.length - 1];
        if (this.parentIds !== null && frame.index !== INVALID_INDEX) {
            this.parentsRefs.push({ child: frame.index, ids: this.parentIds, where: frame.where });
        }
        this.parentIds = null;
    }

    /**
     * Resolve every `pid`, nested and `<parents>` reference read so far now that every node is
     * declared (design section 8.4); an unknown parent is a missing-value error.
     */
    private resolveParents(): void {
        const { sink, report } = this;
        const refs = this.parentRefs;
        const lists = this.parentsRefs;
        this.parentRefs = [];
        this.parentsRefs = [];
        for (const ref of refs) {
            let parent = ref.parentIndex;
            if (ref.parentId !== null) {
                parent = sink.indexOf(ref.parentId);
                if (parent === INVALID_INDEX) {
                    report.error(
                        "missing-value",
                        UNKNOWN_PARENT_CODE,
                        `pid "${String(ref.parentId)}" names a node the document does not declare`,
                        ref.where,
                    );
                    continue;
                }
            }
            try {
                this.parentColumn.set(sink, report, ref.child, parent, ref.where);
            } catch (err) {
                report.recordError(err, ref.where);
            }
        }
        for (const ref of lists) {
            const indices: number[] = [];
            for (const id of ref.ids) {
                const parent = sink.indexOf(id);
                if (parent === INVALID_INDEX) {
                    report.error(
                        "missing-value",
                        UNKNOWN_PARENT_CODE,
                        `<parent for="${String(id)}"> names a node the document does not declare`,
                        ref.where,
                    );
                    continue;
                }
                indices.push(parent);
            }
            try {
                this.parentsColumn.set(sink, report, ref.child, indices, ref.where);
            } catch (err) {
                report.recordError(err, ref.where);
            }
        }
    }

    // ---------------------------------------------------------------- edges

    /**
     * A child of `<edges>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInEdges(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "edge") {
            this.unknownElement(name, line);
            return;
        }
        this.beginEdge(attrs, line);
    }

    /**
     * Push one `<edge>`.
     * @param attrs - its attributes
     * @param line - the line
     */
    private beginEdge(attrs: ReadonlyMap<string, string>, line: number): void {
        const { report, sink, options } = this;
        this.checkAbort();
        this.ctx.push(Ctx.Edge);
        this.edge = null;
        const sourceText = attrs.get("source");
        const targetText = attrs.get("target");
        const idText = attrs.get("id");
        const { where } = this.edgeFrame;
        where.line = line;
        where.element = idText ?? edgeElement(sourceText, targetText);
        this.checkAttributeNames(attrs, EDGE_ATTRIBUTES, "edge", where);
        if (sourceText === undefined || targetText === undefined) {
            report.error(
                "missing-value",
                MISSING_ENDPOINT_CODE,
                `<edge> without a ${sourceText === undefined ? "source" : "target"}`,
                where,
            );
            report.counts.skippedEdges++;
            return;
        }
        let kind: EdgeKind = this.defaultKind;
        const type = attrs.get("type");
        if (type !== undefined) {
            if (!EDGE_TYPES.has(type)) {
                report.error(
                    "validation-error",
                    EDGE_TYPE_CODE,
                    `edge type "${type}" is not directed, undirected or mutual`,
                    where,
                );
                report.counts.skippedEdges++;
                return;
            }
            kind = type as EdgeKind;
        }
        if (idText !== undefined) {
            if (this.edgeIds.has(idText)) {
                report.error(
                    "validation-error",
                    DUPLICATE_EDGE_ID_CODE,
                    `edge id "${idText}" is declared more than once; the edge is skipped`,
                    where,
                );
                report.counts.skippedEdges++;
                return;
            }
            this.edgeIds.add(idText);
        }
        let index: number;
        try {
            const source = this.coerceId(sourceText, where);
            const target = this.coerceId(targetText, where);
            if (!options.addMissingNodes) {
                if (sink.indexOf(source) === INVALID_INDEX) {
                    throw new GraphFormatError("E_UNKNOWN_NODE", `edge references undeclared node "${sourceText}"`, {
                        id: source,
                    });
                }
                if (sink.indexOf(target) === INVALID_INDEX) {
                    throw new GraphFormatError("E_UNKNOWN_NODE", `edge references undeclared node "${targetText}"`, {
                        id: target,
                    });
                }
            }
            const weight = this.xmlWeight(attrs, where);
            const before = sink.edgeCount;
            index = this.resolver.addEdge(source, target, kind, weight, where);
            report.counts.edges += sink.edgeCount - before;
        } catch (err) {
            report.recordError(err, where);
            report.counts.skippedEdges++;
            return;
        }
        const frame = this.edgeFrame;
        frame.index = index;
        frame.mirror = this.resolver.lastMirror;
        this.edge = frame;
        try {
            if (idText !== undefined) {
                this.edgeIdColumn.set(sink, report, index, idText, where);
            }
            const label = attrs.get("label");
            if (label !== undefined) {
                this.edgeColumns.label.set(sink, report, index, label, where);
            }
            const edgeKind = attrs.get("kind");
            if (edgeKind !== undefined) {
                this.edgeKindColumn.set(sink, report, index, edgeKind, where);
            }
            this.writeLifetime(attrs, index, this.edgeColumns, where);
        } catch (err) {
            report.recordError(err, where);
        }
    }

    /**
     * The weight argument of one edge from its `weight` XML attribute; undefined when absent (the
     * sink records a defaulted weight, design section 3.7).
     * @param attrs - the edge's attributes
     * @param where - the edge's location
     * @returns the weight, or undefined
     */
    private xmlWeight(attrs: ReadonlyMap<string, string>, where: IssueLocation): number | undefined {
        const text = attrs.get("weight");
        if (text === undefined) {
            return undefined;
        }
        if (this.options.weightFrom === null) {
            this.report.warnOnce(
                "coercion",
                WEIGHT_IGNORED_CODE,
                "the weight attribute is ignored because weightFrom is null; edges are unweighted",
                where,
            );
            return undefined;
        }
        return parseWeightText(text);
    }

    /**
     * A child of `<edge>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInEdge(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        const { edge } = this;
        if (edge === null) {
            this.ctx.push(Ctx.Skip);
            return;
        }
        switch (local) {
            case "attvalues":
                this.ctx.push(Ctx.Attvalues);
                return;
            case "spells":
                this.spellPairs = [];
                this.spellsWhere = edge.where;
                this.ctx.push(Ctx.Spells);
                return;
            case "color":
            case "thickness":
            case "shape":
                this.writeViz(name, local, attrs, "edge", edge.index, edge.where);
                this.ctx.push(Ctx.Skip);
                return;
            default:
                this.unknownElement(name, line);
        }
    }

    // ---------------------------------------------------------------- attribute values

    /**
     * An `<attvalue>` of the open node or edge; a value that does not parse is recorded and skipped
     * without losing the element or its other values.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInAttvalues(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "attvalue") {
            this.unknownElement(name, line);
            return;
        }
        this.ctx.push(Ctx.Skip);
        const { report, edge } = this;
        const domain: "node" | "edge" = edge === null ? "node" : "edge";
        let index: number;
        let element: string | null;
        if (edge !== null) {
            ({ index } = edge);
            element = edge.where.element ?? null;
        } else {
            const frame = this.nodeStack[this.nodeStack.length - 1];
            ({ index } = frame);
            element = frame.where.element ?? null;
        }
        if (index === INVALID_INDEX) {
            return;
        }
        const forId = attrs.get("for");
        const { valueWhere } = this;
        valueWhere.line = line;
        valueWhere.element = element;
        if (forId === undefined) {
            report.warning("missing-value", ATTVALUE_SHAPE_CODE, "<attvalue> without a for attribute", valueWhere);
            return;
        }
        const value = attrs.get("value");
        if (value === undefined) {
            report.warning(
                "missing-value",
                ATTVALUE_SHAPE_CODE,
                `<attvalue for="${forId}"> without a value attribute`,
                valueWhere,
            );
            return;
        }
        try {
            if (domain === "edge" && this.weightAttr !== null && this.weightAttr.id === forId) {
                if (isTimed(attrs)) {
                    this.writeTimedWeight(attrs, value, index, valueWhere);
                } else {
                    this.applyStaticWeight(value, valueWhere);
                }
                return;
            }
            const tracked = domain === "node" ? this.nodeAttrs : this.edgeAttrs;
            const attribute = tracked.get(forId);
            if (attribute === undefined) {
                const key = `${domain}:${forId}`;
                if (!this.unknownAttributes.has(key)) {
                    this.unknownAttributes.add(key);
                    report.warning(
                        "missing-value",
                        UNKNOWN_ATTRIBUTE_CODE,
                        `<attvalue for="${forId}"> names an undeclared ${domain} attribute; its values are dropped`,
                        { ...valueWhere, element: forId },
                    );
                }
                return;
            }
            if (isTimed(attrs)) {
                this.writeTimedValue(attribute, domain, attrs, value, index, valueWhere);
            } else {
                this.writeStaticValue(attribute, domain, value, index, valueWhere);
            }
        } catch (err) {
            report.recordError(err, { ...valueWhere, element: `${element ?? "?"}[${forId}]` });
        }
    }

    /**
     * A static value of the attribute titled `weightFrom` overrides the XML weight of the open
     * edge (and of its mirror half, design section 3.6).
     * @param text - the value text
     * @param where - the value's location
     */
    private applyStaticWeight(text: string, where: IssueLocation): void {
        const { edge, sink } = this;
        if (edge === null) {
            return;
        }
        const weight = parseWeightText(text);
        if (weight === undefined) {
            throw new GraphFormatError("E_INVALID_WEIGHT", "an edge weight cannot be blank", {
                value: text,
                line: where.line ?? null,
            });
        }
        sink.setEdgeWeight(edge.index, weight);
        if (edge.mirror !== INVALID_INDEX) {
            sink.setEdgeWeight(edge.mirror, weight);
        }
    }

    /**
     * Parse a value by its declaration and write the cell (and the temporal companion, declared
     * on the first value whose text is not canonical, design section 5.1).
     * @param attribute - the declaration
     * @param domain - node or edge
     * @param text - the value text
     * @param index - the element's index
     * @param where - the value's location
     */
    private writeStaticValue(
        attribute: TrackedAttribute,
        domain: "node" | "edge",
        text: string,
        index: number,
        where: IssueLocation,
    ): void {
        const { spec } = attribute;
        if (spec.temporal !== null && !spec.list) {
            const parsed = parseDeclaredTemporal(text, spec);
            this.setCell(domain, attribute.handle, index, parsed.value);
            if (parsed.text !== null && attribute.companionDecl !== null) {
                if (attribute.companion === INVALID_INDEX) {
                    attribute.companion = declareCompanion(this.sink, domain, attribute.companionDecl, this.report);
                }
                this.setCell(domain, attribute.companion, index, parsed.text);
            }
            return;
        }
        const value = parseDeclaredValue(text, spec, attribute.listSyntax);
        if (losesPrecision(spec, text)) {
            this.report.warning(
                "precision",
                PRECISION_CODE,
                `long value ${text.trim()} of "${attribute.name}" exceeds 2^53 and was rounded`,
                where,
            );
        }
        this.setCell(domain, attribute.handle, index, value);
    }

    /**
     * Append a timed value to the attribute's temporal table (design section 5.10), creating the
     * table on first use.
     * @param attribute - the declaration
     * @param domain - node or edge
     * @param attrs - the `<attvalue>` attributes carrying the bounds
     * @param text - the value text
     * @param index - the element's index
     * @param where - the value's location
     */
    private writeTimedValue(
        attribute: TrackedAttribute,
        domain: "node" | "edge",
        attrs: ReadonlyMap<string, string>,
        text: string,
        index: number,
        where: IssueLocation,
    ): void {
        const { sink, report } = this;
        if (!attribute.dynamic) {
            report.warnOnce(
                "validation-error",
                TIMED_STATIC_CODE,
                `attribute "${attribute.name}" is declared static but has timed values; they are stored as dynamic`,
                where,
            );
        }
        const { spec } = attribute;
        const temporalValue = spec.temporal !== null && !spec.list;
        if (attribute.table === INVALID_INDEX) {
            attribute.table = sink.addExtensionTable(
                temporalTableName(domain, attribute.name),
                temporalTableDecls(domain, attribute.valueDecl, temporalValue),
            );
        }
        let value: unknown;
        let valueText: string | null = null;
        if (temporalValue) {
            const parsed = parseDeclaredTemporal(text, spec);
            ({ value } = parsed);
            valueText = parsed.text;
        } else {
            value = parseDeclaredValue(text, spec, attribute.listSyntax);
            if (losesPrecision(spec, text)) {
                report.warning(
                    "precision",
                    PRECISION_CODE,
                    `long value ${text.trim()} of "${attribute.name}" exceeds 2^53 and was rounded`,
                    where,
                );
            }
        }
        const row = this.temporalRow(attrs, index, value, where);
        if (temporalValue) {
            row.push(valueText ?? undefined);
        }
        sink.addExtensionRow(attribute.table, row);
    }

    /**
     * Append a timed weight to the `temporal:edge:weight` table (design section 5.10).
     * @param attrs - the `<attvalue>` attributes carrying the bounds
     * @param text - the value text
     * @param index - the edge index
     * @param where - the value's location
     */
    private writeTimedWeight(
        attrs: ReadonlyMap<string, string>,
        text: string,
        index: number,
        where: IssueLocation,
    ): void {
        const { weightAttr } = this;
        if (weightAttr === null) {
            return;
        }
        const { sink } = this;
        if (weightAttr.table === INVALID_INDEX) {
            weightAttr.table = sink.addExtensionTable(
                temporalTableName("edge", this.options.weightFrom ?? "weight"),
                temporalTableDecls(
                    "edge",
                    {
                        dtype: "f64",
                        origin: { format: GEXF_FORMAT, id: weightAttr.id, type: weightAttr.spec.declared },
                    },
                    false,
                ),
            );
        }
        const value = parseWeightText(text);
        if (value === undefined) {
            throw new GraphFormatError("E_INVALID_WEIGHT", "a timed edge weight cannot be blank", {
                value: text,
                line: where.line ?? null,
            });
        }
        sink.addExtensionRow(weightAttr.table, this.temporalRow(attrs, index, value, where));
    }

    /**
     * The row of a temporal table for one timed value: element, bounds, value, bound texts, open bits.
     * @param attrs - the attributes carrying `start` / `end` / `timestamp` / `startopen` / `endopen`
     * @param index - the element's index
     * @param value - the parsed value
     * @param where - the value's location
     * @returns the row, without the optional value text
     */
    private temporalRow(
        attrs: ReadonlyMap<string, string>,
        index: number,
        value: unknown,
        where: IssueLocation,
    ): unknown[] {
        const bounds = this.readBounds(attrs, where);
        return [
            index,
            bounds.start.value,
            bounds.end.value,
            value,
            bounds.start.text ?? undefined,
            bounds.end.text ?? undefined,
            bounds.open === 0 ? undefined : bounds.open,
        ];
    }

    /**
     * The interval of an element or timed value: `timestamp`, or `start` / `startopen` and `end` /
     * `endopen` (GEXF 1.2: the open form carries the bound's time and marks the bound open, design
     * section 5.1); missing bounds are unbounded.
     * @param attrs - the attributes
     * @param where - the location, for the conflict warning
     * @returns the bounds and the open bits
     */
    private readBounds(
        attrs: ReadonlyMap<string, string>,
        where: IssueLocation,
    ): { start: TemporalValue; end: TemporalValue; open: number } {
        let start: TemporalValue = { value: -Infinity, text: null };
        let end: TemporalValue = { value: Infinity, text: null };
        let open = 0;
        const timestamp = attrs.get("timestamp");
        if (timestamp !== undefined) {
            start = parseTimeText(timestamp, this.timeFormat);
            end = start;
        }
        const startText = attrs.get("start");
        const startOpen = attrs.get("startopen");
        if (startText !== undefined) {
            start = parseTimeText(startText, this.timeFormat);
            if (startOpen !== undefined) {
                this.warnOpenConflict("start", where);
            }
        } else if (startOpen !== undefined) {
            start = parseTimeText(startOpen, this.timeFormat);
            open |= OPEN_START;
        }
        const endText = attrs.get("end");
        const endOpen = attrs.get("endopen");
        if (endText !== undefined) {
            end = parseTimeText(endText, this.timeFormat);
            if (endOpen !== undefined) {
                this.warnOpenConflict("end", where);
            }
        } else if (endOpen !== undefined) {
            end = parseTimeText(endOpen, this.timeFormat);
            open |= OPEN_END;
        }
        return { start, end, open };
    }

    /**
     * Record a `start` next to a `startopen` (or `end` next to `endopen`), which the 1.2 primer forbids.
     * @param bound - which bound
     * @param where - the location
     */
    private warnOpenConflict(bound: "start" | "end", where: IssueLocation): void {
        this.report.warnOnce(
            "validation-error",
            OPEN_BOUND_CONFLICT_CODE,
            `both ${bound} and ${bound}open are given; the closed bound is kept`,
            where,
        );
    }

    // ---------------------------------------------------------------- lifetimes

    /**
     * Write an element's lifetime: `start` / `end` / `timestamp` (with text companions), the 1.2
     * `startopen` / `endopen` bounds and bits, and the 1.3 `timestamps` / `intervals` attributes.
     * @param attrs - the node or edge attributes
     * @param index - its index
     * @param columns - the domain's columns
     * @param where - the element's location
     */
    private writeLifetime(
        attrs: ReadonlyMap<string, string>,
        index: number,
        columns: DomainColumns,
        where: IssueLocation,
    ): void {
        const { sink, report } = this;
        if (
            attrs.has("start") ||
            attrs.has("end") ||
            attrs.has("timestamp") ||
            attrs.has("startopen") ||
            attrs.has("endopen")
        ) {
            const bounds = this.readBounds(attrs, where);
            const timestamp = attrs.get("timestamp");
            if (timestamp !== undefined) {
                columns.timestamp.set(sink, report, index, bounds.start, where);
            }
            if (attrs.has("start") || attrs.has("startopen")) {
                columns.start.set(sink, report, index, bounds.start, where);
            }
            if (attrs.has("end") || attrs.has("endopen")) {
                columns.end.set(sink, report, index, bounds.end, where);
            }
            if (bounds.open !== 0) {
                columns.open.set(sink, report, index, bounds.open, where);
            }
        }
        const timestamps = attrs.get("timestamps");
        if (timestamps !== undefined) {
            const items = splitListText(stripAngles(timestamps), "brackets");
            columns.timestamps.set(
                sink,
                report,
                index,
                items.map((item) => parseTimeText(item, this.timeFormat).value),
                where,
            );
        }
        const intervals = attrs.get("intervals");
        if (intervals !== undefined) {
            columns.spells.set(sink, report, index, this.parseIntervals(intervals, where), where);
        }
    }

    /**
     * Parse a 1.3 `intervals="<[s, e]; [s, e]>"` attribute into spell pairs.
     * @param text - the attribute text
     * @param where - the element's location, for the error
     * @returns the pairs
     */
    private parseIntervals(text: string, where: IssueLocation): number[][] {
        const pairs: number[][] = [];
        for (const part of stripAngles(text).split(";")) {
            if (part.trim().length === 0) {
                continue;
            }
            const items = splitListText(part, "brackets");
            if (items.length !== 2) {
                throw new GraphFormatError("E_COLUMN_TYPE", `interval "${part.trim()}" is not a [start, end] pair`, {
                    value: part,
                    line: where.line ?? null,
                });
            }
            pairs.push([
                parseTimeText(items[0], this.timeFormat).value,
                parseTimeText(items[1], this.timeFormat).value,
            ]);
        }
        return pairs;
    }

    /**
     * A `<spell>` inside `<spells>`.
     * @param name - the element name
     * @param local - its local name
     * @param attrs - its attributes
     * @param line - the line
     */
    private startInSpells(name: string, local: string, attrs: ReadonlyMap<string, string>, line: number): void {
        if (local !== "spell") {
            this.unknownElement(name, line);
            return;
        }
        this.ctx.push(Ctx.Skip);
        const where = this.spellsWhere ?? { line };
        try {
            const bounds = this.readBounds(attrs, where);
            if (bounds.open !== 0) {
                this.report.warnOnce(
                    "unsupported",
                    SPELL_OPEN_CODE,
                    "startopen / endopen on a <spell> cannot be kept; the spell is stored closed",
                    where,
                );
            }
            this.spellPairs?.push([bounds.start.value, bounds.end.value]);
        } catch (err) {
            this.report.recordError(err, where);
        }
    }

    /** Close `<spells>`: write the pairs into the open element's spells column. */
    private finishSpells(): void {
        const pairs = this.spellPairs;
        const where = this.spellsWhere ?? undefined;
        this.spellPairs = null;
        this.spellsWhere = null;
        if (pairs === null) {
            return;
        }
        try {
            if (this.edge !== null) {
                this.edgeColumns.spells.set(this.sink, this.report, this.edge.index, pairs, where);
            } else {
                const frame = this.nodeStack[this.nodeStack.length - 1];
                if (frame.index !== INVALID_INDEX) {
                    this.nodeColumns.spells.set(this.sink, this.report, frame.index, pairs, where);
                }
            }
        } catch (err) {
            this.report.recordError(err, where);
        }
    }

    // ---------------------------------------------------------------- viz

    /**
     * Write one viz element into its role column; a value that does not parse is a recorded
     * warning and the element is skipped.
     * @param name - the element name as written
     * @param local - its local name
     * @param attrs - its attributes
     * @param domain - node or edge
     * @param index - the element's index
     * @param where - the element's location
     */
    private writeViz(
        name: string,
        local: string,
        attrs: ReadonlyMap<string, string>,
        domain: "node" | "edge",
        index: number,
        where: IssueLocation,
    ): void {
        const { sink, report } = this;
        if (!this.viz) {
            report.warnOnce("unsupported", VIZ_SKIPPED_CODE, "viz elements are skipped (option viz: false)", where);
            return;
        }
        if (isTimed(attrs)) {
            report.warnOnce(
                "unsupported",
                VIZ_DYNAMIC_CODE,
                "dynamic viz elements are not supported; the bounds are dropped and the last value kept",
                where,
            );
        }
        const columns = domain === "node" ? this.nodeColumns : this.edgeColumns;
        try {
            switch (local) {
                case "color":
                    columns.color.set(sink, report, index, parseColor(attrs), where);
                    break;
                case "position": {
                    const x = vizNumber(attrs, "x");
                    const y = vizNumber(attrs, "y");
                    const z = attrs.has("z") ? vizNumber(attrs, "z") : 0;
                    this.positionColumn ??= new LazyColumn("node", {
                        ...NODE_DECLS.position,
                        extra: { ...NODE_DECLS.position.extra, sourceDims: attrs.has("z") ? 3 : 2 },
                    });
                    this.positionColumn.set(sink, report, index, [x, y, z], where);
                    break;
                }
                case "size":
                    this.sizeColumn.set(sink, report, index, vizNumber(attrs, "value"), where);
                    break;
                case "thickness":
                    this.thicknessColumn.set(sink, report, index, vizNumber(attrs, "value"), where);
                    break;
                case "shape": {
                    const value = attrs.get("value");
                    if (value === undefined) {
                        throw new GraphFormatError("E_COLUMN_TYPE", "<viz:shape> without a value", { element: name });
                    }
                    columns.shape.set(sink, report, index, value, where);
                    const uri = attrs.get("uri");
                    if (domain === "node" && uri !== undefined) {
                        this.shapeUriColumn.set(sink, report, index, uri, where);
                    }
                    break;
                }
                default:
                    break;
            }
        } catch (err) {
            if (!(err instanceof GraphFormatError) || err.code !== "E_COLUMN_TYPE") {
                throw err;
            }
            report.warning("validation-error", VIZ_VALUE_CODE, `<${name}> skipped: ${err.message}`, where);
        }
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Report an element the schema does not define at this place and skip its subtree.
     * @param name - the element name
     * @param line - the line
     */
    private unknownElement(name: string, line: number): void {
        this.report.warnOnce("unsupported", UNKNOWN_ELEMENT_CODE, `element <${name}> is not GEXF and was skipped`, {
            line,
            element: name,
        });
        this.ctx.push(Ctx.Skip);
    }

    /**
     * Report XML attributes the importer does not read (once per attribute name and domain).
     * @param attrs - the element's attributes
     * @param known - the attribute names read
     * @param domain - node or edge
     * @param where - the element's location
     */
    private checkAttributeNames(
        attrs: ReadonlyMap<string, string>,
        known: ReadonlySet<string>,
        domain: "node" | "edge",
        where: IssueLocation,
    ): void {
        for (const key of attrs.keys()) {
            if (!known.has(key)) {
                const seen = `${domain}@${key}`;
                if (!this.unknownAttributes.has(seen)) {
                    this.unknownAttributes.add(seen);
                    this.report.warning(
                        "unsupported",
                        UNKNOWN_ELEMENT_CODE,
                        `attribute ${key} of <${domain}> is not GEXF and was ignored`,
                        { ...where, element: key },
                    );
                }
            }
        }
    }

    /**
     * Write one cell of a domain.
     * @param domain - node or edge
     * @param handle - the column
     * @param index - the row
     * @param value - the value
     */
    private setCell(domain: "node" | "edge", handle: ColumnHandle, index: number, value: unknown): void {
        if (domain === "node") {
            this.sink.setNodeValue(handle, index, value);
        } else {
            this.sink.setEdgeValue(handle, index, value);
        }
    }

    /**
     * Coerce an id text, recording a merge under ids "number".
     * @param text - the id text
     * @param where - the location, for the merge warning
     * @returns the id
     */
    private coerceId(text: string, where: IssueLocation): NodeId {
        const id = this.coercer.text(text);
        const merge = this.coercer.lastMerge;
        if (merge !== null) {
            this.report.warning(
                "coercion",
                ID_MERGED_CODE,
                `id text "${merge.text}" merged with "${merge.previousText}" as ${merge.id} under ids "number"`,
                where,
            );
        }
        return id;
    }

    /** Check the cancellation signal every ABORT_CHECK_INTERVAL elements. */
    private checkAbort(): void {
        if (++this.elementsSinceCheck >= ABORT_CHECK_INTERVAL) {
            this.elementsSinceCheck = 0;
            throwIfAborted(this.options.signal);
        }
    }
}

/**
 * The element text of an edge without an id, for issues.
 * @param source - the source text, or undefined
 * @param target - the target text, or undefined
 * @returns `source->target`
 */
function edgeElement(source: string | undefined, target: string | undefined): string {
    return `${source ?? "?"}->${target ?? "?"}`;
}

/**
 * Whether an element carries time bounds.
 * @param attrs - the element's attributes
 * @returns true when start, end, timestamp or an open bound is present
 */
function isTimed(attrs: ReadonlyMap<string, string>): boolean {
    return (
        attrs.has("start") ||
        attrs.has("end") ||
        attrs.has("timestamp") ||
        attrs.has("startopen") ||
        attrs.has("endopen")
    );
}

/**
 * Remove the `<` `>` wrapper of a 1.3 `timestamps` / `intervals` attribute value.
 * @param text - the attribute text
 * @returns the inner text
 */
function stripAngles(text: string): string {
    let inner = text.trim();
    if (inner.startsWith("<")) {
        inner = inner.slice(1);
    }
    if (inner.endsWith(">")) {
        inner = inner.slice(0, -1);
    }
    return inner;
}

/**
 * A numeric viz attribute.
 * @param attrs - the element's attributes
 * @param name - the attribute name
 * @returns the number; E_COLUMN_TYPE when absent or not a number
 */
function vizNumber(attrs: ReadonlyMap<string, string>, name: string): number {
    const text = attrs.get(name);
    if (text === undefined) {
        throw new GraphFormatError("E_COLUMN_TYPE", `viz element without a ${name} attribute`, { attribute: name });
    }
    const value = parseDecimalText(text);
    if (Number.isNaN(value)) {
        throw new GraphFormatError("E_COLUMN_TYPE", `viz ${name}="${text}" is not a number`, { attribute: name });
    }
    return value;
}

/**
 * Parse `<viz:color r g b a hex>` into rgba in 0..1 (design section 5.5: role color, f32 x4).
 * @param attrs - the element's attributes
 * @returns [r, g, b, a]
 */
function parseColor(attrs: ReadonlyMap<string, string>): number[] {
    let r: number;
    let g: number;
    let b: number;
    const hex = attrs.get("hex");
    if (hex !== undefined && !attrs.has("r")) {
        const digits = hex.startsWith("#") ? hex.slice(1) : hex;
        if (!/^[0-9a-fA-F]{6}$/.test(digits)) {
            throw new GraphFormatError("E_COLUMN_TYPE", `viz hex="${hex}" is not a #RRGGBB colour`, {
                attribute: "hex",
            });
        }
        r = Number.parseInt(digits.slice(0, 2), 16);
        g = Number.parseInt(digits.slice(2, 4), 16);
        b = Number.parseInt(digits.slice(4, 6), 16);
    } else {
        r = vizNumber(attrs, "r");
        g = vizNumber(attrs, "g");
        b = vizNumber(attrs, "b");
    }
    const a = attrs.has("a") ? vizNumber(attrs, "a") : 1;
    return [r / 255, g / 255, b / 255, a];
}

/**
 * Confidence that a document head is GEXF.
 * @param head - the first bytes
 * @returns 1 for a `<gexf` tag, 0.8 for a gexf.net namespace, 0 otherwise
 */
function sniffGexf(head: Uint8Array): number {
    const text = new TextDecoder("utf-8", { fatal: false }).decode(head);
    if (/<(\w+:)?gexf[\s>]/.test(text)) {
        return 1;
    }
    return text.includes("gexf.net") ? 0.8 : 0;
}

/**
 * Validate the format-specific options.
 * @param options - the caller's options
 * @returns whether viz elements are imported
 */
function resolveGexfOptions(options: (GexfImportOptions & CommonImportOptions) | undefined): boolean {
    const viz = options?.viz;
    if (viz === undefined) {
        return true;
    }
    if (typeof viz !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", `option viz: ${typeof viz} is not a boolean`, {
            option: "viz",
            found: viz,
        });
    }
    return viz;
}

/** The GEXF importer (design section 8.4). */
export const gexfImporter: GraphImporter<GexfImportOptions> = Object.freeze({
    format: GEXF_FORMAT,
    extensions: Object.freeze([".gexf"]),
    mimeTypes: Object.freeze(["application/gexf+xml", "application/xml", "text/xml"]),
    sniff: sniffGexf,
    /**
     * Read a GEXF document into the sink in one streaming pass.
     * @param input - the document as text, bytes, a stream or chunks
     * @param sink - the sink
     * @param options - format-specific and common options
     * @returns the report
     */
    async import(
        input: ImportInput,
        sink: GraphSink,
        options?: GexfImportOptions & CommonImportOptions,
    ): Promise<ImportReport> {
        const resolved = resolveImportOptions(options, {
            ids: "canonical",
            defaultDirected: false,
            weightFrom: "weight",
            addMissingNodes: false,
        });
        const viz = resolveGexfOptions(options);
        const report = new ImportReportBuilder(GEXF_FORMAT, resolved.errorLimit);
        reportSinkOptions(sink, options, report, true);
        // nodeIdFrom is among the options reportUnusedOptions() reports: GEXF ids are mandatory
        reportUnusedOptions(options, report, USED_OPTIONS);
        const reader = new GexfReader(sink, report, resolved, viz);
        try {
            await tokenizeXml(textChunks(input, report, resolved), reader);
        } catch (err) {
            if (err instanceof XmlSyntaxError) {
                report.fail(XML_SYNTAX_CODE, err.message, { line: err.line });
            }
            throw err;
        }
        reader.finish();
        // an abort raised during the last few elements (after the last periodic check) still rejects
        throwIfAborted(resolved.signal);
        return report.finish();
    },
});
