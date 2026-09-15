/**
 * The Pajek NET importer (design sections 8.4 and 8.6; research note 07 section 2.5): a
 * line-oriented streaming reader over the common LineReader that pushes scalars into the sink as
 * each line is read. `*Vertices N` declares vertices 1..N (0..N-1 for the zero-based files some
 * scripts write, detected from the first vertex line), so the id map of a well-formed file is
 * `identity` with offset 1; vertex lines add the label, the coordinates (the position role,
 * units "file"), the shape keyword and the `key value` parameters (a `graphty_originalId`
 * parameter, written by the exporter under `sanitizeIds: "mangle"`, is restored as the vertex's
 * id under `restoreMangledIds`); `*Arcs` sections are directed
 * and `*Edges` sections undirected, resolved through the common DirectionResolver under the
 * mixed-direction policy; the third column of a line is the weight unless `weightFrom` says
 * otherwise; `*Arcslist` / `*Edgeslist` adjacency lists and `*Matrix` rows are read too; time
 * interval tokens `[1-5,7-*]` become the spells role. A second network in one file is refused
 * with an issue; project-file sections (`*Partition`, `*Vector`, ...) are reported as unsupported.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphMetaPatch,
    type GraphSink,
    INVALID_INDEX,
    type NodeId,
    type U32,
} from "@graphty/graph-format";

import { declareResolved, RENAMED_CODE, ROLE_TAKEN_CODE } from "../../common/attributes.js";
import { DUPLICATE_NODE_CODE, INVALID_UTF8_CODE, OPTION_IGNORED_CODE, SYNTAX_CODE } from "../../common/codes.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { coerceIdText, ID_MERGED_CODE, IdCoercer } from "../../common/ids.js";
import { LineReader, throwIfAborted } from "../../common/input.js";
import {
    type ImportFormatDefaults,
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
    SINK_OPTION_CODE,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { isNumericText, TextCellWriter, WIDENING_UNSUPPORTED_CODE } from "../../common/text.js";
import { isWeightField, parseWeightText } from "../../common/weights.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    isCommentOrBlank,
    isIntervalToken,
    isSectionLine,
    isVertexNumber,
    LABEL_COLUMN,
    ORIGINAL_ID_KEY,
    parseIntervals,
    parseSectionHeader,
    POSITION_COLUMN,
    RELATION_COLUMN,
    type SectionHeader,
    SHAPE_COLUMN,
    SHAPES,
    SPELLS_COLUMN,
    tokenize,
    VALUE_COLUMN,
    VALUE_FIELD,
} from "./syntax.js";

/** The format-specific options of the Pajek importer. */
export interface PajekImportOptions {
    /**
     * The number of the first vertex: 1 (Pajek's rule), 0 (files written by zero-based scripts), or
     * "auto" (default): 0 when the first vertex line is numbered 0, 1 otherwise.
     */
    firstVertex?: 0 | 1 | "auto" | undefined;
}

/** Issue codes of the Pajek importer. */
export const PAJEK_ISSUE = Object.freeze({
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** Fatal: no `*Vertices` section (an empty file, or not a Pajek network). */
    NO_VERTICES: "E_PAJEK_NO_VERTICES",
    /** Fatal: `*Vertices` without a vertex count, or one the sink cannot hold. */
    VERTICES_COUNT: "E_PAJEK_VERTICES_COUNT",
    /** Aborts: a second `*Vertices` or `*Network` section; only one network per file is read. */
    MULTIPLE_NETWORKS: "E_PAJEK_MULTIPLE_NETWORKS",
    /** A data line before the first section header. */
    OUTSIDE_SECTION: "E_PAJEK_OUTSIDE_SECTION",
    /** A section header the importer cannot parse. */
    SYNTAX: SYNTAX_CODE,
    /** A double quote not closed before the end of the line. */
    UNTERMINATED_QUOTE: "E_PAJEK_UNTERMINATED_QUOTE",
    /** A vertex line the grammar does not accept. */
    VERTEX_LINE: "E_PAJEK_VERTEX_LINE",
    /** A vertex number outside the declared range. */
    VERTEX_RANGE: "E_PAJEK_VERTEX_RANGE",
    /** Fewer vertex lines than `*Vertices` declares (vertices without a line have no label). */
    VERTEX_COUNT: "E_PAJEK_VERTEX_COUNT",
    /** A second line for the same vertex; the later values overwrite. */
    DUPLICATE_NODE: DUPLICATE_NODE_CODE,
    /** A line (arc, edge, list or matrix row) the grammar does not accept. */
    LINE: "E_PAJEK_LINE",
    /** A line endpoint outside the declared vertex range (the core's code, forwarded). */
    UNKNOWN_NODE: "E_UNKNOWN_NODE",
    /** A malformed time interval token. */
    INTERVAL: "E_PAJEK_INTERVAL",
    /** A `*Matrix` section with the wrong number of rows. */
    MATRIX_ROWS: "E_PAJEK_MATRIX_ROWS",
    /** A project-file section (`*Partition`, `*Vector`, ...) the importer does not read. */
    UNSUPPORTED_SECTION: "E_PAJEK_UNSUPPORTED_SECTION",
    /** Tokens after a section header the grammar does not account for. */
    HEADER_EXTRA: "W_PAJEK_HEADER_EXTRA",
    /** The file declares vertices but no line section. */
    NO_LINES: "W_PAJEK_NO_LINES",
    /** Vertex numbering starts at 0 rather than 1. */
    ZERO_BASED: "W_PAJEK_ZERO_BASED",
    /** Vertex lines mix two and three coordinates. */
    COORD_DIMS: "W_PAJEK_COORD_DIMS",
    /** Two vertices share a label under nodeIdFrom "label" and became one node. */
    LABEL_MERGED: "W_PAJEK_LABEL_MERGED",
    /** Two distinct label texts became one numeric id under ids "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** A parameter column of `2.0`-style text kept at the value-inferred dtype (the sink cannot widen). */
    WIDENING_UNSUPPORTED: WIDENING_UNSUPPORTED_CODE,
    /** A structural column (label, position, shape, spells, relation) renamed `<name>#<id>` because the name was taken (design section 5.6). */
    COLUMN_RENAMED: RENAMED_CODE,
    /** A structural column declared without its role because the sink already holds it. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** Two vertices carry the same `graphty_originalId` under restoreMangledIds and became one node. */
    ORIGINAL_ID_MERGED: "W_PAJEK_ORIGINAL_ID_MERGED",
    /** A vertex line's `graphty_originalId` came after a later vertex's line had created it under its number. */
    ORIGINAL_ID_UNRESTORED: "W_PAJEK_ORIGINAL_ID_UNRESTORED",
    /** A common option the importer has no use for (weightFrom naming a parameter and restoreMangledIds are honoured; long, hyperedges are not). */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
    /** A builder-policy option the caller passed that the caller's sink does not use (the shared W_SINK_OPTION). */
    SINK_OPTION: SINK_OPTION_CODE,
});

/** The common options the Pajek importer reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
    "nodeIdFrom",
    "restoreMangledIds",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "defaultDirected",
    "weightFrom",
    "weightDtype",
    "errorLimit",
    "signal",
    "onProgress",
]);

const DEFAULTS: ImportFormatDefaults = { ids: "canonical", defaultDirected: true, weightFrom: VALUE_FIELD };

/** Lines between two checks of the cancellation signal (a whole string input is one chunk). */
const ABORT_CHECK_INTERVAL = 64;

const FIRST_VERTEX_VALUES: ReadonlySet<unknown> = new Set([0, 1, "auto"]);

const LABEL_DECL: ColumnDecl = {
    name: LABEL_COLUMN,
    dtype: "string",
    nullable: true,
    role: "label",
    origin: { format: "pajek", id: "label" },
};

const SHAPE_DECL: ColumnDecl = {
    name: SHAPE_COLUMN,
    dtype: "dict",
    nullable: true,
    origin: { format: "pajek", id: "shape" },
};

const RELATION_DECL: ColumnDecl = {
    name: RELATION_COLUMN,
    dtype: "dict",
    nullable: true,
    origin: { format: "pajek", id: "relation" },
};

/**
 * The spells column declaration of a domain (design section 5.1: a list of f64 pairs).
 * @returns a fresh declaration
 */
function spellsDecl(): ColumnDecl {
    return {
        name: SPELLS_COLUMN,
        dtype: "list",
        itemDtype: "f64",
        itemComponents: 2,
        nullable: true,
        role: "spells",
        origin: { format: "pajek", id: "interval" },
    };
}

/**
 * The position column declaration (design section 5.2: f32 x 3, role position, units "file").
 * @param sourceDims - 2 or 3, the coordinate count of the first vertex line with coordinates
 * @returns a fresh declaration
 */
function positionDecl(sourceDims: number): ColumnDecl {
    return {
        name: POSITION_COLUMN,
        dtype: "f32",
        components: 3,
        nullable: true,
        role: "position",
        origin: { format: "pajek", id: "coordinates" },
        extra: { sourceDims, units: "file" },
    };
}

/** A problem with one line, recorded as an issue by the per-line catch. */
class LineError extends Error {
    readonly category: "parse-error" | "validation-error" | "missing-value";

    readonly code: string;

    /**
     * Create a line error.
     * @param category - the issue category
     * @param code - the issue code
     * @param message - a plain-ASCII message
     */
    constructor(category: "parse-error" | "validation-error" | "missing-value", code: string, message: string) {
        super(message);
        this.name = "PajekLineError";
        this.category = category;
        this.code = code;
    }
}

/** What a vertex or line row carries besides its numbers, parsed before anything is written. */
interface RowExtras {
    /** Parameter keys, in order. */
    readonly keys: string[];
    /** Parameter value texts, aligned with keys. */
    readonly values: string[];
    /** The spells of an interval token, or null. */
    spells: [number, number][] | null;
}

/** The section the parser is in. */
type Section = "none" | "vertices" | "lines" | "list" | "matrix" | "skip";

/**
 * One import call's state machine: the sections, the vertex range, the columns declared so far
 * and the direction resolver.
 */
class PajekParser {
    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private readonly options: ResolvedImportOptions;

    private readonly resolver: DirectionResolver;

    private readonly coercer: IdCoercer;

    private readonly firstVertex: 0 | 1 | "auto";

    private section: Section = "none";

    /** The direction of the current line section. */
    private kind: EdgeKind = "directed";

    /** The relation name of the current line section, or null. */
    private relation: string | null = null;

    /** `*Vertices N`; -1 before the header. */
    private vertexCount = -1;

    /** The first-mode count of a two-mode network, or null. */
    private firstMode: number | null = null;

    private networkName: string | null = null;

    private base = 1;

    private baseKnown = false;

    private nodesCreated = false;

    private verticesDone = false;

    private sawLineSection = false;

    /** Node index per vertex position (vertex number minus base). */
    private indexOfPos: U32 = new Uint32Array(0);

    /**
     * The node id per vertex position when the vertex lines decide the ids (nodeIdFrom "label",
     * or a restorable `graphty_originalId` under restoreMangledIds); null when the ids are the
     * vertex numbers and every vertex is created up front.
     */
    private idOfPos: NodeId[] | null = null;

    /** Which positions had a vertex line. */
    private seen: Uint8Array = new Uint8Array(0);

    private seenCount = 0;

    private matrixRows = 0;

    private coordDims = 0;

    private labelHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private positionHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private shapeHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private nodeSpellsHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private relationHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private edgeSpellsHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private readonly nodeParams = new Map<string, TextCellWriter>();

    private readonly edgeParams = new Map<string, TextCellWriter>();

    /**
     * Create the parser for one import call.
     * @param sink - the sink
     * @param report - the report
     * @param options - the resolved common options
     * @param firstVertex - the firstVertex format option
     */
    constructor(
        sink: GraphSink,
        report: ImportReportBuilder,
        options: ResolvedImportOptions,
        firstVertex: 0 | 1 | "auto",
    ) {
        this.sink = sink;
        this.report = report;
        this.options = options;
        this.firstVertex = firstVertex;
        this.resolver = new DirectionResolver(sink, report, options.onMixedDirection);
        this.coercer = new IdCoercer(options.ids);
    }

    /**
     * Handle one line; every problem of the line is recorded as an issue and the line skipped.
     * @param text - the line without its terminator
     * @param line - the 1-based line number
     */
    line(text: string, line: number): void {
        if (isCommentOrBlank(text)) {
            return;
        }
        try {
            if (isSectionLine(text)) {
                this.header(text, line);
                return;
            }
            switch (this.section) {
                case "none":
                    throw new LineError(
                        "parse-error",
                        PAJEK_ISSUE.OUTSIDE_SECTION,
                        "data before the first section header; a Pajek network starts with *Vertices",
                    );
                case "vertices":
                    this.vertexLine(text, line);
                    return;
                case "lines":
                    this.edgeLine(text, line);
                    return;
                case "list":
                    this.listLine(text, line);
                    return;
                case "matrix":
                    this.matrixLine(text, line);
                    return;
                case "skip":
                    return;
                default: {
                    const name: string = this.section;
                    throw new Error(`unknown section state ${name}`);
                }
            }
        } catch (err) {
            if (err instanceof LineError) {
                this.report.error(err.category, err.code, err.message, { line });
                return;
            }
            this.report.recordError(err, { line });
        }
    }

    /**
     * End of input: the vertex set is completed, a network without line sections gets the default
     * direction, and the metadata is written.
     */
    finish(): void {
        if (this.vertexCount < 0) {
            this.report.fail(PAJEK_ISSUE.NO_VERTICES, "not a Pajek network: no *Vertices section found");
        }
        this.endSection();
        this.finishVertices();
        if (!this.sawLineSection) {
            this.report.warning(
                "validation-error",
                PAJEK_ISSUE.NO_LINES,
                "the file declares vertices but no *Arcs, *Edges, *Arcslist, *Edgeslist or *Matrix section",
            );
            this.resolver.setHeader(this.options.defaultDirected);
        }
        if (this.coercer.mergeCount > 0) {
            this.report.warning(
                "coercion",
                PAJEK_ISSUE.ID_MERGED,
                `${this.coercer.mergeCount} label(s) merged into an id another label already produced under ids "number"`,
            );
        }
        const meta: GraphMetaPatch = {
            sourceFormat: "pajek",
            ...(this.networkName === null ? {} : { name: this.networkName }),
            ...(this.firstMode === null ? {} : { extra: { pajek: { firstMode: this.firstMode } } }),
        };
        this.sink.setMeta(meta);
    }

    // ============================================================ headers

    /**
     * A section header line.
     * @param text - the line
     * @param line - the line number
     */
    private header(text: string, line: number): void {
        const skipping = this.section === "skip";
        this.endSection();
        const h = parseSectionHeader(text);
        if (h === null) {
            this.section = "skip";
            throw new LineError("parse-error", PAJEK_ISSUE.SYNTAX, `cannot read the section header "${text}"`);
        }
        if (h.extra.length > 0) {
            this.report.warning(
                "validation-error",
                PAJEK_ISSUE.HEADER_EXTRA,
                `ignored ${h.extra.length} unexpected token(s) after *${h.keyword}: ${h.extra.join(" ")}`,
                { line },
            );
        }
        switch (h.kind) {
            case "network":
                if (this.networkName !== null || this.vertexCount >= 0) {
                    this.refuseSecondNetwork(h.keyword, line);
                }
                this.networkName = h.name ?? "";
                this.section = "none";
                return;
            case "vertices":
                if (skipping && this.vertexCount >= 0) {
                    // the `*Vertices N` line of a skipped project-file section (*Partition, *Vector)
                    this.section = "skip";
                    return;
                }
                this.verticesHeader(h, line);
                return;
            case "arcs":
            case "edges":
                this.lineSection(h, line, "lines", h.kind === "arcs" ? "directed" : "undirected");
                return;
            case "arcslist":
            case "edgeslist":
                this.lineSection(h, line, "list", h.kind === "arcslist" ? "directed" : "undirected");
                return;
            case "matrix":
                this.lineSection(h, line, "matrix", "directed");
                this.matrixRows = 0;
                return;
            case "unsupported":
                this.section = "skip";
                this.report.error(
                    "unsupported",
                    PAJEK_ISSUE.UNSUPPORTED_SECTION,
                    `the *${h.keyword} section is not supported; its lines are skipped`,
                    { line },
                );
                return;
            default: {
                const name: string = h.kind;
                throw new Error(`unknown section kind ${name}`);
            }
        }
    }

    /**
     * `*Vertices N [N1]`.
     * @param h - the header
     * @param line - the line number
     */
    private verticesHeader(h: SectionHeader, line: number): void {
        if (this.vertexCount >= 0) {
            this.refuseSecondNetwork(h.keyword, line);
        }
        if (h.count === null) {
            this.report.fail(PAJEK_ISSUE.VERTICES_COUNT, "*Vertices needs a vertex count", { line });
        }
        if (!Number.isSafeInteger(h.count)) {
            this.report.fail(PAJEK_ISSUE.VERTICES_COUNT, `*Vertices ${h.count}: not a count`, { line });
        }
        try {
            // the declared count must be reservable (the sink is the authority on what it can hold,
            // E_TOO_LARGE beyond MAX_COUNT) before any vertex is materialised
            this.sink.reserve(h.count);
        } catch (err) {
            const reason = err instanceof Error ? err.message : String(err);
            this.report.fail(
                PAJEK_ISSUE.VERTICES_COUNT,
                `*Vertices ${h.count}: the sink cannot hold that many (${reason})`,
                {
                    line,
                },
            );
        }
        this.vertexCount = h.count;
        if (h.secondCount !== null) {
            this.firstMode = h.secondCount;
        }
        this.section = "vertices";
        this.indexOfPos = new Uint32Array(h.count);
        this.seen = new Uint8Array(h.count);
        if (this.options.nodeIdFrom === "label" || this.restoresIds()) {
            this.idOfPos = new Array<NodeId>(h.count);
        }
        if (this.firstVertex !== "auto") {
            this.setBase(this.firstVertex, line);
        }
    }

    /**
     * A line section header: the vertex set is completed, the first one sets the sink's direction
     * (rule 1 of design section 8.4), later ones only switch the kind.
     * @param h - the header
     * @param line - the line number
     * @param section - the parser state for the section
     * @param kind - the direction of its lines
     */
    private lineSection(h: SectionHeader, line: number, section: Section, kind: EdgeKind): void {
        if (this.vertexCount < 0) {
            this.report.fail(
                PAJEK_ISSUE.NO_VERTICES,
                `*${h.keyword} before *Vertices; a Pajek network starts with *Vertices`,
                { line },
            );
        }
        this.finishVertices();
        if (!this.sawLineSection) {
            this.sawLineSection = true;
            this.resolver.setHeader(kind === "directed", { line });
        }
        this.section = section;
        this.kind = kind;
        this.relation = h.relation === null ? null : (h.name ?? String(h.relation));
    }

    /**
     * Refuse a second network: an issue, then ImportError with the report so far.
     * @param keyword - the header keyword
     * @param line - the line number
     */
    private refuseSecondNetwork(keyword: string, line: number): never {
        const message = `*${keyword} starts a second network; the importer reads one network per file`;
        this.report.error("unsupported", PAJEK_ISSUE.MULTIPLE_NETWORKS, message, { line });
        throw this.report.abort(message, { code: PAJEK_ISSUE.MULTIPLE_NETWORKS, line });
    }

    /**
     * Checks that run when a section ends.
     */
    private endSection(): void {
        if (this.section === "matrix" && this.matrixRows !== this.vertexCount) {
            this.report.error(
                "validation-error",
                PAJEK_ISSUE.MATRIX_ROWS,
                `*Matrix has ${this.matrixRows} row(s); *Vertices declares ${this.vertexCount}`,
            );
        }
        this.section = "none";
    }

    // ============================================================ vertices

    /**
     * Fix the vertex numbering base and, under nodeIdFrom "id" / "index", create the declared
     * vertices in number order so the id map is identity.
     * @param base - 0 or 1
     * @param line - the line number
     */
    private setBase(base: 0 | 1, line: number): void {
        this.base = base;
        this.baseKnown = true;
        if (base === 0) {
            this.report.warning(
                "coercion",
                PAJEK_ISSUE.ZERO_BASED,
                "vertex numbering starts at 0; Pajek numbers vertices from 1",
                { line },
            );
        }
    }

    /**
     * Push vertices base..base+N-1 in order (nodeIdFrom "id" / "index"), so the id map is identity;
     * called on the first vertex line (once it is known that the lines carry no `graphty_originalId`
     * parameter to restore) or when the vertex section ends without one.
     */
    private createNodes(): void {
        if (this.nodesCreated || this.idOfPos !== null) {
            return;
        }
        this.nodesCreated = true;
        const n = this.vertexCount;
        const { sink } = this;
        for (let pos = 0; pos < n; pos++) {
            this.indexOfPos[pos] = sink.addNode(this.idOfNumber(pos + this.base));
        }
        this.report.counts.nodes += n;
    }

    /**
     * The id of a vertex number under nodeIdFrom "id" (the number coerced by the ids rule) or
     * "index" (the position).
     * @param k - the vertex number
     * @returns the id
     */
    private idOfNumber(k: number): NodeId {
        if (this.options.nodeIdFrom === "index") {
            return k - this.base;
        }
        switch (this.options.ids) {
            case "canonical":
            case "number":
                return k;
            case "keep":
            case "string":
                return String(k);
            default:
                return coerceIdText(String(k), this.options.ids);
        }
    }

    /**
     * The id of a vertex position, for addEdge.
     * @param pos - the vertex number minus the base
     * @returns the id
     */
    private idAt(pos: number): NodeId {
        if (this.idOfPos !== null) {
            return this.idOfPos[pos];
        }
        return this.idOfNumber(pos + this.base);
    }

    /**
     * Complete the vertex set once the vertex section is over: vertices without a line are
     * created (under "label" or a restored `graphty_originalId` with their number as the id), and
     * a partial list is reported.
     */
    private finishVertices(): void {
        if (this.verticesDone) {
            return;
        }
        this.verticesDone = true;
        if (!this.baseKnown) {
            this.setBase(1, 0);
        }
        const n = this.vertexCount;
        this.createNodes();
        if (this.idOfPos !== null) {
            for (let pos = 0; pos < n; pos++) {
                if (this.idOfPos[pos] === undefined) {
                    this.numberedNode(this.idOfPos, pos);
                }
            }
        }
        // a partial list: fewer lines than declared, counting the lines skipped after an error
        const lines = this.seenCount + this.report.counts.skippedNodes;
        if (this.seenCount > 0 && lines < n) {
            this.report.error(
                "validation-error",
                PAJEK_ISSUE.VERTEX_COUNT,
                `*Vertices declares ${n} vertices but ${lines} vertex line(s) were read; the others have no label`,
            );
        }
    }

    /**
     * A vertex line: `number [label] [x y [z]] [shape] [key value ...] [interval]`.
     * @param text - the line
     * @param line - the line number
     */
    private vertexLine(text: string, line: number): void {
        const tokens = this.tokens(text, "node");
        if (!isVertexNumber(tokens[0])) {
            this.report.counts.skippedNodes++;
            throw new LineError(
                "parse-error",
                PAJEK_ISSUE.VERTEX_LINE,
                `a vertex line starts with a vertex number, not "${tokens[0]}"`,
            );
        }
        const k = Number(tokens[0]);
        if (!this.baseKnown) {
            this.setBase(k === 0 ? 0 : 1, line);
        }
        const pos = k - this.base;
        if (pos < 0 || pos >= this.vertexCount) {
            this.report.counts.skippedNodes++;
            throw new LineError(
                "validation-error",
                PAJEK_ISSUE.VERTEX_RANGE,
                `vertex ${k} is outside ${this.base}..${this.base + this.vertexCount - 1}`,
            );
        }
        // parse everything before writing anything
        let i = 1;
        const label = tokens.length > 1 ? tokens[1] : null;
        if (label !== null) {
            i = 2;
        }
        let x = 0;
        let y = 0;
        let z = 0;
        let dims = 0;
        if (i + 1 < tokens.length && isNumericText(tokens[i]) && isNumericText(tokens[i + 1])) {
            x = Number(tokens[i]);
            y = Number(tokens[i + 1]);
            i += 2;
            dims = 2;
            if (i < tokens.length && isNumericText(tokens[i])) {
                z = Number(tokens[i]);
                i++;
                dims = 3;
            }
        } else if (i < tokens.length && isNumericText(tokens[i])) {
            this.report.counts.skippedNodes++;
            throw new LineError(
                "parse-error",
                PAJEK_ISSUE.VERTEX_LINE,
                `vertex ${k}: a single coordinate "${tokens[i]}"; coordinates are x y [z]`,
            );
        }
        let shape: string | null = null;
        if (i < tokens.length && SHAPES.has(tokens[i])) {
            shape = tokens[i];
            i++;
        }
        const extras = this.extras(tokens, i, "node");
        const restored = this.restoredId(extras);
        this.createNodes();
        // write
        const repeated = this.seen[pos] !== 0;
        if (repeated) {
            this.report.warning(
                "merged",
                PAJEK_ISSUE.DUPLICATE_NODE,
                `vertex ${k} has more than one line; the later values overwrite`,
                { line, element: String(k) },
            );
        } else {
            this.seen[pos] = 1;
            this.seenCount++;
        }
        let index: number;
        if (this.idOfPos !== null && !repeated && this.options.nodeIdFrom === "label") {
            index = this.deferredNode(this.idOfPos, pos, k, label, line, "label");
        } else if (this.idOfPos !== null && this.idOfPos[pos] === undefined) {
            // restore mode: the earlier positions without a line yet get their numbers first, so the
            // index order is the vertex number order whatever the line order (the identity id map
            // of an ordered file); then this vertex with its original id, or its number
            for (let p = 0; p < pos; p++) {
                if (this.idOfPos[p] === undefined) {
                    this.numberedNode(this.idOfPos, p);
                }
            }
            index = this.deferredNode(this.idOfPos, pos, k, restored, line, "originalId");
        } else {
            index = this.indexOfPos[pos];
            if (restored !== null && !repeated) {
                // an out-of-order line: the vertex already exists under its number
                this.report.warning(
                    "coercion",
                    PAJEK_ISSUE.ORIGINAL_ID_UNRESTORED,
                    `vertex ${k} carries ${ORIGINAL_ID_KEY} ${JSON.stringify(restored)} but a later vertex's line came first and created it under its number; the original id is not restored`,
                    { line, element: String(k) },
                );
            }
        }
        const { sink } = this;
        if (label !== null) {
            if (this.labelHandle === INVALID_INDEX) {
                this.labelHandle = declareResolved(sink, "node", LABEL_DECL, this.report, { line }).handle;
            }
            sink.setNodeValue(this.labelHandle, index, label);
        }
        if (dims > 0) {
            if (this.positionHandle === INVALID_INDEX) {
                this.coordDims = dims;
                this.positionHandle = declareResolved(sink, "node", positionDecl(dims), this.report, { line }).handle;
            } else if (dims !== this.coordDims) {
                this.report.warnOnce(
                    "validation-error",
                    PAJEK_ISSUE.COORD_DIMS,
                    `vertex lines mix ${this.coordDims} and ${dims} coordinates; the position column records sourceDims ${this.coordDims}`,
                    { line, element: String(k) },
                );
            }
            sink.setNodeValue(this.positionHandle, index, [x, y, z]);
        }
        if (shape !== null) {
            if (this.shapeHandle === INVALID_INDEX && this.nodeParams.has(SHAPE_COLUMN)) {
                // an earlier `shape "<text>"` parameter owns the column: the keyword is written there
                this.writeParameter("node", SHAPE_COLUMN, index, shape);
            } else {
                if (this.shapeHandle === INVALID_INDEX) {
                    this.shapeHandle = declareResolved(sink, "node", SHAPE_DECL, this.report, { line }).handle;
                }
                sink.setNodeValue(this.shapeHandle, index, shape);
            }
        }
        for (let p = 0; p < extras.keys.length; p++) {
            this.writeParameter("node", extras.keys[p], index, extras.values[p]);
        }
        if (extras.spells !== null) {
            if (this.nodeSpellsHandle === INVALID_INDEX) {
                this.nodeSpellsHandle = declareResolved(sink, "node", spellsDecl(), this.report, { line }).handle;
            }
            sink.setNodeValue(this.nodeSpellsHandle, index, extras.spells);
        }
    }

    /**
     * Whether a `graphty_originalId` parameter is restored as the vertex id: restoreMangledIds
     * under nodeIdFrom "id" (under "label" / "index" the parameter stays an ordinary column).
     * @returns true when the vertex lines may carry restorable ids
     */
    private restoresIds(): boolean {
        return this.options.restoreMangledIds && this.options.nodeIdFrom === "id";
    }

    /**
     * Push a vertex whose id is its number (restore mode, a position whose line has not come or
     * carries no original id; or the end of the vertex section under "label").
     * @param ids - the id per vertex position
     * @param pos - the vertex position
     */
    private numberedNode(ids: NodeId[], pos: number): void {
        const id = this.idOfNumber(pos + this.base);
        ids[pos] = id;
        this.indexOfPos[pos] = this.sink.addNode(id);
        this.report.counts.nodes++;
    }

    /**
     * The `graphty_originalId` parameter of a vertex line, taken out of the extras when it is to be
     * restored as the id (restoreMangledIds, under nodeIdFrom "id"); otherwise null and the
     * parameter stays an ordinary column.
     * @param extras - the parsed extras
     * @returns the original id text, or null
     */
    private restoredId(extras: RowExtras): string | null {
        if (!this.restoresIds()) {
            return null;
        }
        const at = extras.keys.indexOf(ORIGINAL_ID_KEY);
        if (at < 0) {
            return null;
        }
        const [text] = extras.values.splice(at, 1);
        extras.keys.splice(at, 1);
        return text;
    }

    /**
     * Push the node of a vertex line whose id comes from the line itself: its label under
     * nodeIdFrom "label", or its restored `graphty_originalId` (the number when the line carries
     * none), reporting a merge when the id was seen before.
     * @param ids - the id per vertex position
     * @param pos - the vertex position
     * @param k - the vertex number
     * @param text - the label or original id text, or null
     * @param line - the line number
     * @param from - which text the id comes from
     * @returns the node index
     */
    private deferredNode(
        ids: NodeId[],
        pos: number,
        k: number,
        text: string | null,
        line: number,
        from: "label" | "originalId",
    ): number {
        const id = text === null ? this.idOfNumber(k) : this.coercer.text(text);
        const before = this.sink.indexOf(id);
        const index = this.sink.addNode(id);
        ids[pos] = id;
        this.indexOfPos[pos] = index;
        this.report.counts.nodes++;
        if (before !== INVALID_INDEX) {
            this.report.warning(
                "merged",
                from === "label" ? PAJEK_ISSUE.LABEL_MERGED : PAJEK_ISSUE.ORIGINAL_ID_MERGED,
                from === "label"
                    ? `vertex ${k} has the label ${JSON.stringify(String(id))} of an earlier vertex; both are one node`
                    : `vertex ${k} has the ${ORIGINAL_ID_KEY} ${JSON.stringify(String(id))} of an earlier vertex; both are one node`,
                { line, element: String(k) },
            );
        }
        return index;
    }

    // ============================================================ lines

    /**
     * An arc or edge line: `u v [value] [key value ...] [interval]`.
     * @param text - the line
     * @param line - the line number
     */
    private edgeLine(text: string, line: number): void {
        const tokens = this.tokens(text, "edge");
        if (tokens.length < 2 || !isVertexNumber(tokens[0]) || !isVertexNumber(tokens[1])) {
            this.report.counts.skippedEdges++;
            throw new LineError(
                "parse-error",
                PAJEK_ISSUE.LINE,
                `a line starts with two vertex numbers, not "${tokens.slice(0, 2).join(" ")}"`,
            );
        }
        const u = this.endpoint(Number(tokens[0]));
        const v = this.endpoint(Number(tokens[1]));
        let i = 2;
        let valueText: string | null = null;
        if (i < tokens.length && isNumericText(tokens[i])) {
            valueText = tokens[i];
            i++;
        }
        const extras = this.extras(tokens, i, "edge");
        const { weightFrom } = this.options;
        let weight: number | undefined;
        let valueIsWeight = false;
        if (isWeightField(VALUE_FIELD, weightFrom)) {
            valueIsWeight = true;
            if (valueText !== null) {
                weight = this.weightOf(valueText);
            }
        } else if (weightFrom !== null) {
            const at = extras.keys.indexOf(weightFrom);
            if (at >= 0) {
                weight = this.weightOf(extras.values[at]);
                extras.keys.splice(at, 1);
                extras.values.splice(at, 1);
            }
        }
        const e = this.pushEdge(u, v, weight, line);
        if (valueText !== null && !valueIsWeight) {
            // inferred like every parameter, so a re-import of the exporter's `value` parameter agrees
            this.writeParameter("edge", VALUE_COLUMN, e, valueText);
        }
        this.writeEdgeExtras(e, extras);
    }

    /**
     * An adjacency list line of `*Arcslist` / `*Edgeslist`: `v n1 n2 ...`.
     * @param text - the line
     * @param line - the line number
     */
    private listLine(text: string, line: number): void {
        const tokens = this.tokens(text, "edge");
        for (const token of tokens) {
            if (!isVertexNumber(token)) {
                this.report.counts.skippedEdges++;
                throw new LineError(
                    "parse-error",
                    PAJEK_ISSUE.LINE,
                    `an adjacency list holds vertex numbers only, not "${token}"`,
                );
            }
        }
        const u = this.endpoint(Number(tokens[0]));
        for (let i = 1; i < tokens.length; i++) {
            const v = this.endpoint(Number(tokens[i]));
            const e = this.pushEdge(u, v, undefined, line);
            this.writeRelation(e);
        }
    }

    /**
     * A `*Matrix` row: N values, every non-zero entry an arc from the row's vertex with that value.
     * @param text - the line
     * @param line - the line number
     */
    private matrixLine(text: string, line: number): void {
        const tokens = this.tokens(text, "edge");
        const row = this.matrixRows;
        this.matrixRows++;
        if (row >= this.vertexCount) {
            throw new LineError(
                "validation-error",
                PAJEK_ISSUE.MATRIX_ROWS,
                `*Matrix row ${row + 1} is beyond the ${this.vertexCount} declared vertices`,
            );
        }
        if (tokens.length !== this.vertexCount) {
            throw new LineError(
                "parse-error",
                PAJEK_ISSUE.LINE,
                `*Matrix row ${row + 1} has ${tokens.length} value(s); *Vertices declares ${this.vertexCount}`,
            );
        }
        for (const token of tokens) {
            if (!isNumericText(token)) {
                throw new LineError(
                    "parse-error",
                    PAJEK_ISSUE.LINE,
                    `*Matrix row ${row + 1}: "${token}" is not a number`,
                );
            }
        }
        for (let j = 0; j < tokens.length; j++) {
            const value = Number(tokens[j]);
            if (value === 0) {
                continue;
            }
            const e = this.pushEdge(row, j, value, line);
            this.writeRelation(e);
        }
    }

    /**
     * Push one line through the direction resolver, counting the logical edges it produced.
     * @param u - the source position
     * @param v - the target position
     * @param weight - the weight, or undefined for none
     * @param line - the line number
     * @returns the primary logical edge index
     */
    private pushEdge(u: number, v: number, weight: number | undefined, line: number): number {
        const { sink } = this;
        const before = sink.edgeCount;
        const e = this.resolver.addEdge(this.idAt(u), this.idAt(v), this.kind, weight, { line });
        this.report.counts.edges += sink.edgeCount - before;
        return e;
    }

    /**
     * Write the relation, parameters and spells of a line on its primary half.
     * @param e - the edge index
     * @param extras - the parsed extras
     */
    private writeEdgeExtras(e: number, extras: RowExtras): void {
        const { sink } = this;
        this.writeRelation(e);
        for (let p = 0; p < extras.keys.length; p++) {
            this.writeParameter("edge", extras.keys[p], e, extras.values[p]);
        }
        if (extras.spells !== null) {
            if (this.edgeSpellsHandle === INVALID_INDEX) {
                this.edgeSpellsHandle = declareResolved(sink, "edge", spellsDecl(), this.report).handle;
            }
            sink.setEdgeValue(this.edgeSpellsHandle, e, extras.spells);
        }
    }

    /**
     * Write the current section's relation name on an edge, when the section has one.
     * @param e - the edge index
     */
    private writeRelation(e: number): void {
        if (this.relation === null) {
            return;
        }
        if (this.relationHandle === INVALID_INDEX) {
            this.relationHandle = declareResolved(this.sink, "edge", RELATION_DECL, this.report).handle;
        }
        this.sink.setEdgeValue(this.relationHandle, e, this.relation);
    }

    /**
     * A line endpoint: the vertex number checked against the declared range; a reference outside
     * it is a missing-value error and the line is skipped (Pajek's `*Vertices N` bounds the id space,
     * so addMissingNodes does not apply).
     * @param k - the vertex number
     * @returns the vertex position
     */
    private endpoint(k: number): number {
        const pos = k - this.base;
        if (pos < 0 || pos >= this.vertexCount) {
            this.report.counts.skippedEdges++;
            throw new LineError(
                "missing-value",
                PAJEK_ISSUE.UNKNOWN_NODE,
                `vertex ${k} is outside ${this.base}..${this.base + this.vertexCount - 1}`,
            );
        }
        return pos;
    }

    /**
     * The weight of a line value text.
     * @param text - the text
     * @returns the weight; E_INVALID_WEIGHT is recorded by the caller's catch
     */
    private weightOf(text: string): number | undefined {
        try {
            return parseWeightText(text);
        } catch (err) {
            this.report.counts.skippedEdges++;
            throw err;
        }
    }

    // ============================================================ shared

    /**
     * Tokenize a data line.
     * @param text - the line
     * @param domain - what is skipped on an unterminated quote
     * @returns the tokens (at least one: blank lines never reach here)
     */
    private tokens(text: string, domain: "node" | "edge"): string[] {
        const tokens = tokenize(text);
        if (tokens === null) {
            if (domain === "node") {
                this.report.counts.skippedNodes++;
            } else {
                this.report.counts.skippedEdges++;
            }
            throw new LineError("parse-error", PAJEK_ISSUE.UNTERMINATED_QUOTE, "a double quote is not closed");
        }
        return tokens;
    }

    /**
     * The `key value` parameters and the interval token at the end of a row, validated before
     * anything is written.
     * @param tokens - the row's tokens
     * @param start - the first token to read
     * @param domain - what is skipped when the row is malformed
     * @returns the parsed extras
     */
    private extras(tokens: readonly string[], start: number, domain: "node" | "edge"): RowExtras {
        const extras: RowExtras = { keys: [], values: [], spells: null };
        for (let i = start; i < tokens.length;) {
            const token = tokens[i];
            if (isIntervalToken(token)) {
                try {
                    extras.spells = parseIntervals(token);
                } catch (err) {
                    this.skip(domain);
                    throw new LineError(
                        "validation-error",
                        PAJEK_ISSUE.INTERVAL,
                        err instanceof Error ? err.message : String(err),
                    );
                }
                i++;
                continue;
            }
            if (i + 1 >= tokens.length) {
                this.skip(domain);
                throw new LineError(
                    "parse-error",
                    domain === "node" ? PAJEK_ISSUE.VERTEX_LINE : PAJEK_ISSUE.LINE,
                    `parameter "${token}" has no value`,
                );
            }
            extras.keys.push(token);
            extras.values.push(tokens[i + 1]);
            i += 2;
        }
        return extras;
    }

    /**
     * Count a skipped element.
     * @param domain - node or edge
     */
    private skip(domain: "node" | "edge"): void {
        if (domain === "node") {
            this.report.counts.skippedNodes++;
        } else {
            this.report.counts.skippedEdges++;
        }
    }

    /**
     * Write one `key value` parameter cell through the column's text writer (the 5.1 grammar per
     * column, the lexical form kept). A `shape` parameter next to the shape keyword column goes
     * into that column (a dict takes any text), and a shape keyword next to a `shape` parameter
     * column into that one, so a file mixing the two spellings reads as one column.
     * @param domain - node or edge
     * @param key - the parameter key (the column name)
     * @param row - the node or edge index
     * @param text - the value text
     */
    private writeParameter(domain: "node" | "edge", key: string, row: number, text: string): void {
        if (domain === "node" && key === SHAPE_COLUMN && this.shapeHandle !== INVALID_INDEX) {
            this.sink.setNodeValue(this.shapeHandle, row, text);
            return;
        }
        const writers = domain === "node" ? this.nodeParams : this.edgeParams;
        let writer = writers.get(key);
        if (writer === undefined) {
            writer = new TextCellWriter(key, domain, this.sink, this.report);
            writers.set(key, writer);
        }
        writer.write(row, text);
    }
}

/**
 * Resolve the firstVertex option.
 * @param value - the caller's value
 * @returns 0, 1 or "auto"; E_UNSUPPORTED for anything else
 */
function firstVertexOption(value: unknown): 0 | 1 | "auto" {
    if (value === undefined) {
        return "auto";
    }
    if (FIRST_VERTEX_VALUES.has(value)) {
        return value as 0 | 1 | "auto";
    }
    throw new GraphFormatError(
        "E_UNSUPPORTED",
        `option firstVertex: ${JSON.stringify(value) ?? typeof value} is not 0, 1 or "auto"`,
        {
            option: "firstVertex",
            found: value,
        },
    );
}

const HEAD_PATTERN = /^\s*\*(vertices|network)\b/i;

/** The Pajek NET importer. */
export const pajekImporter: GraphImporter<PajekImportOptions> = Object.freeze({
    format: "pajek",
    extensions: Object.freeze([".net", ".paj"]),
    mimeTypes: Object.freeze(["text/x-pajek", "text/plain"]),

    /**
     * Confidence that the input is a Pajek network: it starts with `*Vertices` (or `*Network`).
     * @param head - the first bytes
     * @returns 0.9 for a `*Vertices` start, 0.8 for `*Network`, 0 otherwise
     */
    sniff(head: Uint8Array): number {
        const text = new TextDecoder("utf-8").decode(head.subarray(0, Math.min(head.byteLength, 512)));
        const match = HEAD_PATTERN.exec(text.startsWith(String.fromCharCode(0xfeff)) ? text.slice(1) : text);
        if (match === null) {
            return 0;
        }
        return match[1].toLowerCase() === "vertices" ? 0.9 : 0.8;
    },

    /**
     * Read a Pajek network into the sink.
     * @param input - the text, bytes or stream
     * @param sink - the sink
     * @param options - common and Pajek options
     * @returns the report
     */
    async import(
        input: ImportInput,
        sink: GraphSink,
        options?: PajekImportOptions & CommonImportOptions,
    ): Promise<ImportReport> {
        const resolved = resolveImportOptions(options, DEFAULTS);
        const firstVertex = firstVertexOption(options?.firstVertex);
        const report = new ImportReportBuilder("pajek", resolved.errorLimit);
        reportSinkOptions(sink, options, report);
        reportUnusedOptions(options, report, USED_OPTIONS);
        const parser = new PajekParser(sink, report, resolved, firstVertex);
        const reader = new LineReader(input, report, resolved);
        let sinceCheck = 0;
        for await (const text of reader) {
            parser.line(text, reader.line);
            if (++sinceCheck >= ABORT_CHECK_INTERVAL) {
                sinceCheck = 0;
                throwIfAborted(resolved.signal);
            }
        }
        parser.finish();
        throwIfAborted(resolved.signal);
        return report.finish();
    },
});
