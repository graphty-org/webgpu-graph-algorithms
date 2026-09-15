/**
 * The io contract types of @graphty/graph-io (design section 12.4, normative): what every importer
 * and exporter implements and what every caller of one programs against. They are declared here
 * rather than in the core because ImportInput and CommonImportOptions reference ReadableStream and
 * AbortSignal, which need the DOM lib (or @types/node >= 18); the core's public types reference only
 * ES2020 globals (decision D-IO-TYPES). Every declaration is transcribed verbatim from the design;
 * the behaviour behind each option is specified in sections 8.4 (import), 8.5 (export) and 8.6
 * (report and error aggregation).
 */

import {
    type Dtype,
    type DuplicatePolicy,
    GraphFormatError,
    type GraphSink,
    type GraphSnapshot,
    type IdCoercion,
} from "@graphty/graph-format";

/**
 * What an importer reads (design section 8.4): whole text, whole bytes, a byte stream (a browser
 * `File.stream()`, a fetch body) or an async iterable of text or byte chunks. Bytes are decoded as
 * UTF-8 with `fatal: true`, so an invalid sequence is a parse-error and never a silent U+FFFD that
 * could alias two ids.
 */
export type ImportInput = string | Uint8Array | ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>;

/**
 * The options every importer accepts next to its format-specific ones (design section 8.4). The
 * builder-policy fields (`addMissingNodes`, `duplicateEdges`, `selfLoops`, `weightDtype`) seed the
 * registry's builder; on a caller's sink they are read back from `sink.options` and every option the
 * sink cannot honour is reported.
 */
export interface CommonImportOptions {
    /**
     * Id coercion rule applied before an id reaches the sink (design section 4.1): "canonical" by default for
     * text-cell formats, "keep" for JSON.
     */
    ids?: IdCoercion | undefined;
    /** Which field becomes the node id; "id" by default; "label" / "index" resolve the GML / Pajek / d3 ambiguity. */
    nodeIdFrom?: "id" | "label" | "index" | undefined;
    /** Whether an edge may reference an undeclared node (default true; the GEXF importer defaults false for edges). */
    addMissingNodes?: boolean | undefined;
    /** The builder's duplicate-edge policy seed; default "keep". */
    duplicateEdges?: DuplicatePolicy | undefined;
    /** The builder's self-loop policy seed; default "keep". */
    selfLoops?: "keep" | "drop" | "error" | undefined;
    /** What to do with a file whose edges disagree on direction (design section 3.6); default "expand". */
    onMixedDirection?: "expand" | "directed" | "undirected" | "error" | undefined;
    /** The direction assumed for a file that declares none (GEXF: undirected per spec). */
    defaultDirected?: boolean | undefined;
    /** The attribute that becomes THE weight (per-format default: "weight", GML "value"); null = unweighted. */
    weightFrom?: string | null | undefined;
    /** Weight staging precision; "f64" for every importer by default so 0.1 and 16777217 survive. */
    weightDtype?: "f32" | "f64" | undefined;
    /** How a declared `long` column is stored: "f64" (default) or "string" (design section 5.1). */
    long?: "f64" | "string" | undefined;
    /** Restore ids mangled by sanitizeIds "mangle" from the graphty:originalId attribute (default true). */
    restoreMangledIds?: boolean | undefined;
    /** GraphML / JGF hyperedges: refuse, skip with a report entry (default), or expand to a star / clique. */
    hyperedges?: "error" | "skip" | "star" | "clique" | undefined;
    /** Errors tolerated before the importer aborts with E_IMPORT (default 100). */
    errorLimit?: number | undefined;
    /** Cancellation; the importer stops between chunks and rejects with the signal's reason. */
    signal?: AbortSignal | undefined;
    /** Progress in bytes; `bytesTotal` is known for in-memory input only. */
    onProgress?: ((bytesDone: number, bytesTotal?: number) => void) | undefined;
}

/**
 * An importer plugin (design section 8.4): pushes scalars into the caller's sink in one pass and
 * never freezes. `Opts` is its format-specific option set; the default `unknown` lets a caller pass
 * the common options to an importer typed without one.
 */
export interface GraphImporter<Opts = unknown> {
    /** The format name: "gexf", "graphml", "gml", "dot", "pajek", "csv", "json", "neo4j". */
    readonly format: string;
    /** File extensions with the leading dot. */
    readonly extensions: readonly string[];
    /** MIME types the format is served as. */
    readonly mimeTypes: readonly string[];
    /**
     * Confidence that `head` is this format, for the registry's sniff().
     * @param head - the first bytes of the input
     * @returns a confidence in 0..1
     */
    sniff?(head: Uint8Array): number;
    /**
     * Read `input` into `sink`; per-element errors are aggregated into the report until the error
     * limit, then the importer throws ImportError with the partial report.
     * @param input - the text, bytes or stream to read
     * @param sink - the builder (or a recording sink) to push into
     * @param options - format-specific and common options
     * @returns the import report
     */
    import(input: ImportInput, sink: GraphSink, options?: Opts & CommonImportOptions): Promise<ImportReport>;
}

/**
 * What a format can express without loss (design section 8.5): the fidelity matrix of research
 * note 07 section 9 as a table, and the acceptance test list for check().
 */
export interface ExportCapabilities {
    /** Directed and undirected edges in one file. */
    readonly mixedDirection: boolean;
    /** Parallel edges. */
    readonly multiEdges: boolean;
    /** Self-loops. */
    readonly selfLoops: boolean;
    /** Whether edge ids are required (generated when absent), optional or unsupported. */
    readonly edgeIds: "required" | "optional" | "none";
    /** Which node ids can be written unchanged. */
    readonly idCharset: "any" | "nmtoken" | "integer" | "dense-1-based";
    /** The column dtypes the format keeps as declared. */
    readonly dtypes: readonly Dtype[];
    /** Multi-component (stride) columns. */
    readonly components: boolean;
    /** List columns. */
    readonly lists: boolean;
    /** Nested json columns. */
    readonly json: boolean;
    /** Declared defaults. */
    readonly defaults: boolean;
    /** Declared enumerations (GEXF options). */
    readonly options: boolean;
    /** Containment (parent / parents roles). */
    readonly hierarchy: boolean;
    /** Temporal support level. */
    readonly temporal: "none" | "intervals" | "spells" | "dynamic-values";
    /** Graph-level attributes. */
    readonly graphAttributes: boolean;
    /** The position role. */
    readonly positions: boolean;
    /** The visual roles (color, size, shape, thickness). */
    readonly viz: boolean;
}

/** One thing an exporter cannot represent, reported by check() before anything is written (design section 8.5). */
export interface LossNote {
    /** A stable code such as "W_OPEN_INTERVAL". */
    readonly code: string;
    /** A plain-ASCII human-readable explanation. */
    readonly message: string;
    /** The affected column, or null when the note is not about a column. */
    readonly column: string | null;
    /** How many rows or elements are affected, or null when not counted. */
    readonly count: number | null;
}

/** The options every exporter accepts next to its format-specific ones (design section 8.5). */
export interface CommonExportOptions {
    /** "error" (default): never rename a node; "mangle": rewrite ids the format cannot hold and keep the original. */
    sanitizeIds?: "error" | "mangle" | undefined;
    /** For formats without mixed-direction support: refuse (default) or write every edge one way. */
    onMixedDirection?: "error" | "directed" | "undirected" | undefined;
}

/**
 * An exporter plugin (design section 8.5): iterates nodes and logical edges (never arcs) in index
 * order and reads the role columns of section 3.6 / 3.7 to fold expanded pairs and emit explicit
 * weights only.
 */
export interface GraphExporter<Opts = unknown> {
    /** The format name. */
    readonly format: string;
    /** What the format can express. */
    readonly capabilities: ExportCapabilities;
    /**
     * Pre-flight: what export() would lose, without writing anything.
     * @param snapshot - the snapshot to check
     * @param options - format-specific and common options
     * @returns the loss notes, empty when the export is exact
     */
    check(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): readonly LossNote[];
    /**
     * Write the snapshot as UTF-8 chunks.
     * @param snapshot - the snapshot to write
     * @param options - format-specific and common options
     * @returns the encoded chunks
     */
    export(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): AsyncIterable<Uint8Array>;
    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot to write
     * @param options - format-specific and common options
     * @returns the whole document
     */
    exportToString(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): Promise<string>;
}

/** The categories of an ImportIssue (design section 8.6). */
export type IssueCategory =
    "parse-error" | "missing-value" | "validation-error" | "unsupported" | "precision" | "coercion" | "merged";

/** One problem found while importing (design section 8.6). */
export interface ImportIssue {
    /** The category. */
    readonly category: IssueCategory;
    /** Errors count toward the error limit; warnings do not. */
    readonly severity: "error" | "warning";
    /** A stable code such as "E_UNKNOWN_NODE" or "W_WIDENED". */
    readonly code: string;
    /** A plain-ASCII human-readable message. */
    readonly message: string;
    /** The 1-based source line, or null when unknown. */
    readonly line: number | null;
    /** The element (a node or edge id, an attribute name), or null when unknown. */
    readonly element: string | null;
}

/** What an import produced besides the sink's contents (design section 8.6). */
export interface ImportReport {
    /** The importer's format name. */
    readonly format: string;
    /** Element counts. */
    readonly counts: {
        /** Nodes pushed. */
        readonly nodes: number;
        /** Logical edges pushed (both halves of an expanded edge count). */
        readonly edges: number;
        /** Nodes skipped after an error. */
        readonly skippedNodes: number;
        /** Edges skipped after an error. */
        readonly skippedEdges: number;
        /** Edges expanded under onMixedDirection "expand" (design section 3.6). */
        readonly expandedMixed: number;
    };
    /** Every issue recorded, in order. */
    readonly issues: readonly ImportIssue[];
    /** Issues with severity "error". */
    readonly errorCount: number;
    /** Issues with severity "warning". */
    readonly warningCount: number;
    /** Whether the error limit was reached and the import aborted. */
    readonly truncated: boolean;
    /** What the importer could not represent. */
    readonly lossy: readonly LossNote[];
    /** Wall time of the parse phase; the caller's freeze reports separately. */
    readonly durationMs: number;
}

/**
 * The error an importer throws when the error limit is reached or the input cannot be read at all
 * (design section 8.6): a GraphFormatError with code "E_IMPORT" (reserved in the core's union so
 * `err.code === "E_IMPORT"` narrows) carrying the partial report.
 */
export class ImportError extends GraphFormatError {
    /** The report as it stood when the import aborted. */
    readonly report: ImportReport;

    /**
     * Create an ImportError.
     * @param message - a plain-ASCII human-readable message
     * @param report - the partial report
     * @param details - optional machine-readable context
     */
    constructor(message: string, report: ImportReport, details?: Readonly<Record<string, unknown>>) {
        super("E_IMPORT", message, details);
        this.name = "ImportError";
        this.report = report;
    }
}
