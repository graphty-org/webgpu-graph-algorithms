/**
 * The CSV / TSV importer (design sections 8.4 and 8.6; research note 07 section 2.6): a streaming
 * edge-list reader for the generic (`source,target[,weight,...]`), Gephi (`Source,Target,Type,Id,
 * Label,Weight,...`) and headerless (`u v [w]`) dialects, with an optional node table merged by id.
 *
 * - The delimiter is sniffed from a preview unless given; LF, CRLF and lone-CR files all read.
 * - The first row is a header when it holds a known column name or when it is all text over a
 *   numeric second row (`header: "auto"`); a headerless file is positional: source, target,
 *   weight (when `weightFrom` is not null), then `column4`... as attributes.
 * - Endpoint and node ids are text cells coerced by ONE rule per import call (`ids`, default
 *   "canonical": `"1"` becomes the number 1, `"01"` stays a string) over the node table and the
 *   edge table alike, so they agree across the two files.
 * - Direction is per row in the Gephi dialect (`Type` = Directed / Undirected / Mutual, blank =
 *   `defaultDirected`) and `defaultDirected` (true) otherwise; the first edge row sets the sink's
 *   direction and later rows that differ go through `onMixedDirection` (expansion by default).
 * - The weight column (`weightFrom`, default "weight", matched case-insensitively) is parsed per
 *   row; a blank cell means "no weight given" and the edge is pushed without one.
 * - Every other column is an attribute: cells are parsed by the fixed lexical grammar of design
 *   section 5.1 and the sink infers the column dtype (widening per column, never per cell); an
 *   all-text column of low cardinality becomes a dict (design section 5.4); an `id` column of the
 *   edge table is the edge id (role id, unique); a `label` column is the label (role label).
 * - Per-row problems (wrong field count, blank endpoint, invalid weight, bad Type, refused id)
 *   are recorded and the row skipped; the import aborts with ImportError once `errorLimit` is
 *   exceeded, on a malformed or unterminated quoted field, on an empty input and on a header
 *   without endpoint (or id) columns.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphSink,
    INVALID_INDEX,
    type NodeId,
} from "@graphty/graph-format";

import { declareResolved, RENAMED_CODE, uniqueColumnName } from "../../common/attributes.js";
import {
    DUPLICATE_EDGE_ID_CODE as SHARED_DUPLICATE_EDGE_ID_CODE,
    DUPLICATE_NODE_CODE as SHARED_DUPLICATE_NODE_CODE,
    EMPTY_INPUT_CODE as SHARED_EMPTY_INPUT_CODE,
    ID_MERGED_CODE as SHARED_ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE as SHARED_MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE as SHARED_MISSING_ID_CODE,
    ROLE_TAKEN_CODE as SHARED_ROLE_TAKEN_CODE,
} from "../../common/codes.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { IdCoercer } from "../../common/ids.js";
import { throwIfAborted } from "../../common/input.js";
import {
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { parseWeightText } from "../../common/weights.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    type CsvColumnRef,
    EDGE_ID_NAMES,
    findColumn,
    headerNames,
    ID_NAMES,
    LABEL_NAMES,
    looksLikeHeader,
    positionalNames,
    resolveColumnRef,
    SOURCE_NAMES,
    TARGET_NAMES,
    TYPE_NAME,
} from "./header.js";
import { type CsvReaderOptions, CsvRecordReader, sniffDelimiter, sniffNewline } from "./records.js";
import { InferredColumn } from "./values.js";

/** The format-specific options of the CSV importer. */
export interface CsvImportOptions {
    /** The field delimiter; sniffed from the first rows when omitted (`,`, tab, `;`, `|`, space). */
    delimiter?: string | undefined;
    /** Whether the first row is a header; "auto" (default) decides from its content. */
    header?: boolean | "auto" | undefined;
    /**
     * What the input is: an edge table, a node table, or "auto" (default): an edge table when
     * source and target columns resolve, a node table when only an id column does.
     */
    table?: "edges" | "nodes" | "auto" | undefined;
    /** The source column, by name or 0-based position; resolved from the header by default. */
    sourceColumn?: CsvColumnRef | undefined;
    /** The target column, by name or 0-based position; resolved from the header by default. */
    targetColumn?: CsvColumnRef | undefined;
    /**
     * The per-row direction column (Directed / Undirected / Mutual); by default the exact `Type`
     * column of a Gephi table (exact `Source` and `Target` headers); null reads no such column.
     */
    typeColumn?: CsvColumnRef | null | undefined;
    /** The id column of a node table, by name or position; resolved from the header by default. */
    idColumn?: CsvColumnRef | undefined;
    /** A node table read before the edges: its ids become nodes and its other columns node attributes. */
    nodes?: ImportInput | undefined;
}

/** Issue code: the input holds no header row at all. */
export const EMPTY_INPUT_CODE = SHARED_EMPTY_INPUT_CODE;
/** Issue code: the header names no source / target (or, for a node table, no id) column. */
export const NO_ENDPOINT_COLUMNS_CODE = "E_CSV_NO_ENDPOINT_COLUMNS";
/** Issue code: a node table without an id column. */
export const NO_ID_COLUMN_CODE = "E_CSV_NO_ID_COLUMN";
/** Issue code: a row with a different number of fields than the header. */
export const FIELD_COUNT_CODE = "E_CSV_FIELD_COUNT";
/** Issue code: an edge row with a blank source or target cell. */
export const MISSING_ENDPOINT_CODE = SHARED_MISSING_ENDPOINT_CODE;
/** Issue code: a node row with a blank id cell. */
export const MISSING_ID_CODE = SHARED_MISSING_ID_CODE;
/** Issue code: a Type cell that is not Directed, Undirected or Mutual. */
export const BAD_TYPE_CODE = "E_CSV_BAD_TYPE";
/** Issue code: the table has a header and no data rows. */
export const NO_DATA_ROWS_CODE = "W_CSV_NO_DATA_ROWS";
/** Issue code: a node table row repeats an id; its attributes overwrite the earlier row's. */
export const DUPLICATE_NODE_CODE = SHARED_DUPLICATE_NODE_CODE;
/** Issue code: two distinct id cells became one id under `ids: "number"` (design section 4.1). */
export const ID_MERGED_CODE = SHARED_ID_MERGED_CODE;
/** Issue code: an explicitly named weight column the file does not have. */
export const COLUMN_MISSING_CODE = "W_CSV_COLUMN_MISSING";
/** Issue code: a column whose role (id, label) is already held by another column of the sink. */
export const ROLE_TAKEN_CODE = SHARED_ROLE_TAKEN_CODE;
/** Issue code: a repeated edge id (the column is unique); the edge is skipped. */
export const DUPLICATE_EDGE_ID_CODE = SHARED_DUPLICATE_EDGE_ID_CODE;

const TABLE_MODES: ReadonlySet<string> = new Set(["edges", "nodes", "auto"]);

/** The common options an edge-table import reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
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

/** The common options an import with a node table reads: nodeIdFrom applies to the node table. */
const USED_OPTIONS_WITH_NODES: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    ...USED_OPTIONS,
    "nodeIdFrom",
]);
const BAD_DELIMITERS: ReadonlySet<string> = new Set(['"', "\n", "\r"]);

/** The CSV options with defaults applied. */
interface ResolvedCsvOptions {
    readonly delimiter: string | null;
    readonly header: boolean | "auto";
    readonly table: "edges" | "nodes" | "auto";
    readonly sourceColumn: CsvColumnRef | null;
    readonly targetColumn: CsvColumnRef | null;
    /** The direction column reference; null for none; undefined for the Gephi rule. */
    readonly typeColumn: CsvColumnRef | null | undefined;
    readonly idColumn: CsvColumnRef | null;
    readonly nodes: ImportInput | null;
}

/** The columns of an edge table, by index. */
interface EdgePlan {
    readonly kind: "edges";
    readonly names: readonly string[];
    readonly width: number;
    readonly source: number;
    readonly target: number;
    readonly weight: number;
    readonly type: number;
    readonly id: number;
    readonly label: number;
    readonly attributes: readonly number[];
}

/** The columns of a node table, by index. */
interface NodePlan {
    readonly kind: "nodes";
    readonly names: readonly string[];
    readonly width: number;
    /** The column ids are read from, or -1 under nodeIdFrom "index". */
    readonly id: number;
    readonly label: number;
    readonly attributes: readonly number[];
}

/** Everything one import call shares between its tables. */
interface ImportState {
    readonly sink: GraphSink;
    readonly report: ImportReportBuilder;
    readonly common: ResolvedImportOptions;
    readonly csv: ResolvedCsvOptions;
    readonly weightFromExplicit: boolean;
    readonly coercer: IdCoercer;
    readonly resolver: DirectionResolver;
    headerSet: boolean;
    /**
     * The file-level direction a SNAP (`# Directed graph` / `# Undirected graph`) or KONECT
     * (`% asym` / `% sym` / `% bip`) comment header declares; null when the file declares none
     * (the `defaultDirected` option applies).
     */
    commentDirected: boolean | null;
}

/** The comment characters of the SNAP (`#`) and KONECT (`%`) headers (research note 07 section 2.6). */
const COMMENT_CHARS: readonly string[] = Object.freeze(["#", "%"]);

/**
 * The direction a SNAP or KONECT comment header declares (research note 07 section 2.6): SNAP
 * pages write `# Directed graph` / `# Undirected graph`, KONECT's first line is `% sym` (undirected),
 * `% asym` (directed) or `% bip` (bipartite, undirected).
 * @param comments - the leading comment lines
 * @returns true / false when a line declares the direction, null otherwise
 */
function commentDirection(comments: readonly string[]): boolean | null {
    for (const comment of comments) {
        const text = comment.slice(1).trim().toLowerCase();
        if (text.startsWith("directed graph") || text.startsWith("asym")) {
            return true;
        }
        if (text.startsWith("undirected graph") || text.startsWith("sym") || text.startsWith("bip")) {
            return false;
        }
    }
    return null;
}

/**
 * Apply the defaults of the CSV options and check every value.
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for a value outside its set
 */
function resolveCsvOptions(options: (CsvImportOptions & CommonImportOptions) | undefined): ResolvedCsvOptions {
    const o: CsvImportOptions = options ?? {};
    if (o.delimiter !== undefined && (typeof o.delimiter !== "string" || o.delimiter.length === 0)) {
        throw new GraphFormatError("E_UNSUPPORTED", "option delimiter: expected a non-empty string", {
            option: "delimiter",
            found: o.delimiter,
        });
    }
    if (o.delimiter !== undefined && BAD_DELIMITERS.has(o.delimiter)) {
        throw new GraphFormatError("E_UNSUPPORTED", "option delimiter: a quote or a line break cannot delimit", {
            option: "delimiter",
            found: o.delimiter,
        });
    }
    if (o.header !== undefined && o.header !== "auto" && typeof o.header !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", 'option header: expected true, false or "auto"', {
            option: "header",
            found: o.header,
        });
    }
    if (o.table !== undefined && !TABLE_MODES.has(o.table)) {
        throw new GraphFormatError("E_UNSUPPORTED", 'option table: expected "edges", "nodes" or "auto"', {
            option: "table",
            found: o.table,
        });
    }
    for (const name of ["sourceColumn", "targetColumn", "idColumn"] as const) {
        checkColumnRef(name, o[name]);
    }
    if (o.typeColumn !== null) {
        checkColumnRef("typeColumn", o.typeColumn);
    }
    return {
        delimiter: o.delimiter ?? null,
        header: o.header ?? "auto",
        table: o.table ?? "auto",
        sourceColumn: o.sourceColumn ?? null,
        targetColumn: o.targetColumn ?? null,
        typeColumn: o.typeColumn,
        idColumn: o.idColumn ?? null,
        nodes: o.nodes ?? null,
    };
}

/**
 * Check a column reference option: a string name or a non-negative integer position.
 * @param name - the option name
 * @param value - the value
 */
function checkColumnRef(name: string, value: unknown): void {
    if (value === undefined) {
        return;
    }
    if (typeof value === "string" && value.length > 0) {
        return;
    }
    if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
        return;
    }
    throw new GraphFormatError("E_UNSUPPORTED", `option ${name}: expected a column name or a 0-based position`, {
        option: name,
        found: value,
    });
}

/**
 * Whether a cell is empty (nothing or whitespace only): an unset value, a missing endpoint.
 * @param text - the cell text
 * @returns true when blank
 */
function isBlank(text: string): boolean {
    return text.length === 0 || text.trim().length === 0;
}

/**
 * Whether a cell is unset: blank and not quoted. A quoted blank cell (`""`, `" "`) is a set value
 * (the empty string is a legal id and a legal value, design section 4.1); an unquoted one is
 * nothing at all.
 * @param text - the cell text
 * @param quoted - whether the cell was quoted
 * @returns true when the cell carries no value
 */
function isUnset(text: string, quoted: boolean | undefined): boolean {
    return quoted !== true && isBlank(text);
}

/** Rows between two checks of the cancellation signal on an in-memory input. */
const ABORT_CHECK_INTERVAL = 64;

/**
 * Header names made unique within the table: a repeated name becomes `<name>#<position>` (1-based,
 * the CSV analogue of the `#<origin.id>` rule of design section 5.6) and is reported.
 * @param names - the header names
 * @param report - the report
 * @param line - the header line
 * @returns the unique names
 */
function uniqueNames(names: readonly string[], report: ImportReportBuilder, line: number): string[] {
    const seen = new Set<string>();
    return names.map((name, i) => {
        const unique = uniqueColumnName(name, String(i + 1), (n) => seen.has(n));
        seen.add(unique);
        if (unique !== name) {
            report.warning(
                "coercion",
                RENAMED_CODE,
                `column ${i + 1} "${name}" renamed to "${unique}": the header repeats the name`,
                { line, element: name },
            );
        }
        return unique;
    });
}

/**
 * Find a column by candidates, skipping indices already claimed by another role.
 * @param names - the header names
 * @param candidates - the names to look for
 * @param claimed - indices taken
 * @returns the index, or -1
 */
function findFree(names: readonly string[], candidates: readonly string[], claimed: ReadonlySet<number>): number {
    const masked = names.map((name, i) => (claimed.has(i) ? "" : name));
    return findColumn(masked, candidates);
}

/**
 * Declare a role column (edge id, label) on the sink through the shared design section 5.6 rule:
 * a name already declared differently is renamed `<name>#<position>` (reported); a role already
 * held by another column is dropped from this declaration (reported), the values are kept under
 * the name.
 * @param sink - the sink
 * @param domain - node or edge
 * @param decl - the declaration
 * @param position - the 1-based column position, for the rename
 * @param report - the report
 * @param line - the header line
 * @returns the handle
 */
function declareRoleColumn(
    sink: GraphSink,
    domain: "node" | "edge",
    decl: ColumnDecl,
    position: number,
    report: ImportReportBuilder,
    line: number,
): ColumnHandle {
    const withOrigin: ColumnDecl = { ...decl, origin: { ...decl.origin, id: String(position) } };
    const resolved = declareResolved(sink, domain, withOrigin, report, { line, element: decl.name });
    if (resolved.roleDropped) {
        // a unique constraint belongs to the id role; without the role the column is plain text
        return resolved.handle;
    }
    return resolved.handle;
}

/**
 * Parse a Gephi Type cell.
 * @param text - the cell text
 * @returns the kind; undefined for a blank cell (the default applies); null for an unknown word
 */
function parseKind(text: string): EdgeKind | null | undefined {
    if (isBlank(text)) {
        return undefined;
    }
    switch (text.trim().toLowerCase()) {
        case "directed":
            return "directed";
        case "undirected":
            return "undirected";
        case "mutual":
            return "mutual";
        default:
            return null;
    }
}

/**
 * Read one CSV table into the sink.
 */
class TableReader {
    private readonly state: ImportState;

    private readonly reader: CsvRecordReader;

    private readonly kind: "edges" | "nodes" | "auto";

    private plan: EdgePlan | NodePlan | null = null;

    private writers: (InferredColumn | null)[] = [];

    private idHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private labelHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private dataRows = 0;

    private nodeOrdinal = 0;

    /** The edge ids seen so far (the id column is unique; a repeat is skipped with an issue). */
    private readonly edgeIds = new Set<string>();

    private readonly where: { line: number | null; element: string | null } = { line: null, element: null };

    /**
     * Create a table reader.
     * @param state - the import state
     * @param input - the table's input
     * @param kind - what the table is, or "auto"
     * @param progress - whether this table reports byte progress
     */
    constructor(state: ImportState, input: ImportInput, kind: "edges" | "nodes" | "auto", progress: boolean) {
        this.state = state;
        this.kind = kind;
        const readerOptions: CsvReaderOptions = {
            delimiter: state.csv.delimiter,
            comments: COMMENT_CHARS,
            signal: state.common.signal,
            onProgress: progress ? state.common.onProgress : null,
        };
        this.reader = new CsvRecordReader(input, state.report, readerOptions);
    }

    /**
     * Read every row; the reader is closed (and a stream cancelled) when the import aborts midway.
     */
    async read(): Promise<void> {
        const iterator = this.reader[Symbol.asyncIterator]();
        try {
            await this.readRows(iterator);
        } finally {
            await iterator.return(undefined);
        }
    }

    /**
     * Read the header (or decide there is none), resolve the plan, then push every row.
     * @param iterator - the record iterator
     */
    private async readRows(iterator: AsyncGenerator<string[], void, undefined>): Promise<void> {
        const { report } = this.state;
        const first = await iterator.next();
        const firstRow: string[] = first.done
            ? report.fail(EMPTY_INPUT_CODE, "the input is empty: no header row and no records")
            : first.value;
        const firstLine = this.reader.line;
        const firstQuoted = this.reader.quoted.slice(0, firstRow.length);
        const pending: { row: string[]; quoted: readonly boolean[]; line: number }[] = [];
        let header: boolean;
        const { header: mode } = this.state.csv;
        if (mode === "auto") {
            const second = await iterator.next();
            const secondRow: string[] | null = second.done ? null : second.value;
            const secondLine = this.reader.line;
            const secondQuoted = this.reader.quoted.slice(0, secondRow?.length ?? 0);
            header = looksLikeHeader(firstRow, secondRow);
            if (!header) {
                pending.push({ row: firstRow, quoted: firstQuoted, line: firstLine });
            }
            if (secondRow !== null) {
                pending.push({ row: secondRow, quoted: secondQuoted, line: secondLine });
            }
        } else {
            header = mode;
            if (!header) {
                pending.push({ row: firstRow, quoted: firstQuoted, line: firstLine });
            }
        }
        const names = header ? uniqueNames(headerNames(firstRow), report, firstLine) : positionalNames(firstRow.length);
        if (this.kind !== "nodes" && this.state.commentDirected === null) {
            this.state.commentDirected = commentDirection(this.reader.leadingComments);
        }
        this.plan = this.resolvePlan(names, header, firstLine);
        this.prepareColumns(firstLine);
        for (const { row, quoted, line } of pending) {
            this.processRow(row, quoted, line);
        }
        const { signal } = this.state.common;
        let sinceCheck = 0;
        for (;;) {
            const next = await iterator.next();
            if (next.done) {
                break;
            }
            this.processRow(next.value, this.reader.quoted, this.reader.line);
            if (++sinceCheck >= ABORT_CHECK_INTERVAL) {
                sinceCheck = 0;
                throwIfAborted(signal);
            }
        }
        for (const writer of this.writers) {
            writer?.finish();
        }
        if (this.dataRows === 0 && header) {
            report.warning("missing-value", NO_DATA_ROWS_CODE, "the table has a header and no data rows", {
                line: firstLine,
            });
        }
    }

    /**
     * Decide the table's columns from its header.
     * @param names - the column names
     * @param header - whether the file has a header row
     * @param line - the header line
     * @returns the plan; the import aborts when no endpoints (or id) resolve
     */
    private resolvePlan(names: readonly string[], header: boolean, line: number): EdgePlan | NodePlan {
        const { csv, report } = this.state;
        const width = names.length;
        let source = -1;
        let target = -1;
        if (this.kind !== "nodes") {
            if (csv.sourceColumn !== null) {
                source = resolveColumnRef(names, csv.sourceColumn, "sourceColumn");
            } else if (header) {
                source = findColumn(names, SOURCE_NAMES);
            } else {
                source = width >= 2 ? 0 : -1;
            }
            if (csv.targetColumn !== null) {
                target = resolveColumnRef(names, csv.targetColumn, "targetColumn");
            } else if (header) {
                target = findColumn(names, TARGET_NAMES);
            } else {
                target = width >= 2 ? 1 : -1;
            }
            if (source >= 0 && target >= 0 && source === target) {
                throw new GraphFormatError("E_UNSUPPORTED", "sourceColumn and targetColumn name the same column", {
                    option: "targetColumn",
                    found: names[target],
                });
            }
        }
        if (source >= 0 && target >= 0) {
            return this.edgePlan(names, header, source, target);
        }
        const shown = names.map((n) => JSON.stringify(n)).join(", ");
        if (
            this.kind === "edges" ||
            (this.kind === "auto" && (csv.sourceColumn !== null || csv.targetColumn !== null))
        ) {
            report.fail(
                NO_ENDPOINT_COLUMNS_CODE,
                `no source / target columns in the header (${shown}); a node table goes in the nodes option`,
                { line },
                { columns: [...names] },
            );
        }
        const idResolves = header ? findColumn(names, ID_NAMES) >= 0 : width >= 1;
        if (this.kind === "auto" && csv.idColumn === null && !idResolves) {
            report.fail(
                NO_ENDPOINT_COLUMNS_CODE,
                `no source / target columns and no id column in the header (${shown}); the input is neither an edge table nor a node table`,
                { line },
                { columns: [...names] },
            );
        }
        return this.nodePlan(names, header, line);
    }

    /**
     * The columns of an edge table.
     * @param names - the column names
     * @param header - whether the file has a header row
     * @param source - the source column
     * @param target - the target column
     * @returns the plan
     */
    private edgePlan(names: readonly string[], header: boolean, source: number, target: number): EdgePlan {
        const { csv, common, report } = this.state;
        const claimed = new Set<number>([source, target]);
        let weight = -1;
        if (common.weightFrom !== null) {
            if (header) {
                weight = findFree(names, [common.weightFrom], claimed);
                if (weight < 0 && this.state.weightFromExplicit) {
                    report.warning(
                        "missing-value",
                        COLUMN_MISSING_CODE,
                        `weight column ${JSON.stringify(common.weightFrom)} not found; edges are unweighted`,
                        { line: this.reader.line, element: common.weightFrom },
                    );
                }
            } else if (names.length >= 3 && !claimed.has(2)) {
                weight = 2;
            }
        }
        if (weight >= 0) {
            claimed.add(weight);
        }
        let type = -1;
        if (csv.typeColumn === undefined) {
            if (header && names[source] === "Source" && names[target] === "Target") {
                type = findFree(names, [TYPE_NAME], claimed);
                if (type >= 0 && names[type] !== TYPE_NAME) {
                    type = -1;
                }
            }
        } else if (csv.typeColumn !== null) {
            type = resolveColumnRef(names, csv.typeColumn, "typeColumn");
            if (claimed.has(type)) {
                throw new GraphFormatError("E_UNSUPPORTED", "typeColumn names an endpoint or weight column", {
                    option: "typeColumn",
                    found: names[type],
                });
            }
        }
        if (type >= 0) {
            claimed.add(type);
        }
        const id = header ? findFree(names, EDGE_ID_NAMES, claimed) : -1;
        if (id >= 0) {
            claimed.add(id);
        }
        const label = header ? findFree(names, LABEL_NAMES, claimed) : -1;
        if (label >= 0) {
            claimed.add(label);
        }
        const attributes: number[] = [];
        for (let i = 0; i < names.length; i++) {
            if (!claimed.has(i)) {
                attributes.push(i);
            }
        }
        return { kind: "edges", names, width: names.length, source, target, weight, type, id, label, attributes };
    }

    /**
     * The columns of a node table.
     * @param names - the column names
     * @param header - whether the file has a header row
     * @param line - the header line
     * @returns the plan; the import aborts when no id column resolves
     */
    private nodePlan(names: readonly string[], header: boolean, line: number): NodePlan {
        const { csv, common, report } = this.state;
        const claimed = new Set<number>();
        let idColumn = -1;
        if (csv.idColumn !== null) {
            idColumn = resolveColumnRef(names, csv.idColumn, "idColumn");
        } else if (header) {
            idColumn = findColumn(names, ID_NAMES);
        } else if (names.length >= 1) {
            idColumn = 0;
        }
        const label = header ? findFree(names, LABEL_NAMES, new Set(idColumn >= 0 ? [idColumn] : [])) : -1;
        let id: number;
        switch (common.nodeIdFrom) {
            case "label":
                if (label < 0) {
                    report.fail(NO_ID_COLUMN_CODE, 'nodeIdFrom is "label" but the node table has no label column', {
                        line,
                    });
                }
                id = label;
                break;
            case "index":
                id = -1;
                break;
            default:
                if (idColumn < 0) {
                    report.fail(
                        NO_ID_COLUMN_CODE,
                        `no id column in the node table header (${names.map((n) => JSON.stringify(n)).join(", ")})`,
                        { line },
                        { columns: [...names] },
                    );
                }
                id = idColumn;
                claimed.add(idColumn);
                break;
        }
        if (label >= 0) {
            claimed.add(label);
        }
        const attributes: number[] = [];
        for (let i = 0; i < names.length; i++) {
            if (!claimed.has(i)) {
                attributes.push(i);
            }
        }
        return { kind: "nodes", names, width: names.length, id, label, attributes };
    }

    /**
     * Declare the role columns and create a writer per attribute column.
     * @param line - the header line
     */
    private prepareColumns(line: number): void {
        const plan = this.requirePlan();
        const { sink, report } = this.state;
        const domain = plan.kind === "edges" ? "edge" : "node";
        const origin = { format: "csv" };
        if (plan.kind === "edges" && plan.id >= 0) {
            this.idHandle = declareRoleColumn(
                sink,
                "edge",
                { name: plan.names[plan.id], dtype: "string", nullable: true, role: "id", unique: true, origin },
                plan.id + 1,
                report,
                line,
            );
        }
        if (plan.label >= 0) {
            this.labelHandle = declareRoleColumn(
                sink,
                domain,
                { name: plan.names[plan.label], dtype: "string", nullable: true, role: "label", origin },
                plan.label + 1,
                report,
                line,
            );
        }
        this.writers = plan.names.map(() => null);
        for (const index of plan.attributes) {
            this.writers[index] = new InferredColumn(plan.names[index], domain, sink, report);
        }
    }

    /**
     * The plan, which exists once the header was read.
     * @returns the plan
     */
    private requirePlan(): EdgePlan | NodePlan {
        if (this.plan === null) {
            throw new GraphFormatError("E_UNSUPPORTED", "the header has not been read", { reason: "no plan" });
        }
        return this.plan;
    }

    /**
     * Push one data row.
     * @param row - the cells
     * @param quoted - whether each cell was quoted (a quoted empty cell is the empty string)
     * @param line - the row's line
     */
    private processRow(row: string[], quoted: readonly boolean[], line: number): void {
        const plan = this.requirePlan();
        this.dataRows++;
        if (plan.kind === "edges") {
            this.processEdgeRow(plan, row, quoted, line);
        } else {
            this.processNodeRow(plan, row, quoted, line);
        }
    }

    /**
     * Push one edge row: endpoints, weight, direction, then the attribute cells.
     * @param plan - the edge plan
     * @param row - the cells
     * @param quoted - whether each cell was quoted
     * @param line - the row's line
     */
    private processEdgeRow(plan: EdgePlan, row: string[], quoted: readonly boolean[], line: number): void {
        const { report, sink, resolver, common } = this.state;
        const { counts } = report;
        if (row.length !== plan.width) {
            report.error(
                "validation-error",
                FIELD_COUNT_CODE,
                `line ${line}: ${row.length} field(s), the header has ${plan.width}`,
                { line },
            );
            counts.skippedEdges++;
            return;
        }
        const sourceText = row[plan.source];
        const targetText = row[plan.target];
        const sourceMissing = isUnset(sourceText, quoted[plan.source]);
        if (sourceMissing || isUnset(targetText, quoted[plan.target])) {
            report.error(
                "missing-value",
                MISSING_ENDPOINT_CODE,
                `line ${line}: blank ${sourceMissing ? "source" : "target"} cell`,
                { line },
            );
            counts.skippedEdges++;
            return;
        }
        let kind: EdgeKind = (this.state.commentDirected ?? common.defaultDirected) ? "directed" : "undirected";
        if (plan.type >= 0) {
            const parsed = parseKind(row[plan.type]);
            if (parsed === null) {
                report.error(
                    "validation-error",
                    BAD_TYPE_CODE,
                    `line ${line}: Type ${JSON.stringify(row[plan.type])} is not Directed, Undirected or Mutual`,
                    { line },
                );
                counts.skippedEdges++;
                return;
            }
            if (parsed !== undefined) {
                kind = parsed;
            }
        }
        const { where } = this;
        where.line = line;
        where.element = null;
        const idText = plan.id >= 0 && !isUnset(row[plan.id], quoted[plan.id]) ? row[plan.id] : null;
        if (idText !== null) {
            if (this.edgeIds.has(idText)) {
                report.error(
                    "validation-error",
                    DUPLICATE_EDGE_ID_CODE,
                    `line ${line}: edge id ${JSON.stringify(idText)} repeats an earlier row's; the row is skipped`,
                    { line, element: idText },
                );
                counts.skippedEdges++;
                return;
            }
            this.edgeIds.add(idText);
        }
        let edge: number;
        try {
            const source = this.coerce(sourceText);
            const target = this.coerce(targetText);
            const weight = plan.weight >= 0 ? parseWeightText(row[plan.weight]) : undefined;
            if (!this.state.headerSet) {
                this.state.headerSet = true;
                resolver.setHeader(kind !== "undirected", where);
            }
            const sourceNew = sink.indexOf(source) === INVALID_INDEX;
            const targetNew = source !== target && sink.indexOf(target) === INVALID_INDEX;
            const before = sink.edgeCount;
            edge = resolver.addEdge(source, target, kind, weight, where);
            counts.edges += sink.edgeCount - before;
            counts.nodes += (sourceNew ? 1 : 0) + (targetNew ? 1 : 0);
        } catch (err) {
            where.element = idText ?? `${sourceText}->${targetText}`;
            report.recordError(err, where);
            counts.skippedEdges++;
            return;
        }
        if (idText !== null) {
            this.writeRole(this.idHandle, "edge", edge, idText, plan.names[plan.id], line);
        }
        if (plan.label >= 0 && !isUnset(row[plan.label], quoted[plan.label])) {
            this.writeRole(this.labelHandle, "edge", edge, row[plan.label], plan.names[plan.label], line);
        }
        this.writeAttributes(plan, row, quoted, edge, line);
    }

    /**
     * Push one node row: the id, then the label and attribute cells.
     * @param plan - the node plan
     * @param row - the cells
     * @param quoted - whether each cell was quoted
     * @param line - the row's line
     */
    private processNodeRow(plan: NodePlan, row: string[], quoted: readonly boolean[], line: number): void {
        const { report, sink } = this.state;
        const { counts } = report;
        const ordinal = this.nodeOrdinal++;
        if (row.length !== plan.width) {
            report.error(
                "validation-error",
                FIELD_COUNT_CODE,
                `line ${line}: ${row.length} field(s), the header has ${plan.width}`,
                { line },
            );
            counts.skippedNodes++;
            return;
        }
        const idText = plan.id >= 0 ? row[plan.id] : String(ordinal);
        if (plan.id >= 0 && isUnset(idText, quoted[plan.id])) {
            report.error("missing-value", MISSING_ID_CODE, `line ${line}: blank id cell`, { line });
            counts.skippedNodes++;
            return;
        }
        const { where } = this;
        where.line = line;
        where.element = idText;
        let index: number;
        try {
            const id = plan.id >= 0 ? this.coerce(idText) : ordinal;
            if (sink.indexOf(id) !== INVALID_INDEX) {
                report.warning(
                    "merged",
                    DUPLICATE_NODE_CODE,
                    `line ${line}: node ${JSON.stringify(id)} already exists; its attributes are overwritten`,
                    where,
                );
            } else {
                counts.nodes++;
            }
            index = sink.addNode(id);
        } catch (err) {
            report.recordError(err, where);
            counts.skippedNodes++;
            return;
        }
        if (plan.label >= 0 && !isUnset(row[plan.label], quoted[plan.label])) {
            this.writeRole(this.labelHandle, "node", index, row[plan.label], plan.names[plan.label], line);
        }
        this.writeAttributes(plan, row, quoted, index, line);
    }

    /**
     * Coerce an id cell, reporting a merge under `ids: "number"`.
     * @param text - the cell text
     * @returns the id
     */
    private coerce(text: string): NodeId {
        const id = this.state.coercer.text(text);
        const merge = this.state.coercer.lastMerge;
        if (merge !== null) {
            this.state.report.warnOnce(
                "coercion",
                ID_MERGED_CODE,
                `id ${JSON.stringify(merge.text)} merged with ${JSON.stringify(merge.previousText)} as ${merge.id} under ids: "number"`,
                this.where,
            );
        }
        return id;
    }

    /**
     * Write a role column cell (edge id, label).
     * @param handle - the column handle
     * @param domain - node or edge
     * @param row - the node or edge index
     * @param text - the cell text
     * @param name - the column name, for issues
     * @param line - the row's line
     */
    private writeRole(
        handle: ColumnHandle,
        domain: "node" | "edge",
        row: number,
        text: string,
        name: string,
        line: number,
    ): void {
        try {
            if (domain === "node") {
                this.state.sink.setNodeValue(handle, row, text);
            } else {
                this.state.sink.setEdgeValue(handle, row, text);
            }
        } catch (err) {
            this.state.report.recordError(err, { line, element: name });
        }
    }

    /**
     * Write the attribute cells of a row: an unquoted blank cell is unset, a quoted one (`""`,
     * `" "`) is the text it holds.
     * @param plan - the plan
     * @param row - the cells
     * @param quoted - whether each cell was quoted
     * @param index - the node or edge index
     * @param line - the row's line
     */
    private writeAttributes(
        plan: EdgePlan | NodePlan,
        row: readonly string[],
        quoted: readonly boolean[],
        index: number,
        line: number,
    ): void {
        for (const k of plan.attributes) {
            const text = row[k];
            if (isUnset(text, quoted[k])) {
                continue;
            }
            const writer = this.writers[k];
            if (writer === null) {
                continue;
            }
            try {
                writer.write(index, text);
            } catch (err) {
                this.state.report.recordError(err, { line, element: writer.name });
            }
        }
    }
}

const HEAD_BYTES = 4096;
const OTHER_FORMAT = /^\s*(<|[[{]|(strict\s+)?(di)?graph(\s+\S+)?\s*\{|\*vertices|creator\b|graph\s*\[)/i;

/**
 * Sniff confidence for the registry: 0 for XML, JSON, GML, DOT and Pajek openings; otherwise a
 * delimited first row with endpoint headers is 0.9, with an id header 0.6, any consistently
 * delimited rows 0.3, a single column 0.
 * @param head - the first bytes of the input
 * @returns a confidence in 0..1
 */
function sniff(head: Uint8Array): number {
    const text = new TextDecoder("utf-8").decode(head.subarray(0, HEAD_BYTES));
    const body = text.startsWith(String.fromCharCode(0xfeff)) ? text.slice(1) : text;
    if (body.trim().length === 0 || OTHER_FORMAT.test(body)) {
        return 0;
    }
    const newline = sniffNewline(body);
    const delimiter = sniffDelimiter(body, newline);
    if (delimiter === null) {
        return 0;
    }
    const end = body.indexOf(newline);
    const firstLine = end < 0 ? body : body.slice(0, end);
    const names = headerNames(firstLine.replace(/\r$/, "").split(delimiter));
    if (findColumn(names, SOURCE_NAMES) >= 0 && findColumn(names, TARGET_NAMES) >= 0) {
        return 0.9;
    }
    if (findColumn(names, ID_NAMES) >= 0) {
        return 0.6;
    }
    return 0.3;
}

/**
 * Import a CSV edge table (and an optional node table) into a sink.
 * @param input - the edge table (or, with `table: "nodes"` or a header without endpoints, a node table)
 * @param sink - the sink
 * @param options - CSV and common options
 * @returns the report; ImportError with the partial report when the import aborts
 */
async function importCsv(
    input: ImportInput,
    sink: GraphSink,
    options?: CsvImportOptions & CommonImportOptions,
): Promise<ImportReport> {
    const common = resolveImportOptions(options, { ids: "canonical", defaultDirected: true, weightFrom: "weight" });
    const csv = resolveCsvOptions(options);
    const report = new ImportReportBuilder("csv", common.errorLimit);
    reportSinkOptions(sink, options, report);
    reportUnusedOptions(
        options,
        report,
        csv.table === "nodes" || csv.nodes !== null ? USED_OPTIONS_WITH_NODES : USED_OPTIONS,
    );
    const state: ImportState = {
        sink,
        report,
        common,
        csv,
        weightFromExplicit: typeof options?.weightFrom === "string",
        coercer: new IdCoercer(common.ids),
        resolver: new DirectionResolver(sink, report, common.onMixedDirection),
        headerSet: false,
        commentDirected: null,
    };
    if (csv.nodes !== null) {
        await new TableReader(state, csv.nodes, "nodes", false).read();
    }
    await new TableReader(state, input, csv.table, true).read();
    if (state.coercer.mergeCount > 1) {
        report.warning(
            "coercion",
            ID_MERGED_CODE,
            `${state.coercer.mergeCount} id cell(s) merged into ids other cells already produced under ids: "number"`,
        );
    }
    throwIfAborted(common.signal);
    return report.finish();
}

/** The CSV / TSV importer plugin (subpath `@graphty/graph-io/csv`). */
export const csvImporter: GraphImporter<CsvImportOptions> = Object.freeze({
    format: "csv",
    extensions: Object.freeze([".csv", ".tsv", ".edges", ".edgelist"]),
    mimeTypes: Object.freeze(["text/csv", "text/tab-separated-values", "text/plain"]),
    sniff,
    import: importCsv,
});
