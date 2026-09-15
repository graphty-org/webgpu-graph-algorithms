/**
 * The untyped attribute columns of the CSV importer (design sections 5.1 and 5.4): every cell is
 * parsed by the fixed lexical grammar and pushed as a scalar, the sink's per-column inference
 * widens `bool -> i32 -> f64 -> string`, and a column whose sampled values are all text becomes a
 * `dict` when its cardinality stays below half the sampled rows (the 1024-row heuristic of
 * design section 5.4). The sample is the only buffering: at most DICT_SAMPLE_ROWS (row, text)
 * pairs per column, flushed on the decision.
 *
 * An unquoted empty cell is unset (never written); a quoted empty cell is the empty string. A
 * column whose every cell is unset is declared as a nullable string column at the end so the
 * header survives a round trip. Inferred columns go through the shared TextCellWriter, which keeps
 * the column's dtype the one the 5.1 grammar implies and the lexical form of numeric-looking cells
 * that a widening to string would otherwise rewrite.
 */

import { type ColumnHandle, GraphFormatError, type GraphSink, INVALID_INDEX } from "@graphty/graph-format";

import { DICT_SAMPLE_ROWS, DictHeuristic } from "../../common/attributes.js";
import { type ImportReportBuilder } from "../../common/report.js";
import { inferTextDtype, parseTextCell, TextCellWriter } from "../../common/text.js";

/**
 * One inferred attribute column of a node or edge table, written cell by cell.
 */
export class InferredColumn {
    /** The column name in the sink. */
    readonly name: string;

    private readonly domain: "node" | "edge";

    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private handle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    /** Whether values are pushed as text: a dict column, or a caller's string / dict column of the name. */
    private text = false;

    /** Whether the column was declared as a dict by this writer. */
    private dict = false;

    /** Whether the dtype decision was made; until then values are sampled. */
    private decided = false;

    /** Whether every sampled value so far is text under the lexical grammar (a dict candidate). */
    private candidate = true;

    private readonly heuristic: DictHeuristic;

    /** The inferred-column writer, once the column was decided to be inferred (neither dict nor a caller's). */
    private writer: TextCellWriter | null = null;

    private readonly pendingRows: number[] = [];

    private readonly pendingTexts: string[] = [];

    /**
     * Create a column writer; nothing is declared until the first value or finish().
     * @param name - the column name
     * @param domain - node or edge
     * @param sink - the sink
     * @param report - the report a refused sampled value is recorded in (the flush has no line numbers)
     * @param sampleRows - rows sampled before the dict decision; DICT_SAMPLE_ROWS by default
     */
    constructor(
        name: string,
        domain: "node" | "edge",
        sink: GraphSink,
        report: ImportReportBuilder,
        sampleRows: number = DICT_SAMPLE_ROWS,
    ) {
        this.name = name;
        this.domain = domain;
        this.sink = sink;
        this.report = report;
        this.heuristic = new DictHeuristic(sampleRows);
    }

    /**
     * Whether the column was declared as a dict.
     * @returns true after a dict decision
     */
    get isDict(): boolean {
        return this.dict;
    }

    /**
     * Write one set cell.
     * @param row - the node or edge index
     * @param text - the cell text (the empty string for a quoted empty cell)
     */
    write(row: number, text: string): void {
        if (this.decided) {
            this.push(row, text);
            return;
        }
        this.pendingRows.push(row);
        this.pendingTexts.push(text);
        if (this.candidate && inferTextDtype(text) !== "string") {
            this.candidate = false;
        }
        const full = this.heuristic.observe(text);
        if (full || !this.candidate) {
            this.decide();
        }
    }

    /**
     * Decide an undecided column from what was sampled (or declare an all-empty column as string)
     * and flush the sample. Call once at the end of the input.
     */
    finish(): void {
        if (!this.decided) {
            this.decide();
        }
        if (this.handle === INVALID_INDEX) {
            this.handle = this.lookup();
            if (this.handle === INVALID_INDEX) {
                this.handle = this.declare("string");
            }
        }
    }

    /**
     * Make the dtype decision: a dict when every sampled value is text and the cardinality is low
     * and no column of the name exists yet; inference otherwise. A column the sink already holds
     * (a caller's) receives the cell text when it is a string or dict column and the parsed value
     * otherwise. The sample is flushed either way; a sampled value the sink refuses (a caller's
     * typed column of the name) is recorded as an issue of this column and the flush continues.
     */
    private decide(): void {
        this.decided = true;
        const existing = this.lookup();
        if (existing !== INVALID_INDEX) {
            this.handle = existing;
            this.text = this.holdsText();
        } else if (this.candidate && this.heuristic.decide() === "dict") {
            this.dict = true;
            this.text = true;
            this.handle = this.declare("dict");
        } else {
            this.writer = new TextCellWriter(this.name, this.domain, this.sink, this.report);
        }
        const rows = this.pendingRows;
        const texts = this.pendingTexts;
        for (let i = 0; i < rows.length; i++) {
            try {
                this.push(rows[i], texts[i]);
            } catch (err) {
                this.report.recordError(err, { element: this.name });
            }
        }
        rows.length = 0;
        texts.length = 0;
    }

    /**
     * Push one value: text into a dict or a caller's text column, the parsed scalar into a caller's
     * typed column, and everything else through the inferred-column writer.
     * @param row - the node or edge index
     * @param text - the cell text
     */
    private push(row: number, text: string): void {
        if (this.writer !== null) {
            this.writer.write(row, text);
            this.handle = this.writer.column;
            return;
        }
        const value: unknown = this.text ? text : parseTextCell(text);
        this.set(this.handle, row, value);
    }

    /**
     * Set a cell in the sink's table for this domain.
     * @param column - the handle or name
     * @param row - the row
     * @param value - the value
     */
    private set(column: ColumnHandle | string, row: number, value: unknown): void {
        if (this.domain === "node") {
            this.sink.setNodeValue(column, row, value);
        } else {
            this.sink.setEdgeValue(column, row, value);
        }
    }

    /**
     * The sink's handle for this column's name.
     * @returns the handle, or INVALID_INDEX when the sink has no such column yet
     */
    private lookup(): ColumnHandle {
        return this.domain === "node" ? this.sink.nodeColumn(this.name) : this.sink.edgeColumn(this.name);
    }

    /**
     * Whether the sink's existing column of this name is a string or dict column, probed through
     * the sink's declare (the same shape returns the existing handle, another is E_COLUMN_EXISTS).
     * @returns true when cell text is what the column holds
     */
    private holdsText(): boolean {
        for (const dtype of ["string", "dict"] as const) {
            try {
                this.declare(dtype);
                return true;
            } catch (err) {
                if (!(err instanceof GraphFormatError) || err.code !== "E_COLUMN_EXISTS") {
                    throw err;
                }
            }
        }
        return false;
    }

    /**
     * Declare the column with a fixed dtype.
     * @param dtype - dict or string
     * @returns the handle
     */
    private declare(dtype: "dict" | "string"): ColumnHandle {
        const decl = { name: this.name, dtype, nullable: true, origin: { format: "csv" } };
        return this.domain === "node" ? this.sink.declareNodeColumn(decl) : this.sink.declareEdgeColumn(decl);
    }
}
