/**
 * The fixed lexical grammar of design section 5.1 for untyped text sources (CSV cells, GML
 * values, DOT strings, Pajek tokens), reproduced from the core's reference implementation so the
 * two agree (invariant I15). Text importers parse each cell with parseTextCell() and push the JS
 * value; the sink's per-column inference then widens `bool -> i32 -> f64 -> string` and never per
 * cell, so a column that saw `1` and then `"01"` becomes string for all rows.
 *
 * - `bool` is exactly `true` / `false` (case-sensitive);
 * - `i32` is `/^-?(0|[1-9][0-9]*)$/` within `[-2^31, 2^31)`; `0` / `1` are i32, never bool;
 * - `f64` is a decimal or exponent literal accepted by `Number()` that is not empty, not
 *   whitespace, not `Infinity` / `NaN`, not a hex / octal / binary form, has no leading zeros in
 *   its integer part, and whose value is finite (an integer literal outside i32 range is f64);
 * - everything else is `string`.
 */

import { type ColumnHandle, GraphFormatError, type GraphSink, INVALID_INDEX } from "@graphty/graph-format";

import { type ImportReportBuilder } from "./report.js";

const I32_TEXT = /^-?(0|[1-9][0-9]*)$/;
const F64_TEXT = /^[+-]?((0|[1-9][0-9]*)(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

/**
 * The dtype a text cell parses as under the design section 5.1 grammar.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export type TextDtype = "bool" | "i32" | "f64" | "string";

/**
 * Classify one text cell by the fixed grammar.
 * @param text - the cell text, exactly as read (no trimming)
 * @returns bool, i32, f64 or string
 */
export function inferTextDtype(text: string): TextDtype {
    if (text === "true" || text === "false") {
        return "bool";
    }
    if (I32_TEXT.test(text)) {
        const n = Number(text);
        if (n >= I32_MIN && n <= I32_MAX) {
            return "i32";
        }
        return Number.isFinite(n) ? "f64" : "string";
    }
    if (F64_TEXT.test(text) && Number.isFinite(Number(text))) {
        return "f64";
    }
    return "string";
}

/**
 * Parse one text cell into the JS value its grammar class implies: a boolean for bool, a number for
 * i32 / f64, the text itself for string. The sink infers the column dtype from the value.
 * @param text - the cell text, exactly as read
 * @returns the value
 */
export function parseTextCell(text: string): boolean | number | string {
    switch (inferTextDtype(text)) {
        case "bool":
            return text === "true";
        case "i32":
        case "f64":
            return Number(text);
        case "string":
            return text;
        default:
            return text;
    }
}

/**
 * Whether a text cell is a number under the f64 grammar (an i32 or f64 literal).
 * @param text - the cell text
 * @returns true when parseTextCell(text) returns a number
 */
export function isNumericText(text: string): boolean {
    const dtype = inferTextDtype(text);
    return dtype === "i32" || dtype === "f64";
}

/**
 * Whether a numeric cell text is the canonical spelling of its value (`String(value) === text`),
 * so that the sink's own re-formatting of the value would reproduce it.
 * @param text - the cell text
 * @param value - the parsed value
 * @returns true when the lexical form survives a widening to string
 */
function isCanonicalNumberText(text: string, value: number): boolean {
    return String(value) === text;
}

/**
 * The widening rank of a text dtype in the design section 5.1 order.
 * @param dtype - the text dtype
 * @returns 0 for bool, 1 for i32, 2 for f64, 3 for string
 */
function textDtypeRank(dtype: TextDtype): number {
    switch (dtype) {
        case "bool":
            return 0;
        case "i32":
            return 1;
        case "f64":
            return 2;
        case "string":
            return 3;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown text dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * The inferred-column writer of the untyped text formats (CSV, DOT, Pajek; design section 5.1):
 * parses every cell by the fixed grammar and pushes the scalar, and keeps the COLUMN's dtype the
 * one the grammar implies rather than the one the values happen to imply:
 *
 * - a column whose cells are all `2.0`-style f64 text is widened to f64 through the sink's
 *   widening call even though every value is integral (the values alone would infer i32);
 * - a numeric cell whose text is not the canonical spelling of its value (`1e5`, `-0`, `1.0`) is
 *   remembered, and when a later cell widens the column to string the original texts are written
 *   back, so the lexical form is never rewritten by the widening.
 *
 * A sink without the optional widening call keeps the value-inferred dtype and the writer records
 * one W_WIDENING_UNSUPPORTED warning per column.
 * Consumed by the per-format importers under src/formats.
 * @public
 */
export class TextCellWriter {
    /** The column name in the sink. */
    readonly name: string;

    private readonly domain: "node" | "edge";

    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private handle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    /** The widest text dtype seen (the column's dtype under the 5.1 grammar). */
    private textDtype: TextDtype | null = null;

    /** The widest dtype the pushed values imply (what the sink inferred on its own). */
    private valueDtype: TextDtype | null = null;

    private readonly keptRows: number[] = [];

    private readonly keptTexts: string[] = [];

    /**
     * Create a writer; the column is declared by the sink's inference on the first write.
     * @param name - the column name
     * @param domain - node or edge
     * @param sink - the sink
     * @param report - the report the widening warning is recorded in
     */
    constructor(name: string, domain: "node" | "edge", sink: GraphSink, report: ImportReportBuilder) {
        this.name = name;
        this.domain = domain;
        this.sink = sink;
        this.report = report;
    }

    /**
     * The column handle once the first cell was written.
     * @returns the handle, or INVALID_INDEX before the first write
     */
    get column(): ColumnHandle {
        return this.handle;
    }

    /**
     * Write one cell text.
     * @param row - the node or edge index
     * @param text - the cell text, exactly as read
     */
    write(row: number, text: string): void {
        const kind = inferTextDtype(text);
        const value = parseTextCell(text);
        const wasString = this.textDtype === "string";
        const textDtype =
            this.textDtype === null || textDtypeRank(kind) > textDtypeRank(this.textDtype) ? kind : this.textDtype;
        // a string column keeps every cell's lexical form; a numeric text is never re-spelled. The
        // sink may refuse the value (a declared column of another dtype): the state advances only
        // once the cell is written, so a refused cell leaves the writer as it was.
        this.set(row, textDtype === "string" ? text : value);
        this.textDtype = textDtype;
        const valueKind = kindOfValue(value);
        if (this.valueDtype === null || textDtypeRank(valueKind) > textDtypeRank(this.valueDtype)) {
            this.valueDtype = valueKind;
        }
        if (this.textDtype === "string") {
            if (!wasString && this.keptRows.length > 0) {
                // the sink just widened the column to string from the values; restore the texts
                // whose lexical form the values did not carry
                for (let i = 0; i < this.keptRows.length; i++) {
                    this.set(this.keptRows[i], this.keptTexts[i]);
                }
                this.keptRows.length = 0;
                this.keptTexts.length = 0;
            }
            return;
        }
        if (typeof value === "number" && !isCanonicalNumberText(text, value)) {
            this.keptRows.push(row);
            this.keptTexts.push(text);
        }
        if (this.textDtype === "f64" && this.valueDtype !== "f64") {
            this.widen("f64");
        }
    }

    /**
     * Write a value (the parsed scalar, or a text into a widened column).
     * @param row - the row
     * @param value - the value
     */
    private set(row: number, value: unknown): void {
        if (this.handle === INVALID_INDEX) {
            if (this.domain === "node") {
                this.sink.setNodeValue(this.name, row, value);
                this.handle = this.sink.nodeColumn(this.name);
            } else {
                this.sink.setEdgeValue(this.name, row, value);
                this.handle = this.sink.edgeColumn(this.name);
            }
            return;
        }
        if (this.domain === "node") {
            this.sink.setNodeValue(this.handle, row, value);
        } else {
            this.sink.setEdgeValue(this.handle, row, value);
        }
    }

    /**
     * Widen the column to the dtype the text grammar implies, through the sink's optional call.
     * @param dtype - the dtype
     */
    private widen(dtype: "f64"): void {
        const { sink } = this;
        const supported =
            this.domain === "node" ? sink.widenNodeColumn !== undefined : sink.widenEdgeColumn !== undefined;
        if (!supported) {
            this.report.warnOnce(
                "coercion",
                WIDENING_UNSUPPORTED_CODE,
                `column "${this.name}" holds ${dtype} text but the sink cannot widen an inferred column; it keeps the value-inferred dtype`,
                { element: this.name },
            );
            this.valueDtype = dtype;
            return;
        }
        try {
            if (this.domain === "node") {
                sink.widenNodeColumn?.(this.handle, dtype);
            } else {
                sink.widenEdgeColumn?.(this.handle, dtype);
            }
            this.valueDtype = dtype;
        } catch (err) {
            if (!(err instanceof GraphFormatError) || err.code !== "E_COLUMN_TYPE") {
                throw err;
            }
            // a caller's declared column of the name: its dtype stands
            this.valueDtype = dtype;
        }
    }
}

/** Issue code: the sink has no widening call, so a text column keeps the dtype its values imply. */
export const WIDENING_UNSUPPORTED_CODE = "W_WIDENING_UNSUPPORTED";

/**
 * The text dtype a parsed cell value implies on its own (what the sink's inference sees).
 * @param value - the parsed value
 * @returns the dtype
 */
function kindOfValue(value: boolean | number | string): TextDtype {
    if (typeof value === "boolean") {
        return "bool";
    }
    if (typeof value === "string") {
        return "string";
    }
    return Number.isInteger(value) && value >= -2147483648 && value <= 2147483647 && !Object.is(value, -0)
        ? "i32"
        : "f64";
}
