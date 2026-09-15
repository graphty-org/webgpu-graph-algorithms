/**
 * Type inference for untyped sources (design section 5.1, decision C4): CSV cells, GML values, DOT
 * strings, `fromRecords` and `addNodeRecord`. Inference runs per COLUMN, never per cell, widening
 * monotonically in the order `(unset) -> bool -> i32 -> f64 -> string -> json`; it never yields
 * `f32`. The lexical grammar for text is fixed so two implementations agree (invariant I15):
 *
 * - `bool` is exactly `true` / `false` (case-sensitive);
 * - `i32` is `/^-?(0|[1-9][0-9]*)$/` within `[-2^31, 2^31)`;
 * - `f64` is a decimal or exponent literal accepted by `Number()` that is not empty, not whitespace,
 *   not `Infinity` / `NaN` and not a hex / octal / binary form (and whose value is finite);
 * - everything else is `string`. `0` / `1` text is `i32`, never `bool`.
 *
 * JS values map by `typeof`: `boolean -> bool`, integral `number` in i32 range -> `i32`, other
 * `number -> f64`, `string -> string`, array or plain object -> `json`; `null` and `undefined` are
 * unset and do not widen.
 */

import { GraphFormatError } from "../errors.js";

/** The dtypes inference can produce, in widening order. */
export type InferredDtype = "bool" | "i32" | "f64" | "string" | "json";

/** The widening order of design section 5.1; a column only ever moves to the right. */
export const WIDENING_ORDER: readonly InferredDtype[] = ["bool", "i32", "f64", "string", "json"];

const I32_TEXT = /^-?(0|[1-9][0-9]*)$/;
// integer part without leading zeros ("01" is a string, design section 5.1), optional fraction, optional exponent
const F64_TEXT = /^[+-]?((0|[1-9][0-9]*)(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

/**
 * The rank of a dtype in the widening order.
 * @param dtype - an inferred dtype
 * @returns 0 for bool up to 4 for json
 */
export function wideningRank(dtype: InferredDtype): number {
    return WIDENING_ORDER.indexOf(dtype);
}

/**
 * Widen a column's dtype by one observation: the wider of the two, where null (unset, nothing seen
 * yet) is narrower than everything.
 * @param current - the column's dtype so far, or null when no value has been seen
 * @param observed - the dtype of the newly observed value, or null for an unset cell
 * @returns the widened dtype (null only when both are null)
 */
export function widenDtype(current: InferredDtype | null, observed: InferredDtype | null): InferredDtype | null {
    if (current === null) {
        return observed;
    }
    if (observed === null) {
        return current;
    }
    return wideningRank(observed) > wideningRank(current) ? observed : current;
}

/**
 * Classify one text cell by the fixed lexical grammar. Exported as the reference implementation of
 * the text grammar of design section 5.1, which the text importers of graph-io must reproduce
 * exactly (invariant I15); the core itself only infers from JS values, so nothing in src/ calls it.
 * @public
 * @param text - the cell text, exactly as read (no trimming)
 * @returns bool, i32, f64 or string
 */
export function inferTextDtype(text: string): "bool" | "i32" | "f64" | "string" {
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
 * Classify one JS value by `typeof`.
 * @param value - the cell value
 * @returns the dtype, or null for an unset cell (`undefined` or `null`); E_COLUMN_TYPE for a bigint,
 * symbol or function, which no dtype can hold
 */
export function inferValueDtype(value: unknown): InferredDtype | null {
    if (value === undefined || value === null) {
        return null;
    }
    switch (typeof value) {
        case "boolean":
            return "bool";
        case "number":
            return Number.isInteger(value) && value >= I32_MIN && value <= I32_MAX ? "i32" : "f64";
        case "string":
            return "string";
        case "object":
            return "json";
        default:
            throw new GraphFormatError("E_COLUMN_TYPE", `a ${typeof value} value cannot be stored in a column`, {
                reason: "unsupported value type",
                found: typeof value,
            });
    }
}

/**
 * Parse a text cell into the storage value of a dtype the column has widened to. A text column
 * never widens to json, but a column that was inferred from JS values may later receive text; in
 * that case the text is stored as a string value. Part of the reference text grammar of design
 * section 5.1 (see inferTextDtype).
 * @public
 * @param text - the cell text
 * @param dtype - the column's current dtype
 * @returns a boolean for bool, a number for i32 / f64, the text itself for string / json
 */
export function parseText(text: string, dtype: InferredDtype): boolean | number | string {
    switch (dtype) {
        case "bool":
            if (text === "true") {
                return true;
            }
            if (text === "false") {
                return false;
            }
            throw new GraphFormatError("E_COLUMN_TYPE", `text "${text}" is not a bool`, { value: text, dtype });
        case "i32":
        case "f64": {
            const inferred = inferTextDtype(text);
            if (inferred === "string") {
                throw new GraphFormatError("E_COLUMN_TYPE", `text "${text}" is not a ${dtype}`, { value: text, dtype });
            }
            if (inferred === "bool") {
                return text === "true" ? 1 : 0;
            }
            return Number(text);
        }
        case "string":
        case "json":
            return text;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Convert a JS value to the storage value of a (possibly wider) inferred dtype, applying the
 * widening conversions of design section 5.1: a bool widened to a number becomes 1 / 0, a number or
 * boolean widened to string becomes its canonical JS text, anything widened to json is kept as is.
 * @param value - the cell value (never undefined or null; unset cells are handled by the caller)
 * @param dtype - the column's dtype
 * @returns the storage value; E_COLUMN_TYPE when the value cannot be represented in the dtype
 */
export function coerceValue(value: unknown, dtype: InferredDtype): unknown {
    switch (dtype) {
        case "bool":
            if (typeof value === "boolean") {
                return value;
            }
            break;
        case "i32":
        case "f64":
            if (typeof value === "number") {
                return value;
            }
            if (typeof value === "boolean") {
                return value ? 1 : 0;
            }
            break;
        case "string":
            if (typeof value === "string") {
                return value;
            }
            if (typeof value === "number" || typeof value === "boolean") {
                return String(value);
            }
            break;
        case "json":
            return value;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
    throw new GraphFormatError("E_COLUMN_TYPE", `a ${typeof value} value cannot be stored in a ${dtype} column`, {
        dtype,
        found: typeof value,
    });
}

/**
 * Infer the dtype of a whole column of JS values in one pass.
 * @param values - the cell values; undefined and null are unset
 * @returns the widened dtype, or null when every cell is unset
 */
export function inferValuesDtype(values: Iterable<unknown>): InferredDtype | null {
    let dtype: InferredDtype | null = null;
    for (const value of values) {
        dtype = widenDtype(dtype, inferValueDtype(value));
        if (dtype === "json") {
            break;
        }
    }
    return dtype;
}

/**
 * Infer the dtype of a whole column of text cells in one pass. Part of the reference text grammar
 * of design section 5.1 (see inferTextDtype).
 * @public
 * @param cells - the cell texts; an empty string is a string value, not an unset cell (the caller
 * decides how blanks map to unset before calling)
 * @returns the widened dtype, or null when the column is empty
 */
export function inferTextsDtype(cells: Iterable<string>): InferredDtype | null {
    let dtype: InferredDtype | null = null;
    for (const text of cells) {
        dtype = widenDtype(dtype, inferTextDtype(text));
        if (dtype === "string") {
            break;
        }
    }
    return dtype;
}

/**
 * A per-column inference accumulator for row-at-a-time sources (`addNodeRecord`, CSV streaming):
 * observe every cell and read `dtype`; each observe call reports whether the column widened so the
 * caller can reallocate storage and record the widening in FreezeReport.widened.
 */
export class DtypeInferrer {
    private current: InferredDtype | null = null;

    /**
     * The column's dtype so far.
     * @returns the widened dtype, or null until a set cell has been observed
     */
    get dtype(): InferredDtype | null {
        return this.current;
    }

    /**
     * Observe a JS value.
     * @param value - the cell value; undefined and null are unset and never widen
     * @returns true when the column's dtype changed
     */
    observeValue(value: unknown): boolean {
        return this.apply(inferValueDtype(value));
    }

    /**
     * Observe a text cell. Part of the reference text grammar of design section 5.1 (see
     * inferTextDtype).
     * @public
     * @param text - the cell text
     * @returns true when the column's dtype changed
     */
    observeText(text: string): boolean {
        return this.apply(inferTextDtype(text));
    }

    /**
     * Force the dtype to at least the given one (a declared lower bound, or the dtype of a column
     * being merged in).
     * @param dtype - the dtype to widen to
     * @returns true when the column's dtype changed
     */
    widenTo(dtype: InferredDtype): boolean {
        return this.apply(dtype);
    }

    private apply(observed: InferredDtype | null): boolean {
        const next = widenDtype(this.current, observed);
        if (next === this.current) {
            return false;
        }
        this.current = next;
        return true;
    }
}
