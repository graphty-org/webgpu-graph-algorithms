/**
 * Number formatting for exporters (design section 3.7): an f32 weight is written as the shortest
 * decimal that round-trips through Math.fround (never `String(x)`, which prints
 * 0.10000000149011612 for a stored 0.1); an f64 value as its shortest JS text; a GML real always
 * with a decimal point so the dtype survives (design section 8.5); non-finite values as the
 * spelling the target syntax accepts.
 */

/**
 * The shortest decimal text that reads back to the same f32 value through Math.fround.
 * @param value - an f32 value (a JS number holding one)
 * @returns the text; "Infinity" / "-Infinity" / "NaN" for the non-finite values, "0" for both zeros
 */
export function formatF32(value: number): string {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    if (value === 0) {
        return "0";
    }
    for (let digits = 1; digits <= 9; digits++) {
        const text = value.toPrecision(digits);
        if (Math.fround(Number(text)) === value) {
            return String(Number(text));
        }
    }
    return String(value);
}

/**
 * The shortest text of an f64 value: `String(x)`, which is already the shortest round-tripping
 * decimal in JS.
 * @param value - the value
 * @returns the text; "Infinity" / "-Infinity" / "NaN" for the non-finite values
 */
export function formatF64(value: number): string {
    return String(value);
}

/**
 * A decimal text with a decimal point or an exponent guaranteed for a finite value (`2.0`,
 * `1e+21`, `1.5e-7`), so an
 * untyped re-import keeps the column f64 rather than i32 (design section 8.5); negative zero is
 * written `-0.0` so it reads back as -0; non-finite values are the JS spellings (`Infinity`,
 * `-Infinity`, `NaN`), which the CSV / DOT / Pajek importers read back as text. The one
 * implementation of the "decimal point guaranteed" rule for every text format; GML has its own
 * spellings of the non-finite values in formatGmlReal().
 * @param value - the value
 * @param dtype - the column dtype the value comes from; f32 values use the shortest fround-round-trip text
 * @returns the text
 */
export function formatDecimal(value: number, dtype: "f32" | "f64" | "i32" | "u32" | "u8" = "f64"): string {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    if (Object.is(value, -0)) {
        return "-0.0";
    }
    const text = formatNumber(value, dtype);
    // an exponent form (`1e+21`, `1.5e-7`) is already f64 text under the 5.1 grammar
    return text.includes(".") || text.includes("e") ? text : `${text}.0`;
}

/**
 * A GML real: the shortest text with a decimal point guaranteed (`2.0`, `1.0e-7`), so a re-import
 * keeps the column real rather than int. Non-finite values have no GML spelling and are written
 * as the texts NetworkX's writer emits and its reader accepts, `+INF`, `-INF` and `NAN` (the
 * lowercase `inf` / `nan` would lex as keys there).
 * @param value - the value
 * @param dtype - the column dtype the value comes from; f32 values use the shortest fround-round-trip text
 * @returns the text
 */
export function formatGmlReal(value: number, dtype: "f32" | "f64" | "i32" | "u32" | "u8" = "f64"): string {
    if (Number.isNaN(value)) {
        return "NAN";
    }
    if (!Number.isFinite(value)) {
        return value > 0 ? "+INF" : "-INF";
    }
    const text = formatDecimal(value, dtype);
    if (text.includes(".")) {
        return text;
    }
    // NetworkX's real pattern requires a decimal point even in the exponent form (`1.0e-7`)
    const e = text.indexOf("e");
    return `${text.slice(0, e)}.0${text.slice(e)}`;
}

/**
 * An integer text for a value known to be integral (an i32 / u32 / u8 cell, an f64 that holds an
 * integer), avoiding the exponent form `String()` uses above 1e21.
 * @param value - an integral value
 * @returns the digits
 */
export function formatInteger(value: number): string {
    if (Number.isSafeInteger(value) || !Number.isFinite(value)) {
        return String(value);
    }
    return BigInt(value).toString();
}

/**
 * The text of a numeric cell of any numeric dtype, dispatching on the dtype: f32 through
 * formatF32, everything else through formatF64.
 * @param value - the value
 * @param dtype - the column dtype the value came from
 * @returns the text
 */
export function formatNumber(value: number, dtype: "f32" | "f64" | "i32" | "u32" | "u8"): string {
    return dtype === "f32" ? formatF32(value) : formatF64(value);
}
