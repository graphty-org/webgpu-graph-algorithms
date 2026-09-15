/**
 * The declared-type table of design section 5.1: what the type text of a GEXF `<attribute>`, a
 * GraphML `<key>`, a GML value class or a Neo4j CSV header maps to, and how a value's text is
 * parsed once the column dtype is known.
 *
 * Declared types map exactly: `double` and `long` -> f64, `float` -> f32, `int` / `integer` /
 * `short` / `byte` -> i32, `boolean` -> bool, `string` / `anyURI` / `char` -> string, `bigdecimal`
 * / `biginteger` -> string, `date` / `dateTime` and the Neo4j temporal types -> f64 (epoch or
 * time-of-day milliseconds, design section 5.1), `list*` and `[]` -> list of the item type, GML
 * `record` and Neo4j `point` -> json. A declared `long` is stored as f64 with a precision flag
 * (values beyond 2^53 are a `precision` issue) unless the importer option `long: "string"` keeps the
 * whole column as text. Neo4j `duration` has no epoch and is kept as text. The declared type text is
 * recorded in `origin.type` in every case so the exporter can restore the declaration.
 */

import { type Dtype, GraphFormatError, type ScalarDtype } from "@graphty/graph-format";

import { parseTemporal, type TemporalKind } from "./temporal.js";

/** The formats whose declarations this table covers. */
export type DeclaringFormat = "gexf" | "graphml" | "gml" | "neo4j";

/** How the text of one scalar value (or one list item) is parsed. */
type ValueKind =
    "boolean" | "integer" | "long" | "float" | "double" | "string" | "temporal" | "duration" | "point" | "json";

/** The resolved storage of a declared type. */
export interface DeclaredTypeSpec {
    /** The declared type text as given by the file (origin.type). */
    readonly declared: string;
    /** The column dtype. */
    readonly dtype: Dtype;
    /** The item dtype of a list column, null otherwise. */
    readonly itemDtype: ScalarDtype | null;
    /** How each scalar value or list item text is parsed. */
    readonly kind: ValueKind;
    /** Whether the column is a list. */
    readonly list: boolean;
    /** The temporal kind of a temporal value, null otherwise. */
    readonly temporal: TemporalKind | null;
    /** Whether values may exceed 2^53 (a long stored as f64), so the importer checks Number.isSafeInteger. */
    readonly precision: boolean;
}

/** A scalar entry of the table before the list flag and the long mode are applied. */
interface ScalarEntry {
    readonly dtype: ScalarDtype;
    readonly kind: ValueKind;
    readonly temporal?: TemporalKind;
    readonly precision?: boolean;
}

const BOOLEAN: ScalarEntry = { dtype: "bool", kind: "boolean" };
const INTEGER: ScalarEntry = { dtype: "i32", kind: "integer" };
const LONG: ScalarEntry = { dtype: "f64", kind: "long", precision: true };
const FLOAT: ScalarEntry = { dtype: "f32", kind: "float" };
const DOUBLE: ScalarEntry = { dtype: "f64", kind: "double" };
const STRING: ScalarEntry = { dtype: "string", kind: "string" };
const DATE: ScalarEntry = { dtype: "f64", kind: "temporal", temporal: "date" };
const DATE_TIME: ScalarEntry = { dtype: "f64", kind: "temporal", temporal: "dateTime" };
const LOCAL_DATE_TIME: ScalarEntry = { dtype: "f64", kind: "temporal", temporal: "localDateTime" };
const TIME: ScalarEntry = { dtype: "f64", kind: "temporal", temporal: "time" };
const LOCAL_TIME: ScalarEntry = { dtype: "f64", kind: "temporal", temporal: "localTime" };
const DURATION: ScalarEntry = { dtype: "string", kind: "duration" };
const POINT: ScalarEntry = { dtype: "json", kind: "point" };
const RECORD: ScalarEntry = { dtype: "json", kind: "json" };

/** GEXF 1.2 / 1.3 scalar types (the `list*` variants are derived). */
const GEXF_SCALARS: Readonly<Record<string, ScalarEntry>> = {
    integer: INTEGER,
    long: LONG,
    double: DOUBLE,
    float: FLOAT,
    boolean: BOOLEAN,
    string: STRING,
    anyuri: STRING,
    char: STRING,
    byte: INTEGER,
    short: INTEGER,
    bigdecimal: STRING,
    biginteger: STRING,
    date: DATE,
    datetime: DATE_TIME,
};

/** GraphML `attr.type` values. */
const GRAPHML_SCALARS: Readonly<Record<string, ScalarEntry>> = {
    boolean: BOOLEAN,
    int: INTEGER,
    long: LONG,
    float: FLOAT,
    double: DOUBLE,
    string: STRING,
};

/** GML value classes (per value in the file; the GML importer names them per column). */
const GML_SCALARS: Readonly<Record<string, ScalarEntry>> = {
    int: INTEGER,
    real: DOUBLE,
    string: STRING,
    record: RECORD,
};

/** Neo4j CSV header types and Cypher property type names. */
const NEO4J_SCALARS: Readonly<Record<string, ScalarEntry>> = {
    int: INTEGER,
    integer: LONG,
    long: LONG,
    float: FLOAT,
    double: DOUBLE,
    boolean: BOOLEAN,
    byte: INTEGER,
    short: INTEGER,
    char: STRING,
    string: STRING,
    point: POINT,
    date: DATE,
    localtime: LOCAL_TIME,
    time: TIME,
    localdatetime: LOCAL_DATE_TIME,
    datetime: DATE_TIME,
    duration: DURATION,
};

const TABLES: Readonly<Record<DeclaringFormat, Readonly<Record<string, ScalarEntry>>>> = {
    gexf: GEXF_SCALARS,
    graphml: GRAPHML_SCALARS,
    gml: GML_SCALARS,
    neo4j: NEO4J_SCALARS,
};

const INTEGER_TEXT = /^[+-]?[0-9]+$/;
const DECIMAL_TEXT = /^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

/**
 * Resolve a declared type text of one format to its storage.
 * @param format - the declaring format
 * @param type - the type text as written (matched case-insensitively; GEXF `list*` and Neo4j `[]` denote lists)
 * @param long - how declared long / integer-64 columns are stored (the importer option)
 * @returns the spec, or null when the format does not define the type (the importer reports it and keeps the text)
 */
export function mapDeclaredType(
    format: DeclaringFormat,
    type: string,
    long: "f64" | "string",
): DeclaredTypeSpec | null {
    const table = TABLES[format];
    let key = type.trim().toLowerCase();
    let list = false;
    if (format === "gexf" && key.startsWith("list") && key.length > 4) {
        list = true;
        key = key.slice(4);
    } else if (format === "neo4j" && key.endsWith("[]")) {
        list = true;
        key = key.slice(0, -2);
    } else if (format === "gml" && key === "list") {
        return {
            declared: type,
            dtype: "list",
            itemDtype: "json",
            kind: "json",
            list: true,
            temporal: null,
            precision: false,
        };
    }
    let entry = table[key];
    if (entry === undefined) {
        return null;
    }
    if (entry.kind === "long" && long === "string") {
        entry = STRING;
    }
    return {
        declared: type,
        dtype: list ? "list" : entry.dtype,
        itemDtype: list ? entry.dtype : null,
        kind: entry.kind,
        list,
        temporal: entry.temporal ?? null,
        precision: entry.precision ?? false,
    };
}

/**
 * The spec of an untyped attribute: a string column with the given (or absent) type text.
 * @param declared - the type text to record in origin.type, or null
 * @returns a string spec
 */
export function stringSpec(declared: string | null): DeclaredTypeSpec {
    return {
        declared: declared ?? "",
        dtype: "string",
        itemDtype: null,
        kind: "string",
        list: false,
        temporal: null,
        precision: false,
    };
}

/**
 * Parse the text of one scalar value (or list item) by its kind. Numeric and boolean kinds trim
 * surrounding whitespace (pretty-printed XML puts newlines around `<data>` text); string kinds keep
 * the text exactly.
 * @param text - the value text
 * @param kind - the value kind
 * @param temporal - the temporal kind when `kind` is "temporal"
 * @returns a boolean, a number, a string or (point / json) a JSON value; E_COLUMN_TYPE when the text is not of the kind
 */
export function parseScalarText(text: string, kind: ValueKind, temporal: TemporalKind | null = null): unknown {
    switch (kind) {
        case "boolean": {
            const value = parseBooleanText(text);
            if (value === null) {
                throw typeError(text, kind);
            }
            return value;
        }
        case "integer": {
            const trimmed = text.trim();
            if (!INTEGER_TEXT.test(trimmed)) {
                throw typeError(text, kind);
            }
            const value = Number(trimmed);
            if (value < I32_MIN || value > I32_MAX) {
                throw new GraphFormatError("E_COLUMN_TYPE", `"${trimmed}" is outside the 32-bit integer range`, {
                    value: trimmed,
                    kind,
                });
            }
            return value;
        }
        case "long": {
            const trimmed = text.trim();
            if (!INTEGER_TEXT.test(trimmed)) {
                throw typeError(text, kind);
            }
            return Number(trimmed);
        }
        case "float":
        case "double":
            return parseDecimalText(text, kind);
        case "string":
        case "duration":
            return text;
        case "temporal": {
            if (temporal === null) {
                throw new GraphFormatError("E_COLUMN_TYPE", "a temporal value needs its temporal kind", { kind });
            }
            return parseTemporal(text.trim(), temporal).value;
        }
        case "point":
            return parsePointText(text);
        case "json":
            return parseJsonText(text);
        default: {
            const name: string = kind;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown value kind ${name}`, { kind: name });
        }
    }
}

/**
 * Parse a boolean text the way the formats write it: `true` / `false` in any case, or `1` / `0`.
 * @param text - the value text
 * @returns the boolean, or null when the text is neither
 */
export function parseBooleanText(text: string): boolean | null {
    switch (text.trim().toLowerCase()) {
        case "true":
        case "1":
            return true;
        case "false":
        case "0":
            return false;
        default:
            return null;
    }
}

/**
 * Parse a float / double text: a decimal or exponent literal, or the XSD / JSON spellings of the
 * non-finite values (`INF`, `-INF`, `Infinity`, `-Infinity`, `NaN`).
 * @param text - the value text
 * @param kind - float or double, for the error
 * @returns the number
 */
export function parseDecimalText(text: string, kind: ValueKind = "double"): number {
    const trimmed = text.trim();
    if (DECIMAL_TEXT.test(trimmed)) {
        return Number(trimmed);
    }
    switch (trimmed) {
        case "INF":
        case "+INF":
        case "Infinity":
        case "+Infinity":
            return Infinity;
        case "-INF":
        case "-Infinity":
            return -Infinity;
        case "NaN":
            return NaN;
        default:
            throw typeError(text, kind);
    }
}

/**
 * Parse a Neo4j point literal `{x:1.0, y:2.0, crs:'cartesian'}` (also `latitude` / `longitude` /
 * `height` / `z` / `srid`) into a JSON object; a JSON object text is accepted as well.
 * @param text - the value text
 * @returns an object with numeric coordinates and string crs
 */
export function parsePointText(text: string): Readonly<Record<string, number | string>> {
    const trimmed = text.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
        throw typeError(text, "point");
    }
    const out: Record<string, number | string> = {};
    const inner = trimmed.slice(1, -1).trim();
    if (inner.length === 0) {
        return out;
    }
    for (const part of inner.split(",")) {
        const colon = part.indexOf(":");
        if (colon < 0) {
            throw typeError(text, "point");
        }
        const key = part
            .slice(0, colon)
            .trim()
            .replace(/^["']|["']$/g, "");
        const raw = part.slice(colon + 1).trim();
        if (key.length === 0) {
            throw typeError(text, "point");
        }
        if (DECIMAL_TEXT.test(raw)) {
            out[key] = Number(raw);
        } else if ((raw.startsWith("'") && raw.endsWith("'")) || (raw.startsWith('"') && raw.endsWith('"'))) {
            out[key] = raw.slice(1, -1);
        } else {
            throw typeError(text, "point");
        }
    }
    return out;
}

/**
 * Parse JSON text into a value.
 * @param text - the JSON text
 * @returns the parsed value
 */
function parseJsonText(text: string): unknown {
    try {
        return JSON.parse(text) as unknown;
    } catch {
        throw typeError(text, "json");
    }
}

/**
 * The E_COLUMN_TYPE error of a text that is not of a kind.
 * @param text - the text
 * @param kind - the kind
 * @returns the error
 */
function typeError(text: string, kind: ValueKind): GraphFormatError {
    const shown = text.length > 40 ? `${text.slice(0, 40)}...` : text;
    return new GraphFormatError("E_COLUMN_TYPE", `"${shown}" is not a ${kind}`, { value: text, kind });
}
