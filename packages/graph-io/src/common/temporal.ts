/**
 * Temporal values (design section 5.1): `date` / `dateTime` attributes, GEXF time bounds and Neo4j
 * temporal types become f64 epoch milliseconds (or the raw numeric time when the file's timeformat
 * is integer / double). When a value's canonical re-formatting would differ from the source text (a
 * UTC offset, fractional seconds beyond milliseconds, a date without a time in a dateTime column)
 * the importer also keeps the lexical form in a companion `<column>.text` column with role
 * `timeText`; the exporter emits the text when present and formats the number otherwise.
 *
 * Only ISO-8601 forms are parsed (XSD date / dateTime / time and Neo4j's literals), by a fixed
 * grammar rather than Date.parse, so two hosts agree on every value.
 */

import { type ColumnDecl, type ColumnRole, GraphFormatError } from "@graphty/graph-format";

/** The temporal kinds a declared type can map to; each fixes the parse grammar and the canonical text. */
export type TemporalKind = "date" | "dateTime" | "localDateTime" | "time" | "localTime";

/**
 * The GEXF `timeformat` values (design section 5.9).
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export type TimeFormat = "integer" | "double" | "date" | "dateTime";

/** A parsed temporal value: the number stored in the column and the source text when it must be kept. */
export interface TemporalValue {
    /** Epoch milliseconds (date, dateTime, localDateTime), milliseconds since midnight (time, localTime), or the raw number. */
    readonly value: number;
    /** The source text when formatTemporal(value) would not reproduce it; null otherwise. */
    readonly text: string | null;
}

/** The role of a companion text column. */
export const TIME_TEXT_ROLE: ColumnRole = "timeText";

/** The suffix of a companion text column's name. */
export const TIME_TEXT_SUFFIX = ".text";

const DATE_TIME =
    /^(-?[0-9]{4,})-([0-9]{2})-([0-9]{2})(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,9}))?)?)?(Z|z|[+-][0-9]{2}(?::?[0-9]{2})?)?$/;
const TIME = /^([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,9}))?)?(Z|z|[+-][0-9]{2}(?::?[0-9]{2})?)?$/;
const NUMERIC = /^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/;

const MS_PER_MINUTE = 60_000;
const MS_PER_DAY = 86_400_000;

/**
 * Parse an ISO-8601 temporal text of one kind.
 * @param text - the source text (already trimmed)
 * @param kind - the temporal kind
 * @returns the value and, when the canonical form differs, the source text; E_COLUMN_TYPE when the text is not of the kind
 */
export function parseTemporal(text: string, kind: TemporalKind): TemporalValue {
    let value: number;
    switch (kind) {
        case "date":
        case "dateTime":
        case "localDateTime":
            value = parseDateTime(text, kind);
            break;
        case "time":
        case "localTime":
            value = parseTime(text, kind);
            break;
        default: {
            const name: string = kind;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown temporal kind ${name}`, { kind: name });
        }
    }
    const canonical = formatTemporal(value, kind);
    return { value, text: canonical === text ? null : text };
}

/**
 * Parse a GEXF time bound (`start`, `end`, `timestamp`, a spell or an attvalue time) under the
 * graph's timeformat: integer / double values are numbers as written; date / dateTime values are
 * ISO text; when the format is unknown (absent header) numeric text is a number and anything else
 * is tried as dateTime.
 * @param text - the source text
 * @param timeFormat - the graph's timeformat, or null when the file declares none
 * @returns the value and the source text when it must be kept
 */
export function parseTimeText(text: string, timeFormat: TimeFormat | null): TemporalValue {
    const trimmed = text.trim();
    switch (timeFormat) {
        case "integer":
        case "double":
            return parseNumericTime(trimmed);
        case "date":
            return parseTemporal(trimmed, "date");
        case "dateTime":
            return parseTemporal(trimmed, "dateTime");
        case null:
            return NUMERIC.test(trimmed) ? parseNumericTime(trimmed) : parseTemporal(trimmed, "dateTime");
        default: {
            const name: string = timeFormat;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown timeformat ${name}`, { timeFormat: name });
        }
    }
}

/**
 * Format a temporal value in its canonical ISO-8601 form.
 * @param value - the number as stored
 * @param kind - the temporal kind
 * @returns the canonical text: `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS[.mmm]Z`, `YYYY-MM-DDTHH:MM:SS[.mmm]`,
 * `HH:MM:SS[.mmm]Z` or `HH:MM:SS[.mmm]`
 */
export function formatTemporal(value: number, kind: TemporalKind): string {
    switch (kind) {
        case "date": {
            const d = new Date(value);
            return `${year(d)}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
        }
        case "dateTime":
        case "localDateTime": {
            const d = new Date(value);
            const date = `${year(d)}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
            const time = clock(d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds(), d.getUTCMilliseconds());
            return kind === "dateTime" ? `${date}T${time}Z` : `${date}T${time}`;
        }
        case "time":
        case "localTime": {
            const ms = ((value % MS_PER_DAY) + MS_PER_DAY) % MS_PER_DAY;
            const h = Math.floor(ms / 3_600_000);
            const m = Math.floor((ms % 3_600_000) / MS_PER_MINUTE);
            const s = Math.floor((ms % MS_PER_MINUTE) / 1000);
            const text = clock(h, m, s, ms % 1000);
            return kind === "time" ? `${text}Z` : text;
        }
        default: {
            const name: string = kind;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown temporal kind ${name}`, { kind: name });
        }
    }
}

/**
 * Format a GEXF time bound under the graph's timeformat: the number's shortest text for integer /
 * double, the canonical ISO form otherwise.
 * @param value - the value as stored
 * @param timeFormat - the graph's timeformat, or null for a file that declares none (double)
 * @returns the text
 */
export function formatTimeValue(value: number, timeFormat: TimeFormat | null): string {
    switch (timeFormat) {
        case "date":
            return formatTemporal(value, "date");
        case "dateTime":
            return formatTemporal(value, "dateTime");
        case "integer":
        case "double":
        case null:
            return String(value);
        default: {
            const name: string = timeFormat;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown timeformat ${name}`, { timeFormat: name });
        }
    }
}

/**
 * The declaration of the companion text column of a temporal column (design section 5.1).
 * @param column - the temporal column's name
 * @returns a string column named `<column>.text` with role timeText and `extra.for` naming the column
 */
export function timeTextCompanion(column: string): ColumnDecl {
    return {
        name: column + TIME_TEXT_SUFFIX,
        dtype: "string",
        role: TIME_TEXT_ROLE,
        nullable: true,
        extra: { for: column },
    };
}

/**
 * Parse a numeric time bound.
 * @param text - the trimmed source text
 * @returns the number; the text is kept when String(value) differs from it
 */
function parseNumericTime(text: string): TemporalValue {
    if (!NUMERIC.test(text)) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is not a numeric time`, { value: text });
    }
    const value = Number(text);
    return { value, text: String(value) === text ? null : text };
}

/**
 * Parse a date or date-time text into epoch milliseconds.
 * @param text - the source text
 * @param kind - date (a time part is accepted and kept in the companion), dateTime or localDateTime
 * @returns epoch milliseconds
 */
function parseDateTime(text: string, kind: "date" | "dateTime" | "localDateTime"): number {
    const m = DATE_TIME.exec(text);
    if (m === null) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is not an ISO-8601 ${kind}`, { value: text, kind });
    }
    const y = Number(m[1]);
    const month = Number(m[2]);
    const day = Number(m[3]);
    const hour = m[4] === undefined ? 0 : Number(m[4]);
    const minute = m[5] === undefined ? 0 : Number(m[5]);
    const second = m[6] === undefined ? 0 : Number(m[6]);
    const ms = m[7] === undefined ? 0 : fractionMs(m[7]);
    if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 24 || minute > 59 || second > 60) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is out of range for a ${kind}`, { value: text, kind });
    }
    // Date.UTC maps years 0..99 to 1900..1999; the setters do not
    const d = new Date(0);
    d.setUTCFullYear(y, month - 1, day);
    d.setUTCHours(hour, minute, second, ms);
    if (d.getUTCDate() !== day && hour !== 24) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is not a calendar date`, { value: text, kind });
    }
    const offset = m[8] === undefined ? 0 : offsetMinutes(m[8]);
    return d.getTime() - offset * MS_PER_MINUTE;
}

/**
 * Parse a time-of-day text into milliseconds since midnight (UTC for `time` with a zone).
 * @param text - the source text
 * @param kind - time or localTime
 * @returns milliseconds since midnight in 0..86400000
 */
function parseTime(text: string, kind: "time" | "localTime"): number {
    const m = TIME.exec(text);
    if (m === null) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is not an ISO-8601 ${kind}`, { value: text, kind });
    }
    const hour = Number(m[1]);
    const minute = Number(m[2]);
    const second = m[3] === undefined ? 0 : Number(m[3]);
    const ms = m[4] === undefined ? 0 : fractionMs(m[4]);
    if (hour > 24 || minute > 59 || second > 60) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${text}" is out of range for a ${kind}`, { value: text, kind });
    }
    const local = ((hour * 60 + minute) * 60 + second) * 1000 + ms;
    const offset = m[5] === undefined ? 0 : offsetMinutes(m[5]);
    const utc = local - offset * MS_PER_MINUTE;
    return ((utc % MS_PER_DAY) + MS_PER_DAY) % MS_PER_DAY;
}

/**
 * Milliseconds of a fractional-second text (1 to 9 digits), truncated to whole milliseconds.
 * @param digits - the digits after the decimal point
 * @returns 0..999
 */
function fractionMs(digits: string): number {
    return Math.floor(Number(`0.${digits}`) * 1000);
}

/**
 * The minutes of a zone designator.
 * @param zone - "Z", "+HH:MM", "+HHMM" or "+HH"
 * @returns signed minutes east of UTC
 */
function offsetMinutes(zone: string): number {
    if (zone === "Z" || zone === "z") {
        return 0;
    }
    const sign = zone.startsWith("-") ? -1 : 1;
    const digits = zone.slice(1).replace(":", "");
    const hours = Number(digits.slice(0, 2));
    const minutes = digits.length > 2 ? Number(digits.slice(2, 4)) : 0;
    return sign * (hours * 60 + minutes);
}

/**
 * Two-digit zero-padded text.
 * @param n - 0..99
 * @returns the text
 */
function pad2(n: number): string {
    return n < 10 ? `0${n}` : String(n);
}

/**
 * The year of a UTC date as at least four digits (negative years keep their sign).
 * @param d - the date
 * @returns the text
 */
function year(d: Date): string {
    const y = d.getUTCFullYear();
    const abs = String(Math.abs(y)).padStart(4, "0");
    return y < 0 ? `-${abs}` : abs;
}

/**
 * `HH:MM:SS` with `.mmm` appended when the milliseconds are not zero.
 * @param h - hours
 * @param m - minutes
 * @param s - seconds
 * @param ms - milliseconds
 * @returns the text
 */
function clock(h: number, m: number, s: number, ms: number): string {
    const base = `${pad2(h)}:${pad2(m)}:${pad2(s)}`;
    return ms === 0 ? base : `${base}.${String(ms).padStart(3, "0")}`;
}
