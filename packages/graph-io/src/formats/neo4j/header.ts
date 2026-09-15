/**
 * The neo4j-admin import header grammar (research note 07 section 2.6): every header cell is
 * `<name>:<field type>(<id space>){<options>}` where the name and every later part are optional.
 * The field types `ID`, `LABEL`, `START_ID`, `END_ID`, `TYPE` and `IGNORE` are structural; any
 * other field type is a property type of design section 5.1 (`int`, `long`, `float`, `double`,
 * `boolean`, `byte`, `short`, `char`, `string`, `point`, `date`, `localtime`, `time`,
 * `localdatetime`, `datetime`, `duration`, each with an optional `[]` suffix); a cell without a
 * colon is a string property. An id space may follow `ID`, `START_ID` and `END_ID`; brace options
 * (`label`, `id-type`) may follow any structural type.
 */

import { GraphFormatError } from "@graphty/graph-format";

/** The structural field types and the marker for a property column. */
export type FieldKind = "ID" | "LABEL" | "START_ID" | "END_ID" | "TYPE" | "IGNORE" | "PROPERTY";

/** One parsed header cell. */
export interface HeaderField {
    /** The cell text as written. */
    readonly text: string;
    /** The property name (the id property name for `name:ID`); empty when none. */
    readonly name: string;
    /** The field kind. */
    readonly kind: FieldKind;
    /** The property type text for a PROPERTY field (null when the cell declares none); null otherwise. */
    readonly type: string | null;
    /** The id space of an ID / START_ID / END_ID field, or null. */
    readonly space: string | null;
    /** The brace options, keys lower-cased. */
    readonly options: ReadonlyMap<string, string>;
}

const STRUCTURAL: ReadonlySet<string> = new Set(["ID", "LABEL", "START_ID", "END_ID", "TYPE", "IGNORE"]);
const SPACE_KINDS: ReadonlySet<string> = new Set(["ID", "START_ID", "END_ID"]);
const HEADER_MARKER = /^[^:{}()]*:\s*(ID|START_ID|END_ID)\s*(\(|\{|$)/i;
const CELL = /^([^:(){}]*)(?::([^:(){}]*))?(?:\(([^)]*)\))?(?:\{([^}]*)\})?$/;

/**
 * Whether a record looks like a section header: some cell declares an `ID`, `START_ID` or `END_ID`
 * field. The neo4j-admin convention allows several files, each with its own header; graphty's
 * single-file convention (the CSVDataSource of graphty-element) concatenates sections, so a header
 * may appear anywhere.
 * @param cells - the record's cells
 * @param count - how many cells are valid
 * @returns true when the record is a header
 */
export function isHeaderRecord(cells: readonly string[], count: number): boolean {
    for (let i = 0; i < count; i++) {
        const cell = cells[i];
        // every header marker holds a colon; the indexOf keeps the regex off the data rows
        if (cell.indexOf(":") >= 0 && HEADER_MARKER.test(cell.trim())) {
            return true;
        }
    }
    return false;
}

/**
 * Parse one header cell.
 * @param text - the cell text
 * @returns the field; E_UNSUPPORTED (reason "header") for a cell outside the grammar
 */
export function parseHeaderField(text: string): HeaderField {
    const trimmed = text.trim();
    const m = CELL.exec(trimmed);
    if (m === null) {
        throw headerError(text, "not a neo4j header cell");
    }
    const name = m[1].trim();
    const rawType = m[2] === undefined ? null : m[2].trim();
    const space = m[3] === undefined ? null : m[3].trim();
    const options = parseOptions(text, m[4]);
    if (rawType === null) {
        // `name` alone: a string property
        if (name.length === 0) {
            throw headerError(text, "an empty header cell");
        }
        if (space !== null || options.size > 0) {
            throw headerError(text, "an id space or options need a field type");
        }
        return { text, name, kind: "PROPERTY", type: null, space: null, options };
    }
    if (rawType.length === 0) {
        throw headerError(text, "an empty field type");
    }
    const upper = rawType.toUpperCase();
    if (STRUCTURAL.has(upper)) {
        const kind = upper as Exclude<FieldKind, "PROPERTY">;
        if (space !== null && !SPACE_KINDS.has(kind)) {
            throw headerError(text, `an id space is only allowed on ID, START_ID and END_ID, not ${kind}`);
        }
        if (name.length > 0 && kind !== "ID" && kind !== "IGNORE") {
            throw headerError(text, `a ${kind} field cannot be stored as a property`);
        }
        return { text, name, kind, type: null, space: space === "" ? null : space, options };
    }
    if (name.length === 0) {
        throw headerError(text, "a property column needs a name");
    }
    if (space !== null) {
        throw headerError(text, "an id space is only allowed on ID, START_ID and END_ID");
    }
    return { text, name, kind: "PROPERTY", type: rawType, space: null, options };
}

/**
 * Parse the `{key:value, key:value}` options of a header cell.
 * @param text - the whole cell, for errors
 * @param body - the text between the braces, or undefined when absent
 * @returns the options with lower-cased keys
 */
function parseOptions(text: string, body: string | undefined): ReadonlyMap<string, string> {
    const options = new Map<string, string>();
    if (body === undefined || body.trim().length === 0) {
        return options;
    }
    for (const part of body.split(",")) {
        const colon = part.indexOf(":");
        if (colon < 0) {
            throw headerError(text, `option "${part.trim()}" is not key:value`);
        }
        const key = part.slice(0, colon).trim().toLowerCase();
        const value = part.slice(colon + 1).trim();
        if (key.length === 0) {
            throw headerError(text, "an option needs a key");
        }
        options.set(key, value);
    }
    return options;
}

/**
 * The error of a malformed header cell.
 * @param text - the cell
 * @param reason - why
 * @returns the error
 */
function headerError(text: string, reason: string): GraphFormatError {
    return new GraphFormatError("E_UNSUPPORTED", `header cell "${text}": ${reason}`, {
        reason: "header",
        cell: text,
        detail: reason,
    });
}

/**
 * Write a header cell from its parts, the inverse of parseHeaderField.
 * @param name - the property name, or empty
 * @param type - the field type or property type
 * @param space - the id space, or null
 * @returns the cell text
 */
export function formatHeaderField(name: string, type: string, space: string | null): string {
    const base = `${name}:${type}`;
    return space === null ? base : `${base}(${space})`;
}
