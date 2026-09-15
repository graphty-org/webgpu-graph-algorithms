/**
 * Header handling of the CSV importer: whether the first row is a header, which columns hold the
 * endpoints, the weight, the Gephi per-row `Type`, the edge id and the label (by name lists and
 * by position), and the synthesised names of a headerless file (`column1`, `column2`, ...).
 *
 * Name resolution is exact first, then case-insensitive, over a candidate list in priority order,
 * so `source` / `Source` / `SOURCE`, `src`, `from` all resolve without configuration; the Gephi
 * dialect (exact `Source` and `Target` headers) additionally makes an exact `Type` column the
 * per-row direction. A caller may name any column explicitly or give its 0-based position.
 */

import { GraphFormatError } from "@graphty/graph-format";

import { inferTextDtype } from "../../common/text.js";

/** A column named by header text or by 0-based position. */
export type CsvColumnRef = string | number;

/** Header candidates of the source endpoint, in priority order (matched exactly, then case-insensitively). */
export const SOURCE_NAMES: readonly string[] = Object.freeze([
    "source",
    "src",
    "from",
    "start",
    "source_id",
    "sourceid",
    "fromnodeid",
    "start_id",
    ":start_id",
]);

/** Header candidates of the target endpoint. */
export const TARGET_NAMES: readonly string[] = Object.freeze([
    "target",
    "dst",
    "dest",
    "to",
    "end",
    "target_id",
    "targetid",
    "tonodeid",
    "end_id",
    ":end_id",
]);

/** Header candidates of a node table's id column. */
export const ID_NAMES: readonly string[] = Object.freeze(["id", "node", "name", "key"]);

/** Header candidates of a label column (node or edge). */
export const LABEL_NAMES: readonly string[] = Object.freeze(["label"]);

/** Header candidates of an edge id column. */
export const EDGE_ID_NAMES: readonly string[] = Object.freeze(["id"]);

/** The Gephi per-row direction column. */
export const TYPE_NAME = "Type";

/** Every name that marks a first row as a header when the caller leaves detection to the importer. */
const HEADER_MARKERS: ReadonlySet<string> = new Set([
    ...SOURCE_NAMES,
    ...TARGET_NAMES,
    ...ID_NAMES,
    ...LABEL_NAMES,
    "weight",
    "type",
    "value",
]);

/**
 * Find a column by name: candidates are tried in priority order, each first exactly and then
 * case-insensitively, so `Source` (a case variant of the first candidate) beats an exact `from`.
 * @param names - the header names
 * @param candidates - the names to look for, in priority order
 * @returns the column index, or -1
 */
export function findColumn(names: readonly string[], candidates: readonly string[]): number {
    for (const candidate of candidates) {
        const exact = names.indexOf(candidate);
        if (exact >= 0) {
            return exact;
        }
        const lower = candidate.toLowerCase();
        const loose = names.findIndex((name) => name.toLowerCase() === lower);
        if (loose >= 0) {
            return loose;
        }
    }
    return -1;
}

/**
 * Resolve an explicit column option: a 0-based position (checked against the width) or a header
 * name (exact, then case-insensitive). An option naming a column the file does not have is
 * E_UNSUPPORTED (details.option, details.found): the file cannot be read as requested.
 * @param names - the header names
 * @param ref - the option value
 * @param option - the option name, for the error
 * @returns the column index
 */
export function resolveColumnRef(names: readonly string[], ref: CsvColumnRef, option: string): number {
    if (typeof ref === "number") {
        if (!Number.isInteger(ref) || ref < 0 || ref >= names.length) {
            throw new GraphFormatError(
                "E_UNSUPPORTED",
                `option ${option}: column ${ref} does not exist (the file has ${names.length} column(s))`,
                { option, found: ref, columns: names.length },
            );
        }
        return ref;
    }
    const index = findColumn(names, [ref]);
    if (index < 0) {
        throw new GraphFormatError("E_UNSUPPORTED", `option ${option}: no column named ${JSON.stringify(ref)}`, {
            option,
            found: ref,
            columns: [...names],
        });
    }
    return index;
}

/**
 * Decide whether the first row of a file is a header when the caller left it to the importer: it
 * is when any cell is a known column name (source, target, id, label, weight, type...), or when
 * every cell is non-numeric text while the second row has a numeric cell (`u,v` over `1,2`);
 * a first row with a numeric cell is data, and so is one that looks exactly like the rows below it.
 * @param first - the first row
 * @param second - the second row, or null when the file has one row
 * @returns true when the first row is a header
 */
export function looksLikeHeader(first: readonly string[], second: readonly string[] | null): boolean {
    let allText = true;
    for (const cell of first) {
        const text = cell.trim();
        if (HEADER_MARKERS.has(text.toLowerCase())) {
            return true;
        }
        if (text.length === 0 || inferTextDtype(text) !== "string") {
            allText = false;
        }
    }
    if (!allText || second === null) {
        return false;
    }
    return second.some((cell) => cell.trim().length > 0 && inferTextDtype(cell.trim()) !== "string");
}

/**
 * The synthesised header of a headerless file: `column1` ... `columnN`.
 * @param width - the number of cells of the first row
 * @returns the names
 */
export function positionalNames(width: number): string[] {
    const names: string[] = [];
    for (let i = 1; i <= width; i++) {
        names.push(`column${i}`);
    }
    return names;
}

/**
 * Header cells as column names: trimmed, a BOM-free first cell (the reader strips it), and an
 * empty cell named by its position so every column has a name.
 * @param cells - the header row
 * @returns the names
 */
export function headerNames(cells: readonly string[]): string[] {
    return cells.map((cell, i) => {
        const name = cell.trim();
        return name.length === 0 ? `column${i + 1}` : name;
    });
}
