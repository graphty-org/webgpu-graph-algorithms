/**
 * Edge ids are optional data, not structure (design section 4.6): an edge column with role "id"
 * (dtype string, dict, u32 or f64 in practice; any scalar numeric dtype is read the same way) holds
 * them, importers declare it `unique: true`, and `snapshot.edgeIndexOf(id)` resolves an id through
 * a Map built lazily on the first call. This module owns that lazy index and the uniqueness check
 * freeze runs over the SET rows of a unique id column (E_DUPLICATE_EDGE_ID with both edge indices).
 * Equality is SameValueZero (a JS Map key), the same rule as node ids.
 */

import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { type Column, type EdgeId } from "../types/index.js";

/**
 * The E_COLUMN_TYPE error for a column whose dtype cannot hold edge ids.
 * @param column - the column
 * @returns the error
 */
function unsupportedDtype(column: Column): GraphFormatError {
    return new GraphFormatError(
        "E_COLUMN_TYPE",
        `column ${JSON.stringify(column.meta.name)} of dtype ${column.dtype} cannot hold edge ids`,
        { column: column.meta.name, dtype: column.dtype, reason: "edge id dtype" },
    );
}

/**
 * The id stored in one SET row of an edge id column: the decoded string of a string column, the
 * dictionary value of a dict column, or the (first component of the) number of a numeric column.
 * The caller has checked `column.isSet(row)`.
 * @param column - the id column
 * @param row - a set row
 * @returns the id
 */
export function edgeIdOfRow(column: Column, row: number): EdgeId {
    switch (column.dtype) {
        case "string":
            return column.valueAt(row);
        case "dict":
            return column.dictionary[column.codes[row]];
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            return column.data[row * column.meta.components];
        case "bool":
        case "list":
        case "json":
            throw unsupportedDtype(column);
        default:
            throw unsupportedDtype(column);
    }
}

/**
 * Whether a column's dtype can hold edge ids (string, dict or a scalar numeric dtype).
 * @param column - the column
 * @returns true when edgeIdOfRow can read it
 */
export function canHoldEdgeIds(column: Column): boolean {
    switch (column.dtype) {
        case "string":
        case "dict":
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            return true;
        case "bool":
        case "list":
        case "json":
            return false;
        default:
            return false;
    }
}

/**
 * Build the id -> edge index map of an id column over its SET rows in one pass. When an id repeats
 * the lowest edge index wins (a unique column never repeats; `findDuplicateEdgeId` reports it).
 * @param column - the edge id column
 * @returns a fresh Map from id to logical edge index
 */
export function buildEdgeIdMap(column: Column): Map<EdgeId, number> {
    if (!canHoldEdgeIds(column)) {
        throw unsupportedDtype(column);
    }
    const map = new Map<EdgeId, number>();
    const rows = column.length;
    for (let e = 0; e < rows; e++) {
        if (!column.isSet(e)) {
            continue;
        }
        const id = edgeIdOfRow(column, e);
        if (!map.has(id)) {
            map.set(id, e);
        }
    }
    return map;
}

/**
 * The lazily built index behind `snapshot.edgeIndexOf(id)` (design section 4.6): a Map from id to
 * logical edge index over the set rows of the role "id" edge column, built on the first lookup and
 * rebuilt when a mutable column's `version` has changed since.
 */
export class EdgeIdIndex {
    /** The edge id column. */
    private readonly column: Column;

    /** The map, or null before the first lookup. */
    private map: Map<EdgeId, number> | null;

    /** column.version when the map was built. */
    private builtVersion: number;

    /**
     * Wrap an edge id column; nothing is built until the first lookup.
     * @param column - the edge column with role "id"; E_COLUMN_TYPE when its dtype cannot hold ids
     */
    constructor(column: Column) {
        if (!canHoldEdgeIds(column)) {
            throw unsupportedDtype(column);
        }
        this.column = column;
        this.map = null;
        this.builtVersion = -1;
    }

    /**
     * Whether the map has been built (and is current).
     * @returns true when the next lookup will not build
     */
    get built(): boolean {
        return this.map !== null && this.builtVersion === this.column.version;
    }

    /**
     * Number of distinct ids over the set rows; builds the map.
     * @returns the id count
     */
    get size(): number {
        return this.ensure().size;
    }

    /**
     * The logical edge index of an id.
     * @param id - the edge id
     * @returns the edge index, or INVALID_INDEX on a miss
     */
    indexOf(id: EdgeId): number {
        const index = this.ensure().get(id);
        return index === undefined ? INVALID_INDEX : index;
    }

    /**
     * Whether an id is present.
     * @param id - the edge id
     * @returns true when some set row holds it
     */
    has(id: EdgeId): boolean {
        return this.ensure().has(id);
    }

    /**
     * The map, built on first use and rebuilt after the column's version changed.
     * @returns the id -> edge index map
     */
    private ensure(): Map<EdgeId, number> {
        const { version } = this.column;
        if (this.map === null || this.builtVersion !== version) {
            this.map = buildEdgeIdMap(this.column);
            this.builtVersion = version;
        }
        return this.map;
    }
}
