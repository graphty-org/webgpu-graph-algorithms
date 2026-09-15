/**
 * Remap and gather helpers for raw typed arrays, columns and whole tables (design section 5.11):
 * the propagation of attribute data through compaction (freeze step 10), derived graphs (section
 * 7.3) and consumer re-attachment after a builder mutation. One name per shape:
 *
 * - `remapArray(data, remap, newLength, fill, components)` and `remapColumn(column, remap,
 *   newLength)` take an OLD -> NEW map (`INVALID_INDEX` drops the row);
 * - `gatherArray(data, indexMap, components)` and `gatherColumn(column, indexMap)` take a NEW ->
 *   OLD map (`out[i] = source[indexMap[i]]`);
 * - `scatterArray(out, values, indexMap, components)` writes results computed on a derived graph
 *   back into the parent's index space (`out[indexMap[i]] = values[i]`).
 *
 * Index-valued columns (`meta.refersTo`, a u32 column or a list of u32) have their VALUES rewritten
 * through the remap of the space they reference: an in-range value maps to its new index, a
 * dangling reference becomes `INVALID_INDEX` with the row unset (invariant I12; for a list, the
 * dangling items are dropped and a row whose every item dangled becomes unset and empty).
 */

import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { type Column, type ColumnMeta, type F32, type TypedArrayData, type U32 } from "../types/index.js";
import { type MutableColumnParts } from "../types/internal.js";
import { bitmapClear, bitmapGet, bitmapSet, bitmapWordCount, makeBitmap } from "./bitmap.js";
import { allocU8, columnOfValues, createColumn, partsOf } from "./column.js";
import { AttributeTable } from "./table.js";

// ============================================================ raw arrays

/**
 * A zeroed array of the same class as `like` (a padded store for u8).
 * @param like - the array whose class to match
 * @param length - the element count
 * @returns the new array
 */
function allocLike<T extends TypedArrayData>(like: T, length: number): T {
    if (like instanceof Uint8Array) {
        return allocU8(length) as T;
    }
    if (like instanceof Uint32Array) {
        return new Uint32Array(length) as T;
    }
    if (like instanceof Int32Array) {
        return new Int32Array(length) as T;
    }
    if (like instanceof Float32Array) {
        return new Float32Array(length) as T;
    }
    return new Float64Array(length) as T;
}

function checkComponents(components: number, data: ArrayLike<unknown>): void {
    if (!(Number.isInteger(components) && components >= 1)) {
        throw new GraphFormatError("E_COLUMN_TYPE", `components must be a positive integer, found ${components}`, {
            field: "components",
            found: components,
        });
    }
    if (data.length % components !== 0) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `array length ${data.length} is not a multiple of ${components}`,
            {
                expected: components,
                found: data.length,
            },
        );
    }
}

/**
 * Move rows of an index-aligned array from an old index space to a new one through an OLD -> NEW
 * remap (`FreezeReport.nodeRemap` / `edgeRemap`, `DerivedGraph.nodeRemap` / `edgeRemap`): row `i`
 * lands at `remap[i]`, rows mapped to `INVALID_INDEX` are dropped, and rows of the new space no old
 * row maps to hold `fill`.
 * @param data - the old array, `oldRows * components` long
 * @param remap - old index -> new index or INVALID_INDEX; rows beyond `remap.length` are dropped
 * @param newLength - the row count of the new space
 * @param fill - the value of rows no old row maps to
 * @param components - values per row (default 1)
 * @returns a new array of `newLength * components` values
 */
export function remapArray<T extends TypedArrayData>(
    data: T,
    remap: U32,
    newLength: number,
    fill: number,
    components = 1,
): T {
    checkComponents(components, data);
    const out = allocLike(data, newLength * components);
    if (fill !== 0) {
        out.fill(fill);
    }
    const rows = Math.min(remap.length, data.length / components);
    for (let i = 0; i < rows; i++) {
        const target = remap[i];
        if (target === INVALID_INDEX) {
            continue;
        }
        if (target >= newLength) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `remap[${i}] = ${target} is outside the new space of ${newLength} rows`,
                {
                    index: i,
                    found: target,
                    length: newLength,
                },
            );
        }
        if (components === 1) {
            out[target] = data[i];
        } else {
            for (let k = 0; k < components; k++) {
                out[target * components + k] = data[i * components + k];
            }
        }
    }
    return out;
}

/**
 * Gather rows through a NEW -> OLD index map: `out[i] = data[indexMap[i]]` (`DerivedGraph.nodeOrigin`
 * / `edgeOrigin`).
 * @param data - the source array, `oldRows * components` long
 * @param indexMap - new index -> old index, every entry in range
 * @param components - values per row (default 1)
 * @returns a new array of `indexMap.length * components` values
 */
export function gatherArray<T extends TypedArrayData>(data: T, indexMap: U32, components = 1): T {
    checkComponents(components, data);
    const out = allocLike(data, indexMap.length * components);
    const rows = data.length / components;
    for (let i = 0; i < indexMap.length; i++) {
        const source = indexMap[i];
        if (source >= rows) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `indexMap[${i}] = ${source} is outside the source of ${rows} rows`,
                {
                    index: i,
                    found: source,
                    length: rows,
                },
            );
        }
        if (components === 1) {
            out[i] = data[source];
        } else {
            for (let k = 0; k < components; k++) {
                out[i * components + k] = data[source * components + k];
            }
        }
    }
    return out;
}

/**
 * Scatter values computed on a derived graph back into the parent's index space: `out[indexMap[i]]
 * = values[i]` (through `nodeOrigin` / `edgeOrigin`). Rows of `out` no value maps to are untouched.
 * @param out - the destination in the parent's index space
 * @param values - the values in the derived index space, `indexMap.length * components` long
 * @param indexMap - derived index -> parent index, every entry in range
 * @param components - values per row (default 1)
 * @returns `out`
 */
export function scatterArray<T extends TypedArrayData>(out: T, values: T, indexMap: U32, components = 1): T {
    checkComponents(components, values);
    if (values.length !== indexMap.length * components) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `values has ${values.length} entries, expected ${indexMap.length * components}`,
            {
                expected: indexMap.length * components,
                found: values.length,
            },
        );
    }
    const rows = out.length / components;
    for (let i = 0; i < indexMap.length; i++) {
        const target = indexMap[i];
        if (target >= rows) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `indexMap[${i}] = ${target} is outside the destination of ${rows} rows`,
                {
                    index: i,
                    found: target,
                    length: rows,
                },
            );
        }
        if (components === 1) {
            out[target] = values[i];
        } else {
            for (let k = 0; k < components; k++) {
                out[target * components + k] = values[i * components + k];
            }
        }
    }
    return out;
}

// ============================================================ columns

/**
 * Gather the validity bits of a column through a NEW -> OLD index map; entries equal to
 * INVALID_INDEX are unset rows of the result.
 * @param column - the source column
 * @param indexMap - new index -> old index or INVALID_INDEX
 * @returns the validity bitmap of the result (null when every row is set) and its null count
 */
function gatherValidity(column: Column, indexMap: U32): { validity: U32 | null; nullCount: number } {
    const source = column.validity;
    let validity: U32 | null = null;
    let nullCount = 0;
    for (let i = 0; i < indexMap.length; i++) {
        const from = indexMap[i];
        const set = from !== INVALID_INDEX && (source === null || bitmapGet(source, from));
        if (!set) {
            validity ??= makeBitmap(indexMap.length, true);
            bitmapClear(validity, i);
            nullCount++;
        }
    }
    return { validity, nullCount };
}

/**
 * The metadata of a gathered column: the source metadata, made nullable when the gather produced
 * unset rows that the source could not have had.
 * @param meta - the source metadata
 * @param nullCount - the result's null count
 * @returns the metadata to use
 */
function gatheredMeta(meta: ColumnMeta, nullCount: number): ColumnMeta {
    if (meta.nullable || nullCount === 0) {
        return meta;
    }
    return Object.freeze({ ...meta, nullable: true });
}

/**
 * Gather the rows of a column through a NEW -> OLD index map: `out[i] = column[indexMap[i]]`
 * (design section 5.11, `DerivedGraph.nodeOrigin` / `edgeOrigin`). An INVALID_INDEX entry yields an
 * unset row holding the fill (the column becomes nullable if it was not). Values of a refersTo
 * column are NOT rewritten here; compose with remapReferences() when the referenced space changed.
 * @param column - the source column
 * @param indexMap - new index -> old index (or INVALID_INDEX for a row with no source)
 * @returns a new column of `indexMap.length` rows
 */
export function gatherColumn(column: Column, indexMap: U32): Column {
    const { length } = indexMap;
    for (let i = 0; i < length; i++) {
        const from = indexMap[i];
        if (from !== INVALID_INDEX && from >= column.length) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `indexMap[${i}] = ${from} is outside the column of ${column.length} rows`,
                {
                    index: i,
                    found: from,
                    length: column.length,
                    column: column.meta.name,
                },
            );
        }
    }
    const { validity, nullCount } = gatherValidity(column, indexMap);
    const meta = gatheredMeta(column.meta, nullCount);
    const parts: MutableColumnParts = {
        ...partsOf(column),
        meta,
        length,
        validity,
        nullCount,
    };
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            const { components, fill } = meta;
            const out = allocLike(column.data, length * components);
            if (typeof fill === "number" && fill !== 0) {
                out.fill(fill);
            }
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                if (from === INVALID_INDEX) {
                    continue;
                }
                for (let k = 0; k < components; k++) {
                    out[i * components + k] = column.data[from * components + k];
                }
            }
            parts.data = out;
            break;
        }
        case "bool": {
            const out = new Uint32Array(bitmapWordCount(length)).fill(meta.fill === true ? 0xffffffff : 0);
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                if (from === INVALID_INDEX) {
                    continue;
                }
                if (bitmapGet(column.data, from)) {
                    bitmapSet(out, i);
                } else {
                    bitmapClear(out, i);
                }
            }
            parts.data = out;
            break;
        }
        case "dict": {
            const code = typeof meta.fill === "string" ? column.codeOf(meta.fill) : INVALID_INDEX;
            const fillCode = code === INVALID_INDEX ? 0 : code;
            const out = new Uint32Array(length);
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                out[i] = from === INVALID_INDEX ? fillCode : column.codes[from];
            }
            parts.data = out;
            break;
        }
        case "string": {
            const fillText = typeof meta.fill === "string" ? meta.fill : "";
            const strings = new Array<string>(length);
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                strings[i] = from === INVALID_INDEX ? fillText : column.valueAt(from);
            }
            parts.offsets = null;
            parts.utf8 = null;
            parts.strings = strings;
            break;
        }
        case "list": {
            const { offsets, child } = column;
            const newOffsets = new Uint32Array(length + 1);
            let total = 0;
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                if (from !== INVALID_INDEX) {
                    total += offsets[from + 1] - offsets[from];
                }
                newOffsets[i + 1] = total;
            }
            const itemMap = new Uint32Array(total);
            let at = 0;
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                if (from === INVALID_INDEX) {
                    continue;
                }
                for (let j = offsets[from]; j < offsets[from + 1]; j++) {
                    itemMap[at++] = j;
                }
            }
            parts.offsets = newOffsets;
            parts.child = gatherColumn(child, itemMap);
            break;
        }
        case "json": {
            const out = new Array<unknown>(length);
            for (let i = 0; i < length; i++) {
                const from = indexMap[i];
                out[i] = from === INVALID_INDEX ? undefined : column.values[from];
            }
            parts.values = out;
            break;
        }
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
    return createColumn(parts);
}

/**
 * The NEW -> OLD index map equivalent to an OLD -> NEW remap: entries of the new space no old row
 * maps to are INVALID_INDEX.
 * @param remap - old index -> new index or INVALID_INDEX
 * @param oldLength - the number of old rows to consider (entries beyond are ignored)
 * @param newLength - the row count of the new space
 * @returns the inverse map
 */
function invertRemap(remap: U32, oldLength: number, newLength: number): U32 {
    const indexMap = new Uint32Array(newLength).fill(INVALID_INDEX);
    const rows = Math.min(remap.length, oldLength);
    for (let i = 0; i < rows; i++) {
        const target = remap[i];
        if (target === INVALID_INDEX) {
            continue;
        }
        if (target >= newLength) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `remap[${i}] = ${target} is outside the new space of ${newLength} rows`,
                {
                    index: i,
                    found: target,
                    length: newLength,
                },
            );
        }
        indexMap[target] = i;
    }
    return indexMap;
}

/**
 * Rewrite the values of an index-valued column through the remap of the space it references
 * (design section 5.11): an in-range value maps to its new index; a dangling reference (mapped to
 * INVALID_INDEX or out of range) becomes INVALID_INDEX with the row unset. For a list of u32 the
 * dangling items are dropped and a row whose every item dangled becomes unset and empty. A column
 * without refersTo is returned as it is.
 * @param column - the column
 * @param valueRemap - old index -> new index or INVALID_INDEX, over the referenced space
 * @returns a new column when anything was rewritten; the same object otherwise
 */
export function remapReferences(column: Column, valueRemap: U32): Column {
    if (column.meta.refersTo === null) {
        return column;
    }
    const mapValue = (value: number): number => {
        if (value === INVALID_INDEX || value >= valueRemap.length) {
            return INVALID_INDEX;
        }
        return valueRemap[value];
    };
    if (column.dtype === "u32") {
        const { length } = column;
        const data = new Uint32Array(length);
        let validity = column.validity === null ? null : column.validity.slice();
        let { nullCount } = column;
        for (let row = 0; row < length; row++) {
            const mapped = mapValue(column.data[row]);
            data[row] = mapped;
            if (mapped === INVALID_INDEX && (validity === null || bitmapGet(validity, row))) {
                validity ??= makeBitmap(length, true);
                bitmapClear(validity, row);
                nullCount++;
            }
        }
        const parts: MutableColumnParts = {
            ...partsOf(column),
            meta: gatheredMeta(column.meta, nullCount),
            data,
            validity,
            nullCount,
        };
        return createColumn(parts);
    }
    if (column.dtype === "list" && column.child.dtype === "u32") {
        const { length, offsets, child } = column;
        const rows = new Array<number[]>(length);
        let validity = column.validity === null ? null : column.validity.slice();
        let { nullCount } = column;
        for (let row = 0; row < length; row++) {
            const items: number[] = [];
            const start = offsets[row];
            const end = offsets[row + 1];
            for (let j = start; j < end; j++) {
                const mapped = mapValue(child.data[j]);
                if (mapped !== INVALID_INDEX) {
                    items.push(mapped);
                }
            }
            rows[row] = items;
            if (end > start && items.length === 0 && (validity === null || bitmapGet(validity, row))) {
                validity ??= makeBitmap(length, true);
                bitmapClear(validity, row);
                nullCount++;
            }
        }
        const meta = gatheredMeta(column.meta, nullCount);
        const rebuilt = columnOfValues(Object.freeze({ ...meta, nullable: true }), rows);
        const parts: MutableColumnParts = {
            ...partsOf(rebuilt),
            meta,
            validity,
            nullCount,
        };
        return createColumn(parts);
    }
    throw new GraphFormatError(
        "E_COLUMN_TYPE",
        `column "${column.meta.name}" has refersTo but is not u32 or a list of u32`,
        {
            column: column.meta.name,
            field: "refersTo",
        },
    );
}

/**
 * Move a column's rows from an old index space to a new one through an OLD -> NEW remap, rewriting
 * its values through `valueRemap` when it is an index-valued column (design section 5.11): rows
 * mapped to INVALID_INDEX are dropped, rows of the new space no old row maps to are unset and hold
 * the fill (the column becomes nullable if it was not), and dangling references become
 * INVALID_INDEX with the row unset.
 * @param column - the source column
 * @param remap - old row -> new row or INVALID_INDEX
 * @param newLength - the row count of the new space
 * @param valueRemap - the remap of the space the column's values reference, or null to leave values alone
 * @returns a new column of `newLength` rows
 */
export function remapColumnWith(column: Column, remap: U32, newLength: number, valueRemap: U32 | null): Column {
    if (remap.length !== column.length) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `remap has ${remap.length} entries for a column of ${column.length} rows`,
            { column: column.meta.name, expected: column.length, found: remap.length },
        );
    }
    const gathered = gatherColumn(column, invertRemap(remap, column.length, newLength));
    return valueRemap === null ? gathered : remapReferences(gathered, valueRemap);
}

/**
 * Move a column's rows through an OLD -> NEW remap and, when the column refers to its OWN index
 * space (a node column with `refersTo: "node"` such as `parent`, an edge column with `refersTo:
 * "edge"` such as `pair`), rewrite its values through the SAME remap (design sections 5.11 and
 * 12.2). A column whose values reference the OTHER space (a node column referring to edges) has its
 * rows moved and its values left untouched, because `remap` says nothing about that space; the
 * internal remapColumnWith() / remapTable() take a second remap for it.
 * @param column - the source column
 * @param remap - old row -> new row or INVALID_INDEX
 * @param newLength - the row count of the new space
 * @returns a new column of `newLength` rows
 */
export function remapColumn(column: Column, remap: U32, newLength: number): Column {
    const { refersTo, domain } = column.meta;
    const ownSpace = refersTo !== null && refersTo === domain;
    return remapColumnWith(column, remap, newLength, ownSpace ? remap : null);
}

/**
 * Change the stride of an interleaved f32 array (design section 5.2, the generic stride helper):
 * every row of `from` values becomes a row of `to` values, extra lanes filled with `fill` and
 * surplus lanes dropped. A 2D layout result expands to the 3-component position column with
 * `withComponents(xy, 2, 3, 0)`; the inverse drops `z`. Returns `data` itself when `from === to`.
 * @param data - the interleaved values, `rows * from` long
 * @param from - values per row of `data`, 1..16
 * @param to - values per row of the result, 1..16
 * @param fill - the value of the lanes `to` adds beyond `from`
 * @returns a new array of `rows * to` values (the input when the stride is unchanged)
 */
export function withComponents(data: F32, from: number, to: number, fill: number): F32 {
    for (const [field, value] of [
        ["from", from],
        ["to", to],
    ] as const) {
        if (!Number.isInteger(value) || value < 1 || value > 16) {
            throw new GraphFormatError("E_COLUMN_TYPE", `${field} must be an integer in 1..16, found ${value}`, {
                field,
                found: value,
            });
        }
    }
    checkComponents(from, data);
    if (from === to) {
        return data;
    }
    const rows = data.length / from;
    const out = new Float32Array(rows * to);
    const keep = Math.min(from, to);
    if (to > from && fill !== 0) {
        out.fill(fill);
    }
    for (let row = 0; row < rows; row++) {
        const src = row * from;
        const dst = row * to;
        for (let k = 0; k < keep; k++) {
            out[dst + k] = data[src + k];
        }
    }
    return out;
}

// ============================================================ tables

/**
 * The remaps of the two index spaces a column's values may reference; null when that space was not
 * renumbered.
 */
interface ReferenceRemaps {
    /** Old node index -> new node index or INVALID_INDEX; null when nodes were not renumbered. */
    readonly node: U32 | null;
    /** Old edge index -> new edge index or INVALID_INDEX; null when edges were not renumbered. */
    readonly edge: U32 | null;
}

/**
 * The value remap that applies to a column, if any.
 * @param meta - the column metadata
 * @param refs - the reference remaps
 * @returns the remap of the referenced space, or null
 */
function valueRemapFor(meta: ColumnMeta, refs: ReferenceRemaps): U32 | null {
    if (meta.refersTo === "node") {
        return refs.node;
    }
    if (meta.refersTo === "edge") {
        return refs.edge;
    }
    return null;
}

/**
 * Remap every column of a table (freeze step 10, the compaction case; extension tables with
 * `rowRemap` null): rows move through `rowRemap` when given, and every refersTo column's values are
 * rewritten through the remap of the space it references. The result is a new table; the source is
 * untouched.
 * @param table - the source table
 * @param rowRemap - old row -> new row or INVALID_INDEX; null when the table's rows are unchanged
 * @param newLength - the row count of the new table (ignored, and equal to table.rowCount, when rowRemap is null)
 * @param refs - the node and edge remaps for index-valued columns
 * @returns the remapped table
 */
export function remapTable(
    table: AttributeTable,
    rowRemap: U32 | null,
    newLength: number,
    refs: ReferenceRemaps,
): AttributeTable {
    const rowCount = rowRemap === null ? table.rowCount : newLength;
    const columns: Column[] = [];
    for (const column of table) {
        const valueRemap = valueRemapFor(column.meta, refs);
        let out = column;
        if (rowRemap !== null) {
            out = remapColumnWith(column, rowRemap, newLength, valueRemap);
        } else if (valueRemap !== null) {
            out = remapReferences(column, valueRemap);
        }
        columns.push(out);
    }
    return new AttributeTable({ domain: table.domain, rowCount, columns });
}

/**
 * Gather every column of a table through a NEW -> OLD index map (derived graphs: `nodeOrigin` /
 * `edgeOrigin`), rewriting refersTo values through the remaps of the spaces they reference.
 * @param table - the source table
 * @param indexMap - new row -> old row
 * @param refs - the node and edge remaps for index-valued columns
 * @returns the gathered table of `indexMap.length` rows
 */
export function gatherTable(table: AttributeTable, indexMap: U32, refs: ReferenceRemaps): AttributeTable {
    const columns: Column[] = [];
    for (const column of table) {
        const gathered = gatherColumn(column, indexMap);
        const valueRemap = valueRemapFor(column.meta, refs);
        columns.push(valueRemap === null ? gathered : remapReferences(gathered, valueRemap));
    }
    return new AttributeTable({ domain: table.domain, rowCount: indexMap.length, columns });
}
