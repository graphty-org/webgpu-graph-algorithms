/**
 * AttributeTable: a set of index-aligned columns with a fixed row count (design section 5.7). The
 * column SET is a mutable side table of the snapshot (set / remove / rename at any time, by the
 * snapshot owner); the row count is fixed for the life of the table (invariant I17) and column
 * contents are frozen unless the column was declared mutable (design section 5.8). "Absent" is
 * always null on this surface; undefined only ever means "unset row" from value(). Iteration yields
 * columns in declaration order. At most one column per role (E_DUPLICATE_ROLE).
 *
 * The exported helper functions (tableWithColumns, verifyUniqueColumns, declareColumn) are for the
 * snapshot, builder and freeze modules; they are not part of the public surface.
 */

import { GraphFormatError } from "../errors.js";
import {
    type AttributeTableContract,
    type Column,
    type ColumnDeclPatch,
    type ColumnDomain,
    type ColumnInput,
    type ColumnOf,
    type ColumnRole,
    type Dtype,
    type F32,
    type I32,
    type SetOptions,
    type TypedArrayData,
    type U32,
} from "../types/index.js";
import { type TableParts } from "../types/internal.js";
import { assertOneOf } from "../util/options.js";
import { claimHolder, holderCount, isShared, noteShared } from "../util/shared-buffers.js";
import { bitmapGet } from "./bitmap.js";
import {
    columnBuffers,
    columnFromTypedArray,
    columnFromValues,
    gpuViewOf,
    isPackageColumn,
    metaToDecl,
    resolveColumnMeta,
    rewrapColumn,
} from "./column.js";

/**
 * Whether a set() payload is a typed array (as opposed to a Column object or a JS array).
 * @param data - the payload
 * @returns true for one of the five typed-array classes
 */
function isTypedArrayData(data: unknown): data is TypedArrayData {
    return (
        data instanceof Uint32Array ||
        data instanceof Int32Array ||
        data instanceof Float32Array ||
        data instanceof Float64Array ||
        data instanceof Uint8Array
    );
}

/**
 * Whether a set() payload is a Column object (duck-typed on the members every column has).
 * @param data - the payload
 * @returns true for a Column
 */
function isColumn(data: unknown): data is Column {
    if (typeof data !== "object" || data === null || Array.isArray(data) || isTypedArrayData(data)) {
        return false;
    }
    const candidate = data as { dtype?: unknown; meta?: unknown; length?: unknown };
    return (
        typeof candidate.dtype === "string" &&
        typeof candidate.meta === "object" &&
        typeof candidate.length === "number"
    );
}

/**
 * The number of column-set changes a table has seen (set / remove / rename), so a cached role or name
 * lookup can be reused while the set is unchanged.
 * @param table - the table
 * @returns the change count
 */
export function columnSetVersion(table: AttributeTable): number {
    return (table as unknown as { mutations: number }).mutations;
}

/**
 * Record a table as a holder of a column and of every buffer the column views (design section 9.1).
 * @param column - the column being attached
 */
function claimColumn(column: Column): void {
    claimHolder(column);
    for (const buffer of columnBuffers(column)) {
        claimHolder(buffer);
    }
}

/**
 * A set of columns with a fixed row count (design sections 5.7 and 12.2): `nodes` (nodeCount rows),
 * `edges` (edgeCount rows), `graph` (1 row) and extension tables. The column set is a mutable side
 * table; the row count is fixed; iteration is in declaration order.
 */
export class AttributeTable implements AttributeTableContract {
    /** The table the columns belong to. */
    readonly domain: ColumnDomain;

    /** Number of rows of every column; fixed for the life of the table. */
    readonly rowCount: number;

    private readonly columns: Map<string, Column>;

    /** Bumped on every change of the column set (set / remove / rename), for callers that cache a lookup. */
    private mutations = 0;

    /**
     * Create a table from wrapped columns (the constructor the freeze pipeline, fromCsr and the wire
     * reader use). Every column must have the table's row count and domain, names must be unique and
     * at most one column may hold each role.
     * @param parts - the domain, the row count and the columns in declaration order
     * @internal
     */
    constructor(parts: TableParts) {
        const { domain, rowCount } = parts;
        if (!(Number.isInteger(rowCount) && rowCount >= 0)) {
            throw new GraphFormatError("E_COLUMN_LENGTH", `invalid row count ${rowCount} for a ${domain} table`, {
                domain,
                found: rowCount,
            });
        }
        this.domain = domain;
        this.rowCount = rowCount;
        this.columns = new Map();
        for (const column of parts.columns) {
            const { name } = column.meta;
            if (this.columns.has(name)) {
                throw new GraphFormatError("E_COLUMN_EXISTS", `column "${name}" is declared twice`, { column: name });
            }
            this.checkColumn(name, column);
            this.checkRole(name, column.meta.role, false);
            this.columns.set(name, column);
            claimColumn(column);
        }
    }

    /**
     * Column names in declaration order.
     * @returns a fresh array of the names
     */
    names(): readonly string[] {
        return [...this.columns.keys()];
    }

    /**
     * Whether a column exists.
     * @param name - the column name
     * @returns true when present
     */
    has(name: string): boolean {
        return this.columns.has(name);
    }

    /**
     * Total lookup by name.
     * @param name - the column name
     * @returns the column, or null when absent
     */
    get(name: string): Column | null {
        return this.columns.get(name) ?? null;
    }

    /**
     * Checked lookup by name.
     * @param name - the column name
     * @returns the column; E_UNKNOWN_COLUMN when absent
     */
    require(name: string): Column {
        const column = this.columns.get(name);
        if (column === undefined) {
            throw new GraphFormatError("E_UNKNOWN_COLUMN", `no ${this.domain} column named "${name}"`, {
                column: name,
                domain: this.domain,
            });
        }
        return column;
    }

    /**
     * Total lookup by name and dtype.
     * @param name - the column name
     * @param dtype - the expected dtype
     * @returns the column, or null when absent or of another dtype
     */
    typed<D extends Dtype>(name: string, dtype: D): ColumnOf<D> | null {
        const column = this.columns.get(name);
        if (column === undefined || column.dtype !== dtype) {
            return null;
        }
        return column as ColumnOf<D>;
    }

    /**
     * Checked lookup by name and dtype.
     * @param name - the column name
     * @param dtype - the expected dtype
     * @returns the column; E_UNKNOWN_COLUMN when absent, E_COLUMN_TYPE when of another dtype
     */
    requireTyped<D extends Dtype>(name: string, dtype: D): ColumnOf<D> {
        const column = this.require(name);
        if (column.dtype !== dtype) {
            throw new GraphFormatError("E_COLUMN_TYPE", `column "${name}" is ${column.dtype}, expected ${dtype}`, {
                column: name,
                expected: dtype,
                found: column.dtype,
            });
        }
        return column as ColumnOf<D>;
    }

    /**
     * The column holding a role (at most one per table).
     * @param role - the role
     * @returns the column, or null when no column has the role
     */
    byRole(role: ColumnRole): Column | null {
        for (const column of this.columns.values()) {
            if (column.meta.role === role) {
                return column;
            }
        }
        return null;
    }

    /**
     * Typed read of one cell honouring the column's declared default.
     * @param name - the column name; E_UNKNOWN_COLUMN when absent
     * @param row - the row index; E_INDEX_RANGE when out of range
     * @returns the value, the default for an unset row, or undefined for an unset row without a default
     */
    value(name: string, row: number): unknown {
        return this.require(name).value(row);
    }

    /**
     * Whether one cell holds a value.
     * @param name - the column name; E_UNKNOWN_COLUMN when absent
     * @param row - the row index (false when out of range)
     * @returns true when set
     */
    isSet(name: string, row: number): boolean {
        return this.require(name).isSet(row);
    }

    /**
     * Attach a column (design section 5.7). A typed array is adopted by reference after its length is
     * checked (E_COLUMN_LENGTH); a u8 array from which no padded u32 view is constructible is copied
     * unless opts.adopt is "strict" (E_COLUMN_ALIGNMENT); a view over a SharedArrayBuffer or a
     * resizable buffer is E_UNSUPPORTED (decision D-SAB, invariant I10); a JS array builds a column of
     * the declared dtype, or of the dtype inferred from the values (design section 5.1) when the
     * patch names none; a Column object is moved in when its row count matches. Replaces a column of
     * the same name. E_DUPLICATE_ROLE when the declared role is held by another column and
     * opts.replaceRole is not set; with replaceRole the previous holder is removed.
     * @param name - the column name
     * @param data - a Column, a typed array, or a JS array of values
     * @param decl - declaration fields to apply (dtype is inferred from a typed array when omitted)
     * @param opts - role replacement and u8 adoption options
     * @returns the attached column (`column.data !== data` only when a u8 array was copied)
     */
    set(
        name: string,
        data: Column | TypedArrayData | readonly unknown[],
        decl?: ColumnDeclPatch,
        opts?: SetOptions,
    ): Column {
        const patch = decl ?? {};
        let column: Column;
        if (isTypedArrayData(data)) {
            const adopt = assertOneOf("adopt", opts?.adopt, ["copy", "strict"] as const) ?? "copy";
            column = columnFromTypedArray(this.domain, this.rowCount, name, data, patch, adopt);
        } else if (Array.isArray(data)) {
            column = columnFromValues(this.domain, this.rowCount, name, data, patch);
        } else if (isColumn(data)) {
            if (!isPackageColumn(data)) {
                throw new GraphFormatError("E_COLUMN_TYPE", `column "${name}": not a Column of this package`, {
                    column: name,
                    reason: "foreign column",
                });
            }
            column = this.adoptColumn(name, data, decl);
            if (column !== data && holderCount(data) > 0) {
                // a re-wrapped column shares every buffer with `data`, which another table still holds
                noteShared(column);
            }
        } else {
            throw new GraphFormatError("E_COLUMN_TYPE", `column "${name}": unsupported data (${typeof data})`, {
                column: name,
                found: typeof data,
            });
        }
        this.checkRole(name, column.meta.role, opts?.replaceRole ?? false);
        const previous = this.columns.get(name);
        if (previous !== column) {
            // a Column object moved in from another table gains a second holder, and so does every
            // buffer it views (a zero-copy slice of another table's column, design section 9.1);
            // re-wrapping this table's own column under a patch adds no holder of its buffers
            claimHolder(column);
            if (previous !== data) {
                for (const buffer of columnBuffers(column)) {
                    claimHolder(buffer);
                }
            }
        }
        // Map.set on an existing key keeps its position, so a replacement keeps the declaration order
        this.columns.set(name, column);
        this.mutations++;
        return column;
    }

    /**
     * Remove a column.
     * @param name - the column name
     * @returns true when a column was removed
     */
    remove(name: string): boolean {
        const removed = this.columns.delete(name);
        if (removed) {
            this.mutations++;
        }
        return removed;
    }

    /**
     * Rename a column in place (declaration order kept). The Column object is re-wrapped under the new
     * name; its storage is shared with the previous object.
     * @param from - the current name; E_UNKNOWN_COLUMN when absent
     * @param to - the new name; E_COLUMN_EXISTS when another column has it
     */
    rename(from: string, to: string): void {
        const column = this.require(from);
        if (from === to) {
            return;
        }
        if (this.columns.has(to)) {
            throw new GraphFormatError("E_COLUMN_EXISTS", `cannot rename "${from}": a column named "${to}" exists`, {
                column: to,
            });
        }
        const renamed = rewrapColumn(column, resolveColumnMeta(to, this.domain, metaToDecl(column.meta)));
        // the same table keeps holding the same buffers: only the new Column object is claimed
        claimHolder(renamed);
        if (isShared(column)) {
            noteShared(renamed);
        }
        const entries = [...this.columns.entries()];
        this.columns.clear();
        for (const [name, existing] of entries) {
            if (name === from) {
                this.columns.set(to, renamed);
            } else {
                this.columns.set(name, existing);
            }
        }
        this.mutations++;
    }

    /**
     * The array a GPU binds for a column (design section 10.4): its own data for the direct dtypes,
     * the padded u32 view for u8, the packed words for bool, the codes for dict, and a cached f32 copy
     * for f64.
     * @param name - the column name; E_UNKNOWN_COLUMN when absent
     * @returns the bindable array; E_GPU_INELIGIBLE for string / list / json
     */
    gpuView(name: string): U32 | I32 | F32 {
        return gpuViewOf(this.require(name));
    }

    /**
     * A new table with the same columns: the Column objects are shared, the set is independent.
     * @returns the cloned table
     */
    clone(): AttributeTable {
        return new AttributeTable({
            domain: this.domain,
            rowCount: this.rowCount,
            columns: [...this.columns.values()],
        });
    }

    /**
     * Iterate the columns in declaration order.
     * @returns an iterator over the columns
     */
    [Symbol.iterator](): IterableIterator<Column> {
        return this.columns.values();
    }

    /**
     * Move a Column object in: checked for row count and re-wrapped when its name, domain or
     * declaration differ from the target slot.
     * @param name - the target name
     * @param column - the column
     * @param decl - an optional declaration patch
     * @returns the column to store (the same object when nothing changes)
     */
    private adoptColumn(name: string, column: Column, decl: ColumnDeclPatch | undefined): Column {
        this.checkColumn(name, column);
        const { meta } = column;
        const unchanged = decl === undefined && meta.name === name && meta.domain === this.domain;
        if (unchanged) {
            return column;
        }
        const merged: ColumnDeclPatch = { ...metaToDecl(meta), ...decl };
        if (decl?.fill === undefined) {
            merged.fill = meta.fill;
        }
        return rewrapColumn(column, resolveColumnMeta(name, this.domain, merged));
    }

    private checkColumn(name: string, column: Column): void {
        if (column.length !== this.rowCount) {
            throw new GraphFormatError(
                "E_COLUMN_LENGTH",
                `column "${name}" has ${column.length} rows; the ${this.domain} table has ${this.rowCount}`,
                { column: name, expected: this.rowCount, found: column.length },
            );
        }
    }

    /**
     * Enforce "at most one column per role": E_DUPLICATE_ROLE when another column holds the role,
     * unless replaceRole, in which case that column is removed.
     * @param name - the column being attached (a column may keep its own role)
     * @param role - the role being attached, or null
     * @param replaceRole - whether to evict the previous holder
     */
    private checkRole(name: string, role: ColumnRole | null, replaceRole: boolean): void {
        if (role === null) {
            return;
        }
        for (const [otherName, other] of this.columns) {
            if (otherName !== name && other.meta.role === role) {
                if (!replaceRole) {
                    throw new GraphFormatError(
                        "E_DUPLICATE_ROLE",
                        `role "${role}" is already held by column "${otherName}" in the ${this.domain} table`,
                        { role, column: name, holder: otherName },
                    );
                }
                this.columns.delete(otherName);
                return;
            }
        }
    }
}

/**
 * Create an empty table.
 * @param domain - the table's domain
 * @param rowCount - the fixed row count
 * @returns a table with no columns
 */
export function createTable(domain: ColumnDomain, rowCount: number): AttributeTable {
    return new AttributeTable({ domain, rowCount, columns: [] });
}

/**
 * A cloned table plus the given columns (the column half of snapshot.withColumns(), design section
 * 7.3): the Column objects of the source are shared, the set is new, and each entry of `columns` is
 * attached through set() (a typed array, or a ColumnInput carrying data and a declaration patch).
 * @param table - the source table
 * @param columns - the columns to add or replace, keyed by name
 * @returns the new table
 */
export function tableWithColumns(
    table: AttributeTable,
    columns: Readonly<Record<string, TypedArrayData | ColumnInput>>,
): AttributeTable {
    const out = table.clone();
    for (const name of Object.keys(columns)) {
        const input = columns[name];
        if (isTypedArrayData(input)) {
            out.set(name, input);
        } else {
            out.set(name, input.data, input.decl);
        }
    }
    return out;
}

/**
 * The comparable key of one set cell for uniqueness checks: the number, string, boolean or
 * dictionary value; json values are compared by their JSON text.
 * @param column - the column
 * @param row - a set row
 * @returns the key
 */
function uniqueKey(column: Column, row: number): unknown {
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            if (column.meta.components === 1) {
                return column.data[row];
            }
            return Array.from(
                column.data.subarray(row * column.meta.components, (row + 1) * column.meta.components),
            ).join(",");
        case "bool":
            return bitmapGet(column.data, row);
        case "dict":
            return column.codes[row];
        case "string":
            return column.valueAt(row);
        case "list":
            return JSON.stringify(column.sliceOf(row));
        case "json":
            return JSON.stringify(column.values[row]);
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Enforce a unique column over its set rows (design section 5.5; the freeze step 10 hook):
 * E_DUPLICATE_EDGE_ID for an edge column and E_DUPLICATE_ID for every other domain, with the
 * duplicated value and both rows in details.
 * @param column - a column declared unique
 */
export function verifyUniqueColumn(column: Column): void {
    if (!column.meta.unique) {
        return;
    }
    const seen = new Map<unknown, number>();
    for (let row = 0; row < column.length; row++) {
        if (!column.isSet(row)) {
            continue;
        }
        const key = uniqueKey(column, row);
        const first = seen.get(key);
        if (first !== undefined) {
            const code = column.meta.domain === "edge" ? "E_DUPLICATE_EDGE_ID" : "E_DUPLICATE_ID";
            throw new GraphFormatError(
                code,
                `column "${column.meta.name}" is unique but rows ${first} and ${row} hold the same value`,
                {
                    column: column.meta.name,
                    domain: column.meta.domain,
                    rows: [first, row],
                    value: column.value(row),
                },
            );
        }
        seen.set(key, row);
    }
}

/**
 * Enforce every unique column of a table (freeze step 10).
 * @param table - the table to check
 */
export function verifyUniqueColumns(table: AttributeTable): void {
    for (const column of table) {
        verifyUniqueColumn(column);
    }
}
