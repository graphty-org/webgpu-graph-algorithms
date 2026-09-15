/**
 * The builder's staging representation and its compaction (design sections 6.2 and 6.3 step 1).
 *
 * Staging is structure-of-arrays typed storage that grows by doubling: the node side (`ids`,
 * `idToIndex`, `nodeAlive`, the incidence-list heads `firstOut` / `firstIn`), the edge side (`src`,
 * `dst`, `weight`, `weightSet`, `edgeAlive`, the incidence-list links `nextOut` / `nextIn`), the live
 * counts, and one `StagingColumn` per declared node / edge / extension column (a growable typed array,
 * packed bitmap or JS array plus an optional validity bitmap and, for dict columns, an interning
 * dictionary). No per-node or per-edge object is ever allocated.
 *
 * Compaction (`compactStaging`) gathers a NEW staging object through node and edge remaps assigned in
 * index order (invariant I14): tombstoned nodes and edges disappear, `refersTo` column values are
 * rewritten (dangling references become INVALID_INDEX with the row unset), fresh `ids` / `idToIndex`
 * objects are built so the previous snapshot keeps the ones it shares (design section 4.2), and the
 * incidence lists are rebuilt. The source staging is never modified, so a freeze that throws after
 * compacting leaves the builder exactly as it was (design section 11.1); the builder commits the new
 * staging only when the freeze succeeds.
 */

import { bitmapClear, bitmapGet, bitmapWordCount, makeBitmap } from "../columns/bitmap.js";
import {
    allocNumeric,
    assertJsonValue,
    columnOfValues,
    createColumn,
    isArrayLikeNumbers,
    metaToDecl,
    representableNumber,
    resolveColumnMeta,
    seedDictionary,
} from "../columns/column.js";
import { assertWellFormedString, DictionaryBuilder } from "../columns/dictionary.js";
import { GrowableBitmap, GrowableTypedArray } from "../columns/growable.js";
import { coerceValue, DtypeInferrer, type InferredDtype, inferValueDtype } from "../columns/infer.js";
import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { detachString } from "../ids/string-store.js";
import {
    type Column,
    type ColumnDeclPatch,
    type ColumnDomain,
    type ColumnMeta,
    type Dtype,
    type NodeId,
    type ScalarDtype,
    type TypedArrayData,
    type U32,
} from "../types/index.js";
import { type MutableColumnParts } from "../types/internal.js";

// ============================================================ small helpers

/** The shared, frozen empty list every unset / unwritten list row points at. */
const EMPTY_LIST: readonly unknown[] = Object.freeze([]);

/**
 * The E_COLUMN_TYPE error for a value a column cannot hold.
 * @param meta - the column
 * @param row - the row
 * @param message - what is wrong
 * @param found - the offending value's type or shape
 * @returns the error
 */
function cellError(meta: ColumnMeta, row: number, message: string, found: unknown): GraphFormatError {
    return new GraphFormatError("E_COLUMN_TYPE", `row ${row} of ${meta.dtype} column "${meta.name}": ${message}`, {
        column: meta.name,
        domain: meta.domain,
        row,
        found,
    });
}

/**
 * The dtype an inferred (auto-declared) column starts with for a value (design section 5.1).
 * @param value - the first set value
 * @returns the inferred dtype; E_COLUMN_TYPE for a value no column can hold
 */
export function inferInitialDtype(value: unknown): InferredDtype {
    const dtype = inferValueDtype(value);
    if (dtype === null) {
        throw new GraphFormatError("E_COLUMN_TYPE", "an unset value cannot start an inferred column", {
            reason: "unset",
        });
    }
    return dtype;
}

/**
 * The position of a dtype in the widening order of design section 5.1, for the union rule of
 * `addGraph()` (design section 6.6): the dtypes outside the inference order map to the narrowest
 * inferred dtype that can hold every value of theirs.
 * @param dtype - any dtype
 * @returns the inferred dtype the column widens to when it must hold values of `dtype`
 */
export function inferredEquivalent(dtype: Dtype): InferredDtype {
    switch (dtype) {
        case "bool":
            return "bool";
        case "i32":
        case "u8":
            return "i32";
        case "u32":
        case "f32":
        case "f64":
            return "f64";
        case "dict":
        case "string":
            return "string";
        case "list":
        case "json":
            return "json";
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Check one list item against the child dtype of a list column at write time, so a bad item fails
 * at the call that supplied it rather than at freeze.
 * @param meta - the list column
 * @param row - the row
 * @param item - the item
 */
function checkListItem(meta: ColumnMeta, row: number, item: unknown): void {
    const itemDtype = meta.itemDtype as ScalarDtype;
    const components = meta.itemComponents ?? 1;
    switch (itemDtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            if (components === 1) {
                if (typeof item !== "number" || !representableNumber(itemDtype, item)) {
                    throw cellError(meta, row, `list item is not a representable ${itemDtype}`, typeof item);
                }
                return;
            }
            if (!Array.isArray(item) || item.length !== components) {
                throw cellError(meta, row, `list item is not ${components} numbers`, typeof item);
            }
            for (const k of item as unknown[]) {
                if (typeof k !== "number" || !representableNumber(itemDtype, k)) {
                    throw cellError(meta, row, `list item component is not a representable ${itemDtype}`, typeof k);
                }
            }
            return;
        }
        case "bool":
            if (typeof item !== "boolean") {
                throw cellError(meta, row, "list item is not a boolean", typeof item);
            }
            return;
        case "dict":
        case "string":
            if (typeof item !== "string") {
                throw cellError(meta, row, "list item is not a string", typeof item);
            }
            assertWellFormedString(item, { column: meta.name, row });
            return;
        case "json":
            assertJsonValue(item, meta.name);
            return;
        default: {
            const name: string = itemDtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

// ============================================================ staging columns

/** The widening a write caused on an inferred column. */
interface WideningStep {
    /** The dtype before. */
    readonly from: Dtype;
    /** The dtype after. */
    readonly to: Dtype;
}

/**
 * One growable column of the builder (design section 6.2).
 *
 * A DECLARED column stores values in the storage of its dtype (a growable typed array for the numeric
 * dtypes and dict codes, a growable bitmap for bool, a JS array for string / list / json) plus an
 * optional validity bitmap and, for dict columns, the interning dictionary; rows are materialised
 * lazily up to `length` and unset rows physically hold the fill (design section 5.3). It never widens
 * and rejects values it cannot hold with E_COLUMN_TYPE at the write.
 *
 * An INFERRED column (auto-declared by a string column name in `setNodeValue` / `addNodeRecord`, or
 * a declared column widened by the union rule of `addGraph`) keeps the values exactly as written in a
 * JS array and widens its dtype monotonically per design section 5.1 as wider values arrive; the
 * values are coerced ONCE, at freeze, to the final dtype (bool -> 1 / 0 in a number column, number or
 * boolean -> its canonical text in a string column), exactly as `fromRecords` would.
 */
export class StagingColumn {
    /** The resolved metadata; replaced when an inferred column widens. */
    meta: ColumnMeta;

    /** Rows materialised so far. */
    length = 0;

    /** Declared numeric data (rows * components) or dict codes. */
    typed: GrowableTypedArray<TypedArrayData> | null = null;

    /** Declared bool data. */
    bits: GrowableBitmap | null = null;

    /** Declared string / list / json rows, or the raw values of an inferred column. */
    values: unknown[] | null = null;

    /** dict only: the interning dictionary. */
    dictionary: DictionaryBuilder | null = null;

    /** The validity bitmap; null for a non-nullable column. */
    validity: GrowableBitmap | null = null;

    private inferrer: DtypeInferrer | null;

    private readonly resizable: boolean | undefined;

    /**
     * Create an empty column.
     * @param meta - the resolved metadata
     * @param inferred - whether the column stages raw values and may widen
     * @param resizable - the staging buffer kind (tests pass false); undefined = engine default
     * @param dictionary - an existing dictionary to share (compaction); a fresh one when omitted
     */
    constructor(meta: ColumnMeta, inferred: boolean, resizable?: boolean, dictionary?: DictionaryBuilder) {
        this.meta = meta;
        this.resizable = resizable;
        if (inferred) {
            this.inferrer = new DtypeInferrer();
            this.inferrer.widenTo(inferredEquivalent(meta.dtype));
        } else {
            this.inferrer = null;
        }
        this.allocate(dictionary);
    }

    /**
     * Whether the column stages raw values and may widen.
     * @returns true for an inferred column
     */
    get inferred(): boolean {
        return this.inferrer !== null;
    }

    /**
     * The inferred dtype of an inferred column, or null.
     * @returns the current inferred dtype
     */
    get inferredDtype(): InferredDtype | null {
        return this.inferrer === null ? null : this.inferrer.dtype;
    }

    /**
     * Make at least `rows` rows exist, unset (when nullable) and holding the fill.
     * @param rows - the row count to reach
     */
    ensureLength(rows: number): void {
        if (rows <= this.length) {
            return;
        }
        const { meta } = this;
        if (this.typed !== null) {
            this.typed.resize(rows * meta.components, this.fillNumber());
        } else if (this.bits !== null) {
            this.bits.resize(rows, meta.fill === true);
        } else if (this.values !== null) {
            const fill = this.fillValue();
            for (let row = this.length; row < rows; row++) {
                this.values.push(fill);
            }
        }
        if (this.validity !== null) {
            this.validity.resize(rows, false);
        }
        this.length = rows;
    }

    /**
     * Whether a row holds a value.
     * @param row - the row
     * @returns true when set (non-nullable columns: when materialised)
     */
    isSet(row: number): boolean {
        if (row >= this.length) {
            return false;
        }
        return this.validity === null || this.validity.get(row);
    }

    /**
     * Set one cell (design sections 5.1 and 6.6). `undefined` unsets the row, and so does `null`
     * except on a declared json column (JSON null imports as unset, design section 5.3). An inferred
     * column widens when the value is wider than its dtype; the widening is returned so the builder
     * can report it.
     * @param row - the row; rows below it are materialised
     * @param value - the value
     * @returns the widening that took place, or null
     */
    write(row: number, value: unknown): WideningStep | null {
        this.ensureLength(row + 1);
        if (value === undefined || (value === null && (this.inferrer !== null || this.meta.dtype !== "json"))) {
            this.unset(row);
            return null;
        }
        let widened: WideningStep | null = null;
        if (this.inferrer !== null) {
            const before = this.meta.dtype;
            if (this.inferrer.observeValue(value)) {
                const next = this.inferrer.dtype as InferredDtype;
                this.widenTo(next);
                widened = { from: before, to: next };
            }
            this.storeRaw(row, value);
        } else {
            this.store(row, value);
        }
        if (this.validity !== null) {
            this.validity.set(row, true);
        }
        return widened;
    }

    /**
     * Validate a value exactly as `write` would, without writing anything: the same E_COLUMN_TYPE
     * errors for a value the column cannot hold (a non-representable number, a lone surrogate, a
     * non-JSON object, an unset value on a non-nullable column). The record methods of the builder
     * validate every attribute before they apply anything (design section 11.1).
     * @param row - the row the value is meant for (error details only)
     * @param value - the value
     */
    checkValue(row: number, value: unknown): void {
        if (value === undefined || (value === null && (this.inferrer !== null || this.meta.dtype !== "json"))) {
            if (this.validity === null) {
                throw this.unsetError(row);
            }
            return;
        }
        if (this.inferrer !== null) {
            inferValueDtype(value);
            this.checkRaw(row, value);
            return;
        }
        this.coerce(row, value);
    }

    /**
     * Unset one cell: the validity bit is cleared and the fill is written back (design section 5.3).
     * @param row - the row; E_COLUMN_TYPE on a non-nullable column
     */
    unset(row: number): void {
        this.ensureLength(row + 1);
        if (this.validity === null) {
            throw this.unsetError(row);
        }
        this.validity.set(row, false);
        this.writeFill(row);
    }

    /**
     * The E_COLUMN_TYPE error for unsetting a row of a non-nullable column.
     * @param row - the row
     * @returns the error
     */
    private unsetError(row: number): GraphFormatError {
        return new GraphFormatError(
            "E_COLUMN_TYPE",
            `row ${row} of non-nullable column "${this.meta.name}" cannot be unset`,
            { column: this.meta.name, row, field: "nullable" },
        );
    }

    /**
     * The JS value of one cell as the frozen column would hold it: a number (or an array of
     * `components` numbers), a boolean, a string, a list or a json value; undefined for an unset or
     * unmaterialised row. An inferred column's raw value is coerced to its current dtype.
     * @param row - the row
     * @returns the value
     */
    read(row: number): unknown {
        if (!this.isSet(row)) {
            return undefined;
        }
        if (this.inferrer !== null) {
            return coerceValue((this.values as unknown[])[row], this.inferrer.dtype as InferredDtype);
        }
        const { meta } = this;
        const { dtype, components } = meta;
        switch (dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
            case "u8": {
                const typed = this.typed as GrowableTypedArray<TypedArrayData>;
                if (components === 1) {
                    return typed.get(row);
                }
                const out = new Array<number>(components);
                for (let k = 0; k < components; k++) {
                    out[k] = typed.get(row * components + k);
                }
                return out;
            }
            case "bool":
                return (this.bits as GrowableBitmap).get(row);
            case "dict": {
                const code = (this.typed as GrowableTypedArray<TypedArrayData>).get(row);
                return (this.dictionary as DictionaryBuilder).values[code];
            }
            case "string":
            case "list":
            case "json":
                return (this.values as unknown[])[row];
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
    }

    /**
     * Widen the column to a wider dtype of the design section 5.1 order. An inferred column only
     * changes its dtype (its raw values are coerced at freeze); a declared column becomes inferred:
     * its current values are read out and kept as raw values from then on.
     * @param next - the dtype to widen to
     */
    widenTo(next: InferredDtype): void {
        const { meta } = this;
        const nextMeta = resolveColumnMeta(meta.name, meta.domain, {
            dtype: next,
            nullable: true,
            role: meta.role ?? undefined,
            extra: meta.extra,
        });
        if (this.inferrer !== null) {
            this.inferrer.widenTo(next);
            this.meta = nextMeta;
            return;
        }
        const rows = this.length;
        const raw = new Array<unknown>(rows);
        const validity = new GrowableBitmap({ capacity: rows, resizable: this.resizable });
        validity.resize(rows, false);
        for (let row = 0; row < rows; row++) {
            if (this.isSet(row)) {
                raw[row] = this.read(row);
                validity.set(row, true);
            }
        }
        this.meta = nextMeta;
        this.inferrer = new DtypeInferrer();
        this.inferrer.widenTo(next);
        this.typed = null;
        this.bits = null;
        this.dictionary = null;
        this.values = raw;
        this.validity = validity;
    }

    /**
     * A new column of the same shape holding the rows `indexMap` selects (`out[i] = this[indexMap[i]]`),
     * or every row in order when `indexMap` is null, with `refersTo` values rewritten through
     * `valueRemap` when given: an in-range value maps to its new index and a dangling reference
     * becomes INVALID_INDEX with the row unset (for a list, dangling items are dropped and a row whose
     * every item dangled becomes unset and empty). The dictionary of a dict column is shared.
     * @param indexMap - new row -> old row, or null to keep every row
     * @param valueRemap - the remap of the space the values reference, or null
     * @returns the gathered column
     */
    gather(indexMap: U32 | null, valueRemap: U32 | null): StagingColumn {
        const rows = indexMap === null ? this.length : indexMap.length;
        const out = new StagingColumn(this.meta, this.inferred, this.resizable, this.dictionary ?? undefined);
        out.ensureLength(rows);
        const { components } = this.meta;
        const rewrite = valueRemap !== null && this.meta.refersTo !== null;
        for (let i = 0; i < rows; i++) {
            const old = indexMap === null ? i : indexMap[i];
            if (!this.isSet(old)) {
                continue;
            }
            if (out.typed !== null && this.typed !== null) {
                if (rewrite) {
                    const mapped = remapIndex(this.typed.get(old), valueRemap);
                    if (mapped === INVALID_INDEX) {
                        // dangling reference: INVALID_INDEX (the fill of a refersTo column) with the row unset
                        out.unsetDangling(i);
                        continue;
                    }
                    out.typed.set(i, mapped);
                } else {
                    for (let k = 0; k < components; k++) {
                        out.typed.set(i * components + k, this.typed.get(old * components + k));
                    }
                }
            } else if (out.bits !== null && this.bits !== null) {
                out.bits.set(i, this.bits.get(old));
            } else if (out.values !== null && this.values !== null) {
                let value = this.values[old];
                if (rewrite) {
                    const items = value as readonly number[];
                    const kept: number[] = [];
                    for (const item of items) {
                        const mapped = remapIndex(item, valueRemap);
                        if (mapped !== INVALID_INDEX) {
                            kept.push(mapped);
                        }
                    }
                    if (kept.length === 0 && items.length > 0) {
                        out.unsetDangling(i);
                        continue;
                    }
                    value = kept;
                }
                out.values[i] = value;
            }
            if (out.validity !== null) {
                out.validity.set(i, true);
            }
        }
        return out;
    }

    /**
     * Leave row `i` of a gathered column unset because its reference dangled (design section 5.11).
     * A non-nullable column gains a validity bitmap and becomes nullable (invariant I12 forbids a set
     * row holding INVALID_INDEX; `remapColumn` flips `nullable` the same way): every earlier row is
     * marked set except a scalar reference row still holding the INVALID_INDEX fill.
     * @param i - the row being gathered; rows below it are already gathered
     */
    private unsetDangling(i: number): void {
        if (this.validity !== null) {
            return;
        }
        const { meta } = this;
        this.meta = resolveColumnMeta(meta.name, meta.domain, { ...metaToDecl(meta), nullable: true });
        const validity = new GrowableBitmap({ capacity: this.length, resizable: this.resizable });
        validity.resize(this.length, false);
        const { typed } = this;
        for (let row = 0; row < i; row++) {
            if (typed === null || typed.get(row) !== INVALID_INDEX) {
                validity.set(row, true);
            }
        }
        this.validity = validity;
    }

    /**
     * Freeze the column into `rows` rows (design section 6.3 step 10): typed data is copied out of
     * staging into exact-length buffers (invariant I18), the validity bitmap is copied (and dropped when
     * every row is set), a dict dictionary is copied so the snapshot's column never changes (I17), and
     * string / list / json rows and the raw values of an inferred column go through the column factory,
     * which coerces them to the final dtype.
     * @param rows - the row count of the table
     * @returns the frozen column
     */
    toColumn(rows: number): Column {
        this.ensureLength(rows);
        const { meta } = this;
        const { dtype, components } = meta;
        if (this.inferrer !== null) {
            const source = this.values as unknown[];
            const entries = new Array<unknown>(rows);
            for (let row = 0; row < rows; row++) {
                entries[row] = this.isSet(row) ? source[row] : undefined;
            }
            return columnOfValues(meta, entries);
        }
        let validity: U32 | null = null;
        if (this.validity !== null && this.validity.count() !== rows) {
            validity = this.validity.trim();
        }
        const parts: MutableColumnParts = {
            meta,
            length: rows,
            data: null,
            validity,
            nullCount: validity === null ? 0 : rows - (this.validity as GrowableBitmap).count(),
            dictionary: null,
            offsets: null,
            utf8: null,
            strings: null,
            child: null,
            values: null,
        };
        switch (dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
            case "u8": {
                const data = allocNumeric(dtype, rows * components);
                data.set((this.typed as GrowableTypedArray<TypedArrayData>).view().subarray(0, rows * components));
                parts.data = data;
                if (meta.refersTo !== null && this.validity === null) {
                    unsetFillReferences(parts, data as U32, rows);
                }
                return createColumn(parts);
            }
            case "bool":
                parts.data = (this.bits as GrowableBitmap).trim();
                return createColumn(parts);
            case "dict": {
                const codes = new Uint32Array(rows);
                codes.set((this.typed as GrowableTypedArray<TypedArrayData>).view().subarray(0, rows));
                parts.data = codes;
                parts.dictionary = [...(this.dictionary as DictionaryBuilder).values];
                return createColumn(parts);
            }
            case "string":
            case "list":
            case "json": {
                const source = this.values as unknown[];
                const entries = new Array<unknown>(rows);
                for (let row = 0; row < rows; row++) {
                    const value = this.isSet(row) ? source[row] : undefined;
                    entries[row] = value === undefined ? this.unsetEntry() : value;
                }
                return columnOfValues(meta, entries);
            }
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
    }

    /**
     * Bytes of typed staging held by the column (JS arrays are not counted).
     * @returns the byte count
     */
    byteLength(): number {
        let bytes = 0;
        if (this.typed !== null) {
            bytes += this.typed.array.byteLength;
        }
        if (this.bits !== null) {
            bytes += bitmapWordCount(this.bits.length) * 4;
        }
        if (this.validity !== null) {
            bytes += bitmapWordCount(this.validity.length) * 4;
        }
        return bytes;
    }

    /**
     * Allocate the storage of the current dtype (raw values for an inferred column).
     * @param dictionary - a dictionary to share, or undefined for a fresh seeded one
     */
    private allocate(dictionary: DictionaryBuilder | undefined): void {
        const { meta } = this;
        const { dtype } = meta;
        const options = { resizable: this.resizable };
        if (this.inferrer !== null) {
            this.values = [];
            this.validity = new GrowableBitmap(options);
            return;
        }
        switch (dtype) {
            case "f32":
                this.typed = new GrowableTypedArray(Float32Array, options);
                break;
            case "f64":
                this.typed = new GrowableTypedArray(Float64Array, options);
                break;
            case "i32":
                this.typed = new GrowableTypedArray(Int32Array, options);
                break;
            case "u32":
                this.typed = new GrowableTypedArray(Uint32Array, options);
                break;
            case "u8":
                this.typed = new GrowableTypedArray(Uint8Array, options);
                break;
            case "bool":
                this.bits = new GrowableBitmap(options);
                break;
            case "dict": {
                this.typed = new GrowableTypedArray(Uint32Array, options);
                this.dictionary = dictionary ?? seedDictionary(meta);
                break;
            }
            case "string":
            case "list":
            case "json":
                this.values = [];
                break;
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
        this.validity = meta.nullable ? new GrowableBitmap(options) : null;
    }

    /**
     * The number written into unset numeric rows (the dict fill code for dict columns).
     * @returns the fill
     */
    private fillNumber(): number {
        const { meta } = this;
        if (meta.dtype === "dict") {
            const dictionary = this.dictionary as DictionaryBuilder;
            const text = typeof meta.fill === "string" ? meta.fill : "";
            if (this.validity === null) {
                // a set row holding the fill needs a member's code: the nominal "" is interned on demand
                return dictionary.intern(text);
            }
            const code = dictionary.codeOf(text);
            return code === INVALID_INDEX ? 0 : code;
        }
        return typeof meta.fill === "number" ? meta.fill : 0;
    }

    /**
     * The JS value stored in unset rows of a JS-array column (undefined for an inferred column).
     * @returns the fill
     */
    private fillValue(): unknown {
        if (this.inferrer !== null) {
            return undefined;
        }
        switch (this.meta.dtype) {
            case "string":
                return typeof this.meta.fill === "string" ? this.meta.fill : "";
            case "list":
                return EMPTY_LIST;
            default:
                return undefined;
        }
    }

    /**
     * The entry handed to the column factory for an unset row of a declared JS-array column:
     * undefined for a nullable column, the fill for a non-nullable one (json: null, the one JSON value
     * that means "nothing").
     * @returns the entry
     */
    private unsetEntry(): unknown {
        if (this.meta.nullable) {
            return undefined;
        }
        return this.meta.dtype === "json" ? null : this.fillValue();
    }

    private writeFill(row: number): void {
        const { components } = this.meta;
        if (this.typed !== null) {
            const fill = this.fillNumber();
            for (let k = 0; k < components; k++) {
                this.typed.set(row * components + k, fill);
            }
        } else if (this.bits !== null) {
            this.bits.set(row, this.meta.fill === true);
        } else if (this.values !== null) {
            this.values[row] = this.fillValue();
        }
    }

    /**
     * Keep a raw value on an inferred column after checking it is storable at all (a json value must
     * be JSON; a string must be well-formed).
     * @param row - the row
     * @param value - the value as written
     */
    private storeRaw(row: number, value: unknown): void {
        this.checkRaw(row, value);
        (this.values as unknown[])[row] = typeof value === "string" ? detachString(value) : value;
    }

    /**
     * The checks of storeRaw: a string must be well-formed, an object must be JSON.
     * @param row - the row
     * @param value - the value as written
     */
    private checkRaw(row: number, value: unknown): void {
        const { meta } = this;
        if (typeof value === "string") {
            assertWellFormedString(value, { column: meta.name, row });
        } else if (typeof value === "object") {
            assertJsonValue(value, `${meta.name}[${row}]`);
        }
    }

    /**
     * Store a value on a declared column: validate and normalise it (E_COLUMN_TYPE when the column
     * cannot hold it), then write the normalised form.
     * @param row - the row
     * @param value - the value
     */
    private store(row: number, value: unknown): void {
        this.put(row, this.coerce(row, value));
    }

    /**
     * Validate a value against a declared column's dtype and return the form `put` writes: a number
     * or an array of `components` numbers, a boolean, a string, a copied list, or a JSON value.
     * @param row - the row (error details only)
     * @param value - the value
     * @returns the normalised value
     */
    private coerce(row: number, value: unknown): unknown {
        const { meta } = this;
        const { dtype, components } = meta;
        switch (dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
            case "u8": {
                const scalar = typeof value === "boolean" ? Number(value) : value;
                if (typeof scalar === "number") {
                    if (!representableNumber(dtype, scalar)) {
                        throw cellError(meta, row, `${scalar} is not representable`, scalar);
                    }
                    return scalar;
                }
                if (components === 1 || !isArrayLikeNumbers(scalar, components)) {
                    throw cellError(meta, row, `not ${components} number(s)`, typeof scalar);
                }
                for (let k = 0; k < components; k++) {
                    if (!representableNumber(dtype, scalar[k])) {
                        throw cellError(meta, row, `${scalar[k]} is not representable`, scalar[k]);
                    }
                }
                return scalar;
            }
            case "bool":
                return coerceValue(value, "bool") === true;
            case "dict":
            case "string": {
                const text = coerceValue(value, "string");
                if (typeof text !== "string") {
                    throw cellError(meta, row, "not a string", typeof value);
                }
                assertWellFormedString(text, { column: meta.name, row });
                return detachString(text);
            }
            case "list": {
                if (!Array.isArray(value)) {
                    throw cellError(meta, row, "not an array", typeof value);
                }
                for (const item of value as unknown[]) {
                    checkListItem(meta, row, item);
                }
                return (value as unknown[]).map((item) => (typeof item === "string" ? detachString(item) : item));
            }
            case "json":
                assertJsonValue(value, `${meta.name}[${row}]`);
                return value;
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
    }

    /**
     * Write a value `coerce` returned.
     * @param row - the row
     * @param value - the normalised value
     */
    private put(row: number, value: unknown): void {
        const { dtype, components } = this.meta;
        switch (dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
            case "u8": {
                const typed = this.typed as GrowableTypedArray<TypedArrayData>;
                if (typeof value === "number") {
                    if (components === 1) {
                        typed.set(row, value);
                    } else {
                        for (let k = 0; k < components; k++) {
                            typed.set(row * components + k, value);
                        }
                    }
                    return;
                }
                const vector = value as ArrayLike<number>;
                for (let k = 0; k < components; k++) {
                    typed.set(row * components + k, vector[k]);
                }
                return;
            }
            case "bool":
                (this.bits as GrowableBitmap).set(row, value === true);
                return;
            case "dict":
                (this.typed as GrowableTypedArray<TypedArrayData>).set(
                    row,
                    (this.dictionary as DictionaryBuilder).intern(value as string),
                );
                return;
            case "string":
            case "list":
            case "json":
                (this.values as unknown[])[row] = value;
                return;
            default: {
                const name: string = dtype;
                throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
            }
        }
    }
}

/**
 * Map an index-valued cell through a remap: in range and alive -> the new index; otherwise INVALID_INDEX.
 * @param value - the old index
 * @param remap - old -> new or INVALID_INDEX
 * @returns the new index or INVALID_INDEX
 */
function remapIndex(value: number, remap: U32): number {
    return value < remap.length ? remap[value] : INVALID_INDEX;
}

/**
 * The frozen form of a non-nullable scalar `refersTo` column whose rows were never all written: the
 * unwritten rows still hold the INVALID_INDEX fill, and a SET row holding INVALID_INDEX violates
 * invariant I12, so the frozen column becomes nullable with exactly those rows unset (the rule of
 * `gather` for a dangling reference and of `remapColumn`; design sections 5.11 and 6.3, the builder
 * establishes I1-I13 by construction). The builder's own declaration is left alone. A column whose
 * every row holds an index is returned as declared.
 * @param parts - the parts being assembled; `meta`, `validity` and `nullCount` are replaced when needed
 * @param data - the frozen u32 data (stride 1: refersTo requires a scalar u32)
 * @param rows - the row count
 */
function unsetFillReferences(parts: MutableColumnParts, data: U32, rows: number): void {
    let validity: U32 | null = null;
    for (let row = 0; row < rows; row++) {
        if (data[row] === INVALID_INDEX) {
            validity ??= makeBitmap(rows, true);
            bitmapClear(validity, row);
            parts.nullCount++;
        }
    }
    if (validity !== null) {
        const { meta } = parts;
        parts.meta = resolveColumnMeta(meta.name, meta.domain, { ...metaToDecl(meta), nullable: true });
        parts.validity = validity;
    }
}

/**
 * Create a staging column from a declaration patch.
 * @param name - the column name
 * @param domain - the table
 * @param decl - the declaration; dtype required
 * @param inferred - whether the column may widen
 * @param resizable - the staging buffer kind
 * @returns the column
 */
export function createStagingColumn(
    name: string,
    domain: ColumnDomain,
    decl: ColumnDeclPatch,
    inferred: boolean,
    resizable?: boolean,
): StagingColumn {
    return new StagingColumn(resolveColumnMeta(name, domain, decl), inferred, resizable);
}

// ============================================================ extension tables

/** One extension table under construction (design section 5.10). */
export interface ExtensionStaging {
    /** The table name. */
    readonly name: string;
    /** Its columns in declaration order. */
    readonly columns: StagingColumn[];
    /** Rows appended so far. */
    rowCount: number;
}

// ============================================================ staging

/** Construction options of a Staging. */
interface StagingOptions {
    /** Staging weight precision (design section 3.7). */
    readonly weightDtype: "f32" | "f64";
    /** Allocate the weight array up front (`weighted: true`). */
    readonly weighted: boolean;
    /** Node capacity hint. */
    readonly expectedNodes: number | null;
    /** Edge capacity hint. */
    readonly expectedEdges: number | null;
    /** The staging buffer kind; undefined = engine default. */
    readonly resizable: boolean | undefined;
}

/**
 * Whether every edge so far agreed on supplying or omitting its weight (design section 3.7): the
 * `weightSet` bitmap is only allocated once they disagree ("mixed").
 */
type WeightMode = "none" | "explicit" | "omitted" | "mixed";

/**
 * The staging of one builder (design section 6.2). Plain growable storage plus the push and link
 * primitives; every rule (id validation, policies, error codes) lives in the builder.
 */
export class Staging {
    /** The staging options. */
    readonly options: StagingOptions;

    /** The ids in index order, or null while every node is anonymous (id === index). */
    ids: NodeId[] | null = null;

    /** id -> index, shared by reference with frozen snapshots; null while every node is anonymous. */
    idToIndex: Map<NodeId, number> | null = null;

    /** Bit i set: node i is live. */
    readonly nodeAlive: GrowableBitmap;

    /** Head of the out-list of every node (an edge index), INVALID_INDEX = empty. */
    readonly firstOut: GrowableTypedArray<U32>;

    /** Head of the in-list of every node. */
    readonly firstIn: GrowableTypedArray<U32>;

    /** Declared source of every edge. */
    readonly src: GrowableTypedArray<U32>;

    /** Declared target of every edge. */
    readonly dst: GrowableTypedArray<U32>;

    /** Per-edge weights (f32 or f64 per weightDtype), or null while unweighted. */
    weight: GrowableTypedArray<TypedArrayData> | null = null;

    /** Bit e set: edge e's weight was supplied explicitly; null while every edge agrees. */
    weightSet: GrowableBitmap | null = null;

    /** Whether edges so far supplied, omitted, or disagreed about their weight. */
    weightMode: WeightMode = "none";

    /** Bit e set: edge e is live. */
    readonly edgeAlive: GrowableBitmap;

    /** Next edge in the out-list of the source of every edge. */
    readonly nextOut: GrowableTypedArray<U32>;

    /** Next edge in the in-list of the target of every edge. */
    readonly nextIn: GrowableTypedArray<U32>;

    /** Live node count. */
    liveNodeCount = 0;

    /** Live edge count. */
    liveEdgeCount = 0;

    /** Live self-loop count. */
    selfLoopCount = 0;

    /** Node columns in handle order. */
    nodeColumns: StagingColumn[] = [];

    /** Edge columns in handle order. */
    edgeColumns: StagingColumn[] = [];

    /** Extension tables in handle order. */
    extensions: ExtensionStaging[] = [];

    /**
     * Create empty staging.
     * @param options - precision, capacity hints and buffer kind
     */
    constructor(options: StagingOptions) {
        this.options = options;
        const { resizable } = options;
        const nodes = options.expectedNodes ?? 0;
        const edges = options.expectedEdges ?? 0;
        this.nodeAlive = new GrowableBitmap({ capacity: nodes, resizable });
        this.firstOut = new GrowableTypedArray<U32>(Uint32Array, { capacity: nodes, resizable });
        this.firstIn = new GrowableTypedArray<U32>(Uint32Array, { capacity: nodes, resizable });
        this.src = new GrowableTypedArray<U32>(Uint32Array, { capacity: edges, resizable });
        this.dst = new GrowableTypedArray<U32>(Uint32Array, { capacity: edges, resizable });
        this.edgeAlive = new GrowableBitmap({ capacity: edges, resizable });
        this.nextOut = new GrowableTypedArray<U32>(Uint32Array, { capacity: edges, resizable });
        this.nextIn = new GrowableTypedArray<U32>(Uint32Array, { capacity: edges, resizable });
        if (options.weighted) {
            this.weight = this.allocateWeights(edges);
        }
    }

    /**
     * The next node index to be assigned.
     * @returns the node bound
     */
    get nodeBound(): number {
        return this.nodeAlive.length;
    }

    /**
     * The next edge index to be assigned.
     * @returns the edge bound
     */
    get edgeBound(): number {
        return this.edgeAlive.length;
    }

    /**
     * Whether tombstoned nodes or edges exist.
     * @returns true when compaction is needed
     */
    get hasTombstones(): boolean {
        return this.liveNodeCount < this.nodeBound || this.liveEdgeCount < this.edgeBound;
    }

    /**
     * Grow capacity ahead of a bulk push.
     * @param nodes - the node count to fit
     * @param edges - the edge count to fit
     */
    reserve(nodes: number, edges: number): void {
        if (nodes > 0) {
            this.firstOut.ensureCapacity(nodes);
            this.firstIn.ensureCapacity(nodes);
        }
        if (edges > 0) {
            this.src.ensureCapacity(edges);
            this.dst.ensureCapacity(edges);
            this.nextOut.ensureCapacity(edges);
            this.nextIn.ensureCapacity(edges);
            if (this.weight !== null) {
                this.weight.ensureCapacity(edges);
            }
        }
    }

    /**
     * Materialise `ids` and `idToIndex` from the anonymous prefix (every node so far has id === index).
     */
    materialiseIds(): void {
        if (this.ids !== null && this.idToIndex !== null) {
            return;
        }
        const bound = this.nodeBound;
        const ids = new Array<NodeId>(bound);
        const map = new Map<NodeId, number>();
        for (let i = 0; i < bound; i++) {
            ids[i] = i;
            map.set(i, i);
        }
        this.ids = ids;
        this.idToIndex = map;
    }

    /**
     * Append a live node; the id is recorded only when the id structures exist.
     * @param id - the id, or null for an anonymous node (id === index)
     * @returns the new index
     */
    pushNode(id: NodeId | null): number {
        const index = this.nodeAlive.push(true);
        this.firstOut.push(INVALID_INDEX);
        this.firstIn.push(INVALID_INDEX);
        if (this.ids !== null && this.idToIndex !== null) {
            const stored = id ?? index;
            this.ids.push(stored);
            this.idToIndex.set(stored, index);
        }
        this.liveNodeCount++;
        return index;
    }

    /**
     * The weight array, allocated on first use and back-filled with 1 for existing edges.
     * @returns the growable weight array
     */
    ensureWeights(): GrowableTypedArray<TypedArrayData> {
        if (this.weight === null) {
            this.weight = this.allocateWeights(this.edgeBound);
            this.weight.resize(this.edgeBound, 1);
        }
        return this.weight;
    }

    /**
     * Record whether an edge's weight was explicit (design section 3.7): the `weightSet` bitmap is
     * allocated on the first edge that disagrees with every earlier one.
     * @param edge - the edge index (=== edgeBound - 1 when appending)
     * @param explicit - whether the weight was supplied
     */
    trackWeight(edge: number, explicit: boolean): void {
        if (this.weightSet !== null) {
            if (edge >= this.weightSet.length) {
                this.weightSet.resize(edge + 1, false);
            }
            this.weightSet.set(edge, explicit);
            return;
        }
        switch (this.weightMode) {
            case "none":
                this.weightMode = explicit ? "explicit" : "omitted";
                return;
            case "explicit":
                if (!explicit) {
                    this.allocateWeightSet(true, edge);
                }
                return;
            case "omitted":
                if (explicit) {
                    this.allocateWeightSet(false, edge);
                }
                return;
            case "mixed":
                return;
            default: {
                const mode: string = this.weightMode;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown weight mode ${mode}`, { mode });
            }
        }
    }

    /**
     * Whether an edge's weight was supplied explicitly (design section 3.7).
     * @param e - the edge index
     * @returns true when explicit
     */
    weightExplicit(e: number): boolean {
        if (this.weightSet === null) {
            return this.weightMode === "explicit";
        }
        return e < this.weightSet.length && this.weightSet.get(e);
    }

    /**
     * Append a live edge and link it into both incidence lists. The caller has validated the
     * endpoints and the weight.
     * @param u - source index
     * @param v - target index
     * @param weight - the weight, or undefined when omitted (stored as 1 when the array exists)
     * @returns the new edge index
     */
    pushEdge(u: number, v: number, weight: number | undefined): number {
        const e = this.src.push(u);
        this.dst.push(v);
        if (weight !== undefined) {
            this.ensureWeights().push(weight);
        } else if (this.weight !== null) {
            this.weight.push(1);
        }
        this.trackWeight(e, weight !== undefined);
        this.edgeAlive.push(true);
        this.nextOut.push(this.firstOut.get(u));
        this.firstOut.set(u, e);
        this.nextIn.push(this.firstIn.get(v));
        this.firstIn.set(v, e);
        this.liveEdgeCount++;
        if (u === v) {
            this.selfLoopCount++;
        }
        return e;
    }

    /**
     * Append `src.length` live edges at once (the bulk `addEdges` path): the same storage, weight
     * tracking and incidence-list linking as one pushEdge per edge in order, done as typed-array
     * passes. The caller has validated the endpoints and the weights.
     * @param src - source indices
     * @param dst - target indices, as many as `src`
     * @param weights - the explicit weight of every edge, or null when every weight is omitted
     * @returns the first new edge index
     */
    pushEdges(src: U32, dst: U32, weights: TypedArrayData | null): number {
        const count = src.length;
        const first = this.edgeBound;
        const end = first + count;
        if (count === 0) {
            return first;
        }
        this.src.pushAll(src);
        this.dst.pushAll(dst);
        if (weights !== null) {
            this.ensureWeights().pushAll(weights);
        } else if (this.weight !== null) {
            this.weight.resize(end, 1);
        }
        this.trackWeights(first, count, weights !== null);
        this.edgeAlive.resize(end, true);
        this.nextOut.resize(end);
        this.nextIn.resize(end);
        this.selfLoopCount += linkEdgeRange(
            this.src.array,
            this.dst.array,
            first,
            end,
            this.firstOut.array,
            this.firstIn.array,
            this.nextOut.array,
            this.nextIn.array,
        );
        this.liveEdgeCount += count;
        return first;
    }

    /**
     * trackWeight for a batch of `count` edges from `first` that all supplied or all omitted their
     * weight: the same mode transitions and bitmap contents as `count` calls in order.
     * @param first - the first edge of the batch
     * @param count - the batch size, at least 1
     * @param explicit - whether the batch supplied its weights
     */
    private trackWeights(first: number, count: number, explicit: boolean): void {
        const end = first + count;
        let bitmap = this.weightSet;
        if (bitmap === null) {
            switch (this.weightMode) {
                case "none":
                    this.weightMode = explicit ? "explicit" : "omitted";
                    return;
                case "explicit":
                    if (explicit) {
                        return;
                    }
                    bitmap = this.allocateWeightSet(true, first);
                    break;
                case "omitted":
                    if (!explicit) {
                        return;
                    }
                    bitmap = this.allocateWeightSet(false, first);
                    break;
                case "mixed":
                    return;
                default: {
                    const mode: string = this.weightMode;
                    throw new GraphFormatError("E_UNSUPPORTED", `unknown weight mode ${mode}`, { mode });
                }
            }
        }
        if (bitmap.length < first) {
            bitmap.resize(first, false);
        }
        bitmap.resize(end, explicit);
    }

    /**
     * Tombstone a live edge (O(1)).
     * @param e - a live edge index
     */
    killEdge(e: number): void {
        this.edgeAlive.set(e, false);
        this.liveEdgeCount--;
        if (this.src.get(e) === this.dst.get(e)) {
            this.selfLoopCount--;
        }
    }

    /**
     * Live edges leaving a node, walked from the out-list (O(degree)).
     * @param u - the node index
     * @returns the live edge indices in ascending order
     */
    outEdges(u: number): U32 {
        const out: number[] = [];
        for (let e = this.firstOut.get(u); e !== INVALID_INDEX; e = this.nextOut.get(e)) {
            if (this.edgeAlive.get(e)) {
                out.push(e);
            }
        }
        return Uint32Array.from(out).sort();
    }

    /**
     * Live edges entering a node, walked from the in-list (O(degree)).
     * @param v - the node index
     * @returns the live edge indices in ascending order
     */
    inEdges(v: number): U32 {
        const out: number[] = [];
        for (let e = this.firstIn.get(v); e !== INVALID_INDEX; e = this.nextIn.get(e)) {
            if (this.edgeAlive.get(e)) {
                out.push(e);
            }
        }
        return Uint32Array.from(out).sort();
    }

    /**
     * Live edges incident to a node in either orientation (the undirected outEdgesOf / inEdgesOf,
     * design section 6.6 and invariant I7): both lists walked, a self-loop listed once (O(degree)).
     * @param u - the node index
     * @returns the live edge indices in ascending order
     */
    incidentEdges(u: number): U32 {
        const out: number[] = [];
        for (let e = this.firstOut.get(u); e !== INVALID_INDEX; e = this.nextOut.get(e)) {
            if (this.edgeAlive.get(e)) {
                out.push(e);
            }
        }
        for (let e = this.firstIn.get(u); e !== INVALID_INDEX; e = this.nextIn.get(e)) {
            if (this.edgeAlive.get(e) && this.src.get(e) !== u) {
                out.push(e);
            }
        }
        return Uint32Array.from(out).sort();
    }

    /**
     * Bytes of typed staging held (capacity, not length), for `byteLength()`.
     * @returns the byte count
     */
    byteLength(): number {
        let bytes =
            this.firstOut.array.byteLength +
            this.firstIn.array.byteLength +
            this.src.array.byteLength +
            this.dst.array.byteLength +
            this.nextOut.array.byteLength +
            this.nextIn.array.byteLength +
            bitmapWordCount(this.nodeAlive.length) * 4 +
            bitmapWordCount(this.edgeAlive.length) * 4;
        if (this.weight !== null) {
            bytes += this.weight.array.byteLength;
        }
        if (this.weightSet !== null) {
            bytes += bitmapWordCount(this.weightSet.length) * 4;
        }
        for (const column of this.nodeColumns) {
            bytes += column.byteLength();
        }
        for (const column of this.edgeColumns) {
            bytes += column.byteLength();
        }
        for (const table of this.extensions) {
            for (const column of table.columns) {
                bytes += column.byteLength();
            }
        }
        return bytes;
    }

    private allocateWeights(capacity: number): GrowableTypedArray<TypedArrayData> {
        const { resizable } = this.options;
        return this.options.weightDtype === "f64"
            ? new GrowableTypedArray<TypedArrayData>(Float64Array, { capacity, resizable })
            : new GrowableTypedArray<TypedArrayData>(Float32Array, { capacity, resizable });
    }

    /**
     * Allocate the weightSet bitmap over `edge + 1` bits: every earlier edge gets `earlier`, `edge`
     * itself the opposite.
     * @param earlier - the value of every edge before `edge`
     * @param edge - the first disagreeing edge
     * @returns the new bitmap
     */
    private allocateWeightSet(earlier: boolean, edge: number): GrowableBitmap {
        const bitmap = new GrowableBitmap({ capacity: edge + 1, resizable: this.options.resizable });
        bitmap.resize(edge, earlier);
        bitmap.push(!earlier);
        this.weightSet = bitmap;
        this.weightMode = "mixed";
        return bitmap;
    }
}

// ============================================================ compaction

/** The maps of the two index spaces a `refersTo` column's values may live in (old -> new, or source -> builder). */
export interface IndexMaps {
    /** Node index map. */
    readonly node: U32;
    /** Edge index map. */
    readonly edge: U32;
}

/** An old -> new index remap with its inverse and the new count. */
export interface IndexRemap {
    /** old index -> new index or INVALID_INDEX. */
    readonly remap: U32;
    /** new index -> old index. */
    readonly origin: U32;
    /** The new count. */
    readonly count: number;
}

/**
 * The remap that drops every index whose `keep(i)` is false, assigning new indices in index order
 * (invariant I14).
 * @param bound - the old index space size
 * @param keep - whether an old index survives
 * @returns the remap, its inverse and the new count
 */
export function remapDropping(bound: number, keep: (i: number) => boolean): IndexRemap {
    const remap = new Uint32Array(bound);
    let count = 0;
    for (let i = 0; i < bound; i++) {
        remap[i] = keep(i) ? count++ : INVALID_INDEX;
    }
    const origin = new Uint32Array(count);
    for (let i = 0; i < bound; i++) {
        if (remap[i] !== INVALID_INDEX) {
            origin[remap[i]] = i;
        }
    }
    return { remap, origin, count };
}

/**
 * The remap that keeps exactly the indices whose bit is set in `keep` (a bitmap over `bound` bits),
 * assigning new indices in index order (invariant I14): `remapDropping` over the builder's alive
 * bitmaps without a call per index.
 * @param bound - the old index space size
 * @param keep - a bitmap of at least `bound` bits; bit i set = old index i survives
 * @returns the remap, its inverse and the new count
 */
export function remapKeepingBits(bound: number, keep: U32): IndexRemap {
    const remap = new Uint32Array(bound);
    let count = 0;
    for (let i = 0; i < bound; i++) {
        if (((keep[i >>> 5] >>> (i & 31)) & 1) === 1) {
            remap[i] = count++;
        } else {
            remap[i] = INVALID_INDEX;
        }
    }
    const origin = new Uint32Array(count);
    for (let i = 0; i < bound; i++) {
        const mapped = remap[i];
        if (mapped !== INVALID_INDEX) {
            origin[mapped] = i;
        }
    }
    return { remap, origin, count };
}

/**
 * The remap that keeps exactly the survivors of a merge walk (the indices with `survivorOf[e] === e`),
 * assigning new indices in index order (invariant I14): `remapDropping` for design section 6.5's
 * step-7 repeat without a call per edge.
 * @param survivorOf - old edge -> its survivor (itself for a survivor)
 * @returns the remap, its inverse and the new count
 */
export function remapSurvivors(survivorOf: U32): IndexRemap {
    const bound = survivorOf.length;
    const remap = new Uint32Array(bound);
    let count = 0;
    for (let e = 0; e < bound; e++) {
        if (survivorOf[e] === e) {
            remap[e] = count++;
        } else {
            remap[e] = INVALID_INDEX;
        }
    }
    const origin = new Uint32Array(count);
    for (let e = 0; e < bound; e++) {
        const mapped = remap[e];
        if (mapped !== INVALID_INDEX) {
            origin[mapped] = e;
        }
    }
    return { remap, origin, count };
}

/** What a compaction gathers: the node and edge remaps plus optional merged-weight overrides. */
interface CompactionPlan {
    /** The node remap (old -> new). */
    readonly nodes: IndexRemap;
    /** The edge remap (old -> new) of the ROWS; merged edges map to INVALID_INDEX here (their survivor carries them). */
    readonly edges: IndexRemap;
    /**
     * The remap `refersTo: "edge"` VALUES are rewritten through when it differs from the row remap: a
     * merged edge maps to its survivor's new index (design sections 5.11 and 7.3), never to
     * INVALID_INDEX. Null when the row remap applies to values too.
     */
    readonly edgeValues: U32 | null;
    /** Per NEW edge: the reduced weight to store where `mergedWeightStored` is set; null when no merge happened. */
    readonly mergedWeights: Float64Array | null;
    /** Per NEW edge: bit set when `mergedWeights` holds a value to store; null when no merge happened. */
    readonly mergedWeightStored: U32 | null;
    /** Per NEW edge: bit set when the merged group's weight is explicit after the merge; null when no merge happened. */
    readonly mergedWeightSet: U32 | null;
}

/**
 * Link the edges [first, end) into both incidence lists in index order, exactly as pushEdge does one
 * at a time (every list head becomes the highest linked index).
 * @param src - sources (backing view)
 * @param dst - targets (backing view)
 * @param first - the first edge to link
 * @param end - one past the last edge to link
 * @param firstOut - out-list heads
 * @param firstIn - in-list heads
 * @param nextOut - receives the out-list links
 * @param nextIn - receives the in-list links
 * @returns the number of self-loops among the linked edges
 */
function linkEdgeRange(
    src: U32,
    dst: U32,
    first: number,
    end: number,
    firstOut: U32,
    firstIn: U32,
    nextOut: U32,
    nextIn: U32,
): number {
    let loops = 0;
    for (let e = first; e < end; e++) {
        const u = src[e];
        const v = dst[e];
        nextOut[e] = firstOut[u];
        firstOut[u] = e;
        nextIn[e] = firstIn[v];
        firstIn[v] = e;
        if (u === v) {
            loops++;
        }
    }
    return loops;
}

/**
 * The edge gather of a compaction: endpoints through the node remap, in new-index order, with both
 * incidence lists rebuilt (every list head is the highest edge index, as after pushEdge in order).
 * @param src - the source staging's sources (backing view)
 * @param dst - the source staging's targets (backing view)
 * @param origin - new edge index -> old edge index
 * @param nodeRemap - old node index -> new node index
 * @param outSrc - receives the remapped sources
 * @param outDst - receives the remapped targets
 * @param firstOut - out-list heads, INVALID_INDEX on entry
 * @param firstIn - in-list heads, INVALID_INDEX on entry
 * @param nextOut - receives the out-list links
 * @param nextIn - receives the in-list links
 * @returns the self-loop count
 */
function gatherEdges(
    src: U32,
    dst: U32,
    origin: U32,
    nodeRemap: U32,
    outSrc: U32,
    outDst: U32,
    firstOut: U32,
    firstIn: U32,
    nextOut: U32,
    nextIn: U32,
): number {
    const edgeCount = origin.length;
    let loops = 0;
    for (let e = 0; e < edgeCount; e++) {
        const old = origin[e];
        const u = nodeRemap[src[old]];
        const v = nodeRemap[dst[old]];
        outSrc[e] = u;
        outDst[e] = v;
        nextOut[e] = firstOut[u];
        firstOut[u] = e;
        nextIn[e] = firstIn[v];
        firstIn[v] = e;
        if (u === v) {
            loops++;
        }
    }
    return loops;
}

/**
 * `out[i] = values[origin[i]]` for every new index; the element conversion is the target array's
 * (an f64 source into f32 staging rounds exactly as a push would).
 * @param values - the old values (backing view)
 * @param origin - new index -> old index
 * @param out - receives the gathered values
 */
function gatherValues(values: TypedArrayData, origin: U32, out: TypedArrayData): void {
    const count = origin.length;
    for (let i = 0; i < count; i++) {
        out[i] = values[origin[i]];
    }
}

/**
 * Store the reduced weight of every merged group on its survivor (design section 6.5).
 * @param merged - per new edge, the reduced weight where `stored` is set
 * @param stored - per new edge, bit set when a reduced weight is to be stored
 * @param weights - the new staging weights
 * @param edgeCount - the new edge count
 */
function applyMergedWeights(merged: Float64Array, stored: U32, weights: TypedArrayData, edgeCount: number): void {
    for (let e = 0; e < edgeCount; e++) {
        if (bitmapGet(stored, e)) {
            weights[e] = merged[e];
        }
    }
}

/**
 * Gather a new staging through a compaction plan (design section 6.3 step 1). The source is left
 * untouched; the result has no tombstones, fresh `ids` / `idToIndex` objects (or none when every node
 * is anonymous), rebuilt incidence lists, gathered columns with `refersTo` values rewritten, and
 * extension tables with their `refersTo` values rewritten.
 * @param source - the staging to compact
 * @param plan - the remaps and merge overrides
 * @returns the compacted staging
 */
export function compactStaging(source: Staging, plan: CompactionPlan): Staging {
    const { nodes, edges } = plan;
    const out = new Staging({
        ...source.options,
        weighted: source.weight !== null || plan.mergedWeights !== null,
        expectedNodes: nodes.count,
        expectedEdges: edges.count,
    });
    // nodes: every new node is live with empty incidence lists (what pushNode(null) does, in bulk); ids
    // and map are fresh objects; anonymous nodes stay anonymous (their ids are their new indices)
    out.nodeAlive.resize(nodes.count, true);
    out.firstOut.resize(nodes.count, INVALID_INDEX);
    out.firstIn.resize(nodes.count, INVALID_INDEX);
    out.liveNodeCount = nodes.count;
    if (source.ids !== null) {
        const ids = new Array<NodeId>(nodes.count);
        const map = new Map<NodeId, number>();
        for (let i = 0; i < nodes.count; i++) {
            const id = source.ids[nodes.origin[i]];
            ids[i] = id;
            map.set(id, i);
        }
        out.ids = ids;
        out.idToIndex = map;
    }
    // edges: gathered in new-index order through the backing views (one typed-array pass, design
    // section 6.3 step 1), weights and explicit-weight bits carried, lists rebuilt by pushing every
    // edge in index order exactly as pushEdge does (head of each list = highest index)
    const { remap: nodeRemap } = nodes;
    const edgeCount = edges.count;
    const { origin } = edges;
    out.weightMode = source.weightMode;
    if (source.weightSet !== null || plan.mergedWeightSet !== null) {
        out.weightSet = new GrowableBitmap({ capacity: edgeCount, resizable: source.options.resizable });
        out.weightMode = "mixed";
    }
    out.src.resize(edgeCount);
    out.dst.resize(edgeCount);
    out.nextOut.resize(edgeCount);
    out.nextIn.resize(edgeCount);
    out.edgeAlive.resize(edgeCount, true);
    out.liveEdgeCount = edgeCount;
    out.selfLoopCount = gatherEdges(
        source.src.array,
        source.dst.array,
        origin,
        nodeRemap,
        out.src.array,
        out.dst.array,
        out.firstOut.array,
        out.firstIn.array,
        out.nextOut.array,
        out.nextIn.array,
    );
    if (out.weight !== null) {
        out.weight.resize(edgeCount);
        const weights = out.weight.array;
        if (source.weight === null) {
            weights.fill(1, 0, edgeCount);
        } else {
            gatherValues(source.weight.array, origin, weights);
        }
        if (plan.mergedWeights !== null && plan.mergedWeightStored !== null) {
            applyMergedWeights(plan.mergedWeights, plan.mergedWeightStored, weights, edgeCount);
        }
    }
    if (out.weightSet !== null) {
        out.weightSet.resize(edgeCount, false);
        for (let e = 0; e < edgeCount; e++) {
            let explicit = source.weightExplicit(origin[e]);
            if (plan.mergedWeightSet !== null && bitmapGet(plan.mergedWeightSet, e)) {
                explicit = true;
            }
            if (explicit) {
                out.weightSet.set(e, true);
            }
        }
    }
    // columns
    const refs: IndexMaps = { node: nodeRemap, edge: plan.edgeValues ?? edges.remap };
    out.nodeColumns = source.nodeColumns.map((column) => column.gather(nodes.origin, valueRemapFor(column, refs)));
    out.edgeColumns = source.edgeColumns.map((column) => column.gather(edges.origin, valueRemapFor(column, refs)));
    out.extensions = source.extensions.map((table) => ({
        name: table.name,
        rowCount: table.rowCount,
        columns: table.columns.map((column) => column.gather(null, valueRemapFor(column, refs))),
    }));
    return out;
}

/**
 * The value remap that applies to a column's `refersTo`, if any.
 * @param column - the column
 * @param refs - the node and edge remaps
 * @returns the remap of the referenced space, or null
 */
function valueRemapFor(column: StagingColumn, refs: IndexMaps): U32 | null {
    switch (column.meta.refersTo) {
        case "node":
            return refs.node;
        case "edge":
            return refs.edge;
        default:
            return null;
    }
}
