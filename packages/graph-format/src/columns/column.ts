/**
 * Column implementations for every dtype of design section 5.1, the column factory that wraps raw
 * storage (MutableColumnParts) into a Column, the declaration resolver (ColumnDecl -> ColumnMeta,
 * design sections 5.3 and 5.5), the constructors the table uses for typed arrays, JS arrays and
 * empty declared columns. Bitmap and padding arithmetic comes from bitmap.ts and util/typed-array.ts.
 *
 * Storage follows Apache Arrow's layout without the dependency: a typed values buffer, an optional
 * validity bitmap (u32 words, LSB-first, ceil(rows / 32) words -- the one bitmap layout of the
 * package, decision C13), an optional dictionary, offsets or child, and per-column metadata. No
 * per-row objects exist anywhere. Column contents are frozen by contract unless meta.mutable is true
 * (design section 5.8); the mutable gating throws E_COLUMN_IMMUTABLE.
 */

import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { encodeUtf8Rows } from "../ids/string-store.js";
import {
    type BoolColumn,
    type Column,
    type ColumnDeclPatch,
    type ColumnDomain,
    type ColumnMeta,
    type ColumnOf,
    type ColumnOrigin,
    type ColumnOriginInput,
    type DictColumn,
    type Dtype,
    type DtypeValue,
    type F32,
    type F32Column,
    type F64,
    type F64Column,
    type GpuEligibility,
    type I32,
    type I32Column,
    type JsonColumn,
    type ListColumn,
    type ScalarDtype,
    type StringColumn,
    type TypedArrayData,
    type U8,
    type U8Column,
    type U32,
    type U32Column,
} from "../types/index.js";
import { type MutableColumnParts } from "../types/internal.js";
import { claimHolder, noteShared } from "../util/shared-buffers.js";
import { canViewAsPaddedU32, isOverPlainBuffer, paddedU32View, padTo4 } from "../util/typed-array.js";
import { bitmapClear, bitmapCount, bitmapGet, bitmapSet, bitmapSlice, bitmapWordCount, makeBitmap } from "./bitmap.js";
import { assertWellFormedString, buildCodeMap, DictionaryBuilder } from "./dictionary.js";
import { coerceValue, inferValuesDtype } from "./infer.js";

// ============================================================ constants and small helpers

type NumericDtype = "f32" | "f64" | "i32" | "u32" | "u8";

const ALL_DTYPES: ReadonlySet<string> = new Set([
    "f32",
    "f64",
    "i32",
    "u32",
    "u8",
    "bool",
    "dict",
    "string",
    "list",
    "json",
]);
const NUMERIC_DTYPES: ReadonlySet<string> = new Set(["f32", "f64", "i32", "u32", "u8"]);
const MAX_COMPONENTS = 16;
/** The deepest nesting a JSON value (a json cell, a default, options or extra) may have. */
const MAX_JSON_DEPTH = 256;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;
const U32_MAX = 0xffffffff;
const U8_MAX = 0xff;

const decoder = new TextDecoder();

/** Cached f32 copies of f64 columns for gpuView() (design section 10.4); dropped by markDirty(). */
const f32Cache = new WeakMap<object, F32>();

/**
 * Allocate a u8 store from which a zero-copy padded u32 view is constructible (design section 5.7):
 * the backing buffer is a multiple of 4 bytes even when the length is not.
 * @param length - the number of bytes
 * @returns a fresh Uint8Array over a padded buffer
 */
export function allocU8(length: number): U8 {
    return new Uint8Array(new ArrayBuffer(padTo4(length)), 0, length);
}

/**
 * GPU eligibility of a dtype (design section 10.4).
 * @param dtype - the column dtype
 * @returns "direct" for u32 / i32 / f32 / dict, "packed" for u8 / bool, "convert" for f64, "none" otherwise
 */
export function gpuEligibility(dtype: Dtype): GpuEligibility {
    switch (dtype) {
        case "f32":
        case "i32":
        case "u32":
        case "dict":
            return "direct";
        case "u8":
        case "bool":
            return "packed";
        case "f64":
            return "convert";
        case "string":
        case "list":
        case "json":
            return "none";
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Whether a dtype name is one a list child may have (any dtype but list).
 * @param value - the dtype name
 * @returns true for a scalar dtype
 */
function isScalarDtype(value: string): value is ScalarDtype {
    return value !== "list" && ALL_DTYPES.has(value);
}

/**
 * Whether a value is a plain object (prototype Object.prototype or null), the only object shape JSON
 * carries.
 * @param value - the value to test
 * @returns true for a plain object
 */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
    if (typeof value !== "object" || value === null) {
        return false;
    }
    const proto: unknown = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

/**
 * Reject a metadata value that JSON cannot carry (design section 5.9): only null, booleans, numbers
 * (non-finite and -0 included; the wire tags them), well-formed strings, arrays and plain objects of
 * such values are accepted. E_COLUMN_TYPE with details.field otherwise.
 * @param value - the value to check
 * @param field - the metadata field name for the error details
 * @param depth - the nesting depth of `value` (E_COLUMN_TYPE beyond MAX_JSON_DEPTH)
 */
export function assertJsonValue(value: unknown, field: string, depth = 0): void {
    if (value === null || typeof value === "boolean" || typeof value === "number") {
        return;
    }
    if (typeof value === "string") {
        assertWellFormedString(value, { field });
        return;
    }
    if (depth > MAX_JSON_DEPTH) {
        throw new GraphFormatError("E_COLUMN_TYPE", `${field} is nested deeper than ${MAX_JSON_DEPTH} levels`, {
            field,
            reason: "nesting",
        });
    }
    if (Array.isArray(value)) {
        for (const item of value) {
            assertJsonValue(item, field, depth + 1);
        }
        return;
    }
    if (isPlainObject(value)) {
        for (const key of Object.keys(value)) {
            assertJsonValue(value[key], field, depth + 1);
        }
        return;
    }
    throw new GraphFormatError("E_COLUMN_TYPE", `${field} is not a JSON value`, {
        field,
        found: value === undefined ? "undefined" : typeof value,
    });
}

/**
 * A deep copy of a JSON value (already checked by assertJsonValue): arrays and plain objects are
 * copied at every level so a caller's live object never reaches a column's metadata, where a later
 * mutation of the caller's object would change what `value()` returns for unset rows (design
 * section 5.8). The copy is the column's own; like a typed array's contents it is immutable by
 * contract.
 * @param value - the JSON value
 * @returns the copy (primitives are returned as they are)
 */
function cloneJson<T>(value: T): T {
    if (Array.isArray(value)) {
        return value.map((item: unknown) => cloneJson(item)) as T;
    }
    if (isPlainObject(value)) {
        const out: Record<string, unknown> = {};
        for (const key of Object.keys(value)) {
            out[key] = cloneJson(value[key]);
        }
        return out as T;
    }
    return value;
}

// ============================================================ declaration resolution

/**
 * Whether a number is exactly representable in a numeric dtype: any number for f32 / f64, an integer
 * in range for i32 / u32 / u8.
 * @param dtype - the numeric dtype
 * @param value - the number
 * @returns true when the dtype stores the value without change
 */
export function representableNumber(dtype: NumericDtype, value: number): boolean {
    switch (dtype) {
        case "f32":
        case "f64":
            return true;
        case "i32":
            return Number.isInteger(value) && value >= I32_MIN && value <= I32_MAX;
        case "u32":
            return Number.isInteger(value) && value >= 0 && value <= U32_MAX;
        case "u8":
            return Number.isInteger(value) && value >= 0 && value <= U8_MAX;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * The single number a declared numeric default reduces to, when it is representable: the number
 * itself for components 1, or the common value of an array of `components` equal numbers.
 * @param dtype - the numeric dtype
 * @param components - the column stride
 * @param value - the declared default
 * @returns the fill number, or null when the default is not representable as one fill value
 */
function numericDefaultFill(dtype: NumericDtype, components: number, value: unknown): number | null {
    if (typeof value === "number") {
        return representableNumber(dtype, value) ? value : null;
    }
    if (components > 1 && Array.isArray(value) && value.length === components) {
        const first: unknown = value[0];
        if (typeof first !== "number" || !representableNumber(dtype, first)) {
            return null;
        }
        for (let k = 1; k < components; k++) {
            if (!Object.is(value[k], first)) {
                return null;
            }
        }
        return first;
    }
    return null;
}

function typeError(field: string, message: string, extra?: Readonly<Record<string, unknown>>): GraphFormatError {
    return new GraphFormatError("E_COLUMN_TYPE", message, { ...extra, field });
}

function resolveOrigin(origin: ColumnOriginInput | undefined): ColumnOrigin | null {
    if (origin === undefined) {
        return null;
    }
    if (!isPlainObject(origin)) {
        throw typeError("origin", "origin must be a plain object");
    }
    const pick = (key: keyof ColumnOrigin): string | null => {
        const value = origin[key];
        if (value === undefined || value === null) {
            return null;
        }
        if (typeof value !== "string") {
            throw typeError("origin", `origin.${key} must be a string or null`, { found: typeof value });
        }
        return value;
    };
    return {
        format: pick("format"),
        id: pick("id"),
        title: pick("title"),
        type: pick("type"),
        namespace: pick("namespace"),
    };
}

function resolveComponents(field: "components" | "itemComponents", value: number | undefined, dtype: string): number {
    if (value === undefined) {
        return 1;
    }
    if (!Number.isInteger(value) || value < 1 || value > MAX_COMPONENTS) {
        throw typeError(field, `${field} must be an integer in 1..${MAX_COMPONENTS}, found ${value}`, { found: value });
    }
    if (value > 1 && !NUMERIC_DTYPES.has(dtype)) {
        throw typeError(field, `${field} > 1 is only allowed for f32 / f64 / i32 / u32 / u8, not ${dtype}`, {
            found: value,
            dtype,
        });
    }
    return value;
}

/**
 * Resolve the fill value of design section 5.3: the explicit fill when given (type-checked against
 * the dtype), else the declared default when it is representable in the dtype, else the dtype's zero
 * (0 / false / "" ; INVALID_INDEX for a refersTo u32 column so unset references are never in range).
 * @param dtype - the column dtype
 * @param components - the stride
 * @param explicit - the declared fill, if any
 * @param defaultValue - the declared default, if any
 * @param options - the declared options (the initial dictionary of a dict column)
 * @param refersTo - the referenced index space, if any
 * @returns the resolved fill
 */
function resolveFill(
    dtype: Dtype,
    components: number,
    explicit: number | string | boolean | undefined,
    defaultValue: unknown,
    options: readonly unknown[] | null,
    refersTo: "node" | "edge" | null,
): number | string | boolean {
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            if (explicit !== undefined) {
                if (typeof explicit !== "number") {
                    throw typeError("fill", `fill of a ${dtype} column must be a number`, { found: typeof explicit });
                }
                return explicit;
            }
            if (refersTo !== null) {
                return INVALID_INDEX;
            }
            const fromDefault = numericDefaultFill(dtype, components, defaultValue);
            return fromDefault ?? 0;
        }
        case "bool": {
            if (explicit !== undefined) {
                if (typeof explicit !== "boolean") {
                    throw typeError("fill", "fill of a bool column must be a boolean", { found: typeof explicit });
                }
                return explicit;
            }
            return typeof defaultValue === "boolean" ? defaultValue : false;
        }
        case "dict": {
            if (explicit !== undefined) {
                if (typeof explicit !== "string") {
                    throw typeError("fill", "fill of a dict column must be a string", { found: typeof explicit });
                }
                return explicit;
            }
            if (typeof defaultValue === "string" && (options === null || options.includes(defaultValue))) {
                return defaultValue;
            }
            if (options !== null && options.length > 0 && typeof options[0] === "string") {
                return options[0];
            }
            return "";
        }
        case "string": {
            if (explicit !== undefined) {
                if (typeof explicit !== "string") {
                    throw typeError("fill", "fill of a string column must be a string", { found: typeof explicit });
                }
                return explicit;
            }
            return "";
        }
        case "list":
        case "json": {
            // the nominal fill of a list / json column is ""; unset rows physically hold an empty list / undefined
            if (explicit !== undefined && explicit !== "") {
                throw typeError("fill", `a ${dtype} column takes no fill value`, { found: explicit });
            }
            return "";
        }
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Resolve a column declaration into the complete metadata of design section 5.5: every field
 * present, null for none, defaults applied, every rule checked (components, list child, refersTo,
 * JSON-ness of default / options / extra, fill typing). E_COLUMN_TYPE with details.field on every
 * violation.
 * @param name - the column name (overrides decl.name)
 * @param domain - the table the column belongs to
 * @param decl - the declaration; dtype is required
 * @returns the resolved metadata, frozen
 */
export function resolveColumnMeta(name: string, domain: ColumnDomain, decl: ColumnDeclPatch): ColumnMeta {
    const { dtype } = decl;
    if (dtype === undefined || !ALL_DTYPES.has(dtype)) {
        throw typeError("dtype", `column "${name}" has no valid dtype`, { name, found: dtype });
    }
    const components = resolveComponents("components", decl.components, dtype);
    let itemDtype: ScalarDtype | null = null;
    let itemComponents: number | null = null;
    if (dtype === "list") {
        const { itemDtype: declaredItem } = decl;
        if (declaredItem === undefined || !isScalarDtype(declaredItem)) {
            throw typeError("itemDtype", `list column "${name}" needs a scalar itemDtype`, {
                name,
                found: declaredItem,
            });
        }
        itemDtype = declaredItem;
        itemComponents = resolveComponents("itemComponents", decl.itemComponents, declaredItem);
    } else if (decl.itemDtype !== undefined || decl.itemComponents !== undefined) {
        throw typeError("itemDtype", `itemDtype / itemComponents are only allowed on list columns, not ${dtype}`, {
            name,
            dtype,
        });
    }
    const refersTo = decl.refersTo ?? null;
    if (refersTo !== null) {
        if (refersTo !== "node" && refersTo !== "edge") {
            throw typeError("refersTo", `refersTo must be "node" or "edge"`, { found: refersTo });
        }
        const indexColumn =
            (dtype === "u32" && components === 1) || (dtype === "list" && itemDtype === "u32" && itemComponents === 1);
        if (!indexColumn) {
            throw typeError("refersTo", `refersTo requires a u32 column or a list of u32, not ${dtype}`, {
                name,
                dtype,
            });
        }
    }
    const defaultValue = decl.default;
    if (defaultValue !== undefined) {
        assertJsonValue(defaultValue, "default");
    }
    let options: readonly unknown[] | null = null;
    if (decl.options !== undefined) {
        if (!Array.isArray(decl.options)) {
            throw typeError("options", "options must be an array");
        }
        assertJsonValue(decl.options, "options");
        if (dtype === "dict") {
            for (const option of decl.options) {
                if (typeof option !== "string") {
                    throw typeError("options", "options of a dict column must be strings", { found: typeof option });
                }
            }
        }
        options = Object.freeze(cloneJson([...decl.options]));
    }
    const extra = decl.extra ?? {};
    if (!isPlainObject(extra)) {
        throw typeError("extra", "extra must be a plain object");
    }
    assertJsonValue(extra, "extra");
    const role = decl.role ?? null;
    if (role !== null && typeof role !== "string") {
        throw typeError("role", "role must be a string", { found: typeof role });
    }
    const fill = resolveFill(dtype, components, decl.fill, defaultValue, options, refersTo);
    return Object.freeze({
        name,
        domain,
        dtype,
        components,
        itemDtype,
        itemComponents,
        nullable: decl.nullable ?? true,
        mutable: decl.mutable ?? false,
        role,
        refersTo,
        unique: decl.unique ?? false,
        default: cloneJson(defaultValue),
        fill,
        options,
        origin: resolveOrigin(decl.origin),
        dynamic: decl.dynamic ?? false,
        extra: Object.freeze(cloneJson({ ...extra })),
    });
}

/**
 * The declaration that reproduces a column's metadata, for patching an existing column's meta
 * (rename, move between tables, set() with a decl patch).
 * @param meta - the metadata to convert
 * @returns an equivalent declaration patch
 */
export function metaToDecl(meta: ColumnMeta): ColumnDeclPatch {
    return {
        name: meta.name,
        dtype: meta.dtype,
        components: meta.components,
        itemDtype: meta.itemDtype ?? undefined,
        itemComponents: meta.itemComponents ?? undefined,
        nullable: meta.nullable,
        mutable: meta.mutable,
        role: meta.role ?? undefined,
        refersTo: meta.refersTo ?? undefined,
        unique: meta.unique,
        default: meta.default,
        fill: meta.fill,
        options: meta.options ?? undefined,
        origin: meta.origin ?? undefined,
        dynamic: meta.dynamic,
        extra: meta.extra,
    };
}

// ============================================================ column base

/**
 * Members shared by every column implementation (design sections 5.3, 5.7 and 5.8).
 */
abstract class ColumnImpl<D extends Dtype> {
    readonly dtype: D;
    readonly meta: ColumnMeta;
    readonly length: number;
    readonly gpu: GpuEligibility;
    protected validityWords: U32 | null;
    protected nullCountValue: number;
    protected versionValue = 0;
    protected defaultCache: ColumnOf<D> | null = null;
    /** The column's own copy of the declared default, so a mutation of `meta.default` never reaches `value()`. */
    private readonly defaultValue: unknown;

    protected constructor(dtype: D, meta: ColumnMeta, length: number, validity: U32 | null, nullCount: number) {
        this.dtype = dtype;
        this.meta = meta;
        this.length = length;
        this.gpu = gpuEligibility(dtype);
        this.validityWords = validity;
        this.nullCountValue = nullCount;
        this.defaultValue = cloneJson(meta.default);
    }

    get validity(): U32 | null {
        return this.validityWords;
    }

    get nullCount(): number {
        return this.nullCountValue;
    }

    get version(): number {
        return this.versionValue;
    }

    abstract get byteLength(): number;

    get paddedByteLength(): number {
        return this.byteLength;
    }

    protected get validityByteLength(): number {
        return this.validityWords === null ? 0 : this.validityWords.byteLength;
    }

    /**
     * Whether the column's storage was transferred away (design sections 9.1 and 11.3): derived from
     * the array state, since a transferred ArrayBuffer leaves zero-length views behind.
     * @returns true when a typed array the column needs is empty although the column has rows
     */
    get detached(): boolean {
        if (this.length === 0) {
            return false;
        }
        const words = this.validityWords;
        return (words !== null && words.length === 0) || this.storageDetached();
    }

    isSet(row: number): boolean {
        if (!(row >= 0 && row < this.length)) {
            return false;
        }
        const words = this.validityWords;
        if (words === null) {
            return true;
        }
        if (words.length === 0) {
            throw this.detachedError();
        }
        return bitmapGet(words, row);
    }

    value(row: number): DtypeValue<D> | undefined {
        this.assertAttached();
        if (!(Number.isInteger(row) && row >= 0 && row < this.length)) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `row ${row} is out of range for a column of ${this.length} rows`,
                {
                    row,
                    length: this.length,
                    column: this.meta.name,
                },
            );
        }
        const words = this.validityWords;
        if (words !== null && !bitmapGet(words, row)) {
            // an unset row reads the column's own copy of the declared default; a structured default is
            // handed out as a fresh copy so no caller can change what later reads see (design section 5.8)
            const { defaultValue } = this;
            return defaultValue === undefined ? undefined : (cloneJson(defaultValue) as DtypeValue<D>);
        }
        return this.readValue(row) as DtypeValue<D>;
    }

    materializeDefault(): ColumnOf<D> {
        this.assertAttached();
        if (this.meta.default === undefined) {
            throw new GraphFormatError("E_NO_DEFAULT", `column "${this.meta.name}" declares no default`, {
                column: this.meta.name,
            });
        }
        if (this.nullCountValue === 0 || this.fillIsDefault()) {
            return this.self();
        }
        this.defaultCache ??= this.withDefaults();
        return this.defaultCache;
    }

    paddedU32View(): U32 {
        throw new GraphFormatError("E_GPU_INELIGIBLE", `a ${this.dtype} column has no u32 view`, {
            column: this.meta.name,
            dtype: this.dtype,
        });
    }

    markDirty(): void {
        this.assertMutable("markDirty");
        this.assertAttached();
        this.versionValue++;
        this.defaultCache = null;
        f32Cache.delete(this);
        const words = this.validityWords;
        if (words !== null) {
            this.nullCountValue = this.length - bitmapCount(words, this.length);
        }
    }

    mutableValidity(): U32 | null {
        this.assertMutable("mutableValidity");
        return this.validityWords;
    }

    setAll(): void {
        this.assertMutable("setAll");
        this.validityWords = null;
        this.nullCountValue = 0;
        this.versionValue++;
        this.defaultCache = null;
    }

    abstract slice(start: number, end: number): ColumnOf<D>;

    abstract clone(): ColumnOf<D>;

    /**
     * Whether the dtype-specific storage (data, codes, offsets) is a zero-length view although the
     * column has rows; the validity bitmap is checked by the base class.
     * @returns true when detached
     */
    protected abstract storageDetached(): boolean;

    protected abstract readValue(row: number): unknown;

    protected assertAttached(): void {
        if (this.detached) {
            throw this.detachedError();
        }
    }

    protected detachedError(): GraphFormatError {
        return new GraphFormatError("E_DETACHED", `column "${this.meta.name}" was transferred away`, {
            column: this.meta.name,
        });
    }

    protected abstract self(): ColumnOf<D>;

    protected abstract fillIsDefault(): boolean;

    protected abstract withDefaults(): ColumnOf<D>;

    protected assertMutable(operation: string): void {
        if (!this.meta.mutable) {
            throw new GraphFormatError("E_COLUMN_IMMUTABLE", `${operation}() on immutable column "${this.meta.name}"`, {
                column: this.meta.name,
                operation,
            });
        }
    }

    protected checkRange(start: number, end: number): void {
        this.assertAttached();
        if (!(Number.isInteger(start) && Number.isInteger(end) && start >= 0 && start <= end && end <= this.length)) {
            throw new GraphFormatError(
                "E_INDEX_RANGE",
                `row range [${start}, ${end}) is invalid for ${this.length} rows`,
                {
                    start,
                    end,
                    length: this.length,
                    column: this.meta.name,
                },
            );
        }
    }

    /**
     * The validity bitmap and null count of the row range [start, end); zero-copy at word boundaries.
     * @param start - first row
     * @param end - one past the last row
     * @returns the bitmap over the range (null when every row is set) and its null count
     */
    protected sliceValidity(start: number, end: number): { validity: U32 | null; nullCount: number } {
        const words = this.validityWords;
        if (words === null) {
            return { validity: null, nullCount: 0 };
        }
        const rows = end - start;
        const validity =
            start % 32 === 0
                ? shareView(words.subarray(start >>> 5, (start >>> 5) + bitmapWordCount(rows)))
                : bitmapSlice(words, start, end);
        return { validity, nullCount: rows - bitmapCount(validity, rows) };
    }

    protected cloneValidity(): U32 | null {
        this.assertAttached();
        return this.validityWords === null ? null : this.validityWords.slice();
    }
}

/**
 * Record that a zero-copy view aliases a buffer another column still holds (the owner count of
 * design section 9.1): a slice attached to a second snapshot must never let a transfer detach the
 * source, so the shared buffer is copied by `toWire({ transfer: true })` and `transferables()`.
 * @param view - a subarray of a column's storage
 * @returns the view
 */
function shareView<T extends ArrayBufferView>(view: T): T {
    noteShared(view.buffer);
    return view;
}

// ============================================================ fixed-width numeric columns

abstract class FixedColumnImpl<D extends NumericDtype, T extends TypedArrayData> extends ColumnImpl<D> {
    readonly data: T;

    protected constructor(
        dtype: D,
        meta: ColumnMeta,
        length: number,
        data: T,
        validity: U32 | null,
        nullCount: number,
    ) {
        super(dtype, meta, length, validity, nullCount);
        this.data = data;
    }

    override get byteLength(): number {
        return this.data.byteLength + this.validityByteLength;
    }

    mutableData(): T {
        this.assertMutable("mutableData");
        this.assertAttached();
        return this.data;
    }

    protected override storageDetached(): boolean {
        return this.data.length === 0 && this.length * this.meta.components > 0;
    }

    override slice(start: number, end: number): ColumnOf<D> {
        this.checkRange(start, end);
        const { components } = this.meta;
        const data = this.sliceData(start * components, end * components);
        const { validity, nullCount } = this.sliceValidity(start, end);
        return this.wrap(this.meta, end - start, data, validity, nullCount);
    }

    override clone(): ColumnOf<D> {
        this.assertAttached();
        return this.wrap(this.meta, this.length, this.copyData(), this.cloneValidity(), this.nullCountValue);
    }

    protected override readValue(row: number): number | ArrayLike<number> {
        const { components } = this.meta;
        if (components === 1) {
            return this.data[row];
        }
        return this.data.subarray(row * components, (row + 1) * components);
    }

    protected override fillIsDefault(): boolean {
        const { fill, default: defaultValue, components } = this.meta;
        if (typeof fill !== "number") {
            return false;
        }
        return numericDefaultFill(this.dtype, components, defaultValue) === fill;
    }

    protected override withDefaults(): ColumnOf<D> {
        const { components, default: defaultValue } = this.meta;
        const data = this.copyData();
        const words = this.validityWords;
        if (words !== null) {
            for (let row = 0; row < this.length; row++) {
                if (!bitmapGet(words, row)) {
                    writeNumeric(this.dtype, data, row * components, components, defaultValue, this.meta.name, row);
                }
            }
        }
        return this.wrap(this.meta, this.length, data, this.cloneValidity(), this.nullCountValue);
    }

    protected abstract wrap(
        meta: ColumnMeta,
        length: number,
        data: T,
        validity: U32 | null,
        nullCount: number,
    ): ColumnOf<D>;

    protected sliceData(start: number, end: number): T {
        return shareView(this.data.subarray(start, end) as T);
    }

    protected copyData(): T {
        return this.data.slice() as T;
    }
}

/**
 * Write one row's numeric value (a number, or an array of `components` numbers) into a typed array,
 * rejecting values the dtype cannot represent (a non-integer or out-of-range number for i32 / u32 /
 * u8) with E_COLUMN_TYPE instead of letting the typed array wrap them.
 * @param dtype - the numeric dtype
 * @param data - the destination
 * @param at - the element offset of the row
 * @param components - the stride
 * @param value - the value to write
 * @param column - the column name for error details
 * @param row - the row for error details
 */
function writeNumeric(
    dtype: NumericDtype,
    data: TypedArrayData,
    at: number,
    components: number,
    value: unknown,
    column: string,
    row: number,
): void {
    if (typeof value === "number") {
        assertRepresentable(dtype, value, column, row);
        if (components === 1) {
            data[at] = value;
        } else {
            data.fill(value, at, at + components);
        }
        return;
    }
    if (components === 1 || !isArrayLikeNumbers(value, components)) {
        throw new GraphFormatError(
            "E_COLUMN_TYPE",
            `row ${row} of ${dtype} column "${column}" is not ${components} number(s)`,
            { column, row, components, found: typeof value },
        );
    }
    for (let k = 0; k < components; k++) {
        assertRepresentable(dtype, value[k], column, row);
    }
    for (let k = 0; k < components; k++) {
        data[at + k] = value[k];
    }
}

function assertRepresentable(dtype: NumericDtype, value: number, column: string, row: number): void {
    if (!representableNumber(dtype, value)) {
        throw new GraphFormatError(
            "E_COLUMN_TYPE",
            `row ${row} of ${dtype} column "${column}": ${value} is not representable`,
            { column, row, value },
        );
    }
}

/**
 * Whether a value is an array-like of exactly `length` numbers (a multi-component cell).
 * @param value - the value
 * @param length - the required length
 * @returns true for a numeric vector of that length
 */
export function isArrayLikeNumbers(value: unknown, length: number): value is ArrayLike<number> {
    if (typeof value !== "object" || value === null) {
        return false;
    }
    const arrayLike = value as ArrayLike<unknown>;
    if (arrayLike.length !== length) {
        return false;
    }
    for (let k = 0; k < length; k++) {
        if (typeof arrayLike[k] !== "number") {
            return false;
        }
    }
    return true;
}

class F32ColumnImpl extends FixedColumnImpl<"f32", F32> implements F32Column {
    constructor(meta: ColumnMeta, length: number, data: F32, validity: U32 | null, nullCount: number) {
        super("f32", meta, length, data, validity, nullCount);
    }

    protected override self(): F32Column {
        return this;
    }

    protected override wrap(
        meta: ColumnMeta,
        length: number,
        data: F32,
        validity: U32 | null,
        nullCount: number,
    ): F32Column {
        return new F32ColumnImpl(meta, length, data, validity, nullCount);
    }
}

class F64ColumnImpl extends FixedColumnImpl<"f64", F64> implements F64Column {
    constructor(meta: ColumnMeta, length: number, data: F64, validity: U32 | null, nullCount: number) {
        super("f64", meta, length, data, validity, nullCount);
    }

    protected override self(): F64Column {
        return this;
    }

    protected override wrap(
        meta: ColumnMeta,
        length: number,
        data: F64,
        validity: U32 | null,
        nullCount: number,
    ): F64Column {
        return new F64ColumnImpl(meta, length, data, validity, nullCount);
    }
}

class I32ColumnImpl extends FixedColumnImpl<"i32", I32> implements I32Column {
    constructor(meta: ColumnMeta, length: number, data: I32, validity: U32 | null, nullCount: number) {
        super("i32", meta, length, data, validity, nullCount);
    }

    protected override self(): I32Column {
        return this;
    }

    protected override wrap(
        meta: ColumnMeta,
        length: number,
        data: I32,
        validity: U32 | null,
        nullCount: number,
    ): I32Column {
        return new I32ColumnImpl(meta, length, data, validity, nullCount);
    }
}

class U32ColumnImpl extends FixedColumnImpl<"u32", U32> implements U32Column {
    constructor(meta: ColumnMeta, length: number, data: U32, validity: U32 | null, nullCount: number) {
        super("u32", meta, length, data, validity, nullCount);
    }

    override paddedU32View(): U32 {
        return this.data;
    }

    protected override self(): U32Column {
        return this;
    }

    protected override wrap(
        meta: ColumnMeta,
        length: number,
        data: U32,
        validity: U32 | null,
        nullCount: number,
    ): U32Column {
        return new U32ColumnImpl(meta, length, data, validity, nullCount);
    }
}

class U8ColumnImpl extends FixedColumnImpl<"u8", U8> implements U8Column {
    constructor(meta: ColumnMeta, length: number, data: U8, validity: U32 | null, nullCount: number) {
        super("u8", meta, length, data, validity, nullCount);
    }

    override get paddedByteLength(): number {
        return padTo4(this.data.byteLength) + this.validityByteLength;
    }

    override paddedU32View(): U32 {
        return paddedU32View(this.data);
    }

    protected override self(): U8Column {
        return this;
    }

    protected override wrap(
        meta: ColumnMeta,
        length: number,
        data: U8,
        validity: U32 | null,
        nullCount: number,
    ): U8Column {
        return new U8ColumnImpl(meta, length, data, validity, nullCount);
    }

    protected override sliceData(start: number, end: number): U8 {
        if (start % 4 === 0) {
            return shareView(this.data.subarray(start, end));
        }
        const out = allocU8(end - start);
        out.set(this.data.subarray(start, end));
        return out;
    }

    protected override copyData(): U8 {
        const out = allocU8(this.data.length);
        out.set(this.data);
        return out;
    }
}

// ============================================================ bool column

class BoolColumnImpl extends ColumnImpl<"bool"> implements BoolColumn {
    readonly data: U32;

    constructor(meta: ColumnMeta, length: number, data: U32, validity: U32 | null, nullCount: number) {
        super("bool", meta, length, validity, nullCount);
        this.data = data;
    }

    override get byteLength(): number {
        return this.data.byteLength + this.validityByteLength;
    }

    mutableData(): U32 {
        this.assertMutable("mutableData");
        this.assertAttached();
        return this.data;
    }

    override paddedU32View(): U32 {
        this.assertAttached();
        return this.data;
    }

    override slice(start: number, end: number): BoolColumn {
        this.checkRange(start, end);
        const rows = end - start;
        const data =
            start % 32 === 0
                ? shareView(this.data.subarray(start >>> 5, (start >>> 5) + bitmapWordCount(rows)))
                : bitmapSlice(this.data, start, end);
        const { validity, nullCount } = this.sliceValidity(start, end);
        return new BoolColumnImpl(this.meta, rows, data, validity, nullCount);
    }

    override clone(): BoolColumn {
        this.assertAttached();
        return new BoolColumnImpl(this.meta, this.length, this.data.slice(), this.cloneValidity(), this.nullCountValue);
    }

    protected override storageDetached(): boolean {
        return this.data.length === 0;
    }

    protected override self(): BoolColumn {
        return this;
    }

    protected override readValue(row: number): boolean {
        return bitmapGet(this.data, row);
    }

    protected override fillIsDefault(): boolean {
        return this.meta.fill === this.meta.default;
    }

    protected override withDefaults(): BoolColumn {
        const { default: defaultValue, name } = this.meta;
        if (typeof defaultValue !== "boolean") {
            throw new GraphFormatError("E_COLUMN_TYPE", `default of bool column "${name}" is not a boolean`, {
                column: name,
                field: "default",
            });
        }
        const data = this.data.slice();
        const words = this.validityWords;
        if (words !== null) {
            for (let row = 0; row < this.length; row++) {
                if (!bitmapGet(words, row)) {
                    if (defaultValue) {
                        bitmapSet(data, row);
                    } else {
                        bitmapClear(data, row);
                    }
                }
            }
        }
        return new BoolColumnImpl(this.meta, this.length, data, this.cloneValidity(), this.nullCountValue);
    }
}

// ============================================================ dict column

class DictColumnImpl extends ColumnImpl<"dict"> implements DictColumn {
    readonly codes: U32;
    readonly dictionary: readonly string[];
    private codeMap: Map<string, number> | null = null;

    constructor(
        meta: ColumnMeta,
        length: number,
        codes: U32,
        dictionary: readonly string[],
        validity: U32 | null,
        nullCount: number,
    ) {
        super("dict", meta, length, validity, nullCount);
        this.codes = codes;
        this.dictionary = dictionary;
    }

    override get byteLength(): number {
        return this.codes.byteLength + this.validityByteLength;
    }

    codeOf(value: string): number {
        this.codeMap ??= buildCodeMap(this.dictionary);
        const code = this.codeMap.get(value);
        return code === undefined ? INVALID_INDEX : code;
    }

    mutableData(): U32 {
        this.assertMutable("mutableData");
        this.assertAttached();
        return this.codes;
    }

    override paddedU32View(): U32 {
        this.assertAttached();
        return this.codes;
    }

    override slice(start: number, end: number): DictColumn {
        this.checkRange(start, end);
        const { validity, nullCount } = this.sliceValidity(start, end);
        return new DictColumnImpl(
            this.meta,
            end - start,
            shareView(this.codes.subarray(start, end)),
            this.dictionary,
            validity,
            nullCount,
        );
    }

    protected override storageDetached(): boolean {
        return this.codes.length === 0;
    }

    override clone(): DictColumn {
        this.assertAttached();
        return new DictColumnImpl(
            this.meta,
            this.length,
            this.codes.slice(),
            [...this.dictionary],
            this.cloneValidity(),
            this.nullCountValue,
        );
    }

    protected override self(): DictColumn {
        return this;
    }

    protected override readValue(row: number): string {
        return this.dictionary[this.codes[row]];
    }

    protected override fillIsDefault(): boolean {
        return this.meta.fill === this.meta.default;
    }

    protected override withDefaults(): DictColumn {
        const { default: defaultValue, name } = this.meta;
        if (typeof defaultValue !== "string") {
            throw new GraphFormatError("E_COLUMN_TYPE", `default of dict column "${name}" is not a string`, {
                column: name,
                field: "default",
            });
        }
        let dict: readonly string[] = this.dictionary;
        let code = this.codeOf(defaultValue);
        if (code === INVALID_INDEX) {
            code = dict.length;
            dict = [...dict, defaultValue];
        }
        const codes = this.codes.slice();
        const words = this.validityWords;
        if (words !== null) {
            for (let row = 0; row < this.length; row++) {
                if (!bitmapGet(words, row)) {
                    codes[row] = code;
                }
            }
        }
        return new DictColumnImpl(this.meta, this.length, codes, dict, this.cloneValidity(), this.nullCountValue);
    }
}

// ============================================================ string column

/**
 * Encode strings into an Arrow Utf8 store: rows + 1 offsets and the concatenated bytes in a padded
 * u8 store.
 * @param strings - the rows
 * @returns the offsets and bytes
 */
function encodeStrings(strings: readonly string[]): { offsets: U32; utf8: U8 } {
    const encoded = encodeUtf8Rows(strings, strings.length);
    const utf8 = allocU8(encoded.utf8.length);
    utf8.set(encoded.utf8);
    return { offsets: encoded.offsets, utf8 };
}

class StringColumnImpl extends ColumnImpl<"string"> implements StringColumn {
    private offsetsStore: U32 | null;
    private utf8Store: U8 | null;
    private strings: (string | undefined)[] | null;

    constructor(
        meta: ColumnMeta,
        length: number,
        offsets: U32 | null,
        utf8: U8 | null,
        strings: (string | undefined)[] | null,
        validity: U32 | null,
        nullCount: number,
    ) {
        super("string", meta, length, validity, nullCount);
        this.offsetsStore = offsets;
        this.utf8Store = utf8;
        this.strings = strings;
    }

    get offsets(): U32 {
        this.store();
        return this.offsetsStore as U32;
    }

    get utf8(): U8 {
        this.store();
        return this.utf8Store as U8;
    }

    override get byteLength(): number {
        this.store();
        return (this.offsetsStore as U32).byteLength + (this.utf8Store as U8).byteLength + this.validityByteLength;
    }

    valueAt(row: number): string {
        const cached = this.strings?.[row];
        if (cached !== undefined) {
            return cached;
        }
        this.assertAttached();
        this.store();
        const offsets = this.offsetsStore as U32;
        const utf8 = this.utf8Store as U8;
        const text = decoder.decode(utf8.subarray(offsets[row], offsets[row + 1]));
        this.strings ??= new Array<string | undefined>(this.length);
        this.strings[row] = text;
        return text;
    }

    decodeAll(): string[] {
        const out = new Array<string>(this.length);
        for (let row = 0; row < this.length; row++) {
            out[row] = this.valueAt(row);
        }
        return out;
    }

    override slice(start: number, end: number): StringColumn {
        this.checkRange(start, end);
        const rows = end - start;
        const { validity, nullCount } = this.sliceValidity(start, end);
        const strings = this.strings === null ? null : this.strings.slice(start, end);
        if (this.offsetsStore === null || this.utf8Store === null) {
            return new StringColumnImpl(this.meta, rows, null, null, strings, validity, nullCount);
        }
        const base = this.offsetsStore[start];
        const offsets = new Uint32Array(rows + 1);
        for (let i = 0; i <= rows; i++) {
            offsets[i] = this.offsetsStore[start + i] - base;
        }
        const utf8 = shareView(this.utf8Store.subarray(base, this.offsetsStore[end]));
        return new StringColumnImpl(this.meta, rows, offsets, utf8, strings, validity, nullCount);
    }

    override clone(): StringColumn {
        this.assertAttached();
        const strings = this.strings === null ? null : [...this.strings];
        if (this.offsetsStore === null || this.utf8Store === null) {
            return new StringColumnImpl(
                this.meta,
                this.length,
                null,
                null,
                strings,
                this.cloneValidity(),
                this.nullCountValue,
            );
        }
        const utf8 = allocU8(this.utf8Store.length);
        utf8.set(this.utf8Store);
        return new StringColumnImpl(
            this.meta,
            this.length,
            this.offsetsStore.slice(),
            utf8,
            strings,
            this.cloneValidity(),
            this.nullCountValue,
        );
    }

    /**
     * Whether the Utf8 store has been materialised (the wire writer asks before encoding).
     * @returns true when offsets and utf8 exist without being computed
     */
    hasStore(): boolean {
        return this.offsetsStore !== null;
    }

    /**
     * The decoded cache as it stands (sparse), for zero-copy re-wrapping.
     * @returns the cache, or null when no row has been decoded
     */
    decodedCache(): (string | undefined)[] | null {
        return this.strings;
    }

    protected override self(): StringColumn {
        return this;
    }

    protected override storageDetached(): boolean {
        // a materialised store is detached when its offsets vanished; undecoded rows then cannot be read
        return this.offsetsStore !== null && this.offsetsStore.length === 0;
    }

    protected override readValue(row: number): string {
        return this.valueAt(row);
    }

    protected override fillIsDefault(): boolean {
        return this.meta.fill === this.meta.default;
    }

    protected override withDefaults(): StringColumn {
        const { default: defaultValue, name } = this.meta;
        if (typeof defaultValue !== "string") {
            throw new GraphFormatError("E_COLUMN_TYPE", `default of string column "${name}" is not a string`, {
                column: name,
                field: "default",
            });
        }
        const strings = this.decodeAll();
        const words = this.validityWords;
        if (words !== null) {
            for (let row = 0; row < this.length; row++) {
                if (!bitmapGet(words, row)) {
                    strings[row] = defaultValue;
                }
            }
        }
        return new StringColumnImpl(
            this.meta,
            this.length,
            null,
            null,
            strings,
            this.cloneValidity(),
            this.nullCountValue,
        );
    }

    /**
     * Materialise the Utf8 store from the decoded strings on first need (design section 5.7); a
     * store that was transferred away is E_DETACHED.
     */
    private store(): void {
        if (this.offsetsStore !== null && this.utf8Store !== null) {
            this.assertAttached();
            return;
        }
        const strings = this.strings ?? [];
        const dense = new Array<string>(this.length);
        for (let row = 0; row < this.length; row++) {
            dense[row] = strings[row] ?? "";
        }
        const encoded = encodeStrings(dense);
        this.offsetsStore = encoded.offsets;
        this.utf8Store = encoded.utf8;
        // the column is the first holder of the store it materialises (design section 9.1)
        claimHolder(encoded.offsets.buffer);
        claimHolder(encoded.utf8.buffer);
    }
}

// ============================================================ list column

type ChildColumn = Exclude<Column, ListColumn>;

class ListColumnImpl extends ColumnImpl<"list"> implements ListColumn {
    readonly offsets: U32;
    readonly child: ChildColumn;

    constructor(
        meta: ColumnMeta,
        length: number,
        offsets: U32,
        child: ChildColumn,
        validity: U32 | null,
        nullCount: number,
    ) {
        super("list", meta, length, validity, nullCount);
        this.offsets = offsets;
        this.child = child;
    }

    override get byteLength(): number {
        return this.offsets.byteLength + this.child.byteLength + this.validityByteLength;
    }

    sliceOf(row: number): readonly unknown[] {
        this.assertAttached();
        const start = this.offsets[row];
        const end = this.offsets[row + 1];
        const out = new Array<unknown>(end - start);
        for (let i = start; i < end; i++) {
            out[i - start] = this.child.value(i);
        }
        return out;
    }

    override slice(start: number, end: number): ListColumn {
        this.checkRange(start, end);
        const rows = end - start;
        const base = this.offsets[start];
        const offsets = new Uint32Array(rows + 1);
        for (let i = 0; i <= rows; i++) {
            offsets[i] = this.offsets[start + i] - base;
        }
        const { validity, nullCount } = this.sliceValidity(start, end);
        return new ListColumnImpl(
            this.meta,
            rows,
            offsets,
            this.child.slice(base, this.offsets[end]),
            validity,
            nullCount,
        );
    }

    override clone(): ListColumn {
        this.assertAttached();
        return new ListColumnImpl(
            this.meta,
            this.length,
            this.offsets.slice(),
            this.child.clone(),
            this.cloneValidity(),
            this.nullCountValue,
        );
    }

    protected override storageDetached(): boolean {
        return this.offsets.length === 0 || isColumnDetached(this.child);
    }

    protected override self(): ListColumn {
        return this;
    }

    protected override readValue(row: number): readonly unknown[] {
        return this.sliceOf(row);
    }

    protected override fillIsDefault(): boolean {
        return false;
    }

    protected override withDefaults(): ListColumn {
        const { default: defaultValue, name } = this.meta;
        if (!Array.isArray(defaultValue)) {
            throw new GraphFormatError("E_COLUMN_TYPE", `default of list column "${name}" is not an array`, {
                column: name,
                field: "default",
            });
        }
        const rows = new Array<readonly unknown[]>(this.length);
        const words = this.validityWords;
        for (let row = 0; row < this.length; row++) {
            rows[row] = words !== null && !bitmapGet(words, row) ? defaultValue : this.sliceOf(row);
        }
        const parts = listPartsFromValues(this.meta, rows);
        return new ListColumnImpl(
            this.meta,
            this.length,
            parts.offsets,
            parts.child,
            this.cloneValidity(),
            this.nullCountValue,
        );
    }
}

// ============================================================ json column

class JsonColumnImpl extends ColumnImpl<"json"> implements JsonColumn {
    readonly values: readonly unknown[];

    constructor(meta: ColumnMeta, length: number, values: readonly unknown[], validity: U32 | null, nullCount: number) {
        super("json", meta, length, validity, nullCount);
        this.values = values;
    }

    override get byteLength(): number {
        return this.validityByteLength;
    }

    override slice(start: number, end: number): JsonColumn {
        this.checkRange(start, end);
        const { validity, nullCount } = this.sliceValidity(start, end);
        return new JsonColumnImpl(this.meta, end - start, this.values.slice(start, end), validity, nullCount);
    }

    override clone(): JsonColumn {
        this.assertAttached();
        return new JsonColumnImpl(this.meta, this.length, [...this.values], this.cloneValidity(), this.nullCountValue);
    }

    protected override storageDetached(): boolean {
        return false;
    }

    protected override self(): JsonColumn {
        return this;
    }

    protected override readValue(row: number): unknown {
        return this.values[row];
    }

    protected override fillIsDefault(): boolean {
        return false;
    }

    protected override withDefaults(): JsonColumn {
        const { default: defaultValue } = this.meta;
        const values = [...this.values];
        const words = this.validityWords;
        if (words !== null) {
            for (let row = 0; row < this.length; row++) {
                if (!bitmapGet(words, row)) {
                    values[row] = defaultValue;
                }
            }
        }
        return new JsonColumnImpl(this.meta, this.length, values, this.cloneValidity(), this.nullCountValue);
    }
}

/**
 * Whether a value is a Column built by this package's factory (the structural Column type also
 * admits a foreign object with the right members, which the table refuses with E_COLUMN_TYPE).
 * @param value - the candidate
 * @returns true for an instance of a column implementation
 */
export function isPackageColumn(value: unknown): value is Column {
    return value instanceof ColumnImpl;
}

/**
 * Whether a column's storage was transferred away (design section 9.1): derived from the array
 * state. Every accessor of such a column throws E_DETACHED; the wire module and the checksum
 * comparison ask first.
 * @param column - any column
 * @returns true when detached
 */
export function isColumnDetached(column: Column): boolean {
    return (column as unknown as ColumnImpl<Dtype>).detached;
}

// ============================================================ factory

function lengthError(meta: ColumnMeta, what: string, expected: number, found: number): GraphFormatError {
    return new GraphFormatError(
        "E_COLUMN_LENGTH",
        `column "${meta.name}": ${what} has length ${found}, expected ${expected}`,
        {
            column: meta.name,
            what,
            expected,
            found,
        },
    );
}

function slotError(meta: ColumnMeta, message: string): GraphFormatError {
    return new GraphFormatError("E_COLUMN_TYPE", `column "${meta.name}": ${message}`, { column: meta.name });
}

/**
 * Wrap raw column storage into a Column of the dtype named by parts.meta (the column factory of
 * design section 5.7). Checks the slot use and the length rules of invariant I12 (E_COLUMN_LENGTH /
 * E_COLUMN_TYPE / E_COLUMN_ALIGNMENT), recomputes nullCount from the validity bitmap, and adopts
 * every buffer by reference.
 * @param parts - the storage slots; every slot the dtype does not use must be null
 * @returns the column
 */
export function createColumn(parts: MutableColumnParts): Column {
    const { meta, length } = parts;
    if (!(Number.isInteger(length) && length >= 0)) {
        throw new GraphFormatError("E_COLUMN_LENGTH", `column "${meta.name}": invalid row count ${length}`, {
            column: meta.name,
            found: length,
        });
    }
    const { validity: given } = parts;
    let validity: U32 | null = null;
    if (given !== null) {
        if (!meta.nullable) {
            throw slotError(meta, "a non-nullable column cannot carry a validity bitmap");
        }
        if (!(given instanceof Uint32Array)) {
            throw slotError(meta, "validity must be a Uint32Array");
        }
        if (given.length !== bitmapWordCount(length)) {
            throw lengthError(meta, "validity", bitmapWordCount(length), given.length);
        }
        validity = given;
    }
    const nullCount = validity === null ? 0 : length - bitmapCount(validity, length);
    const { dtype } = meta;
    switch (dtype) {
        case "f32":
            return new F32ColumnImpl(meta, length, checkNumeric(parts, Float32Array), validity, nullCount);
        case "f64":
            return new F64ColumnImpl(meta, length, checkNumeric(parts, Float64Array), validity, nullCount);
        case "i32":
            return new I32ColumnImpl(meta, length, checkNumeric(parts, Int32Array), validity, nullCount);
        case "u32":
            return new U32ColumnImpl(meta, length, checkNumeric(parts, Uint32Array), validity, nullCount);
        case "u8": {
            const data = checkNumeric(parts, Uint8Array);
            if (!canViewAsPaddedU32(data)) {
                throw new GraphFormatError(
                    "E_COLUMN_ALIGNMENT",
                    `column "${meta.name}": no padded u32 view over the u8 data`,
                    {
                        column: meta.name,
                        byteOffset: data.byteOffset,
                        byteLength: data.byteLength,
                    },
                );
            }
            return new U8ColumnImpl(meta, length, data, validity, nullCount);
        }
        case "bool": {
            if (!(parts.data instanceof Uint32Array)) {
                throw slotError(meta, "bool data must be packed Uint32Array words");
            }
            if (parts.data.length !== bitmapWordCount(length)) {
                throw lengthError(meta, "data", bitmapWordCount(length), parts.data.length);
            }
            return new BoolColumnImpl(meta, length, parts.data, validity, nullCount);
        }
        case "dict": {
            if (!(parts.data instanceof Uint32Array)) {
                throw slotError(meta, "dict codes must be a Uint32Array");
            }
            if (parts.data.length !== length) {
                throw lengthError(meta, "codes", length, parts.data.length);
            }
            if (parts.dictionary === null) {
                throw slotError(meta, "dict column needs a dictionary");
            }
            return new DictColumnImpl(meta, length, parts.data, parts.dictionary, validity, nullCount);
        }
        case "string": {
            if (parts.offsets === null || parts.utf8 === null) {
                const { strings } = parts;
                if (strings === null || strings.length !== length) {
                    throw lengthError(meta, "strings", length, strings === null ? -1 : strings.length);
                }
                for (let row = 0; row < length; row++) {
                    if (typeof strings[row] !== "string") {
                        throw slotError(meta, `row ${row} is not a decoded string and no Utf8 store is present`);
                    }
                }
                return new StringColumnImpl(meta, length, null, null, strings, validity, nullCount);
            }
            if (parts.offsets.length !== length + 1) {
                throw lengthError(meta, "offsets", length + 1, parts.offsets.length);
            }
            if (parts.offsets[length] !== parts.utf8.length) {
                throw lengthError(meta, "utf8", parts.offsets[length], parts.utf8.length);
            }
            return new StringColumnImpl(meta, length, parts.offsets, parts.utf8, parts.strings, validity, nullCount);
        }
        case "list": {
            if (parts.offsets === null || parts.offsets.length !== length + 1) {
                throw lengthError(meta, "offsets", length + 1, parts.offsets === null ? -1 : parts.offsets.length);
            }
            const { child } = parts;
            if (child === null || child.dtype === "list") {
                throw slotError(meta, "list column needs a non-list child column");
            }
            if (child.validity !== null || child.meta.nullable) {
                throw slotError(meta, "list child must be non-nullable");
            }
            if (child.dtype !== meta.itemDtype || child.meta.components !== meta.itemComponents) {
                throw slotError(
                    meta,
                    `list child is ${child.dtype} x${child.meta.components}, declared ${String(meta.itemDtype)} x${String(meta.itemComponents)}`,
                );
            }
            if (parts.offsets[length] !== child.length) {
                throw lengthError(meta, "child", parts.offsets[length], child.length);
            }
            return new ListColumnImpl(meta, length, parts.offsets, child, validity, nullCount);
        }
        case "json": {
            if (parts.values === null || parts.values.length !== length) {
                throw lengthError(meta, "values", length, parts.values === null ? -1 : parts.values.length);
            }
            return new JsonColumnImpl(meta, length, parts.values, validity, nullCount);
        }
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

function checkNumeric<T extends TypedArrayData>(parts: MutableColumnParts, ctor: new (length: number) => T): T {
    const { meta, length } = parts;
    if (!(parts.data instanceof ctor)) {
        throw slotError(meta, `${meta.dtype} data must be a ${ctor.name}`);
    }
    const expected = length * meta.components;
    if (parts.data.length !== expected) {
        throw lengthError(meta, "data", expected, parts.data.length);
    }
    return parts.data;
}

/**
 * The raw storage of a column, sharing every buffer by reference (the inverse of createColumn): used
 * to re-wrap a column under new metadata and by the remap helpers.
 * @param column - the column to unwrap
 * @returns its storage slots
 */
export function partsOf(column: Column): MutableColumnParts {
    const base: MutableColumnParts = {
        meta: column.meta,
        length: column.length,
        data: null,
        validity: column.validity,
        nullCount: column.nullCount,
        dictionary: null,
        offsets: null,
        utf8: null,
        strings: null,
        child: null,
        values: null,
    };
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
        case "bool":
            base.data = column.data;
            return base;
        case "dict":
            base.data = column.codes;
            base.dictionary = column.dictionary as string[];
            return base;
        case "string": {
            if (column instanceof StringColumnImpl) {
                base.strings = column.decodedCache();
                if (column.hasStore()) {
                    base.offsets = column.offsets;
                    base.utf8 = column.utf8;
                }
                return base;
            }
            base.offsets = column.offsets;
            base.utf8 = column.utf8;
            return base;
        }
        case "list":
            base.offsets = column.offsets;
            base.child = column.child;
            return base;
        case "json":
            base.values = column.values as unknown[];
            return base;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * The distinct backing buffers of a column's materialised typed storage (data or codes, validity,
 * a materialised Utf8 store, a list's offsets and child), for the owner count of design section
 * 9.1: a table claims them when it attaches the column, so a buffer viewed from two tables (a
 * zero-copy slice, a same-realm wire receiver) is recognised as shared. A string store that has not
 * been materialised is not encoded by asking.
 * @param column - the column
 * @returns the buffers, each once
 */
export function columnBuffers(column: Column): ArrayBuffer[] {
    const out = new Set<ArrayBuffer>();
    const add = (view: ArrayBufferView | null): void => {
        if (view !== null) {
            out.add(view.buffer as ArrayBuffer);
        }
    };
    add(column.validity);
    switch (column.dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
        case "bool":
            add(column.data);
            break;
        case "dict":
            add(column.codes);
            break;
        case "string":
            if (!(column instanceof StringColumnImpl) || column.hasStore()) {
                add(column.offsets);
                add(column.utf8);
            }
            break;
        case "list":
            add(column.offsets);
            for (const buffer of columnBuffers(column.child)) {
                out.add(buffer);
            }
            break;
        case "json":
            break;
        default: {
            const unknown: never = column;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${(unknown as Column).dtype}`, {});
        }
    }
    return [...out];
}

/**
 * The same storage under different metadata (a rename, a move to another table, a declaration
 * patch); the buffers are shared, the Column object is new. The dtype, components and list child
 * shape must be unchanged (E_COLUMN_TYPE otherwise); the fill is kept because the unset rows
 * physically hold it; a nullable -> non-nullable change requires no unset rows.
 * @param column - the column to re-wrap
 * @param meta - the new metadata
 * @returns a new column sharing the storage
 */
export function rewrapColumn(column: Column, meta: ColumnMeta): Column {
    const old = column.meta;
    if (
        meta.dtype !== old.dtype ||
        meta.components !== old.components ||
        meta.itemDtype !== old.itemDtype ||
        meta.itemComponents !== old.itemComponents
    ) {
        throw new GraphFormatError(
            "E_COLUMN_TYPE",
            `column "${old.name}" is ${old.dtype} x${old.components}; a patch cannot change its dtype`,
            {
                column: old.name,
                field: "dtype",
            },
        );
    }
    if (meta.fill !== old.fill) {
        throw new GraphFormatError("E_COLUMN_TYPE", `the fill of existing column "${old.name}" cannot change`, {
            column: old.name,
            field: "fill",
        });
    }
    const parts = partsOf(column);
    if (!meta.nullable && column.validity !== null) {
        if (column.nullCount > 0) {
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `column "${old.name}" has ${column.nullCount} unset rows and cannot become non-nullable`,
                {
                    column: old.name,
                    field: "nullable",
                    nullCount: column.nullCount,
                },
            );
        }
        parts.validity = null;
    }
    parts.meta = meta;
    return createColumn(parts);
}

/**
 * Release the cached f32 copy a `gpuView()` of an f64 column keeps (design section 7.2:
 * `dropCaches()` releases every cached gpuView copy). A no-op for every other dtype.
 * @param column - the column
 */
export function dropGpuViewCache(column: Column): void {
    f32Cache.delete(column);
}

/**
 * The array a GPU binds for a column (design section 10.4): its own data for u32 / i32 / f32, the
 * padded view for u8, the packed words for bool, the codes for dict, and a cached f32 copy for f64
 * (dropped by markDirty() and by GraphSnapshot.dropCaches()).
 * @param column - the column
 * @returns the bindable array; E_GPU_INELIGIBLE for string / list / json
 */
export function gpuViewOf(column: Column): U32 | I32 | F32 {
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "i32":
        case "u32":
        case "bool":
            return column.data;
        case "u8":
            return column.paddedU32View();
        case "dict":
            return column.codes;
        case "f64": {
            let cached = f32Cache.get(column);
            if (cached === undefined) {
                cached = new Float32Array(column.data);
                f32Cache.set(column, cached);
            }
            return cached;
        }
        case "string":
        case "list":
        case "json":
            throw new GraphFormatError("E_GPU_INELIGIBLE", `a ${dtype} column cannot be bound by the GPU`, {
                column: column.meta.name,
                dtype,
            });
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

// ============================================================ construction from declarations and values

/**
 * The physical dict code stored in rows that hold the fill: the code of the fill string, interned
 * on demand when those rows are SET (a non-nullable column), or 0 when the fill is not a member and
 * the rows are unset (their code is never read).
 * @param dictionary - the dictionary
 * @param fill - the column's fill
 * @param set - whether the rows holding the fill are set
 * @returns the code
 */
function fillCode(dictionary: DictionaryBuilder, fill: number | string | boolean, set: boolean): number {
    const text = typeof fill === "string" ? fill : "";
    if (set) {
        // a SET row holding the fill (a non-nullable column) needs a code that names a member: the
        // nominal "" is interned on demand rather than borrowing code 0, which may belong to another
        // value or to nothing
        return dictionary.intern(text);
    }
    // an unset row's code is never read; 0 when the fill is not a member
    const code = dictionary.codeOf(text);
    return code === INVALID_INDEX ? 0 : code;
}

/**
 * The initial dictionary of a dict column: the declared options in order, then the fill when it is
 * a string not among them (so the fill code names a member).
 * @param meta - the column metadata
 * @returns the seeded dictionary builder
 */
export function seedDictionary(meta: ColumnMeta): DictionaryBuilder {
    const dictionary = new DictionaryBuilder(meta.options === null ? undefined : (meta.options as readonly string[]));
    if (typeof meta.fill === "string" && meta.fill !== "") {
        dictionary.intern(meta.fill);
    }
    return dictionary;
}

/**
 * A MutableColumnParts record with every storage slot null, for producers that fill one slot.
 * @param meta - the column metadata
 * @param length - the row count
 * @param validity - the validity bitmap, or null
 * @returns the parts
 */
export function emptyParts(meta: ColumnMeta, length: number, validity: U32 | null): MutableColumnParts {
    return {
        meta,
        length,
        data: null,
        validity,
        nullCount: validity === null ? 0 : length,
        dictionary: null,
        offsets: null,
        utf8: null,
        strings: null,
        child: null,
        values: null,
    };
}

function childMeta(meta: ColumnMeta): ColumnMeta {
    if (meta.itemDtype === null || meta.itemComponents === null) {
        throw new GraphFormatError("E_COLUMN_TYPE", `column "${meta.name}" is not a list`, {
            column: meta.name,
            field: "itemDtype",
        });
    }
    return resolveColumnMeta(`${meta.name}.item`, meta.domain, {
        dtype: meta.itemDtype,
        components: meta.itemComponents,
        nullable: false,
        refersTo: meta.refersTo ?? undefined,
    });
}

/**
 * A column of `length` rows with no value set: every row unset (when nullable) and the data holding
 * the fill (design section 5.3). This is what a declared column looks like before any value is
 * written, and what remap uses for rows no source row maps to.
 * @param domain - the table the column belongs to
 * @param length - the number of rows
 * @param decl - the declaration; dtype required
 * @returns the empty column
 */
export function createEmptyColumn(domain: ColumnDomain, length: number, decl: ColumnDeclPatch): Column {
    const name = decl.name ?? "";
    const meta = resolveColumnMeta(name, domain, decl);
    return emptyColumnOf(meta, length);
}

/**
 * An empty column from resolved metadata; see createEmptyColumn.
 * @param meta - the resolved metadata
 * @param length - the number of rows
 * @returns the empty column
 */
function emptyColumnOf(meta: ColumnMeta, length: number): Column {
    const validity = meta.nullable ? new Uint32Array(bitmapWordCount(length)) : null;
    const parts = emptyParts(meta, length, validity);
    const { dtype, components, fill } = meta;
    switch (dtype) {
        case "f32":
            parts.data = new Float32Array(length * components).fill(fill as number);
            break;
        case "f64":
            parts.data = new Float64Array(length * components).fill(fill as number);
            break;
        case "i32":
            parts.data = new Int32Array(length * components).fill(fill as number);
            break;
        case "u32":
            parts.data = new Uint32Array(length * components).fill(fill as number);
            break;
        case "u8":
            parts.data = allocU8(length * components).fill(fill as number);
            break;
        case "bool":
            parts.data = makeBitmap(length, fill === true);
            break;
        case "dict": {
            const dictionary = seedDictionary(meta);
            parts.data = new Uint32Array(length).fill(fillCode(dictionary, fill, validity === null && length > 0));
            parts.dictionary = dictionary.values;
            break;
        }
        case "string": {
            const fillText = typeof fill === "string" ? fill : "";
            parts.strings = new Array<string>(length).fill(fillText);
            break;
        }
        case "list":
            parts.offsets = new Uint32Array(length + 1);
            parts.child = emptyColumnOf(childMeta(meta), 0);
            break;
        case "json":
            parts.values = new Array<unknown>(length).fill(undefined);
            break;
        default: {
            const dtypeName: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${dtypeName}`, { dtype: dtypeName });
        }
    }
    return createColumn(parts);
}

/**
 * Whether a JS array entry is an unset cell: undefined for every dtype, null too except for json,
 * where null is a value (design section 12.1).
 * @param value - the entry
 * @param dtype - the column dtype
 * @returns true when unset
 */
function isUnsetEntry(value: unknown, dtype: Dtype): boolean {
    return value === undefined || (value === null && dtype !== "json");
}

/**
 * Build the offsets and child of a list column from an array of rows (each an array of items).
 * @param meta - the list column's metadata
 * @param rows - one array per row (an empty array for unset rows)
 * @returns the offsets and the wrapped child
 */
function listPartsFromValues(
    meta: ColumnMeta,
    rows: readonly (readonly unknown[])[],
): { offsets: U32; child: ChildColumn } {
    const offsets = new Uint32Array(rows.length + 1);
    const items: unknown[] = [];
    for (let row = 0; row < rows.length; row++) {
        for (const item of rows[row]) {
            items.push(item);
        }
        offsets[row + 1] = items.length;
    }
    const child = columnOfValues(childMeta(meta), items) as ChildColumn;
    return { offsets, child };
}

/**
 * A column from a JS array of values under resolved metadata (one entry per row; undefined, and null
 * except for json, is an unset row). Values are coerced per design section 5.1 (a boolean into a
 * number column becomes 1 / 0, a number or boolean into a string column its canonical text) and
 * rejected with E_COLUMN_TYPE otherwise.
 * @param meta - the resolved metadata
 * @param values - the entries, `length` of them
 * @returns the column
 */
export function columnOfValues(meta: ColumnMeta, values: readonly unknown[]): Column {
    const { length } = values;
    const { dtype, components, fill, name } = meta;
    let validity: U32 | null = null;
    let nullCount = 0;
    for (let row = 0; row < length; row++) {
        if (isUnsetEntry(values[row], dtype)) {
            if (!meta.nullable) {
                throw new GraphFormatError("E_COLUMN_TYPE", `row ${row} of non-nullable column "${name}" is unset`, {
                    column: name,
                    row,
                });
            }
            validity ??= makeBitmap(length, true);
            bitmapClear(validity, row);
            nullCount++;
        }
    }
    const parts = emptyParts(meta, length, validity);
    parts.nullCount = nullCount;
    const isSet = (row: number): boolean => validity === null || bitmapGet(validity, row);
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            const data = allocNumeric(dtype, length * components);
            if (fill !== 0) {
                data.fill(fill as number);
            }
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    continue;
                }
                const raw = values[row];
                const value = typeof raw === "boolean" ? Number(raw) : raw;
                writeNumeric(dtype, data, row * components, components, value, name, row);
            }
            parts.data = data;
            break;
        }
        case "bool": {
            const data = makeBitmap(length, fill === true);
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    continue;
                }
                const value = coerceValue(values[row], "bool");
                if (value === true) {
                    bitmapSet(data, row);
                } else {
                    bitmapClear(data, row);
                }
            }
            parts.data = data;
            break;
        }
        case "dict": {
            const dictionary = seedDictionary(meta);
            const codes = new Uint32Array(length);
            if (validity !== null) {
                codes.fill(fillCode(dictionary, fill, false));
            }
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    continue;
                }
                const value = coerceValue(values[row], "string");
                if (typeof value !== "string") {
                    throw new GraphFormatError("E_COLUMN_TYPE", `row ${row} of dict column "${name}" is not a string`, {
                        column: name,
                        row,
                    });
                }
                codes[row] = dictionary.intern(value);
            }
            parts.data = codes;
            parts.dictionary = dictionary.values;
            break;
        }
        case "string": {
            const fillText = typeof fill === "string" ? fill : "";
            const strings = new Array<string>(length);
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    strings[row] = fillText;
                    continue;
                }
                const value = coerceValue(values[row], "string");
                if (typeof value !== "string") {
                    throw new GraphFormatError(
                        "E_COLUMN_TYPE",
                        `row ${row} of string column "${name}" is not a string`,
                        { column: name, row },
                    );
                }
                assertWellFormedString(value, { column: name, row });
                strings[row] = value;
            }
            parts.strings = strings;
            break;
        }
        case "list": {
            const rows = new Array<readonly unknown[]>(length);
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    rows[row] = [];
                    continue;
                }
                const value = values[row];
                if (!Array.isArray(value)) {
                    throw new GraphFormatError("E_COLUMN_TYPE", `row ${row} of list column "${name}" is not an array`, {
                        column: name,
                        row,
                    });
                }
                rows[row] = value;
            }
            const listParts = listPartsFromValues(meta, rows);
            parts.offsets = listParts.offsets;
            parts.child = listParts.child;
            break;
        }
        case "json": {
            const out = new Array<unknown>(length);
            for (let row = 0; row < length; row++) {
                if (!isSet(row)) {
                    out[row] = undefined;
                    continue;
                }
                assertJsonValue(values[row], `row ${row}`);
                out[row] = values[row];
            }
            parts.values = out;
            break;
        }
        default: {
            const dtypeName: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${dtypeName}`, { dtype: dtypeName });
        }
    }
    return createColumn(parts);
}

/**
 * Allocate a numeric buffer of a dtype (a padded store for u8).
 * @param dtype - the numeric dtype
 * @param length - the element count
 * @returns the zeroed array
 */
export function allocNumeric(dtype: NumericDtype, length: number): TypedArrayData {
    switch (dtype) {
        case "f32":
            return new Float32Array(length);
        case "f64":
            return new Float64Array(length);
        case "i32":
            return new Int32Array(length);
        case "u32":
            return new Uint32Array(length);
        case "u8":
            return allocU8(length);
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_COLUMN_TYPE", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Infer the declaration of a column from a JS array of values when the caller gave no dtype: the
 * widening rules of design section 5.1 over the entries (json when every entry is unset), and for a
 * list the same rules over the items.
 * @param values - the entries
 * @returns the dtype and, for a list, the item dtype
 */
function inferDeclFromValues(values: readonly unknown[]): { dtype: Dtype; itemDtype: ScalarDtype | undefined } {
    let sawArray = false;
    let sawOther = false;
    for (const value of values) {
        if (value === undefined || value === null) {
            continue;
        }
        if (Array.isArray(value)) {
            sawArray = true;
        } else {
            sawOther = true;
        }
    }
    if (sawArray && !sawOther) {
        const items: unknown[] = [];
        for (const value of values) {
            if (Array.isArray(value)) {
                for (const item of value) {
                    items.push(item);
                }
            }
        }
        return { dtype: "list", itemDtype: inferValuesDtype(items) ?? "json" };
    }
    return { dtype: inferValuesDtype(values) ?? "json", itemDtype: undefined };
}

/**
 * A column from a JS array of values with a declaration patch (the set() path for string / list /
 * json and for inferred dtypes). The dtype is inferred when the patch has none; nullable defaults to
 * true.
 * @param domain - the table the column belongs to
 * @param length - the table's row count; E_COLUMN_LENGTH when values.length differs
 * @param name - the column name
 * @param values - the entries, one per row
 * @param decl - the declaration patch
 * @returns the column
 */
export function columnFromValues(
    domain: ColumnDomain,
    length: number,
    name: string,
    values: readonly unknown[],
    decl: ColumnDeclPatch,
): Column {
    if (values.length !== length) {
        throw new GraphFormatError("E_COLUMN_LENGTH", `column "${name}": ${values.length} values for ${length} rows`, {
            column: name,
            expected: length,
            found: values.length,
        });
    }
    let patch: ColumnDeclPatch = { nullable: true, ...decl };
    if (patch.dtype === undefined) {
        const inferred = inferDeclFromValues(values);
        patch = { ...patch, dtype: inferred.dtype };
        if (inferred.dtype === "list" && patch.itemDtype === undefined) {
            patch = { ...patch, itemDtype: inferred.itemDtype };
        }
    }
    const meta = resolveColumnMeta(name, domain, patch);
    return columnOfValues(meta, values);
}

/**
 * The dtype a typed array class implies when the caller gave none.
 * @param data - the array
 * @returns f32 / f64 / i32 / u32 / u8
 */
export function dtypeOfArray(data: TypedArrayData): NumericDtype {
    if (data instanceof Float32Array) {
        return "f32";
    }
    if (data instanceof Float64Array) {
        return "f64";
    }
    if (data instanceof Int32Array) {
        return "i32";
    }
    if (data instanceof Uint32Array) {
        return "u32";
    }
    return "u8";
}

/**
 * A column adopting a typed array by reference (design section 5.7): the dtype comes from the array
 * class unless the patch names one (bool and dict over a Uint32Array); the length must be rowCount *
 * components (bool: ceil(rowCount / 32)) or E_COLUMN_LENGTH; a u8 array from which no padded u32 view
 * is constructible is copied, or refused with E_COLUMN_ALIGNMENT under adopt "strict". nullable
 * defaults to false (every row is set).
 * @param domain - the table the column belongs to
 * @param length - the table's row count
 * @param name - the column name
 * @param data - the typed array
 * @param decl - the declaration patch
 * @param adopt - "copy" (default) or "strict"
 * @returns the column (`column.data !== data` only when a u8 array was copied)
 */
export function columnFromTypedArray(
    domain: ColumnDomain,
    length: number,
    name: string,
    data: TypedArrayData,
    decl: ColumnDeclPatch,
    adopt: "copy" | "strict" = "copy",
): Column {
    const arrayDtype = dtypeOfArray(data);
    const dtype = decl.dtype ?? arrayDtype;
    const compatible = dtype === arrayDtype || (arrayDtype === "u32" && (dtype === "bool" || dtype === "dict"));
    if (!compatible) {
        throw new GraphFormatError(
            "E_COLUMN_TYPE",
            `column "${name}": a ${arrayDtype} array cannot back a ${dtype} column`,
            {
                column: name,
                field: "dtype",
                found: arrayDtype,
                expected: dtype,
            },
        );
    }
    const meta = resolveColumnMeta(name, domain, { nullable: false, ...decl, dtype });
    const parts = emptyParts(meta, length, null);
    let expected: number;
    switch (dtype) {
        case "bool":
            expected = bitmapWordCount(length);
            break;
        case "dict":
            expected = length;
            break;
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            expected = length * meta.components;
            break;
        default: {
            const dtypeName: string = dtype;
            throw new GraphFormatError(
                "E_COLUMN_TYPE",
                `column "${name}": a ${dtypeName} column cannot adopt a typed array`,
                {
                    column: name,
                    field: "dtype",
                },
            );
        }
    }
    if (data.length !== expected) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `column "${name}": array length ${data.length}, expected ${expected}`,
            {
                column: name,
                expected,
                found: data.length,
            },
        );
    }
    if (!isOverPlainBuffer(data)) {
        // decision D-SAB: no SharedArrayBuffer in v1; a resizable buffer's view can change length (I17)
        const reason = data.buffer instanceof ArrayBuffer ? "resizable ArrayBuffer" : "SharedArrayBuffer";
        throw new GraphFormatError("E_UNSUPPORTED", `column "${name}": a view over a ${reason} cannot be adopted`, {
            column: name,
            reason,
        });
    }
    let stored = data;
    if (data instanceof Uint8Array && !canViewAsPaddedU32(data)) {
        if (adopt === "strict") {
            throw new GraphFormatError(
                "E_COLUMN_ALIGNMENT",
                `column "${name}": no zero-copy padded u32 view over the u8 array`,
                {
                    column: name,
                    byteOffset: data.byteOffset,
                    byteLength: data.byteLength,
                    bufferByteLength: data.buffer.byteLength,
                },
            );
        }
        const copy = allocU8(data.length);
        copy.set(data);
        stored = copy;
    }
    parts.data = stored;
    if (dtype === "dict") {
        const dictionary = seedDictionary(meta);
        for (let row = 0; row < length; row++) {
            const code = data[row];
            if (code >= dictionary.size) {
                throw new GraphFormatError(
                    "E_COLUMN_TYPE",
                    `column "${name}": code ${code} at row ${row} is outside the ${dictionary.size} declared options`,
                    { column: name, row, code },
                );
            }
        }
        parts.dictionary = dictionary.values;
    }
    return createColumn(parts);
}
