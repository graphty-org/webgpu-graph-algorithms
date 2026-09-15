/**
 * The wire decoder (design sections 4.5, 5.9, 9.1, 9.5 and 11.3): `fromWire()` rebuilds a snapshot
 * from a WireSnapshot, and `decodeManifest()` is the shared reader the GSNP container functions of
 * bytes.ts drive with their own buffer regions.
 *
 * Rules (design section 9.1): a manifest whose `formatVersion` is not the reader's or whose wire
 * major is unknown is refused with E_UNSUPPORTED_VERSION; a newer wire minor is read by ignoring
 * unknown manifest fields; an unknown column dtype is E_UNSUPPORTED unless `unknownColumns: "skip"`
 * drops the column and names it in `meta.extra["graphty.skippedColumns"]`; an unknown id-map kind
 * is always E_UNSUPPORTED; unknown view entries are ignored and recomputed. Every WireBufferRef is
 * checked against its buffer before a typed array is built over it (E_BAD_SERIALIZATION with
 * details.ref), whatever the validation level, because a RangeError from a typed-array constructor
 * is never an acceptable failure mode; the "structure" and "full" levels add the checks of design
 * section 9.5 and then run the snapshot's own validate().
 *
 * Buffers are adopted by reference by default: the arrays are views into the caller's buffers and
 * the arena descriptor is honoured. A SharedArrayBuffer is copied (decision D-SAB), and `copy: true`
 * copies every buffer before adoption.
 */

import { CORE_ORDER } from "../builder/arena.js";
import { createColumn, emptyParts, isPlainObject, resolveColumnMeta } from "../columns/column.js";
import { AttributeTable } from "../columns/table.js";
import { ALIGNMENT, FORMAT_VERSION, MAX_COUNT, WIRE_FORMAT, WIRE_MAJOR } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { type NodeIdMap, nodeIdMapFromTyped, type NodeIdMapTypedParts } from "../ids/node-id-map.js";
import { checkUtf8Layout, decodeUtf8Rows, Utf8Store } from "../ids/string-store.js";
import { createSnapshot, type GraphSnapshot, seedView } from "../snapshot/graph-snapshot.js";
import { cooViewOf, edgeListViewOf, ReverseAdjacency, rowLengths } from "../snapshot/views.js";
import {
    type ArenaLayout,
    type ArenaSegment,
    type Column,
    type ColumnDeclPatch,
    type ColumnDomain,
    type ColumnMeta,
    type ColumnOrigin,
    type ColumnOriginInput,
    type CoreArrayName,
    type Dtype,
    type F32,
    type F64,
    type FromWireOptions,
    type GraphMeta,
    type NodeIdMapKind,
    type ScalarDtype,
    type SnapshotFlags,
    type TypedArrayData,
    type U8,
    type U32,
    type ValidationLevel,
    type WireBufferRef,
    type WireDtype,
    type WireSnapshot,
} from "../types/index.js";
import { type MutableColumnParts, type SnapshotParts } from "../types/internal.js";
import { canViewAsPaddedU32, copyToPaddedStore, padTo4 } from "../util/typed-array.js";
import {
    checkCooSrc,
    checkDegree,
    checkDegreeOrder,
    checkEdgeList,
    checkF64View,
    checkInDegree,
    checkMate,
    checkOutDegree,
    checkReverse,
    checkSelfLoopArcs,
    checkSelfLoopsPerNode,
    CoreFacts,
} from "./carried-views.js";
import { defineJsonKey } from "./to-wire.js";

// ============================================================ vocabularies

const WIRE_DTYPES: ReadonlySet<string> = new Set(["u32", "i32", "f32", "f64", "u8", "utf8"]);
const COLUMN_DTYPES: ReadonlySet<string> = new Set([
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
const ID_MAP_KINDS: ReadonlySet<string> = new Set(["identity", "dense", "numeric", "string", "mixed"]);
const ID_TYPES: ReadonlySet<GraphMeta["idType"] & string> = new Set(["string", "integer", "mixed"] as const);
const TIME_FORMATS: ReadonlySet<GraphMeta["timeFormat"] & string> = new Set([
    "integer",
    "double",
    "date",
    "dateTime",
] as const);
const TIME_REPRESENTATIONS: ReadonlySet<GraphMeta["timeRepresentation"] & string> = new Set([
    "interval",
    "timestamp",
] as const);
const MODES: ReadonlySet<GraphMeta["mode"] & string> = new Set(["static", "dynamic", "slice"] as const);
const FORBIDDEN_KEYS: ReadonlySet<string> = new Set(["__proto__", "constructor", "prototype"]);
const NUM_TAG = "$num";
const ESC_TAG = "$esc";
/** The deepest JSON nesting a manifest value may have; deeper input is refused instead of overflowing the stack. */
const MAX_JSON_DEPTH = 256;

/**
 * A short plain-ASCII rendering of an untrusted manifest value for an error message: strings are
 * quoted, everything else is named by type (a bigint or a null-prototype object cannot be
 * stringified, and a structuredClone-delivered manifest can carry both).
 * @param value - the value
 * @returns the rendering
 */
function describe(value: unknown): string {
    if (typeof value === "string") {
        return JSON.stringify(value.length > 40 ? `${value.slice(0, 37)}...` : value);
    }
    if (typeof value === "number" || typeof value === "boolean" || value === null) {
        return String(value);
    }
    return value === undefined ? "undefined" : `a ${typeof value}`;
}

/** The meta.extra key naming the columns dropped under unknownColumns: "skip" (design section 9.1). */
export const SKIPPED_COLUMNS_KEY = "graphty.skippedColumns";

// ============================================================ errors and shape readers

/**
 * The E_BAD_SERIALIZATION error of a malformed manifest member or buffer reference.
 * @param ref - the manifest path, e.g. "nodeColumns[2].data"
 * @param reason - what is wrong
 * @param details - extra context
 * @returns the error
 */
export function badWire(
    ref: string,
    reason: string,
    details: Readonly<Record<string, unknown>> = {},
): GraphFormatError {
    return new GraphFormatError("E_BAD_SERIALIZATION", `${ref}: ${reason}`, { ref, reason, ...details });
}

function asObject(value: unknown, path: string): Record<string, unknown> {
    if (!isPlainObject(value)) {
        throw badWire(path, "expected an object");
    }
    return value;
}

function asArray(value: unknown, path: string): readonly unknown[] {
    if (!Array.isArray(value)) {
        throw badWire(path, "expected an array");
    }
    return value as readonly unknown[];
}

function asString(value: unknown, path: string): string {
    if (typeof value !== "string") {
        throw badWire(path, "expected a string");
    }
    return value;
}

function asStringOrNull(value: unknown, path: string): string | null {
    if (value === undefined || value === null) {
        return null;
    }
    return asString(value, path);
}

function asBoolean(value: unknown, path: string): boolean {
    if (typeof value !== "boolean") {
        throw badWire(path, "expected a boolean");
    }
    return value;
}

function asBooleanOrNull(value: unknown, path: string): boolean | null {
    if (value === undefined || value === null) {
        return null;
    }
    return asBoolean(value, path);
}

function asCount(value: unknown, path: string): number {
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > MAX_COUNT) {
        throw badWire(path, `expected an integer in [0, ${MAX_COUNT}]`, { found: value });
    }
    // JSON.parse("-0") is -0; a count is stored as 0 (design section 4.1)
    return value === 0 ? 0 : value;
}

function asEnum<T extends string>(value: unknown, path: string, allowed: ReadonlySet<T>): T | null {
    if (value === undefined || value === null) {
        return null;
    }
    if (typeof value !== "string" || !(allowed as ReadonlySet<string>).has(value)) {
        throw badWire(path, `unexpected value ${describe(value)}`, { found: value });
    }
    return value as T;
}

/**
 * Whether a string names a column dtype this reader knows.
 * @param value - the dtype text
 * @returns true for one of the ten dtypes
 */
function isKnownDtype(value: unknown): value is Dtype {
    return typeof value === "string" && COLUMN_DTYPES.has(value);
}

// ============================================================ JSON values (5.9)

/**
 * Decode a manifest JSON value (design section 5.9): `{ "$num": "Infinity" | "-Infinity" | "NaN" |
 * "-0" }` becomes the number, `{ "$esc": { ... } }` unwraps a user object whose only key collided
 * with a tag (the writer's escape), arrays and plain objects are rebuilt recursively (a key named
 * `__proto__`, `constructor` or `prototype` is refused), everything else is returned as is. Nesting
 * deeper than MAX_JSON_DEPTH is E_BAD_SERIALIZATION rather than a stack overflow.
 * @param value - the value from the manifest
 * @param path - the manifest path for error messages
 * @param depth - the current nesting depth
 * @returns the decoded value
 */
export function decodeJsonValue(value: unknown, path: string, depth = 0): unknown {
    if (depth > MAX_JSON_DEPTH) {
        throw badWire(path, `JSON nesting deeper than ${MAX_JSON_DEPTH}`, { reason: "nesting" });
    }
    if (Array.isArray(value)) {
        return value.map((item: unknown, i) => decodeJsonValue(item, `${path}[${i}]`, depth + 1));
    }
    if (!isPlainObject(value)) {
        return value;
    }
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === NUM_TAG) {
        const tag: unknown = value[NUM_TAG];
        switch (tag) {
            case "NaN":
                return Number.NaN;
            case "Infinity":
                return Infinity;
            case "-Infinity":
                return -Infinity;
            case "-0":
                return -0;
            default:
                throw badWire(path, `unknown number tag ${describe(tag)}`, { found: tag });
        }
    }
    if (keys.length === 1 && keys[0] === ESC_TAG) {
        const inner: unknown = value[ESC_TAG];
        if (!isPlainObject(inner)) {
            throw badWire(path, "an escaped object must hold a plain object", { found: typeof inner });
        }
        return decodeJsonObject(inner, path, depth);
    }
    return decodeJsonObject(value, path, depth);
}

/**
 * Rebuild a plain object's members (no tag interpretation of the object itself).
 * @param value - the plain object
 * @param path - the manifest path
 * @param depth - the nesting depth of the object
 * @returns the rebuilt object
 */
function decodeJsonObject(value: Record<string, unknown>, path: string, depth: number): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value)) {
        if (FORBIDDEN_KEYS.has(key)) {
            throw badWire(path, `forbidden key "${key}"`, { key });
        }
        defineJsonKey(out, key, decodeJsonValue(value[key], `${path}.${key}`, depth + 1));
    }
    return out;
}

/**
 * The JSON.parse reviver of design section 9.5: refuses the prototype-pollution keys.
 * @param key - the property name being revived
 * @param value - the revived value
 * @returns the value unchanged
 */
function guardReviver(this: unknown, key: string, value: unknown): unknown {
    if (FORBIDDEN_KEYS.has(key)) {
        throw badWire("manifest", `forbidden key "${key}"`, { key });
    }
    return value;
}

/**
 * Parse JSON text from a container (the manifest or one json column row) with the guarding
 * reviver. A syntax error is E_BAD_SERIALIZATION.
 * @param text - the JSON text
 * @param path - the manifest path for error messages
 * @returns the parsed value
 */
export function parseGuardedJson(text: string, path: string): unknown {
    try {
        return JSON.parse(text, guardReviver) as unknown;
    } catch (err) {
        if (err instanceof GraphFormatError) {
            throw err;
        }
        throw badWire(path, `not valid JSON (${err instanceof Error ? err.message : String(err)})`);
    }
}

// ============================================================ options and regions

/** The options of fromWire / fromBytes after defaults were applied. */
interface ResolvedFromWireOptions {
    /** The validation level. */
    readonly level: ValidationLevel;
    /** Whether buffers are copied before adoption. */
    readonly copy: boolean;
    /** What to do with an unknown column dtype. */
    readonly unknownColumns: "error" | "skip";
}

/**
 * Apply defaults to FromWireOptions and reject values outside their documented sets (E_UNSUPPORTED).
 * @param options - the caller's options
 * @param defaultLevel - "structure" for fromWire, "full" for fromBytes
 * @returns the resolved options
 */
export function resolveFromWireOptions(
    options: FromWireOptions | undefined,
    defaultLevel: ValidationLevel,
): ResolvedFromWireOptions {
    const level = options?.validate ?? defaultLevel;
    if (level !== "none" && level !== "structure" && level !== "full") {
        throw new GraphFormatError("E_UNSUPPORTED", `unknown validation level ${String(level)}`, {
            option: "validate",
            found: level,
        });
    }
    const unknownColumns = options?.unknownColumns ?? "error";
    if (unknownColumns !== "error" && unknownColumns !== "skip") {
        throw new GraphFormatError("E_UNSUPPORTED", `unknown unknownColumns policy ${String(unknownColumns)}`, {
            option: "unknownColumns",
            found: unknownColumns,
        });
    }
    const copy = options?.copy ?? false;
    if (typeof copy !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", "copy must be a boolean", { option: "copy", found: copy });
    }
    return { level, copy, unknownColumns };
}

/** Where a located byte range lives. */
export interface LocatedRange {
    /** The plain ArrayBuffer holding the bytes. */
    readonly buffer: ArrayBuffer;
    /** The absolute byte offset of the range inside `buffer`. */
    readonly byteOffset: number;
}

/**
 * One buffer index of a manifest as the decoder sees it: a byte length for range checks, a locator
 * that finds a range contiguously in one plain buffer (null when it spans chunks or lives in a
 * SharedArrayBuffer) and a reader that copies a range out.
 */
export interface WireRegion {
    /** The byte length every reference is checked against. */
    readonly byteLength: number;
    /**
     * Find a byte range in one plain buffer.
     * @param byteOffset - region-relative start
     * @param byteLength - length
     * @returns the buffer and absolute offset, or null when the range cannot be adopted
     */
    locate(byteOffset: number, byteLength: number): LocatedRange | null;
    /**
     * Copy a byte range into a fresh buffer.
     * @param byteOffset - region-relative start
     * @param byteLength - length
     * @returns a Uint8Array at offset 0 of a fresh ArrayBuffer of at least byteLength bytes
     */
    read(byteOffset: number, byteLength: number): U8;
}

/**
 * A region over one plain ArrayBuffer.
 * @param buffer - the buffer
 * @param base - the absolute byte offset of the region's start inside the buffer
 * @param byteLength - the region's byte length
 * @returns the region
 */
export function bufferRegion(buffer: ArrayBuffer, base: number, byteLength: number): WireRegion {
    return {
        byteLength,
        locate(byteOffset: number): LocatedRange {
            return { buffer, byteOffset: base + byteOffset };
        },
        read(byteOffset: number, length: number): U8 {
            const out = new Uint8Array(new ArrayBuffer(padTo4(length)), 0, length);
            out.set(new Uint8Array(buffer, base + byteOffset, length));
            return out;
        },
    };
}

/**
 * Copy an ArrayBuffer or SharedArrayBuffer into a fresh ArrayBuffer.
 * @param buffer - the source
 * @returns the copy
 */
function copyBuffer(buffer: ArrayBufferLike): ArrayBuffer {
    const out = new ArrayBuffer(buffer.byteLength);
    new Uint8Array(out).set(new Uint8Array(buffer));
    return out;
}

/**
 * Whether a value is a SharedArrayBuffer (never adopted, decision D-SAB).
 * @param value - the value
 * @returns true for a SharedArrayBuffer
 */
export function isSharedArrayBuffer(value: unknown): value is SharedArrayBuffer {
    return typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer;
}

// ============================================================ the decoder

/** Who a placed byte range belongs to, for the overlap rule of design section 9.5. */
type RangeKind = "core" | "immutable" | "mutable" | "view";

/** An adopted byte range, keyed by the ArrayBuffer it lives in (absolute offsets), for the overlap rule. */
interface PlacedRange {
    readonly buffer: ArrayBuffer;
    readonly start: number;
    readonly end: number;
    readonly kind: RangeKind;
    readonly path: string;
}

interface SkippedColumn {
    readonly domain: ColumnDomain;
    readonly table: string | null;
    readonly name: string;
    readonly dtype: string;
}

/**
 * Bytes per element and constructor of a wire dtype.
 * @param dtype - the wire dtype
 * @returns the element size
 */
function wireElementSize(dtype: WireDtype): number {
    switch (dtype) {
        case "u32":
        case "i32":
        case "f32":
            return 4;
        case "f64":
            return 8;
        case "u8":
        case "utf8":
            return 1;
        default: {
            const name: string = dtype;
            throw badWire("dtype", `unknown wire dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Build the typed array of a wire dtype over a buffer.
 * @param dtype - the wire dtype
 * @param buffer - the buffer
 * @param byteOffset - the absolute start
 * @param length - the element count
 * @returns the array
 */
function buildArray(dtype: WireDtype, buffer: ArrayBuffer, byteOffset: number, length: number): TypedArrayData {
    switch (dtype) {
        case "u32":
            return new Uint32Array(buffer, byteOffset, length);
        case "i32":
            return new Int32Array(buffer, byteOffset, length);
        case "f32":
            return new Float32Array(buffer, byteOffset, length);
        case "f64":
            return new Float64Array(buffer, byteOffset, length);
        case "u8":
        case "utf8":
            return new Uint8Array(buffer, byteOffset, length);
        default: {
            const name: string = dtype;
            throw badWire("dtype", `unknown wire dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Read a WireBufferRef's shape from an untrusted manifest member.
 * @param value - the member
 * @param path - its manifest path
 * @returns the reference
 */
function readRef(value: unknown, path: string): WireBufferRef {
    const o = asObject(value, path);
    const { dtype } = o;
    if (typeof dtype !== "string" || !WIRE_DTYPES.has(dtype)) {
        throw badWire(path, `unknown wire dtype ${describe(dtype)}`, { found: dtype });
    }
    const fields = ["buffer", "byteOffset", "byteLength", "length"] as const;
    for (const field of fields) {
        const v = o[field];
        if (typeof v !== "number" || !Number.isInteger(v) || v < 0) {
            throw badWire(`${path}.${field}`, "expected a non-negative integer", { found: v });
        }
    }
    return {
        buffer: o.buffer as number,
        byteOffset: o.byteOffset as number,
        byteLength: o.byteLength as number,
        dtype: dtype as WireDtype,
        length: o.length as number,
    };
}

/**
 * The error to throw for a failure while building a column or table from untrusted input: a
 * column-construction error (E_COLUMN_*, and E_DUPLICATE_ROLE from the table constructor) becomes
 * E_BAD_SERIALIZATION naming the manifest path, every other Error passes through, and a non-Error
 * value is wrapped.
 * @param err - the caught value
 * @param path - the manifest path of the column or table
 * @returns the error to throw
 */
function asWireError(err: unknown, path: string): Error {
    if (err instanceof GraphFormatError && (err.code.startsWith("E_COLUMN") || err.code === "E_DUPLICATE_ROLE")) {
        return new GraphFormatError("E_BAD_SERIALIZATION", `${path}: ${err.message}`, {
            ref: path,
            reason: err.message,
            cause: err.code,
            ...err.details,
        });
    }
    if (err instanceof Error) {
        return err;
    }
    return badWire(path, String(err));
}

/**
 * The stateful reader of one manifest: resolves references against the regions, records placed
 * ranges for the overlap rule and the columns skipped under unknownColumns: "skip".
 */
class WireDecoder {
    private readonly regions: readonly WireRegion[];
    private readonly options: ResolvedFromWireOptions;
    private readonly ranges: PlacedRange[] = [];
    readonly skipped: SkippedColumn[] = [];

    /**
     * Create a decoder.
     * @param regions - one region per buffer index
     * @param options - the resolved options
     */
    constructor(regions: readonly WireRegion[], options: ResolvedFromWireOptions) {
        this.regions = regions;
        this.options = options;
    }

    /**
     * Whether the level includes the structure checks.
     * @returns true for "structure" and "full"
     */
    get structure(): boolean {
        return this.options.level !== "none";
    }

    /**
     * Whether the level is "full".
     * @returns true for "full"
     */
    get full(): boolean {
        return this.options.level === "full";
    }

    /**
     * Check a reference against its region (design section 9.5) and build the typed array over the
     * referenced bytes, adopting them when they are contiguous and aligned and copying otherwise.
     * @param value - the manifest member holding the reference
     * @param path - its manifest path
     * @param expected - the wire dtype the slot requires
     * @param kind - who the range belongs to
     * @returns the typed array
     */
    array(value: unknown, path: string, expected: WireDtype, kind: RangeKind): TypedArrayData {
        const ref = readRef(value, path);
        if (ref.dtype !== expected) {
            throw badWire(path, `expected dtype ${expected}, found ${ref.dtype}`, {
                expected,
                found: ref.dtype,
            });
        }
        const region = this.regions[ref.buffer] as WireRegion | undefined;
        if (region === undefined) {
            throw badWire(path, `buffer index ${ref.buffer} is out of range (${this.regions.length} buffers)`, {
                found: ref.buffer,
                buffers: this.regions.length,
            });
        }
        const elementSize = wireElementSize(ref.dtype);
        if (ref.byteOffset % elementSize !== 0) {
            throw badWire(path, `byteOffset ${ref.byteOffset} is not a multiple of ${elementSize}`, {
                byteOffset: ref.byteOffset,
                elementSize,
            });
        }
        if (ref.byteLength !== ref.length * elementSize) {
            throw badWire(path, `byteLength ${ref.byteLength} is not length ${ref.length} * ${elementSize}`, {
                byteLength: ref.byteLength,
                length: ref.length,
                elementSize,
            });
        }
        if (ref.byteOffset + ref.byteLength > region.byteLength) {
            throw badWire(path, `range [${ref.byteOffset}, ${ref.byteOffset + ref.byteLength}) exceeds the buffer`, {
                byteOffset: ref.byteOffset,
                byteLength: ref.byteLength,
                bufferByteLength: region.byteLength,
            });
        }
        const located = region.locate(ref.byteOffset, ref.byteLength);
        if (located !== null && located.byteOffset % elementSize === 0) {
            // adopted: the overlap rule is checked on the ArrayBuffer itself, so listing one buffer
            // twice in wire.buffers cannot hide an alias
            this.ranges.push({
                buffer: located.buffer,
                start: located.byteOffset,
                end: located.byteOffset + ref.byteLength,
                kind,
                path,
            });
            return buildArray(ref.dtype, located.buffer, located.byteOffset, ref.length);
        }
        // copied: a private buffer, nothing to overlap
        const copy = region.read(ref.byteOffset, ref.byteLength);
        return buildArray(ref.dtype, copy.buffer, 0, ref.length);
    }

    /**
     * A u32 array slot.
     * @param value - the manifest member
     * @param path - its path
     * @param kind - the range owner
     * @returns the array
     */
    u32(value: unknown, path: string, kind: RangeKind): U32 {
        return this.array(value, path, "u32", kind) as U32;
    }

    /**
     * A Utf8 store slot (offsets + utf8), its layout checked at the structure level and its bytes at
     * the full level.
     * @param value - the manifest member (a WireUtf8)
     * @param path - its path
     * @param rows - the row count the store must have, or null to take it from the offsets (a dictionary)
     * @param kind - the range owner
     * @param always - check the layout at every level (stores decoded eagerly)
     * @returns the offsets and bytes
     */
    utf8(
        value: unknown,
        path: string,
        rows: number | null,
        kind: RangeKind,
        always = false,
    ): { offsets: U32; utf8: U8 } {
        const o = asObject(value, path);
        const offsets = this.u32(o.offsets, `${path}.offsets`, kind);
        const utf8 = this.array(o.utf8, `${path}.utf8`, "utf8", kind) as U8;
        if (offsets.length === 0) {
            throw badWire(`${path}.offsets`, "a Utf8 store needs at least one offset");
        }
        if (this.structure || always) {
            checkUtf8Layout(offsets, utf8, rows ?? offsets.length - 1, path);
        }
        if (this.full) {
            const bad = Utf8Store.fromEncoded(offsets, utf8).firstMalformedRow();
            if (bad !== -1) {
                throw badWire(path, `malformed UTF-8 in row ${bad}`, { row: bad });
            }
        }
        return { offsets, utf8 };
    }

    /**
     * Enforce the overlap rule of design section 9.5: no byte overlap between a mutable column and
     * any core or immutable segment.
     */
    checkOverlaps(): void {
        const fixed = this.ranges.filter((r) => (r.kind === "core" || r.kind === "immutable") && r.end > r.start);
        for (const range of this.ranges) {
            if (range.kind !== "mutable" || range.end === range.start) {
                continue;
            }
            for (const other of fixed) {
                if (other.buffer === range.buffer && range.start < other.end && other.start < range.end) {
                    throw badWire(range.path, `mutable column overlaps ${other.path}`, { overlaps: other.path });
                }
            }
        }
    }

    // ---------------------------------------------------------------- columns

    /**
     * Decode one column (design section 9.1), or drop it under unknownColumns: "skip".
     * @param value - the WireColumn
     * @param domain - the table's domain
     * @param table - the extension table name, or null
     * @param rows - the row count
     * @param path - the manifest path
     * @returns the column, or null when skipped
     */
    column(value: unknown, domain: ColumnDomain, table: string | null, rows: number, path: string): Column | null {
        const w = asObject(value, path);
        const metaRaw = asObject(w.meta, `${path}.meta`);
        const name = asString(metaRaw.name, `${path}.meta.name`);
        const { dtype, itemDtype } = metaRaw;
        const unknown = !isKnownDtype(dtype) || (dtype === "list" && !isKnownDtype(itemDtype));
        if (unknown) {
            const found = dtype === "list" ? itemDtype : dtype;
            if (this.options.unknownColumns === "skip") {
                this.skipped.push({ domain, table, name, dtype: String(found) });
                return null;
            }
            throw new GraphFormatError("E_UNSUPPORTED", `column "${name}" has unknown dtype ${String(found)}`, {
                dtype: found,
                column: name,
                ref: path,
            });
        }
        const meta = this.columnMeta(metaRaw, name, domain, path);
        const kind: RangeKind = meta.mutable ? "mutable" : "immutable";
        const parts: MutableColumnParts = emptyParts(
            meta,
            rows,
            w.validity === null || w.validity === undefined ? null : this.u32(w.validity, `${path}.validity`, kind),
        );
        const declaredNullCount = w.nullCount;
        switch (meta.dtype) {
            case "f32":
            case "f64":
            case "i32":
            case "u32":
                parts.data = this.array(w.data, `${path}.data`, meta.dtype, kind);
                break;
            case "u8": {
                const data = this.array(w.data, `${path}.data`, "u8", kind) as U8;
                parts.data = canViewAsPaddedU32(data) ? data : copyToPaddedStore(data);
                break;
            }
            case "bool":
                parts.data = this.u32(w.data, `${path}.data`, kind);
                break;
            case "dict": {
                parts.data = this.u32(w.data, `${path}.data`, kind);
                const dict = this.utf8(w.dictionary, `${path}.dictionary`, null, kind, true);
                parts.dictionary = decodeUtf8Rows(dict.utf8, dict.offsets, 0, dict.offsets.length - 1);
                break;
            }
            case "string": {
                const store = this.utf8(w.strings, `${path}.strings`, rows, kind);
                parts.offsets = store.offsets;
                parts.utf8 = store.utf8;
                break;
            }
            case "list": {
                const offsets = this.u32(w.offsets, `${path}.offsets`, kind);
                if (offsets.length !== rows + 1) {
                    throw badWire(`${path}.offsets`, `expected ${rows + 1} offsets, found ${offsets.length}`, {
                        expected: rows + 1,
                        found: offsets.length,
                    });
                }
                parts.offsets = offsets;
                const child = this.column(w.child, domain, table, offsets[rows], `${path}.child`);
                if (child === null || child.dtype === "list") {
                    throw badWire(`${path}.child`, "a list column needs a non-list child column");
                }
                parts.child = child;
                break;
            }
            case "json":
                parts.values = this.jsonRows(w.jsonText, rows, `${path}.jsonText`, kind);
                break;
            default: {
                const dtypeName: string = meta.dtype;
                throw badWire(`${path}.meta.dtype`, `unknown dtype ${dtypeName}`, { dtype: dtypeName });
            }
        }
        let column: Column;
        try {
            column = createColumn(parts);
        } catch (err) {
            throw asWireError(err, path);
        }
        if (this.structure && column.nullCount !== declaredNullCount) {
            throw badWire(
                `${path}.nullCount`,
                `manifest says ${String(declaredNullCount)}, the bitmap says ${column.nullCount}`,
                {
                    expected: column.nullCount,
                    found: declaredNullCount,
                },
            );
        }
        return column;
    }

    /**
     * Resolve a column's metadata from its manifest form: the tagged JSON fields decoded, then the
     * declaration resolver of the column module applied so every rule of design section 5.5 holds.
     * @param raw - the manifest meta
     * @param name - the column name
     * @param domain - the table's domain
     * @param path - the column's manifest path
     * @returns the resolved metadata
     */
    private columnMeta(raw: Record<string, unknown>, name: string, domain: ColumnDomain, path: string): ColumnMeta {
        const metaPath = `${path}.meta`;
        const optional = <T>(key: string, check: (v: unknown, p: string) => T): T | undefined => {
            const v = raw[key];
            return v === undefined || v === null ? undefined : check(v, `${metaPath}.${key}`);
        };
        const numberOf = (v: unknown, p: string): number => {
            if (typeof v !== "number") {
                throw badWire(p, "expected a number", { found: v });
            }
            return v;
        };
        const decl: ColumnDeclPatch = {
            name,
            dtype: raw.dtype as Dtype,
            components: optional("components", numberOf),
            itemDtype: optional("itemDtype", (v) => v as ScalarDtype),
            itemComponents: optional("itemComponents", numberOf),
            nullable: optional("nullable", asBoolean),
            mutable: optional("mutable", asBoolean),
            role: optional("role", asString),
            refersTo: optional("refersTo", (v) => v as "node" | "edge"),
            unique: optional("unique", asBoolean),
            default: decodeJsonValue(raw.default, `${metaPath}.default`),
            fill: decodeJsonValue(raw.fill, `${metaPath}.fill`) as number | string | boolean | undefined,
            options: optional("options", (v, p) => decodeJsonValue(v, p) as readonly unknown[]),
            origin: optional("origin", (v) => v as ColumnOriginInput),
            dynamic: optional("dynamic", asBoolean),
            extra: optional("extra", (v, p) => decodeJsonValue(v, p) as Readonly<Record<string, unknown>>),
        };
        try {
            return resolveColumnMeta(name, domain, decl);
        } catch (err) {
            throw asWireError(err, path);
        }
    }

    /**
     * The values of a json column from its JSON text store: empty text is an unset row (undefined),
     * every other row is parsed with the guarding reviver and untagged.
     * @param value - the WireUtf8
     * @param rows - the row count
     * @param path - the manifest path
     * @param kind - the range owner
     * @returns one value per row
     */
    private jsonRows(value: unknown, rows: number, path: string, kind: RangeKind): unknown[] {
        const store = this.utf8(value, path, rows, kind, true);
        const texts = decodeUtf8Rows(store.utf8, store.offsets, 0, rows);
        const values = new Array<unknown>(rows);
        for (let row = 0; row < rows; row++) {
            const text = texts[row];
            values[row] =
                text === "" ? undefined : decodeJsonValue(parseGuardedJson(text, `${path}[${row}]`), `${path}[${row}]`);
        }
        return values;
    }

    /**
     * Decode a table's columns in declaration order.
     * @param value - the WireColumn array
     * @param domain - the domain
     * @param table - the extension table name, or null
     * @param rows - the row count
     * @param path - the manifest path
     * @returns the table
     */
    table(value: unknown, domain: ColumnDomain, table: string | null, rows: number, path: string): AttributeTable {
        const list = asArray(value, path);
        const columns: Column[] = [];
        const names = new Set<string>();
        for (let i = 0; i < list.length; i++) {
            const column = this.column(list[i], domain, table, rows, `${path}[${i}]`);
            if (column === null) {
                continue;
            }
            if (names.has(column.meta.name)) {
                throw badWire(`${path}[${i}].meta.name`, `column "${column.meta.name}" is declared twice`, {
                    column: column.meta.name,
                });
            }
            names.add(column.meta.name);
            columns.push(column);
        }
        try {
            return new AttributeTable({ domain, rowCount: rows, columns });
        } catch (err) {
            throw asWireError(err, path);
        }
    }

    // ---------------------------------------------------------------- id map

    /**
     * Decode the id map (design section 4.5). An unknown kind is E_UNSUPPORTED with details.kind.
     * @param value - the WireIdMap
     * @returns the map
     */
    ids(value: unknown): NodeIdMap {
        const o = asObject(value, "ids");
        const kindRaw = o.kind;
        if (typeof kindRaw !== "string" || !ID_MAP_KINDS.has(kindRaw)) {
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${describe(kindRaw)}`, {
                kind: kindRaw,
                ref: "ids.kind",
            });
        }
        const kind = kindRaw as NodeIdMapKind;
        const size = asCount(o.size, "ids.size");
        const offsetRaw = o.offset ?? 0;
        if (typeof offsetRaw !== "number") {
            throw badWire("ids.offset", "expected a number", { found: offsetRaw });
        }
        const parts: NodeIdMapTypedParts = {
            kind,
            size,
            offset: kind === "identity" ? offsetRaw : 0,
            values: null,
            tags: null,
            numbers: null,
            offsets: null,
            utf8: null,
        };
        const typed = parts as { -readonly [K in keyof NodeIdMapTypedParts]: NodeIdMapTypedParts[K] };
        switch (kind) {
            case "identity":
                break;
            case "dense":
                typed.values = this.u32(o.values, "ids.values", "immutable");
                break;
            case "numeric":
                typed.values = this.array(o.values, "ids.values", "f64", "immutable") as F64;
                break;
            case "string": {
                typed.offsets = this.u32(o.offsets, "ids.offsets", "immutable");
                typed.utf8 = this.array(o.utf8, "ids.utf8", "utf8", "immutable") as U8;
                break;
            }
            case "mixed": {
                typed.tags = this.array(o.tags, "ids.tags", "u8", "immutable") as U8;
                typed.numbers = this.array(o.numbers, "ids.numbers", "f64", "immutable") as F64;
                typed.offsets = this.u32(o.offsets, "ids.offsets", "immutable");
                typed.utf8 = this.array(o.utf8, "ids.utf8", "utf8", "immutable") as U8;
                break;
            }
            default: {
                const name: string = kind;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${name}`, { kind: name });
            }
        }
        return nodeIdMapFromTyped(typed, this.options.level);
    }

    // ---------------------------------------------------------------- views

    /**
     * Resolve the member arrays of the carried views (design section 9.1): every reference is
     * checked like any other, and the arrays are installed on the snapshot by `installViews()`.
     * @param value - the manifest views member
     * @returns the member arrays per view name (unknown names included; they are ignored later)
     */
    views(value: unknown): CarriedViews {
        const out: CarriedViews = new Map();
        if (value === null || value === undefined) {
            return out;
        }
        const views = asObject(value, "views");
        for (const name of Object.keys(views)) {
            const members = asObject(views[name], `views.${name}`);
            const arrays = new Map<string, TypedArrayData>();
            for (const member of Object.keys(members)) {
                const path = `views.${name}.${member}`;
                const ref = readRef(members[member], path);
                arrays.set(member, this.array(ref, path, ref.dtype, "view"));
            }
            out.set(name, arrays);
        }
        return out;
    }
}

// ============================================================ carried views

/** The member arrays of the views a manifest carried, keyed by view name then member name. */
type CarriedViews = Map<string, Map<string, TypedArrayData>>;

/**
 * One member array of a carried view, checked for class and length.
 * @param members - the view's member arrays
 * @param view - the view name (for the error path)
 * @param member - the member name
 * @param ctor - the required typed-array class
 * @param length - the required length
 * @param required - whether an absent member is an error
 * @returns the array, or null when absent and not required
 */
function viewMember<T extends TypedArrayData>(
    members: ReadonlyMap<string, TypedArrayData>,
    view: string,
    member: string,
    ctor: new (length: number) => T,
    length: number,
    required: boolean,
): T | null {
    const path = `views.${view}.${member}`;
    const array = members.get(member);
    if (array === undefined) {
        if (required) {
            throw badWire(path, "the carried view is missing this member");
        }
        return null;
    }
    if (!(array instanceof ctor)) {
        throw badWire(path, `expected ${ctor.name}, found ${array.constructor.name}`, {
            expected: ctor.name,
            found: array.constructor.name,
        });
    }
    if (array.length !== length) {
        throw badWire(path, `expected ${length} entries, found ${array.length}`, {
            expected: length,
            found: array.length,
        });
    }
    return array;
}

/**
 * The per-node or per-arc array of a single-member view.
 * @param members - the view's member arrays
 * @param view - the view name
 * @param ctor - the required typed-array class
 * @param length - the required length
 * @returns the array
 */
function dataMember<T extends TypedArrayData>(
    members: ReadonlyMap<string, TypedArrayData>,
    view: string,
    ctor: new (length: number) => T,
    length: number,
): T {
    return viewMember(members, view, "data", ctor, length, true) as T;
}

/**
 * Install the views a manifest carried into the snapshot's cache (design section 9.1), so the
 * receiver does not recompute them. Views that alias another view on an undirected snapshot
 * (`reverse`, `inDegree`, `weightedInDegree`, `reverseDegreeOrder`) are left to the snapshot, which
 * produces the alias itself; `mate` on a directed snapshot and unknown view names are ignored.
 * Every installed member must have the class and length of the view it claims to be
 * (E_BAD_SERIALIZATION otherwise); at the "structure" level (`verify`) every carried view is also
 * checked against the core before it is installed (src/wire/carried-views.ts), so a corrupt or
 * forged view never enters the cache below "full".
 * @param snapshot - the freshly built snapshot
 * @param carried - the carried member arrays
 * @param verify - whether to check the contents (the "structure" level)
 */
function installViews(snapshot: GraphSnapshot, carried: CarriedViews, verify: boolean): void {
    const { directed, nodeCount, edgeCount, arcCount, selfLoopCount } = snapshot;
    let facts: CoreFacts | null = null;
    const factsOf = (): CoreFacts => {
        facts ??= new CoreFacts(snapshot);
        return facts;
    };
    /**
     * Install an f64 view; under `verify` it is compared with the snapshot's own computation instead
     * (which then stays cached), so a corrupt value array never enters the cache.
     * @param name - the view
     */
    const seedF64 = (name: "weightedOutDegree" | "weightedInDegree" | "weightedDegree" | "selfLoopWeight"): void => {
        const data = dataMember(
            carried.get(name) as ReadonlyMap<string, TypedArrayData>,
            name,
            Float64Array,
            nodeCount,
        );
        if (verify) {
            checkF64View(name, data, snapshot[name]());
            return;
        }
        seedView(snapshot, name, data);
    };
    for (const [name, members] of carried) {
        switch (name) {
            case "reverse": {
                if (!directed) {
                    break;
                }
                const rowPtr = viewMember(members, name, "rowPtr", Uint32Array, nodeCount + 1, true) as U32;
                const colIdx = viewMember(members, name, "colIdx", Uint32Array, arcCount, true) as U32;
                const fwdArc = viewMember(members, name, "fwdArc", Uint32Array, arcCount, true) as U32;
                const weights = viewMember(members, name, "weights", Float32Array, arcCount, snapshot.weights !== null);
                if (weights !== null && snapshot.weights === null) {
                    throw badWire(`views.${name}.weights`, "weights carried for an unweighted snapshot");
                }
                const reverse = new ReverseAdjacency(snapshot, rowPtr, colIdx, weights, fwdArc);
                if (verify) {
                    checkReverse(snapshot, factsOf(), reverse);
                }
                seedView(snapshot, "reverse", reverse);
                break;
            }
            case "coo": {
                const src = viewMember(members, name, "src", Uint32Array, arcCount, true) as U32;
                if (verify) {
                    checkCooSrc(factsOf(), src);
                }
                seedView(snapshot, "coo", cooViewOf(snapshot, src));
                break;
            }
            case "edgeList": {
                const identity = snapshot.flags.arcToEdgeIsIdentity;
                const src = viewMember(members, name, "src", Uint32Array, edgeCount, true) as U32;
                const dst = viewMember(members, name, "dst", Uint32Array, edgeCount, !identity);
                const weights = viewMember(
                    members,
                    name,
                    "weights",
                    Float32Array,
                    edgeCount,
                    snapshot.weights !== null && !identity,
                );
                if (weights !== null && snapshot.weights === null) {
                    throw badWire(`views.${name}.weights`, "weights carried for an unweighted snapshot");
                }
                if (verify) {
                    checkEdgeList(snapshot, factsOf(), src, dst, weights);
                }
                seedView(
                    snapshot,
                    "edgeList",
                    edgeListViewOf(snapshot, src, dst ?? snapshot.colIdx, weights ?? snapshot.weights),
                );
                break;
            }
            case "outDegree": {
                const data = dataMember(members, name, Uint32Array, nodeCount);
                if (verify) {
                    checkOutDegree(snapshot, data);
                }
                seedView(snapshot, name, data);
                break;
            }
            case "degree": {
                const data = dataMember(members, name, Uint32Array, nodeCount);
                if (verify) {
                    checkDegree(snapshot, factsOf(), data);
                }
                seedView(snapshot, name, data);
                break;
            }
            case "selfLoopsPerNode": {
                const data = dataMember(members, name, Uint32Array, nodeCount);
                if (verify) {
                    checkSelfLoopsPerNode(factsOf(), data);
                }
                seedView(snapshot, name, data);
                break;
            }
            case "inDegree":
                if (directed) {
                    const data = dataMember(members, name, Uint32Array, nodeCount);
                    if (verify) {
                        checkInDegree(factsOf(), data);
                    }
                    seedView(snapshot, name, data);
                }
                break;
            case "weightedOutDegree":
            case "weightedDegree":
            case "selfLoopWeight":
                seedF64(name);
                break;
            case "weightedInDegree":
                if (directed) {
                    seedF64(name);
                }
                break;
            case "selfLoopArcs": {
                const data = dataMember(members, name, Uint32Array, selfLoopCount);
                if (verify) {
                    checkSelfLoopArcs(snapshot, factsOf(), data);
                }
                seedView(snapshot, name, data);
                break;
            }
            case "mate":
                if (!directed) {
                    const data = dataMember(members, name, Uint32Array, arcCount);
                    if (verify) {
                        checkMate(snapshot, factsOf(), data);
                    }
                    seedView(snapshot, name, data);
                }
                break;
            case "degreeOrder":
            case "reverseDegreeOrder": {
                if (name === "reverseDegreeOrder" && !directed) {
                    break;
                }
                const perm = viewMember(members, name, "perm", Uint32Array, nodeCount, true) as U32;
                const segmentOffsets = viewMember(members, name, "segmentOffsets", Uint32Array, 5, true) as U32;
                if (verify) {
                    const degree = name === "degreeOrder" ? rowLengths(snapshot.rowPtr, nodeCount) : factsOf().inDegree;
                    checkDegreeOrder(name, degree, perm, segmentOffsets);
                }
                seedView(snapshot, name, Object.freeze({ perm, segmentOffsets }));
                break;
            }
            default:
                // An unknown view name (a newer minor) or a scalar view: ignored and recomputed.
                break;
        }
    }
}

// ============================================================ core, arena, meta

interface CoreRefs {
    readonly rowPtr: WireBufferRef;
    readonly colIdx: WireBufferRef | null;
    readonly weights: WireBufferRef | null;
    readonly arcToEdge: WireBufferRef | null;
    readonly edgeToArc: WireBufferRef | null;
}

/** The counts a manifest declares. */
interface WireCounts {
    readonly nodes: number;
    readonly edges: number;
    readonly arcs: number;
    readonly selfLoops: number;
}

interface CoreArrays {
    readonly rowPtr: U32;
    readonly colIdx: U32;
    readonly weights: F32 | null;
    readonly arcToEdge: U32 | null;
    readonly edgeToArc: U32 | null;
}

/**
 * Read the core references from the manifest, applying the null rules of design section 9.1.
 * @param value - manifest.core
 * @returns the references
 */
function readCoreRefs(value: unknown): CoreRefs {
    const core = asObject(value, "core");
    const optionalRef = (name: string): WireBufferRef | null => {
        const v = core[name];
        return v === null || v === undefined ? null : readRef(v, `core.${name}`);
    };
    return {
        rowPtr: readRef(core.rowPtr, "core.rowPtr"),
        colIdx: optionalRef("colIdx"),
        weights: optionalRef("weights"),
        arcToEdge: optionalRef("arcToEdge"),
        edgeToArc: optionalRef("edgeToArc"),
    };
}

/**
 * Build the core arrays: a null colIdx is a zero-arc graph, a null weights array an unweighted (or
 * zero-arc weighted) graph, a null permutation the identity (when the flag says so) or zero length.
 * @param decoder - the decoder
 * @param refs - the core references
 * @param counts - the manifest counts
 * @param flags - the manifest flags
 * @returns the arrays
 */
function decodeCore(decoder: WireDecoder, refs: CoreRefs, counts: WireCounts, flags: SnapshotFlags): CoreArrays {
    const rowPtr = decoder.u32(refs.rowPtr, "core.rowPtr", "core");
    let colIdx: U32;
    if (refs.colIdx === null) {
        if (counts.arcs !== 0) {
            throw badWire("core.colIdx", `null although counts.arcs is ${counts.arcs}`);
        }
        colIdx = new Uint32Array(0);
    } else {
        colIdx = decoder.u32(refs.colIdx, "core.colIdx", "core");
    }
    let weights: F32 | null = null;
    if (refs.weights !== null) {
        if (!flags.weighted) {
            throw badWire("core.weights", "present although flags.weighted is false");
        }
        weights = decoder.array(refs.weights, "core.weights", "f32", "core") as F32;
    } else if (flags.weighted) {
        if (counts.arcs !== 0) {
            throw badWire("core.weights", `null although flags.weighted is true and counts.arcs is ${counts.arcs}`);
        }
        weights = new Float32Array(0);
    }
    const identity = flags.arcToEdgeIsIdentity;
    const permutation = (ref: WireBufferRef | null, name: "arcToEdge" | "edgeToArc", count: number): U32 | null => {
        if (ref !== null) {
            if (identity) {
                throw badWire(`core.${name}`, "present although flags.arcToEdgeIsIdentity is true");
            }
            return decoder.u32(ref, `core.${name}`, "core");
        }
        if (identity) {
            return null;
        }
        if (count !== 0) {
            throw badWire(
                `core.${name}`,
                `null although the permutation is not the identity and the count is ${count}`,
            );
        }
        return new Uint32Array(0);
    };
    return {
        rowPtr,
        colIdx,
        weights,
        arcToEdge: permutation(refs.arcToEdge, "arcToEdge", counts.arcs),
        edgeToArc: permutation(refs.edgeToArc, "edgeToArc", counts.edges),
    };
}

/**
 * Honour the manifest's arena descriptor (design sections 9.1 and 10.3): the core arrays lying at
 * 256-aligned offsets inside the described range become its segments; hotByteLength is recomputed
 * and, at the structure level, compared with the manifest.
 * @param value - manifest.arena
 * @param regions - the buffer regions
 * @param refs - the core references
 * @param arrays - the core arrays
 * @param structure - whether the structure checks run
 * @returns the arena layout, or null when the manifest carries none or the region is chunked
 */
function decodeArena(
    value: unknown,
    regions: readonly WireRegion[],
    refs: CoreRefs,
    arrays: CoreArrays,
    structure: boolean,
): ArenaLayout | null {
    if (value === null || value === undefined) {
        return null;
    }
    const a = asObject(value, "arena");
    const buffer = asCount(a.buffer, "arena.buffer");
    const byteOffset = asCount(a.byteOffset, "arena.byteOffset");
    const byteLength = asCount(a.byteLength, "arena.byteLength");
    const region = regions[buffer] as WireRegion | undefined;
    if (region === undefined) {
        throw badWire("arena.buffer", `buffer index ${buffer} is out of range`, { found: buffer });
    }
    if (byteOffset + byteLength > region.byteLength) {
        throw badWire("arena", `range [${byteOffset}, ${byteOffset + byteLength}) exceeds the buffer`, {
            byteOffset,
            byteLength,
            bufferByteLength: region.byteLength,
        });
    }
    const located = region.locate(byteOffset, byteLength);
    if (located === null) {
        return null;
    }
    const segments: Record<CoreArrayName, ArenaSegment | null> = {
        rowPtr: null,
        colIdx: null,
        weights: null,
        arcToEdge: null,
        edgeToArc: null,
    };
    let hotByteLength = 0;
    const names: readonly CoreArrayName[] = CORE_ORDER;
    for (const name of names) {
        const ref = refs[name];
        const array = arrays[name];
        if (ref === null || array === null || array.length === 0) {
            continue;
        }
        const inside =
            ref.buffer === buffer &&
            ref.byteOffset >= byteOffset &&
            ref.byteOffset + ref.byteLength <= byteOffset + byteLength;
        if (!inside) {
            continue;
        }
        const relative = ref.byteOffset - byteOffset;
        if (relative % ALIGNMENT !== 0) {
            throw badWire(`core.${name}`, `inside the arena at offset ${relative}, not a multiple of ${ALIGNMENT}`, {
                byteOffset: relative,
            });
        }
        if (array.buffer !== located.buffer || array.byteOffset !== located.byteOffset + relative) {
            throw badWire(`core.${name}`, "the arena describes a segment the array is not a view of");
        }
        segments[name] = Object.freeze({ byteOffset: located.byteOffset + relative, byteLength: ref.byteLength });
        if (name === "rowPtr" || name === "colIdx" || name === "weights") {
            hotByteLength = relative + ref.byteLength;
        }
    }
    if (structure && a.hotByteLength !== hotByteLength) {
        throw badWire(
            "arena.hotByteLength",
            `manifest says ${String(a.hotByteLength)}, the segments say ${hotByteLength}`,
            {
                expected: hotByteLength,
                found: a.hotByteLength,
            },
        );
    }
    return Object.freeze({
        buffer: located.buffer,
        byteOffset: located.byteOffset,
        byteLength,
        alignment: ALIGNMENT,
        segments: Object.freeze(segments),
        hotByteLength,
    });
}

/**
 * Read the flags (every member a boolean).
 * @param value - manifest.flags
 * @returns the flags
 */
function readFlags(value: unknown): SnapshotFlags {
    const o = asObject(value, "flags");
    const flag = (name: keyof SnapshotFlags): boolean => asBoolean(o[name], `flags.${name}`);
    return {
        multigraph: flag("multigraph"),
        hasSelfLoops: flag("hasSelfLoops"),
        arcToEdgeIsIdentity: flag("arcToEdgeIsIdentity"),
        weighted: flag("weighted"),
        allWeightsOne: flag("allWeightsOne"),
        nonNegativeWeights: flag("nonNegativeWeights"),
        finiteWeights: flag("finiteWeights"),
    };
}

/**
 * Read a column origin (every field a string or null).
 * @param value - the manifest member
 * @param path - its path
 * @returns the origin, or null
 */
function readOrigin(value: unknown, path: string): ColumnOrigin | null {
    if (value === null || value === undefined) {
        return null;
    }
    const o = asObject(value, path);
    const field = (key: keyof ColumnOrigin): string | null => asStringOrNull(o[key], `${path}.${key}`);
    return {
        format: field("format"),
        id: field("id"),
        title: field("title"),
        type: field("type"),
        namespace: field("namespace"),
    };
}

/**
 * Read the graph metadata (design section 5.9); unknown fields are ignored, missing ones are null.
 * @param value - manifest.meta
 * @returns the frozen metadata
 */
function readGraphMeta(value: unknown): GraphMeta {
    const o = asObject(value, "meta");
    const keywordsRaw = o.keywords ?? [];
    const keywords = asArray(keywordsRaw, "meta.keywords").map((k, i) => asString(k, `meta.keywords[${i}]`));
    const extraRaw = decodeJsonValue(o.extra ?? {}, "meta.extra");
    if (!isPlainObject(extraRaw)) {
        throw badWire("meta.extra", "expected an object");
    }
    return Object.freeze({
        name: asStringOrNull(o.name, "meta.name"),
        description: asStringOrNull(o.description, "meta.description"),
        creator: asStringOrNull(o.creator, "meta.creator"),
        created: asStringOrNull(o.created, "meta.created"),
        modified: asStringOrNull(o.modified, "meta.modified"),
        keywords: Object.freeze(keywords),
        sourceFormat: asStringOrNull(o.sourceFormat, "meta.sourceFormat"),
        sourceVersion: asStringOrNull(o.sourceVersion, "meta.sourceVersion"),
        idType: asEnum(o.idType, "meta.idType", ID_TYPES),
        timeFormat: asEnum(o.timeFormat, "meta.timeFormat", TIME_FORMATS),
        timeRepresentation: asEnum(o.timeRepresentation, "meta.timeRepresentation", TIME_REPRESENTATIONS),
        mode: asEnum(o.mode, "meta.mode", MODES),
        declaredMultigraph: asBooleanOrNull(o.declaredMultigraph, "meta.declaredMultigraph"),
        weightOrigin: readOrigin(o.weightOrigin, "meta.weightOrigin"),
        extra: Object.freeze(extraRaw),
    });
}

/**
 * Check the manifest's discriminator and versions (design sections 9.1 and 13.5): `format`, the
 * wire major (E_UNSUPPORTED_VERSION kind "wire") and `formatVersion` (kind "format"). A newer wire
 * minor is accepted.
 * @param manifest - the manifest object
 * @returns the wire version pair
 */
export function checkManifestVersions(manifest: Record<string, unknown>): readonly [number, number] {
    if (manifest.format !== WIRE_FORMAT) {
        throw badWire("format", `expected "${WIRE_FORMAT}", found ${describe(manifest.format)}`, {
            found: manifest.format,
        });
    }
    const wire = asArray(manifest.wire, "wire");
    const major = wire[0];
    const minor = wire[1];
    if (
        typeof major !== "number" ||
        typeof minor !== "number" ||
        !Number.isInteger(major) ||
        !Number.isInteger(minor)
    ) {
        throw badWire("wire", "expected [major, minor] integers", { found: manifest.wire });
    }
    if (major !== WIRE_MAJOR) {
        throw new GraphFormatError(
            "E_UNSUPPORTED_VERSION",
            `wire major ${major} is not supported (reader: ${WIRE_MAJOR})`,
            {
                kind: "wire",
                found: major,
                supported: WIRE_MAJOR,
            },
        );
    }
    if (manifest.formatVersion !== FORMAT_VERSION) {
        throw new GraphFormatError(
            "E_UNSUPPORTED_VERSION",
            `formatVersion ${describe(manifest.formatVersion)} is not supported (reader: ${FORMAT_VERSION})`,
            { kind: "format", found: manifest.formatVersion, supported: FORMAT_VERSION },
        );
    }
    return [major, minor];
}

// ============================================================ the manifest reader

/**
 * Rebuild a snapshot from a manifest and the regions its references index (design sections 9.1 and
 * 9.5): versions checked, core arrays and arena adopted, id map and tables decoded, then the
 * snapshot validated at the requested level.
 * @param manifestRaw - the manifest (an object; already parsed for a container)
 * @param regions - one region per buffer index
 * @param options - the resolved options
 * @param honourArena - whether the manifest's arena descriptor is adopted (false for chunked input)
 * @returns the snapshot
 */
export function decodeManifest(
    manifestRaw: unknown,
    regions: readonly WireRegion[],
    options: ResolvedFromWireOptions,
    honourArena: boolean,
): GraphSnapshot {
    const manifest = asObject(manifestRaw, "manifest");
    checkManifestVersions(manifest);
    const decoder = new WireDecoder(regions, options);
    const directed = asBoolean(manifest.directed, "directed");
    const countsRaw = asObject(manifest.counts, "counts");
    const counts: WireCounts = {
        nodes: asCount(countsRaw.nodes, "counts.nodes"),
        edges: asCount(countsRaw.edges, "counts.edges"),
        arcs: asCount(countsRaw.arcs, "counts.arcs"),
        selfLoops: asCount(countsRaw.selfLoops, "counts.selfLoops"),
    };
    const flags = readFlags(manifest.flags);
    if (flags.arcToEdgeIsIdentity && (!directed || counts.arcs !== counts.edges)) {
        throw badWire("flags.arcToEdgeIsIdentity", "set on an undirected graph or with arcs !== edges");
    }
    const coreRefs = readCoreRefs(manifest.core);
    const core = decodeCore(decoder, coreRefs, counts, flags);
    const arena = honourArena ? decodeArena(manifest.arena, regions, coreRefs, core, decoder.structure) : null;
    const ids = decoder.ids(manifest.ids);
    const nodes = decoder.table(manifest.nodeColumns ?? [], "node", null, counts.nodes, "nodeColumns");
    const edges = decoder.table(manifest.edgeColumns ?? [], "edge", null, counts.edges, "edgeColumns");
    const graph = decoder.table(manifest.graphColumns ?? [], "graph", null, 1, "graphColumns");
    const extensions = new Map<string, AttributeTable>();
    const extensionsRaw = asArray(manifest.extensions ?? [], "extensions");
    for (let i = 0; i < extensionsRaw.length; i++) {
        const path = `extensions[${i}]`;
        const ext = asObject(extensionsRaw[i], path);
        const name = asString(ext.name, `${path}.name`);
        if (extensions.has(name)) {
            throw badWire(`${path}.name`, `extension table "${name}" appears twice`, { table: name });
        }
        const rowCount = asCount(ext.rowCount, `${path}.rowCount`);
        extensions.set(name, decoder.table(ext.columns, "extension", name, rowCount, `${path}.columns`));
    }
    let meta = readGraphMeta(manifest.meta ?? {});
    if (decoder.skipped.length > 0) {
        const skipped = Object.freeze(decoder.skipped.map((s) => Object.freeze({ ...s })));
        meta = Object.freeze({ ...meta, extra: Object.freeze({ ...meta.extra, [SKIPPED_COLUMNS_KEY]: skipped }) });
    }
    const carried = decoder.views(manifest.views);
    if (decoder.structure) {
        decoder.checkOverlaps();
    }
    const parts: SnapshotParts = {
        label: asStringOrNull(manifest.label, "label"),
        serial: null,
        directed,
        nodeCount: counts.nodes,
        edgeCount: counts.edges,
        arcCount: counts.arcs,
        selfLoopCount: counts.selfLoops,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags,
        ids,
        nodes,
        edges,
        graph,
        extensions,
        meta,
        arena,
        checksum: false,
    };
    const snapshot = createSnapshot(parts);
    if (options.level === "full") {
        snapshot.validate({ level: "full" });
    } else if (options.level === "structure") {
        snapshot.validate({ level: "structure" });
    }
    if (options.level !== "full") {
        installViews(snapshot, carried, options.level === "structure");
    }
    return snapshot;
}

/**
 * Rebuild a snapshot from its wire form (design section 9.1). Defaults: `validate: "structure"`
 * (the manifest is trusted to have come from this package; the cheap checks still run) and `copy:
 * false` (the arrays are views into `wire.buffers`; a SharedArrayBuffer is copied). The reverse id
 * Map and decoded strings are rebuilt lazily; carried views (`includeViews`) are installed on the
 * receiver below "full" and recomputed under "full", where nothing carried is trusted.
 * @param wire - the wire snapshot
 * @param options - validation level, copy and unknown-column policy
 * @returns the snapshot; E_UNSUPPORTED_VERSION, E_UNSUPPORTED, E_BAD_SERIALIZATION or E_INVALID_SNAPSHOT on bad input
 */
export function fromWire(wire: WireSnapshot, options?: FromWireOptions): GraphSnapshot {
    const resolved = resolveFromWireOptions(options, "structure");
    const w = asObject(wire, "wire");
    const buffersRaw = asArray(w.buffers, "wire.buffers");
    const regions: WireRegion[] = [];
    for (let i = 0; i < buffersRaw.length; i++) {
        const b: unknown = buffersRaw[i];
        let buffer: ArrayBuffer;
        if (b instanceof ArrayBuffer) {
            buffer = resolved.copy ? copyBuffer(b) : b;
        } else if (isSharedArrayBuffer(b)) {
            buffer = copyBuffer(b);
        } else {
            throw badWire(`wire.buffers[${i}]`, "expected an ArrayBuffer");
        }
        regions.push(bufferRegion(buffer, 0, buffer.byteLength));
    }
    return decodeManifest(w.manifest, regions, resolved, true);
}
