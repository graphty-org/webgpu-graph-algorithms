/**
 * The wire encoder (design sections 4.5, 5.9, 9.1 and 12.2): `toWire()` turns a snapshot into a
 * JSON-serialisable manifest plus a list of ArrayBuffers, `transferables()` lists the exclusively
 * owned buffers a worker hand-off may transfer, and the segment collector shared with the GSNP
 * container writer of bytes.ts lays every typed array out as a WireBufferRef.
 *
 * Two layouts exist. In REFERENCE mode (toWire) every distinct backing ArrayBuffer becomes one entry
 * of `buffers` and a reference carries the array's own byteOffset, so the core stays views over the
 * arena (one buffer) and nothing is copied unless `transfer: true` meets a shared buffer. In
 * CONTAINER mode (toBytes / toByteChunks) every array is placed at a 256-byte-aligned offset of one
 * buffer region in manifest order, core arrays first so the region's prefix is the arena.
 *
 * toWire allocates nothing for typed data except the first materialisation of a lazily kept
 * representation: the Utf8 store of string ids and string columns and the f64 / tag arrays of a
 * numeric or mixed id map are cached by their owners, the encoded dictionaries and the JSON text of
 * json columns are cached here. Non-finite numbers and -0 inside metadata are carried as the tagged
 * strings of design section 5.9 so the manifest survives JSON.stringify unchanged.
 */

import { CORE_ORDER } from "../builder/arena.js";
import { isPlainObject } from "../columns/column.js";
import { ALIGNMENT, FORMAT_VERSION, IS_LITTLE_ENDIAN, WIRE_FORMAT, WIRE_MAJOR, WIRE_MINOR } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { NodeIdMap, nodeIdMapToTyped } from "../ids/node-id-map.js";
import { type EncodedUtf8, encodeUtf8Rows } from "../ids/string-store.js";
import { type GraphSnapshot, peekView } from "../snapshot/graph-snapshot.js";
import { viewArrays } from "../snapshot/views.js";
import {
    type AttributeTable,
    type Column,
    type ColumnMeta,
    type CoreArrayName,
    type GraphMeta,
    type JsonColumn,
    type ToWireOptions,
    type TypedArrayData,
    type ViewName,
    type WireArena,
    type WireBufferRef,
    type WireColumn,
    type WireDtype,
    type WireIdMap,
    type WireManifest,
    type WireSnapshot,
    type WireUtf8,
} from "../types/index.js";
import { isShared, noteShared } from "../util/shared-buffers.js";
import { alignUp } from "../util/typed-array.js";

// ============================================================ constants

/**
 * The `producer` string written into every manifest (design section 9.1): the package name and its
 * npm version. Golden fixtures mask it when compared. Bumped together with package.json.
 */
export const WIRE_PRODUCER = "@graphty/graph-format@0.1.0";

/** The views the wire carries when named by includeViews; the scalar views are never carried. */
const SCALAR_VIEWS: ReadonlySet<ViewName> = new Set<ViewName>(["totalWeight", "symmetric"]);

/** The core arrays in arena order (design section 10.3). */

const encoder = new TextEncoder();

// ============================================================ host endianness

/**
 * Refuse to emit wire data on a big-endian host (design section 9.2): every supported platform is
 * little-endian and there is no byte-swapping reader, so a big-endian host must never produce a file
 * other hosts reject. The parameter exists so the refusal path is testable on a little-endian host.
 * @param littleEndian - whether the host is little-endian; defaults to the detected value
 */
export function assertLittleEndianHost(littleEndian: boolean = IS_LITTLE_ENDIAN): void {
    if (!littleEndian) {
        throw new GraphFormatError("E_UNSUPPORTED", "the wire format cannot be written on a big-endian host", {
            reason: "big-endian host",
        });
    }
}

// ============================================================ shared-buffer bookkeeping (9.1)

// ============================================================ JSON tagging (5.9)

/** The tag key of a non-finite or negative-zero number on the wire. */
const NUM_TAG = "$num";
const ESC_TAG = "$esc";

/**
 * Encode a JSON value for the manifest (design section 5.9): `Infinity`, `-Infinity`, `NaN` and `-0`
 * become `{ "$num": "..." }`, arrays and plain objects are copied recursively, everything else is
 * returned as is. A user object whose ONLY key is `$num` or `$esc` would collide with the tags, so
 * it is wrapped as `{ "$esc": { ...the object... } }` and unwrapped by the reader.
 * The input has already passed assertJsonValue (design section 5.9), so no other shape can occur.
 * @param value - the JSON value
 * @returns the tagged copy
 */
export function encodeJsonValue(value: unknown): unknown {
    if (typeof value === "number") {
        if (Number.isNaN(value)) {
            return { [NUM_TAG]: "NaN" };
        }
        if (value === Infinity) {
            return { [NUM_TAG]: "Infinity" };
        }
        if (value === -Infinity) {
            return { [NUM_TAG]: "-Infinity" };
        }
        if (Object.is(value, -0)) {
            return { [NUM_TAG]: "-0" };
        }
        return value;
    }
    if (Array.isArray(value)) {
        return value.map((item: unknown) => encodeJsonValue(item));
    }
    if (isPlainObject(value)) {
        const keys = Object.keys(value);
        const out: Record<string, unknown> = {};
        for (const key of keys) {
            defineJsonKey(out, key, encodeJsonValue(value[key]));
        }
        if (keys.length === 1 && (keys[0] === NUM_TAG || keys[0] === ESC_TAG)) {
            return { [ESC_TAG]: out };
        }
        return out;
    }
    return value;
}

/**
 * Add an own enumerable property to a plain object without going through a setter, so a key such as
 * "__proto__" stays a data property (the manifest reader rejects it anyway).
 * @param target - the object
 * @param key - the property name
 * @param value - the value
 */
export function defineJsonKey(target: Record<string, unknown>, key: string, value: unknown): void {
    Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true });
}

// ============================================================ the segment collector

/** How references are laid out. */
type WireMode = "reference" | "container";

/** One typed array placed in the container region: the source view and where it goes. */
export interface ContainerSegment {
    /** The source array. */
    readonly view: TypedArrayData;
    /** Its reference (buffer 0, region-relative byteOffset). */
    readonly ref: WireBufferRef;
}

/**
 * Collects typed arrays into WireBufferRefs (design sections 9.1 and 9.2). In reference mode
 * `buffers` is the list of distinct backing buffers (a shared buffer copied under `transfer`, its
 * index recorded in `copied`); in container mode `segments` is the placement of every distinct array
 * at a 256-aligned offset and `regionLength` the padded length of the region.
 */
export class WireLayout {
    /** The layout mode. */
    readonly mode: WireMode;
    /** Reference mode: whether shared buffers are copied so the rest can be transferred. */
    readonly transfer: boolean;
    /** Reference mode: the distinct buffers, or their copies. */
    readonly buffers: ArrayBuffer[] = [];
    /** Reference mode: the indices of `buffers` holding copies of shared buffers. */
    readonly copied: number[] = [];
    /** Container mode: every placed array in region order. */
    readonly segments: ContainerSegment[] = [];

    private readonly bufferIndex = new Map<ArrayBuffer, number>();
    private readonly placed = new Map<string, WireBufferRef>();
    private cursor = 0;

    /**
     * Create a collector.
     * @param mode - reference (toWire) or container (toBytes)
     * @param transfer - reference mode only: copy shared buffers so the others can be transferred
     */
    constructor(mode: WireMode, transfer = false) {
        this.mode = mode;
        this.transfer = transfer;
    }

    /**
     * Container mode: the padded byte length of the region (every segment padded to 256 bytes).
     * @returns the region length
     */
    get regionLength(): number {
        return alignUp(this.cursor, ALIGNMENT);
    }

    /**
     * Reference mode: the index a buffer was assigned, or null when no array over it was placed.
     * @param buffer - the buffer
     * @returns the index into `buffers`, or null
     */
    indexOf(buffer: ArrayBuffer): number | null {
        return this.bufferIndex.get(buffer) ?? null;
    }

    /**
     * Place a typed array and return its reference. The same byte range is placed once. When the
     * holder of the array (its column, table or id map) is shared, the buffer is recorded as shared
     * so `finalise()` copies it under `transfer` (design section 9.1).
     * @param view - the array
     * @param dtype - its wire dtype ("utf8" for byte stores, "u8" for u8 columns and tags)
     * @param sharedHolder - whether the object holding the array is held by more than one snapshot or table
     * @returns the reference
     */
    ref(view: TypedArrayData, dtype: WireDtype, sharedHolder = false): WireBufferRef {
        if (sharedHolder) {
            noteShared(view.buffer);
        }
        if (this.mode === "reference") {
            return {
                buffer: this.bufferIndexOf(view.buffer),
                byteOffset: view.byteOffset,
                byteLength: view.byteLength,
                dtype,
                length: view.length,
            };
        }
        const key = `${this.bufferIndexOf(view.buffer)}:${view.byteOffset}:${view.byteLength}:${dtype}`;
        const existing = this.placed.get(key);
        if (existing !== undefined) {
            return existing;
        }
        const byteOffset = alignUp(this.cursor, ALIGNMENT);
        this.cursor = byteOffset + view.byteLength;
        const ref: WireBufferRef = { buffer: 0, byteOffset, byteLength: view.byteLength, dtype, length: view.length };
        this.placed.set(key, ref);
        this.segments.push({ view, ref });
        return ref;
    }

    /**
     * Reference mode with `transfer`: replace every buffer recorded as shared by a copy and list its
     * index in `copied` (design section 9.1). Called once after every array was placed, so the
     * decision does not depend on the order in which shared holders were met.
     */
    finalise(): void {
        if (this.mode !== "reference" || !this.transfer) {
            return;
        }
        for (let i = 0; i < this.buffers.length; i++) {
            const buffer = this.buffers[i];
            if (isShared(buffer)) {
                this.buffers[i] = buffer.slice(0);
                this.copied.push(i);
            }
        }
    }

    /**
     * The index of a backing buffer, assigning the next one on first sight.
     * @param buffer - the buffer
     * @returns the index
     */
    private bufferIndexOf(buffer: ArrayBuffer): number {
        const known = this.bufferIndex.get(buffer);
        if (known !== undefined) {
            return known;
        }
        const index = this.buffers.length;
        this.bufferIndex.set(buffer, index);
        this.buffers.push(buffer);
        return index;
    }
}

// ============================================================ dtype helpers

/**
 * The wire dtype of a typed array by its class.
 * @param view - the array
 * @returns the dtype ("u8" for a Uint8Array; byte stores pass "utf8" explicitly)
 */
function wireDtypeOf(view: TypedArrayData): WireDtype {
    if (view instanceof Uint32Array) {
        return "u32";
    }
    if (view instanceof Int32Array) {
        return "i32";
    }
    if (view instanceof Float32Array) {
        return "f32";
    }
    if (view instanceof Float64Array) {
        return "f64";
    }
    return "u8";
}

// ============================================================ caches of lazily kept representations

/** Encoded dictionaries, keyed by the dictionary array (shared between a static column and its temporal table). */
const DICTIONARY_CACHE = new WeakMap<readonly string[], EncodedUtf8>();

/** JSON text of json columns, keyed by the column, with the column version it was built from. */
const JSON_TEXT_CACHE = new WeakMap<Column, { readonly version: number; readonly encoded: EncodedUtf8 }>();

/**
 * Whether a cached Utf8 store is still usable: the same row count and not detached by a transfer.
 * @param encoded - the cached store
 * @param rows - the expected row count
 * @returns true when the cache can be reused
 */
function cacheUsable(encoded: EncodedUtf8, rows: number): boolean {
    return encoded.offsets.length === rows + 1;
}

/**
 * The encoded form of a dictionary, built once per dictionary array.
 * @param dictionary - the strings in code order
 * @returns the offsets and bytes
 */
function encodedDictionary(dictionary: readonly string[]): EncodedUtf8 {
    const cached = DICTIONARY_CACHE.get(dictionary);
    if (cached !== undefined && cacheUsable(cached, dictionary.length)) {
        return cached;
    }
    const encoded = encodeUtf8Rows(dictionary, dictionary.length);
    DICTIONARY_CACHE.set(dictionary, encoded);
    return encoded;
}

/**
 * The JSON text of every row of a json column as a Utf8 store (an unset row is empty text), built
 * once per column version.
 * @param column - the json column
 * @returns the offsets and bytes
 */
function encodedJsonText(column: JsonColumn): EncodedUtf8 {
    const cached = JSON_TEXT_CACHE.get(column);
    if (cached !== undefined && cached.version === column.version && cacheUsable(cached.encoded, column.length)) {
        return cached.encoded;
    }
    const texts = new Array<string>(column.length);
    for (let row = 0; row < column.length; row++) {
        const value = column.values[row];
        texts[row] = value === undefined || !column.isSet(row) ? "" : JSON.stringify(encodeJsonValue(value));
    }
    const encoded = encodeUtf8Rows(texts, texts.length);
    JSON_TEXT_CACHE.set(column, { version: column.version, encoded });
    return encoded;
}

// ============================================================ column encoding

/**
 * A Utf8 store as two references.
 * @param layout - the collector
 * @param encoded - the offsets and bytes
 * @param shared - whether the holder of the store is shared
 * @returns the wire store
 */
function utf8Ref(layout: WireLayout, encoded: EncodedUtf8, shared: boolean): WireUtf8 {
    return { offsets: layout.ref(encoded.offsets, "u32", shared), utf8: layout.ref(encoded.utf8, "utf8", shared) };
}

/**
 * The manifest copy of a column's metadata: the JSON fields tagged per design section 5.9, the rest
 * copied as is. The tagged fields no longer match ColumnMeta's static types when a value was
 * non-finite; the reader decodes them before resolving the metadata again.
 * @param meta - the resolved metadata
 * @returns the manifest form
 */
function encodeColumnMeta(meta: ColumnMeta): ColumnMeta {
    return {
        name: meta.name,
        domain: meta.domain,
        dtype: meta.dtype,
        components: meta.components,
        itemDtype: meta.itemDtype,
        itemComponents: meta.itemComponents,
        nullable: meta.nullable,
        mutable: meta.mutable,
        role: meta.role,
        refersTo: meta.refersTo,
        unique: meta.unique,
        default: encodeJsonValue(meta.default),
        fill: encodeJsonValue(meta.fill) as number | string | boolean,
        options: meta.options === null ? null : (encodeJsonValue(meta.options) as readonly unknown[]),
        origin: meta.origin === null ? null : { ...meta.origin },
        dynamic: meta.dynamic,
        extra: encodeJsonValue(meta.extra) as Readonly<Record<string, unknown>>,
    };
}

/**
 * Encode one column (design section 9.1): the buffers its dtype uses, null for the rest. A column
 * held by more than one table (or belonging to a table held by more than one snapshot) has every
 * buffer recorded as shared, including the ones it materialises lazily while being encoded.
 * @param layout - the collector
 * @param column - the column
 * @param tableShared - whether the table holding the column is shared
 * @returns the wire column
 */
function encodeColumn(layout: WireLayout, column: Column, tableShared: boolean): WireColumn {
    const shared = tableShared || isShared(column);
    const base = {
        meta: encodeColumnMeta(column.meta),
        data: null as WireBufferRef | null,
        validity: column.validity === null ? null : layout.ref(column.validity, "u32", shared),
        nullCount: column.nullCount,
        dictionary: null as WireUtf8 | null,
        strings: null as WireUtf8 | null,
        offsets: null as WireBufferRef | null,
        child: null as WireColumn | null,
        jsonText: null as WireUtf8 | null,
    };
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
            base.data = layout.ref(column.data, dtype, shared);
            break;
        case "bool":
            base.data = layout.ref(column.data, "u32", shared);
            break;
        case "dict":
            base.data = layout.ref(column.codes, "u32", shared);
            base.dictionary = utf8Ref(layout, encodedDictionary(column.dictionary), shared);
            break;
        case "string":
            base.strings = {
                offsets: layout.ref(column.offsets, "u32", shared),
                utf8: layout.ref(column.utf8, "utf8", shared),
            };
            break;
        case "list":
            base.offsets = layout.ref(column.offsets, "u32", shared);
            base.child = encodeColumn(layout, column.child, shared);
            break;
        case "json":
            base.jsonText = utf8Ref(layout, encodedJsonText(column), shared);
            break;
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dtype ${name}`, { dtype: name });
        }
    }
    return base;
}

/**
 * Encode every column of a table in declaration order.
 * @param layout - the collector
 * @param table - the table
 * @returns the wire columns
 */
function encodeTable(layout: WireLayout, table: AttributeTable): WireColumn[] {
    const out: WireColumn[] = [];
    const tableShared = isShared(table);
    for (const column of table) {
        out.push(encodeColumn(layout, column, tableShared));
    }
    return out;
}

// ============================================================ id map, meta, views

/**
 * Encode the id map (design section 4.5), materialising its typed form on first use.
 * @param layout - the collector
 * @param snapshot - the snapshot
 * @returns the wire id map
 */
function encodeIds(layout: WireLayout, snapshot: GraphSnapshot): WireIdMap {
    const { ids } = snapshot;
    if (!(ids instanceof NodeIdMap)) {
        throw new GraphFormatError("E_UNSUPPORTED", "the snapshot's id map is not a NodeIdMap of this package", {
            reason: "foreign id map",
        });
    }
    const typed = nodeIdMapToTyped(ids);
    const shared = isShared(ids);
    return {
        kind: typed.kind,
        size: typed.size,
        offset: typed.offset,
        values: typed.values === null ? null : layout.ref(typed.values, wireDtypeOf(typed.values), shared),
        tags: typed.tags === null ? null : layout.ref(typed.tags, "u8", shared),
        numbers: typed.numbers === null ? null : layout.ref(typed.numbers, "f64", shared),
        offsets: typed.offsets === null ? null : layout.ref(typed.offsets, "u32", shared),
        utf8: typed.utf8 === null ? null : layout.ref(typed.utf8, "utf8", shared),
    };
}

/**
 * The manifest copy of the graph metadata with `extra` tagged per design section 5.9.
 * @param meta - the metadata
 * @returns the manifest form
 */
function encodeGraphMeta(meta: GraphMeta): GraphMeta {
    return {
        name: meta.name,
        description: meta.description,
        creator: meta.creator,
        created: meta.created,
        modified: meta.modified,
        keywords: [...meta.keywords],
        sourceFormat: meta.sourceFormat,
        sourceVersion: meta.sourceVersion,
        idType: meta.idType,
        timeFormat: meta.timeFormat,
        timeRepresentation: meta.timeRepresentation,
        mode: meta.mode,
        declaredMultigraph: meta.declaredMultigraph,
        weightOrigin: meta.weightOrigin === null ? null : { ...meta.weightOrigin },
        extra: encodeJsonValue(meta.extra) as Readonly<Record<string, unknown>>,
    };
}

/**
 * The cached, non-scalar views named by includeViews, each as its materialised member arrays
 * (design section 9.1). A view that is not resident is skipped: the receiver recomputes it.
 * @param layout - the collector
 * @param snapshot - the snapshot
 * @param names - the requested views
 * @returns the wire views, or null when none was carried
 */
function encodeViews(
    layout: WireLayout,
    snapshot: GraphSnapshot,
    names: readonly ViewName[] | undefined,
): Record<string, Record<string, WireBufferRef>> | null {
    if (names === undefined || names.length === 0) {
        return null;
    }
    const out: Record<string, Record<string, WireBufferRef>> = {};
    let count = 0;
    for (const name of new Set(names)) {
        if (SCALAR_VIEWS.has(name)) {
            continue;
        }
        const value: unknown = peekView(snapshot, name);
        if (value === null) {
            continue;
        }
        const members: Record<string, WireBufferRef> = {};
        let any = false;
        for (const [member, array] of Object.entries(viewArrays(snapshot, name, value))) {
            const view = array as TypedArrayData;
            members[member] = layout.ref(view, wireDtypeOf(view));
            any = true;
        }
        if (any) {
            out[name] = members;
            count++;
        }
    }
    return count === 0 ? null : out;
}

// ============================================================ the manifest

/** The result of collecting a snapshot: the manifest and the layout that placed its arrays. */
interface CollectedWire {
    /** The manifest. */
    readonly manifest: WireManifest;
    /** The collector holding the buffers (reference mode) or the segments (container mode). */
    readonly layout: WireLayout;
}

/**
 * Build the manifest of a snapshot over a collector (design section 9.1): core arrays first (an
 * identity permutation, an absent weights array and a zero-length array are null), then the id map,
 * the tables, the extension tables, the metadata and the requested views.
 * @param snapshot - the snapshot
 * @param layout - the collector
 * @param includeViews - the views to carry
 * @param includeColumns - whether attribute and extension tables are carried (default true)
 * @returns the manifest and the collector
 */
export function collectWire(
    snapshot: GraphSnapshot,
    layout: WireLayout,
    includeViews: readonly ViewName[] | undefined,
    includeColumns = true,
): CollectedWire {
    assertLittleEndianHost();
    const identity = snapshot.flags.arcToEdgeIsIdentity;
    const coreArrays: Record<CoreArrayName, TypedArrayData | null> = {
        rowPtr: snapshot.rowPtr,
        colIdx: snapshot.colIdx,
        weights: snapshot.weights,
        arcToEdge: identity ? null : snapshot.arcToEdge,
        edgeToArc: identity ? null : snapshot.edgeToArc,
    };
    const coreRefs: Record<CoreArrayName, WireBufferRef | null> = {
        rowPtr: null,
        colIdx: null,
        weights: null,
        arcToEdge: null,
        edgeToArc: null,
    };
    const rowPtrRef = layout.ref(snapshot.rowPtr, "u32");
    coreRefs.rowPtr = rowPtrRef;
    for (const name of CORE_ORDER) {
        const array = coreArrays[name];
        if (name !== "rowPtr" && array !== null && array.length > 0) {
            coreRefs[name] = layout.ref(array, wireDtypeOf(array));
        }
    }
    const arena = encodeArena(snapshot, layout, coreRefs);
    const ids = encodeIds(layout, snapshot);
    const nodeColumns = includeColumns ? encodeTable(layout, snapshot.nodes) : [];
    const edgeColumns = includeColumns ? encodeTable(layout, snapshot.edges) : [];
    const graphColumns = includeColumns ? encodeTable(layout, snapshot.graph) : [];
    const extensions: { name: string; rowCount: number; columns: WireColumn[] }[] = [];
    if (includeColumns) {
        for (const [name, table] of snapshot.extensions) {
            extensions.push({ name, rowCount: table.rowCount, columns: encodeTable(layout, table) });
        }
    }
    const views = encodeViews(layout, snapshot, includeViews);
    layout.finalise();
    const manifest: WireManifest = {
        format: WIRE_FORMAT,
        wire: [WIRE_MAJOR, WIRE_MINOR],
        producer: WIRE_PRODUCER,
        formatVersion: FORMAT_VERSION,
        directed: snapshot.directed,
        counts: {
            nodes: snapshot.nodeCount,
            edges: snapshot.edgeCount,
            arcs: snapshot.arcCount,
            selfLoops: snapshot.selfLoopCount,
        },
        flags: { ...snapshot.flags },
        core: {
            rowPtr: rowPtrRef,
            colIdx: coreRefs.colIdx,
            weights: coreRefs.weights,
            arcToEdge: coreRefs.arcToEdge,
            edgeToArc: coreRefs.edgeToArc,
        },
        arena,
        ids,
        nodeColumns,
        edgeColumns,
        graphColumns,
        extensions,
        meta: encodeGraphMeta(snapshot.meta),
        views,
        copied: layout.copied,
        label: snapshot.label,
    };
    return { manifest, layout };
}

/**
 * The arena descriptor: in container mode the prefix of the region holding the core arrays (always
 * present); in reference mode the snapshot's own arena when it has one.
 * @param snapshot - the snapshot
 * @param layout - the collector, after the core arrays were placed
 * @param coreRefs - the core references
 * @returns the descriptor, or null
 */
function encodeArena(
    snapshot: GraphSnapshot,
    layout: WireLayout,
    coreRefs: Readonly<Record<CoreArrayName, WireBufferRef | null>>,
): WireArena | null {
    if (layout.mode === "container") {
        let end = 0;
        let hotByteLength = 0;
        for (const name of CORE_ORDER) {
            const ref = coreRefs[name];
            if (ref === null) {
                continue;
            }
            end = ref.byteOffset + ref.byteLength;
            if (name === "rowPtr" || name === "colIdx" || name === "weights") {
                hotByteLength = end;
            }
        }
        return { buffer: 0, byteOffset: 0, byteLength: end, hotByteLength };
    }
    const { arena } = snapshot;
    if (arena === null) {
        return null;
    }
    const buffer = layout.indexOf(arena.buffer);
    if (buffer === null) {
        return null;
    }
    return { buffer, byteOffset: arena.byteOffset, byteLength: arena.byteLength, hotByteLength: arena.hotByteLength };
}

// ============================================================ public entry points

/**
 * The plain-object wire form of a snapshot (design section 9.1): a JSON-serialisable manifest plus
 * the distinct backing ArrayBuffers of the core, the id map, the columns and the extension tables.
 * Without `transfer` every buffer is a plain reference and postMessage clones; with `transfer: true`
 * every buffer held by more than one snapshot, table or column (design section 9.1, tracked by
 * src/util/shared-buffers.ts) is copied (its index listed in `manifest.copied`) so the remaining
 * buffers can be transferred with `transferables()`.
 * @param snapshot - the snapshot
 * @param options - transfer mode, the cached views to carry and whether columns are carried
 * @returns the wire snapshot; E_UNSUPPORTED on a big-endian host
 */
export function toWire(snapshot: GraphSnapshot, options: ToWireOptions = {}): WireSnapshot {
    const transfer = options.transfer === true;
    const layout = new WireLayout("reference", transfer);
    const { manifest } = collectWire(snapshot, layout, options.includeViews, options.includeColumns !== false);
    if (transfer) {
        // the transfer list of THIS wire: exclusive originals plus the fresh copies of shared buffers
        TRANSFER_LISTS.set(snapshot, [...layout.buffers]);
    }
    return { manifest, buffers: layout.buffers };
}

/** Per snapshot, the buffers its most recent `toWire({ transfer: true })` put into `wire.buffers`. */
const TRANSFER_LISTS = new WeakMap<GraphSnapshot, ArrayBuffer[]>();

/**
 * The postMessage transfer list of design section 9.1: the same set `toWire({ transfer: true })`
 * put into `wire.buffers` -- exclusively owned originals plus the fresh copies made for shared
 * buffers -- so `postMessage(wire, snapshot.transferables())` transfers exactly the buffers the
 * wire carries, whatever `includeColumns` / `includeViews` selected (a buffer the wire does not
 * reference is never listed, so no column is lost on either side). Before any transfer-mode toWire()
 * call the list is computed for the default wire shape: the distinct exclusively owned buffers of
 * the core, the id map, the typed columns, the string stores and the extension tables; a buffer
 * held by more than one snapshot, table or column is excluded and an identity permutation is never
 * included.
 * @param snapshot - the snapshot
 * @returns the buffers, in manifest order
 */
export function transferables(snapshot: GraphSnapshot): ArrayBuffer[] {
    const recorded = TRANSFER_LISTS.get(snapshot);
    if (recorded !== undefined) {
        return [...recorded];
    }
    const layout = new WireLayout("reference", false);
    collectWire(snapshot, layout, undefined, true);
    return layout.buffers.filter((buffer) => !isShared(buffer));
}

/**
 * The UTF-8 bytes of a manifest as the container stores them (design section 9.2).
 * @param manifest - the manifest
 * @returns the encoded JSON text
 */
export function encodeManifest(manifest: WireManifest): Uint8Array {
    return encoder.encode(JSON.stringify(manifest));
}
