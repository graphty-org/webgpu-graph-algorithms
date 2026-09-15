/**
 * Low-level typed-array helpers shared by every module of @graphty/graph-format (design sections 5.7,
 * 6.2, 9.5, 10.2 and 10.3): rounding and alignment arithmetic, arena segment layout, the 4-byte
 * padding rules of I10, the u8 adoption predicate, the wire buffer-range check, per-dtype
 * typed-array constructors, and the growth primitive of the builder staging (doubling into a
 * resizable ArrayBuffer when the engine has one, allocate-and-copy doubling otherwise).
 *
 * Nothing here knows about graphs; everything is arithmetic over byte lengths and typed arrays.
 */

import { ALIGNMENT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { type Dtype, type TypedArrayData, type U8, type U32 } from "../types/index.js";

// ============================================================ resizable ArrayBuffer detection

/**
 * Probe once whether this engine supports resizable ArrayBuffer (ES2024: Node 20+, Chrome 111+,
 * Safari 16.4+, Firefox 128+). The probe constructs a tiny resizable buffer and grows it; any throw or
 * missing member means "not supported", in which case staging grows by allocate-and-copy (design
 * section 6.2).
 * @returns true when `new ArrayBuffer(n, { maxByteLength })` and `resize()` work
 */
function detectResizableArrayBuffer(): boolean {
    try {
        const probe = new ArrayBuffer(0, { maxByteLength: 64 });
        if (!probe.resizable || typeof probe.resize !== "function") {
            return false;
        }
        probe.resize(64);
        return probe.byteLength === 64 && probe.maxByteLength === 64;
    } catch {
        return false;
    }
}

/**
 * Whether resizable ArrayBuffer is available, detected once at module load (design section 6.2). The
 * growth helpers take it as their default; tests pass an explicit value to exercise both paths.
 */
export const HAS_RESIZABLE_ARRAY_BUFFER: boolean = detectResizableArrayBuffer();

// ============================================================ staging growth constants

/** Staging capacities are rounded up to this many elements (design section 6.2). */
export const STAGING_ELEMENT_GRANULE = 16;

/** Staging backing buffers are a multiple of this many bytes (design section 6.2). */
export const STAGING_BYTE_GRANULE = 64;

/**
 * Virtual address space reserved (`maxByteLength`) for a fresh resizable staging buffer: 256 MiB, so
 * a staging array grows in place without copying up to 64M u32 elements. Reservations are
 * address-space only (pages are committed as the buffer grows), and a buffer that outgrows its
 * reservation is reallocated with a doubled one, so the constant bounds the number of copies rather
 * than the size of anything.
 */
export const STAGING_RESERVE_BYTES = 256 * 1024 * 1024;

// ============================================================ rounding and alignment

/**
 * Round `value` up to the next multiple of `multiple`. Written with `%`, not bit masks, because arc
 * counts and byte lengths exceed 2^31 (invariant I3, design section 10.6).
 * @param value - a non-negative integer
 * @param multiple - a positive integer
 * @returns the smallest multiple of `multiple` that is >= value
 */
export function roundUp(value: number, multiple: number): number {
    const remainder = value % multiple;
    return remainder === 0 ? value : value + multiple - remainder;
}

/**
 * Pad a byte length to the next multiple of 4: the size of the padded u32 view of a u8 column and
 * the rule every GPU-bound array satisfies (design sections 5.7 and 10.2).
 * @param byteLength - a non-negative byte count
 * @returns roundUp(byteLength, 4)
 */
export function padTo4(byteLength: number): number {
    return roundUp(byteLength, 4);
}

/**
 * The number of u32 words that span a byte range: ceil(byteLength / 4) (design section 10.2).
 * @param byteLength - a non-negative byte count
 * @returns the word count of the padded u32 view
 */
export function paddedWordCount(byteLength: number): number {
    return Math.ceil(byteLength / 4);
}

/**
 * Align a byte offset up to the next arena segment boundary (design section 10.3; 256 bytes, the
 * WebGPU minStorageBufferOffsetAlignment default).
 * @param byteOffset - a non-negative byte offset
 * @param alignment - the alignment in bytes; defaults to ALIGNMENT (256)
 * @returns roundUp(byteOffset, alignment)
 */
export function alignUp(byteOffset: number, alignment: number = ALIGNMENT): number {
    return roundUp(byteOffset, alignment);
}

// ============================================================ arena segment layout

/**
 * The result of laying out consecutive segments at aligned offsets (design section 10.3).
 */
interface SegmentLayout {
    /**
     * One entry per input length: the segment's byte offset relative to the start of the arena, or null for
     * a zero-length segment, which occupies nothing.
     */
    readonly offsets: readonly (number | null)[];
    /** One entry per input length: offset + length, or null for a zero-length segment. */
    readonly ends: readonly (number | null)[];
    /** The end of the last non-empty segment; 0 when every segment is empty. No trailing padding is added. */
    readonly byteLength: number;
    /** The bytes spent on inter-segment padding. */
    readonly padding: number;
}

/**
 * Lay out segments back to back, each non-empty one starting at a multiple of `alignment`, in the
 * given order (design section 10.3: hot to cold). Zero-length segments get a null offset and occupy
 * nothing, which is how an absent `weights`, a zero-arc `colIdx` and an identity permutation stay out
 * of the arena. For the worked example of section 10.3 (400,004 / 8,000,000 / 8,000,000 / 8,000,000 /
 * 4,000,000) the offsets are 0 / 400,128 / 8,400,128 / 16,400,128 / 24,400,128 and the total is
 * 28,400,128 with 124 bytes of padding.
 * @param byteLengths - the unpadded byte length of every segment in arena order (0 = absent)
 * @param alignment - the segment alignment in bytes; defaults to ALIGNMENT (256)
 * @returns the offsets, ends, total byte length and padding of the layout
 */
export function layoutSegments(byteLengths: readonly number[], alignment: number = ALIGNMENT): SegmentLayout {
    const offsets: (number | null)[] = [];
    const ends: (number | null)[] = [];
    let cursor = 0;
    let padding = 0;
    for (const byteLength of byteLengths) {
        if (byteLength === 0) {
            offsets.push(null);
            ends.push(null);
            continue;
        }
        const offset = alignUp(cursor, alignment);
        padding += offset - cursor;
        offsets.push(offset);
        cursor = offset + byteLength;
        ends.push(cursor);
    }
    return { offsets, ends, byteLength: cursor, padding };
}

// ============================================================ 4-byte rules and adoption (I10, 5.7)

/**
 * The I10 predicate for a GPU-bound array: a view whose byteOffset and byteLength are both multiples
 * of 4 (design section 10.2).
 * @param view - any ArrayBufferView
 * @returns true when both byteOffset and byteLength are multiples of 4
 */
export function isFourByteAligned(view: ArrayBufferView): boolean {
    return view.byteOffset % 4 === 0 && view.byteLength % 4 === 0;
}

/**
 * The "plain ArrayBuffer" half of invariant I10 (decision D-SAB, design section 9.4): the view's
 * buffer is an `ArrayBuffer` that is neither a `SharedArrayBuffer` nor resizable, so its bytes are a
 * `BufferSource` for `GPUQueue.writeBuffer`, transferable, and of fixed length for the life of the
 * snapshot (invariant I17).
 * @param view - any ArrayBufferView
 * @returns true when the buffer is a plain, fixed-length ArrayBuffer
 */
export function isOverPlainBuffer(view: ArrayBufferView): boolean {
    const { buffer } = view;
    return buffer instanceof ArrayBuffer && !buffer.resizable;
}

/**
 * The u8 adoption predicate of design section 5.7: "a zero-copy padded Uint32Array view is
 * constructible", i.e. `byteOffset % 4 === 0 && byteOffset + roundUp(byteLength, 4) <=
 * buffer.byteLength`.
 * @param data - the u8 array a column would adopt
 * @returns true when paddedU32View(data) can be built without copying
 */
export function canViewAsPaddedU32(data: U8): boolean {
    return data.byteOffset % 4 === 0 && data.byteOffset + padTo4(data.byteLength) <= data.buffer.byteLength;
}

/**
 * The zero-copy Uint32Array of ceil(byteLength / 4) words over a u8 array's own byte range (design
 * sections 5.7 and 10.2); the trailing lanes of the last word are whatever the buffer holds there.
 * @param data - a u8 array satisfying canViewAsPaddedU32()
 * @returns the padded u32 view; E_COLUMN_ALIGNMENT when the view is not constructible
 */
export function paddedU32View(data: U8): U32 {
    if (!canViewAsPaddedU32(data)) {
        throw new GraphFormatError(
            "E_COLUMN_ALIGNMENT",
            `u8 array at byteOffset ${data.byteOffset} with byteLength ${data.byteLength} has no zero-copy padded u32 view`,
            { byteOffset: data.byteOffset, byteLength: data.byteLength, bufferByteLength: data.buffer.byteLength },
        );
    }
    return new Uint32Array(data.buffer, data.byteOffset, paddedWordCount(data.byteLength));
}

/**
 * Copy a u8 array into a fresh store over which the padded u32 view is constructible: a new
 * ArrayBuffer of roundUp(length, 4) bytes viewed at offset 0 with the original length (design section
 * 5.7, the `adopt: "copy"` path). The padding bytes are zero.
 * @param data - the u8 array to copy
 * @returns a new U8 of the same length whose store satisfies canViewAsPaddedU32()
 */
export function copyToPaddedStore(data: U8): U8 {
    const store = new Uint8Array(new ArrayBuffer(padTo4(data.byteLength)), 0, data.byteLength);
    store.set(data);
    return store;
}

// ============================================================ per-dtype typed arrays

/**
 * The dtypes whose storage is one flat typed array: the five numeric dtypes plus bool (packed u32
 * words) and dict (u32 codes). string, list and json have offsets, children or JS values instead.
 */
export type TypedDtype = Exclude<Dtype, "string" | "list" | "json">;

/**
 * A typed-array class as the helpers use it: BYTES_PER_ELEMENT plus the (buffer, byteOffset, length)
 * constructor. Every concrete typed-array constructor satisfies it with T fixed to its
 * ArrayBuffer-parameterised instance type.
 */
export interface TypedArrayCtor<T extends TypedArrayData> {
    /** Bytes per element. */
    readonly BYTES_PER_ELEMENT: number;
    /**
     * Construct a view of `length` elements over `buffer` starting at `byteOffset`.
     * @param buffer - the backing buffer
     * @param byteOffset - the start of the view in bytes
     * @param length - the element count of the view
     */
    new (buffer: ArrayBuffer, byteOffset: number, length: number): T;
}

/**
 * Throw E_UNSUPPORTED for a dtype value outside the union; the `default` branch of every dtype switch
 * (design section 13.4) calls this so an unknown dtype coming from untrusted data fails loudly. Always
 * throws.
 * @param dtype - the value that reached the default branch
 */
export function unsupportedDtype(dtype: never): never {
    throw new GraphFormatError("E_UNSUPPORTED", `unsupported dtype ${String(dtype)}`, { dtype });
}

/**
 * The typed-array class that stores a flat dtype (design section 5.1): Float32Array for f32,
 * Float64Array for f64, Int32Array for i32, Uint8Array for u8, and Uint32Array for u32, for the packed
 * words of bool and for the codes of dict.
 * @param dtype - a flat dtype
 * @returns the constructor; E_UNSUPPORTED for an unknown dtype
 */
export function typedArrayCtorOf(dtype: TypedDtype): TypedArrayCtor<TypedArrayData> {
    switch (dtype) {
        case "f32":
            return Float32Array;
        case "f64":
            return Float64Array;
        case "i32":
            return Int32Array;
        case "u8":
            return Uint8Array;
        case "u32":
        case "bool":
        case "dict":
            return Uint32Array;
        default:
            return unsupportedDtype(dtype);
    }
}

// ============================================================ staging growth

/**
 * The next staging capacity, in elements, that holds at least `needed`: doubling from `current`,
 * rounded up to STAGING_ELEMENT_GRANULE (design section 6.2). A capacity of 0 grows to the granule.
 * @param current - the present capacity in elements
 * @param needed - the minimum capacity required
 * @returns the new capacity, >= needed, >= 2 * current, a multiple of 16
 */
export function growCapacity(current: number, needed: number): number {
    return roundUp(Math.max(needed, current * 2, STAGING_ELEMENT_GRANULE), STAGING_ELEMENT_GRANULE);
}

/**
 * The byte length of a staging backing buffer for `capacity` elements of `bytesPerElement` bytes each:
 * rounded up to STAGING_BYTE_GRANULE (design section 6.2). Because every element size divides 64 the
 * result is always a whole number of elements.
 * @param capacity - the capacity in elements
 * @param bytesPerElement - the element size in bytes (1, 4 or 8)
 * @returns the buffer byte length, a multiple of 64
 */
export function stagingByteLength(capacity: number, bytesPerElement: number): number {
    return roundUp(capacity * bytesPerElement, STAGING_BYTE_GRANULE);
}

/**
 * Allocate a staging buffer of `byteLength` bytes. With `resizable` true the buffer is created with a
 * `maxByteLength` of at least `reserveBytes` (and at least `byteLength`) so it can later grow in place;
 * should the engine refuse that reservation with a RangeError (an address-space limit, not a missing
 * capability) the buffer is created resizable with `maxByteLength === byteLength`, which simply means
 * its next growth reallocates. With `resizable` false a plain fixed-length buffer is returned.
 * @param byteLength - the initial byte length
 * @param resizable - whether to create a resizable buffer; defaults to HAS_RESIZABLE_ARRAY_BUFFER
 * @param reserveBytes - the address space to reserve for in-place growth; defaults to STAGING_RESERVE_BYTES
 * @returns the new buffer
 */
export function allocateStagingBuffer(
    byteLength: number,
    resizable: boolean = HAS_RESIZABLE_ARRAY_BUFFER,
    reserveBytes: number = STAGING_RESERVE_BYTES,
): ArrayBuffer {
    if (!resizable) {
        return new ArrayBuffer(byteLength);
    }
    const maxByteLength = Math.max(byteLength, reserveBytes);
    try {
        return new ArrayBuffer(byteLength, { maxByteLength });
    } catch (err) {
        if (err instanceof RangeError && maxByteLength > byteLength) {
            return new ArrayBuffer(byteLength, { maxByteLength: byteLength });
        }
        throw err;
    }
}

/**
 * Grow a staging array so it holds at least `minLength` elements, preserving its contents (design
 * section 6.2). When the array is the leading view of a resizable buffer with room left, the buffer is
 * resized in place and no bytes move; otherwise a new buffer is allocated (resizable per `resizable`,
 * with a reservation of at least twice the old one so repeated overflow stays amortised) and the
 * contents copied. Either way the returned array is a fresh view object of exactly the new capacity;
 * the caller replaces its reference. Elements beyond the old length are zero on the copy path and on a
 * freshly resized region.
 * @param array - the current staging array, viewed from byteOffset 0 of its buffer
 * @param ctor - the array's constructor
 * @param minLength - the minimum element count required
 * @param resizable - whether new buffers are resizable; defaults to HAS_RESIZABLE_ARRAY_BUFFER
 * @returns a view of growCapacity(array.length, minLength) elements holding the old contents
 */
export function growTypedArray<T extends TypedArrayData>(
    array: T,
    ctor: TypedArrayCtor<T>,
    minLength: number,
    resizable: boolean = HAS_RESIZABLE_ARRAY_BUFFER,
): T {
    if (minLength <= array.length) {
        return array;
    }
    const bytesPerElement = ctor.BYTES_PER_ELEMENT;
    const byteLength = stagingByteLength(growCapacity(array.length, minLength), bytesPerElement);
    const capacity = byteLength / bytesPerElement;
    const { buffer } = array;
    if (buffer.resizable && array.byteOffset === 0 && byteLength <= buffer.maxByteLength) {
        buffer.resize(byteLength);
        return new ctor(buffer, 0, capacity);
    }
    const reserve = Math.max(buffer.maxByteLength * 2, STAGING_RESERVE_BYTES);
    const next = new ctor(allocateStagingBuffer(byteLength, resizable, reserve), 0, capacity);
    next.set(array);
    return next;
}
