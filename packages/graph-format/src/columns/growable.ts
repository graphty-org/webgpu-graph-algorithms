/**
 * Growable typed staging for the builder (design section 6.2): structure-of-arrays typed arrays that
 * grow by doubling (capacity rounded up to 16 elements, backing buffers a multiple of 64 bytes) into
 * a resizable ArrayBuffer when the engine has one and by allocate-and-copy otherwise. The edge arrays
 * (src, dst, weight), the incidence lists, the per-declaration column data and the alive / weightSet /
 * validity bitmaps of the builder are all instances of the two classes here.
 *
 * Nothing a snapshot holds may alias staging (invariant I18): view() hands out a zero-copy window for
 * the freeze passes to read, and trim() produces the exact-length copy over a fresh fixed-length
 * buffer that a snapshot may keep.
 */

import { type TypedArrayData, type U32 } from "../types/index.js";
import {
    allocateStagingBuffer,
    growTypedArray,
    HAS_RESIZABLE_ARRAY_BUFFER,
    roundUp,
    STAGING_ELEMENT_GRANULE,
    stagingByteLength,
    type TypedArrayCtor,
} from "../util/typed-array.js";
import { bitmapClearTrailing, bitmapCount, bitmapGet, bitmapSet, bitmapWordCount, bitmapWrite } from "./bitmap.js";

/** Every bit set: the fill word of a range of whole words. */
const ALL_ONES = 0xffffffff;

/** Construction options shared by GrowableTypedArray and GrowableBitmap. */
interface GrowableOptions {
    /** Initial capacity in elements (bits for a GrowableBitmap); defaults to 0, so the first push allocates. */
    readonly capacity?: number | undefined;
    /**
     * Whether backing buffers are resizable ArrayBuffers; defaults to the engine's HAS_RESIZABLE_ARRAY_BUFFER. Tests pass
     * false to exercise the allocate-and-copy path on an engine that has the feature.
     */
    readonly resizable?: boolean | undefined;
}

/**
 * A typed array that grows by doubling (design section 6.2). `length` is the number of elements in
 * use; `capacity` the allocated element count. Elements between length and capacity are unspecified
 * until written. Growth may replace the backing view object, so hot loops that cache `array` re-read
 * it after any call that can grow (push, pushAll, resize, ensureCapacity).
 */
export class GrowableTypedArray<T extends TypedArrayData> {
    private readonly ctor: TypedArrayCtor<T>;
    private readonly resizable: boolean;
    private backing: T;
    private used = 0;

    /**
     * Create an empty growable array.
     * @param ctor - the typed-array class of the elements (Uint32Array, Float32Array, ...)
     * @param options - initial capacity and buffer kind
     */
    constructor(ctor: TypedArrayCtor<T>, options?: GrowableOptions) {
        this.ctor = ctor;
        this.resizable = options?.resizable ?? HAS_RESIZABLE_ARRAY_BUFFER;
        this.backing = this.allocate(options?.capacity ?? 0);
    }

    /**
     * The number of elements in use.
     * @returns the length
     */
    get length(): number {
        return this.used;
    }

    /**
     * The allocated element count; growth happens when length would exceed it.
     * @returns the capacity
     */
    get capacity(): number {
        return this.backing.length;
    }

    /**
     * The backing view over the whole capacity, for hot loops that index it directly. Its identity
     * changes whenever the array grows.
     * @returns the backing typed array
     */
    get array(): T {
        return this.backing;
    }

    /**
     * Append one element.
     * @param value - the element
     * @returns the index it was written at (the old length)
     */
    push(value: number): number {
        const index = this.used;
        if (index >= this.backing.length) {
            this.backing = growTypedArray(this.backing, this.ctor, index + 1, this.resizable);
        }
        this.backing[index] = value;
        this.used = index + 1;
        return index;
    }

    /**
     * Append many elements at once (bulk addEdges input).
     * @param values - the elements, in order
     * @returns the index the first one was written at (the old length)
     */
    pushAll(values: ArrayLike<number>): number {
        const index = this.used;
        const end = index + values.length;
        this.ensureCapacity(end);
        this.backing.set(values, index);
        this.used = end;
        return index;
    }

    /**
     * Read element i.
     * @param i - the index, below length
     * @returns the element
     */
    get(i: number): number {
        return this.backing[i];
    }

    /**
     * Write element i.
     * @param i - the index, below length
     * @param value - the element
     */
    set(i: number, value: number): void {
        this.backing[i] = value;
    }

    /**
     * Make sure at least `n` elements fit without a further reallocation.
     * @param n - the capacity to guarantee
     */
    ensureCapacity(n: number): void {
        if (n > this.backing.length) {
            this.backing = growTypedArray(this.backing, this.ctor, n, this.resizable);
        }
    }

    /**
     * Set the length: growing fills the new elements with `fill` (the incidence lists grow with
     * INVALID_INDEX), shrinking drops the tail without releasing capacity.
     * @param length - the new length
     * @param fill - the value written into elements [old length, length); defaults to 0
     */
    resize(length: number, fill = 0): void {
        if (length > this.used) {
            this.ensureCapacity(length);
            this.backing.fill(fill, this.used, length);
        }
        this.used = length;
    }

    /**
     * A zero-copy window over the elements in use. It aliases staging, so it is for the freeze passes
     * to read from and never for a snapshot to keep (invariant I18); it is invalidated by growth.
     * @returns a view of exactly `length` elements
     */
    view(): T {
        return new this.ctor(this.backing.buffer, 0, this.used);
    }

    /**
     * An exact-length copy over a fresh fixed-length ArrayBuffer: what a snapshot may keep, and what a
     * compaction writes back through.
     * @returns a new typed array of `length` elements
     */
    trim(): T {
        const out = new this.ctor(new ArrayBuffer(this.used * this.ctor.BYTES_PER_ELEMENT), 0, this.used);
        out.set(this.view());
        return out;
    }

    /**
     * Drop every element but keep the capacity (a builder reused after a merging freeze).
     */
    clear(): void {
        this.used = 0;
    }

    /**
     * Drop the elements and the backing storage (freeze({ release: true }) and dispose(), design
     * section 6.1). The array is usable again afterwards; the next push allocates.
     */
    release(): void {
        this.used = 0;
        this.backing = this.allocate(0);
    }

    /**
     * Allocate a backing view of at least `capacity` elements over a fresh staging buffer.
     * @param capacity - the element count wanted; 0 allocates an empty view over an empty buffer
     * @returns the view
     */
    private allocate(capacity: number): T {
        const bytesPerElement = this.ctor.BYTES_PER_ELEMENT;
        const byteLength =
            capacity === 0 ? 0 : stagingByteLength(roundUp(capacity, STAGING_ELEMENT_GRANULE), bytesPerElement);
        return new this.ctor(allocateStagingBuffer(byteLength, this.resizable), 0, byteLength / bytesPerElement);
    }
}

/**
 * A growable packed bitmap (design section 6.2: nodeAlive, edgeAlive, weightSet and the validity of
 * every staging column), in the one LSB-first u32 layout of design section 5.3. `length` is in bits;
 * every bit at or above it is kept clear so view() and trim() are exact bitmaps over `length` bits.
 */
export class GrowableBitmap {
    private readonly words: GrowableTypedArray<U32>;
    private bits = 0;

    /**
     * Create an empty bitmap.
     * @param options - initial capacity in bits and buffer kind
     */
    constructor(options?: GrowableOptions) {
        this.words = new GrowableTypedArray<U32>(Uint32Array, {
            capacity: bitmapWordCount(options?.capacity ?? 0),
            resizable: options?.resizable,
        });
    }

    /**
     * The number of bits in use.
     * @returns the length in bits
     */
    get length(): number {
        return this.bits;
    }

    /**
     * Append one bit.
     * @param value - the bit
     * @returns the index it was written at (the old length)
     */
    push(value: boolean): number {
        const index = this.bits;
        if (index % 32 === 0) {
            this.words.push(0);
        }
        if (value) {
            bitmapWrite(this.words.array, index, true);
        }
        this.bits = index + 1;
        return index;
    }

    /**
     * Read bit i.
     * @param i - the bit index, below length
     * @returns true when set
     */
    get(i: number): boolean {
        return bitmapGet(this.words.array, i);
    }

    /**
     * Write bit i.
     * @param i - the bit index, below length
     * @param value - the bit
     */
    set(i: number, value: boolean): void {
        bitmapWrite(this.words.array, i, value);
    }

    /**
     * The number of set bits below length.
     * @returns the count
     */
    count(): number {
        return bitmapCount(this.words.array, this.bits);
    }

    /**
     * Set the length in bits: growing writes `fill` into the new bits, shrinking clears the dropped
     * bits so the trailing-bits-clear convention holds.
     * @param length - the new length in bits
     * @param fill - the value of bits [old length, length); defaults to false
     */
    resize(length: number, fill = false): void {
        const oldBits = this.bits;
        this.words.resize(bitmapWordCount(length), 0);
        if (length < oldBits) {
            bitmapClearTrailing(this.words.array, length);
        } else if (fill && length > oldBits) {
            this.setRange(oldBits, length);
        }
        this.bits = length;
    }

    /**
     * Set every bit in [from, to): the ragged head and tail bit by bit, the whole words in between
     * with one fill.
     * @param from - the first bit to set
     * @param to - one past the last bit to set
     */
    private setRange(from: number, to: number): void {
        const { array } = this.words;
        let i = from;
        while (i < to && i % 32 !== 0) {
            bitmapSet(array, i++);
        }
        const fullEnd = to - (to % 32);
        if (i < fullEnd) {
            array.fill(ALL_ONES, i / 32, fullEnd / 32);
            i = fullEnd;
        }
        while (i < to) {
            bitmapSet(array, i++);
        }
    }

    /**
     * A zero-copy window over the ceil(length / 32) words in use; aliases staging (invariant I18) and
     * is invalidated by growth.
     * @returns the words
     */
    view(): U32 {
        return this.words.view();
    }

    /**
     * An exact copy of the words in use over a fresh fixed-length ArrayBuffer: a validity bitmap or a
     * bool column's data a snapshot may keep.
     * @returns a new U32 of ceil(length / 32) words
     */
    trim(): U32 {
        return this.words.trim();
    }

    /**
     * Drop every bit but keep the capacity.
     */
    clear(): void {
        this.bits = 0;
        this.words.clear();
    }

    /**
     * Drop the bits and the backing storage.
     */
    release(): void {
        this.bits = 0;
        this.words.release();
    }
}
