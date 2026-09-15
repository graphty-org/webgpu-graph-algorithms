/**
 * The one packed-bitmap layout of @graphty/graph-format (design sections 5.1, 5.3 and 7.4, decision
 * C13): Uint32Array words, LSB-first, ceil(bits / 32) words, bit i of the bitmap at
 * `(words[i >>> 5] >>> (i & 31)) & 1`. Validity bitmaps, bool column data, NodeMask / EdgeMask and
 * the builder's alive / weightSet bitmaps all use these helpers, so a kernel reads every one of them
 * with the same expression and the layout is bit-for-bit Arrow's on a little-endian host.
 *
 * Convention: a bitmap over `bits` bits keeps every bit at position >= bits CLEAR. Every constructor
 * and mutator here maintains that, so word-level counts, ANDs and ORs stay exact without a length
 * argument; callers that write words directly call bitmapClearTrailing() afterwards. Bit indices are
 * plain numbers below 2^32 (invariant I3), so `>>> 5` and `& 31` are safe on them; word indices are
 * multiplied, never shifted, because a word index can exceed 2^26.
 */

import { type U32 } from "../types/index.js";

/** Every bit set: the fill word of an all-true bitmap. */
const ALL_ONES = 0xffffffff;

/**
 * The number of words a bitmap over `bits` bits needs: ceil(bits / 32). Computed with a division, not
 * `(bits + 31) >>> 5`, because bits may be as large as MAX_COUNT and the addition would wrap.
 * @param bits - the bit count, >= 0
 * @returns ceil(bits / 32)
 */
export function bitmapWordCount(bits: number): number {
    return Math.ceil(bits / 32);
}

/**
 * The mask of the bits of the last word that lie below `bits`: all ones when bits is a multiple of
 * 32, otherwise the low `bits % 32` bits. Computed with an unsigned shift so the result is a
 * non-negative number even when 31 bits are kept (`(1 << 31) - 1` is not an int32).
 * @param bits - the bit count
 * @returns the mask as an unsigned 32-bit value
 */
function lastWordMask(bits: number): number {
    const remainder = bits % 32;
    return remainder === 0 ? ALL_ONES : ALL_ONES >>> (32 - remainder);
}

/**
 * Clear every bit at a position >= bits: the tail of the last word and every word beyond. Call it
 * after writing words directly so the bitmap keeps the trailing-bits-clear convention.
 * @param words - the bitmap words
 * @param bits - the bit count the bitmap covers
 */
export function bitmapClearTrailing(words: U32, bits: number): void {
    const wordCount = bitmapWordCount(bits);
    if (wordCount > 0 && wordCount <= words.length && bits % 32 !== 0) {
        words[wordCount - 1] &= lastWordMask(bits);
    }
    if (wordCount < words.length) {
        words.fill(0, wordCount);
    }
}

/**
 * Allocate a bitmap over `bits` bits: ceil(bits / 32) words, every bit clear or, with `fill`, every
 * bit below `bits` set (trailing bits clear).
 * @param bits - the bit count, >= 0
 * @param fill - the initial value of every bit; defaults to false
 * @returns the new words
 */
export function makeBitmap(bits: number, fill = false): U32 {
    const words = new Uint32Array(bitmapWordCount(bits));
    if (fill) {
        words.fill(ALL_ONES);
        bitmapClearTrailing(words, bits);
    }
    return words;
}

/**
 * Read bit i.
 * @param words - the bitmap words
 * @param i - the bit index, below the bitmap's bit count
 * @returns true when the bit is set
 */
export function bitmapGet(words: U32, i: number): boolean {
    return ((words[i >>> 5] >>> (i & 31)) & 1) === 1;
}

/**
 * Set bit i.
 * @param words - the bitmap words
 * @param i - the bit index, below the bitmap's bit count
 */
export function bitmapSet(words: U32, i: number): void {
    words[i >>> 5] |= 1 << (i & 31);
}

/**
 * Clear bit i.
 * @param words - the bitmap words
 * @param i - the bit index, below the bitmap's bit count
 */
export function bitmapClear(words: U32, i: number): void {
    words[i >>> 5] &= ~(1 << (i & 31));
}

/**
 * Write bit i.
 * @param words - the bitmap words
 * @param i - the bit index, below the bitmap's bit count
 * @param value - the value to write
 */
export function bitmapWrite(words: U32, i: number, value: boolean): void {
    if (value) {
        bitmapSet(words, i);
    } else {
        bitmapClear(words, i);
    }
}

/**
 * The number of set bits in one 32-bit word (SWAR popcount; the multiply goes through Math.imul so
 * it never leaves 32 bits).
 * @param word - the word, read as an unsigned 32-bit value
 * @returns the count, 0..32
 */
export function popcount32(word: number): number {
    let w = word - ((word >>> 1) & 0x55555555);
    w = (w & 0x33333333) + ((w >>> 2) & 0x33333333);
    w = (w + (w >>> 4)) & 0x0f0f0f0f;
    return Math.imul(w, 0x01010101) >>> 24;
}

/**
 * The number of set bits below `bits`. Bits at or above `bits` are ignored, so the count is exact
 * even for a bitmap whose trailing bits are dirty.
 * @param words - the bitmap words, at least ceil(bits / 32) of them
 * @param bits - the bit count to count over
 * @returns the number of set bits in [0, bits)
 */
export function bitmapCount(words: U32, bits: number): number {
    const fullWords = Math.floor(bits / 32);
    let count = 0;
    for (let k = 0; k < fullWords; k++) {
        count += popcount32(words[k]);
    }
    if (bits % 32 !== 0) {
        count += popcount32(words[fullWords] & lastWordMask(bits));
    }
    return count;
}

/**
 * The indices of the set bits below `bits`, ascending, as a fresh U32 of exactly bitmapCount()
 * entries (the maskToIndices of design section 7.4).
 * @param words - the bitmap words
 * @param bits - the bit count to collect over
 * @returns the set bit indices
 */
export function bitmapToIndices(words: U32, bits: number): U32 {
    const out = new Uint32Array(bitmapCount(words, bits));
    let n = 0;
    for (let i = 0; i < bits; i++) {
        if (((words[i >>> 5] >>> (i & 31)) & 1) === 1) {
            out[n++] = i;
        }
    }
    return out;
}

/**
 * The bits [start, end) of `words` as a fresh bitmap over end - start bits starting at bit 0 (the
 * copy behind Column.slice() for bool data and validity, design section 5.7). A word-aligned start
 * is a plain copy; any other start shifts across word boundaries.
 * @param words - the source words
 * @param start - the first bit to copy
 * @param end - one past the last bit to copy
 * @returns the new words, trailing bits clear
 */
export function bitmapSlice(words: U32, start: number, end: number): U32 {
    const bits = Math.max(0, end - start);
    const out = makeBitmap(bits);
    if (bits === 0) {
        return out;
    }
    const firstWord = start >>> 5;
    const shift = start & 31;
    if (shift === 0) {
        out.set(words.subarray(firstWord, firstWord + out.length));
    } else {
        const upper = 32 - shift;
        for (let j = 0; j < out.length; j++) {
            const k = firstWord + j;
            const lo = k < words.length ? words[k] >>> shift : 0;
            const hi = k + 1 < words.length ? words[k + 1] << upper : 0;
            out[j] = lo | hi;
        }
    }
    bitmapClearTrailing(out, bits);
    return out;
}
