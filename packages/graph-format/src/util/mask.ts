/**
 * The public packed-mask helpers of design section 7.4 (makeMask, maskTest, maskSet, maskCount,
 * maskToIndices) plus the E_MASK_LENGTH check that filterEdges() and inducedSubgraph({ mask }) apply
 * to a caller's mask. A NodeMask / EdgeMask is the same layout as a validity bitmap and a bool column
 * (decision C13): ceil(length / 32) Uint32Array words, LSB-first, bit i set means index i is included.
 * Masks are not part of the snapshot contract; no kernel is required to honour one.
 */

import {
    bitmapCount,
    bitmapGet,
    bitmapToIndices,
    bitmapWordCount,
    bitmapWrite,
    makeBitmap,
} from "../columns/bitmap.js";
import { GraphFormatError } from "../errors.js";
import { type U32 } from "../types/index.js";

/**
 * Allocate a packed mask over `length` indices: ceil(length / 32) words, every bit clear, or every bit
 * below `length` set when `fill` is true (design section 7.4).
 * @param length - the number of indices the mask covers (nodeCount, edgeCount or arcCount)
 * @param fill - the initial value of every bit; defaults to false
 * @returns the mask words
 */
export function makeMask(length: number, fill?: boolean): U32 {
    return makeBitmap(length, fill === true);
}

/**
 * Whether index i is in the mask.
 * @param mask - the mask words
 * @param i - the index, below the mask's length
 * @returns true when bit i is set
 */
export function maskTest(mask: U32, i: number): boolean {
    return bitmapGet(mask, i);
}

/**
 * Include or exclude index i.
 * @param mask - the mask words
 * @param i - the index, below the mask's length
 * @param value - true to include, false to exclude
 */
export function maskSet(mask: U32, i: number, value: boolean): void {
    bitmapWrite(mask, i, value);
}

/**
 * The number of included indices below `length` (the loop guard of an algorithm-owned alive mask,
 * design section 7.4).
 * @param mask - the mask words, at least ceil(length / 32) of them
 * @param length - the number of indices the mask covers
 * @returns the number of set bits in [0, length)
 */
export function maskCount(mask: U32, length: number): number {
    return bitmapCount(mask, length);
}

/**
 * The included indices below `length` in ascending order, as a fresh U32.
 * @param mask - the mask words, at least ceil(length / 32) of them
 * @param length - the number of indices the mask covers
 * @returns the set bit indices
 */
export function maskToIndices(mask: U32, length: number): U32 {
    return bitmapToIndices(mask, length);
}

/**
 * Check that a caller-supplied mask has enough words for `length` indices; filterEdges() and
 * inducedSubgraph({ mask }) call it before reading the mask (design section 11.2). Throws
 * E_MASK_LENGTH when mask.length < ceil(length / 32).
 * @param mask - the mask words
 * @param length - the number of indices the mask must cover
 * @param what - what the mask is over, for the message ("edges", "nodes")
 */
export function checkMaskLength(mask: U32, length: number, what: string): void {
    const required = bitmapWordCount(length);
    if (mask.length < required) {
        throw new GraphFormatError(
            "E_MASK_LENGTH",
            `mask over ${what} has ${mask.length} words; ${required} needed for ${length} ${what}`,
            { found: mask.length, required, length },
        );
    }
}
