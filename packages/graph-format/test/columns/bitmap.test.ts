import fc from "fast-check";
import { describe, expect, it } from "vitest";

import {
    bitmapClear,
    bitmapClearTrailing,
    bitmapCount,
    bitmapGet,
    bitmapSet,
    bitmapSlice,
    bitmapToIndices,
    bitmapWordCount,
    bitmapWrite,
    makeBitmap,
    popcount32,
} from "../../src/columns/bitmap.js";
import { MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";

/** The bit test every kernel uses (design section 5.3), as an independent oracle. */
function kernelRead(words: Uint32Array, r: number): number {
    return (words[r >>> 5] >>> (r & 31)) & 1;
}

/** A bitmap built from a boolean model, bit by bit through the kernel expression. */
function fromModel(model: readonly boolean[]): Uint32Array<ArrayBuffer> {
    const words = new Uint32Array(Math.ceil(model.length / 32));
    for (let i = 0; i < model.length; i++) {
        if (model[i]) {
            words[i >>> 5] |= 1 << (i & 31);
        }
    }
    return words;
}

function toModel(words: Uint32Array, bits: number): boolean[] {
    const out: boolean[] = [];
    for (let i = 0; i < bits; i++) {
        out.push(kernelRead(words, i) === 1);
    }
    return out;
}

function naivePopcount(word: number): number {
    let count = 0;
    for (let b = 0; b < 32; b++) {
        count += (word >>> b) & 1;
    }
    return count;
}

const BOUNDARY_SIZES = [0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129];

describe("bitmapWordCount", () => {
    it("is ceil(bits / 32) at every boundary", () => {
        expect(bitmapWordCount(0)).toBe(0);
        expect(bitmapWordCount(1)).toBe(1);
        expect(bitmapWordCount(31)).toBe(1);
        expect(bitmapWordCount(32)).toBe(1);
        expect(bitmapWordCount(33)).toBe(2);
        expect(bitmapWordCount(64)).toBe(2);
        expect(bitmapWordCount(65)).toBe(3);
    });

    it("does not wrap at MAX_COUNT (design invariant I3)", () => {
        expect(bitmapWordCount(MAX_COUNT)).toBe(134217728);
        expect(bitmapWordCount(2 ** 32 - 32)).toBe(134217727);
        expect(bitmapWordCount(2 ** 32 - 31)).toBe(134217728);
    });
});

describe("makeBitmap / bitmapClearTrailing", () => {
    it("allocates ceil(bits / 32) clear words", () => {
        for (const bits of BOUNDARY_SIZES) {
            const words = makeBitmap(bits);
            expect(words).toBeInstanceOf(Uint32Array);
            expect(words.length).toBe(bitmapWordCount(bits));
            expect(words.every((w) => w === 0)).toBe(true);
            expect(bitmapCount(words, bits)).toBe(0);
        }
    });

    it("makeBitmap(bits, true) sets exactly the bits below the length (trailing bits clear)", () => {
        for (const bits of BOUNDARY_SIZES) {
            const words = makeBitmap(bits, true);
            expect(bitmapCount(words, bits)).toBe(bits);
            expect(toModel(words, words.length * 32)).toEqual(
                Array.from({ length: words.length * 32 }, (_, i) => i < bits),
            );
        }
        expect(Array.from(makeBitmap(3, true))).toEqual([0b111]);
        expect(Array.from(makeBitmap(33, true))).toEqual([0xffffffff, 1]);
    });

    it("bitmapClearTrailing clears the tail of the last word and every word beyond", () => {
        const words = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
        bitmapClearTrailing(words, 70);
        expect(Array.from(words)).toEqual([0xffffffff, 0xffffffff, 0x3f, 0]);
        const aligned = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff]);
        bitmapClearTrailing(aligned, 64);
        expect(Array.from(aligned)).toEqual([0xffffffff, 0xffffffff, 0]);
        const thirtyOne = new Uint32Array([0xffffffff]);
        bitmapClearTrailing(thirtyOne, 31);
        expect(Array.from(thirtyOne)).toEqual([0x7fffffff]);
        const zero = new Uint32Array([0xffffffff]);
        bitmapClearTrailing(zero, 0);
        expect(Array.from(zero)).toEqual([0]);
        // a short array (fewer words than the length needs) is left alone
        const short = new Uint32Array([0xffffffff]);
        bitmapClearTrailing(short, 70);
        expect(Array.from(short)).toEqual([0xffffffff]);
    });
});

describe("get / set / clear / write", () => {
    it("addresses bits LSB-first within little-endian words, matching the kernel expression", () => {
        const words = makeBitmap(96);
        for (const i of [0, 1, 31, 32, 33, 63, 64, 95]) {
            bitmapSet(words, i);
            expect(bitmapGet(words, i)).toBe(true);
            expect(kernelRead(words, i)).toBe(1);
        }
        expect(Array.from(words)).toEqual([0x80000003, 0x80000003, 0x80000001]);
        // the bytes of word 0 read as Arrow's LSB-first bitmap on a little-endian host
        expect(Array.from(new Uint8Array(words.buffer, 0, 4))).toEqual([0x03, 0x00, 0x00, 0x80]);
        bitmapClear(words, 31);
        expect(bitmapGet(words, 31)).toBe(false);
        expect(words[0]).toBe(3);
        bitmapWrite(words, 31, true);
        expect(words[0]).toBe(0x80000003);
        bitmapWrite(words, 0, false);
        expect(words[0]).toBe(0x80000002);
        expect(bitmapGet(words, 2)).toBe(false);
    });

    it("bit 31 of a word round-trips through set, get, clear and count", () => {
        const words = makeBitmap(32);
        bitmapSet(words, 31);
        expect(words[0]).toBe(0x80000000);
        expect(bitmapGet(words, 31)).toBe(true);
        expect(bitmapCount(words, 32)).toBe(1);
        expect(bitmapCount(words, 31)).toBe(0);
        expect(Array.from(bitmapToIndices(words, 32))).toEqual([31]);
        bitmapClear(words, 31);
        expect(words[0]).toBe(0);
    });

    it("reads beyond the word array as clear", () => {
        expect(bitmapGet(makeBitmap(0), 5)).toBe(false);
        expect(bitmapGet(makeBitmap(32), 40)).toBe(false);
    });
});

describe("bitmapSlice / bitmapToIndices", () => {
    it("bitmapSlice copies [start, end) to bit 0 of a fresh bitmap, at any alignment", () => {
        fc.assert(
            fc.property(
                fc.array(fc.boolean(), { minLength: 0, maxLength: 100 }),
                fc.nat(100),
                fc.nat(100),
                (model, a, b) => {
                    const start = Math.min(a, b, model.length);
                    const end = Math.min(Math.max(a, b), model.length);
                    const words = fromModel(model);
                    const sliced = bitmapSlice(words, start, end);
                    expect(sliced.length).toBe(bitmapWordCount(end - start));
                    expect(toModel(sliced, end - start)).toEqual(model.slice(start, end));
                    // the trailing bits of the slice are clear
                    expect(bitmapCount(sliced, sliced.length * 32)).toBe(bitmapCount(sliced, end - start));
                },
            ),
        );
    });

    it("bitmapToIndices lists the set bits below the length, ascending", () => {
        const words = fromModel([true, false, true, true, false]);
        expect(Array.from(bitmapToIndices(words, 5))).toEqual([0, 2, 3]);
        expect(Array.from(bitmapToIndices(words, 2))).toEqual([0]);
        expect(Array.from(bitmapToIndices(words, 0))).toEqual([]);
        expect(Array.from(bitmapToIndices(new Uint32Array(0), 0))).toEqual([]);
        expect(() => bitmapToIndices(new Uint32Array(0), 0)).not.toThrow(GraphFormatError);
    });
});

describe("popcount32 / bitmapCount", () => {
    it("popcount32 matches the naive count on every kind of word", () => {
        expect(popcount32(0)).toBe(0);
        expect(popcount32(1)).toBe(1);
        expect(popcount32(0x80000000)).toBe(1);
        expect(popcount32(0xffffffff)).toBe(32);
        expect(popcount32(0x7fffffff)).toBe(31);
        expect(popcount32(0x55555555)).toBe(16);
        expect(popcount32(0xf0f0f0f0)).toBe(16);
        fc.assert(
            fc.property(fc.integer({ min: 0, max: 0xffffffff }), (word) => {
                expect(popcount32(word)).toBe(naivePopcount(word));
            }),
        );
    });

    it("bitmapCount counts only the bits below the length", () => {
        const words = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff]);
        expect(bitmapCount(words, 0)).toBe(0);
        expect(bitmapCount(words, 1)).toBe(1);
        expect(bitmapCount(words, 31)).toBe(31);
        expect(bitmapCount(words, 32)).toBe(32);
        expect(bitmapCount(words, 33)).toBe(33);
        expect(bitmapCount(words, 63)).toBe(63);
        expect(bitmapCount(words, 64)).toBe(64);
        expect(bitmapCount(words, 65)).toBe(65);
        expect(bitmapCount(words, 96)).toBe(96);
    });
});
