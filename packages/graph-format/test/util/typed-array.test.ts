import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { ALIGNMENT, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    alignUp,
    allocateStagingBuffer,
    canViewAsPaddedU32,
    copyToPaddedStore,
    growCapacity,
    growTypedArray,
    HAS_RESIZABLE_ARRAY_BUFFER,
    isFourByteAligned,
    layoutSegments,
    paddedU32View,
    paddedWordCount,
    padTo4,
    roundUp,
    STAGING_BYTE_GRANULE,
    STAGING_ELEMENT_GRANULE,
    STAGING_RESERVE_BYTES,
    stagingByteLength,
    typedArrayCtorOf,
    type TypedDtype,
    unsupportedDtype,
} from "../../src/util/typed-array.js";

describe("resizable ArrayBuffer detection", () => {
    it("detects the feature on Node 22", () => {
        // The staging design (section 6.2) relies on it from Node 20 on; the copy path is exercised
        // explicitly below by passing resizable: false.
        expect(HAS_RESIZABLE_ARRAY_BUFFER).toBe(true);
    });
});

describe("roundUp / alignUp / padTo4 / paddedWordCount", () => {
    it("rounds up to a multiple", () => {
        expect(roundUp(0, 4)).toBe(0);
        expect(roundUp(1, 4)).toBe(4);
        expect(roundUp(4, 4)).toBe(4);
        expect(roundUp(5, 4)).toBe(8);
        expect(roundUp(400004, 256)).toBe(400128);
        expect(roundUp(8000000, 256)).toBe(8000000);
    });

    it("works above 2^31 where bit masks would go negative (design section 10.6)", () => {
        expect(roundUp(0x80000041, 64)).toBe(0x80000080);
        expect(roundUp(0x80000040, 64)).toBe(0x80000040);
        expect(roundUp(MAX_COUNT, 256)).toBe(0x100000000);
        expect(roundUp(MAX_COUNT * 4, 4)).toBe(MAX_COUNT * 4);
        expect(alignUp(MAX_COUNT * 4)).toBe(roundUp(MAX_COUNT * 4, 256));
    });

    it("padTo4 pads a byte length to the next multiple of 4", () => {
        expect(padTo4(0)).toBe(0);
        expect(padTo4(1)).toBe(4);
        expect(padTo4(3)).toBe(4);
        expect(padTo4(4)).toBe(4);
        expect(padTo4(5)).toBe(8);
        expect(padTo4(100001)).toBe(100004);
    });

    it("paddedWordCount is ceil(byteLength / 4)", () => {
        expect(paddedWordCount(0)).toBe(0);
        expect(paddedWordCount(1)).toBe(1);
        expect(paddedWordCount(4)).toBe(1);
        expect(paddedWordCount(5)).toBe(2);
        expect(paddedWordCount(100001)).toBe(25001);
    });

    it("alignUp defaults to the 256-byte arena alignment", () => {
        expect(ALIGNMENT).toBe(256);
        expect(alignUp(0)).toBe(0);
        expect(alignUp(1)).toBe(256);
        expect(alignUp(256)).toBe(256);
        expect(alignUp(400004)).toBe(400128);
        expect(alignUp(400004, 4)).toBe(400004);
        expect(alignUp(400005, 4)).toBe(400008);
    });

    it("roundUp agrees with the ceiling definition on random inputs", () => {
        fc.assert(
            fc.property(fc.nat(2 ** 40), fc.integer({ min: 1, max: 4096 }), (value, multiple) => {
                const rounded = roundUp(value, multiple);
                expect(rounded % multiple).toBe(0);
                expect(rounded).toBeGreaterThanOrEqual(value);
                expect(rounded - value).toBeLessThan(multiple);
            }),
        );
    });
});

describe("layoutSegments (design section 10.3 worked numbers)", () => {
    // rowPtr, colIdx, weights, arcToEdge, edgeToArc: n = 100,000, arcCount = 2,000,000, edgeCount = 1,000,000
    it("lays out the undirected weighted benchmark arena", () => {
        const layout = layoutSegments([400004, 8000000, 8000000, 8000000, 4000000]);
        expect(layout.offsets).toEqual([0, 400128, 8400128, 16400128, 24400128]);
        expect(layout.ends).toEqual([400004, 8400128, 16400128, 24400128, 28400128]);
        expect(layout.byteLength).toBe(28400128);
        expect(layout.padding).toBe(124);
        // hotByteLength is the end of the weights segment
        expect(layout.ends[2]).toBe(16400128);
    });

    it("lays out the directed weighted arena with materialised permutations", () => {
        const layout = layoutSegments([400004, 4000000, 4000000, 4000000, 4000000]);
        expect(layout.offsets).toEqual([0, 400128, 4400128, 8400128, 12400128]);
        expect(layout.byteLength).toBe(16400128);
        expect(layout.ends[2]).toBe(8400128);
        expect(layout.padding).toBe(124);
    });

    it("keeps identity permutations out of the arena (null segments occupy nothing)", () => {
        const layout = layoutSegments([400004, 4000000, 4000000, 0, 0]);
        expect(layout.offsets).toEqual([0, 400128, 4400128, null, null]);
        expect(layout.ends).toEqual([400004, 4400128, 8400128, null, null]);
        expect(layout.byteLength).toBe(8400128);
    });

    it("skips an absent weights segment without disturbing later offsets' alignment", () => {
        const undirected = layoutSegments([400004, 8000000, 0, 8000000, 4000000]);
        expect(undirected.offsets).toEqual([0, 400128, null, 8400128, 16400128]);
        expect(undirected.byteLength).toBe(20400128);
        const directed = layoutSegments([400004, 4000000, 0, 4000000, 4000000]);
        expect(directed.offsets).toEqual([0, 400128, null, 4400128, 8400128]);
        expect(directed.byteLength).toBe(12400128);
        // hot prefix when unweighted is the end of colIdx
        expect(directed.ends[1]).toBe(4400128);
    });

    it("handles the empty graph and the all-empty case", () => {
        const empty = layoutSegments([4, 0, 0, 0, 0]);
        expect(empty.offsets).toEqual([0, null, null, null, null]);
        expect(empty.byteLength).toBe(4);
        expect(empty.padding).toBe(0);
        const nothing = layoutSegments([0, 0, 0]);
        expect(nothing.offsets).toEqual([null, null, null]);
        expect(nothing.byteLength).toBe(0);
        expect(layoutSegments([]).byteLength).toBe(0);
    });

    it("honours a custom alignment and pads only between segments", () => {
        const layout = layoutSegments([5, 3, 9], 8);
        expect(layout.offsets).toEqual([0, 8, 16]);
        expect(layout.byteLength).toBe(25);
        expect(layout.padding).toBe(3 + 5);
    });

    it("places every non-empty segment at a multiple of the alignment, in order, without overlap", () => {
        fc.assert(
            fc.property(fc.array(fc.nat(10_000_000), { maxLength: 8 }), (lengths) => {
                const layout = layoutSegments(lengths);
                let previousEnd = 0;
                let total = 0;
                for (let i = 0; i < lengths.length; i++) {
                    const offset = layout.offsets[i];
                    if (lengths[i] === 0) {
                        expect(offset).toBeNull();
                        expect(layout.ends[i]).toBeNull();
                        continue;
                    }
                    expect(offset).not.toBeNull();
                    expect(offset! % ALIGNMENT).toBe(0);
                    expect(offset!).toBeGreaterThanOrEqual(previousEnd);
                    expect(offset! - previousEnd).toBeLessThan(ALIGNMENT);
                    previousEnd = offset! + lengths[i];
                    total = previousEnd;
                    expect(layout.ends[i]).toBe(previousEnd);
                }
                expect(layout.byteLength).toBe(total);
            }),
        );
    });
});

describe("4-byte rules and u8 adoption (I10, design section 5.7)", () => {
    it("isFourByteAligned tests byteOffset and byteLength", () => {
        const buffer = new ArrayBuffer(64);
        expect(isFourByteAligned(new Uint32Array(buffer, 0, 4))).toBe(true);
        expect(isFourByteAligned(new Uint8Array(buffer, 0, 8))).toBe(true);
        expect(isFourByteAligned(new Uint8Array(buffer, 0, 7))).toBe(false);
        expect(isFourByteAligned(new Uint8Array(buffer, 2, 8))).toBe(false);
        expect(isFourByteAligned(new Uint8Array(buffer, 4, 0))).toBe(true);
        expect(isFourByteAligned(new Float64Array(buffer, 8, 2))).toBe(true);
    });

    it("canViewAsPaddedU32 is exactly the design predicate", () => {
        const buffer = new ArrayBuffer(16);
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 0, 16))).toBe(true);
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 0, 13))).toBe(true); // pads to 16 <= 16
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 4, 9))).toBe(true); // 4 + 12 <= 16
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 4, 12))).toBe(true);
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 1, 8))).toBe(false); // unaligned start
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 8, 8))).toBe(true);
        expect(canViewAsPaddedU32(new Uint8Array(buffer, 12, 4))).toBe(true);
        expect(canViewAsPaddedU32(new Uint8Array(13))).toBe(false); // 0 + 16 > 13
        expect(canViewAsPaddedU32(new Uint8Array(16))).toBe(true);
        expect(canViewAsPaddedU32(new Uint8Array(0))).toBe(true);
    });

    it("paddedU32View covers the column's byte range, not the whole buffer", () => {
        const buffer = new ArrayBuffer(32);
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < 32; i++) {
            bytes[i] = i;
        }
        const column = new Uint8Array(buffer, 8, 9); // bytes 8..16, pads to 12 bytes = 3 words
        const view = paddedU32View(column);
        expect(view).toBeInstanceOf(Uint32Array);
        expect(view.buffer).toBe(buffer);
        expect(view.byteOffset).toBe(8);
        expect(view.length).toBe(3);
        expect(view.byteLength).toBe(12);
        expect(view[0]).toBe(0x0b0a0908);
        expect(view[2] & 0xff).toBe(16);
        // exact multiple: no padding word
        expect(paddedU32View(new Uint8Array(buffer, 0, 8)).length).toBe(2);
        expect(paddedU32View(new Uint8Array(0)).length).toBe(0);
    });

    it("paddedU32View throws E_COLUMN_ALIGNMENT when no zero-copy view exists", () => {
        const buffer = new ArrayBuffer(16);
        for (const bad of [new Uint8Array(buffer, 1, 8), new Uint8Array(13)]) {
            let caught: unknown;
            try {
                paddedU32View(bad);
            } catch (err) {
                caught = err;
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            const err = caught as GraphFormatError;
            expect(err.code).toBe("E_COLUMN_ALIGNMENT");
            expect(err.details.byteOffset).toBe(bad.byteOffset);
            expect(err.details.byteLength).toBe(bad.byteLength);
            expect(err.details.bufferByteLength).toBe(bad.buffer.byteLength);
        }
    });

    it("copyToPaddedStore makes an adoptable copy with zero padding bytes", () => {
        const source = new Uint8Array(new ArrayBuffer(16), 1, 13);
        source.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        expect(canViewAsPaddedU32(source)).toBe(false);
        const copy = copyToPaddedStore(source);
        expect(copy).not.toBe(source);
        expect(copy.length).toBe(13);
        expect(copy.byteOffset).toBe(0);
        expect(copy.buffer.byteLength).toBe(16);
        expect(Array.from(copy)).toEqual(Array.from(source));
        expect(canViewAsPaddedU32(copy)).toBe(true);
        const view = paddedU32View(copy);
        expect(view.length).toBe(4);
        expect(view[3]).toBe(13); // padding bytes are zero
        // the copy is independent
        source[0] = 99;
        expect(copy[0]).toBe(1);
        expect(copyToPaddedStore(new Uint8Array(0)).length).toBe(0);
    });
});

describe("typedArrayCtorOf / unsupportedDtype", () => {
    it("maps every flat dtype to its typed-array class", () => {
        const cases: readonly [TypedDtype, unknown][] = [
            ["f32", Float32Array],
            ["f64", Float64Array],
            ["i32", Int32Array],
            ["u32", Uint32Array],
            ["u8", Uint8Array],
            ["bool", Uint32Array],
            ["dict", Uint32Array],
        ];
        for (const [dtype, ctor] of cases) {
            expect(typedArrayCtorOf(dtype)).toBe(ctor);
        }
        let caught: unknown = null;
        try {
            unsupportedDtype("f16" as never);
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        expect((caught as GraphFormatError).code).toBe("E_UNSUPPORTED");
        expect((caught as GraphFormatError).details.dtype).toBe("f16");
    });
});

describe("staging growth arithmetic (design section 6.2)", () => {
    it("growCapacity doubles, covers the need and rounds to 16 elements", () => {
        expect(STAGING_ELEMENT_GRANULE).toBe(16);
        expect(growCapacity(0, 1)).toBe(16);
        expect(growCapacity(16, 17)).toBe(32);
        expect(growCapacity(32, 33)).toBe(64);
        expect(growCapacity(16, 100)).toBe(112);
        expect(growCapacity(16, 97)).toBe(112);
        expect(growCapacity(0, 1000)).toBe(1008);
        expect(growCapacity(5, 6)).toBe(16);
        expect(growCapacity(1000, 1001)).toBe(2000);
    });

    it("stagingByteLength rounds the buffer to 64 bytes and stays a whole number of elements", () => {
        expect(STAGING_BYTE_GRANULE).toBe(64);
        expect(stagingByteLength(16, 4)).toBe(64);
        expect(stagingByteLength(16, 1)).toBe(64);
        expect(stagingByteLength(16, 8)).toBe(128);
        expect(stagingByteLength(17, 4)).toBe(128);
        expect(stagingByteLength(0, 4)).toBe(0);
        for (const bytesPerElement of [1, 4, 8]) {
            for (let capacity = 0; capacity < 200; capacity += 7) {
                const bytes = stagingByteLength(capacity, bytesPerElement);
                expect(bytes % 64).toBe(0);
                expect(bytes % bytesPerElement).toBe(0);
                expect(bytes).toBeGreaterThanOrEqual(capacity * bytesPerElement);
            }
        }
    });
});

describe("allocateStagingBuffer", () => {
    it("returns a plain fixed-length buffer when resizable is false", () => {
        const buffer = allocateStagingBuffer(64, false);
        expect(buffer.byteLength).toBe(64);
        expect(buffer.resizable).toBe(false);
        expect(buffer.maxByteLength).toBe(64);
    });

    it("returns a resizable buffer with the reservation when resizable is true", () => {
        const buffer = allocateStagingBuffer(64, true);
        expect(buffer.byteLength).toBe(64);
        expect(buffer.resizable).toBe(true);
        expect(buffer.maxByteLength).toBe(STAGING_RESERVE_BYTES);
        const bigger = allocateStagingBuffer(STAGING_RESERVE_BYTES + 64, true);
        expect(bigger.maxByteLength).toBe(STAGING_RESERVE_BYTES + 64);
        const small = allocateStagingBuffer(64, true, 1024);
        expect(small.maxByteLength).toBe(1024);
        expect(allocateStagingBuffer(0, true).byteLength).toBe(0);
    });

    it("falls back to a reservation of exactly byteLength when the engine refuses the reservation", () => {
        const buffer = allocateStagingBuffer(64, true, Number.MAX_SAFE_INTEGER);
        expect(buffer.resizable).toBe(true);
        expect(buffer.byteLength).toBe(64);
        expect(buffer.maxByteLength).toBe(64);
    });

    it("propagates a RangeError that is not about the reservation", () => {
        expect(() => allocateStagingBuffer(-1, true)).toThrow(RangeError);
        expect(() => allocateStagingBuffer(-1, false)).toThrow(RangeError);
    });
});

describe("growTypedArray", () => {
    it("returns the same array when it already fits", () => {
        const array = new Uint32Array(allocateStagingBuffer(64, true), 0, 16);
        expect(growTypedArray(array, Uint32Array, 16, true)).toBe(array);
        expect(growTypedArray(array, Uint32Array, 0, true)).toBe(array);
    });

    it("grows in place, keeping the same buffer, when the buffer is resizable and has room", () => {
        const buffer = allocateStagingBuffer(64, true);
        const array = new Uint32Array(buffer, 0, 16);
        array.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        const grown = growTypedArray(array, Uint32Array, 17, true);
        expect(grown).not.toBe(array);
        expect(grown.buffer).toBe(buffer);
        expect(buffer.byteLength).toBe(128);
        expect(grown.length).toBe(32);
        expect(Array.from(grown.subarray(0, 16))).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        expect(Array.from(grown.subarray(16))).toEqual(new Array<number>(16).fill(0));
        // the old view still sees the same memory
        grown[0] = 42;
        expect(array[0]).toBe(42);
    });

    it("copies into a fresh fixed-length buffer on the copy path", () => {
        const array = new Float64Array(allocateStagingBuffer(128, false), 0, 16);
        array.fill(1.5);
        const grown = growTypedArray(array, Float64Array, 17, false);
        expect(grown.buffer).not.toBe(array.buffer);
        expect(grown.buffer.resizable).toBe(false);
        expect(grown.length).toBe(32);
        expect(grown.buffer.byteLength).toBe(256);
        expect(Array.from(grown.subarray(0, 16))).toEqual(new Array<number>(16).fill(1.5));
        expect(Array.from(grown.subarray(16))).toEqual(new Array<number>(16).fill(0));
    });

    it("reallocates a resizable buffer that has outgrown its reservation and copies the contents", () => {
        const buffer = allocateStagingBuffer(64, true, 64); // maxByteLength 64: no room to grow in place
        const array = new Uint8Array(buffer, 0, 64);
        for (let i = 0; i < 64; i++) {
            array[i] = i;
        }
        const grown = growTypedArray(array, Uint8Array, 65, true);
        expect(grown.buffer).not.toBe(buffer);
        expect(grown.buffer.resizable).toBe(true);
        expect(grown.buffer.maxByteLength).toBeGreaterThanOrEqual(STAGING_RESERVE_BYTES);
        expect(grown.length).toBe(128);
        expect(Array.from(grown.subarray(0, 64))).toEqual(Array.from(array));
    });

    it("copies rather than resizes when the array does not start at byteOffset 0", () => {
        const buffer = allocateStagingBuffer(128, true);
        const array = new Uint32Array(buffer, 64, 16);
        array.fill(7);
        const grown = growTypedArray(array, Uint32Array, 17, true);
        expect(grown.buffer).not.toBe(buffer);
        expect(grown.byteOffset).toBe(0);
        expect(Array.from(grown.subarray(0, 16))).toEqual(new Array<number>(16).fill(7));
    });

    it("grows an empty array to the first granule on both paths", () => {
        for (const resizable of [true, false]) {
            const empty = new Uint32Array(allocateStagingBuffer(0, resizable), 0, 0);
            const grown = growTypedArray(empty, Uint32Array, 1, resizable);
            expect(grown.length).toBe(16);
            expect(grown.buffer.byteLength).toBe(64);
            expect(grown.buffer.resizable).toBe(resizable);
        }
    });

    it("keeps doubling: capacities are 16, 32, 64, ... for u32 and 64, 128, ... for u8", () => {
        let u32 = new Uint32Array(allocateStagingBuffer(0, true), 0, 0);
        const u32Capacities: number[] = [];
        for (let i = 0; i < 5; i++) {
            u32 = growTypedArray(u32, Uint32Array, u32.length + 1, true);
            u32Capacities.push(u32.length);
        }
        expect(u32Capacities).toEqual([16, 32, 64, 128, 256]);
        let u8 = new Uint8Array(allocateStagingBuffer(0, false), 0, 0);
        const u8Capacities: number[] = [];
        for (let i = 0; i < 4; i++) {
            u8 = growTypedArray(u8, Uint8Array, u8.length + 1, false);
            u8Capacities.push(u8.length);
        }
        expect(u8Capacities).toEqual([64, 128, 256, 512]);
    });
});
