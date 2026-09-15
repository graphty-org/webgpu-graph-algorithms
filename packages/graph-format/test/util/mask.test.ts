import { describe, expect, it } from "vitest";

import { GraphFormatError } from "../../src/errors.js";
import { type EdgeMask, type NodeMask } from "../../src/types/index.js";
import { checkMaskLength, makeMask, maskCount, maskSet, maskTest, maskToIndices } from "../../src/util/mask.js";

/** The kernel read pattern of design section 7.4 / 10.2. */
function kernelRead(mask: Uint32Array, i: number): number {
    return (mask[i >>> 5] >>> (i & 31)) & 1;
}

describe("makeMask", () => {
    it("allocates ceil(length / 32) clear words", () => {
        for (const [length, words] of [
            [0, 0],
            [1, 1],
            [31, 1],
            [32, 1],
            [33, 2],
            [64, 2],
            [65, 3],
            [1000, 32],
        ] as const) {
            const mask = makeMask(length);
            expect(mask).toBeInstanceOf(Uint32Array);
            expect(mask.length).toBe(words);
            expect(maskCount(mask, length)).toBe(0);
        }
    });

    it("is a plain-ArrayBuffer U32 usable as NodeMask and EdgeMask", () => {
        const nodes: NodeMask = makeMask(10);
        const edges: EdgeMask = makeMask(10);
        expect(nodes.buffer).toBeInstanceOf(ArrayBuffer);
        expect(edges.byteLength % 4).toBe(0);
    });

    it("fill sets every index below length and none beyond", () => {
        for (const length of [1, 31, 32, 33, 64, 65, 100]) {
            const mask = makeMask(length, true);
            expect(maskCount(mask, length)).toBe(length);
            for (let i = 0; i < length; i++) {
                expect(maskTest(mask, i)).toBe(true);
            }
            for (let i = length; i < mask.length * 32; i++) {
                expect(kernelRead(mask, i)).toBe(0);
            }
        }
        expect(Array.from(makeMask(33, true))).toEqual([0xffffffff, 1]);
        expect(maskCount(makeMask(0, true), 0)).toBe(0);
        expect(maskCount(makeMask(5, false), 5)).toBe(0);
    });
});

describe("maskTest / maskSet", () => {
    it("include and exclude indices with the validity bit layout", () => {
        const mask = makeMask(70);
        maskSet(mask, 0, true);
        maskSet(mask, 31, true);
        maskSet(mask, 32, true);
        maskSet(mask, 69, true);
        expect(Array.from(mask)).toEqual([0x80000001, 1, 0x20]);
        expect(maskTest(mask, 0)).toBe(true);
        expect(maskTest(mask, 1)).toBe(false);
        expect(maskTest(mask, 31)).toBe(true);
        expect(maskTest(mask, 32)).toBe(true);
        expect(maskTest(mask, 33)).toBe(false);
        expect(maskTest(mask, 69)).toBe(true);
        for (let i = 0; i < 70; i++) {
            expect(maskTest(mask, i)).toBe(kernelRead(mask, i) === 1);
        }
        maskSet(mask, 31, false);
        maskSet(mask, 69, false);
        expect(Array.from(mask)).toEqual([1, 1, 0]);
        // setting an already-set bit or clearing a clear one is idempotent
        maskSet(mask, 0, true);
        maskSet(mask, 2, false);
        expect(Array.from(mask)).toEqual([1, 1, 0]);
    });
});

describe("maskCount / maskToIndices", () => {
    it("count only the bits below length (the loop guard of an alive mask)", () => {
        const mask = makeMask(64, true);
        expect(maskCount(mask, 64)).toBe(64);
        expect(maskCount(mask, 33)).toBe(33);
        expect(maskCount(mask, 32)).toBe(32);
        expect(maskCount(mask, 1)).toBe(1);
        expect(maskCount(mask, 0)).toBe(0);
        maskSet(mask, 10, false);
        expect(maskCount(mask, 64)).toBe(63);
        expect(maskCount(mask, 11)).toBe(10);
        expect(maskCount(mask, 10)).toBe(10);
    });

    it("maskToIndices lists included indices ascending as a fresh U32", () => {
        const mask = makeMask(100);
        for (const i of [99, 0, 64, 31, 32, 63, 33]) {
            maskSet(mask, i, true);
        }
        const indices = maskToIndices(mask, 100);
        expect(indices).toBeInstanceOf(Uint32Array);
        expect(Array.from(indices)).toEqual([0, 31, 32, 33, 63, 64, 99]);
        expect(Array.from(maskToIndices(mask, 64))).toEqual([0, 31, 32, 33, 63]);
        expect(Array.from(maskToIndices(mask, 33))).toEqual([0, 31, 32]);
        expect(maskToIndices(mask, 0).length).toBe(0);
        expect(maskToIndices(makeMask(50), 50).length).toBe(0);
        const full = maskToIndices(makeMask(70, true), 70);
        expect(Array.from(full)).toEqual(Array.from({ length: 70 }, (_, i) => i));
    });

    it("an algorithm-owned alive mask peels correctly with count as the guard", () => {
        // Girvan-Newman style: remove edges one by one, stop when the mask is empty.
        const edgeCount = 40;
        const alive = makeMask(edgeCount, true);
        let removed = 0;
        while (maskCount(alive, edgeCount) > 0) {
            const next = maskToIndices(alive, edgeCount)[0];
            maskSet(alive, next, false);
            removed++;
        }
        expect(removed).toBe(edgeCount);
        expect(Array.from(alive)).toEqual([0, 0]);
    });
});

describe("checkMaskLength", () => {
    it("accepts a mask with at least ceil(length / 32) words", () => {
        expect(() => checkMaskLength(makeMask(0), 0, "edges")).not.toThrow();
        expect(() => checkMaskLength(makeMask(32), 32, "edges")).not.toThrow();
        expect(() => checkMaskLength(makeMask(33), 33, "edges")).not.toThrow();
        expect(() => checkMaskLength(makeMask(64), 33, "edges")).not.toThrow();
        expect(() => checkMaskLength(new Uint32Array(0), 0, "nodes")).not.toThrow();
    });

    it("throws E_MASK_LENGTH with the found and required word counts for a short mask", () => {
        for (const [words, length] of [
            [0, 1],
            [1, 33],
            [2, 65],
            [31, 1000],
        ] as const) {
            let caught: unknown;
            try {
                checkMaskLength(new Uint32Array(words), length, "edges");
            } catch (err) {
                caught = err;
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            const err = caught as GraphFormatError;
            expect(err.code).toBe("E_MASK_LENGTH");
            expect(err.details).toEqual({ found: words, required: Math.ceil(length / 32), length });
            expect(err.message).toContain("edges");
            expect(err.message).toContain(String(words));
            expect(err.message).toContain(String(Math.ceil(length / 32)));
        }
    });
});
