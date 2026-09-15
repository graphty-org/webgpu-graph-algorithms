import { describe, expect, it } from "vitest";

import { columnFromTypedArray, columnFromValues } from "../../src/columns/column.js";
import { contentHashOf, Fnv1aHasher, hashColumn, hashTypedArray } from "../../src/snapshot/hash.js";
import { makeSnapshot } from "../helpers/parts.js";

const HEX16 = /^[0-9a-f]{16}$/;

describe("Fnv1aHasher", () => {
    it("starts from the FNV bases and renders 16 lowercase hex characters", () => {
        expect(new Fnv1aHasher().digest()).toBe("811c9dc5cbf29ce4");
        expect(new Fnv1aHasher().word(1).digest()).toMatch(HEX16);
    });

    it("lane A is word-wise 32-bit FNV-1a", () => {
        // one word 0: (basis ^ 0) * prime mod 2^32
        const expected = (Math.imul(0x811c9dc5, 0x01000193) >>> 0).toString(16).padStart(8, "0");
        expect(new Fnv1aHasher().word(0).digest().slice(0, 8)).toBe(expected);
    });

    it("is deterministic and sensitive to every word, its order and the length framing", () => {
        const a = new Fnv1aHasher().words(new Uint32Array([1, 2, 3])).digest();
        expect(new Fnv1aHasher().words(new Uint32Array([1, 2, 3])).digest()).toBe(a);
        expect(new Fnv1aHasher().words(new Uint32Array([1, 3, 2])).digest()).not.toBe(a);
        expect(new Fnv1aHasher().words(new Uint32Array([1, 2, 3, 0])).digest()).not.toBe(a);
        expect(
            new Fnv1aHasher()
                .words(new Uint32Array([1, 2]))
                .words(new Uint32Array([3]))
                .digest(),
        ).not.toBe(a);
    });

    it("the two lanes are not the same function of the input", () => {
        const digest = new Fnv1aHasher().words(new Uint32Array([5, 6, 7])).digest();
        expect(digest.slice(0, 8)).not.toBe(digest.slice(8));
    });

    it("identity(n) hashes exactly like words(0..n-1)", () => {
        const explicit = new Fnv1aHasher().words(new Uint32Array([0, 1, 2, 3, 4])).digest();
        expect(new Fnv1aHasher().identity(5).digest()).toBe(explicit);
        expect(new Fnv1aHasher().identity(0).digest()).toBe(new Fnv1aHasher().words(new Uint32Array(0)).digest());
    });

    it("bytes() reads aligned views as words and packs unaligned ones the same way", () => {
        const buffer = new ArrayBuffer(12);
        const bytes = new Uint8Array(buffer);
        bytes.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const aligned = hashTypedArray(new Uint8Array(buffer, 0, 8));
        const copy = hashTypedArray(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
        expect(aligned).toBe(copy);
        // an unaligned view of the same bytes hashes identically to an aligned copy of them
        const unaligned = hashTypedArray(new Uint8Array(buffer, 1, 8));
        expect(unaligned).toBe(hashTypedArray(new Uint8Array([2, 3, 4, 5, 6, 7, 8, 9])));
        // an odd length packs the tail zero-padded and differs from the padded aligned array
        const odd = hashTypedArray(new Uint8Array([1, 2, 3, 4, 5]));
        expect(odd).not.toBe(hashTypedArray(new Uint8Array([1, 2, 3, 4, 5, 0, 0, 0])));
        expect(odd).toMatch(HEX16);
    });

    it("text() mixes the code units", () => {
        expect(new Fnv1aHasher().text("ab").digest()).not.toBe(new Fnv1aHasher().text("ba").digest());
        expect(new Fnv1aHasher().text("").digest()).toBe(new Fnv1aHasher().word(0).digest());
    });
});

describe("hashColumn", () => {
    it("covers data, validity and per-dtype extras", () => {
        const a = columnFromTypedArray("node", 3, "x", new Float32Array([1, 2, 3]), {});
        const b = columnFromTypedArray("node", 3, "x", new Float32Array([1, 2, 4]), {});
        expect(hashColumn(a)).toMatch(HEX16);
        expect(hashColumn(a)).not.toBe(hashColumn(b));
        const dict = columnFromValues("node", 3, "d", ["p", "q", "p"], { dtype: "dict" });
        const dict2 = columnFromValues("node", 3, "d", ["p", "r", "p"], { dtype: "dict" });
        expect(hashColumn(dict)).not.toBe(hashColumn(dict2));
        const str = columnFromValues("node", 2, "s", ["hello", "world"], { dtype: "string" });
        expect(hashColumn(str)).toBe(
            hashColumn(columnFromValues("node", 2, "s", ["hello", "world"], { dtype: "string" })),
        );
        const list = columnFromValues("node", 2, "l", [[1, 2], [3]], { dtype: "list", itemDtype: "u32" });
        expect(hashColumn(list)).not.toBe(
            hashColumn(columnFromValues("node", 2, "l", [[1], [2, 3]], { dtype: "list", itemDtype: "u32" })),
        );
        const json = columnFromValues("node", 2, "j", [{ a: 1 }, null], { dtype: "json" });
        expect(hashColumn(json)).not.toBe(
            hashColumn(columnFromValues("node", 2, "j", [{ a: 2 }, null], { dtype: "json" })),
        );
        const nullable = columnFromValues("node", 2, "n", [1, undefined], { dtype: "u32" });
        expect(hashColumn(nullable)).not.toBe(hashColumn(columnFromValues("node", 2, "n", [1, 0], { dtype: "u32" })));
        const bool = columnFromValues("node", 2, "b", [true, false], { dtype: "bool" });
        expect(hashColumn(bool)).toMatch(HEX16);
    });
});

describe("contentHashOf", () => {
    it("is stable, cached, and independent of whether identity getters were touched (P11)", () => {
        const a = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
            ],
        });
        const b = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
            ],
        });
        expect(a.flags.arcToEdgeIsIdentity).toBe(true);
        const before = a.contentHash();
        expect(before).toMatch(HEX16);
        expect(a.arcToEdge.length).toBe(3);
        expect(a.edgeToArc.length).toBe(3);
        expect(contentHashOf(a)).toBe(before);
        expect(a.contentHash()).toBe(before);
        expect(b.contentHash()).toBe(before);
    });

    it("distinguishes directedness, weights and topology", () => {
        const base = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1],
                    [1, 2],
                ],
            }).contentHash(),
        ).not.toBe(base.contentHash());
        expect(
            makeSnapshot({
                directed: false,
                edges: [
                    [0, 1],
                    [1, 2, 2],
                ],
            }).contentHash(),
        ).not.toBe(base.contentHash());
        expect(
            makeSnapshot({
                directed: false,
                edges: [
                    [0, 1],
                    [0, 2],
                ],
            }).contentHash(),
        ).not.toBe(base.contentHash());
        expect(
            makeSnapshot({
                directed: false,
                edges: [
                    [0, 1],
                    [1, 2],
                ],
                arena: false,
            }).contentHash(),
        ).toBe(base.contentHash());
        expect(
            makeSnapshot({
                directed: false,
                edges: [
                    [0, 1],
                    [1, 2],
                ],
                ids: ["a", "b", "c"],
            }).contentHash(),
        ).toBe(base.contentHash());
    });

    it("hashes the empty snapshot", () => {
        expect(makeSnapshot({ directed: true, nodeCount: 0, edges: [] }).contentHash()).toMatch(HEX16);
    });
});
