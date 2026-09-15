import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    gatherNodeIdMap,
    identityNodeIdMap,
    NodeIdMap,
    nodeIdMapFromF64,
    nodeIdMapFromIds,
    nodeIdMapFromTyped,
    nodeIdMapToTyped,
    type NodeIdMapTypedParts,
    remapNodeIdMap,
    validateNodeId,
} from "../../src/ids/node-id-map.js";
import { type F64, type NodeId, type NodeIdMapKind, type U8, type U32 } from "../../src/types/index.js";

const LONE_HIGH = String.fromCharCode(0xd83d);
const SMILE = String.fromCharCode(0xd83d, 0xde00);
const E_ACUTE = String.fromCharCode(0xe9);

function captureError(fn: () => unknown): GraphFormatError {
    try {
        fn();
    } catch (err) {
        expect(err).toBeInstanceOf(GraphFormatError);
        return err as GraphFormatError;
    }
    throw new Error("expected a GraphFormatError");
}

/** The kind selection rules of design section 4.2, written independently of the implementation. */
function expectedKind(ids: readonly NodeId[]): NodeIdMapKind {
    const n = ids.length;
    if (n === 0) {
        return "identity";
    }
    const allNumbers = ids.every((v) => typeof v === "number");
    const allStrings = ids.every((v) => typeof v === "string");
    if (allNumbers) {
        const numbers = ids.map(Number);
        const offset = numbers[0];
        if (Number.isSafeInteger(offset) && numbers.every((v, i) => v === i + offset)) {
            return "identity";
        }
        if (numbers.every((v) => Number.isInteger(v) && v >= 0 && v < MAX_COUNT)) {
            const maxId = numbers.reduce((a, b) => Math.max(a, b), -1);
            if (maxId + 1 <= 2 * n) {
                return "dense";
            }
        }
        return "numeric";
    }
    if (allStrings) {
        return "string";
    }
    return "mixed";
}

/** -0 is stored as 0. */
function normalise(ids: readonly NodeId[]): NodeId[] {
    return ids.map((v) => (v === 0 ? 0 : v));
}

/** Every public accessor agrees with the given ids (invariant I11 plus the bulk helpers). */
function expectBijection(map: NodeIdMap, ids: readonly NodeId[]): void {
    const expected = normalise(ids);
    expect(map.size).toBe(expected.length);
    expect(map.toArray()).toEqual(expected);
    expect([...map]).toEqual(expected);
    expect(map.idsSlice()).toEqual(expected);
    for (let i = 0; i < expected.length; i++) {
        const id = expected[i];
        expect(map.idOf(i)).toBe(id);
        expect(map.indexOf(id)).toBe(i);
        expect(map.has(id)).toBe(true);
        expect(map.requireIndex(id)).toBe(i);
    }
    expect(map.indexOf("no such id")).toBe(INVALID_INDEX);
    expect(map.indexOf(-123456.5)).toBe(INVALID_INDEX);
    expect(map.indexOf(Number.NaN)).toBe(INVALID_INDEX);
    expect(map.has("no such id")).toBe(false);
    expect(Array.from(map.indicesOf(expected))).toEqual(expected.map((_, i) => i));
}

const numberId = fc.oneof(
    fc.integer({ min: -3, max: 12 }),
    fc.integer(),
    fc.double({ noNaN: true, noDefaultInfinity: true }),
    fc.constantFrom(
        0,
        -0,
        1,
        1.5,
        -1,
        2 ** 40,
        Number.MAX_SAFE_INTEGER,
        -Number.MAX_SAFE_INTEGER,
        0.1,
        MAX_COUNT,
        MAX_COUNT - 1,
    ),
);
const stringId = fc.oneof(
    fc.string(),
    fc.string({ unit: "binary" }),
    fc.constantFrom("1", "0", "1.0", "01", "-1", "", "__proto__", "constructor", SMILE, E_ACUTE),
    fc.integer().map(String),
);
const anyId = fc.oneof(numberId, stringId);
const uniqueIds = fc.uniqueArray(anyId, { comparator: "SameValueZero", maxLength: 60 });
const uniqueNumbers = fc.uniqueArray(numberId, { comparator: "SameValueZero", maxLength: 60 });

describe("validateNodeId / isValidNodeId", () => {
    it("truncates a long string in the message", () => {
        const long = `${"x".repeat(100)}${LONE_HIGH}`;
        const err = captureError(() => validateNodeId(long));
        expect(err.message).toContain("...");
        expect(err.message.length).toBeLessThan(120);
    });
});

describe("identity maps", () => {
    it("has no storage and computes both directions arithmetically", () => {
        const map = identityNodeIdMap(4);
        expect(map.kind).toBe("identity");
        expect(map.size).toBe(4);
        expect(map.offset).toBe(0);
        expect(map.byteLength()).toBe(0);
        expectBijection(map, [0, 1, 2, 3]);
        expect(Object.isFrozen(map)).toBe(true);
    });

    it("covers 1-based files with offset 1 and any safe integer offset", () => {
        const one = identityNodeIdMap(3, 1);
        expect(one.offset).toBe(1);
        expectBijection(one, [1, 2, 3]);
        expect(one.indexOf(0)).toBe(INVALID_INDEX);
        expect(one.indexOf(4)).toBe(INVALID_INDEX);
        const negative = identityNodeIdMap(3, -5);
        expectBijection(negative, [-5, -4, -3]);
        expect(negative.indexOf(0)).toBe(INVALID_INDEX);
    });

    it("misses strings, non-integers, out-of-range numbers and NaN, and returns +0 for -0", () => {
        const map = identityNodeIdMap(3);
        expect(map.indexOf("1")).toBe(INVALID_INDEX);
        expect(map.indexOf("0")).toBe(INVALID_INDEX);
        expect(map.indexOf(1.5)).toBe(INVALID_INDEX);
        expect(map.indexOf(-1)).toBe(INVALID_INDEX);
        expect(map.indexOf(3)).toBe(INVALID_INDEX);
        expect(map.indexOf(Number.NaN)).toBe(INVALID_INDEX);
        expect(map.indexOf(Number.POSITIVE_INFINITY)).toBe(INVALID_INDEX);
        expect(Object.is(map.indexOf(-0), 0)).toBe(true);
        expect(map.indexOf(1.0)).toBe(1);
        expect(map.has("1")).toBe(false);
    });

    it("handles the empty map", () => {
        const map = identityNodeIdMap(0);
        expect(map.size).toBe(0);
        expect(map.toArray()).toEqual([]);
        expect([...map]).toEqual([]);
        expect(map.indexOf(0)).toBe(INVALID_INDEX);
        expect(map.toMap([]).size).toBe(0);
        expect(map.stringIndex().size).toBe(0);
    });

    it("rejects a bad size or offset", () => {
        expect(captureError(() => identityNodeIdMap(-1)).code).toBe("E_TOO_LARGE");
        expect(captureError(() => identityNodeIdMap(1.5)).code).toBe("E_TOO_LARGE");
        expect(captureError(() => identityNodeIdMap(MAX_COUNT + 1)).code).toBe("E_TOO_LARGE");
        expect(captureError(() => identityNodeIdMap(2, 0.5)).code).toBe("E_INVALID_ID");
        expect(captureError(() => identityNodeIdMap(2, 2 ** 53)).code).toBe("E_INVALID_ID");
    });
});

describe("public accessors", () => {
    const ids: NodeId[] = ["a", 1, "1", 2.5, "__proto__"];
    const map = nodeIdMapFromIds(ids);

    it("idOf throws E_INDEX_RANGE for any index outside [0, size)", () => {
        for (const index of [5, -1, 0.5, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 32]) {
            const err = captureError(() => map.idOf(index));
            expect(err.code).toBe("E_INDEX_RANGE");
            expect(err.details).toMatchObject({ index, size: 5 });
        }
    });

    it("requireIndex throws E_UNKNOWN_NODE naming the id", () => {
        const err = captureError(() => map.requireIndex("zzz"));
        expect(err.code).toBe("E_UNKNOWN_NODE");
        expect(err.details).toEqual({ id: "zzz" });
        expect(err.message).toContain('"zzz"');
        expect(captureError(() => map.requireIndex(99)).message).toContain("99");
    });

    it("indicesOf resolves arrays, sets and generators, writing INVALID_INDEX or throwing", () => {
        expect(Array.from(map.indicesOf(["1", 1, "nope", 2.5]))).toEqual([2, 1, INVALID_INDEX, 3]);
        expect(Array.from(map.indicesOf(new Set(["a", "__proto__"])))).toEqual([0, 4]);
        function* many(): Generator<NodeId> {
            for (let i = 0; i < 40; i++) {
                yield i % 2 === 0 ? "a" : 1;
            }
        }
        const result = map.indicesOf(many());
        expect(result).toBeInstanceOf(Uint32Array);
        expect(result.length).toBe(40);
        expect(result.buffer.byteLength).toBe(40 * 4);
        expect(Array.from(result)).toEqual(Array.from({ length: 40 }, (_, i) => (i % 2 === 0 ? 0 : 1)));
        expect(Array.from(map.indicesOf([]))).toEqual([]);
        const err = captureError(() => map.indicesOf(["a", "missing"], "throw"));
        expect(err.code).toBe("E_UNKNOWN_NODE");
        expect(Array.from(map.indicesOf(["a"], "throw"))).toEqual([0]);
    });

    it("idsSlice follows Array.prototype.slice bounds", () => {
        expect(map.idsSlice(1, 3)).toEqual([1, "1"]);
        expect(map.idsSlice(-2)).toEqual([2.5, "__proto__"]);
        expect(map.idsSlice(0, -3)).toEqual(["a", 1]);
        expect(map.idsSlice(4, 2)).toEqual([]);
        expect(map.idsSlice(10)).toEqual([]);
        expect(map.toArray()).not.toBe(map.toArray());
    });

    it("toMap keys by id in index order", () => {
        const result = map.toMap(new Float64Array([10, 11, 12, 13, 14]));
        expect([...result.entries()]).toEqual([
            ["a", 10],
            [1, 11],
            ["1", 12],
            [2.5, 13],
            ["__proto__", 14],
        ]);
    });

    it("toStringMap / toRecord key by String(id) with the higher index winning (legacy assignment order); stringIndex prefers the string id", () => {
        const values = ["v0", "v1", "v2", "v3", "v4"];
        const stringMap = map.toStringMap(values);
        expect([...stringMap.entries()]).toEqual([
            ["a", "v0"],
            ["1", "v2"],
            ["2.5", "v3"],
            ["__proto__", "v4"],
        ]);
        const record = map.toRecord(values);
        expect(Object.getPrototypeOf(record)).toBeNull();
        expect(Object.keys(record)).toEqual(["1", "a", "2.5", "__proto__"]);
        expect(record["1"]).toBe("v2");
        expect(Object.getOwnPropertyDescriptor(record, "__proto__")?.value).toBe("v4");
        const index = map.stringIndex();
        expect(index).toBe(map.stringIndex());
        expect([...index.entries()]).toEqual([
            ["a", 0],
            ["1", 2],
            ["2.5", 3],
            ["__proto__", 4],
        ]);
        // the string id wins whatever its position: a legacy string parameter "1" named the string node
        const reversed = nodeIdMapFromIds(["1", 1]);
        expect(reversed.stringIndex().get("1")).toBe(0);
        expect(reversed.toRecord(["s", "n"])["1"]).toBe("n");
    });

    it("entries pairs ids with values in index order", () => {
        expect([...map.entries([0, 1, 2, 3, 4])]).toEqual([
            ["a", 0],
            [1, 1],
            ["1", 2],
            [2.5, 3],
            ["__proto__", 4],
        ]);
    });
});

describe("kind detection from decoded ids (design section 4.2)", () => {
    const cases: [readonly NodeId[], NodeIdMapKind, number][] = [
        [[], "identity", 0],
        [[0], "identity", 0],
        [[0, 1, 2], "identity", 0],
        [[1, 2, 3], "identity", 1],
        [[5, 6, 7], "identity", 5],
        [[-2, -1, 0], "identity", -2],
        [[0, 2, 1], "dense", 0],
        [[3, 0], "dense", 0],
        [[0, 3], "dense", 0],
        [[0, 4], "numeric", 0],
        [[7], "identity", 7],
        [[1], "identity", 1],
        [[1.5, 2], "numeric", 0],
        [[-1, 0], "identity", -1],
        [[0.5, 1.5], "numeric", 0],
        [[2 ** 40, 0], "numeric", 0],
        [[MAX_COUNT, 0], "numeric", 0],
        [[MAX_COUNT - 1, 0], "numeric", 0],
        [["a", "b"], "string", 0],
        [["0", "1"], "string", 0],
        [[1, "1"], "mixed", 0],
        [["1", 1], "mixed", 0],
        [[0, 1, 2, "x"], "mixed", 0],
    ];

    it.each(cases)("%j is %s (offset %d)", (ids, kind, offset) => {
        const map = nodeIdMapFromIds(ids);
        expect(map.kind).toBe(kind);
        expect(map.offset).toBe(offset);
        expect(expectedKind(ids)).toBe(kind);
        expectBijection(map, ids);
    });

    it("has no partial fast path: a thousand dense ids followed by a string is mixed", () => {
        const ids: NodeId[] = Array.from({ length: 1000 }, (_, i) => i);
        ids.push("s");
        const map = nodeIdMapFromIds(ids);
        expect(map.kind).toBe("mixed");
        expectBijection(map, ids);
    });

    it("chooses dense only while the inverse would be at most twice the node count", () => {
        expect(nodeIdMapFromIds([0, 1, 2, 5]).kind).toBe("dense");
        expect(nodeIdMapFromIds([0, 1, 2, 7]).kind).toBe("dense");
        expect(nodeIdMapFromIds([0, 1, 2, 8]).kind).toBe("numeric");
        expect(nodeIdMapFromIds([10, 11, 12, 13]).kind).toBe("identity");
        expect(nodeIdMapFromIds([10, 12, 11, 13]).kind).toBe("numeric");
    });

    it("follows the rules for random unique id arrays", () => {
        fc.assert(
            fc.property(uniqueIds, (ids) => {
                const map = nodeIdMapFromIds(ids);
                expect(map.kind).toBe(expectedKind(ids));
                expectBijection(map, ids);
            }),
            { numRuns: 300 },
        );
    });
});

describe("equality (SameValueZero)", () => {
    it("keeps 1 and '1' distinct and never coerces", () => {
        const map = nodeIdMapFromIds([1, "1", "1.0", "01"]);
        expect(map.kind).toBe("mixed");
        expect(map.indexOf(1)).toBe(0);
        expect(map.indexOf("1")).toBe(1);
        expect(map.indexOf("1.0")).toBe(2);
        expect(map.indexOf("01")).toBe(3);
        expect(map.indexOf(1.0)).toBe(0);
        expect(map.indexOf(1.5)).toBe(INVALID_INDEX);
        const strings = nodeIdMapFromIds(["1", "2"]);
        expect(strings.indexOf(1)).toBe(INVALID_INDEX);
        expect(strings.indexOf("1")).toBe(0);
        const numbers = nodeIdMapFromIds([1.5, 2.5]);
        expect(numbers.indexOf("1.5")).toBe(INVALID_INDEX);
        expect(numbers.indexOf(1.5)).toBe(0);
    });

    it("stores -0 as 0 and finds it either way in every kind", () => {
        for (const ids of [
            [-0, 1, 2],
            [-0, 5, 3],
            [-0, 0.5],
            [-0, "x"],
        ] as NodeId[][]) {
            const map = nodeIdMapFromIds(ids);
            expect(Object.is(map.idOf(0), 0)).toBe(true);
            expect(map.indexOf(-0)).toBe(0);
            expect(map.indexOf(0)).toBe(0);
        }
        expect(captureError(() => nodeIdMapFromIds([0, -0])).code).toBe("E_DUPLICATE_ID");
    });

    it("never finds NaN", () => {
        for (const ids of [[0, 1], [3, 1], [0.5], ["a"], ["a", 1]] as NodeId[][]) {
            expect(nodeIdMapFromIds(ids).indexOf(Number.NaN)).toBe(INVALID_INDEX);
        }
    });
});

describe("nodeIdMapFromIds validation", () => {
    it("rejects invalid ids with E_INVALID_ID", () => {
        expect(captureError(() => nodeIdMapFromIds([1, Number.NaN])).code).toBe("E_INVALID_ID");
        expect(captureError(() => nodeIdMapFromIds(["a", LONE_HIGH])).details).toMatchObject({
            reason: "lone surrogate",
        });
        expect(captureError(() => nodeIdMapFromIds([null as unknown as NodeId])).code).toBe("E_INVALID_ID");
        expect(captureError(() => nodeIdMapFromIds(["a"], 2)).code).toBe("E_INVALID_ID");
    });

    it("rejects duplicates with E_DUPLICATE_ID naming both indices, in every kind", () => {
        const dense = captureError(() => nodeIdMapFromIds([0, 2, 2]));
        expect(dense.code).toBe("E_DUPLICATE_ID");
        expect(dense.details).toEqual({ id: 2, indices: [1, 2] });
        expect(captureError(() => nodeIdMapFromIds([1.5, 0, 1.5])).details).toEqual({ id: 1.5, indices: [0, 2] });
        expect(captureError(() => nodeIdMapFromIds(["a", "b", "a"])).details).toEqual({ id: "a", indices: [0, 2] });
        expect(captureError(() => nodeIdMapFromIds([1, "a", 1])).details).toEqual({ id: 1, indices: [0, 2] });
        expect(captureError(() => nodeIdMapFromIds([1, 1])).code).toBe("E_DUPLICATE_ID");
    });

    it("rejects a bad size", () => {
        expect(captureError(() => nodeIdMapFromIds([0], -1)).code).toBe("E_TOO_LARGE");
        expect(captureError(() => nodeIdMapFromIds([0], 0.5)).code).toBe("E_TOO_LARGE");
        expect(captureError(() => nodeIdMapFromIds([0], MAX_COUNT + 1)).code).toBe("E_TOO_LARGE");
    });

    it("copies the array when validating and shares it when trusted", () => {
        const ids: NodeId[] = ["a", "b"];
        const validated = nodeIdMapFromIds(ids);
        const trusted = nodeIdMapFromIds(ids, 2, { validate: false });
        ids[0] = "changed";
        expect(validated.idOf(0)).toBe("a");
        expect(trusted.idOf(0)).toBe("changed");
    });

    it("reads only the first `size` entries", () => {
        const map = nodeIdMapFromIds(["a", "b", "c"], 2);
        expect(map.size).toBe(2);
        expect(map.toArray()).toEqual(["a", "b"]);
        expect(map.has("c")).toBe(false);
        expect(nodeIdMapFromIds([0, 1, 5], 2).kind).toBe("identity");
    });
});

describe("the shared builder Map (decision C17)", () => {
    function builderState(ids: NodeId[]): { ids: NodeId[]; map: Map<NodeId, number> } {
        const map = new Map<NodeId, number>();
        ids.forEach((id, i) => map.set(id, i));
        return { ids, map };
    }

    it("shares the Map and the array by reference and never exposes indices >= size", () => {
        for (const all of [
            ["a", "b", "c", "d"],
            [0.5, 1.5, 2.5, 3.5],
            ["a", 1, "b", 2],
        ] as NodeId[][]) {
            const builder = builderState(all);
            const snapshot = nodeIdMapFromIds(builder.ids, 2, { map: builder.map });
            expect(snapshot.size).toBe(2);
            expect(snapshot.toArray()).toEqual(all.slice(0, 2));
            expect(snapshot.indexOf(all[2])).toBe(INVALID_INDEX);
            expect(snapshot.has(all[3])).toBe(false);
            expect(captureError(() => snapshot.requireIndex(all[3])).code).toBe("E_UNKNOWN_NODE");
            expect(snapshot.indexOf(all[1])).toBe(1);
            builder.ids.push("later");
            builder.map.set("later", 4);
            expect(snapshot.has("later")).toBe(false);
            expect(snapshot.toArray()).toEqual(all.slice(0, 2));
            expect([...snapshot]).toEqual(all.slice(0, 2));
            expect(snapshot.stringIndex().size).toBe(2);
            expect(snapshot.toMap([1, 2]).size).toBe(2);
        }
    });

    it("drops the Map for identity and dense kinds", () => {
        const identity = builderState([0, 1, 2]);
        const identityMap = nodeIdMapFromIds(identity.ids, 3, { map: identity.map });
        expect(identityMap.kind).toBe("identity");
        identity.map.set(7, 7);
        expect(identityMap.indexOf(7)).toBe(INVALID_INDEX);
        const dense = builderState([2, 0, 1]);
        const denseMap = nodeIdMapFromIds(dense.ids, 3, { map: dense.map });
        expect(denseMap.kind).toBe("dense");
        expect(denseMap.byteLength()).toBe(3 * 4 + 3 * 4);
        dense.map.set(9, 9);
        expect(denseMap.indexOf(9)).toBe(INVALID_INDEX);
    });

    it("does not validate trusted ids", () => {
        const builder = builderState([Number.NaN]);
        const map = nodeIdMapFromIds(builder.ids, 1, { map: builder.map });
        // the id is stored exactly as the builder holds it: no normalisation, no rejection
        expect(map.kind).toBe("numeric");
        expect(map.size).toBe(1);
        expect(Number.isNaN(map.idOf(0))).toBe(true);
        expect(map.toArray()).toHaveLength(1);
    });
});

describe("nodeIdMapFromF64", () => {
    it("detects identity, dense and numeric", () => {
        expect(nodeIdMapFromF64(new Float64Array([0, 1, 2])).kind).toBe("identity");
        expect(nodeIdMapFromF64(new Float64Array([1, 2, 3])).offset).toBe(1);
        expect(nodeIdMapFromF64(new Float64Array([2, 0, 1])).kind).toBe("dense");
        expect(nodeIdMapFromF64(new Float64Array([0.5, 7])).kind).toBe("numeric");
        expect(nodeIdMapFromF64(new Float64Array([])).kind).toBe("identity");
        const sparse = new Float64Array([100, 200, -3.5]);
        const map = nodeIdMapFromF64(sparse);
        expect(map.kind).toBe("numeric");
        expectBijection(map, [100, 200, -3.5]);
        expect(nodeIdMapToTyped(map).values).toBe(sparse);
        expect(map.byteLength()).toBe(24);
    });

    it("rejects NaN, infinities and duplicates", () => {
        const nan = captureError(() => nodeIdMapFromF64(new Float64Array([1, Number.NaN])));
        expect(nan.code).toBe("E_INVALID_ID");
        expect(nan.details).toMatchObject({ index: 1, reason: "non-finite" });
        expect(captureError(() => nodeIdMapFromF64(new Float64Array([Number.POSITIVE_INFINITY]))).code).toBe(
            "E_INVALID_ID",
        );
        const dup = captureError(() => nodeIdMapFromF64(new Float64Array([0.5, 0.5])));
        expect(dup.code).toBe("E_DUPLICATE_ID");
        expect(dup.details).toEqual({ id: 0.5, indices: [0, 1] });
        expect(captureError(() => nodeIdMapFromF64(new Float64Array([3, 3, 1]))).code).toBe("E_DUPLICATE_ID");
    });

    it("normalises -0 in a copy without touching the caller's array", () => {
        const values = new Float64Array([-0, 2.5]);
        const map = nodeIdMapFromF64(values);
        expect(map.kind).toBe("numeric");
        expect(Object.is(map.idOf(0), 0)).toBe(true);
        expect(Object.is(values[0], -0)).toBe(true);
        expect(nodeIdMapToTyped(map).values).not.toBe(values);
        expect(map.indexOf(-0)).toBe(0);
    });

    it("skips validation when asked", () => {
        const map = nodeIdMapFromF64(new Float64Array([9.5, 9.5]), { validate: false });
        expect(map.kind).toBe("numeric");
        expect(map.idOf(1)).toBe(9.5);
    });

    it("agrees with the decoded-array path", () => {
        fc.assert(
            fc.property(uniqueNumbers, (numbers) => {
                const map = nodeIdMapFromF64(new Float64Array(numbers));
                expect(map.kind).toBe(expectedKind(numbers));
                expectBijection(map, numbers);
            }),
            { numRuns: 200 },
        );
    });
});

describe("typed (wire) form", () => {
    it("identity carries only the offset", () => {
        const parts = nodeIdMapToTyped(identityNodeIdMap(5, 1));
        expect(parts).toEqual({
            kind: "identity",
            size: 5,
            offset: 1,
            values: null,
            tags: null,
            numbers: null,
            offsets: null,
            utf8: null,
        });
        const back = nodeIdMapFromTyped(parts, "full");
        expectBijection(back, [1, 2, 3, 4, 5]);
    });

    it("dense carries u32 values and rebuilds the inverse on load", () => {
        const map = nodeIdMapFromIds([2, 0, 3]);
        const parts = nodeIdMapToTyped(map);
        expect(parts.kind).toBe("dense");
        expect(parts.values).toBeInstanceOf(Uint32Array);
        expect(Array.from(parts.values as Uint32Array)).toEqual([2, 0, 3]);
        expect(nodeIdMapToTyped(map).values).toBe(parts.values);
        const back = nodeIdMapFromTyped(parts, "full");
        expect(back.kind).toBe("dense");
        expectBijection(back, [2, 0, 3]);
        expect(back.byteLength()).toBe(3 * 4 + 4 * 4);
    });

    it("numeric materialises the f64 lazily from a decoded array and caches it", () => {
        const map = nodeIdMapFromIds([1.5, -2, 1e12]);
        expect(map.byteLength()).toBe(0);
        const parts = nodeIdMapToTyped(map);
        expect(parts.values).toBeInstanceOf(Float64Array);
        expect(Array.from(parts.values as Float64Array)).toEqual([1.5, -2, 1e12]);
        expect(map.byteLength()).toBe(24);
        expect(nodeIdMapToTyped(map).values).toBe(parts.values);
        const back = nodeIdMapFromTyped(parts, "full");
        expect(back.kind).toBe("numeric");
        expectBijection(back, [1.5, -2, 1e12]);
        expect(back.byteLength()).toBe(24);
    });

    it("string materialises the Utf8 store lazily and decodes lazily on the other side", () => {
        const ids = ["alpha", "", E_ACUTE, SMILE, "1"];
        const map = nodeIdMapFromIds(ids);
        expect(map.byteLength()).toBe(0);
        const parts = nodeIdMapToTyped(map);
        expect(parts.kind).toBe("string");
        expect(Array.from(parts.offsets as Uint32Array)).toEqual([0, 5, 5, 7, 11, 12]);
        expect(map.byteLength()).toBe(6 * 4 + 12);
        expect(nodeIdMapToTyped(map).utf8).toBe(parts.utf8);
        const back = nodeIdMapFromTyped(parts, "structure");
        expect(back.kind).toBe("string");
        expect(back.byteLength()).toBe(6 * 4 + 12);
        expect(back.idOf(3)).toBe(SMILE);
        expect(back.indexOf(SMILE)).toBe(3);
        expect(back.indexOf("1")).toBe(4);
        expect(back.indexOf(1)).toBe(INVALID_INDEX);
        expectBijection(back, ids);
        expect(nodeIdMapToTyped(back).offsets).toBe(parts.offsets);
    });

    it("mixed materialises tags, numbers and the Utf8 store lazily", () => {
        const ids: NodeId[] = ["a", 1.5, "", -0, "1", 2];
        const map = nodeIdMapFromIds(ids);
        expect(map.byteLength()).toBe(0);
        const parts = nodeIdMapToTyped(map);
        expect(parts.kind).toBe("mixed");
        expect(Array.from(parts.tags as Uint8Array)).toEqual([1, 0, 1, 0, 1, 0]);
        expect(Array.from(parts.numbers as Float64Array)).toEqual([0, 1.5, 0, 0, 0, 2]);
        expect(Array.from(parts.offsets as Uint32Array)).toEqual([0, 1, 1, 1, 1, 2, 2]);
        expect(map.byteLength()).toBe(6 + 6 * 8 + 7 * 4 + 2);
        expect(nodeIdMapToTyped(map).tags).toBe(parts.tags);
        const back = nodeIdMapFromTyped(parts, "full");
        expect(back.kind).toBe("mixed");
        expectBijection(back, ids);
        expect(back.idOf(1)).toBe(1.5);
        expect(back.idOf(4)).toBe("1");
        expect(back.indexOf("1")).toBe(4);
        expect(back.indexOf(1)).toBe(INVALID_INDEX);
        expect(back.idsSlice(2, 5)).toEqual(["", 0, "1"]);
        expect(nodeIdMapToTyped(back)).toEqual(parts);
    });

    it("round trips random unique ids through every kind", () => {
        fc.assert(
            fc.property(uniqueIds, fc.constantFrom<"structure" | "full">("structure", "full"), (ids, level) => {
                const map = nodeIdMapFromIds(ids);
                const parts = nodeIdMapToTyped(map);
                const back = nodeIdMapFromTyped(parts, level);
                expect(back.kind).toBe(map.kind);
                expect(back.offset).toBe(map.offset);
                expectBijection(back, ids);
                expect(back.byteLength()).toBeGreaterThanOrEqual(map.byteLength());
                expect(nodeIdMapToTyped(back).values).toBe(parts.values);
            }),
            { numRuns: 300 },
        );
    });

    it("rejects an object that is not a NodeIdMap of this module", () => {
        const fake = Object.create(NodeIdMap.prototype) as NodeIdMap;
        expect(captureError(() => nodeIdMapToTyped(fake)).code).toBe("E_UNSUPPORTED");
    });
});

describe("nodeIdMapFromTyped validation", () => {
    function parts(patch: Partial<NodeIdMapTypedParts>): NodeIdMapTypedParts {
        return {
            kind: "identity",
            size: 0,
            offset: 0,
            values: null,
            tags: null,
            numbers: null,
            offsets: null,
            utf8: null,
            ...patch,
        };
    }

    it("rejects an unknown kind with E_UNSUPPORTED", () => {
        const err = captureError(() => nodeIdMapFromTyped(parts({ kind: "sparse" as NodeIdMapKind, size: 1 })));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details).toEqual({ kind: "sparse" });
        expect(captureError(() => nodeIdMapFromTyped(parts({ kind: "sparse" as NodeIdMapKind }), "none")).code).toBe(
            "E_UNSUPPORTED",
        );
    });

    it("rejects a bad size or identity offset", () => {
        expect(captureError(() => nodeIdMapFromTyped(parts({ size: -1 }))).details).toMatchObject({ ref: "ids.size" });
        expect(captureError(() => nodeIdMapFromTyped(parts({ size: 1.5 }))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(parts({ size: MAX_COUNT + 1 }))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(parts({ size: 2, offset: 0.5 }))).details).toMatchObject({
            ref: "ids.offset",
        });
        // the O(1) offset rule holds at every level: index + offset must stay a safe integer (I11)
        expect(captureError(() => nodeIdMapFromTyped(parts({ size: 2, offset: 0.5 }), "none")).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(
            captureError(() => nodeIdMapFromTyped(parts({ size: 10, offset: Number.MAX_SAFE_INTEGER - 3 }))).details,
        ).toMatchObject({ ref: "ids.offset", size: 10 });
        expect(nodeIdMapFromTyped(parts({ size: 3, offset: -0 })).offset).toBe(0);
        expect(Object.is(nodeIdMapFromTyped(parts({ size: 3, offset: -0 })).offset, 0)).toBe(true);
    });

    it("checks dense values: type, length, range, bound and duplicates", () => {
        const dense = (values: U32 | F64 | null, size: number): NodeIdMapTypedParts =>
            parts({ kind: "dense", size, values });
        expect(captureError(() => nodeIdMapFromTyped(dense(null, 1))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(dense(new Float64Array([0]), 1))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([0, 1]), 3))).details).toMatchObject({
            ref: "ids.values",
            expected: 3,
            found: 2,
        });
        expect(captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([MAX_COUNT]), 1))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([INVALID_INDEX]), 1))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        const bound = captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([0, 4]), 2)));
        expect(bound.code).toBe("E_BAD_SERIALIZATION");
        expect(bound.details).toMatchObject({ ref: "ids.values", maxId: 4, size: 2 });
        // the dense bounds hold at every level (the inverse array is sized by the largest id)
        expect(captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([0, 4]), 2), "none")).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(nodeIdMapFromTyped(dense(new Uint32Array([0, 3]), 2), "none").indexOf(3)).toBe(1);
        const dup = captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([1, 1]), 2), "structure"));
        expect(dup.code).toBe("E_INVALID_SNAPSHOT");
        expect(dup.details).toMatchObject({ invariant: "I11", id: 1, indices: [0, 1] });
        expect(captureError(() => nodeIdMapFromTyped(dense(new Uint32Array([1, 1]), 2), "none")).code).toBe(
            "E_INVALID_SNAPSHOT",
        );
    });

    it("checks numeric values: type, length, finiteness and duplicates under full", () => {
        const numeric = (values: U32 | F64 | null, size: number): NodeIdMapTypedParts =>
            parts({ kind: "numeric", size, values });
        expect(captureError(() => nodeIdMapFromTyped(numeric(null, 1))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(numeric(new Uint32Array([0]), 1))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(captureError(() => nodeIdMapFromTyped(numeric(new Float64Array([0.5]), 2))).details).toMatchObject({
            ref: "ids.values",
        });
        const nan = new Float64Array([0.5, Number.NaN]);
        expect(nodeIdMapFromTyped(numeric(nan, 2), "structure").kind).toBe("numeric");
        const err = captureError(() => nodeIdMapFromTyped(numeric(nan, 2), "full"));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details).toMatchObject({ invariant: "I11", index: 1 });
        const dup = new Float64Array([0.5, 0.5]);
        expect(nodeIdMapFromTyped(numeric(dup, 2), "structure").kind).toBe("numeric");
        expect(captureError(() => nodeIdMapFromTyped(numeric(dup, 2), "full")).details).toMatchObject({
            invariant: "I11",
            indices: [0, 1],
        });
    });

    it("checks string layout under structure and UTF-8 plus distinctness under full", () => {
        const str = (offsets: U32 | null, utf8: U8 | null, size: number): NodeIdMapTypedParts =>
            parts({ kind: "string", size, offsets, utf8 });
        expect(captureError(() => nodeIdMapFromTyped(str(null, new Uint8Array(0), 0))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(captureError(() => nodeIdMapFromTyped(str(new Uint32Array([0]), null, 0))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        expect(
            captureError(() => nodeIdMapFromTyped(str(new Uint32Array([0, 1]), new Uint8Array(1), 2))).details,
        ).toMatchObject({
            ref: "ids.offsets",
        });
        expect(captureError(() => nodeIdMapFromTyped(str(new Uint32Array([0, 5]), new Uint8Array(1), 1))).code).toBe(
            "E_BAD_SERIALIZATION",
        );
        const bad = str(new Uint32Array([0, 1, 2]), new Uint8Array([0x61, 0xff]), 2);
        expect(nodeIdMapFromTyped(bad, "structure").idOf(1)).toBe(String.fromCharCode(0xfffd));
        const malformed = captureError(() => nodeIdMapFromTyped(bad, "full"));
        expect(malformed.code).toBe("E_INVALID_SNAPSHOT");
        expect(malformed.details).toMatchObject({ invariant: "I11", index: 1 });
        const dup = str(new Uint32Array([0, 1, 2]), new Uint8Array([0x61, 0x61]), 2);
        expect(nodeIdMapFromTyped(dup, "structure").idOf(1)).toBe("a");
        expect(captureError(() => nodeIdMapFromTyped(dup, "full")).details).toMatchObject({
            invariant: "I11",
            id: "a",
        });
        expect(nodeIdMapFromTyped(str(new Uint32Array([0, 1]), new Uint8Array(3), 1), "none").size).toBe(1);
    });

    it("checks mixed layout under structure and values under full", () => {
        const mixed = (patch: Partial<NodeIdMapTypedParts>): NodeIdMapTypedParts =>
            parts({
                kind: "mixed",
                size: 2,
                tags: new Uint8Array([1, 0]),
                numbers: new Float64Array([0, 7]),
                offsets: new Uint32Array([0, 1, 1]),
                utf8: new Uint8Array([0x61]),
                ...patch,
            });
        const good = nodeIdMapFromTyped(mixed({}), "full");
        expectBijection(good, ["a", 7]);
        expect(captureError(() => nodeIdMapFromTyped(mixed({ tags: null }))).details).toMatchObject({
            ref: "ids.tags",
        });
        expect(captureError(() => nodeIdMapFromTyped(mixed({ numbers: null }))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(mixed({ offsets: null }))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(mixed({ utf8: null }))).code).toBe("E_BAD_SERIALIZATION");
        expect(captureError(() => nodeIdMapFromTyped(mixed({ tags: new Uint8Array([1]) }))).details).toMatchObject({
            ref: "ids.tags",
            expected: 2,
            found: 1,
        });
        expect(captureError(() => nodeIdMapFromTyped(mixed({ numbers: new Float64Array(3) }))).details).toMatchObject({
            ref: "ids.numbers",
        });
        expect(captureError(() => nodeIdMapFromTyped(mixed({ tags: new Uint8Array([1, 2]) }))).details).toMatchObject({
            ref: "ids.tags",
            index: 1,
            found: 2,
        });
        expect(
            captureError(() => nodeIdMapFromTyped(mixed({ offsets: new Uint32Array([0, 1]) }))).details,
        ).toMatchObject({
            ref: "ids.offsets",
        });
        const nan = mixed({ numbers: new Float64Array([0, Number.NaN]) });
        expect(nodeIdMapFromTyped(nan, "structure").kind).toBe("mixed");
        expect(captureError(() => nodeIdMapFromTyped(nan, "full")).details).toMatchObject({
            invariant: "I11",
            index: 1,
        });
        const malformed = mixed({ utf8: new Uint8Array([0xff]) });
        expect(captureError(() => nodeIdMapFromTyped(malformed, "full")).details).toMatchObject({
            invariant: "I11",
            index: 0,
        });
        const dup = mixed({
            tags: new Uint8Array([1, 1]),
            offsets: new Uint32Array([0, 1, 2]),
            utf8: new Uint8Array([0x61, 0x61]),
        });
        expect(captureError(() => nodeIdMapFromTyped(dup, "full")).details).toMatchObject({
            invariant: "I11",
            id: "a",
        });
        expect(nodeIdMapFromTyped(mixed({ tags: new Uint8Array([1, 2]) }), "none").size).toBe(2);
    });

    it("builds the reverse map lazily below full and eagerly at full", () => {
        const wire = nodeIdMapToTyped(nodeIdMapFromIds(["x", "y", "z"]));
        const lazy = nodeIdMapFromTyped(wire, "structure");
        expect(lazy.indexOf("y")).toBe(1);
        expect(lazy.indexOf("q")).toBe(INVALID_INDEX);
        expect(lazy.requireIndex("z")).toBe(2);
        const eager = nodeIdMapFromTyped(wire, "full");
        expect(eager.indexOf("y")).toBe(1);
    });
});

describe("gatherNodeIdMap", () => {
    it("keeps identity for a prefix and re-detects otherwise", () => {
        const identity = identityNodeIdMap(5, 1);
        expect(gatherNodeIdMap(identity, new Uint32Array([0, 1, 2])).kind).toBe("identity");
        expect(gatherNodeIdMap(identity, new Uint32Array([0, 1, 2])).offset).toBe(1);
        const shifted = gatherNodeIdMap(identity, new Uint32Array([2, 3, 4]));
        expect(shifted.kind).toBe("identity");
        expect(shifted.offset).toBe(3);
        const permuted = gatherNodeIdMap(identityNodeIdMap(3), new Uint32Array([2, 0, 1]));
        expect(permuted.kind).toBe("dense");
        expectBijection(permuted, [2, 0, 1]);
        const sparse = gatherNodeIdMap(identityNodeIdMap(10), new Uint32Array([9, 0]));
        expect(sparse.kind).toBe("numeric");
        expectBijection(sparse, [9, 0]);
        expect(gatherNodeIdMap(identity, new Uint32Array(0)).kind).toBe("identity");
    });

    it("gathers string, mixed and dense maps", () => {
        const strings = nodeIdMapFromIds(["a", "b", "c"]);
        expectBijection(gatherNodeIdMap(strings, new Uint32Array([2, 0])), ["c", "a"]);
        const mixed = nodeIdMapFromIds(["a", 1, "b", 2]);
        expectBijection(gatherNodeIdMap(mixed, new Uint32Array([3, 1])), [2, 1]);
        expect(gatherNodeIdMap(mixed, new Uint32Array([3, 1])).kind).toBe("dense");
        const dense = nodeIdMapFromIds([3, 1, 0, 2]);
        expectBijection(gatherNodeIdMap(dense, new Uint32Array([1, 3])), [1, 2]);
        expect(gatherNodeIdMap(dense, new Uint32Array([1, 3])).kind).toBe("identity");
    });

    it("throws E_INDEX_RANGE for an out-of-range source index", () => {
        expect(captureError(() => gatherNodeIdMap(identityNodeIdMap(3), new Uint32Array([3]))).code).toBe(
            "E_INDEX_RANGE",
        );
    });

    it("satisfies out.idOf(i) === map.idOf(indexMap[i]) for random selections", () => {
        fc.assert(
            fc.property(uniqueIds, fc.nat(), (ids, seed) => {
                const map = nodeIdMapFromIds(ids);
                const indices = Array.from({ length: ids.length }, (_, i) => i);
                const keep = indices.filter((i) => (i * 7919 + seed) % 3 !== 0).reverse();
                const indexMap = new Uint32Array(keep);
                const out = gatherNodeIdMap(map, indexMap);
                expect(out.size).toBe(keep.length);
                for (let i = 0; i < keep.length; i++) {
                    expect(out.idOf(i)).toBe(map.idOf(indexMap[i]));
                    expect(out.indexOf(map.idOf(indexMap[i]))).toBe(i);
                }
                expect(out.kind).toBe(expectedKind(out.toArray()));
            }),
            { numRuns: 200 },
        );
    });
});

describe("remapNodeIdMap", () => {
    it("compacts through an old -> new map with INVALID_INDEX for dropped nodes", () => {
        const map = nodeIdMapFromIds(["a", "b", "c", "d"]);
        const remap = new Uint32Array([0, INVALID_INDEX, 1, INVALID_INDEX]);
        const out = remapNodeIdMap(map, remap, 2);
        expectBijection(out, ["a", "c"]);
        const identity = remapNodeIdMap(identityNodeIdMap(4, 1), new Uint32Array([INVALID_INDEX, 0, 1, 2]), 3);
        expect(identity.kind).toBe("identity");
        expect(identity.offset).toBe(2);
        const reordered = remapNodeIdMap(nodeIdMapFromIds([5, 6, 7]), new Uint32Array([2, 0, 1]), 3);
        expectBijection(reordered, [6, 7, 5]);
        expect(remapNodeIdMap(map, new Uint32Array(4).fill(INVALID_INDEX), 0).size).toBe(0);
    });

    it("rejects a remap of the wrong length, a repeated or out-of-range target, or a hole", () => {
        const map = nodeIdMapFromIds(["a", "b", "c"]);
        expect(captureError(() => remapNodeIdMap(map, new Uint32Array([0, 1]), 2)).code).toBe("E_INDEX_RANGE");
        expect(captureError(() => remapNodeIdMap(map, new Uint32Array([0, 0, 1]), 2)).code).toBe("E_INDEX_RANGE");
        expect(captureError(() => remapNodeIdMap(map, new Uint32Array([0, 1, 5]), 3)).code).toBe("E_INDEX_RANGE");
        expect(captureError(() => remapNodeIdMap(map, new Uint32Array([0, INVALID_INDEX, 2]), 3)).code).toBe(
            "E_INDEX_RANGE",
        );
    });

    it("agrees with gather through the inverse map for random compactions", () => {
        fc.assert(
            fc.property(uniqueIds, fc.nat(), (ids, seed) => {
                const map = nodeIdMapFromIds(ids);
                const remap = new Uint32Array(ids.length).fill(INVALID_INDEX);
                const kept: number[] = [];
                for (let i = 0; i < ids.length; i++) {
                    if ((i * 104729 + seed) % 4 !== 0) {
                        remap[i] = kept.length;
                        kept.push(i);
                    }
                }
                const out = remapNodeIdMap(map, remap, kept.length);
                const viaGather = gatherNodeIdMap(map, new Uint32Array(kept));
                expect(out.toArray()).toEqual(viaGather.toArray());
                expect(out.kind).toBe(viaGather.kind);
                for (let old = 0; old < ids.length; old++) {
                    if (remap[old] !== INVALID_INDEX) {
                        expect(out.idOf(remap[old])).toBe(map.idOf(old));
                    }
                }
            }),
            { numRuns: 200 },
        );
    });
});

describe("byteLength", () => {
    it("counts only typed storage", () => {
        expect(identityNodeIdMap(1000).byteLength()).toBe(0);
        expect(nodeIdMapFromIds([1, 0, 3]).byteLength()).toBe(3 * 4 + 4 * 4);
        expect(nodeIdMapFromIds([0.5, 1]).byteLength()).toBe(0);
        expect(nodeIdMapFromIds(["ab", "c"]).byteLength()).toBe(0);
        expect(nodeIdMapFromIds(["ab", 1]).byteLength()).toBe(0);
        const wireString = nodeIdMapFromTyped(nodeIdMapToTyped(nodeIdMapFromIds(["ab", "c"])));
        expect(wireString.byteLength()).toBe(3 * 4 + 3);
        wireString.toArray();
        wireString.indexOf("c");
        expect(wireString.byteLength()).toBe(3 * 4 + 3);
    });
});
