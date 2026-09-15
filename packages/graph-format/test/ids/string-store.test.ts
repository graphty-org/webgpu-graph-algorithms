import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    checkUtf8Layout,
    decodeUtf8Row,
    detachString,
    encodeUtf8Into,
    encodeUtf8Rows,
    hasLoneSurrogate,
    resolveRange,
    utf8ByteLength,
    Utf8Store,
} from "../../src/ids/string-store.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Strings over the full code point range plus hand-picked edge cases (empty, ASCII, 2-, 3- and
// 4-byte sequences, lone surrogates on either side and in the middle).
const LONE_HIGH = "\ud83d";
const LONE_LOW = "\ude00";
const E_ACUTE = "\u00e9";
const anyString = fc.oneof(
    fc.string(),
    fc.string({ unit: "binary" }),
    fc.string({ unit: "binary-ascii" }),
    fc.constantFrom("", "a", "abc", "\u00e9", "\u20ac", "\ud83d\ude00", "x\ud83d\ude00y", "\u0000"),
    fc.constantFrom(LONE_HIGH, LONE_LOW, `a${LONE_HIGH}`, `${LONE_LOW}b`, `a${LONE_HIGH}b`, `${LONE_HIGH}${LONE_HIGH}`),
);

function encodeWith(s: string): Uint8Array {
    const out = new Uint8Array(utf8ByteLength(s));
    const end = encodeUtf8Into(s, out, 0);
    expect(end).toBe(out.length);
    return out;
}

describe("utf8ByteLength / encodeUtf8Into", () => {
    it("matches TextEncoder byte for byte, including U+FFFD for lone surrogates", () => {
        fc.assert(
            fc.property(anyString, (s) => {
                const expected = encoder.encode(s);
                expect(utf8ByteLength(s)).toBe(expected.length);
                expect(Array.from(encodeWith(s))).toEqual(Array.from(expected));
            }),
            { numRuns: 500 },
        );
    });

    it("writes at the given position and returns the end", () => {
        const out = new Uint8Array(10).fill(0xaa);
        const end = encodeUtf8Into("hi", out, 3);
        expect(end).toBe(5);
        expect(Array.from(out)).toEqual([0xaa, 0xaa, 0xaa, 0x68, 0x69, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa]);
    });

    it("encodes the four sequence lengths and both lone surrogates", () => {
        expect(Array.from(encodeWith("A"))).toEqual([0x41]);
        expect(Array.from(encodeWith("\u00e9"))).toEqual([0xc3, 0xa9]);
        expect(Array.from(encodeWith("\u20ac"))).toEqual([0xe2, 0x82, 0xac]);
        expect(Array.from(encodeWith("\ud83d\ude00"))).toEqual([0xf0, 0x9f, 0x98, 0x80]);
        expect(Array.from(encodeWith(LONE_HIGH))).toEqual([0xef, 0xbf, 0xbd]);
        expect(Array.from(encodeWith(LONE_LOW))).toEqual([0xef, 0xbf, 0xbd]);
        expect(Array.from(encodeWith(`${LONE_HIGH}a`))).toEqual([0xef, 0xbf, 0xbd, 0x61]);
    });
});

describe("hasLoneSurrogate", () => {
    it("accepts well-formed strings", () => {
        for (const s of ["", "abc", "\u00e9", "\ud83d\ude00", "a\ud83d\ude00b", "\ud83d\ude00\ud83d\ude00"]) {
            expect(hasLoneSurrogate(s)).toBe(false);
        }
    });

    it("rejects lone surrogates anywhere", () => {
        for (const s of [
            LONE_HIGH,
            LONE_LOW,
            `a${LONE_HIGH}`,
            `${LONE_LOW}b`,
            `a${LONE_HIGH}b`,
            `${LONE_HIGH}${LONE_HIGH}`,
            `${LONE_LOW}${LONE_HIGH}\ude00`,
        ]) {
            expect(hasLoneSurrogate(s)).toBe(true);
        }
    });

    it("agrees with a fatal TextEncoder round trip", () => {
        fc.assert(
            fc.property(anyString, (s) => {
                const roundTrip = decoder.decode(encoder.encode(s));
                expect(hasLoneSurrogate(s)).toBe(roundTrip !== s);
            }),
            { numRuns: 500 },
        );
    });
});

describe("decodeUtf8Row", () => {
    it("decodes any well-formed string, short and long, ASCII and not", () => {
        fc.assert(
            fc.property(fc.string({ unit: "binary", maxLength: 3000 }), (s) => {
                const bytes = encoder.encode(s);
                expect(decodeUtf8Row(bytes, 0, bytes.length)).toBe(s);
            }),
            { numRuns: 200 },
        );
    });

    it("takes the ASCII fast path and the decoder path on the same input", () => {
        const long = "a".repeat(5000);
        const bytes = encoder.encode(long);
        expect(decodeUtf8Row(bytes, 0, bytes.length)).toBe(long);
        const mixed = `${"a".repeat(100)}\u00e9`;
        const mixedBytes = encoder.encode(mixed);
        expect(decodeUtf8Row(mixedBytes, 0, mixedBytes.length)).toBe(mixed);
    });

    it("decodes a sub-range and returns an empty string for an empty range", () => {
        const bytes = encoder.encode("hello world");
        expect(decodeUtf8Row(bytes, 6, 11)).toBe("world");
        expect(decodeUtf8Row(bytes, 3, 3)).toBe("");
    });

    it("replaces malformed sequences instead of throwing", () => {
        expect(decodeUtf8Row(new Uint8Array([0x61, 0xff, 0x62]), 0, 3)).toBe("a\ufffdb");
    });
});

describe("encodeUtf8Rows", () => {
    it("builds monotonic offsets starting at 0 and empty rows for non-strings", () => {
        const { offsets, utf8 } = encodeUtf8Rows(["ab", 7, "", "\u00e9", undefined, "z"], 6);
        expect(Array.from(offsets)).toEqual([0, 2, 2, 2, 4, 4, 5]);
        expect(Array.from(utf8)).toEqual([0x61, 0x62, 0xc3, 0xa9, 0x7a]);
    });

    it("reads only the first `length` entries", () => {
        const { offsets, utf8 } = encodeUtf8Rows(["a", "b", "c"], 2);
        expect(Array.from(offsets)).toEqual([0, 1, 2]);
        expect(utf8.length).toBe(2);
    });

    it("produces exactly sized arrays over their own buffers", () => {
        const { offsets, utf8 } = encodeUtf8Rows(["abc"], 1);
        expect(offsets.buffer.byteLength).toBe(offsets.byteLength);
        expect(utf8.buffer.byteLength).toBe(utf8.byteLength);
    });
});

describe("checkUtf8Layout", () => {
    const bytes = new Uint8Array(4);

    it("accepts a well-formed layout, including a non-zero base", () => {
        expect(() => checkUtf8Layout(new Uint32Array([0, 1, 4]), bytes, 2, "x")).not.toThrow();
        expect(() => checkUtf8Layout(new Uint32Array([2, 3, 4]), bytes, 2, "x")).not.toThrow();
        expect(() => checkUtf8Layout(new Uint32Array([0]), new Uint8Array(0), 0, "x")).not.toThrow();
    });

    it("rejects a wrong offsets length", () => {
        const err = captureError(() => checkUtf8Layout(new Uint32Array([0, 1]), bytes, 2, "ids.offsets"));
        expect(err.code).toBe("E_BAD_SERIALIZATION");
        expect(err.details).toMatchObject({ ref: "ids.offsets", reason: "offsets length", expected: 3, found: 2 });
    });

    it("rejects decreasing offsets", () => {
        const err = captureError(() => checkUtf8Layout(new Uint32Array([0, 3, 2]), bytes, 2, "ids.offsets"));
        expect(err.code).toBe("E_BAD_SERIALIZATION");
        expect(err.details).toMatchObject({ ref: "ids.offsets", reason: "offsets not monotonic", row: 1 });
    });

    it("rejects offsets past the byte store", () => {
        const err = captureError(() => checkUtf8Layout(new Uint32Array([0, 2, 5]), bytes, 2, "ids.offsets"));
        expect(err.code).toBe("E_BAD_SERIALIZATION");
        expect(err.details).toMatchObject({ ref: "ids.offsets", reason: "offsets exceed utf8", end: 5, byteLength: 4 });
    });
});

describe("Utf8Store.fromStrings", () => {
    it("shares the source, reads only the first `length` entries and encodes lazily", () => {
        const source = ["a", "bb", "ccc"];
        const store = Utf8Store.fromStrings(source, 2);
        expect(store.length).toBe(2);
        expect(store.decoded).toBe(true);
        expect(store.encoded).toBe(false);
        expect(store.byteLength()).toBe(0);
        expect(store.at(0)).toBe("a");
        expect(store.at(1)).toBe("bb");
        expect(store.decodeAll()).toEqual(["a", "bb"]);
        source.push("dddd");
        expect(store.decodeAll()).toEqual(["a", "bb"]);
        expect(store.encoded).toBe(false);
        const { offsets } = store;
        expect(store.encoded).toBe(true);
        expect(Array.from(offsets)).toEqual([0, 1, 3]);
        expect(Array.from(store.utf8)).toEqual([0x61, 0x62, 0x62]);
        expect(store.byteLength()).toBe(3 * 4 + 3);
        expect(store.offsets).toBe(offsets);
        expect(store.firstMalformedRow()).toBe(-1);
    });

    it("defaults length to source.length", () => {
        expect(Utf8Store.fromStrings(["x", "y"]).length).toBe(2);
        expect(Utf8Store.fromStrings([]).length).toBe(0);
        expect(Array.from(Utf8Store.fromStrings([]).offsets)).toEqual([0]);
    });

    it("slices with Array.prototype.slice semantics", () => {
        const store = Utf8Store.fromStrings(["a", "b", "c", "d"]);
        expect(store.slice()).toEqual(["a", "b", "c", "d"]);
        expect(store.slice(1, 3)).toEqual(["b", "c"]);
        expect(store.slice(-2)).toEqual(["c", "d"]);
        expect(store.slice(0, -1)).toEqual(["a", "b", "c"]);
        expect(store.slice(3, 1)).toEqual([]);
        expect(store.slice(10)).toEqual([]);
        expect(store.slice(-10, 1)).toEqual(["a"]);
    });

    it("throws E_INDEX_RANGE for an out-of-range row", () => {
        const store = Utf8Store.fromStrings(["a"]);
        for (const row of [1, -1, 0.5, Number.NaN]) {
            const err = captureError(() => store.at(row));
            expect(err.code).toBe("E_INDEX_RANGE");
            expect(err.details).toMatchObject({ size: 1 });
        }
    });
});

describe("Utf8Store.fromEncoded", () => {
    function encodedStore(strings: string[]): Utf8Store {
        const { offsets, utf8 } = encodeUtf8Rows(strings, strings.length);
        return Utf8Store.fromEncoded(offsets, utf8);
    }

    it("decodes per row on demand and caches the row", () => {
        const store = encodedStore(["alpha", "\u00e9\u20ac", ""]);
        expect(store.length).toBe(3);
        expect(store.decoded).toBe(false);
        expect(store.encoded).toBe(true);
        const first = store.at(1);
        expect(first).toBe("\u00e9\u20ac");
        expect(store.at(1)).toBe(first);
        expect(store.at(0)).toBe("alpha");
        expect(store.at(2)).toBe("");
        expect(store.byteLength()).toBe(4 * 4 + 10);
    });

    it("decodes in bulk without caching and reuses cached rows", () => {
        const store = encodedStore(["a", "b", "c"]);
        expect(store.slice(1)).toEqual(["b", "c"]);
        expect(store.at(0)).toBe("a");
        expect(store.decodeAll()).toEqual(["a", "b", "c"]);
        expect(store.slice(-1)).toEqual(["c"]);
    });

    it("materialiseDecoded turns an encoded store into a decoded one exactly once", () => {
        const store = encodedStore(["a", E_ACUTE, "ccc"]);
        expect(store.at(2)).toBe("ccc");
        expect(store.decoded).toBe(false);
        store.materialiseDecoded();
        expect(store.decoded).toBe(true);
        expect(store.encoded).toBe(true);
        expect(store.at(0)).toBe("a");
        expect(store.at(1)).toBe(E_ACUTE);
        expect(store.decodeAll()).toEqual(["a", E_ACUTE, "ccc"]);
        store.materialiseDecoded();
        expect(store.decodeAll()).toEqual(["a", E_ACUTE, "ccc"]);
        const fromStrings = Utf8Store.fromStrings(["x"]);
        fromStrings.materialiseDecoded();
        expect(fromStrings.encoded).toBe(false);
        expect(fromStrings.at(0)).toBe("x");
    });

    it("bulk decodes an all-ASCII range in one pass and a mixed range per row", () => {
        const rows = ["alpha", "beta", "", "gamma-delta", "e"];
        expect(encodedStore(rows).slice(1, 4)).toEqual(["beta", "", "gamma-delta"]);
        const mixed = ["alpha", E_ACUTE, "", "x".repeat(100), `${"y".repeat(40)}${E_ACUTE}`];
        expect(encodedStore(mixed).decodeAll()).toEqual(mixed);
        expect(encodedStore(mixed).slice(2)).toEqual(mixed.slice(2));
    });

    it("supports a non-zero offsets base (a zero-copy slice of a parent store)", () => {
        const { offsets, utf8 } = encodeUtf8Rows(["aa", "bbb", "c"], 3);
        const child = Utf8Store.fromEncoded(offsets.subarray(1, 4), utf8);
        expect(child.length).toBe(2);
        expect(child.at(0)).toBe("bbb");
        expect(child.at(1)).toBe("c");
        expect(child.decodeAll()).toEqual(["bbb", "c"]);
    });

    it("handles an empty encoded store", () => {
        const store = Utf8Store.fromEncoded(new Uint32Array([0]), new Uint8Array(0));
        expect(store.length).toBe(0);
        expect(store.decodeAll()).toEqual([]);
        expect(store.firstMalformedRow()).toBe(-1);
        expect(Utf8Store.fromEncoded(new Uint32Array(0), new Uint8Array(0)).length).toBe(0);
    });

    it("throws E_INDEX_RANGE for an out-of-range row", () => {
        const store = encodedStore(["a"]);
        expect(captureError(() => store.at(1)).code).toBe("E_INDEX_RANGE");
    });

    it("reports the first malformed row under a fatal decode", () => {
        expect(encodedStore(["ok", "\u00e9"]).firstMalformedRow()).toBe(-1);
        const bad = Utf8Store.fromEncoded(new Uint32Array([0, 2, 3, 4]), new Uint8Array([0x61, 0x62, 0xff, 0x63]));
        expect(bad.firstMalformedRow()).toBe(1);
        const split = Utf8Store.fromEncoded(new Uint32Array([0, 1, 2]), new Uint8Array([0xc3, 0xa9]));
        expect(split.firstMalformedRow()).toBe(0);
        const together = Utf8Store.fromEncoded(new Uint32Array([0, 2]), new Uint8Array([0xc3, 0xa9]));
        expect(together.firstMalformedRow()).toBe(-1);
        expect(together.at(0)).toBe("\u00e9");
    });

    it("round trips arbitrary strings through the encoded form", () => {
        fc.assert(
            fc.property(fc.array(fc.string({ unit: "binary" }), { maxLength: 40 }), (strings) => {
                const source = Utf8Store.fromStrings(strings);
                const wire = Utf8Store.fromEncoded(source.offsets, source.utf8);
                expect(wire.decodeAll()).toEqual(strings);
                for (let i = 0; i < strings.length; i++) {
                    expect(wire.at(i)).toBe(strings[i]);
                }
                expect(wire.firstMalformedRow()).toBe(-1);
                expect(wire.byteLength()).toBe(source.byteLength());
            }),
            { numRuns: 200 },
        );
    });
});

describe("resolveRange", () => {
    it("clamps and resolves negative bounds like Array.prototype.slice", () => {
        const cases: [number, number, number][] = [
            [0, 5, 5],
            [2, 4, 5],
            [-2, 5, 5],
            [0, -1, 5],
            [4, 2, 5],
            [10, 20, 5],
            [-10, 2, 5],
            [1.7, 3.2, 5],
            [Number.NaN, 3, 5],
            [1, Number.NaN, 5],
            [0, 0, 0],
        ];
        for (const [start, end, length] of cases) {
            const expected = Array.from({ length }, (_, i) => i).slice(start, end);
            const [from, to] = resolveRange(start, end, length);
            expect(Array.from({ length }, (_, i) => i).slice(from, to)).toEqual(expected);
        }
    });
});

function captureError(fn: () => unknown): GraphFormatError {
    try {
        fn();
    } catch (err) {
        expect(err).toBeInstanceOf(GraphFormatError);
        return err as GraphFormatError;
    }
    throw new Error("expected a GraphFormatError");
}

describe("detachString (io round 1: retained ids and cells must not pin their input chunk)", () => {
    it("returns an equal string for every length and keeps short strings as they are", () => {
        fc.assert(
            fc.property(fc.string({ maxLength: 40 }), (s) => {
                const detached = detachString(s);
                expect(detached).toBe(s);
                if (s.length < 13) {
                    expect(Object.is(detached, s)).toBe(true);
                }
            }),
        );
    });

    it("a builder does not keep a 60 KiB chunk alive through the ids sliced out of it", () => {
        // 2000 ids, each sliced from its own 60 KiB chunk: unflattened they would pin about 115 MiB
        const chunks = 2000;
        const b = new GraphBuilder({ directed: true });
        const before = process.memoryUsage().heapUsed;
        for (let i = 0; i < chunks; i++) {
            const chunk = `node-${String(i).padStart(8, "0")}-${"x".repeat(60 * 1024)}`;
            b.addNode(chunk.slice(0, 13));
            b.setNodeValue("s", i, chunk.slice(0, 20));
        }
        global.gc?.();
        const grown = process.memoryUsage().heapUsed - before;
        expect(b.nodeCount).toBe(chunks);
        // with gc exposed the retained growth is a few MiB; without it the bound stays generous
        expect(grown).toBeLessThan(60 * 1024 * 1024);
    });
});
