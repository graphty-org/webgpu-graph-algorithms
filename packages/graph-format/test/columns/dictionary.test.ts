import { describe, expect, it } from "vitest";

import { assertWellFormedString, buildCodeMap, DictionaryBuilder } from "../../src/columns/dictionary.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { hasLoneSurrogate } from "../../src/ids/string-store.js";

function expectError(fn: () => unknown, code: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe(code);
    return error;
}

describe("lone surrogate detection", () => {
    it("flags unpaired surrogates only", () => {
        expect(hasLoneSurrogate("plain")).toBe(false);
        expect(hasLoneSurrogate("\u00e9\u4e2d")).toBe(false);
        expect(hasLoneSurrogate("\ud83d\ude00")).toBe(false);
        expect(hasLoneSurrogate("\ud83d")).toBe(true);
        expect(hasLoneSurrogate("\ude00")).toBe(true);
        expect(hasLoneSurrogate("a\ud800b")).toBe(true);
        expect(hasLoneSurrogate("\ude00\ud800")).toBe(true);
        expect(() => assertWellFormedString("ok")).not.toThrow();
        const error = expectError(() => assertWellFormedString("\ud800", { column: "c", row: 3 }), "E_COLUMN_TYPE");
        expect(error.details).toEqual({ column: "c", row: 3, reason: "lone surrogate" });
    });
});

describe("DictionaryBuilder", () => {
    it("interns in first-seen order with dense codes", () => {
        const dictionary = new DictionaryBuilder();
        expect(dictionary.size).toBe(0);
        expect(dictionary.intern("b")).toBe(0);
        expect(dictionary.intern("a")).toBe(1);
        expect(dictionary.intern("b")).toBe(0);
        expect(dictionary.values).toEqual(["b", "a"]);
        expect(dictionary.size).toBe(2);
        expect(dictionary.codeOf("a")).toBe(1);
        expect(dictionary.codeOf("zzz")).toBe(INVALID_INDEX);
        expect(dictionary.has("a")).toBe(true);
        expect(dictionary.has("zzz")).toBe(false);
    });

    it("seeds from declared options, keeping the first code of duplicates", () => {
        const dictionary = new DictionaryBuilder(["x", "y", "x"]);
        expect(dictionary.values).toEqual(["x", "y"]);
        expect(dictionary.intern("z")).toBe(2);
    });

    it("rejects lone surrogates", () => {
        const dictionary = new DictionaryBuilder();
        expectError(() => dictionary.intern("\udfff"), "E_COLUMN_TYPE");
        expect(dictionary.size).toBe(0);
    });
});

describe("buildCodeMap", () => {
    it("maps values to codes, first occurrence wins", () => {
        const map = buildCodeMap(["a", "b", "a"]);
        expect(map.get("a")).toBe(0);
        expect(map.get("b")).toBe(1);
        expect(map.size).toBe(2);
    });
});
