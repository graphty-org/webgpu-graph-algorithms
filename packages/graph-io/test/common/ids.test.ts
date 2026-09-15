import { GraphFormatError, type IdCoercion } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    canonicalId,
    coerceId,
    coerceIdText,
    ID_MERGED_CODE,
    IdCoercer,
    isCanonicalIntegerText,
} from "../../src/common/ids.js";

function codeOf(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        return err instanceof GraphFormatError ? err.code : "other";
    }
    return null;
}

describe("canonical id rule (design 4.1)", () => {
    it.each<[string, string | number]>([
        ["1", 1],
        ["0", 0],
        ["-1", -1],
        ["42", 42],
        ["9007199254740991", 9007199254740991],
        ["01", "01"],
        ["1.0", "1.0"],
        ["+1", "+1"],
        ["-0", "-0"],
        ["", ""],
        [" 1", " 1"],
        ["1 ", "1 "],
        ["1e3", "1e3"],
        ["0x10", "0x10"],
        ["9007199254740992", "9007199254740992"],
        ["-9007199254740992", "-9007199254740992"],
        ["abc", "abc"],
        ["00", "00"],
        ["true", "true"],
        ["NaN", "NaN"],
        ["Infinity", "Infinity"],
    ])("canonicalId(%j) -> %j", (text, expected) => {
        expect(canonicalId(text)).toBe(expected);
        expect(isCanonicalIntegerText(text)).toBe(typeof expected === "number");
    });

    it("is injective on text: String(id) reproduces the cell", () => {
        const cells = ["1", "01", "1.0", "+1", "-0", "0", "-1", "10", "010", "abc", "9007199254740993"];
        const ids = cells.map((c) => canonicalId(c));
        expect(new Set(ids.map((id) => `${typeof id}:${String(id)}`)).size).toBe(cells.length);
        expect(ids.map((id) => String(id))).toEqual(cells);
    });
});

describe("coerceIdText", () => {
    it("keep and string leave text alone", () => {
        for (const mode of ["keep", "string"] as const) {
            expect(coerceIdText("1", mode)).toBe("1");
            expect(coerceIdText("01", mode)).toBe("01");
            expect(coerceIdText("", mode)).toBe("");
        }
    });

    it("canonical applies the rule", () => {
        expect(coerceIdText("1", "canonical")).toBe(1);
        expect(coerceIdText("01", "canonical")).toBe("01");
    });

    it("number applies Number() and rejects empty, NaN and non-finite text", () => {
        expect(coerceIdText("1", "number")).toBe(1);
        expect(coerceIdText("01", "number")).toBe(1);
        expect(coerceIdText("1.5", "number")).toBe(1.5);
        expect(coerceIdText("1e3", "number")).toBe(1000);
        expect(coerceIdText(" 7 ", "number")).toBe(7);
        expect(codeOf(() => coerceIdText("", "number"))).toBe("E_INVALID_ID");
        expect(codeOf(() => coerceIdText("   ", "number"))).toBe("E_INVALID_ID");
        expect(codeOf(() => coerceIdText("abc", "number"))).toBe("E_INVALID_ID");
        expect(codeOf(() => coerceIdText("Infinity", "number"))).toBe("E_INVALID_ID");
        expect(codeOf(() => coerceIdText("NaN", "number"))).toBe("E_INVALID_ID");
    });

    it("rejects an unknown mode with E_UNSUPPORTED", () => {
        expect(codeOf(() => coerceIdText("1", "weird" as IdCoercion))).toBe("E_UNSUPPORTED");
    });
});

describe("coerceId on typed values", () => {
    it("passes strings through the text rule and finite numbers through unchanged", () => {
        expect(coerceId("1", "canonical")).toBe(1);
        expect(coerceId("1", "keep")).toBe("1");
        expect(coerceId(1.5, "keep")).toBe(1.5);
        expect(coerceId(1.5, "canonical")).toBe(1.5);
        expect(coerceId(2, "number")).toBe(2);
    });

    it("names the shared merge code every importer records", () => {
        expect(ID_MERGED_CODE).toBe("W_ID_MERGED");
    });

    it("string applies String() to numbers (design 4.1, the fromRecords rule)", () => {
        expect(coerceId(2, "string")).toBe("2");
        expect(coerceId(1.5, "string")).toBe("1.5");
        expect(coerceId(-0, "string")).toBe("0");
        expect(new IdCoercer("string").value(7)).toBe("7");
    });

    it("rejects NaN and infinite numbers under every mode", () => {
        for (const mode of ["keep", "canonical", "string", "number"] as const) {
            expect(codeOf(() => coerceId(NaN, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId(Infinity, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId(-Infinity, mode))).toBe("E_INVALID_ID");
        }
    });

    it("string coerces booleans, bigints and null with String(); other modes reject them (design 8.5)", () => {
        expect(coerceId(true, "string")).toBe("true");
        expect(coerceId(false, "string")).toBe("false");
        expect(coerceId(null, "string")).toBe("null");
        expect(coerceId(BigInt(12), "string")).toBe("12");
        for (const mode of ["keep", "canonical", "number"] as const) {
            expect(codeOf(() => coerceId(true, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId(null, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId(BigInt(1), mode))).toBe("E_INVALID_ID");
        }
    });

    it("rejects undefined, objects and arrays under every mode", () => {
        for (const mode of ["keep", "canonical", "string", "number"] as const) {
            expect(codeOf(() => coerceId(undefined, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId({ id: 1 }, mode))).toBe("E_INVALID_ID");
            expect(codeOf(() => coerceId([1], mode))).toBe("E_INVALID_ID");
        }
    });

    it("carries the reason and value in details", () => {
        try {
            coerceId(NaN, "keep");
        } catch (err) {
            expect((err as GraphFormatError).details).toMatchObject({ reason: "not a finite number" });
        }
    });
});

describe("IdCoercer", () => {
    it("counts merges under number when two texts map to one id", () => {
        const c = new IdCoercer("number");
        expect(c.text("1")).toBe(1);
        expect(c.lastMerge).toBeNull();
        expect(c.text("1")).toBe(1);
        expect(c.lastMerge).toBeNull();
        expect(c.text("01")).toBe(1);
        expect(c.lastMerge).toEqual({ id: 1, text: "01", previousText: "1" });
        expect(c.text("1.0")).toBe(1);
        expect(c.mergeCount).toBe(2);
        expect(c.text("2")).toBe(2);
        expect(c.lastMerge).toBeNull();
        expect(c.value("2.0")).toBe(2);
        expect(c.lastMerge).toEqual({ id: 2, text: "2.0", previousText: "2" });
    });

    it("never merges under canonical, keep or string", () => {
        for (const mode of ["canonical", "keep", "string"] as const) {
            const c = new IdCoercer(mode);
            c.text("1");
            c.text("01");
            c.text("1.0");
            expect(c.mergeCount).toBe(0);
            expect(c.lastMerge).toBeNull();
        }
    });

    it("value() routes strings through text() and others through coerceId", () => {
        const c = new IdCoercer("canonical");
        expect(c.value("7")).toBe(7);
        expect(c.value(7)).toBe(7);
        expect(c.value("x")).toBe("x");
        expect(codeOf(() => c.value(null))).toBe("E_INVALID_ID");
        expect(c.mode).toBe("canonical");
    });
});
