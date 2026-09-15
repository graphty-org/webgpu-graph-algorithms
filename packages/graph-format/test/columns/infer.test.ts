import { describe, expect, it } from "vitest";

import {
    coerceValue,
    DtypeInferrer,
    type InferredDtype,
    inferTextDtype,
    inferTextsDtype,
    inferValueDtype,
    inferValuesDtype,
    parseText,
    widenDtype,
    WIDENING_ORDER,
    wideningRank,
} from "../../src/columns/infer.js";
import { GraphFormatError } from "../../src/errors.js";

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

describe("widening order", () => {
    it("is (unset) -> bool -> i32 -> f64 -> string -> json and monotone", () => {
        expect(WIDENING_ORDER).toEqual(["bool", "i32", "f64", "string", "json"]);
        expect(wideningRank("bool")).toBe(0);
        expect(wideningRank("json")).toBe(4);
        expect(widenDtype(null, null)).toBeNull();
        expect(widenDtype(null, "i32")).toBe("i32");
        expect(widenDtype("i32", null)).toBe("i32");
        expect(widenDtype("i32", "bool")).toBe("i32");
        expect(widenDtype("bool", "i32")).toBe("i32");
        expect(widenDtype("f64", "string")).toBe("string");
        expect(widenDtype("string", "f64")).toBe("string");
        expect(widenDtype("string", "json")).toBe("json");
        for (const a of WIDENING_ORDER) {
            for (const b of WIDENING_ORDER) {
                expect(widenDtype(a, b)).toBe(widenDtype(b, a));
                expect(wideningRank(widenDtype(a, b) as InferredDtype)).toBe(
                    Math.max(wideningRank(a), wideningRank(b)),
                );
            }
        }
    });
});

describe("text grammar (design section 5.1)", () => {
    it("recognises bool exactly and case-sensitively", () => {
        expect(inferTextDtype("true")).toBe("bool");
        expect(inferTextDtype("false")).toBe("bool");
        expect(inferTextDtype("True")).toBe("string");
        expect(inferTextDtype("TRUE")).toBe("string");
        expect(inferTextDtype(" true")).toBe("string");
    });

    it("recognises i32 by the canonical integer regex within range", () => {
        expect(inferTextDtype("0")).toBe("i32");
        expect(inferTextDtype("1")).toBe("i32");
        expect(inferTextDtype("-1")).toBe("i32");
        expect(inferTextDtype("-0")).toBe("i32");
        expect(inferTextDtype("2147483647")).toBe("i32");
        expect(inferTextDtype("-2147483648")).toBe("i32");
        expect(inferTextDtype("2147483648")).toBe("f64");
        expect(inferTextDtype("-2147483649")).toBe("f64");
        expect(inferTextDtype("01")).toBe("string");
        expect(inferTextDtype("+1")).toBe("f64");
        expect(inferTextDtype("1 ")).toBe("string");
        expect(inferTextDtype("")).toBe("string");
        expect(inferTextDtype(`1${"0".repeat(400)}`)).toBe("string");
    });

    it("recognises f64 decimal and exponent literals only", () => {
        expect(inferTextDtype("1.5")).toBe("f64");
        expect(inferTextDtype(".5")).toBe("f64");
        expect(inferTextDtype("5.")).toBe("f64");
        expect(inferTextDtype("-2.5e3")).toBe("f64");
        expect(inferTextDtype("1E-7")).toBe("f64");
        expect(inferTextDtype("1e999")).toBe("string");
        expect(inferTextDtype("Infinity")).toBe("string");
        expect(inferTextDtype("-Infinity")).toBe("string");
        expect(inferTextDtype("NaN")).toBe("string");
        expect(inferTextDtype("0x10")).toBe("string");
        expect(inferTextDtype("0b1")).toBe("string");
        expect(inferTextDtype("0o7")).toBe("string");
        expect(inferTextDtype(" ")).toBe("string");
        expect(inferTextDtype("1_000")).toBe("string");
        expect(inferTextDtype("1,5")).toBe("string");
        expect(inferTextDtype("00.5")).toBe("string");
        expect(inferTextDtype("0.5")).toBe("f64");
        expect(inferTextDtype("-0.0")).toBe("f64");
        expect(inferTextDtype("abc")).toBe("string");
    });

    it("infers a whole text column", () => {
        expect(inferTextsDtype([])).toBeNull();
        expect(inferTextsDtype(["true", "false"])).toBe("bool");
        expect(inferTextsDtype(["true", "1"])).toBe("i32");
        expect(inferTextsDtype(["1", "2.5"])).toBe("f64");
        expect(inferTextsDtype(["1", "01"])).toBe("string");
        expect(inferTextsDtype(["0", "1"])).toBe("i32");
    });
});

describe("JS value classification", () => {
    it("maps by typeof with null and undefined unset", () => {
        expect(inferValueDtype(undefined)).toBeNull();
        expect(inferValueDtype(null)).toBeNull();
        expect(inferValueDtype(true)).toBe("bool");
        expect(inferValueDtype(0)).toBe("i32");
        expect(inferValueDtype(-0)).toBe("i32");
        expect(inferValueDtype(2147483647)).toBe("i32");
        expect(inferValueDtype(2147483648)).toBe("f64");
        expect(inferValueDtype(1.5)).toBe("f64");
        expect(inferValueDtype(NaN)).toBe("f64");
        expect(inferValueDtype(Infinity)).toBe("f64");
        expect(inferValueDtype("1")).toBe("string");
        expect(inferValueDtype([1])).toBe("json");
        expect(inferValueDtype({})).toBe("json");
        expect(expectError(() => inferValueDtype(1n), "E_COLUMN_TYPE").details.found).toBe("bigint");
        expectError(() => inferValueDtype(Symbol("s")), "E_COLUMN_TYPE");
        expectError(() => inferValueDtype(() => 1), "E_COLUMN_TYPE");
    });

    it("infers a whole value column, never f32", () => {
        expect(inferValuesDtype([])).toBeNull();
        expect(inferValuesDtype([undefined, null])).toBeNull();
        expect(inferValuesDtype([true, undefined])).toBe("bool");
        expect(inferValuesDtype([true, 1])).toBe("i32");
        expect(inferValuesDtype([1, 0.1])).toBe("f64");
        expect(inferValuesDtype([0.1, "x"])).toBe("string");
        expect(inferValuesDtype(["x", {}])).toBe("json");
        expect(inferValuesDtype([{}, 1n])).toBe("json");
    });
});

describe("parseText and coerceValue", () => {
    it("parses text into the storage value of the widened dtype", () => {
        expect(parseText("true", "bool")).toBe(true);
        expect(parseText("false", "bool")).toBe(false);
        expectError(() => parseText("1", "bool"), "E_COLUMN_TYPE");
        expect(parseText("7", "i32")).toBe(7);
        expect(parseText("true", "i32")).toBe(1);
        expect(parseText("false", "f64")).toBe(0);
        expect(parseText("2.5", "f64")).toBe(2.5);
        expect(parseText("2147483648", "f64")).toBe(2147483648);
        expectError(() => parseText("abc", "f64"), "E_COLUMN_TYPE");
        expect(parseText("abc", "string")).toBe("abc");
        expect(parseText("abc", "json")).toBe("abc");
        expectError(() => parseText("x", "f16" as InferredDtype), "E_COLUMN_TYPE");
    });

    it("coerces JS values with the widening conversions", () => {
        expect(coerceValue(true, "bool")).toBe(true);
        expectError(() => coerceValue(1, "bool"), "E_COLUMN_TYPE");
        expect(coerceValue(true, "i32")).toBe(1);
        expect(coerceValue(false, "f64")).toBe(0);
        expect(coerceValue(2.5, "f64")).toBe(2.5);
        expectError(() => coerceValue("1", "i32"), "E_COLUMN_TYPE");
        expect(coerceValue(1, "string")).toBe("1");
        expect(coerceValue(true, "string")).toBe("true");
        expect(coerceValue(0.1, "string")).toBe("0.1");
        expect(coerceValue("s", "string")).toBe("s");
        expectError(() => coerceValue({}, "string"), "E_COLUMN_TYPE");
        const object = { a: 1 };
        expect(coerceValue(object, "json")).toBe(object);
        expect(coerceValue(1, "json")).toBe(1);
        expectError(() => coerceValue(1, "f16" as InferredDtype), "E_COLUMN_TYPE");
    });
});

describe("DtypeInferrer", () => {
    it("accumulates observations and reports widening", () => {
        const inferrer = new DtypeInferrer();
        expect(inferrer.dtype).toBeNull();
        expect(inferrer.observeValue(undefined)).toBe(false);
        expect(inferrer.observeValue(true)).toBe(true);
        expect(inferrer.dtype).toBe("bool");
        expect(inferrer.observeValue(false)).toBe(false);
        expect(inferrer.observeText("5")).toBe(true);
        expect(inferrer.dtype).toBe("i32");
        expect(inferrer.observeText("true")).toBe(false);
        expect(inferrer.observeValue(0.5)).toBe(true);
        expect(inferrer.dtype).toBe("f64");
        expect(inferrer.widenTo("i32")).toBe(false);
        expect(inferrer.widenTo("json")).toBe(true);
        expect(inferrer.dtype).toBe("json");
        expect(inferrer.observeText("anything")).toBe(false);
    });
});
