import { describe, expect, it } from "vitest";

import { formatF32, formatF64, formatGmlReal, formatInteger, formatNumber } from "../../src/common/format.js";

describe("formatF32 (design 3.7)", () => {
    it("writes the shortest decimal that round-trips through Math.fround", () => {
        expect(formatF32(Math.fround(0.1))).toBe("0.1");
        expect(formatF32(Math.fround(0.3))).toBe("0.3");
        expect(formatF32(Math.fround(1 / 3))).toBe("0.33333334");
        expect(formatF32(1)).toBe("1");
        expect(formatF32(-2.5)).toBe("-2.5");
        expect(formatF32(16777216)).toBe("16777216");
        expect(formatF32(Math.fround(16777217))).toBe("16777216");
        expect(formatF32(Math.fround(1e-7))).toBe("1e-7");
        expect(formatF32(Math.fround(3.4028234663852886e38))).toBe("3.4028235e+38");
    });

    it("handles zeros and non-finite values", () => {
        expect(formatF32(0)).toBe("0");
        expect(formatF32(-0)).toBe("0");
        expect(formatF32(Infinity)).toBe("Infinity");
        expect(formatF32(-Infinity)).toBe("-Infinity");
        expect(formatF32(NaN)).toBe("NaN");
    });

    it("round-trips every value of a spread of f32 values", () => {
        const values = [0.1, 0.2, 0.7, 1.1, 123.456, 1e-20, 1e20, 9.999999e-5, 2.5e-3, 65535.5];
        for (const v of values) {
            const f = Math.fround(v);
            expect(Math.fround(Number(formatF32(f))), String(v)).toBe(f);
            expect(formatF32(f).length, String(v)).toBeLessThanOrEqual(String(f).length);
        }
        expect(formatF32(Math.fround(0.1))).not.toBe(String(Math.fround(0.1)));
    });
});

describe("formatF64 / formatNumber / formatInteger", () => {
    it("writes the shortest JS text", () => {
        expect(formatF64(0.1)).toBe("0.1");
        expect(formatF64(16777217)).toBe("16777217");
        expect(formatF64(-0)).toBe("0");
        expect(formatF64(Infinity)).toBe("Infinity");
        expect(formatF64(NaN)).toBe("NaN");
        expect(formatNumber(Math.fround(0.1), "f32")).toBe("0.1");
        expect(formatNumber(0.1, "f64")).toBe("0.1");
        expect(formatNumber(7, "i32")).toBe("7");
    });

    it("writes integers without an exponent", () => {
        expect(formatInteger(12)).toBe("12");
        expect(formatInteger(-3)).toBe("-3");
        expect(formatInteger(1e21)).toBe("1000000000000000000000");
        expect(formatInteger(Infinity)).toBe("Infinity");
    });
});

describe("formatGmlReal (design 8.5)", () => {
    it("always writes a decimal point", () => {
        expect(formatGmlReal(2)).toBe("2.0");
        expect(formatGmlReal(2.5)).toBe("2.5");
        expect(formatGmlReal(-3)).toBe("-3.0");
        expect(formatGmlReal(0)).toBe("0.0");
        expect(formatGmlReal(1e-7)).toBe("1.0e-7");
        expect(formatGmlReal(1.5e21)).toBe("1.5e+21");
        expect(formatGmlReal(1e21)).toBe("1.0e+21");
    });

    it("writes the NetworkX texts for non-finite values", () => {
        expect(formatGmlReal(Infinity)).toBe("+INF");
        expect(formatGmlReal(-Infinity)).toBe("-INF");
        expect(formatGmlReal(NaN)).toBe("NAN");
    });

    it("formats f32 values with the shortest fround-round-trip text", () => {
        expect(formatGmlReal(Math.fround(0.1), "f32")).toBe("0.1");
        expect(formatGmlReal(Math.fround(0.1))).toBe(String(Math.fround(0.1)));
        expect(formatGmlReal(3, "i32")).toBe("3.0");
    });
});
