import { GraphBuilder, GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DEFAULT_WEIGHT, isWeightField, parseWeightText, weightFromValue } from "../../src/common/weights.js";

function codeOf(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        return err instanceof GraphFormatError ? err.code : "other";
    }
    return null;
}

describe("weight resolution (design 3.7, 8.4)", () => {
    it("isWeightField follows the resolved weightFrom option", () => {
        expect(isWeightField("weight", "weight")).toBe(true);
        expect(isWeightField("Weight", "weight")).toBe(false);
        expect(isWeightField("value", "value")).toBe(true);
        expect(isWeightField("weight", null)).toBe(false);
        expect(DEFAULT_WEIGHT).toBe(1);
    });

    it("parseWeightText: blank is absent, numbers parse, Infinity allowed, NaN and text rejected", () => {
        expect(parseWeightText("")).toBeUndefined();
        expect(parseWeightText("   ")).toBeUndefined();
        expect(parseWeightText("2")).toBe(2);
        expect(parseWeightText(" 0.1 ")).toBe(0.1);
        expect(parseWeightText("16777217")).toBe(16777217);
        expect(parseWeightText("-3.5")).toBe(-3.5);
        expect(parseWeightText("0")).toBe(0);
        expect(parseWeightText("1e-3")).toBe(0.001);
        expect(parseWeightText("Infinity")).toBe(Infinity);
        expect(parseWeightText("-INF")).toBe(-Infinity);
        expect(codeOf(() => parseWeightText("NaN"))).toBe("E_INVALID_WEIGHT");
        expect(codeOf(() => parseWeightText("heavy"))).toBe("E_INVALID_WEIGHT");
        expect(codeOf(() => parseWeightText("1,5"))).toBe("E_INVALID_WEIGHT");
    });

    it("weightFromValue: null / undefined absent, numbers as is, numeric text parsed, others rejected", () => {
        expect(weightFromValue(undefined)).toBeUndefined();
        expect(weightFromValue(null)).toBeUndefined();
        expect(weightFromValue(2.5)).toBe(2.5);
        expect(weightFromValue(-Infinity)).toBe(-Infinity);
        expect(weightFromValue("3")).toBe(3);
        expect(weightFromValue("")).toBeUndefined();
        expect(codeOf(() => weightFromValue(NaN))).toBe("E_INVALID_WEIGHT");
        expect(codeOf(() => weightFromValue(true))).toBe("E_INVALID_WEIGHT");
        expect(codeOf(() => weightFromValue({ w: 1 }))).toBe("E_INVALID_WEIGHT");
        expect(codeOf(() => weightFromValue("x"))).toBe("E_INVALID_WEIGHT");
    });

    it("carries the rejected value in details", () => {
        try {
            weightFromValue("x");
        } catch (err) {
            expect((err as GraphFormatError).details.value).toBe("x");
            expect(typeof (err as GraphFormatError).details.cause).toBe("string");
        }
        try {
            weightFromValue(NaN);
        } catch (err) {
            expect((err as GraphFormatError).message).toBe("invalid edge weight NaN");
        }
    });

    it("an absent weight is pushed without an argument so the builder records it as defaulted", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b", parseWeightText("0.1"));
        b.addEdge("b", "c", parseWeightText(""));
        b.addEdge("c", "d", parseWeightText("16777217"));
        const s = b.freeze();
        const shadow = s.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.isSet(2)).toBe(true);
        expect(shadow?.value(0)).toBe(0.1);
        expect(shadow?.value(2)).toBe(16777217);
        expect(s.edgeList().weights?.[1]).toBe(1);
    });

    it("the E_INVALID_WEIGHT boundary keeps the sink untouched", () => {
        const b = new GraphBuilder({ directed: true });
        let weight: number | undefined;
        expect(codeOf(() => (weight = parseWeightText("bad")))).toBe("E_INVALID_WEIGHT");
        expect(weight).toBeUndefined();
        expect(b.edgeCount).toBe(0);
        expect(b.nodeCount).toBe(0);
    });
});
