import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    inferTextDtype as coreInferTextDtype,
    parseText as coreParseText,
} from "../../../graph-format/src/columns/infer.js";
import { inferTextDtype, isNumericText, parseTextCell } from "../../src/common/text.js";

const CELLS = [
    "true",
    "false",
    "True",
    "TRUE",
    "0",
    "1",
    "-1",
    "01",
    "007",
    "2147483647",
    "2147483648",
    "-2147483648",
    "-2147483649",
    "1.5",
    "-0.5",
    ".5",
    "5.",
    "1e3",
    "1E-3",
    "+1",
    "+1.5",
    "1e400",
    "-1e400",
    "0x10",
    "0o7",
    "0b1",
    "Infinity",
    "-Infinity",
    "NaN",
    "",
    " ",
    " 1",
    "1 ",
    "1,5",
    "abc",
    "9007199254740993",
    "1_000",
    "00.5",
    "0.5",
    "-0",
    "-0.0",
];

describe("text grammar of design 5.1", () => {
    it.each<[string, string]>([
        ["true", "bool"],
        ["false", "bool"],
        ["True", "string"],
        ["0", "i32"],
        ["1", "i32"],
        ["-1", "i32"],
        ["01", "string"],
        ["2147483647", "i32"],
        ["2147483648", "f64"],
        ["-2147483648", "i32"],
        ["-2147483649", "f64"],
        ["1.5", "f64"],
        [".5", "f64"],
        ["5.", "f64"],
        ["1e3", "f64"],
        ["+1", "f64"],
        ["1e400", "string"],
        ["0x10", "string"],
        ["Infinity", "string"],
        ["NaN", "string"],
        ["", "string"],
        [" ", "string"],
        [" 1", "string"],
        ["1,5", "string"],
        ["abc", "string"],
        ["00.5", "string"],
        ["-0", "i32"],
        ["-0.0", "f64"],
    ])("inferTextDtype(%j) is %s", (text, dtype) => {
        expect(inferTextDtype(text)).toBe(dtype);
    });

    it("agrees with the core's reference implementation on every cell (invariant I15)", () => {
        for (const cell of CELLS) {
            expect(inferTextDtype(cell), cell).toBe(coreInferTextDtype(cell));
            const dtype = inferTextDtype(cell);
            if (dtype !== "string") {
                expect(parseTextCell(cell), cell).toBe(coreParseText(cell, dtype));
            }
        }
    });

    it("parseTextCell yields the JS value the sink infers from", () => {
        expect(parseTextCell("true")).toBe(true);
        expect(parseTextCell("false")).toBe(false);
        expect(parseTextCell("1")).toBe(1);
        expect(parseTextCell("1.5")).toBe(1.5);
        expect(parseTextCell("1e3")).toBe(1000);
        expect(parseTextCell("01")).toBe("01");
        expect(parseTextCell("abc")).toBe("abc");
        expect(parseTextCell("")).toBe("");
        expect(isNumericText("1")).toBe(true);
        expect(isNumericText("1.5")).toBe(true);
        expect(isNumericText("01")).toBe(false);
        expect(isNumericText("true")).toBe(false);
    });

    it("pushes into the builder as untyped cells that widen per column, never per cell", () => {
        const b = new GraphBuilder({ directed: true });
        const a = b.addNode("a");
        const c = b.addNode("c");
        const d = b.addNode("d");
        b.setNodeValue("v", a, parseTextCell("1"));
        b.setNodeValue("v", c, parseTextCell("2.5"));
        b.setNodeValue("v", d, parseTextCell("01"));
        b.setNodeValue("w", a, parseTextCell("true"));
        b.setNodeValue("w", c, parseTextCell("0"));
        const { snapshot, report } = b.freezeWithReport();
        const v = snapshot.nodes.require("v");
        expect(v.dtype).toBe("string");
        expect([0, 1, 2].map((r) => v.value(r))).toEqual(["1", "2.5", "01"]);
        const w = snapshot.nodes.require("w");
        expect(w.dtype).toBe("i32");
        expect([0, 1].map((r) => w.value(r))).toEqual([1, 0]);
        expect(report.widened.map((x) => `${x.column}:${x.from}->${x.to}`)).toContain("v:f64->string");
    });
});
