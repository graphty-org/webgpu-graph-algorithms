import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { joinListText, type ListSyntax, splitListText } from "../../src/common/lists.js";

describe("splitListText", () => {
    it("parses GEXF 1.3 bracket lists with quotes", () => {
        expect(splitListText("[1, 2, 3]", "gexf")).toEqual(["1", "2", "3"]);
        expect(splitListText("[foo, bar]", "gexf")).toEqual(["foo", "bar"]);
        expect(splitListText("['a, b', \"c\"]", "gexf")).toEqual(["a, b", "c"]);
        expect(splitListText('["say ""hi""", x]', "gexf")).toEqual(['say "hi"', "x"]);
        expect(splitListText("[]", "gexf")).toEqual([]);
        expect(splitListText("[ ]", "gexf")).toEqual([]);
        expect(splitListText("[a]", "gexf")).toEqual(["a"]);
        expect(splitListText("[a,]", "gexf")).toEqual(["a", ""]);
        expect(splitListText("[a,,b]", "gexf")).toEqual(["a", "", "b"]);
        expect(splitListText("  [ a , b ]  ", "gexf")).toEqual(["a", "b"]);
        expect(splitListText("['unterminated, b]", "gexf")).toEqual(["unterminated, b"]);
        expect(splitListText("['q' tail, b]", "gexf")).toEqual(["qtail", "b"]);
    });

    it("applies the GEXF 1.2 separator rule: pipe, then comma, then semicolon", () => {
        expect(splitListText("a|b|c", "gexf")).toEqual(["a", "b", "c"]);
        expect(splitListText("a,b;c", "gexf")).toEqual(["a", "b;c"]);
        expect(splitListText("a;b", "gexf")).toEqual(["a", "b"]);
        expect(splitListText("a| b ,c", "gexf")).toEqual(["a", "b ,c"]);
        expect(splitListText("single", "gexf")).toEqual(["single"]);
        expect(splitListText("", "gexf")).toEqual([]);
        expect(splitListText("   ", "gexf")).toEqual([]);
    });

    it("splits on one separator for the fixed syntaxes", () => {
        expect(splitListText("a;b; c", "semicolon")).toEqual(["a", "b", "c"]);
        expect(splitListText("a,b", "semicolon")).toEqual(["a,b"]);
        expect(splitListText("a,b", "comma")).toEqual(["a", "b"]);
        expect(splitListText("a|b", "pipe")).toEqual(["a", "b"]);
        expect(splitListText("[a, b]", "brackets")).toEqual(["a", "b"]);
        expect(splitListText("a, b", "brackets")).toEqual(["a, b"]);
    });

    it("rejects an unknown syntax", () => {
        expect(() => splitListText("a", "weird" as ListSyntax)).toThrow(GraphFormatError);
        expect(() => joinListText(["a"], "weird" as ListSyntax)).toThrow(GraphFormatError);
    });
});

describe("joinListText", () => {
    it("writes bracket lists that split back to the same items", () => {
        const items = ["plain", "with, comma", 'with "quote"', "", " padded ", "[bracket]"];
        const text = joinListText(items, "gexf");
        expect(text).toBe('[plain, "with, comma", "with ""quote""", "", " padded ", "[bracket]"]');
        expect(splitListText(text, "gexf")).toEqual(items);
        expect(joinListText([], "brackets")).toBe("[]");
        expect(splitListText(joinListText(["1", "2"], "brackets"), "brackets")).toEqual(["1", "2"]);
    });

    it("joins the separator syntaxes", () => {
        expect(joinListText(["a", "b"], "pipe")).toBe("a|b");
        expect(joinListText(["a", "b"], "comma")).toBe("a,b");
        expect(joinListText(["a", "b"], "semicolon")).toBe("a;b");
        expect(splitListText(joinListText(["x", "y"], "semicolon"), "semicolon")).toEqual(["x", "y"]);
    });
});
