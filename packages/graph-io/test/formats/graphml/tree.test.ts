import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { tokenizeXml, type XmlHandler } from "../../../src/common/xml.js";
import { isXmlTree, treeProblem, writeXmlTree, XmlTreeBuilder } from "../../../src/formats/graphml/tree.js";
import { textChunksOf } from "../../helpers/corpus.js";

/**
 * Parse the content of a `<data>` element (the text between the tags) into a tree the way the
 * importer does: events of the children go to the builder, the outer element is the root.
 */
async function parseContent(inner: string): Promise<unknown> {
    const builder = new XmlTreeBuilder();
    let depth = 0;
    const handler: XmlHandler = {
        start: (name, attrs) => {
            if (depth > 0) {
                builder.start(name, attrs);
            }
            depth++;
        },
        end: () => {
            depth--;
            if (depth > 0) {
                builder.end();
            }
        },
        text: (text) => builder.text(text),
    };
    await tokenizeXml(textChunksOf(`<data>${inner}</data>`, 5), handler);
    return builder.finish();
}

function write(value: unknown): string {
    const out: string[] = [];
    writeXmlTree(value, "", out);
    return out.join("");
}

describe("XmlTreeBuilder", () => {
    it("returns the text of a data element without children", async () => {
        expect(await parseContent("plain text")).toBe("plain text");
        expect(await parseContent("")).toBe("");
        expect(await parseContent("  padded  ")).toBe("  padded  ");
    });

    it("maps elements to objects with @_ attributes, #text and child keys", async () => {
        const tree = await parseContent(
            '<y:ShapeNode>\n  <y:Geometry x="0.0" y="1.5"/>\n  <y:NodeLabel>Start</y:NodeLabel>\n  <y:Shape type="rectangle"/>\n  <y:Empty/>\n  <y:Mixed a="1">text</y:Mixed>\n</y:ShapeNode>',
        );
        expect(tree).toEqual({
            "y:ShapeNode": {
                "y:Geometry": { "@_x": "0.0", "@_y": "1.5" },
                "y:NodeLabel": "Start",
                "y:Shape": { "@_type": "rectangle" },
                "y:Empty": "",
                "y:Mixed": { "@_a": "1", "#text": "text" },
            },
        });
    });

    it("turns repeated child names into arrays and keeps mixed content text", async () => {
        const tree = await parseContent("before<p>one</p><p>two</p><p a='x'>three</p>after");
        expect(tree).toEqual({
            p: ["one", "two", { "@_a": "x", "#text": "three" }],
            "#text": "beforeafter",
        });
    });

    it("drops whitespace-only text between child elements", async () => {
        const tree = await parseContent("\n  <a>\n    <b>x</b>\n  </a>\n");
        expect(tree).toEqual({ a: { b: "x" } });
    });

    it("refuses end() without start() and finish() with open elements", () => {
        const builder = new XmlTreeBuilder();
        expect(() => builder.end()).toThrow(GraphFormatError);
        builder.start("a", new Map());
        expect(builder.open).toBe(true);
        expect(() => builder.finish()).toThrow(GraphFormatError);
        builder.end();
        expect(builder.open).toBe(false);
        expect(builder.finish()).toEqual({ a: "" });
    });
});

describe("writeXmlTree", () => {
    it("writes scalars as escaped text", () => {
        expect(write("a < b & c")).toBe("a &lt; b &amp; c");
        expect(write(42)).toBe("42");
        expect(write(true)).toBe("true");
        expect(write(null)).toBe("");
    });

    it("writes leaf elements inline and elements with children over indented lines", () => {
        const text = write({
            "y:ShapeNode": {
                "y:Geometry": { "@_x": "0.0", "@_y": "1.5" },
                "y:NodeLabel": "Start & end",
                "y:Empty": "",
                "y:Mixed": { "@_a": '"q"', "#text": "text" },
                "y:List": ["one", "two"],
            },
        });
        expect(text).toBe(
            [
                "",
                "<y:ShapeNode>",
                '  <y:Geometry x="0.0" y="1.5"/>',
                "  <y:NodeLabel>Start &amp; end</y:NodeLabel>",
                "  <y:Empty/>",
                '  <y:Mixed a="&quot;q&quot;">text</y:Mixed>',
                "  <y:List>one</y:List>",
                "  <y:List>two</y:List>",
                "</y:ShapeNode>",
            ].join("\n"),
        );
    });

    it("writes mixed content text after the children", () => {
        expect(write({ "#text": "tail", p: "x" })).toBe("\n<p>x</p>tail");
    });

    it("round-trips a parsed tree through the writer and the builder", async () => {
        const source =
            '<y:PolyLineEdge>\n  <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>\n  <y:LineStyle color="#000000" type="line" width="1.0"/>\n  <y:Arrows source="none" target="standard"/>\n  <y:EdgeLabel alignment="center">Label &amp; more</y:EdgeLabel>\n</y:PolyLineEdge>';
        const tree = await parseContent(source);
        const written = write(tree);
        expect(await parseContent(written)).toEqual(tree);
    });

    it("refuses values that are not trees, naming the problem", () => {
        expect(treeProblem("x")).toBeNull();
        expect(treeProblem({ a: { "@_b": "c", "#text": "d" } })).toBeNull();
        expect(treeProblem([[1]])).toMatch(/nested arrays/);
        expect(treeProblem({ "bad name": "x" })).toMatch(/not an XML element name/);
        expect(treeProblem({ "@_bad name": "x" })).toMatch(/not an XML attribute name/);
        expect(treeProblem({ "@_a": { nested: 1 } })).toMatch(/must be a scalar/);
        expect(treeProblem({ "#text": { nested: 1 } })).toMatch(/#text must be a scalar/);
        expect(treeProblem(undefined)).toMatch(/undefined cannot be written/);
        expect(isXmlTree({ a: "b" })).toBe(true);
        expect(isXmlTree({ "1a": "b" })).toBe(false);
        expect(() => write({ "bad name": "x" })).toThrow(GraphFormatError);
    });
});
