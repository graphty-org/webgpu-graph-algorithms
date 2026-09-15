import { describe, expect, it } from "vitest";

import {
    decodeEntities,
    isWhitespace,
    isXmlName,
    localName,
    tokenizeXml,
    type XmlHandler,
    XmlSyntaxError,
    XmlTokenizer,
} from "../../src/common/xml.js";
import { textChunksOf } from "../helpers/corpus.js";

type Event = ["start", string, Record<string, string>, number] | ["end", string, number] | ["text", string, number];

function recorder(): { events: Event[]; handler: XmlHandler } {
    const events: Event[] = [];
    return {
        events,
        handler: {
            start: (name, attrs, line) => events.push(["start", name, Object.fromEntries(attrs), line]),
            end: (name, line) => events.push(["end", name, line]),
            text: (text, line) => events.push(["text", text, line]),
        },
    };
}

async function tokenize(text: string, chunkLength = text.length): Promise<Event[]> {
    const { events, handler } = recorder();
    await tokenizeXml(textChunksOf(text, chunkLength), handler);
    return events;
}

function withoutWhitespace(events: Event[]): Event[] {
    return events.filter((e) => e[0] !== "text" || !isWhitespace(e[1]));
}

async function failure(text: string, chunkLength = text.length): Promise<XmlSyntaxError> {
    try {
        await tokenize(text, chunkLength);
    } catch (err) {
        expect(err).toBeInstanceOf(XmlSyntaxError);
        return err as XmlSyntaxError;
    }
    throw new Error("expected an XmlSyntaxError");
}

const SAMPLE = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE graphml [ <!-- a > comment --> <!ELEMENT x ANY> ]>
<!-- leading comment -->
<root a="1" b='two' c="x &amp; y &#65; &#x42; &lt;tag&gt;">
  <child/>
  <child key="k">text &amp; more</child>
  <![CDATA[raw <cdata> & stuff]]>
  <?pi target?>
  <ns:elem ns:attr="v"></ns:elem>
</root>
<!-- trailing comment -->
`;

describe("tokenizeXml events", () => {
    it("delivers start, end and text events with decoded entities and line numbers", async () => {
        const events = withoutWhitespace(await tokenize(SAMPLE));
        expect(events).toEqual([
            ["start", "root", { a: "1", b: "two", c: "x & y A B <tag>" }, 4],
            ["start", "child", {}, 5],
            ["end", "child", 5],
            ["start", "child", { key: "k" }, 6],
            ["text", "text & more", 6],
            ["end", "child", 6],
            ["text", "\n  raw <cdata> & stuff\n  \n  ", 6],
            ["start", "ns:elem", { "ns:attr": "v" }, 9],
            ["end", "ns:elem", 9],
            ["end", "root", 10],
        ]);
    });

    it("produces the same events whatever the chunk size", async () => {
        const reference = await tokenize(SAMPLE);
        for (const size of [1, 2, 3, 5, 7, 11, 64]) {
            expect(await tokenize(SAMPLE, size), `chunk size ${size}`).toEqual(reference);
        }
    });

    it("normalises CR LF and lone CR to LF, also when the CR ends a chunk", async () => {
        const text = "<r>\r\n<a>x\ry</a>\r\n</r>\r";
        const expected = [
            ["start", "r", {}, 1],
            ["text", "\n", 1],
            ["start", "a", {}, 2],
            ["text", "x\ny", 2],
            ["end", "a", 3],
            ["text", "\n", 3],
            ["end", "r", 4],
        ];
        expect(await tokenize(text)).toEqual(expected);
        expect(await tokenize(text, 4)).toEqual(expected);
        expect(await tokenize(text, 1)).toEqual(expected);
    });

    it("normalises literal tabs and line breaks in attribute values but keeps character references", async () => {
        const events = await tokenize('<r a="one\ttwo\nthree" b="one&#10;two&#9;three"/>');
        expect(events[0]).toEqual(["start", "r", { a: "one two three", b: "one\ntwo\tthree" }, 1]);
    });

    it("keeps an entity reference split across chunks intact", async () => {
        const text = "<r>a &amp; b &#x1F600; c</r>";
        const expected = [
            ["start", "r", {}, 1],
            ["text", `a & b ${String.fromCodePoint(0x1f600)} c`, 1],
            ["end", "r", 1],
        ];
        for (const size of [1, 2, 3, 4, 5]) {
            expect(await tokenize(text, size)).toEqual(expected);
        }
    });

    it("keeps a multi-byte name and text intact across chunks", async () => {
        const eacute = String.fromCharCode(0xe9);
        const text = `<r${eacute}sum${eacute} x="${eacute}">caf${eacute}</r${eacute}sum${eacute}>`;
        const expected = [
            ["start", `r${eacute}sum${eacute}`, { x: eacute }, 1],
            ["text", `caf${eacute}`, 1],
            ["end", `r${eacute}sum${eacute}`, 1],
        ];
        for (const size of [1, 2, 3]) {
            expect(await tokenize(text, size)).toEqual(expected);
        }
    });

    it("accepts a document without an XML declaration and with a UTF-8 BOM already stripped", async () => {
        expect(await tokenize("<a><b/></a>")).toEqual([
            ["start", "a", {}, 1],
            ["start", "b", {}, 1],
            ["end", "b", 1],
            ["end", "a", 1],
        ]);
    });

    it("accepts whitespace and comments outside the root", async () => {
        const events = await tokenize("\n<!-- c -->\n<a/>\n<!-- d -->\n");
        expect(events).toEqual([
            ["start", "a", {}, 3],
            ["end", "a", 3],
        ]);
    });

    it("accepts a tiny document that ends inside the short-markup window", async () => {
        expect(await tokenize("<!---->\n<a/>")).toEqual([
            ["start", "a", {}, 2],
            ["end", "a", 2],
        ]);
        expect(await tokenize("<a/>\n<!---->")).toEqual([
            ["start", "a", {}, 1],
            ["end", "a", 1],
        ]);
    });

    it("reports the line of a multi-line start tag as the line of its <", async () => {
        const events = withoutWhitespace(await tokenize('<a>\n<b\n  x="1"\n  y="2"\n/>\n</a>'));
        expect(events).toEqual([
            ["start", "a", {}, 1],
            ["start", "b", { x: "1", y: "2" }, 2],
            ["end", "b", 2],
            ["end", "a", 6],
        ]);
    });
});

describe("tokenizeXml errors", () => {
    it("rejects an end tag that does not match the open element, with its line", async () => {
        const err = await failure("<a>\n  <b>\n  </a>\n</b>");
        expect(err.message).toMatch(/<\/a> does not match <b>/);
        expect(err.line).toBe(3);
    });

    it("rejects an unclosed element at the end of the input", async () => {
        const err = await failure("<a>\n<b/>\n");
        expect(err.message).toMatch(/unclosed element <a>/);
        expect(err.line).toBe(3);
    });

    it("rejects an empty document and plain text", async () => {
        expect((await failure("")).message).toMatch(/no root element/);
        expect((await failure("   \n  ")).message).toMatch(/no root element/);
        const err = await failure("This is not XML at all.\nJust text.");
        expect(err.message).toMatch(/text outside the root element/);
        expect(err.line).toBe(1);
    });

    it("rejects text after the root and a second root", async () => {
        expect((await failure("<a/>\nstray")).message).toMatch(/text outside the root element/);
        expect((await failure("<a/><b/>")).message).toMatch(/second root element <b>/);
    });

    it("rejects an end tag without an open element", async () => {
        expect((await failure("</a>")).message).toMatch(/unexpected end tag <\/a>/);
    });

    it("rejects unterminated markup at the end of the input", async () => {
        for (const text of [
            "<a",
            '<a x="1',
            "<a><!-- never closed",
            "<a><![CDATA[open",
            "<a><?pi",
            "<a></a",
            "<a>&amp",
        ]) {
            const err = await failure(text);
            expect(err.message, text).toMatch(/unexpected end of input|unterminated entity/);
        }
    });

    it("rejects malformed attributes", async () => {
        expect((await failure("<a x=1/>")).message).toMatch(/not quoted/);
        expect((await failure("<a x/>")).message).toMatch(/has no value/);
        expect((await failure('<a x="1" x="2"/>')).message).toMatch(/duplicate attribute x/);
        expect((await failure("<a =1/>")).message).toMatch(/malformed attribute/);
        expect((await failure("<a /x>")).message).toMatch(/unexpected "\/"/);
    });

    it("rejects malformed names and declarations", async () => {
        expect((await failure("< a/>")).message).toMatch(/expected an element name/);
        expect((await failure("<1a/>")).message).toMatch(/expected an element name/);
        expect((await failure("<a></1>")).message).toMatch(/malformed end tag/);
        expect((await failure("<!ENTITY x 'y'><a/>")).message).toMatch(/unexpected markup declaration/);
    });

    it("rejects unknown entities and invalid character references", async () => {
        expect((await failure("<a>&nbsp;</a>")).message).toMatch(/unknown entity &nbsp;/);
        expect((await failure("<a>&#0;</a>")).message).toMatch(/invalid character reference/);
        expect((await failure("<a>&#x110000;</a>")).message).toMatch(/invalid character reference/);
        expect((await failure("<a>&#xZZ;</a>")).message).toMatch(/invalid character reference/);
        expect((await failure('<a x="&bogus;"/>')).message).toMatch(/unknown entity &bogus;/);
    });

    it("propagates an error thrown by the handler unchanged", async () => {
        const boom = new Error("handler failed");
        const handler: XmlHandler = {
            start: () => {
                throw boom;
            },
            end: () => undefined,
            text: () => undefined,
        };
        await expect(tokenizeXml(textChunksOf("<a/>", 4), handler)).rejects.toBe(boom);
    });

    it("finish() on a pushed-only tokenizer checks well-formedness", () => {
        const { handler } = recorder();
        const tokenizer = new XmlTokenizer(handler);
        tokenizer.push("<a>");
        expect(() => tokenizer.finish()).toThrow(XmlSyntaxError);
    });
});

describe("helpers", () => {
    it("decodeEntities decodes the predefined entities and numeric references", () => {
        expect(decodeEntities("plain", 1)).toBe("plain");
        expect(decodeEntities("&lt;&gt;&amp;&quot;&apos;", 1)).toBe("<>&\"'");
        expect(decodeEntities("&#65;&#x41;&#X41;", 1)).toBe("AAA");
        expect(() => decodeEntities("&amp", 1)).toThrow(/unterminated/);
    });

    it("isXmlName follows the XML Name production", () => {
        expect(isXmlName("a")).toBe(true);
        expect(isXmlName("ns:b-c.d_e1")).toBe(true);
        expect(isXmlName(`x${String.fromCharCode(0xe9)}`)).toBe(true);
        expect(isXmlName("")).toBe(false);
        expect(isXmlName("1a")).toBe(false);
        expect(isXmlName("-a")).toBe(false);
        expect(isXmlName(".a")).toBe(false);
        expect(isXmlName("a b")).toBe(false);
        expect(isXmlName("a<")).toBe(false);
    });

    it("isWhitespace and localName", () => {
        expect(isWhitespace("")).toBe(true);
        expect(isWhitespace(" \t\n\r")).toBe(true);
        expect(isWhitespace(" x ")).toBe(false);
        expect(localName("y:ShapeNode")).toBe("ShapeNode");
        expect(localName("node")).toBe("node");
    });
});
