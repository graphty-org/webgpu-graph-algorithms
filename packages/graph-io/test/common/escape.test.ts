import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    decodeGmlString,
    escapeXmlAttribute,
    escapeXmlText,
    isBareDotId,
    isPajekLabel,
    isWritableDotText,
    quoteCsvCell,
    quoteDotId,
    quoteGmlString,
    quotePajekLabel,
} from "../../src/common/escape.js";

const E_ACUTE = String.fromCodePoint(0xe9);
const EMOJI = String.fromCodePoint(0x1f600);

describe("XML escaping", () => {
    it("escapes text and attribute values", () => {
        expect(escapeXmlText("a < b & c > d \"q\" 'x'")).toBe("a &lt; b &amp; c &gt; d \"q\" 'x'");
        expect(escapeXmlAttribute("a < b & \"q\" 'x'\tt\nn\rr")).toBe(
            "a &lt; b &amp; &quot;q&quot; 'x'&#9;t&#10;n&#13;r",
        );
        expect(escapeXmlText("plain")).toBe("plain");
        expect(escapeXmlText(E_ACUTE)).toBe(E_ACUTE);
    });
});

describe("GML strings", () => {
    it("quotes with entities for quotes, ampersands and non-ASCII, and decodes back", () => {
        expect(quoteGmlString("plain")).toBe('"plain"');
        expect(quoteGmlString('say "hi" & bye')).toBe('"say &#34;hi&#34; &#38; bye"');
        expect(quoteGmlString(`caf${E_ACUTE}`)).toBe('"caf&#233;"');
        expect(quoteGmlString(EMOJI)).toBe('"&#128512;"');
        expect(quoteGmlString("tab\there")).toBe('"tab&#9;here"');
        for (const text of ["plain", 'say "hi" & bye', `caf${E_ACUTE}`, EMOJI, "<>", "a&amp;b"]) {
            const quoted = quoteGmlString(text);
            expect(decodeGmlString(quoted.slice(1, -1)), text).toBe(text);
        }
    });

    it("decodes named and hex entities and leaves unknown ones", () => {
        expect(decodeGmlString("&amp;&quot;&lt;&gt;&apos;")).toBe("&\"<>'");
        expect(decodeGmlString("&#x41;&#66;")).toBe("AB");
        expect(decodeGmlString("&unknown;")).toBe("&unknown;");
    });

    it("refuses a lone surrogate", () => {
        expect(() => quoteGmlString(`a${String.fromCharCode(0xd800)}b`)).toThrow(GraphFormatError);
    });
});

describe("DOT ids", () => {
    it("knows which ids can be bare", () => {
        expect(isBareDotId("node1")).toBe(true);
        expect(isBareDotId("_x")).toBe(true);
        expect(isBareDotId("42")).toBe(true);
        expect(isBareDotId("-1.5")).toBe(true);
        expect(isBareDotId(".5")).toBe(true);
        expect(isBareDotId("5.")).toBe(true);
        expect(isBareDotId(`caf${E_ACUTE}`)).toBe(true);
        expect(isBareDotId("1a")).toBe(false);
        expect(isBareDotId("a b")).toBe(false);
        expect(isBareDotId("")).toBe(false);
        expect(isBareDotId("node")).toBe(false);
        expect(isBareDotId("Graph")).toBe(false);
        expect(isBareDotId("strict")).toBe(false);
        expect(isBareDotId("a-b")).toBe(false);
        expect(isBareDotId("1e5")).toBe(false);
    });

    it("quotes with only the double quote escaped", () => {
        expect(quoteDotId("node1")).toBe("node1");
        expect(quoteDotId("a b")).toBe('"a b"');
        expect(quoteDotId('say "hi"')).toBe('"say \\"hi\\""');
        expect(quoteDotId("back\\slash")).toBe('"back\\slash"');
        expect(quoteDotId("edge")).toBe('"edge"');
        expect(quoteDotId("")).toBe('""');
    });

    it("refuses a trailing backslash and a backslash before a quote as unwritable (Graphviz consumes backslash pairs)", () => {
        expect(isWritableDotText("plain")).toBe(true);
        expect(isWritableDotText("back\\slash")).toBe(true);
        expect(isWritableDotText('say "hi"')).toBe(true);
        expect(isWritableDotText("")).toBe(true);
        expect(isWritableDotText("trailing\\")).toBe(false);
        expect(isWritableDotText("\\")).toBe(false);
        // written as \\" the scanner reads a pair and the closing quote
        expect(isWritableDotText('a\\"b')).toBe(false);
        expect(isWritableDotText('a\\b"c')).toBe(true);
    });
});

describe("CSV cells", () => {
    it("quotes per RFC 4180", () => {
        expect(quoteCsvCell("plain")).toBe("plain");
        // a set empty string is the quoted empty cell; an unset cell is written as nothing
        expect(quoteCsvCell("")).toBe('""');
        expect(quoteCsvCell("a,b")).toBe('"a,b"');
        expect(quoteCsvCell('say "hi"')).toBe('"say ""hi"""');
        expect(quoteCsvCell("line\nbreak")).toBe('"line\nbreak"');
        expect(quoteCsvCell(" padded")).toBe('" padded"');
        expect(quoteCsvCell("a;b")).toBe("a;b");
        expect(quoteCsvCell("a;b", ";")).toBe('"a;b"');
        expect(quoteCsvCell("a\tb", "\t")).toBe('"a\tb"');
    });
});

describe("Pajek labels", () => {
    it("writes bare single tokens and quotes the rest", () => {
        expect(isPajekLabel("ok")).toBe(true);
        expect(isPajekLabel('no "quotes"')).toBe(false);
        expect(isPajekLabel("no\nnewline")).toBe(false);
        expect(quotePajekLabel("ok")).toBe("ok");
        expect(quotePajekLabel("two words")).toBe('"two words"');
        expect(quotePajekLabel("")).toBe('""');
        expect(quotePajekLabel("tab\tbed")).toBe('"tab\tbed"');
        let caught: unknown;
        try {
            quotePajekLabel('has "quote"');
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        expect((caught as GraphFormatError).code).toBe("E_UNSUPPORTED");
        expect((caught as GraphFormatError).details.reason).toBe("pajek label");
    });
});
