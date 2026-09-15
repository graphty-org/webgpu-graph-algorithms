import { describe, expect, it } from "vitest";

import {
    EMPTY_LIST_TEXT,
    GmlSyntaxError,
    isGmlKey,
    isNonFiniteWord,
    LIST_START_MARKER,
    mangleGmlKey,
    numberOfText,
    parseRecord,
    SYNTAX_BRACKET_CODE,
    SYNTAX_STRING_CODE,
    SYNTAX_STRUCTURE_CODE,
    SYNTAX_TOKEN_CODE,
    TOKEN_CLOSE,
    TOKEN_INT,
    TOKEN_OPEN,
    TOKEN_REAL,
    TOKEN_STRING,
    TOKEN_WORD,
    tokenizeGml,
} from "../../../src/formats/gml/syntax.js";

function kinds(text: string): number[] {
    const t = tokenizeGml(text);
    return Array.from(t.kind.subarray(0, t.count));
}

function texts(text: string): string[] {
    const t = tokenizeGml(text);
    const out: string[] = [];
    for (let i = 0; i < t.count; i++) {
        out.push(t.textOf(i));
    }
    return out;
}

function syntaxError(text: string): GmlSyntaxError {
    try {
        tokenizeGml(text);
    } catch (err) {
        expect(err).toBeInstanceOf(GmlSyntaxError);
        return err as GmlSyntaxError;
    }
    throw new Error("expected a GmlSyntaxError");
}

describe("tokenizeGml: lexing", () => {
    it("lexes keys, ints, reals, strings and brackets", () => {
        expect(kinds('graph [ id 1 x 2.5 label "A" ]')).toEqual([
            TOKEN_WORD,
            TOKEN_OPEN,
            TOKEN_WORD,
            TOKEN_INT,
            TOKEN_WORD,
            TOKEN_REAL,
            TOKEN_WORD,
            TOKEN_STRING,
            TOKEN_CLOSE,
        ]);
        expect(texts('graph [ id 1 x 2.5 label "A" ]')).toEqual([
            "graph",
            "[",
            "id",
            "1",
            "x",
            "2.5",
            "label",
            "A",
            "]",
        ]);
    });

    it("accepts every real spelling and signed ints", () => {
        const t = tokenizeGml("a 1. b .5 c -1.5e3 d 1e5 e +2 f -0 g +INF h -inf i NAN j Infinity");
        const numeric = [];
        for (let i = 1; i < t.count; i += 2) {
            numeric.push([t.kind[i], numberOfText(t.textOf(i))]);
        }
        expect(numeric).toEqual([
            [TOKEN_REAL, 1],
            [TOKEN_REAL, 0.5],
            [TOKEN_REAL, -1500],
            [TOKEN_REAL, 100000],
            [TOKEN_INT, 2],
            [TOKEN_INT, -0],
            [TOKEN_REAL, Infinity],
            [TOKEN_REAL, -Infinity],
            [TOKEN_WORD, NaN],
            [TOKEN_WORD, Infinity],
        ]);
        expect(isNonFiniteWord(t, 17)).toBe(true);
        expect(isNonFiniteWord(t, 19)).toBe(true);
        expect(isNonFiniteWord(t, 0)).toBe(false);
    });

    it("splits brackets and quotes from bare words without whitespace", () => {
        expect(texts('node[id 1]edge[label"x"]')).toEqual([
            "node",
            "[",
            "id",
            "1",
            "]",
            "edge",
            "[",
            "label",
            "x",
            "]",
        ]);
    });

    it("skips comments outside strings and keeps # inside strings", () => {
        const text = '# leading comment\ngraph [ # trailing\n  label "a # b" # c\n]';
        expect(texts(text)).toEqual(["graph", "[", "label", "a # b", "]"]);
    });

    it("numbers lines across LF, CRLF and lone CR", () => {
        const t = tokenizeGml("a 1\nb 2\r\nc 3\rd 4");
        expect(Array.from(t.line.subarray(0, t.count))).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });

    it("keeps a string body raw and decodes references on demand", () => {
        const t = tokenizeGml('label "Tom &amp; Jerry &#233; &#x41; TexasA&M"');
        expect(t.textOf(1)).toBe("Tom &amp; Jerry &#233; &#x41; TexasA&M");
        expect(t.stringOf(1)).toBe(`Tom & Jerry ${String.fromCharCode(233)} A TexasA&M`);
    });

    it("grows past the initial capacity", () => {
        const many = Array.from({ length: 5000 }, (_, i) => `k${i} ${i}`).join(" ");
        const t = tokenizeGml(many);
        expect(t.count).toBe(10000);
        expect(t.textOf(9999)).toBe("4999");
    });

    it("rejects a token that is neither a key nor a number", () => {
        const err = syntaxError("graph [\n !@$% 1 ]");
        expect(err.code).toBe(SYNTAX_TOKEN_CODE);
        expect(err.line).toBe(2);
        expect(err.message).toContain("!@$%");
    });

    it("rejects a string that does not close on its line", () => {
        const err = syntaxError('graph [\n  label "open\n]');
        expect(err.code).toBe(SYNTAX_STRING_CODE);
        expect(err.line).toBe(2);
    });
});

describe("tokenizeGml: structure", () => {
    it("records the matching bracket of every open bracket", () => {
        const t = tokenizeGml("graph [ node [ id 1 ] node [ id 2 graphics [ x 1 ] ] ]");
        expect(t.match[1]).toBe(t.count - 1);
        expect(t.match[3]).toBe(6);
        expect(t.isBracket(1)).toBe(true);
        expect(t.isBracket(0)).toBe(false);
        expect(t.nextPair(0)).toBe(t.count);
        expect(t.nextPair(2)).toBe(7);
        expect(t.nextPair(4)).toBe(6);
    });

    it("rejects an unclosed bracket with the line it opened on", () => {
        const err = syntaxError("graph [\n  node [\n    id 1\n");
        expect(err.code).toBe(SYNTAX_BRACKET_CODE);
        expect(err.line).toBe(2);
        expect(err.message).toContain("2 bracket(s) still open");
        const outer = syntaxError("graph [\n  node [\n    id 1\n]");
        expect(outer.line).toBe(1);
    });

    it("rejects a stray close bracket", () => {
        const err = syntaxError("graph [ ] ]");
        expect(err.code).toBe(SYNTAX_STRUCTURE_CODE);
        expect(err.message).toContain('unexpected "]"');
    });

    it("rejects a value without a key and a key without a value", () => {
        expect(syntaxError("1 2").code).toBe(SYNTAX_STRUCTURE_CODE);
        expect(syntaxError("graph [ id ]").code).toBe(SYNTAX_STRUCTURE_CODE);
        const dangling = syntaxError("graph [ id 1 ] tail");
        expect(dangling.code).toBe(SYNTAX_STRUCTURE_CODE);
        expect(dangling.message).toContain('"tail"');
    });

    it("rejects a bare word at a value position unless it is INF or NAN", () => {
        expect(syntaxError("This is not GML").code).toBe(SYNTAX_STRUCTURE_CODE);
        expect(() => tokenizeGml("a INF b NAN c nan")).not.toThrow();
    });

    it("accepts an empty text", () => {
        expect(tokenizeGml("").count).toBe(0);
        expect(tokenizeGml("   \n# only a comment\n").count).toBe(0);
    });
});

describe("parseRecord", () => {
    function record(text: string): Record<string, unknown> {
        const t = tokenizeGml(`r [ ${text} ]`);
        return parseRecord(t, 1);
    }

    it("maps scalars, nested records and strings", () => {
        expect(record('x 1 y 2.5 fill "#ff0000" Line [ point [ x 0 y 1 ] ]')).toEqual({
            x: 1,
            y: 2.5,
            fill: "#ff0000",
            Line: { point: { x: 0, y: 1 } },
        });
    });

    it("turns repeated keys into arrays and honours the NetworkX list conventions", () => {
        expect(record("a 1 a 2 a 3")).toEqual({ a: [1, 2, 3] });
        expect(record(`b "${LIST_START_MARKER}" b 7`)).toEqual({ b: [7] });
        expect(record(`c "${EMPTY_LIST_TEXT}" d "()"`)).toEqual({ c: [], d: [] });
        expect(record("e 1 f 2")).toEqual({ e: 1, f: 2 });
        expect(record(`g "${LIST_START_MARKER}"`)).toEqual({ g: [] });
    });

    it("keeps __proto__ as a plain key", () => {
        const r = record("__proto__ 1");
        expect(Object.keys(r)).toEqual(["__proto__"]);
        expect(r.__proto__).toBe(1);
    });

    it("parses non-finite values", () => {
        expect(record("a +INF b -INF")).toEqual({ a: Infinity, b: -Infinity });
        expect(Number.isNaN(record("c NAN").c)).toBe(true);
    });
});

describe("keys", () => {
    it("accepts NetworkX keys and rejects the rest", () => {
        expect(isGmlKey("id")).toBe(true);
        expect(isGmlKey("Label_2")).toBe(true);
        expect(isGmlKey("_x")).toBe(false);
        expect(isGmlKey("1a")).toBe(false);
        expect(isGmlKey("a.b")).toBe(false);
        expect(isGmlKey("")).toBe(false);
    });

    it("mangles to a valid key", () => {
        expect(mangleGmlKey("graphty.pagerank.rank")).toBe("graphty_pagerank_rank");
        expect(mangleGmlKey("_x")).toBe("x_x");
        expect(mangleGmlKey("1a")).toBe("x1a");
        expect(mangleGmlKey("a b")).toBe("a_b");
        expect(isGmlKey(mangleGmlKey(""))).toBe(true);
    });
});
