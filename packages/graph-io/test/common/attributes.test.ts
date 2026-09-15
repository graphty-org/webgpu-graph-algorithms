import { GraphBuilder, GraphFormatError, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    BAD_DEFAULT_CODE,
    BAD_OPTIONS_CODE,
    declareAttribute,
    declareCompanion,
    declareOn,
    DICT_SAMPLE_ROWS,
    DictHeuristic,
    losesPrecision,
    parseDeclaredTemporal,
    parseDeclaredValue,
    RENAMED_CODE,
    takenIn,
    uniqueColumnName,
    UNKNOWN_TYPE_CODE,
} from "../../src/common/attributes.js";

describe("declareAttribute (design 5.1, 5.5, 5.6)", () => {
    it("names the column after the title and records the source id, type and namespace in origin", () => {
        const d = declareAttribute({ format: "gexf", id: "0", title: "Code", type: "string", long: "f64" });
        expect(d.decl).toEqual({
            name: "Code",
            dtype: "string",
            nullable: true,
            origin: { format: "gexf", id: "0", title: null, type: "string", namespace: null },
        });
        expect(d.spec.kind).toBe("string");
        expect(d.companion).toBeNull();
        expect(d.renamed).toBe(false);
        expect(d.issues).toEqual([]);
        expect(d.listSyntax).toBe("gexf");
    });

    it("falls back to the id when the title is absent (yFiles keys)", () => {
        const d = declareAttribute({
            format: "graphml",
            id: "d6",
            title: null,
            type: null,
            namespace: "yfiles",
            long: "f64",
        });
        expect(d.decl.name).toBe("d6");
        expect(d.decl.dtype).toBe("string");
        expect(d.decl.origin).toEqual({ format: "graphml", id: "d6", title: null, type: null, namespace: "yfiles" });
        expect(d.issues).toEqual([]);
    });

    it("throws E_COLUMN_TYPE without a title or an id", () => {
        expect(() => declareAttribute({ format: "gexf", id: null, title: null, type: "string", long: "f64" })).toThrow(
            GraphFormatError,
        );
        expect(() => declareAttribute({ format: "gexf", id: "", title: "", type: "string", long: "f64" })).toThrow(
            GraphFormatError,
        );
    });

    it("maps every declared type, lists included, and parses defaults", () => {
        const i = declareAttribute({
            format: "gexf",
            id: "1",
            title: "age",
            type: "integer",
            defaultText: "7",
            long: "f64",
        });
        expect(i.decl.dtype).toBe("i32");
        expect(i.decl.default).toBe(7);
        const b = declareAttribute({
            format: "graphml",
            id: "d0",
            title: "flag",
            type: "boolean",
            defaultText: "1",
            long: "f64",
        });
        expect(b.decl.dtype).toBe("bool");
        expect(b.decl.default).toBe(true);
        const l = declareAttribute({
            format: "gexf",
            id: "2",
            title: "tags",
            type: "listinteger",
            defaultText: "[1, 2]",
            long: "f64",
        });
        expect(l.decl.dtype).toBe("list");
        expect(l.decl.itemDtype).toBe("i32");
        expect(l.decl.default).toEqual([1, 2]);
        const f = declareAttribute({
            format: "gexf",
            id: "3",
            title: "w",
            type: "float",
            defaultText: "0.5",
            long: "f64",
        });
        expect(f.decl.dtype).toBe("f32");
        expect(f.decl.default).toBe(0.5);
        const s = declareAttribute({ format: "gexf", id: "4", title: "big", type: "long", long: "string" });
        expect(s.decl.dtype).toBe("string");
        expect(s.decl.origin?.type).toBe("long");
    });

    it("keeps an unknown type as string and reports it", () => {
        const d = declareAttribute({ format: "graphml", id: "d1", title: "geo", type: "yfiles.type", long: "f64" });
        expect(d.decl.dtype).toBe("string");
        expect(d.decl.origin?.type).toBe("yfiles.type");
        expect(d.issues).toEqual([
            {
                category: "unsupported",
                code: UNKNOWN_TYPE_CODE,
                message: expect.stringContaining("yfiles.type") as string,
            },
        ]);
    });

    it("reports an unparsable default and declares without one", () => {
        const d = declareAttribute({
            format: "gexf",
            id: "1",
            title: "n",
            type: "integer",
            defaultText: "many",
            long: "f64",
        });
        expect(d.decl.default).toBeUndefined();
        expect(d.issues.map((i) => [i.category, i.code])).toEqual([["validation-error", BAD_DEFAULT_CODE]]);
    });

    it("turns string options into a dict column with the declared dictionary order", () => {
        const d = declareAttribute({
            format: "gexf",
            id: "5",
            title: "kind",
            type: "string",
            optionsText: "[red, green, blue]",
            defaultText: "green",
            long: "f64",
        });
        expect(d.decl.dtype).toBe("dict");
        expect(d.spec.dtype).toBe("dict");
        expect(d.decl.options).toEqual(["red", "green", "blue"]);
        expect(d.decl.default).toBe("green");
        const pipe = declareAttribute({
            format: "gexf",
            id: "6",
            title: "k",
            type: "string",
            optionsText: "a|b",
            long: "f64",
        });
        expect(pipe.decl.options).toEqual(["a", "b"]);
        expect(pipe.decl.dtype).toBe("dict");
    });

    it("keeps non-string options as metadata on the typed column", () => {
        const d = declareAttribute({
            format: "gexf",
            id: "7",
            title: "lvl",
            type: "integer",
            optionsText: "1|2|3",
            long: "f64",
        });
        expect(d.decl.dtype).toBe("i32");
        expect(d.decl.options).toEqual([1, 2, 3]);
        const l = declareAttribute({
            format: "gexf",
            id: "8",
            title: "tags",
            type: "liststring",
            optionsText: "[a, b]",
            long: "f64",
        });
        expect(l.decl.dtype).toBe("list");
        expect(l.decl.options).toEqual(["a", "b"]);
    });

    it("reports unparsable options and drops them", () => {
        const d = declareAttribute({
            format: "gexf",
            id: "7",
            title: "lvl",
            type: "integer",
            optionsText: "1|x",
            long: "f64",
        });
        expect(d.decl.options).toBeUndefined();
        expect(d.issues.map((i) => i.code)).toEqual([BAD_OPTIONS_CODE]);
    });

    it("declares a companion text column for scalar temporal attributes only", () => {
        const d = declareAttribute({ format: "gexf", id: "9", title: "born", type: "date", long: "f64" });
        expect(d.decl.dtype).toBe("f64");
        expect(d.companion).toEqual({
            name: "born.text",
            dtype: "string",
            role: "timeText",
            nullable: true,
            extra: { for: "born" },
        });
        const n = declareAttribute({ format: "neo4j", id: "seen", title: null, type: "datetime[]", long: "f64" });
        expect(n.decl.dtype).toBe("list");
        expect(n.companion).toBeNull();
    });

    it("renames a taken name to <name>#<id> deterministically and reports a coercion", () => {
        const taken = (name: string): boolean => name === "label";
        const d = declareAttribute({ format: "gexf", id: "3", title: "label", type: "string", taken, long: "f64" });
        expect(d.decl.name).toBe("label#3");
        expect(d.renamed).toBe(true);
        expect(d.decl.origin?.title).toBe("label");
        expect(d.issues.map((i) => [i.category, i.code])).toEqual([["coercion", RENAMED_CODE]]);
        const free = declareAttribute({ format: "gexf", id: "4", title: "other", type: "string", taken, long: "f64" });
        expect(free.decl.name).toBe("other");
        expect(free.renamed).toBe(false);
        expect(free.decl.origin?.title).toBeNull();
    });

    it("passes role and dynamic through", () => {
        const d = declareAttribute({
            format: "gexf",
            id: "w",
            title: "weight",
            type: "double",
            role: "capacity",
            dynamic: true,
            long: "f64",
        });
        expect(d.decl.role).toBe("capacity");
        expect(d.decl.dynamic).toBe(true);
    });
});

describe("uniqueColumnName", () => {
    it("suffixes #id, then #id#n", () => {
        const names = new Set(["label", "label#3", "label#3#2"]);
        const taken = (n: string): boolean => names.has(n);
        expect(uniqueColumnName("free", "1", taken)).toBe("free");
        expect(uniqueColumnName("label", "4", taken)).toBe("label#4");
        expect(uniqueColumnName("label", "3", taken)).toBe("label#3#3");
        expect(uniqueColumnName("label", null, taken)).toBe("label#2");
    });
});

describe("parseDeclaredValue / parseDeclaredTemporal / losesPrecision", () => {
    it("parses scalars and lists by the spec", () => {
        const list = declareAttribute({ format: "gexf", id: "1", title: "l", type: "listdouble", long: "f64" });
        expect(parseDeclaredValue("[1.5, 2]", list.spec, list.listSyntax)).toEqual([1.5, 2]);
        expect(parseDeclaredValue("", list.spec, list.listSyntax)).toEqual([]);
        const neo = declareAttribute({
            format: "neo4j",
            id: "tags",
            title: null,
            type: "string[]",
            listSyntax: "semicolon",
            long: "f64",
        });
        expect(parseDeclaredValue("a;b", neo.spec, neo.listSyntax)).toEqual(["a", "b"]);
        const int = declareAttribute({ format: "gexf", id: "2", title: "i", type: "integer", long: "f64" });
        expect(parseDeclaredValue(" 12 ", int.spec, int.listSyntax)).toBe(12);
        expect(() => parseDeclaredValue("x", int.spec, int.listSyntax)).toThrow(GraphFormatError);
    });

    it("keeps temporal source text for the companion", () => {
        const d = declareAttribute({ format: "gexf", id: "9", title: "born", type: "dateTime", long: "f64" });
        expect(parseDeclaredTemporal("2009-03-01T12:00:00Z", d.spec)).toEqual({
            value: Date.UTC(2009, 2, 1, 12),
            text: null,
        });
        expect(parseDeclaredTemporal("2009-03-01T12:00:00+01:00", d.spec)).toEqual({
            value: Date.UTC(2009, 2, 1, 11),
            text: "2009-03-01T12:00:00+01:00",
        });
        const s = declareAttribute({ format: "gexf", id: "1", title: "s", type: "string", long: "f64" });
        expect(() => parseDeclaredTemporal("x", s.spec)).toThrow(GraphFormatError);
    });

    it("flags precision loss for longs beyond 2^53 only", () => {
        const long = declareAttribute({ format: "gexf", id: "1", title: "l", type: "long", long: "f64" });
        expect(losesPrecision(long.spec, "9007199254740993")).toBe(true);
        expect(losesPrecision(long.spec, "9007199254740992")).toBe(false);
        expect(losesPrecision(long.spec, "-9007199254740993")).toBe(true);
        expect(losesPrecision(long.spec, " 12 ")).toBe(false);
        expect(losesPrecision(long.spec, `1${"0".repeat(400)}`)).toBe(true);
        expect(losesPrecision(long.spec, "abc")).toBe(false);
        const dbl = declareAttribute({ format: "gexf", id: "2", title: "d", type: "double", long: "f64" });
        expect(losesPrecision(dbl.spec, "9007199254740993")).toBe(false);
        const asText = declareAttribute({ format: "gexf", id: "3", title: "t", type: "long", long: "string" });
        expect(losesPrecision(asText.spec, "9007199254740993")).toBe(false);
    });
});

describe("declaring on a real builder", () => {
    it("declares typed columns whose values, defaults, options and origin survive the freeze", () => {
        const b = new GraphBuilder({ directed: true });
        const a = b.addNode("a");
        const c = b.addNode("c");
        const age = declareAttribute({
            format: "gexf",
            id: "0",
            title: "age",
            type: "integer",
            defaultText: "7",
            long: "f64",
        });
        const kind = declareAttribute({
            format: "gexf",
            id: "1",
            title: "kind",
            type: "string",
            optionsText: "[x, y]",
            long: "f64",
        });
        const tags = declareAttribute({ format: "gexf", id: "2", title: "tags", type: "liststring", long: "f64" });
        const hAge = declareOn(b, "node", age.decl);
        const hKind = declareOn(b, "node", kind.decl);
        const hTags = declareOn(b, "node", tags.decl);
        expect(hAge).not.toBe(INVALID_INDEX);
        b.setNodeValue(hAge, a, parseDeclaredValue("41", age.spec, age.listSyntax));
        b.setNodeValue(hKind, a, parseDeclaredValue("y", kind.spec, kind.listSyntax));
        b.setNodeValue(hKind, c, parseDeclaredValue("x", kind.spec, kind.listSyntax));
        b.setNodeValue(hTags, c, parseDeclaredValue("[p, q]", tags.spec, tags.listSyntax));
        const s = b.freeze();
        const ageCol = s.nodes.requireTyped("age", "i32");
        expect(ageCol.value(a)).toBe(41);
        expect(ageCol.isSet(c)).toBe(false);
        expect(ageCol.value(c)).toBe(7);
        expect(ageCol.meta.default).toBe(7);
        expect(ageCol.meta.origin).toEqual({ format: "gexf", id: "0", title: null, type: "integer", namespace: null });
        const kindCol = s.nodes.requireTyped("kind", "dict");
        expect(kindCol.dictionary.slice(0, 2)).toEqual(["x", "y"]);
        expect(kindCol.value(a)).toBe("y");
        expect(kindCol.meta.options).toEqual(["x", "y"]);
        const tagsCol = s.nodes.requireTyped("tags", "list");
        expect(tagsCol.sliceOf(c)).toEqual(["p", "q"]);
        expect(tagsCol.isSet(a)).toBe(false);
        expect(tagsCol.child.dtype).toBe("string");
    });

    it("takenIn answers for the sink's tables and drives the rename", () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
        const nodeTaken = takenIn(b, "node");
        const edgeTaken = takenIn(b, "edge");
        expect(nodeTaken("label")).toBe(true);
        expect(nodeTaken("other")).toBe(false);
        expect(edgeTaken("label")).toBe(false);
        const d = declareAttribute({
            format: "gexf",
            id: "3",
            title: "label",
            type: "string",
            taken: nodeTaken,
            long: "f64",
        });
        expect(d.decl.name).toBe("label#3");
        expect(declareOn(b, "node", d.decl)).not.toBe(INVALID_INDEX);
        expect(nodeTaken("label#3")).toBe(true);
    });

    it("declares companions, dropping the role when another companion already holds it", () => {
        const b = new GraphBuilder({ directed: true });
        const start = declareAttribute({ format: "gexf", id: "s", title: "start", type: "dateTime", long: "f64" });
        const end = declareAttribute({ format: "gexf", id: "e", title: "end", type: "dateTime", long: "f64" });
        declareOn(b, "node", start.decl);
        declareOn(b, "node", end.decl);
        const h1 = declareCompanion(b, "node", start.companion as NonNullable<typeof start.companion>);
        const h2 = declareCompanion(b, "node", end.companion as NonNullable<typeof end.companion>);
        expect(h1).not.toBe(INVALID_INDEX);
        expect(h2).not.toBe(INVALID_INDEX);
        expect(h1).not.toBe(h2);
        const n = b.addNode("n");
        b.setNodeValue(h1, n, "2009-03-01T12:00:00+01:00");
        b.setNodeValue(h2, n, "2009-03-02T12:00:00+01:00");
        const s = b.freeze();
        const c1 = s.nodes.require("start.text");
        const c2 = s.nodes.require("end.text");
        expect(c1.meta.role).toBe("timeText");
        expect(c2.meta.role).toBeNull();
        expect(c1.meta.extra).toEqual({ for: "start" });
        expect(c2.meta.extra).toEqual({ for: "end" });
        expect(s.nodes.byRole("timeText")?.meta.name).toBe("start.text");
        // a second declaration of the same companion returns the existing handle
        expect(declareCompanion(b, "node", start.companion as NonNullable<typeof start.companion>)).toBe(h1);
    });

    it("rethrows other declaration errors from declareCompanion", () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "t.text", dtype: "f64" });
        expect(() => declareCompanion(b, "node", { name: "t.text", dtype: "string", role: "timeText" })).toThrow(
            GraphFormatError,
        );
    });
});

describe("DictHeuristic (design 5.4)", () => {
    it("samples 1024 rows by default and decides dict below half distinct", () => {
        expect(DICT_SAMPLE_ROWS).toBe(1024);
        const h = new DictHeuristic();
        for (let i = 0; i < DICT_SAMPLE_ROWS - 1; i++) {
            expect(h.observe(`v${i % 10}`)).toBe(false);
        }
        expect(h.decided).toBe(false);
        expect(h.observe("v0")).toBe(true);
        expect(h.decided).toBe(true);
        expect(h.rows).toBe(DICT_SAMPLE_ROWS);
        expect(h.distinctCount).toBe(10);
        expect(h.decide()).toBe("dict");
    });

    it("decides string when distinct values reach half the rows, and string for no rows", () => {
        const h = new DictHeuristic(8);
        for (let i = 0; i < 8; i++) {
            h.observe(i < 4 ? `u${i}` : "same");
        }
        expect(h.distinctCount).toBe(5);
        expect(h.decide()).toBe("string");
        const exact = new DictHeuristic(8);
        for (let i = 0; i < 8; i++) {
            exact.observe(`u${i % 4}`);
        }
        expect(exact.decide()).toBe("string");
        const few = new DictHeuristic(8);
        for (let i = 0; i < 8; i++) {
            few.observe(`u${i % 3}`);
        }
        expect(few.decide()).toBe("dict");
        expect(new DictHeuristic(8).decide()).toBe("string");
    });

    it("keeps observing after the sample is full", () => {
        const h = new DictHeuristic(4);
        for (let i = 0; i < 4; i++) {
            h.observe("a");
        }
        expect(h.decide()).toBe("dict");
        h.observe("b");
        h.observe("c");
        h.observe("d");
        expect(h.rows).toBe(7);
        expect(h.decide()).toBe("string");
    });
});
