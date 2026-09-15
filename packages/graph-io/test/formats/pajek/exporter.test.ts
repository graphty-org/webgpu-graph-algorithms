import { type ColumnDecl, GraphBuilder, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { LOSS } from "../../../src/common/export.js";
import { decodeChunks } from "../../../src/common/writer.js";
import { PAJEK_LOSS, pajekExporter, type PajekExportOptions } from "../../../src/formats/pajek/exporter.js";
import { pajekImporter } from "../../../src/formats/pajek/importer.js";
import { type LossNote } from "../../../src/types.js";
import { readCorpusText } from "../../helpers/corpus.js";

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

async function fromPajek(text: string, options?: Parameters<typeof pajekImporter.import>[2]): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await pajekImporter.import(text, builder, options);
    return builder.freeze();
}

function lines(text: string): string[] {
    return text.split("\n").filter((l) => l.length > 0);
}

function builderWith(directed: boolean, nodeDecls: ColumnDecl[] = [], edgeDecls: ColumnDecl[] = []): GraphBuilder {
    const builder = new GraphBuilder({ directed, weightDtype: "f64" });
    for (const decl of nodeDecls) {
        builder.declareNodeColumn(decl);
    }
    for (const decl of edgeDecls) {
        builder.declareEdgeColumn(decl);
    }
    return builder;
}

describe("pajekExporter: capabilities", () => {
    it("declares what Pajek can hold", () => {
        expect(pajekExporter.format).toBe("pajek");
        expect(pajekExporter.capabilities).toMatchObject({
            mixedDirection: true,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "none",
            idCharset: "dense-1-based",
            components: false,
            lists: false,
            json: false,
            defaults: false,
            options: false,
            hierarchy: false,
            temporal: "spells",
            graphAttributes: false,
            positions: true,
            viz: false,
        });
        expect([...pajekExporter.capabilities.dtypes].sort()).toEqual(["bool", "f64", "i32", "string"]);
        expect(Object.isFrozen(pajekExporter.capabilities)).toBe(true);
    });
});

describe("pajekExporter: check()", () => {
    it("is exact for the corpus files with 1-based numbering", async () => {
        for (const name of ["simple.net", "karate.net", "karate-large.net"]) {
            const snapshot = await fromPajek(readCorpusText("pajek", name));
            expect(pajekExporter.check(snapshot), name).toEqual([]);
        }
    });

    it("reports renumbering for ids that are not their 1-based index", async () => {
        const zeroBased = await fromPajek(readCorpusText("pajek", "dolphins.net"));
        const notes = pajekExporter.check(zeroBased);
        expect(codes(notes)).toEqual([LOSS.ID_RENUMBERED]);
        expect(notes[0].count).toBe(62);
        const builder = builderWith(true);
        builder.addEdge("alice", "bob");
        builder.addEdge(3, "bob");
        const strings = pajekExporter.check(builder.freeze());
        expect(codes(strings)).toEqual([LOSS.ID_RENUMBERED]);
        expect(strings[0].count).toBe(2);
        expect(strings[0].message).toContain("labels");
    });

    it("counts text that Pajek cannot write and export() refuses it before writing", async () => {
        const builder = builderWith(true, [{ name: "label", dtype: "string", role: "label" }]);
        builder.addNodeRecord(1, { label: 'say "hi"' });
        builder.addNodeRecord(2, { label: "two\nlines" });
        builder.addNodeRecord(3, { label: "fine" });
        builder.addEdge(1, 2);
        builder.setEdgeValue("note", 0, "a\rb");
        const snapshot = builder.freeze();
        const notes = pajekExporter.check(snapshot);
        expect(codes(notes)).toEqual([PAJEK_LOSS.TEXT, PAJEK_LOSS.TEXT]);
        expect(notes[0]).toMatchObject({ column: "label", count: 2 });
        expect(notes[1]).toMatchObject({ column: "note", count: 1 });
        await expect(pajekExporter.exportToString(snapshot)).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
            details: { reason: "pajek label", column: "label" },
        });
        const chunks = pajekExporter.export(snapshot);
        await expect(decodeChunks(chunks)).rejects.toBeInstanceOf(GraphFormatError);
    });

    it("drops columns whose names cannot be parameter keys", () => {
        const builder = builderWith(false);
        builder.addNode(1);
        builder.addNode(2);
        builder.addEdge(1, 2);
        builder.setNodeValue("has space", 0, "x");
        builder.setNodeValue("box", 0, "y");
        builder.setNodeValue("42", 0, "z");
        builder.setNodeValue("[bracket", 0, "w");
        builder.setEdgeValue("has space", 0, 1);
        builder.setEdgeValue("ok_key", 0, 1);
        const notes = pajekExporter.check(builder.freeze());
        expect(codes(notes)).toEqual([
            PAJEK_LOSS.KEY_DROPPED,
            PAJEK_LOSS.KEY_DROPPED,
            PAJEK_LOSS.KEY_DROPPED,
            PAJEK_LOSS.KEY_DROPPED,
            PAJEK_LOSS.KEY_DROPPED,
        ]);
        expect(notes.map((n) => n.column)).toEqual(["has space", "box", "42", "[bracket", "has space"]);
    });

    it("reports a non-text label column, non-finite numbers and a position stride Pajek lacks", () => {
        const builder = builderWith(true, [
            { name: "label", dtype: "i32", role: "label" },
            { name: "position", dtype: "f32", components: 4, role: "position" },
        ]);
        builder.addNodeRecord(1, { label: 7, position: [1, 2, 3, 4] });
        builder.addNodeRecord(2, { label: 8 });
        builder.addEdge(1, 2);
        builder.setEdgeValue("cost", 0, Infinity);
        builder.setNodeValue("score", 1, NaN);
        const notes = pajekExporter.check(builder.freeze());
        expect(codes(notes)).toEqual([
            PAJEK_LOSS.LABEL_AS_TEXT,
            PAJEK_LOSS.POSITION_STRIDE,
            PAJEK_LOSS.NONFINITE_AS_TEXT,
            PAJEK_LOSS.NONFINITE_AS_TEXT,
        ]);
        expect(notes[1].count).toBe(1);
        expect(notes[2]).toMatchObject({ column: "score", count: 1 });
        expect(notes[3]).toMatchObject({ column: "cost", count: 1 });
    });

    it("writes an integral f64 parameter with a decimal point so the importer keeps f64", async () => {
        const builder = builderWith(true, [
            { name: "whole", dtype: "f64" },
            { name: "mixed", dtype: "f64" },
            { name: "pair", dtype: "f64", components: 2 },
        ]);
        builder.addNodeRecord(1, { whole: 2, mixed: 2, pair: [1, 2] });
        builder.addNodeRecord(2, { whole: -3, mixed: 3.5 });
        builder.addEdge(1, 2);
        const snapshot = builder.freeze();
        const notes = pajekExporter.check(snapshot);
        // the parameters force a label on each vertex line: the id text reads back as a label column
        expect(codes(notes)).toEqual([LOSS.COMPONENTS, PAJEK_LOSS.LABEL_GAINED]);
        const text = await pajekExporter.exportToString(snapshot);
        expect(lines(text).slice(1, 3)).toEqual(['1 1 whole 2.0 mixed 2.0 pair "1.0 2.0"', "2 2 whole -3.0 mixed 3.5"]);
        const back = await fromPajek(text);
        // the 5.1 grammar: "2.0" is f64 text, and the importer widens the column to f64
        expect(back.nodes.get("whole")?.dtype).toBe("f64");
        expect(back.nodes.get("mixed")?.dtype).toBe("f64");
        expect(back.nodes.get("pair")?.dtype).toBe("string");
    });

    it("passes the generic capability gaps through with the shared codes", () => {
        const builder = builderWith(true, [
            { name: "tags", dtype: "list", itemDtype: "string" },
            { name: "blob", dtype: "json" },
            { name: "color", dtype: "string", role: "color" },
            { name: "parent", dtype: "u32", role: "parent", refersTo: "node" },
            { name: "kind", dtype: "dict" },
            { name: "mass", dtype: "f32" },
            { name: "vec", dtype: "f64", components: 2 },
            { name: "opt", dtype: "string", options: ["a", "b"] },
            { name: "dflt", dtype: "i32", default: 3 },
        ]);
        builder.declareEdgeColumn({ name: "eid", dtype: "string", role: "id", unique: true });
        builder.addNodeRecord(1, { tags: ["x"], blob: { a: 1 }, color: "red", kind: "k", mass: 1.5, vec: [1, 2] });
        builder.addNodeRecord(2, { parent: 0 });
        builder.addEdgeRecord(1, 2, { eid: "e1" });
        builder.setGraphValue("title", "graph");
        builder.setMeta({ extra: { pajek: { firstMode: 99 } } });
        const notes = pajekExporter.check(builder.freeze());
        expect(codes(notes)).toEqual([
            LOSS.EDGE_IDS_DROPPED,
            LOSS.LIST,
            LOSS.JSON,
            LOSS.VIZ,
            LOSS.HIERARCHY,
            LOSS.DTYPE,
            LOSS.DTYPE,
            LOSS.COMPONENTS,
            LOSS.OPTIONS,
            LOSS.DEFAULT,
            LOSS.GRAPH_ATTRIBUTES,
            PAJEK_LOSS.LABEL_GAINED,
            PAJEK_LOSS.FIRST_MODE_DROPPED,
        ]);
        expect(notes.filter((n) => n.code === LOSS.DTYPE).map((n) => n.column)).toEqual(["kind", "mass"]);
    });

    it("keeps the shape keyword and relation name columns although they are dict", async () => {
        const text = '*Vertices 2\n1 a box\n2 b\n*Arcs :1 "likes"\n1 2\n';
        const snapshot = await fromPajek(text);
        expect(snapshot.nodes.requireTyped("shape", "dict")).toBeDefined();
        expect(snapshot.edges.requireTyped("relation", "dict")).toBeDefined();
        expect(pajekExporter.check(snapshot)).toEqual([]);
        // a dict column named "shape" on the edge table is a parameter and re-imports as string
        const builder = builderWith(true, [], [{ name: "shape", dtype: "dict" }]);
        builder.addEdgeRecord(1, 2, { shape: "x" });
        expect(codes(pajekExporter.check(builder.freeze()))).toEqual([LOSS.DTYPE]);
    });

    it("reports temporal roles other than spells, mutual edges and odd spells shapes", () => {
        const builder = builderWith(true, [
            { name: "start", dtype: "f64", role: "start" },
            { name: "spells", dtype: "list", itemDtype: "string", role: "spells" },
        ]);
        builder.declareEdgeColumn({ name: "end", dtype: "f64", role: "end" });
        builder.declareEdgeColumn({ name: "graphty.directed", dtype: "bool", role: "directed" });
        builder.declareEdgeColumn({ name: "graphty.pair", dtype: "u32", role: "pair", refersTo: "edge" });
        builder.declareEdgeColumn({ name: "graphty.mutual", dtype: "bool", role: "mutual" });
        builder.addNodeRecord(1, { start: 1, spells: ["a"] });
        builder.addNode(2);
        const e0 = builder.addEdge(1, 2);
        const e1 = builder.addEdge(2, 1);
        builder.setEdgeValue("graphty.directed", e0, true);
        builder.setEdgeValue("graphty.directed", e1, true);
        builder.setEdgeValue("graphty.pair", e0, e1);
        builder.setEdgeValue("graphty.pair", e1, e0);
        builder.setEdgeValue("graphty.mutual", e0, true);
        builder.setEdgeValue("end", e0, 5);
        const notes = pajekExporter.check(builder.freeze());
        expect(codes(notes)).toEqual([
            PAJEK_LOSS.TEMPORAL_DROPPED,
            LOSS.SPELLS,
            PAJEK_LOSS.TEMPORAL_DROPPED,
            PAJEK_LOSS.MUTUAL_AS_UNDIRECTED,
        ]);
        expect(notes[3].count).toBe(1);
    });

    it("reports extension tables, open intervals and an edge position column", async () => {
        const builder = builderWith(
            true,
            [{ name: "graphty.open", dtype: "u8", role: "open" }],
            [{ name: "pos", dtype: "f32", components: 3, role: "position" }],
        );
        builder.addNodeRecord(1, { "graphty.open": 1 });
        builder.addEdgeRecord(1, 2, { pos: [1, 2, 3] });
        const temporal = builder.addExtensionTable("temporal:node:price", [{ name: "value", dtype: "f64" }]);
        builder.addExtensionRow(temporal, [1]);
        const other = builder.addExtensionTable("audit", [{ name: "who", dtype: "string" }]);
        builder.addExtensionRow(other, ["me"]);
        const snapshot = builder.freeze();
        const notes = pajekExporter.check(snapshot);
        expect(codes(notes)).toEqual([LOSS.POSITIONS, LOSS.OPEN_INTERVAL, LOSS.DYNAMIC_VALUES, LOSS.EXTENSION_TABLE]);
        expect(lines(await pajekExporter.exportToString(snapshot))).toEqual(["*Vertices 2", "1", "2", "*Arcs", "1 2"]);
    });

    it("rejects options outside their sets", () => {
        const builder = builderWith(true);
        builder.addEdge(1, 2);
        const snapshot = builder.freeze();
        expect(() => pajekExporter.check(snapshot, { sanitizeIds: "x" as "error" })).toThrow(GraphFormatError);
        expect(() => pajekExporter.check(snapshot, { networkHeader: "yes" as unknown as boolean })).toThrow(
            GraphFormatError,
        );
        expect(() => pajekExporter.export(snapshot, { networkHeader: 1 as unknown as boolean })).toThrow(
            GraphFormatError,
        );
    });
});

describe("pajekExporter: writing", () => {
    it("numbers vertices 1..N, quotes labels only when needed and writes an empty label quoted", async () => {
        const builder = builderWith(false, [{ name: "label", dtype: "string", role: "label" }]);
        builder.addNodeRecord(1, { label: "plain" });
        builder.addNodeRecord(2, { label: "two words" });
        builder.addNodeRecord(3, { label: "" });
        builder.addNode(4);
        builder.addEdge(1, 4);
        const text = await pajekExporter.exportToString(builder.freeze());
        expect(lines(text)).toEqual(["*Vertices 4", "1 plain", '2 "two words"', '3 ""', "4", "*Edges", "1 4"]);
    });

    it("writes ids that are not the vertex number as labels and forces a label before coordinates", async () => {
        const builder = builderWith(true, [{ name: "position", dtype: "f32", components: 3, role: "position" }]);
        builder.addNodeRecord("alice", { position: [0.5, 0.25, 0] });
        builder.addNodeRecord(2, { position: [1, 1, 0] });
        builder.addNode(3);
        builder.addEdge("alice", 2, 2);
        const snapshot = builder.freeze();
        const text = await pajekExporter.exportToString(snapshot);
        expect(lines(text)).toEqual(["*Vertices 3", "1 alice 0.5 0.25 0", "2 2 1 1 0", "3", "*Arcs", "1 2 2"]);
    });

    it("writes z when the importer saw three coordinates or any z is non-zero", async () => {
        const three = await fromPajek("*Vertices 1\n1 a 0.1 0.2 0\n*Arcs\n");
        expect(lines(await pajekExporter.exportToString(three))[1]).toBe("1 a 0.1 0.2 0");
        const two = await fromPajek("*Vertices 1\n1 a 0.1 0.2\n*Arcs\n");
        expect(lines(await pajekExporter.exportToString(two))[1]).toBe("1 a 0.1 0.2");
        const builder = builderWith(true, [
            { name: "position", dtype: "f32", components: 3, role: "position", extra: { sourceDims: 2 } },
        ]);
        builder.addNodeRecord(1, { position: [1, 2, 3] });
        builder.addNodeRecord(2, { position: [4, 5, 0] });
        const text = await pajekExporter.exportToString(builder.freeze());
        expect(lines(text).slice(1, 3)).toEqual(["1 1 1 2 3", "2 2 4 5 0"]);
        const twoComponents = builderWith(true, [{ name: "position", dtype: "f64", components: 2, role: "position" }]);
        twoComponents.addNodeRecord(1, { position: [1.5, 2] });
        expect(lines(await pajekExporter.exportToString(twoComponents.freeze()))[1]).toBe("1 1 1.5 2.0");
    });

    it("writes the shape keywords in their slot, or the whole column as a parameter when a value is not a keyword", async () => {
        const keywords = builderWith(true, [{ name: "shape", dtype: "dict" }]);
        keywords.addNodeRecord(1, { shape: "box" });
        keywords.addNodeRecord(2, { shape: "ellipse" });
        keywords.addNode(3);
        expect(codes(pajekExporter.check(keywords.freeze()))).toEqual([PAJEK_LOSS.LABEL_GAINED]);
        expect(lines(await pajekExporter.exportToString(keywords.freeze())).slice(1, 4)).toEqual([
            "1 1 box",
            "2 2 ellipse",
            "3",
        ]);
        const mixed = builderWith(true, [{ name: "shape", dtype: "dict" }]);
        mixed.addNodeRecord(1, { shape: "box" });
        mixed.addNodeRecord(2, { shape: "star" });
        mixed.addNode(3);
        const snapshot = mixed.freeze();
        const notes = pajekExporter.check(snapshot);
        expect(notes.map((n) => [n.code, n.column])).toEqual([
            [PAJEK_LOSS.SHAPE_AS_PARAMETER, "shape"],
            [PAJEK_LOSS.LABEL_GAINED, "label"],
        ]);
        const text = await pajekExporter.exportToString(snapshot);
        expect(lines(text).slice(1, 4)).toEqual(["1 1 shape box", "2 2 shape star", "3"]);
        // its own importer reads the parameter back as one string column
        const back = await fromPajek(text);
        expect(back.nodes.get("shape")?.dtype).toBe("string");
        expect([0, 1].map((i) => back.nodes.value("shape", i))).toEqual(["box", "star"]);
    });

    it("writes parameters by dtype so the importer keeps the dtype", async () => {
        const builder = builderWith(true, [
            { name: "i", dtype: "i32" },
            { name: "f", dtype: "f64" },
            { name: "b", dtype: "bool" },
            { name: "s", dtype: "string" },
            { name: "u", dtype: "u8" },
            { name: "d", dtype: "dict" },
            { name: "v", dtype: "i32", components: 2 },
            { name: "x", dtype: "f32" },
        ]);
        builder.addNodeRecord(1, { i: -4, f: 2, b: false, s: "a b", u: 200, d: "cat", v: [1, 2], x: 0.1 });
        builder.addNodeRecord(2, { f: 1e21, s: "", b: true });
        const text = await pajekExporter.exportToString(builder.freeze());
        expect(lines(text).slice(1, 3)).toEqual([
            '1 1 i -4 f 2.0 b false s "a b" u 200 d cat v "1 2" x 0.1',
            '2 2 f 1e+21 b true s ""',
        ]);
    });

    it("writes intervals from the spells role on vertices and lines", async () => {
        const source = "*Vertices 2\n1 a [1-5,7-*]\n2 b [3]\n*Edges\n1 2 [*-2]\n";
        const text = await pajekExporter.exportToString(await fromPajek(source));
        expect(lines(text)).toEqual(["*Vertices 2", "1 a [1-5,7-*]", "2 b [3]", "*Edges", "1 2 [*-2]"]);
    });

    it("writes the two-mode header from the metadata and the *Network line on request", async () => {
        const snapshot = await fromPajek("*Network friends\n*Vertices 3 1\n1 a\n2 b\n3 c\n*Edges\n1 2\n");
        expect(lines(await pajekExporter.exportToString(snapshot))[0]).toBe("*Vertices 3 1");
        const options: PajekExportOptions = { networkHeader: true };
        const withHeader = await pajekExporter.exportToString(snapshot, options);
        expect(lines(withHeader).slice(0, 2)).toEqual(["*Network friends", "*Vertices 3 1"]);
        const builder = builderWith(true);
        builder.addEdge(1, 2);
        builder.setMeta({ extra: { pajek: { firstMode: 5 } } });
        expect(lines(await pajekExporter.exportToString(builder.freeze()))[0]).toBe("*Vertices 2");
    });

    it("writes sections in runs of kind and relation, preserving the edge order", async () => {
        const source = [
            "*Vertices 3",
            "*Edges",
            "1 2",
            "*Arcs :1 likes",
            "2 3",
            "3 1",
            "*Edges",
            "1 3",
            '*Arcs :2 "hates a lot"',
            "1 1",
        ].join("\n");
        const snapshot = await fromPajek(source);
        const text = await pajekExporter.exportToString(snapshot);
        expect(lines(text)).toEqual([
            "*Vertices 3",
            "1",
            "2",
            "3",
            "*Edges",
            "1 2",
            "*Arcs :1 likes",
            "2 3",
            "3 1",
            "*Edges",
            "1 3",
            '*Arcs :2 "hates a lot"',
            "1 1",
        ]);
    });

    it("writes only explicit weights and no weight for an unweighted graph", async () => {
        const mixed = await fromPajek("*Vertices 2\n*Arcs\n1 2\n2 1 3.5\n1 1 0.1\n");
        expect(lines(await pajekExporter.exportToString(mixed)).slice(3)).toEqual([
            "*Arcs",
            "1 2",
            "2 1 3.5",
            "1 1 0.1",
        ]);
        const builder = builderWith(true);
        builder.addEdge(1, 2);
        expect(lines(await pajekExporter.exportToString(builder.freeze())).slice(3)).toEqual(["*Arcs", "1 2"]);
        const f32 = new GraphBuilder({ directed: false, weightDtype: "f32" });
        f32.addEdge(1, 2, 0.1);
        f32.addEdge(2, 1, 16777217);
        expect(lines(await pajekExporter.exportToString(f32.freeze())).slice(3)).toEqual([
            "*Edges",
            "1 2 0.1",
            "2 1 16777216",
        ]);
    });

    it("folds expanded pairs and writes a mutual pair as one undirected edge", async () => {
        const builder = builderWith(true);
        builder.declareEdgeColumn({ name: "graphty.directed", dtype: "bool", role: "directed" });
        builder.declareEdgeColumn({ name: "graphty.pair", dtype: "u32", role: "pair", refersTo: "edge" });
        builder.declareEdgeColumn({ name: "graphty.mutual", dtype: "bool", role: "mutual" });
        const e0 = builder.addEdge(1, 2);
        const e1 = builder.addEdge(2, 1);
        builder.setEdgeValue("graphty.directed", e0, true);
        builder.setEdgeValue("graphty.directed", e1, true);
        builder.setEdgeValue("graphty.pair", e0, e1);
        builder.setEdgeValue("graphty.pair", e1, e0);
        builder.setEdgeValue("graphty.mutual", e0, true);
        const e2 = builder.addEdge(2, 3);
        builder.setEdgeValue("graphty.directed", e2, true);
        const text = await pajekExporter.exportToString(builder.freeze());
        expect(lines(text).slice(4)).toEqual(["*Edges", "1 2", "*Arcs", "2 3"]);
    });

    it("writes an empty section header for a graph without edges", async () => {
        const directed = builderWith(true);
        directed.addNode(1);
        expect(lines(await pajekExporter.exportToString(directed.freeze()))).toEqual(["*Vertices 1", "1", "*Arcs"]);
        const undirected = builderWith(false);
        expect(lines(await pajekExporter.exportToString(undirected.freeze()))).toEqual(["*Vertices 0", "*Edges"]);
    });

    it("export() streams the same bytes exportToString() returns", async () => {
        const snapshot = await fromPajek(readCorpusText("pajek", "football.net"));
        const text = await pajekExporter.exportToString(snapshot);
        expect(await decodeChunks(pajekExporter.export(snapshot))).toBe(text);
        expect(text.endsWith("\n")).toBe(true);
    });

    it("skips viz, hierarchy, list and json columns and edge ids in the output", async () => {
        const builder = builderWith(true, [
            { name: "color", dtype: "string", role: "color" },
            { name: "tags", dtype: "list", itemDtype: "string" },
        ]);
        builder.declareEdgeColumn({ name: "eid", dtype: "string", role: "id", unique: true });
        builder.addNodeRecord(1, { color: "red", tags: ["x"] });
        builder.addEdgeRecord(1, 2, { eid: "e1" });
        const text = await pajekExporter.exportToString(builder.freeze());
        expect(lines(text)).toEqual(["*Vertices 2", "1", "2", "*Arcs", "1 2"]);
    });
});
