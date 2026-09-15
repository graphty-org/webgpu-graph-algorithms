import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { decodeChunks } from "../../../src/common/writer.js";
import {
    JSON_DIALECTS,
    JSON_LOSS,
    jsonCapabilities,
    jsonExporter,
    type JsonExportOptions,
    jsonImporter,
} from "../../../src/formats/json/index.js";
import { type CommonExportOptions, type LossNote } from "../../../src/types.js";
import { readCorpusText } from "../../helpers/corpus.js";

type Options = JsonExportOptions & CommonExportOptions;

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

function note(notes: readonly LossNote[], code: string): LossNote | undefined {
    return notes.find((n) => n.code === code);
}

async function imported(text: string, options?: Record<string, unknown>): Promise<GraphSnapshot> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await jsonImporter.import(text, b, options);
    return b.freeze();
}

async function exported(snapshot: GraphSnapshot, options?: Options): Promise<Record<string, unknown>> {
    return JSON.parse(await jsonExporter.exportToString(snapshot, options)) as Record<string, unknown>;
}

/** A directed graph with two undirected pairs (one a self-loop) and one mutual pair. */
function mixed(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
    r.setHeader(true);
    r.addEdge("a", "b", "directed", 2);
    r.addEdge("b", "c", "undirected");
    r.addEdge("c", "c", "undirected", 0.5);
    r.addEdge("a", "c", "mutual");
    b.setEdgeValue("kind", 1, "pair");
    return b.freeze();
}

describe("jsonExporter plugin", () => {
    it("declares the node-link capabilities and one table per dialect", () => {
        expect(jsonExporter.format).toBe("json");
        expect(jsonExporter.capabilities).toBe(jsonCapabilities("node-link"));
        expect(jsonExporter.capabilities.mixedDirection).toBe(false);
        expect(jsonExporter.capabilities.json).toBe(true);
        // JSON declares no types: only the inferred dtypes survive, arrays come back as json
        expect(jsonExporter.capabilities.lists).toBe(false);
        expect(jsonExporter.capabilities.components).toBe(false);
        expect(jsonExporter.capabilities.dtypes).toEqual(["f64", "i32", "bool", "string"]);
        expect(jsonExporter.capabilities.positions).toBe(false);
        expect(jsonCapabilities("cytoscape").positions).toBe(true);
        expect(jsonExporter.capabilities.edgeIds).toBe("none");
        expect(jsonCapabilities("d3").graphAttributes).toBe(false);
        expect(jsonExporter.capabilities.temporal).toBe("none");
        expect(jsonCapabilities("jgf").mixedDirection).toBe(true);
        expect(jsonCapabilities("graphology").mixedDirection).toBe(true);
        expect(jsonCapabilities("cytoscape").hierarchy).toBe(true);
        expect(jsonCapabilities("cytoscape").edgeIds).toBe("required");
        expect(jsonCapabilities("vis").graphAttributes).toBe(false);
        for (const dialect of JSON_DIALECTS) {
            expect(Object.isFrozen(jsonCapabilities(dialect))).toBe(true);
            expect(jsonCapabilities(dialect).idCharset).toBe("any");
        }
    });

    it("rejects bad options with E_UNSUPPORTED", async () => {
        const s = await imported('{"nodes":[{"id":1}],"links":[]}');
        const bad: Options[] = [
            { dialect: "yaml" as never },
            { indent: -1 },
            { indent: 2.5 },
            { indexLinks: "no" as never },
            { edgesKey: "" },
            { weightKey: 4 as never },
            { sanitizeIds: "rename" as never },
        ];
        for (const options of bad) {
            expect(() => jsonExporter.check(s, options)).toThrow(expect.objectContaining({ code: "E_UNSUPPORTED" }));
        }
    });

    it("export() chunks decode to exportToString()", async () => {
        const s = await imported(readCorpusText("json", "miserables.json"));
        const text = await jsonExporter.exportToString(s);
        expect(await decodeChunks(jsonExporter.export(s))).toBe(text);
        expect(JSON.parse(text)).toBeTruthy();
    });

    it("writes compact JSON by default and indented JSON on request", async () => {
        const s = await imported(
            '{"directed":false,"nodes":[{"id":"a","n":{"k":1}}],"links":[{"source":"a","target":"a"}]}',
        );
        const compact = await jsonExporter.exportToString(s);
        expect(compact).not.toContain("\n");
        expect(compact).toBe(
            '{"directed":false,"multigraph":false,"graph":{},"nodes":[{"id":"a","n":{"k":1}}],"links":[{"source":"a","target":"a"}]}',
        );
        const pretty = await jsonExporter.exportToString(s, { indent: 2 });
        expect(pretty.split("\n")[1]).toBe('  "directed": false,');
        expect(pretty).toContain('\n  "nodes": [\n    {\n      "id": "a",');
        expect(JSON.parse(pretty)).toEqual(JSON.parse(compact));
        const empty = new GraphBuilder({ directed: true }).freeze();
        expect(await jsonExporter.exportToString(empty, { indent: 4 })).toBe(
            '{\n    "directed": true,\n    "multigraph": false,\n    "graph": {},\n    "nodes": [],\n    "edges": []\n}',
        );
    });
});

describe("dialect selection", () => {
    it("defaults to the recorded dialect and to node-link otherwise", async () => {
        const cyto = await imported(readCorpusText("json", "cytoscape-format.json"));
        expect(Object.keys(await exported(cyto))).toEqual(["elements"]);
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        const plain = await exported(b.freeze());
        expect(Object.keys(plain)).toEqual(["directed", "multigraph", "graph", "nodes", "edges"]);
        const forced = await exported(cyto, { dialect: "vis" });
        expect(Object.keys(forced)).toEqual(["nodes", "edges"]);
    });

    it("uses the recorded node-link keys only for the same dialect", async () => {
        const d3 = await imported(readCorpusText("json", "miserables.json"));
        const same = await exported(d3);
        expect(Object.keys(same)).toContain("links");
        expect((same.nodes as Record<string, unknown>[])[0]).toEqual({ name: "Myriel", group: 1 });
        expect((same.links as Record<string, unknown>[])[0]).toEqual({ source: 1, target: 0, value: 1 });
        const asNodeLink = await exported(d3, { dialect: "node-link" });
        expect(Object.keys(asNodeLink)).toContain("edges");
        expect((asNodeLink.nodes as Record<string, unknown>[])[0]).toEqual({ id: "Myriel", group: 1 });
        expect((asNodeLink.edges as Record<string, unknown>[])[0]).toEqual({
            source: "Napoleon",
            target: "Myriel",
            value: 1,
        });
        const custom = await exported(d3, {
            dialect: "node-link",
            edgesKey: "arcs",
            nodeIdKey: "key",
            sourceKey: "u",
            targetKey: "v",
            indexLinks: true,
        });
        expect((custom.nodes as Record<string, unknown>[])[0]).toEqual({ key: "Myriel", group: 1 });
        expect((custom.arcs as Record<string, unknown>[])[0]).toEqual({ u: 1, v: 0, value: 1 });
    });
});

describe("values", () => {
    it("writes every dtype as JSON, omits unset rows and formats f32 shortest", async () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.declareNodeColumn({ name: "f", dtype: "f32" });
        b.declareNodeColumn({ name: "d", dtype: "dict" });
        b.declareNodeColumn({ name: "u", dtype: "u8", components: 2 });
        b.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "f32" });
        const a = b.addNode("a");
        const c = b.addNode("c");
        b.setNodeValue("f", a, 0.1);
        b.setNodeValue("d", a, "cat");
        b.setNodeValue("u", a, [1, 2]);
        b.setNodeValue("l", a, [0.1, 2]);
        b.setNodeValue("b", a, true);
        b.setNodeValue("s", a, 'text "quoted"');
        b.setNodeValue("j", a, { deep: [1, null] });
        b.setNodeValue("j", c, null);
        b.setNodeValue("i", c, -7);
        b.setNodeValue("n", c, 1e21);
        b.addEdge("a", "c", 0.1);
        b.addEdge("c", "c");
        b.setGraphValue("title", "G");
        b.setGraphValue("big", 2 ** 53);
        b.setGraphValue("tags", ["x", "y"]);
        const doc = await exported(b.freeze());
        expect(doc.nodes).toEqual([
            { id: "a", f: 0.1, d: "cat", u: [1, 2], l: [0.1, 2], b: true, s: 'text "quoted"', j: { deep: [1, null] } },
            { id: "c", i: -7, n: 1e21 },
        ]);
        expect(doc.edges).toEqual([
            { source: "a", target: "c", weight: 0.1 },
            { source: "c", target: "c" },
        ]);
        expect(doc.graph).toEqual({ title: "G", big: 2 ** 53, tags: ["x", "y"] });
        const text = await jsonExporter.exportToString(b.freeze());
        expect(text).toContain('"f":0.1');
        expect(text).not.toContain("0.10000000149011612");
    });

    it("writes weights from the f32 arc array when every weight is explicit", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f32" });
        b.addEdge("a", "b", 2.5);
        b.addEdge("b", "a", 0.1);
        const doc = await exported(b.freeze());
        expect(doc.edges).toEqual([
            { source: "a", target: "b", weight: 2.5 },
            { source: "b", target: "a", weight: 0.1 },
        ]);
    });

    it("writes the weight under the key it was read from, or weightKey", async () => {
        const s = await imported(readCorpusText("json", "d3-format.json"), { weightFrom: "value" });
        const same = await exported(s);
        expect((same.links as Record<string, unknown>[])[1]).toEqual({ source: "b", target: "c", value: 2 });
        const renamed = await exported(s, { weightKey: "w" });
        expect((renamed.links as Record<string, unknown>[])[1]).toEqual({ source: "b", target: "c", w: 2 });
    });

    it("writes refersTo node columns as ids and drops the key for INVALID_INDEX", async () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "ref", dtype: "u32", refersTo: "node" });
        const a = b.addNode("a");
        const c = b.addNode(7);
        b.setNodeValue("ref", a, c);
        const doc = await exported(b.freeze(), { dialect: "vis" });
        expect(doc.nodes).toEqual([{ id: "a", ref: 7 }, { id: 7 }]);
    });

    it("writes non-finite numbers as null and reports them once", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "f", dtype: "f64" });
        b.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "f64" });
        const a = b.addNode("a");
        b.setNodeValue("f", a, Infinity);
        b.setNodeValue("l", a, [1, NaN]);
        b.addEdge("a", "a", -Infinity);
        b.addEdge("a", "a", 1);
        const s = b.freeze();
        const notes = jsonExporter.check(s);
        expect(codes(notes)).toEqual([LOSS.LIST, JSON_LOSS.NONFINITE_AS_NULL]);
        expect(note(notes, JSON_LOSS.NONFINITE_AS_NULL)?.count).toBe(3);
        const doc = await exported(s);
        expect(doc.nodes).toEqual([{ id: "a", f: null, l: [1, null] }]);
        expect(doc.edges).toEqual([
            { source: "a", target: "a", weight: null },
            { source: "a", target: "a", weight: 1 },
        ]);
    });
});

describe("direction", () => {
    it("node-link refuses mixed direction by default and writes it per onMixedDirection", async () => {
        const s = mixed();
        const refused = jsonExporter.check(s);
        expect(codes(refused)).toEqual([LOSS.MIXED_DIRECTION_ERROR, JSON_LOSS.MUTUAL_EXPANDED]);
        await expect(jsonExporter.exportToString(s)).rejects.toMatchObject({ code: "E_DIRECTED" });
        let failed = false;
        try {
            for await (const chunk of jsonExporter.export(s)) {
                expect(chunk).toBeInstanceOf(Uint8Array);
            }
        } catch (err) {
            failed = true;
            expect(err).toMatchObject({ code: "E_DIRECTED" });
        }
        expect(failed).toBe(true);

        const directed = await exported(s, { onMixedDirection: "directed" });
        expect(codes(jsonExporter.check(s, { onMixedDirection: "directed" }))).toEqual([
            LOSS.MIXED_DIRECTION,
            JSON_LOSS.MUTUAL_EXPANDED,
        ]);
        expect(directed.directed).toBe(true);
        // "directed" folds the pair back to its primary (design section 8.5); the mirror is not written
        expect((directed.edges as unknown[]).length).toBe(5);
        expect(directed.edges).toEqual([
            { source: "a", target: "b", weight: 2 },
            { source: "b", target: "c", kind: "pair" },
            { source: "c", target: "c", weight: 0.5 },
            { source: "a", target: "c" },
            { source: "c", target: "a" },
        ]);

        const undirected = await exported(s, { onMixedDirection: "undirected" });
        expect(undirected.directed).toBe(false);
        expect(undirected.edges).toEqual([
            { source: "a", target: "b", weight: 2 },
            { source: "b", target: "c", kind: "pair" },
            { source: "c", target: "c", weight: 0.5 },
            { source: "a", target: "c" },
            { source: "c", target: "a" },
        ]);
    });

    it("jgf writes per-edge directed: false for pairs and both halves of a mutual pair", async () => {
        const s = mixed();
        const notes = jsonExporter.check(s, { dialect: "jgf" });
        expect(codes(notes)).toEqual([JSON_LOSS.MUTUAL_EXPANDED]);
        expect(note(notes, JSON_LOSS.MUTUAL_EXPANDED)?.count).toBe(1);
        const doc = (await exported(s, { dialect: "jgf" })).graph as Record<string, unknown>;
        expect(doc.directed).toBe(true);
        expect(doc.edges).toEqual([
            { source: "a", target: "b", metadata: { weight: 2 } },
            { source: "b", target: "c", directed: false, metadata: { kind: "pair" } },
            { source: "c", target: "c", directed: false, metadata: { weight: 0.5 } },
            { source: "a", target: "c" },
            { source: "c", target: "a" },
        ]);
    });

    it("graphology writes type mixed with undirected: true on pairs", async () => {
        const doc = await exported(mixed(), { dialect: "graphology" });
        // a snapshot from elsewhere declares neither multi nor allowSelfLoops: only the type is written
        expect(doc.options).toEqual({ type: "mixed" });
        expect(doc.edges).toEqual([
            { source: "a", target: "b", attributes: { weight: 2 } },
            { source: "b", target: "c", undirected: true, attributes: { kind: "pair" } },
            { source: "c", target: "c", undirected: true, attributes: { weight: 0.5 } },
            { source: "a", target: "c" },
            { source: "c", target: "a" },
        ]);
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        const undirected = await exported(b.freeze(), { dialect: "graphology" });
        expect((undirected.options as Record<string, unknown>).type).toBe("undirected");
        const d = new GraphBuilder({ directed: true });
        d.addEdge("a", "b");
        const directed = await exported(d.freeze(), { dialect: "graphology" });
        expect((directed.options as Record<string, unknown>).type).toBe("directed");
    });

    it("cytoscape and vis report the direction they cannot carry", async () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        const undirected = b.freeze();
        expect(codes(jsonExporter.check(undirected, { dialect: "cytoscape" }))).toEqual([
            LOSS.EDGE_IDS_GENERATED,
            JSON_LOSS.DIRECTION_DROPPED,
        ]);
        expect(codes(jsonExporter.check(undirected, { dialect: "vis" }))).toEqual([]);
        const d = new GraphBuilder({ directed: true });
        d.addEdge("a", "b");
        const directed = d.freeze();
        expect(codes(jsonExporter.check(directed, { dialect: "vis" }))).toEqual([JSON_LOSS.DIRECTION_DROPPED]);
        expect(codes(jsonExporter.check(directed, { dialect: "cytoscape" }))).toEqual([LOSS.EDGE_IDS_GENERATED]);
        const doc = await exported(mixed(), { dialect: "cytoscape", onMixedDirection: "undirected" });
        const { edges } = doc.elements as { edges: { data: Record<string, unknown> }[] };
        expect(edges.map((e) => `${String(e.data.source)}-${String(e.data.target)}`)).toEqual([
            "a-b",
            "b-c",
            "c-c",
            "a-c",
            "c-a",
        ]);
    });

    it("writes multigraph true when parallel edges exist or the file declared it", async () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        expect((await exported(b.freeze())).multigraph).toBe(true);
        const declared = await imported('{"multigraph":true,"nodes":[{"id":1}],"links":[]}');
        expect((await exported(declared)).multigraph).toBe(true);
        expect(((await exported(declared, { dialect: "graphology" })).options as Record<string, unknown>).multi).toBe(
            true,
        );
    });
});

describe("check() notes", () => {
    it("reports numeric ids, id collisions and node order for JGF, and export() throws on a collision", async () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode(2);
        b.addNode("1");
        b.addNode("x");
        b.addNode(1);
        const s = b.freeze();
        const notes = jsonExporter.check(s, { dialect: "jgf" });
        expect(codes(notes)).toEqual([
            JSON_LOSS.NUMERIC_IDS_STRINGIFIED,
            JSON_LOSS.ID_TEXT_COLLISION,
            JSON_LOSS.NODE_ORDER,
        ]);
        expect(note(notes, JSON_LOSS.NUMERIC_IDS_STRINGIFIED)?.count).toBe(2);
        expect(note(notes, JSON_LOSS.ID_TEXT_COLLISION)?.count).toBe(1);
        expect(note(notes, JSON_LOSS.NODE_ORDER)?.count).toBe(3);
        await expect(jsonExporter.exportToString(s, { dialect: "jgf" })).rejects.toMatchObject({
            code: "E_INVALID_ID",
        });
        const ordered = new GraphBuilder({ directed: true });
        ordered.addNode(1);
        ordered.addNode(2);
        ordered.addNode("b");
        ordered.addNode("a");
        expect(codes(jsonExporter.check(ordered.freeze(), { dialect: "jgf" }))).toEqual([
            JSON_LOSS.NUMERIC_IDS_STRINGIFIED,
        ]);
        const strings = new GraphBuilder({ directed: true });
        strings.addNode("b");
        strings.addNode("a");
        expect(codes(jsonExporter.check(strings.freeze(), { dialect: "jgf" }))).toEqual([]);
    });

    it("reports reserved key collisions and skips the column", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "id", dtype: "string" });
        const a = b.addNode("a");
        b.setNodeValue("id", a, "shadow");
        const e = b.addEdge("a", "a", 2);
        b.setEdgeValue("weight", e, "w");
        b.setEdgeValue("target", e, "t");
        const s = b.freeze();
        const notes = jsonExporter.check(s);
        expect(codes(notes)).toEqual([JSON_LOSS.RESERVED_KEY, JSON_LOSS.RESERVED_KEY, JSON_LOSS.RESERVED_KEY]);
        expect(notes.map((n) => n.column)).toEqual(["id", "weight", "target"]);
        const doc = await exported(s);
        expect(doc.nodes).toEqual([{ id: "a" }]);
        expect(doc.edges).toEqual([{ source: "a", target: "a", weight: 2 }]);
        const renamed = await exported(s, { nodeIdKey: "key", targetKey: "to", weightKey: "w" });
        expect(renamed.nodes).toEqual([{ key: "a", id: "shadow" }]);
        expect(renamed.edges).toEqual([{ source: "a", to: "a", w: 2, weight: "w", target: "t" }]);
        expect(codes(jsonExporter.check(s, { nodeIdKey: "key", targetKey: "to", weightKey: "w" }))).toEqual([]);
        const nested = await exported(s, { dialect: "graphology" });
        expect(nested.nodes).toEqual([{ key: "a", attributes: { id: "shadow" } }]);
        expect(nested.edges).toEqual([{ source: "a", target: "a", attributes: { weight: 2, target: "t" } }]);
    });

    it("reports a second column that maps to a key another column already uses", async () => {
        const b = new GraphBuilder({ directed: true });
        const a = b.addNode("a");
        b.setNodeValue("position", a, 1);
        b.setNodeValue("position#data", a, 2);
        const s = b.freeze();
        const notes = jsonExporter.check(s, { dialect: "cytoscape" });
        expect(codes(notes)).toEqual([JSON_LOSS.RESERVED_KEY]);
        expect(note(notes, JSON_LOSS.RESERVED_KEY)?.column).toBe("position#data");
        const doc = (await exported(s, { dialect: "cytoscape" })).elements as { nodes: Record<string, unknown>[] };
        expect(doc.nodes).toEqual([{ data: { id: "a", position: 1 } }]);
        expect(codes(jsonExporter.check(s))).toEqual([]);
        expect((await exported(s)).nodes).toEqual([{ id: "a", position: 1, "position#data": 2 }]);
    });

    it("reports the weight key collision only for a weighted graph", async () => {
        const b = new GraphBuilder({ directed: true });
        const e = b.addEdge("a", "a");
        b.setEdgeValue("weight", e, 5);
        const s = b.freeze();
        expect(s.flags.weighted).toBe(false);
        // the column is written; the importer's weightFrom default reads it back as THE weight
        const notes = jsonExporter.check(s);
        expect(codes(notes)).toEqual([JSON_LOSS.WEIGHT_KEY_CLASH]);
        expect(note(notes, JSON_LOSS.WEIGHT_KEY_CLASH)?.column).toBe("weight");
        expect((await exported(s)).edges).toEqual([{ source: "a", target: "a", weight: 5 }]);
        expect(codes(jsonExporter.check(s, { weightKey: "w" }))).toEqual([]);
    });

    it("reports an edge id column written as an attribute in node-link", async () => {
        const s = await imported(readCorpusText("json", "cytoscape-format.json"));
        const notes = jsonExporter.check(s, { dialect: "node-link" });
        expect(codes(notes)).toEqual([JSON_LOSS.EDGE_IDS_DROPPED]);
        expect(note(notes, JSON_LOSS.EDGE_IDS_DROPPED)?.count).toBe(5);
        const doc = await exported(s, { dialect: "node-link" });
        expect((doc.edges as Record<string, unknown>[])[0]).toEqual({
            id: "e1",
            source: "n1",
            target: "n2",
            weight: 1,
        });
    });

    it("writes positions as a Cytoscape position object, as an array elsewhere, and reports z", async () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({
            name: "position",
            dtype: "f32",
            components: 3,
            role: "position",
            extra: { sourceDims: 2 },
        });
        const a = b.addNode("a");
        const c = b.addNode("c");
        b.setNodeValue("position", a, [0.1, 2, 0]);
        b.setNodeValue("position", c, [1, 1, 3]);
        const s = b.freeze();
        const cyto = jsonExporter.check(s, { dialect: "cytoscape" });
        expect(codes(cyto)).toEqual([JSON_LOSS.POSITION_Z_DROPPED]);
        expect(note(cyto, JSON_LOSS.POSITION_Z_DROPPED)?.count).toBe(1);
        const doc = (await exported(s, { dialect: "cytoscape" })).elements as { nodes: Record<string, unknown>[] };
        expect(doc.nodes).toEqual([
            { data: { id: "a" }, position: { x: 0.1, y: 2 } },
            { data: { id: "c" }, position: { x: 1, y: 1 } },
        ]);
        const plain = jsonExporter.check(s);
        expect(codes(plain)).toEqual([JSON_LOSS.POSITIONS_DROPPED]);
        expect((await exported(s)).nodes).toEqual([
            { id: "a", position: [0.1, 2] },
            { id: "c", position: [1, 1] },
        ]);
        const full = new GraphBuilder({ directed: true });
        full.declareNodeColumn({ name: "position", dtype: "f64", components: 3, role: "position" });
        full.setNodeValue("position", full.addNode("a"), [1, 2, 3]);
        expect((await exported(full.freeze(), { dialect: "vis" })).nodes).toEqual([{ id: "a", position: [1, 2, 3] }]);
    });

    it("writes the Cytoscape parent by id and reports a parents column or hierarchy elsewhere", async () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        b.declareNodeColumn({ name: "parents", dtype: "list", itemDtype: "u32", role: "parents", refersTo: "node" });
        const p = b.addNode("p");
        const c = b.addNode("c");
        b.setNodeValue("parent", c, p);
        b.setNodeValue("parents", c, [p]);
        const s = b.freeze();
        const cyto = jsonExporter.check(s, { dialect: "cytoscape" });
        expect(codes(cyto)).toEqual([JSON_LOSS.PARENTS_DROPPED]);
        const doc = (await exported(s, { dialect: "cytoscape" })).elements as { nodes: Record<string, unknown>[] };
        expect(doc.nodes).toEqual([{ data: { id: "p" } }, { data: { id: "c", parent: "p" } }]);
        const plain = jsonExporter.check(s);
        expect(codes(plain)).toEqual([LOSS.HIERARCHY, LOSS.HIERARCHY]);
        expect((await exported(s)).nodes).toEqual([{ id: "p" }, { id: "c" }]);
    });

    it("generates Cytoscape edge ids that avoid existing ids", async () => {
        const b = new GraphBuilder({ directed: true });
        b.declareEdgeColumn({ name: "id", dtype: "string", role: "id" });
        b.addNode("e1");
        const first = b.addEdge("a", "b");
        b.addEdge("b", "a");
        b.addEdge("a", "a");
        b.setEdgeValue("id", first, "e2");
        const s = b.freeze();
        const notes = jsonExporter.check(s, { dialect: "cytoscape" });
        expect(codes(notes)).toEqual([LOSS.EDGE_IDS_GENERATED]);
        expect(note(notes, LOSS.EDGE_IDS_GENERATED)?.count).toBe(2);
        const doc = (await exported(s, { dialect: "cytoscape" })).elements as { edges: { data: { id: string } }[] };
        expect(doc.edges.map((e) => e.data.id)).toEqual(["e2", "e1_2", "e2_2"]);
    });

    it("reports temporal roles, defaults, options, dynamic columns and extension tables through the shared pre-flight", async () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
        b.declareNodeColumn({ name: "start.text", dtype: "string", role: "timeText", extra: { for: "start" } });
        b.declareNodeColumn({ name: "level", dtype: "i32", default: 3 });
        b.declareNodeColumn({ name: "colour", dtype: "string", options: ["red", "blue"] });
        b.declareNodeColumn({ name: "price", dtype: "f64", dynamic: true });
        const a = b.addNode("a");
        b.setNodeValue("start", a, 1);
        b.setNodeValue("start.text", a, "2020");
        b.setNodeValue("colour", a, "red");
        b.setNodeValue("price", a, 2);
        const t = b.addExtensionTable("temporal:node:price", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "end", dtype: "f64" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [a, 0, 1, 2]);
        b.addExtensionTable("other", [{ name: "x", dtype: "i32" }]);
        const s = b.freeze();
        const notes = jsonExporter.check(s);
        expect(codes(notes).sort()).toEqual(
            [
                LOSS.TEMPORAL,
                LOSS.TEMPORAL_TEXT,
                LOSS.DEFAULT,
                LOSS.EMPTY_COLUMN,
                LOSS.OPTIONS,
                LOSS.DYNAMIC_VALUES,
                LOSS.DYNAMIC_VALUES,
                LOSS.EXTENSION_TABLE,
                LOSS.INTEGRAL_F64,
            ].sort(),
        );
        expect((await exported(s)).nodes).toEqual([{ id: "a", colour: "red", price: 2 }]);
    });

    it("reports graph attributes that vis cannot carry", async () => {
        const b = new GraphBuilder({ directed: false });
        b.addNode("a");
        b.setGraphValue("name", "G");
        const s = b.freeze();
        expect(codes(jsonExporter.check(s, { dialect: "vis" }))).toEqual([LOSS.GRAPH_ATTRIBUTES]);
        expect(await exported(s, { dialect: "vis" })).toEqual({ nodes: [{ id: "a" }], edges: [] });
        expect((await exported(s, { dialect: "jgf" })).graph).toMatchObject({ metadata: { name: "G" } });
        expect(await exported(s, { dialect: "cytoscape" })).toEqual({
            elements: { nodes: [{ data: { id: "a" } }], edges: [] },
            data: { name: "G" },
        });
    });
});

describe("dialect shapes", () => {
    it("jgf writes the graph fields and nests element keys", async () => {
        const text = JSON.stringify({
            graph: {
                id: "g",
                label: "L",
                type: "t",
                directed: false,
                metadata: { m: 1 },
                nodes: { A: { label: "a", metadata: { p: 1 }, extra: true }, B: {} },
                edges: [
                    {
                        id: "x",
                        source: "A",
                        target: "B",
                        relation: "r",
                        label: "l",
                        metadata: { weight: 2, k: 1 },
                        custom: 5,
                    },
                ],
            },
        });
        const s = await imported(text);
        expect(jsonExporter.check(s)).toEqual([]);
        const doc = await exported(s, { indent: 2 });
        expect(doc).toEqual(JSON.parse(text));
    });

    it("cytoscape writes element keys, classes, data and the recorded top-level keys", async () => {
        const text = JSON.stringify({
            elements: {
                nodes: [
                    {
                        data: { id: "n", position: "p", weight: 4 },
                        position: { x: 1, y: 2 },
                        classes: "a b",
                        selected: true,
                        custom: 5,
                    },
                ],
                edges: [{ data: { id: "e", source: "n", target: "n", weight: 3 }, classes: "ec", scratch: { k: 1 } }],
            },
            data: { title: "T" },
            zoom: 2,
            style: [{ selector: "node" }],
        });
        const s = await imported(text);
        expect(jsonExporter.check(s)).toEqual([]);
        expect(await exported(s)).toEqual(JSON.parse(text));
    });

    it("graphology writes attributes, options and element keys", async () => {
        const text = JSON.stringify({
            attributes: { name: "G" },
            options: { type: "directed", multi: true, allowSelfLoops: false },
            nodes: [{ key: "a", attributes: { x: 1 }, extra: 1 }, { key: "b" }],
            edges: [{ key: "e", source: "a", target: "b", attributes: { weight: 2, key: "k" }, custom: 2 }],
        });
        const s = await imported(text);
        expect(jsonExporter.check(s)).toEqual([]);
        expect(await exported(s)).toEqual(JSON.parse(text));
        expect((await exported(mixed(), { dialect: "graphology" })).options).toEqual({ type: "mixed" });
        // a recorded allowSelfLoops: false is written back as false only while the graph has no loops
        const loops = new GraphBuilder({ directed: true, weightDtype: "f64" });
        loops.addEdge("a", "a");
        loops.setMeta({ extra: { json: { dialect: "graphology", allowSelfLoops: false } } });
        expect((await exported(loops.freeze())).options).toEqual({ type: "directed", allowSelfLoops: true });
    });

    it("vis writes ids, from / to and attributes", async () => {
        const text = JSON.stringify({
            nodes: [{ id: 1, label: "one" }, { id: 2 }],
            edges: [
                { id: "e", from: 1, to: 2, weight: 0.5, arrows: "to" },
                { from: 2, to: 1 },
            ],
        });
        const s = await imported(text);
        expect(jsonExporter.check(s)).toEqual([]);
        expect(await exported(s)).toEqual(JSON.parse(text));
    });

    it("d3 writes positional nodes without an id key", async () => {
        const s = await imported('{"nodes":[{"group":1},{"group":2}],"links":[{"source":0,"target":1}]}');
        const doc = await exported(s);
        // the bare d3 shape is written back as read: no directed / multigraph / graph keys
        expect(doc).toEqual({
            nodes: [{ group: 1 }, { group: 2 }],
            links: [{ source: 0, target: 1 }],
        });
        expect(codes(jsonExporter.check(s))).toEqual([]);
        const back = await imported(JSON.stringify(doc));
        expect((back.meta.extra.json as { dialect: string }).dialect).toBe("d3");
        // another dialect writes the positional ids it must name
        expect(await exported(s, { dialect: "node-link" })).toEqual({
            directed: false,
            multigraph: false,
            graph: {},
            nodes: [
                { group: 1, id: 0 },
                { group: 2, id: 1 },
            ],
            edges: [{ source: 0, target: 1 }],
        });
    });

    it("writes every dialect of the same snapshot as valid JSON", async () => {
        const s = await imported(readCorpusText("json", "sigma-format.json"));
        for (const dialect of JSON_DIALECTS) {
            const text = await jsonExporter.exportToString(s, { dialect: dialect, indent: 1 });
            expect(() => JSON.parse(text), dialect).not.toThrow();
        }
    });
});
