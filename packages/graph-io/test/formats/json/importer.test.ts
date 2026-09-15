import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DIRECTION_REFUSED_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { JSON_ISSUE, jsonImporter, type JsonImportOptions } from "../../../src/formats/json/index.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../../src/types.js";
import { corpusFiles, inputShapes, readCorpusBytes, readCorpusText } from "../../helpers/corpus.js";

type Options = JsonImportOptions & CommonImportOptions;

async function load(
    text: string | Uint8Array,
    options?: Options,
    builder?: Partial<ConstructorParameters<typeof GraphBuilder>[0]>,
): Promise<{ s: GraphSnapshot; report: ImportReport; b: GraphBuilder }> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64", ...builder });
    const report = await jsonImporter.import(text, b, options);
    return { s: b.freeze(), report, b };
}

function json(value: unknown): string {
    return JSON.stringify(value);
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

function names(s: GraphSnapshot, table: "nodes" | "edges" | "graph"): string[] {
    return [...s[table]].map((c) => c.meta.name);
}

function cell(s: GraphSnapshot, table: "nodes" | "edges" | "graph", name: string, row: number): unknown {
    const column = s[table].require(name);
    return column.dtype === "json" ? column.values[row] : column.value(row);
}

function edge(s: GraphSnapshot, e: number): string {
    const list = s.edgeList();
    return `${String(s.ids.idOf(list.src[e]))}->${String(s.ids.idOf(list.dst[e]))}`;
}

async function expectImportError(text: string, options?: Options): Promise<ImportError> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    let caught: unknown;
    try {
        await jsonImporter.import(text, b, options);
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(ImportError);
    const error = caught as ImportError;
    expect(error.code).toBe("E_IMPORT");
    expect(error.report.format).toBe("json");
    return error;
}

// ============================================================ the plugin shape

describe("jsonImporter plugin", () => {
    it("declares its format, extensions and mime types", () => {
        expect(jsonImporter.format).toBe("json");
        expect(jsonImporter.extensions).toEqual([".json"]);
        expect(jsonImporter.mimeTypes).toEqual(["application/json"]);
    });

    it("sniffs graph documents with a higher confidence than arbitrary JSON", () => {
        const encoder = new TextEncoder();
        const sniff = (text: string): number => jsonImporter.sniff?.(encoder.encode(text)) ?? -1;
        expect(sniff('{"nodes": [], "links": []}')).toBe(0.9);
        expect(sniff('  \n{"elements": {}}')).toBe(0.9);
        expect(sniff('{"graph": {"nodes": {}}}')).toBe(0.9);
        expect(sniff('[{"data": {"id": "a"}}]')).toBe(0.5);
        expect(sniff('{"other": 1}')).toBe(0.5);
        expect(sniff("<graphml/>")).toBe(0);
        expect(sniff("source,target\na,b")).toBe(0);
        expect(sniff("")).toBe(0);
        const bom = new Uint8Array([0xef, 0xbb, 0xbf, ...encoder.encode('{"nodes": []}')]);
        expect(jsonImporter.sniff?.(bom)).toBe(0.9);
    });

    it("rejects bad format-specific options with E_UNSUPPORTED", async () => {
        const b = new GraphBuilder({ directed: true });
        const bad: Options[] = [
            { dialect: "xml" as never },
            { indexLinks: "yes" as never },
            { graphIndex: -1 },
            { graphIndex: 1.5 },
            { nodeIdKey: "" },
            { edgesKey: 3 as never },
            { ids: "guess" as never },
        ];
        for (const options of bad) {
            await expect(jsonImporter.import('{"nodes":[]}', b, options)).rejects.toMatchObject({
                code: "E_UNSUPPORTED",
            });
        }
    });
});

// ============================================================ the corpus

const EXPECTED: Record<string, { dialect: string; directed: boolean }> = {
    "d3-format.json": { dialect: "d3", directed: false },
    "cytoscape-format.json": { dialect: "cytoscape", directed: true },
    "sigma-format.json": { dialect: "graphology", directed: true },
    "visjs-format.json": { dialect: "vis", directed: false },
    "networkx-format.json": { dialect: "node-link", directed: true },
    "karate-d3.json": { dialect: "d3", directed: false },
    "miserables.json": { dialect: "d3", directed: false },
};

describe("corpus", () => {
    for (const entry of corpusFiles("json")) {
        it(`imports ${entry.path} with the manifest's counts`, async () => {
            const { s, report } = await load(readCorpusText("json", entry.path));
            expect(s.nodeCount).toBe(entry.expectedNodes);
            expect(s.edgeCount).toBe(entry.expectedEdges);
            expect(report.counts.nodes).toBe(entry.expectedNodes);
            expect(report.counts.edges).toBe(entry.expectedEdges);
            expect(report.counts.skippedNodes).toBe(0);
            expect(report.counts.skippedEdges).toBe(0);
            expect(report.errorCount).toBe(0);
            expect(report.warningCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(report.durationMs).toBeGreaterThanOrEqual(0);
            const expected = EXPECTED[entry.path];
            expect(s.directed).toBe(expected.directed);
            expect(s.meta.sourceFormat).toBe("json");
            expect((s.meta.extra.json as { dialect: string }).dialect).toBe(expected.dialect);
        });
    }

    it("d3-format.json: string ids, an i32 group, a string label and an unweighted value attribute", async () => {
        const { s } = await load(readCorpusText("json", "d3-format.json"));
        expect(s.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(names(s, "nodes")).toEqual(["group", "label"]);
        expect(s.nodes.require("group").dtype).toBe("i32");
        expect(cell(s, "nodes", "group", 4)).toBe(3);
        expect(cell(s, "nodes", "label", 0)).toBe("Node A");
        expect(s.flags.weighted).toBe(false);
        expect(names(s, "edges")).toEqual(["value"]);
        expect(cell(s, "edges", "value", 3)).toBe(3);
        expect(edge(s, 4)).toBe("a->c");
    });

    it("d3-format.json: weightFrom value makes the value column the weight", async () => {
        const { s } = await load(readCorpusText("json", "d3-format.json"), { weightFrom: "value" });
        expect(s.flags.weighted).toBe(true);
        expect(names(s, "edges")).toEqual([]);
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([1, 2, 1, 3, 1]);
        expect(s.meta.weightOrigin).toEqual({ format: "json", id: "value", title: null, type: null, namespace: null });
    });

    it("cytoscape-format.json: data.id ids, an id-role edge column, data.weight as THE weight", async () => {
        const { s } = await load(readCorpusText("json", "cytoscape-format.json"));
        expect(s.ids.toArray()).toEqual(["n1", "n2", "n3", "n4", "n5"]);
        expect(names(s, "nodes")).toEqual(["label", "weight"]);
        expect(cell(s, "nodes", "weight", 3)).toBe(25);
        const ids = s.edges.byRole("id");
        expect(ids?.meta.name).toBe("id");
        expect(ids?.meta.unique).toBe(true);
        expect(ids?.value(4)).toBe("e5");
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([1, 2.5, 1.5, 3, 0.5]);
        expect(s.edges.byRole("weight")).toBeNull();
        expect(s.edgeIndexOf("e3")).toBe(2);
    });

    it("sigma-format.json: graphology keys, nested attributes, an f64 weight shadow for 0.8", async () => {
        const { s } = await load(readCorpusText("json", "sigma-format.json"));
        expect(s.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(names(s, "nodes")).toEqual(["label", "x", "y", "size"]);
        expect(s.nodes.require("x").dtype).toBe("f64");
        expect(cell(s, "nodes", "x", 3)).toBe(1.5);
        expect(cell(s, "nodes", "size", 1)).toBe(15);
        expect(s.edges.byRole("id")?.value(0)).toBe("e1");
        const shadow = s.edges.byRole("weight");
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.value(4)).toBe(0.8);
        expect(edge(s, 4)).toBe("e->a");
    });

    it("visjs-format.json: numeric ids kept numeric, from / to endpoints, undirected by default", async () => {
        const { s } = await load(readCorpusText("json", "visjs-format.json"));
        expect(s.ids.toArray()).toEqual([1, 2, 3, 4, 5]);
        expect(s.ids.kind).toBe("identity");
        expect(cell(s, "nodes", "color", 2)).toBe("#0000FF");
        expect(cell(s, "edges", "label", 4)).toBe("edge 5");
        expect(edge(s, 4)).toBe("5->1");
        expect(s.edges.byRole("id")).toBeNull();
    });

    it("networkx-format.json: the directed flag, the multigraph flag, the graph dict and explicit weights", async () => {
        const { s } = await load(readCorpusText("json", "networkx-format.json"));
        expect(s.directed).toBe(true);
        expect(s.meta.declaredMultigraph).toBe(false);
        expect(s.graph.names()).toEqual([]);
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([1, 2, 1.5, 1, 0.5]);
        expect(s.meta.extra.json).toEqual({
            dialect: "node-link",
            edgesKey: "links",
            nodeIdKey: "id",
            indexLinks: false,
            sourceKey: "source",
            targetKey: "target",
        });
    });

    it("karate-d3.json: string ids stay strings under ids keep and become numbers under canonical", async () => {
        const kept = await load(readCorpusText("json", "karate-d3.json"));
        expect(kept.s.ids.idOf(0)).toBe("1");
        expect(kept.s.ids.kind).toBe("string");
        expect(cell(kept.s, "nodes", "name", 33)).toBe("Node 34");
        const canonical = await load(readCorpusText("json", "karate-d3.json"), { ids: "canonical" });
        expect(canonical.s.ids.idOf(0)).toBe(1);
        expect(canonical.s.edgeCount).toBe(78);
        expect(edge(canonical.s, 0)).toBe("2->1");
    });

    it("miserables.json: name ids and integer index endpoints", async () => {
        const { s } = await load(readCorpusText("json", "miserables.json"));
        expect(s.ids.idOf(0)).toBe("Myriel");
        expect(s.ids.idOf(11)).toBe("Valjean");
        expect(edge(s, 0)).toBe("Napoleon->Myriel");
        expect(cell(s, "edges", "value", 1)).toBe(8);
        expect(cell(s, "nodes", "group", 12)).toBe(3);
        expect(s.meta.extra.json).toMatchObject({ dialect: "d3", nodeIdKey: "name", indexLinks: true });
    });

    it("reads every input shape to the same graph", async () => {
        const bytes = readCorpusBytes("json", "miserables.json");
        const reference = await load(bytes);
        for (const shape of inputShapes(bytes)) {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const report = await jsonImporter.import(shape.make(), b);
            const s = b.freeze();
            expect(s.nodeCount, shape.name).toBe(77);
            expect(s.edgeCount, shape.name).toBe(254);
            expect(report.errorCount, shape.name).toBe(0);
            expect(s.ids.toArray(), shape.name).toEqual(reference.s.ids.toArray());
        }
    });
});

// ============================================================ node-link

describe("node-link", () => {
    it("reads the new edges key form and records it", async () => {
        const { s } = await load(
            json({ directed: false, nodes: [{ id: 1 }, { id: 2 }], edges: [{ source: 1, target: 2 }] }),
        );
        expect(s.directed).toBe(false);
        expect(s.edgeCount).toBe(1);
        expect(s.meta.extra.json).toMatchObject({ dialect: "node-link", edgesKey: "edges" });
    });

    it("prefers edges over links when both exist unless edgesKey says otherwise", async () => {
        const doc = { nodes: [{ id: 1 }, { id: 2 }], edges: [{ source: 1, target: 2 }], links: [] };
        expect((await load(json(doc))).s.edgeCount).toBe(1);
        expect((await load(json(doc), { edgesKey: "links" })).s.edgeCount).toBe(0);
    });

    it('keeps 1 and "1" distinct under ids keep and merges texts under ids number with a warning', async () => {
        const doc = { nodes: [{ id: 1 }, { id: "1" }, { id: "01" }], links: [] };
        const kept = await load(json(doc));
        expect(kept.s.ids.toArray()).toEqual([1, "1", "01"]);
        const numeric = await load(json(doc), { ids: "number" });
        expect(numeric.s.ids.toArray()).toEqual([1]);
        // "1" merges into the existing node 1 (a duplicate declaration) and "01" is a text merge as well
        expect(codes(numeric.report)).toEqual([
            JSON_ISSUE.DUPLICATE_NODE,
            JSON_ISSUE.ID_MERGED,
            JSON_ISSUE.DUPLICATE_NODE,
        ]);
        expect(numeric.report.issues[1]?.category).toBe("coercion");
        expect(numeric.report.counts.nodes).toBe(1);
        const strings = await load(json(doc), { ids: "string" });
        expect(strings.s.ids.toArray()).toEqual(["1", "01"]);
    });

    it("reports boolean and null ids as unsupported unless ids is string", async () => {
        const doc = { nodes: [{ id: true }, { id: null }, { id: "ok" }], links: [{ source: true, target: "ok" }] };
        const { s, report } = await load(json(doc));
        expect(s.ids.toArray()).toEqual(["ok"]);
        expect(codes(report)).toEqual([
            JSON_ISSUE.UNSUPPORTED_ID,
            JSON_ISSUE.UNSUPPORTED_ID,
            JSON_ISSUE.UNSUPPORTED_ID,
        ]);
        expect(report.issues.every((i) => i.category === "unsupported")).toBe(true);
        expect(report.counts).toEqual({ nodes: 1, edges: 0, skippedNodes: 2, skippedEdges: 1, expandedMixed: 0 });
        const strings = await load(json(doc), { ids: "string" });
        expect(strings.s.ids.toArray()).toEqual(["true", "null", "ok"]);
        expect(strings.s.edgeCount).toBe(1);
    });

    it("rejects non-finite numeric ids through the coercion rule", async () => {
        const { s, report } = await load('{"nodes":[{"id":1.5},{"id":2}],"links":[]}');
        expect(s.ids.toArray()).toEqual([1.5, 2]);
        expect(report.errorCount).toBe(0);
    });

    it("flattens node attributes next to the id and infers columns per column", async () => {
        const doc = {
            nodes: [
                { id: "a", n: 1, f: true, t: "x", nested: { k: [1] }, list: [1, 2], nothing: null },
                { id: "b", n: 2.5, f: 0, t: 3 },
            ],
            links: [],
        };
        const { s } = await load(json(doc));
        expect(names(s, "nodes")).toEqual(["n", "f", "t", "nested", "list"]);
        expect(s.nodes.require("n").dtype).toBe("f64");
        expect(s.nodes.require("f").dtype).toBe("i32");
        expect(s.nodes.require("t").dtype).toBe("string");
        expect(s.nodes.require("nested").dtype).toBe("json");
        expect(s.nodes.require("list").dtype).toBe("json");
        expect(cell(s, "nodes", "nested", 0)).toEqual({ k: [1] });
        expect(cell(s, "nodes", "list", 0)).toEqual([1, 2]);
        expect(s.nodes.require("list").isSet(1)).toBe(false);
        expect(cell(s, "nodes", "t", 1)).toBe("3");
    });

    it("keeps edge attributes, skips the endpoint and weight keys, and records explicit versus defaulted weights", async () => {
        const doc = {
            nodes: [{ id: "a" }, { id: "b" }],
            links: [
                { source: "a", target: "b", weight: 2, key: 0, kind: "x" },
                { source: "b", target: "a", key: 1 },
                { source: "a", target: "a", weight: null },
            ],
        };
        const { s } = await load(json(doc));
        expect(names(s, "edges")).toEqual(["key", "kind", "graphty.weight"]);
        const shadow = s.edges.byRole("weight");
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.isSet(2)).toBe(false);
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([2, 1, 1]);
        expect(cell(s, "edges", "kind", 0)).toBe("x");
        expect(s.selfLoopCount).toBe(1);
    });

    it("reads numeric weight text and reports non-numeric weights as E_INVALID_WEIGHT", async () => {
        const doc = {
            nodes: [{ id: "a" }, { id: "b" }],
            links: [
                { source: "a", target: "b", weight: "2.5" },
                { source: "b", target: "a", weight: "heavy" },
                { source: "a", target: "b", weight: true },
            ],
        };
        const { s, report } = await load(json(doc));
        expect(s.edgeCount).toBe(1);
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([2.5]);
        expect(codes(report)).toEqual(["E_INVALID_WEIGHT", "E_INVALID_WEIGHT"]);
        expect(report.counts.skippedEdges).toBe(2);
        expect(report.issues[0]?.element).toBe("links[1]");
    });

    it("weightFrom null reads an unweighted graph and keeps weight as an attribute", async () => {
        const doc = { nodes: [{ id: "a" }], links: [{ source: "a", target: "a", weight: 3 }] };
        const { s } = await load(json(doc), { weightFrom: null });
        expect(s.flags.weighted).toBe(false);
        expect(cell(s, "edges", "weight", 0)).toBe(3);
        expect(s.meta.weightOrigin).toBeNull();
    });

    it("writes the graph dict as graph columns and records the multigraph flag", async () => {
        const doc = {
            directed: true,
            multigraph: true,
            graph: { name: "G", n: 2, tags: ["x"], skip: null },
            nodes: [],
            links: [],
        };
        const { s } = await load(json(doc));
        expect(s.graph.names()).toEqual(["name", "n", "tags"]);
        expect(cell(s, "graph", "name", 0)).toBe("G");
        expect(s.graph.require("tags").dtype).toBe("list");
        expect(s.meta.declaredMultigraph).toBe(true);
    });

    it("warns about non-boolean flags and a non-object graph dict and uses the defaults", async () => {
        const doc = { directed: "yes", multigraph: 1, graph: [1], nodes: [], links: [] };
        const { s, report } = await load(json(doc));
        expect(s.directed).toBe(false);
        expect(s.meta.declaredMultigraph).toBe(false);
        expect(codes(report)).toEqual([JSON_ISSUE.BAD_FLAG, JSON_ISSUE.BAD_FLAG, JSON_ISSUE.BAD_FLAG]);
        expect(report.errorCount).toBe(0);
        expect(report.warningCount).toBe(3);
    });

    it("uses defaultDirected for a file without the flag", async () => {
        const doc = { nodes: [{ id: "a" }], links: [] };
        expect((await load(json(doc))).s.directed).toBe(false);
        expect((await load(json(doc), { defaultDirected: true })).s.directed).toBe(true);
        expect((await load(json({ ...doc, directed: false }), { defaultDirected: true })).s.directed).toBe(false);
    });

    it("accepts src / dst and from / to endpoint keys and records the keys used", async () => {
        const { s } = await load(json({ nodes: [{ id: 1 }, { id: 2 }], edges: [{ src: 1, dst: 2 }] }));
        expect(edge(s, 0)).toBe("1->2");
        expect(names(s, "edges")).toEqual([]);
        expect(s.meta.extra.json).toMatchObject({ sourceKey: "src", targetKey: "dst" });
        const explicit = await load(json({ nodes: [{ id: 1 }, { id: 2 }], edges: [{ a: 1, b: 2, source: 9 }] }), {
            sourceKey: "a",
            targetKey: "b",
        });
        expect(edge(explicit.s, 0)).toBe("1->2");
        expect(cell(explicit.s, "edges", "source", 0)).toBe(9);
    });

    it("reports edges without an endpoint and non-object records, keeping the rest", async () => {
        const doc = {
            nodes: [{ id: "a" }, 5, null, { id: "b" }],
            links: [{ target: "b" }, { source: "a" }, "x", { source: "a", target: "b" }],
        };
        const { s, report } = await load(json(doc));
        expect(s.nodeCount).toBe(2);
        expect(s.edgeCount).toBe(1);
        expect(codes(report)).toEqual([
            JSON_ISSUE.BAD_ELEMENT,
            JSON_ISSUE.BAD_ELEMENT,
            JSON_ISSUE.MISSING_ENDPOINT,
            JSON_ISSUE.MISSING_ENDPOINT,
            JSON_ISSUE.BAD_ELEMENT,
        ]);
        expect(report.counts).toEqual({ nodes: 2, edges: 1, skippedNodes: 2, skippedEdges: 3, expandedMixed: 0 });
        expect(report.issues[2]?.element).toBe("links[0]");
        expect(report.issues[2]?.category).toBe("missing-value");
    });

    it("reports a node without an id as missing-value and keeps the others", async () => {
        const doc = { nodes: [{ id: "a" }, { x: 1 }], links: [] };
        const { s, report } = await load(json(doc));
        expect(s.ids.toArray()).toEqual(["a"]);
        expect(codes(report)).toEqual([JSON_ISSUE.MISSING_ID]);
        expect(report.issues[0]?.element).toBe("nodes[1]");
    });

    it("creates endpoint nodes that were never declared under addMissingNodes (the default)", async () => {
        const { s } = await load('{"links":[{"source":"a","target":"b"}]}');
        expect(s.ids.toArray()).toEqual(["a", "b"]);
    });

    it("reports E_UNKNOWN_NODE per edge when the sink refuses missing nodes", async () => {
        const doc = {
            nodes: [{ id: "a" }],
            links: [
                { source: "a", target: "zz" },
                { source: "a", target: "a" },
            ],
        };
        const { s, report } = await load(json(doc), undefined, { addMissingNodes: false });
        expect(s.edgeCount).toBe(1);
        expect(codes(report)).toEqual(["E_UNKNOWN_NODE"]);
        expect(report.issues[0]?.category).toBe("missing-value");
    });

    it("nodeIdFrom label uses the label key, nodeIdFrom index uses array positions", async () => {
        const doc = {
            nodes: [
                { id: "x", label: "L1" },
                { id: "y", label: "L2" },
            ],
            links: [{ source: "L1", target: "L2" }],
        };
        const byLabel = await load(json(doc), { nodeIdFrom: "label" });
        expect(byLabel.s.ids.toArray()).toEqual(["L1", "L2"]);
        expect(names(byLabel.s, "nodes")).toEqual(["id"]);
        expect(byLabel.s.edgeCount).toBe(1);
        const positional = await load(json({ ...doc, links: [{ source: 1, target: 0 }] }), { nodeIdFrom: "index" });
        expect(positional.s.ids.toArray()).toEqual([0, 1]);
        expect(names(positional.s, "nodes")).toEqual(["id", "label"]);
        expect(edge(positional.s, 0)).toBe("1->0");
        expect(positional.s.meta.extra.json).toMatchObject({ nodeIdKey: null, indexLinks: true });
    });

    it("uses array positions with a warning when no node has an id or name", async () => {
        const doc = { nodes: [{ group: 1 }, { group: 2 }], links: [{ source: 0, target: 1 }] };
        const { s, report } = await load(json(doc));
        expect(s.ids.toArray()).toEqual([0, 1]);
        expect(codes(report)).toEqual([JSON_ISSUE.POSITIONAL_NODES]);
        expect(report.warningCount).toBe(1);
        expect(s.edgeCount).toBe(1);
    });

    it("index links: integer endpoints below the node count with string ids are positions", async () => {
        const doc = { nodes: [{ id: "a" }, { id: "b" }], links: [{ source: 1, target: 0 }] };
        const auto = await load(json(doc));
        expect(edge(auto.s, 0)).toBe("b->a");
        expect(auto.s.nodeCount).toBe(2);
        const forced = await load(json(doc), { indexLinks: false });
        expect(forced.s.nodeCount).toBe(4);
        expect(edge(forced.s, 0)).toBe("1->0");
        const numericIds = await load(json({ nodes: [{ id: 0 }, { id: 1 }], links: [{ source: 1, target: 0 }] }));
        expect(numericIds.s.nodeCount).toBe(2);
        expect(numericIds.s.meta.extra.json).toMatchObject({ indexLinks: false });
    });

    it("index links: an out-of-range or skipped position is E_BAD_INDEX", async () => {
        const doc = {
            nodes: [{ id: "a" }, { nope: 1 }],
            links: [
                { source: 0, target: 1 },
                { source: 0, target: 5 },
                { source: 0, target: 0 },
            ],
        };
        const { s, report } = await load(json(doc), { indexLinks: true });
        expect(s.edgeCount).toBe(1);
        expect(codes(report)).toEqual([JSON_ISSUE.MISSING_ID, JSON_ISSUE.BAD_INDEX, JSON_ISSUE.BAD_INDEX]);
        expect(report.counts.skippedEdges).toBe(2);
    });

    it("nodeIdKey selects the id field explicitly", async () => {
        const doc = { nodes: [{ id: "a", uid: 7 }], links: [{ source: 7, target: 7 }] };
        const { s } = await load(json(doc), { nodeIdKey: "uid" });
        expect(s.ids.toArray()).toEqual([7]);
        expect(cell(s, "nodes", "id", 0)).toBe("a");
        expect(s.edgeCount).toBe(1);
    });

    it("reserves both nodes and edges as errors when a section is missing, but still imports", async () => {
        const nodesOnly = await load('{"nodes":[{"id":1},{"id":2}]}');
        expect(nodesOnly.s.nodeCount).toBe(2);
        expect(codes(nodesOnly.report)).toEqual([JSON_ISSUE.MISSING_SECTION]);
        expect(nodesOnly.report.issues[0]?.category).toBe("missing-value");
        expect(nodesOnly.report.issues[0]?.severity).toBe("error");
        const edgesOnly = await load('{"edges":[{"source":1,"target":2}]}');
        expect(edgesOnly.s.nodeCount).toBe(2);
        expect(edgesOnly.s.edgeCount).toBe(1);
        expect(codes(edgesOnly.report)).toEqual([JSON_ISSUE.MISSING_SECTION]);
    });

    it("keeps the sink's direction with a coercion warning when the sink is locked", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.lockDirected();
        const report = await jsonImporter.import(
            json({ directed: false, nodes: [{ id: 1 }, { id: 2 }], links: [{ source: 1, target: 2 }] }),
            b,
        );
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        const s = b.freeze();
        expect(s.directed).toBe(true);
        expect(s.edgeCount).toBe(2);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.counts.edges).toBe(2);
    });

    it("dialect can be forced", async () => {
        const doc = { nodes: [{ key: "a" }], edges: [] };
        const auto = await load(json(doc));
        expect(auto.s.meta.extra.json).toMatchObject({ dialect: "graphology" });
        const forced = await load(json(doc), { dialect: "node-link" });
        expect(forced.s.meta.extra.json).toMatchObject({ dialect: "node-link", nodeIdKey: null });
        expect(codes(forced.report)).toEqual([JSON_ISSUE.POSITIONAL_NODES]);
        expect(forced.s.ids.toArray()).toEqual([0]);
        expect(cell(forced.s, "nodes", "key", 0)).toBe("a");
    });
});

// ============================================================ graphology

describe("graphology", () => {
    const mixed = {
        options: { type: "mixed", multi: true, allowSelfLoops: false },
        attributes: { name: "G" },
        nodes: [
            { key: "a", attributes: { label: "A", tags: ["x"] }, extra: 1 },
            { key: "b" },
            { key: "c", attributes: null },
        ],
        edges: [
            { key: "e1", source: "a", target: "b", attributes: { weight: 0.1, key: "shadow" } },
            { key: "e2", source: "b", target: "c", undirected: true, attributes: { w: 2 } },
            { key: "e3", source: "c", target: "c", undirected: true },
        ],
    };

    it("reads a mixed graph: directed header, undirected edges expanded into pairs", async () => {
        const { s, report } = await load(json(mixed));
        expect(s.directed).toBe(true);
        expect(s.nodeCount).toBe(3);
        expect(s.edgeCount).toBe(4);
        expect(report.counts).toEqual({ nodes: 3, edges: 4, skippedNodes: 0, skippedEdges: 0, expandedMixed: 2 });
        expect(report.errorCount).toBe(0);
        const pair = s.edges.byRole("pair");
        expect(pair?.value(1)).toBe(2);
        expect(pair?.value(2)).toBe(1);
        expect(pair?.isSet(3)).toBe(false);
        expect(s.edges.byRole("directed")?.value(3)).toBe(false);
        expect(s.meta.declaredMultigraph).toBe(true);
        expect(s.meta.extra.json).toEqual({ dialect: "graphology", allowSelfLoops: false });
        expect(cell(s, "graph", "name", 0)).toBe("G");
    });

    it("declares the edge key column with role id, and suffixes a colliding attribute", async () => {
        const { s } = await load(json(mixed));
        const ids = s.edges.byRole("id");
        expect(ids?.meta.name).toBe("key");
        expect(ids?.meta.unique).toBe(true);
        expect(ids?.value(0)).toBe("e1");
        expect(cell(s, "edges", "key#data", 0)).toBe("shadow");
        expect(cell(s, "edges", "w", 1)).toBe(2);
        expect(s.edges.require("w").isSet(2)).toBe(false);
    });

    it("keeps unknown element-level keys with the #element suffix", async () => {
        const { s } = await load(json(mixed));
        expect(names(s, "nodes")).toEqual(["label", "tags", "extra#element"]);
        expect(cell(s, "nodes", "extra#element", 0)).toBe(1);
        expect(cell(s, "nodes", "tags", 0)).toEqual(["x"]);
    });

    it("honours options.type directed and undirected over per-edge flags", async () => {
        const directed = await load(json({ ...mixed, options: { type: "directed" } }));
        expect(directed.s.directed).toBe(true);
        expect(directed.s.edgeCount).toBe(3);
        expect(directed.report.counts.expandedMixed).toBe(0);
        const undirected = await load(json({ ...mixed, options: { type: "undirected" } }));
        expect(undirected.s.directed).toBe(false);
        expect(undirected.s.edgeCount).toBe(3);
        expect(undirected.report.counts.expandedMixed).toBe(0);
    });

    it("reads a mixed graph whose edges are all undirected as undirected", async () => {
        const doc = {
            nodes: [{ key: "a" }, { key: "b" }],
            edges: [{ source: "a", target: "b", undirected: true }],
        };
        const { s } = await load(json(doc));
        expect(s.directed).toBe(false);
        expect(s.edgeCount).toBe(1);
        const empty = await load(json({ options: { type: "mixed" }, nodes: [{ key: "a" }], edges: [] }));
        expect(empty.s.directed).toBe(true);
        const emptyDefault = await load(json({ options: { type: "mixed" }, nodes: [{ key: "a" }], edges: [] }), {
            defaultDirected: false,
        });
        expect(emptyDefault.s.directed).toBe(false);
    });

    it("applies onMixedDirection error, directed and undirected", async () => {
        const error = await expectImportError(json(mixed), { onMixedDirection: "error" });
        expect(error.report.issues.at(-1)?.code).toBe("E_MIXED_DIRECTION");
        const forced = await load(json(mixed), { onMixedDirection: "undirected" });
        expect(forced.s.directed).toBe(false);
        expect(forced.s.edgeCount).toBe(3);
        expect(forced.report.issues.map((i) => i.code)).toContain("W_DIRECTION_FORCED");
        const directed = await load(json(mixed), { onMixedDirection: "directed" });
        expect(directed.s.directed).toBe(true);
        expect(directed.s.edgeCount).toBe(3);
    });

    it("warns about a bad options object and bad flags", async () => {
        const doc = { options: "x", nodes: [{ key: "a" }], edges: [{ source: "a", target: "a", undirected: "yes" }] };
        const { s, report } = await load(json(doc), { dialect: "graphology" });
        expect(codes(report)).toEqual([JSON_ISSUE.BAD_FLAG, JSON_ISSUE.BAD_FLAG]);
        expect(s.edgeCount).toBe(1);
        const badType = await load(json({ options: { type: "wrong", multi: "no" }, nodes: [], edges: [] }));
        expect(codes(badType.report)).toEqual([JSON_ISSUE.BAD_FLAG, JSON_ISSUE.BAD_FLAG]);
    });

    it("reports a non-object attributes dict and keeps the node", async () => {
        const doc = { nodes: [{ key: "a", attributes: [1] }], edges: [] };
        const { s, report } = await load(json(doc), { dialect: "graphology" });
        expect(s.nodeCount).toBe(1);
        expect(codes(report)).toEqual([JSON_ISSUE.BAD_VALUE]);
    });

    it("stores mixed numeric and string edge keys as text with one warning", async () => {
        const doc = {
            nodes: [{ key: "a" }],
            edges: [
                { key: 1, source: "a", target: "a" },
                { key: "x", source: "a", target: "a" },
                { key: true, source: "a", target: "a" },
            ],
        };
        const { s, report } = await load(json(doc));
        expect(s.edges.byRole("id")?.dtype).toBe("string");
        expect(s.edges.byRole("id")?.value(0)).toBe("1");
        expect(codes(report)).toEqual([JSON_ISSUE.EDGE_ID_STRINGIFIED, "E_COLUMN_TYPE"]);
        expect(s.edgeCount).toBe(2);
        const numeric = await load(json({ nodes: [{ key: "a" }], edges: [{ key: 7, source: "a", target: "a" }] }));
        expect(numeric.s.edges.byRole("id")?.dtype).toBe("f64");
        expect(numeric.s.edges.byRole("id")?.value(0)).toBe(7);
    });
});

// ============================================================ JGF

describe("jgf", () => {
    const doc = {
        graph: {
            id: "g1",
            label: "Graph",
            type: "t",
            directed: true,
            metadata: { m: 1 },
            nodes: { A: { label: "a", metadata: { p: 1 } }, B: {}, C: { extra: true }, D: null },
            edges: [
                { id: "x", source: "A", target: "B", relation: "knows", label: "L", metadata: { weight: 2 } },
                { source: "B", target: "C", directed: false },
            ],
        },
    };

    it("reads nodes keyed by id, label and relation roles, metadata columns and the graph fields", async () => {
        const { s, report } = await load(json(doc));
        expect(report.errorCount).toBe(0);
        expect(s.ids.toArray()).toEqual(["A", "B", "C", "D"]);
        expect(s.nodes.byRole("label")?.value(0)).toBe("a");
        expect(s.nodes.byRole("label")?.isSet(1)).toBe(false);
        expect(cell(s, "nodes", "p", 0)).toBe(1);
        expect(cell(s, "nodes", "extra#element", 2)).toBe(true);
        expect(s.edges.byRole("id")?.value(0)).toBe("x");
        expect(s.edges.byRole("id")?.meta.unique).toBe(false);
        expect(s.edges.byRole("kind")?.meta.name).toBe("relation");
        expect(s.edges.byRole("kind")?.value(0)).toBe("knows");
        expect(s.edges.byRole("label")?.value(0)).toBe("L");
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([2, 1, 1]);
        expect(s.meta.name).toBe("Graph");
        expect(s.meta.extra.json).toEqual({ dialect: "jgf", id: "g1", type: "t" });
        expect(cell(s, "graph", "m", 0)).toBe(1);
    });

    it("expands a directed: false edge of a directed graph into a pair", async () => {
        const { s, report } = await load(json(doc));
        expect(s.directed).toBe(true);
        expect(s.edgeCount).toBe(3);
        expect(report.counts.expandedMixed).toBe(1);
        expect(s.edges.byRole("pair")?.value(1)).toBe(2);
    });

    it("reads graph.directed false, and a missing flag from the edges", async () => {
        const undirected = await load(json({ graph: { ...doc.graph, directed: false } }));
        expect(undirected.s.directed).toBe(false);
        expect(undirected.s.edgeCount).toBe(2);
        const allFalse = await load(
            json({
                graph: {
                    nodes: { A: {}, B: {} },
                    edges: [{ source: "A", target: "B", directed: false }],
                },
            }),
        );
        expect(allFalse.s.directed).toBe(false);
        const noEdges = await load(json({ graph: { nodes: { A: {} } } }));
        expect(noEdges.s.directed).toBe(true);
        const explicit = await load(json({ graph: { nodes: { A: {} }, edges: [] } }), { defaultDirected: false });
        expect(explicit.s.directed).toBe(false);
        const bad = await load(json({ graph: { nodes: { A: {} }, edges: [], directed: "no" } }));
        expect(codes(bad.report)).toEqual([JSON_ISSUE.BAD_FLAG]);
    });

    it("reads a v1 node array with ids", async () => {
        const v1 = { graph: { nodes: [{ id: "n1", label: "x" }, { id: "n2" }, 4], edges: [] } };
        const { s, report } = await load(json(v1));
        expect(s.ids.toArray()).toEqual(["n1", "n2"]);
        expect(codes(report)).toEqual([JSON_ISSUE.BAD_ELEMENT]);
        expect(s.nodes.byRole("label")?.value(0)).toBe("x");
    });

    it("reads graphs[graphIndex] with a warning about the others", async () => {
        const multi = {
            graphs: [{ nodes: { A: {} } }, { nodes: { B: {}, C: {} }, edges: [{ source: "B", target: "C" }] }],
        };
        const first = await load(json(multi));
        expect(first.s.ids.toArray()).toEqual(["A"]);
        expect(codes(first.report)).toEqual([JSON_ISSUE.MULTIPLE_GRAPHS]);
        const second = await load(json(multi), { graphIndex: 1 });
        expect(second.s.ids.toArray()).toEqual(["B", "C"]);
        expect(second.s.edgeCount).toBe(1);
        const beyond = await expectImportError(json(multi), { graphIndex: 5 });
        expect(beyond.report.issues.at(-1)?.code).toBe(JSON_ISSUE.SHAPE);
        await expectImportError('{"graphs":[]}');
        await expectImportError('{"graphs":[3]}');
        await expectImportError('{"graph":{"nodes":"x"}}');
    });

    it("reports non-string labels, relations and non-object metadata without dropping the element", async () => {
        const bad = {
            graph: {
                nodes: { A: { label: 3, metadata: "m" } },
                edges: [{ source: "A", target: "A", relation: 4, label: [], metadata: 1 }],
            },
        };
        const { s, report } = await load(json(bad));
        expect(s.nodeCount).toBe(1);
        expect(s.edgeCount).toBe(1);
        expect(codes(report)).toEqual([
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_VALUE,
        ]);
    });

    it("does not declare label or relation columns the file never uses", async () => {
        const { s } = await load(json({ graph: { nodes: { A: {} }, edges: [{ source: "A", target: "A" }] } }));
        expect(names(s, "nodes")).toEqual([]);
        expect(names(s, "edges")).toEqual([]);
    });

    describe("hyperedges", () => {
        const hyper = {
            graph: {
                directed: true,
                nodes: { A: {}, B: {}, C: {} },
                edges: [],
                hyperedges: [
                    { nodes: ["A", "B", "C"], relation: "team", metadata: { weight: 3 } },
                    { source: ["A"], target: ["B", "C"], id: "h2" },
                ],
            },
        };

        it("skips them by default with a warning and a loss note", async () => {
            const { s, report } = await load(json(hyper));
            expect(s.edgeCount).toBe(0);
            expect(codes(report)).toEqual([JSON_ISSUE.HYPEREDGES_SKIPPED]);
            expect(report.lossy).toEqual([
                {
                    code: JSON_ISSUE.HYPEREDGES_SKIPPED,
                    message: "2 hyperedge(s) were not imported",
                    column: null,
                    count: 2,
                },
            ]);
        });

        it("refuses them under hyperedges error", async () => {
            const error = await expectImportError(json(hyper), { hyperedges: "error" });
            expect(error.report.issues.at(-1)?.code).toBe(JSON_ISSUE.HYPEREDGE);
            expect(error.report.issues.at(-1)?.category).toBe("unsupported");
        });

        it("expands them as a star or a clique, carrying the hyperedge's fields", async () => {
            const star = await load(json(hyper), { hyperedges: "star" });
            expect(star.s.edgeCount).toBe(4);
            expect([0, 1, 2, 3].map((e) => edge(star.s, e))).toEqual(["A->B", "A->C", "A->B", "A->C"]);
            expect(star.s.edges.byRole("kind")?.value(0)).toBe("team");
            expect(star.s.edges.byRole("id")?.value(2)).toBe("h2");
            expect(Array.from(star.s.edgeList().weights ?? [])).toEqual([3, 3, 1, 1]);
            expect(names(star.s, "edges")).not.toContain("nodes#element");
            const clique = await load(json(hyper), { hyperedges: "clique" });
            expect(clique.s.edgeCount).toBe(5);
            expect([0, 1, 2].map((e) => edge(clique.s, e))).toEqual(["A->B", "A->C", "B->C"]);
            expect(clique.s.directed).toBe(true);
            expect(clique.s.edges.byRole("directed")).toBeNull();
        });

        it("expands an undirected hyperedge into undirected edges, and a directed one expands the sink", async () => {
            const undirected = {
                graph: { directed: false, nodes: { A: {}, B: {}, C: {} }, hyperedges: [{ nodes: ["A", "B", "C"] }] },
            };
            const clique = await load(json(undirected), { hyperedges: "clique" });
            expect(clique.s.directed).toBe(false);
            expect(clique.s.edgeCount).toBe(3);
            const mixed = {
                graph: {
                    directed: false,
                    nodes: { A: {}, B: {}, C: {} },
                    hyperedges: [{ nodes: ["A", "B"] }, { source: ["A"], target: ["C"] }],
                },
            };
            const star = await load(json(mixed), { hyperedges: "star" });
            expect(star.s.directed).toBe(true);
            expect(star.s.edgeCount).toBe(3);
            expect(star.report.counts.expandedMixed).toBe(1);
            expect(star.s.edges.byRole("directed")?.value(0)).toBe(false);
            expect(star.s.edges.byRole("directed")?.value(2)).toBe(true);
        });

        it("reports malformed hyperedges", async () => {
            const bad = {
                graph: {
                    nodes: { A: {} },
                    hyperedges: [
                        { nodes: ["A"] },
                        { source: [], target: ["A"] },
                        { relation: "x" },
                        7,
                        { nodes: [null] },
                    ],
                },
            };
            const { s, report } = await load(json(bad), { hyperedges: "clique" });
            expect(s.edgeCount).toBe(0);
            expect(codes(report)).toEqual([
                "E_INVALID_ID",
                "E_INVALID_ID",
                JSON_ISSUE.HYPEREDGE_SHAPE,
                JSON_ISSUE.BAD_ELEMENT,
                "E_INVALID_ID",
            ]);
            expect(report.counts.skippedEdges).toBe(5);
        });
    });
});

// ============================================================ Cytoscape

describe("cytoscape", () => {
    const doc = {
        elements: {
            nodes: [
                { data: { id: "p" }, selected: true },
                {
                    data: { id: "n1", parent: "p", position: "dataPos", weight: 4 },
                    position: { x: 0.1, y: 2 },
                    classes: "a  b",
                    locked: false,
                    custom: 5,
                },
                { data: { id: "n2", parent: "nope" }, classes: ["c"] },
            ],
            edges: [
                { data: { id: "e1", source: "n1", target: "n2", weight: 3 }, classes: "ec" },
                { data: { source: "n2", target: "p" }, selected: true },
            ],
        },
        data: { title: "T" },
        zoom: 2,
        pan: { x: 1, y: 2 },
    };

    it("reads elements.nodes / elements.edges with positions, classes, parents and element keys", async () => {
        const { s, report } = await load(json(doc));
        expect(s.ids.toArray()).toEqual(["p", "n1", "n2"]);
        expect(s.directed).toBe(true);
        expect(s.edgeCount).toBe(2);
        expect(codes(report)).toEqual([JSON_ISSUE.UNKNOWN_PARENT]);
        expect(report.issues[0]?.category).toBe("missing-value");
        const position = s.nodes.byRole("position");
        expect(position?.meta.name).toBe("position");
        expect(position?.dtype).toBe("f32");
        expect(position?.meta.components).toBe(3);
        expect(position?.meta.mutable).toBe(true);
        expect(position?.meta.extra).toEqual({ sourceDims: 2, units: "file" });
        expect(position?.meta.origin?.namespace).toBe("cytoscape");
        expect(Array.from(position?.value(1) as ArrayLike<number>)).toEqual([Math.fround(0.1), 2, 0]);
        expect(position?.isSet(0)).toBe(false);
        expect(s.nodes.byRole("classes")?.value(1)).toEqual(["a", "b"]);
        expect(s.nodes.byRole("classes")?.value(2)).toEqual(["c"]);
        const parent = s.nodes.byRole("parent");
        expect(parent?.dtype).toBe("u32");
        expect(parent?.meta.refersTo).toBe("node");
        expect(parent?.value(1)).toBe(0);
        expect(parent?.isSet(2)).toBe(false);
        expect(cell(s, "nodes", "selected", 0)).toBe(true);
        expect(cell(s, "nodes", "locked", 1)).toBe(false);
        expect(cell(s, "nodes", "custom#element", 1)).toBe(5);
        expect(cell(s, "nodes", "position#data", 1)).toBe("dataPos");
        expect(cell(s, "nodes", "weight", 1)).toBe(4);
        expect(s.edges.byRole("classes")?.value(0)).toEqual(["ec"]);
        expect(cell(s, "edges", "selected", 1)).toBe(true);
        expect(s.edges.byRole("id")?.value(0)).toBe("e1");
        expect(s.edges.byRole("id")?.isSet(1)).toBe(false);
        expect(Array.from(s.edgeList().weights ?? [])).toEqual([3, 1]);
        expect(cell(s, "graph", "title", 0)).toBe("T");
        expect(s.meta.extra.json).toEqual({ dialect: "cytoscape", cytoscape: { zoom: 2, pan: { x: 1, y: 2 } } });
    });

    it("reads a flat elements array and a top-level array, grouping by group or by endpoints", async () => {
        const flat = [
            { group: "nodes", data: { id: "a" } },
            { data: { id: "b" } },
            { data: { id: "e", source: "a", target: "b" } },
            { group: "edges", data: { source: "b", target: "a" } },
        ];
        const nested = await load(json({ elements: flat }));
        expect(nested.s.ids.toArray()).toEqual(["a", "b"]);
        expect(nested.s.edgeCount).toBe(2);
        const bare = await load(json(flat));
        expect(bare.s.ids.toArray()).toEqual(["a", "b"]);
        expect(bare.s.edgeCount).toBe(2);
        expect(bare.s.meta.extra.json).toEqual({ dialect: "cytoscape" });
        const empty = await load("[]");
        expect(empty.s.nodeCount).toBe(0);
    });

    it("defaults to directed and honours defaultDirected", async () => {
        const undirected = await load(json(doc), { defaultDirected: false });
        expect(undirected.s.directed).toBe(false);
    });

    it("reports elements without data, bad positions and bad classes", async () => {
        const bad = {
            elements: {
                nodes: [{ data: { id: "a" }, position: { x: "1", y: 2 }, classes: 5 }, { position: {} }, "x"],
                edges: [{ data: { id: "e" } }, { nope: 1 }],
            },
        };
        const { s, report } = await load(json(bad));
        expect(s.nodeCount).toBe(1);
        expect(s.edgeCount).toBe(0);
        expect(codes(report)).toEqual([
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_VALUE,
            JSON_ISSUE.BAD_ELEMENT,
            JSON_ISSUE.BAD_ELEMENT,
            JSON_ISSUE.MISSING_ENDPOINT,
            JSON_ISSUE.BAD_ELEMENT,
        ]);
        expect(report.counts).toEqual({ nodes: 1, edges: 0, skippedNodes: 2, skippedEdges: 2, expandedMixed: 0 });
    });

    it("declares no position, classes or parent column when no element uses them", async () => {
        const { s } = await load(json({ elements: { nodes: [{ data: { id: "a", position: 1 } }], edges: [] } }));
        expect(names(s, "nodes")).toEqual(["position"]);
        expect(s.nodes.byRole("position")).toBeNull();
        expect(cell(s, "nodes", "position", 0)).toBe(1);
    });

    it("reports a document without elements and fails on elements of the wrong type", async () => {
        const { s, report } = await load('{"data":{"n":1}}', { dialect: "cytoscape" });
        expect(s.nodeCount).toBe(0);
        expect(codes(report)).toEqual([JSON_ISSUE.MISSING_SECTION]);
        const shape = await expectImportError('{"elements": 5}');
        expect(shape.report.issues.at(-1)?.code).toBe(JSON_ISSUE.SHAPE);
        await expectImportError('{"elements": {"nodes": {}}}');
        await expectImportError("[1, 2]");
        await expectImportError('"text"', { dialect: "cytoscape" });
    });

    it("skips an edge that repeats an id and records it, so the sink still freezes", async () => {
        const dup = {
            elements: {
                nodes: [{ data: { id: "a" } }],
                edges: [
                    { data: { id: "e", source: "a", target: "a" } },
                    { data: { id: "e", source: "a", target: "a", extra: 1 } },
                ],
            },
        };
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await jsonImporter.import(json(dup), b);
        expect(codes(report)).toEqual([JSON_ISSUE.DUPLICATE_EDGE_ID]);
        expect(report.counts).toMatchObject({ edges: 1, skippedEdges: 1 });
        const s = b.freeze();
        expect(s.edgeCount).toBe(1);
        expect(s.edges.get("extra")).toBeNull();
    });

    it("reports a node declared twice and merges its attributes", async () => {
        const { s, report } = await load(
            json({
                nodes: [
                    { id: "a", v: 1, w: 1 },
                    { id: "a", v: 2 },
                ],
                links: [],
            }),
        );
        expect(codes(report)).toEqual([JSON_ISSUE.DUPLICATE_NODE]);
        expect(report.issues[0]?.category).toBe("merged");
        expect(report.counts).toMatchObject({ nodes: 1, skippedNodes: 0 });
        expect(s.nodeCount).toBe(1);
        expect([s.nodes.value("v", 0), s.nodes.value("w", 0)]).toEqual([2, 1]);
    });
});

// ============================================================ vis.js

describe("vis", () => {
    it("reads from / to edges with optional ids, undirected by default", async () => {
        const doc = {
            nodes: [{ id: 1, label: "one" }, { id: 2 }],
            edges: [
                { id: "e", from: 1, to: 2, weight: 0.5, arrows: "to" },
                { from: 2, to: 1 },
            ],
        };
        const { s } = await load(json(doc));
        expect(s.directed).toBe(false);
        expect(s.edges.byRole("id")?.value(0)).toBe("e");
        expect(cell(s, "edges", "arrows", 0)).toBe("to");
        expect(s.edges.byRole("weight")?.value(0)).toBe(0.5);
        expect(s.meta.extra.json).toEqual({ dialect: "vis", nodeIdKey: "id", sourceKey: "from", targetKey: "to" });
        const directed = await load(json(doc), { defaultDirected: true });
        expect(directed.s.directed).toBe(true);
    });

    it("reports edges without from or to", async () => {
        const doc = { nodes: [{ id: 1 }], edges: [{ from: 1 }, { to: 1 }, 3] };
        const { s, report } = await load(json(doc), { dialect: "vis" });
        expect(s.edgeCount).toBe(0);
        expect(codes(report)).toEqual([
            JSON_ISSUE.MISSING_ENDPOINT,
            JSON_ISSUE.MISSING_ENDPOINT,
            JSON_ISSUE.BAD_ELEMENT,
        ]);
    });

    it("honours nodeIdKey and the endpoint key options", async () => {
        const doc = { nodes: [{ uid: "a", id: 1 }], edges: [{ s: "a", t: "a" }] };
        const { s } = await load(json(doc), { dialect: "vis", nodeIdKey: "uid", sourceKey: "s", targetKey: "t" });
        expect(s.ids.toArray()).toEqual(["a"]);
        expect(s.edgeCount).toBe(1);
        expect(cell(s, "nodes", "id", 0)).toBe(1);
    });
});

// ============================================================ input handling

describe("input handling", () => {
    it("strips a BOM and reads bytes", async () => {
        const bytes = new TextEncoder().encode(`${String.fromCharCode(0xfeff)}{"nodes":[{"id":"a"}],"links":[]}`);
        const { s } = await load(bytes);
        expect(s.ids.toArray()).toEqual(["a"]);
    });

    it("aborts on invalid UTF-8 with E_INVALID_UTF8", async () => {
        const bytes = new Uint8Array([0x7b, 0xff, 0x7d]);
        const b = new GraphBuilder({ directed: true });
        let caught: unknown;
        try {
            await jsonImporter.import(bytes, b);
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        expect((caught as ImportError).report.issues[0]?.code).toBe(INVALID_UTF8_CODE);
    });

    it("rejects with the abort reason when the signal is already aborted", async () => {
        const controller = new AbortController();
        controller.abort();
        const b = new GraphBuilder({ directed: true });
        await expect(jsonImporter.import('{"nodes":[]}', b, { signal: controller.signal })).rejects.toMatchObject({
            name: "AbortError",
        });
    });

    it("stops between phases when the signal fires during the read", async () => {
        const controller = new AbortController();
        const b = new GraphBuilder({ directed: true });
        const doc = json({ nodes: [{ id: 1 }], links: [{ source: 1, target: 1 }] });
        await expect(
            jsonImporter.import(doc, b, {
                signal: controller.signal,
                onProgress: () => {
                    controller.abort();
                },
            }),
        ).rejects.toMatchObject({ name: "AbortError" });
        expect(b.nodeCount).toBe(1);
        expect(b.edgeCount).toBe(0);
    });

    it("reports progress for in-memory input", async () => {
        const calls: [number, number | undefined][] = [];
        const text = '{"nodes":[],"links":[]}';
        await load(text, { onProgress: (done, total) => calls.push([done, total]) });
        expect(calls).toEqual([[text.length, text.length]]);
    });

    it("stops at the error limit with a truncated report", async () => {
        const doc = { nodes: [{ x: 1 }, { x: 2 }, { x: 3 }, { id: "ok" }], links: [] };
        const error = await expectImportError(json(doc), { errorLimit: 1 });
        expect(error.report.truncated).toBe(true);
        expect(error.report.errorCount).toBe(2);
        expect(error.report.counts.skippedNodes).toBe(1);
        const unlimited = await load(json(doc), { errorLimit: Infinity });
        expect(unlimited.report.errorCount).toBe(3);
        expect(unlimited.s.nodeCount).toBe(1);
    });

    it("fails fast on an empty, invalid or unrecognised document", async () => {
        const empty = await expectImportError("  \n");
        expect(empty.report.issues[0]?.code).toBe(JSON_ISSUE.EMPTY_INPUT);
        const syntax = await expectImportError("{nodes: []}");
        expect(syntax.report.issues[0]?.code).toBe(JSON_ISSUE.SYNTAX);
        expect(syntax.report.issues[0]?.category).toBe("parse-error");
        const scalar = await expectImportError("42");
        expect(scalar.report.issues[0]?.code).toBe(JSON_ISSUE.DIALECT);
        const unknown = await expectImportError('{"vertices": []}');
        expect(unknown.report.issues[0]?.code).toBe(JSON_ISSUE.DIALECT);
        const forcedScalar = await expectImportError("[]", { dialect: "node-link" });
        expect(forcedScalar.report.issues[0]?.code).toBe(JSON_ISSUE.SHAPE);
    });

    it("warns about builder-policy options the sink does not honour", async () => {
        const { report } = await load('{"nodes":[],"links":[]}', {
            addMissingNodes: false,
            duplicateEdges: "sum",
            selfLoops: "keep",
            weightDtype: "f32",
        });
        expect(codes(report)).toEqual([JSON_ISSUE.SINK_OPTION, JSON_ISSUE.SINK_OPTION, JSON_ISSUE.SINK_OPTION]);
        expect(report.issues.map((i) => i.element)).toEqual(["addMissingNodes", "duplicateEdges", "weightDtype"]);
        expect(report.issues.every((i) => i.category === "coercion" && i.severity === "warning")).toBe(true);
        const honoured = await load('{"nodes":[],"links":[]}', { duplicateEdges: "sum" }, { duplicateEdges: "sum" });
        expect(honoured.report.issues).toEqual([]);
    });

    it("warns about nodeIdFrom for the dialects whose ids are unambiguous", async () => {
        const docs: [string, string][] = [
            ['{"elements":{"nodes":[{"data":{"id":"a"}}]}}', "cytoscape"],
            ['{"graph":{"nodes":{"a":{}}}}', "jgf"],
            ['{"nodes":[{"key":"a"}],"edges":[]}', "graphology"],
            ['{"nodes":[{"id":"a"}],"edges":[{"from":"a","to":"a"}]}', "vis"],
        ];
        for (const [text, dialect] of docs) {
            const { s, report } = await load(text, { nodeIdFrom: "label" });
            expect(s.ids.toArray(), dialect).toEqual(["a"]);
            expect(codes(report), dialect).toEqual([JSON_ISSUE.OPTION_IGNORED]);
            expect(report.issues[0]?.message, dialect).toContain(dialect);
        }
    });

    it("counts edges pushed into a non-empty undirected sink that the file expands", async () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("x", "y");
        const report = await jsonImporter.import(
            json({ directed: true, nodes: [], links: [{ source: "a", target: "b" }] }),
            b,
        );
        expect(b.directed).toBe(true);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.counts.edges).toBe(1);
        expect(b.freeze().edgeCount).toBe(3);
    });
});
