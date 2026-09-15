import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import {
    JSON_DIALECTS,
    JSON_LOSS,
    type JsonDialect,
    jsonExporter,
    type JsonExportOptions,
    jsonImporter,
    type JsonImportOptions,
} from "../../../src/formats/json/index.js";
import { type CommonExportOptions, type CommonImportOptions } from "../../../src/types.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { type CompareOptions, compareSnapshots, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

async function imported(text: string, options?: JsonImportOptions & CommonImportOptions): Promise<GraphSnapshot> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await jsonImporter.import(text, b, options);
    return b.freeze();
}

async function exact(
    snapshot: GraphSnapshot,
    exportOptions?: JsonExportOptions & CommonExportOptions,
    importOptions?: JsonImportOptions & CommonImportOptions,
    compare?: CompareOptions,
): Promise<void> {
    const rt = await roundTrip(snapshot, jsonExporter, jsonImporter, { exportOptions, importOptions });
    expect(rt.report.errorCount).toBe(0);
    expectSameSnapshot(snapshot, rt.snapshot, { allowExtraColumns: false, ...compare });
    // the second trip reproduces the first text exactly
    const again = await jsonExporter.exportToString(rt.snapshot, exportOptions);
    expect(again).toBe(rt.text);
}

/** A directed graph with undirected pairs, a mutual pair, defaulted and f64 weights and every dtype. */
function rich(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
    r.setHeader(true);
    b.declareNodeColumn({ name: "f", dtype: "f32" });
    b.declareNodeColumn({ name: "d", dtype: "dict" });
    b.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "string" });
    b.declareNodeColumn({ name: "u", dtype: "u8", components: 2 });
    const a = b.addNode("a");
    const bb = b.addNode("b");
    const c = b.addNode("c");
    b.setNodeValue("f", a, 0.1);
    b.setNodeValue("d", a, "x");
    b.setNodeValue("d", c, "x");
    b.setNodeValue("l", bb, ["p", "q"]);
    b.setNodeValue("u", c, [1, 2]);
    b.setNodeValue("flag", a, true);
    b.setNodeValue("n", bb, 1.5);
    b.setNodeValue("i", c, 7);
    b.setNodeValue("j", a, { k: [1, "two", null] });
    b.setNodeValue("s", c, `unicode ${String.fromCharCode(0xe9)} text`);
    r.addEdge("a", "b", "directed", 0.1);
    r.addEdge("b", "c", "undirected");
    r.addEdge("c", "c", "undirected", 3);
    r.addEdge("a", "c", "mutual", 2);
    r.addEdge("c", "a", "directed");
    b.setEdgeValue("rel", 0, "knows");
    b.setEdgeValue("rel", 1, "likes");
    b.setEdgeValue("num", 3, 4.5);
    b.setGraphValue("name", "rich");
    b.setGraphValue("nested", { deep: { list: [1, 2] } });
    return b.freeze();
}

describe("corpus round trips (design 16.5)", () => {
    for (const entry of corpusFiles("json")) {
        it(`${entry.path}: export in its own dialect and re-import is exact`, async () => {
            const s = await imported(readCorpusText("json", entry.path));
            expect(jsonExporter.check(s)).toEqual([]);
            await exact(s);
            await exact(s, { indent: 4 });
        });
    }

    it("karate-d3.json under ids canonical keeps numeric ids through the trip", async () => {
        const s = await imported(readCorpusText("json", "karate-d3.json"), { ids: "canonical" });
        expect(s.ids.idOf(0)).toBe(1);
        await exact(s, undefined, { ids: "canonical" });
        const asText = await roundTrip(s, jsonExporter, jsonImporter);
        expect(asText.snapshot.ids.idOf(0)).toBe(1);
    });

    it("d3-format.json with weightFrom value re-imports the same weights when told the key", async () => {
        const s = await imported(readCorpusText("json", "d3-format.json"), { weightFrom: "value" });
        await exact(s, undefined, { weightFrom: "value" });
    });

    it("every corpus file survives every dialect on ids, topology, orientation and weights", async () => {
        const mixedCapable = new Set<JsonDialect>(["jgf", "graphology"]);
        for (const entry of corpusFiles("json")) {
            const s = await imported(readCorpusText("json", entry.path));
            for (const dialect of JSON_DIALECTS) {
                const exportOptions: JsonExportOptions & CommonExportOptions = { dialect };
                const importOptions: JsonImportOptions & CommonImportOptions = { dialect };
                if (dialect === "jgf" && s.ids.toArray().some((id) => typeof id === "number")) {
                    // JGF keys nodes by text: numeric ids come back through the canonical rule
                    importOptions.ids = "canonical";
                }
                if (!mixedCapable.has(dialect)) {
                    importOptions.defaultDirected = s.directed;
                }
                const rt = await roundTrip(s, jsonExporter, jsonImporter, { exportOptions, importOptions });
                const label = `${entry.path} via ${dialect}`;
                expect(rt.report.errorCount, label).toBe(0);
                const diffs = compareSnapshots(s, rt.snapshot, {
                    // the edge id column changes role or name across dialects; the values are compared by name
                    roles: false,
                    ignoreColumns: ["id", "key"],
                });
                expect(diffs, label).toEqual([]);
            }
        }
    });
});

describe("synthetic round trips", () => {
    it("jgf and graphology carry mixed direction, mutual pairs as two edges, weights and every dtype", async () => {
        const s = rich();
        for (const dialect of ["jgf", "graphology"] as const) {
            const notes = jsonExporter.check(s, { dialect });
            // JSON declares no types: the f32, dict, list and u8 x2 columns are announced as changing
            expect(
                notes.map((n) => [n.code, n.column]),
                dialect,
            ).toEqual([
                [LOSS.DTYPE, "f"],
                [LOSS.DTYPE, "d"],
                [LOSS.LIST, "l"],
                [LOSS.DTYPE, "u"],
                [LOSS.COMPONENTS, "u"],
                [JSON_LOSS.MUTUAL_EXPANDED, null],
            ]);
            const rt = await roundTrip(s, jsonExporter, jsonImporter, { exportOptions: { dialect } });
            expect(rt.report.errorCount, dialect).toBe(0);
            // the mutual pair comes back as two plain directed edges (its pair / mutual marks are the one
            // loss); JSON declares no types, so f32 / dict / list columns come back as f64 / string / json
            const diffs = compareSnapshots(s, rt.snapshot, {
                ignoreRoles: ["pair", "mutual", "directed"],
                dtypes: false,
                tolerance: 1e-7,
            });
            expect(diffs, dialect).toEqual([]);
            expect(rt.snapshot.nodes.require("f").dtype, dialect).toBe("f64");
            expect(rt.snapshot.nodes.require("d").dtype, dialect).toBe("string");
            expect(rt.snapshot.nodes.require("l").dtype, dialect).toBe("json");
            expect(rt.snapshot.nodes.require("u").dtype, dialect).toBe("json");
            expect(rt.snapshot.edges.byRole("weight")?.value(0), dialect).toBe(0.1);
            expect(rt.snapshot.edges.byRole("weight")?.isSet(1), dialect).toBe(false);
            expect(rt.snapshot.edges.byRole("pair")?.isSet(3), dialect).toBe(false);
            expect(rt.snapshot.edges.byRole("mutual"), dialect).toBeNull();
        }
    });

    it("node-link folds the undirected pair to its primary under onMixedDirection directed", async () => {
        const s = rich();
        const rt = await roundTrip(s, jsonExporter, jsonImporter, { exportOptions: { onMixedDirection: "directed" } });
        expect(rt.snapshot.directed).toBe(true);
        // the b -- c pair is written as one directed edge b -> c (design section 8.5: pairs fold back)
        expect(rt.snapshot.edgeCount).toBe(s.edgeCount - 1);
        const list = rt.snapshot.edgeList();
        const pairs = Array.from(
            list.src,
            (u, e) => `${String(rt.snapshot.ids.idOf(u))}>${String(rt.snapshot.ids.idOf(list.dst[e]))}`,
        );
        expect(pairs).toEqual(["a>b", "b>c", "c>c", "a>c", "c>a", "c>a"]);
        expect(rt.snapshot.edges.byRole("pair")).toBeNull();
        expect(rt.snapshot.edges.require("rel").value(1)).toBe("likes");
        expect(rt.snapshot.edges.byRole("weight")?.value(2)).toBe(3);
        expect(rt.snapshot.nodes.require("s").value(2)).toBe(s.nodes.require("s").value(2));
    });

    it("a plain undirected weighted graph is exact in every dialect that keeps direction", async () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("x", "y", 0.25);
        b.addEdge("y", "z");
        b.addEdge("z", "z", 1e-7);
        b.setNodeValue("label", 0, "X");
        b.setEdgeValue("note", 1, "n");
        const s = b.freeze();
        for (const dialect of ["node-link", "d3", "jgf", "graphology"] as const) {
            await exact(s, { dialect }, { dialect });
        }
        for (const dialect of ["cytoscape", "vis"] as const) {
            const rt = await roundTrip(s, jsonExporter, jsonImporter, {
                exportOptions: { dialect },
                importOptions: { defaultDirected: false },
            });
            expect(compareSnapshots(s, rt.snapshot, { ignoreColumns: ["id"] }), dialect).toEqual([]);
        }
    });

    it("defaulted weights stay defaulted and explicit ones explicit", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c");
        b.addEdge("c", "a", 1);
        const s = b.freeze();
        const shadow = s.edges.byRole("weight");
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.isSet(2)).toBe(true);
        for (const dialect of JSON_DIALECTS) {
            const rt = await roundTrip(s, jsonExporter, jsonImporter, {
                exportOptions: { dialect },
                importOptions: { defaultDirected: true },
            });
            const back = rt.snapshot.edges.byRole("weight");
            expect(back?.isSet(1), dialect).toBe(false);
            expect(back?.isSet(2), dialect).toBe(true);
            expect(Array.from(rt.snapshot.edgeList().weights ?? []), dialect).toEqual([2, 1, 1]);
        }
    });

    it("Cytoscape keeps positions, classes, parents and element keys", async () => {
        const text = JSON.stringify({
            elements: {
                nodes: [
                    { data: { id: "p" }, selectable: false },
                    {
                        data: { id: "n", parent: "p", k: 1 },
                        position: { x: 0.1, y: -2.5 },
                        classes: "a b",
                        locked: true,
                    },
                ],
                edges: [{ data: { id: "e", source: "n", target: "p", weight: 2, k: "v" }, classes: "x" }],
            },
        });
        const s = await imported(text);
        await exact(s);
        await exact(s, { indent: 2 });
    });

    it("graphology and JGF keep suffixed collisions and element-level keys", async () => {
        const text = JSON.stringify({
            options: { type: "mixed", multi: false, allowSelfLoops: true },
            nodes: [{ key: "a", attributes: { key: "inner" }, level: 2 }],
            edges: [{ key: "e", source: "a", target: "a", undirected: true, attributes: { key: "k" }, flag: true }],
        });
        const s = await imported(text);
        expect(s.nodes.names()).toEqual(["key", "level#element"]);
        expect(s.edges.names()).toContain("key#data");
        await exact(s);
        const jgf = JSON.stringify({
            graph: {
                directed: true,
                nodes: { A: { label: "a", metadata: { label: "inner" }, colour: "red" } },
                edges: [
                    {
                        source: "A",
                        target: "A",
                        directed: false,
                        metadata: { relation: "m" },
                        relation: "r",
                        weight: 9,
                    },
                ],
            },
        });
        const j = await imported(jgf);
        expect(j.nodes.names()).toEqual(["label", "label#data", "colour#element"]);
        expect(j.edges.names()).toContain("relation#data");
        expect(j.edges.names()).toContain("weight#element");
        await exact(j);
    });

    it("keeps a snapshot from another source through node-link with its column set", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({
            name: "score",
            dtype: "f64",
            origin: { format: "gexf", id: "0", title: null, type: "double", namespace: null },
        });
        b.setNodeValue("score", b.addNode("n1"), 0.75);
        b.addEdge("n1", "n2", 0.3);
        const s = b.freeze();
        expect(s.meta.extra).toEqual({});
        await exact(s, undefined, undefined, { originType: false });
    });
});
