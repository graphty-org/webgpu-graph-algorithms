/**
 * Semantic audit of the JSON importer against the corpus files themselves (facts derived here from
 * JSON.parse of the raw text and hand-written counting, never through the importer) and against the
 * quirks graphty-element's JsonDataSource and research note 07 sections 2.7-2.9 describe: the
 * dialect detection order, the id / endpoint key fallbacks, d3 index links, Cytoscape parents and
 * classes, JGF keyed nodes and hyperedges, graphology options, vis from / to. Tests named
 * "DEFECT:" pin a defect and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** The byte-order mark, built from its code so this file stays ASCII. */
const BOM = String.fromCharCode(0xfeff);

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("json", name), { format: "json", ...options });
}

async function parse(doc: unknown, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(typeof doc === "string" ? doc : JSON.stringify(doc), { format: "json", ...options });
}

function dialectOf(snapshot: GraphSnapshot): string | undefined {
    const json = snapshot.meta.extra.json as { dialect?: string } | undefined;
    return json?.dialect;
}

function endpoints(snapshot: GraphSnapshot, e: number): [string | number, string | number] {
    const list = snapshot.edgeList();
    return [snapshot.ids.idOf(list.src[e]), snapshot.ids.idOf(list.dst[e])];
}

function edgeStrings(snapshot: GraphSnapshot): string[] {
    const out: string[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        const [s, t] = endpoints(snapshot, e);
        out.push(`${JSON.stringify(s)}->${JSON.stringify(t)}`);
    }
    return out;
}

describe("JSON corpus facts: miserables.json (d3 v3: name ids, index links, one self-loop)", () => {
    const doc = JSON.parse(readCorpusText("json", "miserables.json")) as {
        nodes: { name: string; group: number }[];
        links: { source: number; target: number; value: number }[];
    };

    it("the raw file has 77 name/group nodes, 254 integer-index links and exactly one self-loop (0,0)", () => {
        expect(Object.keys(doc)).toEqual(["nodes", "links"]);
        expect(doc.nodes).toHaveLength(77);
        expect(doc.links).toHaveLength(254);
        expect(doc.nodes.every((n) => typeof n.name === "string" && !("id" in n))).toBe(true);
        expect(
            doc.links.every(
                (l) => Number.isInteger(l.source) && Number.isInteger(l.target) && l.source < 77 && l.target < 77,
            ),
        ).toBe(true);
        expect(doc.links.filter((l) => l.source === l.target)).toEqual([{ source: 0, target: 0, value: 1 }]);
        expect(doc.nodes[11]).toEqual({ name: "Valjean", group: 2 });
        expect(doc.links[0]).toEqual({ source: 1, target: 0, value: 1 });
    });

    it("imports the d3 dialect with names as string ids and index endpoints resolved by position", async () => {
        const { snapshot, report } = await load("miserables.json");
        expect(report.errorCount).toBe(0);
        expect(dialectOf(snapshot)).toBe("d3");
        expect(snapshot.meta.extra.json).toMatchObject({ edgesKey: "links", nodeIdKey: "name", indexLinks: true });
        expect(snapshot.directed).toBe(false);
        expect(snapshot.nodeCount).toBe(77);
        expect(snapshot.edgeCount).toBe(254);
        expect(snapshot.arcCount).toBe(507);
        expect(snapshot.selfLoopCount).toBe(1);
        expect(snapshot.ids.kind).toBe("string");
        for (let i = 0; i < doc.nodes.length; i++) {
            expect(snapshot.ids.idOf(i)).toBe(doc.nodes[i].name);
            expect(snapshot.nodes.value("group", i)).toBe(doc.nodes[i].group);
        }
        expect(snapshot.nodes.get("group")?.dtype).toBe("i32");
        expect(snapshot.nodes.has("name")).toBe(false);
    });

    it("keeps every link in file order with its value attribute, unweighted (value is not weight)", async () => {
        const { snapshot } = await load("miserables.json");
        const list = snapshot.edgeList();
        for (let e = 0; e < doc.links.length; e++) {
            expect(list.src[e]).toBe(doc.links[e].source);
            expect(list.dst[e]).toBe(doc.links[e].target);
            expect(snapshot.edges.value("value", e)).toBe(doc.links[e].value);
        }
        expect(snapshot.flags.weighted).toBe(false);
        expect(endpoints(snapshot, 0)).toEqual(["Napoleon", "Myriel"]);
        const loop = doc.links.findIndex((l) => l.source === l.target);
        expect(endpoints(snapshot, loop)).toEqual(["Myriel", "Myriel"]);
        expect(snapshot.selfLoopsAt(0)).toBe(1);
        const valjean = doc.links.filter((l) => l.source === 11 || l.target === 11).length;
        expect(valjean).toBe(36);
        expect(snapshot.degree()[11]).toBe(36);
        const weighted = await load("miserables.json", { weightFrom: "value" });
        expect(weighted.snapshot.flags.weighted).toBe(true);
        expect(weighted.snapshot.edgeList().weights?.[1]).toBe(8);
    });
});

describe("JSON corpus facts: sigma-format.json (graphology serialisation)", () => {
    const doc = JSON.parse(readCorpusText("json", "sigma-format.json")) as {
        nodes: { key: string; attributes: { label: string; x: number; y: number; size: number } }[];
        edges: { key: string; source: string; target: string; attributes: { weight: number } }[];
    };

    it("the raw file has key/attributes nodes, keyed edges with weight attributes, one non-integral x", () => {
        expect(doc.nodes.map((n) => n.key)).toEqual(["a", "b", "c", "d", "e"]);
        expect(doc.edges.map((e) => e.key)).toEqual(["e1", "e2", "e3", "e4", "e5"]);
        expect(doc.edges.map((e) => e.attributes.weight)).toEqual([1, 2, 1.5, 1, 0.8]);
        expect(doc.nodes.map((n) => n.attributes.x)).toEqual([0, 1, 2, 1.5, 0.5]);
    });

    it("imports the graphology dialect: directed, keys as edge ids, weights with an f64 shadow, x widened to f64", async () => {
        const { snapshot, freeze } = await load("sigma-format.json");
        expect(dialectOf(snapshot)).toBe("graphology");
        expect(snapshot.directed).toBe(true);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(edgeStrings(snapshot)).toEqual(['"a"->"b"', '"b"->"c"', '"c"->"d"', '"d"->"e"', '"e"->"a"']);
        expect(snapshot.edges.byRole("id")?.meta.name).toBe("key");
        expect([0, 1, 2, 3, 4].map((e) => snapshot.edges.value("key", e))).toEqual(["e1", "e2", "e3", "e4", "e5"]);
        const weight = snapshot.edges.byRole("weight");
        expect(weight?.dtype).toBe("f64");
        expect([0, 1, 2, 3, 4].map((e) => weight?.value(e))).toEqual([1, 2, 1.5, 1, 0.8]);
        expect(snapshot.edgeList().weights?.[4]).toBe(Math.fround(0.8));
        for (let i = 0; i < 5; i++) {
            const attrs = doc.nodes[i].attributes;
            expect(snapshot.nodes.value("label", i)).toBe(attrs.label);
            expect(snapshot.nodes.value("x", i)).toBe(attrs.x);
            expect(snapshot.nodes.value("y", i)).toBe(attrs.y);
            expect(snapshot.nodes.value("size", i)).toBe(attrs.size);
        }
        expect(snapshot.nodes.get("x")?.dtype).toBe("f64");
        expect(snapshot.nodes.get("y")?.dtype).toBe("i32");
        expect(freeze.widened).toEqual([{ column: "x", domain: "node", from: "i32", to: "f64" }]);
    });
});

describe("JSON corpus facts: cytoscape-format.json (elements with data wrappers)", () => {
    const doc = JSON.parse(readCorpusText("json", "cytoscape-format.json")) as {
        elements: {
            nodes: { data: { id: string; label: string; weight: number } }[];
            edges: { data: { id: string; source: string; target: string; weight: number } }[];
        };
    };

    it("imports the cytoscape dialect: directed, edge ids, edge weights, node weight kept as an attribute", async () => {
        expect(doc.elements.nodes.map((n) => n.data.weight)).toEqual([50, 30, 40, 25, 35]);
        expect(doc.elements.edges.map((e) => e.data.weight)).toEqual([1, 2.5, 1.5, 3, 0.5]);
        const { snapshot } = await load("cytoscape-format.json");
        expect(dialectOf(snapshot)).toBe("cytoscape");
        expect(snapshot.directed).toBe(true);
        expect(snapshot.ids.toArray()).toEqual(["n1", "n2", "n3", "n4", "n5"]);
        expect([0, 1, 2, 3, 4].map((e) => snapshot.edges.byRole("id")?.value(e))).toEqual([
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
        ]);
        expect(edgeStrings(snapshot)).toEqual(['"n1"->"n2"', '"n2"->"n3"', '"n3"->"n4"', '"n4"->"n5"', '"n1"->"n3"']);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1, 2.5, 1.5, 3, 0.5]);
        expect(snapshot.nodes.get("weight")?.dtype).toBe("i32");
        expect([0, 1, 2, 3, 4].map((i) => snapshot.nodes.value("weight", i))).toEqual([50, 30, 40, 25, 35]);
        expect(snapshot.nodes.value("label", 0)).toBe("Node 1");
        expect(snapshot.nodes.byRole("parent")).toBeNull();
        expect(snapshot.nodes.byRole("position")).toBeNull();
    });
});

describe("JSON corpus facts: the four small dialect samples", () => {
    it("networkx-format.json: node-link with directed true, multigraph false, an empty graph dict, weights", async () => {
        const doc = JSON.parse(readCorpusText("json", "networkx-format.json")) as {
            directed: boolean;
            multigraph: boolean;
            links: { weight: number }[];
        };
        expect(doc.directed).toBe(true);
        expect(doc.multigraph).toBe(false);
        const { snapshot } = await load("networkx-format.json");
        expect(dialectOf(snapshot)).toBe("node-link");
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta.declaredMultigraph).toBe(false);
        expect(snapshot.ids.toArray()).toEqual(["A", "B", "C", "D", "E"]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual(doc.links.map((l) => l.weight));
        expect(snapshot.graph.names()).toEqual([]);
    });

    it("visjs-format.json: numeric ids kept as numbers, from / to endpoints, undirected, edge labels", async () => {
        const { snapshot } = await load("visjs-format.json");
        expect(dialectOf(snapshot)).toBe("vis");
        expect(snapshot.meta.extra.json).toMatchObject({ sourceKey: "from", targetKey: "to" });
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4, 5]);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.directed).toBe(false);
        expect(edgeStrings(snapshot)).toEqual(["1->2", "2->3", "3->4", "4->5", "5->1"]);
        expect(snapshot.edges.value("label", 4)).toBe("edge 5");
        expect(snapshot.nodes.value("color", 0)).toBe("#FF0000");
    });

    it("d3-format.json: string ids in links (not indices), group and label attributes, value per link", async () => {
        const { snapshot } = await load("d3-format.json");
        expect(dialectOf(snapshot)).toBe("d3");
        expect(snapshot.meta.extra.json).toMatchObject({ nodeIdKey: "id", indexLinks: false });
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(edgeStrings(snapshot)).toEqual(['"a"->"b"', '"b"->"c"', '"c"->"d"', '"d"->"e"', '"a"->"c"']);
        expect([0, 1, 2, 3, 4].map((e) => snapshot.edges.value("value", e))).toEqual([1, 2, 1, 3, 1]);
        expect(snapshot.nodes.value("group", 4)).toBe(3);
    });

    it('karate-d3.json: string ids "1".."34" are kept as strings (ids keep), 78 links by id', async () => {
        const doc = JSON.parse(readCorpusText("json", "karate-d3.json")) as {
            nodes: { id: string; name: string }[];
            links: { source: string; target: string }[];
        };
        expect(doc.nodes).toHaveLength(34);
        expect(doc.links).toHaveLength(78);
        expect(doc.links[0]).toEqual({ source: "2", target: "1" });
        const { snapshot } = await load("karate-d3.json");
        expect(snapshot.ids.kind).toBe("string");
        expect(snapshot.ids.idOf(0)).toBe("1");
        expect(snapshot.ids.has(1)).toBe(false);
        expect(snapshot.edgeCount).toBe(78);
        expect(endpoints(snapshot, 0)).toEqual(["2", "1"]);
        expect(snapshot.nodes.value("name", 33)).toBe("Node 34");
        expect(snapshot.degree()[snapshot.ids.requireIndex("34")]).toBe(17);
        const canonical = await load("karate-d3.json", { ids: "canonical" });
        expect(canonical.snapshot.ids.toArray().slice(0, 3)).toEqual([1, 2, 3]);
    });
});

describe("JSON quirks from JsonDataSource and research note 07", () => {
    it("detects dialects in the documented order: elements, graph.nodes / graphs, options / key / undirected, from / to, links", async () => {
        expect(dialectOf((await parse({ elements: { nodes: [], edges: [] } })).snapshot)).toBe("cytoscape");
        expect(dialectOf((await parse([{ data: { id: "a" } }])).snapshot)).toBe("cytoscape");
        expect(dialectOf((await parse({ graph: { nodes: { a: {} } } })).snapshot)).toBe("jgf");
        expect(dialectOf((await parse({ graphs: [{ nodes: { a: {} } }] })).snapshot)).toBe("jgf");
        expect(
            dialectOf((await parse({ options: { type: "undirected" }, nodes: [{ key: "a" }], edges: [] })).snapshot),
        ).toBe("graphology");
        expect(dialectOf((await parse({ nodes: [{ key: "a" }], edges: [] })).snapshot)).toBe("graphology");
        expect(
            dialectOf(
                (
                    await parse({
                        nodes: [{ id: "a" }, { id: "b" }],
                        edges: [{ source: "a", target: "b", undirected: true }],
                    })
                ).snapshot,
            ),
        ).toBe("graphology");
        expect(dialectOf((await parse({ nodes: [{ id: 1 }, { id: 2 }], edges: [{ from: 1, to: 2 }] })).snapshot)).toBe(
            "vis",
        );
        expect(dialectOf((await parse({ nodes: [{ id: "a" }], links: [] })).snapshot)).toBe("d3");
        expect(dialectOf((await parse({ directed: true, nodes: [{ id: "a" }], links: [] })).snapshot)).toBe(
            "node-link",
        );
        expect(dialectOf((await parse({ nodes: [{ id: "a" }], edges: [] })).snapshot)).toBe("node-link");
    });

    it("accepts id / name / key node identifiers and source|src|from / target|dst|to endpoints", async () => {
        const named = await parse({ nodes: [{ name: "A" }, { name: "B" }], links: [{ source: "B", target: "A" }] });
        expect(named.snapshot.ids.toArray()).toEqual(["A", "B"]);
        expect(edgeStrings(named.snapshot)).toEqual(['"B"->"A"']);
        const srcDst = await parse({ nodes: [{ id: "a" }, { id: "b" }], edges: [{ src: "a", dst: "b" }] });
        expect(edgeStrings(srcDst.snapshot)).toEqual(['"a"->"b"']);
        expect(srcDst.snapshot.meta.extra.json).toMatchObject({ sourceKey: "src", targetKey: "dst" });
        const mixedKeys = await parse({
            nodes: [{ id: "a" }, { id: "b" }],
            edges: [
                { source: "a", target: "b" },
                { from: "b", to: "a" },
            ],
        });
        expect(edgeStrings(mixedKeys.snapshot)).toEqual(['"a"->"b"', '"b"->"a"']);
        const both = await parse({ nodes: [{ id: "a", name: "A" }], links: [] });
        expect(both.snapshot.ids.toArray()).toEqual(["a"]);
        expect(both.snapshot.nodes.value("name", 0)).toBe("A");
    });

    it("does not treat label as an id (JsonDataSource did) but says so: positional ids with a warning, endpoints as errors", async () => {
        const { snapshot, report } = await parse({
            nodes: [{ label: "A" }, { label: "B" }],
            edges: [{ source: "A", target: "B" }],
        });
        expect(report.issues.map((i) => i.code)).toContain("W_POSITIONAL_NODES");
        expect(snapshot.ids.toArray()).toEqual([0, 1]);
        expect(snapshot.edgeCount).toBe(0);
        expect(report.issues.filter((i) => i.code === "E_BAD_INDEX")).toHaveLength(2);
    });

    it('keeps 1 and "1" as distinct ids under keep, refuses true / false / null ids, coerces them only under ids string', async () => {
        const distinct = await parse({ nodes: [{ id: 1 }, { id: "1" }], links: [{ source: 1, target: "1" }] });
        expect(distinct.snapshot.ids.toArray()).toEqual([1, "1"]);
        expect(distinct.snapshot.edgeCount).toBe(1);
        const refused = await parse({
            nodes: [{ id: true }, { id: null }, { id: 1 }],
            links: [{ source: true, target: 1 }],
        });
        expect(refused.snapshot.ids.toArray()).toEqual([1]);
        expect(
            refused.report.issues.filter((i) => i.code === "E_UNSUPPORTED_ID" && i.category === "unsupported"),
        ).toHaveLength(3);
        const coerced = await parse(
            { nodes: [{ id: true }, { id: null }, { id: 1 }], links: [{ source: true, target: 1 }] },
            { ids: "string" },
        );
        expect(coerced.snapshot.ids.toArray()).toEqual(["true", "null", "1"]);
        expect(coerced.snapshot.edgeCount).toBe(1);
        const negativeZero = await parse({ nodes: [{ id: -0 }, { id: 0 }], links: [] });
        expect(negativeZero.snapshot.nodeCount).toBe(1);
        expect(Object.is(negativeZero.snapshot.ids.idOf(0), 0)).toBe(true);
    });

    it("DEFECT: an out-of-range d3 index link must be an error, not silently become new numeric nodes", async () => {
        // With name-only nodes an integer endpoint can only be an array index (note 07: d3 links
        // reference nodes by ARRAY INDEX and must not be coerced). indexLinks "auto" flips to false
        // when one index is out of range, and the integers are then added as nodes 5 and 0.
        const { snapshot, report } = await parse({
            nodes: [{ name: "A" }, { name: "B" }],
            links: [{ source: 5, target: 0 }],
        });
        expect(snapshot.ids.toArray()).toEqual(["A", "B"]);
        expect(snapshot.edgeCount).toBe(0);
        expect(report.issues.map((i) => i.code)).toContain("E_BAD_INDEX");
    });

    it("records multigraph, key attributes and typed graph-dict entries of node-link files", async () => {
        const { snapshot } = await parse({
            directed: true,
            multigraph: true,
            graph: { name: "gn", count: 3, nested: { a: 1 } },
            nodes: [{ id: "a" }, { id: "b" }],
            links: [
                { source: "a", target: "b", key: 0 },
                { source: "a", target: "b", key: 1 },
            ],
        });
        expect(snapshot.meta.declaredMultigraph).toBe(true);
        expect(snapshot.flags.multigraph).toBe(true);
        expect([0, 1].map((e) => snapshot.edges.value("key", e))).toEqual([0, 1]);
        expect(snapshot.graph.value("name", 0)).toBe("gn");
        expect(snapshot.graph.get("count")?.dtype).toBe("i32");
        expect(snapshot.graph.get("nested")?.dtype).toBe("json");
    });

    it("maps nested objects and arrays to json, widens per column, and keeps large integers as f64", async () => {
        const { snapshot } = await parse({
            nodes: [
                { id: "a", pos: { x: 1, y: 2 }, tags: ["x", "y"], v: 1, big: 2147483648 },
                { id: "b", pos: null, v: 1.5 },
                { id: "c", v: "x" },
            ],
            links: [],
        });
        expect(snapshot.nodes.get("pos")?.dtype).toBe("json");
        expect(snapshot.nodes.value("pos", 0)).toEqual({ x: 1, y: 2 });
        expect(snapshot.nodes.isSet("pos", 1)).toBe(false);
        expect(snapshot.nodes.get("tags")?.dtype).toBe("json");
        expect(snapshot.nodes.get("v")?.dtype).toBe("string");
        expect([0, 1, 2].map((i) => snapshot.nodes.value("v", i))).toEqual(["1", "1.5", "x"]);
        expect(snapshot.nodes.get("big")?.dtype).toBe("f64");
    });

    it("reads Cytoscape parents (forward references too), classes, position, flat arrays, and reports an unknown parent", async () => {
        const { snapshot, report } = await parse({
            elements: {
                nodes: [
                    { data: { id: "c", parent: "p" }, position: { x: 1, y: 2 }, classes: "x y" },
                    { data: { id: "p" } },
                    { data: { id: "b", parent: "zz" }, classes: ["m"] },
                ],
                edges: [{ data: { id: "e", source: "c", target: "b", weight: 2 } }],
            },
        });
        expect(snapshot.nodes.byRole("parent")?.value(0)).toBe(1);
        expect(snapshot.nodes.byRole("parent")?.isSet(2)).toBe(false);
        expect(Array.from(snapshot.nodes.byRole("position")?.value(0) as ArrayLike<number>)).toEqual([1, 2, 0]);
        expect(snapshot.nodes.byRole("classes")?.value(0)).toEqual(["x", "y"]);
        expect(snapshot.nodes.byRole("classes")?.value(2)).toEqual(["m"]);
        expect(report.issues.map((i) => i.code)).toContain("E_UNKNOWN_PARENT");
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2]);
        const flat = await parse([
            { data: { id: "a" } },
            { data: { id: "b" } },
            { data: { id: "e", source: "a", target: "b" } },
        ]);
        expect(flat.snapshot.nodeCount).toBe(2);
        expect(flat.snapshot.edgeCount).toBe(1);
    });

    it("reads JGF keyed nodes (JS key order), relation as kind, per-edge directed, label / type / metadata, hyperedges skipped", async () => {
        const { snapshot, report } = await parse({
            graph: {
                directed: false,
                label: "L",
                type: "T",
                metadata: { k: 1 },
                nodes: { "10": { label: "ten" }, "2": { label: "two", metadata: { m: 1 } }, b: {} },
                edges: [
                    { source: "10", target: "2", relation: "r", label: "el", directed: true, metadata: { w: 1 } },
                    { source: "b", target: "2" },
                ],
                hyperedges: [{ nodes: ["10", "2", "b"] }],
            },
        });
        expect(dialectOf(snapshot)).toBe("jgf");
        expect(snapshot.ids.toArray()).toEqual(["2", "10", "b"]);
        expect(snapshot.nodes.value("label", 1)).toBe("ten");
        expect(snapshot.nodes.value("m", 0)).toBe(1);
        expect(snapshot.edges.byRole("kind")?.value(0)).toBe("r");
        expect(snapshot.edges.value("w", 0)).toBe(1);
        expect(snapshot.directed).toBe(true);
        expect(report.counts.expandedMixed).toBe(1);
        expect(snapshot.meta.name).toBe("L");
        expect(snapshot.meta.extra.json).toMatchObject({ type: "T" });
        expect(snapshot.graph.value("k", 0)).toBe(1);
        expect(report.issues.map((i) => i.code)).toContain("W_HYPEREDGES_SKIPPED");
        const graphs = await parse({ graphs: [{ id: "g0", nodes: { a: {} } }, { nodes: { c: {} } }] });
        expect(graphs.report.issues.map((i) => i.code)).toContain("W_MULTIPLE_GRAPHS");
        expect(graphs.snapshot.ids.toArray()).toEqual(["a"]);
    });

    it("reads graphology options / attributes and undirected edges in a mixed graph", async () => {
        const { snapshot } = await parse({
            attributes: { name: "gn" },
            options: { type: "mixed", multi: true, allowSelfLoops: false },
            nodes: [{ key: "a", attributes: { x: 1 } }, { key: "b" }],
            edges: [
                { key: "e1", source: "a", target: "b", undirected: true, attributes: { w: 2 } },
                { source: "b", target: "a" },
            ],
        });
        expect(snapshot.meta.declaredMultigraph).toBe(true);
        expect(snapshot.meta.extra.json).toMatchObject({ allowSelfLoops: false });
        expect(snapshot.graph.value("name", 0)).toBe("gn");
        expect(snapshot.edgeCount).toBe(3);
        expect([0, 1, 2].map((e) => snapshot.edges.byRole("directed")?.value(e))).toEqual([false, false, true]);
        expect(snapshot.edges.byRole("id")?.value(0)).toBe("e1");
        const undirected = await parse({
            options: { type: "undirected" },
            nodes: [{ key: "a" }, { key: "b" }],
            edges: [{ source: "a", target: "b" }],
        });
        expect(undirected.snapshot.directed).toBe(false);
    });

    it("reports missing nodes arrays, missing ids / endpoints, d3 v4 object endpoints and unknown documents explicitly", async () => {
        const noNodes = await parse({ links: [{ source: "a", target: "b" }] });
        expect(noNodes.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(noNodes.report.issues.map((i) => i.code)).toContain("E_MISSING_SECTION");
        const missing = await parse({ nodes: [{ x: 1 }, { id: "b" }], links: [{ source: "b" }] });
        expect(missing.report.issues.map((i) => i.code)).toEqual(
            expect.arrayContaining(["E_MISSING_ID", "E_MISSING_ENDPOINT"]),
        );
        const objects = await parse({
            nodes: [{ id: "a" }, { id: "b" }],
            links: [{ source: { id: "a" }, target: { id: "b" } }],
        });
        expect(objects.snapshot.edgeCount).toBe(0);
        expect(objects.report.issues.filter((i) => i.code === "E_INVALID_ID")).toHaveLength(2);
        await expect(parse({})).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_JSON_DIALECT" }] },
        });
        await expect(parse([1, 2, 3])).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_JSON_DIALECT" }] },
        });
        await expect(parse({ data: { graph: { nodes: [] } } })).rejects.toMatchObject({ code: "E_IMPORT" });
    });

    it("keeps 0.1 and 16777217 weights exact in the f64 shadow and treats a BOM-prefixed document as JSON", async () => {
        const { snapshot } = await parse({
            nodes: [{ id: "a" }, { id: "b" }],
            links: [
                { source: "a", target: "b", weight: 0.1 },
                { source: "b", target: "a", weight: 16777217 },
            ],
        });
        expect([0, 1].map((e) => snapshot.edges.byRole("weight")?.value(e))).toEqual([0.1, 16777217]);
        const bom = await parse(`${BOM}${JSON.stringify({ nodes: [{ id: "a" }], links: [] })}`);
        expect(bom.snapshot.nodeCount).toBe(1);
    });
});
