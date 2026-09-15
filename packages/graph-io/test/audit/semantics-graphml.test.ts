/**
 * Semantic audit of the GraphML importer against the corpus files themselves (facts derived here by
 * regex and a hand-walked element scan over the raw text) and against the quirks graphty-element's
 * GraphMLDataSource and research note 07 section 2.3 describe.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

const NS = 'xmlns="http://graphml.graphdrawing.org/xmlns"';

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("graphml", name), { format: "graphml", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "graphml", ...options });
}

/** Every <key .../> of a document as an attribute map. */
function rawKeys(text: string): Map<string, string>[] {
    return [...text.matchAll(/<key\s([^>]*?)\/>/g)].map((m) => {
        const attrs = new Map<string, string>();
        for (const a of m[1].matchAll(/([\w.]+)="([^"]*)"/g)) {
            attrs.set(a[1], a[2]);
        }
        return attrs;
    });
}

/** Every <node id> ... </node> block with its <data key>text</data> pairs. */
function rawNodes(text: string): { id: string; data: Map<string, string> }[] {
    return [...text.matchAll(/<node id="([^"]*)"(?:\/>|>([\s\S]*?)<\/node>)/g)].map((m) => {
        const data = new Map<string, string>();
        for (const d of (m[2] ?? "").matchAll(/<data key="([^"]*)">([^<]*)<\/data>/g)) {
            data.set(d[1], d[2]);
        }
        return { id: m[1], data };
    });
}

/** Every <edge> block with its attributes and <data key>text</data> pairs. */
function rawEdges(text: string): { id: string | null; source: string; target: string; data: Map<string, string> }[] {
    return [...text.matchAll(/<edge\s([^>]*?)(?:\/>|>([\s\S]*?)<\/edge>)/g)].map((m) => {
        const attrs = new Map<string, string>();
        for (const a of m[1].matchAll(/([\w.]+)="([^"]*)"/g)) {
            attrs.set(a[1], a[2]);
        }
        const data = new Map<string, string>();
        for (const d of (m[2] ?? "").matchAll(/<data key="([^"]*)">([^<]*)<\/data>/g)) {
            data.set(d[1], d[2]);
        }
        return {
            id: attrs.get("id") ?? null,
            source: attrs.get("source") ?? "",
            target: attrs.get("target") ?? "",
            data,
        };
    });
}

function endpoints(snapshot: GraphSnapshot, e: number): [string, string] {
    const list = snapshot.edgeList();
    return [String(snapshot.ids.idOf(list.src[e])), String(snapshot.ids.idOf(list.dst[e]))];
}

describe("GraphML corpus facts: got-network.graphml", () => {
    const text = readCorpusText("graphml", "got-network.graphml");
    const keys = rawKeys(text);
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("the raw file declares three keys, 107 nodes and 352 weighted undirected edges", () => {
        expect(keys.map((k) => [k.get("id"), k.get("attr.name"), k.get("attr.type"), k.get("for")])).toEqual([
            ["label", "label", "string", "node"],
            ["edgelabel", "Edge Label", "string", "edge"],
            ["weight", "weight", "double", "edge"],
        ]);
        expect(nodes).toHaveLength(107);
        expect(edges).toHaveLength(352);
        expect(text).toContain('<graph edgedefault="undirected">');
        expect((text.match(/<graph /g) ?? []).length).toBe(1);
        expect(edges.every((e) => e.data.has("weight"))).toBe(true);
        expect(text).not.toContain('key="edgelabel"');
    });

    it("imports counts, direction, the key declarations and the weight origin", async () => {
        const { snapshot, report } = await load("got-network.graphml");
        expect(snapshot.nodeCount).toBe(107);
        expect(snapshot.edgeCount).toBe(352);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.arcCount).toBe(704);
        expect(report.errorCount).toBe(0);
        expect(report.warningCount).toBe(0);
        const label = snapshot.nodes.get("label");
        expect(label?.meta.role).toBe("label");
        expect(label?.meta.origin?.id).toBe("label");
        expect(label?.meta.origin?.type).toBe("string");
        const edgeLabel = snapshot.edges.get("Edge Label");
        expect(edgeLabel?.meta.origin?.id).toBe("edgelabel");
        expect(edgeLabel?.nullCount).toBe(352);
        expect(snapshot.meta.weightOrigin).toMatchObject({ format: "graphml", id: "weight", type: "double" });
        expect(snapshot.meta.extra.graphml).toMatchObject({ edgedefault: "undirected" });
    });

    it("reads names as string ids in file order with label === id", async () => {
        const { snapshot } = await load("got-network.graphml");
        expect(snapshot.ids.kind).toBe("string");
        for (let i = 0; i < nodes.length; i++) {
            expect(snapshot.ids.idOf(i)).toBe(nodes[i].id);
            expect(snapshot.nodes.value("label", i)).toBe(nodes[i].data.get("label"));
        }
        expect(nodes[50].id).toBe("Sansa");
        expect(snapshot.ids.requireIndex("Sansa")).toBe(50);
    });

    it("reads every edge's endpoints, id and double weight in file order", async () => {
        const { snapshot } = await load("got-network.graphml");
        const list = snapshot.edgeList();
        let sum = 0;
        for (let e = 0; e < edges.length; e++) {
            expect(endpoints(snapshot, e)).toEqual([edges[e].source, edges[e].target]);
            expect(snapshot.edges.value("id", e)).toBe(edges[e].id);
            const w = Number(edges[e].data.get("weight"));
            expect(list.weights?.[e]).toBe(w);
            sum += w;
        }
        expect(sum).toBe(4324);
        expect(edges[100]).toMatchObject({ id: "100", source: "Eddard", target: "Jon" });
        expect(edges[100].data.get("weight")).toBe("8.0");
        expect(list.weights?.[100]).toBe(8);
        expect(edges[351]).toMatchObject({ id: "351", source: "Ygritte", target: "Rattleshirt" });
        expect(list.weights?.[351]).toBe(9);
    });

    it("has no self-loops or parallels, Tyrion has degree 36 and Bran-Hodor is the heaviest edge (96)", async () => {
        const degree = new Map<string, number>();
        const seen = new Set<string>();
        let heaviest = edges[0];
        for (const e of edges) {
            expect(e.source).not.toBe(e.target);
            const key = [e.source, e.target].sort().join("|");
            expect(seen.has(key)).toBe(false);
            seen.add(key);
            degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
            degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
            if (Number(e.data.get("weight")) > Number(heaviest.data.get("weight"))) {
                heaviest = e;
            }
        }
        expect(degree.get("Tyrion")).toBe(36);
        expect(heaviest).toMatchObject({ id: "30", source: "Bran", target: "Hodor" });
        const { snapshot } = await load("got-network.graphml");
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.degree()[snapshot.ids.requireIndex("Tyrion")]).toBe(36);
        const list = snapshot.edgeList();
        expect(list.weights?.[30]).toBe(96);
        expect(endpoints(snapshot, 30)).toEqual(["Bran", "Hodor"]);
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(false);
    });
});

describe("GraphML corpus facts: simple.graphml", () => {
    const text = readCorpusText("graphml", "simple.graphml");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("reads labels through key d0 and weights through key d1; edges have no ids", async () => {
        expect(nodes.map((n) => [n.id, n.data.get("d0")])).toEqual([
            ["n0", "Node A"],
            ["n1", "Node B"],
            ["n2", "Node C"],
            ["n3", "Node D"],
            ["n4", "Node E"],
        ]);
        expect(edges.map((e) => [e.source, e.target, e.data.get("d1")])).toEqual([
            ["n0", "n1", "1.0"],
            ["n0", "n2", "2.0"],
            ["n1", "n3", "1.5"],
            ["n2", "n3", "1.0"],
            ["n3", "n4", "3.0"],
        ]);
        expect(edges.every((e) => e.id === null)).toBe(true);
        const { snapshot } = await load("simple.graphml");
        expect(snapshot.nodeCount).toBe(5);
        expect(snapshot.edgeCount).toBe(5);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.nodes.get("label")?.meta.origin?.id).toBe("d0");
        expect([0, 1, 2, 3, 4].map((i) => snapshot.nodes.value("label", i))).toEqual(
            nodes.map((n) => n.data.get("d0")),
        );
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1, 2, 1.5, 1, 3]);
        expect(snapshot.edges.byRole("id")).toBeNull();
        expect(snapshot.meta.weightOrigin?.id).toBe("d1");
        expect(endpoints(snapshot, 2)).toEqual(["n1", "n3"]);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.selfLoopCount).toBe(0);
    });
});

describe("GraphML corpus facts: yfiles-sample.graphml", () => {
    const text = readCorpusText("graphml", "yfiles-sample.graphml");

    it("keeps the yFiles nested XML as json with the geometry, fill, label and shape of every node", async () => {
        const shapeNodes = [
            ...text.matchAll(
                /<node id="([^"]*)">[\s\S]*?<y:Geometry x="([^"]*)" y="([^"]*)" width="([^"]*)" height="([^"]*)"\/>[\s\S]*?<y:Fill color="([^"]*)"[\s\S]*?<y:NodeLabel>([^<]*)<\/y:NodeLabel>[\s\S]*?<y:Shape type="([^"]*)"\/>/g,
            ),
        ];
        expect(shapeNodes).toHaveLength(4);
        expect(shapeNodes.map((m) => m[7])).toEqual(["Start", "Process A", "Process B", "End"]);
        const { snapshot, report } = await load("yfiles-sample.graphml");
        expect(snapshot.nodeCount).toBe(4);
        expect(snapshot.edgeCount).toBe(4);
        expect(snapshot.directed).toBe(true);
        const d0 = snapshot.nodes.get("d0");
        expect(d0?.dtype).toBe("json");
        expect(d0?.meta.origin?.namespace).toBe("yfiles");
        expect(d0?.meta.origin?.type).toBe("nodegraphics");
        for (let i = 0; i < 4; i++) {
            const m = shapeNodes[i];
            expect(snapshot.ids.idOf(i)).toBe(m[1]);
            const tree = snapshot.nodes.value("d0", i) as {
                "y:ShapeNode": Record<string, Record<string, string> | string>;
            };
            const shape = tree["y:ShapeNode"];
            expect(shape["y:Geometry"]).toEqual({ "@_x": m[2], "@_y": m[3], "@_width": m[4], "@_height": m[5] });
            expect((shape["y:Fill"] as Record<string, string>)["@_color"]).toBe(m[6]);
            expect(shape["y:NodeLabel"]).toBe(m[7]);
            expect((shape["y:Shape"] as Record<string, string>)["@_type"]).toBe(m[8]);
        }
        expect(report.lossy.map((n) => n.code)).toContain("W_GRAPHML_YFILES_JSON");
        expect(snapshot.meta.extra.graphml).toMatchObject({ namespaces: { y: "http://www.yworks.com/xml/graphml" } });
    });

    it("keeps every edge's id, endpoints and PolyLineEdge line style", async () => {
        const polylines = [
            ...text.matchAll(
                /<edge id="([^"]*)" source="([^"]*)" target="([^"]*)">[\s\S]*?<y:LineStyle color="([^"]*)" type="([^"]*)" width="([^"]*)"\/>[\s\S]*?<y:Arrows source="([^"]*)" target="([^"]*)"\/>/g,
            ),
        ];
        expect(polylines).toHaveLength(4);
        const { snapshot } = await load("yfiles-sample.graphml");
        expect(snapshot.edges.get("d1")?.meta.origin?.type).toBe("edgegraphics");
        for (let e = 0; e < 4; e++) {
            const m = polylines[e];
            expect(snapshot.edges.value("id", e)).toBe(m[1]);
            expect(endpoints(snapshot, e)).toEqual([m[2], m[3]]);
            const tree = snapshot.edges.value("d1", e) as { "y:PolyLineEdge": Record<string, Record<string, string>> };
            expect(tree["y:PolyLineEdge"]["y:LineStyle"]).toEqual({ "@_color": m[4], "@_type": m[5], "@_width": m[6] });
            expect(tree["y:PolyLineEdge"]["y:Arrows"]).toEqual({ "@_source": m[7], "@_target": m[8] });
        }
        expect(polylines[2].slice(1, 7)).toEqual(["e2", "n1", "n3", "#FF0000", "line", "2.0"]);
        expect(snapshot.flags.weighted).toBe(false);
    });

    it("skips yFiles keys under yfiles: skip with a warning", async () => {
        const { snapshot, report } = await load("yfiles-sample.graphml", { yfiles: "skip" });
        expect(snapshot.nodes.has("d0")).toBe(false);
        expect(snapshot.edges.has("d1")).toBe(false);
        expect(report.issues.map((i) => i.code)).toContain("W_GRAPHML_YFILES_SKIPPED");
    });
});

describe("GraphML quirks from GraphMLDataSource and research note 07", () => {
    it("applies <default> for unset rows while keeping the row unset", async () => {
        const doc = `<graphml ${NS}><key id="c" for="node" attr.name="color" attr.type="string"><default>yellow</default></key><graph edgedefault="directed"><node id="a"><data key="c">red</data></node><node id="b"/></graph></graphml>`;
        const { snapshot } = await parse(doc);
        const color = snapshot.nodes.require("color");
        expect(color.meta.default).toBe("yellow");
        expect(color.value(0)).toBe("red");
        expect(color.isSet(1)).toBe(false);
        expect(color.value(1)).toBe("yellow");
    });

    it("resolves per-edge directed overrides against edgedefault as a mixed graph", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="undirected"><node id="a"/><node id="b"/><edge source="a" target="b"/><edge source="b" target="a" directed="true"/></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(3);
        expect(report.counts.expandedMixed).toBe(1);
        expect([0, 1, 2].map((e) => snapshot.edges.byRole("directed")?.value(e))).toEqual([false, false, true]);
        expect(snapshot.edges.byRole("pair")?.value(0)).toBe(1);
        await expect(parse(doc, { onMixedDirection: "error" })).rejects.toMatchObject({ code: "E_IMPORT" });
    });

    it("reads a graph without edgedefault as undirected with a warning", async () => {
        const doc = `<graphml ${NS}><graph><node id="a"/><node id="b"/><edge source="a" target="b"/></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.directed).toBe(false);
        expect(report.issues.map((i) => i.code)).toContain("W_GRAPHML_EDGEDEFAULT_MISSING");
    });

    it("maps nested <graph> elements to the parent column and lets edges cross levels", async () => {
        const doc = `<graphml ${NS}><graph id="G" edgedefault="directed"><node id="p"><graph id="p:" edgedefault="directed"><node id="c1"/><node id="c2"/><edge source="c1" target="c2"/></graph></node><node id="x"/><edge source="x" target="c1"/></graph></graphml>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.ids.toArray()).toEqual(["p", "c1", "c2", "x"]);
        const parent = snapshot.nodes.byRole("parent");
        expect(parent?.isSet(0)).toBe(false);
        expect(parent?.value(1)).toBe(0);
        expect(parent?.value(2)).toBe(0);
        expect(parent?.isSet(3)).toBe(false);
        expect(snapshot.edgeCount).toBe(2);
        expect(endpoints(snapshot, 1)).toEqual(["x", "c1"]);
        expect(snapshot.meta.extra.graphml).toMatchObject({ graphId: "G" });
    });

    it("skips hyperedges by default with a warning and expands them under star / clique", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="undirected"><node id="a"/><node id="b"/><node id="c"/><hyperedge><endpoint node="a"/><endpoint node="b"/><endpoint node="c"/></hyperedge></graph></graphml>`;
        const skipped = await parse(doc);
        expect(skipped.snapshot.edgeCount).toBe(0);
        expect(skipped.report.issues.map((i) => i.code)).toContain("W_GRAPHML_HYPEREDGE_SKIPPED");
        const star = await parse(doc, { hyperedges: "star" });
        expect(star.snapshot.nodeCount).toBe(4);
        expect(star.snapshot.edgeCount).toBe(3);
        expect(star.snapshot.nodes.value("graphty.hyperedge", 3)).toBe(true);
        const clique = await parse(doc, { hyperedges: "clique" });
        expect(clique.snapshot.nodeCount).toBe(3);
        expect(clique.snapshot.edgeCount).toBe(3);
    });

    it("keeps sourceport / targetport as port role columns and reports <port> declarations", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="directed"><node id="a"><port name="p1"/></node><node id="b"/><edge source="a" target="b" sourceport="p1" targetport="in"/></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.edges.byRole("sourcePort")?.value(0)).toBe("p1");
        expect(snapshot.edges.byRole("targetPort")?.value(0)).toBe("in");
        expect(report.issues.map((i) => i.code)).toContain("W_GRAPHML_PORT_DECLARATION");
    });

    it('declares a key for="all" in the node, edge and graph tables with the same origin id', async () => {
        const doc = `<graphml ${NS}><key id="k" for="all" attr.name="name" attr.type="string"/><graph edgedefault="directed"><data key="k">gname</data><node id="a"><data key="k">na</data></node><node id="b"/><edge source="a" target="b"><data key="k">ea</data></edge></graph></graphml>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.nodes.get("name")?.meta.origin?.id).toBe("k");
        expect(snapshot.edges.get("name")?.meta.origin?.id).toBe("k");
        expect(snapshot.graph.get("name")?.meta.origin?.id).toBe("k");
        expect(snapshot.nodes.value("name", 0)).toBe("na");
        expect(snapshot.nodes.isSet("name", 1)).toBe(false);
        expect(snapshot.edges.value("name", 0)).toBe("ea");
        expect(snapshot.graph.value("name", 0)).toBe("gname");
    });

    it("reads booleans written as 1 / 0 / True / false (the GraphMLDataSource and NetworkX rule)", async () => {
        const doc = `<graphml ${NS}><key id="b" for="node" attr.name="b" attr.type="boolean"/><graph edgedefault="directed"><node id="a"><data key="b">1</data></node><node id="b"><data key="b">True</data></node><node id="c"><data key="b">false</data></node><node id="d"><data key="b">0</data></node></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(report.errorCount).toBe(0);
        expect([0, 1, 2, 3].map((i) => snapshot.nodes.value("b", i))).toEqual([true, true, false, false]);
    });

    it("maps int / long / float / double exactly and reports long values beyond 2^53", async () => {
        const doc = `<graphml ${NS}><key id="f" for="node" attr.name="f" attr.type="float"/><key id="i" for="node" attr.name="i" attr.type="int"/><key id="d" for="node" attr.name="d" attr.type="double"/><key id="l" for="node" attr.name="l" attr.type="long"/><graph edgedefault="directed"><node id="a"><data key="f">0.1</data><data key="i">42</data><data key="d">0.1</data><data key="l">9007199254740993</data></node></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.nodes.get("f")?.dtype).toBe("f32");
        expect(snapshot.nodes.value("f", 0)).toBe(Math.fround(0.1));
        expect(snapshot.nodes.get("i")?.dtype).toBe("i32");
        expect(snapshot.nodes.get("d")?.dtype).toBe("f64");
        expect(snapshot.nodes.value("d", 0)).toBe(0.1);
        expect(snapshot.nodes.get("l")?.dtype).toBe("f64");
        expect(snapshot.nodes.value("l", 0)).toBe(9007199254740992);
        expect(report.issues.some((i) => i.code === "W_PRECISION")).toBe(true);
        const asString = await parse(doc, { long: "string" });
        expect(asString.snapshot.nodes.get("l")?.dtype).toBe("string");
        expect(asString.snapshot.nodes.value("l", 0)).toBe("9007199254740993");
    });

    it("reports a <data> with an undeclared key, a key used on the wrong element and an unparsable int", async () => {
        const undeclared = `<graphml ${NS}><graph edgedefault="directed"><node id="a"><data key="zz">1</data></node></graph></graphml>`;
        expect((await parse(undeclared)).report.issues.map((i) => i.code)).toContain("E_GRAPHML_UNKNOWN_KEY");
        const wrongDomain = `<graphml ${NS}><key id="e" for="edge" attr.name="e" attr.type="string"/><graph edgedefault="directed"><node id="a"><data key="e">x</data></node></graph></graphml>`;
        expect((await parse(wrongDomain)).report.issues.map((i) => i.code)).toContain("W_GRAPHML_KEY_DOMAIN");
        const badInt = `<graphml ${NS}><key id="i" for="node" attr.name="i" attr.type="int"/><graph edgedefault="directed"><node id="a"><data key="i">abc</data></node></graph></graphml>`;
        const r = await parse(badInt);
        expect(r.report.issues.map((i) => i.code)).toContain("E_COLUMN_TYPE");
        expect(r.snapshot.nodes.isSet("i", 0)).toBe(false);
    });

    it("reads CDATA and entity text, keeps padded strings, and creates an undeclared endpoint by default", async () => {
        const doc = `<graphml ${NS}><key id="d" for="node" attr.name="d" attr.type="string"/><graph edgedefault="directed"><node id="a&amp;b"><data key="d"><![CDATA[a <b> & c]]></data></node><node id="c"><data key="d">  padded  </data></node><edge source="a&amp;b" target="zz"/></graph></graphml>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.ids.toArray()).toEqual(["a&b", "c", "zz"]);
        expect(snapshot.nodes.value("d", 0)).toBe("a <b> & c");
        expect(snapshot.nodes.value("d", 1)).toBe("  padded  ");
        const strict = await parse(doc, { addMissingNodes: false });
        expect(strict.snapshot.nodeCount).toBe(2);
        expect(strict.report.issues.map((i) => i.code)).toContain("E_UNKNOWN_NODE");
    });

    it("reads namespace-prefixed elements and merges multiple top-level graphs with a warning", async () => {
        const prefixed = `<g:graphml xmlns:g="http://graphml.graphdrawing.org/xmlns"><g:graph edgedefault="directed"><g:node id="a"/><g:node id="b"/><g:edge source="a" target="b"/></g:graph></g:graphml>`;
        const r1 = await parse(prefixed);
        expect(r1.snapshot.edgeCount).toBe(1);
        const two = `<graphml ${NS}><graph id="A" edgedefault="directed"><node id="a"/></graph><graph id="B" edgedefault="directed"><node id="b"/></graph></graphml>`;
        const r2 = await parse(two);
        expect(r2.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(r2.report.issues.map((i) => i.code)).toContain("W_MULTIPLE_GRAPHS");
    });

    it("takes the weight from the key whose attr.name is weight, never from a key whose id is weight", async () => {
        const byName = `<graphml ${NS}><key id="w" for="edge" attr.name="weight" attr.type="int"/><graph edgedefault="directed"><node id="a"/><node id="b"/><edge source="a" target="b"><data key="w">3</data></edge></graph></graphml>`;
        expect(Array.from((await parse(byName)).snapshot.edgeList().weights ?? [])).toEqual([3]);
        const byId = `<graphml ${NS}><key id="weight" for="edge" attr.name="cost" attr.type="double"/><graph edgedefault="directed"><node id="a"/><node id="b"/><edge source="a" target="b"><data key="weight">3.5</data></edge></graph></graphml>`;
        const r = await parse(byId);
        expect(r.snapshot.flags.weighted).toBe(false);
        expect(r.snapshot.edges.value("cost", 0)).toBe(3.5);
    });

    it("coerces NMTOKEN ids canonically: 1 becomes a number, 01 stays a string", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="directed"><node id="1"/><node id="01"/><node id="2"/><edge source="1" target="2"/><edge source="01" target="2"/></graph></graphml>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.ids.toArray()).toEqual([1, "01", 2]);
        expect(snapshot.edgeCount).toBe(2);
    });

    it("merges duplicate node declarations with a warning and keeps self-loops and parallels", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="directed"><node id="a"/><node id="a"/><node id="b"/><edge source="a" target="a"/><edge source="a" target="b"/><edge source="a" target="b"/></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.nodeCount).toBe(2);
        expect(report.issues.map((i) => i.code)).toContain("W_DUPLICATE_NODE");
        expect(snapshot.selfLoopCount).toBe(1);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.edgeCount).toBe(3);
    });

    it("keeps edge ids where present as a nullable id column and drops <desc> with a warning", async () => {
        const doc = `<graphml ${NS}><graph edgedefault="directed"><desc>g</desc><node id="a"><desc>nd</desc></node><node id="b"/><edge id="e0" source="a" target="b"/><edge source="b" target="a"/></graph></graphml>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.edges.byRole("id")?.value(0)).toBe("e0");
        expect(snapshot.edges.byRole("id")?.isSet(1)).toBe(false);
        expect(report.issues.map((i) => i.code)).toContain("W_GRAPHML_DESC_DROPPED");
    });

    it("rejects a document that is not UTF-8 with an explicit parse error rather than U+FFFD ids", async () => {
        const latin1 = Uint8Array.from(
            `<?xml version="1.0" encoding="ISO-8859-1"?><graphml ${NS}><graph edgedefault="directed"><node id="caf${String.fromCharCode(0xe9)}"/></graph></graphml>`,
            (c) => c.charCodeAt(0) & 0xff,
        );
        await expect(importGraph(latin1, { format: "graphml" })).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_INVALID_UTF8", category: "parse-error" }] },
        });
    });
});
