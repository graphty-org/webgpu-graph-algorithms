/**
 * Semantic audit of the GEXF importer against the corpus files themselves (facts derived here by
 * regex over the raw text, never through the importer) and against the quirks graphty-element's
 * GEXFDataSource and research note 07 section 2.1 describe. Tests that pin a defect are named
 * "DEFECT:" and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { OPEN_END, OPEN_START } from "../../src/formats/gexf/schema.js";
import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

const NS13 = 'xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3"';
const NS12 = 'xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2"';

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("gexf", name), { format: "gexf", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "gexf", ...options });
}

/** Every `<edge .../>` of a GEXF text, in file order, by hand-walked attribute parsing. */
function rawEdges(text: string): { id: string; source: string; target: string; weight: string | null }[] {
    const out: { id: string; source: string; target: string; weight: string | null }[] = [];
    for (const match of text.matchAll(/<edge\s([^>]*?)\/?>/g)) {
        const attrs = new Map<string, string>();
        for (const attr of match[1].matchAll(/([a-z]+)="([^"]*)"/g)) {
            attrs.set(attr[1], attr[2]);
        }
        out.push({
            id: attrs.get("id") ?? "",
            source: attrs.get("source") ?? "",
            target: attrs.get("target") ?? "",
            weight: attrs.get("weight") ?? null,
        });
    }
    return out;
}

/** Every `<node id label>` start tag of a GEXF text, in file order. */
function rawNodes(text: string): { id: string; label: string }[] {
    return [...text.matchAll(/<node id="([^"]*)" label="([^"]*)"/g)].map((m) => ({ id: m[1], label: m[2] }));
}

function edgeEndpoints(snapshot: GraphSnapshot, e: number): [string, string] {
    const list = snapshot.edgeList();
    return [String(snapshot.ids.idOf(list.src[e])), String(snapshot.ids.idOf(list.dst[e]))];
}

function degreeOf(snapshot: GraphSnapshot, id: string | number): number {
    return snapshot.degree()[snapshot.ids.requireIndex(id)];
}

describe("GEXF corpus facts: lesmiserables.gexf (Gephi 0.7, GEXF 1.1)", () => {
    const text = readCorpusText("gexf", "lesmiserables.gexf");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("the raw file has 77 nodes and 254 edges, 97 of them without a weight attribute", () => {
        expect(nodes).toHaveLength(77);
        expect(edges).toHaveLength(254);
        expect(edges.filter((e) => e.weight === null)).toHaveLength(97);
        expect(text).toContain('<nodes count="77">');
    });

    it("imports the counts, the undirected default and the header metadata", async () => {
        const { snapshot, report } = await load("lesmiserables.gexf");
        expect(snapshot.nodeCount).toBe(77);
        expect(snapshot.edgeCount).toBe(254);
        expect(snapshot.arcCount).toBe(508);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.meta.sourceFormat).toBe("gexf");
        expect(snapshot.meta.sourceVersion).toBe("1.1");
        expect(snapshot.meta.creator).toBe("Gephi 0.7");
        expect(snapshot.meta.modified).toBe("2010-03-03+23:44");
        expect(snapshot.meta.idType).toBe("string");
        expect(report.errorCount).toBe(0);
        expect(report.counts).toMatchObject({ nodes: 77, edges: 254, skippedEdges: 0, expandedMixed: 0 });
    });

    it("keeps the float-looking ids as strings, in file order, with their labels", async () => {
        const { snapshot } = await load("lesmiserables.gexf");
        expect(snapshot.ids.kind).toBe("string");
        for (let i = 0; i < nodes.length; i++) {
            expect(snapshot.ids.idOf(i)).toBe(nodes[i].id);
            expect(snapshot.nodes.value("label", i)).toBe(nodes[i].label);
        }
        expect(snapshot.ids.idOf(11)).toBe("11.0");
        expect(snapshot.nodes.value("label", 11)).toBe("Valjean");
        expect(snapshot.ids.has(11)).toBe(false);
        expect(snapshot.nodes.get("label")?.meta.role).toBe("label");
    });

    it("keeps every edge in file order with its id and endpoints", async () => {
        const { snapshot } = await load("lesmiserables.gexf");
        for (let e = 0; e < edges.length; e++) {
            expect(edgeEndpoints(snapshot, e)).toEqual([edges[e].source, edges[e].target]);
            expect(snapshot.edges.value("id", e)).toBe(edges[e].id);
        }
        expect(edges[253]).toEqual({ id: "249", source: "76.0", target: "66.0", weight: null });
        expect(snapshot.edges.get("id")?.meta.role).toBe("id");
    });

    it("reads explicit weights exactly and leaves the 97 unweighted edges unset (weight 1 on the arcs)", async () => {
        const { snapshot } = await load("lesmiserables.gexf");
        const weightColumn = snapshot.edges.byRole("weight");
        expect(weightColumn).not.toBeNull();
        expect(weightColumn?.nullCount).toBe(97);
        const list = snapshot.edgeList();
        let sum = 0;
        for (let e = 0; e < edges.length; e++) {
            const raw = edges[e].weight;
            if (raw === null) {
                expect(weightColumn?.isSet(e)).toBe(false);
                expect(list.weights?.[e]).toBe(1);
            } else {
                expect(weightColumn?.isSet(e)).toBe(true);
                expect(weightColumn?.value(e)).toBe(Number(raw));
                expect(list.weights?.[e]).toBe(Number(raw));
                sum += Number(raw);
            }
        }
        expect(sum).toBe(723);
        expect(edges[1]).toEqual({ id: "1", source: "2.0", target: "0.0", weight: "8.0" });
        expect(snapshot.edges.value("graphty.weight", 1)).toBe(8);
    });

    it("has no self-loops or parallel edges, and Valjean has degree 36", async () => {
        const seen = new Set<string>();
        for (const e of edges) {
            expect(e.source).not.toBe(e.target);
            const key = [e.source, e.target].sort().join("|");
            expect(seen.has(key)).toBe(false);
            seen.add(key);
        }
        const valjean = edges.filter((e) => e.source === "11.0" || e.target === "11.0").length;
        expect(valjean).toBe(36);
        const { snapshot } = await load("lesmiserables.gexf");
        expect(degreeOf(snapshot, "11.0")).toBe(36);
    });
});

describe("GEXF corpus facts: airlines-sample.gexf (declared attributes, viz:color)", () => {
    const text = readCorpusText("gexf", "airlines-sample.gexf");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);
    const declared = [...text.matchAll(/<attribute id="([^"]*)" title="([^"]*)" type="([^"]*)"/g)].map((m) => ({
        id: m[1],
        title: m[2],
        type: m[3],
    }));
    const blocks = [...text.matchAll(/<node id="([^"]*)" label="([^"]*)">([\s\S]*?)<\/node>/g)].map((m) => {
        const values = new Map<string, string>();
        for (const v of m[3].matchAll(/<attvalue for="([^"]*)" value="([^"]*)"\/>/g)) {
            values.set(v[1], v[2]);
        }
        const color = /<viz:color b="(\d+)" g="(\d+)" r="(\d+)"\/>/.exec(m[3]);
        return {
            id: m[1],
            label: m[2],
            values,
            color: color === null ? null : { r: Number(color[3]), g: Number(color[2]), b: Number(color[1]) },
        };
    });

    it("the raw file declares four node attributes and 235 nodes / 1297 edges without weights", () => {
        expect(declared).toEqual([
            { id: "code", title: "Code", type: "string" },
            { id: "city", title: "City", type: "string" },
            { id: "latitude", title: "latitude", type: "double" },
            { id: "longitude", title: "longitude", type: "double" },
        ]);
        expect(nodes).toHaveLength(235);
        expect(blocks).toHaveLength(235);
        expect(edges).toHaveLength(1297);
        expect(edges.every((e) => e.weight === null)).toBe(true);
        expect(text).toContain('<edges count="1297">');
    });

    it("declares the columns by title with the attribute id and declared type in origin", async () => {
        const { snapshot } = await load("airlines-sample.gexf");
        for (const attr of declared) {
            const column = snapshot.nodes.get(attr.title);
            expect(column, attr.title).not.toBeNull();
            expect(column?.meta.origin?.id).toBe(attr.id);
            expect(column?.meta.origin?.type).toBe(attr.type);
            expect(column?.dtype).toBe(attr.type === "double" ? "f64" : "string");
            expect(column?.nullCount).toBe(0);
        }
        expect(snapshot.nodes.has("code")).toBe(false);
    });

    it("reads node 0 exactly: id, label, the four values and the viz colour", async () => {
        const { snapshot } = await load("airlines-sample.gexf");
        expect(blocks[0].label).toBe("Adams Field Airport");
        expect(blocks[0].values.get("city")).toBe("Little Rock, AR");
        expect(blocks[0].color).toEqual({ r: 88, g: 107, b: 243 });
        expect(snapshot.ids.idOf(0)).toBe(0);
        expect(snapshot.nodes.value("label", 0)).toBe("Adams Field Airport");
        expect(snapshot.nodes.value("Code", 0)).toBe("LIT");
        expect(snapshot.nodes.value("City", 0)).toBe("Little Rock, AR");
        expect(snapshot.nodes.value("latitude", 0)).toBe(34.729444);
        expect(snapshot.nodes.value("longitude", 0)).toBe(-92.224444);
        const color = snapshot.nodes.value("color", 0) as ArrayLike<number>;
        expect(Array.from(color)).toEqual([88 / 255, 107 / 255, 243 / 255, 1].map(Math.fround));
        expect(snapshot.nodes.get("color")?.meta.role).toBe("color");
        expect(snapshot.nodes.get("color")?.meta.origin?.namespace).toBe("viz");
    });

    it("reproduces every node's attribute values and colour", async () => {
        const { snapshot } = await load("airlines-sample.gexf");
        const colorColumn = snapshot.nodes.get("color");
        expect(colorColumn?.nullCount).toBe(0);
        for (let i = 0; i < blocks.length; i++) {
            const block = blocks[i];
            expect(snapshot.ids.idOf(i)).toBe(Number(block.id));
            expect(snapshot.nodes.value("Code", i)).toBe(block.values.get("code"));
            expect(snapshot.nodes.value("City", i)).toBe(block.values.get("city"));
            expect(snapshot.nodes.value("latitude", i)).toBe(Number(block.values.get("latitude")));
            expect(snapshot.nodes.value("longitude", i)).toBe(Number(block.values.get("longitude")));
            expect(block.color).not.toBeNull();
            const color = Array.from(snapshot.nodes.value("color", i) as ArrayLike<number>);
            expect(color[0]).toBe(Math.fround((block.color?.r ?? 0) / 255));
            expect(color[1]).toBe(Math.fround((block.color?.g ?? 0) / 255));
            expect(color[2]).toBe(Math.fround((block.color?.b ?? 0) / 255));
            expect(color[3]).toBe(1);
        }
        const withComma = blocks.filter((b) => (b.values.get("city") ?? "").includes(",")).length;
        expect(withComma).toBe(67);
        const codes = new Set(blocks.map((b) => b.values.get("code")));
        expect(codes.size).toBe(235);
    });

    it("keeps the double-encoded label of node 119 byte for byte", async () => {
        const { snapshot } = await load("airlines-sample.gexf");
        const mojibake = `Louis Armstrong New Orl${String.fromCharCode(0xc3, 0xa9)}ans International Airport`;
        expect(blocks[119].label).toBe(mojibake);
        expect(snapshot.nodes.value("label", 119)).toBe(mojibake);
    });

    it("reads the edges in order, unweighted, with the ids as strings and endpoints as numbers", async () => {
        const { snapshot } = await load("airlines-sample.gexf");
        expect(snapshot.flags.weighted).toBe(false);
        expect(snapshot.edges.byRole("weight")).toBeNull();
        expect(snapshot.ids.kind).toBe("identity");
        for (let e = 0; e < edges.length; e++) {
            const list = snapshot.edgeList();
            expect(list.src[e]).toBe(Number(edges[e].source));
            expect(list.dst[e]).toBe(Number(edges[e].target));
            expect(snapshot.edges.value("id", e)).toBe(edges[e].id);
        }
        expect(edges[1296]).toEqual({ id: "1296", source: "234", target: "164", weight: null });
    });

    it("has no self-loops or parallels; node 136 has the highest degree (130); Bellingham is northernmost", async () => {
        const degree = new Map<string, number>();
        const seen = new Set<string>();
        for (const e of edges) {
            expect(e.source).not.toBe(e.target);
            const key = [e.source, e.target].sort().join("|");
            expect(seen.has(key)).toBe(false);
            seen.add(key);
            degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
            degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
        }
        expect(degree.get("136")).toBe(130);
        expect(Math.max(...degree.values())).toBe(130);
        let north = blocks[0];
        for (const b of blocks) {
            if (Number(b.values.get("latitude")) > Number(north.values.get("latitude"))) {
                north = b;
            }
        }
        expect(north.label).toBe("Bellingham");
        const { snapshot } = await load("airlines-sample.gexf");
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(degreeOf(snapshot, 136)).toBe(130);
        let maxDegree = 0;
        const degrees = snapshot.degree();
        for (let i = 0; i < snapshot.nodeCount; i++) {
            maxDegree = Math.max(maxDegree, degrees[i]);
        }
        expect(maxDegree).toBe(130);
        const lat = snapshot.nodes.require("latitude");
        let best = 0;
        for (let i = 1; i < snapshot.nodeCount; i++) {
            if ((lat.value(i) as number) > (lat.value(best) as number)) {
                best = i;
            }
        }
        expect(snapshot.nodes.value("label", best)).toBe("Bellingham");
        expect(lat.value(best)).toBe(48.8);
    });
});

describe("GEXF corpus facts: minimal.gexf (GEXF 1.2, meta block)", () => {
    it("reads ids as canonical numbers, labels, two undirected edges and the meta block", async () => {
        const text = readCorpusText("gexf", "minimal.gexf");
        expect(rawNodes(text)).toEqual([
            { id: "1", label: "A" },
            { id: "2", label: "B" },
            { id: "3", label: "C" },
        ]);
        expect(rawEdges(text)).toEqual([
            { id: "0", source: "1", target: "2", weight: null },
            { id: "1", source: "2", target: "3", weight: null },
        ]);
        const { snapshot } = await load("minimal.gexf");
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.edgeCount).toBe(2);
        expect(edgeEndpoints(snapshot, 0)).toEqual(["1", "2"]);
        expect(edgeEndpoints(snapshot, 1)).toEqual(["2", "3"]);
        expect([0, 1, 2].map((i) => snapshot.nodes.value("label", i))).toEqual(["A", "B", "C"]);
        expect(snapshot.meta.sourceVersion).toBe("1.2");
        expect(snapshot.meta.creator).toBe("Graphty Test Corpus");
        expect(snapshot.meta.description).toBe("Minimal GEXF test file");
        expect(snapshot.meta.modified).toBe("2025-01-01");
        expect(snapshot.meta.idType).toBeNull();
    });
});

describe("GEXF quirks from GEXFDataSource and research note 07", () => {
    it("defaults to undirected when defaultedgetype is absent and honours defaultDirected", async () => {
        const doc = `<gexf ${NS13}><graph><nodes><node id="a"/><node id="b"/></nodes><edges><edge source="a" target="b"/></edges></graph></gexf>`;
        expect((await parse(doc)).snapshot.directed).toBe(false);
        expect((await parse(doc, { defaultDirected: true })).snapshot.directed).toBe(true);
    });

    it("resolves per-edge type overrides and mutual through the mixed-direction rules", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="undirected"><nodes><node id="a"/><node id="b"/><node id="c"/></nodes><edges><edge id="0" source="a" target="b"/><edge id="1" source="b" target="c" type="directed"/><edge id="2" source="c" target="a" type="mutual"/></edges></graph></gexf>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(5);
        expect(report.counts.expandedMixed).toBe(2);
        const directed = snapshot.edges.byRole("directed");
        const pair = snapshot.edges.byRole("pair");
        const mutual = snapshot.edges.byRole("mutual");
        expect(directed?.value(0)).toBe(false);
        expect(pair?.value(0)).toBe(1);
        expect(directed?.value(2)).toBe(true);
        expect(mutual?.value(3)).toBe(true);
        expect(pair?.value(3)).toBe(4);
    });

    it("keeps GEXF 1.3 kind-distinguished parallel edges as a multigraph with a kind column", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"/><node id="b"/></nodes><edges><edge source="a" target="b" kind="friend"/><edge source="a" target="b" kind="foe"/></edges></graph></gexf>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.edges.get("kind")?.meta.role).toBe("kind");
        expect([0, 1].map((e) => snapshot.edges.value("kind", e))).toEqual(["friend", "foe"]);
    });

    it("reads boolean attvalues written as 1 / 0 (the GEXFDataSource rule) and true / false", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="b" type="boolean"/></attributes><nodes><node id="a"><attvalues><attvalue for="0" value="1"/></attvalues></node><node id="b"><attvalues><attvalue for="0" value="0"/></attvalues></node><node id="c"><attvalues><attvalue for="0" value="true"/></attvalues></node><node id="d"><attvalues><attvalue for="0" value="false"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.nodes.get("b")?.dtype).toBe("bool");
        expect([0, 1, 2, 3].map((i) => snapshot.nodes.value("b", i))).toEqual([true, false, true, false]);
    });

    it("applies <default> to unset rows, records <options>, and marks the rows unset", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="grade" type="string"><default>red</default><options>[red, green, blue]</options></attribute><attribute id="1" title="n" type="integer"><default>5</default></attribute></attributes><nodes><node id="a"><attvalues><attvalue for="0" value="green"/></attvalues></node><node id="b"/></nodes></graph></gexf>`;
        const { snapshot } = await parse(doc);
        const grade = snapshot.nodes.require("grade");
        expect(grade.meta.default).toBe("red");
        expect(grade.meta.options).toEqual(["red", "green", "blue"]);
        expect(grade.value(0)).toBe("green");
        expect(grade.isSet(1)).toBe(false);
        expect(grade.value(1)).toBe("red");
        const n = snapshot.nodes.require("n");
        expect(n.dtype).toBe("i32");
        expect(n.meta.default).toBe(5);
        expect(n.isSet(0)).toBe(false);
        expect(n.value(0)).toBe(5);
    });

    it("reads 1.2 liststring with pipe, comma and semicolon separators and 1.3 bracket lists", async () => {
        const v12 = `<gexf ${NS12}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="tags" type="liststring"/></attributes><nodes><node id="a"><attvalues><attvalue for="0" value="x|y|z"/></attvalues></node><node id="b"><attvalues><attvalue for="0" value="p,q"/></attvalues></node><node id="c"><attvalues><attvalue for="0" value="m;n"/></attvalues></node></nodes></graph></gexf>`;
        const r12 = await parse(v12);
        expect(r12.snapshot.nodes.get("tags")?.dtype).toBe("list");
        expect(r12.snapshot.nodes.get("tags")?.meta.origin?.type).toBe("liststring");
        expect([0, 1, 2].map((i) => r12.snapshot.nodes.value("tags", i))).toEqual([
            ["x", "y", "z"],
            ["p", "q"],
            ["m", "n"],
        ]);
        const v13 = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="tags" type="liststring"/><attribute id="1" title="nums" type="listinteger"/><attribute id="2" title="ds" type="listdouble"/></attributes><nodes><node id="a"><attvalues><attvalue for="0" value="[x, 'y z', &quot;q, r&quot;]"/><attvalue for="1" value="[1, 2, 3]"/><attvalue for="2" value="[0.5, 1.5]"/></attvalues></node></nodes></graph></gexf>`;
        const r13 = await parse(v13);
        expect(r13.snapshot.nodes.value("tags", 0)).toEqual(["x", "y z", "q, r"]);
        expect(r13.snapshot.nodes.get("nums")?.meta.itemDtype).toBe("i32");
        expect(r13.snapshot.nodes.value("nums", 0)).toEqual([1, 2, 3]);
        expect(r13.snapshot.nodes.get("ds")?.meta.itemDtype).toBe("f64");
        expect(r13.snapshot.nodes.value("ds", 0)).toEqual([0.5, 1.5]);
    });

    it("resolves pid forward references, nested <nodes> and <parents>, and reports an unknown pid", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="c1" pid="p"/><node id="p"><nodes><node id="c2"/></nodes></node><node id="c3"><parents><parent for="p"/><parent for="c1"/></parents></node><node id="orphan" pid="zz"/></nodes></graph></gexf>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.ids.toArray()).toEqual(["c1", "p", "c2", "c3", "orphan"]);
        const parent = snapshot.nodes.byRole("parent");
        expect(parent?.value(0)).toBe(1);
        expect(parent?.isSet(1)).toBe(false);
        expect(parent?.value(2)).toBe(1);
        expect(snapshot.nodes.byRole("parents")?.value(3)).toEqual([1, 0]);
        expect(parent?.isSet(4)).toBe(false);
        expect(report.issues.map((i) => i.code)).toContain("E_UNKNOWN_PARENT");
    });

    it("reads start / end / timestamp, <spells>, 1.3 timestamps and intervals into the temporal roles", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed" mode="dynamic" timeformat="integer"><nodes><node id="a" start="1" end="5"/><node id="b"><spells><spell start="1" end="2"/><spell start="4"/></spells></node><node id="c" timestamp="3"/><node id="d" timestamps="[1, 2, 5]"/><node id="e" intervals="[1, 3];[5, 8]"/></nodes><edges><edge source="a" target="b" start="2" end="3"/></edges></graph></gexf>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.meta.mode).toBe("dynamic");
        expect(snapshot.meta.timeFormat).toBe("integer");
        expect(snapshot.nodes.byRole("start")?.value(0)).toBe(1);
        expect(snapshot.nodes.byRole("end")?.value(0)).toBe(5);
        const spells = snapshot.nodes.byRole("spells");
        expect(spells?.meta.itemComponents).toBe(2);
        const spellsB = spells?.value(1) as readonly ArrayLike<number>[];
        expect(spellsB.map((s) => Array.from(s))).toEqual([
            [1, 2],
            [4, Infinity],
        ]);
        expect(snapshot.nodes.byRole("timestamp")?.value(2)).toBe(3);
        expect(snapshot.nodes.byRole("timestamps")?.value(3)).toEqual([1, 2, 5]);
        const spellsE = spells?.value(4) as readonly ArrayLike<number>[];
        expect(spellsE.map((s) => Array.from(s))).toEqual([
            [1, 3],
            [5, 8],
        ]);
        expect(snapshot.edges.byRole("start")?.value(0)).toBe(2);
        expect(snapshot.edges.byRole("end")?.value(0)).toBe(3);
    });

    it("DEFECT: GEXF 1.2 startopen / endopen carry the bound's time (XSD time-type), not a boolean", async () => {
        // gexf.net/1.2draft/dynamics.xsd: <xs:attribute name="startopen" type="time-type"/>; the
        // primer writes open bounds as startopen="2009-03-01" in place of start. The importer only
        // recognises startopen="true", so a real 1.2 file loses the bound AND the open bit silently.
        const doc = `<gexf ${NS12}><graph defaultedgetype="directed" mode="dynamic" timeformat="integer"><nodes><node id="a" start="1" endopen="5"/><node id="b" startopen="2" end="3"/></nodes></graph></gexf>`;
        const { snapshot, report } = await parse(doc);
        const start = snapshot.nodes.byRole("start");
        const end = snapshot.nodes.byRole("end");
        const open = snapshot.nodes.byRole("open");
        expect(report.issues.filter((i) => i.severity === "error")).toHaveLength(0);
        expect(start?.value(0)).toBe(1);
        expect(end?.value(0)).toBe(5);
        expect(start?.value(1)).toBe(2);
        expect(end?.value(1)).toBe(3);
        expect(open).not.toBeNull();
        expect(open?.value(0)).toBe(OPEN_END);
        expect(open?.value(1)).toBe(OPEN_START);
    });

    it("puts dynamic attribute values into a temporal extension table and marks the column dynamic", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed" mode="dynamic" timeformat="integer"><attributes class="node" mode="dynamic"><attribute id="p" title="price" type="double"/></attributes><nodes><node id="a"><attvalues><attvalue for="p" value="1.5" start="1" end="2"/><attvalue for="p" value="2.5" start="2" end="3"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot } = await parse(doc);
        expect(snapshot.nodes.get("price")?.meta.dynamic).toBe(true);
        const table = snapshot.extensions.get("temporal:node:price");
        expect(table?.rowCount).toBe(2);
        expect([0, 1].map((r) => table?.value("element", r))).toEqual([0, 0]);
        expect([0, 1].map((r) => table?.value("start", r))).toEqual([1, 2]);
        expect([0, 1].map((r) => table?.value("end", r))).toEqual([2, 3]);
        expect([0, 1].map((r) => table?.value("value", r))).toEqual([1.5, 2.5]);
    });

    it("converts date / dateTime bounds to epoch milliseconds and keeps a text companion for offsets", async () => {
        const date = `<gexf ${NS13}><graph defaultedgetype="directed" mode="dynamic" timeformat="date"><nodes><node id="a" start="2010-01-01" end="2010-12-31"/></nodes></graph></gexf>`;
        const r1 = await parse(date);
        expect(r1.snapshot.nodes.byRole("start")?.value(0)).toBe(Date.UTC(2010, 0, 1));
        expect(r1.snapshot.nodes.byRole("end")?.value(0)).toBe(Date.UTC(2010, 11, 31));
        const dateTime = `<gexf ${NS13}><graph defaultedgetype="directed" mode="dynamic" timeformat="dateTime"><nodes><node id="a" start="2010-01-01T10:00:00+02:00"/><node id="b" start="2010-01-01T08:00:00Z"/></nodes></graph></gexf>`;
        const r2 = await parse(dateTime);
        expect(r2.snapshot.nodes.byRole("start")?.value(0)).toBe(Date.UTC(2010, 0, 1, 8));
        expect(r2.snapshot.nodes.byRole("start")?.value(1)).toBe(Date.UTC(2010, 0, 1, 8));
        const companion = r2.snapshot.nodes.get("start.text");
        expect(companion?.meta.role).toBe("timeText");
        expect(companion?.meta.extra.for).toBe("start");
        expect(companion?.value(0)).toBe("2010-01-01T10:00:00+02:00");
        expect(companion?.isSet(1)).toBe(false);
    });

    it("reads every viz element: colour (rgba and hex), position (z optional), size, shape, thickness", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"><viz:color r="255" g="0" b="0" a="0.5"/><viz:position x="1.5" y="2.5" z="3.5"/><viz:size value="2"/><viz:shape value="square"/></node><node id="b"><viz:color hex="#00FF00"/><viz:position x="1" y="2"/><viz:shape value="image" uri="http://x/y.png"/></node></nodes><edges><edge source="a" target="b"><viz:color r="0" g="0" b="255"/><viz:thickness value="3"/><viz:shape value="dashed"/></edge></edges></graph></gexf>`;
        const { snapshot } = await parse(doc);
        const n = snapshot.nodes;
        expect(Array.from(n.value("color", 0) as ArrayLike<number>)).toEqual([1, 0, 0, 0.5]);
        expect(Array.from(n.value("color", 1) as ArrayLike<number>)).toEqual([0, 1, 0, 1]);
        expect(Array.from(n.value("position", 0) as ArrayLike<number>)).toEqual([1.5, 2.5, 3.5]);
        expect(Array.from(n.value("position", 1) as ArrayLike<number>)).toEqual([1, 2, 0]);
        expect(n.value("size", 0)).toBe(2);
        expect(n.isSet("size", 1)).toBe(false);
        expect(n.value("shape", 0)).toBe("square");
        expect(n.value("shape", 1)).toBe("image");
        expect(n.value("shape.uri", 1)).toBe("http://x/y.png");
        expect(Array.from(snapshot.edges.value("color", 0) as ArrayLike<number>)).toEqual([0, 0, 1, 1]);
        expect(snapshot.edges.value("thickness", 0)).toBe(3);
        expect(snapshot.edges.value("shape", 0)).toBe("dashed");
    });

    it("distinguishes an absent weight from weight=1.0 by validity and keeps 0.1 exact in the shadow", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"/><node id="b"/></nodes><edges><edge source="a" target="b"/><edge source="b" target="a" weight="1.0"/><edge source="a" target="a" weight="0.1"/></edges></graph></gexf>`;
        const { snapshot } = await parse(doc);
        const weight = snapshot.edges.byRole("weight");
        expect(weight?.dtype).toBe("f64");
        expect(weight?.isSet(0)).toBe(false);
        expect(weight?.isSet(1)).toBe(true);
        expect(weight?.value(1)).toBe(1);
        expect(weight?.value(2)).toBe(0.1);
        expect(snapshot.edgeList().weights?.[2]).toBe(Math.fround(0.1));
        expect(snapshot.selfLoopCount).toBe(1);
    });

    it("refuses an edge to an undeclared node by default and creates it under addMissingNodes", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"/></nodes><edges><edge source="a" target="zz"/></edges></graph></gexf>`;
        const refused = await parse(doc);
        expect(refused.snapshot.edgeCount).toBe(0);
        expect(refused.report.issues.map((i) => i.code)).toContain("E_UNKNOWN_NODE");
        const added = await parse(doc, { addMissingNodes: true });
        expect(added.snapshot.ids.toArray()).toEqual(["a", "zz"]);
        expect(added.snapshot.edgeCount).toBe(1);
    });

    it("reads namespace-prefixed documents (lxml output) and the 1.1 triple-slash viz namespace", async () => {
        const prefixed = `<ns0:gexf xmlns:ns0="http://gexf.net/1.3" version="1.3"><ns0:graph defaultedgetype="directed"><ns0:nodes><ns0:node id="a"/><ns0:node id="b"/></ns0:nodes><ns0:edges><ns0:edge source="a" target="b"/></ns0:edges></ns0:graph></ns0:gexf>`;
        const r1 = await parse(prefixed);
        expect(r1.snapshot.nodeCount).toBe(2);
        expect(r1.snapshot.edgeCount).toBe(1);
        expect(r1.snapshot.directed).toBe(true);
        const gephi11 = `<gexf xmlns:viz="http:///www.gexf.net/1.1draft/viz" version="1.1" xmlns="http://www.gexf.net/1.1draft"><graph defaultedgetype="undirected" idtype="string" type="static"><nodes><node id="0" label="x"><viz:color b="243" g="107" r="88"/></node></nodes></graph></gexf>`;
        const r2 = await parse(gephi11);
        expect(r2.snapshot.meta.mode).toBe("static");
        expect(Array.from(r2.snapshot.nodes.value("color", 0) as ArrayLike<number>)).toEqual(
            [88 / 255, 107 / 255, 243 / 255, 1].map(Math.fround),
        );
    });

    it("maps the declared scalar types exactly: long precision, float to f32, bigdecimal / anyURI / char to string", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="bd" type="bigdecimal"/><attribute id="1" title="l" type="long"/><attribute id="2" title="u" type="anyURI"/><attribute id="3" title="c" type="char"/><attribute id="4" title="by" type="byte"/><attribute id="5" title="sh" type="short"/><attribute id="6" title="f" type="float"/></attributes><nodes><node id="a"><attvalues><attvalue for="0" value="1.000000000000000000001"/><attvalue for="1" value="9007199254740993"/><attvalue for="2" value="http://x"/><attvalue for="3" value="q"/><attvalue for="4" value="127"/><attvalue for="5" value="-5"/><attvalue for="6" value="0.1"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot, report } = await parse(doc);
        const dtypeOf = (name: string): string | undefined => snapshot.nodes.get(name)?.dtype;
        expect(dtypeOf("bd")).toBe("string");
        expect(snapshot.nodes.value("bd", 0)).toBe("1.000000000000000000001");
        expect(dtypeOf("l")).toBe("f64");
        expect(snapshot.nodes.value("l", 0)).toBe(9007199254740992);
        expect(report.issues.some((i) => i.code === "W_PRECISION" && i.category === "precision")).toBe(true);
        expect(dtypeOf("u")).toBe("string");
        expect(snapshot.nodes.get("u")?.meta.origin?.type).toBe("anyURI");
        expect(dtypeOf("c")).toBe("string");
        expect(dtypeOf("by")).toBe("i32");
        expect(dtypeOf("sh")).toBe("i32");
        expect(dtypeOf("f")).toBe("f32");
        expect(snapshot.nodes.value("f", 0)).toBe(Math.fround(0.1));
    });

    it("renames a declared attribute that collides with an XML-derived column and warns", async () => {
        const doc = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="label" type="string"/></attributes><nodes><node id="a" label="L"><attvalues><attvalue for="0" value="attr"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot, report } = await parse(doc);
        expect(snapshot.nodes.value("label", 0)).toBe("L");
        expect(snapshot.nodes.value("label#0", 0)).toBe("attr");
        expect(report.issues.map((i) => i.code)).toContain("W_COLUMN_RENAMED");
    });

    it("reports an attvalue naming an undeclared attribute and one without a value attribute", async () => {
        const unknown = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"><attvalues><attvalue for="zz" value="1"/></attvalues></node></nodes></graph></gexf>`;
        expect((await parse(unknown)).report.issues.map((i) => i.code)).toContain("W_GEXF_UNKNOWN_ATTRIBUTE");
        const textContent = `<gexf ${NS13}><graph defaultedgetype="directed"><attributes class="node"><attribute id="0" title="t" type="string"/></attributes><nodes><node id="a"><attvalues><attvalue for="0">text content</attvalue></attvalues></node></nodes></graph></gexf>`;
        const r = await parse(textContent);
        expect(r.report.issues.map((i) => i.code)).toContain("W_GEXF_ATTVALUE_SHAPE");
        expect(r.snapshot.nodes.isSet("t", 0)).toBe(false);
    });

    it("skips viz under viz: false with one warning, and keeps ids like 0.0 and 1 distinct", async () => {
        const viz = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="a"><viz:color r="1" g="2" b="3"/></node></nodes></graph></gexf>`;
        const r = await parse(viz, { viz: false });
        expect(r.snapshot.nodes.has("color")).toBe(false);
        expect(r.report.issues.map((i) => i.code)).toContain("W_GEXF_VIZ_SKIPPED");
        const ids = `<gexf ${NS13}><graph defaultedgetype="directed"><nodes><node id="0.0"/><node id="1.0"/><node id="1"/></nodes><edges><edge source="0.0" target="1"/></edges></graph></gexf>`;
        const r2 = await parse(ids);
        expect(r2.snapshot.ids.toArray()).toEqual(["0.0", "1.0", 1]);
        expect(edgeEndpoints(r2.snapshot, 0)).toEqual(["0.0", "1"]);
    });
});
