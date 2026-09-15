/**
 * Semantic audit of the GML importer against the corpus files themselves (facts derived here by a
 * hand-walked scan of the `node [ ... ]` / `edge [ ... ]` records in the raw text) and against the
 * quirks graphty-element's GMLDataSource, NetworkX's gml.py and research note 07 section 2.2
 * describe.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("gml", name), { format: "gml", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "gml", ...options });
}

interface RawNode {
    readonly id: number;
    readonly label: string | null;
    /** The raw `value` token: a quoted string keeps its quotes, an integer is bare, null when absent. */
    readonly value: string | null;
}

/** The node records of a Newman-style GML file, in file order (flat records only). */
function rawNodes(text: string): RawNode[] {
    const out: RawNode[] = [];
    for (const m of text.matchAll(/node\s*\[\s*([^\]]*)\]/g)) {
        const body = m[1];
        const id = /\bid (-?\d+)/.exec(body);
        const label = /\blabel "([^"]*)"/.exec(body);
        const value = /\bvalue ("[^"]*"|-?\d+)/.exec(body);
        out.push({
            id: Number(id?.[1]),
            label: label === null ? null : label[1],
            value: value === null ? null : value[1],
        });
    }
    return out;
}

/** The edge records of a Newman-style GML file, in file order. */
function rawEdges(text: string): { source: number; target: number }[] {
    return [...text.matchAll(/edge\s*\[\s*source (-?\d+)\s*target (-?\d+)\s*\]/g)].map((m) => ({
        source: Number(m[1]),
        target: Number(m[2]),
    }));
}

function endpoints(snapshot: GraphSnapshot, e: number): [number, number] {
    const list = snapshot.edgeList();
    return [snapshot.ids.idOf(list.src[e]) as number, snapshot.ids.idOf(list.dst[e]) as number];
}

function checkTopology(
    snapshot: GraphSnapshot,
    nodes: readonly RawNode[],
    edges: readonly { source: number; target: number }[],
): void {
    expect(snapshot.nodeCount).toBe(nodes.length);
    expect(snapshot.edgeCount).toBe(edges.length);
    for (let i = 0; i < nodes.length; i++) {
        expect(snapshot.ids.idOf(i)).toBe(nodes[i].id);
    }
    for (let e = 0; e < edges.length; e++) {
        expect(endpoints(snapshot, e)).toEqual([edges[e].source, edges[e].target]);
    }
    const degree = new Map<number, number>();
    const seen = new Set<string>();
    for (const e of edges) {
        expect(e.source).not.toBe(e.target);
        const key = [e.source, e.target].sort((a, b) => a - b).join("|");
        expect(seen.has(key)).toBe(false);
        seen.add(key);
        degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
        degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
    }
    expect(snapshot.selfLoopCount).toBe(0);
    expect(snapshot.flags.multigraph).toBe(false);
    const degrees = snapshot.degree();
    for (const [id, d] of degree) {
        expect(degrees[snapshot.ids.requireIndex(id)]).toBe(d);
    }
}

describe("GML corpus facts: karate.gml (no directed key, no labels, ids 1..34)", () => {
    const text = readCorpusText("gml", "karate.gml");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("the raw file has 34 nodes numbered 1..34, 78 edges not grouped by source, no directed key", () => {
        expect(nodes).toHaveLength(34);
        expect(nodes.map((n) => n.id)).toEqual(Array.from({ length: 34 }, (_, i) => i + 1));
        expect(nodes.every((n) => n.label === null && n.value === null)).toBe(true);
        expect(edges).toHaveLength(78);
        expect(/\bdirected\b/.test(text)).toBe(false);
        expect(edges.slice(0, 4)).toEqual([
            { source: 2, target: 1 },
            { source: 3, target: 1 },
            { source: 3, target: 2 },
            { source: 4, target: 1 },
        ]);
        expect(edges[77]).toEqual({ source: 34, target: 33 });
    });

    it("imports an undirected identity-id graph in file order with no label column", async () => {
        const { snapshot, report } = await load("karate.gml");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(1);
        checkTopology(snapshot, nodes, edges);
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.flags.weighted).toBe(false);
        expect(snapshot.degree()[snapshot.ids.requireIndex(34)]).toBe(17);
        expect(snapshot.degree()[snapshot.ids.requireIndex(1)]).toBe(16);
        expect(snapshot.meta.creator).toBe("Mark Newman on Fri Jul 21 12:39:27 2006");
        expect(snapshot.meta.sourceFormat).toBe("gml");
    });
});

describe("GML corpus facts: polbooks.gml (directed 0, string value attribute)", () => {
    const text = readCorpusText("gml", "polbooks.gml");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("the raw file has 105 nodes with quoted values c/l/n, 441 edges, directed 0", () => {
        expect(nodes).toHaveLength(105);
        expect(edges).toHaveLength(441);
        expect(text).toMatch(/directed 0/);
        const counts = new Map<string, number>();
        for (const n of nodes) {
            counts.set(n.value ?? "", (counts.get(n.value ?? "") ?? 0) + 1);
        }
        expect(counts.get('"c"')).toBe(49);
        expect(counts.get('"l"')).toBe(43);
        expect(counts.get('"n"')).toBe(13);
        expect(nodes[0]).toEqual({ id: 0, label: "1000 Years for Revenge", value: '"n"' });
        expect(nodes[2].label).toBe("Charlie Wilson's War");
        expect(nodes[10]).toEqual({ id: 10, label: "Dereliction of Duty", value: '"c"' });
        expect(edges[0]).toEqual({ source: 1, target: 0 });
        expect(edges[440]).toEqual({ source: 104, target: 103 });
    });

    it("imports the topology, labels and the value column as a low-cardinality dictionary of strings", async () => {
        const { snapshot, report } = await load("polbooks.gml");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(false);
        checkTopology(snapshot, nodes, edges);
        const value = snapshot.nodes.require("value");
        expect(value.dtype).toBe("dict");
        expect(value.meta.origin?.type).toBe("string");
        expect(value.nullCount).toBe(0);
        let c = 0;
        for (let i = 0; i < nodes.length; i++) {
            expect(snapshot.nodes.value("label", i)).toBe(nodes[i].label);
            expect(value.value(i)).toBe(nodes[i].value?.slice(1, -1));
            if (value.value(i) === "c") {
                c++;
            }
        }
        expect(c).toBe(49);
        expect(snapshot.nodes.value("label", 2)).toBe("Charlie Wilson's War");
        expect(snapshot.degree()[snapshot.ids.requireIndex(8)]).toBe(25);
        expect(snapshot.degree()[snapshot.ids.requireIndex(12)]).toBe(25);
        expect(snapshot.meta.creator).toBe("Mark Newman on Wed Oct 18 16:42:04 2006");
    });

    it("keeps the string value column when the dictionary heuristic is disabled", async () => {
        const { snapshot } = await load("polbooks.gml", { dictionaries: false });
        expect(snapshot.nodes.require("value").dtype).toBe("string");
        expect(snapshot.nodes.value("value", 0)).toBe("n");
    });
});

describe("GML corpus facts: football.gml (integer value attribute)", () => {
    const text = readCorpusText("gml", "football.gml");
    const nodes = rawNodes(text);
    const edges = rawEdges(text);

    it("the raw file has 115 nodes with integer values, 613 edges, directed 0", () => {
        expect(nodes).toHaveLength(115);
        expect(edges).toHaveLength(613);
        expect(nodes.every((n) => n.value !== null && /^\d+$/.test(n.value))).toBe(true);
        expect(nodes[0]).toEqual({ id: 0, label: "BrighamYoung", value: "7" });
        expect(nodes[10]).toEqual({ id: 10, label: "Baylor", value: "3" });
        expect(nodes.filter((n) => n.value === "6")).toHaveLength(13);
        expect(edges[0]).toEqual({ source: 1, target: 0 });
        expect(edges[612]).toEqual({ source: 114, target: 104 });
    });

    it("imports the value column as i32 with origin int and every label", async () => {
        const { snapshot, report } = await load("football.gml");
        expect(report.errorCount).toBe(0);
        checkTopology(snapshot, nodes, edges);
        const value = snapshot.nodes.require("value");
        expect(value.dtype).toBe("i32");
        expect(value.meta.origin?.type).toBe("int");
        let sixes = 0;
        for (let i = 0; i < nodes.length; i++) {
            expect(snapshot.nodes.value("label", i)).toBe(nodes[i].label);
            expect(value.value(i)).toBe(Number(nodes[i].value));
            if (value.value(i) === 6) {
                sixes++;
            }
        }
        expect(sixes).toBe(13);
        expect(snapshot.degree()[snapshot.ids.requireIndex(0)]).toBe(12);
    });
});

describe("GML corpus facts: dolphins.gml and minimal.gml", () => {
    it("dolphins: 62 labelled nodes 0..61 and 159 undirected edges in file order", async () => {
        const text = readCorpusText("gml", "dolphins.gml");
        const nodes = rawNodes(text);
        const edges = rawEdges(text);
        expect(nodes).toHaveLength(62);
        expect(edges).toHaveLength(159);
        expect(nodes[0].label).toBe("Beak");
        expect(nodes[61].label).toBe("Zipfel");
        expect(edges[0]).toEqual({ source: 8, target: 3 });
        const { snapshot } = await load("dolphins.gml");
        checkTopology(snapshot, nodes, edges);
        expect(snapshot.nodes.value("label", 0)).toBe("Beak");
        expect(snapshot.nodes.value("label", 61)).toBe("Zipfel");
        expect(snapshot.degree()[snapshot.ids.requireIndex(14)]).toBe(12);
    });

    it("minimal: three labelled nodes and two edges on one-line records", async () => {
        const { snapshot } = await load("minimal.gml");
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect([0, 1, 2].map((i) => snapshot.nodes.value("label", i))).toEqual(["A", "B", "C"]);
        expect(endpoints(snapshot, 0)).toEqual([1, 2]);
        expect(endpoints(snapshot, 1)).toEqual([2, 3]);
        expect(snapshot.directed).toBe(false);
    });
});

describe("GML quirks from GMLDataSource, NetworkX gml.py and research note 07", () => {
    it("strips # comments outside strings and keeps # inside strings", async () => {
        const { snapshot } = await parse(
            'graph [ # comment\n node [ id 1 label "a # not comment" ] # trailing\n node [ id 2 ]\n edge [ source 1 target 2 ]\n]',
        );
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.nodes.value("label", 0)).toBe("a # not comment");
    });

    it("decodes the numeric and named character entities NetworkX writes for non-ASCII and quotes", async () => {
        const { snapshot } = await parse('graph [ node [ id 1 label "caf&#233; &quot;q&quot; &amp; &#38;" ] ]');
        expect(snapshot.nodes.value("label", 0)).toBe(`caf${String.fromCharCode(0xe9)} "q" & &`);
    });

    it("honours _networkx_list_start so a one-element list re-imports as a list", async () => {
        const { snapshot } = await parse(
            'graph [ node [ id 1 lst "_networkx_list_start" lst "a" ] node [ id 2 lst "x" lst "y" ] ]',
        );
        const lst = snapshot.nodes.require("lst");
        expect(lst.dtype).toBe("list");
        expect(lst.value(0)).toEqual(["a"]);
        expect(lst.value(1)).toEqual(["x", "y"]);
    });

    it("records multigraph 1 and the key attribute of parallel edges", async () => {
        const { snapshot } = await parse(
            "graph [ multigraph 1 node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 key 0 ] edge [ source 1 target 2 key 1 ] ]",
        );
        expect(snapshot.meta.declaredMultigraph).toBe(true);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.edges.byRole("key")?.value(1)).toBe(1);
    });

    it("reads directed 1 as a directed graph, even when the key follows the edges", async () => {
        expect(
            (await parse("graph [ directed 1 node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] ]")).snapshot
                .directed,
        ).toBe(true);
        expect(
            (await parse("graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] directed 1 ]")).snapshot
                .directed,
        ).toBe(true);
    });

    it("maps the graphics record to the position role and the rest of the record to json", async () => {
        const { snapshot } = await parse(
            'graph [ node [ id 1 graphics [ x 1.5 y 2.0 z 0.5 w 10 fill "#ff0000" ] ] node [ id 2 graphics [ x 3.0 y 4.0 ] ] ]',
        );
        const position = snapshot.nodes.byRole("position");
        expect(Array.from(position?.value(0) as ArrayLike<number>)).toEqual([1.5, 2, 0.5]);
        expect(Array.from(position?.value(1) as ArrayLike<number>)).toEqual([3, 4, 0]);
        expect(snapshot.nodes.value("graphics", 0)).toEqual({ w: 10, fill: "#ff0000" });
        expect(snapshot.nodes.isSet("graphics", 1)).toBe(false);
        const whole = await parse("graph [ node [ id 1 graphics [ x 1.5 y 2.0 w 10 ] ] ]", { positions: false });
        expect(whole.snapshot.nodes.byRole("position")).toBeNull();
        expect(whole.snapshot.nodes.value("graphics", 0)).toEqual({ x: 1.5, y: 2, w: 10 });
    });

    it("turns repeated keys into lists and nested records into json", async () => {
        const { snapshot } = await parse(
            'graph [ node [ id 1 tag "a" tag "b" rec [ a 1 sub [ b 2 ] ] ] node [ id 2 tag "c" ] ]',
        );
        expect(snapshot.nodes.value("tag", 0)).toEqual(["a", "b"]);
        expect(snapshot.nodes.value("tag", 1)).toEqual(["c"]);
        expect(snapshot.nodes.get("rec")?.dtype).toBe("json");
        expect(snapshot.nodes.value("rec", 0)).toEqual({ a: 1, sub: { b: 2 } });
    });

    it("keeps the lexical int / real distinction per column and widens int to real when mixed", async () => {
        const { snapshot, report } = await parse(
            "graph [ node [ id 1 a 1E-05 b .5 c -3 d 1. e 2147483648 g 9007199254740993 ] node [ id 2 c 4 ] ]",
        );
        const meta = (name: string): [string, string | null | undefined] => [
            snapshot.nodes.require(name).dtype,
            snapshot.nodes.require(name).meta.origin?.type,
        ];
        expect(meta("a")).toEqual(["f64", "real"]);
        expect(snapshot.nodes.value("a", 0)).toBe(0.00001);
        expect(meta("b")).toEqual(["f64", "real"]);
        expect(meta("c")).toEqual(["i32", "int"]);
        expect(meta("d")).toEqual(["f64", "real"]);
        expect(meta("e")).toEqual(["f64", "int"]);
        expect(snapshot.nodes.value("e", 0)).toBe(2147483648);
        expect(report.issues.some((i) => i.category === "precision")).toBe(true);
        const mixed = await parse("graph [ node [ id 1 v 1 ] node [ id 2 v 2.5 ] ]");
        expect(mixed.snapshot.nodes.require("v").dtype).toBe("f64");
        expect(mixed.snapshot.nodes.require("v").meta.origin?.type).toBe("real");
    });

    it("takes Creator and Version into the meta and any other top-level or graph key into the graph table", async () => {
        const { snapshot } = await parse('Creator "x"\nVersion 1\ngraph [ name "gname" foo 3 node [ id 1 ] ]\nextra 5');
        expect(snapshot.meta.creator).toBe("x");
        expect(snapshot.meta.sourceVersion).toBe("1");
        expect(snapshot.graph.value("name", 0)).toBe("gname");
        expect(snapshot.graph.value("foo", 0)).toBe(3);
        expect(snapshot.graph.value("extra", 0)).toBe(5);
    });

    it("reads the Newman value key as the weight by default and weight only when asked", async () => {
        const { snapshot } = await parse(
            "graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 value 0.1 ] edge [ source 2 target 1 weight 3 ] ]",
        );
        expect(snapshot.edges.byRole("weight")?.value(0)).toBe(0.1);
        expect(snapshot.edges.byRole("weight")?.isSet(1)).toBe(false);
        expect(snapshot.edges.value("weight", 1)).toBe(3);
        const w = await parse("graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 weight 0.5 ] ]", {
            weightFrom: "weight",
        });
        expect(Array.from(w.snapshot.edgeList().weights ?? [])).toEqual([0.5]);
    });

    it("uses the label as the node id under nodeIdFrom: label, keeps brackets inside strings, and CRLF", async () => {
        const byLabel = await parse(
            'graph [ node [ id 1 label "A" ] node [ id 2 label "B" ] edge [ source 1 target 2 ] ]',
            { nodeIdFrom: "label" },
        );
        expect(byLabel.snapshot.ids.toArray()).toEqual(["A", "B"]);
        expect(byLabel.snapshot.edgeCount).toBe(1);
        const brackets = await parse('graph [ node [ id 1 label "a [b] c" ] ]');
        expect(brackets.snapshot.nodes.value("label", 0)).toBe("a [b] c");
        const crlf = await parse(
            "graph [\r\n directed 1\r\n node [ id 1 ]\r\n node [ id 2 ]\r\n edge [ source 1 target 2 value 2.5 ]\r\n]\r\n",
        );
        expect(crlf.snapshot.directed).toBe(true);
        expect(Array.from(crlf.snapshot.edgeList().weights ?? [])).toEqual([2.5]);
    });

    it("creates an endpoint declared only by an edge, accepts edges before nodes, and keeps self-loops", async () => {
        const { snapshot } = await parse(
            "graph [ edge [ source 1 target 9 ] node [ id 1 ] edge [ source 9 target 9 ] ]",
        );
        expect(snapshot.ids.toArray()).toEqual([1, 9]);
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.selfLoopCount).toBe(1);
    });

    it("reports a node without an id, a duplicate id and a non-integer id explicitly", async () => {
        const noId = await parse('graph [ node [ label "x" ] node [ id 2 ] ]');
        expect(noId.report.issues.map((i) => i.code)).toContain("E_MISSING_ID");
        expect(noId.snapshot.nodeCount).toBe(1);
        const dup = await parse('graph [ node [ id 1 label "a" ] node [ id 1 label "b" ] ]');
        expect(dup.report.issues.map((i) => i.code)).toContain("W_DUPLICATE_NODE");
        // NetworkX accepts string-convertible ids; the importer follows the GML spec (C int) and
        // refuses them with an explicit issue rather than silently coercing.
        const strings = await parse('graph [ node [ id "a" ] node [ id "b" ] edge [ source "a" target "b" ] ]');
        expect(strings.report.issues.map((i) => i.code)).toContain("E_GML_ID_TYPE");
        expect(strings.snapshot.nodeCount).toBe(0);
    });

    it("refuses a string spanning lines (NetworkX joins them) with an explicit parse error", async () => {
        await expect(parse('graph [ node [ id 1 label "line1\nline2" ] ]')).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_SYNTAX" }] },
        });
    });

    it("keeps duplicate edges without multigraph 1 (duplicateEdges keep), unlike NetworkX which errors", async () => {
        const { snapshot, report } = await parse(
            "graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] edge [ source 1 target 2 ] ]",
        );
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.meta.declaredMultigraph).toBeNull();
        expect(report.errorCount).toBe(0);
    });

    it("keeps empty strings, the literal None NetworkX writes, and 1 / 0 as integers rather than booleans", async () => {
        const { snapshot } = await parse('graph [ node [ id 1 s "" flag 1 ] node [ id 2 s "None" flag 0 ] ]');
        expect(snapshot.nodes.value("s", 0)).toBe("");
        expect(snapshot.nodes.value("s", 1)).toBe("None");
        expect(snapshot.nodes.require("flag").dtype).toBe("i32");
        expect(snapshot.nodes.value("flag", 0)).toBe(1);
    });
});
