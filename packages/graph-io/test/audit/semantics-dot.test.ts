/**
 * Semantic audit of the DOT importer against the corpus files themselves (facts derived here by
 * hand-walked line parsing of the raw text) and against the quirks graphty-element's
 * DOTDataSource, the Graphviz grammar and research note 07 section 2.4 describe. Tests named
 * "DEFECT:" pin a defect and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** The byte-order mark, built from its code so this file stays ASCII. */
const BOM = String.fromCharCode(0xfeff);

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("dot", name), { format: "dot", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "dot", ...options });
}

function unquote(id: string): string {
    return id.startsWith('"') && id.endsWith('"') ? id.slice(1, -1) : id;
}

/** `key=value` pairs of a DOT attribute list; values unquoted. */
function attrsOf(list: string): Map<string, string> {
    const out = new Map<string, string>();
    for (const m of list.matchAll(/(\w+)=("[^"]*"|[^,\s\]]+)/g)) {
        out.set(m[1], unquote(m[2]));
    }
    return out;
}

function endpoints(snapshot: GraphSnapshot, e: number): [string, string] {
    const list = snapshot.edgeList();
    return [String(snapshot.ids.idOf(list.src[e])), String(snapshot.ids.idOf(list.dst[e]))];
}

function edgeStrings(snapshot: GraphSnapshot): string[] {
    const out: string[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        out.push(endpoints(snapshot, e).join("->"));
    }
    return out;
}

describe("DOT corpus facts: root.gv (1054 node statements, 1083 edge statements)", () => {
    const text = readCorpusText("dot", "root.gv");
    const body = text.slice(text.indexOf("{") + 1);
    const nodeStatements = [...body.matchAll(/^\s*("?[A-Za-z0-9_]+"?)\s*\[([^\]]*)\];/gm)]
        .filter((m) => !["graph", "node", "edge"].includes(m[1]))
        .map((m) => ({ id: unquote(m[1]), attrs: attrsOf(m[2]) }));
    const edgeStatements = [
        ...body.matchAll(/^\s*("?[A-Za-z0-9_]+"?)\s*->\s*("?[A-Za-z0-9_]+"?)\s*(?:\[([^\]]*)\])?;/gm),
    ].map((m) => ({
        source: unquote(m[1]),
        target: unquote(m[2]),
        attrs: attrsOf(m[3] ?? ""),
    }));

    it("the raw file: a digraph named G_component_0 with 1054 distinct nodes and 1083 edges, no undeclared endpoint", () => {
        expect(text).toMatch(/^\s*(\/\*[\s\S]*?\*\/\s*)?digraph G_component_0 \{/);
        expect(nodeStatements).toHaveLength(1054);
        expect(new Set(nodeStatements.map((n) => n.id)).size).toBe(1054);
        expect(edgeStatements).toHaveLength(1083);
        const declared = new Set(nodeStatements.map((n) => n.id));
        for (const e of edgeStatements) {
            expect(declared.has(e.source)).toBe(true);
            expect(declared.has(e.target)).toBe(true);
            expect(e.source).not.toBe(e.target);
        }
        expect(nodeStatements.slice(0, 5).map((n) => n.id)).toEqual(["1", "189E", "790E", "2", "191E"]);
    });

    it("imports the counts, the name, first-mention order and canonical ids (numerals become numbers)", async () => {
        const { snapshot, report } = await load("root.gv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta.name).toBe("G_component_0");
        expect(snapshot.nodeCount).toBe(1054);
        expect(snapshot.edgeCount).toBe(1083);
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.ids.kind).toBe("mixed");
        let numeric = 0;
        for (let i = 0; i < nodeStatements.length; i++) {
            const raw = nodeStatements[i].id;
            const expected = /^(0|[1-9][0-9]*)$/.test(raw) ? Number(raw) : raw;
            expect(snapshot.ids.idOf(i)).toBe(expected);
            if (typeof expected === "number") {
                numeric++;
            }
        }
        expect(numeric).toBe(344);
        expect(snapshot.ids.toArray().slice(0, 5)).toEqual([1, "189E", "790E", 2, "191E"]);
    });

    it("reproduces every node's label, shape, style and colour, including the 7 quoted hex colours", async () => {
        const { snapshot } = await load("root.gv");
        const shapes = new Map<string, number>();
        let noLabel = 0;
        let quotedColors = 0;
        for (let i = 0; i < nodeStatements.length; i++) {
            const { attrs } = nodeStatements[i];
            const shape = attrs.get("shape");
            shapes.set(shape ?? "", (shapes.get(shape ?? "") ?? 0) + 1);
            expect(snapshot.nodes.value("shape", i)).toBe(shape);
            if (attrs.has("label")) {
                expect(snapshot.nodes.value("label", i)).toBe(attrs.get("label"));
            } else {
                noLabel++;
                expect(snapshot.nodes.isSet("label", i)).toBe(false);
            }
            if (attrs.has("color")) {
                expect(snapshot.nodes.value("color", i)).toBe(attrs.get("color"));
                if (attrs.get("color")?.startsWith("#") === true) {
                    quotedColors++;
                }
            } else {
                expect(snapshot.nodes.isSet("color", i)).toBe(false);
            }
            expect(snapshot.nodes.isSet("style", i)).toBe(attrs.has("style"));
        }
        expect(noLabel).toBe(106);
        expect(quotedColors).toBe(7);
        expect([...shapes.entries()].sort()).toEqual([
            ["box", 710],
            ["doubleoctagon", 7],
            ["hexagon", 336],
            ["tripleoctagon", 1],
        ]);
        expect(snapshot.nodes.get("label")?.nullCount).toBe(106);
        expect(snapshot.nodes.get("color")?.nullCount).toBe(1054 - 709 - 258 - 7 - 1 - 7);
        expect(snapshot.nodes.get("label")?.meta.role).toBe("label");
    });

    it("keeps the one tripleoctagon node with its URL and area attributes, and the unquoted numeral label of 727E", async () => {
        const { snapshot } = await load("root.gv");
        const special = nodeStatements.find((n) => n.attrs.get("shape") === "tripleoctagon");
        expect(special?.id).toBe("336");
        expect(special?.attrs.get("URL")).toBe("tes hi");
        expect(special?.attrs.get("area")).toBe("test");
        expect(special?.attrs.get("color")).toBe("#ff0000");
        const index = snapshot.ids.requireIndex(336);
        expect(snapshot.nodes.value("URL", index)).toBe("tes hi");
        expect(snapshot.nodes.value("area", index)).toBe("test");
        expect(snapshot.nodes.get("URL")?.nullCount).toBe(1053);
        expect(snapshot.nodes.value("label", index)).toBe("825c7994d5da13afe519861818");
        expect(snapshot.nodes.value("label", snapshot.ids.requireIndex("727E"))).toBe("2");
    });

    it("reproduces every edge in order with its colour, arrowhead and optional label", async () => {
        const { snapshot } = await load("root.gv");
        let labelled = 0;
        for (let e = 0; e < edgeStatements.length; e++) {
            const stmt = edgeStatements[e];
            expect(endpoints(snapshot, e)).toEqual([stmt.source, stmt.target]);
            expect(snapshot.edges.value("color", e)).toBe(stmt.attrs.get("color"));
            expect(snapshot.edges.value("arrowhead", e)).toBe(stmt.attrs.get("arrowhead"));
            if (stmt.attrs.has("label")) {
                labelled++;
                expect(snapshot.edges.value("label", e)).toBe(stmt.attrs.get("label"));
            } else {
                expect(snapshot.edges.isSet("label", e)).toBe(false);
            }
        }
        expect(labelled).toBe(710);
        expect(snapshot.edges.get("label")?.nullCount).toBe(373);
        expect(edgeStatements[0]).toMatchObject({ source: "1", target: "189E" });
        expect(edgeStatements[0].attrs.get("label")).toBe(" ");
        expect(snapshot.edges.value("label", 0)).toBe(" ");
        expect(edgeStatements[1082]).toMatchObject({ source: "802E", target: "801E" });
        expect(snapshot.edges.value("color", 1082)).toBe("purple");
        expect(snapshot.edges.value("arrowhead", 1082)).toBe("none");
    });

    it("applies the node / edge fontname defaults to every element and reads the graph attributes", async () => {
        const { snapshot } = await load("root.gv");
        expect(text).toContain('node [fontname="Helvetica,Arial,sans-serif"]');
        expect(text).toContain('graph [ranksep=3, root="189E", overlap=prism];');
        expect(snapshot.nodes.get("fontname")?.nullCount).toBe(0);
        expect(snapshot.nodes.value("fontname", 1053)).toBe("Helvetica,Arial,sans-serif");
        expect(snapshot.edges.get("fontname")?.nullCount).toBe(0);
        expect(snapshot.graph.value("fontname", 0)).toBe("Helvetica,Arial,sans-serif");
        expect(snapshot.graph.value("layout", 0)).toBe("sfdp");
        expect(snapshot.graph.get("ranksep")?.dtype).toBe("i32");
        expect(snapshot.graph.value("ranksep", 0)).toBe(3);
        expect(snapshot.graph.value("root", 0)).toBe("189E");
        expect(snapshot.graph.value("overlap", 0)).toBe("prism");
        const outDegree = new Map<string, number>();
        for (const e of edgeStatements) {
            outDegree.set(e.source, (outDegree.get(e.source) ?? 0) + 1);
        }
        expect(outDegree.get("336")).toBe(38);
        expect(snapshot.outDegree()[snapshot.ids.requireIndex(336)]).toBe(38);
    });
});

describe("DOT corpus facts: cluster.gv (subgraph clusters, chains, scoped defaults)", () => {
    it("the raw file has two clusters, two chains of four and seven single edges", () => {
        const text = readCorpusText("dot", "cluster.gv");
        expect(text.match(/subgraph cluster_\d/g)).toEqual(["subgraph cluster_0", "subgraph cluster_1"]);
        expect(text).toContain("a0 -> a1 -> a2 -> a3;");
        expect(text).toContain("b0 -> b1 -> b2 -> b3;");
        expect((text.match(/^\s*\w+ -> \w+;$/gm) ?? []).length).toBe(7);
        expect(text).toContain('label = "process #1";');
        expect(text).toContain("node [style=filled,color=white];");
    });

    it("imports the clusters as container nodes with the members' parent set and 13 edges in order", async () => {
        const { snapshot, report } = await load("cluster.gv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.meta.name).toBe("G");
        expect(snapshot.directed).toBe(true);
        expect(snapshot.nodeCount).toBe(12);
        expect(snapshot.edgeCount).toBe(13);
        expect(snapshot.ids.toArray()).toEqual([
            "cluster_0",
            "a0",
            "a1",
            "a2",
            "a3",
            "cluster_1",
            "b0",
            "b1",
            "b2",
            "b3",
            "start",
            "end",
        ]);
        expect(edgeStrings(snapshot)).toEqual([
            "a0->a1",
            "a1->a2",
            "a2->a3",
            "b0->b1",
            "b1->b2",
            "b2->b3",
            "start->a0",
            "start->b0",
            "a1->b3",
            "b2->a3",
            "a3->a0",
            "a3->end",
            "b3->end",
        ]);
        const parent = snapshot.nodes.byRole("parent");
        const cluster = snapshot.nodes.require("graphty.cluster");
        for (const id of ["a0", "a1", "a2", "a3"]) {
            expect(parent?.value(snapshot.ids.requireIndex(id))).toBe(0);
        }
        for (const id of ["b0", "b1", "b2", "b3"]) {
            expect(parent?.value(snapshot.ids.requireIndex(id))).toBe(5);
        }
        expect(parent?.isSet(snapshot.ids.requireIndex("start"))).toBe(false);
        expect(cluster.value(0)).toBe(true);
        expect(cluster.value(5)).toBe(true);
        expect(cluster.nullCount).toBe(10);
        expect(snapshot.nodes.value("label", 0)).toBe("process #1");
        expect(snapshot.nodes.value("label", 5)).toBe("process #2");
        expect(snapshot.nodes.value("color", 0)).toBe("lightgrey");
        expect(snapshot.nodes.value("style", 0)).toBe("filled");
        expect(snapshot.nodes.value("color", 5)).toBe("blue");
    });

    it("applies the scoped node defaults: a-nodes filled white, b-nodes filled without colour, start / end shapes", async () => {
        const { snapshot } = await load("cluster.gv");
        for (const id of ["a0", "a1", "a2", "a3"]) {
            const i = snapshot.ids.requireIndex(id);
            expect(snapshot.nodes.value("style", i)).toBe("filled");
            expect(snapshot.nodes.value("color", i)).toBe("white");
            expect(snapshot.nodes.value("fontname", i)).toBe("Helvetica,Arial,sans-serif");
        }
        for (const id of ["b0", "b1", "b2", "b3"]) {
            const i = snapshot.ids.requireIndex(id);
            expect(snapshot.nodes.value("style", i)).toBe("filled");
            expect(snapshot.nodes.isSet("color", i)).toBe(false);
        }
        const start = snapshot.ids.requireIndex("start");
        const end = snapshot.ids.requireIndex("end");
        expect(snapshot.nodes.isSet("style", start)).toBe(false);
        expect(snapshot.nodes.value("shape", start)).toBe("Mdiamond");
        expect(snapshot.nodes.value("shape", end)).toBe("Msquare");
        expect(snapshot.nodes.get("shape")?.nullCount).toBe(10);
        expect(snapshot.edges.get("fontname")?.nullCount).toBe(0);
    });
});

describe("DOT corpus facts: datastruct.gv (ports, quoted ids, record labels)", () => {
    const text = readCorpusText("dot", "datastruct.gv");
    const edges = [...text.matchAll(/"(node\d+)":(\w+) -> "(node\d+)":(\w+) \[\s*id = (\d+)\s*\];/g)].map((m) => ({
        source: m[1],
        sourcePort: m[2],
        target: m[3],
        targetPort: m[4],
        id: Number(m[5]),
    }));

    it("the raw file declares 13 record nodes and 17 port-to-port edges numbered 0..16", () => {
        expect((text.match(/^"node\d+" \[/gm) ?? []).length).toBe(13);
        expect(edges).toHaveLength(17);
        expect(edges.map((e) => e.id)).toEqual(Array.from({ length: 17 }, (_, i) => i));
        expect(edges[0]).toEqual({ source: "node0", sourcePort: "f0", target: "node1", targetPort: "f0", id: 0 });
        expect(edges[16]).toEqual({ source: "node11", sourcePort: "f2", target: "node1", targetPort: "f0", id: 16 });
    });

    it("strips the ports from the endpoints and keeps them in the port role columns with the id attribute", async () => {
        const { snapshot, report } = await load("datastruct.gv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.nodeCount).toBe(13);
        expect(snapshot.ids.toArray()).toEqual(Array.from({ length: 13 }, (_, i) => `node${i}`));
        expect(snapshot.edgeCount).toBe(17);
        const sourcePort = snapshot.edges.byRole("sourcePort");
        const targetPort = snapshot.edges.byRole("targetPort");
        for (let e = 0; e < edges.length; e++) {
            expect(endpoints(snapshot, e)).toEqual([edges[e].source, edges[e].target]);
            expect(sourcePort?.value(e)).toBe(edges[e].sourcePort);
            expect(targetPort?.value(e)).toBe(edges[e].targetPort);
            expect(snapshot.edges.value("id", e)).toBe(edges[e].id);
        }
        expect(snapshot.edges.get("id")?.dtype).toBe("i32");
    });

    it("keeps record labels with <f0> field markers literally and applies fontsize / shape defaults", async () => {
        const { snapshot } = await load("datastruct.gv");
        expect(text).toContain('label = "<f0> 3.43322790286038071e-06|44.79998779296875|0"');
        expect(snapshot.nodes.value("label", 3)).toBe("<f0> 3.43322790286038071e-06|44.79998779296875|0");
        expect(snapshot.nodes.value("label", 0)).toBe("<f0> 0x10ba8| <f1>");
        expect(snapshot.nodes.get("fontsize")?.dtype).toBe("i32");
        expect(snapshot.nodes.get("fontsize")?.nullCount).toBe(0);
        expect(snapshot.nodes.value("fontsize", 12)).toBe(16);
        expect(snapshot.nodes.get("shape")?.nullCount).toBe(0);
        expect(snapshot.nodes.value("shape", 5)).toBe("record");
        expect(snapshot.graph.value("rankdir", 0)).toBe("LR");
        expect(snapshot.meta.name).toBe("g");
    });
});

describe("DOT corpus facts: fsm.gv and fdpclust.gv", () => {
    it("fsm: numeral ids in first-mention order, two self-loops, doublecircle for 0 3 4 8, edge labels", async () => {
        const text = readCorpusText("dot", "fsm.gv");
        const edges = [...text.matchAll(/^\s*(\d+) -> (\d+) \[label = "([^"]*)"\];/gm)].map((m) => ({
            s: Number(m[1]),
            t: Number(m[2]),
            label: m[3],
        }));
        expect(edges).toHaveLength(14);
        expect(edges.filter((e) => e.s === e.t).map((e) => e.s)).toEqual([5, 6]);
        expect(text).toContain("node [shape = doublecircle]; 0 3 4 8;");
        const { snapshot } = await load("fsm.gv");
        expect(snapshot.meta.name).toBe("finite_state_machine");
        expect(snapshot.ids.toArray()).toEqual([0, 3, 4, 8, 2, 1, 6, 5, 7]);
        expect(snapshot.edgeCount).toBe(14);
        expect(snapshot.selfLoopCount).toBe(2);
        expect(snapshot.flags.hasSelfLoops).toBe(true);
        for (let e = 0; e < edges.length; e++) {
            const list = snapshot.edgeList();
            expect(snapshot.ids.idOf(list.src[e])).toBe(edges[e].s);
            expect(snapshot.ids.idOf(list.dst[e])).toBe(edges[e].t);
            expect(snapshot.edges.value("label", e)).toBe(edges[e].label);
        }
        expect(snapshot.edges.value("label", 2)).toBe("S($end)");
        for (const id of [0, 3, 4, 8]) {
            expect(snapshot.nodes.value("shape", snapshot.ids.requireIndex(id))).toBe("doublecircle");
        }
        for (const id of [1, 2, 5, 6, 7]) {
            expect(snapshot.nodes.value("shape", snapshot.ids.requireIndex(id))).toBe("circle");
        }
        expect(snapshot.graph.value("rankdir", 0)).toBe("LR");
        expect(snapshot.selfLoopsAt(snapshot.ids.requireIndex(5))).toBe(1);
    });

    it("fdpclust: an undirected graph whose nested clusters become nested container nodes and edge endpoints", async () => {
        const text = readCorpusText("dot", "fdpclust.gv");
        expect(text).toMatch(/^graph G \{/);
        expect((text.match(/ -- /g) ?? []).length).toBe(6);
        const { snapshot } = await load("fdpclust.gv");
        expect(snapshot.directed).toBe(false);
        expect(snapshot.nodeCount).toBe(10);
        expect(snapshot.edgeCount).toBe(6);
        expect(snapshot.ids.toArray()).toEqual(["e", "clusterA", "a", "b", "clusterC", "C", "D", "clusterB", "d", "f"]);
        expect(edgeStrings(snapshot)).toEqual(["a->b", "C->D", "d->f", "d->D", "e->clusterB", "clusterC->clusterB"]);
        const parent = snapshot.nodes.byRole("parent");
        const at = (id: string): number => snapshot.ids.requireIndex(id);
        expect(parent?.value(at("a"))).toBe(at("clusterA"));
        expect(parent?.value(at("clusterC"))).toBe(at("clusterA"));
        expect(parent?.value(at("C"))).toBe(at("clusterC"));
        expect(parent?.value(at("D"))).toBe(at("clusterC"));
        expect(parent?.value(at("d"))).toBe(at("clusterB"));
        expect(parent?.isSet(at("e"))).toBe(false);
        expect(parent?.isSet(at("clusterA"))).toBe(false);
        expect(snapshot.nodes.require("graphty.cluster").nullCount).toBe(7);
        expect(snapshot.graph.value("layout", 0)).toBe("fdp");
    });

    it("hello.gv: the smallest gallery file", async () => {
        const { snapshot } = await load("hello.gv");
        expect(snapshot.ids.toArray()).toEqual(["Hello", "World"]);
        expect(edgeStrings(snapshot)).toEqual(["Hello->World"]);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta.name).toBe("G");
    });
});

describe("DOT quirks from DOTDataSource, the Graphviz grammar and research note 07", () => {
    it('treats a numeral and its quoted form as the same node (1 and "1"), and 007 / 1.5 as strings', async () => {
        const { snapshot } = await parse('digraph { 1 -> "1"; "1" -> 2; 2 -> "2"; 1.5 -> 007; -1 -> .5 }');
        expect(snapshot.ids.toArray()).toEqual([1, 2, "1.5", "007", -1, ".5"]);
        expect(edgeStrings(snapshot)).toEqual(["1->1", "1->2", "2->2", "1.5->007", "-1->.5"]);
        expect(snapshot.selfLoopCount).toBe(2);
    });

    it("strips // and /* */ comments and # preprocessor lines, but not // inside a quoted string", async () => {
        const { snapshot } = await parse(
            '#line 1 "foo.gv"\ndigraph {\n# another\na [URL="http://example.com/x"]; a -> /* c -> */ b // trailing\n}',
        );
        expect(snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.nodes.value("URL", 0)).toBe("http://example.com/x");
    });

    it('keeps HTML strings verbatim including the outer angle brackets, and concatenates "a" + "b"', async () => {
        const html = await parse("digraph { a [label=<<b>bold</b> &amp; <i>x // not a comment</i>>]; a -> b }");
        expect(html.snapshot.nodes.value("label", 0)).toBe("<<b>bold</b> &amp; <i>x // not a comment</i>>");
        const concat = await parse('digraph { a [label="foo" + "bar"]; a -> b }');
        expect(concat.snapshot.nodes.value("label", 0)).toBe("foobar");
    });

    it('decodes \\" inside quoted strings, keeps other escapes, and removes backslash-newline continuations', async () => {
        const escapes = await parse('digraph { a [label="say \\"hi\\"\\nnext \\N \\G"]; a -> b }');
        expect(escapes.snapshot.nodes.value("label", 0)).toBe('say "hi"\\nnext \\N \\G');
        const continuation = await parse('digraph { a [label="long \\\nline"]; a -> b }');
        expect(continuation.snapshot.nodes.value("label", 0)).toBe("long line");
    });

    it('DEFECT: a quoted string ending in an escaped backslash ("x\\\\") is valid DOT and must not be unterminated', async () => {
        // Graphviz's scanner (scan.l) consumes backslash pairs: <qstring>[\\][\\] addstr("\\\\"), so
        // "x\\" is the two-character string x\\ followed by the closing quote. The importer only pairs
        // \" and treats the second backslash as escaping the closing quote, then reports the string as
        // unterminated. `echo 'digraph { a [label="x\\"]; a -> b }' | dot -Tdot` accepts the file.
        const { snapshot, report } = await parse('digraph { a [label="x\\\\"]; a -> b }');
        expect(report.errorCount).toBe(0);
        expect(snapshot.nodes.value("label", 0)).toBe("x\\\\");
        expect(snapshot.edgeCount).toBe(1);
    });

    it("strips ports and compass points from endpoints into the port role columns, on plain and quoted ids", async () => {
        const { snapshot } = await parse('digraph { a:p1:n -> b:p2; c:s -> d; "n 0":f0 -> "n 1":f1:s }');
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "n 0", "n 1"]);
        const sp = snapshot.edges.byRole("sourcePort");
        const tp = snapshot.edges.byRole("targetPort");
        expect([0, 1, 2].map((e) => sp?.value(e))).toEqual(["p1:n", "s", "f0"]);
        expect([0, 1, 2].map((e) => tp?.value(e))).toEqual(["p2", undefined, "f1:s"]);
    });

    it("expands edge chains and subgraph endpoints as a cartesian product, with the subgraph's own edges", async () => {
        const chain = await parse("digraph { a -> {b c} -> d }");
        expect(edgeStrings(chain.snapshot)).toEqual(["a->b", "a->c", "b->d", "c->d"]);
        const both = await parse("digraph { {a b} -> {c d} }");
        expect(edgeStrings(both.snapshot)).toEqual(["a->c", "a->d", "b->c", "b->d"]);
        const nested = await parse("digraph { a -> subgraph s { b -> c } }");
        expect(edgeStrings(nested.snapshot).sort()).toEqual(["a->b", "a->c", "b->c"]);
        const undirected = await parse("graph { a -- b -- c }");
        expect(undirected.snapshot.directed).toBe(false);
        expect(edgeStrings(undirected.snapshot)).toEqual(["a->b", "b->c"]);
    });

    it("applies node / edge / graph defaults statefully and scoped to the enclosing subgraph (DOTDataSource ignored them)", async () => {
        const { snapshot } = await parse(
            "digraph { a; node [color=red]; b; subgraph s { node [color=blue]; c; } d; edge [w=2]; a -> b; }",
        );
        expect(snapshot.nodes.isSet("color", 0)).toBe(false);
        expect(snapshot.nodes.value("color", 1)).toBe("red");
        expect(snapshot.nodes.value("color", 2)).toBe("blue");
        expect(snapshot.nodes.value("color", 3)).toBe("red");
        expect(snapshot.edges.value("w", 0)).toBe(2);
        const later = await parse("digraph { a -> b; a [color=red] }");
        expect(later.snapshot.nodes.value("color", 0)).toBe("red");
    });

    it("reads keywords case-insensitively and keyword-named quoted ids as plain ids", async () => {
        const { snapshot } = await parse(
            'DiGraph G { Node [shape=box]; a -> b; "node" -> "edge"; "graph" -> "digraph" }',
        );
        expect(snapshot.meta.name).toBe("G");
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "node", "edge", "graph", "digraph"]);
        expect(snapshot.nodes.value("shape", 5)).toBe("box");
    });

    it("merges parallel edges of a strict graph with a warning and keeps them otherwise", async () => {
        const strict = await parse("strict digraph { a -> b; a -> b; a -> a; a -> a }");
        expect(strict.snapshot.edgeCount).toBe(2);
        expect(strict.snapshot.flags.multigraph).toBe(false);
        expect(strict.report.issues.filter((i) => i.code === "W_DOT_STRICT_MERGED")).toHaveLength(2);
        const strictUndirected = await parse("strict graph { a -- b; b -- a }");
        expect(strictUndirected.snapshot.edgeCount).toBe(1);
        const loose = await parse("graph { a -- b; b -- a }");
        expect(loose.snapshot.edgeCount).toBe(2);
        expect(loose.snapshot.flags.multigraph).toBe(true);
    });

    it("reports an edge operator that contradicts the graph keyword and resolves it per the option", async () => {
        const op = await parse("digraph { a -- b }");
        expect(op.report.issues.map((i) => i.code)).toContain("W_DOT_EDGE_OPERATOR");
        expect(op.snapshot.edgeCount).toBe(2);
        expect(op.snapshot.edges.byRole("directed")?.value(0)).toBe(false);
        const header = await parse("digraph { a -- b }", { mismatchedEdgeOperator: "header" });
        expect(header.snapshot.edgeCount).toBe(1);
        await expect(parse("digraph { a -- b }", { mismatchedEdgeOperator: "error" })).rejects.toMatchObject({
            code: "E_IMPORT",
        });
    });

    it("reads a cluster's own attributes onto the container node and drops non-cluster subgraph attributes with a warning", async () => {
        const cluster = await parse('digraph { subgraph cluster_0 { label="L"; color=red; a } b }');
        expect(cluster.snapshot.ids.toArray()).toEqual(["cluster_0", "a", "b"]);
        expect(cluster.snapshot.nodes.value("label", 0)).toBe("L");
        expect(cluster.snapshot.nodes.value("color", 0)).toBe("red");
        expect(cluster.snapshot.nodes.byRole("parent")?.value(1)).toBe(0);
        const rank = await parse("digraph { { rank=same; a; b } a -> b }");
        expect(rank.snapshot.nodeCount).toBe(2);
        expect(rank.report.issues.map((i) => i.code)).toContain("W_DOT_SUBGRAPH_ATTRIBUTES_DROPPED");
        const named = await parse("digraph { subgraph s { a; b } }");
        expect(named.snapshot.ids.toArray()).toEqual(["a", "b"]);
    });

    it("reads graph-level ID = ID statements into the graph table, multiple attribute lists, and ; or , separators", async () => {
        const { snapshot } = await parse(
            'digraph { rankdir=LR; a [x=1] [y=2]; a -> b [p=1][q=2]; b [m=1; n=2,o=3]; label="G" }',
        );
        expect(snapshot.graph.value("rankdir", 0)).toBe("LR");
        expect(snapshot.graph.value("label", 0)).toBe("G");
        expect([snapshot.nodes.value("x", 0), snapshot.nodes.value("y", 0)]).toEqual([1, 2]);
        expect([snapshot.edges.value("p", 0), snapshot.edges.value("q", 0)]).toEqual([1, 2]);
        expect(["m", "n", "o"].map((name) => snapshot.nodes.value(name, 1))).toEqual([1, 2, 3]);
    });

    it("reads the weight attribute as the edge weight (design 8.4 weightFrom default) and nothing else as one", async () => {
        const { snapshot } = await parse(
            'digraph { a -> b [weight=2, label="x"]; b -> c [weight=0.5]; c -> d [value=7] }',
        );
        expect(snapshot.flags.weighted).toBe(true);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2, 0.5, 1]);
        expect(snapshot.edges.has("weight")).toBe(false);
        expect(snapshot.edges.value("label", 0)).toBe("x");
        expect(snapshot.edges.value("value", 2)).toBe(7);
        const unweighted = await parse("digraph { a -> b [weight=2] }", { weightFrom: null });
        expect(unweighted.snapshot.flags.weighted).toBe(false);
        expect(unweighted.snapshot.edges.value("weight", 0)).toBe(2);
    });

    it("reads a BOM, CRLF, unicode ids and an empty graph, and warns about a second graph in the input", async () => {
        const bom = await parse(`${BOM}digraph {\r\n a -> b\r\n}\r\n`);
        expect(edgeStrings(bom.snapshot)).toEqual(["a->b"]);
        const cafe = `caf${String.fromCharCode(0xe9)}`;
        const unicode = await parse(`digraph { "${cafe}" -> ${cafe} }`);
        expect(unicode.snapshot.selfLoopCount).toBe(1);
        expect(unicode.snapshot.ids.toArray()).toEqual([cafe]);
        const empty = await parse("digraph {}");
        expect(empty.snapshot.nodeCount).toBe(0);
        const two = await parse("digraph A { a -> b } digraph B { c -> d }");
        expect(two.snapshot.meta.name).toBe("A");
        expect(two.snapshot.nodeCount).toBe(2);
        expect(two.report.issues.map((i) => i.code)).toContain("W_MULTIPLE_GRAPHS");
    });

    it("refuses an attribute without a value and an unterminated string as Graphviz does", async () => {
        await expect(parse("digraph { a [color]; a -> b }")).rejects.toMatchObject({ code: "E_IMPORT" });
        await expect(parse('digraph { a [label="open]; a -> b }')).rejects.toMatchObject({ code: "E_IMPORT" });
    });

    it("DEFECT: a badly delimited numeral such as 1e3 is split into two tokens silently; Graphviz warns", async () => {
        // dot: "Warning: syntax ambiguity - badly delimited number '1e' in line 1 ... splits into two tokens".
        const { snapshot, report } = await parse("digraph { 1e3 -> x }");
        expect(snapshot.ids.toArray()).toEqual([1, "e3", "x"]);
        expect(report.warningCount).toBeGreaterThan(0);
    });
});
