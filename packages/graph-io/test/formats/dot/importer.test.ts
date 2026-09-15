import { type Column, GraphBuilder, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DIRECTION_REFUSED_CODE, PAIR_COLUMN } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import { DOT_ISSUE, dotImporter } from "../../../src/formats/dot/importer.js";
import {
    CLUSTER_COLUMN,
    PARENT_COLUMN,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
} from "../../../src/formats/dot/names.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../../src/types.js";
import {
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
} from "../../helpers/corpus.js";
import { expectSameSnapshot } from "../../helpers/roundtrip.js";

type Options = Parameters<typeof dotImporter.import>[2];

async function load(
    input: string | Uint8Array | ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>,
    options?: Options,
    builder?: GraphBuilder,
): Promise<{ snapshot: GraphSnapshot; report: ImportReport; builder: GraphBuilder }> {
    const b = builder ?? new GraphBuilder({ directed: true, weightDtype: "f64" });
    const report = await dotImporter.import(input, b, options);
    return { snapshot: b.freeze(), report, builder: b };
}

async function failure(input: string | Uint8Array, options?: Options): Promise<ImportError> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        await dotImporter.import(input, b, options);
    } catch (err) {
        if (err instanceof ImportError) {
            return err;
        }
        throw err;
    }
    throw new Error("expected an ImportError");
}

function cell(snapshot: GraphSnapshot, table: "nodes" | "edges" | "graph", name: string, row: number): unknown {
    const column = snapshot[table].get(name);
    if (column === null) {
        return null;
    }
    return column.isSet(row) ? column.value(row) : undefined;
}

function nodeCell(snapshot: GraphSnapshot, id: string | number, name: string): unknown {
    return cell(snapshot, "nodes", name, snapshot.ids.requireIndex(id));
}

function edgeIndex(snapshot: GraphSnapshot, source: string | number, target: string | number): number {
    const list = snapshot.edgeList();
    const u = snapshot.ids.requireIndex(source);
    const v = snapshot.ids.requireIndex(target);
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (list.src[e] === u && list.dst[e] === v) {
            return e;
        }
    }
    throw new Error(`no edge ${String(source)} -> ${String(target)}`);
}

function edgePairs(snapshot: GraphSnapshot): string[] {
    const list = snapshot.edgeList();
    const out: string[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        out.push(`${String(snapshot.ids.idOf(list.src[e]))}>${String(snapshot.ids.idOf(list.dst[e]))}`);
    }
    return out;
}

function parents(snapshot: GraphSnapshot): Record<string, string | undefined> {
    const column = snapshot.nodes.get(PARENT_COLUMN);
    const out: Record<string, string | undefined> = {};
    for (let i = 0; i < snapshot.nodeCount; i++) {
        const p = column !== null && column.isSet(i) ? (column.value(i) as number) : undefined;
        out[String(snapshot.ids.idOf(i))] = p === undefined ? undefined : String(snapshot.ids.idOf(p));
    }
    return out;
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

function dtypeOf(column: Column | null): string | null {
    return column === null ? null : column.dtype;
}

/** Container nodes the importer synthesises per corpus file, on top of the manifest's node count. */
const CONTAINERS: Record<string, string[]> = {
    "hello.gv": [],
    "cluster.gv": ["cluster_0", "cluster_1"],
    "fsm.gv": [],
    "datastruct.gv": [],
    "fdpclust.gv": ["clusterA", "clusterC", "clusterB"],
    "root.gv": [],
};
const DIRECTED: Record<string, boolean> = {
    "hello.gv": true,
    "cluster.gv": true,
    "fsm.gv": true,
    "datastruct.gv": true,
    "fdpclust.gv": false,
    "root.gv": true,
};

describe("dot importer: the corpus", () => {
    for (const file of corpusFiles("dot")) {
        it(`imports ${file.path} with the manifest's counts`, async () => {
            const { snapshot, report } = await load(readCorpusText("dot", file.path));
            const containers = CONTAINERS[file.path];
            expect(snapshot.nodeCount).toBe(file.expectedNodes + containers.length);
            expect(snapshot.edgeCount).toBe(file.expectedEdges);
            expect(snapshot.directed).toBe(DIRECTED[file.path]);
            expect(report.format).toBe("dot");
            expect(report.errorCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(report.counts.nodes).toBe(snapshot.nodeCount);
            expect(report.counts.edges).toBe(snapshot.edgeCount);
            expect(report.counts.skippedNodes).toBe(0);
            expect(report.counts.skippedEdges).toBe(0);
            expect(report.durationMs).toBeGreaterThanOrEqual(0);
            for (const name of containers) {
                expect(nodeCell(snapshot, name, CLUSTER_COLUMN)).toBe(true);
            }
            expect(snapshot.meta.sourceFormat).toBe("dot");
        });
    }

    it("hello.gv: two nodes and one edge in mention order", async () => {
        const { snapshot } = await load(readCorpusText("dot", "hello.gv"));
        expect(snapshot.ids.toArray()).toEqual(["Hello", "World"]);
        expect(edgePairs(snapshot)).toEqual(["Hello>World"]);
        expect(snapshot.meta.name).toBe("G");
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.flags.weighted).toBe(false);
    });

    it("cluster.gv: clusters become container nodes with parents, scoped defaults apply at creation", async () => {
        const { snapshot, report } = await load(readCorpusText("dot", "cluster.gv"));
        expect(report.warningCount).toBe(0);
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
        const p = parents(snapshot);
        expect(p.a0).toBe("cluster_0");
        expect(p.a3).toBe("cluster_0");
        expect(p.b2).toBe("cluster_1");
        expect(p.start).toBeUndefined();
        expect(p.cluster_0).toBeUndefined();
        expect(snapshot.nodes.get(PARENT_COLUMN)?.meta.role).toBe("parent");
        expect(snapshot.nodes.get(PARENT_COLUMN)?.meta.refersTo).toBe("node");
        expect(nodeCell(snapshot, "cluster_0", "label")).toBe("process #1");
        expect(nodeCell(snapshot, "cluster_0", "style")).toBe("filled");
        expect(nodeCell(snapshot, "cluster_0", "color")).toBe("lightgrey");
        expect(nodeCell(snapshot, "cluster_1", "color")).toBe("blue");
        expect(nodeCell(snapshot, "a0", "style")).toBe("filled");
        expect(nodeCell(snapshot, "a0", "color")).toBe("white");
        expect(nodeCell(snapshot, "b0", "style")).toBe("filled");
        expect(nodeCell(snapshot, "b0", "color")).toBeUndefined();
        expect(nodeCell(snapshot, "start", "shape")).toBe("Mdiamond");
        expect(nodeCell(snapshot, "start", "fontname")).toBe("Helvetica,Arial,sans-serif");
        expect(cell(snapshot, "graph", "fontname", 0)).toBe("Helvetica,Arial,sans-serif");
        expect(cell(snapshot, "edges", "fontname", edgeIndex(snapshot, "start", "a0"))).toBe(
            "Helvetica,Arial,sans-serif",
        );
        expect(snapshot.nodes.get("label")?.meta.role).toBe("label");
        expect(edgePairs(snapshot).slice(0, 4)).toEqual(["a0>a1", "a1>a2", "a2>a3", "b0>b1"]);
    });

    it("fsm.gv: numeric ids, self-loops, edge labels and the rankdir graph attribute", async () => {
        const { snapshot } = await load(readCorpusText("dot", "fsm.gv"));
        expect(snapshot.ids.toArray()).toEqual([0, 3, 4, 8, 2, 1, 6, 5, 7]);
        expect(snapshot.selfLoopCount).toBe(2);
        expect(nodeCell(snapshot, 0, "shape")).toBe("doublecircle");
        expect(nodeCell(snapshot, 2, "shape")).toBe("circle");
        expect(cell(snapshot, "edges", "label", edgeIndex(snapshot, 0, 2))).toBe("SS(B)");
        expect(cell(snapshot, "edges", "label", edgeIndex(snapshot, 1, 3))).toBe("S($end)");
        expect(snapshot.edges.get("label")?.meta.role).toBe("label");
        expect(cell(snapshot, "graph", "rankdir", 0)).toBe("LR");
        expect(snapshot.meta.name).toBe("finite_state_machine");
    });

    it("datastruct.gv: quoted ids, ports kept in the port columns, inferred fontsize", async () => {
        const { snapshot, report } = await load(readCorpusText("dot", "datastruct.gv"));
        expect(report.warningCount).toBe(0);
        expect(snapshot.ids.idOf(0)).toBe("node0");
        expect(dtypeOf(snapshot.nodes.get("fontsize"))).toBe("i32");
        expect(nodeCell(snapshot, "node0", "fontsize")).toBe(16);
        expect(nodeCell(snapshot, "node3", "label")).toBe("<f0> 3.43322790286038071e-06|44.79998779296875|0");
        const e = edgeIndex(snapshot, "node0", "node2");
        expect(cell(snapshot, "edges", SOURCE_PORT_COLUMN, e)).toBe("f1");
        expect(cell(snapshot, "edges", TARGET_PORT_COLUMN, e)).toBe("f0");
        expect(snapshot.edges.get(SOURCE_PORT_COLUMN)?.meta.role).toBe("sourcePort");
        expect(snapshot.edges.get(TARGET_PORT_COLUMN)?.meta.role).toBe("targetPort");
        expect(dtypeOf(snapshot.edges.get("id"))).toBe("i32");
        expect(cell(snapshot, "edges", "id", e)).toBe(1);
        expect(snapshot.edges.get("id")?.meta.role).toBeNull();
    });

    it("fdpclust.gv: an undirected graph with nested clusters and edges to cluster names", async () => {
        const { snapshot, report } = await load(readCorpusText("dot", "fdpclust.gv"));
        expect(snapshot.directed).toBe(false);
        expect(report.warningCount).toBe(0);
        expect(snapshot.ids.toArray()).toEqual(["e", "clusterA", "a", "b", "clusterC", "C", "D", "clusterB", "d", "f"]);
        const p = parents(snapshot);
        expect(p.e).toBeUndefined();
        expect(p.a).toBe("clusterA");
        expect(p.C).toBe("clusterC");
        expect(p.clusterC).toBe("clusterA");
        expect(p.clusterA).toBeUndefined();
        expect(p.d).toBe("clusterB");
        expect(edgePairs(snapshot)).toEqual(["a>b", "C>D", "d>f", "d>D", "e>clusterB", "clusterC>clusterB"]);
        expect(cell(snapshot, "graph", "layout", 0)).toBe("fdp");
    });

    it("root.gv: a large file with quoted and numeric ids and typed graph attributes", async () => {
        const { snapshot } = await load(readCorpusText("dot", "root.gv"));
        expect(snapshot.ids.idOf(0)).toBe(1);
        expect(snapshot.ids.idOf(1)).toBe("189E");
        expect(nodeCell(snapshot, 1, "label")).toBe("02f5daf56e299b8a8ecea892");
        expect(nodeCell(snapshot, 1, "shape")).toBe("hexagon");
        expect(nodeCell(snapshot, "189E", "color")).toBe("blue");
        expect(nodeCell(snapshot, "199E", "label")).toBeUndefined();
        expect(cell(snapshot, "graph", "ranksep", 0)).toBe(3);
        expect(cell(snapshot, "graph", "root", 0)).toBe("189E");
        expect(cell(snapshot, "graph", "overlap", 0)).toBe("prism");
        expect(snapshot.meta.name).toBe("G_component_0");
    });

    it("reads the same snapshot from every input shape", async () => {
        const bytes = readCorpusBytes("dot", "cluster.gv");
        const reference = (await load(new TextDecoder().decode(bytes))).snapshot;
        for (const shape of inputShapes(bytes)) {
            const { snapshot } = await load(shape.make());
            expectSameSnapshot(reference, snapshot);
        }
    });
});

describe("dot importer: malformed corpus", () => {
    /** Files Graphviz itself accepts (verified with `dot -Tcanon`): they must import, not fail. */
    const VALID_FOR_GRAPHVIZ: Record<string, { nodes: number; edges: number }> = {
        "invalid-keyword.gv": { nodes: 3, edges: 1 },
        "missing-arrow.gv": { nodes: 4, edges: 0 },
    };

    for (const name of malformedFiles("dot")) {
        const valid = VALID_FOR_GRAPHVIZ[name];
        if (valid !== undefined) {
            it(`${name} is valid DOT (as for Graphviz) and imports`, async () => {
                const { snapshot, report } = await load(readMalformedBytes("dot", name));
                expect(snapshot.nodeCount).toBe(valid.nodes);
                expect(snapshot.edgeCount).toBe(valid.edges);
                expect(report.errorCount).toBe(0);
            });
            continue;
        }
        it(`${name} throws ImportError carrying a report`, async () => {
            const err = await failure(readMalformedBytes("dot", name));
            expect(err.code).toBe("E_IMPORT");
            expect(err.report.format).toBe("dot");
            expect(err.report.errorCount).toBeGreaterThan(0);
            const fatal = err.report.issues.find((i) => i.severity === "error");
            expect(fatal).toBeDefined();
            expect([DOT_ISSUE.SYNTAX, DOT_ISSUE.EMPTY_INPUT]).toContain(fatal?.code);
            expect(fatal?.category).toBe("parse-error");
        });
    }

    it("invalid-keyword.gv reads the bare id as a node and the assignment as a graph attribute", async () => {
        const { snapshot } = await load(readMalformedBytes("dot", "invalid-keyword.gv"));
        expect(snapshot.ids.toArray()).toEqual(["invalid_keyword", "A", "B"]);
        expect(cell(snapshot, "graph", "notavalidstatement", 0)).toBe("broken");
    });

    it("reports the line of a syntax error and keeps the partial counts", async () => {
        const err = await failure("digraph G {\n  a -> b;\n  c [label=];\n}\n");
        expect(err.report.issues[0].line).toBe(3);
        expect(err.report.issues[0].code).toBe(DOT_ISSUE.SYNTAX);
        expect(err.report.counts.nodes).toBe(2);
        expect(err.report.counts.edges).toBe(1);
        expect(err.details.line).toBe(3);
    });

    it("rejects an unterminated block comment, a bad character and a lone dash", async () => {
        expect((await failure("digraph { /* never closed")).report.issues[0].code).toBe(DOT_ISSUE.SYNTAX);
        expect((await failure("digraph { a @ b }")).report.issues[0].message).toContain("unexpected character");
        expect((await failure("digraph { a - b }")).report.issues[0].message).toContain("unexpected '-'");
        expect((await failure("digraph { a -> }")).report.issues[0].message).toContain("expected an identifier");
        expect((await failure("digraph { a [label=<<b>x</b>] }")).report.issues[0].message).toContain(
            "unterminated HTML",
        );
        expect((await failure("digraph { node ; }")).report.issues[0].message).toContain('expected "["');
        expect((await failure("digraph { subgraph x a }")).report.issues[0].message).toContain('expected "{"');
        expect((await failure('digraph { "a" + b }')).report.issues[0].message).toContain("quoted string after");
        expect((await failure("digraph { strict }")).report.issues[0].message).toContain("unexpected keyword");
    });

    it("rejects an input without a graph keyword or header brace", async () => {
        expect((await failure("   \n// only a comment\n")).report.issues[0].code).toBe(DOT_ISSUE.EMPTY_INPUT);
        expect((await failure("graphs { }")).report.issues[0].message).toContain('expected "graph" or "digraph"');
        expect((await failure("digraph G [")).report.issues[0].message).toContain('expected "{"');
    });

    it("rejects invalid UTF-8 as a fatal parse error", async () => {
        const err = await failure(new Uint8Array([0x64, 0x69, 0x67, 0x72, 0x61, 0x70, 0x68, 0x20, 0xff, 0x7b, 0x7d]));
        expect(err.report.issues[0].code).toBe(INVALID_UTF8_CODE);
    });
});

describe("dot importer: the grammar", () => {
    it("strips the three comment forms and a BOM, and treats keywords case-insensitively", async () => {
        const text = `${String.fromCharCode(0xfeff)}# preprocessor line\n/* block\n comment */ Strict DiGraph "my graph" { // trailing\n a -> b // comment\n /* c -> d */ }`;
        const { snapshot, report } = await load(text);
        expect(report.warningCount).toBe(0);
        expect(snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(snapshot.meta.name).toBe("my graph");
        expect(snapshot.meta.extra).toEqual({ dot: { strict: true } });
    });

    it("reads every id spelling: bare, numeral, quoted with escapes and continuations, concatenated, HTML", async () => {
        const text =
            'digraph {\n  x_1; 1.5; -3; .5; "quoted \\"id\\"" ; "line\\\ncontinued"; "a" + "b" + "c"; <<b>html</b>>; "node"; "" ; "back\\\\slash";\n}';
        const { snapshot } = await load(text);
        expect(snapshot.ids.toArray()).toEqual([
            "x_1",
            "1.5",
            -3,
            ".5",
            'quoted "id"',
            "linecontinued",
            "abc",
            "<<b>html</b>>",
            "node",
            "",
            "back\\\\slash",
        ]);
    });

    it("keeps raw line breaks inside quoted strings and counts lines through them", async () => {
        const err = await failure('digraph {\n a [label="two\nlines"];\n b [x=];\n}');
        expect(err.report.issues[0].line).toBe(4);
    });

    it("expands chains and subgraph endpoints in cgraph order", async () => {
        const { snapshot } = await load("digraph { a -> b -> c; {d e} -> {f g}; h -> subgraph s { i j } -> k; }");
        expect(edgePairs(snapshot)).toEqual(["a>b", "b>c", "d>f", "d>g", "e>f", "e>g", "h>i", "h>j", "i>k", "j>k"]);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]);
    });

    it("applies edge attributes to every edge of a chain and accepts every a_list separator", async () => {
        const { snapshot } = await load("digraph { a -> b -> c [color=red; w=2, x=1][y=true] }");
        for (const e of [0, 1]) {
            expect(cell(snapshot, "edges", "color", e)).toBe("red");
            expect(cell(snapshot, "edges", "w", e)).toBe(2);
            expect(cell(snapshot, "edges", "x", e)).toBe(1);
            expect(cell(snapshot, "edges", "y", e)).toBe(true);
        }
        expect(dtypeOf(snapshot.edges.get("y"))).toBe("bool");
    });

    it("applies node and edge defaults to elements created afterwards, scoped to the subgraph", async () => {
        const text = `digraph {
            before;
            node [shape=box, color=red];
            after [color=blue];
            subgraph { node [shape=circle]; inner; }
            outer;
            edge [penwidth=2];
            before -> after;
            after -> outer [penwidth=3];
        }`;
        const { snapshot } = await load(text);
        expect(nodeCell(snapshot, "before", "shape")).toBeUndefined();
        expect(nodeCell(snapshot, "after", "shape")).toBe("box");
        expect(nodeCell(snapshot, "after", "color")).toBe("blue");
        expect(nodeCell(snapshot, "inner", "shape")).toBe("circle");
        expect(nodeCell(snapshot, "inner", "color")).toBe("red");
        expect(nodeCell(snapshot, "outer", "shape")).toBe("box");
        expect(cell(snapshot, "edges", "penwidth", 0)).toBe(2);
        expect(cell(snapshot, "edges", "penwidth", 1)).toBe(3);
    });

    it("does not re-apply defaults to a node mentioned again", async () => {
        const { snapshot } = await load("digraph { a; node [c=1]; a; b; a -> b }");
        expect(nodeCell(snapshot, "a", "c")).toBeUndefined();
        expect(nodeCell(snapshot, "b", "c")).toBe(1);
    });

    it("infers attribute values per column under the 5.1 grammar, labels stay text", async () => {
        const text =
            'digraph { a [n=1, f="2.5", b=true, s=abc, label=42, e=""]; b [n="01", f=3, b=false, label="x"]; }';
        const { snapshot } = await load(text);
        expect(dtypeOf(snapshot.nodes.get("n"))).toBe("string");
        expect(nodeCell(snapshot, "a", "n")).toBe("1");
        expect(dtypeOf(snapshot.nodes.get("f"))).toBe("f64");
        expect(nodeCell(snapshot, "b", "f")).toBe(3);
        expect(dtypeOf(snapshot.nodes.get("b"))).toBe("bool");
        expect(dtypeOf(snapshot.nodes.get("s"))).toBe("string");
        expect(dtypeOf(snapshot.nodes.get("label"))).toBe("string");
        expect(nodeCell(snapshot, "a", "label")).toBe("42");
        expect(nodeCell(snapshot, "a", "e")).toBe("");
    });

    it("reads graph attributes from ID = ID and graph [..] at the root", async () => {
        const { snapshot } = await load('digraph { rankdir=LR; graph [ranksep=3, label="G label"]; a }');
        expect(cell(snapshot, "graph", "rankdir", 0)).toBe("LR");
        expect(cell(snapshot, "graph", "ranksep", 0)).toBe(3);
        expect(cell(snapshot, "graph", "label", 0)).toBe("G label");
        expect(snapshot.graph.get("label")?.dtype).toBe("string");
    });

    it("reports attributes of a subgraph that is not a cluster and keeps its nodes", async () => {
        const { snapshot, report } = await load("digraph { subgraph s { rank=same; a; b } { c } }");
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(codes(report)).toEqual([DOT_ISSUE.SUBGRAPH_ATTRIBUTES_DROPPED]);
        expect(report.issues[0].element).toBe("s");
        expect(report.issues[0].message).toContain("rank");
        expect(snapshot.nodes.get(PARENT_COLUMN)).toBeNull();
    });

    it("turns a subgraph with cluster=true into a container even when the flag follows its members", async () => {
        const { snapshot, report } = await load("digraph { subgraph s { a; b; cluster=true; label=L } }");
        expect(report.warningCount).toBe(0);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "s"]);
        expect(parents(snapshot)).toEqual({ a: "s", b: "s", s: undefined });
        expect(nodeCell(snapshot, "s", CLUSTER_COLUMN)).toBe(true);
        expect(nodeCell(snapshot, "s", "label")).toBe("L");
        expect(snapshot.nodes.get("cluster")).toBeNull();
    });

    it("merges a plain node and a cluster of the same name with a warning", async () => {
        const { snapshot, report } = await load("digraph { x -> cluster_0; subgraph cluster_0 { a } }");
        expect(codes(report)).toEqual([DOT_ISSUE.CLUSTER_NODE_MERGED]);
        expect(snapshot.nodeCount).toBe(3);
        expect(nodeCell(snapshot, "cluster_0", CLUSTER_COLUMN)).toBe(true);
        expect(parents(snapshot).a).toBe("cluster_0");
        expect(edgePairs(snapshot)).toEqual(["x>cluster_0"]);
    });

    it("keeps the first cluster of a node mentioned in two unrelated clusters", async () => {
        const { snapshot, report } = await load(
            "digraph { subgraph cluster_a { n } subgraph cluster_b { n; m } subgraph cluster_a { subgraph cluster_c { n } } }",
        );
        expect(codes(report)).toEqual([DOT_ISSUE.CLUSTER_CONFLICT]);
        expect(report.issues[0].element).toBe("n");
        expect(parents(snapshot).n).toBe("cluster_a");
        expect(parents(snapshot).m).toBe("cluster_b");
        expect(parents(snapshot).cluster_c).toBe("cluster_a");
    });

    it("nests clusters and makes the members of a cluster used as an endpoint the edge targets", async () => {
        const { snapshot } = await load("digraph { x -> subgraph cluster_k { subgraph cluster_j { p } q } }");
        expect(edgePairs(snapshot)).toEqual(["x>p", "x>q"]);
        const p = parents(snapshot);
        expect(p.p).toBe("cluster_j");
        expect(p.cluster_j).toBe("cluster_k");
        expect(p.q).toBe("cluster_k");
    });

    it("keeps ports of edge endpoints and drops a port on a node statement with a warning", async () => {
        const { snapshot, report } = await load('digraph { a:p1:ne -> b:s; c:"x y" -> d; e:n [x=1] }');
        expect(codes(report)).toEqual([DOT_ISSUE.NODE_PORT_DROPPED]);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(cell(snapshot, "edges", SOURCE_PORT_COLUMN, 0)).toBe("p1:ne");
        expect(cell(snapshot, "edges", TARGET_PORT_COLUMN, 0)).toBe("s");
        expect(cell(snapshot, "edges", SOURCE_PORT_COLUMN, 1)).toBe("x y");
        expect(cell(snapshot, "edges", TARGET_PORT_COLUMN, 1)).toBeUndefined();
        expect(nodeCell(snapshot, "e", "x")).toBe(1);
    });

    it("maps a node pos to the position column and a ! suffix to pin", async () => {
        const { snapshot, report } = await load(
            'digraph { a [pos="1.5,2"]; b [pos="3,4,5!"]; c [pos="nope"]; a -> b [pos="e,1,2 3,4"] }',
        );
        expect(codes(report)).toEqual([DOT_ISSUE.BAD_POS]);
        const position = snapshot.nodes.byRole("position");
        expect(position?.meta.name).toBe("pos");
        expect(position?.dtype).toBe("f32");
        expect(position?.meta.components).toBe(3);
        expect(position?.meta.mutable).toBe(true);
        expect(position?.meta.extra).toEqual({ sourceDims: 2, units: "file" });
        expect(Array.from(position?.value(0) as ArrayLike<number>)).toEqual([1.5, 2, 0]);
        expect(Array.from(position?.value(1) as ArrayLike<number>)).toEqual([3, 4, 5]);
        expect(position?.isSet(2)).toBe(false);
        expect(nodeCell(snapshot, "b", "pin")).toBe(true);
        expect(nodeCell(snapshot, "a", "pin")).toBeUndefined();
        expect(cell(snapshot, "edges", "pos", 0)).toBe("e,1,2 3,4");
    });

    it("takes the weight from the weight attribute, explicit only", async () => {
        const { snapshot } = await load('digraph { a -> b [weight=2.5]; b -> c; c -> a [weight=""] }');
        expect(snapshot.flags.weighted).toBe(true);
        const list = snapshot.edgeList();
        expect(Array.from(list.weights ?? [])).toEqual([2.5, 1, 1]);
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.isSet(2)).toBe(false);
        expect(snapshot.edges.get("weight")).toBeNull();
    });

    it("honours weightFrom and weightFrom: null", async () => {
        const custom = await load("digraph { a -> b [w=3, weight=7] }", { weightFrom: "w" });
        expect(Array.from(custom.snapshot.edgeList().weights ?? [])).toEqual([3]);
        expect(cell(custom.snapshot, "edges", "weight", 0)).toBe(7);
        const none = await load("digraph { a -> b [weight=7] }", { weightFrom: null });
        expect(none.snapshot.flags.weighted).toBe(false);
        expect(cell(none.snapshot, "edges", "weight", 0)).toBe(7);
    });

    it("records an invalid weight as an error and skips the edge", async () => {
        const { snapshot, report } = await load("digraph { a -> b [weight=heavy]; b -> c [weight=NaN]; c -> a }");
        expect(snapshot.edgeCount).toBe(1);
        expect(report.errorCount).toBe(2);
        expect(codes(report)).toEqual(["E_INVALID_WEIGHT", "E_INVALID_WEIGHT"]);
        expect(report.issues[0].category).toBe("validation-error");
        expect(report.issues[0].line).toBe(1);
        expect(report.issues[0].element).toBe("a -> b");
        expect(report.counts.skippedEdges).toBe(2);
        expect(report.counts.edges).toBe(1);
    });

    it("aborts with a truncated report beyond the error limit", async () => {
        const err = await failure("digraph { a -> b [weight=x]; b -> c [weight=y]; c -> d [weight=z] }", {
            errorLimit: 1,
        });
        expect(err.report.truncated).toBe(true);
        expect(err.report.errorCount).toBe(2);
        expect(err.report.counts.skippedEdges).toBe(1);
    });

    it("merges parallel edges of a strict graph, attributes and weight overriding", async () => {
        const text =
            "strict graph { a -- b [color=red, weight=2]; b -- a [color=blue]; a -- b [weight=5]; a -- a; a -- a }";
        const { snapshot, report } = await load(text);
        expect(snapshot.edgeCount).toBe(2);
        expect(codes(report)).toEqual([DOT_ISSUE.STRICT_MERGED, DOT_ISSUE.STRICT_MERGED, DOT_ISSUE.STRICT_MERGED]);
        expect(report.issues[0].category).toBe("merged");
        expect(cell(snapshot, "edges", "color", 0)).toBe("blue");
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([5, 1]);
        expect(report.counts.edges).toBe(2);
    });

    it("merges edges with the same endpoints and key, keeping key as the edge id", async () => {
        const { snapshot, report } = await load(
            "digraph { a -> b [key=k, c=1]; a -> b [key=k, c=2]; a -> b [key=other]; b -> a [key=k]; a -> b }",
        );
        expect(snapshot.edgeCount).toBe(4);
        expect(codes(report)).toEqual([DOT_ISSUE.KEY_MERGED]);
        expect(cell(snapshot, "edges", "c", 0)).toBe(2);
        const key = snapshot.edges.byRole("id");
        expect(key?.meta.name).toBe("key");
        expect(key?.dtype).toBe("string");
        expect(key?.value(0)).toBe("k");
        expect(key?.value(1)).toBe("other");
        expect(key?.isSet(3)).toBe(false);
    });

    it("coerces ids canonically by default and per the ids option", async () => {
        const canonical = await load('digraph { 1 -> "1"; "01" -> 2; "-0" -> "1.0" }');
        expect(canonical.snapshot.ids.toArray()).toEqual([1, "01", 2, "-0", "1.0"]);
        expect(canonical.snapshot.edgeCount).toBe(3);
        expect(canonical.snapshot.selfLoopCount).toBe(1);
        const strings = await load("digraph { 1 -> 2 }", { ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["1", "2"]);
        const numbers = await load('digraph { 1 -> "01"; "1.0" -> 2 }', { ids: "number" });
        expect(numbers.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(codes(numbers.report)).toEqual([DOT_ISSUE.ID_MERGED, DOT_ISSUE.ID_MERGED]);
        expect(numbers.report.issues[0].category).toBe("coercion");
        const bad = await load("digraph { x -> 1 }", { ids: "number" });
        expect(codes(bad.report)).toEqual(["E_INVALID_ID"]);
        expect(bad.report.counts.skippedNodes).toBe(1);
        expect(bad.snapshot.nodeCount).toBe(1);
        expect(bad.snapshot.edgeCount).toBe(0);
    });

    it("reads a contradicting edge operator with the operator's direction and expands it", async () => {
        const { snapshot, report } = await load("digraph { a -> b; c -- d }");
        expect(codes(report)).toEqual([DOT_ISSUE.EDGE_OPERATOR]);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(3);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.counts.edges).toBe(3);
        expect(snapshot.edges.get(PAIR_COLUMN)).not.toBeNull();
        const undirectedFirst = await load("graph { a -- b; c -> d }");
        expect(undirectedFirst.snapshot.directed).toBe(true);
        expect(undirectedFirst.snapshot.edgeCount).toBe(3);
        expect(undirectedFirst.report.counts.expandedMixed).toBe(1);
    });

    it("honours mismatchedEdgeOperator header and error", async () => {
        const header = await load("digraph { c -- d }", { mismatchedEdgeOperator: "header" });
        expect(header.snapshot.edgeCount).toBe(1);
        expect(header.snapshot.directed).toBe(true);
        expect(codes(header.report)).toEqual([DOT_ISSUE.EDGE_OPERATOR]);
        const err = await failure("digraph { c -- d }", { mismatchedEdgeOperator: "error" });
        expect(err.report.issues[0].code).toBe(DOT_ISSUE.SYNTAX);
        await expect(load("digraph { }", { mismatchedEdgeOperator: "sometimes" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("applies onMixedDirection error and the forcing policies to a contradicting operator", async () => {
        const err = await failure("digraph { a -> b; c -- d }", { onMixedDirection: "error" });
        expect(err.report.issues.map((i) => i.code)).toContain("E_MIXED_DIRECTION");
        const forced = await load("digraph { a -> b; c -- d }", { onMixedDirection: "directed" });
        expect(forced.snapshot.edgeCount).toBe(2);
        expect(forced.snapshot.directed).toBe(true);
    });

    it("follows a locked sink and expands the file's undirected edges", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        const { snapshot, report } = await load("graph { a -- b }", undefined, builder);
        expect(snapshot.directed).toBe(true);
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        expect(snapshot.edgeCount).toBe(2);
        expect(report.counts.expandedMixed).toBe(1);
    });

    it("reports options the sink does not honour and options DOT has no use for", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64", selfLoops: "keep" });
        const { report } = await load(
            "digraph { a }",
            { weightDtype: "f32", selfLoops: "drop", nodeIdFrom: "label" },
            builder,
        );
        expect(codes(report)).toEqual([DOT_ISSUE.OPTION_IGNORED, SINK_OPTION_CODE, SINK_OPTION_CODE]);
        expect(report.issues[1]).toMatchObject({ element: "selfLoops", category: "coercion" });
        expect(report.issues[2]).toMatchObject({ element: "weightDtype", category: "coercion" });
    });

    it("declares its columns without a role the caller's sink already gave away, and into an existing column", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
        builder.declareEdgeColumn({ name: "key", dtype: "i32" });
        const { snapshot, report } = await load("digraph { a [label=A]; a -> b [key=5] }", undefined, builder);
        expect(codes(report)).toEqual([DOT_ISSUE.ROLE_TAKEN, DOT_ISSUE.COLUMN_RENAMED]);
        expect(snapshot.nodes.get("label")?.meta.role).toBeNull();
        expect(nodeCell(snapshot, "a", "label")).toBe("A");
        // design 5.6: the caller's i32 "key" column keeps its shape; the file's key goes to key#key
        expect(snapshot.edges.get("key")?.isSet(0)).toBe(false);
        expect(cell(snapshot, "edges", "key#key", 0)).toBe("5");
        expect(snapshot.edges.get("key#key")?.meta.role).toBe("id");
    });

    it("records a value the existing column cannot hold as a per-element error and continues", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareNodeColumn({ name: "n", dtype: "i32" });
        const { snapshot, report } = await load("digraph { a [n=text, m=1]; b [n=2] }", undefined, builder);
        expect(codes(report)).toEqual(["E_COLUMN_TYPE"]);
        expect(report.issues[0].element).toBe("a");
        expect(nodeCell(snapshot, "a", "m")).toBe(1);
        expect(nodeCell(snapshot, "b", "n")).toBe(2);
    });

    it("warns about content after the graph and reads only the first graph", async () => {
        const { snapshot, report } = await load("digraph { a } digraph { b }");
        expect(snapshot.ids.toArray()).toEqual(["a"]);
        expect(codes(report)).toEqual([DOT_ISSUE.MULTIPLE_GRAPHS]);
    });

    it("sets the direction from the keyword on an empty unlocked sink of either kind", async () => {
        const undirectedSink = new GraphBuilder({ directed: false, weightDtype: "f64" });
        const a = await load("digraph { a -> b }", undefined, undirectedSink);
        expect(a.snapshot.directed).toBe(true);
        expect(a.snapshot.edges.get(PAIR_COLUMN)).toBeNull();
        const directedSink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const b = await load("graph { a -- b }", undefined, directedSink);
        expect(b.snapshot.directed).toBe(false);
        expect(b.snapshot.arcCount).toBe(2);
    });

    it("handles an empty graph body and a graph without a name", async () => {
        const { snapshot } = await load("graph{}");
        expect(snapshot.nodeCount).toBe(0);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.meta.name).toBeNull();
    });

    it("reports abort before reading and calls onProgress", async () => {
        const controller = new AbortController();
        controller.abort();
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await expect(dotImporter.import("digraph { a }", builder, { signal: controller.signal })).rejects.toMatchObject(
            {
                name: "AbortError",
            },
        );
        const seen: number[] = [];
        await load("digraph { a }", { onProgress: (done) => seen.push(done) });
        expect(seen.length).toBeGreaterThan(0);
        expect(seen[seen.length - 1]).toBe("digraph { a }".length);
    });

    it("stops between statements when the signal fires", async () => {
        const controller = new AbortController();
        const lines: string[] = [];
        for (let i = 0; i < 2000; i++) {
            lines.push(`n${i};`);
        }
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const text = `digraph {\n${lines.join("\n")}\n}`;
        const pending = dotImporter.import(
            (async function* (): AsyncGenerator<string> {
                yield text.slice(0, 20);
                await Promise.resolve();
                controller.abort();
                yield text.slice(20);
            })(),
            builder,
            { signal: controller.signal },
        );
        await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    });
});

describe("dot importer: edge cases of the grammar and the sink", () => {
    it("reads CRLF and CR line endings, continuations and HTML strings across lines with correct line numbers", async () => {
        const text =
            'digraph {\r\n# skipped\r\na [label="x\\\r\ny", h=<<b>\r\nz</b>>];\rb [c="p\r\nq"];\r\nc [x=];\r\n}';
        const err = await failure(text);
        expect(err.report.issues[0].line).toBe(8);
        const { snapshot } = await load(text.replace("c [x=];", "c;"));
        expect(nodeCell(snapshot, "a", "label")).toBe("xy");
        expect(nodeCell(snapshot, "a", "h")).toBe("<<b>\r\nz</b>>");
        expect(nodeCell(snapshot, "b", "c")).toBe("p\r\nq");
    });

    it("names the unclosed subgraph and the offending token in syntax errors", async () => {
        expect((await failure("digraph {\n subgraph s {\n a\n")).report.issues[0].message).toContain(
            "subgraph opened on line 2",
        );
        expect((await failure("digraph { a ->")).report.issues[0].message).toContain("end of input");
        expect((await failure("digraph { = }")).report.issues[0].message).toContain("at the start of a statement");
        expect((await failure("digraph { a [x y] }")).report.issues[0].message).toContain(
            'expected "=" after attribute',
        );
    });

    it("skips a cluster mentioned inside itself and propagates clusters through plain subgraphs", async () => {
        const { snapshot, report } = await load(
            "digraph { subgraph cluster_a { cluster_a; cluster=true; x } subgraph s { subgraph cluster_b { y } } }",
        );
        expect(report.warningCount).toBe(0);
        const p = parents(snapshot);
        expect(p.cluster_a).toBeUndefined();
        expect(p.x).toBe("cluster_a");
        expect(p.y).toBe("cluster_b");
        expect(p.cluster_b).toBeUndefined();
    });

    it("records a cluster name or node the id rule rejects and continues", async () => {
        const { snapshot, report } = await load("digraph { subgraph cluster_x { label=L; 1 } 2 [a=1]; x [b=2]; 3 }", {
            ids: "number",
        });
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(codes(report)).toEqual(["E_INVALID_ID", DOT_ISSUE.SUBGRAPH_ATTRIBUTES_DROPPED, "E_INVALID_ID"]);
        expect(report.counts.skippedNodes).toBe(2);
        expect(snapshot.nodes.get(PARENT_COLUMN)).toBeNull();
    });

    it("records a node id with a lone surrogate as E_INVALID_ID and skips its edge", async () => {
        const bad = `digraph { "${String.fromCharCode(0xd800)}" -> b; c -> d }`;
        const { snapshot, report } = await load(bad);
        expect(codes(report)).toEqual(["E_INVALID_ID"]);
        expect(report.issues[0].message).toContain("surrogate");
        expect(snapshot.ids.toArray()).toEqual(["b", "c", "d"]);
        expect(snapshot.edgeCount).toBe(1);
        expect(report.counts.skippedNodes).toBe(1);
    });

    it("reuses nodes the caller's sink already holds without re-applying defaults", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.setNodeValue("own", 0, 1);
        const { snapshot, report } = await load("digraph { node [c=1]; a -> b }", undefined, builder);
        expect(report.counts.nodes).toBe(1);
        expect(nodeCell(snapshot, "a", "c")).toBeUndefined();
        expect(nodeCell(snapshot, "b", "c")).toBe(1);
        expect(nodeCell(snapshot, "a", "own")).toBe(1);
    });

    it("records sink errors from edge attributes, ports and merged weights", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64", weighted: false });
        builder.declareEdgeColumn({ name: "n", dtype: "i32" });
        builder.declareEdgeColumn({ name: SOURCE_PORT_COLUMN, dtype: "i32" });
        const text = "strict digraph { a -> b [n=text]; a:p -> b [weight=2]; c -> d [weight=3] }";
        const { snapshot, report } = await load(text, undefined, builder);
        expect(snapshot.edgeCount).toBe(1);
        expect(codes(report)).toEqual([
            "E_COLUMN_TYPE",
            DOT_ISSUE.STRICT_MERGED,
            "E_INVALID_WEIGHT",
            DOT_ISSUE.COLUMN_RENAMED,
            "E_INVALID_WEIGHT",
        ]);
        expect(report.issues[0].element).toBe("a -> b");
        expect(report.counts.skippedEdges).toBe(1);
        expect(report.counts.edges).toBe(1);
    });

    it("updates the ports of an edge merged under strict", async () => {
        const { snapshot } = await load("strict digraph { a -> b; a:p -> b:q }");
        expect(cell(snapshot, "edges", SOURCE_PORT_COLUMN, 0)).toBe("p");
        expect(cell(snapshot, "edges", TARGET_PORT_COLUMN, 0)).toBe("q");
    });
});

describe("dot importer: sniff and metadata", () => {
    const encode = (text: string): Uint8Array => new TextEncoder().encode(text);

    it("recognises a DOT header with high confidence", () => {
        expect(dotImporter.sniff?.(readCorpusBytes("dot", "hello.gv"))).toBeGreaterThanOrEqual(0.9);
        expect(dotImporter.sniff?.(encode("/* c */ // d\n# e\nstrict graph G"))).toBe(0.8);
        expect(dotImporter.sniff?.(encode("digraph {"))).toBe(0.95);
        expect(dotImporter.sniff?.(encode("source,target\na,b\n"))).toBe(0);
        expect(dotImporter.sniff?.(encode("graphml"))).toBe(0);
        expect(dotImporter.sniff?.(encode("/* unterminated"))).toBe(0);
        expect(dotImporter.sniff?.(encode("// only"))).toBe(0);
    });

    it("declares its format, extensions and mime types", () => {
        expect(dotImporter.format).toBe("dot");
        expect(dotImporter.extensions).toEqual([".dot", ".gv"]);
        expect(dotImporter.mimeTypes).toEqual(["text/vnd.graphviz"]);
    });

    it("rejects an unknown common option value", async () => {
        const options: CommonImportOptions = { ids: "weird" as never };
        await expect(load("digraph {}", options)).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
    });

    it("never leaves INVALID_INDEX in the parent column of a member", async () => {
        const { snapshot } = await load("digraph { subgraph cluster_a { a } b }");
        const column = snapshot.nodes.get(PARENT_COLUMN);
        expect(column?.dtype).toBe("u32");
        expect(column?.isSet(snapshot.ids.requireIndex("b"))).toBe(false);
        expect((column as { data: Uint32Array }).data[snapshot.ids.requireIndex("b")]).toBe(INVALID_INDEX);
    });
});
