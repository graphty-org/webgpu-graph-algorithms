import { GraphBuilder, GraphFormatError, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DIRECTION_FORCED_CODE, DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE } from "../../../src/common/direction.js";
import { PAJEK_ISSUE, pajekImporter, type PajekImportOptions } from "../../../src/formats/pajek/importer.js";
import { type CommonImportOptions, ImportError, type ImportInput, type ImportReport } from "../../../src/types.js";
import {
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedText,
} from "../../helpers/corpus.js";
import { expectSameSnapshot } from "../../helpers/roundtrip.js";

type Options = PajekImportOptions & CommonImportOptions;

async function load(
    input: ImportInput,
    options?: Options,
    builder?: GraphBuilder,
): Promise<{ snapshot: GraphSnapshot; report: ImportReport; builder: GraphBuilder }> {
    const sink = builder ?? new GraphBuilder({ directed: true, weightDtype: "f64" });
    const report = await pajekImporter.import(input, sink, options);
    return { snapshot: sink.freeze(), report, builder: sink };
}

async function fails(input: ImportInput, options?: Options): Promise<ImportError> {
    try {
        await load(input, options);
    } catch (err) {
        expect(err).toBeInstanceOf(ImportError);
        return err as ImportError;
    }
    throw new Error("expected the import to throw ImportError");
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

/** Source edges: logical edges minus the mirror halves of expanded pairs. */
function sourceEdges(snapshot: GraphSnapshot): number {
    const pair = snapshot.edges.byRole("pair");
    if (pair === null || pair.dtype !== "u32") {
        return snapshot.edgeCount;
    }
    let mirrors = 0;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (pair.isSet(e) && pair.data[e] !== INVALID_INDEX && pair.data[e] < e) {
            mirrors++;
        }
    }
    return snapshot.edgeCount - mirrors;
}

function label(snapshot: GraphSnapshot, index: number): string | undefined {
    return snapshot.nodes.requireTyped("label", "string").value(index);
}

const SIMPLE = readCorpusText("pajek", "simple.net");

describe("pajekImporter: the corpus", () => {
    const expectedDirected: Record<string, boolean> = {
        "simple.net": true,
        "karate.net": false,
        "dolphins.net": false,
        "football.net": false,
    };

    for (const entry of corpusFiles("pajek")) {
        it(`imports ${entry.path} with the manifest's counts`, async () => {
            const { snapshot, report } = await load(readCorpusText("pajek", entry.path));
            expect(snapshot.nodeCount).toBe(entry.expectedNodes);
            expect(sourceEdges(snapshot)).toBe(entry.expectedEdges);
            expect(report.counts.nodes).toBe(entry.expectedNodes);
            expect(snapshot.directed).toBe(expectedDirected[entry.path]);
            expect(report.errorCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(report.format).toBe("pajek");
            expect(report.durationMs).toBeGreaterThanOrEqual(0);
            expect(snapshot.ids.kind).toBe("identity");
            expect(snapshot.meta.sourceFormat).toBe("pajek");
        });
    }

    it("simple.net: labels, coordinates, arcs, edges and explicit weights", async () => {
        const { snapshot, report } = await load(SIMPLE);
        expect(snapshot.ids.offset).toBe(1);
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4, 5]);
        expect(label(snapshot, 0)).toBe("Node A");
        expect(label(snapshot, 4)).toBe("Node E");
        const position = snapshot.nodes.requireTyped("position", "f32");
        expect(position.meta.role).toBe("position");
        expect(position.meta.components).toBe(3);
        expect(position.meta.extra).toEqual({ sourceDims: 3, units: "file" });
        expect(Array.from(position.value(0) as ArrayLike<number>)).toEqual([Math.fround(0.1), Math.fround(0.2), 0]);
        expect(Array.from(position.value(4) as ArrayLike<number>)).toEqual([Math.fround(0.9), 1, 0]);
        // 3 arcs, then 2 undirected edges expanded into pairs
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(7);
        expect(report.counts.edges).toBe(7);
        expect(report.counts.expandedMixed).toBe(2);
        const { src, dst } = snapshot.edgeList();
        expect(Array.from(src)).toEqual([0, 1, 2, 3, 4, 0, 2]);
        expect(Array.from(dst)).toEqual([1, 2, 3, 4, 3, 2, 0]);
        const directed = snapshot.edges.byRole("directed");
        expect(directed).not.toBeNull();
        expect([0, 1, 2, 3, 4, 5, 6].map((e) => directed?.value(e))).toEqual([
            true,
            true,
            true,
            false,
            false,
            false,
            false,
        ]);
        const pair = snapshot.edges.requireTyped("graphty.pair", "u32");
        expect(pair.value(3)).toBe(4);
        expect(pair.value(6)).toBe(5);
        expect(pair.isSet(0)).toBe(false);
        // every weight explicit: no role-weight column, arc weights as written
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.edges.byRole("weight")).toBeNull();
        const { weights } = snapshot.edgeList();
        expect(weights).not.toBeNull();
        expect(Array.from(weights as Float32Array)).toEqual([1, 2, 1.5, 1, 1, 0.5, 0.5]);
        expect(report.issues).toEqual([]);
    });

    it("karate.net: an *Edges-only file is undirected with 78 edges and no pair column", async () => {
        const { snapshot } = await load(readCorpusText("pajek", "karate.net"));
        expect(snapshot.directed).toBe(false);
        expect(snapshot.edgeCount).toBe(78);
        expect(snapshot.edges.byRole("pair")).toBeNull();
        expect(snapshot.edges.byRole("directed")).toBeNull();
        expect(label(snapshot, 33)).toBe("Actor 34");
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(true);
        expect(snapshot.edgeSource(0)).toBe(0);
        expect(snapshot.edgeTarget(0)).toBe(1);
    });

    it("karate-large.net: edges without a value are unweighted", async () => {
        const { snapshot } = await load(readCorpusText("pajek", "karate-large.net"));
        expect(snapshot.flags.weighted).toBe(false);
        expect(snapshot.weights).toBeNull();
        expect(snapshot.edgeCount).toBe(78);
        expect(snapshot.edgeSource(0)).toBe(1);
        expect(snapshot.edgeTarget(0)).toBe(0);
    });

    it("dolphins.net and football.net: zero-based numbering is detected and reported once", async () => {
        const dolphins = await load(readCorpusText("pajek", "dolphins.net"));
        expect(dolphins.snapshot.ids.offset).toBe(0);
        expect(dolphins.snapshot.ids.toArray().slice(0, 3)).toEqual([0, 1, 2]);
        expect(label(dolphins.snapshot, 0)).toBe("Beak");
        expect(label(dolphins.snapshot, 61)).toBe("Zipfel");
        expect(codes(dolphins.report)).toEqual([PAJEK_ISSUE.ZERO_BASED]);
        expect(dolphins.report.issues[0].severity).toBe("warning");
        expect(dolphins.report.issues[0].category).toBe("coercion");
        expect(dolphins.snapshot.edgeSource(0)).toBe(8);
        expect(dolphins.snapshot.edgeTarget(0)).toBe(3);
        const football = await load(readCorpusText("pajek", "football.net"));
        expect(label(football.snapshot, 114)).toBe("Hawaii");
        expect(label(football.snapshot, 0)).toBe("BrighamYoung");
        expect(football.snapshot.edgeCount).toBe(613);
        expect(codes(football.report)).toEqual([PAJEK_ISSUE.ZERO_BASED]);
    });

    it("reads the same snapshot from every input shape", async () => {
        const reference = (await load(SIMPLE)).snapshot;
        for (const shape of inputShapes(readCorpusBytes("pajek", "simple.net"))) {
            const { snapshot, report } = await load(shape.make());
            expect(report.issues, shape.name).toEqual([]);
            expectSameSnapshot(reference, snapshot);
        }
    });

    it("reports progress and honours the abort signal", async () => {
        const calls: [number, number | undefined][] = [];
        await load(SIMPLE, { onProgress: (done, total) => calls.push([done, total]) });
        expect(calls.length).toBeGreaterThan(0);
        expect(calls[calls.length - 1]).toEqual([SIMPLE.length, SIMPLE.length]);
        const controller = new AbortController();
        controller.abort();
        await expect(load(SIMPLE, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    });
});

describe("pajekImporter: sniff", () => {
    const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);

    it("recognises *Vertices and *Network starts", () => {
        expect(pajekImporter.sniff?.(bytes("*Vertices 3\n1 a\n"))).toBe(0.9);
        expect(pajekImporter.sniff?.(bytes("  *vertices 3\n"))).toBe(0.9);
        expect(pajekImporter.sniff?.(bytes(`${String.fromCharCode(0xfeff)}*VERTICES 3\n`))).toBe(0.9);
        expect(pajekImporter.sniff?.(bytes("*Network karate\n*Vertices 3\n"))).toBe(0.8);
    });

    it("rejects other formats", () => {
        expect(pajekImporter.sniff?.(bytes("graph [\n"))).toBe(0);
        expect(pajekImporter.sniff?.(bytes("source,target\n"))).toBe(0);
        expect(pajekImporter.sniff?.(bytes(""))).toBe(0);
        expect(pajekImporter.sniff?.(bytes("*Arcs\n1 2\n"))).toBe(0);
    });

    it("declares its format, extensions and mime types", () => {
        expect(pajekImporter.format).toBe("pajek");
        expect(pajekImporter.extensions).toEqual([".net", ".paj"]);
        expect(pajekImporter.mimeTypes).toContain("text/plain");
    });
});

describe("pajekImporter: vertex lines", () => {
    it("reads bare labels, two coordinates, the shape keyword, parameters and intervals", async () => {
        const text = [
            "*Vertices 3",
            "1 alpha 0.5 0.25 box ic Red bc Black s_size 5 x_fact 1.5 fixed true [1-5,7-*]",
            '2 "beta gamma" 0.1 0.2 ellipse',
            "3 delta",
            "*Arcs",
            "1 2",
        ].join("\n");
        const { snapshot, report } = await load(text);
        expect(report.issues).toEqual([]);
        expect(label(snapshot, 0)).toBe("alpha");
        expect(label(snapshot, 1)).toBe("beta gamma");
        expect(label(snapshot, 2)).toBe("delta");
        const position = snapshot.nodes.requireTyped("position", "f32");
        expect(position.meta.extra.sourceDims).toBe(2);
        expect(Array.from(position.value(0) as ArrayLike<number>)).toEqual([0.5, 0.25, 0]);
        expect(position.isSet(2)).toBe(false);
        const shape = snapshot.nodes.requireTyped("shape", "dict");
        expect(shape.value(0)).toBe("box");
        expect(shape.value(1)).toBe("ellipse");
        expect(shape.isSet(2)).toBe(false);
        expect(shape.meta.origin?.format).toBe("pajek");
        expect(snapshot.nodes.requireTyped("ic", "string").value(0)).toBe("Red");
        expect(snapshot.nodes.requireTyped("bc", "string").value(0)).toBe("Black");
        expect(snapshot.nodes.requireTyped("s_size", "i32").value(0)).toBe(5);
        expect(snapshot.nodes.requireTyped("x_fact", "f64").value(0)).toBe(1.5);
        expect(snapshot.nodes.requireTyped("fixed", "bool").value(0)).toBe(true);
        expect(snapshot.nodes.get("ic")?.isSet(1)).toBe(false);
        const spells = snapshot.nodes.requireTyped("spells", "list");
        expect(spells.meta.role).toBe("spells");
        expect(spells.meta.itemDtype).toBe("f64");
        expect(spells.meta.itemComponents).toBe(2);
        expect([...spells.sliceOf(0)].map((p) => Array.from(p as ArrayLike<number>))).toEqual([
            [1, 5],
            [7, Infinity],
        ]);
        expect(spells.isSet(1)).toBe(false);
    });

    it("parses every interval form and refuses malformed ones", async () => {
        const text = "*Vertices 4\n1 a [3]\n2 b [*-4,6-7]\n3 c [x-y]\n4 d [5-2]\n*Edges\n";
        const { snapshot, report } = await load(text);
        const spells = snapshot.nodes.requireTyped("spells", "list");
        expect([...spells.sliceOf(0)].map((p) => Array.from(p as ArrayLike<number>))).toEqual([[3, 3]]);
        expect([...spells.sliceOf(1)].map((p) => Array.from(p as ArrayLike<number>))).toEqual([
            [-Infinity, 4],
            [6, 7],
        ]);
        expect(spells.isSet(2)).toBe(false);
        expect(spells.isSet(3)).toBe(false);
        expect(label(snapshot, 2)).toBeUndefined();
        expect(codes(report)).toEqual([PAJEK_ISSUE.INTERVAL, PAJEK_ISSUE.INTERVAL]);
        expect(report.issues[0]).toMatchObject({ category: "validation-error", severity: "error", line: 4 });
        expect(report.counts.skippedNodes).toBe(2);
    });

    it("widens a parameter column across lines by the section 5.1 rule", async () => {
        const text = "*Vertices 3\n1 a k 1\n2 b k 2.5\n3 c k 01\n*Edges\n";
        const { snapshot } = await load(text);
        const k = snapshot.nodes.requireTyped("k", "string");
        expect([k.value(0), k.value(1), k.value(2)]).toEqual(["1", "2.5", "01"]);
    });

    it("treats a vertex without a line as declared, with no label", async () => {
        const { snapshot, report } = await load("*Vertices 4\n*Edges\n1 4\n");
        expect(snapshot.nodeCount).toBe(4);
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4]);
        expect(snapshot.nodes.get("label")).toBeNull();
        expect(snapshot.edgeCount).toBe(1);
        expect(report.issues).toEqual([]);
    });

    it("reports a partial vertex list and keeps the declared count", async () => {
        const { snapshot, report } = await load(readMalformedText("pajek", "wrong-vertex-count.net"));
        expect(snapshot.nodeCount).toBe(10);
        expect(label(snapshot, 2)).toBe("C");
        expect(label(snapshot, 3)).toBeUndefined();
        expect(codes(report)).toEqual([PAJEK_ISSUE.VERTEX_COUNT]);
        expect(report.issues[0].severity).toBe("error");
        expect(report.issues[0].message).toContain("10");
    });

    it("skips a malformed vertex line and keeps the vertex", async () => {
        const { snapshot, report } = await load(readMalformedText("pajek", "malformed-vertex.net"));
        expect(snapshot.nodeCount).toBe(3);
        expect(label(snapshot, 1)).toBeUndefined();
        expect(label(snapshot, 2)).toBe("C");
        expect(codes(report)).toEqual([PAJEK_ISSUE.VERTEX_LINE]);
        expect(report.issues[0]).toMatchObject({ category: "parse-error", line: 3 });
        expect(report.counts.skippedNodes).toBe(1);
        expect(snapshot.edgeCount).toBe(2);
    });

    it("reports a vertex number outside the declared range", async () => {
        const { snapshot, report } = await load("*Vertices 2\n1 a\n2 b\n5 e\n0 z\n*Edges\n1 2\n");
        expect(snapshot.nodeCount).toBe(2);
        expect(codes(report)).toEqual([PAJEK_ISSUE.VERTEX_RANGE, PAJEK_ISSUE.VERTEX_RANGE]);
        expect(report.issues[0].message).toContain("1..2");
        expect(report.counts.skippedNodes).toBe(2);
    });

    it("reports a duplicate vertex line and lets the later values overwrite", async () => {
        const { snapshot, report } = await load("*Vertices 2\n1 a\n2 b\n1 c\n*Edges\n");
        expect(label(snapshot, 0)).toBe("c");
        expect(codes(report)).toEqual([PAJEK_ISSUE.DUPLICATE_NODE]);
        expect(report.issues[0]).toMatchObject({ category: "merged", severity: "warning", element: "1", line: 4 });
    });

    it("refuses a single coordinate, a dangling parameter and an unterminated quote", async () => {
        const text = '*Vertices 3\n1 a 0.5\n2 b 0.1 0.2 ic\n3 "c\n*Edges\n';
        const { report, snapshot } = await load(text);
        expect(codes(report)).toEqual([
            PAJEK_ISSUE.VERTEX_LINE,
            PAJEK_ISSUE.VERTEX_LINE,
            PAJEK_ISSUE.UNTERMINATED_QUOTE,
        ]);
        expect(report.issues.map((i) => i.line)).toEqual([2, 3, 4]);
        expect(report.counts.skippedNodes).toBe(3);
        expect(snapshot.nodes.get("label")).toBeNull();
        expect(snapshot.nodes.get("position")).toBeNull();
    });

    it("warns once when vertex lines mix two and three coordinates", async () => {
        const { snapshot, report } = await load("*Vertices 3\n1 a 1 2\n2 b 3 4 5\n3 c 6 7 8\n*Edges\n");
        expect(codes(report)).toEqual([PAJEK_ISSUE.COORD_DIMS]);
        const position = snapshot.nodes.requireTyped("position", "f32");
        expect(position.meta.extra.sourceDims).toBe(2);
        expect(Array.from(position.value(1) as ArrayLike<number>)).toEqual([3, 4, 5]);
    });

    it("declares a two-mode network in the metadata", async () => {
        const { snapshot } = await load("*Vertices 3 2\n1 a\n2 b\n3 c\n*Edges\n1 3\n2 3\n");
        expect(snapshot.meta.extra).toEqual({ pajek: { firstMode: 2 } });
    });

    it("reads a *Network name into the metadata", async () => {
        const { snapshot } = await load("*Network my network\n*Vertices 1\n1 a\n*Arcs\n");
        expect(snapshot.meta.name).toBe("my network");
        expect(snapshot.meta.sourceFormat).toBe("pajek");
    });

    it("skips % comments and blank lines and accepts CRLF and case-insensitive keywords", async () => {
        const text = "% a comment\r\n*VERTICES 2\r\n\r\n1 a\r\n% another\r\n2 b\r\n*arcs\r\n1 2 3\r\n";
        const { snapshot, report } = await load(text);
        expect(report.issues).toEqual([]);
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeList().weights?.[0]).toBe(3);
    });
});

describe("pajekImporter: numbering and ids", () => {
    it("forces the base with firstVertex and rejects other values", async () => {
        const zero = await load("*Vertices 2\n0 a\n1 b\n*Edges\n0 1\n", { firstVertex: 0 });
        expect(zero.snapshot.ids.toArray()).toEqual([0, 1]);
        expect(codes(zero.report)).toEqual([PAJEK_ISSUE.ZERO_BASED]);
        const one = await load("*Vertices 2\n0 a\n1 b\n*Edges\n0 1\n", { firstVertex: 1 });
        expect(one.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(codes(one.report)).toEqual([PAJEK_ISSUE.VERTEX_RANGE, PAJEK_ISSUE.UNKNOWN_NODE]);
        await expect(load("*Vertices 1\n1 a\n*Edges\n", { firstVertex: 2 as unknown as 1 })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("applies the ids coercion rule to vertex numbers", async () => {
        const text = "*Vertices 2\n1 a\n2 b\n*Edges\n1 2\n";
        expect((await load(text, { ids: "canonical" })).snapshot.ids.toArray()).toEqual([1, 2]);
        expect((await load(text, { ids: "number" })).snapshot.ids.toArray()).toEqual([1, 2]);
        expect((await load(text, { ids: "string" })).snapshot.ids.toArray()).toEqual(["1", "2"]);
        expect((await load(text, { ids: "keep" })).snapshot.ids.toArray()).toEqual(["1", "2"]);
        expect((await load(text, { ids: "string" })).snapshot.ids.kind).toBe("string");
    });

    it("nodeIdFrom label: labels become ids, numbers stand in for unlabelled vertices", async () => {
        const text = "*Vertices 4\n1 alice\n2 42\n4 dave\n*Arcs\n1 2\n2 3\n3 4\n";
        const { snapshot, report } = await load(text, { nodeIdFrom: "label" });
        expect(snapshot.ids.toArray()).toEqual(["alice", 42, "dave", 3]);
        expect(label(snapshot, 0)).toBe("alice");
        expect(snapshot.edgeSource(1)).toBe(1);
        expect(snapshot.edgeTarget(1)).toBe(3);
        expect(snapshot.edgeSource(2)).toBe(3);
        expect(snapshot.edgeTarget(2)).toBe(2);
        expect(codes(report)).toEqual([PAJEK_ISSUE.VERTEX_COUNT]);
        expect(report.counts.nodes).toBe(4);
    });

    it("nodeIdFrom label: duplicate labels merge with a warning, ids string keeps numeric labels text", async () => {
        const text = "*Vertices 3\n1 x\n2 x\n3 y\n*Edges\n1 3\n2 3\n";
        const { snapshot, report } = await load(text, { nodeIdFrom: "label" });
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.ids.toArray()).toEqual(["x", "y"]);
        expect(snapshot.edgeCount).toBe(2);
        expect(codes(report)).toEqual([PAJEK_ISSUE.LABEL_MERGED]);
        expect(report.issues[0]).toMatchObject({ category: "merged", severity: "warning", element: "2" });
        const strings = await load("*Vertices 1\n1 7\n*Edges\n", { nodeIdFrom: "label", ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["7"]);
    });

    it("nodeIdFrom label with ids number reports merged label texts", async () => {
        const text = "*Vertices 2\n1 01\n2 1\n*Edges\n1 2\n";
        const { snapshot, report } = await load(text, { nodeIdFrom: "label", ids: "number" });
        expect(snapshot.nodeCount).toBe(1);
        expect(codes(report)).toEqual([PAJEK_ISSUE.LABEL_MERGED, PAJEK_ISSUE.ID_MERGED]);
    });

    it("nodeIdFrom index: ids are the 0-based positions", async () => {
        const { snapshot } = await load(SIMPLE, { nodeIdFrom: "index" });
        expect(snapshot.ids.toArray()).toEqual([0, 1, 2, 3, 4]);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(0);
        expect(label(snapshot, 0)).toBe("Node A");
    });
});

describe("pajekImporter: lines", () => {
    it("reads arcs and edges with parameters, intervals and the relation of the section", async () => {
        const text = [
            "*Vertices 3",
            "1 a",
            "2 b",
            "3 c",
            '*Arcs :1 "likes"',
            '1 2 2.5 c Blue l "a to b" w 3 [1-2]',
            "*Arcs :2",
            "2 3 1",
            "*Edges",
            "3 1",
        ].join("\n");
        const { snapshot, report } = await load(text);
        expect(report.issues).toEqual([]);
        expect(snapshot.edgeCount).toBe(4);
        const relation = snapshot.edges.requireTyped("relation", "dict");
        expect(relation.value(0)).toBe("likes");
        expect(relation.value(1)).toBe("2");
        expect(relation.isSet(2)).toBe(false);
        expect(snapshot.edges.requireTyped("c", "string").value(0)).toBe("Blue");
        expect(snapshot.edges.requireTyped("l", "string").value(0)).toBe("a to b");
        expect(snapshot.edges.requireTyped("w", "i32").value(0)).toBe(3);
        expect(snapshot.edges.get("c")?.isSet(1)).toBe(false);
        // the mirror half of the expanded edge carries no attributes
        expect(snapshot.edges.get("c")?.isSet(3)).toBe(false);
        const spells = snapshot.edges.requireTyped("spells", "list");
        expect([...spells.sliceOf(0)].map((p) => Array.from(p as ArrayLike<number>))).toEqual([[1, 2]]);
        const weights = snapshot.edges.byRole("weight");
        expect(weights).not.toBeNull();
        expect(weights?.isSet(0)).toBe(true);
        expect(weights?.value(0)).toBe(2.5);
        expect(weights?.isSet(2)).toBe(false);
    });

    it("weightFrom null keeps the line value as a column; weightFrom names a parameter", async () => {
        const text = "*Vertices 2\n1 a\n2 b\n*Arcs\n1 2 2.5 w 4\n2 1 w abc\n";
        const asValue = await load(text, { weightFrom: null });
        expect(asValue.snapshot.flags.weighted).toBe(false);
        const value = asValue.snapshot.edges.requireTyped("value", "f64");
        expect(value.value(0)).toBe(2.5);
        expect(value.isSet(1)).toBe(false);
        expect(asValue.snapshot.edges.requireTyped("w", "string").value(1)).toBe("abc");
        const fromW = await load(text, { weightFrom: "w" });
        expect(fromW.snapshot.edgeCount).toBe(1);
        expect(fromW.snapshot.edgeList().weights?.[0]).toBe(4);
        expect(fromW.snapshot.edges.get("w")).toBeNull();
        expect(fromW.snapshot.edges.requireTyped("value", "f64").value(0)).toBe(2.5);
        expect(codes(fromW.report)).toEqual(["E_INVALID_WEIGHT"]);
        expect(fromW.report.issues[0]).toMatchObject({ category: "validation-error", line: 6 });
        expect(fromW.report.counts.skippedEdges).toBe(1);
    });

    it("reads *Edgeslist and *Arcslist adjacency lists", async () => {
        const text = "*Vertices 4\n*Arcslist\n1 2 3\n4\n*Edgeslist\n2 4\n";
        const { snapshot, report } = await load(text);
        expect(report.issues).toEqual([]);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(4);
        const { src, dst } = snapshot.edgeList();
        expect(Array.from(src)).toEqual([0, 0, 1, 3]);
        expect(Array.from(dst)).toEqual([1, 2, 3, 1]);
        expect(snapshot.flags.weighted).toBe(false);
        const bad = await load("*Vertices 2\n*Edgeslist\n1 x\n", { errorLimit: 100 });
        expect(codes(bad.report)).toEqual([PAJEK_ISSUE.LINE]);
        expect(bad.snapshot.edgeCount).toBe(0);
    });

    it("reads a *Matrix as arcs with the entries as weights", async () => {
        const text = "*Vertices 3\n1 a\n2 b\n3 c\n*Matrix\n0 1 2\n0 0 0\n0.5 0 0\n";
        const { snapshot, report } = await load(text);
        expect(report.issues).toEqual([]);
        expect(snapshot.directed).toBe(true);
        const { src, dst, weights } = snapshot.edgeList();
        expect(Array.from(src)).toEqual([0, 0, 2]);
        expect(Array.from(dst)).toEqual([1, 2, 0]);
        expect(Array.from(weights as Float32Array)).toEqual([1, 2, 0.5]);
    });

    it("reports matrix rows that are too few, too many, too short or not numeric", async () => {
        const short = await load("*Vertices 2\n*Matrix\n0 1\n");
        expect(codes(short.report)).toEqual([PAJEK_ISSUE.MATRIX_ROWS]);
        expect(short.snapshot.edgeCount).toBe(1);
        const long = await load("*Vertices 1\n*Matrix\n0\n1\n");
        expect(codes(long.report)).toEqual([PAJEK_ISSUE.MATRIX_ROWS, PAJEK_ISSUE.MATRIX_ROWS]);
        const bad = await load("*Vertices 2\n*Matrix\n0 1 1\nx 0\n");
        expect(codes(bad.report)).toEqual([PAJEK_ISSUE.LINE, PAJEK_ISSUE.LINE]);
        expect(bad.snapshot.edgeCount).toBe(0);
    });

    it("skips a malformed line and an endpoint outside the vertex range", async () => {
        const malformed = await load(readMalformedText("pajek", "malformed-edge.net"));
        expect(malformed.snapshot.edgeCount).toBe(2);
        expect(codes(malformed.report)).toEqual([PAJEK_ISSUE.LINE]);
        expect(malformed.report.issues[0]).toMatchObject({ category: "parse-error", line: 7 });
        expect(malformed.report.counts.skippedEdges).toBe(1);
        const unknown = await load(readMalformedText("pajek", "invalid-edge-reference.net"));
        expect(unknown.snapshot.nodeCount).toBe(3);
        expect(unknown.snapshot.edgeCount).toBe(2);
        expect(codes(unknown.report)).toEqual([PAJEK_ISSUE.UNKNOWN_NODE]);
        expect(unknown.report.issues[0]).toMatchObject({ category: "missing-value", severity: "error", line: 7 });
        expect(unknown.report.issues[0].message).toContain("99");
        expect(unknown.report.counts.skippedEdges).toBe(1);
    });

    it("keeps parallel edges and self-loops", async () => {
        const { snapshot } = await load("*Vertices 2\n*Edges\n1 2\n1 2\n1 1\n2 2\n");
        expect(snapshot.edgeCount).toBe(4);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.selfLoopCount).toBe(2);
        const arcs = await load("*Vertices 1\n*Arcs\n1 1\n*Edges\n1 1\n");
        // an undirected self-loop in a directed graph has no mirror
        expect(arcs.snapshot.edgeCount).toBe(2);
        expect(arcs.report.counts.expandedMixed).toBe(1);
    });
});

describe("pajekImporter: direction", () => {
    const EDGES_THEN_ARCS = "*Vertices 3\n*Edges\n1 2\n2 3\n*Arcs\n3 1\n";

    it("*Edges before *Arcs: the sink is expanded in place once", async () => {
        const { snapshot, report } = await load(EDGES_THEN_ARCS, undefined, new GraphBuilder({ directed: false }));
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(5);
        expect(report.counts.edges).toBe(5);
        expect(report.counts.expandedMixed).toBe(2);
        const directed = snapshot.edges.byRole("directed");
        expect([0, 1, 2, 3, 4].map((e) => directed?.value(e))).toEqual([false, false, false, false, true]);
        expect(report.issues).toEqual([]);
    });

    it("onMixedDirection directed / undirected force every edge and report once", async () => {
        const directed = await load(SIMPLE, { onMixedDirection: "directed" });
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edgeCount).toBe(5);
        expect(directed.snapshot.edges.byRole("pair")).toBeNull();
        expect(codes(directed.report)).toEqual([DIRECTION_FORCED_CODE]);
        const undirected = await load(SIMPLE, { onMixedDirection: "undirected" });
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edgeCount).toBe(5);
        expect(codes(undirected.report)).toEqual([DIRECTION_FORCED_CODE, DIRECTION_FORCED_CODE]);
        expect(undirected.report.issues[0].line).toBe(7);
    });

    it("onMixedDirection error refuses the first edge of the other kind", async () => {
        const err = await fails(SIMPLE, { onMixedDirection: "error" });
        expect(err.code).toBe("E_IMPORT");
        expect(codes(err.report)).toEqual([MIXED_DIRECTION_CODE]);
        expect(err.report.issues[0].line).toBe(12);
        expect(err.report.counts.edges).toBe(3);
    });

    it("a locked sink wins over the file and the difference is reported", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        const { snapshot, report } = await load(readCorpusText("pajek", "karate.net"), undefined, builder);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(156);
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        expect(report.counts.expandedMixed).toBe(78);
    });

    it("a network without line sections takes defaultDirected and warns", async () => {
        const text = readMalformedText("pajek", "missing-edges-section.net");
        const asDirected = await load(text);
        expect(asDirected.snapshot.directed).toBe(true);
        expect(asDirected.snapshot.nodeCount).toBe(3);
        expect(asDirected.snapshot.edgeCount).toBe(0);
        expect(codes(asDirected.report)).toEqual([PAJEK_ISSUE.NO_LINES]);
        expect(asDirected.report.issues[0].severity).toBe("warning");
        const asUndirected = await load(text, { defaultDirected: false });
        expect(asUndirected.snapshot.directed).toBe(false);
    });
});

describe("pajekImporter: structure errors and the error limit", () => {
    it("refuses an empty file, garbage and a file without *Vertices with ImportError and a report", async () => {
        const empty = await fails(readMalformedText("pajek", "empty-file.net"));
        expect(codes(empty.report)).toEqual([PAJEK_ISSUE.NO_VERTICES]);
        expect(empty.report.issues[0].category).toBe("parse-error");
        const garbage = await fails(readMalformedText("pajek", "garbage-content.net"));
        expect(codes(garbage.report)).toEqual([
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.NO_VERTICES,
        ]);
        expect(garbage.report.issues.map((i) => i.line)).toEqual([1, 2, 3, null]);
        const headless = await fails(readMalformedText("pajek", "no-vertices-header.net"));
        expect(codes(headless.report)).toEqual([
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.OUTSIDE_SECTION,
            PAJEK_ISSUE.NO_VERTICES,
        ]);
        expect(headless.report.issues[3].line).toBe(4);
        expect(headless.message).toContain("*Vertices");
    });

    it("refuses *Vertices without a count", async () => {
        const err = await fails("*Vertices\n1 a\n");
        expect(codes(err.report)).toEqual([PAJEK_ISSUE.VERTICES_COUNT]);
        expect(err.report.issues[0].line).toBe(1);
    });

    it("refuses a second network in one file, keeping the report so far", async () => {
        const err = await fails("*Vertices 1\n1 a\n*Arcs\n*Vertices 2\n1 b\n2 c\n");
        expect(codes(err.report)).toEqual([PAJEK_ISSUE.MULTIPLE_NETWORKS]);
        expect(err.report.issues[0]).toMatchObject({ category: "unsupported", severity: "error", line: 4 });
        expect(err.report.counts.nodes).toBe(1);
        const network = await fails("*Network a\n*Vertices 1\n1 a\n*Network b\n");
        expect(codes(network.report)).toEqual([PAJEK_ISSUE.MULTIPLE_NETWORKS]);
    });

    it("reports an unsupported section once and skips its lines", async () => {
        const text = "*Vertices 2\n1 a\n2 b\n*Partition\n1\n2\n*Edges\n1 2\n";
        const { snapshot, report } = await load(text);
        expect(codes(report)).toEqual([PAJEK_ISSUE.UNSUPPORTED_SECTION]);
        expect(report.issues[0]).toMatchObject({ category: "unsupported", severity: "error", line: 4 });
        expect(report.issues[0].message).toContain("*Partition");
        expect(snapshot.edgeCount).toBe(1);
    });

    it("reports extra header tokens and a header it cannot parse", async () => {
        const extra = await load("*Vertices 1 x y\n1 a\n*Edges 5\n");
        expect(codes(extra.report)).toEqual([PAJEK_ISSUE.HEADER_EXTRA, PAJEK_ISSUE.HEADER_EXTRA]);
        expect(extra.report.issues[0].message).toContain("x y");
        const bad = await load('*Vertices 1\n1 a\n*"Arcs\n1 1\n*Edges\n');
        expect(codes(bad.report)).toEqual([PAJEK_ISSUE.SYNTAX]);
        expect(bad.snapshot.edgeCount).toBe(0);
    });

    it("aborts with the partial report when the error limit is exceeded", async () => {
        const err = await fails(readMalformedText("pajek", "malformed-edge.net"), { errorLimit: 0 });
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.truncated).toBe(true);
        expect(err.report.errorCount).toBe(1);
        expect(err.report.counts.nodes).toBe(3);
        expect(err.report.counts.edges).toBe(1);
        expect(err.report.counts.skippedEdges).toBe(1);
        const two = await fails("*Vertices 1\n1 a\n*Edges\nx\ny\nz\n", { errorLimit: 1 });
        expect(two.report.errorCount).toBe(2);
        expect(two.report.truncated).toBe(true);
    });

    it("invalid UTF-8 is a fatal parse error", async () => {
        const bytes = new Uint8Array([
            0x2a, 0x56, 0x65, 0x72, 0x74, 0x69, 0x63, 0x65, 0x73, 0x20, 0x31, 0x0a, 0xff, 0x0a,
        ]);
        const err = await fails(bytes);
        expect(codes(err.report)).toEqual(["E_INVALID_UTF8"]);
    });

    it("reports builder-policy options the caller's sink does not use", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f32", selfLoops: "keep" });
        const explicit = await load(SIMPLE, { weightDtype: "f64", selfLoops: "drop", duplicateEdges: "keep" }, builder);
        expect(codes(explicit.report)).toEqual([PAJEK_ISSUE.SINK_OPTION, PAJEK_ISSUE.SINK_OPTION]);
        expect(explicit.report.issues[0]).toMatchObject({ category: "coercion", severity: "warning" });
        expect(explicit.report.issues[0].message).toContain("selfLoops");
        expect(explicit.report.issues[1].message).toContain("weightDtype");
        // options left at their defaults are never reported against the sink
        const implicit = await load(SIMPLE, undefined, new GraphBuilder({ directed: true, weightDtype: "f32" }));
        expect(implicit.report.issues).toEqual([]);
    });

    it("rejects an option outside its set before reading anything", async () => {
        await expect(load(SIMPLE, { ids: "nope" as "keep" })).rejects.toBeInstanceOf(GraphFormatError);
    });
});

describe("pajekImporter: the malformed corpus", () => {
    /** missing-edges-section.net is a legal vertices-only network; every other case is an error. */
    const recoverable = new Set(["missing-edges-section.net"]);

    for (const name of malformedFiles("pajek")) {
        if (recoverable.has(name)) {
            it(`${name}: imports with a warning`, async () => {
                const { report, snapshot } = await load(readMalformedText("pajek", name), { errorLimit: 0 });
                expect(report.errorCount).toBe(0);
                expect(report.warningCount).toBe(1);
                expect(snapshot.nodeCount).toBe(3);
            });
            continue;
        }
        it(`${name}: throws ImportError with the partial report under errorLimit 0`, async () => {
            const err = await fails(readMalformedText("pajek", name), { errorLimit: 0 });
            expect(err.code).toBe("E_IMPORT");
            expect(err.name).toBe("ImportError");
            expect(err.report.format).toBe("pajek");
            expect(err.report.errorCount).toBeGreaterThan(0);
            expect(err.report.issues.length).toBeGreaterThan(0);
            expect(err.report.issues[0].severity).toBe("error");
        });
    }

    it("under the default limit the recoverable cases import with issues and the structural ones throw", async () => {
        const structural = ["empty-file.net", "garbage-content.net", "no-vertices-header.net"];
        for (const name of malformedFiles("pajek")) {
            if (structural.includes(name)) {
                await fails(readMalformedText("pajek", name));
            } else {
                const { report } = await load(readMalformedText("pajek", name));
                expect(report.truncated, name).toBe(false);
                expect(report.issues.length, name).toBe(1);
            }
        }
    });
});
