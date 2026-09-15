import { GraphBuilder, type GraphBuilderOptions, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import {
    COLUMN_RENAMED_CODE_GML,
    DUPLICATE_NODE_CODE,
    ELEMENT_TYPE_CODE,
    FLAG_TYPE_CODE,
    FLAG_VALUE_CODE,
    gmlImporter,
    ID_DROPPED_CODE,
    ID_TYPE_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MISSING_LABEL_CODE,
    NO_GRAPH_CODE,
    PRECISION_CODE,
    REPEATED_KEY_CODE,
    ROLE_TAKEN_CODE,
    SECOND_GRAPH_CODE,
} from "../../../src/formats/gml/importer.js";
import {
    LIST_START_MARKER,
    ORIGINAL_ID_KEY,
    SYNTAX_BRACKET_CODE,
    SYNTAX_STRING_CODE,
    SYNTAX_TOKEN_CODE,
} from "../../../src/formats/gml/syntax.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../../src/types.js";
import {
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
} from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs } from "../../helpers/roundtrip.js";

type Options = Parameters<typeof gmlImporter.import>[2];

interface Imported {
    snapshot: GraphSnapshot;
    report: ImportReport;
    builder: GraphBuilder;
}

async function importGml(
    input: Parameters<typeof gmlImporter.import>[0],
    options?: Options,
    builderOptions?: Partial<GraphBuilderOptions>,
): Promise<Imported> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64", ...builderOptions });
    const report = await gmlImporter.import(input, builder, options);
    return { snapshot: builder.freeze(), report, builder };
}

async function importError(input: string, options?: Options): Promise<ImportError> {
    try {
        await importGml(input, options);
    } catch (err) {
        expect(err).toBeInstanceOf(ImportError);
        return err as ImportError;
    }
    throw new Error("expected an ImportError");
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

function ids(s: GraphSnapshot): (string | number)[] {
    return s.ids.toArray();
}

function edges(s: GraphSnapshot): string[] {
    const el = s.edgeList();
    const out: string[] = [];
    for (let e = 0; e < s.edgeCount; e++) {
        out.push(`${String(s.ids.idOf(el.src[e]))}-${String(s.ids.idOf(el.dst[e]))}`);
    }
    return out;
}

function column(s: GraphSnapshot, table: "nodes" | "edges" | "graph", name: string): unknown[] {
    const c = s[table].require(name);
    return Array.from({ length: c.length }, (_, r) => (c.isSet(r) ? c.value(r) : undefined));
}

describe("gmlImporter: plugin shape", () => {
    it("declares the format, extensions and mime types", () => {
        expect(gmlImporter.format).toBe("gml");
        expect(gmlImporter.extensions).toEqual([".gml"]);
        expect(gmlImporter.mimeTypes).toContain("text/plain");
        expect(Object.isFrozen(gmlImporter)).toBe(true);
    });

    it("sniffs GML heads", () => {
        const sniff = gmlImporter.sniff as (head: Uint8Array) => number;
        const enc = new TextEncoder();
        expect(sniff(enc.encode("graph [\n  node [ id 1 ]\n]"))).toBe(0.95);
        expect(sniff(enc.encode("# comment\n\ngraph\n[\n"))).toBe(0.95);
        expect(sniff(readCorpusBytes("gml", "karate.gml").subarray(0, 256))).toBe(0.85);
        expect(sniff(enc.encode('Creator "x"\nVersion 1\n'))).toBe(0.4);
        expect(sniff(enc.encode('<?xml version="1.0"?><graphml>'))).toBe(0);
        expect(sniff(enc.encode("Source,Target\n1,2\n"))).toBe(0);
        expect(sniff(enc.encode("digraph g { a -> b }"))).toBe(0);
        expect(sniff(new Uint8Array([0xef, 0xbb, 0xbf, ...enc.encode("graph [")]))).toBe(0.95);
    });
});

describe("gmlImporter: corpus", () => {
    for (const entry of corpusFiles("gml")) {
        it(`imports ${entry.path} with the manifest counts`, async () => {
            const { snapshot, report } = await importGml(readCorpusText("gml", entry.path));
            expect(snapshot.nodeCount).toBe(entry.expectedNodes);
            expect(snapshot.edgeCount).toBe(entry.expectedEdges);
            expect(snapshot.directed).toBe(false);
            expect(report.format).toBe("gml");
            expect(report.counts).toEqual({
                nodes: entry.expectedNodes,
                edges: entry.expectedEdges,
                skippedNodes: 0,
                skippedEdges: 0,
                expandedMixed: 0,
            });
            expect(report.errorCount).toBe(0);
            expect(report.warningCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(report.lossy).toEqual([]);
            expect(snapshot.ids.kind).toBe("identity");
            expect(snapshot.meta.sourceFormat).toBe("gml");
            expect(snapshot.flags.weighted).toBe(false);
        });

        it(`reads ${entry.path} identically from every input shape`, async () => {
            const bytes = readCorpusBytes("gml", entry.path);
            const reference = (await importGml(readCorpusText("gml", entry.path))).snapshot;
            for (const shape of inputShapes(bytes)) {
                const { snapshot } = await importGml(shape.make());
                const diffs = compareSnapshots(reference, snapshot, { allowExtraColumns: false, originType: true });
                expect(diffs, `${shape.name}: ${describeDiffs(diffs)}`).toEqual([]);
            }
        });
    }

    it("karate: 1-based integer ids on the identity fast path, edges in file order, Creator in meta", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "karate.gml"));
        expect(snapshot.ids.offset).toBe(1);
        expect(snapshot.ids.idOf(0)).toBe(1);
        expect(snapshot.ids.idOf(33)).toBe(34);
        expect(edges(snapshot).slice(0, 4)).toEqual(["2-1", "3-1", "3-2", "4-1"]);
        expect(snapshot.meta.creator).toBe("Mark Newman on Fri Jul 21 12:39:27 2006");
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.edges.names()).toEqual([]);
        expect(snapshot.graph.names()).toEqual([]);
    });

    it("dolphins: 0-based ids and a string label column with the label role", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "dolphins.gml"));
        expect(snapshot.ids.offset).toBe(0);
        const label = snapshot.nodes.require("label");
        expect(label.dtype).toBe("string");
        expect(label.meta.role).toBe("label");
        expect(label.meta.origin).toEqual({ format: "gml", id: "label", title: null, type: "string", namespace: null });
        expect(label.value(0)).toBe("Beak");
        expect(label.value(1)).toBe("Beescratch");
        expect(label.value(61)).toBe("Zipfel");
        expect(snapshot.nodes.byRole("label")).toBe(label);
    });

    it("polbooks: value is a dict column by the cardinality heuristic", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "polbooks.gml"));
        const value = snapshot.nodes.require("value");
        expect(value.dtype).toBe("dict");
        expect(value.meta.origin?.type).toBe("string");
        expect(value.value(0)).toBe("n");
        expect(value.value(1)).toBe("c");
        expect(value.value(104)).toBe("n");
        expect(value.value(30)).toBe("l");
        expect(snapshot.nodes.require("label").value(0)).toBe("1000 Years for Revenge");
        expect(snapshot.nodes.require("label").value(2)).toBe("Charlie Wilson's War");
    });

    it("polbooks: the dictionaries option keeps value a string column", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "polbooks.gml"), { dictionaries: false });
        expect(snapshot.nodes.require("value").dtype).toBe("string");
    });

    it("football: value is an i32 column with origin int; a raw & in a label survives", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "football.gml"));
        const value = snapshot.nodes.require("value");
        expect(value.dtype).toBe("i32");
        expect(value.meta.origin?.type).toBe("int");
        expect(value.value(0)).toBe(7);
        expect(value.value(1)).toBe(0);
        const labels = column(snapshot, "nodes", "label");
        expect(labels).toContain("TexasA&M");
        expect(labels[0]).toBe("BrighamYoung");
    });

    it("minimal: a directed 0 header without a Creator", async () => {
        const { snapshot } = await importGml(readCorpusText("gml", "minimal.gml"));
        expect(snapshot.meta.creator).toBeNull();
        expect(ids(snapshot)).toEqual([1, 2, 3]);
        expect(edges(snapshot)).toEqual(["1-2", "2-3"]);
    });
});

describe("gmlImporter: malformed corpus", () => {
    const fatal: Record<string, string> = {
        "empty-file.gml": NO_GRAPH_CODE,
        "garbage-content.gml": SYNTAX_TOKEN_CODE,
        "no-graph-wrapper.gml": NO_GRAPH_CODE,
        "unclosed-bracket.gml": SYNTAX_BRACKET_CODE,
        "unclosed-string.gml": SYNTAX_STRING_CODE,
    };
    const recoverable: Record<string, { code: string; skippedNodes: number; skippedEdges: number }> = {
        "invalid-value-type.gml": { code: ID_TYPE_CODE, skippedNodes: 1, skippedEdges: 0 },
        "missing-edge-target.gml": { code: MISSING_ENDPOINT_CODE, skippedNodes: 0, skippedEdges: 1 },
        "missing-id.gml": { code: MISSING_ID_CODE, skippedNodes: 1, skippedEdges: 0 },
    };

    it("covers every malformed file", () => {
        expect([...Object.keys(fatal), ...Object.keys(recoverable)].sort()).toEqual([...malformedFiles("gml")]);
    });

    for (const name of malformedFiles("gml")) {
        it(`${name}: ImportError carrying a report under errorLimit 0`, async () => {
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            let error: ImportError | null = null;
            try {
                await gmlImporter.import(readMalformedBytes("gml", name), builder, { errorLimit: 0 });
            } catch (err) {
                expect(err).toBeInstanceOf(ImportError);
                error = err as ImportError;
            }
            expect(error).not.toBeNull();
            const { report } = error as ImportError;
            expect(report.format).toBe("gml");
            expect(report.errorCount).toBeGreaterThanOrEqual(1);
            expect(report.issues[0].severity).toBe("error");
            expect(report.issues[0].code).toBe(fatal[name] ?? recoverable[name].code);
            expect(Object.isFrozen(report)).toBe(true);
            expect((error as ImportError).code).toBe("E_IMPORT");
        });
    }

    for (const [name, code] of Object.entries(fatal)) {
        it(`${name}: fatal under the default error limit`, async () => {
            const error = await importError(new TextDecoder().decode(readMalformedBytes("gml", name)));
            expect(error.report.issues).toHaveLength(1);
            expect(error.report.issues[0]).toMatchObject({ category: "parse-error", code });
            expect(error.report.truncated).toBe(false);
        });
    }

    for (const [name, expected] of Object.entries(recoverable)) {
        it(`${name}: recovers with a report under the default error limit`, async () => {
            const { report, snapshot } = await importGml(new TextDecoder().decode(readMalformedBytes("gml", name)));
            expect(codes(report)).toEqual([expected.code]);
            expect(report.errorCount).toBe(1);
            expect(report.counts.skippedNodes).toBe(expected.skippedNodes);
            expect(report.counts.skippedEdges).toBe(expected.skippedEdges);
            expect(report.issues[0].line).toBeGreaterThan(0);
            expect(snapshot.nodeCount).toBeGreaterThan(0);
        });
    }

    it("reports the line of an unclosed string and of an unclosed bracket", async () => {
        const unclosedString = await importError(
            new TextDecoder().decode(readMalformedBytes("gml", "unclosed-string.gml")),
        );
        expect(unclosedString.report.issues[0].line).toBe(4);
        const unclosedBracket = await importError(
            new TextDecoder().decode(readMalformedBytes("gml", "unclosed-bracket.gml")),
        );
        expect(unclosedBracket.report.issues[0].line).toBe(1);
    });
});

describe("gmlImporter: structure and flags", () => {
    it("accepts graph [ on one line and graph\\n[ on two, comments and CRLF", async () => {
        const one = await importGml("graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] ]");
        const two = await importGml(
            "# header\r\ngraph\r\n[\r\n  node\r\n  [\r\n    id 1\r\n  ]\r\n  node [ id 2 ] # trailing\r\n  edge [ source 1 target 2 ]\r\n]\r\n",
        );
        expect(compareSnapshots(one.snapshot, two.snapshot, { allowExtraColumns: false })).toEqual([]);
        expect(ids(two.snapshot)).toEqual([1, 2]);
    });

    it("reads directed 1 as directed and an absent flag as defaultDirected", async () => {
        expect((await importGml("graph [ directed 1 edge [ source 1 target 2 ] ]")).snapshot.directed).toBe(true);
        expect((await importGml("graph [ edge [ source 1 target 2 ] ]")).snapshot.directed).toBe(false);
        expect(
            (await importGml("graph [ edge [ source 1 target 2 ] ]", { defaultDirected: true })).snapshot.directed,
        ).toBe(true);
        expect(
            (await importGml("graph [ directed 0 edge [ source 1 target 2 ] ]", { defaultDirected: true })).snapshot
                .directed,
        ).toBe(false);
    });

    it("reads a directed flag that appears after the edges", async () => {
        const { snapshot } = await importGml("graph [ edge [ source 1 target 2 ] directed 1 ]");
        expect(snapshot.directed).toBe(true);
        expect(edges(snapshot)).toEqual(["1-2"]);
    });

    it("warns on a directed flag outside 0 / 1 and on a repeated flag; errors on a non-integer flag", async () => {
        const odd = await importGml("graph [ directed 2 directed 0 edge [ source 1 target 2 ] ]");
        expect(odd.snapshot.directed).toBe(true);
        expect(codes(odd.report)).toEqual([FLAG_VALUE_CODE, FLAG_VALUE_CODE]);
        expect(odd.report.issues.every((i) => i.severity === "warning")).toBe(true);
        const bad = await importGml('graph [ directed "yes" edge [ source 1 target 2 ] ]', { defaultDirected: true });
        expect(bad.snapshot.directed).toBe(true);
        expect(bad.report.issues[0]).toMatchObject({ code: FLAG_TYPE_CODE, severity: "error", element: "directed" });
    });

    it("records the multigraph flag in meta.declaredMultigraph", async () => {
        expect((await importGml("graph [ multigraph 1 ]")).snapshot.meta.declaredMultigraph).toBe(true);
        expect((await importGml("graph [ multigraph 0 ]")).snapshot.meta.declaredMultigraph).toBe(false);
        expect((await importGml("graph [ ]")).snapshot.meta.declaredMultigraph).toBeNull();
    });

    it("keeps parallel edges and self-loops", async () => {
        const { snapshot } = await importGml(
            "graph [ multigraph 1 edge [ source 1 target 2 ] edge [ source 1 target 2 ] edge [ source 2 target 2 ] ]",
        );
        expect(snapshot.edgeCount).toBe(3);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.selfLoopCount).toBe(1);
    });

    it("orders nodes by node block and resolves forward and interleaved edge references", async () => {
        const text =
            "graph [ edge [ source 9 target 8 ] node [ id 8 ] edge [ source 8 target 7 ] node [ id 7 ] node [ id 9 ] ]";
        const { snapshot, report } = await importGml(text);
        expect(ids(snapshot)).toEqual([8, 7, 9]);
        expect(edges(snapshot)).toEqual(["9-8", "8-7"]);
        expect(report.counts).toMatchObject({ nodes: 3, edges: 2 });
    });

    it("creates undeclared endpoints by default and refuses them under addMissingNodes false", async () => {
        const text = "graph [ node [ id 1 ] edge [ source 1 target 2 ] ]";
        const lenient = await importGml(text);
        expect(ids(lenient.snapshot)).toEqual([1, 2]);
        const strict = await importGml(text, { addMissingNodes: false }, { addMissingNodes: false });
        expect(ids(strict.snapshot)).toEqual([1]);
        expect(strict.snapshot.edgeCount).toBe(0);
        expect(strict.report.issues[0]).toMatchObject({
            category: "missing-value",
            code: "E_UNKNOWN_NODE",
            element: "edge #0",
        });
        expect(strict.report.counts.skippedEdges).toBe(1);
    });

    it("refuses more than one graph block", async () => {
        const error = await importError("graph [ ] graph [ ]");
        expect(error.report.issues[0].code).toBe(SECOND_GRAPH_CODE);
    });

    it("reports node and edge keys whose value is not a block", async () => {
        const { report, snapshot } = await importGml(
            "graph [ node 5 node [ id 1 ] edge 3 edge [ source 1 target 1 ] ]",
        );
        expect(codes(report)).toEqual([ELEMENT_TYPE_CODE, ELEMENT_TYPE_CODE]);
        expect(report.counts).toMatchObject({ nodes: 1, edges: 1, skippedNodes: 1, skippedEdges: 1 });
        expect(snapshot.nodeCount).toBe(1);
    });

    it("reports duplicate node ids, duplicate structural keys and non-integer endpoints", async () => {
        const { report, snapshot } = await importGml(
            'graph [ node [ id 1 ] node [ id 1 ] node [ id 2 id 3 ] edge [ source 1 source 2 target 1 ] edge [ source 1.5 target 1 ] edge [ source 1 target "x" ] ]',
        );
        expect(codes(report)).toEqual([
            DUPLICATE_NODE_CODE,
            REPEATED_KEY_CODE,
            REPEATED_KEY_CODE,
            ID_TYPE_CODE,
            ID_TYPE_CODE,
        ]);
        // a repeated node id merges into the first declaration (a warning, as every importer records it)
        expect(report.issues[0]).toMatchObject({ severity: "warning", category: "merged" });
        expect(report.counts).toMatchObject({ nodes: 1, edges: 0, skippedNodes: 1, skippedEdges: 3 });
        expect(ids(snapshot)).toEqual([1]);
    });

    it("skips a node without an id and a string or real id", async () => {
        const { report, snapshot } = await importGml(
            'graph [ node [ label "a" ] node [ id "n1" ] node [ id 1.5 ] node [ id 4 ] ]',
        );
        expect(codes(report)).toEqual([MISSING_ID_CODE, ID_TYPE_CODE, ID_TYPE_CODE]);
        expect(report.issues[0].category).toBe("missing-value");
        expect(report.issues[1].category).toBe("validation-error");
        expect(report.issues[1].message).toContain('the string "n1"');
        expect(ids(snapshot)).toEqual([4]);
    });
});

describe("gmlImporter: ids", () => {
    it("coerces integer ids to numbers under canonical, keeping 01 as 1 and -0 as 0", async () => {
        const { snapshot } = await importGml(
            "graph [ node [ id 01 ] node [ id -0 ] node [ id 5 ] edge [ source 1 target 0 ] ]",
        );
        expect(ids(snapshot)).toEqual([1, 0, 5]);
        expect(edges(snapshot)).toEqual(["1-0"]);
    });

    it("keeps an integer beyond the safe range as text", async () => {
        const { snapshot } = await importGml(
            "graph [ node [ id 12345678901234567890 ] edge [ source 12345678901234567890 target 1 ] ]",
        );
        expect(ids(snapshot)).toEqual(["12345678901234567890", 1]);
        expect(snapshot.ids.kind).toBe("mixed");
    });

    it("applies ids: string and ids: number", async () => {
        const text = "graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] ]";
        expect(ids((await importGml(text, { ids: "string" })).snapshot)).toEqual(["1", "2"]);
        expect(ids((await importGml(text, { ids: "number" })).snapshot)).toEqual([1, 2]);
        expect(ids((await importGml(text, { ids: "keep" })).snapshot)).toEqual([1, 2]);
    });

    it("rejects an unknown ids option", async () => {
        await expect(importGml("graph [ ]", { ids: "float" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("nodeIdFrom label: labels become the ids, edges resolve through the integer ids, the id keys are a loss note", async () => {
        const text =
            'graph [ node [ id 0 label "a" ] node [ id 1 label "7" ] edge [ source 0 target 1 ] edge [ source 1 target 5 ] ]';
        const { snapshot, report } = await importGml(text, { nodeIdFrom: "label" });
        expect(ids(snapshot)).toEqual(["a", 7, 5]);
        expect(edges(snapshot)).toEqual(["a-7", "7-5"]);
        expect(column(snapshot, "nodes", "label")).toEqual(["a", "7", undefined]);
        expect(snapshot.nodes.require("label").meta.role).toBe("label");
        expect(report.lossy).toEqual([{ code: ID_DROPPED_CODE, message: expect.any(String), column: null, count: 2 }]);
        expect(report.lossy[0].message).toContain("label");
    });

    it("nodeIdFrom label: a missing label and a duplicate label are errors", async () => {
        const { report, snapshot } = await importGml(
            'graph [ node [ id 0 ] node [ id 1 label "a" ] node [ id 2 label "a" ] ]',
            {
                nodeIdFrom: "label",
            },
        );
        expect(codes(report)).toEqual([MISSING_LABEL_CODE, DUPLICATE_NODE_CODE]);
        expect(ids(snapshot)).toEqual(["a"]);
    });

    it("nodeIdFrom index: ids are the node ordinals", async () => {
        const text = "graph [ node [ id 10 ] node [ id 20 ] edge [ source 20 target 10 ] ]";
        const { snapshot } = await importGml(text, { nodeIdFrom: "index" });
        expect(ids(snapshot)).toEqual([0, 1]);
        expect(edges(snapshot)).toEqual(["1-0"]);
    });

    it("restores mangled ids from graphty_originalId and resolves edges through the mangled ids", async () => {
        const text = `graph [ node [ id 0 ${ORIGINAL_ID_KEY} "Little Rock, AR" ] node [ id 1 ${ORIGINAL_ID_KEY} 1.5 ] node [ id 2 ] edge [ source 0 target 1 ] edge [ source 2 target 0 ] ]`;
        const { snapshot } = await importGml(text);
        expect(ids(snapshot)).toEqual(["Little Rock, AR", 1.5, 2]);
        expect(edges(snapshot)).toEqual(["Little Rock, AR-1.5", "2-Little Rock, AR"]);
        expect(snapshot.nodes.names()).toEqual([]);
        const kept = await importGml(text, { restoreMangledIds: false });
        expect(ids(kept.snapshot)).toEqual([0, 1, 2]);
        expect(column(kept.snapshot, "nodes", ORIGINAL_ID_KEY)).toEqual(["Little Rock, AR", "1.5", undefined]);
    });

    it("rejects a record as graphty_originalId", async () => {
        const { report } = await importGml(`graph [ node [ id 0 ${ORIGINAL_ID_KEY} [ a 1 ] ] ]`);
        expect(report.issues[0]).toMatchObject({ code: "E_INVALID_ID", category: "validation-error" });
    });

    it("rejects a string id holding a lone surrogate reference", async () => {
        const { report, snapshot } = await importGml(
            'graph [ node [ id 0 label "&#55296;" ] node [ id 1 label "ok" ] ]',
            {
                nodeIdFrom: "label",
            },
        );
        expect(report.issues).toHaveLength(1);
        expect(report.issues[0].code).toBe("E_INVALID_ID");
        expect(ids(snapshot)).toEqual(["ok"]);
    });
});

describe("gmlImporter: weights", () => {
    it("reads value as the weight by default, int origin", async () => {
        const { snapshot } = await importGml("graph [ edge [ source 1 target 2 value 3 ] edge [ source 2 target 3 ] ]");
        expect(snapshot.flags.weighted).toBe(true);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([3, 1]);
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(1)).toBe(false);
        expect(snapshot.meta.weightOrigin).toEqual({
            format: "gml",
            id: "value",
            title: null,
            type: "int",
            namespace: null,
        });
        expect(snapshot.edges.names()).toEqual(["graphty.weight"]);
    });

    it("keeps real weights exact in f64 and accepts numeric strings", async () => {
        const { snapshot } = await importGml(
            'graph [ edge [ source 1 target 2 value 0.1 ] edge [ source 2 target 3 value "2" ] ]',
        );
        expect(snapshot.edges.byRole("weight")?.value(0)).toBe(0.1);
        expect(snapshot.edges.byRole("weight")?.value(1)).toBe(2);
        expect(snapshot.meta.weightOrigin?.type).toBe("real");
    });

    it("honours weightFrom and weightFrom null", async () => {
        const text = "graph [ edge [ source 1 target 2 value 3 weight 2.5 ] ]";
        const weight = await importGml(text, { weightFrom: "weight" });
        expect(weight.snapshot.edges.byRole("weight")).toBeNull();
        expect(weight.snapshot.edgeList().weights?.[0]).toBe(2.5);
        expect(weight.snapshot.meta.weightOrigin?.id).toBe("weight");
        expect(column(weight.snapshot, "edges", "value")).toEqual([3]);
        const none = await importGml(text, { weightFrom: null });
        expect(none.snapshot.flags.weighted).toBe(false);
        expect(none.snapshot.edges.names()).toEqual(["value", "weight"]);
        expect(none.snapshot.meta.weightOrigin).toBeNull();
    });

    it("skips an edge whose weight is text, NaN or a record", async () => {
        const { report, snapshot } = await importGml(
            'graph [ edge [ source 1 target 2 value "abc" ] edge [ source 1 target 2 value NAN ] edge [ source 1 target 2 value [ a 1 ] ] edge [ source 1 target 2 value +INF ] ]',
        );
        expect(codes(report)).toEqual(["E_INVALID_WEIGHT", "E_INVALID_WEIGHT", "E_INVALID_WEIGHT"]);
        expect(report.counts.skippedEdges).toBe(3);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.edgeList().weights?.[0]).toBe(Infinity);
    });
});

describe("gmlImporter: attribute typing", () => {
    it("types columns by the union of the value classes with origin.type per column", async () => {
        const text = `graph [
            node [ id 1 i 1 r 1.5 s "x" m 1 t 1 w 3000000000 ]
            node [ id 2 i 2 r 2 s "y" m 2.5 t "z" w 1 ]
        ]`;
        const { snapshot } = await importGml(text);
        const meta = (name: string): [string, string | null | undefined] => {
            const c = snapshot.nodes.require(name);
            return [c.dtype, c.meta.origin?.type];
        };
        expect(meta("i")).toEqual(["i32", "int"]);
        expect(meta("r")).toEqual(["f64", "real"]);
        expect(meta("s")).toEqual(["string", "string"]);
        expect(meta("m")).toEqual(["f64", "real"]);
        expect(meta("t")).toEqual(["string", "string"]);
        expect(meta("w")).toEqual(["f64", "int"]);
        expect(column(snapshot, "nodes", "m")).toEqual([1, 2.5]);
        expect(column(snapshot, "nodes", "t")).toEqual(["1", "z"]);
        expect(column(snapshot, "nodes", "w")).toEqual([3000000000, 1]);
    });

    it("warns once about integers beyond 2^53", async () => {
        const { snapshot, report } = await importGml(
            "graph [ node [ id 1 big 9007199254740993 ] node [ id 2 big 9007199254740995 ] ]",
        );
        expect(snapshot.nodes.require("big").dtype).toBe("f64");
        expect(codes(report)).toEqual([PRECISION_CODE]);
        expect(report.issues[0]).toMatchObject({ category: "precision", severity: "warning", element: "big" });
    });

    it("decodes character references and keeps other text exact", async () => {
        const { snapshot } = await importGml(
            'graph [ node [ id 1 label "&#65;&amp;&quot;B&quot; &#x263A; TexasA&M" ] ]',
        );
        expect(snapshot.nodes.require("label").value(0)).toBe(`A&"B" ${String.fromCodePoint(0x263a)} TexasA&M`);
    });

    it("reports a value with a lone surrogate reference and keeps the node", async () => {
        const { snapshot, report } = await importGml(
            'graph [ node [ id 1 s "&#55296;" t "ok" ] node [ id 2 s "fine" ] ]',
        );
        expect(ids(snapshot)).toEqual([1, 2]);
        expect(report.issues).toHaveLength(1);
        expect(report.issues[0]).toMatchObject({ code: "E_COLUMN_TYPE", element: "1", line: 1 });
        expect(report.counts.skippedNodes).toBe(0);
        expect(column(snapshot, "nodes", "s")).toEqual([undefined, "fine"]);
    });

    it("maps nested records to json and keeps NetworkX list conventions inside them", async () => {
        const text = `graph [ node [ id 1 graphics [ w 30 fill "#fff" Line [ point [ x 1 y 2 ] point [ x 3 y 4 ] ] ] ] edge [ source 1 target 1 meta [ tags "a" tags "b" one "${LIST_START_MARKER}" one 5 none "[]" ] ] ]`;
        const { snapshot } = await importGml(text);
        const graphics = snapshot.nodes.require("graphics");
        expect(graphics.dtype).toBe("json");
        expect(graphics.meta.origin?.type).toBe("record");
        expect(graphics.value(0)).toEqual({
            w: 30,
            fill: "#fff",
            Line: {
                point: [
                    { x: 1, y: 2 },
                    { x: 3, y: 4 },
                ],
            },
        });
        expect(snapshot.edges.require("meta").value(0)).toEqual({ tags: ["a", "b"], one: [5], none: [] });
    });

    it("widens a key that is a record in one element and a scalar in another to json", async () => {
        const { snapshot } = await importGml('graph [ node [ id 1 a [ x 1 ] ] node [ id 2 a 5 ] node [ id 3 a "s" ] ]');
        expect(snapshot.nodes.require("a").dtype).toBe("json");
        expect(column(snapshot, "nodes", "a")).toEqual([{ x: 1 }, 5, "s"]);
    });

    it("turns repeated keys into list columns with the marker and empty-list conventions", async () => {
        const text = `graph [
            node [ id 1 tags "a" tags "b" ]
            node [ id 2 tags "${LIST_START_MARKER}" tags "c" ]
            node [ id 3 tags "[]" ]
            node [ id 4 tags "d" ]
            node [ id 5 ]
            edge [ source 1 target 2 ws 1 ws 2.5 ]
            edge [ source 2 target 3 ws 7 ]
        ]`;
        const { snapshot } = await importGml(text);
        const tags = snapshot.nodes.require("tags");
        expect(tags.dtype).toBe("list");
        expect(tags.meta.itemDtype).toBe("string");
        expect(tags.meta.origin?.type).toBe("string");
        expect(column(snapshot, "nodes", "tags")).toEqual([["a", "b"], ["c"], [], ["d"], undefined]);
        const ws = snapshot.edges.require("ws");
        expect(ws.meta.itemDtype).toBe("f64");
        expect(ws.meta.origin?.type).toBe("real");
        expect(column(snapshot, "edges", "ws")).toEqual([[1, 2.5], [7]]);
    });

    it("keeps lists of records and mixed lists", async () => {
        const { snapshot } = await importGml('graph [ node [ id 1 p [ x 1 ] p [ x 2 ] q 1 q "s" ] ]');
        const p = snapshot.nodes.require("p");
        expect(p.dtype).toBe("list");
        expect(p.meta.itemDtype).toBe("json");
        expect(column(snapshot, "nodes", "p")).toEqual([[{ x: 1 }, { x: 2 }]]);
        expect(snapshot.nodes.require("q").meta.itemDtype).toBe("string");
        expect(column(snapshot, "nodes", "q")).toEqual([["1", "s"]]);
    });

    it("gives edge id, key and label keys their roles", async () => {
        const { snapshot } = await importGml('graph [ edge [ source 1 target 2 id 7 key 0 label "e" ] ]');
        expect(snapshot.edges.byRole("id")?.meta.name).toBe("id");
        expect(snapshot.edges.byRole("key")?.meta.name).toBe("key");
        expect(snapshot.edges.byRole("label")?.value(0)).toBe("e");
        expect(snapshot.edgeIndexOf(7)).toBe(0);
    });

    it("stores non-finite reals", async () => {
        const { snapshot } = await importGml("graph [ node [ id 1 a +INF b -inf c NAN ] ]");
        expect(column(snapshot, "nodes", "a")).toEqual([Infinity]);
        expect(column(snapshot, "nodes", "b")).toEqual([-Infinity]);
        expect(Number.isNaN(column(snapshot, "nodes", "c")[0])).toBe(true);
    });
});

describe("gmlImporter: graphics and positions", () => {
    it("maps graphics x / y / z to the position role and keeps the rest as json", async () => {
        const text = `graph [
            node [ id 1 graphics [ x 1.5 y 2 z 3 w 10 fill "#f00" ] ]
            node [ id 2 graphics [ x 4 y 5 ] ]
            node [ id 3 graphics [ w 20 ] ]
            node [ id 4 ]
        ]`;
        const { snapshot } = await importGml(text);
        const position = snapshot.nodes.require("position");
        expect(position.meta.role).toBe("position");
        expect(position.dtype).toBe("f64");
        expect(position.meta.components).toBe(3);
        expect(position.meta.origin).toEqual({
            format: "gml",
            id: "graphics",
            title: null,
            type: "real",
            namespace: null,
        });
        expect(column(snapshot, "nodes", "position")).toEqual([
            Float64Array.of(1.5, 2, 3),
            Float64Array.of(4, 5, 0),
            undefined,
            undefined,
        ]);
        expect(column(snapshot, "nodes", "graphics")).toEqual([
            { w: 10, fill: "#f00" },
            undefined,
            { w: 20 },
            undefined,
        ]);
        expect(snapshot.nodes.byRole("position")).toBe(position);
    });

    it("declares no graphics column when every record is a bare position", async () => {
        const { snapshot } = await importGml("graph [ node [ id 1 graphics [ x 1 y 2 ] ] ]");
        expect(snapshot.nodes.names()).toEqual(["position"]);
    });

    it("keeps an empty graphics record and a non-record graphics value as json", async () => {
        const { snapshot } = await importGml(
            "graph [ node [ id 1 graphics [ ] ] node [ id 2 graphics 5 graphics 6 ] ]",
        );
        expect(snapshot.nodes.names()).toEqual(["graphics"]);
        expect(snapshot.nodes.require("graphics").dtype).toBe("list");
    });

    it("positions: false keeps the whole record in the json column", async () => {
        const { snapshot } = await importGml("graph [ node [ id 1 graphics [ x 1 y 2 ] ] ]", { positions: false });
        expect(snapshot.nodes.names()).toEqual(["graphics"]);
        expect(snapshot.nodes.require("graphics").value(0)).toEqual({ x: 1, y: 2 });
    });

    it("renames the position column when a position key exists", async () => {
        const { snapshot } = await importGml("graph [ node [ id 1 position 3 graphics [ x 1 y 2 ] ] ]");
        expect(snapshot.nodes.names()).toEqual(["position", "position#graphics"]);
        expect(snapshot.nodes.byRole("position")?.meta.name).toBe("position#graphics");
    });

    it("does not map edge graphics", async () => {
        const { snapshot } = await importGml("graph [ edge [ source 1 target 2 graphics [ x 1 y 2 ] ] ]");
        expect(snapshot.edges.require("graphics").value(0)).toEqual({ x: 1, y: 2 });
    });
});

describe("gmlImporter: graph attributes and metadata", () => {
    it("reads in-graph keys as graph columns and Creator / Version into meta", async () => {
        const text =
            'Creator "me"\nVersion 1\ngraph [ name "g" count 2 ratio 0.5 tags "a" tags "b" info [ k 1 ] node [ id 1 ] ]';
        const { snapshot } = await importGml(text);
        expect(snapshot.meta.creator).toBe("me");
        expect(snapshot.meta.sourceVersion).toBe("1");
        expect(snapshot.graph.names()).toEqual(["name", "count", "ratio", "info", "tags"]);
        expect(snapshot.graph.value("name", 0)).toBe("g");
        expect(snapshot.graph.require("count").dtype).toBe("i32");
        expect(snapshot.graph.require("ratio").dtype).toBe("f64");
        expect(snapshot.graph.value("tags", 0)).toEqual(["a", "b"]);
        expect(snapshot.graph.value("info", 0)).toEqual({ k: 1 });
        expect(snapshot.graph.require("count").meta.origin?.type).toBe("int");
        expect(snapshot.graph.require("count").meta.extra).toEqual({});
    });

    it("keeps other top-level keys as graph columns flagged gmlTopLevel", async () => {
        const text = 'Creator "c"\nlicense "MIT"\ngraph [ license "GPL" ]\nseen 1\nseen 2\n';
        const { snapshot } = await importGml(text);
        expect([...snapshot.graph.names()].sort()).toEqual(["license", "license#license", "seen"]);
        expect(snapshot.graph.require("license").meta.extra).toEqual({});
        expect(snapshot.graph.value("license", 0)).toBe("GPL");
        expect(snapshot.graph.require("license#license").meta.extra).toEqual({ gmlTopLevel: true });
        expect(snapshot.graph.value("license#license", 0)).toBe("MIT");
        expect(snapshot.graph.require("seen").meta.extra).toEqual({ gmlTopLevel: true });
        expect(snapshot.graph.value("seen", 0)).toEqual([1, 2]);
    });

    it("accumulates a graph list across node blocks", async () => {
        const { snapshot } = await importGml("graph [ tag 1 node [ id 1 ] tag 2 ]");
        expect(snapshot.graph.value("tag", 0)).toEqual([1, 2]);
    });

    it("reads a numeric Creator as text", async () => {
        expect((await importGml("Creator 7\ngraph [ ]")).snapshot.meta.creator).toBe("7");
        expect((await importGml("Creator [ a 1 ]\ngraph [ ]")).snapshot.graph.value("Creator", 0)).toEqual({ a: 1 });
    });
});

describe("gmlImporter: direction with a caller's sink", () => {
    it("reports builder-policy options the caller's sink does not use (design 8.4 precedence)", async () => {
        const text = "graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] ]";
        const { report } = await importGml(
            text,
            { weightDtype: "f32", selfLoops: "drop", duplicateEdges: "keep" },
            { weightDtype: "f64", selfLoops: "keep", duplicateEdges: "keep" },
        );
        expect(report.issues.map((i) => [i.code, i.element, i.category])).toEqual([
            [SINK_OPTION_CODE, "selfLoops", "coercion"],
            [SINK_OPTION_CODE, "weightDtype", "coercion"],
        ]);
        // options left undefined are defaults, never requests
        const silent = await importGml(text, undefined, { weightDtype: "f32", selfLoops: "drop" });
        expect(codes(silent.report)).toEqual([]);
    });

    it("follows the file on an empty unlocked sink", async () => {
        const { snapshot } = await importGml("graph [ directed 1 edge [ source 1 target 2 ] ]", undefined, {
            directed: false,
        });
        expect(snapshot.directed).toBe(true);
    });

    it("expands an undirected file into a locked directed sink and reports the refusal", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        const report = await gmlImporter.import(
            "graph [ directed 0 edge [ source 1 target 2 ] edge [ source 2 target 2 ] ]",
            builder,
        );
        const snapshot = builder.freeze();
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(3);
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        expect(report.counts).toMatchObject({ edges: 3, expandedMixed: 2 });
        expect(snapshot.edges.byRole("pair")).not.toBeNull();
    });

    it("aborts under onMixedDirection error when the sink refuses the direction", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        let error: ImportError | null = null;
        try {
            await gmlImporter.import("graph [ directed 0 edge [ source 1 target 2 ] ]", builder, {
                onMixedDirection: "error",
            });
        } catch (err) {
            error = err as ImportError;
        }
        expect(error).toBeInstanceOf(ImportError);
        expect(codes((error as ImportError).report)).toEqual([DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE]);
    });

    it("forces the direction under onMixedDirection directed", async () => {
        const { snapshot, report } = await importGml("graph [ directed 0 edge [ source 1 target 2 ] ]", {
            onMixedDirection: "directed",
        });
        expect(snapshot.directed).toBe(true);
        expect(report.issues[0].code).toBe("W_DIRECTION_FORCED");
    });

    it("reuses an existing column of another declaration and drops a taken role", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.declareNodeColumn({ name: "value", dtype: "string" });
        builder.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
        const report = await gmlImporter.import('graph [ node [ id 1 value 3 label "x" ] ]', builder);
        const snapshot = builder.freeze();
        expect(codes(report).sort()).toEqual([COLUMN_RENAMED_CODE_GML, ROLE_TAKEN_CODE]);
        // design 5.6: the caller's string "value" column keeps its shape; the file's int goes to value#value
        expect(snapshot.nodes.require("value").isSet(0)).toBe(false);
        expect(snapshot.nodes.require("value#value").value(0)).toBe(3);
        expect(snapshot.nodes.require("label").meta.role).toBeNull();
        expect(snapshot.nodes.byRole("label")?.meta.name).toBe("name");
    });
});

describe("gmlImporter: limits, cancellation and input handling", () => {
    it("aborts with E_IMPORT and a truncated report beyond the error limit", async () => {
        const text = "graph [ node [ ] node [ ] node [ ] node [ id 1 ] ]";
        const error = await importError(text, { errorLimit: 1 });
        expect(error.report.truncated).toBe(true);
        expect(error.report.errorCount).toBe(2);
        expect(error.report.counts.skippedNodes).toBe(1);
        expect(error.message).toContain("error limit of 1 exceeded");
        const ok = await importGml(text, { errorLimit: 3 });
        expect(ok.report.errorCount).toBe(3);
        expect(ok.report.truncated).toBe(false);
        expect(ok.snapshot.nodeCount).toBe(1);
    });

    it("rejects with the abort reason when the signal is already aborted", async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(importGml("graph [ ]", { signal: controller.signal })).rejects.toMatchObject({
            name: "AbortError",
        });
    });

    it("stops between elements when the signal fires", async () => {
        const controller = new AbortController();
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        const nodes = Array.from({ length: 10000 }, (_, i) => `node [ id ${i} ]`).join(" ");
        const original = builder.addNode.bind(builder);
        let added = 0;
        builder.addNode = (id): number => {
            if (++added === 5000) {
                controller.abort();
            }
            return original(id);
        };
        await expect(
            gmlImporter.import(`graph [ ${nodes} ]`, builder, { signal: controller.signal }),
        ).rejects.toMatchObject({
            name: "AbortError",
        });
        expect(added).toBeLessThan(10000);
    });

    it("reports progress and strips a BOM", async () => {
        const calls: [number, number | undefined][] = [];
        const text = `${String.fromCharCode(0xfeff)}graph [ node [ id 1 ] ]`;
        const { snapshot } = await importGml(text, { onProgress: (done, total) => calls.push([done, total]) });
        expect(ids(snapshot)).toEqual([1]);
        expect(calls).toEqual([[text.length, text.length]]);
    });

    it("fails on invalid UTF-8", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        let error: ImportError | null = null;
        try {
            await gmlImporter.import(new Uint8Array([0x67, 0x72, 0xff, 0xfe]), builder);
        } catch (err) {
            error = err as ImportError;
        }
        expect(error).toBeInstanceOf(ImportError);
        expect((error as ImportError).report.issues[0].code).toBe(INVALID_UTF8_CODE);
    });

    it("lets a non-GraphFormatError thrown by the sink propagate (a bug, not an issue of the file)", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.addEdge = (): number => {
            throw new Error("boom");
        };
        await expect(gmlImporter.import("graph [ edge [ source 1 target 2 ] ]", builder)).rejects.toThrow("boom");
    });

    it("passes the common options through validation", async () => {
        const options: CommonImportOptions = { errorLimit: -1 };
        await expect(importGml("graph [ ]", options)).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
    });

    it("reserves capacity and counts elements for a larger file", async () => {
        const nodes = Array.from({ length: 5000 }, (_, i) => `node [ id ${i} label "n${i}" ]`).join("\n");
        const links = Array.from(
            { length: 5000 },
            (_, i) => `edge [ source ${i} target ${(i + 1) % 5000} value ${i % 7} ]`,
        ).join("\n");
        const { snapshot, report } = await importGml(`graph [ directed 1\n${nodes}\n${links}\n]`);
        expect(snapshot.nodeCount).toBe(5000);
        expect(snapshot.edgeCount).toBe(5000);
        expect(snapshot.ids.kind).toBe("identity");
        expect(report.counts).toMatchObject({ nodes: 5000, edges: 5000 });
        expect(snapshot.edgeList().weights?.[8]).toBe(1);
    });
});
