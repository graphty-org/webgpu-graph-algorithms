import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { decodeChunks } from "../../../src/common/writer.js";
import { DOT_LOSS, dotExporter } from "../../../src/formats/dot/exporter.js";
import { dotImporter } from "../../../src/formats/dot/importer.js";
import { CLUSTER_COLUMN, PARENT_COLUMN } from "../../../src/formats/dot/names.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

type ExportOptions = Parameters<typeof dotExporter.check>[1];

async function imported(text: string, directed = true): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed, weightDtype: "f64" });
    await dotImporter.import(text, builder);
    return builder.freeze();
}

function noteCodes(snapshot: GraphSnapshot, options?: ExportOptions): string[] {
    return dotExporter.check(snapshot, options).map((n) => n.code);
}

function lines(text: string): string[] {
    return text
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
}

describe("dot exporter: capabilities and shape", () => {
    it("declares what DOT can hold", () => {
        expect(dotExporter.format).toBe("dot");
        expect(dotExporter.capabilities).toMatchObject({
            mixedDirection: false,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "optional",
            idCharset: "any",
            dtypes: ["bool", "i32", "f64", "string"],
            components: false,
            lists: false,
            json: false,
            defaults: false,
            options: false,
            hierarchy: true,
            temporal: "none",
            graphAttributes: true,
            positions: true,
            viz: false,
        });
    });

    it("writes header, graph attributes, nodes in index order and edges in logical order", async () => {
        const snapshot = await imported(
            'strict digraph G { rankdir=LR; a [x=1]; b; a -> b [label="e"]; b -> a; c -> c }',
        );
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual([
            "strict digraph G {",
            "graph [rankdir=LR];",
            "a [x=1];",
            "b;",
            "c;",
            "a -> b [label=e];",
            "b -> a;",
            "c -> c;",
            "}",
        ]);
        expect(text.startsWith("strict digraph G {\n    graph")).toBe(true);
    });

    it("export() yields the same bytes exportToString() returns", async () => {
        const snapshot = await imported(readCorpusText("dot", "cluster.gv"));
        const text = await dotExporter.exportToString(snapshot);
        expect(await decodeChunks(dotExporter.export(snapshot))).toBe(text);
    });

    it("honours the name, strict and indent options", async () => {
        const snapshot = await imported("graph { a -- b }");
        const text = await dotExporter.exportToString(snapshot, { name: "my graph", strict: true, indent: "\t" });
        expect(text).toBe('strict graph "my graph" {\n\ta;\n\tb;\n\ta -- b;\n}\n');
        const anonymous = await dotExporter.exportToString(await imported("digraph G {}"), { name: null });
        expect(anonymous).toBe("digraph {\n}\n");
    });

    it("quotes ids and values exactly when the grammar needs it and writes HTML strings bare", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        for (const id of ["plain", "with space", 'has "quote"', "node", "Graph", "", "1e21", "-3", "1.5", "_x9"]) {
            builder.addNode(id);
        }
        builder.addNode(7);
        builder.addNode(-2.5);
        builder.setNodeValue("html", 0, "<<b>bold</b>>");
        builder.setNodeValue("notHtml", 0, "<a> <b>");
        builder.setNodeValue("odd", 0, "<a");
        builder.setNodeValue("t", 0, "true");
        builder.setNodeValue("v", 1, "");
        const text = await dotExporter.exportToString(builder.freeze());
        expect(lines(text).slice(1, 13)).toEqual([
            'plain [html=<<b>bold</b>>, notHtml="<a> <b>", odd="<a", t=true];',
            '"with space" [v=""];',
            '"has \\"quote\\"";',
            '"node";',
            '"Graph";',
            '"";',
            '"1e21";',
            "-3;",
            "1.5;",
            "_x9;",
            "7;",
            "-2.5;",
        ]);
    });

    it("formats bool, i32 and f64 cells so their dtype survives re-import", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.declareNodeColumn({ name: "flag", dtype: "bool" });
        builder.declareNodeColumn({ name: "count", dtype: "i32" });
        builder.declareNodeColumn({ name: "real", dtype: "f64" });
        builder.declareNodeColumn({ name: "big", dtype: "f64" });
        builder.declareNodeColumn({ name: "tiny", dtype: "f64" });
        builder.setNodeValue("flag", 0, false);
        builder.setNodeValue("count", 0, -12);
        builder.setNodeValue("real", 0, 2);
        builder.setNodeValue("big", 0, 1e21);
        builder.setNodeValue("tiny", 0, 1.5e-7);
        const snapshot = builder.freeze();
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)[1]).toBe('a [flag=false, count=-12, real=2.0, big="1e+21", tiny="1.5e-7"];');
        const back = (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot;
        expect(back.nodes.get("flag")?.dtype).toBe("bool");
        expect(back.nodes.get("count")?.dtype).toBe("i32");
        // "2.0" is f64 text under the 5.1 grammar: the importer widens the column to f64 although
        // the parsed value 2 alone would infer i32
        expect(back.nodes.get("real")?.dtype).toBe("f64");
        expect(back.nodes.get("real")?.value(0)).toBe(2);
        expect(back.nodes.get("big")?.dtype).toBe("f64");
        expect(back.nodes.get("big")?.value(0)).toBe(1e21);
        expect(back.nodes.get("tiny")?.dtype).toBe("f64");
        expect(back.nodes.get("tiny")?.value(0)).toBe(1.5e-7);
        expectSameSnapshot(snapshot, back, { dtypes: false });
    });

    it("writes explicit weights only, reading the shadow column's validity", async () => {
        const snapshot = await imported("digraph { a -> b [weight=2.5]; b -> c; c -> a [weight=1] }");
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text).slice(4)).toEqual(["a -> b [weight=2.5];", "b -> c;", "c -> a [weight=1];", "}"]);
        const back = (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot;
        expectSameSnapshot(snapshot, back);
    });

    it("writes f32 arc weights as the shortest round-tripping decimal when every weight is explicit", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f32" });
        builder.addEdge("a", "b", 0.1);
        builder.addEdge("b", "c", Infinity);
        const text = await dotExporter.exportToString(builder.freeze());
        expect(lines(text).slice(4, 6)).toEqual(["a -> b [weight=0.1];", "b -> c [weight=Infinity];"]);
        const back = await imported(text);
        expect(Array.from(back.edgeList().weights ?? [])).toEqual([Math.fround(0.1), Infinity]);
    });

    it("writes the edge id role as key, ports as endpoint suffixes", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "id", dtype: "i32", role: "id" });
        builder.declareEdgeColumn({ name: "from", dtype: "string", role: "sourcePort" });
        builder.declareEdgeColumn({ name: "to", dtype: "string", role: "targetPort" });
        const e = builder.addEdge("a", "b");
        builder.setEdgeValue("id", e, 7);
        builder.setEdgeValue("from", e, "p:ne");
        builder.setEdgeValue("to", e, "x y");
        const f = builder.addEdge("b", "a");
        builder.setEdgeValue("to", f, "s");
        const text = await dotExporter.exportToString(builder.freeze());
        expect(lines(text).slice(3, 5)).toEqual(['a:p:ne -> b:"x y" [key=7];', "b -> a:s;"]);
        const back = await imported(text);
        expect(back.edges.byRole("id")?.value(0)).toBe("7");
        expect(back.edges.byRole("sourcePort")?.value(0)).toBe("p:ne");
        expect(back.edges.byRole("targetPort")?.value(0)).toBe("x y");
    });

    it("writes node positions as pos with the source dimensions, edge and graph positions never", async () => {
        const snapshot = await imported('digraph { a [pos="1.5,2"]; b [pos="3,4!"] }');
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text).slice(1, 3)).toEqual(['a [pos="1.5,2"];', 'b [pin=true, pos="3,4"];']);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
        const threeD = await imported('digraph { a [pos="1,2,3"] }');
        expect(lines(await dotExporter.exportToString(threeD))[1]).toBe('a [pos="1,2,3"];');
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("n");
        builder.declareNodeColumn({ name: "xy", dtype: "f64", components: 2, role: "position" });
        builder.setNodeValue("xy", 0, [0.25, -1]);
        expect(lines(await dotExporter.exportToString(builder.freeze()))[1]).toBe('n [pos="0.25,-1"];');
    });
});

describe("dot exporter: clusters", () => {
    it("emits a cluster block for a container node with its attributes and nested members", async () => {
        const snapshot = await imported(
            'digraph { subgraph cluster_0 { label="one"; a; subgraph cluster_1 { b } } c; subgraph cluster_0 { c } }',
        );
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual([
            "digraph {",
            "subgraph cluster_0 {",
            "graph [label=one];",
            "a;",
            "subgraph cluster_1 {",
            "b;",
            "}",
            "c;",
            "}",
            "}",
        ]);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
    });

    it("writes cluster=true for a container whose id does not start with cluster", async () => {
        const snapshot = await imported("digraph { subgraph grp { cluster=true; a; b } }");
        expect(snapshot.ids.toArray()).toEqual(["grp", "a", "b"]);
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual(["digraph {", "subgraph grp {", "cluster=true;", "a;", "b;", "}", "}"]);
        const late = await imported("digraph { subgraph grp { a; b; cluster=true } }");
        expect(late.ids.toArray()).toEqual(["a", "b", "grp"]);
        expect(lines(await dotExporter.exportToString(late))).toEqual([
            "digraph {",
            "a;",
            "b;",
            "subgraph grp {",
            "cluster=true;",
            "a;",
            "b;",
            "}",
            "}",
        ]);
        expectSameSnapshot(late, (await roundTrip(late, dotExporter, dotImporter)).snapshot);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
    });

    it("writes a parent that is a real node as a node statement plus a grouping cluster", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("p");
        builder.addNode("c1");
        builder.addNode("c2");
        builder.declareNodeColumn({ name: "pid", dtype: "u32", role: "parent", refersTo: "node" });
        builder.setNodeValue("pid", 1, 0);
        builder.setNodeValue("pid", 2, 0);
        builder.setNodeValue("x", 0, 1);
        builder.addEdge("p", "c1");
        const snapshot = builder.freeze();
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual([
            "digraph {",
            "p [x=1];",
            "subgraph p {",
            "cluster=true;",
            "c1;",
            "c2;",
            "}",
            "p -> c1;",
            "}",
        ]);
        const { snapshot: back, report } = await roundTrip(snapshot, dotExporter, dotImporter);
        expect(report.issues.map((i) => i.code)).toEqual(["W_DOT_CLUSTER_NODE_MERGED"]);
        expect(back.ids.toArray()).toEqual(["p", "c1", "c2"]);
        expect(back.nodes.get(CLUSTER_COLUMN)?.value(0)).toBe(true);
        const diffs = compareSnapshots(snapshot, back, { ignoreColumns: [CLUSTER_COLUMN, PARENT_COLUMN, "pid"] });
        expect(describeDiffs(diffs)).toBe("no differences");
        const parent = back.nodes.byRole("parent");
        expect(parent?.value(1)).toBe(0);
        expect(parent?.value(2)).toBe(0);
    });

    it("keeps the node order when a member precedes its container", async () => {
        const snapshot = await imported("digraph { m; subgraph cluster_x { m; n } }");
        expect(snapshot.ids.toArray()).toEqual(["m", "cluster_x", "n"]);
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual(["digraph {", "m;", "subgraph cluster_x {", "m;", "n;", "}", "}"]);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
    });

    it("writes a container's position inside its cluster block", async () => {
        const snapshot = await imported('graph { subgraph cluster_p { pos="1,2"; a } }');
        expect(lines(await dotExporter.exportToString(snapshot))).toEqual([
            "graph {",
            "subgraph cluster_p {",
            'graph [pos="1,2"];',
            "a;",
            "}",
            "}",
        ]);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
    });

    it("ignores a parent that points at itself or out of range", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.declareNodeColumn({ name: "pid", dtype: "u32", role: "parent", refersTo: "node" });
        builder.setNodeValue("pid", 0, 0);
        expect(lines(await dotExporter.exportToString(builder.freeze()))).toEqual(["digraph {", "a;", "}"]);
    });
});

describe("dot exporter: mixed direction", () => {
    async function mixed(): Promise<GraphSnapshot> {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = new ImportReportBuilder("test", 100);
        const resolver = new DirectionResolver(builder, report, "expand");
        resolver.setHeader(true);
        resolver.addEdge("a", "b", "directed");
        resolver.addEdge("b", "c", "undirected");
        resolver.addEdge("c", "d", "mutual");
        builder.setEdgeValue("tag", 1, "pair");
        return Promise.resolve(builder.freeze());
    }

    it("refuses by default and reports the error in check()", async () => {
        const snapshot = await mixed();
        expect(noteCodes(snapshot)).toContain(LOSS.MIXED_DIRECTION_ERROR);
        await expect(dotExporter.exportToString(snapshot)).rejects.toMatchObject({ code: "E_DIRECTED" });
    });

    it("folds expanded pairs and writes every edge one way per onMixedDirection", async () => {
        const snapshot = await mixed();
        const directed = await dotExporter.exportToString(snapshot, { onMixedDirection: "directed" });
        expect(lines(directed).slice(5)).toEqual(["a -> b;", "b -> c [tag=pair];", "c -> d;", "d -> c;", "}"]);
        const notes = noteCodes(snapshot, { onMixedDirection: "directed" });
        expect(notes).toContain(LOSS.MIXED_DIRECTION);
        expect(notes).toContain(DOT_LOSS.MUTUAL_EXPANDED);
        const undirected = await dotExporter.exportToString(snapshot, { onMixedDirection: "undirected" });
        expect(lines(undirected)[0]).toBe("graph {");
        expect(lines(undirected).slice(5)).toEqual(["a -- b;", "b -- c [tag=pair];", "c -- d;", "d -- c;", "}"]);
        const back = await imported(undirected);
        expect(back.directed).toBe(false);
        expect(back.edgeCount).toBe(4);
    });

    it("writes a plain undirected graph with -- and re-imports it equal", async () => {
        const snapshot = await imported("graph { a -- b -- c [w=1]; c -- a }", false);
        expect(noteCodes(snapshot)).toEqual([]);
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
    });
});

describe("dot exporter: check() loss notes", () => {
    it("reports nothing for a snapshot DOT holds exactly", async () => {
        const snapshot = await imported(readCorpusText("dot", "cluster.gv"));
        expect(dotExporter.check(snapshot)).toEqual([]);
    });

    it("reports the generic capability gaps", () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        builder.declareNodeColumn({ name: "blob", dtype: "json" });
        builder.declareNodeColumn({ name: "single", dtype: "f32" });
        builder.declareNodeColumn({ name: "cat", dtype: "dict" });
        builder.declareNodeColumn({ name: "vec", dtype: "f64", components: 2 });
        builder.declareNodeColumn({ name: "dflt", dtype: "i32", default: 3 });
        builder.declareNodeColumn({ name: "opts", dtype: "i32", options: [1, 2] });
        builder.declareNodeColumn({ name: "since", dtype: "f64", role: "start" });
        builder.declareNodeColumn({
            name: "spells",
            dtype: "list",
            itemDtype: "f64",
            itemComponents: 2,
            role: "spells",
        });
        builder.declareNodeColumn({ name: "open", dtype: "u8", role: "open" });
        builder.declareNodeColumn({ name: "rgba", dtype: "f32", components: 4, role: "color" });
        builder.declareNodeColumn({ name: "dyn", dtype: "f64", dynamic: true });
        builder.setNodeValue("tags", 0, ["x"]);
        builder.setNodeValue("blob", 0, { k: 1 });
        builder.setNodeValue("single", 0, 1);
        builder.setNodeValue("cat", 0, "c");
        builder.setNodeValue("vec", 0, [1, 2]);
        builder.setNodeValue("rgba", 0, [1, 0, 0, 1]);
        builder.setNodeValue("dyn", 0, 1);
        const dyn = builder.addExtensionTable("temporal:node:dyn", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "end", dtype: "f64" },
            { name: "value", dtype: "f64" },
        ]);
        builder.addExtensionRow(dyn, [0, 0, 1, 2]);
        const other = builder.addExtensionTable("custom:thing", [{ name: "v", dtype: "i32" }]);
        builder.addExtensionRow(other, [1]);
        const snapshot = builder.freeze();
        const codes = noteCodes(snapshot);
        for (const code of [
            LOSS.LIST,
            LOSS.JSON,
            LOSS.DTYPE,
            LOSS.COMPONENTS,
            LOSS.DEFAULT,
            LOSS.OPTIONS,
            LOSS.TEMPORAL,
            LOSS.SPELLS,
            LOSS.OPEN_INTERVAL,
            LOSS.VIZ,
            LOSS.DYNAMIC_VALUES,
            LOSS.EXTENSION_TABLE,
        ]) {
            expect(codes).toContain(code);
        }
        const dtypeNotes = dotExporter.check(snapshot).filter((n) => n.code === LOSS.DTYPE);
        expect(dtypeNotes.map((n) => n.column).sort()).toEqual(["cat", "single"]);
        for (const absent of [
            LOSS.MULTI_EDGES,
            LOSS.SELF_LOOPS,
            LOSS.HIERARCHY,
            LOSS.POSITIONS,
            LOSS.GRAPH_ATTRIBUTES,
            LOSS.ID_CHARSET,
            LOSS.EDGE_IDS_DROPPED,
            LOSS.EDGE_IDS_GENERATED,
        ]) {
            expect(codes).not.toContain(absent);
        }
    });

    it("never reports parallel edges, self-loops, hierarchy, positions or graph attributes", async () => {
        const snapshot = await imported('digraph { a -> b; a -> b; a -> a; subgraph cluster_0 { a [pos="1,2"] } x=1 }');
        expect(snapshot.flags.multigraph).toBe(true);
        expect(dotExporter.check(snapshot)).toEqual([]);
    });

    it("writes flattened components and skips list / json / viz / temporal columns", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.declareNodeColumn({ name: "vec", dtype: "i32", components: 2 });
        builder.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        builder.declareNodeColumn({ name: "blob", dtype: "json" });
        builder.declareNodeColumn({ name: "rgba", dtype: "f32", components: 4, role: "color" });
        builder.declareNodeColumn({ name: "since", dtype: "f64", role: "start" });
        builder.setNodeValue("vec", 0, [1, 2]);
        builder.setNodeValue("tags", 0, ["x"]);
        builder.setNodeValue("blob", 0, { k: 1 });
        builder.setNodeValue("rgba", 0, [1, 0, 0, 1]);
        builder.setNodeValue("since", 0, 5);
        const text = await dotExporter.exportToString(builder.freeze());
        expect(lines(text)[1]).toBe('a [vec="1,2"];');
    });

    it("reports and refuses ids, names and values ending in a backslash", async () => {
        const bad = new GraphBuilder({ directed: true, weightDtype: "f64" });
        bad.addNode("ok");
        bad.addNode("trailing\\");
        expect(noteCodes(bad.freeze())).toEqual([DOT_LOSS.TRAILING_BACKSLASH]);
        await expect(dotExporter.exportToString(bad.freeze())).rejects.toMatchObject({
            code: "E_INVALID_ID",
            details: { reason: "trailing backslash", kind: "id" },
        });
        const badValue = new GraphBuilder({ directed: true, weightDtype: "f64" });
        badValue.addNode("a");
        badValue.setNodeValue("v", 0, "x\\");
        await expect(dotExporter.exportToString(badValue.freeze())).rejects.toMatchObject({ code: "E_COLUMN_TYPE" });
        const badName = new GraphBuilder({ directed: true, weightDtype: "f64" });
        badName.addNode("a");
        badName.setNodeValue("n\\", 0, 1);
        expect(dotExporter.check(badName.freeze())[0]).toMatchObject({
            code: DOT_LOSS.TRAILING_BACKSLASH,
            count: 1,
        });
        const badGraphName = new GraphBuilder({ directed: true, weightDtype: "f64" });
        badGraphName.setMeta({ name: "g\\" });
        await expect(dotExporter.exportToString(badGraphName.freeze())).rejects.toMatchObject({
            details: { kind: "graph name" },
        });
        const badPort = new GraphBuilder({ directed: true, weightDtype: "f64" });
        badPort.declareEdgeColumn({ name: "p", dtype: "string", role: "sourcePort" });
        badPort.setEdgeValue("p", badPort.addEdge("a", "b"), "q\\");
        // the port column reads back under the importer's fixed name graphty.sourcePort
        expect(noteCodes(badPort.freeze())).toEqual([LOSS.COLUMN_NAME_CHANGED, DOT_LOSS.TRAILING_BACKSLASH]);
    });

    it("writes texts that contain backslashes elsewhere and reads them back unchanged", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode('a\\b"c');
        builder.setNodeValue("v", 0, "x\\\\y\\n");
        const snapshot = builder.freeze();
        expectSameSnapshot(snapshot, (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot);
        // a backslash before a quote is not writable: Graphviz reads the written \\" as a pair and the closing quote
        const unwritable = new GraphBuilder({ directed: true, weightDtype: "f64" });
        unwritable.addNode('a\\"b');
        expect(noteCodes(unwritable.freeze())).toEqual([DOT_LOSS.TRAILING_BACKSLASH]);
    });

    it("reports non-finite numbers and text that would read back typed", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        builder.addNode("b");
        builder.declareNodeColumn({ name: "f", dtype: "f64" });
        builder.setNodeValue("f", 0, Infinity);
        builder.setNodeValue("f", 1, 1);
        builder.setNodeValue("s", 0, "123");
        builder.setNodeValue("s", 1, "abc");
        builder.setNodeValue("label", 0, "42");
        const snapshot = builder.freeze();
        const notes = dotExporter.check(snapshot);
        expect(notes.map((n) => n.code)).toEqual([
            DOT_LOSS.NON_FINITE,
            DOT_LOSS.TEXT_INFERRED,
            DOT_LOSS.ROLE_ASSUMED,
            DOT_LOSS.TEXT_INFERRED,
        ]);
        expect(notes[0]).toMatchObject({ column: "f", count: 1 });
        expect(notes[1]).toMatchObject({ column: "s", count: 1 });
        // a plain "label" column (no role) reads back with the label role, and its text is noted too
        expect(notes[2]).toMatchObject({ column: "label", count: 1 });
        expect(notes[3]).toMatchObject({ column: "label", count: 1 });
        const back = (await roundTrip(snapshot, dotExporter, dotImporter)).snapshot;
        expect(back.nodes.get("f")?.dtype).toBe("string");
        expect(back.nodes.get("f")?.value(0)).toBe("Infinity");
        expect(back.nodes.get("s")?.dtype).toBe("string");
        expect(back.nodes.get("label")?.value(0)).toBe("42");
    });

    it("reports and skips plain columns that clash with weight, key and pos", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b", 2);
        builder.declareEdgeColumn({ name: "eid", dtype: "string", role: "id" });
        builder.setEdgeValue("eid", e, "x");
        builder.setEdgeValue("key", e, "k");
        builder.setEdgeValue("weight", e, 9);
        builder.declareNodeColumn({ name: "xy", dtype: "f32", components: 3, role: "position" });
        builder.setNodeValue("xy", 0, [1, 2, 0]);
        builder.setNodeValue("pos", 0, "text");
        const snapshot = builder.freeze();
        const notes = dotExporter.check(snapshot).filter((n) => n.code === DOT_LOSS.ATTRIBUTE_CLASH);
        expect(notes.map((n) => n.column).sort()).toEqual(["key", "pos", "weight"]);
        const text = await dotExporter.exportToString(snapshot);
        expect(lines(text)).toEqual(["digraph {", 'a [pos="1,2,0"];', "b;", "a -> b [key=x, weight=2];", "}"]);
    });

    it("reports a multi-parent column, odd position shapes and non-canonical numeric ids", () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode(1.5);
        builder.addNode(2);
        builder.declareNodeColumn({ name: "pids", dtype: "list", itemDtype: "u32", role: "parents", refersTo: "node" });
        builder.setNodeValue("pids", 0, [1]);
        builder.declareNodeColumn({ name: "p1", dtype: "f32", components: 1, role: "position" });
        builder.setNodeValue("p1", 0, 3);
        builder.declareEdgeColumn({ name: "epos", dtype: "f32", components: 3, role: "position" });
        builder.setEdgeValue("epos", builder.addEdge(1.5, 2), [1, 2, 3]);
        const snapshot = builder.freeze();
        const notes = dotExporter.check(snapshot);
        const codes = notes.map((n) => n.code);
        expect(codes).toContain(DOT_LOSS.PARENTS_DROPPED);
        expect(codes.filter((c) => c === DOT_LOSS.POSITION_SHAPE)).toHaveLength(2);
        expect(codes).toContain(DOT_LOSS.ID_TEXT_TYPE);
        expect(notes.find((n) => n.code === DOT_LOSS.ID_TEXT_TYPE)?.count).toBe(1);
    });

    it("writes a role-weight column of another dtype, a numeric edge id and f32 cells", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "w", dtype: "i32", role: "weight" });
        builder.declareEdgeColumn({ name: "eid", dtype: "f64", role: "id" });
        builder.declareNodeColumn({ name: "s", dtype: "f32" });
        builder.declareNodeColumn({ name: "where", dtype: "string", role: "position" });
        const e = builder.addEdge("a", "b");
        builder.setEdgeValue("w", e, 3);
        builder.setEdgeValue("eid", e, 2.5);
        builder.setNodeValue("s", 0, 0.1);
        builder.setNodeValue("where", 0, "here");
        const text = await dotExporter.exportToString(builder.freeze());
        expect(lines(text)).toEqual(["digraph {", "a [s=0.1];", "b;", "a -> b [key=2.5, weight=3];", "}"]);
    });

    it("rejects an unknown common export option", async () => {
        const snapshot = await imported("digraph {}");
        expect(() => dotExporter.check(snapshot, { sanitizeIds: "always" as never })).toThrow();
        await expect(dotExporter.exportToString(snapshot, { onMixedDirection: "flip" as never })).rejects.toMatchObject(
            { code: "E_UNSUPPORTED" },
        );
    });
});

describe("dot exporter: corpus round trips", () => {
    for (const file of corpusFiles("dot")) {
        it(`round-trips ${file.path} on ids, topology, orientation, weights and attributes`, async () => {
            const snapshot = await imported(readCorpusText("dot", file.path));
            const result = await roundTrip(snapshot, dotExporter, dotImporter);
            expect(result.notes).toEqual([]);
            expect(result.report.errorCount).toBe(0);
            expect(result.report.warningCount).toBe(0);
            expectSameSnapshot(snapshot, result.snapshot, { allowExtraColumns: false, originType: true });
            const again = await roundTrip(result.snapshot, dotExporter, dotImporter);
            expect(again.text).toBe(result.text);
        });
    }

    it("round-trips a synthetic graph with every feature the format keeps", async () => {
        const text = `strict digraph "the graph" {
            rankdir=LR; graph [label="G", n=3, f=2.5, b=true];
            node [shape=box];
            a [label="A node", pos="1,2!", n=1, f=0.5, s=text, t=true];
            subgraph cluster_x { color=red; b [label=<<i>b</i>>]; subgraph cluster_y { c } }
            a:p -> b:q:n [key=k1, weight=2, label="e1", note="x"];
            a -> b [key=k2, weight=0.25];
            b -> c;
            c -> c [weight=-1];
            d -> a [weight="1e-9"];
        }`;
        const snapshot = await imported(text);
        const result = await roundTrip(snapshot, dotExporter, dotImporter);
        expect(result.notes).toEqual([]);
        expect(result.report.issues).toEqual([]);
        expectSameSnapshot(snapshot, result.snapshot, { allowExtraColumns: false, originType: true });
        expect(result.snapshot.meta.extra).toEqual({ dot: { strict: true } });
        expect(result.snapshot.meta.name).toBe("the graph");
    });
});
