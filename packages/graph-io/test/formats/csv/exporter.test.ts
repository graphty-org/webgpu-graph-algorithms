import { fromRecords, GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { LOSS } from "../../../src/common/export.js";
import { decodeChunks } from "../../../src/common/writer.js";
import { CSV_CAPABILITIES, CSV_LOSS, csvExporter } from "../../../src/formats/csv/exporter.js";
import { csvImporter } from "../../../src/formats/csv/importer.js";
import { type CommonExportOptions, type LossNote } from "../../../src/types.js";
import { corpusFiles, readCorpusBytes, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

type ExportOptions = Parameters<typeof csvExporter.check>[1];
type ImportOptions = Parameters<typeof csvImporter.import>[2];

async function importCsv(text: string, options?: ImportOptions, directed = true): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed, weightDtype: "f64" });
    await csvImporter.import(text, builder, options);
    return builder.freeze();
}

/** Export the edge table and the node table, import them together, and compare. */
async function roundTripWithNodes(
    snapshot: GraphSnapshot,
    exportOptions?: ExportOptions,
    importOptions?: ImportOptions,
): Promise<{ edges: string; nodes: string; notes: readonly LossNote[]; result: GraphSnapshot }> {
    const notes = csvExporter.check(snapshot, exportOptions);
    const edges = await csvExporter.exportToString(snapshot, exportOptions);
    const nodes = await csvExporter.exportToString(snapshot, { ...exportOptions, table: "nodes" });
    const result = await importCsv(edges, { ...importOptions, nodes }, snapshot.directed);
    return { edges, nodes, notes, result };
}

function codesOf(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

describe("csvExporter: capabilities and options", () => {
    it("declares the format and what it keeps", () => {
        expect(csvExporter.format).toBe("csv");
        expect(csvExporter.capabilities).toBe(CSV_CAPABILITIES);
        expect(CSV_CAPABILITIES).toMatchObject({
            mixedDirection: true,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "optional",
            idCharset: "any",
            lists: false,
            json: false,
            defaults: false,
            temporal: "none",
            graphAttributes: false,
            positions: false,
            viz: false,
        });
        expect([...CSV_CAPABILITIES.dtypes]).toEqual(["bool", "i32", "f64", "string", "dict"]);
    });

    it("rejects options outside their sets", () => {
        const s = fromRecords({ directed: true, edges: [{ source: "a", target: "b" }] }).snapshot;
        const bad = [
            { dialect: "excel" },
            { table: "graph" },
            { delimiter: ",," },
            { delimiter: '"' },
            { newline: "\r" },
            { header: "yes" },
            { sanitizeIds: "rename" },
        ] as unknown as CommonExportOptions[];
        for (const options of bad) {
            expect(() => csvExporter.check(s, options)).toThrow(/E_UNSUPPORTED|not one of|expected/);
        }
    });
});

describe("csvExporter: the written text", () => {
    it("writes the Gephi table with RFC 4180 quoting", async () => {
        const s = await importCsv(
            'Source,Target,Type,Id,Label,Weight,note\n"a,1",b,Directed,e1,"say ""hi""",2.5," x "\nb,c,Undirected,e2,,,line\n',
        );
        const text = await csvExporter.exportToString(s);
        expect(text).toBe(
            [
                "Source,Target,Type,Id,Label,Weight,note",
                '"a,1",b,Directed,e1,"say ""hi""",2.5," x "',
                "b,c,Undirected,e2,,,line",
                "",
            ].join("\n"),
        );
    });

    it("writes the generic dialect without a direction column, and honours delimiter, newline and header", async () => {
        const s = await importCsv("source,target,weight\na,b,1\n");
        expect(await csvExporter.exportToString(s, { dialect: "generic" })).toBe("source,target,weight\na,b,1\n");
        expect(await csvExporter.exportToString(s, { dialect: "generic", delimiter: "\t", newline: "\r\n" })).toBe(
            "source\ttarget\tweight\r\na\tb\t1\r\n",
        );
        expect(await csvExporter.exportToString(s, { header: false })).toBe("a,b,Directed,1\n");
    });

    it("quotes a cell that contains the delimiter in use", async () => {
        const s = fromRecords({ directed: true, edges: [{ source: "a;b", target: "c" }] }).snapshot;
        expect(await csvExporter.exportToString(s, { dialect: "generic", delimiter: ";" })).toBe(
            'source;target\n"a;b";c\n',
        );
    });

    it("export() yields the same bytes as exportToString()", async () => {
        const s = await importCsv(readCorpusText("csv", "got-edges.csv"));
        const text = await csvExporter.exportToString(s);
        expect(await decodeChunks(csvExporter.export(s))).toBe(text);
        expect(text.split("\n")).toHaveLength(354);
    });

    it("writes explicit weights only and no Weight column for an unweighted graph", async () => {
        const s = await importCsv("source,target,weight\na,b,2\nb,c,\n");
        expect(await csvExporter.exportToString(s)).toBe("Source,Target,Type,Weight\na,b,Directed,2\nb,c,Directed,\n");
        const unweighted = await importCsv("source,target\na,b\n");
        expect(await csvExporter.exportToString(unweighted)).toBe("Source,Target,Type\na,b,Directed\n");
    });

    it("writes f32 arc weights as the shortest round-tripping text when there is no shadow", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f32" });
        builder.addEdge("a", "b", 0.1);
        const text = await csvExporter.exportToString(builder.freeze(), { dialect: "generic" });
        expect(text).toBe("source,target,weight\na,b,0.1\n");
    });

    it("writes every column dtype as text the importer reads back", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b");
        builder.declareEdgeColumn({ name: "b", dtype: "bool" });
        builder.declareEdgeColumn({ name: "i", dtype: "i32" });
        builder.declareEdgeColumn({ name: "f", dtype: "f64" });
        builder.declareEdgeColumn({ name: "g", dtype: "f32" });
        builder.declareEdgeColumn({ name: "s", dtype: "string" });
        builder.declareEdgeColumn({ name: "d", dtype: "dict" });
        builder.declareEdgeColumn({ name: "l", dtype: "list", itemDtype: "i32" });
        builder.declareEdgeColumn({ name: "j", dtype: "json" });
        builder.declareEdgeColumn({ name: "v", dtype: "f64", components: 3 });
        builder.declareEdgeColumn({ name: "n", dtype: "f64" });
        builder.setEdgeValue("b", e, true);
        builder.setEdgeValue("i", e, -7);
        builder.setEdgeValue("f", e, 2);
        builder.setEdgeValue("g", e, 0.5);
        builder.setEdgeValue("s", e, "text");
        builder.setEdgeValue("d", e, "cat");
        builder.setEdgeValue("l", e, [1, 2, 3]);
        builder.setEdgeValue("j", e, { k: [1, "x"] });
        builder.setEdgeValue("v", e, [1, 2.5, -0]);
        builder.setEdgeValue("n", e, -0);
        const s = builder.freeze();
        const text = await csvExporter.exportToString(s, { dialect: "generic" });
        expect(text).toBe(
            'source,target,b,i,f,g,s,d,l,j,v,n\na,b,true,-7,2.0,0.5,text,cat,1;2;3,"{""k"":[1,""x""]}",1.0;2.5;-0.0,-0.0\n',
        );
        const notes = csvExporter.check(s, { dialect: "generic" });
        expect(notes.map((n) => [n.code, n.column])).toEqual([
            [LOSS.DTYPE, "g"],
            [LOSS.LIST, "l"],
            [LOSS.JSON, "j"],
            [LOSS.COMPONENTS, "v"],
            // one row: a dict of one value reads back as a string column
            [CSV_LOSS.STORAGE_CLASS_CHANGED, "d"],
        ]);
    });

    it("writes a 1e21 f64 with a decimal point and non-finite values as their JS spellings", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b", Infinity);
        builder.declareEdgeColumn({ name: "big", dtype: "f64" });
        builder.declareEdgeColumn({ name: "nan", dtype: "f64" });
        builder.setEdgeValue("big", e, 1e21);
        builder.setEdgeValue("nan", e, NaN);
        const s = builder.freeze();
        expect(await csvExporter.exportToString(s, { dialect: "generic" })).toBe(
            "source,target,weight,big,nan\na,b,Infinity,1e+21,NaN\n",
        );
        const notes = csvExporter.check(s);
        expect(notes.map((n) => [n.code, n.column, n.count])).toEqual([[CSV_LOSS.NONFINITE, "nan", 1]]);
        const back = await importCsv(await csvExporter.exportToString(s));
        expect(back.edges.get("big")?.dtype).toBe("f64");
        expect(back.edges.get("big")?.value(0)).toBe(1e21);
        expect(back.edges.get("nan")?.dtype).toBe("string");
        expect(back.edgeList().weights?.[0]).toBe(Infinity);
    });

    it("folds expanded undirected pairs into one Undirected row and writes mutual pairs as two Directed rows", async () => {
        const s = await importCsv(
            "Source,Target,Type,Label\na,b,Directed,x\nb,c,Undirected,y\nc,c,Undirected,loop\nd,e,Mutual,m\n",
        );
        expect(s.edgeCount).toBe(6);
        const text = await csvExporter.exportToString(s);
        expect(text.split("\n")).toEqual([
            "Source,Target,Type,Label",
            "a,b,Directed,x",
            "b,c,Undirected,y",
            "c,c,Undirected,loop",
            "d,e,Directed,m",
            "e,d,Directed,",
            "",
        ]);
        const notes = csvExporter.check(s);
        expect(notes.map((n) => [n.code, n.count])).toEqual([[CSV_LOSS.MUTUAL_EXPANDED, 1]]);
    });

    it("writes the node table with the id, the label and the attributes", async () => {
        const s = await importCsv("source,target\nb,a\n", { nodes: "id,label,age\na,Alice,30\nb,Bob,\n" });
        expect(await csvExporter.exportToString(s, { table: "nodes" })).toBe("Id,label,age\na,Alice,30\nb,Bob,\n");
        expect(await csvExporter.exportToString(s, { table: "nodes", dialect: "generic", header: false })).toBe(
            "a,Alice,30\nb,Bob,\n",
        );
    });

    it("writes an empty graph as a header only", async () => {
        const s = new GraphBuilder({ directed: false }).freeze();
        expect(await csvExporter.exportToString(s)).toBe("Source,Target,Type\n");
        expect(await csvExporter.exportToString(s, { table: "nodes" })).toBe("Id\n");
        expect(csvExporter.check(s)).toEqual([]);
    });
});

describe("csvExporter: check() loss notes", () => {
    it("is empty for a snapshot the format holds exactly", async () => {
        const s = await importCsv("Source,Target,Type,Id,Label,Weight,n\na,b,Undirected,e1,L,2,1\n");
        expect(csvExporter.check(s)).toEqual([]);
        expect(csvExporter.check(s, { table: "nodes" })).toEqual([]);
    });

    it("reports id text collisions as fatal and refuses to export", () => {
        const s = fromRecords({
            directed: true,
            nodes: [{ id: 1 }, { id: "1" }],
            edges: [{ source: 1, target: "1" }],
        }).snapshot;
        const notes = csvExporter.check(s);
        expect(notes.map((n) => [n.code, n.count])).toEqual([
            [CSV_LOSS.ID_TEXT_COLLISION, 1],
            [CSV_LOSS.ID_TEXT_TYPE, 1],
        ]);
        expect(() => csvExporter.export(s)).toThrow(/E_INVALID_ID|share their text/);
        expect(() => csvExporter.exportToString(s)).toThrow(/share their text/);
    });

    it("reports ids whose type changes under the canonical re-read", () => {
        const s = fromRecords({
            directed: true,
            nodes: [{ id: "7" }, { id: 1.5 }, { id: "x" }, { id: 2 }],
            edges: [{ source: "7", target: 1.5 }],
        }).snapshot;
        const notes = csvExporter.check(s);
        // "x" and 2 have no edge: the edge table cannot carry them (design 8.5: check() says so)
        expect(notes.map((n) => [n.code, n.count])).toEqual([
            [CSV_LOSS.ID_TEXT_TYPE, 2],
            [CSV_LOSS.ISOLATED_NODES, 2],
        ]);
    });

    it("reports the direction the generic dialect drops", async () => {
        const undirected = await importCsv("source,target\na,b\nb,c\n", { defaultDirected: false });
        expect(csvExporter.check(undirected, { dialect: "generic" }).map((n) => [n.code, n.count])).toEqual([
            [CSV_LOSS.DIRECTION_DROPPED, 2],
        ]);
        expect(csvExporter.check(undirected, { dialect: "gephi" })).toEqual([]);
        const mixed = await importCsv("Source,Target,Type\na,b,Directed\nb,c,Undirected\nc,d,Undirected\n");
        expect(csvExporter.check(mixed, { dialect: "generic" }).map((n) => [n.code, n.count])).toEqual([
            [CSV_LOSS.DIRECTION_DROPPED, 2],
        ]);
        expect(await csvExporter.exportToString(mixed, { dialect: "generic" })).toBe("source,target\na,b\nb,c\nc,d\n");
    });

    it("reports attribute names reserved by the headers and does not write them", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b");
        builder.setEdgeValue("source", e, "s");
        builder.setEdgeValue("TYPE", e, "t");
        builder.setEdgeValue("id", e, "e1");
        builder.setEdgeValue("other", e, "o");
        builder.setNodeValue("ID", 0, 5);
        const s = builder.freeze();
        const edgeNotes = csvExporter.check(s);
        expect(edgeNotes.map((n) => [n.code, n.column])).toEqual([
            [CSV_LOSS.RESERVED_NAME, "source"],
            [CSV_LOSS.RESERVED_NAME, "TYPE"],
            [CSV_LOSS.RESERVED_NAME, "id"],
        ]);
        expect(await csvExporter.exportToString(s)).toBe("Source,Target,Type,other\na,b,Directed,o\n");
        // the generic dialect has no Type header, so TYPE is written there
        expect(await csvExporter.exportToString(s, { dialect: "generic" })).toBe("source,target,TYPE,other\na,b,t,o\n");
        const nodeNotes = csvExporter.check(s, { table: "nodes" });
        expect(nodeNotes.map((n) => [n.code, n.column])).toEqual([[CSV_LOSS.RESERVED_NAME, "ID"]]);
        expect(await csvExporter.exportToString(s, { table: "nodes" })).toBe("Id\na\nb\n");
    });

    it("reports a label-named column the importer gives the role and a second label it drops", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addEdge("a", "b");
        builder.setNodeValue("label", 0, "A");
        const s = builder.freeze();
        expect(csvExporter.check(s, { table: "nodes" }).map((n) => n.code)).toEqual([CSV_LOSS.ROLE_ASSUMED]);
        const withRole = new GraphBuilder({ directed: true, weightDtype: "f64" });
        withRole.addEdge("a", "b");
        withRole.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
        withRole.setNodeValue("name", 0, "A");
        withRole.setNodeValue("Label", 0, "B");
        const t = withRole.freeze();
        expect(csvExporter.check(t, { table: "nodes" }).map((n) => [n.code, n.column])).toEqual([
            [CSV_LOSS.RESERVED_NAME, "Label"],
            [CSV_LOSS.ROLE_NAME, "name"],
        ]);
        expect(await csvExporter.exportToString(t, { table: "nodes" })).toBe("Id,name\na,A\nb,\n");
    });

    it("reports role columns that are not text", () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b");
        builder.declareEdgeColumn({ name: "id", dtype: "u32", role: "id", unique: true });
        builder.setEdgeValue("id", e, 4);
        builder.declareNodeColumn({ name: "label", dtype: "i32", role: "label" });
        builder.setNodeValue("label", 0, 9);
        const s = builder.freeze();
        expect(csvExporter.check(s).map((n) => [n.code, n.column])).toEqual([
            [CSV_LOSS.TEXT_ROLE, "id"],
            [CSV_LOSS.NODE_TABLE, null],
        ]);
        expect(csvExporter.check(s, { table: "nodes" }).map((n) => [n.code, n.column])).toEqual([
            [CSV_LOSS.TEXT_ROLE, "label"],
        ]);
    });

    it("reports text that reads back typed and dict / string cardinality changes (a set empty string is the quoted empty cell and reads back exactly)", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        for (let i = 0; i < 10; i++) {
            const e = builder.addEdge(`n${i}`, `n${i + 1}`);
            builder.setEdgeValue("empty", e, i === 0 ? "" : "x");
            builder.setEdgeValue("numeric", e, String(i));
            builder.setEdgeValue("lowcard", e, i % 2 === 0 ? "a" : "b");
        }
        builder.declareEdgeColumn({ name: "wide", dtype: "dict" });
        for (let i = 0; i < 10; i++) {
            builder.setEdgeValue("wide", i, `v${i}`);
        }
        const s = builder.freeze();
        expect(csvExporter.check(s).map((n) => [n.code, n.column, n.count])).toEqual([
            [CSV_LOSS.STORAGE_CLASS_CHANGED, "empty", null],
            [CSV_LOSS.TEXT_INFERRED, "numeric", 10],
            [CSV_LOSS.STORAGE_CLASS_CHANGED, "lowcard", null],
            [CSV_LOSS.STORAGE_CLASS_CHANGED, "wide", null],
        ]);
        const text = await csvExporter.exportToString(s);
        expect(text).toContain('n0,n1,Directed,"",0,a,v0\n');
        const back = (await roundTrip(s, csvExporter, csvImporter)).snapshot;
        expect(back.edges.value("empty", 0)).toBe("");
    });

    it("reports what the format has no place for through the shared checks", () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const e = builder.addEdge("a", "b");
        builder.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, role: "position" });
        builder.declareNodeColumn({ name: "color", dtype: "u8", components: 4, role: "color" });
        builder.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        builder.declareEdgeColumn({ name: "start", dtype: "f64", role: "start" });
        builder.declareEdgeColumn({ name: "kind", dtype: "string", default: "x", options: ["x", "y"] });
        builder.setEdgeValue("start", e, 1);
        builder.setGraphValue("title", "t");
        const s = builder.freeze();
        const notes = csvExporter.check(s);
        expect(codesOf(notes)).toEqual([
            LOSS.POSITIONS,
            LOSS.VIZ,
            LOSS.HIERARCHY,
            LOSS.TEMPORAL,
            LOSS.DEFAULT,
            LOSS.OPTIONS,
            LOSS.GRAPH_ATTRIBUTES,
        ]);
    });

    it("reports node columns an edges export leaves out", async () => {
        const s = await importCsv(readCorpusText("csv", "got-edges.csv"), {
            nodes: readCorpusText("csv", "got-nodes.csv"),
        });
        expect(csvExporter.check(s).map((n) => [n.code, n.count])).toEqual([[CSV_LOSS.NODE_TABLE, 1]]);
        expect(csvExporter.check(s, { table: "nodes" })).toEqual([]);
    });
});

describe("csvExporter: round trips", () => {
    for (const entry of corpusFiles("csv")) {
        it(`${entry.path}: export then import is the same snapshot with no loss notes`, async () => {
            const original = await importCsv(new TextDecoder().decode(readCorpusBytes("csv", entry.path)));
            const { notes, snapshot } = await roundTrip(original, csvExporter, csvImporter);
            expect(notes).toEqual([]);
            expectSameSnapshot(original, snapshot, { allowExtraColumns: false });
        });
    }

    it("got-nodes + got-edges round-trip through both tables", async () => {
        const original = await importCsv(readCorpusText("csv", "got-edges.csv"), {
            nodes: readCorpusText("csv", "got-nodes.csv"),
        });
        const { result, notes, nodes } = await roundTripWithNodes(original);
        expect(codesOf(notes)).toEqual([CSV_LOSS.NODE_TABLE]);
        expect(nodes.split("\n")[0]).toBe("Id,Label");
        expectSameSnapshot(original, result, { allowExtraColumns: false });
    });

    it("gephi-format.csv survives a double round trip", async () => {
        const original = await importCsv(readCorpusText("csv", "gephi-format.csv"));
        const once = await roundTrip(original, csvExporter, csvImporter);
        const twice = await roundTrip(once.snapshot, csvExporter, csvImporter);
        expect(twice.text).toBe(once.text);
        expectSameSnapshot(original, twice.snapshot, { allowExtraColumns: false });
    });

    it("keeps 0.1 and 16777217 weights and a defaulted weight explicit-status", async () => {
        const original = await importCsv("source,target,weight\na,b,0.1\nb,c,16777217\nc,d,\n");
        const { snapshot, text } = await roundTrip(original, csvExporter, csvImporter);
        expect(text).toContain("c,d,Directed,\n");
        expectSameSnapshot(original, snapshot);
        expect(snapshot.edges.byRole("weight")?.isSet(2)).toBe(false);
    });

    it("keeps 01, 1 and 1.0 distinct ids under the canonical rule", async () => {
        const original = await importCsv("source,target\n01,1\n1.0,1\n");
        const { snapshot } = await roundTrip(original, csvExporter, csvImporter);
        expect(snapshot.ids.toArray()).toEqual(["01", 1, "1.0"]);
        expectSameSnapshot(original, snapshot);
    });

    it("keeps an undirected graph undirected and a mixed graph paired", async () => {
        const undirected = await importCsv("source,target\na,b\nb,c\n", { defaultDirected: false });
        const back = await roundTrip(undirected, csvExporter, csvImporter);
        expect(back.snapshot.directed).toBe(false);
        expectSameSnapshot(undirected, back.snapshot, { allowExtraColumns: false });
        const mixed = await importCsv(
            "Source,Target,Type,Weight\na,b,Directed,1\nb,c,Undirected,2\nc,c,Undirected,3\n",
        );
        const paired = await roundTrip(mixed, csvExporter, csvImporter);
        expect(paired.notes).toEqual([]);
        expectSameSnapshot(mixed, paired.snapshot, { allowExtraColumns: false });
        expect(paired.report.counts.expandedMixed).toBe(2);
    });

    it("keeps bool, i32, f64, string and dict columns with their dtypes", async () => {
        const rows = ["source,target,flag,count,ratio,name,kind"];
        for (let i = 0; i < 30; i++) {
            rows.push(`n${i},n${i + 1},${i % 2 === 0},${i},${i / 4},name${i},${i % 3 === 0 ? "x" : "y"}`);
        }
        const original = await importCsv(`${rows.join("\n")}\n`);
        expect(original.edges.get("kind")?.dtype).toBe("dict");
        expect(original.edges.get("ratio")?.dtype).toBe("f64");
        const { snapshot, notes } = await roundTrip(original, csvExporter, csvImporter);
        expect(notes).toEqual([]);
        expectSameSnapshot(original, snapshot, { allowExtraColumns: false });
    });

    it("keeps edge ids and labels with their roles", async () => {
        const original = await importCsv("Source,Target,Id,Label\na,b,e1,first\nb,c,e2,second\n");
        const { snapshot, notes } = await roundTrip(original, csvExporter, csvImporter);
        expect(notes).toEqual([]);
        expectSameSnapshot(original, snapshot, { allowExtraColumns: false });
        expect(snapshot.edges.byRole("id")?.meta.name).toBe("Id");
        expect(snapshot.edgeIndexOf("e2")).toBe(1);
    });

    it("keeps node attributes through the node table", async () => {
        const original = await importCsv("source,target\nb,a\na,c\n", {
            nodes: "id,label,age,city\na,Alice,30,Paris\nb,Bob,,Rome\nc,,5,\n",
        });
        const { result, notes } = await roundTripWithNodes(original);
        // the edge table alone would number the nodes b, a, c; the node table restores the order
        expect(codesOf(notes)).toEqual([CSV_LOSS.NODE_ORDER, CSV_LOSS.NODE_TABLE]);
        expectSameSnapshot(original, result, { allowExtraColumns: false });
    });

    it("round-trips ids and cells that need quoting", async () => {
        const text = 'source,target,note\n"a,b","c""d","x\ny"\n" e ",f,\n';
        const original = await importCsv(text);
        const { snapshot } = await roundTrip(original, csvExporter, csvImporter);
        expectSameSnapshot(original, snapshot, { allowExtraColumns: false });
        expect(snapshot.ids.toArray()).toEqual(["a,b", 'c"d', " e ", "f"]);
    });

    it("round-trips through the generic dialect when the importer is told the direction", async () => {
        const original = await importCsv("source,target,weight\na,b,1\nb,c,2\n", { defaultDirected: false });
        const back = await roundTrip(original, csvExporter, csvImporter, {
            exportOptions: { dialect: "generic" },
            importOptions: { defaultDirected: false },
        });
        expect(codesOf(back.notes)).toEqual([CSV_LOSS.DIRECTION_DROPPED]);
        expectSameSnapshot(original, back.snapshot, { allowExtraColumns: false });
    });

    it("re-imports the mutual expansion as two directed edges (the documented loss)", async () => {
        const original = await importCsv("Source,Target,Type\na,b,Mutual\n");
        const { snapshot, notes } = await roundTrip(original, csvExporter, csvImporter);
        expect(codesOf(notes)).toEqual([CSV_LOSS.MUTUAL_EXPANDED]);
        const diffs = compareSnapshots(original, snapshot);
        expect(describeDiffs(diffs)).toContain("graphty.pair");
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.edges.byRole("mutual")).toBeNull();
    });

    it("round-trips a headerless export through header: false", async () => {
        const original = await importCsv("1 2 0.5\n2 3 1\n");
        const text = await csvExporter.exportToString(original, { dialect: "generic", header: false, delimiter: " " });
        expect(text).toBe("1 2 0.5\n2 3 1\n");
        const back = await importCsv(text);
        expectSameSnapshot(original, back);
    });
});
