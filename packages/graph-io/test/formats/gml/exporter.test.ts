import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { decodeChunks } from "../../../src/common/writer.js";
import {
    DEFAULT_WEIGHT_KEY,
    gmlExporter,
    gmlRealText,
    GRAPHICS_CONFLICT_CODE,
    GRAPHICS_OVERRIDDEN_CODE,
    INVALID_KEY_CODE,
    JSON_ARRAY_CODE,
    KEY_MANGLED_CODE,
    NESTED_ARRAY_CODE,
    POSITION_COMPONENTS_CODE,
    RECORD_BOOLEAN_CODE,
    RECORD_NULL_CODE,
    RECORD_NUMBER_TYPE_CODE,
    RESERVED_KEY_CODE,
} from "../../../src/formats/gml/exporter.js";
import { gmlImporter } from "../../../src/formats/gml/importer.js";
import { LIST_START_MARKER, ORIGINAL_ID_KEY } from "../../../src/formats/gml/syntax.js";
import { type LossNote } from "../../../src/types.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

type ExportOptions = Parameters<typeof gmlExporter.check>[1];

async function importGml(text: string, options?: Parameters<typeof gmlImporter.import>[2]): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await gmlImporter.import(text, builder, options);
    return builder.freeze();
}

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

function lines(text: string): string[] {
    return text.split("\n").map((l) => l.trim());
}

async function expectExportError(snapshot: GraphSnapshot, options: ExportOptions, code: string): Promise<void> {
    await expect(gmlExporter.exportToString(snapshot, options)).rejects.toMatchObject({ code });
    const chunks = gmlExporter.export(snapshot, options);
    await expect(decodeChunks(chunks)).rejects.toMatchObject({ code });
}

async function exact(snapshot: GraphSnapshot, options?: ExportOptions): Promise<string> {
    const notes = gmlExporter.check(snapshot, options);
    expect(notes, describeNotes(notes)).toEqual([]);
    const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, { exportOptions: options });
    expect(rt.report.errorCount).toBe(0);
    expectSameSnapshot(snapshot, rt.snapshot, { originType: true, allowExtraColumns: false });
    const again = await roundTrip(rt.snapshot, gmlExporter, gmlImporter, { exportOptions: options });
    expectSameSnapshot(snapshot, again.snapshot, { originType: true, allowExtraColumns: false });
    expect(again.text).toBe(rt.text);
    return rt.text;
}

function describeNotes(notes: readonly LossNote[]): string {
    return notes.map((n) => `${n.code}: ${n.message}`).join("\n");
}

describe("gmlExporter: plugin shape", () => {
    it("declares the format and the fidelity matrix of note 07", () => {
        expect(gmlExporter.format).toBe("gml");
        expect(gmlExporter.capabilities).toEqual({
            mixedDirection: false,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "optional",
            idCharset: "integer",
            dtypes: ["i32", "f64", "string", "dict", "json"],
            components: false,
            lists: true,
            json: true,
            defaults: false,
            options: false,
            hierarchy: false,
            temporal: "none",
            graphAttributes: true,
            positions: true,
            viz: false,
        });
        expect(Object.isFrozen(gmlExporter)).toBe(true);
    });

    it("writes the same bytes through export() and exportToString()", async () => {
        const snapshot = await importGml(readCorpusText("gml", "karate.gml"));
        const text = await gmlExporter.exportToString(snapshot);
        expect(await decodeChunks(gmlExporter.export(snapshot))).toBe(text);
    });

    it("formats reals with a decimal point and the NetworkX non-finite spellings", () => {
        expect(gmlRealText(2)).toBe("2.0");
        expect(gmlRealText(0.1)).toBe("0.1");
        expect(gmlRealText(1e-7)).toBe("1.0e-7");
        expect(gmlRealText(1e21)).toBe("1.0e+21");
        expect(gmlRealText(Math.fround(0.1), "f32")).toBe("0.1");
        expect(gmlRealText(Infinity)).toBe("+INF");
        expect(gmlRealText(-Infinity)).toBe("-INF");
        expect(gmlRealText(NaN)).toBe("NAN");
    });
});

describe("gmlExporter: corpus round trips (design 16.5)", () => {
    for (const entry of corpusFiles("gml")) {
        it(`${entry.path} survives a double round trip with unchanged dtypes and origin types`, async () => {
            const snapshot = await importGml(readCorpusText("gml", entry.path));
            const text = await exact(snapshot);
            expect(text.startsWith(snapshot.meta.creator === null ? "graph [" : "Creator ")).toBe(true);
            expect(text).toContain("  directed 0\n");
        });
    }

    it("karate: writes the file's own conventions back", async () => {
        const snapshot = await importGml(readCorpusText("gml", "karate.gml"));
        const text = await gmlExporter.exportToString(snapshot);
        const out = lines(text);
        expect(out[0]).toBe('Creator "Mark Newman on Fri Jul 21 12:39:27 2006"');
        expect(out.slice(1, 6)).toEqual(["graph [", "directed 0", "node [", "id 1", "]"]);
        expect(text).toContain("edge [\n    source 2\n    target 1\n  ]");
        expect(text.endsWith("\n]\n")).toBe(true);
        expect(text).not.toContain("value");
    });

    it("polbooks and football: dict values quoted, int values bare", async () => {
        const polbooks = await gmlExporter.exportToString(await importGml(readCorpusText("gml", "polbooks.gml")));
        expect(polbooks).toContain('label "1000 Years for Revenge"\n    value "n"');
        const football = await gmlExporter.exportToString(await importGml(readCorpusText("gml", "football.gml")));
        expect(football).toContain('label "BrighamYoung"\n    value 7');
        expect(football).toContain('label "TexasA&#38;M"');
    });
});

describe("gmlExporter: values", () => {
    it("writes every dtype in its GML spelling and reads it back", async () => {
        const text = `Creator "c"
Version 1
graph [
  directed 1
  multigraph 1
  title "t"
  scale 2.5
  node [ id 1 i 1 r 2 s "a &quot; b &#233;" w 3000000000 label "L" ]
  node [ id 2 i -5 r 0.1 s "" w 1 ]
  edge [ source 1 target 2 value 3 ]
  edge [ source 1 target 2 value 3 key 1 ]
  edge [ source 2 target 2 label "loop" id 9 ]
]
`;
        const snapshot = await importGml(text);
        const out = await exact(snapshot);
        expect(out).toBe(`Creator "c"
Version 1
graph [
  directed 1
  multigraph 1
  title "t"
  scale 2.5
  node [
    id 1
    i 1
    r 2.0
    s "a &#34; b &#233;"
    w 3000000000
    label "L"
  ]
  node [
    id 2
    i -5
    r 0.1
    s ""
    w 1
  ]
  edge [
    source 1
    target 2
    value 3
  ]
  edge [
    source 1
    target 2
    value 3
    key 1
  ]
  edge [
    source 2
    target 2
    id 9
    label "loop"
  ]
]
`);
    });

    it("writes lists with the marker and empty-list conventions and reads them back", async () => {
        const text = `graph [ node [ id 1 t "a" t "b" ] node [ id 2 t "${LIST_START_MARKER}" t "c" ] node [ id 3 t "[]" ] node [ id 4 ] edge [ source 1 target 2 n 1 n 2 ] edge [ source 2 target 3 n 7 ] ]`;
        const out = await exact(await importGml(text));
        expect(out).toContain('t "a"\n    t "b"');
        expect(out).toContain(`t "${LIST_START_MARKER}"\n    t "c"`);
        expect(out).toContain('t "[]"');
        expect(out).toContain(`n "${LIST_START_MARKER}"\n    n 7`);
    });

    it("writes json records, nested records and arrays inside records", async () => {
        const text = `graph [ node [ id 1 meta [ w 30 fill "#fff" Line [ point [ x 1 y 2 ] point [ x 3 y 4 ] ] one "${LIST_START_MARKER}" one 5 none "[]" ] ] ]`;
        const snapshot = await importGml(text);
        const notes = gmlExporter.check(snapshot);
        expect(codes(notes)).toEqual([RECORD_NUMBER_TYPE_CODE]);
        expect(notes[0]).toMatchObject({ column: "meta", count: 1 });
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expectSameSnapshot(snapshot, rt.snapshot, { originType: true, allowExtraColumns: false });
        expect(rt.text).toContain(
            'meta [\n      w 30\n      fill "#fff"\n      Line [\n        point [\n          x 1\n          y 2\n        ]',
        );
        expect(rt.text).toContain(`one "${LIST_START_MARKER}"\n      one 5\n      none "[]"`);
    });

    it("writes the weight from the shadow column for explicit rows only and keeps its precision", async () => {
        const snapshot = await importGml(
            "graph [ edge [ source 1 target 2 value 0.1 ] edge [ source 2 target 3 ] edge [ source 3 target 1 value 16777217 ] ]",
        );
        const out = await exact(snapshot);
        expect(lines(out).filter((l) => l.startsWith("value"))).toEqual(["value 0.1", "value 16777217.0"]);
    });

    it("writes integer weights bare when the origin was int and reads them back as f32-exact arcs", async () => {
        const snapshot = await importGml(
            "graph [ edge [ source 1 target 2 value 3 ] edge [ source 2 target 3 value 4 ] ]",
        );
        const out = await exact(snapshot);
        expect(lines(out).filter((l) => l.startsWith("value"))).toEqual(["value 3", "value 4"]);
    });

    it("uses the weightKey option, the importer's weightFrom by default, and refuses a bad key", async () => {
        const snapshot = await importGml("graph [ edge [ source 1 target 2 weight 2 ] ]", { weightFrom: "weight" });
        expect(await gmlExporter.exportToString(snapshot)).toContain("weight 2");
        expect(await gmlExporter.exportToString(snapshot, { weightKey: "w" })).toContain("w 2");
        expect(() => gmlExporter.check(snapshot, { weightKey: "a.b" })).toThrow(
            expect.objectContaining({ code: "E_UNSUPPORTED" }),
        );
        const builder = new GraphBuilder({ directed: true });
        builder.addEdge(1, 2, 5);
        expect(await gmlExporter.exportToString(builder.freeze())).toContain(`${DEFAULT_WEIGHT_KEY} 5.0`);
    });

    it("writes non-finite reals with the NetworkX spellings and reads them back", async () => {
        const snapshot = await importGml("graph [ node [ id 1 a +INF b -INF c NAN ] ]");
        const out = await gmlExporter.exportToString(snapshot);
        expect(out).toContain("a +INF\n    b -INF\n    c NAN");
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.snapshot.nodes.value("a", 0)).toBe(Infinity);
        expect(rt.snapshot.nodes.value("b", 0)).toBe(-Infinity);
        expect(Number.isNaN(rt.snapshot.nodes.value("c", 0))).toBe(true);
    });

    it("writes columns from other sources by dtype: bool as 1 / 0, f32 and u32 with a note, dict as strings", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "flag", dtype: "bool" });
        builder.declareNodeColumn({ name: "f", dtype: "f32" });
        builder.declareNodeColumn({ name: "u", dtype: "u32" });
        builder.declareNodeColumn({ name: "k", dtype: "dict" });
        builder.declareNodeColumn({ name: "d", dtype: "f64" });
        const a = builder.addNode("a");
        builder.setNodeValue("flag", a, true);
        builder.setNodeValue("f", a, 0.1);
        builder.setNodeValue("u", a, 7);
        builder.setNodeValue("k", a, "cat");
        builder.setNodeValue("d", a, 2);
        builder.addEdge("a", "a");
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot, { sanitizeIds: "mangle" });
        // one row: a dict of one value reads back as a string column (the dictionary heuristic)
        expect(codes(notes)).toEqual([LOSS.ID_MANGLED, LOSS.DTYPE, LOSS.DTYPE, LOSS.DTYPE, LOSS.STORAGE_CLASS]);
        expect(notes.slice(1).map((n) => n.column)).toEqual(["flag", "f", "u", "k"]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, { exportOptions: { sanitizeIds: "mangle" } });
        expect(rt.text).toContain(
            `id 0\n    ${ORIGINAL_ID_KEY} "a"\n    flag 1\n    f 0.1\n    u 7\n    k "cat"\n    d 2.0`,
        );
        expect(rt.snapshot.ids.toArray()).toEqual(["a"]);
        expect(rt.snapshot.nodes.require("flag").dtype).toBe("i32");
        expect(rt.snapshot.nodes.require("f").dtype).toBe("f64");
        expect(rt.snapshot.nodes.require("d").dtype).toBe("f64");
        expect(rt.snapshot.nodes.require("d").meta.origin?.type).toBe("real");
    });

    it("flattens a multi-component column into repeated keys with a note", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "v", dtype: "f64", components: 2 });
        builder.setNodeValue("v", builder.addNode(1), [1, 2]);
        const snapshot = builder.freeze();
        expect(codes(gmlExporter.check(snapshot))).toEqual([LOSS.COMPONENTS]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.text).toContain("v 1.0\n    v 2.0");
        expect(rt.snapshot.nodes.require("v").dtype).toBe("list");
    });
});

describe("gmlExporter: graphics and positions", () => {
    it("round-trips positions through graphics x / y / z merged with the graphics record", async () => {
        const text =
            'graph [ node [ id 1 graphics [ x 1.5 y 2 z 3 w 10 fill "#f00" ] ] node [ id 2 graphics [ x 4 y 5 ] ] node [ id 3 graphics [ w 20 ] ] node [ id 4 ] ]';
        const snapshot = await importGml(text);
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_NUMBER_TYPE_CODE]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expectSameSnapshot(snapshot, rt.snapshot, { originType: true, allowExtraColumns: false });
        const again = await roundTrip(rt.snapshot, gmlExporter, gmlImporter);
        expectSameSnapshot(snapshot, again.snapshot, { originType: true, allowExtraColumns: false });
        const out = rt.text;
        expect(out).toContain(
            'graphics [\n      x 1.5\n      y 2.0\n      z 3.0\n      w 10\n      fill "#f00"\n    ]',
        );
        expect(out).toContain("graphics [\n      x 4.0\n      y 5.0\n      z 0.0\n    ]");
        expect(out).toContain("graphics [\n      w 20\n    ]");
    });

    it("writes a position column from another source with two or four components", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "pos", dtype: "f32", components: 2, role: "position" });
        builder.setNodeValue("pos", builder.addNode(1), [1, 2]);
        const two = builder.freeze();
        // graphics x / y read back as an f64 column named position: the f32 values and the name change
        expect(codes(gmlExporter.check(two))).toEqual([LOSS.DTYPE, LOSS.COLUMN_NAME_CHANGED]);
        expect(await gmlExporter.exportToString(two)).toContain("graphics [\n      x 1.0\n      y 2.0\n    ]");
        const wide = new GraphBuilder({ directed: false });
        wide.declareNodeColumn({ name: "pos", dtype: "f64", components: 4, role: "position" });
        wide.setNodeValue("pos", wide.addNode(1), [1, 2, 3, 4]);
        const four = wide.freeze();
        expect(codes(gmlExporter.check(four))).toEqual([POSITION_COMPONENTS_CODE, LOSS.COLUMN_NAME_CHANGED]);
        expect(await gmlExporter.exportToString(four)).toContain(
            "graphics [\n      x 1.0\n      y 2.0\n      z 3.0\n    ]",
        );
    });

    it("reports graphics records the position overrides and non-record graphics values", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "position", dtype: "f64", components: 3, role: "position" });
        builder.declareNodeColumn({ name: "graphics", dtype: "json" });
        const a = builder.addNode(1);
        builder.setNodeValue("position", a, [1, 2, 3]);
        builder.setNodeValue("graphics", a, { x: 9, w: 1 });
        const b = builder.addNode(2);
        builder.setNodeValue("position", b, [4, 5, 6]);
        builder.setNodeValue("graphics", b, 7);
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot);
        expect(codes(notes)).toEqual([RECORD_NUMBER_TYPE_CODE, GRAPHICS_OVERRIDDEN_CODE, GRAPHICS_CONFLICT_CODE]);
        await expectExportError(snapshot, undefined, "E_UNSUPPORTED");
        const only = new GraphBuilder({ directed: false });
        only.declareNodeColumn({ name: "position", dtype: "f64", components: 3, role: "position" });
        only.declareNodeColumn({ name: "graphics", dtype: "json" });
        const c = only.addNode(1);
        only.setNodeValue("position", c, [1, 2, 3]);
        only.setNodeValue("graphics", c, { x: 9, w: 1 });
        const text = await gmlExporter.exportToString(only.freeze());
        expect(text).toContain("graphics [\n      x 1.0\n      y 2.0\n      z 3.0\n      w 1\n    ]");
    });
});

describe("gmlExporter: ids", () => {
    it("refuses non-integer ids by default and mangles them with the original kept", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addEdge("Little Rock, AR", 1.5);
        builder.addEdge(1.5, 7);
        const snapshot = builder.freeze();
        expect(codes(gmlExporter.check(snapshot))).toEqual([LOSS.ID_CHARSET]);
        await expectExportError(snapshot, undefined, "E_INVALID_ID");
        const notes = gmlExporter.check(snapshot, { sanitizeIds: "mangle" });
        expect(notes).toEqual([expect.objectContaining({ code: LOSS.ID_MANGLED, count: 2 })]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, { exportOptions: { sanitizeIds: "mangle" } });
        expect(rt.text).toContain(`id 0\n    ${ORIGINAL_ID_KEY} "Little Rock, AR"`);
        expect(rt.text).toContain(`id 1\n    ${ORIGINAL_ID_KEY} 1.5`);
        expect(rt.text).toContain("id 7\n  ]");
        expectSameSnapshot(snapshot, rt.snapshot, { allowExtraColumns: false });
        const kept = await roundTrip(snapshot, gmlExporter, gmlImporter, {
            exportOptions: { sanitizeIds: "mangle" },
            importOptions: { restoreMangledIds: false },
        });
        expect(kept.snapshot.ids.toArray()).toEqual([0, 1, 7]);
        expect(kept.snapshot.nodes.value(ORIGINAL_ID_KEY, 0)).toBe("Little Rock, AR");
    });

    it("round-trips a label-keyed import through mangling", async () => {
        const text = 'graph [ node [ id 0 label "a" ] node [ id 1 label "b" ] edge [ source 0 target 1 value 2 ] ]';
        const snapshot = await importGml(text, { nodeIdFrom: "label" });
        expect(snapshot.ids.toArray()).toEqual(["a", "b"]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, { exportOptions: { sanitizeIds: "mangle" } });
        expectSameSnapshot(snapshot, rt.snapshot, { originType: true, allowExtraColumns: false });
        const relabelled = await roundTrip(snapshot, gmlExporter, gmlImporter, {
            exportOptions: { sanitizeIds: "mangle" },
            importOptions: { nodeIdFrom: "label", restoreMangledIds: false },
        });
        expect(relabelled.snapshot.ids.toArray()).toEqual(["a", "b"]);
    });

    it("writes a role-originalId column from another source as graphty_originalId", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "graphty.originalId", dtype: "string", role: "originalId" });
        builder.setNodeValue("graphty.originalId", builder.addNode(3), "x y");
        builder.addNode(4);
        const snapshot = builder.freeze();
        expect(gmlExporter.check(snapshot)).toEqual([]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.text).toContain(`id 3\n    ${ORIGINAL_ID_KEY} "x y"`);
        expect(rt.snapshot.ids.toArray()).toEqual(["x y", 4]);
    });

    it("writes large safe integers as ids", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.addEdge(9007199254740991, -3);
        const snapshot = builder.freeze();
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.snapshot.ids.toArray()).toEqual([9007199254740991, -3]);
        expect(rt.text).toContain("id 9007199254740991");
    });
});

describe("gmlExporter: mixed direction", () => {
    function mixed(): GraphSnapshot {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const resolver = new DirectionResolver(builder, new ImportReportBuilder("t", 10), "expand");
        resolver.setHeader(true);
        resolver.addEdge(1, 2, "directed", 5);
        resolver.addEdge(2, 3, "undirected");
        resolver.addEdge(3, 3, "undirected");
        return builder.freeze();
    }

    it("refuses a mixed snapshot by default", async () => {
        const snapshot = mixed();
        expect(codes(gmlExporter.check(snapshot))).toEqual([LOSS.MIXED_DIRECTION_ERROR]);
        await expectExportError(snapshot, undefined, "E_DIRECTED");
    });

    it("writes each expanded pair once as a directed edge under onMixedDirection directed (design 8.5: pairs fold back)", async () => {
        const snapshot = mixed();
        expect(codes(gmlExporter.check(snapshot, { onMixedDirection: "directed" }))).toEqual([LOSS.MIXED_DIRECTION]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, {
            exportOptions: { onMixedDirection: "directed" },
        });
        expect(rt.text).toContain("directed 1");
        expect(rt.snapshot.directed).toBe(true);
        // the same edge set every format without mixed direction writes under "directed"
        expect(rt.snapshot.edgeCount).toBe(3);
        expect(rt.snapshot.edges.byRole("pair")).toBeNull();
    });

    it("folds the pairs once under onMixedDirection undirected", async () => {
        const snapshot = mixed();
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, {
            exportOptions: { onMixedDirection: "undirected" },
        });
        expect(rt.text).toContain("directed 0");
        expect(rt.snapshot.directed).toBe(false);
        expect(rt.snapshot.edgeCount).toBe(3);
        const el = rt.snapshot.edgeList();
        expect(Array.from(el.src)).toEqual([0, 1, 2]);
        expect(Array.from(el.dst)).toEqual([1, 2, 2]);
        expect(rt.snapshot.edgeList().weights?.[0]).toBe(5);
    });
});

describe("gmlExporter: keys", () => {
    it("refuses column names that are not GML keys unless sanitizeKeys is mangle", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.setNodeValue("graphty.pagerank.rank", builder.addNode(1), 0.5);
        builder.setNodeValue("1st", 0, 2);
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot);
        expect(codes(notes)).toEqual([INVALID_KEY_CODE, INVALID_KEY_CODE]);
        expect(notes[0].column).toBe("graphty.pagerank.rank");
        await expectExportError(snapshot, undefined, "E_UNSUPPORTED");
        const mangled = gmlExporter.check(snapshot, { sanitizeKeys: "mangle" });
        expect(codes(mangled)).toEqual([KEY_MANGLED_CODE, KEY_MANGLED_CODE]);
        const text = await gmlExporter.exportToString(snapshot, { sanitizeKeys: "mangle" });
        expect(text).toContain("graphty_pagerank_rank 0.5\n    x1st 2");
        expect(() => gmlExporter.check(snapshot, { sanitizeKeys: "drop" as never })).toThrow(
            expect.objectContaining({ code: "E_UNSUPPORTED" }),
        );
    });

    it("refuses column names that collide with structural keys", async () => {
        const builder = new GraphBuilder({ directed: false });
        const a = builder.addNode(1);
        builder.setNodeValue("id", a, 5);
        const e = builder.addEdge(1, 1, 2);
        builder.setEdgeValue("source", e, "s");
        builder.setEdgeValue("value", e, "v");
        builder.setGraphValue("node", 1);
        builder.setGraphValue("graph", 1, { extra: { gmlTopLevel: true } });
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot);
        expect(codes(notes)).toEqual([
            RESERVED_KEY_CODE,
            RESERVED_KEY_CODE,
            RESERVED_KEY_CODE,
            RESERVED_KEY_CODE,
            RESERVED_KEY_CODE,
        ]);
        expect(notes.map((n) => n.column)).toEqual(["id", "source", "value", "node", "graph"]);
        await expectExportError(snapshot, undefined, "E_UNSUPPORTED");
        const text = await gmlExporter.exportToString(snapshot, { sanitizeKeys: "mangle" });
        expect(text).toContain("id 1\n    id_2 5");
        expect(text).toContain('source 1\n    target 1\n    value 2.0\n    source_2 "s"\n    value_2 "v"');
        expect(text).toContain("graph_2 1\ngraph [");
        expect(text).toContain("node_2 1");
    });

    it("the weight key is reserved only when weights are written, and the original id key only when mangling", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.setEdgeValue("value", builder.addEdge(1, 2), 3);
        builder.setNodeValue(ORIGINAL_ID_KEY, 0, "o");
        const snapshot = builder.freeze();
        // a plain "value" column reads back as THE weight under the default weightFrom
        expect(codes(gmlExporter.check(snapshot))).toEqual([LOSS.WEIGHT_KEY_CLASH]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, {
            importOptions: { weightFrom: null, restoreMangledIds: false },
        });
        expectSameSnapshot(snapshot, rt.snapshot, { allowExtraColumns: false });
        const mangling = new GraphBuilder({ directed: false });
        mangling.setNodeValue(ORIGINAL_ID_KEY, mangling.addNode("a"), "o");
        expect(codes(gmlExporter.check(mangling.freeze(), { sanitizeIds: "mangle" }))).toEqual([
            LOSS.ID_MANGLED,
            RESERVED_KEY_CODE,
        ]);
    });

    it("writes a column renamed with #id under its original key", async () => {
        const snapshot = await importGml("graph [ node [ id 1 position 3 graphics [ x 1 y 2 ] ] ]");
        expect(snapshot.nodes.names()).toEqual(["position", "position#graphics"]);
        const out = await exact(snapshot);
        expect(out).toContain("position 3\n    graphics [\n      x 1.0\n      y 2.0\n      z 0.0\n    ]");
    });

    it("writes a graph column renamed with #id under its original key at its original level", async () => {
        const snapshot = await importGml('license "MIT"\ngraph [ license "GPL" node [ id 1 ] ]');
        expect([...snapshot.graph.names()].sort()).toEqual(["license", "license#license"]);
        const out = await exact(snapshot);
        expect(out.startsWith('license "MIT"\ngraph [\n  directed 0\n  license "GPL"\n')).toBe(true);
    });

    it("reserves id on edges when a role-id column is written as the edge id", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.declareEdgeColumn({ name: "edgeId", dtype: "string", role: "id" });
        const e = builder.addEdge(1, 2);
        builder.setEdgeValue("edgeId", e, "e0");
        builder.setEdgeValue("id", e, 9);
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot);
        // the role-id column "edgeId" is written as the edge id key and reads back as "id"
        expect(codes(notes)).toEqual([LOSS.COLUMN_NAME_CHANGED, RESERVED_KEY_CODE]);
        expect(notes[0].column).toBe("edgeId");
        expect(notes[1].column).toBe("id");
        const text = await gmlExporter.exportToString(snapshot, { sanitizeKeys: "mangle" });
        expect(text).toContain('source 1\n    target 2\n    id "e0"\n    id_2 9');
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter, { exportOptions: { sanitizeKeys: "mangle" } });
        expect(rt.snapshot.edges.byRole("id")?.value(0)).toBe("e0");
    });

    it("reports two columns that would share a key", async () => {
        const snapshot = await importGml("graph [ node [ id 1 a 1 ] ]");
        const builder = GraphBuilder.from(snapshot);
        builder.declareNodeColumn({ name: "a#2", dtype: "i32", origin: { format: "gml", id: "a", type: "int" } });
        builder.setNodeValue("a#2", 0, 2);
        const notes = gmlExporter.check(builder.freeze());
        expect(codes(notes)).toEqual([RESERVED_KEY_CODE]);
        expect(notes[0].message).toContain("repeats another column's key");
    });

    it("checks record keys inside json values", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "meta", dtype: "json" });
        builder.setNodeValue("meta", builder.addNode(1), { "a.b": 1, ok: "x" });
        const snapshot = builder.freeze();
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_NUMBER_TYPE_CODE, INVALID_KEY_CODE]);
        await expectExportError(snapshot, undefined, "E_UNSUPPORTED");
        expect(codes(gmlExporter.check(snapshot, { sanitizeKeys: "mangle" }))).toEqual([
            RECORD_NUMBER_TYPE_CODE,
            KEY_MANGLED_CODE,
        ]);
        expect(await gmlExporter.exportToString(snapshot, { sanitizeKeys: "mangle" })).toContain(
            'meta [\n      a_b 1\n      ok "x"\n    ]',
        );
    });
});

describe("gmlExporter: json edge cases", () => {
    function withJson(value: unknown): GraphSnapshot {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "j", dtype: "json" });
        builder.setNodeValue("j", builder.addNode(1), value);
        return builder.freeze();
    }

    it("reports booleans and nulls and writes them as 1 / 0 and omission", async () => {
        const snapshot = withJson({ t: true, f: false, n: null, s: "x" });
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_BOOLEAN_CODE, RECORD_NULL_CODE]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.text).toContain('j [\n      t 1\n      f 0\n      s "x"\n    ]');
        expect(rt.snapshot.nodes.value("j", 0)).toEqual({ t: 1, f: 0, s: "x" });
    });

    it("omits a null row value with a note", async () => {
        const snapshot = withJson(null);
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_NULL_CODE]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.snapshot.nodes.has("j")).toBe(false);
    });

    it("writes a top-level array as repeated keys with a note; it re-imports as a list", async () => {
        const snapshot = withJson([1, 2]);
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_NUMBER_TYPE_CODE, JSON_ARRAY_CODE]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.text).toContain("j 1\n    j 2");
        expect(rt.snapshot.nodes.require("j").dtype).toBe("list");
        const single = withJson(["only"]);
        expect(await gmlExporter.exportToString(single)).toContain(`j "${LIST_START_MARKER}"\n    j "only"`);
        const empty = withJson([]);
        expect(await gmlExporter.exportToString(empty)).toContain('j "[]"');
    });

    it("refuses arrays nested in arrays", async () => {
        const snapshot = withJson({ grid: [[1, 2], [3]] });
        expect(codes(gmlExporter.check(snapshot))).toEqual([RECORD_NUMBER_TYPE_CODE, NESTED_ARRAY_CODE]);
        await expectExportError(snapshot, undefined, "E_UNSUPPORTED");
    });

    it("writes scalars in json columns and lists of records", async () => {
        const snapshot = withJson("plain");
        expect(gmlExporter.check(snapshot)).toEqual([]);
        expect(await gmlExporter.exportToString(snapshot)).toContain('j "plain"');
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "p", dtype: "list", itemDtype: "json" });
        builder.setNodeValue("p", builder.addNode(1), [{ x: 1 }, "s", 2.5]);
        const list = builder.freeze();
        expect(codes(gmlExporter.check(list))).toEqual([RECORD_NUMBER_TYPE_CODE]);
        const rt = await roundTrip(list, gmlExporter, gmlImporter);
        expect(rt.text).toContain('p [\n      x 1\n    ]\n    p "s"\n    p 2.5');
        expect(rt.snapshot.nodes.value("p", 0)).toEqual([{ x: 1 }, "s", 2.5]);
    });
});

describe("gmlExporter: graph attributes and metadata", () => {
    it("writes in-graph columns inside graph [ ] and top-level ones before it, with Creator and Version", async () => {
        const text =
            'Creator "me"\nVersion 1\nlicense "MIT"\ngraph [ name "g" tags "a" tags "b" info [ k "v" ] node [ id 1 ] ]';
        const snapshot = await importGml(text);
        const out = await exact(snapshot);
        expect(
            out.startsWith(
                'Creator "me"\nVersion 1\nlicense "MIT"\ngraph [\n  directed 0\n  name "g"\n  info [\n    k "v"\n  ]\n  tags "a"\n  tags "b"\n',
            ),
        ).toBe(true);
    });

    it("quotes a non-numeric Version and skips Version for other source formats", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.setMeta({ sourceFormat: "gml", sourceVersion: "1.0a", creator: "c" });
        expect(await gmlExporter.exportToString(builder.freeze())).toContain('Creator "c"\nVersion "1.0a"\n');
        const other = new GraphBuilder({ directed: false });
        other.setMeta({ sourceFormat: "gexf", sourceVersion: "1.3" });
        expect(await gmlExporter.exportToString(other.freeze())).toBe("graph [\n  directed 0\n]\n");
    });

    it("writes multigraph from the declared flag or the topology", async () => {
        const parallel = new GraphBuilder({ directed: true });
        parallel.addEdge(1, 2);
        parallel.addEdge(1, 2);
        expect(await gmlExporter.exportToString(parallel.freeze())).toContain("directed 1\n  multigraph 1\n");
        const declared = new GraphBuilder({ directed: false });
        declared.setMeta({ declaredMultigraph: false });
        expect(await gmlExporter.exportToString(declared.freeze())).toContain("multigraph 0");
    });
});

describe("gmlExporter: dropped columns", () => {
    it("reports and skips the roles GML cannot hold", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.declareNodeColumn({ name: "color", dtype: "f32", components: 4, role: "color" });
        builder.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        builder.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
        builder.declareNodeColumn({ name: "start.text", dtype: "string", role: "timeText", extra: { for: "start" } });
        builder.declareNodeColumn({ name: "kind", dtype: "string", options: ["a", "b"], default: "a" });
        builder.declareNodeColumn({ name: "price", dtype: "f64", dynamic: true });
        builder.declareEdgeColumn({
            name: "spells",
            dtype: "list",
            itemDtype: "f64",
            itemComponents: 2,
            role: "spells",
        });
        builder.declareEdgeColumn({ name: "open", dtype: "u8", role: "open" });
        const a = builder.addNode(1);
        const b = builder.addNode(2);
        builder.setNodeValue("color", a, [1, 0, 0, 1]);
        builder.setNodeValue("parent", b, a);
        builder.setNodeValue("start", a, 5);
        builder.setNodeValue("start.text", a, "2020");
        builder.setNodeValue("kind", a, "b");
        builder.setNodeValue("price", a, 1.5);
        const e = builder.addEdge(1, 2);
        builder.setEdgeValue("spells", e, [[1, 2]]);
        builder.setEdgeValue("open", e, 1);
        const table = builder.addExtensionTable("temporal:node:price", [
            { name: "row", dtype: "u32", refersTo: "node" },
            { name: "value", dtype: "f64" },
        ]);
        builder.addExtensionRow(table, [a, 2]);
        const other = builder.addExtensionTable("other", [{ name: "x", dtype: "i32" }]);
        builder.addExtensionRow(other, [1]);
        const snapshot = builder.freeze();
        const notes = gmlExporter.check(snapshot);
        expect(codes(notes)).toEqual([
            LOSS.VIZ,
            LOSS.HIERARCHY,
            LOSS.TEMPORAL,
            LOSS.TEMPORAL_TEXT,
            LOSS.DEFAULT,
            LOSS.OPTIONS,
            LOSS.DYNAMIC_VALUES,
            LOSS.SPELLS,
            LOSS.OPEN_INTERVAL,
            LOSS.DYNAMIC_VALUES,
            LOSS.EXTENSION_TABLE,
        ]);
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.snapshot.nodes.names()).toEqual(["kind", "price"]);
        expect(rt.snapshot.edges.names()).toEqual([]);
        expect(rt.snapshot.extensions.size).toBe(0);
        expect(rt.text).toContain('kind "b"\n    price 1.5');
    });

    it("writes only the set rows of a column", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.declareNodeColumn({ name: "s", dtype: "string" });
        builder.addNode(1);
        builder.setNodeValue("s", builder.addNode(2), "two");
        const snapshot = builder.freeze();
        const rt = await roundTrip(snapshot, gmlExporter, gmlImporter);
        expect(rt.text).toContain('node [\n    id 1\n  ]\n  node [\n    id 2\n    s "two"\n  ]');
        const diffs = compareSnapshots(snapshot, rt.snapshot, { allowExtraColumns: false });
        expect(diffs, describeDiffs(diffs)).toEqual([]);
    });
});
