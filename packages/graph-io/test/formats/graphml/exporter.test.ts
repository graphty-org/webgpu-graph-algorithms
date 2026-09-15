import { GraphBuilder, type GraphBuilderOptions, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { decodeChunks } from "../../../src/common/writer.js";
import { GRAPHML_ISSUE, GRAPHML_LOSS } from "../../../src/formats/graphml/constants.js";
import { graphmlExporter, type GraphmlExportOptions } from "../../../src/formats/graphml/exporter.js";
import { graphmlImporter } from "../../../src/formats/graphml/importer.js";
import { type CommonExportOptions, type CommonImportOptions, type ImportReport } from "../../../src/types.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

type ExportOptions = GraphmlExportOptions & CommonExportOptions;

async function importText(
    text: string,
    options?: CommonImportOptions,
    builderOptions: Partial<GraphBuilderOptions> = {},
): Promise<{ snapshot: GraphSnapshot; report: ImportReport }> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64", ...builderOptions });
    const report = await graphmlImporter.import(text, builder, options);
    return { snapshot: builder.freeze(), report };
}

function noteCodes(snapshot: GraphSnapshot, options?: ExportOptions): string[] {
    return graphmlExporter.check(snapshot, options).map((note) => note.code);
}

function builder(directed = true): GraphBuilder {
    return new GraphBuilder({ directed, weightDtype: "f64" });
}

/** A directed snapshot with one directed, one undirected and one mutual edge (through the resolver). */
function mixedSnapshot(): GraphSnapshot {
    const b = builder(true);
    const resolver = new DirectionResolver(b, new ImportReportBuilder("test", 100), "expand");
    resolver.setHeader(true);
    b.addNode("a");
    b.addNode("b");
    b.addNode("c");
    resolver.addEdge("a", "b", "directed", 2);
    resolver.addEdge("b", "c", "undirected");
    resolver.addEdge("c", "a", "mutual", 3);
    return b.freeze();
}

async function reimport(text: string, options?: CommonImportOptions): Promise<GraphSnapshot> {
    return (await importText(text, options)).snapshot;
}

describe("graphmlExporter capabilities and options", () => {
    it("declares what GraphML keeps", () => {
        expect(graphmlExporter.format).toBe("graphml");
        expect(graphmlExporter.capabilities).toMatchObject({
            mixedDirection: true,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "optional",
            idCharset: "nmtoken",
            dtypes: ["bool", "i32", "f32", "f64", "string"],
            lists: false,
            json: false,
            defaults: true,
            options: false,
            hierarchy: true,
            temporal: "none",
            graphAttributes: true,
            positions: false,
            viz: false,
        });
    });

    it("rejects bad format options and bad common options", () => {
        const snapshot = builder().freeze();
        expect(() => graphmlExporter.check(snapshot, { pretty: "yes" } as unknown as ExportOptions)).toThrow(
            GraphFormatError,
        );
        expect(() => graphmlExporter.check(snapshot, { edgedefault: "mixed" } as unknown as ExportOptions)).toThrow(
            GraphFormatError,
        );
        expect(() => graphmlExporter.check(snapshot, { sanitizeIds: "always" } as unknown as ExportOptions)).toThrow(
            GraphFormatError,
        );
    });

    it("export() streams the same bytes exportToString() returns, with or without indentation", async () => {
        const { snapshot } = await importText(readCorpusText("graphml", "simple.graphml"));
        const text = await graphmlExporter.exportToString(snapshot);
        expect(await decodeChunks(graphmlExporter.export(snapshot))).toBe(text);
        expect(text).toContain('\n    <node id="n0">\n      <data key="d0">Node A</data>\n    </node>');
        const flat = await graphmlExporter.exportToString(snapshot, { pretty: false });
        expect(flat).toContain('\n<node id="n0">\n<data key="d0">Node A</data>\n</node>');
        expectSameSnapshot(snapshot, await reimport(flat));
    });

    it("writes the XML declaration, the namespace, the schema location and a graph id", async () => {
        const { snapshot } = await importText(readCorpusText("graphml", "simple.graphml"));
        const text = await graphmlExporter.exportToString(snapshot);
        expect(
            text.startsWith(
                '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            ),
        ).toBe(true);
        expect(text).toContain(
            'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"',
        );
        expect(text).toContain('<graph id="G" edgedefault="undirected">');
        expect(text.endsWith("</graph>\n</graphml>\n")).toBe(true);
    });
});

describe("graphmlExporter corpus round trips", () => {
    for (const entry of corpusFiles("graphml")) {
        it(`round-trips ${entry.path} exactly`, async () => {
            const { snapshot } = await importText(readCorpusText("graphml", entry.path));
            const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter, {
                exportOptions: { sanitizeIds: "mangle" },
            });
            expect(result.report.errorCount).toBe(0);
            const diffs = compareSnapshots(snapshot, result.snapshot, { originType: true, allowExtraColumns: false });
            expect(diffs, describeDiffs(diffs)).toEqual([]);
            expect(result.snapshot.meta.weightOrigin).toEqual(snapshot.meta.weightOrigin);
            expect(result.snapshot.meta.extra).toEqual({
                graphml: { ...(snapshot.meta.extra.graphml as object), graphId: "G" },
            });
            // a second round trip is stable too
            const again = await roundTrip(result.snapshot, graphmlExporter, graphmlImporter, {
                exportOptions: { sanitizeIds: "mangle" },
            });
            expect(again.text).toBe(result.text);
        });
    }

    it("simple.graphml exports without notes and with its declarations restored", async () => {
        const { snapshot } = await importText(readCorpusText("graphml", "simple.graphml"));
        expect(noteCodes(snapshot)).toEqual([]);
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).toContain('<key id="d0" for="node" attr.name="label" attr.type="string"/>');
        expect(text).toContain('<key id="d1" for="edge" attr.name="weight" attr.type="double"/>');
        expect(text).toContain('<edge source="n1" target="n3">\n      <data key="d1">1.5</data>\n    </edge>');
    });

    it("got-network.graphml needs mangling for two ids and restores them on re-import", async () => {
        const { snapshot } = await importText(readCorpusText("graphml", "got-network.graphml"));
        const notes = graphmlExporter.check(snapshot);
        expect(notes.map((n) => n.code)).toEqual([LOSS.ID_CHARSET]);
        expect(notes[0].count).toBe(2);
        await expect(graphmlExporter.exportToString(snapshot)).rejects.toMatchObject({ code: "E_INVALID_ID" });
        expect(noteCodes(snapshot, { sanitizeIds: "mangle" })).toEqual([LOSS.ID_MANGLED]);
        const text = await graphmlExporter.exportToString(snapshot, { sanitizeIds: "mangle" });
        expect(text).toContain('<key id="d0" for="node" attr.name="graphty:originalId" attr.type="string"/>');
        expect(text).toContain('<node id="Jon_Arryn">\n      <data key="d0">Jon Arryn</data>');
        expect(text).toContain('<key id="weight" for="edge" attr.name="weight" attr.type="double"/>');
        expect(text).toContain('<edge id="0" source="Aemon" target="Grenn">');
        const kept = await reimport(text, { restoreMangledIds: false });
        expect(kept.ids.indexOf("Jon_Arryn")).toBeGreaterThanOrEqual(0);
        expect(kept.nodes.require("graphty.originalId").value(kept.ids.indexOf("Jon_Arryn"))).toBe("Jon Arryn");
    });

    it("yfiles-sample.graphml re-exports the nested XML and the y namespace", async () => {
        const { snapshot } = await importText(readCorpusText("graphml", "yfiles-sample.graphml"));
        expect(noteCodes(snapshot)).toEqual([]);
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).toContain('xmlns:y="http://www.yworks.com/xml/graphml"');
        expect(text).toContain('<key id="d0" for="node" yfiles.type="nodegraphics"/>');
        expect(text).toContain(
            '<data key="d0">\n        <y:ShapeNode>\n          <y:Geometry x="0.0" y="0.0" width="60.0" height="30.0"/>',
        );
        expect(text).toContain("<y:NodeLabel>Start</y:NodeLabel>");
    });
});

describe("graphmlExporter check() notes", () => {
    it("reports mutual edges written as undirected and folds expanded pairs", async () => {
        const snapshot = mixedSnapshot();
        expect(noteCodes(snapshot)).toEqual([GRAPHML_LOSS.MUTUAL_AS_UNDIRECTED]);
        const text = await graphmlExporter.exportToString(snapshot);
        // two of the three source edges are undirected (the mutual one folds to undirected): majority wins
        expect(text).toContain('<graph id="G" edgedefault="undirected">');
        expect(text).toContain('<edge source="a" target="b" directed="true">');
        expect(text).toContain('<edge source="b" target="c"/>');
        expect(text).toContain('<edge source="c" target="a">\n      <data key="d0">3</data>');
        expect((text.match(/<edge /g) ?? []).length).toBe(3);
        const back = await reimport(text);
        expect(back.edgeCount).toBe(5);
        expect(back.edges.byRole("mutual")).toBeNull();
        // the mutual flag and the directed=true the resolver gives a mutual pair are what the note announces as lost
        const diffs = compareSnapshots(snapshot, back, { ignoreRoles: ["mutual", "directed"] });
        expect(diffs, describeDiffs(diffs)).toEqual([]);
        expect(Array.from({ length: 5 }, (_, e) => back.edges.require("graphty.directed").value(e))).toEqual([
            true,
            false,
            false,
            false,
            false,
        ]);
    });

    it("reports dtypes GraphML cannot keep and writes them as text", async () => {
        const b = builder(false);
        b.addNode("a");
        b.addNode("b");
        const dict = b.declareNodeColumn({ name: "cat", dtype: "dict" });
        const u32 = b.declareNodeColumn({ name: "big", dtype: "u32" });
        const small = b.declareNodeColumn({ name: "small", dtype: "u32" });
        const u8 = b.declareNodeColumn({ name: "byte", dtype: "u8" });
        const list = b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        const json = b.declareNodeColumn({ name: "blob", dtype: "json" });
        const vec = b.declareNodeColumn({ name: "vec", dtype: "f64", components: 2 });
        b.setNodeValue(dict, 0, "x");
        b.setNodeValue(dict, 1, "y");
        b.setNodeValue(u32, 0, 4294967295);
        b.setNodeValue(small, 0, 7);
        b.setNodeValue(u8, 0, 200);
        b.setNodeValue(list, 0, ["p", "q"]);
        b.setNodeValue(json, 0, { k: [1, 2] });
        b.setNodeValue(vec, 0, [1.5, 2.5]);
        const snapshot = b.freeze();
        const codes = noteCodes(snapshot);
        expect(codes).toEqual([LOSS.DTYPE, LOSS.DTYPE, LOSS.DTYPE, LOSS.DTYPE, LOSS.LIST, LOSS.COMPONENTS, LOSS.JSON]);
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).toContain('<key id="d0" for="node" attr.name="cat" attr.type="string"/>');
        expect(text).toContain('<key id="d1" for="node" attr.name="big" attr.type="long"/>');
        expect(text).toContain('<key id="d2" for="node" attr.name="small" attr.type="int"/>');
        expect(text).toContain('<key id="d3" for="node" attr.name="byte" attr.type="int"/>');
        expect(text).toContain('<key id="d4" for="node" attr.name="tags" attr.type="string"/>');
        expect(text).toContain('<data key="d1">4294967295</data>');
        expect(text).toContain('<data key="d4">["p","q"]</data>');
        expect(text).toContain('<data key="d5">{"k":[1,2]}</data>');
        expect(text).toContain('<data key="d6">[1.5,2.5]</data>');
        const back = await reimport(text);
        expect(back.nodes.require("cat").dtype).toBe("string");
        expect(back.nodes.require("big").dtype).toBe("f64");
        expect(back.nodes.require("big").value(0)).toBe(4294967295);
        expect(back.nodes.require("small").dtype).toBe("i32");
        expect(back.nodes.require("tags").value(0)).toBe('["p","q"]');
    });

    it("reports positions, visual, temporal, hierarchy roles and extension tables, and does not write them", async () => {
        const b = builder(true);
        b.addNode("a");
        b.addNode("b");
        const pos = b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, role: "position" });
        const color = b.declareNodeColumn({ name: "color", dtype: "string", role: "color" });
        const start = b.declareEdgeColumn({ name: "start", dtype: "f64", role: "start" });
        const parents = b.declareNodeColumn({
            name: "parents",
            dtype: "list",
            itemDtype: "u32",
            role: "parents",
            refersTo: "node",
        });
        const kind = b.declareNodeColumn({ name: "kind", dtype: "string", role: "kind" });
        const opts = b.declareNodeColumn({ name: "level", dtype: "i32", options: [1, 2, 3] });
        b.setNodeValue(pos, 0, [1, 2, 3]);
        b.setNodeValue(color, 0, "#ff0000");
        b.setNodeValue(parents, 1, [0]);
        b.setNodeValue(kind, 0, "person");
        b.setNodeValue(opts, 0, 2);
        b.addEdge("a", "b");
        b.setEdgeValue(start, 0, 5);
        const table = b.addExtensionTable("temporal:node:price", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "end", dtype: "f64" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(table, [0, 1, 2, 3.5]);
        const snapshot = b.freeze();
        const codes = noteCodes(snapshot);
        expect(codes).toContain(LOSS.POSITIONS);
        expect(codes).toContain(LOSS.VIZ);
        expect(codes).toContain(LOSS.TEMPORAL);
        expect(codes).toContain(LOSS.OPTIONS);
        expect(codes).toContain(LOSS.DYNAMIC_VALUES);
        expect(codes).toContain(GRAPHML_LOSS.PARENTS_DROPPED);
        expect(codes).toContain(GRAPHML_LOSS.ROLE_DROPPED);
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).not.toContain('attr.name="pos"');
        expect(text).not.toContain('attr.name="color"');
        expect(text).not.toContain('attr.name="start"');
        expect(text).not.toContain('attr.name="parents"');
        expect(text).toContain('attr.name="kind"');
        expect(text).toContain('attr.name="level" attr.type="int"');
        const back = await reimport(text);
        expect(back.nodes.names()).toEqual(["kind", "level"]);
        expect(back.nodes.require("kind").meta.role).toBeNull();
    });

    it("reports ids that change type under the canonical rule and numeric edge ids", async () => {
        const b = builder(true);
        b.addNode("1");
        b.addNode(2.5);
        b.addNode(3);
        const ids = b.declareEdgeColumn({ name: "id", dtype: "f64", role: "id", unique: true });
        b.addEdge("1", 3);
        b.setEdgeValue(ids, 0, 7);
        const snapshot = b.freeze();
        const notes = graphmlExporter.check(snapshot);
        expect(notes.map((n) => n.code)).toEqual([GRAPHML_LOSS.ID_TEXT_TYPE, GRAPHML_LOSS.EDGE_ID_TEXT]);
        expect(notes[0].count).toBe(2);
        const back = await reimport(await graphmlExporter.exportToString(snapshot));
        expect(back.ids.toArray()).toEqual([1, "2.5", 3]);
        expect(back.edges.require("id").value(0)).toBe("7");
    });

    it("refuses or mangles edge ids outside the nmtoken charset", async () => {
        const b = builder(true);
        b.addNode("a");
        const ids = b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
        b.addEdge("a", "a");
        b.addEdge("a", "a");
        b.addEdge("a", "a");
        b.setEdgeValue(ids, 0, "has space");
        b.setEdgeValue(ids, 1, "has_space");
        b.setEdgeValue(ids, 2, "has space!");
        const snapshot = b.freeze();
        expect(noteCodes(snapshot)).toEqual([LOSS.ID_CHARSET]);
        await expect(graphmlExporter.exportToString(snapshot)).rejects.toMatchObject({ code: "E_INVALID_ID" });
        expect(noteCodes(snapshot, { sanitizeIds: "mangle" })).toEqual([LOSS.ID_MANGLED]);
        const text = await graphmlExporter.exportToString(snapshot, { sanitizeIds: "mangle" });
        expect(text).toContain('<edge id="has_space_2" source="a" target="a"/>');
        expect(text).toContain('<edge id="has_space" source="a" target="a"/>');
        expect(text).toContain('<edge id="has_space_" source="a" target="a"/>');
    });

    it("reports yfiles values that are not trees and refuses to write them", async () => {
        const b = builder(true);
        b.addNode("a");
        const graphics = b.declareNodeColumn({
            name: "d0",
            dtype: "json",
            origin: { format: "graphml", id: "d0", type: "nodegraphics", namespace: "yfiles" },
        });
        b.setNodeValue(graphics, 0, { "bad name": 1 });
        const snapshot = b.freeze();
        expect(noteCodes(snapshot)).toEqual([GRAPHML_LOSS.YFILES_TREE]);
        await expect(graphmlExporter.exportToString(snapshot)).rejects.toMatchObject({ code: "E_COLUMN_TYPE" });
    });

    it("reports a weight key colliding with an edge column named weight", () => {
        const b = builder(true);
        b.addNode("a");
        const w = b.declareEdgeColumn({ name: "weight", dtype: "string" });
        b.addEdge("a", "a", 2);
        b.setEdgeValue(w, 0, "heavy");
        expect(noteCodes(b.freeze())).toEqual([LOSS.WEIGHT_KEY_CLASH]);
    });

    it("reports containment reordering and parent cycles", async () => {
        const b = builder(true);
        b.addNode("child");
        b.addNode("parent");
        b.addNode("x");
        b.addNode("y");
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        b.setNodeValue(parent, 0, 1);
        b.setNodeValue(parent, 2, 3);
        b.setNodeValue(parent, 3, 2);
        const snapshot = b.freeze();
        const notes = graphmlExporter.check(snapshot);
        expect(notes.map((n) => n.code)).toEqual([GRAPHML_LOSS.HIERARCHY_REORDERED, GRAPHML_LOSS.PARENT_CYCLE]);
        expect(notes[1].count).toBe(2);
        const text = await graphmlExporter.exportToString(snapshot);
        const back = await reimport(text);
        expect(back.ids.toArray()).toEqual(["parent", "child", "x", "y"]);
        expect(Array.from({ length: 4 }, (_, i) => back.nodes.require("parent").value(i))).toEqual([
            undefined,
            0,
            undefined,
            2,
        ]);
    });

    it("reports graph attributes as kept and extension tables as dropped", () => {
        const b = builder(true);
        b.addNode("a");
        b.setGraphValue("title", "T", { dtype: "string" });
        const table = b.addExtensionTable("other:table", [{ name: "v", dtype: "i32" }]);
        b.addExtensionRow(table, [1]);
        const snapshot = b.freeze();
        expect(noteCodes(snapshot)).toEqual([LOSS.EXTENSION_TABLE]);
    });
});

describe("graphmlExporter round trips of synthetic snapshots", () => {
    it("keeps typed columns, defaults, graph attributes, descriptions and for=all keys", async () => {
        const keys = `<key id="a" for="all" attr.name="score" attr.type="double"><desc>d</desc><default>0.5</default></key>
<key id="n" for="node" attr.name="flag" attr.type="boolean"><default>false</default></key>
<key id="c" for="node" attr.name="count" attr.type="int"/>
<key id="l" for="node" attr.name="big" attr.type="long"/>
<key id="f" for="edge" attr.name="ratio" attr.type="float"/>
<key id="s" for="edge" attr.name="label" attr.type="string"/>
<key id="u" for="node" attr.name="when" attr.type="date"/>
<key id="g" for="graph" attr.name="title" attr.type="string"/>
`;
        const text = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
${keys}<graph id="main" edgedefault="directed">
<desc>The &amp; graph</desc>
<data key="g">Title &lt;1&gt;</data>
<data key="a">7.25</data>
<node id="a"><data key="n">true</data><data key="c">-3</data><data key="l">9007199254740992</data><data key="a">1e21</data><data key="u">2024-01-02</data></node>
<node id="b"><data key="c">2147483647</data></node>
<node id="c"/>
<edge id="e0" source="a" target="b" sourceport="p1" targetport="p2"><data key="f">0.1</data><data key="s">x &amp; y</data><data key="a">2</data></edge>
<edge id="e1" source="b" target="c"/>
</graph>
</graphml>`;
        const { snapshot, report } = await importText(text);
        expect(report.errorCount).toBe(0);
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
        expect(result.notes).toEqual([]);
        const diffs = compareSnapshots(snapshot, result.snapshot, { originType: true, allowExtraColumns: false });
        expect(diffs, describeDiffs(diffs)).toEqual([]);
        expect(result.text).toContain(
            '<key id="a" for="all" attr.name="score" attr.type="double">\n    <desc>d</desc>\n    <default>0.5</default>\n  </key>',
        );
        expect(result.text).toContain('<key id="l" for="node" attr.name="big" attr.type="long"/>');
        expect(result.text).toContain('<key id="u" for="node" attr.name="when" attr.type="date"/>');
        expect(result.text).toContain(
            '<graph id="main" edgedefault="directed">\n    <desc>The &amp; graph</desc>\n    <data key="a">7.25</data>\n    <data key="g">Title &lt;1&gt;</data>',
        );
        expect(result.text).toContain('<edge id="e0" source="a" target="b" sourceport="p1" targetport="p2">');
        expect(result.text).toContain('<data key="l">9007199254740992</data>');
        expect(result.text).toContain('<data key="a">1e+21</data>');
        expect(result.snapshot.meta.description).toBe("The & graph");
        expect(result.snapshot.graph.require("title").value(0)).toBe("Title <1>");
        expect(result.snapshot.edges.require("ratio").value(0)).toBe(Math.fround(0.1));
        // the long column's f64 values are integral, so the declaration is restored; non-integral values demote it
        const b = GraphBuilder.from(snapshot);
        b.setNodeValue("big", 2, 1.5);
        const demoted = await graphmlExporter.exportToString(b.freeze());
        expect(demoted).toContain('<key id="l" for="node" attr.name="big" attr.type="double"/>');
    });

    it("keeps explicit and absent weights apart, and integral declared weights", async () => {
        const text = `<graphml><key id="w" for="edge" attr.name="weight" attr.type="int"/><graph edgedefault="undirected">
<edge source="a" target="b"><data key="w">3</data></edge><edge source="b" target="c"/><edge source="c" target="a"><data key="w">1</data></edge>
</graph></graphml>`;
        const { snapshot } = await importText(text);
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
        expect(result.text).toContain('<key id="w" for="edge" attr.name="weight" attr.type="int"/>');
        expect(result.text).toContain('<edge source="b" target="c"/>');
        expect(result.text).toContain('<data key="w">3</data>');
        expect(result.text).toContain('<data key="w">1</data>');
        expectSameSnapshot(snapshot, result.snapshot);
        const shadow = result.snapshot.edges.byRole("weight");
        expect([0, 1, 2].map((e) => shadow?.isSet(e))).toEqual([true, false, true]);
    });

    it("writes f32 arc weights of a snapshot without a shadow column as the shortest round-tripping text", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f32" });
        b.addEdge("a", "b", 0.1);
        b.addEdge("b", "c", 16777216);
        const snapshot = b.freeze();
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).toContain('<key id="d0" for="edge" attr.name="weight" attr.type="double"/>');
        expect(text).toContain('<data key="d0">0.1</data>');
        expect(text).toContain('<data key="d0">16777216</data>');
        const back = await reimport(text);
        expect(Array.from(back.edgeList().weights ?? [])).toEqual(Array.from(snapshot.edgeList().weights ?? []));
    });

    it("round-trips both mixed-direction layouts exactly by recording edgedefault", async () => {
        const body = `<node id="a"/><node id="b"/><node id="c"/>
<edge source="a" target="b"/>
<edge source="b" target="c" directed="false"/>
<edge source="c" target="a" directed="true"/>
<edge source="c" target="c" directed="false"/>`;
        for (const edgedefault of ["directed", "undirected"]) {
            const text = `<graphml><graph edgedefault="${edgedefault}">${body}</graph></graphml>`;
            const { snapshot } = await importText(text);
            const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
            expect(result.text).toContain(`edgedefault="${edgedefault}"`);
            const diffs = compareSnapshots(snapshot, result.snapshot, { allowExtraColumns: false });
            expect(diffs, `${edgedefault}: ${describeDiffs(diffs)}`).toEqual([]);
        }
    });

    it("chooses the majority direction for a mixed snapshot without recorded metadata, or the override", async () => {
        const snapshot = mixedSnapshot();
        const text = await graphmlExporter.exportToString(snapshot);
        expect(text).toContain('edgedefault="undirected"');
        expect(text).toContain('<edge source="a" target="b" directed="true">');
        const forced = await graphmlExporter.exportToString(snapshot, { edgedefault: "directed" });
        expect(forced).toContain('edgedefault="directed"');
        expect(forced).toContain('<edge source="b" target="c" directed="false"/>');
        const undirected = builder(false);
        undirected.addEdge("a", "b");
        expect(await graphmlExporter.exportToString(undirected.freeze())).toContain('edgedefault="undirected"');
    });

    it("writes nested graphs from the parent column and restores them", async () => {
        const text = `<graphml><graph id="G" edgedefault="directed">
<node id="a"><graph id="a:" edgedefault="directed"><node id="b"/><node id="c"><graph id="c:" edgedefault="directed"><node id="d"/><node id="e"/></graph></node></graph></node>
<node id="f"/>
<edge source="b" target="d"/><edge source="f" target="a"/>
</graph></graphml>`;
        const { snapshot } = await importText(text);
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
        expect(result.notes).toEqual([]);
        expect(result.text).toContain(
            '<node id="a">\n      <graph id="a:" edgedefault="directed">\n        <node id="b"/>\n        <node id="c">\n          <graph id="c:" edgedefault="directed">\n            <node id="d"/>\n            <node id="e"/>\n          </graph>\n        </node>\n      </graph>\n    </node>\n    <node id="f"/>',
        );
        const diffs = compareSnapshots(snapshot, result.snapshot, { allowExtraColumns: false });
        expect(diffs, describeDiffs(diffs)).toEqual([]);
    });

    it("writes a deep containment chain without recursion", async () => {
        const b = builder(true);
        const depth = 5000;
        for (let i = 0; i < depth; i++) {
            b.addNode(`n${i}`);
        }
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        for (let i = 1; i < depth; i++) {
            b.setNodeValue(parent, i, i - 1);
        }
        const snapshot = b.freeze();
        const text = await graphmlExporter.exportToString(snapshot, { pretty: false });
        const back = await reimport(text);
        expect(back.nodeCount).toBe(depth);
        expect(back.nodes.require("parent").value(depth - 1)).toBe(depth - 2);
    });

    it("mangles node ids, keeps the originals, and the importer restores them", async () => {
        const b = builder(true);
        b.addNode("Little Rock, AR");
        b.addNode("Little_Rock__AR");
        b.addNode("");
        b.addNode(7);
        b.addEdge("Little Rock, AR", 7, 2);
        b.addEdge("", "Little_Rock__AR");
        const snapshot = b.freeze();
        const notes = graphmlExporter.check(snapshot, { sanitizeIds: "mangle" });
        expect(notes.map((n) => n.code)).toEqual([LOSS.ID_MANGLED]);
        expect(notes[0].count).toBe(2);
        const text = await graphmlExporter.exportToString(snapshot, { sanitizeIds: "mangle" });
        expect(text).toContain('<node id="Little_Rock__AR_2">\n      <data key="d1">Little Rock, AR</data>');
        expect(text).toContain('<node id="_">\n      <data key="d1"></data>');
        expect(text).toContain('<edge source="Little_Rock__AR_2" target="7">');
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter, {
            exportOptions: { sanitizeIds: "mangle" },
        });
        expect(result.snapshot.ids.toArray()).toEqual(["Little Rock, AR", "Little_Rock__AR", "", 7]);
        expectSameSnapshot(snapshot, result.snapshot, { allowExtraColumns: false });
        expect(result.report.issues).toEqual([]);
    });

    it("writes columns without a GraphML origin with generated key ids and dtype-derived types", async () => {
        const b = builder(true);
        b.addNode("a");
        b.addNode("b");
        const flag = b.declareNodeColumn({ name: "flag", dtype: "bool", default: true });
        const n = b.declareNodeColumn({ name: "n", dtype: "i32", default: 4 });
        const x = b.declareNodeColumn({ name: "x", dtype: "f32" });
        const label = b.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
        const kind = b.declareEdgeColumn({ name: "kind", dtype: "string", default: "plain" });
        b.setNodeValue(flag, 0, false);
        b.setNodeValue(n, 1, 9);
        b.setNodeValue(x, 0, 0.25);
        b.setNodeValue(label, 0, "A & B");
        b.addEdge("a", "b");
        b.setEdgeValue(kind, 0, "special");
        b.setGraphValue("note", 3.5, { dtype: "f64" });
        const snapshot = b.freeze();
        const notes = graphmlExporter.check(snapshot);
        // the label column takes the label slot (a key titled label) and reads back under that name
        expect(notes.map((note) => note.code)).toEqual([LOSS.COLUMN_NAME_CHANGED]);
        expect(notes[0].column).toBe("name");
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
        expect(result.text).toContain('<key id="d0" for="graph" attr.name="note" attr.type="double"/>');
        expect(result.text).toContain(
            '<key id="d1" for="node" attr.name="flag" attr.type="boolean">\n    <default>true</default>\n  </key>',
        );
        expect(result.text).toContain(
            '<key id="d2" for="node" attr.name="n" attr.type="int">\n    <default>4</default>\n  </key>',
        );
        expect(result.text).toContain('<key id="d3" for="node" attr.name="x" attr.type="float"/>');
        expect(result.text).toContain(
            '<key id="d5" for="edge" attr.name="kind" attr.type="string">\n    <default>plain</default>\n  </key>',
        );
        expect(result.text).toContain('<key id="d4" for="node" attr.name="label" attr.type="string"/>');
        expect(result.text).toContain('<data key="d4">A &amp; B</data>');
        expect(result.snapshot.nodes.names()).toEqual(["flag", "n", "x", "label"]);
        const back = result.snapshot.nodes.require("label");
        expect(back.meta.role).toBe("label");
        expect([back.value(0), back.isSet(1)]).toEqual(["A & B", false]);
        expect(result.snapshot.nodes.require("flag").meta.default).toBe(true);
        expect(result.snapshot.nodes.require("n").value(1)).toBe(9);
        expect(result.snapshot.nodes.require("x").value(0)).toBe(0.25);
        expect(result.snapshot.edges.require("kind").value(0)).toBe("special");
        expect(result.snapshot.graph.require("note").value(0)).toBe(3.5);
    });

    it("keeps a label column's own title when another column already holds the label slot", async () => {
        const b = builder(true);
        b.addNode("a");
        const plain = b.declareNodeColumn({ name: "label", dtype: "string" });
        const label = b.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
        b.setNodeValue(plain, 0, "plain");
        b.setNodeValue(label, 0, "A");
        const snapshot = b.freeze();
        const notes = graphmlExporter.check(snapshot);
        expect(notes.map((note) => [note.code, note.column])).toEqual([
            [LOSS.ROLE_ASSUMED, "label"],
            [GRAPHML_LOSS.ROLE_DROPPED, "name"],
        ]);
        const result = await roundTrip(snapshot, graphmlExporter, graphmlImporter);
        expect(result.text).toContain('<key id="d1" for="node" attr.name="name" attr.type="string"/>');
        expect(result.snapshot.nodes.names()).toEqual(["label", "name"]);
        // the importer gives the key titled label the role: what the ROLE_ASSUMED note announces
        expect(result.snapshot.nodes.require("label").meta.role).toBe("label");
        expect(result.snapshot.nodes.require("name").meta.role).toBeNull();
        expect(result.snapshot.nodes.require("name").value(0)).toBe("A");
    });

    it("merges columns sharing a GraphML origin.id into one for=all key only when they agree", async () => {
        const text = `<graphml><key id="k" for="all" attr.name="v" attr.type="int"/><graph edgedefault="directed">
<data key="k">1</data><node id="a"><data key="k">2</data></node><edge source="a" target="a"><data key="k">3</data></edge>
</graph></graphml>`;
        const { snapshot } = await importText(text);
        const out = await graphmlExporter.exportToString(snapshot);
        expect((out.match(/<key /g) ?? []).length).toBe(1);
        expect(out).toContain('<key id="k" for="all" attr.name="v" attr.type="int"/>');
        const b = builder(true);
        b.addNode("a");
        b.setGraphValue("v", 1, { dtype: "i32", origin: { format: "graphml", id: "k", type: "int" } });
        const v = b.declareNodeColumn({
            name: "v",
            dtype: "f64",
            origin: { format: "graphml", id: "k", type: "double" },
        });
        b.setNodeValue(v, 0, 2.5);
        const split = await graphmlExporter.exportToString(b.freeze());
        expect(split).toContain('<key id="k" for="graph" attr.name="v" attr.type="int"/>');
        expect(split).toContain('<key id="d0" for="node" attr.name="v" attr.type="double"/>');
    });

    it("re-declares the namespaces the importer recorded and adds xmlns:y for yfiles columns from elsewhere", async () => {
        const b = builder(true);
        b.addNode("a");
        const graphics = b.declareNodeColumn({
            name: "d0",
            dtype: "json",
            origin: { format: "graphml", id: "d0", type: "nodegraphics", namespace: "yfiles" },
        });
        b.setNodeValue(graphics, 0, { "y:ShapeNode": { "y:NodeLabel": "A" } });
        const plain = b.declareEdgeColumn({ name: "g", dtype: "json", origin: { namespace: "yfiles" } });
        b.addEdge("a", "a");
        b.setEdgeValue(plain, 0, { "y:PolyLineEdge": "" });
        const text = await graphmlExporter.exportToString(b.freeze());
        expect(text).toContain('<key id="d1" for="edge" attr.name="g" yfiles.type="edgegraphics"/>');
        expect(text).toContain('xmlns:y="http://www.yworks.com/xml/graphml"');
        expect(text).toContain("<y:ShapeNode>\n          <y:NodeLabel>A</y:NodeLabel>\n        </y:ShapeNode>");
        const back = await reimport(text);
        expect(back.nodes.require("d0").value(0)).toEqual({ "y:ShapeNode": { "y:NodeLabel": "A" } });
        expect(back.edges.require("g").value(0)).toEqual({ "y:PolyLineEdge": "" });
        expect(back.meta.extra).toEqual({
            graphml: { graphId: "G", edgedefault: "directed", namespaces: { y: "http://www.yworks.com/xml/graphml" } },
        });
    });
});

describe("graphmlExporter and importer issue codes stay distinct", () => {
    it("exposes stable code tables", () => {
        const values = [...Object.values(GRAPHML_ISSUE), ...Object.values(GRAPHML_LOSS)];
        expect(new Set(values).size).toBe(values.length);
    });
});
