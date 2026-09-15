import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { LOSS } from "../../../src/common/export.js";
import { decodeChunks } from "../../../src/common/writer.js";
import {
    GEXF_1_2_CAPABILITIES,
    GEXF_LOSS,
    gexfExporter,
    type GexfExportOptions,
} from "../../../src/formats/gexf/exporter.js";
import { gexfImporter } from "../../../src/formats/gexf/importer.js";
import { type CommonExportOptions, type LossNote } from "../../../src/types.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";
import { ACCENTED, BARE, DYNAMIC_1_3, MIXED_UNDIRECTED_HEADER, OPEN_1_2 } from "./fixtures.js";

type ExportOptions = (GexfExportOptions & CommonExportOptions) | undefined;

async function imported(text: string): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const report = await gexfImporter.import(text, builder);
    expect(report.errorCount).toBe(0);
    return builder.freeze();
}

function noteCodes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

async function exact(snapshot: GraphSnapshot, options?: ExportOptions): Promise<string> {
    const result = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: options });
    expect(result.report.errorCount, result.text).toBe(0);
    expectSameSnapshot(snapshot, result.snapshot);
    return result.text;
}

describe("gexfExporter: identity and capabilities", () => {
    it("declares the 1.3 capabilities and a 1.2 table", () => {
        expect(gexfExporter.format).toBe("gexf");
        expect(gexfExporter.capabilities).toMatchObject({
            mixedDirection: true,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "optional",
            idCharset: "any",
            lists: true,
            json: false,
            components: false,
            defaults: true,
            options: true,
            hierarchy: true,
            temporal: "dynamic-values",
            graphAttributes: false,
            positions: true,
            viz: true,
        });
        expect(gexfExporter.capabilities.dtypes).toEqual(["f32", "f64", "i32", "bool", "dict", "string"]);
        expect(GEXF_1_2_CAPABILITIES).toMatchObject({ multiEdges: false, edgeIds: "required" });
        expect(Object.isFrozen(gexfExporter.capabilities)).toBe(true);
    });

    it("rejects an unknown version and the common option values", async () => {
        const snapshot = await imported(BARE);
        expect(() => gexfExporter.check(snapshot, { version: "2.0" as never })).toThrow(
            expect.objectContaining({ code: "E_UNSUPPORTED" }),
        );
        await expect(gexfExporter.exportToString(snapshot, { sanitizeIds: "drop" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("export() yields the same bytes exportToString() returns", async () => {
        const snapshot = await imported(readCorpusText("gexf", "airlines-sample.gexf"));
        const text = await gexfExporter.exportToString(snapshot);
        const chunks: Uint8Array[] = [];
        for await (const chunk of gexfExporter.export(snapshot)) {
            chunks.push(chunk);
        }
        expect(chunks.length).toBeGreaterThan(1);
        expect(
            await decodeChunks(
                (async function* () {
                    await Promise.resolve();
                    yield* chunks;
                })(),
            ),
        ).toBe(text);
        expect(text.startsWith('<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://gexf.net/1.3"')).toBe(true);
    });
});

describe("gexfExporter: corpus round trips (design 16.5)", () => {
    for (const file of corpusFiles("gexf")) {
        for (const version of ["1.3", "1.2"] as const) {
            it(`${file.path} survives export as ${version} and re-import unchanged`, async () => {
                const snapshot = await imported(readCorpusText("gexf", file.path));
                expect(gexfExporter.check(snapshot, { version })).toEqual([]);
                const text = await exact(snapshot, { version });
                expect(text).toContain(`version="${version}"`);
            });
        }
    }

    it('re-exports lesmiserables.gexf without inventing weight="1" on edge 0', async () => {
        const snapshot = await imported(readCorpusText("gexf", "lesmiserables.gexf"));
        const text = await gexfExporter.exportToString(snapshot);
        expect(text).toContain('<edge id="0" source="1.0" target="0.0"/>');
        expect(text).toContain('<edge id="1" source="2.0" target="0.0" weight="8"/>');
        expect(text).toContain('defaultedgetype="undirected"');
        expect(text).toContain('idtype="string"');
        expect(text).not.toContain("<attributes");
    });

    it("re-exports airlines-sample.gexf with its declarations, values and colours", async () => {
        const snapshot = await imported(readCorpusText("gexf", "airlines-sample.gexf"));
        const text = await gexfExporter.exportToString(snapshot);
        expect(text).toContain('<attribute id="code" title="Code" type="string"/>');
        expect(text).toContain('<attribute id="latitude" title="latitude" type="double"/>');
        expect(text).toContain('<attvalue for="city" value="Little Rock, AR"/>');
        expect(text).toContain('<viz:color r="88" g="107" b="243"/>');
        expect(text).toContain('<meta lastmodifieddate="2010-05-17+15:02">');
        expect(text).toContain("<creator>Gephi 0.7</creator>");
        expect(text).toContain('<nodes count="235">');
        expect(text).toContain('<edges count="1297">');
    });
});

describe("gexfExporter: the 1.3 dynamic document", () => {
    it("round-trips exactly as 1.3 with no loss notes", async () => {
        const snapshot = await imported(DYNAMIC_1_3);
        expect(gexfExporter.check(snapshot)).toEqual([]);
        const text = await exact(snapshot);
        expect(text).toContain(
            '<graph defaultedgetype="directed" mode="dynamic" idtype="string" timeformat="date" timerepresentation="interval" start="2020-01-01" end="2021-01-01">',
        );
        expect(text).toContain('<attribute id="8" title="label" type="string"/>');
        expect(text).toContain('<attribute id="w" title="weight" type="double"/>');
        expect(text).toContain("<options>[a, b, c]</options>");
        expect(text).toContain('<attvalue for="4" value="[x, &quot;y, z&quot;, w]"/>');
        expect(text).toContain('<attvalue for="5" value="2001-02-03T04:05:06+02:00"/>');
        expect(text).toContain('<attvalue for="price" value="12.5" start="2020-07-01"/>');
        expect(text).toContain('<attvalue for="w" value="3.5" start="2020-06-01" end="2020-07-01"/>');
        expect(text).toContain(
            `<node id="b" label="${ACCENTED}" pid="a" timestamps="&lt;[2020-02-01, 2020-03-01]&gt;">`,
        );
        expect(text).toContain('<spell start="2020-03-01"/>');
        expect(text).toContain('<parent for="b"/>');
        expect(text).toContain('<node id="d" label="Dan" pid="c"/>');
        expect(text).toContain('<viz:color r="255" g="0" b="0" a="0.5"/>');
        expect(text).toContain('<viz:position x="1.5" y="-2" z="3"/>');
        expect(text).toContain('<edge id="e2" source="b" target="c" type="undirected"/>');
        expect(text).toContain('<edge id="e3" source="c" target="a" type="mutual" weight="1"/>');
        expect(text).toContain('<edge id="e4" source="a" target="a" type="undirected"/>');
        expect(text).toContain('<edge id="e5" source="a" target="b" kind="second"/>');
        expect(text).toContain('<edge id="e6" source="d" target="1" weight="7"/>');
        expect(text).toContain('<edges count="6">');
        expect(text).toContain("<description>a &amp; b &lt; c");
        expect(text).toContain("<keywords>alpha, beta</keywords>");
    });

    it("reports and applies every 1.2 loss", async () => {
        const snapshot = await imported(DYNAMIC_1_3);
        const notes = gexfExporter.check(snapshot, { version: "1.2" });
        expect(noteCodes(notes).sort()).toEqual(
            [
                LOSS.MULTI_EDGES,
                GEXF_LOSS.TIMESTAMP_AS_INTERVAL,
                GEXF_LOSS.KIND_DROPPED,
                GEXF_LOSS.DECLARED_TYPE,
                GEXF_LOSS.DECLARED_TYPE,
                // 1.2 has no date type: born is written as a double and its text companion is lost
                LOSS.TEMPORAL_TEXT,
            ].sort(),
        );
        expect(notes.find((n) => n.column === "born")?.message).toContain("date");
        expect(notes.find((n) => n.column === "nums")?.message).toContain("listinteger");
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(result.report.errorCount).toBe(0);
        expect(result.text).toContain('xmlns="http://www.gexf.net/1.2draft"');
        expect(result.text).toContain('<attribute id="10" title="nums" type="liststring"/>');
        expect(result.text).toContain('<attribute id="5" title="born" type="double"/>');
        expect(result.text).toContain('<attvalue for="4" value="x|y, z|w"/>');
        expect(result.text).toContain("<options>a|b|c</options>");
        expect(result.text).not.toContain("kind=");
        expect(result.text).not.toContain("timerepresentation");
        expect(result.text).toContain('<spell start="2020-02-01" end="2020-02-01"/>');
        const diffs = compareSnapshots(snapshot, result.snapshot, {
            ignoreColumns: ["born.text", "nums", "timestamps", "spells", "kind"],
        });
        expect(diffs).toEqual([]);
        expect(result.snapshot.nodes.get("nums")?.meta.itemDtype).toBe("string");
        expect(result.snapshot.nodes.value("born", 1)).toBe(snapshot.nodes.value("born", 1));
    });
});

describe("gexfExporter: the 1.2 document", () => {
    it("round-trips exactly as 1.2 and reports open intervals as 1.3", async () => {
        const snapshot = await imported(OPEN_1_2);
        expect(gexfExporter.check(snapshot, { version: "1.2" })).toEqual([
            expect.objectContaining({ code: LOSS.MULTI_EDGES }),
        ]);
        const text = await exact(snapshot, { version: "1.2" });
        // GEXF 1.2 open bounds carry the bound's time (dynamics.xsd: startopen / endopen are time-type)
        expect(text).toContain('<node id="1" label="one" startopen="1" end="5">');
        expect(text).toContain('<attvalue for="2" value="3" start="1" endopen="2"/>');
        expect(text).toContain('<edge id="1" source="2" target="1" start="3" endopen="4"/>');
        expect(text).toContain('<edge id="0" source="1" target="2" type="mutual" weight="0.1"/>');
        expect(text).toContain('timeformat="integer"');
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes).sort()).toEqual([LOSS.OPEN_INTERVAL, LOSS.OPEN_INTERVAL, LOSS.OPEN_INTERVAL].sort());
        expect(notes.map((n) => n.column).sort()).toEqual(["open", "open", "temporal:node:level"]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).not.toContain("startopen");
        expect(compareSnapshots(snapshot, result.snapshot, { ignoreColumns: ["open"] })).toEqual([]);
    });
});

describe("gexfExporter: direction folding (design 3.6)", () => {
    it("folds expanded pairs back into undirected and mutual edges", async () => {
        const snapshot = await imported(MIXED_UNDIRECTED_HEADER);
        const text = await exact(snapshot);
        expect(text).toContain('defaultedgetype="directed"');
        expect(text).toContain('<edges count="3">');
        expect(text).toContain('<edge id="u1" source="p" target="q" type="undirected"/>');
        expect(text).toContain('<edge id="d1" source="q" target="r"/>');
        expect(text).toContain('<edge id="m1" source="r" target="p" type="mutual"/>');
    });

    it("writes an undirected snapshot with an undirected default and no per-edge types", async () => {
        const snapshot = await imported(BARE);
        const text = await exact(snapshot);
        expect(text).toContain('defaultedgetype="undirected"');
        expect(text).not.toContain(' type="');
        expect(text).toContain('<node id="01"/>');
        expect(text).toContain('<edge source="y" target="01" weight="2"/>');
        expect(text).toContain('<edge source="x" target="y"/>');
        expect(text).toContain('idtype="string"');
        // a mixed id map declares no idtype
        const mixed = await imported(DYNAMIC_1_3);
        expect(mixed.ids.kind).toBe("mixed");
        expect(await gexfExporter.exportToString(mixed)).toContain('idtype="string"');
        const builder = new GraphBuilder({ directed: true });
        builder.addEdge(1, "one");
        expect(await gexfExporter.exportToString(builder.freeze())).not.toContain("idtype");
    });

    it("keeps a directed snapshot built by hand as directed edges", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addEdge("a", "b", 0.1);
        builder.addEdge("b", "a", 16777217);
        builder.addEdge("a", "a");
        const snapshot = builder.freeze();
        expect(gexfExporter.check(snapshot)).toEqual([]);
        const text = await exact(snapshot);
        expect(text).toContain('<edge source="a" target="b" weight="0.1"/>');
        expect(text).toContain('<edge source="b" target="a" weight="16777217"/>');
        expect(text).toContain('<edge source="a" target="a"/>');
        expect(text).toContain('idtype="string"');
    });

    it("writes an unweighted f32-weighted snapshot from the arc array", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.addEdge(1, 2, 0.5);
        builder.addEdge(2, 3, 2);
        const snapshot = builder.freeze();
        expect(snapshot.edges.byRole("weight")).toBeNull();
        const text = await exact(snapshot);
        expect(text).toContain('weight="0.5"');
        expect(text).toContain('weight="2"');
        expect(text).toContain('idtype="integer"');
    });
});

describe("gexfExporter: check() loss notes (design 8.5)", () => {
    function build(directed = true): GraphBuilder {
        const builder = new GraphBuilder({ directed, weightDtype: "f64" });
        builder.addNode("a");
        builder.addNode("b");
        builder.addEdge("a", "b");
        return builder;
    }

    it("reports u32 / u8 dtypes and writes them as long / integer", async () => {
        const builder = build();
        const u32 = builder.declareNodeColumn({ name: "count", dtype: "u32" });
        const u8 = builder.declareNodeColumn({ name: "byte", dtype: "u8" });
        builder.setNodeValue(u32, 0, 4000000000);
        builder.setNodeValue(u8, 0, 200);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes)).toEqual([LOSS.DTYPE, LOSS.DTYPE]);
        expect(notes[0]).toMatchObject({ column: "count", count: 1 });
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('<attribute id="count" title="count" type="long"/>');
        expect(result.text).toContain('<attribute id="byte" title="byte" type="integer"/>');
        expect(result.snapshot.nodes.value("count", 0)).toBe(4000000000);
        expect(result.snapshot.nodes.get("count")?.dtype).toBe("f64");
        expect(result.snapshot.nodes.get("byte")?.dtype).toBe("i32");
    });

    it("flattens a stride column into a list and reports it", async () => {
        const builder = build();
        const vec = builder.declareNodeColumn({ name: "vec", dtype: "f64", components: 3 });
        builder.setNodeValue(vec, 1, [1, 2.5, -3]);
        const snapshot = builder.freeze();
        expect(noteCodes(gexfExporter.check(snapshot))).toEqual([LOSS.COMPONENTS]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('type="listdouble"');
        expect(result.text).toContain('<attvalue for="vec" value="[1, 2.5, -3]"/>');
        expect(result.snapshot.nodes.get("vec")?.dtype).toBe("list");
        expect(result.snapshot.nodes.typed("vec", "list")?.sliceOf(1)).toEqual([1, 2.5, -3]);
    });

    it("writes json as text and reports it", async () => {
        const builder = build();
        const json = builder.declareNodeColumn({ name: "meta", dtype: "json" });
        builder.setNodeValue(json, 0, { k: [1, "two"] });
        const snapshot = builder.freeze();
        expect(noteCodes(gexfExporter.check(snapshot))).toEqual([LOSS.JSON]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('value="{&quot;k&quot;:[1,&quot;two&quot;]}"');
        expect(result.snapshot.nodes.value("meta", 0)).toBe('{"k":[1,"two"]}');
    });

    it("reports graph attributes and foreign extension tables, which are not written", async () => {
        const builder = build();
        builder.setGraphValue("title", "hello");
        const table = builder.addExtensionTable("foreign", [{ name: "x", dtype: "i32" }]);
        builder.addExtensionRow(table, [1]);
        const snapshot = builder.freeze();
        expect(noteCodes(gexfExporter.check(snapshot)).sort()).toEqual([LOSS.EXTENSION_TABLE, LOSS.GRAPH_ATTRIBUTES]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.snapshot.graph.names()).toEqual([]);
        expect(result.snapshot.extensions.size).toBe(0);
        expectSameSnapshot(snapshot, result.snapshot, { extensions: false, ignoreColumns: ["title"] });
    });

    it("reports a temporal table without the design's columns and drops it", async () => {
        const builder = build();
        const table = builder.addExtensionTable("temporal:node:x", [{ name: "value", dtype: "i32" }]);
        builder.addExtensionRow(table, [1]);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes)).toEqual([GEXF_LOSS.TEMPORAL_TABLE_SHAPE]);
        expect(notes[0]).toMatchObject({ column: "temporal:node:x", count: 1 });
        const text = await gexfExporter.exportToString(snapshot);
        expect(text).not.toContain("attvalue");
    });

    it("writes an orphan temporal table as a dynamic attribute", async () => {
        const builder = build();
        const table = builder.addExtensionTable("temporal:node:heat", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64", role: "start" },
            { name: "end", dtype: "f64", role: "end" },
            { name: "value", dtype: "f32" },
        ]);
        builder.addExtensionRow(table, [1, 2, 3, 0.5]);
        builder.addExtensionRow(table, [1, 3, Infinity, 0.25]);
        const snapshot = builder.freeze();
        expect(gexfExporter.check(snapshot)).toEqual([]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('<attributes class="node" mode="dynamic">');
        expect(result.text).toContain('<attribute id="heat" title="heat" type="float"/>');
        expect(result.text).toContain('<attvalue for="heat" value="0.5" start="2" end="3"/>');
        expect(result.text).toContain('<attvalue for="heat" value="0.25" start="3"/>');
        expect(result.text).toContain('mode="dynamic" idtype="string" timeformat="double"');
        const heat = result.snapshot.extensions.get("temporal:node:heat");
        expect(heat?.rowCount).toBe(2);
        expect(heat?.value("value", 1)).toBe(0.25);
        expect(heat?.value("element", 1)).toBe(1);
        expect(result.snapshot.nodes.get("heat")?.meta.dynamic).toBe(true);
    });

    it("reports role columns of an unexpected shape and writes them as attributes", async () => {
        const builder = build();
        const start = builder.declareNodeColumn({ name: "start", dtype: "string", role: "start" });
        const color = builder.declareEdgeColumn({ name: "color", dtype: "f32", components: 2, role: "color" });
        builder.setNodeValue(start, 0, "yesterday");
        builder.setEdgeValue(color, 0, [1, 2]);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes)).toEqual([
            GEXF_LOSS.ROLE_SHAPE,
            GEXF_LOSS.ROLE_SHAPE,
            LOSS.COMPONENTS,
            GEXF_LOSS.ATTRIBUTE_RENAMED,
            GEXF_LOSS.ATTRIBUTE_RENAMED,
        ]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('<attribute id="start" title="start" type="string"/>');
        expect(result.snapshot.nodes.value("start#start", 0)).toBe("yesterday");
        expect(result.snapshot.edges.get("color#color")?.dtype).toBe("list");
    });

    it("reports roles GEXF has no field for and writes the column as a plain attribute", async () => {
        const builder = build();
        const community = builder.declareNodeColumn({ name: "community", dtype: "i32", role: "community" });
        builder.setNodeValue(community, 0, 3);
        builder.setNodeValue(community, 1, 4);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(notes).toEqual([expect.objectContaining({ code: LOSS.ROLE, column: "community" })]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.snapshot.nodes.get("community")?.meta.role).toBeNull();
        expect(compareSnapshots(snapshot, result.snapshot, { roles: false })).toEqual([]);
    });

    it("reports an attribute whose title the importer renames", async () => {
        const builder = build();
        const position = builder.declareNodeColumn({ name: "position", dtype: "string" });
        builder.setNodeValue(position, 0, "left");
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(notes).toEqual([expect.objectContaining({ code: GEXF_LOSS.ATTRIBUTE_RENAMED, column: "position" })]);
        expect(notes[0].message).toContain('"position#position"');
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.snapshot.nodes.value("position#position", 0)).toBe("left");
    });

    it("reports 1.2 losses: parallel edges, generated edge ids, kind, timestamps, list separators", async () => {
        const builder = build();
        builder.addEdge("a", "b");
        const kind = builder.declareEdgeColumn({ name: "kind", dtype: "dict", role: "kind" });
        builder.setEdgeValue(kind, 1, "again");
        const stamps = builder.declareNodeColumn({
            name: "timestamps",
            dtype: "list",
            itemDtype: "f64",
            role: "timestamps",
        });
        builder.setNodeValue(stamps, 0, [1, 2]);
        const tags = builder.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        builder.setNodeValue(tags, 0, ["a|b", "c"]);
        const snapshot = builder.freeze();
        expect(gexfExporter.check(snapshot)).toEqual([]);
        const notes = gexfExporter.check(snapshot, { version: "1.2" });
        expect(noteCodes(notes).sort()).toEqual(
            [
                LOSS.MULTI_EDGES,
                LOSS.EDGE_IDS_GENERATED,
                GEXF_LOSS.TIMESTAMP_AS_INTERVAL,
                GEXF_LOSS.KIND_DROPPED,
                GEXF_LOSS.LIST_SEPARATOR,
            ].sort(),
        );
        expect(notes.find((n) => n.code === GEXF_LOSS.LIST_SEPARATOR)).toMatchObject({ column: "tags", count: 1 });
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(result.text).toContain('<edge id="e0" source="a" target="b"/>');
        expect(result.text).toContain('<edge id="e1" source="a" target="b"/>');
        expect(result.text).toContain('<spell start="1" end="1"/>');
        expect(result.snapshot.edges.get("kind")).toBeNull();
        expect(result.snapshot.nodes.typed("tags", "list")?.sliceOf(0)).toEqual(["a", "b", "c"]);
        expect(result.snapshot.nodes.typed("spells", "list")?.sliceOf(0)).toHaveLength(2);
    });

    it("reports a 1.2 export of date and dateTime columns and writes them as doubles", async () => {
        const builder = build();
        const when = builder.declareNodeColumn({
            name: "when",
            dtype: "f64",
            origin: { format: "gexf", id: "w", type: "dateTime" },
        });
        builder.setNodeValue(when, 0, Date.UTC(2020, 0, 1, 12));
        const snapshot = builder.freeze();
        expect(gexfExporter.check(snapshot)).toEqual([]);
        expect(await exact(snapshot)).toContain('<attvalue for="w" value="2020-01-01T12:00:00Z"/>');
        expect(noteCodes(gexfExporter.check(snapshot, { version: "1.2" })).sort()).toEqual(
            [LOSS.EDGE_IDS_GENERATED, GEXF_LOSS.DECLARED_TYPE].sort(),
        );
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(result.text).toContain('<attribute id="w" title="when" type="double"/>');
        expect(result.snapshot.nodes.value("when", 0)).toBe(Date.UTC(2020, 0, 1, 12));
    });

    it("reports options and defaults the declared type cannot express", async () => {
        const builder = build();
        builder.declareNodeColumn({ name: "ratio", dtype: "f32", options: [true, 2] });
        builder.declareNodeColumn({ name: "count", dtype: "i32", default: 0.5 });
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes)).toEqual([GEXF_LOSS.VALUE_UNWRITABLE]);
        expect(notes[0]).toMatchObject({ column: "ratio" });
        const text = await gexfExporter.exportToString(snapshot);
        expect(text).toContain('<attribute id="ratio" title="ratio" type="float"/>');
        expect(text).toContain("<default>0.5</default>");
    });

    it("mangling ids is a no-op for GEXF and never reports", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addEdge("a b", "c<d>&\"'");
        builder.addEdge(-7, "");
        const snapshot = builder.freeze();
        expect(gexfExporter.check(snapshot, { sanitizeIds: "mangle" })).toEqual([]);
        const text = await exact(snapshot, { sanitizeIds: "mangle" });
        expect(text).toContain('<node id="c&lt;d&gt;&amp;&quot;\'"/>');
        expect(text).toContain('<node id="-7"/>');
        expect(text).toContain('<node id=""/>');
        expect(text).toContain('<node id="a b"/>');
    });

    it("reports ids that change type under the canonical rule: non-integer numbers read back as text, integer text as numbers", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addEdge(-1.5, 2);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(notes).toEqual([expect.objectContaining({ code: GEXF_LOSS.ID_TEXT_TYPE, count: 1 })]);
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.text).toContain('<node id="-1.5"/>');
        expect(result.snapshot.ids.toArray()).toEqual(["-1.5", 2]);
        const texts = new GraphBuilder({ directed: true });
        texts.addEdge("1", "2");
        const textNotes = gexfExporter.check(texts.freeze());
        expect(textNotes).toEqual([expect.objectContaining({ code: GEXF_LOSS.ID_TEXT_TYPE, count: 2 })]);
        expect((await roundTrip(texts.freeze(), gexfExporter, gexfImporter)).snapshot.ids.toArray()).toEqual([1, 2]);
    });
});

describe("gexfExporter: values", () => {
    it("writes typed defaults, options, dict dictionaries, booleans, lists and escapes text", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.addNode("n");
        builder.addNode("m");
        builder.addEdge("n", "m");
        const flag = builder.declareNodeColumn({ name: "flag", dtype: "bool", default: false });
        const level = builder.declareNodeColumn({ name: "level", dtype: "i32", options: [1, 2, 3], default: 2 });
        const cat = builder.declareNodeColumn({ name: "cat", dtype: "dict" });
        const note = builder.declareNodeColumn({ name: "note", dtype: "string", default: 'a <b> & "c"' });
        const tags = builder.declareNodeColumn({
            name: "tags",
            dtype: "list",
            itemDtype: "string",
            default: ["x", "y"],
        });
        const bits = builder.declareEdgeColumn({ name: "bits", dtype: "list", itemDtype: "bool" });
        const label = builder.declareEdgeColumn({ name: "label", dtype: "string", role: "label" });
        builder.setNodeValue(flag, 0, true);
        builder.setNodeValue(level, 0, 3);
        builder.setNodeValue(cat, 0, "red");
        builder.setNodeValue(cat, 1, "blue");
        builder.setNodeValue(note, 0, "line\nbreak\ttab");
        builder.setNodeValue(tags, 0, ["p, q", "r'\"s", ""]);
        builder.setEdgeValue(bits, 0, [true, false]);
        builder.setEdgeValue(label, 0, "n & m");
        const snapshot = builder.freeze();
        // the dict column declares no options: its dictionary becomes <options> and reads back declared
        expect(gexfExporter.check(snapshot).map((n) => [n.code, n.column])).toEqual([
            [GEXF_LOSS.OPTIONS_GAINED, "cat"],
        ]);
        const text = await exact(snapshot);
        expect(text).toContain("<default>false</default>");
        expect(text).toContain("<options>[1, 2, 3]</options>");
        expect(text).toContain("<default>2</default>");
        expect(text).toContain("<options>[red, blue]</options>");
        expect(text).toContain('<default>a &lt;b&gt; &amp; "c"</default>');
        expect(text).toContain("<default>[x, y]</default>");
        expect(text).toContain('<attvalue for="note" value="line&#10;break&#9;tab"/>');
        expect(text).toContain(
            '<attvalue for="tags" value="[&quot;p, q&quot;, &quot;r\'&quot;&quot;s&quot;, &quot;&quot;]"/>',
        );
        expect(text).toContain('<attribute id="bits" title="bits" type="listboolean"/>');
        expect(text).toContain('<attvalue for="bits" value="[true, false]"/>');
        expect(text).toContain('label="n &amp; m"');
    });

    it("omits z for a 2-d position, writes u8 colours and the shape uri", async () => {
        const doc = `<gexf version="1.3"><graph><nodes>
            <node id="a"><viz:position x="0.5" y="-1"/><viz:shape value="image" uri="http://x/y.png"/><viz:size value="1"/></node>
            </nodes></graph></gexf>`;
        const snapshot = await imported(doc);
        expect(snapshot.nodes.get("position")?.meta.extra).toEqual({ units: "file", sourceDims: 2 });
        const text = await exact(snapshot);
        expect(text).toContain('<viz:position x="0.5" y="-1"/>');
        expect(text).toContain('<viz:shape value="image" uri="http://x/y.png"/>');
        expect(text).toContain('<viz:size value="1"/>');
        // a 2-d source whose position column later received a z writes z again
        const withZ = GraphBuilder.from(snapshot);
        const handle = withZ.nodeColumn("position");
        withZ.setNodeValue(handle, 0, [0.5, -1, 2]);
        expect(await gexfExporter.exportToString(withZ.freeze())).toContain('<viz:position x="0.5" y="-1" z="2"/>');

        const builder = new GraphBuilder({ directed: true });
        builder.addNode(1);
        const color = builder.declareNodeColumn({ name: "color", dtype: "u8", components: 4, role: "color" });
        builder.setNodeValue(color, 0, [10, 20, 30, 128]);
        const bytes = builder.freeze();
        const written = await gexfExporter.exportToString(bytes);
        expect(written).toContain('<viz:color r="10" g="20" b="30" a="0.5019608"/>');
    });

    it("writes an empty graph and reads it back", async () => {
        const builder = new GraphBuilder({ directed: true });
        const snapshot = builder.freeze();
        const text = await exact(snapshot);
        expect(text).toContain('<nodes count="0">');
        expect(text).toContain('<edges count="0">');
        expect(text).not.toContain("<meta");
    });

    it("writes the timestamp representation and companion texts of element times", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" mode="dynamic" timeformat="dateTime" timerepresentation="timestamp">
            <nodes><node id="a" timestamp="2020-01-01T00:00:00+01:00"/><node id="b" timestamp="2020-06-01T00:00:00Z"/></nodes>
            <edges><edge source="a" target="b" timestamp="2020-03-01T00:00:00Z"/></edges></graph></gexf>`;
        const snapshot = await imported(doc);
        expect(snapshot.nodes.value("timestamp.text", 0)).toBe("2020-01-01T00:00:00+01:00");
        const text = await exact(snapshot);
        expect(text).toContain('timerepresentation="timestamp"');
        expect(text).toContain('<node id="a" timestamp="2020-01-01T00:00:00+01:00"/>');
        expect(text).toContain('<node id="b" timestamp="2020-06-01T00:00:00Z"/>');
        expect(text).toContain('<edge source="a" target="b" timestamp="2020-03-01T00:00:00Z"/>');
        const old = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(old.text).toContain('<node id="b" start="2020-06-01T00:00:00Z" end="2020-06-01T00:00:00Z"/>');
    });

    it("derives the timestamp representation from a timestamp-only snapshot", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addNode("a");
        const stamp = builder.declareNodeColumn({ name: "timestamp", dtype: "f64", role: "timestamp" });
        builder.setNodeValue(stamp, 0, 7);
        const snapshot = builder.freeze();
        const text = await exact(snapshot);
        expect(text).toContain('mode="dynamic" idtype="string" timeformat="double" timerepresentation="timestamp"');
        expect(text).toContain('<node id="a" timestamp="7"/>');
    });
});

describe("gexfExporter: less common paths", () => {
    it("writes timestamp-represented dynamic values, slice mode and companion texts of values", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" mode="slice" timeformat="dateTime" timerepresentation="timestamp">
            <attributes class="node" mode="dynamic"><attribute id="d" title="d" type="dateTime"/></attributes>
            <nodes><node id="a" timestamp="2020-01-01T00:00:00Z"><attvalues>
                <attvalue for="d" value="2020-05-05T00:00:00+02:00" timestamp="2020-01-01T00:00:00Z"/>
                <attvalue for="d" value="2020-06-06T00:00:00Z" start="2020-02-01T00:00:00Z" end="2020-03-01T00:00:00Z"/>
            </attvalues></node></nodes></graph></gexf>`;
        const snapshot = await imported(doc);
        const text = await exact(snapshot);
        expect(text).toContain('mode="slice"');
        expect(text).toContain(
            '<attvalue for="d" value="2020-05-05T00:00:00+02:00" timestamp="2020-01-01T00:00:00Z"/>',
        );
        expect(text).toContain(
            '<attvalue for="d" value="2020-06-06T00:00:00Z" start="2020-02-01T00:00:00Z" end="2020-03-01T00:00:00Z"/>',
        );
        const old = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(old.text).toContain('mode="dynamic"');
        // 1.2 has no dateTime type: the value is a double and its text companion does not apply
        expect(old.text).toContain('<attribute id="d" title="d" type="double"/>');
        expect(old.text).toContain(
            '<attvalue for="d" value="1588629600000" start="2020-01-01T00:00:00Z" end="2020-01-01T00:00:00Z"/>',
        );
        expect(old.snapshot.extensions.get("temporal:node:d")?.rowCount).toBe(2);
    });

    it("writes startopen on 1.2 dynamic values and node bounds", async () => {
        const doc = `<gexf version="1.2"><graph mode="dynamic" timeformat="integer">
            <attributes class="node" mode="dynamic"><attribute id="v" title="v" type="integer"/></attributes>
            <nodes><node id="a" startopen="1" endopen="2"><attvalues><attvalue for="v" value="1" startopen="1" end="2"/></attvalues></node></nodes></graph></gexf>`;
        const snapshot = await imported(doc);
        expect(snapshot.nodes.value("start", 0)).toBe(1);
        expect(snapshot.nodes.value("end", 0)).toBe(2);
        expect(snapshot.nodes.value("open", 0)).toBe(3);
        const text = await exact(snapshot, { version: "1.2" });
        expect(text).toContain('<node id="a" startopen="1" endopen="2">');
        expect(text).toContain('<attvalue for="v" value="1" startopen="1" end="2"/>');
    });

    it("reports json, u32 and list-shaped role columns that fall back to attributes", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addNode("a");
        const color = builder.declareNodeColumn({ name: "color", dtype: "json", role: "color" });
        const size = builder.declareNodeColumn({ name: "size", dtype: "u32", role: "size" });
        const thickness = builder.declareNodeColumn({ name: "thickness", dtype: "u32", role: "thickness" });
        const spells = builder.declareNodeColumn({
            name: "spells",
            dtype: "list",
            itemDtype: "string",
            role: "spells",
        });
        builder.setNodeValue(color, 0, { r: 1 });
        builder.setNodeValue(size, 0, 5);
        builder.setNodeValue(thickness, 0, 6);
        builder.setNodeValue(spells, 0, ["x"]);
        const snapshot = builder.freeze();
        const notes = gexfExporter.check(snapshot);
        expect(noteCodes(notes)).toEqual([
            GEXF_LOSS.ROLE_SHAPE,
            LOSS.JSON,
            GEXF_LOSS.VIZ_DTYPE,
            GEXF_LOSS.ROLE_SHAPE,
            LOSS.DTYPE,
            GEXF_LOSS.ROLE_SHAPE,
            GEXF_LOSS.ATTRIBUTE_RENAMED,
            GEXF_LOSS.ATTRIBUTE_RENAMED,
        ]);
        expect(notes[5].message).toContain("list of string x 1");
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.snapshot.nodes.value("color#color", 0)).toBe('{"r":1}');
        expect(result.snapshot.nodes.value("size", 0)).toBe(5);
        expect(result.snapshot.nodes.get("size")?.dtype).toBe("f32");
        // "thickness" is reserved for edges only, so the node attribute keeps its name
        expect(result.snapshot.nodes.value("thickness", 0)).toBe(6);
    });

    it("keeps attribute ids unique when origin ids collide", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.addNode("n");
        const a = builder.declareNodeColumn({ name: "a", dtype: "string", origin: { format: "gexf", id: "b" } });
        const b = builder.declareNodeColumn({ name: "b", dtype: "string", origin: { format: "gexf", id: "b" } });
        const c = builder.declareNodeColumn({ name: "b#2", dtype: "string", origin: { format: "gexf", id: "b" } });
        builder.setNodeValue(a, 0, "1");
        builder.setNodeValue(b, 0, "2");
        builder.setNodeValue(c, 0, "3");
        const snapshot = builder.freeze();
        const text = await gexfExporter.exportToString(snapshot);
        expect(text).toContain('<attribute id="b" title="a" type="string"/>');
        expect(text).toContain('<attribute id="b#2" title="b" type="string"/>');
        expect(text).toContain('<attribute id="b#2#2" title="b#2" type="string"/>');
        const result = await roundTrip(snapshot, gexfExporter, gexfImporter);
        expect(result.snapshot.nodes.value("a", 0)).toBe("1");
        expect(result.snapshot.nodes.value("b", 0)).toBe("2");
        expect(result.snapshot.nodes.value("b#2", 0)).toBe("3");
    });

    it("restores biginteger, bigdecimal, byte, short and char declarations from origin", async () => {
        const doc = `<gexf version="1.3"><graph>
            <attributes class="node" mode="static">
                <attribute id="0" title="bi" type="biginteger"/><attribute id="1" title="bd" type="bigdecimal"/>
                <attribute id="2" title="by" type="byte"/><attribute id="3" title="sh" type="short"/><attribute id="4" title="ch" type="char"/>
                <attribute id="5" title="lb" type="listboolean"/><attribute id="6" title="ld" type="listdouble"/>
            </attributes>
            <nodes><node id="a"><attvalues>
                <attvalue for="0" value="123456789012345678901234567890"/><attvalue for="1" value="0.1000000000000000000001"/>
                <attvalue for="2" value="7"/><attvalue for="3" value="-3"/><attvalue for="4" value="q"/>
                <attvalue for="5" value="[true, false]"/><attvalue for="6" value="[0.5, 1e21]"/>
            </attvalues></node></nodes></graph></gexf>`;
        const snapshot = await imported(doc);
        expect(snapshot.nodes.get("bi")?.dtype).toBe("string");
        expect(snapshot.nodes.get("by")?.dtype).toBe("i32");
        const text = await exact(snapshot);
        for (const decl of ["biginteger", "bigdecimal", "byte", "short", "char", "listboolean", "listdouble"]) {
            expect(text).toContain(`type="${decl}"`);
        }
        expect(text).toContain('value="123456789012345678901234567890"');
        expect(text).toContain('value="[0.5, 1e+21]"');
        const notes = gexfExporter.check(snapshot, { version: "1.2" });
        expect(noteCodes(notes)).toEqual([
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
            GEXF_LOSS.DECLARED_TYPE,
        ]);
        const old = await roundTrip(snapshot, gexfExporter, gexfImporter, { exportOptions: { version: "1.2" } });
        expect(old.report.errorCount).toBe(0);
        expect(old.text).toContain('<attribute id="0" title="bi" type="string"/>');
        expect(old.text).toContain('<attribute id="2" title="by" type="integer"/>');
        expect(old.snapshot.nodes.value("by", 0)).toBe(7);
    });
});
