import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { neo4jExporter, UNDIRECTED_LOSS } from "../../../src/formats/neo4j/exporter.js";
import { neo4jImporter, type Neo4jImportOptions } from "../../../src/formats/neo4j/importer.js";
import { type CommonImportOptions } from "../../../src/types.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";
import { type Neo4jCorpusFile, neo4jFiles, neo4jText } from "./fixtures.js";

async function load(text: string, options?: Neo4jImportOptions & CommonImportOptions): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await neo4jImporter.import(text, builder, options);
    return builder.freeze();
}

function optionsOf(entry: Neo4jCorpusFile): Neo4jImportOptions & CommonImportOptions {
    return entry.options ?? {};
}

const EXACT = { allowExtraColumns: false, originType: true } as const;

describe("neo4j round trips (design 16.5)", () => {
    for (const entry of neo4jFiles()) {
        it(`${entry.path}: export then re-import is exact`, async () => {
            const original = await load(neo4jText(entry.path), optionsOf(entry));
            expect(neo4jExporter.check(original)).toEqual([]);
            const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
            expect(trip.notes).toEqual([]);
            expect(trip.report.errorCount).toBe(0);
            expectSameSnapshot(original, trip.snapshot, EXACT);
            // a second trip through the written text is byte-stable
            const again = await neo4jExporter.exportToString(trip.snapshot);
            expect(again).toBe(trip.text);
        });

        const paired = entry.with;
        if (paired !== undefined) {
            it(`${entry.path} paired with its relationships: exact through one file and through two`, async () => {
                const original = await load(neo4jText(entry.path), {
                    ...optionsOf(entry),
                    relationships: paired.relationships.map((name) => neo4jText(name)),
                });
                const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
                expectSameSnapshot(original, trip.snapshot, EXACT);

                const nodes = await neo4jExporter.exportToString(original, { part: "nodes" });
                const relationships = await neo4jExporter.exportToString(original, { part: "relationships" });
                expect(nodes + relationships).toBe(trip.text);
                const split = await load(nodes, { relationships });
                expectSameSnapshot(original, split, EXACT);
            });
        }
    }

    it("keeps the tab-delimited fixture exact under its own syntax", async () => {
        const original = await load(neo4jText("crlf-tabs.tsv"), { delimiter: "\t" });
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter, {
            exportOptions: { delimiter: "\t", arrayDelimiter: "|" },
            importOptions: { delimiter: "\t", arrayDelimiter: "|" },
        });
        expect(trip.text.startsWith("nodeId:ID\t:LABEL\tname\tn:int\n")).toBe(true);
        expectSameSnapshot(original, trip.snapshot, EXACT);
    });

    it("keeps explicit and defaulted weights apart, with f32 and f64 values", async () => {
        const original = await load(":START_ID,:END_ID,weight:double\n1,2,0.1\n2,3,\n3,1,16777217\n4,4,0\n");
        expect(original.edges.byRole("weight")?.dtype).toBe("f64");
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
        expectSameSnapshot(original, trip.snapshot, EXACT);
        expect(trip.snapshot.edges.byRole("weight")?.value(2)).toBe(16777217);
    });

    it("keeps numeric, string and mixed-text ids", async () => {
        const original = await load(':ID\n1\n007\n-3\nx y\n"a,b"\n9007199254740993\n');
        expect(original.ids.toArray()).toEqual([1, "007", -3, "x y", "a,b", "9007199254740993"]);
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
        expectSameSnapshot(original, trip.snapshot, EXACT);
    });

    it("keeps a snapshot built by hand, with dict columns compared by value", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "a");
        b.addEdge("a", "a", 0.5);
        b.declareNodeColumn({ name: "kind", dtype: "dict" });
        b.declareNodeColumn({ name: "n", dtype: "i32" });
        b.declareNodeColumn({ name: "flag", dtype: "bool" });
        b.declareEdgeColumn({ name: "note", dtype: "string" });
        b.setNodeValue("kind", 0, "x");
        b.setNodeValue("kind", 1, "y");
        b.setNodeValue("n", 0, 1);
        b.setNodeValue("flag", 1, true);
        b.setEdgeValue("note", 0, "first");
        b.setEdgeValue("note", 2, "");
        const original = b.freeze();
        // the dict column reads back as string, which check() announces
        expect(neo4jExporter.check(original).map((n) => [n.code, n.column])).toEqual([[LOSS.DTYPE, "kind"]]);
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
        expect(trip.snapshot.nodes.require("kind").dtype).toBe("string");
        expectSameSnapshot(original, trip.snapshot, { allowExtraColumns: false, dtypes: false });
        const diffs = compareSnapshots(original, trip.snapshot, { allowExtraColumns: false });
        expect(describeDiffs(diffs)).toContain("nodes.kind.dtype");
    });

    it('re-imports an undirected snapshot as undirected under onMixedDirection "undirected"', async () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("a", "b", 3);
        b.addEdge("b", "c");
        b.addEdge("c", "c");
        const original = b.freeze();
        const notes = neo4jExporter.check(original);
        expect(notes.map((n) => n.code)).toEqual([UNDIRECTED_LOSS]);
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter, {
            importOptions: { onMixedDirection: "undirected" },
        });
        expect(trip.snapshot.directed).toBe(false);
        expectSameSnapshot(original, trip.snapshot, EXACT);
        const asDirected = await roundTrip(original, neo4jExporter, neo4jImporter);
        expect(asDirected.snapshot.directed).toBe(true);
        expect(asDirected.snapshot.edgeCount).toBe(3);
    });

    it('folds expanded pairs back to one directed relationship under onMixedDirection "directed"', async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
        r.setHeader(true);
        r.addEdge("a", "b", "directed", 2);
        r.addEdge("b", "c", "undirected", 3);
        const original = b.freeze();
        expect(original.edgeCount).toBe(3);
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter, {
            exportOptions: { onMixedDirection: "directed" },
        });
        expect(trip.notes.map((n) => n.code)).toEqual([LOSS.MIXED_DIRECTION, UNDIRECTED_LOSS]);
        expect(trip.snapshot.edgeCount).toBe(2);
        const list = trip.snapshot.edgeList();
        expect([...list.src]).toEqual([0, 1]);
        expect([...list.dst]).toEqual([1, 2]);
        expect([...(list.weights ?? [])]).toEqual([2, 3]);
        expect(trip.snapshot.edges.byRole("pair")).toBeNull();
    });

    it("keeps temporal companions, points and lists through a second trip", async () => {
        const text =
            "id:ID,when:datetime,at:time,where:point,tags:string[],days:date[]\n" +
            "1,2020-01-02T03:04:05+01:00,10:00:00+02:00,\"{x:1, y:2, crs:'cartesian'}\",a;b,2020-01-01;2020-02-02\n" +
            "2,2020-01-02T02:04:05Z,08:00:00Z,,,\n";
        const original = await load(text);
        expect(original.nodes.value("when.text", 0)).toBe("2020-01-02T03:04:05+01:00");
        expect(original.nodes.value("at.text", 0)).toBe("10:00:00+02:00");
        expect(original.nodes.isSet("when.text", 1)).toBe(false);
        const trip = await roundTrip(original, neo4jExporter, neo4jImporter);
        expectSameSnapshot(original, trip.snapshot, EXACT);
        expect(trip.text).toContain("2020-01-02T03:04:05+01:00");
        expect(trip.text).toContain("10:00:00+02:00");
        expect(trip.text).toContain("2020-01-02T02:04:05Z");
    });
});
