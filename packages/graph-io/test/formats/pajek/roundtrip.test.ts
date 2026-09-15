import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { LOSS } from "../../../src/common/export.js";
import { PAJEK_LOSS, pajekExporter } from "../../../src/formats/pajek/exporter.js";
import { PAJEK_ISSUE, pajekImporter, type PajekImportOptions } from "../../../src/formats/pajek/importer.js";
import { type CommonImportOptions } from "../../../src/types.js";
import { corpusFiles, readCorpusText } from "../../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip } from "../../helpers/roundtrip.js";

type ImportOpts = PajekImportOptions & CommonImportOptions;

async function fromPajek(text: string, options?: ImportOpts): Promise<GraphSnapshot> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await pajekImporter.import(text, builder, options);
    return builder.freeze();
}

/** The re-import options that restore a snapshot's ids: 0-based identity ids come back through nodeIdFrom "index". */
function reimportOptions(snapshot: GraphSnapshot): ImportOpts {
    return snapshot.ids.kind === "identity" && snapshot.ids.offset === 0 ? { nodeIdFrom: "index" } : {};
}

describe("pajek round trips: the corpus", () => {
    for (const entry of corpusFiles("pajek")) {
        it(`${entry.path}: import -> export -> import is equal, twice`, async () => {
            const first = await fromPajek(readCorpusText("pajek", entry.path));
            const options = reimportOptions(first);
            const once = await roundTrip(first, pajekExporter, pajekImporter, { importOptions: options });
            expect(once.report.errorCount).toBe(0);
            expect(once.notes.map((n) => n.code)).toEqual(first.ids.offset === 0 ? [LOSS.ID_RENUMBERED] : []);
            expectSameSnapshot(first, once.snapshot, { allowExtraColumns: false });
            const twice = await roundTrip(once.snapshot, pajekExporter, pajekImporter, { importOptions: options });
            expectSameSnapshot(first, twice.snapshot, { allowExtraColumns: false });
            expect(twice.text).toBe(once.text);
        });
    }

    it("simple.net re-exports byte-for-byte except the z coordinate spelling", async () => {
        const first = await fromPajek(readCorpusText("pajek", "simple.net"));
        const text = await pajekExporter.exportToString(first);
        expect(text).toBe(readCorpusText("pajek", "simple.net").replace(/([0-9])\.0(?=\s|$)/g, "$1"));
    });
});

describe("pajek round trips: features", () => {
    it("keeps labels, coordinates, shapes, typed parameters and intervals on vertices", async () => {
        const text = [
            "*Vertices 4",
            '1 "Node A" 0.1 0.2 0.3 box ic Red bc Black s_size 5 x_fact 1.5 fixed true [1-5,7-*]',
            "2 plain 0.4 0.5 0.6 ellipse ic LightBlue",
            '3 "" 0.7 0.8 0.9',
            "4 last",
            "*Arcs",
            "1 2 1.5",
            "3 4",
        ].join("\n");
        const first = await fromPajek(text);
        const { snapshot, notes, report } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(notes).toEqual([]);
        expect(report.issues).toEqual([]);
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false, originType: true });
        expect(snapshot.nodes.requireTyped("s_size", "i32").value(0)).toBe(5);
        expect(snapshot.nodes.requireTyped("x_fact", "f64").value(0)).toBe(1.5);
        expect(snapshot.nodes.requireTyped("fixed", "bool").value(0)).toBe(true);
        expect(snapshot.nodes.requireTyped("shape", "dict").value(1)).toBe("ellipse");
        expect(snapshot.nodes.requireTyped("label", "string").value(2)).toBe("");
    });

    it("keeps every direction combination, parameters, relations and intervals on lines", async () => {
        const text = [
            "*Vertices 4",
            "*Edges",
            "1 2 2.5",
            "2 2",
            '*Arcs :1 "likes"',
            '2 3 c Blue l "a to b" w 3 [1-2]',
            "3 1 0.5",
            "*Arcs :2 hates",
            "4 4",
            "*Edges",
            "1 3",
            "*Arcs",
            "1 4 7",
        ].join("\n");
        const first = await fromPajek(text);
        expect(first.directed).toBe(true);
        expect(first.edges.byRole("pair")).not.toBeNull();
        const { snapshot, notes, report, text: exported } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(notes).toEqual([]);
        expect(report.issues).toEqual([]);
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
        expect(exported.split("\n").filter((l) => l.startsWith("*"))).toEqual([
            "*Vertices 4",
            "*Edges",
            "*Arcs :1 likes",
            "*Arcs :2 hates",
            "*Edges",
            "*Arcs",
        ]);
    });

    it("keeps explicit and defaulted weights apart and f64 weights exact", async () => {
        const text = "*Vertices 3\n*Arcs\n1 2\n2 3 0.1\n3 1 16777217\n1 1 1\n";
        const first = await fromPajek(text);
        const shadow = first.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.isSet(0)).toBe(false);
        expect(shadow?.value(1)).toBe(0.1);
        expect(shadow?.value(2)).toBe(16777217);
        const { snapshot, text: exported } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(exported).toBe("*Vertices 3\n1\n2\n3\n*Arcs\n1 2\n2 3 0.1\n3 1 16777217\n1 1 1\n");
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
        expect(compareSnapshots(first, snapshot, { weightExplicitness: true })).toEqual([]);
    });

    it("keeps an unweighted graph unweighted and a line value column as a parameter", async () => {
        const first = await fromPajek("*Vertices 2\n*Edges\n1 2 4\n2 1\n", { weightFrom: null });
        expect(first.flags.weighted).toBe(false);
        const { snapshot, text } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(text).toBe("*Vertices 2\n1\n2\n*Edges\n1 2 value 4\n2 1\n");
        expect(snapshot.flags.weighted).toBe(false);
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("string ids survive as labels and come back through nodeIdFrom label", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.addEdge("alice", "bob", 2);
        builder.addEdge("bob", "carol dane");
        builder.addEdge("carol dane", "alice");
        const first = builder.freeze();
        const plain = await roundTrip(first, pajekExporter, pajekImporter);
        expect(plain.notes.map((n) => n.code)).toEqual([LOSS.ID_RENUMBERED]);
        expect(plain.snapshot.ids.toArray()).toEqual([1, 2, 3]);
        const byLabel = await roundTrip(first, pajekExporter, pajekImporter, {
            importOptions: { nodeIdFrom: "label" },
        });
        expect(byLabel.snapshot.ids.toArray()).toEqual(["alice", "bob", "carol dane"]);
        const diffs = compareSnapshots(first, byLabel.snapshot);
        expect(diffs, describeDiffs(diffs)).toEqual([]);
    });

    it("a 0-based corpus file comes back through nodeIdFrom index with the same ids and labels", async () => {
        const first = await fromPajek(readCorpusText("pajek", "dolphins.net"));
        const { snapshot, report } = await roundTrip(first, pajekExporter, pajekImporter, {
            importOptions: { nodeIdFrom: "index" },
        });
        expect(report.issues).toEqual([]);
        expect(snapshot.ids.toArray()).toEqual(first.ids.toArray());
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("a two-mode network and its name survive", async () => {
        const first = await fromPajek("*Network two mode\n*Vertices 3 2\n1 a\n2 b\n3 c\n*Edges\n1 3\n2 3\n");
        const { snapshot, text } = await roundTrip(first, pajekExporter, pajekImporter, {
            exportOptions: { networkHeader: true },
        });
        expect(text.startsWith("*Network two mode\n*Vertices 3 2\n")).toBe(true);
        expect(snapshot.meta.name).toBe("two mode");
        expect(snapshot.meta.extra).toEqual({ pajek: { firstMode: 2 } });
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("*Edgeslist and *Matrix input re-exports as plain lines with the same topology", async () => {
        const first = await fromPajek("*Vertices 3\n*Arcslist\n1 2 3\n*Matrix\n0 0 0\n0 0 2\n1 0 0\n");
        const { snapshot, text } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(text).toBe("*Vertices 3\n1\n2\n3\n*Arcs\n1 2\n1 3\n2 3 2\n3 1 1\n");
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("a mixed file whose sink was expanded in place round-trips through the pair column", async () => {
        const text = "*Vertices 3\n*Edges\n1 2\n2 3\n*Arcs\n3 1\n";
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        await pajekImporter.import(text, builder);
        const first = builder.freeze();
        expect(first.directed).toBe(true);
        const { snapshot, text: exported } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(exported).toBe("*Vertices 3\n1\n2\n3\n*Edges\n1 2\n2 3\n*Arcs\n3 1\n");
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("the label written for an unlabelled vertex with coordinates is its number", async () => {
        const first = await fromPajek("*Vertices 2\n1 1 0.5 0.5\n2 2 0.1 0.1\n*Arcs\n1 2\n");
        const { snapshot, text } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(text).toBe("*Vertices 2\n1 1 0.5 0.5\n2 2 0.1 0.1\n*Arcs\n1 2\n");
        expectSameSnapshot(first, snapshot, { allowExtraColumns: false });
    });

    it("reports what a lossy export changed on the way back", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareNodeColumn({ name: "mass", dtype: "f32" });
        builder.declareNodeColumn({ name: "kind", dtype: "dict" });
        builder.addNodeRecord(1, { mass: 2, kind: "a" });
        builder.addNodeRecord(2, { mass: 0.5, kind: "b" });
        builder.addEdge(1, 2);
        const first = builder.freeze();
        const { snapshot, notes } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(notes.map((n) => n.code)).toEqual([LOSS.DTYPE, LOSS.DTYPE, PAJEK_LOSS.LABEL_GAINED]);
        expect(snapshot.nodes.get("mass")?.dtype).toBe("f64");
        expect(snapshot.nodes.get("kind")?.dtype).toBe("string");
        const diffs = compareSnapshots(first, snapshot);
        expect(diffs.map((d) => d.path)).toEqual(["nodes.mass.dtype", "nodes.kind.dtype"]);
        expect(compareSnapshots(first, snapshot, { dtypes: false })).toEqual([]);
    });

    it("re-imports under a zero error budget without issues for every corpus file", async () => {
        for (const entry of corpusFiles("pajek")) {
            const first = await fromPajek(readCorpusText("pajek", entry.path));
            const { report } = await roundTrip(first, pajekExporter, pajekImporter, {
                importOptions: { ...reimportOptions(first), errorLimit: 0 },
            });
            expect(
                report.issues.map((i) => i.code),
                entry.path,
            ).toEqual([]);
            expect(report.issues.map((i) => i.code)).not.toContain(PAJEK_ISSUE.ZERO_BASED);
        }
    });
});

describe("pajek round trips: sanitizeIds mangle keeps the original ids (design 8.5)", () => {
    function labelled(): GraphSnapshot {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
        b.addNode("x");
        b.addNode("y");
        b.addNode(3);
        b.addNode("two words");
        b.addEdge("x", "y");
        b.addEdge(3, "two words");
        b.setNodeValue("label", 0, "Ex");
        b.setNodeValue("label", 3, "Two");
        return b.freeze();
    }

    it('under "error" (the default) a labelled node loses its id, which W_ID_RENUMBERED says', async () => {
        const first = labelled();
        const { snapshot, notes, text } = await roundTrip(first, pajekExporter, pajekImporter);
        expect(notes.map((n) => n.code)).toEqual([LOSS.ID_RENUMBERED]);
        expect(notes[0].message).toContain("loses its id");
        expect(text).not.toContain("graphty_originalId");
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('under "mangle" every renumbered vertex carries graphty_originalId and restoreMangledIds (the default) brings the ids back', async () => {
        const first = labelled();
        const exportOptions = { sanitizeIds: "mangle" as const };
        const { snapshot, notes, text, report } = await roundTrip(first, pajekExporter, pajekImporter, {
            exportOptions,
        });
        expect(notes.map((n) => n.code)).toEqual([LOSS.ID_RENUMBERED]);
        expect(notes[0].message).toContain("originalId");
        // vertex 3 has the id 3: its own 1-based index, so it carries no parameter
        expect(text).toBe(
            [
                "*Vertices 4",
                "1 Ex graphty_originalId x",
                "2 y graphty_originalId y",
                "3",
                '4 Two graphty_originalId "two words"',
                "*Arcs",
                "1 2",
                "3 4",
                "",
            ].join("\n"),
        );
        expect(report.issues).toEqual([]);
        expect(snapshot.ids.toArray()).toEqual(["x", "y", 3, "two words"]);
        expect(snapshot.edgeList().src).toEqual(first.edgeList().src);
        expect(snapshot.edgeList().dst).toEqual(first.edgeList().dst);
        expect(snapshot.nodes.names()).toEqual(["label"]);
        // the vertex line of y carries a parameter, so it needs a label: the id text (the note says so)
        expect(notes[0].message).toContain("as labels of the nodes without a label value");
        const label = snapshot.nodes.requireTyped("label", "string");
        expect([0, 1, 2, 3].map((i) => (label.isSet(i) ? label.value(i) : undefined))).toEqual([
            "Ex",
            "y",
            undefined,
            "Two",
        ]);
    });

    it("restoreMangledIds false keeps the parameter as a plain column and the numbered ids", async () => {
        const first = labelled();
        const { snapshot } = await roundTrip(first, pajekExporter, pajekImporter, {
            exportOptions: { sanitizeIds: "mangle" },
            importOptions: { restoreMangledIds: false },
        });
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4]);
        const column = snapshot.nodes.requireTyped("graphty_originalId", "string");
        expect([0, 1, 2, 3].map((i) => (column.isSet(i) ? column.value(i) : null))).toEqual([
            "x",
            "y",
            null,
            "two words",
        ]);
    });

    it('a string id of integer text reads back as a number under ids "canonical", which check() predicts (W_ID_TEXT_TYPE)', async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addNode("7");
        b.addNode("8");
        b.addEdge("7", "8");
        const first = b.freeze();
        const exportOptions = { sanitizeIds: "mangle" as const };
        const { snapshot, notes } = await roundTrip(first, pajekExporter, pajekImporter, { exportOptions });
        expect(notes.map((n) => n.code).sort()).toEqual([LOSS.ID_RENUMBERED, LOSS.ID_TEXT_TYPE].sort());
        expect(snapshot.ids.toArray()).toEqual([7, 8]);
        const kept = await roundTrip(first, pajekExporter, pajekImporter, {
            exportOptions,
            importOptions: { ids: "string" },
        });
        expect(kept.snapshot.ids.toArray()).toEqual(["7", "8"]);
    });

    it('a user column named graphty_originalId is reserved under "mangle" (W_PAJEK_KEY_DROPPED) and written otherwise', async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "graphty_originalId", dtype: "string" });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("graphty_originalId", 0, "mine");
        const first = b.freeze();
        const mangled = await roundTrip(first, pajekExporter, pajekImporter, {
            exportOptions: { sanitizeIds: "mangle" },
        });
        expect(mangled.notes.map((n) => [n.code, n.column])).toEqual([
            [PAJEK_LOSS.KEY_DROPPED, "graphty_originalId"],
            [LOSS.ID_RENUMBERED, null],
        ]);
        expect(mangled.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(mangled.snapshot.nodes.names()).toEqual(["label"]);
        // without "mangle" the column is an ordinary parameter, and the importer restores it as the id
        // (it is the exporter's key by name); restoreMangledIds: false keeps it as the column it was
        const plain = await roundTrip(first, pajekExporter, pajekImporter, {
            importOptions: { restoreMangledIds: false },
        });
        expect(plain.snapshot.nodes.requireTyped("graphty_originalId", "string").value(0)).toBe("mine");
    });

    it("two vertices with the same graphty_originalId become one node with a merge warning", async () => {
        const text = [
            "*Vertices 3",
            "1 a graphty_originalId same",
            "2 b graphty_originalId same",
            "3 c",
            "*Arcs",
            "1 3",
        ].join("\n");
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await pajekImporter.import(text, builder);
        expect(report.issues.map((i) => i.code)).toEqual([PAJEK_ISSUE.ORIGINAL_ID_MERGED]);
        const snapshot = builder.freeze();
        expect(snapshot.ids.toArray()).toEqual(["same", 3]);
        expect(snapshot.edgeCount).toBe(1);
    });

    it("a restored file keeps the vertex-number index order: only the second vertex carries the parameter", async () => {
        // the exporter writes the parameter on renumbered vertices only, so the first line has none
        const text = ["*Vertices 3", "1 a", "2 b graphty_originalId x", "3 c", "*Arcs", "1 2", "3 1"].join("\n");
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await pajekImporter.import(text, builder);
        expect(report.issues).toEqual([]);
        const snapshot = builder.freeze();
        expect(snapshot.ids.toArray()).toEqual([1, "x", 3]);
        expect([...snapshot.edgeList().src]).toEqual([0, 2]);
        expect([...snapshot.edgeList().dst]).toEqual([1, 0]);
    });

    it("out-of-order vertex lines keep the number order; a parameter on a vertex created earlier is reported", async () => {
        const text = [
            "*Vertices 3",
            "3 c graphty_originalId z",
            "1 a graphty_originalId x",
            "2 b",
            "*Arcs",
            "1 2",
        ].join("\n");
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await pajekImporter.import(text, builder);
        // vertex 3's line created vertices 1 and 2 under their numbers first (index = number order)
        expect(report.issues.map((i) => [i.code, i.element])).toEqual([[PAJEK_ISSUE.ORIGINAL_ID_UNRESTORED, "1"]]);
        const snapshot = builder.freeze();
        expect(snapshot.ids.toArray()).toEqual([1, 2, "z"]);
        expect(snapshot.nodes.requireTyped("label", "string").value(0)).toBe("a");
    });

    it("an ordinary file's id map is still identity with offset 1 under restoreMangledIds (the default)", async () => {
        const text = ["*Vertices 3", "1 a", "2 b", "3 c", "*Arcs", "1 2"].join("\n");
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await pajekImporter.import(text, builder);
        const snapshot = builder.freeze();
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(1);
        const out = ["*Vertices 3", "3 c", "1 a", "2 b", "*Arcs", "1 2"].join("\n");
        const b2 = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await pajekImporter.import(out, b2);
        expect(b2.freeze().ids.toArray()).toEqual([1, 2, 3]);
    });

    it("an id holding a double quote cannot be written as a label or a parameter (E_PAJEK_TEXT, export() throws)", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addNode('say "hi"');
        b.addNode("b");
        b.addEdge('say "hi"', "b");
        const first = b.freeze();
        for (const exportOptions of [{}, { sanitizeIds: "mangle" as const }]) {
            const notes = pajekExporter.check(first, exportOptions);
            expect(notes.map((n) => n.code)).toContain(PAJEK_LOSS.TEXT);
            await expect(pajekExporter.exportToString(first, exportOptions)).rejects.toMatchObject({
                code: "E_UNSUPPORTED",
            });
        }
    });
});
