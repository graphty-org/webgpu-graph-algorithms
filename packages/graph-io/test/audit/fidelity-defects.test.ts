/**
 * Fidelity audit, focused reproductions: one deliberately failing test per defect the corpus
 * matrix (fidelity-matrix.test.ts) and the independent readers (fidelity-independent-readers.test.ts)
 * surfaced, reduced to the smallest snapshot that shows it. Each test states the design rule it
 * checks (8.5: check() predicts every loss; 5.1: the text grammar and the timeText companions;
 * 4.1: the canonical id rule; 16.5: equal snapshots after a round trip) and fails with the column
 * and the values.
 */

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { csvExporter } from "../../src/formats/csv/exporter.js";
import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotExporter } from "../../src/formats/dot/exporter.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfExporter } from "../../src/formats/gexf/exporter.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { gmlExporter } from "../../src/formats/gml/exporter.js";
import { gmlImporter } from "../../src/formats/gml/importer.js";
import { graphmlExporter } from "../../src/formats/graphml/exporter.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { jsonExporter } from "../../src/formats/json/exporter.js";
import { jsonImporter } from "../../src/formats/json/importer.js";
import { neo4jExporter } from "../../src/formats/neo4j/exporter.js";
import { neo4jImporter } from "../../src/formats/neo4j/importer.js";
import { pajekExporter } from "../../src/formats/pajek/exporter.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import {
    type CommonExportOptions,
    type CommonImportOptions,
    type GraphExporter,
    type GraphImporter,
    type ImportReport,
    type LossNote,
} from "../../src/types.js";

type AnyExportOptions = Record<string, unknown> & CommonExportOptions;
type AnyImportOptions = Record<string, unknown> & CommonImportOptions;

interface Trip {
    readonly notes: readonly LossNote[];
    readonly text: string;
    readonly snapshot: GraphSnapshot;
    readonly report: ImportReport;
}

async function trip(
    snapshot: GraphSnapshot,
    exporter: GraphExporter<AnyExportOptions>,
    importer: GraphImporter<AnyImportOptions>,
    exportOptions: AnyExportOptions = {},
    importOptions: AnyImportOptions = {},
): Promise<Trip> {
    const notes = exporter.check(snapshot, exportOptions);
    const text = await exporter.exportToString(snapshot, exportOptions);
    const builder = new GraphBuilder({ directed: snapshot.directed, weightDtype: "f64" });
    const report = await importer.import(text, builder, importOptions);
    return { notes, text, snapshot: builder.freeze(), report };
}

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => (n.column === null ? n.code : `${n.code}:${n.column}`));
}

const c = String.fromCharCode;

const GEXF: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "gexf",
    gexfExporter as GraphExporter<AnyExportOptions>,
    gexfImporter as GraphImporter<AnyImportOptions>,
];
const GRAPHML: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "graphml",
    graphmlExporter as GraphExporter<AnyExportOptions>,
    graphmlImporter as GraphImporter<AnyImportOptions>,
];
const GML: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "gml",
    gmlExporter as GraphExporter<AnyExportOptions>,
    gmlImporter as GraphImporter<AnyImportOptions>,
];
const DOT: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "dot",
    dotExporter as GraphExporter<AnyExportOptions>,
    dotImporter as GraphImporter<AnyImportOptions>,
];
const JSON_: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "json",
    jsonExporter as GraphExporter<AnyExportOptions>,
    jsonImporter as GraphImporter<AnyImportOptions>,
];
const NEO4J: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "neo4j",
    neo4jExporter as GraphExporter<AnyExportOptions>,
    neo4jImporter as GraphImporter<AnyImportOptions>,
];
const PAJEK: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "pajek",
    pajekExporter as GraphExporter<AnyExportOptions>,
    pajekImporter as GraphImporter<AnyImportOptions>,
];
const CSV: [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>] = [
    "csv",
    csvExporter as GraphExporter<AnyExportOptions>,
    csvImporter as GraphImporter<AnyImportOptions>,
];

describe("D1 CSV exporter (edge table): isolated nodes and node order are lost without a note", () => {
    it("drops a node with no edges, which check() predicts (an edge table cannot carry it: design 8.5 lists the loss, table nodes keeps it)", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addNode("z");
        b.addNode("a");
        b.addNode("lonely");
        b.addEdge("a", "z");
        const s = b.freeze();
        const t = await trip(
            s,
            ...(CSV.slice(1) as [GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>]),
        );
        expect(t.report.errorCount).toBe(0);
        // design 8.5: check() predicts every loss; the node "lonely" and the node order z, a are gone
        expect(codes(t.notes)).toEqual(["W_CSV_ISOLATED_NODES", "W_CSV_NODE_ORDER"]);
        expect(t.snapshot.ids.toArray()).toEqual(["a", "z"]);
        const nodes = await trip(s, CSV[1], CSV[2], { table: "nodes" }, { table: "nodes" });
        expect(nodes.snapshot.ids.toArray()).toEqual(["z", "a", "lonely"]);
    });

    it("or at least says so: an isolated node or a reordered node table needs a loss note", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addNode("z");
        b.addNode("a");
        b.addNode("lonely");
        b.addEdge("a", "z");
        const s = b.freeze();
        expect(codes(csvExporter.check(s))).not.toEqual([]);
    });
});

describe("D2 Pajek exporter: a `shape` column with values outside the shape keywords", () => {
    it("is written half as the shape slot and half as a `shape` parameter, which its own importer refuses", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "shape", dtype: "string" });
        b.addNode(1);
        b.addNode(2);
        b.addEdge(1, 2);
        b.setNodeValue("shape", 0, "hexagon");
        b.setNodeValue("shape", 1, "box");
        const s = b.freeze();
        const notes = pajekExporter.check(s);
        const text = await pajekExporter.exportToString(s);
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await pajekImporter.import(text, builder);
        expect(
            report.errorCount,
            `re-import of\n${text}\nissues: ${JSON.stringify(report.issues)}; check() said ${JSON.stringify(codes(notes))}`,
        ).toBe(0);
        const back = builder.freeze();
        expect([0, 1].map((i) => back.nodes.require("shape").value(i))).toEqual(["hexagon", "box"]);
    });
});

describe("D3 Pajek exporter: W_ID_RENUMBERED claims the ids are kept as labels", () => {
    it("but a label column wins the label slot, so the original id of a labelled node is lost and unset labels are filled with ids", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
        b.addNode("x");
        b.addNode("y");
        b.addEdge("x", "y");
        b.setNodeValue("label", 0, "Ex");
        const s = b.freeze();
        const t = await trip(s, PAJEK[1], PAJEK[2]);
        expect(codes(t.notes)).toEqual(["W_ID_RENUMBERED"]);
        // the note says exactly what happens: a node with a label value loses its id, one without
        // gets its id text as the label (design 8.5: the loss is predicted, not hidden)
        expect(t.notes[0].message).toContain("kept as labels of the nodes without a label value");
        expect(t.notes[0].message).toContain("loses its id");
        expect(t.text, "the id x appears nowhere in the file").toBe("*Vertices 2\n1 Ex\n2 y\n*Arcs\n1 2\n");
        const label = t.snapshot.nodes.require("label");
        // the unset label of y became its id; the id x is gone (nodeIdFrom "label" would yield "Ex")
        expect([0, 1].map((i) => (label.isSet(i) ? label.value(i) : undefined))).toEqual(["Ex", "y"]);
    });
});

describe("D4 a plain edge column named like the importer's weight key becomes THE weight on re-import", () => {
    async function plain(name: string): Promise<GraphSnapshot> {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareEdgeColumn({ name, dtype: "i32" });
        b.addNode(1);
        b.addNode(2);
        b.addEdge(1, 2);
        b.setEdgeValue(name, 0, 7);
        return Promise.resolve(b.freeze());
    }

    // Design 3.7 / 8.4: which source field becomes THE weight is the importer's weightFrom rule, so
    // a plain column written under the weight key reads back as the weight; what check() owes the
    // caller (8.5) is the prediction, W_WEIGHT_KEY_CLASH naming the column.
    it("GML: `value` (the GML weightFrom default) reads back as THE weight, which check() predicts", async () => {
        const s = await plain("value");
        expect(s.flags.weighted).toBe(false);
        const t = await trip(s, GML[1], GML[2]);
        expect(codes(t.notes)).toEqual(["W_WEIGHT_KEY_CLASH:value"]);
        expect(t.snapshot.flags.weighted).toBe(true);
        expect(t.snapshot.edges.get("value")).toBeNull();
        expect(t.snapshot.edgeList().weights?.[0]).toBe(7);
    });

    for (const [name, exporter, importer] of [GEXF, GRAPHML, JSON_, NEO4J]) {
        it(`${name}: \`weight\` without the weight role reads back as THE weight, which check() predicts`, async () => {
            const s = await plain("weight");
            const t = await trip(s, exporter, importer);
            expect(codes(t.notes)).toContain("W_WEIGHT_KEY_CLASH:weight");
            expect(t.snapshot.flags.weighted).toBe(true);
            expect(t.snapshot.edges.get("weight")).toBeNull();
            expect(t.snapshot.edgeList().weights?.[0]).toBe(7);
        });
    }
});

describe("D5 string ids of canonical integer text (design 4.1) re-import as numbers with no note", () => {
    function stringIds(): GraphSnapshot {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addNode("1");
        b.addNode("2");
        b.addEdge("1", "2");
        return b.freeze();
    }

    // Design 4.1: the canonical id rule reads integer text as numbers whatever the file says (GEXF's
    // idtype is a Gephi constant, "string" on every file it writes); the exporters predict the
    // change with W_ID_TEXT_TYPE and `ids: "string"` on import keeps the texts.
    it('GEXF writes idtype="string"; its importer reads 1 and 2 as numbers under the canonical rule and check() says so', async () => {
        const t = await trip(stringIds(), GEXF[1], GEXF[2]);
        expect(t.text).toContain('idtype="string"');
        expect(codes(t.notes)).toEqual(["W_ID_TEXT_TYPE"]);
        expect(t.snapshot.ids.toArray()).toEqual([1, 2]);
        const kept = await trip(stringIds(), GEXF[1], GEXF[2], {}, { ids: "string" });
        expect(kept.snapshot.ids.toArray()).toEqual(["1", "2"]);
    });

    it("DOT: check() predicts the type change and the ids come back as numbers under the canonical rule", async () => {
        const t = await trip(stringIds(), DOT[1], DOT[2]);
        expect(codes(t.notes)).toEqual(["W_ID_TEXT_TYPE"]);
        expect(t.snapshot.ids.toArray()).toEqual([1, 2]);
        const kept = await trip(stringIds(), DOT[1], DOT[2], {}, { ids: "string" });
        expect(kept.snapshot.ids.toArray()).toEqual(["1", "2"]);
    });
});

describe("D6 the timeText companion of design 5.1 is dropped silently", () => {
    function temporal(originType: string): GraphSnapshot {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "when", dtype: "f64", origin: { type: originType, format: "neo4j" } });
        b.declareNodeColumn({ name: "when.text", dtype: "string", role: "timeText", extra: { for: "when" } });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("when", 0, Date.parse("2024-02-29T13:45:00+02:00"));
        b.setNodeValue("when.text", 0, "2024-02-29T13:45:00+02:00");
        return b.freeze();
    }

    it("GEXF: a Neo4j `datetime` column is written as type double and its +02:00 text is lost; check() is silent (a GEXF-spelled `dateTime` is exact)", async () => {
        const exact = await trip(temporal("dateTime"), GEXF[1], GEXF[2]);
        expect(exact.snapshot.nodes.require("when.text").value(0)).toBe("2024-02-29T13:45:00+02:00");
        const t = await trip(temporal("datetime"), GEXF[1], GEXF[2]);
        expect(
            t.text,
            `written as: ${/<attribute id="when"[^>]*>/.exec(t.text)?.[0]}; check() said ${JSON.stringify(codes(t.notes))}`,
        ).toContain("2024-02-29T13:45:00+02:00");
        expect(t.snapshot.nodes.get("when.text")).not.toBeNull();
    });

    for (const [name, exporter, importer, options] of [
        [...GRAPHML, {}],
        [...GML, { sanitizeIds: "mangle" }],
        [...DOT, {}],
        [...JSON_, {}],
        [...PAJEK, {}],
    ] as [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>, AnyExportOptions][]) {
        it(`${name}: the companion column (role timeText) vanishes and check() says nothing about it`, async () => {
            const t = await trip(temporal("dateTime"), exporter, importer, options);
            const about = t.notes.filter((n) => n.column === "when.text");
            expect(
                t.snapshot.nodes.get("when.text") !== null || about.length > 0,
                `when.text is ${t.snapshot.nodes.get("when.text") === null ? "missing" : "present"}; notes: ${JSON.stringify(codes(t.notes))}`,
            ).toBe(true);
        });
    }
});

describe("D7 the GEXF importer declares an all-unset `.text` companion for a plain date attribute", () => {
    it("(the Neo4j importer declares none for the same lexical value): a phantom column on every cross-format trip into GEXF", async () => {
        const gexf = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://gexf.net/1.3" version="1.3"><graph defaultedgetype="directed">',
            '<attributes class="node"><attribute id="0" title="dt" type="date"/></attributes>',
            '<nodes><node id="a"><attvalues><attvalue for="0" value="2024-02-29"/></attvalues></node><node id="b"/></nodes>',
            '<edges><edge id="e" source="a" target="b"/></edges></graph></gexf>',
        ].join("");
        const gb = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await gexfImporter.import(gexf, gb);
        const fromGexf = gb.freeze();
        const nb = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await neo4jImporter.import(":ID,dt:date\na,2024-02-29\nb,\n", nb);
        const fromNeo4j = nb.freeze();
        expect(fromNeo4j.nodes.get("dt.text")).toBeNull();
        const companion = fromGexf.nodes.get("dt.text");
        expect(
            companion === null || companion.nullCount < companion.length,
            `GEXF declared dt.text with ${companion?.nullCount} of ${companion?.length} rows unset`,
        ).toBe(true);
    });
});

describe("D8 JSON exporter: roles and declared dtypes its capability table claims are lost without a note", () => {
    function viz(): GraphSnapshot {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "color", dtype: "f32", components: 4, role: "color" });
        b.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
        b.declareNodeColumn({ name: "kind", dtype: "dict" });
        b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        b.declareNodeColumn({ name: "level", dtype: "f64" });
        b.declareEdgeColumn({ name: "type", dtype: "dict", role: "kind" });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("color", 0, [0.1, 0.2, 0.3, 1]);
        b.setNodeValue("label", 0, "A");
        b.setNodeValue("kind", 0, "x");
        b.setNodeValue("tags", 0, ["p", "q"]);
        b.setNodeValue("level", 0, 3);
        b.setEdgeValue("type", 0, "KNOWS");
        return b.freeze();
    }

    it("the color, label and kind roles vanish and check() names none of them (positions and edge ids do get a note)", async () => {
        const t = await trip(viz(), JSON_[1], JSON_[2]);
        const flagged = new Set(t.notes.map((n) => n.column));
        const lost = ["color", "label", "kind"].filter((name) => {
            const column = name === "kind" ? t.snapshot.edges.get("type") : t.snapshot.nodes.get(name);
            return column !== null && column.meta.role === null && !flagged.has(name === "kind" ? "type" : name);
        });
        expect(lost, `roles lost without a note; notes: ${JSON.stringify(codes(t.notes))}`).toEqual([]);
    });

    it("f32 x4, dict, list and an integral f64 change dtype through JSON (no declarations: research note 07 2.4), and check() names each", async () => {
        // JSON declares no column types, so the inferred dtypes come back; the capability table says
        // so (dtypes f64 / i32 / bool / string, no lists or strides) and check() names every column
        const s = viz();
        const t = await trip(s, JSON_[1], JSON_[2]);
        const color = t.snapshot.nodes.require("color");
        expect(codes(t.notes)).toEqual(
            expect.arrayContaining([
                "W_VIZ_DROPPED:color",
                "W_DTYPE_UNSUPPORTED:kind",
                "W_LIST_UNSUPPORTED:tags",
                "W_INTEGRAL_F64_AS_I32:level",
            ]),
        );
        expect(t.snapshot.nodes.require("kind").dtype).toBe("string");
        expect(t.snapshot.nodes.require("tags").dtype).toBe("json");
        expect(t.snapshot.nodes.require("level").dtype).toBe("i32");
        expect(color.dtype).toBe("json");
        // f32 values are written as the shortest fround-round-tripping decimals and read as f64
        expect((color.value(0) as number[]).map(Math.fround)).toEqual(
            Array.from(s.nodes.require("color").value(0) as ArrayLike<number>),
        );
    });
});

describe("D12 JSON exporter: the recorded d3 / graphology shape is not written back as read", () => {
    it("a d3 document re-exports with directed / multigraph keys and re-imports as node-link", async () => {
        const d3 = '{"nodes":[{"id":"a"},{"id":"b"}],"links":[{"source":"a","target":"b"}]}';
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await jsonImporter.import(d3, b);
        const s = b.freeze();
        expect((s.meta.extra.json as { dialect: string }).dialect).toBe("d3");
        expect(s.meta.declaredMultigraph).toBeNull();
        const t = await trip(s, JSON_[1], JSON_[2]);
        expect(codes(t.notes)).toEqual([]);
        expect((t.snapshot.meta.extra.json as { dialect: string }).dialect, t.text).toBe("d3");
        expect(t.snapshot.meta.declaredMultigraph).toBeNull();
    });
});

describe("D13 position columns skip the dtype check: an f32 position comes back f64 with different values and no note", () => {
    it("GML: graphics x/y/z of an f32 position column re-import as f64 (GML reals are doubles), which check() predicts", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "position", dtype: "f32", components: 3, role: "position" });
        b.addNode(1);
        b.addNode(2);
        b.addEdge(1, 2);
        b.setNodeValue("position", 0, [0.1, 0.2, 0.3]);
        const s = b.freeze();
        const t = await trip(s, GML[1], GML[2]);
        const position = t.snapshot.nodes.require("position");
        expect(codes(t.notes)).toEqual(["W_DTYPE_UNSUPPORTED:position"]);
        expect(position.dtype).toBe("f64");
        // the f32 values are written exactly (the shortest fround-round-tripping decimal) and read as f64
        expect(Array.from(position.value(0) as ArrayLike<number>).map(Math.fround)).toEqual(
            Array.from(s.nodes.require("position").value(0) as ArrayLike<number>),
        );
    });
});

describe("D16 / D17 XML text fidelity", () => {
    it("GEXF and GraphML write a C0 control character literally; a conforming XML 1.0 parser rejects the document and check() is silent", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "s", dtype: "string" });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("s", 0, `ctrl ${c(1)} char`);
        const s = b.freeze();
        for (const [name, exporter] of [GEXF, GRAPHML]) {
            // the fix: check() names the cell (E_XML_ILLEGAL_CHAR) and export() refuses to write it
            const notes = exporter.check(s);
            let text = "";
            let refused = false;
            try {
                text = await exporter.exportToString(s);
            } catch (err) {
                refused = true;
                expect(err, name).toMatchObject({ code: "E_COLUMN_TYPE" });
            }
            expect(text.includes(c(1)) && notes.length === 0, `${name} writes U+0001 literally with no note`).toBe(
                false,
            );
            expect(refused && notes.some((n) => n.code === "E_XML_ILLEGAL_CHAR"), `${name}: refused with a note`).toBe(
                true,
            );
        }
    });

    it("GraphML writes a carriage return literally in <data> text, which XML end-of-line handling turns into LF on any re-read", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "s", dtype: "string" });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("s", 0, "cr\rreturn");
        b.setNodeValue("s", 1, "crlf\r\nboth");
        const s = b.freeze();
        const t = await trip(s, GRAPHML[1], GRAPHML[2]);
        expect(
            [t.snapshot.nodes.require("s").value(0), t.snapshot.nodes.require("s").value(1)],
            `written as ${JSON.stringify(t.text)}`,
        ).toEqual(["cr\rreturn", "crlf\r\nboth"]);
    });
});

describe("D18 an empty-string node id is written and then refused by the exporter's own importer, with no note", () => {
    for (const [name, exporter, importer, exportOptions, importOptions] of [
        [...CSV, { table: "nodes" }, { table: "nodes", ids: "string" }],
        [...NEO4J, {}, { ids: "string" }],
    ] as [
        string,
        GraphExporter<AnyExportOptions>,
        GraphImporter<AnyImportOptions>,
        AnyExportOptions,
        AnyImportOptions,
    ][]) {
        it(`${name}: the node (and its edges) vanish; the CSV node table even writes a blank line the importer skips without an issue`, async () => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            b.addNode("");
            b.addNode("b");
            b.addEdge("", "b");
            const s = b.freeze();
            const t = await trip(s, exporter, importer, exportOptions, importOptions);
            expect(
                t.snapshot.nodeCount === 2 ||
                    codes(t.notes).some((code) => code.startsWith("E_") || code.includes("EMPTY")),
                `${t.snapshot.nodeCount} node(s) after the trip; issues ${JSON.stringify(t.report.issues.map((i) => i.code))}; notes ${JSON.stringify(codes(t.notes))}`,
            ).toBe(true);
        });
    }
});

describe("D19 text importers rewrite the lexical form of numeric-looking cells in a column that ends up string (design 5.1)", () => {
    const TEXTS = ["x", "1e5", "-0", "1.0", "00123"];

    function texts(): GraphSnapshot {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareEdgeColumn({ name: "es", dtype: "string" });
        TEXTS.forEach((_, i) => {
            b.addNode(`n${i}`);
        });
        for (let i = 1; i < TEXTS.length; i++) {
            b.addEdge(`n${i - 1}`, `n${i}`);
            b.setEdgeValue("es", i - 1, TEXTS[i]);
        }
        b.setEdgeValue("es", 0, TEXTS[0]);
        return b.freeze();
    }

    for (const [name, exporter, importer] of [DOT, PAJEK, CSV]) {
        it(`${name}: "1e5" becomes "100000", "-0" becomes "0", "1.0" becomes "1" inside a string column; check() does not predict a value change`, async () => {
            const t = await trip(texts(), exporter, importer);
            const es = t.snapshot.edges.require("es");
            expect(es.dtype).toBe("string");
            const back = [0, 1, 2, 3].map((e) => es.value(e));
            expect(back, `notes: ${JSON.stringify(codes(t.notes))}`).toEqual(["x", "-0", "1.0", "00123"]);
        });
    }
});

describe("D20 mutual pairs written as two directed edges lose the mutual mark with no note in GML and Neo4j", () => {
    it("GEXF mutual -> GML / Neo4j: graphty.mutual is gone and check() has no mutual note (CSV, DOT, JSON, GraphML and Pajek have one)", async () => {
        const gexf = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://gexf.net/1.3" version="1.3"><graph defaultedgetype="directed">',
            '<nodes><node id="a"/><node id="b"/></nodes>',
            '<edges><edge id="e" source="a" target="b" type="mutual"/></edges></graph></gexf>',
        ].join("");
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await gexfImporter.import(gexf, b);
        const s = b.freeze();
        expect(s.edges.byRole("mutual")).not.toBeNull();
        for (const [name, exporter, importer, options] of [
            [...GML, { sanitizeIds: "mangle" }],
            [...NEO4J, {}],
        ] as [string, GraphExporter<AnyExportOptions>, GraphImporter<AnyImportOptions>, AnyExportOptions][]) {
            const t = await trip(s, exporter, importer, options);
            expect(t.snapshot.edgeCount, name).toBe(2);
            expect(
                t.snapshot.edges.byRole("mutual") !== null || t.notes.some((n) => /MUTUAL/.test(n.code)),
                `${name}: mutual mark lost; notes ${JSON.stringify(codes(t.notes))}`,
            ).toBe(true);
        }
    });
});
