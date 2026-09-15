import {
    GraphBuilder,
    type GraphBuilderOptions,
    GraphFormatError,
    type GraphSnapshot,
    INVALID_INDEX,
} from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { PRECISION_CODE, RENAMED_CODE, ROLE_TAKEN_CODE, UNKNOWN_TYPE_CODE } from "../../../src/common/attributes.js";
import { DIRECTION_FORCED_CODE, DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import { GRAPHML_ISSUE, GRAPHML_LOSS } from "../../../src/formats/graphml/constants.js";
import { graphmlImporter, type GraphmlImportOptions } from "../../../src/formats/graphml/importer.js";
import { type CommonImportOptions, ImportError, type ImportInput, type ImportReport } from "../../../src/types.js";
import {
    corpusEntry,
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
} from "../../helpers/corpus.js";

type Options = GraphmlImportOptions & CommonImportOptions;

interface Loaded {
    readonly snapshot: GraphSnapshot;
    readonly report: ImportReport;
    readonly builder: GraphBuilder;
}

async function load(
    input: ImportInput,
    options?: Options,
    builderOptions: Partial<GraphBuilderOptions> = {},
): Promise<Loaded> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64", ...builderOptions });
    const report = await graphmlImporter.import(input, builder, options);
    return { snapshot: builder.freeze(), report, builder };
}

async function importError(input: ImportInput, options?: Options): Promise<ImportError> {
    try {
        await load(input, options);
    } catch (err) {
        expect(err).toBeInstanceOf(ImportError);
        return err as ImportError;
    }
    throw new Error("expected an ImportError");
}

function codes(report: ImportReport): string[] {
    return report.issues.map((issue) => issue.code);
}

/** The values of a column through value(): the declared default for unset rows, undefined without one. */
function column(snapshot: GraphSnapshot, table: "nodes" | "edges" | "graph", name: string): unknown[] {
    const c = snapshot[table].get(name);
    if (c === null) {
        throw new Error(`no ${table} column ${name}`);
    }
    return Array.from({ length: c.length }, (_, r) => c.value(r));
}

const NS = 'xmlns="http://graphml.graphdrawing.org/xmlns"';

function doc(body: string, edgedefault = "directed", keys = ""): string {
    return `<?xml version="1.0" encoding="UTF-8"?>\n<graphml ${NS}>\n${keys}<graph id="G" edgedefault="${edgedefault}">\n${body}\n</graph>\n</graphml>\n`;
}

describe("graphmlImporter corpus", () => {
    it("declares the format facts", () => {
        expect(graphmlImporter.format).toBe("graphml");
        expect(graphmlImporter.extensions).toContain(".graphml");
        expect(graphmlImporter.mimeTypes).toContain("application/graphml+xml");
    });

    for (const entry of corpusFiles("graphml")) {
        it(`imports ${entry.path} with the manifest's counts`, async () => {
            const { snapshot, report } = await load(readCorpusText("graphml", entry.path));
            expect(snapshot.nodeCount).toBe(entry.expectedNodes);
            expect(snapshot.edgeCount).toBe(entry.expectedEdges);
            expect(report.counts.nodes).toBe(entry.expectedNodes);
            expect(report.counts.edges).toBe(entry.expectedEdges);
            expect(report.errorCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(report.format).toBe("graphml");
            expect(snapshot.meta.sourceFormat).toBe("graphml");
        });
    }

    it("imports simple.graphml: undirected, label column with the label role, double weights", async () => {
        const { snapshot, report } = await load(readCorpusText("graphml", "simple.graphml"));
        expect(snapshot.directed).toBe(false);
        expect(snapshot.ids.toArray()).toEqual(["n0", "n1", "n2", "n3", "n4"]);
        const label = snapshot.nodes.require("label");
        expect(label.meta.role).toBe("label");
        expect(label.meta.dtype).toBe("string");
        expect(label.meta.origin).toEqual({
            format: "graphml",
            id: "d0",
            title: null,
            type: "string",
            namespace: null,
        });
        expect(column(snapshot, "nodes", "label")).toEqual(["Node A", "Node B", "Node C", "Node D", "Node E"]);
        expect(snapshot.flags.weighted).toBe(true);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1, 2, 1.5, 1, 3]);
        expect(snapshot.edges.byRole("weight")).toBeNull();
        expect(snapshot.edges.names()).toEqual([]);
        expect(snapshot.meta.weightOrigin).toEqual({
            format: "graphml",
            id: "d1",
            title: "weight",
            type: "double",
            namespace: null,
        });
        expect(snapshot.meta.extra).toEqual({
            graphml: { graphId: null, edgedefault: "undirected", namespaces: {} },
        });
        expect(report.issues).toEqual([]);
        expect(report.lossy).toEqual([]);
    });

    it("imports got-network.graphml: ids with spaces, edge ids, edge labels, weights", async () => {
        const { snapshot } = await load(readCorpusText("graphml", "got-network.graphml"));
        expect(snapshot.directed).toBe(false);
        expect(snapshot.ids.idOf(0)).toBe("Aemon");
        expect(snapshot.ids.indexOf("Jon Arryn")).not.toBe(INVALID_INDEX);
        const id = snapshot.edges.require("id");
        expect(id.meta.role).toBe("id");
        expect(id.meta.unique).toBe(true);
        expect(id.value(0)).toBe("0");
        expect(snapshot.edgeIndexOf("351")).toBe(351);
        const edgeLabel = snapshot.edges.require("Edge Label");
        expect(edgeLabel.meta.origin?.id).toBe("edgelabel");
        expect(snapshot.flags.weighted).toBe(true);
        const [source, target] = [snapshot.edgeList().src[0], snapshot.edgeList().dst[0]];
        expect([snapshot.ids.idOf(source), snapshot.ids.idOf(target)]).toEqual(["Aemon", "Grenn"]);
    });

    it("imports yfiles-sample.graphml: yFiles graphics as json trees with a yfiles origin", async () => {
        const { snapshot, report } = await load(readCorpusText("graphml", "yfiles-sample.graphml"));
        expect(snapshot.directed).toBe(true);
        const graphics = snapshot.nodes.require("d0");
        expect(graphics.meta.dtype).toBe("json");
        expect(graphics.meta.origin).toEqual({
            format: "graphml",
            id: "d0",
            title: null,
            type: "nodegraphics",
            namespace: "yfiles",
        });
        const start = graphics.value(0) as Record<string, Record<string, unknown>>;
        expect(start["y:ShapeNode"]["y:NodeLabel"]).toBe("Start");
        expect(start["y:ShapeNode"]["y:Geometry"]).toEqual({
            "@_x": "0.0",
            "@_y": "0.0",
            "@_width": "60.0",
            "@_height": "30.0",
        });
        expect(start["y:ShapeNode"]["y:Fill"]).toEqual({ "@_color": "#FFCC00", "@_transparent": "false" });
        const edges = snapshot.edges.require("d1");
        expect(edges.meta.origin?.type).toBe("edgegraphics");
        const e2 = edges.value(2) as Record<string, Record<string, unknown>>;
        expect(e2["y:PolyLineEdge"]["y:LineStyle"]).toEqual({
            "@_color": "#FF0000",
            "@_type": "line",
            "@_width": "2.0",
        });
        expect(column(snapshot, "edges", "id")).toEqual(["e0", "e1", "e2", "e3"]);
        expect(report.lossy.map((note) => note.code)).toEqual([GRAPHML_LOSS.YFILES_JSON]);
        expect(snapshot.meta.extra).toEqual({
            graphml: {
                graphId: null,
                edgedefault: "directed",
                namespaces: { y: "http://www.yworks.com/xml/graphml" },
            },
        });
    });

    it("reads every input shape identically", async () => {
        const entry = corpusEntry("graphml", "got-network.graphml");
        const bytes = readCorpusBytes("graphml", entry.path);
        const reference = await load(bytes);
        for (const shape of inputShapes(bytes)) {
            const { snapshot } = await load(shape.make());
            expect(snapshot.nodeCount, shape.name).toBe(entry.expectedNodes);
            expect(snapshot.edgeCount, shape.name).toBe(entry.expectedEdges);
            expect(snapshot.ids.toArray(), shape.name).toEqual(reference.snapshot.ids.toArray());
            expect(Array.from(snapshot.colIdx), shape.name).toEqual(Array.from(reference.snapshot.colIdx));
        }
    });

    it("reports progress and honours an abort signal", async () => {
        const bytes = readCorpusBytes("graphml", "got-network.graphml");
        const progress: number[] = [];
        await load(bytes, { onProgress: (done) => progress.push(done) });
        expect(progress[progress.length - 1]).toBe(bytes.byteLength);
        const controller = new AbortController();
        controller.abort();
        await expect(load(bytes, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    });

    it("rejects the stray got-social-network.graphml (not XML) with an ImportError", async () => {
        const err = await importError(readCorpusText("graphml", "got-social-network.graphml"));
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.issues[0]).toMatchObject({ code: GRAPHML_ISSUE.XML_SYNTAX, category: "parse-error" });
    });
});

describe("graphmlImporter malformed corpus", () => {
    const fatal: Record<string, string> = {
        "empty-file.graphml": GRAPHML_ISSUE.XML_SYNTAX,
        "invalid-xml.graphml": GRAPHML_ISSUE.XML_SYNTAX,
        "not-xml.graphml": GRAPHML_ISSUE.XML_SYNTAX,
        "unclosed-tag.graphml": GRAPHML_ISSUE.XML_SYNTAX,
        "missing-graph.graphml": GRAPHML_ISSUE.NO_GRAPH,
    };
    const recoverable: Record<string, { code: string; nodes: number; edges: number; options?: Options }> = {
        "invalid-edge-reference.graphml": {
            code: "E_UNKNOWN_NODE",
            nodes: 2,
            edges: 0,
            options: { addMissingNodes: false },
        },
        "missing-edge-source.graphml": { code: GRAPHML_ISSUE.MISSING_ENDPOINT, nodes: 2, edges: 0 },
        "missing-node-id.graphml": { code: GRAPHML_ISSUE.MISSING_ID, nodes: 2, edges: 1 },
    };

    for (const name of malformedFiles("graphml")) {
        it(`${name}: ImportError with a report under errorLimit 0`, async () => {
            const bytes = readMalformedBytes("graphml", name);
            const err = await importError(bytes, { errorLimit: 0, ...recoverable[name]?.options });
            expect(err.code).toBe("E_IMPORT");
            expect(err.report.format).toBe("graphml");
            expect(err.report.errorCount).toBeGreaterThan(0);
            const expected = fatal[name] ?? recoverable[name]?.code;
            expect(expected, `no expectation for ${name}`).toBeDefined();
            expect(codes(err.report)).toContain(expected);
            if (name in fatal) {
                await expect(load(bytes)).rejects.toBeInstanceOf(ImportError);
            }
        });
    }

    for (const [name, expected] of Object.entries(recoverable)) {
        it(`${name}: recoverable under the default error limit`, async () => {
            const { snapshot, report } = await load(readMalformedBytes("graphml", name), expected.options);
            expect(codes(report)).toContain(expected.code);
            expect(report.errorCount).toBeGreaterThan(0);
            expect(report.truncated).toBe(false);
            expect(snapshot.nodeCount).toBe(expected.nodes);
            expect(snapshot.edgeCount).toBe(expected.edges);
        });
    }

    it("invalid-edge-reference.graphml creates the endpoints under addMissingNodes (the default)", async () => {
        const { snapshot, report } = await load(readMalformedBytes("graphml", "invalid-edge-reference.graphml"));
        expect(snapshot.nodeCount).toBe(4);
        expect(snapshot.edgeCount).toBe(2);
        // counts.nodes is the sink's node count: the two declared nodes and the two the edges created
        expect(report.counts).toMatchObject({ nodes: 4, edges: 2, skippedEdges: 0 });
    });

    it("missing-node-id.graphml reports the line of the bad node", async () => {
        const { report } = await load(readMalformedBytes("graphml", "missing-node-id.graphml"));
        const issue = report.issues.find((i) => i.code === GRAPHML_ISSUE.MISSING_ID);
        expect(issue).toMatchObject({ category: "missing-value", severity: "error", line: 4 });
        expect(report.counts.skippedNodes).toBe(1);
    });

    it("invalid-xml.graphml names the mismatched tags and the line", async () => {
        const err = await importError(readMalformedBytes("graphml", "invalid-xml.graphml"));
        const syntax = err.report.issues.find((i) => i.code === GRAPHML_ISSUE.XML_SYNTAX);
        expect(syntax?.message).toMatch(/<\/graph> does not match <node>/);
        expect(syntax?.line).toBe(7);
    });

    it("rejects invalid UTF-8 as a parse error", async () => {
        const bytes = new Uint8Array([...new TextEncoder().encode(doc('<node id="n1"/>').slice(0, 60)), 0xff, 0xfe]);
        const err = await importError(bytes);
        expect(codes(err.report)).toContain(INVALID_UTF8_CODE);
    });

    it("rejects a root element that is not graphml", async () => {
        const err = await importError('<gexf xmlns="http://gexf.net/1.3"><graph/></gexf>');
        expect(codes(err.report)).toEqual([GRAPHML_ISSUE.NOT_GRAPHML]);
    });

    it("aborts with a truncated report when the error limit is exceeded", async () => {
        const body = Array.from({ length: 10 }, () => "<node/>").join("\n");
        const err = await importError(doc(body), { errorLimit: 3 });
        expect(err.report.truncated).toBe(true);
        expect(err.report.errorCount).toBe(4);
        expect(err.report.counts.skippedNodes).toBe(4);
    });
});

describe("graphmlImporter keys and data", () => {
    const KEYS = `<key id="k_bool" for="node" attr.name="flag" attr.type="boolean"><default>true</default></key>
<key id="k_int" for="node" attr.name="count" attr.type="int"/>
<key id="k_long" for="node" attr.name="big" attr.type="long"/>
<key id="k_float" for="node" attr.name="ratio" attr.type="float"/>
<key id="k_double" for="all" attr.name="score" attr.type="double"><desc>a score</desc><default>0.5</default></key>
<key id="k_str" for="edge" attr.name="kind" attr.type="string"/>
<key id="k_graph" for="graph" attr.name="title" attr.type="string"/>
<key id="k_untyped" for="node" attr.name="note"/>
<key id="k_noname" for="node" attr.type="int"/>
`;

    it("declares typed columns with defaults for every domain and parses values by type", async () => {
        const body = `<data key="k_graph">The graph</data>
<data key="k_double">9.5</data>
<node id="a"><data key="k_bool">false</data><data key="k_int">7</data><data key="k_long">9007199254740993</data><data key="k_float">0.5</data><data key="k_double">1.25</data><data key="k_untyped">free text</data><data key="k_noname">3</data></node>
<node id="b"><data key="k_bool">1</data><data key="k_int"> 8 </data></node>
<node id="c"/>
<edge source="a" target="b"><data key="k_str">friend</data><data key="k_double">2</data></edge>`;
        const { snapshot, report } = await load(doc(body, "directed", KEYS));
        expect(snapshot.nodes.require("flag").meta).toMatchObject({ dtype: "bool", default: true });
        expect(column(snapshot, "nodes", "flag")).toEqual([false, true, true]);
        expect(snapshot.nodes.require("flag").isSet(2)).toBe(false);
        expect(snapshot.nodes.require("count").meta.dtype).toBe("i32");
        expect(column(snapshot, "nodes", "count")).toEqual([7, 8, undefined]);
        expect(snapshot.nodes.require("big").meta).toMatchObject({ dtype: "f64" });
        expect(snapshot.nodes.require("big").meta.origin?.type).toBe("long");
        expect(column(snapshot, "nodes", "big")).toEqual([9007199254740992, undefined, undefined]);
        expect(snapshot.nodes.require("ratio").meta.dtype).toBe("f32");
        expect(column(snapshot, "nodes", "ratio")).toEqual([0.5, undefined, undefined]);
        expect(snapshot.nodes.require("score").meta).toMatchObject({ dtype: "f64", default: 0.5 });
        expect(snapshot.nodes.require("score").meta.extra).toEqual({ desc: "a score" });
        expect(column(snapshot, "nodes", "score")).toEqual([1.25, 0.5, 0.5]);
        expect(snapshot.edges.require("score").meta.origin?.id).toBe("k_double");
        expect(column(snapshot, "edges", "score")).toEqual([2]);
        expect(snapshot.graph.require("score").value(0)).toBe(9.5);
        expect(snapshot.graph.require("title").value(0)).toBe("The graph");
        expect(snapshot.graph.require("title").meta.dtype).toBe("string");
        expect(column(snapshot, "edges", "kind")).toEqual(["friend"]);
        expect(snapshot.nodes.require("note").meta.dtype).toBe("string");
        expect(snapshot.nodes.require("note").meta.origin?.type).toBeNull();
        expect(column(snapshot, "nodes", "note")).toEqual(["free text", undefined, undefined]);
        // a key without attr.name is named by its id
        expect(column(snapshot, "nodes", "k_noname")).toEqual([3, undefined, undefined]);
        expect(codes(report)).toEqual([PRECISION_CODE]);
        expect(report.issues[0]).toMatchObject({ category: "precision", severity: "warning", element: "big" });
    });

    it('keeps a declared long as text under long: "string"', async () => {
        const body = `<node id="a"><data key="k_long">9007199254740993</data></node>`;
        const { snapshot, report } = await load(doc(body, "directed", KEYS), { long: "string" });
        expect(snapshot.nodes.require("big").meta.dtype).toBe("string");
        expect(column(snapshot, "nodes", "big")).toEqual(["9007199254740993"]);
        expect(report.issues).toEqual([]);
    });

    it("records type errors per value and keeps the element", async () => {
        const body = `<node id="a"><data key="k_int">seven</data><data key="k_bool">maybe</data><data key="k_float">1.5</data></node>`;
        const { snapshot, report } = await load(doc(body, "directed", KEYS));
        expect(snapshot.nodeCount).toBe(1);
        expect(column(snapshot, "nodes", "ratio")).toEqual([1.5]);
        expect(snapshot.nodes.require("count").isSet(0)).toBe(false);
        expect(codes(report)).toEqual(["E_COLUMN_TYPE", "E_COLUMN_TYPE"]);
        expect(report.issues[0]).toMatchObject({ category: "validation-error", element: "k_int", line: 13 });
    });

    it("treats blank text as unset for non-string types and as an empty string for strings", async () => {
        const body = `<node id="a"><data key="k_int"> </data><data key="k_untyped"></data><data key="k_double"/></node>`;
        const { snapshot, report } = await load(doc(body, "directed", KEYS));
        expect(snapshot.nodes.require("count").isSet(0)).toBe(false);
        expect(snapshot.nodes.require("score").isSet(0)).toBe(false);
        expect(column(snapshot, "nodes", "note")).toEqual([""]);
        expect(report.issues).toEqual([]);
    });

    it("reports unknown keys, keys of another domain, data without a key, and unknown types", async () => {
        const keys = `<key id="d" for="node" attr.name="x" attr.type="date"/>\n<key id="e" for="edge" attr.name="y" attr.type="int"/>\n`;
        const body = `<node id="a"><data key="nope">1</data><data key="e">1</data><data>1</data><data key="d">2024-01-01</data></node>`;
        const { snapshot, report } = await load(doc(body, "directed", keys));
        expect(codes(report)).toEqual([
            UNKNOWN_TYPE_CODE,
            GRAPHML_ISSUE.UNKNOWN_KEY,
            GRAPHML_ISSUE.KEY_DOMAIN,
            GRAPHML_ISSUE.DATA_MISSING_KEY,
        ]);
        expect(snapshot.nodes.require("x").meta.dtype).toBe("string");
        expect(snapshot.nodes.require("x").meta.origin?.type).toBe("date");
        expect(column(snapshot, "nodes", "x")).toEqual(["2024-01-01"]);
        expect(report.errorCount).toBe(2);
    });

    it("reports keys without an id, duplicate keys, invalid and unsupported domains", async () => {
        const keys = `<key for="node" attr.name="a"/>\n<key id="k" for="node" attr.name="b"/>\n<key id="k" for="node" attr.name="c"/>\n<key id="h" for="hyperedge" attr.name="d"/>\n<key id="p" for="port" attr.name="e"/>\n<key id="z" for="zone" attr.name="f"/>\n`;
        const { snapshot, report } = await load(doc('<node id="a"/>', "directed", keys));
        expect(codes(report)).toEqual([
            GRAPHML_ISSUE.KEY_MISSING_ID,
            GRAPHML_ISSUE.DUPLICATE_KEY,
            GRAPHML_ISSUE.KEY_DOMAIN_UNSUPPORTED,
            GRAPHML_ISSUE.KEY_DOMAIN_UNSUPPORTED,
            GRAPHML_ISSUE.KEY_FOR_INVALID,
        ]);
        expect(snapshot.nodes.names()).toEqual(["b"]);
    });

    it("renames a key that collides with an XML-derived or earlier column and records the rename", async () => {
        const keys = `<key id="d0" for="node" attr.name="label" attr.type="string"/>\n<key id="d1" for="node" attr.name="label" attr.type="int"/>\n<key id="d2" for="edge" attr.name="id" attr.type="string"/>\n<key id="d3" for="node" attr.name="parent" attr.type="string"/>\n`;
        const body = `<node id="a"><data key="d0">A</data><data key="d1">1</data><data key="d3">p</data></node><edge id="e1" source="a" target="a"><data key="d2">inner</data></edge>`;
        const { snapshot, report } = await load(doc(body, "directed", keys));
        expect(snapshot.nodes.names()).toEqual(["label", "label#d1", "parent#d3"]);
        expect(snapshot.nodes.require("label").meta.role).toBe("label");
        expect(snapshot.nodes.require("label#d1").meta.role).toBeNull();
        expect(snapshot.nodes.require("label#d1").meta.origin?.title).toBe("label");
        expect(snapshot.edges.names()).toEqual(["id#d2", "id"]);
        expect(column(snapshot, "edges", "id")).toEqual(["e1"]);
        expect(column(snapshot, "edges", "id#d2")).toEqual(["inner"]);
        // the second label key is renamed and, the label role being taken, declared without it
        expect(codes(report)).toEqual([RENAMED_CODE, ROLE_TAKEN_CODE, RENAMED_CODE, RENAMED_CODE]);
    });

    it("declares a graph key up front even without data and keeps its default", async () => {
        const keys = `<key id="g" for="graph" attr.name="version" attr.type="int"><default>2</default></key>\n`;
        const { snapshot } = await load(doc('<node id="a"/>', "directed", keys));
        const version = snapshot.graph.require("version");
        expect(version.meta.dtype).toBe("i32");
        expect(version.isSet(0)).toBe(false);
        expect(version.value(0)).toBe(2);
    });

    it("reports a bad default and keeps the column without one", async () => {
        const keys = `<key id="g" for="node" attr.name="n" attr.type="int"><default>lots</default></key>\n`;
        const { snapshot, report } = await load(doc('<node id="a"/>', "directed", keys));
        expect(snapshot.nodes.require("n").meta.default).toBeUndefined();
        expect(codes(report)).toEqual(["W_BAD_DEFAULT"]);
    });

    it("rejects nested elements inside a typed data element", async () => {
        const keys = `<key id="d" for="node" attr.name="n" attr.type="int"><default><x/></default></key>\n`;
        const body = `<node id="a"><data key="d"><y:Foo/></data></node>`;
        const { snapshot, report } = await load(doc(body, "directed", keys));
        expect(codes(report)).toEqual([GRAPHML_ISSUE.DATA_NESTED, GRAPHML_ISSUE.DATA_NESTED]);
        expect(snapshot.nodes.require("n").isSet(0)).toBe(false);
    });

    it("keeps the graph description and reports node / edge descriptions, ports, locators and unknown elements once", async () => {
        const body = `<desc>My graph</desc>
<node id="a"><desc>node a</desc><port name="p1"><data key="x">1</data></port><locator xlink:href="u"/></node>
<node id="b"><desc>node b</desc><bogus/></node>
<edge source="a" target="b" sourceport="p1" targetport="p2"><desc>e</desc></edge>
<locator xlink:href="v"/>`;
        const { snapshot, report } = await load(doc(body));
        expect(snapshot.meta.description).toBe("My graph");
        expect(codes(report)).toEqual([
            GRAPHML_ISSUE.DESC_DROPPED,
            GRAPHML_ISSUE.PORT_DECLARATION,
            GRAPHML_ISSUE.LOCATOR_DROPPED,
            GRAPHML_ISSUE.UNKNOWN_ELEMENT,
        ]);
        expect(report.issues.every((i) => i.severity === "warning")).toBe(true);
        expect(snapshot.edges.require("sourceport").meta.role).toBe("sourcePort");
        expect(snapshot.edges.require("targetport").meta.role).toBe("targetPort");
        expect(column(snapshot, "edges", "sourceport")).toEqual(["p1"]);
        expect(column(snapshot, "edges", "targetport")).toEqual(["p2"]);
    });

    it("reports stray text and data of a nested graph", async () => {
        const body = `stray words
<node id="a"><graph id="a:" edgedefault="directed"><data key="k">1</data><node id="b"/></graph></node>`;
        const keys = `<key id="k" for="graph" attr.name="k" attr.type="int"/>\n`;
        const { report } = await load(doc(body, "directed", keys));
        expect(codes(report)).toEqual([GRAPHML_ISSUE.STRAY_TEXT, GRAPHML_ISSUE.NESTED_GRAPH_DATA]);
    });
});

describe("graphmlImporter ids and endpoints", () => {
    it("coerces canonical integer text to numbers and keeps other text as strings", async () => {
        const body = `<node id="1"/><node id="01"/><node id="-0"/><node id="2"/><node id="x"/><edge source="1" target="2"/>`;
        const { snapshot } = await load(doc(body));
        expect(snapshot.ids.toArray()).toEqual([1, "01", "-0", 2, "x"]);
    });

    it("honours ids: string, keep and number (with merge reporting)", async () => {
        const body = `<node id="1"/><node id="01"/><node id="2"/><edge source="01" target="2"/>`;
        const strings = await load(doc(body), { ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["1", "01", "2"]);
        const keep = await load(doc(body), { ids: "keep" });
        expect(keep.snapshot.ids.toArray()).toEqual(["1", "01", "2"]);
        const numbers = await load(doc(body), { ids: "number" });
        expect(numbers.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(codes(numbers.report)).toEqual([
            GRAPHML_ISSUE.ID_MERGED,
            GRAPHML_ISSUE.DUPLICATE_NODE,
            GRAPHML_ISSUE.ID_MERGED,
        ]);
        const bad = await load(doc('<node id="x"/>'), { ids: "number" });
        expect(codes(bad.report)).toEqual(["E_INVALID_ID"]);
        expect(bad.snapshot.nodeCount).toBe(0);
    });

    it("merges a node declared twice with a warning and counts it once", async () => {
        const body = `<node id="a"><data key="d">1</data></node><node id="a"><data key="d">2</data></node>`;
        const keys = `<key id="d" for="node" attr.name="v" attr.type="int"/>\n`;
        const { snapshot, report } = await load(doc(body, "directed", keys));
        expect(snapshot.nodeCount).toBe(1);
        expect(column(snapshot, "nodes", "v")).toEqual([2]);
        expect(report.counts.nodes).toBe(1);
        expect(codes(report)).toEqual([GRAPHML_ISSUE.DUPLICATE_NODE]);
    });

    it("reports edges without a target, with a bad directed flag, or a duplicate id, and skips them", async () => {
        const body = `<node id="a"/><node id="b"/>
<edge source="a"/>
<edge source="a" target="b" directed="maybe"/>
<edge id="e" source="a" target="b"/>
<edge id="e" source="b" target="a"/>`;
        const { snapshot, report } = await load(doc(body));
        expect(snapshot.edgeCount).toBe(1);
        expect(codes(report)).toEqual([
            GRAPHML_ISSUE.MISSING_ENDPOINT,
            GRAPHML_ISSUE.INVALID_DIRECTED,
            "E_DUPLICATE_EDGE_ID",
        ]);
        expect(report.counts).toMatchObject({ edges: 1, skippedEdges: 3 });
        expect(report.issues[2]).toMatchObject({ element: "e", line: 8 });
    });

    it("enforces addMissingNodes: false itself and reports the sink's policies when they differ", async () => {
        const body = `<node id="a"/><edge source="a" target="ghost"/>`;
        const { snapshot, report } = await load(doc(body), { addMissingNodes: false, selfLoops: "drop" });
        expect(snapshot.nodeCount).toBe(1);
        expect(snapshot.edgeCount).toBe(0);
        expect(codes(report)).toEqual([SINK_OPTION_CODE, "E_UNKNOWN_NODE"]);
        expect(report.issues[0]).toMatchObject({ element: "selfLoops", category: "coercion" });
        expect(report.issues[0].message).toMatch(/selfLoops: "drop" requested but the sink uses "keep"/);
        // addMissingNodes: true against a refusing sink cannot be honoured; nodeIdFrom has no GraphML meaning
        const strict = await load(
            doc(body),
            { nodeIdFrom: "label", addMissingNodes: true },
            { addMissingNodes: false },
        );
        expect(codes(strict.report)).toEqual([SINK_OPTION_CODE, GRAPHML_ISSUE.OPTION_IGNORED, "E_UNKNOWN_NODE"]);
        // options left undefined are defaults, never requests
        const silent = await load(doc(body), { nodeIdFrom: "label" }, { addMissingNodes: false });
        expect(codes(silent.report)).toEqual([GRAPHML_ISSUE.OPTION_IGNORED, "E_UNKNOWN_NODE"]);
    });

    it("honours parse hints by reserving capacity without changing the result", async () => {
        const text = doc('<node id="a"/><node id="b"/><edge source="a" target="b"/>').replace(
            'edgedefault="directed"',
            'edgedefault="directed" parse.nodes="2" parse.edges="1" parse.order="nodesfirst"',
        );
        const { snapshot } = await load(text);
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.edgeCount).toBe(1);
    });
});

describe("graphmlImporter direction", () => {
    const MIXED = `<node id="a"/><node id="b"/><node id="c"/>
<edge source="a" target="b"/>
<edge source="b" target="c" directed="false"/>
<edge source="c" target="a" directed="true"/>
<edge source="c" target="c" directed="false"/>`;

    it("expands per-edge overrides in a directed graph into pairs (the default policy)", async () => {
        const { snapshot, report } = await load(doc(MIXED, "directed"));
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(5);
        expect(report.counts).toMatchObject({ edges: 5, expandedMixed: 2 });
        const directed = snapshot.edges.byRole("directed");
        const pair = snapshot.edges.byRole("pair");
        expect(directed?.meta.name).toBe("graphty.directed");
        expect(pair?.meta.name).toBe("graphty.pair");
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([true, false, false, true, false]);
        expect(column(snapshot, "edges", "graphty.pair")).toEqual([undefined, 2, 1, undefined, undefined]);
        expect(snapshot.meta.extra).toMatchObject({ graphml: { edgedefault: "directed" } });
    });

    it("expands an undirected graph with directed overrides through the sink", async () => {
        const { snapshot, report } = await load(doc(MIXED, "undirected"));
        expect(snapshot.directed).toBe(true);
        // a->b (u), b->c (u), c->a (d): the first directed edge expands the two earlier ones in place; the
        // undirected self-loop that follows is expanded (one arc) as well
        expect(snapshot.edgeCount).toBe(6);
        expect(report.counts.expandedMixed).toBe(3);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([false, false, false, false, true, false]);
    });

    it("keeps an all-undirected and an all-directed file without pair columns", async () => {
        const undirected = await load(doc('<node id="a"/><edge source="a" target="a"/>', "undirected"));
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edges.byRole("pair")).toBeNull();
        const directed = await load(doc('<edge source="a" target="b"/>', "directed"));
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edges.byRole("directed")).toBeNull();
    });

    it("forces every edge under onMixedDirection directed / undirected and refuses under error", async () => {
        const forced = await load(doc(MIXED, "directed"), { onMixedDirection: "undirected" });
        expect(forced.snapshot.directed).toBe(false);
        expect(forced.snapshot.edgeCount).toBe(4);
        expect(codes(forced.report)).toEqual([DIRECTION_FORCED_CODE, DIRECTION_FORCED_CODE]);
        const directed = await load(doc(MIXED, "undirected"), { onMixedDirection: "directed" });
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edgeCount).toBe(4);
        const err = await importError(doc(MIXED, "directed"), { onMixedDirection: "error" });
        expect(codes(err.report)).toContain(MIXED_DIRECTION_CODE);
    });

    it("uses defaultDirected when edgedefault is missing and reports an invalid one", async () => {
        const missing = doc('<node id="a"/>').replace(' edgedefault="directed"', "");
        const a = await load(missing);
        expect(a.snapshot.directed).toBe(false);
        expect(codes(a.report)).toEqual([GRAPHML_ISSUE.EDGEDEFAULT_MISSING]);
        const b = await load(missing, { defaultDirected: true });
        expect(b.snapshot.directed).toBe(true);
        const c = await load(doc('<node id="a"/>', "sideways"));
        expect(c.snapshot.directed).toBe(false);
        expect(codes(c.report)).toEqual([GRAPHML_ISSUE.INVALID_EDGEDEFAULT]);
        expect(c.report.errorCount).toBe(1);
    });

    it("keeps a locked sink's direction and reports the refusal", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        const report = await graphmlImporter.import(doc('<edge source="a" target="b"/>', "undirected"), builder);
        expect(builder.directed).toBe(true);
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        const snapshot = builder.freeze();
        expect(snapshot.edgeCount).toBe(2);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([false, false]);
    });

    it("merges a second top-level graph into the first with a warning", async () => {
        const text = `<graphml><graph id="one" edgedefault="directed"><node id="a"/><edge source="a" target="b"/></graph>
<graph id="two" edgedefault="undirected"><node id="c"/><edge source="c" target="a"/></graph></graphml>`;
        const { snapshot, report } = await load(text);
        expect(codes(report)).toEqual([GRAPHML_ISSUE.MULTIPLE_GRAPHS]);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(snapshot.edgeCount).toBe(3);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([true, false, false]);
        expect(snapshot.meta.extra).toMatchObject({ graphml: { graphId: "one", edgedefault: "directed" } });
    });

    it("uses each nested graph's edgedefault for its own edges", async () => {
        const body = `<node id="a"><graph id="a:" edgedefault="undirected"><node id="b"/><node id="c"/><edge source="b" target="c"/></graph></node><edge source="a" target="b"/>`;
        const { snapshot } = await load(doc(body, "directed"));
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([false, false, true]);
    });
});

describe("graphmlImporter weights", () => {
    it("passes the weight key to addEdge, omitting absent weights, and rejects bad ones", async () => {
        const keys = `<key id="w" for="edge" attr.name="weight" attr.type="double"/>\n`;
        const body = `<node id="a"/><node id="b"/>
<edge source="a" target="b"><data key="w">2.5</data></edge>
<edge source="b" target="a"/>
<edge source="a" target="a"><data key="w">INF</data></edge>
<edge source="b" target="b"><data key="w">heavy</data></edge>
<edge source="a" target="b"><data key="w"> </data></edge>`;
        const { snapshot, report } = await load(doc(body, "directed", keys));
        expect(snapshot.edgeCount).toBe(4);
        expect(codes(report)).toEqual(["E_INVALID_WEIGHT"]);
        expect(report.counts.skippedEdges).toBe(1);
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect([0, 1, 2, 3].map((e) => shadow?.isSet(e))).toEqual([true, false, true, false]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2.5, 1, Infinity, 1]);
        expect(snapshot.edges.names()).toEqual(["graphty.weight"]);
    });

    it("honours weightFrom: another attribute or null", async () => {
        const keys = `<key id="w" for="edge" attr.name="weight" attr.type="double"/>\n<key id="c" for="edge" attr.name="cost" attr.type="int"/>\n`;
        const body = `<edge source="a" target="b"><data key="w">2.5</data><data key="c">3</data></edge>`;
        const cost = await load(doc(body, "directed", keys), { weightFrom: "cost" });
        expect(Array.from(cost.snapshot.edgeList().weights ?? [])).toEqual([3]);
        expect(column(cost.snapshot, "edges", "weight")).toEqual([2.5]);
        expect(cost.snapshot.meta.weightOrigin?.id).toBe("c");
        const none = await load(doc(body, "directed", keys), { weightFrom: null });
        expect(none.snapshot.flags.weighted).toBe(false);
        expect(none.snapshot.edges.names()).toEqual(["weight", "cost"]);
        expect(none.snapshot.meta.weightOrigin).toBeNull();
    });

    it("keeps weights exact under the default f64 staging", async () => {
        const keys = `<key id="w" for="edge" attr.name="weight" attr.type="double"/>\n`;
        const body = `<edge source="a" target="b"><data key="w">0.1</data></edge><edge source="b" target="c"><data key="w">16777217</data></edge>`;
        const { snapshot } = await load(doc(body, "directed", keys));
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow?.value(0)).toBe(0.1);
        expect(shadow?.value(1)).toBe(16777217);
    });
});

describe("graphmlImporter nesting, hyperedges and yFiles", () => {
    const NESTED = `<node id="a"><graph id="a:" edgedefault="directed"><node id="b"/><node id="c"><graph id="c:" edgedefault="directed"><node id="d"/></graph></node></graph></node><node id="e"/><edge source="b" target="d"/>`;

    it("maps nested graphs to the parent column, never to topology", async () => {
        const { snapshot } = await load(doc(NESTED));
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        const parent = snapshot.nodes.require("parent");
        expect(parent.meta).toMatchObject({ dtype: "u32", role: "parent", refersTo: "node" });
        expect(column(snapshot, "nodes", "parent")).toEqual([undefined, 0, 0, 2, undefined]);
        expect(snapshot.edgeCount).toBe(1);
    });

    it("skips hyperedges with one warning by default and refuses them under error", async () => {
        const body = `<node id="a"/><node id="b"/><hyperedge id="h"><endpoint node="a"/><endpoint node="b"/></hyperedge><hyperedge><endpoint node="a"/></hyperedge>`;
        const skipped = await load(doc(body));
        expect(skipped.snapshot.edgeCount).toBe(0);
        expect(codes(skipped.report)).toEqual([GRAPHML_ISSUE.HYPEREDGE_SKIPPED]);
        expect(skipped.report.counts.skippedEdges).toBe(2);
        const err = await importError(doc(body), { hyperedges: "error" });
        expect(codes(err.report)).toEqual([GRAPHML_ISSUE.HYPEREDGE]);
        expect(err.report.issues[0]).toMatchObject({ category: "unsupported", element: "h" });
    });

    it("expands hyperedges to stars with a marked hub node", async () => {
        const body = `<node id="a"/><node id="b"/><node id="c"/><node id="h"/>
<hyperedge id="h"><endpoint node="a" type="in"/><endpoint node="b" type="out"/><endpoint node="c"/><data key="x">1</data></hyperedge>
<hyperedge><endpoint node="a"/><endpoint node="b"/></hyperedge>`;
        const { snapshot, report } = await load(doc(body), { hyperedges: "star" });
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "h", "h#2", "hyperedge2"]);
        expect(column(snapshot, "nodes", "graphty.hyperedge")).toEqual([
            undefined,
            undefined,
            undefined,
            undefined,
            true,
            true,
        ]);
        const list = snapshot.edgeList();
        const edges = Array.from(list.src).map(
            (s, e) => `${String(snapshot.ids.idOf(s))}->${String(snapshot.ids.idOf(list.dst[e]))}`,
        );
        expect(edges).toEqual([
            "a->h#2",
            "h#2->b",
            "c->h#2",
            "h#2->c",
            "a->hyperedge2",
            "hyperedge2->a",
            "b->hyperedge2",
            "hyperedge2->b",
        ]);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        ]);
        expect(codes(report)).toEqual([GRAPHML_ISSUE.HYPEREDGE_DATA_DROPPED]);
        expect(report.counts.nodes).toBe(6);
    });

    it("expands hyperedges to cliques", async () => {
        const body = `<node id="a"/><node id="b"/><node id="c"/><node id="d"/>
<hyperedge><endpoint node="a" type="in"/><endpoint node="b" type="in"/><endpoint node="c" type="out"/><endpoint node="d"/></hyperedge>`;
        const { snapshot } = await load(doc(body, "undirected"), { hyperedges: "clique" });
        const list = snapshot.edgeList();
        const edges = Array.from(list.src).map(
            (s, e) => `${String(snapshot.ids.idOf(s))}->${String(snapshot.ids.idOf(list.dst[e]))}`,
        );
        // in/in pairs are not joined; in->out is directed; undir joins everything
        expect(edges).toEqual(["a->c", "a->d", "d->a", "b->c", "b->d", "d->b", "c->d", "d->c"]);
    });

    it("reports bad endpoints and skips the hyperedge", async () => {
        const body = `<hyperedge><endpoint/></hyperedge><hyperedge><endpoint node="a" type="sideways"/></hyperedge>`;
        const { snapshot, report } = await load(doc(body), { hyperedges: "clique" });
        expect(snapshot.edgeCount).toBe(0);
        expect(codes(report)).toEqual([GRAPHML_ISSUE.HYPEREDGE_ENDPOINT, GRAPHML_ISSUE.HYPEREDGE_ENDPOINT]);
        expect(report.counts.skippedEdges).toBe(2);
    });

    it('skips yFiles keys under yfiles: "skip" and rejects other values', async () => {
        const { snapshot, report } = await load(readCorpusText("graphml", "yfiles-sample.graphml"), { yfiles: "skip" });
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.edges.names()).toEqual(["id"]);
        expect(codes(report)).toEqual([GRAPHML_ISSUE.YFILES_SKIPPED, GRAPHML_ISSUE.YFILES_SKIPPED]);
        expect(report.lossy).toEqual([]);
        await expect(load("<graphml/>", { yfiles: "xml" } as unknown as Options)).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("keeps a text-only yfiles value and a yfiles default as json", async () => {
        const keys = `<key id="d0" for="node" yfiles.type="nodegraphics"><default><y:ShapeNode><y:Shape type="rectangle"/></y:ShapeNode></default></key>\n`;
        const body = `<node id="a"><data key="d0">just text</data></node><node id="b"/>`;
        const { snapshot } = await load(doc(body, "directed", keys));
        const graphics = snapshot.nodes.require("d0");
        expect(graphics.meta.default).toEqual({ "y:ShapeNode": { "y:Shape": { "@_type": "rectangle" } } });
        expect(graphics.value(0)).toBe("just text");
        expect(graphics.value(1)).toEqual({ "y:ShapeNode": { "y:Shape": { "@_type": "rectangle" } } });
    });
});

describe("graphmlImporter mangled ids", () => {
    const MANGLED = doc(
        `<node id="Little_Rock__AR"><data key="o">Little Rock, AR</data></node><node id="b"/><edge source="Little_Rock__AR" target="b"/>`,
        "directed",
        `<key id="o" for="node" attr.name="graphty:originalId" attr.type="string"/>\n`,
    );

    it("restores mangled ids by default, including edge endpoints", async () => {
        const { snapshot, report } = await load(MANGLED);
        expect(snapshot.ids.toArray()).toEqual(["Little Rock, AR", "b"]);
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.nodes.names()).toEqual([]);
        expect(report.issues).toEqual([]);
    });

    it("keeps the mangled id and the original as a role column under restoreMangledIds: false", async () => {
        const { snapshot } = await load(MANGLED, { restoreMangledIds: false });
        expect(snapshot.ids.toArray()).toEqual(["Little_Rock__AR", "b"]);
        const original = snapshot.nodes.require("graphty.originalId");
        expect(original.meta.role).toBe("originalId");
        expect(column(snapshot, "nodes", "graphty.originalId")).toEqual(["Little Rock, AR", undefined]);
    });
});

describe("graphmlImporter options and sniff", () => {
    it("rejects invalid common options up front", async () => {
        await expect(load("<graphml/>", { ids: "guess" } as unknown as Options)).rejects.toBeInstanceOf(
            GraphFormatError,
        );
    });

    it("sniffs GraphML heads", () => {
        const { sniff } = graphmlImporter;
        if (sniff === undefined) {
            throw new Error("no sniff");
        }
        const encode = (text: string): Uint8Array => new TextEncoder().encode(text);
        expect(sniff(encode(readCorpusText("graphml", "simple.graphml").slice(0, 200)))).toBe(1);
        expect(sniff(encode("<graphml><graph/></graphml>"))).toBe(0.9);
        expect(sniff(encode('<?xml version="1.0"?>\n<gexf/>'))).toBe(0.05);
        expect(sniff(encode("source,target\n1,2\n"))).toBe(0);
        expect(sniff(encode(""))).toBe(0);
    });
});
