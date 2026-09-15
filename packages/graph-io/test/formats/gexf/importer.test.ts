import { type Column, GraphBuilder, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { XML_SYNTAX_CODE } from "../../../src/common/codes.js";
import { DIRECTION_FORCED_CODE, DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import {
    ATTRIBUTE_ID_CODE,
    ATTRIBUTE_TYPE_CODE,
    ATTRIBUTES_CLASS_CODE,
    ATTVALUE_SHAPE_CODE,
    DUPLICATE_ATTRIBUTE_CODE,
    DUPLICATE_NODE_CODE,
    EDGE_TYPE_CODE,
    gexfImporter,
    type GexfImportOptions,
    HEADER_VALUE_CODE,
    ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    MISSING_NODES_CODE,
    NO_GRAPH_CODE,
    NOT_GEXF_CODE,
    SPELL_OPEN_CODE,
    TIMED_STATIC_CODE,
    UNKNOWN_ATTRIBUTE_CODE,
    UNKNOWN_PARENT_CODE,
    VIZ_DYNAMIC_CODE,
    VIZ_SKIPPED_CODE,
    VIZ_VALUE_CODE,
    WEIGHT_IGNORED_CODE,
} from "../../../src/formats/gexf/importer.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../../src/types.js";
import {
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
} from "../../helpers/corpus.js";
import { ACCENTED, BARE, DYNAMIC_1_3, MIXED_UNDIRECTED_HEADER, OPEN_1_2, SLOPPY } from "./fixtures.js";

/** Node 119 of airlines-sample.gexf carries a double-encoded accent in the source file; it is read as written. */
const MOJIBAKE_LABEL = `Louis Armstrong New Orl${String.fromCharCode(0xc3, 0xa9)}ans International Airport`;

type Options = (GexfImportOptions & CommonImportOptions) | undefined;

async function load(
    input: Parameters<typeof gexfImporter.import>[0],
    options?: Options,
    builder?: GraphBuilder,
): Promise<{ snapshot: GraphSnapshot; report: ImportReport; builder: GraphBuilder }> {
    const sink = builder ?? new GraphBuilder({ directed: true, weightDtype: "f64" });
    const report = await gexfImporter.import(input, sink, options);
    return { snapshot: sink.freeze(), report, builder: sink };
}

function values(column: Column | null): unknown[] {
    if (column === null) {
        throw new Error("column missing");
    }
    return Array.from({ length: column.length }, (_, r) => {
        if (!column.isSet(r)) {
            return undefined;
        }
        if (column.dtype === "list") {
            return column.sliceOf(r).map((item) => (ArrayBuffer.isView(item) ? Array.from(item as never) : item));
        }
        const v = column.value(r);
        return ArrayBuffer.isView(v) ? Array.from(v as never) : v;
    });
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

async function expectImportError(
    input: Parameters<typeof gexfImporter.import>[0],
    options?: Options,
): Promise<ImportError> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        await gexfImporter.import(input, builder, options);
    } catch (err) {
        expect(err).toBeInstanceOf(ImportError);
        const error = err as ImportError;
        expect(error.code).toBe("E_IMPORT");
        expect(error.report.format).toBe("gexf");
        return error;
    }
    throw new Error("expected an ImportError");
}

describe("gexfImporter: identity", () => {
    it("declares the format", () => {
        expect(gexfImporter.format).toBe("gexf");
        expect(gexfImporter.extensions).toEqual([".gexf"]);
        expect(gexfImporter.mimeTypes).toContain("application/gexf+xml");
    });

    it("sniffs a gexf head", () => {
        const enc = new TextEncoder();
        expect(gexfImporter.sniff?.(enc.encode('<?xml version="1.0"?>\n<gexf xmlns="http://gexf.net/1.3">'))).toBe(1);
        expect(
            gexfImporter.sniff?.(
                enc.encode('<?xml version="1.0"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'),
            ),
        ).toBe(0);
        expect(gexfImporter.sniff?.(enc.encode('<x xmlns="http://www.gexf.net/1.2draft"'))).toBe(0.8);
        expect(gexfImporter.sniff?.(readCorpusBytes("gexf", "lesmiserables.gexf").subarray(0, 200))).toBe(1);
        expect(gexfImporter.sniff?.(new Uint8Array([0xff, 0xfe, 0x00]))).toBe(0);
    });
});

describe("gexfImporter: corpus", () => {
    for (const file of corpusFiles("gexf")) {
        it(`imports ${file.path} with the manifest's counts`, async () => {
            const { snapshot, report } = await load(readCorpusText("gexf", file.path));
            expect(snapshot.nodeCount).toBe(file.expectedNodes);
            expect(snapshot.edgeCount).toBe(file.expectedEdges);
            expect(report.counts).toMatchObject({
                nodes: file.expectedNodes,
                edges: file.expectedEdges,
                skippedNodes: 0,
                skippedEdges: 0,
                expandedMixed: 0,
            });
            expect(report.errorCount).toBe(0);
            expect(report.truncated).toBe(false);
            expect(snapshot.directed).toBe(false);
            expect(snapshot.edges.byRole("pair")).toBeNull();
            expect(snapshot.meta.sourceFormat).toBe("gexf");
        });
    }

    it("reads minimal.gexf: canonical integer ids, labels, edge ids, meta", async () => {
        const { snapshot, report } = await load(readCorpusText("gexf", "minimal.gexf"));
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(values(snapshot.nodes.byRole("label"))).toEqual(["A", "B", "C"]);
        expect(values(snapshot.edges.byRole("id"))).toEqual(["0", "1"]);
        expect(snapshot.edges.byRole("id")?.meta.unique).toBe(true);
        expect(snapshot.flags.weighted).toBe(false);
        expect(snapshot.meta).toMatchObject({
            creator: "Graphty Test Corpus",
            description: "Minimal GEXF test file",
            modified: "2025-01-01",
            sourceVersion: "1.2",
            idType: null,
        });
        expect(report.issues).toEqual([]);
        const list = snapshot.edgeList();
        expect(Array.from(list.src)).toEqual([0, 1]);
        expect(Array.from(list.dst)).toEqual([1, 2]);
    });

    it("reads lesmiserables.gexf: float-looking string ids, a defaulted weight on edge 0, explicit weights after", async () => {
        const { snapshot } = await load(readCorpusText("gexf", "lesmiserables.gexf"));
        expect(snapshot.ids.idOf(0)).toBe("0.0");
        expect(snapshot.ids.idOf(11)).toBe("11.0");
        expect(snapshot.nodes.value("label", 11)).toBe("Valjean");
        expect(snapshot.flags.weighted).toBe(true);
        const weight = snapshot.edges.byRole("weight");
        expect(weight).not.toBeNull();
        expect(weight?.isSet(0)).toBe(false);
        expect(weight?.isSet(1)).toBe(true);
        expect(weight?.value(1)).toBe(8);
        expect(weight?.value(2)).toBe(10);
        expect(snapshot.edgeList().weights?.[0]).toBe(1);
        expect(snapshot.meta).toMatchObject({
            creator: "Gephi 0.7",
            idType: "string",
            mode: "static",
            sourceVersion: "1.1",
        });
        // edge ids in the file are not in document order ("13" precedes "12") and some are absent
        expect(snapshot.edgeIndexOf("9")).toBe(9);
        expect(snapshot.edgeIndexOf("13")).toBe(10);
        expect(snapshot.edgeIndexOf("12")).toBe(11);
        expect(snapshot.edgeIndexOf("999")).toBe(INVALID_INDEX);
    });

    it("reads airlines-sample.gexf: typed attributes, viz colours, non-ASCII labels", async () => {
        const { snapshot } = await load(readCorpusText("gexf", "airlines-sample.gexf"));
        expect(snapshot.nodes.names()).toEqual(["Code", "City", "latitude", "longitude", "label", "color"]);
        expect(snapshot.nodes.get("Code")?.meta).toMatchObject({
            dtype: "string",
            origin: { format: "gexf", id: "code", title: null, type: "string", namespace: null },
        });
        expect(snapshot.nodes.get("latitude")?.dtype).toBe("f64");
        expect(snapshot.nodes.value("Code", 0)).toBe("LIT");
        expect(snapshot.nodes.value("City", 0)).toBe("Little Rock, AR");
        expect(snapshot.nodes.value("latitude", 0)).toBe(34.729444);
        expect(snapshot.nodes.value("longitude", 2)).toBe(-73.8);
        const color = snapshot.nodes.byRole("color");
        expect(color?.meta).toMatchObject({ dtype: "f32", components: 4, origin: { namespace: "viz" } });
        expect(Array.from(color?.value(0) as ArrayLike<number>)).toEqual([
            Math.fround(88 / 255),
            Math.fround(107 / 255),
            Math.fround(243 / 255),
            1,
        ]);
        expect(snapshot.nodes.value("label", 119)).toBe(MOJIBAKE_LABEL);
        expect(snapshot.ids.kind).not.toBe("string");
        expect(snapshot.ids.idOf(234)).toBe(234);
    });

    it("reads the same graph from every input shape (streaming decode across chunk boundaries)", async () => {
        const bytes = readCorpusBytes("gexf", "airlines-sample.gexf");
        for (const shape of inputShapes(bytes)) {
            const progress: number[] = [];
            const { snapshot, report } = await load(shape.make(), {
                onProgress: (done) => {
                    progress.push(done);
                },
            });
            expect(snapshot.nodeCount, shape.name).toBe(235);
            expect(snapshot.edgeCount, shape.name).toBe(1297);
            expect(snapshot.nodes.value("label", 119), shape.name).toBe(MOJIBAKE_LABEL);
            expect(report.errorCount, shape.name).toBe(0);
            expect(progress.length, shape.name).toBeGreaterThan(0);
        }
    });
});

describe("gexfImporter: a 1.3 dynamic document", () => {
    it("imports every construct into the design's columns and tables", async () => {
        const { snapshot, report } = await load(DYNAMIC_1_3);
        expect(report.errorCount).toBe(0);
        expect(codes(report)).toEqual(["W_COLUMN_RENAMED", "W_PRECISION"]);
        expect(report.issues[0]).toMatchObject({ category: "coercion", element: "8", line: 18 });
        expect(report.issues[1]).toMatchObject({ category: "precision", element: "a", line: 41 });
        expect(report.counts).toEqual({ nodes: 5, edges: 8, skippedNodes: 0, skippedEdges: 0, expandedMixed: 3 });

        // header and meta
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta).toMatchObject({
            sourceVersion: "1.3",
            creator: "graph-io tests",
            description: `a & b < c ${String.fromCharCode(0xe9)} A`,
            keywords: ["alpha", "beta"],
            modified: "2024-05-01",
            timeFormat: "date",
            timeRepresentation: "interval",
            mode: "dynamic",
            idType: "string",
            weightOrigin: { format: "gexf", id: "w", title: "weight", type: "double", namespace: null },
            extra: { gexf: { start: "2020-01-01", end: "2021-01-01" } },
        });

        // ids and topology
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", 1]);
        const list = snapshot.edgeList();
        expect(Array.from(list.src)).toEqual([0, 1, 2, 2, 0, 0, 0, 3]);
        expect(Array.from(list.dst)).toEqual([1, 2, 1, 0, 2, 0, 1, 4]);
        expect(values(snapshot.edges.byRole("directed"))).toEqual([true, false, false, true, true, false, true, true]);
        expect(values(snapshot.edges.byRole("pair"))).toEqual([undefined, 2, 1, 4, 3, undefined, undefined, undefined]);
        expect(values(snapshot.edges.byRole("mutual"))).toEqual([
            undefined,
            undefined,
            undefined,
            true,
            undefined,
            undefined,
            undefined,
            undefined,
        ]);
        expect(snapshot.selfLoopCount).toBe(1);
        expect(snapshot.flags.multigraph).toBe(true);

        // weights: XML attribute, defaulted, static attribute override
        expect(Array.from(list.weights ?? [])).toEqual([2.5, 1, 1, 1, 1, 1, 1, 7]);
        const weight = snapshot.edges.byRole("weight");
        expect(values(weight)).toEqual([2.5, undefined, undefined, 1, 1, undefined, undefined, 7]);
        const weightTable = snapshot.extensions.get("temporal:edge:weight");
        expect(weightTable?.rowCount).toBe(1);
        expect(weightTable?.value("element", 0)).toBe(0);
        expect(weightTable?.value("value", 0)).toBe(3.5);
        expect(weightTable?.value("start", 0)).toBe(Date.UTC(2020, 5, 1));
        expect(weightTable?.value("end", 0)).toBe(Date.UTC(2020, 6, 1));

        // declared node attributes in declaration order, then the XML-derived columns; the text
        // companion of a temporal column is declared on the first value whose text is not
        // canonical (design 5.1), so it follows the columns used before that value
        expect(snapshot.nodes.names()).toEqual([
            "name",
            "age",
            "score",
            "active",
            "tags",
            "born",
            "big",
            "cat",
            "label#8",
            "ratio",
            "nums",
            "url",
            "price",
            "label",
            "start",
            "end",
            "color",
            "position",
            "size",
            "shape",
            "timestamps",
            "born.text",
            "spells",
            "parent",
            "parents",
        ]);
        const dtypes = Object.fromEntries([...snapshot.nodes].map((c) => [c.meta.name, c.meta.dtype]));
        expect(dtypes).toMatchObject({
            name: "string",
            age: "i32",
            score: "f64",
            active: "bool",
            tags: "list",
            born: "f64",
            "born.text": "string",
            big: "f64",
            cat: "dict",
            "label#8": "string",
            ratio: "f32",
            nums: "list",
            url: "string",
            price: "f64",
            label: "string",
            start: "f64",
            end: "f64",
            color: "f32",
            position: "f32",
            size: "f32",
            shape: "dict",
            timestamps: "list",
            spells: "list",
            parent: "u32",
            parents: "list",
        });
        expect(snapshot.nodes.get("name")?.meta.default).toBe("anon");
        expect(snapshot.nodes.get("age")?.meta.default).toBe(0);
        expect(snapshot.nodes.get("active")?.meta.default).toBe(true);
        expect(snapshot.nodes.get("cat")?.meta.options).toEqual(["a", "b", "c"]);
        expect(snapshot.nodes.get("cat")?.meta.default).toBe("b");
        expect(snapshot.nodes.get("label#8")?.meta.origin).toEqual({
            format: "gexf",
            id: "8",
            title: "label",
            type: "string",
            namespace: null,
        });
        expect(snapshot.nodes.get("url")?.meta.origin?.type).toBe("anyURI");
        expect(snapshot.nodes.get("nums")?.meta.itemDtype).toBe("i32");
        expect(snapshot.nodes.get("price")?.meta.dynamic).toBe(true);
        expect(snapshot.nodes.get("born.text")?.meta).toMatchObject({ role: "timeText", extra: { for: "born" } });

        // node a's values
        expect(snapshot.nodes.value("name", 0)).toBe("Alice & co");
        expect(snapshot.nodes.value("name", 1)).toBe("anon");
        expect(snapshot.nodes.isSet("name", 1)).toBe(false);
        expect(snapshot.nodes.value("age", 0)).toBe(30);
        expect(snapshot.nodes.value("score", 0)).toBe(1.5);
        expect(snapshot.nodes.value("active", 0)).toBe(false);
        expect(snapshot.nodes.value("active", 2)).toBe(true);
        expect(values(snapshot.nodes.get("tags"))[0]).toEqual(["x", "y, z", "w"]);
        expect(snapshot.nodes.value("born", 0)).toBe(Date.UTC(1990, 4, 6));
        expect(snapshot.nodes.isSet("born.text", 0)).toBe(false);
        expect(snapshot.nodes.value("born", 1)).toBe(Date.UTC(2001, 1, 3, 2, 5, 6));
        expect(snapshot.nodes.value("born.text", 1)).toBe("2001-02-03T04:05:06+02:00");
        expect(snapshot.nodes.value("big", 0)).toBe(9007199254740992);
        expect(snapshot.nodes.value("cat", 0)).toBe("c");
        expect(snapshot.nodes.value("cat", 3)).toBe("b");
        expect(snapshot.nodes.value("label#8", 0)).toBe("renamed");
        expect(snapshot.nodes.value("ratio", 0)).toBe(0.25);
        expect(values(snapshot.nodes.get("nums"))[0]).toEqual([1, 2, 3]);
        expect(snapshot.nodes.value("url", 0)).toBe("http://example.com/?q=1&r=2");
        expect(snapshot.nodes.value("price", 0)).toBe(11);
        expect(values(snapshot.nodes.get("label"))).toEqual(["Alice", ACCENTED, "Carol", "Dan", "One"]);

        // dynamic values
        const price = snapshot.extensions.get("temporal:node:price");
        expect(price).toBeDefined();
        expect(price?.names()).toEqual(["element", "start", "end", "value", "start.text", "end.text", "open"]);
        expect(price?.get("element")?.meta).toMatchObject({ dtype: "u32", refersTo: "node" });
        expect(price?.get("start")?.meta.role).toBe("start");
        expect(price?.get("start.text")?.meta.role).toBe("timeText");
        expect(values(price?.get("element") ?? null)).toEqual([0, 0]);
        expect(values(price?.get("start") ?? null)).toEqual([Date.UTC(2020, 0, 1), Date.UTC(2020, 6, 1)]);
        expect(values(price?.get("end") ?? null)).toEqual([Date.UTC(2020, 5, 30), Infinity]);
        expect(values(price?.get("value") ?? null)).toEqual([10, 12.5]);
        expect(values(price?.get("open") ?? null)).toEqual([undefined, undefined]);

        // lifetimes
        expect(snapshot.nodes.value("start", 0)).toBe(Date.UTC(2020, 0, 1));
        expect(snapshot.nodes.value("end", 0)).toBe(Date.UTC(2020, 11, 31));
        expect(snapshot.nodes.isSet("start", 1)).toBe(false);
        expect(values(snapshot.nodes.get("timestamps"))[1]).toEqual([Date.UTC(2020, 1, 1), Date.UTC(2020, 2, 1)]);
        expect(values(snapshot.nodes.get("spells"))[1]).toEqual([
            [Date.UTC(2020, 0, 1), Date.UTC(2020, 1, 1)],
            [Date.UTC(2020, 2, 1), Infinity],
        ]);
        expect(snapshot.nodes.get("spells")?.meta).toMatchObject({
            role: "spells",
            itemDtype: "f64",
            itemComponents: 2,
        });

        // viz
        expect(values(snapshot.nodes.get("color"))[0]).toEqual([1, 0, 0, 0.5]);
        expect(values(snapshot.nodes.get("color"))[1]).toEqual([0, 1, 0, 1]);
        expect(values(snapshot.nodes.get("position"))[0]).toEqual([1.5, -2, 3]);
        expect(values(snapshot.nodes.get("position"))[1]).toEqual([0, 0, 0]);
        expect(snapshot.nodes.get("position")?.meta).toMatchObject({
            role: "position",
            components: 3,
            mutable: true,
            extra: { units: "file", sourceDims: 3 },
            origin: { format: "gexf", namespace: "viz" },
        });
        expect(snapshot.nodes.value("size", 0)).toBe(2.5);
        expect(snapshot.nodes.value("shape", 0)).toBe("square");

        // containment: pid, nesting and <parents>
        expect(values(snapshot.nodes.get("parent"))).toEqual([undefined, 0, undefined, 2, undefined]);
        expect(snapshot.nodes.get("parent")?.meta).toMatchObject({ role: "parent", refersTo: "node" });
        expect(values(snapshot.nodes.get("parents"))).toEqual([undefined, undefined, [0, 1], undefined, undefined]);

        // edge columns
        expect(snapshot.edges.names()).toContain("rel");
        expect(values(snapshot.edges.byRole("id"))).toEqual(["e1", "e2", undefined, "e3", undefined, "e4", "e5", "e6"]);
        expect(values(snapshot.edges.byRole("label"))[0]).toBe("knows");
        expect(values(snapshot.edges.byRole("kind"))).toEqual([
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            undefined,
            "second",
            undefined,
        ]);
        expect(snapshot.edges.value("rel", 0)).toBe("friend");
        expect(snapshot.edges.value("start", 0)).toBe(Date.UTC(2020, 0, 1));
        expect(values(snapshot.edges.get("color"))[0]).toEqual([0, 0, 1, 1]);
        expect(snapshot.edges.value("thickness", 0)).toBe(3);
        expect(snapshot.edges.value("shape", 0)).toBe("dotted");
        // mirror halves carry no attribute values
        expect(snapshot.edges.isSet("rel", 2)).toBe(false);
        expect(snapshot.edges.isSet("id", 2)).toBe(false);
    });

    it("skips viz elements under viz: false", async () => {
        const { snapshot, report } = await load(DYNAMIC_1_3, { viz: false });
        expect(snapshot.nodes.byRole("color")).toBeNull();
        expect(snapshot.nodes.byRole("position")).toBeNull();
        expect(snapshot.edges.byRole("thickness")).toBeNull();
        expect(codes(report)).toContain(VIZ_SKIPPED_CODE);
        expect(codes(report).filter((c) => c === VIZ_SKIPPED_CODE)).toHaveLength(1);
    });

    it('keeps long columns as text under long: "string"', async () => {
        const { snapshot, report } = await load(DYNAMIC_1_3, { long: "string" });
        expect(snapshot.nodes.get("big")?.dtype).toBe("string");
        expect(snapshot.nodes.value("big", 0)).toBe("9007199254740993");
        expect(codes(report)).not.toContain("W_PRECISION");
    });

    it("treats the weight attribute as a column under a different weightFrom", async () => {
        const { snapshot, report } = await load(DYNAMIC_1_3, { weightFrom: "cost" });
        expect(report.errorCount).toBe(0);
        expect(snapshot.edges.get("weight")?.meta).toMatchObject({ dtype: "f64", dynamic: true });
        expect(snapshot.extensions.has("temporal:edge:weight")).toBe(true);
        expect(snapshot.extensions.get("temporal:edge:weight")?.get("value")?.meta.origin?.id).toBe("w");
        expect(snapshot.edges.value("weight", 7)).toBe(7);
        expect(snapshot.meta.weightOrigin).toBeNull();
        // the XML weight attribute still applies; the static attvalue of edge e6 no longer overrides it
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2.5, 1, 1, 1, 1, 1, 1, 1]);
    });

    it("forces every edge one way under onMixedDirection directed / undirected", async () => {
        const directed = await load(DYNAMIC_1_3, { onMixedDirection: "directed" });
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edgeCount).toBe(6);
        expect(directed.snapshot.edges.byRole("pair")).toBeNull();
        expect(codes(directed.report)).toContain(DIRECTION_FORCED_CODE);
        const undirected = await load(DYNAMIC_1_3, { onMixedDirection: "undirected" });
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edgeCount).toBe(6);
        expect(codes(undirected.report)).toContain(DIRECTION_FORCED_CODE);
    });

    it("refuses a mixed file under onMixedDirection error", async () => {
        const error = await expectImportError(DYNAMIC_1_3, { onMixedDirection: "error" });
        expect(codes(error.report)).toContain(MIXED_DIRECTION_CODE);
        expect(error.report.counts.nodes).toBe(5);
    });
});

describe("gexfImporter: a 1.2 document", () => {
    it("reads integer times, open bounds, pipe lists, a mutual default and dynamic values", async () => {
        const { snapshot, report } = await load(OPEN_1_2);
        expect(report.errorCount).toBe(0);
        expect(codes(report)).toEqual([VIZ_DYNAMIC_CODE, SPELL_OPEN_CODE]);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta).toMatchObject({ timeFormat: "integer", mode: "dynamic", sourceVersion: "1.2" });
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(snapshot.edgeCount).toBe(3);
        expect(values(snapshot.edges.byRole("mutual"))).toEqual([true, undefined, undefined]);
        expect(values(snapshot.edges.byRole("pair"))).toEqual([1, 0, undefined]);
        expect(values(snapshot.edges.byRole("directed"))).toEqual([true, true, true]);
        expect(values(snapshot.edges.byRole("weight"))).toEqual([0.1, 0.1, undefined]);
        expect(snapshot.edges.byRole("weight")?.dtype).toBe("f64");
        expect(values(snapshot.nodes.get("tags"))[0]).toEqual(["x", "y", "z"]);
        expect(snapshot.nodes.value("flag", 0)).toBe(true);
        expect(snapshot.nodes.value("start", 0)).toBe(1);
        expect(snapshot.nodes.value("end", 0)).toBe(5);
        expect(snapshot.nodes.value("end", 1)).toBe(9);
        expect(values(snapshot.nodes.get("open"))).toEqual([1, 2]);
        expect(snapshot.nodes.get("open")?.meta).toMatchObject({ dtype: "u8", role: "open" });
        expect(values(snapshot.edges.get("open"))).toEqual([undefined, undefined, 2]);
        expect(values(snapshot.nodes.get("spells"))[1]).toEqual([[1, 2]]);
        const level = snapshot.extensions.get("temporal:node:level");
        expect(values(level?.get("value") ?? null)).toEqual([3, 4]);
        expect(values(level?.get("open") ?? null)).toEqual([2, undefined]);
        expect(values(level?.get("end") ?? null)).toEqual([2, Infinity]);
        expect(values(snapshot.nodes.get("color"))[0]).toEqual([
            Math.fround(1 / 255),
            Math.fround(2 / 255),
            Math.fround(3 / 255),
            1,
        ]);
    });
});

describe("gexfImporter: headers and defaults", () => {
    it("assumes undirected when defaultedgetype is absent, and honours defaultDirected", async () => {
        const plain = await load(BARE);
        expect(plain.snapshot.directed).toBe(false);
        expect(plain.snapshot.ids.toArray()).toEqual(["x", "y", "01"]);
        expect(plain.snapshot.meta.sourceVersion).toBe("1.2");
        expect(plain.snapshot.meta.creator).toBeNull();
        expect(plain.snapshot.edges.byRole("id")).toBeNull();
        expect(values(plain.snapshot.edges.byRole("weight"))).toEqual([undefined, 2]);
        const directed = await load(BARE, { defaultDirected: true });
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edges.byRole("pair")).toBeNull();
    });

    it("expands an undirected-header file with directed and mutual edges once", async () => {
        const { snapshot, report } = await load(MIXED_UNDIRECTED_HEADER);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(5);
        expect(report.counts.expandedMixed).toBe(2);
        expect(values(snapshot.edges.byRole("directed"))).toEqual([false, false, true, true, true]);
        expect(values(snapshot.edges.byRole("pair"))).toEqual([1, 0, undefined, 4, 3]);
        expect(values(snapshot.edges.byRole("mutual"))).toEqual([undefined, undefined, undefined, true, undefined]);
        expect(values(snapshot.edges.byRole("id"))).toEqual(["u1", undefined, "d1", "m1", undefined]);
    });

    it("coerces ids per the option", async () => {
        const doc = `<gexf version="1.3"><graph><nodes><node id="1"/><node id="01"/><node id="x"/></nodes><edges><edge source="1" target="x"/></edges></graph></gexf>`;
        const canonical = await load(doc);
        expect(canonical.snapshot.ids.toArray()).toEqual([1, "01", "x"]);
        const strings = await load(doc, { ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["1", "01", "x"]);
        const keep = await load(doc, { ids: "keep" });
        expect(keep.snapshot.ids.toArray()).toEqual(["1", "01", "x"]);
        const numeric = await load(doc, { ids: "number", errorLimit: Infinity });
        expect(numeric.snapshot.ids.toArray()).toEqual([1]);
        expect(codes(numeric.report)).toContain(ID_MERGED_CODE);
        expect(codes(numeric.report)).toContain("E_INVALID_ID");
        expect(numeric.report.counts.skippedNodes).toBe(1);
    });

    it("ignores the weight attribute under weightFrom: null with one warning", async () => {
        const { snapshot, report } = await load(BARE, { weightFrom: null });
        expect(snapshot.flags.weighted).toBe(false);
        expect(codes(report)).toEqual([WEIGHT_IGNORED_CODE]);
    });

    it("creates missing endpoints only under addMissingNodes: true", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed"><nodes><node id="a"/></nodes><edges><edge source="a" target="ghost"/></edges></graph></gexf>`;
        const strict = await load(doc);
        expect(strict.snapshot.nodeCount).toBe(1);
        expect(strict.snapshot.edgeCount).toBe(0);
        expect(strict.report.issues).toHaveLength(1);
        expect(strict.report.issues[0]).toMatchObject({
            category: "missing-value",
            code: "E_UNKNOWN_NODE",
            element: "a->ghost",
        });
        expect(strict.report.counts.skippedEdges).toBe(1);
        const lenient = await load(doc, { addMissingNodes: true });
        expect(lenient.snapshot.nodeCount).toBe(2);
        expect(lenient.snapshot.edgeCount).toBe(1);
        expect(lenient.report.issues).toEqual([]);
    });

    it("reads a slice-mode 1.3 graph with timestamp representation", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" mode="slice" timeformat="integer" timerepresentation="timestamp">
            <attributes class="node" mode="dynamic"><attribute id="v" title="v" type="integer"/></attributes>
            <nodes><node id="a" timestamp="3"><attvalues><attvalue for="v" value="1" timestamp="3"/><attvalue for="v" value="2" timestamp="4"/></attvalues></node><node id="b" timestamp="5"/></nodes>
            <edges><edge source="a" target="b" timestamp="4"/></edges></graph></gexf>`;
        const { snapshot } = await load(doc);
        expect(snapshot.meta).toMatchObject({ mode: "slice", timeRepresentation: "timestamp", timeFormat: "integer" });
        expect(values(snapshot.nodes.byRole("timestamp"))).toEqual([3, 5]);
        expect(values(snapshot.edges.byRole("timestamp"))).toEqual([4]);
        const table = snapshot.extensions.get("temporal:node:v");
        expect(values(table?.get("start") ?? null)).toEqual([3, 4]);
        expect(values(table?.get("end") ?? null)).toEqual([3, 4]);
    });

    it("stores timed values of a static attribute as dynamic with one warning", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" timeformat="double">
            <attributes class="edge" mode="static"><attribute id="w" title="weight" type="double"/><attribute id="c" title="c" type="string"/></attributes>
            <nodes><node id="a"/><node id="b"/></nodes>
            <edges><edge source="a" target="b" weight="9"><attvalues><attvalue for="w" value="2"/><attvalue for="w" value="3" start="1"/><attvalue for="c" value="x" start="1" end="2"/><attvalue for="c" value="y"/></attvalues></edge></edges></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(codes(report)).toEqual([TIMED_STATIC_CODE]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2]);
        expect(snapshot.edges.byRole("weight")).toBeNull();
        expect(snapshot.meta.weightOrigin).toMatchObject({ id: "w", title: "weight", type: "double" });
        expect(values(snapshot.extensions.get("temporal:edge:weight")?.get("value") ?? null)).toEqual([3]);
        expect(snapshot.edges.value("c", 0)).toBe("y");
        expect(snapshot.edges.get("c")?.meta.dynamic).toBe(false);
        expect(values(snapshot.extensions.get("temporal:edge:c")?.get("value") ?? null)).toEqual(["x"]);
        expect(values(snapshot.extensions.get("temporal:edge:c")?.get("start.text") ?? null)).toEqual([undefined]);
    });

    it("resolves a pid that precedes the parent's declaration and decodes entities", async () => {
        const doc = `<gexf version="1.3"><graph><nodes>
            <node id="child" pid="root" label="line&#10;break &amp; &lt;tag&gt; &#x1F600;"/>
            <node id="root"><attvalues/></node>
            </nodes><edges/></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(report.issues).toEqual([]);
        expect(values(snapshot.nodes.byRole("parent"))).toEqual([1, undefined]);
        expect(snapshot.nodes.value("label", 0)).toBe(`line\nbreak & <tag> ${String.fromCodePoint(0x1f600)}`);
    });

    it("reads a CDATA description and keywords", async () => {
        const doc = `<gexf version="1.2"><meta><description><![CDATA[a < b &amp; c]]> tail</description><keywords> one , two,,</keywords></meta><graph><nodes/></graph></gexf>`;
        const { snapshot } = await load(doc);
        expect(snapshot.meta.description).toBe("a < b &amp; c tail");
        expect(snapshot.meta.keywords).toEqual(["one", "two"]);
        expect(snapshot.nodeCount).toBe(0);
    });

    it("keeps a companion for date-typed values whose text is not canonical, and none otherwise", async () => {
        const doc = `<gexf version="1.3"><graph>
            <attributes class="node" mode="static"><attribute id="d" title="when" type="dateTime"/></attributes>
            <nodes><node id="a"><attvalues><attvalue for="d" value="2020-01-02T03:04:05Z"/></attvalues></node>
            <node id="b"><attvalues><attvalue for="d" value="2020-01-02"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot } = await load(doc);
        expect(snapshot.nodes.value("when", 0)).toBe(Date.UTC(2020, 0, 2, 3, 4, 5));
        expect(snapshot.nodes.isSet("when.text", 0)).toBe(false);
        expect(snapshot.nodes.value("when", 1)).toBe(Date.UTC(2020, 0, 2));
        expect(snapshot.nodes.value("when.text", 1)).toBe("2020-01-02");
    });

    it("reads the count hints and nested nodes into the parent column", async () => {
        const doc = `<gexf version="1.3"><graph><nodes count="3"><node id="a"><nodes count="2"><node id="b"/><node id="c" pid="b"/></nodes></node></nodes><edges count="0"/></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(report.counts.nodes).toBe(3);
        expect(values(snapshot.nodes.byRole("parent"))).toEqual([undefined, 0, 1]);
    });

    it("declares an attribute whose name the sink already holds with another shape under #id", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.declareNodeColumn({ name: "score", dtype: "string" });
        const doc = `<gexf version="1.3"><graph defaultedgetype="undirected">
            <attributes class="node" mode="static"><attribute id="s" title="score" type="double"/></attributes>
            <nodes><node id="a"><attvalues><attvalue for="s" value="1.5"/></attvalues></node></nodes></graph></gexf>`;
        const { snapshot, report } = await load(doc, undefined, builder);
        expect(codes(report)).toEqual(["W_COLUMN_RENAMED"]);
        expect(snapshot.nodes.get("score")?.dtype).toBe("string");
        expect(snapshot.nodes.get("score#s")?.dtype).toBe("f64");
        expect(snapshot.nodes.value("score#s", 0)).toBe(1.5);
    });

    it("reuses the same-shaped columns of a sink that already holds a file", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        await gexfImporter.import(readCorpusText("gexf", "airlines-sample.gexf"), builder);
        const second = `<gexf version="1.3"><graph defaultedgetype="undirected">
            <attributes class="node" mode="static"><attribute id="code" title="Code" type="string"/><attribute id="latitude" title="latitude" type="double"/></attributes>
            <nodes><node id="new" label="New"><attvalues><attvalue for="code" value="NEW"/><attvalue for="latitude" value="1.5"/></attvalues><viz:color r="0" g="0" b="0"/></node></nodes>
            <edges><edge source="new" target="0"/></edges></graph></gexf>`;
        const report = await gexfImporter.import(second, builder);
        expect(report.issues).toEqual([]);
        const snapshot = builder.freeze();
        expect(snapshot.nodeCount).toBe(236);
        expect(snapshot.edgeCount).toBe(1298);
        expect(snapshot.nodes.names()).toEqual(["Code", "City", "latitude", "longitude", "label", "color"]);
        expect(snapshot.nodes.value("Code", 235)).toBe("NEW");
        expect(snapshot.nodes.value("latitude", 235)).toBe(1.5);
        expect(snapshot.nodes.value("label", 235)).toBe("New");
    });
});

describe("gexfImporter: sink precedence (design 8.4)", () => {
    it("keeps a locked directed sink and expands an undirected file into it", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.lockDirected();
        const { snapshot, report } = await load(readCorpusText("gexf", "minimal.gexf"), undefined, builder);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(4);
        expect(codes(report)).toContain(DIRECTION_REFUSED_CODE);
        expect(report.counts.expandedMixed).toBe(2);
    });

    it("aborts when a locked undirected sink meets a directed edge under expand", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.lockDirected();
        await expect(gexfImporter.import(MIXED_UNDIRECTED_HEADER, builder)).rejects.toBeInstanceOf(ImportError);
    });

    it("reports builder options the sink does not honour", async () => {
        const builder = new GraphBuilder({
            directed: false,
            weightDtype: "f32",
            duplicateEdges: "sum",
            selfLoops: "drop",
            addMissingNodes: false,
        });
        const { report } = await load(
            readCorpusText("gexf", "minimal.gexf"),
            { addMissingNodes: true, duplicateEdges: "keep", selfLoops: "keep", weightDtype: "f64" },
            builder,
        );
        const sinkIssues = report.issues.filter((i) => i.code === SINK_OPTION_CODE).map((i) => i.element);
        expect(sinkIssues).toEqual(["addMissingNodes", "duplicateEdges", "selfLoops", "weightDtype"]);
        // addMissingNodes: false is enforced by the importer itself, so it is honoured on any sink;
        // options left undefined are defaults, never requests
        const lenient = new GraphBuilder({ directed: false, weightDtype: "f32", addMissingNodes: true });
        const silent = await load(readCorpusText("gexf", "minimal.gexf"), { addMissingNodes: false }, lenient);
        expect(silent.report.issues.filter((i) => i.code === SINK_OPTION_CODE)).toEqual([]);
    });
});

describe("gexfImporter: option validation", () => {
    it("reports nodeIdFrom other than id as an ignored option (design 8.4), and rejects a non-boolean viz and unknown enum values", async () => {
        const builder = new GraphBuilder({ directed: true });
        const report = await gexfImporter.import(BARE, builder, { nodeIdFrom: "label" });
        expect(report.issues.map((i) => [i.code, i.element])).toEqual([["W_OPTION_IGNORED", "nodeIdFrom"]]);
        await expect(gexfImporter.import(BARE, builder, { viz: "yes" as unknown as boolean })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
        await expect(
            gexfImporter.import(BARE, builder, { ids: "weird" as unknown as CommonImportOptions["ids"] }),
        ).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
    });

    it("rejects with the signal's reason when aborted before reading", async () => {
        const controller = new AbortController();
        controller.abort();
        const builder = new GraphBuilder({ directed: true });
        await expect(gexfImporter.import(BARE, builder, { signal: controller.signal })).rejects.toMatchObject({
            name: "AbortError",
        });
    });

    it("stops between phases when the signal fires during the read", async () => {
        const controller = new AbortController();
        const builder = new GraphBuilder({ directed: true });
        const bytes = new TextEncoder().encode(DYNAMIC_1_3);
        async function* chunks(): AsyncGenerator<Uint8Array> {
            yield bytes.subarray(0, 100);
            await Promise.resolve();
            controller.abort();
            yield bytes.subarray(100);
        }
        await expect(gexfImporter.import(chunks(), builder, { signal: controller.signal })).rejects.toMatchObject({
            name: "AbortError",
        });
    });
});

describe("gexfImporter: error aggregation (design 8.6)", () => {
    it("records every problem of a sloppy document and keeps going", async () => {
        const { snapshot, report } = await load(SLOPPY);
        expect(report.truncated).toBe(false);
        expect(snapshot.nodeCount).toBe(3);
        expect(snapshot.edgeCount).toBe(1);
        expect(report.counts).toEqual({ nodes: 3, edges: 1, skippedNodes: 0, skippedEdges: 4, expandedMixed: 0 });
        expect(codes(report)).toEqual([
            HEADER_VALUE_CODE,
            HEADER_VALUE_CODE,
            "W_BAD_DEFAULT",
            DUPLICATE_ATTRIBUTE_CODE,
            "W_UNKNOWN_ATTR_TYPE",
            ATTRIBUTE_TYPE_CODE,
            ATTRIBUTE_ID_CODE,
            ATTRIBUTES_CLASS_CODE,
            "E_COLUMN_TYPE",
            UNKNOWN_ATTRIBUTE_CODE,
            ATTVALUE_SHAPE_CODE,
            ATTVALUE_SHAPE_CODE,
            VIZ_VALUE_CODE,
            DUPLICATE_NODE_CODE,
            UNKNOWN_PARENT_CODE,
            EDGE_TYPE_CODE,
            "E_INVALID_WEIGHT",
            MISSING_ENDPOINT_CODE,
            "E_UNKNOWN_NODE",
        ]);
        expect(report.errorCount).toBe(8);
        expect(report.warningCount).toBe(11);
        expect(report.issues.find((i) => i.code === "E_COLUMN_TYPE")).toMatchObject({
            category: "validation-error",
            severity: "error",
            element: "a[0]",
            line: 17,
        });
        expect(report.issues.find((i) => i.code === UNKNOWN_PARENT_CODE)).toMatchObject({
            category: "missing-value",
            element: "b",
        });
        expect(snapshot.nodes.value("n", 0)).toBe(12);
        expect(snapshot.nodes.get("n")?.meta.default).toBeUndefined();
        expect(snapshot.nodes.value("t", 0)).toBe("kept as text");
        expect(snapshot.nodes.get("u")?.dtype).toBe("string");
        expect(snapshot.nodes.byRole("position")).toBeNull();
        expect(snapshot.meta.mode).toBeNull();
        expect(snapshot.meta.timeFormat).toBeNull();
    });

    it("aborts with the partial report once the error limit is exceeded", async () => {
        const error = await expectImportError(SLOPPY, { errorLimit: 2 });
        expect(error.report.truncated).toBe(true);
        expect(error.report.errorCount).toBe(3);
        expect(error.message).toContain("error limit");
        const zero = await expectImportError(SLOPPY, { errorLimit: 0 });
        expect(zero.report.errorCount).toBe(1);
        expect(zero.report.issues.filter((i) => i.severity === "error")).toHaveLength(1);
    });

    it("aborts on invalid UTF-8", async () => {
        const bytes = new Uint8Array([
            ...new TextEncoder().encode('<gexf version="1.3"><graph><nodes><node id="'),
            0xff,
            0xfe,
        ]);
        const error = await expectImportError(bytes);
        expect(codes(error.report)).toEqual([INVALID_UTF8_CODE]);
        expect(error.report.issues[0].category).toBe("parse-error");
    });

    it("aborts on a document whose root is not gexf", async () => {
        const error = await expectImportError(`<graphml><graph/></graphml>`);
        expect(codes(error.report)).toEqual([NOT_GEXF_CODE]);
    });
});

describe("gexfImporter: malformed corpus", () => {
    const expectedCodes: Record<string, string> = {
        "empty-file.gexf": XML_SYNTAX_CODE,
        "invalid-xml.gexf": XML_SYNTAX_CODE,
        "not-xml.gexf": XML_SYNTAX_CODE,
        "no-graph-element.gexf": NO_GRAPH_CODE,
        "missing-nodes-section.gexf": MISSING_NODES_CODE,
        "missing-node-id.gexf": MISSING_ID_CODE,
        "missing-edge-source.gexf": MISSING_ENDPOINT_CODE,
        "invalid-edge-reference.gexf": "E_UNKNOWN_NODE",
    };

    it("covers every malformed file", () => {
        expect([...malformedFiles("gexf")].sort()).toEqual(Object.keys(expectedCodes).sort());
    });

    for (const name of malformedFiles("gexf")) {
        it(`${name}: ImportError with the report under errorLimit 0, and the error recorded under the default limit`, async () => {
            const bytes = readMalformedBytes("gexf", name);
            const error = await expectImportError(bytes, { errorLimit: 0 });
            expect(error.report.errorCount).toBeGreaterThan(0);
            // the single pass may record a warning (an element GEXF does not define) before the
            // error that aborts; the first ERROR is the expected one
            const firstError = error.report.issues.find((i) => i.severity === "error");
            expect(firstError?.code).toBe(expectedCodes[name]);
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            let report: ImportReport;
            try {
                report = await gexfImporter.import(bytes, builder);
            } catch (err) {
                expect(err).toBeInstanceOf(ImportError);
                ({ report } = err as ImportError);
            }
            expect(report.errorCount).toBeGreaterThan(0);
            expect(report.issues.map((i) => i.code)).toContain(expectedCodes[name]);
        });
    }

    it("locates the syntax error of invalid-xml.gexf on its line", async () => {
        const error = await expectImportError(readMalformedBytes("gexf", "invalid-xml.gexf"));
        const syntax = error.report.issues.find((i) => i.code === XML_SYNTAX_CODE);
        expect(syntax).toMatchObject({ code: XML_SYNTAX_CODE, category: "parse-error", line: 8 });
        expect(syntax?.message).toContain("<node>");
    });

    it("skips a node without an id and the edges that reference it", async () => {
        const { snapshot, report } = await load(readMalformedBytes("gexf", "missing-node-id.gexf"));
        expect(snapshot.nodeCount).toBe(1);
        expect(snapshot.edgeCount).toBe(0);
        expect(report.counts).toMatchObject({ nodes: 1, skippedNodes: 1, skippedEdges: 1 });
        expect(codes(report)).toEqual([MISSING_ID_CODE, "E_UNKNOWN_NODE"]);
        expect(report.issues[0]).toMatchObject({ category: "missing-value", element: "A", line: 5 });
    });

    it("reports both unknown endpoints of invalid-edge-reference.gexf", async () => {
        const { report } = await load(readMalformedBytes("gexf", "invalid-edge-reference.gexf"));
        expect(codes(report)).toEqual(["E_UNKNOWN_NODE", "E_UNKNOWN_NODE"]);
        expect(report.issues.map((i) => i.element)).toEqual(["0", "1"]);
        expect(report.counts.skippedEdges).toBe(2);
    });
});

describe("gexf xml layer (the shared streaming tokenizer of src/common/xml.ts)", () => {
    it("reports unterminated constructs and stray roots", async () => {
        for (const [text, fragment] of [
            ["<gexf><!-- never closed", "end of input"],
            ["<gexf><![CDATA[ x", "end of input"],
            ["<?xml version='1.0'", "end of input"],
            ["<!DOCTYPE gexf [", "end of input"],
            ["<gexf/><gexf/>", "second root"],
            ["<gexf></gexf> trailing", "outside the root"],
            ["<gexf><graph a=b/></gexf>", "not quoted"],
            ["<gexf><graph a/></gexf>", "no value"],
            ['<gexf><graph a="x/></gexf>', "end of input"],
            ["<gexf><graph", "end of input"],
            ["<gexf></graph></gexf>", "does not match"],
            ["</gexf>", "unexpected end tag"],
            ["<gexf><graph /x></graph></gexf>", 'unexpected "/"'],
            ["<gexf>< graph/></gexf>", "expected an element name"],
            ['<gexf><graph ="1"/></gexf>', "malformed attribute"],
            ['<gexf><graph a="1" b/></gexf>', "no value"],
            ['<gexf><graph a="1" b=x/></gexf>', "not quoted"],
            ["<gexf><graph></graph", "end of input"],
        ] as const) {
            const error = await expectImportError(text);
            expect(error.report.issues[0].code, text).toBe(XML_SYNTAX_CODE);
            expect(error.report.issues[0].message, text).toContain(fragment);
        }
    });

    it("accepts comments, a DOCTYPE with an internal subset, processing instructions and a BOM", async () => {
        const doc = `${String.fromCharCode(0xfeff)}<?xml version="1.0"?>\n<!DOCTYPE gexf [ <!ENTITY x "y"> ]>\n<!-- c -->\n<gexf version="1.3"><?pi data?><graph><!-- inner --><nodes><node id="a"/></nodes></graph></gexf>\n<!-- trailing -->\n`;
        const { snapshot, report } = await load(doc);
        expect(report.issues).toEqual([]);
        expect(snapshot.ids.toArray()).toEqual(["a"]);
    });

    it("refuses unknown named entities and invalid character references (XML 1.0 well-formedness)", async () => {
        for (const value of ["&x;", "&#xD800;", "&#99999999;"]) {
            const doc = `<gexf version="1.3"><graph><nodes><node id="a" label="${value}"/></nodes></graph></gexf>`;
            const error = await expectImportError(doc);
            expect(error.report.issues[0].code, value).toBe(XML_SYNTAX_CODE);
        }
    });

    it("decodes the predefined entities and character references", async () => {
        const doc = `<gexf version="1.3"><graph><nodes><node id="a" label="&lt;&amp;&#65;&#x42;"/></nodes></graph></gexf>`;
        const { snapshot } = await load(doc);
        expect(snapshot.nodes.value("label", 0)).toBe("<&AB");
    });
});

describe("gexfImporter: report shape", () => {
    it("returns a frozen report with a duration and no loss notes", async () => {
        const { report } = await load(BARE);
        expect(Object.isFrozen(report)).toBe(true);
        expect(report.durationMs).toBeGreaterThanOrEqual(0);
        expect(report.lossy).toEqual([]);
        expect(report.warningCount + report.errorCount).toBe(report.issues.length);
    });

    it("counts both halves of expanded edges and INVALID_INDEX never appears in the pair column", async () => {
        const { snapshot, report } = await load(MIXED_UNDIRECTED_HEADER);
        expect(report.counts.edges).toBe(snapshot.edgeCount);
        const pair = snapshot.edges.byRole("pair");
        for (let e = 0; e < snapshot.edgeCount; e++) {
            if (pair?.isSet(e)) {
                expect(pair.value(e)).not.toBe(INVALID_INDEX);
            }
        }
    });
});

describe("gexfImporter: less common constructs", () => {
    it("warns on every unknown header value and applies the defaults", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="sideways" timerepresentation="sometimes" idtype="uuid">
            <attributes class="node" mode="sometimes"><attribute id="a" title="a" type="string"/></attributes>
            <attributes class="weird"><attribute id="b" title="b" type="string"/></attributes>
            <nodes><node id="x"/></nodes></graph></gexf>`;
        const { snapshot, report } = await load(doc, { defaultDirected: true });
        expect(snapshot.directed).toBe(true);
        expect(snapshot.meta.timeRepresentation).toBeNull();
        expect(snapshot.meta.idType).toBeNull();
        expect(report.issues.map((i) => [i.code, i.element])).toEqual([
            [HEADER_VALUE_CODE, "defaultedgetype"],
            [HEADER_VALUE_CODE, "timerepresentation"],
            [HEADER_VALUE_CODE, "idtype"],
            [HEADER_VALUE_CODE, "mode"],
            [ATTRIBUTES_CLASS_CODE, null],
        ]);
        expect(report.issues[4].message).toContain('class="weird"');
        expect(snapshot.nodes.get("a")?.meta.dynamic).toBe(false);
        expect(snapshot.nodes.get("b")).toBeNull();
    });

    it("reads 1.3 intervals attributes and edge spells, and rejects a malformed interval", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" timeformat="double">
            <nodes><node id="a" intervals="&lt;[1, 2]; [3.5, 4]&gt;"/><node id="b" intervals="<[1]>"/></nodes>
            <edges><edge source="a" target="b"><spells><spell start="1" end="2"/></spells><viz:color r="1" g="2" b="3"/><viz:shape value="dashed"/></edge></edges></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(values(snapshot.nodes.get("spells"))).toEqual([
            [
                [1, 2],
                [3.5, 4],
            ],
            undefined,
        ]);
        expect(values(snapshot.edges.get("spells"))).toEqual([[[1, 2]]]);
        expect(snapshot.edges.value("shape", 0)).toBe("dashed");
        expect(codes(report)).toEqual(["E_COLUMN_TYPE"]);
        expect(report.issues[0].message).toContain("[start, end] pair");
        expect(report.issues[0].element).toBe("b");
    });

    it("reports a <parent> without for and one naming an unknown node", async () => {
        const doc = `<gexf version="1.3"><graph><nodes>
            <node id="a"><parents><parent/><parent for="ghost"/><parent for="b"/></parents></node><node id="b"/>
            </nodes></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(codes(report)).toEqual([ATTVALUE_SHAPE_CODE, UNKNOWN_PARENT_CODE]);
        expect(report.issues[1].message).toContain("ghost");
        expect(values(snapshot.nodes.get("parents"))).toEqual([[1], undefined]);
    });

    it("keeps the text companion of a timed date value and reports a rounded timed long", async () => {
        const doc = `<gexf version="1.3"><graph mode="dynamic" timeformat="integer">
            <attributes class="node" mode="dynamic"><attribute id="d" title="d" type="date"/><attribute id="l" title="l" type="long"/></attributes>
            <nodes><node id="a"><attvalues>
                <attvalue for="d" value="2020-01-02T03:00:00Z" start="1" end="2"/>
                <attvalue for="d" value="2021-01-02" start="2"/>
                <attvalue for="l" value="9007199254740993" start="1"/>
            </attvalues></node></nodes></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(codes(report)).toEqual(["W_PRECISION"]);
        const table = snapshot.extensions.get("temporal:node:d");
        expect(table?.names()).toEqual([
            "element",
            "start",
            "end",
            "value",
            "start.text",
            "end.text",
            "open",
            "value.text",
        ]);
        expect(values(table?.get("value") ?? null)).toEqual([Date.UTC(2020, 0, 2, 3), Date.UTC(2021, 0, 2)]);
        expect(values(table?.get("value.text") ?? null)).toEqual(["2020-01-02T03:00:00Z", undefined]);
        expect(values(snapshot.extensions.get("temporal:node:l")?.get("value") ?? null)).toEqual([9007199254740992]);
    });

    it("rejects a timed weight that is not a number and a mangled viz element", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" timeformat="double">
            <attributes class="edge" mode="dynamic"><attribute id="w" title="weight" type="double"/></attributes>
            <nodes><node id="a"><viz:color hex="#12345"/><viz:shape/><viz:size/></node><node id="b"><viz:color r="1" g="x" b="3"/></node></nodes>
            <edges><edge source="a" target="b"><attvalues><attvalue for="w" value="heavy" start="1"/><attvalue for="w" value="2" start="2"/></attvalues></edge></edges></graph></gexf>`;
        const { snapshot, report } = await load(doc);
        expect(codes(report)).toEqual([
            VIZ_VALUE_CODE,
            VIZ_VALUE_CODE,
            VIZ_VALUE_CODE,
            VIZ_VALUE_CODE,
            "E_INVALID_WEIGHT",
        ]);
        expect(report.issues.map((i) => i.message)).toEqual([
            expect.stringContaining("#RRGGBB"),
            expect.stringContaining("without a value"),
            expect.stringContaining("without a value attribute"),
            expect.stringContaining('"x" is not a double'),
            expect.stringContaining("heavy"),
        ]);
        expect(snapshot.nodes.byRole("color")).toBeNull();
        expect(values(snapshot.extensions.get("temporal:edge:weight")?.get("value") ?? null)).toEqual([2]);
    });

    it("keeps a companion for an edge end text and writes both lifetime companions", async () => {
        const doc = `<gexf version="1.3"><graph defaultedgetype="directed" mode="dynamic" timeformat="dateTime">
            <nodes><node id="a"/><node id="b"/></nodes>
            <edges><edge source="a" target="b" start="2020-01-01" end="2020-02-01T00:00:00+00:00"/></edges></graph></gexf>`;
        const { snapshot } = await load(doc);
        expect(snapshot.edges.value("start.text", 0)).toBe("2020-01-01");
        expect(snapshot.edges.value("end.text", 0)).toBe("2020-02-01T00:00:00+00:00");
        expect(snapshot.edges.get("start.text")?.meta.role).toBe("timeText");
        expect(snapshot.edges.get("end.text")?.meta.role).toBeNull();
        expect(snapshot.edges.get("end.text")?.meta.extra).toEqual({ for: "end" });
    });
});
