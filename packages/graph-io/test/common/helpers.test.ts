import { existsSync } from "node:fs";

import { GraphBuilder, type GraphSink, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../src/common/direction.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { type GraphExporter, type GraphImporter, type ImportInput, type ImportReport } from "../../src/types.js";
import {
    byteChunks,
    CORPUS_FORMATS,
    corpusEntry,
    corpusFiles,
    corpusPath,
    inputShapes,
    loadManifest,
    malformedFiles,
    malformedPath,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
    readMalformedText,
} from "../helpers/corpus.js";
import { compareSnapshots, describeDiffs, expectSameSnapshot, roundTrip, valuesEqual } from "../helpers/roundtrip.js";

describe("corpus helper", () => {
    it("lists every format's manifest and every listed file exists", () => {
        for (const format of CORPUS_FORMATS) {
            const manifest = loadManifest(format);
            expect(manifest.format).toBe(format);
            expect(manifest.files.length).toBeGreaterThan(0);
            for (const file of corpusFiles(format)) {
                expect(existsSync(corpusPath(format, file.path)), `${format}/${file.path}`).toBe(true);
                expect(file.expectedNodes).toBeGreaterThan(0);
                // a Neo4j node-only file (movies-nodes.csv) legitimately declares no relationships
                expect(file.expectedEdges).toBeGreaterThanOrEqual(0);
                expect(Array.isArray(file.features)).toBe(true);
            }
        }
        expect(corpusEntry("gml", "karate.gml")).toMatchObject({ expectedNodes: 34, expectedEdges: 78 });
        expect(() => corpusEntry("gml", "nope.gml")).toThrow(/no manifest entry/);
    });

    it("reads files as text and bytes", () => {
        const text = readCorpusText("csv", "simple-edges.csv");
        const bytes = readCorpusBytes("csv", "simple-edges.csv");
        expect(text.length).toBeGreaterThan(0);
        expect(new TextDecoder().decode(bytes)).toBe(text);
    });

    it("lists malformed cases for every format and reads them", () => {
        for (const format of CORPUS_FORMATS) {
            const files = malformedFiles(format);
            expect(files.length, format).toBeGreaterThan(0);
            for (const name of files) {
                expect(existsSync(malformedPath(format, name))).toBe(true);
                expect(readMalformedBytes(format, name)).toBeInstanceOf(Uint8Array);
            }
        }
        expect(malformedFiles("csv")).toContain("empty-file.csv");
        expect(readMalformedText("csv", "header-only.csv").length).toBeGreaterThan(0);
        expect(readMalformedBytes("csv", "empty-file.csv").byteLength).toBe(0);
    });

    it("offers every input shape of one file", async () => {
        const bytes = readCorpusBytes("csv", "simple-edges.csv");
        const shapes = inputShapes(bytes);
        expect(shapes.map((s) => s.name)).toEqual([
            "string",
            "Uint8Array",
            "byte chunks of 7",
            "text chunks of 5",
            "ReadableStream of 64",
        ]);
        const text = new TextDecoder().decode(bytes);
        for (const shape of shapes) {
            const input = shape.make();
            let decoded: string;
            if (typeof input === "string") {
                decoded = input;
            } else if (input instanceof Uint8Array) {
                decoded = new TextDecoder().decode(input);
            } else if ("getReader" in input) {
                const reader = input.getReader();
                const parts: Uint8Array[] = [];
                for (;;) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    parts.push(value);
                }
                decoded = new TextDecoder().decode(concat(parts));
            } else {
                const parts: (string | Uint8Array)[] = [];
                for await (const chunk of input) {
                    parts.push(chunk);
                }
                decoded = parts.map((p) => (typeof p === "string" ? p : new TextDecoder().decode(p))).join("");
            }
            expect(decoded, shape.name).toBe(text);
        }
        let count = 0;
        for await (const chunk of byteChunks(new Uint8Array(10), 3)) {
            count += chunk.byteLength;
        }
        expect(count).toBe(10);
    });
});

function concat(parts: Uint8Array[]): Uint8Array {
    const out = new Uint8Array(parts.reduce((n, p) => n + p.byteLength, 0));
    let offset = 0;
    for (const p of parts) {
        out.set(p, offset);
        offset += p.byteLength;
    }
    return out;
}

function sample(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
    r.setHeader(true);
    const label = b.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
    const age = b.declareNodeColumn({ name: "age", dtype: "i32" });
    const tags = b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
    const meta = b.declareNodeColumn({ name: "meta", dtype: "json" });
    const a = b.addNode("a");
    const c = b.addNode("c");
    b.addNode(3);
    b.setNodeValue(label, a, "Alpha");
    b.setNodeValue(age, a, 41);
    b.setNodeValue(tags, c, ["x", "y"]);
    b.setNodeValue(meta, c, { k: [1, { z: true }] });
    r.addEdge("a", "c", "directed", 0.1);
    r.addEdge("c", 3, "undirected");
    r.addEdge(3, "a", "directed");
    b.setGraphValue("name", "sample");
    return b.freeze();
}

describe("roundtrip helper: compareSnapshots", () => {
    it("finds no difference between a snapshot and an equal rebuild", () => {
        const a = sample();
        const b = sample();
        expect(compareSnapshots(a, b)).toEqual([]);
        expect(describeDiffs([])).toBe("no differences");
        expect(() => expectSameSnapshot(a, b)).not.toThrow();
        expect(compareSnapshots(a, b, { allowExtraColumns: false, originType: true })).toEqual([]);
    });

    it("reports direction, counts and ids", () => {
        const a = sample();
        const u = new GraphBuilder({ directed: false });
        u.addEdge("a", "c");
        const diffs = compareSnapshots(a, u.freeze());
        expect(diffs.map((d) => d.path)).toEqual(["directed", "nodeCount", "edgeCount"]);
        const fewer = new GraphBuilder({ directed: true });
        fewer.addEdge("a", "c");
        fewer.addEdge("c", "x");
        const d2 = compareSnapshots(a, fewer.freeze());
        expect(d2.map((d) => d.path)).toEqual(["edgeCount", "ids[2]"]);
        expect(d2[1].message).toBe('ids[2]: expected 3, got "x"');
    });

    it("reports topology and orientation", () => {
        const a = new GraphBuilder({ directed: true });
        a.addEdge("x", "y");
        a.addEdge("y", "z");
        const b = new GraphBuilder({ directed: true });
        b.addEdge("x", "y");
        b.addEdge("z", "y");
        const diffs = compareSnapshots(a.freeze(), b.freeze());
        expect(diffs.map((d) => d.path)).toEqual(["rowPtr", "colIdx", "edge[1]"]);
        expect(diffs[2].message).toBe('edge[1]: expected "1->2", got "2->1"');
    });

    it("reports weights, explicitness and tolerance", () => {
        const make = (w: number | undefined, x: number | undefined): GraphSnapshot => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            b.addEdge("x", "y", w);
            b.addEdge("y", "z", x);
            return b.freeze();
        };
        expect(compareSnapshots(make(0.1, undefined), make(0.1, undefined))).toEqual([]);
        const explicit = compareSnapshots(make(0.1, undefined), make(0.1, 1));
        expect(explicit.map((d) => d.path)).toEqual(["weightExplicit[1]"]);
        expect(compareSnapshots(make(0.1, undefined), make(0.1, 1), { weightExplicitness: false })).toEqual([]);
        const value = compareSnapshots(make(0.1, 2), make(0.1000001, 2));
        expect(value.map((d) => d.path)).toEqual(["weight[0]"]);
        expect(compareSnapshots(make(0.1, 2), make(0.1000001, 2), { tolerance: 1e-6 })).toEqual([]);
        const unweighted = new GraphBuilder({ directed: true });
        unweighted.addEdge("x", "y");
        unweighted.addEdge("y", "z");
        expect(compareSnapshots(make(2, 3), unweighted.freeze()).map((d) => d.path)).toEqual(["flags.weighted"]);
        // f32 arc weights when neither side has a shadow column
        const f32 = (w: number): GraphSnapshot => {
            const b = new GraphBuilder({ directed: true });
            b.addEdge("x", "y", w);
            b.addEdge("y", "z", 1);
            return b.freeze();
        };
        expect(compareSnapshots(f32(2), f32(2))).toEqual([]);
        expect(compareSnapshots(f32(2), f32(3)).map((d) => d.path)).toEqual(["weight[0]"]);
    });

    it("reports column presence, dtype, role, values and unset rows", () => {
        const base = sample();
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
        r.setHeader(true);
        const label = b.declareNodeColumn({ name: "label", dtype: "string" });
        const age = b.declareNodeColumn({ name: "age", dtype: "f64" });
        const tags = b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        const meta = b.declareNodeColumn({ name: "meta", dtype: "json" });
        const a = b.addNode("a");
        const c = b.addNode("c");
        b.addNode(3);
        b.setNodeValue(label, a, "Alpha!");
        b.setNodeValue(age, a, 41);
        b.setNodeValue(age, c, 5);
        b.setNodeValue(tags, c, ["x", "z"]);
        b.setNodeValue(meta, c, { k: [1, { z: false }] });
        r.addEdge("a", "c", "directed", 0.1);
        r.addEdge("c", 3, "undirected");
        r.addEdge(3, "a", "directed");
        const other = b.freeze();
        const diffs = compareSnapshots(base, other);
        expect(diffs.map((d) => d.path)).toEqual([
            "nodes.label.role",
            "nodes.label[0]",
            "nodes.age.dtype",
            "nodes.age[1]",
            "nodes.tags[1]",
            "nodes.meta[1]",
            "graph.name",
        ]);
        expect(
            compareSnapshots(base, other, {
                dtypes: false,
                roles: false,
                ignoreColumns: ["label", "tags", "meta"],
            }).map((d) => d.path),
        ).toEqual(["nodes.age[1]", "graph.name"]);
        expect(compareSnapshots(base, other, { ignoreRoles: ["label"], limit: 2 })).toHaveLength(2);
        expect(() => expectSameSnapshot(base, other)).toThrow(/snapshots differ \(7 difference\(s\)\)/);
        expect(describeDiffs(diffs).split("\n")).toHaveLength(7);
    });

    it("reports extra columns only when asked, and extension tables", () => {
        const a = sample();
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addGraph(a);
        const extra = b.declareNodeColumn({ name: "extra", dtype: "i32" });
        b.setNodeValue(extra, 0, 1);
        const t = b.addExtensionTable("temporal:node:age", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "end", dtype: "f64" },
            { name: "value", dtype: "i32" },
        ]);
        b.addExtensionRow(t, [0, 1, 2, 3]);
        const withExtras = b.freeze();
        expect(compareSnapshots(a, withExtras)).toEqual([]);
        const strict = compareSnapshots(a, withExtras, { allowExtraColumns: false });
        expect(strict.map((d) => d.path)).toEqual(["nodes.extra", "extensions.temporal:node:age"]);
        const missing = compareSnapshots(withExtras, a);
        expect(missing.map((d) => d.path)).toEqual(["nodes.extra", "extensions.temporal:node:age"]);
        expect(compareSnapshots(withExtras, a, { extensions: false, ignoreColumns: ["extra"] })).toEqual([]);
    });

    it("valuesEqual is deep with tolerance", () => {
        expect(valuesEqual(1, 1, 0)).toBe(true);
        expect(valuesEqual(1, 1.5, 0)).toBe(false);
        expect(valuesEqual(1, 1.5, 1)).toBe(true);
        expect(valuesEqual(NaN, NaN, 0)).toBe(true);
        expect(valuesEqual(Infinity, Infinity, 0)).toBe(true);
        expect(valuesEqual([1, [2]], [1, [2]], 0)).toBe(true);
        expect(valuesEqual([1, 2], [1], 0)).toBe(false);
        expect(valuesEqual(new Float32Array([1, 2]), [1, 2], 0)).toBe(true);
        expect(valuesEqual({ a: 1, b: { c: [true] } }, { b: { c: [true] }, a: 1 }, 0)).toBe(true);
        expect(valuesEqual({ a: 1 }, { a: 1, b: 2 }, 0)).toBe(false);
        expect(valuesEqual({ a: 1 }, { b: 1 }, 0)).toBe(false);
        expect(valuesEqual("x", "x", 0)).toBe(true);
        expect(valuesEqual("x", 1, 0)).toBe(false);
        expect(valuesEqual(null, null, 0)).toBe(true);
        expect(valuesEqual(null, {}, 0)).toBe(false);
    });
});

/** A toy line format "u v [w]" per edge, enough to drive roundTrip(). */
const toyExporter: GraphExporter = {
    format: "toy",
    capabilities: {
        mixedDirection: false,
        multiEdges: true,
        selfLoops: true,
        edgeIds: "none",
        idCharset: "any",
        dtypes: [],
        components: false,
        lists: false,
        json: false,
        defaults: false,
        options: false,
        hierarchy: false,
        temporal: "none",
        graphAttributes: false,
        positions: false,
        viz: false,
    },
    check: (): [] => [],
    export(): AsyncIterable<Uint8Array> {
        throw new Error("not used");
    },
    exportToString(snapshot: GraphSnapshot): Promise<string> {
        const list = snapshot.edgeList();
        const shadow = snapshot.edges.byRole("weight");
        const lines = [snapshot.directed ? "directed" : "undirected"];
        for (let e = 0; e < snapshot.edgeCount; e++) {
            const u = String(snapshot.ids.idOf(list.src[e]));
            const v = String(snapshot.ids.idOf(list.dst[e]));
            const explicit = shadow === null ? true : shadow.isSet(e);
            const w = shadow === null ? (list.weights?.[e] ?? 1) : (shadow.value(e) as number);
            lines.push(explicit && snapshot.flags.weighted ? `${u} ${v} ${w}` : `${u} ${v}`);
        }
        return Promise.resolve(`${lines.join("\n")}\n`);
    },
};

const toyImporter: GraphImporter = {
    format: "toy",
    extensions: [".toy"],
    mimeTypes: [],
    import(input: ImportInput, sink: GraphSink): Promise<ImportReport> {
        const report = new ImportReportBuilder("toy", 100);
        const lines = (input as string).split("\n").filter((l) => l.length > 0);
        sink.setDirected(lines[0] === "directed");
        for (const line of lines.slice(1)) {
            const [u, v, w] = line.split(" ");
            sink.addEdge(Number(u), Number(v), w === undefined ? undefined : Number(w));
            report.counts.edges++;
        }
        return Promise.resolve(report.finish());
    },
};

describe("roundtrip helper: roundTrip", () => {
    it("exports, re-imports into a fresh f64 builder and freezes", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge(1, 2, 0.1);
        b.addEdge(2, 3);
        b.addEdge(3, 1, 16777217);
        const original = b.freeze();
        const result = await roundTrip(original, toyExporter, toyImporter);
        expect(result.text).toBe("directed\n1 2 0.1\n2 3\n3 1 16777217\n");
        expect(result.notes).toEqual([]);
        expect(result.report.counts.edges).toBe(3);
        expect(result.freeze.compacted).toBe(false);
        expect(result.snapshot.directed).toBe(true);
        expectSameSnapshot(original, result.snapshot);
    });

    it("honours builder overrides", async () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge(1, 2);
        b.addEdge(1, 2);
        const original = b.freeze();
        const merged = await roundTrip(original, toyExporter, toyImporter, { builder: { duplicateEdges: "first" } });
        expect(merged.snapshot.edgeCount).toBe(1);
        expect(merged.freeze.mergedEdges).toBe(1);
        const kept = await roundTrip(original, toyExporter, toyImporter, { importOptions: { duplicateEdges: "keep" } });
        expect(kept.snapshot.edgeCount).toBe(2);
        expectSameSnapshot(original, kept.snapshot);
    });
});
