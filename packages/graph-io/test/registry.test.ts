import { GraphBuilder, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { csvImporter } from "../src/formats/csv/importer.js";
import { gexfExporter } from "../src/formats/gexf/exporter.js";
import {
    checkExport,
    createRegistry,
    exportGraph,
    exportGraphToString,
    FormatRegistry,
    importGraph,
    registry,
    sniff,
    UNKNOWN_FORMAT_CODE,
} from "../src/registry.js";
import { GRAPH_FORMATS, SNIFF_HEAD_BYTES } from "../src/sniff.js";
import { type GraphImporter, ImportError, type ImportInput } from "../src/types.js";
import {
    byteChunks,
    byteStream,
    CORPUS_FORMATS,
    corpusFiles,
    readCorpusBytes,
    textChunksOf,
} from "./helpers/corpus.js";
import { expectSameSnapshot } from "./helpers/roundtrip.js";

const EDGES = "source,target,weight\na,b,0.1\nb,c,2\nc,a,3\n";

/**
 * The ids of the edges of a snapshot as "source-target" texts, in logical edge order.
 * @param s - the snapshot
 * @returns the texts
 */
function edges(s: GraphSnapshot): string[] {
    const el = s.edgeList();
    return Array.from(
        { length: s.edgeCount },
        (_, e) => `${String(s.ids.idOf(el.src[e]))}-${String(s.ids.idOf(el.dst[e]))}`,
    );
}

/**
 * A neo4j corpus manifest entry's importer options, when it has any.
 * @param format - the corpus format
 * @param path - the file
 * @returns the options or an empty object
 */
function corpusOptions(format: string, path: string): Record<string, unknown> {
    const entry = corpusFiles(format as (typeof CORPUS_FORMATS)[number]).find((f) => f.path === path) as
        { options?: Record<string, unknown> } | undefined;
    return entry?.options ?? {};
}

describe("FormatRegistry", () => {
    it("holds the eight built-in formats in GRAPH_FORMATS order, importers and exporters alike", () => {
        expect(registry).toBeInstanceOf(FormatRegistry);
        expect(registry.formats()).toEqual([...GRAPH_FORMATS]);
        expect(registry.importers().map((i) => i.format)).toEqual([...GRAPH_FORMATS]);
        expect(registry.exporters().map((e) => e.format)).toEqual([...GRAPH_FORMATS]);
        for (const format of GRAPH_FORMATS) {
            expect(registry.hasImporter(format)).toBe(true);
            expect(registry.hasExporter(format)).toBe(true);
            expect(registry.importer(format).format).toBe(format);
            expect(registry.exporter(format).format).toBe(format);
        }
        expect(createRegistry()).not.toBe(registry);
        expect(createRegistry().formats()).toEqual(registry.formats());
    });

    it("rejects an unknown format with E_UNSUPPORTED naming the known ones", () => {
        for (const fn of [(): unknown => registry.importer("nope"), (): unknown => registry.exporter("nope")]) {
            let caught: unknown;
            try {
                fn();
            } catch (err) {
                caught = err;
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            expect(caught).toMatchObject({
                code: "E_UNSUPPORTED",
                details: { option: "format", found: "nope", supported: [...GRAPH_FORMATS] },
            });
        }
        expect(registry.hasImporter("nope")).toBe(false);
        expect(registry.hasExporter("nope")).toBe(false);
    });

    it("registers custom plugins, replaces one of the same name in place and keeps the order", () => {
        const custom = new FormatRegistry();
        expect(custom.formats()).toEqual([]);
        expect(custom.sniff({ filename: "a.csv" })).toBeNull();
        const fake: GraphImporter = {
            format: "fake",
            extensions: [".fake"],
            mimeTypes: [],
            sniff: () => 1,
            import: () => Promise.reject(new Error("no")),
        };
        custom.registerImporter(fake).registerImporter(csvImporter).registerExporter(gexfExporter);
        expect(custom.formats()).toEqual(["fake", "csv", "gexf"]);
        expect(custom.sniff({ head: "source,target\na,b\n" })?.format).toBe("fake");
        const replaced: GraphImporter = { ...fake, sniff: () => 0 };
        custom.registerImporter(replaced);
        expect(custom.importers().map((i) => i.format)).toEqual(["fake", "csv"]);
        expect(custom.importer("fake")).toBe(replaced);
        expect(custom.sniff({ head: "source,target\na,b\n" })?.format).toBe("csv");
        expect(custom.sniffAll({ filename: "x.fake" }).map((r) => r.format)).toEqual(["fake"]);
    });
});

describe("importGraph (design 8.4)", () => {
    it("reads an explicitly named format into a fresh f64 builder and freezes", async () => {
        const result = await importGraph(EDGES, { format: "csv" });
        expect(result.format).toBe("csv");
        expect(result.sniff).toBeNull();
        expect(result.snapshot.nodeCount).toBe(3);
        expect(result.snapshot.edgeCount).toBe(3);
        expect(result.snapshot.directed).toBe(true);
        expect(edges(result.snapshot)).toEqual(["a-b", "b-c", "c-a"]);
        expect(result.report.format).toBe("csv");
        expect(result.report.counts).toMatchObject({ nodes: 3, edges: 3 });
        expect(result.report.issues).toEqual([]);
        expect(result.freeze.compacted).toBe(false);
        // weightDtype f64 by default: 0.1 survives in the role-weight shadow column
        const weight = result.snapshot.edges.byRole("weight");
        expect(weight?.dtype).toBe("f64");
        expect(weight?.value(0)).toBe(0.1);
    });

    it("sniffs the format from the content, the filename and the MIME type", async () => {
        const byContent = await importGraph(EDGES);
        expect(byContent.format).toBe("csv");
        expect(byContent.sniff).toMatchObject({ format: "csv", extension: false, mimeType: false });
        const byName = await importGraph(EDGES, { filename: "edges.csv", mimeType: "text/csv" });
        expect(byName.sniff).toMatchObject({ format: "csv", extension: true, mimeType: true });
        expect(byName.sniff?.confidence).toBeGreaterThan(byContent.sniff?.confidence ?? 1);
        const gml = await importGraph("graph [ node [ id 1 ] node [ id 2 ] edge [ source 1 target 2 ] ]", {
            filename: "misnamed.csv",
        });
        expect(gml.format).toBe("gml");
        expect(gml.snapshot.edgeCount).toBe(1);
        expect((await importGraph('{"nodes":[{"id":"a"}],"links":[]}')).sniff?.dialect).toBe("d3");
    });

    it("reads every input shape through the sniffing path with the same result", async () => {
        const bytes = new TextEncoder().encode(EDGES);
        const expected = (await importGraph(bytes, { format: "csv" })).snapshot;
        const shapes: [string, ImportInput][] = [
            ["string", EDGES],
            ["bytes", bytes],
            ["byte chunks of 7", byteChunks(bytes, 7)],
            ["text chunks of 5", textChunksOf(EDGES, 5)],
            ["stream", byteStream(bytes, 4)],
        ];
        for (const [name, input] of shapes) {
            const result = await importGraph(input);
            expect(result.format, name).toBe("csv");
            expectSameSnapshot(expected, result.snapshot);
        }
    });

    it("replays a stream longer than the sniffed head without losing or repeating bytes", async () => {
        const rows = Array.from({ length: 3000 }, (_, i) => `n${i},n${(i * 7) % 3000},${i % 5}`).join("\n");
        const text = `source,target,weight\n${rows}\n`;
        const bytes = new TextEncoder().encode(text);
        expect(bytes.byteLength).toBeGreaterThan(SNIFF_HEAD_BYTES * 4);
        const expected = (await importGraph(text, { format: "csv" })).snapshot;
        for (const input of [byteStream(bytes, 1000), byteChunks(bytes, 999), textChunksOf(text, 1001)]) {
            const result = await importGraph(input);
            expect(result.format).toBe("csv");
            expect(result.snapshot.edgeCount).toBe(3000);
            expectSameSnapshot(expected, result.snapshot);
        }
        // a stream shorter than the head is replayed whole
        const short = await importGraph(byteStream(new TextEncoder().encode(EDGES), 2));
        expect(short.snapshot.edgeCount).toBe(3);
    });

    it("cancels the source when the importer stops early and honours an aborted signal", async () => {
        const controller = new AbortController();
        controller.abort();
        let pulled = 0;
        const source: AsyncIterable<Uint8Array> = {
            async *[Symbol.asyncIterator](): AsyncGenerator<Uint8Array, void, undefined> {
                pulled++;
                yield await Promise.resolve(new TextEncoder().encode(EDGES));
            },
        };
        await expect(importGraph(source, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
        expect(pulled).toBe(0);
        await expect(importGraph(EDGES, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
        // a stream that fails after the head is peeked surfaces its error, and the head is not lost
        let cancelled = false;
        const failing = new ReadableStream<Uint8Array>({
            pull(controllerStream): void {
                controllerStream.enqueue(new TextEncoder().encode(EDGES.slice(0, 20)));
                controllerStream.error(new Error("network down"));
            },
            cancel(): void {
                cancelled = true;
            },
        });
        await expect(importGraph(failing)).rejects.toThrow("network down");
        expect(cancelled).toBe(false);
    });

    it("fails with an ImportError carrying E_UNKNOWN_FORMAT when nothing recognises the input", async () => {
        let caught: unknown;
        try {
            await importGraph(new Uint8Array([0, 1, 2, 3]), { filename: "blob.bin", mimeType: "image/png" });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const error = caught as ImportError;
        expect(error.code).toBe("E_IMPORT");
        expect(error.message).toContain('filename "blob.bin"');
        expect(error.message).toContain('MIME type "image/png"');
        expect(error.report.format).toBe("unknown");
        expect(error.report.issues).toHaveLength(1);
        expect(error.report.issues[0]).toMatchObject({
            code: UNKNOWN_FORMAT_CODE,
            category: "parse-error",
            severity: "error",
        });
        expect(error.details).toMatchObject({ code: UNKNOWN_FORMAT_CODE, formats: [...GRAPH_FORMATS] });
        await expect(importGraph("")).rejects.toBeInstanceOf(ImportError);
        await expect(importGraph(EDGES, { format: "nope" })).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
    });

    it("seeds the builder from the common options and passes format options through", async () => {
        const parallel = "source,target,weight\na,b,1\na,b,2\na,a,5\n";
        const kept = await importGraph(parallel, { format: "csv" });
        expect(kept.snapshot.edgeCount).toBe(3);
        expect(kept.snapshot.flags.multigraph).toBe(true);
        const merged = await importGraph(parallel, { format: "csv", duplicateEdges: "sum", selfLoops: "drop" });
        expect(merged.snapshot.edgeCount).toBe(1);
        expect(merged.snapshot.edgeList().weights?.[0]).toBe(3);
        expect(merged.freeze.mergedEdges).toBe(1);
        expect(merged.freeze.droppedSelfLoops).toBe(1);
        expect(merged.report.issues).toEqual([]);
        const strict = await importGraph("source,target\na,ghost\n", {
            format: "csv",
            addMissingNodes: false,
            nodes: "id\na\n",
        });
        expect(strict.snapshot.nodeCount).toBe(1);
        expect(strict.snapshot.edgeCount).toBe(0);
        expect(strict.report.issues.map((i) => i.code)).toEqual(["E_UNKNOWN_NODE"]);
        const f32 = await importGraph(EDGES, { format: "csv", weightDtype: "f32" });
        expect(f32.snapshot.edges.byRole("weight")).toBeNull();
        expect(f32.snapshot.edgeList().weights?.[0]).toBe(Math.fround(0.1));
        const semicolons = await importGraph("source;target\na;b\n", { format: "csv", delimiter: ";" });
        expect(semicolons.snapshot.edgeCount).toBe(1);
        const profiled = await importGraph(EDGES, {
            format: "csv",
            freeze: { profile: true },
            builder: { expectedNodes: 3, expectedEdges: 3 },
        });
        expect(Object.keys(profiled.freeze.timings).length).toBeGreaterThan(0);
        const undirected = await importGraph(EDGES, { format: "csv", defaultDirected: false });
        expect(undirected.snapshot.directed).toBe(false);
    });

    it("imports every corpus file by sniffing its name and content", async () => {
        for (const format of CORPUS_FORMATS) {
            for (const file of corpusFiles(format)) {
                const bytes = readCorpusBytes(format, file.path);
                const result = await importGraph(bytes, { filename: file.path, ...corpusOptions(format, file.path) });
                expect(result.format, `${format}/${file.path}`).toBe(format);
                expect(result.sniff?.format, `${format}/${file.path}`).toBe(format);
                expect(result.snapshot.nodeCount, `${format}/${file.path}`).toBeGreaterThan(0);
                expect(result.report.errorCount, `${format}/${file.path}`).toBe(0);
                expect(result.report.truncated).toBe(false);
            }
        }
    });
});

describe("exportGraph / exportGraphToString / checkExport (design 8.5)", () => {
    it("writes through the named exporter and reads back the same graph", async () => {
        const { snapshot } = await importGraph(EDGES, { format: "csv" });
        const text = await exportGraphToString(snapshot, "gexf");
        expect(text).toContain("<gexf");
        const chunks: Uint8Array[] = [];
        for await (const chunk of exportGraph(snapshot, "gexf")) {
            chunks.push(chunk);
        }
        const joined = new TextDecoder().decode(
            chunks.reduce((all, c) => {
                const next = new Uint8Array(all.byteLength + c.byteLength);
                next.set(all);
                next.set(c, all.byteLength);
                return next;
            }, new Uint8Array(0)),
        );
        expect(joined).toBe(text);
        const back = await importGraph(text, { format: "gexf" });
        expect(back.snapshot.nodeCount).toBe(3);
        expect(edges(back.snapshot)).toEqual(edges(snapshot));
        expect(Array.from(back.snapshot.edgeList().weights ?? [])).toEqual(
            Array.from(snapshot.edgeList().weights ?? []),
        );
        expect(snapshot.edgeList().weights).not.toBeNull();
        expect(checkExport(snapshot, "gexf")).toEqual([]);
        expect(await registry.exportGraphToString(snapshot, "json", { dialect: "d3" })).toContain('"links"');
    });

    it("reports losses through check() and refuses an unknown format", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        const parallel = b.freeze();
        const notes = checkExport(parallel, "pajek");
        expect(notes.length).toBeGreaterThan(0);
        expect(notes.every((n) => typeof n.code === "string" && typeof n.message === "string")).toBe(true);
        expect(() => checkExport(parallel, "nope")).toThrow(GraphFormatError);
        expect(() => exportGraph(parallel, "nope")).toThrow(GraphFormatError);
        await expect(exportGraphToString(parallel, "nope")).rejects.toThrow(GraphFormatError);
    });

    it("sniff() at the top level uses the default registry", () => {
        expect(sniff({ filename: "graph.gexf" })?.format).toBe("gexf");
        expect(sniff({ head: '*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2\n' })?.format).toBe("pajek");
        expect(sniff({})).toBeNull();
    });
});
