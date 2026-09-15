/**
 * Streaming audit, throughput lens (design section 15.5 item 5: parse -> builder -> freeze end to
 * end, "text parsing dominates"): the wall time of each importer at one million edges (the three
 * line- or record-oriented formats) and at 200k edges (the two XML formats), through a 64 KiB file
 * stream, against the freeze of the builder it filled, so the parser overhead is a number rather
 * than a slogan. STATUS.md quotes 26 ms for the freeze of 100k nodes / 1M edges directed with
 * weightDtype "f64" on this host, and 21 ms for the builder push; the freeze measured here runs
 * on a cold builder that also carries the columns the file declared.
 *
 * Gated on IO_BENCH=1. The ceilings are ten times the observed values: they catch a regression
 * to a different complexity class, not host noise.
 */

import { open } from "node:fs/promises";

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    csvEdgeList,
    fixtureBytes,
    gexfDocument,
    graphmlDocument,
    neo4jNodes,
    neo4jRelationships,
    pajekNetwork,
} from "../../benchmarks/fixtures.js";
import { fileStream, fullGc, type HeapProfile, mib, sampleHeap } from "../../benchmarks/measure.js";
import { csvExporter, csvImporter } from "../../src/formats/csv/index.js";
import { gexfExporter, gexfImporter } from "../../src/formats/gexf/index.js";
import { graphmlExporter, graphmlImporter } from "../../src/formats/graphml/index.js";
import { neo4jImporter } from "../../src/formats/neo4j/index.js";
import { pajekExporter, pajekImporter } from "../../src/formats/pajek/index.js";
import { type GraphExporter } from "../../src/types.js";

const BENCH = process.env.IO_BENCH === "1";
const LONG = { timeout: 300_000 };
const CHUNK = 64 * 1024;

interface Row {
    readonly format: string;
    readonly edges: number;
    readonly inputBytes: number;
    readonly parseMs: number;
    readonly freezeMs: number;
}

const rows: Row[] = [];

async function measure(
    format: string,
    edges: number,
    inputBytes: number,
    run: (sink: GraphBuilder) => Promise<void>,
): Promise<Row> {
    fullGc();
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const t0 = performance.now();
    await run(sink);
    const parseMs = performance.now() - t0;
    expect(sink.edgeCount).toBe(edges);
    const t1 = performance.now();
    const snapshot = sink.freeze();
    const freezeMs = performance.now() - t1;
    expect(snapshot.edgeCount).toBe(edges);
    const row = { format, edges, inputBytes, parseMs, freezeMs };
    rows.push(row);
    console.log(
        `${format}: ${edges} edges, ${mib(inputBytes)}, parse ${parseMs.toFixed(0)} ms ` +
            `(${((1000 * parseMs) / edges).toFixed(2)} us/edge, ${(inputBytes / 1048576 / (parseMs / 1000)).toFixed(1)} MiB/s), ` +
            `freeze ${freezeMs.toFixed(0)} ms, parse/freeze ${(parseMs / freezeMs).toFixed(0)}x, parse/STATUS-freeze(26 ms) ${(parseMs / 26).toFixed(0)}x`,
    );
    return row;
}

interface ExportRun {
    readonly ms: number;
    readonly bytes: number;
    readonly chunks: number;
    readonly largest: number;
    readonly profile: HeapProfile;
}

let cached: GraphSnapshot | null = null;

async function snapshotOf1M(): Promise<GraphSnapshot> {
    if (cached === null) {
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await csvImporter.import(fileStream(csvEdgeList(1_000_000, 100_000), CHUNK), sink);
        cached = sink.freeze({ release: true });
    }
    return cached;
}

/** Consume export() chunk by chunk into /dev/null (real writes, so the timer sampler runs). */
async function exportToDevNull(name: string, exporter: GraphExporter, snapshot: GraphSnapshot): Promise<ExportRun> {
    const out = await open("/dev/null", "w");
    let bytes = 0;
    let chunks = 0;
    let largest = 0;
    try {
        const profile = await sampleHeap(async () => {
            for await (const chunk of exporter.export(snapshot)) {
                bytes += chunk.byteLength;
                largest = Math.max(largest, chunk.byteLength);
                chunks++;
                await out.write(chunk);
            }
        });
        console.log(
            `${name} export 1M: ${profile.ms.toFixed(0)} ms, ${mib(bytes)} in ${chunks} chunks (largest ${mib(largest)}), ` +
                `${profile.samples} samples, peak heap growth ${mib(profile.peakHeap - profile.baseHeap)}, retained ${mib(profile.retainedHeap - profile.baseHeap)}`,
        );
        return { ms: profile.ms, bytes, chunks, largest, profile };
    } finally {
        await out.close();
    }
}

function expectExportStreams(name: string, run: ExportRun): void {
    expect(run.chunks, `${name}: chunk count`).toBeGreaterThan(100);
    expect(run.largest, `${name}: largest chunk`).toBeLessThan(4 * CHUNK);
    expect(run.profile.samples, `${name}: timer samples`).toBeGreaterThanOrEqual(3);
    // young-generation garbage is bounded by the two 16 MiB semispaces; a buffered document adds its own size
    expect(run.profile.peakHeap - run.profile.baseHeap, `${name}: peak heap growth`).toBeLessThan(
        run.bytes + 32 * 1048576,
    );
}

describe.skipIf(!BENCH)("streaming audit: throughput at 1M edges (IO_BENCH=1)", () => {
    it(
        "CSV imports 1M edges through a file stream",
        async () => {
            // observed: 952 ms (0.95 us/edge, 22 MiB/s); freeze 57 ms; about 37x the STATUS freeze
            const path = csvEdgeList(1_000_000, 100_000);
            const row = await measure("csv", 1_000_000, fixtureBytes(path), async (sink) => {
                await csvImporter.import(fileStream(path, CHUNK), sink);
            });
            expect(row.parseMs).toBeLessThan(10_000);
        },
        LONG,
    );

    it(
        "Pajek imports 1M arcs through a file stream",
        async () => {
            // observed: 1170-1215 ms (1.2 us/edge, 19 MiB/s); about 45x the STATUS freeze
            const path = pajekNetwork(1_000_000, 100_000);
            const row = await measure("pajek", 1_000_000, fixtureBytes(path), async (sink) => {
                await pajekImporter.import(fileStream(path, CHUNK), sink);
            });
            expect(row.parseMs).toBeLessThan(12_000);
        },
        LONG,
    );

    it(
        "Neo4j imports 100k nodes and 1M relationships through file streams",
        async () => {
            // observed: 1440-1645 ms (1.5 us/edge, 18 MiB/s; the slowest line-oriented importer);
            // about 60x the STATUS freeze. Profile (self time): the record state machine 18%, the
            // builder's id lookup 10%, readInput 7.5%, isHeaderRecord + its regexp on every data
            // row 6%, parseDecimalText 5%, canonicalId 4%, a `${start}->${end}` element string per row
            const nodes = neo4jNodes(100_000);
            const rels = neo4jRelationships(1_000_000, 100_000);
            const row = await measure("neo4j", 1_000_000, fixtureBytes(nodes) + fixtureBytes(rels), async (sink) => {
                await neo4jImporter.import(fileStream(nodes, CHUNK), sink);
                await neo4jImporter.import(fileStream(rels, CHUNK), sink);
            });
            expect(row.parseMs).toBeLessThan(16_000);
        },
        LONG,
    );

    it(
        "GraphML imports 200k edges through a file stream",
        async () => {
            // observed: 721-787 ms (3.9 us/edge, 23 MiB/s), so about 4 s and 150x the freeze per 1M
            // edges. Profile: readName (codePointAt + isNameStart / isNameChar per character) 21%,
            // the builder's id lookup 11%, finishEdge (a Set of every edge id) 9%, parseStartTag 8.5%
            const path = graphmlDocument(200_000, 20_000);
            const row = await measure("graphml", 200_000, fixtureBytes(path), async (sink) => {
                await graphmlImporter.import(fileStream(path, CHUNK), sink);
            });
            expect(row.parseMs).toBeLessThan(8_000);
        },
        LONG,
    );

    it(
        "GEXF imports 200k edges through a file stream",
        async () => {
            // observed: 891-981 ms (4.9 us/edge, 15 MiB/s), so about 5 s and 190x the freeze per 1M
            // edges. Profile: fast-xml-parser's tree build (OrderedObjParser, xmlNode, the
            // path-expression matcher) 52%, garbage collection 10%, the well-formedness pre-scan 4%
            const path = gexfDocument(200_000, 20_000);
            const row = await measure("gexf", 200_000, fixtureBytes(path), async (sink) => {
                await gexfImporter.import(fileStream(path, CHUNK), sink);
            });
            expect(row.parseMs).toBeLessThan(10_000);
        },
        LONG,
    );

    it(
        "CSV, Pajek and GraphML export 1M edges as bounded chunks while the heap stays flat",
        async () => {
            // the export side of 8.5: write() is a generator and encodeChunks coalesces its parts into
            // ~64 KiB chunks, so the document is never one string. Observed (chunks written to
            // /dev/null so the loop yields): csv 745 ms, 29 MiB in 467 chunks, peak growth 48 MiB;
            // pajek 490 ms, 22 MiB, peak 21 MiB; graphml 1522 ms, 89 MiB, peak 29 MiB; json, gml,
            // dot and neo4j behave the same (peak 16-23 MiB)
            const snapshot = await snapshotOf1M();
            for (const [name, exporter] of [
                ["csv", csvExporter],
                ["pajek", pajekExporter],
                ["graphml", graphmlExporter],
            ] as const) {
                const run = await exportToDevNull(name, exporter, snapshot);
                expectExportStreams(name, run);
            }
        },
        LONG,
    );

    it(
        "GEXF exports 1M edges as bounded chunks while the heap stays flat (PINS a defect)",
        async () => {
            // FAILS: writeGexf (src/formats/gexf/exporter.ts) pushes every <edge> element into a
            // `parts` array and yields them only after the loop, because <edges count="..."> wants the
            // number of writable edges up front. Observed: 62 MiB of output, peak heap growth
            // 379 MiB, 3 timer samples over 1257 ms. Fix: count the writable edges in a cheap first
            // pass over edgeType() (no strings), then yield each element as it is built; or omit the
            // optional count attribute.
            const snapshot = await snapshotOf1M();
            const run = await exportToDevNull("gexf", gexfExporter, snapshot);
            expectExportStreams("gexf", run);
        },
        LONG,
    );

    it("prints the summary table", () => {
        expect(rows.length).toBeGreaterThan(0);
        const lines = rows.map(
            (r) =>
                `${r.format.padEnd(8)} ${String(r.edges).padStart(8)} edges ${mib(r.inputBytes).padStart(10)} ` +
                `parse ${r.parseMs.toFixed(0).padStart(5)} ms  freeze ${r.freezeMs.toFixed(0).padStart(4)} ms  ` +
                `${((1000 * r.parseMs) / r.edges).toFixed(2)} us/edge  parse/freeze ${(r.parseMs / r.freezeMs).toFixed(0)}x`,
        );
        console.log(`\n${lines.join("\n")}\n`);
    });
});
