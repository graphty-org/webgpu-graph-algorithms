/**
 * Streaming audit, memory lens (design section 8.4: line- and record-oriented importers read a
 * byte stream and never buffer the whole input; GEXF and GraphML are single-pass SAX-style).
 *
 * Each importer reads a generated fixture (see benchmarks/fixtures.ts) through a ReadableStream
 * whose reads are real asynchronous file reads, while a 100 ms timer samples heapUsed; a second
 * pass feeds the same bytes through an async iterable that forces a full GC every 4 MiB and
 * records the heap still in use, the chunk-size-independent high-water mark of what the importer
 * retains while streaming. A whole-input buffer shows up as retained heap of at least the input
 * size (the decoded text alone is one byte per character); a parser that never yields to the
 * event loop shows up as zero timer samples.
 *
 * Gated on IO_BENCH=1: the fixtures are 20-40 MB each and the suite takes about a minute.
 * Measured on Node 22.22.1 / i9-14900; the thresholds leave a wide margin over the observations
 * quoted in each test.
 */

import { readFileSync } from "node:fs";

import { GraphBuilder, type NodeId } from "@graphty/graph-format";
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
import {
    byteChunks,
    fileStream,
    fullGc,
    type HeapProfile,
    mib,
    retainedWhileStreaming,
    sampleHeap,
} from "../../benchmarks/measure.js";
import { csvImporter } from "../../src/formats/csv/index.js";
import { gexfImporter } from "../../src/formats/gexf/index.js";
import { graphmlImporter } from "../../src/formats/graphml/index.js";
import { neo4jImporter } from "../../src/formats/neo4j/index.js";
import { pajekImporter } from "../../src/formats/pajek/index.js";
import { type GraphImporter, type ImportInput } from "../../src/types.js";

const BENCH = process.env.IO_BENCH === "1";
const LONG = { timeout: 180_000 };

const MIB = 1048576;
const CHUNK = 64 * 1024;
/** Chunks between forced-GC readings: 64 x 64 KiB = 4 MiB of input. */
const GC_EVERY = 64;

function builder(): GraphBuilder {
    return new GraphBuilder({ directed: true, weightDtype: "f64" });
}

function describeProfile(label: string, inputBytes: number, p: HeapProfile): void {
    const growth = p.peakHeap - p.baseHeap;
    const retained = p.retainedHeap - p.baseHeap;
    console.log(
        `${label}: input ${mib(inputBytes)}, ${p.ms.toFixed(0)} ms, ${p.samples} timer samples, ` +
            `peak heap growth ${mib(growth)} (${((100 * growth) / inputBytes).toFixed(0)}% of input), ` +
            `retained after GC ${mib(retained)} (${((100 * retained) / inputBytes).toFixed(0)}%)`,
    );
}

/**
 * Feed a file's bytes through the GC-probing iterable and return the high-water mark of the
 * retained heap over the base, in bytes.
 */
async function retainedHighWater(
    importer: GraphImporter,
    path: string,
    sink: GraphBuilder,
    chunkBytes: number = CHUNK,
): Promise<{ readonly highWater: number; readonly readings: number }> {
    const bytes = new Uint8Array(readFileSync(path));
    const readings: number[] = [];
    fullGc();
    const base = process.memoryUsage().heapUsed;
    await importer.import(retainedWhileStreaming(byteChunks(bytes, chunkBytes), GC_EVERY, readings), sink);
    fullGc();
    readings.push(process.memoryUsage().heapUsed);
    return { highWater: Math.max(...readings) - base, readings: readings.length };
}

/**
 * A builder that records how much heap is live when the first edge arrives: everything the reader
 * materialised before delivering its first row is still reachable at that moment, whatever the
 * garbage collector did.
 */
class FirstEdgeProbe extends GraphBuilder {
    readonly baseHeap: number;

    growthAtFirstEdge = -1;

    constructor() {
        super({ directed: true, weightDtype: "f64" });
        fullGc();
        this.baseHeap = process.memoryUsage().heapUsed;
    }

    override addEdge(source: NodeId, target: NodeId, weight?: number): number {
        if (this.growthAtFirstEdge < 0) {
            this.growthAtFirstEdge = process.memoryUsage().heapUsed - this.baseHeap;
        }
        return super.addEdge(source, target, weight);
    }
}

/**
 * The sampled-peak bound of a streaming parse: young-generation garbage between scavenges is
 * bounded by the two 16 MiB semispaces, not by the input, so for an input of tens of MiB the
 * semispace allowance dominates twice the input (the GEXF 200k fixture is 14.8 MiB: twice it is
 * 29.7 MiB, and a scavenge that lands late puts the sampled peak at 33 MiB with 8 MiB retained).
 * A whole-input buffer plus its parse products (9x the input before the streaming rewrite) stays
 * far beyond either term.
 * @param inputBytes - the input size
 * @returns the largest acceptable sampled heap growth
 */
function peakBound(inputBytes: number): number {
    return Math.max(2 * inputBytes, inputBytes + 32 * MIB);
}

/** The assertions every line- or record-oriented importer must satisfy over a 64 KiB stream. */
function expectStreams(label: string, inputBytes: number, p: HeapProfile, highWater: number): void {
    // the parse yields to the event loop between chunks (a whole-input parse never lets the timer run)
    expect(p.samples, `${label}: timer samples during a ${p.ms.toFixed(0)} ms import`).toBeGreaterThanOrEqual(3);
    // a streaming parse peaks at the semispace allowance over the retained graph; a whole-input
    // buffer plus its parse products exceeds it (see peakBound)
    expect(p.peakHeap - p.baseHeap, `${label}: sampled peak heap growth`).toBeLessThan(peakBound(inputBytes));
    // what stays after a GC at any point of the stream is graph data, never the input
    expect(highWater, `${label}: retained heap high-water mark while streaming`).toBeLessThan(inputBytes);
}

describe.skipIf(!BENCH)("streaming audit: memory (IO_BENCH=1)", () => {
    it(
        "CSV: 1M edges through a 64 KiB file stream keeps the retained heap flat",
        async () => {
            // observed: 958 ms, 9 samples, peak growth 17.9 MiB (86%), retained after GC 5.7 MiB (id map)
            const path = csvEdgeList(1_000_000, 100_000);
            const inputBytes = fixtureBytes(path);
            const sink = builder();
            const profile = await sampleHeap(() => csvImporter.import(fileStream(path, CHUNK), sink));
            describeProfile("csv 1M stream", inputBytes, profile);
            expect(sink.edgeCount).toBe(1_000_000);
            const { highWater } = await retainedHighWater(csvImporter, path, builder());
            console.log(`csv 1M retained high-water while streaming: ${mib(highWater)}`);
            expectStreams("csv 1M", inputBytes, profile, highWater);
        },
        LONG,
    );

    it(
        "Pajek: 1M arcs through a 64 KiB file stream keeps the retained heap flat",
        async () => {
            // observed: 1170 ms, 11 samples, peak growth 24.4 MiB (110%), retained 9.0 MiB (labels)
            const path = pajekNetwork(1_000_000, 100_000);
            const inputBytes = fixtureBytes(path);
            const sink = builder();
            const profile = await sampleHeap(() => pajekImporter.import(fileStream(path, CHUNK), sink));
            describeProfile("pajek 1M stream", inputBytes, profile);
            expect(sink.edgeCount).toBe(1_000_000);
            const { highWater } = await retainedHighWater(pajekImporter, path, builder());
            console.log(`pajek 1M retained high-water while streaming: ${mib(highWater)}`);
            expectStreams("pajek 1M", inputBytes, profile, highWater);
        },
        LONG,
    );

    it(
        "Neo4j: 100k nodes + 1M relationships through 64 KiB file streams keep the retained heap flat",
        async () => {
            // observed: 1645 ms, 16 samples, peak growth 34.5 MiB (121%), retained 21.4 MiB (the id
            // property column, the name column and the labels list column of 100k nodes)
            const nodes = neo4jNodes(100_000);
            const rels = neo4jRelationships(1_000_000, 100_000);
            const inputBytes = fixtureBytes(nodes) + fixtureBytes(rels);
            const sink = builder();
            const profile = await sampleHeap(async () => {
                await neo4jImporter.import(fileStream(nodes, CHUNK), sink);
                await neo4jImporter.import(fileStream(rels, CHUNK), sink);
            });
            describeProfile("neo4j 100k+1M stream", inputBytes, profile);
            expect(sink.nodeCount).toBe(100_000);
            expect(sink.edgeCount).toBe(1_000_000);
            const second = builder();
            const nodesRun = await retainedHighWater(neo4jImporter, nodes, second);
            const relsRun = await retainedHighWater(neo4jImporter, rels, second);
            const highWater = Math.max(nodesRun.highWater, relsRun.highWater);
            console.log(`neo4j retained high-water while streaming: ${mib(highWater)}`);
            expectStreams("neo4j", inputBytes, profile, highWater);
        },
        LONG,
    );

    it(
        "GraphML: 200k edges through a 64 KiB file stream keeps the retained heap flat",
        async () => {
            // observed: 787 ms, 7 samples, peak growth 29 MiB (161%), retained 10 MiB (edge ids)
            const path = graphmlDocument(200_000, 20_000);
            const inputBytes = fixtureBytes(path);
            const sink = builder();
            const profile = await sampleHeap(() => graphmlImporter.import(fileStream(path, CHUNK), sink));
            describeProfile("graphml 200k stream", inputBytes, profile);
            expect(sink.edgeCount).toBe(200_000);
            const { highWater } = await retainedHighWater(graphmlImporter, path, builder());
            console.log(`graphml 200k retained high-water while streaming: ${mib(highWater)}`);
            expectStreams("graphml 200k", inputBytes, profile, highWater);
        },
        LONG,
    );

    it(
        "GEXF: 200k edges through a 64 KiB file stream is single-pass and bounded (PINS a defect)",
        async () => {
            // FAILS: the GEXF importer reads the whole stream into one string (readText) and hands it
            // to fast-xml-parser's preserveOrder tree. Observed: 0 timer samples over 981 ms and a
            // heap growth of 137 MiB for a 14.8 MiB input (9.2x), all of it live until the walk ends.
            // Design 8.4 says GEXF is single-pass SAX-style; 8.2 allows fast-xml-parser only as the
            // initial step "replaced by hand-written streaming tokenisers per format".
            const path = gexfDocument(200_000, 20_000);
            const inputBytes = fixtureBytes(path);
            const sink = builder();
            const profile = await sampleHeap(() => gexfImporter.import(fileStream(path, CHUNK), sink));
            describeProfile("gexf 200k stream", inputBytes, profile);
            expect(sink.edgeCount).toBe(200_000);
            expect(profile.samples, "gexf: timer samples during the import").toBeGreaterThanOrEqual(3);
            expect(profile.peakHeap - profile.baseHeap, "gexf: heap growth").toBeLessThan(peakBound(inputBytes));
        },
        LONG,
    );

    it(
        "CSV as one string: rows are delivered incrementally, not as one parse result (PINS a defect)",
        async () => {
            // FAILS: CsvRecordReader hands papaparse the whole carry in one parse() call, so a string
            // input (and each 4 MiB decode slice of a Uint8Array) is turned into every row array at
            // once before the first row reaches the sink. Observed at the first addEdge: 362 MiB of
            // live heap over the 20.7 MiB string (1M line slices, 1M row arrays, 3M cell slices)
            // against 1.4 MiB with 64 KiB chunks; `node --max-old-space-size=160` dies with "heap
            // out of memory" on the string input, needs 256 MiB (GC-thrashing, 3.8 s) to finish, and
            // finishes the stream input inside 96 MiB. Fix: parse the carry in bounded windows cut at
            // a line break (256 KiB keeps the live rows in the low thousands), never the whole
            // in-memory input (or a whole 4 MiB decode slice) in one parse() call.
            const path = csvEdgeList(1_000_000, 100_000);
            const text = new TextDecoder().decode(new Uint8Array(readFileSync(path)));
            const probe = new FirstEdgeProbe();
            await csvImporter.import(text, probe);
            const chunked = new FirstEdgeProbe();
            await csvImporter.import(byteChunks(new Uint8Array(readFileSync(path)), CHUNK), chunked);
            console.log(
                `csv 1M live heap at the first addEdge: string ${mib(probe.growthAtFirstEdge)}, 64 KiB chunks ${mib(chunked.growthAtFirstEdge)}`,
            );
            expect(probe.edgeCount).toBe(1_000_000);
            expect(chunked.growthAtFirstEdge).toBeLessThan(0.25 * text.length);
            expect(probe.growthAtFirstEdge, "live heap at the first addEdge, string input").toBeLessThan(
                0.25 * text.length,
            );
        },
        LONG,
    );

    it(
        "CSV with 12-character string ids: the id map, not the input, is what survives the import",
        async () => {
            // the control for the next test: 12-character ids are below V8's SlicedString minimum
            // (13), so a cell sliced out of a chunk is a copy. Observed: retained 7.4 MiB whatever the
            // chunk size (34 MiB input, 100k ids)
            const path = csvEdgeList(1_000_000, 100_000, "short-string");
            const inputBytes = fixtureBytes(path);
            const keep = builder();
            const { highWater } = await retainedHighWater(csvImporter, path, keep, MIB);
            console.log(`csv 12-char ids, 1 MiB chunks: retained ${mib(highWater)} of ${mib(inputBytes)} input`);
            expect(keep.nodeCount).toBe(100_000);
            expect(highWater).toBeLessThan(0.5 * inputBytes);
        },
        LONG,
    );

    it(
        "CSV with 15-character string ids: the ids pin the decoded input chunks (PINS a defect)",
        async () => {
            // FAILS: every retained string of 13+ characters that papaparse / LineReader / the record
            // readers slice out of a decoded chunk is a V8 SlicedString pointing at that chunk, and the
            // builder's id map (and any string / dict column) keeps it. With 1 MiB chunks, 4 MiB
            // Uint8Array decode slices or a string input, the whole decoded text stays resident for
            // the life of the snapshot. Observed: retained 47 MiB for a 40 MiB input (119%) with 1 MiB
            // chunks and with a Uint8Array; 25 MiB (62%) with 64 KiB chunks; 7.4 MiB with 12-char ids.
            // Fix: flatten a text before it is retained (ids in IdCoercer / the builder's addNode on a
            // Map miss, cell texts in InferredColumn and the declared-column writers), e.g.
            // `(" " + s).slice(1)` (12 ns) or JSON round trip (100 ns), only for s.length >= 13.
            const path = csvEdgeList(1_000_000, 100_000, "long-string");
            const inputBytes = fixtureBytes(path);
            const keep = builder();
            const { highWater } = await retainedHighWater(csvImporter, path, keep, MIB);
            console.log(`csv 15-char ids, 1 MiB chunks: retained ${mib(highWater)} of ${mib(inputBytes)} input`);
            expect(keep.nodeCount).toBe(100_000);
            expect(highWater, "retained heap with 1 MiB chunks").toBeLessThan(0.5 * inputBytes);
        },
        LONG,
    );

    it(
        "CSV with 15-character string ids as a Uint8Array: the snapshot retains the decoded text (PINS the same defect)",
        async () => {
            // FAILS: the Uint8Array path decodes 4 MiB slices; every slice holds a first-seen id, so
            // the whole decoded text (36-40 MiB) is retained through the id map even after
            // freeze({ release: true }). Observed: 47.9 MiB retained for a 36 MiB input.
            const path = csvEdgeList(1_000_000, 100_000, "long-string");
            const bytes = new Uint8Array(readFileSync(path));
            const input: ImportInput = bytes;
            fullGc();
            const base = process.memoryUsage().heapUsed;
            const keep = builder();
            await csvImporter.import(input, keep);
            const snapshot = keep.freeze({ release: true });
            fullGc();
            const retained = process.memoryUsage().heapUsed - base;
            console.log(
                `csv 15-char ids, Uint8Array: retained ${mib(retained)} after freeze(release) of ${mib(bytes.byteLength)} input`,
            );
            expect(snapshot.nodeCount).toBe(100_000);
            expect(retained, "retained heap after freeze(release)").toBeLessThan(0.5 * bytes.byteLength);
        },
        LONG,
    );
});
