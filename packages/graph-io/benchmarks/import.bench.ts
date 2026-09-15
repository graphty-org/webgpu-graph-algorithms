/**
 * The graph-io pipeline benchmark of design section 15.5 item 5 (parse -> builder -> freeze end to
 * end) plus the streaming checks of section 8.4: every importer over its generated fixture
 * (benchmarks/fixtures.ts) through a 64 KiB file stream, as a Uint8Array and as a string, with the
 * median of `runs` wall times, the timer-sampled heap peak and the heap retained after a full GC;
 * the doubling ratios that expose super-linear behaviour; and every exporter over a 1M-edge
 * snapshot written to /dev/null in chunks.
 *
 *   pnpm run benchmark -- [--runs N] [--quick]        (benchmarks/run.ts calls runImportBenchmarks)
 *
 * `--quick` runs the 1M-edge fixtures once and skips the doubling series. Results are printed as
 * a table and appended as JSON to benchmarks/results/io-<host>-node<version>.json.
 */

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { open } from "node:fs/promises";
import { cpus, hostname } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { csvExporter, csvImporter } from "../src/formats/csv/index.js";
import { dotExporter } from "../src/formats/dot/index.js";
import { gexfExporter, gexfImporter } from "../src/formats/gexf/index.js";
import { gmlExporter } from "../src/formats/gml/index.js";
import { graphmlExporter, graphmlImporter } from "../src/formats/graphml/index.js";
import { jsonExporter } from "../src/formats/json/index.js";
import { neo4jExporter, neo4jImporter } from "../src/formats/neo4j/index.js";
import { pajekExporter, pajekImporter } from "../src/formats/pajek/index.js";
import { type GraphExporter, type GraphImporter, type ImportInput } from "../src/types.js";
import {
    csvEdgeList,
    fixtureBytes,
    gexfDocument,
    graphmlDocument,
    neo4jNodes,
    neo4jRelationships,
    pajekNetwork,
} from "./fixtures.js";
import { byteChunks, fileStream, fullGc, mib, sampleHeap } from "./measure.js";

const CHUNK = 64 * 1024;

let quick = false;
let RUNS = 3;

/** One benchmark row. */
interface Result {
    readonly group: string;
    readonly name: string;
    readonly edges: number;
    readonly inputBytes: number;
    readonly medianMs: number;
    readonly minMs: number;
    readonly peakHeapBytes: number;
    readonly retainedHeapBytes: number;
    readonly timerSamples: number;
    readonly freezeMs: number | null;
}

const results: Result[] = [];

/**
 * Print one row and keep it for the JSON record.
 * @param r - the row
 */
function record(r: Result): void {
    results.push(r);
    const perEdge = r.edges > 0 ? `${((1000 * r.medianMs) / r.edges).toFixed(2)} us/edge` : "";
    const rate = `${(r.inputBytes / 1048576 / (r.medianMs / 1000)).toFixed(1)} MiB/s`;
    const freeze =
        r.freezeMs === null ? "" : `freeze ${r.freezeMs.toFixed(0)} ms (${(r.medianMs / r.freezeMs).toFixed(0)}x)`;
    console.log(
        `${r.group.padEnd(8)} ${r.name.padEnd(44)} ${mib(r.inputBytes).padStart(10)} ` +
            `median ${r.medianMs.toFixed(0).padStart(6)} ms  min ${r.minMs.toFixed(0).padStart(6)} ms  ${perEdge.padStart(14)} ${rate.padStart(11)}  ` +
            `peak +${mib(r.peakHeapBytes).padStart(10)}  retained ${mib(r.retainedHeapBytes).padStart(10)}  samples ${String(r.timerSamples).padStart(3)}  ${freeze}`,
    );
}

/** The input shapes an importer is measured with. */
type Shape = "stream" | "bytes" | "string";

/**
 * Build the input of one shape for a fixture.
 * @param path - the fixture
 * @param shape - the shape
 * @param bytes - the file content, for the in-memory shapes
 * @returns the input
 */
function inputOf(path: string, shape: Shape, bytes: Uint8Array): ImportInput {
    switch (shape) {
        case "stream":
            return fileStream(path, CHUNK);
        case "bytes":
            return bytes;
        case "string":
            return new TextDecoder().decode(bytes);
        default:
            return bytes;
    }
}

/**
 * Measure an importer over one or more fixtures read into one builder, `RUNS` times, and freeze
 * the builder of the last run.
 * @param group - the format
 * @param name - the row name
 * @param paths - the fixtures, read in order into one builder
 * @param importer - the importer
 * @param shape - the input shape
 * @param expectedEdges - the edge count every run must produce
 */
async function benchImport(
    group: string,
    name: string,
    paths: readonly string[],
    importer: GraphImporter,
    shape: Shape,
    expectedEdges: number,
): Promise<void> {
    const contents = paths.map((p) => new Uint8Array(readFileSync(p)));
    const inputBytes = paths.reduce((sum, p) => sum + fixtureBytes(p), 0);
    const times: number[] = [];
    let peak = 0;
    let retained = 0;
    let samples = 0;
    let freezeMs: number | null = null;
    for (let run = 0; run < RUNS; run++) {
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const profile = await sampleHeap(async () => {
            for (let i = 0; i < paths.length; i++) {
                await importer.import(inputOf(paths[i], shape, contents[i]), sink);
            }
        });
        if (sink.edgeCount !== expectedEdges) {
            throw new Error(`${group} ${name}: ${sink.edgeCount} edges, expected ${expectedEdges}`);
        }
        times.push(profile.ms);
        peak = Math.max(peak, profile.peakHeap - profile.baseHeap);
        retained = profile.retainedHeap - profile.baseHeap;
        samples = profile.samples;
        if (run === RUNS - 1) {
            const t0 = performance.now();
            sink.freeze();
            freezeMs = performance.now() - t0;
        }
    }
    times.sort((a, b) => a - b);
    record({
        group,
        name,
        edges: expectedEdges,
        inputBytes,
        medianMs: times[Math.floor(times.length / 2)],
        minMs: times[0],
        peakHeapBytes: peak,
        retainedHeapBytes: retained,
        timerSamples: samples,
        freezeMs,
    });
}

/**
 * Measure an exporter writing a snapshot to /dev/null chunk by chunk.
 * @param name - the format
 * @param exporter - the exporter
 * @param snapshot - the snapshot
 */
async function benchExport(name: string, exporter: GraphExporter, snapshot: GraphSnapshot): Promise<void> {
    const times: number[] = [];
    let peak = 0;
    let retained = 0;
    let samples = 0;
    let bytes = 0;
    for (let run = 0; run < RUNS; run++) {
        const out = await open("/dev/null", "w");
        bytes = 0;
        const profile = await sampleHeap(async () => {
            for await (const chunk of exporter.export(snapshot)) {
                bytes += chunk.byteLength;
                await out.write(chunk);
            }
        });
        await out.close();
        times.push(profile.ms);
        peak = Math.max(peak, profile.peakHeap - profile.baseHeap);
        retained = profile.retainedHeap - profile.baseHeap;
        samples = profile.samples;
    }
    times.sort((a, b) => a - b);
    record({
        group: name,
        name: `export 1M edges to /dev/null`,
        edges: snapshot.edgeCount,
        inputBytes: bytes,
        medianMs: times[Math.floor(times.length / 2)],
        minMs: times[0],
        peakHeapBytes: peak,
        retainedHeapBytes: retained,
        timerSamples: samples,
        freezeMs: null,
    });
}

/**
 * The doubling series of one importer: the ratio of consecutive timings, which stays near 2 for
 * a linear parser.
 * @param group - the format
 * @param importer - the importer
 * @param sizes - the edge counts, each double the previous
 * @param fixtureOf - the fixture for a size
 * @param shape - the input shape
 */
async function doubling(
    group: string,
    importer: GraphImporter,
    sizes: readonly number[],
    fixtureOf: (edges: number) => string,
    shape: Shape,
): Promise<void> {
    const times: number[] = [];
    for (const edges of sizes) {
        const path = fixtureOf(edges);
        const bytes = new Uint8Array(readFileSync(path));
        let bestMs = Infinity;
        for (let run = 0; run < RUNS; run++) {
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const t0 = performance.now();
            await importer.import(shape === "stream" ? byteChunks(bytes, CHUNK) : inputOf(path, shape, bytes), sink);
            bestMs = Math.min(bestMs, performance.now() - t0);
        }
        times.push(bestMs);
    }
    const ratios = times.slice(1).map((t, i) => (t / times[i]).toFixed(2));
    console.log(
        `${group.padEnd(8)} doubling ${shape.padEnd(6)} ${sizes.map((s) => `${s / 1000}k`).join(" -> ")}: ` +
            `${times.map((t) => `${t.toFixed(0)} ms`).join(" / ")}  ratios ${ratios.join(", ")}`,
    );
}

/**
 * Run every group and append the JSON record.
 * @param args - the command-line arguments: `--quick` (one run, no doubling series), `--runs N`
 */
export async function runImportBenchmarks(args: readonly string[]): Promise<void> {
    quick = args.includes("--quick");
    const runsArg = args.indexOf("--runs");
    RUNS = quick ? 1 : runsArg >= 0 ? Number(args[runsArg + 1]) : 3;
    results.length = 0;
    console.log(
        `graph-io import benchmark: ${hostname()}, ${cpus()[0]?.model ?? "unknown cpu"}, Node ${process.version}, runs ${RUNS}${quick ? " (quick)" : ""}`,
    );
    console.log("generating fixtures (first run only)...");
    const csv1m = csvEdgeList(1_000_000, 100_000);
    const pajek1m = pajekNetwork(1_000_000, 100_000);
    const neo4jNodes100k = neo4jNodes(100_000);
    const neo4jRels1m = neo4jRelationships(1_000_000, 100_000);
    const gexf200k = gexfDocument(200_000, 20_000);
    const graphml200k = graphmlDocument(200_000, 20_000);

    console.log("\n== importers, 64 KiB file stream ==");
    await benchImport("csv", "1M edges, numeric ids", [csv1m], csvImporter, "stream", 1_000_000);
    await benchImport("pajek", "100k vertices + 1M arcs", [pajek1m], pajekImporter, "stream", 1_000_000);
    await benchImport(
        "neo4j",
        "100k nodes + 1M relationships",
        [neo4jNodes100k, neo4jRels1m],
        neo4jImporter,
        "stream",
        1_000_000,
    );
    await benchImport("graphml", "200k edges", [graphml200k], graphmlImporter, "stream", 200_000);
    await benchImport("gexf", "200k edges", [gexf200k], gexfImporter, "stream", 200_000);

    console.log("\n== importers, in-memory shapes ==");
    await benchImport("csv", "1M edges, Uint8Array", [csv1m], csvImporter, "bytes", 1_000_000);
    await benchImport("csv", "1M edges, string", [csv1m], csvImporter, "string", 1_000_000);
    await benchImport("graphml", "200k edges, Uint8Array", [graphml200k], graphmlImporter, "bytes", 200_000);
    await benchImport("gexf", "200k edges, Uint8Array", [gexf200k], gexfImporter, "bytes", 200_000);
    await benchImport("neo4j", "1M relationships, Uint8Array", [neo4jRels1m], neo4jImporter, "bytes", 1_000_000);
    // Pajek as a Uint8Array is left out on purpose: the LineReader scan is quadratic in the 4 MiB
    // decode slice and the 1M-arc file takes over a minute (see test/audit/streaming-quadratic.test.ts)

    console.log("\n== string ids: 12 characters (copied cells) versus 15 (V8 sliced strings pinning the chunks) ==");
    await benchImport(
        "csv",
        "1M edges, 12-char string ids, stream",
        [csvEdgeList(1_000_000, 100_000, "short-string")],
        csvImporter,
        "stream",
        1_000_000,
    );
    await benchImport(
        "csv",
        "1M edges, 15-char string ids, stream",
        [csvEdgeList(1_000_000, 100_000, "long-string")],
        csvImporter,
        "stream",
        1_000_000,
    );
    await benchImport(
        "csv",
        "1M edges, 15-char string ids, Uint8Array",
        [csvEdgeList(1_000_000, 100_000, "long-string")],
        csvImporter,
        "bytes",
        1_000_000,
    );

    if (!quick) {
        console.log("\n== doubling series ==");
        await doubling("csv", csvImporter, [250_000, 500_000, 1_000_000], (e) => csvEdgeList(e, e / 10), "stream");
        await doubling(
            "graphml",
            graphmlImporter,
            [50_000, 100_000, 200_000, 400_000],
            (e) => graphmlDocument(e, e / 10),
            "stream",
        );
        await doubling(
            "graphml",
            graphmlImporter,
            [12_500, 25_000, 50_000],
            (e) => graphmlDocument(e, e / 10, true),
            "string",
        );
        await doubling(
            "gexf",
            gexfImporter,
            [50_000, 100_000, 200_000, 400_000],
            (e) => gexfDocument(e, e / 10),
            "bytes",
        );
        await doubling("pajek", pajekImporter, [25_000, 50_000, 100_000], (e) => pajekNetwork(e, 100_000), "stream");
        await doubling("pajek", pajekImporter, [25_000, 50_000, 100_000], (e) => pajekNetwork(e, 100_000), "string");
    }

    console.log("\n== exporters, 1M-edge snapshot (from the CSV fixture) ==");
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await csvImporter.import(fileStream(csv1m, CHUNK), sink);
    const snapshot = sink.freeze({ release: true });
    fullGc();
    for (const [name, exporter] of [
        ["csv", csvExporter],
        ["pajek", pajekExporter],
        ["neo4j", neo4jExporter],
        ["graphml", graphmlExporter],
        ["gexf", gexfExporter],
        ["gml", gmlExporter],
        ["dot", dotExporter],
        ["json", jsonExporter],
    ] as const) {
        await benchExport(name, exporter, snapshot);
    }

    // the results file is a JSON array of sessions (the graph-format convention), pretty-printed so
    // it stays a valid, formatted JSON document
    const resultsDir = join(dirname(fileURLToPath(import.meta.url)), "results");
    mkdirSync(resultsDir, { recursive: true });
    const file = join(resultsDir, `io-${hostname()}-node${process.versions.node}.json`);
    const session = {
        at: new Date().toISOString(),
        node: process.version,
        cpu: cpus()[0]?.model ?? null,
        runs: RUNS,
        results,
    };
    const sessions: unknown[] = existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as unknown[]) : [];
    sessions.push(session);
    writeFileSync(file, `${JSON.stringify(sessions, null, 4)}\n`);
    console.log(`\nappended ${results.length} rows to ${file}`);
}
