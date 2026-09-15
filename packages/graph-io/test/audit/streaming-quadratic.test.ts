/**
 * Streaming audit, complexity lens: every importer must be linear in the input, and its cost
 * must not depend on how the same bytes are chunked (a string, a Uint8Array decoded in 4 MiB
 * slices, or 64 KiB stream chunks are the three shapes design section 8.4 admits). Doubling the
 * input must roughly double the time; a super-linear ratio is a defect, and a chunk-size ratio
 * far above one means some per-token work scans the rest of the buffer.
 *
 * Gated on IO_BENCH=1. Ratios are compared against generous bounds (a doubling may cost up to
 * 2.8x, a chunk shape may cost up to 3x another) so a loaded host does not produce false
 * failures; the pinned defects miss these bounds by an order of magnitude or more.
 */

import { readFileSync } from "node:fs";

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { csvEdgeList, fixtureBytes, gexfDocument, graphmlDocument, pajekNetwork } from "../../benchmarks/fixtures.js";
import { byteChunks } from "../../benchmarks/measure.js";
import { LineReader } from "../../src/common/input.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { csvImporter } from "../../src/formats/csv/index.js";
import { gexfImporter } from "../../src/formats/gexf/index.js";
import { graphmlImporter } from "../../src/formats/graphml/index.js";
import { pajekImporter } from "../../src/formats/pajek/index.js";
import { type GraphImporter, type ImportInput } from "../../src/types.js";

const BENCH = process.env.IO_BENCH === "1";
const LONG = { timeout: 300_000 };

/** The most a doubling of the input may multiply the time by and still count as linear. */
const DOUBLING_BOUND = 2.8;
/** The most one input shape may cost relative to another over the same bytes. */
const SHAPE_BOUND = 3;
const CHUNK = 64 * 1024;

async function timeImport(importer: GraphImporter, input: ImportInput): Promise<{ ms: number; edges: number }> {
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const t0 = performance.now();
    await importer.import(input, sink);
    return { ms: performance.now() - t0, edges: sink.edgeCount };
}

/** The fastest of `runs` timings (the least disturbed by the host). */
async function best(
    runs: number,
    body: () => Promise<{ ms: number; edges: number }>,
): Promise<{ ms: number; edges: number }> {
    let result = await body();
    for (let i = 1; i < runs; i++) {
        const again = await body();
        if (again.ms < result.ms) {
            result = again;
        }
    }
    return result;
}

async function* oneChunk(text: string): AsyncGenerator<string, void, undefined> {
    yield text;
    await Promise.resolve();
}

async function* textPieces(text: string, size: number): AsyncGenerator<string, void, undefined> {
    for (let offset = 0; offset < text.length; offset += size) {
        yield text.slice(offset, offset + size);
        await Promise.resolve();
    }
}

async function timeLines(input: AsyncIterable<string>): Promise<number> {
    const reader = new LineReader(input, new ImportReportBuilder("csv", 100), {});
    const t0 = performance.now();
    let chars = 0;
    for await (const line of reader) {
        chars += line.length;
    }
    expect(chars).toBeGreaterThan(0);
    return performance.now() - t0;
}

describe.skipIf(!BENCH)("streaming audit: linearity and chunk-shape independence (IO_BENCH=1)", () => {
    it(
        "CSV: doubling the edge count doubles the time (250k -> 500k -> 1M, 64 KiB chunks)",
        async () => {
            // observed: 338 / 407 / 952 ms
            const times: number[] = [];
            for (const edges of [250_000, 500_000, 1_000_000]) {
                const bytes = new Uint8Array(readFileSync(csvEdgeList(edges, edges / 10)));
                const run = await best(2, () => timeImport(csvImporter, byteChunks(bytes, CHUNK)));
                expect(run.edges).toBe(edges);
                times.push(run.ms);
            }
            console.log(`csv 250k/500k/1M (64 KiB chunks): ${times.map((t) => t.toFixed(0)).join(" / ")} ms`);
            expect(times[1] / times[0]).toBeLessThan(DOUBLING_BOUND);
            expect(times[2] / times[1]).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "CSV: a string, a Uint8Array and 64 KiB chunks cost about the same (1M edges)",
        async () => {
            // observed: 1532 / 1431 / 952 ms (the in-memory shapes pay for one parse result of 4 MiB
            // to 20 MiB of row arrays at a time, see the throughput report)
            const path = csvEdgeList(1_000_000, 100_000);
            const bytes = new Uint8Array(readFileSync(path));
            const text = new TextDecoder().decode(bytes);
            const chunked = await best(2, () => timeImport(csvImporter, byteChunks(bytes, CHUNK)));
            const asBytes = await best(2, () => timeImport(csvImporter, bytes));
            const asString = await best(2, () => timeImport(csvImporter, text));
            console.log(
                `csv 1M: chunks ${chunked.ms.toFixed(0)} ms, Uint8Array ${asBytes.ms.toFixed(0)} ms, string ${asString.ms.toFixed(0)} ms`,
            );
            expect(asBytes.ms / chunked.ms).toBeLessThan(SHAPE_BOUND);
            expect(asString.ms / chunked.ms).toBeLessThan(SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "GraphML: doubling the edge count doubles the time (100k -> 200k -> 400k, 64 KiB chunks)",
        async () => {
            // observed: 389 / 721 / 1547 ms
            const times: number[] = [];
            for (const edges of [100_000, 200_000, 400_000]) {
                const bytes = new Uint8Array(readFileSync(graphmlDocument(edges, edges / 10)));
                const run = await best(2, () => timeImport(graphmlImporter, byteChunks(bytes, CHUNK)));
                expect(run.edges).toBe(edges);
                times.push(run.ms);
            }
            console.log(`graphml 100k/200k/400k (64 KiB chunks): ${times.map((t) => t.toFixed(0)).join(" / ")} ms`);
            expect(times[1] / times[0]).toBeLessThan(DOUBLING_BOUND);
            expect(times[2] / times[1]).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "GEXF: doubling the edge count doubles the time (100k -> 200k -> 400k, Uint8Array)",
        async () => {
            // observed: 461 / 891 / 2030 ms (linear, but 9x the input in heap: see the memory suite)
            const times: number[] = [];
            for (const edges of [100_000, 200_000, 400_000]) {
                const bytes = new Uint8Array(readFileSync(gexfDocument(edges, edges / 10)));
                const run = await best(2, () => timeImport(gexfImporter, bytes));
                expect(run.edges).toBe(edges);
                times.push(run.ms);
            }
            console.log(`gexf 100k/200k/400k (Uint8Array): ${times.map((t) => t.toFixed(0)).join(" / ")} ms`);
            expect(times[1] / times[0]).toBeLessThan(DOUBLING_BOUND);
            expect(times[2] / times[1]).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "LineReader: the cost of a chunk is linear in its size for LF-terminated text (PINS a defect)",
        async () => {
            // FAILS: LineReader (src/common/input.ts) calls text.indexOf("\r", start) for EVERY line;
            // in a file without any CR (every Unix-written file) the search runs to the end of the
            // chunk each time, so one chunk of L lines costs O(L x chunk length). Observed for
            // `i i*7 1.5` lines: 10k lines in one chunk 15 ms, 20k 45 ms, 40k 125 ms, 80k 511 ms,
            // 160k 2304 ms; the same 160k lines in 64 KiB pieces 55 ms; 1M lines in 4 MiB pieces
            // (the Uint8Array DECODE_SLICE) 24.8 s versus 337 ms in 64 KiB pieces. CRLF text is
            // unaffected (the CR is always near). Fix: search for the next line break once per
            // line (a single charCodeAt loop, or indexOf("\n") and a CR check only on the character
            // before it plus a bounded lone-CR scan), never an unbounded indexOf("\r").
            const lines = 160_000;
            const text = `${Array.from({ length: lines }, (_, i) => `${i} ${i * 7} 1.5`).join("\n")}\n`;
            const pieces = await timeLines(textPieces(text, CHUNK));
            const whole = await timeLines(oneChunk(text));
            console.log(
                `LineReader ${lines} LF lines: 64 KiB pieces ${pieces.toFixed(0)} ms, one chunk ${whole.toFixed(0)} ms (${(whole / pieces).toFixed(1)}x)`,
            );
            expect(whole / pieces, "one chunk versus 64 KiB pieces").toBeLessThan(SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "Pajek: a Uint8Array input costs about the same as 64 KiB chunks (PINS the LineReader defect)",
        async () => {
            // FAILS: the Uint8Array path decodes 4 MiB slices, each a single LineReader chunk, so a
            // Pajek file pays the quadratic scan above. Observed: 100k vertices + 200k arcs (5.5 MB)
            // 270 ms chunked versus 14.7 s as one string; the 1M-arc file (23 MB) 1.2 s chunked
            // versus 68.9 s as a Uint8Array (57x) and longer still as a string.
            const path = pajekNetwork(100_000, 100_000);
            const bytes = new Uint8Array(readFileSync(path));
            const chunked = await best(2, () => timeImport(pajekImporter, byteChunks(bytes, CHUNK)));
            const asBytes = await timeImport(pajekImporter, bytes);
            expect(chunked.edges).toBe(100_000);
            expect(asBytes.edges).toBe(100_000);
            console.log(
                `pajek 100k arcs (${(fixtureBytes(path) / 1048576).toFixed(1)} MiB): chunks ${chunked.ms.toFixed(0)} ms, Uint8Array ${asBytes.ms.toFixed(0)} ms (${(asBytes.ms / chunked.ms).toFixed(1)}x)`,
            );
            expect(asBytes.ms / chunked.ms, "Uint8Array versus 64 KiB chunks").toBeLessThan(SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "GraphML: a document without line breaks costs the same as one with them (PINS a defect)",
        async () => {
            // FAILS: the GraphML tokenizer's advanceLine() (src/formats/graphml/xml.ts) counts lines
            // with buffer.indexOf("\n", start) after every token; when the buffer holds no further
            // line break the search runs to its end, so a single-line document costs O(tokens x
            // buffer) per buffer. Observed for the 25k-edge document (2.2 MB): 137 ms in 64 KiB
            // chunks, 1794 ms as one string; 50k edges 9.3 s, 100k edges (8.7 MB) 41 s as a string
            // and 33 s as a Uint8Array, against 479 ms for the same document with line breaks.
            // Fix: bound the scan to `end` (loop with charCodeAt over [start, end), or remember the
            // position of the next line break and only re-search when the buffer is refilled).
            const oneLine = readFileSync(graphmlDocument(25_000, 2_500, true), "utf8");
            const bytes = new TextEncoder().encode(oneLine);
            const chunked = await best(2, () => timeImport(graphmlImporter, byteChunks(bytes, CHUNK)));
            const asString = await timeImport(graphmlImporter, oneLine);
            const multiLine = await best(2, () =>
                timeImport(graphmlImporter, readFileSync(graphmlDocument(25_000, 2_500), "utf8")),
            );
            expect(chunked.edges).toBe(25_000);
            expect(asString.edges).toBe(25_000);
            console.log(
                `graphml 25k one-line: chunks ${chunked.ms.toFixed(0)} ms, string ${asString.ms.toFixed(0)} ms (${(asString.ms / chunked.ms).toFixed(1)}x); multi-line string ${multiLine.ms.toFixed(0)} ms`,
            );
            expect(asString.ms / chunked.ms, "one-line string versus 64 KiB chunks").toBeLessThan(SHAPE_BOUND);
            expect(asString.ms / multiLine.ms, "one-line versus multi-line string").toBeLessThan(SHAPE_BOUND);
        },
        LONG,
    );
});
