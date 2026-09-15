/**
 * Fuzz audit, hang lens (design section 8.4: streaming importers run "over a byte stream" with
 * bounded memory; a hang beyond 10 s on a 50 MB value is a finding): one token that spans many
 * chunks must cost linear time in its length, not in its length times the number of chunks it
 * spans. Three readers re-scan their whole carry on every chunk while a token is incomplete, so
 * a 50 MB quoted cell, a 50 MB line or a 50 MB attribute value arriving in 16 KB chunks (the
 * chunk size of a fetch body in Chromium; File.stream() gives 64 KB) takes 30-40 s instead of
 * well under one second:
 *
 * - LineReader (src/common/input.ts): `text = carry + chunk` then `indexOf("\n", 0)` over the
 *   whole carry on every chunk of a line that has no terminator yet (Pajek);
 * - CsvRecordReader (src/formats/csv/records.ts): `carry += chunk` and papaparse re-parses the
 *   carry from offset 0 on every chunk while a quoted cell (or an unclosed quote) is open;
 * - XmlTokenizer (src/formats/graphml/xml.ts): `buffer + text` then consumeMarkup() re-reads the
 *   incomplete start tag, comment, CDATA or name from its `<` on every push.
 *
 * The Neo4j record reader (a per-character state machine that keeps only the open field) is the
 * linear control. A fourth quadratic lives in the core: GraphBuilder resolves column names by a
 * linear scan (handleOf / checkDeclaration in graph-format's graph-builder.ts), so a 100k-column
 * CSV or Neo4j header, or a JSON node with 100k keys, takes 50-130 s; every importer that
 * declares columns from the input reaches it.
 *
 * The ungated tests pin the complexity with size ratios (doubling a token must at most triple
 * the time) and shape ratios (16 KB chunks against one chunk) at 4-8 MB, taking a few seconds
 * in total; the absolute 50 MB / 10 s checks of the task run under IO_BENCH=1.
 */

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { LineReader } from "../../src/common/input.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { XmlTokenizer } from "../../src/common/xml.js";
import { CsvRecordReader, RecordReader as Neo4jRecordReader } from "../../src/formats/csv/records.js";
import { registry } from "../../src/registry.js";
import { ImportError } from "../../src/types.js";

const MB = 1024 * 1024;
const CHUNK = 16 * 1024;
const BENCH = process.env.IO_BENCH === "1";
const LONG = { timeout: 600_000 };

/** Doubling the token may at most triple the time and still count as linear (2x plus noise). */
const DOUBLING_BOUND = 3;
/** Chunking a token into 16 KB pieces may cost at most this much more than one piece. */
const SHAPE_BOUND = 8;

async function* pieces(text: string, size: number): AsyncGenerator<string, void, undefined> {
    for (let i = 0; i < text.length; i += size) {
        yield text.slice(i, i + size);
        await Promise.resolve();
    }
}

async function* onePiece(text: string): AsyncGenerator<string, void, undefined> {
    yield text;
    await Promise.resolve();
}

function chunkedBytes(text: string, size: number): ReadableStream<Uint8Array> {
    const bytes = new TextEncoder().encode(text);
    let offset = 0;
    return new ReadableStream<Uint8Array>({
        pull(controller): void {
            if (offset >= bytes.byteLength) {
                controller.close();
                return;
            }
            controller.enqueue(bytes.subarray(offset, Math.min(offset + size, bytes.byteLength)));
            offset += size;
        },
    });
}

async function best(runs: number, body: () => Promise<number>): Promise<number> {
    let result = await body();
    for (let i = 1; i < runs; i++) {
        result = Math.min(result, await body());
    }
    return result;
}

async function timeLines(input: AsyncIterable<string>): Promise<number> {
    const reader = new LineReader(input, new ImportReportBuilder("audit", 100));
    const t0 = performance.now();
    let lines = 0;
    for await (const _line of reader) {
        lines++;
    }
    expect(lines).toBeGreaterThan(0);
    return performance.now() - t0;
}

async function timeCsv(input: AsyncIterable<string>): Promise<number> {
    const reader = new CsvRecordReader(input, new ImportReportBuilder("csv", 100));
    const t0 = performance.now();
    let rows = 0;
    for await (const _row of reader) {
        rows++;
    }
    expect(rows).toBe(2);
    return performance.now() - t0;
}

function timeXml(text: string, size: number): number {
    const tokenizer = new XmlTokenizer({ start(): void {}, end(): void {}, text(): void {} });
    const t0 = performance.now();
    for (let i = 0; i < text.length; i += size) {
        tokenizer.push(text.slice(i, i + size));
    }
    tokenizer.finish();
    return performance.now() - t0;
}

async function timeNeo4j(input: AsyncIterable<string>): Promise<number> {
    const reader = new Neo4jRecordReader(
        input,
        new ImportReportBuilder("neo4j", 100),
        { delimiter: ",", quote: '"' },
        {},
    );
    const t0 = performance.now();
    let rows = 0;
    for await (const _cells of reader) {
        rows++;
    }
    expect(rows).toBe(3);
    return performance.now() - t0;
}

function line(mb: number): string {
    return `1 "${"x".repeat(mb * MB)}"\n`;
}

function csvCell(mb: number): string {
    return `source,target,label\na,b,"${"x".repeat(mb * MB)}"\n`;
}

function xmlAttribute(mb: number): string {
    return `<graphml><graph><node id="${"x".repeat(mb * MB)}"/></graph></graphml>`;
}

describe("fuzz audit: a token spanning many chunks costs linear time", () => {
    it(
        "LineReader: doubling a line that spans 16 KB chunks at most triples the time (PINS a defect)",
        async () => {
            // FAILS: observed 4 MB 179 ms, 8 MB 774 ms (4.3x), 50 MB 34.6 s in 16 KB chunks (0 ms whole).
            const four = await best(2, () => timeLines(pieces(line(4), CHUNK)));
            const eight = await best(2, () => timeLines(pieces(line(8), CHUNK)));
            console.log(
                `LineReader one line, 16 KB chunks: 4 MB ${four.toFixed(0)} ms, 8 MB ${eight.toFixed(0)} ms (${(eight / four).toFixed(1)}x)`,
            );
            // a 20 ms noise floor: timings of a few ms are dominated by GC and JIT noise
            expect(eight / Math.max(four, 20)).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "LineReader: one 8 MB line in 16 KB chunks costs about the same as in one chunk (PINS a defect)",
        async () => {
            const chunked = await best(2, () => timeLines(pieces(line(8), CHUNK)));
            const whole = await best(2, () => timeLines(onePiece(line(8))));
            console.log(
                `LineReader 8 MB line: 16 KB chunks ${chunked.toFixed(0)} ms, one chunk ${whole.toFixed(0)} ms`,
            );
            expect(chunked).toBeLessThan(Math.max(whole, 20) * SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "CsvRecordReader: doubling a quoted cell that spans 16 KB chunks at most triples the time (PINS a defect)",
        async () => {
            // FAILS: observed 4 MB 194 ms, 8 MB 803 ms (4.1x), 50 MB 37.7 s in 16 KB chunks (1 ms whole).
            const four = await best(2, () => timeCsv(pieces(csvCell(4), CHUNK)));
            const eight = await best(2, () => timeCsv(pieces(csvCell(8), CHUNK)));
            console.log(
                `CsvRecordReader one quoted cell, 16 KB chunks: 4 MB ${four.toFixed(0)} ms, 8 MB ${eight.toFixed(0)} ms (${(eight / four).toFixed(1)}x)`,
            );
            // a 20 ms noise floor: timings of a few ms are dominated by GC and JIT noise
            expect(eight / Math.max(four, 20)).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "CsvRecordReader: an 8 MB quoted cell in 16 KB chunks costs about the same as in one chunk (PINS a defect)",
        async () => {
            const chunked = await best(2, () => timeCsv(pieces(csvCell(8), CHUNK)));
            const whole = await best(2, () => timeCsv(onePiece(csvCell(8))));
            console.log(
                `CsvRecordReader 8 MB cell: 16 KB chunks ${chunked.toFixed(0)} ms, one chunk ${whole.toFixed(0)} ms`,
            );
            expect(chunked).toBeLessThan(Math.max(whole, 20) * SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "XmlTokenizer: doubling an attribute value that spans 16 KB chunks at most triples the time (PINS a defect)",
        () => {
            // FAILS: observed 4 MB 172 ms, 8 MB 750 ms (4.4x), 50 MB 33.8 s in 16 KB chunks (4 ms whole);
            // a 50 MB comment takes 31 s the same way and a 50 MB element name did not finish in 9 min.
            let four = timeXml(xmlAttribute(4), CHUNK);
            four = Math.min(four, timeXml(xmlAttribute(4), CHUNK));
            let eight = timeXml(xmlAttribute(8), CHUNK);
            eight = Math.min(eight, timeXml(xmlAttribute(8), CHUNK));
            console.log(
                `XmlTokenizer one attribute, 16 KB chunks: 4 MB ${four.toFixed(0)} ms, 8 MB ${eight.toFixed(0)} ms (${(eight / four).toFixed(1)}x)`,
            );
            // a 20 ms noise floor: timings of a few ms are dominated by GC and JIT noise
            expect(eight / Math.max(four, 20)).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "XmlTokenizer: an 8 MB attribute value in 16 KB chunks costs about the same as in one push (PINS a defect)",
        () => {
            const text = xmlAttribute(8);
            const chunked = Math.min(timeXml(text, CHUNK), timeXml(text, CHUNK));
            const whole = Math.min(timeXml(text, text.length), timeXml(text, text.length));
            console.log(
                `XmlTokenizer 8 MB attribute: 16 KB pushes ${chunked.toFixed(0)} ms, one push ${whole.toFixed(0)} ms`,
            );
            expect(chunked).toBeLessThan(Math.max(whole, 20) * SHAPE_BOUND);
        },
        LONG,
    );

    it(
        "Neo4jRecordReader (control): a quoted cell spanning 16 KB chunks is linear and shape-independent",
        async () => {
            const cell = (mb: number): string => `:ID,name,:LABEL\n1,"${"x".repeat(mb * MB)}",P\n2,b,P\n`;
            const eight = await best(3, () => timeNeo4j(pieces(cell(8), CHUNK)));
            const sixteen = await best(3, () => timeNeo4j(pieces(cell(16), CHUNK)));
            const whole = await best(3, () => timeNeo4j(onePiece(cell(16))));
            console.log(
                `Neo4jRecordReader 16 MB cell: 16 KB chunks ${sixteen.toFixed(0)} ms, one chunk ${whole.toFixed(0)} ms; 8 MB ${eight.toFixed(0)} ms`,
            );
            expect(sixteen / Math.max(eight, 20)).toBeLessThan(DOUBLING_BOUND);
            expect(sixteen).toBeLessThan(Math.max(whole, 20) * SHAPE_BOUND);
        },
        LONG,
    );
});

describe("fuzz audit: the number of declared columns", () => {
    function timeColumns(count: number): number {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addNode("a");
        const t0 = performance.now();
        for (let i = 0; i < count; i++) {
            builder.declareNodeColumn({ name: `c${i}`, dtype: "f64" });
        }
        for (let i = 0; i < count; i++) {
            builder.nodeColumn(`c${i}`);
        }
        return performance.now() - t0;
    }

    it(
        "GraphBuilder: declaring and resolving 40k columns costs at most three times 20k (PINS a core defect)",
        () => {
            // FAILS: observed 5k 196 ms, 10k 813 ms, 20k 3302 ms (4x per doubling): handleOf() and
            // checkDeclaration() scan the staging column array by name (graph-format
            // src/builder/graph-builder.ts). Reached from every importer through declareOn() /
            // setNodeValue(name) / nodeColumn(name). With the name index the 20k / 40k pair runs
            // in tens of ms (5k is under 10 ms, too close to GC and JIT noise for a ratio).
            timeColumns(2000); // warm the JIT so the two timings compare like with like
            const twenty = Math.min(timeColumns(20_000), timeColumns(20_000), timeColumns(20_000));
            const forty = Math.min(timeColumns(40_000), timeColumns(40_000), timeColumns(40_000));
            console.log(
                `GraphBuilder columns: 20k ${twenty.toFixed(0)} ms, 40k ${forty.toFixed(0)} ms (${(forty / twenty).toFixed(1)}x)`,
            );
            expect(forty / Math.max(twenty, 20)).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );

    it(
        "CSV: a 10k-column header costs at most three times a 5k-column header (PINS the same defect)",
        async () => {
            // FAILS: observed 100k columns 97 s (600 KB of input); 20k GraphML keys 4 s; 100k JSON keys 64 s.
            const timeHeader = async (columns: number): Promise<number> => {
                const header = ["source", "target", ...Array.from({ length: columns }, (_, i) => `c${i}`)].join(",");
                const text = `${header}\na,b,${new Array<string>(columns).fill("1").join(",")}\n`;
                const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
                const t0 = performance.now();
                await registry.importer("csv").import(text, sink, {});
                return performance.now() - t0;
            };
            const five = await best(2, () => timeHeader(5000));
            const ten = await best(2, () => timeHeader(10_000));
            console.log(
                `CSV header columns: 5k ${five.toFixed(0)} ms, 10k ${ten.toFixed(0)} ms (${(ten / five).toFixed(1)}x)`,
            );
            expect(ten / five).toBeLessThan(DOUBLING_BOUND);
        },
        LONG,
    );
});

describe.skipIf(!BENCH)("fuzz audit: absolute 50 MB / 10 s checks (IO_BENCH=1)", () => {
    const HANG_MS = 10_000;

    async function timeImport(format: string, input: string | ReadableStream<Uint8Array>): Promise<number> {
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const t0 = performance.now();
        try {
            await registry.importer(format).import(input, sink, {});
        } catch (err) {
            if (!(err instanceof ImportError)) {
                throw err;
            }
        }
        return performance.now() - t0;
    }

    it(
        "CSV: a 50 MB quoted cell in 16 KB chunks imports within 10 s (PINS a defect)",
        async () => {
            const ms = await timeImport("csv", chunkedBytes(csvCell(50), CHUNK));
            console.log(`csv 50 MB cell in 16 KB chunks: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "CSV: an unclosed quote followed by 50 MB of rows in 16 KB chunks fails within 10 s (PINS a defect)",
        async () => {
            const ms = await timeImport(
                "csv",
                chunkedBytes(`source,target,label\na,b,"oops\n${"a,b,c\n".repeat((50 * MB) / 6)}`, CHUNK),
            );
            console.log(`csv unclosed quote + 50 MB in 16 KB chunks: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "Pajek: a 50 MB label in 16 KB chunks imports within 10 s (PINS a defect)",
        async () => {
            const ms = await timeImport("pajek", chunkedBytes(`*Vertices 1\n${line(50)}*Edges\n1 1\n`, CHUNK));
            console.log(`pajek 50 MB label in 16 KB chunks: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "GraphML: a 50 MB attribute value in 16 KB chunks imports within 10 s (PINS a defect)",
        async () => {
            const doc = `<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph id="G" edgedefault="directed"><node id="${"x".repeat(50 * MB)}"/></graph></graphml>`;
            const ms = await timeImport("graphml", chunkedBytes(doc, CHUNK));
            console.log(`graphml 50 MB attribute in 16 KB chunks: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "GraphML: a 50 MB comment in 16 KB chunks imports within 10 s (PINS a defect)",
        async () => {
            const doc = `<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><!-- ${"x".repeat(50 * MB)} --><graph id="G" edgedefault="directed"><node id="a"/></graph></graphml>`;
            const ms = await timeImport("graphml", chunkedBytes(doc, CHUNK));
            console.log(`graphml 50 MB comment in 16 KB chunks: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "JSON: a node with 100k keys imports within 10 s (PINS the core column defect)",
        async () => {
            const keys = Array.from({ length: 100_000 }, (_, i) => `"k${i}":${i}`).join(",");
            const ms = await timeImport("json", `{"nodes":[{"id":"a",${keys}}],"links":[]}`);
            console.log(`json 100k keys: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );

    it(
        "CSV: a 100k-column header imports within 10 s (PINS the core column defect)",
        async () => {
            const header = ["source", "target", ...Array.from({ length: 100_000 }, (_, i) => `c${i}`)].join(",");
            const ms = await timeImport("csv", `${header}\na,b,${new Array<string>(100_000).fill("1").join(",")}\n`);
            console.log(`csv 100k columns: ${ms.toFixed(0)} ms`);
            expect(ms).toBeLessThan(HANG_MS);
        },
        LONG,
    );
});
