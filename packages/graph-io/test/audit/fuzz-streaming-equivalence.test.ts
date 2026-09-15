/**
 * Fuzz audit, streaming-equivalence lens (design section 8.4: "chunk boundaries carry no
 * meaning"): for every corpus file under 200 KB, the snapshot produced from a ReadableStream
 * split at an arbitrary byte offset must equal the snapshot produced from the whole string:
 * same contentHash (topology, weights, orientation), same ids, same columns value by value, same
 * issue list. Fifty offsets are sampled per file (every offset for files under 2 KB), the
 * smallest file of every format is also fed as 1-byte chunks, and a few extra shapes are tried:
 * three-way splits, mixed string and byte chunks in one async iterable, a BOM and a multi-byte
 * character split across the boundary, and an empty chunk between two halves.
 */

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { registry } from "../../src/registry.js";
import { type CommonImportOptions, ImportError, type ImportInput, type ImportReport } from "../../src/types.js";
import { CORPUS_FORMATS, type CorpusFile, corpusFiles, readCorpusBytes } from "../helpers/corpus.js";
import { compareSnapshots, describeDiffs } from "../helpers/roundtrip.js";

const MAX_BYTES = 200 * 1024;
const SAMPLED_OFFSETS = 50;
const EXHAUSTIVE_BELOW = 2 * 1024;

/** The manifest entries carry format options for some files (the neo4j TSV needs its delimiter). */
type EntryWithOptions = CorpusFile & { readonly options?: CommonImportOptions | undefined };

interface Loaded {
    readonly snapshot: GraphSnapshot | null;
    readonly report: ImportReport;
}

async function load(format: string, input: ImportInput, options: CommonImportOptions): Promise<Loaded> {
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        const report = await registry.importer(format).import(input, sink, options);
        return { snapshot: sink.freeze(), report };
    } catch (err) {
        if (err instanceof ImportError) {
            return { snapshot: null, report: err.report };
        }
        throw err;
    }
}

function splitStream(bytes: Uint8Array, cuts: readonly number[]): ReadableStream<Uint8Array> {
    const bounds = [0, ...cuts, bytes.byteLength];
    let i = 0;
    return new ReadableStream<Uint8Array>({
        pull(controller): void {
            if (i >= bounds.length - 1) {
                controller.close();
                return;
            }
            controller.enqueue(bytes.subarray(bounds[i], bounds[i + 1]));
            i++;
        },
    });
}

async function* mixedChunks(bytes: Uint8Array, cut: number): AsyncGenerator<string | Uint8Array, void, undefined> {
    // the first part as bytes, an empty byte chunk, then the rest as text (only when the cut is
    // on a character boundary; the caller picks such a cut)
    yield bytes.subarray(0, cut);
    yield new Uint8Array(0);
    yield new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(cut));
    await Promise.resolve();
}

function issueKeys(report: ImportReport): string[] {
    return report.issues.map((i) => `${i.severity}:${i.code}:${i.line ?? "-"}:${i.element ?? "-"}`);
}

function expectEquivalent(reference: Loaded, actual: Loaded, label: string): void {
    expect(issueKeys(actual.report), `${label}: issues`).toEqual(issueKeys(reference.report));
    expect(actual.report.counts, `${label}: counts`).toEqual(reference.report.counts);
    if (reference.snapshot === null) {
        expect(actual.snapshot, `${label}: reference failed but the stream import produced a snapshot`).toBeNull();
        return;
    }
    expect(actual.snapshot, `${label}: reference imported but the stream import failed`).not.toBeNull();
    if (actual.snapshot === null) {
        return;
    }
    expect(actual.snapshot.contentHash(), `${label}: contentHash`).toBe(reference.snapshot.contentHash());
    const diffs = compareSnapshots(reference.snapshot, actual.snapshot, { allowExtraColumns: false, originType: true });
    expect(diffs.length === 0 ? "" : `${label}: ${describeDiffs(diffs)}`).toBe("");
}

/** Deterministic sample of split offsets in 1..length-1. */
function sampledOffsets(length: number, count: number): number[] {
    if (length <= 1) {
        return [];
    }
    if (length - 1 <= count) {
        return Array.from({ length: length - 1 }, (_, i) => i + 1);
    }
    const out = new Set<number>();
    let state = 0x9e3779b9 ^ length;
    while (out.size < count) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        out.add(1 + (state % (length - 1)));
    }
    return [...out].sort((a, b) => a - b);
}

/** A byte offset that starts a UTF-8 character (not a continuation byte). */
function characterBoundary(bytes: Uint8Array, near: number): number {
    let i = Math.min(Math.max(near, 1), bytes.byteLength - 1);
    while (i > 0 && (bytes[i] & 0xc0) === 0x80) {
        i--;
    }
    return i;
}

describe("fuzz audit: a ReadableStream split anywhere imports the same snapshot as the whole string", () => {
    for (const format of CORPUS_FORMATS) {
        const entries = (corpusFiles(format) as readonly EntryWithOptions[]).filter(
            (e) => readCorpusBytes(format, e.path).byteLength <= MAX_BYTES,
        );
        const smallest = [...entries].sort(
            (a, b) => readCorpusBytes(format, a.path).byteLength - readCorpusBytes(format, b.path).byteLength,
        )[0];

        for (const entry of entries) {
            const bytes = readCorpusBytes(format, entry.path);
            const options = entry.options ?? {};
            const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
            const offsets =
                bytes.byteLength < EXHAUSTIVE_BELOW
                    ? sampledOffsets(bytes.byteLength, bytes.byteLength)
                    : sampledOffsets(bytes.byteLength, SAMPLED_OFFSETS);

            it(`${format}/${entry.path}: ${offsets.length} two-chunk splits`, async () => {
                const reference = await load(format, text, options);
                expect(reference.snapshot, "the reference import must succeed").not.toBeNull();
                for (const offset of offsets) {
                    const actual = await load(format, splitStream(bytes, [offset]), options);
                    expectEquivalent(reference, actual, `${format}/${entry.path} split at ${offset}`);
                }
            });

            it(`${format}/${entry.path}: three-way splits, an empty middle chunk and mixed text/byte chunks`, async () => {
                const reference = await load(format, text, options);
                const n = bytes.byteLength;
                const cuts = sampledOffsets(n, 6);
                for (let i = 0; i + 1 < cuts.length; i += 2) {
                    const actual = await load(format, splitStream(bytes, [cuts[i], cuts[i + 1]]), options);
                    expectEquivalent(reference, actual, `${format}/${entry.path} split at ${cuts[i]},${cuts[i + 1]}`);
                }
                const mid = characterBoundary(bytes, Math.floor(n / 2));
                expectEquivalent(
                    reference,
                    await load(format, splitStream(bytes, [mid, mid]), options),
                    `${format}/${entry.path} empty chunk at ${mid}`,
                );
                expectEquivalent(
                    reference,
                    await load(format, mixedChunks(bytes, mid), options),
                    `${format}/${entry.path} bytes then text at ${mid}`,
                );
            });

            it(`${format}/${entry.path}: a UTF-8 BOM split across the first chunk boundary`, async () => {
                const reference = await load(format, text, options);
                const withBom = new Uint8Array(bytes.byteLength + 3);
                withBom.set([0xef, 0xbb, 0xbf]);
                withBom.set(bytes, 3);
                for (const cut of [1, 2, 3]) {
                    const actual = await load(format, splitStream(withBom, [cut]), options);
                    expectEquivalent(reference, actual, `${format}/${entry.path} BOM cut at ${cut}`);
                }
            });
        }

        if (smallest !== undefined) {
            const bytes = readCorpusBytes(format, smallest.path);
            const options = smallest.options ?? {};

            it(`${format}/${smallest.path}: 1-byte chunks`, async () => {
                const reference = await load(format, new TextDecoder("utf-8", { fatal: true }).decode(bytes), options);
                const cuts = Array.from({ length: bytes.byteLength - 1 }, (_, i) => i + 1);
                const actual = await load(format, splitStream(bytes, cuts), options);
                expectEquivalent(reference, actual, `${format}/${smallest.path} 1-byte chunks`);
            });

            it(`${format}/${smallest.path}: every multi-byte character split in the middle`, async () => {
                // a file with a non-ASCII character is cut inside it; ASCII-only files are cut at
                // every byte anyway by the 1-byte run, so this only adds the odd-sized splits
                const reference = await load(format, new TextDecoder("utf-8", { fatal: true }).decode(bytes), options);
                const cuts: number[] = [];
                for (let i = 1; i < bytes.byteLength; i++) {
                    if ((bytes[i] & 0xc0) === 0x80) {
                        cuts.push(i);
                    }
                }
                if (cuts.length === 0) {
                    return;
                }
                const actual = await load(format, splitStream(bytes, cuts), options);
                expectEquivalent(reference, actual, `${format}/${smallest.path} inside multi-byte characters`);
            });
        }
    }

    it("a document with multi-byte ids and a CRLF line ending split at every byte (CSV, Pajek, GraphML, JSON)", async () => {
        const e = String.fromCharCode(0xe9);
        const han = String.fromCharCode(0x4e2d);
        const emoji = String.fromCodePoint(0x1f600);
        const documents: Readonly<Record<string, string>> = {
            csv: `source,target,weight\r\ncaf${e},${han}${han},1.5\r\n${emoji},caf${e},2\r\n`,
            pajek: `*Vertices 3\r\n1 "caf${e}"\r\n2 "${han}${han}"\r\n3 "${emoji}"\r\n*Edges\r\n1 2 1.5\r\n3 1 2\r\n`,
            graphml: `<?xml version="1.0"?>\r\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="l" for="node" attr.name="label" attr.type="string"/><graph id="G" edgedefault="undirected">\r\n<node id="caf${e}"><data key="l">${han}</data></node><node id="${han}${han}"/><node id="${emoji}"/><edge source="caf${e}" target="${han}${han}"/><edge source="${emoji}" target="caf${e}"/></graph></graphml>\r\n`,
            json: `{"nodes":[{"id":"caf${e}","l":"${han}"},{"id":"${han}${han}"},{"id":"${emoji}"}],\r\n"links":[{"source":"caf${e}","target":"${han}${han}","weight":1.5},{"source":"${emoji}","target":"caf${e}"}]}`,
        };
        for (const [format, doc] of Object.entries(documents)) {
            const bytes = new TextEncoder().encode(doc);
            const reference = await load(format, doc, {});
            expect(reference.snapshot).not.toBeNull();
            for (let offset = 1; offset < bytes.byteLength; offset++) {
                expectEquivalent(
                    reference,
                    await load(format, splitStream(bytes, [offset]), {}),
                    `${format} split at ${offset}`,
                );
            }
            const all = Array.from({ length: bytes.byteLength - 1 }, (_, i) => i + 1);
            expectEquivalent(reference, await load(format, splitStream(bytes, all), {}), `${format} 1-byte chunks`);
        }
    });
});
