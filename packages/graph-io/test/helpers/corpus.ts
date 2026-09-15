/**
 * Corpus access for the format tests (design section 16.5): the fixtures under test/corpus/ (moved
 * verbatim from graphty-element/test/helpers/corpus/), their manifests, the malformed cases, and
 * the input shapes that exercise the streaming paths (byte chunks of a chosen size, a
 * ReadableStream, an async iterable of strings).
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * The corpus root: test/corpus.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export const CORPUS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "corpus");

/** The formats the corpus covers, by directory name. */
export const CORPUS_FORMATS = ["csv", "dot", "gexf", "gml", "graphml", "json", "neo4j", "pajek"] as const;

/**
 * A corpus format name.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export type CorpusFormat = (typeof CORPUS_FORMATS)[number];

/**
 * One entry of a manifest.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface CorpusFile {
    /** The file name relative to the format directory. */
    readonly path: string;
    /** Where the file came from. */
    readonly source: string;
    /** Its license. */
    readonly license: string;
    /** Nodes the legacy parsers found. */
    readonly expectedNodes: number;
    /** Edges the legacy parsers found. */
    readonly expectedEdges: number;
    /** Feature tags. */
    readonly features: readonly string[];
}

/**
 * A format's manifest.json.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface CorpusManifest {
    /** The format name. */
    readonly format: string;
    /** A description. */
    readonly description: string;
    /** The files. */
    readonly files: readonly CorpusFile[];
}

/**
 * The manifest of a format.
 * @param format - the format directory
 * @returns the parsed manifest
 */
export function loadManifest(format: CorpusFormat): CorpusManifest {
    return JSON.parse(readFileSync(join(CORPUS_ROOT, format, "manifest.json"), "utf-8")) as CorpusManifest;
}

/**
 * The manifest entries of a format.
 * @param format - the format directory
 * @returns the files, in manifest order
 */
export function corpusFiles(format: CorpusFormat): readonly CorpusFile[] {
    return loadManifest(format).files;
}

/**
 * The manifest entry of one file.
 * @param format - the format directory
 * @param name - the file name
 * @returns the entry; throws when the manifest has no such file
 */
export function corpusEntry(format: CorpusFormat, name: string): CorpusFile {
    const entry = corpusFiles(format).find((f) => f.path === name);
    if (entry === undefined) {
        throw new Error(`no manifest entry for ${format}/${name}`);
    }
    return entry;
}

/**
 * The absolute path of a corpus file.
 * @param format - the format directory
 * @param name - the file name
 * @returns the path
 */
export function corpusPath(format: CorpusFormat, name: string): string {
    return join(CORPUS_ROOT, format, name);
}

/**
 * The absolute path of a malformed corpus file.
 * @param format - the format directory
 * @param name - the file name
 * @returns the path
 */
export function malformedPath(format: CorpusFormat, name: string): string {
    return join(CORPUS_ROOT, "malformed", format, name);
}

/**
 * A corpus file as text.
 * @param format - the format directory
 * @param name - the file name
 * @returns the file content decoded as UTF-8
 */
export function readCorpusText(format: CorpusFormat, name: string): string {
    return readFileSync(corpusPath(format, name), "utf-8");
}

/**
 * A corpus file as bytes.
 * @param format - the format directory
 * @param name - the file name
 * @returns the raw bytes
 */
export function readCorpusBytes(format: CorpusFormat, name: string): Uint8Array {
    return new Uint8Array(readFileSync(corpusPath(format, name)));
}

/**
 * The malformed cases of a format: every file under test/corpus/malformed/<format>.
 * @param format - the format directory
 * @returns the file names, sorted
 */
export function malformedFiles(format: CorpusFormat): readonly string[] {
    const dir = join(CORPUS_ROOT, "malformed", format);
    return existsSync(dir) ? readdirSync(dir).sort() : [];
}

/**
 * A malformed corpus file as bytes (some cases are binary or not UTF-8, so text is not assumed).
 * @param format - the format directory
 * @param name - the file name
 * @returns the raw bytes
 */
export function readMalformedBytes(format: CorpusFormat, name: string): Uint8Array {
    return new Uint8Array(readFileSync(malformedPath(format, name)));
}

/**
 * A malformed corpus file as text (lossy for the binary cases; use readMalformedBytes for those).
 * @param format - the format directory
 * @param name - the file name
 * @returns the file content decoded as UTF-8
 */
export function readMalformedText(format: CorpusFormat, name: string): string {
    return readFileSync(malformedPath(format, name), "utf-8");
}

/**
 * Bytes as an async iterable of chunks of a fixed size, to exercise chunk boundaries (a multi-byte
 * character split across two chunks, a CRLF split, a line spanning chunks).
 * @param bytes - the bytes
 * @param chunkSize - bytes per chunk
 * @yields one chunk at a time
 * @returns nothing
 */
export async function* byteChunks(bytes: Uint8Array, chunkSize: number): AsyncGenerator<Uint8Array, void, undefined> {
    for (let offset = 0; offset < bytes.byteLength; offset += chunkSize) {
        yield bytes.subarray(offset, Math.min(offset + chunkSize, bytes.byteLength));
        await Promise.resolve();
    }
}

/**
 * Text as an async iterable of string chunks of a fixed length.
 * @param text - the text
 * @param chunkLength - UTF-16 code units per chunk
 * @yields one chunk at a time
 * @returns nothing
 */
export async function* textChunksOf(text: string, chunkLength: number): AsyncGenerator<string, void, undefined> {
    for (let offset = 0; offset < text.length; offset += chunkLength) {
        yield text.slice(offset, offset + chunkLength);
        await Promise.resolve();
    }
}

/**
 * Bytes as a ReadableStream of chunks of a fixed size.
 * @param bytes - the bytes
 * @param chunkSize - bytes per chunk
 * @returns the stream
 */
export function byteStream(bytes: Uint8Array, chunkSize: number): ReadableStream<Uint8Array> {
    let offset = 0;
    return new ReadableStream<Uint8Array>({
        pull(controller): void {
            if (offset >= bytes.byteLength) {
                controller.close();
                return;
            }
            controller.enqueue(bytes.subarray(offset, Math.min(offset + chunkSize, bytes.byteLength)));
            offset += chunkSize;
        },
    });
}

/**
 * Every input shape of one file's bytes, for a test that runs an importer over all of them.
 * @param bytes - the bytes
 * @returns named factories; each call produces a fresh input
 */
export function inputShapes(bytes: Uint8Array): readonly {
    readonly name: string;
    readonly make: () => string | Uint8Array | ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>;
}[] {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return [
        { name: "string", make: (): string => text },
        { name: "Uint8Array", make: (): Uint8Array => bytes },
        { name: "byte chunks of 7", make: (): AsyncIterable<Uint8Array> => byteChunks(bytes, 7) },
        { name: "text chunks of 5", make: (): AsyncIterable<string> => textChunksOf(text, 5) },
        { name: "ReadableStream of 64", make: (): ReadableStream<Uint8Array> => byteStream(bytes, 64) },
    ];
}
