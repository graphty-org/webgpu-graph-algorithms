/**
 * The importer / exporter registry (design sections 8.2 and 8.4): the eight built-in formats
 * registered by name, `sniff()` over them, and the two conveniences for callers who do not own a
 * sink: `importGraph()` sniffs the format, creates a builder seeded from the common options
 * (`directed: true` as a placeholder; the importer sets the real value), imports and freezes;
 * `exportGraph()` looks the exporter up by name. A caller who owns a builder uses the importer
 * objects directly (the subpath exports) and this module never touches their sink.
 */

import {
    type FreezeOptions,
    type FreezeReport,
    GraphBuilder,
    type GraphBuilderOptions,
    GraphFormatError,
    type GraphSnapshot,
} from "@graphty/graph-format";

import { throwIfAborted } from "./common/input.js";
import { ImportReportBuilder } from "./common/report.js";
import { csvExporter, csvImporter } from "./formats/csv/index.js";
import { dotExporter, dotImporter } from "./formats/dot/index.js";
import { gexfExporter, gexfImporter } from "./formats/gexf/index.js";
import { gmlExporter, gmlImporter } from "./formats/gml/index.js";
import { graphmlExporter, graphmlImporter } from "./formats/graphml/index.js";
import { jsonExporter, jsonImporter } from "./formats/json/index.js";
import { neo4jExporter, neo4jImporter } from "./formats/neo4j/index.js";
import { pajekExporter, pajekImporter } from "./formats/pajek/index.js";
import { rankFormats, SNIFF_HEAD_BYTES, type SniffHints, type SniffResult } from "./sniff.js";
import {
    type CommonExportOptions,
    type CommonImportOptions,
    type GraphExporter,
    type GraphImporter,
    type ImportInput,
    type ImportReport,
    type LossNote,
} from "./types.js";

/** The issue code of an input whose format no registered importer recognises. */
export const UNKNOWN_FORMAT_CODE = "E_UNKNOWN_FORMAT";

/** The builder options importGraph() accepts beyond the ones the common import options seed. */
export type BuilderSeed = Omit<
    GraphBuilderOptions,
    "directed" | "addMissingNodes" | "duplicateEdges" | "selfLoops" | "weightDtype"
>;

/**
 * The options of importGraph(): the common import options (which also seed the registry's
 * builder, design section 8.4), the format choice and the hints sniffing uses, the builder and
 * freeze options, and any format-specific option (`delimiter`, `dialect`, ...) passed through to
 * the importer unchanged.
 */
export interface ImportGraphOptions extends CommonImportOptions {
    /** The format name, or "auto" (default) to sniff it from the filename, MIME type and content. */
    readonly format?: string | undefined;
    /** The file name or path the input came from, a hint for sniffing. */
    readonly filename?: string | null | undefined;
    /** The MIME type the input was served as, a hint for sniffing. */
    readonly mimeType?: string | null | undefined;
    /** Builder options the common options do not cover (`weighted`, `expectedNodes`, ...). */
    readonly builder?: BuilderSeed | undefined;
    /** Options of the freeze that follows the import. */
    readonly freeze?: FreezeOptions | undefined;
    /** Format-specific options, passed to the importer as they are. */
    readonly [formatOption: string]: unknown;
}

/** What importGraph() returns (design section 8.4). */
export interface ImportGraphResult {
    /** The format the input was read as. */
    readonly format: string;
    /** The sniff that chose the importer, or null when the caller named the format. */
    readonly sniff: SniffResult | null;
    /** The frozen snapshot. */
    readonly snapshot: GraphSnapshot;
    /** The importer's report. */
    readonly report: ImportReport;
    /** The freeze report (design section 6.6). */
    readonly freeze: FreezeReport;
}

/** The options of exportGraph(): the common export options plus any format-specific option, passed through. */
export interface ExportGraphOptions extends CommonExportOptions {
    /** Format-specific options, passed to the exporter as they are. */
    readonly [formatOption: string]: unknown;
}

/** The keys of ImportGraphOptions that belong to the registry, never to an importer. */
const REGISTRY_KEYS: ReadonlySet<string> = new Set(["format", "filename", "mimeType", "builder", "freeze"]);

/**
 * A registry of importers and exporters by format name. Registration order is the tie-break
 * order of sniffing (design section 8.2); the default registry lists the built-in formats in the
 * order of GRAPH_FORMATS.
 */
export class FormatRegistry {
    private readonly importerMap = new Map<string, GraphImporter>();

    private readonly exporterMap = new Map<string, GraphExporter>();

    /**
     * Register an importer under its format name, replacing one of the same name in place (the
     * original registration order is kept).
     * @param importer - the importer
     * @returns this registry, for chaining
     */
    registerImporter(importer: GraphImporter): this {
        this.importerMap.set(importer.format, importer);
        return this;
    }

    /**
     * Register an exporter under its format name, replacing one of the same name.
     * @param exporter - the exporter
     * @returns this registry, for chaining
     */
    registerExporter(exporter: GraphExporter): this {
        this.exporterMap.set(exporter.format, exporter);
        return this;
    }

    /**
     * The importer of a format.
     * @param format - the format name
     * @returns the importer; E_UNSUPPORTED when none is registered
     */
    importer(format: string): GraphImporter {
        const importer = this.importerMap.get(format);
        if (importer === undefined) {
            throw unknownFormat("importer", format, this.importerMap.keys());
        }
        return importer;
    }

    /**
     * The exporter of a format.
     * @param format - the format name
     * @returns the exporter; E_UNSUPPORTED when none is registered
     */
    exporter(format: string): GraphExporter {
        const exporter = this.exporterMap.get(format);
        if (exporter === undefined) {
            throw unknownFormat("exporter", format, this.exporterMap.keys());
        }
        return exporter;
    }

    /**
     * Whether an importer is registered for a format.
     * @param format - the format name
     * @returns true when importer(format) would succeed
     */
    hasImporter(format: string): boolean {
        return this.importerMap.has(format);
    }

    /**
     * Whether an exporter is registered for a format.
     * @param format - the format name
     * @returns true when exporter(format) would succeed
     */
    hasExporter(format: string): boolean {
        return this.exporterMap.has(format);
    }

    /**
     * Every registered importer, in registration order.
     * @returns the importers
     */
    importers(): readonly GraphImporter[] {
        return [...this.importerMap.values()];
    }

    /**
     * Every registered exporter, in registration order.
     * @returns the exporters
     */
    exporters(): readonly GraphExporter[] {
        return [...this.exporterMap.values()];
    }

    /**
     * The names of every format with an importer or an exporter, importers' order first.
     * @returns the format names, each once
     */
    formats(): readonly string[] {
        return [...new Set([...this.importerMap.keys(), ...this.exporterMap.keys()])];
    }

    /**
     * Rank the registered importers for an input (design section 8.2; the successor of
     * graphty-element's detectFormat()).
     * @param hints - the filename, MIME type and / or head of the input
     * @returns the candidates, best first; empty when nothing matches
     */
    sniffAll(hints: SniffHints): readonly SniffResult[] {
        return rankFormats(hints, this.importerMap.values());
    }

    /**
     * The best importer for an input, or null when no registered importer claims it.
     * @param hints - the filename, MIME type and / or head of the input
     * @returns the best candidate, or null
     */
    sniff(hints: SniffHints): SniffResult | null {
        const ranked = this.sniffAll(hints);
        return ranked.length > 0 ? ranked[0] : null;
    }

    /**
     * Read an input into a fresh builder and freeze it (design section 8.4: for callers who do not
     * own a sink). The format is the one named in the options, else sniffed from the filename,
     * the MIME type and the first bytes of the content; the builder is seeded from the common
     * options with `directed: true` as a placeholder that the importer overrides from the file.
     * @param input - the text, bytes, stream or chunks to read
     * @param options - the format, hints, common and format-specific import options
     * @returns the snapshot, the import report and the freeze report
     */
    async importGraph(input: ImportInput, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
        const requested = options.format ?? "auto";
        let importer: GraphImporter;
        let sniff: SniffResult | null = null;
        let source = input;
        let peeked: PeekedInput | null = null;
        if (requested === "auto") {
            peeked = await peekHead(input, SNIFF_HEAD_BYTES, options.signal ?? null);
            source = peeked.input;
            sniff = this.sniff({ filename: options.filename, mimeType: options.mimeType, head: peeked.head });
            if (sniff === null) {
                const report = new ImportReportBuilder("unknown", 0);
                return report.fail(
                    UNKNOWN_FORMAT_CODE,
                    `no registered importer recognises the input${describeHints(options)}; pass the format explicitly`,
                    undefined,
                    { formats: this.formats() },
                );
            }
            importer = this.importer(sniff.format);
        } else {
            importer = this.importer(requested);
        }
        const builder = new GraphBuilder({
            weightDtype: options.weightDtype ?? "f64",
            ...options.builder,
            directed: true,
            addMissingNodes: options.addMissingNodes ?? true,
            duplicateEdges: options.duplicateEdges ?? "keep",
            selfLoops: options.selfLoops ?? "keep",
        });
        let report: ImportReport;
        try {
            report = await importer.import(source, builder, importerOptions(options));
        } catch (err) {
            await peeked?.close();
            throw err;
        }
        const frozen = builder.freezeWithReport(options.freeze);
        return Object.freeze({
            format: importer.format,
            sniff,
            snapshot: frozen.snapshot,
            report,
            freeze: frozen.report,
        });
    }

    /**
     * Write a snapshot in a format, as UTF-8 chunks (design section 8.5).
     * @param snapshot - the snapshot
     * @param format - the format name
     * @param options - the exporter's common and format-specific options
     * @returns the encoded chunks
     */
    exportGraph(snapshot: GraphSnapshot, format: string, options?: ExportGraphOptions): AsyncIterable<Uint8Array> {
        return this.exporter(format).export(snapshot, options);
    }

    /**
     * Write a snapshot in a format, as one string.
     * @param snapshot - the snapshot
     * @param format - the format name
     * @param options - the exporter's common and format-specific options
     * @returns the whole document
     */
    async exportGraphToString(snapshot: GraphSnapshot, format: string, options?: ExportGraphOptions): Promise<string> {
        return this.exporter(format).exportToString(snapshot, options);
    }

    /**
     * What exporting a snapshot in a format would lose, without writing anything.
     * @param snapshot - the snapshot
     * @param format - the format name
     * @param options - the exporter's common and format-specific options
     * @returns the loss notes, empty when the export is exact
     */
    checkExport(snapshot: GraphSnapshot, format: string, options?: ExportGraphOptions): readonly LossNote[] {
        return this.exporter(format).check(snapshot, options);
    }
}

/**
 * A registry holding the eight built-in importers and exporters in the order of GRAPH_FORMATS.
 * @returns a new registry
 */
export function createRegistry(): FormatRegistry {
    return new FormatRegistry()
        .registerImporter(jsonImporter)
        .registerExporter(jsonExporter)
        .registerImporter(graphmlImporter)
        .registerExporter(graphmlExporter)
        .registerImporter(gexfImporter)
        .registerExporter(gexfExporter)
        .registerImporter(csvImporter)
        .registerExporter(csvExporter)
        .registerImporter(gmlImporter)
        .registerExporter(gmlExporter)
        .registerImporter(dotImporter)
        .registerExporter(dotExporter)
        .registerImporter(pajekImporter)
        .registerExporter(pajekExporter)
        .registerImporter(neo4jImporter)
        .registerExporter(neo4jExporter);
}

/** The default registry: every built-in format. */
export const registry: FormatRegistry = createRegistry();

/**
 * Read an input into a fresh builder and freeze it, through the default registry (design section
 * 8.4: `importGraph(input, { format?, ...options })`).
 * @param input - the text, bytes, stream or chunks to read
 * @param options - the format, hints, common and format-specific import options
 * @returns the snapshot, the import report and the freeze report
 */
export function importGraph(input: ImportInput, options?: ImportGraphOptions): Promise<ImportGraphResult> {
    return registry.importGraph(input, options);
}

/**
 * Write a snapshot in a format through the default registry, as UTF-8 chunks.
 * @param snapshot - the snapshot
 * @param format - the format name
 * @param options - the exporter's common and format-specific options
 * @returns the encoded chunks
 */
export function exportGraph(
    snapshot: GraphSnapshot,
    format: string,
    options?: ExportGraphOptions,
): AsyncIterable<Uint8Array> {
    return registry.exportGraph(snapshot, format, options);
}

/**
 * Write a snapshot in a format through the default registry, as one string.
 * @param snapshot - the snapshot
 * @param format - the format name
 * @param options - the exporter's common and format-specific options
 * @returns the whole document
 */
export async function exportGraphToString(
    snapshot: GraphSnapshot,
    format: string,
    options?: ExportGraphOptions,
): Promise<string> {
    return registry.exportGraphToString(snapshot, format, options);
}

/**
 * What exporting a snapshot in a format would lose, through the default registry.
 * @param snapshot - the snapshot
 * @param format - the format name
 * @param options - the exporter's common and format-specific options
 * @returns the loss notes, empty when the export is exact
 */
export function checkExport(
    snapshot: GraphSnapshot,
    format: string,
    options?: ExportGraphOptions,
): readonly LossNote[] {
    return registry.checkExport(snapshot, format, options);
}

/**
 * Sniff an input's format through the default registry.
 * @param hints - the filename, MIME type and / or head of the input
 * @returns the best candidate, or null when no built-in importer claims it
 */
export function sniff(hints: SniffHints): SniffResult | null {
    return registry.sniff(hints);
}

/**
 * The options handed to the importer: everything but the registry's own keys.
 * @param options - the importGraph options
 * @returns the importer's options
 */
function importerOptions(options: ImportGraphOptions): CommonImportOptions {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(options)) {
        if (!REGISTRY_KEYS.has(key)) {
            out[key] = value;
        }
    }
    return out;
}

/**
 * The E_UNSUPPORTED error of a format name nothing is registered under.
 * @param kind - importer or exporter
 * @param format - the requested name
 * @param known - the registered names
 * @returns the error
 */
function unknownFormat(kind: "importer" | "exporter", format: string, known: Iterable<string>): GraphFormatError {
    const supported = [...known];
    return new GraphFormatError(
        "E_UNSUPPORTED",
        `no ${kind} is registered for format ${JSON.stringify(format)}; known: ${supported.join(", ")}`,
        { option: "format", found: format, supported },
    );
}

/**
 * The hints of an import, for the unknown-format message.
 * @param options - the importGraph options
 * @returns " (filename ..., MIME type ...)" or an empty string
 */
function describeHints(options: ImportGraphOptions): string {
    const parts: string[] = [];
    if (typeof options.filename === "string" && options.filename.length > 0) {
        parts.push(`filename ${JSON.stringify(options.filename)}`);
    }
    if (typeof options.mimeType === "string" && options.mimeType.length > 0) {
        parts.push(`MIME type ${JSON.stringify(options.mimeType)}`);
    }
    return parts.length === 0 ? "" : ` (${parts.join(", ")})`;
}

/** A head read from an input, and the input to hand the importer (the same bytes, replayed for a stream). */
interface PeekedInput {
    /** The first bytes (or, for a text input, characters) of the content. */
    readonly head: Uint8Array | string;
    /** The input to read: the original for in-memory input, a replaying iterable for a stream. */
    readonly input: ImportInput;
    /**
     * Close the source when the importer never iterated the replaying input (it threw first, an
     * abort for instance): a stream's reader is cancelled and released, an async generator finalised.
     * @returns when the source is closed
     */
    close(): Promise<void>;
}

/**
 * The first `bytes` of an input without consuming it: in-memory input is sliced; a stream or an
 * async iterable is read until enough is buffered and then replayed (the buffered chunks first,
 * the rest as it arrives) through a new async iterable that cancels the source when the importer
 * stops early.
 * @param input - the input
 * @param bytes - how much to peek
 * @param signal - the cancellation signal, or null
 * @returns the head and the input to read
 */
async function peekHead(input: ImportInput, bytes: number, signal: AbortSignal | null): Promise<PeekedInput> {
    if (typeof input === "string") {
        return { head: input.slice(0, bytes), input, close: (): Promise<void> => Promise.resolve() };
    }
    if (input instanceof Uint8Array) {
        return { head: input.subarray(0, bytes), input, close: (): Promise<void> => Promise.resolve() };
    }
    throwIfAborted(signal);
    const source: AsyncIterator<string | Uint8Array> =
        typeof (input as { getReader?: unknown }).getReader === "function"
            ? readerIterator(input as ReadableStream<Uint8Array>)
            : (input as AsyncIterable<string | Uint8Array>)[Symbol.asyncIterator]();
    const buffered: (string | Uint8Array)[] = [];
    let size = 0;
    let finished = false;
    while (size < bytes) {
        if (signal !== null && signal.aborted) {
            await source.return?.();
            throwIfAborted(signal);
        }
        const next = await source.next();
        if (next.done === true) {
            finished = true;
            break;
        }
        buffered.push(next.value);
        size += typeof next.value === "string" ? next.value.length : next.value.byteLength;
    }
    if (!finished && signal !== null && signal.aborted) {
        // an abort that landed on the read completing the head: close the source before rethrowing
        await source.return?.();
        throwIfAborted(signal);
    }
    const replayed = replay(buffered, source, finished);
    return { head: joinHead(buffered, bytes), input: replayed, close: (): Promise<void> => replayed.close() };
}

/**
 * The head text or bytes of the buffered chunks: bytes when every chunk is bytes, text otherwise
 * (byte chunks decoded leniently; the importer decodes the real input strictly).
 * @param chunks - the buffered chunks
 * @param bytes - the head size
 * @returns the head
 */
function joinHead(chunks: readonly (string | Uint8Array)[], bytes: number): Uint8Array | string {
    if (chunks.every((c) => c instanceof Uint8Array)) {
        const total = Math.min(
            bytes,
            chunks.reduce((sum, c) => sum + c.byteLength, 0),
        );
        const head = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            if (offset >= total) {
                break;
            }
            const part = chunk.subarray(0, Math.min(chunk.byteLength, total - offset));
            head.set(part, offset);
            offset += part.byteLength;
        }
        return head;
    }
    const decoder = new TextDecoder("utf-8", { fatal: false });
    let text = "";
    for (const chunk of chunks) {
        text += typeof chunk === "string" ? chunk : decoder.decode(chunk, { stream: true });
        if (text.length >= bytes) {
            break;
        }
    }
    return text.slice(0, bytes);
}

/**
 * An async iterable that yields the buffered chunks, then the rest of the source; the source is
 * closed when the consumer stops early.
 * @param buffered - the chunks already read
 * @param source - the iterator positioned after them
 * @param finished - whether the source is already exhausted
 * @returns the replaying iterable
 */
function replay(
    buffered: readonly (string | Uint8Array)[],
    source: AsyncIterator<string | Uint8Array>,
    finished: boolean,
): AsyncIterable<string | Uint8Array> & { close(): Promise<void> } {
    let exhausted = finished;
    const close = async (): Promise<void> => {
        if (!exhausted) {
            exhausted = true;
            await source.return?.();
        }
    };
    return {
        close,
        [Symbol.asyncIterator](): AsyncIterator<string | Uint8Array> {
            let position = 0;
            return {
                async next(): Promise<IteratorResult<string | Uint8Array>> {
                    if (position < buffered.length) {
                        return { done: false, value: buffered[position++] };
                    }
                    if (exhausted) {
                        return { done: true, value: undefined };
                    }
                    const next = await source.next();
                    if (next.done === true) {
                        exhausted = true;
                        return { done: true, value: undefined };
                    }
                    return next;
                },
                async return(): Promise<IteratorResult<string | Uint8Array>> {
                    await close();
                    return { done: true, value: undefined };
                },
            };
        },
    };
}

/**
 * An async iterator over a ReadableStream's chunks that cancels the stream when closed early.
 * @param stream - the stream
 * @returns the iterator
 */
function readerIterator(stream: ReadableStream<Uint8Array>): AsyncIterator<Uint8Array> {
    const reader = stream.getReader();
    let done = false;
    return {
        async next(): Promise<IteratorResult<Uint8Array>> {
            if (done) {
                return { done: true, value: undefined };
            }
            const result = await reader.read();
            if (result.done) {
                done = true;
                reader.releaseLock();
                return { done: true, value: undefined };
            }
            return { done: false, value: result.value };
        },
        async return(): Promise<IteratorResult<Uint8Array>> {
            if (!done) {
                done = true;
                await reader.cancel().catch(() => undefined);
                reader.releaseLock();
            }
            return { done: true, value: undefined };
        },
    };
}
