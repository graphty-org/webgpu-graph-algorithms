/**
 * Input handling shared by every importer (design section 8.4): the ImportInput union (whole
 * text, whole bytes, a byte stream, an async iterable of text or byte chunks) read as a sequence of
 * UTF-8 text chunks, as lines, or as one string, with BOM handling, cancellation through an
 * AbortSignal and byte progress.
 *
 * Bytes are decoded with `new TextDecoder("utf-8", { fatal: true })` in streaming mode, so a
 * multi-byte character split across two chunks decodes correctly and an invalid sequence is a
 * parse-error (issue code E_INVALID_UTF8) that aborts the import, never a silent U+FFFD that could
 * alias two ids. A leading U+FEFF is stripped from text and from decoded bytes alike.
 */

import { GraphFormatError } from "@graphty/graph-format";

import { type ImportInput } from "../types.js";
import { INVALID_UTF8_CODE } from "./codes.js";
import { type ImportReportBuilder } from "./report.js";

export { INVALID_UTF8_CODE };

/**
 * Cancellation and progress hooks of the reader; the resolved importer options satisfy this shape.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface ReadOptions {
    /** The cancellation signal, or null / undefined for none. */
    readonly signal?: AbortSignal | null | undefined;
    /** The progress callback, or null / undefined for none. */
    readonly onProgress?: ((bytesDone: number, bytesTotal?: number) => void) | null | undefined;
}

/**
 * Bytes decoded per call when the input is one in-memory Uint8Array: every consumer of textChunks()
 * then sees bounded chunks whatever the input shape (a line reader or a record reader that holds
 * one parse result per chunk never holds more than this much text at once), and progress stays
 * granular.
 */
const DECODE_SLICE = 256 * 1024;

const BOM = String.fromCharCode(0xfeff);

/**
 * Whether a value is an ImportInput this module can read.
 * @param input - any value
 * @returns true for a string, a Uint8Array, a ReadableStream or an async iterable
 */
export function isImportInput(input: unknown): input is ImportInput {
    if (typeof input === "string" || input instanceof Uint8Array) {
        return true;
    }
    if (typeof input !== "object" || input === null) {
        return false;
    }
    return typeof (input as { getReader?: unknown }).getReader === "function" || Symbol.asyncIterator in input;
}

/**
 * The total size of an in-memory input (bytes for a Uint8Array, UTF-16 code units for a string),
 * for the `bytesTotal` argument of onProgress; null for a stream or an iterable.
 * @param input - the input
 * @returns the size, or null when unknown up front
 */
export function inputLength(input: ImportInput): number | null {
    if (typeof input === "string") {
        return input.length;
    }
    if (input instanceof Uint8Array) {
        return input.byteLength;
    }
    return null;
}

/**
 * Throw the signal's reason when it is aborted, exactly as the platform's
 * `AbortSignal.throwIfAborted()` does: the reason as it is (the DOMException named "AbortError"
 * of a reason-less abort, or whatever the caller passed to `abort(reason)`, an Error or not), so a
 * caller can compare the rejection with `signal.reason`. Only a runtime that stores no reason at
 * all gets a synthesised AbortError.
 * @param signal - the signal, or null
 */
export function throwIfAborted(signal: AbortSignal | null | undefined): void {
    if (signal === null || signal === undefined || !signal.aborted) {
        return;
    }
    const { reason }: { reason: unknown } = signal;
    if (reason === undefined) {
        throw abortError();
    }
    // the platform contract: the reason itself, whatever its type
    throw reason as Error;
}

/**
 * An abort error for a signal that carries no reason.
 * @returns a DOMException named "AbortError" where DOMException exists, else an Error with that name
 */
function abortError(): Error {
    const message = "The import was aborted";
    if (typeof DOMException === "function") {
        return new DOMException(message, "AbortError");
    }
    const err = new Error(message);
    err.name = "AbortError";
    return err;
}

/**
 * Read an ImportInput as a sequence of decoded text chunks. Chunk boundaries carry no meaning:
 * a caller that needs lines uses LineReader, one that needs the whole document uses readText().
 * The signal is checked before every chunk; a stream is cancelled when the consumer stops early
 * or the signal fires. Progress is reported after every chunk.
 * @param input - the input
 * @param report - the report the decode error is recorded in (E_INVALID_UTF8, then ImportError)
 * @param options - cancellation and progress
 * @yields decoded text; the first chunk has any leading BOM removed
 * @returns nothing
 */
export async function* textChunks(
    input: ImportInput,
    report: ImportReportBuilder,
    options: ReadOptions = {},
): AsyncGenerator<string, void, undefined> {
    const signal = options.signal ?? null;
    const onProgress = options.onProgress ?? null;
    const total = inputLength(input);
    let done = 0;
    let first = true;
    const emit = (text: string): string => {
        if (first && text.length > 0) {
            first = false;
            return text.startsWith(BOM) ? text.slice(1) : text;
        }
        return text;
    };
    throwIfAborted(signal);
    if (typeof input === "string") {
        const text = emit(input);
        done = input.length;
        if (text.length > 0) {
            yield text;
        }
        onProgress?.(done, total ?? undefined);
        return;
    }
    const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
    const decode = (bytes: Uint8Array, stream: boolean): string => {
        try {
            return decoder.decode(bytes, { stream });
        } catch (err) {
            if (err instanceof TypeError) {
                report.fail(INVALID_UTF8_CODE, `invalid UTF-8 near byte ${done}`, undefined, { byteOffset: done });
            }
            throw err;
        }
    };
    if (input instanceof Uint8Array) {
        for (let offset = 0; offset < input.byteLength; offset += DECODE_SLICE) {
            throwIfAborted(signal);
            const slice = input.subarray(offset, Math.min(offset + DECODE_SLICE, input.byteLength));
            const text = emit(decode(slice, offset + DECODE_SLICE < input.byteLength));
            done = Math.min(offset + DECODE_SLICE, input.byteLength);
            if (text.length > 0) {
                yield text;
            }
            onProgress?.(done, total ?? undefined);
        }
        if (input.byteLength === 0) {
            onProgress?.(0, 0);
        }
        return;
    }
    const chunks = isReadableStream(input) ? streamChunks(input, signal) : input;
    for await (const chunk of chunks) {
        throwIfAborted(signal);
        let text: string;
        if (typeof chunk === "string") {
            // finish any byte sequence still pending in the decoder before switching to text
            const pending = decode(new Uint8Array(0), false);
            text = emit(pending + chunk);
            done += chunk.length;
        } else if (chunk instanceof Uint8Array) {
            text = emit(decode(chunk, true));
            done += chunk.byteLength;
        } else {
            throw new GraphFormatError("E_UNSUPPORTED", "an input chunk must be a string or a Uint8Array", {
                reason: "chunk type",
                found: typeof chunk,
            });
        }
        if (text.length > 0) {
            yield text;
        }
        onProgress?.(done);
    }
    const tail = emit(decode(new Uint8Array(0), false));
    if (tail.length > 0) {
        yield tail;
    }
    onProgress?.(done, done);
}

/**
 * Whether an input is a ReadableStream (by duck type, so a stream from another realm qualifies).
 * @param input - a non-string, non-Uint8Array input
 * @returns true for a ReadableStream
 */
function isReadableStream(
    input: ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>,
): input is ReadableStream<Uint8Array> {
    return typeof (input as { getReader?: unknown }).getReader === "function";
}

/**
 * Iterate a ReadableStream through a reader, cancelling the stream when iteration stops early.
 * @param stream - the stream
 * @param signal - the cancellation signal, or null
 * @yields the stream's chunks
 * @returns nothing
 */
async function* streamChunks(
    stream: ReadableStream<Uint8Array>,
    signal: AbortSignal | null,
): AsyncGenerator<Uint8Array, void, undefined> {
    const reader = stream.getReader();
    let finished = false;
    try {
        for (;;) {
            throwIfAborted(signal);
            const { done, value } = await reader.read();
            if (done) {
                finished = true;
                return;
            }
            yield value;
        }
    } finally {
        if (!finished) {
            await reader.cancel(signal?.reason).catch(() => undefined);
        }
        reader.releaseLock();
    }
}

/**
 * Read the whole input as one string (the GML / DOT / JSON path, design section 8.4).
 * @param input - the input
 * @param report - the report the decode error is recorded in
 * @param options - cancellation and progress
 * @returns the decoded text without a leading BOM
 */
export async function readText(
    input: ImportInput,
    report: ImportReportBuilder,
    options: ReadOptions = {},
): Promise<string> {
    if (typeof input === "string") {
        throwIfAborted(options.signal);
        options.onProgress?.(input.length, input.length);
        return input.startsWith(BOM) ? input.slice(1) : input;
    }
    const parts: string[] = [];
    for await (const chunk of textChunks(input, report, options)) {
        parts.push(chunk);
    }
    return parts.length === 1 ? parts[0] : parts.join("");
}

/**
 * Lines of an ImportInput without allocating anything per line but the string itself: iterate with
 * `for await (const text of reader)` and read `reader.line` (1-based) for the line just yielded.
 * `\n`, `\r\n` and lone `\r` all end a line; the terminator is not part of the text; a final
 * line without a terminator is yielded when non-empty, and every line in between is yielded even
 * when empty (the importer decides what a blank line means).
 */
export class LineReader implements AsyncIterable<string> {
    private readonly input: ImportInput;

    private readonly report: ImportReportBuilder;

    private readonly options: ReadOptions;

    private lineNumber = 0;

    /**
     * Create a reader over an input; nothing is read until iteration starts.
     * @param input - the input
     * @param report - the report decode errors are recorded in
     * @param options - cancellation and progress
     */
    constructor(input: ImportInput, report: ImportReportBuilder, options: ReadOptions = {}) {
        this.input = input;
        this.report = report;
        this.options = options;
    }

    /**
     * The 1-based number of the line most recently yielded.
     * @returns the line number; 0 before the first line
     */
    get line(): number {
        return this.lineNumber;
    }

    /**
     * Iterate the lines.
     * @yields one line at a time, terminator removed
     * @returns nothing
     */
    async *[Symbol.asyncIterator](): AsyncGenerator<string, void, undefined> {
        // The pieces of the line in progress (none of them holds a line break, except that the last
        // may end with a `\r` whose meaning the next chunk decides); joined once when the line ends,
        // so a line spanning many chunks costs its length, not its length times the chunk count.
        const pending: string[] = [];
        let trailingCr = false;
        for await (const chunk of textChunks(this.input, this.report, this.options)) {
            let start = 0;
            const end = chunk.length;
            if (trailingCr) {
                // the previous chunk ended with `\r`: that ended a line, and a leading `\n` here is
                // the second half of the same terminator
                trailingCr = false;
                this.lineNumber++;
                const joined = pending.join("");
                pending.length = 0;
                yield joined.slice(0, -1);
                if (chunk.charCodeAt(0) === 10) {
                    start = 1;
                }
            }
            // the next `\n` and `\r` at or after `start`; each is searched for once per chunk and
            // again only after it was consumed, so a chunk is scanned once whatever its line count
            let nl = chunk.indexOf("\n", start);
            let cr = chunk.indexOf("\r", start);
            while (nl >= 0 || cr >= 0) {
                let cut: number;
                let next: number;
                if (cr >= 0 && (nl < 0 || cr < nl)) {
                    if (cr === end - 1) {
                        // a trailing \r may be the first half of \r\n split across chunks
                        break;
                    }
                    cut = cr;
                    next = chunk.charCodeAt(cr + 1) === 10 ? cr + 2 : cr + 1;
                } else {
                    cut = nl;
                    next = nl + 1;
                }
                this.lineNumber++;
                const piece = chunk.slice(start, cut);
                if (pending.length === 0) {
                    yield piece;
                } else {
                    pending.push(piece);
                    const joined = pending.join("");
                    pending.length = 0;
                    yield joined;
                }
                start = next;
                if (nl >= 0 && nl < next) {
                    nl = chunk.indexOf("\n", next);
                }
                if (cr >= 0 && cr < next) {
                    cr = chunk.indexOf("\r", next);
                }
            }
            if (start < end) {
                pending.push(start === 0 ? chunk : chunk.slice(start));
                trailingCr = chunk.charCodeAt(end - 1) === 13;
            }
        }
        if (pending.length > 0) {
            this.lineNumber++;
            const joined = pending.join("");
            yield joined.endsWith("\r") ? joined.slice(0, -1) : joined;
        }
    }
}
