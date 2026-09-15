/**
 * The output side of the exporter contract (design sections 8.5 and 12.4): an exporter produces
 * its document as a sequence of text parts (a sync or async generator of strings, one per line or
 * element); these helpers turn that sequence into `export()`'s `AsyncIterable<Uint8Array>` of
 * UTF-8 chunks of a bounded size, into `exportToString()`'s one string, or into a
 * `ReadableStream<Uint8Array>` for a caller that wants a stream.
 */

/**
 * Text parts an exporter produces.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export type TextParts = Iterable<string> | AsyncIterable<string>;

/** The target size of one encoded chunk; parts are coalesced up to it and never split. */
export const DEFAULT_CHUNK_BYTES = 64 * 1024;

/**
 * Encode text parts as UTF-8 chunks of about `chunkBytes` each. Small parts are coalesced; a part
 * larger than the chunk size is emitted whole. The last chunk may be short; nothing is emitted for
 * an empty document.
 * @param parts - the text parts
 * @param chunkBytes - the target chunk size in bytes
 * @yields UTF-8 chunks
 * @returns nothing
 */
export async function* encodeChunks(
    parts: TextParts,
    chunkBytes: number = DEFAULT_CHUNK_BYTES,
): AsyncGenerator<Uint8Array, void, undefined> {
    const encoder = new TextEncoder();
    let pending: string[] = [];
    let pendingUnits = 0;
    for await (const part of parts) {
        if (part.length === 0) {
            continue;
        }
        pending.push(part);
        pendingUnits += part.length;
        // UTF-16 code units are a lower bound on the UTF-8 byte count; flush when the lower bound reaches the target
        if (pendingUnits >= chunkBytes) {
            yield encoder.encode(pending.length === 1 ? pending[0] : pending.join(""));
            pending = [];
            pendingUnits = 0;
        }
    }
    if (pending.length > 0) {
        yield encoder.encode(pending.length === 1 ? pending[0] : pending.join(""));
    }
}

/**
 * Join text parts into one string.
 * @param parts - the text parts
 * @returns the whole document
 */
export async function joinText(parts: TextParts): Promise<string> {
    const collected: string[] = [];
    for await (const part of parts) {
        collected.push(part);
    }
    return collected.join("");
}

/**
 * Decode UTF-8 chunks back into one string (for tests and for callers holding an export() result).
 * @param chunks - the chunks
 * @returns the decoded text
 */
export async function decodeChunks(chunks: AsyncIterable<Uint8Array>): Promise<string> {
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const parts: string[] = [];
    for await (const chunk of chunks) {
        parts.push(decoder.decode(chunk, { stream: true }));
    }
    parts.push(decoder.decode());
    return parts.join("");
}

/**
 * Collect UTF-8 chunks into one Uint8Array.
 * @param chunks - the chunks
 * @returns the concatenated bytes
 */
export async function collectBytes(chunks: AsyncIterable<Uint8Array>): Promise<Uint8Array> {
    const parts: Uint8Array[] = [];
    let total = 0;
    for await (const chunk of chunks) {
        parts.push(chunk);
        total += chunk.byteLength;
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
        out.set(part, offset);
        offset += part.byteLength;
    }
    return out;
}

/**
 * Wrap an async iterable of chunks as a ReadableStream, pulling one chunk per read and cancelling
 * the iterable when the stream is cancelled.
 * @param chunks - the chunks
 * @returns a byte stream
 */
export function toReadableStream(chunks: AsyncIterable<Uint8Array>): ReadableStream<Uint8Array> {
    const iterator = chunks[Symbol.asyncIterator]();
    return new ReadableStream<Uint8Array>({
        async pull(controller): Promise<void> {
            const { done, value } = await iterator.next();
            if (done) {
                controller.close();
                return;
            }
            controller.enqueue(value);
        },
        async cancel(): Promise<void> {
            await iterator.return?.(undefined);
        },
    });
}
