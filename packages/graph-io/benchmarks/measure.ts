/**
 * Measurement helpers of the streaming audit: a ReadableStream over a file that performs real
 * asynchronous reads (so the event loop turns between chunks and a timer can sample memory while
 * an importer runs), forced garbage collection without a command-line flag, timer-based heap
 * sampling, and a chunk iterable that records the heap retained after a GC at points during the
 * stream (the high-water mark a streaming parser must keep flat).
 */

import { open } from "node:fs/promises";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";

/** The full garbage collector, obtained through the V8 flag API (no `--expose-gc` needed). */
const gc: () => void = ((): (() => void) => {
    setFlagsFromString("--expose-gc");
    return runInNewContext("gc") as () => void;
})();

/**
 * Collect everything collectable: three full collections, so objects released by finalisers and
 * weak references of the first pass are gone by the last.
 */
export function fullGc(): void {
    gc();
    gc();
    gc();
}

/**
 * A byte ReadableStream over a file, one `read()` system call per chunk through a promise, so
 * every chunk boundary is a macrotask turn (a `setInterval` sampler runs between chunks).
 * @param path - the file
 * @param chunkBytes - bytes per chunk
 * @returns the stream
 */
export function fileStream(path: string, chunkBytes: number = 64 * 1024): ReadableStream<Uint8Array> {
    let handle: Awaited<ReturnType<typeof open>> | null = null;
    return new ReadableStream<Uint8Array>({
        async pull(controller): Promise<void> {
            handle ??= await open(path, "r");
            const buffer = new Uint8Array(chunkBytes);
            const { bytesRead } = await handle.read(buffer, 0, chunkBytes, null);
            if (bytesRead === 0) {
                await handle.close();
                handle = null;
                controller.close();
                return;
            }
            controller.enqueue(bytesRead === chunkBytes ? buffer : buffer.subarray(0, bytesRead));
        },
        async cancel(): Promise<void> {
            await handle?.close();
            handle = null;
        },
    });
}

/**
 * A file as an async iterable of byte chunks of a fixed size, read up front (so the chunk size,
 * not disk latency, is the only variable).
 * @param bytes - the file content
 * @param chunkBytes - bytes per chunk
 * @yields one chunk at a time
 * @returns nothing
 */
export async function* byteChunks(bytes: Uint8Array, chunkBytes: number): AsyncGenerator<Uint8Array, void, undefined> {
    for (let offset = 0; offset < bytes.byteLength; offset += chunkBytes) {
        yield bytes.subarray(offset, Math.min(offset + chunkBytes, bytes.byteLength));
        await Promise.resolve();
    }
}

/** What a sampled run reports. */
export interface HeapProfile {
    /** Wall time of the run, milliseconds. */
    readonly ms: number;
    /** heapUsed before the run (after a full GC), bytes. */
    readonly baseHeap: number;
    /** The highest heapUsed seen by the sampler or at the end of the run, bytes. */
    readonly peakHeap: number;
    /** heapUsed right after the run, before any GC, bytes. */
    readonly endHeap: number;
    /** heapUsed after the run and a full GC: what the run retained, bytes. */
    readonly retainedHeap: number;
    /** How many times the sampler ran during the run (0 when the run never yielded to the event loop). */
    readonly samples: number;
    /** The sampled heapUsed values in order, bytes. */
    readonly timeline: readonly number[];
}

/**
 * Run an async body while a timer samples `process.memoryUsage().heapUsed` every `intervalMs`.
 * @param body - the run
 * @param intervalMs - the sampling interval (100 ms by default)
 * @returns the profile
 */
export async function sampleHeap(body: () => Promise<unknown>, intervalMs = 100): Promise<HeapProfile> {
    fullGc();
    const baseHeap = process.memoryUsage().heapUsed;
    const timeline: number[] = [];
    const timer = setInterval(() => {
        timeline.push(process.memoryUsage().heapUsed);
    }, intervalMs);
    const t0 = performance.now();
    try {
        await body();
    } finally {
        clearInterval(timer);
    }
    const ms = performance.now() - t0;
    const endHeap = process.memoryUsage().heapUsed;
    fullGc();
    const retainedHeap = process.memoryUsage().heapUsed;
    return {
        ms,
        baseHeap,
        peakHeap: Math.max(endHeap, ...timeline),
        endHeap,
        retainedHeap,
        samples: timeline.length,
        timeline,
    };
}

/**
 * Wrap a chunk iterable so that after every `every` chunks a full GC runs and the heap then in
 * use is recorded: the high-water mark of these readings is what the consumer retained while
 * streaming, independent of garbage-collector timing.
 * @param chunks - the chunks
 * @param every - chunks between readings
 * @param readings - receives one heapUsed value per reading, in bytes
 * @yields the chunks unchanged
 * @returns nothing
 */
export async function* retainedWhileStreaming(
    chunks: AsyncIterable<Uint8Array>,
    every: number,
    readings: number[],
): AsyncGenerator<Uint8Array, void, undefined> {
    let count = 0;
    for await (const chunk of chunks) {
        yield chunk;
        count++;
        if (count % every === 0) {
            fullGc();
            readings.push(process.memoryUsage().heapUsed);
        }
    }
}

/**
 * Bytes as mebibytes with one decimal, for reports.
 * @param bytes - a byte count
 * @returns the text
 */
export function mib(bytes: number): string {
    return `${(bytes / 1048576).toFixed(1)} MiB`;
}
