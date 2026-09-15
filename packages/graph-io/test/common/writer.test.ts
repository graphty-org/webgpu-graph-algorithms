import { describe, expect, it } from "vitest";

import {
    collectBytes,
    decodeChunks,
    DEFAULT_CHUNK_BYTES,
    encodeChunks,
    joinText,
    toReadableStream,
} from "../../src/common/writer.js";

const E_ACUTE = String.fromCodePoint(0xe9);

function* parts(): Generator<string> {
    yield "<graph>\n";
    yield "";
    yield `  <node id="caf${E_ACUTE}"/>\n`;
    yield "</graph>\n";
}

async function* asyncParts(): AsyncGenerator<string> {
    for (const p of parts()) {
        yield p;
        await Promise.resolve();
    }
}

const EXPECTED = `<graph>\n  <node id="caf${E_ACUTE}"/>\n</graph>\n`;

describe("encodeChunks", () => {
    it("coalesces parts into chunks of about the target size", async () => {
        const chunks: Uint8Array[] = [];
        for await (const chunk of encodeChunks(parts(), 12)) {
            chunks.push(chunk);
        }
        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks.slice(0, -1)) {
            expect(chunk.byteLength).toBeGreaterThanOrEqual(12);
        }
        expect(await decodeChunks(encodeChunks(parts(), 12))).toBe(EXPECTED);
        expect(await decodeChunks(encodeChunks(asyncParts(), 12))).toBe(EXPECTED);
    });

    it("emits one chunk for a small document and nothing for an empty one", async () => {
        const chunks: Uint8Array[] = [];
        for await (const chunk of encodeChunks(parts())) {
            chunks.push(chunk);
        }
        expect(chunks).toHaveLength(1);
        expect(new TextDecoder().decode(chunks[0])).toBe(EXPECTED);
        const empty: Uint8Array[] = [];
        for await (const chunk of encodeChunks([])) {
            empty.push(chunk);
        }
        expect(empty).toEqual([]);
        for await (const chunk of encodeChunks(["", ""])) {
            empty.push(chunk);
        }
        expect(empty).toEqual([]);
        expect(DEFAULT_CHUNK_BYTES).toBe(65536);
    });

    it("never splits a part", async () => {
        const big = "x".repeat(100);
        const chunks: Uint8Array[] = [];
        for await (const chunk of encodeChunks([big, "y"], 10)) {
            chunks.push(chunk);
        }
        expect(chunks.map((c) => c.byteLength)).toEqual([100, 1]);
    });
});

describe("joinText / collectBytes / toReadableStream", () => {
    it("joins sync and async parts", async () => {
        expect(await joinText(parts())).toBe(EXPECTED);
        expect(await joinText(asyncParts())).toBe(EXPECTED);
        expect(await joinText([])).toBe("");
    });

    it("collects bytes", async () => {
        const bytes = await collectBytes(encodeChunks(parts(), 8));
        expect(new TextDecoder().decode(bytes)).toBe(EXPECTED);
        expect(bytes.byteLength).toBe(new TextEncoder().encode(EXPECTED).byteLength);
    });

    it("wraps chunks as a ReadableStream and cancels the source on cancel", async () => {
        const stream = toReadableStream(encodeChunks(parts(), 8));
        const reader = stream.getReader();
        const got: Uint8Array[] = [];
        for (;;) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            got.push(value);
        }
        expect(
            new TextDecoder().decode(
                await collectBytes(
                    (async function* (): AsyncGenerator<Uint8Array> {
                        for (const g of got) {
                            yield g;
                        }
                        await Promise.resolve();
                    })(),
                ),
            ),
        ).toBe(EXPECTED);

        let finished = false;
        async function* endless(): AsyncGenerator<Uint8Array> {
            try {
                for (;;) {
                    yield new Uint8Array([120]);
                    await Promise.resolve();
                }
            } finally {
                finished = true;
            }
        }
        const cancellable = toReadableStream(endless());
        const r2 = cancellable.getReader();
        await r2.read();
        await r2.cancel();
        expect(finished).toBe(true);
    });
});
