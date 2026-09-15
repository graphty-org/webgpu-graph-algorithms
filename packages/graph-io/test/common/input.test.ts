import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    inputLength,
    INVALID_UTF8_CODE,
    isImportInput,
    LineReader,
    readText,
    textChunks,
    throwIfAborted,
} from "../../src/common/input.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { ImportError, type ImportInput } from "../../src/types.js";
import { byteChunks, byteStream, textChunksOf } from "../helpers/corpus.js";

const encoder = new TextEncoder();

// 2-byte (e-acute), 3-byte (CJK), 4-byte (emoji) sequences built from code points so the source stays ASCII
const MULTIBYTE = `caf${String.fromCodePoint(0xe9)} ${String.fromCodePoint(0x4e2d, 0x6587)} ${String.fromCodePoint(0x1f600)} end`;

function report(): ImportReportBuilder {
    return new ImportReportBuilder("test", 100);
}

async function collect(input: ImportInput, r = report(), options = {}): Promise<string[]> {
    const out: string[] = [];
    for await (const chunk of textChunks(input, r, options)) {
        out.push(chunk);
    }
    return out;
}

async function linesOf(input: ImportInput, r = report()): Promise<{ lines: string[]; numbers: number[] }> {
    const reader = new LineReader(input, r);
    const lines: string[] = [];
    const numbers: number[] = [];
    for await (const line of reader) {
        lines.push(line);
        numbers.push(reader.line);
    }
    return { lines, numbers };
}

describe("isImportInput / inputLength", () => {
    it("recognises every member of the union", () => {
        expect(isImportInput("x")).toBe(true);
        expect(isImportInput(new Uint8Array(2))).toBe(true);
        expect(isImportInput(byteStream(new Uint8Array(2), 1))).toBe(true);
        expect(isImportInput(byteChunks(new Uint8Array(2), 1))).toBe(true);
        expect(isImportInput(42)).toBe(false);
        expect(isImportInput(null)).toBe(false);
        expect(isImportInput({})).toBe(false);
    });

    it("knows the length of in-memory input only", () => {
        expect(inputLength("abc")).toBe(3);
        expect(inputLength(new Uint8Array(5))).toBe(5);
        expect(inputLength(byteStream(new Uint8Array(5), 1))).toBeNull();
        expect(inputLength(byteChunks(new Uint8Array(5), 1))).toBeNull();
    });
});

describe("textChunks", () => {
    it("reads a string as one chunk", async () => {
        expect(await collect("hello")).toEqual(["hello"]);
        expect(await collect("")).toEqual([]);
    });

    it("decodes bytes, streams, byte iterables and text iterables to the same text", async () => {
        const bytes = encoder.encode(MULTIBYTE);
        expect((await collect(bytes)).join("")).toBe(MULTIBYTE);
        expect((await collect(byteStream(bytes, 3))).join("")).toBe(MULTIBYTE);
        expect((await collect(byteChunks(bytes, 2))).join("")).toBe(MULTIBYTE);
        expect((await collect(textChunksOf(MULTIBYTE, 2))).join("")).toBe(MULTIBYTE);
    });

    it("decodes multi-byte characters split across every chunk boundary", async () => {
        const bytes = encoder.encode(MULTIBYTE);
        for (let size = 1; size <= 5; size++) {
            expect((await collect(byteChunks(bytes, size))).join("")).toBe(MULTIBYTE);
            expect((await collect(byteStream(bytes, size))).join("")).toBe(MULTIBYTE);
        }
    });

    it("finishes a pending byte sequence when a text chunk follows in a mixed iterable", async () => {
        const bytes = encoder.encode(String.fromCodePoint(0x1f600));
        async function* mixed(): AsyncGenerator<string | Uint8Array> {
            yield bytes.subarray(0, 2);
            yield bytes.subarray(2);
            yield " text";
            await Promise.resolve();
        }
        expect((await collect(mixed())).join("")).toBe(`${String.fromCodePoint(0x1f600)} text`);
    });

    it("strips a leading BOM from text and from bytes but keeps a later one", async () => {
        const bom = String.fromCharCode(0xfeff);
        expect((await collect(`${bom}abc`)).join("")).toBe("abc");
        expect((await collect(encoder.encode(`${bom}abc`))).join("")).toBe("abc");
        expect((await collect(byteChunks(encoder.encode(`${bom}abc`), 1))).join("")).toBe("abc");
        expect((await collect(textChunksOf(`${bom}abc`, 2))).join("")).toBe("abc");
        expect((await collect(`a${bom}bc`)).join("")).toBe(`a${bom}bc`);
        expect(await collect(bom)).toEqual([]);
    });

    it("reports invalid UTF-8 as a parse-error and aborts with ImportError", async () => {
        const r = report();
        let caught: unknown;
        try {
            await collect(new Uint8Array([0x61, 0xff, 0x62]), r);
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const err = caught as ImportError;
        expect(err.report.issues).toHaveLength(1);
        expect(err.report.issues[0]).toMatchObject({
            category: "parse-error",
            severity: "error",
            code: INVALID_UTF8_CODE,
        });
        expect(err.details.code).toBe(INVALID_UTF8_CODE);
        expect(r.errorCount).toBe(1);
    });

    it("reports a truncated multi-byte sequence at end of input", async () => {
        const bytes = encoder.encode(String.fromCodePoint(0x1f600)).subarray(0, 3);
        await expect(collect(bytes)).rejects.toBeInstanceOf(ImportError);
        await expect(collect(byteChunks(bytes, 1))).rejects.toBeInstanceOf(ImportError);
        await expect(collect(byteStream(bytes, 2))).rejects.toBeInstanceOf(ImportError);
    });

    it("never substitutes U+FFFD", async () => {
        // an overlong encoding of "/" that a lenient decoder would map to U+FFFD
        const bytes = new Uint8Array([0x61, 0xc0, 0xaf, 0x62]);
        await expect(collect(bytes)).rejects.toBeInstanceOf(ImportError);
    });

    it("rejects a chunk that is neither a string nor a Uint8Array", async () => {
        async function* bad(): AsyncGenerator<string | Uint8Array> {
            yield 42 as unknown as string;
            await Promise.resolve();
        }
        let caught: unknown;
        try {
            await collect(bad());
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        expect((caught as GraphFormatError).code).toBe("E_UNSUPPORTED");
    });

    it("reports progress with a total for in-memory input", async () => {
        const calls: [number, number | undefined][] = [];
        const onProgress = (done: number, total?: number): void => {
            calls.push([done, total]);
        };
        await collect("abcd", report(), { onProgress });
        expect(calls).toEqual([[4, 4]]);
        calls.length = 0;
        await collect(encoder.encode("abcd"), report(), { onProgress });
        expect(calls).toEqual([[4, 4]]);
        calls.length = 0;
        await collect(new Uint8Array(0), report(), { onProgress });
        expect(calls).toEqual([[0, 0]]);
        calls.length = 0;
        await collect(byteChunks(encoder.encode("abcdef"), 4), report(), { onProgress });
        expect(calls).toEqual([
            [4, undefined],
            [6, undefined],
            [6, 6],
        ]);
    });

    it("throws the abort reason before reading when the signal is already aborted", async () => {
        const controller = new AbortController();
        controller.abort();
        let caught: unknown;
        try {
            await collect("abc", report(), { signal: controller.signal });
        } catch (err) {
            caught = err;
        }
        expect((caught as { name: string }).name).toBe("AbortError");
    });

    it("stops a stream between chunks when the signal fires and cancels the stream", async () => {
        const controller = new AbortController();
        let cancelled = false;
        let pulls = 0;
        const stream = new ReadableStream<Uint8Array>({
            pull(c): void {
                pulls++;
                c.enqueue(encoder.encode("chunk\n"));
                if (pulls === 2) {
                    controller.abort(new Error("user cancelled"));
                }
            },
            cancel(): void {
                cancelled = true;
            },
        });
        const seen: string[] = [];
        let caught: unknown;
        try {
            for await (const chunk of textChunks(stream, report(), { signal: controller.signal })) {
                seen.push(chunk);
            }
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(Error);
        expect((caught as Error).message).toBe("user cancelled");
        expect(seen.length).toBeLessThan(4);
        expect(cancelled).toBe(true);
    });

    it("cancels a stream when the consumer stops early", async () => {
        let cancelled = false;
        const stream = new ReadableStream<Uint8Array>({
            pull(c): void {
                c.enqueue(encoder.encode("x"));
            },
            cancel(): void {
                cancelled = true;
            },
        });
        for await (const chunk of textChunks(stream, report())) {
            expect(chunk).toBe("x");
            break;
        }
        expect(cancelled).toBe(true);
    });

    it("throwIfAborted rethrows the signal's reason as it is, whatever its type (the platform contract)", () => {
        expect(() => throwIfAborted(null)).not.toThrow();
        expect(() => throwIfAborted(new AbortController().signal)).not.toThrow();
        const withString = new AbortController();
        withString.abort("because");
        try {
            throwIfAborted(withString.signal);
            throw new Error("did not throw");
        } catch (err) {
            expect(err).toBe(withString.signal.reason);
            expect(err).toBe("because");
        }
        const withError = new AbortController();
        const reason = new Error("boom");
        withError.abort(reason);
        expect(() => throwIfAborted(withError.signal)).toThrow(reason);
        const plain = new AbortController();
        plain.abort();
        try {
            throwIfAborted(plain.signal);
            throw new Error("did not throw");
        } catch (err) {
            expect((err as Error).name).toBe("AbortError");
        }
    });
});

describe("readText", () => {
    it("returns the whole text for every input shape, BOM removed", async () => {
        const bom = String.fromCharCode(0xfeff);
        const text = `${bom}${MULTIBYTE}\nsecond line`;
        const bytes = encoder.encode(text);
        const expected = `${MULTIBYTE}\nsecond line`;
        expect(await readText(text, report())).toBe(expected);
        expect(await readText(bytes, report())).toBe(expected);
        expect(await readText(byteChunks(bytes, 3), report())).toBe(expected);
        expect(await readText(byteStream(bytes, 5), report())).toBe(expected);
        expect(await readText(textChunksOf(text, 4), report())).toBe(expected);
    });

    it("honours the signal and reports progress for a string", async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(readText("abc", report(), { signal: controller.signal })).rejects.toMatchObject({
            name: "AbortError",
        });
        const calls: [number, number | undefined][] = [];
        await readText("abc", report(), { onProgress: (d, t): void => void calls.push([d, t]) });
        expect(calls).toEqual([[3, 3]]);
    });
});

describe("LineReader", () => {
    it("splits on LF, CRLF and lone CR with 1-based line numbers", async () => {
        const { lines, numbers } = await linesOf("a\nb\r\nc\rd");
        expect(lines).toEqual(["a", "b", "c", "d"]);
        expect(numbers).toEqual([1, 2, 3, 4]);
    });

    it("keeps empty lines and drops nothing but terminators", async () => {
        expect((await linesOf("a\n\nb\n")).lines).toEqual(["a", "", "b"]);
        expect((await linesOf("\n")).lines).toEqual([""]);
        expect((await linesOf("\r\n")).lines).toEqual([""]);
        expect((await linesOf("\r")).lines).toEqual([""]);
        expect((await linesOf("")).lines).toEqual([]);
        expect((await linesOf("no newline")).lines).toEqual(["no newline"]);
        expect((await linesOf("trailing\r")).lines).toEqual(["trailing"]);
        expect((await linesOf("a\n\r\n")).lines).toEqual(["a", ""]);
    });

    it("gives the same lines whatever the chunk boundaries, CRLF split included", async () => {
        const text = `${MULTIBYTE}\r\nline two\nline three\r\n\r\nlast`;
        const bytes = encoder.encode(text);
        const expected = [MULTIBYTE, "line two", "line three", "", "last"];
        expect((await linesOf(text)).lines).toEqual(expected);
        expect((await linesOf(bytes)).lines).toEqual(expected);
        for (let size = 1; size <= 7; size++) {
            expect((await linesOf(byteChunks(bytes, size))).lines).toEqual(expected);
            expect((await linesOf(textChunksOf(text, size))).lines).toEqual(expected);
        }
        expect((await linesOf(byteStream(bytes, 4))).lines).toEqual(expected);
    });

    it("counts lines across chunks and starts at 0 before iteration", async () => {
        const reader = new LineReader(textChunksOf("a\nb\nc", 1), report());
        expect(reader.line).toBe(0);
        const seen: string[] = [];
        for await (const line of reader) {
            seen.push(`${reader.line}:${line}`);
        }
        expect(seen).toEqual(["1:a", "2:b", "3:c"]);
        expect(reader.line).toBe(3);
    });

    it("surfaces a decode error as ImportError", async () => {
        await expect(linesOf(new Uint8Array([0x61, 0x0a, 0xc3]))).rejects.toBeInstanceOf(ImportError);
    });
});
