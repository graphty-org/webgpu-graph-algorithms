/**
 * Fuzz audit, cancellation lens (design section 8.4 and the CommonImportOptions contract: "the
 * importer stops between chunks and rejects with the signal's reason"): an AbortSignal that fires
 * mid-stream must make import() reject with the reason itself, the source stream must be
 * cancelled and its reader lock released (nothing left dangling for a File.stream() or a fetch
 * body), the caller's sink must be left intact and usable (the importer does not own it, so it
 * is never disposed), and no ImportError or report may be produced for a cancellation.
 *
 * The tests marked FAILS pin three defects: (1) importGraph() under format "auto" peeks the head
 * through a reader, and when the signal fires during the read that completes the head the
 * importer throws before ever touching the replaying input, so the reader lock is never released,
 * the stream is never cancelled and an async-generator input is never finalised; (2) a non-Error
 * abort reason (a string, as AbortController.abort(reason) allows) is replaced by a DOMException
 * instead of being rethrown as the contract says; (3) an abort raised while a chunk is being
 * parsed (from the sink or a callback) is honoured by DOT, GEXF and JSON but ignored by CSV,
 * GML, GraphML, Neo4j and Pajek for a string input, which is one chunk with no "between".
 */

import { GraphBuilder, type GraphSink } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, registry } from "../../src/registry.js";
import { ImportError } from "../../src/types.js";
import { CORPUS_FORMATS, type CorpusFormat, readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** One file per format, big enough to arrive in several 256-byte chunks. */
const FILES: Readonly<Record<CorpusFormat, string>> = {
    csv: "got-edges.csv",
    dot: "root.gv",
    gexf: "airlines-sample.gexf",
    gml: "football.gml",
    graphml: "got-network.graphml",
    json: "miserables.json",
    neo4j: "karate-neo4j.csv",
    pajek: "football.net",
};

interface TrackedStream {
    readonly stream: ReadableStream<Uint8Array>;
    readonly state: { pulls: number; cancelled: boolean; cancelReason: unknown; closed: boolean };
}

/** A stream of fixed-size chunks that records pulls and cancellation; `onPull` runs before each enqueue. */
function tracked(bytes: Uint8Array, size: number, onPull?: (pull: number) => void, highWaterMark = 1): TrackedStream {
    let offset = 0;
    const state = { pulls: 0, cancelled: false, cancelReason: undefined as unknown, closed: false };
    const stream = new ReadableStream<Uint8Array>(
        {
            pull(controller): void {
                state.pulls++;
                onPull?.(state.pulls);
                if (offset >= bytes.byteLength) {
                    state.closed = true;
                    controller.close();
                    return;
                }
                controller.enqueue(bytes.subarray(offset, Math.min(offset + size, bytes.byteLength)));
                offset += size;
            },
            cancel(reason): void {
                state.cancelled = true;
                state.cancelReason = reason;
            },
        },
        { highWaterMark },
    );
    return { stream, state };
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
    try {
        await promise;
    } catch (err) {
        return err;
    }
    throw new Error("expected the import to reject");
}

function expectSinkUsable(sink: GraphBuilder): void {
    // not disposed: every accessor answers and the builder still accepts pushes and freezes
    expect(() => sink.nodeCount).not.toThrow();
    expect(() => sink.addNode("audit-after-abort")).not.toThrow();
    const snapshot = sink.freeze();
    expect(snapshot.ids.has("audit-after-abort")).toBe(true);
}

describe("fuzz audit: AbortSignal mid-stream", () => {
    for (const format of CORPUS_FORMATS) {
        const bytes = readCorpusBytes(format, FILES[format]);

        it(`${format}: rejects with the Error reason itself, cancels the stream and leaves the sink usable`, async () => {
            const ac = new AbortController();
            const reason = new Error("stop now");
            const t = tracked(bytes, 256, (n) => {
                if (n === 3) {
                    ac.abort(reason);
                }
            });
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const err = await rejection(registry.importer(format).import(t.stream, sink, { signal: ac.signal }));
            expect(err).toBe(reason);
            expect(err).not.toBeInstanceOf(ImportError);
            expect(t.state.cancelled, "source stream cancelled").toBe(true);
            expect(t.stream.locked, "reader lock released").toBe(false);
            expect(t.state.closed, "the stream was not read to the end").toBe(false);
            expectSinkUsable(sink);
        });

        it(`${format}: rejects with the DOMException AbortError of a reason-less abort`, async () => {
            const ac = new AbortController();
            const t = tracked(bytes, 256, (n) => {
                if (n === 3) {
                    ac.abort();
                }
            });
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const err = await rejection(registry.importer(format).import(t.stream, sink, { signal: ac.signal }));
            expect(err).toBe(ac.signal.reason);
            expect((err as Error).name).toBe("AbortError");
            expect(t.state.cancelled).toBe(true);
            expect(t.stream.locked).toBe(false);
        });

        it(`${format}: an abort raised from onProgress stops at the next chunk`, async () => {
            const ac = new AbortController();
            const reason = new Error("progress abort");
            const t = tracked(bytes, 256);
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            let calls = 0;
            const err = await rejection(
                registry.importer(format).import(t.stream, sink, {
                    signal: ac.signal,
                    onProgress: (): void => {
                        calls++;
                        if (calls === 2) {
                            ac.abort(reason);
                        }
                    },
                }),
            );
            expect(err).toBe(reason);
            expect(calls, "no progress after the abort").toBe(2);
            expect(t.state.cancelled).toBe(true);
            expect(t.stream.locked).toBe(false);
            expect(t.state.closed).toBe(false);
        });

        it(`${format}: an already-aborted signal rejects before the stream is read or locked`, async () => {
            const ac = new AbortController();
            const reason = new Error("pre-aborted");
            ac.abort(reason);
            const t = tracked(bytes, 256, undefined, 0);
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const err = await rejection(registry.importer(format).import(t.stream, sink, { signal: ac.signal }));
            expect(err).toBe(reason);
            expect(t.state.pulls).toBe(0);
            expect(t.stream.locked).toBe(false);
            expect(sink.nodeCount).toBe(0);
        });

        it(`${format}: an abort mid-stream through importGraph() with an explicit format cancels the stream`, async () => {
            const ac = new AbortController();
            const reason = new Error("registry abort");
            const t = tracked(bytes, 256, (n) => {
                if (n === 5) {
                    ac.abort(reason);
                }
            });
            const err = await rejection(importGraph(t.stream, { format, signal: ac.signal }));
            expect(err).toBe(reason);
            expect(t.state.cancelled).toBe(true);
            expect(t.stream.locked).toBe(false);
        });

        it(`${format}: an abort during the sniff peek of importGraph("auto") cancels the stream`, async () => {
            const ac = new AbortController();
            const reason = new Error("abort during peek");
            const t = tracked(bytes, 256, (n) => {
                if (n === 1) {
                    ac.abort(reason);
                }
            });
            const err = await rejection(importGraph(t.stream, { signal: ac.signal, filename: FILES[format] }));
            expect(err).toBe(reason);
            expect(t.state.cancelled).toBe(true);
            expect(t.stream.locked).toBe(false);
        });

        it.skipIf(bytes.byteLength < 8192)(
            `${format}: an abort as the sniff peek completes still cancels the stream (no dangling reader)`,
            async () => {
                // FAILS: peekHead() (src/registry.ts) checks the signal at the top of its read loop but
                // not after the read that completes the 8 KB head; importer.import() then throws from
                // textChunks()'s first throwIfAborted() before iterating the replaying input, whose
                // return() is the only place the reader is cancelled and released. The stream stays
                // locked and un-cancelled: a File.stream() or fetch body is left open. (A file under
                // 8 KB is read to its end by the peek, which releases the reader, so it is skipped.)
                const ac = new AbortController();
                const reason = new Error("abort as the peek completes");
                const t = tracked(
                    bytes,
                    8192,
                    (n) => {
                        if (n === 1) {
                            ac.abort(reason);
                        }
                    },
                    0,
                );
                const err = await rejection(importGraph(t.stream, { signal: ac.signal, filename: FILES[format] }));
                expect(err).toBe(reason);
                expect(t.stream.locked, "reader lock released").toBe(false);
                expect(t.state.cancelled, "source stream cancelled").toBe(true);
            },
        );
    }

    it("an async-generator input is finalised when the abort lands as the sniff peek completes", async () => {
        // FAILS: same window as above; the generator's finally block never runs because return()
        // is never called on the replaying iterator.
        const bytes = readCorpusBytes("gml", FILES.gml);
        const ac = new AbortController();
        let finalized = false;
        async function* input(): AsyncGenerator<Uint8Array, void, undefined> {
            try {
                ac.abort(new Error("abort before the first chunk is delivered"));
                yield bytes.subarray(0, 8192);
                await Promise.resolve();
                yield bytes.subarray(8192);
            } finally {
                finalized = true;
            }
        }
        await rejection(importGraph(input(), { signal: ac.signal, filename: FILES.gml }));
        expect(finalized, "the generator's finally ran").toBe(true);
    });

    it("an async-generator input is finalised when the abort lands mid-stream (direct importer)", async () => {
        const bytes = readCorpusBytes("csv", FILES.csv);
        const ac = new AbortController();
        let finalized = false;
        let delivered = 0;
        async function* input(): AsyncGenerator<Uint8Array, void, undefined> {
            try {
                for (let offset = 0; offset < bytes.byteLength; offset += 256) {
                    delivered++;
                    if (delivered === 4) {
                        ac.abort(new Error("mid-stream"));
                    }
                    yield bytes.subarray(offset, offset + 256);
                    await Promise.resolve();
                }
            } finally {
                finalized = true;
            }
        }
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await rejection(registry.importer("csv").import(input(), sink, { signal: ac.signal }));
        expect(finalized).toBe(true);
        expect(delivered).toBe(4);
    });

    it("a string abort reason is rethrown as the reason, not replaced by a DOMException", async () => {
        // FAILS: throwIfAborted() (src/common/input.ts) wraps a non-Error reason in a DOMException
        // named AbortError, so the caller cannot compare the rejection with signal.reason (the
        // platform's signal.throwIfAborted() and fetch() both throw the reason as it is).
        const bytes = readCorpusBytes("csv", FILES.csv);
        const ac = new AbortController();
        const t = tracked(bytes, 256, (n) => {
            if (n === 3) {
                ac.abort("user cancelled");
            }
        });
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const err = await rejection(registry.importer("csv").import(t.stream, sink, { signal: ac.signal }));
        expect(t.state.cancelled).toBe(true);
        expect(err).toBe(ac.signal.reason);
    });

    it("a cancellation never produces an ImportError or an issue, whatever chunk it lands on", async () => {
        for (const format of CORPUS_FORMATS) {
            const bytes = readCorpusBytes(format, FILES[format]);
            const chunkCount = Math.ceil(bytes.byteLength / 512);
            for (const at of [1, 2, Math.max(2, Math.floor(chunkCount / 2)), Math.max(2, chunkCount - 1)]) {
                const ac = new AbortController();
                const reason = new Error(`abort at pull ${at}`);
                const t = tracked(bytes, 512, (n) => {
                    if (n === at) {
                        ac.abort(reason);
                    }
                });
                const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
                const err = await rejection(registry.importer(format).import(t.stream, sink, { signal: ac.signal }));
                expect(err, `${format} pull ${at}`).toBe(reason);
                expect(t.state.cancelled, `${format} pull ${at}: cancelled`).toBe(true);
                expect(t.stream.locked).toBe(false);
            }
        }
    });

    describe("an abort raised while a single in-memory chunk is being parsed", () => {
        for (const format of CORPUS_FORMATS) {
            it(`${format}: rejects with the reason instead of finishing the import`, async () => {
                // FAILS for csv, gml, graphml, neo4j and pajek: their only signal check is between
                // chunks, and a string is one chunk, so an abort raised from the sink (or any
                // callback) during the parse is ignored and import() resolves with a full report;
                // DOT, GEXF and JSON check the signal every few hundred elements and reject.
                const text = readCorpusText(format, FILES[format]);
                const ac = new AbortController();
                const reason = new Error("abort from the sink");
                const inner = new GraphBuilder({ directed: true, weightDtype: "f64" });
                let pushes = 0;
                const sink = new Proxy(inner, {
                    get(target, prop, receiver): unknown {
                        if (prop === "addNode" || prop === "addEdge") {
                            return (...args: unknown[]): number => {
                                pushes++;
                                if (pushes === 10) {
                                    ac.abort(reason);
                                }
                                return (Reflect.get(target, prop, receiver) as (...a: unknown[]) => number).apply(
                                    target,
                                    args,
                                );
                            };
                        }
                        const value: unknown = Reflect.get(target, prop, receiver);
                        return typeof value === "function"
                            ? (value as (...a: unknown[]) => unknown).bind(target)
                            : value;
                    },
                }) as unknown as GraphSink;
                const err = await rejection(registry.importer(format).import(text, sink, { signal: ac.signal }));
                expect(err).toBe(reason);
                expect(pushes, "pushes after the abort").toBeLessThan(1000);
            });

            it(`${format}: an abort raised from the sink's 20th addEdge rejects within 64 more edges`, async () => {
                // The claim under test is "every importer checks the signal every 64 elements": the
                // abort lands inside the edge loop (after any check placed between the sections),
                // so a single check between the node and edge sections would not catch it.
                const text = readCorpusText(format, FILES[format]);
                const ac = new AbortController();
                const reason = new Error("abort from addEdge");
                const inner = new GraphBuilder({ directed: true, weightDtype: "f64" });
                let edges = 0;
                const sink = new Proxy(inner, {
                    get(target, prop, receiver): unknown {
                        if (prop === "addEdge") {
                            return (...args: unknown[]): number => {
                                edges++;
                                if (edges === 20) {
                                    ac.abort(reason);
                                }
                                return (Reflect.get(target, prop, receiver) as (...a: unknown[]) => number).apply(
                                    target,
                                    args,
                                );
                            };
                        }
                        const value: unknown = Reflect.get(target, prop, receiver);
                        return typeof value === "function"
                            ? (value as (...a: unknown[]) => unknown).bind(target)
                            : value;
                    },
                }) as unknown as GraphSink;
                const err = await rejection(registry.importer(format).import(text, sink, { signal: ac.signal }));
                expect(err).toBe(reason);
                expect(edges, "edges pushed after the abort").toBeLessThanOrEqual(20 + 64);
            });

            it(`${format}: an abort raised from the sink's last addEdge still rejects (checked before finish())`, async () => {
                // The periodic check is every 64 elements, so an abort during the last few elements
                // is only seen by the check every importer runs before report.finish().
                const text = readCorpusText(format, FILES[format]);
                const plain = new GraphBuilder({ directed: true, weightDtype: "f64" });
                const total = (await registry.importer(format).import(text, plain, {})).counts.edges;
                expect(total).toBeGreaterThan(0);
                const ac = new AbortController();
                const reason = new Error("abort from the last addEdge");
                const inner = new GraphBuilder({ directed: true, weightDtype: "f64" });
                let edges = 0;
                const sink = new Proxy(inner, {
                    get(target, prop, receiver): unknown {
                        if (prop === "addEdge") {
                            return (...args: unknown[]): number => {
                                edges++;
                                if (edges === total) {
                                    ac.abort(reason);
                                }
                                return (Reflect.get(target, prop, receiver) as (...a: unknown[]) => number).apply(
                                    target,
                                    args,
                                );
                            };
                        }
                        const value: unknown = Reflect.get(target, prop, receiver);
                        return typeof value === "function"
                            ? (value as (...a: unknown[]) => unknown).bind(target)
                            : value;
                    },
                }) as unknown as GraphSink;
                const err = await rejection(registry.importer(format).import(text, sink, { signal: ac.signal }));
                expect(err).toBe(reason);
            });
        }
    });

    it("the sink is never disposed by an importer: a caller's builder keeps the partial graph after an abort", async () => {
        const bytes = readCorpusBytes("pajek", FILES.pajek);
        const ac = new AbortController();
        const t = tracked(bytes, 256, (n) => {
            if (n === 6) {
                ac.abort(new Error("stop"));
            }
        });
        const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await rejection(registry.importer("pajek").import(t.stream, sink, { signal: ac.signal }));
        expect(sink.nodeCount).toBeGreaterThan(0);
        const before = sink.nodeCount;
        const snapshot = sink.freeze();
        expect(snapshot.nodeCount).toBe(before);
        snapshot.validate({ level: "full" });
        // a second import into the same sink still works
        const report = await registry.importer("json").import('{"nodes":[{"id":"after-abort"}],"links":[]}', sink, {});
        expect(report.counts.nodes).toBe(1);
        expect(sink.nodeCount).toBe(before + 1);
    });
});
