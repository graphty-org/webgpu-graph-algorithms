/**
 * Audit (design sections 4.1, 8.2, 8.4, 8.5, 8.6 and the decision log): the semantic rules of
 * @graphty/graph-io, exercised through the public surface (the registry's importGraph /
 * exportGraphToString / checkExport and the importer / exporter objects) rather than the helpers.
 *
 * - 4.1: the "canonical" id rule (default for text-cell formats, "keep" for JSON), "number" merges
 *   as coercion issues, one rule across a paired node table and edge table;
 * - 8.4: the registry's importGraph (placeholder direction overridden by the file, f64 weights by
 *   default, the result shape), every ImportInput shape, onProgress, signal, the option defaults
 *   (hyperedges "skip" with a report entry, errorLimit 100, restoreMangledIds true), the two
 *   direction rules under every onMixedDirection value and the locked / non-empty sink precedence;
 * - 8.6: ImportReport counts, errorCount / warningCount, truncation at errorLimit, ImportError
 *   with the partial report, every IssueCategory reachable through a built-in importer;
 * - 8.5: every LOSS code reachable through a built-in exporter's check(), sanitizeIds "error" /
 *   "mangle" (Q26) and restoreMangledIds, onMixedDirection on export;
 *
 * Tests that fail deliberately pin the audit's findings and say so in their names.
 */

import { type Column, GraphBuilder, GraphFormatError, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    checkCapabilities,
    checkExport,
    childrenCsr,
    csvExporter,
    csvImporter,
    DEFAULT_ERROR_LIMIT,
    DirectionResolver,
    exportGraphToString,
    gexfImporter,
    graphmlExporter,
    graphmlImporter,
    ImportError,
    importGraph,
    type ImportReport,
    ImportReportBuilder,
    type IssueCategory,
    LOSS,
    type LossNote,
    NO_CAPABILITIES,
    pajekExporter,
    resolveExportOptions,
} from "../../src/index.js";

// ============================================================ helpers

const GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns";

function graphml(edgedefault: "directed" | "undirected", body: string, keys = ""): string {
    return `<?xml version="1.0"?><graphml xmlns="${GRAPHML_NS}">${keys}<graph edgedefault="${edgedefault}">${body}</graph></graphml>`;
}

/** An undirected header, one undirected edge, one directed edge and one undirected self-loop. */
const MIXED_GRAPHML = graphml(
    "undirected",
    '<node id="a"/><node id="b"/><node id="c"/><edge source="a" target="b"/><edge source="b" target="c" directed="true"/><edge source="c" target="c"/>',
);

function issueCodes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

function categoriesOf(report: ImportReport): IssueCategory[] {
    return report.issues.map((i) => i.category);
}

function noteCodes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

/** The values of a u32 column (the pair role), or [] when absent or of another dtype. */
function u32Values(column: Column | null): number[] {
    return column !== null && column.dtype === "u32" ? Array.from(column.data) : [];
}

async function importError(run: () => Promise<unknown>): Promise<ImportError> {
    try {
        await run();
    } catch (err) {
        if (err instanceof ImportError) {
            return err;
        }
        throw err;
    }
    throw new Error("expected an ImportError");
}

/** A directed snapshot on integer ids with one directed edge, one expanded undirected edge and an undirected loop. */
function mutualSnapshot(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const r = new DirectionResolver(b, new ImportReportBuilder("probe", 100), "expand");
    r.setHeader(true);
    r.addEdge(1, 2, "mutual");
    r.addEdge(2, 3, "directed");
    return b.freeze();
}

function mixedSnapshot(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const r = new DirectionResolver(b, new ImportReportBuilder("probe", 100), "expand");
    r.setHeader(true);
    r.addEdge(1, 2, "directed");
    r.addEdge(2, 3, "undirected");
    r.addEdge(3, 3, "undirected");
    return b.freeze();
}

// ============================================================ 4.1 canonical ids

describe("design 4.1: id coercion is the importer's, canonical by default for text cells", () => {
    it('"canonical": canonical safe-integer text becomes a number, every other text stays a string (injective)', async () => {
        const { snapshot } = await importGraph("source,target\n1,01\n1.0,+1\n-5,9007199254740993\n-0,x y\n", {
            format: "csv",
        });
        expect(snapshot.ids.toArray()).toEqual([1, "01", "1.0", "+1", -5, "9007199254740993", "-0", "x y"]);
        expect(snapshot.nodeCount).toBe(8);
    });

    it('"01" / "1" / "1.0" stay distinct (design 16.5) and String(id) reproduces the cell on export', async () => {
        const { snapshot } = await importGraph("source,target\n01,1\n1.0,1\n", { format: "csv" });
        expect(snapshot.ids.toArray()).toEqual(["01", 1, "1.0"]);
        const text = await exportGraphToString(snapshot, "csv", { dialect: "generic" });
        expect(text).toBe("source,target\n01,1\n1.0,1\n");
    });

    it('"number" can merge "01" and "1"; the merge is a coercion issue', async () => {
        const { snapshot, report } = await importGraph("source,target\n1,01\n", { format: "csv", ids: "number" });
        expect(snapshot.ids.toArray()).toEqual([1]);
        expect(issueCodes(report)).toContain("W_ID_MERGED");
        expect(report.issues.find((i) => i.code === "W_ID_MERGED")?.category).toBe("coercion");
    });

    it('"string" keeps every cell as text; "keep" is the JSON default so typed ids pass through', async () => {
        const csv = await importGraph("source,target\n1,01\n", { format: "csv", ids: "string" });
        expect(csv.snapshot.ids.toArray()).toEqual(["1", "01"]);
        const json = await importGraph(
            '{"nodes":[{"id":1},{"id":"1"},{"id":"01"}],"links":[{"source":1,"target":"1"}]}',
            {
                format: "json",
            },
        );
        expect(json.snapshot.ids.toArray()).toEqual([1, "1", "01"]);
    });

    it("a paired node table and edge table are coerced with the one rule of the call (CSV `nodes`)", async () => {
        const { snapshot, report } = await importGraph("source,target\n1,2\n", {
            format: "csv",
            nodes: "id,label\n1,one\n2,two\n",
        });
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(snapshot.nodes.byRole("label")?.value(0)).toBe("one");
        expect(report.errorCount).toBe(0);
        const numeric = await importGraph("source,target\n01,2\n", {
            format: "csv",
            ids: "number",
            nodes: "id,label\n1,one\n2,two\n",
        });
        expect(numeric.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(numeric.snapshot.nodeCount).toBe(2);
    });

    it('JSON true / false / null ids are unsupported issues, coerced only under ids "string" (design 8.5)', async () => {
        const kept = await importGraph('{"nodes":[{"id":true},{"id":"b"}],"links":[]}', { format: "json" });
        expect(kept.snapshot.ids.toArray()).toEqual(["b"]);
        expect(kept.report.counts.skippedNodes).toBe(1);
        expect(kept.report.issues[0]?.category).toBe("unsupported");
        const text = await importGraph('{"nodes":[{"id":true},{"id":"b"}],"links":[]}', {
            format: "json",
            ids: "string",
        });
        expect(text.snapshot.ids.toArray()).toEqual(["true", "b"]);
    });

    it("an unknown ids value is E_UNSUPPORTED (the core's convention)", async () => {
        await expect(importGraph("a,b\n", { format: "csv", ids: "bogus" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });
});

// ============================================================ 8.4 registry and options

describe("design 8.4: importGraph and the common options", () => {
    it("returns { snapshot, report, freeze } (plus format and sniff), frozen", async () => {
        const result = await importGraph("source,target\n1,2\n", { format: "csv" });
        expect(Object.keys(result)).toEqual(["format", "sniff", "snapshot", "report", "freeze"]);
        expect(Object.isFrozen(result)).toBe(true);
        expect(result.format).toBe("csv");
        expect(result.sniff).toBeNull();
        expect(result.snapshot.edgeCount).toBe(1);
        expect(result.report.format).toBe("csv");
        expect(result.freeze.timings).toBeDefined();
        const sniffed = await importGraph("source,target\n1,2\n", { filename: "x.csv" });
        expect(sniffed.sniff?.format).toBe("csv");
    });

    it("creates the builder with directed: true as a placeholder that the file overrides", async () => {
        const undirected = await importGraph(
            graphml("undirected", '<node id="a"/><node id="b"/><edge source="a" target="b"/>'),
            {
                format: "graphml",
            },
        );
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edges.byRole("pair")).toBeNull();
        const directed = await importGraph(
            graphml("directed", '<node id="a"/><node id="b"/><edge source="a" target="b"/>'),
            {
                format: "graphml",
            },
        );
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edges.byRole("pair")).toBeNull();
    });

    it('stages weights as f64 by default (0.1 and 16777217 survive) and "f32" is an explicit opt-in', async () => {
        const f64 = await importGraph("source,target,weight\n1,2,0.1\n2,3,16777217\n", { format: "csv" });
        const shadow = f64.snapshot.edges.byRole("weight");
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.value(0)).toBe(0.1);
        expect(shadow?.value(1)).toBe(16777217);
        const f32 = await importGraph("source,target,weight\n1,2,0.1\n", { format: "csv", weightDtype: "f32" });
        expect(f32.snapshot.edges.byRole("weight")).toBeNull();
        expect(f32.snapshot.edgeList().weights?.[0]).toBe(Math.fround(0.1));
    });

    it("reads every ImportInput shape and reports progress (bytesTotal known for in-memory input only)", async () => {
        const text = "source,target\n1,2\n2,3\n";
        const bytes = new TextEncoder().encode(text);
        const shapes: Record<
            string,
            string | Uint8Array | ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>
        > = {
            string: text,
            bytes,
            stream: new ReadableStream<Uint8Array>({
                start(c) {
                    c.enqueue(bytes.slice(0, 7));
                    c.enqueue(bytes.slice(7));
                    c.close();
                },
            }),
            iterable: (async function* (): AsyncGenerator<string | Uint8Array> {
                yield "source,tar";
                await Promise.resolve();
                yield bytes.slice(10);
            })(),
        };
        for (const [name, input] of Object.entries(shapes)) {
            const progress: [number, number | undefined][] = [];
            const { snapshot } = await importGraph(input, {
                format: "csv",
                onProgress: (done, total) => {
                    progress.push([done, total]);
                },
            });
            expect(snapshot.nodeCount, name).toBe(3);
            expect(snapshot.edgeCount, name).toBe(2);
            expect(progress.at(-1), name).toEqual([bytes.length, bytes.length]);
            if (name === "string" || name === "bytes") {
                expect(
                    progress.every(([, total]) => total === bytes.length),
                    name,
                ).toBe(true);
            }
        }
    });

    it("rejects with the signal's reason when aborted (G18 cancellation)", async () => {
        const controller = new AbortController();
        controller.abort(new Error("stop"));
        await expect(importGraph("source,target\n1,2\n", { format: "csv", signal: controller.signal })).rejects.toThrow(
            "stop",
        );
    });

    it("the importer never freezes: the caller's builder is still open after import()", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await csvImporter.import("source,target\n1,2\n", builder);
        expect(builder.edgeCount).toBe(1);
        builder.addEdge(2, 3);
        expect(builder.freeze().edgeCount).toBe(2);
    });

    it("an unregistered format is E_UNSUPPORTED; an unrecognised input is E_IMPORT with E_UNKNOWN_FORMAT", async () => {
        await expect(importGraph("x", { format: "nope" })).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
        const err = await importError(() =>
            importGraph(new Uint8Array([0, 1, 2, 3]), { filename: "blob.bin", mimeType: "image/png" }),
        );
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.format).toBe("unknown");
        expect(issueCodes(err.report)).toEqual(["E_UNKNOWN_FORMAT"]);
    });

    it('hyperedges default to "skip" with a report entry (Q28); "star" / "clique" expand; "error" refuses', async () => {
        const hyper = graphml(
            "directed",
            '<node id="a"/><node id="b"/><node id="c"/><hyperedge><endpoint node="a"/><endpoint node="b"/><endpoint node="c"/></hyperedge>',
        );
        const skipped = await importGraph(hyper, { format: "graphml" });
        expect(skipped.snapshot.edgeCount).toBe(0);
        expect(skipped.report.counts.skippedEdges).toBe(1);
        expect(skipped.report.issues.map((i) => [i.category, i.severity])).toEqual([["unsupported", "warning"]]);
        const star = await importGraph(hyper, { format: "graphml", hyperedges: "star" });
        expect(star.snapshot.nodeCount).toBe(4);
        const clique = await importGraph(hyper, { format: "graphml", hyperedges: "clique" });
        expect(clique.snapshot.nodeCount).toBe(3);
        expect(clique.snapshot.edgeCount).toBeGreaterThanOrEqual(3);
        const err = await importError(() => importGraph(hyper, { format: "graphml", hyperedges: "error" }));
        expect(issueCodes(err.report)).toContain("E_HYPEREDGE");
    });

    it("GEXF defaults to undirected (Q22) and refuses undeclared edge endpoints (addMissingNodes false for edges)", async () => {
        const gexf =
            '<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3" version="1.3"><graph><nodes><node id="a"/></nodes><edges><edge id="e" source="a" target="zz"/></edges></graph></gexf>';
        const { snapshot, report } = await importGraph(gexf, { format: "gexf" });
        expect(snapshot.directed).toBe(false);
        expect(snapshot.nodeCount).toBe(1);
        expect(report.counts.skippedEdges).toBe(1);
        expect(report.issues[0]?.category).toBe("missing-value");
        const forced = await importGraph(gexf, { format: "gexf", defaultDirected: true, addMissingNodes: true });
        expect(forced.snapshot.directed).toBe(true);
        expect(forced.snapshot.nodeCount).toBe(2);
    });

    it("weightFrom null reads an unweighted graph; the default error limit is 100", async () => {
        const { snapshot } = await importGraph("source,target,weight\n1,2,7\n", { format: "csv", weightFrom: null });
        expect(snapshot.flags.weighted).toBe(false);
        expect(DEFAULT_ERROR_LIMIT).toBe(100);
    });

    it("a builder-policy option the sink does not honour is reported as a coercion issue (precedence)", async () => {
        const builder = new GraphBuilder({ directed: true, selfLoops: "drop" });
        const report = await csvImporter.import("source,target\n1,1\n", builder, { selfLoops: "keep" });
        const issue = report.issues.find((i) => i.code === "W_SINK_OPTION");
        expect(issue?.category).toBe("coercion");
        expect(issue?.element).toBe("selfLoops");
    });
});

// ============================================================ 8.4 direction rules

describe("design 8.4 / 3.6: direction is resolved by the importer in one pass", () => {
    it('"expand" (default): the mixed file becomes directed with pair / directed roles; expandedMixed counts source edges', async () => {
        const { snapshot, report } = await importGraph(MIXED_GRAPHML, { format: "graphml" });
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(4);
        const pair = snapshot.edges.byRole("pair");
        const directed = snapshot.edges.byRole("directed");
        expect(pair?.dtype).toBe("u32");
        expect(directed?.dtype).toBe("bool");
        expect(u32Values(pair)).toEqual([1, 0, INVALID_INDEX, INVALID_INDEX]);
        expect([0, 1, 2, 3].map((e) => directed?.value(e))).toEqual([false, false, true, false]);
        expect(report.counts.expandedMixed).toBe(2);
        expect(report.counts.edges).toBe(4);
        expect(report.errorCount).toBe(0);
    });

    it('"directed" and "undirected" force every edge and report the distinction once', async () => {
        const directed = await importGraph(MIXED_GRAPHML, { format: "graphml", onMixedDirection: "directed" });
        expect(directed.snapshot.directed).toBe(true);
        expect(directed.snapshot.edgeCount).toBe(3);
        expect(directed.snapshot.edges.byRole("pair")).toBeNull();
        expect(issueCodes(directed.report)).toContain("W_DIRECTION_FORCED");
        expect(categoriesOf(directed.report).every((c) => c === "coercion")).toBe(true);
        const undirected = await importGraph(MIXED_GRAPHML, { format: "graphml", onMixedDirection: "undirected" });
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edgeCount).toBe(3);
        expect(issueCodes(undirected.report)).toEqual(["W_DIRECTION_FORCED"]);
    });

    it('"error" refuses the first edge that differs with E_IMPORT and a validation-error issue', async () => {
        const err = await importError(() =>
            importGraph(MIXED_GRAPHML, { format: "graphml", onMixedDirection: "error" }),
        );
        expect(err.report.issues.map((i) => [i.category, i.code])).toEqual([["validation-error", "E_MIXED_DIRECTION"]]);
        expect(err.report.truncated).toBe(false);
    });

    it("rule 1: a locked directed sink wins over an undirected header; the file is read as the sink's and expanded", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.lockDirected();
        const report = await graphmlImporter.import(
            graphml("undirected", '<node id="a"/><node id="b"/><edge source="a" target="b"/>'),
            builder,
        );
        expect(report.issues.map((i) => [i.category, i.severity, i.code])).toEqual([
            ["coercion", "warning", "W_DIRECTION_REFUSED"],
        ]);
        const snapshot = builder.freeze();
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.edges.byRole("pair")).not.toBeNull();
        expect(report.counts.expandedMixed).toBe(1);
    });

    it('rule 2: a locked undirected sink cannot expand; "expand" aborts, "undirected" applies the policy', async () => {
        const locked = new GraphBuilder({ directed: false });
        locked.lockDirected();
        const err = await importError(() => graphmlImporter.import(MIXED_GRAPHML, locked));
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.issues.map((i) => [i.category, i.severity])).toEqual([["coercion", "error"]]);
        const forced = new GraphBuilder({ directed: false });
        forced.lockDirected();
        const report = await graphmlImporter.import(MIXED_GRAPHML, forced, { onMixedDirection: "undirected" });
        expect(forced.freeze().edgeCount).toBe(3);
        expect(issueCodes(report)).toEqual(["W_DIRECTION_FORCED"]);
    });

    it("rule 1: a non-empty unlocked undirected sink with a directed header is expanded in place", async () => {
        const builder = new GraphBuilder({ directed: false });
        builder.addEdge("x", "y");
        const report = await graphmlImporter.import(
            graphml("directed", '<node id="a"/><node id="b"/><edge source="a" target="b"/>'),
            builder,
        );
        const snapshot = builder.freeze();
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(3);
        expect(u32Values(snapshot.edges.byRole("pair"))).toEqual([1, 0, INVALID_INDEX]);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.errorCount).toBe(0);
    });

    it("an unknown onMixedDirection is E_UNSUPPORTED", async () => {
        await expect(importGraph("a,b\n", { format: "csv", onMixedDirection: "bogus" as never })).rejects.toMatchObject(
            { code: "E_UNSUPPORTED" },
        );
    });
});

// ============================================================ 8.6 report

describe("design 8.6: ImportReport counts, error limit and ImportError", () => {
    // four edges, three of them to an undeclared node under addMissingNodes: false
    const CSV = "source,target\na,b\nb,zz\nc,zz\nd,zz\na,b\n";
    const NODES = "id\na\nb\nc\nd\n";

    it("records per-element errors, skips the element and continues below the limit", async () => {
        const { snapshot, report } = await importGraph(CSV, { format: "csv", addMissingNodes: false, nodes: NODES });
        expect(report.errorCount).toBe(3);
        expect(report.warningCount).toBe(0);
        expect(report.truncated).toBe(false);
        expect(report.counts).toEqual({ nodes: 4, edges: 2, skippedNodes: 0, skippedEdges: 3, expandedMixed: 0 });
        expect(report.issues.map((i) => [i.category, i.severity, i.code, i.line, i.element])).toEqual([
            ["missing-value", "error", "E_UNKNOWN_NODE", 3, "b->zz"],
            ["missing-value", "error", "E_UNKNOWN_NODE", 4, "c->zz"],
            ["missing-value", "error", "E_UNKNOWN_NODE", 5, "d->zz"],
        ]);
        expect(report.durationMs).toBeGreaterThanOrEqual(0);
        expect(Object.isFrozen(report)).toBe(true);
        expect(snapshot.edgeCount).toBe(2);
    });

    it("errorLimit N tolerates N errors; the error beyond it is recorded, truncated is set and E_IMPORT carries the partial report", async () => {
        const tolerated = await importGraph(CSV, {
            format: "csv",
            addMissingNodes: false,
            nodes: NODES,
            errorLimit: 3,
        });
        expect(tolerated.report.truncated).toBe(false);
        expect(tolerated.report.errorCount).toBe(3);
        const err = await importError(() =>
            importGraph(CSV, { format: "csv", addMissingNodes: false, nodes: NODES, errorLimit: 2 }),
        );
        expect(err).toBeInstanceOf(GraphFormatError);
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.truncated).toBe(true);
        expect(err.report.errorCount).toBe(3);
        expect(err.report.issues).toHaveLength(3);
        expect(err.report.counts.edges).toBe(1);
        // NOTE (audit): the element whose error triggers the abort is recorded as an issue but not
        // counted in skippedEdges (2, not 3); an errorLimit of 0 aborts on the first error.
        expect(err.report.counts.skippedEdges).toBe(2);
        const first = await importError(() =>
            importGraph(CSV, { format: "csv", addMissingNodes: false, nodes: NODES, errorLimit: 0 }),
        );
        expect(first.report.errorCount).toBe(1);
        expect(first.report.truncated).toBe(true);
    });

    it("warnings never count toward the limit", async () => {
        const { report } = await importGraph("source,target\n1,01\n2,02\n3,03\n", {
            format: "csv",
            ids: "number",
            errorLimit: 0,
        });
        expect(report.warningCount).toBeGreaterThanOrEqual(2);
        expect(report.errorCount).toBe(0);
        expect(report.truncated).toBe(false);
    });

    it("a fatal parse error aborts at once with a parse-error issue (invalid UTF-8, malformed JSON)", async () => {
        const utf8 = await importError(() =>
            importGraph(new Uint8Array([0x61, 0x2c, 0x62, 0x0a, 0xff, 0xfe, 0x2c, 0x63]), { format: "csv" }),
        );
        expect(utf8.report.issues.map((i) => [i.category, i.code])).toEqual([["parse-error", "E_INVALID_UTF8"]]);
        const json = await importError(() => importGraph("{not json", { format: "json" }));
        expect(json.report.issues.map((i) => [i.category, i.code])).toEqual([["parse-error", "E_SYNTAX"]]);
    });

    it("every IssueCategory of 8.6 is reachable through a built-in importer", async () => {
        const seen = new Map<IssueCategory, string>();
        const record = (report: ImportReport): void => {
            for (const issue of report.issues) {
                if (!seen.has(issue.category)) {
                    seen.set(issue.category, issue.code);
                }
            }
        };
        // parse-error
        record((await importError(() => importGraph("{not json", { format: "json" }))).report);
        // missing-value
        record(
            (await importGraph("source,target\na,zz\n", { format: "csv", addMissingNodes: false, nodes: "id\na\n" }))
                .report,
        );
        // validation-error
        record(
            (await importError(() => importGraph(MIXED_GRAPHML, { format: "graphml", onMixedDirection: "error" })))
                .report,
        );
        // unsupported
        record((await importGraph('{"nodes":[{"id":true}],"links":[]}', { format: "json" })).report);
        // precision
        record(
            (
                await importGraph(
                    graphml(
                        "directed",
                        '<node id="a"><data key="k">9007199254740993</data></node>',
                        '<key id="k" for="node" attr.name="big" attr.type="long"/>',
                    ),
                    { format: "graphml" },
                )
            ).report,
        );
        // coercion
        record((await importGraph("source,target\n1,01\n", { format: "csv", ids: "number" })).report);
        // merged
        record((await importGraph("source,target\na,b\n", { format: "csv", nodes: "id,x\na,1\na,2\n" })).report);
        expect([...seen.keys()].sort()).toEqual(
            [
                "parse-error",
                "missing-value",
                "validation-error",
                "unsupported",
                "precision",
                "coercion",
                "merged",
            ].sort(),
        );
        expect(seen.get("precision")).toBe("W_PRECISION");
        expect(seen.get("coercion")).toBe("W_ID_MERGED");
    });

    it('a declared long beyond 2^53 is a precision issue unless long: "string" (design 5.1)', async () => {
        const doc = graphml(
            "directed",
            '<node id="a"><data key="k">9007199254740993</data></node>',
            '<key id="k" for="node" attr.name="big" attr.type="long"/>',
        );
        const f64 = await importGraph(doc, { format: "graphml" });
        expect(f64.snapshot.nodes.get("big")?.dtype).toBe("f64");
        expect(f64.snapshot.nodes.get("big")?.value(0)).toBe(9007199254740992);
        expect(f64.report.issues.map((i) => [i.category, i.severity])).toEqual([["precision", "warning"]]);
        const text = await importGraph(doc, { format: "graphml", long: "string" });
        expect(text.snapshot.nodes.get("big")?.dtype).toBe("string");
        expect(text.snapshot.nodes.get("big")?.value(0)).toBe("9007199254740993");
        expect(text.report.issues).toEqual([]);
    });
});

// ============================================================ 8.5 export

describe("design 8.5: every LOSS code is reachable through a built-in exporter's check()", () => {
    function cycle(): GraphBuilder {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge(1, 2);
        b.addEdge(2, 3);
        return b;
    }

    const CASES: readonly [
        code: string,
        format: string,
        build: () => GraphSnapshot,
        options?: Record<string, unknown>,
    ][] = [
        [LOSS.MIXED_DIRECTION_ERROR, "gml", mixedSnapshot],
        [LOSS.MIXED_DIRECTION, "gml", mixedSnapshot, { onMixedDirection: "directed" }],
        [
            LOSS.MULTI_EDGES,
            "gexf",
            () => {
                const b = cycle();
                b.addEdge(1, 2);
                return b.freeze();
            },
            { version: "1.2" },
        ],
        [LOSS.EDGE_IDS_GENERATED, "gexf", () => cycle().freeze(), { version: "1.2" }],
        [
            LOSS.EDGE_IDS_DROPPED,
            "pajek",
            () => {
                const b = cycle();
                b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
                b.setEdgeValue("id", 0, "e0");
                return b.freeze();
            },
        ],
        [
            LOSS.ID_MANGLED,
            "graphml",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge("a b", "c");
                return b.freeze();
            },
            { sanitizeIds: "mangle" },
        ],
        [
            LOSS.ID_CHARSET,
            "graphml",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge("a b", "c");
                return b.freeze();
            },
        ],
        [
            LOSS.ID_RENUMBERED,
            "pajek",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge("a", "b");
                return b.freeze();
            },
        ],
        [
            LOSS.DTYPE,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "k", dtype: "dict" });
                b.setNodeValue("k", 0, "x");
                return b.freeze();
            },
        ],
        [
            LOSS.COMPONENTS,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "v", dtype: "f64", components: 2 });
                b.setNodeValue("v", 0, [1, 2]);
                return b.freeze();
            },
        ],
        [
            LOSS.LIST,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "string" });
                b.setNodeValue("l", 0, ["a"]);
                return b.freeze();
            },
        ],
        [
            LOSS.JSON,
            "dot",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "j", dtype: "json" });
                b.setNodeValue("j", 0, { a: 1 });
                return b.freeze();
            },
        ],
        [
            LOSS.DEFAULT,
            "gml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "d", dtype: "string", default: "none" });
                b.setNodeValue("d", 0, "x");
                return b.freeze();
            },
        ],
        [
            LOSS.OPTIONS,
            "gml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "o", dtype: "string", options: ["x", "y"] });
                b.setNodeValue("o", 0, "x");
                return b.freeze();
            },
        ],
        [
            LOSS.HIERARCHY,
            "gml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
                b.setNodeValue("parent", 1, 0);
                return b.freeze();
            },
        ],
        [
            LOSS.TEMPORAL,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
                b.setNodeValue("start", 0, 1);
                return b.freeze();
            },
        ],
        [
            LOSS.SPELLS,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({
                    name: "spells",
                    dtype: "list",
                    itemDtype: "f64",
                    itemComponents: 2,
                    role: "spells",
                });
                b.setNodeValue("spells", 0, [[1, 2]]);
                return b.freeze();
            },
        ],
        [
            LOSS.DYNAMIC_VALUES,
            "graphml",
            () => {
                const b = cycle();
                const t = b.addExtensionTable("temporal:node:price", [
                    { name: "element", dtype: "u32", refersTo: "node" },
                    { name: "start", dtype: "f64", role: "start" },
                    { name: "end", dtype: "f64", role: "end" },
                    { name: "value", dtype: "f64" },
                ]);
                b.addExtensionRow(t, [0, 0, 1, 2]);
                return b.freeze();
            },
        ],
        [
            LOSS.OPEN_INTERVAL,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "open", dtype: "u8", role: "open" });
                b.setNodeValue("open", 0, 1);
                return b.freeze();
            },
        ],
        [
            LOSS.GRAPH_ATTRIBUTES,
            "pajek",
            () => {
                const b = cycle();
                b.setGraphValue("title", "t");
                return b.freeze();
            },
        ],
        [
            LOSS.POSITIONS,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "position", dtype: "f32", components: 3, role: "position" });
                b.setNodeValue("position", 0, [1, 2, 3]);
                return b.freeze();
            },
        ],
        [
            LOSS.VIZ,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "size", dtype: "f32", role: "size" });
                b.setNodeValue("size", 0, 1);
                return b.freeze();
            },
        ],
        [
            LOSS.EXTENSION_TABLE,
            "gexf",
            () => {
                const b = cycle();
                const t = b.addExtensionTable("custom:thing", [{ name: "x", dtype: "i32" }]);
                b.addExtensionRow(t, [1]);
                return b.freeze();
            },
        ],
        // the re-import notes of design section 8.5 (what the format's own importer changes)
        [
            LOSS.TEMPORAL_TEXT,
            "json",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "when", dtype: "f64" });
                b.declareNodeColumn({ name: "when.text", dtype: "string", role: "timeText", extra: { for: "when" } });
                b.setNodeValue("when", 0, 1);
                b.setNodeValue("when.text", 0, "1");
                return b.freeze();
            },
        ],
        [
            LOSS.ROLE,
            "json",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
                b.setNodeValue("name", 0, "one");
                return b.freeze();
            },
        ],
        [
            LOSS.PARENTS,
            "graphml",
            () => {
                const b = cycle();
                b.declareNodeColumn({
                    name: "parents",
                    dtype: "list",
                    itemDtype: "u32",
                    role: "parents",
                    refersTo: "node",
                });
                b.setNodeValue("parents", 0, [1, 2]);
                return b.freeze();
            },
        ],
        [LOSS.MUTUAL_EXPANDED, "json", mutualSnapshot, { dialect: "jgf" }],
        [LOSS.MUTUAL_AS_UNDIRECTED, "graphml", mutualSnapshot],
        [
            LOSS.ID_TEXT_TYPE,
            "graphml",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge("1", "2");
                return b.freeze();
            },
        ],
        [
            LOSS.ID_TEXT_COLLISION,
            "neo4j",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge(1, "1");
                return b.freeze();
            },
        ],
        [
            LOSS.WEIGHT_KEY_CLASH,
            "graphml",
            () => {
                const b = cycle();
                b.declareEdgeColumn({ name: "weight", dtype: "i32" });
                b.setEdgeValue("weight", 0, 7);
                return b.freeze();
            },
        ],
        [
            LOSS.ROLE_ASSUMED,
            "gml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "label", dtype: "string" });
                b.setNodeValue("label", 0, "plain");
                return b.freeze();
            },
            { sanitizeIds: "mangle" },
        ],
        [
            LOSS.COLUMN_NAME_CHANGED,
            "gexf",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "name", dtype: "string", role: "label" });
                b.setNodeValue("name", 0, "one");
                return b.freeze();
            },
        ],
        [
            LOSS.EMPTY_COLUMN,
            "dot",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "never", dtype: "string" });
                return b.freeze();
            },
        ],
        [
            LOSS.STORAGE_CLASS,
            "gml",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "group", dtype: "string" });
                for (let i = 0; i < 3; i++) {
                    b.setNodeValue("group", i, "same");
                }
                return b.freeze();
            },
            { sanitizeIds: "mangle" },
        ],
        [
            LOSS.INTEGRAL_F64,
            "json",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "whole", dtype: "f64" });
                b.setNodeValue("whole", 0, 2);
                return b.freeze();
            },
        ],
        [
            LOSS.TEXT_INFERRED,
            "dot",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "code", dtype: "string" });
                b.setNodeValue("code", 0, "123");
                return b.freeze();
            },
        ],
        [
            LOSS.XML_ILLEGAL_CHAR,
            "gexf",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "s", dtype: "string" });
                b.setNodeValue("s", 0, `bad ${String.fromCharCode(1)} char`);
                return b.freeze();
            },
        ],
        [
            LOSS.OPTIONS_GAINED,
            "gexf",
            () => {
                const b = cycle();
                b.declareNodeColumn({ name: "cat", dtype: "dict" });
                b.setNodeValue("cat", 0, "x");
                return b.freeze();
            },
        ],
    ];

    for (const [code, format, build, options] of CASES) {
        it(`${code} through ${format}`, () => {
            const notes = checkExport(build(), format, options);
            expect(noteCodes(notes), JSON.stringify(noteCodes(notes))).toContain(code);
            const note = notes.find((n) => n.code === code);
            expect(typeof note?.message).toBe("string");
            expect(note?.column === null || typeof note?.column === "string").toBe(true);
            expect(note?.count === null || typeof note?.count === "number").toBe(true);
        });
    }

    it("W_SELF_LOOPS is unreachable through the built-ins (every exporter keeps loops) but the generic check emits it", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge(1, 1);
        const s = b.freeze();
        for (const format of ["gexf", "graphml", "gml", "dot", "pajek", "csv", "json", "neo4j"]) {
            expect(noteCodes(checkExport(s, format))).not.toContain(LOSS.SELF_LOOPS);
        }
        expect(noteCodes(checkCapabilities(s, NO_CAPABILITIES, resolveExportOptions(undefined)))).toContain(
            LOSS.SELF_LOOPS,
        );
        // the cases above plus this one cover the whole table
        const covered = new Set([...CASES.map((c) => c[0]), LOSS.SELF_LOOPS]);
        expect([...Object.values(LOSS)].filter((code) => !covered.has(code))).toEqual([]);
        expect(Object.keys(LOSS)).toHaveLength(40);
    });
});

describe("design 8.5 / Q26: sanitizeIds and onMixedDirection on export", () => {
    it('sanitizeIds defaults to "error": an exporter never renames a node silently', async () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a b", "c");
        const s = b.freeze();
        expect(noteCodes(checkExport(s, "graphml"))).toContain(LOSS.ID_CHARSET);
        await expect(exportGraphToString(s, "graphml")).rejects.toMatchObject({
            code: "E_INVALID_ID",
            details: { reason: "charset", charset: "nmtoken", id: "a b" },
        });
        await expect(exportGraphToString(s, "graphml", { sanitizeIds: "bogus" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it('"mangle" writes graphty:originalId, check() reports it, restoreMangledIds (default true) reads it back', async () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a b", "c");
        const s = b.freeze();
        expect(noteCodes(checkExport(s, "graphml", { sanitizeIds: "mangle" }))).toContain(LOSS.ID_MANGLED);
        const text = await exportGraphToString(s, "graphml", { sanitizeIds: "mangle" });
        expect(text).toContain('attr.name="graphty:originalId"');
        expect(text).toContain('<node id="a_b">');
        const restored = await importGraph(text, { format: "graphml" });
        expect(restored.snapshot.ids.toArray()).toEqual(["a b", "c"]);
        expect(restored.snapshot.nodes.get("graphty:originalId")).toBeNull();
        const kept = await importGraph(text, { format: "graphml", restoreMangledIds: false });
        expect(kept.snapshot.ids.toArray()).toEqual(["a_b", "c"]);
    });

    it("Pajek renumbers 1..N with W_ID_RENUMBERED and never throws (a renumbering, not a mangling)", async () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        const s = b.freeze();
        expect(noteCodes(checkExport(s, "pajek"))).toContain(LOSS.ID_RENUMBERED);
        const text = await exportGraphToString(s, "pajek");
        expect(text).toMatch(/^1 "?a"?$/m);
        const back = await importGraph(text, { format: "pajek" });
        expect(back.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(back.snapshot.nodes.byRole("label")?.value(0)).toBe("a");
        const byLabel = await importGraph(text, { format: "pajek", nodeIdFrom: "label" });
        expect(byLabel.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(pajekExporter.capabilities.idCharset).toBe("dense-1-based");
    });

    it('onMixedDirection defaults to "error": check() reports E_MIXED_DIRECTION and export() throws E_DIRECTED', async () => {
        const s = mixedSnapshot();
        for (const format of ["gml", "dot", "neo4j", "json"]) {
            expect(noteCodes(checkExport(s, format)), format).toContain(LOSS.MIXED_DIRECTION_ERROR);
            await expect(exportGraphToString(s, format), format).rejects.toMatchObject({ code: "E_DIRECTED" });
        }
        await expect(exportGraphToString(s, "gml", { onMixedDirection: "expand" as never })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
        // formats with mixed direction: no note, and the pairs come back
        for (const format of ["gexf", "graphml", "csv", "pajek"]) {
            expect(noteCodes(checkExport(s, format)), format).not.toContain(LOSS.MIXED_DIRECTION_ERROR);
            const back = await importGraph(await exportGraphToString(s, format), { format });
            expect(back.snapshot.edgeCount, format).toBe(4);
            expect(back.snapshot.edges.byRole("pair"), format).not.toBeNull();
        }
    });

    it('"undirected" folds the pairs and writes an undirected graph with W_MIXED_DIRECTION', async () => {
        const s = mixedSnapshot();
        for (const format of ["gml", "dot", "json"]) {
            expect(noteCodes(checkExport(s, format, { onMixedDirection: "undirected" })), format).toContain(
                LOSS.MIXED_DIRECTION,
            );
            const back = await importGraph(await exportGraphToString(s, format, { onMixedDirection: "undirected" }), {
                format,
            });
            expect(back.snapshot.directed, format).toBe(false);
            expect(back.snapshot.edgeCount, format).toBe(3);
        }
    });

    it('FINDING F2: "directed" writes a different edge set per format (GML / JSON keep both halves, DOT / Neo4j drop the mirror)', async () => {
        const s = mixedSnapshot();
        const counts: Record<string, number> = {};
        for (const format of ["gml", "json", "dot", "neo4j"]) {
            expect(noteCodes(checkExport(s, format, { onMixedDirection: "directed" })), format).toContain(
                LOSS.MIXED_DIRECTION,
            );
            const back = await importGraph(await exportGraphToString(s, format, { onMixedDirection: "directed" }), {
                format,
            });
            expect(back.snapshot.directed, format).toBe(true);
            counts[format] = back.snapshot.edgeCount;
        }
        // one semantics for one option: every exporter must agree on the written edge set
        expect(new Set(Object.values(counts)).size, JSON.stringify(counts)).toBe(1);
    });

    it('FINDING F3: Neo4j refuses onMixedDirection "undirected" (a 12.4 value) with E_UNSUPPORTED from check()', () => {
        const s = mixedSnapshot();
        expect(() => checkExport(s, "neo4j", { onMixedDirection: "undirected" })).not.toThrow();
    });
});

// ============================================================ 5.1 text grammar, nodeIdFrom, children

describe("design 5.1 / 8.4: untyped cells, option reporting, the children helper", () => {
    it('FINDING F6: the 5.1 lexical grammar says "2.0" is f64 text, but CSV / DOT / Pajek columns of such cells come back i32', async () => {
        const cases: readonly [format: string, text: string, table: "node" | "edge"][] = [
            ["csv", "source,target,x\n1,2,2.0\n2,3,3.0\n", "edge"],
            ["dot", 'digraph { 1 -> 2 [x="2.0"]; 2 -> 3 [x="3.0"]; }', "edge"],
            ["pajek", '*Vertices 2\n1 "a" x 2.0\n2 "b" x 3.0\n*Arcs\n1 2\n', "node"],
        ];
        const dtypes: Record<string, string | undefined> = {};
        for (const [format, text, table] of cases) {
            const { snapshot } = await importGraph(text, { format });
            const column = table === "edge" ? snapshot.edges.get("x") : snapshot.nodes.get("x");
            dtypes[format] = column?.dtype;
        }
        // GML declares its reals and keeps f64; the untyped text formats must follow the same grammar (I15)
        const gml = await importGraph(
            "graph [ directed 1 node [ id 1 x 2.0 ] node [ id 2 x 3.0 ] edge [ source 1 target 2 ] ]",
            { format: "gml" },
        );
        expect(gml.snapshot.nodes.get("x")?.dtype).toBe("f64");
        expect(dtypes, JSON.stringify(dtypes)).toEqual({ csv: "f64", dot: "f64", pajek: "f64" });
    });

    it("FINDING F7: nodeIdFrom is honoured, ignored silently, warned about (two categories) or refused depending on the format", async () => {
        const docs: readonly [format: string, text: string][] = [
            [
                "gexf",
                '<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3" version="1.3"><graph><nodes><node id="a" label="A"/></nodes></graph></gexf>',
            ],
            ["graphml", graphml("directed", '<node id="a"/>')],
            ["dot", "digraph { a [label=A]; }"],
            ["csv", "source,target\na,b\n"],
            ["json", '{"nodes":[{"id":"a"}],"links":[]}'],
            ["neo4j", ":ID\na\n"],
        ];
        const behaviour: Record<string, string> = {};
        for (const [format, text] of docs) {
            try {
                const { snapshot, report } = await importGraph(text, { format, nodeIdFrom: "label" });
                const issue = report.issues.find((i) => i.element === "nodeIdFrom" || i.code === "W_OPTION_IGNORED");
                if (snapshot.ids.idOf(0) === "A") {
                    behaviour[format] = "honoured";
                } else {
                    behaviour[format] = issue === undefined ? "ignored silently" : `reported as ${issue.category}`;
                }
            } catch (err) {
                behaviour[format] = `refused with ${err instanceof GraphFormatError ? err.code : "?"}`;
            }
        }
        // 8.4: "the importer reports every option it could not honour"; one behaviour, one category
        const distinct = new Set(Object.values(behaviour).filter((b) => b !== "honoured"));
        expect(distinct.size, JSON.stringify(behaviour)).toBe(1);
        expect([...distinct][0], JSON.stringify(behaviour)).toMatch(/^reported as/);
    });

    it("childrenCsr() inverts a parent column into roots / children / depth-first order (7.1)", () => {
        const b = new GraphBuilder({ directed: true });
        for (const id of ["root", "c1", "c2", "g1"]) {
            b.addNode(id);
        }
        b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        b.setNodeValue("parent", 1, 0);
        b.setNodeValue("parent", 2, 0);
        b.setNodeValue("parent", 3, 1);
        const csr = childrenCsr(b.freeze());
        expect(Array.from(csr.roots)).toEqual([0]);
        expect(Array.from(csr.childrenOf(0))).toEqual([1, 2]);
        expect(Array.from(csr.childrenOf(1))).toEqual([3]);
        expect(Array.from(csr.depthFirst().order)).toEqual([0, 1, 3, 2]);
        expect(csr.unreachable).toBe(0);
    });
});

// ============================================================ misc contract points

describe("design 8.4 / 8.5: plugin objects", () => {
    it("exporter.export() yields Uint8Array chunks whose text equals exportToString()", async () => {
        const s = mixedSnapshot();
        const chunks: Uint8Array[] = [];
        for await (const chunk of csvExporter.export(s)) {
            expect(chunk).toBeInstanceOf(Uint8Array);
            chunks.push(chunk);
        }
        const joined = new TextDecoder().decode(new Uint8Array(chunks.flatMap((c) => [...c])));
        expect(joined).toBe(await csvExporter.exportToString(s));
    });

    it("exporter.check() returns loss notes with the 12.4 fields and never writes", () => {
        const notes = graphmlExporter.check(mixedSnapshot());
        expect(Array.isArray(notes)).toBe(true);
        for (const note of notes) {
            expect(Object.keys(note).sort()).toEqual(["code", "column", "count", "message"]);
        }
    });

    it("importer.sniff(head) is a confidence in 0..1", () => {
        const head = new TextEncoder().encode('<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3">');
        const score = gexfImporter.sniff?.(head) ?? -1;
        expect(score).toBeGreaterThan(0);
        expect(score).toBeLessThanOrEqual(1);
        expect(csvImporter.sniff?.(head) ?? -1).toBeLessThan(score);
    });
});
