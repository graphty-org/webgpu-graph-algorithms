/**
 * Fuzz audit, malformed-corpus lens (design sections 8.4 and 8.6): every file under
 * test/corpus/malformed/<format> must make its importer throw ImportError with a populated report
 * whose issues carry a category and whose counts describe the partial result left in the sink.
 *
 * Under errorLimit 0 the first error aborts, so a genuinely malformed file must throw; the files
 * that are in fact well-formed for their format (Graphviz accepts `A B;` and a bare identifier,
 * a `;`-delimited CSV is sniffed, a header-only CSV or a vertices-only Pajek network is an empty
 * graph) are listed in ACCEPTED with the reason and asserted to import with no error. Two of them
 * are corpus defects rather than importer defects and are pinned as such below.
 *
 * The report-versus-sink checks at the end pin an inconsistency: three importers count only the
 * nodes the file declares while the sink also holds the nodes their edges created, so the report
 * of the partial result disagrees with the sink (and with the CSV and Pajek importers).
 */

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { registry } from "../../src/registry.js";
import { type CommonImportOptions, ImportError, type ImportReport, type IssueCategory } from "../../src/types.js";
import { CORPUS_FORMATS, type CorpusFormat, malformedFiles, readMalformedBytes } from "../helpers/corpus.js";

const CATEGORIES: ReadonlySet<IssueCategory> = new Set<IssueCategory>([
    "parse-error",
    "missing-value",
    "validation-error",
    "unsupported",
    "precision",
    "coercion",
    "merged",
]);

/** Malformed-corpus files that are well-formed for their format, with the reason they import. */
const ACCEPTED: Readonly<Record<string, string>> = {
    "csv/header-only.csv": "a header row and no records is an empty edge table (W_CSV_NO_DATA_ROWS)",
    "csv/wrong-delimiter.csv": "a `;`-delimited edge list; the delimiter sniff reads it",
    "dot/invalid-keyword.gv": "Graphviz reads a bare identifier as a node and `a = b` as a graph attribute",
    "dot/missing-arrow.gv": "Graphviz reads `A B;` as two node statements",
    "graphml/invalid-edge-reference.graphml": "undeclared endpoints are created under addMissingNodes (the default)",
    "pajek/missing-edges-section.net": "a vertices-only network is legal (W_PAJEK_NO_LINES)",
};

interface Attempt {
    readonly error: ImportError | null;
    readonly report: ImportReport;
    readonly sink: GraphBuilder;
}

async function attempt(format: CorpusFormat, name: string, options: CommonImportOptions): Promise<Attempt> {
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        const report = await registry.importer(format).import(readMalformedBytes(format, name), sink, options);
        return { error: null, report, sink };
    } catch (err) {
        if (err instanceof ImportError) {
            return { error: err, report: err.report, sink };
        }
        throw err;
    }
}

function expectPopulated(report: ImportReport, format: string): void {
    expect(report.format).toBe(format);
    expect(report.issues.length).toBeGreaterThan(0);
    expect(report.errorCount + report.warningCount).toBe(report.issues.length);
    expect(report.errorCount).toBe(report.issues.filter((i) => i.severity === "error").length);
    for (const issue of report.issues) {
        expect(CATEGORIES.has(issue.category), `category ${issue.category}`).toBe(true);
        expect(issue.code).toMatch(/^[EW]_[A-Z0-9_]+$/);
        expect(issue.message.length).toBeGreaterThan(0);
        // plain ASCII messages (design section 8.6)
        expect(issue.message).toMatch(/^[\x20-\x7e]*$/);
        expect(issue.line === null || (Number.isInteger(issue.line) && issue.line >= 1)).toBe(true);
    }
    // the partial result is described: every counter is a non-negative integer
    for (const [key, value] of Object.entries(report.counts)) {
        expect(Number.isInteger(value) && value >= 0, `counts.${key}`).toBe(true);
    }
    expect(Number.isFinite(report.durationMs) && report.durationMs >= 0).toBe(true);
    expect(Object.isFrozen(report)).toBe(true);
    expect(Object.isFrozen(report.issues)).toBe(true);
}

describe("fuzz audit: every malformed corpus file", () => {
    for (const format of CORPUS_FORMATS) {
        const names = malformedFiles(format);

        it(`${format}: has malformed cases`, () => {
            expect(names.length).toBeGreaterThan(0);
        });

        for (const name of names) {
            const key = `${format}/${name}`;
            const accepted = ACCEPTED[key];

            if (accepted === undefined) {
                it(`${key}: throws ImportError with a populated report under errorLimit 0`, async () => {
                    const { error, report } = await attempt(format, name, { errorLimit: 0 });
                    expect(error).toBeInstanceOf(ImportError);
                    expect(error?.code).toBe("E_IMPORT");
                    expect(error?.name).toBe("ImportError");
                    expect(error?.message.length).toBeGreaterThan(0);
                    expectPopulated(report, format);
                    expect(report.errorCount).toBeGreaterThanOrEqual(1);
                    expect(report.truncated).toBe(true);
                    // the error that aborted is the last issue and is an error
                    const last = report.issues[report.issues.length - 1];
                    expect(last.severity).toBe("error");
                    // the ImportError names the code that aborted
                    expect(typeof error?.details.code).toBe("string");
                });

                it(`${key}: under the default error limit either throws ImportError or reports at least one error`, async () => {
                    const { error, report } = await attempt(format, name, {});
                    if (error !== null) {
                        expectPopulated(report, format);
                    } else {
                        expect(
                            report.errorCount,
                            `recovered without recording an error: ${key}`,
                        ).toBeGreaterThanOrEqual(1);
                    }
                });
            } else {
                it(`${key}: is well-formed for its format (${accepted}) and imports without an error`, async () => {
                    const { error, report } = await attempt(format, name, { errorLimit: 0 });
                    expect(error).toBeNull();
                    expect(report.errorCount).toBe(0);
                });
            }

            it(`${key}: the report counts agree with what is left in the sink`, async () => {
                const { report, sink } = await attempt(format, name, { errorLimit: 0 });
                expect(report.counts.edges).toBe(sink.edgeCount);
                expect(report.counts.nodes, `counts.nodes ${report.counts.nodes} vs sink ${sink.nodeCount}`).toBe(
                    sink.nodeCount,
                );
            });
        }
    }
});

describe("fuzz audit: corpus fixtures exercise what their name says", () => {
    it("csv/binary-content.csv holds non-ASCII bytes, so it tests the invalid-UTF-8 path", () => {
        // The legacy copy was the 14-byte text "Source,Target\n" (no binary byte at all); the
        // fixture now carries an invalid UTF-8 sequence like the Neo4j corpus' binary-content.csv,
        // so the CSV importer's E_INVALID_UTF8 path is corpus-tested (test/formats/csv).
        const bytes = readMalformedBytes("csv", "binary-content.csv");
        let binary = 0;
        for (const b of bytes) {
            if (b >= 0x80 || (b < 0x20 && b !== 0x0a && b !== 0x0d && b !== 0x09)) {
                binary++;
            }
        }
        expect(binary, "non-text bytes in csv/binary-content.csv").toBeGreaterThan(0);
    });
});

describe("fuzz audit: report.counts.nodes versus the sink when edges create nodes", () => {
    // The same graph in five formats: two declared nodes, two edges whose far endpoints are never
    // declared. Every importer leaves four nodes in the sink under addMissingNodes (the default);
    // the report must say so the same way whatever the format.
    const documents: Readonly<Record<string, string>> = {
        csv: "source,target\na,x\nb,y\n",
        json: '{"directed":true,"nodes":[{"id":"a"},{"id":"b"}],"links":[{"source":"a","target":"x"},{"source":"b","target":"y"}]}',
        graphml:
            '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph id="G" edgedefault="directed">' +
            '<node id="a"/><node id="b"/><edge source="a" target="x"/><edge source="b" target="y"/></graph></graphml>',
        gml: "graph [ directed 1 node [ id 1 ] node [ id 2 ] edge [ source 1 target 3 ] edge [ source 2 target 4 ] ]",
        dot: "digraph { a; b; a -> x; b -> y; }",
    };

    for (const [format, text] of Object.entries(documents)) {
        it(`${format}: counts.nodes equals the sink's node count (four)`, async () => {
            const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const report = await registry.importer(format).import(text, sink, { addMissingNodes: true });
            expect(sink.nodeCount).toBe(4);
            expect(report.counts.edges).toBe(2);
            // FAILS for json, graphml and gml: they report 2 (declared nodes only); csv and dot
            // report 4. A consumer reading the report cannot tell what the sink holds.
            expect(report.counts.nodes, `${format} report ${report.counts.nodes} vs sink ${sink.nodeCount}`).toBe(4);
        });
    }
});
