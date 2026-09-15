/**
 * Code-quality audit of the per-element catch (design section 8.6: "the importer catches
 * GraphFormatError per element, records an ImportIssue, skips the element and continues").
 *
 * `ImportReportBuilder.recordError()` (src/common/report.ts) goes further: ANY thrown value that
 * is not a GraphFormatError, an ImportError or an AbortError is recorded as a "parse-error" issue
 * with code E_PARSE and the import continues. A TypeError or RangeError raised by a bug in the
 * importer or in the sink is therefore reported as a defect of the input file (with the file's
 * line number) and the import resolves successfully, so the bug never surfaces as an exception.
 * Every importer routes its per-element catch through recordError, so the behaviour is shared.
 *
 * The tests here fail deliberately: they pin the finding that a non-GraphFormatError must
 * propagate (or at least must not be attributed to the input as a parse error).
 */

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { ImportReportBuilder, PARSE_ERROR_CODE } from "../../src/common/report.js";
import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { jsonImporter } from "../../src/formats/json/importer.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import { type GraphImporter } from "../../src/types.js";

/** A builder whose value setters throw the way a programming error would. */
function brokenSink(): GraphBuilder {
    const builder = new GraphBuilder({ directed: true });
    const boom = (): never => {
        throw new TypeError("Cannot read properties of undefined (reading 'x')");
    };
    builder.setNodeValue = boom;
    builder.setEdgeValue = boom;
    return builder;
}

const INPUTS: readonly (readonly [string, GraphImporter, string])[] = [
    ["dot", dotImporter, "digraph { a [color=red]; a -> b [w=1]; }"],
    ["csv", csvImporter, "source,target,color\na,b,red\n"],
    ["pajek", pajekImporter, '*Vertices 2\n1 "a" x 1\n2 "b"\n*Arcs\n1 2 1 c red\n'],
    [
        "gexf",
        gexfImporter,
        '<gexf xmlns="http://gexf.net/1.3" version="1.3"><graph defaultedgetype="directed"><nodes><node id="a" label="A"/><node id="b"/></nodes><edges><edge id="0" source="a" target="b" label="e"/></edges></graph></gexf>',
    ],
    [
        "graphml",
        graphmlImporter,
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="c" for="node" attr.name="color" attr.type="string"/><graph id="G" edgedefault="directed"><node id="a"><data key="c">red</data></node><node id="b"/><edge source="a" target="b"/></graph></graphml>',
    ],
    ["json", jsonImporter, '{"nodes":[{"id":"a","color":"red"},{"id":"b"}],"links":[{"source":"a","target":"b"}]}'],
];

describe("audit: recordError converts programming errors into input issues", () => {
    it("recordError() records a TypeError as an E_PARSE parse-error instead of rethrowing it", () => {
        const report = new ImportReportBuilder("test", 100);
        expect(() => report.recordError(new TypeError("bug"), { line: 7 })).toThrow(TypeError);
        expect(report.issues.map((i) => i.code)).not.toContain(PARSE_ERROR_CODE);
    });

    for (const [format, importer, text] of INPUTS) {
        it(`${format}: a TypeError thrown by the sink propagates out of import()`, async () => {
            const sink = brokenSink();
            let report = null;
            let thrown: unknown = null;
            try {
                report = await importer.import(text, sink);
            } catch (err) {
                thrown = err;
            }
            // what happens today: the import resolves and the TypeError is a parse-error of the file
            const parseErrors = report === null ? [] : report.issues.filter((i) => i.code === PARSE_ERROR_CODE);
            expect(
                thrown,
                `import resolved with ${parseErrors.length} E_PARSE issue(s): ${parseErrors.map((i) => `${i.category} line ${i.line}: ${i.message}`).join("; ")}`,
            ).toBeInstanceOf(TypeError);
        });
    }
});
