/**
 * Code-quality audit of silent drops (the graph-io CLAUDE.md principle: "Nothing is silently
 * dropped: an unsupported construct is an ImportIssue (report) or a LossNote (check()), never a
 * fallback"; design section 5.6: a colliding declared attribute is renamed `<name>#<origin.id>`
 * "and records a coercion issue").
 *
 * The tests here fail deliberately; each pins a finding of the audit report:
 * - GraphML `declareTarget()` (src/formats/graphml/importer.ts) catches E_DUPLICATE_ROLE and
 *   re-declares the column without its role with no issue at all, so a key titled `label`
 *   silently loses the label role when the caller's sink (or an earlier key) already holds it;
 *   DOT, GML, CSV and Neo4j report the same condition (W_DOT_ROLE_TAKEN, W_GML_ROLE_TAKEN,
 *   W_CSV_ROLE_TAKEN, W_ROLE_TAKEN);
 * - the GEXF importer ignores elements it does not know (a `<foo/>` under `<graph>`, `<node>` or
 *   `<edge>`, an unknown attribute on `<edge>`) with no issue, while the GraphML importer records
 *   W_UNKNOWN_ELEMENT for the same shape;
 * - a common option a format has no use for (`hyperedges`, `long`, `restoreMangledIds`,
 *   `nodeIdFrom` on an edge-list CSV) is ignored without an issue by CSV, GML, Pajek and DOT
 *   (design section 8.4: "the importer reports every option it could not honour"); DOT, GraphML,
 *   JSON and Neo4j report W_OPTION_IGNORED for `nodeIdFrom` only and GEXF throws E_UNSUPPORTED
 *   for it, so one condition has three behaviours.
 */

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { gmlImporter } from "../../src/formats/gml/importer.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import { type CommonImportOptions, type GraphImporter } from "../../src/types.js";

const GRAPHML_LABEL_KEY =
    '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">' +
    '<key id="l" for="node" attr.name="label" attr.type="string"/>' +
    '<graph id="G" edgedefault="directed"><node id="a"><data key="l">A</data></node></graph></graphml>';

const GRAPHML_TWO_LABEL_KEYS =
    '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">' +
    '<key id="l1" for="node" attr.name="label" attr.type="string"/>' +
    '<key id="l2" for="node" attr.name="label" attr.type="string"/>' +
    '<graph id="G" edgedefault="directed"><node id="a"><data key="l1">A</data><data key="l2">B</data></node></graph></graphml>';

const GEXF_UNKNOWN_ELEMENTS =
    '<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3" version="1.3"><graph defaultedgetype="directed">' +
    '<foo/><nodes><node id="1"><foo/><bar x="1"/></node></nodes>' +
    '<edges><edge id="0" source="1" target="1" foo="bar"><baz/></edge></edges></graph></gexf>';

describe("audit: silent drops", () => {
    it("graphml: a label key whose role the sink already holds is reported, not silently demoted", async () => {
        const builder = new GraphBuilder({ directed: true });
        builder.declareNodeColumn({ name: "name", dtype: "string", nullable: true, role: "label" });
        const report = await graphmlImporter.import(GRAPHML_LABEL_KEY, builder);
        const snapshot = builder.freeze();
        // the role was dropped ...
        expect(snapshot.nodes.get("label")?.meta.role).toBeNull();
        expect(snapshot.nodes.byRole("label")?.meta.name).toBe("name");
        // ... and nothing says so
        expect(report.issues.map((i) => `${i.severity}:${i.code}`)).not.toEqual([]);
    });

    it("graphml: the second of two keys titled label loses the role with only a rename issue", async () => {
        const builder = new GraphBuilder({ directed: true });
        const report = await graphmlImporter.import(GRAPHML_TWO_LABEL_KEYS, builder);
        const snapshot = builder.freeze();
        expect(snapshot.nodes.names()).toEqual(["label", "label#l2"]);
        expect(snapshot.nodes.get("label#l2")?.meta.role).toBeNull();
        const codes = report.issues.map((i) => i.code);
        expect(codes).toContain("W_COLUMN_RENAMED");
        // a second issue must name the dropped role (the rename issue is about the NAME)
        expect(codes.length).toBeGreaterThan(1);
    });

    it("gexf: unknown elements and attributes are reported the way GraphML reports them", async () => {
        const gexfBuilder = new GraphBuilder({ directed: true });
        const gexfReport = await gexfImporter.import(GEXF_UNKNOWN_ELEMENTS, gexfBuilder);
        const graphmlBuilder = new GraphBuilder({ directed: true });
        const graphmlReport = await graphmlImporter.import(
            '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph id="G" edgedefault="directed"><node id="a"><foo/></node></graph></graphml>',
            graphmlBuilder,
        );
        expect(graphmlReport.issues.map((i) => i.code)).toEqual(["W_UNKNOWN_ELEMENT"]);
        expect(gexfReport.issues.map((i) => i.code)).not.toEqual([]);
    });

    it("a common option the format cannot honour is reported, never silently ignored", async () => {
        const unusable: CommonImportOptions = { hyperedges: "error", restoreMangledIds: false, long: "string" };
        const cases: readonly (readonly [string, GraphImporter, string, CommonImportOptions])[] = [
            ["csv", csvImporter, "source,target\na,b\n", { ...unusable, nodeIdFrom: "label" }],
            ["gml", gmlImporter, 'graph [ node [ id 1 label "A" ] ]', unusable],
            ["pajek", pajekImporter, '*Vertices 1\n1 "A"\n*Arcs\n', unusable],
            ["dot", dotImporter, "digraph { a }", unusable],
        ];
        const silent: string[] = [];
        for (const [format, importer, text, options] of cases) {
            const builder = new GraphBuilder({ directed: true });
            const report = await importer.import(text, builder, options);
            const reported = report.issues.filter((i) => i.element !== null && i.element in options);
            if (reported.length === 0) {
                silent.push(
                    `${format}: ${Object.keys(options).join(", ")} ignored; issues ${report.issues.map((i) => i.code).join(",") || "none"}`,
                );
            }
        }
        expect(silent).toEqual([]);
    });
});
