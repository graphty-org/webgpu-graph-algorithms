/**
 * Code-quality audit of the five private copies of "declare a column on a sink that already holds
 * the name or the role" (DOT `declare()`, CSV `declareRoleColumn()`, GML `declareColumn` plan,
 * Neo4j `declareReserved()`, GraphML `declareTarget()`; GEXF, Pajek and JSON have none) against
 * design section 5.6, an "io rule, so two importers agree": when a declared attribute collides
 * with an existing name in the same table the importer names it `<name>#<origin.id>` and records a
 * coercion issue.
 *
 * One situation, eight outcomes: the caller's sink already holds a node column `label` (i32, no
 * role) and the file labels its node "A". GraphML and Neo4j rename and keep the value (5.6), CSV
 * keeps it under a different name, GEXF and Pajek record E_COLUMN_EXISTS and lose the label, GML
 * and DOT adopt the caller's i32 column and then record E_COLUMN_TYPE for the text (the label is
 * lost), JSON records E_COLUMN_TYPE. The test fails deliberately: it asserts the 5.6 outcome for
 * every format and reports the divergence.
 */

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { gmlImporter } from "../../src/formats/gml/importer.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { jsonImporter } from "../../src/formats/json/importer.js";
import { neo4jImporter } from "../../src/formats/neo4j/importer.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import { type GraphImporter, ImportError } from "../../src/types.js";

interface Case {
    readonly format: string;
    readonly importer: GraphImporter;
    readonly text: string;
    readonly options?: Record<string, unknown>;
}

const CASES: readonly Case[] = [
    {
        format: "gexf",
        importer: gexfImporter,
        text: '<?xml version="1.0"?><gexf xmlns="http://gexf.net/1.3" version="1.3"><graph defaultedgetype="directed"><nodes><node id="a" label="A"/></nodes></graph></gexf>',
    },
    {
        format: "graphml",
        importer: graphmlImporter,
        text: '<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><key id="l" for="node" attr.name="label" attr.type="string"/><graph id="G" edgedefault="directed"><node id="a"><data key="l">A</data></node></graph></graphml>',
    },
    { format: "gml", importer: gmlImporter, text: 'graph [ directed 1 node [ id 1 label "A" ] ]' },
    { format: "dot", importer: dotImporter, text: 'digraph { a [label="A"]; }' },
    { format: "csv", importer: csvImporter, text: "id,label\na,A\n", options: { table: "nodes" } },
    { format: "pajek", importer: pajekImporter, text: '*Vertices 1\n1 "A"\n*Arcs\n' },
    { format: "json", importer: jsonImporter, text: '{"nodes":[{"id":"a","label":"A"}],"links":[]}' },
    { format: "neo4j", importer: neo4jImporter, text: ":ID,label\na,A\n" },
];

interface Outcome {
    readonly format: string;
    readonly issues: readonly string[];
    readonly columns: readonly string[];
    /** The label text of node 0 in whichever string column received it, or null when it was lost. */
    readonly label: string | null;
}

function labelOf(snapshot: GraphSnapshot): string | null {
    for (const name of snapshot.nodes.names()) {
        const column = snapshot.nodes.get(name);
        if (column !== null && column.dtype === "string" && column.isSet(0) && column.value(0) === "A") {
            return name;
        }
    }
    return null;
}

async function run(c: Case): Promise<Outcome> {
    const builder = new GraphBuilder({ directed: true });
    builder.declareNodeColumn({ name: "label", dtype: "i32", nullable: true });
    try {
        const report = await c.importer.import(c.text, builder, c.options);
        const snapshot = builder.freeze();
        return {
            format: c.format,
            issues: report.issues.map((i) => `${i.severity[0]}:${i.code}`),
            columns: snapshot.nodes.names(),
            label: labelOf(snapshot),
        };
    } catch (err) {
        if (err instanceof ImportError) {
            return { format: c.format, issues: err.report.issues.map((i) => i.code), columns: [], label: null };
        }
        throw err;
    }
}

describe("audit: design 5.6 collision rule across the eight importers", () => {
    it("a label colliding with a caller's column is renamed <name>#<id> and kept in every format", async () => {
        const outcomes = await Promise.all(CASES.map(run));
        // JSON keys are inferred attributes, not declarations (design section 5.6 renames declared
        // attributes): a caller-declared column of another dtype refuses the value per element and
        // the importer records E_COLUMN_TYPE for it, which is the design's answer for untyped cells
        const json = outcomes.find((o) => o.format === "json");
        expect(json?.issues).toEqual(["e:E_COLUMN_TYPE"]);
        const declared = outcomes.filter((o) => o.format !== "json");
        const lost = declared.filter((o) => o.label === null);
        const noCoercionIssue = declared.filter((o) => !o.issues.some((i) => i.startsWith("w:")));
        expect(
            lost.map((o) => `${o.format}: label lost, issues ${o.issues.join(",")}, columns ${o.columns.join(",")}`),
        ).toEqual([]);
        expect(noCoercionIssue.map((o) => o.format)).toEqual([]);
    });
});
