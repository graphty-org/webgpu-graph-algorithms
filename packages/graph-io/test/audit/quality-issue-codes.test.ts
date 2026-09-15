/**
 * Code-quality audit of the issue / loss code tables of the eight formats (design section 8.6:
 * "a stable code such as E_UNKNOWN_NODE"; the CLAUDE.md principle that nothing is silently
 * dropped implies every recorded code is discoverable through its format's table).
 *
 * Three tests here fail deliberately; each pins a finding of the audit report:
 * - "every subpath exports <FMT>_ISSUE and <FMT>_LOSS": DOT exports DOT_ISSUE /
 *   DOT_LOSS and Pajek exports PAJEK_ISSUE, so a consumer cannot look a code table up by
 *   format name;
 * - "a concept shared by several formats has one code": the same condition is spelled
 *   E_MISSING_ID (GEXF, GraphML, JSON, Neo4j), E_GML_MISSING_ID (GML) and E_CSV_MISSING_ID (CSV);
 *   W_DUPLICATE_NODE / E_GML_DUPLICATE_NODE / W_CSV_DUPLICATE_NODE / W_PAJEK_DUPLICATE_VERTEX; and
 *   so on, so a UI cannot switch on a condition across formats;
 * - "every code an importer records is a member of its table": the GML tokenizer's four fatal
 *   codes, GEXF's E_XML_SYNTAX and the Neo4j reader's E_CSV_UNCLOSED_QUOTE / W_CSV_TEXT_AFTER_QUOTE
 *   reach the report but are missing from GML_ISSUE / GEXF_ISSUE / NEO4J_ISSUE, whose JSDoc says
 *   "the issue codes the importer records ... by name";
 * - "a code shared by several modules is one constant": E_MISSING_ID, E_MISSING_ENDPOINT,
 *   W_DUPLICATE_NODE, E_NO_GRAPH, E_XML_SYNTAX, W_OPTION_IGNORED, W_MULTIPLE_GRAPHS,
 *   E_UNKNOWN_PARENT, E_HYPEREDGE, W_ROLE_DROPPED, W_PARENTS_DROPPED are spelled as independent
 *   string literals in two to four modules (and W_PRECISION / W_TEMPORAL_DROPPED once in
 *   src/common and once more in a format), so they agree by coincidence only; W_ID_MERGED and
 *   W_SINK_OPTION show the intended shape (one constant in src/common, aliased by every table).
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import * as csv from "../../src/formats/csv/index.js";
import * as dot from "../../src/formats/dot/index.js";
import * as gexf from "../../src/formats/gexf/index.js";
import * as gml from "../../src/formats/gml/index.js";
import * as graphml from "../../src/formats/graphml/index.js";
import * as json from "../../src/formats/json/index.js";
import * as neo4j from "../../src/formats/neo4j/index.js";
import * as pajek from "../../src/formats/pajek/index.js";
import { type GraphImporter, ImportError, type ImportReport } from "../../src/types.js";

type CodeTable = Readonly<Record<string, string>>;

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "src");

function walk(dir: string, out: string[] = []): string[] {
    for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) {
            walk(path, out);
        } else if (path.endsWith(".ts")) {
            out.push(path);
        }
    }
    return out;
}

/** The codes the core defines (GraphFormatError codes an importer merely forwards). */
const CORE_CODES: ReadonlySet<string> = new Set([
    "E_UNSUPPORTED",
    "E_COLUMN_TYPE",
    "E_INVALID_ID",
    "E_COLUMN_EXISTS",
    "E_DIRECTED",
    "E_UNKNOWN_NODE",
    "E_DUPLICATE_ROLE",
    "E_INVALID_WEIGHT",
    "E_IMPORT",
    "E_DUPLICATE_EDGE_ID",
    "E_UNKNOWN_COLUMN",
    "E_TOO_LARGE",
    "E_GPU_INELIGIBLE",
]);

const SUBPATHS: Readonly<Record<string, Record<string, unknown>>> = {
    GEXF: gexf,
    GRAPHML: graphml,
    GML: gml,
    DOT: dot,
    PAJEK: pajek,
    CSV: csv,
    JSON: json,
    NEO4J: neo4j,
};

function isCodeTable(value: unknown): value is CodeTable {
    return (
        typeof value === "object" &&
        value !== null &&
        !Array.isArray(value) &&
        Object.values(value).length > 0 &&
        Object.values(value).every((v) => typeof v === "string" && /^[EW]_[A-Z0-9_]+$/.test(v))
    );
}

function tablesOf(subpath: Record<string, unknown>): Record<string, CodeTable> {
    const out: Record<string, CodeTable> = {};
    for (const [name, value] of Object.entries(subpath)) {
        if (/^[A-Z0-9_]+$/.test(name) && isCodeTable(value)) {
            out[name] = value;
        }
    }
    return out;
}

async function reportOf(importer: GraphImporter, text: string): Promise<ImportReport> {
    const builder = new GraphBuilder({ directed: true });
    try {
        return await importer.import(text, builder);
    } catch (err) {
        if (err instanceof ImportError) {
            return err.report;
        }
        throw err;
    }
}

describe("audit: issue and loss code tables", () => {
    it("every subpath exports <FMT>_ISSUE and <FMT>_LOSS", () => {
        const missing: string[] = [];
        for (const [fmt, subpath] of Object.entries(SUBPATHS)) {
            const names = Object.keys(tablesOf(subpath)).sort();
            for (const wanted of [`${fmt}_ISSUE`, `${fmt}_LOSS`]) {
                if (!names.includes(wanted)) {
                    missing.push(`${fmt}: has ${names.join(", ")}; no ${wanted}`);
                }
            }
        }
        expect(missing).toEqual([]);
    });

    it("a table key is the code without its E_ / W_ and format prefix", () => {
        const mismatched: string[] = [];
        for (const [fmt, subpath] of Object.entries(SUBPATHS)) {
            for (const [tableName, table] of Object.entries(tablesOf(subpath))) {
                for (const [key, code] of Object.entries(table)) {
                    const suffix = code.replace(/^[EW]_/, "").replace(new RegExp(`^${fmt}_`), "");
                    if (suffix !== key) {
                        mismatched.push(`${tableName}.${key} = ${code}`);
                    }
                }
            }
        }
        expect(mismatched).toEqual([]);
    });

    it("a concept shared by several formats has one code", () => {
        // the same key in the ISSUE tables of two formats must name the same code (a shared
        // vocabulary in src/common, as W_ID_MERGED / W_SINK_OPTION / W_COLUMN_RENAMED already are)
        const byKey = new Map<string, Map<string, string>>();
        for (const [fmt, subpath] of Object.entries(SUBPATHS)) {
            for (const [tableName, table] of Object.entries(tablesOf(subpath))) {
                if (!/ISSUE|IMPORT_CODES/.test(tableName)) {
                    continue;
                }
                for (const [key, code] of Object.entries(table)) {
                    const codes = byKey.get(key) ?? new Map<string, string>();
                    codes.set(fmt, code);
                    byKey.set(key, codes);
                }
            }
        }
        const divergent: string[] = [];
        for (const [key, codes] of byKey) {
            if (codes.size > 1 && new Set(codes.values()).size > 1) {
                divergent.push(`${key}: ${[...codes].map(([fmt, code]) => `${fmt}=${code}`).join(" ")}`);
            }
        }
        expect(divergent.sort()).toEqual([]);
    });

    it("every code an importer records is a member of its table", async () => {
        const cases: readonly (readonly [string, GraphImporter, CodeTable, string])[] = [
            ["gml unclosed bracket", gml.gmlImporter, gml.GML_ISSUE, "graph [ node [ id 1 "],
            ["gml unclosed string", gml.gmlImporter, gml.GML_ISSUE, 'graph [ node [ id 1 label "x ] ]'],
            ["gml bad token", gml.gmlImporter, gml.GML_ISSUE, "graph [ node [ id 1 ] @ ]"],
            ["gml structure", gml.gmlImporter, gml.GML_ISSUE, "graph [ node [ id 1 ] ] ]"],
            ["gexf malformed xml", gexf.gexfImporter, gexf.GEXF_ISSUE, "<gexf><graph>"],
            ["neo4j unclosed quote", neo4j.neo4jImporter, neo4j.NEO4J_ISSUE, ':ID,name\n1,"abc'],
            ["neo4j text after quote", neo4j.neo4jImporter, neo4j.NEO4J_ISSUE, ':ID,name\n1,"abc"def\n'],
        ];
        const unlisted: string[] = [];
        for (const [name, importer, table, text] of cases) {
            const report = await reportOf(importer, text);
            expect(report.issues.length, name).toBeGreaterThan(0);
            const listed = new Set(Object.values(table));
            for (const issue of report.issues) {
                if (!listed.has(issue.code)) {
                    unlisted.push(`${name}: ${issue.code}`);
                }
            }
        }
        expect(unlisted).toEqual([]);
    });

    it("a code shared by several modules is one constant", () => {
        // a code literal written in two modules is two definitions that happen to agree
        const definedIn = new Map<string, Set<string>>();
        for (const file of walk(SRC)) {
            const text = readFileSync(file, "utf8");
            for (const match of text.matchAll(/(?:=|:)\s*"([EW]_[A-Z0-9_]+)"/g)) {
                const code = match[1];
                if (CORE_CODES.has(code)) {
                    continue;
                }
                const files = definedIn.get(code) ?? new Set<string>();
                files.add(relative(SRC, file));
                definedIn.set(code, files);
            }
        }
        const duplicated: string[] = [];
        for (const [code, files] of definedIn) {
            if (files.size > 1) {
                duplicated.push(`${code}: ${[...files].sort().join(", ")}`);
            }
        }
        expect(duplicated.sort()).toEqual([]);
    });
});
