import { GraphBuilder } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { JSON_ISSUE, jsonImporter } from "../../../src/formats/json/index.js";
import { type CommonImportOptions, ImportError } from "../../../src/types.js";
import { malformedFiles, readMalformedBytes } from "../../helpers/corpus.js";

/** What each malformed case does under the default options: fatal at once, or recovered with error issues. */
const CASES: Record<string, { fatal: string } | { recovered: string; nodes: number; edges: number }> = {
    "edges-not-array.json": { fatal: JSON_ISSUE.SHAPE },
    "empty-file.json": { fatal: JSON_ISSUE.EMPTY_INPUT },
    "invalid-json.json": { fatal: JSON_ISSUE.SYNTAX },
    "missing-edges.json": { recovered: JSON_ISSUE.MISSING_SECTION, nodes: 3, edges: 0 },
    "missing-edge-source.json": { recovered: JSON_ISSUE.MISSING_ENDPOINT, nodes: 2, edges: 1 },
    "missing-node-id.json": { recovered: JSON_ISSUE.MISSING_ID, nodes: 2, edges: 1 },
    "missing-nodes.json": { recovered: JSON_ISSUE.MISSING_SECTION, nodes: 3, edges: 2 },
    "nodes-not-array.json": { fatal: JSON_ISSUE.SHAPE },
    "not-json.json": { fatal: JSON_ISSUE.SYNTAX },
    "wrong-structure.json": { fatal: JSON_ISSUE.DIALECT },
};

async function attempt(
    name: string,
    options?: CommonImportOptions,
): Promise<{ error: ImportError | null; b: GraphBuilder }> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        await jsonImporter.import(readMalformedBytes("json", name), b, options);
        return { error: null, b };
    } catch (err) {
        if (err instanceof ImportError) {
            return { error: err, b };
        }
        throw err;
    }
}

describe("malformed corpus", () => {
    it("covers every file under test/corpus/malformed/json", () => {
        expect(malformedFiles("json")).toEqual(Object.keys(CASES).sort());
    });

    for (const name of malformedFiles("json")) {
        const expected = CASES[name];

        it(`${name}: throws ImportError with a report under errorLimit 0`, async () => {
            const { error } = await attempt(name, { errorLimit: 0 });
            expect(error).toBeInstanceOf(ImportError);
            const report = error?.report;
            expect(report?.format).toBe("json");
            expect(report?.errorCount).toBeGreaterThanOrEqual(1);
            expect(report?.issues.some((i) => i.severity === "error")).toBe(true);
            const code = "fatal" in expected ? expected.fatal : expected.recovered;
            expect(report?.issues.map((i) => i.code)).toContain(code);
            if ("recovered" in expected) {
                expect(report?.truncated).toBe(true);
            }
        });

        if ("fatal" in expected) {
            it(`${name}: is fatal under the default options (${expected.fatal})`, async () => {
                const { error } = await attempt(name);
                expect(error).toBeInstanceOf(ImportError);
                expect(error?.report.issues[0]?.code).toBe(expected.fatal);
                expect(error?.report.issues[0]?.category).toBe("parse-error");
                expect(error?.report.issues[0]?.severity).toBe("error");
                expect(error?.report.truncated).toBe(false);
                expect(error?.details.code).toBe(expected.fatal);
            });
        } else {
            it(`${name}: recovers under the default options with ${expected.recovered} recorded`, async () => {
                const { error, b } = await attempt(name);
                expect(error).toBeNull();
                const s = b.freeze();
                expect(s.nodeCount).toBe(expected.nodes);
                expect(s.edgeCount).toBe(expected.edges);
            });
        }
    }

    it("missing-node-id.json skips the node and counts it", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await jsonImporter.import(readMalformedBytes("json", "missing-node-id.json"), b);
        expect(report.counts).toEqual({ nodes: 2, edges: 1, skippedNodes: 1, skippedEdges: 0, expandedMixed: 0 });
        expect(report.issues).toEqual([
            {
                category: "missing-value",
                severity: "error",
                code: JSON_ISSUE.MISSING_ID,
                message: "nodes[1] has no id",
                line: null,
                element: "nodes[1]",
            },
        ]);
        expect(b.freeze().ids.toArray()).toEqual(["1", "3"]);
    });

    it("missing-edge-source.json skips the edge and counts it", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await jsonImporter.import(readMalformedBytes("json", "missing-edge-source.json"), b);
        expect(report.counts).toEqual({ nodes: 2, edges: 1, skippedNodes: 0, skippedEdges: 1, expandedMixed: 0 });
        expect(report.issues[0]).toMatchObject({ code: JSON_ISSUE.MISSING_ENDPOINT, element: "edges[0]" });
    });

    it("the partial report of a fatal error keeps what was counted before the failure", async () => {
        const { error } = await attempt("edges-not-array.json");
        expect(error?.report.counts.nodes).toBe(0);
        expect(error?.message).toContain("edges must be an array");
    });
});
