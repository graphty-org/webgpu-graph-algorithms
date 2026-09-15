import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { ImportReportBuilder, isAbortError, messageOf, PARSE_ERROR_CODE } from "../../src/common/report.js";
import { ImportError, type ImportReport } from "../../src/types.js";

describe("ImportError (design 12.4)", () => {
    const report: ImportReport = Object.freeze({
        format: "csv",
        counts: { nodes: 1, edges: 0, skippedNodes: 0, skippedEdges: 0, expandedMixed: 0 },
        issues: [],
        errorCount: 0,
        warningCount: 0,
        truncated: false,
        lossy: [],
        durationMs: 0,
    });

    it("is a GraphFormatError with code E_IMPORT carrying the partial report", () => {
        const err = new ImportError("boom", report, { reason: "test" });
        expect(err).toBeInstanceOf(GraphFormatError);
        expect(err).toBeInstanceOf(Error);
        expect(err.code).toBe("E_IMPORT");
        expect(err.name).toBe("ImportError");
        expect(err.message).toBe("boom");
        expect(err.report).toBe(report);
        expect(err.details).toEqual({ reason: "test" });
        expect(Object.isFrozen(err.details)).toBe(true);
    });

    it('narrows on err.code === "E_IMPORT"', () => {
        const err: GraphFormatError = new ImportError("x", report);
        if (err.code === "E_IMPORT") {
            expect((err as ImportError).report.format).toBe("csv");
        } else {
            throw new Error("did not narrow");
        }
    });
});

describe("ImportReportBuilder", () => {
    it("records issues by category and severity and counts them", () => {
        const b = new ImportReportBuilder("gexf", 100);
        b.error("parse-error", "E_BAD", "bad thing", { line: 3, element: "n1" });
        b.warning("coercion", "W_COERCED", "coerced", { line: 4 });
        b.warning("precision", "W_PRECISION", "lost bits");
        expect(b.errorCount).toBe(1);
        expect(b.warningCount).toBe(2);
        expect(b.issues).toHaveLength(3);
        expect(b.issues[0]).toEqual({
            category: "parse-error",
            severity: "error",
            code: "E_BAD",
            message: "bad thing",
            line: 3,
            element: "n1",
        });
        expect(b.issues[1].line).toBe(4);
        expect(b.issues[1].element).toBeNull();
        expect(b.issues[2].line).toBeNull();
        expect(Object.isFrozen(b.issues[0])).toBe(true);
        expect(b.truncated).toBe(false);
    });

    it("accepts every category of design 8.6", () => {
        const b = new ImportReportBuilder("x", Infinity);
        const categories = [
            "parse-error",
            "missing-value",
            "validation-error",
            "unsupported",
            "precision",
            "coercion",
            "merged",
        ] as const;
        for (const c of categories) {
            b.warning(c, `W_${c}`, c);
        }
        expect(b.issues.map((i) => i.category)).toEqual([...categories]);
    });

    it("counts and finishes into a frozen report with durationMs", () => {
        const b = new ImportReportBuilder("gml", 10);
        b.counts.nodes += 3;
        b.counts.edges += 2;
        b.counts.skippedNodes++;
        b.counts.skippedEdges++;
        b.counts.expandedMixed++;
        b.loss("W_X", "lost", "col", 2);
        b.loss("W_Y", "lost too");
        const r = b.finish();
        expect(r.format).toBe("gml");
        expect(r.counts).toEqual({ nodes: 3, edges: 2, skippedNodes: 1, skippedEdges: 1, expandedMixed: 1 });
        expect(r.lossy).toEqual([
            { code: "W_X", message: "lost", column: "col", count: 2 },
            { code: "W_Y", message: "lost too", column: null, count: null },
        ]);
        expect(r.durationMs).toBeGreaterThanOrEqual(0);
        expect(Object.isFrozen(r)).toBe(true);
        expect(Object.isFrozen(r.counts)).toBe(true);
        expect(Object.isFrozen(r.issues)).toBe(true);
        // finish() snapshots: later changes do not leak into an earlier report
        b.counts.nodes++;
        expect(r.counts.nodes).toBe(3);
        expect(b.finish().counts.nodes).toBe(4);
    });

    it("throws ImportError with truncated set when an error goes beyond the limit", () => {
        const b = new ImportReportBuilder("csv", 2);
        b.error("parse-error", "E_1", "one");
        b.error("parse-error", "E_2", "two");
        expect(b.truncated).toBe(false);
        let caught: unknown;
        try {
            b.error("parse-error", "E_3", "three");
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const err = caught as ImportError;
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.truncated).toBe(true);
        expect(err.report.errorCount).toBe(3);
        expect(err.report.issues.map((i) => i.code)).toEqual(["E_1", "E_2", "E_3"]);
        expect(err.details).toMatchObject({ code: "E_3", limit: 2 });
        expect(b.truncated).toBe(true);
    });

    it("aborts on the first error with errorLimit 0 and never with Infinity", () => {
        const strict = new ImportReportBuilder("csv", 0);
        expect(() => strict.error("parse-error", "E_1", "one")).toThrow(ImportError);
        expect(strict.errorCount).toBe(1);
        const lax = new ImportReportBuilder("csv", Infinity);
        for (let i = 0; i < 1000; i++) {
            lax.error("parse-error", "E", "x");
        }
        expect(lax.errorCount).toBe(1000);
        expect(lax.truncated).toBe(false);
    });

    it("never counts warnings toward the limit", () => {
        const b = new ImportReportBuilder("csv", 1);
        for (let i = 0; i < 50; i++) {
            b.warning("coercion", "W", "x");
        }
        expect(b.warningCount).toBe(50);
        expect(b.truncated).toBe(false);
    });

    it("warnOnce records a code once", () => {
        const b = new ImportReportBuilder("csv", 1);
        expect(b.warnOnce("coercion", "W_ONCE", "first")).not.toBeNull();
        expect(b.warnOnce("coercion", "W_ONCE", "second")).toBeNull();
        expect(b.warnOnce("coercion", "W_OTHER", "third")).not.toBeNull();
        expect(b.warningCount).toBe(2);
    });

    describe("recordError", () => {
        it("maps GraphFormatError codes to categories and keeps the code", () => {
            const b = new ImportReportBuilder("csv", Infinity);
            b.recordError(new GraphFormatError("E_UNKNOWN_NODE", "no node"), { line: 1 });
            b.recordError(new GraphFormatError("E_INVALID_WEIGHT", "nan"));
            b.recordError(new GraphFormatError("E_DIRECTED", "locked"));
            b.recordError(new GraphFormatError("E_TOO_LARGE", "big"));
            b.recordError(new GraphFormatError("E_UNSUPPORTED", "nope"));
            b.recordError(new GraphFormatError("E_COLUMN_TYPE", "type"));
            expect(b.issues.map((i) => [i.category, i.code])).toEqual([
                ["missing-value", "E_UNKNOWN_NODE"],
                ["validation-error", "E_INVALID_WEIGHT"],
                ["coercion", "E_DIRECTED"],
                ["unsupported", "E_TOO_LARGE"],
                ["unsupported", "E_UNSUPPORTED"],
                ["validation-error", "E_COLUMN_TYPE"],
            ]);
            expect(b.issues[0].line).toBe(1);
            expect(b.issues.every((i) => i.severity === "error")).toBe(true);
        });

        it("re-throws any value that is not a GraphFormatError (a programming error is never an issue of the file)", () => {
            const b = new ImportReportBuilder("csv", Infinity);
            const bug = new TypeError("xml broke");
            expect(() => b.recordError(bug)).toThrow(bug);
            expect(() => b.recordError("a string")).toThrow("a string");
            expect(() => b.recordError(42)).toThrow();
            expect(b.issues).toEqual([]);
            expect(b.issues.map((i) => i.code)).not.toContain(PARSE_ERROR_CODE);
        });

        it("re-throws an ImportError and an AbortError untouched", () => {
            const b = new ImportReportBuilder("csv", Infinity);
            const imp = b.abort("done");
            expect(() => b.recordError(imp)).toThrow(imp);
            const abort = new DOMException("stop", "AbortError");
            expect(() => b.recordError(abort)).toThrow(abort);
            expect(b.errorCount).toBe(0);
        });

        it("aborts through the limit like error()", () => {
            const b = new ImportReportBuilder("csv", 0);
            expect(() => b.recordError(new GraphFormatError("E_COLUMN_TYPE", "x", {}))).toThrow(ImportError);
        });
    });

    it("fail() records a parse-error and throws ImportError even beyond the limit", () => {
        const b = new ImportReportBuilder("csv", 0);
        let caught: unknown;
        try {
            b.fail("E_INVALID_UTF8", "bad bytes", { line: 2 }, { byteOffset: 17 });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(ImportError);
        const err = caught as ImportError;
        expect(err.message).toBe("bad bytes");
        expect(err.details).toEqual({ code: "E_INVALID_UTF8", byteOffset: 17 });
        expect(err.report.issues).toEqual([
            {
                category: "parse-error",
                severity: "error",
                code: "E_INVALID_UTF8",
                message: "bad bytes",
                line: 2,
                element: null,
            },
        ]);
        expect(err.report.truncated).toBe(true);
        const relaxed = new ImportReportBuilder("csv", 10);
        expect(() => relaxed.fail("E_X", "x")).toThrow(ImportError);
        expect(relaxed.truncated).toBe(false);
    });

    it("abort() builds the ImportError without throwing", () => {
        const b = new ImportReportBuilder("csv", 10);
        b.warning("coercion", "W", "w");
        const err = b.abort("stopped", { why: "test" });
        expect(err).toBeInstanceOf(ImportError);
        expect(err.report.warningCount).toBe(1);
        expect(err.details).toEqual({ why: "test" });
    });

    it("helpers: isAbortError and messageOf", () => {
        expect(isAbortError(new DOMException("x", "AbortError"))).toBe(true);
        const named = new Error("x");
        named.name = "AbortError";
        expect(isAbortError(named)).toBe(true);
        expect(isAbortError(new Error("x"))).toBe(false);
        expect(isAbortError(null)).toBe(false);
        expect(messageOf(new Error("m"))).toBe("m");
        expect(messageOf("s")).toBe("s");
        expect(messageOf({})).toBe("non-error thrown (object)");
    });
});
