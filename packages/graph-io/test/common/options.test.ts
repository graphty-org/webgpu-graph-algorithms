import { GraphBuilder, GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    DEFAULT_ERROR_LIMIT,
    type ImportFormatDefaults,
    reportSinkOptions,
    resolveExportOptions,
    resolveImportOptions,
    SINK_OPTION_CODE,
} from "../../src/common/options.js";
import { ImportReportBuilder } from "../../src/common/report.js";

const TEXT_DEFAULTS: ImportFormatDefaults = { ids: "canonical", defaultDirected: false, weightFrom: "weight" };

describe("resolveImportOptions (design 8.4 defaults)", () => {
    it("applies every documented default", () => {
        const o = resolveImportOptions(undefined, TEXT_DEFAULTS);
        expect(o).toEqual({
            ids: "canonical",
            nodeIdFrom: "id",
            addMissingNodes: true,
            duplicateEdges: "keep",
            selfLoops: "keep",
            onMixedDirection: "expand",
            defaultDirected: false,
            weightFrom: "weight",
            weightDtype: "f64",
            long: "f64",
            restoreMangledIds: true,
            hyperedges: "skip",
            errorLimit: DEFAULT_ERROR_LIMIT,
            signal: null,
            onProgress: null,
        });
        expect(DEFAULT_ERROR_LIMIT).toBe(100);
        expect(Object.isFrozen(o)).toBe(true);
    });

    it("takes the per-format defaults for ids, defaultDirected, weightFrom and addMissingNodes", () => {
        const o = resolveImportOptions(
            {},
            { ids: "keep", defaultDirected: true, weightFrom: "value", addMissingNodes: false },
        );
        expect(o.ids).toBe("keep");
        expect(o.defaultDirected).toBe(true);
        expect(o.weightFrom).toBe("value");
        expect(o.addMissingNodes).toBe(false);
        expect(
            resolveImportOptions({}, { ids: "keep", defaultDirected: true, weightFrom: null }).weightFrom,
        ).toBeNull();
    });

    it("lets the caller override every option", () => {
        const { signal } = new AbortController();
        const onProgress = (): void => undefined;
        const o = resolveImportOptions(
            {
                ids: "number",
                nodeIdFrom: "label",
                addMissingNodes: false,
                duplicateEdges: "sum",
                selfLoops: "drop",
                onMixedDirection: "error",
                defaultDirected: true,
                weightFrom: null,
                weightDtype: "f32",
                long: "string",
                restoreMangledIds: false,
                hyperedges: "clique",
                errorLimit: 5,
                signal,
                onProgress,
            },
            TEXT_DEFAULTS,
        );
        expect(o).toEqual({
            ids: "number",
            nodeIdFrom: "label",
            addMissingNodes: false,
            duplicateEdges: "sum",
            selfLoops: "drop",
            onMixedDirection: "error",
            defaultDirected: true,
            weightFrom: null,
            weightDtype: "f32",
            long: "string",
            restoreMangledIds: false,
            hyperedges: "clique",
            errorLimit: 5,
            signal,
            onProgress,
        });
    });

    it("treats an explicit undefined as absent", () => {
        const o = resolveImportOptions({ ids: undefined, errorLimit: undefined, signal: undefined }, TEXT_DEFAULTS);
        expect(o.ids).toBe("canonical");
        expect(o.errorLimit).toBe(100);
        expect(o.signal).toBeNull();
    });

    it.each([
        ["ids", "canonicalish"],
        ["nodeIdFrom", "name"],
        ["duplicateEdges", "merge"],
        ["selfLoops", "allow"],
        ["onMixedDirection", "mixed"],
        ["weightDtype", "f16"],
        ["long", "bigint"],
        ["hyperedges", "ignore"],
    ])("rejects an unknown %s value with E_UNSUPPORTED", (option, value) => {
        let caught: unknown;
        try {
            resolveImportOptions({ [option]: value }, TEXT_DEFAULTS);
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        const err = caught as GraphFormatError;
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.option).toBe(option);
        expect(err.details.found).toBe(value);
        expect(Array.isArray(err.details.supported)).toBe(true);
    });

    it("rejects non-boolean booleans, bad weightFrom, bad errorLimit, bad signal and bad onProgress", () => {
        const bad = (options: Record<string, unknown>): string => {
            try {
                resolveImportOptions(options, TEXT_DEFAULTS);
            } catch (err) {
                return err instanceof GraphFormatError ? `${err.code}:${String(err.details.option)}` : "other";
            }
            return "none";
        };
        expect(bad({ addMissingNodes: "yes" })).toBe("E_UNSUPPORTED:addMissingNodes");
        expect(bad({ defaultDirected: 1 })).toBe("E_UNSUPPORTED:defaultDirected");
        expect(bad({ restoreMangledIds: "no" })).toBe("E_UNSUPPORTED:restoreMangledIds");
        expect(bad({ weightFrom: "" })).toBe("E_UNSUPPORTED:weightFrom");
        expect(bad({ weightFrom: 3 })).toBe("E_UNSUPPORTED:weightFrom");
        expect(bad({ errorLimit: -1 })).toBe("E_UNSUPPORTED:errorLimit");
        expect(bad({ errorLimit: 1.5 })).toBe("E_UNSUPPORTED:errorLimit");
        expect(bad({ errorLimit: "10" })).toBe("E_UNSUPPORTED:errorLimit");
        expect(bad({ signal: {} })).toBe("E_UNSUPPORTED:signal");
        expect(bad({ onProgress: "log" })).toBe("E_UNSUPPORTED:onProgress");
        expect(bad({ errorLimit: Infinity })).toBe("none");
        expect(bad({ errorLimit: 0 })).toBe("none");
    });

    it("accepts a duck-typed signal from another realm", () => {
        const fake = { aborted: false } as AbortSignal;
        expect(resolveImportOptions({ signal: fake }, TEXT_DEFAULTS).signal).toBe(fake);
    });
});

describe("resolveExportOptions (design 8.5 defaults)", () => {
    it("defaults to sanitizeIds error and onMixedDirection error", () => {
        expect(resolveExportOptions(undefined)).toEqual({ sanitizeIds: "error", onMixedDirection: "error" });
        expect(resolveExportOptions({})).toEqual({ sanitizeIds: "error", onMixedDirection: "error" });
    });

    it("accepts the documented values and rejects others", () => {
        expect(resolveExportOptions({ sanitizeIds: "mangle", onMixedDirection: "undirected" })).toEqual({
            sanitizeIds: "mangle",
            onMixedDirection: "undirected",
        });
        expect(resolveExportOptions({ onMixedDirection: "directed" }).onMixedDirection).toBe("directed");
        expect(() => resolveExportOptions({ sanitizeIds: "rename" as "mangle" })).toThrow(GraphFormatError);
        // "expand" is an import-only value (design 8.5)
        expect(() => resolveExportOptions({ onMixedDirection: "expand" as "directed" })).toThrow(GraphFormatError);
    });
});

describe("reportSinkOptions (design 8.4 precedence)", () => {
    const sink = (): GraphBuilder =>
        new GraphBuilder({
            directed: true,
            weightDtype: "f32",
            duplicateEdges: "keep",
            selfLoops: "keep",
            addMissingNodes: false,
        });

    it("reports every explicitly requested builder-policy option the sink does not use, as coercion warnings", () => {
        const report = new ImportReportBuilder("test", 100);
        const n = reportSinkOptions(
            sink(),
            { weightDtype: "f64", duplicateEdges: "sum", selfLoops: "keep", addMissingNodes: true, ids: "keep" },
            report,
        );
        expect(n).toBe(3);
        const finished = report.finish();
        expect(finished.issues.map((i) => [i.code, i.element, i.category, i.severity])).toEqual([
            [SINK_OPTION_CODE, "addMissingNodes", "coercion", "warning"],
            [SINK_OPTION_CODE, "duplicateEdges", "coercion", "warning"],
            [SINK_OPTION_CODE, "weightDtype", "coercion", "warning"],
        ]);
        expect(finished.issues[2].message).toBe(
            'option weightDtype: "f64" requested but the sink uses "f32"; the sink\'s setting applies',
        );
        expect(finished.errorCount).toBe(0);
    });

    it("never reports undefined options or an absent option object", () => {
        const report = new ImportReportBuilder("test", 100);
        expect(reportSinkOptions(sink(), undefined, report)).toBe(0);
        expect(reportSinkOptions(sink(), {}, report)).toBe(0);
        expect(reportSinkOptions(sink(), { ids: "number", errorLimit: 0 }, report)).toBe(0);
        expect(report.finish().issues).toEqual([]);
    });

    it("treats addMissingNodes: false as honoured when the importer enforces it itself", () => {
        const lenient = new GraphBuilder({ directed: true, addMissingNodes: true });
        const report = new ImportReportBuilder("test", 100);
        expect(reportSinkOptions(lenient, { addMissingNodes: false }, report, true)).toBe(0);
        expect(reportSinkOptions(lenient, { addMissingNodes: false }, report)).toBe(1);
        // the looser request against a refusing sink is never honoured
        expect(reportSinkOptions(sink(), { addMissingNodes: true }, report, true)).toBe(1);
        expect(report.finish().issues.map((i) => i.element)).toEqual(["addMissingNodes", "addMissingNodes"]);
    });
});
