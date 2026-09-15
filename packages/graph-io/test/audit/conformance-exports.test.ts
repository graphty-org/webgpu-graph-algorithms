/**
 * Audit (design sections 12.4, 8.2 and 13.1): the public surface of @graphty/graph-io.
 *
 * Pins, without the design document at hand:
 * - the eleven io contract names of section 12.4 (their shapes transcribed verbatim into local
 *   types and compared with expectTypeOf, so a drift in either direction fails `tsc` in lint);
 * - the registry, sniffing and children surfaces of section 8.2 / 13.1;
 * - the eight per-format subpath exports of section 8.2 with `types` first (13.2) and the
 *   identity of every subpath export with the root barrel's;
 * - ImportError: a GraphFormatError whose code "E_IMPORT" narrows and that carries the report.
 *
 * The mechanical comparison against the design listing itself is tmp/io-conformance/check-12-4.ts
 * (see packages/CONFORMANCE.md); this file is its CI-resident shadow.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
    type Dtype,
    type DuplicatePolicy,
    GraphFormatError,
    type GraphFormatErrorCode,
    type GraphSink,
    type GraphSnapshot,
    type IdCoercion,
} from "@graphty/graph-format";
import { describe, expect, expectTypeOf, it } from "vitest";

import * as csv from "../../src/formats/csv/index.js";
import * as dot from "../../src/formats/dot/index.js";
import * as gexf from "../../src/formats/gexf/index.js";
import * as gml from "../../src/formats/gml/index.js";
import * as graphml from "../../src/formats/graphml/index.js";
import * as json from "../../src/formats/json/index.js";
import * as neo4j from "../../src/formats/neo4j/index.js";
import * as pajek from "../../src/formats/pajek/index.js";
import * as root from "../../src/index.js";
import {
    type CommonExportOptions,
    type CommonImportOptions,
    type ExportCapabilities,
    type GraphExporter,
    type GraphImporter,
    ImportError,
    type ImportInput,
    type ImportIssue,
    type ImportReport,
    type IssueCategory,
    type LossNote,
} from "../../src/index.js";

// ============================================================ 12.4, transcribed

type DesignImportInput = string | Uint8Array | ReadableStream<Uint8Array> | AsyncIterable<string | Uint8Array>;
interface DesignCommonImportOptions {
    ids?: IdCoercion | undefined;
    nodeIdFrom?: "id" | "label" | "index" | undefined;
    addMissingNodes?: boolean | undefined;
    duplicateEdges?: DuplicatePolicy | undefined;
    selfLoops?: "keep" | "drop" | "error" | undefined;
    onMixedDirection?: "expand" | "directed" | "undirected" | "error" | undefined;
    defaultDirected?: boolean | undefined;
    weightFrom?: string | null | undefined;
    weightDtype?: "f32" | "f64" | undefined;
    long?: "f64" | "string" | undefined;
    restoreMangledIds?: boolean | undefined;
    hyperedges?: "error" | "skip" | "star" | "clique" | undefined;
    errorLimit?: number | undefined;
    signal?: AbortSignal | undefined;
    onProgress?: ((bytesDone: number, bytesTotal?: number) => void) | undefined;
}
interface DesignGraphImporter<Opts = unknown> {
    readonly format: string;
    readonly extensions: readonly string[];
    readonly mimeTypes: readonly string[];
    sniff?(head: Uint8Array): number;
    import(input: ImportInput, sink: GraphSink, options?: Opts & CommonImportOptions): Promise<ImportReport>;
}
interface DesignExportCapabilities {
    readonly mixedDirection: boolean;
    readonly multiEdges: boolean;
    readonly selfLoops: boolean;
    readonly edgeIds: "required" | "optional" | "none";
    readonly idCharset: "any" | "nmtoken" | "integer" | "dense-1-based";
    readonly dtypes: readonly Dtype[];
    readonly components: boolean;
    readonly lists: boolean;
    readonly json: boolean;
    readonly defaults: boolean;
    readonly options: boolean;
    readonly hierarchy: boolean;
    readonly temporal: "none" | "intervals" | "spells" | "dynamic-values";
    readonly graphAttributes: boolean;
    readonly positions: boolean;
    readonly viz: boolean;
}
interface DesignLossNote {
    readonly code: string;
    readonly message: string;
    readonly column: string | null;
    readonly count: number | null;
}
interface DesignCommonExportOptions {
    sanitizeIds?: "error" | "mangle" | undefined;
    onMixedDirection?: "error" | "directed" | "undirected" | undefined;
}
interface DesignGraphExporter<Opts = unknown> {
    readonly format: string;
    readonly capabilities: ExportCapabilities;
    check(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): readonly LossNote[];
    export(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): AsyncIterable<Uint8Array>;
    exportToString(snapshot: GraphSnapshot, options?: Opts & CommonExportOptions): Promise<string>;
}
type DesignIssueCategory =
    "parse-error" | "missing-value" | "validation-error" | "unsupported" | "precision" | "coercion" | "merged";
interface DesignImportIssue {
    readonly category: IssueCategory;
    readonly severity: "error" | "warning";
    readonly code: string;
    readonly message: string;
    readonly line: number | null;
    readonly element: string | null;
}
interface DesignImportReport {
    readonly format: string;
    readonly counts: {
        readonly nodes: number;
        readonly edges: number;
        readonly skippedNodes: number;
        readonly skippedEdges: number;
        readonly expandedMixed: number;
    };
    readonly issues: readonly ImportIssue[];
    readonly errorCount: number;
    readonly warningCount: number;
    readonly truncated: boolean;
    readonly lossy: readonly LossNote[];
    readonly durationMs: number;
}

describe("design 12.4: the io contract types are exported with the listed shapes", () => {
    it("matches every type alias and interface of the listing", () => {
        expectTypeOf<ImportInput>().toEqualTypeOf<DesignImportInput>();
        expectTypeOf<CommonImportOptions>().toEqualTypeOf<DesignCommonImportOptions>();
        expectTypeOf<GraphImporter>().toEqualTypeOf<DesignGraphImporter>();
        expectTypeOf<GraphImporter<{ delimiter?: string }>>().toEqualTypeOf<
            DesignGraphImporter<{ delimiter?: string }>
        >();
        expectTypeOf<ExportCapabilities>().toEqualTypeOf<DesignExportCapabilities>();
        expectTypeOf<LossNote>().toEqualTypeOf<DesignLossNote>();
        expectTypeOf<CommonExportOptions>().toEqualTypeOf<DesignCommonExportOptions>();
        expectTypeOf<GraphExporter>().toEqualTypeOf<DesignGraphExporter>();
        expectTypeOf<GraphExporter<{ pretty?: boolean }>>().toEqualTypeOf<DesignGraphExporter<{ pretty?: boolean }>>();
        expectTypeOf<IssueCategory>().toEqualTypeOf<DesignIssueCategory>();
        expectTypeOf<ImportIssue>().toEqualTypeOf<DesignImportIssue>();
        expectTypeOf<ImportReport>().toEqualTypeOf<DesignImportReport>();
        // the default Opts = unknown lets a caller pass the common options alone
        const options: Parameters<GraphImporter["import"]>[2] = { ids: "canonical", errorLimit: 3 };
        expect(options).toBeDefined();
    });

    it("ImportError extends GraphFormatError, carries the report and narrows on E_IMPORT", () => {
        const report: ImportReport = Object.freeze({
            format: "probe",
            counts: { nodes: 0, edges: 0, skippedNodes: 0, skippedEdges: 0, expandedMixed: 0 },
            issues: [],
            errorCount: 0,
            warningCount: 0,
            truncated: false,
            lossy: [],
            durationMs: 0,
        });
        const err = new ImportError("stopped", report, { why: "probe" });
        expect(err).toBeInstanceOf(GraphFormatError);
        expect(err).toBeInstanceOf(Error);
        expect(err.name).toBe("ImportError");
        expect(err.code).toBe("E_IMPORT");
        expect(err.report).toBe(report);
        expect(err.details).toEqual({ why: "probe" });
        expectTypeOf(err.report).toEqualTypeOf<ImportReport>();
        // the code is reserved in the core's union (design 8.6)
        const code: GraphFormatErrorCode = "E_IMPORT";
        const thrown: GraphFormatError = err;
        if (thrown.code === code) {
            expect(thrown).toBe(err);
        }
        // NOTE (audit): the listing shows no constructor, so the inherited (code, message, details)
        // is implied; the implementation's (message, report, details) fixes the code and takes the report
        expect(new ImportError("m", report).message).toBe("m");
    });
});

// ============================================================ 8.2 / 13.1 surfaces

const FORMATS = ["gexf", "graphml", "gml", "dot", "pajek", "csv", "json", "neo4j"] as const;
const SUBPATHS: Record<(typeof FORMATS)[number], Record<string, unknown>> = {
    gexf,
    graphml,
    gml,
    dot,
    pajek,
    csv,
    json,
    neo4j,
};

describe("design 8.2 / 13.1: registry, sniff, children and the eight format surfaces", () => {
    it("exports the registry with importGraph / exportGraph / sniff and the children CSR helper", () => {
        expect(typeof root.importGraph).toBe("function");
        expect(typeof root.exportGraph).toBe("function");
        expect(typeof root.sniff).toBe("function");
        expect(root.registry).toBeInstanceOf(root.FormatRegistry);
        expect(typeof root.createRegistry).toBe("function");
        expect(typeof root.childrenCsr).toBe("function");
        expect(root.registry.formats()).toEqual([...root.GRAPH_FORMATS]);
        expect(new Set(root.GRAPH_FORMATS)).toEqual(new Set(FORMATS));
    });

    it("registers one importer and one exporter per format, each typed by the 12.4 contract", () => {
        for (const format of FORMATS) {
            const importer = root.registry.importer(format);
            const exporter = root.registry.exporter(format);
            expect(importer.format).toBe(format);
            expect(exporter.format).toBe(format);
            expect(Array.isArray(importer.extensions)).toBe(true);
            expect(importer.extensions.every((e) => e.startsWith("."))).toBe(true);
            expect(Array.isArray(importer.mimeTypes)).toBe(true);
            expect(typeof importer.import).toBe("function");
            expect(typeof importer.sniff).toBe("function");
            expect(typeof exporter.check).toBe("function");
            expect(typeof exporter.export).toBe("function");
            expect(typeof exporter.exportToString).toBe("function");
            expect(Object.isFrozen(exporter.capabilities)).toBe(true);
            expect(Object.isFrozen(exporter.capabilities.dtypes)).toBe(true);
        }
    });

    it("declares the eight subpath exports in package.json with types first", () => {
        const here = dirname(fileURLToPath(import.meta.url));
        const pkg = JSON.parse(readFileSync(join(here, "..", "..", "package.json"), "utf-8")) as {
            exports: Record<string, Record<string, string>>;
        };
        expect(Object.keys(pkg.exports)).toEqual([".", ...FORMATS.map((f) => `./${f}`)]);
        for (const [key, entry] of Object.entries(pkg.exports)) {
            const name = key === "." ? "graph-io" : key.slice(2);
            expect(Object.keys(entry)[0], `${key}: types must come first`).toBe("types");
            expect(entry.types).toBe(`./dist/${name}.d.ts`);
            expect(entry.import).toBe(`./dist/${name}.js`);
        }
    });

    it("re-exports every subpath export from the root barrel as the same object", () => {
        for (const format of FORMATS) {
            const sub = SUBPATHS[format];
            expect(Object.keys(sub).length).toBeGreaterThanOrEqual(4);
            for (const [name, value] of Object.entries(sub)) {
                expect((root as Record<string, unknown>)[name], `${format}: ${name}`).toBe(value);
            }
            expect(sub[`${format}Importer`]).toBe(root.registry.importer(format));
            expect(sub[`${format}Exporter`]).toBe(root.registry.exporter(format));
        }
    });
});
