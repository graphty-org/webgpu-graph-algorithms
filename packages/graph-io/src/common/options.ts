/**
 * Option normalisation for importers and exporters (design sections 8.4 and 8.5): every common
 * option resolved to its documented default, enum values checked (E_UNSUPPORTED, the core's
 * convention for an option outside its set), and the per-format defaults (`ids`, `defaultDirected`,
 * `weightFrom`, `addMissingNodes`) supplied by the importer that calls resolveImportOptions().
 */

import { type DuplicatePolicy, GraphFormatError, type GraphSink, type IdCoercion } from "@graphty/graph-format";

import { type CommonExportOptions, type CommonImportOptions } from "../types.js";
import { OPTION_IGNORED_CODE, SINK_OPTION_CODE } from "./codes.js";
import { type ImportReportBuilder } from "./report.js";

/** The defaults an importer supplies for the options whose default is per format (design section 8.4). */
export interface ImportFormatDefaults {
    /** "canonical" for text-cell formats, "keep" for JSON. */
    readonly ids: IdCoercion;
    /** The direction assumed when the file declares none. */
    readonly defaultDirected: boolean;
    /** The attribute that becomes the weight ("weight", GML "value"), or null for an unweighted format. */
    readonly weightFrom: string | null;
    /** Whether edges may reference undeclared nodes; true unless the format says otherwise (GEXF: false). */
    readonly addMissingNodes?: boolean | undefined;
}

/**
 * CommonImportOptions with every field present (design section 8.4 defaults applied).
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface ResolvedImportOptions {
    /** The id coercion rule. */
    readonly ids: IdCoercion;
    /** Which field becomes the node id. */
    readonly nodeIdFrom: "id" | "label" | "index";
    /** Whether an edge may reference an undeclared node. */
    readonly addMissingNodes: boolean;
    /** The builder's duplicate-edge policy seed. */
    readonly duplicateEdges: DuplicatePolicy;
    /** The builder's self-loop policy seed. */
    readonly selfLoops: "keep" | "drop" | "error";
    /** The mixed-direction policy. */
    readonly onMixedDirection: "expand" | "directed" | "undirected" | "error";
    /** The direction assumed when the file declares none. */
    readonly defaultDirected: boolean;
    /** The weight attribute, or null for unweighted. */
    readonly weightFrom: string | null;
    /** The weight staging precision. */
    readonly weightDtype: "f32" | "f64";
    /** How declared long columns are stored. */
    readonly long: "f64" | "string";
    /** Whether mangled ids are restored. */
    readonly restoreMangledIds: boolean;
    /** The hyperedge policy. */
    readonly hyperedges: "error" | "skip" | "star" | "clique";
    /** Errors tolerated before aborting. */
    readonly errorLimit: number;
    /** The cancellation signal, or null. */
    readonly signal: AbortSignal | null;
    /** The progress callback, or null. */
    readonly onProgress: ((bytesDone: number, bytesTotal?: number) => void) | null;
}

/** CommonExportOptions with every field present (design section 8.5 defaults applied). */
export interface ResolvedExportOptions {
    /** "error" never renames a node; "mangle" rewrites and keeps the original. */
    readonly sanitizeIds: "error" | "mangle";
    /** What a format without mixed-direction support does with a mixed snapshot. */
    readonly onMixedDirection: "error" | "directed" | "undirected";
}

const ID_COERCIONS: ReadonlySet<string> = new Set(["keep", "canonical", "string", "number"]);
const NODE_ID_SOURCES: ReadonlySet<string> = new Set(["id", "label", "index"]);
const DUPLICATE_POLICIES: ReadonlySet<string> = new Set(["keep", "error", "first", "last", "sum", "min", "max"]);
const SELF_LOOP_POLICIES: ReadonlySet<string> = new Set(["keep", "drop", "error"]);
const IMPORT_MIXED_POLICIES: ReadonlySet<string> = new Set(["expand", "directed", "undirected", "error"]);
const EXPORT_MIXED_POLICIES: ReadonlySet<string> = new Set(["error", "directed", "undirected"]);
const WEIGHT_DTYPES: ReadonlySet<string> = new Set(["f32", "f64"]);
const LONG_MODES: ReadonlySet<string> = new Set(["f64", "string"]);
const HYPEREDGE_POLICIES: ReadonlySet<string> = new Set(["error", "skip", "star", "clique"]);
const SANITIZE_MODES: ReadonlySet<string> = new Set(["error", "mangle"]);

/** The default error limit of design section 8.4. */
export const DEFAULT_ERROR_LIMIT = 100;

export { SINK_OPTION_CODE };

/** The builder-policy options a sink exposes read-only as `sink.options` (design section 8.4). */
const SINK_OPTION_NAMES = ["addMissingNodes", "duplicateEdges", "selfLoops", "weightDtype"] as const;

/**
 * Report every builder-policy option the caller explicitly requested that the sink does not use
 * (design section 8.4 precedence), one `W_SINK_OPTION` warning per option with the option name as
 * the element. Options left undefined are never reported: they are defaults, not requests. On the
 * registry's builder, which is seeded from the same options, nothing is ever reported.
 * @param sink - the sink the importer pushes into
 * @param options - the caller's raw options, possibly undefined
 * @param report - the report to record into
 * @param enforcesMissingNodes - true when the importer applies `addMissingNodes: false` itself (it
 * refuses unknown endpoints before the sink sees them), so that request is honoured on any sink and
 * only `addMissingNodes: true` against a refusing sink is reported
 * @returns the number of warnings recorded
 */
export function reportSinkOptions(
    sink: GraphSink,
    options: CommonImportOptions | undefined,
    report: ImportReportBuilder,
    enforcesMissingNodes = false,
): number {
    if (options === undefined) {
        return 0;
    }
    let recorded = 0;
    for (const name of SINK_OPTION_NAMES) {
        const wanted: unknown = options[name];
        const actual: unknown = sink.options[name];
        if (wanted === undefined || wanted === actual) {
            continue;
        }
        if (name === "addMissingNodes" && enforcesMissingNodes && wanted === false) {
            continue;
        }
        report.warning(
            "coercion",
            SINK_OPTION_CODE,
            `option ${name}: ${JSON.stringify(wanted)} requested but the sink uses ${JSON.stringify(actual)}; the sink's setting applies`,
            { element: name },
        );
        recorded++;
    }
    return recorded;
}

/** The common options a format may leave unused; `signal`, `onProgress` and `errorLimit` apply everywhere. */
const IGNORABLE_OPTION_NAMES = [
    "ids",
    "nodeIdFrom",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "defaultDirected",
    "weightFrom",
    "weightDtype",
    "long",
    "restoreMangledIds",
    "hyperedges",
] as const;

/**
 * Report every common option the caller set to a non-default value that the format has no use
 * for (design section 8.4: "the importer reports every option it could not honour"): one
 * `W_OPTION_IGNORED` warning (category `unsupported`) per option, the option name as the element.
 * The builder-policy options are reportSinkOptions()'s and are skipped here.
 * @param options - the caller's raw options, possibly undefined
 * @param report - the report to record into
 * @param used - the common option names the importer reads
 * @returns the number of warnings recorded
 */
export function reportUnusedOptions(
    options: CommonImportOptions | undefined,
    report: ImportReportBuilder,
    used: ReadonlySet<keyof CommonImportOptions>,
): number {
    if (options === undefined) {
        return 0;
    }
    let recorded = 0;
    for (const name of IGNORABLE_OPTION_NAMES) {
        if (used.has(name) || (SINK_OPTION_NAMES as readonly string[]).includes(name)) {
            continue;
        }
        const value: unknown = options[name];
        if (value === undefined) {
            continue;
        }
        report.warning(
            "unsupported",
            OPTION_IGNORED_CODE,
            `option ${name}: ${JSON.stringify(value)} has no effect on the ${report.format} importer`,
            { element: name },
        );
        recorded++;
    }
    return recorded;
}

/**
 * Apply the design section 8.4 defaults to an importer's common options and check every enum
 * value. Format-specific options in the same object are ignored here.
 * @param options - the caller's options, possibly undefined
 * @param defaults - the importer's per-format defaults
 * @returns the resolved options; E_UNSUPPORTED for a value outside its set
 */
export function resolveImportOptions(
    options: CommonImportOptions | undefined,
    defaults: ImportFormatDefaults,
): ResolvedImportOptions {
    const o: CommonImportOptions = options ?? {};
    return Object.freeze({
        ids: enumOption("ids", o.ids, defaults.ids, ID_COERCIONS),
        nodeIdFrom: enumOption("nodeIdFrom", o.nodeIdFrom, "id", NODE_ID_SOURCES),
        addMissingNodes: booleanOption("addMissingNodes", o.addMissingNodes, defaults.addMissingNodes ?? true),
        duplicateEdges: enumOption("duplicateEdges", o.duplicateEdges, "keep", DUPLICATE_POLICIES),
        selfLoops: enumOption("selfLoops", o.selfLoops, "keep", SELF_LOOP_POLICIES),
        onMixedDirection: enumOption("onMixedDirection", o.onMixedDirection, "expand", IMPORT_MIXED_POLICIES),
        defaultDirected: booleanOption("defaultDirected", o.defaultDirected, defaults.defaultDirected),
        weightFrom: weightFromOption(o.weightFrom, defaults.weightFrom),
        weightDtype: enumOption("weightDtype", o.weightDtype, "f64", WEIGHT_DTYPES),
        long: enumOption("long", o.long, "f64", LONG_MODES),
        restoreMangledIds: booleanOption("restoreMangledIds", o.restoreMangledIds, true),
        hyperedges: enumOption("hyperedges", o.hyperedges, "skip", HYPEREDGE_POLICIES),
        errorLimit: errorLimitOption(o.errorLimit),
        signal: signalOption(o.signal),
        onProgress: progressOption(o.onProgress),
    });
}

/**
 * Apply the design section 8.5 defaults to an exporter's common options and check the enum values.
 * @param options - the caller's options, possibly undefined
 * @returns the resolved options; E_UNSUPPORTED for a value outside its set
 */
export function resolveExportOptions(options: CommonExportOptions | undefined): ResolvedExportOptions {
    const o: CommonExportOptions = options ?? {};
    return Object.freeze({
        sanitizeIds: enumOption("sanitizeIds", o.sanitizeIds, "error", SANITIZE_MODES),
        onMixedDirection: enumOption("onMixedDirection", o.onMixedDirection, "error", EXPORT_MIXED_POLICIES),
    });
}

/**
 * Resolve one enum-valued option.
 * @param name - the option name, for the error
 * @param value - the caller's value
 * @param fallback - the default
 * @param allowed - the accepted values
 * @returns the value or the default; E_UNSUPPORTED when outside the set
 */
function enumOption<T extends string>(name: string, value: unknown, fallback: T, allowed: ReadonlySet<string>): T {
    if (value === undefined) {
        return fallback;
    }
    if (typeof value !== "string" || !allowed.has(value)) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option ${name}: ${describe(value)} is not one of ${list(allowed)}`,
            {
                option: name,
                found: value,
                supported: [...allowed],
            },
        );
    }
    return value as T;
}

/**
 * Resolve one boolean option.
 * @param name - the option name, for the error
 * @param value - the caller's value
 * @param fallback - the default
 * @returns the value or the default; E_UNSUPPORTED when not a boolean
 */
function booleanOption(name: string, value: unknown, fallback: boolean): boolean {
    if (value === undefined) {
        return fallback;
    }
    if (typeof value !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", `option ${name}: ${describe(value)} is not a boolean`, {
            option: name,
            found: value,
        });
    }
    return value;
}

/**
 * Resolve the weightFrom option: a non-empty attribute name, null for unweighted, or the format default.
 * @param value - the caller's value
 * @param fallback - the format default
 * @returns the resolved value; E_UNSUPPORTED for anything else
 */
function weightFromOption(value: unknown, fallback: string | null): string | null {
    if (value === undefined) {
        return fallback;
    }
    if (value === null || (typeof value === "string" && value.length > 0)) {
        return value;
    }
    throw new GraphFormatError(
        "E_UNSUPPORTED",
        `option weightFrom: ${describe(value)} is not an attribute name or null`,
        { option: "weightFrom", found: value },
    );
}

/**
 * Resolve the error limit: a non-negative integer or Infinity.
 * @param value - the caller's value
 * @returns the limit or the default
 */
function errorLimitOption(value: unknown): number {
    if (value === undefined) {
        return DEFAULT_ERROR_LIMIT;
    }
    if (typeof value === "number" && value >= 0 && (Number.isInteger(value) || value === Infinity)) {
        return value;
    }
    throw new GraphFormatError(
        "E_UNSUPPORTED",
        `option errorLimit: ${describe(value)} is not a non-negative integer or Infinity`,
        { option: "errorLimit", found: value },
    );
}

/**
 * Resolve the signal option by duck type, so a signal from another realm is accepted.
 * @param value - the caller's value
 * @returns the signal or null
 */
function signalOption(value: unknown): AbortSignal | null {
    if (value === undefined || value === null) {
        return null;
    }
    if (typeof value === "object" && typeof (value as { aborted?: unknown }).aborted === "boolean") {
        return value as AbortSignal;
    }
    throw new GraphFormatError("E_UNSUPPORTED", `option signal: ${describe(value)} is not an AbortSignal`, {
        option: "signal",
        found: typeof value,
    });
}

/**
 * Resolve the progress callback option.
 * @param value - the caller's value
 * @returns the callback or null
 */
function progressOption(value: unknown): ((bytesDone: number, bytesTotal?: number) => void) | null {
    if (value === undefined || value === null) {
        return null;
    }
    if (typeof value === "function") {
        return value as (bytesDone: number, bytesTotal?: number) => void;
    }
    throw new GraphFormatError("E_UNSUPPORTED", `option onProgress: ${describe(value)} is not a function`, {
        option: "onProgress",
        found: typeof value,
    });
}

/**
 * A short description of an option value for an error message.
 * @param value - the value
 * @returns the JSON text of a primitive, or the type name otherwise
 */
function describe(value: unknown): string {
    switch (typeof value) {
        case "string":
            return JSON.stringify(value);
        case "number":
        case "boolean":
            return String(value);
        default:
            return value === null ? "null" : typeof value;
    }
}

/**
 * The accepted values of an enum option, for an error message.
 * @param allowed - the set
 * @returns the quoted values joined by commas
 */
function list(allowed: ReadonlySet<string>): string {
    return [...allowed].map((v) => JSON.stringify(v)).join(", ");
}
