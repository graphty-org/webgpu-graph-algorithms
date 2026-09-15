/**
 * The export -> import equality checker of design section 16.5: compares two snapshots on ids,
 * topology (rowPtr / colIdx), orientation (edgeList), weights (the role-weight shadow column when
 * both have one, the f32 arc weights otherwise, explicitness included) and the declared columns of
 * every table value by value, returning a list of differences with a description; plus the
 * `roundTrip()` driver that exports a snapshot with one plugin, imports the text with another into
 * a fresh builder and freezes.
 */

import {
    type Column,
    type FreezeReport,
    GraphBuilder,
    type GraphBuilderOptions,
    type GraphSnapshot,
} from "@graphty/graph-format";

import {
    type CommonExportOptions,
    type CommonImportOptions,
    type GraphExporter,
    type GraphImporter,
    type ImportReport,
    type LossNote,
} from "../../src/types.js";

/**
 * One difference between two snapshots.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface SnapshotDiff {
    /** Where: "directed", "ids[3]", "nodes.label[4]", "edges.weight[0]", "extensions.temporal:node:price.rowCount"... */
    readonly path: string;
    /** The value in the expected snapshot. */
    readonly expected: unknown;
    /** The value in the actual snapshot. */
    readonly actual: unknown;
    /** A one-line description. */
    readonly message: string;
}

/**
 * What compareSnapshots() compares.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface CompareOptions {
    /** Column names to skip in every table (format-owned helpers an exporter adds, for instance). */
    readonly ignoreColumns?: readonly string[] | undefined;
    /** Column roles to skip in every table. */
    readonly ignoreRoles?: readonly string[] | undefined;
    /** Whether the actual snapshot may hold columns the expected one lacks (default true). */
    readonly allowExtraColumns?: boolean | undefined;
    /** Whether dtypes must match (default true); false compares values only. */
    readonly dtypes?: boolean | undefined;
    /** Whether roles must match (default true). */
    readonly roles?: boolean | undefined;
    /** Whether origin.type must match (default false). */
    readonly originType?: boolean | undefined;
    /** Absolute tolerance for numeric cell and weight comparisons (default 0: Object.is). */
    readonly tolerance?: number | undefined;
    /** Whether the explicit / defaulted status of weights must match (default true). */
    readonly weightExplicitness?: boolean | undefined;
    /** Whether extension tables are compared (default true). */
    readonly extensions?: boolean | undefined;
    /** Stop after this many differences (default 50). */
    readonly limit?: number | undefined;
}

/**
 * Compare two snapshots.
 * @param expected - the reference (the first import)
 * @param actual - the snapshot under test (the re-import)
 * @param options - what to compare
 * @returns the differences, empty when equal
 */
export function compareSnapshots(
    expected: GraphSnapshot,
    actual: GraphSnapshot,
    options: CompareOptions = {},
): SnapshotDiff[] {
    const diffs: SnapshotDiff[] = [];
    const limit = options.limit ?? 50;
    const tolerance = options.tolerance ?? 0;
    const diff = (path: string, e: unknown, a: unknown, message?: string): boolean => {
        diffs.push({
            path,
            expected: e,
            actual: a,
            message: message ?? `${path}: expected ${show(e)}, got ${show(a)}`,
        });
        return diffs.length >= limit;
    };
    const same = (e: unknown, a: unknown): boolean => valuesEqual(e, a, tolerance);

    if (expected.directed !== actual.directed && diff("directed", expected.directed, actual.directed)) {
        return diffs;
    }
    if (expected.nodeCount !== actual.nodeCount && diff("nodeCount", expected.nodeCount, actual.nodeCount)) {
        return diffs;
    }
    if (expected.edgeCount !== actual.edgeCount && diff("edgeCount", expected.edgeCount, actual.edgeCount)) {
        return diffs;
    }

    // ids in index order
    const expectedIds = expected.ids.toArray();
    const actualIds = actual.ids.toArray();
    const n = Math.min(expectedIds.length, actualIds.length);
    for (let i = 0; i < n; i++) {
        if (!Object.is(expectedIds[i], actualIds[i]) && diff(`ids[${i}]`, expectedIds[i], actualIds[i])) {
            return diffs;
        }
    }
    if (expected.nodeCount !== actual.nodeCount || expected.edgeCount !== actual.edgeCount) {
        return diffs;
    }

    // topology: the CSR arrays are identical when node order and edge order survived
    if (!typedEqual(expected.rowPtr, actual.rowPtr)) {
        if (diff("rowPtr", expected.rowPtr, actual.rowPtr, "rowPtr differs")) {
            return diffs;
        }
    }
    if (!typedEqual(expected.colIdx, actual.colIdx)) {
        if (diff("colIdx", expected.colIdx, actual.colIdx, "colIdx differs")) {
            return diffs;
        }
    }

    // orientation of every logical edge
    const el = expected.edgeList();
    const al = actual.edgeList();
    for (let e = 0; e < expected.edgeCount; e++) {
        if (el.src[e] !== al.src[e] || el.dst[e] !== al.dst[e]) {
            if (diff(`edge[${e}]`, `${el.src[e]}->${el.dst[e]}`, `${al.src[e]}->${al.dst[e]}`)) {
                return diffs;
            }
        }
    }

    // weights
    if (expected.flags.weighted !== actual.flags.weighted) {
        if (diff("flags.weighted", expected.flags.weighted, actual.flags.weighted)) {
            return diffs;
        }
    } else if (expected.flags.weighted) {
        const eShadow = expected.edges.byRole("weight");
        const aShadow = actual.edges.byRole("weight");
        if (eShadow !== null && aShadow !== null) {
            for (let e = 0; e < expected.edgeCount; e++) {
                const eSet = eShadow.isSet(e);
                const aSet = aShadow.isSet(e);
                if (options.weightExplicitness !== false && eSet !== aSet) {
                    if (diff(`weightExplicit[${e}]`, eSet, aSet)) {
                        return diffs;
                    }
                    continue;
                }
                if (eSet && aSet && !same(eShadow.value(e), aShadow.value(e))) {
                    if (diff(`weight[${e}]`, eShadow.value(e), aShadow.value(e))) {
                        return diffs;
                    }
                }
            }
        } else {
            if (options.weightExplicitness !== false && (eShadow === null) !== (aShadow === null)) {
                if (
                    diff(
                        "weightColumn",
                        eShadow !== null,
                        aShadow !== null,
                        "one snapshot has a role-weight column, the other does not",
                    )
                ) {
                    return diffs;
                }
            }
            const ew = el.weights;
            const aw = al.weights;
            for (let e = 0; e < expected.edgeCount; e++) {
                const ev = ew === null ? 1 : ew[e];
                const av = aw === null ? 1 : aw[e];
                if (!same(ev, av) && diff(`weight[${e}]`, ev, av)) {
                    return diffs;
                }
            }
        }
    }

    // columns
    for (const [label, e, a] of [
        ["nodes", expected.nodes, actual.nodes],
        ["edges", expected.edges, actual.edges],
        ["graph", expected.graph, actual.graph],
    ] as const) {
        if (compareTables(label, e, a, options, diff, same)) {
            return diffs;
        }
    }
    if (options.extensions !== false) {
        for (const [name, table] of expected.extensions) {
            const other = actual.extensions.get(name);
            if (other === undefined) {
                if (diff(`extensions.${name}`, "present", "absent")) {
                    return diffs;
                }
                continue;
            }
            if (table.rowCount !== other.rowCount) {
                if (diff(`extensions.${name}.rowCount`, table.rowCount, other.rowCount)) {
                    return diffs;
                }
                continue;
            }
            if (compareTables(`extensions.${name}`, table, other, options, diff, same)) {
                return diffs;
            }
        }
        if (options.allowExtraColumns === false) {
            for (const name of actual.extensions.keys()) {
                if (!expected.extensions.has(name) && diff(`extensions.${name}`, "absent", "present")) {
                    return diffs;
                }
            }
        }
    }
    return diffs;
}

/** A table shape both AttributeTable and extension tables satisfy. */
interface TableLike extends Iterable<Column> {
    readonly rowCount: number;
    names(): readonly string[];
    get(name: string): Column | null;
}

/**
 * Compare the columns of one table.
 * @param label - the table name for paths
 * @param expected - the reference table
 * @param actual - the table under test
 * @param options - the compare options
 * @param diff - the recorder; returns true when the limit was hit
 * @param same - the value comparator
 * @returns true when the limit was hit
 */
function compareTables(
    label: string,
    expected: TableLike,
    actual: TableLike,
    options: CompareOptions,
    diff: (path: string, e: unknown, a: unknown, message?: string) => boolean,
    same: (e: unknown, a: unknown) => boolean,
): boolean {
    const ignoreNames = new Set(options.ignoreColumns ?? []);
    const ignoreRoles = new Set(options.ignoreRoles ?? []);
    // the role-weight shadow column is compared by the weights step, not as a declared column
    const skip = (column: Column): boolean =>
        ignoreNames.has(column.meta.name) ||
        (column.meta.role !== null &&
            (ignoreRoles.has(column.meta.role) || (label === "edges" && column.meta.role === "weight")));
    for (const column of expected) {
        if (skip(column)) {
            continue;
        }
        const { name } = column.meta;
        const other = actual.get(name);
        const path = `${label}.${name}`;
        if (other === null) {
            if (diff(path, "present", "absent", `${path}: column missing after round trip`)) {
                return true;
            }
            continue;
        }
        if (options.dtypes !== false) {
            if (column.meta.dtype !== other.meta.dtype && diff(`${path}.dtype`, column.meta.dtype, other.meta.dtype)) {
                return true;
            }
            if (
                column.meta.itemDtype !== other.meta.itemDtype &&
                diff(`${path}.itemDtype`, column.meta.itemDtype, other.meta.itemDtype)
            ) {
                return true;
            }
            if (
                column.meta.components !== other.meta.components &&
                diff(`${path}.components`, column.meta.components, other.meta.components)
            ) {
                return true;
            }
        }
        if (
            options.roles !== false &&
            column.meta.role !== other.meta.role &&
            diff(`${path}.role`, column.meta.role, other.meta.role)
        ) {
            return true;
        }
        if (options.originType === true) {
            const eType = column.meta.origin?.type ?? null;
            const aType = other.meta.origin?.type ?? null;
            if (eType !== aType && diff(`${path}.origin.type`, eType, aType)) {
                return true;
            }
        }
        if (column.length !== other.length) {
            if (diff(`${path}.length`, column.length, other.length)) {
                return true;
            }
            continue;
        }
        for (let r = 0; r < column.length; r++) {
            const eSet = column.isSet(r);
            const aSet = other.isSet(r);
            if (eSet !== aSet) {
                if (diff(`${path}[${r}]`, eSet ? column.value(r) : undefined, aSet ? other.value(r) : undefined)) {
                    return true;
                }
                continue;
            }
            if (!eSet) {
                continue;
            }
            const ev = cellValue(column, r);
            const av = cellValue(other, r);
            if (!same(ev, av) && diff(`${path}[${r}]`, ev, av)) {
                return true;
            }
        }
    }
    if (options.allowExtraColumns === false) {
        for (const column of actual) {
            if (!skip(column) && expected.get(column.meta.name) === null) {
                if (
                    diff(
                        `${label}.${column.meta.name}`,
                        "absent",
                        "present",
                        `${label}.${column.meta.name}: extra column after round trip`,
                    )
                ) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * The comparable value of a set cell: list rows through sliceOf, json rows through values, the
 * typed accessor otherwise (a components > 1 subarray becomes a plain array).
 * @param column - the column
 * @param row - the row
 * @returns the value
 */
function cellValue(column: Column, row: number): unknown {
    switch (column.dtype) {
        case "list":
            return [...column.sliceOf(row)];
        case "json":
            return column.values[row];
        default: {
            const value = column.value(row);
            return ArrayBuffer.isView(value) ? Array.from(value as ArrayLike<number>) : value;
        }
    }
}

/**
 * Deep equality with a numeric tolerance.
 * @param e - expected
 * @param a - actual
 * @param tolerance - absolute tolerance for numbers
 * @returns true when equal
 */
export function valuesEqual(e: unknown, a: unknown, tolerance: number): boolean {
    if (typeof e === "number" && typeof a === "number") {
        if (Object.is(e, a)) {
            return true;
        }
        return Number.isFinite(e) && Number.isFinite(a) && Math.abs(e - a) <= tolerance;
    }
    if (Array.isArray(e) || ArrayBuffer.isView(e)) {
        const ea = Array.from(e as ArrayLike<unknown>);
        if (!(Array.isArray(a) || ArrayBuffer.isView(a))) {
            return false;
        }
        const aa = Array.from(a as ArrayLike<unknown>);
        return ea.length === aa.length && ea.every((v, i) => valuesEqual(v, aa[i], tolerance));
    }
    if (typeof e === "object" && e !== null && typeof a === "object" && a !== null) {
        const eo = e as Record<string, unknown>;
        const ao = a as Record<string, unknown>;
        const keys = Object.keys(eo);
        if (keys.length !== Object.keys(ao).length) {
            return false;
        }
        return keys.every((k) => Object.prototype.hasOwnProperty.call(ao, k) && valuesEqual(eo[k], ao[k], tolerance));
    }
    return Object.is(e, a);
}

/**
 * Element-wise equality of two typed arrays.
 * @param a - one array
 * @param b - the other
 * @returns true when same length and contents
 */
function typedEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

/**
 * A short text of a value for messages.
 * @param v - the value
 * @returns JSON when possible, else String()
 */
function show(v: unknown): string {
    if (ArrayBuffer.isView(v)) {
        const items = Array.from(v as unknown as ArrayLike<number>);
        return `[${items.slice(0, 8).join(", ")}${items.length > 8 ? ", ..." : ""}]`;
    }
    try {
        const text = JSON.stringify(v);
        return text === undefined ? String(v) : text;
    } catch {
        return String(v);
    }
}

/**
 * A multi-line description of differences.
 * @param diffs - the differences
 * @returns one line per difference, "no differences" when empty
 */
export function describeDiffs(diffs: readonly SnapshotDiff[]): string {
    return diffs.length === 0 ? "no differences" : diffs.map((d) => `  - ${d.message}`).join("\n");
}

/**
 * Assert two snapshots are equal, failing with the difference list.
 * @param expected - the reference
 * @param actual - the snapshot under test
 * @param options - what to compare
 */
export function expectSameSnapshot(expected: GraphSnapshot, actual: GraphSnapshot, options: CompareOptions = {}): void {
    const diffs = compareSnapshots(expected, actual, options);
    if (diffs.length > 0) {
        throw new Error(`snapshots differ (${diffs.length} difference(s)):\n${describeDiffs(diffs)}`);
    }
}

/**
 * What roundTrip() returns.
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface RoundTripResult {
    /** The exported document. */
    readonly text: string;
    /** The exporter's pre-flight notes. */
    readonly notes: readonly LossNote[];
    /** The re-imported snapshot. */
    readonly snapshot: GraphSnapshot;
    /** The importer's report. */
    readonly report: ImportReport;
    /** The freeze report of the re-import. */
    readonly freeze: FreezeReport;
}

/**
 * Options of roundTrip().
 * Consumed by the per-format test suites under test/formats.
 * @public
 */
export interface RoundTripOptions<ExportOpts, ImportOpts> {
    /** Options for the exporter. */
    readonly exportOptions?: (ExportOpts & CommonExportOptions) | undefined;
    /** Options for the importer. */
    readonly importOptions?: (ImportOpts & CommonImportOptions) | undefined;
    /** Overrides for the fresh builder the re-import goes into. */
    readonly builder?: Partial<GraphBuilderOptions> | undefined;
}

/**
 * Export a snapshot as text and import it again into a fresh builder (weightDtype "f64" as every
 * importer expects, the importer setting the direction), then freeze.
 * @param snapshot - the snapshot to round-trip
 * @param exporter - the exporter
 * @param importer - the importer
 * @param options - exporter, importer and builder options
 * @returns the text, the notes, the re-imported snapshot and both reports
 */
export async function roundTrip<ExportOpts, ImportOpts>(
    snapshot: GraphSnapshot,
    exporter: GraphExporter<ExportOpts>,
    importer: GraphImporter<ImportOpts>,
    options: RoundTripOptions<ExportOpts, ImportOpts> = {},
): Promise<RoundTripResult> {
    const notes = exporter.check(snapshot, options.exportOptions);
    const text = await exporter.exportToString(snapshot, options.exportOptions);
    const io = options.importOptions;
    const builder = new GraphBuilder({
        directed: snapshot.directed,
        weightDtype: io?.weightDtype ?? "f64",
        addMissingNodes: io?.addMissingNodes ?? true,
        duplicateEdges: io?.duplicateEdges ?? "keep",
        selfLoops: io?.selfLoops ?? "keep",
        ...options.builder,
    });
    const report = await importer.import(text, builder, io);
    const { snapshot: result, report: freeze } = builder.freezeWithReport();
    return { text, notes, snapshot: result, report, freeze };
}
