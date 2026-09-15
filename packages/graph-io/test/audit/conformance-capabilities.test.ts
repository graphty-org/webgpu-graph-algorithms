/**
 * Audit (design sections 8.5 and 12.4): is every exporter's `capabilities` table TRUTHFUL?
 *
 * For every exporter (GEXF 1.3 and 1.2, GraphML, GML, DOT, Pajek, CSV, the six JSON dialects,
 * Neo4j) and every field of ExportCapabilities, a probe snapshot that exhibits exactly that feature
 * is exported and re-imported:
 *
 * - a capability the table DECLARES must hold: `check()` reports no loss note for it and the
 *   export -> import round trip carries the feature (topology, dtype, values, role);
 * - a capability the table DENIES must be REPORTED: `check()` returns the generic loss code of
 *   `LOSS` (or the format's own code for the same gap) so a caller can tell before writing.
 *
 * The probes use the public surface only (the root barrel and `@graphty/graph-format`), so they run
 * against the same objects a consumer gets. A failing case here is a capability table that lies in
 * one direction or the other; the audit's findings name them.
 */

import { type ColumnDecl, GraphBuilder, type GraphSnapshot, INVALID_INDEX, type NodeId } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    csvExporter,
    csvImporter,
    DirectionResolver,
    dotExporter,
    dotImporter,
    type ExportCapabilities,
    GEXF_1_2_CAPABILITIES,
    gexfExporter,
    gexfImporter,
    gmlExporter,
    gmlImporter,
    type GraphExporter,
    type GraphImporter,
    graphmlExporter,
    graphmlImporter,
    ImportReportBuilder,
    JSON_DIALECTS,
    jsonCapabilities,
    type JsonDialect,
    jsonExporter,
    jsonImporter,
    LOSS,
    type LossNote,
    neo4jExporter,
    neo4jImporter,
    pajekExporter,
    pajekImporter,
} from "../../src/index.js";

// ============================================================ targets

/** One exporter / importer pair under one option set. */
interface Target {
    readonly name: string;
    readonly exporter: GraphExporter;
    readonly importer: GraphImporter;
    readonly caps: ExportCapabilities;
    /** Options every export of this target passes. */
    readonly exportOptions: Record<string, unknown>;
    /** Options every re-import of this target passes. */
    readonly importOptions: Record<string, unknown>;
    /** The table the attribute probes write to: CSV writes the edge table by default. */
    readonly attrDomain: "node" | "edge";
    /** Format-specific loss codes that stand in for a generic one (same gap, the format's own code). */
    readonly aliases: Readonly<Record<string, readonly string[]>>;
}

const JSON_TARGETS: Target[] = JSON_DIALECTS.map((dialect: JsonDialect) => ({
    name: `json:${dialect}`,
    exporter: jsonExporter,
    importer: jsonImporter,
    caps: jsonCapabilities(dialect),
    exportOptions: { dialect },
    importOptions: { dialect },
    attrDomain: "node",
    aliases: {},
}));

const TARGETS: readonly Target[] = [
    {
        name: "gexf:1.3",
        exporter: gexfExporter,
        importer: gexfImporter,
        caps: gexfExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
    {
        name: "gexf:1.2",
        exporter: gexfExporter,
        importer: gexfImporter,
        caps: GEXF_1_2_CAPABILITIES,
        exportOptions: { version: "1.2" },
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
    {
        name: "graphml",
        exporter: graphmlExporter,
        importer: graphmlImporter,
        caps: graphmlExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
    {
        name: "gml",
        exporter: gmlExporter,
        importer: gmlImporter,
        caps: gmlExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
    {
        name: "dot",
        exporter: dotExporter,
        importer: dotImporter,
        caps: dotExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
    {
        name: "pajek",
        exporter: pajekExporter,
        importer: pajekImporter,
        caps: pajekExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: { [LOSS.TEMPORAL]: ["W_TEMPORAL_DROPPED"] },
    },
    {
        name: "csv",
        exporter: csvExporter,
        importer: csvImporter,
        caps: csvExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "edge",
        aliases: {},
    },
    ...JSON_TARGETS,
    {
        name: "neo4j",
        exporter: neo4jExporter,
        importer: neo4jImporter,
        caps: neo4jExporter.capabilities,
        exportOptions: {},
        importOptions: {},
        attrDomain: "node",
        aliases: {},
    },
];

// ============================================================ helpers

const IDS: readonly number[] = [1, 2, 3, 4, 5];

/** A directed 5-cycle on integer ids 1..5 (every id charset holds it; five rows feed the dict heuristic). */
function cycleBuilder(directed = true): GraphBuilder {
    const b = new GraphBuilder({ directed, weightDtype: "f64" });
    for (const id of IDS) {
        b.addNode(id);
    }
    for (let i = 0; i < IDS.length; i++) {
        b.addEdge(IDS[i], IDS[(i + 1) % IDS.length]);
    }
    return b;
}

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

/** The generic code plus the target's aliases for it. */
function expectedCodes(target: Target, generic: readonly string[]): string[] {
    const out: string[] = [];
    for (const code of generic) {
        out.push(code, ...(target.aliases[code] ?? []));
    }
    return out;
}

async function reimport(
    target: Target,
    snapshot: GraphSnapshot,
    extraImport: Record<string, unknown> = {},
): Promise<GraphSnapshot> {
    const text = await target.exporter.exportToString(snapshot, target.exportOptions);
    const builder = new GraphBuilder({ directed: snapshot.directed, weightDtype: "f64" });
    await target.importer.import(text, builder, { ...target.importOptions, ...extraImport });
    return builder.freeze();
}

/** The column of a table by name, in whichever domain the probe wrote it. */
function attrColumn(
    snapshot: GraphSnapshot,
    domain: "node" | "edge",
    name: string,
): ReturnType<GraphSnapshot["nodes"]["get"]> {
    return domain === "node" ? snapshot.nodes.get(name) : snapshot.edges.get(name);
}

function declareAttr(b: GraphBuilder, domain: "node" | "edge", decl: ColumnDecl): void {
    if (domain === "node") {
        b.declareNodeColumn(decl);
    } else {
        b.declareEdgeColumn(decl);
    }
}

function setAttr(b: GraphBuilder, domain: "node" | "edge", name: string, index: number, value: unknown): void {
    if (domain === "node") {
        b.setNodeValue(name, index, value);
    } else {
        b.setEdgeValue(name, index, value);
    }
}

/** A probe snapshot with one attribute column of `decl` holding `values` on the target's attribute domain. */
function withColumn(domain: "node" | "edge", decl: ColumnDecl, values: readonly unknown[]): GraphSnapshot {
    const b = cycleBuilder();
    declareAttr(b, domain, decl);
    values.forEach((v, i) => {
        setAttr(b, domain, decl.name, i, v);
    });
    return b.freeze();
}

/** Same values, cell by cell, over the set rows of the original. */
function sameValues(
    original: GraphSnapshot,
    actual: GraphSnapshot,
    domain: "node" | "edge",
    name: string,
    tolerance = 0,
): boolean {
    const o = attrColumn(original, domain, name);
    const a = attrColumn(actual, domain, name);
    if (o === null || a === null || o.length !== a.length) {
        return false;
    }
    for (let r = 0; r < o.length; r++) {
        if (!o.isSet(r)) {
            continue;
        }
        if (!a.isSet(r)) {
            return false;
        }
        if (!valuesEqual(cell(o, r), cell(a, r), tolerance)) {
            return false;
        }
    }
    return true;
}

function cell(column: NonNullable<ReturnType<GraphSnapshot["nodes"]["get"]>>, row: number): unknown {
    switch (column.dtype) {
        case "list":
            return [...column.sliceOf(row)];
        case "json":
            return column.values[row];
        default: {
            const v = column.value(row);
            return ArrayBuffer.isView(v) ? Array.from(v as ArrayLike<number>) : v;
        }
    }
}

function valuesEqual(e: unknown, a: unknown, tolerance: number): boolean {
    if (typeof e === "number" && typeof a === "number") {
        return Object.is(e, a) || Math.abs(e - a) <= tolerance;
    }
    if (Array.isArray(e) && Array.isArray(a)) {
        return e.length === a.length && e.every((v, i) => valuesEqual(v, a[i], tolerance));
    }
    if (typeof e === "object" && e !== null && typeof a === "object" && a !== null) {
        const eo = e as Record<string, unknown>;
        const ao = a as Record<string, unknown>;
        const keys = Object.keys(eo);
        return keys.length === Object.keys(ao).length && keys.every((k) => valuesEqual(eo[k], ao[k], tolerance));
    }
    return Object.is(e, a);
}

/** How many source-undirected edges a directed snapshot carries (pairs once), through the public roles. */
function mixedCount(snapshot: GraphSnapshot): number {
    if (!snapshot.directed) {
        return 0;
    }
    const directed = snapshot.edges.byRole("directed");
    if (directed === null || directed.dtype !== "bool") {
        return 0;
    }
    const pair = snapshot.edges.byRole("pair");
    let undirected = 0;
    let paired = 0;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (directed.isSet(e) && directed.value(e) === false) {
            undirected++;
            if (pair !== null && pair.dtype === "u32" && pair.isSet(e) && pair.data[e] !== INVALID_INDEX) {
                paired++;
            }
        }
    }
    return paired / 2 + (undirected - paired);
}

// ============================================================ feature probes

/** One capability probe. */
interface Feature {
    readonly name: string;
    /** Whether the table declares the feature. */
    readonly declared: (caps: ExportCapabilities) => boolean;
    /** The generic loss codes check() must report when the feature is denied. */
    readonly lossCodes: readonly string[];
    /** Further codes that report the same gap for a given table (a denied dtype subsumes a denied option set). */
    readonly lossCodesFor?: (caps: ExportCapabilities) => readonly string[];
    /** The probe snapshot. */
    readonly build: (target: Target) => GraphSnapshot;
    /** Whether the re-imported snapshot still carries the feature. */
    readonly preserved: (original: GraphSnapshot, actual: GraphSnapshot, target: Target) => boolean;
    /** Extra import options the round trip needs. */
    readonly importOptions?: Record<string, unknown>;
}

function mixedSnapshot(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    for (const id of IDS) {
        b.addNode(id);
    }
    const r = new DirectionResolver(b, new ImportReportBuilder("probe", 100), "expand");
    r.setHeader(true);
    r.addEdge(1, 2, "directed");
    r.addEdge(2, 3, "undirected");
    r.addEdge(3, 4, "directed");
    return b.freeze();
}

const DTYPE_VALUES: Readonly<Record<string, readonly unknown[]>> = {
    f32: [0.5, 1.5, -2.25, 8, 0.125],
    f64: [0.1, 2.5, 1e-9, -3.75, 1e21],
    i32: [1, -2, 300, 0, 2147483647],
    u32: [1, 2, 4000000000, 0, 7],
    u8: [0, 7, 255, 1, 2],
    bool: [true, false, true, true, false],
    dict: ["x", "x", "x", "y", "x"],
    string: ["hello world", "b", "c", "d e", "f"],
};

function dtypeFeature(dtype: "f32" | "f64" | "i32" | "u32" | "u8" | "bool" | "dict" | "string"): Feature {
    const name = `attr_${dtype}`;
    return {
        name: `dtypes: ${dtype}`,
        declared: (caps) => caps.dtypes.includes(dtype),
        lossCodes: [LOSS.DTYPE],
        build: (t) => withColumn(t.attrDomain, { name, dtype }, DTYPE_VALUES[dtype]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, name);
            return col !== null && col.dtype === dtype && sameValues(o, a, t.attrDomain, name);
        },
    };
}

const FEATURES: readonly Feature[] = [
    {
        name: "mixedDirection",
        declared: (caps) => caps.mixedDirection,
        lossCodes: [LOSS.MIXED_DIRECTION_ERROR, LOSS.MIXED_DIRECTION],
        build: () => mixedSnapshot(),
        preserved: (o, a) => a.directed && mixedCount(a) === mixedCount(o) && a.edgeCount === o.edgeCount,
    },
    {
        name: "multiEdges",
        declared: (caps) => caps.multiEdges,
        lossCodes: [LOSS.MULTI_EDGES],
        build: () => {
            const b = cycleBuilder();
            b.addEdge(1, 2);
            return b.freeze();
        },
        preserved: (o, a) => a.edgeCount === o.edgeCount && a.flags.multigraph,
    },
    {
        name: "selfLoops",
        declared: (caps) => caps.selfLoops,
        lossCodes: [LOSS.SELF_LOOPS],
        build: () => {
            const b = cycleBuilder();
            b.addEdge(3, 3);
            return b.freeze();
        },
        preserved: (o, a) => a.edgeCount === o.edgeCount && a.selfLoopCount === 1,
    },
    {
        name: "edgeIds kept (optional / required)",
        declared: (caps) => caps.edgeIds !== "none",
        lossCodes: [LOSS.EDGE_IDS_DROPPED],
        build: () => {
            const b = cycleBuilder();
            b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
            for (let e = 0; e < IDS.length; e++) {
                b.setEdgeValue("id", e, `edge-${e}`);
            }
            return b.freeze();
        },
        preserved: (o, a) => {
            const col = a.edges.byRole("id");
            if (col === null) {
                return false;
            }
            const orig = o.edges.byRole("id");
            if (orig === null) {
                return false;
            }
            for (let e = 0; e < o.edgeCount; e++) {
                if (String(col.value(e)) !== String(orig.value(e))) {
                    return false;
                }
            }
            return true;
        },
    },
    {
        name: "edgeIds required: generated ids are reported",
        declared: (caps) => caps.edgeIds !== "required",
        lossCodes: [LOSS.EDGE_IDS_GENERATED],
        build: () => cycleBuilder().freeze(),
        preserved: () => true,
    },
    ...(["f32", "f64", "i32", "u32", "u8", "bool", "dict", "string"] as const).map(dtypeFeature),
    {
        name: "components",
        declared: (caps) => caps.components,
        lossCodes: [LOSS.COMPONENTS],
        build: (t) =>
            withColumn(t.attrDomain, { name: "vec", dtype: "f64", components: 2 }, [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, "vec");
            return col !== null && col.meta.components === 2 && sameValues(o, a, t.attrDomain, "vec");
        },
    },
    {
        name: "lists",
        declared: (caps) => caps.lists,
        lossCodes: [LOSS.LIST],
        build: (t) =>
            withColumn(t.attrDomain, { name: "tags", dtype: "list", itemDtype: "string" }, [
                ["a", "b"],
                ["c"],
                [],
                ["d", "e", "f"],
                ["g"],
            ]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, "tags");
            return col !== null && col.dtype === "list" && sameValues(o, a, t.attrDomain, "tags");
        },
    },
    {
        name: "json",
        declared: (caps) => caps.json,
        lossCodes: [LOSS.JSON],
        build: (t) =>
            withColumn(t.attrDomain, { name: "nested", dtype: "json" }, [
                { k: "v", n: [1, 2] },
                { k: "w" },
                { deep: { x: 1 } },
                { k: "z", n: [] },
                { k: "q" },
            ]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, "nested");
            return col !== null && col.dtype === "json" && sameValues(o, a, t.attrDomain, "nested");
        },
    },
    {
        name: "defaults",
        declared: (caps) => caps.defaults,
        lossCodes: [LOSS.DEFAULT],
        build: (t) => withColumn(t.attrDomain, { name: "withdefault", dtype: "string", default: "none" }, ["a", "b"]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, "withdefault");
            return col !== null && col.meta.default === "none" && sameValues(o, a, t.attrDomain, "withdefault");
        },
    },
    {
        name: "options (declared enumeration on a dict column)",
        declared: (caps) => caps.options,
        lossCodes: [LOSS.OPTIONS],
        // a format that cannot keep dict at all reports the dtype; the enumeration goes with it
        lossCodesFor: (caps) => (caps.dtypes.includes("dict") ? [] : [LOSS.DTYPE]),
        build: (t) =>
            withColumn(t.attrDomain, { name: "kind", dtype: "dict", options: ["red", "green", "blue"] }, [
                "red",
                "green",
                "red",
                "blue",
                "red",
            ]),
        preserved: (o, a, t) => {
            const col = attrColumn(a, t.attrDomain, "kind");
            return (
                col !== null &&
                col.meta.options !== null &&
                [...col.meta.options].join(",") === "red,green,blue" &&
                sameValues(o, a, t.attrDomain, "kind")
            );
        },
    },
    {
        name: "hierarchy (parent role)",
        declared: (caps) => caps.hierarchy,
        lossCodes: [LOSS.HIERARCHY],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
            b.setNodeValue("parent", 1, 0);
            b.setNodeValue("parent", 2, 0);
            b.setNodeValue("parent", 4, 3);
            return b.freeze();
        },
        preserved: (o, a) => {
            const col = a.nodes.byRole("parent");
            if (col === null || col.dtype !== "u32") {
                return false;
            }
            const orig = o.nodes.byRole("parent");
            if (orig === null) {
                return false;
            }
            for (let i = 0; i < o.nodeCount; i++) {
                const id = o.ids.idOf(i);
                const j = a.ids.indexOf(id);
                if (j === INVALID_INDEX || orig.isSet(i) !== col.isSet(j)) {
                    return false;
                }
                if (orig.isSet(i) && o.ids.idOf(Number(orig.value(i))) !== a.ids.idOf(Number(col.value(j)))) {
                    return false;
                }
            }
            return true;
        },
    },
    {
        name: "temporal: intervals (start / end roles)",
        // the temporal levels are not a ladder: "spells" (Pajek) carries element lifetimes as spells only
        declared: (caps) => caps.temporal === "intervals" || caps.temporal === "dynamic-values",
        lossCodes: [LOSS.TEMPORAL],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
            b.declareNodeColumn({ name: "end", dtype: "f64", role: "end" });
            for (let i = 0; i < IDS.length; i++) {
                b.setNodeValue("start", i, i * 10);
                b.setNodeValue("end", i, i * 10 + 5);
            }
            return b.freeze();
        },
        preserved: (o, a) => {
            const s = a.nodes.byRole("start");
            const e = a.nodes.byRole("end");
            return (
                s !== null &&
                e !== null &&
                sameValues(o, a, "node", s.meta.name === "start" ? "start" : s.meta.name) &&
                sameValues(o, a, "node", e.meta.name === "end" ? "end" : e.meta.name)
            );
        },
    },
    {
        name: "temporal: spells",
        declared: (caps) => caps.temporal === "spells" || caps.temporal === "dynamic-values",
        lossCodes: [LOSS.SPELLS],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "spells", dtype: "list", itemDtype: "f64", itemComponents: 2, role: "spells" });
            b.setNodeValue("spells", 0, [[1, 2]]);
            b.setNodeValue("spells", 1, [
                [1, 2],
                [4, 6],
            ]);
            b.setNodeValue("spells", 2, [[3, 4]]);
            return b.freeze();
        },
        preserved: (o, a) => {
            const col = a.nodes.byRole("spells");
            return col !== null && col.dtype === "list" && sameValues(o, a, "node", col.meta.name);
        },
    },
    {
        name: "temporal: dynamic values (temporal:* extension table)",
        declared: (caps) => caps.temporal === "dynamic-values",
        lossCodes: [LOSS.DYNAMIC_VALUES],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "price", dtype: "f64", dynamic: true });
            b.setNodeValue("price", 0, 1);
            // the design 5.10 shape: element (refersTo the domain), start, end, value
            const t = b.addExtensionTable("temporal:node:price", [
                { name: "element", dtype: "u32", refersTo: "node" },
                { name: "start", dtype: "f64", role: "start" },
                { name: "end", dtype: "f64", role: "end" },
                { name: "value", dtype: "f64" },
            ]);
            b.addExtensionRow(t, [0, 0, 5, 1]);
            b.addExtensionRow(t, [0, 5, 10, 2]);
            return b.freeze();
        },
        preserved: (_o, a) => {
            const table = a.extensions.get("temporal:node:price");
            return table !== undefined && table.rowCount === 2;
        },
    },
    {
        name: "graphAttributes",
        declared: (caps) => caps.graphAttributes,
        lossCodes: [LOSS.GRAPH_ATTRIBUTES],
        build: () => {
            const b = cycleBuilder();
            b.setGraphValue("title", "probe graph");
            b.setGraphValue("year", 2026);
            return b.freeze();
        },
        preserved: (_o, a) => {
            const title = a.graph.get("title");
            const year = a.graph.get("year");
            return (
                title !== null && title.value(0) === "probe graph" && year !== null && Number(year.value(0)) === 2026
            );
        },
    },
    {
        name: "positions (position role, f32 x3)",
        declared: (caps) => caps.positions,
        lossCodes: [LOSS.POSITIONS],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "position", dtype: "f32", components: 3, role: "position" });
            for (let i = 0; i < IDS.length; i++) {
                b.setNodeValue("position", i, [i, i * 2, i * 3]);
            }
            return b.freeze();
        },
        preserved: (o, a) => {
            const col = a.nodes.byRole("position");
            if (col === null) {
                return false;
            }
            const orig = o.nodes.byRole("position");
            if (orig === null) {
                return false;
            }
            for (let i = 0; i < o.nodeCount; i++) {
                const ov = Array.from(orig.value(i) as ArrayLike<number>);
                const av = Array.from(col.value(i) as ArrayLike<number>);
                if (ov.length < 2 || av.length < 2 || ov[0] !== av[0] || ov[1] !== av[1]) {
                    return false;
                }
            }
            return true;
        },
    },
    {
        name: "viz (color / size roles)",
        declared: (caps) => caps.viz,
        lossCodes: [LOSS.VIZ],
        build: () => {
            const b = cycleBuilder();
            b.declareNodeColumn({ name: "color", dtype: "f32", components: 4, role: "color" });
            b.declareNodeColumn({ name: "size", dtype: "f32", role: "size" });
            for (let i = 0; i < IDS.length; i++) {
                b.setNodeValue("color", i, [0.25, 0.5, 0.75, 1]);
                b.setNodeValue("size", i, i + 1);
            }
            return b.freeze();
        },
        preserved: (o, a) => {
            const size = a.nodes.byRole("size");
            const color = a.nodes.byRole("color");
            if (size === null || color === null) {
                return false;
            }
            for (let i = 0; i < o.nodeCount; i++) {
                if (Number(size.value(i)) !== i + 1) {
                    return false;
                }
                const c = Array.from(color.value(i) as ArrayLike<number>);
                if (c.length < 3 || Math.abs(c[0] - 0.25) > 0.01 || Math.abs(c[1] - 0.5) > 0.01) {
                    return false;
                }
            }
            return true;
        },
    },
];

// ============================================================ the matrix

describe("ExportCapabilities truthfulness (design 8.5 / 12.4)", () => {
    for (const target of TARGETS) {
        describe(target.name, () => {
            for (const feature of FEATURES) {
                const declared = feature.declared(target.caps);
                if (declared) {
                    it(`declares ${feature.name}: check() is silent and the round trip keeps it`, async () => {
                        const snapshot = feature.build(target);
                        const notes = target.exporter.check(snapshot, target.exportOptions);
                        const found = codes(notes).filter((c) => expectedCodes(target, feature.lossCodes).includes(c));
                        expect(found, `check() reported ${JSON.stringify(codes(notes))}`).toEqual([]);
                        const back = await reimport(target, snapshot, feature.importOptions ?? {});
                        expect(
                            feature.preserved(snapshot, back, target),
                            `round trip lost the feature; notes ${JSON.stringify(codes(notes))}`,
                        ).toBe(true);
                    });
                } else {
                    it(`denies ${feature.name}: check() reports it`, () => {
                        const snapshot = feature.build(target);
                        const notes = target.exporter.check(snapshot, target.exportOptions);
                        const wanted = expectedCodes(target, [
                            ...feature.lossCodes,
                            ...(feature.lossCodesFor?.(target.caps) ?? []),
                        ]);
                        const found = codes(notes).filter((c) => wanted.includes(c));
                        expect(
                            found,
                            `expected one of ${JSON.stringify(wanted)} in ${JSON.stringify(codes(notes))}`,
                        ).not.toEqual([]);
                    });
                }
            }
        });
    }
});

// ============================================================ id charsets

/** The ids a charset holds unchanged, and ids it cannot hold. */
const CHARSET_IDS: Readonly<
    Record<ExportCapabilities["idCharset"], { readonly ok: readonly NodeId[]; readonly bad: readonly NodeId[] }>
> = {
    any: { ok: ["a b", "1.0", 7, "x:y", "-"], bad: [] },
    nmtoken: { ok: ["a-b", "x.y", 7, "n:s", "_u"], bad: ["a b", "1.0"] },
    integer: { ok: [1, 2, 3, 10, -4], bad: ["a", "1.0"] },
    "dense-1-based": { ok: [1, 2, 3, 4, 5], bad: ["a", 9] },
};

describe("ExportCapabilities.idCharset truthfulness (design 8.5, Q26)", () => {
    for (const target of TARGETS) {
        const { idCharset } = target.caps;
        const { ok, bad } = CHARSET_IDS[idCharset];

        it(`${target.name}: ids within "${idCharset}" round-trip unchanged with no id note`, async () => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            for (const id of ok) {
                b.addNode(id);
            }
            for (let i = 0; i + 1 < ok.length; i++) {
                b.addEdge(ok[i], ok[i + 1]);
            }
            const snapshot = b.freeze();
            const notes = codes(target.exporter.check(snapshot, target.exportOptions));
            expect(
                notes.filter((c) => c === LOSS.ID_CHARSET || c === LOSS.ID_MANGLED || c === LOSS.ID_RENUMBERED),
            ).toEqual([]);
            const back = await reimport(target, snapshot);
            // JGF / Cytoscape key nodes by JSON object keys: numeric ids come back as text and integer-like
            // keys first; both are reported by the exporter (W_NUMERIC_IDS_STRINGIFIED, W_NODE_ORDER), and the
            // 12.4 idCharset enum has no "text" level, so the test accepts the reported form there.
            const reported = notes.includes("W_NUMERIC_IDS_STRINGIFIED") || notes.includes("W_NODE_ORDER");
            if (reported) {
                expect(back.ids.toArray().map(String).sort()).toEqual(ok.map(String).sort());
            } else {
                expect(back.ids.toArray()).toEqual(ok);
            }
        });

        if (bad.length === 0) {
            continue;
        }

        it(`${target.name}: ids outside "${idCharset}" are reported by check() and refused by export() under sanitizeIds "error"`, async () => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            for (const id of bad) {
                b.addNode(id);
            }
            b.addEdge(bad[0], bad[1]);
            const snapshot = b.freeze();
            const notes = codes(target.exporter.check(snapshot, target.exportOptions));
            if (idCharset === "dense-1-based") {
                expect(notes).toContain(LOSS.ID_RENUMBERED);
                // renumbering is not a mangling: export() succeeds and the file numbers 1..N
                const back = await reimport(target, snapshot);
                expect(back.ids.toArray()).toEqual([1, 2]);
                return;
            }
            expect(notes).toContain(LOSS.ID_CHARSET);
            await expect(target.exporter.exportToString(snapshot, target.exportOptions)).rejects.toMatchObject({
                code: "E_INVALID_ID",
                details: { reason: "charset" },
            });
        });

        it(`${target.name}: sanitizeIds "mangle" rewrites, check() reports it, restoreMangledIds brings the ids back`, async () => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            for (const id of bad) {
                b.addNode(id);
            }
            b.addEdge(bad[0], bad[1]);
            const snapshot = b.freeze();
            const opts = { ...target.exportOptions, sanitizeIds: "mangle" as const };
            const notes = codes(target.exporter.check(snapshot, opts));
            // "dense-1-based" is a renumbering (W_ID_RENUMBERED, whose message names the originalId
            // attribute under "mangle"); every other charset reports W_ID_MANGLED
            expect(notes).toContain(idCharset === "dense-1-based" ? LOSS.ID_RENUMBERED : LOSS.ID_MANGLED);
            expect(notes).not.toContain(LOSS.ID_CHARSET);
            const text = await target.exporter.exportToString(snapshot, opts);
            const restored = new GraphBuilder({ directed: true, weightDtype: "f64" });
            await target.importer.import(text, restored, { ...target.importOptions, restoreMangledIds: true });
            expect(restored.freeze().ids.toArray()).toEqual(bad);
            const kept = new GraphBuilder({ directed: true, weightDtype: "f64" });
            await target.importer.import(text, kept, { ...target.importOptions, restoreMangledIds: false });
            const mangled = kept.freeze();
            expect(mangled.ids.toArray()).not.toEqual(bad);
            // the original ids are kept in the exporter's attribute (design 8.5: graphty:originalId, by name)
            expect(
                [...mangled.nodes].some((c) => c.meta.role === "originalId" || /originalId/i.test(c.meta.name)),
            ).toBe(true);
        });
    }
});
