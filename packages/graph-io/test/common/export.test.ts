import { GraphBuilder, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../src/common/direction.js";
import {
    capabilities,
    checkCapabilities,
    countMixedEdges,
    countParallelArcs,
    countUnrepresentableIds,
    isNameChar,
    isNmtoken,
    isRepresentableId,
    LOSS,
    mangleNmtoken,
    NO_CAPABILITIES,
    sanitizeIds,
} from "../../src/common/export.js";
import { resolveExportOptions } from "../../src/common/options.js";
import { ImportReportBuilder } from "../../src/common/report.js";
import { type ExportCapabilities, type LossNote } from "../../src/types.js";

const ALL: ExportCapabilities = capabilities({
    mixedDirection: true,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "any",
    dtypes: ["f32", "f64", "i32", "u32", "u8", "bool", "dict", "string"],
    components: true,
    lists: true,
    json: true,
    defaults: true,
    options: true,
    hierarchy: true,
    temporal: "dynamic-values",
    graphAttributes: true,
    positions: true,
    viz: true,
});

const DEFAULTS = resolveExportOptions(undefined);

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

function simple(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true });
    b.addEdge("a", "b", 2);
    b.addEdge("b", "c");
    return b.freeze();
}

describe("capabilities()", () => {
    it("starts from NO_CAPABILITIES and freezes", () => {
        const c = capabilities({ selfLoops: true, dtypes: ["f64"] });
        expect(c).toEqual({ ...NO_CAPABILITIES, selfLoops: true, dtypes: ["f64"] });
        expect(Object.isFrozen(c)).toBe(true);
        expect(Object.isFrozen(c.dtypes)).toBe(true);
        expect(NO_CAPABILITIES.edgeIds).toBe("none");
        expect(NO_CAPABILITIES.temporal).toBe("none");
        expect(NO_CAPABILITIES.idCharset).toBe("any");
    });
});

describe("checkCapabilities (design 8.5)", () => {
    it("returns nothing for a plain graph in a format that keeps everything", () => {
        expect(checkCapabilities(simple(), ALL, DEFAULTS)).toEqual([]);
    });

    it("reports mixed direction per onMixedDirection", () => {
        const b = new GraphBuilder({ directed: true });
        const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
        r.setHeader(true);
        r.addEdge("a", "b", "directed");
        r.addEdge("b", "c", "undirected");
        r.addEdge("c", "c", "undirected");
        const s = b.freeze();
        expect(countMixedEdges(s)).toBe(2);
        const no = capabilities({ selfLoops: true, mixedDirection: false });
        const err = checkCapabilities(s, no, DEFAULTS);
        expect(codes(err)).toEqual([LOSS.MIXED_DIRECTION_ERROR]);
        expect(err[0].count).toBe(2);
        const fold = checkCapabilities(s, no, resolveExportOptions({ onMixedDirection: "directed" }));
        expect(codes(fold)).toEqual([LOSS.MIXED_DIRECTION]);
        expect(fold[0].message).toContain("directed");
        expect(checkCapabilities(s, ALL, DEFAULTS)).toEqual([]);
        expect(countMixedEdges(simple())).toBe(0);
        const undirected = new GraphBuilder({ directed: false });
        undirected.addEdge("a", "b");
        expect(countMixedEdges(undirected.freeze())).toBe(0);
    });

    it("reports parallel edges and self-loops", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("c", "c");
        const s = b.freeze();
        expect(countParallelArcs(s)).toBe(2);
        const notes = checkCapabilities(s, capabilities({}), DEFAULTS);
        expect(codes(notes)).toEqual([LOSS.MULTI_EDGES, LOSS.SELF_LOOPS]);
        expect(notes[0].count).toBe(2);
        expect(notes[1].count).toBe(1);
        const u = new GraphBuilder({ directed: false });
        u.addEdge("a", "b");
        u.addEdge("b", "a");
        expect(countParallelArcs(u.freeze())).toBe(1);
        expect(codes(checkCapabilities(s, capabilities({ multiEdges: true, selfLoops: true }), DEFAULTS))).toEqual([]);
    });

    it("reports edge ids generated or dropped", () => {
        const s = simple();
        expect(codes(checkCapabilities(s, capabilities({ edgeIds: "required" }), DEFAULTS))).toEqual([
            LOSS.EDGE_IDS_GENERATED,
        ]);
        expect(codes(checkCapabilities(s, capabilities({ edgeIds: "optional" }), DEFAULTS))).toEqual([]);
        const b = new GraphBuilder({ directed: true });
        const h = b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
        b.setEdgeValue(h, b.addEdge("a", "b"), "e0");
        const withIds = b.freeze();
        const dropped = checkCapabilities(withIds, capabilities({ edgeIds: "none" }), DEFAULTS);
        expect(codes(dropped)).toEqual([LOSS.EDGE_IDS_DROPPED]);
        expect(dropped[0].column).toBe("id");
        expect(
            codes(checkCapabilities(withIds, capabilities({ edgeIds: "required", dtypes: ["string"] }), DEFAULTS)),
        ).toEqual([]);
    });

    it("reports ids outside the charset per sanitizeIds", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("Little Rock, AR", "n2");
        b.addEdge("n2", 3);
        const s = b.freeze();
        const nm = capabilities({ idCharset: "nmtoken" });
        expect(codes(checkCapabilities(s, nm, DEFAULTS))).toEqual([LOSS.ID_CHARSET]);
        expect(codes(checkCapabilities(s, nm, resolveExportOptions({ sanitizeIds: "mangle" })))).toEqual([
            LOSS.ID_MANGLED,
        ]);
        expect(checkCapabilities(s, nm, DEFAULTS)[0].count).toBe(1);
        const int = capabilities({ idCharset: "integer" });
        expect(checkCapabilities(s, int, DEFAULTS)[0].count).toBe(2);
        const dense = capabilities({ idCharset: "dense-1-based" });
        const renumbered = checkCapabilities(s, dense, DEFAULTS);
        expect(codes(renumbered)).toEqual([LOSS.ID_RENUMBERED]);
        expect(renumbered[0].count).toBe(2);
        const oneBased = new GraphBuilder({ directed: true });
        oneBased.addEdge(1, 2);
        oneBased.addEdge(2, 3);
        expect(checkCapabilities(oneBased.freeze(), dense, DEFAULTS)).toEqual([]);
    });

    it("reports column dtypes, lists, json, components, defaults and options", () => {
        const b = new GraphBuilder({ directed: true });
        const n = b.addNode("a");
        b.addNode("b");
        const hf = b.declareNodeColumn({ name: "f", dtype: "f64", default: 1.5 });
        const hl = b.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "i32" });
        const hj = b.declareNodeColumn({ name: "j", dtype: "json" });
        const hp = b.declareNodeColumn({ name: "p", dtype: "f32", components: 2 });
        const hk = b.declareNodeColumn({ name: "k", dtype: "i32", options: [1, 2] });
        const hd = b.declareNodeColumn({ name: "d", dtype: "dict", options: ["x", "y"] });
        b.setNodeValue(hf, n, 2);
        b.setNodeValue(hl, n, [1, 2]);
        b.setNodeValue(hj, n, { a: 1 });
        b.setNodeValue(hp, n, [1, 2]);
        b.setNodeValue(hk, n, 1);
        b.setNodeValue(hd, n, "x");
        const s = b.freeze();
        const notes = checkCapabilities(s, capabilities({ dtypes: ["f32", "i32", "string"] }), DEFAULTS);
        expect(codes(notes).sort()).toEqual(
            [
                LOSS.DTYPE, // f is f64
                LOSS.DEFAULT, // f declares a default
                LOSS.LIST, // l
                LOSS.JSON, // j
                LOSS.COMPONENTS, // p
                LOSS.OPTIONS, // k
                LOSS.DTYPE, // d is dict
                LOSS.OPTIONS, // d declares options too (a dict's enumeration is dropped like any other)
            ].sort(),
        );
        const f = notes.find((x) => x.column === "f" && x.code === LOSS.DTYPE);
        expect(f?.count).toBe(1);
        const listItem = checkCapabilities(
            s,
            capabilities({
                lists: true,
                dtypes: ["f64", "dict", "f32"],
                components: true,
                defaults: true,
                options: true,
                json: true,
            }),
            DEFAULTS,
        );
        expect(codes(listItem)).toEqual([LOSS.DTYPE, LOSS.DTYPE]);
        expect(listItem.map((x) => x.column)).toEqual(["l", "k"]);
        expect(listItem[0].message).toContain("i32 items");
        expect(checkCapabilities(s, ALL, DEFAULTS)).toEqual([]);
    });

    it("reports roles: positions, viz, hierarchy, temporal, spells, open intervals, dynamic", () => {
        const b = new GraphBuilder({ directed: true });
        const n = b.addNode("a");
        const e = b.addEdge("a", "b");
        const set = (name: string, decl: Parameters<GraphBuilder["declareNodeColumn"]>[0], value: unknown): void => {
            b.setNodeValue(b.declareNodeColumn({ ...decl, name }), n, value);
        };
        set("pos", { name: "", dtype: "f32", components: 3, role: "position" }, [1, 2, 3]);
        set("color", { name: "", dtype: "u8", components: 4, role: "color" }, [1, 2, 3, 4]);
        set("size", { name: "", dtype: "f32", role: "size" }, 2);
        set("parent", { name: "", dtype: "u32", role: "parent", refersTo: "node" }, 0);
        set("start", { name: "", dtype: "f64", role: "start" }, 1);
        set("spells", { name: "", dtype: "list", itemDtype: "f64", itemComponents: 2, role: "spells" }, [[1, 2]]);
        set("open", { name: "", dtype: "u8", role: "open" }, 1);
        set("dyn", { name: "", dtype: "f64", dynamic: true }, 1);
        b.setEdgeValue(b.declareEdgeColumn({ name: "end", dtype: "f64", role: "end" }), e, 2);
        const s = b.freeze();
        const none = checkCapabilities(
            s,
            capabilities({ dtypes: ["f64", "f32", "u8", "u32"], components: true }),
            DEFAULTS,
        );
        expect(codes(none).sort()).toEqual(
            [
                LOSS.POSITIONS,
                LOSS.VIZ,
                LOSS.VIZ,
                LOSS.HIERARCHY,
                LOSS.TEMPORAL,
                LOSS.SPELLS,
                LOSS.OPEN_INTERVAL,
                LOSS.DYNAMIC_VALUES,
                LOSS.TEMPORAL,
            ].sort(),
        );
        const intervals = checkCapabilities(
            s,
            capabilities({
                dtypes: ["f64", "f32", "u8", "u32"],
                components: true,
                temporal: "intervals",
                positions: true,
                viz: true,
                hierarchy: true,
            }),
            DEFAULTS,
            { openIntervals: true },
        );
        expect(codes(intervals).sort()).toEqual([LOSS.SPELLS, LOSS.DYNAMIC_VALUES].sort());
        expect(checkCapabilities(s, ALL, DEFAULTS, { openIntervals: true })).toEqual([]);
        expect(codes(checkCapabilities(s, ALL, DEFAULTS))).toEqual([LOSS.OPEN_INTERVAL]);
    });

    it("reports graph attributes and extension tables", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.setGraphValue("name", "g");
        const t = b.addExtensionTable("temporal:node:price", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "end", dtype: "f64" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [0, 1, 2, 3.5]);
        const other = b.addExtensionTable("custom:thing", [{ name: "x", dtype: "i32" }]);
        b.addExtensionRow(other, [1]);
        const s = b.freeze();
        const notes = checkCapabilities(s, capabilities({ dtypes: ["string"] }), DEFAULTS);
        expect(codes(notes).sort()).toEqual([LOSS.GRAPH_ATTRIBUTES, LOSS.DYNAMIC_VALUES, LOSS.EXTENSION_TABLE].sort());
        expect(notes.find((x) => x.code === LOSS.DYNAMIC_VALUES)?.count).toBe(1);
        const kept = checkCapabilities(s, ALL, DEFAULTS);
        expect(codes(kept)).toEqual([LOSS.EXTENSION_TABLE]);
        // graph columns are checked like the others when graph attributes are supported
        const typed = checkCapabilities(
            s,
            capabilities({ graphAttributes: true, dtypes: ["f64"], temporal: "dynamic-values" }),
            DEFAULTS,
        );
        expect(codes(typed).sort()).toEqual([LOSS.DTYPE, LOSS.EXTENSION_TABLE].sort());
        expect(typed.find((x) => x.code === LOSS.DTYPE)?.column).toBe("name");
    });

    it("skips the structural roles and the edge id column", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
        r.setHeader(true);
        r.addEdge("a", "b", "directed", 0.1);
        r.addEdge("b", "c", "undirected");
        r.addEdge("c", "d", "mutual");
        const s = b.freeze();
        expect(s.edges.byRole("weight")).not.toBeNull();
        const notes = checkCapabilities(s, capabilities({ mixedDirection: true, edgeIds: "optional" }), DEFAULTS);
        expect(notes).toEqual([]);
    });
});

describe("NMTOKEN", () => {
    it("isNameChar covers the XML NameChar ranges", () => {
        for (const ch of "aZ09_-.:") {
            expect(isNameChar(ch.codePointAt(0) as number), ch).toBe(true);
        }
        for (const ch of " ,;/\\\"'()!?@#") {
            expect(isNameChar(ch.codePointAt(0) as number), ch).toBe(false);
        }
        expect(isNameChar(0xb7)).toBe(true);
        expect(isNameChar(0xe9)).toBe(true);
        expect(isNameChar(0x300)).toBe(true);
        expect(isNameChar(0x4e2d)).toBe(true);
        expect(isNameChar(0x1f600)).toBe(true);
        expect(isNameChar(0x10000)).toBe(true);
        expect(isNameChar(0xf0000)).toBe(false);
        expect(isNameChar(0x2000)).toBe(false);
        expect(isNameChar(0xfffe)).toBe(false);
        expect(isNameChar(0xd800)).toBe(false);
    });

    it("isNmtoken", () => {
        expect(isNmtoken("n1")).toBe(true);
        expect(isNmtoken("1")).toBe(true);
        expect(isNmtoken("-1.5")).toBe(true);
        expect(isNmtoken("a:b")).toBe(true);
        expect(isNmtoken("")).toBe(false);
        expect(isNmtoken("Little Rock, AR")).toBe(false);
        expect(isNmtoken("1e+21")).toBe(false);
        expect(isNmtoken(`caf${String.fromCodePoint(0xe9)}`)).toBe(true);
    });

    it("mangleNmtoken replaces every other character with an underscore", () => {
        expect(mangleNmtoken("Little Rock, AR")).toBe("Little_Rock__AR");
        expect(mangleNmtoken("")).toBe("_");
        expect(mangleNmtoken("  ")).toBe("__");
        expect(mangleNmtoken("ok")).toBe("ok");
        expect(mangleNmtoken(`a${String.fromCodePoint(0x2000)}b`)).toBe("a_b");
        expect(mangleNmtoken(`a${String.fromCodePoint(0x1f600)}b`)).toBe(`a${String.fromCodePoint(0x1f600)}b`);
    });
});

describe("sanitizeIds (design 8.5)", () => {
    function graph(ids: (string | number)[]): GraphSnapshot {
        const b = new GraphBuilder({ directed: true });
        for (const id of ids) {
            b.addNode(id);
        }
        return b.freeze();
    }

    it("isRepresentableId per charset", () => {
        expect(isRepresentableId("a b", "any")).toBe(true);
        expect(isRepresentableId("a b", "nmtoken")).toBe(false);
        expect(isRepresentableId(1.5, "nmtoken")).toBe(true);
        expect(isRepresentableId(1e21, "nmtoken")).toBe(false);
        expect(isRepresentableId(3, "integer")).toBe(true);
        expect(isRepresentableId(1.5, "integer")).toBe(false);
        expect(isRepresentableId("3", "integer")).toBe(false);
        expect(isRepresentableId(1, "dense-1-based")).toBe(false);
        expect(() => isRepresentableId(1, "weird" as "any")).toThrow(GraphFormatError);
    });

    it("leaves representable ids alone", () => {
        const s = graph(["a", "b", 3]);
        for (const charset of ["any", "nmtoken"] as const) {
            const ids = sanitizeIds(s, charset, "error");
            expect(ids.changed).toBe(0);
            expect([0, 1, 2].map((i) => ids.idAt(i))).toEqual(["a", "b", 3]);
            expect(ids.isChanged(1)).toBe(false);
            expect(ids.originalAt(2)).toBe(3);
        }
        expect(countUnrepresentableIds(s, "any")).toBe(0);
    });

    it("throws E_INVALID_ID under error when an id does not fit", () => {
        const s = graph(["ok", "not ok", "also bad"]);
        let caught: unknown;
        try {
            sanitizeIds(s, "nmtoken", "error");
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        const err = caught as GraphFormatError;
        expect(err.code).toBe("E_INVALID_ID");
        expect(err.details).toMatchObject({ reason: "charset", charset: "nmtoken", count: 2, id: "not ok", index: 1 });
        expect(() => sanitizeIds(graph(["x"]), "integer", "error")).toThrow(GraphFormatError);
    });

    it("mangles NMTOKENs deterministically and uniquely", () => {
        const s = graph(["a b", "a_b", "a c", "a b", "a_b_2", "x"].filter((v, i, arr) => arr.indexOf(v) === i));
        // ids: "a b", "a_b", "a c", "a_b_2", "x"
        const ids = sanitizeIds(s, "nmtoken", "mangle");
        expect(ids.changed).toBe(2);
        expect([0, 1, 2, 3, 4].map((i) => ids.idAt(i))).toEqual(["a_b_3", "a_b", "a_c", "a_b_2", "x"]);
        expect(ids.isChanged(0)).toBe(true);
        expect(ids.isChanged(1)).toBe(false);
        expect(ids.originalAt(0)).toBe("a b");
        expect(new Set([0, 1, 2, 3, 4].map((i) => ids.idAt(i))).size).toBe(5);
        expect(countUnrepresentableIds(s, "nmtoken")).toBe(2);
    });

    it("assigns unused integers under integer / mangle", () => {
        const s = graph(["x", 0, "y", 2, 1.5]);
        const ids = sanitizeIds(s, "integer", "mangle");
        expect([0, 1, 2, 3, 4].map((i) => ids.idAt(i))).toEqual([1, 0, 3, 2, 4]);
        expect(ids.changed).toBe(3);
        expect(ids.isChanged(4)).toBe(true);
        expect(countUnrepresentableIds(s, "integer")).toBe(3);
    });

    it("renumbers 1..N under dense-1-based whatever the mode, keeping the originals", () => {
        const s = graph(["a", 2, 3]);
        for (const mode of ["error", "mangle"] as const) {
            const ids = sanitizeIds(s, "dense-1-based", mode);
            expect([0, 1, 2].map((i) => ids.idAt(i))).toEqual([1, 2, 3]);
            expect([0, 1, 2].map((i) => ids.isChanged(i))).toEqual([true, false, false]);
            expect(ids.changed).toBe(1);
            expect(ids.originalAt(0)).toBe("a");
        }
        const exact = graph([1, 2, 3]);
        expect(sanitizeIds(exact, "dense-1-based", "error").changed).toBe(0);
        expect(exact.ids.kind).toBe("identity");
        expect(countUnrepresentableIds(exact, "dense-1-based")).toBe(0);
    });
});
