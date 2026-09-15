/**
 * Unit tests of the staging representation and its compaction (design sections 6.2 and 6.3 step 1):
 * StagingColumn storage per dtype, inference and widening, gathering with refersTo rewriting,
 * freezing into columns; Staging push / kill / incidence lists / explicit-weight tracking; the
 * remap helpers and compactStaging.
 */

import { describe, expect, it } from "vitest";

import {
    compactStaging,
    createStagingColumn,
    inferInitialDtype,
    inferredEquivalent,
    remapDropping,
    Staging,
    StagingColumn,
} from "../../src/builder/compact.js";
import { resolveColumnMeta } from "../../src/columns/column.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { isIdentity } from "../../src/snapshot/validate.js";
import { type Dtype, type F32Column } from "../../src/types/index.js";

function code(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return err.code;
        }
        throw err;
    }
    return null;
}

function stagingOptions(
    weighted = false,
    weightDtype: "f32" | "f64" = "f32",
): ConstructorParameters<typeof Staging>[0] {
    return { weightDtype, weighted, expectedNodes: null, expectedEdges: null, resizable: undefined };
}

describe("inference helpers", () => {
    it("inferInitialDtype follows the typeof rules and refuses unset values", () => {
        expect(inferInitialDtype(true)).toBe("bool");
        expect(inferInitialDtype(3)).toBe("i32");
        expect(inferInitialDtype(2.5)).toBe("f64");
        expect(inferInitialDtype(2 ** 40)).toBe("f64");
        expect(inferInitialDtype("x")).toBe("string");
        expect(inferInitialDtype([])).toBe("json");
        expect(code(() => inferInitialDtype(null))).toBe("E_COLUMN_TYPE");
        expect(code(() => inferInitialDtype(undefined))).toBe("E_COLUMN_TYPE");
        expect(code(() => inferInitialDtype(Symbol("s")))).toBe("E_COLUMN_TYPE");
    });

    it("inferredEquivalent maps every dtype into the widening order", () => {
        const expected: Record<Dtype, string> = {
            bool: "bool",
            i32: "i32",
            u8: "i32",
            u32: "f64",
            f32: "f64",
            f64: "f64",
            dict: "string",
            string: "string",
            list: "json",
            json: "json",
        };
        for (const [dtype, target] of Object.entries(expected)) {
            expect(inferredEquivalent(dtype as Dtype)).toBe(target);
        }
        expect(code(() => inferredEquivalent("nope" as Dtype))).toBe("E_COLUMN_TYPE");
    });
});

describe("StagingColumn: declared columns", () => {
    it("materialises rows lazily with the fill and tracks validity", () => {
        const column = createStagingColumn("x", "node", { dtype: "f64", default: 7 }, false);
        expect(column.length).toBe(0);
        expect(column.isSet(0)).toBe(false);
        column.write(2, 1.5);
        expect(column.length).toBe(3);
        expect(column.isSet(0)).toBe(false);
        expect(column.isSet(2)).toBe(true);
        expect(column.read(0)).toBeUndefined();
        expect(column.read(2)).toBe(1.5);
        expect(column.typed?.get(0)).toBe(7);
        column.unset(2);
        expect(column.isSet(2)).toBe(false);
        expect(column.typed?.get(2)).toBe(7);
        expect(column.write(0, null)).toBeNull();
        expect(column.isSet(0)).toBe(false);
        const frozen = column.toColumn(4);
        expect(frozen.length).toBe(4);
        expect(frozen.nullCount).toBe(4);
        expect(frozen.dtype).toBe("f64");
        expect(frozen.value(1)).toBe(7);
        expect(column.byteLength()).toBeGreaterThan(0);
    });

    it("stores multi-component rows, dict codes, bool bits, strings, lists and json", () => {
        const pos = createStagingColumn("pos", "node", { dtype: "f32", components: 2 }, false);
        pos.write(1, [3, 4]);
        pos.write(0, 9);
        expect(pos.read(1)).toEqual([3, 4]);
        expect(pos.read(0)).toEqual([9, 9]);
        expect(Array.from((pos.toColumn(2) as F32Column).data)).toEqual([9, 9, 3, 4]);

        const dict = createStagingColumn("cat", "node", { dtype: "dict", options: ["a"], fill: "z" }, false);
        dict.write(0, "b");
        dict.write(1, "a");
        dict.write(2, 5);
        expect(dict.read(0)).toBe("b");
        expect(dict.read(2)).toBe("5");
        const dictColumn = dict.toColumn(4);
        expect(dictColumn.dtype).toBe("dict");
        if (dictColumn.dtype === "dict") {
            expect(dictColumn.dictionary).toEqual(["a", "z", "b", "5"]);
            expect(Array.from(dictColumn.codes)).toEqual([2, 0, 3, 1]);
            expect(dictColumn.dictionary).not.toBe(dict.dictionary?.values);
        }

        const flags = createStagingColumn("flag", "node", { dtype: "bool", fill: true }, false);
        flags.write(0, false);
        flags.write(2, true);
        expect(flags.read(0)).toBe(false);
        expect(flags.bits?.get(1)).toBe(true);
        const boolColumn = flags.toColumn(3);
        expect(boolColumn.value(1)).toBeUndefined();
        expect(boolColumn.value(2)).toBe(true);

        const names = createStagingColumn("name", "node", { dtype: "string", fill: "-" }, false);
        names.write(1, "b");
        names.write(0, 12);
        expect(names.values).toEqual(["12", "b"]);
        names.unset(0);
        expect(names.values).toEqual(["-", "b"]);
        expect(names.toColumn(2).value(1)).toBe("b");

        const lists = createStagingColumn("tags", "node", { dtype: "list", itemDtype: "string" }, false);
        const items = ["a"];
        lists.write(0, items);
        items.push("mutated later");
        expect(lists.read(0)).toEqual(["a"]);
        expect(lists.toColumn(2).value(0)).toEqual(["a"]);

        const json = createStagingColumn("blob", "node", { dtype: "json" }, false);
        json.write(0, null);
        json.write(1, { a: [1] });
        expect(json.isSet(0)).toBe(true);
        expect(json.read(0)).toBeNull();
        const jsonColumn = json.toColumn(3);
        expect(jsonColumn.isSet(0)).toBe(true);
        expect(jsonColumn.value(1)).toEqual({ a: [1] });
        expect(jsonColumn.isSet(2)).toBe(false);
    });

    it("freezes non-nullable JS-array columns with their fills", () => {
        const names = createStagingColumn("name", "node", { dtype: "string", nullable: false }, false);
        names.write(1, "b");
        const column = names.toColumn(2);
        expect(column.validity).toBeNull();
        expect(column.value(0)).toBe("");
        const lists = createStagingColumn("tags", "node", { dtype: "list", itemDtype: "i32", nullable: false }, false);
        lists.write(0, [1]);
        expect(lists.toColumn(2).value(1)).toEqual([]);
        const json = createStagingColumn("blob", "node", { dtype: "json", nullable: false }, false);
        json.write(0, 1);
        const frozen = json.toColumn(2);
        expect(frozen.value(1)).toBeNull();
        expect(code(() => names.unset(0))).toBe("E_COLUMN_TYPE");
    });

    it("rejects values a declared column cannot hold", () => {
        const u8 = createStagingColumn("u", "node", { dtype: "u8" }, false);
        expect(code(() => u8.write(0, 256))).toBe("E_COLUMN_TYPE");
        expect(code(() => u8.write(0, -1))).toBe("E_COLUMN_TYPE");
        expect(code(() => u8.write(0, "1"))).toBe("E_COLUMN_TYPE");
        const pos = createStagingColumn("pos", "node", { dtype: "i32", components: 2 }, false);
        expect(code(() => pos.write(0, [1.5, 2]))).toBe("E_COLUMN_TYPE");
        expect(code(() => pos.write(0, "ab"))).toBe("E_COLUMN_TYPE");
        const lists = createStagingColumn("l", "node", { dtype: "list", itemDtype: "u8" }, false);
        expect(code(() => lists.write(0, [300]))).toBe("E_COLUMN_TYPE");
        const pairs = createStagingColumn("p", "node", { dtype: "list", itemDtype: "f64", itemComponents: 2 }, false);
        expect(code(() => pairs.write(0, [[1, "x"]]))).toBe("E_COLUMN_TYPE");
        expect(code(() => pairs.write(0, [1]))).toBe("E_COLUMN_TYPE");
        const bools = createStagingColumn("b", "node", { dtype: "list", itemDtype: "bool" }, false);
        expect(code(() => bools.write(0, [1]))).toBe("E_COLUMN_TYPE");
        const dicts = createStagingColumn("d", "node", { dtype: "list", itemDtype: "dict" }, false);
        expect(code(() => dicts.write(0, [1]))).toBe("E_COLUMN_TYPE");
        expect(code(() => dicts.write(0, ["\ud800"]))).toBe("E_COLUMN_TYPE");
        const jsons = createStagingColumn("j", "node", { dtype: "list", itemDtype: "json" }, false);
        expect(code(() => jsons.write(0, [() => 1]))).toBe("E_COLUMN_TYPE");
        jsons.write(0, [{ ok: true }]);
        const strings = createStagingColumn("s", "node", { dtype: "string" }, false);
        expect(code(() => strings.write(0, "\ud800"))).toBe("E_COLUMN_TYPE");
    });
});

describe("StagingColumn: inferred columns", () => {
    it("keeps raw values, widens monotonically and coerces once at freeze", () => {
        const column = createStagingColumn("v", "node", { dtype: "bool", nullable: true }, true);
        expect(column.inferred).toBe(true);
        expect(column.inferredDtype).toBe("bool");
        expect(column.write(0, true)).toBeNull();
        expect(column.write(1, 2)).toEqual({ from: "bool", to: "i32" });
        expect(column.read(0)).toBe(1);
        expect(column.write(2, "x")).toEqual({ from: "i32", to: "string" });
        expect(column.read(0)).toBe("true");
        expect(column.write(3, false)).toBeNull();
        expect(column.write(4, null)).toBeNull();
        expect(column.isSet(4)).toBe(false);
        expect(column.byteLength()).toBeGreaterThan(0);
        const frozen = column.toColumn(6);
        expect(frozen.dtype).toBe("string");
        expect([0, 1, 2, 3].map((r) => frozen.value(r))).toEqual(["true", "2", "x", "false"]);
        expect(frozen.isSet(5)).toBe(false);
        expect(column.write(0, { any: 1 })).toEqual({ from: "string", to: "json" });
        expect(column.toColumn(6).value(3)).toBe(false);
    });

    it("validates raw strings and json values at the write", () => {
        const column = createStagingColumn("v", "node", { dtype: "string", nullable: true }, true);
        expect(code(() => column.write(0, "\ud800"))).toBe("E_COLUMN_TYPE");
        expect(code(() => column.write(0, { f: () => 1 }))).toBe("E_COLUMN_TYPE");
        expect(code(() => column.write(0, 1n))).toBe("E_COLUMN_TYPE");
    });

    it("widenTo turns a declared column into an inferred one keeping its values and validity", () => {
        const column = createStagingColumn("v", "node", { dtype: "u8", role: "size" }, false);
        column.write(0, 3);
        column.write(2, 4);
        column.widenTo("string");
        expect(column.inferred).toBe(true);
        expect(column.meta.dtype).toBe("string");
        expect(column.meta.role).toBe("size");
        expect(column.typed).toBeNull();
        expect(column.isSet(1)).toBe(false);
        expect(column.read(0)).toBe("3");
        column.write(1, 2.5);
        const frozen = column.toColumn(3);
        expect([0, 1, 2].map((r) => frozen.value(r))).toEqual(["3", "2.5", "4"]);
    });
});

describe("StagingColumn: gather", () => {
    it("gathers rows through an index map and rewrites refersTo values", () => {
        const parent = createStagingColumn("parent", "node", { dtype: "u32", refersTo: "node" }, false);
        parent.write(0, 2);
        parent.write(1, 0);
        parent.write(3, 1);
        const remap = new Uint32Array([0, INVALID_INDEX, 1, 2]);
        const gathered = parent.gather(new Uint32Array([0, 2, 3]), remap);
        expect(gathered.length).toBe(3);
        expect(gathered.read(0)).toBe(1);
        expect(gathered.isSet(1)).toBe(false);
        expect(gathered.typed?.get(1)).toBe(INVALID_INDEX);
        expect(gathered.isSet(2)).toBe(false);
        const plain = createStagingColumn("x", "node", { dtype: "f32", components: 2 }, false);
        plain.write(0, [1, 2]);
        plain.write(1, [3, 4]);
        const reversed = plain.gather(new Uint32Array([1, 0]), null);
        expect(reversed.read(0)).toEqual([3, 4]);
        expect(reversed.read(1)).toEqual([1, 2]);
        const bits = createStagingColumn("b", "node", { dtype: "bool" }, false);
        bits.write(1, true);
        expect(bits.gather(new Uint32Array([1, 0]), null).read(0)).toBe(true);
    });

    it("gathers list references, dropping dangling items and unsetting emptied rows", () => {
        const parents = createStagingColumn(
            "parents",
            "node",
            { dtype: "list", itemDtype: "u32", refersTo: "node" },
            false,
        );
        parents.write(0, [1, 2]);
        parents.write(1, [1]);
        parents.write(2, []);
        const remap = new Uint32Array([0, INVALID_INDEX, 1]);
        const gathered = parents.gather(null, remap);
        expect(gathered.read(0)).toEqual([1]);
        expect(gathered.isSet(1)).toBe(false);
        expect(gathered.read(2)).toEqual([]);
        const raw = createStagingColumn("raw", "node", { dtype: "string", nullable: true }, true);
        raw.write(1, "x");
        const copy = raw.gather(new Uint32Array([1, 1, 0]), null);
        expect(copy.read(0)).toBe("x");
        expect(copy.read(1)).toBe("x");
        expect(copy.isSet(2)).toBe(false);
    });

    it("shares the dictionary of a dict column", () => {
        const dict = createStagingColumn("d", "node", { dtype: "dict" }, false);
        dict.write(0, "a");
        dict.write(1, "b");
        const gathered = dict.gather(new Uint32Array([1]), null);
        expect(gathered.dictionary).toBe(dict.dictionary);
        expect(gathered.read(0)).toBe("b");
    });
});

describe("Staging", () => {
    it("pushes nodes and edges, links both incidence lists and counts loops", () => {
        const staging = new Staging(stagingOptions());
        expect(staging.pushNode(null)).toBe(0);
        expect(staging.pushNode(null)).toBe(1);
        expect(staging.ids).toBeNull();
        staging.materialiseIds();
        expect(staging.ids).toEqual([0, 1]);
        expect(staging.idToIndex?.get(1)).toBe(1);
        expect(staging.pushNode("x")).toBe(2);
        expect(staging.pushNode(null)).toBe(3);
        expect(staging.ids).toEqual([0, 1, "x", 3]);
        staging.materialiseIds();
        expect(staging.pushEdge(0, 1, undefined)).toBe(0);
        expect(staging.pushEdge(1, 1, undefined)).toBe(1);
        expect(staging.pushEdge(0, 1, 2)).toBe(2);
        expect(staging.weight).not.toBeNull();
        expect(staging.weight?.get(0)).toBe(1);
        expect(staging.weight?.get(2)).toBe(2);
        expect(staging.weightMode).toBe("mixed");
        expect(staging.weightSet?.length).toBe(3);
        expect(staging.weightExplicit(2)).toBe(true);
        expect(staging.weightExplicit(0)).toBe(false);
        expect(staging.selfLoopCount).toBe(1);
        expect(staging.liveEdgeCount).toBe(3);
        expect(Array.from(staging.outEdges(0))).toEqual([0, 2]);
        expect(Array.from(staging.inEdges(1))).toEqual([0, 1, 2]);
        staging.killEdge(1);
        expect(staging.selfLoopCount).toBe(0);
        expect(staging.hasTombstones).toBe(true);
        expect(Array.from(staging.inEdges(1))).toEqual([0, 2]);
        expect(staging.byteLength()).toBeGreaterThan(0);
        staging.reserve(100, 100);
        expect(staging.src.capacity).toBeGreaterThanOrEqual(100);
    });

    it("tracks the explicit-weight mode through every transition", () => {
        const explicitFirst = new Staging(stagingOptions());
        explicitFirst.pushNode(null);
        explicitFirst.pushEdge(0, 0, 2);
        expect(explicitFirst.weightMode).toBe("explicit");
        expect(explicitFirst.weightExplicit(0)).toBe(true);
        explicitFirst.pushEdge(0, 0, undefined);
        expect(explicitFirst.weightMode).toBe("mixed");
        expect(explicitFirst.weightExplicit(0)).toBe(true);
        expect(explicitFirst.weightExplicit(1)).toBe(false);
        const omittedFirst = new Staging(stagingOptions(true));
        omittedFirst.pushNode(null);
        omittedFirst.pushEdge(0, 0, undefined);
        omittedFirst.pushEdge(0, 0, undefined);
        expect(omittedFirst.weightMode).toBe("omitted");
        omittedFirst.trackWeight(0, true);
        expect(omittedFirst.weightMode).toBe("mixed");
        expect(omittedFirst.weightExplicit(0)).toBe(true);
        expect(omittedFirst.weightExplicit(1)).toBe(false);
        omittedFirst.pushEdge(0, 0, undefined);
        expect(omittedFirst.weightExplicit(2)).toBe(false);
        omittedFirst.trackWeight(2, true);
        expect(omittedFirst.weightExplicit(2)).toBe(true);
    });

    it("allocates f64 staging weights on demand and back-fills ones", () => {
        const staging = new Staging(stagingOptions(false, "f64"));
        staging.pushNode(null);
        staging.pushEdge(0, 0, undefined);
        expect(staging.weight).toBeNull();
        const weights = staging.ensureWeights();
        expect(weights).toBe(staging.ensureWeights());
        expect(weights.array).toBeInstanceOf(Float64Array);
        expect(weights.get(0)).toBe(1);
        weights.set(0, 0.1);
        expect(staging.weight?.get(0)).toBe(0.1);
    });
});

describe("remaps and compactStaging", () => {
    it("remapDropping assigns new indices in order and isIdentity detects identities", () => {
        const dropped = remapDropping(5, (i) => i % 2 === 0);
        expect(Array.from(dropped.remap)).toEqual([0, INVALID_INDEX, 1, INVALID_INDEX, 2]);
        expect(Array.from(dropped.origin)).toEqual([0, 2, 4]);
        expect(dropped.count).toBe(3);
        expect(isIdentity(dropped.remap)).toBe(false);
        const kept = remapDropping(3, () => true);
        expect(isIdentity(kept.remap)).toBe(true);
        expect(isIdentity(new Uint32Array(0))).toBe(true);
    });

    it("compacts nodes, edges, weights, columns and extensions into a fresh staging", () => {
        const source = new Staging(stagingOptions(false, "f64"));
        source.materialiseIds();
        source.pushNode("a");
        source.pushNode("b");
        source.pushNode("c");
        source.pushEdge(0, 1, 0.5);
        source.pushEdge(1, 2, undefined);
        source.pushEdge(2, 0, 3);
        source.pushEdge(2, 2, undefined);
        const label = createStagingColumn("label", "node", { dtype: "string" }, false);
        label.write(0, "A");
        label.write(2, "C");
        const pair = createStagingColumn("pair", "edge", { dtype: "u32", refersTo: "edge" }, false);
        pair.write(0, 2);
        pair.write(2, 3);
        pair.write(3, 0);
        source.nodeColumns.push(label);
        source.edgeColumns.push(pair);
        const element = createStagingColumn("element", "extension", { dtype: "u32", refersTo: "node" }, false);
        element.write(0, 1);
        element.write(1, 2);
        source.extensions.push({ name: "ext", columns: [element], rowCount: 2 });
        source.nodeAlive.set(1, false);
        source.liveNodeCount--;
        source.killEdge(0);
        source.killEdge(1);
        const nodes = remapDropping(3, (i) => source.nodeAlive.get(i));
        const edges = remapDropping(4, (e) => source.edgeAlive.get(e));
        const out = compactStaging(source, {
            nodes,
            edges,
            edgeValues: null,
            mergedWeights: null,
            mergedWeightStored: null,
            mergedWeightSet: null,
        });
        expect(out).not.toBe(source);
        expect(out.nodeBound).toBe(2);
        expect(out.edgeBound).toBe(2);
        expect(out.liveNodeCount).toBe(2);
        expect(out.liveEdgeCount).toBe(2);
        expect(out.selfLoopCount).toBe(1);
        expect(out.ids).toEqual(["a", "c"]);
        expect(out.ids).not.toBe(source.ids);
        expect(out.idToIndex).not.toBe(source.idToIndex);
        expect(out.idToIndex?.get("c")).toBe(1);
        expect(source.ids).toEqual(["a", "b", "c"]);
        expect([out.src.get(0), out.dst.get(0)]).toEqual([1, 0]);
        expect([out.src.get(1), out.dst.get(1)]).toEqual([1, 1]);
        expect(out.weight?.get(0)).toBe(3);
        expect(out.weightExplicit(0)).toBe(true);
        expect(out.weightExplicit(1)).toBe(false);
        expect(out.weightMode).toBe("mixed");
        expect(Array.from(out.outEdges(1))).toEqual([0, 1]);
        expect(Array.from(out.inEdges(0))).toEqual([0]);
        expect(out.nodeColumns[0].read(0)).toBe("A");
        expect(out.nodeColumns[0].read(1)).toBe("C");
        expect(out.edgeColumns[0].read(0)).toBe(1);
        expect(out.edgeColumns[0].isSet(1)).toBe(false);
        expect(out.extensions[0].rowCount).toBe(2);
        expect(out.extensions[0].columns[0].isSet(0)).toBe(false);
        expect(out.extensions[0].columns[0].read(1)).toBe(1);
        expect(source.liveEdgeCount).toBe(2);
        expect(source.edgeBound).toBe(4);
    });

    it("applies merged weights and explicit bits during compaction", () => {
        const source = new Staging(stagingOptions());
        source.pushNode(null);
        source.pushEdge(0, 0, undefined);
        source.pushEdge(0, 0, undefined);
        expect(source.weight).toBeNull();
        const nodes = remapDropping(1, () => true);
        const edges = remapDropping(2, (e) => e === 0);
        const out = compactStaging(source, {
            nodes,
            edges,
            edgeValues: null,
            mergedWeights: new Float64Array([2]),
            mergedWeightStored: new Uint32Array([1]),
            mergedWeightSet: new Uint32Array([1]),
        });
        expect(out.weight?.get(0)).toBe(2);
        expect(out.weightExplicit(0)).toBe(true);
        expect(out.ids).toBeNull();
        const keep = compactStaging(source, {
            nodes,
            edges,
            edgeValues: null,
            mergedWeights: new Float64Array([7]),
            mergedWeightStored: new Uint32Array([0]),
            mergedWeightSet: null,
        });
        expect(keep.weight?.get(0)).toBe(1);
        expect(keep.weightExplicit(0)).toBe(false);
    });
});

describe("StagingColumn: constructor guards", () => {
    it("wraps resolved metadata for every dtype", () => {
        for (const dtype of ["f32", "f64", "i32", "u32", "u8", "bool", "dict", "string", "json"] as const) {
            const column = new StagingColumn(resolveColumnMeta("c", "node", { dtype }), false);
            column.ensureLength(2);
            expect(column.length).toBe(2);
            expect(column.toColumn(2).dtype).toBe(dtype);
        }
        const list = new StagingColumn(resolveColumnMeta("l", "node", { dtype: "list", itemDtype: "f32" }), false);
        expect(list.toColumn(1).dtype).toBe("list");
    });
});
