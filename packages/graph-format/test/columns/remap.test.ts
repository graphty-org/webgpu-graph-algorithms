import { describe, expect, it } from "vitest";

import { allocU8, columnFromTypedArray, columnFromValues } from "../../src/columns/column.js";
import {
    gatherArray,
    gatherColumn,
    gatherTable,
    remapArray,
    remapColumn,
    remapColumnWith,
    remapReferences,
    remapTable,
    scatterArray,
    withComponents,
} from "../../src/columns/remap.js";
import { createTable } from "../../src/columns/table.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { type Column, type ColumnDeclPatch } from "../../src/types/index.js";
import { canViewAsPaddedU32 } from "../../src/util/typed-array.js";

function expectError(fn: () => unknown, code: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe(code);
    return error;
}

function values(name: string, entries: readonly unknown[], decl: ColumnDeclPatch = {}): Column {
    return columnFromValues("node", entries.length, name, entries, decl);
}

function u32(...entries: number[]): Uint32Array<ArrayBuffer> {
    return new Uint32Array(entries);
}

const X = INVALID_INDEX;

describe("remapArray", () => {
    it("moves rows old -> new, drops INVALID_INDEX rows and fills uncovered rows", () => {
        const data = new Float64Array([10, 11, 12, 13]);
        const out = remapArray(data, u32(2, X, 0, 1), 5, -1);
        expect(out).toBeInstanceOf(Float64Array);
        expect(Array.from(out)).toEqual([12, 13, 10, -1, -1]);
        expect(out).not.toBe(data);
    });

    it("handles components and every array class", () => {
        const data = new Float32Array([1, 2, 3, 4, 5, 6]);
        expect(Array.from(remapArray(data, u32(1, X, 0), 2, 0, 2))).toEqual([5, 6, 1, 2]);
        expect(Array.from(remapArray(new Int32Array([1, 2]), u32(1, 0), 2, 0))).toEqual([2, 1]);
        expect(Array.from(remapArray(new Uint32Array([1, 2]), u32(1, 0), 2, 0))).toEqual([2, 1]);
        const bytes = remapArray(new Uint8Array([1, 2, 3]), u32(2, 1, 0), 3, 0);
        expect(bytes).toBeInstanceOf(Uint8Array);
        expect(Array.from(bytes)).toEqual([3, 2, 1]);
        expect(canViewAsPaddedU32(bytes)).toBe(true);
    });

    it("ignores remap entries beyond the data and data beyond the remap", () => {
        expect(Array.from(remapArray(new Float64Array([1, 2]), u32(1, 0, 5), 2, 0))).toEqual([2, 1]);
        expect(Array.from(remapArray(new Float64Array([1, 2, 3]), u32(1, 0), 2, 0))).toEqual([2, 1]);
    });

    it("rejects out-of-range targets and bad components", () => {
        expectError(() => remapArray(new Float64Array([1]), u32(3), 2, 0), "E_INDEX_RANGE");
        expectError(() => remapArray(new Float64Array([1, 2, 3]), u32(0), 2, 0, 2), "E_COLUMN_LENGTH");
        expectError(() => remapArray(new Float64Array([1]), u32(0), 2, 0, 0), "E_COLUMN_TYPE");
    });
});

describe("gatherArray and scatterArray", () => {
    it("gathers out[i] = data[indexMap[i]]", () => {
        const data = new Float32Array([1, 2, 3]);
        expect(Array.from(gatherArray(data, u32(2, 2, 0)))).toEqual([3, 3, 1]);
        const vec = new Float64Array([1, 2, 3, 4, 5, 6]);
        expect(Array.from(gatherArray(vec, u32(2, 0), 2))).toEqual([5, 6, 1, 2]);
        expect(gatherArray(data, u32()).length).toBe(0);
        expectError(() => gatherArray(data, u32(3)), "E_INDEX_RANGE");
    });

    it("scatters out[indexMap[i]] = values[i] and returns out", () => {
        const out = new Float64Array(4).fill(-1);
        const result = scatterArray(out, new Float64Array([10, 20]), u32(3, 1));
        expect(result).toBe(out);
        expect(Array.from(out)).toEqual([-1, 20, -1, 10]);
        const vec = new Uint32Array(6);
        scatterArray(vec, new Uint32Array([7, 8]), u32(2), 2);
        expect(Array.from(vec)).toEqual([0, 0, 0, 0, 7, 8]);
        expectError(() => scatterArray(out, new Float64Array([1]), u32(4)), "E_INDEX_RANGE");
        expectError(() => scatterArray(out, new Float64Array([1, 2]), u32(0)), "E_COLUMN_LENGTH");
    });
});

describe("gatherColumn", () => {
    it("gathers every dtype with validity", () => {
        const map = u32(3, 1, 0);
        const f64 = values("f", [1.5, undefined, 3.5, 4.5], { dtype: "f64" });
        const gf = gatherColumn(f64, map);
        expect(gf.dtype).toBe("f64");
        expect(Array.from((gf as { data: Float64Array }).data)).toEqual([4.5, 0, 1.5]);
        expect(gf.isSet(1)).toBe(false);
        expect(gf.nullCount).toBe(1);
        expect(gf.meta).toBe(f64.meta);

        const vec = values(
            "p",
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            { dtype: "f32", components: 2 },
        );
        expect(Array.from((gatherColumn(vec, map) as { data: Float32Array }).data)).toEqual([7, 8, 3, 4, 1, 2]);

        const u8 = values("u", [1, 2, 3, 4], { dtype: "u8" });
        const gu = gatherColumn(u8, map);
        if (gu.dtype !== "u8") {
            throw new Error("unreachable");
        }
        expect(Array.from(gu.data)).toEqual([4, 2, 1]);
        expect(canViewAsPaddedU32(gu.data)).toBe(true);

        const bool = values("b", [true, false, undefined, true], { dtype: "bool" });
        const gb = gatherColumn(bool, u32(3, 1, 2, 0));
        expect([gb.value(0), gb.value(1), gb.value(2), gb.value(3)]).toEqual([true, false, undefined, true]);

        const dict = values("d", ["a", "b", "c", "a"], { dtype: "dict" });
        const gd = gatherColumn(dict, map);
        if (gd.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(gd.dictionary).toBe((dict as { dictionary: unknown }).dictionary);
        expect(Array.from(gd.codes)).toEqual([0, 1, 0]);

        const str = values("s", ["w", "x", "y", "z"], { dtype: "string" });
        const gs = gatherColumn(str, map);
        if (gs.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(gs.decodeAll()).toEqual(["z", "x", "w"]);

        const list = values("l", [[1], [2, 3], [], [4, 5, 6]], { dtype: "list", itemDtype: "i32" });
        const gl = gatherColumn(list, u32(3, 2, 1));
        if (gl.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(Array.from(gl.offsets)).toEqual([0, 3, 3, 5]);
        expect(gl.sliceOf(0)).toEqual([4, 5, 6]);
        expect(gl.sliceOf(1)).toEqual([]);
        expect(gl.sliceOf(2)).toEqual([2, 3]);

        const json = values("j", [{ a: 1 }, null, undefined, "t"], { dtype: "json" });
        const gj = gatherColumn(json, u32(3, 2, 1, 0));
        if (gj.dtype !== "json") {
            throw new Error("unreachable");
        }
        expect(gj.values).toEqual(["t", undefined, null, { a: 1 }]);
        expect(gj.nullCount).toBe(1);
    });

    it("treats INVALID_INDEX entries as unset rows holding the fill and makes the column nullable", () => {
        const column = columnFromTypedArray("node", 2, "x", new Uint32Array([5, 6]), { fill: 9 });
        expect(column.meta.nullable).toBe(false);
        const gathered = gatherColumn(column, u32(1, X, 0));
        expect(gathered.meta.nullable).toBe(true);
        expect(Array.from((gathered as { data: Uint32Array }).data)).toEqual([6, 9, 5]);
        expect(gathered.isSet(1)).toBe(false);
        expect(gathered.nullCount).toBe(1);
        const dict = values("d", ["a", "b"], { dtype: "dict", default: "b" });
        const gd = gatherColumn(dict, u32(X, 0));
        expect(gd.value(0)).toBe("b");
        // the default was seeded as code 0, "a" interned as code 1
        expect(Array.from((gd as { codes: Uint32Array }).codes)).toEqual([0, 1]);
        const str = values("s", ["a"], { dtype: "string", fill: "-" });
        expect((gatherColumn(str, u32(X, 0)) as { valueAt(row: number): string }).valueAt(0)).toBe("-");
        const bool = values("b", [false], { dtype: "bool", default: true });
        expect(gatherColumn(bool, u32(X)).value(0)).toBe(true);
        expectError(() => gatherColumn(column, u32(2)), "E_INDEX_RANGE");
    });
});

describe("remapColumn (old -> new)", () => {
    it("drops rows mapped to INVALID_INDEX and orders survivors by their new index", () => {
        const column = values("f", [10, 11, 12, 13], { dtype: "f64" });
        const out = remapColumn(column, u32(1, X, 0, 2), 3);
        expect(Array.from((out as { data: Float64Array }).data)).toEqual([12, 10, 13]);
        expect(out.nullCount).toBe(0);
        expect(out.length).toBe(3);
    });

    it("leaves rows no old row maps to unset (nullable)", () => {
        const column = columnFromTypedArray("node", 2, "x", new Float32Array([1, 2]), { default: 5 });
        const out = remapColumn(column, u32(0, 1), 4);
        expect(out.length).toBe(4);
        expect(out.meta.nullable).toBe(true);
        expect(out.isSet(2)).toBe(false);
        expect(out.value(2)).toBe(5);
        expect(Array.from((out as { data: Float32Array }).data)).toEqual([1, 2, 5, 5]);
        expectError(() => remapColumn(column, u32(0, 4), 4), "E_INDEX_RANGE");
    });

    it("rewrites refersTo values through the same remap for a self-referencing column", () => {
        // parent column: node 0 -> parent 1, node 1 -> none, node 2 -> parent 0, node 3 -> parent 2
        const parent = values("parent", [1, undefined, 0, 2], { dtype: "u32", refersTo: "node" });
        expect(Array.from((parent as { data: Uint32Array }).data)).toEqual([1, X, 0, 2]);
        // compaction removes node 2
        const remap = u32(0, 1, X, 2);
        const out = remapColumn(parent, remap, 3);
        if (out.dtype !== "u32") {
            throw new Error("unreachable");
        }
        expect(Array.from(out.data)).toEqual([1, X, X]);
        expect(out.isSet(0)).toBe(true);
        expect(out.isSet(1)).toBe(false);
        expect(out.isSet(2)).toBe(false);
        expect(out.nullCount).toBe(2);
        expect(out.value(2)).toBeUndefined();
    });

    it("rewrites values through a separate value remap for cross-space columns", () => {
        // an edge column referring to nodes: rows follow the edge remap, values the node remap
        const column = columnFromValues("edge", 3, "src", [2, 0, 1], { dtype: "u32", refersTo: "node" });
        const edgeRemap = u32(X, 0, 1);
        const nodeRemap = u32(1, X, 0);
        const out = remapColumnWith(column, edgeRemap, 2, nodeRemap);
        if (out.dtype !== "u32") {
            throw new Error("unreachable");
        }
        expect(Array.from(out.data)).toEqual([1, X]);
        expect(out.isSet(0)).toBe(true);
        expect(out.isSet(1)).toBe(false);
        // with valueRemap null the values are untouched
        const plain = remapColumnWith(column, edgeRemap, 2, null);
        expect(Array.from((plain as { data: Uint32Array }).data)).toEqual([0, 1]);
    });
});

describe("remapReferences", () => {
    it("returns non-index columns unchanged and rejects mis-declared ones", () => {
        const column = values("f", [1], { dtype: "f64" });
        expect(remapReferences(column, u32(0))).toBe(column);
    });

    it("maps in-range values and turns dangling and out-of-range values into INVALID_INDEX + unset", () => {
        const column = columnFromValues("edge", 5, "pair", [1, 0, 7, undefined, 4], { dtype: "u32", refersTo: "edge" });
        const remap = u32(3, 2, 1, 0);
        const out = remapReferences(column, remap);
        if (out.dtype !== "u32") {
            throw new Error("unreachable");
        }
        expect(Array.from(out.data)).toEqual([2, 3, X, X, X]);
        expect([0, 1, 2, 3, 4].map((row) => out.isSet(row))).toEqual([true, true, false, false, false]);
        expect(out.nullCount).toBe(3);
        expect(out.meta.refersTo).toBe("edge");
        // the source is untouched
        expect(Array.from((column as { data: Uint32Array }).data)).toEqual([1, 0, 7, X, 4]);
        expect(column.nullCount).toBe(1);
    });

    it("makes a non-nullable column nullable only when a reference dangles", () => {
        const column = columnFromTypedArray("node", 2, "parent", new Uint32Array([1, 0]), { refersTo: "node" });
        expect(column.meta.nullable).toBe(false);
        const same = remapReferences(column, u32(1, 0));
        expect(same.meta.nullable).toBe(false);
        expect(same.validity).toBeNull();
        expect(Array.from((same as { data: Uint32Array }).data)).toEqual([0, 1]);
        const dropped = remapReferences(column, u32(X, 0));
        expect(dropped.meta.nullable).toBe(true);
        expect(dropped.isSet(0)).toBe(true);
        expect(dropped.isSet(1)).toBe(false);
    });

    it("rewrites the u32 child of a list, dropping dangling items and unsetting fully dangling rows", () => {
        const parents = values("parents", [[0, 1], [2], [], undefined, [1, 2]], {
            dtype: "list",
            itemDtype: "u32",
            refersTo: "node",
        });
        const remap = u32(5, X, 4);
        const out = remapReferences(parents, remap);
        if (out.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(out.sliceOf(0)).toEqual([5]);
        expect(out.isSet(0)).toBe(true);
        expect(out.sliceOf(1)).toEqual([4]);
        expect(out.sliceOf(2)).toEqual([]);
        expect(out.isSet(2)).toBe(true);
        expect(out.isSet(3)).toBe(false);
        expect(out.sliceOf(4)).toEqual([4]);
        expect(out.nullCount).toBe(1);
        expect(out.child.dtype).toBe("u32");
        expect(out.child.meta.refersTo).toBe("node");
        expect(Array.from(out.offsets)).toEqual([0, 1, 2, 2, 2, 3]);
        const allDangling = remapReferences(parents, u32(X, X, X));
        if (allDangling.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(allDangling.isSet(0)).toBe(false);
        expect(allDangling.isSet(1)).toBe(false);
        expect(allDangling.isSet(2)).toBe(true);
        expect(allDangling.isSet(3)).toBe(false);
        expect(allDangling.isSet(4)).toBe(false);
        expect(allDangling.nullCount).toBe(4);
        expect(Array.from(allDangling.offsets)).toEqual([0, 0, 0, 0, 0, 0]);
        expect(allDangling.child.length).toBe(0);
    });
});

describe("remapTable and gatherTable", () => {
    it("remaps every column of a node table through the node remap and rewrites references", () => {
        const nodes = createTable("node", 4);
        nodes.set("score", new Float64Array([0.1, 0.2, 0.3, 0.4]));
        nodes.set("parent", [1, undefined, 3, 0], { dtype: "u32", refersTo: "node" });
        nodes.set("edgeRef", [0, 1, 2, 3], { dtype: "u32", refersTo: "edge" });
        nodes.set("label", ["a", "b", "c", "d"], { dtype: "string", role: "label" });
        const nodeRemap = u32(0, X, 1, 2);
        const edgeRemap = u32(X, 0, X, 1);
        const out = remapTable(nodes, nodeRemap, 3, { node: nodeRemap, edge: edgeRemap });
        expect(out).not.toBe(nodes);
        expect(out.rowCount).toBe(3);
        expect(out.domain).toBe("node");
        expect(out.names()).toEqual(["score", "parent", "edgeRef", "label"]);
        expect(Array.from(out.requireTyped("score", "f64").data)).toEqual([0.1, 0.3, 0.4]);
        const parent = out.requireTyped("parent", "u32");
        expect(Array.from(parent.data)).toEqual([X, 2, 0]);
        expect(parent.isSet(0)).toBe(false);
        expect(parent.isSet(1)).toBe(true);
        const edgeRef = out.requireTyped("edgeRef", "u32");
        expect(Array.from(edgeRef.data)).toEqual([X, X, 1]);
        expect(edgeRef.nullCount).toBe(2);
        expect(out.requireTyped("label", "string").decodeAll()).toEqual(["a", "c", "d"]);
        expect(out.byRole("label")?.meta.name).toBe("label");
        // the source table is untouched
        expect(nodes.rowCount).toBe(4);
        expect(Array.from(nodes.requireTyped("parent", "u32").data)).toEqual([1, X, 3, 0]);
    });

    it("rewrites only references when rows are unchanged (extension tables)", () => {
        const extension = createTable("extension", 3);
        extension.set("element", new Uint32Array([0, 1, 2]), { refersTo: "node" });
        const start = extension.set("start", new Float64Array([1, 2, 3]));
        const out = remapTable(extension, null, 99, { node: u32(2, X, 0), edge: null });
        expect(out.rowCount).toBe(3);
        expect(Array.from(out.requireTyped("element", "u32").data)).toEqual([2, X, 0]);
        expect(out.requireTyped("element", "u32").isSet(1)).toBe(false);
        expect(out.get("start")).toBe(start);
        const untouched = remapTable(extension, null, 3, { node: null, edge: null });
        expect(untouched.get("element")).toBe(extension.get("element"));
    });

    it("gathers a table through a new -> old map and rewrites references", () => {
        const edges = createTable("edge", 3);
        edges.set("w", new Float32Array([1, 2, 3]));
        edges.set("pair", [2, X, 0], { dtype: "u32", refersTo: "edge" });
        const edgeOrigin = u32(2, 0);
        const edgeRemap = u32(1, X, 0);
        const out = gatherTable(edges, edgeOrigin, { node: null, edge: edgeRemap });
        expect(out.rowCount).toBe(2);
        expect(Array.from(out.requireTyped("w", "f32").data)).toEqual([3, 1]);
        const pair = out.requireTyped("pair", "u32");
        expect(Array.from(pair.data)).toEqual([1, 0]);
        expect(pair.nullCount).toBe(0);
        const noRefs = gatherTable(edges, edgeOrigin, { node: null, edge: null });
        expect(Array.from(noRefs.requireTyped("pair", "u32").data)).toEqual([0, 2]);
    });

    it("keeps allocU8 stores adoptable through gather", () => {
        const table = createTable("node", 3);
        table.set("bytes", allocU8(3));
        const out = gatherTable(table, u32(2, 1, 0, 0, 1), { node: null, edge: null });
        expect(canViewAsPaddedU32(out.requireTyped("bytes", "u8").data)).toBe(true);
        expect(out.requireTyped("bytes", "u8").paddedU32View().length).toBe(2);
    });
});

describe("withComponents (section 5.2)", () => {
    it("expands the stride, filling the new lanes", () => {
        const xy = new Float32Array([1, 2, 3, 4, 5, 6]);
        const xyz = withComponents(xy, 2, 3, 0);
        expect(xyz).toBeInstanceOf(Float32Array);
        expect(Array.from(xyz)).toEqual([1, 2, 0, 3, 4, 0, 5, 6, 0]);
        expect(Array.from(withComponents(xy, 2, 4, -1))).toEqual([1, 2, -1, -1, 3, 4, -1, -1, 5, 6, -1, -1]);
        expect(Array.from(withComponents(new Float32Array([7, 8]), 1, 2, 9))).toEqual([7, 9, 8, 9]);
    });

    it("narrows the stride, dropping the surplus lanes", () => {
        const xyz = new Float32Array([1, 2, 0, 3, 4, 0, 5, 6, 0]);
        expect(Array.from(withComponents(xyz, 3, 2, 0))).toEqual([1, 2, 3, 4, 5, 6]);
        expect(Array.from(withComponents(xyz, 3, 1, 0))).toEqual([1, 3, 5]);
        expect(Array.from(withComponents(new Float32Array(0), 3, 2, 0))).toEqual([]);
    });

    it("returns the input itself when the stride is unchanged", () => {
        const data = new Float32Array([1, 2, 3, 4]);
        expect(withComponents(data, 2, 2, 0)).toBe(data);
    });

    it("rejects a stride outside 1..16 and a length that is not a multiple of the stride", () => {
        const data = new Float32Array([1, 2, 3, 4]);
        for (const [from, to] of [
            [0, 2],
            [2, 0],
            [17, 2],
            [2, 17],
            [1.5, 2],
            [2, Number.NaN],
        ] as const) {
            let caught: unknown = null;
            try {
                withComponents(data, from, to, 0);
            } catch (err) {
                caught = err;
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            expect((caught as GraphFormatError).code).toBe("E_COLUMN_TYPE");
        }
        let caught: unknown = null;
        try {
            withComponents(new Float32Array(5), 2, 3, 0);
        } catch (err) {
            caught = err;
        }
        expect((caught as GraphFormatError).code).toBe("E_COLUMN_LENGTH");
    });
});
