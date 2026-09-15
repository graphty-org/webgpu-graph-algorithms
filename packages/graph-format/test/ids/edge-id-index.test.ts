import { describe, expect, it } from "vitest";

import { INVALID_INDEX } from "../../src/constants.js";
import { buildEdgeIdMap, canHoldEdgeIds, EdgeIdIndex, edgeIdOfRow } from "../../src/ids/edge-id-index.js";
import { type Column, type ColumnMeta, type Dtype } from "../../src/types/index.js";

/** A mutable stand-in for the column implementation (the columns module is built separately). */
interface FakeColumn {
    dtype: Dtype;
    meta: ColumnMeta;
    length: number;
    version: number;
    unset: Set<number>;
    isSet(row: number): boolean;
    valueAt?(row: number): string;
    data?: ArrayLike<number>;
    codes?: Uint32Array;
    dictionary?: string[];
}

function meta(name: string, dtype: Dtype, components = 1): ColumnMeta {
    return {
        name,
        domain: "edge",
        dtype,
        components,
        itemDtype: null,
        itemComponents: null,
        nullable: true,
        mutable: false,
        role: "id",
        refersTo: null,
        unique: true,
        default: undefined,
        fill: 0,
        options: null,
        origin: null,
        dynamic: false,
        extra: {},
    };
}

function base(name: string, dtype: Dtype, length: number, unset: number[] = [], components = 1): FakeColumn {
    const unsetRows = new Set(unset);
    return {
        dtype,
        meta: meta(name, dtype, components),
        length,
        version: 0,
        unset: unsetRows,
        isSet: (row) => !unsetRows.has(row),
    };
}

function stringColumn(values: string[], unset: number[] = []): Column {
    const column = base("id", "string", values.length, unset);
    column.valueAt = (row) => values[row];
    return column as unknown as Column;
}

function dictColumn(dictionary: string[], codes: number[], unset: number[] = []): Column {
    const column = base("kind", "dict", codes.length, unset);
    column.codes = new Uint32Array(codes);
    column.dictionary = dictionary;
    return column as unknown as Column;
}

function numericColumn(dtype: Dtype, data: ArrayLike<number>, unset: number[] = [], components = 1): Column {
    const column = base("num", dtype, data.length / components, unset, components);
    column.data = data;
    return column as unknown as Column;
}

function otherColumn(dtype: Dtype): Column {
    return base("other", dtype, 2) as unknown as Column;
}

describe("edgeIdOfRow / canHoldEdgeIds", () => {
    it("reads string, dict and every scalar numeric dtype", () => {
        expect(edgeIdOfRow(stringColumn(["e0", "e1"]), 1)).toBe("e1");
        expect(edgeIdOfRow(dictColumn(["x", "y"], [1, 0]), 0)).toBe("y");
        expect(edgeIdOfRow(numericColumn("u32", new Uint32Array([7, 8])), 1)).toBe(8);
        expect(edgeIdOfRow(numericColumn("f64", new Float64Array([0.5, 1.5])), 0)).toBe(0.5);
        expect(edgeIdOfRow(numericColumn("f32", new Float32Array([2, 3])), 1)).toBe(3);
        expect(edgeIdOfRow(numericColumn("i32", new Int32Array([-4, 5])), 0)).toBe(-4);
        expect(edgeIdOfRow(numericColumn("u8", new Uint8Array([9, 10])), 1)).toBe(10);
        for (const dtype of ["string", "dict", "u32", "f64", "f32", "i32", "u8"] as Dtype[]) {
            expect(canHoldEdgeIds(otherColumn(dtype))).toBe(true);
        }
    });

    it("reads the first component of a multi-component numeric column", () => {
        const column = numericColumn("u32", new Uint32Array([10, 11, 20, 21, 30, 31]), [], 2);
        expect(column.length).toBe(3);
        expect(edgeIdOfRow(column, 2)).toBe(30);
    });
});

describe("buildEdgeIdMap", () => {
    it("indexes set rows only, lowest edge index first", () => {
        const map = buildEdgeIdMap(stringColumn(["a", "b", "a", "c"], [3]));
        expect([...map.entries()]).toEqual([
            ["a", 0],
            ["b", 1],
        ]);
    });

    it("uses SameValueZero so -0 and 0 are one id and numbers never equal strings", () => {
        const map = buildEdgeIdMap(numericColumn("f64", new Float64Array([-0, 2])));
        expect(map.get(0)).toBe(0);
        expect(map.get(-0)).toBe(0);
        expect(map.get("0")).toBeUndefined();
    });
});

describe("EdgeIdIndex", () => {
    it("builds lazily on the first lookup and answers with INVALID_INDEX on a miss", () => {
        const index = new EdgeIdIndex(stringColumn(["e0", "e1", "e2"], [1]));
        expect(index.built).toBe(false);
        expect(index.indexOf("e2")).toBe(2);
        expect(index.built).toBe(true);
        expect(index.indexOf("e0")).toBe(0);
        expect(index.indexOf("e1")).toBe(INVALID_INDEX);
        expect(index.indexOf("nope")).toBe(INVALID_INDEX);
        expect(index.indexOf(0)).toBe(INVALID_INDEX);
        expect(index.has("e0")).toBe(true);
        expect(index.has("e1")).toBe(false);
        expect(index.size).toBe(2);
    });

    it("resolves numeric ids through numeric and dict columns", () => {
        const numeric = new EdgeIdIndex(numericColumn("f64", new Float64Array([10, 2.5, 3])));
        expect(numeric.indexOf(2.5)).toBe(1);
        expect(numeric.indexOf("2.5")).toBe(INVALID_INDEX);
        expect(numeric.indexOf(3)).toBe(2);
        const dict = new EdgeIdIndex(dictColumn(["cat", "dog"], [1, 0, 1], [2]));
        expect(dict.indexOf("dog")).toBe(0);
        expect(dict.indexOf("cat")).toBe(1);
        expect(dict.size).toBe(2);
    });

    it("rebuilds after the column version changes", () => {
        const values = ["a", "b"];
        const column = stringColumn(values) as unknown as FakeColumn;
        const index = new EdgeIdIndex(column as unknown as Column);
        expect(index.indexOf("b")).toBe(1);
        values[1] = "c";
        expect(index.indexOf("c")).toBe(INVALID_INDEX);
        expect(index.built).toBe(true);
        column.version = 1;
        expect(index.built).toBe(false);
        expect(index.indexOf("c")).toBe(1);
        expect(index.indexOf("b")).toBe(INVALID_INDEX);
        expect(index.built).toBe(true);
    });

    it("keeps the lowest edge index when the column is not unique", () => {
        const index = new EdgeIdIndex(stringColumn(["x", "x"]));
        expect(index.indexOf("x")).toBe(0);
        expect(index.size).toBe(1);
    });
});
