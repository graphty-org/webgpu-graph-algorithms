import { describe, expect, it } from "vitest";

import { allocU8, columnFromValues } from "../../src/columns/column.js";
import {
    AttributeTable,
    createTable,
    tableWithColumns,
    verifyUniqueColumn,
    verifyUniqueColumns,
} from "../../src/columns/table.js";
import { GraphFormatError } from "../../src/errors.js";
import { type Column, type F32 } from "../../src/types/index.js";

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

describe("AttributeTable construction", () => {
    it("wraps columns in declaration order and validates them", () => {
        const a = columnFromValues("node", 3, "a", [1, 2, 3], { dtype: "i32" });
        const b = columnFromValues("node", 3, "b", ["x", "y", "z"], { dtype: "string", role: "label" });
        const table = new AttributeTable({ domain: "node", rowCount: 3, columns: [a, b] });
        expect(table.domain).toBe("node");
        expect(table.rowCount).toBe(3);
        expect(table.names()).toEqual(["a", "b"]);
        expect([...table]).toEqual([a, b]);
        expect(table.get("a")).toBe(a);
        expect(table.byRole("label")).toBe(b);
        expectError(() => new AttributeTable({ domain: "node", rowCount: 3, columns: [a, a] }), "E_COLUMN_EXISTS");
        expectError(() => new AttributeTable({ domain: "node", rowCount: 2, columns: [a] }), "E_COLUMN_LENGTH");
        expectError(() => new AttributeTable({ domain: "node", rowCount: -1, columns: [] }), "E_COLUMN_LENGTH");
        const c = columnFromValues("node", 3, "c", ["p", "q", "r"], { dtype: "string", role: "label" });
        expectError(() => new AttributeTable({ domain: "node", rowCount: 3, columns: [b, c] }), "E_DUPLICATE_ROLE");
        expect(createTable("graph", 1).rowCount).toBe(1);
        expect(createTable("graph", 1).names()).toEqual([]);
    });
});

describe("AttributeTable lookups", () => {
    const table = createTable("edge", 4);
    table.set("w", new Float64Array([1, 2, 3, 4]), { role: "weight" });
    table.set("kind", ["a", "b", undefined, "a"], { dtype: "dict", role: "kind", default: "z" });

    it("returns null from the total lookups and throws from the checked ones", () => {
        expect(table.has("w")).toBe(true);
        expect(table.has("nope")).toBe(false);
        expect(table.get("nope")).toBeNull();
        expect(table.typed("nope", "f64")).toBeNull();
        expect(table.typed("w", "f32")).toBeNull();
        expect(table.typed("w", "f64")?.data[1]).toBe(2);
        expect(table.byRole("position")).toBeNull();
        expect(table.byRole("weight")?.meta.name).toBe("w");
        const missing = expectError(() => table.require("nope"), "E_UNKNOWN_COLUMN");
        expect(missing.details).toMatchObject({ column: "nope", domain: "edge" });
        expectError(() => table.requireTyped("nope", "f64"), "E_UNKNOWN_COLUMN");
        const wrong = expectError(() => table.requireTyped("w", "u32"), "E_COLUMN_TYPE");
        expect(wrong.details).toMatchObject({ expected: "u32", found: "f64" });
        expect(table.requireTyped("kind", "dict").dictionary).toEqual(["z", "a", "b"]);
    });

    it("reads cells with default and validity semantics", () => {
        expect(table.value("w", 2)).toBe(3);
        expect(table.value("kind", 0)).toBe("a");
        expect(table.value("kind", 2)).toBe("z");
        expect(table.isSet("kind", 2)).toBe(false);
        expect(table.isSet("kind", 3)).toBe(true);
        expect(table.isSet("kind", 4)).toBe(false);
        expectError(() => table.value("w", 4), "E_INDEX_RANGE");
        expectError(() => table.value("nope", 0), "E_UNKNOWN_COLUMN");
        expectError(() => table.isSet("nope", 0), "E_UNKNOWN_COLUMN");
    });
});

describe("AttributeTable.set", () => {
    it("adopts 4-byte arrays and f64 by reference and checks lengths", () => {
        const table = createTable("node", 3);
        const data = new Uint32Array([5, 6, 7]);
        const column = table.set("labels", data);
        expect(column.dtype).toBe("u32");
        expect((column as { data: unknown }).data).toBe(data);
        expect(column.meta.name).toBe("labels");
        expect(column.meta.domain).toBe("node");
        expect(column.meta.nullable).toBe(false);
        const window = new Float32Array(new ArrayBuffer(64), 16, 9);
        expect(
            (table.set("pos", window, { components: 3, role: "position", mutable: true }) as { data: F32 }).data,
        ).toBe(window);
        const error = expectError(() => table.set("bad", new Float32Array(2)), "E_COLUMN_LENGTH");
        expect(error.details).toMatchObject({ expected: 3, found: 2 });
        expectError(() => table.set("bad", new Float32Array(4), { components: 2 }), "E_COLUMN_LENGTH");
        expectError(() => table.set("bad", new Float32Array(3), { dtype: "i32" }), "E_COLUMN_TYPE");
    });

    it("copies or refuses unadoptable u8 arrays per opts.adopt", () => {
        const table = createTable("node", 5);
        const unaligned = new Uint8Array([1, 2, 3, 4, 5]);
        const copied = table.set("flags", unaligned);
        expect((copied as { data: Uint8Array }).data).not.toBe(unaligned);
        expect(Array.from((copied as { data: Uint8Array }).data)).toEqual([1, 2, 3, 4, 5]);
        expectError(() => table.set("flags", unaligned, {}, { adopt: "strict" }), "E_COLUMN_ALIGNMENT");
        const aligned = allocU8(5);
        expect((table.set("flags", aligned, {}, { adopt: "strict" }) as { data: Uint8Array }).data).toBe(aligned);
        expect(table.gpuView("flags").length).toBe(2);
    });

    it("builds string / list / json / inferred columns from JS arrays", () => {
        const table = createTable("node", 2);
        expect(table.set("s", ["a", "b"], { dtype: "string" }).dtype).toBe("string");
        expect(table.set("l", [[1], [2, 3]], { dtype: "list", itemDtype: "i32" }).dtype).toBe("list");
        expect(table.set("j", [{ a: 1 }, null], { dtype: "json" }).dtype).toBe("json");
        expect(table.set("i", [1, undefined]).dtype).toBe("i32");
        expect(table.get("i")?.meta.nullable).toBe(true);
        expect(table.get("i")?.nullCount).toBe(1);
        expectError(() => table.set("short", ["a"], { dtype: "string" }), "E_COLUMN_LENGTH");
        expectError(() => table.set("x", "nope" as unknown as Column), "E_COLUMN_TYPE");
        expectError(() => table.set("x", {} as unknown as Column), "E_COLUMN_TYPE");
    });

    it("replaces a column of the same name in place", () => {
        const table = createTable("node", 2);
        table.set("a", new Uint32Array(2));
        table.set("b", new Uint32Array(2));
        const replacement = table.set("a", new Float32Array(2));
        expect(table.names()).toEqual(["a", "b"]);
        expect(table.get("a")).toBe(replacement);
        expect(table.get("a")?.dtype).toBe("f32");
    });

    it("enforces one column per role with replaceRole eviction", () => {
        const table = createTable("node", 2);
        const seed = table.set("seed", new Float32Array(6), { components: 3, role: "position" });
        const error = expectError(
            () => table.set("pos", new Float32Array(6), { components: 3, role: "position" }),
            "E_DUPLICATE_ROLE",
        );
        expect(error.details).toMatchObject({ role: "position", column: "pos", holder: "seed" });
        expect(table.names()).toEqual(["seed"]);
        // a column may be replaced under its own role
        table.set("seed", new Float32Array(6), { components: 3, role: "position" });
        expect(table.names()).toEqual(["seed"]);
        const pos = table.set("pos", new Float32Array(6), { components: 3, role: "position" }, { replaceRole: true });
        expect(table.names()).toEqual(["pos"]);
        expect(table.byRole("position")).toBe(pos);
        // the evicted column's data is untouched and still usable by its owner
        expect(seed.length).toBe(2);
    });

    it("moves a Column object in, re-wrapping only when the slot differs", () => {
        const source = createTable("node", 2);
        const column = source.set("a", new Uint32Array([1, 2]));
        const target = createTable("node", 2);
        expect(target.set("a", column)).toBe(column);
        const renamed = target.set("b", column);
        expect(renamed).not.toBe(column);
        expect(renamed.meta.name).toBe("b");
        expect((renamed as { data: unknown }).data).toBe((column as { data: unknown }).data);
        const other = createTable("edge", 2);
        const moved = other.set("a", column);
        expect(moved.meta.domain).toBe("edge");
        const patched = other.set("c", column, { role: "community", mutable: true });
        expect(patched.meta.role).toBe("community");
        expect(patched.meta.mutable).toBe(true);
        expect(patched.meta.fill).toBe(column.meta.fill);
        expectError(() => other.set("d", column, { fill: 9 }), "E_COLUMN_TYPE");
        expectError(() => other.set("d", column, { dtype: "i32" }), "E_COLUMN_TYPE");
        const short = createTable("node", 3);
        expectError(() => short.set("a", column), "E_COLUMN_LENGTH");
    });

    it("checks roles for moved columns too", () => {
        const source = createTable("node", 2);
        const ranked = source.set("a", new Uint32Array([1, 2]), { role: "rank" });
        const target = createTable("node", 2);
        expect(target.set("a", ranked)).toBe(ranked);
        expectError(() => target.set("b", ranked), "E_DUPLICATE_ROLE");
        const evicting = target.set("b", ranked, undefined, { replaceRole: true });
        expect(target.names()).toEqual(["b"]);
        expect(target.byRole("rank")).toBe(evicting);
    });
});

describe("AttributeTable mutation helpers", () => {
    it("removes and renames keeping declaration order", () => {
        const table = createTable("node", 2);
        table.set("a", new Uint32Array(2));
        const b = table.set("b", new Uint32Array(2), { role: "rank" });
        table.set("c", new Uint32Array(2));
        expect(table.remove("nope")).toBe(false);
        table.rename("b", "bb");
        expect(table.names()).toEqual(["a", "bb", "c"]);
        const renamed = table.get("bb");
        expect(renamed?.meta.name).toBe("bb");
        expect(renamed?.meta.role).toBe("rank");
        expect((renamed as { data: unknown }).data).toBe((b as { data: unknown }).data);
        expect(table.byRole("rank")).toBe(renamed);
        table.rename("bb", "bb");
        expect(table.names()).toEqual(["a", "bb", "c"]);
        expectError(() => table.rename("nope", "x"), "E_UNKNOWN_COLUMN");
        expectError(() => table.rename("a", "c"), "E_COLUMN_EXISTS");
        expect(table.remove("bb")).toBe(true);
        expect(table.names()).toEqual(["a", "c"]);
        expect(table.byRole("rank")).toBeNull();
    });

    it("clones the set but shares the Column objects", () => {
        const table = createTable("node", 2);
        const a = table.set("a", new Uint32Array(2));
        const copy = table.clone();
        expect(copy).not.toBe(table);
        expect(copy.get("a")).toBe(a);
        copy.set("b", new Uint32Array(2));
        expect(table.has("b")).toBe(false);
        table.remove("a");
        expect(copy.has("a")).toBe(true);
    });

    it("returns GPU views per dtype", () => {
        const table = createTable("node", 2);
        const u32 = new Uint32Array(2);
        table.set("u32", u32);
        table.set("f64", new Float64Array([1.5, 2]));
        table.set("s", ["a", "b"], { dtype: "string" });
        expect(table.gpuView("u32")).toBe(u32);
        const converted = table.gpuView("f64");
        expect(converted).toBeInstanceOf(Float32Array);
        expect(Array.from(converted)).toEqual([1.5, 2]);
        expect(table.gpuView("f64")).toBe(converted);
        expectError(() => table.gpuView("s"), "E_GPU_INELIGIBLE");
        expectError(() => table.gpuView("nope"), "E_UNKNOWN_COLUMN");
    });
});

describe("declareColumn and tableWithColumns", () => {
    it("clones and attaches typed arrays and ColumnInputs", () => {
        const table = createTable("node", 2);
        const a = table.set("a", new Uint32Array(2));
        const rank = new Float64Array([0.5, 0.25]);
        const derived = tableWithColumns(table, {
            "graphty.pagerank.rank": rank,
            label: { data: ["x", "y"], decl: { dtype: "string", role: "label" } },
        });
        expect(derived).not.toBe(table);
        expect(derived.get("a")).toBe(a);
        expect(table.names()).toEqual(["a"]);
        expect(derived.names()).toEqual(["a", "graphty.pagerank.rank", "label"]);
        expect((derived.get("graphty.pagerank.rank") as { data: unknown }).data).toBe(rank);
        expect(derived.byRole("label")?.dtype).toBe("string");
    });
});

describe("unique enforcement", () => {
    it("is a no-op for non-unique columns and passes distinct set rows", () => {
        const table = createTable("edge", 3);
        table.set("id", ["e1", "e1", "e2"], { dtype: "string" });
        table.set("uid", ["e1", undefined, "e2"], { dtype: "string", unique: true, role: "id" });
        expect(() => verifyUniqueColumns(table)).not.toThrow();
    });

    it("throws E_DUPLICATE_EDGE_ID for edge columns and E_DUPLICATE_ID otherwise", () => {
        const edges = createTable("edge", 3);
        edges.set("id", ["e1", "e2", "e1"], { dtype: "string", unique: true, role: "id" });
        const edgeError = expectError(() => verifyUniqueColumns(edges), "E_DUPLICATE_EDGE_ID");
        expect(edgeError.details).toMatchObject({ column: "id", rows: [0, 2], value: "e1" });
        const nodes = createTable("node", 3);
        nodes.set("key", new Uint32Array([7, 8, 7]), { unique: true });
        expect(expectError(() => verifyUniqueColumns(nodes), "E_DUPLICATE_ID").details.rows).toEqual([0, 2]);
        const dict = columnFromValues("node", 3, "k", ["a", "b", "a"], { dtype: "dict", unique: true });
        expectError(() => verifyUniqueColumn(dict), "E_DUPLICATE_ID");
        const bool = columnFromValues("node", 2, "b", [true, true], { dtype: "bool", unique: true });
        expectError(() => verifyUniqueColumn(bool), "E_DUPLICATE_ID");
        const json = columnFromValues("node", 2, "j", [{ a: 1 }, { a: 1 }], { dtype: "json", unique: true });
        expectError(() => verifyUniqueColumn(json), "E_DUPLICATE_ID");
        const list = columnFromValues("node", 2, "l", [[1], [1]], { dtype: "list", itemDtype: "i32", unique: true });
        expectError(() => verifyUniqueColumn(list), "E_DUPLICATE_ID");
        const vec = columnFromValues(
            "node",
            3,
            "v",
            [
                [1, 2],
                [2, 1],
                [1, 2],
            ],
            { dtype: "f32", components: 2, unique: true },
        );
        expect(expectError(() => verifyUniqueColumn(vec), "E_DUPLICATE_ID").details.rows).toEqual([0, 2]);
        const distinctVec = columnFromValues(
            "node",
            2,
            "v",
            [
                [1, 2],
                [2, 1],
            ],
            { dtype: "f32", components: 2, unique: true },
        );
        verifyUniqueColumn(distinctVec);
        expect(distinctVec.nullCount).toBe(0);
        expect(distinctVec.meta.unique).toBe(true);
    });
});
