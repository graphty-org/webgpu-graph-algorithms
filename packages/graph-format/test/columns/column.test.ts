import { describe, expect, it } from "vitest";

import { bitmapClear, bitmapSet } from "../../src/columns/bitmap.js";
import {
    allocU8,
    assertJsonValue,
    columnFromTypedArray,
    columnFromValues,
    createColumn,
    createEmptyColumn,
    gpuEligibility,
    gpuViewOf,
    metaToDecl,
    partsOf,
    resolveColumnMeta,
    rewrapColumn,
} from "../../src/columns/column.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    type Column,
    type ColumnDeclPatch,
    type ColumnOf,
    type Dtype,
    type TypedArrayData,
} from "../../src/types/index.js";
import { type MutableColumnParts } from "../../src/types/internal.js";
import { canViewAsPaddedU32, paddedU32View } from "../../src/util/typed-array.js";

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

function typedValues<D extends Dtype>(
    dtype: D,
    name: string,
    entries: readonly unknown[],
    decl: ColumnDeclPatch = {},
): ColumnOf<D> {
    const column = values(name, entries, { ...decl, dtype });
    if (column.dtype !== dtype) {
        throw new Error("unreachable");
    }
    return column as ColumnOf<D>;
}

function typedArray<D extends Dtype>(
    dtype: D,
    rows: number,
    name: string,
    data: TypedArrayData,
    decl: ColumnDeclPatch = {},
): ColumnOf<D> {
    const column = columnFromTypedArray("node", rows, name, data, { ...decl, dtype });
    if (column.dtype !== dtype) {
        throw new Error("unreachable");
    }
    return column as ColumnOf<D>;
}

function emptyParts(dtype: Dtype, length: number, extra: ColumnDeclPatch = {}): MutableColumnParts {
    return {
        meta: resolveColumnMeta("c", "node", { dtype, ...extra }),
        length,
        data: null,
        validity: null,
        nullCount: 0,
        dictionary: null,
        offsets: null,
        utf8: null,
        strings: null,
        child: null,
        values: null,
    };
}

describe("allocU8", () => {
    it("allocates padded stores from which a u32 view is constructible", () => {
        const store = allocU8(5);
        expect(store.length).toBe(5);
        expect(store.buffer.byteLength).toBe(8);
        expect(canViewAsPaddedU32(store)).toBe(true);
        expect(paddedU32View(store).length).toBe(2);
        expect(allocU8(0).buffer.byteLength).toBe(0);
        expect(allocU8(8).buffer.byteLength).toBe(8);
    });
});

describe("gpuEligibility", () => {
    it("follows the table of design section 10.4", () => {
        expect(gpuEligibility("f32")).toBe("direct");
        expect(gpuEligibility("i32")).toBe("direct");
        expect(gpuEligibility("u32")).toBe("direct");
        expect(gpuEligibility("dict")).toBe("direct");
        expect(gpuEligibility("u8")).toBe("packed");
        expect(gpuEligibility("bool")).toBe("packed");
        expect(gpuEligibility("f64")).toBe("convert");
        expect(gpuEligibility("string")).toBe("none");
        expect(gpuEligibility("list")).toBe("none");
        expect(gpuEligibility("json")).toBe("none");
        expectError(() => gpuEligibility("f16" as Dtype), "E_COLUMN_TYPE");
    });
});

describe("assertJsonValue", () => {
    it("accepts JSON values including non-finite numbers and rejects the rest", () => {
        expect(() => assertJsonValue(null, "x")).not.toThrow();
        expect(() => assertJsonValue(Infinity, "x")).not.toThrow();
        expect(() => assertJsonValue(NaN, "x")).not.toThrow();
        expect(() => assertJsonValue({ a: [1, "b", { c: true }] }, "x")).not.toThrow();
        expect(() => assertJsonValue(Object.create(null), "x")).not.toThrow();
        expect(expectError(() => assertJsonValue(undefined, "extra"), "E_COLUMN_TYPE").details.field).toBe("extra");
        expectError(() => assertJsonValue(() => 1, "x"), "E_COLUMN_TYPE");
        expectError(() => assertJsonValue(new Map(), "x"), "E_COLUMN_TYPE");
        expectError(() => assertJsonValue({ a: undefined }, "x"), "E_COLUMN_TYPE");
        expectError(() => assertJsonValue(new Uint8Array(1), "x"), "E_COLUMN_TYPE");
        expect(expectError(() => assertJsonValue("\ud800", "x"), "E_COLUMN_TYPE").details.reason).toBe(
            "lone surrogate",
        );
    });
});

describe("resolveColumnMeta", () => {
    it("applies the defaults of design section 5.5", () => {
        const meta = resolveColumnMeta("x", "node", { dtype: "f32" });
        expect(meta).toEqual({
            name: "x",
            domain: "node",
            dtype: "f32",
            components: 1,
            itemDtype: null,
            itemComponents: null,
            nullable: true,
            mutable: false,
            role: null,
            refersTo: null,
            unique: false,
            default: undefined,
            fill: 0,
            options: null,
            origin: null,
            dynamic: false,
            extra: {},
        });
        expect(Object.isFrozen(meta)).toBe(true);
    });

    it("resolves the fill from a representable default", () => {
        expect(resolveColumnMeta("x", "node", { dtype: "f64", default: 2.5 }).fill).toBe(2.5);
        expect(resolveColumnMeta("x", "node", { dtype: "i32", default: 7 }).fill).toBe(7);
        expect(resolveColumnMeta("x", "node", { dtype: "i32", default: 7.5 }).fill).toBe(0);
        expect(resolveColumnMeta("x", "node", { dtype: "u8", default: 300 }).fill).toBe(0);
        expect(resolveColumnMeta("x", "node", { dtype: "u32", default: -1 }).fill).toBe(0);
        expect(resolveColumnMeta("x", "node", { dtype: "f32", components: 3, default: [1, 1, 1] }).fill).toBe(1);
        expect(resolveColumnMeta("x", "node", { dtype: "f32", components: 3, default: [1, 2, 3] }).fill).toBe(0);
        expect(resolveColumnMeta("x", "node", { dtype: "bool", default: true }).fill).toBe(true);
        expect(resolveColumnMeta("x", "node", { dtype: "bool", default: "yes" }).fill).toBe(false);
        expect(resolveColumnMeta("x", "node", { dtype: "dict", default: "b", options: ["a", "b"] }).fill).toBe("b");
        expect(resolveColumnMeta("x", "node", { dtype: "dict", default: "z", options: ["a", "b"] }).fill).toBe("a");
        expect(resolveColumnMeta("x", "node", { dtype: "dict", default: "z" }).fill).toBe("z");
        expect(resolveColumnMeta("x", "node", { dtype: "dict" }).fill).toBe("");
        expect(resolveColumnMeta("x", "node", { dtype: "string", default: "s" }).fill).toBe("");
        expect(resolveColumnMeta("x", "node", { dtype: "string", fill: "-" }).fill).toBe("-");
        expect(resolveColumnMeta("x", "node", { dtype: "f64", default: 2.5, fill: NaN }).fill).toBeNaN();
        expect(resolveColumnMeta("x", "node", { dtype: "u32", refersTo: "node" }).fill).toBe(INVALID_INDEX);
        expect(resolveColumnMeta("x", "node", { dtype: "f64", default: Infinity }).fill).toBe(Infinity);
    });

    it("rejects invalid declarations with E_COLUMN_TYPE and details.field", () => {
        expect(expectError(() => resolveColumnMeta("x", "node", {}), "E_COLUMN_TYPE").details.field).toBe("dtype");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", components: 0 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("components");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", components: 17 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("components");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "bool", components: 2 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("components");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "list" }), "E_COLUMN_TYPE").details.field,
        ).toBe("itemDtype");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "list", itemDtype: "list" as "f32" }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("itemDtype");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", itemDtype: "f32" }), "E_COLUMN_TYPE")
                .details.field,
        ).toBe("itemDtype");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "list", itemDtype: "bool", itemComponents: 2 }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("itemComponents");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", refersTo: "node" }), "E_COLUMN_TYPE")
                .details.field,
        ).toBe("refersTo");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "u32", components: 2, refersTo: "node" }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("refersTo");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "u32", refersTo: "arc" as "node" }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("refersTo");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", default: () => 1 }), "E_COLUMN_TYPE")
                .details.field,
        ).toBe("default");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "f32", options: "a" as unknown as [] }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("options");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "dict", options: [1] }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("options");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "f32", extra: { a: new Date() } }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("extra");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "f32", fill: "0" }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("fill");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "bool", fill: 1 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("fill");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "dict", fill: 0 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("fill");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "string", fill: 0 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("fill");
        expect(
            expectError(() => resolveColumnMeta("x", "node", { dtype: "json", fill: 0 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("fill");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "list", itemDtype: "f32", fill: 0 }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("fill");
        expect(
            expectError(
                () => resolveColumnMeta("x", "node", { dtype: "f32", origin: { id: 3 as unknown as string } }),
                "E_COLUMN_TYPE",
            ).details.field,
        ).toBe("origin");
    });

    it("keeps origin, options, extra and list fields", () => {
        const meta = resolveColumnMeta("spells", "edge", {
            dtype: "list",
            itemDtype: "f64",
            itemComponents: 2,
            role: "spells",
            origin: { format: "gexf", type: "spells" },
            options: ["a"],
            extra: { units: "file" },
            dynamic: true,
            unique: true,
            mutable: true,
            nullable: false,
        });
        expect(meta.itemDtype).toBe("f64");
        expect(meta.itemComponents).toBe(2);
        expect(meta.origin).toEqual({ format: "gexf", id: null, title: null, type: "spells", namespace: null });
        expect(meta.options).toEqual(["a"]);
        expect(meta.extra).toEqual({ units: "file" });
        expect(meta.dynamic).toBe(true);
        expect(meta.unique).toBe(true);
        expect(meta.mutable).toBe(true);
        expect(meta.nullable).toBe(false);
        expect(meta.role).toBe("spells");
        expect(resolveColumnMeta("y", "edge", metaToDecl(meta))).toEqual({ ...meta, name: "y" });
    });
});

describe("numeric columns", () => {
    it("adopts every 4-byte array and f64 by reference, whatever the length", () => {
        for (const data of [new Float32Array(7), new Float64Array(7), new Int32Array(7), new Uint32Array(7)]) {
            const column = columnFromTypedArray("node", 7, "x", data, {});
            expect(column.length).toBe(7);
            expect((column as { data: unknown }).data).toBe(data);
            expect(column.validity).toBeNull();
            expect(column.nullCount).toBe(0);
            expect(column.meta.nullable).toBe(false);
            expect(column.isSet(6)).toBe(true);
            expect(column.isSet(7)).toBe(false);
            expect(column.isSet(-1)).toBe(false);
            expect(column.byteLength).toBe(data.byteLength);
            expect(column.paddedByteLength).toBe(data.byteLength);
        }
        const buffer = new ArrayBuffer(64);
        const window = new Uint32Array(buffer, 8, 3);
        expect(typedArray("u32", 3, "x", window).data).toBe(window);
    });

    it("infers the dtype from the array class and rejects mismatches", () => {
        expect(columnFromTypedArray("node", 2, "x", new Float32Array(2), {}).dtype).toBe("f32");
        expect(columnFromTypedArray("node", 2, "x", new Float64Array(2), {}).dtype).toBe("f64");
        expect(columnFromTypedArray("node", 2, "x", new Int32Array(2), {}).dtype).toBe("i32");
        expect(columnFromTypedArray("node", 2, "x", new Uint32Array(2), {}).dtype).toBe("u32");
        expect(columnFromTypedArray("node", 4, "x", new Uint8Array(4), {}).dtype).toBe("u8");
        expect(columnFromTypedArray("node", 40, "x", new Uint32Array(2), { dtype: "bool" }).dtype).toBe("bool");
        expect(
            columnFromTypedArray("node", 2, "x", new Uint32Array(2), { dtype: "dict", options: ["a", "b"] }).dtype,
        ).toBe("dict");
        expectError(() => columnFromTypedArray("node", 2, "x", new Float32Array(2), { dtype: "f64" }), "E_COLUMN_TYPE");
        expectError(
            () => columnFromTypedArray("node", 2, "x", new Uint32Array(2), { dtype: "string" }),
            "E_COLUMN_TYPE",
        );
        expectError(() => columnFromTypedArray("node", 2, "x", new Int32Array(2), { dtype: "bool" }), "E_COLUMN_TYPE");
    });

    it("checks data.length === rowCount * components", () => {
        const error = expectError(
            () => columnFromTypedArray("node", 3, "x", new Float32Array(5), {}),
            "E_COLUMN_LENGTH",
        );
        expect(error.details).toMatchObject({ expected: 3, found: 5 });
        expect(columnFromTypedArray("node", 3, "x", new Float32Array(9), { components: 3 }).meta.components).toBe(3);
        expectError(
            () => columnFromTypedArray("node", 3, "x", new Float32Array(8), { components: 3 }),
            "E_COLUMN_LENGTH",
        );
        expectError(
            () => columnFromTypedArray("node", 33, "x", new Uint32Array(1), { dtype: "bool" }),
            "E_COLUMN_LENGTH",
        );
    });

    it("reads scalar and multi-component values", () => {
        const scalar = columnFromTypedArray("node", 3, "x", new Float64Array([1.5, 2.5, 3.5]), {});
        expect(scalar.value(1)).toBe(2.5);
        const vec = columnFromTypedArray("node", 2, "p", new Float32Array([1, 2, 3, 4, 5, 6]), { components: 3 });
        const row = vec.value(1) as Float32Array;
        expect(row).toBeInstanceOf(Float32Array);
        expect(Array.from(row)).toEqual([4, 5, 6]);
        expect(row.buffer).toBe((vec as { data: Float32Array }).data.buffer);
        expectError(() => scalar.value(3), "E_INDEX_RANGE");
        expectError(() => scalar.value(-1), "E_INDEX_RANGE");
        expectError(() => scalar.value(1.5), "E_INDEX_RANGE");
    });

    it("builds from JS values with validity, fill and default semantics", () => {
        const column = values("x", [1, undefined, 3, null], { dtype: "i32", default: 9 });
        expect(column.dtype).toBe("i32");
        expect(column.meta.fill).toBe(9);
        expect(column.nullCount).toBe(2);
        expect(column.validity).not.toBeNull();
        expect(column.isSet(0)).toBe(true);
        expect(column.isSet(1)).toBe(false);
        expect(column.isSet(3)).toBe(false);
        expect(Array.from((column as { data: Int32Array }).data)).toEqual([1, 9, 3, 9]);
        expect(column.value(1)).toBe(9);
        expect(column.byteLength).toBe(16 + 4);
        const noDefault = values("x", [1, undefined], { dtype: "i32" });
        expect(noDefault.value(1)).toBeUndefined();
        expect(Array.from((noDefault as { data: Int32Array }).data)).toEqual([1, 0]);
    });

    it("coerces booleans into numbers and rejects unrepresentable values", () => {
        expect(Array.from((values("x", [true, false], { dtype: "u8" }) as { data: Uint8Array }).data)).toEqual([1, 0]);
        expectError(() => values("x", ["1"], { dtype: "i32" }), "E_COLUMN_TYPE");
        expectError(() => values("x", [1.5], { dtype: "i32" }), "E_COLUMN_TYPE");
        expectError(() => values("x", [-1], { dtype: "u32" }), "E_COLUMN_TYPE");
        expectError(() => values("x", [256], { dtype: "u8" }), "E_COLUMN_TYPE");
        expectError(() => values("x", [2 ** 31], { dtype: "i32" }), "E_COLUMN_TYPE");
        expect(values("x", [1.5, NaN], { dtype: "f32" }).value(0)).toBe(1.5);
        expectError(() => values("x", [undefined], { dtype: "f32", nullable: false }), "E_COLUMN_TYPE");
    });

    it("builds multi-component rows from arrays or a broadcast number", () => {
        const column = values("p", [[1, 2, 3], 5, undefined], { dtype: "f32", components: 3 });
        expect(Array.from((column as { data: Float32Array }).data)).toEqual([1, 2, 3, 5, 5, 5, 0, 0, 0]);
        expectError(() => values("p", [[1, 2]], { dtype: "f32", components: 3 }), "E_COLUMN_TYPE");
        expectError(() => values("p", [[1, "2", 3]], { dtype: "f32", components: 3 }), "E_COLUMN_TYPE");
    });

    it("slices zero-copy for 4-byte and f64 data and copies bitmaps at unaligned starts", () => {
        const entries: (number | undefined)[] = [];
        for (let i = 0; i < 70; i++) {
            entries.push(i % 3 === 0 ? undefined : i);
        }
        const column = typedValues("f64", "x", entries, { components: 1 });
        const aligned = column.slice(32, 70);
        expect(aligned.length).toBe(38);
        expect(aligned.data.buffer).toBe(column.data.buffer);
        expect(aligned.data.byteOffset).toBe(32 * 8);
        expect(aligned.validity?.buffer).toBe(column.validity?.buffer);
        expect(aligned.nullCount).toBe(entries.slice(32).filter((v) => v === undefined).length);
        const unaligned = column.slice(5, 40);
        expect(unaligned.data.buffer).toBe(column.data.buffer);
        expect(unaligned.validity?.buffer).not.toBe(column.validity?.buffer);
        expect(unaligned.nullCount).toBe(entries.slice(5, 40).filter((v) => v === undefined).length);
        for (let i = 0; i < 35; i++) {
            expect(unaligned.isSet(i)).toBe(entries[5 + i] !== undefined);
            if (entries[5 + i] !== undefined) {
                expect(unaligned.value(i)).toBe(entries[5 + i]);
            }
        }
        expectError(() => column.slice(5, 3), "E_INDEX_RANGE");
        expectError(() => column.slice(0, 71), "E_INDEX_RANGE");
    });

    it("clones deeply", () => {
        const column = typedValues("u32", "x", [1, undefined, 3]);
        const copy = column.clone();
        expect(copy).not.toBe(column);
        expect(copy.data).not.toBe(column.data);
        expect(Array.from(copy.data)).toEqual(Array.from(column.data));
        expect(copy.validity).not.toBe(column.validity);
        expect(copy.nullCount).toBe(1);
        expect(copy.meta).toBe(column.meta);
    });
});

describe("u8 columns", () => {
    it("adopts aligned arrays and copies (or refuses) unaligned ones", () => {
        const aligned = allocU8(5);
        const column = columnFromTypedArray("node", 5, "x", aligned, {});
        expect(column.dtype).toBe("u8");
        expect((column as { data: Uint8Array }).data).toBe(aligned);
        const unaligned = new Uint8Array(5);
        unaligned.set([1, 2, 3, 4, 5]);
        const copied = columnFromTypedArray("node", 5, "x", unaligned, {});
        expect((copied as { data: Uint8Array }).data).not.toBe(unaligned);
        expect(Array.from((copied as { data: Uint8Array }).data)).toEqual([1, 2, 3, 4, 5]);
        expect(copied.paddedU32View().length).toBe(2);
        const error = expectError(
            () => columnFromTypedArray("node", 5, "x", unaligned, {}, "strict"),
            "E_COLUMN_ALIGNMENT",
        );
        expect(error.details.column).toBe("x");
        const offset = new Uint8Array(new ArrayBuffer(16), 2, 5);
        expectError(() => columnFromTypedArray("node", 5, "x", offset, {}, "strict"), "E_COLUMN_ALIGNMENT");
    });

    it("reports padded byte lengths and a padded u32 view over the column range", () => {
        const store = allocU8(6);
        store.set([1, 2, 3, 4, 5, 6]);
        const column = columnFromTypedArray("node", 6, "x", store, {});
        expect(column.byteLength).toBe(6);
        expect(column.paddedByteLength).toBe(8);
        const view = column.paddedU32View();
        expect(view.length).toBe(2);
        expect(view[0]).toBe(0x04030201);
        expect(view.buffer).toBe(store.buffer);
        expect(gpuViewOf(column)).toEqual(view);
        expect(column.gpu).toBe("packed");
    });

    it("slices zero-copy at 4-aligned starts and copies otherwise; clones into padded stores", () => {
        const store = allocU8(12);
        for (let i = 0; i < 12; i++) {
            store[i] = i;
        }
        const column = typedArray("u8", 12, "x", store);
        const aligned = column.slice(4, 10);
        expect(aligned.data.buffer).toBe(store.buffer);
        expect(aligned.data.byteOffset).toBe(4);
        expect(canViewAsPaddedU32(aligned.data)).toBe(true);
        expect(aligned.paddedU32View().length).toBe(2);
        const unaligned = column.slice(3, 10);
        expect(unaligned.data.buffer).not.toBe(store.buffer);
        expect(Array.from(unaligned.data)).toEqual([3, 4, 5, 6, 7, 8, 9]);
        expect(canViewAsPaddedU32(unaligned.data)).toBe(true);
        const copy = column.clone();
        expect(copy.data.buffer.byteLength).toBe(12);
        expect(canViewAsPaddedU32(copy.data)).toBe(true);
        const oddCopy = column.slice(0, 5).clone();
        expect(oddCopy.data.length).toBe(5);
        expect(canViewAsPaddedU32(oddCopy.data)).toBe(true);
        const stride = typedArray("u8", 3, "x", store, { components: 4 });
        expect(stride.slice(1, 3).data.byteOffset).toBe(4);
    });
});

describe("bool columns", () => {
    it("packs bits LSB-first and reads them back", () => {
        const entries = [true, false, undefined, true];
        const column = values("b", entries, { dtype: "bool" });
        expect(column.dtype).toBe("bool");
        if (column.dtype !== "bool") {
            throw new Error("unreachable");
        }
        expect(column.data.length).toBe(1);
        expect(column.data[0]).toBe(0b1001);
        expect(column.value(0)).toBe(true);
        expect(column.value(1)).toBe(false);
        expect(column.value(2)).toBeUndefined();
        expect(column.nullCount).toBe(1);
        expect(column.paddedU32View()).toBe(column.data);
        expect(gpuViewOf(column)).toBe(column.data);
        expect(column.byteLength).toBe(8);
        expectError(() => values("b", [1], { dtype: "bool" }), "E_COLUMN_TYPE");
    });

    it("uses a true default as the fill", () => {
        const column = values("b", [false, undefined], { dtype: "bool", default: true });
        if (column.dtype !== "bool") {
            throw new Error("unreachable");
        }
        expect(column.meta.fill).toBe(true);
        // the trailing lanes of the last word are undefined (here: the fill), so only the two rows count
        expect(column.data[0] & 0b11).toBe(0b10);
        expect(column.value(1)).toBe(true);
        expect(column.materializeDefault()).toBe(column);
    });

    it("slices at word boundaries zero-copy and elsewhere by copying", () => {
        const entries: boolean[] = [];
        for (let i = 0; i < 100; i++) {
            entries.push(i % 7 === 0);
        }
        const column = values("b", entries, { dtype: "bool" });
        if (column.dtype !== "bool") {
            throw new Error("unreachable");
        }
        const aligned = column.slice(64, 100);
        expect(aligned.data.buffer).toBe(column.data.buffer);
        expect(aligned.data.length).toBe(2);
        const unaligned = column.slice(3, 70);
        expect(unaligned.data.buffer).not.toBe(column.data.buffer);
        for (let i = 0; i < 67; i++) {
            expect(unaligned.value(i)).toBe(entries[3 + i]);
        }
        const copy = column.clone();
        expect(copy.data).not.toBe(column.data);
        expect(Array.from(copy.data)).toEqual(Array.from(column.data));
    });
});

describe("dict columns", () => {
    it("interns in first-seen order after the declared options", () => {
        const column = values("k", ["b", "a", undefined, "c", "a"], { dtype: "dict", options: ["a"] });
        if (column.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(column.dictionary).toEqual(["a", "b", "c"]);
        expect(Array.from(column.codes)).toEqual([1, 0, 0, 2, 0]);
        expect(column.value(0)).toBe("b");
        expect(column.value(2)).toBeUndefined();
        expect(column.codeOf("c")).toBe(2);
        expect(column.codeOf("zzz")).toBe(INVALID_INDEX);
        expect(column.paddedU32View()).toBe(column.codes);
        expect(gpuViewOf(column)).toBe(column.codes);
        expect(column.gpu).toBe("direct");
        expect(column.byteLength).toBe(20 + 4);
        expectError(() => values("k", [{}], { dtype: "dict" }), "E_COLUMN_TYPE");
    });

    it("stores the default's code in unset rows when the default is a member", () => {
        const column = values("k", [undefined, "x"], { dtype: "dict", default: "d", options: ["d", "x"] });
        if (column.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(column.meta.fill).toBe("d");
        expect(Array.from(column.codes)).toEqual([0, 1]);
        expect(column.value(0)).toBe("d");
        expect(column.materializeDefault()).toBe(column);
        const seeded = values("k", [undefined, "x"], { dtype: "dict", default: "d" });
        if (seeded.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(seeded.dictionary).toEqual(["d", "x"]);
        expect(seeded.materializeDefault()).toBe(seeded);
    });

    it("materializes a default that is not a member into a copy with an extended dictionary", () => {
        const column = values("k", [undefined, "x"], { dtype: "dict", default: "d", options: ["x", "y"] });
        if (column.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(column.meta.fill).toBe("x");
        const materialized = column.materializeDefault();
        expect(materialized).not.toBe(column);
        expect(materialized.dictionary).toEqual(["x", "y", "d"]);
        expect(Array.from(materialized.codes)).toEqual([2, 0]);
        expect(materialized.isSet(0)).toBe(false);
        expect(materialized.value(0)).toBe("d");
        expect(column.materializeDefault()).toBe(materialized);
    });

    it("slices and clones", () => {
        const column = values("k", ["a", "b", "c"], { dtype: "dict" });
        if (column.dtype !== "dict") {
            throw new Error("unreachable");
        }
        const slice = column.slice(1, 3);
        expect(Array.from(slice.codes)).toEqual([1, 2]);
        expect(slice.dictionary).toBe(column.dictionary);
        expect(slice.codes.buffer).toBe(column.codes.buffer);
        const copy = column.clone();
        expect(copy.dictionary).not.toBe(column.dictionary);
        expect(copy.dictionary).toEqual(column.dictionary);
        expect(copy.codeOf("b")).toBe(1);
    });

    it("rejects lone surrogates", () => {
        expect(expectError(() => values("k", ["\udc00"], { dtype: "dict" }), "E_COLUMN_TYPE").details.reason).toBe(
            "lone surrogate",
        );
    });
});

describe("string columns", () => {
    it("keeps the decoded array and materialises the Utf8 store lazily", () => {
        const column = values("s", ["ab", "", undefined, "\u00e9\u4e2d"], { dtype: "string" });
        if (column.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(column.valueAt(0)).toBe("ab");
        expect(column.valueAt(2)).toBe("");
        expect(column.value(2)).toBeUndefined();
        expect(column.nullCount).toBe(1);
        expect(column.decodeAll()).toEqual(["ab", "", "", "\u00e9\u4e2d"]);
        expect(Array.from(column.offsets)).toEqual([0, 2, 2, 2, 7]);
        expect(column.utf8.length).toBe(7);
        expect(canViewAsPaddedU32(column.utf8)).toBe(true);
        expect(column.byteLength).toBe(5 * 4 + 7 + 4);
        expect(column.gpu).toBe("none");
        expectError(() => column.paddedU32View(), "E_GPU_INELIGIBLE");
        expectError(() => gpuViewOf(column), "E_GPU_INELIGIBLE");
        expect(expectError(() => values("s", ["\ud800x"], { dtype: "string" }), "E_COLUMN_TYPE").details.reason).toBe(
            "lone surrogate",
        );
        expectError(() => values("s", [{}], { dtype: "string" }), "E_COLUMN_TYPE");
    });

    it("decodes lazily from a Utf8 store and caches per row", () => {
        const utf8 = allocU8(5);
        utf8.set([104, 105, 121, 111, 33]);
        const parts = emptyParts("string", 3);
        parts.offsets = new Uint32Array([0, 2, 2, 5]);
        parts.utf8 = utf8;
        const column = createColumn(parts);
        if (column.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(column.valueAt(0)).toBe("hi");
        expect(column.valueAt(1)).toBe("");
        expect(column.valueAt(2)).toBe("yo!");
        expect(column.value(2)).toBe("yo!");
        expect(column.offsets).toBe(parts.offsets);
        expect(column.utf8).toBe(utf8);
        expect(column.decodeAll()).toEqual(["hi", "", "yo!"]);
    });

    it("validates the store lengths", () => {
        const parts = emptyParts("string", 2);
        parts.offsets = new Uint32Array([0, 1]);
        parts.utf8 = allocU8(1);
        expectError(() => createColumn(parts), "E_COLUMN_LENGTH");
        parts.offsets = new Uint32Array([0, 1, 3]);
        expectError(() => createColumn(parts), "E_COLUMN_LENGTH");
        const noStore = emptyParts("string", 2);
        noStore.strings = ["a"];
        expectError(() => createColumn(noStore), "E_COLUMN_LENGTH");
        noStore.strings = ["a", undefined];
        expectError(() => createColumn(noStore), "E_COLUMN_TYPE");
    });

    it("slices zero-copy over the bytes and clones both representations", () => {
        const column = values("s", ["aa", "bbb", "c", "dddd"], { dtype: "string" });
        if (column.dtype !== "string") {
            throw new Error("unreachable");
        }
        const lazy = column.slice(1, 3);
        expect(lazy.decodeAll()).toEqual(["bbb", "c"]);
        expect(column.utf8.length).toBe(10);
        const slice = column.slice(1, 3);
        expect(Array.from(slice.offsets)).toEqual([0, 3, 4]);
        expect(slice.utf8.buffer).toBe(column.utf8.buffer);
        expect(slice.utf8.byteOffset).toBe(2);
        expect(slice.valueAt(1)).toBe("c");
        const copy = column.clone();
        expect(copy.utf8).not.toBe(column.utf8);
        expect(copy.decodeAll()).toEqual(column.decodeAll());
    });

    it("materializes a string default into a copy", () => {
        const column = values("s", ["a", undefined], { dtype: "string", default: "dflt" });
        if (column.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(column.value(1)).toBe("dflt");
        expect(column.valueAt(1)).toBe("");
        const materialized = column.materializeDefault();
        expect(materialized).not.toBe(column);
        expect(materialized.decodeAll()).toEqual(["a", "dflt"]);
        expect(materialized.isSet(1)).toBe(false);
        const explicitFill = values("s", ["a", undefined], { dtype: "string", default: "dflt", fill: "dflt" });
        if (explicitFill.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(explicitFill.valueAt(1)).toBe("dflt");
        expect(explicitFill.materializeDefault()).toBe(explicitFill);
    });
});

describe("list columns", () => {
    it("stores offsets plus a non-nullable child and distinguishes unset from empty", () => {
        const column = values("l", [[1, 2], [], undefined, [3]], { dtype: "list", itemDtype: "i32" });
        if (column.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(Array.from(column.offsets)).toEqual([0, 2, 2, 2, 3]);
        expect(column.child.dtype).toBe("i32");
        expect(column.child.validity).toBeNull();
        expect(column.child.meta.nullable).toBe(false);
        expect(column.sliceOf(0)).toEqual([1, 2]);
        expect(column.sliceOf(1)).toEqual([]);
        expect(column.value(1)).toEqual([]);
        expect(column.isSet(1)).toBe(true);
        expect(column.isSet(2)).toBe(false);
        expect(column.value(2)).toBeUndefined();
        expect(column.value(3)).toEqual([3]);
        expect(column.byteLength).toBe(5 * 4 + 3 * 4 + 4);
        expectError(() => values("l", [1], { dtype: "list", itemDtype: "i32" }), "E_COLUMN_TYPE");
        expectError(() => values("l", [["x"]], { dtype: "list", itemDtype: "i32" }), "E_COLUMN_TYPE");
    });

    it("supports multi-component children (spells)", () => {
        const column = values(
            "spells",
            [
                [
                    [0, 1],
                    [2, 3],
                ],
                [[4, 5]],
            ],
            { dtype: "list", itemDtype: "f64", itemComponents: 2 },
        );
        if (column.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(column.child.length).toBe(3);
        expect(column.child.meta.components).toBe(2);
        const items = column.sliceOf(0) as ArrayLike<number>[];
        expect(Array.from(items[1])).toEqual([2, 3]);
    });

    it("slices and clones through the child", () => {
        const column = values("l", [["a"], ["b", "c"], ["d"]], { dtype: "list", itemDtype: "string" });
        if (column.dtype !== "list") {
            throw new Error("unreachable");
        }
        const slice = column.slice(1, 3);
        expect(Array.from(slice.offsets)).toEqual([0, 2, 3]);
        expect(slice.child.length).toBe(3);
        expect(slice.sliceOf(0)).toEqual(["b", "c"]);
        expect(slice.sliceOf(1)).toEqual(["d"]);
        const copy = column.clone();
        expect(copy.child).not.toBe(column.child);
        expect(copy.sliceOf(1)).toEqual(["b", "c"]);
    });

    it("materializes an array default", () => {
        const column = values("l", [[1], undefined], { dtype: "list", itemDtype: "u8", default: [7, 8] });
        if (column.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(column.value(1)).toEqual([7, 8]);
        const materialized = column.materializeDefault();
        expect(materialized.sliceOf(1)).toEqual([7, 8]);
        expect(Array.from(materialized.offsets)).toEqual([0, 1, 3]);
        expect(materialized.isSet(1)).toBe(false);
        expect(column.materializeDefault()).toBe(materialized);
        expectError(
            () => values("l", [undefined], { dtype: "list", itemDtype: "u8", default: 7 }).materializeDefault(),
            "E_COLUMN_TYPE",
        );
    });

    it("validates the child at construction", () => {
        const parts = emptyParts("list", 2, { itemDtype: "i32" });
        parts.offsets = new Uint32Array([0, 1, 2]);
        parts.child = values("item", [1, 2], { dtype: "i32" });
        expectError(() => createColumn(parts), "E_COLUMN_TYPE");
        parts.child = values("item", [1, 2], { dtype: "f32", nullable: false });
        expectError(() => createColumn(parts), "E_COLUMN_TYPE");
        parts.child = values("item", [1], { dtype: "i32", nullable: false });
        expectError(() => createColumn(parts), "E_COLUMN_LENGTH");
        parts.child = values("item", [[1]], { dtype: "list", itemDtype: "i32", nullable: false });
        expectError(() => createColumn(parts), "E_COLUMN_TYPE");
        parts.child = null;
        expectError(() => createColumn(parts), "E_COLUMN_TYPE");
        parts.offsets = null;
        expectError(() => createColumn(parts), "E_COLUMN_LENGTH");
    });
});

describe("json columns", () => {
    it("holds one value per row with null as a value and undefined as unset", () => {
        const column = values("j", [{ a: 1 }, null, undefined, [1, 2]], { dtype: "json" });
        if (column.dtype !== "json") {
            throw new Error("unreachable");
        }
        expect(column.values.length).toBe(4);
        expect(column.value(0)).toEqual({ a: 1 });
        expect(column.value(1)).toBeNull();
        expect(column.isSet(1)).toBe(true);
        expect(column.isSet(2)).toBe(false);
        expect(column.value(2)).toBeUndefined();
        expect(column.nullCount).toBe(1);
        expect(column.byteLength).toBe(4);
        expect(column.gpu).toBe("none");
        expectError(() => values("j", [new Map()], { dtype: "json" }), "E_COLUMN_TYPE");
        const withDefault = values("j", [undefined], { dtype: "json", default: { d: true } });
        expect(withDefault.value(0)).toEqual({ d: true });
        const materialized = withDefault.materializeDefault();
        if (materialized.dtype !== "json") {
            throw new Error("unreachable");
        }
        expect(materialized.values[0]).toEqual({ d: true });
        expect(materialized.isSet(0)).toBe(false);
        expect(column.slice(1, 3).values).toEqual([null, undefined]);
        expect(column.clone().values).toEqual(column.values);
    });
});

describe("inference from JS arrays without a dtype", () => {
    it("widens per design section 5.1 and picks list for arrays of arrays", () => {
        expect(values("x", [true, false]).dtype).toBe("bool");
        expect(values("x", [1, 2]).dtype).toBe("i32");
        expect(values("x", [1, 2.5]).dtype).toBe("f64");
        expect(values("x", [1, "01"]).dtype).toBe("string");
        expect(values("x", [1, "01"]).value(0)).toBe("1");
        expect(values("x", [true, 5]).value(0)).toBe(1);
        expect(values("x", [1, { a: 1 }]).dtype).toBe("json");
        expect(values("x", [undefined, null]).dtype).toBe("json");
        const list = values("x", [[1, 2], [3]]);
        expect(list.dtype).toBe("list");
        expect(list.meta.itemDtype).toBe("i32");
        const mixed = values("x", [[1], "a"]);
        expect(mixed.dtype).toBe("json");
        expectError(() => values("x", [1n]), "E_COLUMN_TYPE");
    });

    it("checks the value count", () => {
        expect(
            expectError(() => columnFromValues("node", 3, "x", [1, 2], {}), "E_COLUMN_LENGTH").details,
        ).toMatchObject({
            expected: 3,
            found: 2,
        });
    });
});

describe("mutable gating (design section 5.8)", () => {
    it("throws E_COLUMN_IMMUTABLE for every mutator on an immutable column", () => {
        const column = values("x", [1, 2], { dtype: "f32" });
        if (column.dtype !== "f32") {
            throw new Error("unreachable");
        }
        expectError(() => column.mutableData(), "E_COLUMN_IMMUTABLE");
        expectError(() => column.markDirty(), "E_COLUMN_IMMUTABLE");
        expectError(() => column.mutableValidity(), "E_COLUMN_IMMUTABLE");
        expectError(() => column.setAll(), "E_COLUMN_IMMUTABLE");
        for (const dtype of ["bool", "dict"] as const) {
            const packed = values("x", dtype === "bool" ? [true] : ["a"], { dtype });
            expectError(() => (packed as { mutableData(): unknown }).mutableData(), "E_COLUMN_IMMUTABLE");
        }
    });

    it("lets a mutable column be written, marked dirty and set-all", () => {
        const column = values("p", [1, undefined, 3], { dtype: "f64", mutable: true, default: 0.5 });
        if (column.dtype !== "f64") {
            throw new Error("unreachable");
        }
        expect(column.version).toBe(0);
        const before = gpuViewOf(column);
        expect(before).toBeInstanceOf(Float32Array);
        expect(Array.from(before)).toEqual([1, 0.5, 3]);
        expect(gpuViewOf(column)).toBe(before);
        const data = column.mutableData();
        expect(data).toBe(column.data);
        data[0] = 42;
        const materialized = column.materializeDefault();
        expect(materialized).toBe(column);
        column.markDirty();
        expect(column.version).toBe(1);
        const after = gpuViewOf(column);
        expect(after).not.toBe(before);
        expect(after[0]).toBe(42);
        const bits = column.mutableValidity();
        expect(bits).toBe(column.validity);
        expect(bits).not.toBeNull();
        if (bits === null) {
            throw new Error("unreachable");
        }
        bitmapSet(bits, 1);
        expect(column.nullCount).toBe(1);
        column.markDirty();
        expect(column.nullCount).toBe(0);
        bitmapClear(bits, 2);
        column.markDirty();
        expect(column.nullCount).toBe(1);
        column.setAll();
        expect(column.validity).toBeNull();
        expect(column.nullCount).toBe(0);
        expect(column.version).toBe(4);
        expect(column.mutableValidity()).toBeNull();
    });

    it("invalidates the materializeDefault copy on markDirty", () => {
        const column = values("s", ["a", undefined], { dtype: "string", mutable: true, default: "d" });
        const first = column.materializeDefault();
        expect(column.materializeDefault()).toBe(first);
        column.markDirty();
        expect(column.materializeDefault()).not.toBe(first);
    });
});

describe("materializeDefault", () => {
    it("throws E_NO_DEFAULT without a declared default", () => {
        expectError(() => values("x", [1, undefined], { dtype: "i32" }).materializeDefault(), "E_NO_DEFAULT");
    });

    it("returns the column itself when fill equals the default or nothing is unset", () => {
        const column = values("x", [1, undefined], { dtype: "i32", default: 4 });
        expect(column.materializeDefault()).toBe(column);
        const full = values("x", [1, 2], { dtype: "i32", default: 4, fill: 0 });
        expect(full.materializeDefault()).toBe(full);
    });

    it("writes the default into unset rows of a copy when the fill differs", () => {
        const column = values("x", [1, undefined], { dtype: "f64", default: 4, fill: NaN });
        if (column.dtype !== "f64") {
            throw new Error("unreachable");
        }
        expect(column.data[1]).toBeNaN();
        const materialized = column.materializeDefault();
        expect(materialized).not.toBe(column);
        expect(Array.from(materialized.data)).toEqual([1, 4]);
        expect(materialized.isSet(1)).toBe(false);
        expect(materialized.nullCount).toBe(1);
        expect(column.materializeDefault()).toBe(materialized);
        const vec = values("p", [undefined], { dtype: "f32", components: 3, default: [1, 2, 3] });
        if (vec.dtype !== "f32") {
            throw new Error("unreachable");
        }
        expect(Array.from(vec.materializeDefault().data)).toEqual([1, 2, 3]);
        const bad = values("p", [undefined], { dtype: "i32", default: 1.5 });
        expectError(() => bad.materializeDefault(), "E_COLUMN_TYPE");
        const bool = values("b", [undefined, false], { dtype: "bool", default: true, fill: false });
        if (bool.dtype !== "bool") {
            throw new Error("unreachable");
        }
        expect(bool.materializeDefault().data[0]).toBe(1);
    });
});

describe("createColumn (factory) validation", () => {
    it("checks the slot classes and lengths", () => {
        const f32 = emptyParts("f32", 2);
        f32.data = new Float64Array(2);
        expectError(() => createColumn(f32), "E_COLUMN_TYPE");
        f32.data = new Float32Array(3);
        expectError(() => createColumn(f32), "E_COLUMN_LENGTH");
        const bool = emptyParts("bool", 40);
        bool.data = new Uint32Array(1);
        expectError(() => createColumn(bool), "E_COLUMN_LENGTH");
        bool.data = new Uint8Array(8);
        expectError(() => createColumn(bool), "E_COLUMN_TYPE");
        const dict = emptyParts("dict", 2);
        dict.data = new Uint32Array(2);
        expectError(() => createColumn(dict), "E_COLUMN_TYPE");
        dict.dictionary = ["a"];
        expect(createColumn(dict).dtype).toBe("dict");
        dict.data = new Uint32Array(3);
        expectError(() => createColumn(dict), "E_COLUMN_LENGTH");
        const json = emptyParts("json", 2);
        json.values = [1];
        expectError(() => createColumn(json), "E_COLUMN_LENGTH");
        const u8 = emptyParts("u8", 5);
        u8.data = new Uint8Array(5);
        expectError(() => createColumn(u8), "E_COLUMN_ALIGNMENT");
        const negative = emptyParts("f32", -1);
        expectError(() => createColumn(negative), "E_COLUMN_LENGTH");
    });

    it("checks the validity slot and recomputes nullCount", () => {
        const parts = emptyParts("f32", 40);
        parts.data = new Float32Array(40);
        parts.validity = new Uint32Array(1);
        expectError(() => createColumn(parts), "E_COLUMN_LENGTH");
        parts.validity = new Uint32Array(2).fill(0xffffffff);
        bitmapClear(parts.validity, 3);
        bitmapClear(parts.validity, 39);
        parts.nullCount = 99;
        const column = createColumn(parts);
        expect(column.nullCount).toBe(2);
        const strict = emptyParts("f32", 40, { nullable: false });
        strict.data = new Float32Array(40);
        strict.validity = new Uint32Array(2);
        expectError(() => createColumn(strict), "E_COLUMN_TYPE");
        const empty = emptyParts("json", 0);
        empty.values = [];
        empty.validity = new Uint32Array(0);
        expect(createColumn(empty).length).toBe(0);
        expect(createColumn(empty).validity?.length).toBe(0);
    });
});

describe("createEmptyColumn", () => {
    it("creates unset rows holding the fill for every dtype", () => {
        expect(
            Array.from(
                (createEmptyColumn("node", 3, { name: "a", dtype: "f32", default: 2 }) as { data: Float32Array }).data,
            ),
        ).toEqual([2, 2, 2]);
        expect(createEmptyColumn("node", 3, { name: "a", dtype: "f32" }).nullCount).toBe(3);
        expect(createEmptyColumn("node", 3, { name: "a", dtype: "f32", nullable: false }).nullCount).toBe(0);
        expect(createEmptyColumn("node", 3, { name: "a", dtype: "f32", nullable: false }).validity).toBeNull();
        const bool = createEmptyColumn("node", 3, { name: "a", dtype: "bool", default: true });
        if (bool.dtype !== "bool") {
            throw new Error("unreachable");
        }
        // every bit at a position >= rows stays clear (the bitmap convention of src/columns/bitmap.ts)
        expect(bool.data[0]).toBe(0b111);
        const dict = createEmptyColumn("node", 2, { name: "a", dtype: "dict", options: ["x", "y"], default: "y" });
        if (dict.dtype !== "dict") {
            throw new Error("unreachable");
        }
        expect(Array.from(dict.codes)).toEqual([1, 1]);
        expect(dict.dictionary).toEqual(["x", "y"]);
        const refs = createEmptyColumn("edge", 2, { name: "pair", dtype: "u32", refersTo: "edge" });
        if (refs.dtype !== "u32") {
            throw new Error("unreachable");
        }
        expect(Array.from(refs.data)).toEqual([INVALID_INDEX, INVALID_INDEX]);
        const str = createEmptyColumn("node", 2, { name: "a", dtype: "string" });
        if (str.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(str.decodeAll()).toEqual(["", ""]);
        const list = createEmptyColumn("node", 2, { name: "a", dtype: "list", itemDtype: "u32" });
        if (list.dtype !== "list") {
            throw new Error("unreachable");
        }
        expect(Array.from(list.offsets)).toEqual([0, 0, 0]);
        expect(list.child.length).toBe(0);
        const json = createEmptyColumn("node", 2, { name: "a", dtype: "json" });
        if (json.dtype !== "json") {
            throw new Error("unreachable");
        }
        expect(json.values).toEqual([undefined, undefined]);
        const u8 = createEmptyColumn("node", 3, { name: "a", dtype: "u8", components: 3, fill: 7 });
        if (u8.dtype !== "u8") {
            throw new Error("unreachable");
        }
        expect(u8.data.length).toBe(9);
        expect(canViewAsPaddedU32(u8.data)).toBe(true);
        expect(u8.data[8]).toBe(7);
    });
});

describe("partsOf and rewrapColumn", () => {
    it("re-wraps under new metadata sharing the storage", () => {
        const column = values("x", [1, undefined], { dtype: "u32", role: "rank" });
        const meta = resolveColumnMeta("y", "edge", { ...metaToDecl(column.meta), role: "community" });
        const rewrapped = rewrapColumn(column, meta);
        expect(rewrapped.meta.name).toBe("y");
        expect(rewrapped.meta.domain).toBe("edge");
        expect(rewrapped.meta.role).toBe("community");
        expect((rewrapped as { data: Uint32Array }).data).toBe((column as { data: Uint32Array }).data);
        expect(rewrapped.validity).toBe(column.validity);
        expect(rewrapped.nullCount).toBe(1);
        expectError(() => rewrapColumn(column, resolveColumnMeta("y", "node", { dtype: "i32" })), "E_COLUMN_TYPE");
        expectError(
            () => rewrapColumn(column, resolveColumnMeta("y", "node", { dtype: "u32", fill: 5 })),
            "E_COLUMN_TYPE",
        );
        expectError(
            () => rewrapColumn(column, resolveColumnMeta("y", "node", { dtype: "u32", nullable: false })),
            "E_COLUMN_TYPE",
        );
        const full = values("x", [1, 2], { dtype: "u32" });
        const nonNullable = rewrapColumn(full, resolveColumnMeta("x", "node", { dtype: "u32", nullable: false }));
        expect(nonNullable.validity).toBeNull();
        expect(nonNullable.meta.nullable).toBe(false);
    });

    it("exposes every representation of a string column", () => {
        const lazy = values("s", ["a", "b"], { dtype: "string" });
        const parts = partsOf(lazy);
        expect(parts.strings).toEqual(["a", "b"]);
        expect(parts.offsets).toBeNull();
        if (lazy.dtype !== "string") {
            throw new Error("unreachable");
        }
        expect(lazy.offsets.length).toBe(3);
        const after = partsOf(lazy);
        expect(after.offsets).not.toBeNull();
        expect(after.utf8).not.toBeNull();
        const list = values("l", [[1]], { dtype: "list", itemDtype: "i32" });
        expect(partsOf(list).child).not.toBeNull();
        const json = values("j", [1], { dtype: "json" });
        expect(partsOf(json).values).toEqual([1]);
        const dict = values("d", ["a"], { dtype: "dict" });
        expect(partsOf(dict).dictionary).toEqual(["a"]);
    });
});
