import {
    type AttributeTable,
    type BoolColumn,
    type Column,
    type ColumnDecl,
    type ColumnDeclPatch,
    type ColumnMeta,
    type ColumnOf,
    type ColumnRole,
    type DictColumn,
    type Dtype,
    type DtypeValue,
    type EdgeMask,
    type F32,
    type F32Column,
    type F64Column,
    type GraphSnapshot,
    type I32,
    type I32Column,
    type JsonColumn,
    type KnownColumnRole,
    type ListColumn,
    type Loose,
    type NodeMask,
    type ScalarDtype,
    type StringColumn,
    type U8Column,
    type U32,
    type U32Column,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

declare const snapshot: GraphSnapshot;
declare const i: number;
declare const e: number;
declare function use(...values: unknown[]): void;

// ---- the column accessor example of design section 12.3, verbatim: compiles under both
// noUncheckedIndexedAccess settings, no `as number`, no non-null assertion
const pos = snapshot.nodes.requireTyped("position", "f32"); // F32Column; throws E_COLUMN_TYPE otherwise
const stride = pos.meta.components; // 3
const x = pos.data[i * stride]; // number (number | undefined under the strict flag)
const kind = snapshot.edges.byRole("kind"); // Column | null
if (kind !== null && kind.dtype === "dict") {
    const code = kind.codes[e];
    const label = code === undefined ? undefined : kind.dictionary[code];
    use(label);
    expectTypeOf(kind).toEqualTypeOf<DictColumn>();
    expectTypeOf(label).toEqualTypeOf<string | undefined>();
}
const { rowPtr, colIdx } = snapshot; // hot loops read plain numbers
use(x, rowPtr, colIdx);
expectTypeOf(pos).toEqualTypeOf<F32Column>();
expectTypeOf(stride).toBeNumber();
expectTypeOf(kind).toEqualTypeOf<Column | null>();

// ---- ColumnOf<D> narrowing
expectTypeOf<ColumnOf<"f32">>().toEqualTypeOf<F32Column>();
expectTypeOf<ColumnOf<"f64">>().toEqualTypeOf<F64Column>();
expectTypeOf<ColumnOf<"i32">>().toEqualTypeOf<I32Column>();
expectTypeOf<ColumnOf<"u32">>().toEqualTypeOf<U32Column>();
expectTypeOf<ColumnOf<"u8">>().toEqualTypeOf<U8Column>();
expectTypeOf<ColumnOf<"bool">>().toEqualTypeOf<BoolColumn>();
expectTypeOf<ColumnOf<"dict">>().toEqualTypeOf<DictColumn>();
expectTypeOf<ColumnOf<"string">>().toEqualTypeOf<StringColumn>();
expectTypeOf<ColumnOf<"list">>().toEqualTypeOf<ListColumn>();
expectTypeOf<ColumnOf<"json">>().toEqualTypeOf<JsonColumn>();
expectTypeOf<ColumnOf<Dtype>>().toEqualTypeOf<Column>();
expectTypeOf<ScalarDtype>().toEqualTypeOf<Exclude<Dtype, "list">>();

// ---- discriminated narrowing by dtype
declare const column: Column;
if (column.dtype === "string") {
    expectTypeOf(column).toEqualTypeOf<StringColumn>();
    expectTypeOf(column.valueAt(0)).toBeString();
    expectTypeOf(column.decodeAll()).toEqualTypeOf<string[]>();
} else if (column.dtype === "list") {
    expectTypeOf(column.child).toEqualTypeOf<Exclude<Column, ListColumn>>();
    expectTypeOf(column.child.dtype).toEqualTypeOf<ScalarDtype>();
    expectTypeOf(column.sliceOf(0)).toEqualTypeOf<readonly unknown[]>();
} else if (column.dtype === "json") {
    expectTypeOf(column.values).toEqualTypeOf<readonly unknown[]>();
} else if (column.dtype === "bool") {
    expectTypeOf(column.data).toEqualTypeOf<U32>();
    expectTypeOf(column.mutableData()).toEqualTypeOf<U32>();
} else if (column.dtype === "dict") {
    expectTypeOf(column.codes).toEqualTypeOf<U32>();
    expectTypeOf(column.dictionary).toEqualTypeOf<readonly string[]>();
    expectTypeOf(column.codeOf("x")).toBeNumber();
} else if (column.dtype === "i32") {
    expectTypeOf(column.data).toEqualTypeOf<I32>();
}
// the string / list / json shapes have no mutableData(): their contents are never written in place
expectTypeOf<StringColumn>().not.toHaveProperty("mutableData");
expectTypeOf<ListColumn>().not.toHaveProperty("mutableData");
expectTypeOf<JsonColumn>().not.toHaveProperty("mutableData");

// ---- typed() / requireTyped() / clone() / slice() / materializeDefault() return types
declare const table: AttributeTable;
expectTypeOf(table.typed("x", "u32")).toEqualTypeOf<U32Column | null>();
expectTypeOf(table.typed("x", "list")).toEqualTypeOf<ListColumn | null>();
expectTypeOf(table.requireTyped("x", "bool")).toEqualTypeOf<BoolColumn>();
expectTypeOf(table.get("x")).toEqualTypeOf<Column | null>();
expectTypeOf(table.require("x")).toEqualTypeOf<Column>();
expectTypeOf(table.byRole("weight")).toEqualTypeOf<Column | null>();
expectTypeOf(table.value("x", 0)).toBeUnknown();
expectTypeOf(table.isSet("x", 0)).toBeBoolean();
expectTypeOf(table.gpuView("x")).toEqualTypeOf<U32 | I32 | F32>();
expectTypeOf(table.set("x", new Uint32Array(3))).toEqualTypeOf<Column>();
expectTypeOf(table.set("x", ["a", "b"], { dtype: "string" }, { replaceRole: true })).toEqualTypeOf<Column>();
expectTypeOf(table.remove("x")).toBeBoolean();
expectTypeOf(table.rename("x", "y")).toBeVoid();
expectTypeOf(table.clone()).toEqualTypeOf<AttributeTable>();
expectTypeOf(table.names()).toEqualTypeOf<readonly string[]>();
expectTypeOf([...table]).toEqualTypeOf<Column[]>();

declare const f64: F64Column;
expectTypeOf(f64.clone()).toEqualTypeOf<F64Column>();
expectTypeOf(f64.slice(0, 1)).toEqualTypeOf<F64Column>();
expectTypeOf(f64.materializeDefault()).toEqualTypeOf<F64Column>();
expectTypeOf(f64.mutableData()).toEqualTypeOf<Float64Array<ArrayBuffer>>();
expectTypeOf(f64.mutableValidity()).toEqualTypeOf<U32 | null>();
expectTypeOf(f64.validity).toEqualTypeOf<U32 | null>();
expectTypeOf(f64.paddedU32View()).toEqualTypeOf<U32>();
expectTypeOf(f64.gpu).toEqualTypeOf<"direct" | "packed" | "convert" | "none">();

// ---- DtypeValue and value()
expectTypeOf<DtypeValue<"f32">>().toEqualTypeOf<number | ArrayLike<number>>();
expectTypeOf<DtypeValue<"u8">>().toEqualTypeOf<number | ArrayLike<number>>();
expectTypeOf<DtypeValue<"bool">>().toEqualTypeOf<boolean>();
expectTypeOf<DtypeValue<"dict">>().toEqualTypeOf<string>();
expectTypeOf<DtypeValue<"string">>().toEqualTypeOf<string>();
expectTypeOf<DtypeValue<"list">>().toEqualTypeOf<readonly unknown[]>();
expectTypeOf<DtypeValue<"json">>().toBeUnknown();
expectTypeOf(f64.value(0)).toEqualTypeOf<number | ArrayLike<number> | undefined>();
declare const bool: BoolColumn;
expectTypeOf(bool.value(0)).toEqualTypeOf<boolean | undefined>();

// ---- metadata: ColumnMeta has every field present (null for none); ColumnDecl is the loose input
expectTypeOf<Required<ColumnMeta>>().toEqualTypeOf<ColumnMeta>();
expectTypeOf<ColumnMeta["role"]>().toEqualTypeOf<ColumnRole | null>();
expectTypeOf<ColumnMeta["refersTo"]>().toEqualTypeOf<"node" | "edge" | null>();
expectTypeOf<ColumnMeta["origin"]>().toEqualTypeOf<{
    readonly format: string | null;
    readonly id: string | null;
    readonly title: string | null;
    readonly type: string | null;
    readonly namespace: string | null;
} | null>();
const decl: ColumnDecl = { name: "position", dtype: "f32", components: 3, mutable: true, role: "position" };
const declWithUndefined: ColumnDecl = { name: "p", dtype: "u32", refersTo: undefined, origin: { format: undefined } };
const patch: ColumnDeclPatch = { name: undefined, dtype: undefined, role: "custom.role" };
use(decl, declWithUndefined, patch);
expectTypeOf<ColumnDeclPatch>().toEqualTypeOf<Loose<ColumnDecl>>();
expectTypeOf<ColumnDeclPatch["name"]>().toEqualTypeOf<string | undefined>();

// ---- roles: the known literals plus any string, and the literals survive for autocomplete
expectTypeOf<"position">().toMatchTypeOf<ColumnRole>();
expectTypeOf<"graphty.custom">().toMatchTypeOf<ColumnRole>();
expectTypeOf<KnownColumnRole>().toMatchTypeOf<ColumnRole>();
expectTypeOf<number>().not.toMatchTypeOf<ColumnRole>();

// ---- masks share the packed u32 layout
expectTypeOf<NodeMask>().toEqualTypeOf<U32>();
expectTypeOf<EdgeMask>().toEqualTypeOf<U32>();
