import {
    type BuilderOptionsPatch,
    type ColumnHandle,
    type CsrInput,
    type EdgeArraysInput,
    type ExtensionHandle,
    type FlagClaims,
    type FreezeOptions,
    type FreezeReport,
    type GraphBuilder,
    type GraphBuilderOptions,
    type GraphMetaPatch,
    type GraphSink,
    type GraphSnapshot,
    INVALID_INDEX,
    type Loose,
    type RecordsInput,
    type ResolvedBuilderOptions,
    type U32,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

declare function use(...values: unknown[]): void;

// ---- GraphBuilder is assignable to GraphSink (design section 12.3); a sink is not a builder
expectTypeOf<GraphBuilder>().toMatchTypeOf<GraphSink>();
expectTypeOf<GraphSink>().not.toMatchTypeOf<GraphBuilder>();

// ---- branded handles reject a plain number and each other (design section 12.1)
expectTypeOf<number>().not.toMatchTypeOf<ColumnHandle>();
expectTypeOf<number>().not.toMatchTypeOf<ExtensionHandle>();
expectTypeOf<ColumnHandle>().not.toMatchTypeOf<ExtensionHandle>();
expectTypeOf<ExtensionHandle>().not.toMatchTypeOf<ColumnHandle>();
// a handle is still usable where a number is expected (INVALID_INDEX comparison)
expectTypeOf<ColumnHandle>().toMatchTypeOf<number>();
declare const builder: GraphBuilder;
const handle = builder.nodeColumn("position");
expectTypeOf(handle).toEqualTypeOf<ColumnHandle>();
use(handle === INVALID_INDEX);
// the swapped-argument mistake the brand exists to catch
expectTypeOf(builder.setNodeValue).parameter(0).toEqualTypeOf<ColumnHandle | string>();
expectTypeOf(builder.setNodeValue).parameter(1).toBeNumber();
expectTypeOf<Parameters<GraphBuilder["setNodeValue"]>>().not.toMatchTypeOf<[number, ColumnHandle, unknown]>();

// ---- options: directed is required; every optional input also accepts an explicit undefined
expectTypeOf<GraphBuilderOptions["directed"]>().toBeBoolean();
expectTypeOf<Required<GraphBuilderOptions>["directed"]>().toBeBoolean();
const options: GraphBuilderOptions = {
    directed: true,
    weighted: undefined,
    duplicateEdges: "sum",
    selfLoops: undefined,
};
const patch: BuilderOptionsPatch = { directed: undefined, weightDtype: "f64" };
use(options, patch);
expectTypeOf<BuilderOptionsPatch>().toEqualTypeOf<Loose<GraphBuilderOptions>>();
expectTypeOf<BuilderOptionsPatch["directed"]>().toEqualTypeOf<boolean | undefined>();
expectTypeOf<Record<string, never>>().toMatchTypeOf<BuilderOptionsPatch>();
expectTypeOf<Record<string, never>>().not.toMatchTypeOf<GraphBuilderOptions>();

// ---- resolved options: the same shape under every compiler flag, nulls instead of optionals
expectTypeOf<Required<ResolvedBuilderOptions>>().toEqualTypeOf<ResolvedBuilderOptions>();
expectTypeOf<ResolvedBuilderOptions["expectedNodes"]>().toEqualTypeOf<number | null>();
expectTypeOf<ResolvedBuilderOptions["weighted"]>().toEqualTypeOf<boolean | "auto">();
expectTypeOf(builder.options).toEqualTypeOf<ResolvedBuilderOptions>();

// ---- Loose patches accept explicit undefined (compiled under exactOptionalPropertyTypes by the
// strict-consumer config)
const meta: GraphMetaPatch = { name: undefined, keywords: undefined, extra: { a: 1 }, weightOrigin: null };
const claims: FlagClaims = { multigraph: undefined, weighted: true };
const freeze: FreezeOptions = { prepare: undefined, arena: false, checksum: undefined, duplicateEdges: "keep" };
use(meta, claims, freeze);
expectTypeOf<GraphMetaPatch["name"]>().toEqualTypeOf<string | null | undefined>();
expectTypeOf<FlagClaims["multigraph"]>().toEqualTypeOf<boolean | undefined>();

// ---- freeze report: remaps are null, never undefined, when nothing was renumbered (I16)
expectTypeOf<Required<FreezeReport>>().toEqualTypeOf<FreezeReport>();
expectTypeOf<FreezeReport["nodeRemap"]>().toEqualTypeOf<U32 | null>();
expectTypeOf<FreezeReport["edgeRemap"]>().toEqualTypeOf<U32 | null>();
expectTypeOf<FreezeReport["widened"][number]["to"]>().toEqualTypeOf<
    "f32" | "f64" | "i32" | "u32" | "u8" | "bool" | "dict" | "string" | "list" | "json"
>();
expectTypeOf(builder.freeze()).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(builder.freezeWithReport({ label: "x" })).toEqualTypeOf<{
    snapshot: GraphSnapshot;
    report: FreezeReport;
}>();

// ---- builder members
expectTypeOf(builder.directed).toBeBoolean();
expectTypeOf(builder.directedLocked).toBeBoolean();
expectTypeOf(builder.nodeCount).toBeNumber();
expectTypeOf(builder.edgeCount).toBeNumber();
expectTypeOf(builder.nodeBound).toBeNumber();
expectTypeOf(builder.edgeBound).toBeNumber();
expectTypeOf(builder.mutationCount).toBeNumber();
expectTypeOf(builder.dirty).toBeBoolean();
expectTypeOf(builder.addNode("a")).toBeNumber();
expectTypeOf(builder.addNodes(["a", 1], new Uint32Array(2))).toEqualTypeOf<U32>();
expectTypeOf(builder.addAnonymousNodes(10)).toBeNumber();
expectTypeOf(builder.addEdge("a", "b")).toBeNumber();
expectTypeOf(builder.addEdge("a", "b", 2.5)).toBeNumber();
expectTypeOf(builder.addEdgeByIndex(0, 1)).toBeNumber();
expectTypeOf(builder.addEdges(new Uint32Array(2), new Uint32Array(2), new Float64Array(2))).toBeNumber();
expectTypeOf(builder.addEdgesByIds(["a"], [1], [1.5])).toBeNumber();
expectTypeOf(builder.removeNode("a")).toEqualTypeOf<U32>();
expectTypeOf(builder.removeEdge(0)).toBeBoolean();
expectTypeOf(builder.edgeEndpoints(0)).toEqualTypeOf<readonly [source: number, target: number]>();
expectTypeOf(builder.outEdgesOf(0)).toEqualTypeOf<U32>();
expectTypeOf(builder.findEdges(0, 1)).toEqualTypeOf<U32>();
expectTypeOf(builder.declareNodeColumn({ name: "n", dtype: "f64" })).toEqualTypeOf<ColumnHandle>();
expectTypeOf(builder.addExtensionTable("temporal:node:price", [])).toEqualTypeOf<ExtensionHandle>();
expectTypeOf(builder.addNodeRecord("a", { label: "A" })).toBeNumber();
expectTypeOf(builder.addEdgeRecord("a", "b", { w: 1 }, null)).toBeNumber();
expectTypeOf(builder.setDirected(true, { expand: true })).toBeVoid();
expectTypeOf(builder.byteLength()).toBeNumber();

// ---- factory inputs (design section 8.1)
const edgeArrays: EdgeArraysInput = {
    directed: true,
    nodeCount: 3,
    src: new Uint32Array([0, 1]),
    dst: new Uint32Array([1, 2]),
    weights: new Float64Array([1, 2]),
    ids: undefined,
    nodeColumns: { score: new Float32Array(3) },
    edgeColumns: { kind: { data: ["a", "b"], decl: { dtype: "dict" } } },
};
const csr: CsrInput = {
    directed: false,
    nodeCount: 2,
    rowPtr: new Uint32Array([0, 1, 2]),
    colIdx: new Uint32Array([1, 0]),
    weights: null,
    arcToEdge: new Uint32Array([0, 0]),
    flags: { multigraph: false },
};
const records: RecordsInput = {
    directed: true,
    edges: [{ source: "a", target: "b" }],
    nodeId: null,
    edgeWeight: null,
    columns: [{ name: "label", dtype: "string" }],
};
use(edgeArrays, csr, records);
expectTypeOf<EdgeArraysInput["ids"]>().toEqualTypeOf<
    readonly (string | number)[] | Float64Array<ArrayBuffer> | undefined
>();
expectTypeOf<CsrInput["weights"]>().toEqualTypeOf<Float32Array<ArrayBuffer> | null | undefined>();
