import { type Column, type GraphSnapshot, type U32, type ViewName } from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

import {
    type MutableColumnParts,
    type SnapshotParts,
    type TableParts,
    type ViewCache,
    type ViewValues,
} from "../../src/types/internal.js";

// The construction contracts of src/types/internal.ts are not public (they are absent from the
// package barrel), but the implementation modules depend on their shape, so pin it here.

// ---- SnapshotParts carries every public core field of GraphSnapshot with the two lazily
// materialised permutations nullable, plus the construction-only fields
type CoreKeys =
    | "label"
    | "directed"
    | "nodeCount"
    | "edgeCount"
    | "arcCount"
    | "selfLoopCount"
    | "rowPtr"
    | "colIdx"
    | "weights"
    | "flags"
    | "ids"
    | "nodes"
    | "edges"
    | "graph"
    | "extensions"
    | "meta"
    | "arena";
expectTypeOf<Pick<SnapshotParts, CoreKeys>>().toEqualTypeOf<Pick<GraphSnapshot, CoreKeys>>();
expectTypeOf<SnapshotParts["arcToEdge"]>().toEqualTypeOf<U32 | null>();
expectTypeOf<SnapshotParts["edgeToArc"]>().toEqualTypeOf<U32 | null>();
expectTypeOf<SnapshotParts["serial"]>().toEqualTypeOf<number | null>();
expectTypeOf<SnapshotParts["checksum"]>().toBeBoolean();
expectTypeOf<Required<SnapshotParts>>().toEqualTypeOf<SnapshotParts>();
expectTypeOf<SnapshotParts>().not.toHaveProperty("detached");

// ---- ViewValues is keyed exactly by ViewName, and ViewCache is its writable nullable form
expectTypeOf<keyof ViewValues>().toEqualTypeOf<ViewName>();
expectTypeOf<keyof ViewCache>().toEqualTypeOf<ViewName>();
expectTypeOf<ViewCache["outDegree"]>().toEqualTypeOf<U32 | null>();
expectTypeOf<ViewCache["totalWeight"]>().toEqualTypeOf<number | null>();
expectTypeOf<ViewCache["symmetric"]>().toEqualTypeOf<boolean | null>();
expectTypeOf<ViewCache["reverse"]>().toEqualTypeOf<ReturnType<GraphSnapshot["reverse"]> | null>();
expectTypeOf<ViewCache["degreeOrder"]>().toEqualTypeOf<ReturnType<GraphSnapshot["degreeOrder"]> | null>();
expectTypeOf<Readonly<ViewCache>>().not.toEqualTypeOf<ViewCache>();
declare const cache: ViewCache;
cache.outDegree = null;
cache.outDegree = new Uint32Array(1);

// ---- MutableColumnParts: every slot writable and nullable except meta, length and nullCount
expectTypeOf<Readonly<MutableColumnParts>>().not.toEqualTypeOf<MutableColumnParts>();
expectTypeOf<MutableColumnParts["data"]>().toEqualTypeOf<
    | Uint32Array<ArrayBuffer>
    | Int32Array<ArrayBuffer>
    | Float32Array<ArrayBuffer>
    | Float64Array<ArrayBuffer>
    | Uint8Array<ArrayBuffer>
    | null
>();
expectTypeOf<MutableColumnParts["child"]>().toEqualTypeOf<Column | null>();
expectTypeOf<MutableColumnParts["dictionary"]>().toEqualTypeOf<string[] | null>();
expectTypeOf<MutableColumnParts["nullCount"]>().toBeNumber();

// ---- TableParts
expectTypeOf<TableParts["columns"]>().toEqualTypeOf<readonly Column[]>();
expectTypeOf<TableParts["domain"]>().toEqualTypeOf<"node" | "edge" | "graph" | "extension">();
