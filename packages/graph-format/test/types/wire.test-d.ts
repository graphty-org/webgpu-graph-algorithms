import {
    type ColumnMeta,
    type FromWireOptions,
    type GraphMeta,
    type SnapshotFlags,
    type ToBytesOptions,
    type ToWireOptions,
    type ViewName,
    type WireArena,
    type WireBufferRef,
    type WireColumn,
    type WireDtype,
    type WireIdMap,
    type WireManifest,
    type WireSnapshot,
    type WireUtf8,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

declare function use(...values: unknown[]): void;

// ---- the wire form is a plain object: a manifest plus ArrayBuffers (design section 9.1)
declare const wire: WireSnapshot;
expectTypeOf(wire.manifest).toEqualTypeOf<WireManifest>();
expectTypeOf(wire.buffers).toEqualTypeOf<readonly ArrayBuffer[]>();

// ---- absent members are null, never optional (design section 12.1)
expectTypeOf<Required<WireColumn>>().toEqualTypeOf<WireColumn>();
expectTypeOf<Required<WireManifest>>().toEqualTypeOf<WireManifest>();
expectTypeOf<Required<WireIdMap>>().toEqualTypeOf<WireIdMap>();
expectTypeOf<Required<WireBufferRef>>().toEqualTypeOf<WireBufferRef>();
expectTypeOf<Required<WireArena>>().toEqualTypeOf<WireArena>();
expectTypeOf<WireColumn["dictionary"]>().toEqualTypeOf<WireUtf8 | null>();
expectTypeOf<WireColumn["child"]>().toEqualTypeOf<WireColumn | null>();
expectTypeOf<WireColumn["data"]>().toEqualTypeOf<WireBufferRef | null>();
expectTypeOf<WireColumn["meta"]>().toEqualTypeOf<ColumnMeta>();
expectTypeOf<WireManifest["arena"]>().toEqualTypeOf<WireArena | null>();
expectTypeOf<WireManifest["views"]>().toEqualTypeOf<Readonly<
    Record<string, Readonly<Record<string, WireBufferRef>>>
> | null>();
expectTypeOf<WireManifest["label"]>().toEqualTypeOf<string | null>();

// ---- manifest discriminators and shapes
expectTypeOf<WireManifest["format"]>().toEqualTypeOf<"graphty-snapshot">();
expectTypeOf<WireManifest["formatVersion"]>().toEqualTypeOf<1>();
expectTypeOf<WireManifest["wire"]>().toEqualTypeOf<readonly [major: number, minor: number]>();
expectTypeOf<WireManifest["flags"]>().toEqualTypeOf<SnapshotFlags>();
expectTypeOf<WireManifest["meta"]>().toEqualTypeOf<GraphMeta>();
expectTypeOf<WireManifest["core"]["rowPtr"]>().toEqualTypeOf<WireBufferRef>();
expectTypeOf<WireManifest["core"]["arcToEdge"]>().toEqualTypeOf<WireBufferRef | null>();
expectTypeOf<WireManifest["counts"]>().toEqualTypeOf<{
    readonly nodes: number;
    readonly edges: number;
    readonly arcs: number;
    readonly selfLoops: number;
}>();
expectTypeOf<WireManifest["copied"]>().toEqualTypeOf<readonly number[]>();
expectTypeOf<WireManifest["extensions"][number]["columns"]>().toEqualTypeOf<readonly WireColumn[]>();
expectTypeOf<WireIdMap["kind"]>().toEqualTypeOf<"identity" | "dense" | "numeric" | "string" | "mixed">();
expectTypeOf<WireDtype>().toEqualTypeOf<"u32" | "i32" | "f32" | "f64" | "u8" | "utf8">();
expectTypeOf<WireBufferRef["dtype"]>().toEqualTypeOf<WireDtype>();

// ---- options accept explicit undefined
const toWire: ToWireOptions = { transfer: true, includeViews: undefined, includeColumns: false };
const toBytes: ToBytesOptions = { includeViews: ["reverse", "coo"] };
const fromWire: FromWireOptions = { validate: "structure", copy: undefined, unknownColumns: "skip" };
use(toWire, toBytes, fromWire);
expectTypeOf<ToWireOptions["includeViews"]>().toEqualTypeOf<readonly ViewName[] | undefined>();
expectTypeOf<FromWireOptions["validate"]>().toEqualTypeOf<"none" | "structure" | "full" | undefined>();
expectTypeOf<FromWireOptions["unknownColumns"]>().toEqualTypeOf<"error" | "skip" | undefined>();
