/// <reference types="@webgpu/types" />

import {
    type ArenaLayout,
    type F32,
    type F64,
    type GraphSnapshot,
    type I32,
    type NumericVector,
    type TypedArrayData,
    type U8,
    type U32,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

// Design section 12.1: public arrays are the concrete typed-array classes with the buffer parameter
// fixed to ArrayBuffer. That parameter, not Readonly, is what makes them a BufferSource.

// ---- the aliases are exactly the ArrayBuffer-parameterised classes
expectTypeOf<U32>().toEqualTypeOf<Uint32Array<ArrayBuffer>>();
expectTypeOf<I32>().toEqualTypeOf<Int32Array<ArrayBuffer>>();
expectTypeOf<F32>().toEqualTypeOf<Float32Array<ArrayBuffer>>();
expectTypeOf<F64>().toEqualTypeOf<Float64Array<ArrayBuffer>>();
expectTypeOf<U8>().toEqualTypeOf<Uint8Array<ArrayBuffer>>();

// ---- a fresh array needs no annotation (TS 5.7+ produces the ArrayBuffer parameter)
expectTypeOf(new Uint32Array(4)).toEqualTypeOf<U32>();
expectTypeOf(new Int32Array(4)).toEqualTypeOf<I32>();
expectTypeOf(new Float32Array(4)).toEqualTypeOf<F32>();
expectTypeOf(new Float64Array(4)).toEqualTypeOf<F64>();
expectTypeOf(new Uint8Array(4)).toEqualTypeOf<U8>();

// ---- the bare class names (default parameter ArrayBufferLike) and shared-memory arrays are rejected
expectTypeOf<Uint32Array>().not.toMatchTypeOf<U32>();
expectTypeOf<Float32Array>().not.toMatchTypeOf<F32>();
expectTypeOf<Uint32Array<SharedArrayBuffer>>().not.toMatchTypeOf<U32>();

// ---- but every alias is usable where the bare class is expected
expectTypeOf<U32>().toMatchTypeOf<Uint32Array>();
expectTypeOf<F64>().toMatchTypeOf<Float64Array>();

// ---- windows, copies and fills keep the parameter (design section 10.6)
declare const u32: U32;
declare const f32: F32;
expectTypeOf(u32.subarray(1, 3)).toEqualTypeOf<U32>();
expectTypeOf(u32.slice(1, 3)).toEqualTypeOf<U32>();
expectTypeOf(u32.fill(0)).toEqualTypeOf<U32>();
expectTypeOf(f32.subarray(0)).toEqualTypeOf<F32>();
expectTypeOf(u32.buffer).toEqualTypeOf<ArrayBuffer>();

// ---- unions
expectTypeOf<U32>().toMatchTypeOf<TypedArrayData>();
expectTypeOf<U8>().toMatchTypeOf<TypedArrayData>();
expectTypeOf<Uint16Array<ArrayBuffer>>().not.toMatchTypeOf<TypedArrayData>();
expectTypeOf<NumericVector>().toEqualTypeOf<F32 | F64 | U32 | I32>();
expectTypeOf<U8>().not.toMatchTypeOf<NumericVector>();

// ---- the BufferSource proof of design section 16.6 against the real GPUQueue signature of
// @webgpu/types (referenced below): every public array is a view over a plain ArrayBuffer, so it
// is LITERALLY a GPUAllowSharedBufferSource for queue.writeBuffer(buf, 0, array) with no cast.
declare const queue: GPUQueue;
declare const buf: GPUBuffer;
declare const snapshot: GraphSnapshot;
declare const arena: ArenaLayout;
declare const start: number;
declare const end: number;

queue.writeBuffer(buf, 0, snapshot.colIdx);
queue.writeBuffer(buf, 0, snapshot.rowPtr);
queue.writeBuffer(buf, 0, snapshot.nodes.gpuView("position"));
queue.writeBuffer(buf, 0, snapshot.colIdx.subarray(start, end));
queue.writeBuffer(buf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength));
queue.writeBuffer(buf, 0, snapshot.reverse().rowPtr);
queue.writeBuffer(buf, 0, snapshot.edgeList().src);
queue.writeBuffer(buf, 0, snapshot.degreeOrder().perm);
queue.writeBuffer(buf, 0, snapshot.toBytes());
if (snapshot.weights !== null) {
    queue.writeBuffer(buf, 0, snapshot.weights);
}

// the strict BufferSource of the DOM lib (ArrayBufferView<ArrayBuffer> | ArrayBuffer, no
// SharedArrayBuffer) is what a plain ArrayBuffer view satisfies and a SharedArrayBuffer view does not
type StrictBufferSource = ArrayBufferView<ArrayBuffer> | ArrayBuffer;
expectTypeOf<U32>().toMatchTypeOf<StrictBufferSource>();
expectTypeOf<Uint32Array>().not.toMatchTypeOf<StrictBufferSource>();
expectTypeOf<Uint32Array<SharedArrayBuffer>>().not.toMatchTypeOf<StrictBufferSource>();

// weights may be null, so a kernel narrows first
expectTypeOf(snapshot.weights).toEqualTypeOf<F32 | null>();
