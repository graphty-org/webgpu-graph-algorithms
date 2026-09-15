/**
 * The strict-consumer sample (design 16.6; contract 5.5): compiled by `tsc -p tsconfig.strict-consumer.json`
 * against dist/*.d.ts with noUncheckedIndexedAccess and exactOptionalPropertyTypes ON, and by the ordinary
 * `tsc --noEmit -p tsconfig.json` against src/ through `paths`. Never executed. It imports the three entries by
 * their package names. P1-T7 and P3-T3 extend it.
 */

import * as pkg from "@graphty/webgpu-graph-algorithms";
import * as browserEntry from "@graphty/webgpu-graph-algorithms/browser";
import * as nodeEntry from "@graphty/webgpu-graph-algorithms/node";
import { expectTypeOf } from "vitest";

// ---- the barrel and the two entries are ESM with named exports only: no default export
expectTypeOf(pkg).not.toHaveProperty("default");
expectTypeOf(browserEntry).not.toHaveProperty("default");
expectTypeOf(nodeEntry).not.toHaveProperty("default");
expectTypeOf<"createNodeGpu" | "dawnFlags">().toMatchTypeOf<keyof typeof nodeEntry>();

// ---- the closed code union of contract 3.1, in document order
expectTypeOf<pkg.WebGpuGraphErrorCode>().toEqualTypeOf<
    | "E_NO_WEBGPU"
    | "E_NO_ADAPTER"
    | "E_NO_DEVICE"
    | "E_SOFTWARE_ONLY"
    | "E_DEVICE_LOST"
    | "E_DISPOSED"
    | "E_VALIDATION"
    | "E_SHADER_COMPILE"
    | "E_OUT_OF_MEMORY"
    | "E_TOO_LARGE"
    | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT"
    | "E_SNAPSHOT"
    | "E_RELEASED"
    | "E_NOT_LOADED"
    | "E_ABORTED"
>();
expectTypeOf<"E_SOMETHING_ELSE">().not.toMatchTypeOf<pkg.WebGpuGraphErrorCode>();
expectTypeOf<string>().not.toMatchTypeOf<pkg.WebGpuGraphErrorCode>();

// ---- constructing an error and reading its fields
const error = new pkg.WebGpuGraphError("E_NO_ADAPTER", "requestAdapter() returned null", { reason: "null" });
expectTypeOf(error).toMatchTypeOf<Error>();
expectTypeOf(error.code).toEqualTypeOf<pkg.WebGpuGraphErrorCode>();
expectTypeOf(error.name).toEqualTypeOf<"WebGpuGraphError">();
expectTypeOf(error.details).toEqualTypeOf<Readonly<Record<string, unknown>>>();
expectTypeOf(pkg.WebGpuGraphError).toBeConstructibleWith("E_DISPOSED", "disposed");
expectTypeOf(pkg.WebGpuGraphError).toBeConstructibleWith("E_DISPOSED", "disposed", undefined);
expectTypeOf(pkg.WebGpuGraphError).toBeConstructibleWith("E_TOO_LARGE", "big", { needed: 2, limit: 1 });

// ---- narrowing a caught value to a code
function codeOf(x: unknown): pkg.WebGpuGraphErrorCode | null {
    if (pkg.isWebGpuGraphError(x)) {
        return x.code;
    }
    return null;
}
expectTypeOf(codeOf).returns.toEqualTypeOf<pkg.WebGpuGraphErrorCode | null>();
expectTypeOf(pkg.hasErrorCode).parameter(0).toBeUnknown();
expectTypeOf(pkg.hasErrorCode).parameter(1).toEqualTypeOf<pkg.WebGpuGraphErrorCode>();
expectTypeOf(pkg.hasErrorCode).returns.toBeBoolean();

// ---- the constants: literal where declared as a literal, number where computed; the tuple keeps its members
// defined under noUncheckedIndexedAccess. The literal pins use the explicit type-argument form: a `const X = 256`
// has a WIDENING literal type, which `expectTypeOf(value)` widens to number during inference.
expectTypeOf<typeof pkg.WORKGROUP_SIZE>().toEqualTypeOf<256>();
expectTypeOf<typeof pkg.MAX_WORKGROUPS_PER_DIM>().toEqualTypeOf<65535>();
expectTypeOf(pkg.MAX_1D_ITEMS).toBeNumber();
expectTypeOf<typeof pkg.ARC_WINDOW_ALIGN>().toEqualTypeOf<64>();
expectTypeOf<typeof pkg.STORAGE_ALIGN>().toEqualTypeOf<256>();
expectTypeOf(pkg.EXACT_MAX_NODES).toBeNumber();
expectTypeOf(pkg.PASSTHROUGH_FORMAT_CODES).toEqualTypeOf<
    readonly ["E_GPU_INELIGIBLE", "E_UNKNOWN_NODE", "E_UNKNOWN_COLUMN", "E_COLUMN_LENGTH"]
>();
expectTypeOf(pkg.PASSTHROUGH_FORMAT_CODES[0]).toEqualTypeOf<"E_GPU_INELIGIBLE">();
const lastPassthrough: string = pkg.PASSTHROUGH_FORMAT_CODES[3];
expectTypeOf(lastPassthrough).toBeString();

// ---- the adapter info the package reads is structural (no __brand); optional fields may be omitted under
// exactOptionalPropertyTypes, and a real GPUAdapterInfo is assignable to it
const info: pkg.AdapterInfoLike = { vendor: "nvidia", architecture: "lovelace", device: "", description: "" };
const withOptional: pkg.AdapterInfoLike = {
    ...info,
    isFallbackAdapter: false,
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
};
expectTypeOf(pkg.isSoftwareAdapter(info)).toBeBoolean();
expectTypeOf(pkg.isSoftwareAdapter(withOptional)).toBeBoolean();
declare const adapterInfo: GPUAdapterInfo;
expectTypeOf(adapterInfo).toMatchTypeOf<pkg.AdapterInfoLike>();

// ---- the ./node entry under exactOptionalPropertyTypes: every option is `?: T | undefined`
const options: nodeEntry.NodeGpuOptions = {
    adapter: undefined,
    backend: undefined,
    dawnFeatures: undefined,
    software: undefined,
    installGlobals: undefined,
};
expectTypeOf(nodeEntry.dawnFlags(options)).toEqualTypeOf<string[]>();
expectTypeOf(nodeEntry.dawnFlags(undefined)).toEqualTypeOf<string[]>();
expectTypeOf(nodeEntry.dawnFlags({ backend: "null", dawnFeatures: ["allow_unsafe_apis"] })).toEqualTypeOf<string[]>();
expectTypeOf(nodeEntry.createNodeGpu).parameter(0).toEqualTypeOf<nodeEntry.NodeGpuOptions | undefined>();
expectTypeOf(nodeEntry.createNodeGpu).returns.resolves.toEqualTypeOf<nodeEntry.NodeGpuHandle>();
declare const handle: nodeEntry.NodeGpuHandle;
expectTypeOf(handle.gpu).toEqualTypeOf<GPU>();
expectTypeOf(handle.dispose).returns.toBeVoid();
