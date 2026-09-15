/**
 * Context and capability types (spec 2.2, 3.3; contract 3.3). Types only: the file imports the error class
 * as a type, so every layer may `import type` from it without a value edge (the src/types zone of the
 * package eslint config). P0-T3 wrote AdapterInfoLike and AdapterSummary; P1-T1 completes the file.
 */

import type { WebGpuGraphError } from "../errors.js";

/** The limits `"raise"` takes from the adapter (spec 2.2); maxComputeWorkgroupsPerDimension is deliberately absent. */
export type RaisableLimit =
    | "maxBufferSize"
    | "maxStorageBufferBindingSize"
    | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupStorageSize"
    | "maxComputeInvocationsPerWorkgroup"
    | "maxComputeWorkgroupSizeX";

/** How `create()` builds `requiredLimits` (spec 2.2 step 3). */
export type LimitPolicy = "default" | "raise" | Readonly<Partial<Record<RaisableLimit, number>>>;

/** Options of GpuContext.create (spec 2.2). */
export interface GpuContextOptions {
    /** browser: navigator.gpu; Node: the handle of createNodeGpu(). */
    readonly gpu?: GPU | undefined;
    /** Skips requestAdapter(); the adapter must be UNUSED (spec 2.2 step 1). */
    readonly adapter?: GPUAdapter | undefined;
    /** Adopts a device the caller owns (ownsDevice false). */
    readonly device?: GPUDevice | undefined;
    /** Default "high-performance". */
    readonly powerPreference?: GPUPowerPreference | undefined;
    /** Default false; true -> E_SOFTWARE_ONLY instead of a device on lavapipe / SwiftShader. */
    readonly rejectSoftware?: boolean | undefined;
    /** Default "raise". */
    readonly limits?: LimitPolicy | undefined;
    /** Default ["subgroups", "timestamp-query"]; requested only when the adapter has them. */
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;
    /** Default []; a missing one -> E_NO_DEVICE. */
    readonly requiredFeatures?: readonly GPUFeatureName[] | undefined;
    /** The device label and the context label (default "webgpu-graph-algorithms"). */
    readonly label?: string | undefined;
    /** The uncapturederror sink; default: keep the last error and throw it from the next public call (spec 5.7). */
    readonly onError?: ((error: WebGpuGraphError) => void) | undefined;
    /** Default 2: console.warn once when more snapshots than this are resident (spec 4.1). */
    readonly warnUnreleasedSnapshots?: number | undefined;
    /** CONTRACT DECISION: set by the ./browser and ./node entries ("browser" / "node"), never by sniffing globals (spec 2.2 step 4); consumers leave it unset ("unknown"). */
    readonly runtime?: "browser" | "node" | "unknown" | undefined;
}

/** The structural subset of GPUAdapterInfo the package reads (CONTRACT DECISION: structural so tests can fake it without the __brand of @webgpu/types). */
export interface AdapterInfoLike {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly isFallbackAdapter?: boolean | undefined;
    readonly subgroupMinSize?: number | undefined;
    readonly subgroupMaxSize?: number | undefined;
}

/** What the app displays or logs about an adapter (spec 2.2). */
export interface AdapterSummary {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly software: boolean;
    readonly subgroupMinSize: number;
    readonly subgroupMaxSize: number;
    readonly features: readonly string[];
    readonly limits: Readonly<Record<string, number>>;
}

/** Result of GpuContext.probe (spec 2.2); `adapter` is the UNUSED adapter to pass to create({ adapter }). */
export interface ProbeResult {
    readonly ok: boolean;
    readonly code: "OK" | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_SOFTWARE_ONLY";
    readonly reason: string | null;
    readonly adapter: GPUAdapter | null;
    readonly summary: AdapterSummary | null;
}

/** Options of GpuContext.probe (spec 2.2). `gpu` may be undefined so an app can pass `navigator.gpu` without a guard: undefined -> E_NO_WEBGPU. */
export interface ProbeOptions {
    readonly gpu: GPU | undefined;
    readonly powerPreference?: GPUPowerPreference | undefined;
    readonly rejectSoftware?: boolean | undefined;
}

/** The limits every planner reads; GPUSupportedLimits is structurally assignable to it (CONTRACT DECISION: planners take PlanCaps so faked tables need no cast). */
export interface PlanLimits {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxStorageBuffersPerShaderStage: number;
    readonly minStorageBufferOffsetAlignment: number;
    readonly minUniformBufferOffsetAlignment: number;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly maxComputeInvocationsPerWorkgroup: number;
    readonly maxComputeWorkgroupSizeX: number;
    readonly maxComputeWorkgroupStorageSize: number;
    readonly maxUniformBufferBindingSize: number;
}

/** The capability record every planner and kernel reads (spec 3.3). */
export interface GpuCaps {
    /** device.limits (never adapter.limits, spec 2.2 step 4). */
    readonly limits: GPUSupportedLimits;
    /** device.features. */
    readonly features: ReadonlySet<string>;
    /** navigator.gpu.wgslLanguageFeatures / the Dawn equivalent; informational only (never relied on). */
    readonly wgslFeatures: ReadonlySet<string>;
    /** 0 when the "subgroups" feature is absent from the device. */
    readonly subgroupMinSize: number;
    /** 0 when the "subgroups" feature is absent from the device. */
    readonly subgroupMaxSize: number;
    readonly software: boolean;
    readonly runtime: "browser" | "node" | "unknown";
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
}

/** What the pure planners need of GpuCaps: `limits` narrowed to PlanLimits plus the four scalar facts. A GpuCaps IS a PlanCaps. */
export interface PlanCaps {
    readonly limits: PlanLimits;
    readonly features: ReadonlySet<string>;
    readonly subgroupMinSize: number;
    readonly subgroupMaxSize: number;
    readonly software: boolean;
}

/** Mutable test-build flags on a context (@internal; set by test/setup/gpu.ts from GRAPHTY_GPU_INSPECT, spec 11.9 item 2). */
export interface GpuDebugFlags {
    inspect: boolean;
}
