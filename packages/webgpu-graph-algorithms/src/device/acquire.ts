/**
 * Device acquisition (spec 2.2 steps 1-3, 2.6; contract 3.4): the adapter request, the software-adapter
 * test, the adapter summary, the requiredLimits / requiredFeatures builders and the device request. Nothing
 * above the device layer is constructed here (D26); src/context.ts sequences these calls. An adapter is
 * consumed by its first requestDevice (WebGPU 3.5.1; Dawn-node rejects the second call with
 * `OperationError: adapter is "consumed"`), so every device the package creates comes from a fresh adapter.
 */

import { WebGpuGraphError } from "../errors.js";
import type { AdapterInfoLike, AdapterSummary, LimitPolicy, RaisableLimit } from "../types/context.js";

/** The six raisable limits in the order create() requests them (spec 2.2). */
export const RAISABLE_LIMITS: readonly RaisableLimit[] = Object.freeze([
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
]);

/**
 * The WebGPU limit names summarizeAdapter reads (the GPUSupportedLimits members of @webgpu/types 0.1.72).
 * A fixed list because Chromium exposes the limits as prototype getters (Object.keys is empty) and Dawn-node as an
 * addon object; reading each name works identically on both, and a name the runtime lacks is simply absent.
 */
const LIMIT_NAMES: readonly string[] = [
    "maxTextureDimension1D",
    "maxTextureDimension2D",
    "maxTextureDimension3D",
    "maxTextureArrayLayers",
    "maxBindGroups",
    "maxBindGroupsPlusVertexBuffers",
    "maxBindingsPerBindGroup",
    "maxDynamicUniformBuffersPerPipelineLayout",
    "maxDynamicStorageBuffersPerPipelineLayout",
    "maxSampledTexturesPerShaderStage",
    "maxSamplersPerShaderStage",
    "maxStorageBuffersPerShaderStage",
    "maxStorageBuffersInVertexStage",
    "maxStorageBuffersInFragmentStage",
    "maxStorageTexturesPerShaderStage",
    "maxStorageTexturesInVertexStage",
    "maxStorageTexturesInFragmentStage",
    "maxUniformBuffersPerShaderStage",
    "maxUniformBufferBindingSize",
    "maxStorageBufferBindingSize",
    "minUniformBufferOffsetAlignment",
    "minStorageBufferOffsetAlignment",
    "maxVertexBuffers",
    "maxBufferSize",
    "maxVertexAttributes",
    "maxVertexBufferArrayStride",
    "maxInterStageShaderVariables",
    "maxColorAttachments",
    "maxColorAttachmentBytesPerSample",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
    "maxImmediateSize",
];

/** Adapters that created a device through requestDevice(): a second create({ adapter }) fails without a round trip. */
const consumedAdapters = new WeakSet<GPUAdapter>();

/**
 * The message of a thrown value.
 * @param err - what was caught
 * @returns the Error message, or the value as a string
 */
function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

/**
 * Software adapter test (spec 2.2 step 2): architecture "software" (Dawn llvmpipe) or "swiftshader", or
 * isFallbackAdapter === true (Chromium); the ONE reader of these fields.
 * @param info - the adapter info (structural, so tests can fake it)
 * @returns true for a software adapter
 */
export function isSoftwareAdapter(info: AdapterInfoLike): boolean {
    return info.architecture === "software" || info.architecture === "swiftshader" || info.isFallbackAdapter === true;
}

/**
 * The AdapterSummary of an adapter (info + features + limits); never requests a device.
 * @param adapter - an adapter (unused or not; only its info, features and limits are read)
 * @returns a frozen summary; `subgroupMinSize` / `subgroupMaxSize` are 0 when the info lacks them
 */
export function summarizeAdapter(adapter: GPUAdapter): AdapterSummary {
    const { info } = adapter;
    const source = adapter.limits as unknown as Readonly<Record<string, unknown>>;
    const limits: Record<string, number> = {};
    for (const name of LIMIT_NAMES) {
        const value = source[name];
        if (typeof value === "number") {
            limits[name] = value;
        }
    }
    return Object.freeze({
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        software: isSoftwareAdapter(info),
        subgroupMinSize: info.subgroupMinSize ?? 0,
        subgroupMaxSize: info.subgroupMaxSize ?? 0,
        features: Object.freeze(Array.from(adapter.features).sort()),
        limits: Object.freeze(limits),
    });
}

/**
 * Step 1 of create(): requestAdapter with the power preference; null -> E_NO_ADAPTER.
 * @param gpu - navigator.gpu or the Dawn handle
 * @param powerPreference - honoured by browsers, ignored by Dawn-node (spec 2.6)
 * @returns an UNUSED adapter
 */
export async function requestAdapter(gpu: GPU, powerPreference: GPUPowerPreference): Promise<GPUAdapter> {
    let adapter: GPUAdapter | null;
    try {
        adapter = await gpu.requestAdapter({ powerPreference });
    } catch (err) {
        const reason = `requestAdapter() threw: ${messageOf(err)}`;
        throw new WebGpuGraphError("E_NO_ADAPTER", reason, { reason });
    }
    if (adapter === null) {
        const reason = "requestAdapter() returned null: no usable WebGPU adapter";
        throw new WebGpuGraphError("E_NO_ADAPTER", reason, { reason });
    }
    return adapter;
}

/**
 * Step 3 of create(): the requiredLimits record for a policy, clamped to the adapter; an explicit value
 * above the adapter -> E_NO_DEVICE { reason: "limit" } before any requestDevice call, so the failure is
 * diagnosable.
 * @param adapter - the adapter the device will come from
 * @param policy - "default" (nothing), "raise" (every raisable limit at the adapter's value) or explicit values
 * @returns the record to pass as requiredLimits
 */
export function buildRequiredLimits(adapter: GPUAdapter, policy: LimitPolicy): Record<string, number> {
    const limits: Record<string, number> = {};
    if (policy === "default") {
        return limits;
    }
    if (policy === "raise") {
        for (const name of RAISABLE_LIMITS) {
            limits[name] = adapter.limits[name];
        }
        return limits;
    }
    for (const name of RAISABLE_LIMITS) {
        const requested = policy[name];
        if (requested === undefined) {
            continue;
        }
        const available = adapter.limits[name];
        if (requested > available) {
            throw new WebGpuGraphError(
                "E_NO_DEVICE",
                `requested ${name} = ${requested} exceeds the adapter's ${available}`,
                { reason: "limit", adapter: summarizeAdapter(adapter), limit: name, requested, available },
            );
        }
        limits[name] = requested;
    }
    return limits;
}

/**
 * Step 3 of create(): the requiredFeatures list = required + (optional intersect adapter.features); a missing
 * required feature -> E_NO_DEVICE { reason: "feature", requested: <name> }.
 * @param adapter - the adapter the device will come from
 * @param required - features the caller cannot do without
 * @param optional - features requested only when the adapter has them
 * @returns the deduplicated list, required first
 */
export function buildRequiredFeatures(
    adapter: GPUAdapter,
    required: readonly GPUFeatureName[],
    optional: readonly GPUFeatureName[],
): GPUFeatureName[] {
    const features: GPUFeatureName[] = [];
    for (const feature of required) {
        if (!adapter.features.has(feature)) {
            throw new WebGpuGraphError("E_NO_DEVICE", `required feature "${feature}" is not supported by the adapter`, {
                reason: "feature",
                adapter: summarizeAdapter(adapter),
                requested: feature,
            });
        }
        if (!features.includes(feature)) {
            features.push(feature);
        }
    }
    for (const feature of optional) {
        if (adapter.features.has(feature) && !features.includes(feature)) {
            features.push(feature);
        }
    }
    return features;
}

/**
 * The E_NO_DEVICE of a consumed adapter; the runtime's text lives in the error message only (contract 3.1
 * documents `details` as { reason, adapter, limit?, requested?, available? }).
 * @param adapter - the adapter that already created a device
 * @param message - the runtime's rejection message when there was a round trip
 * @returns E_NO_DEVICE { reason: "consumed", adapter }
 */
function consumedError(adapter: GPUAdapter, message: string): WebGpuGraphError {
    return new WebGpuGraphError(
        "E_NO_DEVICE",
        `the adapter already created a device (an adapter is consumed by one requestDevice, spec 2.2 step 1): ${message}`,
        { reason: "consumed", adapter: summarizeAdapter(adapter) },
    );
}

/**
 * Step 3 of create(): requestDevice; a rejection -> E_NO_DEVICE with the adapter summary ("consumed" when
 * the adapter already created a device, detected by the OperationError message or a prior-use record).
 * @param adapter - an UNUSED adapter
 * @param descriptor - requiredLimits, requiredFeatures and the label
 * @returns the device
 */
export async function requestDevice(adapter: GPUAdapter, descriptor: GPUDeviceDescriptor): Promise<GPUDevice> {
    if (consumedAdapters.has(adapter)) {
        throw consumedError(adapter, "recorded by this package");
    }
    let device: GPUDevice;
    try {
        device = await adapter.requestDevice(descriptor);
    } catch (err) {
        const message = messageOf(err);
        if (/consumed|already been used/i.test(message)) {
            throw consumedError(adapter, message);
        }
        throw new WebGpuGraphError("E_NO_DEVICE", `requestDevice() rejected: ${message}`, {
            reason: "requestDevice",
            adapter: summarizeAdapter(adapter),
        });
    }
    consumedAdapters.add(adapter);
    return device;
}
