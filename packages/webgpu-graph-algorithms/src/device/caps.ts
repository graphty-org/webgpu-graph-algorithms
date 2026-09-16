/**
 * Capability capture (spec 2.2 step 4, 3.3, 5.1, 5.2; contract 3.4): the GpuCaps record of a DEVICE (its
 * limits and features, never the adapter's), the zero-cost variant for an adopted device, the two planner
 * facts asserted at create(), and the workgroup size a device runs.
 */

import { MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE } from "../constants.js";
import { WebGpuGraphError } from "../errors.js";
import type { AdapterInfoLike, GpuCaps, PlanCaps } from "../types/context.js";
import { isSoftwareAdapter } from "./acquire.js";

/**
 * Step 4 of create(): the GpuCaps of a DEVICE (device.limits, never adapter.limits) with the adapter info
 * and the runtime tag. The subgroup sizes are the adapter's only when the device has the "subgroups"
 * feature, else 0 (spec 3.3).
 * @param device - the device the caps describe
 * @param info - the adapter's info (vendor, architecture, subgroup sizes)
 * @param runtime - the tag the creating entry sets ("unknown" for consumers of the core)
 * @param wgslFeatures - navigator.gpu.wgslLanguageFeatures or the Dawn equivalent (informational)
 * @returns the capability record
 */
export function captureCaps(
    device: GPUDevice,
    info: AdapterInfoLike,
    runtime: "browser" | "node" | "unknown",
    wgslFeatures: Iterable<string>,
): GpuCaps {
    const features: ReadonlySet<string> = new Set(device.features);
    const subgroups = features.has("subgroups");
    return {
        limits: device.limits,
        features,
        wgslFeatures: new Set(wgslFeatures),
        subgroupMinSize: subgroups ? (info.subgroupMinSize ?? 0) : 0,
        subgroupMaxSize: subgroups ? (info.subgroupMaxSize ?? 0) : 0,
        software: isSoftwareAdapter(info),
        runtime,
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
    };
}

/**
 * GpuContext.from(device, info?): caps from device.limits / device.features and the partial info given;
 * software false and runtime "unknown" unless given; the subgroup sizes are honoured only when the device
 * has the feature.
 * @param device - the adopted device
 * @param info - what the caller knows about the adapter, if anything
 * @returns the capability record
 */
export function capsFromDevice(device: GPUDevice, info: Partial<GpuCaps> | undefined): GpuCaps {
    const features: ReadonlySet<string> = new Set(device.features);
    const subgroups = features.has("subgroups");
    return {
        limits: device.limits,
        features,
        wgslFeatures: info?.wgslFeatures ?? new Set<string>(),
        subgroupMinSize: subgroups ? (info?.subgroupMinSize ?? 0) : 0,
        subgroupMaxSize: subgroups ? (info?.subgroupMaxSize ?? 0) : 0,
        software: info?.software ?? false,
        runtime: info?.runtime ?? "unknown",
        vendor: info?.vendor ?? "",
        architecture: info?.architecture ?? "",
        device: info?.device ?? "",
        description: info?.description ?? "",
    };
}

/**
 * The workgroup size a device runs: min(WORKGROUP_SIZE, caps.limits.maxComputeInvocationsPerWorkgroup) rounded
 * down to a power of two (spec 5.1; the formula of contract 3.2 / 3.4). maxComputeWorkgroupSizeX is not a term:
 * every adapter supports its 256 default (WebGPU 3.6.2), so the result never exceeds it. Rounds down by doubling
 * (no bitwise operator).
 * @param caps - the capability record (a GpuCaps is a PlanCaps)
 * @returns the workgroup size
 */
export function workgroupSizeFor(caps: PlanCaps): number {
    const cap = Math.min(WORKGROUP_SIZE, caps.limits.maxComputeInvocationsPerWorkgroup);
    let size = 1;
    while (size * 2 <= cap) {
        size *= 2;
    }
    return size;
}

/**
 * Asserts the two facts the planners bake in: maxComputeWorkgroupsPerDimension === MAX_WORKGROUPS_PER_DIM
 * (the prelude's linear_id and plan1d / plan2d use the constant) and WG is a power of two >= 64 (spec 2.2,
 * 5.2). The limit is never requested, so a device reports the spec default whatever its adapter says.
 * @param caps - the capability record of the device
 */
export function assertPlanLimits(caps: GpuCaps): void {
    const perDim = caps.limits.maxComputeWorkgroupsPerDimension;
    if (perDim !== MAX_WORKGROUPS_PER_DIM) {
        throw new WebGpuGraphError(
            "E_NO_DEVICE",
            `maxComputeWorkgroupsPerDimension is ${perDim}; the dispatch planner and the WGSL prelude bake ${MAX_WORKGROUPS_PER_DIM}`,
            {
                reason: "maxComputeWorkgroupsPerDimension",
                adapter: null,
                requested: MAX_WORKGROUPS_PER_DIM,
                available: perDim,
            },
        );
    }
    const wg = workgroupSizeFor(caps);
    if (wg < 64) {
        throw new WebGpuGraphError(
            "E_NO_DEVICE",
            `the device allows a workgroup of ${wg} invocations; the kernels need a power of two >= 64`,
            {
                reason: "limit",
                adapter: null,
                limit: "maxComputeInvocationsPerWorkgroup",
                requested: 64,
                available: wg,
            },
        );
    }
}
