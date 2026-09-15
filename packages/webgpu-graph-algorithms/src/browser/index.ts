/// <reference types="@webgpu/types" preserve="true" />
/**
 * The ./browser entry (spec 2.1, 2.3, 3.4; contract 3.6): the only directory of the package allowed to read
 * navigator.gpu. probeBrowserWebGpu never throws; requestGpuContext tags the context runtime "browser".
 */

import { GpuContext } from "../context.js";
import { WebGpuGraphError } from "../errors.js";
import type { GpuContextOptions, ProbeResult } from "../types/context.js";

/** Options of the browser helpers (spec 3.4); a type alias, not an empty `extends` interface, which strictTypeChecked's no-empty-object-type (allowInterfaces "never") reports. */
export type BrowserGpuOptions = Omit<GpuContextOptions, "gpu" | "device" | "runtime">;

/**
 * navigator.gpu, or undefined when the runtime has no navigator or no WebGPU (Node 22 has a navigator
 * without gpu; browsers without WebGPU have navigator.gpu undefined).
 * @returns the GPU object or undefined
 */
function navigatorGpu(): GPU | undefined {
    if (typeof navigator === "undefined") {
        return undefined;
    }
    const candidate: { readonly gpu?: GPU | undefined } = navigator;
    return candidate.gpu;
}

/**
 * navigator.gpu absent -> { ok: false, code: "E_NO_WEBGPU" }; else GpuContext.probe. Never throws.
 * @param options - the power preference and the software policy
 * @returns the probe result
 */
export function probeBrowserWebGpu(options?: BrowserGpuOptions): Promise<ProbeResult> {
    return GpuContext.probe({
        gpu: navigatorGpu(),
        powerPreference: options?.powerPreference ?? "high-performance",
        rejectSoftware: options?.rejectSoftware,
    });
}

/**
 * GpuContext.create({ gpu: navigator.gpu, powerPreference: "high-performance", runtime: "browser", ...options });
 * pass `{ adapter: probe.adapter }` to reuse the probed adapter; create() honours rejectSoftware.
 * @param options - see BrowserGpuOptions
 * @returns the context
 */
export function requestGpuContext(options?: BrowserGpuOptions): Promise<GpuContext> {
    const gpu = navigatorGpu();
    if (gpu === undefined && options?.adapter === undefined) {
        return Promise.reject(
            new WebGpuGraphError("E_NO_WEBGPU", "navigator.gpu is undefined: this browser or context has no WebGPU", {
                reason: "navigator.gpu is undefined",
                hint: "WebGPU needs a supporting browser and a secure context (https or localhost)",
            }),
        );
    }
    return GpuContext.create({ powerPreference: "high-performance", ...options, gpu, runtime: "browser" });
}
