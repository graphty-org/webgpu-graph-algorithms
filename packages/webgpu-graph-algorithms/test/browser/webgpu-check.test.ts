/**
 * The first browser smoke test (contract 5.5; spec 11.6 (1), 13 row P0): navigator.gpu exists under the spec
 * 12.2 flags, an adapter is granted, the adapter matches the policy AND the flag set vitest.config.ts selected
 * (PLAN DECISION 11: a silent SwiftShader fallback under the NVIDIA flags is red), the three variables arrive
 * through import.meta.env (contract 2.5; section 9 item 4), and one trivial compute dispatch round-trips twice.
 */

import { isSoftwareAdapter } from "../../src/device/acquire.js";
import { BufferUsage, MapMode } from "../../src/device/webgpu-constants.js";
import {
    browserExpectedAdapter,
    browserExpectsSoftware,
    browserGpu,
    browserGrantedSoftware,
    browserPolicy,
    browserScale,
    browserWebGpu,
    requireBrowserGpu,
} from "../setup/browser.js";

/** One workgroup per item; every item writes 42 + 2 * its index (the same trivial kernel as test/device/acquire.test.ts). */
const TRIVIAL_WGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = 42u + 2u * gid.x;
}
`;

/** Compiles TRIVIAL_WGSL, dispatches 4 workgroups and reads the 16 bytes back (copy before unmap). */
async function runTrivialKernel(device: GPUDevice): Promise<Uint32Array> {
    const module = device.createShaderModule({ code: TRIVIAL_WGSL, label: "p0-trivial" });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    expect(errors).toEqual([]);
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const out = device.createBuffer({ size: 16, usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC, label: "p0-out" });
    const staging = device.createBuffer({
        size: 16,
        usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
        label: "p0-staging",
    });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: out } }],
    });
    const encoder = device.createCommandEncoder({ label: "p0-trivial" });
    const pass = encoder.beginComputePass({ label: "p0-trivial" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(4);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, staging, 0, 16);
    device.queue.submit([encoder.finish()]);
    const validation = await device.popErrorScope();
    expect(validation).toBeNull();
    await staging.mapAsync(MapMode.READ);
    const result = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    out.destroy();
    staging.destroy();
    return result;
}

/** The adapter Chromium grants for the high-performance preference (never null once requireBrowserGpu passed). */
async function grantedAdapter(): Promise<GPUAdapter> {
    const gpu = browserWebGpu();
    if (gpu === undefined) {
        throw new Error("navigator.gpu is absent");
    }
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) {
        throw new Error("requestAdapter() returned null");
    }
    return adapter;
}

describe("browser: WebGPU check (spec 13 row P0; contract 5.5)", () => {
    it("receives GRAPHTY_GPU_REQUIRE, GRAPHTY_BROWSER_GPU and GRAPHTY_NOISE_FLOOR_WRITE through import.meta.env", () => {
        expect(["nvidia", "swiftshader", "metal", "warp"]).toContain(import.meta.env.GRAPHTY_BROWSER_GPU);
        expect(typeof import.meta.env.GRAPHTY_GPU_REQUIRE).toBe("string");
        expect(typeof import.meta.env.GRAPHTY_NOISE_FLOOR_WRITE).toBe("string");
        expect(browserPolicy().raw).toBe(import.meta.env.GRAPHTY_GPU_REQUIRE);
        expect(browserGpu()).toBe(import.meta.env.GRAPHTY_BROWSER_GPU);
        // nvidia is hardware, swiftshader and warp are software (1 / 50); metal follows the adapter the run got
        const expectedScale = (browserExpectsSoftware() ?? browserGrantedSoftware()) ? 1 / 50 : 1;
        expect(browserScale()).toBe(expectedScale);
        console.warn(
            `[browser] import.meta.env GRAPHTY_GPU_REQUIRE=${import.meta.env.GRAPHTY_GPU_REQUIRE} GRAPHTY_BROWSER_GPU=${import.meta.env.GRAPHTY_BROWSER_GPU} GRAPHTY_NOISE_FLOOR_WRITE=${import.meta.env.GRAPHTY_NOISE_FLOOR_WRITE}`,
        );
    });

    it("exposes navigator.gpu", () => {
        expect(browserWebGpu()).toBeDefined();
    });

    it("grants an adapter that matches the policy and the selected flag set", async (t) => {
        await requireBrowserGpu(t);
        const adapter = await grantedAdapter();
        const { info, limits } = adapter;
        const software = isSoftwareAdapter(info);
        console.warn(
            `[browser] adapter vendor=${info.vendor} architecture=${info.architecture} device=${info.device} description=${info.description} isFallbackAdapter=${String(info.isFallbackAdapter)} software=${software}`,
        );
        console.warn(
            `[browser] adapter limits maxBufferSize=${limits.maxBufferSize} maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize} maxStorageBuffersPerShaderStage=${limits.maxStorageBuffersPerShaderStage} maxComputeWorkgroupsPerDimension=${limits.maxComputeWorkgroupsPerDimension}`,
        );
        // Chromium reports isFallbackAdapter as a boolean (so does Dawn-node 0.4.0; contract correction 5)
        expect(typeof info.isFallbackAdapter).toBe("boolean");
        const named = browserExpectedAdapter();
        if (named !== null) {
            expect(
                { vendor: info.vendor, software },
                "the flag set vitest.config.ts selected (GRAPHTY_BROWSER_GPU) must match the adapter Chromium granted: an NVIDIA flag set that yields SwiftShader means the driver did not initialise (dev box: LD_LIBRARY_PATH or GRAPHTY_EGL_LIB_DIR, spec 12.2; docs/HEADLESS_GPU_REPORT.md)",
            ).toEqual({ vendor: named.vendor, software: browserExpectsSoftware() });
        }
        const policy = browserPolicy();
        if (policy.level === "vendor") {
            expect(info.vendor).toBe(policy.vendor);
            expect(info.isFallbackAdapter).toBe(false);
            expect(software).toBe(false);
        }
        if (policy.level === "hardware") {
            expect(software).toBe(false);
        }
        expect(limits.maxComputeWorkgroupsPerDimension).toBeGreaterThanOrEqual(65535);
    });

    it("runs one trivial compute dispatch twice with identical results", async (t) => {
        await requireBrowserGpu(t);
        const adapter = await grantedAdapter();
        const device = await adapter.requestDevice({ label: "p0-webgpu-check" });
        try {
            const first = await runTrivialKernel(device);
            const second = await runTrivialKernel(device);
            expect([...first]).toEqual([42, 44, 46, 48]);
            expect([...second]).toEqual([...first]);
        } finally {
            device.destroy();
        }
    });
});
