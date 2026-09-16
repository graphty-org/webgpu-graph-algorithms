/**
 * The ./browser entry in Chromium (spec 11.6 item 1; contract 5.5, 3.6): probeBrowserWebGpu reports the software flag the
 * flag set implies (SwiftShader -> software, NVIDIA -> hardware and not a fallback adapter), requestGpuContext creates a
 * context tagged runtime "browser", and the typed E_NO_WEBGPU appears when navigator.gpu is absent (simulated by passing
 * undefined to probe, and by create() without a gpu). Runs on SwiftShader (default lane) and NVIDIA (GPU lane / dev box).
 */

import { probeBrowserWebGpu, requestGpuContext } from "../../src/browser/index.js";
import { MAX_WORKGROUPS_PER_DIM } from "../../src/constants.js";
import { GpuContext } from "../../src/context.js";
import { isSoftwareAdapter } from "../../src/device/acquire.js";
import { browserGpu, browserPolicy, requireBrowserGpu } from "../setup/browser.js";

describe("./browser entry (spec 11.6 item 1)", () => {
    it("probeBrowserWebGpu reports the adapter the flag set selects, with the software flag the policy expects", async (t) => {
        await requireBrowserGpu(t);
        const result = await probeBrowserWebGpu();
        expect(result.ok).toBe(true);
        expect(result.code).toBe("OK");
        expect(result.reason).toBeNull();
        if (result.adapter === null || result.summary === null) {
            throw new Error("probe ok but adapter / summary null");
        }
        const expectedSoftware = browserGpu() === "swiftshader";
        expect(result.summary.software).toBe(expectedSoftware);
        expect(isSoftwareAdapter(result.adapter.info)).toBe(expectedSoftware);
        if (browserGpu() === "nvidia") {
            expect(result.summary.vendor).toBe("nvidia");
            expect(result.adapter.info.isFallbackAdapter).toBe(false);
        } else {
            expect(result.summary.vendor).toBe("google");
            expect(result.summary.architecture).toBe("swiftshader");
        }
        const policy = browserPolicy();
        if (policy.level === "vendor") {
            expect(result.summary.vendor).toBe(policy.vendor);
        }
        console.warn(
            `[entry] probe vendor=${result.summary.vendor} architecture=${result.summary.architecture} software=${String(result.summary.software)} subgroups=${result.summary.subgroupMinSize}/${result.summary.subgroupMaxSize}`,
        );
    });

    it("probeBrowserWebGpu({ rejectSoftware: true }) is E_SOFTWARE_ONLY on SwiftShader and OK on NVIDIA, never a throw", async (t) => {
        await requireBrowserGpu(t);
        const result = await probeBrowserWebGpu({ rejectSoftware: true });
        if (browserGpu() === "swiftshader") {
            expect(result.ok).toBe(false);
            expect(result.code).toBe("E_SOFTWARE_ONLY");
            expect(typeof result.reason).toBe("string");
            if (result.summary !== null) {
                expect(result.summary.software).toBe(true);
            }
        } else {
            expect(result.ok).toBe(true);
            expect(result.code).toBe("OK");
        }
    });

    it("requestGpuContext creates a ready context tagged runtime browser that dispose() destroys", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await requestGpuContext({ label: "entry" });
        try {
            expect(ctx.state).toBe("ready");
            expect(ctx.caps.runtime).toBe("browser");
            expect(ctx.ownsDevice).toBe(true);
            expect(ctx.label).toBe("entry");
            expect(ctx.caps.software).toBe(browserGpu() === "swiftshader");
            expect(ctx.caps.limits.maxComputeWorkgroupsPerDimension).toBe(MAX_WORKGROUPS_PER_DIM);
            expect(ctx.workgroupSize).toBeGreaterThanOrEqual(64);
            expect(ctx.residency.stats().buffers).toBe(0);
        } finally {
            ctx.dispose();
        }
        expect(ctx.state).toBe("disposed");
        ctx.dispose(); // idempotent
    });

    it("GpuContext.probe({ gpu: undefined }) is the typed E_NO_WEBGPU without throwing", async () => {
        const result = await GpuContext.probe({ gpu: undefined });
        expect(result.ok).toBe(false);
        expect(result.code).toBe("E_NO_WEBGPU");
        expect(result.adapter).toBeNull();
        expect(result.summary).toBeNull();
        expect(typeof result.reason).toBe("string");
    });

    it("GpuContext.create without gpu, adapter or device rejects with E_NO_WEBGPU", async () => {
        await expect(GpuContext.create({ runtime: "browser", label: "no-gpu" })).rejects.toMatchObject({
            name: "WebGpuGraphError",
            code: "E_NO_WEBGPU",
        });
    });
});
