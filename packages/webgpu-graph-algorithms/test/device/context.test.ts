/**
 * src/context.ts, src/device/caps.ts, the P1 additions of src/device/acquire.ts and the P1 Profiler shell
 * (contract 3.4, 3.5, 3.9; spec 2.2, 5.7, 11.5). Faked GpuCaps tables give the hand-computed expectations;
 * the device-backed cases use acquireRaw() (a raw device, no context) or acquire() (a context the setup
 * disposes). Contexts created directly here are disposed in a finally block. The deliberately broken bind
 * groups are created on contexts that are NOT acquire()d, so the setup's uncaptured hook never sees them.
 */

import { fromEdgeArrays, type GraphSnapshot } from "@graphty/graph-format";

import { GpuContext } from "../../src/context.js";
import {
    buildRequiredFeatures,
    buildRequiredLimits,
    isSoftwareAdapter,
    RAISABLE_LIMITS,
    requestAdapter,
    requestDevice,
    summarizeAdapter,
} from "../../src/device/acquire.js";
import { assertPlanLimits, capsFromDevice, captureCaps, workgroupSizeFor } from "../../src/device/caps.js";
import { BufferUsage, ShaderStage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError, type WebGpuGraphError, type WebGpuGraphErrorCode } from "../../src/errors.js";
import { type PassTiming, Profiler } from "../../src/kernel/profiler.js";
import type { GpuCaps, PlanLimits, RaisableLimit } from "../../src/types/context.js";
import type { GpuRunOptions } from "../../src/types/run.js";
import { acquire, acquireNullBackend, acquireRaw, isSoftware, requireGpu, uncapturedErrors } from "../setup/gpu.js";

const SPEC_DEFAULT_LIMITS: PlanLimits = {
    maxBufferSize: 268_435_456,
    maxStorageBufferBindingSize: 134_217_728,
    maxStorageBuffersPerShaderStage: 8,
    minStorageBufferOffsetAlignment: 256,
    minUniformBufferOffsetAlignment: 256,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxUniformBufferBindingSize: 65_536,
};

/** A GpuCaps over the spec-default limits with some of them overridden (the branded GPUSupportedLimits needs the cast). */
function fakeCaps(limits: Partial<PlanLimits> = {}, flags: Partial<Omit<GpuCaps, "limits">> = {}): GpuCaps {
    return {
        limits: { ...SPEC_DEFAULT_LIMITS, ...limits } as unknown as GPUSupportedLimits,
        features: new Set<string>(),
        wgslFeatures: new Set<string>(),
        subgroupMinSize: 0,
        subgroupMaxSize: 0,
        software: false,
        runtime: "unknown",
        vendor: "fake",
        architecture: "fake",
        device: "",
        description: "",
        ...flags,
    };
}

/** Undirected 0-1, 1-2, 2-3, 0-2, 0-3 on five nodes; node 4 isolated. outDegree = [3, 2, 3, 2, 0]. */
function smallGraph(): GraphSnapshot {
    const src = new Uint32Array([0, 1, 2, 0, 0]);
    const dst = new Uint32Array([1, 2, 3, 2, 3]);
    return fromEdgeArrays({ directed: false, nodeCount: 5, src, dst });
}

/**
 * A bind group whose 16-byte buffer is smaller than the layout's minBindingSize of 1024: a validation
 * error at createBindGroup, delivered as an uncapturederror event (synchronously under Dawn-node).
 */
function brokenBindGroup(device: GPUDevice, label: string): void {
    const layout = device.createBindGroupLayout({
        label: `${label}/layout`,
        entries: [{ binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "storage", minBindingSize: 1024 } }],
    });
    const small = device.createBuffer({ label: `${label}/small`, size: 16, usage: BufferUsage.STORAGE });
    device.createBindGroup({ label, layout, entries: [{ binding: 0, resource: { buffer: small } }] });
    small.destroy();
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
    try {
        await promise;
    } catch (err) {
        return err;
    }
    throw new Error("expected the promise to reject");
}

function thrown(fn: () => unknown): unknown {
    try {
        fn();
    } catch (err) {
        return err;
    }
    throw new Error("expected the function to throw");
}

function nonNull<T>(value: T | null): T {
    if (value === null) {
        throw new Error("expected a non-null value");
    }
    return value;
}

function expectCode(err: unknown, code: WebGpuGraphErrorCode, details?: Record<string, unknown>): WebGpuGraphError {
    expect(isWebGpuGraphError(err), `expected a WebGpuGraphError, got ${String(err)}`).toBe(true);
    const error = err as WebGpuGraphError;
    expect(error.code).toBe(code);
    if (details !== undefined) {
        expect(error.details).toMatchObject(details);
    }
    return error;
}

describe("caps.ts", () => {
    describe("workgroupSizeFor", () => {
        it("is min(WORKGROUP_SIZE, maxComputeInvocationsPerWorkgroup) rounded down to a power of two (contract 3.2 / 3.4)", () => {
            const wg = (inv: number): number => workgroupSizeFor(fakeCaps({ maxComputeInvocationsPerWorkgroup: inv }));
            expect(wg(1024)).toBe(256);
            expect(wg(256)).toBe(256);
            expect(wg(128)).toBe(128);
            expect(wg(200)).toBe(128);
            expect(wg(64)).toBe(64);
            expect(wg(32)).toBe(32);
            expect(wg(1000)).toBe(256);
            expect(wg(100)).toBe(64);
            // maxComputeWorkgroupSizeX is NOT a term: every adapter supports the 256 default (WebGPU 3.6.2), so the
            // result never exceeds it on a real device; a fake that lowers it does not change the answer.
            expect(
                workgroupSizeFor(fakeCaps({ maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupSizeX: 64 })),
            ).toBe(256);
        });
    });

    describe("assertPlanLimits", () => {
        it("accepts maxComputeWorkgroupsPerDimension 65535 with a workgroup of at least 64", () => {
            expect(() => {
                assertPlanLimits(fakeCaps());
            }).not.toThrow();
            expect(() => {
                assertPlanLimits(fakeCaps({ maxComputeInvocationsPerWorkgroup: 64, maxComputeWorkgroupSizeX: 64 }));
            }).not.toThrow();
        });

        it("rejects any other maxComputeWorkgroupsPerDimension with E_NO_DEVICE { reason: maxComputeWorkgroupsPerDimension }", () => {
            expectCode(
                thrown(() => {
                    assertPlanLimits(fakeCaps({ maxComputeWorkgroupsPerDimension: 65_536 }));
                }),
                "E_NO_DEVICE",
                { reason: "maxComputeWorkgroupsPerDimension", adapter: null, requested: 65_535, available: 65_536 },
            );
            expectCode(
                thrown(() => {
                    assertPlanLimits(fakeCaps({ maxComputeWorkgroupsPerDimension: 32_768 }));
                }),
                "E_NO_DEVICE",
                { reason: "maxComputeWorkgroupsPerDimension", available: 32_768 },
            );
        });

        it("rejects a workgroup below 64 with E_NO_DEVICE { reason: limit }", () => {
            expectCode(
                thrown(() => {
                    assertPlanLimits(fakeCaps({ maxComputeInvocationsPerWorkgroup: 32 }));
                }),
                "E_NO_DEVICE",
                { reason: "limit", limit: "maxComputeInvocationsPerWorkgroup", requested: 64, available: 32 },
            );
        });
    });

    describe("captureCaps", () => {
        it("reads device.limits and device.features, tags the runtime, copies the info and the wgsl features", async (t) => {
            requireGpu(t);
            const { device, info } = await acquireRaw();
            const caps = captureCaps(device, info, "node", ["a", "b"]);
            expect(caps.limits.maxBufferSize).toBe(device.limits.maxBufferSize);
            expect(caps.limits.maxComputeWorkgroupsPerDimension).toBe(device.limits.maxComputeWorkgroupsPerDimension);
            expect(caps.features.has("subgroups")).toBe(device.features.has("subgroups"));
            expect(caps.wgslFeatures).toEqual(new Set(["a", "b"]));
            expect(caps.runtime).toBe("node");
            expect(caps.vendor).toBe(info.vendor);
            expect(caps.architecture).toBe(info.architecture);
            expect(caps.device).toBe(info.device);
            expect(caps.description).toBe(info.description);
            expect(caps.software).toBe(isSoftwareAdapter(info));
            if (device.features.has("subgroups")) {
                expect(caps.subgroupMinSize).toBe(info.subgroupMinSize ?? 0);
                expect(caps.subgroupMaxSize).toBe(info.subgroupMaxSize ?? 0);
                expect(caps.subgroupMaxSize).toBeGreaterThanOrEqual(4);
            }
            expect(() => {
                assertPlanLimits(caps);
            }).not.toThrow();
        });

        it("zeroes the subgroup sizes when the device lacks the feature, whatever the adapter info says", async (t) => {
            requireGpu(t);
            const { device } = await acquireRaw({ optionalFeatures: [] });
            expect(device.features.has("subgroups")).toBe(false);
            const info = {
                vendor: "fake",
                architecture: "fake",
                device: "",
                description: "",
                subgroupMinSize: 32,
                subgroupMaxSize: 32,
            };
            const caps = captureCaps(device, info, "unknown", []);
            expect(caps.subgroupMinSize).toBe(0);
            expect(caps.subgroupMaxSize).toBe(0);
            expect(caps.wgslFeatures.size).toBe(0);
        });
    });

    describe("capsFromDevice", () => {
        it("defaults to runtime unknown, software false and empty strings; honours the partial info given", async (t) => {
            requireGpu(t);
            const { device } = await acquireRaw();
            const plain = capsFromDevice(device, undefined);
            expect(plain.runtime).toBe("unknown");
            expect(plain.software).toBe(false);
            expect(plain.vendor).toBe("");
            expect(plain.description).toBe("");
            expect(plain.wgslFeatures.size).toBe(0);
            expect(plain.limits.maxBufferSize).toBe(device.limits.maxBufferSize);
            expect(plain.features.has("subgroups")).toBe(device.features.has("subgroups"));
            const given = capsFromDevice(device, {
                runtime: "browser",
                software: true,
                vendor: "v",
                architecture: "a",
                description: "d",
                wgslFeatures: new Set(["x"]),
                subgroupMinSize: 8,
                subgroupMaxSize: 32,
            });
            expect(given.runtime).toBe("browser");
            expect(given.software).toBe(true);
            expect(given.vendor).toBe("v");
            expect(given.architecture).toBe("a");
            expect(given.description).toBe("d");
            expect(given.wgslFeatures).toEqual(new Set(["x"]));
            if (device.features.has("subgroups")) {
                expect(given.subgroupMinSize).toBe(8);
                expect(given.subgroupMaxSize).toBe(32);
            } else {
                expect(given.subgroupMinSize).toBe(0);
                expect(given.subgroupMaxSize).toBe(0);
            }
        });
    });
});

describe("acquire.ts (P1 additions)", () => {
    it("RAISABLE_LIMITS is the six-limit list of spec 2.2 without maxComputeWorkgroupsPerDimension", () => {
        const expected: RaisableLimit[] = [
            "maxBufferSize",
            "maxStorageBufferBindingSize",
            "maxStorageBuffersPerShaderStage",
            "maxComputeWorkgroupStorageSize",
            "maxComputeInvocationsPerWorkgroup",
            "maxComputeWorkgroupSizeX",
        ];
        expect([...RAISABLE_LIMITS]).toEqual(expected);
        expect(Object.isFrozen(RAISABLE_LIMITS)).toBe(true);
    });

    it("buildRequiredLimits: default requests nothing, raise requests every raisable limit at the adapter's value", async (t) => {
        requireGpu(t);
        const { adapter } = await acquireRaw();
        expect(buildRequiredLimits(adapter, "default")).toEqual({});
        const raised = buildRequiredLimits(adapter, "raise");
        expect(Object.keys(raised)).toEqual([...RAISABLE_LIMITS]);
        for (const name of RAISABLE_LIMITS) {
            expect(raised[name]).toBe(adapter.limits[name]);
        }
    });

    it("buildRequiredLimits: an object requests exactly its values; a value above the adapter -> E_NO_DEVICE { reason: limit }", async (t) => {
        requireGpu(t);
        const { adapter } = await acquireRaw();
        const available = adapter.limits.maxStorageBuffersPerShaderStage;
        expect(buildRequiredLimits(adapter, { maxStorageBuffersPerShaderStage: available })).toEqual({
            maxStorageBuffersPerShaderStage: available,
        });
        expect(buildRequiredLimits(adapter, { maxBufferSize: undefined, maxComputeWorkgroupSizeX: 64 })).toEqual({
            maxComputeWorkgroupSizeX: 64,
        });
        const error = expectCode(
            thrown(() => buildRequiredLimits(adapter, { maxStorageBuffersPerShaderStage: available + 1 })),
            "E_NO_DEVICE",
            { reason: "limit", limit: "maxStorageBuffersPerShaderStage", requested: available + 1, available },
        );
        expect(error.details.adapter).toEqual(summarizeAdapter(adapter));
    });

    it("buildRequiredFeatures: required + (optional intersect adapter), deduplicated; a missing required feature -> E_NO_DEVICE { reason: feature }", async (t) => {
        requireGpu(t);
        const { adapter } = await acquireRaw();
        const missing = "graphty-nonexistent-feature" as GPUFeatureName;
        expect(adapter.features.has(missing)).toBe(false);
        const present = [...adapter.features] as GPUFeatureName[];
        expect(buildRequiredFeatures(adapter, [], [])).toEqual([]);
        expect(buildRequiredFeatures(adapter, [], [missing])).toEqual([]);
        expect(buildRequiredFeatures(adapter, [], ["subgroups", "timestamp-query", missing])).toEqual(
            (["subgroups", "timestamp-query"] as GPUFeatureName[]).filter((f) => adapter.features.has(f)),
        );
        if (present.length > 0) {
            const one = present[0];
            expect(buildRequiredFeatures(adapter, [one], [one, missing])).toEqual([one]);
        }
        const error = expectCode(
            thrown(() => buildRequiredFeatures(adapter, [missing], [])),
            "E_NO_DEVICE",
            {
                reason: "feature",
                requested: missing,
            },
        );
        expect(error.details.adapter).toEqual(summarizeAdapter(adapter));
    });

    it("requestAdapter returns a fresh adapter; requestDevice consumes it and refuses a second device without a round trip", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const adapter = await requestAdapter(gpu, "high-performance");
        expect(typeof adapter.info.vendor).toBe("string");
        const device = await requestDevice(adapter, { label: "first" });
        device.destroy();
        const error = expectCode(await rejection(requestDevice(adapter, { label: "second" })), "E_NO_DEVICE", {
            reason: "consumed",
        });
        expect(error.details.adapter).toEqual(summarizeAdapter(adapter));
    });

    it("requestDevice recognises an adapter consumed OUTSIDE the package from the OperationError message", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const adapter = await requestAdapter(gpu, "high-performance");
        const device = await adapter.requestDevice({ label: "outside" });
        device.destroy();
        expectCode(await rejection(requestDevice(adapter, { label: "second" })), "E_NO_DEVICE", { reason: "consumed" });
    });

    it("requestDevice maps any other rejection to E_NO_DEVICE { reason: requestDevice } with the adapter summary", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const adapter = await requestAdapter(gpu, "high-performance");
        const error = expectCode(
            await rejection(
                requestDevice(adapter, {
                    label: "too-much",
                    requiredLimits: {
                        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage + 1,
                    },
                }),
            ),
            "E_NO_DEVICE",
            { reason: "requestDevice" },
        );
        expect(error.details.adapter).toEqual(summarizeAdapter(adapter));
        expect(error.message).toContain("requestDevice() rejected");
        expect(Object.keys(error.details).sort()).toEqual(["adapter", "reason"]);
    });
});

describe("Profiler (the P2-T1 form; test/kernel/profiler.test.ts covers the timings)", () => {
    it("is enabled iff the device carries timestamp-query, echoes quantised and destroys without error", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const profiler = new Profiler(device, true, false);
        // P2-T1: enabled follows the granted feature (contract 3.9); acquireRaw() requests it by default
        const granted = device.features.has("timestamp-query");
        expect(profiler.enabled).toBe(granted);
        expect(profiler.quantised).toBe(false);
        expect(profiler.beginPass("any") === undefined).toBe(!granted);
        profiler.destroy();
        profiler.destroy();
        expect(profiler.beginPass("after-destroy")).toBeUndefined();
        const browserLike = new Profiler(device, false, true, 64);
        expect(browserLike.enabled).toBe(false);
        expect(browserLike.quantised).toBe(true);
        expect(browserLike.beginPass("x")).toBeUndefined();
        browserLike.destroy();
        // the two exported types are referenced here until their consumers land (PassTiming: P2-T1; GpuRunOptions: P1-T5)
        const timing: PassTiming = { label: "pass", ns: 1024 };
        expect(timing.ns).toBe(1024);
        const run: GpuRunOptions = { dest: undefined, signal: undefined, onProgress: undefined };
        expect(run.dest).toBeUndefined();
    });
});

describe("GpuContext.probe", () => {
    it("OK: an UNUSED adapter and its summary; the adapter can then be given to create({ adapter })", async (t) => {
        requireGpu(t);
        const { gpu, info } = await acquireRaw();
        const result = await GpuContext.probe({ gpu });
        expect(result.ok).toBe(true);
        expect(result.code).toBe("OK");
        expect(result.reason).toBeNull();
        const adapter = nonNull(result.adapter);
        const summary = nonNull(result.summary);
        expect(summary.vendor).toBe(info.vendor);
        expect(summary.architecture).toBe(info.architecture);
        expect(summary.software).toBe(isSoftwareAdapter(info));
        expect(summary.features).toEqual([...adapter.features].sort());
        expect(summary.limits.maxComputeWorkgroupsPerDimension).toBe(adapter.limits.maxComputeWorkgroupsPerDimension);
        const ctx = await GpuContext.create({ adapter, runtime: "node", label: "from-probe" });
        try {
            expect(ctx.state).toBe("ready");
            expect(ctx.caps.vendor).toBe(info.vendor);
            expect(ctx.caps.software).toBe(summary.software);
        } finally {
            ctx.dispose();
        }
    });

    it("gpu undefined -> { ok: false, code: E_NO_WEBGPU } without throwing", async () => {
        const result = await GpuContext.probe({ gpu: undefined });
        expect(result.ok).toBe(false);
        expect(result.code).toBe("E_NO_WEBGPU");
        expect(typeof result.reason).toBe("string");
        expect(result.adapter).toBeNull();
        expect(result.summary).toBeNull();
    });

    it("a gpu whose requestAdapter returns null or throws -> E_NO_ADAPTER with the reason", async () => {
        const nullGpu = { requestAdapter: () => Promise.resolve(null) } as unknown as GPU;
        const nothing = await GpuContext.probe({ gpu: nullGpu });
        expect(nothing.ok).toBe(false);
        expect(nothing.code).toBe("E_NO_ADAPTER");
        expect(nothing.reason).toContain("null");
        expect(nothing.adapter).toBeNull();
        const throwingGpu = {
            requestAdapter: () => Promise.reject(new Error("no suitable backends found")),
        } as unknown as GPU;
        const threw = await GpuContext.probe({ gpu: throwingGpu });
        expect(threw.ok).toBe(false);
        expect(threw.code).toBe("E_NO_ADAPTER");
        expect(threw.reason).toContain("no suitable backends found");
    });

    it("rejectSoftware: E_SOFTWARE_ONLY on a software adapter (adapter and summary still reported), OK on hardware", async (t) => {
        requireGpu(t);
        const { gpu, info } = await acquireRaw();
        const result = await GpuContext.probe({ gpu, rejectSoftware: true });
        if (isSoftwareAdapter(info)) {
            expect(result.ok).toBe(false);
            expect(result.code).toBe("E_SOFTWARE_ONLY");
            expect(result.reason).toContain(info.architecture);
            expect(result.adapter).not.toBeNull();
            expect(result.summary?.software).toBe(true);
        } else {
            expect(result.ok).toBe(true);
            expect(result.code).toBe("OK");
            expect(result.summary?.software).toBe(false);
        }
    });
});

describe("GpuContext.create / from", () => {
    it("with neither gpu, adapter nor device -> E_NO_WEBGPU { reason, hint }", async () => {
        const error = expectCode(await rejection(GpuContext.create({})), "E_NO_WEBGPU");
        expect(typeof error.details.reason).toBe("string");
        expect(typeof error.details.hint).toBe("string");
    });

    it("create({ gpu }) requests its own adapter and owns the device; dispose() destroys it and resolves ctx.lost", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const ctx = await GpuContext.create({ gpu, runtime: "node", label: "owned" });
        expect(ctx.ownsDevice).toBe(true);
        expect(ctx.state).toBe("ready");
        expect(ctx.label).toBe("owned");
        expect(ctx.caps.runtime).toBe("node");
        // P2-T1: the profiler follows the granted feature (contract 3.5: non-null when "timestamp-query" was granted)
        expect(ctx.profiler === null).toBe(!ctx.caps.features.has("timestamp-query"));
        if (ctx.profiler !== null) {
            expect(ctx.profiler.enabled).toBe(true);
            expect(ctx.profiler.quantised).toBe(false);
        }
        expect(ctx.workgroupSize).toBe(workgroupSizeFor(ctx.caps));
        expect(ctx.debug).toEqual({ inspect: false });
        expect(ctx.nextBatchId()).toBe(1);
        expect(ctx.nextBatchId()).toBe(2);
        ctx.assertReady();
        ctx.dispose();
        expect(ctx.state).toBe("disposed");
        const info = await ctx.lost;
        expect(info.reason).toBe("destroyed");
        expect(ctx.state).toBe("disposed");
        ctx.dispose();
        expectCode(
            thrown(() => {
                ctx.assertReady();
            }),
            "E_DISPOSED",
            { label: "owned" },
        );
    });

    it("create({ device }) adopts: ownsDevice false, runtime unknown, caps from the device; dispose() leaves the device alive", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const ctx = await GpuContext.create({ device, label: "adopted" });
        expect(ctx.ownsDevice).toBe(false);
        expect(ctx.caps.runtime).toBe("unknown");
        expect(ctx.caps.software).toBe(false);
        expect(ctx.caps.limits.maxBufferSize).toBe(device.limits.maxBufferSize);
        ctx.dispose();
        const alive = device.createBuffer({ label: "still-alive", size: 16, usage: BufferUsage.COPY_DST });
        device.queue.writeBuffer(alive, 0, new Uint32Array([1, 2, 3, 4]));
        await device.queue.onSubmittedWorkDone();
        alive.destroy();
    });

    it("from(device, info?) is the zero-cost adoption: runtime unknown and software false unless the info says otherwise", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const plain = GpuContext.from(device);
        expect(plain.ownsDevice).toBe(false);
        expect(plain.caps.runtime).toBe("unknown");
        expect(plain.caps.software).toBe(false);
        expect(plain.label.length).toBeGreaterThan(0);
        expect(plain.state).toBe("ready");
        plain.dispose();
        const tagged = GpuContext.from(device, { runtime: "browser", software: true, vendor: "v" });
        expect(tagged.caps.runtime).toBe("browser");
        expect(tagged.caps.software).toBe(true);
        expect(tagged.caps.vendor).toBe("v");
        tagged.dispose();
    });

    it("limits raise: every raisable limit >= the spec default and <= the adapter's; limits default: the spec defaults from device.limits, never the adapter's", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const probe = await GpuContext.probe({ gpu });
        const adapter = nonNull(probe.adapter);
        const summary = nonNull(probe.summary);
        const raised = await GpuContext.create({ adapter, limits: "raise", runtime: "node" });
        try {
            for (const name of RAISABLE_LIMITS) {
                expect(raised.caps.limits[name], name).toBeGreaterThanOrEqual(SPEC_DEFAULT_LIMITS[name]);
                expect(raised.caps.limits[name], name).toBeLessThanOrEqual(summary.limits[name]);
            }
            expect(raised.caps.limits.maxStorageBuffersPerShaderStage).toBe(
                summary.limits.maxStorageBuffersPerShaderStage,
            );
            expect(raised.caps.limits.maxComputeWorkgroupsPerDimension).toBe(65_535);
        } finally {
            raised.dispose();
        }
        const plain = await GpuContext.create({ gpu, limits: "default", runtime: "node" });
        try {
            expect(plain.caps.limits.maxBufferSize).toBe(268_435_456);
            expect(plain.caps.limits.maxStorageBufferBindingSize).toBe(134_217_728);
            expect(plain.caps.limits.maxStorageBuffersPerShaderStage).toBe(8);
            expect(plain.caps.limits.maxBufferSize).toBe(plain.device.limits.maxBufferSize);
        } finally {
            plain.dispose();
        }
    });

    it("an explicit limit above the adapter -> E_NO_DEVICE { reason: limit } before requestDevice: the adapter stays unused", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const probe = await GpuContext.probe({ gpu });
        const adapter = nonNull(probe.adapter);
        const available = nonNull(probe.summary).limits.maxStorageBuffersPerShaderStage;
        expectCode(
            await rejection(GpuContext.create({ adapter, limits: { maxStorageBuffersPerShaderStage: available + 1 } })),
            "E_NO_DEVICE",
            { reason: "limit", limit: "maxStorageBuffersPerShaderStage", requested: available + 1, available },
        );
        const ctx = await GpuContext.create({ adapter, limits: "default" });
        try {
            expect(ctx.state).toBe("ready");
        } finally {
            ctx.dispose();
        }
    });

    it("a consumed adapter -> E_NO_DEVICE { reason: consumed }", async (t) => {
        requireGpu(t);
        const { adapter } = await acquireRaw();
        const error = expectCode(await rejection(GpuContext.create({ adapter })), "E_NO_DEVICE", {
            reason: "consumed",
        });
        expect(error.details.adapter).toEqual(summarizeAdapter(adapter));
    });

    it("a missing required feature -> E_NO_DEVICE { reason: feature }; optionalFeatures [] -> no subgroups and zero subgroup sizes", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        expectCode(
            await rejection(
                GpuContext.create({ gpu, requiredFeatures: ["graphty-nonexistent-feature" as GPUFeatureName] }),
            ),
            "E_NO_DEVICE",
            { reason: "feature" },
        );
        const bare = await acquire({ subgroups: false });
        expect(bare.caps.features.has("subgroups")).toBe(false);
        expect(bare.caps.features.has("timestamp-query")).toBe(false);
        expect(bare.caps.subgroupMinSize).toBe(0);
        expect(bare.caps.subgroupMaxSize).toBe(0);
        // acquire()'s default follows the run's policy: GRAPHTY_GPU_NO_SUBGROUPS=1 drops subgroups from the
        // optional features for the whole run (spec 11.2; the GPU lane's second pass), so the full context has
        // them only when the adapter offers them AND the run did not opt out.
        const full = await acquire();
        const summary = nonNull((await GpuContext.probe({ gpu })).summary);
        const wantSubgroups = summary.features.includes("subgroups") && process.env.GRAPHTY_GPU_NO_SUBGROUPS !== "1";
        expect(full.caps.features.has("subgroups")).toBe(wantSubgroups);
        if (wantSubgroups) {
            expect(full.caps.subgroupMinSize).toBe(summary.subgroupMinSize);
            expect(full.caps.subgroupMaxSize).toBe(summary.subgroupMaxSize);
        }
    });

    it("rejectSoftware: E_SOFTWARE_ONLY { adapter } before any device on a software adapter; a hardware adapter passes", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        if (isSoftware()) {
            const error = expectCode(
                await rejection(GpuContext.create({ gpu, rejectSoftware: true })),
                "E_SOFTWARE_ONLY",
            );
            expect((error.details.adapter as { software: boolean }).software).toBe(true);
        } else {
            const ctx = await GpuContext.create({ gpu, rejectSoftware: true });
            try {
                expect(ctx.caps.software).toBe(false);
            } finally {
                ctx.dispose();
            }
        }
    });

    it("acquire() tags runtime node, sets debug.inspect from GRAPHTY_GPU_INSPECT; acquireNullBackend() gives a ready context on Dawn's null backend", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "via-setup" });
        expect(ctx.caps.runtime).toBe("node");
        expect(ctx.label).toBe("via-setup");
        expect(ctx.ownsDevice).toBe(true);
        expect(ctx.debug.inspect).toBe(process.env.GRAPHTY_GPU_INSPECT === "1");
        const nul = await acquireNullBackend();
        expect(nul.state).toBe("ready");
        expect(nul.caps.runtime).toBe("node");
        nul.assertReady();
        expect(nul.device).not.toBe(ctx.device);
    });
});

describe("GpuContext: uncaptured errors, loss, disposers, release", () => {
    it("routes an uncaptured error from a deliberately broken bind group to onError; nothing reaches the pending slot or the setup's hook", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const seen: WebGpuGraphError[] = [];
        const ctx = await GpuContext.create({
            gpu,
            runtime: "node",
            label: "sink-ctx",
            onError: (error) => {
                seen.push(error);
            },
        });
        try {
            brokenBindGroup(ctx.device, "sink-ctx/broken");
            await ctx.device.queue.onSubmittedWorkDone();
            expect(seen).toHaveLength(1);
            expect(seen[0].code).toBe("E_VALIDATION");
            expect(seen[0].details.label).toBe(ctx.device.label ?? "");
            expect(String(seen[0].details.message)).toMatch(/1024|binding/i);
            expect(ctx.takePendingError()).toBeNull();
            ctx.assertReady();
            expect(uncapturedErrors()).toEqual([]);
        } finally {
            ctx.dispose();
        }
    });

    it("without onError the error waits in the pending slot: assertReady() throws it once; takePendingError() drains", async (t) => {
        requireGpu(t);
        const { gpu } = await acquireRaw();
        const ctx = await GpuContext.create({ gpu, runtime: "node", label: "slot-ctx" });
        try {
            brokenBindGroup(ctx.device, "slot-ctx/broken");
            await ctx.device.queue.onSubmittedWorkDone();
            const error = expectCode(
                thrown(() => {
                    ctx.assertReady();
                }),
                "E_VALIDATION",
            );
            expect(String(error.details.message)).toMatch(/1024|binding/i);
            ctx.assertReady();
            expect(ctx.takePendingError()).toBeNull();
            brokenBindGroup(ctx.device, "slot-ctx/broken-2");
            await ctx.device.queue.onSubmittedWorkDone();
            expect(ctx.takePendingError()?.code).toBe("E_VALIDATION");
            expect(ctx.takePendingError()).toBeNull();
        } finally {
            ctx.dispose();
        }
    });

    // The one test that exercises the setup's failure path (spec 11.2: "an uncapturederror listener fails the
    // current test"; spec 11.5). Vitest 3.2.7 runs the afterEach hooks BEFORE it flips an `it.fails` result
    // (@vitest/runner chunk-hooks.js: callSuiteHook(..., "afterEach") precedes the `if (test.fails)` flip), so the
    // hook's throw is what makes this test pass; a setup whose hook never throws, or whose acquire() forgot the
    // onError sink, turns it RED with "Expect test to fail". A skipped test stays skipped (the runner returns
    // before the flip), so requireGpu(t) keeps its policy semantics here.
    it.fails(
        "the setup's afterEach hook fails a test whose acquire()d context delivered an uncaptured error",
        async (t) => {
            requireGpu(t);
            const ctx = await acquire({ label: "hook" });
            brokenBindGroup(ctx.device, "hook/broken");
            await ctx.device.queue.onSubmittedWorkDone();
            expect(uncapturedErrors()).toHaveLength(1);
            expect(uncapturedErrors()[0].code).toBe("E_VALIDATION");
            expect(ctx.takePendingError()).toBeNull();
        },
    );

    it("device loss: state lost, residency cleared, every onLost listener runs once, assertReady throws E_DEVICE_LOST, dispose still works", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "loss" });
        const calls: string[] = [];
        const offA = ctx.onLost((info) => {
            calls.push(`a:${info.reason}`);
        });
        const offB = ctx.onLost(() => {
            calls.push("b");
        });
        ctx.onLost(() => {
            throw new Error("a throwing listener must not stop the fan-out");
        });
        const offD = ctx.onLost(() => {
            calls.push("d");
        });
        offB();
        ctx.device.destroy();
        const info = await ctx.lost;
        expect(info.reason).toBe("destroyed");
        expect(ctx.state).toBe("lost");
        expect(calls).toEqual(["a:destroyed", "d"]);
        expect(ctx.residency.stats().buffers).toBe(0);
        const error = expectCode(
            thrown(() => {
                ctx.assertReady();
            }),
            "E_DEVICE_LOST",
            { reason: "destroyed" },
        );
        expect(typeof error.details.message).toBe("string");
        offA();
        offD();
        ctx.dispose();
        expect(ctx.state).toBe("disposed");
        expectCode(
            thrown(() => {
                ctx.assertReady();
            }),
            "E_DISPOSED",
        );
        expect(uncapturedErrors()).toEqual([]);
    });

    it("attachDisposer: runs once at dispose(), immediately when attached after dispose()", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "disposers" });
        let runs = 0;
        ctx.attachDisposer(() => {
            runs += 1;
        });
        expect(runs).toBe(0);
        ctx.dispose();
        expect(runs).toBe(1);
        ctx.dispose();
        expect(runs).toBe(1);
        ctx.attachDisposer(() => {
            runs += 10;
        });
        expect(runs).toBe(11);
    });

    it("dispose() destroys the residency, the pool and the staging ring: the allocator reports 0 live buffers", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "dispose-all" });
        const s = smallGraph();
        ctx.residency.core(s);
        const scratch = ctx.pool.acquire(4096, BufferUsage.STORAGE | BufferUsage.COPY_SRC, "dispose-all/scratch");
        await ctx.readback.read(scratch, 4096);
        ctx.pool.release(scratch);
        expect(ctx.allocator.liveBuffers).toBeGreaterThan(0);
        ctx.dispose();
        expect(ctx.allocator.liveBuffers).toBe(0);
        expect(ctx.allocator.resident).toBe(0);
        expect(ctx.residency.stats().buffers).toBe(0);
        expect(uncapturedErrors()).toEqual([]);
    });

    it("release(snapshot) destroys that snapshot's buffers and trims the pool; safe on an unknown snapshot, idempotent, a no-op after dispose", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "release" });
        const s = smallGraph();
        ctx.release(s);
        ctx.residency.core(s);
        expect(ctx.residency.stats().buffers).toBeGreaterThan(0);
        const scratch = ctx.pool.acquire(4096, BufferUsage.STORAGE, "release/scratch");
        ctx.pool.release(scratch);
        expect(ctx.pool.idleBytes).toBeGreaterThan(0);
        ctx.release(s);
        expect(ctx.residency.stats().buffers).toBe(0);
        expect(ctx.pool.idleBytes).toBe(0);
        ctx.release(s);
        ctx.dispose();
        ctx.release(s);
        expect(uncapturedErrors()).toEqual([]);
    });
});
