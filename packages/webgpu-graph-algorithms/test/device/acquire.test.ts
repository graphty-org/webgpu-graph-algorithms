/**
 * The first device test of the node project (spec 13 row P0, 11.5 seed; contract 5.5): requireGpu, a fresh
 * adapter + device per acquireRaw, distinct adapters across calls, isSoftwareAdapter against the policy
 * expectation, the summarizeAdapter shape, the printed info and limits, one trivial compute dispatch run twice,
 * the uncapturederror sink of the setup (an `it.fails` test: the afterEach hook is what fails it, spec 11.2),
 * and (PLAN DECISION 6) the ./node entry's dawnFlags table and createNodeGpu failure paths through the
 * loadModule seam (no real module involved; the setup already loaded the real one).
 */

import { isSoftwareAdapter, summarizeAdapter } from "../../src/device/acquire.js";
import { BufferUsage, MapMode, ShaderStage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode, isWebGpuGraphError } from "../../src/errors.js";
import { createNodeGpu, dawnFlags, type NodeGpuOptions } from "../../src/node/index.js";
import { type AdapterInfoLike } from "../../src/types/context.js";
import {
    acquireRaw,
    adapterSummary,
    gpuPolicy,
    gpuScale,
    isSoftware,
    requireGpu,
    skipReason,
    uncapturedErrors,
} from "../setup/gpu.js";

const INSTALL_HINT = "install the optional peer dependency webgpu@0.4.0";

/** Runs `fn` and returns what it threw (null when it returned). */
function caught(fn: () => unknown): unknown {
    try {
        fn();
        return null;
    } catch (error) {
        return error;
    }
}

/** One workgroup per item; every item writes 42 + 2 * its index (the trivial device test of spec 13 row P0). */
const TRIVIAL_WGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = 42u + 2u * gid.x;
}
`;

/** Compiles TRIVIAL_WGSL, dispatches 4 workgroups and reads the 16 bytes back (copy BEFORE unmap: Dawn detaches the mapped range on unmap, spec 2.6). */
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

describe("test/setup/gpu.ts: the policy and the probe", () => {
    it("parses GRAPHTY_GPU_REQUIRE into the one policy shape", () => {
        const policy = gpuPolicy();
        expect(["skip", "any", "hardware", "vendor"]).toContain(policy.level);
        expect(policy.raw).toBe(process.env.GRAPHTY_GPU_REQUIRE ?? "");
        if (policy.level === "vendor") {
            expect(policy.vendor).toBe(policy.raw.toLowerCase());
        } else {
            expect(policy.vendor).toBeNull();
        }
    });

    it("either has a probe summary or a printed E_NO_ADAPTER reason, never both", () => {
        const summary = adapterSummary();
        const reason = skipReason();
        expect(summary === null).not.toBe(reason === null);
        if (reason !== null) {
            expect(reason.startsWith("E_NO_ADAPTER: ")).toBe(true);
        }
    });

    it("gpuScale is 1 on hardware and 1 / 50 on a software adapter (spec 11.2)", (t) => {
        requireGpu(t);
        expect(gpuScale()).toBe(isSoftware() ? 1 / 50 : 1);
        expect(isSoftware()).toBe(adapterSummary()?.software);
    });
});

describe("acquireRaw: a fresh adapter and device per call (spec 11.2, 2.2 step 1)", () => {
    it("gives an adapter, a device with raised limits, and the adapter info", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        expect(raw.adapter).toBeDefined();
        expect(raw.device).toBeDefined();
        expect(raw.info.vendor).toBe(raw.adapter.info.vendor);
        const { limits } = raw.device;
        console.warn(
            `[acquire] device limits maxBufferSize=${limits.maxBufferSize} maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize} maxStorageBuffersPerShaderStage=${limits.maxStorageBuffersPerShaderStage} maxComputeWorkgroupsPerDimension=${limits.maxComputeWorkgroupsPerDimension} maxComputeInvocationsPerWorkgroup=${limits.maxComputeInvocationsPerWorkgroup}`,
        );
        // raised = at least the spec defaults; the workgroup-per-dimension limit is the spec minimum on every adapter (spec 2.2)
        expect(limits.maxBufferSize).toBeGreaterThanOrEqual(268435456);
        expect(limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(134217728);
        expect(limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(8);
        expect(limits.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(256);
        expect(limits.maxComputeWorkgroupsPerDimension).toBeGreaterThanOrEqual(65535);
        expect(256 % limits.minStorageBufferOffsetAlignment).toBe(0);
    });

    it("two calls give distinct adapters and distinct devices", async (t) => {
        requireGpu(t);
        const a = await acquireRaw();
        const b = await acquireRaw();
        expect(a.adapter).not.toBe(b.adapter);
        expect(a.device).not.toBe(b.device);
        expect(a.gpu).toBe(b.gpu);
    });

    it("optionalFeatures [] gives a device without subgroups (the twin path of D16)", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw({ optionalFeatures: [] });
        expect(raw.device.features.has("subgroups")).toBe(false);
        expect(raw.device.features.has("timestamp-query")).toBe(false);
    });

    it("limits: default gives the spec-default limits; a record is passed as given (R-22 [M])", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw({ limits: "default" });
        expect(raw.device.limits.maxBufferSize).toBe(268435456);
        expect(raw.device.limits.maxStorageBufferBindingSize).toBe(134217728);
        const record = await acquireRaw({ limits: { maxStorageBuffersPerShaderStage: 8 } });
        expect(record.device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    });

    it("isSoftwareAdapter(info) matches the policy expectation", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        const software = isSoftwareAdapter(raw.info);
        expect(software).toBe(isSoftware());
        // Dawn-node 0.4.0 reports adapter.info.isFallbackAdapter as a boolean (true on llvmpipe, false on NVIDIA),
        // like Chromium; contract correction 5 (spec 2.6 measured the deprecated GPUAdapter attribute)
        expect(typeof raw.info.isFallbackAdapter).toBe("boolean");
        expect(raw.info.isFallbackAdapter).toBe(raw.info.architecture === "software");
        const policy = gpuPolicy();
        if (process.env.GRAPHTY_GPU_ADAPTER === "llvmpipe") {
            expect(software).toBe(true);
            expect(raw.info.architecture).toBe("software");
        }
        if (policy.level === "hardware" || policy.level === "vendor") {
            expect(software).toBe(false);
        }
        if (policy.level === "vendor") {
            expect(raw.info.vendor).toBe(policy.vendor);
        }
    });

    it("summarizeAdapter has the spec 2.2 shape, is frozen, and agrees with the probe summary", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        const summary = summarizeAdapter(raw.adapter);
        expect(typeof summary.vendor).toBe("string");
        expect(typeof summary.architecture).toBe("string");
        expect(typeof summary.device).toBe("string");
        expect(typeof summary.description).toBe("string");
        expect(summary.software).toBe(isSoftwareAdapter(raw.info));
        expect(summary.subgroupMinSize).toBeGreaterThanOrEqual(0);
        expect(summary.subgroupMaxSize).toBeGreaterThanOrEqual(summary.subgroupMinSize);
        expect(Array.isArray(summary.features)).toBe(true);
        expect([...summary.features]).toEqual([...summary.features].sort());
        for (const name of [
            "maxBufferSize",
            "maxStorageBufferBindingSize",
            "maxStorageBuffersPerShaderStage",
            "maxComputeWorkgroupsPerDimension",
        ]) {
            expect(Number.isInteger(summary.limits[name]), name).toBe(true);
            expect(summary.limits[name], name).toBeGreaterThan(0);
        }
        expect(Object.isFrozen(summary)).toBe(true);
        expect(Object.isFrozen(summary.features)).toBe(true);
        expect(Object.isFrozen(summary.limits)).toBe(true);
        const probe = adapterSummary();
        expect(probe).not.toBeNull();
        expect(summary.vendor).toBe(probe?.vendor);
        expect(summary.architecture).toBe(probe?.architecture);
        expect(summary.software).toBe(probe?.software);
        console.warn(
            `[acquire] adapter vendor=${summary.vendor} architecture=${summary.architecture} device=${summary.device} description=${summary.description} software=${summary.software} features=${summary.features.join(",")}`,
        );
    });

    it("runs the trivial compute dispatch twice with bitwise-identical results and no uncaptured error", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        const first = await runTrivialKernel(raw.device);
        const second = await runTrivialKernel(raw.device);
        expect([...first]).toEqual([42, 44, 46, 48]);
        expect([...second]).toEqual([...first]);
        await raw.device.queue.onSubmittedWorkDone();
        expect(uncapturedErrors()).toEqual([]);
    });

    // `it.fails`: the test body's assertions all pass, the setup's afterEach hook then throws on the non-empty
    // uncaptured list (spec 11.2: never a pass), and vitest attributes a hook error to the test, which `fails`
    // inverts. It therefore passes ONLY because the hook fires; a policy violation in requireGpu is inverted too,
    // so the red-run previews of this step count it as a pass.
    it.fails("the uncapturederror sink fails a test that leaves an uncaptured error (spec 11.2)", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        expect(uncapturedErrors()).toEqual([]);
        // a bind group whose 16-byte buffer is smaller than the layout's minBindingSize of 1024: a validation
        // error at createBindGroup, delivered as an uncapturederror event (synchronously under Dawn-node)
        const layout = raw.device.createBindGroupLayout({
            label: "p0-sink/layout",
            entries: [
                { binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "storage", minBindingSize: 1024 } },
            ],
        });
        const small = raw.device.createBuffer({ label: "p0-sink/small", size: 16, usage: BufferUsage.STORAGE });
        raw.device.createBindGroup({
            label: "p0-sink",
            layout,
            entries: [{ binding: 0, resource: { buffer: small } }],
        });
        await raw.device.queue.onSubmittedWorkDone();
        small.destroy();
        const errors = uncapturedErrors();
        expect(errors).toHaveLength(1);
        expect(errors[0].code).toBe("E_VALIDATION");
        expect(errors[0].details.label).toBe("acquireRaw");
        expect(String(errors[0].details.message)).toMatch(/1024|binding/i);
        expect(errors[0].message).toContain(String(errors[0].details.message));
    });
});

describe("isSoftwareAdapter (spec 2.2 step 2, 2.6)", () => {
    const base = { vendor: "nvidia", architecture: "lovelace", device: "", description: "" };
    const rows: readonly [string, AdapterInfoLike, boolean][] = [
        ["Dawn-node llvmpipe", { ...base, vendor: "mesa", architecture: "software" }, true],
        [
            "Chromium SwiftShader",
            { ...base, vendor: "google", architecture: "swiftshader", isFallbackAdapter: true },
            true,
        ],
        ["Dawn-node NVIDIA (isFallbackAdapter false)", { ...base, isFallbackAdapter: false }, false],
        ["Chromium NVIDIA", { ...base, device: "nvidia-geforce-rtx-4070-super", isFallbackAdapter: false }, false],
        ["a fake without the field", base, false],
        ["a hardware architecture flagged as fallback", { ...base, isFallbackAdapter: true }, true],
        ["Dawn null backend", { vendor: "", architecture: "", device: "null-backend", description: "" }, false],
    ];
    for (const [name, info, expected] of rows) {
        it(`${name} -> ${expected}`, () => {
            expect(isSoftwareAdapter(info)).toBe(expected);
        });
    }
});

describe("dawnFlags (spec 2.3; PLAN DECISION 3)", () => {
    const rows: readonly [string, NodeGpuOptions | undefined, string[]][] = [
        ["undefined", undefined, []],
        ["empty", {}, []],
        ["software", { software: true }, ["adapter=llvmpipe"]],
        ["software false", { software: false }, []],
        ["adapter wins over software", { adapter: "4070", software: true }, ["adapter=4070"]],
        ["empty adapter string is unset", { adapter: "", software: true }, ["adapter=llvmpipe"]],
        ["backend", { backend: "null" }, ["backend=null"]],
        ["no features", { dawnFeatures: [] }, []],
        ["one feature", { dawnFeatures: ["allow_unsafe_apis"] }, ["enable-dawn-features=allow_unsafe_apis"]],
        [
            "everything, in order",
            { adapter: "llvmpipe", backend: "vulkan", dawnFeatures: ["a", "b"], software: true, installGlobals: false },
            ["adapter=llvmpipe", "backend=vulkan", "enable-dawn-features=a,b"],
        ],
    ];
    for (const [name, options, expected] of rows) {
        it(name, () => {
            expect(dawnFlags(options)).toEqual(expected);
        });
    }
});

describe("createNodeGpu through the loadModule seam (contract 3.7)", () => {
    interface FakeDawn {
        readonly calls: string[][];
        readonly gpu: GPU;
        readonly create: (flags: string[]) => GPU;
        readonly globals: Record<string, unknown>;
    }

    function fakeDawn(globals: Record<string, unknown> = {}): FakeDawn {
        const calls: string[][] = [];
        const gpu = { requestAdapter: () => Promise.resolve(null) } as unknown as GPU;
        return {
            calls,
            gpu,
            globals,
            create: (flags: string[]): GPU => {
                calls.push(flags);
                return gpu;
            },
        };
    }

    it("a rejected import is E_NO_WEBGPU with the reason and the install hint", async () => {
        const failure = createNodeGpu({ loadModule: () => Promise.reject(new Error("boom")), installGlobals: false });
        await expect(failure).rejects.toSatisfy((error: unknown) => hasErrorCode(error, "E_NO_WEBGPU"));
        try {
            await failure;
        } catch (error) {
            expect(isWebGpuGraphError(error)).toBe(true);
            if (isWebGpuGraphError(error)) {
                expect(error.details.hint).toBe(INSTALL_HINT);
                expect(error.details.reason).toContain("boom");
                expect(error.message).toContain("boom");
                expect(error.message).toContain(INSTALL_HINT);
            }
        }
    });

    it("a module with create() but no globals is accepted and installs nothing", async () => {
        const fake = fakeDawn();
        const before = Object.keys(globalThis).length;
        const handle = await createNodeGpu({
            loadModule: () => Promise.resolve({ create: fake.create }),
            software: true,
        });
        expect(handle.gpu).toBe(fake.gpu);
        expect(fake.calls).toEqual([["adapter=llvmpipe"]]);
        expect(Object.keys(globalThis).length).toBe(before);
        handle.dispose();
    });

    it("a module without create() is E_NO_WEBGPU", async () => {
        const failure = createNodeGpu({ loadModule: () => Promise.resolve({ globals: {} }), installGlobals: false });
        await expect(failure).rejects.toSatisfy((error: unknown) => hasErrorCode(error, "E_NO_WEBGPU"));
        const nonObject = createNodeGpu({ loadModule: () => Promise.resolve("not a module"), installGlobals: false });
        await expect(nonObject).rejects.toSatisfy((error: unknown) => hasErrorCode(error, "E_NO_WEBGPU"));
    });

    it("passes dawnFlags(options) to create() and returns the created GPU; installGlobals false leaves globalThis alone", async () => {
        const fake = fakeDawn({ GRAPHTY_P0_FAKE_GLOBAL: 7 });
        const handle = await createNodeGpu({
            loadModule: () => Promise.resolve(fake),
            adapter: "llvmpipe",
            backend: "vulkan",
            dawnFeatures: ["a", "b"],
            installGlobals: false,
        });
        expect(fake.calls).toEqual([["adapter=llvmpipe", "backend=vulkan", "enable-dawn-features=a,b"]]);
        expect(handle.gpu).toBe(fake.gpu);
        expect((globalThis as Record<string, unknown>).GRAPHTY_P0_FAKE_GLOBAL).toBeUndefined();
        handle.dispose();
    });

    it("installs the module's globals by default", async () => {
        const fake = fakeDawn({ GRAPHTY_P0_FAKE_GLOBAL: 7 });
        const handle = await createNodeGpu({ loadModule: () => Promise.resolve(fake) });
        try {
            expect((globalThis as Record<string, unknown>).GRAPHTY_P0_FAKE_GLOBAL).toBe(7);
            expect(fake.calls).toEqual([[]]);
        } finally {
            Reflect.deleteProperty(globalThis, "GRAPHTY_P0_FAKE_GLOBAL");
            handle.dispose();
        }
    });

    it("dispose() drops the GPU reference: gpu then throws E_DISPOSED, and dispose is idempotent", async () => {
        const fake = fakeDawn();
        const handle = await createNodeGpu({ loadModule: () => Promise.resolve(fake), installGlobals: false });
        expect(handle.gpu).toBe(fake.gpu);
        handle.dispose();
        const thrown = caught(() => handle.gpu);
        expect(hasErrorCode(thrown, "E_DISPOSED")).toBe(true);
        expect(isWebGpuGraphError(thrown) ? thrown.details : null).toEqual({ label: "NodeGpuHandle" });
        handle.dispose();
        expect(
            hasErrorCode(
                caught(() => handle.gpu),
                "E_DISPOSED",
            ),
        ).toBe(true);
    });
});
