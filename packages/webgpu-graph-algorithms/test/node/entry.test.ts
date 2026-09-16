/**
 * The ./node entry (contract 3.7, 5.5; spec 2.3): createNodeGpu / createNodeGpuContext / probeNodeWebGpu on
 * the real Dawn module of the current lane, dawnFlags as a pure mapping, and the loadModule seam for a
 * missing / broken module. The handle drop of NodeGpuHandle.dispose() is observable only as the `gpu`
 * getter throwing E_DISPOSED; that it lets the process exit (spec 2.3) is documented, not asserted.
 */

import { GpuContext } from "../../src/context.js";
import { isWebGpuGraphError, type WebGpuGraphError, type WebGpuGraphErrorCode } from "../../src/errors.js";
import { createNodeGpu, createNodeGpuContext, dawnFlags, probeNodeWebGpu } from "../../src/node/index.js";
import { isSoftware, requireGpu } from "../setup/gpu.js";

/** The lane's Dawn selection, as the setup reads it. */
function laneOptions(): { adapter: string | undefined; dawnFeatures: string[] | undefined } {
    const adapter = process.env.GRAPHTY_GPU_ADAPTER;
    const features = process.env.GRAPHTY_DAWN_FEATURES;
    return {
        adapter: adapter === undefined || adapter === "" ? undefined : adapter,
        dawnFeatures: features === undefined || features === "" ? undefined : features.split(","),
    };
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

function expectCode(err: unknown, code: WebGpuGraphErrorCode, details?: Record<string, unknown>): WebGpuGraphError {
    expect(isWebGpuGraphError(err), `expected a WebGpuGraphError, got ${String(err)}`).toBe(true);
    const error = err as WebGpuGraphError;
    expect(error.code).toBe(code);
    if (details !== undefined) {
        expect(error.details).toMatchObject(details);
    }
    return error;
}

const SENTINEL = "__graphtyEntryTestSentinel";

describe("dawnFlags", () => {
    it("maps every NodeGpuOptions field to its Dawn flag, in the order adapter, backend, enable-dawn-features", () => {
        expect(dawnFlags(undefined)).toEqual([]);
        expect(dawnFlags({})).toEqual([]);
        expect(dawnFlags({ adapter: "llvmpipe" })).toEqual(["adapter=llvmpipe"]);
        expect(dawnFlags({ backend: "null" })).toEqual(["backend=null"]);
        expect(dawnFlags({ dawnFeatures: ["allow_unsafe_apis", "dump_shaders"] })).toEqual([
            "enable-dawn-features=allow_unsafe_apis,dump_shaders",
        ]);
        expect(dawnFlags({ dawnFeatures: [] })).toEqual([]);
        expect(dawnFlags({ software: true })).toEqual(["adapter=llvmpipe"]);
        expect(dawnFlags({ software: false })).toEqual([]);
        expect(dawnFlags({ software: true, adapter: "4070" })).toEqual(["adapter=4070"]);
        expect(dawnFlags({ adapter: "", backend: undefined })).toEqual([]);
        expect(dawnFlags({ adapter: "llvmpipe", backend: "vulkan", dawnFeatures: ["a"] })).toEqual([
            "adapter=llvmpipe",
            "backend=vulkan",
            "enable-dawn-features=a",
        ]);
        expect(dawnFlags({ label: "ignored", installGlobals: false })).toEqual([]);
    });
});

describe("createNodeGpu", () => {
    it("loads Dawn, installs the globals and returns a handle whose dispose() drops the GPU reference", async (t) => {
        requireGpu(t);
        const handle = await createNodeGpu(laneOptions());
        expect(typeof handle.gpu.requestAdapter).toBe("function");
        // Dawn-node 0.4.0 installs GPUBufferUsage as a FUNCTION object carrying the constants (test/device/constants.test.ts)
        expect(["object", "function"]).toContain(
            typeof (globalThis as unknown as { GPUBufferUsage?: unknown }).GPUBufferUsage,
        );
        const adapter = await handle.gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        handle.dispose();
        expectCode(
            thrown(() => handle.gpu),
            "E_DISPOSED",
            { label: "NodeGpuHandle" },
        );
        handle.dispose();
    });

    it("passes dawnFlags(options) to create() and honours installGlobals (fake module through the loadModule seam)", async () => {
        const created: string[][] = [];
        const fakeGpu = { requestAdapter: () => Promise.resolve(null) } as unknown as GPU;
        const module = {
            create: (flags: string[]): GPU => {
                created.push([...flags]);
                return fakeGpu;
            },
            globals: { [SENTINEL]: 42 },
        };
        const quiet = await createNodeGpu({
            adapter: "fake",
            backend: "null",
            installGlobals: false,
            loadModule: () => Promise.resolve(module),
        });
        expect(created).toEqual([["adapter=fake", "backend=null"]]);
        expect(quiet.gpu).toBe(fakeGpu);
        expect(SENTINEL in globalThis).toBe(false);
        const installing = await createNodeGpu({ loadModule: () => Promise.resolve(module) });
        expect(created).toEqual([["adapter=fake", "backend=null"], []]);
        expect((globalThis as unknown as Record<string, unknown>)[SENTINEL]).toBe(42);
        Reflect.deleteProperty(globalThis, SENTINEL);
        quiet.dispose();
        installing.dispose();
    });

    it("a loader that rejects -> E_NO_WEBGPU { reason, hint } with the install hint; a module without create() -> E_NO_WEBGPU", async () => {
        const failing = expectCode(
            await rejection(
                createNodeGpu({ loadModule: () => Promise.reject(new Error("Cannot find module 'webgpu'")) }),
            ),
            "E_NO_WEBGPU",
            { hint: "install the optional peer dependency webgpu@0.4.0" },
        );
        expect(String(failing.details.reason)).toContain("Cannot find module 'webgpu'");
        expect(failing.message).toContain("Cannot find module 'webgpu'");
        const shapeless = expectCode(
            await rejection(createNodeGpu({ loadModule: () => Promise.resolve({ globals: {} }) })),
            "E_NO_WEBGPU",
            { hint: "install the optional peer dependency webgpu@0.4.0" },
        );
        expect(String(shapeless.details.reason)).toContain("create");
        const throwingCreate = expectCode(
            await rejection(
                createNodeGpu({
                    loadModule: () =>
                        Promise.resolve({
                            create: (): GPU => {
                                throw new Error("unrecognised backend 'bogus'");
                            },
                            globals: {},
                        }),
                }),
            ),
            "E_NO_WEBGPU",
            { hint: "install the optional peer dependency webgpu@0.4.0" },
        );
        expect(String(throwingCreate.details.reason)).toContain("unrecognised backend 'bogus'");
        expect(throwingCreate.message).toContain("install the optional peer dependency webgpu@0.4.0");
    });
});

describe("probeNodeWebGpu", () => {
    it("never throws: a broken loader is { ok: false, code: E_NO_WEBGPU, reason }", async () => {
        const result = await probeNodeWebGpu({ loadModule: () => Promise.reject(new Error("no native module")) });
        expect(result.ok).toBe(false);
        expect(result.code).toBe("E_NO_WEBGPU");
        expect(result.reason).toContain("no native module");
        expect(result.adapter).toBeNull();
        expect(result.summary).toBeNull();
    });

    it("reports the adapter of the current lane with its software flag", async (t) => {
        requireGpu(t);
        const result = await probeNodeWebGpu(laneOptions());
        expect(result.ok).toBe(true);
        expect(result.code).toBe("OK");
        expect(result.adapter).not.toBeNull();
        expect(result.summary?.software).toBe(isSoftware());
        expect(Array.isArray(result.summary?.features)).toBe(true);
    });
});

describe("createNodeGpuContext", () => {
    it("creates a context tagged runtime node from the lane's adapter; dispose() disposes the context and the handle", async (t) => {
        requireGpu(t);
        const ctx = await createNodeGpuContext({ ...laneOptions(), label: "entry-test" });
        expect(ctx).toBeInstanceOf(GpuContext);
        expect(ctx.caps.runtime).toBe("node");
        expect(ctx.ownsDevice).toBe(true);
        expect(ctx.label).toBe("entry-test");
        expect(ctx.caps.software).toBe(isSoftware());
        ctx.assertReady();
        ctx.dispose();
        expect(ctx.state).toBe("disposed");
        ctx.dispose();
        // the handle drop is unobservable from here (the handle is internal); spec 2.3 says it lets the process exit
    });

    it("propagates create() failures and drops the handle: rejectSoftware on a software adapter -> E_SOFTWARE_ONLY, OK on hardware", async (t) => {
        requireGpu(t);
        if (isSoftware()) {
            const error = expectCode(
                await rejection(createNodeGpuContext({ ...laneOptions(), rejectSoftware: true })),
                "E_SOFTWARE_ONLY",
            );
            expect((error.details.adapter as { software: boolean }).software).toBe(true);
        } else {
            const ctx = await createNodeGpuContext({ ...laneOptions(), rejectSoftware: true });
            try {
                expect(ctx.caps.software).toBe(false);
            } finally {
                ctx.dispose();
            }
        }
    });
});
