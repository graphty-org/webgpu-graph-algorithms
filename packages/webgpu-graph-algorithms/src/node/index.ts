/// <reference types="@webgpu/types" preserve="true" />
/**
 * The ./node entry (spec 2.1, 2.3, 3.4; contract 3.7): Dawn through the optional peer dependency `webgpu`. The
 * ONLY file of the package that names the "webgpu" module, and only inside `await import("webgpu")` in a
 * function body, so the root and browser bundles never carry the specifier (spec 2.5; test/build-output.test.ts,
 * test/layers.test.ts). P0-T3 wrote createNodeGpu, dawnFlags and their types; P1-T1 adds createNodeGpuContext,
 * probeNodeWebGpu and the `extends Omit<GpuContextOptions, ...>` clause of NodeGpuOptions.
 *
 * Rule of spec 2.5 item 5 (also in CLAUDE.md): "./node" is imported only by Node entry points and tests.
 */

import { GpuContext } from "../context.js";
import { WebGpuGraphError } from "../errors.js";
import type { GpuContextOptions, ProbeResult } from "../types/context.js";

/** Options of the Node helpers (spec 3.4). */
export interface NodeGpuOptions extends Omit<GpuContextOptions, "gpu" | "adapter" | "device" | "runtime"> {
    /** Dawn `adapter=<substring>`: a substring of the adapter name (`llvmpipe`, `4070`); an empty string is ignored; wins over `software`. */
    readonly adapter?: string | undefined;
    /** Dawn `backend=<name>`; `null` compiles pipelines and runs nothing (spec 5.1; the compile tests use it). */
    readonly backend?: "vulkan" | "d3d12" | "d3d11" | "metal" | "opengl" | "opengles" | "null" | undefined;
    /** Dawn toggles, emitted as `enable-dawn-features=a,b`. */
    readonly dawnFeatures?: readonly string[] | undefined;
    /** Shorthand for `adapter=llvmpipe` (Linux / Mesa specific, spec 2.3); ignored when `adapter` is given. */
    readonly software?: boolean | undefined;
    /** Install dawn.globals on globalThis (default true; spec 2.1 rule 2) so user code sees GPUBufferUsage and friends. */
    readonly installGlobals?: boolean | undefined;
    /**
     * Test seam: replaces `() => import("webgpu")` so a missing / broken module can be simulated.
     * @internal
     */
    readonly loadModule?: (() => Promise<unknown>) | undefined;
}

/** The Dawn GPU handle (spec 2.3): dispose() drops the reference so the process can exit. */
export interface NodeGpuHandle {
    /** The GPU of `dawn.create(flags)`; reading it after dispose() throws E_DISPOSED. */
    readonly gpu: GPU;
    /** Drops the GPU reference; idempotent. */
    dispose(): void;
}

/** The install hint of E_NO_WEBGPU (spec 2.5 item 4); P-ENV re-pins the version together with the devDependency. */
const INSTALL_HINT = "install the optional peer dependency webgpu@0.4.0";

/** The shape of the `webgpu` module (its types.d.ts: create(options: string[]): GPU; globals: Object). */
interface DawnModule {
    create(options: string[]): GPU;
    globals?: unknown;
}

/**
 * Whether a loaded module is usable as Dawn.
 * @param loaded - the module namespace
 * @returns true when it has a create() function
 */
function isDawnModule(loaded: unknown): loaded is DawnModule {
    return (
        typeof loaded === "object" && loaded !== null && typeof (loaded as { create?: unknown }).create === "function"
    );
}

/**
 * The message of a thrown value.
 * @param err - what was caught
 * @returns the Error message, or the value as a string
 */
function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

/** The handle returned by createNodeGpu(): `gpu` is a getter so dispose() can drop the reference. */
class DawnHandle implements NodeGpuHandle {
    private ref: GPU | null;

    /**
     * Wraps a GPU object.
     * @param gpu - the object dawn.create() returned
     */
    constructor(gpu: GPU) {
        this.ref = gpu;
    }

    /**
     * The GPU object; E_DISPOSED after dispose().
     * @returns the GPU object
     */
    get gpu(): GPU {
        if (this.ref === null) {
            throw new WebGpuGraphError("E_DISPOSED", "the Dawn GPU handle was disposed", { label: "NodeGpuHandle" });
        }
        return this.ref;
    }

    /** Drops the reference; idempotent. */
    dispose(): void {
        this.ref = null;
    }
}

/**
 * The Dawn flag list a NodeGpuOptions maps to (exported for the tests and scripts/gpu-report.js):
 * adapter=<s>, backend=<s>, enable-dawn-features=a,b, and software -> adapter=llvmpipe; an explicit non-empty
 * `adapter` wins over `software`; an empty feature list emits nothing.
 * @param options - the Node options, or undefined for no flags
 * @returns the strings for dawn.create(), in the order adapter, backend, features
 */
export function dawnFlags(options: NodeGpuOptions | undefined): string[] {
    const flags: string[] = [];
    if (options === undefined) {
        return flags;
    }
    if (options.adapter !== undefined && options.adapter !== "") {
        flags.push(`adapter=${options.adapter}`);
    } else if (options.software === true) {
        flags.push("adapter=llvmpipe");
    }
    if (options.backend !== undefined) {
        flags.push(`backend=${options.backend}`);
    }
    if (options.dawnFeatures !== undefined && options.dawnFeatures.length > 0) {
        flags.push(`enable-dawn-features=${options.dawnFeatures.join(",")}`);
    }
    return flags;
}

/**
 * import("webgpu"), install dawn.globals unless installGlobals === false, dawn.create(flags) (spec 2.3).
 * @param options - adapter / backend / dawnFeatures / software / installGlobals (and the test seam)
 * @returns the handle; `dispose()` drops the GPU reference so the process can exit
 * @throws WebGpuGraphError E_NO_WEBGPU { reason, hint } when the module does not load (missing, or its glibc is too old), has no create(), or create(flags) throws
 */
export async function createNodeGpu(options?: NodeGpuOptions): Promise<NodeGpuHandle> {
    const load = options?.loadModule ?? ((): Promise<unknown> => import("webgpu"));
    let loaded: unknown;
    try {
        loaded = await load();
    } catch (err) {
        const reason = messageOf(err);
        throw new WebGpuGraphError(
            "E_NO_WEBGPU",
            `the webgpu (Dawn) native module did not load: ${reason}; ${INSTALL_HINT}`,
            { reason, hint: INSTALL_HINT },
        );
    }
    if (!isDawnModule(loaded)) {
        const reason = "the webgpu module exports no create() function";
        throw new WebGpuGraphError("E_NO_WEBGPU", `${reason}; ${INSTALL_HINT}`, { reason, hint: INSTALL_HINT });
    }
    if (options?.installGlobals !== false && typeof loaded.globals === "object" && loaded.globals !== null) {
        Object.assign(globalThis, loaded.globals);
    }
    let gpu: GPU;
    try {
        gpu = loaded.create(dawnFlags(options));
    } catch (err) {
        const reason = `dawn.create() threw: ${messageOf(err)}`;
        throw new WebGpuGraphError("E_NO_WEBGPU", `${reason}; ${INSTALL_HINT}`, { reason, hint: INSTALL_HINT });
    }
    return new DawnHandle(gpu);
}

/**
 * The GpuContextOptions part of a NodeGpuOptions, copied key by key so the Node-only `adapter` string never
 * reaches GpuContext.create as a GPUAdapter.
 * @param options - the Node options
 * @returns the context options without gpu / adapter / device / runtime
 */
function contextOptionsOf(options: NodeGpuOptions): Omit<GpuContextOptions, "gpu" | "adapter" | "device" | "runtime"> {
    return {
        powerPreference: options.powerPreference,
        rejectSoftware: options.rejectSoftware,
        limits: options.limits,
        optionalFeatures: options.optionalFeatures,
        requiredFeatures: options.requiredFeatures,
        label: options.label,
        onError: options.onError,
        warnUnreleasedSnapshots: options.warnUnreleasedSnapshots,
    };
}

/**
 * createNodeGpu + GpuContext.create({ gpu, runtime: "node", ...options }); ctx.dispose() also disposes the
 * handle, and a create() failure disposes it before rethrowing. The handle is dropped once `ctx.lost` has
 * settled, never before: under webgpu@0.4.0 a GPU object collected while the device it created is still
 * tearing down (its lost / work-done callbacks in flight) crashes or deadlocks the process (PLAN DECISION,
 * P1-T1; measured with tmp/p1t1/gc-race2.mjs), and device.destroy() reports the loss right away.
 * @param options - the Node options
 * @returns the context
 */
export async function createNodeGpuContext(options?: NodeGpuOptions): Promise<GpuContext> {
    const handle = await createNodeGpu(options);
    let ctx: GpuContext;
    try {
        ctx = await GpuContext.create({ ...contextOptionsOf(options ?? {}), gpu: handle.gpu, runtime: "node" });
    } catch (err) {
        handle.dispose();
        throw err;
    }
    const { lost } = ctx;
    ctx.attachDisposer(() => {
        void lost.then(() => {
            handle.dispose();
        });
    });
    return ctx;
}

/**
 * createNodeGpu + GpuContext.probe + dispose; never throws (a load failure is { code: "E_NO_WEBGPU" }).
 * @param options - the Node options
 * @returns the probe result
 */
export async function probeNodeWebGpu(options?: NodeGpuOptions): Promise<ProbeResult> {
    let handle: NodeGpuHandle;
    try {
        handle = await createNodeGpu(options);
    } catch (err) {
        return { ok: false, code: "E_NO_WEBGPU", reason: messageOf(err), adapter: null, summary: null };
    }
    try {
        return await GpuContext.probe({
            gpu: handle.gpu,
            powerPreference: options?.powerPreference ?? "high-performance",
            rejectSoftware: options?.rejectSoftware,
        });
    } finally {
        handle.dispose();
    }
}
