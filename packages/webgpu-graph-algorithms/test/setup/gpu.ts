/**
 * The Node projects' setup file AND the module every GPU test imports (one instance per worker; spec 11.2,
 * contract 5.1). beforeAll loads Dawn once through createNodeGpu (GRAPHTY_GPU_ADAPTER / GRAPHTY_DAWN_FEATURES),
 * probes ONE adapter for the policy verdict and the printed summary, and never throws: requireGpu(t) and the
 * acquire*() helpers enforce GRAPHTY_GPU_REQUIRE (unset = skip, any, hardware, <vendor>; scripts/gpu-policy.js).
 * Every device comes from a FRESH adapter (an adapter is consumed by one requestDevice, spec 2.2 step 1).
 * acquire() wraps a GpuContext with an onError sink; afterEach fails the test when the sink collected anything
 * ("a wrong result is never a skip"; an uncaptured error is never a pass). afterAll disposes every context and
 * raw device, appends every context's pipeline keys to GRAPHTY_PIPELINE_KEY_LOG (one JSON-encoded key per line
 * in keys-<pid>.jsonl; test/setup/global.ts reads them at P2-T2) and drops both Dawn handles so the fork exits.
 *
 * Environment (spec 12.2; process.env is read ONLY here, in vitest.config.ts and under scripts/):
 * GRAPHTY_GPU_REQUIRE (the policy, parsed by scripts/gpu-policy.js -- the one copy of the rule, D19),
 * GRAPHTY_GPU_ADAPTER (Dawn `adapter=` substring: `llvmpipe` mirrors CI), GRAPHTY_DAWN_FEATURES (comma-separated
 * Dawn toggles), GRAPHTY_GPU_NO_SUBGROUPS ("1" drops `subgroups` from the default optional features, D16),
 * GRAPHTY_GPU_INSPECT ("1" sets ctx.debug.inspect on every acquire()d context, spec 11.9 item 2),
 * GRAPHTY_PIPELINE_KEY_LOG (a directory for the key log), XDG_RUNTIME_DIR (set to /tmp when unset; silences Mesa).
 *
 * "A wrong result is never a skip": requireGpu skips ONLY under the unset policy and only for a missing adapter;
 * under any / hardware / <vendor> a missing or non-conforming adapter FAILS the test.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

import { afterAll, afterEach, beforeAll, type TestContext } from "vitest";

import { checkAdapter, type GpuPolicy, parseGpuRequire } from "../../scripts/gpu-policy.js";
import { GpuContext } from "../../src/context.js";
import {
    buildRequiredFeatures,
    buildRequiredLimits,
    requestDevice,
    summarizeAdapter,
} from "../../src/device/acquire.js";
import type { WebGpuGraphError } from "../../src/errors.js";
import { createNodeGpu, type NodeGpuHandle } from "../../src/node/index.js";
import type { AdapterSummary, LimitPolicy } from "../../src/types/context.js";

/** The optional features acquireRaw() requests by default (the create() default of spec 2.2). */
const DEFAULT_OPTIONAL_FEATURES: readonly GPUFeatureName[] = ["subgroups", "timestamp-query"];

/** The default optional feature list, minus subgroups under GRAPHTY_GPU_NO_SUBGROUPS=1 (spec 11.2: the whole run). */
function defaultOptionalFeatures(): readonly GPUFeatureName[] {
    if (process.env.GRAPHTY_GPU_NO_SUBGROUPS === "1") {
        return DEFAULT_OPTIONAL_FEATURES.filter((name) => name !== "subgroups");
    }
    return DEFAULT_OPTIONAL_FEATURES;
}

if (process.env.XDG_RUNTIME_DIR === undefined || process.env.XDG_RUNTIME_DIR === "") {
    process.env.XDG_RUNTIME_DIR = "/tmp";
}

type Verdict = ReturnType<typeof checkAdapter>;

/** Options of acquire(). `subgroups` defaults to GRAPHTY_GPU_NO_SUBGROUPS !== "1"; false -> optionalFeatures []. */
export interface AcquireOptions {
    readonly subgroups?: boolean | undefined;
    readonly limits?: LimitPolicy | undefined;
    readonly label?: string | undefined;
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;
    readonly rejectSoftware?: boolean | undefined;
    readonly warnUnreleasedSnapshots?: number | undefined;
}

let policy: GpuPolicy = parseGpuRequire(process.env.GRAPHTY_GPU_REQUIRE);
let handle: NodeGpuHandle | null = null;
let nullHandle: NodeGpuHandle | null = null;
let summary: AdapterSummary | null = null;
let reason: string | null = "E_NO_ADAPTER: device acquisition did not run";
let verdict: Verdict = { ok: false, skip: true, reason };
const contexts: GpuContext[] = [];
const rawDevices: GPUDevice[] = [];
const uncaptured: WebGpuGraphError[] = [];
const pipelineKeys: string[] = [];

function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

function envOrUndefined(name: string): string | undefined {
    const value = process.env[name];
    return value === undefined || value === "" ? undefined : value;
}

/** The worker's Dawn handle, or a thrown Error carrying the E_NO_ADAPTER reason. */
function dawn(): NodeGpuHandle {
    if (handle === null) {
        throw new Error(reason ?? "E_NO_ADAPTER: no Dawn handle (call requireGpu(t) before acquire())");
    }
    return handle;
}

/** The parsed policy of GRAPHTY_GPU_REQUIRE (scripts/gpu-policy.js shape). */
export function gpuPolicy(): GpuPolicy {
    return policy;
}

/** The adapter summary of the probe the setup ran once in beforeAll (null when acquisition failed). */
export function adapterSummary(): AdapterSummary | null {
    return summary;
}

/** Why acquisition failed ("E_NO_ADAPTER: ..." text), or null. */
export function skipReason(): string | null {
    return reason;
}

/** True when the setup's probe found a software adapter. */
export function isSoftware(): boolean {
    return summary?.software ?? false;
}

/** 1 on hardware, 1 / 50 on a software adapter (spec 11.2); scales fixture sizes and iteration counts. */
export function gpuScale(): number {
    return isSoftware() ? 1 / 50 : 1;
}

/** The errors the uncapturederror sink collected during the current test (drained by the afterEach hook). */
export function uncapturedErrors(): readonly WebGpuGraphError[] {
    return uncaptured;
}

/**
 * Skips the calling test when no adapter exists and the policy is "skip"; THROWS (fails) under any / hardware /
 * <vendor> when the adapter is absent or violates the policy; returns otherwise.
 */
export function requireGpu(t: TestContext): void {
    if (verdict.ok) {
        return;
    }
    const why = verdict.reason ?? reason ?? "no WebGPU adapter";
    if (verdict.skip) {
        t.skip(why);
    }
    throw new Error(`GRAPHTY_GPU_REQUIRE=${policy.raw === "" ? "(unset)" : policy.raw}: ${why}`);
}

/**
 * A FRESH adapter and device (raised limits) with NO context around them; registered for destroy() in afterAll.
 * The default feature list drops subgroups under GRAPHTY_GPU_NO_SUBGROUPS=1, as createContext does.
 */
export async function acquireRaw(
    options: {
        readonly limits?: LimitPolicy | undefined;
        readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;
    } = {},
): Promise<{
    readonly gpu: GPU;
    readonly adapter: GPUAdapter;
    readonly device: GPUDevice;
    readonly info: GPUAdapterInfo;
}> {
    const { gpu } = dawn();
    const adapter = await gpu.requestAdapter();
    if (adapter === null) {
        throw new Error("E_NO_ADAPTER: requestAdapter() returned null in acquireRaw()");
    }
    const device = await requestDevice(adapter, {
        label: "acquireRaw",
        requiredLimits: buildRequiredLimits(adapter, options.limits ?? "raise"),
        requiredFeatures: buildRequiredFeatures(adapter, [], options.optionalFeatures ?? defaultOptionalFeatures()),
    });
    rawDevices.push(device);
    return { gpu, adapter, device, info: adapter.info };
}

/** A context over `gpu` with the setup's sink, the inspect flag and the afterAll registration. */
async function createContext(gpu: GPU, options: AcquireOptions, defaultLabel: string): Promise<GpuContext> {
    const subgroups = options.subgroups ?? process.env.GRAPHTY_GPU_NO_SUBGROUPS !== "1";
    const ctx = await GpuContext.create({
        gpu,
        runtime: "node",
        limits: options.limits ?? "raise",
        optionalFeatures: subgroups ? options.optionalFeatures : [],
        label: options.label ?? defaultLabel,
        rejectSoftware: options.rejectSoftware,
        warnUnreleasedSnapshots: options.warnUnreleasedSnapshots,
        onError: (error) => {
            uncaptured.push(error);
        },
    });
    ctx.debug.inspect = process.env.GRAPHTY_GPU_INSPECT === "1";
    contexts.push(ctx);
    return ctx;
}

/**
 * P1: a GpuContext over a FRESH adapter each call (spec 11.2), created with runtime "node", onError collecting into
 * the per-test uncaptured list (the afterEach hook fails the test when it is non-empty), ctx.debug.inspect from
 * GRAPHTY_GPU_INSPECT, registered for dispose() in afterAll.
 */
export function acquire(options: AcquireOptions = {}): Promise<GpuContext> {
    return createContext(dawn().gpu, options, "acquire");
}

/**
 * P1: a GpuContext over Dawn's `backend=null` adapter (spec 5.1: compiles pipelines, runs nothing) from a SECOND
 * GPU handle created lazily once per worker through createNodeGpu({ backend: "null", installGlobals: false });
 * a fresh adapter per call; disposed in afterAll with the handle.
 */
export async function acquireNullBackend(options: AcquireOptions = {}): Promise<GpuContext> {
    dawn();
    if (nullHandle === null) {
        nullHandle = await createNodeGpu({ backend: "null", installGlobals: false });
    }
    return createContext(nullHandle.gpu, options, "acquire-null-backend");
}

/** Appends the keys collected so far to GRAPHTY_PIPELINE_KEY_LOG/keys-<pid>.jsonl (one JSON string per line). */
function writeKeyLog(): void {
    const dir = envOrUndefined("GRAPHTY_PIPELINE_KEY_LOG");
    const keys = pipelineKeys.splice(0);
    if (dir === undefined || keys.length === 0) {
        return;
    }
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, `keys-${process.pid}.jsonl`), keys.map((key) => `${JSON.stringify(key)}\n`).join(""));
}

beforeAll(async () => {
    policy = parseGpuRequire(process.env.GRAPHTY_GPU_REQUIRE);
    const dawnFeatures = envOrUndefined("GRAPHTY_DAWN_FEATURES");
    try {
        handle = await createNodeGpu({
            adapter: envOrUndefined("GRAPHTY_GPU_ADAPTER"),
            dawnFeatures: dawnFeatures === undefined ? undefined : dawnFeatures.split(","),
        });
    } catch (err) {
        reason = `E_NO_ADAPTER: ${messageOf(err)}`;
    }
    if (handle !== null) {
        try {
            const adapter = await handle.gpu.requestAdapter();
            if (adapter === null) {
                reason =
                    "E_NO_ADAPTER: requestAdapter() returned null (no usable Vulkan ICD; is libEGL.so.1 on LD_LIBRARY_PATH?)";
            } else {
                summary = summarizeAdapter(adapter);
                reason = null;
                console.warn(
                    `[gpu] adapter vendor=${summary.vendor} architecture=${summary.architecture} device=${summary.device} description=${summary.description} software=${summary.software} subgroups=${summary.subgroupMinSize}-${summary.subgroupMaxSize} features=${summary.features.join(",")}`,
                );
                console.warn(
                    `[gpu] adapter limits maxBufferSize=${summary.limits.maxBufferSize} maxStorageBufferBindingSize=${summary.limits.maxStorageBufferBindingSize} minStorageBufferOffsetAlignment=${summary.limits.minStorageBufferOffsetAlignment} maxComputeWorkgroupsPerDimension=${summary.limits.maxComputeWorkgroupsPerDimension}`,
                );
            }
        } catch (err) {
            reason = `E_NO_ADAPTER: requestAdapter() threw: ${messageOf(err)}`;
        }
    }
    verdict = checkAdapter(
        summary === null ? null : { vendor: summary.vendor, architecture: summary.architecture },
        policy,
    );
    if (reason !== null) {
        console.warn(`[gpu] ${reason}`);
    }
});

afterEach(() => {
    const errors = uncaptured.splice(0);
    if (errors.length > 0) {
        throw new Error(
            `uncaptured GPU errors during the test (spec 11.2: never a pass):\n${errors.map((e) => `${e.code}: ${e.message}`).join("\n")}`,
        );
    }
});

afterAll(async () => {
    const lost: Promise<GPUDeviceLostInfo>[] = [];
    for (const ctx of contexts.splice(0)) {
        try {
            for (const key of ctx.pipelines.keys()) {
                pipelineKeys.push(key);
            }
        } catch {
            // a context the test disposed itself may refuse the read; its keys were compiled all the same
        }
        ctx.dispose();
        lost.push(ctx.lost);
    }
    for (const device of rawDevices.splice(0)) {
        device.destroy();
        lost.push(device.lost);
    }
    // PLAN DECISION (P1-T1, measured on webgpu@0.4.0 with tmp/p1t1/gc-race2.mjs): the Dawn GPU object must stay
    // referenced until every destroyed device has reported its loss -- a GC of the instance while the device
    // teardown callbacks are in flight segfaults, aborts or deadlocks the worker (the futex hang after afterAll).
    // Every lost promise resolves right after destroy(); one macrotask lets the delivered callbacks unwind.
    await Promise.all(lost);
    await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
    });
    writeKeyLog();
    if (nullHandle !== null) {
        nullHandle.dispose();
        nullHandle = null;
    }
    if (handle !== null) {
        handle.dispose();
        handle = null;
    }
});
