/**
 * GpuContext (spec 2.2, 2.8, 3.3, 5.7; contract 3.5): one device, its caps and the singletons built on it --
 * the composition root of D26. probe() never creates a device; create() runs the six steps of spec 2.2 in
 * order; from() adopts a device the caller owns. Step 5 (the uncapturederror sink and the device.lost
 * watcher) and step 6 (the AllocationTracker, GraphResidency, BufferPool, Readback and PipelineCache) run
 * in the private constructor. The Profiler is constructed when "timestamp-query" is on the device (contract
 * 3.5; P2-T1) with `quantised` = runtime !== "node" (Dawn-node is the only runtime whose ticks are known
 * unquantised, spec 2.6). This file never imports ./kernels.js or ./wgsl/**.
 */

import type { GraphSnapshot } from "@graphty/graph-format";

import { DEFAULT_WARN_UNRELEASED_SNAPSHOTS } from "./constants.js";
import {
    buildRequiredFeatures,
    buildRequiredLimits,
    requestAdapter,
    requestDevice,
    summarizeAdapter,
} from "./device/acquire.js";
import { assertPlanLimits, capsFromDevice, captureCaps, workgroupSizeFor } from "./device/caps.js";
import { AllocationTracker } from "./device/error-scope.js";
import { deviceLostError, installUncapturedErrorSink, PendingErrorSlot, watchDeviceLost } from "./device/lost.js";
import { isWebGpuGraphError, WebGpuGraphError } from "./errors.js";
import { PipelineCache } from "./kernel/pipeline-cache.js";
import { Profiler } from "./kernel/profiler.js";
import { BufferPool } from "./memory/buffer-pool.js";
import { Readback } from "./memory/readback.js";
import { GraphResidency } from "./memory/residency.js";
import type { GpuCaps, GpuContextOptions, GpuDebugFlags, ProbeOptions, ProbeResult } from "./types/context.js";

/** The label a context and its device carry when create() is given none. */
const DEFAULT_LABEL = "webgpu-graph-algorithms";
/** The optionalFeatures default of spec 2.2. */
const DEFAULT_OPTIONAL_FEATURES: readonly GPUFeatureName[] = Object.freeze(["subgroups", "timestamp-query"]);
/** The powerPreference default of spec 2.2 (Chrome 145 needs it; Dawn-node ignores it). */
const DEFAULT_POWER_PREFERENCE: GPUPowerPreference = "high-performance";

/** What the private constructor takes: the device, its caps and the create-time choices. */
interface ContextInit {
    readonly device: GPUDevice;
    readonly caps: GpuCaps;
    readonly ownsDevice: boolean;
    readonly label: string;
    readonly onError: ((error: WebGpuGraphError) => void) | null;
    readonly warnUnreleasedSnapshots: number;
}

/**
 * Step 1 of create(): the caller's UNUSED adapter, else one requested from `gpu`; neither -> E_NO_WEBGPU.
 * @param options - the create options
 * @returns an adapter no device was requested from
 */
async function adapterOf(options: GpuContextOptions): Promise<GPUAdapter> {
    if (options.adapter !== undefined) {
        return options.adapter;
    }
    if (options.gpu !== undefined) {
        return requestAdapter(options.gpu, options.powerPreference ?? DEFAULT_POWER_PREFERENCE);
    }
    throw new WebGpuGraphError("E_NO_WEBGPU", "GpuContext.create needs one of gpu, adapter or device", {
        reason: "no gpu, adapter or device given",
        hint: "pass navigator.gpu (browser) or the gpu of createNodeGpu() (Node), or use the ./browser and ./node entries",
    });
}

/** One device, its caps and the singletons built on it (spec 3.3; the composition root of D26). */
export class GpuContext {
    /** The device (owned when create() made it, adopted otherwise). */
    readonly device: GPUDevice;
    /** The capability record captured from device.limits / device.features (spec 3.3). */
    readonly caps: GpuCaps;
    /** Resolves with the lost info once the device is lost or destroyed (spec 2.8). */
    readonly lost: Promise<GPUDeviceLostInfo>;
    /**
     * The upload cache (spec 3.3).
     * @internal
     */
    readonly residency: GraphResidency;
    /** Non-null when "timestamp-query" was granted (P2-T1 makes it functional; P1 always null). */
    readonly profiler: Profiler | null;
    /**
     * The compile-once pipeline cache.
     * @internal
     */
    readonly pipelines: PipelineCache;
    /**
     * The scratch pool.
     * @internal
     */
    readonly pool: BufferPool;
    /**
     * The staging ring.
     * @internal
     */
    readonly readback: Readback;
    /**
     * The OOM-scoped allocator shared by residency and pool.
     * @internal
     */
    readonly allocator: AllocationTracker;
    /**
     * The workgroup size of this device (spec 5.1).
     * @internal
     */
    readonly workgroupSize: number;
    /**
     * True when create() made the device (dispose() destroys it).
     * @internal
     */
    readonly ownsDevice: boolean;
    /**
     * The label given to create().
     * @internal
     */
    readonly label: string;
    /**
     * Mutable test-build flags (GpuDebugFlags).
     * @internal
     */
    readonly debug: GpuDebugFlags;

    private currentState: "ready" | "lost" | "disposed" = "ready";
    private lostError: WebGpuGraphError | null = null;
    private readonly slot = new PendingErrorSlot();
    private readonly uninstallSink: () => void;
    private readonly lostListeners = new Set<(info: GPUDeviceLostInfo) => void>();
    private readonly disposers: (() => void)[] = [];
    private batchCounter = 0;

    /**
     * Steps 5 and 6 of create() (spec 2.2): the sink, the loss watcher and the singletons.
     * @param init - the device, its caps and the create-time choices
     */
    private constructor(init: ContextInit) {
        this.device = init.device;
        this.caps = init.caps;
        this.ownsDevice = init.ownsDevice;
        this.label = init.label;
        this.debug = { inspect: false };
        this.workgroupSize = workgroupSizeFor(init.caps);
        this.allocator = new AllocationTracker(init.device);
        this.residency = new GraphResidency(init.device, init.caps, this.allocator, {
            warnUnreleasedSnapshots: init.warnUnreleasedSnapshots,
        });
        this.pool = new BufferPool(init.device, this.allocator, init.caps.limits.maxBufferSize);
        this.readback = new Readback(init.device, this.allocator);
        this.pipelines = new PipelineCache(init.device, init.caps);
        this.profiler = init.device.features.has("timestamp-query")
            ? new Profiler(init.device, true, init.caps.runtime !== "node")
            : null;
        this.uninstallSink = installUncapturedErrorSink(init.device, this.slot, init.onError);
        this.lost = watchDeviceLost(init.device, (info) => {
            this.handleLost(info);
        });
    }

    /**
     * Probe without creating a device (spec 2.2); never throws. `gpu === undefined` -> E_NO_WEBGPU;
     * requestAdapter() null or a throw -> E_NO_ADAPTER; `rejectSoftware && software` -> E_SOFTWARE_ONLY (the
     * adapter and its summary are still reported); else OK with the UNUSED adapter and its summary.
     * @param options - the GPU object, the power preference and the software policy
     * @returns the probe result
     */
    static async probe(options: ProbeOptions): Promise<ProbeResult> {
        const { gpu } = options;
        if (gpu === undefined) {
            return {
                ok: false,
                code: "E_NO_WEBGPU",
                reason: "no GPU object: navigator.gpu is undefined (no WebGPU in this runtime)",
                adapter: null,
                summary: null,
            };
        }
        let adapter: GPUAdapter;
        try {
            adapter = await requestAdapter(gpu, options.powerPreference ?? DEFAULT_POWER_PREFERENCE);
        } catch (err) {
            const reason = isWebGpuGraphError(err) ? err.message : String(err);
            return { ok: false, code: "E_NO_ADAPTER", reason, adapter: null, summary: null };
        }
        const summary = summarizeAdapter(adapter);
        if (options.rejectSoftware === true && summary.software) {
            return {
                ok: false,
                code: "E_SOFTWARE_ONLY",
                reason: `software adapter rejected: ${summary.vendor}/${summary.architecture}`,
                adapter,
                summary,
            };
        }
        return { ok: true, code: "OK", reason: null, adapter, summary };
    }

    /**
     * The six-step acquisition of spec 2.2: adopt `device`, else take `adapter`, else request one from `gpu`
     * (step 1); the software test (step 2); the limits / features and requestDevice (step 3); the caps from
     * device.limits plus assertPlanLimits (step 4); the sink, the loss watcher and the singletons (steps 5-6,
     * in the constructor).
     * @param options - see GpuContextOptions
     * @returns the context
     */
    static async create(options: GpuContextOptions): Promise<GpuContext> {
        const runtime = options.runtime ?? "unknown";
        const label = options.label ?? DEFAULT_LABEL;
        const onError = options.onError ?? null;
        const warnUnreleasedSnapshots = options.warnUnreleasedSnapshots ?? DEFAULT_WARN_UNRELEASED_SNAPSHOTS;
        if (options.device !== undefined) {
            const adopted = capsFromDevice(options.device, { runtime });
            assertPlanLimits(adopted);
            return new GpuContext({
                device: options.device,
                caps: adopted,
                ownsDevice: false,
                label,
                onError,
                warnUnreleasedSnapshots,
            });
        }
        const adapter = await adapterOf(options);
        const summary = summarizeAdapter(adapter);
        if (options.rejectSoftware === true && summary.software) {
            throw new WebGpuGraphError(
                "E_SOFTWARE_ONLY",
                `software adapter rejected by rejectSoftware: ${summary.vendor}/${summary.architecture}`,
                { adapter: summary },
            );
        }
        const requiredLimits = buildRequiredLimits(adapter, options.limits ?? "raise");
        const requiredFeatures = buildRequiredFeatures(
            adapter,
            options.requiredFeatures ?? [],
            options.optionalFeatures ?? DEFAULT_OPTIONAL_FEATURES,
        );
        const device = await requestDevice(adapter, { label, requiredLimits, requiredFeatures });
        const caps = captureCaps(device, adapter.info, runtime, options.gpu?.wgslLanguageFeatures ?? []);
        try {
            assertPlanLimits(caps);
        } catch (err) {
            device.destroy();
            throw err;
        }
        return new GpuContext({ device, caps, ownsDevice: true, label, onError, warnUnreleasedSnapshots });
    }

    /**
     * Zero-cost adoption of a device the caller owns (ownsDevice false, runtime "unknown" unless info says
     * otherwise). The context is labelled with the device's label, or "GpuContext.from" when it has none.
     * @param device - the device to adopt
     * @param info - what the caller knows about the adapter, if anything
     * @returns the context
     */
    static from(device: GPUDevice, info?: Partial<GpuCaps>): GpuContext {
        const caps = capsFromDevice(device, info);
        assertPlanLimits(caps);
        const deviceLabel: string = device.label ?? "";
        return new GpuContext({
            device,
            caps,
            ownsDevice: false,
            label: deviceLabel === "" ? "GpuContext.from" : deviceLabel,
            onError: null,
            warnUnreleasedSnapshots: DEFAULT_WARN_UNRELEASED_SNAPSHOTS,
        });
    }

    /**
     * "ready" | "lost" | "disposed".
     * @returns the lifecycle state
     */
    get state(): "ready" | "lost" | "disposed" {
        return this.currentState;
    }

    /**
     * Throws E_DEVICE_LOST / E_DISPOSED by state and rethrows a pending uncaptured error (spec 5.7: "thrown
     * from the next public call").
     * @internal
     */
    assertReady(): void {
        if (this.currentState === "disposed") {
            throw new WebGpuGraphError("E_DISPOSED", `GpuContext "${this.label}" is disposed`, { label: this.label });
        }
        if (this.lostError !== null) {
            throw this.lostError;
        }
        const pending = this.slot.take();
        if (pending !== null) {
            throw pending;
        }
    }

    /**
     * Takes the pending uncaptured error (used by CommandBatch right after submit).
     * @internal
     * @returns the error, or null
     */
    takePendingError(): WebGpuGraphError | null {
        return this.slot.take();
    }

    /**
     * Monotonically increasing batch ids, starting at 1.
     * @internal
     * @returns the next id
     */
    nextBatchId(): number {
        this.batchCounter += 1;
        return this.batchCounter;
    }

    /**
     * Registers a device-loss listener (simulations enter "disposed"); returns the unregister function.
     * @internal
     * @param listener - called once with the lost info
     * @returns removes the listener
     */
    onLost(listener: (info: GPUDeviceLostInfo) => void): () => void {
        this.lostListeners.add(listener);
        return () => {
            this.lostListeners.delete(listener);
        };
    }

    /**
     * A disposer run by dispose() (the Node entry attaches the GPU handle's dispose); attached after
     * dispose() it runs immediately.
     * @internal
     * @param dispose - the disposer
     */
    attachDisposer(dispose: () => void): void {
        if (this.currentState === "disposed") {
            dispose();
            return;
        }
        this.disposers.push(dispose);
    }

    /**
     * Destroys every buffer recorded for the snapshot (spec 4.5), then `pool.trim()` (spec 4.4: idle scratch
     * goes with the graph); idempotent; safe on a snapshot never uploaded; a no-op after dispose().
     * @param snapshot - the snapshot whose buffers are released
     */
    release(snapshot: GraphSnapshot): void {
        if (this.currentState === "disposed") {
            return;
        }
        this.residency.release(snapshot);
        this.pool.trim();
    }

    /**
     * Rejects pending work with E_DISPOSED (every later public call), destroys residency, pool and staging
     * ring, destroys the device when owned, runs attached disposers; idempotent (spec 2.8).
     */
    dispose(): void {
        if (this.currentState === "disposed") {
            return;
        }
        this.currentState = "disposed";
        this.uninstallSink();
        this.residency.destroyAll();
        this.pool.destroyAll();
        this.readback.destroyAll();
        if (this.profiler !== null) {
            this.profiler.destroy();
        }
        if (this.ownsDevice) {
            this.device.destroy();
        }
        const disposers = this.disposers.splice(0);
        for (const dispose of disposers) {
            dispose();
        }
    }

    /**
     * Step 5 of spec 2.2 on loss: state "lost", the residency cleared without destroying (the buffers are
     * gone with the device), every registered listener run once; a loss reported after dispose() (the owned
     * device was destroyed by dispose) changes nothing.
     * @param info - the device.lost result
     */
    private handleLost(info: GPUDeviceLostInfo): void {
        if (this.currentState === "disposed") {
            return;
        }
        this.currentState = "lost";
        this.lostError = deviceLostError(info);
        this.residency.clearOnLoss();
        for (const listener of Array.from(this.lostListeners)) {
            try {
                listener(info);
            } catch {
                // a throwing listener must not stop the fan-out; the others still learn of the loss
            }
        }
    }
}
