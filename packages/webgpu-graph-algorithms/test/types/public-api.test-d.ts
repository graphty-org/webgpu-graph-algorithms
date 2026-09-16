import { type F32, type GraphSnapshot, type NumericVector, type U32 } from "@graphty/graph-format";
import {
    type AcceleratorOptions,
    type AdapterInfoLike,
    type AdapterSummary,
    type AlgorithmAccelerator,
    type ApspResultLike,
    ARC_WINDOW_ALIGN,
    type BellmanFordResultLike,
    type BfsResultLike,
    type CommonLayoutOptions,
    type CommunityResultLike,
    type CorenessResultLike,
    type CpuAlgorithmOptions,
    createAccelerator,
    createForceAtlas2,
    degree,
    type EdgeScoresResultLike,
    EXACT_MAX_NODES,
    FA2_DEFAULTS,
    type ForceAtlas2Options,
    type ForceAtlas2Stats,
    type ForceAtlas2TraceRecord,
    type FruchtermanReingoldOptions,
    type GpuAccelerator,
    type GpuCaps,
    GpuContext,
    type GpuContextOptions,
    type GpuLayoutSimulation,
    type GpuLayoutTuning,
    type GpuRunOptions,
    hasErrorCode,
    type HitsResultLike,
    isSoftwareAdapter,
    isWebGpuGraphError,
    type LabelResultLike,
    LAYOUT_TUNING_DEFAULTS,
    type LayoutAccelerator,
    type LayoutSimulation,
    type LayoutStatsBase,
    type LimitPolicy,
    MAX_1D_ITEMS,
    MAX_WORKGROUPS_PER_DIM,
    type MstResultLike,
    type PageRankResultLike,
    PASSTHROUGH_FORMAT_CODES,
    type PassTiming,
    type PlanCaps,
    type PlanLimits,
    type ProbeOptions,
    type ProbeResult,
    type Profiler,
    type RaisableLimit,
    type RunOptions,
    type ScoresResultLike,
    seedPositions,
    type SimulationOptions,
    type SpringElectricalOptions,
    type SsspResultLike,
    STORAGE_ALIGN,
    WebGpuGraphError,
    type WebGpuGraphErrorCode,
    WORKGROUP_SIZE,
} from "@graphty/webgpu-graph-algorithms";
import {
    type BrowserGpuOptions,
    probeBrowserWebGpu,
    requestGpuContext,
} from "@graphty/webgpu-graph-algorithms/browser";
import {
    createNodeGpu,
    createNodeGpuContext,
    dawnFlags,
    type NodeGpuHandle,
    type NodeGpuOptions,
    probeNodeWebGpu,
} from "@graphty/webgpu-graph-algorithms/node";
import { expectTypeOf } from "vitest";

// The strict-consumer sample of spec 11.3 / design 16.6. Compiled by `tsc -p tsconfig.strict-consumer.json` against
// dist/webgpu-graph-algorithms.d.ts, dist/browser.d.ts and dist/node.d.ts with noUncheckedIndexedAccess and
// exactOptionalPropertyTypes ON (after `pnpm run build:all`), and by `tsc --noEmit -p tsconfig.json` against src/;
// never executed. The import list above IS the pinned type list of contract 3.15: noUnusedLocals fails the compile
// for a name the barrel dropped, and every imported name is used below.

declare const ctx: GpuContext;
declare const snapshot: GraphSnapshot;
declare const positions: F32;
declare const info: GPUAdapterInfo;
type Fa2Sim = GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;

// ---- errors (contract 3.1): construct, brand-check, narrow a code
const err = new WebGpuGraphError("E_TOO_LARGE", "needs 3 GiB", {
    needed: 3,
    limit: 2,
    path: "colIdx",
    algorithm: null,
});
expectTypeOf(WebGpuGraphError).toBeConstructibleWith("E_NO_ADAPTER", "no adapter");
expectTypeOf(WebGpuGraphError).toBeConstructibleWith("E_NO_ADAPTER", "no adapter", { reason: "none" });
type CtorParams = ConstructorParameters<typeof WebGpuGraphError>;
expectTypeOf<CtorParams[0]>().toEqualTypeOf<WebGpuGraphErrorCode>();
expectTypeOf<CtorParams[1]>().toBeString();
expectTypeOf<CtorParams[2]>().toEqualTypeOf<Record<string, unknown> | undefined>();
expectTypeOf(err).toMatchTypeOf<Error>();
expectTypeOf(err.code).toEqualTypeOf<WebGpuGraphErrorCode>();
expectTypeOf(err.details).toEqualTypeOf<Readonly<Record<string, unknown>>>();
expectTypeOf(err.details.needed).toBeUnknown();
expectTypeOf(err.name).toEqualTypeOf<"WebGpuGraphError">();
if (err.code === "E_DEVICE_LOST") {
    expectTypeOf(err.code).toEqualTypeOf<"E_DEVICE_LOST">();
}
declare const maybe: unknown;
if (isWebGpuGraphError(maybe)) {
    expectTypeOf(maybe).toEqualTypeOf<WebGpuGraphError>();
}
expectTypeOf(hasErrorCode).parameter(1).toEqualTypeOf<WebGpuGraphErrorCode>();
expectTypeOf(hasErrorCode(maybe, "E_ABORTED")).toBeBoolean();
expectTypeOf<WebGpuGraphErrorCode>().toEqualTypeOf<
    | "E_NO_WEBGPU"
    | "E_NO_ADAPTER"
    | "E_NO_DEVICE"
    | "E_SOFTWARE_ONLY"
    | "E_DEVICE_LOST"
    | "E_DISPOSED"
    | "E_VALIDATION"
    | "E_SHADER_COMPILE"
    | "E_OUT_OF_MEMORY"
    | "E_TOO_LARGE"
    | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT"
    | "E_SNAPSHOT"
    | "E_RELEASED"
    | "E_NOT_LOADED"
    | "E_ABORTED"
>();
expectTypeOf<"E_IN_FLIGHT">().not.toMatchTypeOf<WebGpuGraphErrorCode>(); // spec 3.3: a saturated step() coalesces
expectTypeOf<typeof PASSTHROUGH_FORMAT_CODES>().toEqualTypeOf<
    readonly ["E_GPU_INELIGIBLE", "E_UNKNOWN_NODE", "E_UNKNOWN_COLUMN", "E_COLUMN_LENGTH"]
>();

// ---- constants (contract 3.2): literal types where the value is fixed for good; EXACT_MAX_NODES is re-fixed at G3
expectTypeOf<typeof WORKGROUP_SIZE>().toEqualTypeOf<256>();
expectTypeOf<typeof MAX_WORKGROUPS_PER_DIM>().toEqualTypeOf<65535>();
expectTypeOf<typeof MAX_1D_ITEMS>().toBeNumber();
expectTypeOf<typeof ARC_WINDOW_ALIGN>().toEqualTypeOf<64>();
expectTypeOf<typeof STORAGE_ALIGN>().toEqualTypeOf<256>();
expectTypeOf<typeof EXACT_MAX_NODES>().toBeNumber();
expectTypeOf(FA2_DEFAULTS.maxIter).toEqualTypeOf<100>();
expectTypeOf(FA2_DEFAULTS.dim).toEqualTypeOf<2>();
expectTypeOf(FA2_DEFAULTS.settleThreshold).toEqualTypeOf<0.001>();
expectTypeOf(FA2_DEFAULTS.maxInFlight).toEqualTypeOf<2>();
expectTypeOf(FA2_DEFAULTS.strongGravity).toEqualTypeOf<false>();
expectTypeOf(LAYOUT_TUNING_DEFAULTS.repulsion).toEqualTypeOf<"auto">();
expectTypeOf(LAYOUT_TUNING_DEFAULTS.compat).toEqualTypeOf<"paper">();
expectTypeOf(LAYOUT_TUNING_DEFAULTS.exactMaxNodes).toBeNumber();
expectTypeOf(LAYOUT_TUNING_DEFAULTS.deterministic).toEqualTypeOf<true>();

// ---- context and capabilities (contract 3.3, 3.5)
expectTypeOf(GpuContext.probe).parameter(0).toEqualTypeOf<ProbeOptions>();
expectTypeOf(GpuContext.probe).returns.resolves.toEqualTypeOf<ProbeResult>();
expectTypeOf(GpuContext.create).parameter(0).toEqualTypeOf<GpuContextOptions>();
expectTypeOf(GpuContext.create).returns.resolves.toEqualTypeOf<GpuContext>();
expectTypeOf(GpuContext.from).parameter(0).toEqualTypeOf<GPUDevice>();
expectTypeOf(GpuContext.from).parameter(1).toEqualTypeOf<Partial<GpuCaps> | undefined>();
expectTypeOf(ctx.device).toEqualTypeOf<GPUDevice>();
expectTypeOf(ctx.caps).toEqualTypeOf<GpuCaps>();
expectTypeOf(ctx.state).toEqualTypeOf<"ready" | "lost" | "disposed">();
expectTypeOf(ctx.lost).resolves.toEqualTypeOf<GPUDeviceLostInfo>();
expectTypeOf(ctx.profiler).toEqualTypeOf<Profiler | null>();
expectTypeOf(ctx.release).parameter(0).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(ctx.release).returns.toBeVoid();
expectTypeOf(ctx.dispose).returns.toBeVoid();
expectTypeOf<GpuCaps["limits"]>().toEqualTypeOf<GPUSupportedLimits>();
expectTypeOf<GpuCaps["runtime"]>().toEqualTypeOf<"browser" | "node" | "unknown">();
expectTypeOf<GpuCaps["features"]>().toEqualTypeOf<ReadonlySet<string>>();
expectTypeOf<GpuCaps>().toMatchTypeOf<PlanCaps>(); // "a GpuCaps IS a PlanCaps" (contract 3.3)
expectTypeOf<GPUSupportedLimits>().toMatchTypeOf<PlanLimits>(); // the branded limits are structurally a PlanLimits
expectTypeOf<keyof PlanLimits>().toEqualTypeOf<
    | "maxBufferSize"
    | "maxStorageBufferBindingSize"
    | "maxStorageBuffersPerShaderStage"
    | "minStorageBufferOffsetAlignment"
    | "minUniformBufferOffsetAlignment"
    | "maxComputeWorkgroupsPerDimension"
    | "maxComputeInvocationsPerWorkgroup"
    | "maxComputeWorkgroupSizeX"
    | "maxComputeWorkgroupStorageSize"
    | "maxUniformBufferBindingSize"
>();
expectTypeOf<RaisableLimit>().toEqualTypeOf<
    | "maxBufferSize"
    | "maxStorageBufferBindingSize"
    | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupStorageSize"
    | "maxComputeInvocationsPerWorkgroup"
    | "maxComputeWorkgroupSizeX"
>();
expectTypeOf<LimitPolicy>().toEqualTypeOf<"default" | "raise" | Readonly<Partial<Record<RaisableLimit, number>>>>();
expectTypeOf<GpuContextOptions["limits"]>().toEqualTypeOf<LimitPolicy | undefined>();
expectTypeOf<GpuContextOptions["onError"]>().toEqualTypeOf<((error: WebGpuGraphError) => void) | undefined>();
expectTypeOf<ProbeOptions["gpu"]>().toEqualTypeOf<GPU | undefined>();
expectTypeOf<ProbeResult["code"]>().toEqualTypeOf<"OK" | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_SOFTWARE_ONLY">();
expectTypeOf<ProbeResult["adapter"]>().toEqualTypeOf<GPUAdapter | null>();
expectTypeOf<ProbeResult["summary"]>().toEqualTypeOf<AdapterSummary | null>();
expectTypeOf<ProbeResult["reason"]>().toEqualTypeOf<string | null>();
expectTypeOf<AdapterSummary["limits"]>().toEqualTypeOf<Readonly<Record<string, number>>>();
expectTypeOf<AdapterSummary["software"]>().toBeBoolean();
expectTypeOf(isSoftwareAdapter).parameter(0).toEqualTypeOf<AdapterInfoLike>();
expectTypeOf(isSoftwareAdapter).returns.toBeBoolean();
expectTypeOf(info).toMatchTypeOf<AdapterInfoLike>(); // the branded GPUAdapterInfo passes without a cast
expectTypeOf<PassTiming>().toEqualTypeOf<{ readonly label: string; readonly ns: number }>();
expectTypeOf<Profiler["enabled"]>().toBeBoolean();
expectTypeOf<Profiler["quantised"]>().toBeBoolean();
expectTypeOf<Profiler["beginPass"]>().returns.toEqualTypeOf<GPUComputePassTimestampWrites | undefined>();

// ---- the algorithm surface (P1: degree; contract 3.12) and its run options
expectTypeOf(degree).parameter(1).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(degree).parameter(2).toEqualTypeOf<GpuRunOptions | undefined>();
expectTypeOf(degree).returns.resolves.toEqualTypeOf<U32>();
expectTypeOf<GpuRunOptions["dest"]>().toEqualTypeOf<Float32Array | Uint32Array | undefined>();
expectTypeOf<GpuRunOptions["onProgress"]>().toEqualTypeOf<((done: number, total: number) => void) | undefined>();

// ---- layouts (P3; contract 3.3, 3.13)
expectTypeOf(seedPositions).parameter(2).toEqualTypeOf<number | null>();
expectTypeOf(seedPositions).parameter(3).toEqualTypeOf<2 | 3>();
expectTypeOf(seedPositions).parameter(5).toEqualTypeOf<ArrayLike<number> | null>();
expectTypeOf(seedPositions).parameter(6).toEqualTypeOf<"fa2" | "fr">();
expectTypeOf(seedPositions).returns.toBeVoid();
expectTypeOf(createForceAtlas2).parameter(1).toEqualTypeOf<(ForceAtlas2Options & GpuLayoutTuning) | undefined>();
expectTypeOf(createForceAtlas2).returns.toEqualTypeOf<Fa2Sim>();
expectTypeOf<Fa2Sim>().toMatchTypeOf<LayoutSimulation>();
expectTypeOf<ForceAtlas2Options>().toMatchTypeOf<CommonLayoutOptions & SimulationOptions>();
expectTypeOf<FruchtermanReingoldOptions>().toMatchTypeOf<CommonLayoutOptions & SimulationOptions>();
expectTypeOf<SpringElectricalOptions>().toMatchTypeOf<CommonLayoutOptions & SimulationOptions>();
expectTypeOf<ForceAtlas2Stats>().toMatchTypeOf<LayoutStatsBase>();
expectTypeOf<ForceAtlas2Stats["trace"]>().toEqualTypeOf<ReadonlyArray<ForceAtlas2TraceRecord>>();
expectTypeOf<keyof ForceAtlas2TraceRecord>().toEqualTypeOf<
    "swing" | "traction" | "speed" | "speedEfficiency" | "meanDisplacement" | "settledCount"
>();
expectTypeOf<LayoutStatsBase["centroid"]>().toEqualTypeOf<readonly [number, number, number]>();
expectTypeOf<LayoutStatsBase["repulsionTier"]>().toEqualTypeOf<"exact" | "grid">();
expectTypeOf<LayoutStatsBase["maxCellOccupancy"]>().toEqualTypeOf<number | null>();
expectTypeOf<LayoutStatsBase["outsideGrid"]>().toEqualTypeOf<number | null>();
expectTypeOf<LayoutStatsBase["msPerIteration"]>().toEqualTypeOf<number | null>();
expectTypeOf<RunOptions["signal"]>().toEqualTypeOf<AbortSignal | undefined>();
expectTypeOf<GpuLayoutTuning["exactMaxNodes"]>().toEqualTypeOf<number | undefined>();

// ---- the accelerator (P3; contract 3.14) and the CPU-package mirrors (spec 9.2, 9.3)
expectTypeOf(createAccelerator).parameter(0).toEqualTypeOf<GpuContext>();
expectTypeOf(createAccelerator).parameter(1).toEqualTypeOf<AcceleratorOptions | undefined>();
expectTypeOf(createAccelerator).returns.toEqualTypeOf<GpuAccelerator>();
expectTypeOf<GpuAccelerator>().toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>();
expectTypeOf<AcceleratorOptions["layout"]>().toEqualTypeOf<GpuLayoutTuning | undefined>();
expectTypeOf<CpuAlgorithmOptions>().toEqualTypeOf<Readonly<Record<string, unknown>>>();
expectTypeOf<ScoresResultLike["scores"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<PageRankResultLike>().toMatchTypeOf<ScoresResultLike>();
expectTypeOf<PageRankResultLike["danglingMass"]>().toEqualTypeOf<number | undefined>();
expectTypeOf<HitsResultLike["hubs"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<HitsResultLike["authorities"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<LabelResultLike["labels"]>().toEqualTypeOf<U32>();
expectTypeOf<LabelResultLike["groups"]>().returns.toEqualTypeOf<U32[]>();
expectTypeOf<BfsResultLike["depth"]>().toEqualTypeOf<U32>();
expectTypeOf<BfsResultLike["visitedCount"]>().toBeNumber();
expectTypeOf<SsspResultLike["dist"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<SsspResultLike["predArc"]>().toEqualTypeOf<U32>();
expectTypeOf<BellmanFordResultLike>().toMatchTypeOf<SsspResultLike>();
expectTypeOf<BellmanFordResultLike["hasNegativeCycle"]>().toBeBoolean();
expectTypeOf<EdgeScoresResultLike["scores"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<ApspResultLike["dist"]>().toEqualTypeOf<NumericVector>();
expectTypeOf<ApspResultLike["n"]>().toBeNumber();
expectTypeOf<CorenessResultLike["coreness"]>().toEqualTypeOf<U32>();
expectTypeOf<MstResultLike["edges"]>().toEqualTypeOf<U32>();
expectTypeOf<MstResultLike["totalWeight"]>().toBeNumber();
expectTypeOf<CommunityResultLike>().toMatchTypeOf<LabelResultLike>();
expectTypeOf<CommunityResultLike["modularity"]>().toBeNumber();
expectTypeOf<NonNullable<AlgorithmAccelerator["pageRank"]>>().parameter(0).toEqualTypeOf<GraphSnapshot>();
expectTypeOf<NonNullable<AlgorithmAccelerator["pageRank"]>>()
    .parameter(1)
    .toEqualTypeOf<CpuAlgorithmOptions | undefined>();
expectTypeOf<NonNullable<AlgorithmAccelerator["pageRank"]>>().returns.resolves.toEqualTypeOf<PageRankResultLike>();
expectTypeOf<NonNullable<AlgorithmAccelerator["triangleCount"]>>().returns.resolves.toEqualTypeOf<{
    readonly perNode: U32;
    readonly total: number;
}>();
expectTypeOf<NonNullable<LayoutAccelerator["forceAtlas2"]>>().returns.toEqualTypeOf<LayoutSimulation>();

// ---- the two entries by their package names (contract 3.6, 3.7; the dist d.ts shims of 2.6)
expectTypeOf(probeBrowserWebGpu).parameter(0).toEqualTypeOf<BrowserGpuOptions | undefined>();
expectTypeOf(probeBrowserWebGpu).returns.resolves.toEqualTypeOf<ProbeResult>();
expectTypeOf(requestGpuContext).parameter(0).toEqualTypeOf<BrowserGpuOptions | undefined>();
expectTypeOf(requestGpuContext).returns.resolves.toEqualTypeOf<GpuContext>();
expectTypeOf<BrowserGpuOptions>().not.toHaveProperty("gpu");
expectTypeOf<BrowserGpuOptions>().not.toHaveProperty("device");
expectTypeOf<BrowserGpuOptions>().not.toHaveProperty("runtime");
expectTypeOf<BrowserGpuOptions>().toHaveProperty("adapter");
expectTypeOf(createNodeGpu).parameter(0).toEqualTypeOf<NodeGpuOptions | undefined>();
expectTypeOf(createNodeGpu).returns.resolves.toEqualTypeOf<NodeGpuHandle>();
expectTypeOf(createNodeGpuContext).returns.resolves.toEqualTypeOf<GpuContext>();
expectTypeOf(probeNodeWebGpu).returns.resolves.toEqualTypeOf<ProbeResult>();
expectTypeOf(dawnFlags).parameter(0).toEqualTypeOf<NodeGpuOptions | undefined>();
expectTypeOf(dawnFlags).returns.toEqualTypeOf<string[]>();
expectTypeOf<NodeGpuHandle["gpu"]>().toEqualTypeOf<GPU>();
expectTypeOf<NodeGpuHandle["dispose"]>().returns.toBeVoid();
expectTypeOf<NodeGpuOptions["backend"]>().toEqualTypeOf<
    "vulkan" | "d3d12" | "d3d11" | "metal" | "opengl" | "opengles" | "null" | undefined
>();
expectTypeOf<NodeGpuOptions>().not.toHaveProperty("gpu");
expectTypeOf<NodeGpuOptions>().not.toHaveProperty("device");
expectTypeOf<NodeGpuOptions>().not.toHaveProperty("runtime");
// NodeGpuOptions.adapter is the Dawn adapter NAME (contract 3.7), not GpuContextOptions.adapter
const nodeOptions: NodeGpuOptions = { adapter: "llvmpipe", backend: undefined, software: undefined, label: "sample" };
expectTypeOf(nodeOptions.adapter).toEqualTypeOf<string | undefined>();

/**
 * A consumer's start-up and one layout run in the shape spec 2.4 / 9.4 prescribe: probe, create, inject, lay out,
 * read the stats, release. Type-checked only, never executed: the strict flags are the point. Every option record
 * carries explicit `undefined` members (exactOptionalPropertyTypes) and every index read is guarded
 * (noUncheckedIndexedAccess: `trace[i]` and `degrees[0]` are `T | undefined`).
 * @param gpu - navigator.gpu or a Dawn handle's gpu (undefined -> E_NO_WEBGPU at probe time)
 * @returns the last traced swing plus node 0's out-degree
 */
async function consumerSample(gpu: GPU | undefined): Promise<number> {
    const probe: ProbeResult = await GpuContext.probe({ gpu, rejectSoftware: true, powerPreference: undefined });
    if (!probe.ok || probe.adapter === null) {
        throw new WebGpuGraphError("E_NO_ADAPTER", probe.reason ?? "no adapter", { code: probe.code });
    }
    const context: GpuContext = await GpuContext.create({ adapter: probe.adapter, limits: "raise", label: "sample" });
    const tuning: GpuLayoutTuning = {
        compat: "networkx",
        exactMaxNodes: 4096,
        deterministic: true,
        nearMax: undefined,
    };
    const accelerator: GpuAccelerator = createAccelerator(context, { layout: tuning, algorithms: undefined });
    const injected: AlgorithmAccelerator & LayoutAccelerator = accelerator; // what the element stores (spec 9.4)
    const sim: Fa2Sim = accelerator.forceAtlas2({
        maxIter: 50,
        seed: 7,
        iterationsPerStep: 4,
        weight: true,
        nodeMass: null,
    });
    sim.load(snapshot, positions);
    const stats: ForceAtlas2Stats = await sim.run({ batch: 8, signal: undefined, maxIter: undefined });
    const last: ForceAtlas2TraceRecord | undefined = stats.trace[stats.trace.length - 1];
    const swing: number = last === undefined ? stats.swing : last.swing;
    const degrees: U32 = await degree(context, snapshot, { signal: undefined });
    const first: number | undefined = degrees[0];
    sim.dispose();
    injected.release?.(snapshot);
    accelerator.dispose();
    return swing + (first ?? 0);
}
expectTypeOf(consumerSample).returns.resolves.toBeNumber();
