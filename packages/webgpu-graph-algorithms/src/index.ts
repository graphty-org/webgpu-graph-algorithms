/// <reference types="@webgpu/types" preserve="true" />
/**
 * The public barrel of @graphty/webgpu-graph-algorithms (spec 2.5, 3.3; contract 3.15): explicit named exports only,
 * values and `export type`, no star re-exports, no default export, nothing from src/browser/** or src/node/** (their
 * own entries; imported by nothing else in src/, 2.4) and nothing of the internal surface (GraphResidency,
 * BufferPool, Readback, Lease, PipelineCache, Kernel, CommandBatch, UniformRing, UniformBlock, composeWgsl, KERNELS,
 * ForceSimulation, ForceAtlas2Model: tests import them from their files). P0: the error class, the constants and
 * isSoftwareAdapter. P1 adds GpuContext and degree (values) and the context / run / profiler types. P2 adds nothing
 * (Lease, CommandBatch, UniformRing are internal). P3 adds the layout factory, the accelerator, the two default
 * tables, the seeder and the layout / accelerator types. test/index.test.ts pins the value list and
 * test/types/public-api.test-d.ts the type list; P4+ extends both. This comment must never spell the internal
 * JSDoc tag: it is the leading comment of the first export statement, and stripInternal would drop that statement
 * from the emitted declarations.
 */

// ==================== constants and errors (P0; FA2_DEFAULTS / LAYOUT_TUNING_DEFAULTS public from P3)
export {
    ARC_WINDOW_ALIGN,
    EXACT_MAX_NODES,
    FA2_DEFAULTS,
    LAYOUT_TUNING_DEFAULTS,
    MAX_1D_ITEMS,
    MAX_WORKGROUPS_PER_DIM,
    STORAGE_ALIGN,
    WORKGROUP_SIZE,
} from "./constants.js";
export {
    hasErrorCode,
    isWebGpuGraphError,
    PASSTHROUGH_FORMAT_CODES,
    WebGpuGraphError,
    type WebGpuGraphErrorCode,
} from "./errors.js";

// ==================== context, adapter classifier, profiler (P0 / P1)
export { GpuContext } from "./context.js";
export { isSoftwareAdapter } from "./device/acquire.js";
export type { PassTiming, Profiler } from "./kernel/profiler.js";

// ==================== algorithms (P1: the walking-skeleton diagnostic, spec 3.3)
export { degree } from "./algorithms/degree.js";

// ==================== layouts and the accelerator (P3)
export { createAccelerator } from "./accelerator.js";
export { createForceAtlas2 } from "./layouts/forceatlas2.js";
export { seedPositions } from "./layouts/seed.js";

// ==================== types: the accelerator surface and the CPU-package mirrors (spec 9.2, 9.3; D27)
export type {
    AcceleratorOptions,
    AlgorithmAccelerator,
    ApspResultLike,
    BellmanFordResultLike,
    BfsResultLike,
    CommunityResultLike,
    CorenessResultLike,
    CpuAlgorithmOptions,
    EdgeScoresResultLike,
    GpuAccelerator,
    HitsResultLike,
    LabelResultLike,
    LayoutAccelerator,
    LayoutSimulation,
    MstResultLike,
    PageRankResultLike,
    ScoresResultLike,
    SsspResultLike,
} from "./types/accelerator.js";

// ==================== types: context and capabilities (P0 / P1)
export type {
    AdapterInfoLike,
    AdapterSummary,
    GpuCaps,
    GpuContextOptions,
    LimitPolicy,
    PlanCaps,
    PlanLimits,
    ProbeOptions,
    ProbeResult,
    RaisableLimit,
} from "./types/context.js";

// ==================== types: layouts (P3)
export type {
    ForceAtlas2Stats,
    ForceAtlas2TraceRecord,
    GpuLayoutSimulation,
    GpuLayoutTuning,
    LayoutStatsBase,
    RunOptions,
} from "./types/layout.js";
export type {
    CommonLayoutOptions,
    ForceAtlas2Options,
    FruchtermanReingoldOptions,
    SimulationOptions,
    SpringElectricalOptions,
} from "./types/options.js";

// ==================== types: run options (P1)
export type { GpuRunOptions } from "./types/run.js";
