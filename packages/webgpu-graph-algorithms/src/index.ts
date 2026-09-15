/// <reference types="@webgpu/types" preserve="true" />
/**
 * The public barrel of @graphty/webgpu-graph-algorithms (spec 2.5, 3.3; contract 3.15): explicit named exports
 * only, values and `export type`, no star re-exports. test/index.test.ts pins the value list and
 * test/types/public-api.test-d.ts the type list. Nothing of src/browser/** or src/node/** is exported here (they
 * are the ./browser and ./node entries), and the internal surface (GraphResidency, BufferPool, Readback, Lease,
 * PipelineCache, Kernel, CommandBatch, UniformRing, UniformBlock, composeWgsl, KERNELS, ForceSimulation,
 * ForceAtlas2Model) never is. P1-T7 adds GpuContext / degree and their types; P3-T3 the layouts and the
 * accelerator.
 */

// ============================================================ constants (contract 3.2)
export {
    ARC_WINDOW_ALIGN,
    EXACT_MAX_NODES,
    MAX_1D_ITEMS,
    MAX_WORKGROUPS_PER_DIM,
    STORAGE_ALIGN,
    WORKGROUP_SIZE,
} from "./constants.js";
// ============================================================ device (contract 3.4)
export { isSoftwareAdapter } from "./device/acquire.js";
// ============================================================ errors (contract 3.1)
export {
    hasErrorCode,
    isWebGpuGraphError,
    PASSTHROUGH_FORMAT_CODES,
    WebGpuGraphError,
    type WebGpuGraphErrorCode,
} from "./errors.js";
// ============================================================ types (contract 3.3)
export { type AdapterInfoLike } from "./types/context.js";
