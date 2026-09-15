/**
 * Every numeric constant the package and the WGSL prelude share (spec 3.1, 3.5, 5.2, 7.8, 7.14, 7.17;
 * contract 3.2). The prelude (src/kernel/prelude.ts, P1-T3) interpolates these values and never retypes them
 * as literals (the literal grep of spec 3.5 enforces it); the planners read them; test/device/constants.test.ts
 * pins them. graph-format's INVALID_INDEX is imported by the prelude directly and is deliberately not copied.
 */

/** Workgroup size of every 1D kernel; `WG = min(WORKGROUP_SIZE, caps.limits.maxComputeInvocationsPerWorkgroup)` at runtime (spec 5.1). */
export const WORKGROUP_SIZE = 256;
/** The spec minimum of maxComputeWorkgroupsPerDimension; asserted equal to the device limit at create() (spec 2.2, 5.2). */
export const MAX_WORKGROUPS_PER_DIM = 65535;
/** Items a 1D dispatch of WORKGROUP_SIZE covers: 65535 x 256 = 16,776,960, NOT 2^24 (design 10.6). */
export const MAX_1D_ITEMS = MAX_WORKGROUPS_PER_DIM * WORKGROUP_SIZE;
/** The largest u32 (the `min` identity of the u32 reduce); interpolated into the prelude as `U32_MAX` so no body types the literal (contract 4.1). */
export const U32_MAX = 0xffffffff;
/** Arc-window boundaries are multiples of 64 arcs = 256 bytes (design 10.6). */
export const ARC_WINDOW_ALIGN = 64;
/** Storage-binding offset alignment the package always honours (spec 2.6): the graph-format arena is 256-aligned. */
export const STORAGE_ALIGN = 256;
/** Stride of one UniformRing slot: minUniformBufferOffsetAlignment is 256 on every runtime the package targets (spec 5.3). */
export const UNIFORM_SLOT_BYTES = 256;
/** Exact-tier crossover default: CONSERVATIVE until G3 re-fixes it by the 7.8 rule (spec Q-6; the measured 7.6 curve predicts 32,768). */
export const EXACT_MAX_NODES = 16384;
/** Default number of MAP_READ staging buffers in the Readback ring (spec 4.4). */
export const DEFAULT_STAGING_SLOTS = 3;
/** Timestamp query-set size of the Profiler (spec 5.5: "a query set of 256 slots"; 2 slots per pass). */
export const PROFILER_QUERY_SLOTS = 256;
/** Byte size above which createBuffer runs inside an "out-of-memory" error scope (spec 5.7). */
export const OOM_SCOPE_THRESHOLD_BYTES = 16 * 1024 * 1024;
/** Idle buffers kept per (size class, usage) by the BufferPool (spec 4.4). */
export const POOL_MAX_IDLE_PER_CLASS = 4;
/** Smallest power-of-two pool class (spec 4.4). */
export const POOL_MIN_CLASS_BYTES = 4 * 1024;
/** Largest power-of-two pool class; above it the classes grow linearly (spec 4.4). */
export const POOL_MAX_POW2_CLASS_BYTES = 64 * 1024 * 1024;
/** The linear step of the pool classes above the largest power of two (spec 4.4). */
export const POOL_LINEAR_STEP_BYTES = 16 * 1024 * 1024;
/** Default `warnUnreleasedSnapshots` (spec 2.2, 4.1). */
export const DEFAULT_WARN_UNRELEASED_SNAPSHOTS = 2;
/** CONTRACT DECISION: the largest `iterations` a single step() records (the trace region and the uniform ring are sized by it); larger values are E_INVALID_ARGUMENT. */
export const MAX_ITERATIONS_PER_STEP = 256;
/** Bytes of one Fa2Trace record (spec 7.3: 32-byte records). */
export const TRACE_RECORD_BYTES = 32;
/** CONTRACT DECISION: the state header is padded to 256 bytes so the trace region that follows it in the same buffer starts at a legal 256-aligned binding offset (spec 7.3 says "128 + k x 32"; 128 is not a legal storage offset). */
export const STATE_HEADER_BYTES = 256;
/** Bytes of one per-workgroup partials record (spec 7.3: exactly 64 B). */
export const PARTIAL_BYTES = 64;
/** ForceAtlas2 defaults (spec 7.14, 7.17, 7.19). */
export const FA2_DEFAULTS: Readonly<{
    maxIter: 100;
    jitterTolerance: 1;
    scalingRatio: 2;
    gravity: 1;
    strongGravity: false;
    distributedAction: false;
    linlog: false;
    dissuadeHubs: false;
    dim: 2;
    scale: 1;
    settleThreshold: 0.001;
    settleWindow: 10;
    iterationsPerStep: 1;
    maxInFlight: 2;
}> = Object.freeze({
    maxIter: 100,
    jitterTolerance: 1,
    scalingRatio: 2,
    gravity: 1,
    strongGravity: false,
    distributedAction: false,
    linlog: false,
    dissuadeHubs: false,
    dim: 2,
    scale: 1,
    settleThreshold: 0.001,
    settleWindow: 10,
    iterationsPerStep: 1,
    maxInFlight: 2,
});
/** GPU-only layout tuning defaults (spec 7.14): repulsion "auto", exactMaxNodes EXACT_MAX_NODES, nearMax 64, deterministic true, gridMax2D 512, gridMax3D 128, extentFactor 6, compat "paper". */
export const LAYOUT_TUNING_DEFAULTS: Readonly<{
    repulsion: "auto";
    exactMaxNodes: number;
    nearMax: 64;
    deterministic: true;
    gridMax2D: 512;
    gridMax3D: 128;
    extentFactor: 6;
    compat: "paper";
}> = Object.freeze({
    repulsion: "auto",
    exactMaxNodes: EXACT_MAX_NODES,
    nearMax: 64,
    deterministic: true,
    gridMax2D: 512,
    gridMax3D: 128,
    extentFactor: 6,
    compat: "paper",
});
/** The distance floor `max(d, 0.01)` of spec 7.2; interpolated into the prelude (contract 4.1). */
export const FA2_DISTANCE_FLOOR = 0.01;
/** The square of FA2_DISTANCE_FLOOR. */
export const FA2_DISTANCE_FLOOR_SQ = 0.0001;
/** The coincident threshold `d^2 < 1e-8` of spec 7.2. */
export const FA2_COINCIDENT_SQ = 1e-8;
/** Bits of Fa2Params.flags (contract 4.4). */
export const FA2_FLAG_FIRST = 1;
