/**
 * The numeric WebGPU constants of spec 2.1 rule 1 compared with the runtime globals Dawn installed (contract
 * 5.5), and every value of src/constants.ts pinned by hand (contract 3.2; the pins also keep knip's unused-export
 * check green for the constants nothing in src/ reads before P1).
 */

import {
    ARC_WINDOW_ALIGN,
    DEFAULT_STAGING_SLOTS,
    DEFAULT_WARN_UNRELEASED_SNAPSHOTS,
    EXACT_MAX_NODES,
    FA2_COINCIDENT_SQ,
    FA2_DEFAULTS,
    FA2_DISTANCE_FLOOR,
    FA2_DISTANCE_FLOOR_SQ,
    FA2_FLAG_FIRST,
    LAYOUT_TUNING_DEFAULTS,
    MAX_1D_ITEMS,
    MAX_ITERATIONS_PER_STEP,
    MAX_WORKGROUPS_PER_DIM,
    OOM_SCOPE_THRESHOLD_BYTES,
    PARTIAL_BYTES,
    POOL_LINEAR_STEP_BYTES,
    POOL_MAX_IDLE_PER_CLASS,
    POOL_MAX_POW2_CLASS_BYTES,
    POOL_MIN_CLASS_BYTES,
    PROFILER_QUERY_SLOTS,
    STATE_HEADER_BYTES,
    STORAGE_ALIGN,
    TRACE_RECORD_BYTES,
    U32_MAX,
    UNIFORM_SLOT_BYTES,
    WORKGROUP_SIZE,
} from "../../src/constants.js";
import { BufferUsage, MapMode, ShaderStage } from "../../src/device/webgpu-constants.js";
import { requireGpu } from "../setup/gpu.js";

/**
 * The runtime namespace as the runtime exposes it: a plain record of numbers (an object in browsers, a FUNCTION
 * object whose own properties carry the constants under Dawn-node 0.4.0 -- `typeof GPUBufferUsage === "function"`).
 */
function runtimeNamespace(name: "GPUBufferUsage" | "GPUMapMode" | "GPUShaderStage"): Record<string, number> {
    const value = (globalThis as Record<string, unknown>)[name];
    expect(
        value !== null && (typeof value === "object" || typeof value === "function"),
        `${name} on globalThis after createNodeGpu installed dawn.globals`,
    ).toBe(true);
    return value as Record<string, number>;
}

describe("webgpu-constants versus the runtime globals (spec 2.1 rule 1)", () => {
    it("BufferUsage equals GPUBufferUsage field by field", (t) => {
        requireGpu(t);
        const runtime = runtimeNamespace("GPUBufferUsage");
        for (const [name, bit] of Object.entries(BufferUsage)) {
            expect(runtime[name], `GPUBufferUsage.${name}`).toBe(bit);
        }
    });

    it("MapMode equals GPUMapMode field by field", (t) => {
        requireGpu(t);
        const runtime = runtimeNamespace("GPUMapMode");
        for (const [name, bit] of Object.entries(MapMode)) {
            expect(runtime[name], `GPUMapMode.${name}`).toBe(bit);
        }
    });

    it("ShaderStage equals GPUShaderStage field by field", (t) => {
        requireGpu(t);
        const runtime = runtimeNamespace("GPUShaderStage");
        for (const [name, bit] of Object.entries(ShaderStage)) {
            expect(runtime[name], `GPUShaderStage.${name}`).toBe(bit);
        }
    });
});

describe("webgpu-constants shape", () => {
    it("pins the WebGPU spec values", () => {
        expect(BufferUsage).toEqual({
            MAP_READ: 0x0001,
            MAP_WRITE: 0x0002,
            COPY_SRC: 0x0004,
            COPY_DST: 0x0008,
            INDEX: 0x0010,
            VERTEX: 0x0020,
            UNIFORM: 0x0040,
            STORAGE: 0x0080,
            INDIRECT: 0x0100,
            QUERY_RESOLVE: 0x0200,
        });
        expect(MapMode).toEqual({ READ: 0x0001, WRITE: 0x0002 });
        expect(ShaderStage).toEqual({ VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 });
    });

    it("every bit is a distinct power of two and the namespaces are frozen", () => {
        for (const namespace of [BufferUsage, MapMode, ShaderStage]) {
            expect(Object.isFrozen(namespace)).toBe(true);
            const bits = Object.values(namespace);
            expect(new Set(bits).size).toBe(bits.length);
            for (const bit of bits) {
                expect(Number.isInteger(Math.log2(bit)), `${bit} is a power of two`).toBe(true);
            }
        }
    });
});

describe("constants.ts (contract 3.2)", () => {
    it("pins the dispatch and alignment constants", () => {
        expect(WORKGROUP_SIZE).toBe(256);
        expect(MAX_WORKGROUPS_PER_DIM).toBe(65535);
        // 65535 x 256 = 16,776,960, NOT 2^24 = 16,777,216 (design 10.6)
        expect(MAX_1D_ITEMS).toBe(16776960);
        expect(MAX_1D_ITEMS).toBe(MAX_WORKGROUPS_PER_DIM * WORKGROUP_SIZE);
        expect(MAX_1D_ITEMS).toBeLessThan(2 ** 24);
        expect(U32_MAX).toBe(4294967295);
        expect(ARC_WINDOW_ALIGN).toBe(64);
        expect(STORAGE_ALIGN).toBe(256);
        expect(UNIFORM_SLOT_BYTES).toBe(256);
        // 64 arcs x 4 bytes = one 256-byte storage alignment unit
        expect(ARC_WINDOW_ALIGN * 4).toBe(STORAGE_ALIGN);
    });

    it("pins the memory and kernel-infrastructure constants", () => {
        // re-fixed at G3 by the spec 7.8 rule (docs/decisions/G3.md section 3): a ladder rung, so a power of two in [1024, 65536]
        expect(EXACT_MAX_NODES).toBe(32768);
        expect(EXACT_MAX_NODES).toBeGreaterThanOrEqual(1024);
        expect(EXACT_MAX_NODES).toBeLessThanOrEqual(65536);
        expect(Math.log2(EXACT_MAX_NODES) % 1).toBe(0);
        expect(DEFAULT_STAGING_SLOTS).toBe(3);
        expect(PROFILER_QUERY_SLOTS).toBe(256);
        expect(OOM_SCOPE_THRESHOLD_BYTES).toBe(16777216);
        expect(POOL_MAX_IDLE_PER_CLASS).toBe(4);
        expect(POOL_MIN_CLASS_BYTES).toBe(4096);
        expect(POOL_MAX_POW2_CLASS_BYTES).toBe(67108864);
        expect(POOL_LINEAR_STEP_BYTES).toBe(16777216);
        expect(DEFAULT_WARN_UNRELEASED_SNAPSHOTS).toBe(2);
        expect(MAX_ITERATIONS_PER_STEP).toBe(256);
        expect(TRACE_RECORD_BYTES).toBe(32);
        expect(STATE_HEADER_BYTES).toBe(256);
        // the trace region that follows the state header starts at a legal storage binding offset
        expect(STATE_HEADER_BYTES % STORAGE_ALIGN).toBe(0);
        expect(PARTIAL_BYTES).toBe(64);
    });

    it("pins the ForceAtlas2 defaults, the tuning defaults and the distance floor", () => {
        expect(FA2_DEFAULTS).toEqual({
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
        expect(Object.isFrozen(FA2_DEFAULTS)).toBe(true);
        expect(LAYOUT_TUNING_DEFAULTS).toEqual({
            repulsion: "auto",
            exactMaxNodes: 32768,
            nearMax: 64,
            deterministic: true,
            gridMax2D: 512,
            gridMax3D: 128,
            extentFactor: 6,
            compat: "paper",
        });
        expect(Object.isFrozen(LAYOUT_TUNING_DEFAULTS)).toBe(true);
        expect(LAYOUT_TUNING_DEFAULTS.exactMaxNodes).toBe(EXACT_MAX_NODES);
        expect(FA2_DISTANCE_FLOOR).toBe(0.01);
        expect(FA2_DISTANCE_FLOOR_SQ).toBe(0.0001);
        expect(FA2_DISTANCE_FLOOR_SQ).toBe(FA2_DISTANCE_FLOOR * FA2_DISTANCE_FLOOR);
        expect(FA2_COINCIDENT_SQ).toBe(1e-8);
        expect(FA2_FLAG_FIRST).toBe(1);
    });
});
