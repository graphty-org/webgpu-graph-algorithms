/**
 * The public barrel's VALUE list, pinned (contract 3.15, 5.5): P0's errors / constants / isSoftwareAdapter, P1's
 * GpuContext and degree, P3's layout factory, accelerator, default tables and seeder; each value is the same object
 * its module exports; no default export; the entries, the internal surface and the P4+ names never reach the root.
 * Types are pinned by test/types/public-api.test-d.ts under the strict-consumer compile.
 */

import { createAccelerator } from "../src/accelerator.js";
import { degree } from "../src/algorithms/degree.js";
import * as constants from "../src/constants.js";
import { GpuContext } from "../src/context.js";
import * as acquire from "../src/device/acquire.js";
import * as errors from "../src/errors.js";
import * as api from "../src/index.js";
import { createForceAtlas2 } from "../src/layouts/forceatlas2.js";
import { seedPositions } from "../src/layouts/seed.js";

/**
 * The VALUE exports of contract 3.15 at the end of P3 (the P0, P1 and P3 lists; P2 added none). Types are pinned
 * by test/types/public-api.test-d.ts, which the strict-consumer compile inside `pnpm run lint` type-checks.
 */
const VALUE_EXPORTS = [
    // P0 (contract 3.15): the error class, its brand checks, the constants, the adapter classifier
    "WebGpuGraphError",
    "isWebGpuGraphError",
    "hasErrorCode",
    "PASSTHROUGH_FORMAT_CODES",
    "WORKGROUP_SIZE",
    "MAX_WORKGROUPS_PER_DIM",
    "MAX_1D_ITEMS",
    "ARC_WINDOW_ALIGN",
    "STORAGE_ALIGN",
    "EXACT_MAX_NODES",
    "isSoftwareAdapter",
    // P1: the context and the walking-skeleton diagnostic
    "GpuContext",
    "degree",
    // P3: the layout factory, the accelerator, the two default tables, the seeder
    "createForceAtlas2",
    "createAccelerator",
    "FA2_DEFAULTS",
    "LAYOUT_TUNING_DEFAULTS",
    "seedPositions",
];

/**
 * Names contract 3.15 says are NEVER exported from the root: the entries, the @internal surface, P4+ / P7+ names,
 * plus the entries' helpers, the summarizer and the numeric WebGPU constants (the P0 / P1 pins, retained).
 */
const NEVER_EXPORTED = [
    // the ./node and ./browser entries (their own subpaths)
    "createNodeGpu",
    "createNodeGpuContext",
    "probeNodeWebGpu",
    "dawnFlags",
    "probeBrowserWebGpu",
    "requestGpuContext",
    // the @internal surface (tests import these from their files)
    "GraphResidency",
    "BufferPool",
    "Readback",
    "Lease",
    "PipelineCache",
    "Kernel",
    "CommandBatch",
    "UniformRing",
    "UniformBlock",
    "composeWgsl",
    "KERNELS",
    "kernelSpec",
    "ForceSimulation",
    "ForceAtlas2Model",
    "RepulsionExact",
    "Lcg",
    "resolveForceAtlas2Options",
    "resolveLayoutTuning",
    "Profiler", // exported as a TYPE only (3.15)
    "summarizeAdapter",
    "BufferUsage",
    "MapMode",
    "ShaderStage",
    // P4+ / P5 / P7+
    "calibrateLayout",
    "createFruchtermanReingold",
    "createSpringElectrical",
    "pageRank",
    "connectedComponents",
];

describe("public barrel (contract 3.15; spec 3.3, 11.3 row 'Build output')", () => {
    it("exports exactly the P3 value list and no default export", () => {
        expect(Object.keys(api).sort()).toEqual([...VALUE_EXPORTS].sort());
        expect((api as Record<string, unknown>).default).toBeUndefined();
    });

    it("re-exports the P0, P1 and P3 values as the very objects their modules export", () => {
        expect(api.WebGpuGraphError).toBe(errors.WebGpuGraphError);
        expect(api.isWebGpuGraphError).toBe(errors.isWebGpuGraphError);
        expect(api.hasErrorCode).toBe(errors.hasErrorCode);
        expect(api.PASSTHROUGH_FORMAT_CODES).toBe(errors.PASSTHROUGH_FORMAT_CODES);
        expect(api.WORKGROUP_SIZE).toBe(constants.WORKGROUP_SIZE);
        expect(api.MAX_WORKGROUPS_PER_DIM).toBe(constants.MAX_WORKGROUPS_PER_DIM);
        expect(api.MAX_1D_ITEMS).toBe(constants.MAX_1D_ITEMS);
        expect(api.ARC_WINDOW_ALIGN).toBe(constants.ARC_WINDOW_ALIGN);
        expect(api.STORAGE_ALIGN).toBe(constants.STORAGE_ALIGN);
        expect(api.EXACT_MAX_NODES).toBe(constants.EXACT_MAX_NODES);
        expect(api.FA2_DEFAULTS).toBe(constants.FA2_DEFAULTS);
        expect(api.LAYOUT_TUNING_DEFAULTS).toBe(constants.LAYOUT_TUNING_DEFAULTS);
        expect(api.isSoftwareAdapter).toBe(acquire.isSoftwareAdapter);
        expect(api.GpuContext).toBe(GpuContext);
        expect(api.degree).toBe(degree);
        expect(api.createForceAtlas2).toBe(createForceAtlas2);
        expect(api.createAccelerator).toBe(createAccelerator);
        expect(api.seedPositions).toBe(seedPositions);
        expect(typeof api.WebGpuGraphError).toBe("function");
        expect(typeof api.isWebGpuGraphError).toBe("function");
        expect(typeof api.hasErrorCode).toBe("function");
        expect(typeof api.isSoftwareAdapter).toBe("function");
        expect(typeof api.GpuContext.probe).toBe("function");
        expect(typeof api.GpuContext.create).toBe("function");
        expect(typeof api.GpuContext.from).toBe("function");
        expect(Object.isFrozen(api.PASSTHROUGH_FORMAT_CODES)).toBe(true);
    });

    it("never exports the entries, the @internal surface or a P4+ name", () => {
        for (const name of NEVER_EXPORTED) {
            expect(name in api, name).toBe(false);
        }
    });

    it("carries the P3 default tables with the values of contract 3.2", () => {
        expect(api.FA2_DEFAULTS).toEqual({
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
        expect(api.LAYOUT_TUNING_DEFAULTS).toEqual({
            repulsion: "auto",
            exactMaxNodes: api.EXACT_MAX_NODES,
            nearMax: 64,
            deterministic: true,
            gridMax2D: 512,
            gridMax3D: 128,
            extentFactor: 6,
            compat: "paper",
        });
        expect(Object.isFrozen(api.FA2_DEFAULTS)).toBe(true);
        expect(Object.isFrozen(api.LAYOUT_TUNING_DEFAULTS)).toBe(true);
        expect(api.MAX_1D_ITEMS).toBe(16_776_960);
        expect(api.MAX_1D_ITEMS).toBe(api.MAX_WORKGROUPS_PER_DIM * api.WORKGROUP_SIZE);
    });
});
