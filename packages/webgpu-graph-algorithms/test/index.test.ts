/**
 * The public barrel (contract 3.15, P0 list): exactly the pinned VALUE exports, no default export, and each
 * value is the same object its module exports. P1-T7 and P3-T3 extend the pinned list.
 */

import * as constants from "../src/constants.js";
import * as acquire from "../src/device/acquire.js";
import * as errors from "../src/errors.js";
import * as pkg from "../src/index.js";

/** The end-of-P0 value list of contract 3.15 (types are pinned by test/types/public-api.test-d.ts). */
const VALUE_EXPORTS = [
    // errors (3.1)
    "WebGpuGraphError",
    "isWebGpuGraphError",
    "hasErrorCode",
    "PASSTHROUGH_FORMAT_CODES",
    // constants (3.2)
    "WORKGROUP_SIZE",
    "MAX_WORKGROUPS_PER_DIM",
    "MAX_1D_ITEMS",
    "ARC_WINDOW_ALIGN",
    "STORAGE_ALIGN",
    "EXACT_MAX_NODES",
    // device (3.4)
    "isSoftwareAdapter",
];

describe("public barrel (contract 3.15)", () => {
    it("exports exactly the P0 value surface and no default export", () => {
        expect(Object.keys(pkg).sort()).toEqual([...VALUE_EXPORTS].sort());
        expect((pkg as Record<string, unknown>).default).toBeUndefined();
    });

    it("re-exports the same objects the modules export", () => {
        expect(pkg.WebGpuGraphError).toBe(errors.WebGpuGraphError);
        expect(pkg.isWebGpuGraphError).toBe(errors.isWebGpuGraphError);
        expect(pkg.hasErrorCode).toBe(errors.hasErrorCode);
        expect(pkg.PASSTHROUGH_FORMAT_CODES).toBe(errors.PASSTHROUGH_FORMAT_CODES);
        expect(pkg.WORKGROUP_SIZE).toBe(constants.WORKGROUP_SIZE);
        expect(pkg.MAX_WORKGROUPS_PER_DIM).toBe(constants.MAX_WORKGROUPS_PER_DIM);
        expect(pkg.MAX_1D_ITEMS).toBe(constants.MAX_1D_ITEMS);
        expect(pkg.ARC_WINDOW_ALIGN).toBe(constants.ARC_WINDOW_ALIGN);
        expect(pkg.STORAGE_ALIGN).toBe(constants.STORAGE_ALIGN);
        expect(pkg.EXACT_MAX_NODES).toBe(constants.EXACT_MAX_NODES);
        expect(pkg.isSoftwareAdapter).toBe(acquire.isSoftwareAdapter);
    });

    it("never exports the entries, the internal surface or a summarizer", () => {
        const keys = new Set(Object.keys(pkg));
        for (const name of [
            "createNodeGpu",
            "dawnFlags",
            "summarizeAdapter",
            "BufferUsage",
            "MapMode",
            "ShaderStage",
        ]) {
            expect(keys.has(name), name).toBe(false);
        }
    });
});
