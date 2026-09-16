/**
 * The bind-group budget (spec 3.5, 11.3; contract 3.10.1, 5.5 P2): every bind-group-layout descriptor derived from
 * every KERNELS entry has at most 8 storage entries per (compute) stage -- the WebGPU core default no kernel may
 * exceed; a larger kernel is split, never given a raised limit -- and exactly the per-kernel counts the contract
 * tabulates; uniform entries take a dynamic offset (the UniformRing slot); graph kernels bind the four group-0 slots.
 * Also here (spec 6 / R-8, the faked-caps table with min != max): no planner or composer output depends on the
 * CAPS_INTEL_XE subgroup pair beyond the two SUBGROUP_MIN / SUBGROUP_MAX pipeline constants.
 */

import { ShaderStage } from "../../src/device/webgpu-constants.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { bindGroupLayoutDescriptors, composeWgsl, type WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { type KernelId, KERNELS, kernelSpec } from "../../src/kernels.js";
import { planUpload } from "../../src/memory/upload-plan.js";
import { CAPS_INTEL_XE, CAPS_SPEC_DEFAULT } from "../helpers/caps-tables.js";
import { fixture } from "../helpers/graphs.js";

/** The storage-buffer counts of contract 3.10.1 (read-only and read-write together); P3 ids are listed ahead of their entries. */
const STORAGE_COUNTS: Readonly<Record<string, number | undefined>> = {
    degree: 5,
    reduce: 2,
    fill: 1,
    "segmented-reduce": 5,
    "fa2-stats-finalize": 3,
    "fa2-attraction": 6,
    "fa2-repulsion-exact": 6,
    "fa2-speed-finalize": 3,
    "fa2-integrate": 6,
    "fa2-to-scene": 2,
};

/** A spec of an entry with its defaults; an entry with snippet slots gets a trivial VALUE (the layout ignores it). */
function specOf(id: KernelId): WgslModuleSpec {
    const entry = KERNELS[id];
    const snippets = entry.snippetSlots.length > 0 ? { VALUE: "v = 1.0;" } : undefined;
    return kernelSpec(id, {}, snippets);
}

function entriesOf(descriptors: readonly GPUBindGroupLayoutDescriptor[]): GPUBindGroupLayoutEntry[] {
    const all: GPUBindGroupLayoutEntry[] = [];
    for (const descriptor of descriptors) {
        all.push(...descriptor.entries);
    }
    return all;
}

function isStorage(entry: GPUBindGroupLayoutEntry): boolean {
    const type = entry.buffer?.type;
    return type === "storage" || type === "read-only-storage";
}

function isUniform(entry: GPUBindGroupLayoutEntry): boolean {
    return entry.buffer !== undefined && (entry.buffer.type === undefined || entry.buffer.type === "uniform");
}

describe("bind-group budget", () => {
    const ids = Object.keys(KERNELS) as KernelId[];

    it("every registered id has a row in the 3.10.1 table", () => {
        for (const id of ids) {
            expect(STORAGE_COUNTS[id], id).toBeDefined();
        }
        expect(ids).toContain("segmented-reduce");
    });

    for (const id of ids) {
        it(`${id}: <= 8 storage buffers per stage and exactly the 3.10.1 count`, () => {
            const spec = specOf(id);
            const descriptors = bindGroupLayoutDescriptors(spec);
            const entries = entriesOf(descriptors);
            const storage = entries.filter(isStorage);
            expect(storage.length).toBe(STORAGE_COUNTS[id]);
            expect(storage.length).toBeLessThanOrEqual(8);
            expect(storage.length).toBeLessThanOrEqual(CAPS_SPEC_DEFAULT.limits.maxStorageBuffersPerShaderStage);
            // one entry per BindingDecl, all compute-visible, the uniform ones dynamic
            expect(entries.length).toBe(spec.bindings.length);
            for (const entry of entries) {
                expect(entry.visibility).toBe(ShaderStage.COMPUTE);
                if (isUniform(entry)) {
                    expect(entry.buffer?.hasDynamicOffset).toBe(true);
                }
            }
            // descriptors cover groups 0..maxGroup; empty groups have no entries
            const maxGroup = Math.max(...spec.bindings.map((b) => b.group));
            expect(descriptors.length).toBe(maxGroup + 1);
            // the graph group is always the four slots rowPtr / colIdx / weights / perm, all read-only
            const graph = spec.bindings.filter((b) => b.group === 0);
            if (graph.length > 0) {
                expect(graph.map((b) => b.name)).toEqual(["rowPtr", "colIdx", "weights", "perm"]);
                const group0 = [...descriptors[0].entries];
                expect(group0.length).toBe(4);
                for (const entry of group0) {
                    expect(entry.buffer?.type).toBe("read-only-storage");
                }
            }
        });
    }
});

describe("CAPS_INTEL_XE (spec 6: min != max)", () => {
    it("the dispatch and upload planners give the spec-default answer under the 8 / 32 table", () => {
        for (const items of [0, 1, 255, 256, 16_776_960, 16_776_961]) {
            expect(plan1d(items, 256, CAPS_INTEL_XE), `plan1d(${items})`).toEqual(
                plan1d(items, 256, CAPS_SPEC_DEFAULT),
            );
        }
        const { snapshot } = fixture("karate");
        expect(planUpload(snapshot, CAPS_INTEL_XE, ["rowPtr", "colIdx", "weights"])).toEqual(
            planUpload(snapshot, CAPS_SPEC_DEFAULT, ["rowPtr", "colIdx", "weights"]),
        );
    });

    it("the composer passes the pair through as the two pipeline constants and nothing else changes", () => {
        const spec = kernelSpec("reduce");
        const xe = composeWgsl(spec, CAPS_INTEL_XE);
        expect(xe.subgroups).toBe(true);
        expect(xe.overrides.SUBGROUP_MIN).toBe(8);
        expect(xe.overrides.SUBGROUP_MAX).toBe(32);
        // the same features with a uniform 32-wide subgroup: identical text, only the two constants differ (the
        // subgroup scratch is sized by SUBGROUP_MIN at pipeline creation, contract 4.3)
        const uniform = composeWgsl(spec, { ...CAPS_INTEL_XE, subgroupMinSize: 32, subgroupMaxSize: 32 });
        expect(uniform.code).toBe(xe.code);
        expect(uniform.bodyLine).toBe(xe.bodyLine);
        expect({ ...uniform.overrides, SUBGROUP_MIN: 8, SUBGROUP_MAX: 32 }).toEqual(xe.overrides);
        // the twin (no feature) ignores the pair entirely
        const twin = composeWgsl(spec, CAPS_SPEC_DEFAULT);
        expect(twin.subgroups).toBe(false);
        expect(twin.overrides.SUBGROUP_MIN).toBeUndefined();
        expect(twin.overrides.SUBGROUP_MAX).toBeUndefined();
    });
});
