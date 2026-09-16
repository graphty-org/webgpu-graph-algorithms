/**
 * The compile-matrix seed (contract 5.5 compile.test.ts; spec 5.1, 11.3 "WGSL compile matrix"): every registry
 * entry composes and compiles WITH ITS DEFAULTS through PipelineCache on four lanes -- the real device with the
 * subgroup helpers when the adapter offers them, the real device forced to the workgroup-memory twin
 * (acquire({ subgroups: false })), Dawn's backend=null adapter (spec 5.1: compiles pipelines, runs nothing) and
 * its twin. This is the first time the five P1 bodies and the reduction-helper blocks of src/kernel/prelude.ts
 * meet Tint; P2-T2's wgsl-compile.test.ts adds every override combination of the OVERRIDE_MATRIX and the browser
 * leg. Entries with snippet slots (segmented-reduce at P2) compile with `v = 1.0;` in VALUE, so a later entry
 * joins this test without an edit. The setup's afterEach fails a test on any uncaptured error.
 */

import { type GpuContext } from "../../src/context.js";
import { composeWgsl } from "../../src/kernel/wgsl.js";
import { type KernelEntry, type KernelId, KERNELS, kernelSpec } from "../../src/kernels.js";
import { acquire, acquireNullBackend, requireGpu } from "../setup/gpu.js";

interface Lane {
    readonly name: string;
    /** The lane's context, created on first use and cached (the setup disposes it in afterAll). */
    readonly open: () => Promise<GpuContext>;
}

function cached(open: () => Promise<GpuContext>): () => Promise<GpuContext> {
    let promise: Promise<GpuContext> | null = null;
    return () => {
        if (promise === null) {
            promise = open();
        }
        return promise;
    };
}

const LANES: readonly Lane[] = [
    {
        name: "real device, subgroup helpers when the adapter offers them",
        open: cached(() => acquire({ label: "compile/real" })),
    },
    {
        name: "real device, workgroup-memory twin (subgroups: false)",
        open: cached(() => acquire({ label: "compile/real-twin", subgroups: false })),
    },
    {
        name: "Dawn backend=null, subgroup helpers when offered",
        open: cached(() => acquireNullBackend({ label: "compile/null" })),
    },
    {
        name: "Dawn backend=null, workgroup-memory twin (subgroups: false)",
        open: cached(() => acquireNullBackend({ label: "compile/null-twin", subgroups: false })),
    },
];

const P1_IDS: readonly KernelId[] = ["degree", "reduce", "fill", "fa2-repulsion-exact", "fa2-speed-finalize"];
const ALL_IDS = Object.keys(KERNELS) as KernelId[];

/** Snippet texts that let a body with marker slots compose with its defaults (only VALUE exists in P0-P3). */
const DEFAULT_SNIPPETS: Readonly<Record<string, string>> = { VALUE: "v = 1.0;" };

function defaultSnippets(entry: KernelEntry): Readonly<Record<string, string>> | undefined {
    if (entry.snippetSlots.length === 0) {
        return undefined;
    }
    const snippets: Record<string, string> = {};
    for (const slot of entry.snippetSlots) {
        const text = DEFAULT_SNIPPETS[slot];
        if (text === undefined) {
            throw new Error(`no default snippet for slot ${slot} of ${entry.id}; extend DEFAULT_SNIPPETS`);
        }
        snippets[slot] = text;
    }
    return snippets;
}

function maxGroup(entry: KernelEntry): number {
    return entry.bindings.reduce((acc, b) => Math.max(acc, b.group), 0);
}

describe("every registry entry compiles with its defaults (contract 5.5; spec 5.1)", () => {
    it("the five P1 entries are registered", () => {
        for (const id of P1_IDS) {
            expect(ALL_IDS).toContain(id);
        }
    });

    for (const lane of LANES) {
        describe(lane.name, () => {
            for (const id of ALL_IDS) {
                it(`${id}: composes for the lane's caps and compiles once through PipelineCache`, async (t) => {
                    requireGpu(t);
                    const ctx = await lane.open();
                    const entry = KERNELS[id];
                    const spec = kernelSpec(id, undefined, defaultSnippets(entry));
                    const twin = entry.needs.includes("subgroups");
                    const subgroups = twin && ctx.caps.features.has("subgroups");

                    const composed = composeWgsl(spec, ctx.caps);
                    expect(composed.id).toBe(id);
                    expect(composed.entryPoint).toBe(entry.entryPoint);
                    expect(composed.subgroups).toBe(subgroups);
                    expect(composed.code.includes("enable subgroups;")).toBe(subgroups);
                    if (twin) {
                        expect(composed.code.includes("sg_counter")).toBe(subgroups);
                        expect(composed.code.includes("wg_scratch_v")).toBe(!subgroups);
                    }
                    expect(composed.overrides.WG).toBe(ctx.workgroupSize);
                    const keyFields = ctx.pipelines.key(spec).split("|");
                    expect(keyFields[0]).toBe(id);
                    expect(keyFields[2]).toBe(subgroups ? "subgroups" : "");

                    const before = ctx.pipelines.size;
                    const pipeline = await ctx.pipelines.get(spec);
                    expect(pipeline).toBeDefined();
                    expect(ctx.pipelines.size).toBe(before + 1);
                    const again = await ctx.pipelines.get(spec);
                    expect(again).toBe(pipeline);
                    expect(ctx.pipelines.size).toBe(before + 1);

                    const kernel = await ctx.pipelines.kernel(spec);
                    expect(kernel.entryPoint).toBe(entry.entryPoint);
                    expect(kernel.workgroupSize).toBe(ctx.workgroupSize);
                    expect(kernel.layouts.length).toBe(maxGroup(entry) + 1);
                    expect(ctx.pipelines.size).toBe(before + 1);
                }, 60_000);
            }

            it("warm() over every default spec compiles nothing new once the entries are cached, and keys() lists one key per entry", async (t) => {
                requireGpu(t);
                const ctx = await lane.open();
                const specs = ALL_IDS.map((id) => kernelSpec(id, undefined, defaultSnippets(KERNELS[id])));
                const before = ctx.pipelines.size;
                await ctx.pipelines.warm(specs);
                expect(ctx.pipelines.size).toBe(before);
                expect(ctx.pipelines.size).toBe(ALL_IDS.length);
                expect(new Set(ctx.pipelines.keys()).size).toBe(ALL_IDS.length);
                for (const id of ALL_IDS) {
                    expect(
                        ctx.pipelines.keys().some((key) => key.startsWith(`${id}|`)),
                        id,
                    ).toBe(true);
                }
            }, 60_000);
        });
    }
});
