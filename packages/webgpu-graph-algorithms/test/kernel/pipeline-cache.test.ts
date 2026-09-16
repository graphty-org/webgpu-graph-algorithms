/**
 * PipelineCache (spec 5.1; contract 3.9; 5.5 row pipeline-cache.test.ts): key format and stability (JSON key order
 * independent), get twice -> one pipeline, a WGSL error -> E_SHADER_COMPILE { stage: "compile", messages } with
 * body-relative lines, warm compiles every spec once, keys(), plus a pipeline error that is not a compilation message
 * (-> E_VALIDATION), the Kernel cache and dispose(). Needs a device (acquire()).
 */

import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import { PipelineCache, pipelineKey } from "../../src/kernel/pipeline-cache.js";
import { UniformBlock } from "../../src/kernel/struct-block.js";
import { type WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { type PlanCaps } from "../../src/types/context.js";
import { CAPS_SPEC_DEFAULT } from "../helpers/caps-tables.js";
import { acquire, requireGpu } from "../setup/gpu.js";

async function catchRejection(promise: Promise<unknown>): Promise<WebGpuGraphError> {
    try {
        await promise;
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a rejection with a WebGpuGraphError");
}

const ADD_PARAMS = UniformBlock.define("AddParams", [
    ["count", "u32"],
    ["value", "u32"],
]);
const ADD_BODY = `@compute @workgroup_size(WG)
fn add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    dst[i] = src[i] + P.value;
}`;
// Test-only module ids are prefixed `test-` (PLAN DECISION 16): their keys land in the pipeline key log like any
// other, and P2-T2's coverage check (matrixCovers) ignores every key whose id segment is not a KernelId.
const ADD: WgslModuleSpec = {
    id: "test-add",
    body: ADD_BODY,
    bindings: [
        { group: 1, binding: 0, name: "src", kind: "storage-ro", wgslType: "array<u32>" },
        { group: 1, binding: 1, name: "dst", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "AddParams" },
    ],
    overrideDecls: [{ name: "DOUBLE", type: "bool", default: false }],
    overrides: {},
    needs: [],
    uniforms: [ADD_PARAMS],
};
const SUBGROUP_CAPS: PlanCaps = { ...CAPS_SPEC_DEFAULT, features: new Set(["subgroups"]) };

describe("pipelineKey / PipelineCache.key", () => {
    it("is id | stableJson(overrides) | present needs | hash(snippets), independent of the override key order", () => {
        expect(pipelineKey(ADD, CAPS_SPEC_DEFAULT)).toBe("test-add|{}||0");
        const a = pipelineKey({ ...ADD, overrides: { B: true, A: 1 } }, CAPS_SPEC_DEFAULT);
        const b = pipelineKey({ ...ADD, overrides: { A: 1, B: true } }, CAPS_SPEC_DEFAULT);
        expect(a).toBe('test-add|{"A":1,"B":true}||0');
        expect(b).toBe(a);
        expect(pipelineKey({ ...ADD, overrides: { A: 2, B: true } }, CAPS_SPEC_DEFAULT)).not.toBe(a);
    });

    it("lists a need only when the caps have the feature, and hashes the snippets", () => {
        const reducing: WgslModuleSpec = { ...ADD, needs: ["subgroups"] };
        expect(pipelineKey(reducing, CAPS_SPEC_DEFAULT)).toBe("test-add|{}||0");
        expect(pipelineKey(reducing, SUBGROUP_CAPS)).toBe("test-add|{}|subgroups|0");
        const withSnippet = pipelineKey({ ...ADD, snippets: { VALUE: "v = 1.0;" } }, CAPS_SPEC_DEFAULT);
        expect(withSnippet).toMatch(/^test-add\|\{\}\|\|[0-9a-f]{8}$/);
        expect(pipelineKey({ ...ADD, snippets: { VALUE: "v = 1.0;" } }, CAPS_SPEC_DEFAULT)).toBe(withSnippet);
        expect(pipelineKey({ ...ADD, snippets: { VALUE: "v = 2.0;" } }, CAPS_SPEC_DEFAULT)).not.toBe(withSnippet);
        expect(pipelineKey({ ...ADD, snippets: {} }, CAPS_SPEC_DEFAULT)).toBe("test-add|{}||0");
    });

    it("the cache's key() agrees with pipelineKey on the context's caps", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        expect(ctx.pipelines.key(ADD)).toBe(pipelineKey(ADD, ctx.caps));
    });
});

describe("PipelineCache", () => {
    it("get twice -> one pipeline; concurrent gets share one compile; kernel() is cached with it", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        expect(cache.size).toBe(0);
        const first = await cache.get(ADD);
        const second = await cache.get(ADD);
        expect(second).toBe(first);
        expect(cache.size).toBe(1);
        expect(cache.keys()).toEqual([cache.key(ADD)]);
        const [c1, c2] = await Promise.all([
            cache.get({ ...ADD, overrides: { DOUBLE: true } }),
            cache.get({ ...ADD, overrides: { DOUBLE: true } }),
        ]);
        expect(c2).toBe(c1);
        expect(c1).not.toBe(first);
        expect(cache.size).toBe(2);
        const kernel = await cache.kernel(ADD);
        expect(kernel.pipeline).toBe(first);
        expect(await cache.kernel(ADD)).toBe(kernel);
        expect(kernel.layouts).toHaveLength(3);
        expect(kernel.entryPoint).toBe("add");
        expect(kernel.workgroupSize).toBe(Math.min(256, ctx.caps.limits.maxComputeInvocationsPerWorkgroup));
        expect(cache.layoutsOf(cache.key(ADD))).toBe(kernel.layouts);
        expect(cache.layoutsOf("nope")).toBeNull();
        expect(first.label).toBe(cache.key(ADD));
    });

    it("a body with a WGSL error -> E_SHADER_COMPILE { id, stage: 'compile', messages } with body-relative lines", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        // body line 3 is wrong: a u32 initialised from an abstract float
        const broken: WgslModuleSpec = {
            ...ADD,
            id: "test-broken",
            body: ADD_BODY.replace("let i = linear_id(wid, lid.x);", "let i: u32 = 1.5;"),
        };
        const err = await catchRejection(cache.get(broken));
        expect(err.code).toBe("E_SHADER_COMPILE");
        expect(err.details).toMatchObject({ id: "test-broken", stage: "compile" });
        const messages = err.details.messages as string[];
        expect(messages.length).toBeGreaterThanOrEqual(1);
        expect(messages.join("\n")).toMatch(/abstract-float|u32/);
        expect(err.details.lines).toEqual([3]);
        expect(cache.size).toBe(0);
        expect(cache.keys()).toEqual([]);
        // a failed compile is not cached: fixing the body under the same key compiles
        await cache.get({ ...broken, body: ADD_BODY });
        expect(cache.size).toBe(1);
    });

    it("a compose error rejects get() with stage 'compose' before any device work", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        const err = await catchRejection(cache.get({ ...ADD, overrides: { BOGUS: 1 } }));
        expect(err.code).toBe("E_SHADER_COMPILE");
        expect(err.details).toMatchObject({ id: "test-add", stage: "compose", slot: "override:BOGUS" });
        expect(cache.size).toBe(0);
    });

    it("a pipeline error that is not a compilation message (an oversized WG) -> E_VALIDATION", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        const err = await catchRejection(cache.get({ ...ADD, overrides: { WG: 65536 } }));
        expect(err.code).toBe("E_VALIDATION");
        expect(err.details.label).toBe("test-add|pipeline");
        expect(String(err.details.message)).toMatch(/workgroup/i);
        expect(cache.size).toBe(0);
    });

    it("warm compiles every spec once and keys() lists them in creation order", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        const variant: WgslModuleSpec = { ...ADD, overrides: { DOUBLE: true } };
        const other: WgslModuleSpec = { ...ADD, id: "test-other" };
        await cache.warm([ADD, variant]);
        expect(cache.size).toBe(2);
        const [a, v] = await Promise.all([cache.get(ADD), cache.get(variant)]);
        await cache.warm([ADD, variant, other, ADD]);
        expect(cache.size).toBe(3);
        expect(await cache.get(ADD)).toBe(a);
        expect(await cache.get(variant)).toBe(v);
        expect(cache.keys()).toEqual([cache.key(ADD), cache.key(variant), cache.key(other)]);
    });

    it("dispose() forgets the pipelines and refuses further compiles with E_DISPOSED", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const cache = new PipelineCache(ctx.device, ctx.caps);
        await cache.get(ADD);
        expect(cache.size).toBe(1);
        cache.dispose();
        expect(cache.size).toBe(0);
        expect(cache.layoutsOf(cache.key(ADD))).toBeNull();
        const err = await catchRejection(cache.get(ADD));
        expect(err.code).toBe("E_DISPOSED");
        expect((await catchRejection(cache.warm([ADD]))).code).toBe("E_DISPOSED");
    });
});
