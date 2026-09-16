/**
 * The negative uniform-layout test of spec 5.3 / R-10 on Dawn-node: the WGSL 14.4.5 "Invalid" example (a 4-byte
 * struct-typed member `a: BadInner { x: f32 }` followed by `b: f32` at offset 4: without
 * `uniform_buffer_standard_layout` the gap must be at least roundUp(16, SizeOf(S)) = 16) is ACCEPTED here because
 * Dawn-node exposes the language feature, while UniformBlock's generated text (a flat equivalent) compiles everywhere.
 * The assertion is `accepted === wgslFeatures.has("uniform_buffer_standard_layout")`, which the browser twin
 * (test/browser/uniform-layout.test.ts) pins to "rejected" on Chromium 139. The outcome is printed for the G2 record.
 */

import { UniformBlock } from "../../src/kernel/struct-block.js";
import { withContext } from "../helpers/device.js";
import { requireGpu } from "../setup/gpu.js";

const BAD_WGSL = `struct BadInner { x: f32 }
struct BadParams { a: BadInner, b: f32, tail: vec4f }
@group(0) @binding(0) var<uniform> P: BadParams;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() {
    result[0] = P.a.x + P.b + P.tail.w;
}`;

const GOOD = UniformBlock.define("GoodParams", [
    ["v", "vec4f"],
    ["b", "f32"],
    ["tail", "vec4f"],
]);

const GOOD_WGSL = `${GOOD.wgsl}
@group(0) @binding(0) var<uniform> P: GoodParams;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() {
    result[0] = P.v.x + P.b + P.tail.w;
}`;

interface CompileOutcome {
    readonly ok: boolean;
    readonly messages: readonly string[];
}

/** Compiles a raw module and a pipeline over it inside a validation scope; every error (compilation info, pipeline rejection, scope) is collected. */
async function compileRaw(device: GPUDevice, code: string, label: string): Promise<CompileOutcome> {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const messages = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    let pipelineError: string | null = null;
    if (messages.length === 0) {
        try {
            await device.createComputePipelineAsync({ label, layout: "auto", compute: { module, entryPoint: "main" } });
        } catch (error) {
            pipelineError = error instanceof Error ? error.message : `non-error thrown (${typeof error})`;
        }
    }
    const scoped = await device.popErrorScope();
    const all = [
        ...messages,
        ...(pipelineError === null ? [] : [pipelineError]),
        ...(scoped === null ? [] : [scoped.message]),
    ];
    return { ok: all.length === 0, messages: all };
}

describe("uniform layout: hand-written misaligned struct vs the generated one (spec 5.3, R-10)", () => {
    it("the generated block has the strict offsets", () => {
        expect(GOOD.offsetOf("v")).toBe(0);
        expect(GOOD.offsetOf("b")).toBe(16);
        expect(GOOD.offsetOf("tail")).toBe(32);
        expect(GOOD.byteLength).toBe(48);
        expect(GOOD.wgsl).toContain("struct GoodParams");
    });

    it("the generated struct compiles; the hand-written one is accepted iff uniform_buffer_standard_layout is present (Dawn-node: accepted)", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const relaxed = ctx.caps.wgslFeatures.has("uniform_buffer_standard_layout");
            const good = await compileRaw(ctx.device, GOOD_WGSL, "test/uniform-layout/good");
            expect(good.messages).toEqual([]);
            expect(good.ok).toBe(true);
            const bad = await compileRaw(ctx.device, BAD_WGSL, "test/uniform-layout/bad");
            console.warn(
                `[uniform-layout] runtime=${ctx.caps.runtime} uniform_buffer_standard_layout=${relaxed} hand-written struct accepted=${bad.ok}${bad.ok ? "" : `: ${bad.messages.join(" | ")}`}`,
            );
            expect(bad.ok).toBe(relaxed);
            // Dawn-node 0.4.0 exposes the language feature (note 05 section 2.4): the 14.4.5 Invalid struct passes here, which is
            // exactly why the generator exists and why the browser twin must run on Chromium
            expect(relaxed).toBe(true);
            expect(bad.ok).toBe(true);
        });
    });
});
