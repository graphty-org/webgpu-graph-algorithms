/**
 * The WGSL-side linear_id test at the 1D / 2D boundary (spec 5.2, 11.5; contract 5.5 linear-id.test.ts):
 * `fill` in mode 1 over LINEAR_ID_ITEMS = 16,776,961 words (68 MB, the first count that needs the 2D dispatch:
 * 65,536 workgroups of 256 exceed 65,535) on the current adapter; the LINEAR_ID_SAMPLES positions across the
 * boundary equal `i + LINEAR_ID_VALUE` and linearIdChecksum(result) equals the pinned LINEAR_ID_CHECKSUM, so the
 * same constants pin lavapipe, NVIDIA (here) and SwiftShader / NVIDIA-Chromium (test/browser/skeleton.test.ts,
 * P1-T7) to one bitwise result. The pure tests re-derive the pin in closed form so it never depends on an adapter.
 */

import { MAX_1D_ITEMS } from "../../src/constants.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { bindingOf, readU32, scratchBuffer } from "../helpers/device.js";
import { runKernel } from "../helpers/kernel.js";
import {
    LINEAR_ID_CHECKSUM,
    LINEAR_ID_ITEMS,
    LINEAR_ID_SAMPLES,
    LINEAR_ID_VALUE,
    linearIdChecksum,
} from "../helpers/linear-id.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { acquire, requireGpu } from "../setup/gpu.js";

const TWO_POW_32 = 4_294_967_296n;

describe("test/helpers/linear-id.ts (the constants the node and browser 17M-item tests share)", () => {
    it("pins ITEMS = MAX_1D_ITEMS + 1, an odd VALUE, the samples across the boundary and the checksum's closed form", () => {
        expect(MAX_1D_ITEMS).toBe(16_776_960);
        expect(LINEAR_ID_ITEMS).toBe(MAX_1D_ITEMS + 1);
        expect(LINEAR_ID_ITEMS).toBe(16_776_961);
        expect(LINEAR_ID_VALUE % 2).toBe(1);
        expect(LINEAR_ID_VALUE).toBe(1_000_003);
        expect(LINEAR_ID_SAMPLES).toEqual([0, 1, 255, 256, 16_776_959, 16_776_960, 16_776_960]);
        for (const i of LINEAR_ID_SAMPLES) {
            expect(i).toBeLessThan(LINEAR_ID_ITEMS);
        }
        // sum_{i < N} (i + V) = N V + N (N - 1) / 2, reduced mod 2^32 in BigInt so nothing rounds
        const n = BigInt(LINEAR_ID_ITEMS);
        const v = BigInt(LINEAR_ID_VALUE);
        const closed = Number((n * v + (n * (n - 1n)) / 2n) % TWO_POW_32);
        expect(closed).toBe(LINEAR_ID_CHECKSUM);
        expect(LINEAR_ID_CHECKSUM).toBe(877_493_955);
    });

    it("linearIdChecksum is the word sum modulo 2^32: hand-computed cases, wrap-around, and the full expected array", () => {
        expect(linearIdChecksum(new Uint32Array(0))).toBe(0);
        expect(linearIdChecksum(new Uint32Array([1, 2, 3]))).toBe(6);
        expect(linearIdChecksum(new Uint32Array([4_294_967_295, 1]))).toBe(0);
        expect(linearIdChecksum(new Uint32Array([4_294_967_295, 4_294_967_295]))).toBe(4_294_967_294);
        expect(linearIdChecksum(new Uint32Array([1_000_003, 1_000_004, 1_000_005]))).toBe(3_000_012);
        const expected = new Uint32Array(LINEAR_ID_ITEMS);
        for (let i = 0; i < expected.length; i++) {
            expected[i] = i + LINEAR_ID_VALUE;
        }
        expect(linearIdChecksum(expected)).toBe(LINEAR_ID_CHECKSUM);
    });
});

describe("fill mode 1 over LINEAR_ID_ITEMS words: the 2D dispatch and the prelude's linear_id (spec 5.2, 11.5)", () => {
    it("writes i + LINEAR_ID_VALUE at every sampled index, produces the pinned checksum, and is bitwise stable across two runs", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "linear-id" });
        const plan = plan1d(LINEAR_ID_ITEMS, ctx.workgroupSize, ctx.caps);
        expect(plan.items).toBe(LINEAR_ID_ITEMS);
        expect(plan.x).toBe(65_535);
        expect(plan.y).toBeGreaterThanOrEqual(2);
        expect(plan.x * plan.y * ctx.workgroupSize).toBeGreaterThanOrEqual(LINEAR_ID_ITEMS);
        const byteLength = LINEAR_ID_ITEMS * 4;
        expect(byteLength).toBe(67_107_844);
        expect(byteLength).toBeLessThanOrEqual(ctx.caps.limits.maxStorageBufferBindingSize);

        const dst = scratchBuffer(ctx, byteLength, "linear-id/dst");
        const params = { block: FILL_PARAMS, values: { count: LINEAR_ID_ITEMS, value: LINEAR_ID_VALUE, mode: 1 } };
        await runKernel(ctx, kernelSpec("fill"), { dst: bindingOf(dst) }, plan, params);
        const words = await readU32(ctx, dst, LINEAR_ID_ITEMS);
        expect(words.length).toBe(LINEAR_ID_ITEMS);
        for (const i of LINEAR_ID_SAMPLES) {
            expect(words[i], `word ${i}`).toBe(i + LINEAR_ID_VALUE);
        }
        expect(words[LINEAR_ID_ITEMS - 1]).toBe(LINEAR_ID_ITEMS - 1 + LINEAR_ID_VALUE);
        expect(linearIdChecksum(words)).toBe(LINEAR_ID_CHECKSUM);

        await runKernel(ctx, kernelSpec("fill"), { dst: bindingOf(dst) }, plan, params);
        const again = await readU32(ctx, dst, LINEAR_ID_ITEMS);
        expectBitwiseEqual(words, again, "second run of the 17M-item fill");
    }, 120_000);

    it("the last word is the only item of the second workgroup row at WG = 256 (the boundary the test exists for)", () => {
        // items 0 .. 65_535 x 256 - 1 are the first row of workgroups; item 65_535 x 256 = LINEAR_ID_ITEMS - 1 is
        // workgroup (0, 1), lane 0: linear_id = (0 + 1 x 65535) x 256 + 0
        expect(LINEAR_ID_ITEMS - 1).toBe(65_535 * 256);
        expect(LINEAR_ID_SAMPLES).toContain(65_535 * 256 - 1);
        expect(LINEAR_ID_SAMPLES).toContain(65_535 * 256);
    });
});
