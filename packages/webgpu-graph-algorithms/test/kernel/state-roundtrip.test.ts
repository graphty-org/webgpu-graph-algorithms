/**
 * The state block round trip of spec 5.3 / 11.3 ("the state block round-trips host -> kernel -> host on both
 * runtimes"): Fa2State is written by the host through FA2_STATE.write, incremented by a one-workgroup kernel that
 * touches every field, and read back through FA2_STATE.read -- every field round-trips at the 3.10.2 offsets. The
 * written values are exactly representable in f32 and +1 keeps them exact, so the expected table is hand-written.
 * The browser twin is test/browser/state-roundtrip.test.ts (same body, same tables; keep the two in sync).
 * The ad hoc kernel compiles through a PRIVATE PipelineCache so the override-matrix coverage teardown (P2-T2) never
 * sees a test-only id.
 */

import { STATE_HEADER_BYTES } from "../../src/constants.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { PipelineCache } from "../../src/kernel/pipeline-cache.js";
import type { UniformValues } from "../../src/kernel/struct-block.js";
import type { WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { FA2_STATE } from "../../src/kernels.js";
import { bindingOf, uploadBuffer, withContext } from "../helpers/device.js";
import { requireGpu } from "../setup/gpu.js";

/** Every Fa2State field + 1 (u32 fields + 1u, vec4f fields + (1, 1, 1, 1)); early return is legal: no barrier follows (4.4 rule 1). */
const BUMP_BODY = `@compute @workgroup_size(WG)
fn bump(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; }
    let one = vec4f(1.0, 1.0, 1.0, 1.0);
    S.speed = S.speed + 1.0;
    S.speedEfficiency = S.speedEfficiency + 1.0;
    S.swing = S.swing + 1.0;
    S.traction = S.traction + 1.0;
    S.centroid = S.centroid + one;
    S.rmsRadius = S.rmsRadius + 1.0;
    S.radius = S.radius + 1.0;
    S.meanDisplacement = S.meanDisplacement + 1.0;
    S.iteration = S.iteration + 1u;
    S.min = S.min + one;
    S.max = S.max + one;
    S.gridMin = S.gridMin + one;
    S.eps = S.eps + 1.0;
    S.settledCount = S.settledCount + 1u;
    S.outsideGrid = S.outsideGrid + 1u;
    S.maxCellOccupancy = S.maxCellOccupancy + 1u;
    S.reserved0 = S.reserved0 + one;
    S.reserved1 = S.reserved1 + one;
    S.reserved2 = S.reserved2 + one;
    S.reserved3 = S.reserved3 + one;
    S.reserved4 = S.reserved4 + one;
    S.reserved5 = S.reserved5 + one;
    S.reserved6 = S.reserved6 + one;
    S.reserved7 = S.reserved7 + one;
    S.reserved8 = S.reserved8 + one;
}`;

const BUMP_SPEC: WgslModuleSpec = {
    id: "test-state-bump",
    body: BUMP_BODY,
    bindings: [{ group: 1, binding: 0, name: "S", kind: "storage", wgslType: "Fa2State" }],
    overrideDecls: [],
    overrides: {},
    needs: [],
    uniforms: [FA2_STATE],
};

/** Written by the host: every value exact in f32, and +1 stays exact. */
const WRITTEN: UniformValues = {
    speed: 1.5,
    speedEfficiency: 0.25,
    swing: 3,
    traction: 4,
    centroid: [1, 2, 3, 4],
    rmsRadius: 5.5,
    radius: 6.5,
    meanDisplacement: 0.125,
    iteration: 7,
    min: [-1, -2, -3, 0],
    max: [8, 9, 10, 11],
    gridMin: [0.5, 0.5, 0.5, 2],
    eps: 0.0625,
    settledCount: 9,
    outsideGrid: 10,
    maxCellOccupancy: 11,
    reserved0: [0, 0.5, 100, 0.25],
    reserved1: [1, 1.5, 99, 0.25],
    reserved2: [2, 2.5, 98, 0.25],
    reserved3: [3, 3.5, 97, 0.25],
    reserved4: [4, 4.5, 96, 0.25],
    reserved5: [5, 5.5, 95, 0.25],
    reserved6: [6, 6.5, 94, 0.25],
    reserved7: [7, 7.5, 93, 0.25],
    reserved8: [8, 8.5, 92, 0.25],
};

/** WRITTEN + 1 on every lane, hand-computed. */
const EXPECTED: UniformValues = {
    speed: 2.5,
    speedEfficiency: 1.25,
    swing: 4,
    traction: 5,
    centroid: [2, 3, 4, 5],
    rmsRadius: 6.5,
    radius: 7.5,
    meanDisplacement: 1.125,
    iteration: 8,
    min: [0, -1, -2, 1],
    max: [9, 10, 11, 12],
    gridMin: [1.5, 1.5, 1.5, 3],
    eps: 1.0625,
    settledCount: 10,
    outsideGrid: 11,
    maxCellOccupancy: 12,
    reserved0: [1, 1.5, 101, 1.25],
    reserved1: [2, 2.5, 100, 1.25],
    reserved2: [3, 3.5, 99, 1.25],
    reserved3: [4, 4.5, 98, 1.25],
    reserved4: [5, 5.5, 97, 1.25],
    reserved5: [6, 6.5, 96, 1.25],
    reserved6: [7, 7.5, 95, 1.25],
    reserved7: [8, 8.5, 94, 1.25],
    reserved8: [9, 9.5, 93, 1.25],
};

/** The 3.10.2 byte offsets. */
const OFFSETS: Readonly<Record<string, number>> = {
    speed: 0,
    speedEfficiency: 4,
    swing: 8,
    traction: 12,
    centroid: 16,
    rmsRadius: 32,
    radius: 36,
    meanDisplacement: 40,
    iteration: 44,
    min: 48,
    max: 64,
    gridMin: 80,
    eps: 96,
    settledCount: 100,
    outsideGrid: 104,
    maxCellOccupancy: 108,
    reserved0: 112,
    reserved1: 128,
    reserved2: 144,
    reserved3: 160,
    reserved4: 176,
    reserved5: 192,
    reserved6: 208,
    reserved7: 224,
    reserved8: 240,
};

function hostBytes(): ArrayBuffer {
    const bytes = new ArrayBuffer(STATE_HEADER_BYTES);
    FA2_STATE.write(new DataView(bytes), WRITTEN);
    return bytes;
}

describe("Fa2State round trip host -> kernel -> host (spec 5.3, 11.3)", () => {
    it("FA2_STATE lays every field out at the 3.10.2 offsets inside 256 bytes", () => {
        expect(FA2_STATE.name).toBe("Fa2State");
        expect(FA2_STATE.layout).toBe("storage");
        expect(FA2_STATE.byteLength).toBe(STATE_HEADER_BYTES);
        expect(FA2_STATE.byteLength).toBe(256);
        for (const [field, offset] of Object.entries(OFFSETS)) {
            expect(FA2_STATE.offsetOf(field), field).toBe(offset);
        }
        expect(FA2_STATE.fields.length).toBe(25);
        // the host-side writer and reader agree before any device is involved
        expect(FA2_STATE.read(new DataView(hostBytes()))).toEqual(WRITTEN);
    });

    it("every field round-trips through a one-workgroup kernel that adds 1; twice bitwise", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const cache = new PipelineCache(ctx.device, ctx.caps);
            const kernel = await cache.kernel(BUMP_SPEC);
            const plan = plan1d(1, ctx.workgroupSize, ctx.caps);
            const results: number[][] = [];
            for (let run = 0; run < 2; run++) {
                const state = uploadBuffer(ctx, new Uint8Array(hostBytes()), `test/state/${run}`);
                const batch = new CommandBatch(ctx, `state-roundtrip-${run}`);
                kernel.dispatch(batch.pass("bump"), kernel.bind({ S: bindingOf(state) }), plan);
                const request = batch.readback(state, 0, STATE_HEADER_BYTES);
                const bytes = await batch.submit().readback;
                expect(bytes.byteLength).toBe(256);
                const after = FA2_STATE.read(new DataView(bytes), request.offset);
                expect(after).toEqual(EXPECTED);
                results.push(Array.from(new Uint32Array(bytes, request.offset, 64)));
                state.destroy();
            }
            expect(results[0]).toEqual(results[1]);
        });
    });
});
