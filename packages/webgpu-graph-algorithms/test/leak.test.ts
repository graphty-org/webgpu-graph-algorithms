/**
 * The leak counter of spec 11.3 / 11.5 (contract 5.2, 5.5): a counting proxy around a raw device's createBuffer /
 * destroy / mapAsync proves that `degree` + `release` + `dispose` leave no live buffer and that one `degree` call maps
 * exactly one staging slot. The context ADOPTS the device (GpuContext.from) because the proxies must be installed before
 * the context sees the device; test/setup/gpu.ts destroys the raw device in afterAll.
 */

import { degree } from "../src/algorithms/degree.js";
import { GpuContext } from "../src/context.js";
import { isSoftwareAdapter } from "../src/device/acquire.js";
import { BufferUsage, MapMode } from "../src/device/webgpu-constants.js";
import { type GpuCaps } from "../src/types/context.js";
import { KARATE_EDGES, snapshotOf } from "./helpers/graphs.js";
import { LeakCounter } from "./helpers/leak-counter.js";
import { expectBitwiseEqual } from "./helpers/matchers.js";
import { outDegreeOracle } from "./oracle/degree.js";
import { acquireRaw, requireGpu } from "./setup/gpu.js";

/** The Partial<GpuCaps> GpuContext.from takes, built from a raw adapter's info. */
function infoOf(info: GPUAdapterInfo): Partial<GpuCaps> {
    return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        software: isSoftwareAdapter(info),
        runtime: "node",
    };
}

describe("LeakCounter (contract 5.2)", () => {
    it("counts createBuffer, distinct destroys and mapAsync on a raw device; restore() removes the proxies", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const counter = LeakCounter.wrap(device);
        const a = device.createBuffer({ size: 16, usage: BufferUsage.COPY_DST, label: "leak/a" });
        const b = device.createBuffer({ size: 16, usage: BufferUsage.COPY_DST, label: "leak/b" });
        const c = device.createBuffer({
            size: 16,
            usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
            label: "leak/c",
        });
        expect(counter.created).toBe(3);
        expect(counter.destroyed).toBe(0);
        expect(counter.live).toBe(3);

        a.destroy();
        b.destroy();
        b.destroy(); // WebGPU's destroy is idempotent: the second call is not a second destruction
        expect(counter.destroyed).toBe(2);
        expect(counter.live).toBe(1);

        expect(counter.mapAsyncCalls).toBe(0);
        await c.mapAsync(MapMode.READ);
        c.unmap();
        expect(counter.mapAsyncCalls).toBe(1);
        counter.resetMapAsync();
        expect(counter.mapAsyncCalls).toBe(0);

        c.destroy();
        expect(counter.live).toBe(0);

        counter.restore();
        const d = device.createBuffer({ size: 16, usage: BufferUsage.COPY_DST, label: "leak/d" });
        d.destroy();
        expect(counter.created, "a buffer created after restore() is not counted").toBe(3);
        expect(counter.destroyed).toBe(3);
        expect(counter.live).toBe(0);
        counter.restore(); // idempotent
    });

    it("after degree + release + dispose on an adopted device, live === 0 (spec 11.5)", async (t) => {
        requireGpu(t);
        const { device, info } = await acquireRaw();
        const counter = LeakCounter.wrap(device);
        const ctx = GpuContext.from(device, infoOf(info));
        expect(ctx.ownsDevice).toBe(false);
        const s = snapshotOf(KARATE_EDGES, { label: "leak/karate" });
        try {
            const out = await degree(ctx, s);
            expectBitwiseEqual(out, outDegreeOracle(s), "degree on karate");
            expect(counter.created, "degree creates at least the core, the scratch and a staging slot").toBeGreaterThan(
                0,
            );
            ctx.release(s);
            expect(ctx.residency.stats().buffers).toBe(0);
            expect(ctx.pool.idleBytes, "release trims the pool (spec 4.4)").toBe(0);
        } finally {
            ctx.dispose();
        }
        expect(ctx.state).toBe("disposed");
        expect(counter.live, `live buffers after dispose: ${counter.live} of ${counter.created} created`).toBe(0);
        counter.restore();
    });

    it("degree makes exactly one mapAsync call per invocation", async (t) => {
        requireGpu(t);
        const { device, info } = await acquireRaw();
        const counter = LeakCounter.wrap(device);
        const ctx = GpuContext.from(device, infoOf(info));
        const s = snapshotOf(KARATE_EDGES, { label: "leak/karate-map" });
        try {
            await degree(ctx, s); // the first call uploads the core and compiles the pipeline
            for (let i = 0; i < 3; i++) {
                counter.resetMapAsync();
                await degree(ctx, s);
                expect(counter.mapAsyncCalls, `call ${i}`).toBe(1);
            }
        } finally {
            ctx.release(s);
            ctx.dispose();
            counter.restore();
        }
        expect(counter.live).toBe(0);
    });
});
