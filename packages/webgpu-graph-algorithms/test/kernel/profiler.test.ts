/**
 * Profiler (contract 3.9, spec 5.5): with timestamp-query granted the context owns an enabled profiler; every
 * CommandBatch.pass(label) takes a slot pair, resolveInto(batch) records the resolve + copy into the batch's staging
 * slot, and timings(bytes, request) decodes one { label, ns } per pass with ns > 0; quantised is false under Dawn-node.
 * Without the feature ctx.profiler is null and a Profiler constructed by hand is a no-op (enabled false, beginPass
 * undefined, resolveInto null, timings [], destroy harmless). A full query set drops timings instead of throwing.
 * The passes fill 4M and 2M u32 words (16 MiB and 8 MiB) so each pass lasts many 1,024 ns Dawn ticks.
 */

import { PROFILER_QUERY_SLOTS } from "../../src/constants.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { Profiler } from "../../src/kernel/profiler.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { bindingOf, scratchBuffer, withContext } from "../helpers/device.js";
import { requireGpu } from "../setup/gpu.js";

const ITEMS_A = 4_194_304;
const ITEMS_B = 2_097_152;

describe("Profiler (spec 5.5)", () => {
    it("with timestamp-query granted: a pass timing per pass label, ns > 0, quantised false under Dawn", async (t) => {
        requireGpu(t);
        await withContext({ optionalFeatures: ["timestamp-query", "subgroups"] }, async (ctx) => {
            if (!ctx.caps.features.has("timestamp-query")) {
                console.warn(
                    "[profiler] timestamp-query not granted on this adapter: the enabled-profiler test is skipped",
                );
                t.skip("timestamp-query not granted on this adapter");
            }
            expect(PROFILER_QUERY_SLOTS).toBe(256);
            const { profiler } = ctx;
            expect(profiler).not.toBeNull();
            if (profiler === null) {
                throw new Error("unreachable");
            }
            expect(profiler.enabled).toBe(true);
            expect(profiler.quantised).toBe(false);

            const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
            const ring = new UniformRing(ctx.device, ctx.allocator, 2, "test/profiler/ring");
            const params = ring.binding(FILL_PARAMS);
            const a = scratchBuffer(ctx, ITEMS_A * 4, "test/profiler/a");
            const b = scratchBuffer(ctx, ITEMS_B * 4, "test/profiler/b");
            try {
                ring.write(0, FILL_PARAMS, { count: ITEMS_A, value: 1, mode: 1 });
                ring.write(1, FILL_PARAMS, { count: ITEMS_B, value: 2, mode: 1 });
                ring.flush();
                const batch = new CommandBatch(ctx, "profiled");
                kernel.dispatch(
                    batch.pass("fill-a"),
                    kernel.bind({ dst: bindingOf(a), P: params }),
                    plan1d(ITEMS_A, ctx.workgroupSize, ctx.caps),
                    [ring.offsetOf(0)],
                );
                kernel.dispatch(
                    batch.pass("fill-b"),
                    kernel.bind({ dst: bindingOf(b), P: params }),
                    plan1d(ITEMS_B, ctx.workgroupSize, ctx.caps),
                    [ring.offsetOf(1)],
                );
                const tail = batch.readback(b, 0, 16);
                const request = profiler.resolveInto(batch);
                expect(request).not.toBeNull();
                if (request === null) {
                    throw new Error("unreachable");
                }
                // two passes = 4 query slots = 32 bytes, appended after the 16-byte tail request
                expect(request).toMatchObject({ srcOffset: 0, byteLength: 32, offset: 16 });
                const bytes = await batch.submit().readback;
                expect(bytes.byteLength).toBe(48);
                expect(Array.from(new Uint32Array(bytes, tail.offset, 4))).toEqual([2, 3, 4, 5]);
                const timings = profiler.timings(bytes, request);
                expect(timings.map((timing) => timing.label)).toEqual(["fill-a", "fill-b"]);
                for (const timing of timings) {
                    expect(timing.ns).toBeGreaterThan(0);
                    expect(timing.ns).toBeLessThan(60_000_000_000);
                }
                console.warn(
                    `[profiler] fill-a ${timings[0].ns} ns, fill-b ${timings[1].ns} ns (quantised=${profiler.quantised})`,
                );

                // a request the profiler did not create decodes to nothing
                expect(profiler.timings(bytes, tail)).toEqual([]);
                // the next batch starts from slot 0 again: the same two labels resolve again
                const again = new CommandBatch(ctx, "profiled-2");
                kernel.dispatch(
                    again.pass("fill-a"),
                    kernel.bind({ dst: bindingOf(a), P: params }),
                    plan1d(ITEMS_A, ctx.workgroupSize, ctx.caps),
                    [ring.offsetOf(0)],
                );
                const request2 = profiler.resolveInto(again);
                if (request2 === null) {
                    throw new Error("unreachable");
                }
                expect(request2.offset).toBe(0);
                expect(request2.byteLength).toBe(16);
                const bytes2 = await again.submit().readback;
                const timings2 = profiler.timings(bytes2, request2);
                expect(timings2.map((timing) => timing.label)).toEqual(["fill-a"]);
                expect(timings2[0].ns).toBeGreaterThan(0);
            } finally {
                ring.destroy();
                a.destroy();
                b.destroy();
            }
        });
    });

    it("a full query set drops timings; destroy is idempotent", async (t) => {
        requireGpu(t);
        await withContext({ optionalFeatures: ["timestamp-query", "subgroups"] }, async (ctx) => {
            if (!ctx.caps.features.has("timestamp-query")) {
                t.skip("timestamp-query not granted on this adapter");
            }
            const small = new Profiler(ctx.device, true, false, 4);
            expect(small.enabled).toBe(true);
            const first = small.beginPass("one");
            const second = small.beginPass("two");
            expect(first).toMatchObject({ beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
            expect(second).toMatchObject({ beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 });
            expect(small.beginPass("three")).toBeUndefined();

            const batch = new CommandBatch(ctx, "small");
            const request = small.resolveInto(batch);
            if (request === null) {
                throw new Error("unreachable");
            }
            expect(request.byteLength).toBe(32);
            const bytes = await batch.submit().readback;
            // the two passes were never recorded into the batch: their slots hold whatever the query set held (0 on a fresh set)
            expect(small.timings(bytes, request).map((timing) => timing.label)).toEqual(["one", "two"]);
            // after a resolve the ring restarts at 0
            expect(small.beginPass("four")).toMatchObject({ beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
            small.destroy();
            small.destroy();
            expect(small.beginPass("late")).toBeUndefined();
            expect(small.resolveInto(new CommandBatch(ctx, "late"))).toBeNull();
            // a 1-slot set can hold no pass pair
            const none = new Profiler(ctx.device, true, false, 1);
            expect(none.enabled).toBe(false);
            expect(none.beginPass("x")).toBeUndefined();
            none.destroy();
        });
    });

    it("without the feature: ctx.profiler is null, and a hand-made Profiler is a no-op", async (t) => {
        requireGpu(t);
        await withContext({ subgroups: false }, async (ctx) => {
            expect(ctx.caps.features.has("timestamp-query")).toBe(false);
            expect(ctx.profiler).toBeNull();
            const profiler = new Profiler(ctx.device, true, false);
            expect(profiler.enabled).toBe(false);
            expect(profiler.quantised).toBe(false);
            expect(profiler.beginPass("a")).toBeUndefined();
            const batch = new CommandBatch(ctx, "no-profiler");
            batch.pass("a");
            expect(profiler.resolveInto(batch)).toBeNull();
            const bytes = await batch.submit().readback;
            const slot = ctx.readback.borrowSlot(16);
            expect(profiler.timings(bytes, { src: slot.buffer, srcOffset: 0, byteLength: 16, offset: 0 })).toEqual([]);
            ctx.readback.returnSlot(slot);
            expect(ctx.readback.borrowed).toBe(0);
            profiler.destroy();
            const disabled = new Profiler(ctx.device, false, true);
            expect(disabled.enabled).toBe(false);
            expect(disabled.quantised).toBe(true);
            disabled.destroy();
        });
    });
});
