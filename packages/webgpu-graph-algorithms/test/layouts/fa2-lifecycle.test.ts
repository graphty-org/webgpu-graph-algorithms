/**
 * Lifecycle (spec 11.3 "Device loss / errors / leaks"; contract 5.5 fa2-lifecycle.test.ts): dispose() leaves no
 * live buffer (LeakCounter) and a batch maps at most two staging slots; release(snapshot) during a live simulation
 * makes the next step() reject E_RELEASED and residency.stats().snapshots is 1 after load + release of a previous
 * snapshot; device loss mid-run rejects the pending step with E_DEVICE_LOST, disposes the simulation, leaves the
 * context "lost", and a context from a fresh adapter runs afterwards.
 */

import { GpuContext } from "../../src/context.js";
import {
    BASE_OPTIONS,
    createSim,
    PAPER,
    paritySnapshot,
    rejectionOf,
    startPositions,
    withSim,
} from "../helpers/fa2-parity.js";
import { LeakCounter } from "../helpers/leak-counter.js";
import { acquire, acquireRaw, requireGpu } from "../setup/gpu.js";

/** One macrotask. */
function tick(): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}

describe("FA2 lifecycle (spec 11.3)", () => {
    it("dispose() leaves no live buffer; a step() maps at most two staging slots (scene + state)", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const counter = LeakCounter.wrap(device);
        const own = GpuContext.from(device);
        const s = paritySnapshot("karate", 1, false);
        try {
            const sim = createSim(own, BASE_OPTIONS, PAPER);
            const positions = startPositions(s, BASE_OPTIONS, false);
            sim.load(s, positions);
            await sim.step(1);
            counter.resetMapAsync();
            await sim.step(4);
            expect(counter.mapAsyncCalls, "mapAsync calls of one batch").toBeLessThanOrEqual(2);
            expect(counter.mapAsyncCalls, "a batch reads back").toBeGreaterThanOrEqual(1);
            sim.dispose();
            expect(sim.state).toBe("disposed");
            sim.dispose();
            expect(sim.state, "dispose is idempotent").toBe("disposed");
            own.release(s);
            expect(own.residency.stats().buffers).toBe(0);
        } finally {
            own.dispose();
        }
        expect(counter.live, "live buffers after release + dispose").toBe(0);
        expect(counter.destroyed).toBe(counter.created);
        counter.restore();
    });

    it("release(snapshot) during a live simulation: the next step() rejects E_RELEASED; a load of another snapshot works; snapshots === 1", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "fa2-lifecycle/release" });
        const first = paritySnapshot("karate", 1, false);
        const second = paritySnapshot("grid10", 1, false);
        try {
            await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
                sim.load(first, startPositions(first, BASE_OPTIONS, false));
                await sim.step(2);
                expect(ctx.residency.stats().snapshots).toBe(1);
                ctx.release(first);
                const err = await rejectionOf(sim.step(1));
                expect(err.code).toBe("E_RELEASED");
                expect(err.details.serial).toBe(first.serial);
                // the owner loads another snapshot into the same simulation
                sim.load(second, startPositions(second, BASE_OPTIONS, false));
                await sim.step(2);
                expect(ctx.residency.stats().snapshots).toBe(1);
                expect(sim.iterationsDone).toBe(2);
            });
            // load(next) before release(previous): two resident until the release, then one, the live one keeps stepping
            await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
                sim.load(first, startPositions(first, BASE_OPTIONS, false));
                await sim.step(1);
                sim.load(second, startPositions(second, BASE_OPTIONS, false));
                expect(ctx.residency.stats().snapshots).toBe(2);
                ctx.release(first);
                expect(ctx.residency.stats().snapshots).toBe(1);
                await sim.step(2);
                expect(sim.iterationsDone).toBe(2);
            });
        } finally {
            ctx.release(first);
            ctx.release(second);
        }
    });

    it("device loss mid-run: the pending step rejects E_DEVICE_LOST, the simulation is disposed, the context is lost; a fresh context runs afterwards", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "fa2-lifecycle/lost" });
        const s = paritySnapshot("random1k", 1, false);
        const sim = createSim(ctx, BASE_OPTIONS, PAPER);
        sim.load(s, startPositions(s, BASE_OPTIONS, false));
        await sim.step(1);
        const submittedBefore = sim.lastSubmittedBatchId;
        const pending = sim.step(200);
        while (sim.lastSubmittedBatchId === submittedBefore) {
            await tick();
        }
        ctx.device.destroy();
        const err = await rejectionOf(pending);
        expect(err.code).toBe("E_DEVICE_LOST");
        await ctx.lost;
        expect(ctx.state).toBe("lost");
        expect(sim.state).toBe("disposed");
        const again = await rejectionOf(sim.step(1));
        expect(["E_DISPOSED", "E_DEVICE_LOST"]).toContain(again.code);
        expect(ctx.residency.stats().buffers, "the residency was cleared without destroying").toBe(0);
        // recovery: a context over a fresh adapter (acquire() never reuses one) lays the same graph out
        const fresh = await acquire({ label: "fa2-lifecycle/recovered" });
        try {
            await withSim(fresh, BASE_OPTIONS, PAPER, async (recovered) => {
                const positions = startPositions(s, BASE_OPTIONS, false);
                recovered.load(s, positions);
                await recovered.step(2);
                expect(recovered.iterationsDone).toBe(2);
                expect(positions.every((v) => Number.isFinite(v))).toBe(true);
            });
        } finally {
            fresh.release(s);
        }
    });
});
