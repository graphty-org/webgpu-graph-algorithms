/**
 * Browser leg of the ForceAtlas2 simulation (spec 11.6 item 3; contract 5.5 P3 row test/browser/forceatlas2.test.ts):
 * the smoke on a 500-node graph -- load, step(10) five times with maxInFlight 2, positions written back, setPosition
 * and setFixed honoured, dispose clean -- plus the 11.4 frame-loop test (the 600-tick run, the setPosition-during-
 * flight run and the pause run of test/layouts/frame-loop.test.ts) on Chromium: SwiftShader on the default lane,
 * the NVIDIA card on the GPU lane and the dev box. Sizes follow browserScale() (1 on nvidia, 1 / 50 on swiftshader);
 * the smoke keeps the spec's 500 nodes on both (50 iterations of a 500-node exact tile is well inside SwiftShader's
 * budget).
 */

import { type GraphSnapshot, makeMask, maskSet } from "@graphty/graph-format";

import { MAX_ITERATIONS_PER_STEP } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { createForceAtlas2 } from "../../src/layouts/forceatlas2.js";
import { type ForceAtlas2Stats, type GpuLayoutSimulation } from "../../src/types/layout.js";
import { type ForceAtlas2Options } from "../../src/types/options.js";
import { runFrameLoop } from "../helpers/frame-loop.js";
import { fixture, randomEdges, snapshotOf } from "../helpers/graphs.js";
import {
    acquireBrowser,
    browserExpectsSoftware,
    browserGpu,
    browserGrantedSoftware,
    browserScale,
    requireBrowserGpu,
} from "../setup/browser.js";

/**
 * Pins the adapter the test ran on: requireBrowserGpu only enforces GRAPHTY_GPU_REQUIRE, and under the default
 * lane's `any` a run on the wrong adapter (or a silently different flag set) would pass without a trace.
 * @param ctx - the context acquireBrowser handed out
 */
function expectAdapterMatchesFlagSet(ctx: GpuContext): void {
    expect(ctx.caps.software).toBe(browserExpectsSoftware() ?? browserGrantedSoftware());
    if (browserGpu() === "nvidia") {
        expect(ctx.caps.vendor).toBe("nvidia");
    }
}

/** The four controller fields of a stats record (the "speed trace" of spec 7.19). */
interface ControllerSample {
    readonly speed: number;
    readonly swing: number;
    readonly traction: number;
    readonly speedEfficiency: number;
}

function controllerOf(stats: ForceAtlas2Stats): ControllerSample {
    return { speed: stats.speed, swing: stats.swing, traction: stats.traction, speedEfficiency: stats.speedEfficiency };
}

function lcgUnit(seed: number): number {
    return ((seed * 9301 + 49297) % 233280) / 233280;
}

function nextTick(): Promise<void> {
    return new Promise<void>((resolve) => {
        setTimeout(() => {
            resolve();
        }, 0);
    });
}

/** Chromium clamps nested zero-delay timers to 4 ms; measured (the median of 20 ticks), never assumed. */
async function measureTickMs(): Promise<number> {
    const ticks: number[] = [];
    for (let i = 0; i < 20; i++) {
        const start = performance.now();
        await nextTick();
        ticks.push(performance.now() - start);
    }
    ticks.sort((a, b) => a - b);
    return Math.max(ticks[ticks.length / 2], 0.5);
}

/** One batch of k iterations, timed from the call to the readback landing. */
async function timedStep(sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>, k: number): Promise<number> {
    const start = performance.now();
    await sim.step(k);
    return performance.now() - start;
}

/**
 * The UNCONTENDED wall time of one batch: the minimum of three timings, so a batch queued behind another
 * process's GPU work (a sibling run on the dev box) cannot under-size k (see test/layouts/frame-loop.test.ts).
 */
async function minTimedStep(
    sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>,
    k: number,
): Promise<number> {
    let best = Infinity;
    for (let i = 0; i < 3; i++) {
        best = Math.min(best, await timedStep(sim, k));
    }
    return best;
}

/**
 * iterationsPerStep that makes one batch outlast at least four ticks here: the step(8) estimate is VERIFIED and k
 * doubled until a batch measures >= 4 ticks, each timing the minimum of three runs (see
 * test/layouts/frame-loop.test.ts); throws with the measured ratio
 * when even MAX_ITERATIONS_PER_STEP stays under two ticks. Leaves ctx.pipelines warm (compile-once per context).
 */
async function calibrateHeavyStep(ctx: GpuContext, snapshot: GraphSnapshot, tickMs: number): Promise<number> {
    const scratch = createForceAtlas2(ctx, { seed: 7, maxIter: 1_000_000, settleThreshold: 0 });
    const positions = new Float32Array(3 * snapshot.nodeCount).fill(NaN);
    scratch.load(snapshot, positions);
    await scratch.step(8);
    const batchMs = Math.max(await minTimedStep(scratch, 8), 0.05);
    let k = Math.min(MAX_ITERATIONS_PER_STEP, Math.max(8, Math.ceil((8 * 4 * tickMs) / batchMs)));
    let measuredMs = await minTimedStep(scratch, k);
    while (measuredMs < 4 * tickMs && k < MAX_ITERATIONS_PER_STEP) {
        k = Math.min(MAX_ITERATIONS_PER_STEP, k * 2);
        measuredMs = await minTimedStep(scratch, k);
    }
    scratch.dispose();
    if (measuredMs < 2 * tickMs) {
        throw new Error(
            `calibrateHeavyStep: step(${k}) measured ${measuredMs.toFixed(3)} ms against ${tickMs.toFixed(3)} ms ticks ` +
                `(${(measuredMs / tickMs).toFixed(2)} ticks per batch): even MAX_ITERATIONS_PER_STEP iterations cannot ` +
                "keep a batch in flight across two tick starts on this adapter",
        );
    }
    return k;
}

/**
 * Warms ctx.pipelines on a throwaway simulation so the loop's own simulation starts with untouched counters and a
 * bind promise that resolves inside its first tick (contract 3.13 checks the coalescing before awaiting the bind /
 * warm promise; a cold simulation stepped every tick would submit every call of the compile window at once).
 */
async function warmPipelines(ctx: GpuContext, snapshot: GraphSnapshot): Promise<void> {
    const scratch = createForceAtlas2(ctx, { seed: 7, maxIter: 1_000_000, settleThreshold: 0 });
    scratch.load(snapshot, new Float32Array(3 * snapshot.nodeCount).fill(NaN));
    await scratch.step(1);
    scratch.dispose();
}

function expectMonotone(values: readonly number[], except: ReadonlySet<number> = new Set()): void {
    for (let i = 1; i < values.length; i++) {
        if (except.has(i - 1)) {
            continue;
        }
        expect(values[i], `tick ${i}: ${values[i]} after ${values[i - 1]}`).toBeGreaterThanOrEqual(values[i - 1]);
    }
}

describe("createForceAtlas2 in Chromium: the 11.6 item 3 smoke", () => {
    it("500 nodes: load, step(10) x 5 with maxInFlight 2, positions written back, setPosition / setFixed honoured, dispose clean", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser();
        expectAdapterMatchesFlagSet(ctx);
        const slotsBefore = ctx.readback.slots;
        const liveBefore = ctx.allocator.liveBuffers; // before the simulation's uniform ring exists (3.13)
        const n = 500;
        const snapshot = snapshotOf(randomEdges(n, 1500, 11), { nodeCount: n });
        expect(snapshot.nodeCount).toBe(n);
        const sim = createForceAtlas2(ctx, { seed: 3, maxIter: 1000, maxInFlight: 2 });
        const positions = new Float32Array(3 * n).fill(NaN);
        sim.load(snapshot, positions);
        const seeded = Float32Array.from(positions);
        expect(seeded.every((v) => Number.isFinite(v))).toBe(true);

        // five step(10) calls with two batches in flight: the third call is made only once the first landed, so
        // nothing coalesces and every call runs its ten iterations
        const pending: Promise<void>[] = [];
        for (let i = 0; i < 5; i++) {
            pending.push(sim.step(10));
            expect(sim.inFlight).toBeLessThanOrEqual(2);
            if (pending.length === 2) {
                const oldest = pending.shift();
                if (oldest !== undefined) {
                    await oldest;
                }
            }
        }
        await Promise.all(pending);
        expect(sim.inFlight).toBe(0);
        expect(sim.iterationsDone).toBe(50);
        expect(sim.stats.iteration).toBe(50);
        expect(sim.stats.trace).toHaveLength(10); // the last completed batch's k records
        expect(sim.stats.repulsionTier).toBe("exact");

        // positions written back: every value finite, the array moved from the seeds, z == center.z in 2D
        let moved = 0;
        for (let i = 0; i < n; i++) {
            expect(Number.isFinite(positions[3 * i])).toBe(true);
            expect(Number.isFinite(positions[3 * i + 1])).toBe(true);
            expect(positions[3 * i + 2]).toBe(0);
            if (positions[3 * i] !== seeded[3 * i] || positions[3 * i + 1] !== seeded[3 * i + 1]) {
                moved += 1;
            }
        }
        expect(moved).toBeGreaterThan(n / 2);

        // setPosition: the owner's array at once (7.12), reheat (D8), carried through the batches computed after it
        const mask = makeMask(n);
        maskSet(mask, 3, true);
        sim.setFixed(mask); // pin node 3 first: adding a pin does not reheat
        sim.setPosition(3, 5, -4, 0);
        expect(Array.from(positions.subarray(9, 12))).toEqual([5, -4, 0]);
        expect(sim.iterationsDone).toBe(0);
        // setFixed: pin node 7 where it is
        maskSet(mask, 7, true);
        sim.setFixed(mask);
        const node7 = Array.from(positions.subarray(21, 24));
        const before = Float32Array.from(positions);
        await sim.step(5);
        await sim.step(5);
        expect(sim.iterationsDone).toBe(10);
        expect(Array.from(positions.subarray(9, 12))).toEqual([5, -4, 0]); // honoured, not restored by the GPU
        expect(Array.from(positions.subarray(21, 24))).toEqual(node7); // a pinned node never moves (7.12)
        let othersMoved = 0;
        for (let i = 0; i < n; i++) {
            if (
                i !== 3 &&
                i !== 7 &&
                (positions[3 * i] !== before[3 * i] || positions[3 * i + 1] !== before[3 * i + 1])
            ) {
                othersMoved += 1;
            }
        }
        expect(othersMoved).toBeGreaterThan(0);
        maskSet(mask, 7, false);
        sim.setFixed(mask); // an unpin reheats (7.12)
        expect(sim.iterationsDone).toBe(0);
        await sim.step(5);
        expect(sim.iterationsDone).toBe(5);

        // dispose clean
        sim.dispose();
        sim.dispose(); // idempotent
        await expect(sim.step(1)).rejects.toMatchObject({ code: "E_DISPOSED" });
        expect(ctx.pool.liveBytes).toBe(0);
        ctx.release(snapshot);
        expect(ctx.residency.stats().buffers).toBe(0);
        expect(ctx.pool.idleBytes).toBe(0); // release trims the pool (4.4)
        // every simulation buffer is gone; only the staging ring may have grown (its slots are the context's)
        const grown = ctx.readback.slots - slotsBefore;
        expect(grown).toBeGreaterThanOrEqual(0);
        expect([liveBefore, liveBefore + grown]).toContain(ctx.allocator.liveBuffers);
    });
});

describe("frame loop in Chromium (spec 11.4 last bullet: also on SwiftShader and NVIDIA)", () => {
    it("600 ticks on random1k: submissions <= ticks, at most maxInFlight in flight, iterationsDone monotone, settled reported", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser();
        expectAdapterMatchesFlagSet(ctx);
        const { snapshot } = fixture("random1k", browserScale());
        const n = snapshot.nodeCount;
        await warmPipelines(ctx, snapshot); // the loop's own simulation below is never stepped before the loop
        const positions = new Float32Array(3 * n).fill(NaN);
        const sim = createForceAtlas2(ctx, { seed: 7, maxIter: 100, iterationsPerStep: 4, maxInFlight: 2 });
        sim.load(snapshot, positions);
        const seeded = Float32Array.from(positions);
        const report = await runFrameLoop(sim, positions, { ticks: 600, iterationsPerStep: 4, maxInFlight: 2 });
        expect(report.errors).toEqual([]);
        expect(report.submissions).toBeGreaterThanOrEqual(1);
        expect(report.submissions).toBeLessThanOrEqual(600);
        expect(report.maxObservedInFlight).toBeLessThanOrEqual(2);
        expect(report.iterationsDoneByTick).toHaveLength(600);
        expectMonotone(report.iterationsDoneByTick);
        expect(report.settledAtTick).not.toBeNull();
        expect(report.positionHolds).toEqual([]);
        await sim.flush();
        expect(sim.settled).toBe(true);
        expect(sim.iterationsDone).toBe(report.submissions * 4);
        expect(sim.iterationsDone).toBeLessThanOrEqual(104);
        let moved = 0;
        for (let i = 0; i < n; i++) {
            expect(Number.isFinite(positions[3 * i])).toBe(true);
            expect(positions[3 * i + 2]).toBe(0);
            if (positions[3 * i] !== seeded[3 * i] || positions[3 * i + 1] !== seeded[3 * i + 1]) {
                moved += 1;
            }
        }
        expect(moved).toBeGreaterThan(0);
        sim.dispose();
        ctx.release(snapshot);
    });

    it("a setPosition during flight lands in the following batch and is never overwritten by an older one", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser();
        expectAdapterMatchesFlagSet(ctx);
        const { snapshot } = fixture("random1k", browserScale());
        const n = snapshot.nodeCount;
        const k = await calibrateHeavyStep(ctx, snapshot, await measureTickMs());
        const positions = new Float32Array(3 * n).fill(NaN);
        const sim = createForceAtlas2(ctx, {
            seed: 7,
            maxIter: 1_000_000,
            settleThreshold: 0,
            iterationsPerStep: k,
            maxInFlight: 2,
        });
        sim.load(snapshot, positions);
        const WRITES = 10;
        const writes = Array.from({ length: WRITES }, (_, j) => ({
            tick: 20 + 20 * j,
            index: j,
            x: 100 + j,
            y: -100 - j,
            z: 0,
        }));
        const mask = makeMask(n);
        const inFlightAtWrite: number[] = [];
        const report = await runFrameLoop(sim, positions, {
            ticks: 600,
            iterationsPerStep: k,
            maxInFlight: 2,
            setPositionAt: writes,
            onTick: (tick) => {
                const write = writes.find((w) => w.tick === tick);
                if (write !== undefined) {
                    maskSet(mask, write.index, true);
                    sim.setFixed(mask);
                    inFlightAtWrite.push(sim.inFlight);
                }
            },
        });
        expect(report.errors).toEqual([]);
        expect(report.positionHolds).toHaveLength(WRITES);
        for (const hold of report.positionHolds) {
            expect(hold.held, `write at tick ${hold.tick} on node ${hold.index}`).toBe(true);
        }
        expect(Math.max(...inFlightAtWrite)).toBeGreaterThanOrEqual(1);
        expect(report.maxObservedInFlight).toBeLessThanOrEqual(2);
        expectMonotone(report.iterationsDoneByTick, new Set(writes.map((w) => w.tick)));
        await sim.flush();
        for (const w of writes) {
            expect(Array.from(positions.subarray(3 * w.index, 3 * w.index + 3)), `node ${w.index}`).toEqual([
                w.x,
                w.y,
                0,
            ]);
        }
        sim.dispose();
        ctx.release(snapshot);
    });

    it("pause: exactly the in-flight batches land, flush() resolves, no submission for 100 ticks, a later step() continues", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser();
        expectAdapterMatchesFlagSet(ctx);
        const { snapshot } = fixture("random1k", browserScale());
        const n = snapshot.nodeCount;
        const k = await calibrateHeavyStep(ctx, snapshot, await measureTickMs());
        const options = {
            seed: 7,
            maxIter: 1_000_000,
            settleThreshold: 0,
            iterationsPerStep: k,
            maxInFlight: 2,
        } as const;
        const positions = new Float32Array(3 * n).fill(NaN);
        const sim = createForceAtlas2(ctx, options);
        sim.load(snapshot, positions);
        const pauseAt = 40 + Math.floor(lcgUnit(7) * 60); // 69
        const PAUSE = 100;
        const inFlightByTick: number[] = [];
        const sampled = new Map<number, ControllerSample>();
        const report = await runFrameLoop(sim, positions, {
            ticks: 600,
            iterationsPerStep: k,
            maxInFlight: 2,
            pauseAt,
            pauseTicks: PAUSE,
            onTick: () => {
                inFlightByTick.push(sim.inFlight);
                const { stats } = sim;
                if (stats.iteration > 0 && !sampled.has(stats.iteration)) {
                    sampled.set(stats.iteration, controllerOf(stats));
                }
            },
        });
        expect(report.errors).toEqual([]);
        expect(report.submissionsDuringPause).toBe(0);
        expect(report.settledAtTick).toBeNull();
        const pauseStart = inFlightByTick.findIndex((v, i) => i >= pauseAt && v === 2);
        expect(pauseStart).toBeGreaterThanOrEqual(pauseAt);
        expect(pauseStart + PAUSE).toBeLessThan(600);
        const atPause = report.iterationsDoneByTick[pauseStart];
        const afterPause = report.iterationsDoneByTick[pauseStart + PAUSE];
        expect(afterPause - atPause).toBe(2 * k);
        expect(inFlightByTick[pauseStart + PAUSE]).toBe(0);
        expectMonotone(report.iterationsDoneByTick);
        expect(report.iterationsDoneByTick[599]).toBeGreaterThan(afterPause);
        await sim.flush();
        expect(sim.iterationsDone).toBe(report.submissions * k);
        const maxIteration = Math.max(...sampled.keys());
        const referencePositions = new Float32Array(3 * n).fill(NaN);
        const reference = createForceAtlas2(ctx, options);
        reference.load(snapshot, referencePositions);
        const referenceByIteration = new Map<number, ControllerSample>();
        while (reference.stats.iteration < maxIteration) {
            await reference.step(k);
            referenceByIteration.set(reference.stats.iteration, controllerOf(reference.stats));
        }
        let checkedAfterPause = 0;
        for (const [iteration, sample] of sampled) {
            const expected = referenceByIteration.get(iteration);
            if (expected === undefined) {
                throw new Error(`the unpaused reference never reported iteration ${iteration}`);
            }
            for (const key of ["speed", "swing", "traction", "speedEfficiency"] as const) {
                expect(Object.is(sample[key], expected[key]), `${key} at iteration ${iteration}`).toBe(true);
            }
            if (iteration > afterPause) {
                checkedAfterPause += 1;
            }
        }
        expect(checkedAfterPause).toBeGreaterThan(0);
        reference.dispose();
        sim.dispose();
        ctx.release(snapshot);
    });
});
