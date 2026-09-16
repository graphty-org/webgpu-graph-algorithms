/**
 * The element's frame loop against ForceSimulation (spec 7.19; 11.4 last bullet; contract 5.5 P3 row
 * test/layouts/frame-loop.test.ts).
 *
 * Part 1 pins the bridge logic of test/helpers/frame-loop.ts on a SCRIPTED simulation with hand-computed reports:
 * one synchronous step(k) per tick, coalescing at maxInFlight, the error handler once per distinct promise, the
 * hold / release rule of a setPosition issued during flight, the pause window and its flush().
 *
 * Part 2 runs the same helper against the GPU ForceAtlas2 simulation: the promise-identity check (a coalesced
 * step() returns the OLDEST pending batch's promise as the same object), the 600-tick run (submissions <= ticks, at
 * most maxInFlight in flight, iterationsDone monotone, settled reported, positions written back), the
 * setPosition-during-flight run (every hold true, the pinned rows carry the written values through every following
 * batch, the reheat of D8 visible in the tick series) and the pause run (exactly the in-flight batches land,
 * flush() resolves, no submission for 100 ticks, the later step() continues with iterationsDone and the controller
 * trace bitwise equal to an unpaused run of the same simulation).
 */

import { type F32, type GraphSnapshot, makeMask, maskSet, type NodeMask } from "@graphty/graph-format";

import { MAX_ITERATIONS_PER_STEP } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { createForceAtlas2 } from "../../src/layouts/forceatlas2.js";
import { type ForceAtlas2Stats, type GpuLayoutSimulation, type RunOptions } from "../../src/types/layout.js";
import { type ForceAtlas2Options } from "../../src/types/options.js";
import { type FrameLoopOptions, type FrameLoopReport, runFrameLoop } from "../helpers/frame-loop.js";
import { fixture } from "../helpers/graphs.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

// ============================================================ part 1: the bridge logic on a scripted simulation

/** The stats record the scripted simulation reports (never inspected by part 1; the shape must type-check). */
const FAKE_STATS: ForceAtlas2Stats = {
    iteration: 0,
    meanDisplacement: 0,
    rmsRadius: 1,
    layoutRadius: 1,
    centroid: [0, 0, 0],
    repulsionTier: "exact",
    maxCellOccupancy: null,
    outsideGrid: null,
    msPerIteration: null,
    swing: 0,
    traction: 0,
    speed: 1,
    speedEfficiency: 1,
    trace: [],
};

interface FakeBatch {
    readonly id: number;
    readonly iterations: number;
    /** The positions as the "GPU" saw them at submission; landing writes them back with x + 1. */
    readonly snapshot: Float32Array;
    readonly promise: Promise<void>;
    readonly resolve: () => void;
}

/**
 * A promise with its resolver exposed (lib ES2020 has no Promise.withResolvers).
 * @returns the pair
 */
function deferred(): { readonly promise: Promise<void>; readonly resolve: () => void } {
    let resolve: () => void = () => undefined;
    const promise = new Promise<void>((r) => {
        resolve = r;
    });
    return { promise, resolve };
}

/**
 * A scripted GpuLayoutSimulation: step() submits SYNCHRONOUSLY (a batch id and a snapshot of the positions),
 * coalesces at maxInFlight by returning the OLDEST pending promise (spec 7.19 item 3), resolves at once when
 * settled (item 2); land() plays the GPU for the oldest batch: every row's x moves by +1 from the submission
 * snapshot, skipping rows overridden by a write issued after the batch (spec 7.12) unless honourOverrides is
 * false; setPosition writes the row, records { index -> lastSubmittedBatchId } and reheats (iterationsDone = 0,
 * D8). Row i starts at (i, i, 0).
 */
class FakeSimulation implements GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats> {
    readonly positions: F32;
    readonly maxInFlight: number;
    readonly maxIter: number;
    readonly honourOverrides: boolean;
    lastSubmittedBatchId = 0;
    coalesced = 0;
    iterationsDone = 0;
    private readonly pending: FakeBatch[] = [];
    private readonly overrides = new Map<number, number>();

    constructor(n: number, maxInFlight: number, maxIter: number, honourOverrides: boolean) {
        this.positions = new Float32Array(3 * n);
        for (let i = 0; i < n; i++) {
            this.positions[3 * i] = i;
            this.positions[3 * i + 1] = i;
            this.positions[3 * i + 2] = 0;
        }
        this.maxInFlight = maxInFlight;
        this.maxIter = maxIter;
        this.honourOverrides = honourOverrides;
    }

    get settled(): boolean {
        return this.iterationsDone >= this.maxIter;
    }

    get inFlight(): number {
        return this.pending.length;
    }

    get stats(): ForceAtlas2Stats {
        return FAKE_STATS;
    }

    load(_snapshot: GraphSnapshot, _positions: F32): void {
        /* the scripted simulation is born loaded */
    }

    step(iterations?: number): Promise<void> {
        const k = iterations ?? 1;
        if (this.settled) {
            return Promise.resolve();
        }
        if (this.pending.length >= this.maxInFlight) {
            this.coalesced += 1;
            return this.pending[0].promise;
        }
        this.lastSubmittedBatchId += 1;
        const { promise, resolve } = deferred();
        this.pending.push({
            id: this.lastSubmittedBatchId,
            iterations: k,
            snapshot: Float32Array.from(this.positions),
            promise,
            resolve,
        });
        return promise;
    }

    /** The test's GPU: lands the oldest batch (its readback copied into the owner's array, then its promise resolves). */
    land(): void {
        const batch = this.pending.shift();
        if (batch === undefined) {
            throw new Error("FakeSimulation.land(): nothing in flight");
        }
        const n = this.positions.length / 3;
        for (let i = 0; i < n; i++) {
            const afterBatch = this.overrides.get(i);
            if (this.honourOverrides && afterBatch !== undefined && afterBatch >= batch.id) {
                continue; // computed before the write: the row would move back (spec 7.12)
            }
            this.positions[3 * i] = batch.snapshot[3 * i] + 1;
            this.positions[3 * i + 1] = batch.snapshot[3 * i + 1];
            this.positions[3 * i + 2] = batch.snapshot[3 * i + 2];
        }
        for (const [i, afterBatch] of this.overrides) {
            if (batch.id > afterBatch) {
                this.overrides.delete(i);
            }
        }
        this.iterationsDone += batch.iterations;
        batch.resolve();
    }

    setFixed(_mask: NodeMask): void {
        /* no pins in the scripted simulation */
    }

    setPosition(index: number, x: number, y: number, z: number): void {
        this.positions[3 * index] = x;
        this.positions[3 * index + 1] = y;
        this.positions[3 * index + 2] = z;
        this.overrides.set(index, this.lastSubmittedBatchId);
        this.reheat();
    }

    reheat(): void {
        this.iterationsDone = 0;
    }

    setParams(_patch: Partial<ForceAtlas2Options>): void {
        /* nothing to resolve */
    }

    flush(): Promise<void> {
        return Promise.all(this.pending.map((b) => b.promise)).then(() => undefined);
    }

    run(_options?: RunOptions): Promise<ForceAtlas2Stats> {
        return Promise.resolve(FAKE_STATS);
    }

    dispose(): void {
        /* nothing to destroy */
    }
}

describe("runFrameLoop: the element's bridge logic on a scripted simulation (spec 7.19, 9.4 item 4)", () => {
    it("A: one submission per tick until maxInFlight, coalescing while saturated, landings counted per tick", async () => {
        const fake = new FakeSimulation(4, 2, 1_000_000, true);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 8,
            iterationsPerStep: 1,
            maxInFlight: 2,
            onTick: (tick) => {
                if (tick === 3 || tick === 6) {
                    fake.land();
                }
            },
        });
        // tick 0: submit b1 (inFlight 1) | 1: submit b2 (2) | 2: coalesce (returns b1's promise, already handled)
        // 3: b1 lands in onTick (iterationsDone 1), submit b3 | 4: coalesce | 5: coalesce | 6: b2 lands (2), submit b4
        // 7: coalesce.  The tick-start sample of tick t sees every landing of ticks < t.
        expect(report).toEqual({
            submissions: 4,
            coalesced: 4,
            maxObservedInFlight: 2,
            iterationsDoneByTick: [0, 0, 0, 0, 1, 1, 1, 2],
            settledAtTick: null,
            errors: [],
            positionHolds: [],
            submissionsDuringPause: 0,
        });
        expect(fake.inFlight).toBe(2); // b3 and b4 never landed: the helper never awaits step()
        expect(fake.coalesced).toBe(4);
    });

    it("B: a setPosition during flight holds until a batch submitted after the write lands (override rule honoured)", async () => {
        const fake = new FakeSimulation(4, 2, 1_000_000, true);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 8,
            iterationsPerStep: 1,
            maxInFlight: 2,
            setPositionAt: [{ tick: 2, index: 1, x: 10, y: 20, z: 0 }],
            onTick: (tick) => {
                if (tick === 3 || tick === 4 || tick === 6) {
                    fake.land();
                }
            },
        });
        // 0: submit b1 | 1: submit b2 | 2: write row 1 = (10, 20, 0) with lastSubmittedBatchId 2, then coalesce (b1's promise)
        // 3: b1 lands (older than the write: row 1 skipped, stays (10, 20, 0)); submit b3 -- first returned at tick 3 >= 2
        // 4: b2 lands (older: skipped); submit b4 | 5: coalesce | 6: b3 lands (newer: row 1 <- its snapshot (10, 20, 0) + (1, 0, 0)),
        //    the write is released; submit b5 | 7: coalesce.  iterationsDone: reheat at tick 2 (0 -> 0), then 1, 2, 3.
        expect(report).toEqual({
            submissions: 5,
            coalesced: 3,
            maxObservedInFlight: 2,
            iterationsDoneByTick: [0, 0, 0, 0, 1, 2, 2, 3],
            settledAtTick: null,
            errors: [],
            positionHolds: [{ tick: 2, index: 1, held: true }],
            submissionsDuringPause: 0,
        });
        expect(Array.from(fake.positions.subarray(3, 6))).toEqual([11, 20, 0]); // b3 carried the write forward
    });

    it("B': the same schedule with the override rule IGNORED reports held: false (the helper detects the bug)", async () => {
        const fake = new FakeSimulation(4, 2, 1_000_000, false);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 8,
            iterationsPerStep: 1,
            maxInFlight: 2,
            setPositionAt: [{ tick: 2, index: 1, x: 10, y: 20, z: 0 }],
            onTick: (tick) => {
                if (tick === 3 || tick === 4 || tick === 6) {
                    fake.land();
                }
            },
        });
        // b1's landing at tick 3 overwrites row 1 with its stale snapshot (1, 1, 0) + (1, 0, 0) = (2, 1, 0): the
        // resolution of a promise older than the write finds the row changed -> held false. Counts as in B.
        expect(report.positionHolds).toEqual([{ tick: 2, index: 1, held: false }]);
        expect(report.submissions).toBe(5);
        expect(report.coalesced).toBe(3);
        expect(report.errors).toEqual([]);
    });

    it("C: settled is reported at the first tick that starts settled; later ticks submit nothing", async () => {
        const fake = new FakeSimulation(4, 2, 2, true);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 8,
            iterationsPerStep: 1,
            maxInFlight: 2,
            onTick: (tick) => {
                if (tick === 3 || tick === 6) {
                    fake.land();
                }
            },
        });
        // as A until tick 6: b2 lands there (iterationsDone 2 >= maxIter 2 -> settled), so tick 6's step() resolves at
        // once and is not a submission; tick 7 starts settled -> settledAtTick 7. b3 (tick 3) stays in flight.
        expect(report).toEqual({
            submissions: 3,
            coalesced: 3,
            maxObservedInFlight: 2,
            iterationsDoneByTick: [0, 0, 0, 0, 1, 1, 1, 2],
            settledAtTick: 7,
            errors: [],
            positionHolds: [],
            submissionsDuringPause: 0,
        });
    });

    it("D: the pause starts at the first tick >= pauseAt with inFlight === maxInFlight, flush() resolves inside it, stepping resumes", async () => {
        const fake = new FakeSimulation(4, 2, 1_000_000, true);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 12,
            iterationsPerStep: 1,
            maxInFlight: 2,
            pauseAt: 2,
            pauseTicks: 3,
            onTick: (tick) => {
                if (tick === 3 || tick === 4) {
                    fake.land();
                }
            },
        });
        // 0: submit b1 | 1: submit b2 | 2: inFlight 2 === maxInFlight and 2 >= pauseAt -> pause [2, 5), flush() called
        // 3: paused; b1 lands | 4: paused; b2 lands -> flush() resolves before tick 5 | 5: pause over, submit b3
        // 6: submit b4 | 7..11: coalesce (5 ticks).  No submission was observed inside the window.
        expect(report).toEqual({
            submissions: 4,
            coalesced: 5,
            maxObservedInFlight: 2,
            iterationsDoneByTick: [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2],
            settledAtTick: null,
            errors: [],
            positionHolds: [],
            submissionsDuringPause: 0,
        });
    });

    it("E: a pause that can never start (batches always land before the next tick) is reported in errors", async () => {
        const fake = new FakeSimulation(4, 2, 1_000_000, true);
        const report = await runFrameLoop(fake, fake.positions, {
            ticks: 4,
            iterationsPerStep: 1,
            maxInFlight: 2,
            pauseAt: 0,
            pauseTicks: 2,
            onTick: () => {
                if (fake.inFlight > 0) {
                    fake.land();
                }
            },
        });
        // every tick starts with inFlight 1 at most (0 at tick 0), so inFlight === 2 is never observed:
        // 0: submit b1 | 1: land b1, submit b2 | 2: land b2, submit b3 | 3: land b3, submit b4.
        expect(report.submissions).toBe(4);
        expect(report.coalesced).toBe(0);
        expect(report.maxObservedInFlight).toBe(1);
        expect(report.iterationsDoneByTick).toEqual([0, 0, 1, 2]);
        expect(report.submissionsDuringPause).toBe(0);
        expect(report.errors).toHaveLength(1);
        expect(String(report.errors[0])).toMatch(/pause never started/);
    });

    it("rejects a simulation without the ForceSimulation counters", async () => {
        const bare = { step: () => Promise.resolve() } as unknown as GpuLayoutSimulation<
            ForceAtlas2Options,
            ForceAtlas2Stats
        >;
        await expect(
            runFrameLoop(bare, new Float32Array(0), { ticks: 1, iterationsPerStep: 1, maxInFlight: 1 }),
        ).rejects.toThrow(/lastSubmittedBatchId/);
    });
});

// ============================================================ part 2: the GPU ForceAtlas2 simulation

/** The four controller fields of a stats record (the "speed trace" of spec 7.19). */
interface ControllerSample {
    readonly speed: number;
    readonly swing: number;
    readonly traction: number;
    readonly speedEfficiency: number;
}

/**
 * The controller fields of a stats record.
 * @param stats - the record
 * @returns the four fields
 */
function controllerOf(stats: ForceAtlas2Stats): ControllerSample {
    return { speed: stats.speed, swing: stats.swing, traction: stats.traction, speedEfficiency: stats.speedEfficiency };
}

/**
 * The seeded "random tick" of spec 7.19's pause case (a Park-Miller-style LCG step in [0, 1)), so the run is
 * reproducible: lcgUnit(7) = ((7 * 9301 + 49297) % 233280) / 233280 = 114404 / 233280 = 0.49041 (rounded), so 40 + floor(0.49041 * 60) = 69
 * @param seed - the seed
 * @returns a value in [0, 1)
 */
function lcgUnit(seed: number): number {
    return ((seed * 9301 + 49297) % 233280) / 233280;
}

/**
 * One macrotask, the helper's tick primitive.
 * @returns a promise resolved by a zero-delay timer
 */
function nextTick(): Promise<void> {
    return new Promise<void>((resolve) => {
        setTimeout(() => {
            resolve();
        }, 0);
    });
}

/**
 * The typical length of one tick on this runtime (Node clamps setTimeout 0 to about 1 ms; Chromium to 4 ms once
 * timers nest): the MEDIAN of 20 measured ticks, so a stall of the event loop under a loaded box (parallel test
 * workers) does not inflate the estimate; never reported below 0.5 ms.
 * @returns milliseconds per tick
 */
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

/**
 * Runs one batch of k iterations and returns its wall time (submit to readback landed).
 * @param sim - the simulation
 * @param k - iterations per step
 * @returns milliseconds
 */
async function timedStep(sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>, k: number): Promise<number> {
    const start = performance.now();
    await sim.step(k);
    return performance.now() - start;
}

/**
 * The UNCONTENDED wall time of one batch of k iterations: the minimum of three timings. A single timing can be
 * queued behind another process's GPU work (a parallel test worker of the node project, a sibling run on the dev
 * box) and come out several times too long, which would under-size k and let the flight-dependent runs' batches
 * land inside a tick; contention only ever lengthens a batch, so the minimum is the estimate the premise needs.
 * @param sim - the simulation
 * @param k - iterations per step
 * @returns milliseconds
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
 * Picks iterationsPerStep so that one batch outlasts at least `targetTicks` ticks (default four) on the running
 * adapter -- otherwise a batch lands inside the tick gap and inFlight === maxInFlight is never observable at a
 * tick start (the pause case of spec 7.19 and a setPosition "during flight" both need it). A step(8) timing on a
 * throwaway simulation gives the linear estimate, which under-shoots: a small batch is dominated by the fixed
 * submit / mapAsync latency (spec 10.4 T-3), not by the per-iteration cost. The estimate is therefore VERIFIED:
 * step(k) is timed (the minimum of three runs, see minTimedStep) and k doubled until one batch measures >=
 * targetTicks ticks, clamped to [8, MAX_ITERATIONS_PER_STEP]; when even
 * MAX_ITERATIONS_PER_STEP iterations stay under two ticks the adapter cannot host the flight-dependent runs and
 * the helper throws with the measured ratio (a legible finding, not "pause never started"). Side effect relied on
 * by every caller: ctx.pipelines is warm afterwards (compile-once per context, contract 3.9).
 * @param ctx - the context
 * @param snapshot - the graph of the run
 * @param tickMs - measureTickMs()
 * @param targetTicks - the batch length aimed at, in ticks (default 4; the promise-identity test asks for 64, which
 * clamps at MAX_ITERATIONS_PER_STEP on a fast adapter: its premises hold only while a batch is in flight, and on a
 * loaded box a batch of a few ticks can land inside one stalled tick gap -- a longer batch is the only margin)
 * @returns the iterations per step
 */
async function calibrateHeavyStep(
    ctx: GpuContext,
    snapshot: GraphSnapshot,
    tickMs: number,
    targetTicks = 4,
): Promise<number> {
    const scratch = createForceAtlas2(ctx, { seed: 7, maxIter: 1_000_000, settleThreshold: 0 });
    const positions = new Float32Array(3 * snapshot.nodeCount).fill(NaN);
    scratch.load(snapshot, positions);
    await scratch.step(8); // warm-up: pipeline compile, first submit
    const batchMs = Math.max(await minTimedStep(scratch, 8), 0.05);
    let k = Math.min(MAX_ITERATIONS_PER_STEP, Math.max(8, Math.ceil((8 * targetTicks * tickMs) / batchMs)));
    let measuredMs = await minTimedStep(scratch, k);
    while (measuredMs < targetTicks * tickMs && k < MAX_ITERATIONS_PER_STEP) {
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
 * Asserts a tick series never decreases, except across the ticks in `except` (a reheat resets iterationsDone).
 * @param values - the per-tick series
 * @param except - ticks after which a drop is allowed
 */
function expectMonotone(values: readonly number[], except: ReadonlySet<number> = new Set()): void {
    for (let i = 1; i < values.length; i++) {
        if (except.has(i - 1)) {
            continue;
        }
        expect(values[i], `tick ${i}: ${values[i]} after ${values[i - 1]}`).toBeGreaterThanOrEqual(values[i - 1]);
    }
}

/**
 * Warms ctx.pipelines (compile-once per context, contract 3.9) on a throwaway simulation so a simulation created
 * afterwards starts with untouched counters and a bind promise that resolves inside its first tick. Contract 3.13
 * checks the coalescing BEFORE awaiting the bind / warm promise, so a cold simulation stepped every tick would
 * submit every call of the compile window at once (see the PLAN DECISION above the task).
 * @param ctx - the context
 * @param snapshot - the graph of the run
 */
async function warmPipelines(ctx: GpuContext, snapshot: GraphSnapshot): Promise<void> {
    const scratch = createForceAtlas2(ctx, { seed: 7, maxIter: 1_000_000, settleThreshold: 0 });
    scratch.load(snapshot, new Float32Array(3 * snapshot.nodeCount).fill(NaN));
    await scratch.step(1);
    scratch.dispose();
}

describe("frame loop on the GPU ForceAtlas2 simulation (spec 7.19; 11.4 last bullet)", () => {
    it("a coalesced step() returns the same promise object: the OLDEST pending batch's (spec 7.19 item 3, 9.4 item 4)", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        // 16 x the tile of the other runs (the plan's remedy for a batch that lands inside a tick gap: a larger
        // fixture, never a shorter tick or a looser assertion). Measured on the 4070: step(256) on random1k takes
        // 25-40 ms, on the 16x tile 265-275 ms, against a 1 ms tick; inside the
        // whole-project run (32 forks on the card) one setTimeout(0) tick of this worker exceeded the 1k batch
        // once in four runs and batch 1 had landed before the second tick below (inFlight 1 instead of 2)
        const { snapshot } = fixture("random1k", 16 * gpuScale());
        const n = snapshot.nodeCount;
        // the heaviest batch the adapter offers (64 ticks asked for; MAX_ITERATIONS_PER_STEP on a fast adapter), and
        // ctx.pipelines is warm afterwards: every premise below ("batch 1 still in flight two ticks after its
        // call", "batch 2 still in flight when batch 1 lands") is a wall-clock race against the tick gap, and on a
        // loaded box (parallel test workers, another process on the card) a batch of a few ticks loses it
        const k = await calibrateHeavyStep(ctx, snapshot, await measureTickMs(), 64);
        const positions = new Float32Array(3 * n).fill(NaN);
        const sim = createForceAtlas2(ctx, {
            seed: 7,
            maxIter: 1_000_000,
            settleThreshold: 0,
            iterationsPerStep: k,
            maxInFlight: 2,
        });
        const counters = sim as unknown as { readonly coalesced: number }; // the @internal counter (contract 3.13)
        sim.load(snapshot, positions);
        // the helper's `submissions` (distinct promises) and the element's once-per-distinct-promise .catch
        // (9.4 item 4) both hinge on this identity; an async step() that returns the oldest batch's promise from
        // inside the async function would hand back a fresh wrapper per call and only show up in this suite as an
        // opaque iterationsDone !== submissions * k mismatch
        const p1 = sim.step(k);
        await nextTick(); // the submit happens after the (warm) bind promise; exact at the next macrotask
        expect(sim.inFlight).toBe(1);
        const p2 = sim.step(k); // inFlight 1 < 2: a second batch whatever the open cold-start ordering answer is
        expect(p2).not.toBe(p1);
        await nextTick();
        expect(sim.inFlight).toBe(2);
        expect(counters.coalesced).toBe(0);
        const p3 = sim.step(k);
        const p4 = sim.step(k);
        expect(p3).toBe(p1); // the OLDEST pending batch's promise: the very object the first call returned
        expect(p4).toBe(p3); // and the same object again on the next frame
        expect(counters.coalesced).toBe(2);
        await p1;
        expect(sim.inFlight).toBe(1); // batch 2 lands a full batch time (>= 2 ticks) after batch 1
        const p5 = sim.step(k); // a slot is free again: a new batch, a new promise
        expect(p5).not.toBe(p1);
        expect(p5).not.toBe(p2);
        await nextTick();
        expect(sim.inFlight).toBe(2);
        const p6 = sim.step(k);
        expect(p6).toBe(p2); // the oldest PENDING batch is now the second one, not the first ever
        expect(counters.coalesced).toBe(3);
        await sim.flush();
        expect(sim.inFlight).toBe(0);
        expect(sim.iterationsDone).toBe(3 * k); // p1, p2 and p5 submitted; p3, p4 and p6 ran nothing
        sim.dispose();
        ctx.release(snapshot);
    });

    it("600 ticks on random1k: submissions <= ticks, at most maxInFlight in flight, iterationsDone monotone, settled reported, positions written back", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const { snapshot } = fixture("random1k", gpuScale());
        const n = snapshot.nodeCount;
        await warmPipelines(ctx, snapshot); // the loop's own simulation below is never stepped before the loop
        const positions = new Float32Array(3 * n).fill(NaN);
        const sim = createForceAtlas2(ctx, { seed: 7, maxIter: 100, iterationsPerStep: 4, maxInFlight: 2 });
        sim.load(snapshot, positions);
        const seeded = Float32Array.from(positions);
        expect(seeded.every((v) => Number.isFinite(v))).toBe(true); // load() seeded the NaN rows in place (3.13)
        const loopOptions: FrameLoopOptions = { ticks: 600, iterationsPerStep: 4, maxInFlight: 2 };
        const report: FrameLoopReport = await runFrameLoop(sim, positions, loopOptions);
        expect(report.errors).toEqual([]);
        expect(report.submissions).toBeGreaterThanOrEqual(1);
        expect(report.submissions).toBeLessThanOrEqual(600);
        expect(report.maxObservedInFlight).toBeLessThanOrEqual(2);
        expect(report.iterationsDoneByTick).toHaveLength(600);
        expectMonotone(report.iterationsDoneByTick);
        expect(report.settledAtTick).not.toBeNull();
        expect(report.positionHolds).toEqual([]);
        expect(report.submissionsDuringPause).toBe(0);
        await sim.flush();
        expect(sim.inFlight).toBe(0);
        expect(sim.settled).toBe(true);
        // every submitted batch landed and carried 4 iterations; at most the one batch already in flight when the
        // budget was reached overshoots maxIter (7.19: the element's calls are throttled, not clairvoyant)
        expect(sim.iterationsDone).toBe(report.submissions * 4);
        expect(sim.iterationsDone).toBeLessThanOrEqual(104);
        let moved = 0;
        for (let i = 0; i < n; i++) {
            expect(Number.isFinite(positions[3 * i])).toBe(true);
            expect(Number.isFinite(positions[3 * i + 1])).toBe(true);
            expect(positions[3 * i + 2]).toBe(0); // 2D: z === center.z whatever was uploaded (7.13)
            if (positions[3 * i] !== seeded[3 * i] || positions[3 * i + 1] !== seeded[3 * i + 1]) {
                moved += 1;
            }
        }
        expect(moved).toBeGreaterThan(0);
        sim.dispose();
        ctx.release(snapshot);
    });

    it("a setPosition during flight lands in the following batch and is never overwritten by an older one", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const { snapshot } = fixture("random1k", gpuScale());
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
        const seeded = Float32Array.from(positions);
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
                    // a drag: the node is pinned in the same frame the pointer moves it (7.12), before the helper's
                    // setPosition and step; the pin does not reheat, the write does (D8)
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
        expect(inFlightAtWrite).toHaveLength(WRITES);
        expect(Math.max(...inFlightAtWrite)).toBeGreaterThanOrEqual(1); // at least one write was issued with a batch in flight
        expect(report.maxObservedInFlight).toBeLessThanOrEqual(2);
        expect(report.settledAtTick).toBeNull();
        // monotone except across a write tick: setPosition reheats, iterationsDone -> 0 (D8), visible at the next sample
        expectMonotone(report.iterationsDoneByTick, new Set(writes.map((w) => w.tick)));
        await sim.flush();
        expect(sim.inFlight).toBe(0);
        for (const w of writes) {
            // pinned and written: every batch computed after the write carried (x, y, 0) through toScene unchanged
            expect(positions[3 * w.index], `x of node ${w.index}`).toBe(w.x);
            expect(positions[3 * w.index + 1], `y of node ${w.index}`).toBe(w.y);
            expect(positions[3 * w.index + 2], `z of node ${w.index}`).toBe(0);
        }
        let moved = 0;
        for (let i = WRITES; i < n; i++) {
            if (positions[3 * i] !== seeded[3 * i] || positions[3 * i + 1] !== seeded[3 * i + 1]) {
                moved += 1;
            }
        }
        expect(moved).toBeGreaterThan(0); // the unpinned rows kept integrating after the last write
        sim.dispose();
        ctx.release(snapshot);
    });

    it("pause: exactly the in-flight batches land, flush() resolves, no submission for 100 ticks, a later step() continues", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const { snapshot } = fixture("random1k", gpuScale());
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
        const pauseAt = 40 + Math.floor(lcgUnit(7) * 60); // 69 for seed 7: the seeded "random tick" of spec 7.19
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
                const { stats } = sim; // the LAST COMPLETED batch (7.19); iteration counts every iteration since load()
                if (stats.iteration > 0 && !sampled.has(stats.iteration)) {
                    sampled.set(stats.iteration, controllerOf(stats));
                }
            },
        });
        expect(report.errors).toEqual([]);
        expect(report.submissionsDuringPause).toBe(0);
        expect(report.settledAtTick).toBeNull();
        expect(report.maxObservedInFlight).toBe(2);
        const pauseStart = inFlightByTick.findIndex((v, i) => i >= pauseAt && v === 2);
        expect(pauseStart).toBeGreaterThanOrEqual(pauseAt);
        expect(pauseStart + PAUSE).toBeLessThan(600);
        const atPause = report.iterationsDoneByTick[pauseStart];
        const afterPause = report.iterationsDoneByTick[pauseStart + PAUSE];
        expect(afterPause - atPause).toBe(2 * k); // exactly the two in-flight batches landed inside the window
        expect(inFlightByTick[pauseStart + PAUSE]).toBe(0); // flush() had nothing left to wait for
        for (let i = pauseStart + 1; i <= pauseStart + PAUSE; i++) {
            expect(report.iterationsDoneByTick[i]).toBeLessThanOrEqual(afterPause);
        }
        expectMonotone(report.iterationsDoneByTick); // no reheat anywhere: the pause is not a reset (7.19)
        expect(report.iterationsDoneByTick[599]).toBeGreaterThan(afterPause); // the later step() continued
        await sim.flush();
        expect(sim.iterationsDone).toBe(report.submissions * k);
        // the controller trace is continuous across the pause: every sampled (iteration -> speed, swing, traction,
        // speedEfficiency) equals an UNPAUSED run of the same simulation bitwise (same seed, same device, same
        // batching: the iteration sequence does not depend on when the batches were submitted)
        const maxIteration = Math.max(...sampled.keys());
        const referencePositions = new Float32Array(3 * n).fill(NaN);
        const reference = createForceAtlas2(ctx, options);
        reference.load(snapshot, referencePositions);
        const referenceByIteration = new Map<number, ControllerSample>();
        while (reference.stats.iteration < maxIteration) {
            await reference.step(k);
            referenceByIteration.set(reference.stats.iteration, controllerOf(reference.stats));
        }
        expect(sampled.size).toBeGreaterThan(2);
        let checkedAfterPause = 0;
        for (const [iteration, sample] of sampled) {
            const expected = referenceByIteration.get(iteration);
            if (expected === undefined) {
                throw new Error(`the unpaused reference never reported iteration ${iteration}`);
            }
            for (const key of ["speed", "swing", "traction", "speedEfficiency"] as const) {
                expect(
                    Object.is(sample[key], expected[key]),
                    `${key} at iteration ${iteration}: ${sample[key]} vs ${expected[key]}`,
                ).toBe(true);
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
