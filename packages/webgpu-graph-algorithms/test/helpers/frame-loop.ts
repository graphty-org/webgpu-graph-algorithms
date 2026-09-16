/**
 * test/helpers/frame-loop.ts -- the element's frame loop against a GPU layout simulation (spec 7.19, 9.4 item 4;
 * contract 5.2), for the 11.4 frame-loop test on Node and in the browser.
 *
 * The bridge logic, as SimulationLayoutEngine.step() does it: one SYNCHRONOUS step(k) per tick, the returned
 * promise never awaited, the error handler attached ONCE per distinct promise. The element remembers the last
 * promise it saw; this helper keeps a WeakSet of every promise it has handled, because with maxInFlight >= 2 a
 * coalesced call returns the OLDEST pending promise (spec 7.19 item 3), which a one-slot memory would treat as
 * new every other tick and attach a second handler to. Ticks are separated by one macrotask (setTimeout 0, the
 * same primitive in Node and Chromium) so readbacks land between ticks.
 *
 * Everything the report records is sampled at tick STARTS: ForceSimulation.step() submits after awaiting its
 * bind / warm promise and allocator.check() (contract 3.13), so inFlight and lastSubmittedBatchId lag the call by
 * a few microtasks and are exact only once the next macrotask begins. Submissions are therefore counted as
 * distinct promises returned while the simulation was not settled (a coalesced call returns a promise already
 * seen; a settled call resolves at once and submits nothing). A setPosition write is "released" when a promise
 * first returned at a tick >= the write's tick resolves -- its batch was submitted after the write's writeBuffer
 * (spec 7.12) -- while the resolution of any older promise, and every tick start before the release, must find
 * the written coordinates intact.
 *
 * The pause (spec 7.19): `pauseAt` is the EARLIEST tick of the pause; the pause begins at the first tick >=
 * pauseAt whose start finds inFlight === maxInFlight, calls flush() once (never awaited, so the ticks keep their
 * cadence) and calls no step() for `pauseTicks` ticks; a flush() that has not resolved when the window closes, a
 * pause that never starts, and every rejected promise are recorded in `errors`.
 *
 * Browser-safe: imports nothing from node:*.
 */

import { type F32 } from "@graphty/graph-format";

import { type ForceAtlas2Stats, type GpuLayoutSimulation } from "../../src/types/layout.js";
import { type ForceAtlas2Options } from "../../src/types/options.js";

/** Options of runFrameLoop (contract 5.2). */
export interface FrameLoopOptions {
    readonly ticks: number;
    readonly iterationsPerStep: number;
    readonly maxInFlight: number;
    /** Stop calling step() at this tick for `pauseTicks` ticks (null: no pause). */
    readonly pauseAt?: number | null | undefined;
    readonly pauseTicks?: number | undefined;
    /** setPosition calls to issue at given ticks. */
    readonly setPositionAt?:
        | readonly {
              readonly tick: number;
              readonly index: number;
              readonly x: number;
              readonly y: number;
              readonly z: number;
          }[]
        | undefined;
    readonly onTick?: ((tick: number) => void) | undefined;
}

/** What one run reports (contract 5.2). */
export interface FrameLoopReport {
    readonly submissions: number;
    readonly coalesced: number;
    readonly maxObservedInFlight: number;
    readonly iterationsDoneByTick: readonly number[];
    readonly settledAtTick: number | null;
    readonly errors: readonly unknown[];
    /** For each setPosition issued: whether the written coordinates were still in the owner's array at every later tick until a batch submitted after the write landed. */
    readonly positionHolds: readonly { readonly tick: number; readonly index: number; readonly held: boolean }[];
    readonly submissionsDuringPause: number;
}

/** Spec 7.19: "no submission happens for the next 100 ticks". */
const DEFAULT_PAUSE_TICKS = 100;

/** The @internal ForceSimulation counters the report is built from (contract 3.13). */
interface SimulationCounters {
    readonly lastSubmittedBatchId: number;
    readonly coalesced: number;
}

/** One setPosition issued by the loop and the state of its hold check. */
interface PositionWrite {
    readonly tick: number;
    readonly index: number;
    readonly x: number;
    readonly y: number;
    readonly z: number;
    held: boolean;
    released: boolean;
}

/**
 * The live counters of the simulation (the same object, read at every sample).
 * @param sim - the simulation under test
 * @returns the counters
 */
function countersOf(sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>): SimulationCounters {
    const candidate = sim as unknown as { readonly lastSubmittedBatchId?: unknown; readonly coalesced?: unknown };
    if (typeof candidate.lastSubmittedBatchId !== "number" || typeof candidate.coalesced !== "number") {
        throw new Error(
            "runFrameLoop: the simulation exposes no numeric lastSubmittedBatchId / coalesced counters (the @internal members of ForceSimulation, contract 3.13)",
        );
    }
    return candidate as SimulationCounters;
}

/**
 * One macrotask: what separates two render frames as far as promise resolution is concerned.
 * @returns a promise resolved by a zero-delay timer
 */
function macrotask(): Promise<void> {
    return new Promise<void>((resolve) => {
        setTimeout(() => {
            resolve();
        }, 0);
    });
}

/**
 * Whether the owner's array still carries a write's coordinates (as the f32 the simulation stored).
 * @param positions - the owner's stride-3 scene array
 * @param write - the write
 * @returns true when all three components are intact
 */
function rowHolds(positions: F32, write: PositionWrite): boolean {
    const base = 3 * write.index;
    return (
        positions[base] === Math.fround(write.x) &&
        positions[base + 1] === Math.fround(write.y) &&
        positions[base + 2] === Math.fround(write.z)
    );
}

/**
 * The element's bridge (spec 7.19): one synchronous step(k) per tick, .catch attached once per DISTINCT promise,
 * ticks separated by a macrotask so readbacks land; never awaits step().
 * @param sim - the simulation under test (a ForceSimulation: its @internal counters are read structurally)
 * @param positions - the owner's stride-3 scene array the simulation writes back into
 * @param options - ticks, iterations per step, maxInFlight, the optional pause, the scheduled writes and the hook
 * @returns the report of the run
 */
export async function runFrameLoop(
    sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>,
    positions: F32,
    options: FrameLoopOptions,
): Promise<FrameLoopReport> {
    const counters = countersOf(sim);
    const pauseTicks = options.pauseTicks ?? DEFAULT_PAUSE_TICKS;
    const pauseAt = options.pauseAt ?? null;
    const scheduledWrites = options.setPositionAt ?? [];
    const writes: PositionWrite[] = [];
    const errors: unknown[] = [];
    const iterationsDoneByTick: number[] = [];
    const handled = new WeakSet<Promise<void>>();
    const coalescedAtStart = counters.coalesced;
    let submissions = 0;
    let submissionsDuringPause = 0;
    let maxObservedInFlight = 0;
    let settledAtTick: number | null = null;
    let lastBatchIdSeen = counters.lastSubmittedBatchId;
    let pauseStart: number | null = null;
    let pausing = false;
    let flushResolved = false;

    const checkHolds = (): void => {
        for (const write of writes) {
            if (!write.released && !rowHolds(positions, write)) {
                write.held = false;
            }
        }
    };
    // A promise first returned at `firstTick` resolved: its batch landed (or, when the simulation was settled,
    // nothing was submitted and nothing changed). Newer than a write -> the write is released; older -> the row
    // must still hold.
    const onLanded = (firstTick: number): void => {
        for (const write of writes) {
            if (write.released) {
                continue;
            }
            if (firstTick >= write.tick) {
                write.released = true;
            } else if (!rowHolds(positions, write)) {
                write.held = false;
            }
        }
    };

    for (let tick = 0; tick < options.ticks; tick++) {
        // (a) the tick-start sample: everything that landed before this macrotask is visible now
        const inFlightAtStart = sim.inFlight;
        maxObservedInFlight = Math.max(maxObservedInFlight, inFlightAtStart);
        iterationsDoneByTick.push(sim.iterationsDone);
        if (settledAtTick === null && sim.settled) {
            settledAtTick = tick;
        }
        checkHolds();
        if (pausing && counters.lastSubmittedBatchId !== lastBatchIdSeen) {
            submissionsDuringPause += 1; // nothing called step(): the simulation submitted on its own
        }
        lastBatchIdSeen = counters.lastSubmittedBatchId;
        // (b) the pause window
        if (pausing && pauseStart !== null && tick >= pauseStart + pauseTicks) {
            pausing = false;
            if (!flushResolved) {
                errors.push(
                    new Error(
                        `flush() did not resolve inside the ${pauseTicks}-tick pause that started at tick ${pauseStart}`,
                    ),
                );
            }
        }
        if (
            !pausing &&
            pauseStart === null &&
            pauseAt !== null &&
            tick >= pauseAt &&
            inFlightAtStart === options.maxInFlight
        ) {
            pausing = true;
            pauseStart = tick;
            sim.flush().then(
                () => {
                    flushResolved = true;
                },
                (err: unknown) => {
                    errors.push(err);
                },
            );
        }
        // (c) the caller's hook: sees the state the report recorded for this tick, runs before the writes and the step
        options.onTick?.(tick);
        // (d) the element's frame: scheduled writes, then ONE synchronous step(k), never awaited
        if (!pausing) {
            for (const scheduled of scheduledWrites) {
                if (scheduled.tick === tick) {
                    sim.setPosition(scheduled.index, scheduled.x, scheduled.y, scheduled.z);
                    writes.push({
                        tick,
                        index: scheduled.index,
                        x: scheduled.x,
                        y: scheduled.y,
                        z: scheduled.z,
                        held: true,
                        released: false,
                    });
                }
            }
            const settledBefore = sim.settled;
            const promise = sim.step(options.iterationsPerStep);
            if (!handled.has(promise)) {
                handled.add(promise);
                if (!settledBefore) {
                    submissions += 1;
                }
                promise.then(
                    () => {
                        onLanded(tick);
                    },
                    (err: unknown) => {
                        errors.push(err);
                    },
                );
            }
            maxObservedInFlight = Math.max(maxObservedInFlight, sim.inFlight);
        }
        // (e) the frame ends: readbacks land, handlers run, before the next tick starts
        await macrotask();
    }
    if (pauseAt !== null && pauseStart === null) {
        errors.push(
            new Error(`pause never started: no tick >= ${pauseAt} began with inFlight === ${options.maxInFlight}`),
        );
    }
    if (pausing && !flushResolved) {
        errors.push(
            new Error("flush() did not resolve before the loop ended (the pause window ran past the last tick)"),
        );
    }
    return {
        submissions,
        coalesced: counters.coalesced - coalescedAtStart,
        maxObservedInFlight,
        iterationsDoneByTick,
        settledAtTick,
        errors,
        positionHolds: writes.map((write) => ({ tick: write.tick, index: write.index, held: write.held })),
        submissionsDuringPause,
    };
}
