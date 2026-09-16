/**
 * The exact-tier ForceAtlas2 benchmarks (spec 10.4 T-4 and the Node side of T-5, 7.8, 11.7; contract 6.3): one
 * `createForceAtlas2` simulation per rung of the exact ladder 1k / 4k / 8k / 16k / 32k / 65k (powers of two, E = 10n,
 * seeded G(n, m) with self-loops and parallels, 2D, `compat: "paper"`, `repulsion: "exact"`), warmed by the harness's
 * untimed first run (the bind / warm promise and the first pipeline creation), then `step(1)` timed `runs` times
 * (default 5) with an untimed `reheat()` before every run so a settled simulation never short-circuits. Two rows per
 * rung:
 *
 *   step(1) wall n=<n> m=<m> 2D [<label>]        the wall time of one iteration, its toScene and the 12n-byte readback
 *                                                 into the owner's array -- the per-frame cost an element pays
 *   ms/iteration (profiler|wall) n=<n> [<label>]  stats.msPerIteration after that step: the GPU time of the iteration's
 *                                                 passes when "timestamp-query" was granted (the profiler's sum), else the
 *                                                 batch wall time; the row name says which
 *
 * PLAN DECISION (P3-T7 item 1): the second row is the quantity spec 7.6 measured and the one T-4 and the 7.8 rule read;
 * the readback is the same for either tier and must not decide the crossover. A seventh rung, 10k / 100k (the
 * design-15.3 tier, not a power of two), carries T-4's "10k <= 1 ms" check and the Node side of T-5 (item 2).
 *
 * PLAN DECISION (P3-T7, recorded here and as finding G3-F1 of docs/decisions/G3.md; measured 2026-09-15 on the RTX
 * 4070 SUPER, driver 580.173.02): NVIDIA's power management drops the SM clock to its idle state (P8, 210 MHz, against
 * 2475 MHz at P0) when the GPU sees only sparse, sub-millisecond dispatches -- exactly the step(1) pattern of this group
 * after the CPU-heavy setup of the `upload` group -- and the profiler then reports the ladder 4-15x slower (1k 0.99 ms
 * instead of 0.096 ms) while `nvidia-smi` shows 210 MHz; a run of this group alone right after context creation, or
 * after ~200 ms of dense GPU work, sits at P0. Every rung therefore starts with a clock warm-up burst of at least
 * CLOCK_WARM_MS of back-to-back step(1) calls (untimed, reheat() before each so the budget is never reached) before the
 * harness's own warm-up run and the timed runs, so the rows record the kernels at the card's working clock rather than
 * the power governor. The burst is the same on every adapter (a software adapter is never a baseline anyway).
 * docs/decisions/G3.md section 10 records the measurement; a consumer's frame loop at 60 fps is dense enough not to hit
 * it, a paused one is not.
 *
 * The spec 7.8 re-fix rule lives here as `exactMaxNodesFromLadder` (item 4): the largest measured ladder n with
 * <= EXACT_BUDGET_MS per iteration, rounded down to a power of two. Its second clause ("not slower than the grid tier at
 * the same n") has no grid tier to compare with in P3 and is re-checked at G4. docs/decisions/G3.md section 3 records
 * the run of this rule over the committed baseline that produced the value src/constants.ts carries (appendix A of
 * G3.md keeps the script that applied it; this function is the only copy of the rule).
 */

import { type GpuContext } from "../src/context.js";
import { createForceAtlas2 } from "../src/layouts/forceatlas2.js";
import { type ForceAtlas2Stats, type GpuLayoutSimulation, type GpuLayoutTuning } from "../src/types/layout.js";
import { type ForceAtlas2Options } from "../src/types/options.js";
import { randomEdges, snapshotOf } from "./datasets.js";
import { bench, type BenchResult } from "./harness.js";

/** The group name of every row this file produces (the key of benchmarks/run.ts GROUPS). */
export const LAYOUT_EXACT_GROUP = "layout-exact";

/** One rung of the ladder: the label used in the row names and the node count. */
export interface LadderRung {
    readonly label: string;
    readonly nodes: number;
}

/**
 * The exact ladder of spec 7.8 / 10.4 T-4 / 11.7 -- the one ladder the crossover rule, the benchmarks and P4's
 * `calibrateLayout` (its 8k-65k subset) share. Powers of two, so the 7.8 "rounded down to a power of two" is the identity
 * on a rung.
 */
export const EXACT_LADDER: readonly LadderRung[] = [
    { label: "1k", nodes: 1024 },
    { label: "4k", nodes: 4096 },
    { label: "8k", nodes: 8192 },
    { label: "16k", nodes: 16384 },
    { label: "32k", nodes: 32768 },
    { label: "65k", nodes: 65536 },
];

/** The frame rung: 10k nodes / 100k edges (the design-15.3 tier; T-4's 10k check and the Node side of T-5). */
export const FRAME_RUNG: LadderRung = { label: "10k", nodes: 10_000 };

/** Edges per node of every rung (E = 10n, contract 6.3). */
export const LADDER_EDGE_FACTOR = 10;

/** The spec 7.8 per-iteration budget of the exact tier, milliseconds. */
export const EXACT_BUDGET_MS = 4;

/** The seed of every rung's G(n, m) and of the simulation's LCG (graph-format's benchmark seed). */
const SEED = 12345;

/** The least wall time of the untimed clock warm-up burst before a rung's timed runs (the header's PLAN DECISION, G3-F1). */
const CLOCK_WARM_MS = 500;

/**
 * The simulation options of every rung (PLAN DECISION P3-T7 item 3): the exact tier at any n, one iteration per step,
 * one batch in flight (step(1) awaits its own batch), paper mode, 2D, a maxIter reheat() keeps out of reach.
 */
const RUNG_OPTIONS: ForceAtlas2Options & GpuLayoutTuning = {
    dim: 2,
    seed: SEED,
    maxIter: 1000,
    iterationsPerStep: 1,
    maxInFlight: 1,
    repulsion: "exact",
    compat: "paper",
};

/** One rung's per-iteration figure as the 7.8 rule reads it. */
export interface LadderRow {
    readonly n: number;
    readonly msPerIteration: number;
}

/**
 * The largest power of two that is <= n (multiplication only: the house rule keeps bitwise operators off indices and
 * byte offsets, and this is neither, but one convention is simpler than two).
 * @param n - a positive number
 * @returns the largest power of two not above n
 */
export function floorPow2(n: number): number {
    if (!(n >= 1)) {
        throw new Error(`floorPow2: expected a positive number >= 1, got ${String(n)}`);
    }
    let p = 1;
    while (p * 2 <= n) {
        p *= 2;
    }
    return p;
}

/**
 * The spec 7.8 crossover rule, reduced to its budget clause: the largest measured n with msPerIteration <= budgetMs,
 * rounded down to a power of two. The order of the rows is irrelevant. The "not slower than the grid tier at the same n"
 * clause is not evaluable before P4 (no grid tier exists) and is re-checked at G4; P4 extends this function with it.
 * @param rows - the measured ladder rows (ladderRowsOf of a session's results)
 * @param budgetMs - the per-iteration budget (default EXACT_BUDGET_MS = 4)
 * @returns the exactMaxNodes value src/constants.ts carries
 */
export function exactMaxNodesFromLadder(rows: readonly LadderRow[], budgetMs: number = EXACT_BUDGET_MS): number {
    if (rows.length === 0) {
        throw new Error("exactMaxNodesFromLadder: no ladder rows");
    }
    let best = 0;
    for (const row of rows) {
        if (!Number.isFinite(row.msPerIteration) || row.msPerIteration < 0) {
            throw new Error(
                `exactMaxNodesFromLadder: the row n=${row.n} needs a finite, non-negative msPerIteration, got ${String(row.msPerIteration)}`,
            );
        }
        if (row.msPerIteration <= budgetMs && row.n > best) {
            best = row.n;
        }
    }
    if (best === 0) {
        throw new Error(
            `exactMaxNodesFromLadder: no rung within ${budgetMs} ms per iteration; exactMaxNodes cannot be re-fixed and the owner-decision rule of spec 10.4 applies`,
        );
    }
    return floorPow2(best);
}

/** The row-name prefix of the per-iteration rows and the pattern that extracts their n. */
const PER_ITERATION_ROW = /^ms\/iteration \((?:profiler|wall)\) n=(\d+) /;

/**
 * The per-iteration rows of the six ladder rungs of a session's results (the frame rung and every other group are
 * ignored), as the 7.8 rule reads them, sorted by n.
 * @param results - a session's results (any group)
 * @returns the ladder rows
 */
export function ladderRowsOf(results: readonly BenchResult[]): LadderRow[] {
    const rows: LadderRow[] = [];
    for (const r of results) {
        if (r.group !== LAYOUT_EXACT_GROUP) {
            continue;
        }
        const match = PER_ITERATION_ROW.exec(r.name);
        if (match === null) {
            continue;
        }
        const n = Number(match[1]);
        if (EXACT_LADDER.some((rung) => rung.nodes === n)) {
            rows.push({ n, msPerIteration: r.medianMs });
        }
    }
    rows.sort((a, b) => a.n - b.n);
    return rows;
}

/**
 * The median of a list of numbers (the harness's rule, repeated here because the harness does not export it).
 * @param values - the values
 * @returns the median
 */
function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * A BenchResult built from values the simulation reported (stats.msPerIteration) rather than from the harness's timer:
 * the LAST `runs` samples (the first one is the warm-up run's), median / min / max over them, no memory column.
 * @param name - the row name
 * @param samples - one sample per run, the warm-up's first
 * @param runs - the measured run count (the harness's BenchResult.runs)
 * @param items - items per run for the rate column
 * @param unit - what `items` counts
 * @returns the row
 */
function reportedRow(name: string, samples: readonly number[], runs: number, items: number, unit: string): BenchResult {
    if (samples.length < runs + 1) {
        throw new Error(`${name}: expected ${runs + 1} samples (warm-up + runs), got ${samples.length}`);
    }
    const measured = samples.slice(samples.length - runs);
    for (const s of measured) {
        if (!Number.isFinite(s) || s < 0) {
            throw new Error(
                `${name}: the simulation reported a non-finite msPerIteration (${String(s)}); the row cannot be recorded`,
            );
        }
    }
    const medianMs = median(measured);
    return {
        group: LAYOUT_EXACT_GROUP,
        name,
        medianMs,
        minMs: Math.min(...measured),
        maxMs: Math.max(...measured),
        runs,
        memoryDeltaBytes: 0,
        rate: medianMs === 0 ? null : (items / medianMs) * 1000,
        rateUnit: `${unit}/s`,
    };
}

/**
 * The clock warm-up burst of a rung (the header's PLAN DECISION, G3-F1): back-to-back step(1) calls, reheat() before
 * each, for at least `minMs` of wall time and at least one step, so the SM clock is at its working state when the timed
 * runs start. Untimed; nothing of it enters the rows.
 * @param sim - the rung's loaded simulation
 * @param minMs - the least wall time of the burst
 * @returns the steps run and the wall time spent
 */
async function warmClock(
    sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>,
    minMs: number,
): Promise<{ readonly steps: number; readonly ms: number }> {
    const start = performance.now();
    let steps = 0;
    while (steps < 1 || performance.now() - start < minMs) {
        sim.reheat();
        await sim.step(1);
        steps += 1;
    }
    return { steps, ms: performance.now() - start };
}

/**
 * Run the exact-tier layout benchmarks: the six ladder rungs and the 10k frame rung, two rows each.
 * @param ctx - the context (a hardware adapter; run.ts refuses software ones without --allow-software)
 * @returns the results
 */
export async function runLayoutExactBenchmarks(ctx: GpuContext): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const source = ctx.profiler !== null && ctx.profiler.enabled ? "profiler" : "wall";
    for (const rung of [...EXACT_LADDER, FRAME_RUNG]) {
        const m = rung.nodes * LADDER_EDGE_FACTOR;
        const snapshot = snapshotOf(randomEdges(rung.nodes, m, SEED), { label: `layout-exact/${rung.label}` });
        const positions = new Float32Array(3 * rung.nodes).fill(Number.NaN);
        const sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats> = createForceAtlas2(ctx, RUNG_OPTIONS);
        sim.load(snapshot, positions);
        const pairs = rung.nodes * (rung.nodes - 1);
        const samples: number[] = [];
        try {
            await warmClock(sim, CLOCK_WARM_MS);
            const wall = await bench(
                LAYOUT_EXACT_GROUP,
                `step(1) wall n=${rung.nodes} m=${m} 2D [${rung.label}]`,
                {
                    setup: () => {
                        sim.reheat();
                        return sim;
                    },
                    run: async (input) => {
                        await input.step(1);
                        samples.push(input.stats.msPerIteration ?? Number.NaN);
                        return input.stats;
                    },
                },
                { device: ctx.device, items: pairs, unit: "pairs" },
            );
            results.push(wall);
            results.push(
                reportedRow(
                    `ms/iteration (${source}) n=${rung.nodes} [${rung.label}]`,
                    samples,
                    wall.runs,
                    pairs,
                    "pairs",
                ),
            );
        } finally {
            sim.dispose();
            ctx.release(snapshot);
        }
    }
    return results;
}
