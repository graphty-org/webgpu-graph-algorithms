/**
 * The layout property rows of spec 11.3 (fast-check, numRuns 200; contract 5.5 fa2-properties.test.ts): fixed nodes
 * never move (random masks incl. the all-fixed mask, which settles within settleWindow + 1 steps, PLAN DECISION 12);
 * setPosition visible in the next readback and never clobbered by an older batch; settled within maxIter; reheat on
 * unpin / setPosition / load, not on pin; speed NOT reset by setPosition (untouched at the call, and the next
 * iteration continues from it through the re-synchronised oracle, PLAN DECISIONS 13 / 17); pin A, remove B < A,
 * load(next) with the remapped array and a re-issued mask -> A still fixed; results ArrayBuffer-typed of exact
 * length; z === center.z in 2D; per-node displacement <= speed |F| / (1 + sqrt(speed swing_i)).
 *
 * Sizing (spec 11.3 property row: "generators sized by gpuScale()"): karate cannot shrink (34 nodes), so on a
 * software adapter (gpuScale() < 1) the per-run submission counts shrink instead -- STEPS bounds the before / after
 * step generators of the fixed-node property (4 -> 2) and HALF the speed-continuity horizon on each side of the
 * setPosition (5 -> 3; the re-synchronised oracles follow). numRuns stays at the spec's 200 on every adapter.
 */

import { INVALID_INDEX, makeMask, maskSet } from "@graphty/graph-format";
import fc from "fast-check";

import type { GpuContext } from "../../src/context.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import {
    asF32,
    BASE_OPTIONS,
    debugStages,
    PAPER,
    paritySnapshot,
    pinMask,
    readState,
    resyncReports,
    resyncTrace,
    startPositions,
    withSim,
    xyzOf,
} from "../helpers/fa2-parity.js";
import { assertCheckPasses } from "../helpers/sabotage.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

const NUM_RUNS = 200;
const CASE_TIMEOUT = 300_000;
/** Upper bound of the before / after step generators of the fixed-node property (gpuScale() sizing, file header). */
const STEPS = gpuScale() < 1 ? 2 : 4;
/** Iterations on each side of the setPosition in the speed-continuity property (gpuScale() sizing, file header). */
const HALF = gpuScale() < 1 ? 3 : 5;

/** The smallest normal f32 (2^-126): WGSL permits flushing subnormals to zero, so a subnormal coordinate written to the device may legitimately read back as +0. */
const MIN_NORMAL_F32 = 2 ** -126;

/**
 * A finite f32 in [min, max] that a correct kernel reads back bitwise: -0 excluded (toScene computes pos * scale +
 * center = -0 * 1 + 0 = +0 under IEEE) and subnormals excluded (WGSL may flush them to zero); 0 itself is kept.
 * @param min - the lower bound
 * @param max - the upper bound
 * @returns the arbitrary
 */
function bitwiseStableFloat(min: number, max: number): fc.Arbitrary<number> {
    return fc
        .float({ min, max, noNaN: true })
        .filter((v) => !Object.is(v, -0) && (v === 0 || Math.abs(v) >= MIN_NORMAL_F32));
}

/** One macrotask, so a submission started by step() lands before the caller continues. */
function tick(): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, 0);
    });
}

describe("FA2 properties (spec 11.3; fast-check numRuns 200)", () => {
    let ctx: GpuContext;
    const s = paritySnapshot("karate", 1, false);
    const n = s.nodeCount;
    const start = startPositions(s, BASE_OPTIONS, false);

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-properties" });
    });

    afterAll(() => {
        ctx.release(s);
    });

    it(
        "fixed nodes never move: a random mask set between steps keeps every pinned row bitwise where it was",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                fc.asyncProperty(
                    fc.array(fc.boolean(), { minLength: n, maxLength: n }),
                    fc.integer({ min: 1, max: STEPS }),
                    fc.integer({ min: 1, max: STEPS }),
                    async (bits, before, after) => {
                        const mask = makeMask(n);
                        bits.forEach((b, i) => {
                            maskSet(mask, i, b);
                        });
                        await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
                            const positions = Float32Array.from(start);
                            sim.load(s, positions);
                            await sim.step(before);
                            const pinnedAt = Float32Array.from(positions);
                            sim.setFixed(mask);
                            await sim.step(after);
                            for (let i = 0; i < n; i++) {
                                if (!bits[i]) {
                                    continue;
                                }
                                for (let k = 0; k < 3; k++) {
                                    if (!Object.is(positions[3 * i + k], pinnedAt[3 * i + k])) {
                                        throw new Error(`pinned node ${i} moved on component ${k}`);
                                    }
                                }
                            }
                        });
                    },
                ),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it("the all-fixed mask: nothing moves and settled is reported within settleWindow + 1 single-iteration steps (PLAN DECISION 12)", async (t) => {
        requireGpu(t);
        const settleWindow = 3;
        const mask = makeMask(n);
        for (let i = 0; i < n; i++) {
            maskSet(mask, i, true);
        }
        await withSim(ctx, { ...BASE_OPTIONS, settleThreshold: 1e-3, settleWindow }, PAPER, async (sim) => {
            const positions = Float32Array.from(start);
            sim.load(s, positions);
            sim.setFixed(mask);
            let steps = 0;
            while (!sim.settled && steps < settleWindow + 1) {
                await sim.step(1);
                steps++;
            }
            expect(sim.settled, `settled after ${steps} steps`).toBe(true);
            expect(sim.stats.meanDisplacement).toBe(0);
            for (let i = 0; i < 3 * n; i++) {
                if (!Object.is(positions[i], start[i])) {
                    throw new Error(`component ${i} moved under the all-fixed mask`);
                }
            }
        });
    });

    it(
        "setPosition is visible in the next readback and never clobbered by an older batch (the override list, spec 7.12)",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 0, max: n - 1 }),
                    // the post-batch reads below are bitwise (toBe is Object.is): -0 and subnormals are excluded
                    // because a correct kernel returns +0 for both (bitwiseStableFloat)
                    bitwiseStableFloat(-2, 2),
                    bitwiseStableFloat(-2, 2),
                    async (i, x, y) => {
                        await withSim(ctx, { ...BASE_OPTIONS, maxInFlight: 2 }, PAPER, async (sim) => {
                            const positions = Float32Array.from(start);
                            sim.load(s, positions);
                            sim.setFixed(pinMask(n, i));
                            await sim.step(1);
                            const submittedBefore = sim.lastSubmittedBatchId;
                            const older = sim.step(2);
                            while (sim.lastSubmittedBatchId === submittedBefore) {
                                await tick();
                            }
                            sim.setPosition(i, x, y, 0);
                            expect(sim.overrides.get(i), "the override records the last submitted batch").toBe(
                                sim.lastSubmittedBatchId,
                            );
                            const newer = sim.step(2);
                            await older;
                            expect(positions[3 * i], "x after the older batch landed").toBe(x);
                            expect(positions[3 * i + 1], "y after the older batch landed").toBe(y);
                            await newer;
                            expect(positions[3 * i], "x after the newer batch (pinned, computed from the write)").toBe(
                                x,
                            );
                            expect(positions[3 * i + 1], "y after the newer batch").toBe(y);
                            expect(positions[3 * i + 2]).toBe(0);
                            expect(
                                sim.overrides.has(i),
                                "the override is cleared by a batch newer than the write",
                            ).toBe(false);
                        });
                    },
                ),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it(
        "settled within maxIter: run({ batch: 1 }) stops at maxIter exactly and stats.iteration agrees",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                fc.asyncProperty(fc.integer({ min: 1, max: 30 }), async (maxIter) => {
                    const options: ForceAtlas2Options = { ...BASE_OPTIONS, maxIter };
                    await withSim(ctx, options, PAPER, async (sim) => {
                        const positions = Float32Array.from(start);
                        sim.load(s, positions);
                        const stats = await sim.run({ batch: 1 });
                        expect(sim.settled).toBe(true);
                        expect(sim.iterationsDone).toBe(maxIter);
                        expect(stats.iteration).toBe(maxIter);
                        expect(stats.trace).toHaveLength(1);
                    });
                }),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it("reheat on unpin, setPosition and load, not on pin (D8)", async (t) => {
        requireGpu(t);
        await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
            const positions = Float32Array.from(start);
            sim.load(s, positions);
            await sim.step(5);
            expect(sim.iterationsDone).toBe(5);
            sim.setFixed(pinMask(n, 3));
            expect(sim.iterationsDone, "a pin does not reheat").toBe(5);
            sim.setFixed(makeMask(n));
            expect(sim.iterationsDone, "an unpin reheats").toBe(0);
            await sim.step(3);
            expect(sim.iterationsDone).toBe(3);
            sim.setPosition(0, 0.1, 0.2, 0);
            expect(sim.iterationsDone, "setPosition reheats").toBe(0);
            await sim.step(2);
            expect(sim.iterationsDone).toBe(2);
            sim.load(s, positions);
            expect(sim.iterationsDone, "load reheats").toBe(0);
        });
    });

    it(
        "speed is NOT reset by setPosition (D8): speed / speedEfficiency untouched at the call, and the next iteration continues from them (re-synchronised, PLAN DECISIONS 13 / 17)",
        async (t) => {
            requireGpu(t);
            // D8: setPosition() reheats (iterationsDone = 0, settledCount = 0) and NOTHING else -- the controller
            // keeps its state at the moment of the call, and the next iteration's K4 continues from it. The second
            // half is proved through the re-synchronised oracle (resyncTrace): before every iteration a fresh oracle
            // is seeded with the GPU's own iteration-start state, so at the iteration after the drag the oracle
            // starts from the moved position AND the speed the GPU reported before the call; a controller reset on
            // the device would show as that iteration's speed disagreeing by O(1). The earlier form of this property
            // compared a free-running 10-iteration trace at the f32 trace tolerance and was red on NVIDIA for a few
            // of the 200 drags (a drag that lands a node next to another creates a near-coincident pair whose forces
            // amplify f32 rounding chaotically: G3-F4); the re-synchronised form compares one iteration at a time,
            // free of that amplification, at the measured re-synchronised floors.
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 0, max: n - 1 }),
                    bitwiseStableFloat(-3, 3),
                    bitwiseStableFloat(-3, 3),
                    async (i, x, y) => {
                        let before: { readonly speed: number; readonly speedEfficiency: number } | null = null;
                        let after: { readonly speed: number; readonly speedEfficiency: number } | null = null;
                        const run = await resyncTrace(ctx, s, start, BASE_OPTIONS, PAPER, 2 * HALF, {
                            beforeStep: (sim, k) => {
                                if (k !== HALF) {
                                    return;
                                }
                                before = { speed: sim.stats.speed, speedEfficiency: sim.stats.speedEfficiency };
                                sim.setPosition(i, x, y, 0);
                                after = { speed: sim.stats.speed, speedEfficiency: sim.stats.speedEfficiency };
                                expect(sim.iterationsDone, "setPosition reheats").toBe(0);
                            },
                        });
                        expect(before, "the hook ran").not.toBeNull();
                        expect(after, "stats.speed / speedEfficiency untouched by setPosition").toEqual(before);
                        const reports = resyncReports(run, `node ${i} -> (${x}, ${y})`);
                        assertCheckPasses(reports.f32);
                        assertCheckPasses(reports.f64);
                    },
                ),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it(
        "pin A, remove B < A, load(next) with the remapped array and a re-issued mask -> A is still fixed",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                fc.asyncProperty(
                    fc.integer({ min: 1, max: n - 1 }),
                    fc.integer({ min: 0, max: n - 2 }),
                    async (a, bRaw) => {
                        const b = Math.min(bRaw, a - 1);
                        const keep = Uint32Array.from(Array.from({ length: n }, (_, i) => i).filter((i) => i !== b));
                        const derived = s.inducedSubgraph(keep);
                        const next = derived.snapshot;
                        const remap = derived.nodeRemap;
                        if (remap === null) {
                            throw new Error("inducedSubgraph without a node remap");
                        }
                        try {
                            await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
                                const positions = Float32Array.from(start);
                                sim.load(s, positions);
                                sim.setFixed(pinMask(n, a));
                                await sim.step(2);
                                const nextPositions = new Float32Array(3 * next.nodeCount);
                                for (let i = 0; i < n; i++) {
                                    const j = remap[i];
                                    if (j === INVALID_INDEX) {
                                        continue;
                                    }
                                    nextPositions[3 * j] = positions[3 * i];
                                    nextPositions[3 * j + 1] = positions[3 * i + 1];
                                    nextPositions[3 * j + 2] = positions[3 * i + 2];
                                }
                                const aNew = remap[a];
                                expect(aNew).toBe(a - 1);
                                sim.load(next, nextPositions);
                                sim.setFixed(pinMask(next.nodeCount, aNew));
                                const held = [
                                    nextPositions[3 * aNew],
                                    nextPositions[3 * aNew + 1],
                                    nextPositions[3 * aNew + 2],
                                ];
                                await sim.step(3);
                                expect(nextPositions[3 * aNew]).toBe(held[0]);
                                expect(nextPositions[3 * aNew + 1]).toBe(held[1]);
                                expect(nextPositions[3 * aNew + 2]).toBe(held[2]);
                                // the node that took B's old index is NOT fixed (the mask was re-issued for the new index space)
                                if (b !== aNew) {
                                    const moved = [nextPositions[3 * b], nextPositions[3 * b + 1]];
                                    expect(
                                        moved[0] !== positions[3 * (b + 1)] || moved[1] !== positions[3 * (b + 1) + 1],
                                        "the remapped neighbour moves",
                                    ).toBe(true);
                                }
                            });
                        } finally {
                            ctx.release(next);
                        }
                    },
                ),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it("results are ArrayBuffer-typed of exact length: the owner's array is the one written, stats.trace has k records", async (t) => {
        requireGpu(t);
        await withSim(ctx, BASE_OPTIONS, PAPER, async (sim) => {
            const positions = Float32Array.from(start);
            sim.load(s, positions);
            await sim.step(4);
            expect(positions.buffer).toBeInstanceOf(ArrayBuffer);
            expect(positions).toHaveLength(3 * n);
            expect(sim.stats.trace).toHaveLength(4);
            expect(sim.stats.centroid).toHaveLength(3);
        });
    });

    it(
        "2D writes z === center.z whatever z was uploaded, for random z and random center.z",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                // -0 is excluded from cz: toScene's z is 0 * scale + center.z = +0 for cz = -0 (IEEE), and the
                // read below is bitwise (toBe is Object.is)
                fc.asyncProperty(
                    fc.float({ min: -10, max: 10, noNaN: true }).filter((v) => !Object.is(v, -0)),
                    fc.integer({ min: 0, max: 1000 }),
                    async (cz, zSeed) => {
                        const options: ForceAtlas2Options = { ...BASE_OPTIONS, center: [0, 0, cz] };
                        await withSim(ctx, options, PAPER, async (sim) => {
                            const positions = startPositions(s, options, false);
                            for (let i = 0; i < n; i++) {
                                positions[3 * i + 2] = ((zSeed + 7 * i) % 13) - 6;
                            }
                            sim.load(s, positions);
                            await sim.step(2);
                            for (let i = 0; i < n; i++) {
                                expect(positions[3 * i + 2]).toBe(Math.fround(cz));
                            }
                        });
                    },
                ),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );

    it(
        "per-node displacement never exceeds speed |F| / (1 + sqrt(speed swing_i)) (equality up to f32 rounding in the first iteration)",
        async (t) => {
            requireGpu(t);
            await fc.assert(
                fc.asyncProperty(fc.integer({ min: 1, max: 100_000 }), async (seed) => {
                    const options: ForceAtlas2Options = { ...BASE_OPTIONS, seed };
                    await withSim(ctx, options, PAPER, async (sim) => {
                        const positions = startPositions(s, options, false);
                        sim.load(s, positions);
                        const mass = s.outDegree();
                        const st = debugStages(sim);
                        await st.run("K5");
                        const force = asF32(await st.read("force"));
                        const after = xyzOf(asF32(await st.read("positions")), n);
                        const { speed } = readState(await st.read("state"));
                        for (let i = 0; i < n; i++) {
                            const f = Math.hypot(force[3 * i], force[3 * i + 1], force[3 * i + 2]);
                            const swing = (mass[i] + 1) * f; // paper mode, iteration 1: oldForce is 0
                            const bound = (speed * f) / (1 + Math.sqrt(speed * swing));
                            const dp = Math.hypot(
                                after[3 * i] - positions[3 * i],
                                after[3 * i + 1] - positions[3 * i + 1],
                            );
                            expect(dp, `node ${i}`).toBeLessThanOrEqual(bound * (1 + 1e-5) + 1e-7);
                            expect(after[3 * i + 2], `2D never integrates z (node ${i})`).toBe(0);
                        }
                    });
                }),
                { numRuns: NUM_RUNS },
            );
        },
        CASE_TIMEOUT,
    );
});
