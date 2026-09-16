/**
 * The behaviour pins of layout/test/forceatlas2-layout.test.ts re-expressed on the GPU (spec 11.4; note 01 section
 * 2.1.8; contract 5.5 fa2-behaviour.test.ts): the empty graph (load + step resolve, no GPU work), a single node,
 * disconnected components separated by > 0.03 after 100 iterations, maxIter respected, completeGraph(6) spread > 0.3
 * in width and height, the same seed -> the same layout BITWISE on the same device, different seeds differ, and
 * z === center.z in 2D whatever z was uploaded (spec 7.13). None of these compares a coordinate with the oracle.
 */

import type { F32 } from "@graphty/graph-format";

import { GpuContext } from "../../src/context.js";
import { createForceAtlas2 } from "../../src/layouts/forceatlas2.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import { BASE_OPTIONS, NETWORKX, PAPER, paritySnapshot, startPositions, withSim } from "../helpers/fa2-parity.js";
import { snapshotOf } from "../helpers/graphs.js";
import { LeakCounter } from "../helpers/leak-counter.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { componentSeparation, spread } from "../helpers/metrics.js";
import { acquire, acquireRaw, requireGpu } from "../setup/gpu.js";

const CASE_TIMEOUT = 300_000;

/**
 * The bounding-box extent of one axis of a stride-3 array.
 */
function extent(positions: F32, n: number, axis: number): number {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
        lo = Math.min(lo, positions[3 * i + axis]);
        hi = Math.max(hi, positions[3 * i + axis]);
    }
    return hi - lo;
}

describe("FA2 behaviour pins (spec 11.4; the CPU layout test's pins on the GPU)", () => {
    let ctx: GpuContext;

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-behaviour" });
    });

    it("the empty graph: load() and step() resolve, settled at once, no GPU work (mapAsync 0), dispose leaves no buffer", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const counter = LeakCounter.wrap(device);
        const own = GpuContext.from(device);
        const s = snapshotOf([], { nodeCount: 0, label: "empty" });
        try {
            const sim = createForceAtlas2(own, { ...BASE_OPTIONS, ...PAPER });
            const positions = new Float32Array(0);
            sim.load(s, positions);
            expect(sim.settled).toBe(true);
            counter.resetMapAsync();
            await sim.step();
            await sim.run();
            expect(counter.mapAsyncCalls, "no readback for an empty graph").toBe(0);
            expect(positions).toHaveLength(0);
            expect(sim.iterationsDone).toBe(0);
            sim.dispose();
        } finally {
            own.release(s);
            own.dispose();
        }
        expect(counter.live, "live buffers after dispose").toBe(0);
        counter.restore();
    });

    it("a single node: paper mode never moves it (gravity toward its own centroid is zero); networkx mode pulls it straight toward the origin", async (t) => {
        requireGpu(t);
        const s = snapshotOf([], { nodeCount: 1, label: "one" });
        try {
            const start = Float32Array.from([0.8, 0.6, 0]);
            // settleWindow 200 > maxIter: an unmoving node has meanDisplacement exactly 0, and K1's
            // `meanDisp <= settleThreshold * rmsRadius` is `0 <= 0` = true on every fold, so with the
            // BASE_OPTIONS window of 10 run() would stop settled at iterationsDone 20; the window must
            // not be able to fire before maxIter for the iterationsDone pin to hold.
            await withSim(ctx, { ...BASE_OPTIONS, maxIter: 100, settleWindow: 200 }, PAPER, async (sim) => {
                const positions = Float32Array.from(start);
                sim.load(s, positions);
                await sim.run({ batch: 10 });
                expect(sim.iterationsDone).toBe(100);
                expect(sim.settled).toBe(true);
                expectBitwiseEqual(positions, start, "paper: one node stays put");
            });
            await withSim(ctx, BASE_OPTIONS, NETWORKX, async (sim) => {
                const positions = Float32Array.from(start);
                sim.load(s, positions);
                await sim.step(1);
                const r = Math.hypot(positions[0], positions[1]);
                expect(r, "networkx: closer to the origin after one iteration").toBeLessThan(1);
                expect(r, "networkx: not past the origin").toBeGreaterThan(0);
                // along the same ray: x / y = 0.8 / 0.6
                expect(Math.abs(positions[0] * 0.6 - positions[1] * 0.8)).toBeLessThan(1e-6);
                expect(positions[2]).toBe(0);
            });
        } finally {
            ctx.release(s);
        }
    });

    it("two disconnected triangles end up separated by more than 0.03 after 100 iterations", async (t) => {
        requireGpu(t);
        const s = snapshotOf(
            [
                [0, 1],
                [1, 2],
                [0, 2],
                [3, 4],
                [4, 5],
                [3, 5],
            ],
            { nodeCount: 6, label: "two-triangles" },
        );
        try {
            const options: ForceAtlas2Options = { ...BASE_OPTIONS, maxIter: 100 };
            await withSim(ctx, options, PAPER, async (sim) => {
                const positions = startPositions(s, options, false);
                sim.load(s, positions);
                await sim.run({ batch: 10 });
                expect(positions.every((v) => Number.isFinite(v))).toBe(true);
                expect(componentSeparation(s, positions, 2)).toBeGreaterThan(0.03);
            });
        } finally {
            ctx.release(s);
        }
    });

    it("maxIter is respected: run() stops at exactly maxIter with batch 1, reports settled, and a later step() does no work", async (t) => {
        requireGpu(t);
        const s = paritySnapshot("karate", 1, false);
        try {
            const options: ForceAtlas2Options = { ...BASE_OPTIONS, maxIter: 7 };
            await withSim(ctx, options, PAPER, async (sim) => {
                const positions = startPositions(s, options, false);
                sim.load(s, positions);
                const stats = await sim.run({ batch: 1 });
                expect(sim.iterationsDone).toBe(7);
                expect(stats.iteration).toBe(7);
                expect(sim.settled).toBe(true);
                const frozen = Float32Array.from(positions);
                await sim.step();
                expect(sim.iterationsDone, "a settled simulation submits nothing").toBe(7);
                expect(sim.inFlight).toBe(0);
                expectBitwiseEqual(positions, frozen, "positions untouched after the settled step()");
            });
            await withSim(ctx, options, PAPER, async (sim) => {
                const positions = startPositions(s, options, false);
                sim.load(s, positions);
                const stats = await sim.run();
                expect(sim.settled).toBe(true);
                expect(stats.iteration).toBe(sim.iterationsDone);
                expect(sim.iterationsDone).toBeGreaterThanOrEqual(7);
            });
        } finally {
            ctx.release(s);
        }
    });

    it("completeGraph(6): spread > 0.3 in width and in height after 100 iterations", async (t) => {
        requireGpu(t);
        const s = paritySnapshot("complete6", 1, false);
        try {
            const options: ForceAtlas2Options = { ...BASE_OPTIONS, maxIter: 100 };
            await withSim(ctx, options, PAPER, async (sim) => {
                const positions = startPositions(s, options, false);
                sim.load(s, positions);
                await sim.run({ batch: 10 });
                expect(extent(positions, 6, 0), "width").toBeGreaterThan(0.3);
                expect(extent(positions, 6, 1), "height").toBeGreaterThan(0.3);
                expect(spread(positions, 6, 2)).toBeGreaterThan(0.3);
            });
        } finally {
            ctx.release(s);
        }
    });

    it(
        "the same seed gives the same layout bitwise on the same device; a different seed gives a different layout",
        async (t) => {
            requireGpu(t);
            const s = paritySnapshot("karate", 1, false);
            try {
                const layoutWith = async (seed: number): Promise<F32> => {
                    const options: ForceAtlas2Options = { ...BASE_OPTIONS, seed, maxIter: 20 };
                    return await withSim(ctx, options, PAPER, async (sim) => {
                        const positions = new Float32Array(3 * s.nodeCount);
                        positions.fill(Number.NaN);
                        sim.load(s, positions);
                        await sim.run({ batch: 5 });
                        return positions;
                    });
                };
                const a = await layoutWith(42);
                const b = await layoutWith(42);
                const c = await layoutWith(43);
                expectBitwiseEqual(a, b, "seed 42 twice");
                expect(
                    a.some((v, i) => v !== c[i]),
                    "seed 43 differs",
                ).toBe(true);
            } finally {
                ctx.release(s);
            }
        },
        CASE_TIMEOUT,
    );

    it("2D writes z === center.z on every readback whatever z was uploaded (spec 7.13)", async (t) => {
        requireGpu(t);
        const s = paritySnapshot("karate", 1, false);
        try {
            const options: ForceAtlas2Options = { ...BASE_OPTIONS, center: [0, 0, 0.25] };
            await withSim(ctx, options, PAPER, async (sim) => {
                const positions = startPositions(s, options, false);
                for (let i = 0; i < s.nodeCount; i++) {
                    positions[3 * i + 2] = 5 + i;
                }
                sim.load(s, positions);
                await sim.step(3);
                for (let i = 0; i < s.nodeCount; i++) {
                    expect(positions[3 * i + 2], `z of node ${i}`).toBe(Math.fround(0.25));
                    expect(Number.isFinite(positions[3 * i])).toBe(true);
                    expect(Number.isFinite(positions[3 * i + 1])).toBe(true);
                }
            });
        } finally {
            ctx.release(s);
        }
    });
});
