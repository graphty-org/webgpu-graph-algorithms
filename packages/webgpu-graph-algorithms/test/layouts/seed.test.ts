/**
 * The CPU port's LCG bit for bit (spec 7.2 "Initial positions", 9.3 seedPositions) and the NaN-row seeding rules
 * of contract 3.13: index order, the range box for a fresh layout, the bounding box of the finite rows for a
 * topology change, partial rows keeping their finite axes, 2D writing center.z, and the argument errors.
 */

import type { F32, GraphSnapshot } from "@graphty/graph-format";

import { isWebGpuGraphError } from "../../src/errors.js";
import { Lcg, LCG_A, LCG_C, LCG_M, seedPositions } from "../../src/layouts/seed.js";
import { pathEdges, snapshotOf } from "../helpers/graphs.js";

/** The port's generator, transcribed (layout/src/utils/random.ts), so expectations come from a second copy. */
function referenceLcg(seed: number): () => number {
    const m = 2 ** 35 - 31;
    let state = seed % m;
    return () => {
        state = (185852 * state + 1) % m;
        return state / m;
    };
}

const SEED_42_DRAWS = [
    0.00022717824342667956, 0.2215308973643538, 0.9603369599141427, 0.5446739632719223, 0.7454220133399382,
    0.17202325422353812, 0.8658439530362714, 0.8303596971306586, 0.01043112719557728, 0.6458515524579386,
];

function graph(n: number): GraphSnapshot {
    return snapshotOf(pathEdges(n), { nodeCount: n });
}

function nanRows(n: number): F32 {
    return new Float32Array(3 * n).fill(Number.NaN);
}

function errorOf(fn: () => void): { code: string; details: Readonly<Record<string, unknown>> } {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected the call to throw");
}

describe("Lcg", () => {
    it("carries the port's constants", () => {
        expect(LCG_M).toBe(2 ** 35 - 31);
        expect(LCG_M).toBe(34359738337);
        expect(LCG_A).toBe(185852);
        expect(LCG_C).toBe(1);
    });

    it("reproduces the first 10 draws of seed 42 exactly", () => {
        const lcg = new Lcg(42);
        expect(lcg.seed).toBe(42);
        const draws: number[] = [];
        for (let i = 0; i < 10; i++) {
            draws.push(lcg.next());
        }
        expect(draws).toEqual(SEED_42_DRAWS);
        // the first state is (185852 * 42 + 1) % m = 7805785
        expect(SEED_42_DRAWS[0]).toBe(7805785 / 34359738337);
    });

    it("agrees with the reference transcription over 1000 draws for several seeds", () => {
        for (const seed of [1, 7, 12345, 999999, 2 ** 31]) {
            const lcg = new Lcg(seed);
            const ref = referenceLcg(seed);
            for (let i = 0; i < 1000; i++) {
                expect(lcg.next()).toBe(ref());
            }
        }
    });

    it("preserves the port's quirks: a negative or fractional seed is used as given", () => {
        expect(new Lcg(-3).next()).toBe(-0.00001622698620494445);
        expect(new Lcg(42.5).next()).toBe(0.00022988274597814204);
    });

    it("seed 0 and null draw a random integer seed below 1e6 (the `seed || random` quirk)", () => {
        for (const seed of [0, null, -0]) {
            const lcg = new Lcg(seed);
            expect(Number.isInteger(lcg.seed)).toBe(true);
            expect(lcg.seed).toBeGreaterThanOrEqual(0);
            expect(lcg.seed).toBeLessThan(1000000);
            const v = lcg.next();
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThan(1);
        }
    });
});

describe("seedPositions", () => {
    it("fresh layout: every NaN row draws in [-1, 1) per axis in index order and 2D writes z = center.z", () => {
        const s = graph(3);
        const positions = nanRows(3);
        seedPositions(s, positions, 7, 2, 1, null, "fa2");
        // seed 7, u * 2 - 1, rounded to f32 (row-major: x0 y0 x1 y1 x2 y2)
        expect(Array.from(positions)).toEqual([
            -0.9999243021011353, -0.9261473417282104, 0, 0.6622453331947327, 0.6250821948051453, 0,
            -0.22551928460597992, -0.2091197371482849, 0,
        ]);
        const ref = referenceLcg(7);
        for (let i = 0; i < 3; i++) {
            expect(positions[3 * i]).toBe(Math.fround(ref() * 2 - 1));
            expect(positions[3 * i + 1]).toBe(Math.fround(ref() * 2 - 1));
            expect(positions[3 * i + 2]).toBe(0);
        }
    });

    it("applies scale and center to every drawn value (v * scale + center[axis]) and center.z in 2D", () => {
        const s = graph(3);
        const positions = nanRows(3);
        seedPositions(s, positions, 7, 2, 100, [10, 20, 5], "fa2");
        expect(Array.from(positions)).toEqual([
            -89.99242401123047, -72.61473846435547, 5, 76.22453308105469, 82.50821685791016, 5, -12.55192756652832,
            -0.9119739532470703, 5,
        ]);
    });

    it("3D draws three values per row", () => {
        const s = graph(3);
        const positions = nanRows(3);
        seedPositions(s, positions, 7, 3, 1, null, "fa2");
        expect(Array.from(positions)).toEqual([
            -0.9999243021011353, -0.9261473417282104, 0.6622453331947327, 0.6250821948051453, -0.22551928460597992,
            -0.2091197371482849, -0.32181841135025024, 0.4057227075099945, -0.623420238494873,
        ]);
    });

    it('range "fr" draws in [0, 1)', () => {
        const s = graph(2);
        const positions = nanRows(2);
        seedPositions(s, positions, 21, 2, 1, null, "fr");
        expect(Array.from(positions)).toEqual([
            0.00011358913616277277, 0.1107681542634964, 0, 0.482808381319046, 0.9053942561149597, 0,
        ]);
    });

    it("a partial row keeps its finite axes, draws the NaN ones inside the box, and its finite axes widen the box (PLAN DECISION 12)", () => {
        const s = graph(4);
        // row 0: x given, y NaN; rows 1-2 fully finite; row 3: x NaN, y given
        const positions = new Float32Array([0.5, Number.NaN, Number.NaN, 1, 3, 0, 2, 1, 0, Number.NaN, 2, 0]);
        seedPositions(s, positions, 5, 2, 1, null, "fa2");
        // box per axis over every finite COMPONENT: x [0.5, 2] (row 0's 0.5 widens it below the fully finite
        // rows' [1, 2]), y [1, 3]; draws in index order: row 0's y = 1 + u1 * 2, then row 3's x = 0.5 + u2 * 1.5
        // (under a "fully finite rows only" box row 3's x would be 1 + u2 * 1 = 1.0263774394989014 instead)
        expect(positions[0]).toBe(0.5);
        expect(positions[1]).toBe(1.000054121017456);
        expect(positions[2]).toBe(0); // unseeded 2D row: z = center.z
        expect(Array.from(positions.subarray(3, 9))).toEqual([1, 3, 0, 2, 1, 0]);
        expect(positions[9]).toBe(0.5395662188529968);
        expect(positions[10]).toBe(2);
        expect(positions[11]).toBe(0);
    });

    it("a NaN row among finite rows draws inside the finite rows' bounding box (scale 1)", () => {
        const s = graph(3);
        const positions = new Float32Array([-2, 1, 0, Number.NaN, Number.NaN, Number.NaN, 4, 3, 0]);
        seedPositions(s, positions, 9, 2, 1, null, "fa2");
        expect(Array.from(positions.subarray(3, 6))).toEqual([-1.9997079372406006, 1.0949503183364868, 0]);
        expect(Array.from(positions.subarray(0, 3))).toEqual([-2, 1, 0]);
        expect(Array.from(positions.subarray(6))).toEqual([4, 3, 0]);
    });

    it("the bounding box is taken in layout units, so scale and center round-trip", () => {
        const s = graph(3);
        const positions = new Float32Array([-2, 1, 0, Number.NaN, Number.NaN, Number.NaN, 4, 3, 0]);
        seedPositions(s, positions, 9, 2, 2, [1, -1, 0], "fa2");
        // box in layout units: x [-1.5, 1.5], y [1, 2]; written back as v * 2 + center -> the same scene values
        expect(Array.from(positions.subarray(3, 6))).toEqual([-1.9997079372406006, 1.0949503183364868, 0]);
    });

    it("when no row is fully finite every NaN component draws in the range box (an axis with finite values included)", () => {
        const s = graph(2);
        const positions = new Float32Array([1, Number.NaN, 5, Number.NaN, Number.NaN, 5]);
        seedPositions(s, positions, 13, 2, 1, null, "fa2");
        expect(Array.from(positions)).toEqual([
            1, -0.9998593926429749, 0, -0.8628543615341187, -0.20773831009864807, 0,
        ]);
    });

    it("a fully finite 2D row is never touched, whatever z holds", () => {
        const s = graph(2);
        const positions = new Float32Array([1, 2, 9, 3, 4, Number.NaN]);
        seedPositions(s, positions, 3, 2, 1, null, "fa2");
        expect(Array.from(positions)).toEqual([1, 2, 9, 3, 4, Number.NaN]);
    });

    it("an empty graph is a no-op", () => {
        const s = snapshotOf([], { nodeCount: 0 });
        const positions = new Float32Array(0);
        seedPositions(s, positions, 1, 2, 1, null, "fa2");
        expect(positions.length).toBe(0);
    });

    it("rejects a wrong positions length, a non-positive or non-finite scale, a non-finite center and a bad dim", () => {
        const s = graph(2);
        expect(errorOf(() => seedPositions(s, new Float32Array(5), 1, 2, 1, null, "fa2"))).toEqual({
            code: "E_INVALID_ARGUMENT",
            details: { argument: "positions", value: 5, expected: 6 },
        });
        for (const scale of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
            const e = errorOf(() => seedPositions(s, nanRows(2), 1, 2, scale, null, "fa2"));
            expect(e.code).toBe("E_INVALID_ARGUMENT");
            expect(e.details.argument).toBe("scale");
        }
        const centerError = errorOf(() => seedPositions(s, nanRows(2), 1, 2, 1, [Number.NaN, 0], "fa2"));
        expect(centerError.code).toBe("E_INVALID_ARGUMENT");
        expect(centerError.details.argument).toBe("center");
        const dimError = errorOf(() => seedPositions(s, nanRows(2), 1, 4 as unknown as 2, 1, null, "fa2"));
        expect(dimError.code).toBe("E_INVALID_ARGUMENT");
        expect(dimError.details.argument).toBe("dim");
        // a failed validation leaves the array untouched
        const untouched = nanRows(2);
        expect(() => seedPositions(s, untouched, 1, 2, 0, null, "fa2")).toThrow();
        expect(Array.from(untouched).every((v) => Number.isNaN(v))).toBe(true);
    });
});
