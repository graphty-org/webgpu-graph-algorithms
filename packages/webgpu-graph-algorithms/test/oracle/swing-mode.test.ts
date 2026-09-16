/**
 * The SWING_MODE unit test (spec 11.4 bullet 2 "the paper-mode CONTROLLER differs from NetworkX's by the two
 * SWING_MODE lines, which a unit test pins against hand-computed values on a three-node graph"; contract 5.5 P3-T4).
 *
 * The graph is the path 0-1-2 at p0 = (-1, 0), p1 = (0, 1), p2 = (1, 0) with the default masses degree + 1 =
 * (2, 3, 2), scalingRatio 2, gravity 1 toward the ORIGIN in both modes (so the forces are identical and only the
 * controller differs), jitterTolerance 1, 2D. Every expected number below was derived by hand from the spec 7.2
 * table and networkx 3.4.2 layout.py (never from the oracle): the derivation of iteration 1 in closed form is in
 * the comments, iteration 2 (nested square roots) was evaluated with the same formulas in a straight-line script
 * and pasted to full double precision; the comparison tolerance is 1e-12 relative.
 *
 * Iteration 1, both modes. Attraction F_i = sum_j (p_j - p_i): (1, 1), (0, -2), (-1, 1). Repulsion F_i = sum_j
 * (p_i - p_j) k m_i m_j / d^2 with d01 = d12 = sqrt(2), d02 = 2: node 0: (-1, -1) x 6 + (-2, 0) x 2 = (-10, -6);
 * node 1: (1, 1) x 6 + (-1, 1) x 6 = (0, 12); node 2: (10, -6). Gravity -m_i p_i / |p_i| with |p_i| = 1: (2, 0),
 * (0, -3), (-2, 0). Total F = (-7, -5), (0, 7), (7, -5); |F| = sqrt(74), 7, sqrt(74).
 * networkx mode: swing_i = m_i |p_i - F_i| = 2 sqrt(61), 18, 2 sqrt(61); traction_i = 0.5 m_i |p_i + F_i| =
 * sqrt(89), 12, sqrt(89); global swing = 1 + 18 + 4 sqrt(61), traction = 1 + 12 + 2 sqrt(89) (accumulated from 1).
 * paper mode: F(t-1) = 0, so swing_i = m_i |F_i| = 2 sqrt(74), 21, 2 sqrt(74); traction_i = 0.5 m_i |F_i| =
 * sqrt(74), 10.5, sqrt(74); global = fresh sums 21 + 4 sqrt(74) and 10.5 + 2 sqrt(74) (ratio exactly 2, so the
 * `> 2.0` halving is NOT taken); with node 2 fixed its two terms are excluded: 21 + 2 sqrt(74), 10.5 + sqrt(74).
 * estimateFactor(n = 3): optJitter = 0.05 sqrt(3), minJitter = sqrt(optJitter) = 0.2942830956382712, other =
 * min(10, optJitter traction / 9), jitter = max(minJitter, other); targetSpeed = jitter eff traction / swing;
 * swing > jitter traction -> eff = 0.7; speed = 1 + min(target - 1, 0.5) = target.
 * Local factor_i = speed / (1 + sqrt(speed m_i |F_i|)) in both modes at iteration 1 (paper: |F - 0|); at iteration
 * 2 paper uses m_i |F(2) - F(1)| and networkx m_i |F(2)|, which the displacements below distinguish.
 */

import { makeMask, maskSet } from "@graphty/graph-format";

import { snapshotOf } from "../helpers/graphs.js";
import {
    estimateFactor,
    type EstimateFactorResult,
    ForceAtlas2Oracle,
    kickDir,
    type OracleOptions,
    type OracleStages,
    type OracleTraceRecord,
} from "./forceatlas2.js";

const S61 = 2 * Math.sqrt(61); // 15.620499351813308
const S74 = Math.sqrt(74); // 8.602325267042627
const S89 = Math.sqrt(89); // 9.433981132056603
const START = [-1, 0, 0, 0, 1, 0, 1, 0, 0];
const REL = 1e-12;

const path3 = snapshotOf(
    [
        [0, 1],
        [1, 2],
    ],
    { nodeCount: 3 },
);

function expectClose(actual: number, expected: number, label: string, rel = REL): void {
    const bound = rel * Math.max(1, Math.abs(expected));
    expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(bound);
}

function expectCloseArray(actual: ArrayLike<number>, expected: readonly number[], label: string, rel = REL): void {
    expect(actual.length, `${label}: length`).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
        expectClose(actual[i], expected[i], `${label}[${i}]`, rel);
    }
}

function oracle(compat: "paper" | "networkx", extra: Partial<OracleOptions> = {}): ForceAtlas2Oracle {
    return new ForceAtlas2Oracle(path3, START, { compat, precision: "f64", gravityCenter: 1, dim: 2, ...extra });
}

function fixedNode2() {
    const mask = makeMask(3);
    maskSet(mask, 2, true);
    return mask;
}

describe("iteration-1 forces (shared by both modes; origin gravity)", () => {
    for (const compat of ["paper", "networkx"] as const) {
        it(`${compat}: attraction, repulsion, gravity and force stages equal the hand values`, () => {
            const o = oracle(compat);
            const record: OracleTraceRecord = o.step();
            const { stages }: { stages: OracleStages } = o;
            expectCloseArray(stages.attraction, [1, 1, 0, 0, -2, 0, -1, 1, 0], "attraction");
            expectCloseArray(stages.repulsion, [-10, -6, 0, 0, 12, 0, 10, -6, 0], "repulsion");
            expectCloseArray(stages.gravity, [2, 0, 0, 0, -3, 0, -2, 0, 0], "gravity");
            expectCloseArray(stages.force, [-7, -5, 0, 0, 7, 0, 7, -5, 0], "force");
            expectCloseArray(stages.oldForce, [0, 0, 0, 0, 0, 0, 0, 0, 0], "oldForce before any integrate");
            // K1 on the first iteration keeps load()'s statistics: centroid (0, 1/3), |q|^2 = 10/9, 4/9, 10/9
            expectCloseArray(record.centroid, [0, 1 / 3, 0], "centroid");
            expectClose(record.rmsRadius, Math.sqrt(8 / 9), "rmsRadius"); // 0.9428090415820634
            expectClose(record.layoutRadius, Math.sqrt(10) / 3, "layoutRadius"); // 1.0540925533894598
            expect(record.meanDisplacement).toBe(0);
            expect(record.settledCount).toBe(0);
            expect(o.iteration).toBe(1);
        });
    }
});

describe("paper mode: force-form swing / traction, fresh sums over free nodes (SWING_MODE 0)", () => {
    it("per-node and global values at iteration 1, then the second iteration against F(t-1)", () => {
        const o = oracle("paper");
        const t1 = o.step();
        expectCloseArray(o.stages.swingPerNode, [2 * S74, 21, 2 * S74], "swing_i = m |F - 0|");
        expectCloseArray(o.stages.tractionPerNode, [S74, 10.5, S74], "traction_i = 0.5 m |F + 0|");
        expectClose(o.stages.partials.swing, 21 + 4 * S74, "partials.swing"); // 55.40930106817051
        expectClose(o.stages.partials.traction, 10.5 + 2 * S74, "partials.traction"); // 27.704650534085253
        expectClose(t1.swing, 55.40930106817051, "trace swing (fresh, not accumulated)");
        expectClose(t1.traction, 27.704650534085253, "trace traction");
        expectClose(o.swing, 55.40930106817051, "oracle.swing");
        expectClose(t1.speed, 0.14714154781913558, "speed = jitter x 0.5 (ratio exactly 2: no halving)");
        expectClose(t1.speedEfficiency, 0.7, "eff");
        expectCloseArray(
            o.stages.displacement,
            [
                -0.39751490326156713, -0.2839392166154051, 0, 0, 0.37347848728524713, 0, 0.39751490326156713,
                -0.2839392166154051, 0,
            ],
            "dp = F x speed / (1 + sqrt(speed m |F|))",
        );
        expectCloseArray(
            o.positions,
            [
                -1.397514903261567, -0.2839392166154051, 0, 0, 1.3734784872852472, 0, 1.397514903261567,
                -0.2839392166154051, 0,
            ],
            "positions",
        );
        const t2 = o.step();
        expectCloseArray(o.stages.oldForce, [-7, -5, 0, 0, 7, 0, 7, -5, 0], "oldForce = F(1)");
        expectCloseArray(
            o.stages.force,
            [
                -3.0728142387775375, -2.176001042949008, 0, 0, 2.148427657478288, 0, 3.0728142387775375,
                -2.176001042949008, 0,
            ],
            "F(2)",
        );
        expectCloseArray(
            o.stages.swingPerNode,
            [9.674245833670662, 14.554717027565134, 9.674245833670662],
            "swing_i = m |F(2) - F(1)|",
        );
        expectCloseArray(
            o.stages.tractionPerNode,
            [12.367561508127814, 13.722641486217434, 12.367561508127814],
            "traction_i = 0.5 m |F(2) + F(1)|",
        );
        expectClose(t2.swing, 33.90320869490646, "swing(2) fresh");
        expectClose(t2.traction, 38.45776450247306, "traction(2) fresh");
        expectClose(t2.speed, 0.22071232172870336, "speed(2) = speed(1) x 1.5 (the 0.5 x speed max rise binds)");
        expectClose(t2.speedEfficiency, 0.49, "eff(2) = 0.7 x 0.7");
        expectCloseArray(
            o.stages.displacement,
            [
                -0.2755552849114448, -0.19513336660271344, 0, 0, 0.16981747128215346, 0, 0.2755552849114448,
                -0.19513336660271344, 0,
            ],
            "dp(2) with the local factor from m |F(2) - F(1)|",
        );
        // K1 of iteration 2 folds iteration 1's integrate: centroid of the new positions, radii about the OLD centroid (0, 1/3)
        expectClose(t2.meanDisplacement, 0.4501644954144142, "meanDisplacement = mean |dp(1)| over free nodes");
        expect(t2.settledCount).toBe(0);
        expectCloseArray(t2.centroid, [0, 0.268533351351479, 0], "centroid(2)");
        expectClose(t2.rmsRadius, 1.3844431555450325, "rmsRadius about the previous centroid");
        expectClose(t2.layoutRadius, 1.5277674252838371, "layoutRadius = max |p - c_prev|");
        expect(o.iteration).toBe(2);
        o.reheat();
        expectClose(o.swing, 33.90320869490646, "reheat leaves the fresh sums alone in mode 0");
        expect(o.settledCount).toBe(0);
    });

    it("a fixed node keeps its force but leaves the sums, the mean displacement and its position untouched", () => {
        const o = oracle("paper", { fixed: fixedNode2() });
        const t1 = o.step();
        expectCloseArray(o.stages.swingPerNode, [2 * S74, 21, 0], "swing_i (node 2 excluded)");
        expectCloseArray(o.stages.tractionPerNode, [S74, 10.5, 0], "traction_i (node 2 excluded)");
        expectClose(t1.swing, 21 + 2 * S74, "swing = 38.20465053408525");
        expectClose(t1.traction, 10.5 + S74, "traction = 19.102325267042627");
        expectClose(t1.speed, 0.14714154781913558, "speed unchanged by the exclusion (ratio still exactly 2)");
        expectCloseArray(o.stages.force.subarray(6, 9), [7, -5, 0], "F[2] still computed");
        expectCloseArray(o.stages.displacement.subarray(6, 9), [0, 0, 0], "dp[2] = 0");
        expectCloseArray(o.positions.subarray(6, 9), [1, 0, 0], "p[2] unchanged");
        expect(o.stages.partials.free).toBe(2);
        const t2 = o.step();
        expectCloseArray(o.stages.oldForce.subarray(6, 9), [7, -5, 0], "oldForce[2] stored although fixed (7.11)");
        expectClose(t2.meanDisplacement, 0.43099299338212244, "mean over the two free nodes");
        expectCloseArray(t2.centroid, [-0.13250496775385567, 0.36317975688994736, 0], "centroid(2)");
        expectClose(t2.rmsRadius, 1.2284253278295167, "rmsRadius(2)");
        expectClose(t2.layoutRadius, 1.5277674252838371, "layoutRadius(2)");
        expectClose(t2.swing, 18.253592208707893, "swing(2)");
        expectClose(t2.traction, 29.375771992333277, "traction(2)");
        expectClose(t2.speed, 0.22071232172870336, "speed(2)");
        expectClose(t2.speedEfficiency, 0.49, "eff(2)");
        expectCloseArray(
            o.stages.displacement,
            [-0.32583076346496637, -0.23877018563842758, 0, -0.08834501981434673, 0.3501133708922873, 0, 0, 0, 0],
            "dp(2)",
        );
    });
});

describe("networkx mode: position-mixed swing / traction, accumulated from 1 over every node (SWING_MODE 1)", () => {
    it("per-node and global values at iteration 1, then the second iteration with the m |F| local factor", () => {
        const o = oracle("networkx");
        const t1 = o.step();
        expectCloseArray(o.stages.swingPerNode, [S61, 18, S61], "swing_i = m |p - F|");
        expectCloseArray(o.stages.tractionPerNode, [S89, 12, S89], "traction_i = 0.5 m |p + F|");
        expectClose(o.stages.partials.swing, 18 + 2 * S61, "partials.swing = the fresh sum 49.240998703626616");
        expectClose(o.stages.partials.traction, 12 + 2 * S89, "partials.traction = 30.867962264113203");
        expectClose(t1.swing, 19 + 2 * S61, "trace swing = 1 + fresh = 50.240998703626616");
        expectClose(t1.traction, 13 + 2 * S89, "trace traction = 1 + fresh = 31.867962264113206");
        expectClose(t1.speed, 0.19450843827825515, "speed = target (jitter = other = 0.3066496098618432)");
        expectClose(t1.speedEfficiency, 0.7, "eff");
        expectCloseArray(
            o.stages.displacement,
            [
                -0.4812301414914454, -0.34373581535103237, 0, 0, 0.45068941781693805, 0, 0.4812301414914454,
                -0.34373581535103237, 0,
            ],
            "dp = F x speed / (1 + sqrt(speed m |F|))",
        );
        expectCloseArray(
            o.positions,
            [
                -1.4812301414914453, -0.34373581535103237, 0, 0, 1.450689417816938, 0, 1.4812301414914453,
                -0.34373581535103237, 0,
            ],
            "positions",
        );
        const t2 = o.step();
        expectCloseArray(o.stages.oldForce, [0, 0, 0, 0, 0, 0, 0, 0, 0], "oldForce never written in mode 1");
        expectCloseArray(
            o.stages.force,
            [
                -2.5541064485318543, -1.7307632860901607, 0, 0, 1.3657429232480003, 0, 2.5541064485318543,
                -1.7307632860901607, 0,
            ],
            "F(2)",
        );
        expectCloseArray(
            o.stages.swingPerNode,
            [3.5070835603353676, 0.2548394837068133, 3.5070835603353676],
            "swing_i(2) = m |p - F|",
        );
        expectCloseArray(
            o.stages.tractionPerNode,
            [4.537343707133194, 4.224648511597407, 4.537343707133194],
            "traction_i(2)",
        );
        expectClose(t2.swing, 57.51000530800417, "swing(2) = swing(1) + fresh");
        expectClose(t2.traction, 45.167298189977, "traction(2) = traction(1) + fresh");
        expectClose(t2.speed, 0.23894116682215102, "speed(2)");
        expectClose(t2.speedEfficiency, 0.49, "eff(2)");
        expectCloseArray(
            o.stages.displacement,
            [
                -0.2756152212593928, -0.18676774662918832, 0, 0, 0.16403198092793195, 0, 0.2756152212593928,
                -0.18676774662918832, 0,
            ],
            "dp(2) with the local factor from m |F(2)| (not m |F(2) - F(1)|)",
        );
        expectClose(t2.meanDisplacement, 0.5444867778832041, "meanDisplacement(2)");
        expectCloseArray(t2.centroid, [0, 0.2544059290382911, 0], "centroid(2)");
        expectClose(t2.rmsRadius, 1.4779958669155573, "rmsRadius(2)");
        expectClose(t2.layoutRadius, 1.628639114157258, "layoutRadius(2)");
        o.reheat();
        expect(o.swing).toBe(1);
        expect(o.traction).toBe(1);
        expectClose(o.speed, 0.23894116682215102, "reheat keeps the speed controller (D8)");
        expect(o.iteration).toBe(2);
        expect(o.settledCount).toBe(0);
    });

    it("a fixed node still counts in the sums (every node) and is not moved", () => {
        const o = oracle("networkx", { fixed: fixedNode2() });
        const t1 = o.step();
        expectCloseArray(o.stages.swingPerNode, [S61, 18, S61], "swing_i includes node 2");
        expectClose(t1.swing, 19 + 2 * S61, "swing unchanged by the mask");
        expectClose(t1.traction, 13 + 2 * S89, "traction unchanged by the mask");
        expectCloseArray(o.stages.displacement.subarray(6, 9), [0, 0, 0], "dp[2] = 0");
        expectCloseArray(
            o.positions,
            [-1.4812301414914453, -0.34373581535103237, 0, 0, 1.450689417816938, 0, 1, 0, 0],
            "positions",
        );
        const t2 = o.step();
        expectCloseArray(
            o.stages.force,
            [
                -3.017142356219181, -2.1690161467684694, 0, -1.0634791984645702, 3.3396145477132526, 0,
                4.028851058006053, -3.718490225410944, 0,
            ],
            "F(2)",
        );
        expectCloseArray(
            o.stages.swingPerNode,
            [4.7710269836206995, 6.503171177762884, 9.591894137877278],
            "swing_i(2)",
        );
        expectClose(t2.swing, 71.10709100288747, "swing(2)");
        expectClose(t2.traction, 50.63528169718428, "traction(2)");
        expectClose(t2.speed, 0.2428732649702072, "speed(2)");
        expectClose(t2.meanDisplacement, 0.5210374378666376, "mean over the two free nodes");
        expectCloseArray(t2.centroid, [-0.1604100471638151, 0.3689845341553019, 0], "centroid(2)");
        expectClose(t2.rmsRadius, 1.2925505915058775, "rmsRadius(2)");
        expectClose(t2.layoutRadius, 1.628639114157258, "layoutRadius(2)");
        expectCloseArray(
            o.stages.displacement,
            [-0.3126882928631277, -0.22479083717331966, 0, -0.09941780374814171, 0.3121990013329409, 0, 0, 0, 0],
            "dp(2)",
        );
    });
});

describe("estimateFactor (spec 7.10 line for line; networkx layout.py 1387-1420)", () => {
    it("reproduces both iteration-1 controllers and the iteration-2 max-rise clamp", () => {
        const nx1: EstimateFactorResult = estimateFactor(3, 50.240998703626616, 31.867962264113203, 1, 1, 1, "f64");
        expectClose(nx1.jitter, 0.3066496098618432, "networkx jitter = other (> minJitter)");
        expectClose(nx1.targetSpeed, 0.19450843827825517, "networkx target");
        expectClose(nx1.speed, 0.19450843827825515, "networkx speed");
        expectClose(nx1.speedEfficiency, 0.7, "networkx eff");
        const paper1 = estimateFactor(3, 55.40930106817051, 27.704650534085253, 1, 1, 1, "f64");
        expectClose(paper1.jitter, 0.2942830956382712, "paper jitter = minJitter");
        expectClose(paper1.targetSpeed, 0.1471415478191356, "paper target = jitter / 2");
        expectClose(paper1.speed, 0.14714154781913558, "paper speed");
        expectClose(paper1.speedEfficiency, 0.7, "paper eff");
        const paper2 = estimateFactor(3, 33.90320869490646, 38.45776450247306, 0.14714154781913558, 0.7, 1, "f64");
        expectClose(paper2.jitter, 0.3700600114655676, "paper(2) jitter");
        expectClose(paper2.targetSpeed, 0.29384170184471037, "paper(2) target");
        expectClose(paper2.speed, 0.22071232172870336, "paper(2) speed = 1.5 x speed (max rise 0.5 x speed)");
        expectClose(paper2.speedEfficiency, 0.49, "paper(2) eff");
    });

    it("takes the swing / traction > 2 branch: eff halved, jitter raised to jitterTolerance, then x 0.7", () => {
        const r = estimateFactor(3, 10, 1, 1, 1, 1, "f64");
        expectClose(r.jitter, 1, "jitter = max(0.294..., 1)");
        expectClose(r.targetSpeed, 0.05, "target = 1 x 0.5 x 1 / 10");
        expectClose(r.speedEfficiency, 0.35, "eff = 0.5 x 0.7");
        expectClose(r.speed, 0.05, "speed = 1 + (0.05 - 1)");
    });

    it("takes the 1.3 rise when the layout is calm and speed < 1000, and never raises above 1000", () => {
        const calm = estimateFactor(3, 0.1, 1, 1, 1, 1, "f64");
        expectClose(calm.jitter, 0.2942830956382712, "jitter");
        expectClose(calm.targetSpeed, 2.942830956382712, "target = jitter x 1 x 1 / 0.1");
        expectClose(calm.speedEfficiency, 1.3, "eff x 1.3");
        expectClose(calm.speed, 1.5, "speed = 1 + min(1.94..., 0.5)");
        const fast = estimateFactor(3, 0.1, 1, 2000, 1, 1, "f64");
        expectClose(fast.speedEfficiency, 1, "eff untouched at speed >= 1000");
        const fastHot = estimateFactor(3, 10, 1, 2000, 1, 1, "f64");
        expectClose(fastHot.speedEfficiency, 0.35, "the halving and the 0.7 still apply at speed >= 1000");
    });

    it("treats swing 0 as an infinite target (1e30: the same min() outcome as NetworkX's inf)", () => {
        const r = estimateFactor(3, 0, 1, 2, 1, 1, "f64");
        expect(r.targetSpeed).toBe(1e30);
        expectClose(r.speed, 3, "speed x 1.5");
        expectClose(r.speedEfficiency, 1.3, "eff x 1.3");
    });

    it("skips the conditional multiplies at or below the 0.05 floor (never raises a sub-floor value)", () => {
        const r = estimateFactor(3, 10, 1, 1, 0.05, 1, "f64");
        expect(r.speedEfficiency).toBe(0.05);
        expectClose(r.targetSpeed, 0.005, "target = 1 x 0.05 x 1 / 10");
        expectClose(r.speed, 0.005, "speed");
        const below = estimateFactor(3, 10, 1, 1, 0.02, 1, "f64");
        expect(below.speedEfficiency).toBe(0.02);
    });

    it("f32 precision rounds every operation (values are f32-representable and within 1e-6 of f64)", () => {
        const a = estimateFactor(3, 50.240998703626616, 31.867962264113203, 1, 1, 1, "f32");
        const b = estimateFactor(3, 50.240998703626616, 31.867962264113203, 1, 1, 1, "f64");
        for (const key of ["speed", "speedEfficiency", "jitter", "targetSpeed"] as const) {
            expect(Math.fround(a[key]), key).toBe(a[key]);
            expectClose(a[key], b[key], key, 1e-6);
        }
    });
});

describe("gravity centre (the GRAVITY_CENTER override)", () => {
    it("paper mode defaults to the centroid (0, 1/3): -m q / |q| with q = p - c", () => {
        const o = new ForceAtlas2Oracle(path3, START, { compat: "paper", precision: "f64", dim: 2 });
        o.step();
        // q0 = (-1, -1/3), |q0| = sqrt(10) / 3 -> G0 = 2 (1, 1/3) x 3 / sqrt(10) = (6, 2) / sqrt(10); q1 = (0, 2/3) -> G1 = (0, -3)
        expectCloseArray(
            o.stages.gravity,
            [6 / Math.sqrt(10), 2 / Math.sqrt(10), 0, 0, -3, 0, -6 / Math.sqrt(10), 2 / Math.sqrt(10), 0],
            "centroid gravity",
        );
        expectCloseArray(
            o.stages.force,
            [-7.102633403898972, -4.367544467966324, 0, 0, 7, 0, 7.102633403898972, -4.367544467966324, 0],
            "force",
        );
    });

    it("networkx mode defaults to the origin and the paper mode can be told to use it (identical forces)", () => {
        const a = oracle("paper");
        const b = new ForceAtlas2Oracle(path3, START, { compat: "networkx", precision: "f64", dim: 2 });
        a.step();
        b.step();
        for (let i = 0; i < 9; i++) {
            expect(Object.is(a.stages.force[i], b.stages.force[i]), `force[${i}]`).toBe(true);
        }
    });

    it("strong gravity is -g m q (no guard): the same as regular gravity at |p| = 1", () => {
        const o = oracle("paper", { strongGravity: true });
        o.step();
        expectCloseArray(o.stages.gravity, [2, 0, 0, 0, -3, 0, -2, 0, 0], "strong gravity");
    });
});

describe("kickDir (the prelude's coincident kick, re-implemented)", () => {
    it("is antisymmetric, unit length and planar in 2D", () => {
        for (const [i, j] of [
            [0, 1],
            [3, 7],
            [1000, 5],
            [65535, 65536],
        ] as const) {
            for (const dim of [2, 3] as const) {
                const a = kickDir(i, j, dim);
                const b = kickDir(j, i, dim);
                expectCloseArray(
                    b,
                    [-a[0], -a[1], -a[2]],
                    `kickDir(${j}, ${i}, ${dim}) = -kickDir(${i}, ${j}, ${dim})`,
                );
                expectClose(Math.hypot(a[0], a[1], a[2]), 1, `|kickDir(${i}, ${j}, ${dim})|`);
                if (dim === 2) {
                    expect(a[2] === 0, "planar (0 or -0)").toBe(true);
                }
                const f = kickDir(i, j, dim, "f32");
                expectClose(Math.hypot(f[0], f[1], f[2]), 1, `|kickDir f32|`, 1e-6);
                for (const v of f) {
                    expect(Math.fround(v)).toBe(v);
                }
            }
        }
    });

    it("two coincident isolated nodes receive opposite kicks of magnitude k m_i m_j / 0.01", () => {
        const two = snapshotOf([], { nodeCount: 2 });
        const o = new ForceAtlas2Oracle(two, [0.5, 0.5, 0, 0.5, 0.5, 0], {
            compat: "paper",
            precision: "f64",
            dim: 2,
            gravity: 0,
        });
        o.step();
        const dir = kickDir(0, 1, 2);
        expectCloseArray(
            o.stages.repulsion,
            [200 * dir[0], 200 * dir[1], 0, -200 * dir[0], -200 * dir[1], 0],
            "kick = 2 x 1 x 1 / 0.01 along kickDir",
        );
        expectCloseArray(o.stages.attraction, [0, 0, 0, 0, 0, 0], "no arcs: attraction zero");
        expectClose(o.stages.force[0] + o.stages.force[3], 0, "sum F_x");
        expectClose(o.stages.force[1] + o.stages.force[4], 0, "sum F_y");
    });
});

describe("the empty graph and setPosition", () => {
    it("steps an empty graph without work and reports the state unchanged", () => {
        const empty = snapshotOf([], { nodeCount: 0 });
        const o = new ForceAtlas2Oracle(empty, [], { compat: "networkx", precision: "f64", dim: 2 });
        const record = o.step();
        expect(record.swing).toBe(1);
        expect(record.traction).toBe(1);
        expect(record.speed).toBe(1);
        expect(o.iteration).toBe(1);
        expect(o.positions.length).toBe(0);
    });

    it("setPosition writes layout units, forces z = 0 in 2D and leaves the pending statistics alone", () => {
        const o = oracle("paper");
        o.step();
        o.setPosition(1, 0.25, 0.75, 9);
        expectCloseArray(o.positions.subarray(3, 6), [0.25, 0.75, 0], "p[1]");
        const t2 = o.step();
        expectCloseArray(t2.centroid, [0, 0.268533351351479, 0], "K1 folded the previous integrate, not the override");
        expect(() => {
            o.setPosition(3, 0, 0, 0);
        }).toThrow(/out of range/);
        expect(() => {
            o.setFixed(makeMask(3));
        }).not.toThrow();
        expect(() => {
            o.setFixed(null);
        }).not.toThrow();
        expect(() => {
            o.setFixed(new Uint32Array(0));
        }).toThrow(/mask\.length/);
        const short = new ForceAtlas2Oracle(snapshotOf([[0, 1]], { nodeCount: 40 }), new Float64Array(120), {
            compat: "paper",
            precision: "f64",
        });
        expect(() => {
            short.setFixed(new Uint32Array(1));
        }).toThrow(/mask\.length/);
    });
});
