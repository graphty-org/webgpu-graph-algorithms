/**
 * The walking skeleton's FA2 iteration (spec 11.5; contract 5.5 P1-T6): ONE exact-tile K3 + K4 iteration on karate
 * from host-written state (positions with the mass in `.w`, zero `force` / `oldForce`, a clear fixed mask, `speed =
 * speedEfficiency = swing = traction = 1`, the centroid / bbox / radii computed on the CPU as load() will) in four
 * legs -- paper, networkx, strong gravity, and paper with a non-zero `oldForce` -- each run twice and compared bitwise
 * before it is compared to the inline f64 reference below (the 7.2 table on the f32 inputs: `k m_i m_j / d` with the
 * `max(d, 0.01)` floor, gravity by GRAVITY_CENTER / STRONG_GRAVITY, the two swing / traction forms, estimateFactor
 * line for line). The force is held to the 11.4 floored per-node error, the trace slot / state header / partials to
 * a relative error; both tolerances come from benchmarks/results/noise-floor.json (spec 11.9 item 3), never from a
 * literal. The workgroup twin (acquire({ subgroups: false })) is compared within the twin floors and every committed
 * adapter fixture within the cross-adapter floors; under GRAPHTY_NOISE_FLOOR_WRITE=1 the run writes this adapter's
 * raw outputs, the twin's and the reference's as noise fixtures (contract 5.6).
 *
 * The positions recipe, the state initialisation and the params below are duplicated by test/browser/skeleton.test.ts
 * (P1-T7) so the SwiftShader / NVIDIA-Chromium fixtures compare against the same inputs; a drift between the two
 * copies shows up as a cross-adapter disagreement in test/noise-floor.test.ts.
 */

import { type F32, type GraphSnapshot } from "@graphty/graph-format";

import {
    FA2_COINCIDENT_SQ,
    FA2_DISTANCE_FLOOR,
    FA2_DISTANCE_FLOOR_SQ,
    PARTIAL_BYTES,
    STATE_HEADER_BYTES,
    TRACE_RECORD_BYTES,
} from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode } from "../../src/errors.js";
import { type UniformValues } from "../../src/kernel/struct-block.js";
import { FA2_PARAMS, FA2_PARTIAL, FA2_STATE, FA2_TRACE } from "../../src/kernels.js";
import {
    RepulsionExact,
    type RepulsionExactOverrides,
    type RepulsionExactResources,
} from "../../src/layouts/repulsion-exact.js";
import { bindingOf, readF32, scratchBuffer, uploadBuffer } from "../helpers/device.js";
import { KARATE_EDGES, snapshotOf } from "../helpers/graphs.js";
import { expectBitwiseEqual, flooredRelError, maxRelError } from "../helpers/matchers.js";
import { adapterClass, noiseFloorFor, readNoiseFixtures, writeNoiseFixture } from "../helpers/noise-floor.js";
import { acquire, requireGpu } from "../setup/gpu.js";

// ---------------------------------------------------------------- the fixture (duplicated by test/browser/skeleton.test.ts)

/** The LCG seed of the skeleton positions: 12345 keeps every pair > 0.02 apart and every node > 0.01 from the centroid and the origin (pinned below). */
const SKELETON_SEED = 12345;
/** The CPU port's LCG (spec 7.2; layout/src/utils/random.ts): m = 2^35 - 31, a = 185852, c = 1. */
const LCG_M = 34359738337;
const LCG_A = 185852;
const LCG_C = 1;
/** The params every leg shares (FA2_DEFAULTS: scalingRatio 2, gravity 1; a 2D layout). */
const SCALING_RATIO = 2;
const GRAVITY = 1;
const DIM = 2;
/** The host-written controller state (spec 7.17: load() writes speed = speedEfficiency = 1; contract 3.13 onLoad: swing = traction = 1). */
const INITIAL_SPEED = 1;
const INITIAL_EFFICIENCY = 1;
const INITIAL_SWING = 1;
const INITIAL_TRACTION = 1;
/** The trace slot K4 writes (P.iterationIndex) and the records the trace binding covers; slot 0 must stay zero. */
const TRACE_SLOT = 1;
const TRACE_RECORDS = 4;
const STATE_BYTES = STATE_HEADER_BYTES + TRACE_RECORDS * TRACE_RECORD_BYTES;
/** Spec 11.4: the floored denominator max(|F_cpu(i)|, 1e-3 x max_j |F_cpu(j)|) of the per-node force error. */
const FLOOR_FRACTION = 1e-3;
/** Spec 11.5: the 1e-6 absolute floor of the relative trace comparison. */
const TRACE_ABS_FLOOR = 1e-6;
/** The pinned f64 numbers are reproduced to this relative precision (a fixed evaluation order; ulp-level slack only). */
const PIN_TOLERANCE = 1e-12;
/** The reserved adapter-class strings of the noise fixtures (contract 5.6 as used by this task). */
const ORACLE_CLASS = "oracle-f64";
const TWIN_SUFFIX = "-no-subgroups";

/**
 * The skeleton's device positions: `vec4f` rows (x, y, z = 0, mass = outDegree + 1; spec 7.3, D23), x and y drawn from
 * the port's LCG in index order and mapped to [-1, 1), stored as f32 (the reference reads the f32 values back).
 */
function skeletonPositions(outDegree: ArrayLike<number>): Float32Array<ArrayBuffer> {
    const n = outDegree.length;
    const positions = new Float32Array(4 * n);
    let state = SKELETON_SEED % LCG_M;
    const next = (): number => {
        state = (LCG_A * state + LCG_C) % LCG_M;
        return state / LCG_M;
    };
    for (let i = 0; i < n; i++) {
        positions[4 * i] = 2 * next() - 1;
        positions[4 * i + 1] = 2 * next() - 1;
        positions[4 * i + 2] = 0;
        positions[4 * i + 3] = outDegree[i] + 1;
    }
    return positions;
}

interface CpuStats {
    readonly centroid: readonly [number, number, number];
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
    readonly rmsRadius: number;
    readonly radius: number;
}

/** What load() writes into the state header (contract 3.13): the f32-rounded centroid (what the kernel reads), bbox, RMS radius and max |p - c|, in f64. */
function cpuStats(positions: Float32Array): CpuStats {
    const n = positions.length / 4;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        const x = positions[4 * i];
        const y = positions[4 * i + 1];
        const z = positions[4 * i + 2];
        sx += x;
        sy += y;
        sz += z;
        min[0] = Math.min(min[0], x);
        min[1] = Math.min(min[1], y);
        min[2] = Math.min(min[2], z);
        max[0] = Math.max(max[0], x);
        max[1] = Math.max(max[1], y);
        max[2] = Math.max(max[2], z);
    }
    const centroid: [number, number, number] = [Math.fround(sx / n), Math.fround(sy / n), Math.fround(sz / n)];
    let sumSq = 0;
    let maxSq = 0;
    for (let i = 0; i < n; i++) {
        const qx = positions[4 * i] - centroid[0];
        const qy = positions[4 * i + 1] - centroid[1];
        const qz = positions[4 * i + 2] - centroid[2];
        const q2 = qx * qx + qy * qy + qz * qz;
        sumSq += q2;
        maxSq = Math.max(maxSq, q2);
    }
    return { centroid, min, max, rmsRadius: Math.sqrt(sumSq / n), radius: Math.sqrt(maxSq) };
}

/** The smallest pair distance, the smallest |p - centroid| and the smallest |p| (the floor, kick and gravity-guard regions). */
function guardDistances(
    positions: Float32Array,
    centroid: readonly [number, number, number],
): { readonly pair: number; readonly centroid: number; readonly origin: number } {
    const n = positions.length / 4;
    let pair = Infinity;
    let toCentroid = Infinity;
    let toOrigin = Infinity;
    for (let i = 0; i < n; i++) {
        const px = positions[4 * i];
        const py = positions[4 * i + 1];
        const pz = positions[4 * i + 2];
        const qx = px - centroid[0];
        const qy = py - centroid[1];
        const qz = pz - centroid[2];
        toCentroid = Math.min(toCentroid, Math.sqrt(qx * qx + qy * qy + qz * qz));
        toOrigin = Math.min(toOrigin, Math.sqrt(px * px + py * py + pz * pz));
        for (let j = i + 1; j < n; j++) {
            const dx = px - positions[4 * j];
            const dy = py - positions[4 * j + 1];
            const dz = pz - positions[4 * j + 2];
            pair = Math.min(pair, Math.sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    return { pair, centroid: toCentroid, origin: toOrigin };
}

/**
 * The f64 reference of K3's force on the f32 inputs (spec 7.2, 7.6, 7.9): repulsion `k m_i m_j / d` along
 * `(p_i - p_j) / d` with `d^2 >= 1e-4`, then gravity (regular: `-g m_i q / |q|` when |q| > 0.01; strong: `-g m_i q`)
 * about the f32 centroid (GRAVITY_CENTER 0) or the origin (1). `force` was zero, so the result IS the force after K3.
 * The coincident kick never fires on this fixture (asserted); it is not implemented here.
 */
function referenceForces(
    positions: Float32Array,
    centroid: readonly [number, number, number],
    overrides: RepulsionExactOverrides,
): Float64Array {
    const n = positions.length / 4;
    const force = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        const px = positions[4 * i];
        const py = positions[4 * i + 1];
        const pz = positions[4 * i + 2];
        const mi = positions[4 * i + 3];
        let fx = 0;
        let fy = 0;
        let fz = 0;
        for (let j = 0; j < n; j++) {
            if (j === i) {
                continue;
            }
            const dx = px - positions[4 * j];
            const dy = py - positions[4 * j + 1];
            const dz = pz - positions[4 * j + 2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < FA2_COINCIDENT_SQ) {
                throw new Error(`coincident pair ${i}, ${j}: the skeleton fixture must not reach the kick`);
            }
            d2 = Math.max(d2, FA2_DISTANCE_FLOOR_SQ);
            const s = (SCALING_RATIO * mi * positions[4 * j + 3]) / d2;
            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }
        let qx = px;
        let qy = py;
        let qz = pz;
        if (overrides.GRAVITY_CENTER === 0) {
            qx -= centroid[0];
            qy -= centroid[1];
            qz -= centroid[2];
        }
        if (overrides.STRONG_GRAVITY) {
            fx += -GRAVITY * mi * qx;
            fy += -GRAVITY * mi * qy;
            fz += -GRAVITY * mi * qz;
        } else {
            const d = Math.sqrt(qx * qx + qy * qy + qz * qz);
            if (d > FA2_DISTANCE_FLOOR) {
                fx += (-GRAVITY * mi * qx) / d;
                fy += (-GRAVITY * mi * qy) / d;
                fz += (-GRAVITY * mi * qz) / d;
            }
        }
        force[3 * i] = fx;
        force[3 * i + 1] = fy;
        force[3 * i + 2] = fz;
    }
    return force;
}

/**
 * The f64 sums of the K3 epilogue over every node (the fixed mask is clear): mode 0 `swing_i = m |F - Fold|`,
 * `traction_i = 0.5 m |F + Fold|` (spec 7.2 paper); mode 1 `m |p - F|`, `0.5 m |p + F|` (NetworkX). Sequential in
 * index order.
 */
function referenceSums(
    positions: Float32Array,
    force: Float64Array,
    oldForce: Float32Array | null,
    swingMode: 0 | 1,
): readonly [number, number] {
    const n = positions.length / 4;
    let swing = 0;
    let traction = 0;
    for (let i = 0; i < n; i++) {
        const m = positions[4 * i + 3];
        const fx = force[3 * i];
        const fy = force[3 * i + 1];
        const fz = force[3 * i + 2];
        let ax: number;
        let ay: number;
        let az: number;
        let bx: number;
        let by: number;
        let bz: number;
        if (swingMode === 1) {
            const px = positions[4 * i];
            const py = positions[4 * i + 1];
            const pz = positions[4 * i + 2];
            ax = px - fx;
            ay = py - fy;
            az = pz - fz;
            bx = px + fx;
            by = py + fy;
            bz = pz + fz;
        } else {
            const ox = oldForce === null ? 0 : oldForce[3 * i];
            const oy = oldForce === null ? 0 : oldForce[3 * i + 1];
            const oz = oldForce === null ? 0 : oldForce[3 * i + 2];
            ax = fx - ox;
            ay = fy - oy;
            az = fz - oz;
            bx = fx + ox;
            by = fy + oy;
            bz = fz + oz;
        }
        swing += m * Math.sqrt(ax * ax + ay * ay + az * az);
        traction += 0.5 * m * Math.sqrt(bx * bx + by * by + bz * bz);
    }
    return [swing, traction];
}

/** estimateFactor in TypeScript, line for line the K4 body of contract 4.5 (spec 7.10): returns the new speed and speedEfficiency. */
function estimateFactor(
    swing: number,
    traction: number,
    speed: number,
    speedEfficiency: number,
    n: number,
    jitterTolerance: number,
): { readonly speed: number; readonly speedEfficiency: number } {
    const optJitter = 0.05 * Math.sqrt(n);
    const minJitter = Math.sqrt(optJitter);
    const maxJitter = 10;
    const tr = Math.max(traction, 1e-30);
    const other = Math.min(maxJitter, (optJitter * traction) / (n * n));
    let jitter = jitterTolerance * Math.max(minJitter, other);
    let eff = speedEfficiency;
    if (swing / tr > 2) {
        if (eff > 0.05) {
            eff = eff * 0.5;
        }
        jitter = Math.max(jitter, jitterTolerance);
    }
    const targetSpeed = swing === 0 ? 1e30 : (jitter * eff * traction) / swing;
    if (swing > jitter * traction) {
        if (eff > 0.05) {
            eff = eff * 0.7;
        }
    } else if (speed < 1000) {
        eff = eff * 1.3;
    }
    return { speed: speed + Math.min(targetSpeed - speed, 0.5 * speed), speedEfficiency: eff };
}

/** `fround(-0.5 x fround(f))` per component: the host-written oldForce of the paper-prev leg (exactly minus half the f32-rounded reference force). */
function halfNegated(force: Float64Array): Float32Array<ArrayBuffer> {
    const out = new Float32Array(force.length);
    for (let i = 0; i < force.length; i++) {
        out[i] = Math.fround(-0.5 * Math.fround(force[i]));
    }
    return out;
}

// ---------------------------------------------------------------- the legs and their expected values

interface Leg {
    readonly name: string;
    /** The noise-fixture name of this leg's trace (null: not part of the noise set). */
    readonly fixture: string | null;
    readonly overrides: RepulsionExactOverrides;
    readonly jitterTolerance: number;
    /** oldForce = -0.5 x the paper force (rounded to f32) instead of zero. */
    readonly prev: boolean;
}

const PAPER: Leg = {
    name: "paper",
    fixture: "karate",
    overrides: { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 },
    jitterTolerance: 1,
    prev: false,
};
const NETWORKX: Leg = {
    name: "networkx",
    fixture: "networkx-karate",
    overrides: { SWING_MODE: 1, STRONG_GRAVITY: false, GRAVITY_CENTER: 1 },
    jitterTolerance: 1,
    prev: false,
};
const STRONG: Leg = {
    name: "strong",
    fixture: null,
    overrides: { SWING_MODE: 0, STRONG_GRAVITY: true, GRAVITY_CENTER: 0 },
    jitterTolerance: 1,
    prev: false,
};
const PAPER_PREV: Leg = {
    name: "paper-prev",
    fixture: "prev-karate",
    overrides: { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 },
    jitterTolerance: 0.5,
    prev: true,
};
const LEGS: readonly Leg[] = [PAPER, NETWORKX, STRONG, PAPER_PREV];

interface Expected {
    readonly force: Float64Array;
    readonly oldForce: Float32Array<ArrayBuffer> | null;
    /** partials[0].swingTraction: the sums without the host-written 1. */
    readonly partial: readonly [number, number];
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
}

/** The numbers the reference produced when this test was written (node 22, f64); a drift means the reference changed. */
const PINS = {
    centroid: [-0.07239578664302826, -0.01930960826575756, 0],
    min: [-0.9808534383773804, -0.9781427383422852, 0],
    max: [0.9951686859130859, 0.9690985083580017, 0],
    rmsRadius: 0.8935186175537634,
    radius: 1.389419680706387,
    position0: [-0.8664516806602478, -0.7746485471725464],
    mass0: 17,
    position33: [0.3723199963569641, -0.5822708010673523],
    mass33: 18,
    guards: { pair: 0.027098824484323374, centroid: 0.04381973544344613, origin: 0.11567405500205069 },
    paper: {
        force0: [-3376.030304066258, -1777.8039444967717, 0],
        force33: [2559.865638550776, -3376.268415820058, 0],
        maxForce: 4236.991916808086,
        partial: [366257.0467533182, 183128.5233766591],
        speed: 1.5,
        speedEfficiency: 1.3,
    },
    networkx: {
        force0: [-3375.674210037426, -1778.1900644519676, 0],
        partial: [366123.19218980573, 183196.28278813255],
        swing: 366124.19218980573,
        traction: 183197.28278813255,
        speed: 1.5,
        speedEfficiency: 1.3,
    },
    strong: {
        force0: [-3374.848705520556, -1776.6799588951926, 0],
        partial: [366396.26321916445, 183198.13160958223],
        speed: 1.5,
        speedEfficiency: 1.3,
    },
    paperPrev: {
        partial: [549385.5708797185, 91564.26131345902],
        speed: 0.41666666439218303,
        speedEfficiency: 0.35,
    },
} as const;

const snapshot: GraphSnapshot = snapshotOf(KARATE_EDGES);
const n = snapshot.nodeCount;
const positions = skeletonPositions(snapshot.outDegree());
const stats = cpuStats(positions);
const paperForce = referenceForces(positions, stats.centroid, PAPER.overrides);
const expectedCache = new Map<string, Expected>();

function expectedFor(leg: Leg): Expected {
    const cached = expectedCache.get(leg.name);
    if (cached !== undefined) {
        return cached;
    }
    const force = referenceForces(positions, stats.centroid, leg.overrides);
    const oldForce = leg.prev ? halfNegated(paperForce) : null;
    const partial = referenceSums(positions, force, oldForce, leg.overrides.SWING_MODE);
    const accumulate = leg.overrides.SWING_MODE === 1;
    const swing = accumulate ? INITIAL_SWING + partial[0] : partial[0];
    const traction = accumulate ? INITIAL_TRACTION + partial[1] : partial[1];
    const controller = estimateFactor(swing, traction, INITIAL_SPEED, INITIAL_EFFICIENCY, n, leg.jitterTolerance);
    const expected: Expected = {
        force,
        oldForce,
        partial,
        swing,
        traction,
        speed: controller.speed,
        speedEfficiency: controller.speedEfficiency,
    };
    expectedCache.set(leg.name, expected);
    return expected;
}

// ---------------------------------------------------------------- the P1 driver (a pass the test opens; no ForceSimulation yet)

interface SkeletonOutput {
    readonly force: F32;
    /** The whole state buffer (header + trace records) and the partials, as words, for the bitwise run-to-run check. */
    readonly stateWords: Uint32Array<ArrayBuffer>;
    readonly partialWords: Uint32Array<ArrayBuffer>;
    readonly header: UniformValues;
    readonly trace: UniformValues;
    /** Trace slot 0, which nothing writes. */
    readonly traceUntouched: UniformValues;
    readonly partial: UniformValues;
}

function scalar(values: UniformValues, field: string): number {
    const value = values[field];
    if (typeof value !== "number") {
        throw new Error(`${field}: expected a scalar field`);
    }
    return value;
}

function vector(values: UniformValues, field: string): readonly number[] {
    const value = values[field];
    if (typeof value === "number") {
        throw new Error(`${field}: expected a vector field`);
    }
    return value;
}

/** (swing, traction, speed, speedEfficiency) of a state header or a trace record. */
function controllerOf(values: UniformValues): readonly [number, number, number, number] {
    return [
        scalar(values, "swing"),
        scalar(values, "traction"),
        scalar(values, "speed"),
        scalar(values, "speedEfficiency"),
    ];
}

/**
 * Uploads the host-written buffers of spec 7.3 for karate, records K3 then K4 through RepulsionExact into one pass,
 * submits, and reads force, partials and state back; every buffer is destroyed afterwards.
 */
async function runSkeleton(
    ctx: GpuContext,
    overrides: RepulsionExactOverrides,
    oldForce: Float32Array | null,
    jitterTolerance: number,
): Promise<SkeletonOutput> {
    const stage = await RepulsionExact.create(ctx.pipelines, ctx.caps, overrides);
    const groups = Math.ceil(n / ctx.workgroupSize);
    const stateBytes = new Uint8Array(STATE_BYTES);
    FA2_STATE.write(
        new DataView(stateBytes.buffer),
        {
            speed: INITIAL_SPEED,
            speedEfficiency: INITIAL_EFFICIENCY,
            swing: INITIAL_SWING,
            traction: INITIAL_TRACTION,
            centroid: [stats.centroid[0], stats.centroid[1], stats.centroid[2], 0],
            rmsRadius: stats.rmsRadius,
            radius: stats.radius,
            min: [stats.min[0], stats.min[1], stats.min[2], 0],
            max: [stats.max[0], stats.max[1], stats.max[2], 0],
        },
        0,
    );
    const paramsBytes = new Uint8Array(FA2_PARAMS.byteLength);
    FA2_PARAMS.write(
        new DataView(paramsBytes.buffer),
        {
            n,
            dim: DIM,
            flags: 0,
            tierStart: 0,
            tierEnd: n,
            iterationIndex: TRACE_SLOT,
            seed: 0,
            nearMax: 0,
            scalingRatio: SCALING_RATIO,
            gravity: GRAVITY,
            jitterTolerance,
            scale: 1,
            center: [0, 0, 0, 0],
            settleThreshold: 0.001,
            extentFactor: 6,
        },
        0,
    );
    const pos = uploadBuffer(ctx, positions, "skeleton/pos");
    const force = scratchBuffer(ctx, 12 * n, "skeleton/force");
    const old =
        oldForce === null
            ? scratchBuffer(ctx, 12 * n, "skeleton/oldForce")
            : uploadBuffer(ctx, oldForce, "skeleton/oldForce");
    const fixed = scratchBuffer(ctx, 4 * Math.ceil(n / 32), "skeleton/fixed");
    const partials = scratchBuffer(ctx, PARTIAL_BYTES * groups, "skeleton/partials");
    const state = uploadBuffer(ctx, stateBytes, "skeleton/state");
    const params = uploadBuffer(ctx, paramsBytes, "skeleton/params", BufferUsage.UNIFORM);
    const buffers = [pos, force, old, fixed, partials, state, params];
    try {
        const resources: RepulsionExactResources = {
            pos: bindingOf(pos),
            state: { buffer: state, offset: 0, size: STATE_HEADER_BYTES, window: null },
            trace: {
                buffer: state,
                offset: STATE_HEADER_BYTES,
                size: TRACE_RECORDS * TRACE_RECORD_BYTES,
                window: null,
            },
            force: bindingOf(force),
            oldForce: bindingOf(old),
            fixedMask: bindingOf(fixed),
            partials: bindingOf(partials),
            params: { buffer: params, offset: 0, size: FA2_PARAMS.byteLength, window: null },
        };
        stage.bind(resources);
        const encoder = ctx.device.createCommandEncoder({ label: "skeleton" });
        const pass = encoder.beginComputePass({ label: "skeleton/K3+K4" });
        stage.record(pass, n, 0);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        const forceOut = await readF32(ctx, force, 3 * n);
        const stateWords = new Uint32Array(await ctx.readback.read(state, STATE_BYTES));
        const partialWords = new Uint32Array(await ctx.readback.read(partials, PARTIAL_BYTES * groups));
        ctx.assertReady();
        const stateView = new DataView(stateWords.buffer);
        return {
            force: forceOut,
            stateWords,
            partialWords,
            header: FA2_STATE.read(stateView, 0),
            trace: FA2_TRACE.read(stateView, STATE_HEADER_BYTES + TRACE_SLOT * TRACE_RECORD_BYTES),
            traceUntouched: FA2_TRACE.read(stateView, STATE_HEADER_BYTES),
            partial: FA2_PARTIAL.read(new DataView(partialWords.buffer), 0),
        };
    } finally {
        for (const buffer of buffers) {
            buffer.destroy();
        }
    }
}

let mainContext: GpuContext | null = null;

async function main(): Promise<GpuContext> {
    if (mainContext === null) {
        mainContext = await acquire({ label: "skeleton" });
    }
    return mainContext;
}

const legRuns = new Map<string, readonly [SkeletonOutput, SkeletonOutput]>();

/** Two runs of a leg on the main context (cached per file, so every `it` sees the same outputs). */
async function runsOf(leg: Leg): Promise<readonly [SkeletonOutput, SkeletonOutput]> {
    const cached = legRuns.get(leg.name);
    if (cached !== undefined) {
        return cached;
    }
    const ctx = await main();
    const { oldForce } = expectedFor(leg);
    const first = await runSkeleton(ctx, leg.overrides, oldForce, leg.jitterTolerance);
    const second = await runSkeleton(ctx, leg.overrides, oldForce, leg.jitterTolerance);
    const runs: readonly [SkeletonOutput, SkeletonOutput] = [first, second];
    legRuns.set(leg.name, runs);
    return runs;
}

function forceTolerance(): number {
    return noiseFloorFor("fa2-skeleton.force").value;
}

function traceTolerance(): number {
    return noiseFloorFor("fa2-skeleton.trace").value;
}

function expectPinned(actual: number, pinned: number, label: string): void {
    expect(Math.abs(actual - pinned), `${label}: ${actual} vs pinned ${pinned}`).toBeLessThanOrEqual(
        PIN_TOLERANCE * Math.abs(pinned),
    );
}

function isAdapterFixture(cls: string): boolean {
    return cls !== ORACLE_CLASS && !cls.endsWith(TWIN_SUFFIX);
}

// ---------------------------------------------------------------- the tests

describe("FA2 skeleton iteration (K3 + K4) on karate", () => {
    it("the fixture avoids the distance floor, the coincident kick and the gravity guard", () => {
        expect(n).toBe(34);
        expect(positions.length).toBe(4 * n);
        expectPinned(positions[0], PINS.position0[0], "positions[0].x");
        expectPinned(positions[1], PINS.position0[1], "positions[0].y");
        expect(positions[3]).toBe(PINS.mass0);
        expectPinned(positions[132], PINS.position33[0], "positions[33].x");
        expectPinned(positions[133], PINS.position33[1], "positions[33].y");
        expect(positions[135]).toBe(PINS.mass33);
        for (let i = 0; i < n; i++) {
            expect(positions[4 * i + 2]).toBe(0);
            expect(positions[4 * i + 3]).toBe(snapshot.outDegree()[i] + 1);
        }
        const guards = guardDistances(positions, stats.centroid);
        expect(guards.pair).toBeGreaterThan(2 * FA2_DISTANCE_FLOOR);
        expect(guards.centroid).toBeGreaterThan(FA2_DISTANCE_FLOOR);
        expect(guards.origin).toBeGreaterThan(FA2_DISTANCE_FLOOR);
        expectPinned(guards.pair, PINS.guards.pair, "min pair distance");
        expectPinned(guards.centroid, PINS.guards.centroid, "min distance to the centroid");
        expectPinned(guards.origin, PINS.guards.origin, "min distance to the origin");
        for (let k = 0; k < 3; k++) {
            expectPinned(stats.centroid[k], PINS.centroid[k], `centroid[${k}]`);
            expectPinned(stats.min[k], PINS.min[k], `min[${k}]`);
            expectPinned(stats.max[k], PINS.max[k], `max[${k}]`);
        }
        expectPinned(stats.rmsRadius, PINS.rmsRadius, "rmsRadius");
        expectPinned(stats.radius, PINS.radius, "radius");
    });

    it("the inline f64 reference reproduces its pinned values (and writes the oracle fixtures under GRAPHTY_NOISE_FLOOR_WRITE=1)", () => {
        const paper = expectedFor(PAPER);
        for (let k = 0; k < 3; k++) {
            expectPinned(paper.force[k], PINS.paper.force0[k], `paper force[0].${k}`);
            expectPinned(paper.force[99 + k], PINS.paper.force33[k], `paper force[33].${k}`);
        }
        let maxForce = 0;
        for (let i = 0; i < n; i++) {
            const fx = paper.force[3 * i];
            const fy = paper.force[3 * i + 1];
            const fz = paper.force[3 * i + 2];
            maxForce = Math.max(maxForce, Math.sqrt(fx * fx + fy * fy + fz * fz));
        }
        expectPinned(maxForce, PINS.paper.maxForce, "paper max |F|");
        expectPinned(paper.partial[0], PINS.paper.partial[0], "paper swing sum");
        expectPinned(paper.partial[1], PINS.paper.partial[1], "paper traction sum");
        expect(paper.partial[0] / paper.partial[1]).toBe(2);
        expectPinned(paper.speed, PINS.paper.speed, "paper speed");
        expectPinned(paper.speedEfficiency, PINS.paper.speedEfficiency, "paper speedEfficiency");

        const networkx = expectedFor(NETWORKX);
        for (let k = 0; k < 3; k++) {
            expectPinned(networkx.force[k], PINS.networkx.force0[k], `networkx force[0].${k}`);
        }
        expectPinned(networkx.partial[0], PINS.networkx.partial[0], "networkx swing sum");
        expectPinned(networkx.partial[1], PINS.networkx.partial[1], "networkx traction sum");
        expectPinned(networkx.swing, PINS.networkx.swing, "networkx swing (accumulated from 1)");
        expectPinned(networkx.traction, PINS.networkx.traction, "networkx traction (accumulated from 1)");
        expect(networkx.swing / networkx.traction).toBeLessThan(2);
        expectPinned(networkx.speed, PINS.networkx.speed, "networkx speed");
        expectPinned(networkx.speedEfficiency, PINS.networkx.speedEfficiency, "networkx speedEfficiency");

        const strong = expectedFor(STRONG);
        for (let k = 0; k < 3; k++) {
            expectPinned(strong.force[k], PINS.strong.force0[k], `strong force[0].${k}`);
        }
        expectPinned(strong.partial[0], PINS.strong.partial[0], "strong swing sum");
        expectPinned(strong.partial[1], PINS.strong.partial[1], "strong traction sum");
        expectPinned(strong.speed, PINS.strong.speed, "strong speed");
        expectPinned(strong.speedEfficiency, PINS.strong.speedEfficiency, "strong speedEfficiency");

        const prev = expectedFor(PAPER_PREV);
        expect(prev.oldForce).not.toBeNull();
        expectPinned(prev.partial[0], PINS.paperPrev.partial[0], "paper-prev swing sum");
        expectPinned(prev.partial[1], PINS.paperPrev.partial[1], "paper-prev traction sum");
        expect(prev.swing / prev.traction).toBeGreaterThan(2);
        expectPinned(prev.speed, PINS.paperPrev.speed, "paper-prev speed");
        expectPinned(prev.speedEfficiency, PINS.paperPrev.speedEfficiency, "paper-prev speedEfficiency");

        writeNoiseFixture("fa2-repulsion-exact", "karate", ORACLE_CLASS, paper.force, "f32");
        for (const leg of LEGS) {
            if (leg.fixture === null) {
                continue;
            }
            const e = expectedFor(leg);
            writeNoiseFixture(
                "fa2-speed-finalize",
                leg.fixture,
                ORACLE_CLASS,
                [e.swing, e.traction, e.speed, e.speedEfficiency],
                "f32",
            );
        }
    });

    it("RepulsionExact.specs() names the two registry entries with the override values", () => {
        const [k3, k4] = RepulsionExact.specs({ SWING_MODE: 1, STRONG_GRAVITY: true, GRAVITY_CENTER: 1 });
        expect(k3.id).toBe("fa2-repulsion-exact");
        expect(k3.overrides).toMatchObject({ SWING_MODE: 1, STRONG_GRAVITY: true, GRAVITY_CENTER: 1 });
        expect(k3.needs).toContain("subgroups");
        expect(k4.id).toBe("fa2-speed-finalize");
        expect(k4.overrides).toMatchObject({ SWING_MODE: 1 });
        expect(k4.needs).toContain("subgroups");
        const [p3, p4] = RepulsionExact.specs(PAPER.overrides);
        expect(p3.overrides).toMatchObject({ SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 });
        expect(p4.overrides).toMatchObject({ SWING_MODE: 0 });
    });

    it("record() before bind() throws E_NOT_LOADED; the overrides are frozen", async (t) => {
        requireGpu(t);
        const ctx = await main();
        const stage = await RepulsionExact.create(ctx.pipelines, ctx.caps, PAPER.overrides);
        expect(Object.isFrozen(stage.overrides)).toBe(true);
        expect(stage.overrides).toEqual(PAPER.overrides);
        const encoder = ctx.device.createCommandEncoder({ label: "skeleton/unbound" });
        const pass = encoder.beginComputePass();
        let caught: unknown = null;
        try {
            stage.record(pass, n, 0);
        } catch (err) {
            caught = err;
        }
        expect(hasErrorCode(caught, "E_NOT_LOADED")).toBe(true);
        caught = null;
        try {
            stage.recordSpeedFinalize(pass, 0);
        } catch (err) {
            caught = err;
        }
        expect(hasErrorCode(caught, "E_NOT_LOADED")).toBe(true);
        pass.end();
        encoder.finish();
        ctx.assertReady();
    });

    for (const leg of LEGS) {
        describe(`leg ${leg.name} (SWING_MODE ${leg.overrides.SWING_MODE}, GRAVITY_CENTER ${leg.overrides.GRAVITY_CENTER}, STRONG_GRAVITY ${leg.overrides.STRONG_GRAVITY}, jitterTolerance ${leg.jitterTolerance}, oldForce ${leg.prev ? "-0.5 x F" : "0"})`, () => {
            it("runs twice bitwise-identically and the force after K3 equals the f64 pair sum + gravity", async (t) => {
                requireGpu(t);
                const [first, second] = await runsOf(leg);
                expectBitwiseEqual(first.force, second.force, `${leg.name}: force, run 1 vs run 2`);
                expectBitwiseEqual(first.stateWords, second.stateWords, `${leg.name}: state, run 1 vs run 2`);
                expectBitwiseEqual(first.partialWords, second.partialWords, `${leg.name}: partials, run 1 vs run 2`);
                const expected = expectedFor(leg);
                const err = flooredRelError(first.force, expected.force, FLOOR_FRACTION);
                console.warn(
                    `[skeleton] ${leg.name}: force vs f64 max ${err.max.toExponential(3)} (node ${err.argmax}), rms ${err.rms.toExponential(3)}, p99 ${err.p99.toExponential(3)}`,
                );
                expect(
                    err.max,
                    `${leg.name}: force vs the f64 reference, worst node ${err.argmax}`,
                ).toBeLessThanOrEqual(forceTolerance());
                for (let i = 0; i < n; i++) {
                    expect(Math.abs(first.force[3 * i + 2]), `${leg.name}: force[${i}].z of a 2D layout`).toBe(0);
                }
            });

            it("the trace slot, the state header and partials[0].swingTraction equal the hand-computed controller values", async (t) => {
                requireGpu(t);
                const [first] = await runsOf(leg);
                const expected = expectedFor(leg);
                const want = [expected.swing, expected.traction, expected.speed, expected.speedEfficiency];
                const header = controllerOf(first.header);
                const trace = controllerOf(first.trace);
                const headerErr = maxRelError(header, want, TRACE_ABS_FLOOR);
                const traceErr = maxRelError(trace, want, TRACE_ABS_FLOOR);
                const partialErr = maxRelError(
                    vector(first.partial, "swingTraction"),
                    expected.partial,
                    TRACE_ABS_FLOOR,
                );
                console.warn(
                    `[skeleton] ${leg.name}: trace ${trace.join(", ")} vs ${want.join(", ")} -> header ${headerErr.toExponential(3)}, slot ${traceErr.toExponential(3)}, partials ${partialErr.toExponential(3)}`,
                );
                expect(
                    headerErr,
                    `${leg.name}: state header (swing, traction, speed, speedEfficiency)`,
                ).toBeLessThanOrEqual(traceTolerance());
                expect(traceErr, `${leg.name}: trace slot ${TRACE_SLOT}`).toBeLessThanOrEqual(traceTolerance());
                expect(partialErr, `${leg.name}: partials[0].swingTraction`).toBeLessThanOrEqual(traceTolerance());
                expect(controllerOf(first.traceUntouched), `${leg.name}: trace slot 0 is never written`).toEqual([
                    0, 0, 0, 0,
                ]);
                expect(scalar(first.trace, "meanDisplacement")).toBe(0);
                expect(scalar(first.trace, "settledCount")).toBe(0);
                expect(scalar(first.trace, "iteration")).toBe(0);
                expect(scalar(first.header, "iteration")).toBe(0);
                expect(scalar(first.header, "settledCount")).toBe(0);
                expect(vector(first.header, "centroid")).toEqual([
                    stats.centroid[0],
                    stats.centroid[1],
                    stats.centroid[2],
                    0,
                ]);
                expect(vector(first.partial, "sum")).toEqual([0, 0, 0, 0]);
                expect(vector(first.partial, "dispFree")).toEqual([0, 0]);
                if (leg.fixture !== null) {
                    const ctx = await main();
                    const cls = adapterClass(ctx.caps);
                    if (leg.name === PAPER.name) {
                        writeNoiseFixture("fa2-repulsion-exact", "karate", cls, first.force, "f32");
                    }
                    writeNoiseFixture("fa2-speed-finalize", leg.fixture, cls, trace, "f32");
                }
            });
        });
    }

    it("the workgroup twin (acquire({ subgroups: false })) agrees within the twin floors and runs twice bitwise", async (t) => {
        requireGpu(t);
        const ctx = await main();
        const twin = await acquire({ subgroups: false, label: "skeleton-twin" });
        expect(twin.caps.features.has("subgroups")).toBe(false);
        const first = await runSkeleton(twin, PAPER.overrides, null, PAPER.jitterTolerance);
        const second = await runSkeleton(twin, PAPER.overrides, null, PAPER.jitterTolerance);
        expectBitwiseEqual(first.force, second.force, "twin: force, run 1 vs run 2");
        expectBitwiseEqual(first.stateWords, second.stateWords, "twin: state, run 1 vs run 2");
        const [mainRun] = await runsOf(PAPER);
        const forceErr = flooredRelError(first.force, mainRun.force, FLOOR_FRACTION).max;
        const traceErr = maxRelError(controllerOf(first.trace), controllerOf(mainRun.trace), TRACE_ABS_FLOOR);
        const cls = adapterClass(ctx.caps);
        console.warn(
            `[skeleton] twin on ${cls}: main subgroups=${ctx.caps.features.has("subgroups")} -> force ${forceErr.toExponential(3)}, trace ${traceErr.toExponential(3)}`,
        );
        expect(forceErr, "twin: force vs the subgroup form").toBeLessThanOrEqual(
            noiseFloorFor("fa2-skeleton.force.twin").value,
        );
        expect(traceErr, "twin: trace vs the subgroup form").toBeLessThanOrEqual(
            noiseFloorFor("fa2-skeleton.trace.twin").value,
        );
        writeNoiseFixture("fa2-repulsion-exact", "karate", `${cls}${TWIN_SUFFIX}`, first.force, "f32");
        writeNoiseFixture("fa2-speed-finalize", "karate", `${cls}${TWIN_SUFFIX}`, controllerOf(first.trace), "f32");
    });

    it("agrees with every committed adapter output within the cross-adapter floors (spec 11.5: 1e-5 across adapters)", async (t) => {
        requireGpu(t);
        const ctx = await main();
        const cls = adapterClass(ctx.caps);
        const [paperRun] = await runsOf(PAPER);
        let compared = 0;
        for (const fixture of readNoiseFixtures("fa2-repulsion-exact", "karate")) {
            if (!isAdapterFixture(fixture.adapterClass)) {
                continue;
            }
            const err = flooredRelError(paperRun.force, fixture.values, FLOOR_FRACTION).max;
            console.warn(`[skeleton] force: ${cls} vs committed ${fixture.adapterClass}: ${err.toExponential(3)}`);
            expect(err, `force: ${cls} vs committed ${fixture.adapterClass}`).toBeLessThanOrEqual(
                noiseFloorFor("fa2-skeleton.force.cross").value,
            );
            compared++;
        }
        for (const leg of LEGS) {
            if (leg.fixture === null) {
                continue;
            }
            const [run] = await runsOf(leg);
            for (const fixture of readNoiseFixtures("fa2-speed-finalize", leg.fixture)) {
                if (!isAdapterFixture(fixture.adapterClass)) {
                    continue;
                }
                const err = maxRelError(controllerOf(run.trace), fixture.values, TRACE_ABS_FLOOR);
                console.warn(
                    `[skeleton] trace ${leg.fixture}: ${cls} vs committed ${fixture.adapterClass}: ${err.toExponential(3)}`,
                );
                expect(err, `trace ${leg.fixture}: ${cls} vs committed ${fixture.adapterClass}`).toBeLessThanOrEqual(
                    noiseFloorFor("fa2-skeleton.trace.cross").value,
                );
                compared++;
            }
        }
        if (compared === 0) {
            console.warn(
                "[skeleton] no committed adapter fixture yet: run the Step 7 measurement sequence (GRAPHTY_NOISE_FLOOR_WRITE=1 on every adapter)",
            );
        }
    });
});
