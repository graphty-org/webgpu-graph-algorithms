/**
 * The K3 / K4 sabotage matrix (spec 11.9 item 1; contract 5.5 P1-T6): every mutation P1-T5 lists for
 * "fa2-repulsion-exact" and "fa2-speed-finalize" is spliced into the kernel body (setKernelBodyOverride through
 * withSabotage) and must move the skeleton iteration's force or trace by at least minFactor x the tolerance the
 * skeleton test applies -- the same 11.4 floored per-node force metric and the same relative trace metric, taken
 * from benchmarks/results/noise-floor.json -- measured against the PRISTINE kernel's own output (within the
 * oracle-f64 floor, ~3e-7, of the f64 reference: three orders of magnitude below the 10x margin, so the verdict is
 * the reference's and the reference stays in test/layouts/skeleton.test.ts). A mutation that survives is a bug in
 * the test suite and blocks G1 exactly as a failing test does (spec 13 rule f).
 *
 * The inputs below (positions recipe, state initialisation, params, legs) are the skeleton test's, copied; the
 * paper-prev leg's oldForce is -0.5 x the PRISTINE paper force of this run (rounded to f32) instead of the
 * reference's, which is the same value to f32 noise.
 */

import { type F32 } from "@graphty/graph-format";

import { PARTIAL_BYTES, STATE_HEADER_BYTES, TRACE_RECORD_BYTES } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { type UniformValues } from "../../src/kernel/struct-block.js";
import { FA2_PARAMS, FA2_PARTIAL, FA2_STATE, FA2_TRACE, type KernelId } from "../../src/kernels.js";
import {
    RepulsionExact,
    type RepulsionExactOverrides,
    type RepulsionExactResources,
} from "../../src/layouts/repulsion-exact.js";
import { bindingOf, readF32, scratchBuffer, uploadBuffer } from "../helpers/device.js";
import { KARATE_EDGES, snapshotOf } from "../helpers/graphs.js";
import { flooredRelError, maxRelError } from "../helpers/matchers.js";
import { noiseFloorFor } from "../helpers/noise-floor.js";
import { type Mutation, SABOTAGE, withSabotage } from "../helpers/sabotage.js";
import { acquire, requireGpu } from "../setup/gpu.js";

// ---------------------------------------------------------------- the skeleton's inputs (copied from test/layouts/skeleton.test.ts)

const SKELETON_SEED = 12345;
const LCG_M = 34359738337;
const LCG_A = 185852;
const LCG_C = 1;
const SCALING_RATIO = 2;
const GRAVITY = 1;
const DIM = 2;
const INITIAL_SPEED = 1;
const INITIAL_EFFICIENCY = 1;
const INITIAL_SWING = 1;
const INITIAL_TRACTION = 1;
const TRACE_SLOT = 1;
const TRACE_RECORDS = 4;
const STATE_BYTES = STATE_HEADER_BYTES + TRACE_RECORDS * TRACE_RECORD_BYTES;
const FLOOR_FRACTION = 1e-3;
const TRACE_ABS_FLOOR = 1e-6;
/** The test file every K3 / K4 row must name (contract 5.2 Mutation.test). */
const SKELETON_TEST = "test/layouts/skeleton.test.ts";
/** The two kernels this file covers and the minimum row count per kernel (spec 13 rule f). */
const KERNELS_UNDER_TEST: readonly KernelId[] = ["fa2-repulsion-exact", "fa2-speed-finalize"];
const MIN_ROWS = 3;

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

interface Leg {
    readonly name: string;
    readonly overrides: RepulsionExactOverrides;
    readonly jitterTolerance: number;
    /** oldForce = -0.5 x the pristine paper force (rounded to f32) instead of zero. */
    readonly prev: boolean;
}

const PAPER: Leg = {
    name: "paper",
    overrides: { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 },
    jitterTolerance: 1,
    prev: false,
};
const LEGS: readonly Leg[] = [
    PAPER,
    {
        name: "networkx",
        overrides: { SWING_MODE: 1, STRONG_GRAVITY: false, GRAVITY_CENTER: 1 },
        jitterTolerance: 1,
        prev: false,
    },
    {
        name: "strong",
        overrides: { SWING_MODE: 0, STRONG_GRAVITY: true, GRAVITY_CENTER: 0 },
        jitterTolerance: 1,
        prev: false,
    },
    {
        name: "paper-prev",
        overrides: { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 },
        jitterTolerance: 0.5,
        prev: true,
    },
];

const snapshot = snapshotOf(KARATE_EDGES);
const n = snapshot.nodeCount;
const positions = skeletonPositions(snapshot.outDegree());
const stats = cpuStats(positions);

// ---------------------------------------------------------------- the P1 driver (copied from test/layouts/skeleton.test.ts)

interface SkeletonOutput {
    readonly force: F32;
    readonly header: UniformValues;
    readonly trace: UniformValues;
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

function controllerOf(values: UniformValues): readonly [number, number, number, number] {
    return [
        scalar(values, "swing"),
        scalar(values, "traction"),
        scalar(values, "speed"),
        scalar(values, "speedEfficiency"),
    ];
}

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
    const pos = uploadBuffer(ctx, positions, "sabotage/pos");
    const force = scratchBuffer(ctx, 12 * n, "sabotage/force");
    const old =
        oldForce === null
            ? scratchBuffer(ctx, 12 * n, "sabotage/oldForce")
            : uploadBuffer(ctx, oldForce, "sabotage/oldForce");
    const fixed = scratchBuffer(ctx, 4 * Math.ceil(n / 32), "sabotage/fixed");
    const partials = scratchBuffer(ctx, PARTIAL_BYTES * groups, "sabotage/partials");
    const state = uploadBuffer(ctx, stateBytes, "sabotage/state");
    const params = uploadBuffer(ctx, paramsBytes, "sabotage/params", BufferUsage.UNIFORM);
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
        const encoder = ctx.device.createCommandEncoder({ label: "sabotage" });
        const pass = encoder.beginComputePass({ label: "sabotage/K3+K4" });
        stage.record(pass, n, 0);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        const forceOut = await readF32(ctx, force, 3 * n);
        const stateView = new DataView(await ctx.readback.read(state, STATE_BYTES));
        const partialView = new DataView(await ctx.readback.read(partials, PARTIAL_BYTES * groups));
        ctx.assertReady();
        return {
            force: forceOut,
            header: FA2_STATE.read(stateView, 0),
            trace: FA2_TRACE.read(stateView, STATE_HEADER_BYTES + TRACE_SLOT * TRACE_RECORD_BYTES),
            partial: FA2_PARTIAL.read(partialView, 0),
        };
    } finally {
        for (const buffer of buffers) {
            buffer.destroy();
        }
    }
}

// ---------------------------------------------------------------- the pristine baseline and the detection factor

interface Baseline {
    readonly outputs: ReadonlyMap<string, SkeletonOutput>;
    /** -0.5 x the pristine paper force, rounded to f32: the host-written oldForce of the paper-prev leg. */
    readonly prevOldForce: Float32Array<ArrayBuffer>;
}

function halfNegated(force: ArrayLike<number>): Float32Array<ArrayBuffer> {
    const out = new Float32Array(force.length);
    for (let i = 0; i < force.length; i++) {
        out[i] = Math.fround(-0.5 * Math.fround(force[i]));
    }
    return out;
}

/** Every leg on one context; `prevOldForce` is taken from this context's own paper run (the baseline's when running a mutant). */
async function runLegs(ctx: GpuContext, prevOldForce: Float32Array<ArrayBuffer> | null): Promise<Baseline> {
    const outputs = new Map<string, SkeletonOutput>();
    const paper = await runSkeleton(ctx, PAPER.overrides, null, PAPER.jitterTolerance);
    outputs.set(PAPER.name, paper);
    const old = prevOldForce ?? halfNegated(paper.force);
    for (const leg of LEGS) {
        if (leg.name === PAPER.name) {
            continue;
        }
        outputs.set(leg.name, await runSkeleton(ctx, leg.overrides, leg.prev ? old : null, leg.jitterTolerance));
    }
    return { outputs, prevOldForce: old };
}

let baselinePromise: Promise<Baseline> | null = null;

function baseline(): Promise<Baseline> {
    if (baselinePromise === null) {
        baselinePromise = acquire({ label: "sabotage-pristine" }).then((ctx) => runLegs(ctx, null));
    }
    return baselinePromise;
}

/** A non-finite error (a NaN force) is a detected mutation, not a missing one. */
function finiteOrDetected(value: number): number {
    return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

/** The skeleton test's metrics divided by its tolerances, maximised over the force, the header, the trace slot and the partials. */
function detectionFactor(mutant: SkeletonOutput, pristine: SkeletonOutput, forceTol: number, traceTol: number): number {
    const force = flooredRelError(mutant.force, pristine.force, FLOOR_FRACTION).max / forceTol;
    const header = maxRelError(controllerOf(mutant.header), controllerOf(pristine.header), TRACE_ABS_FLOOR) / traceTol;
    const trace = maxRelError(controllerOf(mutant.trace), controllerOf(pristine.trace), TRACE_ABS_FLOOR) / traceTol;
    const partial =
        maxRelError(
            vector(mutant.partial, "swingTraction"),
            vector(pristine.partial, "swingTraction"),
            TRACE_ABS_FLOOR,
        ) / traceTol;
    return Math.max(
        finiteOrDetected(force),
        finiteOrDetected(header),
        finiteOrDetected(trace),
        finiteOrDetected(partial),
    );
}

function rowsOf(id: KernelId): readonly Mutation[] {
    return SABOTAGE[id] ?? [];
}

// ---------------------------------------------------------------- the tests

describe("sabotage: the K3 / K4 mutations fail the skeleton test by >= minFactor", () => {
    for (const id of KERNELS_UNDER_TEST) {
        describe(id, () => {
            it(`has at least ${MIN_ROWS} rows, each naming ${SKELETON_TEST} with minFactor >= 10`, () => {
                const rows = rowsOf(id);
                expect(rows.length).toBeGreaterThanOrEqual(MIN_ROWS);
                const names = new Set(rows.map((row) => row.name));
                expect(names.size).toBe(rows.length);
                for (const row of rows) {
                    expect(row.test, `${id}/${row.name}`).toBe(SKELETON_TEST);
                    expect(row.minFactor, `${id}/${row.name}`).toBeGreaterThanOrEqual(10);
                    expect(row.find.length, `${id}/${row.name}: find`).toBeGreaterThan(0);
                    expect(row.replace, `${id}/${row.name}: replace differs from find`).not.toBe(row.find);
                }
            });

            for (const mutation of rowsOf(id)) {
                it(`${mutation.name} moves the skeleton by >= ${mutation.minFactor}x its tolerance`, async (t) => {
                    requireGpu(t);
                    const pristine = await baseline();
                    const forceTol = noiseFloorFor("fa2-skeleton.force").value;
                    const traceTol = noiseFloorFor("fa2-skeleton.trace").value;
                    const mutant = await withSabotage(id, mutation, (ctx) => runLegs(ctx, pristine.prevOldForce));
                    let factor = 0;
                    let worstLeg = "";
                    for (const leg of LEGS) {
                        const a = mutant.outputs.get(leg.name);
                        const b = pristine.outputs.get(leg.name);
                        if (a === undefined || b === undefined) {
                            throw new Error(`leg ${leg.name} did not run`);
                        }
                        const legFactor = detectionFactor(a, b, forceTol, traceTol);
                        if (legFactor > factor) {
                            factor = legFactor;
                            worstLeg = leg.name;
                        }
                    }
                    console.warn(
                        `[sabotage] ${id}/${mutation.name}: ${factor.toExponential(2)}x the tolerance (leg ${worstLeg}; required ${mutation.minFactor}x)`,
                    );
                    expect(
                        factor,
                        `${id}/${mutation.name} survived: the skeleton test cannot tell it from the pristine kernel`,
                    ).toBeGreaterThanOrEqual(mutation.minFactor);
                });
            }
        });
    }
});
