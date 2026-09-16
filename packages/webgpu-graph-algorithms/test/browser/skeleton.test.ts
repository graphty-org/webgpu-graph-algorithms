/**
 * The walking skeleton in the browser (spec 11.5, 11.6 item 2; contract 5.5): arena and fromCsr uploads, `degree` on
 * karate / random1k / arcCount 0 against the oracle bitwise, the 17M-item fill on the 2D dispatch with the pinned checksum
 * (the SwiftShader / NVIDIA-Chromium leg of "bitwise across adapters"), the random1k f32 prefix sums through the reduce
 * primitive against the f64 oracle, one K3 + K4 exact-tile FA2 iteration on karate in paper mode from host-written state
 * against the f64 pair sum and the hand-computed controller, twice bitwise, and the SwiftShader / NVIDIA-Chromium noise
 * fixtures written through the commands bridge when GRAPHTY_NOISE_FLOOR_WRITE=1 (spec 11.9 item 3). Both CI browser
 * adapters expose "subgroups" (SwiftShader size 4, NVIDIA 32), so the subgroup twin is the form exercised here (spec 11.6
 * item 5; the workgroup twin is the compile matrix of P2).
 *
 * Every fixture input is the SAME as the Node writers' (Step 31 of the P1-T7 plan), so test/noise-floor.test.ts pairs
 * files computed from identical inputs: degree / random1k is `fixture("random1k", 1)` (test/algorithms/degree.test.ts);
 * reduce / random1k is the f32 sum of `reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED)` at every REDUCE_NOISE_COUNTS
 * prefix (test/helpers/reduce-input.ts, test/primitives/reduce.test.ts, with the 64-element poison pad of
 * test/helpers/reduce-check.ts); the FA2 positions, state header and params are those of test/layouts/skeleton.test.ts's
 * paper leg (the LCG seed 12345 recipe, whose pinned numbers are asserted below so a drift between the two copies is red
 * here before it shows up as a cross-adapter disagreement). The Node helpers themselves import test/setup/gpu.ts
 * (process.env, Dawn) or node:fs and cannot be bundled for Chromium, hence the local copies (PLAN DECISIONS a, b).
 */

import { type GraphSnapshot } from "@graphty/graph-format";

import noiseFloorDoc from "../../benchmarks/results/noise-floor.json";
import { degree } from "../../src/algorithms/degree.js";
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
import { plan1d } from "../../src/kernel/dispatch.js";
import { type UniformValues } from "../../src/kernel/struct-block.js";
import { FA2_PARAMS, FA2_STATE, FA2_TRACE, FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { RepulsionExact, type RepulsionExactOverrides } from "../../src/layouts/repulsion-exact.js";
import { prepareReduce, type ReduceScope } from "../../src/primitives/reduce.js";
import { type GpuCaps } from "../../src/types/context.js";
import { type Binding } from "../../src/types/memory.js";
import { csrSnapshotOf, fixture, KARATE_EDGES, snapshotOf } from "../helpers/graphs.js";
import { runKernel } from "../helpers/kernel.js";
import {
    LINEAR_ID_CHECKSUM,
    LINEAR_ID_ITEMS,
    LINEAR_ID_SAMPLES,
    LINEAR_ID_VALUE,
    linearIdChecksum,
} from "../helpers/linear-id.js";
import { expectBitwiseEqual, flooredRelError, maxRelError } from "../helpers/matchers.js";
import {
    adapterClassBrowser,
    recordNoiseRowBrowser,
    writeNoiseFixtureBrowser,
} from "../helpers/noise-floor-browser.js";
import { RANDOM1K_COUNT, RANDOM1K_SEED, REDUCE_NOISE_COUNTS, reduceInput } from "../helpers/reduce-input.js";
import { outDegreeOracle } from "../oracle/degree.js";
import { reduceOracle } from "../oracle/reduce.js";
import { acquireBrowser, browserAdapterOffersSubgroups, requireBrowserGpu } from "../setup/browser.js";

// ---- local buffer helpers (PLAN DECISION a: test/helpers/device.ts imports the Node setup and cannot be bundled here)

const STORAGE_USAGE = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;

/** A buffer of the given usage created straight on the device (destroyed by the test; never through the allocator). */
function createBuffer(ctx: GpuContext, byteLength: number, label: string, usage: number = STORAGE_USAGE): GPUBuffer {
    return ctx.device.createBuffer({ size: byteLength, usage, label });
}

/** A buffer holding `data` after one writeBuffer. */
function uploadArray(
    ctx: GpuContext,
    data: ArrayBufferView<ArrayBuffer>,
    label: string,
    usage: number = STORAGE_USAGE,
): GPUBuffer {
    const buffer = createBuffer(ctx, data.byteLength, label, usage);
    ctx.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

/** The whole-buffer Binding. */
function whole(buffer: GPUBuffer): Binding {
    return { buffer, offset: 0, size: buffer.size, window: null };
}

/** Reads `count` u32 words through the context's staging ring. */
async function readU32(
    ctx: GpuContext,
    buffer: GPUBuffer,
    count: number,
    byteOffset = 0,
): Promise<Uint32Array<ArrayBuffer>> {
    return new Uint32Array(await ctx.readback.read(buffer, count * 4, undefined, byteOffset));
}

/** Reads `count` f32 values through the context's staging ring. */
async function readF32(
    ctx: GpuContext,
    buffer: GPUBuffer,
    count: number,
    byteOffset = 0,
): Promise<Float32Array<ArrayBuffer>> {
    return new Float32Array(await ctx.readback.read(buffer, count * 4, undefined, byteOffset));
}

/** One scalar field of a block's read(). */
function scalar(values: UniformValues, name: string): number {
    const v = values[name];
    if (typeof v !== "number") {
        throw new Error(`field ${name} is not a scalar`);
    }
    return v;
}

/** The tolerance of a test id from the committed noise-floor file (PLAN DECISION e: a static JSON import, inlined by Vite). */
function toleranceFor(id: string): number {
    const doc = noiseFloorDoc as unknown as {
        readonly tolerances: Readonly<Record<string, { readonly value: number; readonly basis: string } | undefined>>;
    };
    const entry = doc.tolerances[id];
    if (entry === undefined) {
        throw new Error(
            `benchmarks/results/noise-floor.json has no tolerance "${id}" (P1-T6 records it before this test runs)`,
        );
    }
    return entry.value;
}

// ---- the reduce input (reconciled with test/primitives/reduce.test.ts and test/helpers/reduce-check.ts)

/** test/helpers/reduce-check.ts: every uploaded input carries PAD_ELEMENTS poison elements past `count`. */
const REDUCE_PAD_ELEMENTS = 64;
/** test/helpers/reduce-check.ts poisonFor("sum", "f32"): a value a correct sum never reads. */
const REDUCE_SUM_POISON = 1048576;
/** test/helpers/reduce-check.ts OUT_POISON: the output region is poisoned so an unwritten element is visible. */
const OUT_POISON = 0xdeadbeef;

/** The words of the random1k f32 input followed by the poison pad (paddedInput of reduce-check.ts). */
function paddedSumInput(values: Float32Array): Uint32Array<ArrayBuffer> {
    const floats = new Float32Array(values.length + REDUCE_PAD_ELEMENTS);
    floats.set(values);
    floats.fill(REDUCE_SUM_POISON, values.length);
    return new Uint32Array(floats.buffer);
}

/**
 * The 11 f32 prefix sums of the random1k input through the reduce primitive (one prepareReduce, one record + submit +
 * readback per count, the output region poisoned first): the values of the reduce / random1k fixture.
 */
async function reduceSums(ctx: GpuContext, values: Float32Array, src: GPUBuffer): Promise<Float32Array<ArrayBuffer>> {
    const scratch: GPUBuffer[] = [];
    const scope: ReduceScope = {
        device: ctx.device,
        caps: ctx.caps,
        pipelines: ctx.pipelines,
        pool: ctx.pool,
        workgroupSize: ctx.workgroupSize,
        scratch: (byteLength, label) => {
            const buffer = ctx.pool.acquire(byteLength, STORAGE_USAGE, label);
            scratch.push(buffer);
            return buffer;
        },
        params: (block, blockValues) => {
            const buffer = ctx.pool.acquire(
                block.byteLength,
                BufferUsage.UNIFORM | BufferUsage.COPY_DST,
                `${block.name}/params`,
            );
            scratch.push(buffer);
            const bytes = new ArrayBuffer(block.byteLength);
            block.write(new DataView(bytes), blockValues);
            ctx.device.queue.writeBuffer(buffer, 0, bytes);
            return { binding: { buffer, offset: 0, size: block.byteLength, window: null }, offset: 0 };
        },
    };
    const sums = new Float32Array(REDUCE_NOISE_COUNTS.length);
    const out = createBuffer(ctx, 8, "skeleton/reduce-out");
    try {
        const planner = await prepareReduce(scope, "sum", "f32");
        for (const [j, count] of REDUCE_NOISE_COUNTS.entries()) {
            ctx.device.queue.writeBuffer(out, 0, new Uint32Array(2).fill(OUT_POISON));
            const encoder = ctx.device.createCommandEncoder({ label: `skeleton/reduce-${count}` });
            const pass = encoder.beginComputePass({ label: "skeleton/reduce" });
            planner.record(pass, whole(src), count, whole(out), 0);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);
            expect(planner.lastDispatches, `count ${count}`).toBeGreaterThanOrEqual(2);
            const words = await readU32(ctx, out, 2);
            expect(words[1], `count ${count}: the second output word is untouched`).toBe(OUT_POISON);
            sums[j] = new Float32Array(words.buffer, 0, 1)[0];
            const exact = reduceOracle(values.subarray(0, count), "sum", "f32");
            if (typeof exact !== "number") {
                throw new Error("a scalar sum");
            }
            expect(Math.abs(sums[j] - exact) / exact, `sum of ${count} vs the f64 oracle`).toBeLessThanOrEqual(
                count * 2 ** -24,
            );
        }
    } finally {
        out.destroy();
        for (const buffer of scratch) {
            ctx.pool.release(buffer);
        }
    }
    return sums;
}

// ---- the FA2 skeleton reference (PLAN DECISION b: the paper leg of test/layouts/skeleton.test.ts, copied line for line)

/** The LCG seed of the skeleton positions (test/layouts/skeleton.test.ts SKELETON_SEED). */
const SKELETON_SEED = 12345;
/** The CPU port's LCG (spec 7.2; layout/src/utils/random.ts): m = 2^35 - 31, a = 185852, c = 1. */
const LCG_M = 34359738337;
const LCG_A = 185852;
const LCG_C = 1;
/** The params of the paper leg (FA2_DEFAULTS: scalingRatio 2, gravity 1; a 2D layout; jitterTolerance 1). */
const SCALING_RATIO = 2;
const GRAVITY = 1;
const DIM = 2;
const PAPER_JITTER_TOLERANCE = 1;
/** The host-written controller state (spec 7.17: load() writes speed = speedEfficiency = 1; contract 3.13 onLoad: swing = traction = 1). */
const INITIAL_SPEED = 1;
const INITIAL_EFFICIENCY = 1;
const INITIAL_SWING = 1;
const INITIAL_TRACTION = 1;
/** The trace slot K4 writes (P.iterationIndex) and the records the trace binding covers. */
const TRACE_SLOT = 1;
const TRACE_RECORDS = 4;
const STATE_BYTES = STATE_HEADER_BYTES + TRACE_RECORDS * TRACE_RECORD_BYTES;
/** Spec 11.4: the floored denominator of the per-node force error; spec 11.5: the absolute floor of the relative trace comparison. */
const FLOOR_FRACTION = 1e-3;
const TRACE_ABS_FLOOR = 1e-6;
/** The pinned f64 numbers are reproduced to this relative precision (a fixed evaluation order; ulp-level slack only). */
const PIN_TOLERANCE = 1e-12;
const PAPER: RepulsionExactOverrides = { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 };

/** The numbers test/layouts/skeleton.test.ts pins for the same recipe (its PINS constant, paper leg). */
const PINS = {
    centroid: [-0.07239578664302826, -0.01930960826575756, 0],
    position0: [-0.8664516806602478, -0.7746485471725464],
    mass0: 17,
    position33: [0.3723199963569641, -0.5822708010673523],
    mass33: 18,
    guards: { pair: 0.027098824484323374, centroid: 0.04381973544344613 },
    force0: [-3376.030304066258, -1777.8039444967717, 0],
    force33: [2559.865638550776, -3376.268415820058, 0],
    maxForce: 4236.991916808086,
    partial: [366257.0467533182, 183128.5233766591],
    speed: 1.5,
    speedEfficiency: 1.3,
} as const;

/** The skeleton's device positions: vec4f rows (x, y, 0, mass = outDegree + 1), x and y from the port's LCG mapped to [-1, 1). */
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

/** What load() writes into the state header: the f32-rounded centroid, bbox, RMS radius and max |p - c|, in f64. */
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

/** The smallest pair distance and the smallest |p - centroid| (the floor, kick and gravity-guard regions). */
function guardDistances(
    positions: Float32Array,
    centroid: readonly [number, number, number],
): { pair: number; centroid: number } {
    const n = positions.length / 4;
    let pair = Infinity;
    let toCentroid = Infinity;
    for (let i = 0; i < n; i++) {
        const px = positions[4 * i];
        const py = positions[4 * i + 1];
        const qx = px - centroid[0];
        const qy = py - centroid[1];
        toCentroid = Math.min(toCentroid, Math.sqrt(qx * qx + qy * qy));
        for (let j = i + 1; j < n; j++) {
            const dx = px - positions[4 * j];
            const dy = py - positions[4 * j + 1];
            pair = Math.min(pair, Math.sqrt(dx * dx + dy * dy));
        }
    }
    return { pair, centroid: toCentroid };
}

/**
 * The f64 reference of K3's force on the f32 inputs, paper leg (spec 7.2, 7.6, 7.9): repulsion `k m_i m_j / d` along
 * `(p_i - p_j) / d` with `d^2 >= 1e-4`, then regular gravity `-g m_i q / |q|` when |q| > 0.01 about the f32 centroid.
 * `force` was zero, so the result IS the force after K3. The coincident kick never fires on this fixture (asserted).
 */
function referenceForces(positions: Float32Array, centroid: readonly [number, number, number]): Float64Array {
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
        const qx = px - centroid[0];
        const qy = py - centroid[1];
        const qz = pz - centroid[2];
        const d = Math.sqrt(qx * qx + qy * qy + qz * qz);
        if (d > FA2_DISTANCE_FLOOR) {
            fx += (-GRAVITY * mi * qx) / d;
            fy += (-GRAVITY * mi * qy) / d;
            fz += (-GRAVITY * mi * qz) / d;
        }
        force[3 * i] = fx;
        force[3 * i + 1] = fy;
        force[3 * i + 2] = fz;
    }
    return force;
}

/** The f64 sums of the K3 epilogue in paper mode with oldForce 0: swing_i = m |F|, traction_i = 0.5 m |F|, in index order. */
function referenceSums(positions: Float32Array, force: Float64Array): readonly [number, number] {
    const n = positions.length / 4;
    let swing = 0;
    let traction = 0;
    for (let i = 0; i < n; i++) {
        const m = positions[4 * i + 3];
        const len = Math.sqrt(force[3 * i] ** 2 + force[3 * i + 1] ** 2 + force[3 * i + 2] ** 2);
        swing += m * len;
        traction += 0.5 * m * len;
    }
    return [swing, traction];
}

/** estimateFactor in TypeScript, line for line the K4 body of contract 4.5 (spec 7.10): the new speed and speedEfficiency. */
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

/** The host-written inputs of one K3 + K4 iteration and their f64 reference. */
interface SkeletonReference {
    readonly n: number;
    readonly groups: number;
    /** xyz + mass per node (vec4f), f32. */
    readonly positions: Float32Array<ArrayBuffer>;
    /** The whole state buffer: the header (speed / speedEfficiency / swing / traction 1, the centroid, bbox, radii) and a zeroed trace region. */
    readonly state: Uint8Array<ArrayBuffer>;
    /** The Fa2Params record, written. */
    readonly params: Uint8Array<ArrayBuffer>;
    /** The f64 pair sum + gravity, stride 3. */
    readonly force: Float64Array;
    /** [swing, traction, speed, speedEfficiency] after K4 (paper mode, oldForce 0, mask clear). */
    readonly trace: readonly [number, number, number, number];
}

function skeletonReference(s: GraphSnapshot, workgroupSize: number): SkeletonReference {
    const n = s.nodeCount;
    const positions = skeletonPositions(s.outDegree());
    const stats = cpuStats(positions);
    const force = referenceForces(positions, stats.centroid);
    const [swing, traction] = referenceSums(positions, force);
    const controller = estimateFactor(swing, traction, INITIAL_SPEED, INITIAL_EFFICIENCY, n, PAPER_JITTER_TOLERANCE);
    const state = new Uint8Array(STATE_BYTES);
    FA2_STATE.write(
        new DataView(state.buffer),
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
    const params = new Uint8Array(FA2_PARAMS.byteLength);
    FA2_PARAMS.write(
        new DataView(params.buffer),
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
            jitterTolerance: PAPER_JITTER_TOLERANCE,
            scale: 1,
            center: [0, 0, 0, 0],
            settleThreshold: 0.001,
            extentFactor: 6,
        },
        0,
    );
    return {
        n,
        groups: Math.ceil(n / workgroupSize),
        positions,
        state,
        params,
        force,
        trace: [swing, traction, controller.speed, controller.speedEfficiency],
    };
}

function expectPinned(actual: number, pinned: number, label: string): void {
    expect(Math.abs(actual - pinned), `${label}: ${actual} vs pinned ${pinned}`).toBeLessThanOrEqual(
        PIN_TOLERANCE * Math.abs(pinned),
    );
}

interface PaperOutput {
    readonly force: Float32Array<ArrayBuffer>;
    readonly stateWords: Uint32Array<ArrayBuffer>;
    /** [swing, traction, speed, speedEfficiency] of trace slot TRACE_SLOT. */
    readonly trace: Float32Array<ArrayBuffer>;
    readonly header: UniformValues;
}

/** Uploads the host-written buffers, records K3 then K4 through RepulsionExact into one pass, submits, reads back; every buffer destroyed afterwards. */
async function runPaperLeg(ctx: GpuContext, ref: SkeletonReference): Promise<PaperOutput> {
    const { n } = ref;
    const stage = await RepulsionExact.create(ctx.pipelines, ctx.caps, PAPER);
    const pos = uploadArray(ctx, ref.positions, "skeleton/fa2-pos");
    const force = uploadArray(ctx, new Float32Array(3 * n), "skeleton/fa2-force");
    const oldForce = uploadArray(ctx, new Float32Array(3 * n), "skeleton/fa2-oldForce");
    const fixedMask = uploadArray(ctx, new Uint32Array(Math.ceil(n / 32)), "skeleton/fa2-fixedMask");
    const partials = uploadArray(ctx, new Uint8Array(ref.groups * PARTIAL_BYTES), "skeleton/fa2-partials");
    const state = uploadArray(ctx, ref.state, "skeleton/fa2-state");
    const params = uploadArray(ctx, ref.params, "skeleton/fa2-params", BufferUsage.UNIFORM | BufferUsage.COPY_DST);
    const buffers = [pos, force, oldForce, fixedMask, partials, state, params];
    try {
        stage.bind({
            pos: whole(pos),
            state: { buffer: state, offset: 0, size: STATE_HEADER_BYTES, window: null },
            trace: {
                buffer: state,
                offset: STATE_HEADER_BYTES,
                size: TRACE_RECORDS * TRACE_RECORD_BYTES,
                window: null,
            },
            force: whole(force),
            oldForce: whole(oldForce),
            fixedMask: whole(fixedMask),
            partials: whole(partials),
            params: { buffer: params, offset: 0, size: FA2_PARAMS.byteLength, window: null },
        });
        const encoder = ctx.device.createCommandEncoder({ label: "skeleton/fa2" });
        const pass = encoder.beginComputePass({ label: "skeleton/K3+K4" });
        stage.record(pass, n, 0);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        const forceOut = await readF32(ctx, force, 3 * n);
        const stateWords = await readU32(ctx, state, STATE_BYTES / 4);
        ctx.assertReady();
        const view = new DataView(stateWords.buffer);
        const record = FA2_TRACE.read(view, STATE_HEADER_BYTES + TRACE_SLOT * TRACE_RECORD_BYTES);
        return {
            force: forceOut,
            stateWords,
            trace: new Float32Array([
                scalar(record, "swing"),
                scalar(record, "traction"),
                scalar(record, "speed"),
                scalar(record, "speedEfficiency"),
            ]),
            header: FA2_STATE.read(view, 0),
        };
    } finally {
        for (const buffer of buffers) {
            buffer.destroy();
        }
    }
}

// ---- the suite

describe("walking skeleton in the browser (spec 11.5, 11.6 item 2)", () => {
    it("uploads the arena and the fromCsr paths and runs degree on karate, random1k and arcCount 0 against the oracle bitwise", async (t) => {
        await requireBrowserGpu(t);
        // four snapshots are deliberately resident at once below; the spec 4.1 warning is for superseded ones
        const ctx = await acquireBrowser({ label: "skeleton/degree", warnUnreleasedSnapshots: 4 });
        const cls = adapterClassBrowser(ctx.caps);
        console.warn(
            `[skeleton] adapter ${cls} subgroups ${ctx.caps.subgroupMinSize}/${ctx.caps.subgroupMaxSize} workgroup ${ctx.workgroupSize} software ${String(ctx.caps.software)}`,
        );
        // the feature context has subgroups iff the adapter offers them: Chromium's CI adapters do (spec 11.6 item 5;
        // SwiftShader, NVIDIA, Metal, WARP), WebKit 26 on the macOS runner does not
        expect(ctx.caps.features.has("subgroups")).toBe(browserAdapterOffersSubgroups());
        if (browserAdapterOffersSubgroups()) {
            expect(ctx.caps.subgroupMaxSize).toBeGreaterThanOrEqual(4);
        }

        const karate = snapshotOf(KARATE_EDGES, { label: "skeleton/karate" });
        const karateCsr = csrSnapshotOf(KARATE_EDGES, { label: "skeleton/karate-csr" });
        const random1k = fixture("random1k", 1).snapshot;
        const empty = snapshotOf([], { nodeCount: 5, label: "skeleton/empty-arcs" });
        expect(random1k.nodeCount).toBe(1000);
        try {
            expect(ctx.residency.core(karate).plan).toBe("arena");
            expect(karateCsr.arena).toBeNull();
            expect(ctx.residency.core(karateCsr).plan).toBe("perArray");
            const emptyCore = ctx.residency.core(empty);
            expect(emptyCore.colIdx).toBeNull();
            expect(emptyCore.weights).toBeNull();

            for (const s of [karate, karateCsr, random1k, empty]) {
                const first = await degree(ctx, s);
                const second = await degree(ctx, s);
                expectBitwiseEqual(first, second, `${s.label ?? "snapshot"} twice`);
                expectBitwiseEqual(first, outDegreeOracle(s), s.label ?? "snapshot");
                expect(first.length).toBe(s.nodeCount);
            }
            const random1kDegrees = await degree(ctx, random1k);
            const path = await writeNoiseFixtureBrowser("degree", "random1k", cls, random1kDegrees, "u32");
            if (path !== "") {
                console.warn(`[skeleton] wrote ${path}`);
            }
        } finally {
            for (const s of [karate, karateCsr, random1k, empty]) {
                ctx.release(s);
            }
        }
        expect(ctx.residency.stats().buffers, "release leaves no buffers (spec 11.6 item 7)").toBe(0);
        expect(ctx.residency.stats().snapshots).toBe(0);
        expect(ctx.pool.idleBytes).toBe(0);
    });

    it("fills the 17M-item map on the 2D dispatch: the sampled words equal i + value and the checksum is the pinned one", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser({ label: "skeleton/linear-id" });
        const cls = adapterClassBrowser(ctx.caps);
        const plan = plan1d(LINEAR_ID_ITEMS, ctx.workgroupSize, ctx.caps);
        expect(plan.y, "16,776,961 items need the 2D dispatch at WG 256 (spec 5.2)").toBeGreaterThan(1);
        expect(plan.items).toBe(LINEAR_ID_ITEMS);
        const dst = createBuffer(ctx, LINEAR_ID_ITEMS * 4, "skeleton/linear-id");
        try {
            const spec = kernelSpec("fill");
            const params = { block: FILL_PARAMS, values: { count: LINEAR_ID_ITEMS, value: LINEAR_ID_VALUE, mode: 1 } };
            const started = performance.now();
            await runKernel(ctx, spec, { dst: whole(dst) }, plan, params);
            const words = await readU32(ctx, dst, LINEAR_ID_ITEMS);
            const elapsedMs = performance.now() - started;
            expect(words.length).toBe(LINEAR_ID_ITEMS);
            const sampled: number[] = [];
            for (const i of LINEAR_ID_SAMPLES) {
                expect(words[i], `word ${i}`).toBe((i + LINEAR_ID_VALUE) % 4294967296);
                sampled.push(words[i]);
            }
            const checksum = linearIdChecksum(words);
            expect(checksum, "bitwise across adapters: the pinned checksum of test/helpers/linear-id.ts").toBe(
                LINEAR_ID_CHECKSUM,
            );
            console.warn(
                `[skeleton] ${cls}: 17M-item fill + ${(LINEAR_ID_ITEMS * 4) / 1048576} MiB readback in ${elapsedMs.toFixed(0)} ms, checksum ${checksum}`,
            );
            // twice bitwise on the boundary words (spec 11.9 item 4): a second fill of the same buffer
            await runKernel(ctx, spec, { dst: whole(dst) }, plan, params);
            const again = await readU32(ctx, dst, LINEAR_ID_ITEMS);
            expect(linearIdChecksum(again)).toBe(checksum);
            for (const i of LINEAR_ID_SAMPLES) {
                expect(again[i]).toBe(words[i]);
            }
            // PLAN DECISION d: the fixture stores the checksum and the sampled words, not 16.7M values
            await writeNoiseFixtureBrowser("fill", "linear-id", cls, [checksum, ...sampled], "u32");
        } finally {
            dst.destroy();
        }
    });

    it("reduces the random1k f32 prefix sums through the subgroup twin against the f64 oracle, twice bitwise, and writes this adapter's fixture", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser({ label: "skeleton/reduce" });
        const cls = adapterClassBrowser(ctx.caps);
        const values = reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED);
        if (!(values instanceof Float32Array)) {
            throw new Error('reduceInput("f32") must return a Float32Array');
        }
        const src = uploadArray(ctx, paddedSumInput(values), "skeleton/reduce-src");
        const liveBefore = ctx.pool.liveBytes;
        try {
            const first = await reduceSums(ctx, values, src);
            const second = await reduceSums(ctx, values, src);
            expectBitwiseEqual(first, second, "the 11 prefix sums twice");
            expect(first.length).toBe(11);
            console.warn(`[skeleton] ${cls}: reduce random1k sums ${Array.from(first).join(", ")}`);
            await writeNoiseFixtureBrowser("reduce", "random1k", cls, first, "f32");
        } finally {
            src.destroy();
        }
        expect(ctx.pool.liveBytes, "every scratch buffer the scope acquired was released").toBe(liveBefore);
        expect(ctx.pool.idleBytes, "the released scratch is held idle by the pool").toBeGreaterThan(0);
    });

    it("runs one K3 + K4 exact-tile iteration on karate (paper mode) from host-written state: force vs the f64 pair sum and the trace vs the hand-computed controller, twice bitwise", async (t) => {
        await requireBrowserGpu(t);
        const ctx = await acquireBrowser({ label: "skeleton/fa2" });
        const cls = adapterClassBrowser(ctx.caps);
        const forceTolerance = toleranceFor("fa2-skeleton.force");
        const traceTolerance = toleranceFor("fa2-skeleton.trace");
        const s = snapshotOf(KARATE_EDGES, { label: "skeleton/fa2-karate" });
        const ref = skeletonReference(s, ctx.workgroupSize);
        const { n } = ref;
        // the recipe reproduces test/layouts/skeleton.test.ts's pinned numbers, so the two adapters' fixtures share one input
        expect(n).toBe(34);
        expectPinned(ref.positions[0], PINS.position0[0], "positions[0].x");
        expectPinned(ref.positions[1], PINS.position0[1], "positions[0].y");
        expect(ref.positions[3]).toBe(PINS.mass0);
        expectPinned(ref.positions[132], PINS.position33[0], "positions[33].x");
        expectPinned(ref.positions[133], PINS.position33[1], "positions[33].y");
        expect(ref.positions[135]).toBe(PINS.mass33);
        const stats = cpuStats(ref.positions);
        for (let k = 0; k < 3; k++) {
            expectPinned(stats.centroid[k], PINS.centroid[k], `centroid[${k}]`);
            expectPinned(ref.force[k], PINS.force0[k], `force[0].${k}`);
            expectPinned(ref.force[99 + k], PINS.force33[k], `force[33].${k}`);
        }
        const guards = guardDistances(ref.positions, stats.centroid);
        expect(guards.pair, "no pair inside the coincident / floor regions").toBeGreaterThan(2 * FA2_DISTANCE_FLOOR);
        expect(guards.centroid).toBeGreaterThan(FA2_DISTANCE_FLOOR);
        expectPinned(guards.pair, PINS.guards.pair, "min pair distance");
        expectPinned(guards.centroid, PINS.guards.centroid, "min distance to the centroid");
        let maxForce = 0;
        let minForce = Infinity;
        for (let i = 0; i < n; i++) {
            const len = Math.hypot(ref.force[3 * i], ref.force[3 * i + 1], ref.force[3 * i + 2]);
            maxForce = Math.max(maxForce, len);
            minForce = Math.min(minForce, len);
        }
        expectPinned(maxForce, PINS.maxForce, "max |F|");
        expect(minForce / maxForce, "no node is floored by the 1e-3 fraction").toBeGreaterThan(FLOOR_FRACTION);
        expectPinned(ref.trace[0], PINS.partial[0], "swing");
        expectPinned(ref.trace[1], PINS.partial[1], "traction");
        expect(ref.trace[0] / ref.trace[1]).toBe(2);
        expectPinned(ref.trace[2], PINS.speed, "speed");
        expectPinned(ref.trace[3], PINS.speedEfficiency, "speedEfficiency");

        try {
            const first = await runPaperLeg(ctx, ref);
            const second = await runPaperLeg(ctx, ref);
            expectBitwiseEqual(first.force, second.force, "K3 force twice");
            expectBitwiseEqual(first.stateWords, second.stateWords, "state (header + trace) twice");
            expect(scalar(first.header, "speed")).toBe(first.trace[2]);
            expect(scalar(first.header, "speedEfficiency")).toBe(first.trace[3]);

            // PLAN DECISION c: per-node vector relative error with the 11.4 floored denominator; its maximum is this
            // adapter's oracle-f64 noise row
            const err = flooredRelError(first.force, ref.force, FLOOR_FRACTION);
            let maxAbs = 0;
            for (let i = 0; i < 3 * n; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(first.force[i] - ref.force[i]));
            }
            const traceErr = maxRelError(first.trace, ref.trace, TRACE_ABS_FLOOR);
            let traceMaxAbs = 0;
            for (let k = 0; k < 4; k++) {
                traceMaxAbs = Math.max(traceMaxAbs, Math.abs(first.trace[k] - ref.trace[k]));
            }
            console.warn(
                `[skeleton] ${cls}: force vs f64 max ${err.max.toExponential(3)} (node ${err.argmax}), rms ${err.rms.toExponential(3)}; trace ${Array.from(first.trace).join(", ")} vs ${ref.trace.join(", ")} -> ${traceErr.toExponential(3)}`,
            );
            expect(
                err.max,
                `K3 force vs the f64 pair sum: worst node ${err.argmax}, rel ${err.max}`,
            ).toBeLessThanOrEqual(forceTolerance);
            expect(traceErr, "K4 trace vs the hand-computed controller").toBeLessThanOrEqual(traceTolerance);
            for (let i = 0; i < n; i++) {
                expect(first.force[3 * i + 2] === 0, `z of node ${i} in 2D (got ${first.force[3 * i + 2]})`).toBe(true);
            }

            await recordNoiseRowBrowser({
                id: `fa2-skeleton.force.oracle-f64.${cls}`,
                kernel: "fa2-repulsion-exact",
                fixture: "karate",
                comparison: "oracle-f64",
                a: cls,
                b: "oracle-f64",
                maxRelError: err.max,
                maxAbsError: maxAbs,
                samples: 3 * n,
            });
            await recordNoiseRowBrowser({
                id: `fa2-skeleton.trace.oracle-f64.${cls}`,
                kernel: "fa2-speed-finalize",
                fixture: "karate",
                comparison: "oracle-f64",
                a: cls,
                b: "oracle-f64",
                maxRelError: traceErr,
                maxAbsError: traceMaxAbs,
                samples: 4,
            });
            await writeNoiseFixtureBrowser("fa2-repulsion-exact", "karate", cls, first.force, "f32");
            await writeNoiseFixtureBrowser("fa2-speed-finalize", "karate", cls, first.trace, "f32");
        } finally {
            ctx.release(s);
        }
        expect(ctx.residency.stats().buffers).toBe(0);
    });

    it("adapterClassBrowser equals the Node adapterClass form on a faked GpuCaps", () => {
        const fake = {
            limits: {} as GPUSupportedLimits,
            features: new Set<string>(),
            wgslFeatures: new Set<string>(),
            subgroupMinSize: 32,
            subgroupMaxSize: 32,
            software: false,
            runtime: "browser",
            vendor: "nvidia",
            architecture: "lovelace",
            device: "",
            description: "NVIDIA GeForce RTX 4070 SUPER",
        } satisfies GpuCaps;
        expect(adapterClassBrowser(fake)).toBe("nvidia-lovelace-browser");
        expect(adapterClassBrowser({ ...fake, runtime: "node", vendor: "mesa", architecture: "software" })).toBe(
            "mesa-software-node",
        );
        expect(adapterClassBrowser({ ...fake, vendor: "google", architecture: "swiftshader" })).toBe(
            "google-swiftshader-browser",
        );
    });
});
