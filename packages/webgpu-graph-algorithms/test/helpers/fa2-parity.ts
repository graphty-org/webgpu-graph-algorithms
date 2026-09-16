/**
 * Shared machinery of the P3-T5 ForceAtlas2 parity tests (spec 11.4, 11.9 items 1, 2 and 4): the scaled parity
 * graphs and the option matrix, a simulation factory that turns inspect() on, the stage-by-stage capture that compares
 * every kernel's intermediate with the f64 oracle's, the trace / force / distributional reports, the state and
 * partials decoders, the noise-fixture names and the P3 tolerance caps. PLAN DECISION 1 (P3-T5): this file is not in
 * contract 1.2; it exists so that test/layouts/fa2-*.test.ts, test/sabotage/fa2.test.ts and test/noise-floor.test.ts
 * evaluate ONE implementation of each parity metric (spec 11.9 item 1: the sabotage test must run the SAME
 * comparison the parity test runs), exactly as P1-T5's degree-check.ts / reduce-check.ts do for their kernels.
 *
 * Units: the owner's array is in SCENE units; the device and the oracle's `positions` are in LAYOUT units (spec
 * 7.18). Every stage capture runs with scale 1 and a zero center so the two coincide bit for bit; the "units" case of
 * the force-parity matrix is the one place the conversion is exercised, through the oracle's own conversion.
 */

import {
    type F32,
    type F64,
    type GraphSnapshot,
    makeMask,
    maskCount,
    maskSet,
    maskTest,
    type NodeMask,
} from "@graphty/graph-format";

import { FA2_DEFAULTS, PARTIAL_BYTES } from "../../src/constants.js";
import type { GpuContext } from "../../src/context.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import type { UniformBlock } from "../../src/kernel/struct-block.js";
import { FA2_PARTIAL, FA2_STATE, type KernelId } from "../../src/kernels.js";
import { ForceSimulation } from "../../src/layouts/force-simulation.js";
import { createForceAtlas2 } from "../../src/layouts/forceatlas2.js";
import type { ForceAtlas2Stats, ForceAtlas2TraceRecord, GpuLayoutTuning } from "../../src/types/layout.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import {
    ForceAtlas2Oracle,
    forceAtlas2Oracle,
    type OracleOptions,
    type OracleResyncState,
    type OracleTraceRecord,
    seededScenePositions,
} from "../oracle/forceatlas2.js";
import {
    completeEdges,
    type EdgeSpec,
    gridEdges,
    KARATE_EDGES,
    pathEdges,
    randomEdges,
    snapshotOf,
    starEdges,
} from "./graphs.js";
import { flooredRelError, maxRelError } from "./matchers.js";
import { layoutMetrics } from "./metrics.js";
import { noiseFloorFor } from "./noise-floor.js";
import { type CheckReport, ratioOf } from "./sabotage.js";

// ---------------------------------------------------------------- constants

/** Spec 11.4 (DEPARTURE-6): the floored denominator max(|F_cpu(i)|, FLOOR_FRACTION x max_j |F_cpu(j)|). */
const FLOOR_FRACTION = 1e-3;
/** The absolute floor of the elementwise scalar comparisons (P1-T6's TRACE_ABS_FLOOR; PLAN DECISION 11). */
const SCALAR_ABS_FLOOR = 1e-6;
/** The absolute floor of the relative comparisons of the non-histogram distributional metrics (PLAN DECISION 11); the histogram bins are compared as one total-variation distance instead (distributionalValuesError). */
export const DISTRIBUTIONAL_FLOOR = 0.05;
/** The reserved fixture class of the f64 reference (P1-T6; test/noise-floor.test.ts). */
export const ORACLE_F64_CLASS = "oracle-f64";
/** The reserved fixture class of the f32 reference (PLAN DECISION 6; test/noise-floor.test.ts). */
export const ORACLE_F32_CLASS = "oracle-f32";
/** The class suffix of the workgroup twin's fixtures (P1-T6; test/noise-floor.test.ts). */
export const TWIN_SUFFIX = "-no-subgroups";
/** The seed of every seeded case. */
const PARITY_SEED = 7;
const RANDOM_GRAPH_SEED = 1234;
/** True in a recording run: every parity test is then held to the spec cap, not to the derived value (PLAN DECISION 4). */
const WRITE = process.env.GRAPHTY_NOISE_FLOOR_WRITE === "1";

/** The options every parity case starts from: layout units = scene units, seeded, settling disabled (settleThreshold 0 -> K1 never increments settledCount unless every node is fixed). */
export const BASE_OPTIONS: ForceAtlas2Options = Object.freeze({
    dim: 2,
    scale: 1,
    center: [0, 0, 0],
    seed: PARITY_SEED,
    gravity: 1,
    scalingRatio: 2,
    jitterTolerance: 1,
    settleThreshold: 0,
    settleWindow: 10,
    maxIter: 1000,
    iterationsPerStep: 1,
    maxInFlight: 2,
});
/** The paper-mode tuning every parity case uses (exact repulsion, spec 7.2 "paper"). */
export const PAPER: GpuLayoutTuning = Object.freeze({ repulsion: "exact", compat: "paper" });
/** The networkx-mode tuning every parity case uses (exact repulsion, spec 7.2 "networkx"). */
export const NETWORKX: GpuLayoutTuning = Object.freeze({ repulsion: "exact", compat: "networkx" });

/** The simulation createForceAtlas2 builds, with its @internal members visible. @public returned by createSim and received by every withSim callback */
export type Fa2Sim = ForceSimulation<ForceAtlas2Options, ForceAtlas2Stats>;

// ---------------------------------------------------------------- the parity graphs and the case matrix

/** The seven parity graphs paritySnapshot() builds. */
export type ParityGraph = "karate" | "grid10" | "star200" | "random1k" | "path10" | "complete6" | "isolated34";
/** The four graphs of spec 11.4's force parity. */
const PARITY_GRAPHS: readonly ParityGraph[] = Object.freeze(["karate", "grid10", "star200", "random1k"]);

/**
 * The edge list and node count of a parity graph at a gpuScale() (fixed-size graphs ignore the scale; the scaled ones
 * keep a floor so a 1/50 scale still yields a graph with structure).
 * @param name - the graph
 * @param scale - gpuScale() (1 on hardware, 1/50 on a software adapter)
 * @returns edges and node count
 */
function parityGraphSpec(name: ParityGraph, scale: number): { readonly edges: EdgeSpec[]; readonly nodeCount: number } {
    switch (name) {
        case "karate":
            return { edges: [...KARATE_EDGES], nodeCount: 34 };
        case "grid10": {
            const side = Math.max(3, Math.round(10 * Math.sqrt(scale)));
            return { edges: gridEdges(side, side), nodeCount: side * side };
        }
        case "star200": {
            const leaves = Math.max(8, Math.round(200 * scale));
            return { edges: starEdges(leaves), nodeCount: leaves + 1 };
        }
        case "random1k": {
            const n = Math.max(20, Math.round(1000 * scale));
            return { edges: randomEdges(n, 3 * n, RANDOM_GRAPH_SEED), nodeCount: n };
        }
        case "path10":
            return { edges: pathEdges(10), nodeCount: 10 };
        case "complete6":
            return { edges: completeEdges(6), nodeCount: 6 };
        case "isolated34":
            return { edges: [], nodeCount: 34 };
        default:
            throw new Error("unknown parity graph");
    }
}

/**
 * The same edges with deterministic, varied weights 0.5 .. 2.0 (k mod 7 steps of 0.25).
 * @param edges - unweighted or weighted edges
 * @returns weighted edges
 */
function weightedCopy(edges: readonly EdgeSpec[]): EdgeSpec[] {
    return edges.map(([u, v], k) => [u, v, 0.5 + (k % 7) * 0.25] as const);
}

/**
 * A snapshot of a parity graph (undirected; weighted through weightedCopy when asked).
 * @param name - the graph
 * @param scale - gpuScale()
 * @param weighted - whether every edge carries a weight
 * @returns the snapshot
 */
export function paritySnapshot(name: ParityGraph, scale: number, weighted: boolean): GraphSnapshot {
    const { edges, nodeCount } = parityGraphSpec(name, scale);
    return snapshotOf(weighted ? weightedCopy(edges) : edges, { nodeCount, label: `${name}${weighted ? "-w" : ""}` });
}

/**
 * The nodeMass vector of the "nodeMass F32" cases: 1 + (i mod 5) x 0.5, so masses differ from degrees and from
 * coordinates.
 * @param n - node count
 * @returns n masses
 */
function massVector(n: number): F32 {
    return Float32Array.from({ length: n }, (_, i) => 1 + (i % 5) * 0.5);
}

/**
 * The pinned node of the "one pinned node" cases: index 3 (a leaf of the star, an interior node of the grid), or
 * the last node when the graph is smaller.
 * @param n - node count
 * @returns the index
 */
export function pinIndex(n: number): number {
    return Math.min(3, n - 1);
}

/**
 * A mask with one bit set.
 * @param n - node count
 * @param index - the pinned node
 * @returns the mask
 */
export function pinMask(n: number, index: number): NodeMask {
    const mask = makeMask(n);
    maskSet(mask, index, true);
    return mask;
}

/** One case of the force-parity matrix. @public the element type of forceParityCases() */
export interface ParityCase {
    readonly name: string;
    readonly graph: ParityGraph;
    readonly weighted: boolean;
    /** nodeMass = massVector(n), materialised by caseOptions() once n is known. */
    readonly massVector: boolean;
    readonly pinned: boolean;
    readonly unseeded: boolean;
    readonly options: ForceAtlas2Options;
    readonly tuning: GpuLayoutTuning;
}

type CaseFlags = Partial<Pick<ParityCase, "weighted" | "massVector" | "pinned" | "unseeded">>;

function mk(
    graph: ParityGraph,
    variant: string,
    options: ForceAtlas2Options,
    flags: CaseFlags = {},
    tuning: GpuLayoutTuning = PAPER,
): ParityCase {
    return {
        name: `${graph}/${variant}`,
        graph,
        weighted: flags.weighted ?? false,
        massVector: flags.massVector ?? false,
        pinned: flags.pinned ?? false,
        unseeded: flags.unseeded ?? false,
        options: { ...BASE_OPTIONS, ...options },
        tuning,
    };
}

/**
 * Bit k of a small non-negative integer, by arithmetic (the house rule keeps bitwise operators away from indices).
 * @param value - the integer
 * @param k - the bit
 * @returns whether the bit is set
 */
function bit(value: number, k: number): boolean {
    return Math.floor(value / 2 ** k) % 2 === 1;
}

/**
 * The force-parity matrix (PLAN DECISION 8): per parity graph the base case, each single variation (weights,
 * linlog, distributedAction, strongGravity, gravity 0, nodeMass F32, 3D, compat networkx, one pinned node) and an
 * all-on case (44); the full 2^5 cross product of the five law switches x {paper, networkx} x {2D, 3D} on karate
 * (128); karate unseeded, karate in scene units (scale 2.5, center (1, -2, 0.5)) and 34 isolated nodes (3).
 * @returns 175 cases with unique names
 */
export function forceParityCases(): readonly ParityCase[] {
    const cases: ParityCase[] = [];
    for (const graph of PARITY_GRAPHS) {
        cases.push(mk(graph, "base", {}));
        cases.push(mk(graph, "weighted", { weight: true }, { weighted: true }));
        cases.push(mk(graph, "linlog", { linlog: true }));
        cases.push(mk(graph, "distributed", { distributedAction: true }));
        cases.push(mk(graph, "strong", { strongGravity: true }));
        cases.push(mk(graph, "gravity0", { gravity: 0 }));
        cases.push(mk(graph, "nodeMass", {}, { massVector: true }));
        cases.push(mk(graph, "3d", { dim: 3 }));
        cases.push(mk(graph, "networkx", {}, {}, NETWORKX));
        cases.push(mk(graph, "pinned", {}, { pinned: true }));
        cases.push(
            mk(
                graph,
                "all-on",
                { weight: true, linlog: true, distributedAction: true, strongGravity: true, dim: 3 },
                { weighted: true, massVector: true, pinned: true },
                NETWORKX,
            ),
        );
    }
    for (let bits = 0; bits < 32; bits++) {
        for (const tuning of [PAPER, NETWORKX]) {
            for (const dim of [2, 3] as const) {
                const weighted = bit(bits, 3);
                cases.push(
                    mk(
                        "karate",
                        `laws-${bits}-${tuning.compat ?? "paper"}-${dim}d`,
                        {
                            linlog: bit(bits, 0),
                            distributedAction: bit(bits, 1),
                            strongGravity: bit(bits, 2),
                            weight: weighted,
                            gravity: bit(bits, 4) ? 0 : 1,
                            dim,
                        },
                        { weighted },
                        tuning,
                    ),
                );
            }
        }
    }
    cases.push(mk("karate", "unseeded", { seed: null }, { unseeded: true }));
    cases.push(mk("karate", "units", { scale: 2.5, center: [1, -2, 0.5] }));
    cases.push(mk("isolated34", "arcCount0", {}));
    return cases;
}

/**
 * The options of a case once the snapshot's node count is known (materialises the nodeMass vector).
 * @param c - the case
 * @param s - its snapshot
 * @returns the options handed to createForceAtlas2 and to the oracle
 */
export function caseOptions(c: ParityCase, s: GraphSnapshot): ForceAtlas2Options {
    return c.massVector ? { ...c.options, nodeMass: massVector(s.nodeCount) } : c.options;
}

/**
 * The owner's start array in scene units: the LCG seed of the options through seededScenePositions (the same start
 * the GPU seeds, spec 11.4 "same start"), or all-NaN for an unseeded case (PLAN DECISION 9: load() seeds it and the
 * oracle reads the array afterwards).
 * @param s - the snapshot
 * @param options - the case options
 * @param unseeded - whether the array is left for load() to seed
 * @returns a fresh stride-3 array
 */
export function startPositions(s: GraphSnapshot, options: ForceAtlas2Options, unseeded: boolean): F32 {
    if (unseeded) {
        const p = new Float32Array(3 * s.nodeCount);
        p.fill(Number.NaN);
        return p;
    }
    return seededScenePositions(s, options.seed ?? null, options.dim ?? 2, options.scale ?? 1, options.center ?? null);
}

/**
 * The oracle options of a GPU configuration: the same option record, the compat of the tuning, the mask, the mass
 * vector when the options carry one, and the snapshot's per-arc weights when `weight` is true (both the option and the
 * resolved array are passed so the oracle's resolution rule sees consistent inputs).
 * @param s - the snapshot
 * @param options - the GPU options
 * @param tuning - the GPU tuning
 * @param mask - the fixed mask or null
 * @param precision - the oracle's scratch precision
 * @returns the OracleOptions
 */
export function oracleOptionsFor(
    s: GraphSnapshot,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    mask: NodeMask | null,
    precision: "f64" | "f32",
): OracleOptions {
    return {
        ...options,
        compat: tuning.compat ?? "paper",
        precision,
        fixed: mask,
        mass: options.nodeMass instanceof Float32Array ? options.nodeMass : null,
        weights: options.weight === true ? s.weights : null,
    };
}

// ---------------------------------------------------------------- simulations with inspect() on

/**
 * A ForceAtlas2 simulation with ctx.debug.inspect set BEFORE construction (P3-T1 binds inspect / debugRunStages at
 * construction), asserted to be the ForceSimulation the contract's createForceAtlas2 builds.
 * @param ctx - the context
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @returns the simulation
 */
export function createSim(ctx: GpuContext, options: ForceAtlas2Options, tuning: GpuLayoutTuning): Fa2Sim {
    ctx.debug.inspect = true;
    const sim = createForceAtlas2(ctx, { ...options, ...tuning });
    if (!(sim instanceof ForceSimulation)) {
        throw new Error("createForceAtlas2 did not return a ForceSimulation");
    }
    return sim as Fa2Sim;
}

/**
 * Runs `fn` with a fresh simulation and disposes it afterwards (also on throw). The caller releases the snapshot.
 * @param ctx - the context
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @param fn - the work
 * @returns whatever fn returns
 */
export async function withSim<T>(
    ctx: GpuContext,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    fn: (sim: Fa2Sim) => Promise<T>,
): Promise<T> {
    const sim = createSim(ctx, options, tuning);
    try {
        return await fn(sim);
    } finally {
        sim.dispose();
    }
}

/** The inspect() / debugRunStages() pair of a simulation. @public returned by debugStages */
export interface StageIo {
    run(upTo: string): Promise<void>;
    read(name: string): Promise<Float32Array | Uint32Array>;
}

/**
 * The inspect() / debugRunStages() pair of a simulation, or a throw when ctx.debug.inspect was not honoured.
 * @param sim - the simulation
 * @returns the pair
 */
export function debugStages(sim: Fa2Sim): StageIo {
    const run = sim.debugRunStages;
    const read = sim.inspect;
    if (run === undefined || read === undefined) {
        throw new Error("ctx.debug.inspect was not honoured: inspect / debugRunStages are absent");
    }
    return {
        run: (upTo: string): Promise<void> => run.call(sim, upTo),
        read: (name: string): Promise<Float32Array | Uint32Array> => read.call(sim, name),
    };
}

/**
 * A readback as an ArrayBuffer-typed Float32Array copy (throws on a Uint32Array readback).
 * @param x - the readback
 * @returns the copy
 */
export function asF32(x: Float32Array | Uint32Array): F32 {
    if (!(x instanceof Float32Array)) {
        throw new Error("expected a Float32Array readback");
    }
    return Float32Array.from(x);
}

/**
 * The xyz lanes of a vec4f array as a stride-3 array.
 * @param vec4 - stride-4 values
 * @param n - node count
 * @returns stride-3 values
 */
export function xyzOf(vec4: ArrayLike<number>, n: number): F32 {
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        out[3 * i] = vec4[4 * i];
        out[3 * i + 1] = vec4[4 * i + 1];
        out[3 * i + 2] = vec4[4 * i + 2];
    }
    return out;
}

function scalarField(block: UniformBlock, view: DataView, field: string, at: number): number {
    const v = block.readField(view, field, at);
    if (typeof v !== "number") {
        throw new Error(`${block.name}.${field}: expected a scalar field`);
    }
    return v;
}

function vectorField(block: UniformBlock, view: DataView, field: string, at: number): readonly number[] {
    const v = block.readField(view, field, at);
    if (typeof v === "number") {
        throw new Error(`${block.name}.${field}: expected a vector field`);
    }
    return v;
}

/** The decoded Fa2State header. @public returned by readState */
export interface StateFields {
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly swing: number;
    readonly traction: number;
    readonly centroid: readonly [number, number, number];
    readonly rmsRadius: number;
    readonly radius: number;
    readonly meanDisplacement: number;
    readonly iteration: number;
    readonly settledCount: number;
}

/**
 * The Fa2State header fields of an inspect("state") readback, decoded through the generated block (3.10.2 offsets).
 * @param raw - the readback
 * @returns the fields
 */
export function readState(raw: Float32Array | Uint32Array): StateFields {
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const centroid = vectorField(FA2_STATE, view, "centroid", 0);
    return {
        speed: scalarField(FA2_STATE, view, "speed", 0),
        speedEfficiency: scalarField(FA2_STATE, view, "speedEfficiency", 0),
        swing: scalarField(FA2_STATE, view, "swing", 0),
        traction: scalarField(FA2_STATE, view, "traction", 0),
        centroid: [centroid[0], centroid[1], centroid[2]],
        rmsRadius: scalarField(FA2_STATE, view, "rmsRadius", 0),
        radius: scalarField(FA2_STATE, view, "radius", 0),
        meanDisplacement: scalarField(FA2_STATE, view, "meanDisplacement", 0),
        iteration: scalarField(FA2_STATE, view, "iteration", 0),
        settledCount: scalarField(FA2_STATE, view, "settledCount", 0),
    };
}

/** The fold of every Fa2Partial record of a readback. */
interface PartialsSum {
    /** sum of positions (xyz) and of |p - c|^2 (w) over every group. */
    readonly sum: readonly [number, number, number, number];
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
    /** max |p - c|^2 (the max.w lane). */
    readonly maxW: number;
    readonly swing: number;
    readonly traction: number;
    readonly disp: number;
    readonly free: number;
}

/**
 * The per-workgroup Fa2Partial records of an inspect("partials") readback, folded exactly as K1 / K4 fold them
 * (sums added, min / max reduced, the free count as an integer).
 * @param raw - the readback (64 bytes per group)
 * @returns the fold
 */
function readPartials(raw: Float32Array | Uint32Array): PartialsSum {
    const groups = Math.floor(raw.byteLength / PARTIAL_BYTES);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const sum = [0, 0, 0, 0];
    const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let maxW = Number.NEGATIVE_INFINITY;
    let swing = 0;
    let traction = 0;
    let disp = 0;
    let free = 0;
    for (let g = 0; g < groups; g++) {
        const at = g * PARTIAL_BYTES;
        const s = vectorField(FA2_PARTIAL, view, "sum", at);
        const lo = vectorField(FA2_PARTIAL, view, "min", at);
        const hi = vectorField(FA2_PARTIAL, view, "max", at);
        const st = vectorField(FA2_PARTIAL, view, "swingTraction", at);
        const df = vectorField(FA2_PARTIAL, view, "dispFree", at);
        for (let k = 0; k < 4; k++) {
            sum[k] += s[k];
        }
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k], lo[k]);
            max[k] = Math.max(max[k], hi[k]);
        }
        maxW = Math.max(maxW, hi[3]);
        swing += st[0];
        traction += st[1];
        disp += df[0];
        free += df[1];
    }
    return {
        sum: [sum[0], sum[1], sum[2], sum[3]],
        min: [min[0], min[1], min[2]],
        max: [max[0], max[1], max[2]],
        maxW,
        swing,
        traction,
        disp,
        free,
    };
}

// ---------------------------------------------------------------- errors, reports and tolerances

/**
 * |actual - expected| / max(|expected|, floor).
 * @param actual - the measured value
 * @param expected - the reference
 * @param floor - the absolute floor of the denominator
 * @returns the relative error (NaN propagates)
 */
export function rel(actual: number, expected: number, floor: number): number {
    return Math.abs(actual - expected) / Math.max(Math.abs(expected), floor);
}

/**
 * The largest |a_i - b_i|.
 * @param a - values
 * @param b - values of the same length
 * @returns the maximum absolute difference (0 for empty inputs)
 */
export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let worst = 0;
    for (let i = 0; i < a.length; i++) {
        worst = Math.max(worst, Math.abs(a[i] - b[i]));
    }
    return worst;
}

/** The P3 tolerance ids, their spec caps and basis rows (PLAN DECISIONS 4 - 6); spread into test/noise-floor.test.ts's TOLERANCE_CAPS. */
export const P3_TOLERANCE_CAPS: Readonly<Record<string, { readonly cap: number; readonly basis: string }>> =
    Object.freeze({
        "fa2-force-parity": { cap: 1e-4, basis: "fa2-force-parity.oracle-f64" },
        "fa2-force-parity.cross": { cap: 1e-4, basis: "fa2-force-parity.cross" },
        "fa2-force-sum": { cap: 1e-4, basis: "fa2-force-parity.oracle-f64" },
        "fa2-inspect.attraction": { cap: 1e-4, basis: "fa2-inspect.attraction.oracle-f64" },
        "fa2-inspect.attraction.cross": { cap: 1e-4, basis: "fa2-inspect.attraction.cross" },
        "fa2-inspect.epilogue": { cap: 1e-4, basis: "fa2-inspect.epilogue.oracle-f64" },
        "fa2-inspect.epilogue.cross": { cap: 1e-4, basis: "fa2-inspect.epilogue.cross" },
        "fa2-inspect.state": { cap: 1e-4, basis: "fa2-inspect.state.oracle-f64" },
        "fa2-inspect.state.cross": { cap: 1e-4, basis: "fa2-inspect.state.cross" },
        "fa2-inspect.positions": { cap: 1e-4, basis: "fa2-inspect.positions.oracle-f64" },
        "fa2-inspect.positions.cross": { cap: 1e-4, basis: "fa2-inspect.positions.cross" },
        "fa2-inspect.partials": { cap: 1e-4, basis: "fa2-inspect.partials.oracle-f64" },
        "fa2-inspect.partials.cross": { cap: 1e-4, basis: "fa2-inspect.partials.cross" },
        "fa2-inspect.scene": { cap: 1e-4, basis: "fa2-inspect.scene.oracle-f64" },
        "fa2-inspect.scene.cross": { cap: 1e-4, basis: "fa2-inspect.scene.cross" },
        "fa2-inspect.k1": { cap: 1e-4, basis: "fa2-inspect.k1.oracle-f64" },
        "fa2-inspect.k1.cross": { cap: 1e-4, basis: "fa2-inspect.k1.cross" },
        "fa2-trace-parity.f32": { cap: 1e-4, basis: "fa2-trace-parity.oracle-f32" },
        // PLAN DECISION 17: the re-synchronised legs (one GPU iteration against one oracle iteration from the GPU's
        // own iteration-start state) are held to the f32 cap of spec 11.4 against BOTH oracles: the f64 oracle differs
        // from the GPU by one iteration's f32 rounding only, never by the chaotic amplification of the trajectory
        "fa2-trace-parity.resync.f32": { cap: 1e-4, basis: "fa2-trace-parity.resync.oracle-f32" },
        "fa2-trace-parity.resync.f64": { cap: 1e-4, basis: "fa2-trace-parity.resync.oracle-f64" },
        // PLAN DECISION 16: the cross-adapter caps of the chaotic-horizon members are TWICE the oracle cap (triangle
        // inequality: two adapters each within the oracle cap of the reference may differ by up to 2x that cap)
        "fa2-trace-parity.cross10": { cap: 2e-4, basis: "fa2-trace-parity.cross10" },
        "fa2-trace-parity.f64": { cap: 5e-2, basis: "fa2-trace-parity.oracle-f64" },
        "fa2-trace-parity.cross50": { cap: 1e-1, basis: "fa2-trace-parity.cross50" },
        "fa2-distributional": { cap: 0.1, basis: "fa2-distributional.oracle-f64" },
        "fa2-distributional.cross": { cap: 0.2, basis: "fa2-distributional.cross" },
        "fa2-twins.force": { cap: 1e-6, basis: "fa2-twins.force.twin" },
        "fa2-twins.trace": { cap: 1e-6, basis: "fa2-twins.trace.twin" },
        // the per-node vector twin comparisons (K5 positions, toScene scene: the floored stride-3 metric) have their own
        // measured floor -- the maximum over the positions / scene twin members -- so they are never held to a scalar floor
        "fa2-twins.positions": { cap: 1e-6, basis: "fa2-twins.positions.twin" },
    });

/**
 * The tolerance a parity test uses: the derived value of benchmarks/results/noise-floor.json (noiseFloorFor), or the
 * spec cap during a recording run (GRAPHTY_NOISE_FLOOR_WRITE=1), when the derived values do not exist yet.
 * @param id - a P3 tolerance id
 * @returns the tolerance
 */
export function toleranceOf(id: string): number {
    if (WRITE) {
        const spec = P3_TOLERANCE_CAPS[id];
        if (spec === undefined) {
            throw new Error(`${id}: not a P3 tolerance id`);
        }
        return spec.cap;
    }
    return noiseFloorFor(id).value;
}

// ---------------------------------------------------------------- the stage capture (spec 11.9 item 2)

/** The eight inspect() stages in model order (K2, K3, its epilogue, K4, K5, its partials, toScene, the K1 fold). */
export type StageKey = "attraction" | "force" | "epilogue" | "state" | "positions" | "partials" | "scene" | "k1";
/** The eight stage keys as an iteration list, in model order. */
export const STAGE_KEYS: readonly StageKey[] = Object.freeze([
    "attraction",
    "force",
    "epilogue",
    "state",
    "positions",
    "partials",
    "scene",
    "k1",
]);
/** The model stage debugRunStages() runs up to for each key. */
const STAGE_UP_TO: Readonly<Record<StageKey, string>> = Object.freeze({
    attraction: "K2",
    force: "K3",
    epilogue: "K3",
    state: "K4",
    positions: "K5",
    partials: "K5",
    scene: "toScene",
    k1: "K1",
});
/** The kernel each stage's values come from (the sabotage id and the noise fixture's kernel). */
export const STAGE_KERNEL: Readonly<Record<StageKey, KernelId>> = Object.freeze({
    attraction: "fa2-attraction",
    force: "fa2-repulsion-exact",
    epilogue: "fa2-repulsion-exact",
    state: "fa2-speed-finalize",
    positions: "fa2-integrate",
    partials: "fa2-integrate",
    scene: "fa2-to-scene",
    k1: "fa2-stats-finalize",
});
/** The traced tolerance id each stage comparison is held to. */
export const STAGE_TOLERANCE: Readonly<Record<StageKey, string>> = Object.freeze({
    attraction: "fa2-inspect.attraction",
    force: "fa2-force-parity",
    epilogue: "fa2-inspect.epilogue",
    state: "fa2-inspect.state",
    positions: "fa2-inspect.positions",
    partials: "fa2-inspect.partials",
    scene: "fa2-inspect.scene",
    k1: "fa2-inspect.k1",
});

/** A (kernel, fixture) pair of test/fixtures/noise. @public the value type of NOISE_FIXTURES */
export interface NoiseFixtureName {
    readonly kernel: KernelId;
    readonly fixture: string;
}
/**
 * The (kernel, fixture) names of the P3 noise members (PLAN DECISION 7: the stage members, the free-running trace
 * members and the metrics member are recorded on the UNSCALED random1k; PLAN DECISION 17: the two re-synchronised
 * members on the unscaled random1k AND on karate, the worst-conditioned trace fixture -- its per-iteration swing
 * cancellation gives the largest re-synchronised floor, and a tolerance derived from random1k alone would not cover
 * it). `trace10` / `trace50` are the free-running traces of the NETWORKX mode (noiseInputs(NETWORKX)): the free-running
 * legs of fa2-trace-parity.test.ts are asserted in that mode only, the paper-mode trajectory being chaotic beyond
 * any derivable tolerance (docs/decisions/G3.md finding G3-F3), and `trace50Karate` is the same 50-iteration networkx
 * trace on karate, the fixture whose free-running divergence through 50 comes closest to the 5e-2 cap (its row makes
 * the recorded factor of fa2-trace-parity.f64 honest); `resync50` / `resyncKarate50` are the paper-mode
 * re-synchronised traces (resyncTrace over TRACE_ITERATIONS), whose oracle fixtures are per adapter class;
 * `states10` is K4's record of ONE iteration from each of the ten trajectory states of oracleStates() in each mode
 * (the re-synchronised twin comparison of fa2-twins.test.ts: twenty geometries through the same reductions).
 */
export const NOISE_FIXTURES: Readonly<
    Record<
        StageKey | "trace10" | "trace50" | "trace50Karate" | "resync50" | "resyncKarate50" | "states10" | "metrics100",
        NoiseFixtureName
    >
> = Object.freeze({
    attraction: { kernel: "fa2-attraction", fixture: "random1k-K2" },
    force: { kernel: "fa2-repulsion-exact", fixture: "random1k-K3" },
    epilogue: { kernel: "fa2-repulsion-exact", fixture: "random1k-K3-epilogue" },
    state: { kernel: "fa2-speed-finalize", fixture: "random1k-K4" },
    positions: { kernel: "fa2-integrate", fixture: "random1k-K5" },
    partials: { kernel: "fa2-integrate", fixture: "random1k-K5-partials" },
    scene: { kernel: "fa2-to-scene", fixture: "random1k-toScene" },
    k1: { kernel: "fa2-stats-finalize", fixture: "random1k-K1" },
    trace10: { kernel: "fa2-speed-finalize", fixture: "random1k-trace10" },
    trace50: { kernel: "fa2-speed-finalize", fixture: "random1k-trace50" },
    trace50Karate: { kernel: "fa2-speed-finalize", fixture: "karate-trace50" },
    resync50: { kernel: "fa2-speed-finalize", fixture: "random1k-resync50" },
    resyncKarate50: { kernel: "fa2-speed-finalize", fixture: "karate-resync50" },
    states10: { kernel: "fa2-speed-finalize", fixture: "random1k-states10" },
    metrics100: { kernel: "fa2-integrate", fixture: "random1k-metrics100" },
});

/**
 * The inputs a noise member is recorded from: the UNSCALED graph (random1k by default; karate for the karate
 * member), the base options, and the tuning (paper by default; the free-running trace members pass NETWORKX).
 * @param tuning - the compat mode of the member (default PAPER)
 * @param graph - the member's graph (default "random1k")
 * @returns snapshot, start array, options and tuning
 */
export function noiseInputs(
    tuning: GpuLayoutTuning = PAPER,
    graph: ParityGraph = "random1k",
): {
    readonly s: GraphSnapshot;
    readonly start: F32;
    readonly options: ForceAtlas2Options;
    readonly tuning: GpuLayoutTuning;
} {
    const s = paritySnapshot(graph, 1, false);
    return { s, start: startPositions(s, BASE_OPTIONS, false), options: BASE_OPTIONS, tuning };
}

/**
 * The first `count` iteration-start states of the f64 oracle's free-running trajectory (positions after 0 .. count - 1
 * steps, f32-rounded, scene = layout units): adapter-independent states for the multi-state twin comparison of
 * fa2-twins.test.ts and its states10 noise member (PLAN DECISION 17). Requires scale 1 and a zero centre.
 * @param s - the snapshot
 * @param start - the scene start
 * @param options - the FA2 options
 * @param tuning - the GPU tuning (the mode)
 * @param count - the number of states
 * @returns count stride-3 arrays
 */
export function oracleStates(
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    count: number,
): F32[] {
    assertUnitStart(options);
    const oracle = new ForceAtlas2Oracle(
        s,
        layoutStart(start, s.nodeCount, options.dim ?? 2),
        oracleOptionsFor(s, options, tuning, null, "f64"),
    );
    const states: F32[] = [];
    for (let k = 0; k < count; k++) {
        if (k > 0) {
            oracle.step();
        }
        states.push(Float32Array.from(oracle.positions));
    }
    return states;
}

/**
 * The trace record of one iteration of the f64 oracle from a state under load() semantics (a fresh reference:
 * controller 1 / 1, no previous force), as traceValues: the oracle-f64 fixture of the states10 member.
 * @param s - the snapshot
 * @param state - the iteration-start positions
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @returns the six record values
 */
export function oracleRecordFrom(
    s: GraphSnapshot,
    state: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
): F64 {
    const oracle = new ForceAtlas2Oracle(
        s,
        layoutStart(state, s.nodeCount, options.dim ?? 2),
        oracleOptionsFor(s, options, tuning, null, "f64"),
    );
    return traceValues([oracle.step()], 0, 1);
}

/**
 * The fixture class of an oracle reference re-synchronised to ONE adapter's states (the resync members: the oracle's
 * values depend on the adapter's own iteration-start states, so each adapter class carries its own oracle fixture).
 * @param oracleClass - ORACLE_F32_CLASS or ORACLE_F64_CLASS
 * @param adapterClass - the adapter (or twin) class the oracle followed
 * @returns "<oracle class>/<adapter class>"
 */
export function resyncOracleClass(oracleClass: string, adapterClass: string): string {
    return `${oracleClass}/${adapterClass}`;
}

/** One stage of a capture. @public the value type of StageCapture */
export interface StageResult {
    readonly key: StageKey;
    /** Vector stages are compared per node with the floored stride-3 metric, scalar stages elementwise (PLAN DECISION 11). */
    readonly vector: boolean;
    /** This adapter's raw output (f32 values widened). */
    readonly values: F64;
    /** The oracle's expectation in the same layout. */
    readonly expected: F64;
    /** stageError(vector, values, expected).rel */
    readonly error: number;
    readonly maxAbs: number;
}
/** Every stage of one capture. @public returned by captureAllStages */
export type StageCapture = Readonly<Record<StageKey, StageResult>>;

/**
 * The error of two outputs of one stage: the floored per-node metric of spec 11.4 for vector stages, the
 * 1e-6-floored elementwise relative error for scalar stages.
 * @param vector - whether the values are stride-3 per-node vectors
 * @param a - the measured values
 * @param b - the reference values
 * @returns the relative and absolute maximum errors
 */
export function stageError(
    vector: boolean,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
): { readonly rel: number; readonly abs: number } {
    if (a.length !== b.length) {
        return { rel: Number.POSITIVE_INFINITY, abs: Number.POSITIVE_INFINITY };
    }
    if (vector) {
        return { rel: flooredRelError(a, b, FLOOR_FRACTION).max, abs: maxAbsDiff(a, b) };
    }
    return { rel: maxRelError(a, b, SCALAR_ABS_FLOOR), abs: maxAbsDiff(a, b) };
}

function stageResult(
    key: StageKey,
    vector: boolean,
    values: ArrayLike<number>,
    expected: ArrayLike<number>,
): StageResult {
    const v = Float64Array.from(values);
    const e = Float64Array.from(expected);
    const err = stageError(vector, v, e);
    return { key, vector, values: v, expected: e, error: err.rel, maxAbs: err.abs };
}

/**
 * Asserts the unit-identity precondition of a stage capture (scale 1, zero center).
 * @param options - the case options
 */
function assertUnitStart(options: ForceAtlas2Options): void {
    const center = options.center ?? [0, 0, 0];
    const zero = center.length <= 3 && Array.from({ length: center.length }, (_, k) => center[k]).every((v) => v === 0);
    if ((options.scale ?? 1) !== 1 || !zero) {
        throw new Error("captureAllStages needs scale 1 and a zero center (layout units = scene units)");
    }
}

/**
 * The scene start as f64 layout units (scale 1, zero center: the values themselves; z forced to 0 in 2D as load() does).
 * @param start - the scene start
 * @param n - node count
 * @param dim - 2 or 3
 * @returns stride-3 f64 layout positions
 */
function layoutStart(start: F32, n: number, dim: 2 | 3): F64 {
    const out = Float64Array.from(start);
    if (dim === 2) {
        for (let i = 0; i < n; i++) {
            out[3 * i + 2] = 0;
        }
    }
    return out;
}

/**
 * Runs one iteration of every stage on FRESH simulations (PLAN DECISION 10) and compares each intermediate with
 * the f64 oracle's: K2's attraction, K3's force and epilogue partials, K4's controller fields, K5's positions and
 * partials A / C, toScene's scene positions, and K1's fold of iteration 2 (after one real step(1)).
 * @param ctx - the context (inspect is switched on)
 * @param s - the snapshot
 * @param start - the scene start (scale 1, zero center)
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @param mask - the fixed mask or null
 * @returns one result per stage
 */
export async function captureAllStages(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    mask: NodeMask | null,
): Promise<StageCapture> {
    assertUnitStart(options);
    const n = s.nodeCount;
    const dim = options.dim ?? 2;
    const before = layoutStart(start, n, dim);
    const oracleRun = forceAtlas2Oracle(
        s,
        Float32Array.from(start),
        oracleOptionsFor(s, options, tuning, mask, "f64"),
        1,
    );
    const after = Float64Array.from(oracleRun.oracle.positions);
    const { stages } = oracleRun.oracle;
    const record0 = oracleRun.trace[0];
    // the K5 partials A / C of iteration 1 and the K1 fold of iteration 2, from the positions before / after (PLAN DECISION 10)
    const c0 = [0, 0, 0];
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) {
            c0[k] += before[3 * i + k] / n;
        }
    }
    const sum = [0, 0, 0];
    const lo = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const hi = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let sumSq = 0;
    let maxW = 0;
    let disp = 0;
    let free = 0;
    for (let i = 0; i < n; i++) {
        let q2 = 0;
        let dp2 = 0;
        for (let k = 0; k < 3; k++) {
            const p = after[3 * i + k];
            sum[k] += p;
            lo[k] = Math.min(lo[k], p);
            hi[k] = Math.max(hi[k], p);
            const q = p - c0[k];
            q2 += q * q;
            const d = p - before[3 * i + k];
            dp2 += d * d;
        }
        sumSq += q2;
        maxW = Math.max(maxW, q2);
        if (mask === null || !maskTest(mask, i)) {
            free++;
            disp += Math.sqrt(dp2);
        }
    }
    const rms = Math.sqrt(sumSq / n);
    const radius = Math.sqrt(maxW);
    const meanDisp = free === 0 ? 0 : disp / free;
    const settled = meanDisp <= (options.settleThreshold ?? FA2_DEFAULTS.settleThreshold) * rms ? 1 : 0;
    if (mask !== null && maskCount(mask, n) !== n - free) {
        throw new Error("free count mismatch");
    }

    const run = <T>(upTo: string, afterFirstStep: boolean, read: (st: StageIo) => Promise<T>): Promise<T> =>
        withSim(ctx, options, tuning, async (sim) => {
            sim.load(s, Float32Array.from(start));
            if (mask !== null) {
                sim.setFixed(mask);
            }
            if (afterFirstStep) {
                await sim.step(1);
            }
            const st = debugStages(sim);
            await st.run(upTo);
            return read(st);
        });

    const attraction = await run("K2", false, async (st) => asF32(await st.read("force")));
    const k3 = await run("K3", false, async (st) => ({
        force: asF32(await st.read("force")),
        partials: readPartials(await st.read("partials")),
    }));
    const state4 = await run("K4", false, async (st) => readState(await st.read("state")));
    const k5 = await run("K5", false, async (st) => ({
        positions: xyzOf(asF32(await st.read("positions")), n),
        partials: readPartials(await st.read("partials")),
    }));
    const scene = await run("toScene", false, async (st) => asF32(await st.read("scenePositions")));
    const state1 = await run("K1", true, async (st) => readState(await st.read("state")));

    const p = k5.partials;
    return {
        attraction: stageResult("attraction", true, attraction, stages.attraction),
        force: stageResult("force", true, k3.force, stages.force),
        epilogue: stageResult(
            "epilogue",
            false,
            [k3.partials.swing, k3.partials.traction],
            [stages.partials.swing, stages.partials.traction],
        ),
        state: stageResult(
            "state",
            false,
            [state4.speed, state4.speedEfficiency, state4.swing, state4.traction],
            [record0.speed, record0.speedEfficiency, record0.swing, record0.traction],
        ),
        positions: stageResult("positions", true, k5.positions, after),
        partials: stageResult(
            "partials",
            false,
            [
                p.sum[0],
                p.sum[1],
                p.sum[2],
                p.sum[3],
                p.min[0],
                p.min[1],
                p.min[2],
                p.max[0],
                p.max[1],
                p.max[2],
                p.maxW,
                p.disp,
                p.free,
            ],
            [sum[0], sum[1], sum[2], sumSq, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], maxW, disp, free],
        ),
        scene: stageResult("scene", true, scene, oracleRun.positions),
        k1: stageResult(
            "k1",
            false,
            [
                state1.centroid[0],
                state1.centroid[1],
                state1.centroid[2],
                state1.rmsRadius,
                state1.radius,
                state1.meanDisplacement,
                state1.settledCount,
                state1.iteration,
            ],
            [sum[0] / n, sum[1] / n, sum[2] / n, rms, radius, meanDisp, settled, 2],
        ),
    };
}

/**
 * The check report of one stage: its error over its traced tolerance.
 * @param capture - a stage capture
 * @param key - the stage
 * @returns the report (worst = error / tolerance)
 */
export function stageReport(capture: StageCapture, key: StageKey): CheckReport {
    const r = capture[key];
    return {
        worst: ratioOf(r.error, toleranceOf(STAGE_TOLERANCE[key])),
        worstLabel: `${key} (${STAGE_UP_TO[key]})`,
        samples: r.values.length,
    };
}

// ---------------------------------------------------------------- the trace (spec 11.4 trace parity)

/**
 * The six fields of records [from, to) as one flat array (the noise fixture layout of the trace members).
 * @param records - trace records
 * @param from - first index
 * @param to - one past the last index
 * @returns 6 x (to - from) values
 */
export function traceValues(records: readonly ForceAtlas2TraceRecord[], from: number, to: number): F64 {
    const out = new Float64Array(Math.max(0, 6 * (to - from)));
    for (let k = from; k < to; k++) {
        const r = records[k];
        const at = 6 * (k - from);
        out[at] = r.swing;
        out[at + 1] = r.traction;
        out[at + 2] = r.speed;
        out[at + 3] = r.speedEfficiency;
        out[at + 4] = r.meanDisplacement;
        out[at + 5] = r.settledCount;
    }
    return out;
}

/**
 * The worst relative error over records [from, to) of the GPU trace against the oracle's: swing, traction, speed and
 * speedEfficiency relative (floor 1e-9), meanDisplacement relative with a 1e-6 floor, settledCount against a floor
 * of 1 (an integer count). A missing record on either side is Infinity (never a skip).
 * @param gpu - the GPU records (one per step(1))
 * @param oracle - the oracle's records (one per step())
 * @param from - first index
 * @param to - one past the last index
 * @returns the worst error
 */
export function traceError(
    gpu: readonly ForceAtlas2TraceRecord[],
    oracle: readonly OracleTraceRecord[],
    from: number,
    to: number,
): number {
    let worst = 0;
    for (let k = from; k < to; k++) {
        const g: ForceAtlas2TraceRecord | undefined = gpu[k];
        const o: OracleTraceRecord | undefined = oracle[k];
        if (g === undefined || o === undefined) {
            return Number.POSITIVE_INFINITY;
        }
        worst = Math.max(
            worst,
            rel(g.swing, o.swing, 1e-9),
            rel(g.traction, o.traction, 1e-9),
            rel(g.speed, o.speed, 1e-9),
            rel(g.speedEfficiency, o.speedEfficiency, 1e-9),
            rel(g.meanDisplacement, o.meanDisplacement, SCALAR_ABS_FLOOR),
            rel(g.settledCount, o.settledCount, 1),
        );
    }
    return worst;
}

/**
 * The worst relative error of the K1 fold fields the stats carry (centroid, rmsRadius, layoutRadius) against the
 * oracle's records, index-aligned (stats after the k-th step(1) vs the k-th oracle record: both are the fold
 * written at the START of that iteration). The centroid is compared with the record's OWN rmsRadius as the floor of
 * the denominator (P3-T5 implementation note, a deviation from the plan part's fixed 1e-2 floor): the centroid is a
 * coordinate-like quantity that stays near the origin while the layout expands to a radius of 1e2, so a fixed floor
 * turns a 0.9-unit difference in a layout of radius 184 (grid10 / networkx at iteration 28, measured on NVIDIA) into
 * a relative error of 81 while rmsRadius and layoutRadius differ by 0.6%; normalised by the layout's own scale the
 * same centroid difference reads as 5e-3, the same order as the other two fields.
 * @param stats - per-step stats
 * @param oracle - the oracle's records
 * @param from - first index
 * @param to - one past the last index
 * @returns the worst error
 */
export function statsFoldError(
    stats: readonly {
        readonly centroid: readonly [number, number, number];
        readonly rmsRadius: number;
        readonly layoutRadius: number;
    }[],
    oracle: readonly OracleTraceRecord[],
    from: number,
    to: number,
): number {
    let worst = 0;
    for (let k = from; k < to; k++) {
        const g = stats[k];
        const o: OracleTraceRecord | undefined = oracle[k];
        if (g === undefined || o === undefined) {
            return Number.POSITIVE_INFINITY;
        }
        const floor = Math.max(o.rmsRadius, SCALAR_ABS_FLOOR);
        worst = Math.max(
            worst,
            rel(g.centroid[0], o.centroid[0], floor),
            rel(g.centroid[1], o.centroid[1], floor),
            rel(g.centroid[2], o.centroid[2], floor),
            rel(g.rmsRadius, o.rmsRadius, 1e-9),
            rel(g.layoutRadius, o.layoutRadius, 1e-9),
        );
    }
    return worst;
}

/** The free-running part of a resyncTrace run (the GPU's own trajectory). @public the base of ResyncRun */
export interface TraceRun {
    /** One record per step(1). */
    readonly trace: ForceAtlas2TraceRecord[];
    /** The K1 fold fields of stats after each step(1). */
    readonly stats: {
        readonly centroid: readonly [number, number, number];
        readonly rmsRadius: number;
        readonly layoutRadius: number;
    }[];
    /** The owner's array after the last step (scene units). */
    readonly positions: F32;
}

// ---------------------------------------------------------------- the re-synchronised trace (spec 11.4 trace parity; PLAN DECISION 17)

/** The step(1) count of the trace tests and of their re-synchronised legs (spec 11.4: 50 iterations). */
export const TRACE_ITERATIONS = 50;
/** The horizon of the tight free-running leg of spec 11.4 (the first 10 records against the f32 oracle). */
export const TRACE_TIGHT = 10;
/** The values of one re-synchronised record (resyncValues): [swing, traction, speed, speedEfficiency, meanDisplacement, settledCount, centroid x, y, z, rmsRadius]. */
const RESYNC_FIELDS = 10;

/**
 * One iteration of a re-synchronised comparison: K4's four controller fields of the iteration and K1's statistics
 * written at ITS start (the fold of the previous integrate; load()'s statistics for the first) -- every one a SUM
 * over the nodes. K1's layoutRadius (max |p - c|) is deliberately absent: it is an extreme-value statistic, and in
 * paper mode one node whose consecutive forces nearly cancel carries a per-node displacement error of up to 1e-3
 * relative (the speed factor speed / (1 + sqrt(speed m |F(t) - F(t-1)|)) amplifies f32 force rounding; measured
 * 7.5e-4 on random1k at iteration 3 against the f64 oracle) that the max inherits (2.7e-4 measured) while every
 * sum stays under 1e-5; the radius is checked by the K1 stage of fa2-inspect.test.ts and by the free-running fold
 * leg of fa2-trace-parity.test.ts instead.
 */
interface ResyncRecord {
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly meanDisplacement: number;
    readonly settledCount: number;
    readonly centroid: readonly [number, number, number];
    readonly rmsRadius: number;
}

/**
 * The records as one flat array of RESYNC_FIELDS values each (the noise fixture layout of the resync members).
 * @param records - re-synchronised records
 * @returns RESYNC_FIELDS x records.length values
 */
export function resyncValues(records: readonly ResyncRecord[]): F64 {
    const out = new Float64Array(RESYNC_FIELDS * records.length);
    records.forEach((r, k) => {
        const at = RESYNC_FIELDS * k;
        out[at] = r.swing;
        out[at + 1] = r.traction;
        out[at + 2] = r.speed;
        out[at + 3] = r.speedEfficiency;
        out[at + 4] = r.meanDisplacement;
        out[at + 5] = r.settledCount;
        out[at + 6] = r.centroid[0];
        out[at + 7] = r.centroid[1];
        out[at + 8] = r.centroid[2];
        out[at + 9] = r.rmsRadius;
    });
    return out;
}

/**
 * The worst relative error of two resyncValues lists, record by record: swing, traction, speed, speedEfficiency
 * and rmsRadius relative (floor 1e-9), meanDisplacement relative with the 1e-6 floor, settledCount against a floor
 * of 1 (an integer count), and each centroid component against the reference record's OWN rmsRadius as the floor
 * (the centroid is a coordinate-like quantity near the origin of a layout of radius 1e2; the statsFoldError note).
 * The ONE implementation of the metric (spec 11.9 item 1): fa2-trace-parity.test.ts, fa2-twins.test.ts,
 * fa2-properties.test.ts, test/sabotage/fa2.test.ts and the resync members of test/noise-floor.test.ts all call it.
 * Lists of different lengths, or a length that is not a multiple of RESYNC_FIELDS, are Infinity.
 * @param a - the measured values
 * @param b - the reference values
 * @returns the worst error
 */
export function resyncValuesError(a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (a.length !== b.length || a.length % RESYNC_FIELDS !== 0) {
        return Number.POSITIVE_INFINITY;
    }
    let worst = 0;
    for (let at = 0; at < a.length; at += RESYNC_FIELDS) {
        const floor = Math.max(b[at + 9], SCALAR_ABS_FLOOR);
        worst = Math.max(
            worst,
            rel(a[at], b[at], 1e-9),
            rel(a[at + 1], b[at + 1], 1e-9),
            rel(a[at + 2], b[at + 2], 1e-9),
            rel(a[at + 3], b[at + 3], 1e-9),
            rel(a[at + 4], b[at + 4], SCALAR_ABS_FLOOR),
            rel(a[at + 5], b[at + 5], 1),
            rel(a[at + 6], b[at + 6], floor),
            rel(a[at + 7], b[at + 7], floor),
            rel(a[at + 8], b[at + 8], floor),
            rel(a[at + 9], b[at + 9], 1e-9),
        );
    }
    return worst;
}

/** The outputs of resyncTrace. @public returned by resyncTrace */
export interface ResyncRun extends TraceRun {
    /** The GPU's records: K4's fields of record t and K1's statistics of iteration t (the stats after step t). */
    readonly gpu: ResyncRecord[];
    /** The f32 oracle re-synchronised before every iteration: its K4 fields of the iteration and its fold of the previous one. */
    readonly f32: ResyncRecord[];
    /** The same with the f64 oracle. */
    readonly f64: ResyncRecord[];
    /** Informational: the floored per-node error of the GPU's positions after iteration t against each oracle's after its one step. */
    readonly positionError: { readonly f32: number[]; readonly f64: number[] };
}

/** Optional hooks of resyncTrace. @public the last parameter of resyncTrace */
export interface ResyncHooks {
    /** A fixed mask set right after load() (also given to every oracle). */
    readonly mask?: NodeMask | null | undefined;
    /** Called before step t on the simulation (a setPosition / setFixed the next oracle then sees through the GPU's state). */
    readonly beforeStep?: ((sim: Fa2Sim, t: number) => void) | undefined;
}

/**
 * One re-synchronised oracle of precision `precision`: a fresh oracle over the GPU's iteration-start positions,
 * seeded (for t > 0) with the GPU's controller after the previous iteration, K1's statistics of this iteration and
 * the previous oracle's force as F(t-1) (the oracle chains its OWN oldForce: a K5 that forgets to store it stays
 * visible, PLAN DECISION 17).
 */
function resyncOracle(
    s: GraphSnapshot,
    positions: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    mask: NodeMask | null,
    precision: "f32" | "f64",
    seed: OracleResyncState | null,
): ForceAtlas2Oracle {
    const oracle = new ForceAtlas2Oracle(
        s,
        layoutStart(positions, s.nodeCount, options.dim ?? 2),
        oracleOptionsFor(s, options, tuning, mask, precision),
    );
    if (seed !== null) {
        oracle.resync(seed);
    }
    return oracle;
}

/**
 * `iterations` x step(1) on a fresh simulation, and before EVERY iteration a fresh f32 and a fresh f64 oracle
 * seeded with the GPU's iteration-start state (PLAN DECISION 17: the positions the previous step left in the
 * owner's array, the controller the stats report, K1's statistics of the iteration, the oracle's own previous
 * force as F(t-1)), each stepped once: the comparison of one GPU iteration with one oracle iteration, free of the
 * chaotic amplification of the free-running trajectory (spec 7.16). Requires scale 1 and a zero centre (layout
 * units = scene units, so the owner's array IS the layout state).
 * @param ctx - the context
 * @param s - the snapshot
 * @param start - the scene start (copied)
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @param iterations - number of step(1) calls
 * @param hooks - a fixed mask and / or a per-iteration hook
 * @returns the run
 */
export async function resyncTrace(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    iterations: number,
    hooks: ResyncHooks = {},
): Promise<ResyncRun> {
    assertUnitStart(options);
    const mask = hooks.mask ?? null;
    return await withSim(ctx, options, tuning, async (sim) => {
        const positions = Float32Array.from(start);
        sim.load(s, positions);
        if (mask !== null) {
            sim.setFixed(mask);
        }
        const trace: ForceAtlas2TraceRecord[] = [];
        const stats: TraceRun["stats"] = [];
        const gpu: ResyncRecord[] = [];
        const f32: ResyncRecord[] = [];
        const f64: ResyncRecord[] = [];
        const positionError = { f32: [] as number[], f64: [] as number[] };
        let previous: { readonly f32: ForceAtlas2Oracle; readonly f64: ForceAtlas2Oracle } | null = null;
        for (let t = 0; t < iterations; t++) {
            hooks.beforeStep?.(sim, t);
            // the GPU's iteration-start state: positions, and the controller K4 left after the previous iteration
            const before = Float32Array.from(positions);
            const controller = sim.stats;
            await sim.step(1);
            const st = sim.stats;
            if (st.trace.length !== 1) {
                throw new Error(`step(1) left ${st.trace.length} trace records`);
            }
            const record = st.trace[0];
            trace.push(record);
            stats.push({ centroid: st.centroid, rmsRadius: st.rmsRadius, layoutRadius: st.layoutRadius });
            const k1: ResyncRecord = {
                swing: record.swing,
                traction: record.traction,
                speed: record.speed,
                speedEfficiency: record.speedEfficiency,
                meanDisplacement: record.meanDisplacement,
                settledCount: record.settledCount,
                centroid: st.centroid,
                rmsRadius: st.rmsRadius,
            };
            gpu.push(k1);
            // the oracles: for t > 0 seeded with the GPU's state of this iteration (K1's statistics are the GPU's
            // own, checked below against the previous oracle's fold), the previous oracle's force as F(t-1)
            const seedOf = (chain: ForceAtlas2Oracle | null): OracleResyncState | null =>
                chain === null
                    ? null
                    : {
                          speed: controller.speed,
                          speedEfficiency: controller.speedEfficiency,
                          swing: controller.swing,
                          traction: controller.traction,
                          centroid: k1.centroid,
                          rmsRadius: k1.rmsRadius,
                          layoutRadius: st.layoutRadius,
                          meanDisplacement: k1.meanDisplacement,
                          settledCount: k1.settledCount,
                          iteration: t,
                          oldForce: chain.stages.force,
                      };
            const o32 = resyncOracle(s, before, options, tuning, mask, "f32", seedOf(previous?.f32 ?? null));
            const o64 = resyncOracle(s, before, options, tuning, mask, "f64", seedOf(previous?.f64 ?? null));
            const r32 = o32.step();
            const r64 = o64.step();
            // K1's statistics of iteration t: the previous oracle's fold of its step (load()'s statistics at t = 0)
            const foldOf = (chain: ForceAtlas2Oracle | null, r: OracleTraceRecord): ResyncRecord => {
                const fold = chain?.peekFold() ?? null;
                return {
                    swing: r.swing,
                    traction: r.traction,
                    speed: r.speed,
                    speedEfficiency: r.speedEfficiency,
                    meanDisplacement: fold?.meanDisplacement ?? r.meanDisplacement,
                    settledCount: fold?.settledCount ?? r.settledCount,
                    centroid: fold?.centroid ?? r.centroid,
                    rmsRadius: fold?.rmsRadius ?? r.rmsRadius,
                };
            };
            f32.push(foldOf(previous?.f32 ?? null, r32));
            f64.push(foldOf(previous?.f64 ?? null, r64));
            const after = layoutStart(positions, s.nodeCount, options.dim ?? 2);
            positionError.f32.push(flooredRelError(after, o32.positions, FLOOR_FRACTION).max);
            positionError.f64.push(flooredRelError(after, o64.positions, FLOOR_FRACTION).max);
            previous = { f32: o32, f64: o64 };
        }
        return { trace, stats, positions, gpu, f32, f64, positionError };
    });
}

/**
 * The two re-synchronised reports: every record of the run against the f32 oracle under
 * fa2-trace-parity.resync.f32 and against the f64 oracle under fa2-trace-parity.resync.f64.
 * @param run - a resyncTrace run
 * @param label - the case label
 * @returns the two reports
 */
export function resyncReports(run: ResyncRun, label: string): { readonly f32: CheckReport; readonly f64: CheckReport } {
    const gpu = resyncValues(run.gpu);
    const samples = gpu.length;
    return {
        f32: {
            worst: ratioOf(resyncValuesError(gpu, resyncValues(run.f32)), toleranceOf("fa2-trace-parity.resync.f32")),
            worstLabel: `${label}: ${run.gpu.length} re-synchronised iterations vs f32`,
            samples,
        },
        f64: {
            worst: ratioOf(resyncValuesError(gpu, resyncValues(run.f64)), toleranceOf("fa2-trace-parity.resync.f64")),
            worstLabel: `${label}: ${run.gpu.length} re-synchronised iterations vs f64`,
            samples,
        },
    };
}

/**
 * The two trace-parity reports of spec 11.4: the first min(10, length) records against the f32 oracle under
 * fa2-trace-parity.f32, every record against the f64 oracle under fa2-trace-parity.f64.
 * @param gpu - the GPU records
 * @param f32 - the f32 oracle's records
 * @param f64 - the f64 oracle's records
 * @param label - the case label for the reports
 * @returns the two reports
 */
export function traceReports(
    gpu: readonly ForceAtlas2TraceRecord[],
    f32: readonly OracleTraceRecord[],
    f64: readonly OracleTraceRecord[],
    label: string,
): { readonly first10: CheckReport; readonly through50: CheckReport } {
    const tight = Math.min(10, gpu.length);
    return {
        first10: {
            worst: ratioOf(traceError(gpu, f32, 0, tight), toleranceOf("fa2-trace-parity.f32")),
            worstLabel: `${label}: iterations 1-${tight} vs f32`,
            samples: 6 * tight,
        },
        through50: {
            worst: ratioOf(traceError(gpu, f64, 0, gpu.length), toleranceOf("fa2-trace-parity.f64")),
            worstLabel: `${label}: iterations 1-${gpu.length} vs f64`,
            samples: 6 * gpu.length,
        },
    };
}

// ---------------------------------------------------------------- the force report (spec 11.4 force parity)

/**
 * One iteration K2 + K3 on a fresh simulation, `force` read through inspect() after K3, against the f64 oracle's
 * force stage with the floored per-node metric under fa2-force-parity. The unseeded start (all NaN) is seeded by
 * load(); the returned `scene` is the array after load(), which the oracle reads (PLAN DECISION 9).
 * @param ctx - the context
 * @param s - the snapshot
 * @param start - the scene start (copied; may be all NaN)
 * @param options - the FA2 options
 * @param tuning - the GPU tuning
 * @param mask - the fixed mask or null
 * @param label - the case label
 * @returns the report, the force (layout units, stride 3) and the start array the oracle used
 */
export async function forceReport(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options,
    tuning: GpuLayoutTuning,
    mask: NodeMask | null,
    label: string,
): Promise<{ readonly report: CheckReport; readonly force: F32; readonly scene: F32 }> {
    const { force, scene } = await withSim(ctx, options, tuning, async (sim) => {
        const owned = Float32Array.from(start);
        sim.load(s, owned);
        if (mask !== null) {
            sim.setFixed(mask);
        }
        const st = debugStages(sim);
        await st.run("K3");
        return { force: asF32(await st.read("force")), scene: owned };
    });
    const { oracle } = forceAtlas2Oracle(
        s,
        Float32Array.from(scene),
        oracleOptionsFor(s, options, tuning, mask, "f64"),
        1,
    );
    const err = flooredRelError(force, oracle.stages.force, FLOOR_FRACTION);
    return {
        report: {
            worst: ratioOf(err.max, toleranceOf("fa2-force-parity")),
            worstLabel: `${label}: node ${err.argmax}`,
            samples: s.nodeCount,
        },
        force,
        scene,
    };
}

// ---------------------------------------------------------------- distributional metrics (spec 11.4)

/**
 * A metrics record as (sorted keys, values) -- the noise fixture layout of the metrics100 member.
 * @param record - a layoutMetrics() record
 * @returns keys and values in key order
 */
export function metricsValues(record: Readonly<Record<string, number>>): {
    readonly keys: string[];
    readonly values: F64;
} {
    const keys = Object.keys(record).sort();
    return { keys, values: Float64Array.from(keys.map((k) => record[k])) };
}

/** The key prefix of the nearest-neighbour histogram bins of layoutMetrics(); the bins are compared as ONE distribution. */
const NN_BIN_PREFIX = "nnBin";

/**
 * The "within 10%" distance of two metrics value lists in one key order: every non-histogram metric (stress, the
 * edge-length and nearest-neighbour quantiles, spread, separation) as |a - b| / max(|b|, DISTRIBUTIONAL_FLOOR), and
 * the nearest-neighbour histogram (the nnBin* keys) as ONE total-variation distance, sum |a_i - b_i| / 2 -- the
 * fraction of the nodes that sit in a different bin. P3-T5 implementation note (a deviation from the plan part's
 * PLAN DECISION 11, which compared every bin relative to its own fraction under a 0.05 floor): on a 34-node graph one
 * bin holds 1 / 34 = 0.029 of the nodes, so a single node changing bin read as 0.6 under the per-bin rule, and the
 * f64 oracle failed that rule against ITSELF under a one-ulp start perturbation (measured 0.75 on karate); the
 * total-variation form is the standard "within 10% in distribution" and charges that node 0.029.
 * @param keys - the metric names in value order
 * @param a - the measured values
 * @param b - the reference values
 * @returns the worst difference (Infinity when the lengths disagree)
 */
export function distributionalValuesError(keys: readonly string[], a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (keys.length !== a.length || keys.length !== b.length) {
        return Number.POSITIVE_INFINITY;
    }
    let worst = 0;
    let l1 = 0;
    for (let i = 0; i < keys.length; i++) {
        if (keys[i].startsWith(NN_BIN_PREFIX)) {
            l1 += Math.abs(a[i] - b[i]);
        } else {
            worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(Math.abs(b[i]), DISTRIBUTIONAL_FLOOR));
        }
    }
    return Math.max(worst, l1 / 2);
}

/**
 * The "within 10%" distance of two metrics records (distributionalValuesError over the union of keys); a key present
 * on one side only is Infinity (the two layouts disagree on the component structure).
 * @param a - the measured record
 * @param b - the reference record
 * @returns the worst difference
 */
export function distributionalError(a: Readonly<Record<string, number>>, b: Readonly<Record<string, number>>): number {
    const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
    const av: number[] = [];
    const bv: number[] = [];
    for (const key of keys) {
        const x = a[key];
        const y = b[key];
        if (x === undefined || y === undefined) {
            return Number.POSITIVE_INFINITY;
        }
        av.push(x);
        bv.push(y);
    }
    return distributionalValuesError(keys, av, bv);
}

/**
 * The key order of the metrics100 noise member's values (metricsValues over the unscaled random1k; the key set is a
 * property of the graph -- `separation` is present iff it has two or more components -- not of the positions), so
 * test/noise-floor.test.ts can apply distributionalValuesError to the committed value lists.
 * @returns the sorted metric names
 */
export function noiseMetricsKeys(): string[] {
    const { s, start } = noiseInputs();
    return metricsValues(layoutMetrics(s, start, 2)).keys;
}

// ---------------------------------------------------------------- errors

/**
 * Awaits a promise that must reject with a WebGpuGraphError and returns the error.
 * @param p - the promise
 * @returns the error
 */
export async function rejectionOf(p: Promise<unknown>): Promise<WebGpuGraphError> {
    try {
        await p;
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw new Error(`rejected with something that is not a WebGpuGraphError: ${String(err)}`);
    }
    throw new Error("expected a rejection, the promise resolved");
}
