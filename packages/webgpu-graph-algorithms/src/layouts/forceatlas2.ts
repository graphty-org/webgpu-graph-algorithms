/**
 * ForceAtlas2 on the exact repulsion tier (spec 7.1-7.18; contract 3.13): the ForceModel that ForceSimulation
 * drives -- the K1 K2 K3 K4 K5 sequence per iteration and toScene per batch (spec 7.4), the override set of the option
 * record (spec 7.2 with the 4.6 NetworkX corrections), the per-iteration Fa2Params values, the controller resets of
 * spec 7.17 and the stats decoder -- plus the two option resolvers and `createForceAtlas2`. Positions are vec4f
 * (xyz + mass) in layout units on the device (D23, 7.18); the speed controller runs on the device (D15); no
 * displacement clamp (D25); K2 runs the thread-per-row tier over [0, n) with USE_PERM false (P3, spec 7.5).
 *
 * Model decisions this file fixes (the plan of P3-T2 lists the reasons): recordIteration with no `upTo` records every
 * stage including toScene; the K1-K5 dispatches of every iteration of one batch share ONE compute pass (opened by the
 * batch's first recordIteration and remembered by batch id) and toScene runs in a second pass that ends it (contract
 * 4.4); the fill kernel takes its FillParams from a model-owned 256-byte uniform buffer ("fillParams"); the first
 * iteration after every load() zeroes oldForce with a fill (paper mode); `repulsion: "grid"` is E_UNSUPPORTED at
 * load() for any n and `"auto"` above exactMaxNodes.
 */

import { type GraphSnapshot } from "@graphty/graph-format";

import {
    FA2_DEFAULTS,
    LAYOUT_TUNING_DEFAULTS,
    MAX_ITERATIONS_PER_STEP,
    TRACE_RECORD_BYTES,
    UNIFORM_SLOT_BYTES,
} from "../constants.js";
import { type GpuContext } from "../context.js";
import { BufferUsage } from "../device/webgpu-constants.js";
import { WebGpuGraphError } from "../errors.js";
import { type CommandBatch } from "../kernel/batch.js";
import { type DispatchPlan, plan1d } from "../kernel/dispatch.js";
import { type BoundKernel, type Kernel } from "../kernel/kernel.js";
import { type UniformBlock, type UniformValues } from "../kernel/struct-block.js";
import { type WgslModuleSpec } from "../kernel/wgsl.js";
import { FA2_PARAMS, FA2_STATE, FA2_TRACE, FILL_PARAMS, graphBindings, kernelSpec } from "../kernels.js";
import {
    type ForceAtlas2Stats,
    type ForceAtlas2TraceRecord,
    type GpuLayoutSimulation,
    type GpuLayoutTuning,
    type ResolvedLayoutTuning,
} from "../types/layout.js";
import { type Binding } from "../types/memory.js";
import { type ForceAtlas2Options, type ResolvedForceAtlas2Options } from "../types/options.js";
import {
    type BufferSpec,
    type ForceModel,
    ForceSimulation,
    type ModelInputs,
    type ModelResources,
    type StateWriter,
} from "./force-simulation.js";
import { resolveNodeMass, resolveWeights } from "./inputs.js";
import { RepulsionExact, type RepulsionExactOverrides } from "./repulsion-exact.js";

// ============================================================ constants and small helpers

/** An override record as the kernel layer takes it. */
type Overrides = Readonly<Record<string, number | boolean>>;

/** The stage names of one iteration in dispatch order plus the per-batch toScene (spec 7.4; contract 3.13). */
const FA2_STAGES = ["K1", "K2", "K3", "K4", "K5", "toScene"] as const;

/** Bytes of the stride-3 f32 force arrays per node. */
const FORCE_BYTES_PER_NODE = 12;

/** The name of the model-owned FillParams buffer (a BufferSpec, reached through ModelResources.buffer). */
const FILL_PARAMS_BUFFER = "fillParams";

/** The one-workgroup dispatch of K1 (spec 7.4). */
const ONE_WORKGROUP: DispatchPlan = { x: 1, y: 1, z: 1, items: 1, stride: null };

/** Every override K2 accepts, with its default (contract 3.10.1 plus the two standard graph overrides). */
const K2_DEFAULTS: Overrides = { LINLOG: false, DISTRIBUTED: false, TIER: 0, USE_PERM: false, HAS_WEIGHTS: false };

/** Every override K5 accepts, with its default. */
const K5_DEFAULTS: Overrides = { SWING_MODE: 0 };

/** 2^32, the modulus of the u32 seed word (computed with `%`, never a bitwise operator). */
const U32_MODULUS = 4294967296;

/** The resolved record with no option given: FA2_DEFAULTS plus the null / origin defaults of spec 7.14. */
const DEFAULT_RESOLVED: ResolvedForceAtlas2Options = Object.freeze<ResolvedForceAtlas2Options>({
    ...FA2_DEFAULTS,
    nodeMass: null,
    nodeSize: null,
    weight: null,
    center: [0, 0, 0],
    seed: null,
});

/**
 * A short, safe rendering of an argument value for error messages (never String() on an object).
 * @param value - the value
 * @returns the rendering
 */
function describeValue(value: unknown): string {
    if (value === null) {
        return "null";
    }
    if (typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
        return String(value);
    }
    if (typeof value === "undefined") {
        return "undefined";
    }
    if (typeof value === "object" && "length" in value && typeof value.length === "number") {
        return `[${value.length} values]`;
    }
    return typeof value;
}

/**
 * The E_INVALID_ARGUMENT error of an option check (contract 3.1: { argument, value, expected }).
 * @param argument - the option name
 * @param value - the value given
 * @param expected - what was expected
 * @returns the error (not thrown here)
 */
function invalid(argument: string, value: unknown, expected: string): WebGpuGraphError {
    return new WebGpuGraphError("E_INVALID_ARGUMENT", `${argument} must be ${expected}; got ${describeValue(value)}`, {
        argument,
        value,
        expected,
    });
}

/**
 * A numeric option: the given value when defined, else the fallback; validated by `check` (the value is checked as
 * `unknown` so a JS caller's string or object is E_INVALID_ARGUMENT too).
 * @param name - the option name
 * @param given - the value given (undefined = absent)
 * @param fallback - the previous record's value or the default
 * @param check - the range predicate over a finite number
 * @param expected - the range in words (the error message)
 * @returns the value
 */
function pickNumber(
    name: string,
    given: number | undefined,
    fallback: number,
    check: (value: number) => boolean,
    expected: string,
): number {
    const value: unknown = given === undefined ? fallback : given;
    if (typeof value !== "number" || !Number.isFinite(value) || !check(value)) {
        throw invalid(name, value, expected);
    }
    return value;
}

/**
 * A boolean option: the given value when defined, else the fallback; a non-boolean is E_INVALID_ARGUMENT.
 * @param name - the option name
 * @param given - the value given (undefined = absent)
 * @param fallback - the previous record's value or the default
 * @returns the value
 */
function pickBoolean(name: string, given: boolean | undefined, fallback: boolean): boolean {
    const value: unknown = given === undefined ? fallback : given;
    if (typeof value !== "boolean") {
        throw invalid(name, value, "a boolean");
    }
    return value;
}

/**
 * The layout dimension: 2 or 3.
 * @param given - the value given (undefined = absent)
 * @param fallback - the previous record's value or the default
 * @returns 2 or 3
 */
function pickDim(given: 2 | 3 | undefined, fallback: 2 | 3): 2 | 3 {
    const value: unknown = given === undefined ? fallback : given;
    if (value !== 2 && value !== 3) {
        throw invalid("dim", value, "2 or 3");
    }
    return value;
}

/**
 * The scene-unit center: an array-like of 2 (z = 0) or 3 finite numbers.
 * @param given - the value given (undefined = absent)
 * @param fallback - the previous record's value or the default
 * @returns the three components
 */
function pickCenter(
    given: ArrayLike<number> | undefined,
    fallback: readonly [number, number, number],
): readonly [number, number, number] {
    if (given === undefined) {
        return fallback;
    }
    const expected = "an array of 2 or 3 finite numbers";
    const value: unknown = given;
    if (typeof value !== "object" || value === null || !("length" in value)) {
        throw invalid("center", given, expected);
    }
    const { length } = value;
    if (length !== 2 && length !== 3) {
        throw invalid("center", given, expected);
    }
    const x: unknown = given[0];
    const y: unknown = given[1];
    const z: unknown = length === 3 ? given[2] : 0;
    if (
        typeof x !== "number" ||
        typeof y !== "number" ||
        typeof z !== "number" ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(z)
    ) {
        throw invalid("center", given, expected);
    }
    return [x, y, z];
}

/**
 * The seed: a finite number, or null (unseeded; 0 keeps the port's "0 = unseeded" quirk through the Lcg).
 * @param given - the value given (undefined = absent)
 * @param fallback - the previous record's value or the default
 * @returns the seed or null
 */
function pickSeed(given: number | null | undefined, fallback: number | null): number | null {
    if (given === undefined) {
        return fallback;
    }
    const value: unknown = given;
    if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
        throw invalid("seed", given, "a finite number or null");
    }
    return value;
}

/**
 * Integer >= 1.
 * @param value - a finite number
 * @returns whether it is a positive integer
 */
function isPositiveInteger(value: number): boolean {
    return Number.isInteger(value) && value >= 1;
}

/**
 * The u32 word written into Fa2Params.seed: 0 for null, else floor(|seed|) mod 2^32.
 * @param seed - the resolved seed
 * @returns the u32 value
 */
function seedWord(seed: number | null): number {
    if (seed === null) {
        return 0;
    }
    return Math.floor(Math.abs(seed)) % U32_MODULUS;
}

/**
 * A scalar field of a block's read() result.
 * @param values - the values read
 * @param name - the field name
 * @returns the number
 */
function scalar(values: UniformValues, name: string): number {
    const value = values[name];
    if (typeof value !== "number") {
        throw invalid(name, value, "a scalar field");
    }
    return value;
}

/**
 * A vector field of a block's read() result.
 * @param values - the values read
 * @param name - the field name
 * @returns the components
 */
function vector(values: UniformValues, name: string): readonly number[] {
    const value = values[name];
    if (typeof value === "number") {
        throw invalid(name, value, "a vector field");
    }
    return value;
}

/**
 * The override record a kernel gets: its defaults overlaid with the values present in the merged set (contract 3.9:
 * a name a spec does not declare is rejected at compose time, so nothing else is passed through).
 * @param merged - the merged override set of the model (plus USE_PERM / HAS_WEIGHTS from the simulation)
 * @param defaults - the kernel's accepted names with their defaults
 * @returns the kernel's override record, every accepted name explicit
 */
function subset(merged: Overrides, defaults: Overrides): Overrides {
    const out: Record<string, number | boolean> = {};
    for (const name of Object.keys(defaults)) {
        out[name] = name in merged ? merged[name] : defaults[name];
    }
    return out;
}

/**
 * The K3 / K4 override values of a merged set (typed for RepulsionExact).
 * @param merged - the merged override set
 * @returns the three K3 / K4 overrides
 */
function repulsionOverrides(merged: Overrides): RepulsionExactOverrides {
    return {
        SWING_MODE: merged.SWING_MODE === 1 ? 1 : 0,
        STRONG_GRAVITY: merged.STRONG_GRAVITY === true,
        GRAVITY_CENTER: merged.GRAVITY_CENTER === 1 ? 1 : 0,
    };
}

// ============================================================ the resolvers

/**
 * Applies FA2_DEFAULTS to the option record; validates ranges (spec 7.14; contract 3.13). With `previous` the record
 * is a PATCH over it (an absent or explicitly undefined field keeps the previous value) and `maxInFlight` may not
 * change (the uniform ring is sized by it at construction). `nodeSize` is E_UNSUPPORTED { option: "nodeSize" } (the
 * adjustSizes correction is deferred); `dissuadeHubs` is kept and ignored, exactly like the CPU.
 * @param options - the caller's options (or a setParams patch)
 * @param previous - the current resolved record when resolving a patch
 * @returns the frozen resolved record
 */
export function resolveForceAtlas2Options(
    options: ForceAtlas2Options | undefined,
    previous?: ResolvedForceAtlas2Options,
): ResolvedForceAtlas2Options {
    const o: ForceAtlas2Options = options ?? {};
    const base = previous ?? DEFAULT_RESOLVED;
    if (previous !== undefined && o.maxInFlight !== undefined && o.maxInFlight !== previous.maxInFlight) {
        throw new WebGpuGraphError(
            "E_INVALID_ARGUMENT",
            `maxInFlight cannot change after creation (the uniform ring is sized by it): got ${describeValue(o.maxInFlight)}, current ${previous.maxInFlight}`,
            { argument: "maxInFlight", value: o.maxInFlight, expected: previous.maxInFlight },
        );
    }
    const nodeSize = o.nodeSize === undefined ? base.nodeSize : o.nodeSize;
    if (nodeSize !== null) {
        throw new WebGpuGraphError(
            "E_UNSUPPORTED",
            "nodeSize (the adjustSizes correction) is not supported by the GPU ForceAtlas2 yet (spec 7.14)",
            { option: "nodeSize", hint: "pass nodeSize: null; the size-aware repulsion is deferred (spec 7.2, Q-25)" },
        );
    }
    const resolved: ResolvedForceAtlas2Options = {
        maxIter: pickNumber("maxIter", o.maxIter, base.maxIter, isPositiveInteger, "an integer >= 1"),
        jitterTolerance: pickNumber("jitterTolerance", o.jitterTolerance, base.jitterTolerance, (v) => v > 0, "> 0"),
        scalingRatio: pickNumber("scalingRatio", o.scalingRatio, base.scalingRatio, (v) => v > 0, "> 0"),
        gravity: pickNumber("gravity", o.gravity, base.gravity, (v) => v >= 0, ">= 0"),
        strongGravity: pickBoolean("strongGravity", o.strongGravity, base.strongGravity),
        distributedAction: pickBoolean("distributedAction", o.distributedAction, base.distributedAction),
        linlog: pickBoolean("linlog", o.linlog, base.linlog),
        nodeMass: o.nodeMass === undefined ? base.nodeMass : o.nodeMass,
        nodeSize,
        weight: o.weight === undefined ? base.weight : o.weight,
        dissuadeHubs: pickBoolean("dissuadeHubs", o.dissuadeHubs, base.dissuadeHubs),
        dim: pickDim(o.dim, base.dim),
        scale: pickNumber("scale", o.scale, base.scale, (v) => v > 0, "> 0"),
        center: pickCenter(o.center, base.center),
        seed: pickSeed(o.seed, base.seed),
        settleThreshold: pickNumber("settleThreshold", o.settleThreshold, base.settleThreshold, (v) => v >= 0, ">= 0"),
        settleWindow: pickNumber(
            "settleWindow",
            o.settleWindow,
            base.settleWindow,
            isPositiveInteger,
            "an integer >= 1",
        ),
        iterationsPerStep: pickNumber(
            "iterationsPerStep",
            o.iterationsPerStep,
            base.iterationsPerStep,
            (v) => isPositiveInteger(v) && v <= MAX_ITERATIONS_PER_STEP,
            `an integer in [1, ${MAX_ITERATIONS_PER_STEP}]`,
        ),
        maxInFlight: pickNumber("maxInFlight", o.maxInFlight, base.maxInFlight, isPositiveInteger, "an integer >= 1"),
    };
    return Object.freeze(resolved);
}

/**
 * Applies LAYOUT_TUNING_DEFAULTS (spec 7.14); the grid knobs are validated and stored but only `repulsion`,
 * `exactMaxNodes`, `deterministic` and `compat` have an effect in P3 (contract 3.3).
 * @param tuning - the GPU-only knobs given (any object carrying them, e.g. the createForceAtlas2 options)
 * @returns the frozen resolved tuning
 */
export function resolveLayoutTuning(tuning: GpuLayoutTuning | undefined): ResolvedLayoutTuning {
    const t: GpuLayoutTuning = tuning ?? {};
    const repulsion: unknown = t.repulsion ?? LAYOUT_TUNING_DEFAULTS.repulsion;
    if (repulsion !== "exact" && repulsion !== "grid" && repulsion !== "auto") {
        throw invalid("repulsion", repulsion, '"exact", "grid" or "auto"');
    }
    const compat: unknown = t.compat ?? LAYOUT_TUNING_DEFAULTS.compat;
    if (compat !== "paper" && compat !== "networkx") {
        throw invalid("compat", compat, '"paper" or "networkx"');
    }
    const resolved: ResolvedLayoutTuning = {
        repulsion,
        exactMaxNodes: pickNumber(
            "exactMaxNodes",
            t.exactMaxNodes,
            LAYOUT_TUNING_DEFAULTS.exactMaxNodes,
            isPositiveInteger,
            "an integer >= 1",
        ),
        nearMax: pickNumber("nearMax", t.nearMax, LAYOUT_TUNING_DEFAULTS.nearMax, isPositiveInteger, "an integer >= 1"),
        deterministic: pickBoolean("deterministic", t.deterministic, LAYOUT_TUNING_DEFAULTS.deterministic),
        gridMax2D: pickNumber(
            "gridMax2D",
            t.gridMax2D,
            LAYOUT_TUNING_DEFAULTS.gridMax2D,
            isPositiveInteger,
            "an integer >= 1",
        ),
        gridMax3D: pickNumber(
            "gridMax3D",
            t.gridMax3D,
            LAYOUT_TUNING_DEFAULTS.gridMax3D,
            isPositiveInteger,
            "an integer >= 1",
        ),
        extentFactor: pickNumber(
            "extentFactor",
            t.extentFactor,
            LAYOUT_TUNING_DEFAULTS.extentFactor,
            (v) => v > 0,
            "> 0",
        ),
        compat,
    };
    return Object.freeze(resolved);
}

// ============================================================ the model

/** Everything bind() produced for one load(): the kernels, their bind groups and the dispatch plans of this n. */
interface BoundModel {
    readonly n: number;
    /** plan1d(n): K2, K5, toScene (every kernel compiles with the same device-derived WG). */
    readonly plan: DispatchPlan;
    /** plan1d(3n): the fills of force / oldForce (3 words per node). */
    readonly fillPlan: DispatchPlan;
    readonly k1: Kernel;
    readonly k1Bound: BoundKernel;
    readonly k2: Kernel;
    /** null when arcCount === 0 (K2 is not recorded; the fill below zeroes force instead, spec 7.5). */
    readonly k2Bound: BoundKernel | null;
    readonly repulsion: RepulsionExact;
    readonly k5: Kernel;
    readonly k5Bound: BoundKernel;
    readonly toScene: Kernel;
    readonly toSceneBound: BoundKernel;
    readonly fill: Kernel;
    /** The fill of `force` (arcCount === 0 only). */
    readonly fillForceBound: BoundKernel | null;
    /** The fill of `oldForce` on the first iteration after load() (paper mode only). */
    readonly fillOldBound: BoundKernel | null;
}

/** The ForceAtlas2 model (spec 7.4: K1 K2 K3 K4 K5 per iteration; toScene once per batch). Stages: ["K1", "K2", "K3", "K4", "K5", "toScene"]. */
export class ForceAtlas2Model implements ForceModel<ForceAtlas2Options, ForceAtlas2Stats> {
    /** The model kind of spec 7.19. */
    readonly kind = "forceatlas2";
    /** The stage names in dispatch order (the `upTo` vocabulary of recordIteration and debugRunStages). */
    readonly stages: readonly ["K1", "K2", "K3", "K4", "K5", "toScene"] = FA2_STAGES;
    /** Fa2Params: the per-iteration uniform block (the simulation writes the shared fields into it). */
    readonly params: UniformBlock = FA2_PARAMS;
    /** Fa2State: the state header block (the simulation allocates and initialises it through this layout). */
    readonly state: UniformBlock = FA2_STATE;
    /** Fa2Trace: one record per iteration of a batch. */
    readonly trace: UniformBlock = FA2_TRACE;
    /** The resolved GPU-only tuning this model was created with. */
    readonly tuning: ResolvedLayoutTuning;

    /**
     * The option record the model holds: the constructor's record, replaced by onSetParams() ONLY. The query hooks
     * (inputs, overrides, paramsFor) never assign it: the simulation calls overrides(next) BEFORE onSetParams(patch)
     * (3.13 setParams: the recompile decision precedes the controller reset), so a hook that tracked the record would
     * hide every law change from onSetParams (PLAN DECISION 12).
     */
    private current: ResolvedForceAtlas2Options;
    /** The resources of the last bind(), or null before the first. */
    private resources: ModelResources | null = null;
    /** The kernels and bind groups of the last bind(), or null before it (and for n === 0). */
    private bound: BoundModel | null = null;
    /** Armed by onLoad(): the next recordIteration zeroes oldForce first (paper mode). */
    private resetOldForce = false;
    /**
     * The K1-K5 compute pass of the batch being recorded, keyed by CommandBatch.id (unique per batch): every
     * recordIteration of one batch dispatches into it (ONE pass per batch, contract 4.4); null between batches and
     * after the toScene pass ended it (PLAN DECISION 2).
     */
    private openPass: { readonly id: number; readonly pass: GPUComputePassEncoder } | null = null;

    /**
     * Creates the model for one simulation.
     * @param tuning - the resolved GPU-only tuning (compat selects SWING_MODE / GRAVITY_CENTER; repulsion and
     *   exactMaxNodes the tier rule)
     * @param resolved - the resolved option record at creation
     */
    constructor(tuning: ResolvedLayoutTuning, resolved: ResolvedForceAtlas2Options) {
        this.tuning = tuning;
        this.current = resolved;
    }

    /**
     * The SWING_MODE of this model: 1 in networkx mode (accumulated, position-mixed sums; m|F| local swing), else 0.
     * @returns 0 or 1
     */
    private get swingMode(): 0 | 1 {
        return this.tuning.compat === "networkx" ? 1 : 0;
    }

    /**
     * force 12n and oldForce 12n (zeroed) in BOTH swing modes (3.10.1: a writable slot is never aliased; mode 1 leaves
     * oldForce unread and unwritten), plus the 256-byte FillParams uniform buffer the fill dispatches read. n = 0
     * reports one node's worth of bytes so no zero-length buffer is ever created (spec 3.6).
     * @param n - the node count
     * @param _dim - the layout dimension (the force arrays are stride 3 in both)
     * @returns the three model-owned buffer specs
     */
    buffers(n: number, _dim: 2 | 3): readonly BufferSpec[] {
        const bytes = Math.max(1, n) * FORCE_BYTES_PER_NODE;
        const usage = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;
        return [
            { name: "force", byteLength: bytes, usage, zero: true },
            { name: "oldForce", byteLength: bytes, usage, zero: true },
            {
                name: FILL_PARAMS_BUFFER,
                byteLength: UNIFORM_SLOT_BYTES,
                usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
                zero: false,
            },
        ];
    }

    /**
     * { mass: resolveNodeMass(s, resolved.nodeMass), weights: resolveWeights(s, resolved.weight) } (3.13 inputs.ts),
     * after the tier rule of spec 7.8 / lead f: `repulsion: "grid"` is E_UNSUPPORTED { feature: "repulsion.grid" }
     * for any n and `"auto"` when n > exactMaxNodes (the grid tier lands in P4); `"exact"` always runs.
     * @param s - the snapshot being loaded
     * @param options - the simulation's current option record
     * @returns the per-load inputs
     */
    inputs(s: GraphSnapshot, options: ForceAtlas2Options): ModelInputs {
        const resolved = resolveForceAtlas2Options(options, this.current);
        const { repulsion, exactMaxNodes } = this.tuning;
        const n = s.nodeCount;
        if (repulsion === "grid" || (repulsion === "auto" && n > exactMaxNodes)) {
            throw new WebGpuGraphError(
                "E_UNSUPPORTED",
                repulsion === "grid"
                    ? 'repulsion: "grid" is not available yet (the grid tier lands in P4)'
                    : `the graph has ${n} nodes, above exactMaxNodes ${exactMaxNodes}, and the grid tier lands in P4`,
                {
                    feature: "repulsion.grid",
                    hint: 'pass repulsion: "exact" (or raise exactMaxNodes) to run the exact tier at this size',
                },
            );
        }
        return { mass: resolveNodeMass(s, resolved.nodeMass), weights: resolveWeights(s, resolved.weight) };
    }

    /**
     * { LINLOG, DISTRIBUTED, TIER: 0, SWING_MODE: compat === "networkx" ? 1 : 0, STRONG_GRAVITY, GRAVITY_CENTER:
     * compat === "networkx" ? 1 : 0 }; USE_PERM / HAS_WEIGHTS are merged in by the simulation from ModelResources
     * (3.10, never from core.hasWeights). A pure query: the simulation calls it with the current AND the next record
     * inside setParams() to decide the recompile, so it never touches the model's record (PLAN DECISION 12).
     * @param options - an option record (the simulation's current one, or the next one of a setParams patch)
     * @returns the model's own override set
     */
    overrides(options: ForceAtlas2Options): Overrides {
        const resolved = resolveForceAtlas2Options(options, this.current);
        const mode = this.swingMode;
        return {
            LINLOG: resolved.linlog,
            DISTRIBUTED: resolved.distributedAction,
            TIER: 0,
            SWING_MODE: mode,
            STRONG_GRAVITY: resolved.strongGravity,
            GRAVITY_CENTER: mode,
        };
    }

    /**
     * The seven module specs of an override set in dispatch order -- K1, K2, K3, K4, K5, toScene, fill -- each with
     * only the override names its entry declares (K2 also USE_PERM / HAS_WEIGHTS), for warm() and the compile matrix.
     * @param overrides - the merged override set (the model's plus USE_PERM / HAS_WEIGHTS)
     * @param _subgroups - accepted for the ForceModel interface and unused: every reducing FA2 body carries
     *   needs: ["subgroups"] in its registry entry and the composer picks the twin from caps.features (contract 4.3)
     * @returns the specs
     */
    specs(overrides: Overrides, _subgroups: boolean): readonly WgslModuleSpec[] {
        const [repulsionSpec, speedSpec] = RepulsionExact.specs(repulsionOverrides(overrides));
        return [
            kernelSpec("fa2-stats-finalize"),
            kernelSpec("fa2-attraction", subset(overrides, K2_DEFAULTS)),
            repulsionSpec,
            speedSpec,
            kernelSpec("fa2-integrate", subset(overrides, K5_DEFAULTS)),
            kernelSpec("fa2-to-scene"),
            kernelSpec("fill"),
        ];
    }

    /**
     * Compiles (through the cache) and binds every kernel against the buffers of this load(): K1, K2 (or the fill of
     * force when arcCount === 0), K3 + K4 through RepulsionExact, K5, toScene, and the fill of oldForce; writes the
     * FillParams { count: 3n, value: 0, mode: 0 } into the model's uniform buffer. With n === 0 nothing is bound.
     * @param resources - the graph, the shared and model buffers, the ring and the cache
     * @param overrides - the merged override set
     */
    async bind(resources: ModelResources, overrides: Overrides): Promise<void> {
        this.dropBound();
        this.resources = resources;
        const { n, pipelines, caps, core, perm, ring, device } = resources;
        if (n === 0) {
            return;
        }
        const [k1, k2, k5, toScene, fill] = await Promise.all([
            pipelines.kernel(kernelSpec("fa2-stats-finalize")),
            pipelines.kernel(kernelSpec("fa2-attraction", subset(overrides, K2_DEFAULTS))),
            pipelines.kernel(kernelSpec("fa2-integrate", subset(overrides, K5_DEFAULTS))),
            pipelines.kernel(kernelSpec("fa2-to-scene")),
            pipelines.kernel(kernelSpec("fill")),
        ]);
        const repulsion = await RepulsionExact.create(pipelines, caps, repulsionOverrides(overrides));
        if (this.resources !== resources) {
            // a newer bind() superseded this one while the pipelines compiled; its own bind groups stand
            return;
        }
        const pos = resources.buffer("positions");
        const scene = resources.buffer("scenePositions");
        const fixed = resources.buffer("fixed");
        const partials = resources.buffer("partials");
        const state = resources.buffer("state");
        const trace = resources.buffer("trace");
        const force = resources.buffer("force");
        const oldForce = resources.buffer("oldForce");
        const fillParamsBuffer = resources.buffer(FILL_PARAMS_BUFFER);
        const params = ring.binding(FA2_PARAMS);
        const fillParams: Binding = {
            buffer: fillParamsBuffer.buffer,
            offset: fillParamsBuffer.offset,
            size: FILL_PARAMS.byteLength,
            window: null,
        };
        const fillBytes = new ArrayBuffer(FILL_PARAMS.byteLength);
        FILL_PARAMS.write(new DataView(fillBytes), { count: 3 * n, value: 0, mode: 0 });
        device.queue.writeBuffer(fillParamsBuffer.buffer, fillParamsBuffer.offset, fillBytes);
        const hasArcs = core.colIdx !== null;
        repulsion.bind({ pos, state, trace, force, oldForce, fixedMask: fixed, partials, params });
        const wg = k1.workgroupSize;
        this.bound = {
            n,
            plan: plan1d(n, wg, caps),
            fillPlan: plan1d(3 * n, wg, caps),
            k1,
            k1Bound: k1.bind({ partials, S: state, T: trace, P: params }),
            k2,
            k2Bound: hasArcs
                ? k2.bind({ ...graphBindings(core, perm, resources.weights), pos, force, P: params })
                : null,
            repulsion,
            k5,
            k5Bound: k5.bind({ force, oldForce, fixedMask: fixed, S: state, pos, partials, P: params }),
            toScene,
            toSceneBound: toScene.bind({ pos, scene, P: params }),
            fill,
            fillForceBound: hasArcs ? null : fill.bind({ dst: force, P: fillParams }),
            fillOldBound: this.swingMode === 0 ? fill.bind({ dst: oldForce, P: fillParams }) : null,
        };
    }

    /**
     * The Fa2Params values of one iteration (the simulation overwrites the shared fields n, dim, flags,
     * iterationIndex, seed, scale, center and settleThreshold with the same values plus the flags).
     * @param iteration - the trace slot of the iteration inside its batch
     * @param options - the simulation's current option record
     * @returns the uniform values
     */
    paramsFor(iteration: number, options: ForceAtlas2Options): UniformValues {
        const { n } = this.requireResources();
        const resolved = resolveForceAtlas2Options(options, this.current);
        const { nearMax, extentFactor } = this.tuning;
        return {
            n,
            dim: resolved.dim,
            flags: 0,
            tierStart: 0,
            tierEnd: n,
            iterationIndex: iteration,
            seed: seedWord(resolved.seed),
            nearMax,
            scalingRatio: resolved.scalingRatio,
            gravity: resolved.gravity,
            jitterTolerance: resolved.jitterTolerance,
            scale: resolved.scale,
            center: [resolved.center[0], resolved.center[1], resolved.center[2], 0],
            settleThreshold: resolved.settleThreshold,
            extentFactor,
            gridMax: 0,
            levels: 0,
            pad: [0, 0, 0, 0],
        };
    }

    /**
     * Records one iteration into the batch: K1, K2 (or the fill of force when arcCount === 0), K3, K4, K5 in the
     * batch's ONE K1-K5 compute pass (opened by the first call of a batch and reused by every later call with the
     * same batch.id, PLAN DECISION 2), then toScene in a second pass that ends it, stopping after stage `upTo` when
     * given (spec 7.4; debugRunStages / inspect, spec 11.9 item 2). The simulation passes "K5" for iterations
     * 0..k-2 and undefined for the last, so toScene runs once per batch. The first call after load() zeroes
     * oldForce before K1 (paper mode). With n === 0 nothing is recorded (PLAN DECISION 9); a call before bind()
     * completed is E_NOT_LOADED (never a silent no-op).
     * @param batch - the batch being recorded
     * @param slot - the UniformRing slot holding this iteration's Fa2Params
     * @param tier - "exact" (the grid tier is E_UNSUPPORTED until P4; the simulation never passes "grid")
     * @param upTo - a stage name to stop after; undefined records every stage including toScene
     */
    recordIteration(batch: CommandBatch, slot: number, tier: "exact" | "grid", upTo?: string): void {
        if (tier === "grid") {
            throw new WebGpuGraphError("E_UNSUPPORTED", "the grid repulsion tier lands in P4", {
                feature: "repulsion.grid",
                hint: 'pass repulsion: "exact"',
            });
        }
        const resources = this.requireResources();
        const stop = upTo === undefined ? FA2_STAGES.length - 1 : this.stageIndex(upTo);
        const { bound } = this;
        if (bound === null) {
            if (resources.n === 0) {
                return;
            }
            throw new WebGpuGraphError(
                "E_NOT_LOADED",
                "the ForceAtlas2 model is not bound (bind() has not completed)",
                {
                    state: "loaded",
                },
            );
        }
        const offset = resources.ring.offsetOf(slot);
        const pass = this.openPass !== null && this.openPass.id === batch.id ? this.openPass.pass : batch.pass("fa2");
        this.openPass = { id: batch.id, pass };
        if (this.resetOldForce) {
            this.resetOldForce = false;
            if (bound.fillOldBound !== null) {
                bound.fill.dispatch(pass, bound.fillOldBound, bound.fillPlan, [0]);
            }
        }
        bound.k1.dispatch(pass, bound.k1Bound, ONE_WORKGROUP, [offset]);
        if (stop < 1) {
            return;
        }
        if (bound.k2Bound !== null) {
            bound.k2.dispatch(pass, bound.k2Bound, bound.plan, [offset]);
        } else if (bound.fillForceBound !== null) {
            bound.fill.dispatch(pass, bound.fillForceBound, bound.fillPlan, [0]);
        }
        if (stop < 2) {
            return;
        }
        bound.repulsion.recordRepulsion(pass, bound.n, offset);
        if (stop < 3) {
            return;
        }
        bound.repulsion.recordSpeedFinalize(pass, offset);
        if (stop < 4) {
            return;
        }
        bound.k5.dispatch(pass, bound.k5Bound, bound.plan, [offset]);
        if (stop < 5) {
            return;
        }
        // the second pass ends the K1-K5 pass; the batch is complete after toScene, so nothing reuses it
        this.openPass = null;
        const scenePass = batch.pass("fa2-to-scene");
        bound.toScene.dispatch(scenePass, bound.toSceneBound, bound.plan, [offset]);
    }

    /**
     * speed = 1, speedEfficiency = 1, swing = 1, traction = 1 (mode 1 accumulates from 1; mode 0 overwrites them each
     * iteration, the initial value is irrelevant); arms the oldForce reset of the next recordIteration.
     * @param state - the state writer of the simulation
     */
    onLoad(state: StateWriter): void {
        state.set("speed", 1);
        state.set("speedEfficiency", 1);
        state.set("swing", 1);
        state.set("traction", 1);
        this.resetOldForce = true;
    }

    /**
     * Mode 0: nothing (D8). Mode 1 (networkx): swing = traction = 1 (spec 7.2 "load() and reheat() reset them to 1").
     * @param state - the state writer of the simulation
     */
    onReheat(state: StateWriter): void {
        if (this.swingMode === 1) {
            state.set("swing", 1);
            state.set("traction", 1);
        }
    }

    /**
     * Resets speed / speedEfficiency to 1 only when linlog, strongGravity or distributedAction changed (spec 7.17);
     * a numeric tweak leaves the controller alone. The patch is validated by the same resolver the simulation uses
     * and applied over the record of the constructor / the previous onSetParams -- the only place `current` moves
     * (PLAN DECISION 12), so the comparison sees the record from BEFORE this setParams even though the simulation
     * already queried overrides(next).
     * @param patch - the setParams patch
     * @param state - the state writer of the simulation
     */
    onSetParams(patch: Partial<ForceAtlas2Options>, state: StateWriter): void {
        const next = resolveForceAtlas2Options(patch, this.current);
        const lawChanged =
            next.linlog !== this.current.linlog ||
            next.strongGravity !== this.current.strongGravity ||
            next.distributedAction !== this.current.distributedAction;
        this.current = next;
        if (lawChanged) {
            state.set("speed", 1);
            state.set("speedEfficiency", 1);
        }
    }

    /**
     * Decodes the state header and the k trace records of a completed batch (k = trace.byteLength / 32) into
     * ForceAtlas2Stats: the exact tier with null grid fields; msPerIteration null (the simulation owns the clock).
     * @param state - a DataView over the 256-byte state header
     * @param trace - a DataView over the k Fa2Trace records of the batch
     * @returns the stats
     */
    readStats(state: DataView, trace: DataView): ForceAtlas2Stats {
        const header = FA2_STATE.read(state);
        const centroid = vector(header, "centroid");
        const records: ForceAtlas2TraceRecord[] = [];
        const count = Math.floor(trace.byteLength / TRACE_RECORD_BYTES);
        for (let i = 0; i < count; i++) {
            const record = FA2_TRACE.read(trace, i * TRACE_RECORD_BYTES);
            records.push({
                swing: scalar(record, "swing"),
                traction: scalar(record, "traction"),
                speed: scalar(record, "speed"),
                speedEfficiency: scalar(record, "speedEfficiency"),
                meanDisplacement: scalar(record, "meanDisplacement"),
                settledCount: scalar(record, "settledCount"),
            });
        }
        return {
            iteration: scalar(header, "iteration"),
            meanDisplacement: scalar(header, "meanDisplacement"),
            rmsRadius: scalar(header, "rmsRadius"),
            layoutRadius: scalar(header, "radius"),
            centroid: [centroid[0], centroid[1], centroid[2]],
            repulsionTier: "exact",
            maxCellOccupancy: null,
            outsideGrid: null,
            msPerIteration: null,
            swing: scalar(header, "swing"),
            traction: scalar(header, "traction"),
            speed: scalar(header, "speed"),
            speedEfficiency: scalar(header, "speedEfficiency"),
            trace: records,
        };
    }

    /**
     * The resources of the last bind(), or E_NOT_LOADED before it.
     * @returns the resources
     */
    private requireResources(): ModelResources {
        if (this.resources === null) {
            throw new WebGpuGraphError("E_NOT_LOADED", "the ForceAtlas2 model has not been bound (load() first)", {
                state: "created",
            });
        }
        return this.resources;
    }

    /**
     * The index of a stage name in FA2_STAGES, or E_INVALID_ARGUMENT.
     * @param upTo - the stage name
     * @returns its index
     */
    private stageIndex(upTo: string): number {
        for (let i = 0; i < FA2_STAGES.length; i++) {
            if (FA2_STAGES[i] === upTo) {
                return i;
            }
        }
        throw invalid("upTo", upTo, FA2_STAGES.join(" | "));
    }

    /**
     * Drops the bind groups of the previous bind() (the buffers changed) so the cached kernels do not accumulate stale
     * groups across reloads; K3 / K4 live inside RepulsionExact and keep the P1-T6 behaviour. Also forgets the pass
     * of a batch recorded before the rebind.
     */
    private dropBound(): void {
        this.openPass = null;
        const { bound } = this;
        if (bound === null) {
            return;
        }
        for (const kernel of [bound.k1, bound.k2, bound.k5, bound.toScene, bound.fill]) {
            kernel.invalidate();
        }
        this.bound = null;
    }
}

// ============================================================ the factory

/**
 * The resolve callback of the simulation's setParams: the patch over the current record, re-validated (the current
 * record is re-resolved first so the callback is typed without a cast).
 * @param patch - the setParams patch
 * @param current - the simulation's current option record
 * @returns the new record
 */
function resolvePatch(patch: Partial<ForceAtlas2Options>, current: ForceAtlas2Options): ForceAtlas2Options {
    return resolveForceAtlas2Options(patch, resolveForceAtlas2Options(current));
}

/**
 * Spec 3.3 createForceAtlas2, verbatim: a GpuLayoutSimulation running ForceAtlas2 on the exact repulsion tier with
 * the option defaults of spec 7.14 and the GPU-only tuning of GpuLayoutTuning (contract 3.13 "Contracts").
 * @param ctx - the context (E_DISPOSED / E_DEVICE_LOST through assertReady)
 * @param options - the ForceAtlas2 options and the GPU-only tuning knobs in one record
 * @returns the simulation in state "created"; load() next
 */
export function createForceAtlas2(
    ctx: GpuContext,
    options?: ForceAtlas2Options & GpuLayoutTuning,
): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats> {
    ctx.assertReady();
    const resolved = resolveForceAtlas2Options(options);
    const tuning = resolveLayoutTuning(options);
    const model = new ForceAtlas2Model(tuning, resolved);
    return new ForceSimulation<ForceAtlas2Options, ForceAtlas2Stats>(ctx, model, resolved, tuning, resolvePatch);
}
