/**
 * The shared layout state machine (spec 7.19): the buffers every model shares (positions, scenePositions, fixed,
 * partials, state + trace), the in-flight batches and their readbacks, the settle window, the fixed mask and the
 * setPosition override list (spec 7.12), the trace and the batch driver; consumes a ForceModel by composition (the
 * per-model kernel sequence, buffers, overrides, per-iteration params, controller hooks and stats decoding).
 * Positions are layout units on the device and scene units in the owner's array (spec 7.18): load() repacks
 * (scene - center) / scale into vec4f with the mass in .w (D23); the model's toScene stage, recorded once per batch
 * on the last iteration (PLAN DECISION 2), writes p * scale + center back into scenePositions.
 */

import { type F32, type GraphSnapshot, makeMask, maskTest, type NodeMask, type U32 } from "@graphty/graph-format";

import {
    FA2_DEFAULTS,
    FA2_FLAG_FIRST,
    MAX_1D_ITEMS,
    MAX_ITERATIONS_PER_STEP,
    PARTIAL_BYTES,
    STATE_HEADER_BYTES,
    UNIFORM_SLOT_BYTES,
} from "../constants.js";
import { type GpuContext } from "../context.js";
import { BufferUsage } from "../device/webgpu-constants.js";
import { hasErrorCode, WebGpuGraphError } from "../errors.js";
import { CommandBatch, type ReadbackRequest, type SubmittedBatch } from "../kernel/batch.js";
import { type PipelineCache } from "../kernel/pipeline-cache.js";
import { type UniformBlock, type UniformFieldType, type UniformValues } from "../kernel/struct-block.js";
import { UniformRing } from "../kernel/uniform-ring.js";
import { type WgslModuleSpec } from "../kernel/wgsl.js";
import { graphOverrides } from "../kernels.js";
import { type ArrayBinding, type CoreBinding } from "../memory/residency.js";
import { type PlanCaps } from "../types/context.js";
import {
    type GpuLayoutSimulation,
    type LayoutStatsBase,
    type ResolvedLayoutTuning,
    type RunOptions,
} from "../types/layout.js";
import { type Binding } from "../types/memory.js";
import { type CommonLayoutOptions, type SimulationOptions } from "../types/options.js";
import { type ResolvedWeights } from "./inputs.js";
import { seedPositions } from "./seed.js";

// ============================================================ the model hook interface (contract 3.13)

/** A model-owned buffer beyond the shared set (spec 7.19: oldForce, velocity, the grid tier's). */
export interface BufferSpec {
    readonly name: string;
    readonly byteLength: number;
    readonly usage: number;
    readonly zero: boolean;
}

/** Host writes into the state header collected between submits (spec 7.19 onLoad / onReheat / onSetParams). */
export interface StateWriter {
    /** Queues a field write (flushed by one writeBuffer before the next submit; also applied to the host shadow immediately). */
    set(field: string, value: number | readonly number[]): void;
    /** The host shadow of a field (last known value). */
    get(field: string): number | readonly number[];
}

/** What a model gets at bind time (after load() / a resize): the graph, the shared and model buffers, the ring, the cache. */
export interface ModelResources {
    readonly device: GPUDevice;
    readonly caps: PlanCaps;
    readonly pipelines: PipelineCache;
    readonly core: CoreBinding;
    readonly perm: Binding | null;
    /**
     * The RESOLVED weights binding (model.inputs(): source "arcs" -> core.weights (null on an unweighted snapshot),
     * "column" -> the registered ArrayBinding's binding, "none" -> null); group 0 is built as
     * graphBindings(core, perm, weights) (3.10).
     */
    readonly weights: Binding | null;
    readonly n: number;
    readonly dim: 2 | 3;
    readonly tier: "exact" | "grid";
    readonly ring: UniformRing;
    /** A shared or model-owned buffer by name: "positions", "scenePositions", "fixed", "partials", "state", "trace", plus every BufferSpec name. */
    buffer(name: string): Binding;
}

/** The per-load inputs a model resolves from the snapshot and its options (mass into the `.w` lane; weights into the group-0 slot). */
export interface ModelInputs {
    readonly mass: F32;
    readonly weights: ResolvedWeights;
}

/**
 * Spec 7.19 ForceModel (kind, buffers, overrides, paramsFor, recordIteration, onLoad, onReheat, onSetParams,
 * readStats) with SEVEN additions (CONTRACT DECISION, each with its reason): `stages` (the kernel stage names for
 * debugRunStages / inspect, spec 11.9 item 2); `params` (the model's uniform block, which must declare the shared
 * field names n, dim, flags, iterationIndex, seed, scale, center, settleThreshold, so the simulation can write
 * them); `state` and `trace` (the model's generated Fa2State / Fa2Trace blocks, so the simulation allocates,
 * initialises and decodes the state buffer through the model's own layouts, D20); `specs` (the module specs of an
 * override set, for warm() and the compile matrix); `bind` (the model compiles and binds after load(), when the
 * buffers exist); `inputs` (mass and weight resolution need the model's option names -- nodeMass / weight are
 * ForceAtlas2Options, not CommonLayoutOptions -- so the simulation calls it at load() and derives USE_PERM /
 * HAS_WEIGHTS from ModelResources, 3.10); and the optional `upTo` argument of `recordIteration` (truncates the
 * sequence after a stage, for debugRunStages / inspect). PLAN DECISION 4: the state block must declare centroid,
 * min, max (vec4f), rmsRadius, radius, meanDisplacement (f32), iteration, settledCount (u32) and fit in
 * STATE_HEADER_BYTES; the params block must declare the eight shared names with center a vec4f and fit in
 * UNIFORM_SLOT_BYTES. PLAN DECISION 6: paramsFor receives the GLOBAL iteration index; the shared fields win.
 * PLAN DECISION 2: recordIteration with `upTo` undefined records EVERY stage (toScene included); `upTo = <stage>`
 * stops after that stage.
 */
export interface ForceModel<Options, Stats extends LayoutStatsBase> {
    readonly kind: "forceatlas2" | "fruchtermanReingold" | "springElectrical";
    readonly stages: readonly string[];
    readonly params: UniformBlock;
    readonly state: UniformBlock;
    readonly trace: UniformBlock;
    buffers(n: number, dim: 2 | 3): readonly BufferSpec[];
    /** Called at load() before the upload: FA2 = { mass: resolveNodeMass(s, nodeMass), weights: resolveWeights(s, weight) }. */
    inputs(s: GraphSnapshot, options: Options): ModelInputs;
    /** The model's OWN override set; the simulation merges USE_PERM / HAS_WEIGHTS from graphOverrides(core, perm, resources.weights) before specs() / bind(). */
    overrides(options: Options): Readonly<Record<string, number | boolean>>;
    specs(overrides: Readonly<Record<string, number | boolean>>, subgroups: boolean): readonly WgslModuleSpec[];
    bind(resources: ModelResources, overrides: Readonly<Record<string, number | boolean>>): Promise<void>;
    paramsFor(iteration: number, options: Options): UniformValues;
    recordIteration(batch: CommandBatch, slot: number, tier: "exact" | "grid", upTo?: string): void;
    onLoad(state: StateWriter): void;
    onReheat(state: StateWriter): void;
    onSetParams(patch: Partial<Options>, state: StateWriter): void;
    readStats(state: DataView, trace: DataView): Stats;
}

// ============================================================ module-private helpers

/** The shared params fields (name, width) every model params block must declare (PLAN DECISION 4). */
const SHARED_PARAM_FIELDS: readonly (readonly [string, number])[] = [
    ["n", 1],
    ["dim", 1],
    ["flags", 1],
    ["iterationIndex", 1],
    ["seed", 1],
    ["scale", 1],
    ["center", 4],
    ["settleThreshold", 1],
];

/** The state fields (name, width) the simulation writes at load() and reads after every batch (PLAN DECISION 4). */
const SHARED_STATE_FIELDS: readonly (readonly [string, number])[] = [
    ["centroid", 4],
    ["min", 4],
    ["max", 4],
    ["rmsRadius", 1],
    ["radius", 1],
    ["meanDisplacement", 1],
    ["iteration", 1],
    ["settledCount", 1],
];

/** The names of the shared buffers a BufferSpec may not reuse. */
const SHARED_BUFFER_NAMES: readonly string[] = ["positions", "scenePositions", "fixed", "partials", "state", "trace"];

/** The per-batch epilogue stage: iterations 0..k-2 stop after the stage that precedes it (PLAN DECISION 2). */
const EPILOGUE_STAGE = "toScene";

let simulationCounter = 0;

/**
 * An E_INVALID_ARGUMENT with the documented details shape.
 * @param argument - the argument name
 * @param value - the offending value
 * @param expected - what was expected
 * @param message - the message
 * @returns the error
 */
function invalidArgument(argument: string, value: unknown, expected: unknown, message: string): WebGpuGraphError {
    return new WebGpuGraphError("E_INVALID_ARGUMENT", message, { argument, value, expected });
}

/**
 * The E_DISPOSED of a simulation.
 * @param label - the simulation label
 * @returns the error
 */
function disposedError(label: string): WebGpuGraphError {
    return new WebGpuGraphError("E_DISPOSED", `${label} is disposed`, { label });
}

/**
 * Coerces a caught value to an Error for Promise rejections.
 * @param err - the caught value
 * @returns the value when it is an Error, else a wrapping Error
 */
function asError(err: unknown): Error {
    if (err instanceof Error) {
        return err;
    }
    return new Error(typeof err === "string" ? err : "unknown error");
}

/**
 * Reads a field of an option record the generic type does not name (maxIter, iterations, nodeSize).
 * @param record - the options or patch
 * @param key - the field name
 * @returns the value, or undefined
 */
function optionField(record: object, key: string): unknown {
    return (record as Record<string, unknown>)[key];
}

/**
 * The number of scalars of a uniform field type.
 * @param type - the field type
 * @returns 1, 2 or 4
 */
function fieldWidth(type: UniformFieldType): number {
    switch (type) {
        case "u32":
        case "i32":
        case "f32":
            return 1;
        case "vec2f":
        case "vec2u":
            return 2;
        case "vec4f":
        case "vec4u":
            return 4;
        default:
            throw invalidArgument("type", type, "a UniformFieldType", "unknown uniform field type");
    }
}

/**
 * The type of a block field, or null when the block does not declare it.
 * @param block - the block
 * @param field - the field name
 * @returns the type or null
 */
function fieldTypeOrNull(block: UniformBlock, field: string): UniformFieldType | null {
    for (const [name, type] of block.fields) {
        if (name === field) {
            return type;
        }
    }
    return null;
}

/**
 * The type of a block field.
 * @param block - the block
 * @param field - the field name
 * @returns the type; E_INVALID_ARGUMENT when the block does not declare the field
 */
function fieldTypeOf(block: UniformBlock, field: string): UniformFieldType {
    const type = fieldTypeOrNull(block, field);
    if (type === null) {
        throw invalidArgument(
            "field",
            field,
            block.fields.map((f) => f[0]),
            `block ${block.name} has no field ${field}`,
        );
    }
    return type;
}

/**
 * Asserts a block declares every (name, width) pair (PLAN DECISION 4).
 * @param block - the block
 * @param required - the required fields
 * @param what - "model.params" or "model.state", for the error
 */
function requireFields(block: UniformBlock, required: readonly (readonly [string, number])[], what: string): void {
    for (const [field, width] of required) {
        const type = fieldTypeOrNull(block, field);
        if (type === null) {
            throw invalidArgument(
                what,
                field,
                required.map((r) => r[0]),
                `${what} block ${block.name} must declare ${field}`,
            );
        }
        if (fieldWidth(type) !== width) {
            throw invalidArgument(
                what,
                field,
                `a field of width ${width}`,
                `${what}.${field} must have width ${width}`,
            );
        }
    }
}

/**
 * A stable string of an override record, for the law-change comparison of setParams.
 * @param record - the overrides
 * @returns the JSON of the sorted entries
 */
function stableKey(record: Readonly<Record<string, number | boolean>>): string {
    return JSON.stringify(
        Object.keys(record)
            .sort()
            .map((k) => [k, record[k]]),
    );
}

/**
 * The stats record with msPerIteration filled by the simulation (PLAN DECISION 3).
 * @param stats - the model's record
 * @param msPerIteration - the measured value
 * @returns the record
 */
function withMs<S extends LayoutStatsBase>(stats: S, msPerIteration: number | null): S {
    return { ...stats, msPerIteration };
}

/**
 * True for a SharedArrayBuffer (the package never writes into one, spec 5.7).
 * @param buffer - the backing buffer
 * @returns true when shared
 */
function isSharedBuffer(buffer: ArrayBufferLike): boolean {
    return typeof SharedArrayBuffer !== "undefined" && buffer instanceof SharedArrayBuffer;
}

/**
 * The validated scale of an option record (FA2_DEFAULTS.scale when absent).
 * @param options - the options
 * @returns the scale
 */
function scaleOf(options: CommonLayoutOptions): number {
    const scale = options.scale ?? FA2_DEFAULTS.scale;
    if (!Number.isFinite(scale) || scale <= 0) {
        throw invalidArgument("scale", scale, "a finite number > 0", `scale must be a finite number > 0, got ${scale}`);
    }
    return scale;
}

/**
 * The validated center of an option record (missing components are 0).
 * @param options - the options
 * @returns [x, y, z]
 */
function centerOf(options: CommonLayoutOptions): [number, number, number] {
    const out: [number, number, number] = [0, 0, 0];
    const { center } = options;
    if (center === undefined) {
        return out;
    }
    for (let axis = 0; axis < 3 && axis < center.length; axis++) {
        const v = center[axis];
        if (!Number.isFinite(v)) {
            throw invalidArgument("center", v, "finite components", `center[${axis}] is not finite`);
        }
        out[axis] = v;
    }
    return out;
}

/**
 * The repulsion tier of a node count under a tuning (PLAN DECISION 11).
 * @param tuning - the resolved tuning
 * @param n - the node count
 * @returns "exact" or "grid"
 */
function tierFor(tuning: ResolvedLayoutTuning, n: number): "exact" | "grid" {
    if (tuning.repulsion === "exact") {
        return "exact";
    }
    if (tuning.repulsion === "grid") {
        return "grid";
    }
    return n <= tuning.exactMaxNodes ? "exact" : "grid";
}

/**
 * A whole-buffer binding.
 * @param buffer - the buffer
 * @param size - its byte length
 * @returns the binding
 */
function wholeBinding(buffer: GPUBuffer, size: number): Binding {
    return { buffer, offset: 0, size, window: null };
}

/**
 * The host side of the state header (PLAN DECISION 5): a shadow of every field, a queue of the fields set since
 * the last flush, and the byte image the queued fields are copied from.
 */
class HeaderWriter implements StateWriter {
    private readonly block: UniformBlock;
    private readonly shadow = new Map<string, number | readonly number[]>();
    private readonly queued = new Map<string, number | readonly number[]>();

    /**
     * Creates a writer over a model's state block with every field 0.
     * @param block - the model's state block
     */
    constructor(block: UniformBlock) {
        this.block = block;
        this.reset();
    }

    /**
     * Queues a field write and applies it to the shadow.
     * @param field - the field name (E_INVALID_ARGUMENT when the block lacks it or the width differs)
     * @param value - a number for a scalar field, an array of the vector's width otherwise
     */
    set(field: string, value: number | readonly number[]): void {
        const width = fieldWidth(fieldTypeOf(this.block, field));
        const ok = width === 1 ? typeof value === "number" : typeof value !== "number" && value.length === width;
        if (!ok) {
            throw invalidArgument(
                "value",
                value,
                `a value of width ${width}`,
                `state field ${field} takes a value of width ${width}`,
            );
        }
        this.shadow.set(field, value);
        this.queued.set(field, value);
    }

    /**
     * The host shadow of a field: the last value set, or the last landed header's value.
     * @param field - the field name
     * @returns the value
     */
    get(field: string): number | readonly number[] {
        const value = this.shadow.get(field);
        if (value === undefined) {
            throw invalidArgument(
                "field",
                field,
                this.block.fields.map((f) => f[0]),
                `state block has no field ${field}`,
            );
        }
        return value;
    }

    /** Every field back to 0 and the queue cleared (load()). */
    reset(): void {
        this.shadow.clear();
        this.queued.clear();
        const zero = this.block.read(new DataView(new ArrayBuffer(this.block.byteLength)));
        for (const [name, value] of Object.entries(zero)) {
            this.shadow.set(name, value);
        }
    }

    /**
     * The byte image of the shadow (the block's own writer, D20).
     * @returns block.byteLength bytes
     */
    headerBytes(): ArrayBuffer {
        const bytes = new ArrayBuffer(this.block.byteLength);
        this.block.write(new DataView(bytes), Object.fromEntries(this.shadow));
        return bytes;
    }

    /**
     * Refreshes the shadow from a landed header; a field set since the last flush keeps its queued value.
     * @param view - the landed header
     */
    absorb(view: DataView): void {
        for (const [name, value] of Object.entries(this.block.read(view))) {
            if (!this.queued.has(name)) {
                this.shadow.set(name, value);
            }
        }
    }

    /**
     * The fields set since the last flush; clears the queue.
     * @returns the field names
     */
    takeQueued(): string[] {
        const fields = [...this.queued.keys()];
        this.queued.clear();
        return fields;
    }

    /** Drops the queue (after load() wrote the whole header). */
    clearQueued(): void {
        this.queued.clear();
    }
}

/** One batch from the step() call that created it until its readback settled (PLAN DECISION 7). */
interface PendingBatch {
    readonly k: number;
    readonly generation: number;
    readonly promise: Promise<void>;
    readonly resolve: () => void;
    readonly reject: (error: Error) => void;
    id: number;
    submitted: SubmittedBatch | null;
    stale: boolean;
    startedAt: number;
    sceneOffset: number;
    stateOffset: number;
    profile: ReadbackRequest | null;
}

/**
 * A pending record whose promise the simulation settles.
 * @param k - iterations of the batch
 * @param generation - the generation at the step() call
 * @returns the record
 */
function createPending(k: number, generation: number): PendingBatch {
    let resolveFn: () => void = () => undefined;
    let rejectFn: (error: Error) => void = () => undefined;
    const promise = new Promise<void>((resolve, reject) => {
        resolveFn = resolve;
        rejectFn = reject;
    });
    return {
        k,
        generation,
        promise,
        resolve: () => {
            resolveFn();
        },
        reject: (error: Error) => {
            rejectFn(error);
        },
        id: 0,
        submitted: null,
        stale: false,
        startedAt: 0,
        sceneOffset: 0,
        stateOffset: 0,
        profile: null,
    };
}

/** The simulation's buffers of one load() (spec 7.3) plus the name -> Binding map the model and inspect() read. */
interface SimulationBuffers {
    readonly positions: GPUBuffer;
    readonly scene: GPUBuffer;
    readonly fixed: GPUBuffer;
    readonly partials: GPUBuffer;
    readonly state: GPUBuffer;
    readonly model: ReadonlyMap<string, { readonly buffer: GPUBuffer; readonly spec: BufferSpec }>;
    readonly bindings: ReadonlyMap<string, Binding>;
    readonly traceRegionBytes: number;
}

/** The registered expanded weight column of the current load (re-expanded only when column.version changed). */
interface WeightsUpload {
    readonly serial: number;
    readonly column: object;
    readonly version: number;
    readonly data: F32;
    readonly upload: ArrayBinding;
}

// ============================================================ the simulation

/**
 * The shared layout state machine (spec 7.19): buffers, in-flight batches, readback, settle window, fixed mask,
 * setPosition overrides, trace, batch driver; consumes a ForceModel by composition.
 */
export class ForceSimulation<
    Options extends CommonLayoutOptions & SimulationOptions,
    Stats extends LayoutStatsBase,
> implements GpuLayoutSimulation<Options, Stats> {
    readonly ctx: GpuContext;
    readonly model: ForceModel<Options, Stats>;
    readonly tuning: ResolvedLayoutTuning;
    /**
     * The uniform ring, sized (maxInFlight + 1) x MAX_ITERATIONS_PER_STEP slots at construction (CONTRACT DECISION:
     * reserve() wraps to 0 when the tail is short, so maxInFlight batches of up to MAX_ITERATIONS_PER_STEP slots
     * plus one wasted tail always fit without a slot being rewritten while a submitted batch still reads it; hence
     * setParams cannot change maxInFlight, 3.13 forceatlas2.ts).
     * @internal
     */
    readonly ring: UniformRing;
    /**
     * Present when ctx.debug.inspect is true: reads back any named buffer (a shared name, a BufferSpec name,
     * "force", "state", "trace") after the last submitted kernel; resolves a Float32Array except for "fixed",
     * "trace" (raw words) which resolve Uint32Array.
     */
    inspect?: (name: string) => Promise<Float32Array | Uint32Array>;
    /**
     * Present when ctx.debug.inspect is true: records ONE iteration truncated after stage `upTo` (a model stage
     * name), submits, awaits it (no readback into positions, no stats update).
     * @internal
     */
    debugRunStages?: (upTo: string) => Promise<void>;

    private readonly resolveOptions: (patch: Partial<Options>, current: Options) => Options;
    private readonly maxInFlight: number;
    private readonly label: string;
    private readonly traceBytes: number;
    private readonly writer: HeaderWriter;
    private readonly overrideList = new Map<number, number>();
    private readonly pending: PendingBatch[] = [];
    private readonly unregisterLost: () => void;
    private optionsValue: Options;
    private stateValue: "created" | "loaded" | "disposed" = "created";
    private generationValue = 0;
    private tierValue: "exact" | "grid";
    private dimValue: 2 | 3;
    private snapshot: GraphSnapshot | null = null;
    private serial: number | null = null;
    private owner: F32 | null = null;
    private n = 0;
    private scale = 1;
    private center: [number, number, number] = [0, 0, 0];
    private buffers: SimulationBuffers | null = null;
    private core: CoreBinding | null = null;
    private resources: ModelResources | null = null;
    private weightsUpload: WeightsUpload | null = null;
    private ready: Promise<void> = Promise.resolve();
    private submitChain: Promise<void> = Promise.resolve();
    private fixedWords: U32 = new Uint32Array(0);
    private fixedDirty = false;
    private iterationsSubmitted = 0;
    private iterationsDoneValue = 0;
    private settledCountValue = 0;
    private settledValue = false;
    private firstPending = true;
    private statsValue: Stats | null = null;
    private lastSubmittedBatchIdValue = 0;
    /** The id of the last batch submitted BEFORE the most recent reheat() (PLAN DECISION 21); 0 = none. */
    private reheatedAfterBatchId = 0;
    private coalescedValue = 0;
    private torndown = false;

    /**
     * Creates a simulation over a context and a model (state "created"; load() makes it "loaded").
     * @param ctx - the context (ready)
     * @param model - the force model
     * @param options - the option record (defaults applied by the caller's factory)
     * @param tuning - the resolved GPU tuning
     * @param resolve - how setParams merges a patch into the current record
     */
    constructor(
        ctx: GpuContext,
        model: ForceModel<Options, Stats>,
        options: Options,
        tuning: ResolvedLayoutTuning,
        resolve: (patch: Partial<Options>, current: Options) => Options,
    ) {
        ctx.assertReady();
        requireFields(model.params, SHARED_PARAM_FIELDS, "model.params");
        requireFields(model.state, SHARED_STATE_FIELDS, "model.state");
        if (model.params.byteLength > UNIFORM_SLOT_BYTES) {
            throw invalidArgument(
                "model.params",
                model.params.byteLength,
                `<= ${UNIFORM_SLOT_BYTES} bytes`,
                "the params block must fit one ring slot",
            );
        }
        if (model.state.byteLength > STATE_HEADER_BYTES) {
            throw invalidArgument(
                "model.state",
                model.state.byteLength,
                `<= ${STATE_HEADER_BYTES} bytes`,
                "the state block must fit the state header",
            );
        }
        if (model.stages.length === 0) {
            throw invalidArgument(
                "model.stages",
                model.stages,
                "at least one stage name",
                "a model declares its stages",
            );
        }
        const maxInFlight = options.maxInFlight ?? FA2_DEFAULTS.maxInFlight;
        if (!Number.isInteger(maxInFlight) || maxInFlight < 1) {
            throw invalidArgument(
                "maxInFlight",
                maxInFlight,
                "an integer >= 1",
                `maxInFlight must be an integer >= 1, got ${maxInFlight}`,
            );
        }
        const dim = options.dim ?? FA2_DEFAULTS.dim;
        if (dim !== 2 && dim !== 3) {
            throw invalidArgument("dim", dim, "2 or 3", `dim must be 2 or 3, got ${String(dim)}`);
        }
        // scale and center are validated now so a bad option never reaches load()
        scaleOf(options);
        centerOf(options);
        simulationCounter++;
        this.ctx = ctx;
        this.model = model;
        this.tuning = tuning;
        this.resolveOptions = resolve;
        this.optionsValue = options;
        this.maxInFlight = maxInFlight;
        this.dimValue = dim;
        this.tierValue = tuning.repulsion === "grid" ? "grid" : "exact";
        this.label = `${model.kind}#${simulationCounter}`;
        this.traceBytes = model.trace.byteLength;
        this.writer = new HeaderWriter(model.state);
        this.ring = new UniformRing(
            ctx.device,
            ctx.allocator,
            (maxInFlight + 1) * MAX_ITERATIONS_PER_STEP,
            `${this.label}/ring`,
        );
        this.unregisterLost = ctx.onLost((info) => {
            this.onDeviceLost(info);
        });
        if (ctx.debug.inspect) {
            this.inspect = (name: string): Promise<Float32Array | Uint32Array> => this.inspectBuffer(name);
            this.debugRunStages = (upTo: string): Promise<void> => this.runStages(upTo);
        }
    }

    // ---------------------------------------------------------------- read-only state

    /**
     * "created" | "loaded" | "disposed".
     * @returns the state
     */
    get state(): "created" | "loaded" | "disposed" {
        return this.stateValue;
    }

    /**
     * The current options record (defaults applied).
     * @returns the record
     */
    get options(): Options {
        return this.optionsValue;
    }

    /**
     * The repulsion tier of the current load ("exact" until P4 lifts the grid tier).
     * @returns the tier
     */
    get tier(): "exact" | "grid" {
        return this.tierValue;
    }

    /**
     * The generation counter bumped by every load() (stale readbacks are discarded).
     * @internal
     * @returns the generation
     */
    get generation(): number {
        return this.generationValue;
    }

    /**
     * Nodes of the current load (0 before load()).
     * @returns the node count
     */
    get nodeCount(): number {
        return this.n;
    }

    /**
     * 2 or 3, fixed at construction (setParams({ dim }) is rejected, spec 7.13).
     * @returns the dimension
     */
    get dim(): 2 | 3 {
        return this.dimValue;
    }

    /**
     * iterationsDone >= the budget OR settledCount >= settleWindow of the last completed batch (spec 7.17); true for an empty graph.
     * @returns whether the layout is settled
     */
    get settled(): boolean {
        return this.settledValue;
    }

    /**
     * Batches created by step() and not yet landed (PLAN DECISION 7: counted from the step() call).
     * @returns the count
     */
    get inFlight(): number {
        return this.pending.length;
    }

    /**
     * Iterations landed since load() or the last reheat().
     * @returns the count
     */
    get iterationsDone(): number {
        return this.iterationsDoneValue;
    }

    /**
     * The stats of the last completed batch; before one lands, the host-written header decoded through the model
     * with an empty trace and msPerIteration null.
     * @returns the stats
     */
    get stats(): Stats {
        if (this.statsValue !== null) {
            return this.statsValue;
        }
        const header = new DataView(this.writer.headerBytes());
        const trace = new DataView(new ArrayBuffer(0));
        return withMs(this.model.readStats(header, trace), null);
    }

    /**
     * The override list (spec 7.12): rows whose readback is skipped while a batch older than the write is in flight.
     * @internal
     * @returns row -> lastSubmittedBatchId at the write
     */
    get overrides(): ReadonlyMap<number, number> {
        return this.overrideList;
    }

    /**
     * The last submitted batch id.
     * @internal
     * @returns the id (0 before any submit)
     */
    get lastSubmittedBatchId(): number {
        return this.lastSubmittedBatchIdValue;
    }

    /**
     * Number of step() calls that returned an existing batch's promise instead of submitting (spec 7.19 item 3;
     * read by test/helpers/frame-loop.ts).
     * @internal
     * @returns the count
     */
    get coalesced(): number {
        return this.coalescedValue;
    }

    // ---------------------------------------------------------------- load

    /**
     * Uploads the snapshot's core, seeds the NaN rows of `positions`, repacks it into layout-unit vec4f with the
     * mass in .w, writes the initial state, starts the model's compile + bind and enters "loaded" (spec 7.19; a
     * load() during flight bumps the generation and discards the in-flight batches). Order of the checks (PLAN
     * DECISION 16): E_DISPOSED, the context's assertReady, E_SNAPSHOT (directed), E_TOO_LARGE (nodeCount), the
     * positions array, the tuning, then the core upload and model.inputs() BEFORE any state is touched.
     * PLAN DECISION 10: nodeCount 0 loads with no GPU work at all. PLAN DECISION 11: the tier is resolved here
     * and "grid" is E_UNSUPPORTED until P4. PLAN DECISION 17: a same-size load() keeps the buffers and re-zeroes
     * partials, the trace region and every `zero: true` model buffer.
     * @param snapshot - an undirected snapshot
     * @param positions - the owner's stride-3 scene-unit array (NaN rows are seeded in place)
     */
    load(snapshot: GraphSnapshot, positions: F32): void {
        this.assertNotDisposed();
        this.ctx.assertReady();
        if (snapshot.directed) {
            throw new WebGpuGraphError(
                "E_SNAPSHOT",
                "a layout needs an undirected snapshot: pass toUndirected().snapshot",
                {
                    reason: "directed",
                    serial: snapshot.serial,
                },
            );
        }
        const n = snapshot.nodeCount;
        if (n > MAX_1D_ITEMS) {
            throw new WebGpuGraphError(
                "E_TOO_LARGE",
                `${n} nodes exceed ${MAX_1D_ITEMS} (the third partials level is P4)`,
                {
                    needed: n,
                    limit: MAX_1D_ITEMS,
                    path: "partials",
                    algorithm: this.model.kind,
                },
            );
        }
        if (positions.length !== 3 * n) {
            throw invalidArgument(
                "positions",
                positions.length,
                3 * n,
                `positions has ${positions.length} entries, expected ${3 * n}`,
            );
        }
        if (isSharedBuffer(positions.buffer)) {
            throw invalidArgument(
                "positions",
                "SharedArrayBuffer",
                "an ArrayBuffer-backed Float32Array",
                "positions must not be backed by a SharedArrayBuffer",
            );
        }
        const scale = scaleOf(this.optionsValue);
        const center = centerOf(this.optionsValue);
        const tier = tierFor(this.tuning, n);
        if (tier === "grid") {
            throw new WebGpuGraphError(
                "E_UNSUPPORTED",
                `the grid repulsion tier lands at P4 (n = ${n}, exactMaxNodes = ${this.tuning.exactMaxNodes})`,
                {
                    feature: "repulsion.grid",
                    hint: 'pass repulsion: "exact" or raise exactMaxNodes',
                },
            );
        }
        const positionsBytes = 16 * n;
        if (positionsBytes > this.ctx.caps.limits.maxBufferSize) {
            throw new WebGpuGraphError("E_TOO_LARGE", `${positionsBytes} bytes of positions exceed maxBufferSize`, {
                needed: positionsBytes,
                limit: this.ctx.caps.limits.maxBufferSize,
                path: "positions",
                algorithm: this.model.kind,
            });
        }
        let core: CoreBinding | null = null;
        let inputs: ModelInputs | null = null;
        if (n > 0) {
            core = this.ctx.residency.core(snapshot);
            if (core.plan === "windowed") {
                throw new WebGpuGraphError(
                    "E_TOO_LARGE",
                    "a windowed core cannot be walked by the layout kernels until P4",
                    {
                        needed: snapshot.arcCount,
                        limit: this.ctx.caps.limits.maxStorageBufferBindingSize,
                        path: "windowed",
                        algorithm: this.model.kind,
                    },
                );
            }
            inputs = this.model.inputs(snapshot, this.optionsValue);
            if (inputs.mass.length !== n) {
                throw invalidArgument(
                    "nodeMass",
                    inputs.mass.length,
                    n,
                    `the model resolved ${inputs.mass.length} masses for ${n} nodes`,
                );
            }
        }

        // ---- every check passed: mutate
        this.generationValue++;
        this.discardPending();
        const resized = this.buffers === null || n !== this.n || snapshot.serial !== this.serial;
        this.snapshot = snapshot;
        this.serial = snapshot.serial;
        this.owner = positions;
        this.n = n;
        this.tierValue = tier;
        this.scale = scale;
        this.center = center;
        this.core = core;
        this.iterationsSubmitted = 0;
        this.iterationsDoneValue = 0;
        this.settledCountValue = 0;
        this.firstPending = true;
        this.statsValue = null;
        this.writer.reset();
        if (resized) {
            this.destroyBuffers();
            this.overrideList.clear();
            this.fixedWords = makeMask(n);
            this.fixedDirty = false;
        }
        if (n === 0 || core === null || inputs === null) {
            // PLAN DECISION 10: an empty graph loads with no GPU work
            this.weightsUpload = null;
            this.resources = null;
            // keep the chain (PLAN DECISION 20): a bind of the previous load may still be running and the next
            // non-empty load() must start its bind after it; its outcome is irrelevant here
            this.ready = this.ready.catch(() => undefined);
            this.settledValue = true;
            this.stateValue = "loaded";
            return;
        }
        const buffers = resized ? this.allocate(n) : this.requireBuffers();
        if (!resized) {
            this.clearKept(buffers);
        }
        this.buffers = buffers;
        const range = this.model.kind === "fruchtermanReingold" ? "fr" : "fa2";
        seedPositions(snapshot, positions, this.optionsValue.seed ?? null, this.dimValue, scale, center, range);
        this.uploadPositions(buffers, positions, inputs.mass);
        const weights = this.resolveWeightsBinding(snapshot, core, inputs.weights);
        const overrides = { ...this.model.overrides(this.optionsValue), ...graphOverrides(core, null, weights) };
        const resources = this.makeResources(core, weights, buffers);
        this.resources = resources;
        this.startBind(resources, overrides);
        this.settledValue = false;
        this.stateValue = "loaded";
    }

    // ---------------------------------------------------------------- step and the batch driver

    /**
     * Submits k iterations (spec 7.19 items 1-6): "created" -> E_NOT_LOADED; "disposed" -> E_DISPOSED; a released
     * snapshot -> E_RELEASED; k outside [1, MAX_ITERATIONS_PER_STEP] -> E_INVALID_ARGUMENT; settled -> resolves at
     * once; inFlight >= maxInFlight -> the OLDEST pending batch's promise (coalesced); else the batch is queued on
     * the submit chain (after the bind promise and allocator.check()) and its promise resolves when its readback
     * landed in the owner's array. The same promise object is returned for every coalesced call. PLAN DECISION 7:
     * the batch counts in `inFlight` from this call, not from its submission.
     * @param iterations - k (default options.iterationsPerStep, default 1)
     * @returns resolves when the batch landed (or was discarded); rejects E_VALIDATION / E_DEVICE_LOST / E_OUT_OF_MEMORY / E_SHADER_COMPILE
     */
    step(iterations?: number): Promise<void> {
        if (this.stateValue === "created") {
            return Promise.reject(
                new WebGpuGraphError("E_NOT_LOADED", `${this.label}: load() first`, { state: "created" }),
            );
        }
        if (this.stateValue === "disposed") {
            return Promise.reject(disposedError(this.label));
        }
        try {
            this.ctx.assertReady();
        } catch (err) {
            return Promise.reject(asError(err));
        }
        const { snapshot } = this;
        if (snapshot !== null && this.ctx.residency.isReleased(snapshot.serial)) {
            return Promise.reject(
                new WebGpuGraphError(
                    "E_RELEASED",
                    `snapshot ${snapshot.serial} was released while ${this.label} used it`,
                    {
                        serial: snapshot.serial,
                    },
                ),
            );
        }
        const k = iterations ?? this.optionsValue.iterationsPerStep ?? FA2_DEFAULTS.iterationsPerStep;
        if (!Number.isInteger(k) || k < 1 || k > MAX_ITERATIONS_PER_STEP) {
            return Promise.reject(
                invalidArgument(
                    "iterations",
                    k,
                    `an integer in [1, ${MAX_ITERATIONS_PER_STEP}]`,
                    `step(${k}): iterations must be an integer in [1, ${MAX_ITERATIONS_PER_STEP}]`,
                ),
            );
        }
        if (this.settledValue || this.n === 0) {
            return Promise.resolve();
        }
        if (this.pending.length >= this.maxInFlight) {
            this.coalescedValue++;
            return this.pending[0].promise;
        }
        const record = createPending(k, this.generationValue);
        this.pending.push(record);
        this.submitChain = this.submitChain.then(() => this.submitBatch(record));
        return record.promise;
    }

    /**
     * Resolves when nothing is in flight (every pending batch landed or was discarded).
     * @returns the promise
     */
    async flush(): Promise<void> {
        while (this.pending.length > 0) {
            await Promise.allSettled(this.pending.map((r) => r.promise));
        }
    }

    /**
     * Node batch driver: loops step(batch) until settled, the budget (`maxIter ?? options.maxIter`, PLAN DECISION
     * 1) is reached or the signal aborts (E_ABORTED; the batch in flight at the abort is discarded, spec 5.7 / Q-15).
     * @param options - maxIter, batch (default 8), signal
     * @returns the stats of the last completed batch
     */
    async run(options?: RunOptions): Promise<Stats> {
        this.assertNotDisposed();
        const batch = options?.batch ?? 8;
        if (!Number.isInteger(batch) || batch < 1 || batch > MAX_ITERATIONS_PER_STEP) {
            throw invalidArgument(
                "batch",
                batch,
                `an integer in [1, ${MAX_ITERATIONS_PER_STEP}]`,
                "run(): batch must be an integer in [1, MAX_ITERATIONS_PER_STEP]",
            );
        }
        const budget = options?.maxIter ?? this.iterationBudget();
        if (Number.isNaN(budget) || budget < 0) {
            throw invalidArgument("maxIter", budget, "a number >= 0", "run(): maxIter must be >= 0");
        }
        const signal = options?.signal;
        const onAbort = (): void => {
            this.discardPending();
        };
        if (signal !== undefined) {
            if (signal.aborted) {
                throw this.abortedError();
            }
            signal.addEventListener("abort", onAbort, { once: true });
        }
        try {
            while (!this.settled && this.iterationsDoneValue < budget) {
                if (signal?.aborted === true) {
                    throw this.abortedError();
                }
                const k = Math.min(batch, budget - this.iterationsDoneValue);
                await this.step(k);
            }
            if (signal?.aborted === true) {
                throw this.abortedError();
            }
            return this.stats;
        } finally {
            signal?.removeEventListener("abort", onAbort);
        }
    }

    // ---------------------------------------------------------------- pins, drags, reheat, params

    /**
     * Copies the mask words (spec 7.12): E_INVALID_ARGUMENT when shorter than ceil(n / 32); the buffer is marked
     * dirty and re-uploaded before the next submit; reheat() iff some bit went 1 -> 0 (an unpin). PLAN DECISION
     * 18: the comparison is bit by bit below n (bits at or above n never count as an unpin).
     * @param mask - the NodeMask (LSB-first words)
     */
    setFixed(mask: NodeMask): void {
        this.assertLoaded();
        const { n } = this;
        const words = Math.ceil(n / 32);
        if (mask.length < words) {
            throw invalidArgument(
                "mask",
                mask.length,
                words,
                `setFixed: the mask has ${mask.length} words, ${words} needed for ${n} nodes`,
            );
        }
        let unpinned = false;
        for (let w = 0; w < words && !unpinned; w++) {
            if (this.fixedWords[w] === mask[w]) {
                continue;
            }
            const last = Math.min(n, w * 32 + 32);
            for (let i = w * 32; i < last; i++) {
                if (maskTest(this.fixedWords, i) && !maskTest(mask, i)) {
                    unpinned = true;
                    break;
                }
            }
        }
        this.fixedWords.set(mask.subarray(0, words));
        this.fixedDirty = true;
        if (unpinned) {
            this.reheat();
        }
    }

    /**
     * Writes a scene-unit position (spec 7.12): into the owner's array at once, into the device (layout units, z 0
     * in 2D, 12 bytes at 16 i, queue-ordered before the next submit), into the override list keyed by the last
     * submitted batch id, then reheat(). PLAN DECISION 18: non-finite coordinates are E_INVALID_ARGUMENT.
     * @param index - the node index (< n)
     * @param x - scene x
     * @param y - scene y
     * @param z - scene z (ignored on the device in 2D)
     */
    setPosition(index: number, x: number, y: number, z: number): void {
        this.assertLoaded();
        const { n } = this;
        if (!Number.isInteger(index) || index < 0 || index >= n) {
            throw invalidArgument(
                "index",
                index,
                `an integer in [0, ${n})`,
                `setPosition(${index}): index out of range`,
            );
        }
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            throw invalidArgument(
                "position",
                [x, y, z],
                "finite coordinates",
                "setPosition: coordinates must be finite",
            );
        }
        const { owner, buffers } = this;
        if (owner === null || buffers === null) {
            throw new WebGpuGraphError("E_NOT_LOADED", `${this.label}: load() first`, { state: this.stateValue });
        }
        owner[3 * index] = x;
        owner[3 * index + 1] = y;
        owner[3 * index + 2] = z;
        const [cx, cy, cz] = this.center;
        const layout = new Float32Array([
            (x - cx) / this.scale,
            (y - cy) / this.scale,
            this.dimValue === 2 ? 0 : (z - cz) / this.scale,
        ]);
        this.ctx.device.queue.writeBuffer(buffers.positions, 16 * index, layout);
        this.overrideList.set(index, this.lastSubmittedBatchIdValue);
        this.reheat();
    }

    /**
     * iterationsDone = 0, settledCount = 0 (a queued state write), model.onReheat(writer); nothing else (D8: the
     * speed controller keeps its state). `settled` is recomputed at once (PLAN DECISION 8). Allowed before load().
     * PLAN DECISION 21: the id of the last batch submitted so far is remembered; a batch with that id or an older
     * one was computed from the pre-reheat counter, so its landed settledCount is taken as 0 (see onLanded) and a
     * drag or unpin can never leave the simulation settled through a batch that was already in flight.
     */
    reheat(): void {
        this.assertNotDisposed();
        this.iterationsDoneValue = 0;
        this.settledCountValue = 0;
        this.reheatedAfterBatchId = this.lastSubmittedBatchIdValue;
        this.writer.set("settledCount", 0);
        this.model.onReheat(this.writer);
        this.settledValue = this.computeSettled();
    }

    /**
     * Live tuning (spec 3.3, 7.17): `dim` differing from the current -> E_INVALID_ARGUMENT; `maxInFlight` differing
     * -> E_INVALID_ARGUMENT (the ring is sized by it); a non-null `nodeSize` -> E_UNSUPPORTED; the record is
     * replaced through `resolve(patch, current)`; a change of the model's override set (a force LAW) recompiles
     * and re-binds; scale / center changes reach the next batch through the shared params of every slot (the
     * model's toScene reads them); then model.onSetParams(patch, writer) and reheat().
     * @param patch - the fields to change
     */
    setParams(patch: Partial<Options>): void {
        this.assertNotDisposed();
        const current = this.optionsValue;
        const { dim, maxInFlight } = patch;
        if (dim !== undefined && dim !== this.dimValue) {
            throw invalidArgument(
                "dim",
                dim,
                this.dimValue,
                "dim is fixed at construction; graphty-element re-creates the engine on a view-mode switch (spec 7.13)",
            );
        }
        if (maxInFlight !== undefined && maxInFlight !== this.maxInFlight) {
            throw invalidArgument(
                "maxInFlight",
                maxInFlight,
                this.maxInFlight,
                "maxInFlight is fixed at construction (the uniform ring is sized by it)",
            );
        }
        const nodeSize = optionField(patch, "nodeSize");
        if (nodeSize !== undefined && nodeSize !== null) {
            throw new WebGpuGraphError("E_UNSUPPORTED", "nodeSize (adjustSizes) is deferred (spec 7.14)", {
                option: "nodeSize",
                hint: "leave nodeSize null; the size correction ships in a later slice",
            });
        }
        const next = this.resolveOptions(patch, current);
        const scale = scaleOf(next);
        const center = centerOf(next);
        const before = stableKey(this.model.overrides(current));
        const after = stableKey(this.model.overrides(next));
        this.optionsValue = next;
        this.scale = scale;
        this.center = center;
        const { resources, core } = this;
        if (this.stateValue === "loaded" && resources !== null && core !== null && before !== after) {
            const overrides = { ...this.model.overrides(next), ...graphOverrides(core, null, resources.weights) };
            this.startBind(resources, overrides);
        }
        this.model.onSetParams(patch, this.writer);
        this.reheat();
    }

    // ---------------------------------------------------------------- dispose and device loss

    /**
     * Discards in-flight batches (their promises resolve), destroys every simulation buffer and the ring, trims
     * the pool, unregisters the loss listener; state "disposed"; idempotent.
     */
    dispose(): void {
        if (this.stateValue === "disposed") {
            return;
        }
        this.stateValue = "disposed";
        this.teardown();
    }

    // ---------------------------------------------------------------- private: the batch driver

    /**
     * The serialised submission of one pending batch: awaits the bind / warm promise and allocator.check(), then
     * records and submits unless the record went stale meanwhile; wires the readback handlers. A record that went
     * stale while waiting (a load() or dispose() superseded it) RESOLVES even when the wait rejected (PLAN
     * DECISION 8: the rejection belongs to the superseded generation -- a bind that failed E_SHADER_COMPILE, or
     * the ring's E_DISPOSED after dispose() tore it down under a running bind).
     * @param record - the pending batch
     */
    private async submitBatch(record: PendingBatch): Promise<void> {
        try {
            await this.ready;
            await this.ctx.allocator.check();
        } catch (err) {
            this.finish(record);
            if (this.isStale(record)) {
                record.resolve();
                return;
            }
            if (hasErrorCode(err, "E_OUT_OF_MEMORY")) {
                this.outOfMemory();
            }
            record.reject(asError(err));
            return;
        }
        if (this.isStale(record)) {
            this.finish(record);
            record.resolve();
            return;
        }
        let submitted: SubmittedBatch;
        try {
            submitted = this.recordAndSubmit(record);
        } catch (err) {
            this.finish(record);
            record.reject(asError(err));
            return;
        }
        submitted.readback.then(
            (bytes) => {
                this.onLanded(record, bytes);
            },
            (err: unknown) => {
                this.onFailed(record, err);
            },
        );
    }

    /**
     * Flushes the host writes, fills k ring slots, records k iterations (only the last one runs the model's toScene
     * epilogue, PLAN DECISION 2) + the two readbacks into one CommandBatch and submits it (spec 7.19 items 4-5).
     * @param record - the pending batch
     * @returns the submitted batch
     */
    private recordAndSubmit(record: PendingBatch): SubmittedBatch {
        const buffers = this.requireBuffers();
        const { k } = record;
        const { device } = this.ctx;
        if (this.fixedDirty) {
            device.queue.writeBuffer(buffers.fixed, 0, this.fixedWords);
            this.fixedDirty = false;
        }
        this.flushStateWrites(buffers.state);
        const first = this.ring.reserve(k);
        for (let i = 0; i < k; i++) {
            const flags = i === 0 && this.firstPending ? FA2_FLAG_FIRST : 0;
            this.ring.write(first + i, this.model.params, this.paramsForSlot(this.iterationsSubmitted + i, i, flags));
        }
        this.ring.flush();
        const batch = new CommandBatch(this.ctx, `${this.label}/batch`, this.generationValue);
        const beforeEpilogue = this.lastIterationStage();
        for (let i = 0; i < k; i++) {
            // the last iteration records every stage (toScene included); the others stop before the epilogue
            this.model.recordIteration(batch, first + i, this.tierValue, i === k - 1 ? undefined : beforeEpilogue);
        }
        batch.endPass();
        const scene = batch.readback(buffers.scene, 0, 12 * this.n);
        const state = batch.readback(buffers.state, 0, STATE_HEADER_BYTES + k * this.traceBytes);
        const { profiler } = this.ctx;
        const profile = profiler === null ? null : profiler.resolveInto(batch);
        const submitted = batch.submit();
        record.submitted = submitted;
        record.id = submitted.id;
        record.sceneOffset = scene.offset;
        record.stateOffset = state.offset;
        record.profile = profile;
        record.startedAt = performance.now();
        this.lastSubmittedBatchIdValue = submitted.id;
        this.iterationsSubmitted += k;
        this.firstPending = false;
        return submitted;
    }

    /**
     * A landed readback (spec 7.19 item 6): stale -> discard; else the scene bytes go into the owner's array row by
     * row (skipping overridden rows, clearing overrides older than this batch), the header refreshes the shadow,
     * the stats are decoded through the model (msPerIteration from the profiler or the wall time, PLAN DECISION 3),
     * iterationsDone += k and settled is recomputed -- with the landed settledCount taken as 0 when the batch was
     * submitted before the last reheat() (PLAN DECISION 21; the header is still absorbed and the stats decoded).
     * @param record - the pending batch
     * @param bytes - the batch's readback bytes
     */
    private onLanded(record: PendingBatch, bytes: ArrayBuffer): void {
        this.finish(record);
        const { owner } = this;
        if (this.isStale(record) || owner === null) {
            record.resolve();
            return;
        }
        try {
            const { k } = record;
            const { n } = this;
            const needed = Math.max(
                record.sceneOffset + 12 * n,
                record.stateOffset + STATE_HEADER_BYTES + k * this.traceBytes,
            );
            if (bytes.byteLength < needed) {
                // a discarded readback resolves with an empty buffer
                record.resolve();
                return;
            }
            this.copyScene(owner, bytes, record);
            const header = new DataView(bytes, record.stateOffset, STATE_HEADER_BYTES);
            const trace = new DataView(bytes, record.stateOffset + STATE_HEADER_BYTES, k * this.traceBytes);
            this.writer.absorb(header);
            const settledCount = this.model.state.readField(header, "settledCount");
            if (typeof settledCount !== "number") {
                throw invalidArgument("settledCount", settledCount, "a scalar", "settledCount must be a scalar field");
            }
            const ms = this.batchMilliseconds(record, bytes);
            this.statsValue = withMs(this.model.readStats(header, trace), ms / k);
            this.iterationsDoneValue += k;
            // PLAN DECISION 21: a batch in flight at reheat() carries the pre-reheat settle counter
            this.settledCountValue = record.id <= this.reheatedAfterBatchId ? 0 : settledCount;
            this.settledValue = this.computeSettled();
            record.resolve();
        } catch (err) {
            record.reject(asError(err));
        }
    }

    /**
     * A failed readback: a discarded batch's E_ABORTED resolves (PLAN DECISION 8); a lost device rejects
     * E_DEVICE_LOST; everything else (E_VALIDATION, E_DISPOSED) rejects as is.
     * @param record - the pending batch
     * @param err - the rejection
     */
    private onFailed(record: PendingBatch, err: unknown): void {
        this.finish(record);
        if (hasErrorCode(err, "E_ABORTED") && this.isStale(record)) {
            record.resolve();
            return;
        }
        if (this.ctx.state === "lost" && !hasErrorCode(err, "E_DEVICE_LOST")) {
            record.reject(
                new WebGpuGraphError("E_DEVICE_LOST", "the device was lost while the batch was in flight", {
                    reason: "unknown",
                    message: asError(err).message,
                }),
            );
            return;
        }
        record.reject(asError(err));
    }

    /**
     * Copies the scene bytes of a landed batch into the owner's array, honouring the override list (spec 7.12).
     * @param owner - the owner's array
     * @param bytes - the readback
     * @param record - the batch
     */
    private copyScene(owner: F32, bytes: ArrayBuffer, record: PendingBatch): void {
        const { n } = this;
        const scene = new Float32Array(bytes, record.sceneOffset, 3 * n);
        if (this.overrideList.size === 0) {
            owner.set(scene);
            return;
        }
        for (let i = 0; i < n; i++) {
            const after = this.overrideList.get(i);
            if (after !== undefined && after >= record.id) {
                continue;
            }
            owner[3 * i] = scene[3 * i];
            owner[3 * i + 1] = scene[3 * i + 1];
            owner[3 * i + 2] = scene[3 * i + 2];
        }
        for (const [i, after] of this.overrideList) {
            if (after < record.id) {
                this.overrideList.delete(i);
            }
        }
    }

    /**
     * The batch's duration in milliseconds: the profiler's pass timings summed when present, else wall time.
     * @param record - the batch
     * @param bytes - its readback (the profiler's resolve lands in it)
     * @returns milliseconds
     */
    private batchMilliseconds(record: PendingBatch, bytes: ArrayBuffer): number {
        const { profiler } = this.ctx;
        if (profiler !== null && record.profile !== null) {
            const timings = profiler.timings(bytes, record.profile);
            if (timings.length > 0) {
                let ns = 0;
                for (const timing of timings) {
                    ns += timing.ns;
                }
                return ns / 1e6;
            }
        }
        return performance.now() - record.startedAt;
    }

    /**
     * Whether a record belongs to an earlier generation, was discarded, or the simulation left "loaded".
     * @param record - the batch
     * @returns true when its readback must be ignored
     */
    private isStale(record: PendingBatch): boolean {
        return record.stale || this.stateValue !== "loaded" || record.generation !== this.generationValue;
    }

    /**
     * Removes a record from the pending list (idempotent).
     * @param record - the batch
     */
    private finish(record: PendingBatch): void {
        const at = this.pending.indexOf(record);
        if (at >= 0) {
            this.pending.splice(at, 1);
        }
    }

    /** Marks every pending batch stale and discards the submitted ones (their readbacks resolve empty or E_ABORTED). */
    private discardPending(): void {
        for (const record of this.pending) {
            record.stale = true;
            record.submitted?.discard();
        }
    }

    /** An OOM surfaced by allocator.check(): every simulation buffer is destroyed and the state returns to "created". */
    private outOfMemory(): void {
        this.discardPending();
        this.destroyBuffers();
        this.ctx.allocator.reset();
        this.settledValue = false;
        this.stateValue = "created";
    }

    /**
     * The E_ABORTED of run().
     * @returns the error
     */
    private abortedError(): WebGpuGraphError {
        return new WebGpuGraphError("E_ABORTED", `${this.label}: run() aborted by the signal`, {
            batchId: this.lastSubmittedBatchIdValue,
        });
    }

    // ---------------------------------------------------------------- private: params, state writes, toScene

    /**
     * The uniform values of one iteration slot: the model's values with the shared fields on top (PLAN DECISION 6).
     * @param global - the global iteration index (iterationsSubmitted + i)
     * @param index - the slot index within the batch (the trace slot)
     * @param flags - FA2_FLAG_FIRST for the first iteration after load(), else 0
     * @returns the values
     */
    private paramsForSlot(global: number, index: number, flags: number): UniformValues {
        const [cx, cy, cz] = this.center;
        const shared: UniformValues = {
            n: this.n,
            dim: this.dimValue,
            flags,
            iterationIndex: index,
            seed: this.seedU32(),
            scale: this.scale,
            center: [cx, cy, cz, 0],
            settleThreshold: this.settleThreshold(),
        };
        return { ...this.model.paramsFor(global, this.optionsValue), ...shared };
    }

    /**
     * The option seed as a u32 (0 when unseeded), the near-field hash seed of P4.
     * @returns the seed
     */
    private seedU32(): number {
        const { seed } = this.optionsValue;
        if (seed === null || seed === undefined || !Number.isFinite(seed)) {
            return 0;
        }
        return Math.floor(Math.abs(seed)) % 4294967296;
    }

    /**
     * Writes the queued state fields, each as one writeBuffer of its own byte range (PLAN DECISION 5).
     * @param state - the state buffer
     */
    private flushStateWrites(state: GPUBuffer): void {
        const fields = this.writer.takeQueued();
        if (fields.length === 0) {
            return;
        }
        const bytes = this.writer.headerBytes();
        const block = this.model.state;
        for (const field of fields) {
            const offset = block.offsetOf(field);
            const size = 4 * fieldWidth(fieldTypeOf(block, field));
            this.ctx.device.queue.writeBuffer(state, offset, bytes, offset, size);
        }
    }

    /**
     * The stage that precedes the toScene epilogue in model.stages (the `upTo` of every iteration but the last), or
     * undefined when the model has no epilogue or lists it first (every iteration then records every stage).
     * @returns the stage name or undefined
     */
    private lastIterationStage(): string | undefined {
        const at = this.model.stages.indexOf(EPILOGUE_STAGE);
        if (at <= 0) {
            return undefined;
        }
        return this.model.stages[at - 1];
    }

    /**
     * The settle threshold of the current options (FA2_DEFAULTS when absent).
     * @returns the threshold
     */
    private settleThreshold(): number {
        return this.optionsValue.settleThreshold ?? FA2_DEFAULTS.settleThreshold;
    }

    /**
     * The settle window of the current options (FA2_DEFAULTS when absent).
     * @returns the window
     */
    private settleWindow(): number {
        return this.optionsValue.settleWindow ?? FA2_DEFAULTS.settleWindow;
    }

    /**
     * The iteration budget (PLAN DECISION 1): options.maxIter, else options.iterations, else no budget.
     * @returns the budget
     */
    private iterationBudget(): number {
        const maxIter = optionField(this.optionsValue, "maxIter");
        if (typeof maxIter === "number") {
            return maxIter;
        }
        const iterations = optionField(this.optionsValue, "iterations");
        if (typeof iterations === "number") {
            return iterations;
        }
        return Number.POSITIVE_INFINITY;
    }

    /**
     * The settle rule of spec 7.17 over the counters of the last completed batch.
     * @returns whether the layout is settled
     */
    private computeSettled(): boolean {
        if (this.stateValue !== "loaded") {
            return false;
        }
        return (
            this.n === 0 ||
            this.iterationsDoneValue >= this.iterationBudget() ||
            this.settledCountValue >= this.settleWindow()
        );
    }

    // ---------------------------------------------------------------- private: buffers, upload, bind

    /**
     * Allocates the shared buffers (spec 7.3 sizes) and the model's BufferSpecs through the allocator (every buffer
     * labelled); a `zero: true` spec must carry COPY_DST so it can be re-zeroed when kept across loads (PLAN
     * DECISION 17). The state buffer is STATE_HEADER_BYTES + MAX_ITERATIONS_PER_STEP x model.trace.byteLength:
     * contract 3.13 writes TRACE_RECORD_BYTES (32, the FA2 record) for the last factor; the model's block is the
     * same number for FA2 and is what the simulation decodes the trace with, so it is the general rule (PLAN
     * DECISION 4) and the constant is not imported here.
     * @param n - the node count (> 0)
     * @returns the buffers and the name -> Binding map
     */
    private allocate(n: number): SimulationBuffers {
        const { allocator } = this.ctx;
        const { label } = this;
        const groups = Math.ceil(n / this.ctx.workgroupSize);
        const traceRegionBytes = MAX_ITERATIONS_PER_STEP * this.traceBytes;
        const storageRw = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;
        const positions = allocator.createBuffer({ label: `${label}/positions`, size: 16 * n, usage: storageRw });
        const scene = allocator.createBuffer({
            label: `${label}/scenePositions`,
            size: 12 * n,
            usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC,
        });
        const fixed = allocator.createBuffer({
            label: `${label}/fixed`,
            size: 4 * Math.ceil(n / 32),
            usage: storageRw,
        });
        const partials = allocator.createBuffer({
            label: `${label}/partials`,
            size: PARTIAL_BYTES * groups,
            usage: storageRw,
        });
        const state = allocator.createBuffer({
            label: `${label}/state`,
            size: STATE_HEADER_BYTES + traceRegionBytes,
            usage: storageRw,
        });
        const bindings = new Map<string, Binding>();
        bindings.set("positions", wholeBinding(positions, 16 * n));
        bindings.set("scenePositions", wholeBinding(scene, 12 * n));
        bindings.set("fixed", wholeBinding(fixed, 4 * Math.ceil(n / 32)));
        bindings.set("partials", wholeBinding(partials, PARTIAL_BYTES * groups));
        bindings.set("state", { buffer: state, offset: 0, size: STATE_HEADER_BYTES, window: null });
        bindings.set("trace", { buffer: state, offset: STATE_HEADER_BYTES, size: traceRegionBytes, window: null });
        const model = new Map<string, { readonly buffer: GPUBuffer; readonly spec: BufferSpec }>();
        for (const spec of this.model.buffers(n, this.dimValue)) {
            if (SHARED_BUFFER_NAMES.includes(spec.name) || bindings.has(spec.name)) {
                throw invalidArgument(
                    "BufferSpec.name",
                    spec.name,
                    "a name no other buffer uses",
                    `BufferSpec "${spec.name}" reuses a buffer name`,
                );
            }
            if (!Number.isInteger(spec.byteLength) || spec.byteLength <= 0 || spec.byteLength % 4 !== 0) {
                throw invalidArgument(
                    "BufferSpec.byteLength",
                    spec.byteLength,
                    "a positive multiple of 4",
                    `BufferSpec "${spec.name}" has byteLength ${spec.byteLength}`,
                );
            }
            if (spec.zero && (spec.usage & BufferUsage.COPY_DST) === 0) {
                throw invalidArgument(
                    "BufferSpec.usage",
                    spec.usage,
                    "COPY_DST on a zero: true spec",
                    `BufferSpec "${spec.name}" is zero: true but lacks COPY_DST`,
                );
            }
            const buffer = allocator.createBuffer({
                label: `${label}/${spec.name}`,
                size: spec.byteLength,
                usage: spec.usage,
            });
            model.set(spec.name, { buffer, spec });
            bindings.set(spec.name, wholeBinding(buffer, spec.byteLength));
        }
        return { positions, scene, fixed, partials, state, model, bindings, traceRegionBytes };
    }

    /**
     * Re-zeroes the buffers a same-size load() keeps: partials, the trace region and every `zero: true` model buffer.
     * @param buffers - the kept buffers
     */
    private clearKept(buffers: SimulationBuffers): void {
        const { device } = this.ctx;
        const encoder = device.createCommandEncoder({ label: `${this.label}/clear` });
        encoder.clearBuffer(buffers.partials);
        encoder.clearBuffer(buffers.state, STATE_HEADER_BYTES, buffers.traceRegionBytes);
        for (const { buffer, spec } of buffers.model.values()) {
            if (spec.zero) {
                encoder.clearBuffer(buffer);
            }
        }
        device.queue.submit([encoder.finish()]);
    }

    /**
     * Destroys every simulation buffer through the allocator (the ring lives on until dispose()). While batches are
     * still in flight the destruction is DEFERRED until their readbacks have settled: the GPU may still be copying
     * out of `scene` / `state` and Dawn's Metal backend (dawn-node 0.4.0 on macOS) has taken the worker process
     * down when a buffer with pending work was destroyed (the Vulkan backends defer internally). The simulation
     * drops its references at once either way, so nothing here is reachable afterwards.
     */
    private destroyBuffers(): void {
        const { buffers } = this;
        if (buffers === null) {
            return;
        }
        this.buffers = null;
        this.resources = null;
        const doomed = [buffers.positions, buffers.scene, buffers.fixed, buffers.partials, buffers.state];
        for (const { buffer } of buffers.model.values()) {
            doomed.push(buffer);
        }
        this.afterInFlight(() => {
            const { allocator } = this.ctx;
            for (const buffer of doomed) {
                allocator.destroy(buffer);
            }
        });
    }

    /**
     * Runs `action` now when no submitted batch is in flight, otherwise once every in-flight readback has settled
     * (resolved, discarded or rejected -- the staging slot is returned and the GPU work is done either way).
     * @param action - the destruction to run
     */
    private afterInFlight(action: () => void): void {
        const waits: Promise<unknown>[] = [];
        for (const record of this.pending) {
            if (record.submitted !== null) {
                waits.push(record.submitted.readback.catch(() => undefined));
            }
        }
        if (waits.length === 0) {
            action();
            return;
        }
        void Promise.all(waits).then(action, action);
    }

    /**
     * The buffers of the current load.
     * @returns the buffers; E_NOT_LOADED when there are none
     */
    private requireBuffers(): SimulationBuffers {
        const { buffers } = this;
        if (buffers === null) {
            throw new WebGpuGraphError("E_NOT_LOADED", `${this.label}: load() first`, { state: this.stateValue });
        }
        return buffers;
    }

    /**
     * Repacks the seeded scene array into layout-unit vec4f (mass in .w, z = 0 in 2D), uploads it, and writes the
     * initial centroid / bbox / rmsRadius / radius (f64 over the f32 values uploaded) with iteration 0,
     * settledCount 0, meanDisplacement 0 into the state header, then model.onLoad(writer) and one whole-header
     * writeBuffer (spec 7.4, 7.17, 7.18).
     * @param buffers - the buffers
     * @param positions - the seeded owner array
     * @param mass - the resolved masses
     */
    private uploadPositions(buffers: SimulationBuffers, positions: F32, mass: F32): void {
        const { n, scale } = this;
        const [cx, cy, cz] = this.center;
        const packed = new Float32Array(4 * n);
        for (let i = 0; i < n; i++) {
            packed[4 * i] = (positions[3 * i] - cx) / scale;
            packed[4 * i + 1] = (positions[3 * i + 1] - cy) / scale;
            packed[4 * i + 2] = this.dimValue === 2 ? 0 : (positions[3 * i + 2] - cz) / scale;
            packed[4 * i + 3] = mass[i];
        }
        this.ctx.device.queue.writeBuffer(buffers.positions, 0, packed);
        let sx = 0;
        let sy = 0;
        let sz = 0;
        const lo = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
        const hi = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
        for (let i = 0; i < n; i++) {
            const x = packed[4 * i];
            const y = packed[4 * i + 1];
            const z = packed[4 * i + 2];
            sx += x;
            sy += y;
            sz += z;
            lo[0] = Math.min(lo[0], x);
            lo[1] = Math.min(lo[1], y);
            lo[2] = Math.min(lo[2], z);
            hi[0] = Math.max(hi[0], x);
            hi[1] = Math.max(hi[1], y);
            hi[2] = Math.max(hi[2], z);
        }
        const mx = sx / n;
        const my = sy / n;
        const mz = sz / n;
        let sumSq = 0;
        let maxSq = 0;
        for (let i = 0; i < n; i++) {
            const dx = packed[4 * i] - mx;
            const dy = packed[4 * i + 1] - my;
            const dz = packed[4 * i + 2] - mz;
            const q = dx * dx + dy * dy + dz * dz;
            sumSq += q;
            maxSq = Math.max(maxSq, q);
        }
        const { writer } = this;
        writer.set("iteration", 0);
        writer.set("settledCount", 0);
        writer.set("meanDisplacement", 0);
        writer.set("centroid", [mx, my, mz, 0]);
        writer.set("min", [lo[0], lo[1], lo[2], 0]);
        writer.set("max", [hi[0], hi[1], hi[2], 0]);
        writer.set("rmsRadius", Math.sqrt(sumSq / n));
        writer.set("radius", Math.sqrt(maxSq));
        this.model.onLoad(writer);
        this.ctx.device.queue.writeBuffer(buffers.state, 0, writer.headerBytes());
        writer.clearQueued();
    }

    /**
     * The weights binding of a load (3.10): "arcs" -> core.weights, "none" -> null, "column" -> the expanded array
     * registered with residency.array(expanded, "weights", snapshot), the previous array object reused when the
     * column and its version are unchanged (so the upload cache finds it).
     * @param snapshot - the snapshot
     * @param core - its core
     * @param resolved - model.inputs().weights
     * @returns the binding for the group-0 weights slot, or null
     */
    private resolveWeightsBinding(
        snapshot: GraphSnapshot,
        core: CoreBinding,
        resolved: ResolvedWeights,
    ): Binding | null {
        if (resolved.source === "arcs") {
            this.weightsUpload = null;
            return core.weights;
        }
        if (resolved.source === "none" || resolved.data === null || resolved.column === null) {
            this.weightsUpload = null;
            return null;
        }
        const previous = this.weightsUpload;
        const reusable =
            previous !== null &&
            previous.serial === snapshot.serial &&
            previous.column === resolved.column &&
            previous.version === resolved.column.version &&
            previous.data.length === resolved.data.length;
        const data = reusable && previous !== null ? previous.data : resolved.data;
        const upload = this.ctx.residency.array(data, "weights", snapshot);
        this.weightsUpload = {
            serial: snapshot.serial,
            column: resolved.column,
            version: resolved.column.version,
            data,
            upload,
        };
        return upload.binding;
    }

    /**
     * The ModelResources of a load.
     * @param core - the core
     * @param weights - the resolved weights binding
     * @param buffers - the buffers
     * @returns the resources
     */
    private makeResources(core: CoreBinding, weights: Binding | null, buffers: SimulationBuffers): ModelResources {
        const { bindings } = buffers;
        return {
            device: this.ctx.device,
            caps: this.ctx.caps,
            pipelines: this.ctx.pipelines,
            core,
            perm: null,
            weights,
            n: this.n,
            dim: this.dimValue,
            tier: this.tierValue,
            ring: this.ring,
            buffer: (name: string): Binding => {
                const binding = bindings.get(name);
                if (binding === undefined) {
                    throw invalidArgument("name", name, [...bindings.keys()], `no simulation buffer named "${name}"`);
                }
                return binding;
            },
        };
    }

    /**
     * Chains the compile + bind promise the next step() awaits on the PREVIOUS one (PLAN DECISION 20: binds are
     * serialised on `ready` for load() and setParams() alike, so a superseded load()'s bind either fails its
     * generation check before model.bind() or completes before its successor's bind starts -- it can never bind
     * the model to destroyed buffers AFTER the new bind finished). The previous promise's rejection is swallowed by
     * the chain (it belonged to the superseded generation); this promise's rejection is kept for the next step()
     * (the derived catch only silences the unhandled-rejection warning).
     * @param resources - the resources
     * @param overrides - the merged override set
     */
    private startBind(resources: ModelResources, overrides: Readonly<Record<string, number | boolean>>): void {
        const generation = this.generationValue;
        const run = (): Promise<void> => this.compileAndBind(resources, overrides, generation);
        const promise = this.ready.catch(() => undefined).then(run);
        void promise.catch(() => undefined);
        this.ready = promise;
    }

    /**
     * warm(model.specs()) then model.bind(); the await re-checks that the load is current before binding.
     * @param resources - the resources
     * @param overrides - the merged override set
     * @param generation - the generation the bind belongs to
     */
    private async compileAndBind(
        resources: ModelResources,
        overrides: Readonly<Record<string, number | boolean>>,
        generation: number,
    ): Promise<void> {
        const subgroups = this.ctx.caps.features.has("subgroups");
        await this.ctx.pipelines.warm(this.model.specs(overrides, subgroups));
        if (this.stateValue !== "loaded" || generation !== this.generationValue) {
            return;
        }
        await this.model.bind(resources, overrides);
    }

    // ---------------------------------------------------------------- private: inspect and debug runs

    /**
     * inspect(name): flush, read the named buffer back, Uint32Array for "fixed" and "trace", Float32Array otherwise.
     * @param name - a shared or BufferSpec name
     * @returns the words
     */
    private async inspectBuffer(name: string): Promise<Float32Array | Uint32Array> {
        this.assertLoaded();
        const { buffers } = this;
        const binding = buffers?.bindings.get(name);
        if (buffers === null || binding === undefined) {
            throw invalidArgument(
                "name",
                name,
                buffers === null ? [] : [...buffers.bindings.keys()],
                `inspect("${name}"): no such buffer`,
            );
        }
        await this.flush();
        const bytes = await this.ctx.readback.read(binding.buffer, binding.size, undefined, binding.offset);
        return name === "fixed" || name === "trace" ? new Uint32Array(bytes) : new Float32Array(bytes);
    }

    /**
     * debugRunStages(upTo): one iteration truncated after a stage, submitted alone after flush(), awaited through
     * a header readback; never advances the counters or the FA2_FLAG_FIRST flag (PLAN DECISION 9).
     * @param upTo - a model stage name
     */
    private async runStages(upTo: string): Promise<void> {
        this.assertLoaded();
        if (!this.model.stages.includes(upTo)) {
            throw invalidArgument(
                "upTo",
                upTo,
                [...this.model.stages],
                `debugRunStages("${upTo}"): not a stage of ${this.model.kind}`,
            );
        }
        if (this.n === 0) {
            return;
        }
        await this.flush();
        await this.ready;
        await this.ctx.allocator.check();
        this.assertLoaded();
        const buffers = this.requireBuffers();
        const { device } = this.ctx;
        if (this.fixedDirty) {
            device.queue.writeBuffer(buffers.fixed, 0, this.fixedWords);
            this.fixedDirty = false;
        }
        this.flushStateWrites(buffers.state);
        const slot = this.ring.reserve(1);
        this.ring.write(
            slot,
            this.model.params,
            this.paramsForSlot(this.iterationsSubmitted, 0, this.firstPending ? FA2_FLAG_FIRST : 0),
        );
        this.ring.flush();
        const batch = new CommandBatch(this.ctx, `${this.label}/debug`, this.generationValue);
        this.model.recordIteration(batch, slot, this.tierValue, upTo);
        batch.endPass();
        batch.readback(buffers.state, 0, STATE_HEADER_BYTES);
        const submitted = batch.submit();
        this.lastSubmittedBatchIdValue = submitted.id;
        await submitted.readback;
    }

    // ---------------------------------------------------------------- private: lifecycle

    /**
     * Device loss (spec 5.7): every pending promise rejects E_DEVICE_LOST, the simulation is disposed.
     * @param info - the loss info
     */
    private onDeviceLost(info: GPUDeviceLostInfo): void {
        if (this.stateValue === "disposed") {
            return;
        }
        const error = new WebGpuGraphError("E_DEVICE_LOST", `device lost (${info.reason}): ${info.message}`, {
            reason: info.reason,
            message: info.message,
        });
        const pending = this.pending.splice(0);
        this.stateValue = "disposed";
        for (const record of pending) {
            record.stale = true;
            record.reject(error);
        }
        this.teardown();
    }

    /** Releases everything once: pending batches discarded, buffers and ring destroyed, pool trimmed, listener unregistered. */
    private teardown(): void {
        if (this.torndown) {
            return;
        }
        this.torndown = true;
        this.discardPending();
        this.destroyBuffers();
        this.afterInFlight(() => {
            this.ring.destroy();
        });
        if (this.ctx.state === "ready") {
            this.ctx.pool.trim();
        }
        this.unregisterLost();
        this.core = null;
        this.snapshot = null;
        this.owner = null;
        this.weightsUpload = null;
    }

    /** E_DISPOSED after dispose() or device loss. */
    private assertNotDisposed(): void {
        if (this.stateValue === "disposed") {
            throw disposedError(this.label);
        }
    }

    /** E_NOT_LOADED before load(), E_DISPOSED after dispose(). */
    private assertLoaded(): void {
        if (this.stateValue === "created") {
            throw new WebGpuGraphError("E_NOT_LOADED", `${this.label}: load() first`, { state: "created" });
        }
        this.assertNotDisposed();
    }
}
