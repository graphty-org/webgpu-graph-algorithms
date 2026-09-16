import { type F32, type NodeId, type NodeMask } from "@graphty/graph-format";
import {
    type AcceleratorOptions,
    type CommonLayoutOptions,
    createAccelerator,
    createForceAtlas2,
    type ForceAtlas2Options,
    type ForceAtlas2Stats,
    type FruchtermanReingoldOptions,
    type GpuContext,
    type GpuContextOptions,
    type GpuLayoutSimulation,
    type GpuLayoutTuning,
    type GpuRunOptions,
    type ProbeOptions,
    type RunOptions,
    type SimulationOptions,
    type SpringElectricalOptions,
} from "@graphty/webgpu-graph-algorithms";
import { expectTypeOf } from "vitest";

// Every option interface of the public surface accepts an explicit `undefined` in every optional field (spec 3.6:
// options are `?: T | undefined`), which is exactly what exactOptionalPropertyTypes forbids for a plain `?: T`.
// tsconfig.strict-consumer.json compiles this file with that flag ON against dist/*.d.ts (and tsconfig.json with it
// OFF against src/), so a field declared `?: T` anywhere in the surface turns `pnpm run lint` red.

declare const ctx: GpuContext;
declare const mass: F32;
declare const mask: NodeMask;
type Fa2Sim = GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;

// ---- layouts (spec 9.3 mirrors; contract 3.3 src/types/options.ts and src/types/layout.ts)
const common: CommonLayoutOptions = { dim: undefined, scale: undefined, center: undefined, seed: undefined };
const simulation: SimulationOptions = {
    settleThreshold: undefined,
    settleWindow: undefined,
    iterationsPerStep: undefined,
    maxInFlight: undefined,
};
const fa2: ForceAtlas2Options = {
    ...common,
    ...simulation,
    maxIter: undefined,
    jitterTolerance: undefined,
    scalingRatio: undefined,
    gravity: undefined,
    strongGravity: undefined,
    distributedAction: undefined,
    linlog: undefined,
    nodeMass: undefined,
    nodeSize: undefined,
    weight: undefined,
    dissuadeHubs: undefined,
};
const fr: FruchtermanReingoldOptions = {
    ...common,
    ...simulation,
    k: undefined,
    iterations: undefined,
    fixed: undefined,
};
const spring: SpringElectricalOptions = {
    ...common,
    ...simulation,
    springLength: undefined,
    springCoefficient: undefined,
    gravity: undefined,
    dragCoefficient: undefined,
    timeStep: undefined,
};
const tuning: GpuLayoutTuning = {
    repulsion: undefined,
    exactMaxNodes: undefined,
    nearMax: undefined,
    deterministic: undefined,
    gridMax2D: undefined,
    gridMax3D: undefined,
    extentFactor: undefined,
    compat: undefined,
};
const run: RunOptions = { maxIter: undefined, batch: undefined, signal: undefined };
expectTypeOf(common).toEqualTypeOf<CommonLayoutOptions>();
expectTypeOf(simulation).toEqualTypeOf<SimulationOptions>();
expectTypeOf(fa2).toEqualTypeOf<ForceAtlas2Options>();
expectTypeOf(fr).toEqualTypeOf<FruchtermanReingoldOptions>();
expectTypeOf(spring).toEqualTypeOf<SpringElectricalOptions>();
expectTypeOf(tuning).toEqualTypeOf<GpuLayoutTuning>();
expectTypeOf(run).toEqualTypeOf<RunOptions>();

// the populated forms, every field at its widest declared type
const fa2Full: ForceAtlas2Options = {
    dim: 3,
    scale: 2,
    center: [0, 0, 0],
    seed: 42,
    settleThreshold: 0.001,
    settleWindow: 10,
    iterationsPerStep: 4,
    maxInFlight: 2,
    maxIter: 100,
    jitterTolerance: 1,
    scalingRatio: 2,
    gravity: 1,
    strongGravity: false,
    distributedAction: false,
    linlog: false,
    nodeMass: mass,
    nodeSize: null,
    weight: "capacity",
    dissuadeHubs: false,
};
const fa2ByName: ForceAtlas2Options = { seed: null, nodeMass: "mass", weight: true, center: new Float32Array(3) };
const fa2ByRecord: ForceAtlas2Options = { nodeMass: { alice: 2, 7: 1 }, weight: null };
const frFull: FruchtermanReingoldOptions = { k: null, iterations: 50, fixed: mask };
const frByName: FruchtermanReingoldOptions = { k: 0.5, fixed: "pinned" };
expectTypeOf(fa2Full.nodeMass).toEqualTypeOf<F32 | string | Readonly<Record<NodeId, number>> | null | undefined>();
expectTypeOf(fa2ByName.weight).toEqualTypeOf<boolean | string | null | undefined>();
expectTypeOf(fa2ByRecord.nodeSize).toEqualTypeOf<F32 | string | Readonly<Record<NodeId, number>> | null | undefined>();
expectTypeOf(frFull.fixed).toEqualTypeOf<NodeMask | string | null | undefined>();
expectTypeOf(frByName.k).toEqualTypeOf<number | null | undefined>();
expectTypeOf<ForceAtlas2Options["dim"]>().toEqualTypeOf<2 | 3 | undefined>();
expectTypeOf<ForceAtlas2Options["center"]>().toEqualTypeOf<ArrayLike<number> | undefined>();
expectTypeOf<ForceAtlas2Options["seed"]>().toEqualTypeOf<number | null | undefined>();
expectTypeOf<GpuLayoutTuning["repulsion"]>().toEqualTypeOf<"exact" | "grid" | "auto" | undefined>();
expectTypeOf<GpuLayoutTuning["compat"]>().toEqualTypeOf<"paper" | "networkx" | undefined>();

// the member lists are spec 9.3 / 3.3 verbatim
expectTypeOf<keyof CommonLayoutOptions>().toEqualTypeOf<"dim" | "scale" | "center" | "seed">();
expectTypeOf<keyof SimulationOptions>().toEqualTypeOf<
    "settleThreshold" | "settleWindow" | "iterationsPerStep" | "maxInFlight"
>();
expectTypeOf<keyof ForceAtlas2Options>().toEqualTypeOf<
    | keyof CommonLayoutOptions
    | keyof SimulationOptions
    | "maxIter"
    | "jitterTolerance"
    | "scalingRatio"
    | "gravity"
    | "strongGravity"
    | "distributedAction"
    | "linlog"
    | "nodeMass"
    | "nodeSize"
    | "weight"
    | "dissuadeHubs"
>();
expectTypeOf<keyof FruchtermanReingoldOptions>().toEqualTypeOf<
    keyof CommonLayoutOptions | keyof SimulationOptions | "k" | "iterations" | "fixed"
>();
expectTypeOf<keyof SpringElectricalOptions>().toEqualTypeOf<
    | keyof CommonLayoutOptions
    | keyof SimulationOptions
    | "springLength"
    | "springCoefficient"
    | "gravity"
    | "dragCoefficient"
    | "timeStep"
>();
expectTypeOf<keyof GpuLayoutTuning>().toEqualTypeOf<
    "repulsion" | "exactMaxNodes" | "nearMax" | "deterministic" | "gridMax2D" | "gridMax3D" | "extentFactor" | "compat"
>();
expectTypeOf<keyof RunOptions>().toEqualTypeOf<"maxIter" | "batch" | "signal">();

// ---- ForceAtlas2Options & GpuLayoutTuning is what createForceAtlas2 takes (spec 3.3); the accelerator method
// takes the CPU option type ONLY -- GPU tuning reaches it through createAccelerator's options.layout (spec 9.2)
const merged: ForceAtlas2Options & GpuLayoutTuning = { ...fa2Full, ...tuning, compat: "networkx", exactMaxNodes: 4096 };
expectTypeOf(createForceAtlas2).parameter(1).toEqualTypeOf<(ForceAtlas2Options & GpuLayoutTuning) | undefined>();
expectTypeOf(createForceAtlas2(ctx, merged)).toEqualTypeOf<Fa2Sim>();
expectTypeOf(createForceAtlas2(ctx, fa2)).toEqualTypeOf<Fa2Sim>();
expectTypeOf(createForceAtlas2(ctx, tuning)).toEqualTypeOf<Fa2Sim>();
expectTypeOf(createForceAtlas2(ctx)).toEqualTypeOf<Fa2Sim>();
expectTypeOf(createForceAtlas2(ctx).setParams).parameter(0).toEqualTypeOf<Partial<ForceAtlas2Options>>();
expectTypeOf(createForceAtlas2(ctx).run).parameter(0).toEqualTypeOf<RunOptions | undefined>();
expectTypeOf(createForceAtlas2(ctx).run(run)).resolves.toEqualTypeOf<ForceAtlas2Stats>();

// ---- the accelerator and context option records
const accelerator: AcceleratorOptions = { layout: undefined, algorithms: undefined };
const acceleratorDeep: AcceleratorOptions = {
    layout: tuning,
    algorithms: { betweenness: { k: undefined, sources: undefined } },
};
const acceleratorEmptyAlgorithms: AcceleratorOptions = { algorithms: {} };
expectTypeOf(accelerator).toEqualTypeOf<AcceleratorOptions>();
expectTypeOf(acceleratorDeep.algorithms).toEqualTypeOf<AcceleratorOptions["algorithms"]>();
expectTypeOf(acceleratorEmptyAlgorithms.layout).toEqualTypeOf<GpuLayoutTuning | undefined>();
expectTypeOf<keyof AcceleratorOptions>().toEqualTypeOf<"layout" | "algorithms">();
expectTypeOf(createAccelerator).parameter(1).toEqualTypeOf<AcceleratorOptions | undefined>();
expectTypeOf(createAccelerator(ctx, acceleratorDeep).forceAtlas2)
    .parameter(0)
    .toEqualTypeOf<ForceAtlas2Options | undefined>();

const gpuRun: GpuRunOptions = { dest: undefined, signal: undefined, onProgress: undefined };
expectTypeOf(gpuRun.dest).toEqualTypeOf<Float32Array | Uint32Array | undefined>();
expectTypeOf<keyof GpuRunOptions>().toEqualTypeOf<"dest" | "signal" | "onProgress">();

const context: GpuContextOptions = {
    gpu: undefined,
    adapter: undefined,
    device: undefined,
    powerPreference: undefined,
    rejectSoftware: undefined,
    limits: undefined,
    optionalFeatures: undefined,
    requiredFeatures: undefined,
    label: undefined,
    onError: undefined,
    warnUnreleasedSnapshots: undefined,
    runtime: undefined,
};
const probe: ProbeOptions = { gpu: undefined, powerPreference: undefined, rejectSoftware: undefined };
expectTypeOf(context.limits).toEqualTypeOf<GpuContextOptions["limits"]>();
expectTypeOf(probe.gpu).toEqualTypeOf<GPU | undefined>();
expectTypeOf<keyof GpuContextOptions>().toEqualTypeOf<
    | "gpu"
    | "adapter"
    | "device"
    | "powerPreference"
    | "rejectSoftware"
    | "limits"
    | "optionalFeatures"
    | "requiredFeatures"
    | "label"
    | "onError"
    | "warnUnreleasedSnapshots"
    | "runtime"
>();
expectTypeOf<keyof ProbeOptions>().toEqualTypeOf<"gpu" | "powerPreference" | "rejectSoftware">();
