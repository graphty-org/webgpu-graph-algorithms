import { type GraphSnapshot } from "@graphty/graph-format";
import {
    type AcceleratorOptions,
    type AlgorithmAccelerator,
    createAccelerator,
    type ForceAtlas2Options,
    type ForceAtlas2Stats,
    type GpuAccelerator,
    type GpuContext,
    type GpuLayoutSimulation,
    type GpuLayoutTuning,
    type LayoutAccelerator,
    type LayoutSimulation,
    type LayoutStatsBase,
    type RunOptions,
} from "@graphty/webgpu-graph-algorithms";
import { expectTypeOf } from "vitest";

// Compiled twice and never executed: by `tsc --noEmit -p tsconfig.json` against src/ and by
// `tsc -p tsconfig.strict-consumer.json` against dist/*.d.ts with noUncheckedIndexedAccess and
// exactOptionalPropertyTypes ON (spec 11.3 row "Type-level"). `declare const` stands in for every value.

declare const ctx: GpuContext;
type Fa2Sim = GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;

// ---- forward direction (spec 9.1, D27): the accelerator satisfies both CPU-side interfaces structurally
expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>();
expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator>();
expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<LayoutAccelerator>();
expectTypeOf(createAccelerator(ctx)).toEqualTypeOf<GpuAccelerator>();
expectTypeOf<GpuAccelerator>().toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>();

// injection needs no cast (spec 9.4: the element's `accelerator` property is typed by the CPU packages' interfaces)
const injectedLayout: LayoutAccelerator = createAccelerator(ctx);
const injectedAlgorithms: AlgorithmAccelerator = createAccelerator(ctx);
expectTypeOf(injectedLayout.kind).toBeString();
expectTypeOf(injectedAlgorithms.kind).toBeString();

// ---- reverse direction: the mirrors are strictly wider (kind: string, every member optional), so a bare mirror
// is NOT a GpuAccelerator ...
expectTypeOf<AlgorithmAccelerator & LayoutAccelerator>().not.toMatchTypeOf<GpuAccelerator>();
expectTypeOf<LayoutAccelerator>().not.toMatchTypeOf<GpuAccelerator>();
expectTypeOf<AlgorithmAccelerator>().not.toMatchTypeOf<GpuAccelerator>();
// ... while every member the accelerator DOES declare fits the mirror's optional slot for it
expectTypeOf<GpuAccelerator["kind"]>().toEqualTypeOf<"webgpu">();
expectTypeOf<GpuAccelerator["kind"]>().toMatchTypeOf<LayoutAccelerator["kind"]>();
expectTypeOf<GpuAccelerator["kind"]>().toMatchTypeOf<AlgorithmAccelerator["kind"]>();
expectTypeOf<GpuAccelerator["forceAtlas2"]>().toMatchTypeOf<NonNullable<LayoutAccelerator["forceAtlas2"]>>();
expectTypeOf<GpuAccelerator["release"]>().toMatchTypeOf<NonNullable<LayoutAccelerator["release"]>>();
expectTypeOf<GpuAccelerator["release"]>().toMatchTypeOf<NonNullable<AlgorithmAccelerator["release"]>>();
expectTypeOf<GpuAccelerator["dispose"]>().toMatchTypeOf<NonNullable<LayoutAccelerator["dispose"]>>();
expectTypeOf<GpuAccelerator["dispose"]>().toMatchTypeOf<NonNullable<AlgorithmAccelerator["dispose"]>>();

// the 9.2 / 9.3 dispatcher shape: an optional member is tested for undefined and then called without a cast
declare const injected: LayoutAccelerator;
const routed = injected.forceAtlas2 !== undefined ? injected.forceAtlas2({ maxIter: 10 }) : null;
expectTypeOf(routed).toEqualTypeOf<LayoutSimulation | null>();

// ---- the mirrors are the spec 9.2 / 9.3 declarations verbatim (member lists pinned; at W1 the mirrors become
// `import type` of the real packages and these lines must still compile)
expectTypeOf<keyof LayoutSimulation>().toEqualTypeOf<
    "load" | "step" | "settled" | "setFixed" | "setPosition" | "dispose"
>();
expectTypeOf<keyof LayoutAccelerator>().toEqualTypeOf<
    "kind" | "forceAtlas2" | "fruchtermanReingold" | "springElectrical" | "release" | "dispose"
>();
expectTypeOf<keyof AlgorithmAccelerator>().toEqualTypeOf<
    | "kind"
    | "pageRank"
    | "personalizedPageRank"
    | "hits"
    | "eigenvectorCentrality"
    | "katzCentrality"
    | "connectedComponents"
    | "weaklyConnectedComponents"
    | "breadthFirstSearch"
    | "sssp"
    | "bellmanFord"
    | "closenessCentrality"
    | "betweennessCentrality"
    | "edgeBetweennessCentrality"
    | "allPairsShortestPath"
    | "kCoreDecomposition"
    | "triangleCount"
    | "labelPropagation"
    | "minimumSpanningTree"
    | "louvain"
    | "release"
    | "dispose"
>();
expectTypeOf<keyof GpuAccelerator>().toEqualTypeOf<
    keyof AlgorithmAccelerator | keyof LayoutAccelerator | "ctx" | "options"
>();
expectTypeOf<LayoutAccelerator["kind"]>().toBeString();
expectTypeOf<LayoutSimulation["step"]>().returns.toEqualTypeOf<void | Promise<void>>();
expectTypeOf<LayoutSimulation["step"]>().parameter(0).toEqualTypeOf<number | undefined>();
expectTypeOf<LayoutSimulation["settled"]>().toBeBoolean();

// ---- GpuLayoutSimulation extends LayoutSimulation (spec 3.3): a narrower step(), the GPU additions on top
expectTypeOf<Fa2Sim>().toMatchTypeOf<LayoutSimulation>();
expectTypeOf<LayoutSimulation>().not.toMatchTypeOf<Fa2Sim>();
expectTypeOf(createAccelerator(ctx).forceAtlas2()).toEqualTypeOf<Fa2Sim>();
expectTypeOf(createAccelerator(ctx).forceAtlas2()).toMatchTypeOf<LayoutSimulation>();
expectTypeOf<Fa2Sim["step"]>().returns.toEqualTypeOf<Promise<void>>();
expectTypeOf<Fa2Sim["step"]>().parameter(0).toEqualTypeOf<number | undefined>();
expectTypeOf<Fa2Sim["stats"]>().toEqualTypeOf<ForceAtlas2Stats>();
expectTypeOf<Fa2Sim["stats"]>().toMatchTypeOf<LayoutStatsBase>();
expectTypeOf<Fa2Sim["inFlight"]>().toBeNumber();
expectTypeOf<Fa2Sim["iterationsDone"]>().toBeNumber();
expectTypeOf<Fa2Sim["flush"]>().returns.toEqualTypeOf<Promise<void>>();
expectTypeOf<Fa2Sim["reheat"]>().returns.toBeVoid();
expectTypeOf<Fa2Sim["setParams"]>().parameter(0).toEqualTypeOf<Partial<ForceAtlas2Options>>();
expectTypeOf<Fa2Sim["run"]>().parameter(0).toEqualTypeOf<RunOptions | undefined>();
expectTypeOf<Fa2Sim["run"]>().returns.resolves.toEqualTypeOf<ForceAtlas2Stats>();
expectTypeOf<NonNullable<Fa2Sim["inspect"]>>().returns.resolves.toEqualTypeOf<Float32Array | Uint32Array>();

// ---- the accelerator's own members (spec 3.3 GpuAccelerator)
declare const acc: GpuAccelerator;
expectTypeOf(acc.kind).toEqualTypeOf<"webgpu">();
expectTypeOf(acc.ctx).toEqualTypeOf<GpuContext>();
expectTypeOf(acc.options).toEqualTypeOf<Readonly<AcceleratorOptions>>();
expectTypeOf(acc.options.layout).toEqualTypeOf<GpuLayoutTuning | undefined>();
expectTypeOf<Readonly<Pick<GpuAccelerator, "kind" | "ctx" | "options">>>().toEqualTypeOf<
    Pick<GpuAccelerator, "kind" | "ctx" | "options">
>();
expectTypeOf(acc.forceAtlas2).parameter(0).toEqualTypeOf<ForceAtlas2Options | undefined>();
expectTypeOf(acc.forceAtlas2).returns.toEqualTypeOf<Fa2Sim>();
expectTypeOf(acc.release).parameter(0).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(acc.release).returns.toBeVoid();
expectTypeOf(acc.dispose).returns.toBeVoid();
// members P3 does not implement keep the mirror's optional type -- the dispatchers read them as `undefined`
expectTypeOf(acc.pageRank).toEqualTypeOf<AlgorithmAccelerator["pageRank"]>();
expectTypeOf(acc.connectedComponents).toEqualTypeOf<AlgorithmAccelerator["connectedComponents"]>();
expectTypeOf(acc.fruchtermanReingold).toEqualTypeOf<LayoutAccelerator["fruchtermanReingold"]>();
expectTypeOf(acc.springElectrical).toEqualTypeOf<LayoutAccelerator["springElectrical"]>();
