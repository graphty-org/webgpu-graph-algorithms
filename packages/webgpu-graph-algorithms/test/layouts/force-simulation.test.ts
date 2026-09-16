/**
 * The shared layout state machine (spec 7.19, 7.12, 7.17; contract 3.13 ForceSimulation) driven by a FAKE
 * ForceModel whose kernels are the registry's `fill` body, so every rule is testable without ForceAtlas2:
 * transitions and every error code, coalescing at maxInFlight, the uniform-ring invariant across a wrap, generation
 * discard on load(), the setPosition override list, settle at maxIter and at settleWindow, reheat, flush, run() with
 * a signal, inspect / debugRunStages, dispose leak 0 and device loss.
 */

import { type F32, type GraphSnapshot, makeMask, maskSet } from "@graphty/graph-format";

import {
    LAYOUT_TUNING_DEFAULTS,
    MAX_1D_ITEMS,
    MAX_ITERATIONS_PER_STEP,
    STATE_HEADER_BYTES,
    UNIFORM_SLOT_BYTES,
} from "../../src/constants.js";
import { GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError } from "../../src/errors.js";
import { type CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { type BoundKernel, type Kernel } from "../../src/kernel/kernel.js";
import { UniformBlock, type UniformValues } from "../../src/kernel/struct-block.js";
import { type WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { FILL_PARAMS, KERNELS } from "../../src/kernels.js";
import {
    type BufferSpec,
    type ForceModel,
    ForceSimulation,
    type ModelInputs,
    type ModelResources,
    type StateWriter,
} from "../../src/layouts/force-simulation.js";
import { resolveNodeMass, resolveWeights } from "../../src/layouts/inputs.js";
import { type LayoutStatsBase, type ResolvedLayoutTuning } from "../../src/types/layout.js";
import { type CommonLayoutOptions, type SimulationOptions } from "../../src/types/options.js";
import { pathEdges, snapshotOf } from "../helpers/graphs.js";
import { LeakCounter } from "../helpers/leak-counter.js";
import { acquire, acquireRaw, requireGpu } from "../setup/gpu.js";

// ============================================================ the fake model

interface FakeOptions extends CommonLayoutOptions, SimulationOptions {
    readonly maxIter?: number | undefined;
    /** A pretend force law: a change recompiles (the simulation compares model.overrides() before and after). */
    readonly law?: number | undefined;
}

interface FakeStats extends LayoutStatsBase {
    /** Trace word 0 of the landed batch = the ring value of its last iteration (a u32 bit pattern). */
    readonly traceWord: number;
}

const FAKE_PARAMS = UniformBlock.define("FakeParams", [
    ["n", "u32"],
    ["dim", "u32"],
    ["flags", "u32"],
    ["iterationIndex", "u32"],
    ["seed", "u32"],
    ["scale", "f32"],
    ["center", "vec4f"],
    ["settleThreshold", "f32"],
    ["count", "u32"],
    ["value", "u32"],
    ["mode", "u32"],
]);

const FAKE_STATE_FIELDS = [
    ["centroid", "vec4f"],
    ["min", "vec4f"],
    ["max", "vec4f"],
    ["rmsRadius", "f32"],
    ["radius", "f32"],
    ["meanDisplacement", "f32"],
    ["iteration", "u32"],
    ["settledCount", "u32"],
] as const;

const FAKE_STATE = UniformBlock.define("FakeState", FAKE_STATE_FIELDS, {
    layout: "storage",
    padTo: STATE_HEADER_BYTES,
});

const FAKE_TRACE = UniformBlock.define(
    "FakeTrace",
    [
        ["value", "u32"],
        ["pad0", "u32"],
        ["pad1", "u32"],
        ["pad2", "u32"],
    ],
    { layout: "storage" },
);

/** The fill body bound to a u32 array with the fake's ring block. */
const FILL_SPEC: WgslModuleSpec = {
    id: "fake-fill",
    body: KERNELS.fill.body,
    bindings: [
        { group: 1, binding: 0, name: "dst", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "FakeParams" },
    ],
    overrideDecls: [],
    overrides: {},
    needs: [],
    uniforms: [FAKE_PARAMS],
};

/** The fill body with the registry's own FillParams (the header fill reads a model-owned uniform buffer). */
const HEADER_SPEC: WgslModuleSpec = {
    id: "fake-fill-header",
    body: KERNELS.fill.body,
    bindings: [
        { group: 1, binding: 0, name: "dst", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "FillParams" },
    ],
    overrideDecls: [],
    overrides: {},
    needs: [],
    uniforms: [FILL_PARAMS],
};

function floatBits(v: number): number {
    return new Uint32Array(new Float32Array([v]).buffer)[0];
}

function bitsToFloat(w: number): number {
    return new Float32Array(new Uint32Array([w]).buffer)[0];
}

class FakeModel implements ForceModel<FakeOptions, FakeStats> {
    readonly kind = "forceatlas2" as const;
    readonly stages = ["positions", "trace", "header", "toScene"] as const;
    readonly params: UniformBlock;
    readonly state: UniformBlock;
    readonly trace = FAKE_TRACE;
    /** When non-null every iteration also fills the 64 header words with this value. */
    headerValue: number | null = null;
    /** Extra positions fills per iteration, to make a batch heavy enough to still be in flight when a test acts. */
    workFactor = 1;
    /** Milliseconds bind() sleeps after entry and before binding (a slow bind, for the superseded-load test). */
    bindDelayMs = 0;
    /** `resources.n` of the last bind() that COMPLETED (null before the first). */
    boundN: number | null = null;
    /** Trace word 0 of every landed batch, in landing order. */
    readonly seen: number[] = [];
    readonly calls = { load: 0, reheat: 0, setParams: 0, bind: 0, inputs: 0 };
    lastPatch: Partial<FakeOptions> | null = null;
    lastWriter: StateWriter | null = null;

    private res: ModelResources | null = null;
    private kernel: Kernel | null = null;
    private headerKernel: Kernel | null = null;
    private boundPositions: BoundKernel | null = null;
    private boundTrace: BoundKernel | null = null;
    private boundHeader: BoundKernel | null = null;
    private boundScene: BoundKernel | null = null;

    constructor(blocks?: { readonly params?: UniformBlock; readonly state?: UniformBlock }) {
        this.params = blocks?.params ?? FAKE_PARAMS;
        this.state = blocks?.state ?? FAKE_STATE;
    }

    buffers(_n: number, _dim: 2 | 3): readonly BufferSpec[] {
        return [
            {
                name: "headerParams",
                byteLength: UNIFORM_SLOT_BYTES,
                usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
                zero: true,
            },
        ];
    }

    inputs(s: GraphSnapshot, _options: FakeOptions): ModelInputs {
        this.calls.inputs++;
        return { mass: resolveNodeMass(s, null), weights: resolveWeights(s, null) };
    }

    overrides(options: FakeOptions): Readonly<Record<string, number | boolean>> {
        return { LAW: options.law ?? 0 };
    }

    specs(_overrides: Readonly<Record<string, number | boolean>>, _subgroups: boolean): readonly WgslModuleSpec[] {
        return [FILL_SPEC, HEADER_SPEC];
    }

    async bind(resources: ModelResources, _overrides: Readonly<Record<string, number | boolean>>): Promise<void> {
        this.calls.bind++;
        this.res = resources;
        if (this.bindDelayMs > 0) {
            await sleep(this.bindDelayMs);
        }
        this.kernel = await resources.pipelines.kernel(FILL_SPEC);
        this.headerKernel = await resources.pipelines.kernel(HEADER_SPEC);
        const P = resources.ring.binding(this.params);
        this.boundPositions = this.kernel.bind({ dst: resources.buffer("positions"), P });
        this.boundTrace = this.kernel.bind({ dst: resources.buffer("trace"), P });
        this.boundScene = this.kernel.bind({ dst: resources.buffer("scenePositions"), P });
        const headerParams = resources.buffer("headerParams");
        this.boundHeader = this.headerKernel.bind({
            dst: resources.buffer("state"),
            P: { buffer: headerParams.buffer, offset: 0, size: FILL_PARAMS.byteLength, window: null },
        });
        this.boundN = resources.n;
    }

    paramsFor(iteration: number, _options: FakeOptions): UniformValues {
        const n = this.res?.n ?? 0;
        return { count: 4 * n, value: floatBits(iteration + 1), mode: 0 };
    }

    recordIteration(batch: CommandBatch, slot: number, _tier: "exact" | "grid", upTo?: string): void {
        const { res, kernel, headerKernel } = this;
        if (res === null || kernel === null || headerKernel === null || this.boundPositions === null) {
            throw new Error("fake model: recordIteration before bind");
        }
        const pass = batch.pass(`fake/${slot}`);
        const offsets = [res.ring.offsetOf(slot)];
        const plan = plan1d(4 * res.n, kernel.workgroupSize, res.caps);
        for (let w = 0; w < this.workFactor; w++) {
            kernel.dispatch(pass, this.boundPositions, plan, offsets);
        }
        if (upTo === "positions") {
            return;
        }
        if (this.boundTrace !== null) {
            kernel.dispatch(pass, this.boundTrace, plan, offsets);
        }
        if (upTo === "trace") {
            return;
        }
        if (this.headerValue !== null && this.boundHeader !== null) {
            const bytes = new ArrayBuffer(FILL_PARAMS.byteLength);
            FILL_PARAMS.write(new DataView(bytes), { count: STATE_HEADER_BYTES / 4, value: this.headerValue, mode: 0 });
            res.device.queue.writeBuffer(res.buffer("headerParams").buffer, 0, bytes);
            headerKernel.dispatch(
                pass,
                this.boundHeader,
                plan1d(STATE_HEADER_BYTES / 4, headerKernel.workgroupSize, res.caps),
                [0],
            );
        }
        if (upTo === "header") {
            return;
        }
        // the per-batch epilogue "toScene": recorded when upTo is undefined (the simulation's last iteration of a
        // batch) or "toScene" (a debug run); the simulation passes "header" for every other iteration
        if (this.boundScene !== null) {
            kernel.dispatch(pass, this.boundScene, plan, offsets);
        }
    }

    onLoad(state: StateWriter): void {
        this.calls.load++;
        this.lastWriter = state;
    }

    onReheat(_state: StateWriter): void {
        this.calls.reheat++;
    }

    onSetParams(patch: Partial<FakeOptions>, _state: StateWriter): void {
        this.calls.setParams++;
        this.lastPatch = patch;
    }

    readStats(state: DataView, trace: DataView): FakeStats {
        const s = this.state.read(state);
        const traceWord = trace.byteLength >= 4 ? trace.getUint32(0, true) : 0;
        if (trace.byteLength > 0) {
            this.seen.push(traceWord);
        }
        const centroid = s.centroid as readonly number[];
        return {
            iteration: s.iteration as number,
            meanDisplacement: s.meanDisplacement as number,
            rmsRadius: s.rmsRadius as number,
            layoutRadius: s.radius as number,
            centroid: [centroid[0], centroid[1], centroid[2]],
            repulsionTier: "exact",
            maxCellOccupancy: null,
            outsideGrid: null,
            msPerIteration: null,
            traceWord,
        };
    }
}

// ============================================================ helpers

type FakeSim = ForceSimulation<FakeOptions, FakeStats>;

function makeSim(
    ctx: GpuContext,
    fake: FakeModel,
    options: FakeOptions = {},
    tuning: Partial<ResolvedLayoutTuning> = {},
): FakeSim {
    return new ForceSimulation<FakeOptions, FakeStats>(
        ctx,
        fake,
        { maxIter: 1000, seed: 1, ...options },
        { ...LAYOUT_TUNING_DEFAULTS, ...tuning },
        (patch, current) => ({ ...current, ...patch }),
    );
}

function nanPositions(n: number): F32 {
    return new Float32Array(3 * n).fill(Number.NaN);
}

function graph(n: number): GraphSnapshot {
    return snapshotOf(pathEdges(n), { nodeCount: n });
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

async function rejectionOf(p: Promise<unknown>): Promise<{ code: string; details: Readonly<Record<string, unknown>> }> {
    try {
        await p;
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected the promise to reject");
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

/**
 * A batch created by step() is submitted on the serialised chain (after the bind promise, which compiles on the
 * first step); a test that must act while it is IN FLIGHT waits for the submission (a 1 ms poll, so the test's
 * continuation runs before any readback callback, which Dawn delivers from a later macrotask) and uses a heavy
 * batch (heavyFake(): HEAVY_N nodes, 256 iterations, 8 positions fills each = about 4 GB of writes per batch,
 * about 8 ms on the RTX 4070 SUPER and about a second on lavapipe) so the GPU cannot have finished inside the
 * poll interval. Every caller asserts the in-flight state explicitly right after the wait (`inFlight` and an
 * empty `fake.seen`) so a batch that landed early fails there, not in a later assertion.
 */
async function waitForSubmission(s: FakeSim, id: number): Promise<void> {
    while (s.lastSubmittedBatchId < id) {
        await sleep(1);
    }
}

const HEAVY_N = 131072;

/**
 * HEAVY_N is above LAYOUT_TUNING_DEFAULTS.exactMaxNodes, so the "auto" tier would resolve to the grid tier (PLAN
 * DECISION 11: E_UNSUPPORTED until P4); the fake has no repulsion stage, so the heavy tests pin the exact tier.
 */
const EXACT_TIER: Partial<ResolvedLayoutTuning> = { repulsion: "exact" };

function heavyFake(): FakeModel {
    const fake = new FakeModel();
    fake.workFactor = 8;
    return fake;
}

/** Every row of the owner's array equals (x, y, z) except the skipped rows (one scan; the first bad row is reported). */
function expectRows(
    positions: F32,
    n: number,
    x: number,
    y: number,
    z: number,
    skip: ReadonlySet<number> = new Set(),
): void {
    let bad = -1;
    for (let i = 0; i < n && bad < 0; i++) {
        if (skip.has(i)) {
            continue;
        }
        if (positions[3 * i] !== x || positions[3 * i + 1] !== y || positions[3 * i + 2] !== z) {
            bad = i;
        }
    }
    if (bad >= 0) {
        expect([positions[3 * bad], positions[3 * bad + 1], positions[3 * bad + 2]], `row ${bad}`).toEqual([x, y, z]);
    }
    expect(bad).toBe(-1);
}

// ============================================================ the tests

describe("ForceSimulation (fake model)", () => {
    // one context for the describe, acquired lazily AFTER requireGpu() so a "skip" policy skips instead of failing
    let shared: GpuContext | null = null;
    const sims: FakeSim[] = [];

    async function ctxOf(): Promise<GpuContext> {
        if (shared === null) {
            shared = await acquire({ label: "force-simulation" });
        }
        return shared;
    }

    afterEach(() => {
        for (const sim of sims.splice(0)) {
            sim.dispose();
        }
    });

    function sim(
        ctx: GpuContext,
        fake: FakeModel,
        options?: FakeOptions,
        tuning?: Partial<ResolvedLayoutTuning>,
    ): FakeSim {
        const s = makeSim(ctx, fake, options, tuning);
        sims.push(s);
        return s;
    }

    it("validates the model's blocks and the options at construction", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const noIterationIndex = UniformBlock.define("BadParams", [
            ["n", "u32"],
            ["dim", "u32"],
            ["flags", "u32"],
            ["seed", "u32"],
            ["scale", "f32"],
            ["center", "vec4f"],
            ["settleThreshold", "f32"],
        ]);
        const e1 = errorOf(() => makeSim(ctx, new FakeModel({ params: noIterationIndex })));
        expect(e1.code).toBe("E_INVALID_ARGUMENT");
        expect(String(e1.details.value)).toContain("iterationIndex");

        const noSettledCount = UniformBlock.define(
            "BadState",
            [
                ["centroid", "vec4f"],
                ["min", "vec4f"],
                ["max", "vec4f"],
                ["rmsRadius", "f32"],
                ["radius", "f32"],
                ["meanDisplacement", "f32"],
                ["iteration", "u32"],
            ],
            { layout: "storage" },
        );
        const e2 = errorOf(() => makeSim(ctx, new FakeModel({ state: noSettledCount })));
        expect(e2.code).toBe("E_INVALID_ARGUMENT");
        expect(String(e2.details.value)).toContain("settledCount");

        const tooBig = UniformBlock.define("BigState", [...FAKE_STATE_FIELDS, ["pad", "vec4f"]], {
            layout: "storage",
            padTo: STATE_HEADER_BYTES + 16,
        });
        expect(errorOf(() => makeSim(ctx, new FakeModel({ state: tooBig }))).code).toBe("E_INVALID_ARGUMENT");

        expect(errorOf(() => makeSim(ctx, new FakeModel(), { maxInFlight: 0 })).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => makeSim(ctx, new FakeModel(), { dim: 4 as unknown as 2 })).code).toBe(
            "E_INVALID_ARGUMENT",
        );

        const ok = sim(ctx, new FakeModel(), { maxInFlight: 3 });
        expect(ok.ring.slots).toBe((3 + 1) * MAX_ITERATIONS_PER_STEP);
        expect(ok.state).toBe("created");
        expect(ok.dim).toBe(2);
        expect(ok.tier).toBe("exact");
        expect(ok.nodeCount).toBe(0);
        expect(ok.settled).toBe(false);
        expect(ok.inFlight).toBe(0);
        expect(ok.iterationsDone).toBe(0);
        expect(ok.stats.iteration).toBe(0);
        expect(ok.inspect).toBeUndefined();
        expect(ok.debugRunStages).toBeUndefined();
    });

    it("created: step / setFixed / setPosition are E_NOT_LOADED; setParams and reheat are allowed", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake);
        const e = await rejectionOf(s.step());
        expect(e).toEqual({ code: "E_NOT_LOADED", details: { state: "created" } });
        expect(errorOf(() => s.setFixed(makeMask(8))).code).toBe("E_NOT_LOADED");
        expect(errorOf(() => s.setPosition(0, 0, 0, 0)).code).toBe("E_NOT_LOADED");
        s.setParams({ settleWindow: 5 });
        expect(s.options.settleWindow).toBe(5);
        expect(fake.calls.setParams).toBe(1);
        s.reheat();
        expect(fake.calls.reheat).toBeGreaterThanOrEqual(1);
        expect(s.state).toBe("created");
    });

    it("load() rejects a directed snapshot, a too-large graph, a bad positions array and the grid tier", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel());
        const directed = snapshotOf(pathEdges(4), { nodeCount: 4, directed: true });
        const e1 = errorOf(() => s.load(directed, nanPositions(4)));
        expect(e1.code).toBe("E_SNAPSHOT");
        expect(e1.details.reason).toBe("directed");
        expect(e1.details.serial).toBe(directed.serial);

        const huge = { directed: false, nodeCount: MAX_1D_ITEMS + 1, serial: 4242 } as unknown as GraphSnapshot;
        const e2 = errorOf(() => s.load(huge, new Float32Array(0)));
        expect(e2.code).toBe("E_TOO_LARGE");
        expect(e2.details.path).toBe("partials");
        expect(e2.details.limit).toBe(MAX_1D_ITEMS);

        const g = graph(8);
        const e3 = errorOf(() => s.load(g, new Float32Array(23)));
        expect(e3).toEqual({ code: "E_INVALID_ARGUMENT", details: { argument: "positions", value: 23, expected: 24 } });
        const shared = new Float32Array(new SharedArrayBuffer(4 * 24)) as unknown as F32;
        const e4 = errorOf(() => s.load(g, shared));
        expect(e4.code).toBe("E_INVALID_ARGUMENT");
        expect(e4.details.argument).toBe("positions");
        expect(s.state).toBe("created");

        const grid = sim(ctx, new FakeModel(), {}, { repulsion: "grid" });
        const e5 = errorOf(() => grid.load(g, nanPositions(8)));
        expect(e5.code).toBe("E_UNSUPPORTED");
        expect(e5.details.feature).toBe("repulsion.grid");
        const auto = sim(ctx, new FakeModel(), {}, { repulsion: "auto", exactMaxNodes: 4 });
        expect(errorOf(() => auto.load(g, nanPositions(8))).details.feature).toBe("repulsion.grid");
        const exact = sim(ctx, new FakeModel(), {}, { repulsion: "exact", exactMaxNodes: 4 });
        exact.load(g, nanPositions(8));
        expect(exact.tier).toBe("exact");
        expect(exact.state).toBe("loaded");
    });

    it("load() seeds NaN rows, uploads, writes the initial stats and enters 'loaded'", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake, { seed: 7 });
        const g = graph(16);
        const positions = nanPositions(16);
        s.load(g, positions);
        expect(s.state).toBe("loaded");
        expect(s.nodeCount).toBe(16);
        expect(s.dim).toBe(2);
        expect(s.tier).toBe("exact");
        expect(s.settled).toBe(false);
        expect(s.generation).toBe(1);
        expect(fake.calls.load).toBe(1);
        expect(fake.calls.inputs).toBe(1);
        expect(fake.lastWriter?.get("iteration")).toBe(0);
        expect(positions.every((v) => Number.isFinite(v))).toBe(true);
        // the initial stats are computed on the CPU from the seeded array (scale 1, center 0: layout = scene)
        let cx = 0;
        let cy = 0;
        for (let i = 0; i < 16; i++) {
            cx += positions[3 * i];
            cy += positions[3 * i + 1];
        }
        cx /= 16;
        cy /= 16;
        let sumSq = 0;
        let maxSq = 0;
        for (let i = 0; i < 16; i++) {
            const dx = positions[3 * i] - cx;
            const dy = positions[3 * i + 1] - cy;
            sumSq += dx * dx + dy * dy;
            maxSq = Math.max(maxSq, dx * dx + dy * dy);
        }
        const { stats } = s;
        expect(stats.iteration).toBe(0);
        expect(stats.centroid[0]).toBeCloseTo(cx, 6);
        expect(stats.centroid[1]).toBeCloseTo(cy, 6);
        expect(stats.centroid[2]).toBe(0);
        expect(stats.rmsRadius).toBeCloseTo(Math.sqrt(sumSq / 16), 6);
        expect(stats.layoutRadius).toBeCloseTo(Math.sqrt(maxSq), 6);
        expect(stats.meanDisplacement).toBe(0);
        expect(stats.repulsionTier).toBe("exact");
        expect(stats.msPerIteration).toBeNull();
        expect(fake.lastWriter?.get("settledCount")).toBe(0);
        // the first step lands the model's values
        await s.step(1);
        expect(s.iterationsDone).toBe(1);
    });

    it("a step writes the fill value into every position and advances the counters", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake);
        const g = graph(20);
        const positions = nanPositions(20);
        s.load(g, positions);
        await s.step(1);
        expectRows(positions, 20, 1, 1, 1);
        expect(s.iterationsDone).toBe(1);
        expect(s.inFlight).toBe(0);
        expect(s.lastSubmittedBatchId).toBeGreaterThanOrEqual(1);
        expect(bitsToFloat(s.stats.traceWord)).toBe(1);
        expect(fake.seen.map(bitsToFloat)).toEqual([1]);
        expect(typeof s.stats.msPerIteration).toBe("number");
        expect(s.stats.msPerIteration).toBeGreaterThanOrEqual(0);
        expect(s.stats.iteration).toBe(0); // the fake never writes the header: the host-written value stands
        await s.step(3); // global iterations 1, 2, 3 -> the last one writes 4
        expectRows(positions, 20, 4, 4, 4);
        expect(s.iterationsDone).toBe(4);
        expect(fake.seen.map(bitsToFloat)).toEqual([1, 4]);
        // step() without an argument uses iterationsPerStep
        const s2 = sim(ctx, new FakeModel(), { iterationsPerStep: 2 });
        const p2 = nanPositions(20);
        s2.load(g, p2);
        await s2.step();
        expectRows(p2, 20, 2, 2, 2);
    });

    it("scale and center are validated and applied on the way IN (seeding, setPosition); the host never rescales a readback", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        ctx.debug.inspect = true;
        try {
            expect(errorOf(() => makeSim(ctx, new FakeModel(), { scale: 0 })).code).toBe("E_INVALID_ARGUMENT");
            expect(errorOf(() => makeSim(ctx, new FakeModel(), { center: [Number.NaN, 0, 0] })).code).toBe(
                "E_INVALID_ARGUMENT",
            );
            const s = sim(ctx, new FakeModel(), { scale: 2, center: [1, 2, 7], seed: 7 });
            const positions = nanPositions(8);
            s.load(graph(8), positions);
            // seeded in scene units (v * 2 + center, z = center.z); uploaded as v in layout units with z 0 and the mass in .w
            const uploaded = await s.inspect?.("positions");
            expect(uploaded?.length).toBe(4 * 8);
            for (let i = 0; i < 8; i++) {
                expect(uploaded?.[4 * i]).toBeCloseTo((positions[3 * i] - 1) / 2, 6);
                expect(uploaded?.[4 * i + 1]).toBeCloseTo((positions[3 * i + 1] - 2) / 2, 6);
                expect(uploaded?.[4 * i + 2]).toBe(0);
                expect(uploaded?.[4 * i + 3]).toBeGreaterThan(0);
                expect(positions[3 * i + 2]).toBe(7);
            }
            // setPosition inverts the same transform (z forced to 0 in 2D) and leaves the mass lane alone
            s.setPosition(3, 5, 8, 9);
            const after = await s.inspect?.("positions");
            expect(Array.from((after ?? new Float32Array(0)).subarray(12, 15))).toEqual([2, 3, 0]);
            expect(after?.[15]).toBe(uploaded?.[15]);
            expect(Array.from(positions.subarray(9, 12))).toEqual([5, 8, 9]);
            expect(errorOf(() => s.setParams({ scale: -1 })).code).toBe("E_INVALID_ARGUMENT");
            // the fake's toScene stage writes the raw value: nothing on the host rescales it
            await s.step(1);
            expectRows(positions, 8, 1, 1, 1);
        } finally {
            ctx.debug.inspect = false;
        }
    });

    it("step() validates the iteration count", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel());
        s.load(graph(4), nanPositions(4));
        for (const k of [0, -1, 1.5, MAX_ITERATIONS_PER_STEP + 1, Number.NaN]) {
            const e = await rejectionOf(s.step(k));
            expect(e.code).toBe("E_INVALID_ARGUMENT");
            expect(e.details.argument).toBe("iterations");
        }
        expect(s.inFlight).toBe(0);
    });

    it("an empty graph loads with no GPU work and is settled at once", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel());
        s.load(snapshotOf([], { nodeCount: 0 }), new Float32Array(0));
        expect(s.state).toBe("loaded");
        expect(s.settled).toBe(true);
        await s.step(4);
        expect(s.lastSubmittedBatchId).toBe(0);
        expect(s.iterationsDone).toBe(0);
        const stats = await s.run();
        expect(stats.iteration).toBe(0);
    });

    it("E_RELEASED after release() during a live simulation; a fresh load() re-uploads", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel());
        const g = graph(8);
        const positions = nanPositions(8);
        s.load(g, positions);
        await s.step(1);
        ctx.release(g);
        const e = await rejectionOf(s.step(1));
        expect(e).toEqual({ code: "E_RELEASED", details: { serial: g.serial } });
        s.load(g, positions);
        await s.step(1);
        expectRows(positions, 8, 1, 1, 1);
        ctx.release(g);
    });

    it("coalesces at maxInFlight and returns the OLDEST pending promise", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel(), { maxInFlight: 2 });
        const positions = nanPositions(64);
        s.load(graph(64), positions);
        const p1 = s.step(8);
        const p2 = s.step(8);
        const p3 = s.step(8);
        expect(p3).toBe(p1);
        expect(p2).not.toBe(p1);
        expect(s.coalesced).toBe(1);
        expect(s.inFlight).toBe(2);
        await p1;
        expect(s.inFlight).toBeLessThanOrEqual(1);
        const p4 = s.step(8);
        expect(p4).not.toBe(p1);
        expect(p4).not.toBe(p2);
        await s.flush();
        expect(s.inFlight).toBe(0);
        expect(s.iterationsDone).toBe(24); // p3 ran nothing
        expect(s.coalesced).toBe(1);
        expectRows(positions, 64, 24, 24, 24);
    });

    it("the ring never rewrites a slot a submitted batch still reads (maxInFlight 3, step(256) across a wrap)", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake, { maxInFlight: 3, maxIter: 100000 });
        const positions = nanPositions(64);
        s.load(graph(64), positions);
        expect(s.ring.slots).toBe(4 * MAX_ITERATIONS_PER_STEP);
        const inflight: Promise<void>[] = [s.step(256), s.step(256), s.step(256)];
        expect(s.inFlight).toBe(3);
        expect(s.coalesced).toBe(0);
        // 8 batches of 256 slots on a 1024-slot ring: batches 4..7 reuse the slots of batches 0..3
        for (let b = 3; b < 8; b++) {
            const oldest = inflight.shift();
            if (oldest !== undefined) {
                await oldest;
            }
            inflight.push(s.step(256));
        }
        await s.flush();
        expect(s.coalesced).toBe(0);
        expect(s.iterationsDone).toBe(8 * 256);
        // each landed batch saw ITS OWN last-iteration value: 256, 512, ..., 2048
        expect(fake.seen.map(bitsToFloat)).toEqual([256, 512, 768, 1024, 1280, 1536, 1792, 2048]);
        expectRows(positions, 64, 2048, 2048, 2048);
    });

    it("setParams: dim and maxInFlight are rejected, nodeSize is unsupported, a law change rebinds", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake, { maxInFlight: 2 });
        s.load(graph(8), nanPositions(8));
        await s.step(1);
        expect(fake.calls.bind).toBe(1);
        expect(errorOf(() => s.setParams({ dim: 3 })).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => s.setParams({ maxInFlight: 3 })).code).toBe("E_INVALID_ARGUMENT");
        s.setParams({ maxInFlight: 2 }); // the same value is not a change
        const nodeSize = errorOf(() =>
            s.setParams({ nodeSize: new Float32Array(8) } as unknown as Partial<FakeOptions>),
        );
        expect(nodeSize.code).toBe("E_UNSUPPORTED");
        expect(nodeSize.details.option).toBe("nodeSize");
        const bindsBefore = fake.calls.bind;
        // a numeric tweak: no rebind, but onSetParams and reheat
        s.setParams({ settleThreshold: 0.5 });
        expect(fake.calls.bind).toBe(bindsBefore);
        expect(fake.lastPatch).toEqual({ settleThreshold: 0.5 });
        expect(s.iterationsDone).toBe(0);
        expect(s.options.settleThreshold).toBe(0.5);
        // a law change: model.overrides() differs -> rebind
        s.setParams({ law: 1 });
        await s.step(1);
        expect(fake.calls.bind).toBe(bindsBefore + 1);
        expect(fake.lastPatch).toEqual({ law: 1 });
        expect(s.iterationsDone).toBe(1);
        // scale / center changes are validated and kept in the record (the model's toScene reads them per slot)
        s.setParams({ scale: 10, center: [5, 5, 5] });
        expect(s.options.scale).toBe(10);
        expect(errorOf(() => s.setParams({ center: [0, Number.POSITIVE_INFINITY] })).code).toBe("E_INVALID_ARGUMENT");
    });

    it("load() during flight bumps the generation and discards the in-flight batch", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = heavyFake();
        const s = sim(ctx, fake, { maxInFlight: 2 }, EXACT_TIER);
        const gA = graph(HEAVY_N);
        const posA = nanPositions(HEAVY_N);
        s.load(gA, posA);
        const seededA = Array.from(posA);
        const { generation } = s;
        const p = s.step(256);
        await waitForSubmission(s, 1); // submitted and still executing
        expect(s.inFlight).toBe(1);
        expect(fake.seen).toEqual([]);
        const gB = graph(12);
        const posB = nanPositions(12);
        s.load(gB, posB);
        expect(s.generation).toBe(generation + 1);
        expect(s.nodeCount).toBe(12);
        await p; // resolves (discarded), never rejects
        expect(Array.from(posA)).toEqual(seededA);
        expect(s.iterationsDone).toBe(0);
        expect(fake.seen).toEqual([]);
        // a record that is created but not yet submitted at load() time is discarded the same way
        const p2 = s.step(2);
        const gC = graph(12);
        s.load(gC, posB);
        await p2;
        expect(fake.seen).toEqual([]);
        await s.step(1);
        expectRows(posB, 12, 1, 1, 1);
        expect(fake.seen.map(bitsToFloat)).toEqual([1]);
        expect(s.inFlight).toBe(0);
        // a load() of the same snapshot and size keeps the buffers and the fixed words
        const mask = makeMask(12);
        maskSet(mask, 2, true);
        s.setFixed(mask);
        s.load(gC, posB);
        expect(s.generation).toBe(generation + 3);
        await s.step(1);
        expect(s.iterationsDone).toBe(1);
    });

    it("a load() that supersedes a bind still inside model.bind() never binds the old buffers after the new ones (PLAN DECISION 20)", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        fake.bindDelayMs = 50;
        const s = sim(ctx, fake);
        const gA = graph(4096);
        s.load(gA, nanPositions(4096));
        // wait until gA's bind is INSIDE the fake's bind() (past the generation guard, sleeping)
        while (fake.calls.bind < 1) {
            await sleep(1);
        }
        expect(fake.boundN).toBeNull();
        const gB = graph(12);
        const posB = nanPositions(12);
        s.load(gB, posB); // resizes: gA's buffers are destroyed while gA's bind is still running
        await s.step(1); // waits for the serialised chain: gA's bind completes, THEN gB's bind runs and binds
        expect(fake.calls.bind).toBe(2);
        expect(fake.boundN).toBe(12);
        expectRows(posB, 12, 1, 1, 1);
        expect(fake.seen.map(bitsToFloat)).toEqual([1]);
        // the same for a setParams() rebind racing a load(): the last bind to run is the new load's
        fake.bindDelayMs = 0;
        s.setParams({ law: 1 });
        const gC = graph(12);
        s.load(gC, posB);
        await s.step(1);
        expect(fake.boundN).toBe(12);
        expectRows(posB, 12, 1, 1, 1);
    });

    it("setPosition: validation, the immediate owner write, and the override list across a batch", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = heavyFake();
        const s = sim(ctx, fake, { maxInFlight: 2 }, EXACT_TIER);
        const n = HEAVY_N;
        const positions = nanPositions(n);
        s.load(graph(n), positions);
        expect(errorOf(() => s.setPosition(n, 0, 0, 0)).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => s.setPosition(-1, 0, 0, 0)).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => s.setPosition(1.5, 0, 0, 0)).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => s.setPosition(0, Number.NaN, 0, 0)).code).toBe("E_INVALID_ARGUMENT");
        const p = s.step(256);
        await waitForSubmission(s, 1); // the batch is in flight
        expect(s.inFlight).toBe(1);
        expect(fake.seen).toEqual([]);
        const reheats = fake.calls.reheat;
        s.setPosition(3, 100, 200, 5);
        expect(positions[9]).toBe(100);
        expect(positions[10]).toBe(200);
        expect(positions[11]).toBe(5);
        expect(s.overrides.get(3)).toBe(s.lastSubmittedBatchId);
        expect(s.lastSubmittedBatchId).toBeGreaterThanOrEqual(1);
        expect(fake.calls.reheat).toBe(reheats + 1);
        await p;
        // the batch computed before the write does not move the node back
        expect(positions[9]).toBe(100);
        expect(positions[10]).toBe(200);
        expect(positions[11]).toBe(5);
        expectRows(positions, n, 256, 256, 256, new Set([3]));
        expect(s.overrides.size).toBe(1);
        // the next batch was submitted after the write and lands normally; the override is cleared
        await s.step(1);
        expectRows(positions, n, 257, 257, 257);
        expect(s.overrides.size).toBe(0);
        expect(s.iterationsDone).toBe(257);
    });

    it("setFixed: validation, the words reach the device, an unpin reheats and a pin does not", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        ctx.debug.inspect = true;
        try {
            const fake = new FakeModel();
            const s = sim(ctx, fake);
            const positions = nanPositions(40);
            s.load(graph(40), positions);
            expect(errorOf(() => s.setFixed(new Uint32Array(1))).code).toBe("E_INVALID_ARGUMENT");
            await s.step(1);
            const mask = makeMask(40);
            maskSet(mask, 1, true);
            maskSet(mask, 33, true);
            const reheats = fake.calls.reheat;
            s.setFixed(mask);
            expect(fake.calls.reheat).toBe(reheats);
            expect(s.iterationsDone).toBe(1);
            await s.step(1);
            const words = await s.inspect?.("fixed");
            expect(words).toBeInstanceOf(Uint32Array);
            expect(Array.from(words ?? [])).toEqual(Array.from(mask));
            maskSet(mask, 33, false);
            s.setFixed(mask);
            expect(fake.calls.reheat).toBe(reheats + 1);
            expect(s.iterationsDone).toBe(0);
            await s.step(1);
            expect(Array.from((await s.inspect?.("fixed")) ?? [])).toEqual(Array.from(mask));
        } finally {
            ctx.debug.inspect = false;
        }
    });

    it("settles at maxIter and a settled simulation submits nothing", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel(), { maxIter: 3 });
        s.load(graph(8), nanPositions(8));
        await s.step(1);
        await s.step(1);
        expect(s.settled).toBe(false);
        await s.step(1);
        expect(s.settled).toBe(true);
        expect(s.iterationsDone).toBe(3);
        const last = s.lastSubmittedBatchId;
        await s.step(1);
        expect(s.lastSubmittedBatchId).toBe(last);
        expect(s.inFlight).toBe(0);
        const stats = await s.run();
        expect(stats.iteration).toBe(0);
        expect(s.lastSubmittedBatchId).toBe(last);
    });

    it("settles at settleWindow; reheat resets only the two counters and calls the model hook", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake, { settleWindow: 2, maxIter: 1000 });
        s.load(graph(8), nanPositions(8));
        fake.headerValue = 2; // every header word = 2: settledCount = 2 >= settleWindow, iteration = 2
        await s.step(1);
        expect(s.settled).toBe(true);
        expect(s.stats.iteration).toBe(2);
        expect(s.iterationsDone).toBe(1);
        const reheats = fake.calls.reheat;
        s.reheat();
        expect(fake.calls.reheat).toBe(reheats + 1);
        expect(s.settled).toBe(false);
        expect(s.iterationsDone).toBe(0);
        fake.headerValue = null;
        await s.step(1);
        expect(s.settled).toBe(false); // settledCount was written back to 0 before the batch
        expect(s.iterationsDone).toBe(1);
        expect(s.stats.iteration).toBe(2); // the header's other words were NOT rewritten by reheat
        expect(fake.lastWriter?.get("settledCount")).toBe(0);
        expect(fake.lastWriter?.get("iteration")).toBe(2);
    });

    it("a batch in flight at reheat() cannot settle the simulation when it lands (PLAN DECISION 21)", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = heavyFake();
        const s = sim(ctx, fake, { settleWindow: 2, maxIter: 1000, maxInFlight: 2 }, EXACT_TIER);
        s.load(graph(HEAVY_N), nanPositions(HEAVY_N));
        fake.headerValue = 2; // the batch lands with settledCount = 2 >= settleWindow
        const p = s.step(256);
        await waitForSubmission(s, 1);
        expect(s.inFlight).toBe(1);
        expect(fake.seen).toEqual([]);
        s.reheat(); // an unpin or a drag while the batch is still executing
        expect(s.settled).toBe(false);
        await p;
        expect(s.settled).toBe(false); // the landed settledCount predates the reheat
        expect(s.iterationsDone).toBe(256); // the contract's iterationsDone += k still applies
        expect(s.stats.iteration).toBe(2); // the rest of the header is absorbed as usual
        // the next batch starts from the settledCount the reheat wrote (0); without the header fill it stays 0
        fake.headerValue = null;
        await s.step(1);
        expect(s.settled).toBe(false);
        expect(s.iterationsDone).toBe(257);
        // a batch submitted AFTER the reheat settles the simulation as usual
        fake.headerValue = 2;
        await s.step(1);
        expect(s.settled).toBe(true);
    });

    it("a state write flushes only its own field: the in-flight batch's GPU-written header words survive the next submit (PLAN DECISION 5)", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = heavyFake();
        const s = sim(ctx, fake, { settleWindow: 2, maxIter: 1000, maxInFlight: 2 }, EXACT_TIER);
        s.load(graph(HEAVY_N), nanPositions(HEAVY_N));
        fake.headerValue = 2; // batch A fills every header word with 2 on the GPU
        const a = s.step(256);
        await waitForSubmission(s, 1);
        expect(s.inFlight).toBe(1);
        expect(fake.seen).toEqual([]);
        // reheat while A is still executing (the shadow still holds load()'s header: iteration 0), then submit B
        // at once: its flush must write the ONE queued word (settledCount = 0), not the whole shadow -- a
        // whole-header writeBuffer would land after A and put iteration back to 0
        s.reheat();
        fake.headerValue = null;
        const b = s.step(1);
        await waitForSubmission(s, 2);
        expect(s.inFlight).toBe(2);
        expect(fake.seen).toEqual([]); // A had not landed when B's state writes were flushed
        await a;
        expect(s.stats.iteration).toBe(2);
        await b;
        expect(s.iterationsDone).toBe(257);
        expect(s.settled).toBe(false); // B's header carries the reheat's settledCount = 0
        expect(s.stats.iteration).toBe(2); // A's GPU-written word survived B's flush
        expect(s.stats.rmsRadius).toBe(bitsToFloat(2)); // and so did the rest of A's header
        expect(fake.lastWriter?.get("settledCount")).toBe(0);
        expect(fake.lastWriter?.get("iteration")).toBe(2);
    });

    it("flush() resolves when nothing is in flight", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel(), { maxInFlight: 2 });
        s.load(graph(32), nanPositions(32));
        await s.flush(); // nothing pending
        void s.step(16);
        void s.step(16);
        expect(s.inFlight).toBe(2);
        await s.flush();
        expect(s.inFlight).toBe(0);
        expect(s.iterationsDone).toBe(32);
    });

    it("run() batches until the budget; a signal aborts with E_ABORTED and discards the in-flight batch", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const fake = new FakeModel();
        const s = sim(ctx, fake, { maxIter: 1000 });
        const positions = nanPositions(16);
        s.load(graph(16), positions);
        const stats = await s.run({ maxIter: 10, batch: 4 });
        expect(s.iterationsDone).toBe(10); // 4 + 4 + 2
        expect(fake.seen.map(bitsToFloat)).toEqual([4, 8, 10]);
        expect(stats.traceWord).toBe(floatBits(10));
        expectRows(positions, 16, 10, 10, 10);
        expect((await rejectionOf(s.run({ batch: 0 }))).code).toBe("E_INVALID_ARGUMENT");

        const pre = new AbortController();
        pre.abort();
        const before = s.lastSubmittedBatchId;
        expect((await rejectionOf(s.run({ signal: pre.signal }))).code).toBe("E_ABORTED");
        expect(s.lastSubmittedBatchId).toBe(before);

        const controller = new AbortController();
        const running = s.run({ maxIter: 100000, batch: 8, signal: controller.signal });
        await sleep(5);
        const doneAtAbort = s.iterationsDone;
        controller.abort();
        expect((await rejectionOf(running)).code).toBe("E_ABORTED");
        await s.flush();
        expect(s.iterationsDone).toBe(doneAtAbort); // the batch in flight at the abort was discarded
        expect(s.inFlight).toBe(0);
        expect(s.state).toBe("loaded");
    });

    it("inspect() and debugRunStages() are present only with ctx.debug.inspect", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        ctx.debug.inspect = true;
        try {
            const fake = new FakeModel();
            const s = sim(ctx, fake, { maxInFlight: 2 });
            expect(s.inspect).toBeDefined();
            expect(s.debugRunStages).toBeDefined();
            expect((await rejectionOf(s.inspect?.("positions") ?? Promise.resolve())).code).toBe("E_NOT_LOADED");
            const positions = nanPositions(16);
            s.load(graph(16), positions);
            await s.step(2);
            const pos = await s.inspect?.("positions");
            expect(pos).toBeInstanceOf(Float32Array);
            expect(pos?.length).toBe(4 * 16);
            expect(Array.from(pos ?? []).every((v) => v === 2)).toBe(true);
            const trace = await s.inspect?.("trace");
            expect(trace).toBeInstanceOf(Uint32Array);
            expect(trace?.length).toBe((MAX_ITERATIONS_PER_STEP * FAKE_TRACE.byteLength) / 4);
            expect(trace?.[0]).toBe(floatBits(2));
            const header = await s.inspect?.("headerParams");
            expect(header).toBeInstanceOf(Float32Array);
            expect(header?.length).toBe(UNIFORM_SLOT_BYTES / 4);
            expect((await rejectionOf(s.inspect?.("nope") ?? Promise.resolve())).code).toBe("E_INVALID_ARGUMENT");
            expect((await rejectionOf(s.debugRunStages?.("bogus") ?? Promise.resolve())).code).toBe(
                "E_INVALID_ARGUMENT",
            );
            // a debug run truncated after "positions": the positions get iteration 2's value (iterationsSubmitted = 2
            // -> value 3) while the trace region is untouched and nothing is counted
            await s.debugRunStages?.("positions");
            expect(Array.from((await s.inspect?.("positions")) ?? []).every((v) => v === 3)).toBe(true);
            expect((await s.inspect?.("trace"))?.[0]).toBe(floatBits(2));
            expect(s.iterationsDone).toBe(2);
            expect(fake.seen.length).toBe(1);
            expect(Array.from(positions.subarray(0, 3))).toEqual([2, 2, 2]); // no readback into the owner's array
            await s.step(1);
            expectRows(positions, 16, 3, 3, 3);
        } finally {
            ctx.debug.inspect = false;
        }
    });

    it("dispose(): idempotent, every later call is E_DISPOSED, pending steps resolve", async (t) => {
        requireGpu(t);
        const ctx = await ctxOf();
        const s = sim(ctx, new FakeModel(), { maxInFlight: 2 });
        const g = graph(8);
        s.load(g, nanPositions(8));
        const p = s.step(4);
        s.dispose();
        expect(s.state).toBe("disposed");
        await p;
        s.dispose();
        expect((await rejectionOf(s.step(1))).code).toBe("E_DISPOSED");
        expect(errorOf(() => s.load(g, nanPositions(8))).code).toBe("E_DISPOSED");
        expect(errorOf(() => s.setFixed(makeMask(8))).code).toBe("E_DISPOSED");
        expect(errorOf(() => s.setPosition(0, 0, 0, 0)).code).toBe("E_DISPOSED");
        expect(errorOf(() => s.setParams({ law: 2 })).code).toBe("E_DISPOSED");
        expect(errorOf(() => s.reheat()).code).toBe("E_DISPOSED");
        expect((await rejectionOf(s.run())).code).toBe("E_DISPOSED");
        expect(s.inFlight).toBe(0);
    });
});

describe("ForceSimulation lifecycle on a raw device", () => {
    it("dispose leaves no simulation buffer behind (LeakCounter), and the context ends at 0 live buffers", async (t) => {
        requireGpu(t);
        const raw = await acquireRaw();
        const counter = LeakCounter.wrap(raw.device);
        const ctx = GpuContext.from(raw.device, { runtime: "node" });
        try {
            const g = graph(16);
            const round = async (): Promise<void> => {
                const s = makeSim(ctx, new FakeModel(), { maxInFlight: 2 });
                s.load(g, nanPositions(16));
                void s.step(2);
                await s.step(2);
                await s.flush();
                s.dispose();
            };
            await round();
            const steady = counter.live;
            await round();
            expect(counter.live).toBe(steady);
            ctx.release(g);
        } finally {
            ctx.dispose();
        }
        expect(counter.live).toBe(0);
        counter.restore();
    });

    it("device loss disposes the simulation and rejects the pending steps with E_DEVICE_LOST", async (t) => {
        requireGpu(t);
        const lossCtx = await acquire({ label: "force-simulation-loss" });
        const fake = heavyFake();
        const s = makeSim(lossCtx, fake, { maxInFlight: 2 }, EXACT_TIER);
        s.load(graph(HEAVY_N), nanPositions(HEAVY_N));
        await s.step(1); // compiles and binds on this FRESH context, so the two heavy submissions below are immediate
        fake.seen.length = 0;
        const p1 = s.step(256);
        const p2 = s.step(256);
        await waitForSubmission(s, 2);
        // both batches are submitted and still executing: their readbacks are pending at the loss
        expect(s.inFlight).toBe(2);
        expect(fake.seen).toEqual([]);
        lossCtx.device.destroy();
        expect((await rejectionOf(p1)).code).toBe("E_DEVICE_LOST");
        expect((await rejectionOf(p2)).code).toBe("E_DEVICE_LOST");
        await lossCtx.lost;
        await sleep(0);
        expect(s.state).toBe("disposed");
        expect(s.inFlight).toBe(0);
        expect((await rejectionOf(s.step(1))).code).toBe("E_DISPOSED");
        s.dispose();
    });
});
