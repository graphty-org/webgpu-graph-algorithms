/**
 * P3-T2 option tests for createForceAtlas2 (contract 5.5 row fa2-options.test.ts; spec 7.5, 7.8, 7.14, 7.17):
 * the four P3 registry entries, the option resolvers' defaults and every E_INVALID_ARGUMENT range, nodeSize ->
 * E_UNSUPPORTED, the repulsion tier rule at load(), dissuadeHubs ignored, the setParams rejections, a law change
 * recompiling and resetting the controller while a numeric tweak keeps it, the arcCount 0 path (fill instead of
 * K2, hand-computed two-node numbers), scale / center through the real toScene (spec 7.18, z = center.z in 2D),
 * the weights axis of contract 3.10, the stats shape with the radius pair against the f64 oracle, and every
 * override combination compiling on the device with its pipeline keys covered by OVERRIDE_MATRIX.
 *
 * Every GPU test acquires a FRESH context (spec 11.2) and releases its snapshots; a wrong result is never a skip.
 */

import { type F32, type GraphSnapshot } from "@graphty/graph-format";

import { EXACT_MAX_NODES, MAX_ITERATIONS_PER_STEP, TRACE_RECORD_BYTES } from "../../src/constants.js";
import { type GpuContext } from "../../src/context.js";
import { WebGpuGraphError, type WebGpuGraphErrorCode } from "../../src/errors.js";
import { type CommandBatch } from "../../src/kernel/batch.js";
import { FA2_STATE, FA2_TRACE, KERNELS, kernelSpec } from "../../src/kernels.js";
import { ForceSimulation, type StateWriter } from "../../src/layouts/force-simulation.js";
import {
    createForceAtlas2,
    ForceAtlas2Model,
    resolveForceAtlas2Options,
    resolveLayoutTuning,
} from "../../src/layouts/forceatlas2.js";
import { seedPositions } from "../../src/layouts/seed.js";
import { type ForceAtlas2Stats, type GpuLayoutSimulation, type GpuLayoutTuning } from "../../src/types/layout.js";
import { type ForceAtlas2Options } from "../../src/types/options.js";
import { KARATE_EDGES, pathEdges, snapshotOf } from "../helpers/graphs.js";
import { expectAllClose, expectBitwiseEqual } from "../helpers/matchers.js";
import { noiseFloorFor } from "../helpers/noise-floor.js";
import { matrixCovers } from "../helpers/override-matrix.js";
import { ForceAtlas2Oracle } from "../oracle/forceatlas2.js";
import { acquire, requireGpu } from "../setup/gpu.js";

// ============================================================ helpers

type Fa2Sim = ForceSimulation<ForceAtlas2Options, ForceAtlas2Stats>;

/** A JS-shaped option record (out-of-type values for the range tests). */
function loose(record: Record<string, unknown>): ForceAtlas2Options & GpuLayoutTuning {
    return record;
}

function assertCode(caught: unknown, code: WebGpuGraphErrorCode, details?: Record<string, unknown>): void {
    expect(caught).toBeInstanceOf(WebGpuGraphError);
    if (!(caught instanceof WebGpuGraphError)) {
        return;
    }
    expect(caught.code).toBe(code);
    if (details !== undefined) {
        expect(caught.details).toMatchObject(details);
    }
}

function expectCode(fn: () => unknown, code: WebGpuGraphErrorCode, details?: Record<string, unknown>): void {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    assertCode(caught, code, details);
}

async function expectRejects(
    promise: Promise<unknown>,
    code: WebGpuGraphErrorCode,
    details?: Record<string, unknown>,
): Promise<void> {
    let caught: unknown = null;
    try {
        await promise;
    } catch (err) {
        caught = err;
    }
    assertCode(caught, code, details);
}

/** Narrows the public simulation to the class so the @internal members (model, options, inspect, debugRunStages) are reachable. */
function asSim(sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>): Fa2Sim {
    if (!(sim instanceof ForceSimulation)) {
        throw new Error("createForceAtlas2 must return a ForceSimulation");
    }
    return sim;
}

function stageRunner(sim: Fa2Sim): (upTo: string) => Promise<void> {
    const { debugRunStages } = sim;
    if (debugRunStages === undefined) {
        throw new Error("ctx.debug.inspect must be true before createForceAtlas2");
    }
    return (upTo) => debugRunStages.call(sim, upTo);
}

function inspector(sim: Fa2Sim): (name: string) => Promise<Float32Array | Uint32Array> {
    const { inspect } = sim;
    if (inspect === undefined) {
        throw new Error("ctx.debug.inspect must be true before createForceAtlas2");
    }
    return (name) => inspect.call(sim, name);
}

function nanPositions(n: number): F32 {
    return new Float32Array(3 * n).fill(Number.NaN);
}

/** Scene-unit start positions seeded by the package's own LCG (the same start the GPU load() would produce). */
function seeded(s: GraphSnapshot, seed: number): F32 {
    const positions = nanPositions(s.nodeCount);
    seedPositions(s, positions, seed, 2, 1, null, "fa2");
    return positions;
}

async function oneIteration(
    ctx: GpuContext,
    s: GraphSnapshot,
    start: F32,
    options: ForceAtlas2Options & GpuLayoutTuning,
): Promise<F32> {
    const sim = createForceAtlas2(ctx, options);
    const positions = start.slice();
    sim.load(s, positions);
    await sim.step(1);
    sim.dispose();
    return positions;
}

function stateWord(state: Float32Array | Uint32Array, field: string): number {
    return state[FA2_STATE.offsetOf(field) / 4];
}

/** Karate with integer weights 1..10 in edge order (edge e carries 1 + e % 10). */
const WEIGHTED_KARATE: (readonly [number, number, number])[] = KARATE_EDGES.map(
    ([a, b], e) => [a, b, 1 + (e % 10)] as const,
);
const KARATE_WEIGHTS: F32 = Float32Array.from(WEIGHTED_KARATE, (e) => e[2]);

class FakeWriter implements StateWriter {
    readonly writes: [string, number | readonly number[]][] = [];
    set(field: string, value: number | readonly number[]): void {
        this.writes.push([field, value]);
    }
    get(field: string): number | readonly number[] {
        for (let i = this.writes.length - 1; i >= 0; i--) {
            if (this.writes[i][0] === field) {
                return this.writes[i][1];
            }
        }
        return 0;
    }
}

const DEFAULT_RESOLVED = {
    maxIter: 100,
    jitterTolerance: 1,
    scalingRatio: 2,
    gravity: 1,
    strongGravity: false,
    distributedAction: false,
    linlog: false,
    nodeMass: null,
    nodeSize: null,
    weight: null,
    dissuadeHubs: false,
    dim: 2,
    scale: 1,
    center: [0, 0, 0],
    seed: null,
    settleThreshold: 0.001,
    settleWindow: 10,
    iterationsPerStep: 1,
    maxInFlight: 2,
};

const DEFAULT_TUNING = {
    repulsion: "auto",
    exactMaxNodes: EXACT_MAX_NODES,
    nearMax: 64,
    deterministic: true,
    gridMax2D: 512,
    gridMax3D: 128,
    extentFactor: 6,
    compat: "paper",
};

// ============================================================ the P3 registry entries

const P3_IDS = ["fa2-stats-finalize", "fa2-attraction", "fa2-integrate", "fa2-to-scene"] as const;

type BindingRow = readonly [group: number, binding: number, name: string, kind: string, wgslType: string];

interface EntryTable {
    readonly entryPoint: string;
    readonly bindings: readonly BindingRow[];
    readonly overrides: readonly string[];
    readonly uniforms: readonly string[];
    readonly needs: readonly string[];
    readonly storage: number;
}

const GRAPH_GROUP: readonly BindingRow[] = [
    [0, 0, "rowPtr", "storage-ro", "array<u32>"],
    [0, 1, "colIdx", "storage-ro", "array<u32>"],
    [0, 2, "weights", "storage-ro", "array<f32>"],
    [0, 3, "perm", "storage-ro", "array<u32>"],
];

/** Contract 3.10.1, the four P3 rows. */
const P3_ENTRIES: Record<(typeof P3_IDS)[number], EntryTable> = {
    "fa2-stats-finalize": {
        entryPoint: "stats_finalize",
        bindings: [
            [1, 0, "partials", "storage-ro", "array<Fa2Partial>"],
            [1, 1, "S", "storage", "Fa2State"],
            [1, 2, "T", "storage", "array<Fa2Trace>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrides: [],
        uniforms: ["Fa2Params", "Fa2State", "Fa2Trace", "Fa2Partial"],
        needs: ["subgroups"],
        storage: 3,
    },
    "fa2-attraction": {
        entryPoint: "attraction",
        bindings: [
            ...GRAPH_GROUP,
            [1, 0, "pos", "storage-ro", "array<vec4f>"],
            [1, 1, "force", "storage", "array<f32>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrides: ["LINLOG:bool=false", "DISTRIBUTED:bool=false", "TIER:u32=0"],
        uniforms: ["Fa2Params"],
        needs: [],
        storage: 6,
    },
    "fa2-integrate": {
        entryPoint: "integrate",
        bindings: [
            [1, 0, "force", "storage-ro", "array<f32>"],
            [1, 1, "oldForce", "storage", "array<f32>"],
            [1, 2, "fixedMask", "storage-ro", "array<u32>"],
            [1, 3, "S", "storage", "Fa2State"],
            [1, 4, "pos", "storage", "array<vec4f>"],
            [1, 5, "partials", "storage", "array<Fa2Partial>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrides: ["SWING_MODE:u32=0"],
        uniforms: ["Fa2Params", "Fa2State", "Fa2Partial"],
        needs: ["subgroups"],
        storage: 6,
    },
    "fa2-to-scene": {
        entryPoint: "to_scene",
        bindings: [
            [1, 0, "pos", "storage-ro", "array<vec4f>"],
            [1, 1, "scene", "storage", "array<f32>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrides: [],
        uniforms: ["Fa2Params"],
        needs: [],
        storage: 2,
    },
};

describe("the P3 registry entries (contract 3.10.1)", () => {
    it("declares the four entries with the binding tables, override declarations, uniforms and needs of 3.10.1", () => {
        for (const id of P3_IDS) {
            const entry = KERNELS[id];
            const want = P3_ENTRIES[id];
            expect(entry.id).toBe(id);
            expect(entry.phase).toBe("P3");
            expect(entry.entryPoint).toBe(want.entryPoint);
            expect(entry.bindings.map((b) => [b.group, b.binding, b.name, b.kind, b.wgslType])).toEqual(want.bindings);
            expect(entry.overrideDecls.map((d) => `${d.name}:${d.type}=${String(d.default)}`)).toEqual(want.overrides);
            expect(entry.uniforms.map((u) => u.name)).toEqual(want.uniforms);
            expect(entry.needs).toEqual(want.needs);
            expect(entry.snippetSlots).toEqual([]);
            expect(entry.bindings.filter((b) => b.kind !== "uniform")).toHaveLength(want.storage);
            expect(entry.body).not.toContain("@group(");
            expect(entry.body).not.toContain("override ");
            expect(entry.body).toContain(`fn ${want.entryPoint}(`);
        }
    });

    it("compiles each entry with its defaults on the device", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        for (const id of P3_IDS) {
            await ctx.pipelines.get(kernelSpec(id));
        }
        expect(ctx.pipelines.size).toBe(P3_IDS.length);
    });
});

// ============================================================ the resolvers (pure)

describe("resolveForceAtlas2Options (contract 3.13; spec 7.14)", () => {
    it("applies FA2_DEFAULTS and the null / origin defaults; explicit undefined counts as absent", () => {
        expect(resolveForceAtlas2Options(undefined)).toEqual(DEFAULT_RESOLVED);
        expect(resolveForceAtlas2Options({})).toEqual(DEFAULT_RESOLVED);
        expect(resolveForceAtlas2Options({ gravity: undefined, seed: undefined, center: undefined })).toEqual(
            DEFAULT_RESOLVED,
        );
        expect(Object.isFrozen(resolveForceAtlas2Options({}))).toBe(true);
    });

    it("keeps every given value, widens a 2-vector center and accepts a typed-array center", () => {
        const resolved = resolveForceAtlas2Options({
            maxIter: 7,
            jitterTolerance: 0.5,
            scalingRatio: 4,
            gravity: 0,
            strongGravity: true,
            distributedAction: true,
            linlog: true,
            nodeMass: "mass",
            weight: "w",
            dissuadeHubs: true,
            dim: 3,
            scale: 100,
            center: [1, 2],
            seed: 42,
            settleThreshold: 0,
            settleWindow: 3,
            iterationsPerStep: 4,
            maxInFlight: 1,
        });
        expect(resolved).toEqual({
            maxIter: 7,
            jitterTolerance: 0.5,
            scalingRatio: 4,
            gravity: 0,
            strongGravity: true,
            distributedAction: true,
            linlog: true,
            nodeMass: "mass",
            nodeSize: null,
            weight: "w",
            dissuadeHubs: true,
            dim: 3,
            scale: 100,
            center: [1, 2, 0],
            seed: 42,
            settleThreshold: 0,
            settleWindow: 3,
            iterationsPerStep: 4,
            maxInFlight: 1,
        });
        expect(resolveForceAtlas2Options({ center: new Float32Array([1, 2, 3]) }).center).toEqual([1, 2, 3]);
        expect(resolveForceAtlas2Options({ seed: 0 }).seed).toBe(0);
        expect(resolveForceAtlas2Options({ seed: null }).seed).toBeNull();
        expect(resolveForceAtlas2Options({ iterationsPerStep: MAX_ITERATIONS_PER_STEP }).iterationsPerStep).toBe(
            MAX_ITERATIONS_PER_STEP,
        );
    });

    it("rejects every out-of-range field with E_INVALID_ARGUMENT naming the argument", () => {
        const bad: [Record<string, unknown>, string][] = [
            [{ gravity: -1 }, "gravity"],
            [{ gravity: Number.NaN }, "gravity"],
            [{ gravity: "1" }, "gravity"],
            [{ scalingRatio: 0 }, "scalingRatio"],
            [{ scalingRatio: Number.POSITIVE_INFINITY }, "scalingRatio"],
            [{ jitterTolerance: 0 }, "jitterTolerance"],
            [{ maxIter: 0 }, "maxIter"],
            [{ maxIter: 1.5 }, "maxIter"],
            [{ settleWindow: 0 }, "settleWindow"],
            [{ settleThreshold: -0.001 }, "settleThreshold"],
            [{ maxInFlight: 0 }, "maxInFlight"],
            [{ maxInFlight: 2.5 }, "maxInFlight"],
            [{ iterationsPerStep: 0 }, "iterationsPerStep"],
            [{ iterationsPerStep: MAX_ITERATIONS_PER_STEP + 1 }, "iterationsPerStep"],
            [{ dim: 4 }, "dim"],
            [{ dim: 2.5 }, "dim"],
            [{ scale: 0 }, "scale"],
            [{ scale: -1 }, "scale"],
            [{ center: [1] }, "center"],
            [{ center: [1, 2, 3, 4] }, "center"],
            [{ center: [1, Number.NaN] }, "center"],
            [{ seed: Number.POSITIVE_INFINITY }, "seed"],
            [{ linlog: 1 }, "linlog"],
            [{ strongGravity: "yes" }, "strongGravity"],
        ];
        for (const [record, argument] of bad) {
            expectCode(() => resolveForceAtlas2Options(loose(record)), "E_INVALID_ARGUMENT", { argument });
        }
    });

    it("rejects nodeSize with E_UNSUPPORTED { option } and keeps nodeSize null otherwise", () => {
        expectCode(() => resolveForceAtlas2Options({ nodeSize: new Float32Array(3) }), "E_UNSUPPORTED", {
            option: "nodeSize",
        });
        expectCode(() => resolveForceAtlas2Options({ nodeSize: "size" }), "E_UNSUPPORTED", { option: "nodeSize" });
        expectCode(() => resolveForceAtlas2Options({ nodeSize: { a: 1 } }), "E_UNSUPPORTED", { option: "nodeSize" });
        let caught: unknown = null;
        try {
            resolveForceAtlas2Options({ nodeSize: "size" });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(WebGpuGraphError);
        if (caught instanceof WebGpuGraphError) {
            expect(typeof caught.details.hint).toBe("string");
            expect(caught.details.feature).toBeUndefined();
        }
        expect(resolveForceAtlas2Options({ nodeSize: null }).nodeSize).toBeNull();
    });

    it("merges a patch over a previous record and pins maxInFlight", () => {
        const previous = resolveForceAtlas2Options({ gravity: 3, seed: 5, maxInFlight: 4, dim: 3 });
        expect(resolveForceAtlas2Options({ linlog: true }, previous)).toEqual({ ...previous, linlog: true });
        expect(resolveForceAtlas2Options({}, previous)).toEqual(previous);
        expect(resolveForceAtlas2Options({ maxInFlight: 4 }, previous)).toEqual(previous);
        expect(resolveForceAtlas2Options({ seed: null }, previous).seed).toBeNull();
        expectCode(() => resolveForceAtlas2Options({ maxInFlight: 3 }, previous), "E_INVALID_ARGUMENT", {
            argument: "maxInFlight",
            value: 3,
            expected: 4,
        });
        expectCode(() => resolveForceAtlas2Options({ gravity: -2 }, previous), "E_INVALID_ARGUMENT", {
            argument: "gravity",
        });
    });
});

describe("resolveLayoutTuning (contract 3.13; spec 7.14)", () => {
    it("applies LAYOUT_TUNING_DEFAULTS", () => {
        expect(resolveLayoutTuning(undefined)).toEqual(DEFAULT_TUNING);
        expect(resolveLayoutTuning({})).toEqual(DEFAULT_TUNING);
        expect(Object.isFrozen(resolveLayoutTuning({}))).toBe(true);
        expect(
            resolveLayoutTuning({
                repulsion: "exact",
                exactMaxNodes: 8,
                nearMax: 16,
                deterministic: false,
                gridMax2D: 64,
                gridMax3D: 32,
                extentFactor: 4,
                compat: "networkx",
            }),
        ).toEqual({
            repulsion: "exact",
            exactMaxNodes: 8,
            nearMax: 16,
            deterministic: false,
            gridMax2D: 64,
            gridMax3D: 32,
            extentFactor: 4,
            compat: "networkx",
        });
    });

    it("rejects an unknown enum or an out-of-range number with E_INVALID_ARGUMENT", () => {
        const bad: [Record<string, unknown>, string][] = [
            [{ repulsion: "bogus" }, "repulsion"],
            [{ compat: "gephi" }, "compat"],
            [{ exactMaxNodes: 0 }, "exactMaxNodes"],
            [{ exactMaxNodes: 1.5 }, "exactMaxNodes"],
            [{ nearMax: 0 }, "nearMax"],
            [{ gridMax2D: 0 }, "gridMax2D"],
            [{ gridMax3D: -1 }, "gridMax3D"],
            [{ extentFactor: 0 }, "extentFactor"],
            [{ deterministic: "no" }, "deterministic"],
        ];
        for (const [record, argument] of bad) {
            expectCode(() => resolveLayoutTuning(loose(record)), "E_INVALID_ARGUMENT", { argument });
        }
    });
});

// ============================================================ the model without a device

describe("ForceAtlas2Model (no device; contract 3.13)", () => {
    const paper = new ForceAtlas2Model(resolveLayoutTuning(undefined), resolveForceAtlas2Options(undefined));
    const networkx = new ForceAtlas2Model(
        resolveLayoutTuning({ compat: "networkx" }),
        resolveForceAtlas2Options(undefined),
    );

    it("has the FA2 kind, stages, blocks and the three model buffers", () => {
        expect(paper.kind).toBe("forceatlas2");
        expect(paper.stages).toEqual(["K1", "K2", "K3", "K4", "K5", "toScene"]);
        expect(paper.params.name).toBe("Fa2Params");
        expect(paper.state.name).toBe("Fa2State");
        expect(paper.trace.name).toBe("Fa2Trace");
        // the simulation sizes the trace region by model.trace.byteLength; the 3.2 constant stays the FA2 truth
        expect(FA2_TRACE.byteLength).toBe(TRACE_RECORD_BYTES);
        const specs = paper.buffers(34, 2);
        expect(specs.map((b) => [b.name, b.byteLength, b.zero])).toEqual([
            ["force", 408, true],
            ["oldForce", 408, true],
            ["fillParams", 256, false],
        ]);
        expect(networkx.buffers(34, 3).map((b) => b.name)).toEqual(["force", "oldForce", "fillParams"]);
        expect(paper.buffers(0, 2).map((b) => b.byteLength)).toEqual([12, 12, 256]);
    });

    it("compiles the option record to the override set of spec 7.2 with the 4.6 corrections in networkx mode", () => {
        expect(paper.overrides(resolveForceAtlas2Options(undefined))).toEqual({
            LINLOG: false,
            DISTRIBUTED: false,
            TIER: 0,
            SWING_MODE: 0,
            STRONG_GRAVITY: false,
            GRAVITY_CENTER: 0,
        });
        expect(
            paper.overrides(resolveForceAtlas2Options({ linlog: true, distributedAction: true, strongGravity: true })),
        ).toEqual({
            LINLOG: true,
            DISTRIBUTED: true,
            TIER: 0,
            SWING_MODE: 0,
            STRONG_GRAVITY: true,
            GRAVITY_CENTER: 0,
        });
        expect(networkx.overrides(resolveForceAtlas2Options(undefined))).toEqual({
            LINLOG: false,
            DISTRIBUTED: false,
            TIER: 0,
            SWING_MODE: 1,
            STRONG_GRAVITY: false,
            GRAVITY_CENTER: 1,
        });
    });

    it("hands each kernel only the override names its entry declares (K2 plus USE_PERM / HAS_WEIGHTS)", () => {
        const merged = {
            LINLOG: true,
            DISTRIBUTED: false,
            TIER: 0,
            SWING_MODE: 1,
            STRONG_GRAVITY: true,
            GRAVITY_CENTER: 1,
            USE_PERM: false,
            HAS_WEIGHTS: true,
        };
        const specs = networkx.specs(merged, true);
        expect(specs.map((s) => s.id)).toEqual([
            "fa2-stats-finalize",
            "fa2-attraction",
            "fa2-repulsion-exact",
            "fa2-speed-finalize",
            "fa2-integrate",
            "fa2-to-scene",
            "fill",
        ]);
        expect(specs.map((s) => s.overrides)).toEqual([
            {},
            { LINLOG: true, DISTRIBUTED: false, TIER: 0, USE_PERM: false, HAS_WEIGHTS: true },
            { SWING_MODE: 1, STRONG_GRAVITY: true, GRAVITY_CENTER: 1 },
            { SWING_MODE: 1 },
            { SWING_MODE: 1 },
            {},
            {},
        ]);
        const bare = paper.specs(paper.overrides(resolveForceAtlas2Options(undefined)), false);
        expect(bare[1].overrides).toEqual({
            LINLOG: false,
            DISTRIBUTED: false,
            TIER: 0,
            USE_PERM: false,
            HAS_WEIGHTS: false,
        });
        expect(bare[2].overrides).toEqual({ SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 });
    });

    it("resets the controller on load, on reheat only in networkx mode, and on setParams only when a law changed", () => {
        const w = new FakeWriter();
        paper.onLoad(w);
        expect(w.writes).toEqual([
            ["speed", 1],
            ["speedEfficiency", 1],
            ["swing", 1],
            ["traction", 1],
        ]);
        const r0 = new FakeWriter();
        paper.onReheat(r0);
        expect(r0.writes).toEqual([]);
        const r1 = new FakeWriter();
        networkx.onReheat(r1);
        expect(r1.writes).toEqual([
            ["swing", 1],
            ["traction", 1],
        ]);
        const model = new ForceAtlas2Model(resolveLayoutTuning(undefined), resolveForceAtlas2Options(undefined));
        const tweak = new FakeWriter();
        model.onSetParams({ gravity: 5, scalingRatio: 9, jitterTolerance: 2 }, tweak);
        expect(tweak.writes).toEqual([]);
        const law = new FakeWriter();
        model.onSetParams({ linlog: true }, law);
        expect(law.writes).toEqual([
            ["speed", 1],
            ["speedEfficiency", 1],
        ]);
        const same = new FakeWriter();
        model.onSetParams({ linlog: true }, same);
        expect(same.writes).toEqual([]);
        const strong = new FakeWriter();
        model.onSetParams({ strongGravity: true, gravity: 2 }, strong);
        expect(strong.writes).toEqual([
            ["speed", 1],
            ["speedEfficiency", 1],
        ]);
        const distributed = new FakeWriter();
        model.onSetParams({ distributedAction: true }, distributed);
        expect(distributed.writes).toHaveLength(2);
        expectCode(
            () => {
                model.onSetParams({ maxInFlight: 9 }, new FakeWriter());
            },
            "E_INVALID_ARGUMENT",
            { argument: "maxInFlight" },
        );
    });

    it("decodes the state header and k trace records through the generated blocks", () => {
        const bytes = new ArrayBuffer(FA2_STATE.byteLength + 2 * FA2_TRACE.byteLength);
        const view = new DataView(bytes);
        FA2_STATE.write(view, {
            speed: 1.5,
            speedEfficiency: 0.75,
            swing: 4,
            traction: 2,
            centroid: [0.25, -0.5, 0.125, 0],
            rmsRadius: 0.875,
            radius: 1.75,
            meanDisplacement: 0.0625,
            iteration: 12,
            min: [-1, -2, 0, 0],
            max: [3, 4, 0, 0],
            settledCount: 3,
        });
        FA2_TRACE.write(
            view,
            {
                swing: 4,
                traction: 2,
                speed: 1.25,
                speedEfficiency: 0.5,
                meanDisplacement: 0.03125,
                settledCount: 1,
                iteration: 11,
            },
            FA2_STATE.byteLength,
        );
        FA2_TRACE.write(
            view,
            {
                swing: 3,
                traction: 1.5,
                speed: 1.5,
                speedEfficiency: 0.75,
                meanDisplacement: 0.0625,
                settledCount: 3,
                iteration: 12,
            },
            FA2_STATE.byteLength + FA2_TRACE.byteLength,
        );
        const stats = paper.readStats(
            new DataView(bytes, 0, FA2_STATE.byteLength),
            new DataView(bytes, FA2_STATE.byteLength, 2 * FA2_TRACE.byteLength),
        );
        expect(stats).toEqual({
            iteration: 12,
            meanDisplacement: 0.0625,
            rmsRadius: 0.875,
            layoutRadius: 1.75,
            centroid: [0.25, -0.5, 0.125],
            repulsionTier: "exact",
            maxCellOccupancy: null,
            outsideGrid: null,
            msPerIteration: null,
            swing: 4,
            traction: 2,
            speed: 1.5,
            speedEfficiency: 0.75,
            trace: [
                {
                    swing: 4,
                    traction: 2,
                    speed: 1.25,
                    speedEfficiency: 0.5,
                    meanDisplacement: 0.03125,
                    settledCount: 1,
                },
                {
                    swing: 3,
                    traction: 1.5,
                    speed: 1.5,
                    speedEfficiency: 0.75,
                    meanDisplacement: 0.0625,
                    settledCount: 3,
                },
            ],
        });
        const empty = paper.readStats(new DataView(bytes, 0, FA2_STATE.byteLength), new DataView(bytes, 0, 0));
        expect(empty.trace).toEqual([]);
    });

    it("refuses paramsFor before bind() with E_NOT_LOADED", () => {
        const model = new ForceAtlas2Model(resolveLayoutTuning(undefined), resolveForceAtlas2Options(undefined));
        expectCode(() => model.paramsFor(0, resolveForceAtlas2Options(undefined)), "E_NOT_LOADED", {
            state: "created",
        });
    });

    it("inputs() applies the tier rule itself (ForceSimulation.load() checks first, so this is the model's own guard)", () => {
        const ten = snapshotOf(pathEdges(10));
        const options = resolveForceAtlas2Options(undefined);
        const modelFor = (tuning: GpuLayoutTuning): ForceAtlas2Model =>
            new ForceAtlas2Model(resolveLayoutTuning(tuning), options);
        expectCode(() => modelFor({ exactMaxNodes: 8 }).inputs(ten, options), "E_UNSUPPORTED", {
            feature: "repulsion.grid",
        });
        expectCode(() => modelFor({ exactMaxNodes: 8, repulsion: "auto" }).inputs(ten, options), "E_UNSUPPORTED", {
            feature: "repulsion.grid",
        });
        expectCode(() => modelFor({ repulsion: "grid" }).inputs(ten, options), "E_UNSUPPORTED", {
            feature: "repulsion.grid",
        });
        expectCode(() => modelFor({ exactMaxNodes: 64, repulsion: "grid" }).inputs(ten, options), "E_UNSUPPORTED", {
            feature: "repulsion.grid",
        });
        expect(modelFor({ exactMaxNodes: 10 }).inputs(ten, options).mass).toHaveLength(10);
        expect(modelFor({ exactMaxNodes: 8, repulsion: "exact" }).inputs(ten, options).mass).toHaveLength(10);
        const inputs = modelFor({}).inputs(ten, options);
        expect(inputs.mass).toHaveLength(10);
        expect(inputs.weights).toEqual({ data: null, source: "none", column: null });
        // recordIteration refuses the grid tier before it touches the batch or the resources (never bound here)
        expectCode(() => modelFor({}).recordIteration({} as CommandBatch, 0, "grid"), "E_UNSUPPORTED", {
            feature: "repulsion.grid",
        });
    });
});

// ============================================================ creation and the tier rule

describe("createForceAtlas2: creation errors and the repulsion tier rule (spec 7.8, 7.14)", () => {
    it("rejects nodeSize and out-of-range options at creation, accepts nodeSize null and dissuadeHubs", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        expectCode(() => createForceAtlas2(ctx, { nodeSize: new Float32Array(34) }), "E_UNSUPPORTED", {
            option: "nodeSize",
        });
        expectCode(() => createForceAtlas2(ctx, { nodeSize: "size" }), "E_UNSUPPORTED", { option: "nodeSize" });
        expectCode(() => createForceAtlas2(ctx, { gravity: -1 }), "E_INVALID_ARGUMENT", { argument: "gravity" });
        expectCode(() => createForceAtlas2(ctx, loose({ compat: "gephi" })), "E_INVALID_ARGUMENT", {
            argument: "compat",
        });
        expectCode(() => createForceAtlas2(ctx, { maxInFlight: 0 }), "E_INVALID_ARGUMENT", {
            argument: "maxInFlight",
        });
        const sim = createForceAtlas2(ctx, {
            nodeSize: null,
            dissuadeHubs: true,
            exactMaxNodes: 99,
            compat: "networkx",
        });
        const narrowed = asSim(sim);
        expect(narrowed.tuning.exactMaxNodes).toBe(99);
        expect(narrowed.tuning.compat).toBe("networkx");
        expect(narrowed.options.dissuadeHubs).toBe(true);
        expect(narrowed.model.kind).toBe("forceatlas2");
        sim.dispose();
    });

    it("throws E_DISPOSED on a disposed context", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        ctx.dispose();
        expectCode(() => createForceAtlas2(ctx), "E_DISPOSED");
    });

    it("applies the tier rule at load(): auto above exactMaxNodes and an explicit grid throw E_UNSUPPORTED { feature }, exact never", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const ten = snapshotOf(pathEdges(10));
        expect(ten.nodeCount).toBe(10);
        const cases: [GpuLayoutTuning, boolean][] = [
            [{ exactMaxNodes: 8 }, false],
            [{ exactMaxNodes: 8, repulsion: "auto" }, false],
            [{ exactMaxNodes: 8, repulsion: "grid" }, false],
            [{ exactMaxNodes: 64, repulsion: "grid" }, false],
            [{ exactMaxNodes: 10 }, true],
            [{ exactMaxNodes: 8, repulsion: "exact" }, true],
            [{}, true],
        ];
        for (const [tuning, ok] of cases) {
            const sim = createForceAtlas2(ctx, tuning);
            const positions = nanPositions(10);
            if (ok) {
                sim.load(ten, positions);
                await sim.step(1);
                expect(sim.stats.repulsionTier).toBe("exact");
                for (const v of positions) {
                    expect(Number.isFinite(v)).toBe(true);
                }
            } else {
                expectCode(
                    () => {
                        sim.load(ten, positions);
                    },
                    "E_UNSUPPORTED",
                    { feature: "repulsion.grid" },
                );
                await expectRejects(sim.step(1), "E_NOT_LOADED");
            }
            sim.dispose();
        }
        ctx.release(ten);
    });

    it("seeds unseeded rows deterministically per seed; seed null and seed 0 still seed every row", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        const load = (options: ForceAtlas2Options): F32 => {
            const sim = createForceAtlas2(ctx, options);
            const positions = nanPositions(s.nodeCount);
            sim.load(s, positions);
            sim.dispose();
            return positions;
        };
        const a = load({ seed: 7 });
        const b = load({ seed: 7 });
        expectBitwiseEqual(a, b, "seed 7 twice");
        expectBitwiseEqual(a, seeded(s, 7), "seed 7 equals seedPositions");
        for (const positions of [load({ seed: null }), load({ seed: 0 }), load({})]) {
            for (const v of positions) {
                expect(Number.isFinite(v)).toBe(true);
            }
        }
        ctx.release(s);
    });
});

// ============================================================ live option changes

describe("createForceAtlas2: dissuadeHubs and setParams (spec 7.13, 7.14, 7.17)", () => {
    it("accepts and ignores dissuadeHubs: no extra pipeline and bitwise identical positions", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        const start = seeded(s, 7);
        const plain = await oneIteration(ctx, s, start, {});
        const pipelines = ctx.pipelines.size;
        const hubs = await oneIteration(ctx, s, start, { dissuadeHubs: true });
        expect(ctx.pipelines.size).toBe(pipelines);
        expectBitwiseEqual(plain, hubs, "dissuadeHubs is ignored");
        ctx.release(s);
    });

    it("toScene applies scale and center (scene = p * scale + center); 2D writes z = center.z", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        // both loads seed the SAME LCG rows in layout units (seedPositions writes v * scale + center, load() applies
        // the inverse), so the layout positions agree up to the f32 rounding of that round trip and the scene
        // arrays differ by exactly the affine map of spec 7.18; z is written from P.center.z in 2D (spec 7.13)
        const plain = await oneIteration(ctx, s, nanPositions(s.nodeCount), { seed: 7 });
        const scaled = await oneIteration(ctx, s, nanPositions(s.nodeCount), {
            seed: 7,
            scale: 100,
            center: [1, 2, 3],
        });
        for (let i = 0; i < s.nodeCount; i++) {
            expectAllClose(
                [scaled[3 * i], scaled[3 * i + 1]],
                [plain[3 * i] * 100 + 1, plain[3 * i + 1] * 100 + 2],
                { rel: 1e-4, abs: 1e-4 },
                `row ${i} (2D)`,
            );
            expect(scaled[3 * i + 2]).toBe(3);
        }
        expect(Array.from(plain).some((v, i) => i % 3 !== 2 && v !== 0)).toBe(true);
        ctx.release(s);
    });

    it("toScene applies scale and center to all three axes in 3D", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        const plain = await oneIteration(ctx, s, nanPositions(s.nodeCount), { seed: 7, dim: 3 });
        const scaled = await oneIteration(ctx, s, nanPositions(s.nodeCount), {
            seed: 7,
            dim: 3,
            scale: 100,
            center: [1, 2, 3],
        });
        for (let i = 0; i < s.nodeCount; i++) {
            expectAllClose(
                [scaled[3 * i], scaled[3 * i + 1], scaled[3 * i + 2]],
                [plain[3 * i] * 100 + 1, plain[3 * i + 1] * 100 + 2, plain[3 * i + 2] * 100 + 3],
                { rel: 1e-4, abs: 1e-4 },
                `row ${i} (3D)`,
            );
        }
        expect(Array.from(plain).some((v, i) => i % 3 === 2 && v !== 0)).toBe(true);
        ctx.release(s);
    });

    it("rejects setParams({ dim }), setParams({ maxInFlight }) and setParams({ nodeSize }); unchanged values pass", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        const sim = createForceAtlas2(ctx, { dim: 2, maxInFlight: 2 });
        sim.load(s, seeded(s, 7));
        expectCode(() => {
            sim.setParams({ dim: 3 });
        }, "E_INVALID_ARGUMENT");
        expectCode(
            () => {
                sim.setParams({ maxInFlight: 3 });
            },
            "E_INVALID_ARGUMENT",
            { argument: "maxInFlight" },
        );
        expectCode(
            () => {
                sim.setParams({ nodeSize: new Float32Array(s.nodeCount) });
            },
            "E_UNSUPPORTED",
            { option: "nodeSize" },
        );
        expectCode(
            () => {
                sim.setParams({ gravity: -1 });
            },
            "E_INVALID_ARGUMENT",
            { argument: "gravity" },
        );
        sim.setParams({ dim: 2 });
        sim.setParams({ maxInFlight: 2 });
        sim.setParams({ gravity: 2 });
        await sim.step(1);
        expect(sim.iterationsDone).toBe(1);
        sim.dispose();
        ctx.release(s);
    });

    it("a force-law change recompiles and resets the controller; a numeric tweak keeps both", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        ctx.debug.inspect = true;
        const s = snapshotOf(KARATE_EDGES);
        const sim = asSim(createForceAtlas2(ctx, { seed: 7 }));
        const run = stageRunner(sim);
        const read = inspector(sim);
        sim.load(s, seeded(s, 7));
        await sim.step(5);
        await run("K1");
        const running = await read("state");
        expect(stateWord(running, "speed")).not.toBe(1);
        const pipelines = ctx.pipelines.size;

        sim.setParams({ gravity: 2, scalingRatio: 3 });
        await run("K1");
        const afterTweak = await read("state");
        expect(stateWord(afterTweak, "speed")).toBe(stateWord(running, "speed"));
        expect(stateWord(afterTweak, "speedEfficiency")).toBe(stateWord(running, "speedEfficiency"));
        expect(ctx.pipelines.size).toBe(pipelines);

        sim.setParams({ linlog: true });
        await run("K1");
        const afterLaw = await read("state");
        expect(stateWord(afterLaw, "speed")).toBe(1);
        expect(stateWord(afterLaw, "speedEfficiency")).toBe(1);
        expect(ctx.pipelines.size).toBe(pipelines + 1);
        expect(ctx.pipelines.keys().some((k) => k.startsWith("fa2-attraction|") && k.includes('"LINLOG":true'))).toBe(
            true,
        );

        await sim.step(3);
        await run("K1");
        const moved = await read("state");
        expect(stateWord(moved, "speed")).not.toBe(1);
        sim.setParams({ linlog: true });
        await run("K1");
        const unchanged = await read("state");
        expect(stateWord(unchanged, "speed")).toBe(stateWord(moved, "speed"));
        expect(ctx.pipelines.size).toBe(pipelines + 1);
        sim.dispose();
        ctx.release(s);
    });
});

// ============================================================ arcCount 0

describe("createForceAtlas2: arcCount 0 (spec 7.5: fill instead of K2)", () => {
    it("runs two isolated nodes: the hand-computed repulsion after K3, controller after K4, positions after K5; the fill erases the previous force", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        ctx.debug.inspect = true;
        const s = snapshotOf([], { nodeCount: 2 });
        expect(s.arcCount).toBe(0);
        const start = new Float32Array([-0.5, 0, 0, 0.5, 0, 0]);
        const sim = asSim(createForceAtlas2(ctx, { gravity: 0, dim: 2 }));
        const run = stageRunner(sim);
        const read = inspector(sim);

        sim.load(s, start.slice());
        await run("K3");
        expectAllClose(await read("force"), [-2, 0, 0, 2, 0, 0], { rel: 0, abs: 1e-6 }, "force after K3");

        sim.load(s, start.slice());
        await run("K4");
        const state = await read("state");
        expectAllClose(
            ["speed", "speedEfficiency", "swing", "traction"].map((f) => stateWord(state, f)),
            [0.13295739742362472, 0.7, 4, 2],
            { rel: 1e-5, abs: 1e-6 },
            "controller after K4",
        );

        const out = start.slice();
        sim.load(s, out);
        await sim.step(1);
        expectAllClose(out, [-0.6754438123873462, 0, 0, 0.6754438123873462, 0, 0], { rel: 0, abs: 1e-5 }, "positions");
        const { stats } = sim;
        expect(stats.trace).toHaveLength(1);
        expectAllClose(
            [stats.swing, stats.traction, stats.speed, stats.speedEfficiency],
            [4, 2, 0.13295739742362472, 0.7],
            { rel: 1e-5, abs: 1e-6 },
            "stats controller",
        );
        expect(stats.meanDisplacement).toBe(0);
        expect(stats.iteration).toBe(1);
        expect(stats.trace[0].settledCount).toBe(0);

        // WITHOUT reloading (a reload re-zeroes every zero: true buffer, P3-T1 clearKept, and would hide a missing
        // fill): force still holds K3's (-2, 0, 0, 2, 0, 0) from step(1); the debug run's fill must erase it before
        // K3 accumulates, so after "K2" it is all zeros and after "K3" it is the repulsion at the NEW distance
        // 2 x 0.6754438 = 1.3508876 (k = 2, masses 1: 2 / 1.3508876 = 1.4805080), not -2 - 1.48 (accumulation)
        await run("K2");
        expect(Array.from(await read("force"))).toEqual([0, 0, 0, 0, 0, 0]);
        await run("K3");
        expectAllClose(
            await read("force"),
            [-1.480508047687216, 0, 0, 1.480508047687216, 0, 0],
            { rel: 1e-5, abs: 1e-6 },
            "force after K3 without a reload: the fill ran, nothing accumulated",
        );
        sim.dispose();
        ctx.release(s);
    });
});

// ============================================================ the weights axis

describe("createForceAtlas2: the weights axis (contract 3.10 CONTRACT DECISION; spec 7.5)", () => {
    it("weight false on a weighted snapshot equals the unweighted graph bitwise; weight true differs; an edge column equals arc weights bitwise", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const weighted = snapshotOf(WEIGHTED_KARATE);
        const unweighted = snapshotOf(KARATE_EDGES);
        expect(weighted.flags.weighted).toBe(true);
        expect(unweighted.flags.weighted).toBe(false);
        const withColumn = unweighted.withColumns(undefined, { w: KARATE_WEIGHTS });
        const start = seeded(unweighted, 7);

        const a = await oneIteration(ctx, weighted, start, { weight: false });
        const b = await oneIteration(ctx, unweighted, start, {});
        expectBitwiseEqual(a, b, "weight: false on a weighted snapshot == unweighted");

        const c = await oneIteration(ctx, weighted, start, { weight: true });
        expect(Array.from(c).some((v, i) => v !== a[i])).toBe(true);

        const d = await oneIteration(ctx, withColumn, start, { weight: "w" });
        expectBitwiseEqual(d, c, 'weight: "w" on an unweighted snapshot == arc weights');

        expect(ctx.pipelines.keys().filter((k) => k.startsWith("fa2-attraction|"))).toHaveLength(2);
        ctx.release(weighted);
        ctx.release(unweighted);
    });
});

// ============================================================ stats shape and the radius pair

describe("createForceAtlas2: stats shape and the radius pair vs the f64 oracle (spec 3.3; contract 4.5 K1)", () => {
    it("reports the exact tier with null grid fields, k trace records, the params values, and rmsRadius / layoutRadius within the traced tolerance", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const s = snapshotOf(KARATE_EDGES);
        const start = seeded(s, 7);
        const sim = asSim(createForceAtlas2(ctx, { seed: 7 }));
        const oracle = new ForceAtlas2Oracle(s, start, { compat: "paper", precision: "f64" });
        const positions = start.slice();
        sim.load(s, positions);
        const tolerance = noiseFloorFor("fa2-skeleton.force");
        for (let i = 0; i < 3; i++) {
            const record = oracle.step();
            await sim.step(1);
            const { stats } = sim;
            expect(stats.trace).toHaveLength(1);
            expect(stats.iteration).toBe(i + 1);
            expectAllClose(
                [stats.rmsRadius, stats.layoutRadius],
                [record.rmsRadius, record.layoutRadius],
                { rel: tolerance.value, abs: 0 },
                `radius pair after iteration ${i + 1}`,
            );
        }
        await sim.step(3);
        expect(sim.model.paramsFor(3, sim.options)).toMatchObject({
            n: 34,
            dim: 2,
            flags: 0,
            tierStart: 0,
            tierEnd: 34,
            iterationIndex: 3,
            seed: 7,
            nearMax: 64,
            scalingRatio: 2,
            gravity: 1,
            jitterTolerance: 1,
            scale: 1,
            center: [0, 0, 0, 0],
            settleThreshold: 0.001,
            extentFactor: 6,
        });
        const { stats } = sim;
        expect(stats.repulsionTier).toBe("exact");
        expect(stats.maxCellOccupancy).toBeNull();
        expect(stats.outsideGrid).toBeNull();
        expect(stats.msPerIteration === null || typeof stats.msPerIteration === "number").toBe(true);
        expect(stats.trace).toHaveLength(3);
        expect(stats.iteration).toBe(6);
        expect(stats.centroid).toHaveLength(3);
        for (const record of stats.trace) {
            for (const v of [
                record.swing,
                record.traction,
                record.speed,
                record.speedEfficiency,
                record.meanDisplacement,
            ]) {
                expect(Number.isFinite(v)).toBe(true);
            }
            expect(Number.isInteger(record.settledCount)).toBe(true);
        }
        expect(stats.speed).toBe(stats.trace[2].speed);
        expect(stats.speedEfficiency).toBe(stats.trace[2].speedEfficiency);
        expect(stats.swing).toBe(stats.trace[2].swing);
        expect(stats.traction).toBe(stats.trace[2].traction);
        sim.dispose();
        ctx.release(s);
    });
});

// ============================================================ every override combination compiles

describe("createForceAtlas2: every option combination compiles through the override matrix (spec 5.1, 11.3)", () => {
    it("16 law x compat combinations over weights / dim run one iteration each; 19 pipelines; every key covered", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const weighted = snapshotOf(WEIGHTED_KARATE);
        for (const compat of ["paper", "networkx"] as const) {
            for (const linlog of [false, true]) {
                for (const distributedAction of [false, true]) {
                    for (const strongGravity of [false, true]) {
                        const dim = distributedAction ? 3 : 2;
                        const weight = !strongGravity;
                        const sim = createForceAtlas2(ctx, {
                            compat,
                            linlog,
                            distributedAction,
                            strongGravity,
                            dim,
                            weight,
                            seed: 11,
                        });
                        const positions = nanPositions(weighted.nodeCount);
                        sim.load(weighted, positions);
                        await sim.step(1);
                        expect(sim.iterationsDone).toBe(1);
                        for (const v of positions) {
                            expect(Number.isFinite(v)).toBe(true);
                        }
                        if (dim === 2) {
                            for (let i = 0; i < weighted.nodeCount; i++) {
                                expect(positions[3 * i + 2]).toBe(0);
                            }
                        }
                        sim.dispose();
                    }
                }
            }
        }
        expect(ctx.pipelines.size).toBe(19);
        const keys = ctx.pipelines.keys();
        const cover = matrixCovers(keys, ctx.caps);
        expect(cover.missing, `keys missing from OVERRIDE_MATRIX: ${cover.missing.join(" ; ")}`).toEqual([]);
        expect(cover.ok).toBe(true);
        ctx.release(weighted);
    });
});
