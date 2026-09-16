/**
 * THE kernel registry (spec 3.5; contract 3.10): every WGSL module the package compiles, keyed by id, with the
 * binding table, the override declarations, the generated uniform / storage blocks and the feature needs from
 * which src/kernel/wgsl.ts emits the bind declarations and src/kernel/pipeline-cache.ts derives the explicit
 * bind-group layouts. A variant cannot exist without an entry here: the compile tests, the bind-group-budget
 * test and PipelineCache.warm() iterate this table. P1-T4 lands degree, reduce, fill, fa2-repulsion-exact (K3)
 * and fa2-speed-finalize (K4) together with every generated block of contract 3.10.2; P2-T2 lands
 * segmented-reduce; P3-T2 adds fa2-stats-finalize (K1), fa2-attraction (K2), fa2-integrate (K5) and
 * fa2-to-scene. This file is the only importer of src/wgsl/** (spec 3.2; test/layers.test.ts).
 */

import { STATE_HEADER_BYTES } from "./constants.js";
import { WebGpuGraphError } from "./errors.js";
import { UniformBlock } from "./kernel/struct-block.js";
import { type BindingDecl, type OverrideDecl, type WgslModuleSpec } from "./kernel/wgsl.js";
import { type CoreBinding } from "./memory/residency.js";
import { type Binding } from "./types/memory.js";
import { degreeWgsl } from "./wgsl/degree.wgsl.js";
import { fa2AttractionWgsl } from "./wgsl/fa2-attraction.wgsl.js";
import { fa2IntegrateWgsl } from "./wgsl/fa2-integrate.wgsl.js";
import { fa2RepulsionExactWgsl } from "./wgsl/fa2-repulsion-exact.wgsl.js";
import { fa2SpeedFinalizeWgsl } from "./wgsl/fa2-speed-finalize.wgsl.js";
import { fa2StatsFinalizeWgsl } from "./wgsl/fa2-stats-finalize.wgsl.js";
import { fa2ToSceneWgsl } from "./wgsl/fa2-to-scene.wgsl.js";
import { fillWgsl } from "./wgsl/fill.wgsl.js";
import { reduceWgsl } from "./wgsl/reduce.wgsl.js";
import { segmentedReduceWgsl } from "./wgsl/segmented-reduce.wgsl.js";

/** Every module id of P1-P3 (P4+ ids are appended, never renamed). */
export type KernelId =
    | "degree"
    | "reduce"
    | "fill"
    | "segmented-reduce"
    | "fa2-stats-finalize"
    | "fa2-attraction"
    | "fa2-repulsion-exact"
    | "fa2-speed-finalize"
    | "fa2-integrate"
    | "fa2-to-scene";

/** One registry entry: everything of a WgslModuleSpec except the per-variant overrides and snippets. */
export interface KernelEntry {
    readonly id: KernelId;
    readonly body: string;
    readonly entryPoint: string;
    readonly bindings: readonly BindingDecl[];
    readonly overrideDecls: readonly OverrideDecl[];
    readonly uniforms: readonly UniformBlock[];
    /** ["subgroups"] when the body calls a reduction helper (the twin axis), else []. */
    readonly needs: readonly "subgroups"[];
    /** The snippet marker names the body carries (segmented-reduce: ["VALUE"]). */
    readonly snippetSlots: readonly string[];
    /** The phase the entry landed in (documentation and the compile-matrix filter). */
    readonly phase: "P1" | "P2" | "P3";
}

// ---- the generated blocks (spec 5.3; contract 3.10.2): field order = byte order, offsets in the JSDoc

/** `RangeParams` (uniform, 32 B): rows `[start, end)` @0 / @4 of the dispatch, the bound arc window `[arcBase, arcEnd)` @8 / @12 (0 and arcCount when not windowed), `accumulate` @16 (1 combines into `out[i]` instead of overwriting: the P4 windowed loop), `n` @20 (the node count that bounds neighbour indices), `pad0` @24, `pad1` @28. */
export const RANGE_PARAMS: UniformBlock = UniformBlock.define("RangeParams", [
    ["start", "u32"],
    ["end", "u32"],
    ["arcBase", "u32"],
    ["arcEnd", "u32"],
    ["accumulate", "u32"],
    ["n", "u32"],
    ["pad0", "u32"],
    ["pad1", "u32"],
]);

/** `ReduceParams` (uniform, 16 B): `count` @0 elements of the level, `outOffset` @4 (the element index the level writes at), `level` @8, `pad0` @12. */
export const REDUCE_PARAMS: UniformBlock = UniformBlock.define("ReduceParams", [
    ["count", "u32"],
    ["outOffset", "u32"],
    ["level", "u32"],
    ["pad0", "u32"],
]);

/** `FillParams` (uniform, 16 B): `count` @0 words, `value` @4, `mode` @8 (0 = the constant `value`, 1 = iota `i + value`), `pad0` @12. */
export const FILL_PARAMS: UniformBlock = UniformBlock.define("FillParams", [
    ["count", "u32"],
    ["value", "u32"],
    ["mode", "u32"],
    ["pad0", "u32"],
]);

/** `Fa2Params` (uniform, 96 B; spec 7.3): the per-iteration ForceAtlas2 parameters -- `n` @0, `dim` @4, `flags` @8 (bit 0 = FA2_FLAG_FIRST), `tierStart` @12, `tierEnd` @16, `iterationIndex` @20, `seed` @24, `nearMax` @28, `scalingRatio` @32, `gravity` @36, `jitterTolerance` @40, `scale` @44, `center` @48 (xyz, w 0), `settleThreshold` @64, `extentFactor` @68, `gridMax` @72, `levels` @76, `pad` @80 (reserved for the P4 GridSpec). */
export const FA2_PARAMS: UniformBlock = UniformBlock.define("Fa2Params", [
    ["n", "u32"],
    ["dim", "u32"],
    ["flags", "u32"],
    ["tierStart", "u32"],
    ["tierEnd", "u32"],
    ["iterationIndex", "u32"],
    ["seed", "u32"],
    ["nearMax", "u32"],
    ["scalingRatio", "f32"],
    ["gravity", "f32"],
    ["jitterTolerance", "f32"],
    ["scale", "f32"],
    ["center", "vec4f"],
    ["settleThreshold", "f32"],
    ["extentFactor", "f32"],
    ["gridMax", "u32"],
    ["levels", "u32"],
    ["pad", "vec4f"],
]);

/** `Fa2State` (storage, padded to STATE_HEADER_BYTES = 256; spec 7.3): the device-resident controller state the finalize kernels write and the host reads back for stats -- `speed` @0, `speedEfficiency` @4, `swing` @8, `traction` @12, `centroid` @16, `rmsRadius` @32, `radius` @36, `meanDisplacement` @40, `iteration` @44, `min` @48, `max` @64, `gridMin` @80 (P4), `eps` @96 (P4), `settledCount` @100, `outsideGrid` @104 (P4), `maxCellOccupancy` @108 (P4), `reserved0` .. `reserved8` @112 .. @240. */
export const FA2_STATE: UniformBlock = UniformBlock.define(
    "Fa2State",
    [
        ["speed", "f32"],
        ["speedEfficiency", "f32"],
        ["swing", "f32"],
        ["traction", "f32"],
        ["centroid", "vec4f"],
        ["rmsRadius", "f32"],
        ["radius", "f32"],
        ["meanDisplacement", "f32"],
        ["iteration", "u32"],
        ["min", "vec4f"],
        ["max", "vec4f"],
        ["gridMin", "vec4f"],
        ["eps", "f32"],
        ["settledCount", "u32"],
        ["outsideGrid", "u32"],
        ["maxCellOccupancy", "u32"],
        ["reserved0", "vec4f"],
        ["reserved1", "vec4f"],
        ["reserved2", "vec4f"],
        ["reserved3", "vec4f"],
        ["reserved4", "vec4f"],
        ["reserved5", "vec4f"],
        ["reserved6", "vec4f"],
        ["reserved7", "vec4f"],
        ["reserved8", "vec4f"],
    ],
    { layout: "storage", padTo: STATE_HEADER_BYTES },
);

/** `Fa2Trace` (storage record, 32 B; spec 7.3): one per-iteration trace record -- `swing` @0, `traction` @4, `speed` @8, `speedEfficiency` @12 (written by K4), `meanDisplacement` @16, `settledCount` @20, `iteration` @24 (written by K1), `pad0` @28; the trace region is `array<Fa2Trace>` at byte offset STATE_HEADER_BYTES of the state buffer. */
export const FA2_TRACE: UniformBlock = UniformBlock.define(
    "Fa2Trace",
    [
        ["swing", "f32"],
        ["traction", "f32"],
        ["speed", "f32"],
        ["speedEfficiency", "f32"],
        ["meanDisplacement", "f32"],
        ["settledCount", "u32"],
        ["iteration", "u32"],
        ["pad0", "u32"],
    ],
    { layout: "storage" },
);

/** `Fa2Partial` (storage record, 64 B; spec 7.3): one per-workgroup partial -- `sum` @0 (xyz = sum of positions, w = sum of |p - centroid|^2), `min` @16, `max` @32 (w = max of |p - centroid|^2, the exact layoutRadius source), written by K5; `swingTraction` @48 (written by K3's epilogue); `dispFree` @56 (x = sum |dp| over free rows, y = the free count as an f32 <= 256, written by K5). */
export const FA2_PARTIAL: UniformBlock = UniformBlock.define(
    "Fa2Partial",
    [
        ["sum", "vec4f"],
        ["min", "vec4f"],
        ["max", "vec4f"],
        ["swingTraction", "vec2f"],
        ["dispFree", "vec2f"],
    ],
    { layout: "storage" },
);

// ---- the entries (contract 3.10.1; group 0 = graph, 1 = state, 2 = params, 3 = cold)

/**
 * One binding declaration of contract 3.9, so the tables below read like the rows of 3.10.1.
 * @param group - the bind group (0 graph, 1 state, 2 params, 3 cold)
 * @param binding - the slot inside the group
 * @param name - the WGSL variable name (also the KernelBindings key)
 * @param kind - "storage" (read_write), "storage-ro" (read) or "uniform"
 * @param wgslType - the element / struct type text
 * @returns the declaration
 */
function decl(
    group: 0 | 1 | 2 | 3,
    binding: number,
    name: string,
    kind: BindingDecl["kind"],
    wgslType: string,
): BindingDecl {
    return { group, binding, name, kind, wgslType };
}

/** The four group-0 graph slots of every row-walking kernel (spec 3.5): rowPtr, colIdx, weights | dummy, perm | dummy -- all read-only, so the dummies never alias a writable slot (3.10.1). */
const GRAPH_SLOTS: readonly BindingDecl[] = [
    decl(0, 0, "rowPtr", "storage-ro", "array<u32>"),
    decl(0, 1, "colIdx", "storage-ro", "array<u32>"),
    decl(0, 2, "weights", "storage-ro", "array<f32>"),
    decl(0, 3, "perm", "storage-ro", "array<u32>"),
];

/** `degree` (3.10.1): 5 storage bindings; only the standard USE_PERM / HAS_WEIGHTS overrides. */
const DEGREE: KernelEntry = {
    id: "degree",
    body: degreeWgsl,
    entryPoint: "degree",
    bindings: GRAPH_SLOTS.concat(decl(1, 0, "out", "storage", "array<u32>"), decl(2, 0, "P", "uniform", "RangeParams")),
    overrideDecls: [],
    uniforms: [RANGE_PARAMS],
    needs: [],
    snippetSlots: [],
    phase: "P1",
};

/** `reduce` (3.10.1): the multi-level reduction; OP 0 sum / 1 min / 2 max, DTYPE 0 f32 / 1 u32 / 2 vec4f, FINAL for the one-workgroup level; 2 storage bindings; calls the reduction helpers. */
const REDUCE: KernelEntry = {
    id: "reduce",
    body: reduceWgsl,
    entryPoint: "reduce",
    bindings: [
        decl(1, 0, "src", "storage-ro", "array<u32>"),
        decl(1, 1, "out", "storage", "array<u32>"),
        decl(2, 0, "P", "uniform", "ReduceParams"),
    ],
    overrideDecls: [
        { name: "OP", type: "u32", default: 0 },
        { name: "DTYPE", type: "u32", default: 0 },
        { name: "FINAL", type: "bool", default: false },
    ],
    uniforms: [REDUCE_PARAMS],
    needs: ["subgroups"],
    snippetSlots: [],
    phase: "P1",
};

/** `fill` (3.10.1): dst[i] = value (mode 0) or i + value (mode 1) over `count` words; 1 storage binding. */
const FILL: KernelEntry = {
    id: "fill",
    body: fillWgsl,
    entryPoint: "fill",
    bindings: [decl(1, 0, "dst", "storage", "array<u32>"), decl(2, 0, "P", "uniform", "FillParams")],
    overrideDecls: [],
    uniforms: [FILL_PARAMS],
    needs: [],
    snippetSlots: [],
    phase: "P1",
};

/** `segmented-reduce` (3.10.1): the per-row fold of the VALUE snippet over the CSR rows; OP 0 sum / 1 min / 2 max, TIER 0 (thread-per-row; the degreeOrder tiers land at P4) plus the standard USE_PERM / HAS_WEIGHTS; 5 storage bindings; one snippet slot. */
const SEGMENTED_REDUCE: KernelEntry = {
    id: "segmented-reduce",
    body: segmentedReduceWgsl,
    entryPoint: "segmented_reduce",
    bindings: GRAPH_SLOTS.concat(decl(1, 0, "out", "storage", "array<f32>"), decl(2, 0, "P", "uniform", "RangeParams")),
    overrideDecls: [
        { name: "OP", type: "u32", default: 0 },
        { name: "TIER", type: "u32", default: 0 },
    ],
    uniforms: [RANGE_PARAMS],
    needs: [],
    snippetSlots: ["VALUE"],
    phase: "P2",
};

/** `fa2-stats-finalize` (K1, 3.10.1): the one-workgroup fold of the previous integrate's partials into the state block and the K1 half of the trace record; 3 storage bindings; calls the reduction helpers. */
const FA2_STATS_FINALIZE: KernelEntry = {
    id: "fa2-stats-finalize",
    body: fa2StatsFinalizeWgsl,
    entryPoint: "stats_finalize",
    bindings: [
        decl(1, 0, "partials", "storage-ro", "array<Fa2Partial>"),
        decl(1, 1, "S", "storage", "Fa2State"),
        decl(1, 2, "T", "storage", "array<Fa2Trace>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ],
    overrideDecls: [],
    uniforms: [FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL],
    needs: ["subgroups"],
    snippetSlots: [],
    phase: "P3",
};

/** `fa2-attraction` (K2, 3.10.1): the thread-per-row attraction gather over the CSR rows (the first writer of `force` each iteration); LINLOG / DISTRIBUTED / TIER 0 plus the standard USE_PERM / HAS_WEIGHTS; 6 storage bindings. */
const FA2_ATTRACTION: KernelEntry = {
    id: "fa2-attraction",
    body: fa2AttractionWgsl,
    entryPoint: "attraction",
    bindings: GRAPH_SLOTS.concat(
        decl(1, 0, "pos", "storage-ro", "array<vec4f>"),
        decl(1, 1, "force", "storage", "array<f32>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ),
    overrideDecls: [
        { name: "LINLOG", type: "bool", default: false },
        { name: "DISTRIBUTED", type: "bool", default: false },
        { name: "TIER", type: "u32", default: 0 },
    ],
    uniforms: [FA2_PARAMS],
    needs: [],
    snippetSlots: [],
    phase: "P3",
};

/** `fa2-repulsion-exact` (K3, 3.10.1): the tiled all-pairs repulsion with the gravity and swing / traction epilogue; 6 storage bindings (`oldForce` read-only: it only calls load_old); calls the reduction helpers. */
const FA2_REPULSION_EXACT: KernelEntry = {
    id: "fa2-repulsion-exact",
    body: fa2RepulsionExactWgsl,
    entryPoint: "repulsion",
    bindings: [
        decl(1, 0, "pos", "storage-ro", "array<vec4f>"),
        decl(1, 1, "S", "storage", "Fa2State"),
        decl(1, 2, "force", "storage", "array<f32>"),
        decl(1, 3, "oldForce", "storage-ro", "array<f32>"),
        decl(1, 4, "fixedMask", "storage-ro", "array<u32>"),
        decl(1, 5, "partials", "storage", "array<Fa2Partial>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ],
    overrideDecls: [
        { name: "SWING_MODE", type: "u32", default: 0 },
        { name: "STRONG_GRAVITY", type: "bool", default: false },
        { name: "GRAVITY_CENTER", type: "u32", default: 0 },
    ],
    uniforms: [FA2_PARAMS, FA2_STATE, FA2_PARTIAL],
    needs: ["subgroups"],
    snippetSlots: [],
    phase: "P1",
};

/** `fa2-speed-finalize` (K4, 3.10.1): the one-workgroup partials fold and estimateFactor; 3 storage bindings; calls the reduction helpers. */
const FA2_SPEED_FINALIZE: KernelEntry = {
    id: "fa2-speed-finalize",
    body: fa2SpeedFinalizeWgsl,
    entryPoint: "speed_finalize",
    bindings: [
        decl(1, 0, "partials", "storage-ro", "array<Fa2Partial>"),
        decl(1, 1, "S", "storage", "Fa2State"),
        decl(1, 2, "T", "storage", "array<Fa2Trace>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ],
    overrideDecls: [{ name: "SWING_MODE", type: "u32", default: 0 }],
    uniforms: [FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL],
    needs: ["subgroups"],
    snippetSlots: [],
    phase: "P1",
};

/** `fa2-integrate` (K5, 3.10.1): the per-node speed factor and position update (no clamp, D25), `oldForce` stored in SWING_MODE 0, and the partials A / C of the next K1; 6 storage bindings; calls the reduction helpers. */
const FA2_INTEGRATE: KernelEntry = {
    id: "fa2-integrate",
    body: fa2IntegrateWgsl,
    entryPoint: "integrate",
    bindings: [
        decl(1, 0, "force", "storage-ro", "array<f32>"),
        decl(1, 1, "oldForce", "storage", "array<f32>"),
        decl(1, 2, "fixedMask", "storage-ro", "array<u32>"),
        decl(1, 3, "S", "storage", "Fa2State"),
        decl(1, 4, "pos", "storage", "array<vec4f>"),
        decl(1, 5, "partials", "storage", "array<Fa2Partial>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ],
    overrideDecls: [{ name: "SWING_MODE", type: "u32", default: 0 }],
    uniforms: [FA2_PARAMS, FA2_STATE, FA2_PARTIAL],
    needs: ["subgroups"],
    snippetSlots: [],
    phase: "P3",
};

/** `fa2-to-scene` (3.10.1): the per-batch unpack of the vec4f layout positions into the stride-3 scene array with `scale` / `center` applied (z = center.z in 2D); 2 storage bindings. */
const FA2_TO_SCENE: KernelEntry = {
    id: "fa2-to-scene",
    body: fa2ToSceneWgsl,
    entryPoint: "to_scene",
    bindings: [
        decl(1, 0, "pos", "storage-ro", "array<vec4f>"),
        decl(1, 1, "scene", "storage", "array<f32>"),
        decl(2, 0, "P", "uniform", "Fa2Params"),
    ],
    overrideDecls: [],
    uniforms: [FA2_PARAMS],
    needs: [],
    snippetSlots: [],
    phase: "P3",
};

/**
 * The entries by id, in dispatch order. PLAN DECISION: `KernelId` is declared in full (contract 3.10) while the
 * entries landed phase by phase, so the table is built as a Partial record and exported below through the
 * contract's `Readonly<Record<KernelId, KernelEntry>>` type by one assertion; the runtime membership check of
 * `entryOf` is the E_INVALID_ARGUMENT the contract documents for a JS caller's unknown id. P1-T4 landed the five
 * P1 entries, P2-T2 `"segmented-reduce"`, and P3-T2 `"fa2-stats-finalize"`, `"fa2-attraction"`, `"fa2-integrate"`
 * and `"fa2-to-scene"`, so every member of `KernelId` is present and the assertion is exact.
 */
const REGISTRY: Readonly<Partial<Record<KernelId, KernelEntry>>> = Object.freeze({
    degree: DEGREE,
    reduce: REDUCE,
    fill: FILL,
    "segmented-reduce": SEGMENTED_REDUCE,
    "fa2-stats-finalize": FA2_STATS_FINALIZE,
    "fa2-attraction": FA2_ATTRACTION,
    "fa2-repulsion-exact": FA2_REPULSION_EXACT,
    "fa2-speed-finalize": FA2_SPEED_FINALIZE,
    "fa2-integrate": FA2_INTEGRATE,
    "fa2-to-scene": FA2_TO_SCENE,
});

/** THE registry (spec 3.5): every entry, keyed by id. */
export const KERNELS: Readonly<Record<KernelId, KernelEntry>> = REGISTRY as Readonly<Record<KernelId, KernelEntry>>;

/** The bodies installed by setKernelBodyOverride (the sabotage seam of spec 11.9 item 1), by id. */
const bodyOverrides = new Map<KernelId, string>();

/**
 * The entry of an id, or E_INVALID_ARGUMENT for an id that is not (yet) registered.
 * @param id - the module id
 * @returns the entry
 */
function entryOf(id: KernelId): KernelEntry {
    const entry = REGISTRY[id];
    if (entry === undefined) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `unknown kernel id "${id}"`, {
            argument: "id",
            value: id,
            expected: Object.keys(REGISTRY),
        });
    }
    return entry;
}

/**
 * A WgslModuleSpec for a variant: the entry plus the overrides / snippets given; unknown override names are
 * rejected at compose time. A body override installed by setKernelBodyOverride is used instead of the entry's
 * body.
 * @param id - the module id
 * @param overrides - the override values of this variant (the five standard names and the entry's overrideDecls)
 * @param snippets - the texts substituted at the body's `//@@NAME@@` markers
 * @returns the spec the PipelineCache compiles
 */
export function kernelSpec(
    id: KernelId,
    overrides?: Readonly<Record<string, number | boolean>>,
    snippets?: Readonly<Record<string, string>>,
): WgslModuleSpec {
    const entry = entryOf(id);
    return {
        id: entry.id,
        body: bodyOverrides.get(id) ?? entry.body,
        bindings: entry.bindings,
        overrideDecls: entry.overrideDecls,
        overrides: overrides ?? {},
        needs: entry.needs,
        uniforms: entry.uniforms,
        snippets,
    };
}

/**
 * The sabotage seam (spec 11.9 item 1): replaces an entry's body for specs created afterwards (null restores).
 * Tests use a FRESH context per mutation because the pipeline key does not include the body.
 * @internal
 * @param id - the module id
 * @param body - the replacement body, or null to restore the normative one
 */
export function setKernelBodyOverride(id: KernelId, body: string | null): void {
    entryOf(id);
    if (body === null) {
        bodyOverrides.delete(id);
    } else {
        bodyOverrides.set(id, body);
    }
}

/**
 * The group-0 graph bindings of a core with the dummy rules applied (spec 3.5, 4.1): colIdx <- rowPtr when null,
 * weights <- colIdx ?? rowPtr when null, perm <- rowPtr when null. `weights` is the binding to use in the weights
 * slot: omitted -> core.weights (degree, segmented-reduce); a layout passes its RESOLVED weights
 * (ModelResources.weights, 3.13), null meaning "attract with 1.0" even on a weighted snapshot.
 * @param core - the resident core arrays of a snapshot
 * @param perm - the degreeOrder row permutation, or null (rowPtr is bound as the dummy)
 * @param weights - the weights binding to use; undefined takes the core's, null binds the colIdx dummy
 * @returns the four bindings in slot order
 */
export function graphBindings(
    core: CoreBinding,
    perm: Binding | null,
    weights?: Binding | null,
): Readonly<Record<"rowPtr" | "colIdx" | "weights" | "perm", Binding>> {
    const colIdx = core.colIdx ?? core.rowPtr;
    const resolved = weights === undefined ? core.weights : weights;
    return {
        rowPtr: core.rowPtr,
        colIdx,
        weights: resolved ?? colIdx,
        perm: perm ?? core.rowPtr,
    };
}

/**
 * The override values graphBindings implies: USE_PERM = perm !== null, HAS_WEIGHTS = (weights === undefined ?
 * core.weights : weights) !== null.
 * @param core - the resident core arrays of a snapshot
 * @param perm - the row permutation binding, or null
 * @param weights - the weights binding to use; undefined takes the core's, null means "unweighted"
 * @returns the two standard override values of the variant
 */
export function graphOverrides(
    core: CoreBinding,
    perm: Binding | null,
    weights?: Binding | null,
): Readonly<{ USE_PERM: boolean; HAS_WEIGHTS: boolean }> {
    const resolved = weights === undefined ? core.weights : weights;
    return { USE_PERM: perm !== null, HAS_WEIGHTS: resolved !== null };
}
