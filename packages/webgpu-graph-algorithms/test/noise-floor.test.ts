/**
 * The noise-floor file (spec 11.9 item 3; contract 5.6): benchmarks/results/noise-floor.json holds, per kernel and
 * fixture of the noise set, the spread that pure summation-order noise produces -- the same kernel on its subgroup
 * twin, on one adapter vs another (NVIDIA Dawn, lavapipe Dawn, SwiftShader Chromium), and against the f64 reference
 * on f32-rounded inputs -- and the tolerances the parity tests take from it, each DERIVED as
 * `min(spec cap, 10 x floor)` and never typed. This file measures the rows from the committed fixtures
 * (test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json), writes this adapter's FA2 paper-leg outputs and the
 * rows / adapters / tolerances under GRAPHTY_NOISE_FLOOR_WRITE=1, and validates the committed file otherwise.
 *
 * Members: degree / random1k (u32, bitwise; written by test/algorithms/degree.test.ts), reduce / random1k (the f32 sum;
 * written by test/primitives/reduce.test.ts; checked against the analytic bound 2 x 1000 x 2^-24 of spec 11.3, no
 * derived tolerance), fa2-repulsion-exact / karate (the paper-leg force, stride 3, the 11.4 floored per-node metric),
 * fa2-speed-finalize / karate, networkx-karate, prev-karate (the trace [swing, traction, speed, speedEfficiency] of the
 * skeleton's legs). Reserved adapter classes: "oracle-f64" (the f64 reference written by test/layouts/skeleton.test.ts)
 * and "<class>-no-subgroups" (the workgroup twin). One writer per row id: the rows degree.cross, reduce.sum.twin and
 * reduce.sum.cross are recorded by P1-T5's tests (this file requires them and never records them); the six
 * fa2-skeleton.* rows are recorded only here, each the maximum over the whole committed set; a browser run (P1-T7) may
 * add per-class rows under its own ids (fa2-skeleton.force.oracle-f64.<class>), which the validation accepts and no
 * tolerance is derived from.
 *
 * P3 members (P3-T5, contract 5.6 "extended at P3-T5 with K1-K5"; every one on the UNSCALED random1k, P3-T5 PLAN
 * DECISION 7): fa2-attraction / random1k-K2 (K2's attraction, floored stride-3), fa2-repulsion-exact / random1k-K3
 * (K3's force; its oracle-f64 row is the basis of fa2-force-parity and fa2-force-sum, its twin row of fa2-twins.force)
 * and random1k-K3-epilogue ([swing, traction]), fa2-speed-finalize / random1k-K4 ([speed, speedEfficiency, swing,
 * traction]), fa2-integrate / random1k-K5 (positions, stride 3) and random1k-K5-partials (the 13 folded partial
 * values), fa2-to-scene / random1k-toScene (scene positions), fa2-stats-finalize / random1k-K1 (the fold of
 * iteration 2), fa2-speed-finalize / random1k-trace10 (60 trace values of the free-running NETWORKX-mode trajectory
 * -- the mode the free-running legs of fa2-trace-parity.test.ts are asserted in, G3-F3; its reference is the f32
 * oracle, reserved class "oracle-f32", P3-T5 PLAN DECISION 6: the row carries comparison "oracle-f64" with b
 * "oracle-f32", a known literal mismatch against the closed union of contract 5.6 -- open item for the owner, the b
 * field is authoritative), random1k-trace50 (300 networkx values vs the f64 oracle) and karate-trace50 (the same on
 * karate, whose 50-iteration networkx divergence comes closest to the 5e-2 cap, so the recorded factor of
 * fa2-trace-parity.f64 is honest), fa2-speed-finalize / random1k-resync50 and karate-resync50 (PLAN DECISION 17: the PAPER-mode 50-iteration trajectory as RESYNC_FIELDS
 * values per iteration, compared with the f32 / f64 oracles RE-SYNCHRONISED to the adapter's own iteration-start
 * state before every iteration through the "resync-trace" metric resyncValuesError; the oracle fixtures of such a
 * member are per adapter class, "oracle-f32/<class>" and "oracle-f64/<class>", because the oracle's values follow
 * that adapter's states; its cross-adapter and twin pairs are free-running paper-mode traces, chaotic beyond any
 * derivable tolerance (G3-F3), and are printed but neither checked nor recorded; a twin class's fixture is compared
 * with ITS own oracle fixtures and feeds the same rows), fa2-speed-finalize / random1k-states10 (K4's six-value
 * record of ONE iteration under load() semantics from each of ten f64-oracle trajectory states, oracleStates(): the
 * multi-state twin comparison of fa2-twins.test.ts; it feeds the K4 state rows and the scalar twin row so those
 * floors cover ten geometries, not one), fa2-integrate / random1k-metrics100 (the layoutMetrics record
 * after 100 iterations, the "distributional" metric of test/helpers/fa2-parity.ts: the non-histogram metrics relative
 * under a 0.05 floor, the nearest-neighbour histogram as one total-variation distance). Writers: fa2-inspect.test.ts
 * (adapter and oracle-f64 of the eight stages), fa2-twins.test.ts (the -no-subgroups twins and their resync oracles),
 * fa2-trace-parity.test.ts (the traces and their references), fa2-distributional.test.ts (the metrics). The scalar
 * twin rows all feed one row id, fa2-twins.trace.twin (the maximum over the members), the basis of fa2-twins.trace;
 * the two per-node vector twin rows (K5 positions, toScene scene) feed fa2-twins.positions.twin (the maximum over the
 * two), the basis of fa2-twins.positions. Every pair is MEASURED (bump) before it is CHECKED (P3-T5): a comparison
 * above its cap still records its row, so a floor above the spec cap surfaces as the validation's "value >= its
 * floor" finding (spec 10.4) rather than as an unrecorded basis row that would abort every derivation.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { type F32 } from "@graphty/graph-format";

import { PARTIAL_BYTES, STATE_HEADER_BYTES, TRACE_RECORD_BYTES } from "../src/constants.js";
import { type GpuContext } from "../src/context.js";
import { BufferUsage } from "../src/device/webgpu-constants.js";
import { type UniformValues } from "../src/kernel/struct-block.js";
import { FA2_PARAMS, FA2_STATE, FA2_TRACE } from "../src/kernels.js";
import {
    RepulsionExact,
    type RepulsionExactOverrides,
    type RepulsionExactResources,
} from "../src/layouts/repulsion-exact.js";
import { bindingOf, readF32, scratchBuffer, uploadBuffer } from "./helpers/device.js";
import {
    distributionalValuesError,
    NOISE_FIXTURES,
    noiseMetricsKeys,
    ORACLE_F32_CLASS,
    P3_TOLERANCE_CAPS,
    resyncOracleClass,
    resyncValuesError,
} from "./helpers/fa2-parity.js";
import { KARATE_EDGES, snapshotOf } from "./helpers/graphs.js";
import { expectBitwiseEqual, flooredRelError, maxRelError } from "./helpers/matchers.js";
import {
    adapterClass,
    noiseFloorFor,
    type NoiseRow,
    readNoiseFixtures,
    recordNoiseRow,
    writeNoiseFixture,
} from "./helpers/noise-floor.js";
import { acquire, requireGpu } from "./setup/gpu.js";

// ---------------------------------------------------------------- the file, the set and the caps

const NOISE_FLOOR_FILE = fileURLToPath(new URL("../benchmarks/results/noise-floor.json", import.meta.url));
const WRITE = process.env.GRAPHTY_NOISE_FLOOR_WRITE === "1";
/** One f32 ulp: the smallest floor a tolerance is derived from (a bitwise agreement so far is not evidence the next adapter agrees bitwise). */
const MIN_FLOOR = 2 ** -24;
/** Distinct adapter classes every member must have committed: NVIDIA Dawn + lavapipe Dawn at G1 (SwiftShader Chromium joins through P1-T7). */
const MIN_ADAPTER_CLASSES = 2;
const ORACLE_CLASS = "oracle-f64";
const TWIN_SUFFIX = "-no-subgroups";
/** Spec 11.4: the floored denominator of the per-node force error; spec 11.5: the absolute floor of the relative comparisons. */
const FLOOR_FRACTION = 1e-3;
const TRACE_ABS_FLOOR = 1e-6;
/** The reduce noise fixture holds f32 prefix sums of up to 1,000 values (spec 11.3: each within count x 2^-24 of the f64 sum). */
const REDUCE_COUNT = 1000;
/** Two adapters' f32 sums are each within count x 2^-24 of the exact sum, so within 2 x count x 2^-24 of each other (the bound test/primitives/reduce.test.ts applies). */
const REDUCE_CROSS_ADAPTER_BOUND = 2 * REDUCE_COUNT * MIN_FLOOR;
/** The `a` / `b` value of a provisional Step 1 row (samples 0): resolvable by noiseFloorFor, rejected by the validation, dropped before deriving. */
const SEED_CLASS = "seed";
/** Rows recorded by P1-T5's tests (one writer per row id): required to be present, never recorded here, no tolerance derived from them. */
const REQUIRED_ROWS: readonly string[] = ["degree.cross", "reduce.sum.twin", "reduce.sum.cross"];

type Metric = "bitwise" | "elementwise" | "floored-stride3" | "distributional" | "resync-trace";
type Comparison = "cross-adapter" | "twin" | "oracle-f64";
/** What a comparison is held to: a tolerance id of noise-floor.json, an analytic bound (a number), or null (bitwise). */
type Limit = string | number | null;

interface NoiseMember {
    readonly kernel: string;
    readonly fixture: string;
    readonly dtype: "u32" | "f32";
    readonly metric: Metric;
    /** The row id each comparison contributes to (null: not recorded for this member). */
    readonly rows: Readonly<Record<Comparison, string | null>>;
    /** What each comparison is checked against: a tolerance id, an analytic bound, or null (bitwise). */
    readonly tolerances: Readonly<Record<Comparison, Limit>>;
    /** Who writes the adapter fixtures of this member under GRAPHTY_NOISE_FLOOR_WRITE=1. */
    readonly writer: string;
    /** P3: the row id of the comparison against the f32 reference (class "oracle-f32"), null when the member has none. */
    readonly oracleF32Row?: string | null | undefined;
    /** P3: the tolerance id that comparison is checked against. */
    readonly oracleF32Tolerance?: string | null | undefined;
    /**
     * P3-T5 PLAN DECISION 17: a re-synchronised member -- the oracle fixtures are per adapter class (resyncOracleClass),
     * every adapter AND twin fixture is compared with its own two oracle fixtures (the rows["oracle-f64"] and
     * oracleF32Row rows), and the cross-adapter / twin pairs (free-running paper-mode traces) are printed only.
     */
    readonly resync?: boolean | undefined;
}

const NO_ROWS: Readonly<Record<Comparison, string | null>> = { "cross-adapter": null, twin: null, "oracle-f64": null };
const FA2_TRACE_ROWS: Readonly<Record<Comparison, string | null>> = {
    "cross-adapter": "fa2-skeleton.trace.cross",
    twin: "fa2-skeleton.trace.twin",
    "oracle-f64": "fa2-skeleton.trace.oracle-f64",
};
const FA2_TRACE_TOLERANCES: Readonly<Record<Comparison, Limit>> = {
    "cross-adapter": "fa2-skeleton.trace.cross",
    twin: "fa2-skeleton.trace.twin",
    "oracle-f64": "fa2-skeleton.trace",
};

// ---------------------------------------------------------------- the P3 members (P3-T5)

const P3_STAGE_WRITERS =
    "test/layouts/fa2-inspect.test.ts (adapter + oracle-f64) and test/layouts/fa2-twins.test.ts (-no-subgroups)";

/** The three rows / tolerances of an inspect stage: <stem>.cross, the given twin id, <stem>.oracle-f64 / <stem>. */
function stageRows(stem: string, twin: string | null): Readonly<Record<Comparison, string | null>> {
    return { "cross-adapter": `${stem}.cross`, twin, "oracle-f64": `${stem}.oracle-f64` };
}

function stageTolerances(stem: string, twin: string | null): Readonly<Record<Comparison, string | null>> {
    return { "cross-adapter": `${stem}.cross`, twin, "oracle-f64": stem };
}

function p3Member(
    name: keyof typeof NOISE_FIXTURES,
    metric: Metric,
    rows: Readonly<Record<Comparison, string | null>>,
    tolerances: Readonly<Record<Comparison, string | null>>,
    writer: string,
    f32?: { readonly row: string; readonly tolerance: string },
    resync = false,
): NoiseMember {
    const { kernel, fixture } = NOISE_FIXTURES[name];
    return {
        kernel,
        fixture,
        dtype: "f32",
        metric,
        rows,
        tolerances,
        writer,
        oracleF32Row: f32 === undefined ? null : f32.row,
        oracleF32Tolerance: f32 === undefined ? null : f32.tolerance,
        resync,
    };
}

/** The two re-synchronised members share their rows and tolerances (the floor is the maximum over both graphs). */
const RESYNC_ROWS: Readonly<Record<Comparison, string | null>> = {
    "cross-adapter": null,
    twin: null,
    "oracle-f64": "fa2-trace-parity.resync.oracle-f64",
};
const RESYNC_TOLERANCES: Readonly<Record<Comparison, string | null>> = {
    "cross-adapter": null,
    twin: null,
    "oracle-f64": "fa2-trace-parity.resync.f64",
};
const RESYNC_F32 = { row: "fa2-trace-parity.resync.oracle-f32", tolerance: "fa2-trace-parity.resync.f32" };
const RESYNC_WRITERS =
    "test/layouts/fa2-trace-parity.test.ts (adapter + its oracle-f32 / oracle-f64) and test/layouts/fa2-twins.test.ts (-no-subgroups + its oracles)";

/** The P3 noise set (contract 5.6; P3-T5 PLAN DECISIONS 6, 7, 11). */
const P3_NOISE_SET: readonly NoiseMember[] = [
    p3Member(
        "attraction",
        "floored-stride3",
        stageRows("fa2-inspect.attraction", null),
        stageTolerances("fa2-inspect.attraction", null),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "force",
        "floored-stride3",
        stageRows("fa2-force-parity", "fa2-twins.force.twin"),
        stageTolerances("fa2-force-parity", "fa2-twins.force"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "epilogue",
        "elementwise",
        stageRows("fa2-inspect.epilogue", "fa2-twins.trace.twin"),
        stageTolerances("fa2-inspect.epilogue", "fa2-twins.trace"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "state",
        "elementwise",
        stageRows("fa2-inspect.state", "fa2-twins.trace.twin"),
        stageTolerances("fa2-inspect.state", "fa2-twins.trace"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "positions",
        "floored-stride3",
        stageRows("fa2-inspect.positions", "fa2-twins.positions.twin"),
        stageTolerances("fa2-inspect.positions", "fa2-twins.positions"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "partials",
        "elementwise",
        stageRows("fa2-inspect.partials", "fa2-twins.trace.twin"),
        stageTolerances("fa2-inspect.partials", "fa2-twins.trace"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "scene",
        "floored-stride3",
        stageRows("fa2-inspect.scene", "fa2-twins.positions.twin"),
        stageTolerances("fa2-inspect.scene", "fa2-twins.positions"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "k1",
        "elementwise",
        stageRows("fa2-inspect.k1", "fa2-twins.trace.twin"),
        stageTolerances("fa2-inspect.k1", "fa2-twins.trace"),
        P3_STAGE_WRITERS,
    ),
    p3Member(
        "trace10",
        "elementwise",
        { "cross-adapter": "fa2-trace-parity.cross10", twin: "fa2-twins.trace.twin", "oracle-f64": null },
        { "cross-adapter": "fa2-trace-parity.cross10", twin: "fa2-twins.trace", "oracle-f64": null },
        "test/layouts/fa2-trace-parity.test.ts (adapter + oracle-f32, networkx mode) and test/layouts/fa2-twins.test.ts (-no-subgroups)",
        { row: "fa2-trace-parity.oracle-f32", tolerance: "fa2-trace-parity.f32" },
    ),
    p3Member(
        "trace50",
        "elementwise",
        { "cross-adapter": "fa2-trace-parity.cross50", twin: null, "oracle-f64": "fa2-trace-parity.oracle-f64" },
        { "cross-adapter": "fa2-trace-parity.cross50", twin: null, "oracle-f64": "fa2-trace-parity.f64" },
        "test/layouts/fa2-trace-parity.test.ts (adapter + oracle-f64, networkx mode)",
    ),
    p3Member(
        "trace50Karate",
        "elementwise",
        { "cross-adapter": "fa2-trace-parity.cross50", twin: null, "oracle-f64": "fa2-trace-parity.oracle-f64" },
        { "cross-adapter": "fa2-trace-parity.cross50", twin: null, "oracle-f64": "fa2-trace-parity.f64" },
        "test/layouts/fa2-trace-parity.test.ts (adapter + oracle-f64, networkx mode, karate)",
    ),
    p3Member("resync50", "resync-trace", RESYNC_ROWS, RESYNC_TOLERANCES, RESYNC_WRITERS, RESYNC_F32, true),
    p3Member("resyncKarate50", "resync-trace", RESYNC_ROWS, RESYNC_TOLERANCES, RESYNC_WRITERS, RESYNC_F32, true),
    // PLAN DECISION 17: K4's record of one iteration from each of ten oracle-trajectory states (the multi-state twin
    // comparison of fa2-twins.test.ts) widens the K4 state rows and the scalar twin row to ten geometries
    p3Member(
        "states10",
        "elementwise",
        stageRows("fa2-inspect.state", "fa2-twins.trace.twin"),
        stageTolerances("fa2-inspect.state", "fa2-twins.trace"),
        "test/layouts/fa2-twins.test.ts (both twins + oracle-f64)",
    ),
    p3Member(
        "metrics100",
        "distributional",
        { "cross-adapter": "fa2-distributional.cross", twin: null, "oracle-f64": "fa2-distributional.oracle-f64" },
        { "cross-adapter": "fa2-distributional.cross", twin: null, "oracle-f64": "fa2-distributional" },
        "test/layouts/fa2-distributional.test.ts (adapter + oracle-f64)",
    ),
];

/** The P1 noise set (contract 5.6, G1: degree, reduce and the FA2 skeleton from every adapter) followed by the P3 set. */
const NOISE_SET: readonly NoiseMember[] = [
    {
        kernel: "degree",
        fixture: "random1k",
        dtype: "u32",
        metric: "bitwise",
        rows: NO_ROWS,
        tolerances: NO_ROWS,
        writer: "test/algorithms/degree.test.ts (fixtures and the degree.cross row)",
    },
    {
        kernel: "reduce",
        fixture: "random1k",
        dtype: "f32",
        metric: "elementwise",
        rows: NO_ROWS,
        tolerances: { "cross-adapter": REDUCE_CROSS_ADAPTER_BOUND, twin: null, "oracle-f64": null },
        writer: "test/primitives/reduce.test.ts (fixtures and the reduce.sum.twin / reduce.sum.cross rows)",
    },
    {
        kernel: "fa2-repulsion-exact",
        fixture: "karate",
        dtype: "f32",
        metric: "floored-stride3",
        rows: {
            "cross-adapter": "fa2-skeleton.force.cross",
            twin: "fa2-skeleton.force.twin",
            "oracle-f64": "fa2-skeleton.force.oracle-f64",
        },
        tolerances: {
            "cross-adapter": "fa2-skeleton.force.cross",
            twin: "fa2-skeleton.force.twin",
            "oracle-f64": "fa2-skeleton.force",
        },
        writer: "test/layouts/skeleton.test.ts and this file",
    },
    {
        kernel: "fa2-speed-finalize",
        fixture: "karate",
        dtype: "f32",
        metric: "elementwise",
        rows: FA2_TRACE_ROWS,
        tolerances: FA2_TRACE_TOLERANCES,
        writer: "test/layouts/skeleton.test.ts and this file",
    },
    {
        kernel: "fa2-speed-finalize",
        fixture: "networkx-karate",
        dtype: "f32",
        metric: "elementwise",
        rows: FA2_TRACE_ROWS,
        tolerances: FA2_TRACE_TOLERANCES,
        writer: "test/layouts/skeleton.test.ts",
    },
    {
        kernel: "fa2-speed-finalize",
        fixture: "prev-karate",
        dtype: "f32",
        metric: "elementwise",
        rows: FA2_TRACE_ROWS,
        tolerances: FA2_TRACE_TOLERANCES,
        writer: "test/layouts/skeleton.test.ts",
    },
    ...P3_NOISE_SET,
];

/** Every tolerance the file carries: the spec cap it is derived under and the row it is derived from (the P1 entries, then the P3 caps of test/helpers/fa2-parity.ts, one table, never retyped). */
const TOLERANCE_CAPS: Readonly<Record<string, { readonly cap: number; readonly basis: string }>> = {
    "fa2-skeleton.force": { cap: 1e-5, basis: "fa2-skeleton.force.oracle-f64" },
    "fa2-skeleton.trace": { cap: 1e-5, basis: "fa2-skeleton.trace.oracle-f64" },
    "fa2-skeleton.force.twin": { cap: 1e-6, basis: "fa2-skeleton.force.twin" },
    "fa2-skeleton.trace.twin": { cap: 1e-6, basis: "fa2-skeleton.trace.twin" },
    "fa2-skeleton.force.cross": { cap: 1e-5, basis: "fa2-skeleton.force.cross" },
    "fa2-skeleton.trace.cross": { cap: 1e-5, basis: "fa2-skeleton.trace.cross" },
    ...P3_TOLERANCE_CAPS,
};

interface NoiseAdapter {
    readonly class: string;
    readonly vendor: string;
    readonly architecture: string;
    readonly description: string;
    readonly runtime: string;
}

interface NoiseTolerance {
    readonly value: number;
    readonly basis: string;
    readonly factor: number;
}

interface NoiseFloorDoc {
    recordedAt: string;
    adapters: NoiseAdapter[];
    rows: NoiseRow[];
    tolerances: Record<string, NoiseTolerance>;
}

function readDoc(): NoiseFloorDoc {
    if (!existsSync(NOISE_FLOOR_FILE)) {
        throw new Error(`${NOISE_FLOOR_FILE} does not exist (Step 1 of P1-T6 seeds it)`);
    }
    return JSON.parse(readFileSync(NOISE_FLOOR_FILE, "utf8")) as NoiseFloorDoc;
}

function writeDoc(doc: NoiseFloorDoc): void {
    writeFileSync(NOISE_FLOOR_FILE, `${JSON.stringify(doc, null, 4)}\n`);
}

function isTwinClass(cls: string): boolean {
    return cls.endsWith(TWIN_SUFFIX);
}

/** The reserved classes start with "oracle-" (oracle-f64, oracle-f32 and the per-adapter "oracle-<p>/<class>" of the resync members). */
function isOracleClass(cls: string): boolean {
    return cls.startsWith("oracle-");
}

function isAdapterClass(cls: string): boolean {
    return !isOracleClass(cls) && !isTwinClass(cls);
}

/** An adapters[] entry for a class seen only through its fixtures: <vendor>-<architecture>-<runtime> split back, description unknown. */
function adapterFromClass(cls: string): NoiseAdapter {
    const parts = cls.split("-");
    return {
        class: cls,
        vendor: parts[0],
        architecture: parts.slice(1, -1).join("-"),
        description: "",
        runtime: parts[parts.length - 1],
    };
}

function capOf(id: string): number {
    const spec = TOLERANCE_CAPS[id];
    if (spec === undefined) {
        throw new Error(`${id}: no cap in TOLERANCE_CAPS`);
    }
    return spec.cap;
}

/** In a write run every comparison is held to the spec cap (a floor above the cap is a finding); otherwise to the committed, derived tolerance. */
function limitOf(id: string): number {
    return WRITE ? capOf(id) : noiseFloorFor(id).value;
}

function floorOf(rows: readonly NoiseRow[], basis: string): number | null {
    const row = rows.find((r) => r.id === basis);
    return row === undefined ? null : row.maxRelError;
}

// ---------------------------------------------------------------- pair errors and the measured rows

interface PairError {
    readonly rel: number;
    readonly abs: number;
    readonly mismatches: number;
}

function pairError(metric: Metric, a: ArrayLike<number>, b: ArrayLike<number>): PairError {
    expect(a.length, "the two fixtures hold the same number of values").toBe(b.length);
    let abs = 0;
    let mismatches = 0;
    for (let i = 0; i < a.length; i++) {
        const d = Math.abs(a[i] - b[i]);
        abs = Math.max(abs, d);
        if (d !== 0) {
            mismatches++;
        }
    }
    switch (metric) {
        case "bitwise":
            return { rel: mismatches === 0 ? 0 : Number.POSITIVE_INFINITY, abs, mismatches };
        case "elementwise":
            return { rel: maxRelError(a, b, TRACE_ABS_FLOOR), abs, mismatches };
        case "floored-stride3":
            return {
                rel: Math.max(flooredRelError(a, b, FLOOR_FRACTION).max, flooredRelError(b, a, FLOOR_FRACTION).max),
                abs,
                mismatches,
            };
        case "distributional":
            // the ONE implementation of the metrics100 comparison (test/helpers/fa2-parity.ts): the values are in
            // the sorted key order of metricsValues(), reconstructed from the member's own graph
            return { rel: distributionalValuesError(noiseMetricsKeys(), a, b), abs, mismatches };
        case "resync-trace":
            // the ONE implementation of the re-synchronised comparison (test/helpers/fa2-parity.ts): b is the oracle
            return { rel: resyncValuesError(a, b), abs, mismatches };
        default:
            throw new Error("unknown metric");
    }
}

/** The rows this run measured, by id: the maximum over every pair / member that contributes to the id. */
const measured = new Map<string, NoiseRow>();

function bump(
    id: string | null,
    member: NoiseMember,
    comparison: Comparison,
    a: string,
    b: string,
    err: PairError,
    samples: number,
): void {
    if (id === null || !Number.isFinite(err.rel)) {
        return;
    }
    const existing = measured.get(id);
    if (existing !== undefined && existing.maxRelError >= err.rel) {
        return;
    }
    measured.set(id, {
        id,
        kernel: member.kernel,
        fixture: member.fixture,
        comparison,
        a,
        b,
        maxRelError: err.rel,
        maxAbsError: err.abs,
        samples,
    });
}

/**
 * The verdict of one comparison: null when it is within its limit, else the failure message. P3-T5: a verdict, not a
 * throwing expect, so a member test can MEASURE every pair (bump) before it fails on any of them -- with an immediate
 * throw the pairs after the first failing one would never be recorded and the derivation of every tolerance would
 * abort on the missing basis row instead of surfacing the finding through the validation.
 * @param member - the member
 * @param comparison - which comparison
 * @param err - the measured error
 * @param label - the pair label
 * @returns null or the failure message
 */
function check(member: NoiseMember, comparison: Comparison, err: PairError, label: string): string | null {
    const tolerance = member.tolerances[comparison];
    if (tolerance === null) {
        return err.mismatches === 0
            ? null
            : `${label}: ${comparison} u32 outputs must agree bitwise (${err.mismatches} mismatches)`;
    }
    if (typeof tolerance === "number") {
        return err.rel <= tolerance
            ? null
            : `${label}: ${comparison} error ${err.rel.toExponential(3)} vs the analytic bound ${tolerance.toExponential(3)}`;
    }
    const limit = limitOf(tolerance);
    return err.rel <= limit
        ? null
        : `${label}: ${comparison} error ${err.rel.toExponential(3)} vs ${WRITE ? "the spec cap of" : "the derived tolerance"} ${tolerance} (${limit.toExponential(3)})`;
}

/**
 * Appends a non-null verdict to the member's failure list.
 * @param failures - the list
 * @param verdict - the verdict of check()
 */
function collect(failures: string[], verdict: string | null): void {
    if (verdict !== null) {
        failures.push(verdict);
    }
}

// ---------------------------------------------------------------- this adapter's paper-leg output (the driver of test/layouts/skeleton.test.ts, paper leg only)

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
const PAPER: RepulsionExactOverrides = { SWING_MODE: 0, STRONG_GRAVITY: false, GRAVITY_CENTER: 0 };
const PAPER_JITTER_TOLERANCE = 1;

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

const snapshot = snapshotOf(KARATE_EDGES);
const n = snapshot.nodeCount;
const positions = skeletonPositions(snapshot.outDegree());
const stats = cpuStats(positions);

interface PaperOutput {
    readonly force: F32;
    readonly stateWords: Uint32Array<ArrayBuffer>;
    /** [swing, traction, speed, speedEfficiency] of the trace slot. */
    readonly trace: readonly [number, number, number, number];
}

function scalar(values: UniformValues, field: string): number {
    const value = values[field];
    if (typeof value !== "number") {
        throw new Error(`${field}: expected a scalar field`);
    }
    return value;
}

async function runPaperLeg(ctx: GpuContext): Promise<PaperOutput> {
    const stage = await RepulsionExact.create(ctx.pipelines, ctx.caps, PAPER);
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
            jitterTolerance: PAPER_JITTER_TOLERANCE,
            scale: 1,
            center: [0, 0, 0, 0],
            settleThreshold: 0.001,
            extentFactor: 6,
        },
        0,
    );
    const pos = uploadBuffer(ctx, positions, "noise/pos");
    const force = scratchBuffer(ctx, 12 * n, "noise/force");
    const old = scratchBuffer(ctx, 12 * n, "noise/oldForce");
    const fixed = scratchBuffer(ctx, 4 * Math.ceil(n / 32), "noise/fixed");
    const partials = scratchBuffer(ctx, PARTIAL_BYTES * groups, "noise/partials");
    const state = uploadBuffer(ctx, stateBytes, "noise/state");
    const params = uploadBuffer(ctx, paramsBytes, "noise/params", BufferUsage.UNIFORM);
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
        const encoder = ctx.device.createCommandEncoder({ label: "noise" });
        const pass = encoder.beginComputePass({ label: "noise/K3+K4" });
        stage.record(pass, n, 0);
        pass.end();
        ctx.device.queue.submit([encoder.finish()]);
        const forceOut = await readF32(ctx, force, 3 * n);
        const stateWords = new Uint32Array(await ctx.readback.read(state, STATE_BYTES));
        ctx.assertReady();
        const record = FA2_TRACE.read(
            new DataView(stateWords.buffer),
            STATE_HEADER_BYTES + TRACE_SLOT * TRACE_RECORD_BYTES,
        );
        return {
            force: forceOut,
            stateWords,
            trace: [
                scalar(record, "swing"),
                scalar(record, "traction"),
                scalar(record, "speed"),
                scalar(record, "speedEfficiency"),
            ],
        };
    } finally {
        for (const buffer of buffers) {
            buffer.destroy();
        }
    }
}

let runningAdapter: NoiseAdapter | null = null;

// ---------------------------------------------------------------- the tests (in file order: produce, measure, record, validate)

describe("noise floor (benchmarks/results/noise-floor.json)", () => {
    it("this adapter's FA2 paper-leg output: bitwise across two runs, written under GRAPHTY_NOISE_FLOOR_WRITE=1, within the cross-adapter floors of every committed adapter output", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "noise-floor" });
        const cls = adapterClass(ctx.caps);
        runningAdapter = {
            class: cls,
            vendor: ctx.caps.vendor,
            architecture: ctx.caps.architecture,
            description: ctx.caps.description,
            runtime: ctx.caps.runtime,
        };
        const first = await runPaperLeg(ctx);
        const second = await runPaperLeg(ctx);
        expectBitwiseEqual(first.force, second.force, "force, run 1 vs run 2");
        expectBitwiseEqual(first.stateWords, second.stateWords, "state, run 1 vs run 2");
        writeNoiseFixture("fa2-repulsion-exact", "karate", cls, first.force, "f32");
        writeNoiseFixture("fa2-speed-finalize", "karate", cls, first.trace, "f32");
        let compared = 0;
        for (const fixture of readNoiseFixtures("fa2-repulsion-exact", "karate")) {
            if (!isAdapterClass(fixture.adapterClass)) {
                continue;
            }
            const err = pairError("floored-stride3", first.force, fixture.values);
            console.warn(
                `[noise-floor] live force ${cls} vs committed ${fixture.adapterClass}: ${err.rel.toExponential(3)}`,
            );
            expect(err.rel, `live force vs committed ${fixture.adapterClass}`).toBeLessThanOrEqual(
                limitOf("fa2-skeleton.force.cross"),
            );
            compared++;
        }
        for (const fixture of readNoiseFixtures("fa2-speed-finalize", "karate")) {
            if (!isAdapterClass(fixture.adapterClass)) {
                continue;
            }
            const err = pairError("elementwise", first.trace, fixture.values);
            console.warn(
                `[noise-floor] live trace ${cls} vs committed ${fixture.adapterClass}: ${err.rel.toExponential(3)}`,
            );
            expect(err.rel, `live trace vs committed ${fixture.adapterClass}`).toBeLessThanOrEqual(
                limitOf("fa2-skeleton.trace.cross"),
            );
            compared++;
        }
        expect(
            compared,
            "at least this adapter's own committed fixtures (or the ones just written) exist",
        ).toBeGreaterThan(0);
    });

    for (const member of NOISE_SET) {
        const label = `${member.kernel} / ${member.fixture}`;
        it(`${label}: every committed output agrees (${member.metric}; rows ${
            Object.values(member.rows)
                .filter((r) => r !== null)
                .join(", ") || "none recorded here"
        })`, () => {
            const fixtures = readNoiseFixtures(member.kernel, member.fixture);
            const adapters = fixtures.filter((f) => isAdapterClass(f.adapterClass));
            const classes = new Set(adapters.map((f) => f.adapterClass));
            expect(
                classes.size,
                `${label}: committed adapter classes (written by ${member.writer} under GRAPHTY_NOISE_FLOOR_WRITE=1 on each adapter)`,
            ).toBeGreaterThanOrEqual(MIN_ADAPTER_CLASSES);
            expect(classes.size, `${label}: one fixture per adapter class`).toBe(adapters.length);
            for (const fixture of fixtures) {
                expect(fixture.dtype, `${label}: ${fixture.adapterClass} dtype`).toBe(member.dtype);
            }
            const samples = adapters[0].values.length;
            const failures: string[] = [];
            const resync = member.resync === true;
            for (let i = 0; i < adapters.length; i++) {
                for (let j = i + 1; j < adapters.length; j++) {
                    const err = pairError(member.metric, adapters[i].values, adapters[j].values);
                    console.warn(
                        `[noise-floor] ${label} cross ${adapters[i].adapterClass} vs ${adapters[j].adapterClass}: rel ${err.rel.toExponential(3)} abs ${err.abs.toExponential(3)}${resync ? " (free-running paper-mode traces: informational, G3-F3)" : ""}`,
                    );
                    if (resync) {
                        continue;
                    }
                    bump(
                        member.rows["cross-adapter"],
                        member,
                        "cross-adapter",
                        adapters[i].adapterClass,
                        adapters[j].adapterClass,
                        err,
                        samples,
                    );
                    collect(
                        failures,
                        check(
                            member,
                            "cross-adapter",
                            err,
                            `${label}: ${adapters[i].adapterClass} vs ${adapters[j].adapterClass}`,
                        ),
                    );
                }
            }
            for (const adapter of adapters) {
                const twin = fixtures.find((f) => f.adapterClass === `${adapter.adapterClass}${TWIN_SUFFIX}`);
                if (twin === undefined || (member.rows.twin === null && !resync)) {
                    continue;
                }
                const err = pairError(member.metric, adapter.values, twin.values);
                console.warn(
                    `[noise-floor] ${label} twin ${adapter.adapterClass}: rel ${err.rel.toExponential(3)} abs ${err.abs.toExponential(3)}${resync ? " (free-running paper-mode traces: informational, G3-F3)" : ""}`,
                );
                if (resync) {
                    continue;
                }
                bump(
                    member.rows.twin,
                    member,
                    "twin",
                    adapter.adapterClass,
                    `${adapter.adapterClass}/no-subgroups`,
                    err,
                    samples,
                );
                collect(
                    failures,
                    check(member, "twin", err, `${label}: ${adapter.adapterClass} vs its workgroup twin`),
                );
            }
            // the classes compared with the oracle references: every adapter, plus every twin of a resync member
            // (PLAN DECISION 17: the twin's oracle fixtures followed the twin's own states)
            const compared = resync ? fixtures.filter((f) => !isOracleClass(f.adapterClass)) : adapters;
            const oracleOf = (oracleClass: string, cls: string): string =>
                resync ? resyncOracleClass(oracleClass, cls) : oracleClass;
            if (member.rows["oracle-f64"] !== null) {
                for (const adapter of compared) {
                    const oracleClass = oracleOf(ORACLE_CLASS, adapter.adapterClass);
                    const oracle = fixtures.find((f) => f.adapterClass === oracleClass);
                    if (oracle === undefined) {
                        throw new Error(
                            `${label}: the ${oracleClass} fixture is missing (${member.writer} writes it under GRAPHTY_NOISE_FLOOR_WRITE=1)`,
                        );
                    }
                    const err = pairError(member.metric, adapter.values, oracle.values);
                    console.warn(
                        `[noise-floor] ${label} ${oracleClass} vs ${adapter.adapterClass}: rel ${err.rel.toExponential(3)} abs ${err.abs.toExponential(3)}`,
                    );
                    bump(
                        member.rows["oracle-f64"],
                        member,
                        "oracle-f64",
                        adapter.adapterClass,
                        oracleClass,
                        err,
                        samples,
                    );
                    collect(
                        failures,
                        check(member, "oracle-f64", err, `${label}: ${adapter.adapterClass} vs the f64 reference`),
                    );
                }
            }
            if (member.oracleF32Row !== undefined && member.oracleF32Row !== null) {
                const toleranceId = member.oracleF32Tolerance ?? null;
                for (const adapter of compared) {
                    const oracleClass = oracleOf(ORACLE_F32_CLASS, adapter.adapterClass);
                    const f32 = fixtures.find((f) => f.adapterClass === oracleClass);
                    if (f32 === undefined) {
                        throw new Error(
                            `${label}: the ${oracleClass} fixture is missing (${member.writer} writes it under GRAPHTY_NOISE_FLOOR_WRITE=1)`,
                        );
                    }
                    const err = pairError(member.metric, adapter.values, f32.values);
                    console.warn(
                        `[noise-floor] ${label} ${oracleClass} vs ${adapter.adapterClass}: rel ${err.rel.toExponential(3)} abs ${err.abs.toExponential(3)}`,
                    );
                    // P3-T5 PLAN DECISION 6: KNOWN literal mismatch -- NoiseRow.comparison is the closed union
                    // "twin" | "cross-adapter" | "oracle-f64" (contract 5.6) and "oracle-f64" is defined as the f64
                    // reference, which this comparison is NOT (b names the real reference class "oracle-f32"); the
                    // row is NOT schema-conformant until the owner adds "oracle-f32" to the union (checkpoint note,
                    // open item for P3-T7 / G3.md); readers of noise-floor.json must go by the b field of this row
                    bump(member.oracleF32Row, member, "oracle-f64", adapter.adapterClass, oracleClass, err, samples);
                    if (toleranceId !== null) {
                        const limit = limitOf(toleranceId);
                        if (!(err.rel <= limit)) {
                            failures.push(
                                `${label}: ${adapter.adapterClass} vs the f32 reference error ${err.rel.toExponential(3)} vs ${toleranceId} (${limit.toExponential(3)})`,
                            );
                        }
                    }
                }
            }
            expect(failures, `${label}: every comparison within its limit`).toEqual([]);
        });
    }

    it("records the rows, the adapters and the derived tolerances (GRAPHTY_NOISE_FLOOR_WRITE=1 only)", () => {
        if (!WRITE) {
            console.warn("[noise-floor] read-only run (GRAPHTY_NOISE_FLOOR_WRITE is not 1): nothing recorded");
            return;
        }
        for (const row of measured.values()) {
            recordNoiseRow(row);
        }
        const doc = readDoc();
        doc.recordedAt = new Date().toISOString();
        // a Step 1 seed row whose basis was not measured must not derive a tolerance: drop it so floorOf() reports the missing fixtures
        doc.rows = doc.rows.filter((r) => r.a !== SEED_CLASS);
        const classes = new Set<string>();
        for (const member of NOISE_SET) {
            for (const fixture of readNoiseFixtures(member.kernel, member.fixture)) {
                if (isAdapterClass(fixture.adapterClass)) {
                    classes.add(fixture.adapterClass);
                }
            }
        }
        for (const cls of classes) {
            const known =
                runningAdapter !== null && runningAdapter.class === cls ? runningAdapter : adapterFromClass(cls);
            const index = doc.adapters.findIndex((a) => a.class === cls);
            if (index === -1) {
                doc.adapters.push(known);
            } else if (doc.adapters[index].description === "" && known.description !== "") {
                doc.adapters[index] = known;
            }
        }
        for (const [id, spec] of Object.entries(TOLERANCE_CAPS)) {
            const floor = floorOf(doc.rows, spec.basis);
            if (floor === null) {
                throw new Error(`${id}: the basis row ${spec.basis} was not recorded (its fixtures are missing)`);
            }
            const effective = Math.max(floor, MIN_FLOOR);
            const value = Math.min(spec.cap, 10 * effective);
            const previous = doc.tolerances[id];
            doc.tolerances[id] = { value, basis: spec.basis, factor: value / effective };
            console.warn(
                `[noise-floor] tolerance ${id}: floor ${floor.toExponential(3)} (${spec.basis}) -> value ${value.toExponential(3)}, factor ${(value / effective).toFixed(2)}${previous === undefined ? "" : ` (was ${previous.value.toExponential(3)}, factor ${previous.factor.toFixed(2)})`}`,
            );
        }
        writeDoc(doc);
        console.warn(
            `[noise-floor] wrote ${NOISE_FLOOR_FILE}: ${doc.rows.length} rows, ${doc.adapters.length} adapters, ${Object.keys(doc.tolerances).length} tolerances`,
        );
    });

    it("every tolerance is derived from a recorded basis row: floor <= value <= 10 x floor, factor = value / floor, no seed entry left", () => {
        const doc = readDoc();
        expect(typeof doc.recordedAt).toBe("string");
        expect(Array.isArray(doc.adapters)).toBe(true);
        expect(Array.isArray(doc.rows)).toBe(true);
        expect(typeof doc.tolerances).toBe("object");
        for (const id of Object.keys(TOLERANCE_CAPS)) {
            expect(doc.tolerances[id], `${id}: a tolerance the file must carry`).toBeDefined();
        }
        for (const row of doc.rows) {
            expect(Number.isFinite(row.maxRelError), `${row.id}: finite maxRelError`).toBe(true);
            expect(row.maxRelError, `${row.id}: maxRelError >= 0`).toBeGreaterThanOrEqual(0);
            expect(
                row.samples,
                `${row.id}: samples > 0 (0 marks a Step 1 seed row; run this file with GRAPHTY_NOISE_FLOOR_WRITE=1 after every adapter's fixtures are committed)`,
            ).toBeGreaterThan(0);
            expect(row.a, `${row.id}: a is a measured class, not the seed`).not.toBe(SEED_CLASS);
        }
        for (const member of NOISE_SET) {
            for (const rowId of Object.values(member.rows)) {
                if (rowId !== null) {
                    expect(
                        doc.rows.some((r) => r.id === rowId),
                        `${rowId}: recorded (run this file with GRAPHTY_NOISE_FLOOR_WRITE=1 after every adapter's fixtures are committed)`,
                    ).toBe(true);
                }
            }
        }
        for (const member of NOISE_SET) {
            if (member.oracleF32Row !== undefined && member.oracleF32Row !== null) {
                expect(
                    doc.rows.some((r) => r.id === member.oracleF32Row),
                    `${member.oracleF32Row}: recorded (${member.writer})`,
                ).toBe(true);
            }
        }
        for (const rowId of REQUIRED_ROWS) {
            expect(
                doc.rows.some((r) => r.id === rowId),
                `${rowId}: recorded by P1-T5's degree / reduce tests (run them with GRAPHTY_NOISE_FLOOR_WRITE=1 on every adapter)`,
            ).toBe(true);
        }
        for (const [id, tolerance] of Object.entries(doc.tolerances)) {
            const floor = floorOf(doc.rows, tolerance.basis);
            if (floor === null) {
                throw new Error(`${id}: the basis row ${tolerance.basis} is missing from rows[]`);
            }
            const effective = Math.max(floor, MIN_FLOOR);
            expect(
                tolerance.value,
                `${id}: value >= its floor ${effective.toExponential(3)} (a floor above the value is a finding: re-fix by the spec 10.4 rule, never edit the value)`,
            ).toBeGreaterThanOrEqual(effective);
            expect(tolerance.value, `${id}: value <= 10 x its floor`).toBeLessThanOrEqual(10 * effective * (1 + 1e-9));
            expect(tolerance.factor, `${id}: factor > 0 (0 is the Step 1 seed, never derived)`).toBeGreaterThan(0);
            expect(
                Math.abs(tolerance.factor - tolerance.value / effective),
                `${id}: factor = value / floor`,
            ).toBeLessThanOrEqual(1e-6 * tolerance.factor);
            const spec = TOLERANCE_CAPS[id];
            if (spec !== undefined) {
                expect(tolerance.basis, `${id}: basis`).toBe(spec.basis);
                expect(tolerance.value, `${id}: value <= the spec cap`).toBeLessThanOrEqual(spec.cap);
            }
        }
        const classes = new Set<string>();
        for (const member of NOISE_SET) {
            for (const fixture of readNoiseFixtures(member.kernel, member.fixture)) {
                if (isAdapterClass(fixture.adapterClass)) {
                    classes.add(fixture.adapterClass);
                }
            }
        }
        for (const cls of classes) {
            expect(
                doc.adapters.some((a) => a.class === cls),
                `${cls}: listed in adapters[]`,
            ).toBe(true);
        }
    });
});
