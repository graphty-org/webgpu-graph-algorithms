/**
 * The sabotage matrix (spec 11.9 item 1, 13 rule f; contract 5.2): named textual mutations of the normative kernel
 * bodies of 4.5, each with the minimum factor by which it must break the named test's tolerance. A test compiles a
 * mutant through `withSabotage` (the `setKernelBodyOverride` seam of kernels.ts and a FRESH context, because the
 * pipeline key does not include the body) and asserts that the SAME check that passes on the real kernel fails on
 * the mutant by at least `minFactor`. A mutation that survives is a bug in the test suite and blocks the gate.
 *
 * The `find` strings below are copied from the contract's 4.5 bodies; test/sabotage/coverage.test.ts asserts that
 * each occurs exactly once in the registry's body, so a re-formatted body is caught, never silently un-mutated.
 *
 * Rows: degree, reduce, K3 (fa2-repulsion-exact) and K4 (fa2-speed-finalize) at P1-T5; segmented-reduce at P2-T2
 * (which lists "P2"); P3-T5 adds K1 / K2 / K5 and lists "P3". SABOTAGE_P3_ADDENDUM carries the rows P3 adds on the P1
 * kernels (measured by the P3 checks of test/sabotage/fa2.test.ts only).
 */

import { type GpuContext } from "../../src/context.js";
import { type KernelId, KERNELS, setKernelBodyOverride } from "../../src/kernels.js";
import { acquire } from "../setup/gpu.js";

/** One named mutation of a kernel body: a unique `find` string replaced by `replace`, and the minimum factor by which it must break the named test's tolerance. */
export interface Mutation {
    readonly name: string;
    readonly find: string;
    readonly replace: string;
    readonly minFactor: number;
    readonly test: string;
}

const DEGREE_TEST = "test/algorithms/degree.test.ts";
const REDUCE_TEST = "test/primitives/reduce.test.ts";
const SKELETON_TEST = "test/layouts/skeleton.test.ts";
const SEGMENTED_REDUCE_TEST = "test/primitives/segmented-reduce.test.ts";
const INSPECT_TEST = "test/layouts/fa2-inspect.test.ts";
const TRACE_TEST = "test/layouts/fa2-trace-parity.test.ts";

/** At least three mutations per kernel that has rows (spec 13 rule f); PARTIAL so a phase's kernels can land before its rows (the coverage test below gates by phase). */
export const SABOTAGE: Readonly<Partial<Record<KernelId, readonly Mutation[]>>> = Object.freeze({
    degree: Object.freeze([
        {
            // the last row of the dispatch range is never written: `n - 1` instead of `n` in the bound (spec 11.9)
            name: "last-row-skipped",
            find: "if (row >= P.end) { return; }",
            replace: "if (row >= P.end - 1u) { return; }",
            minFactor: 10,
            test: DEGREE_TEST,
        },
        {
            // the USE_PERM select reads the dummy (rowPtr) as the row index when USE_PERM is false
            name: "use-perm-select-swapped",
            find: "select(row, perm[row], USE_PERM)",
            replace: "select(perm[row], row, USE_PERM)",
            minFactor: 10,
            test: DEGREE_TEST,
        },
        {
            // the rebase uniform is ignored: visible only through the row-window leg of the degree check (arcBase != 0)
            name: "rebase-ignored",
            find: "colIdx[arc - P.arcBase]",
            replace: "colIdx[arc]",
            minFactor: 10,
            test: DEGREE_TEST,
        },
        {
            // a neighbour == n is impossible on a valid CSR, so the bounds check is sabotaged by counting each target twice
            name: "target-counted-twice",
            find: "select(0u, 1u, nbr < P.n)",
            replace: "select(0u, 2u, nbr < P.n)",
            minFactor: 10,
            test: DEGREE_TEST,
        },
    ]),
    reduce: Object.freeze([
        {
            // the FINAL level writes out[0] instead of out[P.outOffset]
            name: "final-writes-out-zero",
            find: "let o = P.outOffset + g;",
            replace: "let o = g;",
            minFactor: 10,
            test: REDUCE_TEST,
        },
        {
            // the u32 min identity becomes 0: every u32 min collapses to 0
            name: "u32-min-identity-zero",
            find: "return U32_MAX;",
            replace: "return 0u;",
            minFactor: 10,
            test: REDUCE_TEST,
        },
        {
            // the f32 min identity becomes 0.0: every f32 / vec4f min of positive inputs collapses to 0
            name: "f32-min-identity-zero",
            find: "return F32_MAX;",
            replace: "return 0.0;",
            minFactor: 10,
            test: REDUCE_TEST,
        },
        {
            // the level-1 bound admits i == count: the element past the range (the check's poison tail) is folded in
            name: "level-bound-inclusive",
            find: "if (i < P.count) {",
            replace: "if (i <= P.count) {",
            minFactor: 10,
            test: REDUCE_TEST,
        },
    ]),
    "fa2-repulsion-exact": Object.freeze([
        {
            // regular gravity pushes away from the centre instead of towards it (7.9)
            name: "gravity-sign-flipped",
            find: "return -P.gravity * pi.w * q / d;",
            replace: "return P.gravity * pi.w * q / d;",
            minFactor: 10,
            test: SKELETON_TEST,
        },
        {
            // |F| = k m_i m_j / d^2 instead of the 7.2 law k m_i m_j / d
            name: "inverse-square-law",
            find: "f = f + d * (k / d2);",
            replace: "f = f + d * (k / (d2 * length(d)));",
            minFactor: 10,
            test: SKELETON_TEST,
        },
        {
            // the self pair is no longer excluded: it takes the coincident kick of magnitude k m_i m_i / 0.01
            name: "self-pair-guard-removed",
            find: "if (o.w > 0.0 && jj != i) {",
            replace: "if (o.w > 0.0) {",
            minFactor: 10,
            test: SKELETON_TEST,
        },
        {
            // the mass lane read as .x (4.4 rule 5)
            name: "mass-lane-x",
            find: "let k = P.scalingRatio * pi.w * o.w;",
            replace: "let k = P.scalingRatio * pi.x * o.x;",
            minFactor: 10,
            test: SKELETON_TEST,
        },
    ]),
    "fa2-speed-finalize": Object.freeze([
        {
            // the CPU's conditional halving of the efficiency (7.2) weakened; visible when swing / traction > 2 and eff > 0.05
            name: "efficiency-halving-weakened",
            find: "eff = eff * 0.5;",
            replace: "eff = eff * 0.9;",
            minFactor: 10,
            test: SKELETON_TEST,
        },
        {
            // the 1.3 rise removed; visible when swing <= jitter * traction and S.speed < 1000
            name: "efficiency-rise-removed",
            find: "eff = eff * 1.3;",
            replace: "eff = eff * 1.0;",
            minFactor: 10,
            test: SKELETON_TEST,
        },
        {
            // swing and traction swapped before the reduction: S and T both see the swap, in both SWING_MODEs
            name: "swing-traction-swapped",
            find: "vec4f(st, 0.0, 0.0)",
            replace: "vec4f(st.yx, 0.0, 0.0)",
            minFactor: 10,
            test: SKELETON_TEST,
        },
    ]),
    "segmented-reduce": Object.freeze([
        {
            // contract 5.5 spells this row `row >= P.end` -> `row > P.end`; that mutant's extra invocation writes
            // out[n], an in-bounds slot under the pool's size classes, so it is not reliably caught. The form here
            // skips the last row deterministically (spec 11.9 item 1: "n - 1 instead of n in the bound"). Pending
            // owner confirmation, docs/decisions/G2.md.
            name: "last-row-skipped",
            find: "row >= P.end",
            replace: "row + 1u >= P.end",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
        {
            // the weight read under HAS_WEIGHTS dropped: every arc weighs 1.0 (the check set's weights average 2.1)
            name: "weight-read-dropped",
            find: "weight = weights[arc - P.arcBase]",
            replace: "weight = 1.0",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
        {
            // the HAS_WEIGHTS select inverted: a weighted core folds 1.0 per arc
            name: "has-weights-inverted",
            find: "if (HAS_WEIGHTS)",
            replace: "if (!HAS_WEIGHTS)",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
        {
            // the min identity becomes 0.0: every row's min collapses to 0 against weights >= 0.25
            name: "min-identity-zero",
            find: "return F32_MAX;",
            replace: "return 0.0;",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
        {
            // a row with no arcs is never written: the check's 100 empty rows keep the sentinel
            name: "empty-rows-skipped",
            find: "out[i] = select(acc, comb(out[i], acc), P.accumulate == 1u);",
            replace: "if (a1 > a0) { out[i] = select(acc, comb(out[i], acc), P.accumulate == 1u); }",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
        {
            // the USE_PERM select reads the dummy (rowPtr) as the row index when USE_PERM is false
            name: "perm-select-swapped",
            find: "select(row, perm[row], USE_PERM)",
            replace: "select(perm[row], row, USE_PERM)",
            minFactor: 10,
            test: SEGMENTED_REDUCE_TEST,
        },
    ]),
    "fa2-stats-finalize": Object.freeze([
        {
            // the settle counter is never reset: with settleThreshold 0 the oracle keeps 0, the mutant counts every iteration (visible at the K1 fold of iteration 2)
            name: "settled-count-never-reset",
            find: "S.settledCount = select(0u, S.settledCount + 1u, meanDisp <= P.settleThreshold * S.rmsRadius);",
            replace: "S.settledCount = S.settledCount + 1u;",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // the centroid divides by n - 1: a 1 / (n - 1) relative error on every component (3% on karate)
            name: "centroid-over-n-minus-1",
            find: "let c = tSum.xyz / n;",
            replace: "let c = tSum.xyz / (n - 1.0);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // the RMS radius without its square root: r^2 instead of r (tens of percent on every layout)
            name: "rms-radius-without-sqrt",
            find: "S.rmsRadius = sqrt(max(tSum.w, 0.0) / n);",
            replace: "S.rmsRadius = max(tSum.w, 0.0) / n;",
            minFactor: 10,
            test: INSPECT_TEST,
        },
    ]),
    "fa2-attraction": Object.freeze([
        {
            // attraction pushes away from the neighbour instead of towards it
            name: "attraction-sign-flipped",
            find: "f = f + d * mag;",
            replace: "f = f - d * mag;",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // the LINLOG select arguments swapped: the linear law runs the linlog magnitude and vice versa (30% at |d| = 1)
            name: "linlog-select-swapped",
            find: "let mag = select(w, w * log(1.0 + len) / len, LINLOG);",
            replace: "let mag = select(w * log(1.0 + len) / len, w, LINLOG);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // PLAN DECISION 2 (replaces the inert self-loop row): the row bound off by one reads the next row's first arc into every row
            name: "row-bound-inclusive",
            find: "for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {",
            replace: "for (var a = rowPtr[i]; a <= rowPtr[i + 1u]; a = a + 1u) {",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // spec 11.9 item 1: select arguments swapped in the USE_PERM read; with USE_PERM false the dummy rowPtr is read as the row index
            name: "use-perm-select-swapped",
            find: "let i = select(row, perm[row], USE_PERM);",
            replace: "let i = select(perm[row], row, USE_PERM);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
    ]),
    "fa2-integrate": Object.freeze([
        {
            // a displacement clamp inserted (D25 forbids one): the first iteration's O(1..10) steps collapse to 0.01
            name: "dp-clamped",
            find: "dp = select(f * factor, vec3f(0.0), fixed);",
            replace: "dp = select(clamp(f * factor, vec3f(-0.01), vec3f(0.01)), vec3f(0.0), fixed);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // oldForce never stored: iteration 2's swing becomes m |F| instead of m |F - F_old| (paper mode)
            name: "store-old-skipped",
            find: "if (SWING_MODE == 0u) { store_old(i, f); }",
            replace: "if (SWING_MODE == 2u) { store_old(i, f); }",
            minFactor: 10,
            test: TRACE_TEST,
        },
        {
            // PLAN DECISION 2 (replaces the inert 2D-z row): the stale oldForce read in place of the force (spec 11.9 item 1); iteration 1 then moves nothing
            name: "stale-force-read",
            find: "let f = load_force(i);",
            replace: "let f = load_old(i);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
        {
            // the fixed select swapped: free nodes stay, fixed nodes would move
            name: "fixed-select-swapped",
            find: "dp = select(f * factor, vec3f(0.0), fixed);",
            replace: "dp = select(vec3f(0.0), f * factor, fixed);",
            minFactor: 10,
            test: INSPECT_TEST,
        },
    ]),
});

/**
 * Rows added on the P1 kernels by P3-T5's review fix (docs/decisions/G3.md, the review of 2026-09-16), kept out of
 * SABOTAGE because test/sabotage/coverage.test.ts pins the P1 rows by name and test/sabotage/fa2-skeleton.test.ts
 * requires every K4 row to name the skeleton test (both P1-T5 files): test/sabotage/fa2.test.ts runs these rows with
 * the P3 checks (the re-synchronised trace legs) exactly as it runs SABOTAGE's, with the same find-uniqueness check.
 */
export const SABOTAGE_P3_ADDENDUM: Readonly<Partial<Record<KernelId, readonly Mutation[]>>> = Object.freeze({
    "fa2-speed-finalize": Object.freeze([
        {
            // CONTRACT DECISION K4-1 (G3-F6): the halving predicate at its exact-equality knife edge. After load() the
            // paper-mode traction is EXACTLY half the swing (oldForce = 0), so `>=` halves the efficiency at every
            // first iteration where `>` (the port's `swing / traction > 2`) never does; the difference is a factor 2
            // in speedEfficiency from iteration 1 on, measured 7.4e+3x the re-synchronised f32 tolerance on karate
            name: "halving-at-exact-equality",
            find: "swing > 2.0 * tr",
            replace: "swing >= 2.0 * tr",
            minFactor: 10,
            test: TRACE_TEST,
        },
    ]),
});

/** The phases whose kernels ALL have their rows: ["P1"] at P1-T5, + "P2" at P2-T2, + "P3" at P3-T5; test/sabotage/coverage.test.ts asserts every KERNELS entry whose `phase` is listed here has >= 3 rows, except SABOTAGE_EXEMPT. */
export const SABOTAGE_PHASES: readonly ("P1" | "P2" | "P3")[] = Object.freeze(["P1", "P2", "P3"]);

/** Kernels with no oracle-sensitive arithmetic to mutate (a wrong fill / toScene fails the exact-equality tests directly): ["fill", "fa2-to-scene"]. */
export const SABOTAGE_EXEMPT: readonly KernelId[] = Object.freeze(["fill", "fa2-to-scene"]);

/**
 * The mutated body (throws when `find` is absent or not unique in the normative body).
 * @param id - the kernel
 * @param mutation - the row to apply
 * @returns the registry body with the one occurrence of `find` replaced by `replace`
 */
export function sabotagedBody(id: KernelId, mutation: Mutation): string {
    const { body } = KERNELS[id];
    const first = body.indexOf(mutation.find);
    if (first < 0) {
        throw new Error(`sabotage ${id}/${mutation.name}: the find string is absent from the normative body`);
    }
    if (body.indexOf(mutation.find, first + mutation.find.length) >= 0) {
        throw new Error(`sabotage ${id}/${mutation.name}: the find string is not unique in the normative body`);
    }
    return body.slice(0, first) + mutation.replace + body.slice(first + mutation.find.length);
}

/**
 * Installs the mutation through setKernelBodyOverride, runs `fn` with a FRESH context, restores.
 * @param id - the kernel
 * @param mutation - the row to apply
 * @param fn - the check to run against the mutant (the context is disposed afterwards)
 * @returns whatever `fn` returns
 */
export async function withSabotage<T>(
    id: KernelId,
    mutation: Mutation,
    fn: (ctx: GpuContext) => Promise<T>,
): Promise<T> {
    setKernelBodyOverride(id, sabotagedBody(id, mutation));
    try {
        const ctx = await acquire({ label: `sabotage/${id}/${mutation.name}` });
        try {
            return await fn(ctx);
        } finally {
            ctx.dispose();
        }
    } finally {
        setKernelBodyOverride(id, null);
    }
}

/** The outcome of one measured check: the largest error / tolerance ratio over its samples (Infinity for a mismatch under a tolerance of 0). */
export interface CheckReport {
    readonly worst: number;
    readonly worstLabel: string;
    readonly samples: number;
}

/**
 * The error / tolerance ratio of one sample: 0 for an exact match, Infinity for any mismatch under a tolerance of 0
 * (a bitwise test), NaN counts as Infinity.
 * @param error - the measured error (absolute or relative, the caller's choice, the same as the tolerance's)
 * @param tolerance - the test's tolerance in the same units
 * @returns the ratio the sabotage tests compare with `minFactor`
 */
export function ratioOf(error: number, tolerance: number): number {
    if (Number.isNaN(error)) {
        return Infinity;
    }
    if (tolerance <= 0) {
        return error === 0 ? 0 : Infinity;
    }
    return error / tolerance;
}

/**
 * The report of several checks: the largest ratio and the label it came from.
 * @param reports - the reports to merge
 * @returns one report
 */
export function mergeReports(reports: readonly CheckReport[]): CheckReport {
    let worst = 0;
    let worstLabel = "(no samples)";
    let samples = 0;
    for (const r of reports) {
        samples += r.samples;
        const w = Number.isNaN(r.worst) ? Infinity : r.worst;
        if (w > worst) {
            worst = w;
            ({ worstLabel } = r);
        }
    }
    return { worst, worstLabel, samples };
}

/**
 * THE assertion of a check (spec 11.9 item 1: the sabotage test asserts this same function fails on the mutant).
 * @param report - the measured report
 */
export function assertCheckPasses(report: CheckReport): void {
    if (!(report.worst <= 1)) {
        throw new Error(
            `check failed: worst error / tolerance ratio ${report.worst} > 1 at ${report.worstLabel} (${report.samples} samples)`,
        );
    }
}
