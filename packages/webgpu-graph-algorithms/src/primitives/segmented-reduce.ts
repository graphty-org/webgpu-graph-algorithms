/**
 * The segmented (per-row) reduction primitive of spec 6 row 3 in its thread-per-row tier (P2-P3): one invocation per
 * CSR row folds the caller's VALUE snippet over the row's arcs into `out[row]` (f32), a row with no arcs receiving
 * the identity element. The degree tiers of degreeOrder() (subgroup-per-row for the mid tier, workgroup-per-row for
 * the high tier) land at P4; until then `tiers !== null` is E_UNSUPPORTED and USE_PERM is always false (the perm
 * slot carries the rowPtr dummy of graphBindings). The row and arc counts come from the core's binding sizes: the
 * residency binds every array at its exact byte length (contract 3.8), so rowPtr is 4(n + 1) bytes and colIdx
 * 4 x arcCount.
 */

import { WebGpuGraphError } from "../errors.js";
import { plan1d } from "../kernel/dispatch.js";
import { type Kernel } from "../kernel/kernel.js";
import { graphBindings, graphOverrides, kernelSpec, RANGE_PARAMS } from "../kernels.js";
import { type CoreBinding } from "../memory/residency.js";
import { type Binding } from "../types/memory.js";
import { type ReduceOp, type ReduceScope } from "./reduce.js";

/** The degree tiers of degreeOrder(): the permutation binding and the CPU-side segmentOffsets [0, hiEnd, midEnd, lowEnd, n]. */
export interface DegreeTiers {
    readonly perm: Binding;
    readonly segmentOffsets: readonly [number, number, number, number, number];
}

/** Options of segmentedReduce. `valueSnippet` is the Gunrock-style functor: WGSL statements assigning `v` from (row, arc, nbr, weight) (4.5; `nbr` because `target` is a WGSL reserved word). */
export interface SegmentedReduceOptions {
    readonly op: ReduceOp;
    readonly valueSnippet: string;
    readonly tiers: DegreeTiers | null;
    readonly accumulate?: boolean | undefined;
}

/** A prepared segmented reduce (P2-P3: the thread-per-row tier only; `tiers !== null` -> E_UNSUPPORTED { feature: "segmentedReduce.tiers" } until P4). */
export interface SegmentedReducePlanner {
    /** Records one dispatch over rows [0, n) (tiers null) writing out[i] (f32) per row; a row with no arcs gets the identity element. */
    record(pass: GPUComputePassEncoder, core: CoreBinding, out: Binding): void;
}

/** The identifiers a VALUE snippet may name (contract 3.11, 4.5); every other identifier is rejected textually before compose. */
const VALUE_SNIPPET_VOCABULARY: ReadonlySet<string> = new Set(["row", "arc", "nbr", "weight", "v"]);

/**
 * The WGSL words a snippet statement may use that are not identifiers: the statement keywords (never the flow
 * keywords `return` / `break` / `continue`, which would leave the fold), the scalar type constructors and the
 * builtin math functions a value expression may call (a bounded list; a binding, a uniform or a module function such
 * as `identity` / `comb` / `linear_id` is NOT in it and is rejected as an identifier).
 */
const VALUE_SNIPPET_WGSL_WORDS: ReadonlySet<string> = new Set([
    "if",
    "else",
    "let",
    "var",
    "const",
    "true",
    "false",
    "f32",
    "u32",
    "i32",
    "bool",
    "abs",
    "ceil",
    "clamp",
    "exp",
    "exp2",
    "floor",
    "fract",
    "inverseSqrt",
    "log",
    "log2",
    "max",
    "min",
    "mix",
    "pow",
    "round",
    "select",
    "sign",
    "sqrt",
    "step",
    "trunc",
]);

/**
 * The OP override value of an operator (the body's `OP == 1u` / `OP == 2u` tests, contract 4.5).
 * @param op - the operator
 * @returns 0 for sum, 1 for min, 2 for max
 */
function opCode(op: ReduceOp): number {
    switch (op) {
        case "sum":
            return 0;
        case "min":
            return 1;
        case "max":
            return 2;
        default:
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", `segmentedReduce: unknown op ${String(op)}`, {
                argument: "op",
                value: op,
                expected: "sum | min | max",
            });
    }
}

/**
 * WGSL comments removed (a comment may mention `target` or any other word).
 * @param text - WGSL text
 * @returns the text with block and line comments replaced by spaces
 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
}

/**
 * The textual vocabulary check of contract 3.11: the snippet must assign `v` (`=`, `+=`, `-=`, `*=` or `/=`) and its
 * identifiers may be only row, arc, nbr, weight and v; numeric literals and the WGSL words of
 * VALUE_SNIPPET_WGSL_WORDS are not identifiers and pass; anything else -- a binding, a uniform, a module function,
 * a locally declared name, a flow keyword, a reserved word such as `target` -- is E_SHADER_COMPILE { stage:
 * "compose", slot: "VALUE", identifier } on every device, before any shader is created.
 * @param snippet - the VALUE snippet text
 */
function validateValueSnippet(snippet: string): void {
    const code = stripComments(snippet);
    if (!/\bv\s*[-+*/]?=(?!=)/.test(code)) {
        throw new WebGpuGraphError("E_SHADER_COMPILE", "segmentedReduce: the VALUE snippet never assigns v", {
            id: "segmented-reduce",
            stage: "compose",
            slot: "VALUE",
        });
    }
    const tokens = code.match(/[A-Za-z_][A-Za-z0-9_]*|[0-9][0-9A-Za-z_.]*/g) ?? [];
    for (const token of tokens) {
        if (/^[0-9]/.test(token) || VALUE_SNIPPET_VOCABULARY.has(token) || VALUE_SNIPPET_WGSL_WORDS.has(token)) {
            continue;
        }
        throw new WebGpuGraphError(
            "E_SHADER_COMPILE",
            `segmentedReduce: the VALUE snippet names "${token}"; the only identifiers allowed are row, arc, nbr, weight and v`,
            { id: "segmented-reduce", stage: "compose", slot: "VALUE", identifier: token },
        );
    }
}

/**
 * The row count of a core from its rowPtr binding (4(n + 1) bytes).
 * @param core - the core
 * @returns n
 */
function rowCountOf(core: CoreBinding): number {
    const bytes = core.rowPtr.size;
    if (bytes < 4 || bytes % 4 !== 0) {
        throw new WebGpuGraphError(
            "E_INVALID_ARGUMENT",
            `segmentedReduce: a rowPtr binding of ${bytes} bytes is not 4(n + 1)`,
            {
                argument: "core.rowPtr",
                value: bytes,
                expected: "a positive multiple of 4",
            },
        );
    }
    return bytes / 4 - 1;
}

/**
 * Rejects a windowed core (executed at P4).
 * @param core - the core
 */
function assertNotWindowed(core: CoreBinding): void {
    if (core.plan === "windowed" || core.windows !== null) {
        throw new WebGpuGraphError("E_UNSUPPORTED", "segmentedReduce: windowed cores are executed at P4", {
            feature: "segmentedReduce.windowed",
        });
    }
}

/** The thread-per-row planner: ONE `segmented-reduce` dispatch with TIER 0 over every row. */
class ThreadPerRowPlanner implements SegmentedReducePlanner {
    private readonly scope: ReduceScope;
    private readonly kernel: Kernel;
    private readonly hasWeights: boolean;
    private readonly accumulate: boolean;

    /**
     * Wraps a compiled thread-per-row pipeline with the pattern it was compiled for.
     * @param scope - the scope the pipeline was prepared in
     * @param kernel - the compiled kernel
     * @param hasWeights - the HAS_WEIGHTS the pipeline was compiled with
     * @param accumulate - whether record() combines into out instead of overwriting
     */
    constructor(scope: ReduceScope, kernel: Kernel, hasWeights: boolean, accumulate: boolean) {
        this.scope = scope;
        this.kernel = kernel;
        this.hasWeights = hasWeights;
        this.accumulate = accumulate;
    }

    /**
     * Records the dispatch: rows [0, n), arcs [0, arcCount), plan1d(n); nothing for n = 0 (no zero-length binding is
     * ever created).
     * @param pass - the pass to record into
     * @param core - a core with the SAME weights pattern as the one prepared (any snapshot)
     * @param out - at least 4n bytes of f32
     */
    record(pass: GPUComputePassEncoder, core: CoreBinding, out: Binding): void {
        assertNotWindowed(core);
        if ((core.weights !== null) !== this.hasWeights) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                "segmentedReduce: the core's weights pattern differs from the one prepared",
                {
                    argument: "core",
                    value: core.weights !== null,
                    expected: this.hasWeights,
                },
            );
        }
        const n = rowCountOf(core);
        if (n === 0) {
            return;
        }
        const arcCount = core.colIdx === null ? 0 : core.colIdx.size / 4;
        if (out.size < 4 * n) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `segmentedReduce: out holds ${out.size} bytes, ${4 * n} needed`,
                {
                    argument: "out",
                    value: out.size,
                    expected: `>= ${4 * n}`,
                },
            );
        }
        const params = this.scope.params(RANGE_PARAMS, {
            start: 0,
            end: n,
            arcBase: 0,
            arcEnd: arcCount,
            accumulate: this.accumulate ? 1 : 0,
            n,
        });
        const bound = this.kernel.bind({ ...graphBindings(core, null), out, P: params.binding });
        this.kernel.dispatch(pass, bound, plan1d(n, this.scope.workgroupSize, this.scope.caps), [params.offset]);
    }
}

/**
 * Prepares the thread-per-row pipeline for a snapshot's dummy pattern (USE_PERM, HAS_WEIGHTS) and snippet.
 * @param scope - the reduce scope (pipelines, pool, params writer)
 * @param core - the core whose weights pattern selects HAS_WEIGHTS (USE_PERM is false: no tiers at P2)
 * @param options - operator, snippet, tiers (must be null), accumulate
 * @returns the planner
 */
export async function prepareSegmentedReduce(
    scope: ReduceScope,
    core: CoreBinding,
    options: SegmentedReduceOptions,
): Promise<SegmentedReducePlanner> {
    if (options.tiers !== null) {
        throw new WebGpuGraphError("E_UNSUPPORTED", "segmentedReduce: the degree tiers land at P4; pass tiers: null", {
            feature: "segmentedReduce.tiers",
        });
    }
    assertNotWindowed(core);
    const op = opCode(options.op);
    validateValueSnippet(options.valueSnippet);
    const overrides = { ...graphOverrides(core, null), OP: op, TIER: 0 };
    const spec = kernelSpec("segmented-reduce", overrides, { VALUE: options.valueSnippet });
    const kernel = await scope.pipelines.kernel(spec);
    return new ThreadPerRowPlanner(scope, kernel, core.weights !== null, options.accumulate === true);
}
