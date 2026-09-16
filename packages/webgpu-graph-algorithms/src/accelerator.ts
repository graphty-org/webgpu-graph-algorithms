/**
 * createAccelerator (spec 3.3, 9; contract 3.14): the injectable object that satisfies the CPU packages'
 * AlgorithmAccelerator and LayoutAccelerator interfaces STRUCTURALLY (spec 9.2, 9.3; the mirrors of
 * src/types/accelerator.ts until W1, D27). At P3 it carries `forceAtlas2`, `release` and `dispose` and nothing
 * else: the CPU-side dispatchers (`accelerated()`, `createSimulation()`) test `acc.pageRank !== undefined` and
 * route to the CPU when the member is absent (spec 2.4 row "method missing"), so a method the GPU does not
 * implement must not exist here -- never a throwing stub. The algorithm members arrive one per shipped algorithm
 * from P7; `fruchtermanReingold` / `springElectrical` with P5.
 */

import { type GraphSnapshot } from "@graphty/graph-format";

import { type GpuContext } from "./context.js";
import { createForceAtlas2 } from "./layouts/forceatlas2.js";
import { type AcceleratorOptions, type GpuAccelerator } from "./types/accelerator.js";
import { type ForceAtlas2Stats, type GpuLayoutSimulation, type GpuLayoutTuning } from "./types/layout.js";
import { type ForceAtlas2Options } from "./types/options.js";

/** The `algorithms` record of AcceleratorOptions (spec 3.3), named for the copy helpers. */
type AlgorithmDefaults = NonNullable<AcceleratorOptions["algorithms"]>;

/** The `algorithms.betweenness` record of AcceleratorOptions (spec 3.3: the `k` / `sources` defaults until A2). */
type BetweennessDefaults = NonNullable<AlgorithmDefaults["betweenness"]>;

/**
 * Frozen copy of the betweenness defaults; `sources` is copied into a fresh frozen array so a later mutation of
 * the caller's list is never seen by the accelerator.
 * @param defaults - the caller's record
 * @returns the frozen copy
 */
function copyBetweenness(defaults: BetweennessDefaults): BetweennessDefaults {
    const copy: { k?: number | undefined; sources?: readonly number[] | undefined } = { ...defaults };
    if (defaults.sources !== undefined) {
        copy.sources = Object.freeze([...defaults.sources]);
    }
    return Object.freeze(copy);
}

/**
 * Frozen copy of the algorithms record, one level deeper for `betweenness`.
 * @param algorithms - the caller's record
 * @returns the frozen copy
 */
function copyAlgorithms(algorithms: AlgorithmDefaults): AlgorithmDefaults {
    const copy: { betweenness?: BetweennessDefaults | undefined } = { ...algorithms };
    if (algorithms.betweenness !== undefined) {
        copy.betweenness = copyBetweenness(algorithms.betweenness);
    }
    return Object.freeze(copy);
}

/**
 * The frozen deep copy of the accelerator options (spec 3.3 GpuAccelerator.options: "the tuning defaults given
 * to createAccelerator; frozen"; contract 3.14): the record, its `layout` tuning, its `algorithms` record, the
 * `betweenness` record and the `sources` list are each fresh frozen objects, so neither the caller's later edits
 * nor a consumer's writes can change what the accelerator's simulations inherit.
 * @param options - the caller's options, or undefined
 * @returns the frozen copy (an empty frozen record when nothing was given)
 */
function freezeOptions(options: AcceleratorOptions | undefined): Readonly<AcceleratorOptions> {
    if (options === undefined) {
        return Object.freeze({});
    }
    const copy: { layout?: GpuLayoutTuning | undefined; algorithms?: AlgorithmDefaults | undefined } = { ...options };
    if (options.layout !== undefined) {
        copy.layout = Object.freeze({ ...options.layout });
    }
    if (options.algorithms !== undefined) {
        copy.algorithms = copyAlgorithms(options.algorithms);
    }
    return Object.freeze(copy);
}

/**
 * Spec 3.3 createAccelerator, verbatim: the object implementing AlgorithmAccelerator & LayoutAccelerator
 * structurally; at P3 it carries forceAtlas2, release and dispose. One per call (the app creates one and injects
 * it, spec 2.4); `kind` is "webgpu"; `options` is a frozen deep copy; `forceAtlas2(o)` is
 * `createForceAtlas2(ctx, { ...o, ...options.layout })`, so the GPU tuning given here wins over anything the
 * CPU-typed option object carries (spec 3.3: tuning never comes from the caller of the accelerator method);
 * `release(s)` is `ctx.release(s)`; `dispose()` is `ctx.dispose()`.
 * @param ctx - the context every simulation the accelerator creates runs on
 * @param options - GPU-only defaults inherited by every simulation (`layout`) and, from P9, the algorithm defaults
 * @returns the injectable accelerator
 * @throws E_DISPOSED / E_DEVICE_LOST from `ctx.assertReady()` (here and inside `forceAtlas2()`); `forceAtlas2()`
 *   also throws what createForceAtlas2 throws (E_UNSUPPORTED for `nodeSize`, E_INVALID_ARGUMENT for a bad range)
 */
export function createAccelerator(ctx: GpuContext, options?: AcceleratorOptions): GpuAccelerator {
    ctx.assertReady();
    const frozen = freezeOptions(options);
    return {
        kind: "webgpu",
        ctx,
        options: frozen,
        /**
         * The ForceAtlas2 simulation with this accelerator's layout tuning (spec 3.3; contract 3.14).
         * @param o - the CPU option type (spec 9.3 ForceAtlas2Options); GPU tuning keys come from `options.layout`
         * @returns a fresh simulation in state "created"
         */
        forceAtlas2(o?: ForceAtlas2Options): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats> {
            ctx.assertReady();
            return createForceAtlas2(ctx, { ...o, ...frozen.layout });
        },
        /**
         * Destroys every device buffer recorded for the snapshot (spec 4.5); delegates to ctx.release.
         * @param s - the snapshot the app is done with
         */
        release(s: GraphSnapshot): void {
            ctx.release(s);
        },
        /**
         * Disposes the context (spec 2.8); delegates to ctx.dispose, idempotent.
         */
        dispose(): void {
            ctx.dispose();
        },
    };
}
