/**
 * The layout-facing public types (spec 3.3, 7.19): the stats records, the GPU simulation interface that extends the
 * design-14.3 LayoutSimulation mirror, the run options and the GPU-only tuning knobs. Types only.
 */

import type { F32, GraphSnapshot, NodeMask } from "@graphty/graph-format";

import type { LayoutSimulation } from "./accelerator.js";

/** Spec 3.3 LayoutStatsBase, verbatim; the three grid fields are null on the exact tier (P3 always). */
export interface LayoutStatsBase {
    readonly iteration: number;
    readonly meanDisplacement: number;
    readonly rmsRadius: number;
    readonly layoutRadius: number;
    readonly centroid: readonly [number, number, number];
    readonly repulsionTier: "exact" | "grid";
    readonly maxCellOccupancy: number | null;
    readonly outsideGrid: number | null;
    readonly msPerIteration: number | null;
}

/**
 * One per-iteration trace record of the last completed batch (spec 3.3 ForceAtlas2Stats.trace element).
 * Exported: the element type of ForceAtlas2Stats.trace; re-exported from src/index.ts at P3-T3.
 * @public
 */
export interface ForceAtlas2TraceRecord {
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly meanDisplacement: number;
    readonly settledCount: number;
}

/** Spec 3.3 ForceAtlas2Stats, verbatim. */
export interface ForceAtlas2Stats extends LayoutStatsBase {
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly trace: ReadonlyArray<ForceAtlas2TraceRecord>;
}

/** Options of GpuLayoutSimulation.run (spec 3.3). */
export interface RunOptions {
    readonly maxIter?: number | undefined;
    readonly batch?: number | undefined;
    readonly signal?: AbortSignal | undefined;
}

/** Spec 3.3 GpuLayoutSimulation, verbatim (LayoutSimulation is the design-14.3 mirror of accelerator.ts). */
export interface GpuLayoutSimulation<Options, Stats extends LayoutStatsBase> extends LayoutSimulation {
    load(snapshot: GraphSnapshot, positions: F32): void;
    /**
     * Submits k iterations; resolves when their batch has been read back into `positions`. With `maxInFlight`
     * batches in flight the call COALESCES: nothing is queued and the OLDEST pending batch's promise is returned
     * (spec 7.19 item 3).
     * @param iterations - the number of iterations to submit (default `iterationsPerStep`)
     * @returns resolves when the batch carrying them has landed in the owner's array
     */
    step(iterations?: number): Promise<void>;
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
    readonly inFlight: number;
    readonly iterationsDone: number;
    readonly stats: Stats;
    flush(): Promise<void>;
    reheat(): void;
    setParams(patch: Partial<Options>): void;
    run(options?: RunOptions): Promise<Stats>;
    inspect?(name: string): Promise<Float32Array | Uint32Array>;
}

/**
 * Spec 3.3 GpuLayoutTuning, verbatim; the grid knobs are accepted and stored in P3 but only `repulsion`,
 * `exactMaxNodes`, `deterministic` and `compat` have an effect (grid tier = P4).
 */
export interface GpuLayoutTuning {
    readonly repulsion?: "exact" | "grid" | "auto" | undefined;
    readonly exactMaxNodes?: number | undefined;
    readonly nearMax?: number | undefined;
    readonly deterministic?: boolean | undefined;
    readonly gridMax2D?: number | undefined;
    readonly gridMax3D?: number | undefined;
    readonly extentFactor?: number | undefined;
    readonly compat?: "paper" | "networkx" | undefined;
}

/** The resolved tuning record (defaults from constants.ts LAYOUT_TUNING_DEFAULTS applied). */
export interface ResolvedLayoutTuning {
    readonly repulsion: "exact" | "grid" | "auto";
    readonly exactMaxNodes: number;
    readonly nearMax: number;
    readonly deterministic: boolean;
    readonly gridMax2D: number;
    readonly gridMax3D: number;
    readonly extentFactor: number;
    readonly compat: "paper" | "networkx";
}
