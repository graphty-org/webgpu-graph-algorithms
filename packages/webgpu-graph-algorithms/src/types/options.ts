/**
 * The option records of the layouts (spec 9.3, 7.14): structural mirrors of @graphty/layout's option types (D27),
 * copied field for field so a `ForceAtlas2Options` object the element parses is accepted here without a cast. Types
 * only: this file imports nothing at runtime.
 */

import type { F32, NodeId, NodeMask } from "@graphty/graph-format";

/** Design 14.3 CommonLayoutOptions, mirrored verbatim. */
export interface CommonLayoutOptions {
    readonly dim?: 2 | 3 | undefined;
    readonly scale?: number | undefined;
    readonly center?: ArrayLike<number> | undefined;
    readonly seed?: number | null | undefined;
}

/** Spec 9.3 SimulationOptions, mirrored verbatim (layout-owned; the CPU simulations ignore maxInFlight). */
export interface SimulationOptions {
    readonly settleThreshold?: number | undefined;
    readonly settleWindow?: number | undefined;
    readonly iterationsPerStep?: number | undefined;
    readonly maxInFlight?: number | undefined;
}

/**
 * Spec 9.3 ForceAtlas2Options, mirrored verbatim (same names and defaults as
 * layout/src/layouts/force-directed/forceatlas2.ts lines 26-42).
 */
export interface ForceAtlas2Options extends CommonLayoutOptions, SimulationOptions {
    readonly maxIter?: number | undefined;
    readonly jitterTolerance?: number | undefined;
    readonly scalingRatio?: number | undefined;
    readonly gravity?: number | undefined;
    readonly strongGravity?: boolean | undefined;
    readonly distributedAction?: boolean | undefined;
    readonly linlog?: boolean | undefined;
    readonly nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    readonly nodeSize?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    readonly weight?: boolean | string | null | undefined;
    readonly dissuadeHubs?: boolean | undefined;
}

/** Spec 9.3 FruchtermanReingoldOptions, mirrored for the LayoutAccelerator mirror's method signature (P5 implements it). */
export interface FruchtermanReingoldOptions extends CommonLayoutOptions, SimulationOptions {
    readonly k?: number | null | undefined;
    readonly iterations?: number | undefined;
    readonly fixed?: NodeMask | string | null | undefined;
}

/** Spec 9.3 SpringElectricalOptions, mirrored for the LayoutAccelerator mirror's method signature (P5 implements it). */
export interface SpringElectricalOptions extends CommonLayoutOptions, SimulationOptions {
    readonly springLength?: number | undefined;
    readonly springCoefficient?: number | undefined;
    readonly gravity?: number | undefined;
    readonly dragCoefficient?: number | undefined;
    readonly timeStep?: number | undefined;
}

/**
 * The resolved (defaults applied) ForceAtlas2 option record the simulation keeps; every field present.
 * Exported: consumed by src/layouts/forceatlas2.ts (P3-T2, resolveForceAtlas2Options) and the option tests.
 * @public
 */
export interface ResolvedForceAtlas2Options {
    readonly maxIter: number;
    readonly jitterTolerance: number;
    readonly scalingRatio: number;
    readonly gravity: number;
    readonly strongGravity: boolean;
    readonly distributedAction: boolean;
    readonly linlog: boolean;
    readonly nodeMass: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly nodeSize: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly weight: boolean | string | null;
    readonly dissuadeHubs: boolean;
    readonly dim: 2 | 3;
    readonly scale: number;
    readonly center: readonly [number, number, number];
    readonly seed: number | null;
    readonly settleThreshold: number;
    readonly settleWindow: number;
    readonly iterationsPerStep: number;
    readonly maxInFlight: number;
}
