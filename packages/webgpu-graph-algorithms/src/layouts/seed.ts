/**
 * The CPU port's random number generator, bit for bit, and the NaN-row seeding both paths share (spec 7.2 "Initial
 * positions", 9.3 seedPositions, 7.14 `pos`, 7.19 topology change): a seed gives the same start on the CPU and the
 * GPU because both write the same f32 values in index order. The package carries its own copy of the LCG for its
 * whole life (D27: it cannot import @graphty/layout); W1 cross-tests it against the real RandomNumberGenerator.
 */

import type { F32, GraphSnapshot } from "@graphty/graph-format";

import { WebGpuGraphError } from "../errors.js";

/** The CPU port's LCG constants (layout/src/utils/random.ts): m = 2^35 - 31, a = 185852, c = 1. */
export const LCG_M = 34359738337;
/** The multiplier a of the port's LCG. */
export const LCG_A = 185852;
/** The increment c of the port's LCG. */
export const LCG_C = 1;

/**
 * The port's RandomNumberGenerator, bit for bit: `seed || Math.floor(Math.random() * 1000000)` (seed 0 / null =
 * unseeded, the quirk preserved), state = seed % m, next = (a * state + c) % m, value = state / m. The product
 * a * state stays below 2^53 (a < 2^18, state < 2^35), so every step is exact in f64, as it is in the port.
 */
export class Lcg {
    /** The seed actually used (a random one when unseeded). */
    readonly seed: number;

    private state: number;

    /**
     * Creates the generator; a seed of 0, -0, NaN or null means unseeded (the port's `seed || random` quirk).
     * @param seed - the seed, or null for a random seed
     */
    constructor(seed: number | null) {
        this.seed = seed === null || seed === 0 || Number.isNaN(seed) ? Math.floor(Math.random() * 1000000) : seed;
        this.state = this.seed % LCG_M;
    }

    /**
     * Next value in [0, 1).
     * @returns state / m after one LCG step
     */
    next(): number {
        this.state = (LCG_A * this.state + LCG_C) % LCG_M;
        return this.state / LCG_M;
    }
}

/**
 * The three center components of a CommonLayoutOptions.center (missing components are 0).
 * @param center - the caller's center, or null
 * @returns [x, y, z]; E_INVALID_ARGUMENT when a component is not finite
 */
function resolveCenter(center: ArrayLike<number> | null): [number, number, number] {
    const out: [number, number, number] = [0, 0, 0];
    if (center === null) {
        return out;
    }
    for (let axis = 0; axis < 3 && axis < center.length; axis++) {
        const v = center[axis];
        if (!Number.isFinite(v)) {
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", `center[${axis}] is not finite`, {
                argument: "center",
                value: v,
                expected: "finite components",
            });
        }
        out[axis] = v;
    }
    return out;
}

/**
 * Seeds the unseeded rows of the owner's stride-3 SCENE array in index order (spec 9.3 seedPositions, 7.14 `pos`,
 * 7.19 topology change): a row is unseeded when any of its `dim` components is not finite; every such component
 * draws one LCG value; when NO row is fully finite the draw is uniform in [-1, 1) layout units per axis (`range:
 * "fa2"`; `"fr"` is [0, 1)), otherwise uniform inside the [min, max] box of the finite components per axis,
 * converted to layout units (an axis with no finite value falls back to the range box); the value is written as
 * `v * scale + center[axis]`, which for scale 1 / center 0 is bit-identical to the port's `u * 2 - 1` (resp. `u`);
 * in 2D the third component of a seeded row is written as center[2]. Finite components are never changed and a
 * fully finite row is never touched. No random number is drawn when nothing needs seeding.
 * PLAN DECISION 12 (P3-T1): the box is taken over every finite COMPONENT (the port's rule), not over the fully
 * finite ROWS only; the two differ only when a partially finite row's finite axis lies outside the fully finite
 * rows' box (pinned by test/layouts/seed.test.ts). PLAN DECISION 13: a 2D row whose x and y are finite is seeded
 * whatever its z holds.
 * @param s - the snapshot (nodeCount rows)
 * @param positions - the owner's stride-3 scene-unit array, length 3 * nodeCount, modified in place
 * @param seed - the LCG seed (0 / null = unseeded, the port's quirk)
 * @param dim - 2 or 3
 * @param scale - the scene scale (> 0)
 * @param center - the scene center, or null for the origin
 * @param range - "fa2" for [-1, 1), "fr" for [0, 1)
 */
export function seedPositions(
    s: GraphSnapshot,
    positions: F32,
    seed: number | null,
    dim: 2 | 3,
    scale: number,
    center: ArrayLike<number> | null,
    range: "fa2" | "fr",
): void {
    const n = s.nodeCount;
    if (dim !== 2 && dim !== 3) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `dim must be 2 or 3, got ${String(dim)}`, {
            argument: "dim",
            value: dim,
            expected: "2 or 3",
        });
    }
    if (positions.length !== 3 * n) {
        throw new WebGpuGraphError(
            "E_INVALID_ARGUMENT",
            `positions has ${positions.length} entries, expected ${3 * n}`,
            {
                argument: "positions",
                value: positions.length,
                expected: 3 * n,
            },
        );
    }
    if (!Number.isFinite(scale) || scale <= 0) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `scale must be a finite number > 0, got ${scale}`, {
            argument: "scale",
            value: scale,
            expected: "a finite number > 0",
        });
    }
    const c = resolveCenter(center);

    // classify the rows and collect the per-axis box of the finite components (scene units)
    const lo = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const hi = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let finiteRows = 0;
    let unseededRows = 0;
    for (let i = 0; i < n; i++) {
        let finite = true;
        for (let axis = 0; axis < dim; axis++) {
            const v = positions[3 * i + axis];
            if (Number.isFinite(v)) {
                if (v < lo[axis]) {
                    lo[axis] = v;
                }
                if (v > hi[axis]) {
                    hi[axis] = v;
                }
            } else {
                finite = false;
            }
        }
        if (finite) {
            finiteRows++;
        } else {
            unseededRows++;
        }
    }
    if (unseededRows === 0) {
        return;
    }

    // the per-axis draw box in layout units
    const rangeLo = range === "fr" ? 0 : -1;
    const rangeHi = 1;
    const boxLo = [rangeLo, rangeLo, rangeLo];
    const boxHi = [rangeHi, rangeHi, rangeHi];
    if (finiteRows > 0) {
        for (let axis = 0; axis < dim; axis++) {
            if (Number.isFinite(lo[axis]) && Number.isFinite(hi[axis])) {
                boxLo[axis] = (lo[axis] - c[axis]) / scale;
                boxHi[axis] = (hi[axis] - c[axis]) / scale;
            }
        }
    }

    const rng = new Lcg(seed);
    for (let i = 0; i < n; i++) {
        let unseeded = false;
        for (let axis = 0; axis < dim; axis++) {
            if (!Number.isFinite(positions[3 * i + axis])) {
                unseeded = true;
            }
        }
        if (!unseeded) {
            continue;
        }
        for (let axis = 0; axis < dim; axis++) {
            const at = 3 * i + axis;
            if (!Number.isFinite(positions[at])) {
                const u = rng.next();
                const v = boxLo[axis] + u * (boxHi[axis] - boxLo[axis]);
                positions[at] = v * scale + c[axis];
            }
        }
        if (dim === 2) {
            positions[3 * i + 2] = c[2];
        }
    }
}
