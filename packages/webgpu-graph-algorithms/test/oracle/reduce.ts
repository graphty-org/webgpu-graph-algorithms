/**
 * The CPU reference of `reduce` (spec 6 row 1 "Array.reduce in f64", contract 5.3): a sequential f64 fold over f32 /
 * u32 / vec4f inputs. `sum` returns the f64 total (the tests scale the tolerance by count: the GPU's f32 tree sum is
 * within count x 2^-24 relative); `min` / `max` are exact. The identity of the empty range is the GPU's (4.5
 * reduce body: 0, F32_MAX, -F32_MAX, U32_MAX), so a count-0 reduce has an exact expectation.
 */

import { type ReduceDtype, type ReduceOp } from "../../src/primitives/reduce.js";

/** The largest finite f32, the value of the prelude's `F32_MAX = 0x1.fffffep+127` (4.1). */
const F32_MAX = 3.4028234663852886e38;
/** The largest u32, the prelude's `U32_MAX` (4.1). */
const U32_MAX = 4294967295;

/**
 * The identity element the GPU's `identity_f` / `identity_u` return (4.5 reduce body).
 * @param op - the operator
 * @param dtype - the element type (vec4f lanes are f32)
 * @returns 0 for sum, F32_MAX / U32_MAX for min, -F32_MAX / 0 for max
 */
export function reduceIdentity(op: ReduceOp, dtype: ReduceDtype): number {
    if (op === "sum") {
        return 0;
    }
    if (dtype === "u32") {
        return op === "min" ? U32_MAX : 0;
    }
    return op === "min" ? F32_MAX : -F32_MAX;
}

/**
 * One f64 combining step.
 * @param acc - the accumulator
 * @param value - the next element
 * @param op - the operator
 * @returns the combined value
 */
function combine(acc: number, value: number, op: ReduceOp): number {
    if (op === "min") {
        return Math.min(acc, value);
    }
    if (op === "max") {
        return Math.max(acc, value);
    }
    return acc + value;
}

/**
 * f64 sequential reduction of f32 / u32 / vec4f inputs; sum returns the f64 total (tests scale the tolerance by
 * count); min / max exact.
 * @param values - the elements (vec4f: four words per element, lane-major)
 * @param op - the operator
 * @param dtype - the element type
 * @returns the scalar result, or the four lane results for vec4f
 */
export function reduceOracle(
    values: ArrayLike<number>,
    op: ReduceOp,
    dtype: ReduceDtype,
): number | readonly [number, number, number, number] {
    const identity = reduceIdentity(op, dtype);
    if (dtype === "vec4f") {
        const n = Math.floor(values.length / 4);
        let x = identity;
        let y = identity;
        let z = identity;
        let w = identity;
        for (let i = 0; i < n; i++) {
            x = combine(x, values[4 * i], op);
            y = combine(y, values[4 * i + 1], op);
            z = combine(z, values[4 * i + 2], op);
            w = combine(w, values[4 * i + 3], op);
        }
        return [x, y, z, w];
    }
    let acc = identity;
    for (let i = 0; i < values.length; i++) {
        acc = combine(acc, values[i], op);
    }
    return acc;
}
