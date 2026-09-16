/**
 * Numeric matchers shared by every kernel, primitive and layout test (contract 5.2; spec 11.3, 11.4): element-wise
 * closeness with the worst index in the failure message, TRUE bitwise equality of typed arrays (the bytes, so -0 / 0
 * and NaN payloads count), the floored per-node relative error of spec 11.4 over stride-3 vectors, and a plain
 * maximum relative error with an absolute floor.
 */

import { type TypedArrayData } from "@graphty/graph-format";

/**
 * Relative and absolute tolerance: |a - e| <= abs + rel * |e|. Exported by contract 5.2 for the kernel and layout
 * tests that name their tolerances; nothing at P1-T2 imports it by name.
 * @public
 */
export interface Tolerance {
    readonly rel: number;
    readonly abs: number;
}

/**
 * Asserts |actual[i] - expected[i]| <= abs + rel * |expected[i]| for every i (two NaNs at one index are equal), with
 * equal lengths; the failure message names the worst index and both values.
 * @param actual - the values under test
 * @param expected - the reference values
 * @param tolerance - the tolerance
 * @param label - a name for the failure message
 */
export function expectAllClose(
    actual: ArrayLike<number>,
    expected: ArrayLike<number>,
    tolerance: Tolerance,
    label?: string,
): void {
    const name = label ?? "values";
    expect(actual.length, `${name}: length`).toBe(expected.length);
    let worst = -1;
    let worstExcess = 0;
    for (let i = 0; i < expected.length; i++) {
        const a = actual[i];
        const e = expected[i];
        if (Number.isNaN(a) && Number.isNaN(e)) {
            continue;
        }
        const excess = Math.abs(a - e) - (tolerance.abs + tolerance.rel * Math.abs(e));
        const bad = Number.isNaN(excess) ? Number.POSITIVE_INFINITY : excess;
        if (bad > worstExcess) {
            worst = i;
            worstExcess = bad;
        }
    }
    const detail =
        worst < 0
            ? ""
            : ` worst index ${worst}: actual ${actual[worst]} expected ${expected[worst]} (rel ${tolerance.rel}, abs ${tolerance.abs})`;
    expect(worst, `${name}: not all close;${detail}`).toBe(-1);
}

/**
 * Asserts two typed arrays are bitwise identical: same constructor, same length, same bytes (so Object.is holds on
 * every element and NaN payloads agree).
 * @param a - the first array
 * @param b - the second array
 * @param label - a name for the failure message
 */
export function expectBitwiseEqual(a: TypedArrayData, b: TypedArrayData, label?: string): void {
    const name = label ?? "arrays";
    expect(a.constructor.name, `${name}: dtype`).toBe(b.constructor.name);
    expect(a.length, `${name}: length`).toBe(b.length);
    expect(a.byteLength, `${name}: byteLength`).toBe(b.byteLength);
    const ab = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const bb = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < ab.length; i++) {
        if (ab[i] !== bb[i]) {
            const index = Math.floor(i / a.BYTES_PER_ELEMENT);
            expect.fail(`${name}: bitwise difference at element ${index}: ${a[index]} vs ${b[index]} (byte ${i})`);
        }
    }
}

/**
 * The spec 11.4 per-node relative error over stride-3 vectors with the floored denominator
 * max(|F_cpu(i)|, floorFraction * max_j |F_cpu(j)|).
 * @param gpu - the GPU forces, 3 per node
 * @param cpu - the oracle forces, 3 per node
 * @param floorFraction - the fraction of the largest oracle norm used as the denominator floor
 * @returns the maximum error, its node index (-1 when there are no nodes), the RMS and the 99th percentile
 */
export function flooredRelError(
    gpu: ArrayLike<number>,
    cpu: ArrayLike<number>,
    floorFraction: number,
): { readonly max: number; readonly argmax: number; readonly rms: number; readonly p99: number } {
    expect(gpu.length, "flooredRelError: length").toBe(cpu.length);
    const n = Math.floor(cpu.length / 3);
    if (n === 0) {
        return { max: 0, argmax: -1, rms: 0, p99: 0 };
    }
    const norms = new Float64Array(n);
    let largest = 0;
    for (let i = 0; i < n; i++) {
        const x = cpu[3 * i];
        const y = cpu[3 * i + 1];
        const z = cpu[3 * i + 2];
        norms[i] = Math.sqrt(x * x + y * y + z * z);
        largest = Math.max(largest, norms[i]);
    }
    const floor = floorFraction * largest;
    const errors = new Float64Array(n);
    let max = 0;
    let argmax = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const dx = gpu[3 * i] - cpu[3 * i];
        const dy = gpu[3 * i + 1] - cpu[3 * i + 1];
        const dz = gpu[3 * i + 2] - cpu[3 * i + 2];
        const numerator = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const denominator = Math.max(norms[i], floor);
        let error = 0;
        if (denominator > 0) {
            error = numerator / denominator;
        } else if (numerator > 0) {
            error = Number.POSITIVE_INFINITY;
        }
        errors[i] = error;
        sumSq += error * error;
        if (error > max) {
            max = error;
            argmax = i;
        }
    }
    const sorted = Float64Array.from(errors).sort();
    const p99 = sorted[Math.max(0, Math.ceil(0.99 * n) - 1)];
    return { max, argmax, rms: Math.sqrt(sumSq / n), p99 };
}

/**
 * max_i |actual[i] - expected[i]| / max(|expected[i]|, absFloor).
 * @param actual - the values under test
 * @param expected - the reference values
 * @param absFloor - the denominator floor
 * @returns the maximum relative error (0 for empty inputs)
 */
export function maxRelError(actual: ArrayLike<number>, expected: ArrayLike<number>, absFloor: number): number {
    expect(actual.length, "maxRelError: length").toBe(expected.length);
    let max = 0;
    for (let i = 0; i < expected.length; i++) {
        const error = Math.abs(actual[i] - expected[i]) / Math.max(Math.abs(expected[i]), absFloor);
        if (error > max) {
            max = error;
        }
    }
    return max;
}
