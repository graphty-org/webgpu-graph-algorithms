/**
 * The seeded reduce inputs (spec 11.3, 11.9 item 3) shared by test/helpers/reduce-check.ts, test/primitives/reduce.test.ts,
 * P1-T6's test/noise-floor.test.ts and P1-T7's test/browser/skeleton.test.ts. PURE: no imports, no node:*, no device --
 * a browser test imports it as-is. The reduce noise fixture `reduce-random1k-<adapterClass>.json` is DEFINED here: the
 * f32 SUM of `reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED)` at every prefix count of REDUCE_NOISE_COUNTS, in that
 * order, dtype "f32" (11 values). Every writer of that fixture imports these constants; none re-types them, or the
 * cross-adapter comparison of the committed files compares different inputs and fails on every later run.
 *
 * Type note: `ReduceDtype` is spelled out as a string union here rather than imported from src/primitives/reduce.ts
 * (a type-only import would be erased, but keeping the module import-free is the point); the union is identical to
 * the contract's 3.11 declaration and test/oracle/oracles.test.ts pins the two together.
 */

/** The element types of the reduce primitive (the same union as ReduceDtype of src/primitives/reduce.ts). */
type InputDtype = "f32" | "u32" | "vec4f";

/** The seed every case input derives from (plus the case's count). */
export const INPUT_SEED = 20260915;
/** The element count of the "random1k" reduce noise fixture. */
export const RANDOM1K_COUNT = 1000;
/** The seed of the "random1k" reduce noise fixture (deliberately not INPUT_SEED + 1000: the fixture is its own input). */
export const RANDOM1K_SEED = 1001;
/** The prefix counts whose f32 sums make up the "random1k" reduce noise fixture, in fixture order (11 values). */
export const REDUCE_NOISE_COUNTS: readonly number[] = Object.freeze([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]);

/**
 * Words per element of a dtype.
 * @param dtype - the element type
 * @returns 1, or 4 for vec4f
 */
export function lanesOf(dtype: InputDtype): number {
    return dtype === "vec4f" ? 4 : 1;
}

/**
 * A seeded LCG (Numerical Recipes constants); the state is a PRNG word, not an arc index or a byte offset.
 * @param seed - the seed
 * @returns the next-word function
 */
function lcg(seed: number): () => number {
    let x = seed >>> 0;
    return (): number => {
        x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
        return x;
    };
}

/**
 * The seeded input of a case: f32 / vec4f in [1, 2) (count x lanes floats), u32 in [1, 127] (a u32 sum of 16,777,217
 * elements stays below 2^32).
 * @param dtype - the element type
 * @param count - the element count
 * @param seed - the seed
 * @returns the values
 */
export function reduceInput(
    dtype: InputDtype,
    count: number,
    seed: number,
): Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer> {
    const next = lcg(seed);
    if (dtype === "u32") {
        const out = new Uint32Array(count);
        for (let i = 0; i < count; i++) {
            out[i] = 1 + (next() % 127);
        }
        return out;
    }
    const out = new Float32Array(count * lanesOf(dtype));
    for (let i = 0; i < out.length; i++) {
        out[i] = Math.fround(1 + next() / 4294967296);
    }
    return out;
}
