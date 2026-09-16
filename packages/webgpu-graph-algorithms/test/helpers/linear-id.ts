/**
 * The constants of the 17M-item linear_id test (spec 5.2, 11.5; contract 5.2), shared by
 * test/kernel/linear-id.test.ts (Node: lavapipe / NVIDIA through Dawn) and test/browser/skeleton.test.ts
 * (Chromium: SwiftShader / NVIDIA, P1-T7), so the two legs of "the 17M-item map bitwise across adapters" cannot
 * drift. Imports only src/constants.ts -- no node:* module -- so the browser bundle can load it. The checksum
 * arithmetic uses `%`, never a bitwise operator (house rule I3).
 */

import { MAX_1D_ITEMS } from "../../src/constants.js";

/** 16,776,961 = MAX_1D_ITEMS + 1: the first item count that needs the 2D dispatch. */
export const LINEAR_ID_ITEMS: number = MAX_1D_ITEMS + 1;

/** The fill `value` of the test (a fixed odd constant). */
export const LINEAR_ID_VALUE: number = 1_000_003;

/**
 * Sampled indices across the 1D / 2D boundary (0, 1, 255, 256, 65_535 x 256 - 1, 65_535 x 256,
 * LINEAR_ID_ITEMS - 1). The last two coincide (LINEAR_ID_ITEMS - 1 = 65_535 x 256): that index is the single item
 * of the second workgroup row, lane 0 of workgroup (0, 1).
 */
export const LINEAR_ID_SAMPLES: readonly number[] = Object.freeze([
    0,
    1,
    255,
    256,
    MAX_1D_ITEMS - 1,
    MAX_1D_ITEMS,
    LINEAR_ID_ITEMS - 1,
]);

/**
 * The u32 checksum (sum mod 2^32 of every word) the fill must produce, recorded once and pinned (bitwise across
 * adapters): (N x V + N (N - 1) / 2) mod 2^32 for N = 16,776,961 and V = 1,000,003, re-derived in BigInt by
 * test/kernel/linear-id.test.ts.
 */
export const LINEAR_ID_CHECKSUM: number = 877_493_955;

/** 2^32 as a decimal, so the reduction below is plain modular arithmetic. */
const CHECKSUM_MODULUS = 4_294_967_296;

/**
 * The checksum of a result array (the same arithmetic the pin was recorded with): every word added into a running
 * sum reduced modulo 2^32 after each step, so the intermediate never exceeds 2^33 and stays exact in a double.
 * @param words - the words read back from the device
 * @returns the sum of every word modulo 2^32
 */
export function linearIdChecksum(words: Uint32Array): number {
    let sum = 0;
    for (let k = 0; k < words.length; k++) {
        sum = (sum + words[k]) % CHECKSUM_MODULUS;
    }
    return sum;
}
