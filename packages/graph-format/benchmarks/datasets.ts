/**
 * Synthetic inputs of the benchmarks (design section 15.5): uniformly random edge arrays with
 * integer weights (the shape of the section 15.4 freeze target), and string / numeric id lists for
 * the id-map benchmarks. Every generator is seeded, so the inputs are identical across runs and
 * hosts.
 */

import { makeRandom } from "./harness.js";

/** A graph as parallel typed arrays, the `fromEdgeArrays` input shape. */
export interface EdgeArrays {
    /** Node count. */
    readonly nodeCount: number;
    /** Source node index of every edge. */
    readonly src: Uint32Array<ArrayBuffer>;
    /** Target node index of every edge. */
    readonly dst: Uint32Array<ArrayBuffer>;
    /** Integer weights in 1..10. */
    readonly weights: Float32Array<ArrayBuffer>;
}

/**
 * Uniformly random edges over `nodeCount` nodes (a G(n, m) graph with self-loops and parallel edges
 * allowed).
 * @param nodeCount - the node count
 * @param edgeCount - the edge count
 * @param seed - the generator seed
 * @returns the edge arrays
 */
export function randomEdges(nodeCount: number, edgeCount: number, seed = 12345): EdgeArrays {
    const random = makeRandom(seed);
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    const weights = new Float32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        src[e] = Math.floor(random() * nodeCount);
        dst[e] = Math.floor(random() * nodeCount);
        weights[e] = 1 + Math.floor(random() * 10);
    }
    return { nodeCount, src, dst, weights };
}

/**
 * Distinct string ids of the form "node-<i>" in a random order.
 * @param count - how many ids
 * @param seed - the generator seed
 * @returns the ids
 */
export function stringIds(count: number, seed = 777): string[] {
    const ids = new Array<string>(count);
    for (let i = 0; i < count; i++) {
        ids[i] = `node-${i}`;
    }
    shuffle(ids, seed);
    return ids;
}

/**
 * Distinct sparse numeric ids (every id is a multiple of 7 plus a random offset below 7, so the map
 * is "numeric", not "dense") in a random order.
 * @param count - how many ids
 * @param seed - the generator seed
 * @returns the ids
 */
export function sparseNumericIds(count: number, seed = 778): number[] {
    const random = makeRandom(seed);
    const ids = new Array<number>(count);
    for (let i = 0; i < count; i++) {
        ids[i] = i * 7 + Math.floor(random() * 7);
    }
    shuffle(ids, seed + 1);
    return ids;
}

/**
 * Shuffle an array in place (Fisher-Yates) with a seeded generator.
 * @param items - the array
 * @param seed - the generator seed
 */
function shuffle<T>(items: T[], seed: number): void {
    const random = makeRandom(seed);
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}
