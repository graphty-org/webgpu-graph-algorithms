/**
 * Synthetic inputs of the benchmarks (spec 11.7; contract 6.2): seeded edge arrays in the `fromEdgeArrays` input shape --
 * graph-format's G(n, m) generator (self-loops and parallels allowed, integer weights 1..10), an R-MAT-like hub graph, a
 * grid, and Zachary's karate club -- plus the design-15.3 tiers and a snapshot builder. Every generator is seeded, so the
 * inputs are identical across runs and hosts.
 */

import { type F32, fromEdgeArrays, type GraphSnapshot, type U32 } from "@graphty/graph-format";

import { makeRandom } from "./harness.js";

/**
 * A graph as parallel typed arrays, the `fromEdgeArrays` input shape.
 * @public consumed by P3-T7's benchmarks/layout-exact.bench.ts and layout-run.ts (contract 6.3); referenced only
 * through the generators' return types inside this file at P1
 */
export interface EdgeArrays {
    /** Node count. */
    readonly nodeCount: number;
    /** Source node index of every edge. */
    readonly src: U32;
    /** Target node index of every edge. */
    readonly dst: U32;
    /** Weights; integers 1..10 from the random generators, 1 for the grid and karate. */
    readonly weights: F32;
}

/**
 * G(n, m) with self-loops and parallels, weights 1..10 (graph-format's).
 * @param nodeCount - the node count
 * @param edgeCount - the edge count
 * @param seed - the generator seed (default 12345, graph-format's)
 * @returns the edge arrays
 */
export function randomEdges(nodeCount: number, edgeCount: number, seed?: number | undefined): EdgeArrays {
    const random = makeRandom(seed ?? 12345);
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
 * R-MAT-like hub graph (0.57 / 0.19 / 0.19 / 0.05), no self-loops: 2^scale nodes, edgeFactor edges per node, each edge
 * placed by `scale` quadrant draws (the Chakrabarti / Zhan / Faloutsos recursion), a self-loop moved to (v + 1) % n as the
 * gpu-upload.test.ts generator does. Weights 1..10.
 * @public consumed by the P3-T7 layout-exact group and layout-run.ts (contract 6.3); test/benchmarks.test.ts pins it
 * @param scale - log2 of the node count
 * @param edgeFactor - edges per node
 * @param seed - the generator seed (default 12345)
 * @returns the edge arrays
 */
export function rmatEdges(scale: number, edgeFactor: number, seed?: number | undefined): EdgeArrays {
    const nodeCount = 2 ** scale;
    const edgeCount = nodeCount * edgeFactor;
    const random = makeRandom(seed ?? 12345);
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    const weights = new Float32Array(edgeCount);
    for (let e = 0; e < edgeCount; e++) {
        let u = 0;
        let v = 0;
        for (let level = 0; level < scale; level++) {
            const r = random();
            let bitU = 0;
            let bitV = 0;
            if (r < 0.57) {
                bitU = 0;
                bitV = 0;
            } else if (r < 0.76) {
                bitU = 0;
                bitV = 1;
            } else if (r < 0.95) {
                bitU = 1;
                bitV = 0;
            } else {
                bitU = 1;
                bitV = 1;
            }
            u = u * 2 + bitU;
            v = v * 2 + bitV;
        }
        if (u === v) {
            v = (v + 1) % nodeCount;
        }
        src[e] = u;
        dst[e] = v;
        weights[e] = 1 + Math.floor(random() * 10);
    }
    return { nodeCount, src, dst, weights };
}

/**
 * A w x h grid (4-neighbour), row-major node indices, in graph-format's edge order (right, then down, per node).
 * @public consumed by the P3-T7 layout-exact group (contract 6.3); test/benchmarks.test.ts pins it
 * @param w - columns
 * @param h - rows
 * @returns the edge arrays (unit weights)
 */
export function gridEdges(w: number, h: number): EdgeArrays {
    const pairs: number[] = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const u = y * w + x;
            if (x + 1 < w) {
                pairs.push(u, u + 1);
            }
            if (y + 1 < h) {
                pairs.push(u, u + w);
            }
        }
    }
    const edgeCount = pairs.length / 2;
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    const weights = new Float32Array(edgeCount).fill(1);
    for (let e = 0; e < edgeCount; e++) {
        src[e] = pairs[2 * e];
        dst[e] = pairs[2 * e + 1];
    }
    return { nodeCount: w * h, src, dst, weights };
}

/** Zachary's karate club, 78 edges over nodes 0..33 (graph-format's test/helpers/parts.ts KARATE_EDGES). */
const KARATE_PAIRS: readonly (readonly [number, number])[] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [0, 17],
    [0, 19],
    [0, 21],
    [0, 31],
    [1, 2],
    [1, 3],
    [1, 7],
    [1, 13],
    [1, 17],
    [1, 19],
    [1, 21],
    [1, 30],
    [2, 3],
    [2, 7],
    [2, 8],
    [2, 9],
    [2, 13],
    [2, 27],
    [2, 28],
    [2, 32],
    [3, 7],
    [3, 12],
    [3, 13],
    [4, 6],
    [4, 10],
    [5, 6],
    [5, 10],
    [5, 16],
    [6, 16],
    [8, 30],
    [8, 32],
    [8, 33],
    [9, 33],
    [13, 33],
    [14, 32],
    [14, 33],
    [15, 32],
    [15, 33],
    [18, 32],
    [18, 33],
    [19, 33],
    [20, 32],
    [20, 33],
    [22, 32],
    [22, 33],
    [23, 25],
    [23, 27],
    [23, 29],
    [23, 32],
    [23, 33],
    [24, 25],
    [24, 27],
    [24, 31],
    [25, 31],
    [26, 29],
    [26, 33],
    [27, 33],
    [28, 31],
    [28, 33],
    [29, 32],
    [29, 33],
    [30, 32],
    [30, 33],
    [31, 32],
    [31, 33],
    [32, 33],
];

/**
 * Zachary's karate club as EdgeArrays (unit weights).
 * @public consumed by the P3-T7 layout-exact group (contract 6.3); test/benchmarks.test.ts pins it
 */
export const KARATE_EDGES: EdgeArrays = {
    nodeCount: 34,
    src: Uint32Array.from(KARATE_PAIRS, (pair) => pair[0]),
    dst: Uint32Array.from(KARATE_PAIRS, (pair) => pair[1]),
    weights: new Float32Array(KARATE_PAIRS.length).fill(1),
};

/** The design-15.3 tiers as (nodes, edges) pairs: 10k/100k, 100k/1M, 1M/10M. */
export const TIERS: readonly { readonly name: string; readonly nodes: number; readonly edges: number }[] = [
    { name: "10k/100k", nodes: 10_000, edges: 100_000 },
    { name: "100k/1M", nodes: 100_000, edges: 1_000_000 },
    { name: "1M/10M", nodes: 1_000_000, edges: 10_000_000 },
];

/**
 * An undirected weighted fromEdgeArrays snapshot of an EdgeArrays.
 * @param edges - the edge arrays
 * @param options - `directed` (default false) and the snapshot label
 * @returns the snapshot (arena path: fromEdgeArrays freezes with `arena: true`)
 */
export function snapshotOf(
    edges: EdgeArrays,
    options?: { readonly directed?: boolean | undefined; readonly label?: string | undefined } | undefined,
): GraphSnapshot {
    return fromEdgeArrays(
        {
            directed: options?.directed ?? false,
            nodeCount: edges.nodeCount,
            src: edges.src,
            dst: edges.dst,
            weights: edges.weights,
        },
        options?.label === undefined ? {} : { label: options.label },
    );
}
