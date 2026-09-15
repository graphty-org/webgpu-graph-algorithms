/**
 * Freeze benchmarks (design section 15.5 item 1): `fromEdgeArrays` with identity ids at
 * 10k nodes / 100k edges and 100k nodes / 1M edges, directed and undirected, from typed arrays;
 * the same graphs through the builder (`addAnonymousNodes` + `addEdges` + `freeze`), with
 * `duplicateEdges: "sum"`, with `weightDtype: "f64"`, and a compacting re-freeze after 10% edge
 * removals. Times are the freeze alone unless the name says otherwise; the section 15.4 targets are
 * 25-30 ms directed and 45-55 ms undirected for the 1M-edge graph.
 */

import { GraphBuilder } from "../src/builder/graph-builder.js";
import { fromEdgeArrays } from "../src/populate/from-edge-arrays.js";
import { type EdgeArrays, randomEdges } from "./datasets.js";
import { bench, type BenchResult, makeRandom } from "./harness.js";

/** The graph sizes measured, as [nodes, edges]. */
const SIZES: readonly (readonly [number, number])[] = [
    [10_000, 100_000],
    [100_000, 1_000_000],
];

/**
 * A builder loaded with the edges, ready to freeze.
 * @param edges - the input
 * @param directed - the orientation
 * @param options - extra builder options
 * @returns the builder
 */
function loadedBuilder(
    edges: EdgeArrays,
    directed: boolean,
    options: { duplicateEdges?: "keep" | "sum"; weightDtype?: "f32" | "f64" } = {},
): GraphBuilder {
    const builder = new GraphBuilder({
        directed,
        expectedNodes: edges.nodeCount,
        expectedEdges: edges.src.length,
        ...options,
    });
    builder.addAnonymousNodes(edges.nodeCount);
    builder.addEdges(edges.src, edges.dst, edges.weights);
    return builder;
}

/**
 * Run the freeze benchmarks.
 * @returns the results
 */
export function runFreezeBenchmarks(): BenchResult[] {
    const results: BenchResult[] = [];
    for (const [nodeCount, edgeCount] of SIZES) {
        const edges = randomEdges(nodeCount, edgeCount);
        const label = `${nodeCount / 1000}k/${edgeCount >= 1_000_000 ? `${edgeCount / 1_000_000}M` : `${edgeCount / 1000}k`}`;
        for (const directed of [true, false]) {
            const dir = directed ? "directed" : "undirected";
            results.push(
                bench(
                    "freeze",
                    `fromEdgeArrays ${label} ${dir} (typed arrays, identity ids)`,
                    {
                        setup: () => edges,
                        run: (input) =>
                            fromEdgeArrays({
                                directed,
                                nodeCount: input.nodeCount,
                                src: input.src,
                                dst: input.dst,
                                weights: input.weights,
                            }),
                    },
                    { items: edgeCount, unit: "edges" },
                ),
            );
            results.push(
                bench(
                    "freeze",
                    `builder push ${label} ${dir} (addAnonymousNodes + addEdges)`,
                    {
                        setup: () => edges,
                        run: (input) => loadedBuilder(input, directed),
                    },
                    { items: edgeCount, unit: "edges" },
                ),
            );
            results.push(
                bench(
                    "freeze",
                    `freeze ${label} ${dir} (builder already loaded)`,
                    {
                        setup: () => loadedBuilder(edges, directed),
                        run: (builder) => builder.freeze(),
                    },
                    { items: edgeCount, unit: "edges" },
                ),
            );
        }
        results.push(
            bench(
                "freeze",
                `freeze ${label} directed duplicateEdges: "sum"`,
                {
                    setup: () => loadedBuilder(edges, true, { duplicateEdges: "sum" }),
                    run: (builder) => builder.freeze(),
                },
                { items: edgeCount, unit: "edges" },
            ),
        );
        results.push(
            bench(
                "freeze",
                `freeze ${label} directed weightDtype: "f64"`,
                {
                    setup: () => loadedBuilder(edges, true, { weightDtype: "f64" }),
                    run: (builder) => builder.freeze(),
                },
                { items: edgeCount, unit: "edges" },
            ),
        );
        results.push(
            bench(
                "freeze",
                `re-freeze ${label} directed after removing 10% of the edges (compaction)`,
                {
                    setup: () => {
                        const builder = loadedBuilder(edges, true);
                        builder.freeze();
                        const random = makeRandom(99);
                        const removals = Math.floor(edgeCount / 10);
                        for (let i = 0; i < removals; i++) {
                            builder.removeEdge(Math.floor(random() * edgeCount));
                        }
                        return builder;
                    },
                    run: (builder) => builder.freeze(),
                },
                { items: edgeCount, unit: "edges" },
            ),
        );
    }
    return results;
}
