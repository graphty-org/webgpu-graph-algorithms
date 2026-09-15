/**
 * I1-I18 as executable checks (design section 16.1) over the hand-written fixtures of that section
 * and over generated graphs. The checks themselves live in test/helpers/invariants.ts so the module
 * tests reuse them; this file is the fixture runner plus the worked layout example of design
 * section 10.3.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { allocateCoreArrays } from "../src/builder/arena.js";
import { INVALID_INDEX } from "../src/constants.js";
import { buildCore } from "../src/snapshot/derived.js";
import { equalsTopology } from "../src/snapshot/graph-snapshot.js";
import { assertInvariants, assertMatchesSpec } from "./helpers/invariants.js";
import { type EdgeSpec, type GraphSpec, gridEdges, KARATE_EDGES, makeSnapshot, naiveCsr } from "./helpers/parts.js";

/** The fixtures of design section 16.1 that need no builder. */
const FIXTURES: readonly [string, GraphSpec][] = [
    ["empty directed", { directed: true, nodeCount: 0, edges: [] }],
    ["empty undirected", { directed: false, nodeCount: 0, edges: [] }],
    ["one node directed", { directed: true, nodeCount: 1, edges: [] }],
    ["one node undirected", { directed: false, nodeCount: 1, edges: [] }],
    ["one self-loop directed", { directed: true, edges: [[0, 0]] }],
    ["one self-loop undirected", { directed: false, edges: [[0, 0]] }],
    [
        "undirected, every edge a self-loop (arcCount === edgeCount, identity perm, flag false)",
        {
            directed: false,
            nodeCount: 3,
            edges: [
                [0, 0],
                [1, 1],
                [2, 2, 2.5],
            ],
        },
    ],
    [
        "parallel edges directed",
        {
            directed: true,
            edges: [
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1, 3],
            ],
        },
    ],
    [
        "parallel edges undirected",
        {
            directed: false,
            edges: [
                [0, 1],
                [1, 0],
                [0, 1],
                [2, 2],
                [2, 2],
            ],
        },
    ],
    [
        "directed sorted input (identity)",
        {
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 0],
            ],
        },
    ],
    [
        "directed unsorted input",
        {
            directed: true,
            edges: [
                [2, 0],
                [0, 2],
                [1, 2],
                [0, 1],
            ],
        },
    ],
    [
        "directed weighted with zero, negative and infinite weights",
        {
            directed: true,
            edges: [
                [0, 1, 0],
                [1, 2, -1],
                [2, 0, Infinity],
                [0, 2, -Infinity],
            ],
        },
    ],
    [
        "expanded mixed graph with a pair column",
        {
            directed: true,
            edges: [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 3],
                [3, 2],
            ],
            edgeColumns: {
                "graphty.directed": {
                    data: new Uint32Array([0b00100]),
                    decl: { dtype: "bool", role: "directed", nullable: false },
                },
                "graphty.pair": {
                    values: [1, 0, undefined, 4, 3],
                    decl: { dtype: "u32", role: "pair", refersTo: "edge", nullable: true },
                },
            },
        },
    ],
    ["karate (undirected)", { directed: false, edges: KARATE_EDGES }],
    [
        "karate (directed, string ids)",
        { directed: true, edges: KARATE_EDGES, ids: Array.from({ length: 34 }, (_, i) => `n${i}`) },
    ],
    ["grid 4x3 undirected", { directed: false, edges: gridEdges(4, 3) }],
    ["grid 3x3 directed", { directed: true, edges: gridEdges(3, 3) }],
    [
        "isolated nodes with string ids",
        { directed: false, nodeCount: 5, ids: ["a", "b", "c", "d", "e"], edges: [[1, 3]] },
    ],
    [
        "numeric ids with holes",
        {
            directed: true,
            ids: [10, 20, 30, 1.5],
            edges: [
                [3, 0],
                [0, 1],
                [1, 2],
            ],
        },
    ],
    [
        "no arena",
        {
            directed: false,
            edges: [
                [0, 1],
                [1, 2],
            ],
            arena: false,
        },
    ],
    [
        "checksummed",
        {
            directed: true,
            edges: [
                [1, 0],
                [0, 1, 2],
            ],
            checksum: true,
        },
    ],
];

describe("invariants I1-I18 over the fixtures", () => {
    for (const [name, spec] of FIXTURES) {
        it(name, () => {
            const snapshot = makeSnapshot(spec);
            assertMatchesSpec(snapshot, spec);
        });
    }

    it("the expanded mixed graph's pair column is rewritten through the identity", () => {
        const spec = FIXTURES.find(([name]) => name.startsWith("expanded"))?.[1];
        expect(spec).toBeDefined();
        const snapshot = makeSnapshot(spec as GraphSpec);
        const pair = snapshot.edges.requireTyped("graphty.pair", "u32");
        expect(pair.isSet(2)).toBe(false);
        expect(Array.from(pair.data)).toEqual([1, 0, INVALID_INDEX, 4, 3]);
        const directed = snapshot.edges.requireTyped("graphty.directed", "bool");
        expect(directed.value(2)).toBe(true);
        expect(directed.value(0)).toBe(false);
    });

    it("the all-self-loop undirected fixture has arcCount === edgeCount and a materialised identity that is NOT flagged", () => {
        const snapshot = makeSnapshot({
            directed: false,
            nodeCount: 3,
            edges: [
                [0, 0],
                [1, 1],
                [2, 2],
            ],
        });
        expect(snapshot.arcCount).toBe(snapshot.edgeCount);
        expect(Array.from(snapshot.arcToEdge)).toEqual([0, 1, 2]);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
        expect(snapshot.arena?.segments.arcToEdge).not.toBeNull();
    });
});

/** A random edge list over n nodes. */
function edgesArb(
    maxNodes: number,
    maxEdges: number,
    weighted: boolean,
): fc.Arbitrary<{ nodeCount: number; edges: EdgeSpec[] }> {
    return fc.integer({ min: 0, max: maxNodes }).chain((nodeCount) => {
        if (nodeCount === 0) {
            return fc.constant({ nodeCount, edges: [] as EdgeSpec[] });
        }
        const endpoint = fc.integer({ min: 0, max: nodeCount - 1 });
        const weight = fc.oneof(
            fc.constant(1),
            fc.constant(0),
            fc.constant(-2),
            fc.constant(Infinity),
            fc.float({ noNaN: true, noDefaultInfinity: true }),
        );
        const edge: fc.Arbitrary<EdgeSpec> = weighted
            ? fc.tuple(endpoint, endpoint, weight).map(([u, v, w]) => [u, v, w] as const)
            : fc.tuple(endpoint, endpoint).map(([u, v]) => [u, v] as const);
        return fc.array(edge, { maxLength: maxEdges }).map((edges) => ({ nodeCount, edges }));
    });
}

describe("invariants over generated graphs (P1, P4, P5, P8)", () => {
    for (const directed of [true, false]) {
        for (const weighted of [false, true]) {
            it(`hold for random ${directed ? "directed" : "undirected"} ${weighted ? "weighted" : "unweighted"} multigraphs`, () => {
                fc.assert(
                    fc.property(edgesArb(12, 40, weighted), ({ nodeCount, edges }) => {
                        const spec: GraphSpec = { directed, nodeCount, edges };
                        const snapshot = makeSnapshot(spec);
                        assertMatchesSpec(snapshot, spec);
                        // P5: edgeCount equals the number of arcs holding the declared orientation
                        let declared = 0;
                        for (let a = 0; a < snapshot.arcCount; a++) {
                            if (snapshot.edgeToArc[snapshot.arcToEdge[a]] === a) {
                                declared++;
                            }
                        }
                        expect(declared).toBe(snapshot.edgeCount);
                    }),
                    { numRuns: 60 },
                );
            });
        }
    }

    it("the counting-sort core builder agrees with the naive comparator construction", () => {
        fc.assert(
            fc.property(fc.boolean(), edgesArb(10, 30, true), (directed, { nodeCount, edges }) => {
                const spec: GraphSpec = { directed, nodeCount, edges };
                const naive = naiveCsr(spec);
                const built = buildCore({
                    directed,
                    nodeCount,
                    edgeCount: naive.edgeCount,
                    src: naive.src,
                    dst: naive.dst,
                    weights: naive.edgeWeights,
                });
                expect(Array.from(built.rowPtr)).toEqual(Array.from(naive.rowPtr));
                expect(Array.from(built.colIdx)).toEqual(Array.from(naive.colIdx));
                expect(built.selfLoopCount).toBe(naive.selfLoopCount);
                if (built.arcToEdge === null) {
                    expect(built.flags.arcToEdgeIsIdentity).toBe(true);
                    expect(Array.from(naive.arcToEdge)).toEqual(Array.from(naive.arcToEdge, (_, i) => i));
                } else {
                    expect(Array.from(built.arcToEdge)).toEqual(Array.from(naive.arcToEdge));
                    expect(Array.from(built.edgeToArc as Uint32Array)).toEqual(Array.from(naive.edgeToArc));
                }
                if (naive.weights !== null) {
                    expect(Array.from(built.weights as Float32Array)).toEqual(Array.from(naive.weights));
                }
                const snapshot = makeSnapshot(spec);
                const fromBuilt = makeSnapshot({ ...spec });
                expect(equalsTopology(snapshot, fromBuilt)).toBe(true);
            }),
            { numRuns: 60 },
        );
    });
});

describe("worked examples of the design", () => {
    it("lays out the benchmark undirected weighted arena as in section 10.3", () => {
        const core = allocateCoreArrays(
            { nodeCount: 100_000, arcCount: 2_000_000, edgeCount: 1_000_000, weighted: true, identity: false },
            true,
        );
        const { arena } = core;
        expect(arena).not.toBeNull();
        if (arena === null) {
            return;
        }
        expect(arena.segments.rowPtr).toEqual({ byteOffset: 0, byteLength: 400_004 });
        expect(arena.segments.colIdx).toEqual({ byteOffset: 400_128, byteLength: 8_000_000 });
        expect(arena.segments.weights).toEqual({ byteOffset: 8_400_128, byteLength: 8_000_000 });
        expect(arena.segments.arcToEdge).toEqual({ byteOffset: 16_400_128, byteLength: 8_000_000 });
        expect(arena.segments.edgeToArc).toEqual({ byteOffset: 24_400_128, byteLength: 4_000_000 });
        expect(arena.byteLength).toBe(28_400_128);
        expect(arena.hotByteLength).toBe(16_400_128);
        expect(core.rowPtr.buffer).toBe(arena.buffer);
        expect(core.edgeToArc?.byteOffset).toBe(24_400_128);
    });

    it("lays out the directed weighted arena with and without materialised permutations", () => {
        const materialised = allocateCoreArrays(
            { nodeCount: 100_000, arcCount: 1_000_000, edgeCount: 1_000_000, weighted: true, identity: false },
            true,
        ).arena;
        expect(materialised?.segments.arcToEdge?.byteOffset).toBe(8_400_128);
        expect(materialised?.segments.edgeToArc?.byteOffset).toBe(12_400_128);
        expect(materialised?.byteLength).toBe(16_400_128);
        expect(materialised?.hotByteLength).toBe(8_400_128);
        const identity = allocateCoreArrays(
            { nodeCount: 100_000, arcCount: 1_000_000, edgeCount: 1_000_000, weighted: true, identity: true },
            true,
        ).arena;
        expect(identity?.segments.arcToEdge).toBeNull();
        expect(identity?.segments.edgeToArc).toBeNull();
        expect(identity?.byteLength).toBe(8_400_128);
        expect(identity?.hotByteLength).toBe(8_400_128);
    });

    it("reports the degree conventions of section 3.4 for a self-loop", () => {
        const directed = makeSnapshot({
            directed: true,
            edges: [
                [0, 0],
                [0, 1],
            ],
        });
        expect(Array.from(directed.outDegree())).toEqual([2, 0]);
        expect(Array.from(directed.inDegree())).toEqual([1, 1]);
        expect(Array.from(directed.degree())).toEqual([3, 1]);
        const undirected = makeSnapshot({
            directed: false,
            edges: [
                [0, 0],
                [0, 1],
            ],
        });
        expect(Array.from(undirected.outDegree())).toEqual([2, 1]);
        expect(undirected.inDegree()).toBe(undirected.outDegree());
        expect(Array.from(undirected.degree())).toEqual([3, 1]);
        assertInvariants(directed);
        assertInvariants(undirected);
    });
});
