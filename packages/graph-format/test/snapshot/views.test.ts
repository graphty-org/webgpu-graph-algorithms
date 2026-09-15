import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { peekView } from "../../src/snapshot/graph-snapshot.js";
import {
    computeDegreeOrder,
    DEGREE_TIER_HIGH,
    DEGREE_TIER_MID,
    expandEdges,
    foldArcs,
    identityPermutation,
    ReverseAdjacency,
    viewByteLength,
} from "../../src/snapshot/views.js";
import { type GraphSnapshot } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { type GraphSpec, KARATE_EDGES, makeSnapshot, weightOf } from "../helpers/parts.js";

function expectError(fn: () => unknown, code: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe(code);
    return error;
}

/** Brute-force per-node quantities from the spec. */
function brute(spec: GraphSpec, s: GraphSnapshot) {
    const n = s.nodeCount;
    const out = new Array<number>(n).fill(0);
    const inn = new Array<number>(n).fill(0);
    const loops = new Array<number>(n).fill(0);
    const wOut = new Array<number>(n).fill(0);
    const wIn = new Array<number>(n).fill(0);
    const wLoop = new Array<number>(n).fill(0);
    let total = 0;
    for (const edge of spec.edges) {
        const [u, v] = edge;
        const w = s.weights === null ? 1 : Math.fround(weightOf(edge));
        total += w;
        if (u === v) {
            loops[u]++;
            wLoop[u] += w;
            out[u]++;
            wOut[u] += w;
            inn[u]++;
            wIn[u] += w;
            continue;
        }
        out[u]++;
        wOut[u] += w;
        inn[v]++;
        wIn[v] += w;
        if (!spec.directed) {
            out[v]++;
            wOut[v] += w;
            inn[u]++;
            wIn[u] += w;
        }
    }
    return { out, inn, loops, wOut, wIn, wLoop, total };
}

const SPECS: readonly [string, GraphSpec][] = [
    [
        "directed weighted multigraph with loops",
        {
            directed: true,
            edges: [
                [0, 1, 2],
                [1, 0, 3],
                [0, 1, 0.5],
                [2, 2, 4],
                [1, 2, -1],
                [3, 0, 1],
            ],
        },
    ],
    [
        "undirected weighted multigraph with loops",
        {
            directed: false,
            edges: [
                [0, 1, 2],
                [1, 0, 3],
                [2, 2, 4],
                [2, 2, 0.5],
                [1, 2, -1],
                [3, 0, 1],
            ],
        },
    ],
    [
        "directed unweighted identity",
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
        "undirected unweighted",
        {
            directed: false,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 0],
            ],
        },
    ],
    ["karate", { directed: false, edges: KARATE_EDGES }],
    ["empty", { directed: false, nodeCount: 0, edges: [] }],
    ["single node", { directed: true, nodeCount: 1, edges: [] }],
];

describe("degree views agree with brute force (P8)", () => {
    for (const [name, spec] of SPECS) {
        it(name, () => {
            const s = makeSnapshot(spec);
            const b = brute(spec, s);
            expect(Array.from(s.outDegree())).toEqual(b.out);
            expect(Array.from(s.inDegree())).toEqual(b.inn);
            expect(Array.from(s.selfLoopsPerNode())).toEqual(b.loops);
            const degree = spec.directed ? b.out.map((d, u) => d + b.inn[u]) : b.out.map((d, u) => d + b.loops[u]);
            expect(Array.from(s.degree())).toEqual(degree);
            const close = (actual: Float64Array, expected: number[]) => {
                expect(actual.length).toBe(expected.length);
                for (let i = 0; i < expected.length; i++) {
                    expect(actual[i]).toBeCloseTo(expected[i], 6);
                }
            };
            close(s.weightedOutDegree(), b.wOut);
            close(s.weightedInDegree(), b.wIn);
            close(s.selfLoopWeight(), b.wLoop);
            const wDegree = spec.directed ? b.wOut.map((d, u) => d + b.wIn[u]) : b.wOut.map((d, u) => d + b.wLoop[u]);
            close(s.weightedDegree(), wDegree);
            expect(s.totalWeight()).toBeCloseTo(b.total, 6);
            if (!spec.directed) {
                let sum = 0;
                for (const d of s.weightedDegree()) {
                    sum += d;
                }
                expect(sum).toBeCloseTo(2 * s.totalWeight(), 6);
            }
            expect(s.weightedOutDegree()).toBeInstanceOf(Float64Array);
            assertInvariants(s);
        });
    }

    it("totalWeight is edgeCount when unweighted and reads every edge once when the permutation is the identity", () => {
        expect(
            makeSnapshot({
                directed: false,
                edges: [
                    [0, 1],
                    [1, 2],
                ],
            }).totalWeight(),
        ).toBe(2);
        const identity = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 2],
                [0, 2, 3],
                [1, 2, 4],
            ],
        });
        expect(identity.flags.arcToEdgeIsIdentity).toBe(true);
        expect(identity.totalWeight()).toBe(9);
        expect(peekView(identity, "edgeList")).toBeNull();
    });

    it("weightedOutDegree widens outDegree when unweighted and may be zero for a node with arcs", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 0],
                [0, 2, 0],
            ],
        });
        expect(Array.from(s.weightedOutDegree())).toEqual([0, 0, 0]);
        expect(Array.from(s.outDegree())).toEqual([2, 0, 0]);
        const un = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
        });
        expect(Array.from(un.weightedOutDegree())).toEqual([2, 0, 0]);
        expect(Array.from(un.weightedInDegree())).toEqual([0, 1, 1]);
    });
});

describe("sharing and aliasing rules (I17, section 7.2)", () => {
    it("every view returns the same cached object; .slice() gives a private copy", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 2, 3],
            ],
        });
        expect(s.outDegree()).toBe(s.outDegree());
        expect(s.inDegree()).toBe(s.outDegree());
        expect(s.weightedInDegree()).toBe(s.weightedOutDegree());
        expect(s.reverse()).toBe(s.reverse());
        expect(s.coo()).toBe(s.coo());
        expect(s.edgeList()).toBe(s.edgeList());
        expect(s.mate()).toBe(s.mate());
        expect(s.degreeOrder()).toBe(s.degreeOrder({ of: "reverse" }));
        const copy = s.outDegree().slice();
        copy[0] = 99;
        expect(s.outDegree()[0]).toBe(1);
    });

    it("undirected reverse() is the forward arrays with a lazy identity fwdArc and an aliased arcToEdge", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 2, 3],
            ],
        });
        const r = s.reverse();
        expect(r.rowPtr).toBe(s.rowPtr);
        expect(r.colIdx).toBe(s.colIdx);
        expect(r.weights).toBe(s.weights);
        expect(r.arcToEdge).toBe(s.arcToEdge);
        expect(r.directed).toBe(false);
        expect(r.nodeCount).toBe(3);
        expect(r.arcCount).toBe(4);
        expect(viewByteLength(s, "reverse", r)).toBe(0);
        expect(Array.from(r.fwdArc)).toEqual([0, 1, 2, 3]);
        expect(r.fwdArc).toBe(r.fwdArc);
        expect(viewByteLength(s, "reverse", r)).toBe(16);
    });

    it("directed reverse() is a counting sort with gathered weights and a lazily gathered arcToEdge", () => {
        const spec: GraphSpec = {
            directed: true,
            edges: [
                [2, 0, 5],
                [0, 2, 6],
                [1, 2, 7],
                [0, 1, 8],
                [0, 2, 9],
            ],
        };
        const s = makeSnapshot(spec);
        const r = s.reverse();
        expect(r.directed).toBe(true);
        // in-neighbours: 0 <- {2}; 1 <- {0}; 2 <- {0 (e1), 0 (e4), 1}
        expect(Array.from(r.rowPtr)).toEqual([0, 1, 2, 5]);
        expect(Array.from(r.colIdx)).toEqual([2, 0, 0, 0, 1]);
        const fwd = r.fwdArc;
        for (let v = 0; v < r.nodeCount; v++) {
            for (let k = r.rowPtr[v]; k < r.rowPtr[v + 1]; k++) {
                // reverse arc k in row v is the forward arc fwd[k] targeting v from source colIdx'[k]
                expect(s.colIdx[fwd[k]]).toBe(v);
                expect(s.arcSource(fwd[k])).toBe(r.colIdx[k]);
                expect(r.arcToEdge[k]).toBe(s.arcToEdge[fwd[k]]);
                expect(r.weights?.[k]).toBe(s.weights?.[fwd[k]]);
            }
        }
        // reverse rows are sorted by source with parallels in edge order
        expect(r.arcToEdge[2]).toBeLessThan(r.arcToEdge[3]);
        expect(r.arcToEdge[2]).toBe(1);
        expect(r.arcToEdge[3]).toBe(4);
        expect(viewByteLength(s, "reverse", r)).toBe(
            r.rowPtr.byteLength + r.colIdx.byteLength * 3 + r.arcToEdge.byteLength,
        );
        expect(r instanceof ReverseAdjacency && Object.keys(r.materialised()).sort()).toEqual([
            "arcToEdge",
            "colIdx",
            "fwdArc",
            "rowPtr",
            "weights",
        ]);
    });

    it("directed identity reverse() aliases arcToEdge to fwdArc without materialising the snapshot's identity", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
            ],
        });
        const r = s.reverse();
        expect(r.arcToEdge).toBe(r.fwdArc);
        expect(peekView(s, "reverse")).toBe(r);
        expect(Array.from(r.fwdArc)).toEqual([0, 1, 2]);
    });

    it("coo() aliases dst and weights and reaches arcToEdge through a getter", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [1, 0, 2],
                [0, 1, 3],
                [0, 0, 4],
            ],
        });
        const coo = s.coo();
        expect(coo.dst).toBe(s.colIdx);
        expect(coo.weights).toBe(s.weights);
        expect(coo.arcToEdge).toBe(s.arcToEdge);
        expect(Array.from(coo.src)).toEqual([0, 0, 1]);
        expect(viewByteLength(s, "coo", coo)).toBe(12);
    });

    it("edgeList() gives declared orientations; identity snapshots alias dst and weights", () => {
        const spec: GraphSpec = {
            directed: false,
            edges: [
                [2, 0, 5],
                [0, 2, 6],
                [1, 1, 7],
                [3, 1, 8],
            ],
        };
        const s = makeSnapshot(spec);
        const list = s.edgeList();
        expect(Array.from(list.src)).toEqual([2, 0, 1, 3]);
        expect(Array.from(list.dst)).toEqual([0, 2, 1, 1]);
        expect(Array.from(list.weights as Float32Array)).toEqual([5, 6, 7, 8]);
        expect(list.arc).toBe(s.edgeToArc);
        expect(viewByteLength(s, "edgeList", list)).toBe(16 * 3);
        const identity = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 2],
                [0, 2, 3],
                [1, 2, 4],
            ],
        });
        const il = identity.edgeList();
        expect(il.dst).toBe(identity.colIdx);
        expect(il.weights).toBe(identity.weights);
        expect(Array.from(il.src)).toEqual([0, 0, 1]);
        expect(viewByteLength(identity, "edgeList", il)).toBe(12);
        const unweighted = makeSnapshot({ directed: false, edges: [[1, 0]] });
        expect(unweighted.edgeList().weights).toBeNull();
    });
});

describe("self-loop views", () => {
    it("selfLoopArcs lists loop arcs ascending and selfLoopsPerNode counts them", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 0],
                [1, 2],
                [2, 2],
                [2, 2],
                [0, 1],
            ],
        });
        const arcs = s.selfLoopArcs();
        expect(arcs.length).toBe(s.selfLoopCount);
        for (const a of arcs) {
            expect(s.colIdx[a]).toBe(s.arcSource(a));
        }
        expect(Array.from(arcs)).toEqual(Array.from(arcs).sort((a, b) => a - b));
        expect(Array.from(s.selfLoopsPerNode())).toEqual([1, 0, 2]);
        expect(s.selfLoopArcs()).toBe(s.selfLoopArcs());
        expect(s.cachedViews()).toContain("selfLoopsPerNode");
    });

    it("selfLoopWeight sums loop weights, widened counts when unweighted", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 0, 2],
                [0, 0, 3],
                [1, 1, -1],
                [0, 1, 9],
            ],
        });
        expect(Array.from(s.selfLoopWeight())).toEqual([5, -1]);
        const un = makeSnapshot({
            directed: true,
            edges: [
                [0, 0],
                [0, 0],
                [0, 1],
            ],
        });
        expect(Array.from(un.selfLoopWeight())).toEqual([2, 0]);
    });
});

describe("mate()", () => {
    it("pairs k-th with k-th for parallels, maps self-loops to themselves (P4)", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 1],
                [2, 0],
            ],
        });
        const mate = s.mate();
        for (let a = 0; a < s.arcCount; a++) {
            expect(mate[mate[a]]).toBe(a);
            expect(s.arcToEdge[mate[a]]).toBe(s.arcToEdge[a]);
        }
        const [lo, hi] = s.arcsBetween(0, 1);
        const [lo2] = s.arcsBetween(1, 0);
        for (let i = 0; i < hi - lo; i++) {
            expect(mate[lo + i]).toBe(lo2 + i);
        }
        const loop = s.findArc(1, 1);
        expect(mate[loop]).toBe(loop);
    });

    it("throws E_DIRECTED on a directed snapshot", () => {
        const s = makeSnapshot({ directed: true, edges: [[0, 1]] });
        expectError(() => s.mate(), "E_DIRECTED");
        expectError(() => s.prepare(["mate"]), "E_DIRECTED");
    });

    it("throws E_INVALID_SNAPSHOT (I7) when the doubled storage is broken", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        // corrupt: replace the mate of 0->1 in row 1 with a self-loop target
        s.colIdx[s.findArc(1, 0)] = 1;
        const error = expectError(() => s.mate(), "E_INVALID_SNAPSHOT");
        expect(error.details.invariant).toBe("I7");
    });

    it("random undirected multigraphs satisfy mate(mate(a)) === a", () => {
        const edgesArb = fc.integer({ min: 1, max: 7 }).chain((n) =>
            fc
                .array(fc.tuple(fc.integer({ min: 0, max: n - 1 }), fc.integer({ min: 0, max: n - 1 })), {
                    maxLength: 25,
                })
                .map((edges) => ({ n, edges: edges.map(([u, v]) => [u, v] as const) })),
        );
        fc.assert(
            fc.property(edgesArb, ({ n, edges }) => {
                const s = makeSnapshot({ directed: false, nodeCount: n, edges });
                const mate = s.mate();
                for (let a = 0; a < s.arcCount; a++) {
                    expect(mate[mate[a]]).toBe(a);
                    expect(s.arcToEdge[mate[a]]).toBe(s.arcToEdge[a]);
                    expect(s.colIdx[mate[a]]).toBe(s.arcSource(a));
                }
            }),
            { numRuns: 40 },
        );
    });
});

describe("degreeOrder()", () => {
    it("orders by descending degree with stable ties and reports the cuGraph tiers", () => {
        const s = makeSnapshot({
            directed: true,
            nodeCount: 5,
            edges: [
                [1, 0],
                [1, 2],
                [3, 0],
                [3, 1],
                [3, 2],
                [0, 4],
            ],
        });
        const order = s.degreeOrder();
        expect(Array.from(order.perm)).toEqual([3, 1, 0, 2, 4]);
        expect(Array.from(order.segmentOffsets)).toEqual([0, 0, 0, 3, 5]);
        const reverse = s.degreeOrder({ of: "reverse" });
        // in-degrees: 0 <- 2, 1 <- 1, 2 <- 2, 3 <- 0, 4 <- 1
        expect(Array.from(reverse.perm)).toEqual([0, 2, 1, 4, 3]);
        expect(reverse).not.toBe(order);
        expect(s.degreeOrder({ of: "reverse" })).toBe(reverse);
        expect(s.cachedViews()).toEqual(expect.arrayContaining(["degreeOrder", "reverseDegreeOrder", "reverse"]));
    });

    it("computes the tier boundaries at 1024 / 32 / 1", () => {
        const rowPtr = new Uint32Array([
            0,
            DEGREE_TIER_HIGH,
            DEGREE_TIER_HIGH + DEGREE_TIER_MID,
            DEGREE_TIER_HIGH + DEGREE_TIER_MID + 5,
            DEGREE_TIER_HIGH + DEGREE_TIER_MID + 5,
        ]);
        const order = computeDegreeOrder(rowPtr, 4);
        expect(Array.from(order.perm)).toEqual([0, 1, 2, 3]);
        expect(Array.from(order.segmentOffsets)).toEqual([0, 1, 2, 3, 4]);
        const empty = computeDegreeOrder(new Uint32Array([0]), 0);
        expect(Array.from(empty.segmentOffsets)).toEqual([0, 0, 0, 0, 0]);
        expect(empty.perm.length).toBe(0);
    });

    it("never contains INVALID_INDEX and is a permutation", () => {
        const s = makeSnapshot({ directed: false, edges: KARATE_EDGES });
        const { perm } = s.degreeOrder();
        expect(new Set(perm).size).toBe(34);
        expect(perm.includes(INVALID_INDEX)).toBe(false);
        for (let i = 1; i < perm.length; i++) {
            expect(s.outDegreeOf(perm[i - 1])).toBeGreaterThanOrEqual(s.outDegreeOf(perm[i]));
        }
    });
});

describe("isSymmetric()", () => {
    it("is true for undirected snapshots without computing anything", () => {
        const s = makeSnapshot({ directed: false, edges: [[0, 1]] });
        expect(s.isSymmetric()).toBe(true);
        expect(s.cachedViews()).toEqual(["symmetric"]);
    });

    it("compares forward and reverse rows with weights on directed snapshots", () => {
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1],
                    [1, 0],
                    [2, 2],
                ],
            }).isSymmetric(),
        ).toBe(true);
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                ],
            }).isSymmetric(),
        ).toBe(false);
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1, 2],
                    [1, 0, 2],
                ],
            }).isSymmetric(),
        ).toBe(true);
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1, 2],
                    [1, 0, 3],
                ],
            }).isSymmetric(),
        ).toBe(false);
        expect(makeSnapshot({ directed: true, nodeCount: 2, edges: [] }).isSymmetric()).toBe(true);
        expect(
            makeSnapshot({
                directed: true,
                edges: [
                    [0, 1],
                    [0, 1],
                    [1, 0],
                ],
            }).isSymmetric(),
        ).toBe(false);
    });
});

describe("prepare / dropCaches / cachedViews", () => {
    it("prepare materialises the named views and dropCaches releases them", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1, 2],
                [1, 2],
            ],
        });
        expect(s.cachedViews()).toEqual([]);
        expect(s.prepare(["outDegree", "mate", "totalWeight", "symmetric", "reverseDegreeOrder"])).toBe(s);
        expect(s.cachedViews()).toEqual([
            "outDegree",
            "totalWeight",
            "mate",
            "degreeOrder",
            "reverseDegreeOrder",
            "symmetric",
        ]);
        expect(s.byteLength({ views: true })).toBe(s.byteLength() + 12 + 16 + 12 + 20);
        s.dropCaches();
        expect(s.cachedViews()).toEqual([]);
        expect(s.byteLength({ views: true })).toBe(s.byteLength());
        expect(s.outDegree()).toEqual(new Uint32Array([1, 2, 1]));
    });

    it("prepare accepts every view name on a directed snapshot except mate", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [1, 0, 2],
                [0, 1],
            ],
        });
        s.prepare([
            "reverse",
            "coo",
            "edgeList",
            "outDegree",
            "inDegree",
            "degree",
            "weightedOutDegree",
            "weightedInDegree",
            "weightedDegree",
            "selfLoopWeight",
            "totalWeight",
            "selfLoopArcs",
            "selfLoopsPerNode",
            "degreeOrder",
            "reverseDegreeOrder",
            "symmetric",
        ]);
        expect(s.cachedViews().length).toBe(16);
        expect(s.byteLength({ views: true })).toBeGreaterThan(s.byteLength());
    });
});

describe("foldArcs / expandEdges (section 7.5)", () => {
    const s = makeSnapshot({
        directed: false,
        edges: [
            [0, 1],
            [1, 2],
            [2, 2],
        ],
    });

    it("expandEdges gathers per-edge values to both arcs; foldArcs reduces them back", () => {
        const perEdge = new Float64Array([10, 20, 30]);
        const perArc = expandEdges(s, perEdge);
        expect(perArc.length).toBe(s.arcCount);
        for (let a = 0; a < s.arcCount; a++) {
            expect(perArc[a]).toBe(perEdge[s.arcToEdge[a]]);
        }
        expect(Array.from(foldArcs(s, perArc, "first"))).toEqual([10, 20, 30]);
        expect(Array.from(foldArcs(s, perArc, "sum"))).toEqual([20, 40, 30]);
        expect(Array.from(foldArcs(s, perArc, "max"))).toEqual([10, 20, 30]);
        const mixed = perArc.slice();
        mixed[s.mate()[s.edgeToArc[0]]] = 1;
        expect(Array.from(foldArcs(s, mixed, "min"))).toEqual([1, 20, 30]);
        expect(Array.from(foldArcs(s, mixed, "max"))).toEqual([10, 20, 30]);
        expect(Array.from(foldArcs(s, mixed, "first"))).toEqual([10, 20, 30]);
        const out = new Float64Array(3);
        expect(foldArcs(s, perArc, "sum", out)).toBe(out);
        const outArcs = new Float64Array(s.arcCount);
        expect(expandEdges(s, perEdge, outArcs)).toBe(outArcs);
    });

    it("returns the input itself when the permutation is the identity, copies into out when given", () => {
        const id = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
        });
        const vec = new Uint32Array([3, 4]);
        expect(foldArcs(id, vec, "sum")).toBe(vec);
        expect(expandEdges(id, vec)).toBe(vec);
        const out = new Uint32Array(2);
        expect(foldArcs(id, vec, "first", out)).toBe(out);
        expect(Array.from(out)).toEqual([3, 4]);
        expect(Array.from(expandEdges(id, vec, new Uint32Array(2)))).toEqual([3, 4]);
    });

    it("works for every NumericVector class", () => {
        for (const vec of [new Float32Array([1, 2, 3]), new Int32Array([1, 2, 3]), new Uint32Array([1, 2, 3])]) {
            const arcs = expandEdges(s, vec);
            expect(arcs.constructor).toBe(vec.constructor);
            expect(foldArcs(s, arcs, "sum").constructor).toBe(vec.constructor);
        }
    });

    it("rejects vectors of the wrong length with E_COLUMN_LENGTH", () => {
        expectError(() => foldArcs(s, new Float64Array(2), "sum"), "E_COLUMN_LENGTH");
        expectError(() => expandEdges(s, new Float64Array(2)), "E_COLUMN_LENGTH");
    });
});

describe("identityPermutation", () => {
    it("is 0..n-1 over a fresh buffer", () => {
        expect(Array.from(identityPermutation(4))).toEqual([0, 1, 2, 3]);
        expect(identityPermutation(0).length).toBe(0);
    });
});
