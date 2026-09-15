/**
 * Adversarial audit of undirected storage (I7, design sections 3.3, 3.4, 6.4): the lockstep mate
 * pairing under parallel edges in BOTH declared orientations mixed with self-loops, the declared
 * orientation of edgeToArc, the duplicate policies over undirected groups, and what mate() /
 * validate() do when the doubled storage is corrupted.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { createSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type DuplicatePolicy, type GraphSnapshot } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { type EdgeSpec, type GraphSpec, makeParts, type MutableParts } from "../helpers/parts.js";

const RUNS = Number(process.env.FC_RUNS ?? "150");

interface RawEdge {
    readonly u: number;
    readonly v: number;
    readonly w: number | undefined;
}

/** Undirected multigraphs dense enough that parallels in both orientations and loops are common. */
const arbUndirected: fc.Arbitrary<{ nodeCount: number; edges: RawEdge[] }> = fc
    .integer({ min: 1, max: 4 })
    .chain((nodeCount) =>
        fc
            .array(
                fc.record({
                    u: fc.integer({ min: 0, max: nodeCount - 1 }),
                    v: fc.integer({ min: 0, max: nodeCount - 1 }),
                    w: fc.option(fc.constantFrom(1, 0, -1, 2.5, 0.1, Infinity), { nil: undefined }),
                }),
                { maxLength: 24 },
            )
            .map((edges) => ({ nodeCount, edges })),
    );

function build(
    g: { nodeCount: number; edges: readonly RawEdge[] },
    options: { duplicateEdges?: DuplicatePolicy } = {},
): GraphSnapshot {
    const b = new GraphBuilder({ directed: false, ...options });
    b.addAnonymousNodes(g.nodeCount);
    for (const e of g.edges) {
        b.addEdgeByIndex(e.u, e.v, e.w);
    }
    return b.freeze();
}

function expectInvariant(fn: () => unknown, invariants: readonly string[]): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe("E_INVALID_SNAPSHOT");
    expect(invariants).toContain(error.details.invariant);
    return error;
}

describe("I7 pairing on random undirected multigraphs with loops and mixed orientations (P4)", () => {
    it("mate() is an involution that pairs the k-th u->v arc with the k-th v->u arc of the same edge and weight", () => {
        fc.assert(
            fc.property(arbUndirected, (g) => {
                const s = build(g);
                assertInvariants(s);
                const mate = s.mate();
                expect(mate).toBe(s.mate());
                expect(mate.length).toBe(s.arcCount);
                for (let a = 0; a < s.arcCount; a++) {
                    const b = mate[a];
                    expect(b).toBeLessThan(s.arcCount);
                    expect(mate[b]).toBe(a);
                    expect(s.arcToEdge[b]).toBe(s.arcToEdge[a]);
                    expect(s.arcSource(b)).toBe(s.colIdx[a]);
                    expect(s.colIdx[b]).toBe(s.arcSource(a));
                    if (s.weights !== null) {
                        expect(Object.is(s.weights[a], s.weights[b])).toBe(true);
                    }
                    expect(b === a).toBe(s.colIdx[a] === s.arcSource(a));
                }
                // k-th / k-th: the arc ranges of (u, v) and (v, u) list the same edges in the same order
                for (let u = 0; u < s.nodeCount; u++) {
                    for (let v = u + 1; v < s.nodeCount; v++) {
                        const [lo, hi] = s.arcsBetween(u, v);
                        const [lo2, hi2] = s.arcsBetween(v, u);
                        expect(hi - lo).toBe(hi2 - lo2);
                        expect(s.multiplicity(u, v)).toBe(s.multiplicity(v, u));
                        for (let k = 0; k < hi - lo; k++) {
                            expect(s.arcToEdge[lo + k]).toBe(s.arcToEdge[lo2 + k]);
                            expect(mate[lo + k]).toBe(lo2 + k);
                        }
                        // parallels are in ascending edge order (I4) on both sides
                        for (let k = 1; k < hi - lo; k++) {
                            expect(s.arcToEdge[lo + k]).toBeGreaterThan(s.arcToEdge[lo + k - 1]);
                        }
                    }
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("edgeToArc[e] is the arc in the row of the DECLARED source and its mate lives in the row of the declared target", () => {
        fc.assert(
            fc.property(arbUndirected, (g) => {
                const s = build(g);
                const mate = s.mate();
                g.edges.forEach((e, i) => {
                    const a = s.edgeToArc[i];
                    expect(s.arcSource(a)).toBe(e.u);
                    expect(s.colIdx[a]).toBe(e.v);
                    expect(s.arcToEdge[a]).toBe(i);
                    expect(s.arcSource(mate[a])).toBe(e.v);
                    expect(s.edgeList().src[i]).toBe(e.u);
                    expect(s.edgeList().dst[i]).toBe(e.v);
                });
                // exactly edgeCount arcs hold the declared orientation (P5)
                let declared = 0;
                for (let a = 0; a < s.arcCount; a++) {
                    if (s.edgeToArc[s.arcToEdge[a]] === a) {
                        declared++;
                    }
                }
                expect(declared).toBe(s.edgeCount);
                expect(s.arcCount).toBe(2 * s.edgeCount - s.selfLoopCount);
                // reverse() is the forward adjacency itself (I7 consequence)
                const r = s.reverse();
                expect(r.rowPtr).toBe(s.rowPtr);
                expect(r.colIdx).toBe(s.colIdx);
                expect(r.weights).toBe(s.weights);
                expect(r.arcToEdge).toBe(s.arcToEdge);
                expect(Array.from(r.fwdArc)).toEqual(Array.from({ length: s.arcCount }, (_, k) => k));
            }),
            { numRuns: RUNS },
        );
    });

    it("degree conventions (design 3.4): undirected degree counts a loop twice, outDegree once; sums match the counting vocabulary", () => {
        fc.assert(
            fc.property(arbUndirected, (g) => {
                const s = build(g);
                const out = s.outDegree();
                const degree = s.degree();
                const loops = s.selfLoopsPerNode();
                let sumOut = 0;
                let sumDegree = 0;
                for (let u = 0; u < s.nodeCount; u++) {
                    expect(out[u]).toBe(s.rowPtr[u + 1] - s.rowPtr[u]);
                    expect(degree[u]).toBe(out[u] + loops[u]);
                    expect(loops[u]).toBe(s.selfLoopsAt(u));
                    expect(loops[u]).toBe(g.edges.filter((e) => e.u === u && e.v === u).length);
                    sumOut += out[u];
                    sumDegree += degree[u];
                }
                expect(s.inDegree()).toBe(out);
                expect(sumOut).toBe(s.arcCount);
                expect(sumDegree).toBe(2 * s.edgeCount);
                // weighted: sum(weightedDegree) === 2 * totalWeight (the modularity identity)
                const wd = s.weightedDegree();
                let sumWd = 0;
                for (let u = 0; u < s.nodeCount; u++) {
                    sumWd += wd[u];
                }
                const total = s.totalWeight();
                if (Number.isFinite(total)) {
                    expect(sumWd).toBeCloseTo(2 * total, 6);
                } else {
                    expect(Number.isFinite(sumWd)).toBe(false);
                }
            }),
            { numRuns: RUNS },
        );
    });
});

describe("duplicate policies over undirected groups (design section 6.5)", () => {
    const merging: readonly DuplicatePolicy[] = ["first", "last", "sum", "min", "max"];

    it("(u, v) and (v, u) parallels are ONE group: the survivor is the lowest (or highest) edge index across both orientations and keeps its own declared orientation", () => {
        fc.assert(
            fc.property(arbUndirected, fc.constantFrom(...merging), (g, policy) => {
                const b = new GraphBuilder({ directed: false });
                b.addAnonymousNodes(g.nodeCount);
                for (const e of g.edges) {
                    b.addEdgeByIndex(e.u, e.v, e.w);
                }
                const { snapshot: s, report } = b.freezeWithReport({ duplicateEdges: policy });
                assertInvariants(s);
                expect(s.flags.multigraph).toBe(false);
                // expected survivors from the unordered-pair grouping
                const groups = new Map<string, number[]>();
                g.edges.forEach((e, i) => {
                    const key = e.u <= e.v ? `${e.u}-${e.v}` : `${e.v}-${e.u}`;
                    const group = groups.get(key) ?? [];
                    group.push(i);
                    groups.set(key, group);
                });
                const survivors: number[] = [];
                const survivorOf = new Map<number, number>();
                for (const group of groups.values()) {
                    const survivor = policy === "last" ? group[group.length - 1] : group[0];
                    survivors.push(survivor);
                    for (const e of group) {
                        survivorOf.set(e, survivor);
                    }
                }
                survivors.sort((x, y) => x - y);
                expect(s.edgeCount).toBe(survivors.length);
                expect(report.mergedEdges).toBe(g.edges.length - survivors.length);
                survivors.forEach((old, e) => {
                    expect(s.edgeSource(e)).toBe(g.edges[old].u);
                    expect(s.edgeTarget(e)).toBe(g.edges[old].v);
                });
                if (report.edgeRemap === null) {
                    expect(report.mergedEdges).toBe(0);
                } else {
                    g.edges.forEach((_, old) => {
                        const survivor = survivorOf.get(old) as number;
                        expect(report.edgeRemap?.[old]).toBe(survivors.indexOf(survivor));
                    });
                }
                // mates still pair after the merge and carry the reduced weight on both arcs
                const mate = s.mate();
                for (let a = 0; a < s.arcCount; a++) {
                    expect(mate[mate[a]]).toBe(a);
                    if (s.weights !== null) {
                        expect(Object.is(s.weights[a], s.weights[mate[a]])).toBe(true);
                    }
                }
                // the builder was rewritten to the merged edge set: a second freeze is a no-op
                const again = b.freezeWithReport({ duplicateEdges: policy });
                expect(again.report.mergedEdges).toBe(0);
                expect(again.report.edgeRemap).toBeNull();
                expect(again.snapshot.contentHash()).toBe(s.contentHash());
            }),
            { numRuns: RUNS },
        );
    });

    it('"error" names an undirected pair once, from the row of the lower endpoint, in the builder\'s index space', () => {
        const b = new GraphBuilder({ directed: false });
        b.addAnonymousNodes(4);
        b.addEdgeByIndex(3, 1);
        b.addEdgeByIndex(2, 2);
        b.addEdgeByIndex(1, 3);
        let caught: unknown = null;
        try {
            b.freeze({ duplicateEdges: "error" });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        const error = caught as GraphFormatError;
        expect(error.code).toBe("E_DUPLICATE_EDGE");
        expect(error.details.source).toBe(1);
        expect(error.details.target).toBe(3);
        expect(error.details.edges).toEqual([0, 2]);
        // the refused freeze left the builder untouched
        expect(b.edgeBound).toBe(3);
        expect(b.edgeEndpoints(0)).toEqual([3, 1]);
        // a repeated self-loop is a group too
        const loops = new GraphBuilder({ directed: false });
        loops.addEdge("x", "x");
        loops.addEdge("x", "x");
        expect(() => loops.freeze({ duplicateEdges: "error" })).toThrow(
            expect.objectContaining({
                code: "E_DUPLICATE_EDGE",
                details: expect.objectContaining({ source: 0, target: 0 }),
            }),
        );
    });
});

describe("corrupted doubled storage: mate() and validate() refuse it with I7 (or the earlier invariant that breaks first)", () => {
    const BASE: GraphSpec = {
        directed: false,
        edges: [
            [0, 1, 2],
            [1, 0, 3],
            [1, 1, 4],
            [0, 2, 5],
            [2, 0, 5],
        ],
    };

    function corrupt(mutate: (parts: MutableParts) => void, spec: GraphSpec = BASE): GraphSnapshot {
        const parts = makeParts(spec);
        mutate(parts);
        return createSnapshot(parts);
    }

    it("the fixture itself is valid and its pairing is what the audit assumes", () => {
        const s = createSnapshot(makeParts(BASE));
        assertInvariants(s);
        expect(Array.from(s.rowPtr)).toEqual([0, 4, 7, 9]);
        expect(Array.from(s.colIdx)).toEqual([1, 1, 2, 2, 0, 0, 1, 0, 0]);
        expect(Array.from(s.arcToEdge)).toEqual([0, 1, 3, 4, 0, 1, 2, 3, 4]);
        expect(Array.from(s.mate())).toEqual([4, 5, 7, 8, 0, 1, 6, 2, 3]);
    });

    it("a mate whose colIdx points elsewhere", () => {
        const s = corrupt((p) => {
            p.colIdx[5] = 2; // row 1's second arc to 0 now targets 2: row 1 keeps sorted order [0, 2, 1]? no: [0, 2, 1] is unsorted
        });
        expectInvariant(() => s.mate(), ["I7"]);
        expectInvariant(() => s.validate(), ["I4", "I7"]);
        expect(() => s.validate({ level: "structure" })).not.toThrow();
    });

    it("row v holds fewer arcs to u than row u holds to v (counts still agree)", () => {
        // move one 1->0 arc to 2->1: rows stay sorted, arcCount unchanged, but pairing breaks
        const s = corrupt((p) => {
            p.colIdx[5] = 1; // row 1: [0, 1, 1] : the loop group grows, the 1->0 group shrinks
        });
        expectInvariant(() => s.mate(), ["I7"]);
        expectInvariant(() => s.validate(), ["I7", "I9"]);
    });

    it("mates that carry different logical edges", () => {
        const s = corrupt((p) => {
            const perm = p.arcToEdge as Uint32Array;
            [perm[4], perm[5]] = [perm[5], perm[4]]; // swap the two 1->0 mates: I4 tie order and the k-th pairing break
        });
        expectInvariant(() => s.validate(), ["I4", "I7"]);
        // mate() pairs positionally without reading arcToEdge, so it does not notice by itself
        expect(() => s.mate()).not.toThrow();
    });

    it("mates that carry different weights", () => {
        const s = corrupt((p) => {
            (p.weights as Float32Array)[4] = 99;
        });
        const error = expectInvariant(() => s.validate(), ["I7"]);
        expect(error.details.arc).toBeDefined();
        expect(() => s.mate()).not.toThrow();
    });

    it("a self-loop edge stored as two arcs", () => {
        const s = corrupt((p) => {
            // make the loop appear twice: relabel the (0,2)/(2,0) mates as edge 2 and the loop's mate... use edge ids only
            const perm = p.arcToEdge as Uint32Array;
            perm[2] = 2; // row 0's first arc to 2 now claims the loop edge
        });
        expectInvariant(() => s.validate(), ["I5", "I7"]);
    });

    it("an edge appearing three times, and an edge appearing never", () => {
        const s = corrupt((p) => {
            const perm = p.arcToEdge as Uint32Array;
            perm[3] = 3; // edge 3 now three times, edge 4 twice -> edge 4 still twice, edge 3 x3
            perm[8] = 3;
        });
        expectInvariant(() => s.validate(), ["I4", "I5", "I7"]);
    });

    it("selfLoopCount lies: the loop arc is counted as a non-loop", () => {
        const s = corrupt((p) => {
            p.selfLoopCount = 0;
            p.arcCount = 9;
            p.edgeCount = 5;
        });
        // arcCount (9) !== 2 * 5 - 0: the I7 count rule catches it at "structure"
        expectInvariant(() => s.validate({ level: "structure" }), ["I7"]);
    });

    it("mate() on a directed snapshot is E_DIRECTED and validate() ignores pairing there", () => {
        const s = createSnapshot(
            makeParts({
                directed: true,
                edges: [
                    [0, 1],
                    [1, 0],
                ],
            }),
        );
        expect(() => s.mate()).toThrow(expect.objectContaining({ code: "E_DIRECTED" }));
        expect(() => s.validate()).not.toThrow();
    });

    it("random single-cell corruption of an undirected core is either detected by validate() or leaves a snapshot every view can walk", () => {
        const specs: readonly (readonly EdgeSpec[])[] = [
            BASE.edges,
            [
                [0, 0],
                [0, 1],
                [1, 2],
                [2, 0],
                [1, 1],
                [2, 2],
            ],
        ];
        fc.assert(
            fc.property(
                fc.constantFrom(...specs),
                fc.constantFrom<"colIdx" | "arcToEdge" | "edgeToArc" | "rowPtr">(
                    "colIdx",
                    "arcToEdge",
                    "edgeToArc",
                    "rowPtr",
                ),
                fc.nat({ max: 20 }),
                fc.nat({ max: 20 }),
                (edges, array, position, value) => {
                    const spec: GraphSpec = { directed: false, edges, weighted: true };
                    const s = corrupt((p) => {
                        const target = p[array] as Uint32Array;
                        if (target.length === 0) {
                            return;
                        }
                        target[position % target.length] = value === 20 ? INVALID_INDEX : value;
                    }, spec);
                    let verdict: unknown = null;
                    try {
                        s.validate({ level: "full" });
                    } catch (err) {
                        verdict = err;
                    }
                    if (verdict !== null) {
                        expect(verdict).toBeInstanceOf(GraphFormatError);
                        expect((verdict as GraphFormatError).code).toBe("E_INVALID_SNAPSHOT");
                        expect(typeof (verdict as GraphFormatError).details.invariant).toBe("string");
                        return;
                    }
                    // validate accepted it: then every view and query must be computable and self-consistent
                    assertInvariants(s);
                    const mate = s.mate();
                    for (let a = 0; a < s.arcCount; a++) {
                        expect(mate[mate[a]]).toBe(a);
                    }
                    for (let e = 0; e < s.edgeCount; e++) {
                        expect(s.arcToEdge[s.edgeToArc[e]]).toBe(e);
                    }
                },
            ),
            { numRuns: RUNS * 2 },
        );
    });
});
