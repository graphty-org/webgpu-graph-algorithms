/**
 * Adversarial audit of the query methods (design section 3.9, P6), INVALID_INDEX handling at every
 * index- and id-taking boundary (section 11.1), the degree conventions of section 3.4 for directed
 * graphs, the degree-order tiers, the fold / expand helpers (7.5) and every view over the empty and
 * single-node edge cases (section 11.3).
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { expandEdges, foldArcs } from "../../src/snapshot/views.js";
import { type GraphSnapshot, type ViewName } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";

const RUNS = Number(process.env.FC_RUNS ?? "150");

const VIEW_NAMES: readonly ViewName[] = [
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
    "mate",
    "degreeOrder",
    "reverseDegreeOrder",
    "symmetric",
];

interface RawEdge {
    readonly u: number;
    readonly v: number;
    readonly w: number | undefined;
}

interface RawGraph {
    readonly directed: boolean;
    readonly nodeCount: number;
    readonly edges: readonly RawEdge[];
}

const arbGraph: fc.Arbitrary<RawGraph> = fc
    .record({ directed: fc.boolean(), nodeCount: fc.integer({ min: 1, max: 6 }) })
    .chain(({ directed, nodeCount }) =>
        fc
            .array(
                fc.record({
                    u: fc.integer({ min: 0, max: nodeCount - 1 }),
                    v: fc.integer({ min: 0, max: nodeCount - 1 }),
                    w: fc.option(fc.constantFrom(1, 0, -1, 2.5, 0.1, 3), { nil: undefined }),
                }),
                { maxLength: 24 },
            )
            .map((edges) => ({ directed, nodeCount, edges })),
    );

function build(g: RawGraph, weighted?: boolean): GraphSnapshot {
    const b = new GraphBuilder({ directed: g.directed, ...(weighted === undefined ? {} : { weighted }) });
    b.addAnonymousNodes(g.nodeCount);
    for (const e of g.edges) {
        b.addEdgeByIndex(e.u, e.v, e.w);
    }
    return b.freeze();
}

/** The arcs u -> v by a linear scan of row u. */
function scan(s: GraphSnapshot, u: number, v: number): number[] {
    const out: number[] = [];
    for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
        if (s.colIdx[a] === v) {
            out.push(a);
        }
    }
    return out;
}

describe("queries agree with linear scans (P6) and INVALID_INDEX is handled at every boundary", () => {
    it("findArc / hasArc / arcsBetween / multiplicity / selfLoopsAt for every (u, v) pair, plus INVALID_INDEX and out-of-range targets", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const s = build(g);
                for (let u = 0; u < s.nodeCount; u++) {
                    for (let v = 0; v < s.nodeCount; v++) {
                        const arcs = scan(s, u, v);
                        const first = arcs.length === 0 ? INVALID_INDEX : arcs[0];
                        expect(s.findArc(u, v)).toBe(first);
                        expect(s.hasArc(u, v)).toBe(arcs.length > 0);
                        const [lo, hi] = s.arcsBetween(u, v);
                        expect(hi - lo).toBe(arcs.length);
                        expect(s.multiplicity(u, v)).toBe(arcs.length);
                        if (arcs.length > 0) {
                            expect(lo).toBe(arcs[0]);
                            expect(hi).toBe(arcs[arcs.length - 1] + 1);
                            // the first arc is the lowest logical edge index among the parallels (3.5)
                            for (const a of arcs) {
                                expect(s.arcToEdge[first]).toBeLessThanOrEqual(s.arcToEdge[a]);
                            }
                        } else {
                            expect(lo).toBe(hi);
                            expect(lo).toBeGreaterThanOrEqual(s.rowPtr[u]);
                            expect(lo).toBeLessThanOrEqual(s.rowPtr[u + 1]);
                        }
                    }
                    expect(s.selfLoopsAt(u)).toBe(scan(s, u, u).length);
                    // a target that is not a node: total, never a hit
                    for (const v of [INVALID_INDEX, s.nodeCount, s.nodeCount + 7]) {
                        expect(s.findArc(u, v)).toBe(INVALID_INDEX);
                        expect(s.hasArc(u, v)).toBe(false);
                        expect(s.multiplicity(u, v)).toBe(0);
                        const [lo, hi] = s.arcsBetween(u, v);
                        expect(lo).toBe(hi);
                        expect(lo).toBe(s.rowPtr[u + 1]);
                    }
                }
                // a source that is not a node: findArc / hasArc report a miss
                expect(s.findArc(INVALID_INDEX, 0)).toBe(INVALID_INDEX);
                expect(s.hasArc(INVALID_INDEX, 0)).toBe(false);
                expect(s.findArc(s.nodeCount, 0)).toBe(INVALID_INDEX);
                expect(s.hasArc(INVALID_INDEX, INVALID_INDEX)).toBe(false);
            }),
            { numRuns: RUNS },
        );
    });

    it("arcSource / edgeSource / edgeTarget agree with the rows before and after the coo / edgeList views are cached", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const s = build(g);
                const rowOf = (a: number): number => {
                    let u = 0;
                    while (s.rowPtr[u + 1] <= a) {
                        u++;
                    }
                    return u;
                };
                const check = (): void => {
                    for (let a = 0; a < s.arcCount; a++) {
                        expect(s.arcSource(a)).toBe(rowOf(a));
                    }
                    g.edges.forEach((e, i) => {
                        expect(s.edgeSource(i)).toBe(e.u);
                        expect(s.edgeTarget(i)).toBe(e.v);
                    });
                };
                check();
                expect(s.cachedViews()).not.toContain("coo");
                s.coo();
                check();
                s.edgeList();
                check();
                s.dropCaches();
                expect(s.cachedViews()).toEqual([]);
                check();
            }),
            { numRuns: RUNS },
        );
    });

    it("id- and row-taking boundary calls are checked: idOf / value throw E_INDEX_RANGE, indexOf / edgeIndexOf return INVALID_INDEX, requireIndex throws E_UNKNOWN_NODE", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("label", 0, "x");
        b.setEdgeValue("id", 0, "e0");
        const s = b.freeze();
        s.edges.set("eid", ["k0"], { role: "id", dtype: "string", unique: true });
        for (const bad of [INVALID_INDEX, s.nodeCount, -1, 1.5, NaN]) {
            expect(() => s.ids.idOf(bad)).toThrow(expect.objectContaining({ code: "E_INDEX_RANGE" }));
            expect(() => s.nodes.value("label", bad)).toThrow(expect.objectContaining({ code: "E_INDEX_RANGE" }));
            expect(() => s.nodes.require("label").value(bad)).toThrow(
                expect.objectContaining({ code: "E_INDEX_RANGE" }),
            );
        }
        expect(s.ids.indexOf(INVALID_INDEX)).toBe(INVALID_INDEX);
        expect(s.ids.indexOf("zzz")).toBe(INVALID_INDEX);
        expect(s.ids.indexOf(NaN)).toBe(INVALID_INDEX);
        expect(() => s.ids.requireIndex("zzz")).toThrow(expect.objectContaining({ code: "E_UNKNOWN_NODE" }));
        expect(s.edgeIndexOf(INVALID_INDEX)).toBe(INVALID_INDEX);
        expect(s.edgeIndexOf("nope")).toBe(INVALID_INDEX);
        expect(s.edgeIndexOf("k0")).toBe(0);
        expect(() => s.nodes.value("missing", 0)).toThrow(expect.objectContaining({ code: "E_UNKNOWN_COLUMN" }));
        // a numeric id equal to INVALID_INDEX is a legal id and resolves (SameValueZero, never coerced)
        const c = new GraphBuilder({ directed: true });
        c.addNode(INVALID_INDEX);
        const t = c.freeze();
        expect(t.ids.indexOf(INVALID_INDEX)).toBe(0);
        expect(t.ids.idOf(0)).toBe(INVALID_INDEX);
        assertInvariants(t);
    });
});

describe("degree conventions on directed graphs (design 3.4) and the degree-order tiers (7.2)", () => {
    it("degree = in + out; a directed loop is counted once in each; sums equal edgeCount; weighted variants follow the same shape", () => {
        fc.assert(
            fc.property(
                arbGraph.filter((g) => g.directed),
                (g) => {
                    const s = build(g);
                    const out = s.outDegree();
                    const inn = s.inDegree();
                    const degree = s.degree();
                    const wOut = s.weightedOutDegree();
                    const wIn = s.weightedInDegree();
                    const wDeg = s.weightedDegree();
                    let sumOut = 0;
                    let sumIn = 0;
                    for (let u = 0; u < s.nodeCount; u++) {
                        const expectedOut = g.edges.filter((e) => e.u === u).length;
                        const expectedIn = g.edges.filter((e) => e.v === u).length;
                        expect(out[u]).toBe(expectedOut);
                        expect(inn[u]).toBe(expectedIn);
                        expect(degree[u]).toBe(expectedOut + expectedIn);
                        const weightOf = (e: RawEdge): number => Math.fround(e.w ?? 1);
                        const expectedWOut = g.edges.filter((e) => e.u === u).reduce((acc, e) => acc + weightOf(e), 0);
                        const expectedWIn = g.edges.filter((e) => e.v === u).reduce((acc, e) => acc + weightOf(e), 0);
                        expect(wOut[u]).toBeCloseTo(expectedWOut, 5);
                        expect(wIn[u]).toBeCloseTo(expectedWIn, 5);
                        expect(wDeg[u]).toBeCloseTo(expectedWOut + expectedWIn, 5);
                        sumOut += out[u];
                        sumIn += inn[u];
                    }
                    expect(sumOut).toBe(s.edgeCount);
                    expect(sumIn).toBe(s.edgeCount);
                    expect(inn).not.toBe(out);
                    const total = g.edges.reduce((acc, e) => acc + Math.fround(e.w ?? 1), 0);
                    expect(s.totalWeight()).toBeCloseTo(total, 5);
                },
            ),
            { numRuns: RUNS },
        );
    });

    it("degreeOrder(): perm is a permutation sorted by descending degree with ties in ascending index, tiers at 1024 / 32 / 1, reverse uses the in-degree", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const s = build(g);
                for (const of of ["forward", "reverse"] as const) {
                    const view = s.degreeOrder({ of });
                    const degree = of === "forward" || !s.directed ? s.outDegree() : s.inDegree();
                    const seen = new Set(view.perm);
                    expect(seen.size).toBe(s.nodeCount);
                    for (let i = 1; i < view.perm.length; i++) {
                        const a = view.perm[i - 1];
                        const b = view.perm[i];
                        expect(degree[a] > degree[b] || (degree[a] === degree[b] && a < b)).toBe(true);
                    }
                    const count = (predicate: (d: number) => boolean): number =>
                        Array.from(degree).filter(predicate).length;
                    expect(Array.from(view.segmentOffsets)).toEqual([
                        0,
                        count((d) => d >= 1024),
                        count((d) => d >= 32),
                        count((d) => d >= 1),
                        s.nodeCount,
                    ]);
                }
                if (!s.directed) {
                    expect(s.degreeOrder({ of: "reverse" })).toBe(s.degreeOrder());
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("foldArcs / expandEdges (7.5) round-trip through arcToEdge and return the input itself only on the identity path", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const s = build(g);
                const perEdge = Float64Array.from({ length: s.edgeCount }, (_, e) => e + 0.5);
                const perArc = expandEdges(s, perEdge);
                expect(perArc.length).toBe(s.arcCount);
                for (let a = 0; a < s.arcCount; a++) {
                    expect(perArc[a]).toBe(perEdge[s.arcToEdge[a]]);
                }
                for (const reducer of ["first", "sum", "max", "min"] as const) {
                    const folded = foldArcs(s, perArc, reducer);
                    expect(folded.length).toBe(s.edgeCount);
                    for (let e = 0; e < s.edgeCount; e++) {
                        const arcs = s.directed || s.edgeSource(e) === s.edgeTarget(e) ? 1 : 2;
                        expect(folded[e]).toBe(reducer === "sum" ? perEdge[e] * arcs : perEdge[e]);
                    }
                    expect(folded === perArc).toBe(s.flags.arcToEdgeIsIdentity);
                }
                expect(perArc === perEdge).toBe(s.flags.arcToEdgeIsIdentity);
                expect(() => foldArcs(s, new Float64Array(s.arcCount + 1), "sum")).toThrow(
                    expect.objectContaining({ code: "E_COLUMN_LENGTH" }),
                );
                expect(() => expandEdges(s, new Float64Array(s.edgeCount + 1))).toThrow(
                    expect.objectContaining({ code: "E_COLUMN_LENGTH" }),
                );
            }),
            { numRuns: RUNS },
        );
    });
});

describe("empty and single-node snapshots through every view (design 11.3)", () => {
    const cases: readonly [string, () => GraphSnapshot][] = [
        ["empty directed", () => new GraphBuilder({ directed: true }).freeze()],
        ["empty undirected", () => new GraphBuilder({ directed: false }).freeze()],
        ["empty directed, weighted: true", () => new GraphBuilder({ directed: true, weighted: true }).freeze()],
        [
            "one node directed",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addNode("only");
                return b.freeze();
            },
        ],
        [
            "one node undirected",
            () => {
                const b = new GraphBuilder({ directed: false });
                b.addNode("only");
                return b.freeze();
            },
        ],
        [
            "one node, one directed loop, weight 0",
            () => {
                const b = new GraphBuilder({ directed: true });
                b.addEdge("only", "only", 0);
                return b.freeze();
            },
        ],
        [
            "one node, two undirected loops",
            () => {
                const b = new GraphBuilder({ directed: false });
                b.addEdge("only", "only", 2.5);
                b.addEdge("only", "only");
                return b.freeze();
            },
        ],
        [
            "nodes only, no edges, weighted: true (node-only graph before its edges arrive)",
            () => {
                const b = new GraphBuilder({ directed: false, weighted: true });
                b.addNodes(["a", "b", "c"]);
                return b.freeze();
            },
        ],
    ];

    for (const [name, make] of cases) {
        it(name, () => {
            const s = make();
            assertInvariants(s);
            const n = s.nodeCount;
            expect(s.rowPtr.length).toBe(n + 1);
            expect(s.rowPtr[0]).toBe(0);
            expect(s.rowPtr[n]).toBe(s.arcCount);
            expect(s.colIdx.length).toBe(s.arcCount);
            expect(s.arcToEdge.length).toBe(s.arcCount);
            expect(s.edgeToArc.length).toBe(s.edgeCount);
            if (s.edgeCount === 0) {
                expect(s.weights === null).toBe(!s.flags.weighted);
                expect(s.flags.arcToEdgeIsIdentity).toBe(s.directed);
                expect(s.flags.multigraph).toBe(false);
                expect(s.flags.hasSelfLoops).toBe(false);
                expect(s.flags.allWeightsOne).toBe(true);
                expect(s.edges.byRole("weight")).toBeNull();
            }
            // every view is computable, has the right length and is memoised
            for (const view of VIEW_NAMES) {
                if (view === "mate" && s.directed) {
                    expect(() => s.mate()).toThrow(expect.objectContaining({ code: "E_DIRECTED" }));
                    continue;
                }
                s.prepare([view]);
                expect(s.cachedViews()).toContain(view);
            }
            expect(s.outDegree().length).toBe(n);
            expect(s.inDegree().length).toBe(n);
            expect(s.degree().length).toBe(n);
            expect(s.weightedOutDegree().length).toBe(n);
            expect(s.weightedInDegree().length).toBe(n);
            expect(s.weightedDegree().length).toBe(n);
            expect(s.selfLoopWeight().length).toBe(n);
            expect(s.selfLoopsPerNode().length).toBe(n);
            expect(s.selfLoopArcs().length).toBe(s.selfLoopCount);
            expect(s.coo().src.length).toBe(s.arcCount);
            expect(s.coo().dst).toBe(s.colIdx);
            expect(s.edgeList().src.length).toBe(s.edgeCount);
            expect(s.edgeList().dst.length).toBe(s.edgeCount);
            expect(s.edgeList().arc).toBe(s.edgeToArc);
            expect(s.reverse().rowPtr.length).toBe(n + 1);
            expect(s.reverse().colIdx.length).toBe(s.arcCount);
            expect(s.reverse().fwdArc.length).toBe(s.arcCount);
            expect(s.reverse().arcToEdge.length).toBe(s.arcCount);
            expect(s.degreeOrder().perm.length).toBe(n);
            expect(Array.from(s.degreeOrder().segmentOffsets)).toEqual([0, 0, 0, s.arcCount > 0 ? 1 : 0, n]);
            expect(s.isSymmetric()).toBe(true);
            expect(s.totalWeight()).toBe(
                s.weights === null ? s.edgeCount : Array.from(s.edgeList().weights ?? []).reduce((a, w) => a + w, 0),
            );
            if (!s.directed) {
                expect(s.mate().length).toBe(s.arcCount);
                for (let a = 0; a < s.arcCount; a++) {
                    expect(s.mate()[a]).toBe(a);
                }
            }
            for (let u = 0; u < n; u++) {
                expect(s.degree()[u]).toBe(
                    s.directed ? s.outDegree()[u] + s.inDegree()[u] : s.outDegree()[u] + s.selfLoopsPerNode()[u],
                );
                expect(s.outArcs(u)).toEqual([s.rowPtr[u], s.rowPtr[u + 1]]);
            }
            expect(s.byteLength({ views: true, columns: true, ids: true })).toBeGreaterThanOrEqual(s.byteLength());
            s.dropCaches();
            expect(s.cachedViews()).toEqual([]);
            s.validate({ level: "full" });
            s.validate({ level: "structure" });
        });
    }

    it("a single self-loop is its own mate, counts once in outDegree and twice in degree, and is the only declared arc", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("x", "x", 3);
        const s = b.freeze();
        expect(Array.from(s.rowPtr)).toEqual([0, 1]);
        expect(Array.from(s.colIdx)).toEqual([0]);
        expect(Array.from(s.arcToEdge)).toEqual([0]);
        expect(Array.from(s.edgeToArc)).toEqual([0]);
        expect(s.flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(s.mate())).toEqual([0]);
        expect(Array.from(s.outDegree())).toEqual([1]);
        expect(Array.from(s.degree())).toEqual([2]);
        expect(Array.from(s.weightedOutDegree())).toEqual([3]);
        expect(Array.from(s.weightedDegree())).toEqual([6]);
        expect(s.totalWeight()).toBe(3);
        expect(Array.from(s.selfLoopWeight())).toEqual([3]);
        expect(s.selfLoopsAt(0)).toBe(1);
        expect(s.findArc(0, 0)).toBe(0);
        expect(s.arcsBetween(0, 0)).toEqual([0, 1]);
        expect(s.multiplicity(0, 0)).toBe(1);
    });
});
