/**
 * Adversarial audit of row ordering (I4), the (target, edge) tie-break, arcToEdge identity
 * detection (design sections 3.1, 3.8, 6.3 step 4), prefix stability (I16), determinism (I15) and
 * the no-aliasing rule (I18) across every builder entry point the model test of
 * test/builder/model.test.ts does NOT drive: addEdges (index path), addEdgesByIds,
 * GraphBuilder.from() re-freezes, addGraph(), setDirected(true, { expand: true }), removals that
 * leave a sorted remainder, and arena on / off.
 *
 * Tests marked "PINS DEFECT" are expected to FAIL against the current implementation; each names
 * the finding it pins.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { equalsTopology } from "../../src/snapshot/graph-snapshot.js";
import { type GraphSnapshot } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";

const RUNS = Number(process.env.FC_RUNS ?? "150");

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

const arbWeight: fc.Arbitrary<number> = fc.constantFrom(1, 0, -1, 2.5, 0.1, 16777217, Infinity, -Infinity);

/** A random multigraph with self-loops, parallels in both orientations and mixed weights. */
const arbGraph: fc.Arbitrary<RawGraph> = fc
    .record({ directed: fc.boolean(), nodeCount: fc.integer({ min: 1, max: 7 }) })
    .chain(({ directed, nodeCount }) =>
        fc
            .array(
                fc.record({
                    u: fc.integer({ min: 0, max: nodeCount - 1 }),
                    v: fc.integer({ min: 0, max: nodeCount - 1 }),
                    w: fc.option(arbWeight, { nil: undefined }),
                }),
                { maxLength: 30 },
            )
            .map((edges) => ({ directed, nodeCount, edges })),
    );

/** An edge list that is already grouped by source and sorted by target (the identity path). */
const arbSortedGraph: fc.Arbitrary<RawGraph> = arbGraph.map((g) => ({
    ...g,
    edges: [...g.edges].sort((a, b) => a.u - b.u || a.v - b.v),
}));

function isSorted(edges: readonly RawEdge[]): boolean {
    for (let e = 1; e < edges.length; e++) {
        if (edges[e].u < edges[e - 1].u || (edges[e].u === edges[e - 1].u && edges[e].v < edges[e - 1].v)) {
            return false;
        }
    }
    return true;
}

/** The comparator-sorted rows a graph must produce: per node the [target, edge] pairs. */
function naiveRows(g: RawGraph): [number, number][][] {
    const rows: [number, number][][] = Array.from({ length: g.nodeCount }, () => []);
    g.edges.forEach((edge, e) => {
        rows[edge.u].push([edge.v, e]);
        if (!g.directed && edge.u !== edge.v) {
            rows[edge.v].push([edge.u, e]);
        }
    });
    for (const row of rows) {
        row.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    }
    return rows;
}

function snapshotRows(s: GraphSnapshot): [number, number][][] {
    const rows: [number, number][][] = [];
    for (let u = 0; u < s.nodeCount; u++) {
        const row: [number, number][] = [];
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            row.push([s.colIdx[a], s.arcToEdge[a]]);
        }
        rows.push(row);
    }
    return rows;
}

/** Every construction path of the same graph; every one must yield the same core. */
function constructions(g: RawGraph): [string, () => GraphSnapshot][] {
    const src = Uint32Array.from(g.edges, (e) => e.u);
    const dst = Uint32Array.from(g.edges, (e) => e.v);
    const anyWeight = g.edges.some((e) => e.w !== undefined);
    const weights = anyWeight ? Float64Array.from(g.edges, (e) => e.w ?? 1) : undefined;
    return [
        [
            "addEdgeByIndex one by one",
            () => {
                const b = new GraphBuilder({ directed: g.directed });
                b.addAnonymousNodes(g.nodeCount);
                for (const e of g.edges) {
                    b.addEdgeByIndex(e.u, e.v, e.w);
                }
                return b.freeze();
            },
        ],
        [
            "addEdges bulk (index path)",
            () => {
                const b = new GraphBuilder({ directed: g.directed });
                b.addAnonymousNodes(g.nodeCount);
                b.addEdges(src, dst, weights);
                return b.freeze();
            },
        ],
        [
            "addEdgesByIds bulk (id path)",
            () => {
                const b = new GraphBuilder({ directed: g.directed });
                b.addAnonymousNodes(g.nodeCount);
                b.addEdgesByIds(
                    Array.from(src),
                    Array.from(dst),
                    weights === undefined ? undefined : Array.from(weights),
                );
                return b.freeze();
            },
        ],
        [
            "addEdge by numeric id (id path, ids equal to indices) then arena: false",
            () => {
                const b = new GraphBuilder({ directed: g.directed });
                for (let i = 0; i < g.nodeCount; i++) {
                    b.addNode(i);
                }
                for (const e of g.edges) {
                    b.addEdge(e.u, e.v, e.w);
                }
                return b.freeze({ arena: false });
            },
        ],
    ];
}

describe("I4 / I14: every construction path yields the comparator-sorted rows with the (target, edge) tie-break", () => {
    it("addEdgeByIndex, addEdges, addEdgesByIds and addEdge agree with each other and with the naive sort", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const expected = naiveRows(g);
                const snapshots = constructions(g).map(([name, build]) => {
                    const s = build();
                    assertInvariants(s);
                    expect(snapshotRows(s), name).toEqual(expected);
                    expect(s.flags.arcToEdgeIsIdentity, name).toBe(g.directed && isSorted(g.edges));
                    g.edges.forEach((e, i) => {
                        expect(s.edgeSource(i), `${name} edgeSource(${i})`).toBe(e.u);
                        expect(s.edgeTarget(i), `${name} edgeTarget(${i})`).toBe(e.v);
                    });
                    return s;
                });
                for (const s of snapshots.slice(1)) {
                    expect(equalsTopology(snapshots[0], s)).toBe(true);
                    expect(s.contentHash()).toBe(snapshots[0].contentHash());
                    expect(s.flags).toEqual(snapshots[0].flags);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("directed input that happens to be sorted takes the identity path and the lazily materialised permutation IS the identity", () => {
        fc.assert(
            fc.property(arbSortedGraph, (g) => {
                const [, build] = constructions(g)[0];
                const s = build();
                assertInvariants(s);
                expect(s.flags.arcToEdgeIsIdentity).toBe(g.directed);
                if (g.directed) {
                    expect(s.arena?.segments.arcToEdge ?? null).toBeNull();
                    expect(s.arena?.segments.edgeToArc ?? null).toBeNull();
                    expect(s.byteLength()).toBe(
                        s.rowPtr.byteLength + s.colIdx.byteLength + (s.weights?.byteLength ?? 0),
                    );
                    const before = s.byteLength();
                    expect(Array.from(s.arcToEdge)).toEqual(g.edges.map((_, i) => i));
                    expect(Array.from(s.edgeToArc)).toEqual(g.edges.map((_, i) => i));
                    // touching the getters never changes the resident size or the wire bytes (I15, P11)
                    expect(s.byteLength()).toBe(before);
                    expect(s.arcToEdge.buffer).not.toBe(s.arena?.buffer);
                } else {
                    // an undirected graph never reports identity, even when every edge is a self-loop
                    expect(s.arena?.segments.arcToEdge === null).toBe(s.arcCount === 0);
                }
                expect(snapshotRows(s)).toEqual(naiveRows(g));
            }),
            { numRuns: RUNS },
        );
    });

    it("a directed edge list that is sorted except for one late swap is NOT identity, and its rows are still sorted", () => {
        fc.assert(
            fc.property(
                arbSortedGraph.filter((g) => g.directed && g.edges.length >= 2),
                (g) => {
                    const edges = [...g.edges];
                    // swap the last two edges when they differ; otherwise the list stays sorted
                    const last = edges.length - 1;
                    [edges[last - 1], edges[last]] = [edges[last], edges[last - 1]];
                    const swapped = { ...g, edges };
                    const s = constructions(swapped)[0][1]();
                    assertInvariants(s);
                    expect(s.flags.arcToEdgeIsIdentity).toBe(isSorted(edges));
                    expect(snapshotRows(s)).toEqual(naiveRows(swapped));
                },
            ),
            { numRuns: RUNS },
        );
    });
});

describe("I16 / identity detection across re-freezes", () => {
    it("removing edges so that the remainder is sorted flips the identity flag on the next freeze without touching the earlier snapshot", () => {
        fc.assert(
            fc.property(
                arbGraph.filter((g) => g.directed),
                fc.array(fc.nat({ max: 40 }), { maxLength: 12 }),
                (g, kills) => {
                    const b = new GraphBuilder({ directed: true });
                    b.addAnonymousNodes(g.nodeCount);
                    for (const e of g.edges) {
                        b.addEdgeByIndex(e.u, e.v, e.w);
                    }
                    const first = b.freeze({ checksum: true });
                    const firstHash = first.contentHash();
                    const firstFlag = first.flags.arcToEdgeIsIdentity;
                    const removed = new Set<number>();
                    for (const k of kills) {
                        if (g.edges.length > 0) {
                            const e = k % g.edges.length;
                            if (b.removeEdge(e)) {
                                removed.add(e);
                            }
                        }
                    }
                    const remaining = g.edges.filter((_, e) => !removed.has(e));
                    const { snapshot: second, report } = b.freezeWithReport();
                    assertInvariants(second);
                    expect(second.flags.arcToEdgeIsIdentity).toBe(isSorted(remaining));
                    expect(snapshotRows(second)).toEqual(naiveRows({ ...g, edges: remaining }));
                    // I16: the report says exactly whether the edge space was renumbered
                    expect(report.edgeRemap === null).toBe(removed.size === 0);
                    expect(report.nodeRemap).toBeNull();
                    if (report.edgeRemap !== null) {
                        let next = 0;
                        g.edges.forEach((_, e) => {
                            expect(report.edgeRemap?.[e]).toBe(removed.has(e) ? INVALID_INDEX : next++);
                        });
                    }
                    // I17 / I18: the first snapshot is byte-identical to what it was
                    first.validate({ level: "full", checksum: true });
                    expect(first.contentHash()).toBe(firstHash);
                    expect(first.flags.arcToEdgeIsIdentity).toBe(firstFlag);
                },
            ),
            { numRuns: RUNS },
        );
    });

    it("appending edges after a freeze keeps the earlier indices as a prefix (I16) and both snapshots stay valid", () => {
        fc.assert(
            fc.property(arbGraph, arbGraph, (g, extra) => {
                const nodeCount = Math.max(g.nodeCount, extra.nodeCount);
                const b = new GraphBuilder({ directed: g.directed });
                b.addAnonymousNodes(nodeCount);
                for (const e of g.edges) {
                    b.addEdgeByIndex(e.u, e.v, e.w);
                }
                const first = b.freeze();
                for (const e of extra.edges) {
                    b.addEdgeByIndex(e.u, e.v, e.w);
                }
                const { snapshot: second, report } = b.freezeWithReport();
                assertInvariants(first);
                assertInvariants(second);
                expect(report.nodeRemap).toBeNull();
                expect(report.edgeRemap).toBeNull();
                expect(report.compacted).toBe(false);
                for (let e = 0; e < first.edgeCount; e++) {
                    expect(second.edgeSource(e)).toBe(first.edgeSource(e));
                    expect(second.edgeTarget(e)).toBe(first.edgeTarget(e));
                }
                expect(snapshotRows(second)).toEqual(
                    naiveRows({ directed: g.directed, nodeCount, edges: [...g.edges, ...extra.edges] }),
                );
            }),
            { numRuns: RUNS },
        );
    });
});

describe("GraphBuilder.from() and addGraph() round trips (design section 6.6, I14, I15)", () => {
    it("from(snapshot).freeze() reproduces the topology, orientation, weights and flags of a builder-made snapshot", () => {
        fc.assert(
            fc.property(arbGraph, fc.constantFrom<"f32" | "f64">("f32", "f64"), (g, weightDtype) => {
                const b = new GraphBuilder({ directed: g.directed, weightDtype });
                b.addAnonymousNodes(g.nodeCount);
                for (const e of g.edges) {
                    b.addEdgeByIndex(e.u, e.v, e.w);
                }
                const source = b.freeze();
                const again = GraphBuilder.from(source).freeze();
                assertInvariants(again);
                expect(equalsTopology(source, again)).toBe(true);
                expect(again.flags).toEqual(source.flags);
                const shadowA = source.edges.byRole("weight");
                const shadowB = again.edges.byRole("weight");
                expect(shadowB === null).toBe(shadowA === null);
                if (shadowA !== null && shadowB !== null) {
                    expect(shadowB.dtype).toBe(shadowA.dtype);
                    expect(shadowB.nullCount).toBe(shadowA.nullCount);
                    for (let e = 0; e < source.edgeCount; e++) {
                        expect(shadowB.isSet(e)).toBe(shadowA.isSet(e));
                        expect(Object.is(shadowB.value(e), shadowA.value(e))).toBe(true);
                    }
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("PINS DEFECT (finding: from() drops a declared weights array): a weighted: true snapshot whose edges all omitted the weight loses `weights` across from().freeze()", () => {
        const b = new GraphBuilder({ directed: true, weighted: true });
        b.addEdge(0, 1);
        b.addEdge(1, 2);
        const source = b.freeze();
        expect(source.flags.weighted).toBe(true);
        expect(source.edges.byRole("weight")?.nullCount).toBe(2);
        const again = GraphBuilder.from(source).freeze();
        // design 3.7: "the format never drops a declared array"
        expect(again.flags.weighted).toBe(true);
        expect(equalsTopology(source, again)).toBe(true);
    });

    it("addGraph() of two disjoint snapshots appends edges in logical order with their declared orientation", () => {
        fc.assert(
            fc.property(
                arbGraph,
                arbGraph.filter((h) => h.edges.length > 0),
                (g, h) => {
                    const { directed } = g;
                    const make = (raw: RawGraph, prefix: string): GraphSnapshot => {
                        const b = new GraphBuilder({ directed });
                        for (let i = 0; i < raw.nodeCount; i++) {
                            b.addNode(`${prefix}${i}`);
                        }
                        for (const e of raw.edges) {
                            b.addEdge(`${prefix}${e.u}`, `${prefix}${e.v}`, e.w);
                        }
                        return b.freeze();
                    };
                    const a = make(g, "a");
                    const c = make(h, "c");
                    const union = new GraphBuilder({ directed });
                    union.addGraph(a);
                    union.addGraph(c);
                    const s = union.freeze();
                    assertInvariants(s);
                    expect(s.nodeCount).toBe(a.nodeCount + c.nodeCount);
                    expect(s.edgeCount).toBe(a.edgeCount + c.edgeCount);
                    const offset = a.nodeCount;
                    g.edges.forEach((e, i) => {
                        expect(s.edgeSource(i)).toBe(e.u);
                        expect(s.edgeTarget(i)).toBe(e.v);
                    });
                    h.edges.forEach((e, i) => {
                        expect(s.edgeSource(a.edgeCount + i)).toBe(offset + e.u);
                        expect(s.edgeTarget(a.edgeCount + i)).toBe(offset + e.v);
                    });
                    const combined: RawGraph = {
                        directed,
                        nodeCount: s.nodeCount,
                        edges: [...g.edges, ...h.edges.map((e) => ({ u: e.u + offset, v: e.v + offset, w: e.w }))],
                    };
                    expect(snapshotRows(s)).toEqual(naiveRows(combined));
                },
            ),
            { numRuns: Math.ceil(RUNS / 2) },
        );
    });
});

describe("setDirected(true, { expand: true }) (design sections 3.6 and 6.6)", () => {
    it("mirrors every non-loop edge at index E + k, links the halves through graphty.pair, keeps I4 and yields a symmetric directed snapshot", () => {
        fc.assert(
            fc.property(
                arbGraph.filter((g) => !g.directed),
                (g) => {
                    const b = new GraphBuilder({ directed: false });
                    b.addAnonymousNodes(g.nodeCount);
                    for (const e of g.edges) {
                        b.addEdgeByIndex(e.u, e.v, e.w);
                    }
                    const undirected = b.freeze();
                    b.setDirected(true, { expand: true });
                    const s = b.freeze();
                    assertInvariants(s);
                    expect(s.directed).toBe(true);
                    const loops = g.edges.filter((e) => e.u === e.v).length;
                    expect(s.edgeCount).toBe(2 * g.edges.length - loops);
                    if (!s.flags.multigraph) {
                        // on a multigraph isSymmetric() is positional (STATUS.md known gap); see the pinned test below
                        expect(s.isSymmetric()).toBe(true);
                    }
                    const pair = s.edges.byRole("pair");
                    const directed = s.edges.byRole("directed");
                    if (g.edges.length === 0) {
                        // an empty builder simply changes direction (design 6.6): no reserved columns
                        expect(pair).toBeNull();
                        expect(directed).toBeNull();
                        return;
                    }
                    expect(pair).not.toBeNull();
                    expect(directed).not.toBeNull();
                    const pairs = pair as NonNullable<typeof pair>;
                    let mirror = g.edges.length;
                    g.edges.forEach((e, i) => {
                        expect(s.edgeSource(i)).toBe(e.u);
                        expect(s.edgeTarget(i)).toBe(e.v);
                        expect(directed?.value(i)).toBe(false);
                        if (e.u === e.v) {
                            expect(pairs.isSet(i)).toBe(false);
                            return;
                        }
                        expect(pairs.value(i)).toBe(mirror);
                        expect(pairs.value(mirror)).toBe(i);
                        expect(s.edgeSource(mirror)).toBe(e.v);
                        expect(s.edgeTarget(mirror)).toBe(e.u);
                        if (s.weights !== null && undirected.weights !== null) {
                            expect(s.weights[s.edgeToArc[mirror]]).toBe(undirected.weights[undirected.edgeToArc[i]]);
                        }
                        mirror++;
                    });
                    // the doubled arc multiset of the undirected snapshot equals the arc multiset of the expansion
                    const arcs = (t: GraphSnapshot): string[] => {
                        const out: string[] = [];
                        for (let u = 0; u < t.nodeCount; u++) {
                            for (let a = t.rowPtr[u]; a < t.rowPtr[u + 1]; a++) {
                                out.push(`${u}>${t.colIdx[a]}`);
                            }
                        }
                        return out;
                    };
                    expect(arcs(s)).toEqual(arcs(undirected));
                },
            ),
            { numRuns: RUNS },
        );
    });
});

describe("isSymmetric() on an expanded undirected multigraph (design sections 3.6 and 7.2)", () => {
    it("PINS DEFECT (finding: isSymmetric is positional on multigraphs): the expansion of an undirected multigraph with unequal parallel weights is closed under reversal with equal weights but reports false", () => {
        const b = new GraphBuilder({ directed: false });
        b.addAnonymousNodes(4);
        b.addEdgeByIndex(1, 3, -Infinity);
        b.addEdgeByIndex(3, 1);
        b.setDirected(true, { expand: true });
        const s = b.freeze();
        assertInvariants(s);
        // every arc u -> v with weight w has an arc v -> u with weight w (multiset closure under reversal)
        const multiset = (t: GraphSnapshot, flip: boolean): string[] => {
            const out: string[] = [];
            for (let u = 0; u < t.nodeCount; u++) {
                for (let a = t.rowPtr[u]; a < t.rowPtr[u + 1]; a++) {
                    const v = t.colIdx[a];
                    const w = t.weights === null ? 1 : t.weights[a];
                    out.push(flip ? `${v}>${u}:${w}` : `${u}>${v}:${w}`);
                }
            }
            return out.sort();
        };
        expect(multiset(s, false)).toEqual(multiset(s, true));
        expect(s.isSymmetric()).toBe(true);
    });
});

describe("I15 / I18: determinism and no aliasing", () => {
    it("arena: true and arena: false give byte-identical cores and the same content hash", () => {
        fc.assert(
            fc.property(arbGraph, (g) => {
                const build = (arena: boolean): GraphSnapshot => {
                    const b = new GraphBuilder({ directed: g.directed });
                    b.addAnonymousNodes(g.nodeCount);
                    for (const e of g.edges) {
                        b.addEdgeByIndex(e.u, e.v, e.w);
                    }
                    return b.freeze({ arena });
                };
                const a = build(true);
                const c = build(false);
                expect(a.arena).not.toBeNull();
                expect(c.arena).toBeNull();
                expect(equalsTopology(a, c)).toBe(true);
                expect(a.contentHash()).toBe(c.contentHash());
                expect(a.flags).toEqual(c.flags);
            }),
            { numRuns: RUNS },
        );
    });

    it("mutating the builder after a freeze never changes the snapshot (checksum-verified) and the shared id map never exposes later ids", () => {
        fc.assert(
            fc.property(arbGraph, arbGraph, (g, extra) => {
                const b = new GraphBuilder({ directed: g.directed });
                for (let i = 0; i < g.nodeCount; i++) {
                    b.addNode(`n${i}`);
                }
                for (const e of g.edges) {
                    b.addEdge(`n${e.u}`, `n${e.v}`, e.w);
                }
                const s = b.freeze({ checksum: true, prepare: ["reverse", "coo", "edgeList", "degreeOrder"] });
                const hash = s.contentHash();
                const ids = s.ids.toArray();
                for (let i = 0; i < extra.nodeCount; i++) {
                    b.addNode(`x${i}`);
                }
                for (const e of extra.edges) {
                    b.addEdge(`x${e.u}`, `x${e.v}`, e.w);
                }
                if (g.edges.length > 0) {
                    b.setEdgeWeight(0, 99);
                    b.removeEdge(0);
                }
                b.freeze();
                s.validate({ level: "full", checksum: true });
                expect(s.contentHash()).toBe(hash);
                expect(s.ids.toArray()).toEqual(ids);
                expect(s.ids.size).toBe(g.nodeCount);
                expect(s.ids.indexOf("x0")).toBe(INVALID_INDEX);
                expect(s.ids.idsSlice(0)).toEqual(ids);
                expect([...s.ids.entries(new Array(s.nodeCount).fill(0))].length).toBe(g.nodeCount);
            }),
            { numRuns: Math.ceil(RUNS / 2) },
        );
    });
});
