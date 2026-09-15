/**
 * Differential tests against the legacy `Graph` class, part 4 (design section 16.2): the derived
 * graphs of section 7.3 and the builder composition entry points of section 6.6 compared with the
 * legacy class configured to the equivalent policy:
 *
 * - `toUndirected()` of a directed simple graph vs a legacy UNDIRECTED graph fed the same edge log
 *   with `allowParallelEdges: false` (the legacy rejects the second half of a reciprocal pair, which
 *   is the keep-first / weights "first" rule of 7.3);
 * - `simplified()` of a multigraph vs a legacy graph with `allowParallelEdges: false` (the legacy
 *   keeps the FIRST parallel, the survivor rule of 7.3), with and without self-loops;
 * - `withoutSelfLoops()` vs `allowSelfLoops: false`;
 * - `transpose()` vs the legacy graph fed the swapped endpoints;
 * - `inducedSubgraph({ mask })` vs legacy `removeNode` of the unselected ids;
 * - `filterEdges(keep)` vs legacy `removeEdge` of the dropped edges;
 * - `GraphBuilder.from(snapshot)` followed by more operations vs the legacy graph continuing;
 * - `addGraph(snapshot)` vs the legacy graph fed both logs.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type DerivedGraph, type GraphSnapshot, type NodeId } from "../../src/types/index.js";
import { makeMask, maskSet } from "../../src/util/mask.js";
import { assertInvariants } from "../helpers/invariants.js";
import { LegacyGraph } from "../helpers/legacy-graph.js";

const RUNS = Number(process.env.FC_RUNS ?? "60");

// ============================================================ generators (self-contained)

type IdKind = "ints" | "strings" | "mixed";

function idOf(kind: IdKind, i: number): NodeId {
    switch (kind) {
        case "ints":
            return i;
        case "strings":
            return `n${i}`;
        case "mixed":
            return i % 2 === 0 ? i / 2 : `${(i - 1) / 2}`;
        default:
            throw new Error(`unknown kind ${String(kind)}`);
    }
}

const WEIGHTS: readonly (number | undefined)[] = [undefined, 1, 0, 2, 0.5, 0.1, -3, 7, Infinity];

interface Log {
    readonly source: NodeId;
    readonly target: NodeId;
    readonly weight: number | undefined;
}

interface DerivedCase {
    readonly directed: boolean;
    readonly idKind: IdKind;
    readonly n: number;
    readonly log: readonly Log[];
    readonly extra: readonly Log[];
    /** Node selection bits for inducedSubgraph and edge keep bits for filterEdges. */
    readonly keepNodes: readonly boolean[];
    readonly keepEdges: readonly boolean[];
}

const derivedCaseArb: fc.Arbitrary<DerivedCase> = fc
    .record({
        directed: fc.boolean(),
        idKind: fc.constantFrom<IdKind>("ints", "strings", "mixed"),
        n: fc.oneof(
            { weight: 5, arbitrary: fc.integer({ min: 1, max: 12 }) },
            { weight: 2, arbitrary: fc.integer({ min: 13, max: 300 }) },
            { weight: 1, arbitrary: fc.integer({ min: 1500, max: 2000 }) },
        ),
    })
    .chain((head) => {
        const index = fc.nat({ max: head.n - 1 });
        const edge = fc
            .tuple(index, index, fc.constantFrom(...WEIGHTS))
            .map(([s, t, w]) => ({ source: idOf(head.idKind, s), target: idOf(head.idKind, t), weight: w }));
        const maxEdges = Math.min(4000, 3 * head.n);
        return fc.record({
            head: fc.constant(head),
            log: fc.array(edge, { maxLength: maxEdges, size: "max" }),
            extra: fc.array(edge, { maxLength: Math.min(200, head.n + 2), size: "max" }),
            keepNodes: fc.array(fc.boolean(), { minLength: head.n, maxLength: head.n, size: "max" }),
            keepEdges: fc.array(fc.boolean(), { minLength: maxEdges, maxLength: maxEdges, size: "max" }),
        });
    })
    .map(({ head, log, extra, keepNodes, keepEdges }) => ({ ...head, log, extra, keepNodes, keepEdges }));

function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

/** Feed a log to a legacy graph, skipping what its policy refuses; returns the accepted log. */
function feedLegacy(legacy: LegacyGraph, log: readonly Log[]): Log[] {
    const accepted: Log[] = [];
    for (const e of log) {
        try {
            legacy.addEdge(e.source, e.target, e.weight);
        } catch {
            continue;
        }
        accepted.push(e);
    }
    return accepted;
}

/** Feed a log to a builder (every edge accepted; nodes created in legacy order). */
function feedBuilder(builder: GraphBuilder, log: readonly Log[]): void {
    for (const e of log) {
        builder.addNode(e.source);
        builder.addNode(e.target);
        builder.addEdge(e.source, e.target, e.weight);
    }
}

/** Compare a snapshot with a legacy graph: node order, distinct neighbour sets, hasEdge, first-arc weight. */
function compare(
    legacy: LegacyGraph,
    s: GraphSnapshot,
    options: { weights?: boolean; parallels?: boolean } = {},
): void {
    assertInvariants(s);
    expect(s.directed).toBe(legacy.isDirected);
    expect(s.nodeCount).toBe(legacy.nodeCount);
    expect(s.ids.toArray()).toEqual([...legacy.nodes()].map((node) => node.id));
    if (options.parallels !== true) {
        expect(s.edgeCount).toBe(legacy.totalEdgeCount);
        expect(s.flags.multigraph).toBe(false);
    }
    const outDegree = s.outDegree();
    const reverse = s.reverse();
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        const row = new Set<string>();
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            row.add(keyOf(s.ids.idOf(s.colIdx[a])));
        }
        expect(row).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
        const inRow = new Set<string>();
        for (let a = reverse.rowPtr[u]; a < reverse.rowPtr[u + 1]; a++) {
            inRow.add(keyOf(s.ids.idOf(reverse.colIdx[a])));
        }
        expect(inRow).toEqual(new Set([...legacy.inNeighbors(id)].map(keyOf)));
        if (options.parallels !== true) {
            expect(outDegree[u]).toBe(legacy.outDegree(id));
        }
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            expect(legacy.hasEdge(id, s.ids.idOf(s.colIdx[a]))).toBe(true);
        }
    }
    if (options.weights !== false) {
        for (const edge of legacy.edges()) {
            const u = s.ids.requireIndex(edge.source);
            const v = s.ids.requireIndex(edge.target);
            const arc = s.findArc(u, v);
            expect(arc).not.toBe(INVALID_INDEX);
            const w = s.weights === null ? 1 : s.weights[arc];
            expect(Object.is(w, Math.fround(edge.weight ?? 1))).toBe(true);
        }
    }
}

/** P9b: edgeRemap / edgeOrigin agree, and every source edge maps to a survivor with the same endpoints as a set. */
function checkEdgeMaps(source: GraphSnapshot, d: DerivedGraph): void {
    const { snapshot, edgeRemap, edgeOrigin } = d;
    if (edgeRemap === null || edgeOrigin === null) {
        expect(edgeRemap).toBeNull();
        expect(edgeOrigin).toBeNull();
        expect(snapshot.edgeCount).toBe(source.edgeCount);
        return;
    }
    expect(edgeRemap.length).toBe(source.edgeCount);
    expect(edgeOrigin.length).toBe(snapshot.edgeCount);
    for (let dEdge = 0; dEdge < snapshot.edgeCount; dEdge++) {
        expect(edgeRemap[edgeOrigin[dEdge]]).toBe(dEdge);
    }
    for (let e = 0; e < source.edgeCount; e++) {
        const t = edgeRemap[e];
        if (t === INVALID_INDEX) {
            continue;
        }
        // endpoints compared in the derived node space (identity when nodeRemap is null)
        const mapNode = (i: number): number => (d.nodeRemap === null ? i : d.nodeRemap[i]);
        const ends = new Set([mapNode(source.edgeSource(e)), mapNode(source.edgeTarget(e))]);
        const derivedEnds = new Set([snapshot.edgeSource(t), snapshot.edgeTarget(t)]);
        expect(derivedEnds).toEqual(ends);
    }
}

// ============================================================ derived graphs vs legacy policies

describe("differential: derived graphs vs the equivalent legacy policy (design section 7.3)", () => {
    it("toUndirected() of a directed simple graph equals the legacy undirected graph fed the same log", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                // the directed source: simple (legacy allowParallelEdges false decides acceptance)
                const directedLegacy = new LegacyGraph({ directed: true, allowParallelEdges: false });
                const accepted = feedLegacy(directedLegacy, c.log);
                const builder = new GraphBuilder({ directed: true });
                feedBuilder(builder, accepted);
                const source = builder.freeze();
                compare(directedLegacy, source);
                // the legacy undirected graph over the same accepted log rejects the reciprocal half
                const undirectedLegacy = new LegacyGraph({ directed: false, allowParallelEdges: false });
                feedLegacy(undirectedLegacy, accepted);
                const d = source.toUndirected();
                compare(undirectedLegacy, d.snapshot);
                checkEdgeMaps(source, d);
                expect(d.nodeRemap).toBeNull();
                expect(d.snapshot.nodes).toBe(source.nodes);
                // idempotent (P9)
                const again = d.snapshot.toUndirected();
                expect(again.snapshot).toBe(d.snapshot);
                expect(again.edgeRemap).toBeNull();
            }),
            { numRuns: RUNS },
        );
    });

    it("simplified() of a multigraph equals the legacy graph with allowParallelEdges: false (survivor = first)", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const builder = new GraphBuilder({ directed: c.directed });
                feedBuilder(builder, c.log);
                const source = builder.freeze();
                const simpleLegacy = new LegacyGraph({ directed: c.directed, allowParallelEdges: false });
                feedLegacy(simpleLegacy, c.log);
                const d = source.simplified();
                compare(simpleLegacy, d.snapshot);
                checkEdgeMaps(source, d);
                // and dropping loops on top of it
                const noLoops = new LegacyGraph({
                    directed: c.directed,
                    allowParallelEdges: false,
                    allowSelfLoops: false,
                });
                feedLegacy(noLoops, c.log);
                const d2 = source.simplified({ selfLoops: "drop" });
                compare(noLoops, d2.snapshot);
                expect(d2.snapshot.selfLoopCount).toBe(0);
                checkEdgeMaps(source, d2);
            }),
            { numRuns: RUNS },
        );
    });

    it("withoutSelfLoops() equals the legacy graph with allowSelfLoops: false", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const builder = new GraphBuilder({ directed: c.directed });
                feedBuilder(builder, c.log);
                const source = builder.freeze();
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowParallelEdges: true,
                    allowSelfLoops: false,
                });
                const accepted = feedLegacy(legacy, c.log);
                const d = source.withoutSelfLoops();
                compare(legacy, d.snapshot, { parallels: true, weights: false });
                expect(d.snapshot.edgeCount).toBe(accepted.length);
                expect(d.snapshot.selfLoopCount).toBe(0);
                expect(d.snapshot.flags.hasSelfLoops).toBe(false);
                checkEdgeMaps(source, d);
                expect(d.report.droppedEdges).toBe(c.log.length - accepted.length);
            }),
            { numRuns: RUNS },
        );
    });

    it("transpose() equals the legacy directed graph fed the swapped endpoints", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const forward = new LegacyGraph({ directed: true, allowParallelEdges: false });
                const accepted = feedLegacy(forward, c.log);
                const builder = new GraphBuilder({ directed: true });
                feedBuilder(builder, accepted);
                const source = builder.freeze();
                const swapped = new LegacyGraph({ directed: true, allowParallelEdges: false });
                // the legacy node order follows first appearance of the swapped log, which differs from
                // the source order; pre-add the nodes in source order so the orders are comparable
                for (const id of source.ids) {
                    swapped.addNode(id);
                }
                for (const e of accepted) {
                    swapped.addEdge(e.target, e.source, e.weight);
                }
                const d = source.transpose();
                compare(swapped, d.snapshot);
                expect(d.nodeRemap).toBeNull();
                expect(d.edgeRemap).toBeNull();
                for (let e = 0; e < source.edgeCount; e++) {
                    expect(d.snapshot.edgeSource(e)).toBe(source.edgeTarget(e));
                    expect(d.snapshot.edgeTarget(e)).toBe(source.edgeSource(e));
                }
                // twice is the identity (P9)
                const back = d.snapshot.transpose().snapshot;
                expect(Array.from(back.rowPtr)).toEqual(Array.from(source.rowPtr));
                expect(Array.from(back.colIdx)).toEqual(Array.from(source.colIdx));
            }),
            { numRuns: RUNS },
        );
    });

    it("inducedSubgraph({ mask }) equals the legacy graph after removeNode of every unselected id", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const legacy = new LegacyGraph({ directed: c.directed, allowParallelEdges: false });
                const accepted = feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed });
                feedBuilder(builder, accepted);
                const source = builder.freeze();
                const mask = makeMask(source.nodeCount);
                for (let i = 0; i < source.nodeCount; i++) {
                    const keep = c.keepNodes[i % c.keepNodes.length];
                    maskSet(mask, i, keep);
                    if (!keep) {
                        expect(legacy.removeNode(source.ids.idOf(i))).toBe(true);
                    }
                }
                const d = source.inducedSubgraph({ mask });
                compare(legacy, d.snapshot);
                // nodeOrigin / nodeRemap are inverse on the kept set; null only when nothing was dropped
                if (d.nodeRemap === null || d.nodeOrigin === null) {
                    expect(d.nodeRemap).toBeNull();
                    expect(d.nodeOrigin).toBeNull();
                    expect(legacy.nodeCount).toBe(source.nodeCount);
                } else {
                    for (let i = 0; i < source.nodeCount; i++) {
                        const j = d.nodeRemap[i];
                        if (j === INVALID_INDEX) {
                            expect(legacy.hasNode(source.ids.idOf(i))).toBe(false);
                        } else {
                            expect(d.nodeOrigin[j]).toBe(i);
                            expect(d.snapshot.ids.idOf(j)).toBe(source.ids.idOf(i));
                        }
                    }
                }
                checkEdgeMaps(source, d);
            }),
            { numRuns: RUNS },
        );
    });

    it("filterEdges(keep) equals the legacy graph after removeEdge of every dropped edge", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const legacy = new LegacyGraph({ directed: c.directed, allowParallelEdges: false });
                const accepted = feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed });
                feedBuilder(builder, accepted);
                const source = builder.freeze();
                const keep = makeMask(source.edgeCount);
                for (let e = 0; e < source.edgeCount; e++) {
                    const bit = c.keepEdges[e % c.keepEdges.length];
                    maskSet(keep, e, bit);
                    if (!bit) {
                        expect(legacy.removeEdge(accepted[e].source, accepted[e].target)).toBe(true);
                    }
                }
                const d = source.filterEdges(keep);
                compare(legacy, d.snapshot);
                expect(d.nodeRemap).toBeNull();
                checkEdgeMaps(source, d);
                // kept edges keep their relative order and orientation (I14)
                let next = 0;
                for (let e = 0; e < source.edgeCount; e++) {
                    if (c.keepEdges[e % c.keepEdges.length]) {
                        expect(d.edgeRemap === null ? e : d.edgeRemap[e]).toBe(next);
                        expect(d.snapshot.edgeSource(next)).toBe(source.edgeSource(e));
                        expect(d.snapshot.edgeTarget(next)).toBe(source.edgeTarget(e));
                        next++;
                    }
                }
            }),
            { numRuns: RUNS },
        );
    });
});

// ============================================================ builder composition (6.6)

describe("differential: GraphBuilder.from / addGraph vs a legacy graph fed both logs (design section 6.6)", () => {
    it("GraphBuilder.from(snapshot) continues like the legacy graph: indices preserved, new nodes appended", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const legacy = new LegacyGraph({ directed: c.directed, allowParallelEdges: true });
                feedLegacy(legacy, c.log);
                const first = new GraphBuilder({ directed: c.directed });
                feedBuilder(first, c.log);
                const source = first.freeze();
                const resumed = GraphBuilder.from(source);
                expect(resumed.nodeCount).toBe(source.nodeCount);
                expect(resumed.edgeCount).toBe(source.edgeCount);
                for (let i = 0; i < source.nodeCount; i++) {
                    expect(resumed.idOf(i)).toBe(source.ids.idOf(i));
                }
                for (let e = 0; e < source.edgeCount; e++) {
                    expect(resumed.edgeEndpoints(e)).toEqual([source.edgeSource(e), source.edgeTarget(e)]);
                }
                feedLegacy(legacy, c.extra);
                feedBuilder(resumed, c.extra);
                const s = resumed.freeze();
                compare(legacy, s, { parallels: true, weights: false });
                expect(s.edgeCount).toBe(legacy.totalEdgeCount);
                // prefix stability (I16): the resumed snapshot extends the source
                expect(s.ids.toArray().slice(0, source.nodeCount)).toEqual(source.ids.toArray());
                for (let e = 0; e < source.edgeCount; e++) {
                    expect(s.edgeSource(e)).toBe(source.edgeSource(e));
                    expect(s.edgeTarget(e)).toBe(source.edgeTarget(e));
                    const w = s.weights === null ? 1 : s.weights[s.edgeToArc[e]];
                    const sw = source.weights === null ? 1 : source.weights[source.edgeToArc[e]];
                    expect(Object.is(w, sw)).toBe(true);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("addGraph(snapshot) equals the legacy graph fed the first log then the second", () => {
        fc.assert(
            fc.property(derivedCaseArb, (c) => {
                const legacy = new LegacyGraph({ directed: c.directed, allowParallelEdges: true });
                feedLegacy(legacy, c.log);
                feedLegacy(legacy, c.extra);
                const a = new GraphBuilder({ directed: c.directed });
                feedBuilder(a, c.log);
                const b = new GraphBuilder({ directed: c.directed });
                feedBuilder(b, c.extra);
                const second = b.freeze();
                a.addGraph(second, { onDuplicateNode: "merge" });
                const s = a.freeze();
                compare(legacy, s, { parallels: true, weights: false });
                expect(s.edgeCount).toBe(legacy.totalEdgeCount);
                // every edge of the second graph is appended in order with its weight
                const offset = s.edgeCount - second.edgeCount;
                for (let e = 0; e < second.edgeCount; e++) {
                    expect(s.ids.idOf(s.edgeSource(offset + e))).toBe(second.ids.idOf(second.edgeSource(e)));
                    expect(s.ids.idOf(s.edgeTarget(offset + e))).toBe(second.ids.idOf(second.edgeTarget(e)));
                    const w = s.weights === null ? 1 : s.weights[s.edgeToArc[offset + e]];
                    const sw = second.weights === null ? 1 : second.weights[second.edgeToArc[e]];
                    expect(Object.is(w, sw)).toBe(true);
                }
            }),
            { numRuns: Math.max(10, Math.floor(RUNS / 2)) },
        );
    });
});
