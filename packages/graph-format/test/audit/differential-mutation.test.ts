/**
 * Differential tests against the legacy `Graph` class, part 3 (design section 16.2): random
 * removal sequences (`removeNode` by id, `removeEdge` by endpoint pair) applied to both structures,
 * followed by a freeze, must leave the same node set in the same order (I14 compaction preserves
 * relative order), the same neighbour sets, the same `hasEdge` answers and the same weights; the
 * `freezeWithReport` remaps must map every surviving old index to the index of the same id (P3).
 * The one documented divergence -- reviving a removed id before the next freeze keeps its OLD index
 * (design 6.6, NetworKit restoreNode semantics) where the legacy Map re-inserts it at the END -- is
 * pinned as intentional, and the post-compaction case (revive after a freeze) is shown to agree with
 * the legacy order again.
 *
 * Removal is restricted to simple graphs because the legacy class cannot hold parallel edges (its
 * `removeEdge` drops the only stored entry but decrements `totalEdgeCount` by one, so on a
 * multigraph its own counts are wrong).
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type GraphSnapshot, type NodeId } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { LegacyGraph } from "../helpers/legacy-graph.js";

const RUNS = Number(process.env.FC_RUNS ?? "60");

// ============================================================ generators (self-contained)

type IdKind = "ints" | "strings" | "mixed";

function idOf(kind: IdKind, i: number): NodeId {
    switch (kind) {
        case "ints":
            return i * 3 + 2;
        case "strings":
            return `v${i}`;
        case "mixed":
            // 0, "0", 1, "1", ...: a numeric id and a string id sharing the String() form
            return i % 2 === 0 ? i / 2 : `${(i - 1) / 2}`;
        default:
            throw new Error(`unknown kind ${String(kind)}`);
    }
}

type MutationOp =
    | { readonly type: "addEdge"; readonly s: number; readonly t: number; readonly w: number | undefined }
    | { readonly type: "removeNode"; readonly u: number }
    | { readonly type: "removeEdge"; readonly s: number; readonly t: number }
    | { readonly type: "freeze" };

interface MutationCase {
    readonly directed: boolean;
    readonly allowSelfLoops: boolean;
    readonly idKind: IdKind;
    readonly n: number;
    readonly ops: readonly MutationOp[];
}

const WEIGHTS: readonly (number | undefined)[] = [undefined, 1, 0, 2, 0.5, 0.1, -3, Infinity];

const mutationCaseArb: fc.Arbitrary<MutationCase> = fc
    .record({
        directed: fc.boolean(),
        allowSelfLoops: fc.boolean(),
        idKind: fc.constantFrom<IdKind>("ints", "strings", "mixed"),
        n: fc.oneof(
            { weight: 5, arbitrary: fc.integer({ min: 1, max: 12 }) },
            { weight: 2, arbitrary: fc.integer({ min: 13, max: 300 }) },
            { weight: 1, arbitrary: fc.integer({ min: 1500, max: 2000 }) },
        ),
    })
    .chain((head) => {
        const index = fc.nat({ max: head.n - 1 });
        const op: fc.Arbitrary<MutationOp> = fc.oneof(
            {
                weight: 6,
                arbitrary: fc.record({
                    type: fc.constant("addEdge" as const),
                    s: index,
                    t: index,
                    w: fc.constantFrom(...WEIGHTS),
                }),
            },
            { weight: 1, arbitrary: fc.record({ type: fc.constant("removeNode" as const), u: index }) },
            { weight: 2, arbitrary: fc.record({ type: fc.constant("removeEdge" as const), s: index, t: index }) },
            { weight: 1, arbitrary: fc.constant({ type: "freeze" as const }) },
        );
        return fc.record({
            head: fc.constant(head),
            ops: fc.array(op, { maxLength: Math.min(3000, 3 * head.n + 8), size: "max" }),
        });
    })
    .map(({ head, ops }) => ({ ...head, ops }));

function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

function legacyIds(legacy: LegacyGraph): NodeId[] {
    return [...legacy.nodes()].map((node) => node.id);
}

/** The cheap agreement check run at every intermediate freeze: counts and id order. */
function compareCounts(legacy: LegacyGraph, s: GraphSnapshot): void {
    expect(s.nodeCount).toBe(legacy.nodeCount);
    expect(s.edgeCount).toBe(legacy.totalEdgeCount);
    expect(s.ids.toArray()).toEqual(legacyIds(legacy));
}

/** Every structural answer both sides can give, compared after a freeze. */
function compare(legacy: LegacyGraph, s: GraphSnapshot, pool: readonly NodeId[]): void {
    assertInvariants(s);
    expect(s.nodeCount).toBe(legacy.nodeCount);
    expect(s.edgeCount).toBe(legacy.totalEdgeCount);
    expect(s.ids.toArray()).toEqual(legacyIds(legacy));
    const outDegree = s.outDegree();
    const degree = s.degree();
    const loops = s.selfLoopsPerNode();
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        expect(legacy.hasNode(id)).toBe(true);
        const row = new Set<string>();
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            row.add(keyOf(s.ids.idOf(s.colIdx[a])));
        }
        expect(row).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
        expect(outDegree[u]).toBe(legacy.outDegree(id));
        if (s.directed) {
            expect(degree[u]).toBe(legacy.degree(id));
        } else {
            expect(degree[u]).toBe(legacy.degree(id) + loops[u]);
        }
    }
    const checkPair = (a: NodeId, b: NodeId): void => {
        const u = s.ids.indexOf(a);
        const v = s.ids.indexOf(b);
        const legacyHas = legacy.hasEdge(a, b);
        if (u === INVALID_INDEX || v === INVALID_INDEX) {
            expect(legacyHas).toBe(false);
            return;
        }
        expect(s.hasArc(u, v)).toBe(legacyHas);
        if (legacyHas) {
            const w = s.weights === null ? 1 : s.weights[s.findArc(u, v)];
            expect(Object.is(w, Math.fround(legacy.getEdge(a, b)?.weight ?? Number.NaN))).toBe(true);
        }
    };
    for (const a of pool) {
        expect(s.ids.has(a)).toBe(legacy.hasNode(a));
    }
    // every ordered pair of a prefix of the pool (absent pairs included), then every legacy edge
    const prefix = pool.slice(0, 48);
    for (const a of prefix) {
        for (const b of prefix) {
            checkPair(a, b);
        }
    }
    for (const edge of legacy.edges()) {
        checkPair(edge.source, edge.target);
        checkPair(edge.target, edge.source);
    }
}

interface Driven {
    readonly legacy: LegacyGraph;
    readonly builder: GraphBuilder;
    readonly pool: readonly NodeId[];
    /** The snapshot of the last freeze op, when any. */
    readonly last: GraphSnapshot | null;
}

/**
 * Apply the operation log to both sides. removeNode is only applied when the legacy has the node
 * (its own removeNode returns false; the builder throws E_UNKNOWN_NODE); removeEdge resolves the
 * pair through findEdges (either orientation when undirected, like the legacy hasEdge check).
 */
function drive(c: MutationCase, freezeAtEnd: boolean): Driven {
    const pool = Array.from({ length: c.n }, (_, i) => idOf(c.idKind, i));
    const legacy = new LegacyGraph({
        directed: c.directed,
        allowSelfLoops: c.allowSelfLoops,
        allowParallelEdges: false,
    });
    const builder = new GraphBuilder({ directed: c.directed });
    let last: GraphSnapshot | null = null;
    // ids removed since the last freeze: re-adding one would REVIVE its old index (design 6.6), which
    // the legacy Map cannot mirror (it re-appends); the intentional divergence is pinned by its own
    // test below, so the random driver compacts first (a freeze) and both sides then append.
    const tombstoned = new Set<NodeId>();
    for (const op of c.ops) {
        switch (op.type) {
            case "addEdge": {
                const source = pool[op.s];
                const target = pool[op.t];
                if (tombstoned.has(source) || tombstoned.has(target)) {
                    last = builder.freeze();
                    compareCounts(legacy, last);
                    tombstoned.clear();
                }
                builder.addNode(source);
                builder.addNode(target);
                try {
                    legacy.addEdge(source, target, op.w);
                } catch {
                    break;
                }
                builder.addEdge(source, target, op.w);
                break;
            }
            case "removeNode": {
                const id = pool[op.u];
                const legacyEdgesBefore = legacy.totalEdgeCount;
                const removed = legacy.removeNode(id);
                if (removed) {
                    const removedEdges = builder.removeNode(id);
                    expect(removedEdges.length).toBe(legacyEdgesBefore - legacy.totalEdgeCount);
                    tombstoned.add(id);
                } else {
                    expect(builder.hasNode(id)).toBe(false);
                }
                break;
            }
            case "removeEdge": {
                const source = pool[op.s];
                const target = pool[op.t];
                const removed = legacy.removeEdge(source, target);
                const u = builder.indexOf(source);
                const v = builder.indexOf(target);
                if (u === INVALID_INDEX || v === INVALID_INDEX) {
                    expect(removed).toBe(false);
                    break;
                }
                const edges = builder.findEdges(u, v);
                expect(edges.length > 0).toBe(removed);
                if (removed) {
                    expect(edges.length).toBe(1);
                    expect(builder.removeEdge(edges[0])).toBe(true);
                    expect(builder.hasEdge(edges[0])).toBe(false);
                }
                break;
            }
            case "freeze": {
                last = builder.freeze();
                compareCounts(legacy, last);
                tombstoned.clear();
                break;
            }
            default:
                throw new Error("unreachable");
        }
        expect(builder.nodeCount).toBe(legacy.nodeCount);
        expect(builder.edgeCount).toBe(legacy.totalEdgeCount);
    }
    if (freezeAtEnd) {
        last = builder.freeze();
        compare(legacy, last, pool);
    }
    return { legacy, builder, pool, last };
}

// ============================================================ the properties

describe("differential: removals, compaction and index order (design sections 4.4, 6.6, I14, I16)", () => {
    it("after any sequence of adds / removes / freezes both structures agree (simple graphs)", () => {
        fc.assert(
            fc.property(mutationCaseArb, (c) => {
                drive(c, true);
            }),
            { numRuns: RUNS },
        );
    });

    it("freezeWithReport remaps carry every surviving old index to the index of the same id (P3)", () => {
        fc.assert(
            fc.property(mutationCaseArb, (c) => {
                // the log without freezes, split in two halves: freeze, mutate, freezeWithReport
                const ops = c.ops.filter((op) => op.type !== "freeze");
                const half = Math.floor(ops.length / 2);
                const first = drive({ ...c, ops: ops.slice(0, half) }, true);
                const before = first.last;
                expect(before).not.toBeNull();
                if (before === null) {
                    return;
                }
                const oldIds = before.ids.toArray();
                const oldEdges: [NodeId, NodeId][] = [];
                const list = before.edgeList();
                for (let e = 0; e < before.edgeCount; e++) {
                    oldEdges.push([before.ids.idOf(list.src[e]), before.ids.idOf(list.dst[e])]);
                }
                // replay the second half against the same builder and legacy
                const { legacy, builder, pool } = first;
                let removedSomething = false;
                const tombstoned = new Set<NodeId>();
                for (const op of ops.slice(half)) {
                    switch (op.type) {
                        case "addEdge": {
                            const source = pool[op.s];
                            const target = pool[op.t];
                            if (tombstoned.has(source) || tombstoned.has(target)) {
                                // a revival would keep the old index (6.6): skip such ops here so the
                                // report is measured against one uninterrupted freeze interval
                                break;
                            }
                            builder.addNode(source);
                            builder.addNode(target);
                            try {
                                legacy.addEdge(source, target, op.w);
                            } catch {
                                break;
                            }
                            builder.addEdge(source, target, op.w);
                            break;
                        }
                        case "removeNode": {
                            const id = pool[op.u];
                            if (legacy.removeNode(id)) {
                                builder.removeNode(id);
                                removedSomething = true;
                                tombstoned.add(id);
                            }
                            break;
                        }
                        case "removeEdge": {
                            const source = pool[op.s];
                            const target = pool[op.t];
                            if (legacy.removeEdge(source, target)) {
                                const edges = builder.findEdges(builder.indexOf(source), builder.indexOf(target));
                                expect(edges.length).toBe(1);
                                builder.removeEdge(edges[0]);
                                removedSomething = true;
                            }
                            break;
                        }
                        default:
                            break;
                    }
                }
                const { snapshot, report } = builder.freezeWithReport();
                compare(legacy, snapshot, pool);
                if (!removedSomething) {
                    // I16 prefix stability: no renumbering
                    expect(report.nodeRemap).toBeNull();
                    expect(report.edgeRemap).toBeNull();
                    expect(snapshot.ids.toArray().slice(0, oldIds.length)).toEqual(oldIds);
                    return;
                }
                // a removal happened: every old index maps to the index holding the same id, or to
                // INVALID_INDEX exactly when the id is gone (a removed-then-revived id keeps its index)
                if (report.nodeRemap !== null) {
                    // 4.4: "old" is the previous freeze's index space PLUS anything appended since
                    expect(report.nodeRemap.length).toBeGreaterThanOrEqual(oldIds.length);
                    for (let i = oldIds.length; i < report.nodeRemap.length; i++) {
                        const j = report.nodeRemap[i];
                        expect(j === INVALID_INDEX || j < snapshot.nodeCount).toBe(true);
                    }
                    for (let i = 0; i < oldIds.length; i++) {
                        const j = report.nodeRemap[i];
                        if (j === INVALID_INDEX) {
                            expect(legacy.hasNode(oldIds[i])).toBe(false);
                        } else {
                            expect(snapshot.ids.idOf(j)).toBe(oldIds[i]);
                        }
                    }
                } else {
                    for (let i = 0; i < oldIds.length; i++) {
                        expect(snapshot.ids.idOf(i)).toBe(oldIds[i]);
                    }
                }
                if (report.edgeRemap !== null) {
                    expect(report.edgeRemap.length).toBeGreaterThanOrEqual(oldEdges.length);
                    for (let e = 0; e < oldEdges.length; e++) {
                        const d = report.edgeRemap[e];
                        if (d === INVALID_INDEX) {
                            continue;
                        }
                        expect(snapshot.ids.idOf(snapshot.edgeSource(d))).toBe(oldEdges[e][0]);
                        expect(snapshot.ids.idOf(snapshot.edgeTarget(d))).toBe(oldEdges[e][1]);
                    }
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("INTENTIONAL (6.6): reviving a removed id before the next freeze keeps its old index; the legacy Map re-appends it", () => {
        const legacy = new LegacyGraph({ directed: true });
        const b = new GraphBuilder({ directed: true });
        for (const [s, t] of [
            ["a", "b"],
            ["b", "c"],
            ["c", "a"],
        ] as const) {
            legacy.addEdge(s, t);
            b.addEdge(s, t);
        }
        expect(legacy.removeNode("a")).toBe(true);
        expect(Array.from(b.removeNode("a")).sort()).toEqual([0, 2]);
        legacy.addEdge("a", "c");
        b.addEdge("a", "c");
        const s = b.freeze();
        // same node SET, same edges ...
        expect(new Set(s.ids.toArray())).toEqual(new Set(legacyIds(legacy)));
        expect(s.edgeCount).toBe(legacy.totalEdgeCount);
        expect(s.hasArc(s.ids.requireIndex("a"), s.ids.requireIndex("c"))).toBe(true);
        // ... but the ORDER differs by design: NetworKit restoreNode semantics keep index 0 for "a"
        expect(legacyIds(legacy)).toEqual(["b", "c", "a"]);
        expect(s.ids.toArray()).toEqual(["a", "b", "c"]);
        // the edge index space compacted: the surviving edge b -> c became edge 0, a -> c edge 1
        expect(s.ids.idOf(s.edgeSource(0))).toBe("b");
        expect(s.ids.idOf(s.edgeSource(1))).toBe("a");
    });

    it("after a compacting freeze a re-added id gets a NEW index at the end, matching the legacy order", () => {
        const legacy = new LegacyGraph({ directed: false });
        const b = new GraphBuilder({ directed: false });
        for (const [s, t] of [
            ["a", "b"],
            ["b", "c"],
            ["c", "a"],
        ] as const) {
            legacy.addEdge(s, t);
            b.addEdge(s, t);
        }
        legacy.removeNode("a");
        b.removeNode("a");
        const mid = b.freeze();
        expect(mid.ids.toArray()).toEqual(legacyIds(legacy));
        legacy.addEdge("a", "c", 4);
        b.addEdge("a", "c", 4);
        const s = b.freeze();
        compare(legacy, s, ["a", "b", "c"]);
        expect(s.ids.toArray()).toEqual(["b", "c", "a"]);
    });

    it("removeNode returns exactly the incident edges the legacy removeNode dropped, in both directions", () => {
        fc.assert(
            fc.property(mutationCaseArb, (c) => {
                const adds = c.ops.filter((op) => op.type === "addEdge");
                const { legacy, builder, pool } = drive({ ...c, ops: adds }, false);
                for (let i = 0; i < pool.length && i < 40; i++) {
                    const id = pool[i];
                    if (!legacy.hasNode(id)) {
                        expect(builder.hasNode(id)).toBe(false);
                        continue;
                    }
                    const incident = new Set<string>();
                    for (const v of legacy.neighbors(id)) {
                        incident.add(`${keyOf(id)}>${keyOf(v)}`);
                    }
                    for (const v of legacy.inNeighbors(id)) {
                        incident.add(`${keyOf(v)}>${keyOf(id)}`);
                    }
                    if (!c.directed) {
                        // undirected: the legacy adjacency is symmetric; one entry per edge suffices
                        incident.clear();
                        for (const v of legacy.neighbors(id)) {
                            incident.add(keyOf(v));
                        }
                    }
                    const before = legacy.totalEdgeCount;
                    legacy.removeNode(id);
                    const removed = builder.removeNode(id);
                    expect(removed.length).toBe(before - legacy.totalEdgeCount);
                    expect(removed.length).toBe(incident.size);
                    for (const e of removed) {
                        expect(builder.hasEdge(e)).toBe(false);
                    }
                }
                const s = builder.freeze();
                compare(legacy, s, pool);
            }),
            { numRuns: Math.max(10, Math.floor(RUNS / 2)) },
        );
    });
});
