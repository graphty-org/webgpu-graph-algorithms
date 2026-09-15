/**
 * Differential tests against the legacy `Graph` class, part 6 (design sections 14.2 and 16.2):
 * the worked ports of section 14.2 -- breadthFirstSearch (port 1, over the snapshot and over
 * `reverse()` for the in-neighbour walk), dijkstra (port 2, an O(n^2) selection instead of the
 * indexed heap; the relaxation loop is the design's), connectedComponents (port 4, union-find over
 * `edgeList()`) and commonNeighborsScore (port 6, the sorted-row merge) -- run over snapshots and
 * compared with straightforward implementations over the legacy class on the same random graphs.
 * Depths, distances, component partitions and common-neighbour counts are order-independent, so
 * they must be equal exactly; BFS parents and visiting order are allowed to differ (rows are sorted
 * by index, the legacy visits in insertion order; design 3.9 declares that change).
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type AdjacencyView, type GraphSnapshot, type NodeId } from "../../src/types/index.js";
import { LegacyGraph, type LegacyNodeId } from "../helpers/legacy-graph.js";

const RUNS = Number(process.env.FC_RUNS ?? "60");

// ============================================================ generators (self-contained)

type IdKind = "ints" | "strings" | "mixed";

function idOf(kind: IdKind, i: number): NodeId {
    switch (kind) {
        case "ints":
            return i * 2 + 1;
        case "strings":
            return `n${i}`;
        case "mixed":
            return i % 2 === 0 ? i / 2 : `${(i - 1) / 2}`;
        default:
            throw new Error(`unknown kind ${String(kind)}`);
    }
}

/** f32-exact small positive weights so that path sums are exact in both f32 and f64. */
const WEIGHTS: readonly number[] = [1, 2, 3, 4, 5, 6, 7, 0.5, 0.25];

interface Log {
    readonly source: NodeId;
    readonly target: NodeId;
    readonly weight: number;
}

interface PortCase {
    readonly directed: boolean;
    readonly allowParallelEdges: boolean;
    readonly idKind: IdKind;
    readonly n: number;
    readonly log: readonly Log[];
    readonly start: number;
}

const portCaseArb: fc.Arbitrary<PortCase> = fc
    .record({
        directed: fc.boolean(),
        allowParallelEdges: fc.boolean(),
        idKind: fc.constantFrom<IdKind>("ints", "strings", "mixed"),
        n: fc.oneof(
            { weight: 5, arbitrary: fc.integer({ min: 1, max: 12 }) },
            { weight: 3, arbitrary: fc.integer({ min: 13, max: 300 }) },
            { weight: 1, arbitrary: fc.integer({ min: 1500, max: 2000 }) },
        ),
    })
    .chain((head) => {
        const index = fc.nat({ max: head.n - 1 });
        const edge = fc
            .tuple(index, index, fc.constantFrom(...WEIGHTS))
            .map(([s, t, w]) => ({ source: idOf(head.idKind, s), target: idOf(head.idKind, t), weight: w }));
        return fc.record({
            head: fc.constant(head),
            log: fc.array(edge, { maxLength: Math.min(4000, 3 * head.n), size: "max" }),
            start: index,
        });
    })
    .map(({ head, log, start }) => ({ ...head, log, start }));

function buildBoth(c: PortCase): { legacy: LegacyGraph; snapshot: GraphSnapshot } {
    const legacy = new LegacyGraph({
        directed: c.directed,
        allowSelfLoops: true,
        allowParallelEdges: c.allowParallelEdges,
    });
    const builder = new GraphBuilder({ directed: c.directed });
    for (const e of c.log) {
        builder.addNode(e.source);
        builder.addNode(e.target);
        try {
            legacy.addEdge(e.source, e.target, e.weight);
        } catch {
            continue;
        }
        builder.addEdge(e.source, e.target, e.weight);
    }
    return { legacy, snapshot: builder.freeze() };
}

// ============================================================ the ports (design 14.2, verbatim where possible)

interface BfsResult {
    readonly order: Uint32Array;
    readonly parent: Uint32Array;
    readonly depth: Uint32Array;
    readonly visitedCount: number;
}

/** Port 1 of design 14.2. */
function breadthFirstSearch(g: AdjacencyView, start: number): BfsResult {
    const { nodeCount, rowPtr, colIdx } = g;
    const parent = new Uint32Array(nodeCount).fill(INVALID_INDEX);
    const depth = new Uint32Array(nodeCount).fill(INVALID_INDEX);
    const order = new Uint32Array(nodeCount);
    let head = 0;
    let tail = 0;
    order[tail++] = start;
    depth[start] = 0;
    while (head < tail) {
        const u = order[head++];
        const d = depth[u];
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            const v = colIdx[a];
            if (depth[v] === INVALID_INDEX) {
                depth[v] = d + 1;
                parent[v] = u;
                order[tail++] = v;
            }
        }
    }
    return { order: order.subarray(0, tail), parent, depth, visitedCount: tail };
}

/** The legacy BFS: depths by id over neighbors() (or inNeighbors() for the reverse walk). */
function legacyBfs(legacy: LegacyGraph, start: LegacyNodeId, reverse: boolean): Map<string, number> {
    const depth = new Map<string, number>();
    const key = (id: LegacyNodeId): string => (typeof id === "number" ? `n:${String(id)}` : `s:${id}`);
    const queue: LegacyNodeId[] = [start];
    depth.set(key(start), 0);
    for (let head = 0; head < queue.length; head++) {
        const u = queue[head];
        const d = depth.get(key(u)) ?? 0;
        const neighbours = reverse ? legacy.inNeighbors(u) : legacy.neighbors(u);
        for (const v of neighbours) {
            if (!depth.has(key(v))) {
                depth.set(key(v), d + 1);
                queue.push(v);
            }
        }
    }
    return depth;
}

/** Port 2 of design 14.2 with an O(n^2) extract-min instead of the indexed heap. */
function dijkstra(g: AdjacencyView, source: number): Float64Array {
    const { nodeCount, rowPtr, colIdx, weights } = g;
    const dist = new Float64Array(nodeCount).fill(Infinity);
    const done = new Uint8Array(nodeCount);
    dist[source] = 0;
    for (let round = 0; round < nodeCount; round++) {
        let u = INVALID_INDEX;
        let best = Infinity;
        for (let i = 0; i < nodeCount; i++) {
            if (done[i] === 0 && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u === INVALID_INDEX) {
            break;
        }
        done[u] = 1;
        const du = dist[u];
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            const w = weights === null ? 1 : weights[a];
            const v = colIdx[a];
            const dv = du + w;
            if (dv < dist[v]) {
                dist[v] = dv;
            }
        }
    }
    return dist;
}

/** The legacy Dijkstra over neighbors() and getEdge().weight, keyed by id. */
function legacyDijkstra(legacy: LegacyGraph, source: LegacyNodeId): Map<string, number> {
    const key = (id: LegacyNodeId): string => (typeof id === "number" ? `n:${String(id)}` : `s:${id}`);
    const dist = new Map<string, number>();
    const ids = [...legacy.nodes()].map((node) => node.id);
    for (const id of ids) {
        dist.set(key(id), Infinity);
    }
    dist.set(key(source), 0);
    const done = new Set<string>();
    for (let round = 0; round < ids.length; round++) {
        let u: LegacyNodeId | null = null;
        let best = Infinity;
        for (const id of ids) {
            const d = dist.get(key(id)) ?? Infinity;
            if (!done.has(key(id)) && d < best) {
                best = d;
                u = id;
            }
        }
        if (u === null) {
            break;
        }
        done.add(key(u));
        for (const v of legacy.neighbors(u)) {
            const w = legacy.getEdge(u, v)?.weight ?? 1;
            const dv = best + w;
            if (dv < (dist.get(key(v)) ?? Infinity)) {
                dist.set(key(v), dv);
            }
        }
    }
    return dist;
}

/** Port 4 of design 14.2: union-find over edgeList(), labels renumbered in first-seen node order. */
function connectedComponents(s: GraphSnapshot): Uint32Array {
    const parent = new Uint32Array(s.nodeCount);
    for (let i = 0; i < s.nodeCount; i++) {
        parent[i] = i;
    }
    const find = (x: number): number => {
        let r = x;
        while (parent[r] !== r) {
            r = parent[r];
        }
        let y = x;
        while (parent[y] !== r) {
            const next = parent[y];
            parent[y] = r;
            y = next;
        }
        return r;
    };
    const { src, dst } = s.edgeList();
    for (let e = 0; e < s.edgeCount; e++) {
        const a = find(src[e]);
        const b = find(dst[e]);
        if (a !== b) {
            parent[Math.max(a, b)] = Math.min(a, b);
        }
    }
    const labels = new Uint32Array(s.nodeCount);
    const renumber = new Map<number, number>();
    for (let i = 0; i < s.nodeCount; i++) {
        const r = find(i);
        let label = renumber.get(r);
        if (label === undefined) {
            label = renumber.size;
            renumber.set(r, label);
        }
        labels[i] = label;
    }
    return labels;
}

/** The legacy weakly connected components: BFS over neighbors() and inNeighbors(), first-seen order. */
function legacyComponents(legacy: LegacyGraph): Map<string, number> {
    const key = (id: LegacyNodeId): string => (typeof id === "number" ? `n:${String(id)}` : `s:${id}`);
    const label = new Map<string, number>();
    let count = 0;
    for (const node of legacy.nodes()) {
        if (label.has(key(node.id))) {
            continue;
        }
        const queue: LegacyNodeId[] = [node.id];
        label.set(key(node.id), count);
        for (let head = 0; head < queue.length; head++) {
            const u = queue[head];
            for (const v of [...legacy.neighbors(u), ...legacy.inNeighbors(u)]) {
                if (!label.has(key(v))) {
                    label.set(key(v), count);
                    queue.push(v);
                }
            }
        }
        count++;
    }
    return label;
}

/** Port 6 of design 14.2. */
function commonNeighborsScore(s: GraphSnapshot, u: number, v: number, directed: boolean): number {
    const fwd: AdjacencyView = s;
    const bwd: AdjacencyView = directed ? s.reverse() : s;
    let i = fwd.rowPtr[u];
    const iEnd = fwd.rowPtr[u + 1];
    let j = bwd.rowPtr[v];
    const jEnd = bwd.rowPtr[v + 1];
    let count = 0;
    while (i < iEnd && j < jEnd) {
        const a = fwd.colIdx[i];
        const b = bwd.colIdx[j];
        if (a === b) {
            count++;
            i++;
            j++;
            while (i < iEnd && fwd.colIdx[i] === a) {
                i++;
            }
            while (j < jEnd && bwd.colIdx[j] === b) {
                j++;
            }
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

// ============================================================ the properties

describe("differential: the worked ports of design 14.2 vs legacy implementations", () => {
    it("port 1: BFS depths over the snapshot equal the legacy BFS; over reverse() they equal the in-neighbour BFS", () => {
        fc.assert(
            fc.property(portCaseArb, (c) => {
                const { legacy, snapshot: s } = buildBoth(c);
                fc.pre(s.nodeCount > 0);
                const start = c.start % s.nodeCount;
                const startId = s.ids.idOf(start);
                const forward = breadthFirstSearch(s, start);
                const expected = legacyBfs(legacy, startId, false);
                expect(forward.visitedCount).toBe(expected.size);
                for (let i = 0; i < s.nodeCount; i++) {
                    const d = expected.get(keyOf(s.ids.idOf(i)));
                    expect(forward.depth[i]).toBe(d === undefined ? INVALID_INDEX : d);
                    if (forward.parent[i] !== INVALID_INDEX) {
                        // the parent is a genuine predecessor one level up
                        expect(forward.depth[forward.parent[i]]).toBe(forward.depth[i] - 1);
                        expect(legacy.hasEdge(s.ids.idOf(forward.parent[i]), s.ids.idOf(i))).toBe(true);
                    }
                }
                // the order is a valid BFS order: non-decreasing depth
                for (let k = 1; k < forward.visitedCount; k++) {
                    expect(forward.depth[forward.order[k]]).toBeGreaterThanOrEqual(forward.depth[forward.order[k - 1]]);
                }
                const backward = breadthFirstSearch(s.reverse(), start);
                const expectedIn = legacyBfs(legacy, startId, true);
                expect(backward.visitedCount).toBe(expectedIn.size);
                for (let i = 0; i < s.nodeCount; i++) {
                    const d = expectedIn.get(keyOf(s.ids.idOf(i)));
                    expect(backward.depth[i]).toBe(d === undefined ? INVALID_INDEX : d);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("port 2: Dijkstra distances over weights[a] equal the legacy Dijkstra on simple graphs", () => {
        fc.assert(
            fc.property(portCaseArb, (c) => {
                fc.pre(!c.allowParallelEdges && c.n <= 300);
                const { legacy, snapshot: s } = buildBoth(c);
                fc.pre(s.nodeCount > 0);
                expect(s.flags.nonNegativeWeights).toBe(true);
                const start = c.start % s.nodeCount;
                const dist = dijkstra(s, start);
                const expected = legacyDijkstra(legacy, s.ids.idOf(start));
                for (let i = 0; i < s.nodeCount; i++) {
                    expect(dist[i]).toBe(expected.get(keyOf(s.ids.idOf(i))));
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("port 2 on a multigraph: the format relaxes every parallel arc while the legacy only sees the last one", () => {
        // Not a defect: the legacy adjacency Map holds one Edge per pair (last write wins, and it
        // even overwrites a cheaper earlier parallel); the format keeps every arc (3.5). With
        // parallels present the format's distance is the true shortest path, never longer.
        fc.assert(
            fc.property(portCaseArb, (c) => {
                fc.pre(c.allowParallelEdges && c.n <= 200);
                const { legacy, snapshot: s } = buildBoth(c);
                fc.pre(s.nodeCount > 0);
                const start = c.start % s.nodeCount;
                const dist = dijkstra(s, start);
                const expected = legacyDijkstra(legacy, s.ids.idOf(start));
                for (let i = 0; i < s.nodeCount; i++) {
                    expect(dist[i]).toBeLessThanOrEqual(expected.get(keyOf(s.ids.idOf(i))) ?? Infinity);
                }
            }),
            { numRuns: Math.max(10, Math.floor(RUNS / 2)) },
        );
    });

    it("port 4: connected components over edgeList() partition the nodes like the legacy BFS components", () => {
        fc.assert(
            fc.property(portCaseArb, (c) => {
                const { legacy, snapshot: s } = buildBoth(c);
                const labels = connectedComponents(s);
                const expected = legacyComponents(legacy);
                // both renumber in first-seen node order (design port 4 says so), so labels are equal
                for (let i = 0; i < s.nodeCount; i++) {
                    expect(labels[i]).toBe(expected.get(keyOf(s.ids.idOf(i))));
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("port 6: commonNeighborsScore equals the legacy neighbour-set intersection (parallels counted once)", () => {
        fc.assert(
            fc.property(
                portCaseArb,
                fc.array(fc.tuple(fc.nat(), fc.nat()), { maxLength: 100, size: "max" }),
                (c, pairs) => {
                    const { legacy, snapshot: s } = buildBoth(c);
                    fc.pre(s.nodeCount > 0);
                    for (const [x, y] of pairs) {
                        const u = x % s.nodeCount;
                        const v = y % s.nodeCount;
                        const uId = s.ids.idOf(u);
                        const vId = s.ids.idOf(v);
                        const left = new Set([...legacy.neighbors(uId)].map(keyOf));
                        const right = c.directed ? legacy.inNeighbors(vId) : legacy.neighbors(vId);
                        let expected = 0;
                        for (const z of right) {
                            if (left.has(keyOf(z))) {
                                expected++;
                            }
                        }
                        expect(commonNeighborsScore(s, u, v, c.directed)).toBe(expected);
                    }
                },
            ),
            { numRuns: RUNS },
        );
    });
});
