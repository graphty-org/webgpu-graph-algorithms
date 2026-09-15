/**
 * Differential tests against the legacy `Graph` class, part 5 (design section 16.2): the builder
 * policies of sections 3.4 / 3.5 / 6.5 compared with the legacy `allowSelfLoops` /
 * `allowParallelEdges` configuration, and the alternative population paths of section 8.1
 * (`fromRecords`, `fromEdgeArrays`, `addEdgesByIds`) compared with the legacy class fed the same
 * data one call at a time:
 *
 * - `selfLoops: "drop"` vs `allowSelfLoops: false` (the legacy creates both endpoints BEFORE it
 *   refuses the loop, so the node sets agree);
 * - `duplicateEdges: "first"` vs `allowParallelEdges: false` (the legacy keeps the first parallel);
 * - `duplicateEdges: "last"` vs `allowParallelEdges: true` (the legacy adjacency Map keeps the LAST
 *   Edge object per pair, so `getEdge().weight` is the last weight);
 * - `selfLoops: "error"` / `duplicateEdges: "error"` throw at freeze exactly when the legacy threw
 *   at `addEdge` at least once;
 * - `fromRecords` node-link records with attributes vs `addNode(id, data)` / `addEdge(..., data)`;
 * - `fromEdgeArrays` with external ids vs the legacy fed the ids then the edges;
 * - `addEdgesByIds` (bulk) vs sequential legacy `addEdge` (node order = first appearance).
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { fromEdgeArrays } from "../../src/populate/from-edge-arrays.js";
import { fromRecords } from "../../src/populate/from-records.js";
import { type GraphSnapshot, type NodeId } from "../../src/types/index.js";
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

interface PolicyCase {
    readonly directed: boolean;
    readonly idKind: IdKind;
    readonly n: number;
    readonly log: readonly Log[];
}

const policyCaseArb: fc.Arbitrary<PolicyCase> = fc
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
        return fc.record({
            head: fc.constant(head),
            log: fc.array(edge, { maxLength: Math.min(4000, 3 * head.n), size: "max" }),
        });
    })
    .map(({ head, log }) => ({ ...head, log }));

function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

/** Feed a log to a legacy graph; returns how many requests it refused. */
function feedLegacy(legacy: LegacyGraph, log: readonly Log[]): number {
    let refused = 0;
    for (const e of log) {
        try {
            legacy.addEdge(e.source, e.target, e.weight);
        } catch {
            refused++;
        }
    }
    return refused;
}

/** Feed the whole log to a builder (nodes created in legacy order: source, then target). */
function feedBuilder(builder: GraphBuilder, log: readonly Log[]): void {
    for (const e of log) {
        builder.addNode(e.source);
        builder.addNode(e.target);
        builder.addEdge(e.source, e.target, e.weight);
    }
}

/** Node order, distinct neighbour sets, hasEdge and (optionally) the weight the legacy getEdge holds. */
function compare(legacy: LegacyGraph, s: GraphSnapshot, weightArc: "first" | "last" | null): void {
    assertInvariants(s);
    expect(s.nodeCount).toBe(legacy.nodeCount);
    expect(s.ids.toArray()).toEqual([...legacy.nodes()].map((node) => node.id));
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        const row = new Set<string>();
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            row.add(keyOf(s.ids.idOf(s.colIdx[a])));
        }
        expect(row).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
        expect(row.size).toBe(legacy.outDegree(id));
    }
    if (weightArc === null) {
        return;
    }
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        for (const v of legacy.neighbors(id)) {
            const [lo, hi] = s.arcsBetween(u, s.ids.requireIndex(v));
            expect(hi).toBeGreaterThan(lo);
            const arc = weightArc === "first" ? lo : hi - 1;
            const w = s.weights === null ? 1 : s.weights[arc];
            expect(Object.is(w, Math.fround(legacy.getEdge(id, v)?.weight ?? Number.NaN))).toBe(true);
        }
    }
}

// ============================================================ policies (3.4, 3.5, 6.5)

describe("differential: builder policies vs the legacy configuration (design sections 3.4, 3.5, 6.5)", () => {
    it('selfLoops: "drop" equals allowSelfLoops: false', () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: false,
                    allowParallelEdges: true,
                });
                const refused = feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed, selfLoops: "drop" });
                feedBuilder(builder, c.log);
                const { snapshot, report } = builder.freezeWithReport();
                compare(legacy, snapshot, null);
                expect(snapshot.edgeCount).toBe(legacy.totalEdgeCount);
                expect(snapshot.selfLoopCount).toBe(0);
                expect(report.droppedSelfLoops).toBe(refused);
                // the builder was rewritten to the loop-free edge set (6.3 step 2): a re-freeze agrees
                expect(builder.edgeCount).toBe(legacy.totalEdgeCount);
                const again = builder.freezeWithReport();
                expect(again.report.droppedSelfLoops).toBe(0);
                expect(again.snapshot.contentHash()).toBe(snapshot.contentHash());
            }),
            { numRuns: RUNS },
        );
    });

    it('duplicateEdges: "first" equals allowParallelEdges: false (the legacy keeps the first parallel)', () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: false,
                });
                const refused = feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed, duplicateEdges: "first" });
                feedBuilder(builder, c.log);
                const { snapshot, report } = builder.freezeWithReport();
                compare(legacy, snapshot, "first");
                expect(snapshot.edgeCount).toBe(legacy.totalEdgeCount);
                expect(snapshot.flags.multigraph).toBe(false);
                expect(report.mergedEdges).toBe(refused);
                // survivors are the lowest indices, in order: the accepted log of the legacy
                const accepted: Log[] = [];
                const seen = new LegacyGraph({ directed: c.directed, allowParallelEdges: false });
                for (const e of c.log) {
                    try {
                        seen.addEdge(e.source, e.target, e.weight);
                    } catch {
                        continue;
                    }
                    accepted.push(e);
                }
                expect(snapshot.edgeCount).toBe(accepted.length);
                for (let e = 0; e < accepted.length; e++) {
                    expect(snapshot.ids.idOf(snapshot.edgeSource(e))).toBe(accepted[e].source);
                    expect(snapshot.ids.idOf(snapshot.edgeTarget(e))).toBe(accepted[e].target);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it('duplicateEdges: "last" holds the weight the legacy getEdge holds (last write wins in its Map)', () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: true,
                });
                feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed, duplicateEdges: "last" });
                feedBuilder(builder, c.log);
                const snapshot = builder.freeze();
                compare(legacy, snapshot, "last");
                expect(snapshot.flags.multigraph).toBe(false);
                // one edge per distinct pair
                const pairs = new Set<string>();
                for (const e of c.log) {
                    const a = keyOf(e.source);
                    const b = keyOf(e.target);
                    pairs.add(c.directed || a <= b ? `${a}|${b}` : `${b}|${a}`);
                }
                expect(snapshot.edgeCount).toBe(pairs.size);
            }),
            { numRuns: RUNS },
        );
    });

    it('the "error" policies throw at freeze exactly when the legacy refused at least one addEdge', () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const noLoops = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: false,
                    allowParallelEdges: true,
                });
                const loopsRefused = feedLegacy(noLoops, c.log);
                const loopBuilder = new GraphBuilder({ directed: c.directed, selfLoops: "error" });
                feedBuilder(loopBuilder, c.log);
                let loopCode: string | null = null;
                try {
                    loopBuilder.freeze();
                } catch (err) {
                    if (!(err instanceof GraphFormatError)) {
                        throw err;
                    }
                    loopCode = err.code;
                }
                expect(loopCode).toBe(loopsRefused > 0 ? "E_SELF_LOOP" : null);
                // the builder is untouched by the throw (11.1): the same edges are still there
                expect(loopBuilder.edgeCount).toBe(c.log.length);

                const noParallels = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: false,
                });
                const parallelsRefused = feedLegacy(noParallels, c.log);
                const dupBuilder = new GraphBuilder({ directed: c.directed, duplicateEdges: "error" });
                feedBuilder(dupBuilder, c.log);
                let dupCode: string | null = null;
                try {
                    dupBuilder.freeze();
                } catch (err) {
                    if (!(err instanceof GraphFormatError)) {
                        throw err;
                    }
                    dupCode = err.code;
                }
                expect(dupCode).toBe(parallelsRefused > 0 ? "E_DUPLICATE_EDGE" : null);
                expect(dupBuilder.edgeCount).toBe(c.log.length);
            }),
            { numRuns: RUNS },
        );
    });
});

// ============================================================ population paths (8.1)

describe("differential: fromRecords / fromEdgeArrays / addEdgesByIds vs sequential legacy calls (design section 8.1)", () => {
    it("fromRecords with node and edge attribute records equals addNode(id, data) / addEdge(..., data)", () => {
        fc.assert(
            fc.property(policyCaseArb, fc.array(fc.nat({ max: 5 }), { maxLength: 64 }), (c, tags) => {
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: true,
                });
                const nodeRecords: Record<string, unknown>[] = [];
                const nodeIds = new Set<NodeId>();
                // node records for a prefix of the ids (the rest appear through edges only)
                for (let i = 0; i < Math.min(c.n, tags.length); i++) {
                    const id = idOf(c.idKind, i);
                    nodeIds.add(id);
                    nodeRecords.push({ id, tag: tags[i], label: `L${tags[i]}` });
                    legacy.addNode(id, { tag: tags[i], label: `L${tags[i]}` });
                }
                const edgeRecords: Record<string, unknown>[] = [];
                c.log.forEach((e, k) => {
                    const record: Record<string, unknown> = { source: e.source, target: e.target, seq: k };
                    if (e.weight !== undefined) {
                        record.weight = e.weight;
                    }
                    edgeRecords.push(record);
                    legacy.addEdge(e.source, e.target, e.weight, { seq: k });
                });
                const { snapshot: s } = fromRecords(
                    { directed: c.directed, nodes: nodeRecords, edges: edgeRecords },
                    { weightDtype: "f64" },
                );
                assertInvariants(s);
                expect(s.nodeCount).toBe(legacy.nodeCount);
                expect(s.ids.toArray()).toEqual([...legacy.nodes()].map((node) => node.id));
                expect(s.edgeCount).toBe(legacy.totalEdgeCount);
                for (let u = 0; u < s.nodeCount; u++) {
                    const id = s.ids.idOf(u);
                    const row = new Set<string>();
                    for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                        row.add(keyOf(s.ids.idOf(s.colIdx[a])));
                    }
                    expect(row).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
                    const data = legacy.getNode(id)?.data;
                    if (nodeIds.has(id)) {
                        expect(s.nodes.value("tag", u)).toBe(data?.tag);
                        expect(s.nodes.value("label", u)).toBe(data?.label);
                    } else if (s.nodes.has("tag")) {
                        expect(s.nodes.isSet("tag", u)).toBe(false);
                    }
                }
                // edge attributes are per LOGICAL edge (I13) in record order
                const list = s.edgeList();
                for (let e = 0; e < s.edgeCount; e++) {
                    expect(s.edges.value("seq", e)).toBe(e);
                    expect(s.ids.idOf(list.src[e])).toBe(c.log[e].source);
                    expect(s.ids.idOf(list.dst[e])).toBe(c.log[e].target);
                    const shadow = s.edges.byRole("weight");
                    const exact =
                        shadow !== null && shadow.dtype === "f64" && shadow.isSet(e)
                            ? (shadow.value(e) as number)
                            : (list.weights?.[e] ?? 1);
                    expect(Object.is(exact, c.log[e].weight ?? 1)).toBe(true);
                }
                // no reserved key became a column
                expect(s.edges.has("source")).toBe(false);
                expect(s.edges.has("target")).toBe(false);
                expect(s.nodes.has("id")).toBe(false);
            }),
            { numRuns: RUNS },
        );
    });

    it("fromEdgeArrays with external ids equals the legacy fed the ids then the index edges", () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const ids = Array.from({ length: c.n }, (_, i) => idOf(c.idKind, i));
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: true,
                });
                for (const id of ids) {
                    legacy.addNode(id);
                }
                const index = new Map<string, number>(ids.map((id, i) => [keyOf(id), i]));
                const src = new Uint32Array(c.log.length);
                const dst = new Uint32Array(c.log.length);
                const weights = new Float64Array(c.log.length);
                c.log.forEach((e, k) => {
                    src[k] = index.get(keyOf(e.source)) ?? INVALID_INDEX;
                    dst[k] = index.get(keyOf(e.target)) ?? INVALID_INDEX;
                    weights[k] = e.weight ?? 1;
                    legacy.addEdge(e.source, e.target, e.weight ?? 1);
                });
                const s = fromEdgeArrays({ directed: c.directed, ids, src, dst, weights });
                assertInvariants(s);
                expect(s.ids.toArray()).toEqual([...legacy.nodes()].map((node) => node.id));
                expect(s.edgeCount).toBe(legacy.totalEdgeCount);
                for (let u = 0; u < s.nodeCount; u++) {
                    const id = s.ids.idOf(u);
                    const row = new Set<string>();
                    for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                        row.add(keyOf(s.ids.idOf(s.colIdx[a])));
                    }
                    expect(row).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
                }
                for (let e = 0; e < c.log.length; e++) {
                    expect(s.edgeSource(e)).toBe(src[e]);
                    expect(s.edgeTarget(e)).toBe(dst[e]);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("addEdgesByIds assigns node indices in the same first-appearance order as sequential legacy addEdge", () => {
        fc.assert(
            fc.property(policyCaseArb, (c) => {
                const legacy = new LegacyGraph({
                    directed: c.directed,
                    allowSelfLoops: true,
                    allowParallelEdges: true,
                });
                feedLegacy(legacy, c.log);
                const builder = new GraphBuilder({ directed: c.directed });
                const first = builder.addEdgesByIds(
                    c.log.map((e) => e.source),
                    c.log.map((e) => e.target),
                    c.log.map((e) => e.weight ?? 1),
                );
                expect(first).toBe(0);
                const s = builder.freeze();
                assertInvariants(s);
                expect(s.ids.toArray()).toEqual([...legacy.nodes()].map((node) => node.id));
                expect(s.edgeCount).toBe(legacy.totalEdgeCount);
                const sequential = new GraphBuilder({ directed: c.directed });
                for (const e of c.log) {
                    sequential.addEdge(e.source, e.target, e.weight ?? 1);
                }
                const t = sequential.freeze();
                expect(t.contentHash()).toBe(s.contentHash());
                expect(t.ids.toArray()).toEqual(s.ids.toArray());
            }),
            { numRuns: RUNS },
        );
    });
});
