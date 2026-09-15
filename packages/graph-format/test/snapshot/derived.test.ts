import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { buildCore, renumberPartition } from "../../src/snapshot/derived.js";
import { equalsTopology, hasChecksums, peekView } from "../../src/snapshot/graph-snapshot.js";
import { type DerivedGraph, type GraphSnapshot, type U32 } from "../../src/types/index.js";
import { makeMask, maskSet } from "../../src/util/mask.js";
import { assertInvariants } from "../helpers/invariants.js";
import { type EdgeSpec, type GraphSpec, KARATE_EDGES, makeSnapshot } from "../helpers/parts.js";

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

/** The logical edges of a snapshot as [source, target, weight] in index order. */
function edgesOf(s: GraphSnapshot): [number, number, number][] {
    const out: [number, number, number][] = [];
    for (let e = 0; e < s.edgeCount; e++) {
        out.push([s.edgeSource(e), s.edgeTarget(e), s.weights === null ? 1 : s.weights[s.edgeToArc[e]]]);
    }
    return out;
}

/** P9b: remap / origin consistency for every derived graph. */
function assertMaps(source: GraphSnapshot, d: DerivedGraph): void {
    assertInvariants(d.snapshot);
    if (d.edgeOrigin !== null || d.edgeRemap !== null) {
        expect(d.edgeOrigin).not.toBeNull();
        expect(d.edgeRemap).not.toBeNull();
        const origin = d.edgeOrigin as Uint32Array;
        const remap = d.edgeRemap as Uint32Array;
        expect(origin.length).toBe(d.snapshot.edgeCount);
        expect(remap.length).toBe(source.edgeCount);
        for (let x = 0; x < origin.length; x++) {
            expect(remap[origin[x]]).toBe(x);
            expect(origin[x]).not.toBe(INVALID_INDEX);
        }
        for (let e = 0; e < remap.length; e++) {
            if (remap[e] !== INVALID_INDEX) {
                expect(remap[e]).toBeLessThan(d.snapshot.edgeCount);
            }
        }
        // survivors are numbered in ascending source order
        for (let x = 1; x < origin.length; x++) {
            expect(origin[x]).toBeGreaterThan(origin[x - 1]);
        }
    } else {
        expect(d.snapshot.edgeCount).toBe(source.edgeCount);
    }
    if (d.nodeOrigin !== null || d.nodeRemap !== null) {
        expect(d.nodeOrigin).not.toBeNull();
        expect(d.nodeRemap).not.toBeNull();
        const origin = d.nodeOrigin as Uint32Array;
        const remap = d.nodeRemap as Uint32Array;
        expect(origin.length).toBe(d.snapshot.nodeCount);
        expect(remap.length).toBe(source.nodeCount);
        for (let x = 0; x < origin.length; x++) {
            expect(remap[origin[x]]).toBe(x);
        }
    } else {
        expect(d.snapshot.nodeCount).toBe(source.nodeCount);
        expect(d.snapshot.nodes).toBe(source.nodes);
        expect(d.snapshot.ids).toBe(source.ids);
    }
    expect(d.snapshot.graph).toBe(source.graph);
    expect(d.snapshot.meta).toBe(source.meta);
}

const edgesArb = (weighted: boolean) =>
    fc.integer({ min: 1, max: 7 }).chain((n) => {
        const endpoint = fc.integer({ min: 0, max: n - 1 });
        const weight = fc.integer({ min: -3, max: 5 });
        const edge: fc.Arbitrary<EdgeSpec> = weighted
            ? fc.tuple(endpoint, endpoint, weight).map(([u, v, w]) => [u, v, w] as const)
            : fc.tuple(endpoint, endpoint).map(([u, v]) => [u, v] as const);
        return fc.array(edge, { maxLength: 20 }).map((edges) => ({ n, edges }));
    });

describe("buildCore", () => {
    it("builds an empty core and a core without an arena", () => {
        const empty = buildCore({
            directed: false,
            nodeCount: 0,
            edgeCount: 0,
            src: new Uint32Array(0),
            dst: new Uint32Array(0),
            weights: null,
        });
        expect(Array.from(empty.rowPtr)).toEqual([0]);
        expect(empty.colIdx.length).toBe(0);
        expect(empty.arcToEdge?.length).toBe(0);
        expect(empty.arena?.segments.colIdx).toBeNull();
        expect(empty.arena?.byteLength).toBe(4);
        const separate = buildCore(
            {
                directed: true,
                nodeCount: 2,
                edgeCount: 1,
                src: new Uint32Array([1]),
                dst: new Uint32Array([0]),
                weights: new Float32Array([2]),
            },
            { arena: false },
        );
        expect(separate.arena).toBeNull();
        expect(separate.flags.arcToEdgeIsIdentity).toBe(true);
        expect(separate.rowPtr.buffer).not.toBe(separate.colIdx.buffer);
    });
});

describe("filterEdges / withoutSelfLoops", () => {
    const spec: GraphSpec = {
        directed: false,
        edges: [
            [0, 1, 1],
            [1, 1, 2],
            [1, 2, 3],
            [2, 0, 4],
            [2, 2, 5],
        ],
        edgeColumns: { label: { values: ["a", "b", "c", "d", "e"], decl: { dtype: "string" } } },
        nodeColumns: { x: new Float32Array([1, 2, 3]) },
    };

    it("filterEdges keeps the masked edges, gathers edge columns and shares the node table", () => {
        const s = makeSnapshot(spec);
        const keep = makeMask(5);
        maskSet(keep, 0, true);
        maskSet(keep, 2, true);
        maskSet(keep, 4, true);
        const d = s.filterEdges(keep);
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [1, 2, 3],
            [2, 2, 5],
        ]);
        expect(Array.from(d.edgeOrigin as Uint32Array)).toEqual([0, 2, 4]);
        expect(Array.from(d.edgeRemap as Uint32Array)).toEqual([0, INVALID_INDEX, 1, INVALID_INDEX, 2]);
        expect(d.report).toEqual({ droppedEdges: 2, mergedEdges: 0 });
        expect(d.snapshot.nodes).toBe(s.nodes);
        expect(d.snapshot.edges).not.toBe(s.edges);
        expect(d.snapshot.edges.value("label", 1)).toBe("c");
        expect(d.snapshot.selfLoopCount).toBe(1);
        expect(d.snapshot.serial).not.toBe(s.serial);
        expect(d.snapshot.arena).not.toBeNull();
        expect(d.blockSizes).toBeNull();
    });

    it("filterEdges with every bit set is a copy with null maps", () => {
        const s = makeSnapshot(spec);
        const d = s.filterEdges(makeMask(5, true));
        assertMaps(s, d);
        expect(d.snapshot).not.toBe(s);
        expect(d.edgeOrigin).toBeNull();
        expect(d.edgeRemap).toBeNull();
        expect(d.snapshot.edges).toBe(s.edges);
        expect(equalsTopology(s, d.snapshot)).toBe(true);
    });

    it("filterEdges rejects a short mask with E_MASK_LENGTH", () => {
        const s = makeSnapshot({
            directed: true,
            nodeCount: 2,
            edges: Array.from({ length: 40 }, () => [0, 1] as const),
        });
        const error = expectError(() => s.filterEdges(new Uint32Array(1)), "E_MASK_LENGTH");
        expect(error.details.required).toBe(2);
        const kept = s.filterEdges(new Uint32Array(2));
        expect(kept.snapshot.edgeCount).toBe(0);
        expect(kept.report.droppedEdges).toBe(40);
    });

    it("withoutSelfLoops drops loops and reports them; a loop-free graph gets null maps", () => {
        const s = makeSnapshot(spec);
        const d = s.withoutSelfLoops();
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [1, 2, 3],
            [2, 0, 4],
        ]);
        expect(d.snapshot.flags.hasSelfLoops).toBe(false);
        expect(d.report.droppedEdges).toBe(2);
        const clean = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        const same = clean.withoutSelfLoops();
        expect(same.edgeRemap).toBeNull();
        expect(equalsTopology(clean, same.snapshot)).toBe(true);
    });

    it("rewrites refersTo edge columns and extension tables through the edge remap", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 0],
                [1, 2],
            ],
            edgeColumns: { pair: { values: [1, 0, undefined], decl: { dtype: "u32", refersTo: "edge" } } },
            extensions: {
                "temporal:edge:w": {
                    rowCount: 3,
                    columns: { element: { values: [0, 2, 1], decl: { dtype: "u32", refersTo: "edge" } } },
                },
            },
        });
        const keep = makeMask(3);
        maskSet(keep, 1, true);
        maskSet(keep, 2, true);
        const d = s.filterEdges(keep);
        assertMaps(s, d);
        const pair = d.snapshot.edges.requireTyped("pair", "u32");
        expect(Array.from(pair.data)).toEqual([INVALID_INDEX, INVALID_INDEX]);
        expect(pair.isSet(0)).toBe(false);
        const ext = d.snapshot.extensions.get("temporal:edge:w");
        expect(ext).toBeDefined();
        const element = ext?.requireTyped("element", "u32");
        expect(Array.from(element?.data ?? [])).toEqual([INVALID_INDEX, 1, 0]);
        expect(element?.isSet(0)).toBe(false);
    });
});

describe("inducedSubgraph", () => {
    const spec: GraphSpec = {
        directed: true,
        ids: ["a", "b", "c", "d"],
        edges: [
            [0, 1, 1],
            [1, 2, 2],
            [2, 3, 3],
            [3, 0, 4],
            [1, 1, 5],
        ],
        nodeColumns: {
            parent: { values: [undefined, 0, 1, 2], decl: { dtype: "u32", refersTo: "node" } },
            x: new Float64Array([10, 20, 30, 40]),
        },
        edgeColumns: { w2: new Float32Array([1, 2, 3, 4, 5]) },
    };

    it("takes an index list in the given order, gathering ids, node columns and edges", () => {
        const s = makeSnapshot(spec);
        const d = s.inducedSubgraph(new Uint32Array([2, 1]));
        assertMaps(s, d);
        expect(d.snapshot.nodeCount).toBe(2);
        expect(d.snapshot.ids.toArray()).toEqual(["c", "b"]);
        expect(Array.from(d.nodeOrigin as Uint32Array)).toEqual([2, 1]);
        expect(Array.from(d.nodeRemap as Uint32Array)).toEqual([INVALID_INDEX, 1, 0, INVALID_INDEX]);
        expect(edgesOf(d.snapshot)).toEqual([
            [1, 0, 2],
            [1, 1, 5],
        ]);
        expect(Array.from(d.edgeOrigin as Uint32Array)).toEqual([1, 4]);
        expect(d.report.droppedEdges).toBe(3);
        expect(Array.from(d.snapshot.nodes.requireTyped("x", "f64").data)).toEqual([30, 20]);
        const parent = d.snapshot.nodes.requireTyped("parent", "u32");
        expect(Array.from(parent.data)).toEqual([1, INVALID_INDEX]);
        expect(parent.isSet(1)).toBe(false);
        expect(Array.from(d.snapshot.edges.requireTyped("w2", "f32").data)).toEqual([2, 5]);
    });

    it("takes a mask in ascending order", () => {
        const s = makeSnapshot(spec);
        const mask = makeMask(4);
        maskSet(mask, 3, true);
        maskSet(mask, 0, true);
        const d = s.inducedSubgraph({ mask });
        assertMaps(s, d);
        expect(Array.from(d.nodeOrigin as Uint32Array)).toEqual([0, 3]);
        expect(edgesOf(d.snapshot)).toEqual([[1, 0, 4]]);
        expect(d.snapshot.ids.kind).toBe("string");
    });

    it("the identity selection shares the node table and id map (P9)", () => {
        const s = makeSnapshot(spec);
        for (const selection of [new Uint32Array([0, 1, 2, 3]), { mask: makeMask(4, true) }]) {
            const d = s.inducedSubgraph(selection);
            assertMaps(s, d);
            expect(d.nodeOrigin).toBeNull();
            expect(d.edgeOrigin).toBeNull();
            expect(d.snapshot.nodes).toBe(s.nodes);
            expect(d.snapshot.edges).toBe(s.edges);
            expect(equalsTopology(s, d.snapshot)).toBe(true);
        }
    });

    it("rejects out-of-range and repeated indices with E_INDEX_RANGE, a short mask with E_MASK_LENGTH", () => {
        const s = makeSnapshot(spec);
        let error = expectError(() => s.inducedSubgraph(new Uint32Array([0, 4])), "E_INDEX_RANGE");
        expect(error.details.index).toBe(1);
        expect(error.details.found).toBe(4);
        error = expectError(() => s.inducedSubgraph(new Uint32Array([2, 0, 2])), "E_INDEX_RANGE");
        expect(error.details.reason).toBe("repeated");
        expectError(() => s.inducedSubgraph({ mask: new Uint32Array(0) }), "E_MASK_LENGTH");
    });

    it("an empty selection yields the empty graph", () => {
        const s = makeSnapshot(spec);
        const d = s.inducedSubgraph(new Uint32Array(0));
        assertMaps(s, d);
        expect(d.snapshot.nodeCount).toBe(0);
        expect(d.snapshot.edgeCount).toBe(0);
        expect(d.snapshot.ids.size).toBe(0);
    });
});

describe("relabel", () => {
    it("permutes the node space with perm[new] = old, id map following, edge order preserved", () => {
        const s = makeSnapshot({
            directed: false,
            ids: ["a", "b", "c"],
            edges: [
                [0, 1, 1],
                [1, 2, 2],
                [2, 2, 3],
            ],
            nodeColumns: { parent: { values: [2, undefined, 0], decl: { dtype: "u32", refersTo: "node" } } },
            edgeColumns: { anchor: { values: [2, 0, 1], decl: { dtype: "u32", refersTo: "node" } } },
        });
        const d = s.relabel(new Uint32Array([2, 0, 1]));
        assertMaps(s, d);
        expect(d.snapshot.ids.toArray()).toEqual(["c", "a", "b"]);
        expect(Array.from(d.nodeRemap as Uint32Array)).toEqual([1, 2, 0]);
        expect(edgesOf(d.snapshot)).toEqual([
            [1, 2, 1],
            [2, 0, 2],
            [0, 0, 3],
        ]);
        expect(d.edgeOrigin).toBeNull();
        // new0 = c (parent a -> new 1), new1 = a (parent c -> new 0), new2 = b (unset)
        expect(Array.from(d.snapshot.nodes.requireTyped("parent", "u32").data)).toEqual([1, 0, INVALID_INDEX]);
        expect(d.snapshot.nodes.requireTyped("parent", "u32").isSet(2)).toBe(false);
        expect(Array.from(d.snapshot.edges.requireTyped("anchor", "u32").data)).toEqual([0, 1, 2]);
        expect(d.snapshot.edges).not.toBe(s.edges);
        const back = d.snapshot.relabel(d.nodeRemap as U32);
        expect(equalsTopology(back.snapshot, s)).toBe(true);
    });

    it("the identity permutation is a copy sharing the tables", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [1, 0],
                [0, 2],
            ],
        });
        const d = s.relabel(new Uint32Array([0, 1, 2]));
        assertMaps(s, d);
        expect(d.nodeOrigin).toBeNull();
        expect(d.snapshot).not.toBe(s);
        expect(d.snapshot.edges).toBe(s.edges);
        expect(equalsTopology(s, d.snapshot)).toBe(true);
    });

    it("relabel(degreeOrder().perm) is the cuGraph renumbering", () => {
        const s = makeSnapshot({ directed: false, edges: KARATE_EDGES });
        const d = s.relabel(s.degreeOrder().perm);
        assertMaps(s, d);
        expect(d.snapshot.outDegreeOf(0)).toBe(17);
        expect(d.snapshot.ids.idOf(0)).toBe(33);
    });

    it("rejects a non-permutation with E_INVALID_PERMUTATION", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        expectError(() => s.relabel(new Uint32Array([0, 1])), "E_INVALID_PERMUTATION");
        expectError(() => s.relabel(new Uint32Array([0, 1, 1])), "E_INVALID_PERMUTATION");
        expectError(() => s.relabel(new Uint32Array([0, 1, 3])), "E_INVALID_PERMUTATION");
    });
});

describe("toUndirected", () => {
    it("returns this with null maps on an undirected snapshot", () => {
        const s = makeSnapshot({ directed: false, edges: [[0, 1]] });
        const d = s.toUndirected();
        expect(d.snapshot).toBe(s);
        expect(d.edgeRemap).toBeNull();
        expect(d.nodeOrigin).toBeNull();
        expect(d.report).toEqual({ droppedEdges: 0, mergedEdges: 0 });
    });

    it("collapses reciprocal pairs keep-first with the weight reducer (Q36)", () => {
        const spec: GraphSpec = {
            directed: true,
            edges: [
                [0, 1, 1],
                [1, 2, 2],
                [1, 0, 3],
                [2, 2, 4],
                [2, 1, 5],
                [1, 2, 6],
                [3, 0, 7],
            ],
            edgeColumns: { tag: { values: ["a", "b", "c", "d", "e", "f", "g"], decl: { dtype: "string" } } },
        };
        const s = makeSnapshot(spec);
        const first = s.toUndirected();
        assertMaps(s, first);
        expect(first.snapshot.directed).toBe(false);
        expect(edgesOf(first.snapshot)).toEqual([
            [0, 1, 1],
            [1, 2, 2],
            [2, 2, 4],
            [1, 2, 6],
            [3, 0, 7],
        ]);
        expect(Array.from(first.edgeRemap as Uint32Array)).toEqual([0, 1, 0, 2, 1, 3, 4]);
        expect(first.report).toEqual({ droppedEdges: 0, mergedEdges: 2 });
        expect(first.snapshot.edges.value("tag", 1)).toBe("b");
        expect(first.snapshot.nodes).toBe(s.nodes);
        expect(edgesOf(s.toUndirected({ weights: "sum" }).snapshot).map((e) => e[2])).toEqual([4, 7, 4, 6, 7]);
        expect(edgesOf(s.toUndirected({ weights: "last" }).snapshot).map((e) => e[2])).toEqual([3, 5, 4, 6, 7]);
        expect(edgesOf(s.toUndirected({ weights: "min" }).snapshot).map((e) => e[2])).toEqual([1, 2, 4, 6, 7]);
        expect(edgesOf(s.toUndirected({ weights: "max" }).snapshot).map((e) => e[2])).toEqual([3, 5, 4, 6, 7]);
        // idempotent
        const again = first.snapshot.toUndirected();
        expect(again.snapshot).toBe(first.snapshot);
    });

    it("reciprocal: true keeps only paired edges (and self-loops)", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
                [1, 0],
                [2, 2],
                [3, 0],
            ],
        });
        const d = s.toUndirected({ reciprocal: true });
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [2, 2, 1],
        ]);
        expect(Array.from(d.edgeRemap as Uint32Array)).toEqual([0, INVALID_INDEX, 0, 1, INVALID_INDEX]);
        expect(d.report).toEqual({ droppedEdges: 2, mergedEdges: 1 });
    });

    it("an unweighted source stays unweighted unless the reducer is sum (multiplicities)", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 0],
                [1, 2],
            ],
        });
        expect(s.toUndirected().snapshot.weights).toBeNull();
        const summed = s.toUndirected({ weights: "sum" }).snapshot;
        expect(edgesOf(summed).map((e) => e[2])).toEqual([2, 1]);
    });

    it("with no reciprocal pair the edge space is unchanged and the edge table shared", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        const d = s.toUndirected();
        assertMaps(s, d);
        expect(d.edgeRemap).toBeNull();
        expect(d.snapshot.edges).toBe(s.edges);
        expect(d.snapshot.arcCount).toBe(4);
    });
});

describe("transpose", () => {
    it("returns this on an undirected snapshot", () => {
        const s = makeSnapshot({ directed: false, edges: [[0, 1]] });
        expect(s.transpose().snapshot).toBe(s);
    });

    it("swaps every orientation using the reverse view's arrays, sharing both tables (P9)", () => {
        const spec: GraphSpec = {
            directed: true,
            edges: [
                [2, 0, 5],
                [0, 2, 6],
                [1, 2, 7],
                [0, 1, 8],
                [0, 2, 9],
                [1, 1, 10],
            ],
        };
        const s = makeSnapshot(spec);
        const d = s.transpose();
        assertMaps(s, d);
        expect(d.snapshot.rowPtr).toBe(s.reverse().rowPtr);
        expect(d.snapshot.colIdx).toBe(s.reverse().colIdx);
        expect(d.snapshot.arena).toBeNull();
        expect(d.snapshot.edges).toBe(s.edges);
        expect(edgesOf(d.snapshot)).toEqual(spec.edges.map(([u, v, w]) => [v, u, w]));
        expect(d.snapshot.flags.multigraph).toBe(true);
        expect(d.snapshot.selfLoopCount).toBe(1);
        const back = d.snapshot.transpose();
        expect(equalsTopology(back.snapshot, s)).toBe(true);
        expect(Array.from(back.snapshot.edgeToArc)).toEqual(Array.from(s.edgeToArc));
    });

    it("recognises an identity permutation after transposition", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [1, 0],
                [2, 0],
                [2, 1],
            ],
        });
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        const d = s.transpose();
        assertMaps(s, d);
        expect(d.snapshot.flags.arcToEdgeIsIdentity).toBe(true);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [0, 2, 1],
            [1, 2, 1],
        ]);
    });
});

describe("simplified", () => {
    const spec: GraphSpec = {
        directed: false,
        edges: [
            [0, 1, 1],
            [1, 0, 2],
            [1, 2, 3],
            [0, 1, 4],
            [2, 2, 5],
            [2, 2, 6],
        ],
        edgeColumns: {
            w: new Float64Array([1, 2, 3, 4, 5, 6]),
            n: new Uint32Array([1, 2, 3, 4, 5, 6]),
            tag: { values: ["a", "b", "c", "d", "e", "f"], decl: { dtype: "string" } },
            kept: { values: [true, false, true, false, true, false], decl: { dtype: "bool" } },
        },
    };

    it("keeps one edge per group, survivor lowest index, weights per reducer", () => {
        const s = makeSnapshot(spec);
        const d = s.simplified();
        assertMaps(s, d);
        expect(d.snapshot.flags.multigraph).toBe(false);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [1, 2, 3],
            [2, 2, 5],
        ]);
        expect(Array.from(d.edgeRemap as Uint32Array)).toEqual([0, 0, 1, 0, 2, 2]);
        expect(d.report).toEqual({ droppedEdges: 0, mergedEdges: 3 });
        expect(d.snapshot.edges.value("tag", 0)).toBe("a");
        expect(edgesOf(s.simplified({ weights: "sum" }).snapshot).map((e) => e[2])).toEqual([7, 3, 11]);
        expect(edgesOf(s.simplified({ weights: "max" }).snapshot).map((e) => e[2])).toEqual([4, 3, 6]);
        expect(edgesOf(s.simplified({ weights: "last" }).snapshot).map((e) => e[2])).toEqual([4, 3, 6]);
    });

    it("drops self-loops on request", () => {
        const s = makeSnapshot(spec);
        const d = s.simplified({ selfLoops: "drop" });
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [1, 2, 3],
        ]);
        expect(d.report).toEqual({ droppedEdges: 2, mergedEdges: 2 });
    });

    it("applies per-column reducers", () => {
        const s = makeSnapshot(spec);
        const d = s.simplified({
            edgeReducers: { w: "mean", n: "sum", tag: "last", kept: "count" },
        });
        assertMaps(s, d);
        const { edges } = d.snapshot;
        expect(Array.from(edges.requireTyped("w", "f64").data)).toEqual([7 / 3, 3, 5.5]);
        // a "sum" over an integer column widens to f64 (it cannot wrap); "min" / "max" keep the dtype
        expect(Array.from(edges.requireTyped("n", "f64").data)).toEqual([7, 3, 11]);
        expect(edges.requireTyped("tag", "string").decodeAll()).toEqual(["d", "c", "f"]);
        expect(Array.from(edges.requireTyped("kept", "u32").data)).toEqual([3, 1, 2]);
        const dropped = s.simplified({ edgeReducers: { w: "drop", n: "min" } }).snapshot.edges;
        expect(dropped.has("w")).toBe(false);
        expect(Array.from(dropped.requireTyped("n", "u32").data)).toEqual([1, 3, 5]);
        expect(
            Array.from(s.simplified({ edgeReducers: { n: "max" } }).snapshot.edges.requireTyped("n", "u32").data),
        ).toEqual([4, 3, 6]);
        expect(
            Array.from(s.simplified({ edgeReducers: { n: "first" } }).snapshot.edges.requireTyped("n", "u32").data),
        ).toEqual([1, 3, 5]);
    });

    it("rejects a numeric reducer on a non-numeric column with E_COLUMN_TYPE", () => {
        const s = makeSnapshot(spec);
        expectError(() => s.simplified({ edgeReducers: { tag: "sum" } }), "E_COLUMN_TYPE");
        expectError(() => s.simplified({ edgeReducers: { kept: "mean" } }), "E_COLUMN_TYPE");
    });

    it("a simple graph simplifies to a copy with null maps", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 0],
            ],
        });
        const d = s.simplified();
        assertMaps(s, d);
        expect(d.edgeRemap).toBeNull();
        expect(d.snapshot.edges).toBe(s.edges);
    });

    it("directed groups are ordered pairs", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 1],
                [1, 0, 2],
                [0, 1, 3],
            ],
        });
        const d = s.simplified({ weights: "sum" });
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 4],
            [1, 0, 2],
        ]);
    });
});

describe("contract", () => {
    const spec: GraphSpec = {
        directed: false,
        edges: [
            [0, 1, 1],
            [1, 2, 2],
            [2, 3, 3],
            [3, 0, 4],
            [0, 2, 5],
            [1, 1, 6],
        ],
        nodeColumns: {
            mass: new Float64Array([1, 2, 3, 4]),
            parent: { values: [1, 1, 3, 3], decl: { dtype: "u32", refersTo: "node" } },
        },
        edgeColumns: { cap: new Float32Array([10, 20, 30, 40, 50, 60]) },
    };

    it("aggregates at the logical-edge level: one self-loop of weight w per intra-block edge, sums by default", () => {
        const s = makeSnapshot(spec);
        const d = s.contract(new Uint32Array([0, 0, 1, 1]));
        assertMaps(s, d);
        expect(d.snapshot.nodeCount).toBe(2);
        expect(Array.from(d.blockSizes as Uint32Array)).toEqual([2, 2]);
        expect(Array.from(d.nodeOrigin as Uint32Array)).toEqual([0, 2]);
        expect(Array.from(d.nodeRemap as Uint32Array)).toEqual([0, 0, 1, 1]);
        expect(d.snapshot.ids.kind).toBe("identity");
        // groups: {0,0}: e0, e5 -> loop weight 7 ; {0,1}: e1, e3, e4 -> 11 ; {1,1}: e2 -> 3
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 0, 7],
            [0, 1, 11],
            [1, 1, 3],
        ]);
        expect(Array.from(d.edgeRemap as Uint32Array)).toEqual([0, 1, 2, 1, 1, 0]);
        expect(Array.from(d.edgeOrigin as Uint32Array)).toEqual([0, 1, 2]);
        expect(d.report).toEqual({ droppedEdges: 0, mergedEdges: 3 });
        expect(d.snapshot.nodes.names()).toEqual([]);
        expect(d.snapshot.edges.names()).toEqual([]);
        expect(d.snapshot.totalWeight()).toBe(s.totalWeight());
    });

    it("drops intra-block edges and keeps parallels on request", () => {
        const s = makeSnapshot(spec);
        const d = s.contract(new Uint32Array([0, 0, 1, 1]), { selfLoops: "drop", parallel: "keep" });
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 2],
            [1, 0, 4],
            [0, 1, 5],
        ]);
        expect(d.report).toEqual({ droppedEdges: 3, mergedEdges: 0 });
        expect(d.snapshot.flags.multigraph).toBe(true);
    });

    it("reduces node and edge columns per the named reducers, dropping refersTo columns", () => {
        const s = makeSnapshot(spec);
        const d = s.contract(new Uint32Array([0, 0, 1, 1]), {
            nodeReducers: { mass: "sum", parent: "first" },
            edgeReducers: { cap: "max" },
        });
        expect(Array.from(d.snapshot.nodes.requireTyped("mass", "f64").data)).toEqual([3, 7]);
        expect(d.snapshot.nodes.has("parent")).toBe(false);
        expect(Array.from(d.snapshot.edges.requireTyped("cap", "f32").data)).toEqual([60, 50, 30]);
    });

    it("keeps dense labels as block indices and renumbers others in first-seen order (P9c)", () => {
        const s = makeSnapshot(spec);
        const dense = s.contract(new Uint32Array([1, 0, 1, 0]));
        expect(Array.from(dense.nodeRemap as Uint32Array)).toEqual([1, 0, 1, 0]);
        expect(Array.from(dense.nodeOrigin as Uint32Array)).toEqual([1, 0]);
        const sparse = s.contract(new Uint32Array([7, 3, 7, 3]));
        expect(Array.from(sparse.nodeRemap as Uint32Array)).toEqual([0, 1, 0, 1]);
        const image = renumberPartition(new Uint32Array([7, 3, 7, 3]));
        expect(image.count).toBe(2);
        expect(equalsTopology(sparse.snapshot, s.contract(image.labels).snapshot)).toBe(true);
    });

    it("contract(identity) preserves weightedDegree and totalWeight; blockSizes sum to nodeCount (P9c)", () => {
        const s = makeSnapshot({ directed: false, edges: KARATE_EDGES });
        const d = s.contract(new Uint32Array(Array.from({ length: 34 }, (_, i) => i)), { weights: "first" });
        assertMaps(s, d);
        expect(d.snapshot.weights).toBeNull();
        expect(Array.from(d.snapshot.weightedDegree())).toEqual(Array.from(s.weightedDegree()));
        expect(d.snapshot.totalWeight()).toBe(s.totalWeight());
        expect(Array.from(d.blockSizes as Uint32Array).reduce((a, b) => a + b, 0)).toBe(34);
        expect(equalsTopology(d.snapshot, s)).toBe(true);
        expect(d.edgeRemap).toBeNull();
    });

    it("on an unweighted source sum materialises multiplicities and the other reducers keep null", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 1],
                [1, 2],
                [2, 0],
            ],
        });
        const summed = s.contract(new Uint32Array([0, 0, 1]));
        expect(edgesOf(summed.snapshot)).toEqual([
            [0, 0, 2],
            [0, 1, 1],
            [1, 0, 1],
        ]);
        expect(summed.snapshot.flags.weighted).toBe(true);
        const first = s.contract(new Uint32Array([0, 0, 1]), { weights: "first" });
        expect(first.snapshot.weights).toBeNull();
    });

    it("rejects a wrong length or an INVALID_INDEX label with E_PARTITION", () => {
        const s = makeSnapshot(spec);
        expectError(() => s.contract(new Uint32Array([0, 0, 1])), "E_PARTITION");
        expectError(() => s.contract(new Uint32Array([0, 0, INVALID_INDEX, 1])), "E_PARTITION");
        expectError(() => renumberPartition(new Uint32Array([0, INVALID_INDEX])), "E_PARTITION");
    });

    it("directed contraction keeps orientation between blocks", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 1],
                [1, 0, 2],
                [2, 1, 3],
            ],
        });
        const d = s.contract(new Uint32Array([0, 1, 1]));
        assertMaps(s, d);
        expect(edgesOf(d.snapshot)).toEqual([
            [0, 1, 1],
            [1, 0, 2],
            [1, 1, 3],
        ]);
    });

    it("rewrites extension element references through the block map", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 1],
                [1, 2],
            ],
            extensions: {
                t: { rowCount: 2, columns: { element: { values: [2, 0], decl: { dtype: "u32", refersTo: "node" } } } },
            },
        });
        const d = s.contract(new Uint32Array([0, 0, 1]));
        const table = d.snapshot.extensions.get("t");
        expect(Array.from(table?.requireTyped("element", "u32").data ?? [])).toEqual([1, 0]);
    });
});

describe("renumberPartition", () => {
    it("renumbers in first-seen order, in place when out is the input, with a Map for huge labels", () => {
        const labels = new Uint32Array([5, 5, 9, 1, 9]);
        const result = renumberPartition(labels, labels);
        expect(result.labels).toBe(labels);
        expect(Array.from(labels)).toEqual([0, 0, 1, 2, 1]);
        expect(result.count).toBe(3);
        const huge = renumberPartition(new Uint32Array([4_000_000_000, 7, 4_000_000_000]));
        expect(Array.from(huge.labels)).toEqual([0, 1, 0]);
        expect(huge.count).toBe(2);
        expect(renumberPartition(new Uint32Array(0)).count).toBe(0);
        expectError(() => renumberPartition(new Uint32Array([1]), new Uint32Array(2)), "E_COLUMN_LENGTH");
    });
});

describe("withColumns", () => {
    it("shares core, ids and serial with a cloned column set plus the given columns", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
            nodeColumns: { x: new Float32Array([1, 2, 3]) },
            checksum: true,
        });
        const w = s.withColumns(
            { y: new Float32Array([4, 5, 6]) },
            { z: { data: new Uint32Array([7, 8]), decl: { role: "rank" } } },
        );
        expect(w).not.toBe(s);
        expect(w.serial).toBe(s.serial);
        expect(w.rowPtr).toBe(s.rowPtr);
        expect(w.colIdx).toBe(s.colIdx);
        expect(w.ids).toBe(s.ids);
        expect(w.arena).toBe(s.arena);
        expect(w.nodes).not.toBe(s.nodes);
        expect(w.nodes.get("x")).toBe(s.nodes.get("x"));
        expect(w.nodes.has("y")).toBe(true);
        expect(s.nodes.has("y")).toBe(false);
        expect(w.edges.byRole("rank")?.meta.name).toBe("z");
        expect(hasChecksums(w)).toBe(true);
        w.validate({ checksum: true });
        assertInvariants(w);
        expect(equalsTopology(w, s)).toBe(true);
        expect(s.withColumns().nodes.names()).toEqual(["x"]);
    });

    it("keeps an identity permutation lazy on both snapshots", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
        });
        const w = s.withColumns();
        expect(w.flags.arcToEdgeIsIdentity).toBe(true);
        expect(Array.from(w.arcToEdge)).toEqual([0, 1]);
    });
});

describe("derived graphs over random inputs (P9)", () => {
    for (const directed of [true, false]) {
        it(`${directed ? "directed" : "undirected"}: every operation yields a valid snapshot with consistent maps`, () => {
            fc.assert(
                fc.property(edgesArb(true), fc.integer({ min: 1, max: 3 }), ({ n, edges }, blocks) => {
                    const s = makeSnapshot({ directed, nodeCount: n, edges });
                    const partition = new Uint32Array(n).map((_, i) => i % blocks);
                    const keep = makeMask(edges.length);
                    edges.forEach((_, e) => maskSet(keep, e, e % 2 === 0));
                    const half = new Uint32Array(Math.ceil(n / 2)).map((_, i) => i);
                    const derived: DerivedGraph[] = [
                        s.toUndirected(),
                        s.toUndirected({ reciprocal: true, weights: "sum" }),
                        s.transpose(),
                        s.simplified({ weights: "sum" }),
                        s.simplified({ selfLoops: "drop" }),
                        s.withoutSelfLoops(),
                        s.filterEdges(keep),
                        s.inducedSubgraph(half),
                        s.contract(partition),
                        s.contract(partition, { parallel: "keep", selfLoops: "drop", weights: "max" }),
                        s.relabel(s.degreeOrder().perm),
                    ];
                    // contract(identity) with parallels kept is topologically the source; with the default
                    // merge it preserves the weighted degrees and the total weight (P9c)
                    const identityLabels = new Uint32Array(n).map((_, i) => i);
                    const identity = s.contract(identityLabels, { weights: "first", parallel: "keep" });
                    expect(equalsTopology(identity.snapshot, s)).toBe(true);
                    const merged = s.contract(identityLabels);
                    expect(merged.snapshot.totalWeight()).toBeCloseTo(s.totalWeight(), 6);
                    const mergedDegree = merged.snapshot.weightedDegree();
                    s.weightedDegree().forEach((w, u) => expect(mergedDegree[u]).toBeCloseTo(w, 6));
                    // derived graphs do not grow the source's resident views
                    expect(peekView(s, "edgeList")).toBeNull();
                    for (const d of [...derived, identity, merged]) {
                        assertMaps(s, d);
                    }
                    // simplified never leaves parallels; withoutSelfLoops never leaves loops
                    expect(derived[3].snapshot.flags.multigraph).toBe(false);
                    expect(derived[5].snapshot.flags.hasSelfLoops).toBe(false);
                }),
                { numRuns: 40 },
            );
        });
    }
});
