/**
 * Unit tests of the freeze pipeline (design sections 6.3-6.7, 3.7, 4.2 and 10.3): remaps and prefix
 * stability, the self-loop and duplicate policies with their error codes, the f64 / explicit-weight
 * shadow column, the identity fast path, the arena layout, id-map kind selection and sharing, the
 * undirected pairing, and the freeze options (label, prepare, checksum, profile, arena).
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { permutationMaterialised } from "../../src/snapshot/graph-snapshot.js";
import { type DuplicatePolicy } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";

function code(fn: () => unknown): GraphFormatErrorCode | null {
    try {
        fn();
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return err.code;
        }
        throw err;
    }
    return null;
}

function thrown(fn: () => unknown): GraphFormatError {
    try {
        fn();
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a GraphFormatError");
}

function remap(array: Uint32Array | null): number[] | null {
    return array === null ? null : Array.from(array);
}

describe("freeze: empty and node-only builders", () => {
    it("freezes an empty builder to a valid snapshot", () => {
        for (const directed of [true, false]) {
            const s = new GraphBuilder({ directed }).freeze();
            assertInvariants(s);
            expect(s.nodeCount).toBe(0);
            expect(s.edgeCount).toBe(0);
            expect(s.arcCount).toBe(0);
            expect(Array.from(s.rowPtr)).toEqual([0]);
            expect(s.colIdx.length).toBe(0);
            expect(s.arcToEdge.length).toBe(0);
            expect(s.edgeToArc.length).toBe(0);
            expect(s.weights).toBeNull();
            expect(s.flags.arcToEdgeIsIdentity).toBe(directed);
            expect(s.arena).not.toBeNull();
            expect(s.arena?.segments.colIdx).toBeNull();
            expect(s.ids.kind).toBe("identity");
        }
    });

    it("freezes nodes without edges", () => {
        const b = new GraphBuilder({ directed: false });
        b.addNodes(["x", "y"]);
        const s = b.freeze();
        assertInvariants(s);
        expect(s.nodeCount).toBe(2);
        expect(Array.from(s.rowPtr)).toEqual([0, 0, 0]);
        expect(s.ids.kind).toBe("string");
        expect(s.edgeToArc.length).toBe(0);
    });
});

describe("freeze: remaps and prefix stability (I16)", () => {
    it("reports null remaps for append-only freezes and shares index prefixes", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        const first = b.freezeWithReport();
        expect(first.report.nodeRemap).toBeNull();
        expect(first.report.edgeRemap).toBeNull();
        expect(first.report.compacted).toBe(false);
        b.addEdge("c", "a");
        b.addEdge("b", "c");
        const second = b.freezeWithReport();
        expect(second.report.nodeRemap).toBeNull();
        expect(second.report.edgeRemap).toBeNull();
        expect(second.snapshot.ids.toArray().slice(0, 2)).toEqual(first.snapshot.ids.toArray());
        expect(second.snapshot.edgeSource(0)).toBe(first.snapshot.edgeSource(0));
        expect(second.snapshot.edgeTarget(0)).toBe(first.snapshot.edgeTarget(0));
    });

    it("reports remaps over the builder's own index space on the first freeze after removals", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.addEdge("c", "d");
        b.removeNode("b");
        const { snapshot, report } = b.freezeWithReport();
        expect(remap(report.nodeRemap)).toEqual([0, INVALID_INDEX, 1, 2]);
        expect(remap(report.edgeRemap)).toEqual([INVALID_INDEX, INVALID_INDEX, 0]);
        expect(report.compacted).toBe(true);
        expect(report.droppedEdges).toBe(2);
        expect(snapshot.ids.toArray()).toEqual(["a", "c", "d"]);
        expect(snapshot.edgeSource(0)).toBe(1);
        expect(snapshot.edgeTarget(0)).toBe(2);
        assertInvariants(snapshot);
        // the builder now uses the snapshot's indices
        expect(b.indexOf("d")).toBe(2);
        expect(b.edgeEndpoints(0)).toEqual([1, 2]);
        // and the next freeze is relative to this one
        b.addEdge("d", "a");
        const next = b.freezeWithReport();
        expect(next.report.nodeRemap).toBeNull();
        expect(next.report.edgeRemap).toBeNull();
        expect(next.snapshot.edgeCount).toBe(2);
    });

    it("keeps node and edge remaps independent", () => {
        const nodesOnly = new GraphBuilder({ directed: true });
        nodesOnly.addNodes(["a", "b", "c"]);
        nodesOnly.addEdge("a", "c");
        nodesOnly.removeNode("b");
        const r1 = nodesOnly.freezeWithReport().report;
        expect(remap(r1.nodeRemap)).toEqual([0, INVALID_INDEX, 1]);
        expect(r1.edgeRemap).toBeNull();
        const edgesOnly = new GraphBuilder({ directed: true });
        edgesOnly.addEdge("a", "b");
        edgesOnly.addEdge("b", "a");
        edgesOnly.removeEdge(0);
        const r2 = edgesOnly.freezeWithReport().report;
        expect(r2.nodeRemap).toBeNull();
        expect(remap(r2.edgeRemap)).toEqual([INVALID_INDEX, 0]);
        expect(r2.compacted).toBe(true);
    });

    it("carries node and edge columns and refersTo values through compaction", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.addEdge("c", "a");
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node" });
        b.setNodeValue("parent", 0, 1);
        b.setNodeValue("parent", 2, 0);
        b.declareEdgeColumn({ name: "pair", dtype: "u32", refersTo: "edge" });
        b.setEdgeValue("pair", 0, 1);
        b.setEdgeValue("pair", 2, 0);
        b.declareNodeColumn({ name: "parents", dtype: "list", itemDtype: "u32", refersTo: "node" });
        b.setNodeValue("parents", 2, [0, 1]);
        b.setNodeValue("parents", 0, [1]);
        b.setNodeValue("label", 2, "C");
        b.removeNode("b");
        const s = b.freeze();
        assertInvariants(s);
        expect(s.ids.toArray()).toEqual(["a", "c"]);
        const parent = s.nodes.requireTyped("parent", "u32");
        expect(parent.isSet(0)).toBe(false);
        expect(parent.data[0]).toBe(INVALID_INDEX);
        expect(parent.value(1)).toBe(0);
        const pair = s.edges.requireTyped("pair", "u32");
        expect(s.edgeCount).toBe(1);
        expect(pair.isSet(0)).toBe(false);
        expect(s.nodes.value("parents", 1)).toEqual([0]);
        expect(s.nodes.isSet("parents", 0)).toBe(false);
        expect(s.nodes.value("label", 1)).toBe("C");
        expect(s.nodes.isSet("label", 0)).toBe(false);
    });
});

describe("freeze: self-loop policy", () => {
    it("keeps loops by default, counting each once (I7)", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "a");
        b.addEdge("a", "b");
        const s = b.freeze();
        assertInvariants(s);
        expect(s.selfLoopCount).toBe(1);
        expect(s.arcCount).toBe(3);
        expect(s.flags.hasSelfLoops).toBe(true);
        expect(s.outDegree()[0]).toBe(2);
        expect(s.degree()[0]).toBe(3);
    });

    it("drops loops at freeze, reporting them and rewriting the builder", () => {
        const b = new GraphBuilder({ directed: true, selfLoops: "drop" });
        b.addEdge("a", "a");
        b.addEdge("a", "b");
        b.addEdge("b", "b");
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.selfLoopCount).toBe(0);
        expect(report.droppedSelfLoops).toBe(2);
        expect(report.droppedEdges).toBe(2);
        expect(remap(report.edgeRemap)).toEqual([INVALID_INDEX, 0, INVALID_INDEX]);
        expect(report.nodeRemap).toBeNull();
        expect(b.edgeCount).toBe(1);
        expect(b.edgeBound).toBe(1);
        expect(b.freezeWithReport().report.edgeRemap).toBeNull();
    });

    it("refuses loops with E_SELF_LOOP naming the first one, leaving the builder untouched", () => {
        const b = new GraphBuilder({ directed: true, selfLoops: "error" });
        b.addEdge("a", "b");
        b.addEdge("c", "c");
        b.addEdge("d", "d");
        b.removeNode("a");
        const before = { m: b.mutationCount, n: b.nodeBound, e: b.edgeBound };
        const err = thrown(() => b.freeze());
        expect(err.code).toBe("E_SELF_LOOP");
        expect(err.details.edge).toBe(1);
        expect(err.details.node).toBe(2);
        expect(b.mutationCount).toBe(before.m);
        expect(b.nodeBound).toBe(before.n);
        expect(b.edgeBound).toBe(before.e);
        b.removeEdge(1);
        b.removeEdge(2);
        expect(b.freeze().selfLoopCount).toBe(0);
    });
});

describe("freeze: duplicate policy", () => {
    function parallel(directed: boolean, policy: DuplicatePolicy): GraphBuilder {
        const b = new GraphBuilder({ directed, duplicateEdges: policy });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c");
        b.addEdge("a", "b", 5);
        b.addEdge("b", "a", 3);
        b.addEdge("c", "c", 4);
        b.addEdge("c", "c", 6);
        b.setEdgeValue("kind", 0, "first-ab");
        b.setEdgeValue("kind", 2, "second-ab");
        b.setEdgeValue("kind", 3, "ba");
        return b;
    }

    it("keeps parallels by default, adjacent and in edge order", () => {
        const s = parallel(true, "keep").freeze();
        assertInvariants(s);
        expect(s.flags.multigraph).toBe(true);
        expect(s.multiplicity(0, 1)).toBe(2);
        expect(s.edgeCount).toBe(6);
        const [lo, hi] = s.arcsBetween(0, 1);
        expect(Array.from(s.arcToEdge.subarray(lo, hi))).toEqual([0, 2]);
    });

    it("throws E_DUPLICATE_EDGE with the pair and both edges under error, leaving the builder untouched", () => {
        const b = parallel(true, "error");
        b.removeEdge(1);
        const bound = b.edgeBound;
        const err = thrown(() => b.freeze());
        expect(err.code).toBe("E_DUPLICATE_EDGE");
        expect(err.details).toEqual({ source: 0, target: 1, edges: [0, 2] });
        expect(b.edgeBound).toBe(bound);
        expect(b.edgeCount).toBe(5);
        b.removeEdge(2);
        b.removeEdge(5);
        const s = b.freeze();
        expect(s.flags.multigraph).toBe(false);
        expect(s.edgeCount).toBe(3);
    });

    it("error on an undirected graph treats (u, v) and (v, u) as one group", () => {
        const b = new GraphBuilder({ directed: false, duplicateEdges: "error" });
        b.addEdge("a", "b");
        b.addEdge("b", "a");
        const err = thrown(() => b.freeze());
        expect(err.details).toEqual({ source: 0, target: 1, edges: [0, 1] });
        const ok = new GraphBuilder({ directed: true, duplicateEdges: "error" });
        ok.addEdge("a", "b");
        ok.addEdge("b", "a");
        expect(ok.freeze().edgeCount).toBe(2);
    });

    it.each<[DuplicatePolicy, number[], number[], (string | undefined)[]]>([
        ["first", [2, 1, 3, 4], [0, 1, 0, 2, 3, 3], ["first-ab", undefined, "ba", undefined]],
        ["last", [1, 5, 3, 6], [1, 0, 1, 2, 3, 3], [undefined, "second-ab", "ba", undefined]],
        ["sum", [7, 1, 3, 10], [0, 1, 0, 2, 3, 3], ["first-ab", undefined, "ba", undefined]],
        ["min", [2, 1, 3, 4], [0, 1, 0, 2, 3, 3], ["first-ab", undefined, "ba", undefined]],
        ["max", [5, 1, 3, 6], [0, 1, 0, 2, 3, 3], ["first-ab", undefined, "ba", undefined]],
    ])("merges directed parallels under %s", (policy, weights, edgeRemap, kinds) => {
        const b = parallel(true, policy);
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.edgeCount).toBe(4);
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual(weights);
        expect(remap(report.edgeRemap)).toEqual(edgeRemap);
        expect(report.mergedEdges).toBe(2);
        expect(report.droppedEdges).toBe(2);
        expect(report.nodeRemap).toBeNull();
        expect(report.compacted).toBe(true);
        kinds.forEach((kind, e) => {
            expect(snapshot.edges.isSet("kind", e)).toBe(kind !== undefined);
            if (kind !== undefined) {
                expect(snapshot.edges.value("kind", e)).toBe(kind);
            }
        });
        // the builder was rewritten to the merged edge set, so re-freezing is idempotent
        expect(b.edgeCount).toBe(4);
        expect(b.edgeBound).toBe(4);
        expect(b.edgeWeight(0)).toBe(weights[0]);
        const again = b.freezeWithReport();
        expect(again.report.edgeRemap).toBeNull();
        expect(again.report.mergedEdges).toBe(0);
        expect(Array.from(again.snapshot.edgeList().weights as Float32Array)).toEqual(weights);
    });

    it("merges undirected parallels in either orientation as one group, keeping the survivor's orientation", () => {
        const b = parallel(false, "sum");
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(snapshot.edgeCount).toBe(3);
        expect(remap(report.edgeRemap)).toEqual([0, 1, 0, 0, 2, 2]);
        expect(report.mergedEdges).toBe(3);
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual([10, 1, 10]);
        expect(snapshot.edgeSource(0)).toBe(0);
        expect(snapshot.edgeTarget(0)).toBe(1);
        expect(Array.from(snapshot.mate())).toEqual(Array.from(snapshot.mate()));
        const last = parallel(false, "last").freezeWithReport();
        expect(last.snapshot.edgeSource(1)).toBe(1);
        expect(last.snapshot.edgeTarget(1)).toBe(0);
        expect(Array.from(last.snapshot.edgeList().weights as Float32Array)).toEqual([1, 3, 6]);
        expect(remap(last.report.edgeRemap)).toEqual([1, 0, 1, 1, 2, 2]);
    });

    it("sum on an unweighted builder yields multiplicities and allocates weights", () => {
        const b = new GraphBuilder({ directed: true, duplicateEdges: "sum" });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("b", "a");
        const s = b.freeze();
        expect(s.flags.weighted).toBe(true);
        expect(Array.from(s.edgeList().weights as Float32Array)).toEqual([3, 1]);
        expect(b.edgeWeight(0)).toBe(3);
        const minB = new GraphBuilder({ directed: true, duplicateEdges: "min" });
        minB.addEdge("a", "b");
        minB.addEdge("a", "b");
        expect(minB.freeze().weights).toBeNull();
    });

    it("applies a per-freeze policy override that rewrites the builder permanently", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 2);
        expect(b.freeze().edgeCount).toBe(2);
        const merged = b.freezeWithReport({ duplicateEdges: "max" });
        expect(merged.snapshot.edgeCount).toBe(1);
        expect(merged.report.mergedEdges).toBe(1);
        expect(remap(merged.report.edgeRemap)).toEqual([0, 0]);
        expect(b.edgeCount).toBe(1);
        expect(b.edgeWeight(0)).toBe(2);
        expect(b.freeze().edgeCount).toBe(1);
        expect(code(() => b.freeze({ duplicateEdges: "error" }))).toBeNull();
    });

    it("a non-keep policy with no duplicates still yields an arena-backed core", () => {
        const b = new GraphBuilder({ directed: false, duplicateEdges: "sum" });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c", 3);
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.arena).not.toBeNull();
        expect(snapshot.colIdx.buffer).toBe(snapshot.arena?.buffer);
        expect(report.edgeRemap).toBeNull();
        expect(report.mergedEdges).toBe(0);
        const separate = b.freeze({ arena: false });
        expect(separate.arena).toBeNull();
    });

    it("merges edges whose explicit-weight record differs, marking the survivor explicit", () => {
        const b = new GraphBuilder({ directed: true, duplicateEdges: "sum" });
        b.addEdge("a", "b");
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c");
        const s = b.freeze();
        const shadow = s.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.value(0)).toBe(3);
        expect(shadow?.isSet(1)).toBe(false);
    });
});

describe("freeze: weights and the shadow column (design section 3.7)", () => {
    it("stores no column when every weight is explicit and f32-exact", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c", 16777216);
        const s = b.freeze();
        expect(s.edges.byRole("weight")).toBeNull();
        expect(s.edges.names()).toEqual([]);
    });

    it("keeps an f64 shadow when a value is not f32-exact", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b", 0.1);
        b.addEdge("b", "c", 16777217);
        const s = b.freeze();
        assertInvariants(s);
        const shadow = s.edges.requireTyped("graphty.weight", "f64");
        expect(shadow.meta.role).toBe("weight");
        expect(shadow.validity).toBeNull();
        expect(Array.from(shadow.data)).toEqual([0.1, 16777217]);
        expect(Array.from(s.weights as Float32Array)).toEqual([Math.fround(0.1), 16777216]);
    });

    it("f32 staging never keeps a shadow for inexact values", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f32" });
        b.addEdge("a", "b", 0.1);
        expect(b.freeze().edges.byRole("weight")).toBeNull();
        expect(b.edgeWeight(0)).toBe(Math.fround(0.1));
    });

    it("keeps an f32 column with validity when some edges omitted their weight", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 2);
        b.addEdge("b", "c");
        b.addEdge("c", "a", 4);
        const s = b.freeze();
        const shadow = s.edges.requireTyped("graphty.weight", "f32");
        expect(shadow.nullCount).toBe(1);
        expect([0, 1, 2].map((e) => shadow.isSet(e))).toEqual([true, false, true]);
        expect(Array.from(shadow.data)).toEqual([2, 1, 4]);
        expect(shadow.value(1)).toBeUndefined();
    });

    it("keeps an f64 column with validity when both rules apply", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.addEdge("a", "b");
        b.addEdge("b", "c", 0.1);
        const shadow = b.freeze().edges.requireTyped("graphty.weight", "f64");
        expect(shadow.isSet(0)).toBe(false);
        expect(shadow.value(1)).toBe(0.1);
    });

    it("keeps an all-clear column for a weighted: true builder whose edges omitted every weight", () => {
        const b = new GraphBuilder({ directed: true, weighted: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        const s = b.freeze();
        const shadow = s.edges.requireTyped("graphty.weight", "f32");
        expect(shadow.nullCount).toBe(2);
        expect(s.flags.allWeightsOne).toBe(true);
        expect(new GraphBuilder({ directed: true, weighted: true }).freeze().edges.byRole("weight")).toBeNull();
    });

    it("both arcs of an undirected edge carry the same weight and the flags are truthful", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", -2);
        b.addEdge("b", "c", Infinity);
        b.addEdge("c", "c", 0);
        const s = b.freeze();
        assertInvariants(s);
        const mate = s.mate();
        for (let a = 0; a < s.arcCount; a++) {
            expect((s.weights as Float32Array)[a]).toBe((s.weights as Float32Array)[mate[a]]);
        }
        expect(s.flags).toEqual({
            multigraph: false,
            hasSelfLoops: true,
            arcToEdgeIsIdentity: false,
            weighted: true,
            allWeightsOne: false,
            nonNegativeWeights: false,
            finiteWeights: false,
        });
    });
});

describe("freeze: identity permutation and the arena (design sections 3.1 and 10.3)", () => {
    it("detects sorted directed input, keeps the permutations out of the arena and materialises them lazily", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        b.addEdges(new Uint32Array([0, 0, 1, 2]), new Uint32Array([1, 2, 2, 0]), new Float32Array([1, 2, 3, 4]));
        const s = b.freeze();
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        expect(permutationMaterialised(s)).toBe(false);
        expect(s.arena?.segments.arcToEdge).toBeNull();
        expect(s.arena?.segments.edgeToArc).toBeNull();
        expect(s.arena?.byteLength).toBe(256 + 256 + 16);
        expect(s.arena?.hotByteLength).toBe(256 + 256 + 16);
        expect(Array.from(s.arcToEdge)).toEqual([0, 1, 2, 3]);
        expect(s.arcToEdge.buffer).not.toBe(s.arena?.buffer);
        expect(permutationMaterialised(s)).toBe(true);
        assertInvariants(s);
    });

    it("materialises the permutations for unsorted directed input and for every undirected graph", () => {
        const unsorted = new GraphBuilder({ directed: true });
        unsorted.addNodes(["a", "b", "c"]);
        unsorted.addEdge("a", "c");
        unsorted.addEdge("a", "b");
        const s1 = unsorted.freeze();
        expect(s1.flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(s1.arcToEdge)).toEqual([1, 0]);
        expect(Array.from(s1.edgeToArc)).toEqual([1, 0]);
        expect(s1.arena?.segments.arcToEdge).not.toBeNull();
        const loops = new GraphBuilder({ directed: false });
        loops.addEdge("a", "a");
        loops.addEdge("b", "b");
        const s2 = loops.freeze();
        expect(s2.arcCount).toBe(s2.edgeCount);
        expect(s2.flags.arcToEdgeIsIdentity).toBe(false);
        expect(s2.arena?.segments.arcToEdge).not.toBeNull();
        assertInvariants(s2);
    });

    it("lays the arena out hot to cold at 256-byte offsets with the documented hotByteLength", () => {
        const b = new GraphBuilder({ directed: false });
        for (let i = 0; i < 100; i++) {
            b.addEdge(i, (i * 7) % 100, i);
        }
        const s = b.freeze();
        assertInvariants(s);
        const arena = s.arena as NonNullable<typeof s.arena>;
        expect(arena.byteOffset).toBe(0);
        expect(arena.alignment).toBe(256);
        const rowPtrBytes = 4 * 101;
        const arcBytes = 4 * s.arcCount;
        expect(arena.segments.rowPtr).toEqual({ byteOffset: 0, byteLength: rowPtrBytes });
        expect(arena.segments.colIdx).toEqual({ byteOffset: 512, byteLength: arcBytes });
        const weightsOffset = 512 + Math.ceil(arcBytes / 256) * 256;
        expect(arena.segments.weights).toEqual({ byteOffset: weightsOffset, byteLength: arcBytes });
        expect(arena.hotByteLength).toBe(weightsOffset + arcBytes);
        const arcToEdgeOffset = weightsOffset + Math.ceil(arcBytes / 256) * 256;
        expect(arena.segments.arcToEdge).toEqual({ byteOffset: arcToEdgeOffset, byteLength: arcBytes });
        const edgeToArcOffset = arcToEdgeOffset + Math.ceil(arcBytes / 256) * 256;
        expect(arena.segments.edgeToArc).toEqual({ byteOffset: edgeToArcOffset, byteLength: 4 * s.edgeCount });
        expect(arena.byteLength).toBe(edgeToArcOffset + 4 * s.edgeCount);
        for (const array of [s.rowPtr, s.colIdx, s.weights as Float32Array, s.arcToEdge, s.edgeToArc]) {
            expect(array.buffer).toBe(arena.buffer);
        }
    });

    it("arena: false allocates separate buffers that still satisfy I10", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", 2);
        const s = b.freeze({ arena: false });
        assertInvariants(s);
        expect(s.arena).toBeNull();
        expect(s.rowPtr.buffer).not.toBe(s.colIdx.buffer);
    });

    it("never aliases staging (I18): mutating the builder afterwards leaves the snapshot alone", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 2);
        b.setNodeValue("x", 0, 1);
        const s = b.freeze({ checksum: true });
        b.addEdge("b", "c", 5);
        b.setEdgeWeight(0, 9);
        b.setNodeValue("x", 0, 2);
        b.removeNode("a");
        s.validate({ checksum: true, level: "full" });
        expect(Array.from(s.weights as Float32Array)).toEqual([2]);
        expect(s.nodes.value("x", 0)).toBe(1);
        expect(s.ids.size).toBe(2);
    });
});

describe("freeze: id map (design section 4.2)", () => {
    it("selects identity for 0-based numeric ids and for anonymous nodes", () => {
        const named = new GraphBuilder({ directed: true });
        named.addEdge(0, 1);
        named.addEdge(1, 2);
        expect(named.freeze().ids.kind).toBe("identity");
        const oneBased = new GraphBuilder({ directed: true });
        oneBased.addNodes([1, 2, 3]);
        const s = oneBased.freeze();
        expect(s.ids.kind).toBe("identity");
        expect(s.ids.offset).toBe(1);
        expect(s.ids.indexOf(3)).toBe(2);
    });

    it("selects dense, numeric, string and mixed kinds", () => {
        const dense = new GraphBuilder({ directed: true });
        dense.addNodes([5, 0, 3]);
        expect(dense.freeze().ids.kind).toBe("dense");
        const numeric = new GraphBuilder({ directed: true });
        numeric.addNodes([1.5, 1000]);
        expect(numeric.freeze().ids.kind).toBe("numeric");
        const strings = new GraphBuilder({ directed: true });
        strings.addNodes(["x", "y"]);
        expect(strings.freeze().ids.kind).toBe("string");
        const mixed = new GraphBuilder({ directed: true });
        mixed.addNodes([1, "1"]);
        const s = mixed.freeze();
        expect(s.ids.kind).toBe("mixed");
        expect(s.ids.indexOf("1")).toBe(1);
        expect(s.ids.indexOf(1)).toBe(0);
    });

    it("shares the builder's map by reference, guarded by nodeCount, until a compacting freeze", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b"]);
        const first = b.freeze();
        b.addNode("c");
        expect(first.ids.indexOf("c")).toBe(INVALID_INDEX);
        expect(first.ids.has("c")).toBe(false);
        expect(first.ids.size).toBe(2);
        expect(first.ids.toArray()).toEqual(["a", "b"]);
        const second = b.freeze();
        expect(second.ids.indexOf("c")).toBe(2);
        b.removeNode("a");
        const third = b.freeze();
        expect(third.ids.toArray()).toEqual(["b", "c"]);
        expect(second.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(second.ids.indexOf("a")).toBe(0);
        expect(first.ids.indexOf("b")).toBe(1);
        b.addNode("d");
        expect(third.ids.indexOf("d")).toBe(INVALID_INDEX);
        assertInvariants(third);
    });
});

describe("freeze: undirected pairing (design section 6.4)", () => {
    it("pairs the k-th (u, v) arc with the k-th (v, u) arc for parallel edges", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", 1);
        b.addEdge("b", "a", 2);
        b.addEdge("a", "b", 3);
        b.addEdge("a", "a", 4);
        const s = b.freeze();
        assertInvariants(s);
        const mate = s.mate();
        const [lo, hi] = s.arcsBetween(0, 1);
        const [lo2] = s.arcsBetween(1, 0);
        for (let k = 0; k < hi - lo; k++) {
            expect(s.arcToEdge[lo + k]).toBe(s.arcToEdge[lo2 + k]);
            expect(mate[lo + k]).toBe(lo2 + k);
            expect(mate[lo2 + k]).toBe(lo + k);
        }
        expect(Array.from(s.arcToEdge.subarray(lo, hi))).toEqual([0, 1, 2]);
        const loop = s.findArc(0, 0);
        expect(mate[loop]).toBe(loop);
        expect(s.edgeSource(1)).toBe(1);
        expect(s.edgeTarget(1)).toBe(0);
        expect(s.reverse().rowPtr).toBe(s.rowPtr);
    });
});

describe("freeze: options", () => {
    it("carries the label, prepares views, records checksums and profiles phases", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "a");
        const { snapshot, report } = b.freezeWithReport({
            label: "test",
            prepare: ["reverse", "outDegree"],
            checksum: true,
            profile: true,
        });
        expect(snapshot.label).toBe("test");
        expect(snapshot.cachedViews()).toEqual(expect.arrayContaining(["reverse", "outDegree"]));
        snapshot.validate({ checksum: true });
        expect(Object.keys(report.timings)).toEqual(
            expect.arrayContaining(["compact", "sort", "weights", "ids", "columns", "snapshot", "total"]),
        );
        expect(report.timings.total).toBeGreaterThanOrEqual(0);
        const plain = b.freezeWithReport();
        expect(plain.snapshot.label).toBeNull();
        expect(plain.report.timings).toEqual({});
        expect(plain.snapshot.cachedViews()).toEqual([]);
        expect(code(() => plain.snapshot.validate({ checksum: true }))).toBe("E_INVALID_SNAPSHOT");
    });

    it("freezes with an empty prepare list and duplicate-policy profiling", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        const { report } = b.freezeWithReport({ prepare: [], profile: true, duplicateEdges: "first" });
        expect(report.timings.duplicates).toBeGreaterThanOrEqual(0);
        expect(report.mergedEdges).toBe(1);
    });
});
