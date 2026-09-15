/**
 * Adversarial audit of the builder's remaps (design sections 4.4, 6.3 step 1, 6.6 and invariant
 * I16): every FreezeReport remap must be relative to the PREVIOUS successful freeze of the same
 * builder, through interleaved add / remove / revive / freeze sequences three and four freezes deep;
 * a freeze that throws after compacting must not move the "previous freeze" baseline; the report's
 * counts and timings must be exact; and the index-space paths must refuse dead or out-of-range
 * indices with the documented code before pushing anything.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { type GraphSnapshot, type U32 } from "../../src/types/index.js";
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

function details(fn: () => unknown): Readonly<Record<string, unknown>> {
    try {
        fn();
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return err.details;
        }
        throw err;
    }
    throw new Error("expected a GraphFormatError");
}

function list(array: U32 | null): number[] | null {
    return array === null ? null : Array.from(array);
}

function edgesOf(s: GraphSnapshot): [number, number][] {
    const out: [number, number][] = [];
    for (let e = 0; e < s.edgeCount; e++) {
        out.push([s.edgeSource(e), s.edgeTarget(e)]);
    }
    return out;
}

/**
 * Check a report against the previous snapshot: every previous live node maps to the node with the
 * same id (or INVALID_INDEX when it no longer exists) and every previous edge maps to an edge with
 * the same endpoint ids (or INVALID_INDEX when it no longer exists), which is property P3 of design
 * section 16.1 applied to one freeze pair.
 */
function checkRemapAgainst(
    previous: GraphSnapshot,
    next: GraphSnapshot,
    nodeRemap: U32 | null,
    edgeRemap: U32 | null,
): void {
    for (let i = 0; i < previous.nodeCount; i++) {
        const id = previous.ids.idOf(i);
        const mapped = nodeRemap === null ? i : nodeRemap[i];
        if (mapped === INVALID_INDEX) {
            expect(next.ids.has(id)).toBe(false);
        } else {
            expect(next.ids.idOf(mapped)).toBe(id);
        }
    }
    for (let e = 0; e < previous.edgeCount; e++) {
        const u = previous.ids.idOf(previous.edgeSource(e));
        const v = previous.ids.idOf(previous.edgeTarget(e));
        const mapped = edgeRemap === null ? e : edgeRemap[e];
        if (mapped === INVALID_INDEX) {
            continue;
        }
        expect(next.ids.idOf(next.edgeSource(mapped))).toBe(u);
        expect(next.ids.idOf(next.edgeTarget(mapped))).toBe(v);
    }
}

describe("audit: remaps relative to the previous freeze (I16, P3)", () => {
    it("four freezes deep: removal, append, edge removal, revival, each reported against the freeze before", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b"); // e0
        b.addEdge("b", "c"); // e1
        b.addEdge("c", "d"); // e2
        b.addEdge("d", "a"); // e3
        const f1 = b.freezeWithReport();
        assertInvariants(f1.snapshot);
        expect(f1.report.nodeRemap).toBeNull();
        expect(f1.report.edgeRemap).toBeNull();
        expect(f1.report.compacted).toBe(false);

        // freeze 2: remove b (kills e0, e1), append node e and edge e -> a
        expect(Array.from(b.removeNode("b"))).toEqual([0, 1]);
        b.addEdge("e", "a"); // node 4, e4
        const f2 = b.freezeWithReport();
        assertInvariants(f2.snapshot);
        // the old space is f1's plus the appended node / edge
        expect(list(f2.report.nodeRemap)).toEqual([0, INVALID_INDEX, 1, 2, 3]);
        expect(list(f2.report.edgeRemap)).toEqual([INVALID_INDEX, INVALID_INDEX, 0, 1, 2]);
        expect(f2.report.droppedEdges).toBe(2);
        expect(f2.snapshot.ids.toArray()).toEqual(["a", "c", "d", "e"]);
        expect(edgesOf(f2.snapshot)).toEqual([
            [1, 2],
            [2, 0],
            [3, 0],
        ]);
        checkRemapAgainst(f1.snapshot, f2.snapshot, f2.report.nodeRemap, f2.report.edgeRemap);
        // the builder speaks the new indices
        expect(b.indexOf("e")).toBe(3);
        expect(b.edgeEndpoints(0)).toEqual([1, 2]);
        expect(Array.from(b.outEdgesOf(0))).toEqual([]);
        expect(Array.from(b.inEdgesOf(0))).toEqual([1, 2]);

        // freeze 3: remove the edge that is now index 0 (old e2), append a -> e and a new node b
        expect(b.removeEdge(0)).toBe(true);
        b.addEdge("a", "e"); // e3 in the builder's space
        expect(b.addNode("b")).toBe(4); // a fresh index: the compacting freeze forgot the old b
        const f3 = b.freezeWithReport();
        assertInvariants(f3.snapshot);
        expect(f3.report.nodeRemap).toBeNull();
        expect(list(f3.report.edgeRemap)).toEqual([INVALID_INDEX, 0, 1, 2]);
        expect(f3.report.compacted).toBe(true);
        expect(f3.report.droppedEdges).toBe(1);
        expect(f3.snapshot.ids.toArray()).toEqual(["a", "c", "d", "e", "b"]);
        expect(edgesOf(f3.snapshot)).toEqual([
            [2, 0],
            [3, 0],
            [0, 3],
        ]);
        checkRemapAgainst(f2.snapshot, f3.snapshot, f3.report.nodeRemap, f3.report.edgeRemap);

        // freeze 4: remove e (kills e1, e2 of freeze 3), revive it before freezing
        expect(Array.from(b.removeNode("e"))).toEqual([1, 2]);
        expect(b.addNode("e")).toBe(3);
        const f4 = b.freezeWithReport();
        assertInvariants(f4.snapshot);
        expect(f4.report.nodeRemap).toBeNull();
        expect(list(f4.report.edgeRemap)).toEqual([0, INVALID_INDEX, INVALID_INDEX]);
        expect(f4.snapshot.ids.toArray()).toEqual(["a", "c", "d", "e", "b"]);
        expect(edgesOf(f4.snapshot)).toEqual([[2, 0]]);
        checkRemapAgainst(f3.snapshot, f4.snapshot, f4.report.nodeRemap, f4.report.edgeRemap);

        // every earlier snapshot is untouched by the later compactions
        expect(f1.snapshot.ids.toArray()).toEqual(["a", "b", "c", "d"]);
        expect(edgesOf(f1.snapshot)).toEqual([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]);
        expect(f2.snapshot.ids.toArray()).toEqual(["a", "c", "d", "e"]);
    });

    it("undirected: removing a node with a self-loop and parallels reports every incident edge once", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "a"); // e0 loop
        b.addEdge("a", "b"); // e1
        b.addEdge("b", "a"); // e2 parallel in the other orientation
        b.addEdge("c", "c"); // e3 loop elsewhere
        b.addEdge("b", "c"); // e4
        const f1 = b.freezeWithReport();
        expect(f1.snapshot.selfLoopCount).toBe(2);
        const removed = b.removeNode("a");
        expect(Array.from(removed)).toEqual([0, 1, 2]);
        expect(b.edgeCount).toBe(2);
        expect(b.nodeCount).toBe(2);
        const f2 = b.freezeWithReport();
        assertInvariants(f2.snapshot);
        expect(list(f2.report.nodeRemap)).toEqual([INVALID_INDEX, 0, 1]);
        expect(list(f2.report.edgeRemap)).toEqual([INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, 0, 1]);
        expect(f2.report.droppedEdges).toBe(3);
        expect(f2.report.droppedSelfLoops).toBe(0);
        expect(f2.snapshot.selfLoopCount).toBe(1);
        expect(f2.snapshot.arcCount).toBe(3);
        checkRemapAgainst(f1.snapshot, f2.snapshot, f2.report.nodeRemap, f2.report.edgeRemap);
    });

    it("a freeze that throws after compacting leaves the baseline at the last successful freeze", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b"); // e0
        b.addEdge("b", "c"); // e1
        b.addEdge("c", "d"); // e2
        b.addEdge("d", "e"); // e3
        const f1 = b.freezeWithReport();
        b.declareEdgeColumn({ name: "eid", dtype: "string", role: "id", unique: true });
        b.setEdgeValue("eid", 2, "x");
        b.setEdgeValue("eid", 3, "x");
        b.removeNode("a"); // kills e0; the unique check then fails AFTER step 1 compacted
        const mutations = b.mutationCount;
        expect(code(() => b.freeze())).toBe("E_DUPLICATE_EDGE_ID");
        // untouched: the builder still holds the tombstone and its own index space
        expect(b.nodeBound).toBe(5);
        expect(b.edgeBound).toBe(4);
        expect(b.nodeCount).toBe(4);
        expect(b.edgeCount).toBe(3);
        expect(b.hasNode("a")).toBe(false);
        expect(b.indexOf("e")).toBe(4);
        expect(b.dirty).toBe(true);
        expect(b.mutationCount).toBe(mutations);
        // fix the duplicate; the report is relative to f1, not to the failed attempt
        b.setEdgeValue("eid", 3, "y");
        const f2 = b.freezeWithReport();
        assertInvariants(f2.snapshot);
        expect(list(f2.report.nodeRemap)).toEqual([INVALID_INDEX, 0, 1, 2, 3]);
        expect(list(f2.report.edgeRemap)).toEqual([INVALID_INDEX, 0, 1, 2]);
        expect(f2.snapshot.ids.toArray()).toEqual(["b", "c", "d", "e"]);
        expect([0, 1, 2].map((e) => f2.snapshot.edges.value("eid", e))).toEqual([undefined, "x", "y"]);
        checkRemapAgainst(f1.snapshot, f2.snapshot, f2.report.nodeRemap, f2.report.edgeRemap);
    });

    it("a merging freeze composes tombstones and survivors into one remap; the next freeze is null again", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1); // e0
        b.addEdge("x", "y", 9); // e1, removed below
        b.addEdge("a", "b", 2); // e2 parallel of e0
        b.addEdge("b", "c", 3); // e3
        b.addEdge("a", "b", 4); // e4 parallel of e0
        b.freeze();
        b.removeEdge(1);
        const f2 = b.freezeWithReport({ duplicateEdges: "sum" });
        assertInvariants(f2.snapshot);
        expect(list(f2.report.edgeRemap)).toEqual([0, INVALID_INDEX, 0, 1, 0]);
        expect(f2.report.nodeRemap).toBeNull();
        expect(f2.report.mergedEdges).toBe(2);
        expect(f2.report.droppedEdges).toBe(3);
        expect(f2.report.compacted).toBe(true);
        expect(f2.snapshot.edgeCount).toBe(2);
        expect(Array.from(f2.snapshot.weights as Float32Array)).toEqual([7, 3]);
        // the builder was rewritten to the merged set and a re-freeze is a no-op
        expect(b.edgeCount).toBe(2);
        expect(b.edgeWeight(0)).toBe(7);
        expect(b.dirty).toBe(false);
        const f3 = b.freezeWithReport({ duplicateEdges: "sum" });
        expect(f3.report.edgeRemap).toBeNull();
        expect(f3.report.mergedEdges).toBe(0);
        expect(f3.report.droppedEdges).toBe(0);
        expect(f3.report.compacted).toBe(false);
    });

    it("counts: droppedEdges is tombstoned + dropped loops + merged, with a tombstoned loop counted once", () => {
        const b = new GraphBuilder({ directed: true, selfLoops: "drop" });
        b.addEdge("a", "a"); // e0 loop, tombstoned below (must not also count as a dropped loop)
        b.addEdge("b", "b"); // e1 loop, dropped by policy
        b.addEdge("a", "b"); // e2
        b.addEdge("a", "b"); // e3 merged into e2
        b.addEdge("c", "d"); // e4 tombstoned
        b.removeEdge(0);
        b.removeEdge(4);
        const { report, snapshot } = b.freezeWithReport({ duplicateEdges: "first" });
        assertInvariants(snapshot);
        expect(report.droppedSelfLoops).toBe(1);
        expect(report.mergedEdges).toBe(1);
        expect(report.droppedEdges).toBe(4);
        expect(list(report.edgeRemap)).toEqual([INVALID_INDEX, INVALID_INDEX, 0, 0, INVALID_INDEX]);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.selfLoopCount).toBe(0);
    });

    it("profile timings name every phase that ran and nothing else", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.removeEdge(1);
        const merged = b.freezeWithReport({ profile: true, duplicateEdges: "sum" });
        expect(Object.keys(merged.report.timings)).toEqual([
            "compact",
            "sort",
            "duplicates",
            "weights",
            "ids",
            "columns",
            "snapshot",
            "total",
        ]);
        for (const value of Object.values(merged.report.timings)) {
            expect(value).toBeGreaterThanOrEqual(0);
        }
        expect(Object.isFrozen(merged.report.timings)).toBe(true);
        const keep = b.freezeWithReport({ profile: true });
        expect(Object.keys(keep.report.timings)).toEqual([
            "compact",
            "sort",
            "weights",
            "ids",
            "columns",
            "snapshot",
            "total",
        ]);
        const silent = b.freezeWithReport();
        expect(Object.keys(silent.report.timings)).toEqual([]);
    });
});

describe("audit: index-space path (addAnonymousNodes + addEdges)", () => {
    it("refuses dead and out-of-range indices with E_UNKNOWN_NODE (details.index) before pushing anything", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        b.removeNodeByIndex(1);
        const mutations = b.mutationCount;
        const cases: [number[], number[], number][] = [
            [[0, 3], [2, 2], 3],
            [[0, INVALID_INDEX], [2, 0], INVALID_INDEX],
            [[0, 1], [2, 2], 1],
            [[2], [1], 1],
        ];
        for (const [src, dst, bad] of cases) {
            const err = details(() => b.addEdges(new Uint32Array(src), new Uint32Array(dst)));
            expect(code(() => b.addEdges(new Uint32Array(src), new Uint32Array(dst)))).toBe("E_UNKNOWN_NODE");
            expect(err.index).toBe(bad);
        }
        expect(b.edgeBound).toBe(0);
        expect(b.edgeCount).toBe(0);
        expect(b.mutationCount).toBe(mutations);
        // a weight problem is also found before anything is pushed
        expect(
            code(() => b.addEdges(new Uint32Array([0, 2]), new Uint32Array([2, 0]), new Float32Array([1, Number.NaN]))),
        ).toBe("E_INVALID_WEIGHT");
        expect(b.edgeBound).toBe(0);
        // the valid call pushes in order and returns the first index
        expect(b.addEdges(new Uint32Array([0, 2]), new Uint32Array([2, 0]))).toBe(0);
        expect(b.edgeEndpoints(1)).toEqual([2, 0]);
        const s = b.freeze();
        assertInvariants(s);
        expect(s.nodeCount).toBe(2);
        expect(s.ids.kind).toBe("identity");
    });

    it("addEdgeByIndex and addEdges agree on the same code for a revived node's stale edges", () => {
        const b = new GraphBuilder({ directed: false });
        b.addAnonymousNodes(2);
        b.addEdges(new Uint32Array([0]), new Uint32Array([1]));
        b.removeNodeByIndex(1);
        expect(code(() => b.addEdgeByIndex(0, 1))).toBe("E_UNKNOWN_NODE");
        expect(code(() => b.addEdges(new Uint32Array([0]), new Uint32Array([1])))).toBe("E_UNKNOWN_NODE");
        // anonymous nodes have no id to revive through, so only a fresh node can be added
        expect(b.addAnonymousNodes(1)).toBe(2);
        b.addEdges(new Uint32Array([0]), new Uint32Array([2]));
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(list(report.nodeRemap)).toEqual([0, INVALID_INDEX, 1]);
        expect(list(report.edgeRemap)).toEqual([INVALID_INDEX, 0]);
        expect(snapshot.ids.toArray()).toEqual([0, 1]);
    });
});
