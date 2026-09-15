/**
 * Unit tests of GraphBuilder's API (design sections 6.1, 6.6, 8.3 and 11.3): ids and revival,
 * incidence lists, tombstones, direction changes and in-place expansion, columns and inference,
 * records, composition, lifecycle, and every error code the builder names.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { equalsTopology } from "../../src/snapshot/graph-snapshot.js";
import { type ColumnHandle } from "../../src/types/index.js";
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

describe("GraphBuilder options", () => {
    it("resolves defaults and exposes the current direction", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.options).toEqual({
            directed: true,
            weighted: "auto",
            weightDtype: "f32",
            duplicateEdges: "keep",
            selfLoops: "keep",
            addMissingNodes: true,
            expectedNodes: null,
            expectedEdges: null,
        });
        expect(Object.isFrozen(b.options)).toBe(true);
        b.setDirected(false);
        expect(b.options.directed).toBe(false);
        expect(b.directed).toBe(false);
    });

    it("keeps explicit option values and hints", () => {
        const b = new GraphBuilder({
            directed: false,
            weighted: true,
            weightDtype: "f64",
            duplicateEdges: "sum",
            selfLoops: "drop",
            addMissingNodes: false,
            expectedNodes: 10,
            expectedEdges: 20,
        });
        expect(b.options.weighted).toBe(true);
        expect(b.options.weightDtype).toBe("f64");
        expect(b.options.duplicateEdges).toBe("sum");
        expect(b.options.selfLoops).toBe("drop");
        expect(b.options.addMissingNodes).toBe(false);
        expect(b.options.expectedNodes).toBe(10);
        expect(b.options.expectedEdges).toBe(20);
    });

    it("rejects option values outside their documented sets with E_UNSUPPORTED", () => {
        expect(code(() => new GraphBuilder({ directed: "yes" as unknown as boolean }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, duplicateEdges: "avg" as never }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, selfLoops: "ignore" as never }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, weightDtype: "f16" as never }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, weighted: "maybe" as never }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, expectedNodes: -1 }))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder({ directed: true, addMissingNodes: 1 as never }))).toBe("E_UNSUPPORTED");
    });
});

describe("GraphBuilder nodes", () => {
    it("assigns indices in first-seen order and is idempotent", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.addNode("a")).toBe(0);
        expect(b.addNode("b")).toBe(1);
        expect(b.addNode("a")).toBe(0);
        expect(b.nodeCount).toBe(2);
        expect(b.nodeBound).toBe(2);
        expect(b.hasNode("a")).toBe(true);
        expect(b.hasNode("z")).toBe(false);
        expect(b.indexOf("b")).toBe(1);
        expect(b.indexOf("z")).toBe(INVALID_INDEX);
        expect(b.idOf(1)).toBe("b");
    });

    it('keeps SameValueZero semantics: 1 and "1" differ, -0 is 0', () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.addNode(1)).toBe(0);
        expect(b.addNode("1")).toBe(1);
        expect(b.addNode(-0)).toBe(2);
        expect(b.addNode(0)).toBe(2);
        expect(Object.is(b.idOf(2), 0)).toBe(true);
        expect(b.indexOf(-0)).toBe(2);
        expect(b.freeze().ids.toArray()).toEqual([1, "1", 0]);
    });

    it("rejects illegal ids with E_INVALID_ID and leaves lookups total", () => {
        const b = new GraphBuilder({ directed: true });
        expect(code(() => b.addNode(Number.NaN))).toBe("E_INVALID_ID");
        expect(code(() => b.addNode(Infinity))).toBe("E_INVALID_ID");
        expect(code(() => b.addNode(null as unknown as string))).toBe("E_INVALID_ID");
        expect(code(() => b.addNode(undefined as unknown as string))).toBe("E_INVALID_ID");
        expect(code(() => b.addNode({} as unknown as string))).toBe("E_INVALID_ID");
        expect(code(() => b.addNode(10n as unknown as string))).toBe("E_INVALID_ID");
        expect(details(() => b.addNode("\ud800")).reason).toBe("lone surrogate");
        expect(b.hasNode(Number.NaN)).toBe(false);
        expect(b.indexOf(null as unknown as string)).toBe(INVALID_INDEX);
        expect(b.nodeBound).toBe(0);
    });

    it("addNodes writes every index into out and allocates it when omitted", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("x");
        const out = b.addNodes(["a", "x", "b"]);
        expect(Array.from(out)).toEqual([1, 0, 2]);
        const given = new Uint32Array(4);
        expect(b.addNodes(new Set(["b", "c"]), given)).toBe(given);
        expect(Array.from(given)).toEqual([2, 3, 0, 0]);
        expect(code(() => b.addNodes(["d", "e"], new Uint32Array(1)))).toBe("E_COLUMN_LENGTH");
        expect(b.nodeBound).toBe(4);
    });

    it("addAnonymousNodes never builds an id map and freezes to an identity map", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.addAnonymousNodes(3)).toBe(0);
        expect(b.addAnonymousNodes(0)).toBe(3);
        expect(b.addAnonymousNodes(2)).toBe(3);
        expect(b.nodeCount).toBe(5);
        expect(b.hasNode(4)).toBe(true);
        expect(b.hasNode(5)).toBe(false);
        expect(b.hasNode("4")).toBe(false);
        expect(b.idOf(4)).toBe(4);
        const s = b.freeze();
        expect(s.ids.kind).toBe("identity");
        expect(s.ids.byteLength()).toBe(0);
        assertInvariants(s);
    });

    it("addAnonymousNodes after named nodes uses index ids and refuses a taken id", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        expect(b.addAnonymousNodes(2)).toBe(1);
        expect(b.idOf(1)).toBe(1);
        expect(b.indexOf(2)).toBe(2);
        b.addNode(4);
        expect(details(() => b.addAnonymousNodes(1)).id).toBe(4);
        expect(b.nodeBound).toBe(4);
        expect(b.freeze().ids.kind).toBe("mixed");
    });

    it("addAnonymousNodes validates its count (E_INDEX_RANGE, E_TOO_LARGE)", () => {
        const b = new GraphBuilder({ directed: true });
        expect(code(() => b.addAnonymousNodes(-1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.addAnonymousNodes(1.5))).toBe("E_INDEX_RANGE");
        expect(code(() => b.addAnonymousNodes(MAX_COUNT + 1))).toBe("E_TOO_LARGE");
        expect(b.nodeBound).toBe(0);
    });

    it("idOf throws E_INDEX_RANGE for out-of-range and removed indices", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        expect(code(() => b.idOf(1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.idOf(-1))).toBe("E_INDEX_RANGE");
        b.removeNode("a");
        expect(code(() => b.idOf(0))).toBe("E_INDEX_RANGE");
    });

    it("reserve grows capacity and validates its arguments", () => {
        const b = new GraphBuilder({ directed: true });
        const before = b.byteLength();
        b.reserve(1000, 5000);
        expect(b.byteLength()).toBeGreaterThan(before);
        expect(code(() => b.reserve(-1))).toBe("E_TOO_LARGE");
        expect(code(() => b.reserve(1, MAX_COUNT + 1))).toBe("E_TOO_LARGE");
        b.reserve();
    });
});

describe("GraphBuilder removal and revival", () => {
    it("removeNode tombstones the node and every live incident edge, returning them once each", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "a");
        b.addEdge("b", "b");
        b.addEdge("c", "b");
        b.addEdge("a", "c");
        b.removeEdge(3);
        const removed = b.removeNode("b");
        expect(Array.from(removed)).toEqual([0, 1, 2]);
        expect(b.nodeCount).toBe(2);
        expect(b.edgeCount).toBe(1);
        expect(b.hasNode("b")).toBe(false);
        expect(b.indexOf("b")).toBe(INVALID_INDEX);
        expect(b.hasEdge(0)).toBe(false);
        expect(b.hasEdge(4)).toBe(true);
        expect(b.nodeBound).toBe(3);
        expect(b.edgeBound).toBe(5);
        expect(code(() => b.removeNode("b"))).toBe("E_UNKNOWN_NODE");
        expect(code(() => b.removeNode("nope"))).toBe("E_UNKNOWN_NODE");
        expect(details(() => b.removeNodeByIndex(1)).index).toBe(1);
        expect(code(() => b.removeNodeByIndex(9))).toBe("E_UNKNOWN_NODE");
    });

    it("removeNodeByIndex works for anonymous nodes", () => {
        const b = new GraphBuilder({ directed: false });
        b.addAnonymousNodes(3);
        b.addEdges(new Uint32Array([0, 1, 2]), new Uint32Array([1, 2, 0]));
        expect(Array.from(b.removeNodeByIndex(1))).toEqual([0, 1]);
        expect(b.edgeCount).toBe(1);
        const s = b.freeze();
        expect(s.nodeCount).toBe(2);
        expect(s.ids.toArray()).toEqual([0, 1]);
        assertInvariants(s);
    });

    it("revives a removed id at its old index before the next freeze without reviving its edges (I16)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.removeNode("b");
        expect(b.addNode("b")).toBe(1);
        expect(b.nodeCount).toBe(3);
        expect(b.edgeCount).toBe(0);
        b.addEdge("b", "a");
        const { snapshot, report } = b.freezeWithReport();
        expect(report.nodeRemap).toBeNull();
        expect(report.edgeRemap).not.toBeNull();
        expect(Array.from(report.edgeRemap as Uint32Array)).toEqual([INVALID_INDEX, INVALID_INDEX, 0]);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(snapshot.edgeCount).toBe(1);
        expect(snapshot.edgeSource(0)).toBe(1);
        assertInvariants(snapshot);
    });

    it("assigns a new index to a re-added id after a compacting freeze", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        b.addNode("c");
        b.removeNode("b");
        b.freeze();
        expect(b.nodeBound).toBe(2);
        expect(b.indexOf("c")).toBe(1);
        expect(b.addNode("b")).toBe(2);
    });

    it("removeEdge is O(1), total and bumps mutationCount only when something was removed", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge(1, 2);
        const m = b.mutationCount;
        expect(b.removeEdge(0)).toBe(true);
        expect(b.mutationCount).toBe(m + 1);
        expect(b.removeEdge(0)).toBe(false);
        expect(b.removeEdge(7)).toBe(false);
        expect(b.removeEdge(-1)).toBe(false);
        expect(b.mutationCount).toBe(m + 1);
        expect(b.edgeCount).toBe(0);
        expect(b.edgeBound).toBe(1);
    });
});

describe("GraphBuilder edges", () => {
    it("addEdge creates missing endpoints by default and returns edge indices in order", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.addEdge("a", "b")).toBe(0);
        expect(b.addEdge("b", "c", 2)).toBe(1);
        expect(b.nodeCount).toBe(3);
        expect(b.edgeEndpoints(1)).toEqual([1, 2]);
        expect(b.edgeWeight(0)).toBe(1);
        expect(b.edgeWeight(1)).toBe(2);
    });

    it("addEdge with addMissingNodes false throws E_UNKNOWN_NODE without touching the builder", () => {
        const b = new GraphBuilder({ directed: true, addMissingNodes: false });
        b.addNode("a");
        const m = b.mutationCount;
        expect(details(() => b.addEdge("a", "b")).id).toBe("b");
        expect(code(() => b.addEdge("z", "a"))).toBe("E_UNKNOWN_NODE");
        b.addNode("b");
        b.removeNode("b");
        expect(code(() => b.addEdge("a", "b"))).toBe("E_UNKNOWN_NODE");
        expect(b.edgeBound).toBe(0);
        expect(b.mutationCount).toBe(m + 2);
    });

    it("rejects a NaN weight with E_INVALID_WEIGHT and accepts every other number", () => {
        const b = new GraphBuilder({ directed: true });
        expect(code(() => b.addEdge("a", "b", Number.NaN))).toBe("E_INVALID_WEIGHT");
        expect(code(() => b.addEdge("a", "b", "3" as unknown as number))).toBe("E_INVALID_WEIGHT");
        expect(b.edgeBound).toBe(0);
        b.addEdge("a", "b", 0);
        b.addEdge("a", "b", -2);
        b.addEdge("a", "b", Infinity);
        const s = b.freeze();
        expect(s.flags.finiteWeights).toBe(false);
        expect(s.flags.nonNegativeWeights).toBe(false);
        expect(s.flags.allWeightsOne).toBe(false);
    });

    it("a builder declared unweighted never allocates weights and refuses a weight other than 1", () => {
        const b = new GraphBuilder({ directed: true, weighted: false });
        b.addEdge("a", "b", 1);
        expect(code(() => b.addEdge("a", "b", 2))).toBe("E_INVALID_WEIGHT");
        expect(code(() => b.setEdgeWeight(0, 0.5))).toBe("E_INVALID_WEIGHT");
        b.setEdgeWeight(0, 1);
        expect(code(() => b.addEdges(new Uint32Array([0]), new Uint32Array([1]), new Float32Array([2])))).toBe(
            "E_INVALID_WEIGHT",
        );
        b.addEdges(new Uint32Array([0]), new Uint32Array([1]), new Float32Array([1]));
        const s = b.freeze();
        expect(s.weights).toBeNull();
        expect(s.flags.weighted).toBe(false);
        expect(s.edges.byRole("weight")).toBeNull();
    });

    it("weighted: true allocates the array even when every value is 1", () => {
        const b = new GraphBuilder({ directed: true, weighted: true });
        b.addEdge("a", "b");
        const s = b.freeze();
        expect(s.weights).not.toBeNull();
        expect(Array.from(s.weights as Float32Array)).toEqual([1]);
        expect(s.flags.allWeightsOne).toBe(true);
        expect(new GraphBuilder({ directed: true, weighted: true }).freeze().weights).toEqual(new Float32Array(0));
    });

    it("addEdgeByIndex validates both endpoints (E_UNKNOWN_NODE with details.index)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(2);
        expect(b.addEdgeByIndex(0, 1, 3)).toBe(0);
        expect(details(() => b.addEdgeByIndex(0, 2)).index).toBe(2);
        expect(code(() => b.addEdgeByIndex(-1, 0))).toBe("E_UNKNOWN_NODE");
        b.removeNodeByIndex(1);
        expect(details(() => b.addEdgeByIndex(1, 0)).index).toBe(1);
        expect(code(() => b.addEdgeByIndex(0, 0, Number.NaN))).toBe("E_INVALID_WEIGHT");
        expect(b.edgeBound).toBe(1);
    });

    it("addEdges validates everything before pushing anything", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        expect(code(() => b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1])))).toBe("E_COLUMN_LENGTH");
        expect(code(() => b.addEdges(new Uint32Array([0]), new Uint32Array([1]), new Float32Array(2)))).toBe(
            "E_COLUMN_LENGTH",
        );
        expect(details(() => b.addEdges(new Uint32Array([0, 5]), new Uint32Array([1, 1]))).index).toBe(5);
        expect(
            details(() =>
                b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1, 2]), new Float64Array([1, Number.NaN])),
            ).edge,
        ).toBe(1);
        expect(b.edgeBound).toBe(0);
        expect(b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1, 2]), new Float64Array([0.5, 2]))).toBe(0);
        expect(b.addEdges(new Uint32Array(0), new Uint32Array(0))).toBe(2);
        expect(b.addEdges(new Uint32Array([2]), new Uint32Array([0]))).toBe(2);
        expect(b.edgeCount).toBe(3);
        expect(b.edgeWeight(0)).toBe(0.5);
        expect(b.edgeWeight(2)).toBe(1);
        const s = b.freeze();
        assertInvariants(s);
        expect(Array.from(s.weights as Float32Array)).toEqual([0.5, 2, 1]);
    });

    it("addEdgesByIds validates ids and weights before adding any node or edge", () => {
        const b = new GraphBuilder({ directed: false });
        expect(code(() => b.addEdgesByIds(["a", "b"], ["b"]))).toBe("E_COLUMN_LENGTH");
        expect(code(() => b.addEdgesByIds(["a"], ["b"], [1, 2]))).toBe("E_COLUMN_LENGTH");
        expect(code(() => b.addEdgesByIds(["a", "b"], ["b", Number.NaN]))).toBe("E_INVALID_ID");
        expect(code(() => b.addEdgesByIds(["a", "b"], ["b", "c"], [1, Number.NaN]))).toBe("E_INVALID_WEIGHT");
        expect(b.nodeBound).toBe(0);
        expect(b.addEdgesByIds(["a", "b"], ["b", "c"], [2, 3])).toBe(0);
        expect(b.addEdgesByIds(["c"], ["a"])).toBe(2);
        expect(b.nodeCount).toBe(3);
        expect(b.edgeWeight(1)).toBe(3);
        const strict = new GraphBuilder({ directed: true, addMissingNodes: false });
        strict.addNode("a");
        expect(code(() => strict.addEdgesByIds(["a", "a"], ["a", "b"]))).toBe("E_UNKNOWN_NODE");
        expect(strict.edgeBound).toBe(0);
        expect(strict.addEdgesByIds(["a"], ["a"])).toBe(0);
    });

    it("edgeEndpoints / edgeWeight / setEdgeWeight check the edge (E_INDEX_RANGE)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        expect(code(() => b.edgeEndpoints(1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.edgeWeight(-1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.setEdgeWeight(1, 2))).toBe("E_INDEX_RANGE");
        expect(code(() => b.setEdgeWeight(0, Number.NaN))).toBe("E_INVALID_WEIGHT");
        b.setEdgeWeight(0, 4);
        expect(b.edgeWeight(0)).toBe(4);
        b.removeEdge(0);
        expect(code(() => b.edgeEndpoints(0))).toBe("E_INDEX_RANGE");
        expect(code(() => b.edgeWeight(0))).toBe("E_INDEX_RANGE");
    });

    it("setEdgeWeight allocates the weight array and back-fills earlier edges with 1", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.setEdgeWeight(1, 7);
        const s = b.freeze();
        expect(Array.from(s.weights as Float32Array)).toEqual([1, 7]);
        const shadow = s.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.isSet(0)).toBe(false);
        expect(shadow?.isSet(1)).toBe(true);
    });

    it("walks the incidence lists: outEdgesOf, inEdgesOf, findEdges", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "c");
        b.addEdge("c", "a");
        b.addEdge("a", "a");
        b.addEdge("a", "b");
        b.removeEdge(1);
        expect(Array.from(b.outEdgesOf(0))).toEqual([0, 3, 4]);
        expect(Array.from(b.inEdgesOf(0))).toEqual([2, 3]);
        expect(Array.from(b.inEdgesOf(1))).toEqual([0, 4]);
        expect(Array.from(b.findEdges(0, 1))).toEqual([0, 4]);
        expect(Array.from(b.findEdges(1, 0))).toEqual([]);
        expect(Array.from(b.findEdges(0, 0))).toEqual([3]);
        expect(Array.from(b.findEdges(2, 0))).toEqual([2]);
        expect(code(() => b.outEdgesOf(3))).toBe("E_INDEX_RANGE");
        expect(code(() => b.inEdgesOf(-1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.findEdges(0, 9))).toBe("E_INDEX_RANGE");
        b.removeNode("a");
        expect(Array.from(b.outEdgesOf(0))).toEqual([]);
        expect(Array.from(b.inEdgesOf(1))).toEqual([]);
    });

    it("findEdges on an undirected builder matches either orientation", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        b.addEdge("b", "a");
        b.addEdge("b", "b");
        expect(Array.from(b.findEdges(0, 1))).toEqual([0, 1]);
        expect(Array.from(b.findEdges(1, 0))).toEqual([0, 1]);
        expect(Array.from(b.findEdges(1, 1))).toEqual([2]);
        // undirected: every incident edge leaves the node (I7), a self-loop once, both queries agree
        expect(Array.from(b.outEdgesOf(1))).toEqual([0, 1, 2]);
        expect(Array.from(b.inEdgesOf(1))).toEqual([0, 1, 2]);
        expect(Array.from(b.outEdgesOf(0))).toEqual([0, 1]);
        expect(Array.from(b.inEdgesOf(0))).toEqual([0, 1]);
    });

    it("counts mutations for topology and weight changes only", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.mutationCount).toBe(0);
        expect(b.dirty).toBe(true);
        b.addNode("a");
        expect(b.mutationCount).toBe(1);
        b.addNode("a");
        expect(b.mutationCount).toBe(1);
        b.addEdge("a", "b");
        expect(b.mutationCount).toBe(3);
        b.setNodeValue("x", 0, 1);
        b.declareEdgeColumn({ name: "y", dtype: "f32" });
        b.setEdgeValue("y", 0, 2);
        b.setGraphValue("g", 1);
        b.setMeta({ name: "n" });
        expect(b.mutationCount).toBe(3);
        b.setEdgeWeight(0, 2);
        expect(b.mutationCount).toBe(4);
        b.freeze();
        expect(b.mutationCount).toBe(4);
        expect(b.dirty).toBe(false);
        b.setNodeValue("x", 0, 2);
        expect(b.dirty).toBe(false);
        b.removeEdge(0);
        expect(b.dirty).toBe(true);
        expect(b.mutationCount).toBe(5);
    });
});

describe("GraphBuilder direction", () => {
    it("setDirected is a no-op for the same value and free while no live edge exists", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addEdge("a", "b");
        const m = b.mutationCount;
        b.setDirected(true);
        expect(b.mutationCount).toBe(m);
        b.removeEdge(0);
        b.setDirected(false);
        expect(b.directed).toBe(false);
        expect(b.mutationCount).toBe(m + 2);
        b.lockDirected();
        b.setDirected(false);
        expect(b.directedLocked).toBe(true);
    });

    it("refuses changes with E_DIRECTED: locked, directed -> undirected with edges, or no expand", () => {
        const locked = new GraphBuilder({ directed: true });
        locked.lockDirected();
        expect(details(() => locked.setDirected(false)).reason).toBe("locked");
        const withEdges = new GraphBuilder({ directed: true });
        withEdges.addEdge("a", "b");
        expect(details(() => withEdges.setDirected(false)).reason).toBe("edges present");
        const undirected = new GraphBuilder({ directed: false });
        undirected.addEdge("a", "b");
        expect(details(() => undirected.setDirected(true)).reason).toBe("expand required");
        expect(code(() => undirected.setDirected(true, { expand: false }))).toBe("E_DIRECTED");
        expect(undirected.directed).toBe(false);
        undirected.lockDirected();
        expect(details(() => undirected.setDirected(true, { expand: true })).reason).toBe("locked");
    });

    it("expands an undirected builder in place with graphty.directed / graphty.pair columns", () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("a", "b", 2.5);
        b.addEdge("b", "c");
        b.addEdge("c", "c", 4);
        b.addEdge("c", "a");
        b.removeEdge(3);
        b.setEdgeValue("kind", 0, "road");
        const m = b.mutationCount;
        b.setDirected(true, { expand: true });
        expect(b.directed).toBe(true);
        expect(b.mutationCount).toBe(m + 1);
        expect(b.edgeCount).toBe(5);
        expect(b.edgeBound).toBe(6);
        expect(b.edgeEndpoints(4)).toEqual([1, 0]);
        expect(b.edgeEndpoints(5)).toEqual([2, 1]);
        expect(b.edgeWeight(4)).toBe(2.5);
        expect(b.edgeWeight(5)).toBe(1);
        const { snapshot, report } = b.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.edgeCount).toBe(5);
        expect(Array.from(report.edgeRemap as Uint32Array)).toEqual([0, 1, 2, INVALID_INDEX, 3, 4]);
        const directed = snapshot.edges.requireTyped("graphty.directed", "bool");
        const pair = snapshot.edges.requireTyped("graphty.pair", "u32");
        expect(directed.meta.role).toBe("directed");
        expect(pair.meta.role).toBe("pair");
        expect(pair.meta.refersTo).toBe("edge");
        for (let e = 0; e < 5; e++) {
            expect(directed.value(e)).toBe(false);
        }
        expect(Array.from(pair.data)).toEqual([3, 4, INVALID_INDEX, 0, 1]);
        expect(pair.isSet(2)).toBe(false);
        const kind = snapshot.edges.require("kind");
        expect(kind.value(0)).toBe("road");
        expect(kind.isSet(3)).toBe(false);
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow?.isSet(3)).toBe(true);
        expect(shadow?.isSet(4)).toBe(false);
        expect(shadow?.value(3)).toBe(2.5);
    });

    it("expansion can be repeated after a fresh setDirected(false) on an emptied builder", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        b.setDirected(true, { expand: true });
        b.setDirected(true, { expand: true });
        expect(b.edgeCount).toBe(2);
        b.clear();
        b.setDirected(false);
        b.addEdge("x", "y");
        b.setDirected(true, { expand: true });
        expect(b.edgeCount).toBe(2);
        assertInvariants(b.freeze());
    });
});

describe("GraphBuilder columns", () => {
    it("declares columns with stable handles and returns the same handle for the same shape", () => {
        const b = new GraphBuilder({ directed: true });
        const h = b.declareNodeColumn({ name: "score", dtype: "f64", role: "rank" });
        expect(h).toBe(0);
        expect(b.declareNodeColumn({ name: "score", dtype: "f64" })).toBe(h);
        expect(b.nodeColumn("score")).toBe(h);
        expect(b.nodeColumn("missing")).toBe(INVALID_INDEX);
        expect(b.edgeColumn("score")).toBe(INVALID_INDEX);
        expect(code(() => b.declareNodeColumn({ name: "score", dtype: "f32" }))).toBe("E_COLUMN_EXISTS");
        expect(code(() => b.declareNodeColumn({ name: "score", dtype: "f64", components: 2 }))).toBe("E_COLUMN_EXISTS");
        expect(code(() => b.declareNodeColumn({ name: "other", dtype: "u32", role: "rank" }))).toBe("E_DUPLICATE_ROLE");
        expect(code(() => b.declareNodeColumn({ name: "", dtype: "u32" }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.declareEdgeColumn({ name: "bad", dtype: "nope" as never }))).toBe("E_COLUMN_TYPE");
        expect(b.declareEdgeColumn({ name: "w", dtype: "f32" })).toBe(0);
    });

    it("writes cells by handle and by name and checks the row (E_INDEX_RANGE, E_UNKNOWN_COLUMN)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        const h = b.declareNodeColumn({ name: "score", dtype: "f64" });
        b.setNodeValue(h, 0, 1.5);
        b.setNodeValue("score", 1, 2.5);
        expect(code(() => b.setNodeValue(h, 2, 1))).toBe("E_INDEX_RANGE");
        expect(code(() => b.setNodeValue(7 as ColumnHandle, 0, 1))).toBe("E_UNKNOWN_COLUMN");
        expect(code(() => b.setEdgeValue(3 as ColumnHandle, 0, 1))).toBe("E_UNKNOWN_COLUMN");
        expect(code(() => b.setEdgeValue("x", 1, 1))).toBe("E_INDEX_RANGE");
        b.setEdgeValue("x", 0, "v");
        const s = b.freeze();
        expect(s.nodes.value("score", 0)).toBe(1.5);
        expect(s.nodes.value("score", 1)).toBe(2.5);
        expect(s.edges.value("x", 0)).toBe("v");
    });

    it("infers and widens auto-declared columns and reports the widening once", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c", "d"]);
        b.setNodeValue("v", 0, true);
        b.setNodeValue("v", 1, 3);
        b.setNodeValue("v", 2, 2.5);
        b.setNodeValue("v", 3, "01");
        b.setNodeValue("u", 0, 1);
        b.setNodeValue("u", 1, { nested: true });
        b.setNodeValue("skip", 0, null);
        const { snapshot, report } = b.freezeWithReport();
        expect(snapshot.nodes.names()).toEqual(["v", "u"]);
        const v = snapshot.nodes.require("v");
        expect(v.dtype).toBe("string");
        expect([0, 1, 2, 3].map((r) => v.value(r))).toEqual(["true", "3", "2.5", "01"]);
        expect(snapshot.nodes.require("u").dtype).toBe("json");
        expect(report.widened).toEqual([
            { column: "v", domain: "node", from: "bool", to: "i32" },
            { column: "v", domain: "node", from: "i32", to: "f64" },
            { column: "v", domain: "node", from: "f64", to: "string" },
            { column: "u", domain: "node", from: "i32", to: "json" },
        ]);
        expect(b.freezeWithReport().report.widened).toEqual([]);
    });

    it("keeps an inferred column's dtype sticky and coerces values once at freeze", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b"]);
        b.setNodeValue("v", 0, 1);
        b.setNodeValue("v", 1, 2.5);
        b.setNodeValue("v", 1, 3);
        const s = b.freeze();
        expect(s.nodes.require("v").dtype).toBe("f64");
        expect(s.nodes.value("v", 1)).toBe(3);
        expect(code(() => b.setNodeValue("v", 0, 10n))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("w", 0, "\ud800"))).toBe("E_COLUMN_TYPE");
    });

    it("declared columns reject values they cannot hold (E_COLUMN_TYPE) and accept every dtype", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        b.declareNodeColumn({ name: "f", dtype: "f32" });
        b.declareNodeColumn({ name: "i", dtype: "i32" });
        b.declareNodeColumn({ name: "u", dtype: "u8" });
        b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, role: "position", mutable: true });
        b.declareNodeColumn({ name: "flag", dtype: "bool" });
        b.declareNodeColumn({ name: "cat", dtype: "dict", options: ["x", "y"] });
        b.declareNodeColumn({ name: "name", dtype: "string" });
        b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        b.declareNodeColumn({ name: "spells", dtype: "list", itemDtype: "f64", itemComponents: 2 });
        b.declareNodeColumn({ name: "blob", dtype: "json" });
        b.declareNodeColumn({ name: "solid", dtype: "u32", nullable: false });
        expect(code(() => b.setNodeValue("f", 0, "no"))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("i", 0, 1.5))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("u", 0, 300))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("pos", 0, [1, 2]))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("pos", 0, [1, 2, "3"]))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("flag", 0, 1))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("cat", 0, {}))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("name", 0, {}))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("tags", 0, "x"))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("tags", 0, [1]))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("spells", 0, [[1]]))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("blob", 0, () => 1))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeValue("solid", 0, undefined))).toBe("E_COLUMN_TYPE");
        b.setNodeValue("f", 0, 1.5);
        b.setNodeValue("f", 1, true);
        b.setNodeValue("i", 0, -4);
        b.setNodeValue("u", 0, 255);
        b.setNodeValue("pos", 0, [1, 2, 3]);
        b.setNodeValue("pos", 1, 5);
        b.setNodeValue("flag", 0, true);
        b.setNodeValue("flag", 1, false);
        b.setNodeValue("cat", 0, "y");
        b.setNodeValue("cat", 1, "z");
        b.setNodeValue("name", 0, 12);
        b.setNodeValue("tags", 0, ["a", "b"]);
        b.setNodeValue("tags", 1, []);
        b.setNodeValue("spells", 0, [
            [1, 2],
            [3, 4],
        ]);
        b.setNodeValue("blob", 0, null);
        b.setNodeValue("blob", 1, { deep: [1, "x"] });
        b.setNodeValue("solid", 0, 9);
        b.setNodeValue("flag", 0, undefined);
        const s = b.freeze();
        assertInvariants(s);
        expect(s.nodes.value("f", 0)).toBe(1.5);
        expect(s.nodes.value("f", 1)).toBe(1);
        expect(s.nodes.isSet("f", 2)).toBe(false);
        expect(s.nodes.value("i", 0)).toBe(-4);
        expect(s.nodes.value("u", 0)).toBe(255);
        expect(Array.from(s.nodes.value("pos", 0) as ArrayLike<number>)).toEqual([1, 2, 3]);
        expect(Array.from(s.nodes.value("pos", 1) as ArrayLike<number>)).toEqual([5, 5, 5]);
        expect(s.nodes.isSet("flag", 0)).toBe(false);
        expect(s.nodes.value("flag", 1)).toBe(false);
        const cat = s.nodes.requireTyped("cat", "dict");
        expect(cat.dictionary).toEqual(["x", "y", "z"]);
        expect(cat.value(1)).toBe("z");
        expect(s.nodes.value("name", 0)).toBe("12");
        expect(s.nodes.value("tags", 0)).toEqual(["a", "b"]);
        expect(s.nodes.value("tags", 1)).toEqual([]);
        expect(s.nodes.isSet("tags", 2)).toBe(false);
        expect((s.nodes.value("spells", 0) as ArrayLike<number>[]).map((item) => Array.from(item))).toEqual([
            [1, 2],
            [3, 4],
        ]);
        expect(s.nodes.isSet("blob", 0)).toBe(true);
        expect(s.nodes.value("blob", 0)).toBeNull();
        expect(s.nodes.value("blob", 1)).toEqual({ deep: [1, "x"] });
        const solid = s.nodes.requireTyped("solid", "u32");
        expect(solid.validity).toBeNull();
        expect(Array.from(solid.data)).toEqual([9, 0, 0]);
        expect(s.nodes.require("pos").meta.mutable).toBe(true);
    });

    it("drops the validity bitmap of a declared column whose every row is set", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b"]);
        b.declareNodeColumn({ name: "x", dtype: "i32" });
        b.setNodeValue("x", 0, 1);
        b.setNodeValue("x", 1, 2);
        const s = b.freeze();
        expect(s.nodes.require("x").validity).toBeNull();
        expect(s.nodes.require("x").nullCount).toBe(0);
    });

    it("setNodeColumn / setEdgeColumn adopt typed arrays by copy with length checks", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        b.addEdge("a", "b");
        expect(code(() => b.setNodeColumn("x", new Float32Array(2)))).toBe("E_COLUMN_LENGTH");
        expect(code(() => b.setNodeColumn("x", new Float32Array(3), { dtype: "i32" }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setNodeColumn("x", new Float32Array(3), { dtype: "string" }))).toBe("E_COLUMN_TYPE");
        const data = new Float32Array([1, 2, 3]);
        b.setNodeColumn("x", data);
        data[0] = 99;
        b.setNodeColumn("pos", new Float64Array(6), { components: 2 });
        b.setNodeColumn("flags", new Uint32Array([0b101]), { dtype: "bool" });
        b.setNodeColumn("cat", new Uint32Array([1, 0, 1]), { dtype: "dict", options: ["p", "q"] });
        expect(code(() => b.setNodeColumn("bad", new Uint32Array([2, 0, 1]), { dtype: "dict", options: ["p"] }))).toBe(
            "E_COLUMN_TYPE",
        );
        b.setNodeColumn("bytes", new Uint8Array([1, 2, 3]), { nullable: true });
        b.setEdgeColumn("w", new Int32Array([7]));
        b.setNodeColumn("x", new Float32Array([4, 5, 6]));
        b.declareNodeColumn({ name: "r", dtype: "u32", role: "rank" });
        expect(code(() => b.setNodeColumn("x", new Float32Array(3), { role: "rank" }))).toBe("E_DUPLICATE_ROLE");
        const s = b.freeze();
        assertInvariants(s);
        expect(Array.from(s.nodes.requireTyped("x", "f32").data)).toEqual([4, 5, 6]);
        expect(s.nodes.require("x").validity).toBeNull();
        expect(s.nodes.require("pos").meta.components).toBe(2);
        expect([0, 1, 2].map((r) => s.nodes.value("flags", r))).toEqual([true, false, true]);
        expect([0, 1, 2].map((r) => s.nodes.value("cat", r))).toEqual(["q", "p", "q"]);
        expect(s.nodes.require("bytes").validity).toBeNull();
        expect(s.nodes.value("bytes", 2)).toBe(3);
        expect(s.edges.value("w", 0)).toBe(7);
    });

    it("stores graph-level values and metadata", () => {
        const b = new GraphBuilder({ directed: true });
        b.setGraphValue("title", "hello");
        b.setGraphValue("count", 3, { dtype: "i32" });
        b.setGraphValue("nothing", undefined);
        b.setGraphValue("title", "again");
        expect(code(() => b.setGraphValue("bad", () => 1))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setGraphValue("bad", 1, { dtype: "nope" as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setGraphValue("bad", "x", { dtype: "i32" }))).toBe("E_COLUMN_TYPE");
        b.setMeta({ name: "g", keywords: ["a"], idType: "string", extra: { k: [1, 2] } });
        b.setMeta({
            description: "d",
            weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
        });
        expect(code(() => b.setMeta({ name: 5 as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ keywords: [1] as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ idType: "float" as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ mode: "x" as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ declaredMultigraph: "no" as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ weightOrigin: 1 as never }))).toBe("E_COLUMN_TYPE");
        expect(code(() => b.setMeta({ weightOrigin: { format: 2 } as never }))).toBe("E_COLUMN_TYPE");
        expect(details(() => b.setMeta({ extra: { f: () => 1 } })).field).toBe("extra");
        const s = b.freeze();
        expect(s.graph.value("title", 0)).toBe("again");
        expect(s.graph.require("count").dtype).toBe("i32");
        expect(s.graph.isSet("nothing", 0)).toBe(false);
        expect(s.meta.name).toBe("g");
        expect(s.meta.description).toBe("d");
        expect(s.meta.keywords).toEqual(["a"]);
        expect(s.meta.idType).toBe("string");
        expect(s.meta.extra).toEqual({ k: [1, 2] });
        expect(s.meta.weightOrigin).toEqual({ format: "gexf", id: null, title: null, type: "double", namespace: null });
        expect(s.meta.timeFormat).toBeNull();
        b.setMeta({ weightOrigin: null, timeFormat: "date", timeRepresentation: "interval", mode: "dynamic" });
        expect(b.freeze().meta.weightOrigin).toBeNull();
    });

    it("graph values with a duplicated role fail at freeze with E_DUPLICATE_ROLE", () => {
        const b = new GraphBuilder({ directed: true });
        b.setGraphValue("a", 1, { role: "rank" });
        b.setGraphValue("b", 2, { role: "rank" });
        expect(code(() => b.freeze())).toBe("E_DUPLICATE_ROLE");
    });

    it("builds extension tables row by row and rewrites their references through compaction", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        const t = b.addExtensionTable("temporal:node:price", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "start", dtype: "f64" },
            { name: "value", dtype: "f64" },
        ]);
        expect(t).toBe(0);
        expect(code(() => b.addExtensionTable("temporal:node:price", []))).toBe("E_COLUMN_EXISTS");
        expect(
            code(() =>
                b.addExtensionTable("dup", [
                    { name: "x", dtype: "u32" },
                    { name: "x", dtype: "u32" },
                ]),
            ),
        ).toBe("E_COLUMN_EXISTS");
        expect(
            code(() =>
                b.addExtensionTable("dup", [
                    { name: "x", dtype: "u32", role: "start" },
                    { name: "y", dtype: "u32", role: "start" },
                ]),
            ),
        ).toBe("E_DUPLICATE_ROLE");
        expect(b.addExtensionRow(t, [0, 1, 10])).toBe(0);
        expect(b.addExtensionRow(t, [1, 2, 20])).toBe(1);
        expect(b.addExtensionRow(t, [2, 3, 30])).toBe(2);
        expect(code(() => b.addExtensionRow(t, [1, 2]))).toBe("E_COLUMN_LENGTH");
        expect(code(() => b.addExtensionRow(5 as never, [1, 2, 3]))).toBe("E_INDEX_RANGE");
        b.removeNode("b");
        const s = b.freeze();
        assertInvariants(s);
        const table = s.extensions.get("temporal:node:price");
        expect(table?.rowCount).toBe(3);
        const element = table?.requireTyped("element", "u32");
        expect(Array.from(element?.data ?? [])).toEqual([0, INVALID_INDEX, 1]);
        expect(element?.isSet(1)).toBe(false);
        expect(table?.value("value", 2)).toBe(30);
    });

    it("addNodeRecord is last-write-wins per attribute and addEdgeRecord reads the weight key", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.addNodeRecord("a", { color: "red", size: 1 })).toBe(0);
        expect(b.addNodeRecord("a", { size: 2, extra: true })).toBe(0);
        expect(b.addEdgeRecord("a", "b", { weight: 2.5, kind: "k" })).toBe(0);
        expect(b.addEdgeRecord("a", "b", { weight: null, kind: "j" })).toBe(1);
        expect(b.addEdgeRecord("a", "b", { w: 4, weight: 9 }, "w")).toBe(2);
        expect(b.addEdgeRecord("a", "b", { weight: 7 }, null)).toBe(3);
        expect(code(() => b.addEdgeRecord("a", "b", { weight: "heavy" }))).toBe("E_INVALID_WEIGHT");
        expect(code(() => b.addEdgeRecord("a", "b", { weight: Number.NaN }))).toBe("E_INVALID_WEIGHT");
        expect(b.edgeCount).toBe(4);
        const s = b.freeze();
        expect(s.nodes.value("color", 0)).toBe("red");
        expect(s.nodes.value("size", 0)).toBe(2);
        expect(s.nodes.value("extra", 0)).toBe(true);
        expect(s.nodes.isSet("color", 1)).toBe(false);
        expect(Array.from(s.edgeList().weights as Float32Array)).toEqual([2.5, 1, 4, 1]);
        expect(s.edges.value("kind", 1)).toBe("j");
        expect(s.edges.has("w")).toBe(false);
        expect(s.edges.value("weight", 2)).toBe(9);
        expect(s.edges.value("weight", 3)).toBe(7);
        expect(s.edges.isSet("weight", 0)).toBe(false);
    });

    it("enforces unique columns at freeze (E_DUPLICATE_EDGE_ID / E_DUPLICATE_ID) leaving the builder intact", () => {
        const b = new GraphBuilder({ directed: true });
        b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
        b.declareNodeColumn({ name: "key", dtype: "i32", unique: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.setEdgeValue("id", 0, "e1");
        b.setEdgeValue("id", 1, "e1");
        b.addNode("d");
        b.removeNode("d");
        const bound = b.nodeBound;
        const d = details(() => b.freeze());
        expect(code(() => b.freeze())).toBe("E_DUPLICATE_EDGE_ID");
        expect(d.rows).toEqual([0, 1]);
        expect(b.nodeBound).toBe(bound);
        expect(b.nodeCount).toBe(3);
        b.setEdgeValue("id", 1, "e2");
        b.setNodeValue("key", 0, 1);
        b.setNodeValue("key", 1, 1);
        expect(code(() => b.freeze())).toBe("E_DUPLICATE_ID");
        b.setNodeValue("key", 1, 2);
        const s = b.freeze();
        expect(s.edgeIndexOf("e2")).toBe(1);
        expect(s.nodeCount).toBe(3);
    });
});

describe("GraphBuilder lifecycle", () => {
    it("clear empties everything but keeps options and the lock", () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.lockDirected();
        b.addEdge("a", "b", 2);
        b.setNodeValue("x", 0, 1);
        b.setGraphValue("g", 1);
        b.setMeta({ name: "n" });
        b.addExtensionTable("t", [{ name: "v", dtype: "f64" }]);
        b.freeze();
        b.clear();
        expect(b.nodeCount).toBe(0);
        expect(b.nodeBound).toBe(0);
        expect(b.edgeBound).toBe(0);
        expect(b.nodeColumn("x")).toBe(INVALID_INDEX);
        expect(b.dirty).toBe(true);
        expect(b.directedLocked).toBe(true);
        expect(b.options.weightDtype).toBe("f64");
        const s = b.freeze();
        expect(s.nodeCount).toBe(0);
        expect(s.graph.names()).toEqual([]);
        expect(s.meta.name).toBeNull();
        expect(s.extensions.size).toBe(0);
        expect(b.addExtensionTable("t", [])).toBe(0);
    });

    it("freeze({ release: true }) returns the snapshot and empties staging", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 2);
        b.setNodeValue("x", 0, 1);
        const before = b.byteLength();
        expect(before).toBeGreaterThan(0);
        const s = b.freeze({ release: true });
        expect(s.nodeCount).toBe(2);
        expect(s.nodes.value("x", 0)).toBe(1);
        expect(b.nodeBound).toBe(0);
        expect(b.edgeBound).toBe(0);
        expect(b.byteLength()).toBeLessThan(before);
        expect(b.nodeColumn("x")).toBe(INVALID_INDEX);
        expect(s.ids.toArray()).toEqual(["a", "b"]);
        b.addNode("c");
        expect(s.ids.indexOf("c")).toBe(INVALID_INDEX);
        expect(b.indexOf("c")).toBe(0);
    });

    it("dispose makes every further call throw E_BUILDER_DISPOSED", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.dispose();
        b.dispose();
        const calls: (() => unknown)[] = [
            () => b.addNode("x"),
            () => b.addNodes(["x"]),
            () => b.addAnonymousNodes(1),
            () => b.hasNode("a"),
            () => b.indexOf("a"),
            () => b.idOf(0),
            () => b.removeNode("a"),
            () => b.removeNodeByIndex(0),
            () => b.reserve(1, 1),
            () => b.addEdge("a", "b"),
            () => b.addEdgeByIndex(0, 1),
            () => b.addEdges(new Uint32Array(0), new Uint32Array(0)),
            () => b.addEdgesByIds([], []),
            () => b.removeEdge(0),
            () => b.hasEdge(0),
            () => b.edgeEndpoints(0),
            () => b.edgeWeight(0),
            () => b.setEdgeWeight(0, 1),
            () => b.outEdgesOf(0),
            () => b.inEdgesOf(0),
            () => b.findEdges(0, 1),
            () => b.declareNodeColumn({ name: "x", dtype: "f32" }),
            () => b.declareEdgeColumn({ name: "x", dtype: "f32" }),
            () => b.nodeColumn("x"),
            () => b.edgeColumn("x"),
            () => b.setNodeValue("x", 0, 1),
            () => b.setEdgeValue("x", 0, 1),
            () => b.setNodeColumn("x", new Float32Array(0)),
            () => b.setEdgeColumn("x", new Float32Array(0)),
            () => b.setGraphValue("x", 1),
            () => b.setMeta({}),
            () => b.addExtensionTable("t", []),
            () => b.addExtensionRow(0 as never, []),
            () => b.addNodeRecord("a", {}),
            () => b.addEdgeRecord("a", "b", {}),
            () => b.addGraph(new GraphBuilder({ directed: true }).freeze()),
            () => b.freeze(),
            () => b.freezeWithReport(),
            () => b.clear(),
            () => b.byteLength(),
            () => b.setDirected(false),
            () => b.lockDirected(),
        ];
        for (const call of calls) {
            expect(code(call)).toBe("E_BUILDER_DISPOSED");
        }
        // the getters too (design section 6.1: every further call)
        expect(code(() => b.options)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.directed)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.directedLocked)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.nodeCount)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.dirty)).toBe("E_BUILDER_DISPOSED");
    });

    it("byteLength reports typed staging only", () => {
        const b = new GraphBuilder({ directed: true });
        expect(b.byteLength()).toBe(0);
        b.addEdge("a", "b");
        const withEdge = b.byteLength();
        expect(withEdge).toBeGreaterThan(0);
        b.declareNodeColumn({ name: "x", dtype: "f64" });
        b.setNodeValue("x", 0, 1);
        expect(b.byteLength()).toBeGreaterThan(withEdge);
    });
});

describe("GraphBuilder composition", () => {
    it("from(snapshot) reproduces the topology, ids, weights, columns, extensions and meta", () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("a", "b", 0.1);
        b.addEdge("b", "c");
        b.addEdge("c", "c", 3);
        b.setNodeValue("label", 0, "A");
        b.declareNodeColumn({ name: "cat", dtype: "dict" });
        b.setNodeValue("cat", 1, "x");
        b.setNodeValue("cat", 2, "y");
        b.declareEdgeColumn({ name: "pair", dtype: "u32", refersTo: "edge" });
        b.setEdgeValue("pair", 0, 2);
        b.setEdgeValue("tags", 1, ["t"]);
        b.setGraphValue("title", "g");
        b.setMeta({ name: "meta", extra: { a: 1 } });
        const t = b.addExtensionTable("temporal:node:v", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [2, 5]);
        const original = b.freeze();
        const copy = GraphBuilder.from(original);
        expect(copy.options.weightDtype).toBe("f64");
        expect(copy.directed).toBe(false);
        expect(copy.nodeBound).toBe(3);
        expect(copy.edgeBound).toBe(3);
        expect(copy.dirty).toBe(true);
        const again = copy.freeze();
        assertInvariants(again);
        expect(equalsTopology(again, original)).toBe(true);
        expect(again.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(again.nodes.value("label", 0)).toBe("A");
        expect(again.nodes.isSet("label", 1)).toBe(false);
        expect(again.nodes.requireTyped("cat", "dict").dictionary).toEqual(["x", "y"]);
        expect(again.nodes.value("cat", 2)).toBe("y");
        expect(again.edges.value("pair", 0)).toBe(2);
        expect(again.edges.value("tags", 1)).toEqual(["t"]);
        expect(again.graph.value("title", 0)).toBe("g");
        expect(again.meta.name).toBe("meta");
        expect(again.meta.extra).toEqual({ a: 1 });
        expect(again.extensions.get("temporal:node:v")?.value("element", 0)).toBe(2);
        const shadow = again.edges.byRole("weight");
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.value(0)).toBe(0.1);
        expect(shadow?.isSet(1)).toBe(false);
        expect(shadow?.value(2)).toBe(3);
        expect(copy.edgeWeight(0)).toBe(0.1);
        expect(copy.dirty).toBe(false);
    });

    it("from() honours option overrides and anonymous identity ids", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1, 2]), new Float32Array([2, 3]));
        const original = b.freeze();
        const copy = GraphBuilder.from(original, { duplicateEdges: "sum", directed: undefined });
        expect(copy.options.duplicateEdges).toBe("sum");
        expect(copy.options.weightDtype).toBe("f32");
        const again = copy.freeze();
        expect(again.ids.kind).toBe("identity");
        expect(equalsTopology(again, original)).toBe(true);
        expect(again.edges.byRole("weight")).toBeNull();
    });

    it("addGraph appends a disjoint graph, merges by id, and refuses duplicates with onDuplicateNode error", () => {
        const first = new GraphBuilder({ directed: true });
        first.addEdge("a", "b", 2);
        first.setNodeValue("n", 0, 1);
        first.declareEdgeColumn({ name: "kind", dtype: "string" });
        first.setEdgeValue("kind", 0, "one");
        const one = first.freeze();
        const second = new GraphBuilder({ directed: true });
        second.addEdge("b", "c");
        second.addEdge("c", "b");
        second.setNodeValue("n", 0, "x");
        second.setNodeValue("m", 1, true);
        second.setEdgeValue("kind", 1, "two");
        const two = second.freeze();

        const merged = new GraphBuilder({ directed: true });
        merged.addGraph(one);
        merged.addGraph(two);
        expect(merged.nodeCount).toBe(3);
        expect(merged.edgeCount).toBe(3);
        expect(merged.edgeEndpoints(1)).toEqual([1, 2]);
        expect(merged.edgeWeight(0)).toBe(2);
        const { snapshot, report } = merged.freezeWithReport();
        assertInvariants(snapshot);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(snapshot.nodes.require("n").dtype).toBe("string");
        expect(snapshot.nodes.value("n", 0)).toBe("1");
        expect(snapshot.nodes.value("n", 1)).toBe("x");
        expect(snapshot.nodes.value("m", 2)).toBe(true);
        expect(snapshot.edges.value("kind", 0)).toBe("one");
        expect(snapshot.edges.value("kind", 2)).toBe("two");
        expect(report.widened).toEqual([{ column: "n", domain: "node", from: "i32", to: "string" }]);

        const strict = new GraphBuilder({ directed: true });
        strict.addGraph(one);
        expect(code(() => strict.addGraph(two, { onDuplicateNode: "error" }))).toBe("E_DUPLICATE_ID");
        strict.addGraph(two, { onDuplicateNode: "merge" });
        expect(strict.nodeCount).toBe(3);
    });

    it("addGraph widens declared columns to the union dtype and re-interns dictionaries", () => {
        const a = new GraphBuilder({ directed: true });
        a.addNodes(["x", "y"]);
        a.declareNodeColumn({ name: "c", dtype: "dict" });
        a.setNodeValue("c", 0, "p");
        a.setNodeValue("c", 1, "q");
        a.declareNodeColumn({ name: "u", dtype: "u8" });
        a.setNodeValue("u", 0, 7);
        a.declareNodeColumn({ name: "l", dtype: "list", itemDtype: "i32" });
        a.setNodeValue("l", 0, [1, 2]);
        const first = a.freeze();
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["y", "z"]);
        b.declareNodeColumn({ name: "c", dtype: "dict", options: ["q", "r"] });
        b.setNodeValue("c", 1, "r");
        b.setNodeValue("c", 0, "q");
        b.declareNodeColumn({ name: "u", dtype: "f32" });
        b.setNodeValue("u", 1, 2.5);
        b.declareNodeColumn({ name: "l", dtype: "json" });
        b.setNodeValue("l", 1, { k: 1 });
        const secondSnapshot = b.freeze();
        const union = new GraphBuilder({ directed: true });
        union.addGraph(first);
        union.addGraph(secondSnapshot);
        const s = union.freeze();
        assertInvariants(s);
        const c = s.nodes.requireTyped("c", "dict");
        expect(c.dictionary).toEqual(["p", "q", "r"]);
        expect([0, 1, 2].map((r) => c.value(r))).toEqual(["p", "q", "r"]);
        expect(s.nodes.require("u").dtype).toBe("f64");
        expect(s.nodes.value("u", 0)).toBe(7);
        expect(s.nodes.value("u", 2)).toBe(2.5);
        expect(s.nodes.require("l").dtype).toBe("json");
        expect(s.nodes.value("l", 0)).toEqual([1, 2]);
        expect(s.nodes.value("l", 2)).toEqual({ k: 1 });
    });

    it("addGraph rewrites refersTo columns and extension references into the builder's index space", () => {
        const a = new GraphBuilder({ directed: true });
        a.addEdge("p", "q");
        a.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", role: "parent" });
        a.setNodeValue("parent", 1, 0);
        a.declareEdgeColumn({ name: "pair", dtype: "u32", refersTo: "edge" });
        a.setEdgeValue("pair", 0, 0);
        const t = a.addExtensionTable("ext", [{ name: "element", dtype: "u32", refersTo: "node" }]);
        a.addExtensionRow(t, [1]);
        const donor = a.freeze();
        const b = new GraphBuilder({ directed: true });
        b.addEdge("m", "n");
        b.addGraph(donor);
        b.addGraph(donor);
        const s = b.freeze();
        assertInvariants(s);
        expect(s.ids.toArray()).toEqual(["m", "n", "p", "q"]);
        expect(s.nodes.value("parent", 3)).toBe(2);
        expect(s.nodes.isSet("parent", 0)).toBe(false);
        expect(s.edges.value("pair", 1)).toBe(1);
        expect(s.edges.value("pair", 2)).toBe(2);
        const ext = s.extensions.get("ext");
        expect(ext?.rowCount).toBe(2);
        expect(Array.from(ext?.requireTyped("element", "u32").data ?? [])).toEqual([3, 3]);
    });

    it("widenNodeColumn / widenEdgeColumn widen an inferred column in the 5.1 order without changing values (io round 1)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.setNodeValue("x", 0, 2);
        b.setNodeValue("x", 1, 3);
        b.setEdgeValue("w", 0, 5);
        // a text importer knows from the lexical grammar that "2.0" cells are f64 (design 5.1)
        b.widenNodeColumn("x", "f64");
        b.widenEdgeColumn(b.edgeColumn("w"), "f64");
        // widening to the same dtype again is a no-op; narrowing is refused
        b.widenNodeColumn("x", "f64");
        expect(() => b.widenNodeColumn("x", "i32")).toThrow(GraphFormatError);
        expect(() => b.widenNodeColumn("x", "u32")).toThrow(GraphFormatError);
        expect(() => b.widenNodeColumn("missing", "f64")).toThrow(GraphFormatError);
        b.declareNodeColumn({ name: "d", dtype: "i32" });
        expect(() => b.widenNodeColumn("d", "f64")).toThrow(GraphFormatError);
        const { snapshot, report } = b.freezeWithReport();
        expect(snapshot.nodes.get("x")?.dtype).toBe("f64");
        expect(snapshot.nodes.value("x", 0)).toBe(2);
        expect(snapshot.edges.get("w")?.dtype).toBe("f64");
        expect(report.widened).toEqual([
            { column: "x", domain: "node", from: "i32", to: "f64" },
            { column: "w", domain: "edge", from: "i32", to: "f64" },
        ]);
    });

    it("declaring and resolving many columns by name costs linear time (a Map beside the column array)", () => {
        const time = (count: number): number => {
            const b = new GraphBuilder({ directed: true });
            b.addNode("a");
            const t0 = performance.now();
            for (let i = 0; i < count; i++) {
                b.declareNodeColumn({ name: `c${i}`, dtype: "f64" });
            }
            for (let i = 0; i < count; i++) {
                b.nodeColumn(`c${i}`);
                b.setNodeValue(`c${i}`, 0, i);
            }
            return performance.now() - t0;
        };
        const small = Math.min(time(4000), time(4000), time(4000));
        const large = Math.min(time(16_000), time(16_000), time(16_000));
        // 4x the columns may cost at most 12x (linear plus noise); the former scan cost 16x and more
        expect(large / Math.max(small, 5)).toBeLessThan(12);
    });
});
