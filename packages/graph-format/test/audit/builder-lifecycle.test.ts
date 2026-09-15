/**
 * Adversarial audit of the builder lifecycle (design sections 6.1, 6.5, 6.6, 6.7, 11.1 and 12.2):
 * dispose() must make every further use throw E_BUILDER_DISPOSED, freeze({ release: true }) must hand
 * everything to the snapshot and leave a working empty builder, clear() must reset the staging but
 * keep the options and the lock, an unsupported freeze option must be refused rather than acted on,
 * a merging freeze must leave the builder consistent with its own declared options, and a failing
 * composition call must not be half-applied.
 *
 * Tests marked PINS FINDING fail deliberately; each names the defect of the audit report it pins.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { type ColumnHandle, type ExtensionHandle } from "../../src/types/index.js";
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

describe("audit: dispose()", () => {
    it("is idempotent, keeps every snapshot usable and refuses every method afterwards", () => {
        const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
        b.addEdge("a", "b", 2.5);
        b.setNodeValue("x", 0, 1);
        const s = b.freeze();
        b.dispose();
        b.dispose();
        expect(s.ids.toArray()).toEqual(["a", "b"]);
        expect(s.nodes.value("x", 0)).toBe(1);
        expect(s.edgeCount).toBe(1);
        assertInvariants(s);
        expect(code(() => b.freeze())).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.freeze({ release: true }))).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.freezeWithReport({ profile: true }))).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.addNode("a"))).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.removeNode("a"))).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.hasEdge(0))).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.byteLength())).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.clear())).toBe("E_BUILDER_DISPOSED");
        // a builder seeded from the disposed builder's snapshot is fully independent of it
        const seeded = GraphBuilder.from(s);
        expect(code(() => seeded.addGraph(s, { onDuplicateNode: "error" }))).toBe("E_DUPLICATE_ID");
        expect(seeded.freeze().ids.toArray()).toEqual(["a", "b"]);
    });

    // PINS FINDING: the state getters answer after dispose() instead of throwing.
    it("the state getters throw E_BUILDER_DISPOSED too instead of reporting an empty, dirty builder", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.dispose();
        // design 6.1: dispose() "makes every further call throw E_BUILDER_DISPOSED"; a getter that
        // returns 0 nodes and dirty === true describes a builder that no longer exists
        expect(code(() => b.nodeCount)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.edgeCount)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.nodeBound)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.edgeBound)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.dirty)).toBe("E_BUILDER_DISPOSED");
        expect(code(() => b.mutationCount)).toBe("E_BUILDER_DISPOSED");
    });
});

describe("audit: freeze({ release: true })", () => {
    it("returns a complete snapshot, empties staging, keeps options, lock, meta and graph values", () => {
        const b = new GraphBuilder({ directed: true, weighted: true, weightDtype: "f64" });
        b.lockDirected();
        b.addEdge("a", "b", 0.1);
        b.addEdge("b", "c");
        const kind: ColumnHandle = b.declareNodeColumn({ name: "kind", dtype: "dict" });
        b.setNodeValue(kind, 0, "x");
        b.setEdgeValue("tag", 1, "t");
        const t: ExtensionHandle = b.addExtensionTable("t", [{ name: "v", dtype: "f64" }]);
        b.addExtensionRow(t, [1]);
        b.setGraphValue("g", 5);
        b.setMeta({ name: "graph", keywords: ["k"] });
        b.removeNode("c");
        const { snapshot: s, report } = b.freezeWithReport({ release: true, checksum: true });
        assertInvariants(s);
        expect(Array.from(report.nodeRemap as Uint32Array)).toEqual([0, 1, INVALID_INDEX]);
        expect(Array.from(report.edgeRemap as Uint32Array)).toEqual([0, INVALID_INDEX]);
        expect(s.nodes.value("kind", 0)).toBe("x");
        expect(s.edges.byRole("weight")?.dtype).toBe("f64");
        expect(s.extensions.get("t")?.rowCount).toBe(1);
        expect(s.graph.value("g", 0)).toBe(5);
        expect(s.meta.name).toBe("graph");
        // the builder is empty but alive, with its configuration intact
        expect(b.nodeBound).toBe(0);
        expect(b.edgeBound).toBe(0);
        expect(b.nodeCount).toBe(0);
        expect(b.directedLocked).toBe(true);
        expect(b.options.weightDtype).toBe("f64");
        expect(b.nodeColumn("kind")).toBe(INVALID_INDEX);
        expect(b.edgeColumn("tag")).toBe(INVALID_INDEX);
        expect(code(() => b.setNodeValue(kind, 0, "y"))).toBe("E_INDEX_RANGE");
        expect(code(() => b.addExtensionRow(t, [2]))).toBe("E_INDEX_RANGE");
        expect(code(() => b.setDirected(false))).toBe("E_DIRECTED");
        // the second freeze is a fresh, empty graph that still carries the graph-level state
        b.addNode("z");
        expect(code(() => b.setNodeValue(kind, 0, "y"))).toBe("E_UNKNOWN_COLUMN");
        const next = b.freezeWithReport();
        assertInvariants(next.snapshot);
        expect(next.report.nodeRemap).toBeNull();
        expect(next.report.compacted).toBe(false);
        expect(next.snapshot.nodeCount).toBe(1);
        expect(next.snapshot.nodes.names()).toEqual([]);
        expect(next.snapshot.extensions.size).toBe(0);
        expect(next.snapshot.graph.value("g", 0)).toBe(5);
        expect(next.snapshot.meta.name).toBe("graph");
        // the released snapshot is untouched and its checksums still hold
        expect(s.nodeCount).toBe(2);
        s.validate({ level: "full", checksum: true });
    });

    it("clear() resets staging, graph values and meta, bumps mutationCount and keeps the lock", () => {
        const b = new GraphBuilder({ directed: true });
        b.lockDirected();
        b.addEdge("a", "b");
        b.setGraphValue("g", 1);
        b.setMeta({ name: "n" });
        const s = b.freeze();
        const mutations = b.mutationCount;
        b.clear();
        expect(b.mutationCount).toBe(mutations + 1);
        expect(b.dirty).toBe(true);
        expect(b.directedLocked).toBe(true);
        expect(b.hasNode("a")).toBe(false);
        expect(b.freeze().graph.names()).toEqual([]);
        expect(b.freeze().meta.name).toBeNull();
        expect(s.ids.toArray()).toEqual(["a", "b"]);
        expect(s.graph.value("g", 0)).toBe(1);
    });
});

describe("audit: freeze options are validated, never silently substituted", () => {
    // PINS FINDING: an unsupported per-freeze duplicateEdges value acts like "first" and rewrites the builder.
    it("an unsupported duplicateEdges override throws E_UNSUPPORTED and leaves the builder untouched", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 2);
        // the constructor refuses the same value
        expect(code(() => new GraphBuilder({ directed: true, duplicateEdges: "bogus" as never }))).toBe(
            "E_UNSUPPORTED",
        );
        const before = b.freeze();
        expect(before.edgeCount).toBe(2);
        expect(code(() => b.freeze({ duplicateEdges: "bogus" as never }))).toBe("E_UNSUPPORTED");
        // nothing merged, nothing rewritten
        expect(b.edgeCount).toBe(2);
        expect(b.freeze().edgeCount).toBe(2);
    });

    it("an unknown prepare name is E_UNSUPPORTED and does not commit a compaction", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.removeNode("c");
        expect(code(() => b.freeze({ prepare: ["nope" as never] }))).toBe("E_UNSUPPORTED");
        expect(b.nodeBound).toBe(3);
        expect(b.edgeBound).toBe(2);
        expect(b.dirty).toBe(true);
    });
});

describe("audit: merging freezes and the builder's declared options", () => {
    // PINS FINDING: "sum" multiplicities on omitted weights are recorded as omitted and lost by from().
    it("multiplicities produced by duplicateEdges sum survive GraphBuilder.from() (design 6.6: from() seeds weights)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        const s = b.freeze({ duplicateEdges: "sum" });
        assertInvariants(s);
        expect(s.flags.weighted).toBe(true);
        expect(Array.from(s.weights as Float32Array)).toEqual([2, 1]);
        expect(b.edgeWeight(0)).toBe(2);
        // the role-"weight" column exists (some edge omitted its weight); the reduced weight is THE
        // weight of edge 0 (design 3.7: "the single weight per edge is THE weight"), so the column
        // must not report it as absent, or exporters and from() drop the 2
        const shadow = s.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        if (shadow === null) {
            throw new Error("unreachable");
        }
        expect(shadow.isSet(0)).toBe(true);
        const again = GraphBuilder.from(s).freeze();
        expect(again.weights).not.toBeNull();
        expect(Array.from(again.weights as Float32Array)).toEqual([2, 1]);
    });

    // PINS FINDING: a merge allocates weights on a builder declared weighted: false.
    it("a weighted: false builder never gains an arc weight array through a merge", () => {
        const b = new GraphBuilder({ directed: true, weighted: false });
        b.addEdge("a", "b");
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        const s = b.freeze({ duplicateEdges: "sum" });
        assertInvariants(s);
        expect(b.options.weighted).toBe(false);
        // design 12.2 / STATUS: weighted: false means no arc weight array; the builder refuses any
        // weight other than 1, so it cannot also hold a weight of 2 for edge 0
        expect(code(() => b.addEdge("a", "c", 2))).toBe("E_INVALID_WEIGHT");
        expect(b.edgeWeight(0)).toBe(1);
        expect(s.weights).toBeNull();
        expect(s.flags.weighted).toBe(false);
    });

    it("merge policies on explicit weights keep the builder and the snapshot in agreement", () => {
        for (const policy of ["sum", "min", "max", "first", "last"] as const) {
            const b = new GraphBuilder({ directed: false, weightDtype: "f64" });
            b.addEdge("a", "b", 0.1);
            b.addEdge("b", "a", 0.2);
            b.addEdge("a", "b", 0.7);
            const s = b.freeze({ duplicateEdges: policy });
            assertInvariants(s);
            expect(s.edgeCount).toBe(1);
            const shadow = s.edges.byRole("weight");
            const exact = shadow !== null && shadow.dtype === "f64" ? shadow.data[0] : undefined;
            const stored = exact ?? (s.weights as Float32Array)[s.edgeToArc[0]];
            expect(b.edgeWeight(0)).toBe(stored);
            const expected = { sum: 1, min: 0.1, max: 0.7, first: 0.1, last: 0.7 }[policy];
            expect(b.edgeWeight(0)).toBeCloseTo(expected, 12);
            expect(b.dirty).toBe(false);
            // re-freezing with the same policy changes nothing
            const again = b.freezeWithReport({ duplicateEdges: policy });
            expect(again.report.edgeRemap).toBeNull();
            expect(again.report.mergedEdges).toBe(0);
        }
    });
});

describe("audit: composition calls are atomic on failure (design 11.1)", () => {
    // PINS FINDING: addGraph with onDuplicateNode "error" appends nodes before it throws.
    it("addGraph(..., { onDuplicateNode: 'error' }) leaves the builder unchanged when it throws", () => {
        const other = new GraphBuilder({ directed: true });
        other.addNodes(["a", "b", "c"]);
        other.addEdge("a", "b");
        const incoming = other.freeze();
        const b = new GraphBuilder({ directed: true });
        b.addNode("c");
        const mutations = b.mutationCount;
        expect(code(() => b.addGraph(incoming, { onDuplicateNode: "error" }))).toBe("E_DUPLICATE_ID");
        expect(b.nodeCount).toBe(1);
        expect(b.nodeBound).toBe(1);
        expect(b.hasNode("a")).toBe(false);
        expect(b.edgeCount).toBe(0);
        expect(b.mutationCount).toBe(mutations);
    });
});
