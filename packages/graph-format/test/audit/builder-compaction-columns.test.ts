/**
 * Adversarial audit of attribute data through compaction (design sections 5.11, 6.3 step 10, 5.10
 * and invariant I12): every dtype's values and validity must survive a compacting freeze, refersTo
 * columns must be rewritten through the SAME remap the FreezeReport returns, dangling references must
 * become INVALID_INDEX with the row UNSET (never a set row holding INVALID_INDEX), and extension
 * tables must be remapped without losing rows.
 *
 * Two tests here fail deliberately; each pins a defect named in the audit report:
 * - "merged edges: refersTo values follow the survivor" (design 5.11 / 6.5 / 7.3: a merged edge is
 *   not dangling, it maps to its survivor; the compaction rewrites through a remap that sends it to
 *   INVALID_INDEX instead, so graphty.pair and temporal extension rows are destroyed by a merging
 *   freeze while simplified() on the same graph keeps them);
 * - "a non-nullable refersTo column with a dangling reference" (the compaction leaves the row SET
 *   with INVALID_INDEX, so the frozen snapshot violates I12 and validate() rejects it).
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { type ExtensionHandle, type U32 } from "../../src/types/index.js";
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

function list(array: U32 | null): number[] | null {
    return array === null ? null : Array.from(array);
}

describe("audit: column values and validity through compaction", () => {
    it("every dtype keeps its set rows, unset rows and fills when a node and its edges compact away", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c", "d"]);
        b.addEdge("a", "b"); // e0 dies
        b.addEdge("b", "c"); // e1 dies
        b.addEdge("c", "d"); // e2 -> 0
        b.addEdge("d", "a"); // e3 -> 1
        b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, fill: -1 });
        b.setNodeValue("pos", 0, [1, 2, 3]);
        b.setNodeValue("pos", 1, [4, 5, 6]);
        b.setNodeValue("pos", 3, [7, 8, 9]);
        b.declareNodeColumn({ name: "flag", dtype: "bool", fill: true });
        b.setNodeValue("flag", 0, false);
        b.setNodeValue("flag", 1, false);
        b.setNodeValue("flag", 2, true);
        b.declareNodeColumn({ name: "kind", dtype: "dict", options: ["x", "y"], fill: "y" });
        b.setNodeValue("kind", 0, "z");
        b.setNodeValue("kind", 1, "w");
        b.setNodeValue("kind", 3, "x");
        b.declareNodeColumn({ name: "name", dtype: "string", fill: "?" });
        b.setNodeValue("name", 0, "A");
        b.setNodeValue("name", 1, "B");
        b.setNodeValue("name", 2, "C");
        b.declareNodeColumn({ name: "vals", dtype: "list", itemDtype: "f32" });
        b.setNodeValue("vals", 0, [1.5]);
        b.setNodeValue("vals", 1, [9]);
        b.setNodeValue("vals", 3, [2.5, 3.5]);
        b.declareNodeColumn({ name: "meta", dtype: "json" });
        b.setNodeValue("meta", 0, { k: 1 });
        b.setNodeValue("meta", 1, { k: 2 });
        b.setNodeValue("meta", 2, null);
        b.setNodeValue("meta", 3, [1, 2]);
        b.declareNodeColumn({ name: "small", dtype: "u8", nullable: false, fill: 9 });
        b.setNodeValue("small", 1, 100);
        b.setNodeValue("small", 3, 200);
        b.setNodeValue("auto", 0, 1);
        b.setNodeValue("auto", 1, 2);
        b.setNodeValue("auto", 2, "two");
        b.setNodeValue("auto", 3, true);
        b.setNodeColumn("bulk", new Float64Array([0.5, 1.5, 2.5, 3.5]));
        b.setEdgeValue("num", 0, 10);
        b.setEdgeValue("num", 1, 11);
        b.setEdgeValue("num", 3, 40);
        b.declareEdgeColumn({ name: "endOf", dtype: "u32", refersTo: "node" });
        b.setEdgeValue("endOf", 2, 3); // c -> d refers to d (3 -> 2)
        b.setEdgeValue("endOf", 3, 1); // d -> a refers to b, which dies
        b.removeNode("b");
        const { snapshot: s, report } = b.freezeWithReport();
        assertInvariants(s);
        expect(list(report.nodeRemap)).toEqual([0, INVALID_INDEX, 1, 2]);
        expect(list(report.edgeRemap)).toEqual([INVALID_INDEX, INVALID_INDEX, 0, 1]);
        expect(s.ids.toArray()).toEqual(["a", "c", "d"]);

        const pos = s.nodes.requireTyped("pos", "f32");
        expect(Array.from(pos.data)).toEqual([1, 2, 3, -1, -1, -1, 7, 8, 9]);
        expect([0, 1, 2].map((r) => pos.isSet(r))).toEqual([true, false, true]);
        const flag = s.nodes.requireTyped("flag", "bool");
        expect([0, 1, 2].map((r) => [flag.isSet(r), flag.value(r)])).toEqual([
            [true, false],
            [true, true],
            [false, undefined],
        ]);
        const kind = s.nodes.requireTyped("kind", "dict");
        expect([0, 1, 2].map((r) => [kind.isSet(r), kind.value(r)])).toEqual([
            [true, "z"],
            [false, undefined],
            [true, "x"],
        ]);
        expect(kind.dictionary[kind.codes[1]]).toBe("y");
        const name = s.nodes.requireTyped("name", "string");
        expect([0, 1, 2].map((r) => [name.isSet(r), name.valueAt(r)])).toEqual([
            [true, "A"],
            [true, "C"],
            [false, "?"],
        ]);
        expect([0, 1, 2].map((r) => [s.nodes.isSet("vals", r), s.nodes.value("vals", r)])).toEqual([
            [true, [1.5]],
            [false, undefined],
            [true, [2.5, 3.5]],
        ]);
        expect([0, 1, 2].map((r) => [s.nodes.isSet("meta", r), s.nodes.value("meta", r)])).toEqual([
            [true, { k: 1 }],
            [true, null],
            [true, [1, 2]],
        ]);
        const small = s.nodes.requireTyped("small", "u8");
        expect(small.meta.nullable).toBe(false);
        expect(small.validity).toBeNull();
        expect(Array.from(small.data.subarray(0, 3))).toEqual([9, 9, 200]);
        const auto = s.nodes.require("auto");
        expect(auto.dtype).toBe("string");
        expect([0, 1, 2].map((r) => auto.value(r))).toEqual(["1", "two", "true"]);
        const bulk = s.nodes.requireTyped("bulk", "f64");
        expect(bulk.meta.nullable).toBe(false);
        expect(Array.from(bulk.data)).toEqual([0.5, 2.5, 3.5]);
        expect([0, 1].map((e) => [s.edges.isSet("num", e), s.edges.value("num", e)])).toEqual([
            [false, undefined],
            [true, 40],
        ]);
        const endOf = s.edges.requireTyped("endOf", "u32");
        expect(Array.from(endOf.data)).toEqual([2, INVALID_INDEX]);
        expect([0, 1].map((e) => endOf.isSet(e))).toEqual([true, false]);
    });

    it("a second compaction rewrites references that the first one rewrote", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c", "d", "e"]);
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node" });
        b.declareNodeColumn({ name: "kin", dtype: "list", itemDtype: "u32", refersTo: "node" });
        b.setNodeValue("parent", 4, 3);
        b.setNodeValue("parent", 3, 2);
        b.setNodeValue("parent", 2, 1);
        b.setNodeValue("kin", 4, [0, 1, 2, 3]);
        b.setNodeValue("kin", 0, [1]);
        b.removeNode("b");
        const s1 = b.freeze();
        assertInvariants(s1);
        expect(s1.ids.toArray()).toEqual(["a", "c", "d", "e"]);
        expect(Array.from(s1.nodes.requireTyped("parent", "u32").data)).toEqual([INVALID_INDEX, INVALID_INDEX, 1, 2]);
        expect(s1.nodes.value("kin", 3)).toEqual([0, 1, 2]);
        expect(s1.nodes.isSet("kin", 0)).toBe(false);
        // the builder's own column now holds the rewritten values
        b.removeNode("c");
        const s2 = b.freeze();
        assertInvariants(s2);
        expect(s2.ids.toArray()).toEqual(["a", "d", "e"]);
        expect(Array.from(s2.nodes.requireTyped("parent", "u32").data)).toEqual([INVALID_INDEX, INVALID_INDEX, 1]);
        expect([0, 1, 2].map((r) => s2.nodes.isSet("parent", r))).toEqual([false, false, true]);
        expect(s2.nodes.value("kin", 2)).toEqual([0, 1]);
        // the first snapshot is unchanged
        expect(Array.from(s1.nodes.requireTyped("parent", "u32").data)).toEqual([INVALID_INDEX, INVALID_INDEX, 1, 2]);
    });

    it("extension tables keep every row, rewriting node and edge references through both remaps", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b"); // e0
        b.addEdge("b", "c"); // e1
        b.addEdge("c", "a"); // e2
        const t: ExtensionHandle = b.addExtensionTable("temporal:node:x", [
            { name: "element", dtype: "u32", refersTo: "node" },
            { name: "edgeRef", dtype: "u32", refersTo: "edge" },
            { name: "start", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [0, 0, 1]);
        b.addExtensionRow(t, [1, 1, 2]);
        b.addExtensionRow(t, [2, 2, 3]);
        b.freeze();
        b.removeNode("a"); // kills e0 and e2
        const s2 = b.freeze();
        assertInvariants(s2);
        const t2 = s2.extensions.get("temporal:node:x");
        expect(t2).not.toBeNull();
        if (t2 === undefined) {
            throw new Error("extension table missing");
        }
        expect(t2.rowCount).toBe(3);
        expect(Array.from(t2.requireTyped("element", "u32").data)).toEqual([INVALID_INDEX, 0, 1]);
        expect(Array.from(t2.requireTyped("edgeRef", "u32").data)).toEqual([INVALID_INDEX, 0, INVALID_INDEX]);
        expect([0, 1, 2].map((r) => [t2.isSet("element", r), t2.isSet("edgeRef", r)])).toEqual([
            [false, false],
            [true, true],
            [true, false],
        ]);
        expect(Array.from(t2.requireTyped("start", "f64").data)).toEqual([1, 2, 3]);
        // rows appended after the first compaction use the builder's (renumbered) index space
        b.addEdge("c", "b"); // edge 1
        b.addExtensionRow(t, [0, 1, 4]);
        b.removeNode("c"); // kills both edges
        const s3 = b.freeze();
        assertInvariants(s3);
        const t3 = s3.extensions.get("temporal:node:x");
        if (t3 === undefined) {
            throw new Error("extension table missing");
        }
        expect(s3.ids.toArray()).toEqual(["b"]);
        expect(t3.rowCount).toBe(4);
        expect(Array.from(t3.requireTyped("element", "u32").data)).toEqual([INVALID_INDEX, 0, INVALID_INDEX, 0]);
        expect(Array.from(t3.requireTyped("edgeRef", "u32").data)).toEqual(new Array<number>(4).fill(INVALID_INDEX));
        expect(Array.from(t3.requireTyped("start", "f64").data)).toEqual([1, 2, 3, 4]);
        // the earlier snapshot's table is untouched
        expect(Array.from(t2.requireTyped("element", "u32").data)).toEqual([INVALID_INDEX, 0, 1]);
    });

    it("non-survivor rows of a merge are gone, survivor rows keep the documented edge's attributes", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", 1); // e0
        b.addEdge("b", "a", 2); // e1 same undirected edge as e0
        b.addEdge("a", "b", 3); // e2 same again
        b.addEdge("b", "c", 4); // e3
        b.declareEdgeColumn({ name: "tag", dtype: "string" });
        b.setEdgeValue("tag", 1, "one");
        b.setEdgeValue("tag", 2, "two");
        b.setEdgeValue("tag", 3, "three");
        b.setEdgeValue("num", 0, 10);
        b.setEdgeValue("num", 2, 12);
        const expected = {
            first: { weight: 1, tag: undefined, num: 10, orientation: [0, 1] },
            sum: { weight: 6, tag: undefined, num: 10, orientation: [0, 1] },
            min: { weight: 1, tag: undefined, num: 10, orientation: [0, 1] },
            max: { weight: 3, tag: undefined, num: 10, orientation: [0, 1] },
            last: { weight: 3, tag: "two", num: 12, orientation: [0, 1] },
        } as const;
        for (const policy of ["first", "sum", "min", "max", "last"] as const) {
            const fresh = GraphBuilder.from(b.freeze());
            const { snapshot, report } = fresh.freezeWithReport({ duplicateEdges: policy });
            assertInvariants(snapshot);
            expect(snapshot.edgeCount).toBe(2);
            expect(snapshot.flags.multigraph).toBe(false);
            expect(list(report.edgeRemap)).toEqual([0, 0, 0, 1]);
            expect(report.mergedEdges).toBe(2);
            const want = expected[policy];
            expect([snapshot.edgeSource(0), snapshot.edgeTarget(0)]).toEqual(want.orientation);
            expect((snapshot.weights as Float32Array)[snapshot.edgeToArc[0]]).toBe(want.weight);
            expect(snapshot.edges.value("tag", 0)).toBe(want.tag);
            expect(snapshot.edges.value("num", 0)).toBe(want.num);
            expect(snapshot.edges.value("tag", 1)).toBe("three");
            expect(snapshot.edges.isSet("num", 1)).toBe(false);
        }
    });
});

describe("audit: refersTo rewriting agrees with the report's remap", () => {
    // PINS FINDING: merged edges are treated as dangling by the compaction's refersTo rewrite.
    it("merged edges: refersTo values follow the survivor, as edgeRemap and simplified() say", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", 1); // e0
        b.addEdge("b", "a", 2); // e1
        b.setDirected(true, { expand: true }); // e2 mirrors e0 as b -> a, e3 mirrors e1 as a -> b
        const t = b.addExtensionTable("temporal:edge:w", [
            { name: "element", dtype: "u32", refersTo: "edge" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [0, 10]);
        b.addExtensionRow(t, [3, 13]); // refers to e3, which merges into e0
        b.declareNodeColumn({ name: "firstEdge", dtype: "u32", refersTo: "edge" });
        b.setNodeValue("firstEdge", 0, 3); // a -> e3 (merges into e0 -> new 0)
        b.setNodeValue("firstEdge", 1, 2); // b -> e2 (merges into e1 -> new 1)

        // the reference behaviour: the same merge through simplified() keeps every reference
        const keep = b.freeze();
        const simplified = keep.simplified({ weights: "sum" });
        expect(list(simplified.edgeRemap)).toEqual([0, 1, 1, 0]);
        expect(Array.from(simplified.snapshot.edges.requireTyped("graphty.pair", "u32").data)).toEqual([1, 0]);

        const { snapshot, report } = b.freezeWithReport({ duplicateEdges: "sum" });
        assertInvariants(snapshot);
        expect(list(report.edgeRemap)).toEqual([0, 1, 1, 0]);
        expect(report.mergedEdges).toBe(2);
        expect(snapshot.edgeCount).toBe(2);
        // graphty.pair (design 3.6): each survivor's mate is the other survivor
        const pair = snapshot.edges.requireTyped("graphty.pair", "u32");
        expect([0, 1].map((e) => pair.isSet(e))).toEqual([true, true]);
        expect(Array.from(pair.data)).toEqual([1, 0]);
        // a node column referring to edges follows the survivors
        const firstEdge = snapshot.nodes.requireTyped("firstEdge", "u32");
        expect([0, 1].map((r) => firstEdge.isSet(r))).toEqual([true, true]);
        expect(Array.from(firstEdge.data)).toEqual([0, 1]);
        // an extension row on a merged edge follows the survivor instead of being dropped
        const table = snapshot.extensions.get("temporal:edge:w");
        if (table === undefined) {
            throw new Error("extension table missing");
        }
        const element = table.requireTyped("element", "u32");
        expect([0, 1].map((r) => element.isSet(r))).toEqual([true, true]);
        expect(Array.from(element.data)).toEqual([0, 0]);
        // and the builder, rewritten to the merged set, agrees with its own snapshot
        expect(b.freeze().edges.requireTyped("graphty.pair", "u32").isSet(0)).toBe(true);
    });

    it("directed parallels: an extension row on a merged edge follows edgeRemap", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1); // e0
        b.addEdge("a", "b", 2); // e1 merges into e0
        b.addEdge("b", "c", 3); // e2
        const t = b.addExtensionTable("temporal:edge:w", [
            { name: "element", dtype: "u32", refersTo: "edge" },
            { name: "value", dtype: "f64" },
        ]);
        b.addExtensionRow(t, [0, 10]);
        b.addExtensionRow(t, [1, 11]);
        b.addExtensionRow(t, [2, 12]);
        const { snapshot, report } = b.freezeWithReport({ duplicateEdges: "sum" });
        assertInvariants(snapshot);
        const { edgeRemap } = report;
        expect(list(edgeRemap)).toEqual([0, 0, 1]);
        if (edgeRemap === null) {
            throw new Error("expected an edge remap");
        }
        const table = snapshot.extensions.get("temporal:edge:w");
        if (table === undefined) {
            throw new Error("extension table missing");
        }
        const element = table.requireTyped("element", "u32");
        // every row's reference equals edgeRemap[old reference]; none dangles
        expect(Array.from(element.data)).toEqual([edgeRemap[0], edgeRemap[1], edgeRemap[2]]);
        expect([0, 1, 2].map((r) => element.isSet(r))).toEqual([true, true, true]);
    });
});

describe("audit: dangling references and I12", () => {
    // PINS FINDING: a non-nullable refersTo column is left SET with INVALID_INDEX by compaction.
    it("a non-nullable refersTo column with a dangling reference freezes to a snapshot that satisfies I12", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", nullable: false });
        b.setNodeValue("parent", 0, 1);
        b.setNodeValue("parent", 1, 2);
        b.setNodeValue("parent", 2, 0);
        b.removeNode("b");
        const s = b.freeze();
        const parent = s.nodes.requireTyped("parent", "u32");
        expect(Array.from(parent.data)).toEqual([INVALID_INDEX, 0]);
        // design 5.11: a dangling reference is INVALID_INDEX with the row UNSET; I12 forbids a set row
        // holding INVALID_INDEX; remapColumn() flips such a column to nullable for exactly this case
        expect(parent.isSet(0)).toBe(false);
        expect(code(() => s.validate({ level: "full" }))).toBeNull();
    });

    // VERIFIER (round 1): the same defect on the path without compaction. A non-nullable refersTo
    // column whose rows were never (all) written holds the INVALID_INDEX fill in the unwritten rows;
    // the frozen column must not carry those rows SET (I12), so it becomes nullable with exactly the
    // unwritten rows unset, and the builder's own declaration stays non-nullable.
    it("a non-nullable refersTo column with unwritten rows freezes to a snapshot that satisfies I12", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", nullable: false });
        const untouched = b.freeze();
        const none = untouched.nodes.requireTyped("parent", "u32");
        expect(Array.from(none.data)).toEqual([INVALID_INDEX, INVALID_INDEX, INVALID_INDEX]);
        expect(none.meta.nullable).toBe(true);
        expect(none.nullCount).toBe(3);
        expect([none.isSet(0), none.isSet(1), none.isSet(2)]).toEqual([false, false, false]);
        expect(code(() => untouched.validate({ level: "full" }))).toBeNull();

        b.setNodeValue("parent", 1, 0);
        const partial = b.freeze();
        const some = partial.nodes.requireTyped("parent", "u32");
        expect(Array.from(some.data)).toEqual([INVALID_INDEX, 0, INVALID_INDEX]);
        expect(some.meta.nullable).toBe(true);
        expect(some.nullCount).toBe(2);
        expect([some.isSet(0), some.isSet(1), some.isSet(2)]).toEqual([false, true, false]);
        expect(code(() => partial.validate({ level: "full" }))).toBeNull();
        assertInvariants(partial);

        // the builder still refuses to unset a row of its non-nullable column
        expect(code(() => b.setNodeValue("parent", 0, INVALID_INDEX))).toBe("E_COLUMN_TYPE");

        b.setNodeValue("parent", 0, 2);
        b.setNodeValue("parent", 2, 1);
        const full = b.freeze();
        const all = full.nodes.requireTyped("parent", "u32");
        expect(Array.from(all.data)).toEqual([2, 0, 1]);
        expect(all.meta.nullable).toBe(false);
        expect(all.validity).toBeNull();
        expect(code(() => full.validate({ level: "full" }))).toBeNull();

        // an edge column referring to edges, and a refersTo column with an explicit in-range fill
        b.addEdge("a", "b");
        b.declareEdgeColumn({ name: "twin", dtype: "u32", refersTo: "edge", nullable: false });
        b.declareNodeColumn({ name: "root", dtype: "u32", refersTo: "node", nullable: false, fill: 0 });
        const mixed = b.freeze();
        expect(mixed.edges.requireTyped("twin", "u32").meta.nullable).toBe(true);
        expect(mixed.edges.require("twin").isSet(0)).toBe(false);
        const root = mixed.nodes.requireTyped("root", "u32");
        expect(root.meta.nullable).toBe(false);
        expect(Array.from(root.data)).toEqual([0, 0, 0]);
        expect(code(() => mixed.validate({ level: "full" }))).toBeNull();
    });

    it("the public remapColumn helper handles the same column correctly (the reference behaviour)", async () => {
        const { remapColumn } = await import("../../src/columns/remap.js");
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", nullable: false });
        b.setNodeValue("parent", 0, 1);
        b.setNodeValue("parent", 1, 2);
        b.setNodeValue("parent", 2, 0);
        const before = b.freeze();
        b.removeNode("b");
        const { report } = b.freezeWithReport();
        const { nodeRemap } = report;
        if (nodeRemap === null) {
            throw new Error("expected a node remap");
        }
        const moved = remapColumn(before.nodes.require("parent"), nodeRemap, 2);
        expect(moved.dtype).toBe("u32");
        if (moved.dtype !== "u32") {
            throw new Error("unreachable");
        }
        expect(Array.from(moved.data)).toEqual([INVALID_INDEX, 0]);
        expect(moved.isSet(0)).toBe(false);
        expect(moved.meta.nullable).toBe(true);
    });

    it("a nullable refersTo column and a refersTo list drop dangling items and stay valid", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b", "c"]);
        b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node" });
        b.declareNodeColumn({ name: "kin", dtype: "list", itemDtype: "u32", refersTo: "node" });
        b.setNodeValue("parent", 0, 1);
        b.setNodeValue("parent", 2, 1);
        b.setNodeValue("kin", 0, [1]);
        b.setNodeValue("kin", 2, [1, 0, 1]);
        b.setNodeValue("kin", 1, []);
        b.removeNode("b");
        const s = b.freeze();
        assertInvariants(s);
        const parent = s.nodes.requireTyped("parent", "u32");
        expect(Array.from(parent.data)).toEqual([INVALID_INDEX, INVALID_INDEX]);
        expect(parent.nullCount).toBe(2);
        expect(s.nodes.isSet("kin", 0)).toBe(false);
        expect(s.nodes.value("kin", 1)).toEqual([0]);
    });
});
