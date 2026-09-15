/**
 * Adversarial audit of the id map shared by reference between the builder and its snapshots (design
 * sections 4.2, 4.4 and 6.3 step 1, decision C17): a snapshot's map is the builder's `Map` / `ids`
 * array guarded by `index < nodeCount`; a compacting freeze must build FRESH objects so every earlier
 * snapshot keeps the ones it was frozen with; appends, revivals, clear() and dispose() after a freeze
 * must never leak into a snapshot; and the wire form of an old snapshot must encode the old ids.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type GraphSnapshot, type NodeId } from "../../src/types/index.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { assertInvariants } from "../helpers/invariants.js";

/** Every public read of an id map, compared against an expected id list. */
function expectIds(s: GraphSnapshot, expected: readonly NodeId[], absent: readonly NodeId[]): void {
    const { ids } = s;
    expect(ids.size).toBe(expected.length);
    expect(ids.toArray()).toEqual(expected);
    expect([...ids]).toEqual(expected);
    expect(ids.idsSlice()).toEqual(expected);
    expect(ids.idsSlice(1)).toEqual(expected.slice(1));
    expected.forEach((id, i) => {
        expect(ids.idOf(i)).toBe(id);
        expect(ids.indexOf(id)).toBe(i);
        expect(ids.has(id)).toBe(true);
        expect(ids.requireIndex(id)).toBe(i);
        expect(ids.stringIndex().get(String(id))).toBe(i);
    });
    for (const id of absent) {
        expect(ids.indexOf(id)).toBe(INVALID_INDEX);
        expect(ids.has(id)).toBe(false);
        expect(ids.stringIndex().has(String(id))).toBe(false);
    }
    expect(Array.from(ids.indicesOf([...expected, ...absent]))).toEqual([
        ...expected.map((_, i) => i),
        ...absent.map(() => INVALID_INDEX),
    ]);
    expect(() => ids.idOf(expected.length)).toThrow();
    const values = expected.map((_, i) => i * 10);
    expect([...ids.toMap(values).entries()]).toEqual(expected.map((id, i) => [id, i * 10]));
    expect([...ids.entries(values)]).toEqual(expected.map((id, i) => [id, i * 10]));
    // the wire form carries exactly these ids
    const back = fromWire(s.toWire(), { validate: "full" });
    expect(back.ids.toArray()).toEqual(expected);
    expect(back.ids.kind).toBe(ids.kind);
}

const KINDS: readonly { readonly name: string; readonly ids: readonly NodeId[]; readonly kind: string }[] = [
    { name: "string", ids: ["a", "b", "c", "d"], kind: "string" },
    { name: "numeric", ids: [10, 20, 300, 4.5], kind: "numeric" },
    { name: "mixed", ids: ["a", 1, "b", 2], kind: "mixed" },
    { name: "dense", ids: [0, 2, 4, 6], kind: "dense" },
    { name: "identity", ids: [0, 1, 2, 3], kind: "identity" },
];

describe("audit: a snapshot's id map survives every later builder mutation", () => {
    for (const { name, ids, kind } of KINDS) {
        it(`${name}: three compacting freezes, appends and a revival leave earlier snapshots intact`, () => {
            const b = new GraphBuilder({ directed: true });
            for (const id of ids) {
                b.addNode(id);
            }
            const s1 = b.freeze();
            expect(s1.ids.kind).toBe(kind);
            expectIds(s1, ids, ["zz", -1, 99]);

            // append without compaction: s1 must not see the new id
            b.addNode("late");
            expectIds(s1, ids, ["late"]);

            // compaction 1: drop ids[1]
            b.removeNode(ids[1]);
            const s2 = b.freeze();
            assertInvariants(s2);
            const expected2 = [ids[0], ids[2], ids[3], "late"];
            expectIds(s2, expected2, [ids[1]]);
            expectIds(s1, ids, ["late"]);

            // revive-then-remove churn and compaction 2: drop ids[0], add ids[1] back at a NEW index
            b.removeNode(ids[0]);
            expect(b.addNode(ids[0])).toBe(0);
            b.removeNode(ids[0]);
            expect(b.addNode(ids[1])).toBe(4);
            const s3 = b.freeze();
            assertInvariants(s3);
            const expected3 = [ids[2], ids[3], "late", ids[1]];
            expectIds(s3, expected3, [ids[0]]);
            expectIds(s2, expected2, [ids[1]]);
            expectIds(s1, ids, ["late"]);

            // compaction 3 through a merging freeze (edges only) plus clear() and dispose()
            b.addEdge(ids[2], ids[3]);
            b.addEdge(ids[2], ids[3]);
            const s4 = b.freeze({ duplicateEdges: "first" });
            expectIds(s4, expected3, [ids[0]]);
            b.clear();
            b.addNodes(["p", "q"]);
            const s5 = b.freeze();
            expectIds(s5, ["p", "q"], [...ids, "late"]);
            b.dispose();
            expectIds(s1, ids, ["late"]);
            expectIds(s2, expected2, [ids[1]]);
            expectIds(s3, expected3, [ids[0]]);
            expectIds(s4, expected3, [ids[0]]);
            expectIds(s5, ["p", "q"], ids);
        });
    }

    it("the shared Map never exposes a builder index at or beyond nodeCount, even for a revived id", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b"]);
        const s = b.freeze();
        b.addNode("c");
        b.removeNode("c");
        b.addNode("c");
        expect(b.indexOf("c")).toBe(2);
        expect(s.ids.indexOf("c")).toBe(INVALID_INDEX);
        expect(s.ids.has("c")).toBe(false);
        expect(() => s.ids.requireIndex("c")).toThrow();
        expect(Array.from(s.ids.indicesOf(["c", "a"]))).toEqual([INVALID_INDEX, 0]);
        expect(s.ids.toArray()).toEqual(["a", "b"]);
        expect(s.ids.stringIndex().has("c")).toBe(false);
        // the wire form of the old snapshot encodes only its own ids although the builder's array grew
        const back = fromWire(s.toWire(), { validate: "full" });
        expect(back.ids.toArray()).toEqual(["a", "b"]);
    });

    it("freeze({ release: true }) hands the map to the snapshot and the builder starts fresh", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes(["a", "b"]);
        b.removeNode("a");
        const s = b.freeze({ release: true });
        expect(s.ids.toArray()).toEqual(["b"]);
        expect(b.addNode("z")).toBe(0);
        expect(b.addNode("b")).toBe(1);
        expect(s.ids.indexOf("z")).toBe(INVALID_INDEX);
        expect(s.ids.indexOf("b")).toBe(0);
        const next = b.freeze();
        expect(next.ids.toArray()).toEqual(["z", "b"]);
        expect(s.ids.toArray()).toEqual(["b"]);
    });

    it("anonymous builders keep an identity map whose ids renumber with the compaction", () => {
        const b = new GraphBuilder({ directed: false });
        b.addAnonymousNodes(4);
        b.addEdges(new Uint32Array([0, 1, 2]), new Uint32Array([1, 2, 3]));
        const s1 = b.freeze();
        expect(s1.ids.kind).toBe("identity");
        b.removeNodeByIndex(0);
        const { snapshot: s2, report } = b.freezeWithReport();
        assertInvariants(s2);
        expect(s2.ids.kind).toBe("identity");
        expect(s2.ids.toArray()).toEqual([0, 1, 2]);
        expect(Array.from(report.nodeRemap as Uint32Array)).toEqual([INVALID_INDEX, 0, 1, 2]);
        expect(s1.ids.toArray()).toEqual([0, 1, 2, 3]);
        // the builder now answers by the new ids
        expect(b.idOf(0)).toBe(0);
        expect(b.indexOf(3)).toBe(INVALID_INDEX);
        expect(b.hasNode(2)).toBe(true);
    });
});
