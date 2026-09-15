/**
 * Regression tests for the audit round 1 fixes that no pinned audit test covers on its own: the
 * uniform enum-option refusal (E_UNSUPPORTED, never a silent default), the freeze option checks,
 * addGraph direction and option checks, the atomic expansion, the $esc wrapper of the wire's JSON
 * tags, the carried-view content checks at the structure level, fromCsr's presence-flag rule,
 * refusal of resizable-buffer views, the dict fill code, and the transferables() list recorded by
 * a transfer-mode toWire().
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { fromCsr } from "../../src/populate/from-csr.js";
import { fromEdgeArrays } from "../../src/populate/from-edge-arrays.js";
import { fromRecords } from "../../src/populate/from-records.js";
import { type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { foldArcs } from "../../src/snapshot/views.js";
import { type ViewName, type WireBufferRef, type WireSnapshot } from "../../src/types/index.js";
import { fromBytes } from "../../src/wire/bytes.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { assertInvariants } from "../helpers/invariants.js";
import { richSnapshot } from "../wire/fixture-graph.js";

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

function error(fn: () => unknown): GraphFormatError {
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

/** A small directed graph with parallels and a loop. */
function small(): GraphSnapshot {
    return fromEdgeArrays({
        directed: true,
        nodeCount: 4,
        src: new Uint32Array([0, 0, 1, 2, 3]),
        dst: new Uint32Array([1, 1, 2, 3, 3]),
        weights: new Float32Array([1, 2, 3, 4, 5]),
    });
}

describe("round 1: enum options are refused, never substituted (design 11.1)", () => {
    it("every enum-valued option throws E_UNSUPPORTED with details.field", () => {
        const s = small();
        const cases: readonly [string, () => unknown][] = [
            ["validate.level", () => s.validate({ level: "bogus" as never })],
            ["degreeOrder.of", () => s.degreeOrder({ of: "bogus" as never })],
            ["toUndirected.weights", () => s.toUndirected({ weights: "bogus" as never })],
            ["simplified.weights", () => s.simplified({ weights: "bogus" as never })],
            ["simplified.selfLoops", () => s.simplified({ selfLoops: "bogus" as never })],
            ["simplified.edgeReducers", () => s.simplified({ edgeReducers: { w: "bogus" as never } })],
            ["contract.parallel", () => s.contract(new Uint32Array([0, 0, 1, 1]), { parallel: "bogus" as never })],
            ["contract.weights", () => s.contract(new Uint32Array([0, 0, 1, 1]), { weights: "bogus" as never })],
            ["foldArcs.reducer", () => foldArcs(s, new Float64Array(s.arcCount), "bogus" as never)],
            ["set.adopt", () => s.nodes.set("x", new Uint32Array(4), undefined, { adopt: "bogus" as never })],
            ["indicesOf.onMissing", () => s.ids.indicesOf([0], "bogus" as never)],
            ["fromRecords.ids", () => fromRecords({ directed: true, nodes: [], edges: [], ids: "bogus" as never })],
            [
                "addGraph.onDuplicateNode",
                () => new GraphBuilder({ directed: true }).addGraph(s, { onDuplicateNode: "bogus" as never }),
            ],
            ["freeze.arena", () => new GraphBuilder({ directed: true }).freeze({ arena: "yes" as never })],
            ["freeze.label", () => new GraphBuilder({ directed: true }).freeze({ label: 5 as never })],
            ["freeze.prepare", () => new GraphBuilder({ directed: true }).freeze({ prepare: "coo" as never })],
        ];
        for (const [label, call] of cases) {
            const err = error(call);
            expect(err.code, label).toBe("E_UNSUPPORTED");
            expect(err.details.reason, label).toBe("unsupported option");
            expect(typeof err.details.field, label).toBe("string");
        }
        expect(code(() => new GraphBuilder(undefined as never))).toBe("E_UNSUPPORTED");
        expect(code(() => new GraphBuilder(null as never))).toBe("E_UNSUPPORTED");
    });

    it("the documented values still work", () => {
        const s = small();
        expect(s.degreeOrder({ of: "forward" }).perm.length).toBe(4);
        expect(s.degreeOrder({ of: "reverse" }).perm.length).toBe(4);
        s.validate({ level: "structure" });
        expect(s.simplified({ weights: "max", selfLoops: "drop" }).snapshot.edgeCount).toBe(3);
        expect(Array.from(foldArcs(s, new Float64Array([1, 1, 1, 1, 1]), "sum"))).toEqual([1, 1, 1, 1, 1]);
        expect(Array.from(s.ids.indicesOf([9], "invalid"))).toEqual([INVALID_INDEX]);
    });
});

describe("round 1: addGraph and setDirected expansion", () => {
    it("addGraph refuses a snapshot of the other direction with E_DIRECTED and changes nothing", () => {
        const directed = small();
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        const mutations = b.mutationCount;
        const err = error(() => b.addGraph(directed));
        expect(err.code).toBe("E_DIRECTED");
        expect(err.details).toMatchObject({ reason: "direction mismatch", directed: false, found: true });
        expect(b.nodeCount).toBe(2);
        expect(b.edgeCount).toBe(1);
        expect(b.mutationCount).toBe(mutations);
        const undirected = new GraphBuilder({ directed: false }).freeze();
        expect(code(() => new GraphBuilder({ directed: true }).addGraph(undirected))).toBe("E_DIRECTED");
    });

    it("addGraph refuses a weight the unweighted builder cannot hold before it adds anything", () => {
        const b = new GraphBuilder({ directed: true, weighted: false });
        b.addNode("z");
        expect(code(() => b.addGraph(small()))).toBe("E_INVALID_WEIGHT");
        expect(b.nodeCount).toBe(1);
        expect(b.edgeCount).toBe(0);
    });

    it("addGraph keeps a declared (all-omitted) weight array through a weighted-auto builder", () => {
        const source = new GraphBuilder({ directed: true, weighted: true });
        source.addEdge("a", "b");
        const s = source.freeze();
        expect(s.flags.weighted).toBe(true);
        const b = new GraphBuilder({ directed: true });
        b.addGraph(s);
        const out = b.freeze();
        expect(out.flags.weighted).toBe(true);
        expect(Array.from(out.weights as Float32Array)).toEqual([1]);
    });

    it("setDirected(true, { expand: true }) applies nothing when the pair role is already taken", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        b.declareEdgeColumn({ name: "mine", dtype: "u32", role: "pair", refersTo: "edge" });
        expect(code(() => b.setDirected(true, { expand: true }))).toBe("E_DUPLICATE_ROLE");
        expect(b.directed).toBe(false);
        expect(b.edgeColumn("graphty.directed")).toBe(INVALID_INDEX);
        expect(b.edgeCount).toBe(1);
    });

    it("a record with a bad value is not applied at all, even when it would create the node", () => {
        const b = new GraphBuilder({ directed: true });
        b.declareNodeColumn({ name: "n", dtype: "u8" });
        expect(code(() => b.addNodeRecord("fresh", { n: 300 }))).toBe("E_COLUMN_TYPE");
        expect(b.hasNode("fresh")).toBe(false);
        expect(b.nodeCount).toBe(0);
        expect(code(() => b.addEdgeRecord("p", "q", { n: 1, bad: 1n }))).toBe("E_COLUMN_TYPE");
        expect(b.nodeCount).toBe(0);
        expect(b.edgeCount).toBe(0);
        expect(b.edgeColumn("bad")).toBe(INVALID_INDEX);
    });
});

describe("round 1: the wire's JSON tags", () => {
    it("user objects whose only key is $num or $esc round-trip through the wire and the container", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        b.declareNodeColumn({ name: "j", dtype: "json" });
        b.setNodeValue("j", 0, { $num: "text" });
        b.setNodeValue("j", 1, { $esc: { $num: "NaN" } });
        b.setNodeValue("j", 2, { $num: Number.NaN, other: 1 });
        b.setGraphValue("g", { $esc: [1, Infinity] });
        b.setMeta({ extra: { $num: "-0", nested: { $esc: null } } });
        const s = b.freeze();
        for (const back of [fromWire(structuredClone(s.toWire()), { validate: "full" }), fromBytes(s.toBytes())]) {
            expect(back.nodes.value("j", 0)).toEqual({ $num: "text" });
            expect(back.nodes.value("j", 1)).toEqual({ $esc: { $num: "NaN" } });
            const two = back.nodes.value("j", 2) as { $num: number; other: number };
            expect(Number.isNaN(two.$num)).toBe(true);
            expect(two.other).toBe(1);
            expect(back.graph.value("g", 0)).toEqual({ $esc: [1, Infinity] });
            expect(back.meta.extra).toEqual({ $num: "-0", nested: { $esc: null } });
        }
    });

    it("an $esc wrapper around a non-object is E_BAD_SERIALIZATION", () => {
        const s = small();
        const wire = structuredClone(s.toWire());
        (wire.manifest.meta as { extra: Record<string, unknown> }).extra = { x: { $esc: 5 } };
        expect(code(() => fromWire(wire))).toBe("E_BAD_SERIALIZATION");
    });
});

describe("round 1: carried views are content-checked at the structure level", () => {
    const names: readonly ViewName[] = [
        "reverse",
        "coo",
        "edgeList",
        "outDegree",
        "inDegree",
        "degree",
        "weightedOutDegree",
        "weightedInDegree",
        "weightedDegree",
        "selfLoopWeight",
        "selfLoopArcs",
        "selfLoopsPerNode",
        "degreeOrder",
        "reverseDegreeOrder",
    ];

    function poisoned(
        edit: (views: Record<string, Record<string, WireBufferRef>>, w: WireSnapshot) => void,
    ): WireSnapshot {
        const s = richSnapshot({ directed: true });
        s.prepare(names);
        const wire = structuredClone(s.toWire({ includeViews: names }));
        edit(wire.manifest.views as Record<string, Record<string, WireBufferRef>>, wire);
        return wire;
    }

    function viewOf(w: WireSnapshot, ref: WireBufferRef): Uint32Array | Float32Array | Float64Array {
        const buffer = w.buffers[ref.buffer];
        if (ref.dtype === "f64") {
            return new Float64Array(buffer, ref.byteOffset, ref.length);
        }
        if (ref.dtype === "f32") {
            return new Float32Array(buffer, ref.byteOffset, ref.length);
        }
        return new Uint32Array(buffer, ref.byteOffset, ref.length);
    }

    it("intact views install at structure (and at none) and match a fresh computation", () => {
        const wire = poisoned(() => undefined);
        const back = fromWire(wire, { validate: "structure" });
        const fresh = fromWire(structuredClone(wire), { validate: "full" });
        expect(back.cachedViews()).toEqual(fresh.prepare(names).cachedViews());
        for (const name of ["outDegree", "inDegree", "degree", "selfLoopArcs", "selfLoopsPerNode"] as const) {
            expect(Array.from(back[name]())).toEqual(Array.from(fresh[name]()));
        }
        expect(Array.from(back.reverse().colIdx)).toEqual(Array.from(fresh.reverse().colIdx));
        expect(Array.from(back.degreeOrder({ of: "reverse" }).perm)).toEqual(
            Array.from(fresh.degreeOrder({ of: "reverse" }).perm),
        );
        expect(fromWire(structuredClone(wire), { validate: "none" }).cachedViews()).toEqual(back.cachedViews());
    });

    type Views = Record<string, Record<string, WireBufferRef>>;
    /** Bump one element of a member array. */
    const bump =
        (view: string, member: string, at: number, by = 1) =>
        (v: Views, w: WireSnapshot): void => {
            const array = viewOf(w, v[view][member]);
            array[at] += by;
        };
    const corruptions: readonly [string, (v: Views, w: WireSnapshot) => void][] = [
        ["outDegree.data", bump("outDegree", "data", 0)],
        ["inDegree.data", bump("inDegree", "data", 1)],
        ["degree.data", bump("degree", "data", 0)],
        ["selfLoopsPerNode.data", bump("selfLoopsPerNode", "data", 0)],
        ["selfLoopArcs.data", bump("selfLoopArcs", "data", 0)],
        ["coo.src", bump("coo", "src", 0)],
        ["edgeList.src", bump("edgeList", "src", 0)],
        ["edgeList.dst", bump("edgeList", "dst", 0)],
        ["edgeList.weights", bump("edgeList", "weights", 0)],
        ["reverse.colIdx", bump("reverse", "colIdx", 0)],
        [
            "reverse.fwdArc",
            (v, w): void => {
                const a = viewOf(w, v.reverse.fwdArc);
                [a[0], a[1]] = [a[1], a[0]];
            },
        ],
        ["reverse.weights", bump("reverse", "weights", 0)],
        [
            "reverse.rowPtr",
            (v, w): void => {
                const r = viewOf(w, v.reverse.rowPtr);
                r[1] = r[2];
            },
        ],
        [
            "degreeOrder.perm",
            (v, w): void => {
                const p = viewOf(w, v.degreeOrder.perm);
                [p[0], p[p.length - 1]] = [p[p.length - 1], p[0]];
            },
        ],
        ["degreeOrder.segmentOffsets", bump("degreeOrder", "segmentOffsets", 1)],
        [
            "reverseDegreeOrder.perm",
            (v, w): void => {
                viewOf(w, v.reverseDegreeOrder.perm).fill(0);
            },
        ],
        ["weightedOutDegree.data", bump("weightedOutDegree", "data", 0, 0.5)],
        ["weightedInDegree.data", bump("weightedInDegree", "data", 0, 0.5)],
        ["weightedDegree.data", bump("weightedDegree", "data", 0, 0.5)],
        ["selfLoopWeight.data", bump("selfLoopWeight", "data", 0, 0.5)],
    ];

    it.each(corruptions)(
        "a corrupted %s is E_INVALID_SNAPSHOT at structure and installed only at none",
        (member, edit) => {
            const wire = poisoned(edit);
            const err = error(() => fromWire(wire, { validate: "structure" }));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details.reason).toBe("view");
            expect(err.details.view).toBe(member.split(".")[0]);
            // "none" trusts the manifest (design 9.5) and "full" recomputes
            expect(fromWire(structuredClone(wire), { validate: "none" }).cachedViews().length).toBeGreaterThan(0);
            expect(fromWire(structuredClone(wire), { validate: "full" }).cachedViews()).toEqual([]);
        },
    );
});

describe("round 1: fromCsr presence flags and buffer classes", () => {
    it("a claimed arcToEdgeIsIdentity never drops a supplied permutation, at any level", () => {
        const input = {
            directed: true,
            nodeCount: 2,
            rowPtr: new Uint32Array([0, 2, 2]),
            colIdx: new Uint32Array([0, 1]),
            arcToEdge: new Uint32Array([1, 0]),
            edgeToArc: new Uint32Array([1, 0]),
            edgeColumns: { tag: new Uint32Array([10, 11]) },
        };
        for (const level of ["none", "structure", "full"] as const) {
            // the claim contradicts the array: refused (I9) instead of renumbering the edges
            const err = error(() => fromCsr({ ...input, flags: { arcToEdgeIsIdentity: true } }, { validate: level }));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details.flag).toBe("arcToEdgeIsIdentity");
            const honest = fromCsr({ ...input, flags: { arcToEdgeIsIdentity: false } }, { validate: level });
            expect(Array.from(honest.arcToEdge)).toEqual([1, 0]);
            expect(honest.edges.value("tag", 1)).toBe(11);
        }
    });

    it("a presence-flag claim that contradicts the arrays is I9 at every level", () => {
        const base = {
            directed: true,
            nodeCount: 2,
            rowPtr: new Uint32Array([0, 1, 1]),
            colIdx: new Uint32Array([1]),
            weights: new Float32Array([2]),
        };
        for (const level of ["none", "structure", "full"] as const) {
            const err = error(() => fromCsr({ ...base, flags: { weighted: false } }, { validate: level }));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details.invariant).toBe("I9");
            expect(err.details.flag).toBe("weighted");
            const s = fromCsr(
                {
                    ...base,
                    arcToEdge: new Uint32Array([0]),
                    flags: { arcToEdgeIsIdentity: true, weighted: true },
                },
                { validate: level },
            );
            expect(s.flags.arcToEdgeIsIdentity).toBe(true);
            expect(s.flags.weighted).toBe(true);
        }
    });

    it("a supplied non-identity arcToEdge is kept even when claimed identity below full", () => {
        const s = fromCsr(
            {
                directed: true,
                nodeCount: 2,
                rowPtr: new Uint32Array([0, 2, 2]),
                colIdx: new Uint32Array([0, 1]),
                arcToEdge: new Uint32Array([1, 0]),
                edgeToArc: new Uint32Array([1, 0]),
                edgeColumns: { tag: new Uint32Array([10, 11]) },
                flags: { arcToEdgeIsIdentity: false },
            },
            { validate: "structure" },
        );
        expect(Array.from(s.arcToEdge)).toEqual([1, 0]);
        expect(Array.from(s.edgeList().src)).toEqual([0, 0]);
        expect(s.edges.value("tag", 0)).toBe(10);
        expect(
            code(() =>
                fromCsr(
                    {
                        directed: true,
                        nodeCount: 2,
                        rowPtr: new Uint32Array([0, 2, 2]),
                        colIdx: new Uint32Array([0, 1]),
                        arcToEdge: new Uint32Array([1, 0]),
                        flags: { arcToEdgeIsIdentity: true },
                    },
                    { validate: "structure" },
                ),
            ),
        ).toBe("E_INVALID_SNAPSHOT");
    });

    it("columns over a SharedArrayBuffer or a resizable buffer are copied by fromCsr and refused by set()", () => {
        const shared = new Uint32Array(new SharedArrayBuffer(8)) as unknown as Uint32Array<ArrayBuffer>;
        shared.set([5, 6]);
        const resizable = new ArrayBuffer(8, { maxByteLength: 64 });
        const growing = new Uint32Array(resizable);
        growing.set([7, 8]);
        const s = fromCsr({
            directed: true,
            nodeCount: 2,
            rowPtr: new Uint32Array([0, 1, 1]),
            colIdx: new Uint32Array([1]),
            nodeColumns: { a: shared, b: growing, c: { data: growing, decl: { dtype: "u32" } } },
        });
        for (const name of ["a", "b", "c"]) {
            const column = s.nodes.requireTyped(name, "u32");
            expect(column.data.buffer).toBeInstanceOf(ArrayBuffer);
            expect(column.data.buffer.resizable).toBe(false);
        }
        expect(s.nodes.value("a", 1)).toBe(6);
        expect(s.nodes.value("b", 1)).toBe(8);
        resizable.resize(64);
        expect(s.nodes.requireTyped("b", "u32").data.length).toBe(2);
        assertInvariants(s);
        const sab = error(() => s.nodes.set("d", shared));
        expect(sab.code).toBe("E_UNSUPPORTED");
        expect(sab.details.reason).toBe("SharedArrayBuffer");
        const rz = error(() => s.nodes.set("e", new Uint32Array(resizable, 0, 2)));
        expect(rz.code).toBe("E_UNSUPPORTED");
        expect(rz.details.reason).toBe("resizable ArrayBuffer");
        expect(s.nodes.has("d")).toBe(false);
        expect(s.nodes.has("e")).toBe(false);
    });

    it("a foreign object shaped like a Column is refused by set()", () => {
        const s = small();
        const fake = { dtype: "f32", meta: { name: "z", domain: "node", role: null }, length: 4 };
        const err = error(() => s.nodes.set("z", fake as never));
        expect(err.code).toBe("E_COLUMN_TYPE");
        expect(err.details.reason).toBe("foreign column");
        expect(s.nodes.has("z")).toBe(false);
    });
});

describe("round 1: dictionary fill codes and unique-column codes", () => {
    it("a non-nullable dict column with no options holds a real member in its unwritten rows", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        b.declareNodeColumn({ name: "k", dtype: "dict", nullable: false });
        b.setNodeValue("k", 2, "x");
        const s = b.freeze();
        const k = s.nodes.requireTyped("k", "dict");
        expect(k.value(0)).toBe("");
        expect(k.value(2)).toBe("x");
        expect(k.dictionary).toContain("");
        s.validate({ level: "full" });
        // a nullable one needs no member for its unset rows
        const c = new GraphBuilder({ directed: true });
        c.addAnonymousNodes(2);
        c.setNodeValue("k", 1, "y");
        c.declareNodeColumn({ name: "d", dtype: "dict" });
        c.setNodeValue("d", 1, "y");
        const t = c.freeze();
        expect(t.nodes.requireTyped("d", "dict").dictionary).toEqual(["y"]);
        t.validate({ level: "full" });
    });

    it("a violated unique column is E_INVALID_SNAPSHOT from validate() and E_DUPLICATE_* from the freeze", () => {
        const s = small();
        s.nodes.set("uq", [1, 1, 2, 3], { dtype: "i32", unique: true });
        const err = error(() => s.validate());
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details).toMatchObject({
            invariant: "I12",
            reason: "unique",
            cause: "E_DUPLICATE_ID",
            column: "uq",
        });
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b");
        b.addEdge("b", "c");
        b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
        b.setEdgeValue("id", 0, "same");
        b.setEdgeValue("id", 1, "same");
        expect(code(() => b.freeze())).toBe("E_DUPLICATE_EDGE_ID");
    });
});

describe("round 1: transferables() follows the transfer-mode wire", () => {
    it("lists exactly the buffers of the last toWire({ transfer: true }), copies included", () => {
        const s = richSnapshot();
        const plain = s.transferables();
        expect(plain.length).toBeGreaterThan(0);
        const wire = s.toWire({ transfer: true, includeColumns: false });
        const list = s.transferables();
        expect(list).toEqual(wire.buffers);
        const moved = structuredClone(wire, { transfer: list });
        expect(fromWire(moved, { validate: "full" }).nodes.names()).toEqual([]);
        // the columns were not in the wire, so the sender still has them
        expect(s.nodes.value("label", 0)).toBe("node 0 \u00e9");
        expect(s.detached).toBe(true);
    });
});
