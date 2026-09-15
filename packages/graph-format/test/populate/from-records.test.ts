/**
 * Unit tests of fromRecords (design section 8.1, entry point 3): node-link records against the
 * builder's record API, the key defaults and fallbacks, positional (d3 v3) nodes, id coercion, the
 * weight key, the column modes (infer / json / none / declared) with inference and widening through
 * the freeze report, the builder policies and freeze options, and every rejection code.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { coerceId, fromRecords } from "../../src/populate/from-records.js";
import { type GraphSnapshot, type RecordsInput } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { KARATE_EDGES } from "../helpers/parts.js";

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

function expectSameCore(a: GraphSnapshot, b: GraphSnapshot): void {
    expect(a.directed).toBe(b.directed);
    expect(a.nodeCount).toBe(b.nodeCount);
    expect(a.edgeCount).toBe(b.edgeCount);
    expect(a.arcCount).toBe(b.arcCount);
    expect(Array.from(a.rowPtr)).toEqual(Array.from(b.rowPtr));
    expect(Array.from(a.colIdx)).toEqual(Array.from(b.colIdx));
    expect(Array.from(a.arcToEdge)).toEqual(Array.from(b.arcToEdge));
    expect(Array.from(a.edgeToArc)).toEqual(Array.from(b.edgeToArc));
    expect(a.weights === null).toBe(b.weights === null);
    if (a.weights !== null && b.weights !== null) {
        expect(Array.from(a.weights)).toEqual(Array.from(b.weights));
    }
    expect(a.flags).toEqual(b.flags);
    expect(a.ids.toArray()).toEqual(b.ids.toArray());
}

const NODES = [
    { id: "a", label: "Alice", age: 30, active: true },
    { id: "b", label: "Bob", age: 25, active: false, tags: ["x", "y"] },
    { id: "c", label: "Carol", age: null },
];

const EDGES = [
    { source: "a", target: "b", weight: 2, kind: "friend" },
    { source: "b", target: "c", weight: 0.5, kind: "colleague", since: 2020 },
    { source: "c", target: "a", kind: "friend" },
    { source: "a", target: "d" },
];

describe("fromRecords: round trips", () => {
    it("matches the builder's record API for node-link records", () => {
        const { snapshot, report } = fromRecords({ directed: true, nodes: NODES, edges: EDGES });
        assertInvariants(snapshot);
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        for (const node of NODES) {
            const { id, ...attrs } = node;
            builder.addNodeRecord(id, attrs);
        }
        for (const edge of EDGES) {
            const { source, target, ...attrs } = edge;
            builder.addEdgeRecord(source, target, attrs);
        }
        const expected = builder.freeze();
        expectSameCore(snapshot, expected);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d"]);
        expect(snapshot.nodes.names()).toEqual(expected.nodes.names());
        expect(snapshot.edges.names()).toEqual(expected.edges.names());
        for (const name of snapshot.nodes.names()) {
            for (let i = 0; i < snapshot.nodeCount; i++) {
                expect(snapshot.nodes.value(name, i)).toEqual(expected.nodes.value(name, i));
            }
        }
        expect(report.compacted).toBe(false);
        expect(report.widened).toEqual([]);
    });

    it("infers column dtypes per column, keeps the id out of the columns and unsets missing keys", () => {
        const { snapshot } = fromRecords({ directed: false, nodes: NODES, edges: EDGES });
        assertInvariants(snapshot);
        expect(snapshot.nodes.has("id")).toBe(false);
        expect(snapshot.edges.has("source")).toBe(false);
        expect(snapshot.edges.has("target")).toBe(false);
        expect(snapshot.edges.has("weight")).toBe(false);
        expect(snapshot.nodes.requireTyped("label", "string").value(1)).toBe("Bob");
        expect(snapshot.nodes.requireTyped("label", "string").isSet(3)).toBe(false);
        const age = snapshot.nodes.requireTyped("age", "i32");
        expect(age.value(0)).toBe(30);
        expect(age.isSet(2)).toBe(false);
        expect(age.nullCount).toBe(2);
        expect(snapshot.nodes.requireTyped("active", "bool").value(0)).toBe(true);
        expect(snapshot.nodes.requireTyped("tags", "json").value(1)).toEqual(["x", "y"]);
        expect(snapshot.edges.requireTyped("kind", "string").value(2)).toBe("friend");
        expect(snapshot.edges.requireTyped("since", "i32").value(1)).toBe(2020);
        expect(snapshot.edges.requireTyped("since", "i32").isSet(0)).toBe(false);
    });

    it("stores the weight in the arc array with an f64 shadow only when not f32-exact", () => {
        const { snapshot } = fromRecords({ directed: true, nodes: NODES, edges: EDGES });
        expect(snapshot.flags.weighted).toBe(true);
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual([2, 0.5, 1, 1]);
        const shadow = snapshot.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.dtype).toBe("f32");
        expect(shadow?.isSet(0)).toBe(true);
        expect(shadow?.isSet(2)).toBe(false);
        const inexact = fromRecords({
            directed: true,
            edges: [{ source: 1, target: 2, weight: 0.1 }],
        }).snapshot;
        expect(inexact.edges.byRole("weight")?.dtype).toBe("f64");
        expect(inexact.edges.byRole("weight")?.value(0)).toBe(0.1);
        expect(inexact.weights?.[0]).toBe(Math.fround(0.1));
        const f32 = fromRecords(
            { directed: true, edges: [{ source: 1, target: 2, weight: 0.1 }] },
            { weightDtype: "f32" },
        );
        expect(f32.snapshot.edges.byRole("weight")).toBeNull();
    });

    it("creates nodes from edges alone in first-seen order and preserves edge order", () => {
        const { snapshot } = fromRecords({
            directed: false,
            edges: KARATE_EDGES.map(([source, target]) => ({ source, target })),
        });
        assertInvariants(snapshot);
        expect(snapshot.nodeCount).toBe(34);
        expect(snapshot.edgeCount).toBe(KARATE_EDGES.length);
        // ids are the integers 0..33 but indices follow first appearance in the edge list
        expect(snapshot.ids.kind).toBe("dense");
        const seen: number[] = [];
        for (const [u, v] of KARATE_EDGES) {
            for (const id of [u, v]) {
                if (!seen.includes(id)) {
                    seen.push(id);
                }
            }
        }
        expect(snapshot.ids.toArray()).toEqual(seen);
        KARATE_EDGES.forEach(([u, v], e) => {
            expect(snapshot.edgeSource(e)).toBe(snapshot.ids.indexOf(u));
            expect(snapshot.edgeTarget(e)).toBe(snapshot.ids.indexOf(v));
        });
        expect(snapshot.weights).toBeNull();
        expect(snapshot.edges.byRole("weight")).toBeNull();
    });

    it("keeps parallel edges and self-loops from records", () => {
        const { snapshot } = fromRecords({
            directed: true,
            edges: [
                { source: "a", target: "b" },
                { source: "a", target: "b" },
                { source: "a", target: "a" },
            ],
        });
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.selfLoopCount).toBe(1);
    });

    it("accepts any iterable of records and an empty edge list", () => {
        function* nodes(): Generator<Record<string, unknown>> {
            yield { id: 1 };
            yield { id: 2 };
        }
        const { snapshot } = fromRecords({ directed: true, nodes: nodes(), edges: new Set() });
        assertInvariants(snapshot);
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.edgeCount).toBe(0);
        const empty = fromRecords({ directed: false, edges: [] }).snapshot;
        expect(empty.nodeCount).toBe(0);
    });
});

describe("fromRecords: keys", () => {
    it("falls back to src / from and dst / to for the endpoints", () => {
        const { snapshot } = fromRecords({
            directed: true,
            edges: [
                { src: "a", dst: "b" },
                { from: "b", to: "c" },
                { source: "c", target: "a", src: "ignored" },
            ],
        });
        assertInvariants(snapshot);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(snapshot.edgeSource(2)).toBe(2);
        expect(snapshot.edgeTarget(2)).toBe(0);
        // the key that was not used stays an attribute
        expect(snapshot.edges.requireTyped("src", "string").value(2)).toBe("ignored");
        expect(snapshot.edges.requireTyped("src", "string").isSet(0)).toBe(false);
    });

    it("uses explicit keys only and treats the defaults as attributes", () => {
        const { snapshot } = fromRecords({
            directed: true,
            nodes: [{ name: "n1", id: "attr" }],
            edges: [{ u: "n1", v: "n2", cost: 3, source: "s", weight: 9 }],
            nodeId: "name",
            edgeSource: "u",
            edgeTarget: "v",
            edgeWeight: "cost",
        });
        assertInvariants(snapshot);
        expect(snapshot.ids.toArray()).toEqual(["n1", "n2"]);
        expect(snapshot.nodes.requireTyped("id", "string").value(0)).toBe("attr");
        expect(snapshot.edges.requireTyped("source", "string").value(0)).toBe("s");
        expect(snapshot.edges.requireTyped("weight", "i32").value(0)).toBe(9);
        expect(snapshot.weights?.[0]).toBe(3);
    });

    it("treats weight as an attribute under edgeWeight: null", () => {
        const { snapshot } = fromRecords({
            directed: true,
            edges: [{ source: "a", target: "b", weight: 5 }],
            edgeWeight: null,
        });
        expect(snapshot.weights).toBeNull();
        expect(snapshot.edges.requireTyped("weight", "i32").value(0)).toBe(5);
    });

    it("rejects a record without its id or an endpoint (E_INVALID_ID)", () => {
        const node = thrown(() => fromRecords({ directed: true, nodes: [{ label: "x" }], edges: [] }));
        expect(node.code).toBe("E_INVALID_ID");
        expect(node.details).toMatchObject({ field: "id", node: 0, reason: "missing id" });
        const source = thrown(() => fromRecords({ directed: true, edges: [{ target: "b" }] }));
        expect(source.code).toBe("E_INVALID_ID");
        expect(source.details).toMatchObject({ field: "source", edge: 0, keys: ["source", "src", "from"] });
        const target = thrown(() =>
            fromRecords({ directed: true, edges: [{ source: "a", dst: "b" }], edgeTarget: "to" }),
        );
        expect(target.details).toMatchObject({ field: "target", keys: ["to"] });
        expect(code(() => fromRecords({ directed: true, edges: [{ source: "a", target: undefined }] }))).toBe(
            "E_INVALID_ID",
        );
    });

    it("rejects a non-numeric weight (E_INVALID_WEIGHT) and a NaN weight", () => {
        const err = thrown(() => fromRecords({ directed: true, edges: [{ source: 1, target: 2, weight: "3" }] }));
        expect(err.code).toBe("E_INVALID_WEIGHT");
        expect(err.details).toMatchObject({ key: "weight", found: "string", edge: 0 });
        expect(code(() => fromRecords({ directed: true, edges: [{ source: 1, target: 2, weight: Number.NaN }] }))).toBe(
            "E_INVALID_WEIGHT",
        );
        // null and undefined weights are "no weight"
        const { snapshot } = fromRecords({
            directed: true,
            edges: [
                { source: 1, target: 2, weight: null },
                { source: 2, target: 3, weight: 4 },
            ],
        });
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual([1, 4]);
    });
});

describe("fromRecords: positional nodes (nodeId: null)", () => {
    it("uses the record position as the index and endpoints as indices", () => {
        const { snapshot } = fromRecords({
            directed: true,
            nodeId: null,
            nodes: [{ id: "ignored-as-attribute", name: "n0" }, { name: "n1" }, { name: "n2" }],
            edges: [
                { source: 0, target: 2 },
                { source: 2, target: 1, weight: 3 },
            ],
        });
        assertInvariants(snapshot);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.idOf(2)).toBe(2);
        expect(snapshot.nodes.requireTyped("id", "string").value(0)).toBe("ignored-as-attribute");
        expect(snapshot.edgeSource(1)).toBe(2);
        expect(snapshot.edgeTarget(1)).toBe(1);
        expect(snapshot.weights?.[snapshot.edgeToArc[1]]).toBe(3);
    });

    it("extends the node range for an unseen index under addMissingNodes and refuses it otherwise", () => {
        const { snapshot } = fromRecords({ directed: true, nodeId: null, edges: [{ source: 0, target: 4 }] });
        expect(snapshot.nodeCount).toBe(5);
        expect(snapshot.edgeSource(0)).toBe(0);
        expect(snapshot.edgeTarget(0)).toBe(4);
        const err = thrown(() =>
            fromRecords(
                { directed: true, nodeId: null, nodes: [{}], edges: [{ source: 0, target: 4 }] },
                { addMissingNodes: false },
            ),
        );
        expect(err.code).toBe("E_UNKNOWN_NODE");
        expect(err.details).toMatchObject({ index: 4, field: "target", edge: 0 });
    });

    it("rejects an endpoint that is not an index (E_INVALID_ID)", () => {
        for (const bad of ["1", 1.5, -1, null, undefined]) {
            const err = thrown(() =>
                fromRecords({ directed: true, nodeId: null, edges: [{ source: bad, target: 0 }] }),
            );
            expect(err.code).toBe("E_INVALID_ID");
            expect(err.details).toMatchObject({ field: "source", reason: "not an index" });
        }
    });
});

describe("fromRecords: id coercion", () => {
    it("keep leaves typed values alone and rejects other shapes", () => {
        expect(coerceId(1, "keep", "id")).toBe(1);
        expect(coerceId("1", "keep", "id")).toBe("1");
        expect(code(() => coerceId(true, "keep", "id"))).toBe("E_INVALID_ID");
        expect(code(() => coerceId(null, "keep", "id"))).toBe("E_INVALID_ID");
        expect(code(() => coerceId({}, "keep", "id"))).toBe("E_INVALID_ID");
        const { snapshot } = fromRecords({ directed: true, edges: [{ source: 1, target: "1" }] });
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.ids.toArray()).toEqual([1, "1"]);
    });

    it("canonical turns canonical integer text into numbers and nothing else", () => {
        expect(coerceId("1", "canonical", "id")).toBe(1);
        expect(coerceId("-12", "canonical", "id")).toBe(-12);
        expect(coerceId("0", "canonical", "id")).toBe(0);
        expect(coerceId("01", "canonical", "id")).toBe("01");
        expect(coerceId("1.0", "canonical", "id")).toBe("1.0");
        expect(coerceId("+1", "canonical", "id")).toBe("+1");
        expect(coerceId(" 1", "canonical", "id")).toBe(" 1");
        expect(coerceId("-0", "canonical", "id")).toBe("-0");
        expect(coerceId("9007199254740993", "canonical", "id")).toBe("9007199254740993");
        expect(coerceId(2.5, "canonical", "id")).toBe(2.5);
        expect(code(() => coerceId(false, "canonical", "id"))).toBe("E_INVALID_ID");
        const { snapshot } = fromRecords({
            directed: true,
            nodes: [{ id: "1" }, { id: "x" }],
            edges: [{ source: 1, target: "x" }],
            ids: "canonical",
        });
        expect(snapshot.nodeCount).toBe(2);
        expect(snapshot.ids.kind).toBe("mixed");
        expect(snapshot.ids.indexOf(1)).toBe(0);
    });

    it("string and number force one type", () => {
        expect(coerceId(1, "string", "id")).toBe("1");
        expect(coerceId(true, "string", "id")).toBe("true");
        expect(coerceId(null, "string", "id")).toBe("null");
        expect(coerceId("s", "string", "id")).toBe("s");
        expect(code(() => coerceId({}, "string", "id"))).toBe("E_INVALID_ID");
        expect(coerceId("01", "number", "id")).toBe(1);
        expect(coerceId(3, "number", "id")).toBe(3);
        expect(code(() => coerceId(true, "number", "id"))).toBe("E_INVALID_ID");
        // "abc" becomes NaN, which the builder refuses
        expect(code(() => fromRecords({ directed: true, edges: [{ source: "abc", target: 1 }], ids: "number" }))).toBe(
            "E_INVALID_ID",
        );
        const merged = fromRecords({ directed: true, edges: [{ source: "01", target: 1 }], ids: "number" }).snapshot;
        expect(merged.nodeCount).toBe(1);
        expect(merged.selfLoopCount).toBe(1);
        const strings = fromRecords({ directed: true, edges: [{ source: 1, target: "1" }], ids: "string" }).snapshot;
        expect(strings.nodeCount).toBe(1);
        expect(strings.ids.kind).toBe("string");
    });

    it("rejects an unknown coercion rule (E_UNSUPPORTED)", () => {
        const err = thrown(() => coerceId(1, "upper" as unknown as "keep", "id"));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.field).toBe("ids");
    });
});

describe("fromRecords: column modes", () => {
    const input: RecordsInput = {
        directed: true,
        nodes: [
            { id: "a", n: 1, s: "x", nested: { k: 1 } },
            { id: "b", n: "two", s: "y" },
        ],
        edges: [{ source: "a", target: "b", n: 2.5, flag: true }],
    };

    it("infer widens per column and reports the widenings", () => {
        const { snapshot, report } = fromRecords(input);
        assertInvariants(snapshot);
        expect(snapshot.nodes.requireTyped("n", "string").decodeAll()).toEqual(["1", "two"]);
        expect(snapshot.nodes.requireTyped("nested", "json").value(0)).toEqual({ k: 1 });
        expect(snapshot.edges.requireTyped("n", "f64").value(0)).toBe(2.5);
        expect(snapshot.edges.requireTyped("flag", "bool").value(0)).toBe(true);
        expect(report.widened).toEqual([{ column: "n", domain: "node", from: "i32", to: "string" }]);
    });

    it("json stores every key as a json column verbatim", () => {
        const { snapshot, report } = fromRecords({ ...input, columns: "json" });
        assertInvariants(snapshot);
        for (const column of snapshot.nodes) {
            expect(column.dtype).toBe("json");
        }
        expect(snapshot.nodes.requireTyped("n", "json").value(0)).toBe(1);
        expect(snapshot.nodes.requireTyped("n", "json").value(1)).toBe("two");
        expect(snapshot.nodes.requireTyped("s", "json").isSet(1)).toBe(true);
        expect(snapshot.nodes.requireTyped("nested", "json").isSet(1)).toBe(false);
        expect(snapshot.edges.requireTyped("flag", "json").value(0)).toBe(true);
        expect(report.widened).toEqual([]);
    });

    it("none loads the structure and the weight only", () => {
        const { snapshot } = fromRecords({
            ...input,
            edges: [{ source: "a", target: "b", weight: 3, n: 2.5 }],
            columns: "none",
        });
        assertInvariants(snapshot);
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.edges.names().filter((name) => !name.startsWith("graphty."))).toEqual([]);
        expect(snapshot.weights?.[0]).toBe(3);
    });

    it("a declaration list stores only the declared keys with their dtypes", () => {
        const { snapshot } = fromRecords({
            ...input,
            nodes: [
                { id: "a", n: 1, s: "x", nested: { k: 1 } },
                { id: "b", s: "y" },
            ],
            columns: [
                { name: "n", dtype: "f64", role: "size" },
                { name: "flag", dtype: "bool" },
                { name: "missing", dtype: "string" },
            ],
        });
        assertInvariants(snapshot);
        expect(snapshot.nodes.names()).toEqual(["n"]);
        expect(snapshot.nodes.byRole("size")?.dtype).toBe("f64");
        expect(snapshot.nodes.requireTyped("n", "f64").value(0)).toBe(1);
        expect(snapshot.nodes.requireTyped("n", "f64").isSet(1)).toBe(false);
        expect(snapshot.edges.names()).toEqual(["n", "flag"]);
        expect(snapshot.edges.requireTyped("flag", "bool").value(0)).toBe(true);
        // a declared column never widens: a value its dtype cannot hold is refused at the cell
        const err = thrown(() => fromRecords({ ...input, columns: [{ name: "n", dtype: "i32" }] }));
        expect(err.code).toBe("E_COLUMN_TYPE");
        expect(err.details).toMatchObject({ column: "n", row: 1 });
    });

    it("rejects a bad declaration list", () => {
        const twice = thrown(() =>
            fromRecords({
                ...input,
                columns: [
                    { name: "n", dtype: "f64" },
                    { name: "n", dtype: "string" },
                ],
            }),
        );
        expect(twice.code).toBe("E_COLUMN_EXISTS");
        const unnamed = thrown(() => fromRecords({ ...input, columns: [{ name: "", dtype: "f64" }] }));
        expect(unnamed.code).toBe("E_COLUMN_TYPE");
        const mode = thrown(() => fromRecords({ ...input, columns: "all" as unknown as "infer" }));
        expect(mode.code).toBe("E_UNSUPPORTED");
        expect(mode.details.field).toBe("columns");
    });
});

describe("fromRecords: options", () => {
    const input: RecordsInput = {
        directed: false,
        edges: [
            { source: "a", target: "b", weight: 1 },
            { source: "b", target: "a", weight: 2 },
            { source: "a", target: "a" },
        ],
    };

    it("applies the builder policies", () => {
        expect(code(() => fromRecords(input, { duplicateEdges: "error" }))).toBe("E_DUPLICATE_EDGE");
        expect(code(() => fromRecords(input, { selfLoops: "error" }))).toBe("E_SELF_LOOP");
        const merged = fromRecords(input, { duplicateEdges: "sum", selfLoops: "drop" });
        assertInvariants(merged.snapshot);
        expect(merged.snapshot.edgeCount).toBe(1);
        expect(merged.snapshot.weights?.[0]).toBe(3);
        expect(merged.report.mergedEdges).toBe(1);
        expect(merged.report.droppedSelfLoops).toBe(1);
        expect(merged.report.edgeRemap).not.toBeNull();
        expect(Array.from(merged.report.edgeRemap as Uint32Array)).toEqual([0, 0, INVALID_INDEX]);
        const unknown = thrown(() =>
            fromRecords(
                { directed: true, nodes: [{ id: "a" }], edges: [{ source: "a", target: "zz" }] },
                { addMissingNodes: false },
            ),
        );
        expect(unknown.code).toBe("E_UNKNOWN_NODE");
        expect(fromRecords(input, { weighted: true }).snapshot.weights).not.toBeNull();
        expect(
            fromRecords({ directed: true, edges: [{ source: 1, target: 2 }] }, { weighted: true }).snapshot.weights,
        ).not.toBeNull();
    });

    it("applies the freeze options", () => {
        const { snapshot } = fromRecords(input, { label: "recs", prepare: ["degree"], checksum: true, arena: false });
        expect(snapshot.label).toBe("recs");
        expect(snapshot.cachedViews()).toContain("degree");
        expect(snapshot.arena).toBeNull();
        snapshot.validate({ checksum: true, level: "full" });
        // the checksum records cover the prepared views: a mutated one is caught
        const degree = snapshot.degree();
        degree[0] = degree[0] + 1;
        expect(() => snapshot.validate({ checksum: true })).toThrow(GraphFormatError);
    });

    it("rejects an illegal id in a record (E_INVALID_ID)", () => {
        expect(code(() => fromRecords({ directed: true, edges: [{ source: Infinity, target: 1 }] }))).toBe(
            "E_INVALID_ID",
        );
        expect(code(() => fromRecords({ directed: true, nodes: [{ id: "\ud800" }], edges: [] }))).toBe("E_INVALID_ID");
    });
});
