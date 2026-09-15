/**
 * Unit tests of fromEdgeArrays (design section 8.1, entry point 1): round trips against the builder
 * and the naive CSR, the identity id map without ids, id-map kinds with ids, F32 / F64 weights and
 * the f64 shadow, node / edge columns (typed arrays, ColumnInput, JS arrays) including under a
 * merging duplicate policy, meta, the builder policies and freeze options mirrored through the
 * options object, and every rejection code.
 */

import { describe, expect, it } from "vitest";

import { WEIGHT_COLUMN_NAME } from "../../src/builder/freeze.js";
import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { fromEdgeArrays, splitOptions } from "../../src/populate/from-edge-arrays.js";
import { type EdgeArraysInput, type F32, type GraphSnapshot, type U32 } from "../../src/types/index.js";
import { assertInvariants, assertMatchesSpec } from "../helpers/invariants.js";
import { type EdgeSpec, type GraphSpec, gridEdges, KARATE_EDGES } from "../helpers/parts.js";

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

/** The COO arrays of a spec. */
function cooOf(spec: GraphSpec): { src: U32; dst: U32; weights: F32 | null } {
    const src = new Uint32Array(spec.edges.map((e) => e[0]));
    const dst = new Uint32Array(spec.edges.map((e) => e[1]));
    const weighted = spec.weighted ?? spec.edges.some((e) => e.length === 3);
    const weights = weighted ? new Float32Array(spec.edges.map((e) => (e.length === 3 ? e[2] : 1))) : null;
    return { src, dst, weights };
}

function inputOf(spec: GraphSpec, nodeCount?: number): EdgeArraysInput {
    const coo = cooOf(spec);
    return {
        directed: spec.directed,
        nodeCount: nodeCount ?? spec.nodeCount ?? Math.max(-1, ...spec.edges.flatMap((e) => [e[0], e[1]])) + 1,
        src: coo.src,
        dst: coo.dst,
        weights: coo.weights ?? undefined,
    };
}

/** Byte-for-byte comparison of two snapshots' core arrays, flags and counts. */
function expectSameCore(a: GraphSnapshot, b: GraphSnapshot): void {
    expect(a.directed).toBe(b.directed);
    expect(a.nodeCount).toBe(b.nodeCount);
    expect(a.edgeCount).toBe(b.edgeCount);
    expect(a.arcCount).toBe(b.arcCount);
    expect(a.selfLoopCount).toBe(b.selfLoopCount);
    expect(Array.from(a.rowPtr)).toEqual(Array.from(b.rowPtr));
    expect(Array.from(a.colIdx)).toEqual(Array.from(b.colIdx));
    expect(Array.from(a.arcToEdge)).toEqual(Array.from(b.arcToEdge));
    expect(Array.from(a.edgeToArc)).toEqual(Array.from(b.edgeToArc));
    expect(a.weights === null).toBe(b.weights === null);
    if (a.weights !== null && b.weights !== null) {
        expect(Array.from(a.weights)).toEqual(Array.from(b.weights));
    }
    expect(a.flags).toEqual(b.flags);
    expect(a.contentHash()).toBe(b.contentHash());
}

const WEIGHTED_EDGES: readonly EdgeSpec[] = [
    [0, 1, 2],
    [1, 2, 0.5],
    [2, 0, 3],
    [0, 0, 1],
    [1, 2, 4],
    [3, 1, -1],
];

describe("fromEdgeArrays: round trips", () => {
    const specs: readonly [string, GraphSpec][] = [
        ["directed karate", { directed: true, edges: KARATE_EDGES }],
        ["undirected karate", { directed: false, edges: KARATE_EDGES }],
        ["directed weighted with loop and parallels", { directed: true, edges: WEIGHTED_EDGES }],
        ["undirected weighted with loop and parallels", { directed: false, edges: WEIGHTED_EDGES }],
        ["directed grid", { directed: true, edges: gridEdges(4, 3) }],
        ["undirected grid with isolates", { directed: false, nodeCount: 20, edges: gridEdges(4, 3) }],
        ["empty directed", { directed: true, nodeCount: 0, edges: [] }],
        ["nodes only undirected", { directed: false, nodeCount: 5, edges: [] }],
    ];

    it.each(specs)("matches the naive CSR and the builder: %s", (_name, spec) => {
        const snapshot = fromEdgeArrays(inputOf(spec));
        assertMatchesSpec(snapshot, spec);
        const builder = new GraphBuilder({ directed: spec.directed });
        builder.addAnonymousNodes(snapshot.nodeCount);
        const coo = cooOf(spec);
        builder.addEdges(coo.src, coo.dst, coo.weights ?? undefined);
        expectSameCore(snapshot, builder.freeze());
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(0);
        expect(snapshot.ids.byteLength()).toBe(0);
        expect(snapshot.arena).not.toBeNull();
    });

    it("preserves isolates from nodeCount and never sorts nodes or edges", () => {
        const snapshot = fromEdgeArrays({
            directed: true,
            nodeCount: 6,
            src: new Uint32Array([4, 0, 4]),
            dst: new Uint32Array([1, 5, 0]),
        });
        assertInvariants(snapshot);
        expect(snapshot.nodeCount).toBe(6);
        expect(snapshot.edgeCount).toBe(3);
        expect(snapshot.edgeSource(0)).toBe(4);
        expect(snapshot.edgeTarget(0)).toBe(1);
        expect(snapshot.edgeSource(1)).toBe(0);
        expect(snapshot.edgeTarget(2)).toBe(0);
        expect(snapshot.outDegreeOf(2)).toBe(0);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
    });

    it("takes the identity fast path for a source-grouped, target-sorted directed input", () => {
        const snapshot = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0, 0, 1, 2]),
            dst: new Uint32Array([1, 2, 2, 0]),
            weights: new Float32Array([1, 2, 3, 4]),
        });
        assertInvariants(snapshot);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(true);
        expect(snapshot.arena?.segments.arcToEdge).toBeNull();
        expect(snapshot.edgeList().weights).toBe(snapshot.weights);
    });

    it("is deterministic (I15)", () => {
        const spec: GraphSpec = { directed: false, edges: WEIGHTED_EDGES };
        const a = fromEdgeArrays(inputOf(spec), { label: "a" });
        const b = fromEdgeArrays(inputOf(spec), { label: "b" });
        expectSameCore(a, b);
        expect(a.label).toBe("a");
        expect(b.label).toBe("b");
    });

    it("does not alias the caller's arrays (I18) and leaves them untouched", () => {
        const src = new Uint32Array([2, 1, 0]);
        const dst = new Uint32Array([0, 0, 2]);
        const weights = new Float32Array([5, 6, 7]);
        const snapshot = fromEdgeArrays({ directed: true, nodeCount: 3, src, dst, weights });
        expect(snapshot.colIdx).not.toBe(dst);
        expect(snapshot.weights).not.toBe(weights);
        expect(Array.from(src)).toEqual([2, 1, 0]);
        expect(Array.from(dst)).toEqual([0, 0, 2]);
        expect(Array.from(weights)).toEqual([5, 6, 7]);
    });
});

describe("fromEdgeArrays: ids", () => {
    it("derives nodeCount from ids and builds a string id map", () => {
        const snapshot = fromEdgeArrays({
            directed: false,
            ids: ["a", "b", "c", "d"],
            src: new Uint32Array([0, 1]),
            dst: new Uint32Array([1, 2]),
        });
        assertInvariants(snapshot);
        expect(snapshot.nodeCount).toBe(4);
        expect(snapshot.ids.kind).toBe("string");
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d"]);
        expect(snapshot.ids.indexOf("d")).toBe(3);
        expect(snapshot.ids.indexOf("e")).toBe(INVALID_INDEX);
    });

    it("detects identity with offset 1 for 1-based F64 ids", () => {
        const snapshot = fromEdgeArrays({
            directed: true,
            ids: new Float64Array([1, 2, 3]),
            src: new Uint32Array([0, 1]),
            dst: new Uint32Array([1, 2]),
        });
        assertInvariants(snapshot);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(1);
        expect(snapshot.ids.idOf(2)).toBe(3);
        expect(snapshot.ids.indexOf(1)).toBe(0);
    });

    it("detects dense, numeric and mixed kinds", () => {
        const edges = { src: new Uint32Array([0, 1]), dst: new Uint32Array([1, 2]) };
        expect(fromEdgeArrays({ directed: true, ids: [0, 2, 4], ...edges }).ids.kind).toBe("dense");
        expect(fromEdgeArrays({ directed: true, ids: [0.5, 100, 7], ...edges }).ids.kind).toBe("numeric");
        expect(fromEdgeArrays({ directed: true, ids: [0, "x", 2], ...edges }).ids.kind).toBe("mixed");
        expect(fromEdgeArrays({ directed: true, ids: new Float64Array([10, 20, 1e9]), ...edges }).ids.kind).toBe(
            "numeric",
        );
    });

    it("accepts a matching explicit nodeCount next to ids", () => {
        const snapshot = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            ids: ["p", "q"],
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        });
        expect(snapshot.nodeCount).toBe(2);
    });

    it("rejects a nodeCount that disagrees with ids (E_COLUMN_LENGTH)", () => {
        const err = thrown(() =>
            fromEdgeArrays({
                directed: true,
                nodeCount: 3,
                ids: ["p", "q"],
                src: new Uint32Array([0]),
                dst: new Uint32Array([1]),
            }),
        );
        expect(err.code).toBe("E_COLUMN_LENGTH");
        expect(err.details.field).toBe("ids");
    });

    it("rejects repeated ids (E_DUPLICATE_ID) and illegal ids (E_INVALID_ID)", () => {
        const edges = { src: new Uint32Array([0]), dst: new Uint32Array([1]) };
        const dup = thrown(() => fromEdgeArrays({ directed: true, ids: ["a", "b", "a"], ...edges }));
        expect(dup.code).toBe("E_DUPLICATE_ID");
        expect(dup.details).toMatchObject({ id: "a", index: 0, position: 2 });
        expect(code(() => fromEdgeArrays({ directed: true, ids: ["a", Number.NaN], ...edges }))).toBe("E_INVALID_ID");
        expect(code(() => fromEdgeArrays({ directed: true, ids: new Float64Array([1, Infinity]), ...edges }))).toBe(
            "E_INVALID_ID",
        );
    });
});

describe("fromEdgeArrays: weights", () => {
    it("keeps F32 weights exactly and computes the weight flags", () => {
        const snapshot = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0, 1, 2]),
            dst: new Uint32Array([1, 2, 0]),
            weights: new Float32Array([0.5, -1, Infinity]),
        });
        assertInvariants(snapshot);
        expect(snapshot.flags).toMatchObject({
            weighted: true,
            allWeightsOne: false,
            nonNegativeWeights: false,
            finiteWeights: false,
        });
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual([0.5, -1, Infinity]);
        expect(snapshot.edges.byRole("weight")).toBeNull();
    });

    it("downcasts F64 weights to f32 and keeps an f64 shadow only when a value is not f32-exact", () => {
        const exact = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0, 1]),
            dst: new Uint32Array([1, 0]),
            weights: new Float64Array([2, 0.5]),
        });
        assertInvariants(exact);
        expect(exact.weights).toBeInstanceOf(Float32Array);
        expect(exact.edges.byRole("weight")).toBeNull();

        const inexact = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0, 1]),
            dst: new Uint32Array([1, 0]),
            weights: new Float64Array([0.1, 16777217]),
        });
        assertInvariants(inexact);
        expect(inexact.weights?.[inexact.edgeToArc[0]]).toBe(Math.fround(0.1));
        const shadow = inexact.edges.byRole("weight");
        expect(shadow).not.toBeNull();
        expect(shadow?.meta.name).toBe(WEIGHT_COLUMN_NAME);
        expect(shadow?.dtype).toBe("f64");
        expect(shadow?.value(0)).toBe(0.1);
        expect(shadow?.value(1)).toBe(16777217);
    });

    it("honours an explicit weightDtype over the array type", () => {
        const input: EdgeArraysInput = {
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
            weights: new Float64Array([0.1]),
        };
        expect(fromEdgeArrays(input, { weightDtype: "f32" }).edges.byRole("weight")).toBeNull();
        const f32Input: EdgeArraysInput = { ...input, weights: new Float32Array([0.1]) };
        expect(fromEdgeArrays(f32Input, { weightDtype: "f64" }).edges.byRole("weight")).toBeNull();
    });

    it("allocates an all-ones array under weighted: true and refuses weights under weighted: false", () => {
        const input: EdgeArraysInput = {
            directed: false,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        };
        const declared = fromEdgeArrays(input, { weighted: true });
        expect(declared.weights).not.toBeNull();
        expect(Array.from(declared.weights as Float32Array)).toEqual([1, 1]);
        expect(declared.flags.allWeightsOne).toBe(true);
        const ones = fromEdgeArrays({ ...input, weights: new Float32Array([1]) }, { weighted: false });
        expect(ones.weights).toBeNull();
        const err = thrown(() => fromEdgeArrays({ ...input, weights: new Float32Array([7]) }, { weighted: false }));
        expect(err.code).toBe("E_INVALID_WEIGHT");
        expect(err.details.reason).toBe("unweighted builder");
        expect(fromEdgeArrays(input).weights).toBeNull();
    });

    it("rejects NaN weights (E_INVALID_WEIGHT) and a wrong weights length (E_COLUMN_LENGTH)", () => {
        const base = { directed: true, nodeCount: 2, src: new Uint32Array([0, 1]), dst: new Uint32Array([1, 0]) };
        const nan = thrown(() => fromEdgeArrays({ ...base, weights: new Float32Array([1, Number.NaN]) }));
        expect(nan.code).toBe("E_INVALID_WEIGHT");
        const short = thrown(() => fromEdgeArrays({ ...base, weights: new Float32Array([1]) }));
        expect(short.code).toBe("E_COLUMN_LENGTH");
        expect(short.details).toMatchObject({ field: "weights", expected: 2, found: 1 });
    });
});

describe("fromEdgeArrays: columns and meta", () => {
    const base: EdgeArraysInput = {
        directed: true,
        nodeCount: 3,
        src: new Uint32Array([0, 1, 1]),
        dst: new Uint32Array([1, 2, 2]),
    };

    it("attaches typed-array node and edge columns with their declarations", () => {
        const snapshot = fromEdgeArrays({
            ...base,
            nodeColumns: {
                size: new Float32Array([1, 2, 3]),
                pos: { data: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]), decl: { components: 3, role: "position" } },
                flag: { data: new Uint32Array([0b101]), decl: { dtype: "bool" } },
            },
            edgeColumns: {
                cap: new Uint32Array([10, 20, 30]),
                kind: { data: new Uint32Array([0, 1, 0]), decl: { dtype: "dict", options: ["a", "b"] } },
            },
        });
        assertInvariants(snapshot);
        expect(snapshot.nodes.names()).toEqual(["size", "pos", "flag"]);
        expect(snapshot.nodes.requireTyped("size", "f32").value(2)).toBe(3);
        const pos = snapshot.nodes.byRole("position");
        expect(pos?.meta.components).toBe(3);
        expect(Array.from(pos?.value(1) as ArrayLike<number>)).toEqual([1, 1, 1]);
        expect(snapshot.nodes.requireTyped("flag", "bool").value(2)).toBe(true);
        expect(snapshot.nodes.requireTyped("flag", "bool").value(1)).toBe(false);
        expect(snapshot.edges.requireTyped("cap", "u32").value(2)).toBe(30);
        const kind = snapshot.edges.requireTyped("kind", "dict");
        expect(kind.value(1)).toBe("b");
        expect(kind.dictionary).toEqual(["a", "b"]);
    });

    it("attaches JS-array columns with inference (string, list, json) and declared dtypes", () => {
        const snapshot = fromEdgeArrays({
            ...base,
            nodeColumns: {
                name: { data: ["x", "y", null], decl: {} },
                tags: { data: [["a", "b"], [], ["c"]], decl: {} },
                blob: { data: [{ k: 1 }, [1, 2], "s"], decl: {} },
                score: { data: [1, 2.5, undefined], decl: { dtype: "f64" } },
            },
            edgeColumns: {
                label: { data: ["e0", "e1", "e2"], decl: { dtype: "string", role: "label" } },
            },
        });
        assertInvariants(snapshot);
        const name = snapshot.nodes.requireTyped("name", "string");
        expect(name.value(0)).toBe("x");
        expect(name.isSet(2)).toBe(false);
        expect(name.nullCount).toBe(1);
        const tags = snapshot.nodes.requireTyped("tags", "list");
        expect(tags.child.dtype).toBe("string");
        expect(tags.value(0)).toEqual(["a", "b"]);
        expect(tags.value(1)).toEqual([]);
        expect(snapshot.nodes.requireTyped("blob", "json").value(1)).toEqual([1, 2]);
        const score = snapshot.nodes.requireTyped("score", "f64");
        expect(score.value(1)).toBe(2.5);
        expect(score.isSet(2)).toBe(false);
        expect(snapshot.edges.byRole("label")?.value(2)).toBe("e2");
    });

    it("gathers edge columns through a merging duplicate policy and a dropped self-loop", () => {
        const snapshot = fromEdgeArrays(
            {
                directed: true,
                nodeCount: 3,
                src: new Uint32Array([0, 1, 1, 2]),
                dst: new Uint32Array([1, 2, 2, 2]),
                weights: new Float32Array([1, 2, 3, 4]),
                edgeColumns: {
                    tag: { data: ["a", "b", "c", "d"], decl: {} },
                    n: new Uint32Array([10, 20, 30, 40]),
                },
            },
            { duplicateEdges: "sum", selfLoops: "drop" },
        );
        assertInvariants(snapshot);
        expect(snapshot.edgeCount).toBe(2);
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
        expect(Array.from(snapshot.edgeList().weights as Float32Array)).toEqual([1, 5]);
        expect(snapshot.edges.requireTyped("tag", "string").decodeAll()).toEqual(["a", "b"]);
        expect(Array.from(snapshot.edges.requireTyped("n", "u32").data)).toEqual([10, 20]);
    });

    it("rejects a column of the wrong length (E_COLUMN_LENGTH) and a taken role (E_DUPLICATE_ROLE)", () => {
        const short = thrown(() => fromEdgeArrays({ ...base, nodeColumns: { x: new Float32Array([1, 2]) } }));
        expect(short.code).toBe("E_COLUMN_LENGTH");
        const shortList = thrown(() => fromEdgeArrays({ ...base, edgeColumns: { x: { data: ["a"], decl: {} } } }));
        expect(shortList.code).toBe("E_COLUMN_LENGTH");
        const role = thrown(() =>
            fromEdgeArrays({
                ...base,
                nodeColumns: {
                    a: { data: new Float32Array(3), decl: { role: "size" } },
                    b: { data: new Float32Array(3), decl: { role: "size" } },
                },
            }),
        );
        expect(role.code).toBe("E_DUPLICATE_ROLE");
    });

    it("applies meta and rejects a non-JSON extra (E_COLUMN_TYPE)", () => {
        const snapshot = fromEdgeArrays({
            ...base,
            meta: { name: "g", keywords: ["k"], extra: { source: "test" }, declaredMultigraph: false },
        });
        expect(snapshot.meta.name).toBe("g");
        expect(snapshot.meta.keywords).toEqual(["k"]);
        expect(snapshot.meta.extra).toEqual({ source: "test" });
        expect(snapshot.meta.declaredMultigraph).toBe(false);
        expect(snapshot.meta.creator).toBeNull();
        const err = thrown(() => fromEdgeArrays({ ...base, meta: { extra: { f: () => 1 } } }));
        expect(err.code).toBe("E_COLUMN_TYPE");
    });
});

describe("fromEdgeArrays: options", () => {
    const parallel: EdgeArraysInput = {
        directed: false,
        nodeCount: 2,
        src: new Uint32Array([0, 1, 0]),
        dst: new Uint32Array([1, 0, 0]),
        weights: new Float32Array([1, 2, 3]),
    };

    it("splits builder and freeze fields and takes directed from the input", () => {
        // an options.directed that disagrees with the input is refused rather than ignored
        const err = thrown(() => splitOptions(false, { directed: true }));
        expect(err.code).toBe("E_DIRECTED");
        expect(err.details).toMatchObject({ reason: "direction mismatch", directed: false, found: true });
        const split = splitOptions(false, {
            directed: false,
            weighted: true,
            duplicateEdges: "max",
            label: "L",
            arena: false,
            checksum: true,
        });
        expect(split.builder.directed).toBe(false);
        expect(split.builder.weighted).toBe(true);
        expect(split.builder.duplicateEdges).toBe("max");
        expect(split.freeze.label).toBe("L");
        expect(split.freeze.arena).toBe(false);
        expect(split.freeze.checksum).toBe(true);
        expect(Object.keys(split.freeze)).not.toContain("duplicateEdges");
    });

    it("applies duplicateEdges: error / keep / merge policies", () => {
        const err = thrown(() => fromEdgeArrays(parallel, { duplicateEdges: "error" }));
        expect(err.code).toBe("E_DUPLICATE_EDGE");
        expect(err.details).toMatchObject({ source: 0, target: 1, edges: [0, 1] });
        const kept = fromEdgeArrays(parallel);
        expect(kept.flags.multigraph).toBe(true);
        expect(kept.edgeCount).toBe(3);
        const merged = fromEdgeArrays(parallel, { duplicateEdges: "max" });
        assertInvariants(merged);
        expect(merged.edgeCount).toBe(2);
        expect(merged.weights?.[merged.edgeToArc[0]]).toBe(2);
    });

    it("applies selfLoops: error / drop", () => {
        const err = thrown(() => fromEdgeArrays(parallel, { selfLoops: "error" }));
        expect(err.code).toBe("E_SELF_LOOP");
        expect(err.details).toMatchObject({ edge: 2, node: 0 });
        const dropped = fromEdgeArrays(parallel, { selfLoops: "drop" });
        expect(dropped.selfLoopCount).toBe(0);
        expect(dropped.edgeCount).toBe(2);
    });

    it("passes label, prepare, arena, checksum and profile to the freeze", () => {
        const snapshot = fromEdgeArrays(parallel, {
            label: "tagged",
            prepare: ["outDegree", "coo"],
            checksum: true,
        });
        expect(snapshot.label).toBe("tagged");
        expect(snapshot.cachedViews()).toEqual(expect.arrayContaining(["outDegree", "coo"]));
        expect(() => {
            snapshot.validate({ checksum: true });
        }).not.toThrow();
        const separate = fromEdgeArrays(parallel, { arena: false });
        expect(separate.arena).toBeNull();
        assertInvariants(separate);
        expect(() => {
            separate.validate({ checksum: true });
        }).toThrow(GraphFormatError);
    });

    it("rejects an unsupported policy value (E_UNSUPPORTED)", () => {
        const err = thrown(() => fromEdgeArrays(parallel, { duplicateEdges: "bogus" as unknown as "keep" }));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.field).toBe("duplicateEdges");
    });
});

describe("fromEdgeArrays: structural rejections", () => {
    it("requires nodeCount without ids (E_INDEX_RANGE) and a non-negative integer nodeCount", () => {
        const edges = { src: new Uint32Array([0]), dst: new Uint32Array([1]) };
        const missing = thrown(() => fromEdgeArrays({ directed: true, ...edges }));
        expect(missing.code).toBe("E_INDEX_RANGE");
        expect(missing.details.field).toBe("nodeCount");
        expect(code(() => fromEdgeArrays({ directed: true, nodeCount: -1, ...edges }))).toBe("E_INDEX_RANGE");
        expect(code(() => fromEdgeArrays({ directed: true, nodeCount: 1.5, ...edges }))).toBe("E_INDEX_RANGE");
    });

    it("rejects src / dst of different lengths (E_COLUMN_LENGTH)", () => {
        const err = thrown(() =>
            fromEdgeArrays({ directed: true, nodeCount: 2, src: new Uint32Array([0, 1]), dst: new Uint32Array([1]) }),
        );
        expect(err.code).toBe("E_COLUMN_LENGTH");
        expect(err.details).toMatchObject({ field: "dst", expected: 2, found: 1 });
    });

    it("rejects an endpoint at or above nodeCount (E_UNKNOWN_NODE with details.index)", () => {
        const err = thrown(() =>
            fromEdgeArrays({ directed: true, nodeCount: 2, src: new Uint32Array([0]), dst: new Uint32Array([2]) }),
        );
        expect(err.code).toBe("E_UNKNOWN_NODE");
        expect(err.details.index).toBe(2);
    });
});
