/**
 * nodeMass and weight resolution by graph-format ROLE and name (spec 7.14, D28; contract 3.13 inputs.ts): the
 * role-"mass" column in any numeric dtype, the outDegree + 1 default, the explicit F32, the named column, the
 * Record rejection with the hint; weights from the arc array, from a named edge column through expandEdges (f32,
 * f64, u32) and the errors, including the E_GPU_INELIGIBLE pass-through of a string column.
 */

import type { GraphSnapshot } from "@graphty/graph-format";

import { isWebGpuGraphError } from "../../src/errors.js";
import { resolveNodeMass, resolveWeights } from "../../src/layouts/inputs.js";
import { type EdgeSpec, pathEdges, snapshotOf } from "../helpers/graphs.js";

const TRIANGLE: readonly EdgeSpec[] = [
    [0, 1, 0.5],
    [1, 2, 2],
    [0, 2, 3],
];

function errorOf(fn: () => void): { code: string; details: Readonly<Record<string, unknown>> } {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected the call to throw");
}

function formatErrorOf(fn: () => void): { name: string; code: string } {
    try {
        fn();
    } catch (err) {
        const e = err as { name?: unknown; code?: unknown };
        return { name: String(e.name), code: String(e.code) };
    }
    throw new Error("expected the call to throw");
}

describe("resolveNodeMass", () => {
    let path4: GraphSnapshot;

    beforeEach(() => {
        path4 = snapshotOf(pathEdges(4));
    });

    it("null without a role column -> outDegree + 1", () => {
        const mass = resolveNodeMass(path4, null);
        expect(mass).toBeInstanceOf(Float32Array);
        expect(Array.from(mass)).toEqual([2, 3, 3, 2]);
        expect(Array.from(resolveNodeMass(path4, undefined))).toEqual([2, 3, 3, 2]);
    });

    it("null with a role-mass column: f32 as is, f64 through the cached f32 copy, u32 converted", () => {
        const f32 = new Float32Array([1.5, 2.5, 3.5, 4.5]);
        path4.nodes.set("m", f32, { role: "mass" });
        expect(resolveNodeMass(path4, null)).toBe(f32);

        path4.nodes.set("m64", new Float64Array([1, 2, 3, 4]), { role: "mass" }, { replaceRole: true });
        const fromF64 = resolveNodeMass(path4, null);
        expect(fromF64).toBeInstanceOf(Float32Array);
        expect(Array.from(fromF64)).toEqual([1, 2, 3, 4]);

        path4.nodes.set("mu", new Uint32Array([7, 8, 9, 10]), { role: "mass" }, { replaceRole: true });
        const fromU32 = resolveNodeMass(path4, null);
        expect(fromU32).toBeInstanceOf(Float32Array);
        expect(Array.from(fromU32)).toEqual([7, 8, 9, 10]);

        path4.nodes.set("mu8", new Uint8Array([1, 2, 3, 4]), { role: "mass" }, { replaceRole: true });
        expect(Array.from(resolveNodeMass(path4, null))).toEqual([1, 2, 3, 4]);
    });

    it("an F32 of length n is returned as is", () => {
        const given = new Float32Array([1, 1, 1, 1]);
        expect(resolveNodeMass(path4, given)).toBe(given);
    });

    it("a column name resolves through nodes.get (any numeric dtype)", () => {
        path4.nodes.set("weightish", new Float64Array([4, 3, 2, 1]));
        expect(Array.from(resolveNodeMass(path4, "weightish"))).toEqual([4, 3, 2, 1]);
    });

    it("a Record -> E_UNSUPPORTED { option: 'nodeMass', hint }", () => {
        const e = errorOf(() => resolveNodeMass(path4, { 0: 1, 1: 2, 2: 3, 3: 4 }));
        expect(e.code).toBe("E_UNSUPPORTED");
        expect(e.details.option).toBe("nodeMass");
        expect(String(e.details.hint)).toContain("role 'mass'");
        expect(String(e.details.hint)).toContain("Float32Array");
        expect(String(e.details.hint)).toContain("resolveNodeVector");
    });

    it("wrong length, missing column, non-numeric column, non-positive / NaN mass -> E_INVALID_ARGUMENT", () => {
        expect(errorOf(() => resolveNodeMass(path4, new Float32Array(3)))).toEqual({
            code: "E_INVALID_ARGUMENT",
            details: { argument: "nodeMass", value: 3, expected: "4 values" },
        });
        const missing = errorOf(() => resolveNodeMass(path4, "absent"));
        expect(missing.code).toBe("E_INVALID_ARGUMENT");
        expect(missing.details.argument).toBe("nodeMass");
        expect(missing.details.value).toBe("absent");

        path4.nodes.set("label", ["a", "b", "c", "d"]);
        const nonNumeric = errorOf(() => resolveNodeMass(path4, "label"));
        expect(nonNumeric.code).toBe("E_INVALID_ARGUMENT");
        expect(nonNumeric.details.argument).toBe("nodeMass");

        path4.nodes.set("flag", [true, false, true, false]);
        expect(errorOf(() => resolveNodeMass(path4, "flag")).code).toBe("E_INVALID_ARGUMENT");

        expect(errorOf(() => resolveNodeMass(path4, new Float32Array([1, 0, 1, 1]))).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => resolveNodeMass(path4, new Float32Array([1, -2, 1, 1]))).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => resolveNodeMass(path4, new Float32Array([1, Number.NaN, 1, 1]))).code).toBe(
            "E_INVALID_ARGUMENT",
        );
        path4.nodes.set("badmass", new Float32Array([1, 2, 0, 4]), { role: "mass" });
        expect(errorOf(() => resolveNodeMass(path4, null)).code).toBe("E_INVALID_ARGUMENT");

        // a typed array that is not a Float32Array is an argument error, not a Record
        const f64 = new Float64Array([1, 2, 3, 4]) as unknown as Readonly<Record<string, number>>;
        expect(errorOf(() => resolveNodeMass(path4, f64)).code).toBe("E_INVALID_ARGUMENT");
    });

    it("a multi-component column is rejected", () => {
        path4.nodes.set("pos", new Float32Array(12), { components: 3 });
        expect(errorOf(() => resolveNodeMass(path4, "pos")).code).toBe("E_INVALID_ARGUMENT");
    });
});

describe("resolveWeights", () => {
    it("true on a weighted snapshot -> the arc array itself, source 'arcs'", () => {
        const s = snapshotOf(TRIANGLE);
        const r = resolveWeights(s, true);
        expect(r.source).toBe("arcs");
        expect(r.column).toBeNull();
        expect(r.data).toBe(s.weights);
        expect(Array.from(r.data ?? [])).toEqual([0.5, 3, 0.5, 2, 3, 2]);
    });

    it("true on an unweighted snapshot -> data null (attract with 1.0), source 'arcs'", () => {
        const s = snapshotOf(pathEdges(3));
        expect(s.weights).toBeNull();
        expect(resolveWeights(s, true)).toEqual({ data: null, source: "arcs", column: null });
    });

    it("false / null / undefined -> none", () => {
        const s = snapshotOf(TRIANGLE);
        for (const spec of [false, null, undefined]) {
            expect(resolveWeights(s, spec)).toEqual({ data: null, source: "none", column: null });
        }
    });

    it("an f32 edge column expands through expandEdges (both arcs of an edge carry its value)", () => {
        const s = snapshotOf(pathEdges(3));
        s.edges.set("w", new Float32Array([10, 20]));
        const r = resolveWeights(s, "w");
        expect(r.source).toBe("column");
        expect(r.column?.meta.name).toBe("w");
        expect(r.data).toBeInstanceOf(Float32Array);
        // arcs: 0->1 (e0), 1->0 (e0), 1->2 (e1), 2->1 (e1)
        expect(Array.from(r.data ?? [])).toEqual([10, 10, 20, 20]);
        expect(r.data?.length).toBe(s.arcCount);
    });

    it("the triangle's edge column expands in arc order [10, 30, 10, 20, 30, 20]", () => {
        const s = snapshotOf(TRIANGLE);
        s.edges.set("w", new Float32Array([10, 20, 30]));
        expect(Array.from(resolveWeights(s, "w").data ?? [])).toEqual([10, 30, 10, 20, 30, 20]);
        const perEdge = [10, 20, 30];
        const data = resolveWeights(s, "w").data ?? new Float32Array(0);
        for (let a = 0; a < s.arcCount; a++) {
            expect(data[a]).toBe(perEdge[s.arcToEdge[a]]);
        }
    });

    it("an f64 column arrives as f32 values, a u32 column is converted", () => {
        const s = snapshotOf(TRIANGLE);
        s.edges.set("w64", new Float64Array([10, 20, 30]));
        const f64 = resolveWeights(s, "w64");
        expect(f64.data).toBeInstanceOf(Float32Array);
        expect(Array.from(f64.data ?? [])).toEqual([10, 30, 10, 20, 30, 20]);
        s.edges.set("wu", new Uint32Array([1, 2, 3]));
        const u32 = resolveWeights(s, "wu");
        expect(u32.data).toBeInstanceOf(Float32Array);
        expect(Array.from(u32.data ?? [])).toEqual([1, 3, 1, 2, 3, 2]);
    });

    it("a missing column, a packed dtype, a multi-component column or a bad spec -> E_INVALID_ARGUMENT", () => {
        const s = snapshotOf(TRIANGLE);
        const missing = errorOf(() => resolveWeights(s, "nope"));
        expect(missing.code).toBe("E_INVALID_ARGUMENT");
        expect(missing.details.argument).toBe("weight");
        expect(missing.details.value).toBe("nope");

        s.edges.set("u8", new Uint8Array([1, 2, 3]));
        expect(errorOf(() => resolveWeights(s, "u8")).code).toBe("E_INVALID_ARGUMENT");
        s.edges.set("b", [true, false, true]);
        expect(errorOf(() => resolveWeights(s, "b")).code).toBe("E_INVALID_ARGUMENT");
        s.edges.set("v3", new Float32Array(9), { components: 3 });
        expect(errorOf(() => resolveWeights(s, "v3")).code).toBe("E_INVALID_ARGUMENT");
        expect(errorOf(() => resolveWeights(s, 3 as unknown as string)).code).toBe("E_INVALID_ARGUMENT");
    });

    it("a string column passes gpuView's E_GPU_INELIGIBLE through unchanged (D12)", () => {
        const s = snapshotOf(TRIANGLE);
        s.edges.set("name", ["a", "b", "c"]);
        expect(formatErrorOf(() => resolveWeights(s, "name"))).toEqual({
            name: "GraphFormatError",
            code: "E_GPU_INELIGIBLE",
        });
    });
});
