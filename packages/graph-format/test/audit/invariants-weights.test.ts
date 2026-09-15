/**
 * Adversarial audit of the weight rules (design sections 3.7, 3.8, 6.3 step 8, 6.5 and invariants
 * I8 / I9): f32 rounding of the arc array, the f64 shadow rule, the explicit / omitted validity
 * record, the reducer arithmetic of the duplicate policies and the NaN re-check of freeze step 8.
 *
 * Tests marked "PINS DEFECT" are expected to FAIL against the current implementation; each names
 * the finding it pins.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { GraphFormatError } from "../../src/errors.js";
import { type GraphSnapshot } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { ModelError, ModelGraph } from "../helpers/model-graph.js";

const RUNS = Number(process.env.FC_RUNS ?? "150");

/** A weight pool that exercises rounding, overflow, underflow, signed zero and the infinities. */
const arbWeight: fc.Arbitrary<number> = fc.oneof(
    fc.constant(1),
    fc.constant(0),
    fc.constant(-0),
    fc.constant(-1),
    fc.constant(0.1),
    fc.constant(2.5),
    fc.constant(16777217),
    fc.constant(1 + 2 ** -30),
    fc.constant(-1e-50),
    fc.constant(1e39),
    fc.constant(3e38),
    fc.constant(Infinity),
    fc.constant(-Infinity),
    fc.double({ noNaN: true, noDefaultInfinity: true }),
    fc.float({ noNaN: true }),
);

interface WeightedEdge {
    readonly u: number;
    readonly v: number;
    readonly w: number | undefined;
}

const arbEdges: fc.Arbitrary<WeightedEdge[]> = fc.array(
    fc.record({
        u: fc.integer({ min: 0, max: 5 }),
        v: fc.integer({ min: 0, max: 5 }),
        w: fc.option(arbWeight, { nil: undefined }),
    }),
    { minLength: 1, maxLength: 24 },
);

function shadowOf(s: GraphSnapshot): { dtype: string; data: number[]; validity: boolean[] | null } | null {
    const column = s.edges.byRole("weight");
    if (column === null) {
        return null;
    }
    if (column.dtype !== "f32" && column.dtype !== "f64") {
        throw new Error(`shadow column has dtype ${column.dtype}`);
    }
    return {
        dtype: column.dtype,
        data: Array.from(column.data),
        validity: column.validity === null ? null : Array.from({ length: s.edgeCount }, (_, e) => column.isSet(e)),
    };
}

describe("f32 storage and the f64 shadow rule (design section 3.7)", () => {
    for (const directed of [true, false]) {
        it(`weightDtype "f64": arc weights are fround(w), the shadow exists iff some weight is not f32-exact or some edge omitted its weight (${directed ? "directed" : "undirected"})`, () => {
            fc.assert(
                fc.property(arbEdges, (edges) => {
                    const builder = new GraphBuilder({ directed, weightDtype: "f64" });
                    for (const { u, v, w } of edges) {
                        builder.addEdge(u, v, w);
                    }
                    const s = builder.freeze();
                    assertInvariants(s);
                    const anyExplicit = edges.some((e) => e.w !== undefined);
                    const anyOmitted = edges.some((e) => e.w === undefined);
                    expect(s.weights === null).toBe(!anyExplicit);
                    const shadow = shadowOf(s);
                    if (!anyExplicit) {
                        expect(shadow).toBeNull();
                        return;
                    }
                    const weights = s.weights as Float32Array;
                    const values = edges.map((e) => e.w ?? 1);
                    const inexact = values.some((w) => Math.fround(w) !== w);
                    values.forEach((w, e) => {
                        const stored = weights[s.edgeToArc[e]];
                        expect(Object.is(stored, Math.fround(w))).toBe(true);
                    });
                    if (!inexact && !anyOmitted) {
                        expect(shadow).toBeNull();
                        return;
                    }
                    expect(shadow).not.toBeNull();
                    const column = shadow as NonNullable<typeof shadow>;
                    expect(column.dtype).toBe(inexact ? "f64" : "f32");
                    values.forEach((w, e) => {
                        expect(Object.is(column.data[e], inexact ? w : Math.fround(w))).toBe(true);
                    });
                    if (anyOmitted) {
                        expect(column.validity).not.toBeNull();
                        edges.forEach((edge, e) => {
                            expect((column.validity as boolean[])[e]).toBe(edge.w !== undefined);
                        });
                    } else {
                        expect(column.validity).toBeNull();
                    }
                }),
                { numRuns: RUNS },
            );
        });
    }

    it('weightDtype "f32" never keeps an f64 shadow and edgeWeight() reads the rounded value back', () => {
        fc.assert(
            fc.property(arbEdges, fc.boolean(), (edges, directed) => {
                const builder = new GraphBuilder({ directed, weightDtype: "f32" });
                for (const { u, v, w } of edges) {
                    builder.addEdge(u, v, w);
                }
                const s = builder.freeze();
                assertInvariants(s);
                const shadow = shadowOf(s);
                if (shadow !== null) {
                    expect(shadow.dtype).toBe("f32");
                    expect(shadow.validity).not.toBeNull();
                }
                edges.forEach((edge, e) => {
                    expect(Object.is(builder.edgeWeight(e), Math.fround(edge.w ?? 1))).toBe(true);
                });
            }),
            { numRuns: RUNS },
        );
    });

    it("a declared weighted builder whose edges all omit the weight keeps the array with an all-clear validity (design 3.7)", () => {
        const builder = new GraphBuilder({ directed: true, weighted: true });
        builder.addEdge(0, 1);
        builder.addEdge(1, 2);
        const s = builder.freeze();
        expect(s.weights).not.toBeNull();
        expect(s.flags.allWeightsOne).toBe(true);
        const shadow = shadowOf(s);
        expect(shadow).not.toBeNull();
        expect(shadow?.dtype).toBe("f32");
        expect(shadow?.validity).toEqual([false, false]);
        expect(s.edges.byRole("weight")?.nullCount).toBe(2);
    });

    it("flags describe the f32 ARC array, not the f64 shadow (design 3.8 predicate; a shadow consumer sees allWeightsOne / nonNegativeWeights that do not match its values)", () => {
        // 1 + 2^-30 rounds to 1 in f32; -1e-50 rounds to -0; 1e39 rounds to Infinity
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.addEdge(0, 1, 1 + 2 ** -30);
        builder.addEdge(1, 2, 1);
        const s = builder.freeze();
        expect(s.flags.allWeightsOne).toBe(true);
        expect(Array.from(s.weights as Float32Array)).toEqual([1, 1]);
        expect(shadowOf(s)?.data[0]).toBe(1 + 2 ** -30);
        const negative = new GraphBuilder({ directed: true, weightDtype: "f64" });
        negative.addEdge(0, 1, -1e-50);
        const n = negative.freeze();
        expect(n.flags.nonNegativeWeights).toBe(true);
        expect(Object.is((n.weights as Float32Array)[0], -0)).toBe(true);
        expect(shadowOf(n)?.data[0]).toBe(-1e-50);
        const overflow = new GraphBuilder({ directed: true, weightDtype: "f64" });
        overflow.addEdge(0, 1, 1e39);
        const o = overflow.freeze();
        expect(o.flags.finiteWeights).toBe(false);
        expect((o.weights as Float32Array)[0]).toBe(Infinity);
        expect(shadowOf(o)?.data[0]).toBe(1e39);
        for (const snapshot of [s, n, o]) {
            assertInvariants(snapshot);
        }
    });
});

describe("duplicate-policy reducers (design section 6.5) and the NaN re-check of freeze step 8", () => {
    it('PINS DEFECT (finding: reducer NaN silently replaced): "sum" over +Infinity and -Infinity must not freeze a snapshot whose survivor silently keeps one operand', () => {
        for (const order of [
            [Infinity, -Infinity],
            [-Infinity, Infinity],
        ]) {
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            builder.addEdge(0, 1, order[0]);
            builder.addEdge(0, 1, order[1]);
            let caught: unknown = null;
            let snapshot: GraphSnapshot | null = null;
            try {
                snapshot = builder.freeze({ duplicateEdges: "sum" });
            } catch (err) {
                caught = err;
            }
            // design 6.3 step 8: a NaN produced for the merged edge is E_INVALID_WEIGHT (I8), never a
            // silently substituted operand; the stored weight must not be one of the two operands
            if (snapshot !== null) {
                const stored = (snapshot.weights as Float32Array)[0];
                expect(Object.is(stored, order[0]) || Object.is(stored, order[1])).toBe(false);
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            expect((caught as GraphFormatError).code).toBe("E_INVALID_WEIGHT");
        }
    });

    it("PINS DEFECT (finding: reducer NaN silently replaced): the reference model and the builder disagree on the merged weight", () => {
        const model = new ModelGraph({
            directed: true,
            weighted: "auto",
            weightDtype: "f64",
            duplicateEdges: "sum",
            selfLoops: "keep",
            addMissingNodes: true,
        });
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64", duplicateEdges: "sum" });
        expect(builder.addEdge("a", "b", Infinity)).toBe(model.addEdge("a", "b", Infinity));
        expect(builder.addEdge("a", "b", -Infinity)).toBe(model.addEdge("a", "b", -Infinity));
        // the reducer's arithmetic result is NaN, which no snapshot may hold (I8): both constructions
        // of section 16.1 refuse the freeze with E_INVALID_WEIGHT (step 8's re-check) and the builder
        // is left untouched
        let modelCode: string | null = null;
        try {
            model.freeze();
        } catch (err) {
            modelCode = err instanceof ModelError ? err.code : null;
        }
        expect(modelCode).toBe("E_INVALID_WEIGHT");
        let builderCode: string | null = null;
        try {
            builder.freeze();
        } catch (err) {
            builderCode = err instanceof GraphFormatError ? err.code : null;
        }
        expect(builderCode).toBe("E_INVALID_WEIGHT");
        expect(builder.edgeCount).toBe(2);
        expect(builder.freeze({ duplicateEdges: "keep" }).edgeCount).toBe(2);
    });

    it('"sum" / "min" / "max" agree with f64 arithmetic over the staged values, rounded to f32 on the arc array', () => {
        fc.assert(
            fc.property(
                fc.array(fc.option(arbWeight, { nil: undefined }), { minLength: 2, maxLength: 6 }),
                fc.constantFrom<"sum" | "min" | "max">("sum", "min", "max"),
                fc.constantFrom<"f32" | "f64">("f32", "f64"),
                fc.boolean(),
                (weights, policy, weightDtype, directed) => {
                    const stage = (w: number | undefined): number => {
                        if (w === undefined) {
                            return 1;
                        }
                        return weightDtype === "f32" ? Math.fround(w) : w;
                    };
                    const staged = weights.map(stage);
                    const combine = (acc: number, w: number): number => {
                        if (policy === "sum") {
                            return acc + w;
                        }
                        return policy === "min" ? Math.min(acc, w) : Math.max(acc, w);
                    };
                    let reduced = staged[0];
                    for (let i = 1; i < staged.length; i++) {
                        reduced = combine(reduced, staged[i]);
                    }
                    fc.pre(!Number.isNaN(reduced));
                    const builder = new GraphBuilder({ directed, weightDtype });
                    for (const w of weights) {
                        builder.addEdge(0, 1, w);
                    }
                    const anyExplicit = weights.some((w) => w !== undefined);
                    const s = builder.freeze({ duplicateEdges: policy });
                    assertInvariants(s);
                    expect(s.edgeCount).toBe(1);
                    expect(s.flags.multigraph).toBe(false);
                    if (!anyExplicit && policy !== "sum") {
                        expect(s.weights).toBeNull();
                        return;
                    }
                    const stored = (s.weights as Float32Array)[s.edgeToArc[0]];
                    expect(Object.is(stored, Math.fround(reduced))).toBe(true);
                    const shadow = shadowOf(s);
                    if (weightDtype === "f64" && Math.fround(reduced) !== reduced) {
                        expect(shadow?.dtype).toBe("f64");
                        expect(shadow?.data[0]).toBe(reduced);
                    }
                },
            ),
            { numRuns: RUNS },
        );
    });

    it("f32 overflow of a sum is reported truthfully by finiteWeights and preserved by an f64 shadow", () => {
        const f32 = new GraphBuilder({ directed: true, weightDtype: "f32" });
        f32.addEdge(0, 1, 3e38);
        f32.addEdge(0, 1, 3e38);
        const a = f32.freeze({ duplicateEdges: "sum" });
        expect((a.weights as Float32Array)[0]).toBe(Infinity);
        expect(a.flags.finiteWeights).toBe(false);
        expect(shadowOf(a)).toBeNull();
        const f64 = new GraphBuilder({ directed: true, weightDtype: "f64" });
        f64.addEdge(0, 1, 3e38);
        f64.addEdge(0, 1, 3e38);
        const b = f64.freeze({ duplicateEdges: "sum" });
        expect((b.weights as Float32Array)[0]).toBe(Infinity);
        expect(b.flags.finiteWeights).toBe(false);
        expect(shadowOf(b)?.dtype).toBe("f64");
        expect(shadowOf(b)?.data[0]).toBe(6e38);
        assertInvariants(a);
        assertInvariants(b);
    });

    it("NaN is refused at every entry point and the builder is left untouched", () => {
        const builder = new GraphBuilder({ directed: false });
        builder.addEdge(0, 1, 2);
        const mutations = builder.mutationCount;
        const attempts: (() => unknown)[] = [
            () => builder.addEdge(1, 2, NaN),
            () => builder.addEdgeByIndex(0, 1, NaN),
            () => builder.setEdgeWeight(0, NaN),
            () => builder.addEdges(new Uint32Array([0]), new Uint32Array([1]), new Float32Array([NaN])),
            () => builder.addEdgesByIds([0], [1], [NaN]),
            () => builder.addEdgeRecord(0, 1, { weight: NaN }),
        ];
        for (const attempt of attempts) {
            expect(attempt).toThrow(expect.objectContaining({ code: "E_INVALID_WEIGHT" }));
        }
        expect(builder.mutationCount).toBe(mutations);
        expect(builder.edgeBound).toBe(1);
        expect(builder.edgeWeight(0)).toBe(2);
    });
});
