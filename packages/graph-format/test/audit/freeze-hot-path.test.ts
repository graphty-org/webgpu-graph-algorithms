/**
 * Adversarial audit of the freeze hot path (design sections 6.3 steps 3-8, 6.4, 15.4 and invariants
 * I4, I5, I7, I8, I9): the counting sort was restructured for speed (one loop per function, a
 * paired pass-1 transient, scatters that carry no dependent random loads, weight predicates over
 * the edges), so every semantic the old scatter established implicitly is pinned here explicitly --
 * on graphs far larger than the unit tests use, through both the plain-array entry point and the
 * builder's resizable-buffer staging views.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { countSelfLoops, type SortInput, sortIntoCore } from "../../src/builder/counting-sort.js";
import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { GraphFormatError } from "../../src/errors.js";
import { type F32, type F64, type GraphSnapshot, type SnapshotFlags, type U32 } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";

const RUNS = Number(process.env.FC_RUNS ?? "60");

/** A deterministic xorshift32 generator. */
function makeRandom(seed: number): () => number {
    let state = seed >>> 0 || 1;
    return () => {
        state ^= state << 13;
        state >>>= 0;
        state ^= state >>> 17;
        state ^= state << 5;
        state >>>= 0;
        return state / 0x100000000;
    };
}

/** The reference CSR by a comparator sort over (source, target, edge), with the flags over the STORED f32 values. */
interface NaiveCore {
    rowPtr: number[];
    colIdx: number[];
    arcToEdge: number[];
    edgeToArc: number[];
    weights: number[] | null;
    multigraph: boolean;
    allOne: boolean;
    nonNegative: boolean;
    finite: boolean;
}

/**
 * Sort an input naively.
 * @param input - the per-edge arrays
 * @returns the reference core
 */
function naive(input: SortInput): NaiveCore {
    const arcs: { source: number; target: number; edge: number; declared: boolean }[] = [];
    for (let e = 0; e < input.edgeCount; e++) {
        arcs.push({ source: input.src[e], target: input.dst[e], edge: e, declared: true });
        if (!input.directed && input.src[e] !== input.dst[e]) {
            arcs.push({ source: input.dst[e], target: input.src[e], edge: e, declared: false });
        }
    }
    arcs.sort((a, b) => a.source - b.source || a.target - b.target || a.edge - b.edge);
    const rowPtr = new Array<number>(input.nodeCount + 1).fill(0);
    for (const arc of arcs) {
        rowPtr[arc.source + 1]++;
    }
    for (let u = 0; u < input.nodeCount; u++) {
        rowPtr[u + 1] += rowPtr[u];
    }
    const edgeToArc = new Array<number>(input.edgeCount).fill(-1);
    let multigraph = false;
    arcs.forEach((arc, a) => {
        if (arc.declared) {
            edgeToArc[arc.edge] = a;
        }
        if (a > 0 && arcs[a - 1].source === arc.source && arcs[a - 1].target === arc.target) {
            multigraph = true;
        }
    });
    const weights =
        input.weights === null ? null : arcs.map((arc) => Math.fround((input.weights as Float64Array)[arc.edge]));
    return {
        rowPtr,
        colIdx: arcs.map((arc) => arc.target),
        arcToEdge: arcs.map((arc) => arc.edge),
        edgeToArc,
        weights,
        multigraph,
        allOne: weights === null || weights.every((w) => w === 1),
        nonNegative: weights === null || weights.every((w) => !(w < 0)),
        finite: weights === null || weights.every((w) => Number.isFinite(w)),
    };
}

/**
 * Compare a sort result with the reference.
 * @param input - the input
 * @param useArena - the arena flag
 */
function expectMatchesNaive(input: SortInput, useArena: boolean): void {
    const expected = naive(input);
    const { core, arcCount, selfLoopCount, flags } = sortIntoCore(input, useArena);
    expect(arcCount).toBe(expected.colIdx.length);
    expect(selfLoopCount).toBe(countSelfLoops(input.src, input.dst, input.edgeCount));
    expect(Array.from(core.rowPtr)).toEqual(expected.rowPtr);
    expect(Array.from(core.colIdx)).toEqual(expected.colIdx);
    if (core.arcToEdge === null || core.edgeToArc === null) {
        expect(flags.arcToEdgeIsIdentity).toBe(true);
        expect(expected.arcToEdge).toEqual(expected.arcToEdge.map((_, i) => i));
    } else {
        expect(flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(core.arcToEdge)).toEqual(expected.arcToEdge);
        expect(Array.from(core.edgeToArc)).toEqual(expected.edgeToArc);
    }
    if (expected.weights === null) {
        expect(core.weights).toBeNull();
    } else {
        expect(Array.from(core.weights as Float32Array)).toEqual(expected.weights);
    }
    expect(flags.multigraph).toBe(expected.multigraph);
    expect(flags.hasSelfLoops).toBe(selfLoopCount > 0);
    expect(flags.weighted).toBe(input.weights !== null);
    expect(flags.allWeightsOne).toBe(expected.allOne);
    expect(flags.nonNegativeWeights).toBe(expected.nonNegative);
    expect(flags.finiteWeights).toBe(expected.finite);
    expect(core.arena === null).toBe(!useArena);
}

/** A weight pool that exercises rounding, overflow, underflow, signed zero and the infinities. */
const arbWeight: fc.Arbitrary<number> = fc.oneof(
    { weight: 6, arbitrary: fc.constant(1) },
    { weight: 1, arbitrary: fc.constant(0) },
    { weight: 1, arbitrary: fc.constant(-0) },
    { weight: 1, arbitrary: fc.constant(-1) },
    { weight: 1, arbitrary: fc.constant(1 + 2 ** -30) },
    { weight: 1, arbitrary: fc.constant(-1e-50) },
    { weight: 1, arbitrary: fc.constant(1e39) },
    { weight: 1, arbitrary: fc.constant(Infinity) },
    { weight: 1, arbitrary: fc.constant(-Infinity) },
    { weight: 2, arbitrary: fc.double({ noNaN: true }) },
);

/** Random inputs: dense enough for many parallels and self-loops, larger than the unit tests. */
const arbInput: fc.Arbitrary<SortInput> = fc
    .record({
        directed: fc.boolean(),
        nodeCount: fc.integer({ min: 1, max: 300 }),
        weighted: fc.constantFrom("none", "f32", "f64"),
        edgeCount: fc.integer({ min: 0, max: 3000 }),
        seed: fc.integer({ min: 1, max: 2 ** 31 - 1 }),
        weights: fc.array(arbWeight, { minLength: 8, maxLength: 8 }),
    })
    .map(({ directed, nodeCount, weighted, edgeCount, seed, weights }) => {
        const random = makeRandom(seed);
        const src = new Uint32Array(edgeCount);
        const dst = new Uint32Array(edgeCount);
        const w = new Float64Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            src[e] = Math.floor(random() * nodeCount);
            dst[e] = Math.floor(random() * nodeCount);
            w[e] = weights[Math.floor(random() * weights.length)];
        }
        let stagingWeights: F32 | F64 | null = null;
        if (weighted === "f32") {
            stagingWeights = Float32Array.from(w);
        } else if (weighted === "f64") {
            stagingWeights = w;
        }
        return { directed, nodeCount, edgeCount, src, dst, weights: stagingWeights };
    });

describe("sortIntoCore differential on large random multigraphs (design 6.3 steps 3-6, I4, I5, I7)", () => {
    it("matches the comparator construction, with the arena and without", () => {
        fc.assert(
            fc.property(arbInput, fc.boolean(), (input, useArena) => {
                expectMatchesNaive(input, useArena);
            }),
            { numRuns: RUNS },
        );
    });

    it("matches on a 20k-node / 200k-edge graph in both directions and both weight precisions", () => {
        const nodeCount = 20_000;
        const edgeCount = 200_000;
        const random = makeRandom(4242);
        const src = new Uint32Array(edgeCount);
        const dst = new Uint32Array(edgeCount);
        const w = new Float64Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            src[e] = Math.floor(random() * nodeCount);
            dst[e] = e % 97 === 0 ? src[e] : Math.floor(random() * nodeCount);
            w[e] = e % 13 === 0 ? -0.5 : 1 + Math.floor(random() * 4);
        }
        for (const directed of [true, false]) {
            for (const weights of [null, Float32Array.from(w), w]) {
                expectMatchesNaive({ directed, nodeCount, edgeCount, src, dst, weights }, true);
            }
        }
    });

    it("takes the identity path for sorted directed input with f64 staging weights, rounding them to f32", () => {
        const input: SortInput = {
            directed: true,
            nodeCount: 4,
            edgeCount: 5,
            src: new Uint32Array([0, 0, 1, 2, 3]),
            dst: new Uint32Array([1, 1, 3, 2, 0]),
            weights: new Float64Array([16777217, 1 + 2 ** -30, -1e-50, 1e39, -Infinity]),
        };
        expectMatchesNaive(input, true);
        expectMatchesNaive(input, false);
        const { core, flags } = sortIntoCore(input, true);
        expect(flags.arcToEdgeIsIdentity).toBe(true);
        expect(flags.multigraph).toBe(true);
        expect(Array.from(core.weights as Float32Array)).toEqual([16777216, 1, -0, Infinity, -Infinity]);
        expect(flags.finiteWeights).toBe(false);
        expect(flags.nonNegativeWeights).toBe(false);
        expect(flags.allWeightsOne).toBe(false);
    });
});

describe("flags are predicates over the STORED f32 arc values (design 3.8, I9), not over the staging values", () => {
    /**
     * Sort one directed unsorted edge list with the given f64 weights.
     * @param weights - the staging weights
     * @returns the flags
     */
    function flagsOf(weights: number[]): SnapshotFlags {
        const n = weights.length;
        const src = new Uint32Array(n);
        const dst = new Uint32Array(n);
        for (let e = 0; e < n; e++) {
            src[e] = n - 1 - e;
            dst[e] = e;
        }
        return sortIntoCore(
            { directed: true, nodeCount: n, edgeCount: n, src, dst, weights: Float64Array.from(weights) },
            true,
        ).flags;
    }

    it("an f64 weight that rounds to 1 keeps allWeightsOne", () => {
        const flags = flagsOf([1, 1 + 2 ** -30, 1 - 2 ** -30]);
        expect(flags.allWeightsOne).toBe(true);
        expect(flags.nonNegativeWeights).toBe(true);
        expect(flags.finiteWeights).toBe(true);
    });

    it("a tiny negative f64 that rounds to -0 keeps nonNegativeWeights (the stored value is not < 0)", () => {
        const flags = flagsOf([1, -1e-50, -0]);
        expect(flags.nonNegativeWeights).toBe(true);
        expect(flags.allWeightsOne).toBe(false);
        expect(flags.finiteWeights).toBe(true);
    });

    it("an f64 beyond the f32 range rounds to Infinity and clears finiteWeights but not nonNegativeWeights", () => {
        const flags = flagsOf([1, 1e39]);
        expect(flags.finiteWeights).toBe(false);
        expect(flags.nonNegativeWeights).toBe(true);
        expect(flags.allWeightsOne).toBe(false);
    });

    it("every weight -Infinity clears all three; every weight +Infinity keeps nonNegativeWeights", () => {
        const negative = flagsOf([-Infinity, -Infinity]);
        expect(negative.allWeightsOne).toBe(false);
        expect(negative.nonNegativeWeights).toBe(false);
        expect(negative.finiteWeights).toBe(false);
        const positive = flagsOf([Infinity, Infinity]);
        expect(positive.allWeightsOne).toBe(false);
        expect(positive.nonNegativeWeights).toBe(true);
        expect(positive.finiteWeights).toBe(false);
    });

    it("an unweighted sort and a weighted sort of zero edges both report the three weight flags true", () => {
        const empty = sortIntoCore(
            {
                directed: false,
                nodeCount: 3,
                edgeCount: 0,
                src: new Uint32Array(0),
                dst: new Uint32Array(0),
                weights: new Float32Array(0),
            },
            true,
        );
        expect(empty.flags.weighted).toBe(true);
        expect(empty.flags.allWeightsOne).toBe(true);
        expect(empty.flags.nonNegativeWeights).toBe(true);
        expect(empty.flags.finiteWeights).toBe(true);
        expect(empty.flags.multigraph).toBe(false);
        const unweighted = sortIntoCore(
            {
                directed: true,
                nodeCount: 2,
                edgeCount: 2,
                src: new Uint32Array([1, 0]),
                dst: new Uint32Array([0, 1]),
                weights: null,
            },
            false,
        );
        expect(unweighted.flags.weighted).toBe(false);
        expect(unweighted.flags.allWeightsOne).toBe(true);
        expect(unweighted.flags.nonNegativeWeights).toBe(true);
        expect(unweighted.flags.finiteWeights).toBe(true);
        expect(unweighted.core.weights).toBeNull();
    });
});

describe("the NaN re-check of freeze step 8 (I8)", () => {
    it("names the LOWEST NaN edge with its weight, before any core array is allocated (unsorted and sorted input)", () => {
        for (const [src, dst] of [
            [new Uint32Array([2, 1, 0, 2]), new Uint32Array([0, 1, 2, 1])],
            [new Uint32Array([0, 0, 1, 2]), new Uint32Array([0, 1, 1, 2])],
        ]) {
            const input: SortInput = {
                directed: true,
                nodeCount: 3,
                edgeCount: 4,
                src,
                dst,
                weights: new Float64Array([1, Number.NaN, 2, Number.NaN]),
            };
            let caught: GraphFormatError | null = null;
            try {
                sortIntoCore(input, true);
            } catch (err) {
                caught = err as GraphFormatError;
            }
            expect(caught).toBeInstanceOf(GraphFormatError);
            expect(caught?.code).toBe("E_INVALID_WEIGHT");
            expect(caught?.details.edge).toBe(1);
            expect(caught?.details.weight).toBeNaN();
        }
    });

    it("is unreachable through the builder for pushed edges: every add path validates NaN first, so the freeze never throws it", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        expect(() =>
            b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1, 2]), new Float32Array([1, Number.NaN])),
        ).toThrow(GraphFormatError);
        expect(b.edgeCount).toBe(0);
        b.addEdges(new Uint32Array([0, 1]), new Uint32Array([1, 2]), new Float32Array([1, 2]));
        expect(() => b.setEdgeWeight(0, Number.NaN)).toThrow(GraphFormatError);
        const s = b.freeze();
        expect(Array.from(s.weights as Float32Array)).toEqual([1, 2]);
        expect(s.flags.finiteWeights).toBe(true);
    });
});

describe("freeze through the builder's resizable-buffer staging (design 6.2) at scale", () => {
    /**
     * Build and freeze a random multigraph with self-loops.
     * @param directed - the direction
     * @param nodeCount - nodes
     * @param edgeCount - edges
     * @param weightDtype - the staging precision
     * @returns the builder and the snapshot
     */
    function frozen(
        directed: boolean,
        nodeCount: number,
        edgeCount: number,
        weightDtype: "f32" | "f64",
    ): { builder: GraphBuilder; src: U32; dst: U32; weights: F64; snapshot: GraphSnapshot } {
        const random = makeRandom(7 + edgeCount);
        const src = new Uint32Array(edgeCount);
        const dst = new Uint32Array(edgeCount);
        const weights = new Float64Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            src[e] = Math.floor(random() * nodeCount);
            dst[e] = e % 53 === 0 ? src[e] : Math.floor(random() * nodeCount);
            weights[e] = e % 11 === 0 ? 1 + 2 ** -30 : 1 + Math.floor(random() * 5);
        }
        const builder = new GraphBuilder({ directed, weightDtype, expectedNodes: 16, expectedEdges: 16 });
        builder.addAnonymousNodes(nodeCount);
        builder.addEdges(src, dst, weights);
        return { builder, src, dst, weights, snapshot: builder.freeze() };
    }

    it("a 30k-node / 300k-edge undirected multigraph satisfies I1-I13, pairs its mates (6.4) and stores the declared orientation (I5)", () => {
        const { src, dst, weights, snapshot } = frozen(false, 30_000, 300_000, "f32");
        assertInvariants(snapshot);
        expect(snapshot.arcCount).toBe(2 * 300_000 - snapshot.selfLoopCount);
        expect(snapshot.selfLoopCount).toBe(countSelfLoops(src, dst, 300_000));
        const { rowPtr, colIdx, arcToEdge, edgeToArc } = snapshot;
        const mate = snapshot.mate();
        // row of every arc, by a walk over rowPtr
        const rowOf = new Uint32Array(snapshot.arcCount);
        for (let u = 0; u < snapshot.nodeCount; u++) {
            rowOf.fill(u, rowPtr[u], rowPtr[u + 1]);
        }
        const stored = snapshot.weights as Float32Array;
        let mismatches = 0;
        for (let e = 0; e < 300_000; e++) {
            const a = edgeToArc[e];
            const m = mate[a];
            if (rowOf[a] !== src[e] || colIdx[a] !== dst[e] || arcToEdge[a] !== e) {
                mismatches++;
            }
            if (stored[a] !== Math.fround(weights[e])) {
                mismatches++;
            }
            if (src[e] === dst[e]) {
                if (m !== a) {
                    mismatches++;
                }
            } else if (
                m === a ||
                arcToEdge[m] !== e ||
                rowOf[m] !== dst[e] ||
                colIdx[m] !== src[e] ||
                stored[m] !== stored[a]
            ) {
                mismatches++;
            }
        }
        expect(mismatches).toBe(0);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.flags.hasSelfLoops).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(false);
    });

    it("a 30k-node / 300k-edge directed multigraph with f64 staging keeps I1-I13 and rounds the arc weights", () => {
        const { weights, snapshot } = frozen(true, 30_000, 300_000, "f64");
        assertInvariants(snapshot);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
        const { arcToEdge } = snapshot;
        const stored = snapshot.weights as Float32Array;
        let mismatches = 0;
        for (let a = 0; a < snapshot.arcCount; a++) {
            if (stored[a] !== Math.fround(weights[arcToEdge[a]])) {
                mismatches++;
            }
        }
        expect(mismatches).toBe(0);
        // the f64 shadow column exists exactly because 1 + 2^-30 is not f32-exact
        expect(snapshot.edges.has("graphty.weight")).toBe(true);
    });

    it("re-freezing the same builder is deterministic (I15) and never aliases staging (I18)", () => {
        const { builder, snapshot } = frozen(false, 5_000, 40_000, "f32");
        const again = builder.freeze();
        expect(Array.from(again.rowPtr)).toEqual(Array.from(snapshot.rowPtr));
        expect(Array.from(again.colIdx)).toEqual(Array.from(snapshot.colIdx));
        expect(Array.from(again.arcToEdge)).toEqual(Array.from(snapshot.arcToEdge));
        expect(Array.from(again.edgeToArc)).toEqual(Array.from(snapshot.edgeToArc));
        expect(again.rowPtr).not.toBe(snapshot.rowPtr);
        expect(again.colIdx.buffer).not.toBe(snapshot.colIdx.buffer);
        expect(snapshot.colIdx.buffer.resizable).toBe(false);
        expect(again.flags).toEqual(snapshot.flags);
    });
});

describe("bulk addEdges is one pushEdge per edge, in order (design 6.6, 3.7): differential against the per-edge path", () => {
    /** One builder operation of the scenario. */
    type Op =
        | {
              readonly kind: "bulk";
              readonly src: number[];
              readonly dst: number[];
              readonly weights: number[] | null;
              readonly dtype: "f32" | "f64";
          }
        | { readonly kind: "single"; readonly u: number; readonly v: number; readonly weight: number | null }
        | { readonly kind: "setWeight"; readonly slot: number; readonly weight: number }
        | { readonly kind: "remove"; readonly slot: number };

    const NODES = 6;
    const arbIndex = fc.integer({ min: 0, max: NODES - 1 });
    const arbW = fc.oneof(
        fc.constant(1),
        fc.constant(2.5),
        fc.constant(-1),
        fc.constant(1 + 2 ** -30),
        fc.double({ noNaN: true }),
    );
    const arbOp: fc.Arbitrary<Op> = fc.oneof(
        fc
            .record({
                n: fc.integer({ min: 0, max: 7 }),
                weighted: fc.boolean(),
                dtype: fc.constantFrom("f32", "f64"),
                seed: fc.integer({ min: 1, max: 1_000_000 }),
                ws: fc.array(arbW, { minLength: 7, maxLength: 7 }),
            })
            .map(({ n, weighted, dtype, seed, ws }): Op => {
                const random = makeRandom(seed);
                const src: number[] = [];
                const dst: number[] = [];
                for (let i = 0; i < n; i++) {
                    src.push(Math.floor(random() * NODES));
                    dst.push(Math.floor(random() * NODES));
                }
                return { kind: "bulk", src, dst, weights: weighted ? ws.slice(0, n) : null, dtype };
            }),
        fc
            .record({ u: arbIndex, v: arbIndex, weight: fc.option(arbW, { nil: null }) })
            .map((r): Op => ({ kind: "single", ...r })),
        fc.record({ slot: fc.nat({ max: 40 }), weight: arbW }).map((r): Op => ({ kind: "setWeight", ...r })),
        fc.record({ slot: fc.nat({ max: 40 }) }).map((r): Op => ({ kind: "remove", ...r })),
    );

    /**
     * Apply a scenario to a builder, bulk ops as addEdges (bulk = true) or as addEdgeByIndex calls.
     * @param ops - the scenario
     * @param bulk - which path the bulk ops take
     * @param options - builder options
     * @returns the builder
     */
    function apply(
        ops: readonly Op[],
        bulk: boolean,
        options: { weighted?: boolean; weightDtype: "f32" | "f64" },
    ): GraphBuilder {
        const b = new GraphBuilder({ directed: true, ...options });
        b.addAnonymousNodes(NODES);
        for (const op of ops) {
            switch (op.kind) {
                case "bulk": {
                    if (options.weighted === false && op.weights !== null && op.weights.some((w) => w !== 1)) {
                        break;
                    }
                    const src = Uint32Array.from(op.src);
                    const dst = Uint32Array.from(op.dst);
                    let weights: F32 | F64 | undefined;
                    if (op.weights !== null) {
                        weights = op.dtype === "f32" ? Float32Array.from(op.weights) : Float64Array.from(op.weights);
                    }
                    if (bulk) {
                        b.addEdges(src, dst, weights);
                    } else {
                        for (let i = 0; i < src.length; i++) {
                            b.addEdgeByIndex(src[i], dst[i], weights === undefined ? undefined : weights[i]);
                        }
                    }
                    break;
                }
                case "single":
                    if (options.weighted === false && op.weight !== null && op.weight !== 1) {
                        break;
                    }
                    b.addEdgeByIndex(op.u, op.v, op.weight ?? undefined);
                    break;
                case "setWeight":
                    if (b.edgeCount > 0 && !(options.weighted === false && op.weight !== 1)) {
                        const live = b.outEdgesOf(op.slot % NODES);
                        if (live.length > 0) {
                            b.setEdgeWeight(live[op.slot % live.length], op.weight);
                        }
                    }
                    break;
                case "remove":
                    if (b.edgeBound > 0) {
                        b.removeEdge(op.slot % b.edgeBound);
                    }
                    break;
                default:
                    throw new Error("unreachable");
            }
        }
        return b;
    }

    /**
     * Every observable of a frozen snapshot that the push path determines.
     * @param s - the snapshot
     * @returns a comparable record
     */
    function observe(s: GraphSnapshot): Record<string, unknown> {
        const shadow = s.edges.get("graphty.weight");
        return {
            counts: [s.nodeCount, s.edgeCount, s.arcCount, s.selfLoopCount],
            rowPtr: Array.from(s.rowPtr),
            colIdx: Array.from(s.colIdx),
            arcToEdge: Array.from(s.arcToEdge),
            edgeToArc: Array.from(s.edgeToArc),
            weights: s.weights === null ? null : Array.from(s.weights),
            flags: s.flags,
            shadow:
                shadow === null
                    ? null
                    : {
                          dtype: shadow.meta.dtype,
                          nullable: shadow.meta.nullable,
                          data: Array.from(
                              shadow.meta.dtype === "f64"
                                  ? s.edges.requireTyped("graphty.weight", "f64").data
                                  : s.edges.requireTyped("graphty.weight", "f32").data,
                          ),
                          set: Array.from({ length: s.edgeCount }, (_, e) => shadow.isSet(e)),
                      },
        };
    }

    it("weighted: 'auto' builders, f32 and f64 staging", () => {
        fc.assert(
            fc.property(fc.array(arbOp, { maxLength: 12 }), fc.constantFrom("f32", "f64"), (ops, weightDtype) => {
                const viaBulk = apply(ops, true, { weightDtype });
                const viaSingle = apply(ops, false, { weightDtype });
                expect(viaBulk.edgeBound).toBe(viaSingle.edgeBound);
                for (let e = 0; e < viaBulk.edgeBound; e++) {
                    expect(viaBulk.outEdgesOf(e % NODES)).toEqual(viaSingle.outEdgesOf(e % NODES));
                    expect(viaBulk.inEdgesOf(e % NODES)).toEqual(viaSingle.inEdgesOf(e % NODES));
                }
                expect(observe(viaBulk.freeze())).toEqual(observe(viaSingle.freeze()));
            }),
            { numRuns: RUNS * 3 },
        );
    });

    it("weighted: false and weighted: true builders", () => {
        fc.assert(
            fc.property(fc.array(arbOp, { maxLength: 12 }), fc.boolean(), (ops, weighted) => {
                const viaBulk = apply(ops, true, { weighted, weightDtype: "f32" });
                const viaSingle = apply(ops, false, { weighted, weightDtype: "f32" });
                expect(observe(viaBulk.freeze())).toEqual(observe(viaSingle.freeze()));
            }),
            { numRuns: RUNS * 2 },
        );
    });

    it("a batch that flips the explicit / omitted mode allocates the same validity as the per-edge path", () => {
        for (const first of [true, false]) {
            const bulk = new GraphBuilder({ directed: false });
            const single = new GraphBuilder({ directed: false });
            for (const b of [bulk, single]) {
                b.addAnonymousNodes(3);
                b.addEdgeByIndex(0, 1, first ? 2 : undefined);
            }
            const src = new Uint32Array([1, 2, 0]);
            const dst = new Uint32Array([2, 0, 0]);
            const w = new Float64Array([3, 4, 5]);
            bulk.addEdges(src, dst, first ? undefined : w);
            for (let i = 0; i < 3; i++) {
                single.addEdgeByIndex(src[i], dst[i], first ? undefined : w[i]);
            }
            // a second batch in the now-mixed state
            bulk.addEdges(new Uint32Array([2]), new Uint32Array([2]), new Float32Array([7]));
            single.addEdgeByIndex(2, 2, 7);
            expect(observe(bulk.freeze())).toEqual(observe(single.freeze()));
            const shadow = bulk.freeze().edges.get("graphty.weight");
            expect(shadow).not.toBeNull();
            expect(Array.from({ length: 5 }, (_, e) => shadow?.isSet(e))).toEqual(
                first ? [true, false, false, false, true] : [false, true, true, true, true],
            );
        }
    });
});
