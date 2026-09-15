/**
 * Unit tests of the two-pass counting sort (design section 6.3 steps 3-6 and 8): against a naive
 * comparator construction, the identity fast path, undirected doubling with the pairing invariant,
 * NaN rejection and the flag predicates.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { countSelfLoops, isSortedEdgeList, type SortInput, sortIntoCore } from "../../src/builder/counting-sort.js";
import { GraphFormatError } from "../../src/errors.js";

interface NaiveCore {
    rowPtr: number[];
    colIdx: number[];
    arcToEdge: number[];
    edgeToArc: number[];
    weights: number[] | null;
    multigraph: boolean;
}

/** The CSR of an edge list by a comparator sort over (source, target, edge). */
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
    return {
        rowPtr,
        colIdx: arcs.map((arc) => arc.target),
        arcToEdge: arcs.map((arc) => arc.edge),
        edgeToArc,
        weights:
            input.weights === null ? null : arcs.map((arc) => Math.fround((input.weights as Float64Array)[arc.edge])),
        multigraph,
    };
}

const arbInput: fc.Arbitrary<SortInput> = fc
    .record({
        directed: fc.boolean(),
        nodeCount: fc.integer({ min: 1, max: 12 }),
        weighted: fc.boolean(),
    })
    .chain(({ directed, nodeCount, weighted }) =>
        fc
            .array(
                fc.tuple(fc.nat({ max: nodeCount - 1 }), fc.nat({ max: nodeCount - 1 }), fc.double({ noNaN: true })),
                {
                    maxLength: 40,
                },
            )
            .map((edges) => ({
                directed,
                nodeCount,
                edgeCount: edges.length,
                src: Uint32Array.from(edges.map((e) => e[0])),
                dst: Uint32Array.from(edges.map((e) => e[1])),
                weights: weighted ? Float64Array.from(edges.map((e) => e[2])) : null,
            })),
    );

describe("sortIntoCore", () => {
    it("matches a comparator sort on random inputs, in and out of the arena", () => {
        fc.assert(
            fc.property(arbInput, fc.boolean(), (input, useArena) => {
                const expected = naive(input);
                const { core, arcCount, selfLoopCount, flags } = sortIntoCore(input, useArena);
                expect(arcCount).toBe(expected.colIdx.length);
                expect(selfLoopCount).toBe(countSelfLoops(input.src, input.dst, input.edgeCount));
                expect(Array.from(core.rowPtr)).toEqual(expected.rowPtr);
                expect(Array.from(core.colIdx)).toEqual(expected.colIdx);
                const identity = input.directed && isSortedEdgeList(input.src, input.dst, input.edgeCount);
                expect(flags.arcToEdgeIsIdentity).toBe(identity);
                expect(core.arcToEdge === null).toBe(identity);
                expect(core.edgeToArc === null).toBe(identity);
                if (core.arcToEdge !== null && core.edgeToArc !== null) {
                    expect(Array.from(core.arcToEdge)).toEqual(expected.arcToEdge);
                    expect(Array.from(core.edgeToArc)).toEqual(expected.edgeToArc);
                } else {
                    expect(expected.arcToEdge).toEqual(expected.arcToEdge.map((_, i) => i));
                }
                if (expected.weights === null) {
                    expect(core.weights).toBeNull();
                } else {
                    expect(Array.from(core.weights as Float32Array)).toEqual(expected.weights);
                    const w = expected.weights;
                    expect(flags.allWeightsOne).toBe(w.every((x) => x === 1));
                    expect(flags.nonNegativeWeights).toBe(w.every((x) => x >= 0));
                    expect(flags.finiteWeights).toBe(w.every((x) => Number.isFinite(x)));
                }
                expect(flags.multigraph).toBe(expected.multigraph);
                expect(flags.hasSelfLoops).toBe(selfLoopCount > 0);
                expect(flags.weighted).toBe(input.weights !== null);
                expect(core.arena === null).toBe(!useArena);
                if (core.arena !== null) {
                    expect(core.rowPtr.buffer).toBe(core.arena.buffer);
                }
            }),
            { numRuns: 300 },
        );
    });

    it("takes the identity path for sorted directed input", () => {
        const input: SortInput = {
            directed: true,
            nodeCount: 3,
            edgeCount: 4,
            src: new Uint32Array([0, 0, 1, 2]),
            dst: new Uint32Array([1, 1, 0, 2]),
            weights: new Float32Array([1, 1, 1, 1]),
        };
        const result = sortIntoCore(input, true);
        expect(result.flags.arcToEdgeIsIdentity).toBe(true);
        expect(result.flags.multigraph).toBe(true);
        expect(result.flags.allWeightsOne).toBe(true);
        expect(Array.from(result.core.rowPtr)).toEqual([0, 2, 3, 4]);
        expect(Array.from(result.core.colIdx)).toEqual([1, 1, 0, 2]);
        expect(result.core.arena?.segments.arcToEdge).toBeNull();
        expect(isSortedEdgeList(new Uint32Array([1, 0]), new Uint32Array([0, 0]), 2)).toBe(false);
        expect(isSortedEdgeList(new Uint32Array([0, 0]), new Uint32Array([2, 1]), 2)).toBe(false);
        expect(isSortedEdgeList(new Uint32Array(0), new Uint32Array(0), 0)).toBe(true);
    });

    it("never takes the identity path for undirected input", () => {
        const input: SortInput = {
            directed: false,
            nodeCount: 2,
            edgeCount: 2,
            src: new Uint32Array([0, 1]),
            dst: new Uint32Array([0, 1]),
            weights: null,
        };
        const result = sortIntoCore(input, true);
        expect(result.flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(result.core.arcToEdge as Uint32Array)).toEqual([0, 1]);
        expect(result.arcCount).toBe(2);
    });

    it("rejects NaN weights with E_INVALID_WEIGHT naming the edge", () => {
        const input: SortInput = {
            directed: true,
            nodeCount: 2,
            edgeCount: 2,
            src: new Uint32Array([1, 0]),
            dst: new Uint32Array([0, 1]),
            weights: new Float64Array([1, Number.NaN]),
        };
        let caught: GraphFormatError | null = null;
        try {
            sortIntoCore(input, true);
        } catch (err) {
            caught = err as GraphFormatError;
        }
        expect(caught?.code).toBe("E_INVALID_WEIGHT");
        expect(caught?.details.edge).toBe(1);
        const sorted: SortInput = { ...input, src: new Uint32Array([0, 1]), dst: new Uint32Array([1, 0]) };
        expect(() => sortIntoCore(sorted, false)).toThrow(GraphFormatError);
    });

    it("reads only the first edgeCount entries of longer staging views", () => {
        const input: SortInput = {
            directed: true,
            nodeCount: 2,
            edgeCount: 1,
            src: new Uint32Array([1, 0, 0, 0]),
            dst: new Uint32Array([0, 1, 1, 1]),
            weights: new Float32Array([2, 9, 9, 9]),
        };
        const result = sortIntoCore(input, false);
        expect(result.arcCount).toBe(1);
        expect(Array.from(result.core.rowPtr)).toEqual([0, 0, 1]);
        expect(Array.from(result.core.weights as Float32Array)).toEqual([2]);
    });
});
