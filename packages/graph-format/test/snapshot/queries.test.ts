import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { INVALID_INDEX } from "../../src/constants.js";
import {
    arcRangeIn,
    arcSourceIn,
    findArcIn,
    lowerBound,
    multiplicityIn,
    upperBound,
} from "../../src/snapshot/queries.js";
import { type GraphSpec, makeSnapshot } from "../helpers/parts.js";

describe("lowerBound / upperBound", () => {
    const sorted = new Uint32Array([1, 3, 3, 3, 7, 9]);
    it("find the first >= and the first > within a range", () => {
        expect(lowerBound(sorted, 0, 6, 3)).toBe(1);
        expect(upperBound(sorted, 0, 6, 3)).toBe(4);
        expect(lowerBound(sorted, 0, 6, 0)).toBe(0);
        expect(lowerBound(sorted, 0, 6, 10)).toBe(6);
        expect(upperBound(sorted, 0, 6, 9)).toBe(6);
        expect(lowerBound(sorted, 2, 4, 3)).toBe(2);
        expect(lowerBound(sorted, 4, 4, 3)).toBe(4);
    });
});

describe("findArc / arcsBetween / multiplicity (design section 3.5 semantics)", () => {
    // row 0: 0->1 (e0), 0->1 (e1), 0->1 (e3), 0->2 (e4); row 1: 1->0 (e2)
    const spec: GraphSpec = {
        directed: true,
        edges: [
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1, 3],
            [0, 2],
        ],
    };
    const s = makeSnapshot(spec);

    it("findArc returns the FIRST matching arc, the lowest logical edge among parallels", () => {
        const a = s.findArc(0, 1);
        expect(a).toBe(0);
        expect(s.arcToEdge[a]).toBe(0);
        expect(s.findArc(0, 2)).toBe(3);
        expect(s.findArc(1, 0)).toBe(4);
        expect(s.findArc(1, 2)).toBe(INVALID_INDEX);
        expect(s.findArc(2, 0)).toBe(INVALID_INDEX);
    });

    it("hasArc mirrors findArc", () => {
        expect(s.hasArc(0, 1)).toBe(true);
        expect(s.hasArc(2, 1)).toBe(false);
    });

    it("arcsBetween is the half-open range and multiplicity its length; total, never throws", () => {
        expect(s.arcsBetween(0, 1)).toEqual([0, 3]);
        expect(s.multiplicity(0, 1)).toBe(3);
        expect(s.arcsBetween(0, 2)).toEqual([3, 4]);
        const [lo, hi] = s.arcsBetween(2, 2);
        expect(lo).toBe(hi);
        expect(s.multiplicity(2, 2)).toBe(0);
        expect(s.arcsBetween(0, 1)).not.toBe(s.arcsBetween(0, 1));
    });

    it("outArcs and outDegreeOf read rowPtr", () => {
        expect(s.outArcs(0)).toEqual([0, 4]);
        expect(s.outArcs(2)).toEqual([5, 5]);
        expect(s.outDegreeOf(0)).toBe(4);
        expect(s.outDegreeOf(2)).toBe(0);
    });

    it("out-of-range indices are unchecked and yield a miss rather than throwing", () => {
        expect(s.findArc(7, 1)).toBe(INVALID_INDEX);
        expect(s.hasArc(0, 99)).toBe(false);
        expect(() => s.multiplicity(99, 0)).not.toThrow();
        expect(() => s.arcsBetween(99, 0)).not.toThrow();
    });
});

describe("arcSource / edgeSource / edgeTarget", () => {
    it("arcSource skips empty rows by binary search and is O(1) once coo() is cached", () => {
        const s = makeSnapshot({
            directed: true,
            nodeCount: 6,
            edges: [
                [0, 1],
                [0, 2],
                [3, 3],
                [5, 0],
            ],
        });
        // rows: 0 -> [0,2), 1 -> [2,2), 2 -> [2,2), 3 -> [2,3), 4 -> [3,3), 5 -> [3,4)
        expect(arcSourceIn(s.rowPtr, 0)).toBe(0);
        expect(arcSourceIn(s.rowPtr, 1)).toBe(0);
        expect(arcSourceIn(s.rowPtr, 2)).toBe(3);
        expect(arcSourceIn(s.rowPtr, 3)).toBe(5);
        for (let a = 0; a < s.arcCount; a++) {
            expect(s.arcSource(a)).toBe(arcSourceIn(s.rowPtr, a));
        }
        s.coo();
        for (let a = 0; a < s.arcCount; a++) {
            expect(s.arcSource(a)).toBe(s.coo().src[a]);
        }
    });

    it("edgeSource / edgeTarget give the declared orientation, on undirected snapshots too", () => {
        const spec: GraphSpec = {
            directed: false,
            edges: [
                [3, 1],
                [1, 3],
                [2, 2],
                [0, 3],
            ],
        };
        const s = makeSnapshot(spec);
        spec.edges.forEach(([u, v], e) => {
            expect(s.edgeSource(e)).toBe(u);
            expect(s.edgeTarget(e)).toBe(v);
        });
        s.edgeList();
        spec.edges.forEach(([u, v], e) => {
            expect(s.edgeSource(e)).toBe(u);
            expect(s.edgeTarget(e)).toBe(v);
        });
    });

    it("selfLoopsAt counts loop arcs at one node", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 0],
                [0, 0],
                [0, 1],
                [1, 1],
            ],
        });
        expect(s.selfLoopsAt(0)).toBe(2);
        expect(s.selfLoopsAt(1)).toBe(1);
        expect(s.selfLoopsAt(2)).toBe(0);
    });
});

describe("queries agree with a linear scan (P6)", () => {
    const edgesArb = fc.integer({ min: 1, max: 8 }).chain((n) =>
        fc
            .array(fc.tuple(fc.integer({ min: 0, max: n - 1 }), fc.integer({ min: 0, max: n - 1 })), {
                maxLength: 30,
            })
            .map((edges) => ({ n, edges: edges.map(([u, v]) => [u, v] as const) })),
    );

    for (const directed of [true, false]) {
        it(directed ? "directed" : "undirected", () => {
            fc.assert(
                fc.property(edgesArb, ({ n, edges }) => {
                    const s = makeSnapshot({ directed, nodeCount: n, edges });
                    for (let u = 0; u < n; u++) {
                        for (let v = 0; v < n; v++) {
                            let first = INVALID_INDEX;
                            let count = 0;
                            for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                                if (s.colIdx[a] === v) {
                                    count++;
                                    if (first === INVALID_INDEX) {
                                        first = a;
                                    }
                                }
                            }
                            expect(findArcIn(s.rowPtr, s.colIdx, u, v)).toBe(first);
                            expect(multiplicityIn(s.rowPtr, s.colIdx, u, v)).toBe(count);
                            const [lo, hi] = arcRangeIn(s.rowPtr, s.colIdx, u, v);
                            expect(hi - lo).toBe(count);
                            if (count > 0) {
                                expect(lo).toBe(first);
                            }
                        }
                    }
                }),
                { numRuns: 40 },
            );
        });
    }
});
