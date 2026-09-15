/**
 * The invariants I1-I18 of design section 3.2 as executable checks over any GraphSnapshot, reused by
 * test/invariants.test.ts and by the module tests after every construction. I1-I13 run the same
 * helpers validate() uses (src/snapshot/validate.ts) AND independent naive re-derivations; I14-I16
 * are builder properties and take the spec the fixture was built from (I15 as "same parts twice
 * give byte-identical cores"); I17 checks freezing, view sharing and (when recorded) checksums;
 * I18 cannot be observed without a builder and is asserted as "the snapshot's arrays are not the
 * spec's arrays".
 */

import { expect } from "vitest";

import { INVALID_INDEX, MAX_COUNT, SNAPSHOT_BRAND } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { contentHashOf } from "../../src/snapshot/hash.js";
import {
    checkI1,
    checkI2,
    checkI3,
    checkI4,
    checkI5Orientation,
    checkI5Ranges,
    checkI6,
    checkI7Counts,
    checkI7Pairing,
    checkI8Length,
    checkI8NaN,
    checkI9,
    checkI10,
    checkI11Bijection,
    checkI11Size,
    checkI12,
    checkI13,
} from "../../src/snapshot/validate.js";
import { type GraphSnapshot } from "../../src/types/index.js";
import { type GraphSpec, makeSnapshot, naiveCsr, naiveFlags, naiveOutArcs, weightOf } from "./parts.js";

/** One executable invariant: throws (an Error or a GraphFormatError) when violated. */
type InvariantCheck = (snapshot: GraphSnapshot) => void;

function fail(invariant: string, message: string): never {
    throw new Error(`${invariant}: ${message}`);
}

/** I1-I13 as the validate() helpers plus naive re-derivations. */
const INVARIANTS: Readonly<Record<string, InvariantCheck>> = {
    I1: (s) => {
        checkI1(s);
        let total = 0;
        for (let u = 0; u < s.nodeCount; u++) {
            total += s.rowPtr[u + 1] - s.rowPtr[u];
        }
        if (total !== s.arcCount) {
            fail("I1", "row lengths do not sum to arcCount");
        }
    },
    I2: (s) => {
        checkI2(s);
        const scan = (name: string, array: ArrayLike<number>): void => {
            for (let i = 0; i < array.length; i++) {
                if (array[i] === INVALID_INDEX) {
                    fail("I2", `${name}[${i}] is INVALID_INDEX`);
                }
            }
        };
        scan("rowPtr", s.rowPtr);
        scan("colIdx", s.colIdx);
        scan("arcToEdge", s.arcToEdge);
        scan("edgeToArc", s.edgeToArc);
        scan("coo.src", s.coo().src);
        scan("edgeList.src", s.edgeList().src);
        scan("edgeList.dst", s.edgeList().dst);
        scan("degreeOrder.perm", s.degreeOrder().perm);
        if (!s.directed) {
            scan("mate", s.mate());
        }
    },
    I3: (s) => {
        checkI3(s);
        expect(s.nodeCount).toBeLessThanOrEqual(MAX_COUNT);
        expect(s.edgeCount).toBeLessThanOrEqual(s.arcCount);
    },
    I4: (s) => {
        checkI4(s);
        for (let u = 0; u < s.nodeCount; u++) {
            for (let a = s.rowPtr[u] + 1; a < s.rowPtr[u + 1]; a++) {
                const sameTarget = s.colIdx[a] === s.colIdx[a - 1];
                if (s.colIdx[a] < s.colIdx[a - 1] || (sameTarget && s.arcToEdge[a] <= s.arcToEdge[a - 1])) {
                    fail("I4", `row ${u} unsorted at arc ${a}`);
                }
            }
        }
    },
    I5: (s) => {
        checkI5Ranges(s);
        checkI5Orientation(s);
        for (let e = 0; e < s.edgeCount; e++) {
            const a = s.edgeToArc[e];
            if (s.arcToEdge[a] !== e) {
                fail("I5", `arcToEdge[edgeToArc[${e}]] !== ${e}`);
            }
            if (s.edgeSource(e) !== s.arcSource(a) || s.edgeTarget(e) !== s.colIdx[a]) {
                fail("I5", `edge ${e} orientation disagrees with its arc`);
            }
        }
    },
    I6: (s) => {
        checkI6(s, true);
        if (s.directed) {
            expect(s.arcCount).toBe(s.edgeCount);
            const seen = new Set<number>();
            for (let a = 0; a < s.arcCount; a++) {
                seen.add(s.arcToEdge[a]);
            }
            expect(seen.size).toBe(s.edgeCount);
        }
    },
    I7: (s) => {
        checkI7Counts(s);
        checkI7Pairing(s);
        if (!s.directed) {
            expect(s.arcCount).toBe(2 * s.edgeCount - s.selfLoopCount);
            const mate = s.mate();
            for (let a = 0; a < s.arcCount; a++) {
                const b = mate[a];
                if (mate[b] !== a || s.arcToEdge[a] !== s.arcToEdge[b]) {
                    fail("I7", `mate(mate(${a})) !== ${a} or edges differ`);
                }
                if (s.colIdx[a] !== s.arcSource(b) || s.colIdx[b] !== s.arcSource(a)) {
                    fail("I7", `arc ${a} and its mate ${b} are not opposite orientations`);
                }
                if (s.weights !== null && s.weights[a] !== s.weights[b]) {
                    fail("I7", `arc ${a} and its mate ${b} carry different weights`);
                }
                if (s.colIdx[a] === s.arcSource(a) && b !== a) {
                    fail("I7", `self-loop arc ${a} is not its own mate`);
                }
            }
            const reverse = s.reverse();
            expect(reverse.rowPtr).toBe(s.rowPtr);
            expect(reverse.colIdx).toBe(s.colIdx);
            expect(reverse.weights).toBe(s.weights);
        }
    },
    I8: (s) => {
        checkI8Length(s);
        checkI8NaN(s);
        if (s.weights !== null) {
            expect(s.weights.length).toBe(s.arcCount);
        }
    },
    I9: (s) => {
        checkI9(s);
        const { flags } = s;
        expect(flags.weighted).toBe(s.weights !== null);
        expect(flags.hasSelfLoops).toBe(s.selfLoopCount > 0);
        if (!s.directed) {
            expect(flags.arcToEdgeIsIdentity).toBe(false);
        }
        if (s.weights !== null) {
            const w = Array.from(s.weights);
            expect(flags.allWeightsOne).toBe(w.every((x) => x === 1));
            expect(flags.nonNegativeWeights).toBe(w.every((x) => x >= 0));
            expect(flags.finiteWeights).toBe(w.every((x) => Number.isFinite(x)));
        } else {
            expect(flags.allWeightsOne).toBe(true);
            expect(flags.nonNegativeWeights).toBe(true);
            expect(flags.finiteWeights).toBe(true);
        }
        let parallel = false;
        for (let u = 0; u < s.nodeCount; u++) {
            for (let a = s.rowPtr[u] + 1; a < s.rowPtr[u + 1]; a++) {
                parallel ||= s.colIdx[a] === s.colIdx[a - 1];
            }
        }
        expect(flags.multigraph).toBe(parallel);
        expect(Object.isFrozen(flags)).toBe(true);
    },
    I10: (s) => {
        checkI10(s);
        const aligned = (name: string, view: ArrayBufferView, multiple = 4): void => {
            if (!(view.buffer instanceof ArrayBuffer)) {
                fail("I10", `${name} is not over a plain ArrayBuffer`);
            }
            if (view.byteOffset % multiple !== 0 || view.byteLength % 4 !== 0) {
                fail("I10", `${name} is not ${multiple}-byte aligned`);
            }
        };
        aligned("rowPtr", s.rowPtr);
        aligned("colIdx", s.colIdx);
        aligned("arcToEdge", s.arcToEdge);
        aligned("edgeToArc", s.edgeToArc);
        if (s.weights !== null) {
            aligned("weights", s.weights);
        }
        aligned("outDegree", s.outDegree());
        aligned("inDegree", s.inDegree());
        aligned("degree", s.degree());
        aligned("coo.src", s.coo().src);
        aligned("edgeList.src", s.edgeList().src);
        aligned("edgeList.dst", s.edgeList().dst);
        aligned("selfLoopArcs", s.selfLoopArcs());
        aligned("selfLoopsPerNode", s.selfLoopsPerNode());
        aligned("degreeOrder.perm", s.degreeOrder().perm);
        aligned("degreeOrder.segmentOffsets", s.degreeOrder().segmentOffsets);
        aligned("reverse.rowPtr", s.reverse().rowPtr);
        aligned("reverse.colIdx", s.reverse().colIdx);
        aligned("reverse.fwdArc", s.reverse().fwdArc);
        aligned("weightedOutDegree", s.weightedOutDegree(), 8);
        aligned("weightedInDegree", s.weightedInDegree(), 8);
        aligned("weightedDegree", s.weightedDegree(), 8);
        aligned("selfLoopWeight", s.selfLoopWeight(), 8);
        if (!s.directed) {
            aligned("mate", s.mate());
        }
        if (s.arena !== null) {
            expect(s.arena.alignment).toBe(256);
            for (const [name, segment] of Object.entries(s.arena.segments)) {
                if (segment !== null) {
                    expect((segment.byteOffset - s.arena.byteOffset) % 256).toBe(0);
                    expect(segment.byteLength).toBeGreaterThan(0);
                } else {
                    const array = (s as unknown as Record<string, ArrayBufferView | null>)[name];
                    const identity = s.flags.arcToEdgeIsIdentity && (name === "arcToEdge" || name === "edgeToArc");
                    if (array !== null && !identity && array.byteLength > 0 && array.buffer === s.arena.buffer) {
                        fail("I10", `${name} lives in the arena but has no segment`);
                    }
                }
            }
        }
    },
    I11: (s) => {
        checkI11Size(s);
        checkI11Bijection(s);
        expect(s.ids.size).toBe(s.nodeCount);
        const seen = new Set<unknown>();
        for (let i = 0; i < s.nodeCount; i++) {
            const id = s.ids.idOf(i);
            expect(Number.isNaN(id)).toBe(false);
            expect(seen.has(id)).toBe(false);
            seen.add(id);
            expect(s.ids.indexOf(id)).toBe(i);
        }
    },
    I12: (s) => {
        checkI12(s);
        expect(s.nodes.rowCount).toBe(s.nodeCount);
        expect(s.edges.rowCount).toBe(s.edgeCount);
        expect(s.graph.rowCount).toBe(1);
        for (const table of [s.nodes, s.edges, s.graph, ...s.extensions.values()]) {
            for (const column of table) {
                expect(column.length).toBe(table.rowCount);
            }
        }
    },
    I13: (s) => {
        checkI13(s);
        for (const column of s.edges) {
            expect(column.length).toBe(s.edgeCount);
        }
    },
    I17: (s) => {
        expect(Object.isFrozen(s)).toBe(true);
        expect(s[SNAPSHOT_BRAND]).toBe(true);
        const views: readonly (() => unknown)[] = [
            () => s.reverse(),
            () => s.coo(),
            () => s.edgeList(),
            () => s.outDegree(),
            () => s.inDegree(),
            () => s.degree(),
            () => s.weightedOutDegree(),
            () => s.weightedInDegree(),
            () => s.weightedDegree(),
            () => s.selfLoopWeight(),
            () => s.selfLoopArcs(),
            () => s.selfLoopsPerNode(),
            () => s.degreeOrder(),
        ];
        for (const view of views) {
            expect(view()).toBe(view());
        }
        if (!s.directed) {
            expect(s.mate()).toBe(s.mate());
            expect(s.inDegree()).toBe(s.outDegree());
            expect(s.weightedInDegree()).toBe(s.weightedOutDegree());
            expect(s.degreeOrder({ of: "reverse" })).toBe(s.degreeOrder());
        }
        expect(s.arcToEdge).toBe(s.arcToEdge);
        expect(s.edgeToArc).toBe(s.edgeToArc);
        expect(() => {
            (s as unknown as { rowPtr: unknown }).rowPtr = null;
        }).toThrow(TypeError);
    },
};

/**
 * Run I1-I13 and I17 on a snapshot; the first failure throws.
 */
export function assertInvariants(snapshot: GraphSnapshot): void {
    for (const [name, check] of Object.entries(INVARIANTS)) {
        try {
            check(snapshot);
        } catch (err) {
            const detail = err instanceof GraphFormatError ? `${err.code} ${JSON.stringify(err.details)}` : "";
            throw new Error(`invariant ${name} failed: ${err instanceof Error ? err.message : String(err)} ${detail}`);
        }
    }
    snapshot.validate({ level: "full" });
}

/**
 * Check a snapshot against the spec it was built from: the neighbour structure (I14 order: node
 * index = spec index, edge index = spec order), I15 determinism (a second construction from the same
 * spec has byte-identical core arrays and the same content hash) and I18 (no aliasing with the naive
 * construction's arrays).
 */
export function assertMatchesSpec(snapshot: GraphSnapshot, spec: GraphSpec): void {
    assertInvariants(snapshot);
    const naive = naiveCsr(spec);
    expect(snapshot.nodeCount).toBe(naive.nodeCount);
    expect(snapshot.edgeCount).toBe(naive.edgeCount);
    expect(snapshot.arcCount).toBe(naive.arcCount);
    expect(snapshot.selfLoopCount).toBe(naive.selfLoopCount);
    expect(Array.from(snapshot.rowPtr)).toEqual(Array.from(naive.rowPtr));
    expect(Array.from(snapshot.colIdx)).toEqual(Array.from(naive.colIdx));
    expect(Array.from(snapshot.arcToEdge)).toEqual(Array.from(naive.arcToEdge));
    expect(Array.from(snapshot.edgeToArc)).toEqual(Array.from(naive.edgeToArc));
    if (naive.weights === null) {
        expect(snapshot.weights).toBeNull();
    } else {
        expect(snapshot.weights).not.toBeNull();
        expect(Array.from(snapshot.weights as Float32Array)).toEqual(Array.from(naive.weights));
    }
    expect(snapshot.flags).toEqual(naiveFlags(spec.directed, naive));
    // I14: declared orientation and edge order follow the spec
    spec.edges.forEach((edge, e) => {
        expect(snapshot.edgeSource(e)).toBe(edge[0]);
        expect(snapshot.edgeTarget(e)).toBe(edge[1]);
        if (snapshot.weights !== null) {
            expect(snapshot.weights[snapshot.edgeToArc[e]]).toBe(Math.fround(weightOf(edge)));
        }
    });
    for (let u = 0; u < snapshot.nodeCount; u++) {
        const [start, end] = snapshot.outArcs(u);
        const actual: [number, number][] = [];
        for (let a = start; a < end; a++) {
            actual.push([snapshot.colIdx[a], snapshot.arcToEdge[a]]);
        }
        expect(actual).toEqual(naiveOutArcs(spec, u));
    }
    // I15: determinism
    const again = makeSnapshot(spec);
    expect(Array.from(again.rowPtr)).toEqual(Array.from(snapshot.rowPtr));
    expect(Array.from(again.colIdx)).toEqual(Array.from(snapshot.colIdx));
    expect(again.contentHash()).toBe(snapshot.contentHash());
    expect(contentHashOf(again)).toBe(snapshot.contentHash());
    // I18: the snapshot does not alias the naive construction's arrays
    expect(snapshot.rowPtr).not.toBe(naive.rowPtr);
}
