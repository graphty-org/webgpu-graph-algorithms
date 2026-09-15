/**
 * Adversarial audit of validate() (design sections 3.2, 9.5, 11.4): an INDEPENDENT naive checker
 * of I1-I9 and I11 is run next to validate() over randomly corrupted cores, so that every verdict
 * validate() gives is cross-checked (never a pass on a broken core, never a non-GraphFormatError,
 * always an invariant number the naive checker agrees with), plus the level table of 9.5 and the
 * checks 9.5 lists that the implementation does not perform.
 *
 * Tests marked "PINS DEFECT" are expected to FAIL against the current implementation; each names
 * the finding it pins.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { createSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type GraphSnapshot } from "../../src/types/index.js";
import { type GraphSpec, makeParts } from "../helpers/parts.js";

const RUNS = Number(process.env.FC_RUNS ?? "200");

/** The invariants an independent reading of design section 3.2 finds violated on a snapshot. */
function naiveViolations(s: GraphSnapshot): Set<string> {
    const out = new Set<string>();
    const { nodeCount, edgeCount, arcCount, rowPtr, colIdx, weights, directed } = s;
    // I3
    for (const count of [nodeCount, edgeCount, arcCount, s.selfLoopCount]) {
        if (!Number.isInteger(count) || count < 0 || count > MAX_COUNT) {
            out.add("I3");
        }
    }
    if (edgeCount > arcCount) {
        out.add("I3");
    }
    // I1
    if (
        rowPtr.length !== nodeCount + 1 ||
        rowPtr[0] !== 0 ||
        rowPtr[nodeCount] !== arcCount ||
        colIdx.length !== arcCount
    ) {
        out.add("I1");
    }
    for (let u = 0; u < nodeCount; u++) {
        if (rowPtr[u + 1] < rowPtr[u]) {
            out.add("I1");
        }
    }
    if (out.has("I1")) {
        return out; // rows cannot be walked
    }
    // I2
    for (let a = 0; a < arcCount; a++) {
        if (colIdx[a] >= nodeCount) {
            out.add("I2");
        }
    }
    const identity = s.flags.arcToEdgeIsIdentity;
    const arcToEdge = identity ? Uint32Array.from({ length: arcCount }, (_, a) => a) : s.arcToEdge;
    const edgeToArc = identity ? Uint32Array.from({ length: edgeCount }, (_, e) => e) : s.edgeToArc;
    // I5 lengths and ranges
    if (arcToEdge.length !== arcCount || edgeToArc.length !== edgeCount) {
        out.add("I5");
        return out;
    }
    for (let a = 0; a < arcCount; a++) {
        if (arcToEdge[a] >= edgeCount) {
            out.add("I5");
        }
    }
    for (let e = 0; e < edgeCount; e++) {
        if (edgeToArc[e] >= arcCount) {
            out.add("I5");
        }
    }
    if (identity && (!directed || arcCount !== edgeCount)) {
        out.add("I5");
    }
    if (out.has("I2") || out.has("I5")) {
        return out;
    }
    // I4
    for (let u = 0; u < nodeCount; u++) {
        for (let a = rowPtr[u] + 1; a < rowPtr[u + 1]; a++) {
            if (colIdx[a] < colIdx[a - 1] || (colIdx[a] === colIdx[a - 1] && arcToEdge[a] <= arcToEdge[a - 1])) {
                out.add("I4");
            }
        }
    }
    // I5 orientation
    for (let e = 0; e < edgeCount; e++) {
        if (arcToEdge[edgeToArc[e]] !== e) {
            out.add("I5");
        }
    }
    const rowOf = new Uint32Array(arcCount);
    for (let u = 0; u < nodeCount; u++) {
        for (let a = rowPtr[u]; a < rowPtr[u + 1]; a++) {
            rowOf[a] = u;
        }
    }
    let loopArcs = 0;
    for (let a = 0; a < arcCount; a++) {
        if (colIdx[a] === rowOf[a]) {
            loopArcs++;
        }
    }
    // I6 / I7
    const arcsOf: number[][] = Array.from({ length: edgeCount }, () => []);
    for (let a = 0; a < arcCount; a++) {
        arcsOf[arcToEdge[a]].push(a);
    }
    if (directed) {
        if (arcCount !== edgeCount || arcsOf.some((arcs) => arcs.length !== 1)) {
            out.add("I6");
        }
    } else {
        if (arcCount !== 2 * edgeCount - s.selfLoopCount) {
            out.add("I7");
        }
        for (const arcs of arcsOf) {
            if (arcs.length === 1) {
                if (colIdx[arcs[0]] !== rowOf[arcs[0]]) {
                    out.add("I7");
                }
            } else if (arcs.length === 2) {
                const [a, b] = arcs;
                const opposite = rowOf[a] === colIdx[b] && rowOf[b] === colIdx[a] && rowOf[a] !== colIdx[a];
                if (!opposite || (weights !== null && weights[a] !== weights[b])) {
                    out.add("I7");
                }
            } else {
                out.add("I7");
            }
        }
    }
    // I8
    if (weights !== null && weights.length !== arcCount) {
        out.add("I8");
    }
    if (weights !== null) {
        for (let a = 0; a < weights.length; a++) {
            if (Number.isNaN(weights[a])) {
                out.add("I8");
            }
        }
    }
    // I9
    let multigraph = false;
    for (let u = 0; u < nodeCount; u++) {
        for (let a = rowPtr[u] + 1; a < rowPtr[u + 1]; a++) {
            multigraph ||= colIdx[a] === colIdx[a - 1];
        }
    }
    const w = weights === null ? [] : Array.from(weights);
    const expected = {
        multigraph,
        hasSelfLoops: s.selfLoopCount > 0,
        arcToEdgeIsIdentity: directed && Array.from(arcToEdge).every((e, a) => e === a),
        weighted: weights !== null,
        allWeightsOne: w.every((x) => x === 1),
        nonNegativeWeights: w.every((x) => x >= 0),
        finiteWeights: w.every((x) => Number.isFinite(x)),
    };
    if (loopArcs !== s.selfLoopCount) {
        out.add("I9");
    }
    for (const [name, value] of Object.entries(expected)) {
        if (s.flags[name as keyof typeof expected] !== value) {
            out.add("I9");
        }
    }
    // I11
    if (s.ids.size !== nodeCount) {
        out.add("I11");
    }
    for (let i = 0; i < Math.min(nodeCount, s.ids.size); i++) {
        if (s.ids.indexOf(s.ids.idOf(i)) !== i) {
            out.add("I11");
        }
    }
    return out;
}

type CoreArray = "colIdx" | "arcToEdge" | "edgeToArc" | "rowPtr" | "weights";

interface Corruption {
    readonly array: CoreArray;
    readonly position: number;
    readonly value: number;
}

const arbCorruption: fc.Arbitrary<Corruption> = fc.record({
    array: fc.constantFrom<CoreArray>("colIdx", "arcToEdge", "edgeToArc", "rowPtr", "weights"),
    position: fc.nat({ max: 30 }),
    value: fc.oneof(fc.nat({ max: 12 }), fc.constant(INVALID_INDEX), fc.constant(NaN), fc.constant(-3)),
});

const SPECS: readonly GraphSpec[] = [
    {
        directed: true,
        edges: [
            [0, 1, 2],
            [1, 2, 3],
            [2, 2, 4],
            [0, 1, 5],
            [3, 0, 1],
        ],
    },
    {
        directed: true,
        edges: [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [1, 2, 5],
            [2, 2, 1],
        ],
    }, // sorted: identity
    {
        directed: false,
        edges: [
            [0, 1, 2],
            [1, 0, 3],
            [1, 1, 4],
            [0, 2, 5],
            [2, 0, 5],
            [3, 3, 1],
        ],
    },
    {
        directed: false,
        edges: [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
    },
    { directed: true, edges: [], nodeCount: 3 },
];

function corrupt(spec: GraphSpec, c: Corruption): GraphSnapshot | null {
    const parts = makeParts(spec);
    const target = parts[c.array] as Uint32Array | Float32Array | null;
    if (target === null || target.length === 0) {
        return null;
    }
    const at = c.position % target.length;
    if (target instanceof Float32Array) {
        target[at] = c.value;
    } else {
        target[at] = c.value < 0 ? INVALID_INDEX : c.value;
    }
    let snapshot: GraphSnapshot | null = null;
    try {
        snapshot = createSnapshot(parts);
    } catch (err) {
        // the constructor's O(1) checks refuse some corruptions outright
        expect(err).toBeInstanceOf(GraphFormatError);
        expect((err as GraphFormatError).code).toBe("E_INVALID_SNAPSHOT");
        return null;
    }
    return snapshot;
}

describe("validate() versus an independent naive checker over randomly corrupted cores", () => {
    it("full: throws E_INVALID_SNAPSHOT naming an invariant the naive checker also finds violated, and passes only when the naive checker passes", () => {
        fc.assert(
            fc.property(fc.constantFrom(...SPECS), arbCorruption, (spec, c) => {
                const s = corrupt(spec, c);
                if (s === null) {
                    return;
                }
                const naive = naiveViolations(s);
                let verdict: unknown = null;
                try {
                    s.validate({ level: "full" });
                } catch (err) {
                    verdict = err;
                }
                if (verdict === null) {
                    expect([...naive], `validate passed but the naive checker found ${[...naive].join(",")}`).toEqual(
                        [],
                    );
                    return;
                }
                expect(verdict).toBeInstanceOf(GraphFormatError);
                const error = verdict as GraphFormatError;
                expect(error.code).toBe("E_INVALID_SNAPSHOT");
                expect(
                    naive.size,
                    `validate reported ${String(error.details.invariant)} but the naive checker passes`,
                ).toBeGreaterThan(0);
                expect([...naive]).toContain(error.details.invariant);
            }),
            { numRuns: RUNS * 3 },
        );
    });

    it("structure: never throws for a corruption that only the O(m log d) checks of 9.5 can see, and never passes one the O(n + m) checks must catch", () => {
        const structural = new Set(["I1", "I2", "I3", "I8", "I10", "I11", "I12", "I13"]);
        fc.assert(
            fc.property(fc.constantFrom(...SPECS), arbCorruption, (spec, c) => {
                const s = corrupt(spec, c);
                if (s === null) {
                    return;
                }
                const naive = naiveViolations(s);
                let verdict: unknown = null;
                try {
                    s.validate({ level: "structure" });
                } catch (err) {
                    verdict = err;
                }
                const structuralOnly = [...naive].filter((i) => structural.has(i));
                if (verdict === null) {
                    // I8 NaN and I11 bijection are full-level; only lengths / ranges / counts are structural
                    const mustCatch = structuralOnly.filter((i) => i !== "I8" && i !== "I11");
                    expect(mustCatch, `structure passed but ${mustCatch.join(",")} is an O(n + m) check`).toEqual([]);
                    return;
                }
                expect(verdict).toBeInstanceOf(GraphFormatError);
                expect(naive.size).toBeGreaterThan(0);
            }),
            { numRuns: RUNS * 2 },
        );
    });

    it("a valid snapshot of every spec passes both levels and the naive checker", () => {
        for (const spec of SPECS) {
            const s = createSnapshot(makeParts(spec));
            expect([...naiveViolations(s)]).toEqual([]);
            s.validate({ level: "structure" });
            s.validate({ level: "full" });
            const built = new GraphBuilder({ directed: spec.directed });
            built.addAnonymousNodes(spec.nodeCount ?? Math.max(-1, ...spec.edges.flatMap((e) => [e[0], e[1]])) + 1);
            for (const e of spec.edges) {
                built.addEdgeByIndex(e[0], e[1], e.length === 3 ? e[2] : undefined);
            }
            expect([...naiveViolations(built.freeze())]).toEqual([]);
        }
    });
});

describe("checks design section 9.5 lists for the structure level", () => {
    it("PINS DEFECT (finding: nullCount not recomputed): a mutable column whose validity bitmap was edited without markDirty() keeps a stale nullCount that validate() accepts", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge(0, 1);
        b.addEdge(1, 2);
        b.addEdge(2, 0);
        const s = b.freeze();
        s.nodes.set("m", [1, undefined, 3], { mutable: true, nullable: true, dtype: "i32" });
        const column = s.nodes.require("m");
        expect(column.nullCount).toBe(1);
        const validity = column.mutableValidity();
        expect(validity).not.toBeNull();
        (validity as Uint32Array)[0] = 0; // every row unset now: 3 nulls
        expect(column.isSet(0)).toBe(false);
        expect(column.nullCount).toBe(1); // stale by construction
        // 9.5 "structure": "validity length, nullCount recomputed from the bitmap"
        expect(() => s.validate({ level: "structure" })).toThrow(
            expect.objectContaining({
                code: "E_INVALID_SNAPSHOT",
                details: expect.objectContaining({ invariant: "I12" }),
            }),
        );
    });

    it("PINS DEFECT (finding: I10 plain-ArrayBuffer rule unchecked): a core array over a SharedArrayBuffer passes validate()", () => {
        const parts = makeParts({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
            arena: false,
        });
        const shared = new Uint32Array(new SharedArrayBuffer(parts.colIdx.byteLength));
        shared.set(parts.colIdx);
        parts.colIdx = shared as unknown as Uint32Array<ArrayBuffer>;
        const s = createSnapshot(parts);
        // I10: "a view over a plain ArrayBuffer"
        expect(() => s.validate({ level: "structure" })).toThrow(
            expect.objectContaining({
                code: "E_INVALID_SNAPSHOT",
                details: expect.objectContaining({ invariant: "I10" }),
            }),
        );
    });

    it("column-level checks that ARE performed: length, validity words, dictionary codes, refersTo range and unset rule", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge(0, 1);
        b.addEdge(1, 2);
        b.addEdge(2, 0);
        const s = b.freeze();
        s.nodes.set("ref", new Uint32Array([1, 2, 0]), { refersTo: "node", mutable: true });
        const ref = s.nodes.requireTyped("ref", "u32");
        expect(() => s.validate({ level: "structure" })).not.toThrow();
        ref.mutableData()[1] = 3; // not below nodeCount
        expect(() => s.validate({ level: "structure" })).toThrow(
            expect.objectContaining({
                code: "E_INVALID_SNAPSHOT",
                details: expect.objectContaining({ invariant: "I12", column: "ref" }),
            }),
        );
        ref.mutableData()[1] = INVALID_INDEX; // INVALID_INDEX on a SET row (nullable false -> every row set)
        expect(() => s.validate({ level: "structure" })).toThrow(
            expect.objectContaining({
                code: "E_INVALID_SNAPSHOT",
                details: expect.objectContaining({ invariant: "I12", column: "ref", row: 1 }),
            }),
        );
        s.nodes.remove("ref");
        s.nodes.set("d", ["a", "b", "a"], { dtype: "dict", mutable: true });
        const dict = s.nodes.requireTyped("d", "dict");
        expect(() => s.validate({ level: "structure" })).not.toThrow();
        dict.mutableData()[2] = 99;
        expect(() => s.validate({ level: "structure" })).toThrow(
            expect.objectContaining({
                code: "E_INVALID_SNAPSHOT",
                details: expect.objectContaining({ invariant: "I12", column: "d", row: 2 }),
            }),
        );
    });
});

describe("I3 near MAX_COUNT (design 11.3: E_TOO_LARGE at the add* call that crosses the limit)", () => {
    it("addAnonymousNodes refuses to cross MAX_COUNT before allocating anything and leaves the builder untouched", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(3);
        const mutations = b.mutationCount;
        expect(() => b.addAnonymousNodes(MAX_COUNT - 3 + 1)).toThrow(expect.objectContaining({ code: "E_TOO_LARGE" }));
        expect(() => b.addAnonymousNodes(Number.MAX_SAFE_INTEGER)).toThrow(
            expect.objectContaining({ code: "E_TOO_LARGE" }),
        );
        expect(b.nodeBound).toBe(3);
        expect(b.mutationCount).toBe(mutations);
        expect(() => b.reserve(MAX_COUNT + 1)).toThrow(expect.objectContaining({ code: "E_TOO_LARGE" }));
        expect(() => b.reserve(0, MAX_COUNT + 1)).toThrow(expect.objectContaining({ code: "E_TOO_LARGE" }));
        expect(() => b.reserve(-1)).toThrow(expect.objectContaining({ code: "E_TOO_LARGE" }));
        expect(() => b.addAnonymousNodes(-1)).toThrow(expect.objectContaining({ code: "E_INDEX_RANGE" }));
        expect(() => b.addAnonymousNodes(1.5)).toThrow(expect.objectContaining({ code: "E_INDEX_RANGE" }));
        expect(b.nodeBound).toBe(3);
    });

    it("validate() rejects counts above MAX_COUNT and count relations that cannot hold, whatever the arrays say", () => {
        const s = createSnapshot(makeParts({ directed: true, edges: [[0, 1]] }));
        const fake = (overrides: Record<string, unknown>): GraphSnapshot => {
            const f = Object.create(s) as GraphSnapshot;
            for (const [key, value] of Object.entries(overrides)) {
                Object.defineProperty(f, key, { value, enumerable: true });
            }
            return f;
        };
        for (const overrides of [
            { nodeCount: MAX_COUNT + 1 },
            { edgeCount: MAX_COUNT + 1 },
            { arcCount: MAX_COUNT + 1 },
            { selfLoopCount: MAX_COUNT + 1 },
            { selfLoopCount: 2 },
            { edgeCount: 2 },
            { nodeCount: -1 },
            { arcCount: 1.5 },
            { arcCount: INVALID_INDEX },
        ]) {
            expect(() => fake(overrides).validate({ level: "structure" })).toThrow(
                expect.objectContaining({
                    code: "E_INVALID_SNAPSHOT",
                    details: expect.objectContaining({ invariant: "I3" }),
                }),
            );
        }
    });
});
