import { describe, expect, it } from "vitest";

import { columnFromTypedArray, columnFromValues } from "../../src/columns/column.js";
import { createTable } from "../../src/columns/table.js";
import { INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { identityNodeIdMap, nodeIdMapFromIds } from "../../src/ids/node-id-map.js";
import { createSnapshot } from "../../src/snapshot/graph-snapshot.js";
import {
    checkColumnRules,
    checkI3,
    checkI6,
    checkI10,
    computeFlags,
    validateFull,
    validateStructure,
} from "../../src/snapshot/validate.js";
import { type ArenaLayout, type GraphSnapshot } from "../../src/types/index.js";
import { type GraphSpec, makeParts, makeSnapshot, type MutableParts } from "../helpers/parts.js";

function expectInvariant(fn: () => unknown, invariant: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe("E_INVALID_SNAPSHOT");
    expect(error.details.invariant).toBe(invariant);
    return error;
}

/** A snapshot whose listed public fields are overridden (prototype delegation), for checks the constructor would refuse. */
function fakeWith(s: GraphSnapshot, overrides: Readonly<Record<string, unknown>>): GraphSnapshot {
    const fake = Object.create(s) as GraphSnapshot;
    for (const [key, value] of Object.entries(overrides)) {
        Object.defineProperty(fake, key, { value, enumerable: true });
    }
    return fake;
}

/** Build a snapshot from parts, corrupting a field first; parts are copied so the arena views stay independent. */
function corrupted(spec: GraphSpec, mutate: (parts: MutableParts) => void): GraphSnapshot {
    const parts = makeParts(spec);
    mutate(parts);
    return createSnapshot(parts);
}

const UNDIRECTED: GraphSpec = {
    directed: false,
    edges: [
        [0, 1, 2],
        [1, 2, 3],
        [2, 2, 4],
        [0, 1, 5],
    ],
};
const DIRECTED: GraphSpec = {
    directed: true,
    edges: [
        [1, 0, 2],
        [0, 2, 3],
        [2, 2, 4],
        [0, 1, 5],
    ],
};

describe("validate levels", () => {
    it("accept every fixture at both levels and default to full", () => {
        for (const spec of [UNDIRECTED, DIRECTED, { directed: true, nodeCount: 0, edges: [] }]) {
            const s = makeSnapshot(spec);
            expect(() => s.validate()).not.toThrow();
            expect(() => s.validate({ level: "structure" })).not.toThrow();
            expect(() => s.validate({ level: "full" })).not.toThrow();
            expect(() => validateStructure(s)).not.toThrow();
            expect(() => validateFull(s)).not.toThrow();
        }
    });

    it("structure skips the O(m log d) checks that full runs", () => {
        const s = corrupted(DIRECTED, (parts) => {
            // swap two targets inside row 0 so the row is unsorted but every length and range holds
            const [a, b] = [parts.rowPtr[0], parts.rowPtr[0] + 1];
            const tmp = parts.colIdx[a];
            parts.colIdx[a] = parts.colIdx[b];
            parts.colIdx[b] = tmp;
        });
        expect(() => s.validate({ level: "structure" })).not.toThrow();
        expectInvariant(() => s.validate(), "I4");
    });
});

describe("invariant error codes and numbers", () => {
    it("I1: rowPtr shape", () => {
        expectInvariant(() => corrupted(UNDIRECTED, (p) => (p.rowPtr[0] = 1)).validate(), "I1");
        expectInvariant(
            () =>
                corrupted(UNDIRECTED, (p) => {
                    p.rowPtr[1] = 5;
                    p.rowPtr[2] = 2;
                }).validate(),
            "I1",
        );
        expectInvariant(() => corrupted(UNDIRECTED, (p) => (p.rowPtr[p.rowPtr.length - 1] = 1)).validate(), "I1");
        const error = expectInvariant(
            () => createSnapshot({ ...makeParts(UNDIRECTED), rowPtr: new Uint32Array(2) }),
            "I1",
        );
        expect(error.details.found).toBe(2);
    });

    it("I2: colIdx range", () => {
        const error = expectInvariant(
            () => corrupted(UNDIRECTED, (p) => (p.colIdx[3] = 99)).validate({ level: "structure" }),
            "I2",
        );
        expect(error.details.arc).toBe(3);
        expectInvariant(() => corrupted(UNDIRECTED, (p) => (p.colIdx[0] = INVALID_INDEX)).validate(), "I2");
    });

    it("I3: counts", () => {
        expectInvariant(() => corrupted(UNDIRECTED, (p) => (p.selfLoopCount = 40)).validate(), "I3");
        expectInvariant(() => corrupted(UNDIRECTED, (p) => (p.selfLoopCount = -1)).validate(), "I3");
        const s = makeSnapshot(UNDIRECTED);
        expectInvariant(() => checkI3(fakeWith(s, { nodeCount: MAX_COUNT + 1 })), "I3");
        expectInvariant(() => checkI3(fakeWith(s, { edgeCount: s.arcCount + 1 })), "I3");
        expectInvariant(() => checkI3(fakeWith(s, { arcCount: 1.5 })), "I3");
    });

    it("I4: sorted rows with ties in edge order", () => {
        const s = corrupted(UNDIRECTED, (p) => {
            // the two parallel arcs 0->1 in row 0: swap their edge order
            const a = p.rowPtr[0];
            const perm = p.arcToEdge as Uint32Array;
            const tmp = perm[a];
            perm[a] = perm[a + 1];
            perm[a + 1] = tmp;
        });
        const error = expectInvariant(() => s.validate(), "I4");
        expect(error.details.row).toBe(0);
    });

    it("I5: permutation lengths, ranges and orientation", () => {
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), arcToEdge: new Uint32Array(1) }), "I5");
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), edgeToArc: new Uint32Array(1) }), "I5");
        expectInvariant(
            () => corrupted(UNDIRECTED, (p) => ((p.arcToEdge as Uint32Array)[0] = 9)).validate({ level: "structure" }),
            "I5",
        );
        expectInvariant(
            () => corrupted(UNDIRECTED, (p) => ((p.edgeToArc as Uint32Array)[0] = 99)).validate({ level: "structure" }),
            "I5",
        );
        const error = expectInvariant(
            () =>
                corrupted(
                    UNDIRECTED,
                    (p) => ((p.edgeToArc as Uint32Array)[0] = (p.edgeToArc as Uint32Array)[1]),
                ).validate(),
            "I5",
        );
        expect(error.details.edge).toBe(0);
        // an identity claim on an undirected snapshot
        expectInvariant(
            () =>
                createSnapshot({
                    ...makeParts(UNDIRECTED),
                    arcToEdge: null,
                    edgeToArc: null,
                    flags: { ...makeParts(UNDIRECTED).flags, arcToEdgeIsIdentity: true },
                }),
            "I9",
        );
    });

    it("I6: directed arcCount === edgeCount and a permutation", () => {
        const directed = makeSnapshot(DIRECTED);
        expectInvariant(() => checkI6(fakeWith(directed, { edgeCount: 3 }), false), "I6");
        const s = corrupted(DIRECTED, (p) => {
            const perm = p.arcToEdge as Uint32Array;
            perm[1] = perm[0];
        });
        const error = expectInvariant(() => s.validate(), "I5");
        expect(error.code).toBe("E_INVALID_SNAPSHOT");
        // I5 orientation implies the permutation property, so the I6 duplicate check is reached directly
        const error6 = expectInvariant(() => checkI6(s, true), "I6");
        expect(error6.details.arc).toBe(1);
    });

    it("I7: undirected counts and pairing", () => {
        expectInvariant(
            () => corrupted(UNDIRECTED, (p) => (p.selfLoopCount = 0)).validate({ level: "structure" }),
            "I7",
        );
        const broken = corrupted(UNDIRECTED, (p) => {
            // make the two arcs of edge 1 (1-2) carry different weights
            const w = p.weights as Float32Array;
            w[p.rowPtr[1]] = 100;
        });
        const error = expectInvariant(() => broken.validate(), "I7");
        expect(error.details.arc).toBeDefined();
        const mismatch = corrupted(UNDIRECTED, (p) => {
            // point the mate arc of 1->2 at edge 0 instead of edge 1 (also breaks the occurrence count)
            const perm = p.arcToEdge as Uint32Array;
            const mateArc = p.rowPtr[2];
            perm[mateArc] = 0;
        });
        expectInvariant(() => mismatch.validate(), "I7");
    });

    it("I8: weights length and NaN", () => {
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), weights: new Float32Array(1) }), "I8");
        const nan = corrupted(UNDIRECTED, (p) => {
            const w = p.weights as Float32Array;
            const a = p.rowPtr[2];
            w[a] = NaN;
            w[a + 1] = NaN;
        });
        expect(() => nan.validate({ level: "structure" })).not.toThrow();
        expectInvariant(() => nan.validate(), "I8");
    });

    it("I9: truthful flags and selfLoopCount", () => {
        for (const flag of [
            "multigraph",
            "hasSelfLoops",
            "weighted",
            "allWeightsOne",
            "nonNegativeWeights",
            "finiteWeights",
        ] as const) {
            const s = corrupted(UNDIRECTED, (p) => (p.flags = { ...p.flags, [flag]: !p.flags[flag] }));
            const error = expectInvariant(() => s.validate(), "I9");
            expect(error.details.flag).toBe(flag);
        }
        const sorted = makeParts({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
        });
        expect(sorted.arcToEdge).toBeNull();
        const materialisedIdentity = createSnapshot({
            ...sorted,
            arcToEdge: new Uint32Array([0, 1]),
            edgeToArc: new Uint32Array([0, 1]),
            flags: { ...sorted.flags, arcToEdgeIsIdentity: false },
        });
        const error = expectInvariant(() => materialisedIdentity.validate(), "I9");
        expect(error.details.flag).toBe("arcToEdgeIsIdentity");
        const loops = corrupted(UNDIRECTED, (p) => {
            // turn the loop arc into a non-loop target without touching counts: I9 must notice
            p.colIdx[p.rowPtr[2] + p.rowPtr[3] - p.rowPtr[2] - 1] = 1;
        });
        const which = (() => {
            try {
                loops.validate();
            } catch (err) {
                return (err as GraphFormatError).details.invariant;
            }
            return null;
        })();
        expect(["I4", "I7", "I9"]).toContain(which);
    });

    it("I10: alignment and arena consistency", () => {
        const unaligned = new Uint32Array(new ArrayBuffer(64), 4, 5);
        unaligned.set([0, 2, 4, 6, 6]);
        const parts = makeParts({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 3],
                [3, 0],
            ],
            arena: false,
        });
        expect(() => createSnapshot(parts).validate()).not.toThrow();
        const badArena: ArenaLayout = {
            ...(makeParts(UNDIRECTED).arena as ArenaLayout),
        };
        const wrongSegment = corrupted(UNDIRECTED, (p) => {
            p.arena = { ...badArena, segments: { ...badArena.segments, colIdx: { byteOffset: 8, byteLength: 4 } } };
        });
        expectInvariant(() => wrongSegment.validate({ level: "structure" }), "I10");
        const wrongHot = corrupted(UNDIRECTED, (p) => {
            p.arena = { ...(p.arena as ArenaLayout), hotByteLength: 1 };
        });
        const error = expectInvariant(() => wrongHot.validate({ level: "structure" }), "I10");
        expect(error.details.reason).toBe("hotByteLength");
        const wrongAlignment = corrupted(UNDIRECTED, (p) => {
            p.arena = { ...(p.arena as ArenaLayout), alignment: 128 as unknown as 256 };
        });
        expectInvariant(() => wrongAlignment.validate({ level: "structure" }), "I10");
        const absent = corrupted(UNDIRECTED, (p) => {
            const arena = p.arena as ArenaLayout;
            p.weights = null;
            p.flags = { ...p.flags, weighted: false, allWeightsOne: true };
            p.arena = { ...arena };
        });
        expectInvariant(() => absent.validate({ level: "structure" }), "I10");
        const notInArena = corrupted(UNDIRECTED, (p) => {
            p.colIdx = p.colIdx.slice();
        });
        expectInvariant(() => notInArena.validate({ level: "structure" }), "I10");
        // a misaligned 4-byte view cannot be constructed at all (the typed-array guarantee I10 relies on)
        expect(() => new Uint32Array(new ArrayBuffer(16), 2, 3)).toThrow(RangeError);
        const misaligned = fakeWith(makeSnapshot({ ...DIRECTED, arena: false }), {
            weights: new Float32Array(new ArrayBuffer(20), 4, 4),
        });
        expect(() => checkI10(misaligned)).not.toThrow();
        const shortBytes = fakeWith(makeSnapshot({ ...DIRECTED, arena: false }), {
            weights: { buffer: new ArrayBuffer(8), byteOffset: 0, byteLength: 6, length: 4 },
        });
        expectInvariant(() => checkI10(shortBytes), "I10");
    });

    it("I11: id map size and bijection", () => {
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), ids: identityNodeIdMap(2) }), "I11");
        // a map whose lookups disagree with idOf: wrap a real map
        const real = nodeIdMapFromIds(["a", "b", "c"]);
        const liar = Object.create(real) as typeof real;
        Object.defineProperty(liar, "indexOf", { value: () => 0 });
        const s = createSnapshot({ ...makeParts(UNDIRECTED), ids: liar });
        const error = expectInvariant(() => s.validate(), "I11");
        expect(error.details.index).toBe(1);
    });

    it("I12: table row counts and column length rules", () => {
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), nodes: createTable("node", 2) }), "I12");
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), graph: createTable("graph", 2) }), "I12");
        const okay = makeSnapshot({
            ...UNDIRECTED,
            nodeColumns: {
                d: { values: ["x", "y", "x"], decl: { dtype: "dict" } },
                s: { values: ["p", "q", "r"], decl: { dtype: "string" } },
                l: { values: [[1], [2, 3], []], decl: { dtype: "list", itemDtype: "u32" } },
                j: { values: [1, "a", null], decl: { dtype: "json" } },
                b: { values: [true, false, true], decl: { dtype: "bool" } },
                ref: { values: [1, undefined, 0], decl: { dtype: "u32", refersTo: "node" } },
                refList: { values: [[0, 1], [], [2]], decl: { dtype: "list", itemDtype: "u32", refersTo: "node" } },
            },
            extensions: {
                ext: {
                    rowCount: 2,
                    columns: { element: { values: [3, 1], decl: { dtype: "u32", refersTo: "edge" } } },
                },
            },
        });
        expect(() => okay.validate()).not.toThrow();
        // corrupt a dictionary code
        const dict = okay.nodes.requireTyped("d", "dict");
        dict.codes[0] = 7;
        let error = expectInvariant(() => okay.validate(), "I12");
        expect(error.details.column).toBe("d");
        dict.codes[0] = 0;
        // corrupt a reference
        okay.nodes.requireTyped("ref", "u32").data[0] = 3;
        error = expectInvariant(() => okay.validate(), "I12");
        expect(error.details.column).toBe("ref");
        okay.nodes.requireTyped("ref", "u32").data[0] = INVALID_INDEX;
        error = expectInvariant(() => okay.validate(), "I12");
        expect(error.details.row).toBe(0);
        okay.nodes.requireTyped("ref", "u32").data[0] = 1;
        // a list child reference out of range
        const list = okay.nodes.requireTyped("refList", "list");
        const { child } = list;
        if (child.dtype === "u32") {
            child.data[0] = 9;
        }
        error = expectInvariant(() => okay.validate(), "I12");
        expect(error.details.column).toBe("refList");
        if (child.dtype === "u32") {
            child.data[0] = 0;
        }
        // an extension element out of range
        okay.extensions.get("ext")?.requireTyped("element", "u32").data.set([9], 0);
        error = expectInvariant(() => okay.validate(), "I12");
        expect(error.details.table).toBe("extensions[ext]");
        expect(() => okay.validate({ level: "structure" })).toThrow();
    });

    it("I12 column rules through checkColumnRules", () => {
        const s = makeSnapshot(UNDIRECTED);
        const good = columnFromTypedArray("node", 3, "x", new Float32Array(6), { components: 2 });
        expect(() => checkColumnRules("nodes", good, 3, s)).not.toThrow();
        expectInvariant(() => checkColumnRules("nodes", good, 4, s), "I12");
        const str = columnFromValues("node", 3, "s", ["a", "bb", "c"], { dtype: "string" });
        expect(() => checkColumnRules("nodes", str, 3, s)).not.toThrow();
        if (str.dtype === "string") {
            str.offsets[1] = 5;
        }
        expectInvariant(() => checkColumnRules("nodes", str, 3, s), "I12");
        const bool = columnFromValues("node", 40, "b", new Array<boolean>(40).fill(true), { dtype: "bool" });
        expect(() => checkColumnRules("nodes", bool, 40, s)).not.toThrow();
        const json = columnFromValues("node", 3, "j", [1, 2, 3], { dtype: "json" });
        expect(() => checkColumnRules("nodes", json, 3, s)).not.toThrow();
        const nullable = columnFromValues("node", 3, "n", [1, undefined, 3], { dtype: "u32" });
        expect(nullable.validity?.length).toBe(1);
        expect(() => checkColumnRules("nodes", nullable, 3, s)).not.toThrow();
    });

    it("I13: the edge table has edgeCount rows", () => {
        expectInvariant(() => createSnapshot({ ...makeParts(UNDIRECTED), edges: createTable("edge", 1) }), "I12");
        const s = makeSnapshot(UNDIRECTED);
        const fake = Object.create(s) as GraphSnapshot;
        Object.defineProperty(fake, "edges", { value: createTable("edge", 1) });
        expectInvariant(() => validateStructure(fake), "I13");
    });

    it("full validation enforces unique columns", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
            ],
            edgeColumns: { id: { values: ["x", "x"], decl: { dtype: "string", role: "id", unique: true } } },
        });
        let caught: unknown = null;
        try {
            s.validate();
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        // a validation failure is E_INVALID_SNAPSHOT (design sections 9.5 and 11.4); the freeze-time
        // code of the unique rule travels in details.cause
        expect((caught as GraphFormatError).code).toBe("E_INVALID_SNAPSHOT");
        expect((caught as GraphFormatError).details).toMatchObject({
            invariant: "I12",
            reason: "unique",
            cause: "E_DUPLICATE_EDGE_ID",
            column: "id",
        });
        expect(() => s.validate({ level: "structure" })).not.toThrow();
    });
});

describe("computeFlags", () => {
    it("computes every predicate of section 3.8 from the arrays", () => {
        const rowPtr = new Uint32Array([0, 2, 3]);
        const colIdx = new Uint32Array([1, 1, 0]);
        const base = { directed: true, nodeCount: 2, rowPtr, colIdx, selfLoopCount: 0 };
        expect(computeFlags({ ...base, weights: null, arcToEdge: null })).toEqual({
            multigraph: true,
            hasSelfLoops: false,
            arcToEdgeIsIdentity: true,
            weighted: false,
            allWeightsOne: true,
            nonNegativeWeights: true,
            finiteWeights: true,
        });
        const flags = computeFlags({
            ...base,
            weights: new Float32Array([1, -1, Infinity]),
            arcToEdge: new Uint32Array([1, 0, 2]),
            selfLoopCount: 1,
        });
        expect(flags).toEqual({
            multigraph: true,
            hasSelfLoops: true,
            arcToEdgeIsIdentity: false,
            weighted: true,
            allWeightsOne: false,
            nonNegativeWeights: false,
            finiteWeights: false,
        });
        expect(
            computeFlags({ ...base, directed: false, weights: null, arcToEdge: new Uint32Array([0, 1, 2]) })
                .arcToEdgeIsIdentity,
        ).toBe(false);
    });
});

describe("checksums (design section 5.8)", () => {
    it("throws no-checksum on a snapshot built without records", () => {
        const s = makeSnapshot(UNDIRECTED);
        let caught: unknown = null;
        try {
            s.validate({ checksum: true });
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(GraphFormatError);
        expect((caught as GraphFormatError).code).toBe("E_INVALID_SNAPSHOT");
        expect((caught as GraphFormatError).details.reason).toBe("no-checksum");
    });

    it("passes on an untouched snapshot and catches a write into a core array", () => {
        const s = makeSnapshot({ ...UNDIRECTED, checksum: true });
        s.prepare(["outDegree", "reverse", "coo", "edgeList", "degreeOrder", "mate", "selfLoopArcs"]);
        expect(() => s.validate({ checksum: true })).not.toThrow();
        const same = s.colIdx[0];
        s.colIdx[0] = same;
        expect(() => s.validate({ checksum: true })).not.toThrow();
        const w = s.weights as Float32Array;
        const original = w[0];
        w[0] = 42;
        let caught: unknown = null;
        try {
            s.validate({ checksum: true });
        } catch (err) {
            caught = err;
        }
        expect((caught as GraphFormatError).details.reason).toBe("checksum");
        expect((caught as GraphFormatError).details.array).toBe("weights");
        w[0] = original;
        expect(() => s.validate({ checksum: true })).not.toThrow();
    });

    it("catches a write into a shared view and into an immutable column", () => {
        const s = makeSnapshot({ ...UNDIRECTED, checksum: true, nodeColumns: { x: new Float32Array([1, 2, 3]) } });
        const degree = s.outDegree();
        degree[0] = 99;
        let caught: unknown = null;
        try {
            s.validate({ checksum: true });
        } catch (err) {
            caught = err;
        }
        expect((caught as GraphFormatError).details.view).toBe("outDegree");
        s.dropCaches();
        expect(() => s.validate({ checksum: true })).not.toThrow();
        s.nodes.requireTyped("x", "f32").data[1] = 0;
        caught = null;
        try {
            s.validate({ checksum: true });
        } catch (err) {
            caught = err;
        }
        expect((caught as GraphFormatError).details.column).toBe("x");
        // columns attached after freeze have no record and are not checked
        s.nodes.remove("x");
        s.nodes.set("y", new Float32Array([7, 8, 9]));
        s.nodes.requireTyped("y", "f32").data[0] = 1;
        expect(() => s.validate({ checksum: true })).not.toThrow();
    });

    it("records a lazily materialised identity permutation and a mutable column is never recorded", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
            checksum: true,
            nodeColumns: { pos: { data: new Float32Array(3), decl: { mutable: true } } },
        });
        const perm = s.arcToEdge;
        expect(() => s.validate({ checksum: true })).not.toThrow();
        perm[0] = 1;
        let caught: unknown = null;
        try {
            s.validate({ checksum: true });
        } catch (err) {
            caught = err;
        }
        expect((caught as GraphFormatError).details.array).toBe("arcToEdge");
        s.dropCaches();
        expect(() => s.validate({ checksum: true })).not.toThrow();
        s.nodes.requireTyped("pos", "f32").mutableData()[0] = 5;
        expect(() => s.validate({ checksum: true })).not.toThrow();
    });
});
