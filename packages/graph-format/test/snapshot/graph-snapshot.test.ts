import { describe, expect, it } from "vitest";

import { FORMAT_VERSION, INVALID_INDEX, SNAPSHOT_BRAND } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    CLONE_GUARD_KEY,
    createSnapshot,
    EMPTY_GRAPH_META,
    equalsTopology,
    GraphSnapshot,
    hasChecksums,
    isGraphSnapshot,
    peekView,
    permutationMaterialised,
} from "../../src/snapshot/graph-snapshot.js";
import { type ArenaLayout, type GraphSnapshotContract } from "../../src/types/index.js";
import { fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { assertInvariants } from "../helpers/invariants.js";
import { type GraphSpec, makeParts, makeSnapshot } from "../helpers/parts.js";

function expectError(fn: () => unknown, code: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe(code);
    return error;
}

const SPEC: GraphSpec = {
    directed: false,
    edges: [
        [0, 1, 2],
        [1, 2, 3],
        [2, 2, 4],
    ],
    label: "tri",
};

describe("createSnapshot and identity", () => {
    it("wraps parts into a frozen, branded snapshot with a fresh serial", () => {
        const a = makeSnapshot(SPEC);
        const b = makeSnapshot(SPEC);
        expect(a).toBeInstanceOf(GraphSnapshot);
        expect(Object.isFrozen(a)).toBe(true);
        expect(a[SNAPSHOT_BRAND]).toBe(true);
        expect(a.formatVersion).toBe(FORMAT_VERSION);
        expect(a.label).toBe("tri");
        expect(a.serial).not.toBe(b.serial);
        expect(a.meta).toBe(EMPTY_GRAPH_META);
        expect(a.detached).toBe(false);
        expect(a.extensions.size).toBe(0);
        expect(() => {
            (a as unknown as { nodeCount: number }).nodeCount = 5;
        }).toThrow(TypeError);
        expect(() => {
            (a as unknown as { extra: number }).extra = 5;
        }).toThrow(TypeError);
        assertInvariants(a);
    });

    it("isGraphSnapshot is structural: brand plus formatVersion, never instanceof", () => {
        const s = makeSnapshot(SPEC);
        expect(isGraphSnapshot(s)).toBe(true);
        expect(isGraphSnapshot(null)).toBe(false);
        expect(isGraphSnapshot(5)).toBe(false);
        expect(isGraphSnapshot({})).toBe(false);
        expect(isGraphSnapshot({ [SNAPSHOT_BRAND]: true, formatVersion: 2 })).toBe(false);
        expect(isGraphSnapshot({ [SNAPSHOT_BRAND]: true, formatVersion: 1 })).toBe(true);
        const lookalike = Object.create(null) as Record<symbol | string, unknown>;
        lookalike[SNAPSHOT_BRAND] = true;
        lookalike.formatVersion = 1;
        expect(isGraphSnapshot(lookalike)).toBe(true);
    });

    it("carries an enumerable own function property so structuredClone throws DataCloneError", () => {
        const s = makeSnapshot(SPEC);
        expect(Object.keys(s)).toContain(CLONE_GUARD_KEY);
        let caught: unknown = null;
        try {
            structuredClone(s);
        } catch (err) {
            caught = err;
        }
        expect(caught).toBeInstanceOf(Error);
        expect((caught as Error).name).toBe("DataCloneError");
    });

    it("shares a serial when parts say so and copies the flags into a frozen object", () => {
        const parts = makeParts(SPEC);
        const a = createSnapshot({ ...parts, serial: 12345 });
        expect(a.serial).toBe(12345);
        expect(a.flags).not.toBe(parts.flags);
        expect(a.flags).toEqual(parts.flags);
        expect(Object.isFrozen(a.flags)).toBe(true);
    });

    it("refuses inconsistent parts with E_INVALID_SNAPSHOT at construction", () => {
        const parts = makeParts(SPEC);
        expectError(() => createSnapshot({ ...parts, colIdx: new Uint32Array(1) }), "E_INVALID_SNAPSHOT");
        expectError(() => createSnapshot({ ...parts, arcToEdge: null }), "E_INVALID_SNAPSHOT");
        expect(
            expectError(
                () => createSnapshot({ ...parts, ids: makeParts({ directed: true, nodeCount: 1, edges: [] }).ids }),
                "E_INVALID_SNAPSHOT",
            ).details.invariant,
        ).toBe("I11");
        const directed = makeParts({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
        });
        expect(directed.arcToEdge).toBeNull();
        expectError(() => createSnapshot({ ...directed, arcToEdge: new Uint32Array([0, 1]) }), "E_INVALID_SNAPSHOT");
        expectError(
            () =>
                createSnapshot({
                    ...directed,
                    edgeCount: 1,
                    edges: makeParts({ directed: true, edges: [[0, 1]] }).edges,
                }),
            "E_INVALID_SNAPSHOT",
        );
    });
});

describe("identity permutations", () => {
    it("are materialised on first access outside the arena and never counted in byteLength", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1, 1],
                [0, 2, 1],
                [1, 2, 1],
            ],
        });
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        expect(permutationMaterialised(s)).toBe(false);
        expect(s.arena?.segments.arcToEdge).toBeNull();
        const before = s.byteLength();
        expect(before).toBe(s.rowPtr.byteLength + s.colIdx.byteLength + 12);
        const perm = s.arcToEdge;
        expect(Array.from(perm)).toEqual([0, 1, 2]);
        expect(Array.from(s.edgeToArc)).toEqual([0, 1, 2]);
        expect(permutationMaterialised(s)).toBe(true);
        expect(perm.buffer).not.toBe(s.arena?.buffer);
        expect(s.byteLength()).toBe(before);
        expect(s.arcToEdge).toBe(perm);
        s.dropCaches();
        expect(permutationMaterialised(s)).toBe(false);
        expect(s.arcToEdge).not.toBe(perm);
    });

    it("are supplied and counted when not identity", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [1, 0],
                [0, 1],
            ],
        });
        expect(s.flags.arcToEdgeIsIdentity).toBe(false);
        expect(permutationMaterialised(s)).toBe(true);
        expect(s.byteLength()).toBe(s.rowPtr.byteLength + s.colIdx.byteLength + 8 + 8);
        s.dropCaches();
        expect(permutationMaterialised(s)).toBe(true);
    });

    it("undirected snapshots always hold both arrays even when their contents are the identity", () => {
        const s = makeSnapshot({
            directed: false,
            edges: [
                [0, 0],
                [1, 1],
            ],
        });
        expect(s.flags.arcToEdgeIsIdentity).toBe(false);
        expect(permutationMaterialised(s)).toBe(true);
    });
});

describe("detached snapshots", () => {
    function detach(s: GraphSnapshot): void {
        const arena = s.arena as ArenaLayout;
        structuredClone(arena.buffer, { transfer: [arena.buffer] });
    }

    it("report detached and throw E_DETACHED from every accessor that reads the core", () => {
        const s = makeSnapshot(SPEC);
        detach(s);
        expect(s.detached).toBe(true);
        expect(s.rowPtr.length).toBe(0);
        for (const call of [
            () => s.arcToEdge,
            () => s.edgeToArc,
            () => s.outArcs(0),
            () => s.outDegreeOf(0),
            () => s.findArc(0, 1),
            () => s.arcsBetween(0, 1),
            () => s.multiplicity(0, 1),
            () => s.arcSource(0),
            () => s.selfLoopsAt(0),
            () => s.reverse(),
            () => s.outDegree(),
            () => s.mate(),
            () => s.contentHash(),
            () => s.validate(),
            () => s.toUndirected(),
            () => s.transpose(),
            () => s.simplified(),
            () => s.withoutSelfLoops(),
            () => s.filterEdges(new Uint32Array(1)),
            () => s.inducedSubgraph(new Uint32Array(0)),
            () => s.contract(new Uint32Array(3)),
            () => s.relabel(new Uint32Array([0, 1, 2])),
            () => s.withColumns(),
            () => s.toWire(),
            () => s.toBytes(),
            () => s.toByteChunks(),
            () => s.transferables(),
        ]) {
            expectError(call, "E_DETACHED");
        }
    });

    it("every holder of the same core agrees", () => {
        const s = makeSnapshot(SPEC);
        const w = s.withColumns();
        detach(s);
        expect(w.detached).toBe(true);
        expectError(() => w.outDegree(), "E_DETACHED");
    });
});

describe("byteLength", () => {
    it("counts the core, then views, columns and ids on request", () => {
        const s = makeSnapshot({
            directed: false,
            ids: ["a", "b", "c"],
            edges: [
                [0, 1, 2],
                [1, 2, 3],
            ],
            nodeColumns: { x: new Float32Array(3) },
            edgeColumns: { y: new Float64Array(2) },
            extensions: { t: { rowCount: 2, columns: { e: new Uint32Array(2) } } },
        });
        const core = 16 + 16 + 16 + 16 + 8;
        expect(s.byteLength()).toBe(core);
        expect(s.byteLength({ columns: true })).toBe(core + 12 + 16 + 8);
        expect(s.byteLength({ ids: true })).toBe(core + s.ids.byteLength());
        s.outDegree();
        s.reverse();
        expect(s.byteLength({ views: true })).toBe(core + 12);
        expect(s.reverse().fwdArc.length).toBe(4);
        expect(s.byteLength({ views: true })).toBe(core + 12 + 16);
        s.coo();
        s.edgeList();
        expect(s.byteLength({ views: true })).toBe(core + 12 + 16 + 16 + 8 + 8 + 8);
    });
});

describe("edgeIndexOf", () => {
    it("resolves ids through the role id column, follows replacements, misses with INVALID_INDEX", () => {
        const s = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            edgeColumns: { id: { values: ["e0", "e1", "e2"], decl: { dtype: "string", role: "id" } } },
        });
        expect(s.edgeIndexOf("e1")).toBe(1);
        expect(s.edgeIndexOf("nope")).toBe(INVALID_INDEX);
        expect(s.edgeIndexOf(1)).toBe(INVALID_INDEX);
        s.edges.set("id", [10, 20, 30], { dtype: "u32", role: "id" });
        expect(s.edgeIndexOf(20)).toBe(1);
        expect(s.edgeIndexOf("e1")).toBe(INVALID_INDEX);
        s.edges.remove("id");
        expect(s.edgeIndexOf(20)).toBe(INVALID_INDEX);
        const plain = makeSnapshot(SPEC);
        expect(plain.edgeIndexOf("x")).toBe(INVALID_INDEX);
    });
});

describe("equalsTopology", () => {
    it("compares directedness, counts, core arrays and ids element by element", () => {
        const a = makeSnapshot({
            directed: true,
            ids: ["x", "y", "z"],
            edges: [
                [0, 1, 1],
                [1, 2, 2],
            ],
        });
        const b = makeSnapshot({
            directed: true,
            ids: ["x", "y", "z"],
            edges: [
                [0, 1, 1],
                [1, 2, 2],
            ],
            arena: false,
        });
        expect(equalsTopology(a, b)).toBe(true);
        expect(
            equalsTopology(
                a,
                makeSnapshot({
                    directed: true,
                    ids: ["x", "y", "w"],
                    edges: [
                        [0, 1, 1],
                        [1, 2, 2],
                    ],
                }),
            ),
        ).toBe(false);
        expect(
            equalsTopology(
                a,
                makeSnapshot({
                    directed: true,
                    ids: ["x", "y", "z"],
                    edges: [
                        [0, 1, 1],
                        [1, 2, 3],
                    ],
                }),
            ),
        ).toBe(false);
        expect(
            equalsTopology(
                a,
                makeSnapshot({
                    directed: true,
                    ids: ["x", "y", "z"],
                    edges: [
                        [0, 1],
                        [1, 2],
                    ],
                }),
            ),
        ).toBe(false);
        expect(
            equalsTopology(
                a,
                makeSnapshot({
                    directed: false,
                    ids: ["x", "y", "z"],
                    edges: [
                        [0, 1, 1],
                        [1, 2, 2],
                    ],
                }),
            ),
        ).toBe(false);
        expect(
            equalsTopology(
                a,
                makeSnapshot({
                    directed: true,
                    ids: ["x", "y", "z"],
                    edges: [
                        [1, 2, 2],
                        [0, 1, 1],
                    ],
                }),
            ),
        ).toBe(false);
        expect(
            equalsTopology(a, makeSnapshot({ directed: true, ids: ["x", "y", "z"], nodeCount: 3, edges: [[0, 1, 1]] })),
        ).toBe(false);
        const identity = makeSnapshot({
            directed: true,
            edges: [
                [0, 1],
                [0, 2],
            ],
        });
        const explicit = makeSnapshot({
            directed: true,
            edges: [
                [0, 2],
                [0, 1],
            ],
        });
        expect(equalsTopology(identity, explicit)).toBe(false);
        expect(
            equalsTopology(
                identity,
                makeSnapshot({
                    directed: true,
                    edges: [
                        [0, 1],
                        [0, 2],
                    ],
                }),
            ),
        ).toBe(true);
    });
});

describe("wire methods (sections 9.1, 9.2)", () => {
    it("reach the wire module directly and round-trip the topology", () => {
        const s = makeSnapshot(SPEC);
        expect(s.transferables()).toContain(s.arena?.buffer);
        const wire = s.toWire();
        expect(wire.manifest.format).toBe("graphty-snapshot");
        expect(equalsTopology(fromWire(wire), s)).toBe(true);
        // fromWire adopted the same buffers in this realm, so the arena is now held twice
        expect(s.transferables()).not.toContain(s.arena?.buffer);
        const bytes = s.toBytes();
        expect(bytes.length % 256).toBe(0);
        expect(equalsTopology(fromBytes(bytes), s)).toBe(true);
        const chunks = [...s.toByteChunks()];
        expect(chunks.reduce((n, c) => n + c.length, 0)).toBe(bytes.length);
        expect(equalsTopology(fromByteChunks(chunks), s)).toBe(true);
    });

    it("throw E_DETACHED once the core was transferred away", () => {
        const s = makeSnapshot(SPEC);
        const wire = s.toWire({ transfer: true });
        structuredClone(wire, { transfer: s.transferables() });
        expect(s.detached).toBe(true);
        expectError(() => s.toWire(), "E_DETACHED");
        expectError(() => s.toBytes(), "E_DETACHED");
        expectError(() => s.toByteChunks(), "E_DETACHED");
        expectError(() => s.transferables(), "E_DETACHED");
    });
});

describe("misc surface", () => {
    it("peekView and hasChecksums expose the private state for tests and the wire module", () => {
        const s = makeSnapshot({ ...SPEC, checksum: true });
        expect(hasChecksums(s)).toBe(true);
        expect(hasChecksums(makeSnapshot(SPEC))).toBe(false);
        expect(peekView(s, "outDegree")).toBeNull();
        const degree = s.outDegree();
        expect(peekView(s, "outDegree")).toBe(degree);
        const notASnapshot = Object.create(GraphSnapshot.prototype) as GraphSnapshot;
        expect(peekView(notASnapshot, "outDegree")).toBeNull();
        expect(hasChecksums(notASnapshot)).toBe(false);
    });

    it("the class satisfies the public contract type", () => {
        const s: GraphSnapshotContract = makeSnapshot(SPEC);
        expect(s.nodeCount).toBe(3);
        expect(typeof s.reverse).toBe("function");
    });

    it("cachedViews reports names in the fixed order and never includes reverseDegreeOrder twice", () => {
        const s = makeSnapshot(SPEC);
        s.isSymmetric();
        s.degreeOrder({ of: "reverse" });
        s.coo();
        expect(s.cachedViews()).toEqual(["coo", "degreeOrder", "reverseDegreeOrder", "symmetric"]);
        expect(s.byteLength({ views: true })).toBe(s.byteLength() + 20 + 12 + 20);
    });

    it("dropCaches releases the cached gpuView() f32 copies of f64 columns (section 7.2)", () => {
        const s = makeSnapshot({ ...SPEC, nodeColumns: { score: new Float64Array([0.5, 1.5, 2.5]) } });
        const first = s.nodes.gpuView("score");
        expect(first).toBeInstanceOf(Float32Array);
        expect(s.nodes.gpuView("score")).toBe(first);
        s.dropCaches();
        const second = s.nodes.gpuView("score");
        expect(second).not.toBe(first);
        expect(Array.from(second)).toEqual([0.5, 1.5, 2.5]);
    });
});
