/**
 * Unit tests of fromCsr (design sections 8.1, 9.5, 10.3 and 11.4): adoption by reference versus
 * copying (byte-level identity), arena detection of an already-aligned shared buffer, the derived
 * arcToEdge / edgeToArc / edgeCount, the identity fast path and its edgeList aliasing, the sortRows
 * rebuild through the freeze pipeline, every validation level, flag claims, ids, columns, meta, and
 * every rejection code with its invariant.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { ALIGNMENT, INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { detectArena, fromCsr, resolveGraphMeta } from "../../src/populate/from-csr.js";
import { fromEdgeArrays } from "../../src/populate/from-edge-arrays.js";
import { permutationMaterialised } from "../../src/snapshot/graph-snapshot.js";
import { type CsrInput, type GraphSnapshot, type U32 } from "../../src/types/index.js";
import { assertInvariants, assertMatchesSpec } from "../helpers/invariants.js";
import { type EdgeSpec, type GraphSpec, gridEdges, KARATE_EDGES, naiveCsr } from "../helpers/parts.js";

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

/** The CsrInput of a spec, from the naive construction (fresh arrays every call). */
function csrOf(spec: GraphSpec, extra: Partial<CsrInput> = {}): CsrInput {
    const naive = naiveCsr(spec);
    return {
        directed: spec.directed,
        nodeCount: naive.nodeCount,
        rowPtr: naive.rowPtr,
        colIdx: naive.colIdx,
        weights: naive.weights,
        arcToEdge: naive.arcToEdge,
        edgeToArc: naive.edgeToArc,
        edgeCount: naive.edgeCount,
        ...extra,
    };
}

/** Lay the core arrays of a spec out in one 256-aligned buffer, arena order, at a base offset. */
function packedCsr(spec: GraphSpec, base = 0): { input: CsrInput; buffer: ArrayBuffer } {
    const naive = naiveCsr(spec);
    const identity = spec.directed && naive.arcToEdge.every((e, a) => e === a);
    const parts: (Uint32Array | Float32Array | null)[] = [
        naive.rowPtr,
        naive.colIdx,
        naive.weights,
        identity ? null : naive.arcToEdge,
        identity ? null : naive.edgeToArc,
    ];
    let cursor = base;
    const offsets: (number | null)[] = [];
    for (const part of parts) {
        if (part === null || part.byteLength === 0) {
            offsets.push(null);
            continue;
        }
        cursor = base + Math.ceil((cursor - base) / ALIGNMENT) * ALIGNMENT;
        offsets.push(cursor);
        cursor += part.byteLength;
    }
    const buffer = new ArrayBuffer(cursor + 64);
    const view = (i: number, ctor: typeof Uint32Array | typeof Float32Array): Uint32Array | Float32Array | null => {
        const part = parts[i];
        const offset = offsets[i];
        if (part === null) {
            return null;
        }
        if (offset === null) {
            return new ctor(0);
        }
        const out = new ctor(buffer, offset, part.length);
        out.set(part);
        return out;
    };
    const input: CsrInput = {
        directed: spec.directed,
        nodeCount: naive.nodeCount,
        rowPtr: view(0, Uint32Array) as U32,
        colIdx: view(1, Uint32Array) as U32,
        weights: view(2, Float32Array) as Float32Array<ArrayBuffer> | null,
        arcToEdge: (view(3, Uint32Array) as U32 | null) ?? undefined,
        edgeToArc: (view(4, Uint32Array) as U32 | null) ?? undefined,
        edgeCount: naive.edgeCount,
    };
    return { input, buffer };
}

function expectSameCore(a: GraphSnapshot, b: GraphSnapshot): void {
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
}

const WEIGHTED_EDGES: readonly EdgeSpec[] = [
    [0, 1, 2],
    [1, 2, 0.5],
    [2, 0, 3],
    [0, 0, 1],
    [1, 2, 4],
    [3, 1, -1],
];

const SPECS: readonly [string, GraphSpec][] = [
    ["directed karate", { directed: true, edges: KARATE_EDGES }],
    ["undirected karate", { directed: false, edges: KARATE_EDGES }],
    ["directed weighted with loop and parallels", { directed: true, edges: WEIGHTED_EDGES }],
    ["undirected weighted with loop and parallels", { directed: false, edges: WEIGHTED_EDGES }],
    ["directed grid (identity)", { directed: true, edges: gridEdges(4, 3) }],
    ["undirected grid with isolates", { directed: false, nodeCount: 20, edges: gridEdges(4, 3) }],
    [
        "undirected all self-loops",
        {
            directed: false,
            edges: [
                [0, 0],
                [1, 1],
                [2, 2],
            ],
        },
    ],
    ["empty directed", { directed: true, nodeCount: 0, edges: [] }],
    ["nodes only undirected", { directed: false, nodeCount: 5, edges: [] }],
];

describe("fromCsr: round trips", () => {
    it.each(SPECS)("adopts a valid CSR and matches the spec and the builder: %s", (_name, spec) => {
        const snapshot = fromCsr(csrOf(spec));
        assertMatchesSpec(snapshot, spec);
        const src = new Uint32Array(spec.edges.map((e) => e[0]));
        const dst = new Uint32Array(spec.edges.map((e) => e[1]));
        const weighted = spec.edges.some((e) => e.length === 3);
        const weights = weighted ? new Float32Array(spec.edges.map((e) => (e.length === 3 ? e[2] : 1))) : undefined;
        const nodeCount = spec.nodeCount ?? snapshot.nodeCount;
        expectSameCore(snapshot, fromEdgeArrays({ directed: spec.directed, nodeCount, src, dst, weights }));
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.label).toBeNull();
        expect(snapshot.meta.name).toBeNull();
        expect(snapshot.graph.rowCount).toBe(1);
        expect(snapshot.extensions.size).toBe(0);
    });

    it("round-trips a builder snapshot through its own arrays", () => {
        const builder = new GraphBuilder({ directed: false });
        builder.addEdge("a", "b", 2);
        builder.addEdge("b", "c", 3);
        builder.addEdge("c", "a");
        builder.addEdge("a", "a", 0.25);
        const frozen = builder.freeze();
        const snapshot = fromCsr({
            directed: false,
            nodeCount: frozen.nodeCount,
            rowPtr: frozen.rowPtr,
            colIdx: frozen.colIdx,
            weights: frozen.weights,
            arcToEdge: frozen.arcToEdge,
            edgeToArc: frozen.edgeToArc,
            ids: frozen.ids.toArray(),
        });
        assertInvariants(snapshot);
        expectSameCore(snapshot, frozen);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        // the builder's arena is detected as the adopted snapshot's arena
        expect(snapshot.arena?.buffer).toBe(frozen.arena?.buffer);
        expect(snapshot.arena).toEqual(frozen.arena);
    });
});

describe("fromCsr: adoption versus copy", () => {
    const spec: GraphSpec = { directed: false, edges: WEIGHTED_EDGES };

    it("adopts every array by reference by default with arena null for separate buffers", () => {
        const input = csrOf(spec);
        const snapshot = fromCsr(input);
        expect(snapshot.rowPtr).toBe(input.rowPtr);
        expect(snapshot.colIdx).toBe(input.colIdx);
        expect(snapshot.weights).toBe(input.weights);
        expect(snapshot.arcToEdge).toBe(input.arcToEdge);
        expect(snapshot.edgeToArc).toBe(input.edgeToArc);
        expect(snapshot.arena).toBeNull();
        assertInvariants(snapshot);
    });

    it("copies into a fresh arena under copy: true, leaving the caller's arrays untouched", () => {
        const input = csrOf(spec);
        const before = Array.from(input.colIdx);
        const snapshot = fromCsr(input, { copy: true });
        assertInvariants(snapshot);
        expect(snapshot.rowPtr).not.toBe(input.rowPtr);
        expect(snapshot.colIdx).not.toBe(input.colIdx);
        expect(snapshot.weights).not.toBe(input.weights);
        expect(snapshot.arcToEdge).not.toBe(input.arcToEdge);
        expect(snapshot.edgeToArc).not.toBe(input.edgeToArc);
        expect(snapshot.rowPtr.buffer).not.toBe(input.rowPtr.buffer);
        expect(Array.from(snapshot.colIdx)).toEqual(before);
        expect(snapshot.arena).not.toBeNull();
        expect(snapshot.arena?.byteOffset).toBe(0);
        expect(snapshot.colIdx.buffer).toBe(snapshot.arena?.buffer);
        expect(snapshot.arena?.segments.colIdx?.byteOffset).toBe(ALIGNMENT);
        // mutating the caller's array afterwards does not reach the snapshot
        input.colIdx[0] = 99;
        expect(snapshot.colIdx[0]).toBe(before[0]);
    });

    it("copies columns and F64 ids under copy: true and adopts them otherwise", () => {
        const size = new Float32Array([1, 2, 3, 4]);
        const ids = new Float64Array([10, 20, 30, 40]);
        const adopted = fromCsr(csrOf(spec, { nodeColumns: { size }, ids }));
        expect(adopted.nodes.requireTyped("size", "f32").data).toBe(size);
        expect(adopted.ids.kind).toBe("numeric");
        const copied = fromCsr(csrOf(spec, { nodeColumns: { size }, ids }), { copy: true });
        const column = copied.nodes.requireTyped("size", "f32");
        expect(column.data).not.toBe(size);
        expect(Array.from(column.data)).toEqual([1, 2, 3, 4]);
        expect(copied.ids.toArray()).toEqual([10, 20, 30, 40]);
        ids[0] = 11;
        expect(copied.ids.idOf(0)).toBe(10);
        expect(adopted.ids.idOf(0)).toBe(11);
    });

    it("never adopts a SharedArrayBuffer-backed array", () => {
        const naive = naiveCsr(spec);
        const shared = new SharedArrayBuffer(naive.colIdx.byteLength);
        const colIdx = new Uint32Array(shared);
        colIdx.set(naive.colIdx);
        const snapshot = fromCsr(csrOf(spec, { colIdx: colIdx as unknown as U32 }));
        assertInvariants(snapshot);
        expect(snapshot.colIdx).not.toBe(colIdx);
        expect(snapshot.colIdx.buffer).toBeInstanceOf(ArrayBuffer);
        expect(snapshot.arena).not.toBeNull();
        expect(Array.from(snapshot.colIdx)).toEqual(Array.from(naive.colIdx));
    });
});

describe("fromCsr: arena detection", () => {
    it("adopts one aligned shared buffer as the arena", () => {
        for (const spec of [
            { directed: false, edges: WEIGHTED_EDGES },
            { directed: true, edges: KARATE_EDGES },
            { directed: true, edges: gridEdges(4, 3) },
        ] as GraphSpec[]) {
            const { input, buffer } = packedCsr(spec);
            const snapshot = fromCsr(input);
            assertInvariants(snapshot);
            expect(snapshot.arena).not.toBeNull();
            expect(snapshot.arena?.buffer).toBe(buffer);
            expect(snapshot.arena?.byteOffset).toBe(0);
            expect(snapshot.arena?.alignment).toBe(ALIGNMENT);
            expect(snapshot.rowPtr).toBe(input.rowPtr);
            expect(snapshot.colIdx).toBe(input.colIdx);
            const { segments } = snapshot.arena as NonNullable<GraphSnapshot["arena"]>;
            expect(segments.rowPtr).toEqual({ byteOffset: 0, byteLength: input.rowPtr.byteLength });
            expect(segments.colIdx?.byteOffset).toBe(input.colIdx.byteOffset);
            if (spec.directed && snapshot.flags.arcToEdgeIsIdentity) {
                expect(segments.arcToEdge).toBeNull();
                expect(segments.edgeToArc).toBeNull();
            } else {
                expect(segments.arcToEdge?.byteOffset).toBe(input.arcToEdge?.byteOffset);
                expect(segments.edgeToArc?.byteOffset).toBe(input.edgeToArc?.byteOffset);
            }
            const hot = input.weights === null || input.weights === undefined ? input.colIdx : input.weights;
            expect(snapshot.arena?.hotByteLength).toBe(hot.byteOffset + hot.byteLength);
            const last = input.edgeToArc ?? hot;
            expect(snapshot.arena?.byteLength).toBe(last.byteOffset + last.byteLength);
        }
    });

    it("measures offsets relative to the first array, not the buffer", () => {
        const spec: GraphSpec = { directed: false, edges: WEIGHTED_EDGES };
        const { input, buffer } = packedCsr(spec, 1024 + 128);
        // rowPtr sits at 1152, which is not 256-aligned inside the buffer but is the arena origin
        expect(input.rowPtr.byteOffset).toBe(1152);
        const snapshot = fromCsr(input);
        assertInvariants(snapshot);
        expect(snapshot.arena?.buffer).toBe(buffer);
        expect(snapshot.arena?.byteOffset).toBe(1152);
        expect(snapshot.arena?.segments.colIdx?.byteOffset).toBe(1152 + ALIGNMENT);
        expect(snapshot.arena?.hotByteLength).toBe(
            (input.weights as Float32Array).byteOffset + (input.weights as Float32Array).byteLength - 1152,
        );
    });

    it("gives arena null when the arrays share a buffer at unaligned offsets or out of order", () => {
        const naive = naiveCsr({ directed: false, edges: WEIGHTED_EDGES });
        const total =
            4 * (naive.rowPtr.length + naive.colIdx.length + 2 * naive.arcToEdge.length + naive.edgeToArc.length);
        const buffer = new ArrayBuffer(total);
        let offset = 0;
        const take = (length: number, ctor: typeof Uint32Array | typeof Float32Array): Uint32Array | Float32Array => {
            const out = new ctor(buffer, offset, length);
            offset += 4 * length;
            return out;
        };
        const rowPtr = take(naive.rowPtr.length, Uint32Array) as U32;
        rowPtr.set(naive.rowPtr);
        const colIdx = take(naive.colIdx.length, Uint32Array) as U32;
        colIdx.set(naive.colIdx);
        const weights = take(naive.arcToEdge.length, Float32Array) as Float32Array<ArrayBuffer>;
        weights.set(naive.weights as Float32Array);
        const arcToEdge = take(naive.arcToEdge.length, Uint32Array) as U32;
        arcToEdge.set(naive.arcToEdge);
        const edgeToArc = take(naive.edgeToArc.length, Uint32Array) as U32;
        edgeToArc.set(naive.edgeToArc);
        const unaligned = fromCsr({
            directed: false,
            nodeCount: naive.nodeCount,
            rowPtr,
            colIdx,
            weights,
            arcToEdge,
            edgeToArc,
        });
        assertInvariants(unaligned);
        expect(unaligned.arena).toBeNull();
        expect(unaligned.colIdx).toBe(colIdx);
        // out of order: weights before colIdx in one aligned buffer
        expect(
            detectArena({
                rowPtr: new Uint32Array(new ArrayBuffer(1024), 0, 2),
                colIdx: new Uint32Array(new ArrayBuffer(1024), 512, 2),
                weights: null,
                arcToEdge: null,
                edgeToArc: null,
            }),
        ).toBeNull();
        const shared = new ArrayBuffer(1024);
        expect(
            detectArena({
                rowPtr: new Uint32Array(shared, 0, 2),
                colIdx: new Uint32Array(shared, 512, 2),
                weights: new Float32Array(shared, 256, 2),
                arcToEdge: null,
                edgeToArc: null,
            }),
        ).toBeNull();
        const ordered = detectArena({
            rowPtr: new Uint32Array(shared, 0, 2),
            colIdx: new Uint32Array(shared, 256, 2),
            weights: new Float32Array(shared, 512, 2),
            arcToEdge: null,
            edgeToArc: null,
        });
        expect(ordered?.hotByteLength).toBe(520);
        expect(ordered?.byteLength).toBe(520);
        expect(ordered?.segments.weights).toEqual({ byteOffset: 512, byteLength: 8 });
    });

    it("gives arena null when a derived edgeToArc lives outside the shared buffer", () => {
        const { input } = packedCsr({ directed: false, edges: WEIGHTED_EDGES });
        const snapshot = fromCsr({ ...input, edgeToArc: undefined });
        assertInvariants(snapshot);
        expect(snapshot.arena).toBeNull();
    });
});

describe("fromCsr: derived arrays and counts", () => {
    it("treats an absent arcToEdge as the identity of a directed graph and keeps it lazy", () => {
        const naive = naiveCsr({ directed: true, edges: gridEdges(3, 3) });
        const snapshot = fromCsr({
            directed: true,
            nodeCount: naive.nodeCount,
            rowPtr: naive.rowPtr,
            colIdx: naive.colIdx,
            weights: naive.weights,
        });
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(true);
        expect(snapshot.edgeCount).toBe(naive.edgeCount);
        expect(permutationMaterialised(snapshot)).toBe(false);
        assertInvariants(snapshot);
        expect(Array.from(snapshot.arcToEdge)).toEqual(Array.from(naive.arcToEdge));
        expect(Array.from(snapshot.edgeToArc)).toEqual(Array.from(naive.edgeToArc));
    });

    it("aliases edgeList().weights to weights for a directed identity input", () => {
        const weights = new Float32Array([1, 2, 3]);
        const snapshot = fromCsr({
            directed: true,
            nodeCount: 3,
            rowPtr: new Uint32Array([0, 2, 3, 3]),
            colIdx: new Uint32Array([1, 2, 0]),
            weights,
        });
        expect(snapshot.edgeList().weights).toBe(weights);
        expect(snapshot.edgeList().dst).toBe(snapshot.colIdx);
    });

    it("drops an explicit identity arcToEdge (and an identity edgeToArc) for a directed graph", () => {
        const naive = naiveCsr({ directed: true, edges: gridEdges(3, 3) });
        const snapshot = fromCsr(csrOf({ directed: true, edges: gridEdges(3, 3) }));
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(true);
        expect(snapshot.arena).toBeNull();
        expect(permutationMaterialised(snapshot)).toBe(false);
        expect(snapshot.arcToEdge).not.toBe(naive.arcToEdge);
        expect(snapshot.arcToEdge.length).toBe(naive.arcCount);
    });

    it("derives edgeToArc for a directed permutation and for an undirected graph", () => {
        const directed: GraphSpec = { directed: true, edges: WEIGHTED_EDGES };
        const directedSnapshot = fromCsr(csrOf(directed, { edgeToArc: undefined }));
        assertMatchesSpec(directedSnapshot, directed);
        expect(Array.from(directedSnapshot.edgeToArc)).toEqual(Array.from(naiveCsr(directed).edgeToArc));
        // undirected: the derivation picks the arc in the lower row, i.e. the low -> high orientation
        const lowHigh: GraphSpec = {
            directed: false,
            edges: WEIGHTED_EDGES.map((e) => (e[0] <= e[1] ? e : [e[1], e[0], e[2] as number])),
        };
        const undirectedSnapshot = fromCsr(csrOf(lowHigh, { edgeToArc: undefined }));
        assertMatchesSpec(undirectedSnapshot, lowHigh);
        expect(Array.from(undirectedSnapshot.edgeToArc)).toEqual(Array.from(naiveCsr(lowHigh).edgeToArc));
    });

    it("derives an undirected edgeToArc as the arc in the lower row when orientation is unknown", () => {
        // edge 0 declared 2 -> 0 by the naive construction; without edgeToArc the row-0 arc is chosen
        const spec: GraphSpec = {
            directed: false,
            edges: [
                [2, 0],
                [1, 2],
            ],
        };
        const snapshot = fromCsr(csrOf(spec, { edgeToArc: undefined }));
        assertInvariants(snapshot);
        expect(snapshot.edgeSource(0)).toBe(0);
        expect(snapshot.edgeTarget(0)).toBe(2);
        expect(snapshot.edgeSource(1)).toBe(1);
        expect(snapshot.edgeTarget(1)).toBe(2);
    });

    it("defaults edgeCount to arcCount (directed) or derives it from arcToEdge (undirected)", () => {
        const directed = fromCsr(csrOf({ directed: true, edges: WEIGHTED_EDGES }, { edgeCount: undefined }));
        expect(directed.edgeCount).toBe(WEIGHTED_EDGES.length);
        const undirected = fromCsr(csrOf({ directed: false, edges: WEIGHTED_EDGES }, { edgeCount: undefined }));
        expect(undirected.edgeCount).toBe(WEIGHTED_EDGES.length);
        assertInvariants(undirected);
        const empty = fromCsr({
            directed: false,
            nodeCount: 2,
            rowPtr: new Uint32Array([0, 0, 0]),
            colIdx: new Uint32Array(0),
            arcToEdge: new Uint32Array(0),
        });
        expect(empty.edgeCount).toBe(0);
        expect(empty.edgeToArc.length).toBe(0);
        assertInvariants(empty);
    });

    it("accepts an empty directed graph without any optional array and detects a rowPtr-only arena", () => {
        const rowPtr = new Uint32Array(new ArrayBuffer(256), 0, 1);
        const snapshot = fromCsr({ directed: true, nodeCount: 0, rowPtr, colIdx: new Uint32Array(0) });
        assertInvariants(snapshot);
        expect(snapshot.arena?.byteLength).toBe(4);
        expect(snapshot.arena?.hotByteLength).toBe(4);
        expect(snapshot.arena?.segments.colIdx).toBeNull();
        expect(snapshot.weights).toBeNull();
        expect(snapshot.flags.weighted).toBe(false);
    });

    it("keeps a zero-length weights array as weighted", () => {
        const snapshot = fromCsr({
            directed: true,
            nodeCount: 1,
            rowPtr: new Uint32Array([0, 0]),
            colIdx: new Uint32Array(0),
            weights: new Float32Array(0),
        });
        assertInvariants(snapshot);
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(true);
    });
});

describe("fromCsr: sortRows", () => {
    /** A CSR whose rows are permuted arbitrarily (targets out of order), from a spec. */
    function shuffledCsr(spec: GraphSpec, seed = 7): CsrInput {
        const naive = naiveCsr(spec);
        let state = seed;
        const random = (): number => {
            state = (state * 1103515245 + 12345) % 2147483648;
            return state / 2147483648;
        };
        const colIdx = naive.colIdx.slice();
        const arcToEdge = naive.arcToEdge.slice();
        const weights = naive.weights === null ? null : naive.weights.slice();
        const edgeToArc = naive.edgeToArc.slice();
        const where = new Uint32Array(naive.arcCount);
        for (let u = 0; u < naive.nodeCount; u++) {
            const start = naive.rowPtr[u];
            const end = naive.rowPtr[u + 1];
            for (let a = end - 1; a > start; a--) {
                const b = start + Math.floor(random() * (a - start + 1));
                [colIdx[a], colIdx[b]] = [colIdx[b], colIdx[a]];
                [arcToEdge[a], arcToEdge[b]] = [arcToEdge[b], arcToEdge[a]];
                if (weights !== null) {
                    [weights[a], weights[b]] = [weights[b], weights[a]];
                }
            }
            for (let a = start; a < end; a++) {
                where[a] = a;
            }
        }
        // recompute edgeToArc: the declared arc of e is the one in the row of naive's declared source
        const declaredRow = new Uint32Array(naive.edgeCount);
        for (let u = 0; u < naive.nodeCount; u++) {
            for (let a = naive.rowPtr[u]; a < naive.rowPtr[u + 1]; a++) {
                if (naive.edgeToArc[naive.arcToEdge[a]] === a) {
                    declaredRow[naive.arcToEdge[a]] = u;
                }
            }
        }
        for (let u = 0; u < naive.nodeCount; u++) {
            for (let a = naive.rowPtr[u]; a < naive.rowPtr[u + 1]; a++) {
                const e = arcToEdge[a];
                if (declaredRow[e] === u && (spec.directed || colIdx[a] !== u || naive.src[e] === naive.dst[e])) {
                    edgeToArc[e] = a;
                }
            }
        }
        return {
            directed: spec.directed,
            nodeCount: naive.nodeCount,
            rowPtr: naive.rowPtr,
            colIdx,
            weights,
            arcToEdge,
            edgeToArc,
            edgeCount: naive.edgeCount,
        };
    }

    it.each(SPECS)("rebuilds unsorted rows through the freeze pipeline: %s", (_name, spec) => {
        const input = shuffledCsr(spec);
        const naive = naiveCsr(spec);
        const before = { colIdx: Array.from(input.colIdx), arcToEdge: Array.from(input.arcToEdge as U32) };
        const shuffled =
            before.colIdx.some((v, a) => v !== naive.colIdx[a]) ||
            before.arcToEdge.some((e, a) => e !== naive.arcToEdge[a]);
        const snapshot = fromCsr(input);
        assertMatchesSpec(snapshot, spec);
        // the caller's arrays are never modified
        expect(Array.from(input.colIdx)).toEqual(before.colIdx);
        expect(Array.from(input.arcToEdge as U32)).toEqual(before.arcToEdge);
        if (shuffled) {
            // a rebuild: fresh arrays in a fresh arena
            expect(snapshot.arena).not.toBeNull();
            expect(snapshot.rowPtr).not.toBe(input.rowPtr);
            expect(snapshot.colIdx).not.toBe(input.colIdx);
        } else {
            // nothing to sort (rows of at most one arc): adopted as given
            expect(snapshot.colIdx).toBe(input.colIdx);
        }
    });

    it("rebuilds an unsorted directed identity input into a permuted core preserving edge indices", () => {
        const snapshot = fromCsr({
            directed: true,
            nodeCount: 3,
            rowPtr: new Uint32Array([0, 2, 3, 3]),
            colIdx: new Uint32Array([2, 1, 0]),
            weights: new Float32Array([5, 6, 7]),
        });
        assertInvariants(snapshot);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(snapshot.colIdx)).toEqual([1, 2, 0]);
        expect(Array.from(snapshot.arcToEdge)).toEqual([1, 0, 2]);
        expect(snapshot.edgeTarget(0)).toBe(2);
        expect(snapshot.weights?.[snapshot.edgeToArc[0]]).toBe(5);
    });

    it("rebuilds rows whose parallel arcs are out of edge order", () => {
        const snapshot = fromCsr({
            directed: true,
            nodeCount: 2,
            rowPtr: new Uint32Array([0, 2, 2]),
            colIdx: new Uint32Array([1, 1]),
            arcToEdge: new Uint32Array([1, 0]),
            edgeToArc: new Uint32Array([1, 0]),
        });
        assertInvariants(snapshot);
        expect(Array.from(snapshot.arcToEdge)).toEqual([0, 1]);
        expect(snapshot.flags.multigraph).toBe(true);
    });

    it("adopts sorted rows unchanged even under sortRows: true", () => {
        const input = csrOf({ directed: false, edges: WEIGHTED_EDGES });
        const snapshot = fromCsr(input, { sortRows: true });
        expect(snapshot.colIdx).toBe(input.colIdx);
    });

    it("sortRows: false asserts sorted rows under full validation (E_INVALID_SNAPSHOT I4)", () => {
        const input = shuffledCsr({ directed: false, edges: KARATE_EDGES });
        const err = thrown(() => fromCsr(input, { sortRows: false }));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details.invariant).toBe("I4");
        expect(typeof err.details.row).toBe("number");
        // structure and none do not check sortedness
        expect(fromCsr(input, { sortRows: false, validate: "structure" }).colIdx).toBe(input.colIdx);
        expect(fromCsr(input, { sortRows: false, validate: "none" }).colIdx).toBe(input.colIdx);
    });

    it("rebuilds under validate: none and structure too", () => {
        const spec: GraphSpec = { directed: false, edges: WEIGHTED_EDGES };
        for (const validate of ["none", "structure"] as const) {
            const snapshot = fromCsr(shuffledCsr(spec), { validate });
            assertMatchesSpec(snapshot, spec);
        }
    });

    it("verifies flag claims against the rebuilt core", () => {
        const input = shuffledCsr({ directed: false, edges: WEIGHTED_EDGES });
        const err = thrown(() => fromCsr({ ...input, flags: { multigraph: false } }));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details).toMatchObject({ invariant: "I9", flag: "multigraph", found: false, expected: true });
        expect(fromCsr({ ...input, flags: { multigraph: true, hasSelfLoops: true } }).flags.multigraph).toBe(true);
    });

    it("catches an unpaired arc in an unsorted undirected input under full validation (I7)", () => {
        // undirected triangle 0-1, 1-2, 2-0 with row 0 unsorted and the mate of 1 -> 2 replaced by 1 -> 0
        const err = thrown(() =>
            fromCsr({
                directed: false,
                nodeCount: 3,
                rowPtr: new Uint32Array([0, 2, 4, 6]),
                colIdx: new Uint32Array([2, 1, 0, 0, 1, 0]),
                arcToEdge: new Uint32Array([2, 0, 0, 1, 1, 2]),
                edgeToArc: new Uint32Array([1, 3, 5]),
                edgeCount: 3,
            }),
        );
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details.invariant).toBe("I7");
    });

    it("catches unequal mate weights in an unsorted undirected input under full validation (I7)", () => {
        const input = shuffledCsr({ directed: false, edges: WEIGHTED_EDGES });
        const weights = (input.weights as Float32Array).slice();
        // change the weight of one non-loop arc only
        const loopFree = Array.from(input.colIdx).findIndex((v, a) => {
            let row = 0;
            while (input.rowPtr[row + 1] <= a) {
                row++;
            }
            return v !== row;
        });
        weights[loopFree] += 10;
        const err = thrown(() => fromCsr({ ...input, weights }));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details.invariant).toBe("I7");
        // the same input is accepted (and built) without the pairing checks
        assertInvariants(fromCsr({ ...input, weights }, { validate: "structure" }));
    });

    it("runs the order-independent full checks on the input before a rebuild (I6 duplicate edge)", () => {
        const err = thrown(() =>
            fromCsr({
                directed: true,
                nodeCount: 2,
                rowPtr: new Uint32Array([0, 2, 2]),
                colIdx: new Uint32Array([1, 0]),
                arcToEdge: new Uint32Array([0, 0]),
                edgeToArc: new Uint32Array([0, 0]),
            }),
        );
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(["I5", "I6"]).toContain(err.details.invariant);
    });
});

describe("fromCsr: validation levels and flag claims", () => {
    const spec: GraphSpec = { directed: false, edges: WEIGHTED_EDGES };

    it("defaults to full validation and rejects an out-of-range target (I2)", () => {
        const input = csrOf(spec);
        input.colIdx[1] = 99;
        const err = thrown(() => fromCsr(input));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details).toMatchObject({ invariant: "I2", arc: 1, found: 99 });
        expect(thrown(() => fromCsr(input, { validate: "structure" })).details.invariant).toBe("I2");
        // "none" trusts the caller
        expect(fromCsr(input, { validate: "none", sortRows: false }).colIdx[1]).toBe(99);
    });

    it("rejects a NaN weight only under full (I8)", () => {
        const input = csrOf(spec);
        (input.weights as Float32Array)[0] = Number.NaN;
        const err = thrown(() => fromCsr(input));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details.invariant).toBe("I8");
        expect(() => fromCsr(input, { validate: "structure" })).not.toThrow();
    });

    it("rejects a bad rowPtr (I1) at every level except none", () => {
        const input = csrOf(spec);
        const rowPtr = input.rowPtr.slice();
        rowPtr[2] = rowPtr[1] - 1;
        for (const validate of ["structure", "full"] as const) {
            const err = thrown(() => fromCsr({ ...input, rowPtr }, { validate }));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details.invariant).toBe("I1");
        }
        const short = thrown(() => fromCsr({ ...input, rowPtr: rowPtr.subarray(0, 2) }, { validate: "none" }));
        expect(short.details.invariant).toBe("I1");
    });

    it("rejects a bad permutation range (I5) and a wrong directed edgeCount (I6)", () => {
        const input = csrOf(spec);
        const arcToEdge = (input.arcToEdge as U32).slice();
        arcToEdge[0] = 1000;
        const range = thrown(() => fromCsr({ ...input, arcToEdge }, { validate: "structure" }));
        expect(range.details.invariant).toBe("I5");
        const directed = csrOf({ directed: true, edges: WEIGHTED_EDGES });
        const count = thrown(() => fromCsr({ ...directed, edgeCount: directed.colIdx.length - 1 }));
        expect(count.details.invariant).toBe("I6");
    });

    it("rejects a wrong undirected arc count (I7) and a missing undirected arcToEdge (I5)", () => {
        const input = csrOf(spec);
        const count = thrown(() =>
            fromCsr({
                directed: false,
                nodeCount: 2,
                rowPtr: new Uint32Array([0, 1, 1]),
                colIdx: new Uint32Array([1]),
                arcToEdge: new Uint32Array([0]),
                edgeCount: 1,
            }),
        );
        expect(count.details.invariant).toBe("I7");
        const extra = thrown(() => fromCsr({ ...input, edgeCount: (input.edgeCount as number) + 1 }));
        expect(extra.details.invariant).toBe("I5");
        const missing = thrown(() => fromCsr({ ...input, arcToEdge: undefined }));
        expect(missing.code).toBe("E_INVALID_SNAPSHOT");
        expect(missing.details).toMatchObject({ invariant: "I5", array: "arcToEdge" });
        // even under none: the array is structurally required
        expect(thrown(() => fromCsr({ ...input, arcToEdge: undefined }, { validate: "none" })).details.invariant).toBe(
            "I5",
        );
    });

    it("rejects a directed identity arcToEdge paired with a non-identity edgeToArc (I5)", () => {
        const naive = naiveCsr({ directed: true, edges: gridEdges(3, 2) });
        const edgeToArc = naive.edgeToArc.slice();
        [edgeToArc[0], edgeToArc[1]] = [edgeToArc[1], edgeToArc[0]];
        const err = thrown(() =>
            fromCsr({
                directed: true,
                nodeCount: naive.nodeCount,
                rowPtr: naive.rowPtr,
                colIdx: naive.colIdx,
                edgeToArc,
            }),
        );
        expect(err.details).toMatchObject({ invariant: "I5", array: "edgeToArc" });
        // an identity edgeToArc is accepted and dropped
        const ok = fromCsr({
            directed: true,
            nodeCount: naive.nodeCount,
            rowPtr: naive.rowPtr,
            colIdx: naive.colIdx,
            edgeToArc: naive.edgeToArc,
        });
        expect(ok.flags.arcToEdgeIsIdentity).toBe(true);
        expect(permutationMaterialised(ok)).toBe(false);
    });

    it("verifies flag claims (I9) and accepts true ones", () => {
        const input = csrOf(spec);
        const truth = fromCsr(input).flags;
        expect(truth).toEqual({
            multigraph: true,
            hasSelfLoops: true,
            arcToEdgeIsIdentity: false,
            weighted: true,
            allWeightsOne: false,
            nonNegativeWeights: false,
            finiteWeights: true,
        });
        expect(fromCsr(csrOf(spec, { flags: { ...truth } })).flags).toEqual(truth);
        for (const name of Object.keys(truth) as (keyof typeof truth)[]) {
            const err = thrown(() => fromCsr(csrOf(spec, { flags: { [name]: !truth[name] } })));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details).toMatchObject({
                invariant: "I9",
                flag: name,
                found: !truth[name],
                expected: truth[name],
            });
        }
    });

    it("trusts the claims under structure and none, computing only what is unclaimed", () => {
        const input = csrOf(spec);
        const truth = fromCsr(input).flags;
        for (const validate of ["structure", "none"] as const) {
            const computed = fromCsr(csrOf(spec), { validate });
            expect(computed.flags).toEqual(truth);
            const lie = fromCsr(csrOf(spec, { flags: { ...truth, allWeightsOne: true } }), { validate });
            expect(lie.flags.allWeightsOne).toBe(true);
            expect(lie.flags.multigraph).toBe(true);
            const partial = fromCsr(csrOf(spec, { flags: { finiteWeights: false } }), { validate });
            expect(partial.flags.finiteWeights).toBe(false);
            expect(partial.flags.multigraph).toBe(true);
            expect(partial.selfLoopCount).toBe(1);
        }
        // a wrong claim on a rebuilt core is stored as given below "full" and rejected under "full"
        const unsorted = csrOf(spec);
        [unsorted.colIdx[0], unsorted.colIdx[1]] = [unsorted.colIdx[1], unsorted.colIdx[0]];
        const arcToEdge = unsorted.arcToEdge as U32;
        [arcToEdge[0], arcToEdge[1]] = [arcToEdge[1], arcToEdge[0]];
        const weights = unsorted.weights as Float32Array;
        [weights[0], weights[1]] = [weights[1], weights[0]];
        const edgeToArc = unsorted.edgeToArc as U32;
        for (let e = 0; e < edgeToArc.length; e++) {
            if (edgeToArc[e] === 0) {
                edgeToArc[e] = 1;
            } else if (edgeToArc[e] === 1) {
                edgeToArc[e] = 0;
            }
        }
        const trusted = fromCsr({ ...unsorted, flags: { finiteWeights: false } }, { validate: "structure" });
        expect(trusted.flags.finiteWeights).toBe(false);
        expect(trusted.colIdx).not.toBe(unsorted.colIdx);
        expect(thrown(() => fromCsr({ ...unsorted, flags: { finiteWeights: false } })).details.flag).toBe(
            "finiteWeights",
        );
    });

    it("trusts an identity claim below full for a directed input", () => {
        const naive = naiveCsr({ directed: true, edges: WEIGHTED_EDGES });
        for (const validate of ["structure", "none"] as const) {
            const snapshot = fromCsr(
                {
                    directed: true,
                    nodeCount: naive.nodeCount,
                    rowPtr: naive.rowPtr,
                    colIdx: naive.colIdx,
                    arcToEdge: naive.arcToEdge,
                    edgeToArc: naive.edgeToArc,
                    flags: { arcToEdgeIsIdentity: false },
                },
                { validate },
            );
            expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
            expect(snapshot.arcToEdge).toBe(naive.arcToEdge);
        }
    });

    it("rejects an unknown validation level (E_UNSUPPORTED)", () => {
        const err = thrown(() => fromCsr(csrOf(spec), { validate: "loose" as unknown as "full" }));
        expect(err.code).toBe("E_UNSUPPORTED");
        expect(err.details.field).toBe("validate");
    });
});

describe("fromCsr: ids, columns and meta", () => {
    const spec: GraphSpec = { directed: true, edges: WEIGHTED_EDGES };

    it("builds every id-map kind from the ids", () => {
        expect(fromCsr(csrOf(spec, { ids: ["a", "b", "c", "d"] })).ids.kind).toBe("string");
        expect(fromCsr(csrOf(spec, { ids: [1, 2, 3, 4] })).ids.kind).toBe("identity");
        expect(fromCsr(csrOf(spec, { ids: new Float64Array([1, 2, 3, 4]) })).ids.offset).toBe(1);
        expect(fromCsr(csrOf(spec, { ids: [0, 2, 4, 6] })).ids.kind).toBe("dense");
        expect(fromCsr(csrOf(spec, { ids: [0, "x", 2, 3] })).ids.kind).toBe("mixed");
        const snapshot = fromCsr(csrOf(spec, { ids: ["a", "b", "c", "d"] }));
        assertInvariants(snapshot);
        expect(snapshot.ids.indexOf("c")).toBe(2);
    });

    it("rejects ids of the wrong length (E_COLUMN_LENGTH), duplicates and illegal ids", () => {
        const short = thrown(() => fromCsr(csrOf(spec, { ids: ["a", "b"] })));
        expect(short.code).toBe("E_COLUMN_LENGTH");
        expect(short.details).toMatchObject({ field: "ids", expected: 4, found: 2 });
        expect(thrown(() => fromCsr(csrOf(spec, { ids: ["a", "b", "a", "c"] }))).code).toBe("E_DUPLICATE_ID");
        expect(thrown(() => fromCsr(csrOf(spec, { ids: [1, 2, 2, 3] }))).code).toBe("E_DUPLICATE_ID");
        expect(thrown(() => fromCsr(csrOf(spec, { ids: ["a", "b", "c", Number.NaN] }))).code).toBe("E_INVALID_ID");
        expect(thrown(() => fromCsr(csrOf(spec, { ids: new Float64Array([1, 2, 3, Infinity]) }))).code).toBe(
            "E_INVALID_ID",
        );
        // none skips the id checks but the length rule still holds
        expect(thrown(() => fromCsr(csrOf(spec, { ids: ["a"] }), { validate: "none" })).code).toBe("E_COLUMN_LENGTH");
    });

    it("attaches node and edge columns by reference and validates their lengths (I12)", () => {
        const size = new Float32Array([1, 2, 3, 4]);
        const tag = ["a", "b", "c", "d", "e", "f"];
        const snapshot = fromCsr(
            csrOf(spec, {
                nodeColumns: { size, pos: { data: new Float32Array(12), decl: { components: 3, role: "position" } } },
                edgeColumns: {
                    tag: { data: tag, decl: {} },
                    flag: { data: new Uint32Array([0b1010]), decl: { dtype: "bool" } },
                },
            }),
        );
        assertInvariants(snapshot);
        expect(snapshot.nodes.requireTyped("size", "f32").data).toBe(size);
        expect(snapshot.nodes.byRole("position")?.meta.components).toBe(3);
        expect(snapshot.edges.requireTyped("tag", "string").decodeAll()).toEqual(tag);
        expect(snapshot.edges.requireTyped("flag", "bool").value(3)).toBe(true);
        const short = thrown(() => fromCsr(csrOf(spec, { nodeColumns: { size: new Float32Array(3) } })));
        expect(short.code).toBe("E_COLUMN_LENGTH");
        const badRef = thrown(() =>
            fromCsr(
                csrOf(spec, {
                    edgeColumns: { p: { data: new Uint32Array([0, 1, 2, 3, 4, 50]), decl: { refersTo: "edge" } } },
                }),
            ),
        );
        expect(badRef.code).toBe("E_INVALID_SNAPSHOT");
        expect(badRef.details.invariant).toBe("I12");
    });

    it("enforces unique columns under full validation (E_INVALID_SNAPSHOT, cause E_DUPLICATE_EDGE_ID)", () => {
        const ids = {
            data: ["x", "y", "x", "z", "w", "v"],
            decl: { dtype: "string", role: "id", unique: true },
        } as const;
        const err = thrown(() => fromCsr(csrOf(spec, { edgeColumns: { id: ids } })));
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details).toMatchObject({ invariant: "I12", reason: "unique", cause: "E_DUPLICATE_EDGE_ID" });
        expect(() => fromCsr(csrOf(spec, { edgeColumns: { id: ids } }), { validate: "structure" })).not.toThrow();
    });

    it("resolves meta with the builder's rules", () => {
        const snapshot = fromCsr(
            csrOf(spec, {
                meta: {
                    name: "n",
                    keywords: ["a", "b"],
                    idType: "integer",
                    timeFormat: "date",
                    mode: "dynamic",
                    declaredMultigraph: true,
                    weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
                    extra: { x: [1, 2] },
                },
            }),
        );
        expect(snapshot.meta).toEqual({
            name: "n",
            description: null,
            creator: null,
            created: null,
            modified: null,
            keywords: ["a", "b"],
            sourceFormat: null,
            sourceVersion: null,
            idType: "integer",
            timeFormat: "date",
            timeRepresentation: null,
            mode: "dynamic",
            declaredMultigraph: true,
            weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
            extra: { x: [1, 2] },
        });
        expect(Object.isFrozen(snapshot.meta)).toBe(true);
        expect(resolveGraphMeta(undefined).name).toBeNull();
        expect(resolveGraphMeta({ name: undefined }).name).toBeNull();
        const builder = new GraphBuilder({ directed: true });
        builder.setMeta({
            name: "n",
            keywords: ["a", "b"],
            idType: "integer",
            weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
        });
        const viaBuilder = builder.freeze().meta;
        expect(
            resolveGraphMeta({
                name: "n",
                keywords: ["a", "b"],
                idType: "integer",
                weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
            }),
        ).toEqual(viaBuilder);
    });

    it("rejects bad meta fields (E_COLUMN_TYPE with details.field)", () => {
        const cases: readonly [string, unknown][] = [
            ["name", 5],
            ["keywords", "k"],
            ["keywords", [1]],
            ["idType", "float"],
            ["timeFormat", "epoch"],
            ["timeRepresentation", "x"],
            ["mode", "x"],
            ["declaredMultigraph", "yes"],
            ["weightOrigin", "gexf"],
            ["weightOrigin.type", { type: 1 }],
            ["extra", { f: () => 1 }],
        ];
        for (const [field, value] of cases) {
            const key = field.split(".")[0];
            const err = thrown(() => fromCsr(csrOf(spec, { meta: { [key]: value } })));
            expect(err.code).toBe("E_COLUMN_TYPE");
            expect(err.details.field).toBe(field);
        }
    });
});

describe("fromCsr: shape rejections", () => {
    const spec: GraphSpec = { directed: true, edges: WEIGHTED_EDGES };

    it("rejects arrays of the wrong class (E_INVALID_SNAPSHOT, details.array)", () => {
        const input = csrOf(spec);
        const cases: readonly [string, Partial<CsrInput>][] = [
            ["rowPtr", { rowPtr: new Int32Array(input.rowPtr.length) as unknown as U32 }],
            ["colIdx", { colIdx: Array.from(input.colIdx) as unknown as U32 }],
            ["weights", { weights: new Float64Array(input.colIdx.length) as unknown as Float32Array<ArrayBuffer> }],
            ["arcToEdge", { arcToEdge: new Uint16Array(input.colIdx.length) as unknown as U32 }],
            ["edgeToArc", { edgeToArc: new Uint8Array(input.colIdx.length) as unknown as U32 }],
        ];
        for (const [name, patch] of cases) {
            const err = thrown(() => fromCsr({ ...input, ...patch }, { validate: "none" }));
            expect(err.code).toBe("E_INVALID_SNAPSHOT");
            expect(err.details).toMatchObject({ array: name, reason: "dtype" });
        }
    });

    it("rejects a bad nodeCount (E_TOO_LARGE) and a bad edgeCount (I3)", () => {
        const input = csrOf(spec);
        expect(thrown(() => fromCsr({ ...input, nodeCount: -1 })).code).toBe("E_TOO_LARGE");
        expect(thrown(() => fromCsr({ ...input, nodeCount: 1.5 })).code).toBe("E_TOO_LARGE");
        expect(thrown(() => fromCsr({ ...input, nodeCount: MAX_COUNT + 1 })).code).toBe("E_TOO_LARGE");
        const count = thrown(() => fromCsr({ ...input, edgeCount: -2 }));
        expect(count.details.invariant).toBe("I3");
    });

    it("rejects a directed input whose arcToEdge is not a permutation (I6) and a mis-oriented edgeToArc (I5)", () => {
        const input = csrOf({
            directed: true,
            edges: [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
        });
        const dup = thrown(() =>
            fromCsr({ ...input, arcToEdge: new Uint32Array([0, 0, 2]), edgeToArc: new Uint32Array([0, 0, 2]) }),
        );
        expect(dup.code).toBe("E_INVALID_SNAPSHOT");
        expect(["I5", "I6"]).toContain(dup.details.invariant);
        const swapped = csrOf({
            directed: true,
            edges: [
                [1, 0],
                [0, 1],
            ],
        });
        expect(Array.from(swapped.arcToEdge as U32)).toEqual([1, 0]);
        const orientation = thrown(() => fromCsr({ ...swapped, edgeToArc: new Uint32Array([0, 1]) }));
        expect(orientation.details.invariant).toBe("I5");
        expect(orientation.details.edge).toBe(0);
    });

    it("rejects an INVALID_INDEX left by a derived edgeToArc for an edge no arc holds (I5)", () => {
        const err = thrown(() =>
            fromCsr({
                directed: false,
                nodeCount: 2,
                rowPtr: new Uint32Array([0, 1, 2]),
                colIdx: new Uint32Array([1, 0]),
                arcToEdge: new Uint32Array([0, 0]),
                edgeCount: 2,
            }),
        );
        expect(err.code).toBe("E_INVALID_SNAPSHOT");
        expect(err.details.invariant).toBe("I5");
        expect(err.details.found).toBe(INVALID_INDEX);
    });
});
