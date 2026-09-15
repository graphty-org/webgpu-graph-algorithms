/**
 * Adversarial audit of the GPU memcpy contract (design section 10, invariant I10, sections 3.3-3.5
 * and 5.7) WITHOUT a device: every array a GPU consumer binds is a 4-byte-aligned view over a plain,
 * fixed-length ArrayBuffer; the freeze arena reproduces the worked offsets of section 10.3; u8 / bool
 * columns pack as documented; f64 columns convert exactly once through gpuView(); chunk windows
 * satisfy the 256-byte rebase rule of section 10.6. The tests marked "PINS DEFECT" fail on purpose
 * and are listed in the audit report.
 */

import { describe, expect, it } from "vitest";

import { allocateArenaCore, allocateSeparateCore } from "../../src/builder/arena.js";
import { ALIGNMENT, INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import {
    type ArenaLayout,
    type Column,
    type CoreArrayName,
    fromBytes,
    fromCsr,
    fromEdgeArrays,
    fromWire,
    GraphBuilder,
    type GraphSnapshot,
    paddedU32View,
    type U32,
} from "../../src/index.js";
import { layoutSegments } from "../../src/util/typed-array.js";

// ============================================================ helpers

/** The I10 predicate a GPU consumer relies on, plus the "plain fixed-length ArrayBuffer" half. */
function expectGpuBindable(name: string, view: ArrayBufferView): void {
    expect(view.buffer, `${name}: buffer class`).toBeInstanceOf(ArrayBuffer);
    expect((view.buffer as ArrayBuffer).resizable, `${name}: resizable buffer`).toBe(false);
    expect(view.byteOffset % 4, `${name}: byteOffset ${view.byteOffset}`).toBe(0);
    expect(view.byteLength % 4, `${name}: byteLength ${view.byteLength}`).toBe(0);
}

function errorCode(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return err.code;
        }
        throw err;
    }
    return null;
}

const CORE_NAMES: readonly CoreArrayName[] = ["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"];

/** Every core array a GPU binds, keyed by segment name; identity permutations are NOT touched. */
function coreArrays(s: GraphSnapshot): Record<CoreArrayName, ArrayBufferView | null> {
    const identity = s.flags.arcToEdgeIsIdentity;
    return {
        rowPtr: s.rowPtr,
        colIdx: s.colIdx,
        weights: s.weights,
        arcToEdge: identity ? null : s.arcToEdge,
        edgeToArc: identity ? null : s.edgeToArc,
    };
}

/** Section 10.3 as a check: segment order, 256-alignment relative to byteOffset, hot prefix, no aliasing. */
function expectArenaContract(s: GraphSnapshot): ArenaLayout {
    const { arena } = s;
    expect(arena).not.toBeNull();
    const a = arena as ArenaLayout;
    expect(a.alignment).toBe(256);
    expect(a.buffer).toBeInstanceOf(ArrayBuffer);
    expect(a.buffer.resizable).toBe(false);
    expect(a.byteOffset % 4).toBe(0);
    expect(a.byteLength % 4).toBe(0);
    expect(a.hotByteLength % 4).toBe(0);
    expect(a.hotByteLength).toBeLessThanOrEqual(a.byteLength);
    expect(a.byteOffset + a.byteLength).toBeLessThanOrEqual(a.buffer.byteLength);
    const arrays = coreArrays(s);
    let cursor = 0;
    let hotEnd = 0;
    for (const name of CORE_NAMES) {
        const segment = a.segments[name];
        const array = arrays[name];
        if (segment === null) {
            // null = absent, zero-length, or identity: never in the arena
            if (array !== null && array.byteLength > 0) {
                expect(array.buffer, `${name} outside the arena must not alias it`).not.toBe(a.buffer);
            }
            continue;
        }
        expect(array, `${name}: segment without array`).not.toBeNull();
        const view = array as ArrayBufferView;
        const relative = segment.byteOffset - a.byteOffset;
        expect(relative % ALIGNMENT, `${name}: relative offset ${relative}`).toBe(0);
        expect(relative, `${name}: hot-to-cold order`).toBeGreaterThanOrEqual(cursor);
        expect(segment.byteLength).toBeGreaterThan(0);
        expect(segment.byteLength % 4).toBe(0);
        expect(view.buffer).toBe(a.buffer);
        expect(view.byteOffset).toBe(segment.byteOffset);
        expect(view.byteLength).toBe(segment.byteLength);
        cursor = relative + segment.byteLength;
        if (name === "rowPtr" || name === "colIdx" || name === "weights") {
            hotEnd = cursor;
        }
    }
    expect(a.byteLength).toBe(cursor);
    expect(a.hotByteLength).toBe(hotEnd);
    // the literal upload expressions of section 10.3 / 16.6 construct without throwing
    const hot = new Uint8Array(a.buffer, a.byteOffset, a.hotByteLength);
    const all = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    expect(hot.byteLength % 4).toBe(0);
    expect(all.byteLength % 4).toBe(0);
    return a;
}

/** Every array section 10.1 lists, on a snapshot. */
function expectAllBindablesAligned(s: GraphSnapshot): void {
    const arrays = coreArrays(s);
    for (const name of CORE_NAMES) {
        const array = arrays[name];
        if (array !== null) {
            expectGpuBindable(name, array);
        }
    }
    const rev = s.reverse();
    expectGpuBindable("reverse.rowPtr", rev.rowPtr);
    expectGpuBindable("reverse.colIdx", rev.colIdx);
    expectGpuBindable("reverse.fwdArc", rev.fwdArc);
    if (rev.weights !== null) {
        expectGpuBindable("reverse.weights", rev.weights);
    }
    expectGpuBindable("coo.src", s.coo().src);
    const el = s.edgeList();
    expectGpuBindable("edgeList.src", el.src);
    expectGpuBindable("edgeList.dst", el.dst);
    if (el.weights !== null) {
        expectGpuBindable("edgeList.weights", el.weights);
    }
    expectGpuBindable("outDegree", s.outDegree());
    expectGpuBindable("inDegree", s.inDegree());
    expectGpuBindable("degree", s.degree());
    expectGpuBindable("selfLoopsPerNode", s.selfLoopsPerNode());
    expectGpuBindable("selfLoopArcs", s.selfLoopArcs());
    expectGpuBindable("degreeOrder.perm", s.degreeOrder().perm);
    expectGpuBindable("degreeOrder(reverse).perm", s.degreeOrder({ of: "reverse" }).perm);
    if (!s.directed) {
        expectGpuBindable("mate", s.mate());
    }
    for (const table of [s.nodes, s.edges, s.graph]) {
        for (const column of table) {
            if (column.gpu !== "none") {
                expectGpuBindable(`gpuView(${column.meta.name})`, table.gpuView(column.meta.name));
            }
            if (column.validity !== null) {
                expectGpuBindable(`validity(${column.meta.name})`, column.validity);
            }
        }
    }
}

/** A deterministic random edge list. */
function randomEdges(n: number, m: number, seed: number): { src: U32; dst: U32 } {
    const src = new Uint32Array(m);
    const dst = new Uint32Array(m);
    let state = seed >>> 0;
    const next = (): number => {
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        return state;
    };
    for (let e = 0; e < m; e++) {
        src[e] = next() % n;
        dst[e] = next() % n;
    }
    return { src, dst };
}

function smallGraph(directed: boolean): GraphSnapshot {
    const b = new GraphBuilder({ directed });
    b.addEdge("c", "a", 3);
    b.addEdge("a", "b", 1);
    b.addEdge("b", "c", 2);
    b.addEdge("a", "a", 5);
    b.addEdge("a", "b", 7);
    b.addEdge("d", "a", 9);
    b.declareNodeColumn({ name: "bytes", dtype: "u8", components: 3, nullable: false });
    b.declareNodeColumn({ name: "flag", dtype: "bool", nullable: false });
    b.declareNodeColumn({ name: "score", dtype: "f64", mutable: true, nullable: false });
    b.declareNodeColumn({ name: "kind", dtype: "dict", nullable: false });
    b.declareNodeColumn({ name: "maybe", dtype: "u32", nullable: true, default: 4 });
    b.declareEdgeColumn({ name: "cost", dtype: "f32", nullable: false });
    for (let i = 0; i < 4; i++) {
        b.setNodeValue("bytes", i, [i, i + 1, i + 2]);
        b.setNodeValue("flag", i, i % 2 === 0);
        b.setNodeValue("score", i, i * 1.1);
        b.setNodeValue("kind", i, i === 0 ? "x" : "y");
    }
    b.setNodeValue("maybe", 1, 42);
    for (let e = 0; e < 6; e++) {
        b.setEdgeValue("cost", e, e * 0.5);
    }
    return b.freeze();
}

// ============================================================ 10.3 worked layout

describe("audit: arena layout (design section 10.3)", () => {
    it("reproduces the worked offsets for the benchmark shape (n = 100k, A = 2M, E = 1M, weighted)", () => {
        const core = allocateArenaCore({
            nodeCount: 100000,
            arcCount: 2000000,
            edgeCount: 1000000,
            weighted: true,
            identity: false,
        });
        const arena = core.arena as ArenaLayout;
        expect(arena.segments).toEqual({
            rowPtr: { byteOffset: 0, byteLength: 400004 },
            colIdx: { byteOffset: 400128, byteLength: 8000000 },
            weights: { byteOffset: 8400128, byteLength: 8000000 },
            arcToEdge: { byteOffset: 16400128, byteLength: 8000000 },
            edgeToArc: { byteOffset: 24400128, byteLength: 4000000 },
        });
        expect(arena.byteLength).toBe(28400128);
        expect(arena.hotByteLength).toBe(16400128);
        expect(arena.buffer.byteLength).toBe(28400128);
        expect(layoutSegments([400004, 8000000, 8000000, 8000000, 4000000]).padding).toBe(124);
    });

    it("reproduces the directed worked totals: 16,400,128 with permutations, 8,400,128 identity", () => {
        const perm = allocateArenaCore({
            nodeCount: 100000,
            arcCount: 1000000,
            edgeCount: 1000000,
            weighted: true,
            identity: false,
        });
        const permArena = perm.arena as ArenaLayout;
        expect(permArena.byteLength).toBe(16400128);
        expect(permArena.hotByteLength).toBe(8400128);
        expect(permArena.segments.arcToEdge?.byteOffset).toBe(8400128);
        expect(permArena.segments.edgeToArc?.byteOffset).toBe(12400128);
        const identity = allocateArenaCore({
            nodeCount: 100000,
            arcCount: 1000000,
            edgeCount: 1000000,
            weighted: true,
            identity: true,
        });
        const idArena = identity.arena as ArenaLayout;
        expect(idArena.byteLength).toBe(8400128);
        expect(idArena.hotByteLength).toBe(8400128);
        expect(idArena.segments.arcToEdge).toBeNull();
        expect(idArena.segments.edgeToArc).toBeNull();
        expect(identity.arcToEdge).toBeNull();
        expect(identity.edgeToArc).toBeNull();
    });

    it("gives zero-length arrays a null segment and keeps them out of the arena", () => {
        const empty = allocateArenaCore({ nodeCount: 3, arcCount: 0, edgeCount: 0, weighted: true, identity: false });
        const arena = empty.arena as ArenaLayout;
        expect(arena.segments.colIdx).toBeNull();
        expect(arena.segments.weights).toBeNull();
        expect(arena.segments.arcToEdge).toBeNull();
        expect(arena.segments.edgeToArc).toBeNull();
        expect(arena.byteLength).toBe(16);
        expect(arena.hotByteLength).toBe(16);
        expect(empty.colIdx.length).toBe(0);
        expect(empty.weights?.length).toBe(0);
        expect(empty.colIdx.buffer).not.toBe(arena.buffer);
        const unweighted = allocateArenaCore({
            nodeCount: 1,
            arcCount: 5,
            edgeCount: 5,
            weighted: false,
            identity: true,
        });
        expect((unweighted.arena as ArenaLayout).hotByteLength).toBe(256 + 20);
        expect(unweighted.weights).toBeNull();
    });

    it("separate cores satisfy I10 with arena === null", () => {
        const core = allocateSeparateCore({ nodeCount: 5, arcCount: 7, edgeCount: 7, weighted: true, identity: false });
        expect(core.arena).toBeNull();
        for (const array of [core.rowPtr, core.colIdx, core.weights, core.arcToEdge, core.edgeToArc]) {
            expectGpuBindable("separate", array as ArrayBufferView);
        }
    });

    it("the public freeze path lays the 100k / 1M benchmark graph out exactly as section 10.3 says", () => {
        const n = 100000;
        const m = 1000000;
        const { src, dst } = randomEdges(n, m, 7);
        for (let e = 0; e < m; e++) {
            if (src[e] === dst[e]) {
                dst[e] = (dst[e] + 1) % n;
            }
        }
        const weights = new Float32Array(m).fill(2);
        const undirected = fromEdgeArrays({ directed: false, nodeCount: n, src, dst, weights });
        expect(undirected.arcCount).toBe(2000000);
        const ua = expectArenaContract(undirected);
        expect(ua.byteLength).toBe(28400128);
        expect(ua.hotByteLength).toBe(16400128);
        expect(ua.segments.colIdx?.byteOffset).toBe(400128);
        expect(ua.segments.edgeToArc?.byteOffset).toBe(24400128);
        const directed = fromEdgeArrays({ directed: true, nodeCount: n, src, dst, weights });
        expect(directed.flags.arcToEdgeIsIdentity).toBe(false);
        const da = expectArenaContract(directed);
        expect(da.byteLength).toBe(16400128);
        expect(da.hotByteLength).toBe(8400128);
    });
});

// ============================================================ I10 over every producer

describe("audit: I10 on every producer and every 10.1 binding", () => {
    it("builder freeze, directed and undirected, arena and separate", () => {
        for (const directed of [true, false]) {
            const s = smallGraph(directed);
            expectArenaContract(s);
            expectAllBindablesAligned(s);
            const b = new GraphBuilder({ directed });
            b.addGraph(s);
            const separate = b.freeze({ arena: false });
            expect(separate.arena).toBeNull();
            expectAllBindablesAligned(separate);
        }
    });

    it("an identity-permutation directed core has no arcToEdge in the arena and never materialises it on upload paths", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge(0, 1);
        b.addEdge(0, 2);
        b.addEdge(1, 2);
        const s = b.freeze();
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        const arena = expectArenaContract(s);
        expect(arena.segments.arcToEdge).toBeNull();
        expect(arena.segments.edgeToArc).toBeNull();
        expect(s.byteLength()).toBe(s.rowPtr.byteLength + s.colIdx.byteLength);
        // views that a GPU binds must not force the identity permutation into existence
        s.coo();
        s.edgeList();
        s.reverse();
        s.outDegree();
        s.degreeOrder();
        expect(s.byteLength()).toBe(s.rowPtr.byteLength + s.colIdx.byteLength);
        expect(s.cachedViews()).toContain("edgeList");
    });

    it("undirected: reverse() is the forward arrays themselves (one upload serves pull kernels)", () => {
        const s = smallGraph(false);
        const rev = s.reverse();
        expect(rev.rowPtr).toBe(s.rowPtr);
        expect(rev.colIdx).toBe(s.colIdx);
        expect(rev.weights).toBe(s.weights);
        expect(rev.fwdArc.length).toBe(s.arcCount);
        expect(s.inDegree()).toBe(s.outDegree());
        expect(s.mate().length).toBe(s.arcCount);
        // I7: both arcs of an edge carry the same weight and arcToEdge
        const mate = s.mate();
        for (let a = 0; a < s.arcCount; a++) {
            const m = mate[a];
            expect(m).not.toBe(INVALID_INDEX);
            expect(s.arcToEdge[m]).toBe(s.arcToEdge[a]);
            expect((s.weights as Float32Array)[m]).toBe((s.weights as Float32Array)[a]);
        }
    });

    it("fromCsr adopts separate arrays with arena === null and detects an arena-shaped buffer", () => {
        const rowPtr = new Uint32Array([0, 2, 3, 3]);
        const colIdx = new Uint32Array([1, 2, 2]);
        const weights = new Float32Array([1, 2, 3]);
        const separate = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx, weights });
        expect(separate.arena).toBeNull();
        expect(separate.rowPtr).toBe(rowPtr);
        expectAllBindablesAligned(separate);
        const buffer = new ArrayBuffer(128 + 3 * 256);
        const rp = new Uint32Array(buffer, 128, 4);
        rp.set(rowPtr);
        const ci = new Uint32Array(buffer, 128 + 256, 3);
        ci.set(colIdx);
        const ww = new Float32Array(buffer, 128 + 512, 3);
        ww.set(weights);
        const shaped = fromCsr({ directed: true, nodeCount: 3, rowPtr: rp, colIdx: ci, weights: ww });
        const arena = expectArenaContract(shaped);
        expect(arena.byteOffset).toBe(128);
        expect(arena.hotByteLength).toBe(512 + 12);
        expect(shaped.rowPtr).toBe(rp);
        const copied = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx, weights }, { copy: true });
        expectArenaContract(copied);
        expect(copied.rowPtr).not.toBe(rowPtr);
    });

    it("fromCsr never adopts a SharedArrayBuffer-backed core (D-SAB)", () => {
        const sab = new SharedArrayBuffer(16);
        const rowPtr = new Uint32Array(sab) as unknown as U32;
        rowPtr.set([0, 2, 3, 3]);
        const s = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx: new Uint32Array([1, 2, 2]) });
        expect(s.rowPtr.buffer).toBeInstanceOf(ArrayBuffer);
        expectAllBindablesAligned(s);
    });

    it("fromBytes at byteOffset 0 and 8 adopts the container region as the arena; offset 4 copies", () => {
        const s = smallGraph(true);
        const bytes = s.toBytes();
        expect(bytes.byteLength % 256).toBe(0);
        const r0 = fromBytes(bytes);
        const a0 = expectArenaContract(r0);
        expect(a0.buffer).toBe(bytes.buffer);
        expectAllBindablesAligned(r0);
        for (const shift of [4, 8]) {
            const shifted = new Uint8Array(bytes.byteLength + shift);
            shifted.set(bytes, shift);
            const r = fromBytes(new Uint8Array(shifted.buffer, shift, bytes.byteLength));
            const a = expectArenaContract(r);
            if (shift === 8) {
                expect(a.buffer).toBe(shifted.buffer);
                expect(a.byteOffset % 8).toBe(0);
            } else {
                expect(a.buffer).not.toBe(shifted.buffer);
                expect(a.byteOffset).toBe(0);
            }
            expectAllBindablesAligned(r);
        }
    });

    it("fromWire keeps the arena descriptor and shares the buffer without copying", () => {
        const s = smallGraph(false);
        const r = fromWire(s.toWire());
        const a = expectArenaContract(r);
        expect(a.buffer).toBe((s.arena as ArenaLayout).buffer);
        expect(r.colIdx.buffer).toBe(s.colIdx.buffer);
        expectAllBindablesAligned(r);
        const cloned = structuredClone(s.toWire({ transfer: false }));
        const rc = fromWire(cloned);
        expectArenaContract(rc);
        expectAllBindablesAligned(rc);
    });

    it("every derived graph satisfies I10 (arena or not)", () => {
        const s = smallGraph(true);
        const derived = [
            s.toUndirected(),
            s.transpose(),
            s.simplified(),
            s.withoutSelfLoops(),
            s.inducedSubgraph(new Uint32Array([0, 1, 2])),
            s.relabel(new Uint32Array([3, 2, 1, 0])),
            s.contract(new Uint32Array([0, 0, 1, 1])),
        ];
        for (const d of derived) {
            expectAllBindablesAligned(d.snapshot);
            if (d.snapshot.arena !== null) {
                expectArenaContract(d.snapshot);
            }
        }
        expect(s.transpose().snapshot.arena).toBeNull();
        expect(s.toUndirected().snapshot.arena).not.toBeNull();
    });
});

// ============================================================ 10.2 / 10.4 columns

describe("audit: column packing and gpuView (design sections 10.2, 10.4, 5.7)", () => {
    it("bool columns are ceil(rows / 32) words for every boundary", () => {
        for (const rows of [0, 1, 31, 32, 33, 63, 64, 65, 100]) {
            const b = new GraphBuilder({ directed: true });
            b.addAnonymousNodes(rows);
            b.declareNodeColumn({ name: "bit", dtype: "bool", nullable: false });
            b.declareNodeColumn({ name: "maybe", dtype: "u32", nullable: true });
            for (let i = 0; i < rows; i++) {
                b.setNodeValue("bit", i, i % 3 === 0);
            }
            const s = b.freeze();
            const column = s.nodes.requireTyped("bit", "bool");
            expect(column.data.length, `rows ${rows}`).toBe(Math.ceil(rows / 32));
            expect(s.nodes.gpuView("bit")).toBe(column.data);
            const maybe = s.nodes.requireTyped("maybe", "u32");
            if (rows > 0) {
                expect(maybe.validity?.length, `validity rows ${rows}`).toBe(Math.ceil(rows / 32));
            }
            for (let i = 0; i < rows; i++) {
                const word = column.data[i >>> 5];
                expect((word >>> (i & 31)) & 1, `bit ${i} of ${rows}`).toBe(i % 3 === 0 ? 1 : 0);
            }
        }
    });

    it("u8 columns expose ceil(rows * components / 4) words over the column's own bytes", () => {
        for (const [rows, components] of [
            [1, 1],
            [3, 1],
            [4, 1],
            [5, 1],
            [3, 3],
            [5, 3],
            [2, 16],
        ]) {
            const b = new GraphBuilder({ directed: true });
            b.addAnonymousNodes(rows);
            b.declareNodeColumn({ name: "bytes", dtype: "u8", components, nullable: false });
            for (let i = 0; i < rows; i++) {
                const value = new Array<number>(components);
                for (let k = 0; k < components; k++) {
                    value[k] = (i * components + k) & 0xff;
                }
                b.setNodeValue("bytes", i, components === 1 ? value[0] : value);
            }
            const s = b.freeze();
            const column = s.nodes.requireTyped("bytes", "u8");
            const view = column.paddedU32View();
            expect(view.length, `${rows}x${components}`).toBe(Math.ceil((rows * components) / 4));
            expect(view.buffer).toBe(column.data.buffer);
            expect(view.byteOffset).toBe(column.data.byteOffset);
            expect(s.nodes.gpuView("bytes")).toEqual(view);
            expect(column.paddedByteLength).toBe(Math.ceil((rows * components) / 4) * 4);
            // unpack4xU8 semantics: lane i & 3 of word i >> 2, little-endian
            for (let i = 0; i < rows * components; i++) {
                expect((view[i >>> 2] >>> ((i & 3) * 8)) & 0xff, `lane ${i}`).toBe(column.data[i]);
            }
        }
    });

    it("paddedU32View covers a subarray at the tail of a padded store and refuses a store it cannot", () => {
        const backing = new Uint8Array(new ArrayBuffer(8));
        expect(paddedU32View(new Uint8Array(backing.buffer, 4, 3)).length).toBe(1);
        expect(errorCode(() => paddedU32View(new Uint8Array(backing.buffer, 5, 3)))).toBe("E_COLUMN_ALIGNMENT");
        expect(errorCode(() => paddedU32View(new Uint8Array(new ArrayBuffer(7), 4, 3)))).toBe("E_COLUMN_ALIGNMENT");
        expect(errorCode(() => paddedU32View(new Uint8Array(new ArrayBuffer(7), 0, 7)))).toBe("E_COLUMN_ALIGNMENT");
        expect(paddedU32View(new Uint8Array(new ArrayBuffer(8), 0, 7)).length).toBe(2);
    });

    it("set() of an unadoptable u8 array copies by default and refuses under adopt: strict", () => {
        const s = smallGraph(true);
        const odd = new Uint8Array(new ArrayBuffer(16), 1, 4);
        odd.set([9, 8, 7, 6]);
        const column = s.nodes.set("odd", odd);
        expect(column.dtype).toBe("u8");
        expectGpuBindable("gpuView(odd)", s.nodes.gpuView("odd"));
        expect(Array.from((column as { data: Uint8Array }).data)).toEqual([9, 8, 7, 6]);
        expect(errorCode(() => s.nodes.set("odd2", odd, undefined, { adopt: "strict" }))).toBe("E_COLUMN_ALIGNMENT");
    });

    it("f64 columns convert exactly once, the copy is invalidated by markDirty(), dropCaches() and replacement", () => {
        const s = smallGraph(true);
        const first = s.nodes.gpuView("score");
        expect(first).toBeInstanceOf(Float32Array);
        expect(first.length).toBe(4);
        expect(s.nodes.gpuView("score")).toBe(first);
        expect(Array.from(first)).toEqual(Array.from(new Float32Array([0, 1.1, 2.2, 3.3])));
        const column = s.nodes.requireTyped("score", "f64");
        column.mutableData()[0] = 99;
        expect(s.nodes.gpuView("score")).toBe(first);
        expect(first[0]).toBe(0);
        column.markDirty();
        const second = s.nodes.gpuView("score");
        expect(second).not.toBe(first);
        expect(second[0]).toBe(99);
        expect(s.nodes.gpuView("score")).toBe(second);
        s.dropCaches();
        const third = s.nodes.gpuView("score");
        expect(third).not.toBe(second);
        expect(Array.from(third)).toEqual(Array.from(second));
        const attached = new Float64Array([1.5, 2.5, 3.5, 4.5]);
        s.nodes.set("score", attached);
        const fourth = s.nodes.gpuView("score");
        expect(fourth).not.toBe(third);
        expect(Array.from(fourth)).toEqual([1.5, 2.5, 3.5, 4.5]);
        // the column keeps its f64 data untouched
        expect(s.nodes.requireTyped("score", "f64").data).toBe(attached);
    });

    it("gpuView returns the data itself for every direct dtype and throws for the ineligible ones", () => {
        const s = smallGraph(true);
        expect(s.nodes.gpuView("kind")).toBe(s.nodes.requireTyped("kind", "dict").codes);
        expect(s.nodes.gpuView("maybe")).toBe(s.nodes.requireTyped("maybe", "u32").data);
        expect(s.edges.gpuView("cost")).toBe(s.edges.requireTyped("cost", "f32").data);
        expect(s.nodes.gpuView("flag")).toBe(s.nodes.requireTyped("flag", "bool").data);
        s.nodes.set("names", ["a", "b", "c", "d"]);
        s.nodes.set("lists", [[1], [2], [3], [4]]);
        s.nodes.set("json", [{ a: 1 }, null, 3, "x"]);
        for (const name of ["names", "lists", "json"]) {
            expect(
                errorCode(() => s.nodes.gpuView(name)),
                name,
            ).toBe("E_GPU_INELIGIBLE");
            expect(s.nodes.require(name).gpu).toBe("none");
        }
        expect(errorCode(() => s.nodes.gpuView("nope"))).toBe("E_UNKNOWN_COLUMN");
    });

    it("declared defaults are already in the data of unset rows (a kernel binds gpuView() as is)", () => {
        const s = smallGraph(true);
        const maybe = s.nodes.requireTyped("maybe", "u32");
        expect(maybe.nullCount).toBe(3);
        expect(Array.from(maybe.data)).toEqual([4, 42, 4, 4]);
        expect(s.nodes.gpuView("maybe")).toBe(maybe.data);
    });

    it("slices keep I10 and eligibility for packed stores at unaligned starts", () => {
        const s = smallGraph(true);
        const bytes = s.nodes.requireTyped("bytes", "u8");
        const flag = s.nodes.requireTyped("flag", "bool");
        const maybe = s.nodes.requireTyped("maybe", "u32");
        for (const [start, end] of [
            [0, 4],
            [1, 3],
            [2, 4],
            [3, 4],
        ]) {
            const bs = bytes.slice(start, end);
            expectGpuBindable(`bytes.slice(${start},${end})`, bs.paddedU32View());
            expect(bs.paddedU32View().length).toBe(Math.ceil((3 * (end - start)) / 4));
            expect(Array.from(bs.data)).toEqual(Array.from(bytes.data.subarray(3 * start, 3 * end)));
            const fs = flag.slice(start, end);
            expectGpuBindable(`flag.slice(${start},${end})`, fs.data);
            expect(fs.data.length).toBe(Math.ceil((end - start) / 32));
            for (let i = 0; i < end - start; i++) {
                expect(fs.value(i)).toBe(flag.value(start + i));
            }
            const ms = maybe.slice(start, end);
            expectGpuBindable(`maybe.slice(${start},${end})`, ms.data);
            if (ms.validity !== null) {
                expectGpuBindable(`maybe.slice(${start},${end}).validity`, ms.validity);
            }
        }
    });

    it("results attached by reference (any length) are the bound arrays; f64 CPU results convert", () => {
        const s = smallGraph(true);
        const labels = new Uint32Array([0, 0, 1, 1]);
        const column = s.nodes.set("graphty.cc.component", labels);
        expect((column as { data: Uint32Array }).data).toBe(labels);
        expect(s.nodes.gpuView("graphty.cc.component")).toBe(labels);
        const scores = new Float64Array([0.25, 0.5, 0.75, 1]);
        s.nodes.set("pr", scores);
        const view = s.nodes.gpuView("pr");
        expect(view).toBeInstanceOf(Float32Array);
        expect(Array.from(view)).toEqual([0.25, 0.5, 0.75, 1]);
    });

    it("the weighted out-degree normaliser is f64 and NOT a GPU upload (section 10.1)", () => {
        const s = smallGraph(true);
        expect(s.weightedOutDegree()).toBeInstanceOf(Float64Array);
        expect(s.weightedInDegree()).toBeInstanceOf(Float64Array);
        expect(s.outDegree()).toBeInstanceOf(Uint32Array);
        expect(s.degreeOrder().segmentOffsets.length).toBe(5);
    });
});

// ============================================================ 10.6 chunk windows

describe("audit: chunk windows (design section 10.6)", () => {
    it("colIdx windows rebased to a multiple of 64 arcs start at a 256-byte boundary and stay bindable", () => {
        const n = 3000;
        const m = 40000;
        const { src, dst } = randomEdges(n, m, 11);
        const s = fromEdgeArrays({ directed: true, nodeCount: n, src, dst });
        const { colIdx, rowPtr } = s;
        const colBase = colIdx.byteOffset;
        for (const [v0, v1] of [
            [0, 1000],
            [1000, 2000],
            [1234, 2345],
            [2999, 3000],
            [0, n],
        ]) {
            const start = rowPtr[v0] - (rowPtr[v0] % 64);
            const window = colIdx.subarray(start, rowPtr[v1]);
            expectGpuBindable(`colIdx window [${v0},${v1})`, window);
            expect((window.byteOffset - colBase) % 256).toBe(0);
            expect(window.buffer).toBe(colIdx.buffer);
            expect(window.length).toBe(rowPtr[v1] - start);
            const rows = rowPtr.subarray(v0, v1 + 1);
            expectGpuBindable(`rowPtr window [${v0},${v1}]`, rows);
            expect(rows.length).toBe(v1 - v0 + 1);
            // the rebase constant makes rowPtr entries valid indices into the window
            for (let v = v0; v < v1; v++) {
                for (let a = rowPtr[v]; a < rowPtr[v + 1]; a++) {
                    expect(window[a - start]).toBe(colIdx[a]);
                }
            }
        }
        // the 1D dispatch ceiling is 65,535 x 256, not 2^24
        expect(65535 * 256).toBe(16776960);
        expect(Math.ceil(16776961 / 256)).toBeGreaterThan(65535);
    });

    it("arc windows above 2^31 are computed with %, never with bitwise operators", () => {
        const big = 0x80000040;
        expect(big - (big % 64)).toBe(0x80000040);
        expect(big & ~63).toBeLessThan(0);
        const odd = 0x8000007f;
        expect(odd - (odd % 64)).toBe(0x80000040);
    });
});

// ============================================================ deliberately failing: I10 "plain ArrayBuffer" is not enforced

describe("audit: I10 plain-ArrayBuffer enforcement (PINS DEFECT)", () => {
    it("set() must not adopt a SharedArrayBuffer-backed array: gpuView / transferables / validate all lie", () => {
        const s = smallGraph(true);
        const shared = new Uint32Array(new SharedArrayBuffer(16)) as unknown as U32;
        shared.set([1, 2, 3, 4]);
        let adopted: Column | null = null;
        const code = errorCode(() => {
            adopted = s.nodes.set("shared", shared);
        });
        if (code === null) {
            // adoption happened: the snapshot now violates I10 and its transfer list is one postMessage rejects
            expect(adopted).not.toBeNull();
            const view = s.nodes.gpuView("shared");
            expect(view.buffer, "gpuView() over a SharedArrayBuffer violates I10").toBeInstanceOf(ArrayBuffer);
            expect(() => s.validate({ level: "full" })).toThrow();
            const wire = s.toWire({ transfer: true });
            const transferables = s.transferables();
            for (const buffer of transferables) {
                expect(buffer).toBeInstanceOf(ArrayBuffer);
            }
            expect(() => structuredClone(wire, { transfer: transferables })).not.toThrow();
        } else {
            expect(code).toBe("E_UNSUPPORTED");
        }
    });

    it("fromCsr must not adopt a length-tracking view over a resizable ArrayBuffer (I10 / I17)", () => {
        const resizable = new ArrayBuffer(16, { maxByteLength: 4096 });
        const rowPtr = new Uint32Array(resizable);
        rowPtr.set([0, 2, 3, 3]);
        const colIdx = new Uint32Array([1, 2, 2]);
        const s = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx });
        expect(s.rowPtr.buffer.resizable, "adopted a resizable buffer as the core").toBe(false);
        resizable.resize(4096);
        expect(s.rowPtr.length).toBe(4);
        expect(() => s.validate({ level: "structure" })).not.toThrow();
    });
});
