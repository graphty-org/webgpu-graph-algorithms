/**
 * The upload contract of spec 11.3 ported from graph-format's test/audit/gpu-upload.test.ts lines 254-666 onto
 * GraphResidency (contract 3.8, 5.5): arena bindings equal the CPU views byte for byte; per-array on fromCsr and
 * transpose(); cold segments on demand; arena.byteOffset !== 0 from fromBytes (rich-v1.gsnp); packed u8 / bool
 * columns through column(); identity permutations never materialised (a getter spy plus byteLength({ views: true }));
 * the same array object uploads once; a column version bump re-uploads in place; ctx.release destroys every buffer
 * and trims the pool; E_RELEASED / isReleased; release idempotent and safe on an unknown snapshot; withColumns
 * siblings share one record; the once-only warning; views, ad hoc arrays, the windowed E_TOO_LARGE at P1, a detached
 * snapshot, clearOnLoss / destroyAll. Where gpu-upload.test.ts summed rows in a kernel, this file reads the bound
 * ranges back and compares bytes (the kernel path is P1-T5's degree test); "no uncaptured error" is the setup's
 * afterEach hook. Every test runs in a fresh context so the buffer counts are exact.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
    type ArenaLayout,
    type F32,
    fromBytes,
    fromEdgeArrays,
    GraphBuilder,
    GraphSnapshot,
    type U32,
} from "@graphty/graph-format";
import { type TestContext } from "vitest";

import { type GpuContext } from "../../src/context.js";
import { BufferUsage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError } from "../../src/errors.js";
import { GraphResidency } from "../../src/memory/residency.js";
import { type Binding } from "../../src/types/memory.js";
import { fakeCaps } from "../helpers/caps-tables.js";
import { readF32, readU32, withContext } from "../helpers/device.js";
import { csrSnapshotOf, KARATE_EDGES, pathEdges, randomEdges, snapshotOf } from "../helpers/graphs.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { requireGpu } from "../setup/gpu.js";

const MIB = 1024 * 1024;
const STORAGE = BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST;
const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "rich-v1.gsnp");
const N = 4096;
const M = 50_000;

function caught(fn: () => unknown): { code: string; details: Readonly<Record<string, unknown>> } {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return { code: err.code, details: err.details };
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

/** The code of a GraphFormatError a call lets propagate (the pass-through codes of contract 3.1). */
function formatCode(fn: () => unknown): string {
    try {
        fn();
    } catch (err) {
        const e = err as { name?: unknown; code?: unknown };
        if (e.name === "GraphFormatError" && typeof e.code === "string") {
            return e.code;
        }
        throw err;
    }
    throw new Error("expected a GraphFormatError");
}

/** gpu-upload.test.ts directedGraph(): n = 4096, m = 50,000, weights 1..7, a `cost` edge column, a non-identity arcToEdge. */
function directedGraph(): GraphSnapshot {
    const edges = randomEdges(N, M, 42);
    const src = new Uint32Array(M);
    const dst = new Uint32Array(M);
    const weights = new Float32Array(M);
    const cost = new Float32Array(M);
    edges.forEach((edge, e) => {
        src[e] = edge[0];
        dst[e] = edge[1];
        weights[e] = 1 + (e % 7);
        cost[e] = (e % 13) * 0.5;
    });
    const s = fromEdgeArrays(
        { directed: true, nodeCount: N, src, dst, weights, edgeColumns: { cost } },
        { label: "directed" },
    );
    expect(s.flags.arcToEdgeIsIdentity).toBe(false);
    expect(s.arena).not.toBeNull();
    return s;
}

/** gpu-upload.test.ts undirectedGraph(): weights 1..5, doubled arcs. */
function undirectedGraph(): GraphSnapshot {
    const edges = randomEdges(N, M, 43);
    const src = new Uint32Array(M);
    const dst = new Uint32Array(M);
    const weights = new Float32Array(M);
    edges.forEach((edge, e) => {
        src[e] = edge[0];
        dst[e] = edge[1];
        weights[e] = 1 + (e % 5);
    });
    const s = fromEdgeArrays({ directed: false, nodeCount: N, src, dst, weights }, { label: "undirected" });
    expect(s.arcCount).toBe(2 * M);
    expect(s.arena).not.toBeNull();
    return s;
}

async function expectBindingU32(ctx: GpuContext, binding: Binding, expected: U32, label: string): Promise<void> {
    expect(binding.size, `${label}: size`).toBe(expected.byteLength);
    expect(binding.window, `${label}: window`).toBeNull();
    expectBitwiseEqual(await readU32(ctx, binding.buffer, expected.length, binding.offset), expected, label);
}

async function expectBindingF32(ctx: GpuContext, binding: Binding, expected: F32, label: string): Promise<void> {
    expect(binding.size, `${label}: size`).toBe(expected.byteLength);
    expect(binding.window, `${label}: window`).toBeNull();
    expectBitwiseEqual(await readF32(ctx, binding.buffer, expected.length, binding.offset), expected, label);
}

function segment(
    arena: ArenaLayout,
    name: "rowPtr" | "colIdx" | "weights" | "arcToEdge" | "edgeToArc",
): { byteOffset: number; byteLength: number } {
    const seg = arena.segments[name];
    expect(seg, name).not.toBeNull();
    return seg as { byteOffset: number; byteLength: number };
}

describe("GraphResidency: the upload contract (spec 11.3, gpu-upload.test.ts lines 254-666)", () => {
    it("the device accepts the format's 256-byte segment alignment and default-limit planning assumptions", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const { limits } = ctx.caps;
            expect(256 % limits.minStorageBufferOffsetAlignment).toBe(0);
            expect(limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(128 * MIB);
            expect(limits.maxBufferSize).toBeGreaterThanOrEqual(256 * MIB);
            expect(limits.maxComputeWorkgroupsPerDimension).toBeGreaterThanOrEqual(65_535);
            // Step 19 reads these two numbers off the NVIDIA and lavapipe runs to re-fix the caps tables
            console.warn(
                `[residency] device limits: maxBufferSize=${limits.maxBufferSize} maxStorageBufferBindingSize=${limits.maxStorageBufferBindingSize} vendor=${ctx.caps.vendor} architecture=${ctx.caps.architecture}`,
            );
            await ctx.allocator.check();
        });
    });

    it("arena path: ONE buffer of the hot prefix; per-segment bindings at segment.byteOffset - arena.byteOffset equal the CPU views", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = directedGraph();
            const arena = s.arena as ArenaLayout;
            const core = ctx.residency.core(s);
            expect(core.plan).toBe("arena");
            expect(core.serial).toBe(s.serial);
            expect(core.windows).toBeNull();
            expect(core.hasWeights).toBe(true);
            expect(core.arcToEdge).toBeNull();
            expect(core.edgeToArc).toBeNull();
            const { buffer } = core.rowPtr;
            expect(buffer.size).toBe(arena.hotByteLength);
            expect(buffer.label).toBe(`residency:core:${s.serial}:arena`);
            const colIdx = core.colIdx as Binding;
            const weights = core.weights as Binding;
            expect(colIdx.buffer).toBe(buffer);
            expect(weights.buffer).toBe(buffer);
            expect(core.rowPtr).toEqual({ buffer, offset: 0, size: segment(arena, "rowPtr").byteLength, window: null });
            expect(colIdx).toEqual({
                buffer,
                offset: segment(arena, "colIdx").byteOffset - arena.byteOffset,
                size: 4 * s.arcCount,
                window: null,
            });
            expect(weights).toEqual({
                buffer,
                offset: segment(arena, "weights").byteOffset - arena.byteOffset,
                size: 4 * s.arcCount,
                window: null,
            });
            expect(colIdx.offset % 256).toBe(0);
            expect(weights.offset % 256).toBe(0);
            await expectBindingU32(ctx, core.rowPtr, s.rowPtr, "rowPtr");
            await expectBindingU32(ctx, colIdx, s.colIdx, "colIdx");
            await expectBindingF32(ctx, weights, s.weights as F32, "weights");
            expect(ctx.residency.stats()).toEqual({
                buffers: 1,
                bytes: arena.hotByteLength,
                snapshots: 1,
                perSnapshot: [{ serial: s.serial, label: "directed", bytes: arena.hotByteLength, buffers: 1 }],
            });
            expect(ctx.residency.residentBytes).toBe(arena.hotByteLength);
            // a second core() finds the record: the same bindings, no second buffer
            expect(ctx.residency.core(s)).toEqual(core);
            expect(ctx.residency.stats().buffers).toBe(1);
            await ctx.allocator.check();
        });
    });

    it("cold segments: named in the first core() they ride the full arena; needed later they get their own buffer", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = directedGraph();
            const arena = s.arena as ArenaLayout;
            const full = ctx.residency.core(s, ["rowPtr", "colIdx", "arcToEdge", "edgeToArc"]);
            const { buffer } = full.rowPtr;
            expect(buffer.size).toBe(arena.byteLength);
            expect(full.weights).toBeNull();
            expect(full.hasWeights).toBe(false);
            const arcToEdge = full.arcToEdge as Binding;
            const edgeToArc = full.edgeToArc as Binding;
            expect(arcToEdge.buffer).toBe(buffer);
            expect(edgeToArc.buffer).toBe(buffer);
            expect(arcToEdge.offset).toBe(segment(arena, "arcToEdge").byteOffset - arena.byteOffset);
            expect(edgeToArc.offset).toBe(segment(arena, "edgeToArc").byteOffset - arena.byteOffset);
            await expectBindingU32(ctx, arcToEdge, s.arcToEdge, "arcToEdge");
            await expectBindingU32(ctx, edgeToArc, s.edgeToArc, "edgeToArc");
            // the weights segment sits inside the uploaded bytes: a later default need binds it with no upload
            const later = ctx.residency.core(s);
            expect(later.weights?.buffer).toBe(buffer);
            expect(later.hasWeights).toBe(true);
            expect(later.arcToEdge).toEqual(arcToEdge);
            expect(ctx.residency.stats().buffers).toBe(1);
            // a second snapshot: the hot prefix first, then arcToEdge on demand as its own zero-copy buffer
            const u = undirectedGraph();
            const uArena = u.arena as ArenaLayout;
            const hot = ctx.residency.core(u);
            expect(hot.rowPtr.buffer.size).toBe(uArena.hotByteLength);
            const cold = ctx.residency.core(u, ["rowPtr", "arcToEdge"]);
            expect(cold.rowPtr.buffer).toBe(hot.rowPtr.buffer);
            expect(cold.colIdx).toEqual(hot.colIdx);
            const uArc = cold.arcToEdge as Binding;
            expect(uArc.buffer).not.toBe(hot.rowPtr.buffer);
            expect(uArc).toEqual({ buffer: uArc.buffer, offset: 0, size: 4 * u.arcCount, window: null });
            expect(uArc.buffer.label).toBe(`residency:core:${u.serial}:arcToEdge`);
            await expectBindingU32(ctx, uArc, u.arcToEdge, "undirected arcToEdge");
            expect(cold.edgeToArc).toBeNull();
            const stats = ctx.residency.stats();
            expect(stats.snapshots).toBe(2);
            expect(stats.buffers).toBe(3);
            expect(stats.perSnapshot[1]).toEqual({
                serial: u.serial,
                label: "undirected",
                bytes: uArena.hotByteLength + 4 * u.arcCount,
                buffers: 2,
            });
            await ctx.allocator.check();
        });
    });

    it("perArray path: fromCsr (arena === null) and transpose() upload one buffer per array with whole-buffer bindings", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = csrSnapshotOf(KARATE_EDGES, { weighted: true });
            expect(s.arena).toBeNull();
            const core = ctx.residency.core(s);
            expect(core.plan).toBe("perArray");
            const colIdx = core.colIdx as Binding;
            const weights = core.weights as Binding;
            expect(new Set([core.rowPtr.buffer, colIdx.buffer, weights.buffer]).size).toBe(3);
            expect(core.rowPtr).toEqual({ buffer: core.rowPtr.buffer, offset: 0, size: 140, window: null });
            expect(core.rowPtr.buffer.size).toBe(140);
            expect(core.rowPtr.buffer.label).toBe(`residency:core:${s.serial}:rowPtr`);
            expect(colIdx.buffer.size).toBe(624);
            await expectBindingU32(ctx, core.rowPtr, s.rowPtr, "rowPtr");
            await expectBindingU32(ctx, colIdx, s.colIdx, "colIdx");
            await expectBindingF32(ctx, weights, s.weights as F32, "weights");
            expect(ctx.residency.stats()).toMatchObject({ buffers: 3, bytes: 140 + 624 + 624, snapshots: 1 });
            // transpose() of a directed snapshot adopts the reverse arrays as its core: no arena, a new serial
            const d = directedGraph();
            const transposed = d.transpose().snapshot;
            expect(transposed.arena).toBeNull();
            expect(transposed.serial).not.toBe(d.serial);
            const tc = ctx.residency.core(transposed);
            expect(tc.plan).toBe("perArray");
            await expectBindingU32(ctx, tc.rowPtr, transposed.rowPtr, "transposed rowPtr");
            await expectBindingU32(ctx, tc.colIdx as Binding, transposed.colIdx, "transposed colIdx");
            await expectBindingF32(ctx, tc.weights as Binding, transposed.weights as F32, "transposed weights");
            expectBitwiseEqual(transposed.colIdx, d.reverse().colIdx, "the transposed core is the reverse view");
            expect(ctx.residency.stats().snapshots).toBe(2);
            expect(ctx.residency.stats().buffers).toBe(6);
            await ctx.allocator.check();
        });
    });

    it("undirected: reverse() aliases the forward arrays, inDegree() is outDegree(), both degree orders are one view", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const u = undirectedGraph();
            expect(u.reverse().rowPtr).toBe(u.rowPtr);
            expect(u.reverse().colIdx).toBe(u.colIdx);
            ctx.residency.core(u);
            const out = ctx.residency.view(u, "outDegree");
            const inn = ctx.residency.view(u, "inDegree");
            expect(out.view).toBe("outDegree");
            expect(inn.view).toBe("inDegree");
            expect(Object.keys(out.bindings)).toEqual(["outDegree"]);
            expect(Object.keys(inn.bindings)).toEqual(["inDegree"]);
            expect(inn.bindings.inDegree.buffer).toBe(out.bindings.outDegree.buffer);
            expect(out.scalars).toEqual({});
            expect(out.bindings.outDegree.buffer.label).toBe(`residency:view:${u.serial}:outDegree`);
            await expectBindingU32(ctx, out.bindings.outDegree, u.outDegree(), "outDegree");
            const order = ctx.residency.view(u, "degreeOrder");
            const reverseOrder = ctx.residency.view(u, "reverseDegreeOrder");
            expect(Object.keys(order.bindings)).toEqual(["perm"]);
            expect(reverseOrder.bindings.perm.buffer).toBe(order.bindings.perm.buffer);
            expect(order.scalars).toEqual({ segmentOffsets: Array.from(u.degreeOrder().segmentOffsets) });
            expect(order.scalars.segmentOffsets).toHaveLength(5);
            expect(order.scalars.segmentOffsets[4]).toBe(u.nodeCount);
            await expectBindingU32(ctx, order.bindings.perm, u.degreeOrder().perm, "perm");
            expect(ctx.residency.stats().buffers).toBe(3);
            await ctx.allocator.check();
        });
    });

    it("identity permutations are never materialised: the arcToEdge / edgeToArc getters are not read, byteLength({ views: true }) is unchanged", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(pathEdges(50), { directed: true });
            expect(s.flags.arcToEdgeIsIdentity).toBe(true);
            const before = s.byteLength({ views: true });
            const arcSpy = vi.spyOn(GraphSnapshot.prototype, "arcToEdge", "get");
            const edgeSpy = vi.spyOn(GraphSnapshot.prototype, "edgeToArc", "get");
            try {
                const core = ctx.residency.core(s, ["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"]);
                expect(core.arcToEdge).toBeNull();
                expect(core.edgeToArc).toBeNull();
                expect(core.weights).toBeNull();
                expect(core.hasWeights).toBe(false);
                expect(s.byteLength({ views: true })).toBe(before);
                ctx.residency.view(s, "outDegree");
                expect(arcSpy).not.toHaveBeenCalled();
                expect(edgeSpy).not.toHaveBeenCalled();
            } finally {
                arcSpy.mockRestore();
                edgeSpy.mockRestore();
            }
            await expectBindingU32(ctx, ctx.residency.core(s).colIdx as Binding, s.colIdx, "colIdx");
            expect(ctx.residency.stats().buffers).toBe(2);
            await ctx.allocator.check();
        });
    });

    it("packed columns through column(): u8 -> padded u32 words, bool -> bitmap words, f64 -> the cached f32 copy, u8 x3", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const rows = 1000;
            const b = new GraphBuilder({ directed: true });
            b.addAnonymousNodes(rows);
            b.declareNodeColumn({ name: "bytes", dtype: "u8", nullable: false });
            b.declareNodeColumn({ name: "flag", dtype: "bool", nullable: false });
            b.declareNodeColumn({ name: "score", dtype: "f64", nullable: false });
            b.declareNodeColumn({ name: "triple", dtype: "u8", components: 3, nullable: false });
            b.declareNodeColumn({ name: "name", dtype: "string" });
            for (let i = 0; i < rows; i++) {
                b.setNodeValue("bytes", i, (i * 37) % 256);
                b.setNodeValue("flag", i, i % 3 === 0);
                b.setNodeValue("score", i, i / 8 + 0.25);
                b.setNodeValue("triple", i, [i % 256, Math.floor(i / 256) % 256, 7]);
            }
            const s = b.freeze();
            const bytes = ctx.residency.column(s.nodes, "bytes", s);
            expect(bytes.eligibility).toBe("packed");
            expect(bytes.components).toBe(1);
            expect(bytes.version).toBe(0);
            expect(bytes.column).toBe(s.nodes.require("bytes"));
            expect(bytes.binding.size).toBe(1000);
            expect(bytes.binding.buffer.label).toBe(`residency:column:${s.serial}:node.bytes`);
            await expectBindingU32(ctx, bytes.binding, s.nodes.requireTyped("bytes", "u8").paddedU32View(), "bytes");
            const flag = ctx.residency.column(s.nodes, "flag", s);
            expect(flag.eligibility).toBe("packed");
            expect(flag.binding.size).toBe(4 * Math.ceil(rows / 32));
            await expectBindingU32(ctx, flag.binding, s.nodes.requireTyped("flag", "bool").data, "flag");
            const score = ctx.residency.column(s.nodes, "score", s);
            expect(score.eligibility).toBe("convert");
            expect(score.binding.size).toBe(4 * rows);
            await expectBindingF32(
                ctx,
                score.binding,
                new Float32Array(s.nodes.requireTyped("score", "f64").data),
                "score",
            );
            const triple = ctx.residency.column(s.nodes, "triple", s);
            expect(triple.components).toBe(3);
            expect(triple.binding.size).toBe(3000);
            await expectBindingU32(ctx, triple.binding, s.nodes.requireTyped("triple", "u8").paddedU32View(), "triple");
            expect(formatCode(() => ctx.residency.column(s.nodes, "name", s))).toBe("E_GPU_INELIGIBLE");
            expect(formatCode(() => ctx.residency.column(s.nodes, "missing", s))).toBe("E_UNKNOWN_COLUMN");
            expect(ctx.residency.stats()).toMatchObject({ buffers: 4, snapshots: 1 });
            await ctx.allocator.check();
        });
    });

    it("the same array object uploads once: a view twice, an ad hoc array twice, a u8 column twice", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            ctx.residency.core(s);
            const a = ctx.residency.view(s, "outDegree");
            const b = ctx.residency.view(s, "outDegree");
            expect(b.bindings.outDegree.buffer).toBe(a.bindings.outDegree.buffer);
            const mask = new Uint32Array(2);
            const m1 = ctx.residency.array(mask, "mask", s);
            const m2 = ctx.residency.array(mask, "mask", s);
            expect(m2.binding.buffer).toBe(m1.binding.buffer);
            const rich = fromBytes(new Uint8Array(readFileSync(FIXTURE)));
            // gpuView() of a u8 column is a fresh view on every call: the Column object is the key (PLAN DECISION 8)
            expect(rich.nodes.gpuView("byte")).not.toBe(rich.nodes.gpuView("byte"));
            const c1 = ctx.residency.column(rich.nodes, "byte", rich);
            const c2 = ctx.residency.column(rich.nodes, "byte", rich);
            expect(c2.binding.buffer).toBe(c1.binding.buffer);
            expect(ctx.residency.stats()).toMatchObject({ buffers: 4, snapshots: 2 });
            await ctx.allocator.check();
        });
    });

    it("a column version bump re-uploads in place (same buffer, new version); an unchanged column is never re-uploaded", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const b = new GraphBuilder({ directed: false });
            b.addAnonymousNodes(3);
            b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, nullable: false, mutable: true });
            b.declareNodeColumn({ name: "score", dtype: "f64", nullable: false, mutable: true });
            for (let i = 0; i < 3; i++) {
                b.setNodeValue("pos", i, [i, 2 * i, 3 * i]);
                b.setNodeValue("score", i, i + 0.25);
            }
            const s = b.freeze();
            const first = ctx.residency.column(s.nodes, "pos", s);
            expect(first.version).toBe(0);
            expect(first.components).toBe(3);
            expect(first.eligibility).toBe("direct");
            expect(ctx.residency.column(s.nodes, "pos", s).binding.buffer).toBe(first.binding.buffer);
            expect(ctx.residency.stats().buffers).toBe(1);
            const pos = s.nodes.requireTyped("pos", "f32");
            pos.mutableData()[0] = 42;
            pos.markDirty();
            expect(pos.version).toBe(1);
            const bumped = ctx.residency.column(s.nodes, "pos", s);
            expect(bumped.binding.buffer).toBe(first.binding.buffer);
            expect(bumped.version).toBe(1);
            const back = await readF32(ctx, bumped.binding.buffer, 9, 0);
            expect(back[0]).toBe(42);
            expect(back[4]).toBe(2);
            expect(ctx.residency.stats().buffers).toBe(1);
            // an f64 column: markDirty makes a NEW f32 copy, but the Column object is the key, so it is an in-place write too
            const c1 = ctx.residency.column(s.nodes, "score", s);
            const score = s.nodes.requireTyped("score", "f64");
            score.mutableData()[1] = 9.5;
            score.markDirty();
            const c2 = ctx.residency.column(s.nodes, "score", s);
            expect(c2.binding.buffer).toBe(c1.binding.buffer);
            expect(c2.version).toBe(1);
            expect(Array.from(await readF32(ctx, c2.binding.buffer, 3, 0))).toEqual([0.25, 9.5, 2.25]);
            expect(ctx.residency.stats().buffers).toBe(2);
            await ctx.allocator.check();
        });
    });

    it("ctx.release destroys every buffer of the snapshot, leaves allocator.liveBuffers at 0 and trims the pool", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const base = ctx.allocator.liveBuffers;
            expect(base, "the context creates no allocator-tracked buffer of its own").toBe(0);
            const s = directedGraph();
            ctx.residency.core(s, ["rowPtr", "colIdx", "weights", "arcToEdge"]);
            ctx.residency.view(s, "outDegree");
            ctx.residency.view(s, "inDegree");
            ctx.residency.column(s.edges, "cost", s);
            ctx.residency.array(new Float32Array(N), "mass", s);
            expect(ctx.residency.stats()).toMatchObject({ buffers: 5, snapshots: 1 });
            expect(ctx.allocator.liveBuffers).toBe(5);
            const scratch = ctx.pool.acquire(4096, STORAGE, "scratch");
            ctx.pool.release(scratch);
            expect(ctx.pool.idleBytes).toBe(4096);
            expect(ctx.allocator.liveBuffers).toBe(6);
            ctx.release(s);
            expect(ctx.residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            expect(ctx.residency.residentBytes).toBe(0);
            expect(ctx.allocator.liveBuffers).toBe(0);
            expect(ctx.pool.idleBytes).toBe(0);
            expect(ctx.pool.liveBytes).toBe(0);
            expect(ctx.residency.isReleased(s.serial)).toBe(true);
            await ctx.allocator.check();
        });
    });

    it("E_RELEASED: a live user sees isReleased; view / column / array on the released snapshot throw; core() re-uploads", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES, { label: "k" });
            const core = ctx.residency.core(s);
            expect(ctx.residency.isReleased(s.serial)).toBe(false);
            ctx.release(s);
            expect(ctx.residency.isReleased(s.serial)).toBe(true);
            for (const call of [
                () => ctx.residency.view(s, "outDegree"),
                () => ctx.residency.column(s.nodes, "anything", s),
                () => ctx.residency.array(new Uint32Array(4), "m", s),
            ]) {
                const err = caught(call);
                expect(err.code).toBe("E_RELEASED");
                expect(err.details).toEqual({ serial: s.serial });
            }
            expect(ctx.residency.stats().snapshots).toBe(0);
            const again = ctx.residency.core(s);
            expect(again.rowPtr.buffer).not.toBe(core.rowPtr.buffer);
            expect(ctx.residency.isReleased(s.serial)).toBe(false);
            await expectBindingU32(ctx, again.rowPtr, s.rowPtr, "rowPtr after re-upload");
            expect(ctx.residency.view(s, "outDegree").bindings.outDegree.size).toBe(4 * 34);
            expect(ctx.residency.stats()).toMatchObject({ buffers: 2, snapshots: 1 });
            await ctx.allocator.check();
        });
    });

    it("release is idempotent and safe on a snapshot never uploaded", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            const other = snapshotOf(pathEdges(5));
            expect(() => {
                ctx.release(other);
                ctx.release(other);
            }).not.toThrow();
            expect(ctx.residency.isReleased(other.serial)).toBe(false);
            ctx.residency.core(s);
            ctx.release(s);
            ctx.release(s);
            expect(ctx.residency.stats().snapshots).toBe(0);
            expect(ctx.residency.isReleased(s.serial)).toBe(true);
            expect(ctx.residency.isReleased(123_456_789)).toBe(false);
            await ctx.allocator.check();
        });
    });

    it("withColumns() siblings share one record: one core buffer, one stats entry, a release of either tombstones both", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            const sibling = s.withColumns({ extra: new Float32Array(34) });
            expect(sibling.serial).toBe(s.serial);
            expect(sibling).not.toBe(s);
            const a = ctx.residency.core(s);
            const b = ctx.residency.core(sibling);
            expect(b.rowPtr.buffer).toBe(a.rowPtr.buffer);
            const extra = ctx.residency.column(sibling.nodes, "extra", sibling);
            expect(extra.binding.size).toBe(4 * 34);
            expect(ctx.residency.stats()).toMatchObject({ buffers: 2, snapshots: 1 });
            ctx.release(sibling);
            expect(ctx.residency.isReleased(s.serial)).toBe(true);
            expect(caught(() => ctx.residency.view(s, "outDegree")).code).toBe("E_RELEASED");
            expect(ctx.residency.stats().buffers).toBe(0);
            await ctx.allocator.check();
        });
    });

    it("warns once through options.warn when more than warnUnreleasedSnapshots snapshots are resident", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const warn = vi.fn();
            const residency = new GraphResidency(ctx.device, ctx.caps, ctx.allocator, {
                warnUnreleasedSnapshots: 2,
                warn,
            });
            const snapshots = [0, 1, 2, 3].map((i) => snapshotOf(pathEdges(3 + i)));
            residency.core(snapshots[0]);
            residency.core(snapshots[1]);
            expect(warn).not.toHaveBeenCalled();
            residency.core(snapshots[2]);
            expect(warn).toHaveBeenCalledTimes(1);
            expect(String(warn.mock.calls[0][0])).toMatch(/3 snapshots/);
            expect(String(warn.mock.calls[0][0])).toMatch(/warnUnreleasedSnapshots = 2/);
            residency.core(snapshots[3]);
            expect(warn).toHaveBeenCalledTimes(1);
            expect(residency.stats().snapshots).toBe(4);
            residency.destroyAll();
            expect(residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            const disposed = caught(() => residency.core(snapshots[0]));
            expect(disposed.code).toBe("E_DISPOSED");
            expect(disposed.details.label).toBe("residency");
            await ctx.allocator.check();
        });
    });

    it("a container adopted by fromBytes uploads straight from the file bytes (arena.byteOffset !== 0), also through a view at byte 8", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const golden = new Uint8Array(readFileSync(FIXTURE));
            const s = fromBytes(golden);
            const arena = s.arena as ArenaLayout;
            expect(arena.byteOffset).toBeGreaterThan(0);
            expect(arena.byteOffset % 256).toBe(0);
            expect(Array.from(s.rowPtr)).toEqual([0, 3, 6, 8, 9]);
            expect(Array.from(s.colIdx)).toEqual([1, 1, 3, 0, 0, 2, 1, 2, 0]);
            expect(Array.from(s.weights as F32)).toEqual([0.5, Math.fround(0.1), 1, 0.5, Math.fround(0.1), 2, 2, 3, 1]);
            const core = ctx.residency.core(s);
            expect(core.plan).toBe("arena");
            expect(core.rowPtr.buffer.size).toBe(arena.hotByteLength);
            await expectBindingU32(ctx, core.rowPtr, s.rowPtr, "container rowPtr");
            await expectBindingU32(ctx, core.colIdx as Binding, s.colIdx, "container colIdx");
            await expectBindingF32(ctx, core.weights as Binding, s.weights as F32, "container weights");
            // every GPU-eligible dtype of the container through column()
            const expected = [
                ["byte", "packed", 1],
                ["flag", "packed", 1],
                ["solid", "direct", 1],
                ["score", "direct", 1],
                ["pos", "direct", 3],
                ["end", "convert", 1],
                ["cat", "direct", 1],
            ] as const;
            for (const [name, eligibility, components] of expected) {
                const c = ctx.residency.column(s.nodes, name, s);
                expect(c.eligibility, name).toBe(eligibility);
                expect(c.components, name).toBe(components);
                expect(c.binding.size, name).toBe(s.nodes.gpuView(name).byteLength);
            }
            await expectBindingU32(
                ctx,
                ctx.residency.column(s.nodes, "solid", s).binding,
                s.nodes.requireTyped("solid", "u32").data,
                "solid",
            );
            expect(Array.from(s.nodes.requireTyped("solid", "u32").data)).toEqual([0, 100, 200, 300]);
            await expectBindingF32(
                ctx,
                ctx.residency.column(s.nodes, "pos", s).binding,
                s.nodes.requireTyped("pos", "f32").data,
                "pos",
            );
            expect(formatCode(() => ctx.residency.column(s.nodes, "label", s))).toBe("E_GPU_INELIGIBLE");
            expect(ctx.residency.stats()).toMatchObject({ buffers: 8, snapshots: 1 });
            // the same container through a view at byte 8 of a larger buffer (gpu-upload.test.ts lines 616-666)
            const shifted = new Uint8Array(golden.byteLength + 8);
            shifted.set(golden, 8);
            const s8 = fromBytes(new Uint8Array(shifted.buffer, 8, golden.byteLength), { validate: "structure" });
            const arena8 = s8.arena as ArenaLayout;
            expect(arena8.buffer).toBe(shifted.buffer);
            expect(arena8.byteOffset).toBe(arena.byteOffset + 8);
            const core8 = ctx.residency.core(s8);
            await expectBindingU32(ctx, core8.rowPtr, s8.rowPtr, "shifted rowPtr");
            await expectBindingU32(ctx, core8.colIdx as Binding, s8.colIdx, "shifted colIdx");
            await expectBindingF32(ctx, core8.weights as Binding, s8.weights as F32, "shifted weights");
            expectBitwiseEqual(s8.colIdx, s.colIdx, "the two decodings agree");
            expect(ctx.residency.stats().snapshots).toBe(2);
            await ctx.allocator.check();
        });
    });

    it("array(): keyed on the object, registered against an owner or destroyed by the caller; invalid inputs rejected", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            const orphanData = new Float32Array([1, 2, 3, 4]);
            const orphan = ctx.residency.array(orphanData, "orphan");
            expect(orphan.owner).toBeNull();
            expect(orphan.byteLength).toBe(16);
            expect(orphan.binding).toEqual({ buffer: orphan.binding.buffer, offset: 0, size: 16, window: null });
            expect(orphan.binding.buffer.label).toBe("residency:array:orphan");
            await expectBindingF32(ctx, orphan.binding, orphanData, "orphan");
            expect(ctx.residency.stats()).toEqual({ buffers: 1, bytes: 16, snapshots: 0, perSnapshot: [] });
            expect(ctx.residency.array(orphanData, "orphan").binding.buffer).toBe(orphan.binding.buffer);
            orphan.destroy();
            orphan.destroy();
            expect(ctx.residency.stats().buffers).toBe(0);
            const owned = ctx.residency.array(new Uint32Array(2), "mask", s);
            expect(owned.owner).toBe(s);
            expect(ctx.residency.stats()).toMatchObject({ buffers: 1, snapshots: 1 });
            ctx.release(s);
            expect(ctx.residency.stats().buffers).toBe(0);
            expect(() => {
                owned.destroy();
            }).not.toThrow();
            const s2 = snapshotOf(pathEdges(3));
            const early = ctx.residency.array(new Uint32Array(1), "early", s2);
            early.destroy();
            expect(ctx.residency.stats()).toMatchObject({ buffers: 0, snapshots: 1 });
            ctx.release(s2);
            const bad: [Parameters<typeof ctx.residency.array>[0], string][] = [
                [new Uint32Array(0), "key"],
                [new Uint8Array(3), "key"],
                [new Float32Array(new SharedArrayBuffer(16)) as unknown as F32, "key"],
            ];
            for (const [key, argument] of bad) {
                const err = caught(() => ctx.residency.array(key, "bad"));
                expect(err.code).toBe("E_INVALID_ARGUMENT");
                expect(err.details.argument).toBe(argument);
            }
            await ctx.allocator.check();
        });
    });

    it("view(): P7 names and packViews are E_UNSUPPORTED, an empty snapshot is E_INVALID_ARGUMENT; core() of arcCount 0 binds rowPtr only", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            for (const name of ["reverse", "coo", "edgeList", "mate"] as const) {
                const err = caught(() => ctx.residency.view(s, name));
                expect(err.code).toBe("E_UNSUPPORTED");
                expect(err.details.feature).toBe(`view:${name}`);
                expect(err.details.option).toBeUndefined();
            }
            const packed = caught(() => ctx.residency.view(s, "outDegree", { packViews: true }));
            expect(packed.code).toBe("E_UNSUPPORTED");
            expect(packed.details.option).toBe("packViews");
            expect(packed.details.feature).toBeUndefined();
            expect(ctx.residency.view(s, "outDegree", { packViews: false }).view).toBe("outDegree");
            const empty = snapshotOf([], { nodeCount: 0 });
            const err = caught(() => ctx.residency.view(empty, "outDegree"));
            expect(err.code).toBe("E_INVALID_ARGUMENT");
            expect(err.details.argument).toBe("snapshot");
            const emptyCore = ctx.residency.core(empty);
            expect(emptyCore.rowPtr.size).toBe(4);
            expect(emptyCore.colIdx).toBeNull();
            const five = snapshotOf([], { nodeCount: 5 });
            const core = ctx.residency.core(five, ["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"]);
            expect(core.plan).toBe("arena");
            expect(core.rowPtr.size).toBe(24);
            expect(core.colIdx).toBeNull();
            expect(core.weights).toBeNull();
            expect(core.arcToEdge).toBeNull();
            expect(core.edgeToArc).toBeNull();
            expect(core.hasWeights).toBe(false);
            await expectBindingU32(ctx, core.rowPtr, five.rowPtr, "rowPtr of an arc-less graph");
            // karate's outDegree view (the packViews: false call), the empty core and the five-node core
            expect(ctx.residency.stats()).toMatchObject({ buffers: 3, snapshots: 3 });
            await ctx.allocator.check();
        });
    });

    it("a windowed plan is E_TOO_LARGE { path: 'windowed' } until P4 executes windows (PLAN DECISION 2)", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            // karate: rowPtr (140 B) fits a 256-byte binding, colIdx (624 B) does not
            const caps = fakeCaps(ctx.caps, { maxStorageBufferBindingSize: 256 });
            const residency = new GraphResidency(ctx.device, caps, ctx.allocator, { warnUnreleasedSnapshots: 2 });
            const s = snapshotOf(KARATE_EDGES);
            const err = caught(() => residency.core(s));
            expect(err.code).toBe("E_TOO_LARGE");
            expect(err.details).toEqual({ needed: 624, limit: 256, path: "windowed", algorithm: null });
            expect(residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            expect(ctx.allocator.liveBuffers).toBe(0);
            residency.destroyAll();
            await ctx.allocator.check();
        });
    });

    it("a detached snapshot is E_SNAPSHOT { reason: 'detached' } from core / view / column / array", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES);
            const arena = s.arena as ArenaLayout;
            structuredClone(arena.buffer, { transfer: [arena.buffer] });
            expect(s.detached).toBe(true);
            for (const call of [
                () => ctx.residency.core(s),
                () => ctx.residency.view(s, "outDegree"),
                () => ctx.residency.column(s.nodes, "x", s),
                () => ctx.residency.array(new Uint32Array(1), "m", s),
            ]) {
                const err = caught(call);
                expect(err.code).toBe("E_SNAPSHOT");
                expect(err.details).toEqual({ reason: "detached", serial: s.serial });
            }
            expect(ctx.residency.stats().snapshots).toBe(0);
            await ctx.allocator.check();
        });
    });

    it("clearOnLoss drops every record without destroying; destroyAll destroys everything and disposes", async (t: TestContext) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const residency = new GraphResidency(ctx.device, ctx.caps, ctx.allocator, { warnUnreleasedSnapshots: 2 });
            const s = snapshotOf(KARATE_EDGES);
            const core = residency.core(s);
            const degree = residency.view(s, "outDegree").bindings.outDegree.buffer;
            const orphan = residency.array(new Uint32Array(4), "o");
            const live = ctx.allocator.liveBuffers;
            expect(live).toBe(3);
            residency.clearOnLoss();
            expect(residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            // nothing destroyed: on a real loss the buffers are gone with the device (spec 5.7)
            expect(ctx.allocator.liveBuffers).toBe(live);
            expect(residency.isReleased(s.serial)).toBe(false);
            const again = residency.core(s);
            expect(again.rowPtr.buffer).not.toBe(core.rowPtr.buffer);
            expect(residency.stats()).toMatchObject({ buffers: 1, snapshots: 1 });
            expect(() => {
                orphan.destroy();
            }).not.toThrow();
            residency.destroyAll();
            expect(residency.stats().buffers).toBe(0);
            expect(ctx.allocator.liveBuffers).toBe(live);
            expect(caught(() => residency.core(s)).code).toBe("E_DISPOSED");
            expect(caught(() => residency.view(s, "outDegree")).details.label).toBe("residency");
            expect(caught(() => residency.column(s.nodes, "x", s)).code).toBe("E_DISPOSED");
            expect(caught(() => residency.array(new Uint32Array(1), "a")).code).toBe("E_DISPOSED");
            expect(() => {
                residency.destroyAll();
            }).not.toThrow();
            // the three buffers clearOnLoss orphaned in this synthetic (device not lost) scenario: the test destroys them
            core.rowPtr.buffer.destroy();
            degree.destroy();
            orphan.binding.buffer.destroy();
            await ctx.allocator.check();
        });
    });
});

describe("P2: residentBytes and stats().perSnapshot (spec 4.1)", () => {
    it("residentBytes equals the sum of the uploaded byte lengths; perSnapshot itemises them; release zeroes them", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES, { label: "karate" });
            if (s.arena === null) {
                throw new Error("karate is expected to carry an arena");
            }
            // karate, undirected: rowPtr 35 x 4 = 140 B at 0, colIdx 156 x 4 = 624 B at 256 -> hot prefix 880 B (unweighted)
            expect(s.nodeCount).toBe(34);
            expect(s.arcCount).toBe(156);
            expect(s.arena.hotByteLength).toBe(880);
            expect(ctx.residency.residentBytes).toBe(0);

            ctx.residency.core(s);
            expect(ctx.residency.residentBytes).toBe(880);
            ctx.residency.view(s, "outDegree"); // 34 x 4 = 136 B
            expect(ctx.residency.residentBytes).toBe(1016);
            s.nodes.set("mass", new Float32Array(34));
            ctx.residency.column(s.nodes, "mass", s); // 34 x 4 = 136 B
            expect(ctx.residency.residentBytes).toBe(1152);
            const adHoc = ctx.residency.array(new Uint32Array(16), "adhoc", s); // 64 B
            expect(adHoc.byteLength).toBe(64);
            expect(ctx.residency.residentBytes).toBe(1216);
            expect(ctx.residency.stats()).toEqual({
                buffers: 4,
                bytes: 1216,
                snapshots: 1,
                perSnapshot: [{ serial: s.serial, label: "karate", bytes: 1216, buffers: 4 }],
            });

            // a second snapshot adds its own bytes: path(10) undirected = 10 nodes, 18 arcs: rowPtr 44 B at 0, colIdx 72 B at 256 -> 328 B
            const p = snapshotOf(pathEdges(10), { label: "path10" });
            if (p.arena === null) {
                throw new Error("path10 is expected to carry an arena");
            }
            expect(p.nodeCount).toBe(10);
            expect(p.arcCount).toBe(18);
            expect(p.arena.hotByteLength).toBe(328);
            ctx.residency.core(p);
            expect(ctx.residency.residentBytes).toBe(1216 + 328);
            expect(ctx.residency.residentBytes).toBe(1544);
            expect(ctx.residency.stats().snapshots).toBe(2);
            expect(ctx.residency.stats().perSnapshot.map((row) => row.bytes)).toEqual([1216, 328]);

            ctx.release(s);
            expect(ctx.residency.residentBytes).toBe(328);
            expect(ctx.residency.stats()).toEqual({
                buffers: 1,
                bytes: 328,
                snapshots: 1,
                perSnapshot: [{ serial: p.serial, label: "path10", bytes: 328, buffers: 1 }],
            });
            ctx.release(p);
            expect(ctx.residency.residentBytes).toBe(0);
            expect(ctx.residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            // idempotent and safe on a snapshot never uploaded
            ctx.release(p);
            ctx.release(snapshotOf(pathEdges(3)));
            expect(ctx.residency.residentBytes).toBe(0);
            await Promise.resolve();
        });
    });

    it("clearOnLoss() forgets every record without destroying and zeroes residentBytes", async (t) => {
        requireGpu(t);
        await withContext(undefined, async (ctx) => {
            const s = snapshotOf(KARATE_EDGES, { label: "karate" });
            ctx.residency.core(s);
            ctx.residency.view(s, "outDegree");
            const live = ctx.allocator.liveBuffers;
            expect(ctx.residency.residentBytes).toBe(1016);
            ctx.residency.clearOnLoss();
            expect(ctx.residency.residentBytes).toBe(0);
            expect(ctx.residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
            // nothing was destroyed (the device is what frees them on loss); the context's dispose() destroys the device
            expect(ctx.allocator.liveBuffers).toBe(live);
            expect(ctx.residency.isReleased(s.serial)).toBe(false);
            await Promise.resolve();
        });
    });
});
