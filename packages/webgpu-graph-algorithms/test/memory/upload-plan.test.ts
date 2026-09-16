/**
 * Pure planner tests (spec 11.3 row "Planner unit tests"; contract 5.5): planUpload over the four caps tables x
 * { arena, no arena } x { fits, exceeds buffer, exceeds binding } with hand-computed plans, the spec 4.2 tier
 * arithmetic (100k / 1M, 1M / 10M, 10M / 100M), planArcWindows (start % 64 === 0 through `%`, a hub row split across
 * windows, a synthetic rowPtr above 2^31 arcs), arcsPerWindowFor, and the pure checks of the test helpers this task
 * ships (caps-tables, matchers, graphs -- test/helpers/** is outside the node project's include globs and contract
 * 1.2 declares no helper test file). No device is touched.
 */

import { type ArenaLayout, type CoreArrayName, type GraphSnapshot } from "@graphty/graph-format";
import * as fc from "fast-check";

import { ARC_WINDOW_ALIGN } from "../../src/constants.js";
import { isWebGpuGraphError } from "../../src/errors.js";
import { arcsPerWindowFor, planArcWindows, planUpload, type UploadPlan } from "../../src/memory/upload-plan.js";
import { type ArcWindow } from "../../src/types/memory.js";
import {
    CAPS_INTEL_XE,
    CAPS_LAVAPIPE,
    CAPS_NVIDIA_4070,
    CAPS_SPEC_DEFAULT,
    CAPS_SWIFTSHADER,
    CAPS_TABLES,
    fakeCaps,
} from "../helpers/caps-tables.js";
import {
    completeEdges,
    csrSnapshotOf,
    cycleEdges,
    fixture,
    FIXTURE_NAMES,
    gridEdges,
    KARATE_EDGES,
    pathEdges,
    randomEdges,
    randomEdgesLoose,
    rmatEdges,
    snapshotOf,
    starEdges,
} from "../helpers/graphs.js";
import { expectAllClose, expectBitwiseEqual, flooredRelError, maxRelError } from "../helpers/matchers.js";

const MIB = 1024 * 1024;
const ALL_CORE: readonly CoreArrayName[] = ["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"];
const DEFAULT_NEED: readonly CoreArrayName[] = ["rowPtr", "colIdx", "weights"];

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

// ============================================================ fake snapshots for the tier arithmetic

/** graph-format's builder/arena.ts layout: each present segment starts at the cursor rounded up to 256; no trailing padding. */
function layoutArena(byteLengths: Readonly<Record<CoreArrayName, number>>): ArenaLayout {
    const segments: Record<CoreArrayName, { byteOffset: number; byteLength: number } | null> = {
        rowPtr: null,
        colIdx: null,
        weights: null,
        arcToEdge: null,
        edgeToArc: null,
    };
    let cursor = 0;
    let hot = 0;
    ALL_CORE.forEach((name, i) => {
        const len = byteLengths[name];
        if (len === 0) {
            return;
        }
        const offset = Math.ceil(cursor / 256) * 256;
        segments[name] = { byteOffset: offset, byteLength: len };
        cursor = offset + len;
        if (i <= 2) {
            hot = cursor;
        }
    });
    return {
        buffer: new ArrayBuffer(0),
        byteOffset: 0,
        byteLength: cursor,
        alignment: 256,
        segments,
        hotByteLength: hot,
    };
}

interface TierSpec {
    readonly n: number;
    readonly edges: number;
    readonly directed?: boolean;
    readonly weighted?: boolean;
    readonly identity?: boolean;
    readonly arena?: boolean;
    readonly rowPtr?: Uint32Array;
}

/** A snapshot descriptor with the fields planUpload reads (arena, counts, flags, rowPtr); nothing else exists on it. */
function fakeSnapshot(spec: TierSpec): GraphSnapshot {
    const directed = spec.directed ?? false;
    const weighted = spec.weighted ?? true;
    const identity = spec.identity ?? false;
    const arcs = directed ? spec.edges : 2 * spec.edges;
    const lengths: Record<CoreArrayName, number> = {
        rowPtr: 4 * (spec.n + 1),
        colIdx: 4 * arcs,
        weights: weighted ? 4 * arcs : 0,
        arcToEdge: identity ? 0 : 4 * arcs,
        edgeToArc: identity ? 0 : 4 * spec.edges,
    };
    const fake = {
        serial: 0,
        label: "fake",
        directed,
        nodeCount: spec.n,
        edgeCount: spec.edges,
        arcCount: arcs,
        rowPtr: spec.rowPtr ?? new Uint32Array(0),
        arena: (spec.arena ?? true) ? layoutArena(lengths) : null,
        flags: {
            multigraph: false,
            hasSelfLoops: false,
            arcToEdgeIsIdentity: identity,
            weighted,
            allWeightsOne: !weighted,
            nonNegativeWeights: true,
            finiteWeights: true,
        },
    };
    return fake as unknown as GraphSnapshot;
}

function uniformRowPtr(n: number, perRow: number): Uint32Array {
    const rowPtr = new Uint32Array(n + 1);
    for (let i = 0; i <= n; i++) {
        rowPtr[i] = perRow * i;
    }
    return rowPtr;
}

const TIER_100K: TierSpec = { n: 100_000, edges: 1_000_000 };
const TIER_1M: TierSpec = { n: 1_000_000, edges: 10_000_000 };
const TIER_10M: TierSpec = { n: 10_000_000, edges: 100_000_000 };
let rowPtr1M: Uint32Array;
let rowPtr10M: Uint32Array;

beforeAll(() => {
    rowPtr1M = uniformRowPtr(1_000_000, 20);
    rowPtr10M = uniformRowPtr(10_000_000, 20);
});

/** The six windows of the 10M / 100M colIdx at 33,554,432 arcs per window over rows of 20 arcs (PLAN DECISION 3). */
const WINDOWS_10M_SINGLE_BUFFER: readonly ArcWindow[] = [
    { start: 0, end: 33_554_420, rowFirst: 0, rowLast: 1_677_720, bufferIndex: 0, offset: 0 },
    {
        start: 33_554_368,
        end: 67_108_800,
        rowFirst: 1_677_721,
        rowLast: 3_355_439,
        bufferIndex: 0,
        offset: 134_217_472,
    },
    {
        start: 67_108_800,
        end: 100_663_220,
        rowFirst: 3_355_440,
        rowLast: 5_033_160,
        bufferIndex: 0,
        offset: 268_435_200,
    },
    {
        start: 100_663_168,
        end: 134_217_600,
        rowFirst: 5_033_161,
        rowLast: 6_710_879,
        bufferIndex: 0,
        offset: 402_652_672,
    },
    {
        start: 134_217_600,
        end: 167_772_020,
        rowFirst: 6_710_880,
        rowLast: 8_388_600,
        bufferIndex: 0,
        offset: 536_870_400,
    },
    {
        start: 167_771_968,
        end: 200_000_000,
        rowFirst: 8_388_601,
        rowLast: 9_999_999,
        bufferIndex: 0,
        offset: 671_087_872,
    },
];

/** The same windows placed into three 256 MiB-bounded buffers (two windows each; the third holds 263,129,600 B). */
const WINDOWS_10M_SPLIT: readonly ArcWindow[] = [
    { start: 0, end: 33_554_420, rowFirst: 0, rowLast: 1_677_720, bufferIndex: 0, offset: 0 },
    {
        start: 33_554_368,
        end: 67_108_800,
        rowFirst: 1_677_721,
        rowLast: 3_355_439,
        bufferIndex: 0,
        offset: 134_217_472,
    },
    { start: 67_108_800, end: 100_663_220, rowFirst: 3_355_440, rowLast: 5_033_160, bufferIndex: 1, offset: 0 },
    {
        start: 100_663_168,
        end: 134_217_600,
        rowFirst: 5_033_161,
        rowLast: 6_710_879,
        bufferIndex: 1,
        offset: 134_217_472,
    },
    { start: 134_217_600, end: 167_772_020, rowFirst: 6_710_880, rowLast: 8_388_600, bufferIndex: 2, offset: 0 },
    {
        start: 167_771_968,
        end: 200_000_000,
        rowFirst: 8_388_601,
        rowLast: 9_999_999,
        bufferIndex: 2,
        offset: 134_217_472,
    },
];
const BUFFERS_10M_SPLIT = [
    { byteOffset: 0, byteLength: 268_435_200 },
    { byteOffset: 268_435_200, byteLength: 268_435_200 },
    { byteOffset: 536_870_400, byteLength: 263_129_600 },
];

// ============================================================ arcsPerWindowFor

describe("arcsPerWindowFor", () => {
    it("is floor(maxStorageBufferBindingSize / 4) rounded down to a multiple of 64", () => {
        expect(arcsPerWindowFor(CAPS_SPEC_DEFAULT.limits)).toBe(33_554_432);
        expect(arcsPerWindowFor(CAPS_SWIFTSHADER.limits)).toBe(33_554_432);
        expect(arcsPerWindowFor(CAPS_LAVAPIPE.limits)).toBe(33_554_432);
        // 2,147,483,644 / 4 = 536,870,911 = 64 x 8,388,607 + 63
        expect(arcsPerWindowFor(CAPS_NVIDIA_4070.limits)).toBe(536_870_848);
        expect(arcsPerWindowFor(fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 4096 }).limits)).toBe(1024);
        // floor(1000 / 4) = 250 -> 250 - 250 % 64 = 192
        expect(arcsPerWindowFor(fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 1000 }).limits)).toBe(192);
        expect(arcsPerWindowFor(fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 255 }).limits)).toBe(0);
        expect(arcsPerWindowFor(fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 1024 * MIB }).limits)).toBe(
            268_435_456,
        );
    });
});

// ============================================================ planArcWindows

describe("planArcWindows", () => {
    it("returns no window for arcCount 0", () => {
        expect(planArcWindows(new Uint32Array([0]), 0, 64)).toEqual([]);
        expect(planArcWindows(new Uint32Array([0, 0, 0]), 0, 64)).toEqual([]);
    });

    it("covers a small graph with one window ending at the last row (empty rows included)", () => {
        expect(planArcWindows(new Uint32Array([0, 2, 5, 5, 9]), 9, 64)).toEqual([
            { start: 0, end: 9, rowFirst: 0, rowLast: 3, bufferIndex: 0, offset: 0 },
        ]);
        expect(planArcWindows(new Uint32Array([0, 3, 3, 3, 7]), 7, 64)).toEqual([
            { start: 0, end: 7, rowFirst: 0, rowLast: 3, bufferIndex: 0, offset: 0 },
        ]);
        expect(planArcWindows(new Uint32Array([0, 0, 0, 5]), 5, 64)).toEqual([
            { start: 0, end: 5, rowFirst: 0, rowLast: 2, bufferIndex: 0, offset: 0 },
        ]);
        expect(planArcWindows(new Uint32Array([0, 5, 5, 5]), 5, 64)).toEqual([
            { start: 0, end: 5, rowFirst: 0, rowLast: 2, bufferIndex: 0, offset: 0 },
        ]);
    });

    it("ends windows at row boundaries and aligns the next start down to 64 with `%`", () => {
        // rows of 20 arcs, 10 rows, 128 arcs per window: rows 0-5 end at 120 <= 128; the next start is 120 - 120 % 64 = 64,
        // limit 192, rows 6-8 end at 180; start 180 - 180 % 64 = 128, limit 256, row 9 ends at 200.
        expect(planArcWindows(uniformRowPtr(10, 20), 200, 128)).toEqual([
            { start: 0, end: 120, rowFirst: 0, rowLast: 5, bufferIndex: 0, offset: 0 },
            { start: 64, end: 180, rowFirst: 6, rowLast: 8, bufferIndex: 0, offset: 256 },
            { start: 128, end: 200, rowFirst: 9, rowLast: 9, bufferIndex: 0, offset: 512 },
        ]);
    });

    it("splits a hub row longer than a window into aligned chunks that share the row (PLAN DECISION 4)", () => {
        // row 1 holds 200 arcs [5, 205); a 64-arc window opened at 5 - 5 % 64 = 0 cannot hold it, so row 0 closes the
        // first window at 5 and row 1 is split into [0, 64), [64, 128), [128, 192), then [192, 210) also takes row 2.
        expect(planArcWindows(new Uint32Array([0, 5, 205, 210]), 210, 64)).toEqual([
            { start: 0, end: 5, rowFirst: 0, rowLast: 0, bufferIndex: 0, offset: 0 },
            { start: 0, end: 64, rowFirst: 1, rowLast: 1, bufferIndex: 0, offset: 0 },
            { start: 64, end: 128, rowFirst: 1, rowLast: 1, bufferIndex: 0, offset: 256 },
            { start: 128, end: 192, rowFirst: 1, rowLast: 1, bufferIndex: 0, offset: 512 },
            { start: 192, end: 210, rowFirst: 1, rowLast: 2, bufferIndex: 0, offset: 768 },
        ]);
        // a row exactly two windows long, followed by an empty row
        expect(planArcWindows(new Uint32Array([0, 128, 128]), 128, 64)).toEqual([
            { start: 0, end: 64, rowFirst: 0, rowLast: 0, bufferIndex: 0, offset: 0 },
            { start: 64, end: 128, rowFirst: 0, rowLast: 1, bufferIndex: 0, offset: 256 },
        ]);
    });

    it("computes starts above 2^31 arcs exactly (a bitwise `& ~63` would give -2,147,483,520 for 2,147,483,800)", () => {
        // three rows: [0, 2,147,483,700), [2,147,483,700, 2,147,483,800), [2,147,483,800, 4,294,967,000);
        // 536,870,848 arcs per window (the 4070's 2 GiB - 4 binding): row 0 = four full chunks + a tail that also takes row 1;
        // row 2 opens at 2,147,483,800 - 24 = 2,147,483,776 (2,147,483,800 = 64 x 33,554,434 + 24) and splits into four.
        const rowPtr = new Uint32Array([0, 2_147_483_700, 2_147_483_800, 4_294_967_000]);
        expect(planArcWindows(rowPtr, 4_294_967_000, 536_870_848)).toEqual([
            { start: 0, end: 536_870_848, rowFirst: 0, rowLast: 0, bufferIndex: 0, offset: 0 },
            { start: 536_870_848, end: 1_073_741_696, rowFirst: 0, rowLast: 0, bufferIndex: 0, offset: 2_147_483_392 },
            {
                start: 1_073_741_696,
                end: 1_610_612_544,
                rowFirst: 0,
                rowLast: 0,
                bufferIndex: 0,
                offset: 4_294_966_784,
            },
            {
                start: 1_610_612_544,
                end: 2_147_483_392,
                rowFirst: 0,
                rowLast: 0,
                bufferIndex: 0,
                offset: 6_442_450_176,
            },
            {
                start: 2_147_483_392,
                end: 2_147_483_800,
                rowFirst: 0,
                rowLast: 1,
                bufferIndex: 0,
                offset: 8_589_933_568,
            },
            {
                start: 2_147_483_776,
                end: 2_684_354_624,
                rowFirst: 2,
                rowLast: 2,
                bufferIndex: 0,
                offset: 8_589_935_104,
            },
            {
                start: 2_684_354_624,
                end: 3_221_225_472,
                rowFirst: 2,
                rowLast: 2,
                bufferIndex: 0,
                offset: 10_737_418_496,
            },
            {
                start: 3_221_225_472,
                end: 3_758_096_320,
                rowFirst: 2,
                rowLast: 2,
                bufferIndex: 0,
                offset: 12_884_901_888,
            },
            {
                start: 3_758_096_320,
                end: 4_294_967_000,
                rowFirst: 2,
                rowLast: 2,
                bufferIndex: 0,
                offset: 15_032_385_280,
            },
        ]);
        for (const w of planArcWindows(rowPtr, 4_294_967_000, 536_870_848)) {
            expect(w.start % ARC_WINDOW_ALIGN).toBe(0);
            expect(w.start).toBeGreaterThanOrEqual(0);
        }
    });

    it("rejects a bad arcsPerWindow, an empty rowPtr and an arcCount that disagrees with rowPtr", () => {
        expect(caught(() => planArcWindows(new Uint32Array([0, 5]), 5, 0)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => planArcWindows(new Uint32Array([0, 5]), 5, 100)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => planArcWindows(new Uint32Array([0, 5]), 5, 64.5)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => planArcWindows(new Uint32Array(0), 0, 64)).code).toBe("E_INVALID_ARGUMENT");
        expect(caught(() => planArcWindows(new Uint32Array([0, 5]), 6, 64)).code).toBe("E_INVALID_ARGUMENT");
    });

    it("partitions the arcs of every row exactly once, with aligned starts and bounded windows (fast-check)", () => {
        const arbRowPtr = fc.array(fc.integer({ min: 0, max: 300 }), { minLength: 0, maxLength: 40 }).map((lengths) => {
            const rowPtr = new Uint32Array(lengths.length + 1);
            let acc = 0;
            lengths.forEach((len, i) => {
                acc += len;
                rowPtr[i + 1] = acc;
            });
            return rowPtr;
        });
        fc.assert(
            fc.property(arbRowPtr, fc.constantFrom(64, 128, 192, 256, 512), (rowPtr, arcsPerWindow) => {
                const n = rowPtr.length - 1;
                const arcCount = rowPtr[n];
                const windows = planArcWindows(rowPtr, arcCount, arcsPerWindow);
                if (arcCount === 0) {
                    expect(windows).toEqual([]);
                    return;
                }
                const covered = new Uint8Array(arcCount);
                expect(windows[0].rowFirst).toBe(0);
                expect(windows[windows.length - 1].rowLast).toBe(n - 1);
                expect(windows[windows.length - 1].end).toBe(arcCount);
                for (let k = 0; k < windows.length; k++) {
                    const w = windows[k];
                    expect(w.start % 64).toBe(0);
                    expect(w.end).toBeGreaterThan(w.start);
                    expect(w.end - w.start).toBeLessThanOrEqual(arcsPerWindow);
                    expect(w.bufferIndex).toBe(0);
                    expect(w.offset).toBe(4 * w.start);
                    expect(w.rowLast).toBeGreaterThanOrEqual(w.rowFirst);
                    if (k > 0) {
                        const prev = windows[k - 1];
                        expect(w.start).toBeGreaterThanOrEqual(prev.start);
                        expect(w.end).toBeGreaterThan(prev.end);
                        expect([prev.rowLast, prev.rowLast + 1]).toContain(w.rowFirst);
                    }
                    for (let u = w.rowFirst; u <= w.rowLast; u++) {
                        const lo = Math.max(rowPtr[u], w.start);
                        const hi = Math.min(rowPtr[u + 1], w.end);
                        for (let a = lo; a < hi; a++) {
                            covered[a] += 1;
                        }
                    }
                }
                expect(covered.every((c) => c === 1)).toBe(true);
            }),
            { numRuns: 200 },
        );
    });
});

// ============================================================ planUpload on real small snapshots

describe("planUpload on real snapshots", () => {
    it("plans karate's arena hot prefix (880 B) by default: every segment inside the prefix is bound, the cold ones are null", () => {
        const s = snapshotOf(KARATE_EDGES);
        const arena = s.arena as ArenaLayout;
        expect(arena.hotByteLength).toBe(880);
        expect(arena.byteLength).toBe(2104);
        expect(planUpload(s, CAPS_SPEC_DEFAULT, DEFAULT_NEED)).toEqual({
            kind: "arena",
            bytes: 880,
            includesCold: false,
            segments: {
                rowPtr: { offset: 0, size: 140 },
                colIdx: { offset: 256, size: 624 },
                weights: null,
                arcToEdge: null,
                edgeToArc: null,
            },
        });
    });

    it("plans the full arena (2104 B) when a cold segment is needed and fits, listing every segment in the buffer", () => {
        const s = snapshotOf(KARATE_EDGES);
        expect(planUpload(s, CAPS_SPEC_DEFAULT, ["rowPtr", "colIdx", "arcToEdge"])).toEqual({
            kind: "arena",
            bytes: 2104,
            includesCold: true,
            segments: {
                rowPtr: { offset: 0, size: 140 },
                colIdx: { offset: 256, size: 624 },
                weights: null,
                arcToEdge: { offset: 1024, size: 624 },
                edgeToArc: { offset: 1792, size: 312 },
            },
        });
    });

    it("binds a weights segment when the snapshot is weighted (hot prefix 1648 B)", () => {
        const s = snapshotOf(KARATE_EDGES, { weighted: true });
        const plan = planUpload(s, CAPS_SPEC_DEFAULT, DEFAULT_NEED);
        expect(plan.kind).toBe("arena");
        if (plan.kind === "arena") {
            expect(plan.bytes).toBe(1648);
            expect(plan.segments.weights).toEqual({ offset: 1024, size: 624 });
            expect(plan.segments.arcToEdge).toBeNull();
        }
    });

    it("plans one buffer per needed present array on a fromCsr snapshot (arena === null)", () => {
        const s = csrSnapshotOf(KARATE_EDGES);
        expect(s.arena).toBeNull();
        expect(planUpload(s, CAPS_SPEC_DEFAULT, DEFAULT_NEED)).toEqual({
            kind: "perArray",
            arrays: [
                { name: "rowPtr", byteLength: 140, buffers: [{ byteOffset: 0, byteLength: 140 }] },
                { name: "colIdx", byteLength: 624, buffers: [{ byteOffset: 0, byteLength: 624 }] },
            ],
        });
        const weighted = csrSnapshotOf(KARATE_EDGES, { weighted: true });
        expect(planUpload(weighted, CAPS_SPEC_DEFAULT, ALL_CORE)).toEqual({
            kind: "perArray",
            arrays: [
                { name: "rowPtr", byteLength: 140, buffers: [{ byteOffset: 0, byteLength: 140 }] },
                { name: "colIdx", byteLength: 624, buffers: [{ byteOffset: 0, byteLength: 624 }] },
                { name: "weights", byteLength: 624, buffers: [{ byteOffset: 0, byteLength: 624 }] },
                { name: "arcToEdge", byteLength: 624, buffers: [{ byteOffset: 0, byteLength: 624 }] },
                { name: "edgeToArc", byteLength: 312, buffers: [{ byteOffset: 0, byteLength: 312 }] },
            ],
        });
    });

    it("plans rowPtr only for arcCount 0 whatever `need` says", () => {
        const s = snapshotOf([], { nodeCount: 5 });
        expect(planUpload(s, CAPS_SPEC_DEFAULT, ALL_CORE)).toEqual({
            kind: "arena",
            bytes: 24,
            includesCold: false,
            segments: {
                rowPtr: { offset: 0, size: 24 },
                colIdx: null,
                weights: null,
                arcToEdge: null,
                edgeToArc: null,
            },
        });
        // fromCsr keeps a trivial one-array arena for an arc-less graph, so the per-array case uses arena: false
        const separate = snapshotOf([], { nodeCount: 5, arena: false });
        expect(separate.arena).toBeNull();
        expect(planUpload(separate, CAPS_SPEC_DEFAULT, ALL_CORE)).toEqual({
            kind: "perArray",
            arrays: [{ name: "rowPtr", byteLength: 24, buffers: [{ byteOffset: 0, byteLength: 24 }] }],
        });
        const empty = snapshotOf([], { nodeCount: 0 });
        const plan = planUpload(empty, CAPS_SPEC_DEFAULT, DEFAULT_NEED);
        expect(plan.kind).toBe("arena");
        if (plan.kind === "arena") {
            expect(plan.segments.rowPtr).toEqual({ offset: 0, size: 4 });
        }
    });

    it("never plans an identity permutation (a directed path): the cold names are null and the plan is the hot prefix", () => {
        const s = snapshotOf(pathEdges(4), { directed: true });
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        expect(planUpload(s, CAPS_SPEC_DEFAULT, ALL_CORE)).toEqual({
            kind: "arena",
            bytes: 268,
            includesCold: false,
            segments: {
                rowPtr: { offset: 0, size: 20 },
                colIdx: { offset: 256, size: 12 },
                weights: null,
                arcToEdge: null,
                edgeToArc: null,
            },
        });
        expect(planUpload(csrSnapshotOf(pathEdges(4), { directed: true }), CAPS_SPEC_DEFAULT, ALL_CORE)).toEqual({
            kind: "perArray",
            arrays: [
                { name: "rowPtr", byteLength: 20, buffers: [{ byteOffset: 0, byteLength: 20 }] },
                { name: "colIdx", byteLength: 12, buffers: [{ byteOffset: 0, byteLength: 12 }] },
            ],
        });
    });

    it("always plans rowPtr even when `need` omits it, and is deterministic", () => {
        const s = csrSnapshotOf(KARATE_EDGES);
        const plan = planUpload(s, CAPS_SPEC_DEFAULT, ["colIdx"]);
        expect(plan.kind).toBe("perArray");
        if (plan.kind === "perArray") {
            expect(plan.arrays.map((a) => a.name)).toEqual(["rowPtr", "colIdx"]);
        }
        expect(planUpload(s, CAPS_SPEC_DEFAULT, ["colIdx"])).toEqual(plan);
        const arena = snapshotOf(KARATE_EDGES);
        expect(planUpload(arena, CAPS_SPEC_DEFAULT, ["colIdx"])).toEqual(
            planUpload(arena, CAPS_SPEC_DEFAULT, ["rowPtr", "colIdx"]),
        );
    });

    it("depends on the limits only: the subgroup facts and the software flag never change a plan", () => {
        const s = snapshotOf(KARATE_EDGES, { weighted: true });
        for (const { caps } of CAPS_TABLES) {
            const twisted = fakeCaps(caps, {}, { subgroupMinSize: 8, subgroupMaxSize: 32, software: !caps.software });
            expect(planUpload(s, twisted, ALL_CORE)).toEqual(planUpload(s, caps, ALL_CORE));
        }
    });
});

// ============================================================ the spec 4.2 tiers x the caps tables

describe("planUpload at the spec 4.2 tiers", () => {
    it("the fake arenas reproduce the spec 4.2 / graph-format numbers", () => {
        const a100k = fakeSnapshot(TIER_100K).arena as ArenaLayout;
        expect(a100k.hotByteLength).toBe(16_400_128);
        expect(a100k.byteLength).toBe(28_400_128);
        expect(a100k.segments).toEqual({
            rowPtr: { byteOffset: 0, byteLength: 400_004 },
            colIdx: { byteOffset: 400_128, byteLength: 8_000_000 },
            weights: { byteOffset: 8_400_128, byteLength: 8_000_000 },
            arcToEdge: { byteOffset: 16_400_128, byteLength: 8_000_000 },
            edgeToArc: { byteOffset: 24_400_128, byteLength: 4_000_000 },
        });
        const a1m = fakeSnapshot(TIER_1M).arena as ArenaLayout;
        expect(a1m.hotByteLength).toBe(164_000_256);
        expect(a1m.byteLength).toBe(284_000_256);
        expect(a1m.segments.colIdx).toEqual({ byteOffset: 4_000_256, byteLength: 80_000_000 });
        expect(a1m.segments.edgeToArc).toEqual({ byteOffset: 244_000_256, byteLength: 40_000_000 });
        const a10m = fakeSnapshot(TIER_10M).arena as ArenaLayout;
        expect(a10m.hotByteLength).toBe(1_640_000_256);
        expect(a10m.byteLength).toBe(2_840_000_256);
        expect(a10m.segments.colIdx).toEqual({ byteOffset: 40_000_256, byteLength: 800_000_000 });
        expect(a10m.segments.weights).toEqual({ byteOffset: 840_000_256, byteLength: 800_000_000 });
    });

    describe.each(CAPS_TABLES)("$name", ({ caps }) => {
        it("100k / 1M: the hot prefix (16,400,128 B) by default and the full arena (28,400,128 B) with a cold need", () => {
            const s = fakeSnapshot(TIER_100K);
            const hot = planUpload(s, caps, DEFAULT_NEED);
            expect(hot).toEqual({
                kind: "arena",
                bytes: 16_400_128,
                includesCold: false,
                segments: {
                    rowPtr: { offset: 0, size: 400_004 },
                    colIdx: { offset: 400_128, size: 8_000_000 },
                    weights: { offset: 8_400_128, size: 8_000_000 },
                    arcToEdge: null,
                    edgeToArc: null,
                },
            });
            const full = planUpload(s, caps, ["rowPtr", "colIdx", "weights", "edgeToArc"]);
            expect(full).toEqual({
                kind: "arena",
                bytes: 28_400_128,
                includesCold: true,
                segments: {
                    rowPtr: { offset: 0, size: 400_004 },
                    colIdx: { offset: 400_128, size: 8_000_000 },
                    weights: { offset: 8_400_128, size: 8_000_000 },
                    arcToEdge: { offset: 16_400_128, size: 8_000_000 },
                    edgeToArc: { offset: 24_400_128, size: 4_000_000 },
                },
            });
        });

        it("1M / 10M: the hot prefix (164,000,256 B) fits every table; the full arena (284,000,256 B) only the 4070", () => {
            const s = fakeSnapshot({ ...TIER_1M, rowPtr: rowPtr1M });
            expect(planUpload(s, caps, DEFAULT_NEED)).toEqual({
                kind: "arena",
                bytes: 164_000_256,
                includesCold: false,
                segments: {
                    rowPtr: { offset: 0, size: 4_000_004 },
                    colIdx: { offset: 4_000_256, size: 80_000_000 },
                    weights: { offset: 84_000_256, size: 80_000_000 },
                    arcToEdge: null,
                    edgeToArc: null,
                },
            });
            const cold = planUpload(s, caps, ["rowPtr", "colIdx", "weights", "arcToEdge"]);
            if (caps === CAPS_NVIDIA_4070) {
                expect(cold).toEqual({
                    kind: "arena",
                    bytes: 284_000_256,
                    includesCold: true,
                    segments: {
                        rowPtr: { offset: 0, size: 4_000_004 },
                        colIdx: { offset: 4_000_256, size: 80_000_000 },
                        weights: { offset: 84_000_256, size: 80_000_000 },
                        arcToEdge: { offset: 164_000_256, size: 80_000_000 },
                        edgeToArc: { offset: 244_000_256, size: 40_000_000 },
                    },
                });
            } else {
                // 284,000,256 > 268,435,456: the hot prefix is uploaded and the residency uploads arcToEdge per array
                expect(cold).toEqual(planUpload(s, caps, DEFAULT_NEED));
            }
        });

        it("1M / 10M without an arena: one buffer per array on every table", () => {
            const s = fakeSnapshot({ ...TIER_1M, arena: false, rowPtr: rowPtr1M });
            expect(planUpload(s, caps, DEFAULT_NEED)).toEqual({
                kind: "perArray",
                arrays: [
                    { name: "rowPtr", byteLength: 4_000_004, buffers: [{ byteOffset: 0, byteLength: 4_000_004 }] },
                    { name: "colIdx", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
                    { name: "weights", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
                ],
            });
        });

        it("10M / 100M: windowed at the 128 MiB binding (six windows, three buffers), arena / perArray on the 4070", () => {
            const withArena = fakeSnapshot({ ...TIER_10M, rowPtr: rowPtr10M });
            const noArena = fakeSnapshot({ ...TIER_10M, arena: false, rowPtr: rowPtr10M });
            if (caps === CAPS_NVIDIA_4070) {
                expect(planUpload(withArena, caps, DEFAULT_NEED)).toEqual({
                    kind: "arena",
                    bytes: 1_640_000_256,
                    includesCold: false,
                    segments: {
                        rowPtr: { offset: 0, size: 40_000_004 },
                        colIdx: { offset: 40_000_256, size: 800_000_000 },
                        weights: { offset: 840_000_256, size: 800_000_000 },
                        arcToEdge: null,
                        edgeToArc: null,
                    },
                });
                const full = planUpload(withArena, caps, ALL_CORE);
                expect(full.kind).toBe("arena");
                if (full.kind === "arena") {
                    expect(full.bytes).toBe(2_840_000_256);
                    expect(full.includesCold).toBe(true);
                    expect(full.segments.edgeToArc).toEqual({ offset: 2_440_000_256, size: 400_000_000 });
                }
                // the format freezes with arena: false at this tier (spec 4.2): 800 MB < 2 GiB - 4 -> per array
                expect(planUpload(noArena, caps, DEFAULT_NEED)).toEqual({
                    kind: "perArray",
                    arrays: [
                        {
                            name: "rowPtr",
                            byteLength: 40_000_004,
                            buffers: [{ byteOffset: 0, byteLength: 40_000_004 }],
                        },
                        {
                            name: "colIdx",
                            byteLength: 800_000_000,
                            buffers: [{ byteOffset: 0, byteLength: 800_000_000 }],
                        },
                        {
                            name: "weights",
                            byteLength: 800_000_000,
                            buffers: [{ byteOffset: 0, byteLength: 800_000_000 }],
                        },
                    ],
                });
            } else {
                // colIdx = 800,000,000 B exceeds the 128 MiB binding and the 256 MiB buffer limit: SIX windows of at most
                // 33,554,432 arcs over rows of 20 arcs (PLAN DECISION 3: the spec's "25" mixes bytes and arcs), split
                // across three buffers at window boundaries (2 x 268,435,200 B + 263,129,600 B = 800,000,000 B)
                const expected: UploadPlan = {
                    kind: "windowed",
                    arcsPerWindow: 33_554_432,
                    windows: WINDOWS_10M_SPLIT,
                    arrays: [
                        {
                            name: "rowPtr",
                            byteLength: 40_000_004,
                            buffers: [{ byteOffset: 0, byteLength: 40_000_004 }],
                        },
                        { name: "colIdx", byteLength: 800_000_000, buffers: BUFFERS_10M_SPLIT },
                        { name: "weights", byteLength: 800_000_000, buffers: BUFFERS_10M_SPLIT },
                    ],
                };
                expect(planUpload(withArena, caps, DEFAULT_NEED)).toEqual(expected);
                expect(planUpload(noArena, caps, DEFAULT_NEED)).toEqual(expected);
                expect(BUFFERS_10M_SPLIT.reduce((sum, b) => sum + b.byteLength, 0)).toBe(800_000_000);
            }
        });
    });

    it("exceeds buffer, fits binding: the 1M / 10M arena at a faked 100,000,000 B maxBufferSize goes per array", () => {
        const caps = fakeCaps(CAPS_SPEC_DEFAULT, { maxBufferSize: 100_000_000 });
        const s = fakeSnapshot({ ...TIER_1M, rowPtr: rowPtr1M });
        expect(planUpload(s, caps, DEFAULT_NEED)).toEqual({
            kind: "perArray",
            arrays: [
                { name: "rowPtr", byteLength: 4_000_004, buffers: [{ byteOffset: 0, byteLength: 4_000_004 }] },
                { name: "colIdx", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
                { name: "weights", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
            ],
        });
    });

    it("exceeds binding, fits buffer: the 1M / 10M colIdx at a faked 64 MiB binding is windowed into one buffer", () => {
        const caps = fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 64 * MIB });
        const s = fakeSnapshot({ ...TIER_1M, rowPtr: rowPtr1M });
        // 16,777,216 arcs per window; rows of 20: window 0 ends at 16,777,200 (row 838,859), the next start is
        // 16,777,200 - 48 = 16,777,152 and its limit 33,554,368 covers the remaining 3,222,800 arcs
        expect(planUpload(s, caps, DEFAULT_NEED)).toEqual({
            kind: "windowed",
            arcsPerWindow: 16_777_216,
            windows: [
                { start: 0, end: 16_777_200, rowFirst: 0, rowLast: 838_859, bufferIndex: 0, offset: 0 },
                {
                    start: 16_777_152,
                    end: 20_000_000,
                    rowFirst: 838_860,
                    rowLast: 999_999,
                    bufferIndex: 0,
                    offset: 67_108_608,
                },
            ],
            arrays: [
                { name: "rowPtr", byteLength: 4_000_004, buffers: [{ byteOffset: 0, byteLength: 4_000_004 }] },
                { name: "colIdx", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
                { name: "weights", byteLength: 80_000_000, buffers: [{ byteOffset: 0, byteLength: 80_000_000 }] },
            ],
        });
    });

    it("the 10M / 100M windows placed in one buffer keep offset = 4 x start", () => {
        const caps = fakeCaps(CAPS_SPEC_DEFAULT, { maxBufferSize: 1024 * MIB });
        const s = fakeSnapshot({ ...TIER_10M, arena: false, rowPtr: rowPtr10M });
        const plan = planUpload(s, caps, ["rowPtr", "colIdx"]);
        expect(plan.kind).toBe("windowed");
        if (plan.kind === "windowed") {
            expect(plan.windows).toEqual(WINDOWS_10M_SINGLE_BUFFER);
            expect(plan.arrays[1]).toEqual({
                name: "colIdx",
                byteLength: 800_000_000,
                buffers: [{ byteOffset: 0, byteLength: 800_000_000 }],
            });
        }
    });

    it("exceeds binding on the 4070: a 2.4 GB colIdx over 100 hub rows is windowed at 536,870,848 arcs into one buffer", () => {
        // colIdx = 2,400,000,000 B > 2,147,483,644 (the 2 GiB - 4 binding) but < 1 TiB, so one buffer; 536,870,848 /
        // 6,000,000 = 89.47 -> 89 whole rows = 534,000,000 arcs, which is 64-aligned, so window 1 starts there and its
        // limit 1,070,870,848 covers the remaining 11 rows. The planner never reads colIdx: the fake costs its rowPtr.
        const hub = uniformRowPtr(100, 6_000_000);
        const s = fakeSnapshot({
            n: 100,
            edges: 600_000_000,
            directed: true,
            weighted: false,
            identity: true,
            arena: false,
            rowPtr: hub,
        });
        expect(planUpload(s, CAPS_NVIDIA_4070, DEFAULT_NEED)).toEqual({
            kind: "windowed",
            arcsPerWindow: 536_870_848,
            windows: [
                { start: 0, end: 534_000_000, rowFirst: 0, rowLast: 88, bufferIndex: 0, offset: 0 },
                {
                    start: 534_000_000,
                    end: 600_000_000,
                    rowFirst: 89,
                    rowLast: 99,
                    bufferIndex: 0,
                    offset: 2_136_000_000,
                },
            ],
            arrays: [
                { name: "rowPtr", byteLength: 404, buffers: [{ byteOffset: 0, byteLength: 404 }] },
                {
                    name: "colIdx",
                    byteLength: 2_400_000_000,
                    buffers: [{ byteOffset: 0, byteLength: 2_400_000_000 }],
                },
            ],
        });
    });

    it("exceeds buffer on the 4070: the full 10M / 100M arena above a faked 2,000,000,000 B maxBufferSize falls back to the hot prefix", () => {
        // the full arena (2,840,000,256 B) exceeds 2e9 but the hot prefix (1,640,000,256 B) fits, so the plan is the
        // hot arena with the cold segments null: identical to the default-need plan on the real table
        const caps = fakeCaps(CAPS_NVIDIA_4070, { maxBufferSize: 2_000_000_000 });
        const s = fakeSnapshot({ ...TIER_10M, rowPtr: rowPtr10M });
        const plan = planUpload(s, caps, ALL_CORE);
        expect(plan).toEqual(planUpload(s, CAPS_NVIDIA_4070, DEFAULT_NEED));
        expect(plan.kind).toBe("arena");
        if (plan.kind === "arena") {
            expect(plan.bytes).toBe(1_640_000_256);
            expect(plan.includesCold).toBe(false);
            expect(plan.segments.arcToEdge).toBeNull();
            expect(plan.segments.edgeToArc).toBeNull();
        }
    });

    it("throws E_TOO_LARGE { path: 'rowPtr' } when even rowPtr exceeds the binding limit", () => {
        const caps = fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 1024 });
        const s = fakeSnapshot({ n: 1000, edges: 0, weighted: false });
        const err = caught(() => planUpload(s, caps, DEFAULT_NEED));
        expect(err.code).toBe("E_TOO_LARGE");
        expect(err.details).toEqual({ needed: 4004, limit: 1024, path: "rowPtr", algorithm: null });
    });

    it("throws E_TOO_LARGE { path: 'binding' } when a needed edge-indexed array cannot be windowed", () => {
        const caps = fakeCaps(CAPS_SPEC_DEFAULT, { maxStorageBufferBindingSize: 8_000_000 });
        const s = fakeSnapshot({ ...TIER_1M, rowPtr: rowPtr1M });
        const err = caught(() => planUpload(s, caps, ["rowPtr", "edgeToArc"]));
        expect(err.code).toBe("E_TOO_LARGE");
        expect(err.details).toEqual({ needed: 40_000_000, limit: 8_000_000, path: "binding", algorithm: null });
    });
});

// ============================================================ the pure helpers of this task

describe("test/helpers/caps-tables.ts", () => {
    it("ships five distinct tables and fakeCaps overrides limits and flags without touching the base", () => {
        expect(CAPS_TABLES.map((t) => t.name)).toEqual([
            "spec-default",
            "swiftshader",
            "lavapipe",
            "nvidia-4070",
            "intel-xe",
        ]);
        expect(CAPS_INTEL_XE.subgroupMinSize).toBe(8);
        expect(CAPS_INTEL_XE.subgroupMaxSize).toBe(32);
        expect(CAPS_INTEL_XE.limits).toBe(CAPS_SPEC_DEFAULT.limits);
        expect(CAPS_SPEC_DEFAULT.limits.maxBufferSize).toBe(268_435_456);
        expect(CAPS_SPEC_DEFAULT.limits.maxStorageBufferBindingSize).toBe(134_217_728);
        expect(CAPS_SWIFTSHADER.software).toBe(true);
        expect(CAPS_LAVAPIPE.subgroupMinSize).toBe(8);
        expect(CAPS_NVIDIA_4070.limits.maxStorageBufferBindingSize).toBe(2_147_483_644);
        const faked = fakeCaps(
            CAPS_LAVAPIPE,
            { maxBufferSize: 4_294_967_295, maxStorageBuffersPerShaderStage: undefined },
            { software: false },
        );
        expect(faked.limits.maxBufferSize).toBe(4_294_967_295);
        expect(faked.limits.maxStorageBuffersPerShaderStage).toBe(16);
        expect(faked.software).toBe(false);
        expect(faked.subgroupMaxSize).toBe(8);
        expect(faked.features).toBe(CAPS_LAVAPIPE.features);
        expect(CAPS_LAVAPIPE.limits.maxBufferSize).toBe(268_435_456);
        expect(CAPS_LAVAPIPE.software).toBe(true);
    });
});

describe("test/helpers/matchers.ts", () => {
    it("expectAllClose: |a - e| <= abs + rel |e|, NaN pairs equal, the worst index reported", () => {
        expectAllClose([1, 2.05], [1, 2], { rel: 0.03, abs: 0 });
        expect(() => expectAllClose([1, 2.05], [1, 2], { rel: 0.02, abs: 0 })).toThrow(/worst index 1/);
        expectAllClose([1, 2.05], [1, 2], { rel: 0, abs: 0.05 });
        expectAllClose([Number.NaN], [Number.NaN], { rel: 0, abs: 0 });
        expect(() => expectAllClose([Number.NaN], [1], { rel: 1, abs: 1 })).toThrow(/worst index 0/);
        expect(() => expectAllClose([1, 2, 3], [1, 2], { rel: 1, abs: 1 })).toThrow();
    });

    it("expectBitwiseEqual: bytes, dtype and length; -0 differs from 0 and NaN payloads count", () => {
        expectBitwiseEqual(new Uint32Array([1, 2, 3]), new Uint32Array([1, 2, 3]));
        expect(() => expectBitwiseEqual(new Float32Array([0]), new Float32Array([-0]))).toThrow(/element 0/);
        expect(() => expectBitwiseEqual(new Uint32Array([1]), new Float32Array([1]))).toThrow(/dtype/);
        expect(() => expectBitwiseEqual(new Uint32Array([1, 2]), new Uint32Array([1]))).toThrow(/length/);
        const quiet = new Float32Array(new Uint32Array([0x7fc00000]).buffer);
        const payload = new Float32Array(new Uint32Array([0x7fc00001]).buffer);
        expect(() => expectBitwiseEqual(quiet, payload)).toThrow(/element 0/);
        expectBitwiseEqual(quiet, new Float32Array(new Uint32Array([0x7fc00000]).buffer));
    });

    it("flooredRelError: hand-computed floored per-node errors over stride-3 vectors", () => {
        // node 0 = (1, 2, 2) |F| = 3; node 1 = (0.001, 0, 0) |F| = 0.001; floor = 0.01 x 3 = 0.03;
        // errors: 0 / 3 = 0 and 0.001 / max(0.001, 0.03) = 0.033333; rms = sqrt((0 + 0.0011111) / 2) = 0.023570; p99 = errors[1]
        const r = flooredRelError([1, 2, 2, 0, 0, 0], [1, 2, 2, 0.001, 0, 0], 0.01);
        expect(r.max).toBeCloseTo(0.0333333, 6);
        expect(r.argmax).toBe(1);
        expect(r.rms).toBeCloseTo(0.0235702, 6);
        expect(r.p99).toBeCloseTo(0.0333333, 6);
        expect(flooredRelError([1, 2, 2], [1, 2, 2], 0.01)).toEqual({ max: 0, argmax: 0, rms: 0, p99: 0 });
        expect(flooredRelError([], [], 0.01)).toEqual({ max: 0, argmax: -1, rms: 0, p99: 0 });
        // an error against an all-zero oracle is infinite unless the numerator is zero too
        expect(flooredRelError([0, 0, 1], [0, 0, 0], 0.01).max).toBe(Number.POSITIVE_INFINITY);
        expect(flooredRelError([0, 0, 0], [0, 0, 0], 0.01).max).toBe(0);
    });

    it("maxRelError: max |a - e| / max(|e|, floor)", () => {
        // 0.1 / 1 = 0.1 and 0.001 / max(0, 0.01) = 0.1
        expect(maxRelError([1.1, 0.001], [1, 0], 0.01)).toBeCloseTo(0.1, 12);
        expect(maxRelError([], [], 0.01)).toBe(0);
        expect(maxRelError([2], [1], 0.01)).toBe(1);
    });
});

describe("test/helpers/graphs.ts", () => {
    it("generators: karate, grid, path, star, cycle, complete", () => {
        expect(KARATE_EDGES).toHaveLength(78);
        expect(Math.max(...KARATE_EDGES.map((e) => Math.max(e[0], e[1])))).toBe(33);
        expect(gridEdges(3, 2)).toEqual([
            [0, 1],
            [0, 3],
            [1, 2],
            [1, 4],
            [2, 5],
            [3, 4],
            [4, 5],
        ]);
        expect(pathEdges(4)).toEqual([
            [0, 1],
            [1, 2],
            [2, 3],
        ]);
        expect(pathEdges(1)).toEqual([]);
        expect(starEdges(3)).toEqual([
            [0, 1],
            [0, 2],
            [0, 3],
        ]);
        expect(cycleEdges(4)).toEqual([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ]);
        expect(cycleEdges(2)).toEqual([[0, 1]]);
        expect(cycleEdges(1)).toEqual([]);
        expect(completeEdges(4)).toHaveLength(6);
        expect(completeEdges(1)).toEqual([]);
    });

    it("random generators are seeded and shaped as documented", () => {
        const edges = randomEdges(50, 300, 7);
        expect(edges).toHaveLength(300);
        expect(randomEdges(50, 300, 7)).toEqual(edges);
        expect(randomEdges(50, 300, 8)).not.toEqual(edges);
        const pairs = new Set<string>();
        for (const [u, v] of edges) {
            expect(u).not.toBe(v);
            expect(u).toBeLessThan(50);
            expect(v).toBeLessThan(50);
            const key = `${Math.min(u, v)}-${Math.max(u, v)}`;
            expect(pairs.has(key)).toBe(false);
            pairs.add(key);
        }
        expect(() => randomEdges(4, 7, 1)).toThrow(RangeError);
        const loose = randomEdgesLoose(10, 100, 3);
        expect(loose).toHaveLength(100);
        for (const e of loose) {
            expect(e).toHaveLength(3);
            expect(Number.isInteger(e[2])).toBe(true);
            expect(e[2]).toBeGreaterThanOrEqual(1);
            expect(e[2]).toBeLessThanOrEqual(10);
        }
        expect(randomEdgesLoose(10, 100, 3)).toEqual(loose);
        const rmat = rmatEdges(4, 2, 5);
        expect(rmat).toHaveLength(32);
        for (const [u, v] of rmat) {
            expect(u).toBeLessThan(16);
            expect(v).toBeLessThan(16);
        }
        expect(rmatEdges(4, 2, 5)).toEqual(rmat);
    });

    it("snapshotOf / csrSnapshotOf build the same graph on the arena and per-array paths", () => {
        const a = snapshotOf(KARATE_EDGES, { label: "k" });
        const c = csrSnapshotOf(KARATE_EDGES);
        expect(a.arena).not.toBeNull();
        expect(a.label).toBe("k");
        expect(c.arena).toBeNull();
        expect(Array.from(c.rowPtr)).toEqual(Array.from(a.rowPtr));
        expect(Array.from(c.colIdx)).toEqual(Array.from(a.colIdx));
        expect(a.nodeCount).toBe(34);
        expect(a.edgeCount).toBe(78);
        expect(a.arcCount).toBe(156);
        expect(a.weights).toBeNull();
        expect(snapshotOf(KARATE_EDGES, { weighted: true }).flags.weighted).toBe(true);
        expect(snapshotOf(KARATE_EDGES, { arena: false }).arena).toBeNull();
        const d = csrSnapshotOf(pathEdges(4), { directed: true });
        expect(d.directed).toBe(true);
        expect(d.flags.arcToEdgeIsIdentity).toBe(true);
        expect(d.arena).toBeNull();
        // a lone rowPtr is a trivial arena for fromCsr: documented, not an error
        expect(csrSnapshotOf([], { nodeCount: 5 }).arena).not.toBeNull();
    });

    it("every named fixture builds at scale 1 / 50 with the documented shape", () => {
        const scale = 1 / 50;
        const built = new Map(FIXTURE_NAMES.map((name) => [name, fixture(name, scale)]));
        expect(built.size).toBe(13);
        for (const [name, f] of built) {
            expect(f.name).toBe(name);
            expect(f.snapshot.label).toBe(name);
        }
        const counts = (name: string): [number, number] => {
            const { snapshot } = built.get(name) as { snapshot: GraphSnapshot };
            return [snapshot.nodeCount, snapshot.edgeCount];
        };
        expect(counts("empty")).toEqual([0, 0]);
        expect(counts("one")).toEqual([1, 0]);
        expect(counts("self-loop")).toEqual([3, 3]);
        expect((built.get("self-loop") as { snapshot: GraphSnapshot }).snapshot.arcCount).toBe(5);
        expect(counts("karate")).toEqual([34, 78]);
        expect(counts("grid10")).toEqual([100, 180]);
        expect(counts("path1k")).toEqual([20, 19]);
        expect(counts("star200")).toEqual([5, 4]);
        expect(counts("complete6")).toEqual([6, 15]);
        expect(counts("random1k")).toEqual([20, 100]);
        expect(counts("hub10k")).toEqual([200, 599]);
        expect(counts("coincident")).toEqual([34, 78]);
        // giant = max(64, round(1000 / 50)) = 64 nodes (63 path + 192 random edges), 2 triangles, 1 isolated node
        expect(counts("isolated")).toEqual([71, 261]);
        expect(counts("parallel")).toEqual([4, 5]);
        const coincident = built.get("coincident") as { positions: Float32Array | null };
        expect(coincident.positions).not.toBeNull();
        const p = coincident.positions as Float32Array;
        expect(p).toHaveLength(102);
        expect([p[3], p[4], p[5]]).toEqual([p[0], p[1], p[2]]);
        expect([p[9], p[10], p[11]]).toEqual([p[6], p[7], p[8]]);
        expect(built.get("karate")?.positions).toBeNull();
        const parallel = (built.get("parallel") as { snapshot: GraphSnapshot }).snapshot;
        expect(parallel.flags.weighted).toBe(true);
        expect(parallel.flags.multigraph).toBe(true);
        expect(fixture("hub10k").snapshot.nodeCount).toBe(10_000);
        expect(fixture("isolated").snapshot.nodeCount).toBe(1310);
        expect(() => fixture("nope")).toThrow(RangeError);
    });
});
