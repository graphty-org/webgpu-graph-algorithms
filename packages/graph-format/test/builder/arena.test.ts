/**
 * Unit tests of the arena allocation (design section 10.3, invariant I10): the hot-to-cold layout at
 * 256-byte offsets with the worked example of the design, null segments for absent arrays, separate
 * allocation, and copying a core into an arena.
 */

import { describe, expect, it } from "vitest";

import {
    allocateArenaCore,
    allocateCoreArrays,
    allocateSeparateCore,
    copyCoreIntoArena,
    shapeOfCore,
} from "../../src/builder/arena.js";

describe("arena allocation", () => {
    it("reproduces the worked layout of design section 10.3", () => {
        const core = allocateArenaCore({
            nodeCount: 100_000,
            arcCount: 2_000_000,
            edgeCount: 1_000_000,
            weighted: true,
            identity: false,
        });
        const { arena } = core;
        expect(arena).not.toBeNull();
        if (arena === null) {
            return;
        }
        expect(arena.byteOffset).toBe(0);
        expect(arena.alignment).toBe(256);
        expect(arena.segments.rowPtr).toEqual({ byteOffset: 0, byteLength: 400_004 });
        expect(arena.segments.colIdx).toEqual({ byteOffset: 400_128, byteLength: 8_000_000 });
        expect(arena.segments.weights).toEqual({ byteOffset: 8_400_128, byteLength: 8_000_000 });
        expect(arena.segments.arcToEdge).toEqual({ byteOffset: 16_400_128, byteLength: 8_000_000 });
        expect(arena.segments.edgeToArc).toEqual({ byteOffset: 24_400_128, byteLength: 4_000_000 });
        expect(arena.byteLength).toBe(28_400_128);
        expect(arena.hotByteLength).toBe(16_400_128);
        expect(arena.buffer.byteLength).toBe(28_400_128);
        expect(core.rowPtr.length).toBe(100_001);
        expect(core.colIdx.byteOffset).toBe(400_128);
        expect(core.weights?.byteOffset).toBe(8_400_128);
        expect(core.arcToEdge?.byteOffset).toBe(16_400_128);
        expect(core.edgeToArc?.byteOffset).toBe(24_400_128);
        expect(core.edgeToArc?.length).toBe(1_000_000);
        expect(Object.isFrozen(arena)).toBe(true);
        expect(Object.isFrozen(arena.segments)).toBe(true);
    });

    it("gives the directed identity case only the hot prefix", () => {
        const core = allocateArenaCore({
            nodeCount: 100_000,
            arcCount: 1_000_000,
            edgeCount: 1_000_000,
            weighted: true,
            identity: true,
        });
        const { arena } = core;
        expect(arena?.byteLength).toBe(8_400_128);
        expect(arena?.hotByteLength).toBe(8_400_128);
        expect(arena?.segments.arcToEdge).toBeNull();
        expect(arena?.segments.edgeToArc).toBeNull();
        expect(core.arcToEdge).toBeNull();
        expect(core.edgeToArc).toBeNull();
        const unweighted = allocateArenaCore({
            nodeCount: 10,
            arcCount: 5,
            edgeCount: 5,
            weighted: false,
            identity: true,
        });
        expect(unweighted.weights).toBeNull();
        expect(unweighted.arena?.segments.weights).toBeNull();
        expect(unweighted.arena?.hotByteLength).toBe(256 + 20);
        expect(unweighted.arena?.byteLength).toBe(276);
    });

    it("leaves zero-length arrays out of the arena", () => {
        const empty = allocateArenaCore({ nodeCount: 0, arcCount: 0, edgeCount: 0, weighted: true, identity: false });
        expect(empty.arena?.byteLength).toBe(4);
        expect(empty.arena?.hotByteLength).toBe(4);
        expect(empty.arena?.segments).toEqual({
            rowPtr: { byteOffset: 0, byteLength: 4 },
            colIdx: null,
            weights: null,
            arcToEdge: null,
            edgeToArc: null,
        });
        expect(empty.colIdx.length).toBe(0);
        expect(empty.weights?.length).toBe(0);
        expect(empty.arcToEdge?.length).toBe(0);
        expect(empty.edgeToArc?.length).toBe(0);
        expect(empty.rowPtr.buffer).toBe(empty.arena?.buffer);
    });

    it("allocates separate 4-byte-aligned buffers when asked", () => {
        const core = allocateSeparateCore({ nodeCount: 3, arcCount: 5, edgeCount: 4, weighted: true, identity: false });
        expect(core.arena).toBeNull();
        expect(core.rowPtr.length).toBe(4);
        expect(core.colIdx.length).toBe(5);
        expect(core.weights?.length).toBe(5);
        expect(core.arcToEdge?.length).toBe(5);
        expect(core.edgeToArc?.length).toBe(4);
        expect(core.rowPtr.buffer).not.toBe(core.colIdx.buffer);
        const identity = allocateSeparateCore({
            nodeCount: 3,
            arcCount: 5,
            edgeCount: 5,
            weighted: false,
            identity: true,
        });
        expect(identity.weights).toBeNull();
        expect(identity.arcToEdge).toBeNull();
        expect(
            allocateCoreArrays({ nodeCount: 1, arcCount: 1, edgeCount: 1, weighted: false, identity: false }, false)
                .arena,
        ).toBeNull();
        expect(
            allocateCoreArrays({ nodeCount: 1, arcCount: 1, edgeCount: 1, weighted: false, identity: false }, true)
                .arena,
        ).not.toBeNull();
    });

    it("copies a separate core into a fresh arena with the same contents", () => {
        const core = allocateSeparateCore({ nodeCount: 2, arcCount: 3, edgeCount: 2, weighted: true, identity: false });
        core.rowPtr.set([0, 2, 3]);
        core.colIdx.set([0, 1, 1]);
        core.weights?.set([1, 2, 3]);
        core.arcToEdge?.set([1, 0, 0]);
        core.edgeToArc?.set([1, 0]);
        expect(shapeOfCore(core)).toEqual({ nodeCount: 2, arcCount: 3, edgeCount: 2, weighted: true, identity: false });
        const copied = copyCoreIntoArena(core);
        expect(copied.arena).not.toBeNull();
        expect(Array.from(copied.rowPtr)).toEqual([0, 2, 3]);
        expect(Array.from(copied.colIdx)).toEqual([0, 1, 1]);
        expect(Array.from(copied.weights as Float32Array)).toEqual([1, 2, 3]);
        expect(Array.from(copied.arcToEdge as Uint32Array)).toEqual([1, 0, 0]);
        expect(Array.from(copied.edgeToArc as Uint32Array)).toEqual([1, 0]);
        expect(copied.colIdx.buffer).toBe(copied.arena?.buffer);
        core.colIdx[0] = 9;
        expect(copied.colIdx[0]).toBe(0);
        const identity = allocateSeparateCore({
            nodeCount: 1,
            arcCount: 2,
            edgeCount: 2,
            weighted: false,
            identity: true,
        });
        expect(shapeOfCore(identity)).toEqual({
            nodeCount: 1,
            arcCount: 2,
            edgeCount: 2,
            weighted: false,
            identity: true,
        });
        const copiedIdentity = copyCoreIntoArena(identity);
        expect(copiedIdentity.arcToEdge).toBeNull();
        expect(copiedIdentity.weights).toBeNull();
    });
});
