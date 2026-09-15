/**
 * Core-array allocation for the freeze pipeline (design sections 6.3 step 11 and 10.3): the five core
 * arrays of a snapshot laid out hot to cold (rowPtr, colIdx, weights, arcToEdge, edgeToArc) inside ONE
 * ArrayBuffer at 256-byte-aligned offsets, or -- for the transient pass of a non-"keep" duplicate
 * policy and for `freeze({ arena: false })` -- as separate fresh buffers. Either way every array is
 * 4-byte aligned over a plain ArrayBuffer (invariant I10). A zero-length array, an absent weights
 * array and an identity permutation have a null segment and occupy nothing in the arena.
 *
 * The counting-sort scatter of counting-sort.ts writes straight into the arrays returned here, so a
 * "keep"-policy freeze touches the arena exactly once (design section 6.3 step 6).
 */

import { ALIGNMENT } from "../constants.js";
import { type ArenaLayout, type ArenaSegment, type CoreArrayName, type F32, type U32 } from "../types/index.js";
import { layoutSegments } from "../util/typed-array.js";

/** The shape of a core to allocate: the counts and which optional arrays exist. */
interface CoreShape {
    /** The node count n. */
    readonly nodeCount: number;
    /** The arc count A. */
    readonly arcCount: number;
    /** The logical edge count E. */
    readonly edgeCount: number;
    /** Whether the weights array exists. */
    readonly weighted: boolean;
    /** Whether the permutation is the identity, in which case arcToEdge / edgeToArc are not allocated. */
    readonly identity: boolean;
}

/** The five core arrays plus the arena they live in (null for separate buffers). */
export interface CoreArrays {
    /** nodeCount + 1 row offsets. */
    readonly rowPtr: U32;
    /** arcCount targets. */
    readonly colIdx: U32;
    /** arcCount f32 weights, or null when unweighted. */
    readonly weights: F32 | null;
    /** arcCount entries, or null when the permutation is the identity. */
    readonly arcToEdge: U32 | null;
    /** edgeCount entries, or null when the permutation is the identity. */
    readonly edgeToArc: U32 | null;
    /** The arena, or null when the arrays are separate buffers. */
    readonly arena: ArenaLayout | null;
}

/** The order of the core arrays inside the arena: hot to cold (design section 10.3). */
/** The core arrays in arena order, hot to cold (design section 10.3); the wire and fromCsr use the same order. */
export const CORE_ORDER: readonly CoreArrayName[] = ["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"];

/**
 * The unpadded byte length of every core array in arena order for a shape.
 * @param shape - the core shape
 * @returns five byte lengths; 0 for an absent array
 */
function byteLengthsOf(shape: CoreShape): number[] {
    const permArcs = shape.identity ? 0 : shape.arcCount;
    const permEdges = shape.identity ? 0 : shape.edgeCount;
    return [
        4 * (shape.nodeCount + 1),
        4 * shape.arcCount,
        shape.weighted ? 4 * shape.arcCount : 0,
        4 * permArcs,
        4 * permEdges,
    ];
}

/**
 * Allocate the core arrays as separate fresh buffers (design section 6.3 step 11 with `arena: false`,
 * and the transient target of a non-"keep" duplicate policy). Every array is a fresh typed array, so
 * I10 holds by the constructor guarantee.
 * @param shape - the core shape
 * @returns the arrays with `arena: null`
 */
export function allocateSeparateCore(shape: CoreShape): CoreArrays {
    return {
        rowPtr: new Uint32Array(shape.nodeCount + 1),
        colIdx: new Uint32Array(shape.arcCount),
        weights: shape.weighted ? new Float32Array(shape.arcCount) : null,
        arcToEdge: shape.identity ? null : new Uint32Array(shape.arcCount),
        edgeToArc: shape.identity ? null : new Uint32Array(shape.edgeCount),
        arena: null,
    };
}

/**
 * Allocate the core arrays inside one arena (design section 10.3): one ArrayBuffer of the total padded
 * size, every non-empty segment starting at a multiple of 256 bytes, ordered rowPtr, colIdx, weights,
 * arcToEdge, edgeToArc. `hotByteLength` is the end of the weights segment (or of colIdx when
 * unweighted). For the worked example of section 10.3 (n = 100,000, A = 2,000,000, E = 1,000,000,
 * weighted, permutations materialised) the segments start at 0 / 400,128 / 8,400,128 / 16,400,128 /
 * 24,400,128 and the arena is 28,400,128 bytes.
 * @param shape - the core shape
 * @returns the arrays as views into the arena, plus its layout
 */
export function allocateArenaCore(shape: CoreShape): CoreArrays {
    const byteLengths = byteLengthsOf(shape);
    const layout = layoutSegments(byteLengths, ALIGNMENT);
    const buffer = new ArrayBuffer(layout.byteLength);
    const segments: Record<CoreArrayName, ArenaSegment | null> = {
        rowPtr: null,
        colIdx: null,
        weights: null,
        arcToEdge: null,
        edgeToArc: null,
    };
    let hotByteLength = 0;
    for (let i = 0; i < CORE_ORDER.length; i++) {
        const offset = layout.offsets[i];
        if (offset === null) {
            continue;
        }
        const segment: ArenaSegment = Object.freeze({ byteOffset: offset, byteLength: byteLengths[i] });
        segments[CORE_ORDER[i]] = segment;
        if (i <= 2) {
            hotByteLength = offset + byteLengths[i];
        }
    }
    const u32 = (name: CoreArrayName, length: number): U32 => {
        const segment = segments[name];
        return segment === null ? new Uint32Array(0) : new Uint32Array(buffer, segment.byteOffset, length);
    };
    let weights: F32 | null = null;
    if (shape.weighted) {
        const segment = segments.weights;
        weights = segment === null ? new Float32Array(0) : new Float32Array(buffer, segment.byteOffset, shape.arcCount);
    }
    const arena: ArenaLayout = Object.freeze({
        buffer,
        byteOffset: 0,
        byteLength: layout.byteLength,
        alignment: ALIGNMENT,
        segments: Object.freeze(segments),
        hotByteLength,
    });
    return {
        rowPtr: u32("rowPtr", shape.nodeCount + 1),
        colIdx: u32("colIdx", shape.arcCount),
        weights,
        arcToEdge: shape.identity ? null : u32("arcToEdge", shape.arcCount),
        edgeToArc: shape.identity ? null : u32("edgeToArc", shape.edgeCount),
        arena,
    };
}

/**
 * Allocate the core arrays for a shape, in one arena or as separate buffers.
 * @param shape - the core shape
 * @param useArena - true for one 256-aligned arena (the freeze default), false for separate buffers
 * @returns the arrays
 */
export function allocateCoreArrays(shape: CoreShape, useArena: boolean): CoreArrays {
    return useArena ? allocateArenaCore(shape) : allocateSeparateCore(shape);
}

/**
 * The shape of an already built core.
 * @param core - the core arrays
 * @returns the shape
 */
export function shapeOfCore(core: CoreArrays): CoreShape {
    return {
        nodeCount: core.rowPtr.length - 1,
        arcCount: core.colIdx.length,
        edgeCount: core.edgeToArc === null ? core.colIdx.length : core.edgeToArc.length,
        weighted: core.weights !== null,
        identity: core.arcToEdge === null,
    };
}

/**
 * Copy a core built in separate buffers into a fresh arena of the same shape (the non-"keep" duplicate
 * policy path of design section 6.3 step 7 when nothing was merged: the transient arrays are final,
 * only their home changes). The source arrays are not modified.
 * @param core - the core in separate buffers
 * @returns the same contents as views into a new arena
 */
export function copyCoreIntoArena(core: CoreArrays): CoreArrays {
    const out = allocateArenaCore(shapeOfCore(core));
    out.rowPtr.set(core.rowPtr);
    out.colIdx.set(core.colIdx);
    if (out.weights !== null && core.weights !== null) {
        out.weights.set(core.weights);
    }
    if (out.arcToEdge !== null && core.arcToEdge !== null) {
        out.arcToEdge.set(core.arcToEdge);
    }
    if (out.edgeToArc !== null && core.edgeToArc !== null) {
        out.edgeToArc.set(core.edgeToArc);
    }
    return out;
}
