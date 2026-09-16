/**
 * The pure upload planner of spec 4.2: from a snapshot's shape, the device limits and the core arrays a kernel
 * needs, choose the one-buffer arena upload (the hot prefix, or the full arena when a cold segment is needed and
 * fits), one buffer per array, or arc windows over the arc-indexed arrays. No GPU object is touched here:
 * GraphResidency (./residency.ts) executes arena and perArray plans and P4 executes windowed plans (P1-P3 only
 * plan them). Every byte length comes from counts and flags, so an identity permutation is never materialised
 * (spec 4.1), and every arithmetic step uses `%` and Math.floor, never a bitwise operator: arc indices and byte
 * offsets exceed 2^31 at the 10M / 100M tier (graph-format invariant I3, spec 4.2).
 */

import { type CoreArrayName, type GraphSnapshot } from "@graphty/graph-format";

import { ARC_WINDOW_ALIGN } from "../constants.js";
import { WebGpuGraphError } from "../errors.js";
import { type PlanCaps, type PlanLimits } from "../types/context.js";
import { type ArcWindow } from "../types/memory.js";

/** The five core arrays in arena (hot-to-cold) order. */
const CORE_ORDER: readonly CoreArrayName[] = Object.freeze(["rowPtr", "colIdx", "weights", "arcToEdge", "edgeToArc"]);

/** The arc-indexed arrays: the only ones a window can cover (rowPtr is row-indexed, edgeToArc edge-indexed). */
const ARC_INDEXED: ReadonlySet<CoreArrayName> = new Set<CoreArrayName>(["colIdx", "weights", "arcToEdge"]);

/**
 * Where one core array's bytes go: one or more buffers (split at window boundaries only when the array exceeds
 * maxBufferSize). Exported with the plan union (contract 3.8): P4's windowed execution reads it; nothing at P1
 * imports it by name.
 * @public
 */
export interface PlannedArray {
    readonly name: CoreArrayName;
    readonly byteLength: number;
    readonly buffers: readonly { readonly byteOffset: number; readonly byteLength: number }[];
}

/** The arena path: ONE buffer of `bytes`, one writeBuffer, per-segment bindings at `segment.byteOffset - arena.byteOffset`. */
export interface ArenaPlan {
    readonly kind: "arena";
    readonly bytes: number;
    readonly includesCold: boolean;
    readonly segments: Readonly<Record<CoreArrayName, { readonly offset: number; readonly size: number } | null>>;
}

/** The per-array path: one buffer per needed array, whole-buffer bindings. */
export interface PerArrayPlan {
    readonly kind: "perArray";
    readonly arrays: readonly PlannedArray[];
}

/** The windowed path: per-array buffers plus the arc windows kernels iterate (P4 executes; P1-P3 only plan). */
export interface WindowedPlan {
    readonly kind: "windowed";
    readonly arrays: readonly PlannedArray[];
    readonly windows: readonly ArcWindow[];
    readonly arcsPerWindow: number;
}

/** One of the three plans. */
export type UploadPlan = ArenaPlan | PerArrayPlan | WindowedPlan;

/** One buffer range of a PlannedArray. */
interface BufferRange {
    readonly byteOffset: number;
    readonly byteLength: number;
}

/**
 * Byte length of every core array from the counts and flags alone; 0 means absent (unweighted, an identity
 * permutation, or zero length: spec 5.6 never binds a zero-length array). No array is read.
 * @param s - the snapshot
 * @returns the byte lengths keyed by core array name
 */
function coreByteLengths(s: GraphSnapshot): Readonly<Record<CoreArrayName, number>> {
    const arcs = s.arcCount;
    const identity = s.flags.arcToEdgeIsIdentity;
    return {
        rowPtr: 4 * (s.nodeCount + 1),
        colIdx: 4 * arcs,
        weights: s.flags.weighted && arcs > 0 ? 4 * arcs : 0,
        arcToEdge: !identity && arcs > 0 ? 4 * arcs : 0,
        edgeToArc: !identity && s.edgeCount > 0 ? 4 * s.edgeCount : 0,
    };
}

/**
 * rowPtr always, then every requested PRESENT array, in arena order (absent names are dropped, never an error).
 * @param need - the requested names
 * @param lengths - the byte lengths of coreByteLengths
 * @returns the names to plan
 */
function neededNames(
    need: readonly CoreArrayName[],
    lengths: Readonly<Record<CoreArrayName, number>>,
): CoreArrayName[] {
    const wanted = new Set<CoreArrayName>(need);
    wanted.add("rowPtr");
    return CORE_ORDER.filter((name) => wanted.has(name) && lengths[name] > 0);
}

/**
 * Rounds an arc count down to a multiple of ARC_WINDOW_ALIGN with `%` (a bitwise `& ~63` breaks above 2^31).
 * @param value - an arc count
 * @returns the largest multiple of 64 not above it
 */
function alignDown(value: number): number {
    return value - (value % ARC_WINDOW_ALIGN);
}

/**
 * The E_TOO_LARGE error of spec 5.7 for a plan that cannot exist.
 * @param needed - the bytes that do not fit
 * @param limit - the limit they exceed
 * @param path - "rowPtr" (no window can hold rowPtr) or "binding" (an edge-indexed array cannot be windowed)
 * @returns the error to throw
 */
function tooLarge(needed: number, limit: number, path: "rowPtr" | "binding"): WebGpuGraphError {
    return new WebGpuGraphError("E_TOO_LARGE", `${path}: ${needed} bytes exceed the device limit of ${limit} bytes`, {
        needed,
        limit,
        path,
        algorithm: null,
    });
}

/**
 * The E_INVALID_ARGUMENT error of a bad planner input.
 * @param argument - the argument name
 * @param value - the value given
 * @param expected - what was expected
 * @returns the error to throw
 */
function invalid(argument: string, value: unknown, expected: string): WebGpuGraphError {
    return new WebGpuGraphError("E_INVALID_ARGUMENT", `${argument}: expected ${expected}`, {
        argument,
        value,
        expected,
    });
}

/**
 * The largest 64-aligned arc count one binding holds: floor(maxStorageBufferBindingSize / 4) rounded down to a
 * multiple of ARC_WINDOW_ALIGN.
 * @param limits - the device limits
 * @returns the arcs per window (0 when the binding limit is below 256 bytes)
 */
export function arcsPerWindowFor(limits: PlanLimits): number {
    return alignDown(Math.floor(limits.maxStorageBufferBindingSize / 4));
}

/**
 * The window list for a rowPtr (pure; unit-tested with a synthetic rowPtr above 2^31 arcs): start = rowPtr[v0] -
 * rowPtr[v0] % ARC_WINDOW_ALIGN, at most `arcsPerWindow` arcs per window, a row longer than a window split across
 * windows. Whole rows first: a window ends at the last row boundary within `start + arcsPerWindow`; a row that does
 * not fit a window opened at its own aligned start is split into aligned chunks whose `rowFirst === rowLast` is that
 * row, and the next window continues the row at the chunk end. Every row belongs to a window (leading and trailing
 * empty rows included); every window is placed in buffer 0 at offset 4 x start (planUpload re-places them when an
 * array is split across buffers).
 * @param rowPtr - nodeCount + 1 row offsets (rowPtr[nodeCount] === arcCount)
 * @param arcCount - the arc count
 * @param arcsPerWindow - a positive multiple of ARC_WINDOW_ALIGN
 * @returns the windows in arc order (empty for arcCount 0)
 */
export function planArcWindows(rowPtr: Uint32Array, arcCount: number, arcsPerWindow: number): ArcWindow[] {
    if (rowPtr.length === 0) {
        throw invalid("rowPtr", rowPtr.length, "nodeCount + 1 entries");
    }
    if (
        !Number.isInteger(arcsPerWindow) ||
        arcsPerWindow < ARC_WINDOW_ALIGN ||
        arcsPerWindow % ARC_WINDOW_ALIGN !== 0
    ) {
        throw invalid("arcsPerWindow", arcsPerWindow, `a positive multiple of ${ARC_WINDOW_ALIGN}`);
    }
    const n = rowPtr.length - 1;
    if (rowPtr[n] !== arcCount) {
        throw invalid("arcCount", arcCount, `rowPtr[${n}] === ${rowPtr[n]}`);
    }
    const windows: ArcWindow[] = [];
    if (arcCount === 0) {
        return windows;
    }
    let pos = 0;
    let v = 0;
    let rowFirst = 0;
    while (pos < arcCount) {
        while (rowPtr[v + 1] <= pos) {
            v++;
        }
        const start = alignDown(pos);
        const limit = start + arcsPerWindow;
        if (rowPtr[v + 1] > limit) {
            windows.push({ start, end: limit, rowFirst, rowLast: v, bufferIndex: 0, offset: 4 * start });
            pos = limit;
            rowFirst = v;
        } else {
            let u = v;
            while (u + 1 < n && rowPtr[u + 2] <= limit) {
                u++;
            }
            const end = rowPtr[u + 1];
            windows.push({ start, end, rowFirst, rowLast: u, bufferIndex: 0, offset: 4 * start });
            pos = end;
            rowFirst = u + 1;
        }
    }
    return windows;
}

/**
 * Groups consecutive windows into buffers of at most `bufferLimit` bytes (spec 4.2: an array above maxBufferSize is
 * split across buffers at window boundaries) and re-places every window with its buffer index and offset.
 * @param windows - the windows of planArcWindows
 * @param arcBytes - the byte length of one arc-indexed array
 * @param bufferLimit - maxBufferSize
 * @returns the buffer ranges shared by every arc-indexed array and the placed windows
 */
function placeWindows(
    windows: readonly ArcWindow[],
    arcBytes: number,
    bufferLimit: number,
): { readonly buffers: BufferRange[]; readonly placed: ArcWindow[] } {
    if (arcBytes <= bufferLimit) {
        return { buffers: [{ byteOffset: 0, byteLength: arcBytes }], placed: [...windows] };
    }
    const buffers: BufferRange[] = [];
    const placed: ArcWindow[] = [];
    let first = 0;
    while (first < windows.length) {
        let last = first;
        while (last + 1 < windows.length && 4 * (windows[last + 1].end - windows[first].start) <= bufferLimit) {
            last++;
        }
        const base = windows[first].start;
        const bufferIndex = buffers.length;
        buffers.push({ byteOffset: 4 * base, byteLength: 4 * (windows[last].end - base) });
        for (let i = first; i <= last; i++) {
            const w = windows[i];
            placed.push({
                start: w.start,
                end: w.end,
                rowFirst: w.rowFirst,
                rowLast: w.rowLast,
                bufferIndex,
                offset: 4 * (w.start - base),
            });
        }
        first = last + 1;
    }
    return { buffers, placed };
}

/**
 * The pure planner of spec 4.2: arena (hot prefix unless `need` names a cold segment and the full arena fits) ->
 * perArray -> windowed, in that order. `need` always contains "rowPtr" (it is added when absent); absent arrays
 * (unweighted `weights`, identity permutations, zero-length arrays) are dropped; a snapshot with arcCount === 0
 * plans rowPtr only. The arena path needs `bytes <= maxBufferSize` and every needed segment within both limits;
 * ArenaPlan.segments lists every non-null segment inside the uploaded bytes (a needed cold segment outside them is
 * null: the residency uploads it as its own buffer). The perArray path needs every needed array within both limits;
 * otherwise the arc-indexed arrays are windowed (rowPtr and edgeToArc must fit, else E_TOO_LARGE).
 * @param s - the snapshot (only arena, the counts, the flags and rowPtr are read)
 * @param caps - the device capabilities (only `limits` is read)
 * @param need - the core arrays the kernel binds
 * @returns the plan
 */
export function planUpload(s: GraphSnapshot, caps: PlanCaps, need: readonly CoreArrayName[]): UploadPlan {
    const { limits } = caps;
    const binding = limits.maxStorageBufferBindingSize;
    const bufferLimit = limits.maxBufferSize;
    const fits = (bytes: number): boolean => bytes <= binding && bytes <= bufferLimit;
    const violated = (bytes: number): number => (bytes > binding ? binding : bufferLimit);
    const lengths = coreByteLengths(s);
    const names = neededNames(need, lengths);
    if (!fits(lengths.rowPtr)) {
        throw tooLarge(lengths.rowPtr, violated(lengths.rowPtr), "rowPtr");
    }
    const { arena } = s;
    if (arena !== null) {
        const coldNeeded = names.some(
            (name) => (name === "arcToEdge" || name === "edgeToArc") && arena.segments[name] !== null,
        );
        const includesCold = coldNeeded && arena.byteLength <= bufferLimit;
        const bytes = includesCold ? arena.byteLength : arena.hotByteLength;
        const bindable = names.every((name) => {
            const segment = arena.segments[name];
            return segment === null || fits(segment.byteLength);
        });
        if (bytes <= bufferLimit && bindable) {
            const segmentOf = (name: CoreArrayName): { readonly offset: number; readonly size: number } | null => {
                const segment = arena.segments[name];
                if (segment === null) {
                    return null;
                }
                const offset = segment.byteOffset - arena.byteOffset;
                return offset + segment.byteLength <= bytes ? { offset, size: segment.byteLength } : null;
            };
            return {
                kind: "arena",
                bytes,
                includesCold,
                segments: {
                    rowPtr: segmentOf("rowPtr"),
                    colIdx: segmentOf("colIdx"),
                    weights: segmentOf("weights"),
                    arcToEdge: segmentOf("arcToEdge"),
                    edgeToArc: segmentOf("edgeToArc"),
                },
            };
        }
    }
    if (names.every((name) => fits(lengths[name]))) {
        return {
            kind: "perArray",
            arrays: names.map((name) => ({
                name,
                byteLength: lengths[name],
                buffers: [{ byteOffset: 0, byteLength: lengths[name] }],
            })),
        };
    }
    if (names.includes("edgeToArc") && !fits(lengths.edgeToArc)) {
        throw tooLarge(lengths.edgeToArc, violated(lengths.edgeToArc), "binding");
    }
    const arcsPerWindow = Math.min(arcsPerWindowFor(limits), alignDown(Math.floor(bufferLimit / 4)));
    if (arcsPerWindow < ARC_WINDOW_ALIGN) {
        throw tooLarge(4 * ARC_WINDOW_ALIGN, binding, "binding");
    }
    const windows = planArcWindows(s.rowPtr, s.arcCount, arcsPerWindow);
    const { buffers, placed } = placeWindows(windows, 4 * s.arcCount, bufferLimit);
    const arrays: PlannedArray[] = names.map((name) =>
        ARC_INDEXED.has(name)
            ? { name, byteLength: lengths[name], buffers }
            : { name, byteLength: lengths[name], buffers: [{ byteOffset: 0, byteLength: lengths[name] }] },
    );
    return { kind: "windowed", arrays, windows: placed, arcsPerWindow };
}
