/**
 * `validate()` (design sections 3.2, 9.5 and 11.4): the invariants I1-I13 as one function each, the
 * two levels "structure" (O(n + m + columns): lengths, ranges, counts, alignment, column length
 * rules) and "full" (O(m log d): sortedness, orientation, pairing, NaN, flag recomputation, id
 * bijection, unique columns), and the flag predicates of design section 3.8 shared with the derived
 * graph core builder so that every flag has exactly one definition.
 *
 * Every check throws `E_INVALID_SNAPSHOT` on the first violation with `details.invariant` naming the
 * number ("I4") and the location (`row`, `arc`, `edge`, `column`, `table`, `flag`, `reason`). The
 * functions take the public `GraphSnapshot` contract, so test/invariants.test.ts runs the same
 * helpers over hand-written fixtures. Checksum comparison needs the snapshot's private records and
 * lives in graph-snapshot.ts.
 */

import { bitmapCount, bitmapGet } from "../columns/bitmap.js";
import { verifyUniqueColumn } from "../columns/table.js";
import { ALIGNMENT, INVALID_INDEX, MAX_COUNT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import {
    type ArenaLayout,
    type AttributeTable,
    type Column,
    type CoreArrayName,
    type F32,
    type GraphSnapshot,
    type SnapshotFlags,
    type U32,
} from "../types/index.js";
import { isFourByteAligned, isOverPlainBuffer } from "../util/typed-array.js";

// ============================================================ errors

/**
 * Build the E_INVALID_SNAPSHOT error of one violated invariant.
 * @param invariant - the invariant number, e.g. "I4"
 * @param message - a plain-ASCII description naming the location
 * @param details - the location (row, arc, edge, column, ...); `invariant` is added
 * @returns the error
 */
export function invariantViolation(
    invariant: string,
    message: string,
    details: Readonly<Record<string, unknown>> = {},
): GraphFormatError {
    return new GraphFormatError("E_INVALID_SNAPSHOT", `invariant ${invariant} violated: ${message}`, {
        invariant,
        ...details,
    });
}

// ============================================================ flags (3.8)

/** The arrays the flag predicates read; a subset of the core so the builder can call it before the snapshot exists. */
interface FlagSource {
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** The node count. */
    readonly nodeCount: number;
    /** The row offsets. */
    readonly rowPtr: U32;
    /** The sorted targets. */
    readonly colIdx: U32;
    /** The per-arc weights, or null. */
    readonly weights: F32 | null;
    /** The arc -> edge permutation, or null when it is known to be the identity. */
    readonly arcToEdge: U32 | null;
    /** The number of self-loop logical edges. */
    readonly selfLoopCount: number;
}

/**
 * Whether some row holds two arcs with equal colIdx (parallel edges); adjacent by invariant I4.
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param nodeCount - the node count
 * @returns true for a multigraph
 */
function hasParallelArcs(rowPtr: U32, colIdx: U32, nodeCount: number): boolean {
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u] + 1; a < end; a++) {
            if (colIdx[a] === colIdx[a - 1]) {
                return true;
            }
        }
    }
    return false;
}

/**
 * The number of arcs a with colIdx[a] === row(a).
 * @param rowPtr - the row offsets
 * @param colIdx - the sorted targets
 * @param nodeCount - the node count
 * @returns the loop arc count
 */
export function countLoopArcs(rowPtr: U32, colIdx: U32, nodeCount: number): number {
    let loops = 0;
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            if (colIdx[a] === u) {
                loops++;
            }
        }
    }
    return loops;
}

/**
 * Whether a permutation array is the identity.
 * @param perm - the array
 * @returns true when perm[i] === i for every i
 */
export function isIdentity(perm: U32): boolean {
    for (let i = 0; i < perm.length; i++) {
        if (perm[i] !== i) {
            return false;
        }
    }
    return true;
}

/**
 * Compute every flag of design section 3.8 from the arrays (invariant I9: never guessed).
 * @param core - the arrays and counts the predicates read
 * @returns the truthful flags
 */
export function computeFlags(core: FlagSource): SnapshotFlags {
    const { directed, nodeCount, rowPtr, colIdx, weights, arcToEdge } = core;
    let allWeightsOne = true;
    let nonNegativeWeights = true;
    let finiteWeights = true;
    if (weights !== null) {
        for (let a = 0; a < weights.length; a++) {
            const w = weights[a];
            if (w !== 1) {
                allWeightsOne = false;
            }
            if (!(w >= 0)) {
                nonNegativeWeights = false;
            }
            if (!Number.isFinite(w)) {
                finiteWeights = false;
            }
        }
    }
    return {
        multigraph: hasParallelArcs(rowPtr, colIdx, nodeCount),
        hasSelfLoops: core.selfLoopCount > 0,
        arcToEdgeIsIdentity: directed && (arcToEdge === null || isIdentity(arcToEdge)),
        weighted: weights !== null,
        allWeightsOne,
        nonNegativeWeights,
        finiteWeights,
    };
}

// ============================================================ I1 - I3: structure

/**
 * I1: rowPtr has nodeCount + 1 entries, starts at 0, is non-decreasing and ends at arcCount ===
 * colIdx.length.
 * @param s - the snapshot
 */
export function checkI1(s: GraphSnapshot): void {
    const { rowPtr, colIdx, nodeCount, arcCount } = s;
    if (rowPtr.length !== nodeCount + 1) {
        throw invariantViolation("I1", `rowPtr has ${rowPtr.length} entries, expected ${nodeCount + 1}`, {
            expected: nodeCount + 1,
            found: rowPtr.length,
        });
    }
    if (rowPtr[0] !== 0) {
        throw invariantViolation("I1", `rowPtr[0] is ${rowPtr[0]}, expected 0`, { row: 0, found: rowPtr[0] });
    }
    for (let u = 0; u < nodeCount; u++) {
        if (rowPtr[u + 1] < rowPtr[u]) {
            throw invariantViolation("I1", `rowPtr decreases at row ${u}: ${rowPtr[u]} -> ${rowPtr[u + 1]}`, {
                row: u,
            });
        }
    }
    if (rowPtr[nodeCount] !== arcCount || colIdx.length !== arcCount) {
        throw invariantViolation(
            "I1",
            `rowPtr[${nodeCount}] = ${rowPtr[nodeCount]}, colIdx.length = ${colIdx.length}, arcCount = ${arcCount}`,
            { row: nodeCount, found: rowPtr[nodeCount], expected: arcCount, colIdxLength: colIdx.length },
        );
    }
}

/**
 * I2: every colIdx entry is below nodeCount (so INVALID_INDEX never appears).
 * @param s - the snapshot
 */
export function checkI2(s: GraphSnapshot): void {
    const { colIdx, nodeCount } = s;
    for (let a = 0; a < colIdx.length; a++) {
        if (colIdx[a] >= nodeCount) {
            throw invariantViolation("I2", `colIdx[${a}] = ${colIdx[a]} is not below nodeCount ${nodeCount}`, {
                arc: a,
                found: colIdx[a],
            });
        }
    }
}

/**
 * I3: the counts are non-negative integers at most MAX_COUNT, edgeCount <= arcCount and selfLoopCount
 * <= edgeCount.
 * @param s - the snapshot
 */
export function checkI3(s: GraphSnapshot): void {
    const counts: readonly [string, number][] = [
        ["nodeCount", s.nodeCount],
        ["edgeCount", s.edgeCount],
        ["arcCount", s.arcCount],
        ["selfLoopCount", s.selfLoopCount],
    ];
    for (const [name, value] of counts) {
        if (!Number.isInteger(value) || value < 0 || value > MAX_COUNT) {
            throw invariantViolation("I3", `${name} = ${value} is not an integer in [0, MAX_COUNT]`, {
                count: name,
                found: value,
            });
        }
    }
    if (s.edgeCount > s.arcCount) {
        throw invariantViolation("I3", `edgeCount ${s.edgeCount} exceeds arcCount ${s.arcCount}`, {
            edgeCount: s.edgeCount,
            arcCount: s.arcCount,
        });
    }
    if (s.selfLoopCount > s.edgeCount) {
        throw invariantViolation("I3", `selfLoopCount ${s.selfLoopCount} exceeds edgeCount ${s.edgeCount}`, {
            selfLoopCount: s.selfLoopCount,
            edgeCount: s.edgeCount,
        });
    }
}

// ============================================================ I4 - I8: rows, permutations, pairing, weights

/**
 * I4: within every row colIdx is non-decreasing and arcs with equal colIdx are ordered by ascending
 * arcToEdge. Full level.
 * @param s - the snapshot
 */
export function checkI4(s: GraphSnapshot): void {
    const { rowPtr, colIdx, nodeCount } = s;
    const identity = s.flags.arcToEdgeIsIdentity;
    const arcToEdge = identity ? null : s.arcToEdge;
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u] + 1; a < end; a++) {
            if (colIdx[a] < colIdx[a - 1]) {
                throw invariantViolation("I4", `row ${u} is not sorted at arc ${a}`, { row: u, arc: a });
            }
            if (colIdx[a] === colIdx[a - 1] && arcToEdge !== null && arcToEdge[a] <= arcToEdge[a - 1]) {
                throw invariantViolation("I4", `parallel arcs ${a - 1} and ${a} in row ${u} are not in edge order`, {
                    row: u,
                    arc: a,
                });
            }
        }
    }
}

/**
 * I5 (structure part): arcToEdge has arcCount entries below edgeCount and edgeToArc has edgeCount
 * entries below arcCount; an identity flag requires arcCount === edgeCount and a directed snapshot.
 * @param s - the snapshot
 */
export function checkI5Ranges(s: GraphSnapshot): void {
    const { arcCount, edgeCount } = s;
    if (s.flags.arcToEdgeIsIdentity) {
        if (!s.directed || arcCount !== edgeCount) {
            throw invariantViolation("I5", "identity permutation claimed on a snapshot where it cannot hold", {
                directed: s.directed,
                arcCount,
                edgeCount,
            });
        }
        return;
    }
    const { arcToEdge, edgeToArc } = s;
    if (arcToEdge.length !== arcCount) {
        throw invariantViolation("I5", `arcToEdge has ${arcToEdge.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: arcToEdge.length,
        });
    }
    if (edgeToArc.length !== edgeCount) {
        throw invariantViolation("I5", `edgeToArc has ${edgeToArc.length} entries, expected ${edgeCount}`, {
            expected: edgeCount,
            found: edgeToArc.length,
        });
    }
    for (let a = 0; a < arcCount; a++) {
        if (arcToEdge[a] >= edgeCount) {
            throw invariantViolation("I5", `arcToEdge[${a}] = ${arcToEdge[a]} is not below edgeCount ${edgeCount}`, {
                arc: a,
                found: arcToEdge[a],
            });
        }
    }
    for (let e = 0; e < edgeCount; e++) {
        if (edgeToArc[e] >= arcCount) {
            throw invariantViolation("I5", `edgeToArc[${e}] = ${edgeToArc[e]} is not below arcCount ${arcCount}`, {
                edge: e,
                found: edgeToArc[e],
            });
        }
    }
}

/**
 * I5 (orientation part): arcToEdge[edgeToArc[e]] === e for every edge. Full level.
 * @param s - the snapshot
 */
export function checkI5Orientation(s: GraphSnapshot): void {
    if (s.flags.arcToEdgeIsIdentity) {
        return;
    }
    const { arcToEdge, edgeToArc, edgeCount } = s;
    for (let e = 0; e < edgeCount; e++) {
        if (arcToEdge[edgeToArc[e]] !== e) {
            throw invariantViolation("I5", `arcToEdge[edgeToArc[${e}]] = ${arcToEdge[edgeToArc[e]]}, expected ${e}`, {
                edge: e,
                arc: edgeToArc[e],
            });
        }
    }
}

/**
 * I6: a directed snapshot has arcCount === edgeCount (structure) and arcToEdge is a permutation of
 * 0..edgeCount-1 (full: every edge exactly once).
 * @param s - the snapshot
 * @param full - whether to run the permutation check
 */
export function checkI6(s: GraphSnapshot, full: boolean): void {
    if (!s.directed) {
        return;
    }
    const { arcCount, edgeCount } = s;
    if (arcCount !== edgeCount) {
        throw invariantViolation("I6", `directed snapshot has arcCount ${arcCount} but edgeCount ${edgeCount}`, {
            arcCount,
            edgeCount,
        });
    }
    if (!full || s.flags.arcToEdgeIsIdentity) {
        return;
    }
    const { arcToEdge } = s;
    const seen = new Uint8Array(edgeCount);
    for (let a = 0; a < arcCount; a++) {
        const e = arcToEdge[a];
        if (e >= edgeCount || seen[e] === 1) {
            throw invariantViolation("I6", `edge ${e} appears more than once in arcToEdge (arc ${a})`, {
                arc: a,
                edge: e,
            });
        }
        seen[e] = 1;
    }
}

/**
 * I7 (structure part): an undirected snapshot has arcCount === 2 * edgeCount - selfLoopCount.
 * @param s - the snapshot
 */
export function checkI7Counts(s: GraphSnapshot): void {
    if (s.directed) {
        return;
    }
    const expected = 2 * s.edgeCount - s.selfLoopCount;
    if (s.arcCount !== expected) {
        throw invariantViolation(
            "I7",
            `undirected snapshot has arcCount ${s.arcCount}, expected 2 * ${s.edgeCount} - ${s.selfLoopCount} = ${expected}`,
            { arcCount: s.arcCount, expected },
        );
    }
}

/**
 * I7 (pairing part): every arc u -> v with u !== v has exactly one mate v -> u with the same logical
 * edge and weight, a self-loop edge has exactly one arc, and every edge appears once (loop) or twice
 * (mates). The lockstep walk of design section 6.4. Full level.
 * @param s - the snapshot
 */
export function checkI7Pairing(s: GraphSnapshot): void {
    if (s.directed) {
        return;
    }
    const { nodeCount, arcCount, edgeCount, rowPtr, colIdx, weights, arcToEdge } = s;
    const occurrences = new Uint8Array(edgeCount);
    for (let a = 0; a < arcCount; a++) {
        const e = arcToEdge[a];
        if (occurrences[e] === 2) {
            throw invariantViolation("I7", `edge ${e} appears more than twice in arcToEdge (arc ${a})`, {
                arc: a,
                edge: e,
            });
        }
        occurrences[e]++;
    }
    const cursor = rowPtr.slice(0, nodeCount);
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        let a = cursor[u];
        if (a < rowPtr[u] || a > end) {
            throw invariantViolation("I7", `row ${u} was over-consumed by mates from earlier rows`, { row: u });
        }
        while (a < end) {
            const v = colIdx[a];
            let g = a + 1;
            while (g < end && colIdx[g] === v) {
                g++;
            }
            const k = g - a;
            if (v === u) {
                for (let i = a; i < g; i++) {
                    if (occurrences[arcToEdge[i]] !== 1) {
                        throw invariantViolation("I7", `self-loop edge ${arcToEdge[i]} has more than one arc`, {
                            row: u,
                            arc: i,
                            edge: arcToEdge[i],
                        });
                    }
                }
            } else {
                if (v < u) {
                    throw invariantViolation("I7", `arc ${a} in row ${u} targets ${v} but row ${v} holds no mate`, {
                        row: u,
                        arc: a,
                        target: v,
                    });
                }
                const b = cursor[v];
                const rowEnd = rowPtr[v + 1];
                for (let i = 0; i < k; i++) {
                    const arc = a + i;
                    const mate = b + i;
                    if (mate >= rowEnd || colIdx[mate] !== u) {
                        throw invariantViolation(
                            "I7",
                            `arc ${arc} in row ${u} targeting ${v} has no mate in row ${v}`,
                            {
                                row: u,
                                arc,
                                target: v,
                            },
                        );
                    }
                    if (arcToEdge[mate] !== arcToEdge[arc]) {
                        throw invariantViolation(
                            "I7",
                            `arcs ${arc} and ${mate} should share a logical edge but hold ${arcToEdge[arc]} and ${arcToEdge[mate]}`,
                            { row: u, arc, mate },
                        );
                    }
                    if (occurrences[arcToEdge[arc]] !== 2) {
                        throw invariantViolation("I7", `edge ${arcToEdge[arc]} of arc ${arc} does not have two arcs`, {
                            row: u,
                            arc,
                            edge: arcToEdge[arc],
                        });
                    }
                    if (weights !== null && weights[arc] !== weights[mate]) {
                        throw invariantViolation(
                            "I7",
                            `arcs ${arc} and ${mate} carry different weights ${weights[arc]} and ${weights[mate]}`,
                            { row: u, arc, mate },
                        );
                    }
                }
                if (b + k < rowEnd && colIdx[b + k] === u) {
                    throw invariantViolation("I7", `row ${v} holds more arcs to ${u} than row ${u} holds to ${v}`, {
                        row: v,
                        arc: b + k,
                        target: u,
                    });
                }
                cursor[v] = b + k;
            }
            a = g;
        }
        cursor[u] = end;
    }
}

/**
 * I8 (length part): weights is null or has arcCount entries.
 * @param s - the snapshot
 */
export function checkI8Length(s: GraphSnapshot): void {
    const { weights, arcCount } = s;
    if (weights !== null && weights.length !== arcCount) {
        throw invariantViolation("I8", `weights has ${weights.length} entries, expected ${arcCount}`, {
            expected: arcCount,
            found: weights.length,
        });
    }
}

/**
 * I8 (value part): no weight is NaN. Full level.
 * @param s - the snapshot
 */
export function checkI8NaN(s: GraphSnapshot): void {
    const { weights } = s;
    if (weights === null) {
        return;
    }
    for (let a = 0; a < weights.length; a++) {
        if (Number.isNaN(weights[a])) {
            throw invariantViolation("I8", `weights[${a}] is NaN`, { arc: a });
        }
    }
}

// ============================================================ I9 - I13: flags, alignment, ids, columns

/**
 * I9: every flag equals its predicate over the arrays, and selfLoopCount equals the number of loop
 * arcs. Full level.
 * @param s - the snapshot
 */
export function checkI9(s: GraphSnapshot): void {
    const loopArcs = countLoopArcs(s.rowPtr, s.colIdx, s.nodeCount);
    if (loopArcs !== s.selfLoopCount) {
        throw invariantViolation("I9", `selfLoopCount is ${s.selfLoopCount} but ${loopArcs} loop arcs exist`, {
            reason: "selfLoopCount",
            expected: loopArcs,
            found: s.selfLoopCount,
        });
    }
    const recomputed = computeFlags({
        directed: s.directed,
        nodeCount: s.nodeCount,
        rowPtr: s.rowPtr,
        colIdx: s.colIdx,
        weights: s.weights,
        arcToEdge: s.flags.arcToEdgeIsIdentity ? null : s.arcToEdge,
        selfLoopCount: s.selfLoopCount,
    });
    const names: readonly (keyof SnapshotFlags)[] = [
        "multigraph",
        "hasSelfLoops",
        "arcToEdgeIsIdentity",
        "weighted",
        "allWeightsOne",
        "nonNegativeWeights",
        "finiteWeights",
    ];
    for (const name of names) {
        if (s.flags[name] !== recomputed[name]) {
            throw invariantViolation("I9", `flag ${name} is ${s.flags[name]} but the arrays say ${recomputed[name]}`, {
                flag: name,
                found: s.flags[name],
                expected: recomputed[name],
            });
        }
    }
}

/**
 * Check one core array against its arena segment (design section 10.3).
 * @param arena - the arena
 * @param name - the core array name
 * @param array - the array, or null when absent
 */
function checkSegment(arena: ArenaLayout, name: CoreArrayName, array: ArrayBufferView | null): void {
    const segment = arena.segments[name];
    if (segment === null) {
        return;
    }
    if (array === null) {
        throw invariantViolation("I10", `arena names a segment for absent core array ${name}`, { array: name });
    }
    const relative = segment.byteOffset - arena.byteOffset;
    if (relative < 0 || relative % ALIGNMENT !== 0 || relative + segment.byteLength > arena.byteLength) {
        throw invariantViolation(
            "I10",
            `arena segment ${name} at ${segment.byteOffset} is not 256-aligned inside the arena`,
            {
                array: name,
                byteOffset: segment.byteOffset,
            },
        );
    }
    if (
        array.buffer !== arena.buffer ||
        array.byteOffset !== segment.byteOffset ||
        array.byteLength !== segment.byteLength
    ) {
        throw invariantViolation("I10", `core array ${name} is not the view its arena segment describes`, {
            array: name,
            byteOffset: array.byteOffset,
            byteLength: array.byteLength,
        });
    }
}

/**
 * I10: every core array is 4-byte aligned over a plain ArrayBuffer and, when an arena exists, every
 * array with a non-null segment is the 256-aligned view the segment describes and hotByteLength is the
 * end of the weights (or colIdx, or rowPtr) segment.
 * @param s - the snapshot
 */
export function checkI10(s: GraphSnapshot): void {
    const identity = s.flags.arcToEdgeIsIdentity;
    const arrays: readonly [CoreArrayName, ArrayBufferView | null][] = [
        ["rowPtr", s.rowPtr],
        ["colIdx", s.colIdx],
        ["weights", s.weights],
        ["arcToEdge", identity ? null : s.arcToEdge],
        ["edgeToArc", identity ? null : s.edgeToArc],
    ];
    for (const [name, array] of arrays) {
        if (array === null) {
            continue;
        }
        if (!isFourByteAligned(array)) {
            throw invariantViolation("I10", `core array ${name} is not 4-byte aligned`, {
                array: name,
                byteOffset: array.byteOffset,
                byteLength: array.byteLength,
            });
        }
        if (!isOverPlainBuffer(array)) {
            throw invariantViolation("I10", `core array ${name} is not a view over a plain ArrayBuffer`, {
                array: name,
                reason: "buffer",
            });
        }
    }
    const { arena } = s;
    if (arena === null) {
        return;
    }
    const { alignment }: { alignment: number } = arena;
    if (alignment !== ALIGNMENT) {
        throw invariantViolation("I10", `arena alignment is ${alignment}, expected ${ALIGNMENT}`, {
            found: alignment,
        });
    }
    for (const [name, array] of arrays) {
        checkSegment(arena, name, array);
    }
    let hotEnd = 0;
    for (const name of ["rowPtr", "colIdx", "weights"] as const) {
        const segment = arena.segments[name];
        if (segment !== null) {
            hotEnd = segment.byteOffset + segment.byteLength - arena.byteOffset;
        }
    }
    if (arena.hotByteLength !== hotEnd) {
        throw invariantViolation("I10", `arena.hotByteLength is ${arena.hotByteLength}, expected ${hotEnd}`, {
            reason: "hotByteLength",
            found: arena.hotByteLength,
            expected: hotEnd,
        });
    }
}

/**
 * I11 (size part): the id map has nodeCount ids.
 * @param s - the snapshot
 */
export function checkI11Size(s: GraphSnapshot): void {
    if (s.ids.size !== s.nodeCount) {
        throw invariantViolation("I11", `ids.size is ${s.ids.size}, expected nodeCount ${s.nodeCount}`, {
            expected: s.nodeCount,
            found: s.ids.size,
        });
    }
}

/**
 * I11 (bijection part): ids.indexOf(ids.idOf(i)) === i for every i and no id is NaN. Full level.
 * @param s - the snapshot
 */
export function checkI11Bijection(s: GraphSnapshot): void {
    const { ids, nodeCount } = s;
    for (let i = 0; i < nodeCount; i++) {
        const id = ids.idOf(i);
        if (typeof id === "number" && Number.isNaN(id)) {
            throw invariantViolation("I11", `id of node ${i} is NaN`, { index: i });
        }
        const back = ids.indexOf(id);
        if (back !== i) {
            throw invariantViolation("I11", `ids.indexOf(ids.idOf(${i})) is ${back}`, { index: i, found: back });
        }
    }
}

/**
 * The E_INVALID_SNAPSHOT error of one column length-rule violation (I12).
 * @param table - the table name for the message
 * @param column - the column
 * @param message - what is wrong
 * @param details - extra location details
 * @returns the error
 */
function columnViolation(
    table: string,
    column: Column,
    message: string,
    details: Readonly<Record<string, unknown>> = {},
): GraphFormatError {
    return invariantViolation("I12", `column "${column.meta.name}" of table ${table}: ${message}`, {
        table,
        column: column.meta.name,
        ...details,
    });
}

/**
 * Check that offsets are rows + 1 long, non-decreasing and end at `end`.
 * @param table - the table name for the message
 * @param column - the column
 * @param offsets - the offsets
 * @param rows - the row count
 * @param end - the expected last offset
 */
function checkOffsets(table: string, column: Column, offsets: U32, rows: number, end: number): void {
    if (offsets.length !== rows + 1) {
        throw columnViolation(table, column, `offsets has ${offsets.length} entries, expected ${rows + 1}`, {
            expected: rows + 1,
            found: offsets.length,
        });
    }
    if (offsets[0] !== 0) {
        throw columnViolation(table, column, `offsets[0] is ${offsets[0]}`, { row: 0 });
    }
    for (let r = 0; r < rows; r++) {
        if (offsets[r + 1] < offsets[r]) {
            throw columnViolation(table, column, `offsets decrease at row ${r}`, { row: r });
        }
    }
    if (offsets[rows] !== end) {
        throw columnViolation(table, column, `offsets[${rows}] is ${offsets[rows]}, expected ${end}`, {
            row: rows,
            found: offsets[rows],
            expected: end,
        });
    }
}

/**
 * Check the values of a refersTo u32 array: every entry below `bound` or INVALID_INDEX; when
 * `rowValidity` is given an INVALID_INDEX entry requires its row to be unset.
 * @param table - the table name for the message
 * @param column - the column
 * @param values - the u32 values
 * @param components - values per row
 * @param bound - the referenced space's row count
 * @param rowValidity - the validity bitmap of the rows (null = all set), or undefined to skip the unset rule
 */
function checkReferences(
    table: string,
    column: Column,
    values: U32,
    components: number,
    bound: number,
    rowValidity: U32 | null | undefined,
): void {
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (value === INVALID_INDEX) {
            const row = Math.floor(i / components);
            if (rowValidity !== undefined && (rowValidity === null || bitmapGet(rowValidity, row))) {
                throw columnViolation(table, column, `row ${row} is set but holds INVALID_INDEX`, { row });
            }
            continue;
        }
        if (value >= bound) {
            throw columnViolation(table, column, `value ${value} at ${i} is not below ${bound}`, {
                row: Math.floor(i / components),
                found: value,
                expected: bound,
            });
        }
    }
}

/**
 * The refersTo rules of invariant I12 for one column: a u32 column's values (or a list column's u32
 * child items) are below `bound` or INVALID_INDEX, and a scalar INVALID_INDEX sits in an unset row.
 * @param table - the table name for the message
 * @param column - a column whose meta.refersTo is set
 * @param bound - the referenced space's row count
 */
function checkColumnReferences(table: string, column: Column, bound: number): void {
    switch (column.dtype) {
        case "u32":
            checkReferences(table, column, column.data, column.meta.components, bound, column.validity);
            return;
        case "list": {
            const { child } = column;
            if (child.dtype !== "u32") {
                throw columnViolation(table, column, `refersTo list requires a u32 child, found ${child.dtype}`);
            }
            checkReferences(table, column, child.data, 1, bound, undefined);
            return;
        }
        default:
            throw columnViolation(table, column, `refersTo requires dtype u32, found ${column.dtype}`);
    }
}

/**
 * The typed arrays a column exposes to a GPU consumer (data, validity, codes, list child data): each
 * must be a view over a plain ArrayBuffer (I10, decision D-SAB).
 * @param column - the column
 * @returns the arrays with their names
 */
function columnArrays(column: Column): readonly [string, ArrayBufferView | null][] {
    switch (column.dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
        case "bool":
            return [
                ["data", column.data],
                ["validity", column.validity],
            ];
        case "dict":
            return [
                ["codes", column.codes],
                ["validity", column.validity],
            ];
        case "string":
            return [
                ["offsets", column.offsets],
                ["utf8", column.utf8],
                ["validity", column.validity],
            ];
        case "list":
            return [
                ["offsets", column.offsets],
                ["validity", column.validity],
                ...columnArrays(column.child).map(([name, array]): [string, ArrayBufferView | null] => [
                    `child.${name}`,
                    array,
                ]),
            ];
        case "json":
            return [["validity", column.validity]];
        default: {
            const unknown: never = column;
            throw invariantViolation("I12", `unknown dtype ${(unknown as Column).dtype}`);
        }
    }
}

/**
 * The length rules of design section 5.7 for one column, plus dictionary code ranges, refersTo
 * ranges, the recorded nullCount (recomputed from the bitmap, design section 9.5) and the plain
 * ArrayBuffer rule of I10 for every typed array the column exposes (I12).
 * @param table - the table name for the message
 * @param column - the column
 * @param rows - the table's row count
 * @param s - the snapshot (for the referenced row counts)
 */
export function checkColumnRules(table: string, column: Column, rows: number, s: GraphSnapshot): void {
    if (column.length !== rows) {
        throw columnViolation(table, column, `has ${column.length} rows, expected ${rows}`, {
            expected: rows,
            found: column.length,
        });
    }
    const { validity, meta } = column;
    const validityWords = Math.ceil(rows / 32);
    if (validity !== null && validity.length !== validityWords) {
        throw columnViolation(table, column, `validity has ${validity.length} words, expected ${validityWords}`, {
            expected: validityWords,
            found: validity.length,
        });
    }
    const nullCount = validity === null ? 0 : rows - bitmapCount(validity, rows);
    if (column.nullCount !== nullCount) {
        throw columnViolation(
            table,
            column,
            `nullCount is ${column.nullCount}, the validity bitmap says ${nullCount}`,
            {
                expected: nullCount,
                found: column.nullCount,
                reason: "nullCount",
            },
        );
    }
    for (const [name, array] of columnArrays(column)) {
        if (array !== null && !isOverPlainBuffer(array)) {
            throw columnViolation(table, column, `${name} is not a view over a plain ArrayBuffer`, {
                array: name,
                reason: "buffer",
            });
        }
    }
    const refBound = meta.refersTo === "node" ? s.nodeCount : s.edgeCount;
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            const expected = rows * meta.components;
            if (column.data.length !== expected) {
                throw columnViolation(table, column, `data has ${column.data.length} values, expected ${expected}`, {
                    expected,
                    found: column.data.length,
                });
            }
            if (meta.refersTo !== null) {
                checkColumnReferences(table, column, refBound);
            }
            break;
        }
        case "bool":
            if (column.data.length !== validityWords) {
                throw columnViolation(
                    table,
                    column,
                    `bool data has ${column.data.length} words, expected ${validityWords}`,
                    {
                        expected: validityWords,
                        found: column.data.length,
                    },
                );
            }
            break;
        case "dict": {
            if (column.codes.length !== rows) {
                throw columnViolation(table, column, `codes has ${column.codes.length} entries, expected ${rows}`, {
                    expected: rows,
                    found: column.codes.length,
                });
            }
            const size = column.dictionary.length;
            for (let r = 0; r < rows; r++) {
                if (column.codes[r] >= size && (validity === null || bitmapGet(validity, r))) {
                    throw columnViolation(table, column, `code ${column.codes[r]} at row ${r} is not below ${size}`, {
                        row: r,
                        found: column.codes[r],
                    });
                }
            }
            break;
        }
        case "string":
            checkOffsets(table, column, column.offsets, rows, column.utf8.length);
            break;
        case "list": {
            const { child } = column;
            checkOffsets(table, column, column.offsets, rows, child.length);
            if (child.validity !== null) {
                throw columnViolation(table, column, "list child column has a validity bitmap");
            }
            if (meta.refersTo !== null) {
                checkColumnReferences(table, column, refBound);
            }
            break;
        }
        case "json":
            if (column.values.length !== rows) {
                throw columnViolation(table, column, `values has ${column.values.length} entries, expected ${rows}`, {
                    expected: rows,
                    found: column.values.length,
                });
            }
            break;
        default: {
            const unknown: never = dtype;
            throw columnViolation(table, column, `unknown dtype ${String(unknown)}`);
        }
    }
}

/**
 * Check every column of a table against its row count.
 * @param name - the table name for the message
 * @param table - the table
 * @param rows - the expected row count
 * @param s - the snapshot
 */
function checkTable(name: string, table: AttributeTable, rows: number, s: GraphSnapshot): void {
    for (const column of table) {
        checkColumnRules(name, column, rows, s);
    }
}

/**
 * I12: the node, edge and graph tables have nodeCount, edgeCount and 1 rows, every column (extension
 * tables included) obeys its length rules, dictionary codes are in range and refersTo values are in
 * range or INVALID_INDEX with the row unset.
 * @param s - the snapshot
 */
export function checkI12(s: GraphSnapshot): void {
    const tables: readonly [string, AttributeTable, number][] = [
        ["nodes", s.nodes, s.nodeCount],
        ["edges", s.edges, s.edgeCount],
        ["graph", s.graph, 1],
    ];
    for (const [name, table, rows] of tables) {
        if (table.rowCount !== rows) {
            throw invariantViolation("I12", `table ${name} has ${table.rowCount} rows, expected ${rows}`, {
                table: name,
                expected: rows,
                found: table.rowCount,
            });
        }
        checkTable(name, table, rows, s);
    }
    for (const [name, table] of s.extensions) {
        checkTable(`extensions[${name}]`, table, table.rowCount, s);
    }
}

/**
 * I13: edge attribute columns are indexed by logical edge: the edge table has edgeCount rows.
 * @param s - the snapshot
 */
export function checkI13(s: GraphSnapshot): void {
    if (s.edges.rowCount !== s.edgeCount) {
        throw invariantViolation("I13", `edge table has ${s.edges.rowCount} rows, expected edgeCount ${s.edgeCount}`, {
            expected: s.edgeCount,
            found: s.edges.rowCount,
        });
    }
}

// ============================================================ levels

/**
 * The "structure" level of design section 9.5: I1-I3, I5 lengths and ranges, I6 / I7 counts, I8
 * length, I10, I11 size, I12 lengths and ranges, I13. O(n + m + columns).
 * @param s - the snapshot
 */
export function validateStructure(s: GraphSnapshot): void {
    checkI3(s);
    checkI1(s);
    checkI2(s);
    checkI5Ranges(s);
    checkI6(s, false);
    checkI7Counts(s);
    checkI8Length(s);
    checkI10(s);
    checkI11Size(s);
    checkI13(s);
    checkI12(s);
}

/**
 * The "full" level of design section 9.5: structure plus I4, I5 orientation, I6 permutation, I8 NaN
 * (before the pairing walk, so a NaN weight is reported as I8 rather than as an unequal mate), I7
 * pairing, I9 flags, I11 bijection and unique columns. O(m log d).
 * @param s - the snapshot
 */
export function validateFull(s: GraphSnapshot): void {
    validateStructure(s);
    checkI4(s);
    checkI5Orientation(s);
    checkI6(s, true);
    checkI8NaN(s);
    checkI7Pairing(s);
    checkI9(s);
    checkI11Bijection(s);
    const tables: readonly [string, AttributeTable][] = [
        ["nodes", s.nodes],
        ["edges", s.edges],
        ["graph", s.graph],
        ...[...s.extensions].map(([name, table]): [string, AttributeTable] => [`extensions[${name}]`, table]),
    ];
    for (const [name, table] of tables) {
        for (const column of table) {
            checkUniqueColumn(name, column);
        }
    }
}

/**
 * The `unique` rule of a column as a validation failure (design sections 9.5 and 11.4): the
 * freeze-time codes E_DUPLICATE_ID / E_DUPLICATE_EDGE_ID of `verifyUniqueColumn` become
 * E_INVALID_SNAPSHOT with `details.invariant` "I12", `details.reason` "unique" and the original
 * code in `details.cause`, so a reader of untrusted input sees one error family.
 * @param table - the table name for the message
 * @param column - the column
 */
export function checkUniqueColumn(table: string, column: Column): void {
    try {
        verifyUniqueColumn(column);
    } catch (err) {
        if (err instanceof GraphFormatError && (err.code === "E_DUPLICATE_ID" || err.code === "E_DUPLICATE_EDGE_ID")) {
            throw columnViolation(table, column, err.message, { ...err.details, reason: "unique", cause: err.code });
        }
        throw err;
    }
}
