/**
 * `fromCsr()` (design section 8.1, entry point 2; decision C12): adopt prebuilt CSR arrays -- from
 * a file, a worker, another package or a generator that emits CSR directly -- without copying by
 * default (GraphBLAS pack semantics: the caller transfers ownership and must not mutate them
 * afterwards), with full validation by default because adopted arrays typically come from a file or
 * the network (design section 9.5).
 *
 * The pipeline: resolve the optional arrays (an absent `arcToEdge` is the identity of a directed
 * graph, an absent `edgeToArc` is derived in one O(m) pass, `edgeCount` defaults to the arc count or
 * is derived from `arcToEdge`); adopt or copy; detect an already-aligned shared buffer as the arena
 * (design section 10.3); build a candidate snapshot; check the rows (invariant I4) when `sortRows`
 * is set and, when some row is unsorted, rebuild a fresh sorted core through the freeze pipeline's
 * counting sorts over the declared edge list (the caller's arrays are not modified; the cost is a
 * freeze); validate at the requested level; verify the caller's flag claims (invariant I9).
 *
 * `detectArena()` is exported for `fromWire`, which applies the same detection when a manifest
 * carries no arena descriptor (design section 8.1).
 */

import { copyCoreIntoArena, CORE_ORDER, type CoreArrays } from "../builder/arena.js";
import { sortIntoCore } from "../builder/counting-sort.js";
import { createTable, tableWithColumns } from "../columns/table.js";
import { ALIGNMENT, INVALID_INDEX, MAX_COUNT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { identityNodeIdMap, nodeIdMapFromF64, nodeIdMapFromIds } from "../ids/node-id-map.js";
import { EMPTY_GRAPH_META, resolveGraphMeta as resolveSharedGraphMeta } from "../snapshot/graph-meta.js";
import { createSnapshot, type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import {
    checkI5Orientation,
    checkI6,
    checkI8NaN,
    checkI11Bijection,
    checkUniqueColumn,
    computeFlags,
    countLoopArcs,
    invariantViolation,
    isIdentity,
    validateFull,
    validateStructure,
} from "../snapshot/validate.js";
import {
    type ArenaLayout,
    type ArenaSegment,
    type AttributeTable,
    type ColumnInput,
    type CoreArrayName,
    type CsrInput,
    type F32,
    type F64,
    type FlagClaims,
    type FromCsrOptions,
    type GraphMeta,
    type GraphMetaPatch,
    type NodeIdMap,
    type SnapshotFlags,
    type TypedArrayData,
    type U32,
    type ValidationLevel,
} from "../types/index.js";
import { type SnapshotParts } from "../types/internal.js";
import { isOverPlainBuffer } from "../util/typed-array.js";

// ============================================================ constants

/** The flag names, for claim verification. */
const FLAG_NAMES: readonly (keyof SnapshotFlags)[] = [
    "multigraph",
    "hasSelfLoops",
    "arcToEdgeIsIdentity",
    "weighted",
    "allWeightsOne",
    "nonNegativeWeights",
    "finiteWeights",
];

const VALIDATION_LEVELS: ReadonlySet<string> = new Set(["none", "structure", "full"]);
// ============================================================ input shape

/** The core arrays after the optional ones are resolved; `arcToEdge` / `edgeToArc` are null when identity. */
interface ResolvedCore {
    readonly rowPtr: U32;
    readonly colIdx: U32;
    readonly weights: F32 | null;
    readonly arcToEdge: U32 | null;
    readonly edgeToArc: U32 | null;
    readonly edgeCount: number;
    readonly arcCount: number;
    readonly selfLoopCount: number;
}

/**
 * The E_INVALID_SNAPSHOT error for an input array of the wrong class.
 * @param name - the array
 * @param expected - the class name expected
 * @returns the error
 */
function arrayTypeError(name: string, expected: string): GraphFormatError {
    return new GraphFormatError("E_INVALID_SNAPSHOT", `${name} must be a ${expected}`, {
        array: name,
        expected,
        reason: "dtype",
    });
}

/**
 * Check the classes of the input arrays (a Float64Array passed as weights or a plain array passed as
 * colIdx would otherwise produce a snapshot whose bytes are wrong on the GPU) and the node count.
 * @param input - the input
 */
function checkShape(input: CsrInput): void {
    if (!Number.isInteger(input.nodeCount) || input.nodeCount < 0 || input.nodeCount > MAX_COUNT) {
        throw new GraphFormatError("E_TOO_LARGE", `nodeCount ${input.nodeCount} is not an integer in [0, MAX_COUNT]`, {
            count: input.nodeCount,
            max: MAX_COUNT,
        });
    }
    if (!(input.rowPtr instanceof Uint32Array)) {
        throw arrayTypeError("rowPtr", "Uint32Array");
    }
    if (!(input.colIdx instanceof Uint32Array)) {
        throw arrayTypeError("colIdx", "Uint32Array");
    }
    if (input.weights !== undefined && input.weights !== null && !(input.weights instanceof Float32Array)) {
        throw arrayTypeError("weights", "Float32Array");
    }
    if (input.arcToEdge !== undefined && !(input.arcToEdge instanceof Uint32Array)) {
        throw arrayTypeError("arcToEdge", "Uint32Array");
    }
    if (input.edgeToArc !== undefined && !(input.edgeToArc instanceof Uint32Array)) {
        throw arrayTypeError("edgeToArc", "Uint32Array");
    }
    if (input.edgeCount !== undefined && (!Number.isInteger(input.edgeCount) || input.edgeCount < 0)) {
        throw invariantViolation("I3", `edgeCount ${input.edgeCount} is not a non-negative integer`, {
            count: "edgeCount",
            found: input.edgeCount,
        });
    }
    if (input.rowPtr.length !== input.nodeCount + 1) {
        throw invariantViolation("I1", `rowPtr has ${input.rowPtr.length} entries, expected ${input.nodeCount + 1}`, {
            expected: input.nodeCount + 1,
            found: input.rowPtr.length,
        });
    }
}

/**
 * Whether a typed array is backed by a plain, fixed-length ArrayBuffer (a SharedArrayBuffer is never
 * adopted, decision D-SAB; a resizable buffer's view can change length, invariants I10 and I17).
 * @param array - the array
 * @returns true for a plain ArrayBuffer
 */
function overPlainBuffer(array: ArrayBufferView): boolean {
    return isOverPlainBuffer(array);
}

/**
 * Derive `edgeToArc` from `arcToEdge`: the first (lowest) arc holding each edge, which for a directed
 * graph is the edge's only arc and for an undirected graph the arc in the lower-numbered row
 * (design section 8.1: "derived in one O(m) pass"). An edge no arc holds is left INVALID_INDEX so the
 * I5 range check names it.
 * @param arcToEdge - the arc -> edge map
 * @param edgeCount - the edge count
 * @returns the edge -> arc map
 */
function deriveEdgeToArc(arcToEdge: U32, edgeCount: number): U32 {
    const edgeToArc = new Uint32Array(edgeCount).fill(INVALID_INDEX);
    for (let a = 0; a < arcToEdge.length; a++) {
        const e = arcToEdge[a];
        if (e < edgeCount && edgeToArc[e] === INVALID_INDEX) {
            edgeToArc[e] = a;
        }
    }
    return edgeToArc;
}

/**
 * The E_INVALID_SNAPSHOT error for an identity-flag contradiction between the input arrays.
 * @param message - what is wrong
 * @param details - the location
 * @returns the error
 */
function identityError(message: string, details: Readonly<Record<string, unknown>>): GraphFormatError {
    return invariantViolation("I5", message, details);
}

/**
 * Resolve the optional core arrays and counts (design section 8.1). For a directed input an absent
 * `arcToEdge` is the identity and a supplied one that happens to be the identity is dropped (the
 * snapshot materialises identity permutations lazily, design section 3.1); an undirected input must
 * supply `arcToEdge` (E_INVALID_SNAPSHOT, invariant I5). `edgeCount` defaults to the arc count
 * (directed) or to max(arcToEdge) + 1 (undirected). `selfLoopCount` is the number of loop arcs.
 * @param input - the input
 * @param level - the validation level ("none" skips the edgeToArc identity check)
 * @returns the resolved core
 */
function resolveCore(input: CsrInput, level: ValidationLevel): ResolvedCore {
    const { directed, nodeCount, rowPtr, colIdx } = input;
    const arcCount = colIdx.length;
    const weights = input.weights ?? null;
    const selfLoopCount = countLoopArcs(rowPtr, colIdx, nodeCount);
    let arcToEdge: U32 | null = input.arcToEdge ?? null;
    let edgeToArc: U32 | null = input.edgeToArc ?? null;
    let edgeCount: number;
    if (directed) {
        edgeCount = input.edgeCount ?? arcCount;
        if (edgeCount !== arcCount) {
            throw invariantViolation("I6", `directed input has arcCount ${arcCount} but edgeCount ${edgeCount}`, {
                arcCount,
                edgeCount,
            });
        }
        // decided from the supplied array at every level (O(m)): a flag claim never drops an array
        const identity = arcToEdge === null || (arcToEdge.length === arcCount && isIdentity(arcToEdge));
        if (identity) {
            if (edgeToArc !== null && level !== "none" && (edgeToArc.length !== edgeCount || !isIdentity(edgeToArc))) {
                throw identityError("arcToEdge is the identity but edgeToArc is not", {
                    array: "edgeToArc",
                    found: edgeToArc.length,
                    expected: edgeCount,
                });
            }
            arcToEdge = null;
            edgeToArc = null;
        } else if (edgeToArc === null) {
            edgeToArc = deriveEdgeToArc(arcToEdge as U32, edgeCount);
        }
    } else {
        if (arcToEdge === null) {
            throw identityError("an undirected input must supply arcToEdge", {
                array: "arcToEdge",
                reason: "missing arcToEdge",
            });
        }
        const { edgeCount: given } = input;
        if (given === undefined) {
            let max = -1;
            for (let a = 0; a < arcToEdge.length; a++) {
                if (arcToEdge[a] > max) {
                    max = arcToEdge[a];
                }
            }
            edgeCount = max + 1;
        } else {
            edgeCount = given;
        }
        edgeToArc ??= deriveEdgeToArc(arcToEdge, edgeCount);
    }
    return { rowPtr, colIdx, weights, arcToEdge, edgeToArc, edgeCount, arcCount, selfLoopCount };
}

// ============================================================ adoption and the arena

/**
 * The core arrays of a resolved core in the shape the arena helpers take.
 * @param core - the resolved core
 * @param arena - the arena, or null
 * @returns the core arrays
 */
function coreArraysOf(core: ResolvedCore, arena: ArenaLayout | null): CoreArrays {
    return {
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        arena,
    };
}

/**
 * Detect whether adopted core arrays already form an arena (design sections 8.1 and 10.3): every
 * present, non-empty, non-identity core array is a view over ONE plain ArrayBuffer, laid out hot to
 * cold (rowPtr, colIdx, weights, arcToEdge, edgeToArc) without overlap, each starting at a multiple
 * of 256 bytes relative to `rowPtr`. When they do, that buffer is the arena: `byteOffset` is
 * rowPtr's, `byteLength` spans to the end of the last array and `hotByteLength` ends at the weights
 * (or colIdx, or rowPtr) segment. Otherwise null. Arrays whose segment would be null (a zero-length
 * array, an absent weights array, an identity permutation given as null) do not take part.
 * @param core - the core arrays; `arcToEdge` / `edgeToArc` null when identity
 * @returns the arena layout, or null when the arrays are not an aligned shared buffer
 */
export function detectArena(core: Omit<CoreArrays, "arena">): ArenaLayout | null {
    const { buffer } = core.rowPtr;
    if (!(buffer instanceof ArrayBuffer)) {
        return null;
    }
    const base = core.rowPtr.byteOffset;
    const arrays: readonly (ArrayBufferView | null)[] = [
        core.rowPtr,
        core.colIdx,
        core.weights,
        core.arcToEdge,
        core.edgeToArc,
    ];
    const segments: Record<CoreArrayName, ArenaSegment | null> = {
        rowPtr: null,
        colIdx: null,
        weights: null,
        arcToEdge: null,
        edgeToArc: null,
    };
    let cursor = base;
    let hotEnd = base;
    for (let i = 0; i < CORE_ORDER.length; i++) {
        const array = arrays[i];
        if (array === null || array.byteLength === 0) {
            continue;
        }
        if (array.buffer !== buffer || array.byteOffset < cursor || (array.byteOffset - base) % ALIGNMENT !== 0) {
            return null;
        }
        segments[CORE_ORDER[i]] = Object.freeze({ byteOffset: array.byteOffset, byteLength: array.byteLength });
        cursor = array.byteOffset + array.byteLength;
        if (i <= 2) {
            hotEnd = cursor;
        }
    }
    return Object.freeze({
        buffer,
        byteOffset: base,
        byteLength: cursor - base,
        alignment: ALIGNMENT,
        segments: Object.freeze(segments),
        hotByteLength: hotEnd - base,
    });
}

/**
 * Whether any present core array is not over a plain ArrayBuffer, which forces the copy path.
 * @param core - the resolved core
 * @returns true when a copy is required
 */
function needsCopy(core: ResolvedCore): boolean {
    const arrays = [core.rowPtr, core.colIdx, core.weights, core.arcToEdge, core.edgeToArc];
    return arrays.some((array) => array !== null && !overPlainBuffer(array));
}

// ============================================================ ids, columns, meta

/**
 * The id map of a CsrInput (design section 4.2): identity with no storage when no ids are given;
 * otherwise the kind detected from the ids, validated (E_INVALID_ID / E_DUPLICATE_ID) unless the
 * level is "none". A Float64Array is adopted by reference unless `copy` is set.
 * @param input - the input
 * @param level - the validation level
 * @param copy - whether to copy an F64 array
 * @returns the id map; E_COLUMN_LENGTH when ids.length !== nodeCount
 */
function idMapOf(input: CsrInput, level: ValidationLevel, copy: boolean): NodeIdMap {
    const { ids, nodeCount } = input;
    if (ids === undefined) {
        return identityNodeIdMap(nodeCount);
    }
    if (ids.length !== nodeCount) {
        throw new GraphFormatError(
            "E_COLUMN_LENGTH",
            `ids has ${ids.length} entries, expected nodeCount ${nodeCount}`,
            {
                field: "ids",
                expected: nodeCount,
                found: ids.length,
            },
        );
    }
    const validate = level !== "none";
    if (ids instanceof Float64Array) {
        const values: F64 = copy || !overPlainBuffer(ids) ? new Float64Array(ids) : ids;
        return nodeIdMapFromF64(values, { validate });
    }
    return nodeIdMapFromIds(copy ? [...ids] : ids, nodeCount, { validate });
}

/**
 * The node or edge table of a CsrInput: the caller's columns attached through the same path as
 * `withColumns()` (typed arrays adopted by reference, JS arrays resolved with inference; design
 * section 5.7), copied first when `copy` is set.
 * @param domain - "node" or "edge"
 * @param rowCount - the table's row count
 * @param columns - the caller's columns, or undefined
 * @param copy - whether to copy typed arrays
 * @returns the table
 */
function tableOf(
    domain: "node" | "edge",
    rowCount: number,
    columns: Readonly<Record<string, TypedArrayData | ColumnInput>> | undefined,
    copy: boolean,
): AttributeTable {
    const table = createTable(domain, rowCount);
    if (columns === undefined) {
        return table;
    }
    // a typed array over a SharedArrayBuffer or a resizable buffer is copied like the core (D-SAB)
    const copied: Record<string, TypedArrayData | ColumnInput> = {};
    for (const name of Object.keys(columns)) {
        const value = columns[name];
        if (ArrayBuffer.isView(value)) {
            copied[name] = copy || !overPlainBuffer(value) ? value.slice() : value;
        } else if (ArrayBuffer.isView(value.data)) {
            copied[name] =
                copy || !overPlainBuffer(value.data) ? { data: value.data.slice(), decl: value.decl } : value;
        } else {
            copied[name] = value;
        }
    }
    return tableWithColumns(table, copied);
}

/**
 * Resolve a GraphMetaPatch into a complete GraphMeta (design section 5.9) with the builder's
 * `setMeta` rules over empty metadata (E_COLUMN_TYPE with details.field on a bad field).
 * @param patch - the patch, or undefined for no metadata
 * @returns the metadata, frozen
 */
export function resolveGraphMeta(patch: GraphMetaPatch | undefined): GraphMeta {
    return patch === undefined ? EMPTY_GRAPH_META : resolveSharedGraphMeta(EMPTY_GRAPH_META, patch);
}

// ============================================================ flags

/**
 * Whether every flag is claimed, so no pass over the arrays is needed to know them.
 * @param claims - the caller's claims
 * @returns true when all seven flags are defined
 */
function fullyClaimed(claims: FlagClaims | undefined): claims is SnapshotFlags {
    return claims !== undefined && FLAG_NAMES.every((name) => claims[name] !== undefined);
}

/**
 * The truthful flags of a core (design section 3.8) computed from the arrays.
 * @param directed - the direction
 * @param nodeCount - the node count
 * @param core - the core arrays
 * @param selfLoopCount - the loop count
 * @returns the flags
 */
function flagsOf(directed: boolean, nodeCount: number, core: CoreArrays, selfLoopCount: number): SnapshotFlags {
    return computeFlags({
        directed,
        nodeCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        selfLoopCount,
    });
}

/**
 * Verify the caller's flag claims against the computed flags (invariant I9): every claimed flag must
 * equal the truth.
 * @param claims - the claims, or undefined
 * @param flags - the truthful flags
 */
function verifyClaims(claims: FlagClaims | undefined, flags: SnapshotFlags): void {
    if (claims === undefined) {
        return;
    }
    for (const name of FLAG_NAMES) {
        const claim = claims[name];
        if (claim !== undefined && claim !== flags[name]) {
            throw invariantViolation("I9", `flag ${name} is claimed ${claim} but the arrays say ${flags[name]}`, {
                flag: name,
                found: claim,
                expected: flags[name],
            });
        }
    }
}

// ============================================================ rows and the rebuild

/**
 * Whether every row satisfies invariant I4: colIdx non-decreasing within the row, and parallel arcs in
 * ascending arcToEdge order.
 * @param s - the candidate snapshot
 * @returns true when sorted
 */
function rowsSorted(s: GraphSnapshot): boolean {
    const { rowPtr, colIdx, nodeCount } = s;
    const arcToEdge = s.flags.arcToEdgeIsIdentity ? null : s.arcToEdge;
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u] + 1; a < end; a++) {
            if (colIdx[a] < colIdx[a - 1]) {
                return false;
            }
            if (colIdx[a] === colIdx[a - 1] && arcToEdge !== null && arcToEdge[a] <= arcToEdge[a - 1]) {
                return false;
            }
        }
    }
    return true;
}

/**
 * The declared edge list of a candidate whose rows are unsorted: for every logical edge the arc
 * `edgeToArc[e]` gives the declared source (its row), target and weight.
 * @param s - the candidate snapshot
 * @returns the per-edge arrays
 */
function declaredEdges(s: GraphSnapshot): { readonly src: U32; readonly dst: U32; readonly weights: F32 | null } {
    const { nodeCount, edgeCount, rowPtr, colIdx, weights } = s;
    const identity = s.flags.arcToEdgeIsIdentity;
    const src = new Uint32Array(edgeCount);
    const dst = new Uint32Array(edgeCount);
    const edgeWeights = weights === null ? null : new Float32Array(edgeCount);
    const arcToEdge = identity ? null : s.arcToEdge;
    const edgeToArc = identity ? null : s.edgeToArc;
    for (let u = 0; u < nodeCount; u++) {
        const end = rowPtr[u + 1];
        for (let a = rowPtr[u]; a < end; a++) {
            const e = arcToEdge === null ? a : arcToEdge[a];
            if (edgeToArc !== null && edgeToArc[e] !== a) {
                continue;
            }
            src[e] = u;
            dst[e] = colIdx[a];
            if (edgeWeights !== null && weights !== null) {
                edgeWeights[e] = weights[a];
            }
        }
    }
    return { src, dst, weights: edgeWeights };
}

/**
 * Compare the rebuilt sorted core with the unsorted input row by row as multisets of (target, edge,
 * weight) (the "full" level's pairing and weight checks of invariant I7 for an input that could not
 * be walked in sorted order): an arc without a mate, a mate on a different edge, or unequal mate
 * weights all leave the rebuilt rows different from the input rows.
 * @param input - the candidate over the caller's arrays
 * @param rebuilt - the rebuilt snapshot
 */
function compareRows(input: GraphSnapshot, rebuilt: GraphSnapshot): void {
    const { nodeCount } = input;
    for (let u = 0; u <= nodeCount; u++) {
        if (input.rowPtr[u] !== rebuilt.rowPtr[u]) {
            throw invariantViolation("I7", `row ${u - 1} holds arcs whose mates are missing from their rows`, {
                row: Math.max(0, u - 1),
                found: input.rowPtr[u],
                expected: rebuilt.rowPtr[u],
            });
        }
    }
    const inArcToEdge = input.arcToEdge;
    const outArcToEdge = rebuilt.arcToEdge;
    const inWeights = input.weights;
    const outWeights = rebuilt.weights;
    for (let u = 0; u < nodeCount; u++) {
        const start = input.rowPtr[u];
        const end = input.rowPtr[u + 1];
        const order = new Uint32Array(end - start);
        for (let i = 0; i < order.length; i++) {
            order[i] = start + i;
        }
        order.sort((x, y) => input.colIdx[x] - input.colIdx[y] || inArcToEdge[x] - inArcToEdge[y]);
        for (let i = 0; i < order.length; i++) {
            const a = order[i];
            const b = start + i;
            const sameArc = input.colIdx[a] === rebuilt.colIdx[b] && inArcToEdge[a] === outArcToEdge[b];
            const sameWeight = inWeights === null || outWeights === null || inWeights[a] === outWeights[b];
            if (!sameArc || !sameWeight) {
                throw invariantViolation("I7", `arc ${a} in row ${u} has no matching mate`, {
                    row: u,
                    arc: a,
                    target: input.colIdx[a],
                    edge: inArcToEdge[a],
                });
            }
        }
    }
}

/**
 * Rebuild a sorted core from a candidate whose rows are unsorted through the freeze pipeline's
 * counting sorts (design section 6.3 steps 3-6 and 8) over the declared edge list, into a fresh arena
 * as `freeze()` would. Logical edge indices are preserved; the caller's arrays are not modified.
 * @param candidate - the candidate over the caller's arrays
 * @param parts - the candidate's parts (ids, tables, meta are reused)
 * @returns the rebuilt parts, with the truthful flags of the sorted core
 */
function rebuildSorted(candidate: GraphSnapshot, parts: SnapshotParts): SnapshotParts {
    const edges = declaredEdges(candidate);
    const sorted = sortIntoCore(
        {
            directed: candidate.directed,
            nodeCount: candidate.nodeCount,
            edgeCount: candidate.edgeCount,
            src: edges.src,
            dst: edges.dst,
            weights: edges.weights,
        },
        true,
    );
    return {
        ...parts,
        arcCount: sorted.arcCount,
        selfLoopCount: sorted.selfLoopCount,
        rowPtr: sorted.core.rowPtr,
        colIdx: sorted.core.colIdx,
        weights: sorted.core.weights,
        arcToEdge: sorted.core.arcToEdge,
        edgeToArc: sorted.core.edgeToArc,
        flags: sorted.flags,
        arena: sorted.core.arena,
    };
}

/**
 * The order-independent checks of the "full" level over a candidate that is about to be rebuilt:
 * the permutation rules (I5 orientation, I6), NaN weights (I8), the id bijection (I11) and unique
 * columns. Sortedness (I4), pairing (I7) and the flags (I9) are established by the rebuild itself.
 * @param s - the candidate
 */
function orderIndependentFullChecks(s: GraphSnapshot): void {
    checkI5Orientation(s);
    checkI6(s, true);
    checkI8NaN(s);
    checkI11Bijection(s);
    const tables: readonly [string, AttributeTable][] = [
        ["nodes", s.nodes],
        ["edges", s.edges],
        ["graph", s.graph],
    ];
    for (const [name, table] of tables) {
        for (const column of table) {
            checkUniqueColumn(name, column);
        }
    }
}

// ============================================================ the entry point

/**
 * Resolve the options with their defaults, checking the validation level.
 * @param options - the caller's options
 * @returns the resolved values
 */
function resolveOptions(options: FromCsrOptions): {
    readonly level: ValidationLevel;
    readonly copy: boolean;
    readonly sortRows: boolean;
} {
    const level = options.validate ?? "full";
    if (!VALIDATION_LEVELS.has(level)) {
        throw new GraphFormatError("E_UNSUPPORTED", `unsupported validation level ${level}`, {
            field: "validate",
            found: level,
            reason: "unsupported option",
        });
    }
    return { level, copy: options.copy === true, sortRows: options.sortRows !== false };
}

/**
 * Build a snapshot from prebuilt CSR arrays (design section 8.1). The arrays are ADOPTED without
 * copying by default (`copy: false`): the caller transfers ownership and must not mutate them
 * afterwards; when they already share one plain ArrayBuffer at 256-byte-aligned offsets in arena
 * order that buffer becomes the snapshot's arena, otherwise `arena` is null. With `copy: true` (or
 * when an array is over a SharedArrayBuffer) the core is copied into a fresh arena. `validate`
 * defaults to "full" (design section 9.5), which computes the flags and verifies the caller's claims
 * (invariant I9); "structure" and "none" trust every claimed flag and compute only the unclaimed ones
 * (design section 8.1: "structure" skips the sortedness / multigraph / flag checks and trusts
 * `flags`). With `sortRows: true` (the default) invariant I4 is checked in O(m) and unsorted rows
 * are rebuilt through the freeze pipeline into a fresh arena; `sortRows: false` asserts sorted rows,
 * checked under "full". A directed input without `arcToEdge` (or with the identity) keeps identity
 * permutations lazy, so `edgeList().weights` aliases `weights`.
 * @param input - the arrays and counts; an undirected input must supply arcToEdge
 * @param options - validation level, copy and sortRows
 * @returns the snapshot; E_INVALID_SNAPSHOT (details.invariant, location) on every rejected input,
 *   E_INVALID_ID / E_DUPLICATE_ID for bad ids, E_COLUMN_LENGTH for a column or id array of the wrong
 *   length, E_TOO_LARGE for a node count above MAX_COUNT
 */
export function fromCsr(input: CsrInput, options: FromCsrOptions = {}): GraphSnapshot {
    const { level, copy, sortRows } = resolveOptions(options);
    checkShape(input);
    const resolved = resolveCore(input, level);
    const { directed, nodeCount } = input;
    const claims = input.flags;

    // adopt (detecting an arena) or copy
    let core: CoreArrays;
    if (copy || needsCopy(resolved)) {
        core = copyCoreIntoArena(coreArraysOf(resolved, null));
    } else {
        core = coreArraysOf(resolved, detectArena(resolved));
    }

    // flags (design sections 3.8, 8.1 and 9.5): "full" computes them and verifies the claims (I9);
    // "structure" and "none" trust every claimed flag and compute only the unclaimed ones
    const trustClaims = level !== "full";
    let flags: SnapshotFlags;
    if (trustClaims && fullyClaimed(claims)) {
        flags = applyClaims(
            {
                ...claims,
                weighted: core.weights !== null,
                arcToEdgeIsIdentity: directed && core.arcToEdge === null,
            },
            claims,
        );
    } else {
        const computed = flagsOf(directed, nodeCount, core, resolved.selfLoopCount);
        flags = trustClaims ? applyClaims(computed, claims) : computed;
    }

    const parts: SnapshotParts = {
        label: null,
        serial: null,
        directed,
        nodeCount,
        edgeCount: resolved.edgeCount,
        arcCount: resolved.arcCount,
        selfLoopCount: resolved.selfLoopCount,
        rowPtr: core.rowPtr,
        colIdx: core.colIdx,
        weights: core.weights,
        arcToEdge: core.arcToEdge,
        edgeToArc: core.edgeToArc,
        flags,
        ids: idMapOf(input, level, copy),
        nodes: tableOf("node", nodeCount, input.nodeColumns, copy),
        edges: tableOf("edge", resolved.edgeCount, input.edgeColumns, copy),
        graph: createTable("graph", 1),
        extensions: new Map(),
        meta: resolveGraphMeta(input.meta),
        arena: core.arena,
        checksum: false,
    };
    const candidate = createSnapshot(parts);

    if (sortRows && !rowsSorted(candidate)) {
        // unsorted rows: check what can be checked without sorted rows, then rebuild
        if (level !== "none") {
            validateStructure(candidate);
        }
        if (level === "full") {
            orderIndependentFullChecks(candidate);
        }
        const rebuiltParts = rebuildSorted(candidate, parts);
        const rebuilt = createSnapshot({
            ...rebuiltParts,
            flags: trustClaims ? applyClaims(rebuiltParts.flags, claims) : rebuiltParts.flags,
        });
        if (level === "full") {
            if (!directed) {
                compareRows(candidate, rebuilt);
            }
            verifyClaims(claims, rebuilt.flags);
        }
        return rebuilt;
    }
    if (level === "structure") {
        validateStructure(candidate);
    } else if (level === "full") {
        validateFull(candidate);
        verifyClaims(claims, candidate.flags);
    }
    return candidate;
}

/** The flags that describe which arrays are PRESENT; they are never taken from a claim (I9 at every level). */
const PRESENCE_FLAGS: readonly (keyof SnapshotFlags)[] = ["weighted", "arcToEdgeIsIdentity"];

/**
 * The trusted flags of a "structure" or "none" level construction: every claimed predicate flag as
 * given, every other one as computed. The two presence flags (`weighted`, `arcToEdgeIsIdentity`)
 * follow the arrays at every level, and a claim that contradicts them is I9 immediately: a claim
 * must never change which arrays the snapshot holds.
 * @param computed - the flags computed from the arrays
 * @param claims - the caller's claims, or undefined
 * @returns the flags to store
 */
function applyClaims(computed: SnapshotFlags, claims: FlagClaims | undefined): SnapshotFlags {
    if (claims === undefined) {
        return computed;
    }
    const out: { -readonly [K in keyof SnapshotFlags]: boolean } = { ...computed };
    for (const name of FLAG_NAMES) {
        const value = claims[name];
        if (value === undefined) {
            continue;
        }
        if (PRESENCE_FLAGS.includes(name)) {
            if (value !== computed[name]) {
                throw invariantViolation(
                    "I9",
                    `flag ${name} is claimed ${value} but the arrays say ${computed[name]}`,
                    {
                        flag: name,
                        found: value,
                        expected: computed[name],
                    },
                );
            }
            continue;
        }
        out[name] = value;
    }
    return out;
}
