/**
 * The immutable id-to-index bijection of a snapshot (design section 4): five storage kinds chosen by
 * inspecting the ids once in O(n), SameValueZero equality, `INVALID_INDEX` on a miss, a dense-integer
 * fast path with no Map at all, lazily materialised typed / decoded representations in either
 * direction, and a reverse `Map` that is shared with the builder (guarded by `index < size`) or built
 * lazily on the first lookup.
 *
 * The class exposes exactly the public surface of design section 12.2. Everything the sibling
 * modules need beyond it (construction from ids, typed arrays or the wire; the typed form for the
 * wire; gather and remap for derived graphs and compaction) is a module-level function that reaches
 * the private storage through a module-private WeakMap, so nothing internal appears on the instance
 * type.
 */

import { INVALID_INDEX, MAX_COUNT } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import {
    type F64,
    type NodeId,
    type NodeIdMapContract,
    type NodeIdMapKind,
    type U8,
    type U32,
    type ValidationLevel,
} from "../types/index.js";
import { assertOneOf } from "../util/options.js";
import { claimHolder } from "../util/shared-buffers.js";
import { checkUtf8Layout, encodeUtf8Rows, hasLoneSurrogate, resolveRange, Utf8Store } from "./string-store.js";

// ============================================================ id validation

/**
 * A short, plain-ASCII rendering of an id for error messages.
 * @param id - the value
 * @returns the number, a quoted (possibly truncated) string, or the type name
 */
export function describeId(id: unknown): string {
    if (typeof id === "number") {
        return String(id);
    }
    if (typeof id === "string") {
        const text = id.length > 40 ? `${id.slice(0, 37)}...` : id;
        return JSON.stringify(text);
    }
    if (id === null) {
        return "null";
    }
    if (typeof id === "bigint") {
        return `${String(id)}n`;
    }
    return `a value of type ${typeof id}`;
}

/**
 * Check a value as a node id and normalise it (design section 4.1): a finite number (`-0` becomes
 * `0`) or a string without a lone surrogate. NaN, non-finite numbers, bigints, objects, null and
 * undefined are E_INVALID_ID; a lone surrogate is E_INVALID_ID with details.reason "lone surrogate".
 * @param id - the value to check
 * @returns the id as stored
 */
export function validateNodeId(id: unknown): NodeId {
    if (typeof id === "number") {
        if (!Number.isFinite(id)) {
            throw new GraphFormatError("E_INVALID_ID", `invalid node id ${String(id)}: not a finite number`, {
                id,
                reason: "non-finite",
            });
        }
        return id === 0 ? 0 : id;
    }
    if (typeof id === "string") {
        if (hasLoneSurrogate(id)) {
            throw new GraphFormatError("E_INVALID_ID", `invalid node id ${describeId(id)}: lone surrogate`, {
                id,
                reason: "lone surrogate",
            });
        }
        return id;
    }
    throw new GraphFormatError("E_INVALID_ID", `invalid node id: ${describeId(id)} is not a string or a number`, {
        reason: "unsupported type",
        type: typeof id,
    });
}

// ============================================================ storage

/**
 * The private state of a NodeIdMap (design section 4.2). Which slots are non-null depends on the
 * kind and on which representations have been materialised so far:
 *
 * - identity: nothing;
 * - dense: denseValues (index -> id) and denseInverse (id -> index, INVALID_INDEX where absent);
 * - numeric: decoded (the builder's array, shared) and / or numbers (F64, the wire form); map;
 * - string: strings (a Utf8Store over the builder's array or over the wire buffers); map;
 * - mixed: decoded (shared) and / or tags + numbers + strings (the wire form); map.
 *
 * `map` is null until first needed for numeric / string / mixed and always null for identity / dense.
 * The object is mutable so that lazily materialised representations can be cached; the map instance
 * itself never changes. Exported only because the public constructor names it; never exported from
 * the package barrel.
 */
interface NodeIdMapStorage {
    /** The storage kind. */
    readonly kind: NodeIdMapKind;
    /** Number of ids. */
    readonly size: number;
    /** identity only: id === index + offset; 0 otherwise. */
    readonly offset: number;
    /** dense: the id of every index. */
    readonly denseValues: U32 | null;
    /** dense: index of every id in [0, maxId], INVALID_INDEX where absent. */
    readonly denseInverse: U32 | null;
    /** numeric / mixed: the decoded ids, shared by reference; entries at or beyond `size` are invisible. */
    readonly decoded: readonly NodeId[] | null;
    /** numeric: the ids as f64 (the wire form); mixed: the number rows, 0 where the row is a string. */
    numbers: F64 | null;
    /** mixed: 0 for a number row, 1 for a string row (the wire form). */
    tags: U8 | null;
    /** string / mixed: the Utf8 store (string: every row; mixed: string rows, number rows empty). */
    strings: Utf8Store | null;
    /** numeric / string / mixed: the reverse map; shared with the builder or built on first lookup. */
    map: Map<NodeId, number> | null;
    /** String(idOf(i)) -> i, built lazily by stringIndex(). */
    stringIndex: Map<string, number> | null;
}

/** The typed representation of an id map, one slot per wire member (design section 4.5). */
export interface NodeIdMapTypedParts {
    /** The storage kind. */
    readonly kind: NodeIdMapKind;
    /** Number of ids. */
    readonly size: number;
    /** identity only: id === index + offset; 0 otherwise. */
    readonly offset: number;
    /** dense: u32 ids per index; numeric: f64 ids per index; null otherwise. */
    readonly values: U32 | F64 | null;
    /** mixed: u8 tag per index (0 number, 1 string); null otherwise. */
    readonly tags: U8 | null;
    /** mixed: f64 per index (0 where string); null otherwise. */
    readonly numbers: F64 | null;
    /** string / mixed: rows + 1 offsets of the Utf8 store; null otherwise. */
    readonly offsets: U32 | null;
    /** string / mixed: the UTF-8 bytes; null otherwise. */
    readonly utf8: U8 | null;
}

/** Storage behind every instance, reachable by the module-level functions but not by consumers. */
const INTERNALS = new WeakMap<NodeIdMap, NodeIdMapStorage>();

/**
 * The storage of an instance.
 * @param map - the id map
 * @returns its private storage
 */
function storageOf(map: NodeIdMap): NodeIdMapStorage {
    const storage = INTERNALS.get(map);
    if (storage === undefined) {
        throw new GraphFormatError("E_UNSUPPORTED", "not a NodeIdMap of this package", { reason: "foreign id map" });
    }
    return storage;
}

/**
 * A storage record with every slot null except the given ones.
 * @param kind - the storage kind
 * @param size - the id count
 * @param slots - the non-null slots
 * @returns the storage
 */
function makeStorage(kind: NodeIdMapKind, size: number, slots: Partial<NodeIdMapStorage> = {}): NodeIdMapStorage {
    return {
        kind,
        size,
        offset: 0,
        denseValues: null,
        denseInverse: null,
        decoded: null,
        numbers: null,
        tags: null,
        strings: null,
        map: null,
        stringIndex: null,
        ...slots,
    };
}

/**
 * Whether a typed array the storage still reads from was transferred away.
 * @param storage - the storage
 * @returns true when detached
 */
function storageDetached(storage: NodeIdMapStorage): boolean {
    if (storage.size === 0) {
        return false;
    }
    return (
        (storage.denseValues !== null && storage.denseValues.length === 0) ||
        (storage.decoded === null && storage.numbers !== null && storage.numbers.length === 0) ||
        (storage.decoded === null && storage.tags !== null && storage.tags.length === 0) ||
        (storage.decoded === null && storage.strings !== null && storage.strings.detached)
    );
}

/**
 * Whether an id map's typed storage was transferred away (design section 9.1), for validate().
 * @param map - the id map
 * @returns true when detached
 */
export function idMapDetached(map: NodeIdMap): boolean {
    const storage = INTERNALS.get(map);
    return storage !== undefined && storageDetached(storage);
}

/**
 * -0 is stored and reported as 0 (SameValueZero, design section 4.1).
 * @param value - a numeric id
 * @returns the id with -0 folded into 0
 */
function normaliseZero(value: number): number {
    return value === 0 ? 0 : value;
}

/**
 * The E_DUPLICATE_ID error for two indices holding the same id.
 * @param id - the id
 * @param first - the lower index
 * @param second - the higher index
 * @returns the error
 */
function duplicateIdError(id: NodeId, first: number, second: number): GraphFormatError {
    return new GraphFormatError(
        "E_DUPLICATE_ID",
        `duplicate node id ${describeId(id)} at indices ${first} and ${second}`,
        {
            id,
            indices: [first, second],
        },
    );
}

/**
 * The E_INVALID_SNAPSHOT error for an id bijection failure of untrusted typed input (invariant I11).
 * @param reason - what failed
 * @param details - extra context
 * @returns the error
 */
function bijectionError(reason: string, details: Readonly<Record<string, unknown>>): GraphFormatError {
    return new GraphFormatError("E_INVALID_SNAPSHOT", `id map violates invariant I11: ${reason}`, {
        invariant: "I11",
        reason,
        ...details,
    });
}

/**
 * The E_BAD_SERIALIZATION error for a malformed typed id map.
 * @param ref - the manifest member
 * @param reason - what is wrong
 * @param details - extra context
 * @returns the error
 */
function layoutError(ref: string, reason: string, details: Readonly<Record<string, unknown>> = {}): GraphFormatError {
    return new GraphFormatError("E_BAD_SERIALIZATION", `${ref}: ${reason}`, { ref, reason, ...details });
}

/**
 * Build the dense inverse array (id -> index) from the dense values, detecting duplicates.
 * @param values - the id of every index, integers in [0, MAX_COUNT)
 * @param maxId - the largest id
 * @param onDuplicate - builds the error thrown for a repeated id
 * @returns a fresh inverse of maxId + 1 entries filled with INVALID_INDEX where absent
 */
function buildDenseInverse(
    values: U32,
    maxId: number,
    onDuplicate: (id: number, first: number, second: number) => GraphFormatError,
): U32 {
    const inverse = new Uint32Array(maxId + 1).fill(INVALID_INDEX);
    for (let i = 0; i < values.length; i++) {
        const id = values[i];
        if (inverse[id] !== INVALID_INDEX) {
            throw onDuplicate(id, inverse[id], i);
        }
        inverse[id] = i;
    }
    return inverse;
}

/**
 * Build the reverse map of `size` decoded ids, detecting duplicates.
 * @param ids - the ids in index order
 * @param size - how many entries to index
 * @param onDuplicate - builds the error thrown for a repeated id
 * @returns a fresh Map from id to index
 */
function buildReverseMap(
    ids: ArrayLike<NodeId>,
    size: number,
    onDuplicate: (id: NodeId, first: number, second: number) => GraphFormatError,
): Map<NodeId, number> {
    const map = new Map<NodeId, number>();
    for (let i = 0; i < size; i++) {
        const id = ids[i];
        const seen = map.get(id);
        if (seen !== undefined) {
            throw onDuplicate(id, seen, i);
        }
        map.set(id, i);
    }
    return map;
}

// ============================================================ detection

/** What one O(n) pass over the ids learns (design section 4.2). */
interface IdScan {
    /** Every id is a number. */
    readonly allNumbers: boolean;
    /** Every id is a string. */
    readonly allStrings: boolean;
    /** ids[i] === i + offset for every i, with an integer offset. */
    readonly identity: boolean;
    /** The identity offset (meaningful when identity). */
    readonly offset: number;
    /** Every id is an integer in [0, MAX_COUNT). */
    readonly allDenseIntegers: boolean;
    /** The largest id when allDenseIntegers, else -1. */
    readonly maxId: number;
}

/**
 * One pass over the first `size` ids that decides the storage kind. The ids are assumed valid
 * (finite numbers or strings).
 * @param ids - the ids in index order
 * @param size - how many to inspect
 * @returns the scan result
 */
function scanIds(ids: ArrayLike<NodeId>, size: number): IdScan {
    let allNumbers = true;
    let allStrings = true;
    let allDenseIntegers = true;
    let maxId = -1;
    let offset = 0;
    let identity = true;
    if (size > 0) {
        const first = ids[0];
        if (typeof first === "number" && Number.isSafeInteger(first)) {
            offset = first;
        } else {
            identity = false;
        }
    }
    for (let i = 0; i < size; i++) {
        const id = ids[i];
        if (typeof id === "number") {
            allStrings = false;
            if (identity && id !== i + offset) {
                identity = false;
            }
            if (allDenseIntegers) {
                if (Number.isInteger(id) && id >= 0 && id < MAX_COUNT) {
                    if (id > maxId) {
                        maxId = id;
                    }
                } else {
                    allDenseIntegers = false;
                }
            }
        } else {
            allNumbers = false;
            identity = false;
            allDenseIntegers = false;
        }
    }
    if (!allNumbers) {
        maxId = -1;
    }
    return { allNumbers, allStrings, identity, offset, allDenseIntegers, maxId };
}

/**
 * The kind a scan selects (design section 4.2): identity, else dense when the inverse array would be
 * at most twice the node count, else numeric, string or mixed.
 * @param scan - the scan result
 * @param size - the id count
 * @returns the kind
 */
function kindOf(scan: IdScan, size: number): NodeIdMapKind {
    if (size === 0 || scan.identity) {
        return "identity";
    }
    if (scan.allDenseIntegers && scan.maxId + 1 <= 2 * size) {
        return "dense";
    }
    if (scan.allNumbers) {
        return "numeric";
    }
    if (scan.allStrings) {
        return "string";
    }
    return "mixed";
}

/**
 * Copy the first `size` ids into a fresh U32 (dense values).
 * @param ids - the ids, all integers in [0, MAX_COUNT)
 * @param size - how many
 * @returns the values
 */
function denseValuesOf(ids: ArrayLike<NodeId>, size: number): U32 {
    const values = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
        const id = ids[i];
        values[i] = typeof id === "number" ? id : INVALID_INDEX;
    }
    return values;
}

// ============================================================ the class

/**
 * Bijection between node ids and node indices for one snapshot (design section 4, invariant I11):
 * `indexOf(idOf(i)) === i` for every i, `size === nodeCount`, SameValueZero equality (so `1` and
 * `"1"` are distinct, `-0` is `0`, `1.5` is legal and `NaN` never occurs). Immutable. Lookups by id
 * are total and return `INVALID_INDEX` on a miss (decision C9); `requireIndex` is the checked form.
 *
 * Instances are created by the module-level factories (`nodeIdMapFromIds`, `nodeIdMapFromF64`,
 * `identityNodeIdMap`, `nodeIdMapFromTyped`) and by `gatherNodeIdMap` / `remapNodeIdMap`; the
 * constructor takes a prepared storage record and is not meant to be called by consumers.
 */
export class NodeIdMap implements NodeIdMapContract {
    /** The storage kind (design section 4.2). */
    readonly kind: NodeIdMapKind;

    /** Number of ids; equals nodeCount. */
    readonly size: number;

    /** identity only: id === index + offset (0 or 1 in practice); 0 for every other kind. */
    readonly offset: number;

    /** The private storage; also registered in INTERNALS for the module-level functions. */
    private readonly storage: NodeIdMapStorage;

    /**
     * Wrap a prepared storage record. Sibling modules use the factories of this module instead.
     * @param storage - the storage; the kind's slots must be filled as documented on NodeIdMapStorage
     * @internal
     */
    constructor(storage: NodeIdMapStorage) {
        this.kind = storage.kind;
        this.size = storage.size;
        this.offset = storage.kind === "identity" ? storage.offset : 0;
        this.storage = storage;
        INTERNALS.set(this, storage);
        Object.freeze(this);
    }

    /**
     * The id of a node index.
     * @param index - the node index
     * @returns the id; E_INDEX_RANGE when index is not an integer in [0, size)
     */
    idOf(index: number): NodeId {
        if (!(index >= 0 && index < this.size) || !Number.isInteger(index)) {
            throw new GraphFormatError("E_INDEX_RANGE", `node index ${index} out of range (size ${this.size})`, {
                index,
                size: this.size,
            });
        }
        return this.idAt(index);
    }

    /**
     * Total lookup by id: SameValueZero, never coerces, so `indexOf("1")` misses a map of numbers.
     * @param id - the node id
     * @returns the node index, or INVALID_INDEX when absent
     */
    indexOf(id: NodeId): number {
        const { storage } = this;
        switch (storage.kind) {
            case "identity": {
                if (typeof id !== "number") {
                    return INVALID_INDEX;
                }
                const index = id - storage.offset;
                if (!Number.isInteger(index) || index < 0 || index >= storage.size) {
                    return INVALID_INDEX;
                }
                return index === 0 ? 0 : index;
            }
            case "dense": {
                const inverse = storage.denseInverse as U32;
                if (typeof id !== "number" || !Number.isInteger(id) || id < 0 || id >= inverse.length) {
                    return INVALID_INDEX;
                }
                return inverse[id];
            }
            case "numeric":
            case "string":
            case "mixed": {
                const index = this.reverseMap().get(id);
                return index === undefined || index >= storage.size ? INVALID_INDEX : index;
            }
            default:
                throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(storage.kind)}`, {
                    kind: storage.kind,
                });
        }
    }

    /**
     * Whether an id is present.
     * @param id - the node id
     * @returns true when present
     */
    has(id: NodeId): boolean {
        return this.indexOf(id) !== INVALID_INDEX;
    }

    /**
     * Checked lookup by id, for algorithms that today throw "node not found".
     * @param id - the node id
     * @returns the node index; E_UNKNOWN_NODE when absent
     */
    requireIndex(id: NodeId): number {
        const index = this.indexOf(id);
        if (index === INVALID_INDEX) {
            throw new GraphFormatError("E_UNKNOWN_NODE", `unknown node id ${describeId(id)}`, { id });
        }
        return index;
    }

    /**
     * Bulk lookup.
     * @param ids - the ids to resolve
     * @param onMissing - "invalid" (default) writes INVALID_INDEX for a miss; "throw" raises E_UNKNOWN_NODE
     * @returns a fresh U32 of indices in input order
     */
    indicesOf(ids: Iterable<NodeId>, onMissing: "invalid" | "throw" = "invalid"): U32 {
        assertOneOf("onMissing", onMissing, ["invalid", "throw"] as const);
        const lookup =
            onMissing === "throw"
                ? (id: NodeId): number => this.requireIndex(id)
                : (id: NodeId): number => this.indexOf(id);
        if (Array.isArray(ids)) {
            const list = ids as readonly NodeId[];
            const out = new Uint32Array(list.length);
            for (let i = 0; i < list.length; i++) {
                out[i] = lookup(list[i]);
            }
            return out;
        }
        let out = new Uint32Array(16);
        let count = 0;
        for (const id of ids) {
            if (count === out.length) {
                const grown = new Uint32Array(out.length * 2);
                grown.set(out);
                out = grown;
            }
            out[count++] = lookup(id);
        }
        return out.slice(0, count);
    }

    /**
     * Bulk decode of a range of ids in one pass. Bounds follow Array.prototype.slice (negative values
     * count from the end, out-of-range values are clamped).
     * @param start - first index (default 0)
     * @param end - one past the last index (default size)
     * @returns a fresh array of the ids in index order
     */
    idsSlice(start = 0, end = this.size): NodeId[] {
        const [from, to] = resolveRange(start, end, this.size);
        const { storage } = this;
        this.assertAttached();
        switch (storage.kind) {
            case "identity": {
                const out = new Array<NodeId>(to - from);
                for (let i = from; i < to; i++) {
                    out[i - from] = i + storage.offset;
                }
                return out;
            }
            case "dense":
                return Array.from((storage.denseValues as U32).subarray(from, to));
            case "numeric":
                if (storage.decoded !== null) {
                    return storage.decoded.slice(from, to);
                }
                return Array.from((storage.numbers as F64).subarray(from, to), normaliseZero);
            case "string":
                return (storage.strings as Utf8Store).slice(from, to);
            case "mixed": {
                if (storage.decoded !== null) {
                    return storage.decoded.slice(from, to);
                }
                const tags = storage.tags as U8;
                const numbers = storage.numbers as F64;
                const strings = storage.strings as Utf8Store;
                const out = new Array<NodeId>(to - from);
                for (let i = from; i < to; i++) {
                    out[i - from] = tags[i] === 0 ? normaliseZero(numbers[i]) : strings.at(i);
                }
                return out;
            }
            default:
                throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(storage.kind)}`, {
                    kind: storage.kind,
                });
        }
    }

    /**
     * Every id in index order.
     * @returns a fresh array
     */
    toArray(): NodeId[] {
        return this.idsSlice(0, this.size);
    }

    /**
     * Iterate the ids in index order.
     * @yields each id, decoding lazily
     */
    *[Symbol.iterator](): IterableIterator<NodeId> {
        for (let i = 0; i < this.size; i++) {
            yield this.idAt(i);
        }
    }

    /**
     * Key an index-aligned result vector by id (boundary helper, decision C11).
     * @param values - one value per node index
     * @returns a Map from id to value, in index order
     */
    toMap<T>(values: ArrayLike<T>): Map<NodeId, T> {
        const out = new Map<NodeId, T>();
        for (let i = 0; i < this.size; i++) {
            out.set(this.idAt(i), values[i]);
        }
        return out;
    }

    /**
     * Key an index-aligned result vector by String(id), for legacy Map<string, T> result shapes. When
     * two ids share a string form (1 and "1") the HIGHER index wins, exactly as the legacy
     * algorithms' assignment in node order does (design section 14.2).
     * @param values - one value per node index
     * @returns a Map from String(id) to value
     */
    toStringMap<T>(values: ArrayLike<T>): Map<string, T> {
        const out = new Map<string, T>();
        for (let i = 0; i < this.size; i++) {
            out.set(String(this.idAt(i)), values[i]);
        }
        return out;
    }

    /**
     * Key an index-aligned result vector by String(id), for legacy record result shapes only. The
     * record has a null prototype so ids such as "__proto__" are ordinary keys; when two ids share a
     * string form the HIGHER index wins (the legacy assignment order, design section 14.2).
     * @param values - one value per node index
     * @returns a record from String(id) to value
     */
    toRecord<T>(values: ArrayLike<T>): Record<string, T> {
        const out = Object.create(null) as Record<string, T>;
        for (let i = 0; i < this.size; i++) {
            out[String(this.idAt(i))] = values[i];
        }
        return out;
    }

    /**
     * Iterate [id, value] pairs of an index-aligned result vector.
     * @param values - one value per node index
     * @yields the pairs in index order
     */
    *entries<T>(values: ArrayLike<T>): IterableIterator<[NodeId, T]> {
        for (let i = 0; i < this.size; i++) {
            yield [this.idAt(i), values[i]];
        }
    }

    /**
     * String(idOf(i)) -> i, built lazily once, for legacy string-typed id parameters. When two ids
     * share a string form (always one number and one string, since ids are distinct) the STRING id
     * wins: a legacy string parameter "1" addressed the node whose id is the string "1".
     * @returns the read-only string index
     */
    stringIndex(): ReadonlyMap<string, number> {
        const { storage } = this;
        if (storage.stringIndex === null) {
            const index = new Map<string, number>();
            for (let i = 0; i < storage.size; i++) {
                const id = this.idAt(i);
                const key = String(id);
                if (typeof id === "string" || !index.has(key)) {
                    index.set(key, i);
                }
            }
            storage.stringIndex = index;
        }
        return storage.stringIndex;
    }

    /**
     * Bytes of typed storage currently materialised; excludes the reverse Map and JS strings.
     * @returns the byte count
     */
    byteLength(): number {
        const { storage } = this;
        let bytes = 0;
        if (storage.denseValues !== null) {
            bytes += storage.denseValues.byteLength;
        }
        if (storage.denseInverse !== null) {
            bytes += storage.denseInverse.byteLength;
        }
        if (storage.numbers !== null) {
            bytes += storage.numbers.byteLength;
        }
        if (storage.tags !== null) {
            bytes += storage.tags.byteLength;
        }
        if (storage.strings !== null) {
            bytes += storage.strings.byteLength();
        }
        return bytes;
    }

    /**
     * Unchecked id of an in-range index.
     * @param index - a node index in [0, size)
     * @returns the id
     */
    private idAt(index: number): NodeId {
        const { storage } = this;
        this.assertAttached();
        switch (storage.kind) {
            case "identity":
                return index + storage.offset;
            case "dense":
                return (storage.denseValues as U32)[index];
            case "numeric":
                return storage.decoded !== null
                    ? storage.decoded[index]
                    : normaliseZero((storage.numbers as F64)[index]);
            case "string":
                return (storage.strings as Utf8Store).at(index);
            case "mixed":
                if (storage.decoded !== null) {
                    return storage.decoded[index];
                }
                return (storage.tags as U8)[index] === 0
                    ? normaliseZero((storage.numbers as F64)[index])
                    : (storage.strings as Utf8Store).at(index);
            default:
                throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(storage.kind)}`, {
                    kind: storage.kind,
                });
        }
    }

    /**
     * E_DETACHED when a typed array the map reads from was transferred away (design sections 9.1
     * and 11.3): derived from the array state (a transferred buffer leaves zero-length views).
     */
    private assertAttached(): void {
        if (storageDetached(this.storage)) {
            throw new GraphFormatError("E_DETACHED", "the id map's storage was transferred away", {
                kind: this.storage.kind,
            });
        }
    }

    /**
     * The reverse map of a numeric / string / mixed map, built on first use (one pass over the ids,
     * after a bulk decode of a wire-decoded Utf8 store) when it was not shared by the builder.
     * @returns the map from id to index
     */
    private reverseMap(): Map<NodeId, number> {
        const { storage } = this;
        if (storage.map === null) {
            // the bulk decode below reads the Utf8 store: E_DETACHED, never a TypeError, after a transfer
            this.assertAttached();
            if (storage.decoded === null && storage.strings !== null) {
                storage.strings.materialiseDecoded();
            }
            const map = new Map<NodeId, number>();
            for (let i = 0; i < storage.size; i++) {
                map.set(this.idAt(i), i);
            }
            storage.map = map;
        }
        return storage.map;
    }
}

// ============================================================ factories

/**
 * An identity map: id === index + offset, no storage at all (design section 4.2). What
 * `fromEdgeArrays` without ids and `contract()` produce; offset 1 covers 1-based files.
 * @param size - the node count
 * @param offset - the integer added to every index; default 0
 * @returns the map
 */
export function identityNodeIdMap(size: number, offset = 0): NodeIdMap {
    if (!Number.isInteger(size) || size < 0 || size > MAX_COUNT) {
        throw new GraphFormatError("E_TOO_LARGE", `node count ${size} is not an integer in [0, MAX_COUNT]`, {
            count: size,
            max: MAX_COUNT,
        });
    }
    if (!Number.isSafeInteger(offset)) {
        throw new GraphFormatError("E_INVALID_ID", `identity offset ${offset} is not a safe integer`, {
            offset,
            reason: "non-integer offset",
        });
    }
    return new NodeIdMap(makeStorage("identity", size, { offset }));
}

/** Options of nodeIdMapFromIds. */
interface FromIdsOptions {
    /**
     * The builder's reverse map, shared by reference and guarded by `index < size` (decision C17). When
     * given, the ids are trusted (already validated and distinct) and `validate` is ignored.
     */
    readonly map?: Map<NodeId, number> | undefined;
    /**
     * Check every id (E_INVALID_ID) and their distinctness (E_DUPLICATE_ID), copying the array and
     * normalising -0; default true. false trusts the caller (derived graphs) and shares the array.
     */
    readonly validate?: boolean | undefined;
}

/**
 * Build a map from decoded ids in index order, choosing the kind in one O(n) pass (design section
 * 4.2, freeze step 9). With `options.map` (the builder's Map) the array and the Map are shared by
 * reference and nothing is validated; identity and dense maps drop the Map. Without it the ids are
 * validated and copied unless `validate` is false.
 * @param ids - the ids; only entries below `size` are read
 * @param size - the node count; default ids.length
 * @param options - the shared map and validation switch
 * @returns the map
 */
export function nodeIdMapFromIds(
    ids: readonly NodeId[],
    size: number = ids.length,
    options: FromIdsOptions = {},
): NodeIdMap {
    if (!Number.isInteger(size) || size < 0 || size > MAX_COUNT) {
        throw new GraphFormatError("E_TOO_LARGE", `node count ${size} is not an integer in [0, MAX_COUNT]`, {
            count: size,
            max: MAX_COUNT,
        });
    }
    const trusted = options.map !== undefined || options.validate === false;
    let source: readonly NodeId[] = ids;
    if (!trusted) {
        const copy = new Array<NodeId>(size);
        for (let i = 0; i < size; i++) {
            copy[i] = validateNodeId(ids[i]);
        }
        source = copy;
    }
    const scan = scanIds(source, size);
    const kind = kindOf(scan, size);
    switch (kind) {
        case "identity":
            return new NodeIdMap(makeStorage("identity", size, { offset: size === 0 ? 0 : scan.offset }));
        case "dense": {
            const denseValues = denseValuesOf(source, size);
            const denseInverse = buildDenseInverse(denseValues, scan.maxId, duplicateIdError);
            return new NodeIdMap(makeStorage("dense", size, { denseValues, denseInverse }));
        }
        case "numeric":
        case "mixed": {
            const map = options.map ?? (trusted ? null : buildReverseMap(source, size, duplicateIdError));
            return new NodeIdMap(makeStorage(kind, size, { decoded: source, map }));
        }
        case "string": {
            const map = options.map ?? (trusted ? null : buildReverseMap(source, size, duplicateIdError));
            const strings = Utf8Store.fromStrings(source as readonly string[], size);
            return new NodeIdMap(makeStorage("string", size, { strings, map }));
        }
        default:
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(kind)}`, { kind });
    }
}

/** Options of nodeIdMapFromF64. */
interface FromF64Options {
    /**
     * Check that every value is finite (E_INVALID_ID) and that the values are distinct (E_DUPLICATE_ID);
     * default true. The array is adopted by reference either way; -0 entries are rewritten to 0 in a
     * copy when validating.
     */
    readonly validate?: boolean | undefined;
}

/**
 * Build a map from numeric ids in a Float64Array (the `ids: F64` form of `fromEdgeArrays` /
 * `fromCsr`), choosing identity, dense or numeric in one pass. A numeric map adopts the array as its
 * typed representation.
 * @param values - one id per index
 * @param options - the validation switch
 * @returns the map
 */
export function nodeIdMapFromF64(values: F64, options: FromF64Options = {}): NodeIdMap {
    const size = values.length;
    if (size > MAX_COUNT) {
        throw new GraphFormatError("E_TOO_LARGE", `node count ${size} exceeds MAX_COUNT`, {
            count: size,
            max: MAX_COUNT,
        });
    }
    const validate = options.validate !== false;
    let numbers = values;
    if (validate) {
        let negativeZero = false;
        for (let i = 0; i < size; i++) {
            const v = values[i];
            if (!Number.isFinite(v)) {
                throw new GraphFormatError(
                    "E_INVALID_ID",
                    `invalid node id ${String(v)} at index ${i}: not a finite number`,
                    {
                        id: v,
                        index: i,
                        reason: "non-finite",
                    },
                );
            }
            if (Object.is(v, -0)) {
                negativeZero = true;
            }
        }
        if (negativeZero) {
            numbers = values.slice();
            for (let i = 0; i < size; i++) {
                if (numbers[i] === 0) {
                    numbers[i] = 0;
                }
            }
        }
    }
    const scan = scanIds(numbers, size);
    const kind = kindOf(scan, size);
    switch (kind) {
        case "identity":
            return new NodeIdMap(makeStorage("identity", size, { offset: size === 0 ? 0 : scan.offset }));
        case "dense": {
            const denseValues = denseValuesOf(numbers, size);
            const denseInverse = buildDenseInverse(denseValues, scan.maxId, duplicateIdError);
            return new NodeIdMap(makeStorage("dense", size, { denseValues, denseInverse }));
        }
        case "numeric": {
            const map = validate ? buildReverseMap(numbers, size, duplicateIdError) : null;
            return new NodeIdMap(makeStorage("numeric", size, { numbers, map }));
        }
        default:
            throw new GraphFormatError("E_UNSUPPORTED", `unexpected id map kind ${kind} for numeric ids`, { kind });
    }
}

/**
 * The distinct backing buffers of an id map's materialised typed storage (dense values and inverse,
 * numbers, tags, an encoded Utf8 store), for the owner count of design section 9.1.
 * @param map - the id map
 * @returns the buffers, each once
 */
export function idMapBuffers(map: NodeIdMap): ArrayBuffer[] {
    const storage = INTERNALS.get(map);
    const out = new Set<ArrayBuffer>();
    if (storage === undefined) {
        // a foreign object standing in for a map (tests wrap one): nothing typed to claim
        return [];
    }
    for (const view of [storage.denseValues, storage.denseInverse, storage.numbers, storage.tags]) {
        if (view !== null) {
            out.add(view.buffer);
        }
    }
    if (storage.strings !== null && storage.strings.encoded) {
        out.add(storage.strings.offsets.buffer);
        out.add(storage.strings.utf8.buffer);
    }
    return [...out];
}

/**
 * Build a map from its typed (wire) representation (design sections 4.5 and 9.5), adopting the arrays
 * by reference. "structure" checks lengths, tag values, the dense bound and the Utf8 layout
 * (E_BAD_SERIALIZATION with details.ref); "full" additionally checks invariant I11: finite numbers,
 * well-formed UTF-8 (a fatal decode) and distinct ids (E_INVALID_SNAPSHOT with details.invariant
 * "I11"), building the reverse map eagerly. A dense duplicate is always detected because the inverse
 * array is rebuilt on load. An unknown kind is E_UNSUPPORTED with details.kind.
 * @param parts - the typed representation
 * @param validate - the validation level; default "structure"
 * @returns the map
 */
export function nodeIdMapFromTyped(parts: NodeIdMapTypedParts, validate: ValidationLevel = "structure"): NodeIdMap {
    const { kind, size } = parts;
    const structure = validate !== "none";
    const full = validate === "full";
    if (structure && (!Number.isInteger(size) || size < 0 || size > MAX_COUNT)) {
        throw layoutError("ids.size", `size ${size} is not an integer in [0, MAX_COUNT]`, { found: size });
    }
    switch (kind) {
        case "identity": {
            // O(1) at every level: every id index + offset (the largest is offset + size - 1) must be a
            // safe integer (I11)
            if (!Number.isSafeInteger(parts.offset) || (size > 0 && !Number.isSafeInteger(parts.offset + size - 1))) {
                const reason = `offset ${parts.offset} with ${size} ids leaves the safe-integer range`;
                throw full
                    ? bijectionError(reason, { found: parts.offset, size })
                    : layoutError("ids.offset", reason, { found: parts.offset, size });
            }
            return new NodeIdMap(makeStorage("identity", size, { offset: normaliseZero(parts.offset) }));
        }
        case "dense": {
            const { values } = parts;
            if (!(values instanceof Uint32Array)) {
                throw layoutError("ids.values", "a dense id map needs u32 values");
            }
            if (structure && values.length !== size) {
                throw layoutError("ids.values", `expected ${size} values, found ${values.length}`, {
                    expected: size,
                    found: values.length,
                });
            }
            let maxId = -1;
            for (let i = 0; i < values.length; i++) {
                if (values[i] > maxId) {
                    maxId = values[i];
                }
            }
            // the dense bounds are checked at every level: the scan is already paid, and the inverse
            // array below is allocated over [0, maxId], which an unchecked id could make enormous
            if (maxId >= MAX_COUNT) {
                throw layoutError("ids.values", `dense id ${maxId} is not below MAX_COUNT`, { found: maxId });
            }
            if (maxId + 1 > 2 * Math.max(size, values.length)) {
                throw layoutError("ids.values", `dense bound violated: max id ${maxId} with ${size} nodes`, {
                    maxId,
                    size,
                });
            }
            const denseInverse = buildDenseInverse(values, maxId, (id, first, second) =>
                bijectionError(`duplicate id ${id}`, { id, indices: [first, second] }),
            );
            return new NodeIdMap(makeStorage("dense", size, { denseValues: values, denseInverse }));
        }
        case "numeric": {
            const numbers = parts.values;
            if (!(numbers instanceof Float64Array)) {
                throw layoutError("ids.values", "a numeric id map needs f64 values");
            }
            if (structure && numbers.length !== size) {
                throw layoutError("ids.values", `expected ${size} values, found ${numbers.length}`, {
                    expected: size,
                    found: numbers.length,
                });
            }
            let map: Map<NodeId, number> | null = null;
            if (full) {
                for (let i = 0; i < size; i++) {
                    if (!Number.isFinite(numbers[i])) {
                        throw bijectionError(`non-finite id at index ${i}`, { index: i, id: numbers[i] });
                    }
                }
                map = buildReverseMap(numbers, size, (id, first, second) =>
                    bijectionError(`duplicate id ${describeId(id)}`, { id, indices: [first, second] }),
                );
            }
            return new NodeIdMap(makeStorage("numeric", size, { numbers, map }));
        }
        case "string": {
            const { offsets, utf8 } = parts;
            if (offsets === null || utf8 === null) {
                throw layoutError("ids.offsets", "a string id map needs offsets and utf8");
            }
            if (structure) {
                checkUtf8Layout(offsets, utf8, size, "ids.offsets");
            }
            const strings = Utf8Store.fromEncoded(offsets, utf8);
            let map: Map<NodeId, number> | null = null;
            if (full) {
                const bad = strings.firstMalformedRow();
                if (bad !== -1) {
                    throw bijectionError(`malformed UTF-8 in id at index ${bad}`, { index: bad });
                }
                map = buildReverseMap(strings.decodeAll(), size, (id, first, second) =>
                    bijectionError(`duplicate id ${describeId(id)}`, { id, indices: [first, second] }),
                );
            }
            return new NodeIdMap(makeStorage("string", size, { strings, map }));
        }
        case "mixed": {
            const { tags, numbers, offsets, utf8 } = parts;
            if (tags === null || numbers === null || offsets === null || utf8 === null) {
                throw layoutError("ids.tags", "a mixed id map needs tags, numbers, offsets and utf8");
            }
            if (structure) {
                if (tags.length !== size) {
                    throw layoutError("ids.tags", `expected ${size} tags, found ${tags.length}`, {
                        expected: size,
                        found: tags.length,
                    });
                }
                if (numbers.length !== size) {
                    throw layoutError("ids.numbers", `expected ${size} numbers, found ${numbers.length}`, {
                        expected: size,
                        found: numbers.length,
                    });
                }
                for (let i = 0; i < size; i++) {
                    if (tags[i] > 1) {
                        throw layoutError("ids.tags", `tag ${tags[i]} at index ${i} is not 0 or 1`, {
                            index: i,
                            found: tags[i],
                        });
                    }
                }
                checkUtf8Layout(offsets, utf8, size, "ids.offsets");
            }
            const strings = Utf8Store.fromEncoded(offsets, utf8);
            let map: Map<NodeId, number> | null = null;
            if (full) {
                const bad = strings.firstMalformedRow();
                if (bad !== -1) {
                    throw bijectionError(`malformed UTF-8 in id at index ${bad}`, { index: bad });
                }
                const decoded = new Array<NodeId>(size);
                for (let i = 0; i < size; i++) {
                    if (tags[i] === 0) {
                        if (!Number.isFinite(numbers[i])) {
                            throw bijectionError(`non-finite id at index ${i}`, { index: i, id: numbers[i] });
                        }
                        decoded[i] = numbers[i];
                    } else {
                        decoded[i] = strings.at(i);
                    }
                }
                map = buildReverseMap(decoded, size, (id, first, second) =>
                    bijectionError(`duplicate id ${describeId(id)}`, { id, indices: [first, second] }),
                );
            }
            return new NodeIdMap(makeStorage("mixed", size, { tags, numbers, strings, map }));
        }
        default:
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(kind)}`, { kind });
    }
}

// ============================================================ typed form, gather, remap

/**
 * The typed (wire) representation of a map (design section 4.5), materialising and caching whatever
 * the map only held in decoded form: the F64 of a numeric map, the Utf8 store of a string map, the
 * tags + numbers + Utf8 store of a mixed map. The dense inverse and the reverse Map are never part
 * of it. Repeated calls return the same arrays.
 * @param map - the id map
 * @returns the typed parts; every slot the kind does not use is null
 */
export function nodeIdMapToTyped(map: NodeIdMap): NodeIdMapTypedParts {
    const storage = storageOf(map);
    const { kind, size } = storage;
    switch (kind) {
        case "identity":
            return {
                kind,
                size,
                offset: storage.offset,
                values: null,
                tags: null,
                numbers: null,
                offsets: null,
                utf8: null,
            };
        case "dense":
            return {
                kind,
                size,
                offset: 0,
                values: storage.denseValues,
                tags: null,
                numbers: null,
                offsets: null,
                utf8: null,
            };
        case "numeric": {
            if (storage.numbers === null) {
                const decoded = storage.decoded as readonly NodeId[];
                const numbers = new Float64Array(size);
                for (let i = 0; i < size; i++) {
                    const id = decoded[i];
                    if (typeof id !== "number") {
                        throw new GraphFormatError(
                            "E_INVALID_ID",
                            `numeric id map holds a ${typeof id} at index ${i}`,
                            {
                                index: i,
                                found: typeof id,
                            },
                        );
                    }
                    numbers[i] = id;
                }
                storage.numbers = numbers;
                claimHolder(numbers.buffer);
            }
            return {
                kind,
                size,
                offset: 0,
                values: storage.numbers,
                tags: null,
                numbers: null,
                offsets: null,
                utf8: null,
            };
        }
        case "string": {
            const strings = storage.strings as Utf8Store;
            return {
                kind,
                size,
                offset: 0,
                values: null,
                tags: null,
                numbers: null,
                offsets: strings.offsets,
                utf8: strings.utf8,
            };
        }
        case "mixed": {
            if (storage.tags === null || storage.numbers === null || storage.strings === null) {
                const decoded = storage.decoded as readonly NodeId[];
                const tags = new Uint8Array(size);
                const numbers = new Float64Array(size);
                for (let i = 0; i < size; i++) {
                    const id = decoded[i];
                    if (typeof id === "number") {
                        numbers[i] = id;
                    } else {
                        tags[i] = 1;
                    }
                }
                const { offsets, utf8 } = encodeUtf8Rows(decoded, size);
                storage.tags = tags;
                storage.numbers = numbers;
                storage.strings = Utf8Store.fromEncoded(offsets, utf8);
                // the map is the first holder of the buffers it materialises (design section 9.1)
                for (const buffer of [tags.buffer, numbers.buffer, offsets.buffer, utf8.buffer]) {
                    claimHolder(buffer);
                }
            }
            return {
                kind,
                size,
                offset: 0,
                values: null,
                tags: storage.tags,
                numbers: storage.numbers,
                offsets: storage.strings.offsets,
                utf8: storage.strings.utf8,
            };
        }
        default:
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${String(kind)}`, { kind });
    }
}

/**
 * The map of a gathered node space: `out.idOf(i) === map.idOf(indexMap[i])` (design section 7.3:
 * inducedSubgraph, relabel). The kind is re-detected from the gathered ids, so an identity map
 * gathered through a prefix stays identity and otherwise becomes dense or numeric per the rules of
 * design section 4.2. The caller guarantees indexMap has no repeats (a permutation or a selection),
 * so the result is a bijection without re-validation.
 * @param map - the source map
 * @param indexMap - new index -> source index; E_INDEX_RANGE when an entry is out of range
 * @returns the gathered map
 */
export function gatherNodeIdMap(map: NodeIdMap, indexMap: U32): NodeIdMap {
    const n = indexMap.length;
    const ids = new Array<NodeId>(n);
    for (let i = 0; i < n; i++) {
        ids[i] = map.idOf(indexMap[i]);
    }
    return nodeIdMapFromIds(ids, n, { validate: false });
}

/**
 * The map of a compacted node space: `out.idOf(remap[old]) === map.idOf(old)` for every old index
 * whose entry is not INVALID_INDEX (design sections 4.4 and 6.3 step 1). Every new index must be hit
 * exactly once.
 * @param map - the source map
 * @param remap - old index -> new index or INVALID_INDEX; length === map.size
 * @param newSize - the compacted node count
 * @returns the remapped map; E_INDEX_RANGE when remap is not a surjection onto [0, newSize)
 */
export function remapNodeIdMap(map: NodeIdMap, remap: U32, newSize: number): NodeIdMap {
    if (remap.length !== map.size) {
        throw new GraphFormatError("E_INDEX_RANGE", `remap has ${remap.length} entries for ${map.size} ids`, {
            expected: map.size,
            found: remap.length,
        });
    }
    const ids = new Array<NodeId>(newSize);
    const seen = new Uint8Array(newSize);
    let filled = 0;
    for (let old = 0; old < remap.length; old++) {
        const target = remap[old];
        if (target === INVALID_INDEX) {
            continue;
        }
        if (target >= newSize || seen[target] === 1) {
            throw new GraphFormatError("E_INDEX_RANGE", `remap entry ${target} at ${old} is out of range or repeated`, {
                index: target,
                size: newSize,
            });
        }
        seen[target] = 1;
        ids[target] = map.idOf(old);
        filled++;
    }
    if (filled !== newSize) {
        throw new GraphFormatError("E_INDEX_RANGE", `remap fills ${filled} of ${newSize} new indices`, {
            filled,
            size: newSize,
        });
    }
    return nodeIdMapFromIds(ids, newSize, { validate: false });
}
