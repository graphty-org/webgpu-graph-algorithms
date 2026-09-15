/**
 * Content hashing and checksums of @graphty/graph-format (design sections 5.8 and 9.3).
 *
 * `contentHash()` is a 64-bit hash of the core arrays computed as two independent 32-bit FNV-1a-style
 * lanes over u32 words with `Math.imul`, returned as 16 hex characters: the key a cache uses for a
 * snapshot's topology. The same hasher produces the per-array checksums that `freeze({ checksum:
 * true })` records and `validate({ checksum: true })` compares, so a consumer that wrote into a frozen
 * array or a shared view is caught by the tests that opt in (invariant I17).
 *
 * The hash is a pure function of the topology: it covers `directed`, the four counts, `rowPtr`,
 * `colIdx`, `weights` (as raw f32 bit patterns), `arcToEdge` and `edgeToArc`. An identity permutation
 * is hashed as the sequence 0..len-1 WITHOUT materialising it, so two snapshots with the same
 * topology hash identically whether or not their getters were touched (P11). Nothing here depends on
 * the arena, the id map or the columns.
 */

import { isColumnDetached } from "../columns/column.js";
import { GraphFormatError } from "../errors.js";
import { type Column, type GraphSnapshot } from "../types/index.js";
import { unsupportedDtype } from "../util/typed-array.js";

/** Lane A: the 32-bit FNV-1a offset basis. */
const LANE_A_BASIS = 0x811c9dc5;
/** Lane A: the 32-bit FNV prime. */
const LANE_A_PRIME = 0x01000193;
/** Lane B: the high word of the 64-bit FNV offset basis (0xcbf29ce484222325), as an independent seed. */
const LANE_B_BASIS = 0xcbf29ce4;
/** Lane B: an odd multiplier unrelated to the FNV prime (the MurmurHash2 constant), so the lanes decorrelate. */
const LANE_B_MULTIPLIER = 0x5bd1e995;

/**
 * Two-lane FNV-1a-style hasher over u32 words (design section 9.3). Lane A is standard 32-bit FNV-1a
 * applied word-wise (`h = imul(h ^ w, 0x01000193)`); lane B uses a different basis and multiplier
 * (`h = imul(h ^ w, 0x5bd1e995)`) and additionally rotates the word by 16 bits before mixing so the
 * two lanes never agree on which bits they weight. `digest()` renders both lanes as 8 zero-padded hex
 * characters each, lane A first.
 */
export class Fnv1aHasher {
    private laneA: number;
    private laneB: number;

    /** Start a fresh hasher at the two offset bases. */
    constructor() {
        this.laneA = LANE_A_BASIS;
        this.laneB = LANE_B_BASIS;
    }

    /**
     * Mix one u32 word (any number is coerced with `>>> 0`).
     * @param word - the word to mix
     * @returns this hasher
     */
    word(word: number): this {
        const w = word >>> 0;
        this.laneA = Math.imul(this.laneA ^ w, LANE_A_PRIME) >>> 0;
        const rotated = ((w << 16) | (w >>> 16)) >>> 0;
        this.laneB = Math.imul(this.laneB ^ rotated, LANE_B_MULTIPLIER) >>> 0;
        return this;
    }

    /**
     * Mix a length prefix followed by every word of a u32 array, so arrays of different lengths with a
     * common prefix hash differently and consecutive arrays cannot be re-split.
     * @param words - the words to mix
     * @returns this hasher
     */
    words(words: Uint32Array): this {
        this.word(words.length);
        for (let i = 0; i < words.length; i++) {
            this.word(words[i]);
        }
        return this;
    }

    /**
     * Mix the sequence 0..length-1 (an identity permutation) with the same framing as `words()`, without
     * allocating it.
     * @param length - the permutation length
     * @returns this hasher
     */
    identity(length: number): this {
        this.word(length);
        for (let i = 0; i < length; i++) {
            this.word(i);
        }
        return this;
    }

    /**
     * Mix a byte length prefix followed by the bytes of any ArrayBufferView, four bytes per word
     * (little-endian packing, the last word zero-padded). A view whose byte range is 4-byte aligned
     * is read as u32 words directly.
     * @param view - the bytes to mix
     * @returns this hasher
     */
    bytes(view: ArrayBufferView): this {
        const { byteLength, byteOffset } = view;
        this.word(byteLength);
        if (byteOffset % 4 === 0 && byteLength % 4 === 0) {
            const words = new Uint32Array(view.buffer, byteOffset, byteLength / 4);
            for (let i = 0; i < words.length; i++) {
                this.word(words[i]);
            }
            return this;
        }
        const bytes = new Uint8Array(view.buffer, byteOffset, byteLength);
        let i = 0;
        for (; i + 4 <= byteLength; i += 4) {
            this.word(bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16) | (bytes[i + 3] << 24));
        }
        if (i < byteLength) {
            let tail = 0;
            for (let k = 0; i + k < byteLength; k++) {
                tail |= bytes[i + k] << (8 * k);
            }
            this.word(tail);
        }
        return this;
    }

    /**
     * Mix a JS string as its length followed by its UTF-16 code units.
     * @param text - the string to mix
     * @returns this hasher
     */
    text(text: string): this {
        this.word(text.length);
        for (let i = 0; i < text.length; i++) {
            this.word(text.charCodeAt(i));
        }
        return this;
    }

    /**
     * The 16-character lowercase hex digest: lane A then lane B, each 8 zero-padded characters.
     * @returns the digest
     */
    digest(): string {
        return this.laneA.toString(16).padStart(8, "0") + this.laneB.toString(16).padStart(8, "0");
    }
}

/**
 * The digest of one typed array's bytes (a core array, a view array, a column buffer).
 * @param view - the array to hash
 * @returns the 16-character hex digest
 */
export function hashTypedArray(view: ArrayBufferView): string {
    return new Fnv1aHasher().bytes(view).digest();
}

/**
 * Mix everything that determines a column's contents: dtype, length, null count, validity words,
 * and per dtype the data / codes / dictionary / offsets / utf8 / child / json text. Used for the
 * immutable-column checksums of design section 5.8.
 * @param hasher - the hasher to mix into
 * @param column - the column
 */
function mixColumn(hasher: Fnv1aHasher, column: Column): void {
    hasher.text(column.dtype).word(column.length).word(column.nullCount);
    if (column.validity !== null) {
        hasher.words(column.validity);
    } else {
        hasher.word(0);
    }
    const { dtype } = column;
    switch (dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8":
        case "bool":
            hasher.bytes(column.data);
            break;
        case "dict":
            hasher.words(column.codes).word(column.dictionary.length);
            for (const entry of column.dictionary) {
                hasher.text(entry);
            }
            break;
        case "string":
            hasher.words(column.offsets).bytes(column.utf8);
            break;
        case "list":
            hasher.words(column.offsets);
            mixColumn(hasher, column.child);
            break;
        case "json":
            hasher.text(JSON.stringify(column.values));
            break;
        default:
            unsupportedDtype(dtype);
    }
}

/**
 * The digest of a column's contents (data, validity, dictionary, strings, child, json text).
 * @param column - the column to hash
 * @returns the 16-character hex digest
 */
export function hashColumn(column: Column): string {
    if (isColumnDetached(column)) {
        throw new GraphFormatError("E_DETACHED", `column "${column.meta.name}" was transferred away`, {
            column: column.meta.name,
        });
    }
    const hasher = new Fnv1aHasher();
    mixColumn(hasher, column);
    return hasher.digest();
}

/**
 * The content hash of a snapshot's topology (design section 9.3): `directed`, the four counts, then
 * `rowPtr`, `colIdx`, `weights` (raw f32 bit patterns; an absent array hashes as length 0),
 * `arcToEdge` and `edgeToArc`, with an identity permutation hashed as 0..len-1 without being
 * materialised (P11: the digest does not depend on which getters were touched).
 * @param snapshot - the snapshot to hash
 * @returns the 16-character hex digest
 */
export function contentHashOf(snapshot: GraphSnapshot): string {
    const hasher = new Fnv1aHasher();
    hasher
        .word(snapshot.directed ? 1 : 0)
        .word(snapshot.nodeCount)
        .word(snapshot.edgeCount)
        .word(snapshot.arcCount)
        .word(snapshot.selfLoopCount);
    hasher.words(snapshot.rowPtr).words(snapshot.colIdx);
    if (snapshot.weights === null) {
        hasher.word(0);
    } else {
        hasher.bytes(snapshot.weights);
    }
    if (snapshot.flags.arcToEdgeIsIdentity) {
        hasher.identity(snapshot.arcCount).identity(snapshot.edgeCount);
    } else {
        hasher.words(snapshot.arcToEdge).words(snapshot.edgeToArc);
    }
    return hasher.digest();
}
