/**
 * The Arrow Utf8 string store (design sections 4.2 and 5.1): `Uint32Array(rows + 1)` offsets plus one
 * `Uint8Array` of concatenated UTF-8 bytes, kept next to a decoded JS-string representation. The two
 * are caches of one logical value and are materialised lazily in either direction:
 *
 * - a store built from strings (`Utf8Store.fromStrings`, what the builder has) shares the source
 *   array by reference and encodes on the first read of `offsets` / `utf8` (toWire / toBytes /
 *   transferables), caching the result;
 * - a store built from a wire buffer (`Utf8Store.fromEncoded`) decodes per row on `at()`, cached in
 *   a sparse array, and in bulk on `slice()` / `decodeAll()`, which are not cached.
 *
 * Nothing is encoded or decoded at construction. Row `i` occupies the byte range
 * `[offsets[i], offsets[i + 1])`; `offsets[0]` need not be zero so that a zero-copy slice of a
 * string column can share its parent's arrays. The encoder is a plain JS UTF-8 encoder that
 * produces exactly the bytes `TextEncoder` would (a lone surrogate becomes U+FFFD); the decoder is
 * `TextDecoder` behind an ASCII fast path.
 */

import { GraphFormatError } from "../errors.js";
import { type U8, type U32 } from "../types/index.js";
import { claimHolder } from "../util/shared-buffers.js";

/**
 * Rows of at most this many bytes are decoded by a char-code loop when they are pure ASCII, which is
 * about three times faster than TextDecoder for short ids; longer rows go straight to TextDecoder.
 */
const ASCII_FAST_PATH_MAX_BYTES = 32;

/** The instance type of the host's TextDecoder (a global value in Node and in browsers; typed through its constructor). */
type Decoder = InstanceType<typeof TextDecoder>;

let sharedDecoder: Decoder | null = null;
let fatalDecoder: Decoder | null = null;

/**
 * The lazily created shared (replacing) decoder.
 * @returns a TextDecoder that replaces malformed sequences with U+FFFD
 */
function replacingDecoder(): Decoder {
    sharedDecoder ??= new TextDecoder("utf-8");
    return sharedDecoder;
}

/**
 * The lazily created fatal decoder used by validation.
 * @returns a TextDecoder that throws on a malformed sequence
 */
function strictDecoder(): Decoder {
    fatalDecoder ??= new TextDecoder("utf-8", { fatal: true });
    return fatalDecoder;
}

/**
 * Number of UTF-8 bytes `encodeUtf8Into` writes for a string (a lone surrogate counts as the three
 * bytes of U+FFFD, matching TextEncoder).
 * @param s - the string
 * @returns the encoded byte length
 */
export function utf8ByteLength(s: string): number {
    let bytes = 0;
    const n = s.length;
    for (let i = 0; i < n; i++) {
        const c = s.charCodeAt(i);
        if (c < 0x80) {
            bytes += 1;
        } else if (c < 0x800) {
            bytes += 2;
        } else if (c >= 0xd800 && c <= 0xdbff && i + 1 < n) {
            const d = s.charCodeAt(i + 1);
            if (d >= 0xdc00 && d <= 0xdfff) {
                bytes += 4;
                i++;
            } else {
                bytes += 3;
            }
        } else {
            bytes += 3;
        }
    }
    return bytes;
}

/**
 * Encode a string as UTF-8 into `out` starting at `position`; the caller has sized `out` with
 * `utf8ByteLength`. Produces the bytes TextEncoder would, including U+FFFD for a lone surrogate.
 * @param s - the string
 * @param out - the destination bytes
 * @param position - the first byte to write
 * @returns one past the last byte written
 */
export function encodeUtf8Into(s: string, out: Uint8Array, position: number): number {
    let p = position;
    const n = s.length;
    for (let i = 0; i < n; i++) {
        let c = s.charCodeAt(i);
        if (c < 0x80) {
            out[p++] = c;
            continue;
        }
        if (c < 0x800) {
            out[p++] = 0xc0 | (c >> 6);
            out[p++] = 0x80 | (c & 0x3f);
            continue;
        }
        if (c >= 0xd800 && c <= 0xdfff) {
            if (c <= 0xdbff && i + 1 < n) {
                const d = s.charCodeAt(i + 1);
                if (d >= 0xdc00 && d <= 0xdfff) {
                    c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00);
                    i++;
                    out[p++] = 0xf0 | (c >> 18);
                    out[p++] = 0x80 | ((c >> 12) & 0x3f);
                    out[p++] = 0x80 | ((c >> 6) & 0x3f);
                    out[p++] = 0x80 | (c & 0x3f);
                    continue;
                }
            }
            c = 0xfffd;
        }
        out[p++] = 0xe0 | (c >> 12);
        out[p++] = 0x80 | ((c >> 6) & 0x3f);
        out[p++] = 0x80 | (c & 0x3f);
    }
    return p;
}

/**
 * The length from which V8 represents a substring as a SlicedString that keeps its parent alive
 * (`SlicedString::kMinLength`); shorter substrings are copied.
 */
const SLICED_STRING_MIN_LENGTH = 13;

/**
 * A copy of a string that no longer references the buffer it was sliced from. Importers cut ids
 * and cell texts out of decoded input chunks, and V8 represents such a substring of 13 or more
 * characters as a view on the chunk, so a retained id or value would keep the whole chunk alive
 * for the life of the builder and the snapshot (measured: a 40 MiB CSV kept 47 MiB of decoded
 * text resident through 100k retained ids). Concatenating one character and slicing it off makes
 * V8 flatten the string into fresh storage; shorter strings are already copies and are returned
 * as they are.
 * @param s - the string
 * @returns an equal string that owns its characters
 */
export function detachString(s: string): string {
    return s.length < SLICED_STRING_MIN_LENGTH ? s : ` ${s}`.slice(1);
}

/**
 * Whether a string contains a lone UTF-16 surrogate, which TextEncoder would replace with U+FFFD on
 * the wire (design section 4.1: such an id is E_INVALID_ID).
 * @param s - the string
 * @returns true when a high surrogate lacks its low partner or a low surrogate stands alone
 */
export function hasLoneSurrogate(s: string): boolean {
    const n = s.length;
    for (let i = 0; i < n; i++) {
        const c = s.charCodeAt(i);
        if (c < 0xd800 || c > 0xdfff) {
            continue;
        }
        if (c > 0xdbff) {
            return true;
        }
        if (i + 1 >= n) {
            return true;
        }
        const d = s.charCodeAt(i + 1);
        if (d < 0xdc00 || d > 0xdfff) {
            return true;
        }
        i++;
    }
    return false;
}

/**
 * Decode one row of UTF-8 bytes with an ASCII fast path.
 * @param utf8 - the byte store
 * @param start - first byte of the row
 * @param end - one past the last byte of the row
 * @returns the decoded string (malformed sequences become U+FFFD)
 */
export function decodeUtf8Row(utf8: Uint8Array, start: number, end: number): string {
    const len = end - start;
    if (len === 0) {
        return "";
    }
    if (len <= ASCII_FAST_PATH_MAX_BYTES) {
        let s = "";
        let i = start;
        for (; i < end; i++) {
            const b = utf8[i];
            if (b >= 0x80) {
                break;
            }
            s += String.fromCharCode(b);
        }
        if (i === end) {
            return s;
        }
    }
    return replacingDecoder().decode(utf8.subarray(start, end));
}

/**
 * Whether every byte of a range is ASCII.
 * @param utf8 - the byte store
 * @param start - first byte
 * @param end - one past the last byte
 * @returns true when no byte is >= 0x80
 */
function isAsciiRange(utf8: Uint8Array, start: number, end: number): boolean {
    for (let i = start; i < end; i++) {
        if (utf8[i] >= 0x80) {
            return false;
        }
    }
    return true;
}

/**
 * Decode a range of rows in one pass: when the whole byte range is ASCII, one TextDecoder call plus
 * one substring per row (byte offsets equal char offsets); otherwise one decode per row.
 * @param utf8 - the byte store
 * @param offsets - the row offsets
 * @param from - first row
 * @param to - one past the last row
 * @returns the decoded rows in order
 */
export function decodeUtf8Rows(utf8: Uint8Array, offsets: Uint32Array, from: number, to: number): string[] {
    const out = new Array<string>(to - from);
    if (to <= from) {
        return out;
    }
    const base = offsets[from];
    const end = offsets[to];
    if (isAsciiRange(utf8, base, end)) {
        const whole = replacingDecoder().decode(utf8.subarray(base, end));
        for (let i = from; i < to; i++) {
            out[i - from] = whole.substring(offsets[i] - base, offsets[i + 1] - base);
        }
        return out;
    }
    for (let i = from; i < to; i++) {
        out[i - from] = decodeUtf8Row(utf8, offsets[i], offsets[i + 1]);
    }
    return out;
}

/**
 * Whether one row of bytes is well-formed UTF-8 (a fatal decode).
 * @param utf8 - the byte store
 * @param start - first byte of the row
 * @param end - one past the last byte of the row
 * @returns true when the row decodes without error
 */
function isWellFormedUtf8Row(utf8: Uint8Array, start: number, end: number): boolean {
    let ascii = true;
    for (let i = start; i < end; i++) {
        if (utf8[i] >= 0x80) {
            ascii = false;
            break;
        }
    }
    if (ascii) {
        return true;
    }
    try {
        strictDecoder().decode(utf8.subarray(start, end));
        return true;
    } catch {
        return false;
    }
}

/**
 * Encode the first `length` entries of an array as an Arrow Utf8 store in one sizing pass and one
 * writing pass. Entries that are not strings (the number rows of a mixed id map) encode as empty
 * rows; offsets start at 0.
 * @param source - the rows; only entries below `length` are read
 * @param length - the number of rows to encode
 * @returns fresh, exactly sized offsets (length + 1) and utf8 arrays
 */
export function encodeUtf8Rows(source: ArrayLike<unknown>, length: number): EncodedUtf8 {
    const offsets = new Uint32Array(length + 1);
    let total = 0;
    for (let i = 0; i < length; i++) {
        const v = source[i];
        if (typeof v === "string") {
            total += utf8ByteLength(v);
        }
        offsets[i + 1] = total;
    }
    const utf8 = new Uint8Array(total);
    let p = 0;
    for (let i = 0; i < length; i++) {
        const v = source[i];
        if (typeof v === "string") {
            p = encodeUtf8Into(v, utf8, p);
        }
    }
    return { offsets, utf8 };
}

/**
 * Structure-level check of a Utf8 store's layout (design section 9.5): `rows + 1` offsets,
 * non-decreasing, ending inside the byte store.
 * @param offsets - the offsets array
 * @param utf8 - the byte store
 * @param rows - the expected row count
 * @param ref - the manifest path named in the error, e.g. "ids.offsets"
 */
export function checkUtf8Layout(offsets: Uint32Array, utf8: Uint8Array, rows: number, ref: string): void {
    if (offsets.length !== rows + 1) {
        throw new GraphFormatError(
            "E_BAD_SERIALIZATION",
            `${ref}: expected ${rows + 1} offsets, found ${offsets.length}`,
            { ref, reason: "offsets length", expected: rows + 1, found: offsets.length },
        );
    }
    let previous = offsets[0];
    for (let i = 1; i <= rows; i++) {
        const current = offsets[i];
        if (current < previous) {
            throw new GraphFormatError("E_BAD_SERIALIZATION", `${ref}: offsets decrease at row ${i - 1}`, {
                ref,
                reason: "offsets not monotonic",
                row: i - 1,
            });
        }
        previous = current;
    }
    if (previous > utf8.length) {
        throw new GraphFormatError(
            "E_BAD_SERIALIZATION",
            `${ref}: offsets end at ${previous} but the byte store has ${utf8.length} bytes`,
            { ref, reason: "offsets exceed utf8", end: previous, byteLength: utf8.length },
        );
    }
}

/** The encoded representation of a store: offsets plus bytes, always materialised together. */
export interface EncodedUtf8 {
    /** rows + 1 non-decreasing offsets. */
    readonly offsets: U32;
    /** The concatenated UTF-8 bytes. */
    readonly utf8: U8;
}

/**
 * An Arrow Utf8 string store with lazily materialised encoded and decoded representations (design
 * sections 4.2 and 5.1). Immutable once built; every lazily computed representation is cached.
 */
export class Utf8Store {
    /** Number of rows. */
    readonly length: number;

    /**
     * The decoded rows, shared by reference for a store built from strings (only entries below
     * `length` are read) or produced by `materialiseDecoded()` for an encoded store.
     */
    private source: ArrayLike<string> | null;

    /** Sparse per-row decode cache of an encoded-only store; allocated on the first decode. */
    private cache: (string | undefined)[] | null;

    /** The offsets and bytes, once encoded or when adopted. */
    private encodedArrays: EncodedUtf8 | null;

    /**
     * Construct a store; use the static factories. At least one representation is always given.
     * @param length - the row count
     * @param source - the decoded rows or null
     * @param encoded - the offsets and bytes or null
     */
    private constructor(length: number, source: ArrayLike<string> | null, encoded: EncodedUtf8 | null) {
        this.length = length;
        this.source = source;
        this.cache = null;
        this.encodedArrays = encoded;
    }

    /**
     * A store over decoded strings, shared by reference: the builder-side representation. Encoding
     * happens on the first read of `offsets` / `utf8`.
     * @param source - the rows; only entries below `length` are ever read, so a builder array that
     * keeps growing can be shared (design section 4.2)
     * @param length - the row count; default source.length
     * @returns the store
     */
    static fromStrings(source: ArrayLike<string>, length: number = source.length): Utf8Store {
        return new Utf8Store(length, source, null);
    }

    /**
     * A store over an encoded Utf8 layout, adopted by reference: the wire-side representation.
     * Decoding happens per row on `at()` and in bulk on `slice()` / `decodeAll()`. The layout is not
     * checked here; `checkUtf8Layout` does that for untrusted input.
     * @param offsets - rows + 1 non-decreasing offsets
     * @param utf8 - the concatenated bytes
     * @returns the store
     */
    static fromEncoded(offsets: U32, utf8: U8): Utf8Store {
        return new Utf8Store(Math.max(0, offsets.length - 1), null, { offsets, utf8 });
    }

    /**
     * Whether the encoded representation exists (adopted or already materialised).
     * @returns true when `offsets` / `utf8` can be read without encoding
     */
    get encoded(): boolean {
        return this.encodedArrays !== null;
    }

    /**
     * Whether the store is backed by a decoded source array, so `at()` never decodes.
     * @returns true for a store built from strings
     */
    get decoded(): boolean {
        return this.source !== null;
    }

    /**
     * Whether the encoded arrays were transferred away while rows still depend on them (design
     * section 9.1): a transferred buffer leaves zero-length views.
     * @returns true when detached
     */
    get detached(): boolean {
        return (
            this.source === null &&
            this.length > 0 &&
            this.encodedArrays !== null &&
            this.encodedArrays.offsets.length === 0
        );
    }

    /**
     * The `length + 1` offsets, encoding the source on first access.
     * @returns the offsets array (cached)
     */
    get offsets(): U32 {
        return this.ensureEncoded().offsets;
    }

    /**
     * The concatenated UTF-8 bytes, encoding the source on first access.
     * @returns the byte array (cached)
     */
    get utf8(): U8 {
        return this.ensureEncoded().utf8;
    }

    /**
     * The string of one row, decoded on first access and cached per row.
     * @param row - the row index
     * @returns the string; E_INDEX_RANGE when row is out of range
     */
    at(row: number): string {
        if (!(row >= 0 && row < this.length) || !Number.isInteger(row)) {
            throw new GraphFormatError("E_INDEX_RANGE", `row ${row} out of range (length ${this.length})`, {
                index: row,
                size: this.length,
            });
        }
        if (this.source !== null) {
            return this.source[row];
        }
        this.cache ??= new Array<string | undefined>(this.length);
        const cached = this.cache[row];
        if (cached !== undefined) {
            return cached;
        }
        const { offsets, utf8 } = this.ensureEncoded();
        const value = decodeUtf8Row(utf8, offsets[row], offsets[row + 1]);
        this.cache[row] = value;
        return value;
    }

    /**
     * The strings of a row range decoded in one pass (one TextDecoder call for an all-ASCII range);
     * nothing is cached. Bounds follow Array.prototype.slice (negative values count from the end,
     * out-of-range values are clamped).
     * @param start - first row (default 0)
     * @param end - one past the last row (default length)
     * @returns a fresh array of the rows in order
     */
    slice(start = 0, end = this.length): string[] {
        const [from, to] = resolveRange(start, end, this.length);
        const { source } = this;
        if (source !== null) {
            const out = new Array<string>(to - from);
            for (let i = from; i < to; i++) {
                out[i - from] = source[i];
            }
            return out;
        }
        const { offsets, utf8 } = this.ensureEncoded();
        return decodeUtf8Rows(utf8, offsets, from, to);
    }

    /**
     * Every row decoded in one pass; not cached.
     * @returns a fresh array of all rows
     */
    decodeAll(): string[] {
        return this.slice(0, this.length);
    }

    /**
     * Decode every row of an encoded store once, in bulk, and keep the result as the decoded
     * representation so that later `at()` calls are O(1) without a per-row decode. For a store that
     * already has its decoded rows this is a no-op. Used before a pass that will touch every row
     * anyway (building the reverse map of a wire-decoded id map).
     */
    materialiseDecoded(): void {
        if (this.source === null) {
            this.source = this.decodeAll();
            this.cache = null;
        }
    }

    /**
     * Bytes of typed storage: the offsets and utf8 arrays when they exist, 0 for a store that has not
     * been encoded yet. JS strings are never counted.
     * @returns the byte count
     */
    byteLength(): number {
        const enc = this.encodedArrays;
        return enc === null ? 0 : enc.offsets.byteLength + enc.utf8.byteLength;
    }

    /**
     * The first row whose bytes are not well-formed UTF-8, for full validation of untrusted input
     * (design section 9.5: string ids are decoded with a fatal decoder). A source-backed store is
     * always well-formed.
     * @returns the row index, or -1 when every row is well-formed
     */
    firstMalformedRow(): number {
        const enc = this.encodedArrays;
        if (this.source !== null || enc === null) {
            return -1;
        }
        const { offsets, utf8 } = enc;
        for (let i = 0; i < this.length; i++) {
            if (!isWellFormedUtf8Row(utf8, offsets[i], offsets[i + 1])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * The encoded representation, encoding the source once on first use. A store always has a source
     * when it has no encoded arrays (the private constructor is only reached through the factories).
     * @returns the offsets and bytes
     */
    private ensureEncoded(): EncodedUtf8 {
        let enc = this.encodedArrays;
        if (enc === null) {
            enc = encodeUtf8Rows(this.source as ArrayLike<string>, this.length);
            this.encodedArrays = enc;
            // the store is the first holder of the buffers it materialises (design section 9.1)
            claimHolder(enc.offsets.buffer);
            claimHolder(enc.utf8.buffer);
        }
        return enc;
    }
}

/**
 * Resolve Array.prototype.slice-style bounds against a length.
 * @param start - the requested start (negative counts from the end)
 * @param end - the requested end (negative counts from the end)
 * @param length - the length of the sequence
 * @returns the clamped [from, to] with from <= to
 */
export function resolveRange(start: number, end: number, length: number): [from: number, to: number] {
    let from = Math.trunc(start);
    let to = Math.trunc(end);
    if (Number.isNaN(from)) {
        from = 0;
    }
    if (Number.isNaN(to)) {
        to = 0;
    }
    from = from < 0 ? Math.max(length + from, 0) : Math.min(from, length);
    to = to < 0 ? Math.max(length + to, 0) : Math.min(to, length);
    return [from, Math.max(from, to)];
}
