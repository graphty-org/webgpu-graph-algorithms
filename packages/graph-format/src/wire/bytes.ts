/**
 * The GSNP byte container (design section 9.2): `toBytes()` writes one contiguous buffer,
 * `toByteChunks()` the same bytes as a sequence (header + manifest, then every 256-padded segment
 * in manifest order), `fromBytes()` adopts a container's buffer region as the arena and
 * `fromByteChunks()` adopts each chunk's buffer per segment. The snapshot's `toBytes` /
 * `toByteChunks` methods call the functions of this module directly.
 *
 * Layout, all integers little-endian:
 *
 * ```
 * offset  size   field
 * 0       4      magic "GSNP"
 * 4       2      wire major (u16)
 * 6       2      wire minor (u16)
 * 8       4      endianness probe: 0x01020304 written through a host-order Uint32Array
 * 12      4      manifest byte length L (u32)
 * 16      L      manifest: UTF-8 JSON of the WireManifest, every reference into buffer 0 with a
 *                region-relative byteOffset
 * 16+L    pad    zero padding to the next multiple of 256
 * B       ...    buffer region: each array at a 256-aligned offset in manifest order, core arrays
 *                first so the region's prefix is the arena; every segment padded to 256 bytes
 * ```
 *
 * The reader refuses a bad magic, an unexpected probe, a truncated header or manifest, a manifest
 * whose versions disagree with the header (E_BAD_SERIALIZATION), an unknown wire major
 * (E_UNSUPPORTED_VERSION) and everything the manifest reader of from-wire.ts refuses.
 */

import { ALIGNMENT, CONTAINER_MAGIC, ENDIAN_PROBE, IS_LITTLE_ENDIAN, WIRE_MAJOR, WIRE_MINOR } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import { type FromWireOptions, type ToBytesOptions, type TypedArrayData, type U8 } from "../types/index.js";
import { alignUp, padTo4 } from "../util/typed-array.js";
import {
    badWire,
    bufferRegion,
    checkManifestVersions,
    decodeManifest,
    isSharedArrayBuffer,
    type LocatedRange,
    parseGuardedJson,
    resolveFromWireOptions,
    type WireRegion,
} from "./from-wire.js";
import { collectWire, type ContainerSegment, encodeManifest, WireLayout } from "./to-wire.js";

// ============================================================ constants

/** Bytes before the manifest: magic, major, minor, probe and manifest length. */
export const CONTAINER_HEADER_BYTES = 16;

const fatalDecoder = new TextDecoder("utf-8", { fatal: true });

// ============================================================ writing

/** Everything a container write needs, computed once per call. */
interface ContainerPlan {
    /** The manifest's UTF-8 bytes. */
    readonly manifestBytes: Uint8Array;
    /** B: the start of the buffer region (header + manifest padded to 256). */
    readonly regionStart: number;
    /** The placed arrays in region order. */
    readonly segments: readonly ContainerSegment[];
    /** The padded region length. */
    readonly regionLength: number;
}

/**
 * Lay a snapshot out for the container.
 * @param snapshot - the snapshot
 * @param options - the views to carry
 * @returns the plan
 */
function planContainer(snapshot: GraphSnapshot, options: ToBytesOptions): ContainerPlan {
    const layout = new WireLayout("container");
    const { manifest } = collectWire(snapshot, layout, options.includeViews, true);
    const manifestBytes = encodeManifest(manifest);
    return {
        manifestBytes,
        regionStart: alignUp(CONTAINER_HEADER_BYTES + manifestBytes.byteLength, ALIGNMENT),
        segments: layout.segments,
        regionLength: layout.regionLength,
    };
}

/**
 * Write the 16-byte header into a fresh buffer (design section 9.2).
 * @param out - a Uint8Array at offset 0 of its buffer with at least 16 bytes
 * @param manifestLength - L
 */
function writeHeader(out: U8, manifestLength: number): void {
    out.set(CONTAINER_MAGIC, 0);
    const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
    view.setUint16(4, WIRE_MAJOR, true);
    view.setUint16(6, WIRE_MINOR, true);
    new Uint32Array(out.buffer, out.byteOffset + 8, 1)[0] = ENDIAN_PROBE;
    view.setUint32(12, manifestLength, true);
}

/**
 * The bytes of a typed array as a Uint8Array view.
 * @param view - the array
 * @returns the byte view
 */
function bytesOf(view: TypedArrayData): Uint8Array {
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

/**
 * The GSNP container of a snapshot as one contiguous buffer (design section 9.2).
 * @param snapshot - the snapshot
 * @param options - the cached views to carry
 * @returns the container bytes at offset 0 of a fresh ArrayBuffer; E_UNSUPPORTED on a big-endian host
 */
export function toBytes(snapshot: GraphSnapshot, options: ToBytesOptions = {}): U8 {
    const plan = planContainer(snapshot, options);
    const out = new Uint8Array(plan.regionStart + plan.regionLength);
    writeHeader(out, plan.manifestBytes.byteLength);
    out.set(plan.manifestBytes, CONTAINER_HEADER_BYTES);
    for (const segment of plan.segments) {
        out.set(bytesOf(segment.view), plan.regionStart + segment.ref.byteOffset);
    }
    return out;
}

/**
 * Yield the chunks of a plan: the padded header + manifest, then every non-empty segment padded to
 * 256 bytes.
 * @param plan - the plan
 * @yields the chunks in order
 */
function* chunksOf(plan: ContainerPlan): Generator<U8, void, undefined> {
    const head = new Uint8Array(plan.regionStart);
    writeHeader(head, plan.manifestBytes.byteLength);
    head.set(plan.manifestBytes, CONTAINER_HEADER_BYTES);
    yield head;
    for (const segment of plan.segments) {
        if (segment.ref.byteLength === 0) {
            continue;
        }
        const chunk = new Uint8Array(alignUp(segment.ref.byteLength, ALIGNMENT));
        chunk.set(bytesOf(segment.view));
        yield chunk;
    }
}

/**
 * The GSNP container as a sequence of chunks whose concatenation equals `toBytes()` (design section
 * 9.2): the header + manifest padded to 256, then each non-empty segment padded to 256, in manifest
 * order. The layout is computed when this function is called; the chunk copies are made lazily.
 * @param snapshot - the snapshot
 * @param options - the cached views to carry
 * @returns the chunks; E_UNSUPPORTED on a big-endian host
 */
export function toByteChunks(snapshot: GraphSnapshot, options: ToBytesOptions = {}): Iterable<U8> {
    return chunksOf(planContainer(snapshot, options));
}

// ============================================================ reading

/** The fields of a container header. */
interface ContainerHeader {
    /** The wire major. */
    readonly major: number;
    /** The wire minor. */
    readonly minor: number;
    /** L, the manifest byte length. */
    readonly manifestLength: number;
}

/**
 * Check and read the 16-byte header (design section 9.2).
 * @param head - at least the first 16 bytes of the container
 * @param totalLength - the container's total byte length
 * @returns the header fields
 */
function readHeader(head: Uint8Array, totalLength: number): ContainerHeader {
    if (totalLength < CONTAINER_HEADER_BYTES) {
        throw badWire("header", `truncated: ${totalLength} bytes, expected at least ${CONTAINER_HEADER_BYTES}`, {
            byteLength: totalLength,
        });
    }
    for (let i = 0; i < CONTAINER_MAGIC.length; i++) {
        if (head[i] !== CONTAINER_MAGIC[i]) {
            throw badWire("magic", "not a GSNP container", { found: Array.from(head.subarray(0, 4)) });
        }
    }
    const view = new DataView(head.buffer, head.byteOffset, head.byteLength);
    const major = view.getUint16(4, true);
    const minor = view.getUint16(6, true);
    if (major !== WIRE_MAJOR) {
        throw new GraphFormatError(
            "E_UNSUPPORTED_VERSION",
            `wire major ${major} is not supported (reader: ${WIRE_MAJOR})`,
            {
                kind: "wire",
                found: major,
                supported: WIRE_MAJOR,
            },
        );
    }
    const probe = view.getUint32(8, IS_LITTLE_ENDIAN);
    if (probe !== ENDIAN_PROBE) {
        throw badWire(
            "endianness",
            `probe reads 0x${probe.toString(16)}; the container was written on another endianness`,
            {
                found: probe,
                expected: ENDIAN_PROBE,
            },
        );
    }
    const manifestLength = view.getUint32(12, true);
    if (CONTAINER_HEADER_BYTES + manifestLength > totalLength) {
        throw badWire(
            "manifest",
            `truncated: ${manifestLength} manifest bytes declared, ${totalLength} bytes in total`,
            {
                manifestLength,
                byteLength: totalLength,
            },
        );
    }
    return { major, minor, manifestLength };
}

/**
 * Parse the manifest bytes: a fatal UTF-8 decode, the guarding JSON reviver, then the version check
 * against the header.
 * @param bytes - the manifest bytes
 * @param header - the header the manifest must agree with
 * @returns the manifest object
 */
function readManifest(bytes: Uint8Array, header: ContainerHeader): Record<string, unknown> {
    let text: string;
    try {
        text = fatalDecoder.decode(bytes);
    } catch {
        throw badWire("manifest", "not valid UTF-8");
    }
    const manifest: unknown = parseGuardedJson(text, "manifest");
    if (typeof manifest !== "object" || manifest === null || Array.isArray(manifest)) {
        throw badWire("manifest", "expected an object");
    }
    const record = manifest as Record<string, unknown>;
    const [major, minor] = checkManifestVersions(record);
    if (major !== header.major || minor !== header.minor) {
        throw badWire(
            "wire",
            `manifest says [${major}, ${minor}], the header says [${header.major}, ${header.minor}]`,
            {
                found: [major, minor],
                header: [header.major, header.minor],
            },
        );
    }
    return record;
}

/**
 * Rebuild a snapshot from a GSNP container (design section 9.2). Defaults: `validate: "full"` and
 * `copy: false`. The buffer region is adopted as the arena when the container is a plain
 * ArrayBuffer viewed at a byteOffset that is a multiple of 8 (so f64 segments align); a
 * SharedArrayBuffer, an odd offset or `copy: true` take the copy path into a fresh 256-aligned
 * buffer.
 * @param bytes - the container: a Uint8Array, an ArrayBuffer or a SharedArrayBuffer
 * @param options - validation level, copy and unknown-column policy
 * @returns the snapshot; E_BAD_SERIALIZATION / E_UNSUPPORTED_VERSION / E_UNSUPPORTED / E_INVALID_SNAPSHOT on bad input
 */
export function fromBytes(bytes: Uint8Array | ArrayBufferLike, options?: FromWireOptions): GraphSnapshot {
    const resolved = resolveFromWireOptions(options, "full");
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const header = readHeader(view, view.byteLength);
    const manifest = readManifest(
        view.subarray(CONTAINER_HEADER_BYTES, CONTAINER_HEADER_BYTES + header.manifestLength),
        header,
    );
    const regionStart = alignUp(CONTAINER_HEADER_BYTES + header.manifestLength, ALIGNMENT);
    const regionLength = Math.max(0, view.byteLength - regionStart);
    const { buffer } = view;
    let region: WireRegion;
    if (!resolved.copy && !isSharedArrayBuffer(buffer) && view.byteOffset % 8 === 0) {
        region = bufferRegion(buffer, view.byteOffset + regionStart, regionLength);
    } else {
        const copy = new Uint8Array(alignUp(regionLength, ALIGNMENT));
        if (regionLength > 0) {
            copy.set(view.subarray(regionStart));
        }
        region = bufferRegion(copy.buffer, 0, regionLength);
    }
    return decodeManifest(manifest, [region], resolved, true);
}

/**
 * A byte stream made of chunks, addressed by absolute offset: ranges inside one chunk are located
 * for adoption, ranges spanning chunks are copied.
 */
class ChunkedBytes {
    private readonly chunks: readonly Uint8Array[];
    private readonly starts: number[];
    /** The total byte length. */
    readonly byteLength: number;

    /**
     * Wrap the chunks.
     * @param chunks - the chunks in order
     */
    constructor(chunks: readonly Uint8Array[]) {
        this.chunks = chunks;
        this.starts = [];
        let total = 0;
        for (const chunk of chunks) {
            this.starts.push(total);
            total += chunk.byteLength;
        }
        this.byteLength = total;
    }

    /**
     * The chunk index holding an absolute offset (the last chunk starting at or before it).
     * @param offset - the absolute offset
     * @returns the index, or -1 when there is no chunk
     */
    private chunkAt(offset: number): number {
        let lo = 0;
        let hi = this.starts.length - 1;
        let found = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >>> 1;
            if (this.starts[mid] <= offset) {
                found = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return found;
    }

    /**
     * Find a range inside one chunk over a plain ArrayBuffer.
     * @param offset - the absolute start
     * @param length - the byte length
     * @returns the buffer and absolute offset, or null when the range spans chunks or a SharedArrayBuffer
     */
    locate(offset: number, length: number): LocatedRange | null {
        const i = this.chunkAt(offset);
        if (i < 0) {
            return null;
        }
        const chunk = this.chunks[i];
        const start = this.starts[i];
        if (offset + length > start + chunk.byteLength || isSharedArrayBuffer(chunk.buffer)) {
            return null;
        }
        return { buffer: chunk.buffer, byteOffset: chunk.byteOffset + (offset - start) };
    }

    /**
     * Copy a range out of the stream.
     * @param offset - the absolute start
     * @param length - the byte length
     * @returns a fresh Uint8Array of `length` bytes at offset 0 of a padded buffer
     */
    read(offset: number, length: number): U8 {
        if (offset + length > this.byteLength) {
            throw badWire("chunks", `range [${offset}, ${offset + length}) exceeds the ${this.byteLength} bytes given`);
        }
        const out = new Uint8Array(new ArrayBuffer(padTo4(length)), 0, length);
        let written = 0;
        let i = this.chunkAt(offset);
        while (written < length && i < this.chunks.length) {
            const chunk = this.chunks[i];
            const start = this.starts[i];
            const from = offset + written - start;
            const take = Math.min(length - written, chunk.byteLength - from);
            out.set(chunk.subarray(from, from + take), written);
            written += take;
            i++;
        }
        return out;
    }

    /**
     * A view when the range lies inside one chunk, a copy otherwise.
     * @param offset - the absolute start
     * @param length - the byte length
     * @returns the bytes
     */
    span(offset: number, length: number): Uint8Array {
        const i = this.chunkAt(offset);
        if (i >= 0) {
            const chunk = this.chunks[i];
            const start = this.starts[i];
            if (offset + length <= start + chunk.byteLength) {
                return chunk.subarray(offset - start, offset - start + length);
            }
        }
        return this.read(offset, length);
    }

    /**
     * The stream from `base` onward as a decoder region.
     * @param base - the absolute offset of the region's start
     * @returns the region
     */
    region(base: number): WireRegion {
        return {
            byteLength: Math.max(0, this.byteLength - base),
            locate: (byteOffset: number, byteLength: number): LocatedRange | null =>
                this.locate(base + byteOffset, byteLength),
            read: (byteOffset: number, byteLength: number): U8 => this.read(base + byteOffset, byteLength),
        };
    }
}

/**
 * Rebuild a snapshot from the chunks `toByteChunks()` produced (or any split of a container's
 * bytes): each segment is adopted as a view into the chunk that holds it (a segment spanning two
 * chunks, or one in a SharedArrayBuffer, is copied) and the snapshot has `arena === null` (design
 * section 9.2). Defaults: `validate: "full"`, `copy: false`.
 * @param chunks - the chunks in order
 * @param options - validation level, copy and unknown-column policy
 * @returns the snapshot
 */
export function fromByteChunks(chunks: Iterable<Uint8Array>, options?: FromWireOptions): GraphSnapshot {
    const resolved = resolveFromWireOptions(options, "full");
    const list: Uint8Array[] = [];
    for (const chunk of chunks) {
        if (!(chunk instanceof Uint8Array)) {
            throw badWire(`chunks[${list.length}]`, "expected a Uint8Array");
        }
        list.push(resolved.copy ? chunk.slice() : chunk);
    }
    const bytes = new ChunkedBytes(list);
    const header = readHeader(bytes.span(0, Math.min(CONTAINER_HEADER_BYTES, bytes.byteLength)), bytes.byteLength);
    const manifest = readManifest(bytes.span(CONTAINER_HEADER_BYTES, header.manifestLength), header);
    const regionStart = alignUp(CONTAINER_HEADER_BYTES + header.manifestLength, ALIGNMENT);
    return decodeManifest(manifest, [bytes.region(regionStart)], resolved, false);
}
