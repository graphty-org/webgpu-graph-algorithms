/**
 * Package-wide constants of @graphty/graph-format (design sections 2, 3.2, 9.2, 10.3 and 12.2).
 *
 * The four constants listed in design section 12.2 (INVALID_INDEX, MAX_COUNT, FORMAT_VERSION and
 * SNAPSHOT_BRAND) are public and re-exported by src/index.ts. The remaining constants are shared by
 * the builder, arena and wire modules but are not part of the public surface.
 */

/**
 * The one and only "no index" sentinel: 0xFFFFFFFF. Used for node, edge and arc indices alike, both as
 * a JS return value (indexOf, findArc, edgeIndexOf, nodeColumn, edgeColumn, codeOf) and inside
 * Uint32Array vectors (remaps, parents, labels). It never appears in rowPtr, colIdx, arcToEdge,
 * edgeToArc or any view array (invariant I2), and because every count is bounded by MAX_COUNT it can
 * never collide with a valid index (invariant I3).
 */
export const INVALID_INDEX = 0xffffffff;

/**
 * Upper bound on nodeCount, edgeCount and arcCount: 0xFFFFFFFE (invariant I3). Crossing it at an add*
 * call throws E_TOO_LARGE. One below INVALID_INDEX so the sentinel is always free.
 */
export const MAX_COUNT = 0xfffffffe;

/**
 * The data-model major version carried by every snapshot as `formatVersion` and compared by
 * isGraphSnapshot(), fromWire() and fromBytes() (design section 13.5). A mismatch is
 * E_UNSUPPORTED_VERSION with details.kind "format". It bumps only when the meaning of an invariant, a
 * field, a flag or a sentinel changes.
 */
export const FORMAT_VERSION = 1;

/**
 * Brand property read by isGraphSnapshot() (design section 7.5). It is a Symbol.for() symbol rather
 * than a private one so that two copies of the package inside one application still recognise each
 * other's snapshots structurally, without instanceof.
 */
export const SNAPSHOT_BRAND: unique symbol = Symbol.for("@graphty/graph-format/snapshot");

/**
 * Alignment, in bytes, of every core-array segment inside the arena and of every segment inside the
 * GSNP byte container (design sections 9.2 and 10.3). It is WebGPU's minStorageBufferOffsetAlignment
 * default, so each segment can be bound as a storage buffer window without copying.
 * @internal
 */
export const ALIGNMENT = 256;

/**
 * Whether this host stores multi-byte integers little-endian, computed once at module load. Every
 * supported platform (x86-64, ARM64, WebAssembly) is little-endian; on a big-endian host toBytes() and
 * toWire() throw E_UNSUPPORTED with details.reason "big-endian host" so that no file is ever written
 * that other hosts would reject (design section 9.2).
 * @internal
 */
export const IS_LITTLE_ENDIAN: boolean = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * Wire major version written into the GSNP container header and the manifest (design section 9.2).
 * A reader refuses any other major with E_UNSUPPORTED_VERSION (details.kind "wire").
 * @internal
 */
export const WIRE_MAJOR = 1;

/**
 * Wire minor version written into the GSNP container header and the manifest (design section 9.2).
 * Minors are additive: older readers ignore unknown manifest fields. Wire 1.0 tags non-finite
 * numbers as `{ "$num": ... }` and wraps a JSON object whose only key is a tag name as
 * `{ "$esc": ... }` (design section 5.9).
 * @internal
 */
export const WIRE_MINOR = 0;

/**
 * The manifest `format` discriminator of a WireManifest (design section 9.1).
 * @internal
 */
export const WIRE_FORMAT = "graphty-snapshot";

/**
 * The four magic bytes at the start of a GSNP byte container: "GSNP" in ASCII (design section 9.2).
 * @internal
 */
export const CONTAINER_MAGIC: readonly [number, number, number, number] = [0x47, 0x53, 0x4e, 0x50];

/**
 * The u32 value written through a host-order Uint32Array at container offset 8 so a reader can detect
 * an endianness mismatch (design section 9.2): a little-endian reader sees the bytes 04 03 02 01.
 * @internal
 */
export const ENDIAN_PROBE = 0x01020304;
