/**
 * The wire form of a snapshot (design sections 9.1, 9.2 and 12.2): a JSON-serialisable manifest
 * plus a list of ArrayBuffers. The same object is what postMessage, structuredClone, IndexedDB and
 * the GSNP byte container carry; `buffers` is literally the postMessage transfer list. Absent
 * members are null, never optional (design section 12.1).
 */

import { type ColumnMeta, type GraphMeta, type ValidationLevel } from "./columns.js";
import { type NodeIdMapKind, type SnapshotFlags, type ViewName } from "./snapshot.js";

/** Element type of a wire buffer reference: the five typed-array dtypes plus "utf8" for byte stores. */
export type WireDtype = "u32" | "i32" | "f32" | "f64" | "u8" | "utf8";

/**
 * One typed array inside WireSnapshot.buffers (design section 9.1). Validation at level "structure"
 * checks byteOffset % elementSize === 0, byteLength === length * elementSize and that the range lies
 * inside the buffer (E_BAD_SERIALIZATION with details.ref otherwise).
 */
export interface WireBufferRef {
    /** Index into WireSnapshot.buffers (always 0 inside a byte container). */
    readonly buffer: number;
    /** Byte offset inside the buffer (relative to the buffer region inside a container). */
    readonly byteOffset: number;
    /** Byte length of the array. */
    readonly byteLength: number;
    /** Element type. */
    readonly dtype: WireDtype;
    /** Element count. */
    readonly length: number;
}

/**
 * An Arrow Utf8 store on the wire: u32 offsets plus UTF-8 bytes (string ids, string columns, dictionaries, JSON text).
 */
export interface WireUtf8 {
    /** rows + 1 offsets. */
    readonly offsets: WireBufferRef;
    /** The concatenated UTF-8 bytes. */
    readonly utf8: WireBufferRef;
}

/**
 * One column on the wire (design section 9.1): the resolved metadata plus the buffers its dtype
 * uses, each null when not applicable. meta.default, options and extra are JSON values with the
 * tagged encoding of design section 5.9 for non-finite numbers.
 */
export interface WireColumn {
    /** Resolved metadata. */
    readonly meta: ColumnMeta;
    /** Values (numeric dtypes), packed words (bool) or codes (dict); null for string / list / json. */
    readonly data: WireBufferRef | null;
    /** Validity bitmap words; null when every row is set. */
    readonly validity: WireBufferRef | null;
    /** Number of unset rows (recomputed from the bitmap by validation). */
    readonly nullCount: number;
    /** dict only: the dictionary strings. */
    readonly dictionary: WireUtf8 | null;
    /** string only: the Utf8 store. */
    readonly strings: WireUtf8 | null;
    /** list only: rows + 1 offsets into the child. */
    readonly offsets: WireBufferRef | null;
    /** list only: the child column. */
    readonly child: WireColumn | null;
    /** json only: the JSON text of every row as a Utf8 store. */
    readonly jsonText: WireUtf8 | null;
}

/** The arena region inside one wire buffer, so a receiver keeps the one-writeBuffer path (design section 9.1). */
export interface WireArena {
    /** Index into WireSnapshot.buffers. */
    readonly buffer: number;
    /** Start of the arena inside the buffer. */
    readonly byteOffset: number;
    /** Total padded length of the arena. */
    readonly byteLength: number;
    /** Hot prefix length (design section 10.3). */
    readonly hotByteLength: number;
}

/**
 * The id map on the wire (design section 4.5), one shape for every kind: the members a kind does not
 * use are null. The reverse Map is never serialised; a receiver rebuilds it lazily.
 */
export interface WireIdMap {
    /** The storage kind. */
    readonly kind: NodeIdMapKind;
    /** Number of ids. */
    readonly size: number;
    /** identity only: id === index + offset; 0 otherwise. */
    readonly offset: number;
    /** dense: u32 ids per index; numeric: f64 ids per index. */
    readonly values: WireBufferRef | null;
    /** mixed: u8 tag per index (0 number, 1 string). */
    readonly tags: WireBufferRef | null;
    /** mixed: f64 per index (0 where string). */
    readonly numbers: WireBufferRef | null;
    /** string / mixed: u32 offsets of the Utf8 store. */
    readonly offsets: WireBufferRef | null;
    /** string / mixed: the UTF-8 bytes. */
    readonly utf8: WireBufferRef | null;
}

/**
 * The JSON-serialisable manifest of a wire snapshot (design section 9.1). Readers refuse a
 * formatVersion or wire major they do not know (E_UNSUPPORTED_VERSION), ignore unknown manifest
 * fields of a newer minor, refuse an unknown column dtype unless unknownColumns is "skip", refuse an
 * unknown id-map kind, and ignore unknown view entries.
 */
export interface WireManifest {
    /** Discriminator. */
    readonly format: "graphty-snapshot";
    /** Wire [major, minor]. */
    readonly wire: readonly [major: number, minor: number];
    /** The producing package version (masked when golden fixtures are compared). */
    readonly producer: string;
    /** The data-model major. */
    readonly formatVersion: 1;
    /** Whether the graph is directed. */
    readonly directed: boolean;
    /** The four counts. */
    readonly counts: {
        readonly nodes: number;
        readonly edges: number;
        readonly arcs: number;
        readonly selfLoops: number;
    };
    /** The flags (invariant I9). */
    readonly flags: SnapshotFlags;
    /**
     * The core arrays; null for an identity permutation (always, whether or not a getter materialised it), an absent
     * weights array or a zero-length array.
     */
    readonly core: {
        readonly rowPtr: WireBufferRef;
        readonly colIdx: WireBufferRef | null;
        readonly weights: WireBufferRef | null;
        readonly arcToEdge: WireBufferRef | null;
        readonly edgeToArc: WireBufferRef | null;
    };
    /** The arena region when the core lives in one buffer; null otherwise. */
    readonly arena: WireArena | null;
    /** The id map. */
    readonly ids: WireIdMap;
    /** Node columns in declaration order. */
    readonly nodeColumns: readonly WireColumn[];
    /** Edge columns in declaration order. */
    readonly edgeColumns: readonly WireColumn[];
    /** Graph columns in declaration order. */
    readonly graphColumns: readonly WireColumn[];
    /** Extension tables with their own row counts (design section 5.10). */
    readonly extensions: readonly {
        readonly name: string;
        readonly rowCount: number;
        readonly columns: readonly WireColumn[];
    }[];
    /** Graph metadata. */
    readonly meta: GraphMeta;
    /** Cached views included on request: view name -> member name -> buffer; null unless includeViews named some. */
    readonly views: Readonly<Record<string, Readonly<Record<string, WireBufferRef>>>> | null;
    /** Buffer indices that were copied rather than transferred (shared storage, design section 9.1). */
    readonly copied: readonly number[];
    /** The snapshot label. */
    readonly label: string | null;
}

/**
 * The plain-object, transferable, versioned representation of a snapshot: the only cloneable shape (design section
 * 9.1).
 */
export interface WireSnapshot {
    /** The manifest. */
    readonly manifest: WireManifest;
    /** The buffers the manifest's references index; the postMessage transfer list when produced with transfer: true. */
    readonly buffers: readonly ArrayBuffer[];
}

/** Options of toWire() (design section 9.1). */
export interface ToWireOptions {
    /**
     * Put only exclusively owned buffers into `buffers` as transferables and copy every shared one (listed in
     * manifest.copied); default false (nothing is transferable).
     */
    readonly transfer?: boolean | undefined;
    /** Cached views to carry; the receiver recomputes the rest. Scalar views are ignored. */
    readonly includeViews?: readonly ViewName[] | undefined;
    /** Whether attribute columns are carried; default true. */
    readonly includeColumns?: boolean | undefined;
}

/** Options of toBytes() and toByteChunks() (design section 9.2). */
export interface ToBytesOptions {
    /** Cached views to carry. */
    readonly includeViews?: readonly ViewName[] | undefined;
}

/** Options of fromWire(), fromBytes() and fromByteChunks() (design sections 9.1 and 9.2). */
export interface FromWireOptions {
    /** Validation level; default "structure" for fromWire, "full" for fromBytes. */
    readonly validate?: ValidationLevel | undefined;
    /** Copy the buffers instead of adopting them; default false. */
    readonly copy?: boolean | undefined;
    /**
     * What to do with a column whose dtype the reader does not know; default "error" (E_UNSUPPORTED); "skip" drops it
     * and records the name in meta.extra["graphty.skippedColumns"].
     */
    readonly unknownColumns?: "error" | "skip" | undefined;
}
