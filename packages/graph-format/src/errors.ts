/**
 * The error class and the closed set of error codes of @graphty/graph-format (design section 11).
 *
 * Policy (design section 11.1): the builder and the constructors throw on malformed input, on the
 * first error, with a GraphFormatError carrying a stable `code`; snapshot methods that take an index
 * are total for in-range arguments and unchecked otherwise; only GraphFormatError instances are ever
 * thrown by this package. Messages are plain ASCII and name the offending index, id or value when it
 * is a number or a short string. ImportError (code E_IMPORT) is declared by @graphty/graph-io.
 */

/**
 * The closed union of error codes a GraphFormatError can carry (design section 11.2). Each code is
 * stable across versions and is the value consumers branch on; the message is for humans.
 *
 * - E_INVALID_ID: NaN, a non-finite number, a bigint, an object, null, undefined or a string with a
 *   lone surrogate used as an id (details.reason "lone surrogate").
 * - E_UNKNOWN_NODE: addEdge with addMissingNodes false, addEdgeByIndex with a dead or out-of-range
 *   index (details.index), or a requireIndex miss.
 * - E_INDEX_RANGE: ids.idOf / table.value out of range; inducedSubgraph with a repeated or
 *   out-of-range index.
 * - E_TOO_LARGE: nodeCount, edgeCount or arcCount would exceed MAX_COUNT.
 * - E_INVALID_WEIGHT: a NaN weight at addEdge / setEdgeWeight / addEdges.
 * - E_DIRECTED: mate() on a directed snapshot; setDirected() refused (locked, or a change the edge
 *   set does not allow).
 * - E_SELF_LOOP: a self-loop under selfLoops "error".
 * - E_DUPLICATE_EDGE: a parallel edge under duplicateEdges "error" (details source, target, edges).
 * - E_DUPLICATE_EDGE_ID: a unique role "id" edge column violated (both edge indices in details).
 * - E_DUPLICATE_ID: a unique node column violated; addGraph with onDuplicateNode "error"; a
 *   repeated id in the `ids` array or F64 given to fromEdgeArrays / fromCsr.
 * - E_DUPLICATE_ROLE: two columns with the same role in one table.
 * - E_UNKNOWN_COLUMN: require() of an absent column.
 * - E_COLUMN_TYPE: requireTyped() dtype mismatch; a non-JSON default / fill / options / extra value
 *   (details.field); a string value with a lone surrogate.
 * - E_COLUMN_LENGTH: set() with data.length !== rowCount * components.
 * - E_COLUMN_ALIGNMENT: a u8 array without a constructible padded u32 view under adopt "strict".
 * - E_COLUMN_EXISTS: declareNodeColumn / declareEdgeColumn twice with a different declaration.
 * - E_COLUMN_IMMUTABLE: mutableData() / markDirty() / mutableValidity() / setAll() on an immutable
 *   column.
 * - E_NO_DEFAULT: materializeDefault() on a column without a declared default.
 * - E_PARTITION: contract() partition of the wrong length or holding INVALID_INDEX.
 * - E_INVALID_PERMUTATION: relabel() with a perm that is not a permutation of 0..n-1.
 * - E_MASK_LENGTH: filterEdges() / inducedSubgraph({ mask }) with a short mask.
 * - E_GPU_INELIGIBLE: gpuView() / paddedU32View() on a string, list or json column.
 * - E_INVALID_SNAPSHOT: validate() failure (details.invariant, row / arc / edge / column) or
 *   validate({ checksum: true }) on a snapshot without recorded checksums (details.reason
 *   "no-checksum").
 * - E_BAD_SERIALIZATION: bad magic, endianness probe, manifest or buffer reference (details.ref).
 * - E_UNSUPPORTED_VERSION: unknown wire major or formatVersion (details.kind "wire" | "format",
 *   details.found, details.supported).
 * - E_DETACHED: a core accessor after a consuming transfer.
 * - E_BUILDER_DISPOSED: any call on a disposed builder.
 * - E_UNSUPPORTED: big-endian host (details.reason); unknown wire dtype (details.dtype) or id-map
 *   kind (details.kind).
 * - E_IMPORT: reserved for @graphty/graph-io's ImportError.
 */
export type GraphFormatErrorCode =
    | "E_INVALID_ID"
    | "E_UNKNOWN_NODE"
    | "E_INDEX_RANGE"
    | "E_TOO_LARGE"
    | "E_INVALID_WEIGHT"
    | "E_DIRECTED"
    | "E_SELF_LOOP"
    | "E_DUPLICATE_EDGE"
    | "E_DUPLICATE_EDGE_ID"
    | "E_DUPLICATE_ID"
    | "E_DUPLICATE_ROLE"
    | "E_UNKNOWN_COLUMN"
    | "E_COLUMN_TYPE"
    | "E_COLUMN_LENGTH"
    | "E_COLUMN_ALIGNMENT"
    | "E_COLUMN_EXISTS"
    | "E_COLUMN_IMMUTABLE"
    | "E_NO_DEFAULT"
    | "E_PARTITION"
    | "E_INVALID_PERMUTATION"
    | "E_MASK_LENGTH"
    | "E_GPU_INELIGIBLE"
    | "E_INVALID_SNAPSHOT"
    | "E_BAD_SERIALIZATION"
    | "E_UNSUPPORTED_VERSION"
    | "E_DETACHED"
    | "E_BUILDER_DISPOSED"
    | "E_UNSUPPORTED"
    | "E_IMPORT";

const EMPTY_DETAILS: Readonly<Record<string, unknown>> = Object.freeze({});

/**
 * The only error type this package throws (design section 11.2). `code` is the stable, programmatic
 * discriminator; `details` carries the machine-readable context named per code in the
 * GraphFormatErrorCode documentation (index, id, edge, invariant, row, found, supported, ...) as a
 * frozen shallow copy of what the thrower supplied. `name` is "GraphFormatError" so stack traces and
 * test matchers read naturally.
 */
export class GraphFormatError extends Error {
    /** The stable error code (design section 11.2). */
    readonly code: GraphFormatErrorCode;

    /** Machine-readable context for the failure; an empty frozen object when none was supplied. */
    readonly details: Readonly<Record<string, unknown>>;

    /**
     * Create a GraphFormatError.
     * @param code - the stable error code the caller can branch on
     * @param message - a plain-ASCII human-readable message naming the offending index, id or value
     * @param details - optional machine-readable context; copied shallowly and frozen
     */
    constructor(code: GraphFormatErrorCode, message: string, details?: Readonly<Record<string, unknown>>) {
        super(message);
        this.name = "GraphFormatError";
        this.code = code;
        this.details = details === undefined ? EMPTY_DETAILS : Object.freeze({ ...details });
    }
}
