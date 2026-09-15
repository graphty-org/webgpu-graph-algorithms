/**
 * Scalars, typed-array aliases and the attribute column model of @graphty/graph-format (design
 * sections 2, 5 and 12.2).
 *
 * This is the base layer of the type surface: the scalar aliases and the typed-array aliases live
 * here because every other types file depends on them, and the column model depends on nothing
 * outside this file. Every declaration is transcribed verbatim from design section 12.2; the
 * AttributeTableContract declared here is the instance contract of the AttributeTable class
 * implemented in src/columns/table.ts (design section 12.1 makes it a class so the JSDoc-on-method
 * lint applies). The public name AttributeTable is the class itself, re-exported type-only below so
 * every type on this surface names the class the barrel exports.
 */

import { type AttributeTable } from "../columns/table.js";

export type { AttributeTable };

// ============================================================ scalars, sentinels, typed-array aliases

/**
 * External identity of a node: a string or a number, never coerced by the core (design section 4.1).
 * Equality is SameValueZero, so 1 and "1" are two different nodes and -0 is stored as 0. NaN,
 * non-finite numbers, bigints, objects, null, undefined and strings with lone surrogates are rejected
 * with E_INVALID_ID.
 */
export type NodeId = string | number;

/**
 * Optional external identity of a logical edge, carried by an edge column with role "id" (design
 * section 4.6). Never part of the CSR structure.
 */
export type EdgeId = string | number;

/**
 * Dense node index, 0 <= i < nodeCount, assigned in insertion order (invariant I14). A plain number
 * alias for documentation: typed-array reads produce numbers, so a brand would cost a cast per read.
 */
export type NodeIndex = number;

/**
 * Logical edge index, 0 <= e < edgeCount, in addEdge order minus removals (invariant I14). Edge
 * attribute columns are indexed by it, never by arc (invariant I13).
 */
export type EdgeIndex = number;

/**
 * Arc index, 0 <= a < arcCount, in CSR order: one entry of colIdx. A directed edge is one arc; an
 * undirected edge with distinct endpoints is two arcs; an undirected self-loop is one arc (invariant
 * I7). Arc indices may exceed 2^31, so consumers never apply JS bitwise operators to them (I3).
 */
export type ArcIndex = number;

/**
 * Uint32Array over a plain ArrayBuffer (design section 12.1). The buffer type parameter is what makes
 * the array a BufferSource for GPUQueue.writeBuffer without a cast; a bare Uint32Array (whose default
 * parameter is ArrayBufferLike) is rejected there. subarray(), slice(), fill() and `new Uint32Array(n)`
 * all preserve or produce this type.
 */
export type U32 = Uint32Array<ArrayBuffer>;

/** Int32Array over a plain ArrayBuffer; see U32 for why the buffer parameter is fixed. */
export type I32 = Int32Array<ArrayBuffer>;

/** Float32Array over a plain ArrayBuffer; see U32 for why the buffer parameter is fixed. */
export type F32 = Float32Array<ArrayBuffer>;

/** Float64Array over a plain ArrayBuffer; see U32 for why the buffer parameter is fixed. */
export type F64 = Float64Array<ArrayBuffer>;

/** Uint8Array over a plain ArrayBuffer; see U32 for why the buffer parameter is fixed. */
export type U8 = Uint8Array<ArrayBuffer>;

/** Any typed array a column can be built from or adopted as (design section 5.7). */
export type TypedArrayData = U32 | I32 | F32 | F64 | U8;

/**
 * An index-aligned numeric result vector: CPU results are F64, GPU results F32, labels and parents
 * U32 or I32. The boundary helpers foldArcs / expandEdges are generic over it (decision C11).
 */
export type NumericVector = F32 | F64 | U32 | I32;

/**
 * Patch shape: every field optional AND accepting an explicit undefined (design section 12.1).
 * TypeScript's Partial adds `?` but not `| undefined`, so a consumer compiled with
 * exactOptionalPropertyTypes could not spread a partial object into it; Loose can be spread under
 * either setting.
 */
export type Loose<T> = { [K in keyof T]?: T[K] | undefined };

/**
 * How much of an untrusted snapshot is checked (design section 9.5): "none" is manifest shape only,
 * "structure" checks lengths, ranges and counts in O(n + m), "full" adds sortedness, pairing, NaN,
 * flag recomputation and the id bijection in O(m log d).
 */
export type ValidationLevel = "none" | "structure" | "full";

/**
 * What freeze() does with parallel edges (design section 6.5): "keep" stores them (the default),
 * "error" throws E_DUPLICATE_EDGE, and the merge policies keep one edge per (u, v) with the weight
 * of the first or last edge or the sum / min / max over the group. A merge policy rewrites the
 * builder's edge set.
 */
export type DuplicatePolicy = "keep" | "error" | "first" | "last" | "sum" | "min" | "max";

/** How the weights of merged edges combine in a derived graph or a merging freeze (design section 7.3). */
export type WeightReducer = "first" | "last" | "sum" | "min" | "max";

/**
 * How an attribute column of merged rows combines in simplified() / contract(): a WeightReducer,
 * the mean, the group size, or "drop" to omit the column (design section 7.3).
 */
export type ColumnReducer = WeightReducer | "mean" | "count" | "drop";

/**
 * Id coercion rule applied by importers before an id reaches the builder (design section 4.1):
 * "keep" leaves typed values alone, "canonical" turns canonical integer text into a number and keeps
 * everything else a string (injective on text), "string" and "number" force one type.
 */
export type IdCoercion = "keep" | "canonical" | "string" | "number";

// ============================================================ columns

/**
 * Column data types (design section 5.1). f32 / f64 / i32 / u32 / u8 are flat numeric buffers with
 * an optional stride; bool is bit-packed u32 words (the Arrow boolean layout, shared with validity
 * bitmaps and masks); dict is u32 codes plus a string dictionary; string is an Arrow Utf8 store; list
 * is offsets plus one non-list child column; json is one JS value per row.
 */
export type Dtype = "f32" | "f64" | "i32" | "u32" | "u8" | "bool" | "dict" | "string" | "list" | "json";

/** Every dtype a list column's child may have: lists nest one level only (design section 5.1). */
export type ScalarDtype = Exclude<Dtype, "list">;

/**
 * The table a column belongs to: nodes (nodeCount rows), edges (edgeCount rows), graph (1 row) or an
 * extension table with its own row count (design section 5.10).
 */
export type ColumnDomain = "node" | "edge" | "graph" | "extension";

/**
 * How a column reaches the GPU (design section 10.4): "direct" binds its data as is, "packed" binds
 * u8 lanes or bool bits inside u32 words, "convert" binds a cached f32 copy of an f64 column, and
 * "none" throws E_GPU_INELIGIBLE.
 */
export type GpuEligibility = "direct" | "packed" | "convert" | "none";

/**
 * The format-neutral roles the package and its importers recognise (design section 5.5): what a
 * column means, independent of its name, so consumers find "the position column" without per-format
 * knowledge and exporters map roles back to reserved fields. At most one column per role per table.
 */
export type KnownColumnRole =
    | "id"
    | "label"
    | "weight"
    | "capacity"
    | "position"
    | "color"
    | "size"
    | "shape"
    | "thickness"
    | "parent"
    | "parents"
    | "kind"
    | "labels"
    | "classes"
    | "start"
    | "end"
    | "timestamp"
    | "timestamps"
    | "spells"
    | "open"
    | "timeText"
    | "key"
    | "directed"
    | "pair"
    | "mutual"
    | "originalId"
    | "sourcePort"
    | "targetPort"
    | "idSpace"
    | "fixed"
    | "mass"
    | "subset"
    | "hidden"
    | "component"
    | "community"
    | "rank";

/**
 * A column role: the known roles plus any other string. The `(string & {})` intersection is the
 * deliberate lint-clean spelling that keeps the literals alive for autocomplete (design section
 * 12.1).
 */
export type ColumnRole = KnownColumnRole | (string & {});

/**
 * Where a column came from, so an exporter can restore the source declaration (design section 5.5).
 * Every field is null when unknown.
 */
export interface ColumnOrigin {
    /** Source format name: "gexf", "graphml", "gml", "csv", ... */
    readonly format: string | null;
    /** GEXF attribute id or GraphML key id. */
    readonly id: string | null;
    /** GEXF title / GraphML attr.name when it differs from the column name. */
    readonly title: string | null;
    /** Declared source type text: "liststring", "anyURI", "long", "date", "int", "real", "yfiles". */
    readonly type: string | null;
    /** Declaring namespace: "viz", "yfiles", "neo4j". */
    readonly namespace: string | null;
}

/** Input form of ColumnOrigin: every field optional and accepting an explicit undefined. */
export type ColumnOriginInput = Loose<ColumnOrigin>;

/**
 * Input shape for declaring a column through declareNodeColumn / declareEdgeColumn / set() (design
 * section 5.5). Optional fields take their documented defaults; ColumnMeta is the resolved output.
 */
export interface ColumnDecl {
    /** Unique within its table; case-sensitive; dotted names allowed (design section 5.6). */
    name: string;
    /** The column dtype (design section 5.1). */
    dtype: Dtype;
    /** Values per row for f32 / f64 / i32 / u32 / u8, 1..16; default 1 (design section 5.2). */
    components?: number | undefined;
    /** list only: the child dtype. */
    itemDtype?: ScalarDtype | undefined;
    /** list only: the child's components (spells: 2). */
    itemComponents?: number | undefined;
    /** Whether a validity bitmap may exist; default true for declared columns, false for typed bulk sets. */
    nullable?: boolean | undefined;
    /** Whether contents may be written in place on a snapshot (design section 5.8); default false. */
    mutable?: boolean | undefined;
    /** The column's role (design section 5.5); at most one column per role per table. */
    role?: ColumnRole | undefined;
    /** Marks a u32 column (or a list of u32) as holding indices that every remap rewrites (design section 5.11). */
    refersTo?: "node" | "edge" | undefined;
    /** Enforced at freeze over set rows (E_DUPLICATE_EDGE_ID / E_DUPLICATE_ID). */
    unique?: boolean | undefined;
    /** Declared default, a JSON value; non-finite numbers are allowed (design section 5.9). */
    default?: unknown;
    /** Value physically stored in unset rows; default: the declared default when representable, else 0 / "" / false. */
    fill?: number | string | boolean | undefined;
    /** Declared enumeration (GEXF options); for dict the initial dictionary. */
    options?: readonly unknown[] | undefined;
    /** Source declaration for exporters. */
    origin?: ColumnOriginInput | undefined;
    /** GEXF dynamic attribute: values live in a temporal extension table (design section 5.10). */
    dynamic?: boolean | undefined;
    /** Anything an importer wants to survive; must be JSON-serialisable. */
    extra?: Readonly<Record<string, unknown>> | undefined;
}

/** Patch form of ColumnDecl: every field, including name and dtype, optional. */
export type ColumnDeclPatch = Loose<ColumnDecl>;

/**
 * Output shape on every column: every field present, null for "none", never an optional property
 * (design sections 5.5 and 12.1), so it reads identically under every compiler flag.
 */
export interface ColumnMeta {
    /** Unique within its table; case-sensitive. */
    readonly name: string;
    /** The table the column belongs to. */
    readonly domain: ColumnDomain;
    /** The column dtype. */
    readonly dtype: Dtype;
    /** Values per row; > 1 only for f32 / f64 / i32 / u32 / u8. */
    readonly components: number;
    /** list only: the child dtype; null otherwise. */
    readonly itemDtype: ScalarDtype | null;
    /** list only: the child's components; null otherwise. */
    readonly itemComponents: number | null;
    /** Whether a validity bitmap may exist. */
    readonly nullable: boolean;
    /** Whether contents may be written in place on a snapshot (design section 5.8). */
    readonly mutable: boolean;
    /** The column's role, or null. */
    readonly role: ColumnRole | null;
    /** Which index space the values reference, or null. */
    readonly refersTo: "node" | "edge" | null;
    /** Whether uniqueness over set rows is enforced at freeze. */
    readonly unique: boolean;
    /** The declared default; undefined when none was declared (returned by value() for unset rows). */
    readonly default: unknown;
    /** The value physically stored in unset rows. */
    readonly fill: number | string | boolean;
    /** The declared enumeration, or null. */
    readonly options: readonly unknown[] | null;
    /** The source declaration, or null. */
    readonly origin: ColumnOrigin | null;
    /** Whether the column is a GEXF dynamic attribute. */
    readonly dynamic: boolean;
    /** Importer-owned JSON-serialisable extras; an empty object when none. */
    readonly extra: Readonly<Record<string, unknown>>;
}

/**
 * Column data plus a declaration patch, the long form of a column value in *Input.nodeColumns / edgeColumns and
 * withColumns().
 */
export interface ColumnInput {
    /** The values: a typed array for the numeric / bool dtypes, a JS array for string / list / json. */
    readonly data: TypedArrayData | readonly unknown[];
    /** The declaration; name and dtype may be omitted and inferred. */
    readonly decl: ColumnDeclPatch;
}

/**
 * Options of AttributeTable.set() (design section 5.7): replaceRole removes the previous holder of
 * the role instead of throwing E_DUPLICATE_ROLE; adopt "strict" throws E_COLUMN_ALIGNMENT instead of
 * copying a u8 array from which no zero-copy padded u32 view is constructible.
 */
export interface SetOptions {
    /** Remove the previous holder of the declared role instead of throwing E_DUPLICATE_ROLE. */
    readonly replaceRole?: boolean | undefined;
    /** "copy" (default) copies an unadoptable u8 array; "strict" throws E_COLUMN_ALIGNMENT instead. */
    readonly adopt?: "copy" | "strict" | undefined;
}

/**
 * What value(row) returns for a set row of each dtype (design section 5.3): a number (or a subarray
 * when components > 1) for the numeric dtypes, a boolean for bool, a string for dict and string, a
 * read-only array for list, and anything for json.
 */
export type DtypeValue<D extends Dtype> = D extends "f32" | "f64" | "i32" | "u32" | "u8"
    ? number | ArrayLike<number>
    : D extends "bool"
      ? boolean
      : D extends "dict" | "string"
        ? string
        : D extends "list"
          ? readonly unknown[]
          : unknown;

/**
 * Members shared by every column (design sections 5.3, 5.7 and 5.8). A column is immutable unless
 * meta.mutable is true; the column SET of a table is always a mutable side table.
 */
export interface ColumnBase<D extends Dtype> {
    /** The dtype discriminator; narrowing on it selects the concrete column interface. */
    readonly dtype: D;
    /** Resolved metadata; every field present, null for none. */
    readonly meta: ColumnMeta;
    /** Number of rows. */
    readonly length: number;
    /** Validity bitmap: LSB-first words, ceil(length / 32); null means every row is set (design section 5.3). */
    readonly validity: U32 | null;
    /** Number of unset rows; kept exact by the setters. */
    readonly nullCount: number;
    /** GPU eligibility derived from the dtype (design section 10.4). */
    readonly gpu: GpuEligibility;
    /** Bytes of data plus validity. */
    readonly byteLength: number;
    /** byteLength, or roundUp(byteLength, 4) for u8: the size of paddedU32View() (design section 5.7). */
    readonly paddedByteLength: number;
    /** Bumped by markDirty(); lets a consumer cache derived data per column contents. */
    readonly version: number;
    /**
     * Whether row `row` holds a value.
     * @param row - the row index
     * @returns true when the row is set (validity null or bit set)
     */
    isSet(row: number): boolean;
    /**
     * Typed read honouring meta.default; components > 1 returns a subarray view.
     * @param row - the row index; E_INDEX_RANGE when out of range
     * @returns the value, the declared default for an unset row, or undefined for an unset row without a default
     */
    value(row: number): DtypeValue<D> | undefined;
    /**
     * This column when fill equals the default; otherwise a cached copy with the default written into
     * unset rows (invalidated by markDirty()).
     * @returns a column whose data holds the default in every unset row; E_NO_DEFAULT when none is declared
     */
    materializeDefault(): ColumnOf<D>;
    /**
     * The u32 words a GPU binds: for u8 a zero-copy padded view over the column's byte range, for u32
     * and bool the data itself.
     * @returns the padded u32 view; E_GPU_INELIGIBLE for the other dtypes
     */
    paddedU32View(): U32;
    /**
     * Invalidate the cached gpuView() f32 copy and materializeDefault() copy after in-place writes and
     * bump `version` (design section 5.8). Mutable columns only; E_COLUMN_IMMUTABLE otherwise.
     */
    markDirty(): void;
    /**
     * The validity bitmap itself, for owners that set rows one at a time. Mutable columns only.
     * @returns the bitmap words, or null when every row is set
     */
    mutableValidity(): U32 | null;
    /** Mark every row set: drops the bitmap and sets nullCount to 0. Mutable columns only. */
    setAll(): void;
    /**
     * A column over the row range [start, end): zero-copy except packed stores at unaligned starts
     * (design section 5.7).
     * @param start - first row
     * @param end - one past the last row
     * @returns a column of the same dtype over the range
     */
    slice(start: number, end: number): ColumnOf<D>;
    /**
     * A deep copy of the column.
     * @returns a new column of the same dtype with copied buffers
     */
    clone(): ColumnOf<D>;
}

/** f32 column: Float32Array(rows * components); GPU eligibility "direct". */
export interface F32Column extends ColumnBase<"f32"> {
    /** The values, row-major, components interleaved. */
    readonly data: F32;
    /**
     * The data array for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): F32;
}

/** f64 column: Float64Array(rows * components); GPU eligibility "convert" (cached f32 copy). */
export interface F64Column extends ColumnBase<"f64"> {
    /** The values, row-major, components interleaved. */
    readonly data: F64;
    /**
     * The data array for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): F64;
}

/** i32 column: Int32Array(rows * components); GPU eligibility "direct". */
export interface I32Column extends ColumnBase<"i32"> {
    /** The values, row-major, components interleaved. */
    readonly data: I32;
    /**
     * The data array for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): I32;
}

/** u32 column: Uint32Array(rows * components); index references, labels, partitions; GPU eligibility "direct". */
export interface U32Column extends ColumnBase<"u32"> {
    /** The values, row-major, components interleaved; INVALID_INDEX marks an unset refersTo entry. */
    readonly data: U32;
    /**
     * The data array for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): U32;
}

/**
 * u8 column: Uint8Array(rows * components) over a store with a constructible padded u32 view; GPU eligibility "packed".
 */
export interface U8Column extends ColumnBase<"u8"> {
    /** The values, row-major, components interleaved. */
    readonly data: U8;
    /**
     * The data array for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): U8;
}

/** bool column: bit-packed, ceil(rows / 32) u32 words, LSB-first (the validity layout); GPU eligibility "packed". */
export interface BoolColumn extends ColumnBase<"bool"> {
    /** The packed words; row r is `(data[r >>> 5] >>> (r & 31)) & 1`. */
    readonly data: U32;
    /**
     * The packed words for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `data`
     */
    mutableData(): U32;
}

/** dict column: u32 codes into a string dictionary in first-seen (or declared) order (design section 5.4). */
export interface DictColumn extends ColumnBase<"dict"> {
    /** One code per row, `< dictionary.length` for every set row. */
    readonly codes: U32;
    /** The dictionary; codes are dense 0..length-1. */
    readonly dictionary: readonly string[];
    /**
     * The code of a dictionary value, via a lazily built reverse map.
     * @param value - the string to look up
     * @returns its code, or INVALID_INDEX when absent
     */
    codeOf(value: string): number;
    /**
     * The codes for in-place writes; mutable columns only (E_COLUMN_IMMUTABLE otherwise).
     * @returns the same typed array as `codes`
     */
    mutableData(): U32;
}

/** string column: an Arrow Utf8 store (offsets + utf8) with a lazily decoded per-row cache (design section 5.1). */
export interface StringColumn extends ColumnBase<"string"> {
    /** rows + 1 offsets into utf8, non-decreasing; materialised lazily from the decoded cache when needed. */
    readonly offsets: U32;
    /** The UTF-8 bytes of every row, concatenated. */
    readonly utf8: U8;
    /**
     * The decoded string of one row, cached per row.
     * @param row - the row index
     * @returns the string (empty for an unset row)
     */
    valueAt(row: number): string;
    /**
     * Decode every row in one pass; not cached.
     * @returns a fresh array of rows strings
     */
    decodeAll(): string[];
}

/** list column: rows + 1 offsets into one non-nullable child column of any dtype except list (design section 5.1). */
export interface ListColumn extends ColumnBase<"list"> {
    /** rows + 1 offsets into the child, non-decreasing; unset and empty rows both have equal offsets. */
    readonly offsets: U32;
    /** The child column holding every item of every row; never nullable, never a list. */
    readonly child: Exclude<Column, ListColumn>;
    /**
     * The items of one row, read through the child.
     * @param row - the row index
     * @returns the row's items; narrow the element type through child.dtype
     */
    sliceOf(row: number): readonly unknown[];
}

/** json column: one JS value per row, serialised as JSON text on the wire (design section 5.1). */
export interface JsonColumn extends ColumnBase<"json"> {
    /** One value per row; undefined in unset rows. */
    readonly values: readonly unknown[];
}

/** Every column shape, discriminated by `dtype`. */
export type Column =
    | F32Column
    | F64Column
    | I32Column
    | U32Column
    | U8Column
    | BoolColumn
    | DictColumn
    | StringColumn
    | ListColumn
    | JsonColumn;

/** The column interface of one dtype: `ColumnOf<"f32">` is F32Column. */
export type ColumnOf<D extends Dtype> = Extract<Column, { dtype: D }>;

/**
 * A set of columns with a fixed row count (design section 5.7): `nodes` (nodeCount rows), `edges`
 * (edgeCount rows), `graph` (1 row) and extension tables. The column SET is a mutable side table of
 * the snapshot (set / remove / rename at any time); the row count is fixed (invariant I17). "Absent"
 * is always null on this surface; undefined only ever means "unset row" from value(). Iteration
 * yields columns in declaration order.
 *
 * Instance contract of the AttributeTable class in src/columns/table.ts, which implements it; the
 * public barrel exports the class and every public type names the class.
 */
export interface AttributeTableContract extends Iterable<Column> {
    /** The table the columns belong to. */
    readonly domain: ColumnDomain;
    /** Number of rows of every column; fixed for the life of the table. */
    readonly rowCount: number;
    /**
     * Column names in declaration order.
     * @returns the names
     */
    names(): readonly string[];
    /**
     * Whether a column exists.
     * @param name - the column name
     * @returns true when present
     */
    has(name: string): boolean;
    /**
     * Total lookup by name.
     * @param name - the column name
     * @returns the column, or null when absent
     */
    get(name: string): Column | null;
    /**
     * Checked lookup by name.
     * @param name - the column name
     * @returns the column; E_UNKNOWN_COLUMN when absent
     */
    require(name: string): Column;
    /**
     * Total lookup by name and dtype.
     * @param name - the column name
     * @param dtype - the expected dtype
     * @returns the column, or null when absent or of another dtype
     */
    typed<D extends Dtype>(name: string, dtype: D): ColumnOf<D> | null;
    /**
     * Checked lookup by name and dtype.
     * @param name - the column name
     * @param dtype - the expected dtype
     * @returns the column; E_UNKNOWN_COLUMN when absent, E_COLUMN_TYPE when of another dtype
     */
    requireTyped<D extends Dtype>(name: string, dtype: D): ColumnOf<D>;
    /**
     * The column holding a role (at most one per table).
     * @param role - the role
     * @returns the column, or null when no column has the role
     */
    byRole(role: ColumnRole): Column | null;
    /**
     * Typed read of one cell honouring the column's declared default.
     * @param name - the column name
     * @param row - the row index; E_INDEX_RANGE when out of range
     * @returns the value, the default for an unset row, or undefined for an unset row without a default
     */
    value(name: string, row: number): unknown;
    /**
     * Whether one cell holds a value.
     * @param name - the column name
     * @param row - the row index
     * @returns true when set
     */
    isSet(name: string, row: number): boolean;
    /**
     * Attach a column, adopting a typed array by reference after validating its length
     * (E_COLUMN_LENGTH), or move an existing Column object in. Replaces a column of the same name;
     * E_DUPLICATE_ROLE when the role is taken and replaceRole is not set (design section 5.7).
     * @param name - the column name
     * @param data - a Column, a typed array, or a JS array for string / list / json
     * @param decl - declaration fields to apply
     * @param opts - role replacement and u8 adoption options
     * @returns the attached column (`column.data !== data` only when a u8 array was copied)
     */
    set(
        name: string,
        data: Column | TypedArrayData | readonly unknown[],
        decl?: ColumnDeclPatch,
        opts?: SetOptions,
    ): Column;
    /**
     * Remove a column.
     * @param name - the column name
     * @returns true when a column was removed
     */
    remove(name: string): boolean;
    /**
     * Rename a column in place.
     * @param from - the current name; E_UNKNOWN_COLUMN when absent
     * @param to - the new name
     */
    rename(from: string, to: string): void;
    /**
     * The array a GPU binds for a column (design section 10.4): its own data for the direct dtypes,
     * the padded u32 view for u8, the packed words for bool, the codes for dict, and a cached f32 copy
     * for f64.
     * @param name - the column name
     * @returns the bindable array; E_GPU_INELIGIBLE for string / list / json
     */
    gpuView(name: string): U32 | I32 | F32;
    /**
     * A new table with the same columns (the Column objects are shared, the set is independent).
     * @returns the cloned table
     */
    clone(): AttributeTable;
    /**
     * Iterate the columns in declaration order.
     * @returns an iterator over the columns
     */
    [Symbol.iterator](): IterableIterator<Column>;
}

/**
 * Graph-level metadata (design section 5.9): what a file header declares, kept so exporters can
 * re-emit it. Every field is null when unknown; `extra` must hold JSON values (non-finite numbers are
 * carried on the wire through the tagged encoding of section 5.9).
 */
export interface GraphMeta {
    /** Graph name. */
    readonly name: string | null;
    /** Free-text description. */
    readonly description: string | null;
    /** Creator (GEXF meta creator). */
    readonly creator: string | null;
    /** Creation time, ISO-8601. */
    readonly created: string | null;
    /** Modification time, ISO-8601. */
    readonly modified: string | null;
    /** Keywords. */
    readonly keywords: readonly string[];
    /** Source format name, e.g. "gexf". */
    readonly sourceFormat: string | null;
    /** Source format version, e.g. "1.3". */
    readonly sourceVersion: string | null;
    /** GEXF idtype, for exporters. */
    readonly idType: "string" | "integer" | "mixed" | null;
    /** GEXF timeformat. */
    readonly timeFormat: "integer" | "double" | "date" | "dateTime" | null;
    /** GEXF 1.3 timerepresentation. */
    readonly timeRepresentation: "interval" | "timestamp" | null;
    /** GEXF graph mode. */
    readonly mode: "static" | "dynamic" | "slice" | null;
    /** The node-link / graphology "multigraph" flag as declared, independent of flags.multigraph. */
    readonly declaredMultigraph: boolean | null;
    /** Declared source type of the weight (design section 3.7). */
    readonly weightOrigin: ColumnOrigin | null;
    /** JSON-serialisable extras; reserved per-format keys in design section 8.5. */
    readonly extra: Readonly<Record<string, unknown>>;
}

/** Patch form of GraphMeta for setMeta() and the *Input.meta fields. */
export type GraphMetaPatch = Loose<GraphMeta>;

// ============================================================ masks (packed bitmaps; not a snapshot contract)

/**
 * Packed node bitmap: ceil(n / 32) words, bit i set means node i is included; the same layout as
 * validity bitmaps and bool columns (design section 7.4). No kernel or algorithm is required to
 * honour a mask; it is an input to inducedSubgraph() and an algorithm-internal scratch shape.
 */
export type NodeMask = U32;

/**
 * Packed edge bitmap over LOGICAL edges, ceil(edgeCount / 32) words; the input of filterEdges() (design section 7.4).
 */
export type EdgeMask = U32;
