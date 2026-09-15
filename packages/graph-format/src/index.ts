/**
 * The public barrel of @graphty/graph-format: a frozen compressed-sparse-row graph data format over
 * typed arrays, with a mutable builder, columnar typed attributes and an id map kept outside the CSR.
 * This is the only public barrel; it exports exactly the surface of design section 12.2, as named
 * exports (no star re-exports, nothing from src/types/internal.ts, no helper that a sibling module
 * exports for the package's own use).
 */

// ============================================================ constants and errors
export { FORMAT_VERSION, INVALID_INDEX, MAX_COUNT, SNAPSHOT_BRAND } from "./constants.js";
export { GraphFormatError, type GraphFormatErrorCode } from "./errors.js";

// ============================================================ classes
export { GraphBuilder } from "./builder/graph-builder.js";
export { AttributeTable } from "./columns/table.js";
export { NodeIdMap } from "./ids/node-id-map.js";
export { GraphSnapshot } from "./snapshot/graph-snapshot.js";

// ============================================================ functions
export { gpuEligibility } from "./columns/column.js";
export { gatherArray, gatherColumn, remapArray, remapColumn, scatterArray, withComponents } from "./columns/remap.js";
export { fromCsr } from "./populate/from-csr.js";
export { fromEdgeArrays } from "./populate/from-edge-arrays.js";
export { fromRecords } from "./populate/from-records.js";
export { renumberPartition } from "./snapshot/derived.js";
export { equalsTopology, isGraphSnapshot } from "./snapshot/graph-snapshot.js";
export { expandEdges, foldArcs } from "./snapshot/views.js";
export { makeMask, maskCount, maskSet, maskTest, maskToIndices } from "./util/mask.js";
export { paddedU32View } from "./util/typed-array.js";
export { fromByteChunks, fromBytes } from "./wire/bytes.js";
export { fromWire } from "./wire/from-wire.js";

// ============================================================ types: builder
export type {
    BuilderOptionsPatch,
    ColumnHandle,
    ExtensionHandle,
    FreezeOptions,
    FreezeReport,
    GraphBuilderOptions,
    GraphSink,
    ResolvedBuilderOptions,
    SetDirectedOptions,
} from "./types/builder.js";

// ============================================================ types: scalars, columns, tables, masks
export type {
    ArcIndex,
    BoolColumn,
    Column,
    ColumnBase,
    ColumnDecl,
    ColumnDeclPatch,
    ColumnDomain,
    ColumnInput,
    ColumnMeta,
    ColumnOf,
    ColumnOrigin,
    ColumnOriginInput,
    ColumnReducer,
    ColumnRole,
    DictColumn,
    Dtype,
    DtypeValue,
    DuplicatePolicy,
    EdgeId,
    EdgeIndex,
    EdgeMask,
    F32,
    F32Column,
    F64,
    F64Column,
    GpuEligibility,
    GraphMeta,
    GraphMetaPatch,
    I32,
    I32Column,
    IdCoercion,
    JsonColumn,
    KnownColumnRole,
    ListColumn,
    Loose,
    NodeId,
    NodeIndex,
    NodeMask,
    NumericVector,
    ScalarDtype,
    SetOptions,
    StringColumn,
    TypedArrayData,
    U8,
    U8Column,
    U32,
    U32Column,
    ValidationLevel,
    WeightReducer,
} from "./types/columns.js";

// ============================================================ types: id map, flags, arena, views, derived graphs, factory inputs
export type {
    AdjacencyView,
    ArenaLayout,
    ArenaSegment,
    ByteLengthOptions,
    ContractOptions,
    CooView,
    CoreArrayName,
    CsrInput,
    DegreeOrderOptions,
    DegreeOrderView,
    DerivedGraph,
    EdgeArraysInput,
    EdgeListView,
    FlagClaims,
    FromCsrOptions,
    NodeIdMapKind,
    RecordsInput,
    ReverseView,
    SimplifyOptions,
    SnapshotFlags,
    ToUndirectedOptions,
    ValidateOptions,
    ViewName,
} from "./types/snapshot.js";

// ============================================================ types: wire form
export type {
    FromWireOptions,
    ToBytesOptions,
    ToWireOptions,
    WireArena,
    WireBufferRef,
    WireColumn,
    WireDtype,
    WireIdMap,
    WireManifest,
    WireSnapshot,
    WireUtf8,
} from "./types/wire.js";
