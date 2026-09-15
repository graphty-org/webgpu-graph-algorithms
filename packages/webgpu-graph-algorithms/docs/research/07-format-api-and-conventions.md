# 07. graph-format API consumed by the GPU package, package skeleton, and conventions

Status: research note for the WebGPU plan. Everything below about existing code was read from the
files cited (paths relative to `/home/apowers/Projects/webgpu-graph-algorithms/` unless absolute);
everything about the design cites `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
(hereafter "design") by section and line. Plain ASCII throughout.

Contents

1. Verified API reference (what `@graphty/webgpu-graph-algorithms` will call)
2. Where the implementation differs from, or refines, design section 10
3. Consumption rules the GPU package must follow (derived from 1 and 2)
4. Package skeleton for `packages/webgpu-graph-algorithms/`
5. Scaffold triage: delete / keep / move
6. Reusable test and benchmark assets
7. Conventions checklist
8. Sources

---

## 1. Verified API reference

The public barrel is `packages/graph-format/src/index.ts` (139 lines). It exports exactly 126 names
with explicit named exports, no star re-exports (`packages/STATUS.md` line 20; the value list is
pinned by `packages/graph-format/test/index.test.ts` lines 6-40). The names the GPU package needs are
listed below with the source of truth for each signature.

### 1.1 Constants and the error class

From `packages/graph-format/src/constants.ts` (public four; re-exported at `src/index.ts` line 10):

| Name | Value | Line | Meaning for the GPU package |
| --- | --- | --- | --- |
| `INVALID_INDEX` | `0xffffffff` | 16 | The only "no index" sentinel; `0xFFFFFFFFu` in WGSL. Never appears in `rowPtr`, `colIdx`, `arcToEdge`, `edgeToArc` or any view array (I2). Free as a sentinel in every `array<u32>` result (parents, labels). |
| `MAX_COUNT` | `0xfffffffe` | 22 | Upper bound of `nodeCount`, `edgeCount`, `arcCount`. Guarantees the sentinel never collides. Also: arc indices may exceed 2^31, so NEVER apply JS bitwise operators to arc indices (design 10.6 lines 2497-2502; `packages/graph-format/CLAUDE.md` invariant I3 line). |
| `FORMAT_VERSION` | `1` | 30 | `isGraphSnapshot()` compares it. |
| `SNAPSHOT_BRAND` | `Symbol.for("@graphty/graph-format/snapshot")` | 37 | Brand read by `isGraphSnapshot()`; structural, survives two package copies. |

NOT exported (internal, `@internal`): `ALIGNMENT = 256` (`constants.ts` line 45). The GPU package must
NOT import it; read `snapshot.arena.alignment` (typed as the literal `256`) or hard-code 256 with a
comment citing design 10.3.

`GraphFormatError` (`packages/graph-format/src/errors.ts` lines 96-120): `readonly code:
GraphFormatErrorCode`, `readonly details: Readonly<Record<string, unknown>>` (frozen shallow copy),
`name === "GraphFormatError"`. Codes the GPU package will see from the format (lines 56-86):
`E_GPU_INELIGIBLE` (`gpuView` on string / list / json), `E_DIRECTED` (`mate()` on a directed
snapshot), `E_DETACHED` (core accessor after a consuming transfer), `E_COLUMN_LENGTH` (`foldArcs` /
`expandEdges` / `scatterArray` length mismatch), `E_UNSUPPORTED` (unknown enum option, e.g.
`foldArcs(.., "bogus")`, `degreeOrder({ of: "bogus" })`), `E_UNKNOWN_COLUMN` / `E_COLUMN_TYPE`
(`require` / `requireTyped`), `E_COLUMN_IMMUTABLE` (`markDirty` / `mutableData` on an immutable
column), `E_MASK_LENGTH`. The GPU package should define its OWN error class (see section 4) and never
throw `GraphFormatError` itself; `E_IMPORT` is reserved for graph-io (line 54), so there is no
reserved code for a GPU package.

### 1.2 Typed-array aliases (`packages/graph-format/src/types/columns.ts` lines 59-80)

```ts
export type U32 = Uint32Array<ArrayBuffer>;    // line 59
export type I32 = Int32Array<ArrayBuffer>;     // 62
export type F32 = Float32Array<ArrayBuffer>;   // 65
export type F64 = Float64Array<ArrayBuffer>;   // 68
export type U8  = Uint8Array<ArrayBuffer>;     // 71
export type TypedArrayData = U32 | I32 | F32 | F64 | U8;   // 74
export type NumericVector = F32 | F64 | U32 | I32;         // 80 (foldArcs / expandEdges generic bound)
export type NodeMask = U32;   // line 722: ceil(n / 32) words, bit i = node i included
export type EdgeMask = U32;   // line 727: over LOGICAL edges
```

The `<ArrayBuffer>` type parameter is load-bearing: `packages/graph-format/test/types/typed-arrays.test-d.ts`
lines 58-74 prove that `snapshot.colIdx`, `snapshot.rowPtr`, `snapshot.nodes.gpuView("position")`,
`snapshot.colIdx.subarray(start, end)`, `new Uint8Array(arena.buffer, arena.byteOffset,
arena.hotByteLength)`, `snapshot.reverse().rowPtr` and `snapshot.edgeList().src` are each accepted
by `GPUQueue.writeBuffer(buf, 0, x)` from `@webgpu/types` 0.1.72 with no cast. The GPU package's own
result arrays and scratch must be typed the same way (`new Uint32Array(n)` infers
`Uint32Array<ArrayBuffer>` under TS 5.7+), and it must never hand out `Uint32Array<ArrayBufferLike>`.

### 1.3 GraphSnapshot core (`packages/graph-format/src/types/snapshot.ts` lines 425-484; class at `src/snapshot/graph-snapshot.ts` lines 311-475)

`GraphSnapshotContract extends AdjacencyView` (line 425). `AdjacencyView` (lines 235-252) is the
row-walking subset implemented by BOTH `GraphSnapshot` and `ReverseView`, so a kernel that only walks
rows can take either.

| Member | Type | Invariant / note (line in snapshot.ts) |
| --- | --- | --- |
| `serial` | `number` | process-unique identity of the CORE, shared by `withColumns()` snapshots (429). Use for cache keys on the CORE only; two snapshots with equal `serial` share `rowPtr` etc. |
| `label` | `string \| null` | debugging label (431). |
| `formatVersion` | `1` | (433) |
| `directed` | `boolean` | no tri-state (435). |
| `nodeCount` | `number` | n <= MAX_COUNT (437). |
| `edgeCount` | `number` | logical edges; every edge column has `edgeCount` rows (439). |
| `arcCount` | `number` | `colIdx.length`; `edgeCount` when directed, `2 * edgeCount - selfLoopCount` when undirected (441). |
| `selfLoopCount` | `number` | (443) |
| `rowPtr` | `U32` | `nodeCount + 1` entries, `rowPtr[0] === 0`, non-decreasing, `rowPtr[n] === arcCount` (I1). Plain data property (graph-snapshot.ts line 11-12: "hot loops pay nothing"). |
| `colIdx` | `U32` | `arcCount` entries, `< nodeCount`, sorted within each row, ties by ascending `arcToEdge` (I2, I4); LENGTH 0 when `arcCount === 0` (447). Plain data property. |
| `weights` | `F32 \| null` | `arcCount` f32; null when unweighted (every weight 1); both arcs of an undirected edge carry the same value (I7, I8: never NaN) (452). Plain data property. |
| `arcToEdge` | `U32` | GETTER (graph-snapshot.ts lines 448-456): MATERIALISES an identity permutation of `arcCount` entries OUTSIDE the arena on first access when `flags.arcToEdgeIsIdentity`. Test the flag first. |
| `edgeToArc` | `U32` | GETTER (lines 463-471): same materialisation rule, `edgeCount` entries. |
| `flags` | `SnapshotFlags` | frozen (section 1.4). |
| `ids` | `NodeIdMap` | section 1.8. |
| `nodes`, `edges`, `graph` | `AttributeTable` | rowCount n / edgeCount / 1 (I12, I13: edge columns are indexed by LOGICAL edge, never by arc). |
| `extensions` | `ReadonlyMap<string, AttributeTable>` | (471) |
| `meta` | `GraphMeta` | (473) |
| `arena` | `ArenaLayout \| null` | null when arrays were adopted from separate buffers (477); section 1.5. |
| `detached` | `boolean` | derived: `rowPtr.length === 0` after a consuming transfer (479; class line 438-440). Every core accessor throws `E_DETACHED` afterwards. |

Queries the GPU package may use on the CPU side (lines 483-560): `outArcs(u)` (allocates a tuple; hot
loops read `rowPtr` directly), `outDegreeOf(u)`, `findArc(u, v)` (binary search, `INVALID_INDEX` when
absent), `hasArc`, `arcsBetween`, `multiplicity`, `arcSource(a)` (O(1) after `coo()`, else binary
search on `rowPtr`), `edgeSource(e)`, `edgeTarget(e)`, `edgeIndexOf(id)`.

Memory / identity (lines 780-830): `byteLength(options?: { views?, columns?, ids? })` -- core only by
default, identity permutations count zero even when materialised (class line 948); `contentHash()`
(16 hex chars, lazy, cached); `transferables()`; `validate(options?)`.

`isGraphSnapshot(x: unknown): x is GraphSnapshot` (graph-snapshot.ts lines 1351-1357: brand plus
`formatVersion === 1`, never instanceof). `equalsTopology(a, b): boolean` (lines 1386-1420) compares
counts, `rowPtr`, `colIdx`, `weights`, the permutations (identity compared without materialising) and
every id.

### 1.4 Flags (`types/snapshot.ts` lines 155-181)

```ts
export interface SnapshotFlags {
    readonly multigraph: boolean;          // some row has two arcs with equal colIdx
    readonly hasSelfLoops: boolean;        // selfLoopCount > 0
    readonly arcToEdgeIsIdentity: boolean; // directed && arcToEdge[a] === a for all a; ALWAYS false when !directed
    readonly weighted: boolean;            // weights !== null
    readonly allWeightsOne: boolean;       // weights === null || every value === 1 -> SSSP degrades to BFS
    readonly nonNegativeWeights: boolean;  // weights === null || every value >= 0 -> Dijkstra / delta-stepping legal
    readonly finiteWeights: boolean;       // weights === null || every value finite
}
```

Every flag is a truthful predicate over the arrays, computed at freeze or by validation, never
guessed (I9). Caveat recorded in the doc comment (lines 155-161) and in `packages/STATUS.md` lines
459-465: the three weight flags describe the f32 ARC array; an f64 role-"weight" shadow column can
disagree (`1 + 2^-30` rounds to 1). A GPU kernel binds the f32 arc array, so the flags are exactly
right for it; a GPU package must NOT substitute the shadow column and still branch on the flags.

### 1.5 ArenaLayout (`types/snapshot.ts` lines 183-224; producer `src/builder/arena.ts` lines 96-144)

```ts
export type CoreArrayName = "rowPtr" | "colIdx" | "weights" | "arcToEdge" | "edgeToArc";   // hot-to-cold order
export interface ArenaSegment { readonly byteOffset: number; readonly byteLength: number; }  // ABSOLUTE offset in buffer; unpadded length
export interface ArenaLayout {
    readonly buffer: ArrayBuffer;
    readonly byteOffset: number;     // 0 for builder output; bytes.byteOffset + B for a fromBytes container
    readonly byteLength: number;     // end of the LAST non-empty segment (no trailing padding: STATUS.md line 422)
    readonly alignment: 256;
    readonly segments: Readonly<Record<CoreArrayName, ArenaSegment | null>>;  // null = absent, zero-length, or identity
    readonly hotByteLength: number;  // end of weights (or colIdx when unweighted) RELATIVE to byteOffset
}
```

Verified facts (arena.ts lines 107-118): `hotByteLength` is set from the last non-null segment among
indices 0..2 (rowPtr, colIdx, weights); segments are `Object.freeze`d; the arena object is frozen;
identity permutations (`shape.identity`) and `weights === null` get `null` segments and zero bytes.
`ArenaSegment.byteOffset` is ABSOLUTE in `buffer`, so the storage-binding offset is
`segment.byteOffset - arena.byteOffset` (design 10.3 lines 2426-2428; exercised on a real device by
`packages/graph-format/test/audit/gpu-upload.test.ts` lines 295-364 and, for `arena.byteOffset !== 0`
after `fromBytes` at an 8-byte offset, lines 616-666).

Worked numbers pinned by `test/audit/gpu-contract.test.ts` lines 207-315: undirected weighted n = 100k,
A = 2M, E = 1M gives segments at 0 / 400,128 / 8,400,128 / 16,400,128 / 24,400,128 and
`byteLength` 28,400,128 with `hotByteLength` 16,400,128; directed with identity permutations is
8,400,128 total.

Which producers give `arena !== null`: `freeze()` with the default `arena: true` (`FreezeOptions.arena`,
`types/builder.ts` line 77), `fromEdgeArrays` (goes through freeze), `fromBytes` at `byteOffset % 8 === 0`,
`fromWire` when the manifest carries an arena descriptor, `fromCsr` only when `detectArena` finds the
five arrays in hot-to-cold order inside one aligned buffer (`src/populate/from-csr.ts` lines 275-300;
`packages/CONFORMANCE.md` line 412). `arena === null` for: `fromCsr` on separate arrays, `transpose()`
(adopts the reverse arrays as the core; CONFORMANCE.md line 269, 407), `fromByteChunks`.

### 1.6 Views (`types/snapshot.ts` lines 226-330 and 562-660; implementations `src/snapshot/views.ts`)

All views are pure functions of the core, memoised once per snapshot instance and SHARED: writing
into a view is a contract violation (I17; class doc lines 300-310). Aliasing is deliberate and is
what makes an upload cache keyed on the ARRAY OBJECT upload each distinct array once.

| Call | Returns | Shape, aliasing, cost (views.ts line) |
| --- | --- | --- |
| `reverse()` | `ReverseView` = `AdjacencyView & { fwdArc: U32 }` | Directed: fresh `rowPtr(n+1)`, `colIdx(A)` (= forward sources, sorted per reverse row), `weights(A)` gathered, `fwdArc(A)` (reverse arc -> forward arc) all materialised at once (`computeReverse` 197-229). Undirected: `rowPtr`, `colIdx`, `weights` ARE the forward array objects (I7); `fwdArc` is an identity permutation ALLOCATED ON FIRST READ of the getter (lines 116-119) -- treat it like `arcToEdge`: do not touch it when `!directed`. `reverse().arcToEdge` is a getter too (line 121+): for a directed identity snapshot it aliases `fwdArc` (zero bytes); otherwise it gathers `source.arcToEdge[fwdArc[k]]`, which materialises the identity on the source if the flag is set. |
| `coo()` | `CooView { src, dst, arcToEdge, weights }` | `src` is the only new array (`expandRowPtr`, A entries); `dst` ALIASES `colIdx`; `weights` aliases; `arcToEdge` is a getter reaching `snapshot.arcToEdge` (materialises identity) -- read the flag first. Per-ARC edge-parallel kernels (both directions of an undirected edge). |
| `edgeList()` | `EdgeListView { src, dst, arc, weights }` | E entries each: declared orientation of every logical edge; `arc` aliases `edgeToArc` (getter); `weights` gathered through `edgeToArc`, ALIASED to `weights` when the permutation is the identity (types line 308). Each-edge-once kernels; correct on directed and undirected alike. |
| `outDegree()` | `U32(n)` | `rowPtr` differences; self-loop counted once. |
| `inDegree()` | `U32(n)` | via `reverse().rowPtr`; the SAME OBJECT as `outDegree()` when undirected. |
| `degree()` | `U32(n)` | in + out (directed), out + self-loops (undirected). |
| `weightedOutDegree()` / `weightedInDegree()` / `weightedDegree()` | `F64(n)` | f64 on purpose; NOT a GPU upload (design 10.1 line 2344; `gpu-contract.test.ts` line 644). May be 0 for a node with out-arcs. The GPU package computes its normaliser on the device. |
| `selfLoopWeight()` | `F64(n)` | |
| `totalWeight()` | `number` | never cached on the wire. |
| `selfLoopArcs()` | `U32(selfLoopCount)` | |
| `selfLoopsPerNode()` | `U32(n)` | |
| `selfLoopsAt(u)` | `number` | O(log d) point query. |
| `mate()` | `U32(A)` | undirected only, `E_DIRECTED` otherwise; the arc holding the opposite orientation (a self-loop maps to itself). |
| `degreeOrder(options?: { of?: "forward" \| "reverse" })` | `DegreeOrderView { perm: U32(n), segmentOffsets: U32(5) }` | counting sort by DESCENDING degree, ties ascending node index; `segmentOffsets = [0, hiEnd, midEnd, lowEnd, n]` for thresholds `DEGREE_TIER_HIGH = 1024`, `DEGREE_TIER_MID = 32`, low = 1 (views.ts lines 41-45, 568-609); degree-0 nodes are the trailing segment `[lowEnd, n)`. `"reverse"` orders by in-degree (pull kernels). Both variants cached; on an undirected snapshot both are the same object (CONFORMANCE.md line 257). Frozen object. `segmentOffsets` is read on the CPU to size dispatches (a 20-byte `array<u32,5>` is not a legal uniform layout, design 10.1 line 2346). |
| `isSymmetric()` | `boolean` | |
| `prepare(views: readonly ViewName[]): this` | | eager materialisation (also `FreezeOptions.prepare`). |
| `dropCaches(): void` | | drops EVERY cached view, every materialised identity permutation (`arcToEdge` / `edgeToArc` go back to lazy when the flag is set), and every cached `gpuView()` f32 copy of an f64 column (class lines 797-813). After it, the array OBJECTS of views change: an upload cache keyed on the object must not assume it survives `dropCaches()`. |
| `cachedViews(): readonly ViewName[]` | | in the fixed `ViewName` order. |

`ViewName` (types lines 226-244): `"reverse" | "coo" | "edgeList" | "outDegree" | "inDegree" |
"degree" | "weightedOutDegree" | "weightedInDegree" | "weightedDegree" | "selfLoopWeight" |
"totalWeight" | "selfLoopArcs" | "selfLoopsPerNode" | "mate" | "degreeOrder" | "reverseDegreeOrder" |
"symmetric"`.

`DEGREE_TIER_HIGH` / `DEGREE_TIER_MID` are exported from `views.ts` but NOT from the barrel; the GPU
package reads the tiers from `segmentOffsets` and does not import them.

### 1.7 Columns, `gpuView()`, mutability (`types/columns.ts` lines 342-540, 564-660; `src/columns/column.ts`, `src/columns/table.ts`)

`AttributeTable` (table.ts; contract lines 564-660): `domain`, `rowCount`, `names()`, `has(name)`,
`get(name): Column | null`, `require(name)`, `typed(name, dtype)`, `requireTyped(name, dtype):
ColumnOf<D>`, `byRole(role): Column | null`, `value`, `isSet`, `set(name, data, decl?, opts?): Column`
(adopts a typed array BY REFERENCE after checking length; `E_COLUMN_LENGTH`; refuses SharedArrayBuffer
/ resizable buffers -- STATUS.md line 500), `remove`, `rename`, `gpuView(name): U32 | I32 | F32`
(table.ts lines 371-373 -> `gpuViewOf`, column.ts lines 1971-2009), `clone()`, iteration in
declaration order.

`gpuEligibility(dtype): GpuEligibility` (column.ts lines 98-116; exported from the barrel):

| dtype | eligibility | `gpuView` returns (column.ts 1971-2009) | WGSL read |
| --- | --- | --- | --- |
| `f32`, `i32`, `u32` | `direct` | `column.data` itself (any `components`) | `array<f32/i32/u32>`, flat interleaved; `components: 3` must NOT be `array<vec3<f32>>` (stride 16 != 12), `components: 4` may be `vec4` (design 10.2) |
| `dict` | `direct` | `column.codes` (`U32`); dictionary stays on the CPU | `array<u32>` |
| `u8` | `packed` | `column.paddedU32View()` = zero-copy `Uint32Array(ceil(byteLength / 4))` over the column's own bytes (util/typed-array.ts lines 196-206); trailing lanes of the last word are undefined | `unpack4xU8(w[i >> 2u])[i & 3u]` with a bound check `i < rows * components` |
| `bool` | `packed` | `column.data` (`U32`, `ceil(rows / 32)` words, LSB-first; trailing bits are kept CLEAR -- STATUS.md line 429) | `(w[r >> 5u] >> (r & 31u)) & 1u` |
| `f64` | `convert` | a CACHED `new Float32Array(column.data)` copy kept in a module WeakMap (column.ts lines 81, 1993-1999); the one documented non-memcpy conversion | `array<f32>` |
| `string`, `list`, `json` | `none` | throws `E_GPU_INELIGIBLE` | -- |

Every column has (`ColumnBase`, lines 342-410): `dtype`, `meta: ColumnMeta` (every field present,
`null` for none; `meta.mutable` says whether in-place writes are legal; `meta.role`; `meta.components`;
`meta.refersTo`), `length`, `validity: U32 | null` (same bitmap layout; `null` means every row set),
`nullCount`, `gpu: GpuEligibility`, `byteLength`, `paddedByteLength`, `version: number` (bumped by
`markDirty()` and `setAll()`), `paddedU32View()` (u8 / u32 / bool only), `markDirty()` (mutable only:
bumps `version`, drops the cached f32 copy and the `materializeDefault()` copy -- column.ts lines
638-648), `mutableValidity()`, `setAll()`, `slice(start, end)`, `clone()`. Typed columns add `data`
and `mutableData()` (mutable only, `E_COLUMN_IMMUTABLE` otherwise), e.g. `F32Column.data: F32`
(lines 413-421).

Declared defaults are ALREADY written into `data` for unset rows of numeric / bool / dict columns
(design 10.4; `gpu-contract.test.ts` line 594), so a kernel binds `gpuView()` without a copy.

Result attachment by reference: `snapshot.nodes.set("graphty.cc.component", labels)` adopts a
`Uint32Array` readback whatever its length (design 10.4 lines 2467-2470; `gpu-contract.test.ts` line
631). Roles the layout slice cares about (`KnownColumnRole`, columns.ts lines 152-193): `"position"`,
`"fixed"`, `"mass"`, `"size"`, `"weight"`, `"component"`, `"community"`, `"rank"`.

Cache keys the format provides for uploaded columns: the ARRAY OBJECT returned by `gpuView()` (stable
until `markDirty()` for f64 -- a new copy is made -- and until `dropCaches()`), plus `column.version`
to detect in-place writes to a mutable column whose array object did not change (design 14.5 lines
4230-4233: "dropped on `markDirty()` via `column.version`").

### 1.8 NodeIdMap boundary helpers (`types/snapshot.ts` lines 58-153)

The GPU package returns index-aligned arrays and never keys anything by id (design 10.7 lines
2532-2536). The helpers a CALLER uses at the boundary, verified on the contract: `ids.kind`,
`ids.size`, `ids.offset`, `idOf(i)`, `indexOf(id)` (`INVALID_INDEX` on a miss), `has`,
`requireIndex` (`E_UNKNOWN_NODE`), `indicesOf(ids, onMissing?)`, `toMap<T>(values: ArrayLike<T>):
Map<NodeId, T>`, `toStringMap`, `toRecord`, `entries<T>(values)`, `stringIndex()`. The one place the
GPU package itself touches ids is an options field like `source: NodeId | number` for BFS / SSSP;
resolve through `snapshot.ids.requireIndex(id)` only when the value is not already an index (the
CPU package `indexed.*` convention will be index-only; keep the GPU entry points index-only too and
let graphty-element resolve ids).

### 1.9 Masks (`src/util/mask.ts` lines 27-78; barrel line 28)

`makeMask(length, fill?): U32` (ceil(length / 32) words), `maskTest(mask, i)`, `maskSet(mask, i,
value)`, `maskCount(mask, length)`, `maskToIndices(mask, length): U32`. The layout `setFixed(mask:
NodeMask)` of design 14.3 uses this exact layout, which is also the bit layout of a `bool` column with
role `"fixed"` (upload `snapshot.nodes.gpuView(name)` of that column directly). `checkMaskLength` is
NOT exported; the GPU package validates `mask.length >= Math.ceil(n / 32)` itself.

### 1.10 Per-arc / per-edge folding and index-space helpers

`foldArcs<T extends NumericVector>(snapshot, perArc: T, reducer: "first" | "sum" | "max" | "min",
out?: T): T` (views.ts lines 852-909): `E_COLUMN_LENGTH` unless `perArc.length === arcCount`;
returns `perArc` ITSELF when `flags.arcToEdgeIsIdentity` and no `out`; otherwise
`result[e] = perArc[edgeToArc[e]]` then the reducer over the other arcs. `"first"` when both arcs of
an undirected edge hold the same value (edge betweenness); `"sum"` / `"max"` / `"min"` only for
per-direction contributions (design 10.7). `expandEdges<T>(snapshot, perEdge: T, out?: T): T` (lines
922-942) is the inverse (`result[a] = perEdge[arcToEdge[a]]`), same identity short-cut. Both are CPU
loops; the GPU package uses them on readbacks, and may implement the same gathers on the device with
`arcToEdge` / `edgeToArc` bindings when the result stays resident.

`remapArray(data, remap, newLength, fill, components = 1)` (columns/remap.ts lines 82-127: OLD -> NEW,
`INVALID_INDEX` drops), `gatherArray(data, indexMap, components = 1)` (line 130: NEW -> OLD,
`out[i] = data[indexMap[i]]`), `scatterArray(out, values, indexMap, components = 1)` (line 167:
`out[indexMap[i]] = values[i]`, length-checked), `gatherColumn`, `remapColumn`, `withComponents(data:
F32, from, to, fill): F32` (line 554; returns `data` itself when `from === to`). These are the tools
for running a kernel on a derived graph (`inducedSubgraph`, `toUndirected`, `simplified`,
`contract`) and writing results back through `DerivedGraph.nodeOrigin` / `edgeOrigin` /
`nodeRemap` / `edgeRemap` (types/snapshot.ts lines 332-362).

`renumberPartition(labels: U32, out?: U32): { labels: U32; count: number }` (derived.ts lines
1155-1190): dense 0..k-1 relabelling in first-seen order; `out` may be `labels` itself. This is the
CPU dense-relabel of component ids that design 10.8 leaves to the GPU package: reuse it after a CC
readback instead of writing a second one.

### 1.11 Derived graphs the GPU package may be handed (`types/snapshot.ts` lines 611-700)

`toUndirected(options?)`, `transpose()`, `simplified(options?)`, `withoutSelfLoops()`,
`filterEdges(keep: EdgeMask)`, `inducedSubgraph(selection: U32 | { mask: NodeMask })`,
`contract(partition: U32, options?)`, `relabel(perm: U32)`, `withColumns(nodes?, edges?)`. Each
returns `DerivedGraph { snapshot, nodeOrigin, edgeOrigin, nodeRemap, edgeRemap, blockSizes, report }`
with a NEW `serial` except `withColumns()` (same serial, shared core) and the identity cases
(`transpose()` / `toUndirected()` of an undirected snapshot return `{ snapshot: this, null maps }`).
The layout slice receives `dm.undirected(s)` from graphty-element (design 14.3 lines 3990-3996) and
never converts itself.

### 1.12 Factories the tests use (`src/populate/*.ts`, `src/builder/graph-builder.ts`)

- `fromEdgeArrays(input: EdgeArraysInput, options: BuilderOptionsPatch & FreezeOptions = {}):
  GraphSnapshot` (from-edge-arrays.ts lines 216-257). `EdgeArraysInput` (types/snapshot.ts lines
  704-727): `directed`, `nodeCount?` (required unless `ids`), `ids?`, `src: U32`, `dst: U32`,
  `weights?: F32 | F64`, `nodeColumns?`, `edgeColumns?`, `meta?`. Options include `duplicateEdges`,
  `selfLoops`, `weighted`, `label`, `prepare`, `arena` (default true), `checksum`. ~22 ms directed /
  ~48 ms undirected for 100k / 1M (STATUS.md line 437). This is THE test-graph constructor.
- `fromCsr(input: CsrInput, options?: FromCsrOptions)` (from-csr.ts lines 675+): adopts arrays by
  reference (`copy: false` default), `validate` default `"full"`, `sortRows` default true; an
  undirected input must supply `arcToEdge`. Useful to build a snapshot around a pre-computed GPU
  result (e.g. on-device COO -> CSR) -- note `arena === null` unless the five arrays share one
  aligned buffer in hot-to-cold order.
- `GraphBuilder` (`new GraphBuilder({ directed })`, `addAnonymousNodes(n)`, `addEdges(src, dst,
  weights?)`, `declareNodeColumn({ name, dtype, components?, nullable?, mutable? })`,
  `setNodeValue(name, row, value)`, `freeze(options?)`, `freezeWithReport`) as used by
  `gpu-upload.test.ts` lines 481-499 to make u8 / bool / f64 / stride-3 columns.
- `fromRecords`, `fromWire`, `fromBytes`, `fromByteChunks` for fixture loading.

---

## 2. Where the implementation differs from, or refines, design section 10

None of these contradicts section 10; each is a detail the GPU package must know.

1. `reverse().fwdArc` on an UNDIRECTED snapshot is not "the same array object as forward": it is an
   identity permutation allocated on first read (`views.ts` lines 116-119). Design 10.1 line 2338
   lists `fwdArc` among the arrays that alias when undirected. Rule: treat `fwdArc` exactly like
   `arcToEdge` -- do not read it when `!snapshot.directed`; a pull kernel on an undirected snapshot
   uses `USE_PERM = false`.
2. `coo().arcToEdge` and `edgeList().arc` are getters that materialise the identity permutation on
   the source snapshot (`views.ts` header lines 14-17). Design 10.1 only warns about
   `snapshot.arcToEdge`. Rule: read `flags.arcToEdgeIsIdentity` BEFORE touching any of the three.
3. `dropCaches()` also resets materialised identity permutations and the f64 `gpuView` copies
   (`graph-snapshot.ts` lines 797-813; STATUS.md line 332-333). Design 7.2 / 10.4 mention only views
   and gpuView copies. Rule: the upload cache keyed on array objects must tolerate a key that is no
   longer reachable from the snapshot (it simply becomes garbage in the WeakMap; the GPU buffer it
   references is NOT freed until `release(snapshot)` -- so the residency record must ALSO be
   reachable from the snapshot (a `WeakMap<GraphSnapshot, Residency>`) so `release` can enumerate
   every buffer, including ones whose CPU key was dropped.
4. `arena.byteLength` excludes trailing padding (STATUS.md lines 421-423): it equals the end of the
   last non-empty segment, so `new Uint8Array(arena.buffer, arena.byteOffset, arena.byteLength)`
   may have a length that is not a multiple of 4 only if the last segment's length is not (never:
   every segment is 4-byte elements). `hotByteLength` likewise. Both are safe `writeBuffer` sizes
   (WebGPU requires the size to be a multiple of 4; every core array is 4-byte elements, I10).
5. `fromCsr` gives `arena === null` for separately allocated arrays and `transpose()` gives
   `arena === null` (CONFORMANCE.md lines 269, 288, 412). Rule: the per-array upload path is not an
   exception path; it is the normal path for every adopted or transposed snapshot and must be as
   well tested as the arena path.
6. `AttributeTable.set()` does not range-check `refersTo` values (STATUS.md lines 443-449). The GPU
   package must not trust a `u32` column with `refersTo` as an in-range index without its own bound
   check when it dereferences it in a kernel.
7. The bool bitmap keeps bits `>= N` clear (STATUS.md line 429) -- stronger than design 10.2's
   "undefined trailing lanes" for u8. Kernels still bound-check because `paddedU32View()` of a u8
   SLICE at an unaligned start shares its last word with neighbouring bytes (`gpu-contract.test.ts`
   line 602).
8. `ReverseView.arcToEdge` on a directed identity snapshot ALIASES `fwdArc` (`views.ts` lines
   61-65): the GPU package can bind `fwdArc` and skip a second upload.
9. `equalsTopology` compares ids too (graph-snapshot.ts lines 1411-1420): it is a test-oracle
   helper, not a cheap cache key. Use `serial` (core identity) plus array-object identity for caching.
10. `MAX_COUNT` means arc indices up to `0xFFFFFFFE` while WebGPU `array<u32>` indices are u32: fine.
    The 1D dispatch ceiling 65,535 x 256 = 16,776,960 invocations (design 10.6 lines 2490-2496) is
    exercised as an assertion in `gpu-upload.test.ts` line 189 and must become a planner rule.

---

## 3. Consumption rules the GPU package must follow

Derived from sections 1-2 and design 10.1-10.8 / 14.5 (lines 4212-4244):

- Entry points take `GraphSnapshot` (not `AdjacencyView`) so they can read `flags` and `arena`
  (design 10.1 line 2350). Row-walking helpers inside the package may take `AdjacencyView` so one
  implementation serves `snapshot` and `snapshot.reverse()`.
- Upload plan (design 10.3 lines 2416-2440; 14.5 lines 4224-4229): (1) whole-arena when
  `arena !== null && arena.byteLength <= device.limits.maxBufferSize` AND every non-null
  `segment.byteLength <= device.limits.maxStorageBufferBindingSize`: one `createBuffer` + one
  `writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength))` (or
  `arena.byteLength` when a cold segment is needed), bind `{ buffer: gbuf, offset: seg.byteOffset -
  arena.byteOffset, size: seg.byteLength }`; (2) else per array (`writeBuffer(bufX, 0,
  snapshot.colIdx)`), also the path for `arena === null`; (3) else windows on ARC ranges at 64-arc
  boundaries (`start = rowPtr[v0] - (rowPtr[v0] % 64)`, `%` not `& ~63`).
- Request raised limits from the adapter first (`maxBufferSize`, `maxStorageBufferBindingSize`,
  `maxComputeWorkgroupsPerDimension`, `maxStorageBuffersPerShaderStage`) and plan against
  `device.limits`, never against the spec defaults.
- Never bind a zero-length array: `colIdx`, `weights`, `arcToEdge` have length 0 when
  `arcCount === 0`; `edgeToArc` when `edgeCount === 0`; a nullable column of an empty table has an
  empty validity array (design 10.5 lines 2477-2484). Dispatch nothing for empty ranges.
- Identity permutations: `override USE_PERM: bool` in WGSL, `select(a, arcToEdge[a], USE_PERM)`;
  in the identity case fill the `arcToEdge` slot with `colIdx` and the `edgeToArc` slot with
  `rowPtr` (already uploaded, never read) so one bind-group layout serves both (design 10.1 lines
  2334-2336). Applies equally to `reverse().fwdArc` when `!directed` (section 2 item 1).
- Weighted normalisers are computed on the device by a segmented reduce over `rowPtr` / `weights`,
  guarding division by a zero sum (design 10.1 line 2344, 10.5 lines 2484-2487).
- Upload cache: `WeakMap` keyed on the typed-array OBJECT (`rowPtr` for a core, the view array for a
  view, the `gpuView()` array for a column) plus `column.version`, and a per-snapshot residency
  record so `release(snapshot)` frees every `GPUBuffer` (design 14.5 lines 4229-4234). GPU memory
  is never freed by garbage collection; graphty-element calls `release(previous)` from
  `snapshot-replaced`.
- Readback: copy out of `getMappedRange()` BEFORE `unmap()` (the mapped range is detached at
  unmap; `gpu-upload.test.ts` lines 122-131 show the idiom `staging.getMappedRange().slice(0)`),
  into a fresh `Uint32Array<ArrayBuffer>` / `Float32Array<ArrayBuffer>` or a caller-supplied
  destination (`dest?` parameter on every algorithm, design 10.7 lines 2528-2531).
- Results are index-aligned typed arrays of length n (per node), `arcCount` (per arc, fold with
  `foldArcs`) or `edgeCount` (per edge, from the `edgeToArc` writeback path). `parents` is
  `Uint32Array` with `INVALID_INDEX`, never `Int32Array` with -1 (design 10.1 lines 2352-2354).
- The package throws when no device is available and never falls back (project rule
  `/home/apowers/Projects/webgpu-graph-algorithms/CLAUDE.md`; design 10.8 lines 2540-2546).
- Never write into a view array or a core array (I17); scratch is `.slice()` or a fresh array.
- Never apply `|`, `&`, `>>>` to arc indices or byte offsets that can exceed 2^31 (I3); use `%`,
  `Math.floor`, `Math.trunc`.

---

## 4. Package skeleton for `packages/webgpu-graph-algorithms/`

Mirror `packages/graph-format/` and `packages/graph-io/` file for file so the W1 move is a plain
`mv` and the `packages/README.md` checklist (lines 44-198) applies verbatim.

### 4.1 Directory tree

```
packages/webgpu-graph-algorithms/
+-- package.json                  # @graphty/webgpu-graph-algorithms (section 4.2)
+-- project.json                  # Nx project "webgpu-graph-algorithms" (section 4.3)
+-- webgpu-graph-algorithms.ts    # one line: export * from "./src/index.js"; (the algorithms.ts / graph-format.ts pattern, README.md lines 168-176)
+-- tsconfig.json                 # lint/typecheck, noEmit, noUncheckedIndexedAccess OFF (design 14.5)
+-- tsconfig.build.json           # emit src/ + root entry -> dist/src/ + dist/webgpu-graph-algorithms.*
+-- tsconfig.strict-consumer.json # test/types/*.test-d.ts against dist/*.d.ts with the strict flags ON
+-- vitest.config.ts              # projects: "node" (Dawn) and "browser" (Playwright Chromium); thresholds 80/80/75/80
+-- scripts/entries.js            # { "webgpu-graph-algorithms": "src/index.ts", node: "src/node/index.ts" } (graph-io pattern)
+-- scripts/build-bundle.js       # multi-entry vite lib build, externals = dependencies + peerDependencies + "webgpu"
+-- scripts/bundle-types.js       # dist/<entry>.d.ts one-line re-exports of dist/src/**
+-- README.md  CLAUDE.md  LICENSE
+-- src/
|   +-- index.ts                  # the ONLY public barrel (browser-safe: no node: imports, no "webgpu" import)
|   +-- node/index.ts             # the ./node subpath: createNodeGpu() over the "webgpu" (Dawn) package
|   +-- errors.ts                 # WebGpuGraphError { code, details } (its own class; never GraphFormatError)
|   +-- constants.ts              # WORKGROUP_SIZE = 256, MAX_1D_INVOCATIONS = 65535 * 256, ARC_WINDOW_ALIGN = 64
|   +-- types/                    # public option / result / context types (index.ts barrel of types only)
|   +-- device/                   # GpuContext: adapter + device acquisition, limit requests, feature probes
|   +-- memory/                   # residency.ts (upload cache + release), buffer-pool.ts, readback.ts, upload-plan.ts
|   +-- kernel/                   # pipeline-cache.ts, dispatch.ts (1D/2D planner), uniforms.ts, wgsl.ts (prelude + composition)
|   +-- wgsl/                     # *.wgsl.ts: WGSL source as exported template-string constants (section 4.6)
|   +-- primitives/               # scan.ts reduce.ts segmented-reduce.ts compact.ts radix-sort.ts histogram.ts frontier.ts advance.ts spmv.ts coo-to-csr.ts
|   +-- algorithms/               # one file per algorithm, grouped by primitive family
|   +-- layouts/                  # LayoutSimulation implementations (force-directed first)
+-- test/
|   +-- setup/gpu.ts              # acquire(): Dawn in Node, navigator.gpu in the browser; requireGpu(t) skip-with-reason
|   +-- helpers/                  # graphs.ts (seeded generators), oracle.ts (CPU reference implementations), device.ts
|   +-- <module>/*.test.ts        # node project: unit tests per src module (run on Dawn)
|   +-- browser/*.test.ts         # browser project: light smoke suite on Playwright Chromium (real GPU locally)
|   +-- types/*.test-d.ts         # strict-consumer type tests
|   +-- index.test.ts             # barrel exports exactly the documented value surface
|   +-- build-output.test.ts      # package.json shape, exports map incl. ./node, dist checks
+-- benchmarks/                   # run.ts harness.ts datasets.ts <group>.bench.ts results/
```

### 4.2 package.json

Fields copied from `packages/graph-io/package.json` (lines 1-131) with these values:

```jsonc
{
    "name": "@graphty/webgpu-graph-algorithms",
    "version": "0.1.0",
    "description": "WebGPU-accelerated graph algorithms and layouts over the @graphty/graph-format snapshot",
    "author": "Adam Powers <apowers@ato.ms>",
    "type": "module",
    "main": "dist/webgpu-graph-algorithms.js",
    "types": "dist/webgpu-graph-algorithms.d.ts",
    "exports": {
        ".": { "types": "./dist/webgpu-graph-algorithms.d.ts", "import": "./dist/webgpu-graph-algorithms.js", "default": "./dist/webgpu-graph-algorithms.js" },
        "./node": { "types": "./dist/node.d.ts", "import": "./dist/node.js", "default": "./dist/node.js" }
    },
    "sideEffects": false,
    "files": ["dist/", "src/", "README.md", "LICENSE"],
    "publishConfig": { "access": "public", "provenance": true },
    "engines": { "node": ">=18.19.0" },
    "scripts": {
        "build": "tsc -p tsconfig.build.json",
        "build:bundle": "node scripts/build-bundle.js",
        "build:all": "npm run build && npm run build:bundle",
        "lint": "eslint && tsc --noEmit -p tsconfig.json",
        "lint:fix": "eslint --fix",
        "typecheck": "tsc --noEmit -p tsconfig.json",
        "typecheck:strict-consumer": "tsc -p tsconfig.strict-consumer.json",
        "test": "vitest",
        "test:ui": "vitest --ui",
        "test:run": "vitest run",
        "test:node": "vitest run --project=node",
        "test:browser": "vitest run --project=browser",
        "coverage": "vitest run --coverage",
        "coverage:preview": "npx serve coverage -p 9058",
        "benchmark": "tsx benchmarks/run.ts",
        "ready:commit": "npm run build:all && npm run lint && npm run typecheck:strict-consumer && npm run test:run"
    },
    "repository": { "type": "git", "url": "git+https://github.com/graphty-org/graphty-monorepo.git", "directory": "webgpu-graph-algorithms" },
    "keywords": ["graph", "webgpu", "wgsl", "gpu", "graph-algorithms", "force-directed", "layout", "graph-format"],
    "license": "MIT",
    "bugs": { "url": "https://github.com/graphty-org/graphty-monorepo/issues" },
    "homepage": "https://github.com/graphty-org/graphty-monorepo/tree/master/webgpu-graph-algorithms#readme",
    "dependencies": {
        "@graphty/graph-format": "workspace:*",
        "@webgpu/types": "^0.1.72"
    },
    "peerDependencies": {
        "@graphty/graph-format": "^0.1.0",
        "webgpu": "^0.4.0"
    },
    "peerDependenciesMeta": {
        "webgpu": { "optional": true }
    },
    "devDependencies": {
        "@vitest/browser": "^3.2.4",
        "@vitest/coverage-v8": "^3.2.4",
        "@vitest/ui": "^3.2.4",
        "fast-check": "^4.2.0",
        "playwright": "^1.54.1",
        "tsx": "^4.20.3",
        "typescript": "^5.9.3",
        "vite": "^7.0.5",
        "vitest": "^3.2.4",
        "webgpu": "^0.4.0"
    }
}
```

Notes and rationale:

- `@graphty/graph-format` in BOTH `dependencies` (`workspace:*`) and `peerDependencies` (`^0.1.0`),
  exactly as `packages/graph-io/package.json` lines 106-111 and `graph-io/CLAUDE.md` lines 122-123.
  The peer range bumps to `^1.0.0` when F2 cuts 1.0 (design 14.6); no consumer PR merges to master
  before that.
- `webgpu` (Dawn for Node, `dawn-gpu/node-webgpu`) is an OPTIONAL peer (browser consumers never
  install it) AND a devDependency (the node test project needs it). Pin `^0.4.0`: graph-format pins
  the same range (`packages/graph-format/package.json` line 73) and the lockfile rehearsal resolved
  `webgpu@0.4.0` (`packages/README.md` line 145); the plan brief records that 0.5+ / 0.6+ need glibc
  2.38 while the dev container is Ubuntu 22.04 / glibc 2.35 (`HEADLESS_GPU_REPORT.md` line 31 gives
  the container's glibc; the 0.5+ requirement itself was not re-verified here). Its API
  (`packages/graph-format/node_modules/webgpu/types.d.ts`): `create(options: string[]): GPU` and
  `globals: Object`; usage per its README lines 12-24: `Object.assign(globalThis, globals); const gpu
  = create([])`. Dawn toggles go in the string list (`enable-dawn-features=...`, `backend=vulkan`,
  `adapter=<name>`; README lines 30-70). Lifetime note (README lines 84-98): the process does not
  exit while a reference to the object returned by `create()` is alive -- `test/setup/gpu.ts` must
  drop it in `afterAll` (the graph-format test destroys the device; also null the `gpu` reference).
  `pnpm install` prints "dependencies have build scripts that were ignored: webgpu" -- expected, the
  postinstall only strips a macOS quarantine attribute (README.md lines 151-156).
- `@webgpu/types` is a runtime-free package whose ambient declarations the PUBLIC d.ts of this
  package needs (`GPUDevice`, `GPUBuffer`, `GPUAdapter` appear in exported signatures), so it goes in
  `dependencies`, and `src/index.ts` starts with `/// <reference types="@webgpu/types" />` so tsc
  carries the reference into `dist/src/index.d.ts` for consumers (graph-format only needed it in a
  test-d file: `test/types/typed-arrays.test-d.ts` line 1). Version `^0.1.72` matches graph-format
  (`packages/graph-format/package.json` line 66) and what the monorepo lock re-resolved to
  (README.md lines 147-150).
- Playwright: the monorepo root pins `playwright ^1.54.1` (`/home/apowers/Projects/graphty-monorepo/package.json`
  line 148) and `@vitest/browser ^3.2.4` (line 130) with a `pnpm.overrides` entry
  `"@vitest/browser@<3.2.7": "^3.2.7"` (line 60); declare the same carets here. Playwright 1.54.1 wants
  Chromium build 1181 (Chromium 139), the build `HEADLESS_GPU_REPORT.md` line 36 verified on the RTX
  4070 SUPER.
- No `browserslist` (design 14.5 line 4216: "no `browserslist`").
- `private` is NOT set (the scaffold's `"private": true`, `package.json` line 6, goes away): the
  package publishes with provenance like its siblings.
- `coverage:preview` port 9058 (graph-format 9056, graph-io 9057; `packages/README.md` line 41).

### 4.3 project.json

Copy `packages/graph-io/project.json` (lines 1-58) with `name` / `sourceRoot` / every `cwd` set to
`webgpu-graph-algorithms`. Same seven targets (`build` with `"dependsOn": ["^build"]` and `outputs:
["{projectRoot}/dist"]`, `test`, `test:ui`, `coverage`, `lint`, `typecheck`, `benchmark`). Add two
targets the CI shard split needs:

```jsonc
"test:node":    { "executor": "nx:run-commands", "options": { "command": "npm run test:node",    "cwd": "webgpu-graph-algorithms" } },
"test:browser": { "executor": "nx:run-commands", "options": { "command": "npm run test:browser", "cwd": "webgpu-graph-algorithms" } }
```

### 4.4 tsconfig trio

`tsconfig.json` (from `packages/graph-io/tsconfig.json` lines 1-18):

```jsonc
{
    "extends": "../tsconfig.base.json",
    "compilerOptions": {
        "composite": true,
        "noEmit": true,
        "noImplicitOverride": true,
        "noUncheckedIndexedAccess": false,      // design 14.5 line 4218: OFF so row loops and the result-writing loop are shared with the other consumers
        "exactOptionalPropertyTypes": false,
        "lib": ["ES2020", "DOM", "DOM.Iterable"],   // DOM for navigator.gpu in the browser entry; the node entry never uses it
        "types": ["node", "vitest/globals", "@webgpu/types"],
        "paths": {
            "@graphty/webgpu-graph-algorithms": ["./src/index.ts"],
            "@graphty/webgpu-graph-algorithms/node": ["./src/node/index.ts"],
            "@graphty/graph-format": ["../graph-format/src/index.ts"]
        }
    },
    "include": ["webgpu-graph-algorithms.ts", "src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "../graph-format/src/**/*.ts"],
    "exclude": ["node_modules", "dist", "coverage", "tmp"]
}
```

`tsconfig.base.json` (`packages/tsconfig.base.json`) already gives `target ES2020`, `module ES2020`,
`moduleResolution bundler`, `strict`, `isolatedModules`, `noUnusedLocals/Parameters`,
`noImplicitReturns`, `noFallthroughCasesInSwitch`, `declaration`, `sourceMap`. Do NOT re-add the
scaffold's `noPropertyAccessFromIndexSignature`, `allowImportingTsExtensions`,
`allowArbitraryExtensions`, `jsx`, `moduleDetection`, `importsNotUsedAsValues` (scaffold
`tsconfig.json` lines 2-40; several are incompatible with a `tsc -p tsconfig.build.json` emit).

`tsconfig.build.json` (from `packages/graph-io/tsconfig.build.json`): `noEmit: false`, `rootDir: "."`,
`outDir: "./dist"`, `declaration`, `declarationMap`, `stripInternal: true` (graph-format sets it,
`packages/graph-format/tsconfig.build.json` line 9; it strips `@internal` members from the d.ts),
`types: ["node", "@webgpu/types"]`, `paths: {}` (so `@graphty/graph-format` resolves through the
workspace symlink to the BUILT format package -- graph-format must be built first, graph-io/CLAUDE.md
lines 85-88), `include: ["webgpu-graph-algorithms.ts", "src/**/*.ts"]`, exclude `test`, `benchmarks`.

`tsconfig.strict-consumer.json` (from `packages/graph-io/tsconfig.strict-consumer.json`):
`noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`, `composite: false`, `paths`
mapping the package name and `./node` to `./dist/*.d.ts` and `@graphty/graph-format` to
`../graph-format/dist/graph-format.d.ts`, `include: ["test/types/**/*.test-d.ts"]`. Purpose: the
public d.ts must compile for a consumer running the strict flags (design 16.6; graphty-element
compiles with them).

### 4.5 vitest.config.ts

Two projects in ONE config (the `algorithms/vitest.config.ts` shape in the monorepo, lines 1-60,
with coverage thresholds skipped when `--project=` is passed, lines 75-88), environment `node` (NOT
`happy-dom`: the format package uses `environment: "node"`, `pool: "forks"`,
`packages/graph-format/vitest.config.ts` lines 4-8):

```ts
import { defineConfig } from "vitest/config";

const single = !process.argv.some((a) => a.startsWith("--project"));
export default defineConfig({
    test: {
        projects: [
            {
                test: {
                    name: "node",
                    globals: true,
                    environment: "node",
                    pool: "forks",
                    testTimeout: 30000,
                    hookTimeout: 60000,
                    include: ["test/**/*.test.ts"],
                    exclude: ["test/browser/**"],
                    setupFiles: ["test/setup/gpu.ts"],
                },
            },
            {
                test: {
                    name: "browser",
                    include: ["test/browser/**/*.test.ts"],
                    testTimeout: 60000,
                    hookTimeout: 60000,
                    browser: {
                        enabled: true,
                        headless: true,
                        provider: "playwright",
                        instances: [
                            {
                                browser: "chromium",
                                launch: {
                                    args: [
                                        "--enable-unsafe-webgpu",
                                        "--enable-features=Vulkan",
                                        "--use-angle=vulkan",
                                        "--disable-vulkan-surface",
                                    ],
                                },
                            },
                        ],
                    },
                },
            },
        ],
        coverage: {
            all: true,
            provider: "v8",
            reporter: ["text", "json-summary", "json", "lcov", "html"],
            reportsDirectory: process.env.COVERAGE_DIR ?? "coverage",
            include: ["src/**/*.ts"],
            exclude: ["**/*.d.ts", "**/*.test.ts", "**/index.ts", "src/wgsl/**"],
            thresholds: single ? { lines: 80, functions: 80, branches: 75, statements: 80 } : undefined,
        },
        reporters: ["verbose"],
    },
});
```

The four Chromium flags are the exact set `HEADLESS_GPU_REPORT.md` lines 19-22 and 165-175 proved
(all four, no more; `--use-gl=swiftshader` / `--use-vulkan=swiftshader` from the scaffold force
software rendering, report line 226-227). Where the per-instance `launch` option lives in Vitest 3's
`browser.instances[]` must be confirmed against the installed version at implementation time (the
report's recipe on lines 204-224 targets Vitest 2.1's `providerOptions.launch`); record the verified
spelling in the package CLAUDE.md. `LD_LIBRARY_PATH` for `libEGL.so.1` (report appendix D, lines
389-397: `tmp/egl/root/usr/lib/x86_64-linux-gnu`) is an ENVIRONMENT concern, set by the developer's
shell or a `tools/` script, never by the config (report recommendation 1, lines 200-203).

Coverage: `src/wgsl/**` is excluded because template-string modules have no branches worth
measuring; everything else in `src/` counts. The browser project is a SMOKE suite (owner: "light
browser testing to prove that they will work"); it does not carry coverage.

### 4.6 Build scripts and WGSL handling

Copy `packages/graph-io/scripts/{entries.js,build-bundle.js,bundle-types.js}` with
`ENTRIES = { "webgpu-graph-algorithms": "src/index.ts", node: "src/node/index.ts" }`.
`build-bundle.js`'s `externalDependencies()` (lines 33-47) already externalises every
`dependencies` + `peerDependencies` name including subpaths, so `@graphty/graph-format` and `webgpu`
stay external and the browser bundle never references Dawn. `node/index.ts` must import `webgpu`
DYNAMICALLY (`await import("webgpu")`, as `gpu-upload.test.ts` lines 40-45 do) so that a static
analysis of the root entry finds no Node-only module, and so the `./node` bundle can report a clear
error when Dawn is absent.

WGSL: keep shader source as TypeScript template-string modules under `src/wgsl/*.wgsl.ts`
(`export const bfsAdvance = /* wgsl */ \`...\`;`), NOT as `.wgsl` files with `?raw` imports. Reasons
(all verified against the toolchain in `packages/`): the pre-push hook builds with plain
`pnpm -r run build` = `tsc -p tsconfig.build.json` (README.md lines 168-176), and tsc neither copies
`.wgsl` files into `dist/src/` nor understands `?raw`, so a tsc-only dist would be broken; knip's
`project` globs are `src/**/*.ts` (`packages/knip.config.ts` lines 31-32) and would report `.wgsl`
files as unused; the shared eslint config lints `.ts` only. Composition (prelude + operator
snippets) is string concatenation in `src/kernel/wgsl.ts`; the WGSL prelude carries
`const INVALID_INDEX: u32 = 0xFFFFFFFFu;` and the `override USE_PERM: bool;` declaration. Editor
highlighting comes from the `/* wgsl */` tag. If `.wgsl` files are ever wanted, a
`scripts/gen-wgsl.js` that emits `src/wgsl/*.wgsl.ts` with the auto-generated header (owner rule:
"THIS FILE IS AUTO GENERATED: DO NOT EDIT THIS FILE. INSTEAD EDIT src/wgsl/<name>.wgsl") is the
compatible route; not needed for the first slices.

### 4.7 knip, commitlint and the root touch points

`packages/knip.config.ts` gets a third workspace in the monorepo's shape (lines 30-40):

```ts
"webgpu-graph-algorithms": {
    entry: ["src/index.ts", "src/node/index.ts", "test/**/*.test.ts", "test/types/**/*.test-d.ts", "scripts/**/*.{ts,js}"],
    project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"],
},
```

(`src/node/index.ts` is a second entry because the root barrel does not re-export it, unlike
graph-io's format barrels -- README.md line 65-66.) `packages/pnpm-workspace.yaml` gains
`- "webgpu-graph-algorithms"`.

At W1 the `packages/README.md` checklist (lines 44-198) repeats with these additions to
`move/root-touch-points.diff`: `pnpm-workspace.yaml` (after `graph-io`, before `algorithms`),
`commitlint.config.js` `scope-enum` + `"webgpu-graph-algorithms"`
(`/home/apowers/Projects/graphty-monorepo/commitlint.config.js` lines 4-22; note the existing
`"gpu-3d-force-layout"` scope on line 15 is a stale name of an older experiment and can stay),
`knip.config.ts` (the entry above plus the monorepo's `ignore: ["dist/**", "coverage/**",
"node_modules/**"]`), root `package.json` `coverage:preview:webgpu-graph-algorithms` on port 9058,
root `README.md` package block, `tools/prepush.sh` (ONLY the node project: `(cd
webgpu-graph-algorithms && npm run test:node)` -- the browser project is not a fast test),
`tools/merge-coverage.sh` `PACKAGES`, `.github/workflows/ci.yml` build upload/download steps and
TWO shards `webgpu-graph-algorithms-node` and `webgpu-graph-algorithms-browser` (the
`algorithms-default` / `algorithms-browser` split, ci.yml lines 262-271), the `Build ... (PR)`
step, `release.yml` download step, and root `CLAUDE.md` rows via the script. The GPU shards are
where the "one runner for GPU and a default runner for other tests" decision lands (a separate
plan note covers the runner); the package-level split into `node` / `browser` projects is what makes
that routing a one-line `runs-on` change per shard.

### 4.8 Device acquisition in Node (test setup and the `./node` entry)

`packages/graph-format/test/audit/gpu-upload.test.ts` lines 26-102 is the working reference:
`import("webgpu")`, `Object.assign(globalThis, dawn.globals)` (installs `GPUBufferUsage`,
`GPUMapMode`, `GPUShaderStage` and friends), `dawn.create([])`, `requestAdapter()`, `adapter.info`
(vendor / architecture / device / description), `requestDevice()`, `device.destroy()` in `afterAll`,
and a `requireGpu(t)` that calls `t.skip(reason)` with an `E_NO_ADAPTER` string -- "a wrong result is
never a skip" (line 16-17). The GPU package's `test/setup/gpu.ts` generalises it: acquire once per
worker, print `adapter.info` and the four limits (lines 82-88), export `requireGpu`, and set
`REQUIRE_GPU=1` semantics (fail instead of skip) for the local box and the GPU runner so a
SwiftShader / llvmpipe adapter cannot silently pass (report recommendation 3, lines 228-233; the
Dawn fallback to Mesa llvmpipe is noted at gpu-upload.test.ts lines 13-15). The `./node` entry's
`createNodeGpu(options?: { dawnFlags?: string[] })` is the same sequence minus vitest, returning
`{ gpu, dispose() }`, and it throws (never falls back) when the module or an adapter is missing.

---

## 5. Scaffold triage: delete / keep / move

The repository root `/home/apowers/Projects/webgpu-graph-algorithms/` currently holds a July 2025
npm scaffold (git status: every file untracked). Nothing in it is imported by anything. Decision per
file (the new package lives under `packages/webgpu-graph-algorithms/`; the root becomes a thin
staging wrapper like it already is for graph-format / graph-io):

| Path | Action | Reason |
| --- | --- | --- |
| `src/types/index.ts` | DELETE | `CSRGraph { numVertices, numEdges, rowPtr, colIdx, edgeWeights? }`, `EdgeListGraph`, `GraphAlgorithm<TInput,TOutput>` with `initialize/execute/dispose`, `ShortestPathResult.parents: Int32Array`: every one is superseded by `GraphSnapshot`, the plain-async-function convention and `Uint32Array` + `INVALID_INDEX` (design 14.5 lines 4219-4223 say to delete `CSRGraph` / `EdgeListGraph`; 10.1 line 2352 renames the fields). `GPUConfig` / `PageRankResult` / `ConnectedComponentsResult` are redefined in the new `src/types/`. |
| `src/index.ts` | DELETE (rewrite) | exports `version` and `export * from "./types"`; the new barrel uses explicit named exports (graph-format rule, `src/index.ts` lines 4-6). |
| `src/{algorithms,core,formats,utils}/` | DELETE | empty directories; the new layout is `device/ memory/ kernel/ wgsl/ primitives/ algorithms/ layouts/`. "formats" is graph-format's job. |
| `test/setup/browser.ts`, `test/setup/webgpu-global.ts` | DELETE (replace) | `requestAdapterInfo()` is removed from the API (report line 232 recommends `adapter.info`); the console output uses a non-ASCII check mark (`webgpu-global.ts` lines 58, 61: forbidden by the ASCII rule); the throw-on-missing-WebGPU intent is kept in the new `test/setup/gpu.ts`. |
| `test/setup/node.ts` | DELETE | a `console.log` placeholder. |
| `test/helpers/webgpu.ts`, `test/helpers/test-utils.ts` | DELETE (fold in) | `createBuffer` via `mappedAtCreation` and `readBuffer` returning `Float32Array` are less general than the graph-format `upload` / `readback` helpers (section 6); `expectFloatArraysEqual` is replaced by a tolerance matcher in `test/helpers/oracle.ts`. `withTestDevice` (fresh device per test) is a useful pattern to keep in `test/helpers/device.ts`. |
| `test/browser/webgpu-check.test.ts` | MOVE + harden | becomes the first browser smoke test; add `expect(adapter.info.vendor)` / `isFallbackAdapter === false` under `REQUIRE_GPU=1` (report recommendation 3). |
| `test/browser/webgpu.test.ts` | DELETE | a doubling-shader demo that the walking-skeleton test supersedes. |
| `test/unit/index.test.ts` | DELETE | asserts `version === "0.1.0"`; the new `test/index.test.ts` pins the export list instead. |
| `vitest.config.ts` | DELETE (rewrite) | browser-only, SwiftShader flags, Vitest 2.1 shapes (`name: "chromium"`, top-level `launch` rejected -- report line 106-109). Section 4.5 replaces it. |
| `vite.config.ts` | DELETE | dev server with dotenv / HTTPS / `examples/` opener; the packages use the programmatic `scripts/build-bundle.js` and no dev server (`packages/README.md` line 173: "vite.shared.config.ts (no dev server)"). Any demo page belongs in graphty-element stories, not here. |
| `tsconfig.json` | DELETE (rewrite) | `noUncheckedIndexedAccess: true` and `exactOptionalPropertyTypes: true` contradict design 14.5; `jsx`, `allowImportingTsExtensions`, `moduleDetection`, `importsNotUsedAsValues` are foreign to the package convention. |
| `eslint.config.js` | DELETE | `@stylistic` formatting rules and `naming-convention` conflict with the monorepo flat config (`packages/eslint.config.js`: "ERROR PREVENTION, not stylistic rules", formatting by Prettier). The package is linted by `packages/eslint.config.js` like its siblings. |
| `knip.json`, `.husky/`, `package-lock.json`, `.env`, `.env.example`, `.github/workflows/test.yml`, `.vscode/`, `dist/`, `examples/` | DELETE | npm + husky + dotenv tooling of the scaffold; the staging root has "no Nx, husky, commitlint or release tooling" (README.md line 179); the workflow installs Mesa/Xvfb and runs on GitHub-hosted runners without a GPU (the plan's CI note replaces it); `examples/{data}` is an empty template artifact. |
| `package.json` (root) | REWRITE as a thin staging root or DELETE | the real package.json is `packages/webgpu-graph-algorithms/package.json`; `packages/package.json` is already the pnpm workspace root (lines 1-45). Keep a root README pointing at `packages/`. |
| `STRATEGY.md`, `IMPLEMENTATION_CHECKLIST.md` | DELETE or move to `tmp/legacy/` | stale, pre-design, mention CPU / WebGL fallbacks that the project rule forbids. |
| `README.md` (root) | REWRITE | one screen: what the repo stages, pointer to `packages/README.md` and the plan. |
| `HEADLESS_GPU_REPORT.md` | KEEP (move under `packages/webgpu-graph-algorithms/docs/` or reference from CLAUDE.md) | the verified recipe for the real GPU under headless Chromium; design 14.5 line 4215 cites it by name. |
| `CLAUDE.md` (root, one line: no fallbacks) | KEEP | still the project rule; the package CLAUDE.md repeats it. |
| `.gitignore` (root) | REWRITE | keep `node_modules`, `dist`, `coverage`, `tmp/` (currently NOT ignored at the root -- `tmp/` holds this plan; decide whether the plan directory is committed), `*.tsbuildinfo`, drop the `*.test.js` / `.env` lines. |
| `packages/**` | KEEP untouched | graph-format and graph-io are done and audited. |

---

## 6. Reusable test and benchmark assets

Import rule: after W1 each package is an independent Nx project and `graph-format/test/**` is not
published, so the GPU package must not `import` from `../graph-format/test/...` or
`../graph-format/benchmarks/...` (graph-io has exactly one cross-package test import, of a SOURCE
module through the tsconfig `include` of `../graph-format/src/**`: `packages/graph-io/test/common/text.test.ts`
line 7; benchmarks were re-implemented: `packages/graph-io/benchmarks/measure.ts`). COPY the small
pure helpers below into `test/helpers/` and `benchmarks/`, keeping a comment naming the origin.

| Asset | Where | What to reuse |
| --- | --- | --- |
| Device acquisition, `requireGpu(t)`, `upload()`, `outputBuffer()`, `readback()` (copy-before-unmap), `dispatch()` (bind-group layout from `Binding[]`, `getCompilationInfo()` errors surfaced as failures, `pushErrorScope("validation")`, the 65,535 dispatch assertion), `u32Uniform()` (16-byte padded) | `packages/graph-format/test/audit/gpu-upload.test.ts` lines 26-198 | Becomes `test/setup/gpu.ts` + `test/helpers/device.ts`. The `dispatch()` helper is the seed of the package's own `Kernel` class but stays a test-side copy. |
| `randomEdges(n, m, seed)` LCG with no self-loops, `directedGraph()` (n=4096, m=50000, weights 1..7, `cost` edge column), `undirectedGraph()` | same file lines 200-251 | `test/helpers/graphs.ts`. |
| `randomEdges(nodeCount, edgeCount, seed)` (G(n, m) WITH self-loops and parallels, integer weights 1..10), `stringIds`, `sparseNumericIds`, `makeRandom(seed)` xorshift32 | `packages/graph-format/benchmarks/datasets.ts` lines 1-88, `harness.ts` lines 239-249 | `benchmarks/datasets.ts`. Seeded, identical across hosts. |
| `bench(group, name, { setup, run }, { runs, items, unit })` -> `BenchResult { group, name, medianMs, minMs, maxMs, runs, memoryDeltaBytes, rate, rateUnit }`, `printTable`, `appendSession` -> `benchmarks/results/<host>-node<version>.json` (array of sessions with `date`, `host`, `node`, `cpu`, `exposeGc`, `results`), `run.ts` group selection and `--no-save` | `packages/graph-format/benchmarks/harness.ts` lines 16-249, `run.ts` lines 1-45, `results/dev.ato.ms-node22.22.1.json` | `benchmarks/harness.ts`: `run` becomes `async` (GPU work awaits readback) and a `gpu` field (adapter vendor / architecture / device, requested limits) is added to the session record so results from the RTX 4070 SUPER, a CI runner and a browser are distinguishable. Keep the same JSON shape otherwise so a future merge-benchmarks tool can read both packages. |
| `KARATE_EDGES` (78 edges), `gridEdges(w, h)`, `EdgeSpec` | `packages/graph-format/test/helpers/parts.ts` lines 26-31, 369-467 | `test/helpers/graphs.ts` (pure data; NOT `makeSnapshot` / `makeParts`, which import `src/types/internal.ts`). Zachary's karate club is the canonical small oracle graph for BFS / CC / betweenness / Louvain checks. |
| `assertInvariants(snapshot)` | `test/helpers/invariants.ts` line 300 | do not copy; the GPU package never constructs snapshots except through public factories, and `snapshot.validate()` is the public equivalent. |
| `ModelGraph`, `arbScenario` (fast-check) | `test/helpers/model-graph.ts`, `random-ops.ts` | builder-specific; not needed. |
| Legacy `Graph` class copy | `test/helpers/legacy-graph.ts` | not needed; the GPU package's oracles are the CPU `@graphty/algorithms` results (differential tests, design 16) or hand-written O(n + m) references in `test/helpers/oracle.ts`. |
| Fixture container | `test/fixtures/rich-v1.gsnp` | a small `fromBytes` fixture with columns of every dtype; copy for the "adopted container uploads from file bytes at `arena.byteOffset !== 0`" case. |

Assertions worth carrying over verbatim from `gpu-upload.test.ts` into the walking-skeleton test:
the device accepts 256-byte offsets (`256 % minStorageBufferOffsetAlignment === 0`) and the default
limits (`maxStorageBufferBindingSize >= 128 MiB`, `maxBufferSize >= 256 MiB`,
`maxComputeWorkgroupsPerDimension >= 65535`) (lines 254-262); whole-arena upload with per-segment
bindings equals the CPU views (lines 295-364); cold segments gather / write back (365-433);
undirected doubled storage writes once per edge with equal values (434-480); packed columns
(481-554); 64-arc windowed binding equals a copied window (555-614); container bytes upload at a
non-zero offset (616-666).

---

## 7. Conventions checklist

Every item is enforced today on graph-format / graph-io and applies unchanged:

- Coverage thresholds lines 80 / functions 80 / branches 75 / statements 80 (`packages/graph-format/vitest.config.ts`
  lines 17-22; `packages/vitest.shared.config.ts` lines 22-27), measured by the `node` project on Dawn;
  `src/wgsl/**` excluded; no threshold on the browser smoke project.
- Lint = `eslint` (the root flat config `packages/eslint.config.js`: `strictTypeChecked`, jsdoc
  `flat/recommended-typescript-error` with `require-jsdoc` for exported functions / methods / classes,
  `explicit-function-return-type`, `no-floating-promises`, `no-non-null-assertion`, `eqeqeq`,
  `curly`, `default-case`, `no-nested-ternary`, `prefer-template`, `camelcase` with
  `properties: "always"`, `no-console` allowing only `warn` / `error`, `simple-import-sort`) plus
  `tsc --noEmit`. Test files get the relaxed block (lines 208-245). Benchmarks and scripts are
  ignored by eslint (lines 33, 41) but type-checked by `tsconfig.json`.
- No `eslint-disable`, no `@ts-expect-error`, no `@ts-ignore` (owner rule in
  `/home/apowers/.claude/CLAUDE.md`; graph-io CLAUDE.md lines 163-168 states the house style: JSDoc
  on every exported symbol with `@param name - description` and `@returns`, explicit return types,
  inline `import { type X, y }` qualifiers, `.js` suffixes on relative imports, no default exports,
  no `console.log` in `src/`).
- Plain ASCII in every source, test, script, doc and commit message; `--` for dashes; non-ASCII
  test data built with `String.fromCharCode`.
- Prettier: `tabWidth 4`, `printWidth 120`, `trailingComma all` (`packages/.prettierrc`); not
  enforced by CI (README.md lines 189-195) but run `pnpm exec prettier --write` before hand-off.
- knip clean under the monorepo's pinned version (5.77.4 reports exports used only in their own
  file, README.md lines 169-176): un-export or tag `@public` with a reason.
- No fallbacks: the GPU package throws when no adapter / device exists; there is no CPU path, no
  WebGL path, no SwiftShader acceptance in `src/` (root `CLAUDE.md`; design 10.8). The TEST layer may
  skip with a printed `E_NO_ADAPTER` reason where no device exists, and must FAIL under
  `REQUIRE_GPU=1`; a wrong result is never a skip.
- `GraphFormatError` semantics carried over: absent output is `null`, never `undefined`; optional
  input is `?: T | undefined`; every throwing call leaves state unchanged; errors carry a stable
  `code` and frozen `details` (graph-format CLAUDE.md lines 137-139; errors.ts lines 88-120).
- Arrays: results are `Uint32Array<ArrayBuffer>` / `Float32Array<ArrayBuffer>`, index-aligned,
  attached to tables by reference; `INVALID_INDEX` is the only sentinel; never a bitwise operator on
  an arc index or byte offset; never write into a view.
- Zero runtime dependencies beyond `@graphty/graph-format` (and the types-only `@webgpu/types`);
  `webgpu` (Dawn) reachable only through the `./node` subpath and only dynamically.
- Commit scope `webgpu-graph-algorithms`; `feat!:` / format-version rules of design 13.5 apply.
- Ports: any local server in 9000-9099; the coverage preview is 9058.
- Never `git add` / `commit` / `push`, never `sudo` (owner rules).

---

## 8. Sources

Local files (all read in full or at the cited lines):

- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/src/index.ts` (139 lines, the barrel)
- `.../graph-format/src/constants.ts` (lines 16, 22, 30, 37, 45)
- `.../graph-format/src/errors.ts` (lines 50-120)
- `.../graph-format/src/types/index.ts`, `types/columns.ts` (lines 59-80, 120-193, 255-540, 555-660, 715-727), `types/snapshot.ts` (lines 1-886), `types/builder.ts` (lines 20-125)
- `.../graph-format/src/snapshot/graph-snapshot.ts` (lines 1-30, 300-475, 780-830, 1351-1420, 1480-1503)
- `.../graph-format/src/snapshot/views.ts` (lines 1-130, 192-232, 560-609, 835-943)
- `.../graph-format/src/snapshot/derived.ts` (lines 1145-1160), `snapshot/queries.ts` (export list)
- `.../graph-format/src/builder/arena.ts` (lines 1-192)
- `.../graph-format/src/columns/column.ts` (lines 75-116, 545-665, 1950-2010), `columns/table.ts` (lines 360-380), `columns/remap.ts` (lines 70-175)
- `.../graph-format/src/util/typed-array.ts` (lines 190-215), `util/mask.ts` (lines 20-89), `util/options.ts`, `util/shared-buffers.ts` (export lists)
- `.../graph-format/src/populate/from-edge-arrays.ts` (lines 200-257), `from-csr.ts` (lines 275-300, 660-700), `from-records.ts` (export list)
- `.../graph-format/test/audit/gpu-upload.test.ts` (lines 1-667), `test/audit/gpu-contract.test.ts` (lines 1-60, describe/it list)
- `.../graph-format/test/helpers/{invariants,legacy-graph,model-graph,parts,random-ops}.ts` (headers and export lists), `test/index.test.ts` (lines 1-40), `test/build-output.test.ts` (lines 1-71), `test/types/typed-arrays.test-d.ts` (lines 1, 58-74)
- `.../graph-format/benchmarks/{datasets,harness,run}.ts`, `benchmarks/results/dev.ato.ms-node22.22.1.json`
- `.../graph-format/{package.json,project.json,tsconfig.json,tsconfig.build.json,tsconfig.strict-consumer.json,vitest.config.ts,graph-format.ts,CLAUDE.md}`, `scripts/{build-bundle,bundle-types}.js`
- `.../graph-format/node_modules/webgpu/{package.json,types.d.ts,index.js,README.md}` (webgpu 0.4.0; `@webgpu/types` 0.1.72)
- `.../packages/graph-io/{package.json,project.json,tsconfig.json,tsconfig.build.json,tsconfig.strict-consumer.json,vitest.config.ts,CLAUDE.md}`, `scripts/{entries,build-bundle}.js`, `test/common/text.test.ts` line 7, `benchmarks/run.ts`
- `.../packages/{README.md,STATUS.md,CONFORMANCE.md,package.json,pnpm-workspace.yaml,knip.config.ts,tsconfig.base.json,vitest.shared.config.ts,vite.shared.config.ts,eslint.config.js,.prettierrc,.prettierignore,.npmrc,.gitignore}`, `packages/move/root-touch-points.diff`
- `/home/apowers/Projects/webgpu-graph-algorithms/{package.json,tsconfig.json,vitest.config.ts,vite.config.ts,eslint.config.js,knip.json,.gitignore,.env.example,CLAUDE.md,HEADLESS_GPU_REPORT.md}`, `src/index.ts`, `src/types/index.ts`, `test/setup/*.ts`, `test/helpers/*.ts`, `test/browser/*.test.ts`, `test/unit/index.test.ts`, `.github/workflows/test.yml`
- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` lines 2320-2553 (section 10), 3959-4047 (14.3), 4212-4278 (14.5, 14.6)
- `/home/apowers/Projects/graphty-monorepo/{commitlint.config.js,pnpm-workspace.yaml,package.json (lines 55-61, 130, 148, 158-160)}`, `.github/workflows/ci.yml` (lines 236-330), `algorithms/vitest.config.ts` (read-only)

Not fetched / not verified in this note: the exact Vitest 3.x spelling of per-instance Playwright
launch arguments (`browser.instances[].launch` vs `providerOptions`) -- to be confirmed against the
installed `@vitest/browser` when the config is written; whether `webgpu@0.4.0`'s Dawn build honours
`adapter=` / `backend=` on this container (only `create([])` was exercised by graph-format's audit);
the monorepo's `tools/prepush.sh` and `tools/merge-coverage.sh` contents (only the diff hunks in
`packages/move/root-touch-points.diff` were read); GitHub GPU runner options (a separate note).
