# Staging status: @graphty/graph-format and @graphty/graph-io

Staging workspace for two packages destined for the graphty monorepo (`design/graph-format/graph-format-design.md`
is the normative design; section numbers below refer to it). Each package directory moves into the
monorepo with a plain `mv`; the shared configs here are verbatim copies of the monorepo's.

Last updated: 2026-09-14 (graph-io implemented and integrated, phase IO1; see the graph-io sections at the end).

## Summary

| Package                                      | State                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graph-format` (@graphty/graph-format 0.1.0) | IMPLEMENTED and AUDITED (round 1, verified). Every module of design section 13.1, the full 12.2 surface (126 exported names, checked mechanically against the design listing), 1309 tests (unit + audit suites), coverage 96 percent, lint / typecheck / strict-consumer / knip / build clean.                                                                                                                                                                         |
| `graph-io` (@graphty/graph-io 0.1.0)         | IMPLEMENTED and AUDITED (round 1, verified). The 12.4 contract types, the eight importer / exporter pairs (GEXF, GraphML, GML, DOT, Pajek, CSV, JSON, Neo4j) under per-format subpath exports, the registry with `importGraph` / `exportGraph` / `sniff`, the `children` CSR helper; 4237 tests (unit suites over the full corpus plus the seven audit suites), coverage 97 percent statements / 95 branches, lint / typecheck / strict-consumer / knip / build clean. |

## graph-format: what is implemented

| Module (src/)               | Design sections   | Contents                                                                                                                                                                                                                                                                                                                                               |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `index.ts`                  | 12.2              | The only public barrel: explicit named exports of exactly the 126 names of 12.2 (4 classes, 24 functions, 5 constants incl. the error class, 93 types).                                                                                                                                                                                                |
| `constants.ts`, `errors.ts` | 11.2, 12.2        | `INVALID_INDEX`, `MAX_COUNT`, `FORMAT_VERSION`, `SNAPSHOT_BRAND`; internal `ALIGNMENT`, `IS_LITTLE_ENDIAN`, `WIRE_*`, `CONTAINER_MAGIC`, `ENDIAN_PROBE`; `GraphFormatError` with the 29 codes.                                                                                                                                                         |
| `types/`                    | 12.2              | `columns.ts`, `snapshot.ts`, `builder.ts`, `wire.ts` transcribe every interface; the four classes are `*Contract` interfaces the classes implement, with the public names re-exported type-only from the class modules. `internal.ts` holds the construction contracts (SnapshotParts, MutableColumnParts, TableParts, ViewValues / ViewCache).        |
| `ids/`                      | 4                 | `NodeIdMap` (identity / dense / numeric / string / mixed, lazy reverse map, typed wire form), `Utf8Store` (lazy encode / decode), `EdgeIdIndex`.                                                                                                                                                                                                       |
| `columns/`                  | 5                 | Every dtype (`f32 f64 i32 u32 u8 bool dict string list json`), strides, validity bitmaps, defaults and fills, roles, mutability, `AttributeTable`, dictionaries, dtype inference and widening, growable staging arrays, remap / gather / scatter helpers, `withComponents`.                                                                            |
| `builder/`                  | 6                 | `GraphBuilder` (the complete 12.2 member set incl. `static from`, `addGraph`, `setDirected` expansion, records, extension tables), the 12-step freeze pipeline, compaction, the two stable counting sorts, the 256-aligned arena; `options.ts` holds the shared DuplicatePolicy check of the constructor and the per-freeze override.                  |
| `snapshot/`                 | 3, 7, 11.4        | `GraphSnapshot` (frozen, branded, structuredClone-guarded), every 7.2 view, every 7.3 derived graph, the 3.9 queries, `validate()` at both levels with I1-I13 checks, FNV-1a content hashes and checksum records; `graph-meta.ts` is the one GraphMetaPatch resolver (builder `setMeta` and `fromCsr`).                                                |
| `populate/`                 | 8.1               | `fromEdgeArrays`, `fromCsr` (adopt / copy, arena detection, three validation levels, row sorting), `fromRecords` (node-link records, id coercion, column modes).                                                                                                                                                                                       |
| `wire/`                     | 9                 | `toWire` / `fromWire` (transfer-aware through the 9.1 owner count), the GSNP container (`toBytes` / `toByteChunks` / `fromBytes` / `fromByteChunks`), `includeViews` carried, content-checked (`carried-views.ts`) and installed on the receiver, tagged JSON for non-finite numbers with the `$esc` wrapper, three validation levels, golden fixture. |
| `util/`                     | 5.7, 7.4, 9.1, 10 | Typed-array helpers (alignment, padded u32 views, the plain-buffer predicate of I10, resizable-buffer staging), the public mask helpers, the shared-storage owner count, the enum-option check (`options.ts`, E_UNSUPPORTED for every enum-valued option).                                                                                             |

Docs: `graph-format/README.md` (badges, install, quick start that is executed as a check, invariants,
counting vocabulary, view and derived-graph tables), `graph-format/CLAUDE.md` (structure, commands,
invariants, adding a view / a dtype). Benchmarks: `graph-format/benchmarks/` (`run.ts`, `harness.ts`,
`datasets.ts`, `freeze.bench.ts`, `ids.bench.ts`, `views.bench.ts`, `results/`).

## Verification (graph-format, run 2026-09-14 by the graph-io audit round 1 verification pass, from packages/graph-format)

| Command                                                      | Result                                                                                                                                                                              |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm run lint` (eslint + tsc --noEmit)                      | pass, 0 findings                                                                                                                                                                    |
| `pnpm run typecheck`                                         | pass                                                                                                                                                                                |
| `pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json`    | pass (noUncheckedIndexedAccess + exactOptionalPropertyTypes over dist/graph-format.d.ts; the 16.6 `writeBuffer` proof compiles against the real `GPUQueue` of @webgpu/types 0.1.72) |
| `pnpm run test:run`                                          | 59 files, 1309 tests, 0 failures, ~40 s (the audit suites carry the fast-check soaks and the 100k / 1M graphs)                                                                      |
| `pnpm run coverage`                                          | statements 95.96, branches 95.88, functions 98.83, lines 95.96 (thresholds 80 / 80 / 75 / 80)                                                                                       |
| `pnpm run build:all`                                         | pass; dist/graph-format.js (459 KB unminified ESM + map) and dist/graph-format.d.ts; test/build-output.test.ts passes                                                               |
| `pnpm exec knip --workspace graph-format` (from packages/)   | pass, nothing reported                                                                                                                                                              |
| `pnpm exec prettier --check .`                               | pass                                                                                                                                                                                |
| `node --expose-gc --import tsx benchmarks/run.ts`            | pass (numbers below)                                                                                                                                                                |
| Bundle smoke (`tmp/bundle-smoke.mjs`, `tmp/cycle-smoke.mjs`) | the built bundle and the unbundled dist/src load in both module orders and round-trip a graph                                                                                       |

Test counts by directory: builder 131 (7 files), columns 159 (7), ids 118 (3), populate 126 (3),
snapshot 139 (6), wire 65 (3), util 42 (2), root 52 (errors, invariants, index, build-output),
audit 477 (24 files under test/audit: the eight audit lenses' suites plus `round1-fixes.test.ts`).
The graph-io audit added the widening, column-index and detachString tests, and the
verification pass fixed a latent bug of the Map-of-Maps test model (`test/helpers/model-graph.ts`
dropped self-loops before it could refuse a duplicate edge, so a refused freeze left the model
changed while the builder was, correctly, untouched; one fast-check seed in 200 hit it) and
pinned the counterexample as a deterministic test.
Type-level tests (`test/types/*.test-d.ts`, 8 files) are compile-only and run under both tsconfigs.
Source: ~24.8k lines under src/, ~31k under test/.

## Benchmarks (i9-14900, Node 22.22.1, single thread, median of 5, `--expose-gc`)

Freeze (design 15.4 targets: 25-30 ms directed, 45-55 ms undirected for 100k nodes / 1M edges):

| Benchmark                                                                      | Median                 |
| ------------------------------------------------------------------------------ | ---------------------- |
| fromEdgeArrays 10k nodes / 100k edges, directed / undirected                   | 9.5 ms / 12 ms         |
| builder push 10k/100k (addAnonymousNodes + addEdges), directed / undirected    | 3.4 ms / 2.4 ms        |
| freeze 10k/100k (builder loaded), directed / undirected                        | 2.1 ms / 3.0 ms        |
| fromEdgeArrays 100k nodes / 1M edges, directed / undirected                    | 54 ms / 81 ms          |
| builder push 100k/1M, directed / undirected                                    | 21 ms / 30 ms          |
| freeze 100k/1M (builder loaded), directed / undirected                         | 22 ms / 55 ms (min 46) |
| freeze 100k/1M directed, duplicateEdges "sum"                                  | 87 ms (min 73)         |
| freeze 100k/1M directed, weightDtype "f64"                                     | 26 ms                  |
| re-freeze 100k/1M directed after removing 10 percent of the edges (compaction) | 43 ms                  |

The freeze and ids rows were re-measured by the verification pass of audit round 1 (the full
`node --expose-gc --import tsx benchmarks/run.ts` session, appended to the results file) on a host
with a load average of about 15 from unrelated work, which is why the undirected freeze median sits
at the top of its 45-55 ms band while its minimum (46 ms) matches the earlier quiet-host run (48 ms
median, 45 ms min); the views rows below are the pre-audit numbers (those modules' hot paths were not
changed by the audit, and none of the audit tests reports a regression).

Ids (1M ids unless stated):

| Benchmark                                                           | Median                           |
| ------------------------------------------------------------------- | -------------------------------- |
| intern 100k string ids (addNode x n + freeze) / (addNode only)      | 16 ms / 11 ms                    |
| intern 1M string ids (addNode x n + freeze) / (addNode only)        | 359 ms / 304 ms                  |
| intern 1M sparse numeric ids (addNode x n + freeze)                 | 158 ms                           |
| indexOf, 1M lookups, identity / dense / numeric / string map (warm) | 2.6 ms / 7.7 ms / 65 ms / 113 ms |
| toMap over 1M string ids                                            | 153 ms                           |
| entries over 1M string ids (consumed)                               | 13 ms                            |
| idsSlice of 1M string ids                                           | 4 ms                             |
| toWire of a 1M string id map (encodes the Utf8 store)               | 65 ms                            |

Views and derived graphs (100k nodes / 1M edges, cold snapshot per run):

| Benchmark                                                                         | Median                           |
| --------------------------------------------------------------------------------- | -------------------------------- |
| reverse() directed                                                                | 11 ms                            |
| coo() / edgeList() directed                                                       | 1.5 ms / 11 ms                   |
| edgeList() undirected                                                             | 20 ms                            |
| outDegree() / inDegree() (incl. reverse) / degree() undirected                    | 0.2 ms / 9.5 ms / 6.7 ms         |
| weightedOutDegree() / weightedDegree() undirected                                 | 1.3 ms / 8 ms                    |
| selfLoopArcs() / mate() undirected                                                | 5 ms / 18 ms                     |
| degreeOrder() / degreeOrder({ of: "reverse" }) / isSymmetric() directed           | 0.4 ms / 10 ms / 9.5 ms          |
| toUndirected() directed / withoutSelfLoops() undirected / simplified() undirected | 151 ms / 146 ms / 151 ms         |
| transpose() directed (zero copy)                                                  | 4 ms                             |
| filterEdges(50 percent) / inducedSubgraph(50 percent) directed / undirected       | 63 ms / 50 ms / 82 ms            |
| contract(1000 blocks, weights "sum") undirected                                   | 140 ms                           |
| toWire() / fromWire none / structure / full                                       | 0.1 ms / 0.2 ms / 3 ms / 12 ms   |
| toBytes() / fromBytes none / structure / full                                     | 1.8 ms / 0.2 ms / 3.5 ms / 13 ms |
| validate({ level: "full" }) directed                                              | 12 ms                            |

The full benchmark session is appended to `graph-format/benchmarks/results/<host>-node<version>.json`.

## Deviations from the design (collected from every module)

Where the design is silent or the implementation differs, in the order of the design. Each is a
candidate for confirmation by the owner; none contradicts an invariant.

### Types and surface (12.1, 12.2)

- The four classes are implemented as classes; `src/types/*` holds `GraphSnapshotContract`,
  `NodeIdMapContract`, `AttributeTableContract` and `GraphBuilderContract` (the verbatim 12.2 member
  sets, implemented by the classes) and re-exports the class types under the public names. Every
  public type (`DerivedGraph.snapshot`, `GraphSnapshot.ids` / `nodes`, ...) therefore names the class,
  so class instance types and the types in signatures are interchangeable for consumers.
- `constants.ts` also defines internal `ALIGNMENT`, `IS_LITTLE_ENDIAN`, `WIRE_MAJOR` / `WIRE_MINOR`,
  `WIRE_FORMAT`, `CONTAINER_MAGIC`, `ENDIAN_PROBE`, not exported from the barrel.
- `GraphFormatError.details` is a frozen shallow copy of the supplied record.
- `NodeIdMap` has a public constructor taking an internal storage record; it appears in the d.ts
  but not on the instance type.
- `withComponents(data, from, to, fill)` returns `data` itself when `from === to`; strides outside
  1..16 are `E_COLUMN_TYPE`, a length that is not a multiple of `from` is `E_COLUMN_LENGTH`.
- Negative type tests use `expectTypeOf(...).not...` instead of `@ts-expect-error` (owner rule).
- `test/types/internal.test-d.ts` is excluded from the strict-consumer compile (it imports
  `src/types/internal.ts`, which is not public and would pull the whole source tree under the
  consumer flags); it is still checked by `tsconfig.json`.

### Id map (4)

- Duplicate ids in the `ids` array / F64 of `fromEdgeArrays` / `fromCsr` throw `E_DUPLICATE_ID`
  (details `id`, `indices`); the same in untrusted wire input is `E_INVALID_SNAPSHOT` I11.
- A wire `dense` map violating its own condition (`maxId + 1 > 2 * size` or a value `>= MAX_COUNT`)
  is `E_BAD_SERIALIZATION` at EVERY level (the scan is already paid and the inverse array is sized by
  the largest id); duplicates are detected at every level too. An identity `offset` must keep
  `offset + size - 1` a safe integer at every level (`E_BAD_SERIALIZATION`, I11 at "full"); `-0`
  counts, offsets and numeric ids read as `0`.
- Mixed-kind Utf8 store: `size + 1` offsets covering every row, number rows empty.
- `idsSlice(start, end)` uses `Array.prototype.slice` bounds semantics.
- `toRecord` returns a null-prototype object; when two ids share a `String()` form (always one number
  and one string) `toStringMap` / `toRecord` let the HIGHER index win, exactly as the legacy
  algorithms' assignment in node order does (14.2), and `stringIndex()` prefers the string id (a
  legacy string parameter named the string node). Audit round 1; owner decision pending.
- `gatherNodeIdMap` re-detects the kind (a contiguous run of an identity map stays identity with a new
  offset; a permutation becomes `dense`; a sparse selection `numeric`); 7.3's "identity becomes
  numeric unless prefix" is read as loose wording.
- Identity offset may be any safe integer whose ids stay safe integers.
- Edge id columns may also be `f32` / `i32` / `u8`; `bool` / `list` / `json` are `E_COLUMN_TYPE`;
  `EdgeIdIndex` rebuilds when a mutable id column's version changes.
- Structure-level layout checks of the typed id map live in `nodeIdMapFromTyped`; "full" adds
  finiteness, fatal UTF-8 decode, distinctness and builds the reverse Map eagerly.
- Bulk decodes are not written into the per-row cache; `materialiseDecoded()` converts in one pass.
- UTF-8 encoding is hand-written (byte-identical to TextEncoder, lone surrogates become U+FFFD);
  decoding uses TextDecoder behind an ASCII fast path.

### Columns (5)

- `meta.fill` of `list` / `json` columns is the nominal `""`; an explicit other fill is `E_COLUMN_TYPE`.
- `dict` fill is a string; a string default with no declared options seeds the dictionary as code 0;
  with options the default is representable iff it is a member, else fill is `options[0]`.
- `u32` `refersTo` columns (and the `u32` child of a `refersTo` list) default to fill `INVALID_INDEX`.
- Only integers in range are representable defaults for `i32` / `u32` / `u8`; array defaults reduce
  to one fill only when every lane is the same representable number.
- `markDirty()` recomputes `nullCount` from the validity bitmap; `setAll()` bumps the version.
- `remapColumn(column, remap, newLength)` rewrites `refersTo` values through the SAME remap only when
  the column refers to its OWN index space (a node column with `refersTo: "node"`, an edge column with
  `refersTo: "edge"`); a column referring to the other space has its rows moved and its values left
  alone; a remap of the wrong length is `E_COLUMN_LENGTH`. The cross-space forms are internal
  (`remapColumnWith`, `remapReferences`, `remapTable`, `gatherTable`).
- Dangling references: scalar `u32` becomes `INVALID_INDEX` + unset; list-of-u32 drops dangling items,
  a row whose every item dangled becomes unset and empty.
- Rows of a new space no old row maps to are unset with the fill; the RESULT flips `nullable` to true.
- `set()` with a typed array defaults `nullable` to false, with a JS array to true; inference follows
  5.1 widening (json when every entry is unset; list when every set entry is an array); values are
  coerced per 5.1; a non-representable value is `E_COLUMN_TYPE` instead of typed-array wrapping. A
  typed array over a SharedArrayBuffer or a resizable buffer is `E_UNSUPPORTED` (D-SAB, I10 / I17);
  dict codes beyond the seeded dictionary are `E_COLUMN_TYPE`; a foreign object shaped like a Column
  is `E_COLUMN_TYPE` (reason "foreign column").
- `rename()`, a move to another name / domain, and a decl patch re-wrap into a NEW Column object
  sharing every buffer; a patch that changes shape or fill is `E_COLUMN_TYPE`; `nullable: false`
  on a column with unset rows is `E_COLUMN_TYPE`.
- `paddedU32View()` also returns the codes of a `dict` column; `i32` / `f32` throw
  `E_GPU_INELIGIBLE` from it (their `gpuView()` returns the data).
- `byteLength`: json 0 data bytes (+ validity); dict codes + validity; string materialises the Utf8
  store; list offsets + child + validity.
- `column.isSet(row)` is total; `table.isSet(name, row)` throws `E_UNKNOWN_COLUMN` for an absent column.
- `createColumn()` recomputes `nullCount`; a validity bitmap on a non-nullable meta is `E_COLUMN_TYPE`;
  a u8 data array failing the padded-view predicate is `E_COLUMN_ALIGNMENT` at the factory.
- Every bitmap the package builds keeps its trailing bits clear (the bitmap module's convention), so
  a column's validity words and checksum do not depend on the construction path.
- `ColumnMeta.default` / `options` / `extra` are deep copies of the declaration (a caller's live
  object never reaches the metadata); each column keeps its own copy of the default and `value()` hands
  out a fresh copy of a structured default for an unset row. JSON values deeper than 256 levels are
  `E_COLUMN_TYPE` (reason "nesting").
- A `dict` fill that is not a dictionary member is interned on demand only where rows holding it are
  SET (a non-nullable column); unset rows keep code 0 without adding a member.
- A `"sum"` reducer over an i32 / u32 / u8 column (`simplified`, `contract`) widens the result to f64
  (as `"mean"` does) so a total never wraps; `"min"` / `"max"` keep the dtype.
- `clone()` deep-copies every buffer; a json column's values array is copied, the values shared.
- f64 text inference: no leading zeros in the integer part, sign and exponent allowed, must be finite;
  an integer text outside i32 range is f64.
- `AttributeTable` carries exactly the 12.2 members; declare / withColumns / uniqueness helpers are
  module functions (`createTable`, `declareColumn`, `tableWithColumns`, `verifyUniqueColumn(s)`).

### Builder (6)

- `GraphSink` (8.3, 12.2) has two optional members the design listing lacks,
  `widenNodeColumn?(column, dtype)` / `widenEdgeColumn?(column, dtype)`, and `GraphBuilder`
  implements them: an inferred (auto-declared) column is widened along the 5.1 order
  (bool -> i32 -> f64 -> string) so a text importer can give a column its text dtype whatever
  the values imply (`E_UNKNOWN_COLUMN` for an unknown name, `E_COLUMN_TYPE` for a declared column
  or a narrowing). Added in the graph-io audit round 1 (silence-fill (b): 8.3 says the importer
  reproduces the 5.1 grammar, which needs it). Column names resolve through a `Map` index rebuilt
  when the column array changes (`ColumnNameIndex`), and string ids / interned strings are copied
  out of V8 sliced strings (`detachString`, strings of 13+ characters) so a chunk is not retained
  by the ids cut from it.
- `weighted: false`: no arc weight array; a weight other than exactly 1 is `E_INVALID_WEIGHT`
  (reason "unweighted builder"); a `"sum"` merge on such a builder stores no multiplicities (the
  builder never gains a weight array).
- `addAnonymousNodes` after named nodes: the new id is the index; if that number is already an id,
  `E_DUPLICATE_ID`; a negative or non-integer count is `E_INDEX_RANGE`.
- Codes the design does not assign: `removeNode` of an unknown / removed id and `removeNodeByIndex`
  of a dead index are `E_UNKNOWN_NODE`; `edgeEndpoints` / `edgeWeight` / `setEdgeWeight` on a dead
  edge, out-of-range indices in value setters / `outEdgesOf` / `inEdgesOf` / `findEdges`, and a bad
  extension handle are `E_INDEX_RANGE`; a stale `ColumnHandle` is `E_UNKNOWN_COLUMN`; length
  mismatches in bulk adds are `E_COLUMN_LENGTH`; a constructor option, a per-freeze `FreezeOptions`
  field or an `addGraph` option outside its set is `E_UNSUPPORTED` (as is every other enum-valued
  option of the package, through `util/options.ts`); a bad `setMeta` field is `E_COLUMN_TYPE`.
- A `refersTo` value is range-checked at the write (`E_INDEX_RANGE` unless below the referenced
  space's current bound or `INVALID_INDEX`); a scalar `INVALID_INDEX` written to a nullable refersTo
  column unsets the row, to a non-nullable one it is `E_COLUMN_TYPE`. A dangling reference at
  compaction becomes `INVALID_INDEX` with the row unset, and a non-nullable refersTo column becomes
  nullable for it (the rule of `remapColumn`); a non-nullable scalar refersTo column whose rows were
  not all written (the unwritten rows hold the `INVALID_INDEX` fill) freezes the same way, nullable
  with exactly those rows unset, while the builder's own declaration stays non-nullable. So every
  frozen snapshot satisfies I12 (the snapshot-side `AttributeTable.set()` is the one write path that
  does not range-check references; see the known gaps).
- A merging freeze rewrites `refersTo: "edge"` values to the SURVIVOR of a merged edge, the same map
  `FreezeReport.edgeRemap` reports (5.11, 7.3); a reducer that produces NaN (`"sum"` over +Infinity and
  -Infinity) is `E_INVALID_WEIGHT` before anything is renumbered (6.3 step 8); a reduced weight makes
  the survivor's weight explicit (a set row of the shadow column), so multiplicities survive
  `GraphBuilder.from()` and exporters.
- The shadow weight column (3.7) is named `graphty.weight`; not created for 0 edges; a user column
  already holding role `weight` makes the freeze throw `E_DUPLICATE_ROLE` when the shadow is needed.
- Dict dictionaries are copied at freeze (I17: the builder keeps interning into its own array).
- Compaction gathers into a fresh staging object adopted only on success, so every freeze-time throw
  leaves the builder untouched.
- `FreezeReport.compacted` is true whenever anything was renumbered.
- `setDirected(true, { expand: true })`: a self-loop is not mirrored; only live edges are mirrored;
  mirror weights copy the original's value and explicit / omitted status.
- Inferred columns stage values as written and coerce once at freeze to the widened dtype
  (bool -> i32 -> string yields "true", not "1").
- `FreezeReport.widened` lists widenings since the previous freeze.
- `freeze({ release: true })` empties staging but keeps graph-level values, meta, options and the
  direction lock; `clear()` drops graph values and meta too.
- `GraphBuilder.from(snapshot)` consumes the role-`weight` edge column as the weight source; `weightDtype`
  defaults to "f64" when that column is f64; `weighted` defaults to true when the snapshot is weighted
  (a declared array is never dropped, 3.7); expected counts default to the snapshot's.
- `addGraph`: graph columns overwrite by name, meta is left alone, extension tables are appended by
  name, a snapshot with an offset-0 identity map appended to an empty anonymous builder uses
  `addAnonymousNodes`; declared columns present in both graphs widen to the union dtype; a snapshot of
  the other direction is `E_DIRECTED` (reason "direction mismatch"; the design is silent); a weighted
  snapshot keeps its weight array unless the builder is `weighted: false`; everything that can be
  refused (duplicate ids under "error", weights, roles, extension columns, the count limits) is checked
  before the first mutation.
- `E_DUPLICATE_EDGE` details name the edges in the builder's index space.
- An unset value written by name to a column that does not exist is a no-op; `null` is unset on
  inferred columns and a value on a declared json column.
- Counting sort: arc materialisation is fused into pass 1; pass 1 scatters (source, edge) pairs into
  one `U32(2A)` transient with a bitmap over pass-1 positions for the declared orientation (undirected
  only), plus 2 x U32(n + 1) counts; the weights and (directed) `edgeToArc` of 6.3 step 6 are written
  in sequential post-passes rather than inside the pass-2 scatter (same result, measured 2x faster);
  the design's 15.2 one-lane 4A directed transient measured slower than the 8A pair layout.
- `outEdgesOf` / `inEdgesOf` follow the snapshot's semantics: directed, the declared orientation;
  undirected, every incident edge (a self-loop once) from both queries (I7, `inDegree === outDegree`).
- `dirty` is true for a never-frozen builder; bulk adds bump `mutationCount` once per call.
- `addNodeRecord` / `addEdgeRecord` validate every attribute value (and the weight) before the element
  is added, so a bad value leaves the builder untouched (11.1); `setDirected(true, { expand: true })`
  checks both column declarations before writing either; a null / undefined weight value is omitted,
  a non-number is `E_INVALID_WEIGHT`.
- `dispose()` makes every getter throw `E_BUILDER_DISPOSED` too (6.1: every further call).
- `E_TOO_LARGE` is checked at every add call (a conservative 2-arcs-per-edge estimate in
  `addEdgesByIds` before ids resolve) and again at freeze.

### Snapshot, views, derived graphs (3, 7)

- The structuredClone guard is an ENUMERABLE own function-valued property `__graphtyNoStructuredClone`
  (a non-enumerable one is skipped by the structured clone algorithm on Node 22, so the guard would
  be silent); it appears in `Object.keys(snapshot)`.
- Lazy state lives in a TS `private readonly state` record defined as a NON-enumerable own property
  (plus a module-private WeakMap), so `Object.keys(snapshot)` lists exactly the public fields and the
  clone guard.
- `validate()` compares recorded checksums BEFORE the level checks; "full" runs the I8 NaN check
  before the I7 pairing walk.
- `contract(identity, { parallel: "keep" })` reports `edgeRemap === null` (edge maps null iff the edge
  space is unchanged, applied to contract too).
- `contract` remaps extension tables' `refersTo` values through the block map rather than dropping
  them; `refersTo` node / edge columns are dropped even when named in the reducers.
- `toUndirected` / `simplified` on an unweighted source with `weights: "sum"` materialise
  multiplicities (the design states this rule only for contract); other reducers keep `null`.
- `isSymmetric` compares forward and reverse rows position by position on a simple graph and, on a
  multigraph, each run of parallel arcs as a sorted weight multiset (3.6: closed under reversal with
  equal weights, whatever the edge order).
- `validate()` rejects a detached column or id store with `E_DETACHED` before any level check, refuses
  an unknown `level` with `E_UNSUPPORTED`, recomputes every column's `nullCount` from its bitmap, checks
  the plain-ArrayBuffer half of I10 for every core array and every column array, and reports a
  violated `unique` column as `E_INVALID_SNAPSHOT` (I12, reason "unique", the freeze-time code in
  `details.cause`); the freeze keeps `E_DUPLICATE_ID` / `E_DUPLICATE_EDGE_ID`.
- The view objects (`coo()`, `edgeList()`, `degreeOrder()`) and `ArenaSegment` records are frozen.
- Column and id-map accessors on storage that was transferred away throw `E_DETACHED` (derived from the
  array state, like the core), including the lazy reverse-map build behind `indexOf` / `has` /
  `requireIndex` / `indicesOf` of a wire-decoded string or mixed map decoded below "full" (the level
  that builds the map eagerly); `cachedViews()` of a detached snapshot drops the caches and reports none.
- `transpose()` adopts the cached reverse arrays as the core (arena null); every other derived graph
  goes through the builder's counting sort (`sortIntoCore`) into a fresh arena.
- Derived graphs carry the source's `label` and `meta` and share its `graph` table; `withColumns()`
  clones the graph table too and shares the extensions map; it propagates the checksum setting.
- `dropCaches()` also drops materialised identity permutations and their checksum records, and the
  cached `gpuView()` f32 copies of f64 columns.
- Checksum records of a reverse view cover the members materialised when the view was first computed.

### Populate (8.1)

- `fromCsr` flag claims are verified only under "full" (8.1 + 9.5); under "structure" and "none"
  every claimed PREDICATE flag is stored as given and only unclaimed flags are computed; the two
  presence flags (`weighted`, `arcToEdgeIsIdentity`) always follow the arrays and a contradicting claim
  is I9 at every level (a claim never changes which arrays are adopted; identity is decided from the
  supplied `arcToEdge` itself).
- `fromCsr` arena detection additionally requires the present, non-empty, non-identity arrays to lie
  in hot-to-cold order without overlap; a derived `edgeToArc` makes the arena null.
- `fromCsr({ copy: true })` copies the core into a fresh freeze-style arena and every typed column
  and ids array; core and column arrays over a SharedArrayBuffer or a resizable buffer are always
  copied (D-SAB, I10 / I17).
- `fromCsr` derives a missing `edgeToArc` as the lowest arc holding each edge; an explicit identity
  `arcToEdge` is treated like an absent one; an `edgeToArc` next to an identity `arcToEdge` must be
  the identity (I5) unless the level is "none".
- `fromCsr` "full" on an unsorted undirected input rebuilds and compares rows as multisets.
- `fromCsr` codes: wrong array class is `E_INVALID_SNAPSHOT` (reason "dtype"); a bad `nodeCount` is
  `E_TOO_LARGE`; `ids.length !== nodeCount` is `E_COLUMN_LENGTH`; a directed `edgeCount !== arcCount`
  is I6 and a missing undirected `arcToEdge` is I5 at every level; an unknown validate value is
  `E_UNSUPPORTED`. `CsrInput.meta` is resolved by a local `resolveGraphMeta()` mirroring `setMeta`.
- `fromEdgeArrays`: a bad `nodeCount` without ids is `E_INDEX_RANGE`; count / length disagreements
  are `E_COLUMN_LENGTH`; a repeated id is `E_DUPLICATE_ID`; an `options.directed` that disagrees with
  `input.directed` is `E_DIRECTED`; F64 weights select `weightDtype: "f64"` unless overridden; node / edge columns are
  staged in the builder (so merging policies gather them) rather than attached after the freeze.
- `fromRecords` defaults `weightDtype` to "f64"; reserved keys never become columns; the record is
  pushed one scalar at a time; columns modes "json" / "none" / a `ColumnDecl[]` as described in the
  from-records module; a bad decl list is `E_COLUMN_EXISTS` / `E_COLUMN_TYPE`; id coercion
  "canonical" excludes the text "-0"; "string" applies `String()` to numbers, booleans, bigints and
  null; "number" applies `Number()` to strings only; "keep" rejects anything else; a record missing
  its id or an endpoint is `E_INVALID_ID`; `nodeId: null` (d3 v3) appends nodes anonymously with
  index endpoints.

### Wire (9)

- "none" still checks the manifest shape AND every `WireBufferRef` (dtype, alignment, byteLength,
  range) before a typed array is built; overlap, nullCount, arena agreement, name uniqueness and
  `validate()` run only at "structure" / "full".
- The 9.1 owner count is `src/util/shared-buffers.ts`: every `AttributeTable` claims its columns AND
  the buffers they view, every `GraphSnapshot` claims its core buffers, the id map and its materialised
  buffers, and its tables at construction; a lazily materialised store (a string column's Utf8 store,
  a numeric / mixed id map's arrays) claims its buffers when it is built; a zero-copy `slice()` notes
  its source buffer shared; the second claim marks the target shared, so a same-realm `fromWire`
  receiver, a slice attached elsewhere or a Column moved between tables all make a later transfer copy
  rather than empty a sibling. `transpose()` marks the adopted reverse arrays shared. The wire module
  treats a buffer as shared when the buffer, its column or its table is shared and decides copies after
  every array is placed. `transferables()` returns the buffer list of the most recent
  `toWire({ transfer: true })` (exclusive originals plus the fresh copies) so
  `postMessage(wire, snapshot.transferables())` never transfers a buffer the wire does not carry
  (`includeColumns: false`); before any transfer-mode call it is the exclusive set of the default wire
  shape.
- `includeViews`: only resident views are carried; on receipt they are installed on the snapshot at
  "none" (class and length checked, `E_BAD_SERIALIZATION` otherwise) and at "structure" after an exact
  content check against the core (`src/wire/carried-views.ts`: index views arc by arc, the f64 views
  against the snapshot's own computation; `E_INVALID_SNAPSHOT` I2 / I17 with `details.view`), and
  recomputed under "full"; views that alias another view on an undirected snapshot (`reverse`,
  `inDegree`, `weightedInDegree`, `reverseDegreeOrder`), `mate` on a directed snapshot and unknown names
  are ignored.
- `includeColumns: false` omits extension tables as well.
- json column values use the same `{ "$num": ... }` tagging as metadata; a user object whose ONLY key
  is `$num` or `$esc` is wrapped as `{ "$esc": { ... } }` so it never collides with the tag (wire 1.0
  as shipped; the golden fixture was regenerated); a set row is JSON text, an unset row empty text.
  Manifest values nested deeper than 256 levels are `E_BAD_SERIALIZATION` (reason "nesting").
- `meta.extra["graphty.skippedColumns"]` holds `{ domain, table, name, dtype }` records.
- Every container segment, including the last, is padded to 256 bytes, so the container length is a
  multiple of 256; `toByteChunks` yields header + manifest then one padded chunk per segment.
- `fromByteChunks` accepts any split; a segment inside one chunk over a plain ArrayBuffer is adopted,
  one spanning chunks or inside a SharedArrayBuffer is copied; arena is always null.
- `fromWire` copies SharedArrayBuffer entries and rejects non-buffer entries; `copy: true` copies
  buffers wholesale so the arena descriptor still applies; the mutable-column overlap rule is keyed by
  the ArrayBuffer object (absolute offsets), so listing one buffer twice in `wire.buffers` hides
  nothing; error messages describe untrusted values by type (a bigint or null-prototype value never
  reaches `JSON.stringify`).
- Column factory errors while rebuilding are rethrown as `E_BAD_SERIALIZATION` with `details.cause`;
  O(1) parts inconsistencies stay `E_INVALID_SNAPSHOT`.
- Inconsistent identity flags / weights / permutation refs in a manifest are `E_BAD_SERIALIZATION`
  at every level; a header minor differing from `manifest.wire[1]` is `E_BAD_SERIALIZATION`; a newer
  minor is accepted and unknown fields ignored.
- `producer` is the constant `WIRE_PRODUCER = "@graphty/graph-format@0.1.0"`, pinned to package.json by
  `test/build-output.test.ts` (importing package.json into src would emit it under dist/).
- Encoded dictionaries and json text are cached in wire-module WeakMaps and re-encoded when detached.
- `manifest.copied` lists BUFFER indices.
- The wire module and the snapshot module import each other (hoisted functions only, call-time use);
  no codec registry, so `toWire()` and friends can never throw for a missing wire module.

### Util (5.7, 6.2, 7.4, 10.3)

- `layoutSegments()` reports the arena byte length as the end of the last non-empty segment (no
  trailing padding), matching the 10.3 worked numbers.
- A fresh resizable staging buffer reserves `max(byteLength, 256 MiB)` of address space; growth past
  it reallocates with a doubled reservation; a RangeError on the reservation itself retries with
  `maxByteLength === byteLength`.
- Growable capacity is rounded to 16 elements then 64 bytes.
- Every bitmap keeps bits `>= N` clear; counting / iteration take an explicit bit count.
- The bitmap module holds only the helpers the package calls (`bitmapAnd` / `bitmapOr` /
  `bitmapAndNot` / `bitmapFromIndices` / `bitmapGather` / `bitmapResize` / `bitmapNextSet` /
  `bitmapIsFull` / `bitmapFill` / `bitmapSetBits` were dead and were removed in audit round 1).
- `checkMaskLength` is the single implementation of `E_MASK_LENGTH`.
- Growable staging is `GrowableTypedArray` (any of the five numeric classes, also dict codes) and
  `GrowableBitmap` (bool and validity).

## Known gaps and open issues

- Freeze speed: the 15.4 targets are met (22 ms directed vs 25-30, 44-48 ms undirected vs 45-55 for
  100k / 1M). Two derived costs stay above the design's estimates: compaction after removals is ~32 ms
  of the 44 ms re-freeze (6.7 budgets 10-15 ms; the floor is the incidence-list rebuild, four random
  accesses per edge), and a merging freeze is 76-110 ms (6.7 implies ~47 ms; the compaction floor is
  paid once more for the step-7 repeat plus the survivor bookkeeping of `walkDuplicates`). Derived
  graphs that rebuild a core (`toUndirected`, `simplified`, `withoutSelfLoops`, `contract`) take
  140-150 ms on the same graph, mostly in edge-list extraction and grouping before the sort.
- `AttributeTable.set()` on a snapshot checks a typed array's length, dtype, dictionary codes and
  buffer class (5.7) but not `refersTo` values: a table knows only its own row count, not the other
  index space's, so `nodes.set("p", data, { refersTo: "edge" })` cannot be bounded there. A consumer
  can therefore attach a column with an out-of-range reference (or `INVALID_INDEX` in a non-nullable
  column) and `validate()` is what reports it (I12). The builder range-checks every write. Found in the
  verification pass of audit round 1; owner decision pending on whether `set()` should check the
  same-space case and refuse the cross-space declaration.
- `E_TOO_LARGE` at the 0xFFFFFFFF-th element is covered through count validation only.
- The big-endian writer refusal is tested through `assertLittleEndianHost(false)` only.
- Checksum records of a reverse view do not cover members materialised after the view was recorded.
- The three weight flags (`allWeightsOne`, `nonNegativeWeights`, `finiteWeights`) describe the f32 arc
  array, as 3.8 defines them; with `weightDtype: "f64"` the kept shadow can disagree (1 + 2^-30 rounds
  to 1, -1e-50 to -0, 1e39 to Infinity). A consumer that substitutes the f64 shadow for the arc
  weights (14.1) must not branch on these flags; `SnapshotFlags` says so. Owner decision pending on
  whether the flags should instead describe the exact staged weights when a shadow is kept.
- `runFreeze` (about 170 lines) and five other functions between 100 and 150 lines
  (`nodeIdMapFromTyped`, `columnOfValues`, `gatherColumn`, `sortIntoCore`, `createColumn`) exceed or
  approach the 13.4 threshold; each is one dtype / kind switch with independent cases. Not split in
  audit round 1 (no behavioural benefit, real regression risk).
- `remapColumnWith` / `invertRemap` (columns/remap.ts) and `invertOrigin` (snapshot/derived.ts),
  `allocLike` / `allocVector` / `allocNumeric`, and the builder's `checkRole` versus the table's are
  near-duplicates with different signatures or semantics; left as they are.
- The doc-silent choices listed above (notably `weighted: false` semantics, the `graphty.weight`
  shadow column name, the carried-views policy under "full") should be confirmed by the owner before
  graph-io depends on them.
- `test/builder/model.test.ts` runs 200 fast-check scenarios by default; `FC_RUNS=4000` is the soak.
- The `perf` vitest project and the pre-push `BENCH=1` hook of 15.5 are not wired (no CI here);
  `benchmarks/run.ts` is the manual equivalent and appends results.
- graph-io (phase IO1) is implemented; see the graph-io sections below. Of the two graph-io
  findings that touched the core, the widening one is resolved (`GraphSink.widenNodeColumn` /
  `widenEdgeColumn`, optional on the sink and implemented by `GraphBuilder`, so a text importer
  can give a `2.0`-style column its f64 dtype); `GraphBuilder.setMeta()` still replaces `extra`
  as a whole, so an importer that records `meta.extra.<format>` overwrites extras a caller had set
  on the sink (owner decision below). The graph-io audit also made the builder's column-name
  lookup an index (`ColumnNameIndex`) and detaches V8 sliced strings on the id and dictionary
  paths (`detachString`).

## Audit round 1 (2026-09-13)

Eight adversarial audits (data-model invariants, builder lifecycle and compaction, serialisation and
transfer, the GPU memcpy contract on real hardware, differential tests against the legacy Graph class,
freeze hot-path performance, API conformance to section 12.2, code quality) reported 66 findings: 27
major, 39 minor (four of the performance findings were fixed by that auditor directly and are counted
as applied). Every finding pinned by a deliberately failing test now passes by a code change; no test
was weakened, and where an auditor's own two findings (or their test and their proposed fix)
contradicted each other the test was rewritten to the design's semantics and the change is named
here.

Applied (by area):

- Builder: reducer NaN is `E_INVALID_WEIGHT` and the reference model agrees; a merge stores an
  explicit weight on the survivor (multiplicities survive `from()`); `weighted: false` never gains a
  weight array; refersTo edge values follow the survivor; a dangling reference on a non-nullable
  refersTo column makes it nullable; refersTo writes are range-checked (`E_INDEX_RANGE`); the
  per-freeze `duplicateEdges` override and every other `FreezeOptions` field are validated; `addGraph`,
  `addNodeRecord`, `addEdgeRecord` and the expansion are atomic; `addGraph` refuses a direction
  mismatch (`E_DIRECTED`); `from()` / `addGraph` keep a declared weight array; every getter throws after
  `dispose()`; `new GraphBuilder(undefined)` is `E_UNSUPPORTED`; `outEdgesOf` / `inEdgesOf` on an
  undirected builder answer with every incident edge.
- Columns: trailing validity bits clear everywhere; deep-copied metadata JSON and per-column default;
  `set()` refuses SharedArrayBuffer / resizable views, unchecked dict codes and foreign objects;
  `remapColumn` rewrites only own-space references and refuses a short remap; integer `"sum"` widens to
  f64; the dict fill code always names a member for set rows; `E_DETACHED` from every accessor of a
  transferred column; frozen view objects.
- Snapshot: non-enumerable `state`; `isSymmetric` as a multiset per parallel run; `validate()` checks
  detachment, `nullCount`, the plain-buffer half of I10 and reports `unique` violations as
  `E_INVALID_SNAPSHOT`; enum options (`validate.level`, `degreeOrder.of`, the reducers, `parallel`,
  `selfLoops`, `adopt`, `onMissing`, `foldArcs`) are `E_UNSUPPORTED` when unknown.
- Ids: `toRecord` / `toStringMap` keep the legacy last-wins rule, `stringIndex` prefers the string id;
  dense and identity bounds hold at every level; `-0` never leaks; a numeric map encoder refuses a
  non-number; `E_DETACHED` after a transfer.
- Populate: presence flags never come from a claim; SharedArrayBuffer / resizable columns are copied;
  `options.directed` must agree with the input.
- Wire: the `$esc` wrapper (golden regenerated); type-based error rendering; a 256-level nesting limit;
  the overlap rule by buffer identity; carried views content-checked at "structure"; `transferables()`
  follows the last transfer-mode wire; buffer-level holders for tables, id maps and materialised
  stores; `ArenaSegment` records frozen; `E_DETACHED` through `validate({ checksum: true })`.
- Quality: 24 dead exports removed with their tests (the 5.1 text grammar stays, tagged `@public` as
  the reference implementation graph-io must reproduce); `hasLoneSurrogate`, `seedDictionary`,
  `representableNumber`, `isArrayLikeNumbers`, `dtypeOfArray`, `describeId`, `isPlainObject`,
  `isIdentity`, `CORE_ORDER`, `copyCoreIntoArena`, `encodeUtf8Rows`, `emptyParts` and the graph-meta
  resolver each exist once; five JSDoc blocks corrected; the three internal constructors are
  `@internal` and stripped from the d.ts; `edgeIndexOf` caches its role lookup on the table's
  column-set version; string column getters no longer allocate; the 16.6 proof compiles against the
  real `GPUQueue` (`@webgpu/types` devDependency); the seven `not.toThrow()`-only tests assert values.
- Performance: the perf auditor's counting-sort / compaction / bulk-push rewrites are kept; the
  numbers above are the post-audit measurements.

Rejected or deferred (design wins, or an owner decision):

- The weight flags describe the f32 arc array (3.8 letter); documented, owner decision pending.
- `ids: "number"` applies `Number()` per cell (4.1 says so explicitly: it may merge cells and the
  importer counts the merges); `""` becoming `0` is that rule, not a fallback.
- The knip entry list (packages/knip.config.ts mirrors the monorepo's) was not changed.
- The wire minor stays 0: 9.2 pins the header at `minor = 0` and no 1.0 container has shipped, so
  the `$esc` wrapper is part of wire 1.0.
- `includeViews` naming a non-resident view stays a silent omission (G7 pairs it with `prepare`).
- Function-length refactors (runFreeze and five others) deferred.

Verification pass (same day, adversarial re-check of every critical / major finding against the code,
the tests and the design): every claimed fix reproduced as resolved, with two residuals found and
fixed:

- The "non-nullable refersTo column left SET with INVALID_INDEX" fix covered the compaction path
  (`gather`) only; the path with nothing to compact froze a non-nullable scalar refersTo column whose
  rows were never all written with those rows SET at the `INVALID_INDEX` fill, and `validate()`
  rejected the snapshot (I12). `StagingColumn.toColumn` now hands out a nullable column with exactly
  those rows unset (`unsetFillReferences`, src/builder/compact.ts); pinned in
  `test/audit/builder-compaction-columns.test.ts`.
- The "detached accessors throw E_DETACHED" fix guarded `idAt`, but the lazy reverse-map build of a
  wire-decoded string / mixed map (decoded below "full", which builds the map eagerly) bulk-decoded
  the Utf8 store first, so the first `indexOf` / `has` / `requireIndex` / `indicesOf` after a
  consuming transfer was a raw TypeError; `reverseMap()` now asserts attachment first
  (src/ids/node-id-map.ts); the detached-accessor test in `test/audit/wire-probes.test.ts` now covers
  every id-map lookup on string, mixed and numeric maps at the default level.

One new minor gap recorded above (snapshot-side `AttributeTable.set()` does not range-check
`refersTo` values). The eight rejections were checked against the design text they cite (4.1 for
`Number()` per cell, 3.8's predicate definitions, the 9.2 header pin at minor 0, 11.1 for the
refersTo write, G7 for `includeViews`) and stand.

Test changes that were not simple additions: `test/audit/invariants-weights.test.ts` (the model and
the builder now both refuse the NaN reducer instead of the test expecting a NaN weight, which I8
forbids), `test/audit/quality-defects.test.ts` (an out-of-range refersTo write is refused at the
write, as 11.1 requires, instead of being silently unset), `test/audit/wire-probes.test.ts` (the
detached-column checksum scenario now detaches a column from outside the owner count, since the
same auditor's holder finding makes the original same-realm scenario copy instead),
`test/audit/differential-structure.test.ts` (identity offsets are safe integers), and the unit tests
that pinned the old deviations (undirected `outEdgesOf`, `options` after `dispose()`, the u32 `sum`
dtype, `bool` trailing bits, the unique-column code, the lower-index-wins rule).

## graph-io: what is implemented (phase IO1, 2026-09-14)

Design sections 8.2-8.6, 12.4, 13.1 and the io rows of the decision log. The eight formats were
implemented by one agent each on top of a shared foundation (src/common), then integrated (root
barrel, registry, sniffing, children CSR, subpath exports, multi-entry build, docs).

| Module (src/)      | Design     | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index.ts`         | 13.1       | The root barrel: the 12.4 types and `ImportError`, the registry, sniffing and children helpers, every format's subpath surface, and the shared helpers a third-party plugin builds on (report builder, input reader, option resolution, direction resolver, id coercion, text grammar, capability check, loss / issue codes).                                                                                                                                                                                                                                                     |
| `types.ts`         | 12.4       | `ImportInput`, `CommonImportOptions`, `GraphImporter`, `ExportCapabilities`, `LossNote`, `CommonExportOptions`, `GraphExporter`, `IssueCategory`, `ImportIssue`, `ImportReport`, `ImportError` (`E_IMPORT`, carries the partial report), transcribed verbatim.                                                                                                                                                                                                                                                                                                                    |
| `registry.ts`      | 8.2, 8.4   | `FormatRegistry` (register / lookup / `sniff` / `sniffAll` / `importGraph` / `exportGraph` / `exportGraphToString` / `checkExport`), `createRegistry()`, the default `registry`, and the top-level `importGraph` (sniff, builder seeded from the common options with `directed: true` as the placeholder, import, `freezeWithReport`), `exportGraph`, `exportGraphToString`, `checkExport`, `sniff`. A stream is peeked for at most 8 KiB and replayed to the importer.                                                                                                           |
| `sniff.ts`         | 8.2        | `rankFormats` / `sniffFormat` over the registered importers' `extensions`, `mimeTypes` and `sniff(head)`: content matches score `0.5 + 0.35 * content + 0.1 * ext + 0.05 * MIME`, hints alone at most 0.4, ties in registration order (`GRAPH_FORMATS`: json, graphml, gexf, csv, gml, dot, pajek, neo4j). `sniffJsonDialectHead` classifies a whole or truncated JSON head by the importer's own dialect rule (a key-skeleton scan for truncated heads).                                                                                                                         |
| `children.ts`      | 7.1, 5.10  | `childrenCsr(snapshot, { column })` / `childrenFromColumn`: the inverse of a `parent` (u32) or `parents` (list of u32) column as a CSR (`rowPtr`, `children`, `roots`, `dropped`, `unreachable`) with `childrenOf` / `childCount` / `hasChildren` / `isRoot` / `parentCountOf` and `depthFirst()` (pre-order from the roots, cycle members forced to the top, every node once). Used by the GraphML (nested graphs) and DOT (cluster blocks) exporters.                                                                                                                           |
| `common/`          | 8.4-8.6, 5 | codes.ts (the one definition of every shared issue / loss code), report.ts, options.ts (`reportSinkOptions` / `reportUnusedOptions`), input.ts, ids.ts, text.ts (the 5.1 grammar, `TextCellWriter`), declared-types.ts, temporal.ts, lists.ts, attributes.ts (`declareResolved`: the 5.6 rename rule), weights.ts (`explicitWeights`), direction.ts (`DirectionResolver`, `pairFolding`), export.ts (`LOSS`, `checkCapabilities` + `CheckExtras`, `sanitizeIds`), xml.ts (the streaming XML tokenizer, `xmlIllegalTextNotes`), escape.ts, format.ts (`formatDecimal`), writer.ts. |
| `formats/gexf/`    | 8.4, 8.5   | 1.2 and 1.3; a streaming reader over the shared XML tokenizer (single pass, bounded memory, line numbers); declared attributes, viz roles, lifetimes / spells / timestamps / intervals (XSD `startopen` / `endopen` in 1.2), dynamic values as `temporal:*` extension tables, pid / parents deferred; exporter folds pairs, writes explicit weights only, streams the edges.                                                                                                                                                                                                      |
| `formats/graphml/` | 8.4, 8.5   | The shared streaming XML tokenizer (single pass, bounded memory, line numbers); keys of every attr.type, `for="all"`, defaults and descs, ports, nested graphs, hyperedges (error / skip / star / clique), yFiles trees as json; exporter writes nested graphs from the children CSR, the label slot, mangles ids to NMTOKEN on request, folds mutual pairs to undirected edges.                                                                                                                                                                                                  |
| `formats/gml/`     | 8.4, 8.5   | A NetworkX-compatible lexer, two-walk import over one token list (schema first, typed declarations second, so `origin.type` is exact), graphics -> position, records -> json, dictionary heuristic; exporter writes reals with a decimal point, `+INF` / `-INF` / `NAN`, integer ids (mangle renumbers), key sanitising.                                                                                                                                                                                                                                                          |
| `formats/dot/`     | 8.4, 8.5   | A Graphviz-faithful tokenizer and parser (verified against `dot -Tcanon`), clusters as container nodes with the parent role, ports, HTML strings, scoped defaults expanded at creation, `pos` -> position; exporter writes nested cluster blocks from the children CSR.                                                                                                                                                                                                                                                                                                           |
| `formats/pajek/`   | 8.4, 8.5   | Line-oriented state machine over `LineReader`; `*Vertices N` pre-creates ids 1..N (0-based files detected), `*Arcs` / `*Edges` / `*Arcslist` / `*Edgeslist` / `*Matrix`, labels, positions, shapes, key-value parameters, time intervals -> spells; exporter numbers 1..N and writes sections in runs so a mixed file keeps its edge order.                                                                                                                                                                                                                                       |
| `formats/csv/`     | 8.4, 8.5   | The shared streaming RFC 4180 record reader (`records.ts`, also Neo4j's) with delimiter / newline sniffing, SNAP `#` / KONECT `%` comment headers, header resolution (`source` / `target`, Gephi `Source` / `Target` / `Type` / `Id` / `Label` / `Weight`), positional headerless files, a paired node table (`nodes`), per-row direction; exporter dialects `gephi` and `generic`.                                                                                                                                                                                               |
| `formats/json/`    | 8.2, 8.5   | Node-link, d3, JGF, Cytoscape, graphology and vis dialects sniffed from the document; structural columns (position, classes, parent, labels, relation, edge ids) declared on use; `meta.extra.json` shape record for exact re-export; exporter per dialect with its own capability table.                                                                                                                                                                                                                                                                                         |
| `formats/neo4j/`   | 8.4, 8.5   | The shared streaming RFC 4180 reader (quoted empty versus absent fields, multi-line fields); `neo4j-admin import` headers with id spaces, labels, types, typed properties, arrays, temporal values with `.text` companions, `:IGNORE`, brace options; several sections per input; exporter writes nodes and relationships in id-space runs.                                                                                                                                                                                                                                       |

Distribution: `package.json` exports `.` plus `./gexf`, `./graphml`, `./gml`, `./dot`, `./pajek`,
`./csv`, `./json`, `./neo4j` (types first); `scripts/entries.js` names the entries once for both
`scripts/build-bundle.js` (one multi-entry vite lib build: `dist/graph-io.js`, `dist/<format>.js`,
shared code under `dist/chunks/` so every entry sees one `ImportError` class and one importer
object; `@graphty/graph-format` external; no other runtime dependency) and
`scripts/bundle-types.js` (`dist/<entry>.d.ts` = one-line re-export of the tsc barrel);
`test/build-output.test.ts` checks the exports, the entries file and the built files agree and
that the subpath bundles share module instances with the root bundle.

Docs: `graph-io/README.md` (install, quick start for `importGraph` and a subpath import, the common
options, the capability matrix and known losses per format, format detection, the report shape,
the plugin toolkit), `graph-io/CLAUDE.md` (structure, commands, principles, the adding-a-format
recipe).

## Verification (graph-io, run 2026-09-14 from packages/graph-io by the audit round 1 verification pass)

| Command                                                   | Result                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm run lint` (eslint + tsc --noEmit)                   | pass, 0 findings                                                                                                                                                                                                                                                                                                                                                                                                      |
| `pnpm run typecheck`                                      | pass                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json` | pass (test/types/package-shape.test-d.ts and subpaths.test-d.ts against dist/graph-io.d.ts and dist/<format>.d.ts under noUncheckedIndexedAccess + exactOptionalPropertyTypes)                                                                                                                                                                                                                                        |
| `pnpm exec vitest run`                                    | 75 files, 4271 tests: 4237 pass, 34 skipped (the `IO_BENCH=1` absolute checks and the streaming benchmarks), 0 failing; `IO_BENCH=1 pnpm exec vitest run test/audit/fuzz-chunk-quadratic.test.ts test/audit/streaming-*.test.ts` runs the 40 gated checks (all pass, about 60 s, fixtures under `../../tmp/io-bench`; the sampled-peak bounds allow the 32 MiB young-generation allowance, see the verification pass) |
| `pnpm run coverage`                                       | statements 97.16, branches 95.23, functions 98.89, lines 97.16 (thresholds 80 / 80 / 75 / 80)                                                                                                                                                                                                                                                                                                                         |
| `pnpm run build:all`                                      | pass; dist/graph-io.js + 8 subpath bundles + chunks + 9 d.ts shims; test/build-output.test.ts passes including the bundle identity check                                                                                                                                                                                                                                                                              |
| `pnpm exec knip` (from packages/, both workspaces)        | pass, nothing reported (fast-xml-parser is a devDependency of the independent-reader probes only; papaparse is gone)                                                                                                                                                                                                                                                                                                  |
| `pnpm exec prettier --check .` (from packages/)           | pass for every file prettier can parse; the two deliberately malformed JSON corpus fixtures (`test/corpus/malformed/json/invalid-json.json`, `not-json.json`) make prettier's JSON parser error exactly as the same files do in the monorepo's graphty-element copy (`.prettierignore` is a verbatim copy and is not edited here)                                                                                     |
| Bundle smoke (`graph-io/tmp/bundle-smoke.mjs`, Node 22)   | the root bundle and the subpath bundles load through the package's own exports map, import a CSV, export GEXF, share one importer object                                                                                                                                                                                                                                                                              |
| README code blocks (`tmp/readme-run/`, Node 22)           | every code block of graph-format/README.md and graph-io/README.md extracted and executed against the built dist through the packages' exports maps (the two `npm install` blocks are not runnable before publication; the ImportReport shape block is type-checked against dist/graph-io.d.ts); all pass                                                                                                              |

Test counts by directory: root 62 (5 files: build-output, children, index, registry, sniff),
common 342 (17), formats/csv 134 (2), dot 111 (3), gexf 105 (2), gml 164 (3), graphml 109 (3),
json 159 (4), neo4j 207 (6), pajek 114 (3), audit 2764 (27 files: the seven auditors' suites,
kept as regression tests; 34 skipped without `IO_BENCH=1`). Type-level tests (`test/types/*.test-d.ts`, 2 files) are
compile-only and run under both tsconfigs. Source: ~36k lines under src/, ~33k under test/ plus
the corpus (7 legacy format directories copied from graphty-element, 102 files, of which one was
replaced by the verification pass (`malformed/csv/binary-content.csv`, see the known gaps), and 6
Neo4j fixtures + 13 malformed cases written for this package).

## graph-io: integration changes applied to the agents' typeChangesNeeded

- `src/common/ids.ts`: `coerceId(value, "string")` now applies `String()` to numbers (design 4.1,
  parity with `fromRecords`); the GML and JSON importers' local workarounds stay valid. The
  `W_ID_MERGED` code is defined once (`ID_MERGED_CODE`) and aliased by every format's code table.
- `src/common/format.ts`: `formatGmlReal(value, dtype?)` writes `+INF` / `-INF` / `NAN` (NetworkX's
  spellings) and takes the column dtype for the f32 shortest text; the GML exporter's `gmlRealText`
  delegates to it.
- `src/common/export.ts`: `checkCapabilities` no longer reports `W_DTYPE_UNSUPPORTED` for the u8
  `open` role column when the format writes open intervals, and no longer reports
  `W_EDGE_IDS_GENERATED` for a snapshot with no edges (the GEXF exporter's post-filters were
  removed; the JSON exporter tests were adjusted).
- `src/common/options.ts`: the shared `reportSinkOptions(sink, options, report, enforcesMissingNodes)`
  with one code `W_SINK_OPTION` (category `coercion`, element = option name) replaces the seven
  private variants (CSV, Pajek, JSON, GEXF, GraphML, DOT) and is added to GML and Neo4j, which had
  none. Semantics unified to "explicit requests only": an option left undefined is a default, never
  a request (GEXF and GraphML previously reported resolved defaults against the sink); GEXF and
  GraphML pass `enforcesMissingNodes: true` because they refuse unknown endpoints themselves. DOT's
  `W_OPTION_NOT_HONOURED` and GraphML's sink-mismatch use of `W_OPTION_IGNORED` are gone (both
  keep `W_OPTION_IGNORED` for `nodeIdFrom`).
- `src/common/escape.ts`: `isWritableDotText()` names the one text DOT cannot quote (a trailing
  backslash); the DOT exporter uses it.
- `src/formats/json/dialect.ts`: the dialect rule is the pure `sniffJsonDialect(root)`; the
  importer's `detectDialect` wraps it with its issue codes and the registry's sniff reuses it.
- `src/formats/gexf/exporter.ts`: the `positionWritesZ` type error (`Column.data` on the union) was
  fixed by narrowing to the numeric variants.
- `src/formats/neo4j/importer.ts`: claims `.tsv` and `text/tab-separated-values` as well.
- `src/formats/graphml/exporter.ts` and `src/formats/dot/exporter.ts` build their containment order
  from `childrenCsr()` instead of private CSR / map code.
- Per-format index files export a frozen code table where one was missing (`CSV_ISSUE`,
  `GEXF_ISSUE`, `GML_ISSUE` / `GML_LOSS`, `GRAPHML_ISSUE` (the constants' `ISSUE`), `NEO4J_ISSUE` /
  `NEO4J_LOSS`); the agents' existing table names (`DOT_IMPORT_CODES`, `DOT_EXPORT_CODES`,
  `PAJEK_ISSUES`, `PAJEK_LOSS`, `JSON_ISSUE`, `JSON_LOSS`, `GEXF_LOSS`, `CSV_LOSS`, `GRAPHML_LOSS`)
  were kept as they are.
- `test/helpers/corpus.ts`: `neo4j` added to `CORPUS_FORMATS` (a node-only Neo4j fixture may declare
  0 expected edges).
- Applied by audit round 1 (below): the one streaming XML tokenizer in `src/common/xml.ts` for
  GEXF and GraphML (fast-xml-parser gone from the runtime), the one RFC 4180 record reader for CSV
  and Neo4j (papaparse gone), and the core-side widening API (`GraphSink.widenNodeColumn` /
  `widenEdgeColumn`) the DOT / CSV / Pajek text importers use through `TextCellWriter`.

## graph-io: deviations from the design (collected from every agent)

Foundation (src/common):

- tsconfig.json maps `@graphty/graph-format` to `../graph-format/src/index.ts` and, because
  `composite: true` refuses files outside the project, also includes `../graph-format/src/**/*.ts`;
  tsconfig.build.json overrides `include` / `paths` so the emitted build resolves the built dist;
  tsconfig.strict-consumer.json maps the format to its dist d.ts and `@graphty/graph-io/*` to
  `dist/*.d.ts`. vitest resolves the format through the workspace symlink to its dist, so
  graph-format must be built before graph-io tests run.
- The legacy corpus runners (corpus.test.ts, malformed.test.ts) were not copied: they import
  graphty-element's DataSource classes; their role is taken by test/helpers/corpus.ts plus the
  per-format suites.
- Error limit: the error that takes `errorCount` above `errorLimit` is still recorded, `truncated`
  is set and `ImportError` is thrown; `errorLimit: 0` aborts on the first error. Warnings never
  count. `report.fail()` (fatal parse errors) aborts regardless of the limit.
- Abort: an aborted signal makes the reader throw the signal's reason (AbortError DOMException or
  the caller's Error), not an `ImportError`.
- Temporal: Neo4j `duration` has no epoch and is stored as a string column; localtime / time map
  to milliseconds since midnight; only ISO-8601 forms are parsed. A table holds one column per
  role, so a second temporal `.text` companion in the same table is declared without the
  `timeText` role (exporters find companions through `extra.for`).
- GEXF `<options>` on a string attribute make the column `dict` with the options as the
  dictionary; `W_OPTIONS_DROPPED` is reported for non-dict columns only.
- Unknown declared type texts (e.g. `yfiles.type`) map to a string column with `origin.type` kept
  and a `W_UNKNOWN_ATTR_TYPE` issue; unparsable defaults / options are dropped with `W_BAD_DEFAULT`
  / `W_BAD_OPTIONS`.
- Precision (5.1): `losesPrecision(spec, text)` decides from the integer text via BigInt; only
  `long` / Neo4j `integer` columns stored as f64 are checked.
- `onMixedDirection` `"directed"` / `"undirected"` force the sink's direction at header time and
  push every edge once; both the header difference and the first forced edge are reported
  (`W_DIRECTION_FORCED`). Mutual edges expand as a pair with `graphty.directed = true` on both
  halves and `graphty.mutual = true` on the primary; an undirected self-loop is not mirrored.
- `sanitizeIds`: `dense-1-based` always renumbers 1..N and never throws (`W_ID_RENUMBERED`; under
  `"mangle"` the Pajek exporter also writes every renumbered vertex's original id as a
  `graphty_originalId` parameter its importer restores under `restoreMangledIds`, the 8.5 contract);
  `integer` mangling assigns the smallest unused non-negative integers; `nmtoken` mangling replaces
  non-NameChars with `_` and suffixes `_2`, `_3` on collision; under `"error"` the code is
  `E_INVALID_ID` with `details.reason: "charset"`.
- LossNote codes beyond the two the design names are defined in `LOSS` (mixed direction,
  multi-edges, self-loops, edge ids, id mangling / charset / renumbering, dtype, components, lists,
  json, defaults, options, hierarchy, temporal, spells, dynamic values, graph attributes, positions,
  viz, extension tables); open-interval support is passed to `checkCapabilities` as an extra because
  `ExportCapabilities` (12.4) has no field for it.
- DOT quoting follows the Graphviz grammar literally (only `"` is escaped); Pajek labels holding a
  double quote or a line break cannot be written (`E_UNSUPPORTED`); GML strings escape `"`, `&` and
  every non-printable-ASCII character as `&#NN;` (NetworkX convention).

GEXF:

- Parsed in one pass by the shared streaming XML tokenizer (`common/xml.ts`, also GraphML's):
  well-formedness, entities and line numbers come from the tokenizer; fast-xml-parser is no longer
  a runtime dependency (a devDependency of the independent-reader probes only).
- Declared `string` attributes are declared as `string` immediately (dict only through
  `<options>`); the 5.4 dictionary heuristic is not applied to GEXF declared strings.
- XML- and viz-derived column names are reserved: a declared attribute with such a title is always
  renamed `<title>#<id>` (`W_COLUMN_RENAMED`), whether or not the file uses the XML field.
- `viz:color` is an f32 x4 rgba column in 0..1, `viz:position` an f32 x3 column with `extra`
  `{ units: "file", sourceDims }`, `viz:size` / `viz:thickness` f32 scalars, `viz:shape` a dict.
- Graph-level `start` / `end` / `timestamp` are kept as raw text in `meta.extra.gexf`.
- The edge attribute titled `weightFrom` is never a column: its static value overrides the XML
  weight attribute, its timed values go to `temporal:edge:weight`; under `weightFrom: null` the XML
  weight is ignored with one `W_WEIGHT_IGNORED`.
- Edge ids are kept as text in a string column (role id, unique). Missing element start / end are
  left unset; in the temporal tables missing bounds are stored explicitly as -Infinity / +Infinity.
- Timed values on an attribute declared in a static group are stored as dynamic anyway
  (`W_TIMED_VALUE_ON_STATIC`); `<attvalue for>` naming an undeclared attribute is a warning once
  per id. 1.2 `startopen` / `endopen` are XSD time values: on a node or edge they set the bound
  and the `open` bit (declaring both `start` and `startopen` is `W_GEXF_OPEN_BOUND_CONFLICT`); on a
  spell the bound is read and the openness dropped (`W_GEXF_SPELL_OPEN_DROPPED`).
- `restoreMangledIds`, `nodeIdFrom` and `hyperedges` have no use in GEXF and are reported
  (`W_OPTION_IGNORED`). `GEXF_1_2_CAPABILITIES` is exported for `version: "1.2"`. Node ids whose
  text reads back as the other type under the canonical rule (a non-integer number, a string of
  integer text) are reported (`W_ID_TEXT_TYPE`); non-f32 viz role columns are written but reported
  (`W_GEXF_VIZ_DTYPE`).
- Open: 1.3 `timestamps` / `intervals` syntax follows Gephi's toString form.

GraphML:

- Parsed by the shared streaming XML tokenizer (`common/xml.ts`; fast-xml-parser accepts
  mismatched tags silently, does not decode numeric character references without a deprecated
  option, and its builder is deprecated too).
- `hyperedges: "error"` aborts at once with `ImportError` (`E_HYPEREDGE`); `"skip"` warns once.
- `<port>` declarations are reported once and dropped; `sourceport` / `targetport` are kept as
  edge columns with the `sourcePort` / `targetPort` roles. A key named `label` gets the `label`
  role when free.
- `restoreMangledIds` also maps edge endpoints written with the mangled id. A second top-level
  `<graph>` is merged with `W_MULTIPLE_GRAPHS`. `meta.extra.graphml` records `graphId`,
  `edgedefault` and namespaces so a mixed file re-exports with the same layout.
- String cells holding characters XML 1.0 cannot carry are `E_XML_ILLEGAL_CHAR` in `check()` and
  refused by `export()`; `escapeXmlText` writes a carriage return as `&#13;` so it survives.
- Open: yFiles graphics are kept only as json trees (no extraction of geometry / fill / labels);
  nested `<graph id>` values and node / edge `<desc>` texts are not preserved; `W_PRECISION` is
  recorded once per file.

GML:

- Two-walk import over one token list (typed arrays, 17 bytes per token): schema first, typed
  declarations with `origin.type` second. Nodes are pushed before edges.
- Non-integer node ids / endpoints are `E_GML_ID_TYPE` (element skipped); under `nodeIdFrom`
  `label` / `index` the integer ids are not kept as a column (`W_GML_ID_DROPPED`).
- `graphics [ x y z ]` becomes an f64 x3 position column (z = 0 when absent, renamed
  `position#graphics` on collision) and the remaining keys stay a `graphics` json column
  (`positions: false` keeps the record); the dictionary heuristic applies to string-only scalar
  columns (`dictionaries: false` disables it).
- Top-level keys other than `graph` / `Creator` / `Version` become graph columns with
  `extra.gmlTopLevel`; `Creator` -> `meta.creator`, `Version` -> `meta.sourceVersion`.
- Capabilities: no mixed direction (`E_MIXED_DIRECTION` by default), `multigraph 1` from
  `meta.declaredMultigraph` / `flags.multigraph`, integer ids (mangling renumbers and keeps
  `graphty_originalId`), `sanitizeKeys: "error" | "mangle"` for keys outside the GML grammar or
  named like structural keys; json values report `W_GML_RECORD_NUMBER_TYPE`, `W_GML_RECORD_BOOLEAN`,
  `W_GML_RECORD_NULL`, `W_GML_JSON_ARRAY`, `E_GML_NESTED_ARRAY`, graphics conflicts.
- Open: the position column's z = 0 re-exports as `z 0.0` (snapshot-exact, not byte-exact); the
  dictionary sample covers the first 1024 rows.

DOT:

- `invalid-keyword.gv` and `missing-arrow.gv` of the malformed corpus are valid DOT (verified with
  `dot -Tcanon`); the tests assert a successful import with the Graphviz-verified counts. Grammar
  violations are fatal (`E_DOT_SYNTAX`), as in Graphviz.
- Clusters are container nodes (design 5.10): `cluster.gv` imports 12 nodes (manifest 10 + 2
  clusters), `fdpclust.gv` 10 (manifest 7). A subgraph is a container when its name starts with
  `cluster` or it carries `cluster=true`.
- Edge endpoint ports are kept (`graphty.sourcePort` / `graphty.targetPort`); a port on a node
  statement is dropped with `W_DOT_NODE_PORT_DROPPED`. A contradicting edge operator is read with
  the operator's direction and resolved through the mixed-direction policy (`W_DOT_EDGE_OPERATOR`;
  `mismatchedEdgeOperator: "header" | "error"` for the other behaviours).
- `key` becomes the edge id role column; `strict` merges parallel edges last-write-wins. HTML
  strings keep their outer brackets. `pos` -> `pos` (f32 x3, role position); `!` sets a `pin`
  bool attribute. `viz: false`; non-cluster subgraph attributes are reported.
- Untyped attribute cells go through `TextCellWriter` (the 5.1 grammar per column, backed by the
  sink's `widenNodeColumn` / `widenEdgeColumn`): an all-`2.0` column stays f64 and the lexical
  forms of a column that ends up string (`1e5`, `-0`, `1.0`) are kept; a sink without the widening
  members keeps the value-inferred dtype and `W_WIDENING_UNSUPPORTED` says so. Subgraphs nested
  deeper than 1024 levels are `E_DOT_NESTING`; a quoted string follows Graphviz's backslash-pair
  rule (two backslashes are a pair, so a string ending in an escaped backslash is closed), which is
  why a text ending in a backslash cannot be written (`E_DOT_TRAILING_BACKSLASH`).

Pajek:

- `missing-edges-section.net` is legal Pajek (vertices only): 3 nodes, `W_PAJEK_NO_LINES`. The
  `*Vertices` count is normative (`E_PAJEK_VERTEX_COUNT` for a partial list); an endpoint outside
  1..N is `E_UNKNOWN_NODE` regardless of `addMissingNodes`.
- 0-based files (dolphins.net, football.net) are detected (`firstVertex: 0 | 1 | "auto"`) and
  reported once (`W_PAJEK_ZERO_BASED`); the exporter always numbers 1..N (`W_ID_RENUMBERED`), so
  their round trips re-import with `nodeIdFrom: "index"`.
- `weightFrom` defaults to the field name `value` (the third column); a `value` column is written
  as a `value <v>` parameter, never into the third slot. Untyped parameters go through
  `TextCellWriter` (an all-`2.0` column stays f64; `W_WIDENING_UNSUPPORTED` on a sink that cannot
  widen). Shape and relation are dict columns without roles (a `shape` column with a value outside
  the keywords is written as a parameter, `W_PAJEK_SHAPE_AS_PARAMETER`); time intervals map to the
  spells role; a mutual pair is written as one undirected edge (`W_MUTUAL_AS_UNDIRECTED`); a vertex
  line whose parameters force a label gets its id text there (`W_PAJEK_LABEL_GAINED`).
- The `*Vertices` count is validated through `sink.reserve()` before any vertex exists (fatal
  `E_PAJEK_VERTICES_COUNT`). A second `*Vertices` / `*Network` is `E_PAJEK_MULTIPLE_NETWORKS`
  (abort); `*Matrix` is read as arcs; `*Partition` / `*Vector` sections are unsupported errors
  skipped together with the `*Vertices` line each of them carries in a `.paj` project file.
- Open: unknown shape values and the `value` parameter are untested against Pajek itself.

CSV:

- Header-only files import as an empty graph with `W_CSV_NO_DATA_ROWS` (the exporter writes exactly
  that for an empty snapshot); a malformed closing quote is fatal like an unterminated one; a header
  with neither endpoints nor an id column is fatal (`E_CSV_NO_ENDPOINT_COLUMNS`), headerless files
  are positional. `wrong-delimiter.csv` is valid under delimiter sniffing.
- `defaultDirected` is true; the Gephi `Type` column gives per-row direction with the first row as
  the header direction. The dictionary heuristic runs per attribute column (1024-value sample), so
  column declaration order follows first-write order. A repeated edge id is a per-row
  `E_DUPLICATE_EDGE_ID` issue (the row is skipped). A quoted empty cell is a set empty string (an
  empty-string id round-trips), a bare one is unset; text after a closing quote is fatal
  (`E_CSV_QUOTE`). Leading `#` (SNAP) and `%` (KONECT) comment lines are skipped and read for the
  direction they declare.
- Mutual pairs export as two Directed rows (`W_MUTUAL_EXPANDED`); check() reports storage-class
  changes (`W_STORAGE_CLASS_CHANGED`), text that reads back typed (`W_TEXT_INFERRED`), non-finite
  numbers, id type changes (`W_ID_TEXT_TYPE`), id text collisions (`E_ID_TEXT_COLLISION`), and the
  isolated nodes and the node order an edge table cannot carry (`W_CSV_ISOLATED_NODES`,
  `W_CSV_NODE_ORDER`). Lists / json / multi-component values are written as `;`-joined or JSON
  text. Gephi `timeset` is not implemented.
- The record reader (`csv/records.ts`) is the package's one RFC 4180 reader, shared with Neo4j;
  papaparse is gone.

JSON:

- The importer does not call `fromRecords()` (it creates and freezes its own builder); the
  node-link path reproduces its rules. A missing `nodes` / `edges` array is a recoverable
  `E_MISSING_SECTION`; empty input, invalid JSON, an unrecognised shape and a wrong section type
  are fatal.
- Exporter dialect defaults to the one recorded under `meta.extra.json` (node-link otherwise); the
  static `capabilities` is the node-link table, `jsonCapabilities(dialect)` gives the others; the
  tables list what the untyped re-import restores (dtypes f64 / i32 / bool / string, no lists /
  strides / viz, positions for Cytoscape only, edge ids for JGF / Cytoscape / graphology / vis) and
  `check()` names every column that changes (`W_DTYPE_UNSUPPORTED`, `W_LIST_UNSUPPORTED`,
  `W_COMPONENTS_FLATTENED`, `W_INTEGRAL_F64_AS_I32`, `W_ROLE_DROPPED`, `W_POSITIONS_DROPPED`,
  `W_EDGE_IDS_DROPPED`, `W_WEIGHT_KEY_CLASH`, `W_EMPTY_COLUMN_DROPPED`). A d3 document is written
  back bare (no `directed` / `multigraph` / `graph` keys) and a graphology one with only the
  options it declared. JGF node order follows JSON's integer-key ordering (`W_NODE_ORDER`). Mutual
  pairs are written as two directed edges everywhere (`W_MUTUAL_EXPANDED`); non-finite numbers as
  null (`W_NONFINITE_AS_NULL`).
- Open: `GraphBuilder.setMeta()` replaces `extra` as a whole, so the importer's `meta.extra.json`
  overwrites extras a caller set on the sink; JSON declares no types, so f32 / dict / list dtypes
  from another format come back f64 / string / json; vis `arrows` is a plain attribute.

Neo4j:

- The shared streaming RFC 4180 reader (`csv/records.ts`) tells a quoted empty field (empty
  string property; a quoted empty `:ID` is the id `""`) from an unquoted one (property not set)
  and handles multi-line quoted fields.
- Reserved columns have plain names with roles: `:LABEL` -> `labels` (list of dict, role labels),
  `:TYPE` -> `type` (dict, role kind), `:ID(Space)` -> `idSpace` (dict, role idSpace); a colliding
  user property is renamed `<name>#<name>`. A stored id `name:ID(Space)` becomes a string column
  with `origin.namespace`. A node id declared in two id spaces is `E_ID_SPACE_COLLISION`; a repeated
  id in one space is a `merged` warning with last-write-wins.
- Files are directed (no direction header); the sink is set directed lazily before the first
  relationship row; `onMixedDirection: "undirected"` reads a file as undirected
  (`W_DIRECTION_FORCED`). `{label:X}` on `:ID` is applied; other brace options are reported
  (`W_NEO4J_HEADER_OPTION_IGNORED`); `:IGNORE` columns are counted in one loss note.
- Exporter: an undirected snapshot, and the folded pairs of a mixed one under `onMixedDirection`
  `"directed"` / `"undirected"`, are written with every edge as a relationship
  (`W_NEO4J_UNDIRECTED_AS_DIRECTED`); untyped headers are restored as written; u32 / u8 written as
  long / int; json as JSON text; dict columns read back as string and position / visual columns as
  plain properties (the capability table says `dict` no, `positions` / `viz` false); edge ids,
  hierarchy, lifetimes, spells, graph attributes and extension tables are skipped with notes;
  format-specific notes `W_ID_TEXT_TYPE`, `E_ID_TEXT_COLLISION`, `W_WEIGHT_KEY_CLASH`,
  `E_NEO4J_WEIGHT_COLUMN_TAKEN`, `E_NEO4J_ID_COLUMN_TAKEN`, `W_NEO4J_MULTIPLE_ID_PROPERTIES`,
  `W_NEO4J_DECLARED_TYPE_CHANGED`, `W_NEO4J_ARRAY_DELIMITER`. Nodes and relationships are written
  in id-space runs so the order round-trips.
- Temporal `.text` companions are declared lazily on the first value whose canonical form differs.
  No public corpus exists: the six fixtures and thirteen malformed cases were written for this
  package (`tmp/make-neo4j-corpus.py`); the manifest has two extra optional fields (`options`,
  `with`).
- Open: the Neo4j 5 brace-option set was not verified against a live `neo4j-admin`.

Integration (registry, sniffing, children, build):

- `importGraph()` returns `{ format, sniff, snapshot, report, freeze }` (the design lists
  `{ snapshot; report; freeze }`; `format` and the `SniffResult` were added so a caller sees what
  was chosen). Format-specific options are passed in the same object (`importGraph(input,
{ format: "csv", delimiter: ";" })`); the registry keeps `format`, `filename`, `mimeType`,
  `builder` and `freeze` for itself. `weightDtype` seeds the builder with `"f64"` by default (8.4).
- An input no importer recognises is an `ImportError` (`E_IMPORT`) whose report has format
  `"unknown"` and one parse-error issue `E_UNKNOWN_FORMAT`; an unregistered format name is
  `E_UNSUPPORTED` (the core's convention for an option outside its set).
- `exportGraph()` returns the exporter's `AsyncIterable<Uint8Array>`; `exportGraphToString()` and
  `checkExport()` are the string and pre-flight forms (the design names only `exportGraph`).
- Sniffing combines hints and content by a fixed formula (module comment of `src/sniff.ts`) rather
  than the legacy "extension first, content second"; content therefore beats a misleading
  extension and decides shared ones. The JSON dialect from a truncated head is a hint built from a
  key skeleton; the importer's detection on the parsed document is authoritative.
- The subpath bundles share chunks with the root bundle (one vite build) instead of being
  self-contained, so a consumer that loads two entries gets one module instance of every shared
  class; `remote-logger`'s precedent points its subpaths at the tsc output instead, so the monorepo
  build convention for multi-entry vite lib bundles is set here.

## graph-io: known gaps and open issues

- Core-side: `GraphBuilder.setMeta()` replaces `extra` as a whole; importers that record
  `meta.extra.<format>` (JSON, GEXF, GraphML, Pajek) overwrite extras a caller set on the sink.
- JSON attributes are inferred, not declared (research note 07 section 2.4): a JSON key whose value a
  caller-declared column of another dtype refuses is a per-element `E_COLUMN_TYPE` issue and the
  value is dropped; the 5.6 rename rule applies to declared attributes only. A snapshot's f32 / u32 /
  u8 / dict / list / vector columns come back f64 / i32 / string / json through JSON, which the
  capability tables and `check()` now say.
- The independent XML reader probes (`fidelity-independent-readers`) use `fast-xml-parser` as a
  devDependency; the package has no runtime dependency beyond the format.
- GraphML: yFiles graphics are kept as json trees only (no geometry / fill / label extraction);
  nested `<graph id>` values and element `<desc>` texts are not preserved; `W_PRECISION` is once
  per file.
- GEXF: 1.3 `timestamps` / `intervals` attribute syntax follows Gephi's toString form only; the
  file's `idtype` is not honoured by the importer (the canonical id rule of 4.1 applies; `ids:
"string"` keeps the texts and `check()` says which ids change type).
- Pajek: unknown shape values and the `value` parameter are untested against Pajek itself.
- Neo4j: the Neo4j 5 brace-option set was not verified against a live `neo4j-admin`; a
  relationship row's `element` string for issues is built per row (a minor allocation the
  streaming auditor flagged; left as is, the row text is needed by four issue sites). Per-element
  issue-location objects and eager element strings remain in the GEXF node path (a frame per
  `<node>`, kept because nested nodes and deferred parents hold it), DOT (a copy of the scope's
  edge defaults, an element string and a location per edge) and Pajek (a token array and an
  extras record per line) hot loops; measured at 2-3 us per edge, the same order as the other
  importers, so they are recorded rather than removed. The GEXF edge frame and the `<attvalue>`
  location are reused (one object each per import).
- The DOT, GML and JSON importers read the whole text (8.4 allows it); every importer checks the
  signal every 64 pushed elements and once more before `report.finish()` (the verification pass
  added the per-element check to JSON and the pre-finish check to DOT, GML and GEXF; pinned by
  `fuzz-abort`'s in-memory cases).
- The strict-consumer flags (`noUncheckedIndexedAccess`) are not applied to src by design (Q19);
  the public d.ts files pass the strict-consumer compile.
- Timing-based audit tests (`fuzz-chunk-quadratic`, `streaming-*`) compare ratios with a 20 ms
  noise floor; on a loaded host a GC pause can still make one of them fail once.
- `test/corpus/malformed/csv/binary-content.csv` is no longer the verbatim graphty-element copy:
  the legacy file was the 14-byte text `Source,Target` (no binary byte at all, so it could not
  exercise the invalid-UTF-8 path its name promises, the fuzz auditor's pinned corpus defect); the
  verification pass gave it an invalid UTF-8 sequence like the Neo4j corpus' `binary-content.csv`,
  and the CSV importer test expects the fatal `E_INVALID_UTF8` (owner decision below: keep, or
  restore the verbatim copy and drop the pin).
- Pajek `sanitizeIds: "mangle"` restores ids through a `graphty_originalId` parameter (8.5);
  a string id of canonical integer text still reads back as a number under `ids: "canonical"`
  (`W_ID_TEXT_TYPE`, `ids: "string"` keeps the texts), because Pajek tokens carry no type.

## graph-io audit round 1 (2026-09-14)

Seven adversarial audits (fidelity across formats, fuzz / malformed input / streaming equivalence,
streaming performance, API conformance, per-format semantics, code quality, the monorepo rehearsal)
reported about 80 findings, pinned by 347 deliberately failing tests under `test/audit/` (of 4228).
Every pinned test now passes by a code change, except the tests re-pinned to the design where an
auditor asked for a semantic change the design had settled (listed under "Rejected" with the
section); the corpus-fixture defect was resolved by the verification pass (the fixture replaced,
see the known gaps). The audit suites stay in `test/audit/` as regression tests (2764 tests).

Applied (by area):

- Streaming (linear time, bounded memory): one shared XML tokenizer (`common/xml.ts`, piece-buffered
  pending tokens, attribute values skipped with `indexOf`) replaces fast-xml-parser for GEXF and the
  GraphML-private tokenizer; one shared RFC 4180 record reader (`csv/records.ts`, a resumable
  per-character state machine with quoted runs skipped with `indexOf`, the sniff preview capped)
  replaces papaparse and the Neo4j-private reader; `LineReader` keeps pending pieces instead of
  re-concatenating; a 50 MB quoted cell, line or attribute value in 16 KB chunks costs about the
  same as in one piece (8 MB attribute: 11 ms chunked, 4 ms whole; the 50 MB / 10 s checks pass
  under `IO_BENCH=1`). The core's column name lookup is an index (`ColumnNameIndex` in
  graph-format's builder, 5k -> 10k columns 2.1x) and V8 sliced-string retention of ids and
  interned strings is cut by `detachString`. Every importer checks the cancellation signal every
  64 elements and once before `report.finish()`; `importGraph()`'s stream peek closes the source
  when the abort lands as the head completes or the importer throws before iterating.
- Fidelity notes (8.5: `check()` predicts every difference the format's own importer produces):
  `checkCapabilities` gained `CheckExtras` (`roles`, `roleNames`, `temporalText`, `positionDtype`,
  `openIntervals`) and reports the role of every column the format has no slot for
  (`W_ROLE_DROPPED`), a slot column read back under the importer's fixed name
  (`W_COLUMN_NAME_CHANGED`), a dropped `.text` companion (`W_TEMPORAL_TEXT_DROPPED`), dict options
  on dict columns, list items of dict dtype, a position column of the other float dtype; the
  exporters add `W_WEIGHT_KEY_CLASH` (a plain column read back as THE weight), `W_ROLE_ASSUMED`,
  `W_ID_TEXT_TYPE` / `E_ID_TEXT_COLLISION` (every text format), `W_STORAGE_CLASS_CHANGED`,
  `W_INTEGRAL_F64_AS_I32` (JSON), `W_TEXT_INFERRED`, `W_EMPTY_COLUMN_DROPPED` (DOT, GML, JSON),
  `W_OPTIONS_GAINED` (GEXF), `E_XML_ILLEGAL_CHAR` (GEXF, GraphML: `export()` refuses XML-illegal
  text), `W_MUTUAL_EXPANDED` / `W_MUTUAL_AS_UNDIRECTED` in every exporter, GraphML's reserved-name
  renames, Pajek's `W_PAJEK_LABEL_GAINED` and `W_PAJEK_SHAPE_AS_PARAMETER`, CSV's
  `W_CSV_ISOLATED_NODES` / `W_CSV_NODE_ORDER`. The JSON dialect tables, GraphML `json` and Neo4j
  `dict` / `positions` / `viz` are truthful (F1, F4, F5); every one of the 40 `LOSS` codes is
  reachable through a built-in exporter.
- Direction: under `onMixedDirection: "directed"` every exporter without mixed direction folds a
  pair to one directed edge (F2); Neo4j honours `"undirected"` (F3); GraphML and Pajek fold a mutual
  pair to one undirected edge, the others write two directed edges; the shared `pairFolding()` and
  `explicitWeights()` helpers replace eight private copies of the mirror and weight-text rules.
- Text cells (5.1): `TextCellWriter` tracks the text dtype per column, widens an all-`x.0` column
  to f64 through the new optional `GraphSink.widenNodeColumn` / `widenEdgeColumn` (implemented by
  `GraphBuilder`), keeps the lexical form of `1e5` / `-0` / `1.0` in a column that ends up string
  and commits its state only after the sink accepted the cell (F6, D19); CSV, DOT and Pajek write
  untyped cells through it; `formatDecimal` / `formatGmlReal` are the one decimal-point rule.
- Importer semantics: the 5.6 collision rule through `declareResolved` in every importer
  (`<name>#<id>` + `W_COLUMN_RENAMED`, and `W_ROLE_TAKEN` when the role is held; Pajek's structural
  columns included); `reportUnusedOptions` (`W_OPTION_IGNORED`, one category) in every importer
  (F7); duplicate edge ids are per-element `E_DUPLICATE_EDGE_ID` issues (GEXF, CSV, JSON) instead
  of a raw freeze error; duplicate node declarations are merged with `W_DUPLICATE_NODE` everywhere
  (JSON included); `counts.nodes` counts the endpoints an edge created; a quoted empty CSV / Neo4j
  cell is a set empty string (an empty-string id round-trips), text after a closing quote is a
  fatal `E_CSV_QUOTE`; SNAP `#` and KONECT `%` comment headers are skipped and read for their
  direction; an out-of-range d3 index link is `E_BAD_INDEX`; the GEXF XSD open bounds
  (`startopen` / `endopen`), `applyStaticWeight` on both halves of a pair, the Pajek `*Vertices`
  count validated by the sink's `reserve()` before any vertex is created and the `.paj`
  `*Partition` / `*Vector` sections skipped with their own `*Vertices` line; DOT nesting bounded
  (`E_DOT_NESTING`), the GML record parser iterative (10k-deep lists no longer overflow the
  stack), Graphviz's backslash-pair rule (a text with a backslash before a quote or at its end is
  `E_DOT_TRAILING_BACKSLASH`); `recordError` rethrows anything that is not a `GraphFormatError`.
- Issue codes (the quality auditor's scheme, applied to every format now): one code per concept,
  defined once in `common/codes.ts` and aliased by every table; format-specific codes carry the
  format prefix; every subpath exports `<FMT>_ISSUE` and `<FMT>_LOSS` with keys = code minus the
  prefixes, and every code an importer records is a member of its table. The renames (about 90
  codes, e.g. `E_CSV_MISSING_ID` -> `E_MISSING_ID`, `W_YFILES_JSON` -> `W_GRAPHML_YFILES_JSON`,
  `W_ID_READ_AS_TEXT` -> `W_ID_TEXT_TYPE`, `E_HEADER` -> `E_NEO4J_HEADER`, `PAJEK_ISSUES` ->
  `PAJEK_ISSUE`) are breaking changes of the subpath surfaces; nothing has shipped.
- Quality: no function over 150 lines (`plan`, `planExport`, `writeParts`, `collectAttributes`,
  `importCytoscape`, `checkColumns` and the CSV scanner split), no duplicated mirror / weight /
  decimal / entity / whitespace helpers, the root barrel exports the shared machinery for plugin
  authors (`TextCellWriter`, `pairFolding`, `explicitWeights`, `declareResolved`,
  `reportUnusedOptions`, `tokenizeXml`, the codes), the graph-io benchmark results file is a JSON
  array of sessions like graph-format's, `fast-xml-parser` is a declared devDependency of the
  independent-reader probes.

Rejected or re-pinned (the design wins; each test was re-pinned to the design behaviour with the
section named in the test):

- D4 (a plain `weight` / `value` column must survive a round trip as a column): 3.7 / 8.4 make the
  importer's `weightFrom` rule decide what becomes THE weight, so the column reads back as the
  weight; the fix is the prediction (`W_WEIGHT_KEY_CLASH`), not a rename of the column.
- D5 (GEXF must honour `idtype="string"`, DOT must keep string ids of integer text): 4.1's
  canonical rule applies to every text-cell format (Gephi writes `idtype="string"` on every file);
  `check()` predicts the change (`W_ID_TEXT_TYPE`) and `ids: "string"` keeps the texts.
- D8 second half (dict / list / f32 x4 / integral f64 must survive JSON): JSON declares no types
  (research note 07 section 2.4); the tables and the notes are truthful instead.
- D13 (GML must keep an f32 position): GML reals are doubles; `W_DTYPE_UNSUPPORTED` names the
  position column and the values are exact at f32 precision.
- D1 first half (a CSV edge table must keep an isolated node): an edge table cannot; the notes
  (`W_CSV_ISOLATED_NODES`, `W_CSV_NODE_ORDER`) say so and the node table keeps it.
- D3 (Pajek must keep the original id of a labelled node): the file has no second slot; the
  `W_ID_RENUMBERED` note now says exactly which ids are lost and which become labels (the
  auditor's own expectations contradicted the file text they pinned).
- GML NetworkX leniencies (string node ids, strings spanning lines): the malformed corpus fixtures
  (`invalid-value-type.gml`, `unclosed-string.gml`) and the semantics tests pin the strict reading.
- JSON duplicate node counting (`counts.nodes + skippedNodes` = 2): the shared convention of every
  other importer counts a merged declaration as neither; the test asserts that convention now.
- JSON label collision with a caller's typed column (5.6 rename): JSON keys are inferred
  attributes, not declarations; the per-element `E_COLUMN_TYPE` issue stands (known gap above).
- The fidelity auditor's D16 / D17 expected an escaped or literal write; `check()` names the
  cells (`E_XML_ILLEGAL_CHAR`) and `export()` refuses them, which the re-pinned test accepts as
  "refused".
- Neo4j's per-row `element` string (a streaming minor) is kept: four issue sites need it.

Test changes that were not simple additions or code renames: the `fidelity-defects` re-pins
above; `fidelity-matrix`'s EXPLAINS table rewritten for the new codes and taught the rename target
of a name-change note; `conformance-semantics`' reachability cases extended to all 40 `LOSS`
codes; `quality-declare-divergence` pins JSON's inferred-attribute outcome; `fuzz-mutations`'
duplicate-node count; `fuzz-chunk-quadratic`'s 20 ms noise floor on the three doubling checks and a
JIT warm-up for the column-scaling check; `fidelity-independent-readers`' quoted-empty-cell
expectation (the empty string now round-trips) and the XML illegal-character probe (refusal
accepted); the per-format unit tests adapted to the new codes, notes and semantics (GraphML label
slot and reserved-name renames, JSON fold-under-directed and bare d3, Pajek label / shape notes,
Neo4j undirected policy and quoted-empty ids, DOT backslash rule, GEXF id / options notes, CSV
comment lines).

### Verification pass (2026-09-14)

An independent verifier re-ran every command above, probed the resolved findings against the
built dist through the packages' exports maps (fidelity notes, duplicate ids, count hints, the
abort paths, the 40 `IO_BENCH=1` checks, 8 MB tokens in 16 KB chunks for every reader, DOT at
400k edges) and the README code blocks, and found four things to change:

- `test/helpers/model-graph.ts` (graph-format): the Map-of-Maps model dropped self-loops before it
  could refuse a duplicate edge, so a refused freeze (`duplicateEdges: "error"`) left the model
  changed while the builder was, correctly, untouched (design 11.1); one fast-check seed in about
  200 runs of `model.test.ts` failed on `droppedSelfLoops`. The model now restores its edge state
  when a freeze throws; the counterexample is a deterministic test.
- Cancellation: the JSON importer checked the signal once between its node and edge sections, not
  every 64 elements, and DOT, GML and GEXF had no check between their last periodic check and
  `report.finish()`, so an abort raised from the sink during the last elements of an in-memory
  document resolved instead of rejecting. Every importer now checks every 64 pushed elements and
  once before `finish()`; `fuzz-abort` pins both (an abort at the 20th and at the last `addEdge`).
- Pajek honours `sanitizeIds: "mangle"` (design 8.5: the exporter names the attribute the original
  id is kept in, `restoreMangledIds` reads it back): every renumbered vertex carries a
  `graphty_originalId` parameter, the importer restores it (`W_PAJEK_ORIGINAL_ID_MERGED` when two
  lines carry the same one; the index order stays the vertex-number order, so a parameter on a
  vertex an out-of-order later line already created under its number is
  `W_PAJEK_ORIGINAL_ID_UNRESTORED`), a user column of that name is reserved under "mangle"
  (`W_PAJEK_KEY_DROPPED`), an unwritable id text is `E_PAJEK_TEXT`, and a string id of integer
  text is predicted to come back as a number (`W_ID_TEXT_TYPE`). Before, the option was silently a
  no-op and a labelled node's id was written nowhere (the D3 finding's first alternative).
- `test/corpus/malformed/csv/binary-content.csv` now holds invalid UTF-8 (see the known gaps), so
  the suite has no test that fails by design; the graph-io README's issue-table names
  (`DOT_ISSUE`, `PAJEK_ISSUE`) and this file's numbers and stale GEXF / GraphML / CSV / Neo4j /
  DOT / Pajek / JSON bullets were brought in line with the code.
- The `IO_BENCH=1` suites were flaky in three runs out of six: the GEXF 200k sampled peak swung
  between 24 and 33 MiB against a bound of twice the 14.8 MiB input (young-generation garbage is
  bounded by V8's two 16 MiB semispaces, not by the input, so a late scavenge alone crosses that
  bound while 8.4 MiB stay retained), and the builder's 5k / 10k column ratio was measured on
  8-9 ms timings (2.0x-3.2x). The GEXF importer now reuses one edge frame and one `<attvalue>`
  location (the per-edge allocations of the hot-loop finding; peak 24-26 MiB in isolation), the
  sampled-peak bound of the memory suite is `max(2 x input, input + 32 MiB)` (the export suite
  already allowed the semispaces), and the column check compares 20k with 40k (38 / 76 ms, 2.0x;
  the pinned quadratic version took 3.3 s at 20k). Six further runs, sequential and parallel:
  40 / 40 each.

## Owner decisions needed (consolidated)

Core (from the graph-format audit round 1 and this pass):

1. Whether `AttributeTable.set()` on a snapshot should range-check same-space `refersTo` values
   and refuse a cross-space declaration (today `validate()` reports it, I12).
2. Whether the three weight flags (`allWeightsOne`, `nonNegativeWeights`, `finiteWeights`) should
   describe the exact staged weights when an f64 shadow is kept, instead of the f32 arc array (3.8).
3. The doc-silent choices to confirm before consumers depend on them: `weighted: false` semantics,
   the `graphty.weight` shadow column name, the carried-views policy under "full", the legacy
   last-wins rule of `toRecord` / `toStringMap`.
4. Whether `GraphBuilder.setMeta()` should merge `extra` instead of replacing it (importers
   recording `meta.extra.<format>` overwrite a caller's extras).
5. `runFreeze` and five other builder functions above the 13.4 length threshold: split, or accept
   as one dtype switch each.

graph-io (this pass):

6. `test/corpus/malformed/csv/binary-content.csv` was replaced (it now holds invalid UTF-8 and the
   CSV importer test expects the fatal `E_INVALID_UTF8`), the one departure from the verbatim
   graphty-element corpus; confirm, or restore the legacy 14-byte file and drop the fuzz auditor's
   pin (the legacy `malformed.test.ts` of graphty-element expected the file to import).
7. The issue-code renames (about 90 codes and the `PAJEK_ISSUES` -> `PAJEK_ISSUE` table) change
   every subpath's public surface; confirm before anything ships.
8. `W_ID_RENUMBERED` (Pajek, under the default `sanitizeIds: "error"`: a labelled node's id is
   written nowhere; `"mangle"` keeps it in `graphty_originalId`) and the JSON / Neo4j / GEXF /
   GraphML / GML weight-key clash are predicted, not avoided: confirm that a loss note is the
   design's answer (8.5) rather than a rename of the written column, and whether Pajek should
   default to writing the original ids (the `dense-1-based` charset never throws under `"error"`,
   a recorded deviation from 8.5's "never silently renames").
9. JSON keys colliding with a caller-declared column of another dtype are per-element
   `E_COLUMN_TYPE` issues (no 5.6 rename for inferred attributes); confirm.
10. GEXF's `idtype` is not honoured on import (4.1 canonical rule; `ids: "string"` opts out);
    confirm that the importer should not switch on the file's declaration.
11. The `E_EMPTY_ID` loss code was dropped (an empty-string id now round-trips through CSV and
    Neo4j as the quoted empty cell); confirm the quoted-empty-cell rule for the CSV exporter
    against other readers (RFC 4180 readers give the empty string; Gephi's importer may treat it
    as blank).
