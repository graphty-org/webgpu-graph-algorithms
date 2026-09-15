# CLAUDE.md

This file provides guidance to Claude Code when working with the @graphty/graph-format package.

## Project Overview

@graphty/graph-format is a zero-runtime-dependency, framework-agnostic graph DATA FORMAT: a frozen
compressed-sparse-row (CSR) representation over typed arrays, a mutable builder that produces frozen
snapshots, columnar typed node/edge attribute storage, and an id-to-index map kept outside the CSR.
It contains no graph algorithms, no rendering and no parsers (parsers live in @graphty/graph-io).

The normative design is `design/graph-format/graph-format-design.md` in the monorepo:

- section 3.2: the numbered invariants I1-I18 (the contract every consumer relies on)
- section 12.2: the complete public type surface (everything `src/index.ts` exports, nothing else)
- section 13: package layout, build, lint and versioning rules
- section 17: the decision log

Implement the design as written. Where it is silent, choose the simplest option consistent with the
invariants and say so in the PR. `packages/STATUS.md` (staging) lists every deviation made so far.

## Package Structure

```
graph-format/
+-- package.json                  # @graphty/graph-format, ESM only, sideEffects false, zero dependencies
+-- project.json                  # Nx project "graph-format"
+-- graph-format.ts               # root entry, one line (export * from "./src/index.js"); tsc alone emits dist/graph-format.{js,d.ts}
+-- tsconfig.json                 # lint/typecheck config: graph-format.ts src/ test/ benchmarks/, noEmit
+-- tsconfig.build.json           # emit config: graph-format.ts + src/, rootDir ".", outDir dist (-> dist/src/ + dist/graph-format.*)
+-- tsconfig.strict-consumer.json # public d.ts under noUncheckedIndexedAccess + exactOptionalPropertyTypes
+-- vitest.config.ts              # single project, environment node, pool forks, thresholds 80/80/75/80
+-- scripts/build-bundle.js       # programmatic vite lib build -> dist/graph-format.js (+ map)
+-- scripts/bundle-types.js       # writes dist/graph-format.d.ts = export * from "./src/index.js"
+-- src/
|   +-- index.ts                  # the ONLY public barrel; every 12.2 name listed explicitly (no star re-exports)
|   +-- constants.ts              # INVALID_INDEX, MAX_COUNT, FORMAT_VERSION, SNAPSHOT_BRAND (public) + ALIGNMENT, WIRE_* (internal)
|   +-- errors.ts                 # GraphFormatError and its codes
|   +-- types/                    # columns.ts snapshot.ts builder.ts wire.ts (12.2 types), internal.ts (construction contracts)
|   +-- ids/                      # node-id-map.ts (five kinds), string-store.ts (Utf8 store, lazy decode), edge-id-index.ts
|   +-- columns/                  # column.ts table.ts bitmap.ts dictionary.ts infer.ts growable.ts remap.ts
|   +-- builder/                  # graph-builder.ts freeze.ts (6.3 pipeline) compact.ts counting-sort.ts arena.ts
|   +-- snapshot/                 # graph-snapshot.ts views.ts queries.ts derived.ts validate.ts hash.ts
|   +-- populate/                 # from-edge-arrays.ts from-csr.ts from-records.ts
|   +-- wire/                     # to-wire.ts from-wire.ts bytes.ts (GSNP container)
|   +-- util/                     # typed-array.ts mask.ts shared-buffers.ts (the 9.1 owner count)
|   +-- lib-resizable-array-buffer.d.ts   # local declaration for resizable / resize / maxByteLength
+-- test/
|   +-- helpers/                  # parts.ts (naive CSR + SnapshotParts), invariants.ts (I1-I18 checks), model-graph.ts, random-ops.ts
|   +-- fixtures/                 # golden .gsnp containers (regenerate with tmp/make-golden.ts when the layout changes deliberately)
|   +-- <module>/*.test.ts        # unit tests, one directory per src module
|   +-- types/*.test-d.ts         # expectTypeOf tests; also compiled by tsconfig.strict-consumer.json
|   +-- invariants.test.ts        # I1-I18 and P1-P12 over fixtures and fast-check graphs
|   +-- index.test.ts             # the barrel exports exactly the 12.2 value surface
|   +-- build-output.test.ts      # package.json shape, dist checks, WIRE_PRODUCER pinned to the version
+-- benchmarks/                   # run.ts harness.ts datasets.ts freeze.bench.ts ids.bench.ts views.bench.ts results/
```

### How the four public classes relate to the type files

`GraphSnapshot`, `NodeIdMap`, `AttributeTable` and `GraphBuilder` are CLASSES (design 12.1/12.2).
Each class `implements` an interface named `<Name>Contract` in `src/types/*` that transcribes the
12.2 member set with the JSDoc; the public type name is the class itself, which `src/types/*`
re-export type-only (`export type { GraphSnapshot } from "../snapshot/graph-snapshot.js"`), so
`DerivedGraph.snapshot`, `SnapshotParts.ids`, `GraphSnapshot.nodes` and every other reference names
the class. The imports are type-only, so there is no runtime cycle. Keep the contract interfaces
and the classes in sync: tsc fails when a class drops a member.

### The snapshot <-> wire import cycle

`snapshot/graph-snapshot.ts` imports `toWire` / `toBytes` / `toByteChunks` / `transferables` from
the wire module, and the wire module imports `createSnapshot` / `peekView` / `seedView` back. This is
safe because each side only reaches the other through hoisted function declarations invoked at call
time; never add a top-level statement in `src/wire/*` that touches a `const`, `let` or `class`
binding of `graph-snapshot.ts` (it would be in its temporal dead zone when the wire module happens
to evaluate first). `tmp/cycle-smoke.mjs`-style checks load `dist/src` in both orders.

### Storage sharing (design 9.1)

`src/util/shared-buffers.ts` keeps the owner count. Every `AttributeTable` claims its columns and
every `GraphSnapshot` claims its core buffers, id map and tables at construction; the second claim
of the same object marks it shared, and `toWire({ transfer: true })` / `transferables()` then copy
instead of transferring. `transpose()` marks the adopted reverse arrays shared explicitly. When you
add a new way to alias storage between two holders, either construct through those paths or call
`noteShared()` yourself.

## Essential Commands

```bash
# Build
npm run build            # tsc -p tsconfig.build.json -> dist/src/ plus the unbundled dist/graph-format.{js,d.ts} entry
npm run build:bundle     # vite lib bundle overwrites dist/graph-format.js, then rewrites dist/graph-format.d.ts
npm run build:all        # both, in order

# Testing
npm test                 # vitest in watch mode
npm run test:run         # run all tests once (~860 tests, ~2 s)
npm run coverage         # run with v8 coverage (thresholds 80/80/75/80)
npm run coverage:preview # serve coverage report on port 9056

# Linting and types
npm run lint             # eslint (root flat config) + tsc --noEmit over src, test and benchmarks
npm run typecheck        # tsc --noEmit only
npm run typecheck:strict-consumer  # compile test/types/*.test-d.ts against dist/graph-format.d.ts
                                   # with noUncheckedIndexedAccess and exactOptionalPropertyTypes on
                                   # (needs npm run build:all first; internal.test-d.ts is excluded there
                                   # because it imports src/types/internal.ts, which is not public)
pnpm exec knip --workspace graph-format   # from packages/: dead exports (un-export or tag @public with a reason)

# Benchmarks (design section 15.5)
npm run benchmark                          # every group, appends benchmarks/results/<host>-node<version>.json
npx tsx benchmarks/run.ts freeze --no-save # one group, no results file
node --expose-gc --import tsx benchmarks/run.ts   # exact memory deltas

# Everything before a commit
npm run ready:commit
```

## Key Design Principles

- The invariants I1-I18 (design section 3.2), `INVALID_INDEX`, the counting vocabulary and the view
  tables are public API. Changing any of them is a breaking change even when no TypeScript signature
  changes (design section 13.5).
- Zero runtime dependencies. No DOM lib in the core: `tsconfig.json` sets `lib: ["ES2020"]`; anything
  needing `ReadableStream` or `AbortSignal` belongs in @graphty/graph-io.
- `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` are OFF for `src/` (every consumer
  compiles that way). The public declarations are still verified under both flags by
  `tsconfig.strict-consumer.json`, which compiles `test/types/*.test-d.ts` against the BUILT
  `dist/graph-format.d.ts`. Type tests therefore import the package by its bare name
  (`import { ... } from "@graphty/graph-format"`); `tsconfig.json` maps that name to `src/index.ts`
  for lint, typecheck and vitest, and the strict-consumer config maps it to `dist/graph-format.d.ts`.
- Resizable `ArrayBuffer` is feature-detected at runtime; the only type declarations for it are in
  `src/lib-resizable-array-buffer.d.ts`.
- Never add a fallback that hides a missing capability.
- Absent output is `null`, never `undefined`; optional input is `?: T | undefined` (design 12.1).
- Every index-taking method is total for in-range arguments and unchecked otherwise; every throwing
  call leaves the builder or snapshot unchanged (design 11.1).

## Invariants (design 3.2, one line each)

- I1 `rowPtr` has `nodeCount + 1` non-decreasing entries from 0 to `arcCount === colIdx.length`.
- I2 `colIdx[a] < nodeCount`; `INVALID_INDEX` never appears in a core or view array.
- I3 counts `<= MAX_COUNT`; `edgeCount <= arcCount`; never apply bitwise operators to arc indices.
- I4 rows sorted by `colIdx`, ties by ascending `arcToEdge`; no unsorted mode.
- I5 `arcToEdge` / `edgeToArc` are consistent permutations; `edgeToArc[e]` is the declared orientation.
- I6 directed: `arcCount === edgeCount`, `arcToEdge` a permutation.
- I7 undirected: doubled storage with equal-weight mates, one arc per self-loop.
- I8 `weights` null or `arcCount` long, never NaN.
- I9 flags are truthful predicates over the arrays.
- I10 4-byte alignment everywhere; arena segments at multiples of 256.
- I11 `ids` is a SameValueZero bijection over `[0, nodeCount)`.
- I12 table row counts match; column length rules; codes and `refersTo` in range.
- I13 edge columns are indexed by logical edge, never by arc.
- I14 indices follow first appearance / `addEdge` order; nothing is sorted by id or degree.
- I15 determinism across runs and engines (stable counting sorts, insertion-order Maps).
- I16 prefix stability between freezes without removals or merges; `freezeWithReport` says otherwise.
- I17 nothing frozen ever changes; views memoised once and shared; only the column set / mutable contents move.
- I18 no snapshot array aliases builder staging.

`test/helpers/invariants.ts` has every one of them as an executable check (`assertInvariants`,
`assertMatchesSpec`); run it over any new construction path.

## House Style (design section 13.4)

- 4-space indent, double quotes, Prettier formats everything (`.prettierrc` at the workspace root).
- JSDoc on every exported function, class and method: `@param name - description`, `@returns ...`.
- Explicit return types; `import { type X, y }` inline type qualifiers (never a separate
  `import type` next to a value import from the same module); relative imports end in `.js`.
- camelCase field names (`rowPtr`, `colIdx`, never `row_ptr`); `curly` everywhere; `default-case`
  in every dtype `switch`; no `enum`, no `namespace`, no static-only classes, no default exports.
- No exported type named `Node`, `Edge` or `Graph`.
- No `console.log` in `src/`; no `eslint-disable`, no `@ts-expect-error` (negative type tests use
  `expectTypeOf(...).not...`).
- Plain ASCII in every file (the Write tool unescapes `\uXXXX`; re-escape and check with `grep -P '[^\x00-\x7F]'`).
- An export used only inside its own file is dead to knip: un-export it (same-file types in exported
  signatures are still emitted into the d.ts) or tag it `@public` with the reason.

## Testing Guidelines

- Tests live in `test/<module>/` mirroring `src/<module>/`, named `*.test.ts`, `expect` style.
- Property tests use fast-check; the Map-of-Maps reference model lives in `test/helpers/model-graph.ts`
  and the builder scenario arbitraries in `test/helpers/random-ops.ts` (`FC_RUNS=4000` for a soak).
- Every error code the module can throw has a test that triggers it.
- `test/wire/golden.test.ts` compares `test/fixtures/rich-v1.gsnp` byte for byte (modulo `producer`);
  a deliberate layout change regenerates it with `tmp/make-golden.ts`.
- `test/build-output.test.ts` checks the package.json shape and, when `dist/` exists, the build output.

## Adding a View

1. Add the name to `ViewName` in `src/types/snapshot.ts` and a slot to `ViewValues` in
   `src/types/internal.ts` (`ViewCache` derives from it; `test/types/internal.test-d.ts` pins the key set).
2. Implement the pure computation in `src/snapshot/views.ts` (a function of the core arrays; typed
   arrays over plain `ArrayBuffer`, 4-byte aligned unless it is an f64 view) and extend `viewArrays()`
   (checksums, byte accounting, wire) and `viewByteLength()`.
3. Add the method to `GraphSnapshotContract` (`src/types/snapshot.ts`) and the class
   (`this.cached(name, () => compute(this))`), and a case in `materialiseView()` (the `prepare()`
   dispatch) and in `VIEW_NAMES`.
4. Wire: `encodeViews()` in `src/wire/to-wire.ts` carries whatever `viewArrays()` returns; add the
   receiver case to `installViews()` in `src/wire/from-wire.ts` (class and length checks, then
   `seedView`). Scalar views go in `SCALAR_VIEWS`.
5. Tests: `test/snapshot/views.test.ts` (against a naive computation and the fast-check graphs),
   `test/wire/round-trip.test.ts` (carried and installed), and the README view table. Adding a view
   is a minor version bump.

## Adding a Dtype

1. Add it to `Dtype` in `src/types/columns.ts`, a `<Name>Column` interface extending `ColumnBase`,
   the `Column` union and `DtypeValue`.
2. `src/columns/column.ts`: a column implementation class, `resolveColumnMeta()` rules, `createColumn()`,
   `partsOf()`, `columnOfValues()`, `emptyColumnOf()`, `clone`, `byteLength`, `gpuEligibility()`
   and `gpuViewOf()`; every `switch (dtype)` in `src/columns/*.ts`, `src/snapshot/hash.ts`,
   `src/snapshot/derived.ts` (reducers) and `src/util/typed-array.ts` has a `default-case` that
   will now fail to compile until the new case is added (that is the point).
3. Inference (`src/columns/infer.ts`) and builder staging (`src/builder/compact.ts` StagingColumn)
   if the dtype can be inferred or written through `setNodeValue`.
4. Wire: `encodeColumn()` in `src/wire/to-wire.ts` and the decoder in `src/wire/from-wire.ts`,
   including the `unknownColumns: "skip"` path so OLDER readers skip it (bump the wire minor).
5. Tests in `test/columns/`, a row in the README dtype list, `test/types/columns.test-d.ts`.
   Adding a dtype is a minor version bump; changing the meaning of an existing one is a major.

## Distribution

- Main entry: `dist/graph-format.js` (bundled ES module); types: `dist/graph-format.d.ts`
  (a one-line re-export of `dist/src/index.d.ts`).
- `files` ships `dist/`, `src/`, `README.md`, `LICENSE`; always run `npm run build:all` first.
- `WIRE_PRODUCER` in `src/wire/to-wire.ts` must equal `<name>@<version>` of package.json
  (`test/build-output.test.ts` enforces it); bump both together.
