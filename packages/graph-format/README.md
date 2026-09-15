# @graphty/graph-format

[![CI](https://github.com/graphty-org/graphty-monorepo/actions/workflows/ci.yml/badge.svg)](https://github.com/graphty-org/graphty-monorepo/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/graphty-org/graphty-monorepo/badge.svg?branch=master)](https://coveralls.io/github/graphty-org/graphty-monorepo?branch=master)
[![npm version](https://img.shields.io/npm/v/@graphty/graph-format.svg)](https://www.npmjs.com/package/@graphty/graph-format)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Frozen CSR graph data format over typed arrays, shared by CPU and WebGPU graph algorithms.

@graphty/graph-format is a zero-runtime-dependency, framework-agnostic graph DATA FORMAT: a frozen
compressed-sparse-row (CSR) representation over typed arrays, a mutable builder that produces frozen
snapshots, columnar typed node/edge attribute storage, and an id-to-index map kept outside the CSR.
It contains no graph algorithms, no rendering and no parsers (see @graphty/graph-io).

## Features

- **Frozen CSR snapshot** (`GraphSnapshot`): `rowPtr` / `colIdx` / `weights` / `arcToEdge` /
  `edgeToArc` as 4-byte-aligned typed arrays in one 256-byte-aligned arena, so the whole core uploads
  to a GPU with one `writeBuffer`. Topology, counts, flags and id map never change after `freeze()`.
- **Builder** (`GraphBuilder`): long-lived, mutable, repeatedly freezable; ids or indices; directed
  or undirected (with expansion); duplicate-edge and self-loop policies; f32 or f64 weights; two
  stable counting sorts (never a comparator sort) so every freeze is deterministic and prefix-stable.
  Column names resolve through an index (declaring and looking up 100k columns is linear), and
  an inferred column can be widened by name through `widenNodeColumn()` / `widenEdgeColumn()`
  (also on the `GraphSink` contract, optional) so a text importer can give a `2.0`-style column its
  f64 dtype whatever its values imply.
- **Typed attribute columns** (`AttributeTable`, `Column`): `f32 f64 i32 u32 u8 bool dict string
list json`, multi-component strides, Arrow-style validity bitmaps, roles, provenance, mutable
  columns for layouts, extension tables for temporal attributes, and `gpuView()` for what the GPU
  binds.
- **Id map** (`NodeIdMap`): five storage kinds (`identity`, `dense`, `numeric`, `string`, `mixed`)
  chosen at freeze; zero bytes for the common `0..n-1` case; lazy reverse map.
- **Views** (lazy, cached, shared): `reverse()`, `coo()`, `edgeList()`, degrees, weighted degrees,
  self-loops, `mate()`, `degreeOrder()`, `isSymmetric()`.
- **Derived graphs** (new snapshots with index maps back): `toUndirected()`, `transpose()`,
  `simplified()`, `withoutSelfLoops()`, `filterEdges()`, `inducedSubgraph()`, `contract()`,
  `relabel()`, `withColumns()`.
- **Wire form and GSNP container**: `toWire()` / `fromWire()` for `postMessage` and IndexedDB
  (transfer-aware: shared buffers are copied, exclusive ones transferred), `toBytes()` / `fromBytes()`
  / `fromByteChunks()` for files and the network, with `none` / `structure` / `full` validation of
  untrusted input.
- **Factories**: `fromEdgeArrays()`, `fromCsr()` (adopts caller arrays with validation),
  `fromRecords()` (node-link JSON shapes).

## Installation

```bash
npm install @graphty/graph-format
```

ESM only. Node >= 18.19.0 or any browser with ES2020 support. No runtime dependencies.

## Quick Start

```typescript
import { fromBytes, GraphBuilder } from "@graphty/graph-format";

// 1. Build. Ids are strings or numbers; indices are assigned in first-appearance order.
const builder = new GraphBuilder({ directed: false, weightDtype: "f32" });
builder.declareNodeColumn({ name: "label", dtype: "string", role: "label" });
builder.addNodeRecord("alice", { label: "Alice" });
builder.addNodeRecord("bob", { label: "Bob" });
builder.addEdge("alice", "bob", 2.5);
builder.addEdge("bob", "carol", 1); // "carol" is added on first mention

// 2. Freeze: a frozen CSR snapshot; the builder stays usable and can be frozen again.
const snapshot = builder.freeze();
console.log(snapshot.nodeCount, snapshot.edgeCount, snapshot.arcCount); // 3 2 4 (undirected: two arcs per edge)

// 3. Walk rows. Every out-arc of node u lives in [rowPtr[u], rowPtr[u + 1]).
const { rowPtr, colIdx, weights } = snapshot;
const strength = new Float64Array(snapshot.nodeCount);
for (let u = 0; u < snapshot.nodeCount; u++) {
    for (let a = rowPtr[u]; a < rowPtr[u + 1]; a++) {
        const v = colIdx[a];
        strength[u] += weights === null ? 1 : weights[a];
        void v; // the neighbour index
    }
}

// 4. Map results back to ids only at the boundary.
const byId = snapshot.ids.toMap(strength); // Map<NodeId, number>: "alice" -> 2.5, "bob" -> 3.5, "carol" -> 1
const labels = snapshot.nodes.requireTyped("label", "string");
console.log(labels.valueAt(snapshot.ids.requireIndex("bob"))); // "Bob"

// 5. Views are lazy, cached and shared (call .slice() for scratch); derived graphs are new snapshots.
const degree = snapshot.degree(); // Uint32Array(nodeCount)
const { snapshot: simple, edgeRemap } = snapshot.simplified();
void degree;
void simple;
void edgeRemap;

// 6. Serialise: one contiguous GSNP container (or toWire() for postMessage / IndexedDB).
const bytes = snapshot.toBytes();
const back = fromBytes(bytes); // validates fully by default
console.log(back.ids.idOf(0), byId.get("bob")); // "alice" 3.5
```

## Invariants

Every snapshot, whether frozen by the builder, adopted by `fromCsr()` or decoded by `fromBytes()`,
satisfies all of the following (design section 3.2). Kernels and algorithms may assume them without
checking; `validate()` re-checks I1-I13. Changing any of them is a data-model major bump.

- **I1** `rowPtr.length === nodeCount + 1`, `rowPtr[0] === 0`, `rowPtr` non-decreasing,
  `rowPtr[nodeCount] === arcCount === colIdx.length`.
- **I2** `colIdx[a] < nodeCount` for every arc; `INVALID_INDEX` never appears in a core or view array.
- **I3** `nodeCount`, `edgeCount`, `arcCount <= MAX_COUNT (0xFFFFFFFE)`; `edgeCount <= arcCount`.
  Arc indices may exceed 2^31: never apply JS bitwise operators to them.
- **I4** Sorted rows: within a row `colIdx` is non-decreasing and equal targets are ordered by
  ascending `arcToEdge`. There is no unsorted mode.
- **I5** Permutations: `arcToEdge` (arcCount entries, values `< edgeCount`) and `edgeToArc`
  (edgeCount entries, values `< arcCount`) with `arcToEdge[edgeToArc[e]] === e`; the arc at
  `edgeToArc[e]` stores the declared `source -> target` orientation of edge `e`.
- **I6** Directed: `arcCount === edgeCount` and `arcToEdge` is a permutation of `0..edgeCount-1`.
- **I7** Undirected: doubled storage; every non-loop arc has exactly one mate in the opposite row
  with the same edge and weight; a self-loop has one arc; `arcCount === 2 * edgeCount - selfLoopCount`.
- **I8** `weights === null` or `weights.length === arcCount`; no `NaN` (infinities are legal and
  reported by `flags.finiteWeights`).
- **I9** Flags are truthful: every `SnapshotFlags` member is a predicate over the arrays, computed at
  freeze or by validation, never guessed.
- **I10** Alignment: every core array, 4-byte view array and GPU-bindable column array is a view over
  a plain `ArrayBuffer` with `byteOffset % 4 === 0` and `byteLength % 4 === 0`; arena segments start
  at multiples of 256 bytes.
- **I11** `ids` is a bijection between `[0, nodeCount)` and the id set (SameValueZero, no `NaN`).
- **I12** `nodes.rowCount === nodeCount`, `edges.rowCount === edgeCount`, `graph.rowCount === 1`;
  every column obeys its length rules; dictionary codes and `refersTo` indices are in range.
- **I13** Edge attribute columns are indexed by logical edge, never by arc.
- **I14** Node index = order of first appearance; edge index = order of `addEdge` minus removals and
  merges, relative order preserved. Nothing is ever sorted by id or degree at freeze.
- **I15** Determinism: the same builder operations with the same options give byte-identical core
  arrays, flags, id map and columns on every run and engine.
- **I16** Prefix stability: without removals or merging policies between two freezes, the earlier
  snapshot's indices are a prefix of the later one's; `freezeWithReport` says exactly when a remap
  happened.
- **I17** After `freeze()` no core array, count, flag, id-map entry or immutable column changes; views
  are memoised once and shared; only the column SET and the CONTENTS of `mutable` columns may change.
- **I18** No core array, view array or column aliases memory a builder can still write.

## Counting Vocabulary

```
nodeCount      n
edgeCount      number of logical edges (NetworkX number_of_edges);
               every EDGE column has edgeCount rows
arcCount       colIdx.length === rowPtr[nodeCount];
               every ARC-aligned array (weights, arcToEdge, per-arc results) has arcCount entries
selfLoopCount  logical edges with source === target
directed:      arcCount === edgeCount
undirected:    arcCount === 2 * edgeCount - selfLoopCount
INVALID_INDEX  0xFFFFFFFF, the one "no index" sentinel for node, edge and arc indices,
               in JS return values and inside Uint32Array vectors alike
```

A **node index** is a dense integer `0 <= i < nodeCount`; a **node id** (`string | number`) lives only
in the id map and at API boundaries and is never coerced. A **logical edge** is one edge as declared;
an **arc** is one entry of `colIdx` (a directed edge is one arc, an undirected edge two, an undirected
self-loop one). A **row** is the arc range `[rowPtr[u], rowPtr[u + 1])` of node `u`.

## Views and Derived Graphs

Views are pure functions of the core, computed on first call, cached for the life of the snapshot
and SHARED (writing into one is a contract violation; `.slice()` first). `prepare(names)` computes
a set eagerly, `dropCaches()` releases them, `cachedViews()` lists what is resident.

| Method                                                            | Returns             | Cost                   | Notes                                                       |
| ----------------------------------------------------------------- | ------------------- | ---------------------- | ----------------------------------------------------------- |
| `reverse()`                                                       | `ReverseView`       | O(n + m)               | in-adjacency; undirected: the forward arrays themselves     |
| `coo()`                                                           | `CooView`           | O(m)                   | `src` is the only new array                                 |
| `edgeList()`                                                      | `EdgeListView`      | O(m)                   | each logical edge once, declared orientation                |
| `outDegree()` / `inDegree()` / `degree()`                         | `U32(n)`            | O(n) / O(m) / O(n + m) | graph-theoretic degree counts an undirected self-loop twice |
| `weightedOutDegree()` / `weightedInDegree()` / `weightedDegree()` | `F64(n)`            | O(m)                   | f64 for modularity / PageRank consumers                     |
| `selfLoopWeight()` / `totalWeight()`                              | `F64(n)` / `number` | O(n log d) / O(m)      |                                                             |
| `selfLoopArcs()` / `selfLoopsPerNode()` / `selfLoopsAt(u)`        | `U32`               | O(n log d)             |                                                             |
| `mate()`                                                          | `U32(arcCount)`     | O(m)                   | undirected only (`E_DIRECTED`)                              |
| `degreeOrder({ of })`                                             | `DegreeOrderView`   | O(n + maxDegree)       | `perm` plus segment offsets at the 1024 / 32 / 1 thresholds |
| `isSymmetric()`                                                   | `boolean`           | O(m)                   |                                                             |

Derived graphs are new snapshots with maps back to the source (`nodeOrigin`, `edgeOrigin`,
`nodeRemap`, `edgeRemap`, each `null` when that index space is unchanged); the format never caches
them.

| Method                                 | Node space     | Edge space | Semantics                                                                        |
| -------------------------------------- | -------------- | ---------- | -------------------------------------------------------------------------------- |
| `toUndirected(opts?)`                  | same           | new        | reciprocal pairs collapse keep-first; `reciprocal: true` keeps mutual pairs only |
| `transpose()`                          | same           | same       | every orientation swapped, zero copy of `reverse()`                              |
| `simplified(opts?)`                    | same           | new        | one edge per `(u, v)` group                                                      |
| `withoutSelfLoops()`                   | same           | new        |                                                                                  |
| `filterEdges(mask)`                    | same           | new        | packed bitmap over logical edges                                                 |
| `inducedSubgraph(indices \| { mask })` | new            | new        | edges with both endpoints kept                                                   |
| `contract(partition, opts?)`           | new (k blocks) | new        | Leiden / Louvain aggregation and condensation                                    |
| `relabel(perm)`                        | permuted       | same       | `perm[newIndex] = oldIndex`                                                      |
| `withColumns(nodes?, edges?)`          | same           | same       | shares the core, clones the column set                                           |

## GPU Contract

`snapshot.arena` describes one `ArrayBuffer` holding the core arrays at 256-byte-aligned offsets
(`hotByteLength` covers `rowPtr`, `colIdx` and `weights`); every core array, every 4-byte view and
`table.gpuView(name)` are plain-`ArrayBuffer` views accepted by `GPUQueue.writeBuffer` without a
cast. `flags` (`weighted`, `multigraph`, `hasSelfLoops`, `arcToEdgeIsIdentity`, ...) let kernels
branch once at dispatch.

## Errors

Every failure is a `GraphFormatError` with a stable `code` (`E_INVALID_ID`, `E_DUPLICATE_EDGE`,
`E_COLUMN_TYPE`, `E_INVALID_SNAPSHOT`, `E_BAD_SERIALIZATION`, ...) and a `details` record naming
the offending index, column or invariant. A throwing call leaves the builder or snapshot unchanged.

## Versioning

Three version numbers apply: the npm package version (conventional commits), `FORMAT_VERSION` (the
data-model major, compared by `isGraphSnapshot()` and the wire readers) and the wire `[major, minor]`
of the byte container. The invariants, `INVALID_INDEX`, the counting vocabulary and the view tables
are public API: changing any of them is a breaking change even when no TypeScript signature changes.

## License

MIT
