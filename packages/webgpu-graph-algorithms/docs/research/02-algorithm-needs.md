# 02 - Algorithm inventory and how optional / detected GPU acceleration plugs in

Status: planning note, 2026-09-14. Input to the WebGPU implementation plan.
Everything about existing code below cites a file path; everything external
cites a URL in the Sources section. Nothing here is implemented.

Scope: (1) what `@graphty/algorithms` and `@graphty/layout` export today and
what result shape each thing must produce for parity with the planned
`indexed.*` namespace; (2) how graphty-element invokes algorithms and layouts
and writes results back; (3) the four candidate plug-in mechanisms for
"optional / detected" acceleration, compared and one recommended; (4) a
prioritised, scored list of GPU targets with the result shape each must
return.

---

## 0. Summary of findings that shape the plan

1. Every public algorithm in `@graphty/algorithms` is a SYNCHRONOUS function
   over the Map-of-Maps `Graph` (e.g. `pageRank(graph, options): PageRankResult`,
   `algorithms/src/algorithms/centrality/pagerank.ts:83`). A WebGPU result
   needs `mapAsync`, so GPU acceleration can NEVER be spliced transparently
   into the existing sync entry points without changing their return type.
   The plug-in point must be an async layer: graphty-element's adapters are
   already `async run()` (`graphty-element/src/algorithms/Algorithm.ts:283`),
   and the design's `LayoutSimulation.step()` is already `void | Promise<void>`
   (design doc lines 3977-3984). So the acceleration seam is (a) the
   graphty-element adapter / engine layer and (b) a new async dispatcher in
   the CPU packages, not the legacy sync facades.
2. The accepted design already decides the shape at the graphty-element
   level: adapters call `getSnapshot()` once, run `indexed.*` "(or the
   injected GPU accelerator)", then the same result-writing loop (design doc
   line 4180); the WebGPU package is "an OPTIONAL accelerator injected by the
   caller" (line 70) and graphty-element "injects it as `runAlgorithm(snapshot,
   { accelerator: gpu })`" (lines 4239-4240). Mechanism (a) below is therefore
   not a free choice; it is the accepted baseline. The open question is only
   how "detected" is layered on top of "injected".
3. graphty-element already has the pattern and the scar tissue for an
   optional peer loaded by dynamic import: `@mlc-ai/web-llm` is an optional
   peer (`graphty-element/package.json` `peerDependenciesMeta`), externalised
   in `vite.config.ts:39`, and its provider module is deliberately NOT
   imported from the barrel because "Safari fails on dynamic imports of
   non-existent modules even before the import is called"
   (`graphty-element/src/ai/providers/index.ts:9-13`). Any (c)-style wiring
   must copy that isolation.
4. graphty-element's DEFAULT layout is `ngraph` (`graphty-element/src/config/
   GraphBehavior.ts:13`), a per-frame stepped force simulation
   (`NGraphLayoutEngine.ts:168-191`), not one of the `@graphty/layout`
   force layouts. The GPU force-directed layout must therefore be a
   per-frame `LayoutSimulation` that the `LayoutManager` steps
   (`managers/LayoutManager.ts:241-244`), i.e. an interactive replacement
   for the ngraph experience at large N, not only a faster one-shot
   `forceatlas2Layout`.
5. The GPU package must take the `GPU`/`GPUDevice` from the caller and never
   touch `navigator` (research note 09 section 7.2, "Design consequence").
   That is what makes one code base run in Node (Dawn via the `webgpu` npm
   package, already used by `packages/graph-format/test/audit/gpu-upload.test.ts:41-49`)
   and in the browser, and it keeps the GPU package at zero runtime
   dependencies (the `webgpu` native module is a devDependency for tests
   and benchmarks only).
6. The highest-value / lowest-risk GPU algorithms are the SpMV family
   (PageRank, HITS, eigenvector, Katz: one pull kernel over `reverse()`),
   then connected components (edge-parallel hook/compress over
   `edgeList()`), then BFS/SSSP/closeness (frontier machinery), then
   betweenness (the most expensive thing graphty users run, and the one
   with the best literature). Louvain/Leiden have the highest user value
   but the highest risk and belong after the frontier and SpMV primitives
   exist.

---

## 1. Inventory of `@graphty/algorithms` (v1.7.2)

Source: `algorithms/src/index.ts` re-exports `./algorithms/index.js`,
`./research/index.js`, `./data-structures/index.js`, `./optimized/index.js`
plus the `Graph` class and the result types in `algorithms/src/types/index.ts`.
Category barrels: `algorithms/src/algorithms/index.ts` (traversal,
shortest-path, centrality, components, mst, community, pathfinding, flow,
clustering, matching, link-prediction).

Core result types (`algorithms/src/types/index.ts:31-108`):

```ts
type NodeId = string | number;
interface ShortestPathResult { distance: number; path: NodeId[]; predecessor: Map<NodeId, NodeId | null>; }
type CentralityResult = Record<string, number>;
interface TraversalResult { visited: Set<NodeId>; order: NodeId[]; tree?: Map<NodeId, NodeId | null>; }
interface CommunityResult { communities: NodeId[][]; modularity: number; iterations?: number; }
interface ComponentResult { components: NodeId[][]; componentMap: Map<NodeId, number>; }
interface MSTResult { edges: Edge[]; totalWeight: number; }
interface FloydWarshallResult { distances: Map<NodeId, Map<NodeId, number>>; predecessors: Map<...>; hasNegativeCycle: boolean; }
interface BellmanFordResult { distances: Map<NodeId, number>; previous: Map<NodeId, NodeId | null>; hasNegativeCycle: boolean; negativeCycleNodes?: NodeId[]; }
```

The table lists every exported algorithm function (signatures from the
`grep "^export function"` over `algorithms/src`, file:line given), its
legacy result shape, the planned `indexed.*` shape (design doc section
14.2, lines 3728-3760, "Result-type conventions"), and whether
graphty-element has an adapter for it (`graphty-element/src/algorithms/index.ts`
registers 23 adapters). "GPU" is the verdict developed in section 7.

### 1.1 Traversal (`algorithms/src/algorithms/traversal/`)

| Export | Signature (file:line) | Legacy result | `indexed` shape (design 14.2) | Element adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `breadthFirstSearch` | `(graph, startNode, options?: TraversalOptions): TraversalResult` (`bfs-unified.ts:46`); auto-switches to `DirectionOptimizedBFS` above 10,000 nodes (`bfs-unified.ts:53-55`) | `TraversalResult` | `BfsResult { order: U32; parent: U32; depth: U32; visitedCount }` (design lines 3822-3843) | `BFSAlgorithm` | yes (frontier) |
| `shortestPathBFS` | `(graph, source, target): ShortestPathResult \| null` (`bfs-unified.ts:174`) | path + predecessor Map | `SsspResult` unweighted | via Dijkstra adapter | derived from BFS |
| `singleSourceShortestPathBFS` | `(graph, source): Map<NodeId, ShortestPathResult>` (`bfs-unified.ts:290`) | Map per node (O(n^2) pathology noted in design) | `SsspResult { dist; predArc; pathTo(); pathEdges() }` | -- | derived from BFS |
| `isBipartite` | `(graph): boolean` (`bfs-unified.ts:395`) | boolean | 2-colouring `U8`/mask + boolean | -- | derived from BFS (parity of depth) |
| `depthFirstSearch` | `(graph, startNode, options?: DFSOptions): TraversalResult` (`dfs.ts:26`) | `TraversalResult` | same as BFS | `DFSAlgorithm` | no (sequential) |
| `hasCycleDFS` | `(graph): boolean` (`dfs.ts:165`) | boolean | boolean | -- | no |
| `topologicalSort` | `(graph): NodeId[] \| null` (`dfs.ts:246`) | order or null | `U32` order or null | -- | no (Kahn's is parallelisable but low value) |
| `findStronglyConnectedComponents` | `(graph): NodeId[][]` (`dfs.ts:296`) | groups | `{ labels: U32; count; groups() }` | `StronglyConnectedComponentsAlgorithm` (via components) | later (FW-BW) |
| `bfs-variants.ts` (`bfsWithPathCounting`, `bfsDistancesOnly`, `bfsColoringWithPartitions`, `bfsAugmentingPath`, `bfsWeightedDistances`) | internal helpers, not in the barrel | -- | -- | -- | building blocks of betweenness / bipartite / flow |

### 1.2 Shortest path (`algorithms/src/algorithms/shortest-path/`)

| Export | Signature | Legacy result | `indexed` shape | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `dijkstra` | `(graph, source, options?: DijkstraOptions): Map<NodeId, ShortestPathResult>` (`dijkstra.ts:21`) | Map per node | `SsspResult { dist: F64; predArc: U32; pathTo; pathEdges }` (design lines 3849-3869; predecessor is the relaxing ARC) | `DijkstraAlgorithm` (calls `dijkstraPath` then `dijkstra`, `DijkstraAlgorithm.ts:167-171`) | yes (delta-stepping / edge-relax) |
| `dijkstraPath` | `(graph, source, target, options?)` (`dijkstra.ts:128`) | `ShortestPathResult \| null` | `pathTo(target)` on `SsspResult` | Dijkstra adapter | derived |
| `singleSourceShortestPath` | `(graph, source, cutoff?): Map<NodeId, number>` (`dijkstra.ts:172`) | distance Map | `F64`/`F32` dist | -- | derived |
| `allPairsShortestPath` | `(graph): Map<NodeId, Map<NodeId, number>>` (`dijkstra.ts:255`) | n x n Maps | `F32(n*n)` (design line 4014: KK `dist` accepts `Float32Array(n*n)` "from `@graphty/algorithms` or the GPU package") | -- | yes (n batched BFS / blocked FW), n <= ~10k |
| `BidirectionalDijkstra` (class, `bidirectional-dijkstra.ts:26`) | internal | -- | -- | -- | no |
| `bellmanFord` | `(graph, source, options?): BellmanFordResult` (`bellman-ford.ts:52`) | Maps + negative cycle flag | `SsspResult` + `hasNegativeCycle` | `BellmanFordAlgorithm` | yes (edge-parallel relax over `edgeList()`, design section 10.1) |
| `bellmanFordPath`, `hasNegativeCycle` (`bellman-ford.ts:138,174`) | -- | -- | derived | -- | derived |
| `floydWarshall` | `(graph): FloydWarshallResult` (`floyd-warshall.ts:15`) | n x n Maps | `{ dist: F32(n*n); pred: U32(n*n); hasNegativeCycle }` | `FloydWarshallAlgorithm` | yes (blocked FW), n <= ~10k |
| `floydWarshallPath`, `transitiveClosure` (`floyd-warshall.ts:120,172`) | -- | -- | derived / `U8(n*n)` bitmap | -- | derived |

### 1.3 Centrality (`algorithms/src/algorithms/centrality/`)

| Export | Signature | Legacy result | `indexed` shape | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `degreeCentrality` | `(graph, options?: CentralityOptions): CentralityResult` (`degree.ts:17`) | `Record<string, number>` | `F64(n)` (CPU) | `DegreeAlgorithm` | trivial: `rowPtr[v+1]-rowPtr[v]`; use as the walking-skeleton kernel, not a product feature |
| `nodeDegreeCentrality` (`degree.ts:67`) | single node | number | number | -- | no |
| `pageRank` | `(graph, options?: PageRankOptions): PageRankResult` (`pagerank.ts:83`) | `{ ranks: Record<string, number>; iterations; converged }` | `{ scores: F64(n); iterations; converged }` (design lines 3873-3898; pull over `reverse()`, dangling mass) | `PageRankAlgorithm` (writes `rank`, `rankPct`, graph `iterations`/`converged`/`maxRank`, `PageRankAlgorithm.ts:224-240`) | YES, first algorithm |
| `personalizedPageRank` (`pagerank.ts:292`) | + personalization Map | same | same + `F32(n)` personalization vector | via adapter option (programmatic only) | same kernel, different teleport vector |
| `pageRankCentrality`, `topPageRankNodes` (`pagerank.ts:332,344`) | Record / `{ node, rank }[]` | -- | `U32` indices + `F64` (design 14.2 last row) | -- | derived (top-k = readback + CPU partial sort) |
| `DeltaPageRank`, `PriorityDeltaPageRank` (classes, `delta-pagerank.ts:52,293`), `SimpleDeltaPageRank` (`delta-pagerank-simple.ts:11`) | incremental | -- | "a second indexed implementation over the same view" (design line 3900) | adapter option `useDelta` | no (the GPU converges the plain iteration faster than delta bookkeeping) |
| `betweennessCentrality` | `(graph, options?: BetweennessCentralityOptions): Record<string, number>` (`betweenness.ts:204`) | Record | `F64(n)` | `BetweennessCentralityAlgorithm` (`score`, `scorePct` min-max normalised, `BetweennessCentralityAlgorithm.ts:59-78`) | YES (Brandes forward BFS + backward dependency, McLaughlin-Bader hybrid) |
| `nodeBetweennessCentrality` (`betweenness.ts:248`) | one node | number | number | -- | derived |
| `edgeBetweennessCentrality` (`betweenness.ts:267`) | `Map<string, number>` keyed `"v-w"` | `F64(edgeCount)` (design 14.2 "edge betweenness" row) | -- | same kernel, per-arc accumulate + `foldArcs` |
| `closenessCentrality`, `nodeClosenessCentrality`, `weightedClosenessCentrality`, `nodeWeightedClosenessCentrality` (`closeness.ts:114,132,155,176`) | Record | `F64(n)` | `ClosenessCentralityAlgorithm` | yes (multi-source BFS batches; weighted = batched SSSP) |
| `eigenvectorCentrality` (`eigenvector.ts:28`) | Record | `F64(n)` | `EigenvectorCentralityAlgorithm` | yes (power iteration = SpMV + norm reduce) |
| `hits` (`hits.ts:34`) | `HITSResult { hubs; authorities }` Records | `{ hubs: F64(n); authorities: F64(n); iterations; converged }` | `HITSAlgorithm` | yes (two SpMVs: forward and `reverse()`) |
| `katzCentrality` (`katz.ts:31`) | Record | `F64(n)` | `KatzCentralityAlgorithm` | yes (SpMV with alpha/beta) |
| `node*` single-node variants (`hits.ts:194`, `eigenvector.ts:150`, `katz.ts:124`) | number | number | -- | derived |

### 1.4 Components (`algorithms/src/algorithms/components/connected.ts`)

| Export | Signature | Legacy result | `indexed` shape | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `connectedComponents` (`:17`), `connectedComponentsDFS` (`:45`), `weaklyConnectedComponents` (`:233`) | `(graph): NodeId[][]` | groups | `{ labels: U32(n); count; groups(): U32[] }` with labels "renumbered 0..count-1 in first-seen node order" (design lines 3902-3906, Port 4) | `ConnectedComponentsAlgorithm` | YES (edge-parallel hook / compress over `edgeList()`) |
| `numberOfConnectedComponents` (`:87`), `isConnected` (`:96`), `isWeaklyConnected` (`:261`) | scalar | -- | `count` | -- | derived |
| `largestConnectedComponent` (`:105`), `getConnectedComponent` (`:121`) | `NodeId[]` | `U32` index list | -- | derived (histogram of labels) |
| `stronglyConnectedComponents` (`:143`), `isStronglyConnected` (`:218`) | groups | `{ labels; count; groups() }`; labels in Tarjan completion order for `condensationGraph` parity (design 14.2 "condensationGraph" row) | `StronglyConnectedComponentsAlgorithm` | later (forward-backward reachability; label order will NOT match Tarjan's, so parity is set-equality only) |
| `condensationGraph` (`:274`) | `Graph` | `DerivedGraph` from `contract()` | -- | no (CPU `contract()`) |

### 1.5 MST (`algorithms/src/algorithms/mst/`)

| Export | Signature | Legacy | `indexed` | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `kruskalMST` (`kruskal.ts:18`), `minimumSpanningTree` (`kruskal.ts:74`) | `(graph): MSTResult` | `{ edges: Edge[]; totalWeight }` | `{ edges: U32 (logical edge indices); totalWeight }` (design Port 5, lines 3908-3916) | `KruskalAlgorithm` (matches MST edges by index through `edgeRemap`, design line 3915) | yes but medium value (Boruvka over `edgeList()`, per-component min-edge via atomicMin on packed (weight bits, edge) u32 pairs) |
| `primMST` (`prim.ts:15`) | `(graph, startNode?): MSTResult`; throws on directed | same | same, "discovery arc" convention (design 14.2) | `PrimAlgorithm` | no (sequential heap); the GPU Boruvka result is the same set on distinct weights; ties differ |

### 1.6 Community (`algorithms/src/algorithms/community/`)

| Export | Signature | Legacy | `indexed` | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `louvain` (`louvain.ts:20`), `louvainOptimized` (`louvain-optimized.ts:471`) | `(graph, options?: LouvainOptions): CommunityResult` | groups + modularity | `{ labels: U32; count; groups(); modularity; iterations }`; aggregation through `snapshot.contract(partition)` per level (design lines 3790-3795) | `LouvainAlgorithm` (`communityId` per node, `groupCount`, `modularity`, `LouvainAlgorithm.ts:166-184`) | later, high risk (cuGraph-style hash-based gain, contraction on CPU per level) |
| `leiden` (`leiden.ts:591`) | `LeidenResult` (Map keyed `String(id)`) | `U32` labels via `toStringMap` | `LeidenAlgorithm` | later, after Louvain |
| `labelPropagation`, `labelPropagationAsync`, `labelPropagationSemiSupervised` (`label-propagation.ts:378,391,405`) | `LabelPropagationResult` (Map keyed `String(id)`) | `{ labels: U32; iterations; converged }` | `LabelPropagationAlgorithm` | yes (Jacobi-style; per-node neighbour-label mode via sorted-row scan or workgroup hash) |
| `girvanNewman` (`girvan-newman.ts:23`) | `CommunityResult[]` | `{ levels: U32[]; modularity: F64 }` | `GirvanNewmanAlgorithm` | no (repeated edge betweenness + removal; could reuse the GPU edge-betweenness kernel per level, but O(m^2 n) makes it a small-graph algorithm by nature) |
| `modularity-utils.ts` (`calculateModularity` etc.) | helpers | -- | `weightedDegree()`-based (design line 3787) | -- | a modularity reduce kernel is cheap and useful for parity checks |

### 1.7 Clustering, flow, matching, link prediction, pathfinding, research

| Export | File:line | Legacy | `indexed` | Adapter | GPU |
| --- | --- | --- | --- | --- | --- |
| `kCoreDecomposition` (`clustering/k-core.ts:488`), `getKCore` (`:223`) | -- | `KCoreResult` (coreness Map keyed string), `Set<string>` | `U32(n)` coreness; packed `NodeMask` | -- | yes, medium (parallel peeling, one level per dispatch) |
| `kTruss`, `degeneracyOrdering`, `getKCoreSubgraph`, `toUndirected` (`k-core.ts:325,270,243,449`) | not all in barrel | -- | -- | -- | k-truss later (triangle counting) |
| `hierarchicalClustering` (`clustering/hierarchical.ts:544`), `cutDendrogram*`, `modularityHierarchicalClustering` | -- | dendrogram of `ClusterNode` | tree of `U32` | -- | no |
| `markovClustering`, `calculateMCLModularity` (`clustering/mcl.ts:38,467`) | -- | `MCLResult` | labels | -- | no for v1 (SpGEMM on device is a project of its own) |
| `spectralClustering` (`clustering/spectral.ts:44`) | -- | `SpectralClusteringResult` | labels | -- | later (Laplacian SpMV + k-means; the CPU version's eigen-solver is suspect, see research note 03 section 12.5) |
| `fordFulkerson`, `edmondsKarp`, `createBipartiteFlowNetwork` (`flow/ford-fulkerson.ts:394,410,335`) | -- | `MaxFlowResult` Maps | `{ maxFlow; flow: F64(edgeCount); sourceSide: NodeMask; cutEdges: U32 }` | `MaxFlowAlgorithm` | no (augmenting paths are sequential; push-relabel on GPU is out of scope) |
| `minSTCut`, `stoerWagner`, `kargerMinCut` (`flow/min-cut.ts:28,67,318`) | -- | `MinCutResult` | `{ cutValue; side: NodeMask; cutEdges: U32 }` | `MinCutAlgorithm` | no |
| `maximumBipartiteMatching`, `greedyBipartiteMatching`, `bipartitePartition` (`matching/bipartite.ts:33,118,97`) | -- | `BipartiteMatchingResult` | `U32(n)` mate with `INVALID_INDEX` | `BipartiteMatchingAlgorithm` | no |
| `isGraphIsomorphic`, `findAllIsomorphisms` (`matching/isomorphism.ts:45,374`) | -- | mapping Map | `U32(n)` | -- | no |
| `commonNeighborsScore/Prediction/ForPairs`, `getTopCandidatesForNode`, `evaluateCommonNeighbors` (`link-prediction/common-neighbors.ts:36,63,114,133,178`) | -- | `LinkPredictionScore[]` | sorted-row merge (design Port 6) | -- | later (edge-parallel intersection is the triangle-counting kernel) |
| `adamicAdar*` (`link-prediction/adamic-adar.ts:28-269`) | -- | same | same + `1/log(deg)` | -- | later, same kernel |
| `astar`, `astarWithDetails` (`pathfinding/astar.ts:17,110`) | generic `<T>` over a Map graph | path | -- | -- | no |
| `syncClustering`, `teraHAC`, `grsbm` (`research/*.ts`) | research | -- | -- | -- | no |

### 1.8 Data structures and `optimized/` (the existing CSR-ish code)

`algorithms/src/optimized/index.ts` exports `CSRGraph`, `DirectionOptimizedBFS`,
`CompactDistanceArray`, `GraphBitSet`, `VisitedBitArray`, `toCSRGraph`,
`isCSRGraph`, `createOptimizedGraph`, and the deprecated no-op
`configureOptimizations` / `getOptimizationConfig` (`graph-adapter.ts:182-195`).

- `CSRGraph` (`optimized/csr-graph.ts:44`) is built from a
  `Map<TNodeId, TNodeId[]>` adjacency list; it sorts node ids
  (`csr-graph.ts:78-80`), keeps `rowPointers: Uint32Array`, `columnIndices:
  Uint32Array`, optional `edgeWeights: Float32Array`, optional reverse
  arrays for bottom-up BFS, and `nodeIdToIndex: Map` / `indexToNodeId: []`
  (`csr-graph.ts:18-36`). Weights are keyed by `"source-target"` strings
  during conversion (`graph-adapter.ts:33-52`), so parallel edges collapse.
- `DirectionOptimizedBFS` (`optimized/direction-optimized-bfs.ts:32`) is
  Beamer's top-down / bottom-up switch with `alpha = 15`, `beta = 18`,
  `Int32Array parent` (-1 unvisited, -2 source).
- `bfs-unified.ts:19-33` caches the CSR in a `WeakMap<Graph, CSRGraph>`
  with no mutation check (the stale-cache bug the design's `mutationCount`
  memoisation fixes, design 14.1 rule 4).
- The design deletes all of this at 2.0: "`graphToMap`, `GraphAdapter`,
  `toCSRGraph`, `createOptimizedGraph`, `optimized/csr-graph.ts` and
  `optimized/graph-adapter.ts` are unreachable after A2 and deleted at 2.0"
  (design lines 3813-3816), and `DirectionOptimizedBFS` becomes
  `indexed.directionOptimizedBfs(snapshot)` using `reverse()` (lines 3845-3847).
- Consequence for the GPU package: there is NOTHING in `optimized/` to
  reuse; the measured cost of `toCSRGraph` is 2376 ms on 100k/1M (design
  line 76) versus 20 ms for the format's freeze. The GPU package consumes
  `GraphSnapshot` only (design 14.5) and the existing `CSRGraph` type in
  `/home/apowers/Projects/webgpu-graph-algorithms/src/types/index.ts:6-17`
  is deleted at move-in.

### 1.9 Layout inventory (`@graphty/layout` v1.6.2)

Force-directed exports (`layout/src/layouts/force-directed/*.ts`):
`forceatlas2Layout` (`forceatlas2.ts:26`), `fruchtermanReingoldLayout`
(`fruchterman-reingold.ts:24`), `springLayout` (`spring.ts:21`, alias of FR),
`arfLayout` (`arf.ts:16`), `kamadaKawaiLayout` (`kamada-kawai.ts:19`). All
take the `{ nodes(), edges(), getEdgeData? }` duck type and positional
parameters (ForceAtlas2 takes 15 positional arguments as called from
`graphty-element/src/layout/ForceAtlas2LayoutEngine.ts:155-171`) and return
a `PositionMap` (`Record<NodeId, number[]>`). Per research note 03 section 4
(table rows 6-9) FA2 allocates N x N x dim `diff` per iteration, ARF a dense
N x N `K`, KK an O(N^3) Floyd-Warshall, FR loops through `Record` lookups;
the practical ceilings are ~5k (FR), ~10k (FA2 memory), ~1-2k (KK).

After L1 the layout package exposes `indexed.*` entry points returning
`LayoutResult { positions: F32; dim; n }` plus `LayoutSimulation` for
steppable layouts (design lines 3963-3984). The GPU layouts implement
`LayoutSimulation` only; they never return a `PositionMap`.

---

## 2. How graphty-element invokes algorithms and layouts today

### 2.1 Algorithms

- Registry: `Algorithm.register(cls)` keys a `Map` by `"namespace:type"`
  (`graphty-element/src/algorithms/Algorithm.ts:318-325`); `Algorithm.get(g,
  ns, type, options)` instantiates (`:335-342`). 23 adapters are registered in
  `graphty-element/src/algorithms/index.ts:33-63`.
- Invocation: `Graph.runAlgorithm(namespace, type, options)` queues an
  `"algorithm-run"` operation (`graphty-element/src/Graph.ts:1207-1235`) and
  `AlgorithmManager.runAlgorithm` does `Algorithm.get(...)` then `await
  alg.run(this.graph)` (`managers/AlgorithmManager.ts:85-99`). The
  `run()` method is abstract and async (`Algorithm.ts:283`).
- Every adapter today converts with `toAlgorithmGraph(g, { directed?,
  addReverseEdges? })` (`graphty-element/src/algorithms/utils/graphConverter.ts:35`),
  which rebuilds an algorithms `Graph` from `DataManager.nodes/edges` per
  call, adding reverse edges for "undirected" (`graphConverter.ts:71-74`).
- Results: `addNodeResult(nodeId, name, value)` does a lodash `deepSet` on
  the `Node` object at `algorithmResults.<ns>.<type>.<name>`
  (`Algorithm.ts:236-247`); `addEdgeResult(edge, ...)` takes the Edge
  OBJECT (`:255-258`); `addGraphResult` writes `dm.graphResults` (`:265-272`).
  Every adapter also derives a normalised `*Pct` field (PageRank
  `rankPct = rank / maxRank`, `PageRankAlgorithm.ts:230-234`; betweenness
  min-max `scorePct`, `BetweennessCentralityAlgorithm.ts:72-78`; Louvain
  `communityId` from the groups, `LouvainAlgorithm.ts:166-178`).
- Styles consume these paths: e.g. PageRank's `suggestedStyles` maps
  `algorithmResults.graphty.pagerank.rankPct` to node size
  (`PageRankAlgorithm.ts:191-209`).

After E1 (design 14.4, line 4180) the adapter body becomes: `const s =
dm.getSnapshot();` -> `indexed.x(s)` or the injected accelerator -> `for (i <
n) addNodeResult(s.ids.idOf(i), name, vec[i])`; edge results go through
`edgesByIndex[e]` and `edgeRemap` (lines 4110-4113). The `*Pct`
normalisation is an O(n) pass over the readback and stays on the CPU.
`DataManager` emits `snapshot-replaced { previous, next, report }` and "the
GPU accelerator's `release(previous)` destroys its `GPUBuffer`s" (lines
4121-4126).

### 2.2 Layouts

- `LayoutEngine` base (`graphty-element/src/layout/LayoutEngine.ts:36-63`)
  requires `init()`, `addNode/addEdge`, `getNodePosition`, `setNodePosition`,
  `step()`, `pin/unpin`, `isSettled`. Registry `LayoutEngine.register` /
  `LayoutEngine.get(type, opts)` (`:96-118`).
- Thirteen `SimpleLayoutEngine` subclasses (`LayoutEngine.ts:194`) are
  one-shot: `doLayout()` re-materialises `nodes()` / `edges()` arrays per call
  and stores a `positions: Record` (`ForceAtlas2LayoutEngine.ts:150-171`,
  research note 03 section 8.1).
- The default engine is `ngraph` (`config/GraphBehavior.ts:13`), which is
  stepped every frame by `LayoutManager.step()` while `!isSettled`
  (`managers/LayoutManager.ts:241-244`), with a settle heuristic of average
  movement <= 0.05 or 1000 steps (`NGraphLayoutEngine.ts:184-190`). `d3`
  (`D3GraphLayoutEngine.ts:233`) is the other stepped engine.
- Post-L1/E1: `LayoutManager._setLayoutInternal` calls
  `engine.load(dm.undirected(getSnapshot()).snapshot, positions)` for
  `SimpleLayoutEngine` subclasses and `LayoutSimulation` engines "keep
  stepping on the new array" across topology changes (design line 4181-4182);
  drag writes `simulation.setPosition` "while a simulation is stepping, the
  GPU buffer being authoritative then" (line 4183).

The GPU force layout therefore plugs into the ELEMENT as a `LayoutEngine`
whose `step()` drives a `LayoutSimulation`, and into the LAYOUT PACKAGE as an
alternative `LayoutSimulation` implementation behind the same options.

---

## 3. Package manifests: dependency and export facts

| Package | Version | Runtime deps | Peer deps | Exports | Notes |
| --- | --- | --- | --- | --- | --- |
| `@graphty/algorithms` | 1.7.2 | `typedfastbitset` | none | single `"."` ESM entry `dist/algorithms.js` (`algorithms/package.json:8-14`) | `engines.node >= 18.19.0`; `publishConfig.provenance: true`; browser + default vitest projects |
| `@graphty/layout` | 1.6.2 | none | none | single `"."` ESM entry `dist/layout.js` (`layout/package.json:8-13`) | `provenance: true` |
| `@graphty/graphty-element` | 1.9.4 | `@graphty/algorithms`, `@graphty/layout`, `@graphty/remote-logger` (`workspace:*`), babylon-free deps, `ngraph.forcelayout`, `d3-force-3d`, ... | `@babylonjs/core ^8`, `lit ^3`, `@mlc-ai/web-llm >=0.2.0` OPTIONAL (`peerDependenciesMeta`) (`graphty-element/package.json:170-179`) | ESM + UMD; `sideEffects` lists the registration barrels `src/layout/index.ts`, `src/data/index.ts`, `src/algorithms/index.ts` (`:16-22`) | vite externalises `@mlc-ai/web-llm` (`vite.config.ts:39`) |
| `@graphty/webgpu-graph-algorithms` (scaffold) | 0.1.0, `private: true` | none | none | single ESM entry (`/home/apowers/Projects/webgpu-graph-algorithms/package.json:24-30`) | `webgpu@^0.4.0` is a devDependency of the staged graph-format package (`packages/graph-format/package.json:73`) |

Post-format rule (design 13.5 rule 3, lines 3658-3665): every consumer
declares `@graphty/graph-format` in BOTH `dependencies` (`workspace:*`) and
`peerDependencies` (`^<major>`) so an app gets one copy, and
`isGraphSnapshot()` is a `Symbol.for` brand check "never `instanceof`", so
duplicate copies interoperate. The GPU package is a consumer under the same
rule.

Release: `nx release` with `projectsRelationship: "independent"`,
conventional commits, `updateDependents: "auto"`, GitHub releases per
project (`graphty-monorepo/nx.json:4-34`); OIDC trusted publishing with npm
11 pinned (`.github/workflows/release.yml:36-52`), triggered only when the
CI workflow succeeded (`release.yml:19`). CI runs a per-package test matrix
on `ubuntu-latest` (`.github/workflows/ci.yml:235-300`).

---

## 4. The four plug-in mechanisms compared

Definitions used below. "Accelerator" is an object the GPU package produces
once per device: `const gpu = await GpuContext.create({ gpu: navigator.gpu })`
in the browser, `create({ gpu: dawn.create([]) })` in Node. It implements
two structural interfaces that the CPU packages OWN:

```ts
// @graphty/algorithms (indexed layer), no WebGPU types anywhere
export interface AlgorithmAccelerator {
    pageRank?(s: GraphSnapshot, o?: PageRankOptions): Promise<PageRankResult>;
    breadthFirstSearch?(s: GraphSnapshot, start: number, o?: BfsOptions): Promise<BfsResult>;
    connectedComponents?(s: GraphSnapshot): Promise<LabelResult>;
    // ... one optional method per accelerated indexed.* function, same option and result types
    release?(s: GraphSnapshot): void;
}
// @graphty/layout
export interface LayoutAccelerator {
    forceAtlas2?(o: ForceAtlas2Options): LayoutSimulation;
    fruchtermanReingold?(o: FrOptions): LayoutSimulation;
}
```

The GPU package implements both (it depends on `@graphty/graph-format` at
runtime and on `@graphty/algorithms` / `@graphty/layout` as
`devDependencies` for the type-level conformance test `expectTypeOf(gpu)
.toMatchTypeOf<AlgorithmAccelerator>()`, so no runtime import of the CPU
packages is needed and the dependency graph stays acyclic).

### 4.1 (a) Explicit injection: `{ accelerator }` option

How: `indexed.run(acc, "pageRank", s, o)` or per-function async twins in the
CPU packages: `indexed.pageRankAsync(s, o, { accelerator })` returns
`acc?.pageRank ? acc.pageRank(s, o) : Promise.resolve(indexed.pageRank(s,
o))`. graphty-element's adapters read `this.graph.accelerator` and pass it.
The app (or a test) constructs the accelerator and injects it.

| Criterion | Assessment |
| --- | --- |
| Dependency direction | GPU -> graph-format (runtime); GPU -> algorithms/layout (dev, types only); algorithms/layout -> nothing new; graphty-element -> nothing new. Acyclic. |
| Bundling / tree-shaking | Zero impact on any CPU package or the element: no import of the GPU package exists outside the app. The GPU package is imported by whoever injects it (statically or via `import()` for code splitting, the app's choice). |
| SSR / Node | Nothing GPU-related executes unless the caller injects. In Node the caller passes Dawn's `GPU`; the GPU package never references `navigator`. |
| Testability | Best: a fake accelerator (`{ pageRank: async () => fixture }`) tests the dispatcher and adapters without a GPU; the GPU package tests itself differentially against `indexed.*`. |
| Release process | GPU package is an independent nx project with its own conventional-commit versioning; no version coupling with the element beyond the shared graph-format peer range. |
| Never-falls-back rule | Cleanest: if `accelerator` is present its method is called and any throw propagates; if absent the CPU runs. There is no place where a CPU fallback could be written "by accident". |
| "Detected" | Not by itself: detection is the injector's job (section 4.5). |

### 4.2 (b) Registry: `registerAccelerator(acc)` / `setAccelerator(acc)` consulted by the CPU packages

How: a module-level singleton in `@graphty/algorithms` and `@graphty/layout`;
the GPU package (or the app) calls `registerAccelerator(gpu)` at start-up and
every `indexed.*Async` call consults it.

| Criterion | Assessment |
| --- | --- |
| Dependency direction | If the GPU package self-registers it must import `@graphty/algorithms` and `@graphty/layout` at runtime: the dependency direction inverts (GPU -> CPU packages), and the GPU package's own tests then need both CPU packages' `dist`. If the app registers, it is just (a) with global state. |
| Bundling / tree-shaking | A self-registering import (`import "@graphty/webgpu-graph-algorithms/register"`) is a side-effect module, so it defeats tree-shaking of the GPU package by construction and pulls the algorithms barrel (which has no `sideEffects: false` declaration in `algorithms/package.json`) into every bundle that includes the register module. |
| SSR / Node | Global mutable state across requests; a registration in one test file leaks into the next unless every test resets it. |
| Testability | Worst of the four: tests must reset the singleton; two copies of `@graphty/algorithms` in one app (allowed by the format's brand-check rule) have two registries, so a registration silently applies to one copy only. |
| Release process | Unaffected mechanically, but the "register" entry adds a second public export path that has to be kept stable. |
| Never-falls-back rule | Worst fit: the CPU package now owns the decision "GPU registered but threw: what next?", which is exactly the fallback temptation the project rule forbids; the honest implementation re-throws, which surprises a caller who never wrote "gpu" anywhere. |

Rejected.

### 4.3 (c) Optional peer dependency + dynamic `import()` inside the CPU packages or the element

How: `@graphty/graphty-element` (or algorithms) lists
`@graphty/webgpu-graph-algorithms` under `peerDependenciesMeta.optional` and
does `await import("@graphty/webgpu-graph-algorithms")` behind a probe.
npm's semantics: "Npm will not automatically install optional peer
dependencies" (npm package.json docs, `peerDependenciesMeta`).

| Criterion | Assessment |
| --- | --- |
| Dependency direction | element -> GPU (optional peer). Fine for the element (it already does this for web-llm); WRONG for algorithms/layout: a library that reaches for its own accelerator has the same inversion problem as (b) for type imports, and it puts WebGPU knowledge into packages the design says have none. |
| Bundling / tree-shaking | Requires externalising the specifier in the library's vite config (as `vite.config.ts:39` does for web-llm) and requires the import to live in a module that is NOT statically reachable from the barrel; graphty-element learned that "Safari fails on dynamic imports of non-existent modules even before the import is called" (`src/ai/providers/index.ts:11-12`). Consumers' bundlers (Next, webpack) also warn on a bare dynamic specifier that is not installed unless it is declared external. |
| SSR / Node | Works when the package is installed; the probe must not touch `navigator` at module top level. In Node the element would additionally need to obtain a Dawn `GPU`, which is a native module the element must NOT depend on, so Node detection cannot live in the element at all. |
| Testability | Module mocking of `import()`; a test for "not installed" needs a resolver that fails on purpose. Doable but fiddly. |
| Release process | `nx release` versions the element independently; an optional peer on a 0.x GPU package is exactly the situation the format's rule 5 forbids for hard deps (design lines 3670-3680); an optional peer range is tolerable but must be widened by hand at every GPU major. Whether nx's `updateDependents: "auto"` treats `peerDependencies` as local dependencies was not verified (see Unverified). |
| Never-falls-back rule | Detection happens at construction (probe returns null -> no accelerator is created); never at run time. That is compatible with the rule provided the element's code path is "no accelerator -> CPU", not "accelerator threw -> CPU". |

Acceptable only as an OPTIONAL convenience layer in graphty-element
("auto" mode), never in algorithms or layout, and only behind the same
isolation the web-llm loader uses. Not the primary mechanism.

### 4.4 (d) graphty-element-only wiring

How: the element exposes `accelerator` (a property on the element, a
`GraphBehavior` config key, and a constructor option on `Graph`); adapters and
layout engines consult `this.graph.accelerator`; the element itself does not
know how to create one. The graphty APP (which owns its bundle and knows its
targets) imports the GPU package, probes, and sets `element.accelerator`.

| Criterion | Assessment |
| --- | --- |
| Dependency direction | element -> nothing new (it types the property against the `AlgorithmAccelerator & LayoutAccelerator` interfaces exported by the CPU packages it already depends on). App -> GPU. |
| Bundling / tree-shaking | The element bundle does not change size. The app decides between a static import (simplest, +GPU package bytes) and a code-split `import()` on probe success. |
| SSR / Node | The element never touches WebGPU; a Node test of the element with a Dawn-backed accelerator is just injection. |
| Testability | Element tests inject a fake; the adapter loop is tested once for CPU and once for the fake accelerator. |
| Release process | Unchanged. |
| Never-falls-back rule | Same as (a). |
| Limitation | Users of the raw CPU packages outside the element get no "detected" behaviour; they use (a) directly, which is what the design already prescribes. |

### 4.5 Recommendation

Adopt (a) as the contract and (d) as the wiring, with (c) as a later,
optional "auto" convenience inside the element only:

1. `@graphty/algorithms` and `@graphty/layout` OWN the accelerator
   interfaces (`AlgorithmAccelerator`, `LayoutAccelerator`) next to their
   `indexed.*` result types, and add ONE async dispatcher per package
   (`indexed.accelerated(acc)` returning an object whose methods are
   `pageRank(s, o): Promise<PageRankResult>` etc., each delegating to
   `acc.pageRank` when present and to the CPU `indexed.pageRank` otherwise).
   The sync `indexed.*` functions and the legacy facades never change.
   Reasons: no new runtime dependency anywhere; the return-type problem
   (finding 1) is solved by construction; the interface being owned by the
   CPU package means the GPU package is checked against it at compile time
   in its own CI without a runtime import.
2. The GPU package exports `GpuContext.create({ gpu, limits? })` (throws
   `E_NO_ADAPTER` / `E_NO_DEVICE`; never returns a CPU stand-in) and
   `GpuContext.probe({ gpu })` (returns `{ ok: boolean; reason?: string }`
   without creating a device). The accelerator object it returns
   implements both interfaces plus `release(snapshot)` and `dispose()`.
   "Detected" = the injector calls `probe()` then `create()`. The GPU package
   never reads `navigator.gpu` itself; the caller passes it (browser) or
   Dawn's `create([])` result (Node), which is what keeps one code base for
   both runtimes and keeps `webgpu` out of the runtime dependency list.
3. graphty-element gets an `accelerator: AlgorithmAccelerator &
   LayoutAccelerator | null` property; adapters call
   `indexed.accelerated(this.graph.accelerator).pageRank(s, o)`; the
   `snapshot-replaced` listener calls `accelerator?.release(previous)`;
   `LayoutEngine` subclasses for `forceatlas2` / `spring` ask
   `accelerator?.forceAtlas2?.(opts) ?? cpuSimulation(opts)` when a
   `LayoutSimulation` is created.
4. The graphty app does the detection (probe + `import()` + `create` +
   `element.accelerator = gpu`) and surfaces a "GPU acceleration: on/off"
   indicator. Optionally, later, the element adds `accelerator: "auto"`
   implemented with the web-llm isolation pattern (a separate module, the
   specifier externalised in vite, the peer optional); this is a
   convenience for third-party element users and is not on the critical
   path.
5. Reject (b) for the reasons in 4.2 (global state, inverted dependency,
   fallback temptation).

Interplay with the rule "never fall back": the only branch that chooses CPU
is `acc?.x === undefined` evaluated BEFORE any GPU work; a throw from a GPU
method propagates to `runAlgorithm`'s operation queue and is reported to
the user as a failed algorithm run. The element may offer "disable
accelerator" as a user action after a failure, which is a user decision,
not a fallback.

Timing versus the landing order (design 14.6): the interfaces and the
dispatcher land in A2 (algorithms) and L1 (layout); the element property in
E1; the GPU package's conformance against the interfaces in W1. Until A2
exists in the monorepo the GPU package develops against `@graphty/graph-format`
alone with its own CPU reference implementations in `test/` (small,
index-based, written from the design's Port 1-6 code) and switches its
differential tests to `indexed.*` at W1.

---

## 5. Result-shape contract the GPU package must honour

Rules from the design that the GPU results must follow, so that
graphty-element's single result-writing loop works for CPU and GPU alike:

- Index-aligned typed arrays, never id-keyed (design 10 and research note
  09 section 6): per-node `Float32Array(n)` / `Uint32Array(n)`, per-edge
  results folded from per-arc buffers with `foldArcs` (exported by
  `packages/graph-format/src/snapshot/views.ts:852`).
- "CPU scores are `Float64Array`, GPU scores `Float32Array`; parents, labels
  and matchings are `Uint32Array` with `INVALID_INDEX`" (design 14.1 rule 5,
  lines 3723-3726). The `AlgorithmAccelerator` result types therefore use
  `NumericVector` (`F32 | F64`) for scores so both satisfy the same
  interface; consumers that need `F64` call `Float64Array.from`.
- Labels are dense `0..count-1`; the GPU package calls `renumberPartition`
  (`packages/graph-format/src/snapshot/derived.ts:1155`) on the readback so
  `{ labels, count }` matches the CPU shape; ORDER of labels differs from
  the CPU's first-seen order unless the GPU package renumbers in first-seen
  order on the CPU (O(n), recommended so `groups()` output is identical).
- Differential tolerance for f32 scores: the design's `1e-9` tolerance
  (section 16.2) applies to the f64 CPU path only; GPU parity tests compare
  with a relative tolerance derived from iteration count (PageRank ~1e-5
  relative after 100 iterations) and compare RANK ORDER of the top-k.

Per-target result shapes (the GPU method returns exactly this; the
`*Pct` derivation and id mapping stay in the adapter):

| GPU method | Returns | Parity check against `indexed.*` |
| --- | --- | --- |
| `pageRank(s, o)` | `{ scores: F32(n); iterations; converged }` | rel. error <= 1e-5 per node; identical `converged` on tolerance-scaled check |
| `personalizedPageRank(s, personalization: F32(n), o)` | same | same |
| `hits(s, o)` | `{ hubs: F32(n); authorities: F32(n); iterations; converged }` | same |
| `eigenvectorCentrality(s, o)` / `katzCentrality(s, o)` | `F32(n)` (+ iterations) | same |
| `degree(s)` (skeleton only) | `U32(n)` | exact |
| `breadthFirstSearch(s, start, o)` | `{ parent: U32; depth: U32; visitedCount; order: U32 }` | `depth` exact; `parent[v]` any vertex with `depth[parent[v]] === depth[v]-1` and an arc to `v` (NOT byte-identical to the CPU FIFO parent); `order` grouped by level, not FIFO-identical |
| `directionOptimizedBfs` | same | same |
| `connectedComponents(s)` / `weaklyConnectedComponents(s)` | `{ labels: U32(n); count }` | set-equality of partitions; identical after first-seen renumbering |
| `sssp(s, source, o)` (Dijkstra / delta-stepping replacement) | `{ dist: F32(n) (Infinity = unreached); predArc: U32(n) }` | `dist` within 1e-5 relative; `predArc` any arc that attains `dist` (ties differ) |
| `bellmanFord(s, source, o)` | `{ dist: F32; predArc: U32; hasNegativeCycle }` | same; negative cycle flag exact |
| `betweennessCentrality(s, o)` | `F32(n)` (raw; normalised/halved per `directed` in the SAME convention as `indexed`) | rel. error <= 1e-4 (f32 accumulation over many sources); top-k order |
| `edgeBetweennessCentrality(s, o)` | `F32(edgeCount)` via `foldArcs` | same |
| `closenessCentrality(s, o)` | `F32(n)` | rel. 1e-5 (unweighted exact in integers before division) |
| `allPairsShortestPath(s)` / `floydWarshall(s)` | `{ dist: F32(n*n); pred?: U32(n*n) }`, n bounded by `maxBufferSize` | exact for unweighted; 1e-5 weighted |
| `labelPropagation(s, o)` | `{ labels: U32(n); iterations; converged }` | set-equality on synthetic graphs with planted partitions (LPA is nondeterministic on ties even on the CPU) |
| `kCoreDecomposition(s)` | `U32(n)` coreness | exact |
| `minimumSpanningTree(s)` (Boruvka) | `{ edges: U32; totalWeight }` | `totalWeight` within 1e-5; edge set identical when weights are distinct |
| `louvain(s, o)` (later) | `{ labels: U32; count; modularity; iterations }` | modularity within a band of the CPU's (not identical partitions) |
| Layouts: `forceAtlas2(o)`, `fruchtermanReingold(o)` | a `LayoutSimulation` writing the owner's stride-3 `F32` | no numeric parity (chaotic); tests assert energy decrease, settle, `setFixed`/`setPosition` semantics, and pin invariance |

---

## 6. Scoring model for the prioritised list

Score = value x speedup / risk.

- Value (1-5): 5 = has a graphty-element adapter with `suggestedStyles`
  and is a headline visual analytic (PageRank, betweenness, communities,
  the force layout); 4 = has an adapter and is a common building block;
  3 = adapter exists but rarely used at scale, or no adapter but a cheap
  add-on of a 4/5; 2 = programmatic users only; 1 = niche.
- Speedup potential at 100k-1M nodes (1-5): 5 = O(N^2) or many-sweep
  algorithm that maps to dense parallel kernels with no per-iteration
  readback; 4 = O(m) per iteration edge-parallel with tens of iterations;
  3 = frontier-shaped with load-balancing sensitivity; 2 = mostly
  sequential or readback-bound; 1 = O(n) and cheaper on the CPU than the
  upload.
- Risk (1-5): 1 = one kernel plus reductions, no atomics on floats, well
  known; 2 = atomics on u32 / label convergence; 3 = frontier queue,
  degree-tier load balancing, prefix scans; 4 = multi-phase with per-level
  contraction or nondeterministic tie handling; 5 = research-grade
  (SpGEMM, hash-based modularity gain on device).

WGSL constraint that drives several risk ratings: WGSL has atomic `i32`/`u32`
only, no float atomics (research note 03 section 12, first paragraph;
research note 09 section 2). Consequences: non-negative `f32` distances can
use `atomicMin` on their bit pattern (order-preserving for non-negative IEEE
floats); accumulations (PageRank push, betweenness dependencies) must be
PULL-shaped over `reverse()` or use fixed-point `i32` accumulation with a
CPU rescale; Bellman-Ford with negative weights needs a compare-and-swap
loop.

---

## 7. Prioritised targets

### 7.1 Layouts (first slice after the walking skeleton, per the owner)

| Rank | Target | Value | Speedup | Risk | Score | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | ForceAtlas2 as `LayoutSimulation` | 5 | 5 | 3 | 8.3 | graphty-element's flagship force layout; today allocates N x N x dim per iteration (note 03 row 7). Buffers per note 03 section 12.1: positions, undirected `rowPtr/colIdx`, `weights`, `mass = outDegree()+1` (design line 4019), `size`, force / swing / traction scratch, params uniform. Repulsion: tiled all-pairs in workgroup memory is exact and simplest (O(N^2) but 1M x 1M pairs per iteration is 10^12 -- too slow above ~100k), so a Barnes-Hut / grid approximation is REQUIRED for the 1M-node target; GraphWaGu demonstrates Barnes-Hut FR in WebGPU (its README names "Fruchterman-Reingold and Barnes-Hut algorithms"); cosmos.gl uses a many-body force with a "spatial-hash grid" for collisions in WebGL 2 (README). Recommend: uniform-grid / spatial-hash many-body approximation first (sort by cell key with the radix-sort primitive, per-cell centroid, far-field from cell centroids, near-field exact), quadtree/octree later. Per-frame `step()` with no readback except a periodic movement reduce for `settled`. |
| L2 | Fruchterman-Reingold / `spring` as `LayoutSimulation` | 3 | 5 | 2 | 7.5 | Two kernels per iteration (note 03 section 12.2); shares the many-body kernel with L1; simplest correctness test; `fixed` mask maps to `setFixed`. |
| L3 | "ngraph-like" spring-electrical defaults | 4 | 5 | 2 | 10 (but not a new kernel) | The element's default layout is ngraph (finding 4). Rather than porting ngraph, ship L2 with an option preset that mimics ngraph's spring length / repulsion constants and its settle heuristic so the element can route `ngraph` to the GPU simulation above a node-count threshold chosen by the app. Product decision for the element, listed here so the plan reserves it. |
| L4 | Spectral layout via Laplacian SpMV | 2 | 3 | 3 | 2 | Reuses the SpMV kernel; the CPU version's eigen-solver is known-suspect (note 03 section 12.5), so it is a correctness fix, not a parity port. |
| L5 | Kamada-Kawai (APSP on GPU + gradient kernel) | 2 | 4 | 3 | 2.7 | Bounded to n <= ~10k by the n x n matrix (note 03 section 12.4); depends on the APSP kernel (A9). |
| L6 | ARF | 1 | 4 | 2 | 2 | Same shape as FR (note 03 section 12.3); rarely used. |

### 7.2 Algorithms

| Rank | Target | Value | Speedup | Risk | Score | Primitives needed | Views bound (design 10.1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | PageRank (+ personalized) | 5 | 5 | 1 | 25 | SpMV pull kernel, two reductions (dangling mass, L1 delta), weighted-out-degree segmented reduce on device (design 10.1 last row) | `reverse().rowPtr/colIdx/weights`, `outDegree` computed from `rowPtr` |
| A2 | HITS, eigenvector, Katz | 3 | 5 | 1 | 15 | same SpMV + norm reduce | forward CSR and `reverse()` |
| A3 | Connected components (WCC) | 4 | 4 | 2 | 8 | edge-parallel hook + pointer-jump compress (Shiloach-Vishkin / Afforest style), `atomicMin` on u32 labels, `renumberPartition` on readback | `edgeList().src/.dst` (each edge once, correct for directed and undirected without special case) |
| A4 | BFS (single source; depth + parent) | 4 | 4 | 3 | 5.3 | Frontier (prefix-scan compaction), degree-tier advance (workgroup / subgroup / thread per vertex using `degreeOrder()` segments; the `subgroups` feature is present on the dev GPU per note 09 section 0), optional bottom-up phase via `reverse()` (Beamer switch, same alpha/beta as `direction-optimized-bfs.ts:49-50`). Merrill-Garland-Grimshaw's prefix-sum frontier expansion is the reference: "asymptotically optimal O(\|V\|+\|E\|) work complexity", "3.3 billion ... traversed edges per second" on one GPU (NVIDIA research page). | `rowPtr`, `colIdx`, `reverse()`, `degreeOrder()` |
| A5 | Closeness centrality | 4 | 4 | 3 | 5.3 | Batched multi-source BFS (bitmask frontier, 32 sources per u32 word) built on A4; weighted variant = batched A6 | as A4 |
| A6 | SSSP (Dijkstra replacement: delta-stepping or Bellman-Ford-style edge relax with `atomicMin` on non-negative f32 bit patterns) | 4 | 3 | 3 | 4 | Frontier + bucketing (delta-stepping) or simple iterate-to-fixpoint with an active mask; `predArc` written by the winning relax (CAS on (dist bits, arc) pair) | `rowPtr`, `colIdx`, `weights`, or `edgeList()` for the relax-all variant |
| A7 | Betweenness centrality (node + edge) | 5 | 5 | 4 | 6.3 | Brandes: per-source forward BFS recording `sigma` (u32 path counts, atomicAdd) and `depth`, then backward level-by-level dependency accumulation as a PULL over successors (no float atomics), batching several sources per dispatch; McLaughlin-Bader's hybrid chooses between "work-efficient" (active vertices only) and "edge-parallel" per iteration (CACM 2018 abstract via search result); sampling k sources gives the approximate variant graphty needs at 1M nodes (exact BC is O(nm) and is not interactive at that scale on any device). The Buffalo 2023-06 report is titled "Accelerating Betweenness Centrality on GPU" (its content could not be extracted, see Unverified). | as A4 plus `sigma`, `delta`, `depth` scratch |
| A8 | Label propagation | 3 | 4 | 3 | 4 | Per-node mode of neighbour labels: for rows sorted by target the labels are NOT sorted, so either a workgroup-local hash (degree <= ~1024) or a segmented sort of (label) per row (radix primitive) ; Jacobi-style synchronous update with a changed-count reduce | `rowPtr`, `colIdx`, `weights` |
| A9 | APSP / Floyd-Warshall (feeds KK, `allPairsShortestPath`, `floydWarshall`) | 2 | 5 | 2 | 5 | blocked FW (tiled, classic GPU kernel) for weighted; n batched BFS for unweighted; memory `n*n*4` bytes bounded by `maxBufferSize` (256 MiB default -> n <= 8192 unless raised limits) | `rowPtr`, `colIdx`, `weights` |
| A10 | k-core decomposition | 3 | 3 | 2 | 4.5 | parallel peeling: one dispatch per k level with a compaction of the peeled set; `outDegree().slice()` semantics honoured on the device copy | `rowPtr`, `colIdx` |
| A11 | Bellman-Ford (negative weights) | 3 | 4 | 2 | 6 | edge-parallel relax over `edgeList()` (both directions on undirected, design 10.1), `n-1` rounds max with early exit on a changed flag; negative-cycle detection = one more round; CAS loop for signed floats | `edgeList()` |
| A12 | MST (Boruvka) | 3 | 3 | 3 | 3 | per-component min edge via `atomicMin` on packed 64-bit (weight-bits << 32 \| edge) split across two u32 atomics or a two-pass scheme; union via A3's compress | `edgeList()` |
| A13 | Common neighbours / Adamic-Adar / triangle counting | 2 | 4 | 2 | 4 | edge-parallel sorted-row intersection (binary or merge), the same kernel as k-truss; not in the adapter set today | `rowPtr`, `colIdx`, `coo().src` |
| A14 | Louvain / Leiden | 5 | 3 | 5 | 3 | per-node best-community gain over neighbour communities (workgroup hash), synchronous colour-ordered or randomised updates, modularity reduce, then `snapshot.contract(partition)` on the CPU per level (design line 3792) and re-upload; cuGraph precedent exists but WGSL has no float atomics for the community-weight tables (fixed-point i32). Highest user value, highest risk: schedule after A1-A8 primitives exist. | `rowPtr`, `colIdx`, `weights`, `weightedDegree()` |
| A15 | SCC | 3 | 2 | 4 | 1.5 | forward-backward reachability with trimming; needs `reverse()`; label order differs from Tarjan's so `condensationGraph` parity is not achievable; low priority |
| A16 | Spectral clustering | 2 | 3 | 4 | 1.5 | Laplacian SpMV + Lanczos/LOBPCG + k-means; nvGRAPH-era precedent ("spectral and hierarchical clustering/partitioning techniques", NVIDIA cluster-analysis page); later |
| -- | Not GPU targets | -- | -- | -- | -- | DFS, topological sort, cycle detection, Prim, Girvan-Newman, hierarchical / TeraHAC / GRSBM / SynC, MCL, max-flow / min-cut, bipartite matching, isomorphism, A*: sequential, research-grade, or inherently small-graph. graphty-element's adapters for these keep the CPU path. |

Suggested vertical-slice order after the walking skeleton (device + upload
hot prefix + `degree` kernel + readback + `release`, in Node and browser):

1. L2 (FR) then L1 (FA2) as `LayoutSimulation` -- the owner's first need;
   they pull in the position-buffer lifecycle, `setFixed` / `setPosition`
   writes, the many-body approximation (radix sort + grid), and the reduce
   primitive.
2. A1 (PageRank) then A2 -- pull SpMV, segmented reduce, weighted
   normaliser; first algorithm parity tests against `indexed.pageRank`.
3. A3 (WCC) -- edge-parallel kernels over `edgeList()`, u32 atomics,
   `renumberPartition`.
4. A4 -> A5 -> A6 -> A7 -- the frontier family, ending in betweenness with
   sampling.
5. A8, A9, A10, A11, A12 as demand appears; A14 (Louvain) once 1-4 are
   stable.

---

## 8. What the GPU package's public surface must include for the recommended mechanism

- `GpuContext.probe({ gpu })` and `GpuContext.create({ gpu, requiredLimits? })`
  (throws; no fallback), `ctx.dispose()`.
- `ctx.accelerator(): AlgorithmAccelerator & LayoutAccelerator &
  { release(snapshot): void }` -- the object the app injects.
- One async function per algorithm target in section 7.2 with the indexed
  option types re-declared structurally (no runtime import of algorithms).
- One `LayoutSimulation` factory per layout target in section 7.1.
- Upload cache keyed on the typed-array object (`rowPtr` for a core, the
  view array for a view, the `gpuView()` array for a column), explicit
  `release(snapshot)` (design 14.5 lines 4226-4233).
- No `navigator` access, no `webgpu` runtime dependency, `@webgpu/types`
  as a devDependency; `noUncheckedIndexedAccess` off (design 14.5).

---

## Unverified

- The Buffalo tech report 2023-06 PDF was fetched (849 KB) but its text
  could not be extracted (no `pdftotext` / `pdftoppm` in this container); only
  its title "Accelerating Betweenness Centrality on GPU" is known from the
  search result. Its technique is not used in this note.
- The ACM DL page for McLaughlin and Bader (10.1145/3230485) returned 403;
  the abstract summary comes from the web search result snippet only.
- Whether nx release's `updateDependents: "auto"` treats a `workspace:*`
  entry under `peerDependencies` as a local dependency for version bumping
  was not verified in nx documentation.
- cosmos.gl's exact many-body algorithm (quadtree vs grid) is not stated in
  its README; only "Many-Body force implementation" (via the
  `EXT_float_blend` extension) and a "spatial-hash grid" for collision are
  quoted. GraphWaGu's README names Barnes-Hut but gives no scale numbers.
- `jaredmcqueen/analytics` claims "60 FPS simulation for scenes with 1
  million nodes" for a WebGL FR simulation; not independently measured.
- The subgroup size on the dev GPU is asserted as 32 in research note 09
  (section 3) from NVIDIA convention; the WebGPU `subgroups` feature does
  not guarantee a fixed size, so kernels must read `subgroup_size`.

---

## Sources

Local (paths absolute; line numbers as of 2026-09-14):

- `/home/apowers/Projects/graphty-monorepo/algorithms/src/index.ts`,
  `algorithms/src/algorithms/index.ts` and the category barrels under
  `algorithms/src/algorithms/*/index.ts`, `algorithms/src/research/index.ts`,
  `algorithms/src/optimized/index.ts`, `algorithms/src/types/index.ts`
- `/home/apowers/Projects/graphty-monorepo/algorithms/src/optimized/csr-graph.ts`,
  `graph-adapter.ts`, `direction-optimized-bfs.ts`;
  `algorithms/src/algorithms/traversal/bfs-unified.ts`
- `/home/apowers/Projects/graphty-monorepo/algorithms/package.json`,
  `layout/package.json`, `graphty-element/package.json`, `nx.json`,
  `pnpm-workspace.yaml`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`
- `/home/apowers/Projects/graphty-monorepo/graphty-element/src/algorithms/Algorithm.ts`,
  `index.ts`, `PageRankAlgorithm.ts`, `BetweennessCentralityAlgorithm.ts`,
  `LouvainAlgorithm.ts`, `DijkstraAlgorithm.ts`, `utils/graphConverter.ts`;
  `graphty-element/src/managers/AlgorithmManager.ts`, `managers/LayoutManager.ts`;
  `graphty-element/src/Graph.ts`; `graphty-element/src/layout/LayoutEngine.ts`,
  `ForceAtlas2LayoutEngine.ts`, `NGraphLayoutEngine.ts`;
  `graphty-element/src/config/GraphBehavior.ts`;
  `graphty-element/src/ai/providers/index.ts`, `WebLlmProvider.ts`;
  `graphty-element/vite.config.ts`
- `/home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/*.ts`
- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
  (lines 60-80, 2320-2340, 3590-3727, 3728-3958, 3959-4047, 4110-4130,
  4170-4190, 4212-4278)
- `/home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/03-layout-needs.md`
  (sections 4, 8.1, 9, 12), `04-graphty-element-usage.md` (section 4.3),
  `09-webgpu-requirements.md` (sections 0, 3, 6, 7.2)
- `/home/apowers/Projects/webgpu-graph-algorithms/package.json`, `src/index.ts`,
  `src/types/index.ts`, `packages/graph-format/src/index.ts`,
  `packages/graph-format/src/snapshot/views.ts`, `derived.ts`,
  `packages/graph-format/test/audit/gpu-upload.test.ts`,
  `packages/graph-format/package.json`
- Clone: `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`
  (self-hosted runner label `cudaffi-gpu-runner`, container
  `ghcr.io/apowers313/roc-dev:1.5.2` with `--gpus all`; CI detail belongs to
  the CI note, recorded here only as verified)

External:

- cosmos.gl: https://github.com/cosmosgl/cosmos (WebGL 2 via luma.gl;
  "Many-Body force implementation"; "spatial-hash grid" collision; typed
  array `setPointPositions` / `setLinks`; "hundreds of thousands of points
  and links")
- Cosmograph examples: https://cosmograph.app/examples ; Python package:
  https://pypi.org/project/cosmograph/ (owner-supplied; not fetched)
- GraphWaGu: https://github.com/harp-lab/GraphWaGu (WebGPU;
  "Fruchterman-Reingold and Barnes-Hut algorithms"; paper "GraphWaGu: GPU
  Powered Large Scale Graph Layout Computation and Rendering for the Web")
- jaredmcqueen/analytics: https://github.com/jaredmcqueen/analytics
  ("fruchterman reingold force-directed simulation, all performed on the
  GPU"; "60 FPS simulation for scenes with 1 million nodes")
- Merrill, Garland, Grimshaw, "High Performance and Scalable GPU Graph
  Traversal", UVA TR CS-2011-05:
  https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal
- McLaughlin and Bader, "Accelerating GPU betweenness centrality", CACM
  61(8), 2018: https://dl.acm.org/doi/10.1145/3230485 (403 on fetch);
  https://cacm.acm.org/research/accelerating-gpu-betweenness-centrality/
- University at Buffalo CSE tech report 2023-06, "Accelerating Betweenness
  Centrality on GPU": https://cse.buffalo.edu/tech-reports/2023-06.pdf
  (text not extractable here)
- NVIDIA cluster analysis: https://developer.nvidia.com/discover/cluster-analysis
  (nvGRAPH "spectral and hierarchical clustering/partitioning techniques")
- npm `peerDependenciesMeta`: https://docs.npmjs.com/cli/v10/configuring-npm/package-json
  ("Npm will not automatically install optional peer dependencies")
- atoms-org/cuda-ffi: https://github.com/atoms-org/cuda-ffi
