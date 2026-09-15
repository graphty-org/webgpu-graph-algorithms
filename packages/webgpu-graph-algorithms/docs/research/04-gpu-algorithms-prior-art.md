# 04 - GPU graph algorithm prior art, translated to WebGPU

Date: 2026-09-14
Scope: what NVIDIA/cuGraph, Gunrock, GAP and the papers the owner supplied
do for each graph algorithm on a GPU, and what each of those strategies
becomes under WebGPU/WGSL constraints and the @graphty/graph-format GPU
contract (design doc section 10). This note feeds the WebGPU package plan;
it is not the plan itself.

Everything stated about external code was read from a local clone under
/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/
(paths below are relative to that directory) or from a PDF converted to
text under tmp/webgpu-plan/papers/. Things that could not be fetched are
listed in section 12.

Local inputs read:

- /home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md
  lines 2320-2553 (section 10, GPU contract)
- /home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/09-webgpu-requirements.md
  (probed 4070 SUPER limits, WGSL type rules, per-algorithm view table)
- /home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/08-prior-art.md
  sections 4 (cuGraph), 5 (Gunrock), 11.10 (WebGPU constraints)
- /home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/03-layout-needs.md
  sections 4 and 12 (layout buffers, first GPU targets)

## 0. One-page summary

1. Every GPU graph library surveyed is built on the same five primitives:
   exclusive scan, segmented reduce over rowPtr, stream compaction,
   radix sort (or sort-by-key), and a load-balanced "advance" that expands
   a vertex frontier into its neighbours (Merrill 2011 scan+warp+CTA;
   Gunrock block_mapped / merge_path_v2; Davidson 2014 load-balanced
   partitioning). Build those once, well, and every algorithm below is a
   few small kernels around them.
2. The CUDA techniques that do NOT translate to WebGPU and need a
   replacement pattern are: float atomicAdd (Gunrock BC/PR/TC, cuGraph
   FA2 attraction), spin-lock/CAS tree insertion with inter-block fences
   (Burtscher-Pingali Barnes-Hut in cuGraph FA2), warp-synchronous
   voting/shuffles assumed at width 32, dynamic parallelism, and
   kernel-side `atomicAdd` on a global queue counter for frontier
   enqueue (works in WGSL, but the returned offset is per-workgroup so
   the compaction must be workgroup-granular; see 2.4).
3. Gather ("pull") formulations avoid float atomics entirely and are the
   default for PageRank/Katz/eigenvector/HITS (cuGraph already runs these
   on the transposed graph: `pull_graph_view`,
   cugraph-algos/cpp/src/link_analysis/pagerank_impl.cuh:220-290), for
   Brandes dependency accumulation (McLaughlin-Bader Algorithm 2 checks
   successors "to eliminate the use of atomics"), and for force-directed
   attraction (per-row gather over the symmetric CSR).
4. cuGraph's direction-optimizing BFS constants are verified in source:
   alpha = average degree (times 0.267 on multi-GPU), beta = 24, and it
   switches back to top-down when `next_frontier * 24 < unvisited` and the
   frontier is shrinking (cugraph-algos/cpp/src/traversal/bfs_impl.cuh:291-297,
   637-638, 843-846). Beamer SC12 uses alpha = 14, beta = 24 with
   `m_f > m_u / alpha` and `n_f < n / beta`.
5. cuGraph SSSP is Davidson et al. 2014 near-far with a two-level near
   queue; delta = 32 * avg_weight / avg_degree, 16 subpartitions, near
   queue capped at SMs * 2048 / avg_degree (sssp_impl.cuh:189-262).
6. cuGraph betweenness batches up to 65,535 sources into one tagged
   multi-source BFS with n x sources 2D sigma/distance arrays, capped at
   25 % of device memory (betweenness_centrality_impl.cuh:660-700,
   1380-1400). Dependency accumulation is a pull over successors with a
   plus-reduce, no atomics (lines 348-368).
7. cuGraph Louvain is synchronous: every vertex computes its best
   neighbour community in one pass, the "up_down" flag lets a vertex move
   only to a higher-id (or only lower-id) community in alternating passes
   to prevent swaps, then contraction builds the coarse graph
   (louvain_impl.cuh:172-215; detail/common_methods.cuh:120-152, 402-446).
   nu-Louvain calls the same trick "Pick-Less every 4 iterations"
   (arXiv 2501.19004).
8. The Buffalo tech report 2023-06 is an MS thesis (Utkarsh Kumar,
   "Accelerating Betweenness Centrality on GPU") on DENSE adjacency-matrix
   BC via cuBLAS and a Katz-walk approximation; it only beats McLaughlin-
   Bader at 50-75 % density and admits accuracy problems. Not applicable to
   sparse graphs; noted for completeness only.
9. The NVIDIA "cluster analysis" page is a short nvGRAPH-era overview of
   spectral (Laplacian eigenvectors) and multilevel hierarchical
   (coarsen + Kernighan-Lin refine) graph partitioning with modularity /
   balanced-cut / flow metrics. It confirms the spectral and multilevel
   families but carries no kernel-level detail.
10. Existing WebGPU graph work is thin: GraphWaGu (FR + Barnes-Hut layout,
    WGSL radix sort + Morton-code cluster tree, i32 fixed-point atomics for
    the bounding box) is the only substantive one; @antv/webgpu-graph is a
    dense-matrix PageRank and all-vertex Bellman-Ford with per-iteration
    readback in pre-1.0 WGSL. There is no WebGPU BFS/CC/BC/Louvain library
    to reuse; the owner's package would be first.

## 1. CUDA -> WebGPU constraint translation

| CUDA/prior-art concept | WebGPU/WGSL reality (probed on 4070 SUPER, 09-webgpu-requirements.md section 0) | Pattern to use |
| --- | --- | --- |
| warp (32 lanes), `__shfl`, `__ballot`, warp vote | optional `subgroups` feature: subgroupAdd/Min/Max/Ballot/Broadcast/Shuffle/Elect, `subgroup_size` builtin, size NOT guaranteed constant (WGSL 17.12, adapter minSubgroupSize/maxSubgroupSize) | write every kernel with workgroup-memory fallbacks; specialise with `override USE_SUBGROUPS` when the feature is present; never assume width 32 |
| CTA / thread block + `__shared__` (48 KB) + `__syncthreads` | workgroup size <= 256 invocations at core defaults (1024 on the adapter), 16 KB workgroup storage default (48 KB adapter) | tile size 256; scan of 256 degrees in workgroup memory; keep per-workgroup scratch <= 16 KB unless the device was created with the raised limit |
| `atomicAdd(float*)` (Gunrock BC deltas, PR, TC counts; cuGraph FA2 attraction) | WGSL `atomic<T>` only for `u32` / `i32`, only in `storage, read_write` or `workgroup` (WGSL 6.2) | (a) gather/pull formulation so each output is owned by one invocation; (b) i32 fixed-point accumulation (GraphWaGu bounding box: `atomicMin(&bounding.x_min, i32(floor(x * 1000.0)))`, GraphWaGu/src/wgsl/apply_forces.wgsl:79-82); (c) CAS loop on `bitcast<u32>(f32)` for rare updates (correct but slow under contention); (d) per-workgroup partial sums + second-pass reduce |
| `atomicMin(float*)` (SSSP relax) | none for f32 | non-negative f32 has the same ordering as its `u32` bit pattern, so `atomicMin(&dist_u32[v], bitcast<u32>(d))` is exact for d >= 0 (sign bit clear). Requires `flags.nonNegativeWeights` (section 10.5), which delta-stepping needs anyway |
| `atomicCAS` | `atomicCompareExchangeWeak` on u32/i32 | fine for BFS visited claim, Afforest hook, queue slot claim |
| global queue counter `atomicAdd(&len, count)` returning a base offset (Merrill 5.1 step iv, Gunrock block_mapped.hxx:137) | works: one invocation per workgroup does `atomicAdd` on a storage atomic and broadcasts via workgroup memory | keep it (workgroup-granular allocation); output order is non-deterministic just as in CUDA |
| dynamic parallelism / device-side launch; host loop with `cudaMemcpy` of a scalar per iteration | no device-side launch; host readback is `mapAsync` (a full round trip, hundreds of us) | run k iterations per submit; check convergence every k iterations; use `dispatchWorkgroupsIndirect(buffer)` with the frontier length written by the previous kernel so the next expansion needs no readback |
| grid size up to 2^31 blocks | 1D dispatch <= 65,535 workgroups, i.e. 16,776,960 invocations at size 256 (design doc 10.6) | 2D grid or grid-stride loop for n or arcCount above that; the DispatchPlanner owns this |
| unlimited kernel arguments | 8 storage buffers per stage default (10 on the adapter), 4 bind groups, uniform bindings 64 KB | arena sub-range bindings (rowPtr/colIdx/weights in one buffer, section 10.3); split cold arrays into a second bind group; scalar tier bounds as uniforms |
| 64-bit indices / `double` accumulation | u32 / f32 only; no f64, no u64 | counts <= 0xFFFFFFFE (I3); sigma path counts in BC as f32 (cuGraph uses edge_t integers but overflows the same way on big graphs) |
| inter-block spin locks and `__threadfence` (Burtscher-Pingali tree build) | no forward-progress guarantee across workgroups is documented for WebGPU; a workgroup spinning on another workgroup's write can hang the device | level-by-level dispatches instead (GraphWaGu create_tree.wgsl: one dispatch per tree level); never spin across workgroups |
| `cudaMalloc` per iteration (cuGraph frontier buffers resize freely) | buffer creation is cheap-ish but not free; mapping is async | preallocate 2 x n-slot vertex queues and, when using an edge frontier, an arcCount-slot queue; BufferPool by size class |
| texture cache for the bitmask (Merrill 4.2) | no read-only texture path worth using for u32 bitsets | plain storage bitset; the visited bitset is 1 bit/vertex either way |

## 2. Primitive catalogue (what the algorithms below consume)

### 2.1 Scan (exclusive prefix sum)

Every frontier-based algorithm uses two scans per iteration: one over
degrees to lay out the edge frontier, one over validity flags to compact
the next vertex frontier (Merrill 2011 section 5.1 steps iii-v; Gunrock
block_mapped.hxx:123 `block_scan_t(scan).ExclusiveSum(th_deg, th_deg,
aggregate_degree_per_block)`). WebGPU: workgroup-level scan of 256 items in
workgroup memory (or subgroupExclusiveAdd + one cross-subgroup pass when
available), device-level scan as reduce-then-scan (3 dispatches: block
sums, scan of block sums, add back) or single-pass decoupled look-back
(needs storage atomics with acquire/release; WGSL atomics are relaxed
only, so decoupled look-back must poll with `atomicLoad` and cannot rely
on ordering of neighbouring non-atomic stores -- prefer reduce-then-scan
in v1, revisit with subgroups).

### 2.2 Segmented reduce over rowPtr (CSR SpMV shape)

Used by: weighted out-degree normaliser (design doc 10.1 explicitly
delegates it to the GPU package), pull PageRank/Katz/HITS, Brandes
dependency accumulation, per-row best-community in Louvain, closeness
partial sums. Per-row gather with the degree-tier dispatch (thread /
subgroup / workgroup per row from `degreeOrder().segmentOffsets`,
thresholds 1024 / 32 / 1 copied from cuGraph graph_view.hpp, cited in
09-webgpu-requirements.md section 3) is the WebGPU form of Gunrock's
`neighborreduce` operator (gunrock/include/gunrock/framework/operators/
neighborreduce/neighborreduce.hxx) and of cuGraph's
`per_v_transform_reduce_incoming_e` (cugraph-algos/cpp/include/cugraph/
prims/per_v_transform_reduce_incoming_outgoing_e.cuh). Merge-path
(Gunrock merge_path_v2.hxx:40-56 and 111, Merrill-Garland SpMV) gives perfect
balance without tiers but needs a binary search per tile over rowPtr; it
is a v2 optimisation.

### 2.3 Stream compaction and dedupe

Compaction = flag + scan + scatter. Dedupe of a vertex frontier without
sorting: Davidson 2014 "ownership" trick -- each queue entry writes its
queue index into `owner[v]`, then reads it back; the entry whose index
survives is the unique owner, all others are compacted away
(papers/davidson2014.txt lines 614-625). This is one plain store + one
load per entry and needs no atomics (last-writer-wins is fine). Merrill's
warp-cull / history-cull heuristics (papers/merrill2011-bfs.txt Algorithm 7,
section 4.3: hash into a 128-entry per-warp scratch, then vie with the
thread id) map to workgroup memory hashing; they are "best effort", and
the exact dedupe above is the safety net that the CUDA papers get from
atomic CAS on the label.

### 2.4 Frontier (vertex queue and edge queue)

Merrill's four couplings (papers/merrill2011-bfs.txt section 5):
expand-contract (in-core edge frontier, 2n storage, 5n+2m traffic),
contract-expand (out-of-core edge queue, 2m storage, 3n+4m traffic, best
for "small, fleeting" iterations), two-phase (n+m storage, 5n+4m traffic,
best bulk throughput), hybrid (contract-expand when the edge queue is
smaller than the resident thread count, else two-phase). WebGPU
recommendation: two-phase as the workhorse (separate expand kernel and
contract kernel; each is a scan + scatter) plus a fused
expand-contract kernel for tiny frontiers so a long-tail road-network
BFS does not pay two dispatches and two scans per level. Frontier length
lives in a 4-byte storage atomic that doubles as the
`dispatchWorkgroupsIndirect` argument after a tiny "count -> workgroups"
kernel (ceil(len/256), clamped to 65,535 with a second dimension).

### 2.5 Advance with load balancing

Merrill's scan+warp+CTA (Algorithm 5 and 6, section 4.1): rows with
degree >= CTA width are strip-mined by the whole CTA, rows >= warp width
by a warp, the "loose ends" packed by a CTA-wide scan so "no SIMD lanes
are unutilized during global reads from C". Serial per-thread gathering
had "terrible coalescing"; warp-only gathering wasted lanes on graphs
with average degree <= 10; scan-only suffered when one thread held a huge
row. Davidson 2014 confirms (papers/davidson2014.txt lines 500-580):
CTA+Warp+Scan is poor on medium-degree graphs, load-balanced partitioning
(sorted search of block boundaries into the scanned degree array) wins on
high-degree graphs. Gunrock block_mapped (block_mapped.hxx:123-165):
scan the tile's degrees in shared memory, then every thread loops
`for i in local_idx .. aggregate_degree step blockDim` and binary-searches
the scanned degrees to find its source vertex -- this is the simplest
form that is already balanced within a workgroup, and it is the
recommended v1 advance for WebGPU (256-entry workgroup scan + `upper_bound`
over workgroup memory). Add the workgroup-per-row tier for rows longer
than the tile (degree >= 1024, from `segmentOffsets`) and the
subgroup-per-row tier only when `subgroups` is present.

### 2.6 Radix sort / sort-by-key

Needed by: Louvain contraction (sort arcs by (community(src),
community(dst)) then segmented reduce), triangle counting on unsorted
inputs (not needed: `flags.sortedRows` is a format invariant, section
10.5), Barnes-Hut Morton ordering (GraphWaGu/src/webgpu/sort.ts, 8 bits per
pass, 4 passes for 32-bit keys, re-implemented from Fuchsia's Vulkan radix
sort), degree ordering when `degreeOrder()` is not precomputed (the format
does it on the CPU; keep it there). A 4-pass 8-bit LSD radix sort
(histogram + scan + scatter per pass) is the v1 target; onesweep-style
single-pass sorts need the same decoupled look-back caveat as 2.1.

### 2.7 Histogram / count-by-key and on-device COO -> CSR

Counting sort by source: histogram of `src` (atomicAdd u32 per arc),
exclusive scan, scatter with a per-row atomic cursor. cuGraph and Gunrock
both build CSR from COO this way (gunrock/include/gunrock/formats/csr.hxx
`from_coo`: count per row, exclusive scan, scatter; 08-prior-art.md
section 5). On WebGPU this is the Louvain coarse-graph builder and the
residual-graph builder; the initial graph never needs it because the
format ships CSR.

### 2.8 Bitset

Visited / frontier-membership bitset as `array<u32>` with `atomicOr`
(cuGraph bfs_impl.cuh:774 `word.fetch_or(packed_bool_mask(v_offset))`),
plus the non-atomic bulk path cuGraph takes when the new frontier is
>= 40 % of the vertices (bfs_impl.cuh:729-765: one thread per word,
binary search into the sorted frontier). Beamer's bottom-up step needs
"a frontier bitmap to allow a constant-time test" (papers/beamer2012.txt
lines 432-441).

## 3. Breadth-first search

Strategy (Merrill 2011; Beamer 2012; cuGraph bfs_impl.cuh):

- Top-down, level-synchronous, work-efficient O(n+m): expand the vertex
  frontier through CSR rows (advance), claim unvisited neighbours,
  compact into the next frontier. Merrill reports 3.3 GTEPS single GPU,
  7-29x over sequential CPU, and that quadratic (all-vertex-per-level)
  kernels are 2,300x slower on europe.osm (papers/merrill2011-bfs.txt
  section 5.5). Quadratic per-level kernels (Harish et al. style, and
  @antv/webgpu-graph's SSSP) are only acceptable as a debugging baseline.
- Direction-optimizing (Beamer): switch to bottom-up when
  `m_f > m_u / alpha` and the frontier is growing (alpha = 14); switch back
  when `n_f < n / beta` and shrinking (beta = 24); convert the frontier
  queue <-> bitmap at the switch (papers/beamer2012.txt lines 396-441).
  Speedups 3.3-7.8x on Kronecker/RMAT and 2.4-4.6x on real social graphs;
  no benefit on high-diameter meshes/roads. cuGraph's version:
  alpha = m/n (avg degree, x0.267 multi-GPU), beta = 24, tracks
  `m_f` as the sum of frontier degrees and `m_u` as approximate degrees
  of unvisited vertices per degree segment (bfs_impl.cuh:486-560), keeps
  the list of non-zero-degree unvisited vertices (`nzd_unvisited_vertices`)
  and bottom-up iterates over that list, not all n (bfs_impl.cuh:640-660,
  765-790). Bottom-up requires the in-adjacency (`reverse()`), which for
  undirected snapshots is the forward CSR (section 10.1) -- so
  direction-optimizing BFS costs nothing extra on undirected graphs and
  4n+8m bytes of reverse view on directed ones.

Primitives: advance (2.5), compaction with dedupe (2.3), bitset (2.8),
scan (2.1), indirect dispatch.

Format views: `rowPtr`, `colIdx`; `reverse()` for bottom-up on directed
graphs; `degreeOrder()` for tiering and for the `m_u` estimate (cuGraph
computes approximate degrees for the high/mid segments only,
bfs_impl.cuh:320-330).

WebGPU adjustments:

- Visited claim: `atomicCompareExchangeWeak(&dist[v], INVALID_INDEX,
  level)` (Gunrock uses `atomic::min(&distances[neighbor], iteration+1)`
  and keeps the neighbour iff the old value was larger,
  gunrock/include/gunrock/algorithms/bfs.hxx:125-127; either works with
  u32 atomics). Predecessor labelling: write `parent[v] = u` only from the
  invocation whose CAS won; Merrill notes predecessor variants cost up to
  19 % more traffic (Table 2 discussion).
- Duplicates in the edge frontier: local warp/history culling becomes
  workgroup-memory hashing (128-entry scratch per subgroup or per
  workgroup); the CAS on the label is what makes correctness independent
  of the heuristics.
- Tiny frontiers: run the fused expand-contract kernel and skip the
  bitmask lookup, as Merrill does "for fleeting iterations having
  edge-frontiers smaller than the number of resident threads"
  (section 4.2). Batch several levels per `queue.submit` with indirect
  dispatch; read the frontier length back only every k levels (k ~ 8-32)
  or on a timestamp budget.
- 1D dispatch limit: expansion over an edge frontier of > 16,776,960
  entries needs the 2D grid; the planner handles it.

Complexity: O(n+m) work, O(diameter) dispatch rounds; expected 10-30x over
the CPU Map-of-Maps BFS at 1M edges once the graph is resident, but the
whole win is lost if each level pays a mapAsync round trip on a
high-diameter graph (europe.osm has ~19,000 levels, Merrill Table 1).

## 4. Single-source shortest paths (weighted)

Strategy: Davidson-Baxter-Garland-Owens 2014 near-far (cuGraph's
documented basis, sssp_impl.cuh:189-194). Bellman-Ford over all vertices
is the baseline; Workfront Sweep prunes to an active-vertex queue with the
ownership dedupe; Near-Far adds a priority threshold: process only queue
entries with `dist < (i+1) * delta` (near pile), defer the rest (far
pile); when the near pile empties, raise i, compact invalid/duplicate far
entries, split again (papers/davidson2014.txt lines 638-665, 719-740).
delta = c * w / d with c = warp width 32, w = average edge weight, d =
average degree, after Meyer-Sanders delta = Theta(1/d). Results: up to
14x over GPU Bellman-Ford on low-degree graphs, 340x on scale-free, 20-60x
over serial CPU on dense graphs (abstract). cuGraph refinement: a
two-level near queue (near-near / near-far, 16 subpartitions) so the
processed queue stays "just large enough to saturate GPU resources"
(sssp_impl.cuh:192-194, 246-262: cap = SMs * 2048 / avg_degree).

Primitives: advance, compaction/dedupe, scan, `atomicMin` on distance,
histogram of far-pile entries by subpartition (cuGraph
compute_new_near_near_partition_range, sssp_impl.cuh:85-161).

Format views: `rowPtr`, `colIdx`, `weights` (arc-aligned; `flags.weighted`,
`flags.nonNegativeWeights` required, `flags.allWeightsOne` -> run BFS
instead, section 10.5). Predecessors need the winning arc's source.

WebGPU adjustments:

- Distance array as `array<atomic<u32>>` holding f32 bit patterns;
  `atomicMin` is exact for non-negative floats (section 1 table).
  +Infinity (0x7F800000) sorts above every finite value, so
  `INVALID = 0x7F800000` doubles as "unreached". The result buffer is
  reinterpreted as `Float32Array` on readback with no conversion.
- Predecessor consistency: Gunrock SSSP does not record predecessors;
  cuGraph reduces `(dist, pred)` tuples with `reduce_op::minimum` per dst
  (sssp_impl.cuh:334). On WebGPU: after the relax kernel settles a level,
  a second pass over the same edge frontier writes `pred[v] = u` where
  `dist[u] + w == dist[v]` with `atomicMin(&pred[v], u)` for determinism
  (this is what @antv/webgpu-graph's updatePred kernel does,
  antv-webgpu-graph/package/es/traversal/sssp.js:51). Two passes over the
  frontier instead of a 64-bit packed atomic.
- The near/far split needs the CPU to know when the near pile is empty;
  use the same k-iterations-per-submit + indirect dispatch pattern as BFS,
  with a device-side "near empty" flag that a tiny kernel turns into a
  zero indirect dispatch so extra queued iterations are no-ops.
- Integer weights (`u32` column) can use a plain integer `atomicMin`;
  offer both entry points.

Complexity: near-far is O(m) relaxations amortised, with O(D/delta) rounds;
expect the same order of speedup as BFS on road-like graphs and larger
on scale-free graphs.

## 5. PageRank, Katz, eigenvector centrality, HITS (pull SpMV family)

Strategy: all four are power iterations with a per-vertex gather over
in-neighbours plus a global reduction for convergence. cuGraph runs them
on the transposed graph (`pull_graph_view`) with
`per_v_transform_reduce_incoming_e` and a `plus` reduce; PageRank
per iteration (pagerank_impl.cuh:222-320): copy old ranks, dangling sum =
`transform_reduce_v` of ranks where out_weight_sum == 0, ranks /=
out_weight_sum (guarded), gather `alpha * src_rank * w` +
`(dangling_sum * alpha + (1 - alpha)) / n`, personalization scatter,
`diff_sum` L1 reduce, stop when `diff_sum < epsilon` or max_iterations.
Katz (katz_centrality_impl.cuh:101-152) is the same gather with
`alpha * src * w + beta`; eigenvector (eigenvector_centrality_impl.cuh:
93-148) adds an L2 normalise each iteration and converges on
`diff_sum < n * epsilon`; HITS (hits_impl.cuh:87-93 onward) alternates a
pull for authorities and a pull for hubs and normalises by sum.
Gunrock's PR (gunrock/include/gunrock/algorithms/pr.hxx:145) instead
pushes with `atomic::add(p + dst, update)` -- the float-atomic form that
WebGPU cannot use.

Primitives: segmented reduce over reverse rowPtr (2.2), device reduce
(sum / L1 / L2 / max), elementwise map. No frontier, no sort.

Format views: `reverse().rowPtr / colIdx / weights` (identity arrays for
undirected snapshots); weighted out-degree normaliser computed on device
by a segmented reduce over forward `rowPtr` / `weights` (design doc 10.1:
"NOT a format upload"); `outDegree()` optional; personalization vector
as a node column via `gpuView()`.

WebGPU adjustments:

- Pull formulation only; one invocation owns `rankOut[v]`; no atomics.
  Degree-tiered dispatch by IN-degree (`degreeOrder({ of: "reverse" })`,
  design doc 10.1) so hub vertices with millions of in-arcs go to a
  workgroup-per-row kernel.
- Two rank buffers ping-pong; the dangling sum and diff sum are
  workgroup partial sums (256 per workgroup) reduced by a second tiny
  kernel into a 16-byte result that the next iteration's uniform reads
  via `copyBufferToBuffer` -- no host readback inside the loop. Check
  convergence on the host every k iterations (k = 4-8; PageRank typically
  needs 20-60).
- f32 accumulation: cuGraph and NetworkX use f64; the CPU package returns
  Float64Array (design doc 10.7). Document the f32 tolerance (epsilon
  >= 1e-6 * n is meaningful, tighter is noise) and reduce partial sums in
  a tree to limit error; Kahan in the per-row loop is cheap for hubs.
- Bindings: revRowPtr, revColIdx, revWeights, outWeightSum, rankIn,
  rankOut, personalization, partials = 8 -> exactly the default budget;
  put the uniform in the same group (uniforms are a separate limit) and
  drop `personalization` into a second bind group when absent.

Complexity: O(m) per iteration; the 100k/1M benchmark graph is ~8 MB of
CSR, so each iteration is bandwidth-trivial (< 0.1 ms) and the loop is
dispatch-latency-bound; batching iterations matters more than kernel
tuning here.

## 6. Betweenness centrality (Brandes)

Strategy:

- McLaughlin-Bader (CACM 2018 / SC14): work-efficient forward pass with
  explicit `Qcurr`/`Qnext` queues, `atomicCAS(d[w], inf, d[v]+1)` to
  enqueue each vertex once (so Qnext is O(n) not O(m)), `atomicAdd(sigma[w],
  sigma[v])` for path counts, and an `S`/`ends` array recording the
  vertices of each level contiguously "analogous to CSR"
  (papers/mclaughlin-bader-2018.txt Algorithm 1). Dependency accumulation
  (Algorithm 2) processes level d from `ends[d]..ends[d+1]` and has each
  vertex w sum over its successors v with `d[v] == d[w]+1`:
  `delta[w] += sigma[w]/sigma[v] * (1 + delta[v])` -- "we are able to
  eliminate the use of atomics by checking successors rather than the
  predecessors". The hybrid picks edge-parallel (all m edges every level,
  best memory throughput, wins on small-diameter scale-free graphs) vs
  work-efficient (wins ~10x on roads/meshes) online: run the work-efficient
  method on a few sample sources, take the median BFS depth, and switch to
  edge-parallel if `median < gamma * log2(n)` (Algorithm 3). Results:
  2.71x average over the prior best GPU code, up to 13x on high-diameter
  graphs, near-linear scaling to 192 GPUs by distributing sources.
- cuGraph (betweenness_centrality_impl.cuh): forward pass is a frontier
  BFS whose per-dst reduce is `plus` on sigma
  (`transform_reduce_if_v_frontier_outgoing_e_by_dst`, lines 184-193) so
  sigma needs no atomics either -- the reduce-by-key over the edge
  frontier does it; backward pass sorts vertices by distance once and
  walks levels from the diameter down with `per_v_transform_reduce_outgoing_e`
  (lines 283-368). Single-GPU it runs a TAGGED multi-source BFS: the
  frontier holds (vertex, source_idx u16) pairs and sigma/distance are
  `num_sources x n` 2D arrays, batch size capped so those arrays use at
  most 25 % of device memory and at most 65,535 sources (lines 660-700,
  1380-1400). Approximate BC = a subset of sources; both papers say the
  exact algorithm "can be trivially adjusted for approximation"
  (McLaughlin-Bader section 5.1) -- k sources cost k times one source.
- Gunrock bc.hxx:130-135 and 171-172 uses `atomic::add` on f32 sigma, delta and
  bc_values -- not usable as-is on WebGPU.

Primitives: BFS machinery (section 3) with level bookkeeping, segmented
reduce (2.2), reduce-by-key over the edge frontier (sort-free: scatter
with u32 atomicAdd for sigma when sigma is an integer that fits), sort by
distance (or keep the per-level `S`/`ends` layout so no sort is needed).

Format views: `rowPtr`, `colIdx` (forward pass); the backward pass reads
successors, i.e. the same forward rows (McLaughlin-Bader) -- no reverse
view needed for undirected; for directed graphs the successor gather is
still over forward rows, so `reverse()` is never needed for vertex BC.
Edge BC writes per arc; fold with `foldArcs(..., "first")` (design doc
10.7) and, for undirected, divide by 2 as both papers do.

WebGPU adjustments:

- sigma as `array<atomic<u32>>` with integer `atomicAdd` (exact until
  2^32 paths; cuGraph uses `edge_t` which is also 32-bit in the v32
  builds). Overflow is a real risk on small-world graphs with many
  equal-length paths; detect with `atomicAdd` return + carry into a
  second word, or switch to f32 sigma via the CAS loop for a slower exact
  path. Recommended: u32 sigma + saturation flag; document it.
- delta and bc accumulation: pull over successors per level
  (McLaughlin-Bader Algorithm 2) so each `delta[w]` is written by one
  invocation and `bc[w] += delta[w]` is a plain add; no float atomics
  anywhere. Per-level vertex ranges come from the `S`/`ends` arrays
  produced by the forward pass (the compaction already writes them
  contiguously; `ends` is the running frontier offset).
- Batching sources: the tagged multi-source BFS is the right structure
  for WebGPU because it turns k tiny per-level dispatches into one big
  one; the (vertex, source) key packs into a u32 as
  `v << 16 | s` only for n < 65,536, so use a two-word frontier entry
  (two `array<u32>` bindings) in general. Memory per batch is
  8 bytes * n * k; at n = 100k, k = 256 that is 200 MB -- fits the
  raised limits but not the 256 MiB default buffer for larger n, so the
  batch size is a planner decision from `device.limits.maxBufferSize`.
- Host loop: sources are processed in batches, each batch is a full BFS
  (up to diameter levels) plus the same number of backward levels, so a
  high-diameter graph does thousands of dispatches per batch; use
  indirect dispatch and long submits.
- The McLaughlin-Bader hybrid decision maps directly: run the first batch
  work-efficiently, read back the max depth, then choose the all-edges
  kernel (`edgeList()`/`coo().src` view, each arc relaxed if its source is
  at the current level) for small-diameter graphs.

Complexity: O(n * m) exact; per-source cost equals one BFS plus one
backward sweep; expected speedup vs the CPU Brandes on 100k/1M is large
(tens to hundreds of x with source batching) but exact BC on 100k nodes
is still ~100k BFS traversals: minutes on the GPU, so the API must expose
`k` sampled sources and a progress/cancel hook.

## 7. Closeness, harmonic, eccentricity, APSP

Same forward-pass machinery as BC without the backward sweep: a batched
multi-source BFS (unweighted) or repeated near-far SSSP (weighted) whose
per-source distance rows are reduced (sum for closeness, sum of 1/d for
harmonic, max for eccentricity) on the device without materialising
n x n. Kamada-Kawai's APSP need (03-layout-needs.md section 12.4) is the
one consumer that does want the full n x n Float32Array, which bounds it
to ~10k nodes (400 MB) as that note says. Views: `rowPtr`, `colIdx`,
`weights`. WebGPU adjustments identical to sections 3 and 6.

## 8. Connected components

Strategies:

- Shiloach-Vishkin / Afforest (Sutton, Ben-Nun, Barak IPDPS 2018; GAP
  gapbs/cc.cc:40-150): `comp[v] = v`; `Link(u, v)`: hook the higher root to
  the lower with `compare_and_swap(comp[high], high, low)`, retrying up
  the trees (lines 41-56); `Compress`: pointer-jump every vertex to its
  root (59-66). Afforest first does `neighbor_rounds = 2` link rounds over
  only the r-th neighbour of every vertex (a sampled subgraph), compresses,
  samples `comp` to find the giant component, then links the remaining
  vertices' remaining edges, skipping vertices already in the giant
  component (lines 104-149). Work is close to O(n) on graphs with a giant
  component. Directed graphs: process the reverse graph too (line 142-144).
- cuGraph WCC (weakly_connected_components_impl.cuh:294-320, 418-560):
  multi-root frontier expansion -- pick roots until their degree sum hits
  SMs * 1024, BFS from all roots simultaneously, record "conflict" edges
  between different roots' frontiers, recurse on the much smaller conflict
  graph; degree threshold `ceil(sqrt(degree_sum_threshold * 2))`
  guarantees >= 50 % compression per level. More machinery than Afforest
  for the same asymptotics; it exists because cuGraph is multi-GPU.
- Label propagation (min-label, Jacobi): O(diameter) rounds of
  `label[v] = min(label[v], min over nbrs)` -- simplest kernel, poor on
  high-diameter graphs.

Primitives: edge-parallel map over `edgeList()` (each undirected edge
once, design doc 10.1) with `atomicCompareExchangeWeak` on u32; vertex
map for compress; device reduce/histogram for the frequent-element
sample; compaction for the "remaining vertices" list.

Format views: `edgeList().src/.dst` (each edge once, correct for both
directed and undirected per section 10.1); `rowPtr`/`colIdx` for the
per-vertex r-th-neighbour sampling rounds (`colIdx[rowPtr[v] + r]` if
`r < degree`); for directed WCC also `reverse()` or just treat
`edgeList()` edges as undirected (which is exactly what WCC means, so no
reverse view is required).

WebGPU adjustments: Afforest maps one-to-one (all-u32 atomics). Compress
is a pointer-jumping loop per invocation reading `comp[comp[v]]` written
by other invocations in the same dispatch -- benign race in CUDA, and in
WGSL the reads must be `atomicLoad` on the same `array<atomic<u32>>`
(mixing atomic and non-atomic access to one buffer element is not
allowed within a shader). Host loop: fixed 2 sampling rounds + compress,
one readback for the frequent-element sample (a 1024-entry histogram
reduce is enough), then link + compress until a device-side "changed"
flag stays 0 (check every few iterations). Dense relabel of roots to
0..k-1 is a compaction + scatter (design doc 10.8 says the GPU package
owns it).

Complexity: near O(n + sampled m) work, a handful of rounds; expected to
beat the CPU union-find by 10x+ mostly because the CPU cost is dominated
by Map iteration, not because CC is GPU-friendly.

## 9. Triangle counting, k-truss, k-core

- Triangle counting (cuGraph triangle_count_impl.cuh:344-470; Gunrock
  tc.hxx:63-95): drop self-loops, prune to the 2-core, orient each edge
  from lower to higher degree (tie-break by id; pred op at lines 74-84),
  then per oriented edge (u, v) intersect the oriented adjacency lists of
  u and v (`transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v`).
  Sorted rows (`flags.sortedRows`, design doc 10.5) make the intersection a
  merge (or binary search of the shorter list into the longer, better when
  degrees differ by > 32x). Per-vertex counts: Gunrock does
  `atomic::add(&vertex_triangles_count[intersection_vertex], 1)` (tc.hxx:90)
  -- u32, fine on WebGPU.
- k-truss (k_truss_impl.cuh:183-300): (k-1)-core prune, orient
  low-to-high, count triangles per edge, iteratively remove edges with
  support < k-2 and recount only affected edges ("unroll weak edges").
  Needs an edge mask (bool column, `array<u32>` bitset) and per-edge
  support (`Uint32Array(edgeCount)` written through `edgeToArc`).
- k-core / core number (core_number_impl.cuh:97-230): initialise core
  numbers to degrees, keep a "remaining vertices" list, each round move
  vertices with degree bound < k into a frontier, decrement neighbours'
  counts via the frontier expansion, repeat; cuGraph notes the remaining-
  vertex scan "can add significant overhead" when many distinct core
  numbers exist (line 216-218). GPU peeling is inherently O(max core)
  rounds; the Batagelj-Zaversnik bucket order is sequential. Practical
  WebGPU form: rounds of (frontier of vertices with count < k) ->
  (atomicSub neighbour counts, u32) -> compaction; k increases when the
  frontier empties.

Primitives: advance with per-edge intersection (one invocation per
oriented arc; the intersection loop length is min(deg u, deg v), balance
by putting high-degree pairs in a workgroup-per-arc kernel), u32 atomics,
compaction, bitset masks.

Format views: `rowPtr`, `colIdx` sorted; `outDegree()`; `edgeList()` or
`coo({canonical})`-style "each edge once" for orientation (the oriented
edge set is a compaction of arcs where `(deg[u], u) < (deg[v], v)`);
`arcToEdge` / `edgeToArc` for per-edge support results; `mate()` is not
required.

WebGPU adjustments: the intersection kernel binds rowPtr, colIdx, the
oriented arc list (2 arrays), counts, mask = 6 storage buffers; the
degree array can be read from rowPtr. Result totals via workgroup partial
sums, not one global atomic (contention). k-truss iteration is host-driven
with an indirect dispatch on the "affected edges" list.

Complexity: TC O(sum over oriented edges of min-degree) -- the standard
bound; large speedups vs CPU (the intersection is bandwidth-friendly).

## 10. Community detection: Louvain, Leiden, label propagation, ECG

cuGraph Louvain (louvain_impl.cuh:60-300, detail/common_methods.cuh):

1. Per level: vertex weights = out-weight sums (segmented reduce);
   cluster weights start equal; modularity Q computed by
   `transform_reduce_e` over edges whose endpoints share a cluster.
2. Move phase, synchronous: `update_clustering_by_delta_modularity`
   aggregates for every vertex the edge weight into each neighbouring
   cluster (`per_v_transform_reduce_dst_key_aggregated_outgoing_e`, a
   per-row group-by-key on `cluster[dst]`, common_methods.cuh:409-424),
   evaluates `delta_Q = 2 * ((new_cluster_sum - old_cluster_sum) /
   total - resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
   total^2)` (lines 70-95), reduces to the best (cluster, gain) with a
   deterministic tie-break (lines 99-116), counts moves, and applies a
   move only if `delta_modularity > min_gain` AND the direction matches
   `up_down` (`(new_cluster > old_cluster) != up_down ? old : new`,
   lines 142-152); `up_down` flips every pass, and flips immediately when
   a pass yields zero moves (line 439). This is the swap-avoidance rule;
   nu-Louvain calls its variant "Pick-Less" (move only to a lower-id
   community) applied every 4th iteration (arXiv 2501.19004).
3. Repeat moves while `new_Q > cur_Q + threshold`; then
   `graph_contraction` relabels vertices to clusters, sorts/aggregates
   edges by (cluster src, cluster dst) and builds the coarse CSR; loop
   until modularity stops improving; `flatten_dendrogram` maps the
   original vertices through every level.

Leiden (leiden_impl.cuh, detail/refine_impl.cuh:60-130): Louvain move
phase, then a refinement phase inside each Louvain community that only
merges a vertex into a refined cluster when the cut condition
`E(Cr, S - Cr) > gamma * ||Cr|| * (||S|| - ||Cr||)` holds, with random
moves currently disabled ("FIXME: Disable random moves in refinement
phase for now", line 107); refinement uses a maximal-independent-set
kernel (detail/maximal_independent_moves.cuh) so adjacent vertices do
not move in the same step. ECG (ecg_impl.cuh:66-118) runs Louvain
`ensemble_size` times with random initial permutations, re-weights each
edge by `min_weight + (1 - min_weight) * co-cluster-frequency`, then runs
Louvain once more on the re-weighted graph.

Other GPU Louvain results (secondary sources; Naim et al. 2017 itself
could not be fetched, see section 12): nu-Louvain uses thread-per-vertex
below degree 64 (128 for aggregation) and block-per-vertex above, per-
vertex open-addressing hash tables of size 2 * degree in one contiguous
global buffer, f32 values, prefix-sum-built CSR for the coarse graph;
5.0x faster than cuGraph Louvain, only 1.03x faster than the 64-thread
CPU GVE-Louvain, and concludes multicore CPUs suit Louvain better because
later passes have too little parallelism (arXiv 2501.19004). Gilbert-
Madduri 2026 (arXiv 2608.01503) adds hash-table sizing to |E| total,
kernel fission (enumerate then argmax), reduction instead of atomics for
connection strength, and a symmetry-breaking "afterburner filter" with an
annealing temperature; 3.1x over nu-Louvain, contraction 15x faster than
competitors.

Label propagation (not in cuGraph C++; nx-cugraph has none; the CPU
package has it): per vertex, the most frequent (weighted) neighbour label
= the same per-row group-by-key as the Louvain move step with `count`
instead of `delta_Q`, synchronous with the same swap-avoidance rule.

Primitives: per-row group-by-key (sort the row's neighbour keys in
workgroup memory for degree <= 256, hash in workgroup memory for larger
rows, global per-vertex hash region sized 2 * degree for the largest
rows), segmented reduce, device reduce for Q and move counts, radix sort
by (cluster src, cluster dst) + segmented reduce for contraction, COO ->
CSR builder (2.7), compaction for relabelling.

Format views: `rowPtr`, `colIdx`, `weights` (symmetric CSR; Louvain is
undirected -- run `toUndirected()` first for directed inputs, design doc
section 14.2), `edgeList()` for the modularity edge sum and for
contraction, `outDegree()` / weighted degree via segmented reduce. After
level 0 the coarse graphs are built on the device and never touch the
format.

WebGPU adjustments:

- Cluster weight updates: cuGraph recomputes `cluster_keys/weights` with
  a reduce-by-key after each pass (common_methods.cuh
  `compute_cluster_keys_and_values`) rather than atomically adjusting
  them during moves; keep that (no float atomics, and it is what makes
  the pass synchronous/deterministic).
- Per-row group-by-key in workgroup memory is limited by 16 KB: 256 keys
  + 256 f32 = 2 KB per row for the thread-per-row tier is fine; the
  workgroup-per-row tier for hubs needs the global hash region.
- Contraction = sort-by-key (2.6) + segmented reduce + CSR build; all
  u32/f32 without atomics except the histogram in COO -> CSR.
- Termination and level loop are host-driven; each level needs a
  readback of Q and the move count (two floats) -- acceptable, since a
  level is many kernels.
- Expectation management: the GPU wins at level 0 on large graphs and
  loses parallelism as the graph coarsens (nu-Louvain's conclusion);
  budget for a hybrid where levels below ~50k vertices could be handed to
  the CPU algorithms package -- but the project rule says the GPU package
  never falls back, so the GPU package should simply run the small
  levels on the GPU too and accept the latency; the caller decides which
  package to call.

Complexity: O(m) per move pass, typically 5-20 passes per level, 3-8
levels; expect 2-10x over the CPU package at 1M edges, not 100x.

## 11. Spectral clustering / spectral layout, k-means

NVIDIA's cluster-analysis page (nvGRAPH era) describes spectral
clustering as "constructs the graph Laplacian matrix, solves an
associated eigenvalue problem, and extracts splitting information from
the calculated eigenvector(s)", with modularity `B_ij = A_ij - k_i k_j /
2m`, balanced cut and flow metrics, and multilevel partitioning that
"collapses nodes and edges together" then refines with Kernighan-Lin
smoothing. cuGraph keeps a legacy spectral_clustering.cu
(cugraph-algos/cpp/src/community/legacy/spectral_clustering.cu). The GPU
kernel content is: Laplacian SpMV (`L v = deg .* v - A v`, one segmented
reduce over rowPtr/colIdx/weights), dot products and norms (device
reduce), Lanczos / LOBPCG orthogonalisation on the host or in small
dense kernels, then k-means on the k-dimensional embedding (distance map
+ argmin + reduce-by-key). 03-layout-needs.md section 12.5 already flags
that the CPU spectral layout uses plain power iteration and so finds the
wrong (largest) eigenvectors; the GPU port should implement inverse /
shifted iteration or Lanczos. Views: `rowPtr`, `colIdx`, `weights`,
`outDegree()`. No atomics; f32 orthogonalisation loses precision for
k > ~8 vectors, so re-orthogonalise every few iterations.

## 12. Force-directed layout prior art (summary; the layout plan note owns the detail)

Read for this note because the owner made layout the first slice:

- cuGraph ForceAtlas2 (read-only sibling checkout
  cugraph/cpp/src/layout/legacy/): attraction is edge-parallel over COO
  with four `atomicAdd(float)` per edge (fa2_kernels.cuh:77-80); repulsion
  is either exact O(n^2) (exact_repulsion.cuh) or Burtscher-Pingali
  Barnes-Hut (bh_kernels.cuh: BoundingBoxKernel, TreeBuildingKernel with
  lock/CAS insertion, SummarizationKernel, SortKernel, RepulsionKernel;
  THREADS1..7 = 512/512/768/128/1024/1024/1024). The float atomics and the
  lock-based tree build are both non-portable to WebGPU; the gather form
  of attraction over the symmetric CSR row (03-layout-needs.md 12.1) and a
  level-by-level tree build replace them.
- GraphWaGu (harp-lab, MIT; WebGPU): FR with an O(n^2) kernel at
  `@workgroup_size(1,1,1)` plus a bit-packed adjacency matrix
  (compute_forces.wgsl -- clearly a baseline), a CSR-like
  `EdgeInfo{source_start, source_degree, dest_start, dest_degree}` with two
  neighbour lists per node built by a single-thread kernel
  (create_sourcelist.wgsl, `@workgroup_size(1,1,1)`, serial over all
  edges -- do not copy), a per-node attraction gather over both lists
  (compute_attractive_new.wgsl), Barnes-Hut via Morton codes ->
  WGSL radix sort (8 bits x 4 passes, Fuchsia port) -> bottom-up
  `cluster_size`-ary tree built one level per dispatch (create_tree.wgsl)
  -> per-node traversal with a 64-entry private stack and theta = 0.8
  (compute_forcesBH.wgsl:49-99), bounding box via i32 fixed-point
  `atomicMin/Max` x1000 (apply_forces.wgsl:79-82). 2D, undirected only.
  This is the closest existing WebGPU design and validates: radix sort in
  WGSL, level-wise tree build, i32 fixed-point atomics for global
  min/max, CSR gather for attraction.
- cosmos.gl (@cosmos.gl/graph 3.4.1, MIT; WebGL2 fragment shaders): exact
  all-pairs below 4,096 points, otherwise a grid PYRAMID (not a tree):
  finest grid ~2 sqrt(n) cells per axis capped at 512, each level
  aggregates [sum x, sum y, count] per cell, far field from cell
  centroids, near field (3x3 neighbourhood) by per-tick depth-peeled
  random subsets (32/16/8 slots by n) with Horvitz-Thompson weighting
  (cosmos/src/modules/ForceManyBody/index.ts:20-110). Link force loops per
  point over a CSR-like first-index+count texture, bounded by the max
  degree (`MAX_LINKS`), in two passes (outgoing / incoming)
  (cosmos/src/modules/ForceLink/index.ts:12-60, force-spring.ts:36-49).
  The pyramid is a compelling alternative to Barnes-Hut on WebGPU: no
  sort, no tree pointers, every level is a fixed-size grid reduction; its
  cost is O(n + cells) per tick and it degrades gracefully to a
  stochastic estimate in dense cells. Worth carrying into the layout plan
  as option B next to GraphWaGu-style BH (option A).
- jaredmcqueen/analytics: WebGL1 GPGPU FR with an O(n^2) texture loop
  (analytics/shaders/sim-velocity.glsl:80-95); the README's "60 FPS with
  1 million nodes" claim is not consistent with the O(n^2) kernel and
  should not be treated as a benchmark. Nothing to reuse.
- Cosmograph (cosmograph.app, the Python package on PyPI) is the product
  on top of cosmos.gl; no additional algorithmic content was verifiable
  from the links.

## 13. Prior-art assessment of the owner-supplied links

| Link | What it is | Reusable for this package |
| --- | --- | --- |
| Merrill-Garland-Grimshaw 2011 | the reference for scan-based frontier expansion, gather tiers, duplicate culling, expand/contract couplings | yes: sections 2.3-2.5, 3 |
| McLaughlin-Bader CACM 2018 | work-efficient vs edge-parallel BC, atomic-free dependency accumulation, sampling-based online switch, source batching across GPUs | yes: section 6 |
| cse.buffalo.edu/tech-reports/2023-06.pdf | MS thesis: dense-matrix BC via cuBLAS and a Katz-walk approximation; beats McLaughlin-Bader only at >= 50 % density; accuracy issues admitted | no (sparse graphs); cite only as a negative result |
| developer.nvidia.com/discover/cluster-analysis | nvGRAPH-era overview of spectral and multilevel graph partitioning, modularity/balanced-cut/flow metrics | background for section 11 only |
| GraphWaGu | WebGPU FR + Barnes-Hut, WGSL radix sort | yes: layout slice, radix sort design |
| cosmos.gl | WebGL grid-pyramid many-body + CSR link force | yes as an algorithm design (pyramid), not as code |
| jaredmcqueen/analytics | WebGL1 O(n^2) FR | no |
| Cosmograph | product / Python bindings over cosmos.gl | no |

## 14. Per-algorithm summary matrix

| Algorithm | Parallel strategy | Primitives | Format views | WebGPU-specific | Rounds (host-visible) |
| --- | --- | --- | --- | --- | --- |
| BFS | two-phase frontier + fused small-frontier kernel; direction-optimizing on graphs with cheap reverse | advance, scan, compaction/dedupe, bitset | rowPtr, colIdx, reverse() (directed), degreeOrder() | u32 CAS claim; indirect dispatch; k levels per submit | O(diameter) |
| SSSP | near-far (delta = 32 w / d), two-level near queue | advance, atomicMin, compaction/dedupe, histogram | rowPtr, colIdx, weights, flags | f32-as-u32 atomicMin (non-negative), 2-pass predecessor | O(D / delta) |
| PageRank / Katz / eigenvector / HITS | pull SpMV, ping-pong ranks | segmented reduce, device reduce | reverse(), device out-weight sums, gpuView columns | no atomics; tier by in-degree; batch iterations | O(iters / k) |
| Betweenness | tagged multi-source BFS forward, successor-pull backward; hybrid edge-parallel for small diameter | BFS set + reduce-by-key + level ranges | rowPtr, colIdx, edgeList() (edge-parallel), arcToEdge for edge BC | u32 sigma (overflow flag), no float atomics, batch memory planned from limits | O(sources / batch * 2 diameter) |
| Closeness / harmonic / eccentricity | multi-source BFS/SSSP with on-device row reductions | as BFS/SSSP | rowPtr, colIdx, weights | same | same |
| Connected components | Afforest (2 sampled rounds + giant-component skip + link/compress) | edge map with CAS, compress, histogram sample, compaction | edgeList(), rowPtr/colIdx for r-th neighbour | all u32; atomicLoad in compress | ~5-10 |
| Triangle count / k-truss / k-core | degree-oriented edge intersection; peeling rounds | intersection advance, u32 atomics, masks, compaction | sorted rowPtr/colIdx, outDegree(), edgeToArc | workgroup-per-arc tier for hub pairs | 1 / O(iters) / O(max core) |
| Louvain / Leiden / LPA / ECG | synchronous best-move with up/down swap rule; contraction by sort + segmented reduce | per-row group-by-key, sort-by-key, segmented reduce, COO->CSR | symmetric rowPtr/colIdx/weights, edgeList() | workgroup hash for rows <= 256, global hash for hubs; no float atomics | O(levels * passes) |
| Spectral | Laplacian SpMV + Lanczos/inverse iteration + k-means | segmented reduce, dot/norm reduce | rowPtr, colIdx, weights, outDegree() | f32 re-orthogonalisation | O(iters) |
| Force-directed (FA2 / FR) | per-node CSR attraction gather; repulsion by BH tree (Morton sort + level build) or grid pyramid | radix sort, level dispatches, reduce (bbox, swing/traction), i32 fixed-point atomics | symmetric rowPtr/colIdx/weights, node columns (mass, size, fixed) | no float atomics; no cross-workgroup locks | 1 submit per k steps |

## 15. Recommendations feeding the plan

1. Primitive order that unblocks the most algorithms: workgroup scan +
   device scan -> segmented reduce (tiered) -> compaction with ownership
   dedupe -> block_mapped advance with indirect dispatch -> radix sort ->
   COO->CSR. The force-directed slice needs only reduce, segmented reduce
   (attraction gather), radix sort (BH) or a grid reduction (pyramid), so
   it does not block on the frontier machinery.
2. Adopt cuGraph's verified constants as defaults and expose them as
   options: BFS alpha = m/n, beta = 24; SSSP delta = 32 * avg_w / avg_deg,
   16 subpartitions; BC batch capped by memory; Louvain up/down rule and
   `threshold` on modularity gain; degree tiers 1024 / 32.
3. Write every kernel in the pull/gather form first; treat push + atomics
   as an optimisation only where the value type is u32/i32.
4. Design the frontier so that k rounds run per submit with indirect
   dispatch; a per-level `mapAsync` would make BFS on a road network
   slower than the CPU.
5. Treat @antv/webgpu-graph as a cautionary example (dense matrices,
   per-iteration readback, all-vertex Bellman-Ford), not as prior art to
   build on; treat GraphWaGu's radix sort and level-wise tree build as
   the WebGPU-proven patterns to re-derive (MIT licensed).
6. Keep the Buffalo thesis and the analytics repo out of the design.

## Sources

Local repositories (cloned under tmp/webgpu-plan/repos/, commit as of
2026-09-14):

- cugraph-algos/ = https://github.com/rapidsai/cugraph (sparse: cpp/src/
  traversal, link_analysis, centrality, community, components, cores,
  structure, detail; cpp/include/cugraph). Files cited:
  cpp/src/traversal/bfs_impl.cuh, cpp/src/traversal/sssp_impl.cuh,
  cpp/src/link_analysis/pagerank_impl.cuh, hits_impl.cuh,
  cpp/src/centrality/betweenness_centrality_impl.cuh,
  katz_centrality_impl.cuh, eigenvector_centrality_impl.cuh,
  cpp/src/community/louvain_impl.cuh, leiden_impl.cuh, ecg_impl.cuh,
  triangle_count_impl.cuh, k_truss_impl.cuh, detail/common_methods.cuh,
  detail/refine_impl.cuh, legacy/spectral_clustering.cu,
  cpp/src/components/weakly_connected_components_impl.cuh,
  cpp/src/cores/core_number_impl.cuh, cpp/include/cugraph/prims/*.cuh
- cugraph/ (sibling agent's read-only sparse checkout, cpp/src/layout/
  legacy/bh_kernels.cuh, fa2_kernels.cuh, exact_repulsion.cuh)
- gunrock/ = https://github.com/gunrock/gunrock (sparse: include/gunrock/
  framework, algorithms, graph, formats). Files cited:
  framework/operators/configs.hxx, advance/block_mapped.hxx,
  advance/merge_path_v2.hxx, filter/filter.hxx,
  neighborreduce/neighborreduce.hxx, algorithms/bfs.hxx, bc.hxx,
  sssp.hxx, pr.hxx, tc.hxx
- GraphWaGu/ = https://github.com/harp-lab/GraphWaGu (src/wgsl/*.wgsl,
  src/webgpu/force_directed.ts, sort.ts, README.md)
- cosmos/ = https://github.com/cosmosgl/cosmos (src/modules/ForceManyBody/
  index.ts, src/modules/ForceLink/index.ts, force-spring.ts, README.md)
- analytics/ = https://github.com/jaredmcqueen/analytics
  (shaders/sim-velocity.glsl, README.md)
- hybrid_BC/ = https://github.com/Adam27X/hybrid_BC (cloned, referenced
  by the CACM paper; not read in detail)
- gapbs/cc.cc = https://raw.githubusercontent.com/sbeamer/gapbs/master/src/cc.cc
- antv-webgpu-graph/ = npm tarball @antv/webgpu-graph@1.0.0
  (https://registry.npmjs.org/@antv/webgpu-graph/-/webgpu-graph-1.0.0.tgz;
  es/link-analysis/pageRank.js, es/traversal/sssp.js)

Papers (PDF -> text under tmp/webgpu-plan/papers/):

- Merrill, Garland, Grimshaw, "High Performance and Scalable GPU Graph
  Traversal", UVA TR CS-2011-05:
  https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal
  and PDF https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-and/BFS%20TR.pdf
- McLaughlin, Bader, "Accelerating GPU Betweenness Centrality", CACM
  61(8) 2018, DOI 10.1145/3230485; landing page
  https://davidbader.net/publication/2018-mb/ ; PDF
  https://davidbader.net/publication/2018-mb/2018-mb.pdf
  (https://dl.acm.org/doi/10.1145/3230485 and the cacm.acm.org fulltext
  returned HTTP 403)
- Utkarsh Kumar, "Accelerating Betweenness Centrality on GPU", MS thesis,
  University at Buffalo, Aug 2023: https://cse.buffalo.edu/tech-reports/2023-06.pdf
- Beamer, Asanovic, Patterson, "Direction-Optimizing Breadth-First
  Search", SC12: https://scottbeamer.net/pubs/beamer-sc2012.pdf
- Davidson, Baxter, Garland, Owens, "Work-Efficient Parallel GPU Methods
  for Single-Source Shortest Paths", IPDPS 2014:
  https://escholarship.org/content/qt8qr166v2/qt8qr166v2.pdf ; abstract
  page https://mgarland.org/papers/2014/sssp/

Web pages:

- NVIDIA cluster analysis: https://developer.nvidia.com/discover/cluster-analysis
- GraphWaGu README: https://github.com/harp-lab/GraphWaGu
- @antv/webgpu-graph docs: https://g.antv.antgroup.com/en/api/gpgpu/webgpu-graph
  and https://www.npmjs.com/package/@antv/webgpu-graph
- WGSL subgroup builtins: https://gpuweb.github.io/gpuweb/wgsl/#subgroup-builtin-functions
- Sahu, "CPU vs. GPU for Community Detection: Performance Insights from
  GVE-Louvain and nu-Louvain": https://arxiv.org/html/2501.19004
- Gilbert, Madduri, "GPU-Accelerated Multilevel Graph Clustering: A
  Parallel Perspective on Louvain and Leiden": https://arxiv.org/html/2608.01503
- Afforest repository README: https://github.com/michaelsutton/afforest/blob/master/README.md
- Search-result pages used only to locate the above (Naim et al. 2017
  citation: https://dblp.org/pid/175/5742.html ; McLaughlin-Bader:
  https://cacm.acm.org/magazines/2018/8/229768-accelerating-gpu-betweenness-centrality/fulltext ;
  Davidson: https://dl.acm.org/doi/10.1109/IPDPS.2014.45 ; Sutton et al.:
  https://ieeexplore.ieee.org/document/8425156/ )

Not fetched / not verified (see also the structured summary):

- Naim, Manne, Halappanavar, Tumeo, "Community Detection on the GPU",
  IPDPS 2017 (http://ieeexplore.ieee.org/document/7967153/): paywalled;
  the UiB mirror URL returned an HTML page, not the PDF; described here
  only through nu-Louvain / Gilbert-Madduri secondary sources.
- Burtscher and Pingali, "An Efficient CUDA Implementation of the
  Tree-Based Barnes Hut n-Body Algorithm" (GPU Computing Gems 2011):
  known only through cuGraph's bh_kernels.cuh, which is a port of it.
- The GraphWaGu paper PDF (Google Drive link in the README) was not
  fetched; statements come from the source code.
- Cosmograph (cosmograph.app/examples, pypi cosmograph) pages were not
  fetched; cosmos.gl source was read instead.
- Gunrock's published performance numbers and the Gunrock paper
  (arXiv 1501.05387) were not read; only Gunrock source was used.
