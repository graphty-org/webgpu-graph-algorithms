# 03 -- GPU force-directed layout: prior art at kernel level

Purpose: enough kernel-level detail about existing GPU force-directed layouts
to design the WebGPU layout slice of this package (the owner's first need).
Everything below was read from cloned source or downloaded papers under
`/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/`
(commit hashes recorded per project). Where a fact could not be verified it is
marked UNVERIFIED. Plain ASCII throughout.

Conventions: n = node count, m = edge count (graph-format `arcCount` is 2m for
undirected snapshots), d = distance between two points, "1/d law" means the
force MAGNITUDE falls off as 1/d (so the per-component update is
`delta * k / d^2`).

---

## 0. Executive summary

- Every serious GPU graph layout solves the same three sub-problems: (a) an
  n-body repulsion approximation, (b) an O(m) edge attraction, (c) a global
  step-size control. (b) and (c) are cheap and nearly identical everywhere;
  (a) is 80-95 % of the per-iteration time in every measurement found
  (Brinkmann 2017 Table I: BH kernels ~80 %; GraphWaGu 2025 Fig. 6:
  repulsion dominates; Burtscher 2011 Table 6.2: kernel 5 = 81 % of GPU
  time).
- Four repulsion strategies are in production use on GPUs:
  1. exact tiled O(n^2) (cosmos <= 4,096 points; cuGraph `exact_fa2`;
     d3-force-webgpu; GraphGPU; jaredmcqueen/analytics; cosmos 3D branch);
  2. Barnes-Hut tree built top-down with CAS locks (Burtscher & Pingali 2011,
     reused verbatim by cuGraph and by Brinkmann et al. 2017);
  3. Barnes-Hut-like tree built bottom-up from a spatial sort (GraphWaGu
     2025: Hilbert code + radix sort + level-by-level merge);
  4. uniform grid pyramid with an exact/Monte-Carlo near field (cosmos.gl
     >= 4,097 points, "P3M": particle-particle / particle-mesh).
- WGSL has atomics only on `u32`/`i32` (WGSL spec 6.2.8), no float atomics,
  no warp vote/shuffle in core, no guarantee that all workgroups are
  co-resident (so the Burtscher spin-wait summarization is unsafe in
  WebGPU). This rules out a literal port of cuGraph's kernels and of any
  edge-parallel scatter with `atomicAdd(float)`. Every WebGPU/WebGL project
  surveyed copes by gathering per node instead of scattering per edge, by
  fixed-point `i32` atomics (GraphWaGu bounding box), by additive blending in
  a render pass (cosmos grid aggregation), or -- badly -- by accepting data
  races (d3-force-webgpu, GraphGPU).
- Recommendation (section 8): one force law abstraction (all the laws graphty
  needs are 1/d magnitude with different strength factors) over TWO
  repulsion back-ends chosen by n and dim: an exact tiled all-pairs kernel
  (also the test oracle) up to roughly 16k nodes on a discrete GPU, and a
  cell-sorted uniform-grid pyramid (cosmos's P3M re-expressed as compute
  kernels: counting sort with `u32` atomics, segmented reductions for the
  centroids, fixed 3x3 / 6x6 (3D: 3^3 / 6^3) far-field loops, exact near
  field over sorted cell ranges) for everything larger, in both 2D and 3D. A
  Hilbert-sorted cluster tree (GraphWaGu style) is the documented fallback
  experiment for pathological non-uniform distributions; a Burtscher-style
  locked quadtree is NOT recommended for WebGPU.

---

## 1. cosmos.gl (Cosmograph's engine) -- WebGL 2, fragment-shader GPGPU

Repo: `repos/cosmos` = https://github.com/cosmosgl/cosmos (redirects to
cosmosgl/graph), commit `6843f5d9` (2026-09-13), package `@cosmos.gl/graph`
3.4.1, MIT (`repos/cosmos/LICENCE`). Authors per `CITATION.cff`: Rokotyan,
Stukova, Ovsyannikov. The Python package `cosmograph` on PyPI is a Jupyter
widget wrapping the same JS engine, GPL-3.0-or-later (PyPI page).
cosmograph.app's public docs (docs-general/concept) add nothing algorithmic
beyond "the entire force simulation on the GPU"; the only technical
write-ups are inside the repo (`docs/many-body-force/README.md`,
`docs/collision-force/README.md`, `history/2026/*.md`).

### 1.1 Data layout

- Every per-point quantity is an `rgba32float` texture of side
  `pointsTextureSize = ceil(sqrt(n))`; point i lives at texel
  `(i % size, i / size)` (`src/modules/ForceManyBody/force-nearfield.frag`
  lines 74-88). Positions: `.rg` = x,y (3D branch: z in `.a`). A separate
  velocity texture accumulates forces by additive blending; each force is a
  full-screen fragment pass writing one fragment per point.
- Links are stored TWICE, CSR-like, in textures: `linkFirstIndicesAndAmount`
  (per point: first link texel x,y + count) and `indices` (per link: texel
  coords of the other endpoint) plus per-link `bias`, `strength`, random
  distance factor (`src/modules/ForceLink/index.ts` lines 43-78). One
  `ForceLink` instance is built for OUTGOING and one for INCOMING
  (`src/index.ts` lines 429-430), so attraction runs as two per-node gather
  passes; the shader loop bound is the graph's max degree baked into the
  GLSL as a constant (`force-spring.ts` line 39: `const float MAX_LINKS`),
  and the program is recompiled when max degree changes.
- Space is a fixed square `[0, spaceSize]`, default 4096 (larger crashes iOS,
  `src/variables.ts` line 14-15); positions are clamped to it every tick
  (`src/modules/Points/update-position.frag`).

### 1.2 Per-tick pipeline (`src/index.ts` `runSimulationStep`, lines 2011-2100)

Gauss-Seidel style: each force pass is immediately followed by an
`updatePosition` pass (friction + integrate + clamp), swapping the position
FBO before each write:

1. gravity (if `simulationGravity`): velocity += alpha * gravity * 0.1 * dist
   toward the space centre (`ForceGravity/force-gravity.frag`);
2. center (if `simulationCenter`): toward the centre of mass (a separate
   reduction pass);
3. many-body repulsion (section 1.3);
4. link INCOMING pass, then link OUTGOING pass;
5. cluster force (optional), collision force (optional, spatial hash grid,
   section 1.5);
6. alpha decay: `alpha += (alphaTarget - alpha) * (1 - ALPHA_MIN^(1/decay))`
   with `ALPHA_MIN = 0.001`, `simulationDecay` default 5000
   (`src/modules/Store/index.ts` lines 8, 329-332).

Force laws (all d3-style):
- repulsion: `addV = alpha * repulsion * mass / d` with the d3 minimum-distance
  clamp `if (l < 1) l = sqrt(l)` on the squared distance
  (`force-level.frag` lines 46-57) -- i.e. a 1/d law, identical to
  d3-force `manyBody` (`repos/d3/manyBody.js` lines 69-71,
  `node.vx += x * quad.value * alpha / l`);
- link spring: `l = max(l, r*0.99); f = (l - r)/l * linkSpring * alpha *
  strength * bias` where `r = linkDistance * random in
  linkDistRandomVariationRange`, `strength = sqrt(1/min(deg))` by default and
  `bias = deg(other)/(deg(a)+deg(b))` (`force-spring.ts` lines 79-93,
  `ForceLink/index.ts` lines 55-68) -- d3-force `link` semantics;
- parameters (`src/config.ts` lines 372-456, defaults `src/variables.ts`
  lines 69-78): `simulationDecay` 5000, `simulationGravity` 0.25,
  `simulationCenter` 0, `simulationRepulsion` 1.0, `simulationRepulsionTheta`
  1.15 (DEPRECATED no-op since #240), `simulationLinkSpring` 1,
  `simulationLinkDistance` 10, `simulationLinkDistRandomVariationRange`
  [1, 1.2], `simulationRepulsionFromMouse` 2, `simulationFriction` 0.85,
  `simulationCluster` 0.1, `simulationCollision` 0 (+ radius, padding).

### 1.3 Many-body repulsion: grid pyramid + Monte-Carlo near field (P3M)

Source of truth: `docs/many-body-force/README.md` and
`src/modules/ForceManyBody/index.ts`. Two paths:

**Exact path, n <= 4,096** (`ALL_PAIRS_MAX_POINTS`, index.ts line 75;
`force-allpairs.frag`): one fragment per point loops over every other point
(texelFetch), sums the 1/d pairwise force, splits pairs into "near" (within
`2 * finestCellSize`, jittered and magnitude-clamped) and "far" (unbounded)
so dynamics match the grid path across the threshold. Coincident points get
a per-point random kick. Measured 1.81 ms/step at 2k points (history
2026-08-14).

**Grid path, n > 4,096** (three stages, all render passes):

1. `drawLevels()`: a pyramid of grids 4^2, 8^2, ... up to a finest
   resolution of `2^ceil(log2(2*sqrt(n)))` per axis, floored at 8, capped at
   `MAX_GRID_SIZE = 512` (index.ts lines 20-29). Each level is a
   `rgba32float` render target; every point is drawn as a 1-pixel point with
   ADDITIVE BLENDING accumulating `[sum x, sum y, count, 0]` per cell
   (`calculate-level.vert/frag`). This is the WebGL way to do a float
   scatter-add without float atomics (needs `EXT_float_blend`, README
   "Known Issues"). Average finest-cell occupancy is 1/4 point.
2. `drawNearFieldSlots()`: K "depth-peeling" passes over the finest grid;
   pass k selects per cell the point with the smallest per-tick integer hash
   (lowbias32, `build-nearfield-slots.vert` line 63) not selected by passes
   0..k-1, using the depth test as a per-cell argmin. Result: a
   `sampler2DArray` of K layers holding a uniform random K-subset of each
   cell's points, redrawn every tick. K = 32 (n <= 16,384), 16 (<= 65,536),
   8 above (index.ts lines 51-55).
3. `drawForces()`: per point,
   - `force-level.frag` per level: at the COARSEST level, sum centroid forces
     from every cell except the 3x3 Chebyshev-1 neighbourhood; at each finer
     level, sum the aligned 6x6 child block of the parent's 3x3 minus this
     level's own 3x3. Space is tiled exactly once; there is no theta
     (lines 60-96).
   - `force-nearfield.frag`: for the finest 3x3 neighbourhood, sum true
     pairwise forces from the K sampled slots and scale each cell's sum by
     `others / sampled` (Horvitz-Thompson unbiased estimator, line 139);
     cells with <= K points are therefore exact. Then jitter
     (`velocity += velocity * random`) and clamp the per-tick step to
     `2 * cellSize` (line 154).

Cost model stated in the docs: K sequential peel passes (~0.1 ms fixed each)
plus a fixed 9 + 27-per-level texel loop per point; memory at the 512^2 cap
with K = 8 is ~20 MB. Benchmark (history 2026-08-14, GPU not named; the
"Repulsion Benchmark" story forces a readback so numbers are not vsync
capped): 2k 1.81 ms/step (exact), 5k 3.80, 20k 2.29, 50k 3.90, 100k 6.63,
200k 13.79 ms/step. README headline: "hundreds of thousands of points and
links"; the performance story `Hyperbolic Graph (140k points, ~1M links)`
exists (`src/stories/performance.stories.ts` line 36). The docs also record
WHY the earlier theta-banded quadtree was abandoned: centroid-only near
field is purely radial (hubs collapse into disks/petals), theta was a
tuning footgun, and the once-tiling is 1.2-4x faster per step.

Known failure mode and fix (history 2026-08-14): when links/gravity keep a
cell's occupancy far above K while alpha stays high, the per-tick re-sampled
estimate shows as permanent shimmer (163-node country graph: 0.46 units/tick
random walk). Fixed by the exact path below 4,096 and adaptive K.

### 1.4 3D

`feat/3d` branch (fetched, HEAD `9cc081d0`,
`src/modules/ForceManyBody/force-many-body-3d.frag`): "Exact O(n^2) 3D
repulsion. The 2D force uses a quadtree approximation ... which does not port
to 3D without an octree -- this brute-force pass is used in 3D mode instead
... practical up to roughly 10-20k points on discrete GPUs." (The 2D docs'
claim that a P3M 3D force exists on `feat/3d` was not found at that HEAD;
UNVERIFIED for other branches such as `feat/3d-merged`.)

### 1.5 Collision force (relevant near-field technique)

`docs/collision-force/README.md`: a uniform grid sized from the physics
(`cell = max(2R, 8)`, grid = min(512, floor(spaceSize/cell))) so the 3x3
neighbourhood is exhaustive; per-cell AVERAGE position/size (again via
additive blending), evaluated on FOUR half-cell-offset grids so a contact
split by a cell boundary is caught by one of them. O(n) with a fixed
constant; the "cell-average contact" is the approximation.

### 1.6 What to take from cosmos

- The exact-once tiling of a grid pyramid with a fixed 3x3 / 6x6 loop
  structure is deterministic, branch-light and theta-free; it is the best
  fit for a compute-shader port (section 8).
- The near-field problem is real: hub cells can hold hundreds of points.
  Cosmos answers it statistically; with compute shaders and a cell-sorted
  index we can answer it exactly with a bounded loop plus a statistical
  fallback above an occupancy cap.
- Two link passes (in/out) are an artefact of their edge textures; with an
  undirected CSR snapshot (both directions stored) one gather pass suffices.
- Their sequential force->integrate->force ordering is not required for
  quality; cuGraph/FA2 accumulate all forces then integrate once.

---

## 2. GraphWaGu -- WebGPU compute, Barnes-Hut on a Hilbert-sorted tree

Repo: `repos/GraphWaGu` = https://github.com/harp-lab/GraphWaGu, commit
`bee7b7b8` (2025-04-28), MIT (Landon Dyken). Papers:
- Dyken, Poudel, Usher, Petruzza, Bhatia, Kumar, "GraphWaGu: GPU Powered
  Large Scale Graph Layout Computation and Rendering for the Web", EGPGV 2022
  (`repos/papers/graphwagu-2022.pdf`, fetched from stevepetruzza.io);
- Dyken, Usher, Petruzza, Sintos, Kumar, "Accelerating Web-Based Graph
  Drawing with Bottom-Up GPU Quadtree Construction", PacificVis (year not
  printed in the PDF; cites 2024 NSF awards)
  (`repos/papers/pacificvis-graphwagu.pdf`, fetched from evl.uic.edu with
  certificate verification disabled because the site's certificate has
  expired).
The checked-out code is the 2025 (bottom-up) version.

### 2.1 Data layout

- `Node { value, x, y, size }` storage array (16 B/node); forces as a flat
  `array<f32>` of 2n; positions normalised to roughly [0,1] then drifting
  (bounding box recomputed each iteration).
- Attraction: CSR in BOTH directions built by SINGLE-THREAD kernels
  (`create_sourcelist.wgsl` / `create_targetlist.wgsl`, `@workgroup_size(1)`,
  a serial loop over all edges -- an obvious O(m) serial bottleneck done
  once at load). Per node `EdgeInfo { source_start, source_degree,
  dest_start, dest_degree }`.
- Tree: `TreeNode { boundary: vec4f, CoM: vec2f, mass: f32, test, code,
  level, test2, test3: u32, pointers: array<u32, cluster_size> }` = 64 B
  with `cluster_size = 4` (`force_directed.ts` line 48; substituted into the
  WGSL via `CHANGEME`). Tree slots: n leaves + n/4 + n/16 + ... ~ 1.33 n.

### 2.2 Per-iteration kernels (`src/webgpu/force_directed.ts` lines 572-960)

1. `morton_codes.wgsl` (@workgroup_size 128): normalise x,y by the
   bounding box, quantise to 16-bit fixed point (`float_to_fixed`, line 52),
   compute a 32-bit HILBERT code (`hilbert_xy_to_d`; the Morton variant is
   computed but unused, line 174), write `morton_codes[i]`,
   `morton_indices[i] = i`, and the leaf `TreeNode` at `tree[i+1]` (mass 1,
   CoM = position, level 16).
2. Radix sort of (code, index) pairs, 32-bit keys, 8 bits per pass, a WGSL
   re-implementation of the Fuchsia Vulkan radix sort (`sort.ts` header).
3. `create_tree.wgsl`, dispatched `log_4(n)` times from the host, one
   `queue.writeBuffer` of the step index per level (lines 668-690): each
   thread merges `cluster_size` consecutive nodes of the previous level into
   one parent (mass sum, mass-weighted CoM, level = min over children of the
   common-prefix level of their Hilbert codes, `find_morton_split_level`),
   appending parents after the current level's end. The "quadtree" is thus a
   4-ary CLUSTER TREE over the space-filling-curve order, not a true
   quadtree: node extent is the smallest Hilbert-prefix box containing the
   cluster.
4. `compute_attractive_new.wgsl` (128 threads): per node, gather over both
   CSR lists; FR attraction `d^2 / l` toward each neighbour (no weights).
5. `compute_forcesBH.wgsl` (128 threads): per node, depth-first traversal
   with a PRIVATE `array<u32, 64>` stack (line 49); accept a tree node when
   `theta > (2 * boundary.w) / dist` (line 70) and add
   `mass * (l^2 / d) * dir` (FR repulsion); leaves (mass 1) are always
   direct. Note: the pop is `counter--; if (counter < 0u) break;` on a
   `u32`, which never fires (wraps); termination relies on `tree_idx == 0`
   -- a latent bug, not load-bearing for our design.
6. `apply_forces.wgsl` (128 threads, 2 nodes per thread): clamp force to
   `cooling_factor` (FR temperature), integrate, zero forces, and update the
   bounding box with `atomicMin/atomicMax` on `i32` FIXED-POINT
   (`floor(x * 1000)`, lines 79-82) -- the WebGPU-compatible answer to "no
   float atomics" for a min/max reduction.
7. Host: `coolingFactor *= 0.9` per iteration (line 953; class default 0.985
   line 24), theta default 0.8 (line 41), stop when cooling < 1e-4.

### 2.3 Reported performance

PacificVis paper, RTX 4070 Laptop GPU, DX12 backend, 1000 iterations, theta
= 2 for all benchmarks (section 4.2; Fig. 4): speedups 15.7x (sf_ba6000) to
69.5x (finance256) over 2022 GraphWaGu; pkustk13 (94,893 nodes, 6,616,827
edges) 5.48 ms/iteration ("182 fps"); comYoutube (1,134,890 nodes, 5,975,248
edges) ~160 ms/iteration, breakdown "Create Tree ~14 ms, Attractive ~10 ms,
Repulsive ~160 ms" (Fig. 6, with sync overhead); finance256 (37,376 /
298,496): total 8 ms (tree 2, attractive 3, repulsive 3). Integrated Iris Xe
speedups 15.0x-35.2x. The 2022 version (single-thread top-down tree
insertion on the GPU plus a `|V| log4 |V|` global stack buffer for BFS
traversal) crashed with out-of-memory on the three largest graphs.
EGPGV 2022 (RTX 2060): GraphWaGu FR (adjacency BITMAP, O(n^2)) had the best
iteration time up to ~5,000 nodes, BH beyond; rendering stayed >= 10 fps to
100k nodes / 2M edges.

### 2.4 What to take from GraphWaGu

- Sort-then-merge tree construction is the only tree build in the survey
  that needs no locks, no spinning and no float atomics: sort (u32 keys),
  then `log_b n` fully parallel merge dispatches. It is WebGPU-safe.
- Its weaknesses: (a) `log_b n` host round trips per iteration (fixable
  with a pre-filled uniform per level + dynamic offsets, or one dispatch per
  level from a pre-recorded command buffer); (b) traversal divergence with a
  64-deep private stack and no warp broadcast; (c) cluster-tree boxes
  overlap, so theta acceptance is looser than a true quadtree's; (d) 64 B
  per tree node with unused `test*` fields; (e) FR-only force law, no
  weights, no mass, 2D only.
- Radix sort and the Hilbert/Morton code kernels are reusable primitives for
  our primitives layer (both MIT).

---

## 3. jaredmcqueen/analytics -- Three.js GPGPU (WebGL 1), exact all-pairs

Repo: `repos/analytics` = https://github.com/jaredmcqueen/analytics, commit
`6bd1c586` (2021-06-29), GPL-3.0 (`LICENSE`). README claims "60 FPS
simulation for scenes with 1 million nodes" and "fruchterman reingold
force-directed simulation, all performed on the GPU".

Kernel (`shaders/sim-velocity.glsl`): one fragment per node; a nested loop
over the ENTIRE position texture (`nodesTexWidth^2` texels) applying FR
repulsion `k^2/d` (lines 19-24, 66-84); then attraction by scanning the
ENTIRE edge texture and applying `d^2/k` only for texels in the node's
`[start, end)` range (lines 86-134) -- O(n * (n + m)) per frame, with every
node reading every edge. Velocity is normalised to `temperature`
(`velocity = normalize(velocity) * temperature`, line 143), speed-limited,
damped by 0.25; `sim-position.glsl` integrates `pos += vel * delta * 50`.
No spatial structure, no CSR gather bound, no Barnes-Hut. The 1M-node
60 fps claim cannot be consistent with an O(n^2) fragment loop at 1M
(10^12 pair evaluations per frame); most plausibly it refers to rendering,
not simulating (UNVERIFIED; no benchmark in the repo). Value to us: a
cautionary example only; nothing to reuse (and GPL).

---

## 4. RAPIDS cuGraph ForceAtlas2 (CUDA) -- Burtscher-Pingali Barnes-Hut

Repo: `repos/cugraph` = https://github.com/rapidsai/cugraph, sparse checkout
of `cpp/src/layout/legacy/` at commit `4f1606a8` (2026-09-14), Apache-2.0
(SPDX headers). Files: `force_atlas2.cu` (dispatch), `exact_fa2.cuh` +
`exact_repulsion.cuh` (O(n^2)), `barnes_hut.cuh` + `bh_kernels.cuh` (BH),
`fa2_kernels.cuh` (attraction, gravity, speed, apply). Lineage: the BH
kernels are the Burtscher & Pingali 2011 code (kernel names, `THREADS1..6`,
`FACTOR1..6`, `-1` null / `-2` lock convention are verbatim), reduced to 2D,
as first done by Brinkmann, Rietveld & Takes (ICPP 2017, section 4.6).

### 4.1 Parameters (`cpp/include/cugraph/algorithms.hpp` lines 175-256; Python
docs docs.nvidia.com/cugraph)

`max_iter` 500, `outbound_attraction_distribution` true, `lin_log_mode`
false, `prevent_overlapping` false (+ `vertex_radius`,
`overlap_scaling_ratio` 100), `edge_weight_influence` 1.0,
`jitter_tolerance` 1.0, `barnes_hut_optimize` true, `barnes_hut_theta` 0.5,
`scaling_ratio` 2.0, `strong_gravity_mode` false, `gravity` 1.0,
`vertex_mobility`, `vertex_mass`, `callback` (removed 25.10). Docs: "Good
short-term quality can be achieved with 50-100 iterations. Above 1000
iterations is discouraged." Peak memory: header says 17*V, Python doc says
30*V (floats; both statements exist -- treat as ~70-120 B/node).

### 4.2 Data layout (`barnes_hut.cuh` lines 55-125)

Structure-of-arrays, bodies first and cells last in the SAME arrays
(Burtscher Fig. 6.7): `nnodes = max(2n, 1024 * SMs)` rounded to a multiple
of 32, minus 1 (lines 62-66); `childl[(nnodes+1)*4]` (4 children per cell,
`-1` null), `massl`, `nodes_pos` (x block then y block, each `nnodes+1`),
`startl`, `countl`, `sortl`, `rep_forces[(nnodes+1)*2]`, plus n-sized
`attract`, `old_forces`, `swinging`, `traction`. Edges are COO
(`src_indices`, `dst_indices`, `edge_data`), sorted once for coalescing
(line 172). Default mass = out-degree + 1 (lines 176-187) -- exactly the
graph-format plan (`outDegree()` + 1).

### 4.3 Per-iteration kernels (`barnes_hut.cuh` lines 220-343)

1. fills (rep_forces, attract, swing, traction = 0), `ResetKernel`;
2. `BoundingBoxKernel` (512 threads x 3*SMs blocks): block-local min/max
   reduction in shared memory, last block (via `atomicInc` limiter) combines
   and writes the root cell (radius = half max extent + 1e-5);
3. `ClearKernel1` (children = -1), `TreeBuildingKernel` (512 threads):
   Burtscher's iterative insertion: descend to a leaf slot; `atomicCAS(slot,
   -1, i)` to insert into an empty slot; else `atomicCAS(slot, ch, -2)` to
   LOCK, allocate new cells with `atomicSub(bottomd, 1)` walking down until
   the two bodies separate, then `__threadfence(); childd[locked] = patch`
   to publish/unlock (lines 178-248). Cell budget exhaustion is handled by
   clamping `bottomd` to N (silently degrading);
4. `ClearKernel2`, `SummarizationKernel` (768 threads): bottom-up centre of
   mass; cells assigned in reverse allocation order so children are usually
   ready; a cell whose children are not ready is retried in five wait-free
   pre-passes and then SPIN-WAITED on a `volatile` mass array (lines
   313-427). This requires all blocks to make progress concurrently;
5. `SortKernel` (128 threads): in-order traversal permutation `sortl`
   (bodies of the same cell contiguous) and compaction of children to the
   front (lines 432-470);
6. `RepulsionKernel` (1024 threads, `__launch_bounds__`): bodies processed
   in `sortl` order; a per-WARP shared-memory stack (`pos[]`, `node[]`
   written by lane 0 only) and a per-depth acceptance table
   `dq[depth] = (radius^2 / theta^2) * 0.25^depth + epssq` (lines 496-509);
   a cell is accepted when `__all_sync(mask, dxy1 >= dq[depth])` -- ALL
   lanes of the warp agree (line 583), otherwise the whole warp descends;
   force `scaling_ratio * mass_i * mass_n / (d^2 + epssq)` times `(dx, dy)`
   (a 1/d law with a softening epssq = 0.0025, line 59);
7. `apply_gravity` (linear: `mass*gravity/d`, strong: `scaling_ratio * mass
   * gravity`, `fa2_kernels.cuh` lines 130-175);
8. `apply_attraction` (edge-parallel, 256 threads): per COO edge,
   `weight^edge_weight_influence`, `factor = -coef * w` (`coef =
   sum(mass)/n` compensation when outbound distribution is on), LinLog
   `log(1+d)/d`, prevent-overlap using `d' = d - r_src - r_dst` (0 force
   when overlapping), `/ mass[src]` for outbound distribution, then FOUR
   `atomicAdd(float)` (lines 15-84) -- the pattern WebGPU cannot use;
9. `compute_local_speed`: per node `swing = mass * |F(t) - F(t-1)|`,
   `traction = 0.5 * mass * |F(t) + F(t-1)|`; two `thrust::reduce` to the
   HOST; `adapt_speed` on the CPU (jt = jitter_tolerance * clamp(0.05 sqrt(n)
   * t / n^2, sqrt(0.05 sqrt n), 10); speed_efficiency *= 0.5 / 0.7 / 1.3
   rules; `speed += min(target - speed, 0.5 * speed)`; lines 252-286 -- a
   line-by-line port of Gephi's `ForceAtlas2.java` lines 296-328);
10. `apply_forces_bh`: `factor = speed / (1 + sqrt(speed * swing_i))`
    (x0.1 and capped at 10 units when preventing overlap), `pos += F *
    mobility * factor`, `old = F`.

`exact_fa2` differs only in step 6: `repulsion_kernel` with a 2D grid
(32x32 threads, <= 256x256 blocks), each thread handling pairs `(i, j<i)`
and adding the symmetric contributions with four float `atomicAdd`s
(`exact_repulsion.cuh` lines 11-53).

### 4.4 Reported performance

RAPIDS could not be fetched directly (Medium returns 403). Search snippets
of "Large Graph Visualization with RAPIDS cuGraph" (Linsenmaier, Dec 2020)
claim "2788x faster than existing Python packages and 3x-6x faster than the
previous GPU version" and "graphs with more than 50M vertices and edges ...
in just a few minutes" (UNVERIFIED beyond the snippet). The measured
numbers we can cite are Brinkmann et al. 2017 (same kernels, GTX Titan X,
500 iterations, `repos/papers/brinkmann2017.pdf` Table I): com-youtube
(1,134,890 / 2,987,624) 1.925 min = 231 ms/iteration; com-dblp (317,080 /
1,049,866) 0.208 min = 25 ms/iteration; ca-AstroPh (17,903 / 196,972)
0.015 min = 1.8 ms/iteration; CORPNET-4 (4.6M / 123M) 13.83 min. Overall
40x-123x over their C++ CPU port; BH force kernel only ~45x and "the body
repulsion kernels constitute approximately 80 % of the execution time"
(section V-D); attraction speedup drops with average degree because of
atomic contention on hub nodes.

Burtscher & Pingali 2011 (`repos/papers/burtscher-pingali-gcg11.pdf`,
3D octree, Quadro FX 5800, 5,000,000 bodies): 5.2 s per step; kernel
breakdown ms: bbox 0.8, tree build 868, summarize 100, sort 39, force
4,203, integrate 4; "for problem sizes below about 10,000 bodies, the
O(n^2) CUDA implementation is the fastest"; BH is 0.2x, 3.3x, 35x, 314x the
O(n^2) speed at 5k, 50k, 500k, 5M bodies; O(n^2) hits 305 GFLOP/s vs BH 76.

### 4.5 What to take from cuGraph / Burtscher

- The FA2 force pipeline (mass = degree+1, outbound compensation, LinLog,
  overlap, swing/traction/adaptive global speed with the 50 % rise cap,
  local speed) is exactly what `@graphty/layout`'s FA2 implements and is
  cheap to keep on the GPU; only two global reductions per iteration are
  needed and they can stay on-device (cuGraph brings them to the host).
- Their tree build depends on three CUDA properties WebGPU does not offer:
  warp-synchronous throttling (`__syncthreads` as a lock back-off), a
  `volatile` spin-wait across blocks (needs co-residency), and warp vote
  (`__all_sync`) with a warp-shared stack. Not portable; see section 6.
- Softening (`epssq`) and the per-depth acceptance table are cheap tricks
  worth keeping in any tree or grid kernel.

---

## 5. ForceAtlas2 reference semantics (paper + Gephi)

Paper: Jacomy, Venturini, Heymann, Bastian, PLOS ONE 2014, CC-BY
(journals.plos.org). Gephi source (GPL-3/CDDL dual licence, so NOT copyable
into an MIT package -- only the published formulas are used):
`repos/gephi/ForceAtlas2.java`, `ForceFactory.java`, `Region.java` (raw
files from github.com/gephi/gephi master).

Formulas confirmed against the paper and code:
- attraction `Fa = d` (LinLog: `ln(1 + d)`), with edge weight `w^delta`
  (`edge_weight_influence`), "dissuade hubs" / outbound attraction
  distribution divides by `deg(n1) + 1` (code: `factor = -coef * e /
  n1.mass`, `ForceFactory.java` lines 355-372);
- repulsion `Fr = kr * (deg(n1)+1)(deg(n2)+1) / d` (code: `factor = coef *
  m1 * m2 / d / d` applied to the component vector, lines 132-148: a 1/d
  magnitude);
- gravity `Fg = kg * (deg+1)` (unit direction), strong gravity `kg *
  (deg+1) * d`;
- prevent overlap: `d' = d - size1 - size2`; `d' > 0`: use d'; `d' < 0`:
  repulsion `100 * kr * m1 * m2` (no division) and no attraction; `d' = 0`:
  nothing (lines 206-240);
- swing `swg(n) = |F(t) - F(t-1)|`, traction `tra(n) = |F(t) + F(t-1)| / 2`,
  global speed `s(G) = tau * tra(G) / swg(G)` (limited to +50 % per step),
  local speed `s(n) = ks * s(G) / (1 + s(G) * sqrt(swg(n)))`, maximum local
  speed `ksmax / |F(n)|` (paper); Gephi defaults (`ForceAtlas2.java` lines
  519-547): scalingRatio 2.0 (10.0 under 100 nodes), gravity 1, jitter
  tolerance 1, Barnes-Hut on when n >= 1000, theta 1.2, threads = cores-1;
- Gephi's BH region acceptance: `distance * theta > size` where `size` is
  twice the max distance of a member from the region's centre of mass
  (`Region.java` lines 91-97, 190), i.e. the standard `s/d < theta` test
  with theta = 1.2 -- a much coarser default than cuGraph's 0.5 or
  GraphWaGu's 0.8/2.0. Speed constants: the paper's ks / ksmax numeric
  defaults could not be extracted from the HTML (UNVERIFIED; the Gephi code
  uses `factor = speed / (1 + sqrt(speed * swinging))` with no ks).

`@graphty/layout`'s CPU FA2 (`layout/src/layouts/force-directed/
forceatlas2.ts` lines 26-42) exposes `maxIter` 100, `jitterTolerance` 1,
`scalingRatio` 2, `gravity` 1, `distributedAction`, `strongGravity`,
`nodeMass`, `nodeSize`, `weight`, `linlog`, `seed`, `dim` (2 or 3) -- the GPU
options object must be a superset of this.

---

## 6. Other WebGPU / GPU implementations found (and not found)

| Project | Repulsion | Attraction | Verdict |
| --- | --- | --- | --- |
| `repos/d3-force-webgpu` (jamescarruthers, commit `b19c463b` 2026-08-10, ISC/BSD-style d3 licence) | tiled exact O(n^2): `TILE_SIZE 256`, `var<workgroup> tile: array<vec4f, 256>`, two `workgroupBarrier()` per tile, d3 law with `distanceMin2/Max2` and LCG jiggle (`src/gpu/shaders/manyBody.wgsl`); comment claims "faster than Barnes-Hut for n < ~50k nodes" (UNVERIFIED, no benchmark) | edge-parallel with plain non-atomic read-modify-write on `nodes[].vx` -- documented data race (`link.wgsl` line 109); an alternative per-node kernel scans ALL links (O(n*m)) | the tiled kernel is the right shape for the exact tier; the link kernel is a counter-example |
| `repos/GraphGPU` (drkameleon, commit `4456f02b` 2026-03-02, MIT) | untiled O(n^2) per node, vis.js law `G / d^2` (1/d^2 magnitude), `@workgroup_size(64)` (`src/shaders/index.ts` lines 264-291) | edge-parallel `forces[src] += fx` with acknowledged races (line 310); CPU path has a Barnes-Hut quadtree | counter-example |
| `repos/webgpu-compute-exploration` (scttfrdmn, MIT) | Barnes-Hut listed only in `FUTURE_EXAMPLES.md`; not implemented | -- | nothing |
| bneukom/gpu-nbody (OpenCL octree, from search) | not read | -- | not consulted |
| t-FDP (Zhong et al., TVCG 2023, arXiv 2303.03964) | FFT-accelerated interpolation of a bounded t-distribution force on a grid, "one order of magnitude faster ... two orders faster on the GPU" | -- | far-field via FFT is a documented option; not pursued for v1 (different force law) |
| Yunis, Yokota, Ahmadia 2012 (FMM for graph layout, from search) | fast multipole | -- | not pursued |

No WebGPU octree/quadtree n-body implementation other than GraphWaGu was
found by the searches run ("WebGPU Barnes-Hut", "wgsl octree n-body",
"WebGPU force directed graph"); the field is thin, which argues for
building on the grid approach that has a working production reference
(cosmos) rather than on tree code that has none in WGSL beyond GraphWaGu.

---

## 7. Comparison table

| | cosmos.gl (grid P3M) | cosmos.gl (exact) | GraphWaGu 2025 | cuGraph FA2 BH (Burtscher) | cuGraph exact | d3-force-webgpu | analytics | GraphGPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| API | WebGL 2 fragment passes | WebGL 2 | WebGPU compute | CUDA | CUDA | WebGPU compute | WebGL 1 (three.js) | WebGPU compute |
| Dim | 2 (3D branch: exact only) | 2/3 | 2 | 2 | 2 | 2 | 3 | 2 |
| Repulsion | grid pyramid 4^2..512^2, exact once-tiling, Monte-Carlo near field (K = 32/16/8) | all pairs, one fragment per point | Hilbert sort + 4-ary cluster tree, DFS with private stack, theta | top-down quadtree with CAS locks, spin-wait summarize, warp-vote traversal, theta 0.5 | all pairs, symmetric 2D grid | tiled all pairs (256-tile shared mem) | all pairs, no tiling | all pairs, no tiling |
| Force law | d3 1/d, min-distance clamp | same | FR `l^2/d` | FA2 `kr m_i m_j / d` softened | same | d3 1/d | FR | vis.js 1/d^2 |
| Tree/grid build | additive-blend point draws per level + K depth-peel passes | none | Hilbert codes (1 kernel), radix sort (4x 8-bit passes), log_4 n merge dispatches | bbox reduce, CAS insert, spin-wait summarize, in-order sort | none | none | none | none |
| Kernels / iteration | per force: 1 pass + integrate; many-body = levels (~9) + K peels + levels + near field | ~6 passes | 1 + sort (~12 dispatches) + log_4 n + 1 + 1 + 1 | 2 fills + 8 kernels + 2 host reductions | 2 fills + 6 kernels + 2 host reductions | 5-7 dispatches | 2 passes | 5 dispatches |
| Attraction | per-node gather, 2 passes (in/out), texture CSR | same | per-node gather, 2 CSR lists | edge-parallel COO, float atomicAdd | same | edge-parallel, RACY | per-node scan of all edges | edge-parallel, RACY |
| Atomics | none (blending, depth test) | none | i32 fixed-point min/max bbox; radix sort u32 | int CAS/Sub/Inc/Max + float atomicAdd | float atomicAdd | none | none | none |
| Step control | d3 alpha decay, friction 0.85, sequential force/integrate | same | FR cooling `*= 0.9`, force clamp | FA2 swing/traction adaptive speed, host scalar | same | d3 alpha | temperature normalisation | velocity integration |
| Memory for repulsion structure | ~20 MB at cap (independent of n) | 0 | ~64 B x 1.33 n + 8 B/node codes + sort scratch ~ 110 B/node | ~40 B/slot x 2-3 slots/body = 80-120 B/node (docs: 17-30 floats/node) | 0 | 0 | 0 | 0 |
| Measured | 100k: 6.6 ms/step, 200k: 13.8 ms/step (GPU unnamed) | 2k: 1.8 ms/step | 95k n / 6.6M m: 5.5 ms/iter; 1.13M n: ~160 ms/iter (RTX 4070 Laptop, theta 2) | 1.13M n / 3M m: 231 ms/iter; 317k / 1M: 25 ms/iter (Titan X, 2017, theta not stated) | -- | none | none | none |
| Licence | MIT | MIT | MIT | Apache-2.0 | Apache-2.0 | d3 (ISC-like) | GPL-3 | MIT |

---

## 8. Analysis and recommendation

### 8.1 WebGPU constraints that shape the choice

1. Atomics: `atomic<u32>` / `atomic<i32>` only, with add/sub/min/max/
   and/or/xor/exchange/compareExchangeWeak (WGSL spec sections 6.2.8, 17.8).
   Consequences: no edge-parallel scatter of float forces; no float
   centroid accumulation by atomics. Options that DO exist: (a) gather per
   node (attraction over CSR rows, repulsion per point) -- deterministic and
   race-free; (b) counting sort by cell using `atomicAdd(u32)` on histogram
   and cursor arrays -- order of items inside a cell is nondeterministic but
   the SET is deterministic, and a per-cell sort by index restores full
   determinism if wanted; (c) fixed-point `i32` accumulation relative to the
   cell origin (16-bit fraction of a cell side; sums of up to 2^15 points fit
   in 32 bits) -- exact enough for centroids, deterministic; (d) segmented
   reduction over the cell-sorted order -- deterministic, no atomics at all;
   (e) additive blending into `r32float`/`rgba32float` render targets, only
   with the optional `float32-blendable` feature (Chrome 132+; WebGPU spec
   25.15) -- available on the dev RTX 4070 SUPER but not guaranteed on
   integrated/mobile parts; and it forces render passes into a compute
   library. Recommend (b)+(d), with (c) as the cheaper alternative if the
   segmented reduce proves slow.
2. No warp intrinsics in core (`subgroups` is optional) and no
   guarantee of workgroup co-residency: Burtscher's spin-wait summarization
   and warp-vote traversal are out. Bottom-up level-by-level merging
   (GraphWaGu) and pyramid downsampling (cosmos) are both barrier-free
   between dispatches and therefore safe.
3. Limits (design doc section 10; research note 09 table): default
   `maxComputeWorkgroupStorageSize` 16 KiB (a 256 x vec4f tile is 4 KiB),
   `maxComputeWorkgroupSizeX` 256, 65,535 workgroups per dimension -> at most
   16,776,960 invocations per 1D dispatch; `maxStorageBuffersPerShaderStage`
   8. A `components: 3` position column must be read as `array<f32>` with
   `3*i` indexing, never `array<vec3<f32>>` (stride 16 != 12; design doc
   10.2). No recursion, no dynamic allocation, private arrays are fine
   (GraphWaGu's 64-entry stack).
4. Readback: `mapAsync` is asynchronous; the `LayoutSimulation.step()`
   contract already allows a Promise and says the GPU buffer is
   authoritative while stepping (design doc 14.3). FA2's two global
   reductions (swing, traction) and the adaptive speed update must stay on
   the device (a reduce kernel + a 1-invocation "adapt speed" kernel writing
   a storage-buffer scalar block) or every iteration costs a round trip;
   cuGraph pays that round trip (`thrust::reduce` to host) because CUDA
   launches are cheap and synchronous -- WebGPU's are not.
5. 3D is a first-class requirement (graphty-element renders 3D; positions
   are stride-3; `@graphty/layout` FA2/FR take `dim`). Only the exact
   kernels among the surveyed systems support 3D; cosmos's grid pyramid and
   GraphWaGu's tree are 2D. The recommended grid design extends to 3D at
   the cost of 27/216-cell loops and a coarser finest level (section 8.4).

### 8.2 Cost model and crossovers

Per iteration, repulsion work:
- exact: `n^2` pair evaluations (~20-25 flops + 2 loads from shared memory
  each). Burtscher measured ~305 GFLOP/s for the O(n^2) CUDA kernel on a
  2009 GPU; a 2024 discrete GPU sustains several TFLOP/s on this kernel
  shape, i.e. on the order of 10^11 pairs/s. Estimated ms/iteration at
  10^11 pairs/s: n = 4k -> 0.16 (dispatch-overhead bound), 16k -> 2.6,
  32k -> 10, 65k -> 43, 100k -> 100, 1M -> 10^4. Integrated GPUs are
  roughly 5-10x slower. Evidence: cosmos exact 2k in 1.8 ms (WebGL, fixed
  overheads dominate); Burtscher: O(n^2) fastest below ~10k bodies (2009);
  GraphWaGu 2022: FR bitmap best below ~5k (RTX 2060, but their FR also did
  an n^2 adjacency probe); d3-force-webgpu's "< ~50k" is an unbacked
  comment.
- grid pyramid (cosmos structure as compute): O(n * (K_far + occ_near)) with
  `K_far = |coarsest grid| - 9 + 27 * (levels - 1)` (for 4^2 coarsest and a
  512^2 finest: 7 + 27*7 = 196 centroid evaluations per point in 2D) and
  `occ_near` = points in the 3x3 finest neighbourhood (mean ~2.25 at 1/4
  point per cell; hub cells are the tail). Build: O(n) cell ids + histogram
  + scan over `cells` + scatter + O(cells * levels) downsample. At n = 1M
  in 2D this is ~2 x 10^8 far-field evaluations, comparable to GraphWaGu's
  measured 160 ms/iteration budget at 1.1M but with coherent, branch-free
  memory access; a few tens of ms is a reasonable expectation on the 4070
  SUPER (UNVERIFIED until the walking skeleton measures it).
- Barnes-Hut tree: O(n log n) with a large constant from divergent
  traversal (GraphWaGu: 3 ms of 8 ms at 37k; 160 ms at 1.1M with theta 2;
  Brinkmann/Burtscher kernels: 231 ms at 1.13M on a 2015 GPU). Memory
  ~100 B/node plus sort scratch.

Crossover recommendation (discrete GPU; halve for integrated):
- n <= 16,384: exact tiled all-pairs (deterministic, no build cost, exact
  hubs, trivially 3D). Cosmos chose 4,096 because its WebGL peel passes
  cost ~0.1 ms each; a single compute dispatch has no such floor, so the
  threshold can be higher. Make it a configurable option
  (`repulsion: "exact" | "grid" | "auto"`, `exactMaxNodes` default 16,384)
  and measure.
- n > 16,384: grid pyramid. The grid's cost is nearly flat in n between
  the 512^2 cap and ~1M points (finest level saturates at 2 sqrt(n) = 512
  around n = 65k), so from 65k upward per-node cost is dominated by the
  near field, whose worst case is hub-cell occupancy -- cap it (section
  8.4).
- Memory per node (GPU, dim = 2 / 3), excluding the snapshot itself:
  - exact: positions 12 B (the owner's stride-3 column) + force 8/12 +
    old force 8/12 + swing/traction 8 + mass 4 = ~40-48 B/node; zero
    structure.
  - grid: + cell id 4 + sorted index 4 + histogram cursor (per cell) and per
    cell `[sum x, sum y, (sum z), count]` 16 B x 1.33 (2D pyramid) or x
    1.14 (3D) -- at the 512^2 cap 5.6 MB total in 2D; in 3D a 128^3 finest
    grid is 2.1M cells x 16 B = 34 MB (pyramid ~38 MB), 256^3 would be
    268 MB and is NOT recommended -- cap 3D at 128^3 (or 160^3) and let
    occupancy rise. Per node ~56-64 B plus the fixed grid.
  - GraphWaGu tree: ~110 B/node; Burtscher tree: ~80-120 B/node (2D),
    ~130-170 B/node (3D, 8 children).
  Node counts at which a 100 MB GPU budget is exhausted: exact ~2M,
  grid ~1.5M (2D), tree ~0.9M -- all far beyond the interactive
  per-iteration budget, so time, not memory, is the limit.

### 8.3 Why grid over tree for the primary large-n back-end

1. Build is a counting sort (u32 atomics) + scan + scatter + downsample:
   every stage is a primitive the package needs anyway (histogram, scan,
   compaction, segmented reduce -- the assistant's earlier layered
   recommendation), no locks, no spinning, no float atomics, deterministic.
2. Traversal is a FIXED loop (3x3 exclusion, 6x6 child block per level):
   no per-thread stack, no divergence beyond the near-field occupancy loop,
   coherent memory access (Cosmos measured this loop shape 1.2-4x faster
   than their previous theta-banded traversal).
3. It has a production reference with documented failure modes and fixes
   (cosmos: radial-only near field, coincident points, over-full cells,
   sin-hash instability, per-tick clamp) and published per-step numbers.
4. It extends to 3D by changing loop bounds (27 / 216) and the finest cap.
5. Cell-sorted order doubles as Burtscher's "kernel 4" locality sort: if
   positions, mass and the force scratch are permuted into cell order each
   iteration (or every k iterations), the attraction gather over CSR rows
   also reads spatially coherent positions. graph-format's `INVALID_INDEX`
   and permutation conventions apply (a `perm` array, positions read via
   `perm[i]`; with `override USE_PERM` for the identity case, design doc
   10.1).
Trees win only when the distribution is extremely non-uniform (most of
space empty, a few dense clumps); a grid then wastes far-field work on
empty cells and the near field degrades. Real graph layouts are clumpy, so
keep the Hilbert-sorted cluster tree (GraphWaGu 2025) as the documented
second experiment, sharing the same sort primitive, and decide with
measurements on the hub-heavy test graphs.

### 8.4 Sketch of the recommended kernels (per iteration, grid tier)

Positions are the owner's stride-3 scene-unit column (`positions[3*i +
k]`), uploaded once per `load()` and thereafter GPU-authoritative.

1. `bbox`: workgroup min/max reduce of positions to `i32` fixed point via
   `atomicMin/Max` (GraphWaGu apply_forces lines 79-82) or a two-level
   reduce; derive `cellSize` for the finest level (target 2 sqrt(n) cells
   per axis in 2D, `n^(1/3) * 2` in 3D, capped 512^2 / 128^3). Cosmos
   avoids this by clamping to a fixed `spaceSize`; we should NOT clamp
   scene-unit positions.
2. `cellId`: `cell[i] = floor((p - min) / cellSize)` linearised; also
   accumulate the finest histogram with `atomicAdd(&count[cell], 1u)`.
3. `scan` over `cells` (exclusive prefix sum) -> `cellStart`.
4. `scatter`: `sorted[cellStart[cell] + atomicAdd(&cursor[cell], 1u)] = i`.
5. `centroid` (finest): segmented reduce over `sorted` ranges (or
   fixed-point atomics in step 2) -> `[sum x, sum y, sum z, count]` per
   cell, stored as `array<vec4<f32>>` (16 B, legal stride). Masses: FA2
   needs mass-weighted centroids and total mass (Gephi `Region.java`), so
   sum `mass * p` and `mass`, with count kept for the near-field weight.
6. `downsample` per coarser level: each parent sums its 4 (8) children --
   one dispatch per level, no atomics.
7. `farField`: per point, coarsest-level full loop minus 3x3 (3x3x3), then
   per level the 6x6 (6x6x6) block minus own 3x3 (3x3x3), `F += k * m_i *
   M_cell / (d^2 + eps) * delta` -- the same 1/d law parameterised for d3
   (`mass = 1`, strength), FA2 (`mass = deg+1`, scalingRatio), FR (`k^2`).
8. `nearField`: per point, for each of the 9 (27) finest cells iterate
   `sorted[cellStart .. cellStart + count)` computing exact pairwise forces,
   bounded by `NEAR_MAX` (e.g. 64) samples per cell; above the cap, apply
   cosmos's Horvitz-Thompson weight `count / sampled` to the first
   `NEAR_MAX` entries with a per-iteration hashed offset into the range
   (unbiased, no depth peeling needed because we can index the sorted
   range). Coincident points: random kick from a per-node hash (lowbias32,
   integer hash -- cosmos's reason: `sin()` hashes diverge across vendors).
   Clamp the per-iteration near-field step to `2 * cellSize`.
9. `attraction`: per node, CSR row gather `for a in rowPtr[u]..rowPtr[u+1]`
   over `colIdx[a]`, `weights === null ? 1 : weights[a]` (via `override
   HAS_WEIGHTS`), with the FA2 outbound-distribution / LinLog / overlap
   variants as `override` constants (pipeline cache keyed by the variant).
   Undirected snapshots store both directions, so ONE pass replaces
   cosmos's two and cuGraph's atomics. For degree skew use graph-format's
   `degreeOrder().segmentOffsets` tiers: one thread per row for low-degree
   rows, one workgroup per row (with a workgroup reduce) for the hi tier,
   like the advance operator planned for the algorithms layer.
10. `gravity` + `sum forces`: fused into 9 or 11.
11. `speed`: FA2 -- per-node swing/traction, one reduce kernel to two
    scalars, one 1-invocation `adaptSpeed` kernel (Gephi/cuGraph rules)
    writing `speed`, `speedEfficiency` to a storage block; d3/cosmos --
    host-side alpha decay is a uniform write, no readback.
12. `integrate`: `pos += F * factor` (FA2 local speed) or `v = (v + F) *
    friction; pos += v` (d3), honouring the `fixed` bitmap (u32 words, 32
    nodes each) and `setPosition` writes (12-byte `writeBuffer`), writing
    the stride-3 column in place; staging copy + `mapAsync` only when the
    caller's `step()` promise resolves (design doc 14.3: the GPU buffer is
    authoritative while stepping; readback per step is n * 12 B = 12 MB at
    1M nodes, so batch several iterations per `step(iterations)`).

Dispatch count per iteration: ~12 + levels (2D: up to 8) -- all recorded
into one command buffer, no host round trips except the optional readback.

### 8.5 How this maps onto the graph-format snapshot

| Need | Snapshot source | Binding / note |
| --- | --- | --- |
| attraction rows | `rowPtr` (n+1), `colIdx` (arcCount) | storage, read; windowed at 64-arc boundaries if over the binding limit (design doc 10.6) |
| edge weights | `weights` (Float32Array or null = all ones) | `override HAS_WEIGHTS`; null -> bind `colIdx` in the slot (never read) like the `USE_PERM` pattern |
| FA2 mass | `outDegree()` view (+1 in-shader) or a `mass` node column via `gpuView` | Float32Array(n) |
| node size (overlap) | `size` node column | optional |
| fixed nodes | `bool` column role "fixed" -> packed u32 bitmap (`NodeMask`) | `setFixed(mask)` uploads ceil(n/32) words |
| positions | owner's stride-3 `Float32Array(3n)` | `array<f32>` with `3*i`; `dim` uniform selects whether z participates |
| degree tiers | `degreeOrder().segmentOffsets` (CPU) | three dispatches for the attraction gather |
| release | `gpu.release(snapshot)` on snapshot-replaced | drops CSR uploads; the simulation's own scratch is owned by the `LayoutSimulation` and freed in `dispose()` |
| multigraph | parallel arcs summed (documented behaviour change in 14.3) | matches CSR gather semantics |
| directed input | `toLayoutSnapshot` -> undirected snapshot | layouts always see doubled arcs, self-loops once |

### 8.6 Testing implications (for the plan)

- The exact tiled kernel is the oracle: assert grid/tree forces agree with
  it within a tolerance on random and clumpy inputs (cosmos's country-graph
  and 1,024-points-in-one-cell cases are good fixtures), and assert the
  FA2 pipeline matches `@graphty/layout`'s CPU FA2 step for one iteration
  on small graphs (same formulas, f32 vs f64 tolerance).
- Determinism: with segmented reductions and index-sorted cells the grid
  tier is bitwise reproducible across runs on the same device; document
  that the Monte-Carlo near-field fallback is seeded per iteration from a
  `seed` option.
- Performance fixtures: sizes 4k / 16k / 65k / 262k / 1M nodes, average
  degree 2-10, plus a scale-free graph with a 10k-degree hub, in Node via
  Dawn (`webgpu@0.4.0`) and lightly in headless Chromium on the RTX 4070
  SUPER; report ms/iteration for exact and grid tiers to fix the `auto`
  threshold empirically.

---

## 9. Open questions / unverified items

- cosmos benchmark GPU model is not stated in the history note; numbers are
  indicative only.
- GraphWaGu paper year (PacificVis) is not printed in the PDF; the 2022
  EGPGV venue is confirmed by the diglib/NSF listings.
- cuGraph FA2 performance blog (Medium) could not be fetched (403); only
  search-snippet claims are recorded.
- ForceAtlas2 paper ks / ksmax numeric defaults not extracted from the
  PLOS HTML (the formulas are).
- cosmos `feat/3d` P3M 3D force: not present at the fetched HEAD.
- The 10^11 pairs/s exact-kernel throughput and the grid-tier per-iteration
  estimates are extrapolations, not measurements; the walking skeleton must
  measure them on the dev GPU before the `exactMaxNodes` default is fixed.
- jaredmcqueen/analytics "1M nodes at 60 fps" is not reconcilable with its
  O(n^2) shader; treated as a rendering claim.

---

## 10. Sources

Cloned repositories (under `tmp/webgpu-plan/repos/`):
- https://github.com/cosmosgl/cosmos (-> cosmosgl/graph), commit 6843f5d9,
  branch `feat/3d` commit 9cc081d0; files cited: `README.md`,
  `docs/many-body-force/README.md`, `docs/collision-force/README.md`,
  `history/2026/2026-07-08-many-body-repulsion.md`,
  `history/2026/2026-08-14-nearfield-jitter.md`,
  `src/modules/ForceManyBody/{index.ts,force-level.frag,force-nearfield.frag,force-allpairs.frag,build-nearfield-slots.vert,force-many-body-3d.frag}`,
  `src/modules/ForceLink/{index.ts,force-spring.ts}`,
  `src/modules/ForceGravity/force-gravity.frag`,
  `src/modules/Points/update-position.frag`, `src/modules/Store/index.ts`,
  `src/index.ts`, `src/config.ts`, `src/variables.ts`, `CITATION.cff`,
  `LICENCE`.
- https://github.com/harp-lab/GraphWaGu, commit bee7b7b8; files:
  `README.md`, `LICENSE`, `src/components/tutorial.md`,
  `src/webgpu/{force_directed.ts,sort.ts}`,
  `src/wgsl/{morton_codes,create_tree,compute_forcesBH,compute_forces,compute_attractive_new,apply_forces,create_sourcelist}.wgsl`.
- https://github.com/jaredmcqueen/analytics, commit 6bd1c586; files:
  `README.md`, `LICENSE`, `shaders/{sim-velocity,sim-position}.glsl`,
  `app/simulator.js`.
- https://github.com/rapidsai/cugraph (sparse: `cpp/src/layout/legacy`,
  `cpp/include/cugraph`), commit 4f1606a8; files: `force_atlas2.cu`,
  `barnes_hut.cuh`, `bh_kernels.cuh`, `fa2_kernels.cuh`,
  `exact_repulsion.cuh`, `exact_fa2.cuh`, `algorithms.hpp`.
- https://github.com/jamescarruthers/d3-force-webgpu, commit b19c463b;
  `README.md`, `src/gpu/shaders/{manyBody,link}.wgsl`.
- https://github.com/drkameleon/GraphGPU, commit 4456f02b; `README.md`,
  `src/shaders/index.ts`.
- https://github.com/scttfrdmn/webgpu-compute-exploration, commit
  743f629b; `FUTURE_EXAMPLES.md`.
- Raw files: https://raw.githubusercontent.com/gephi/gephi/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/{ForceAtlas2,ForceFactory,Region}.java ;
  https://raw.githubusercontent.com/d3/d3-force/main/src/manyBody.js .

Papers (downloaded to `tmp/webgpu-plan/repos/papers/`, text extracted with
pypdf):
- https://www2.evl.uic.edu/documents/pacificvisgraphwagu.pdf (Dyken et al.,
  "Accelerating Web-Based Graph Drawing with Bottom-Up GPU Quadtree
  Construction"; fetched with `curl -k`, expired certificate)
- https://stevepetruzza.io/pubs/graphwagu-2022.pdf (Dyken et al., EGPGV 2022)
- https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf (Burtscher &
  Pingali, GPU Computing Gems Emerald Edition ch. 6, 2011)
- https://liacs.leidenuniv.nl/~takesfw/pdf/exploiting-gpus-fast.pdf
  (Brinkmann, Rietveld, Takes, ICPP 2017)

Web pages read via WebFetch / WebSearch:
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679
  (ForceAtlas2, Jacomy et al. 2014)
- https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.force_atlas2/
  (redirect target of docs.rapids.ai stable)
- https://cosmograph.app/examples , https://cosmograph.app/docs-general/concept/ ,
  https://pypi.org/project/cosmograph/
- https://gpuweb.github.io/gpuweb/wgsl/#atomic-types (WGSL atomic types)
- https://arxiv.org/abs/2303.03964 (t-FDP)
- Search-result pages (titles/snippets only, not fetched):
  https://par.nsf.gov/biblio/10384648-graphwagu-gpu-powered-large-scale-graph-layout-computation-rendering-web ,
  https://diglib.eg.org/items/b9dc1e24-9dea-4483-9229-f40315220a29 ,
  https://medium.com/rapids-ai/large-graph-visualization-with-rapids-cugraph-590d07edce33 (403 on fetch),
  https://x.com/rapidsai/status/1334585393790914561 ,
  https://chromestatus.com/feature/5173655901044736 (float32-blendable),
  https://developer.chrome.com/blog/new-in-webgpu-132 ,
  https://github.com/gpuweb/gpuweb/issues/3556 ,
  https://www.semanticscholar.org/paper/Exploiting-GPUs-for-Fast-Force-Directed-of-Networks-Brinkmann-Rietveld/ab5c82679d36ebbde1553923f77660a2afaaa357 ,
  https://github.com/govertb/GPUGraphLayout (Brinkmann's code; licence
  reported by the GitHub API as NOASSERTION, not read),
  https://www.bu.edu/exafmm/files/2012/02/YunisYokotaAhmadia2012.pdf ,
  https://arxiv.org/pdf/2108.00529 (BigGraphVis),
  https://github.com/bneukom/gpu-nbody .

Owner-supplied links NOT applicable to this note (BFS / centrality /
clustering; covered by other notes): research.nvidia.com Merrill 2011,
cse.buffalo.edu 2023-06, dl.acm.org 10.1145/3230485,
developer.nvidia.com/discover/cluster-analysis.

Project files consulted (read-only):
- /home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md
  (sections 10 and 14.3)
- /home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/03-layout-needs.md
  (section 12), 09-webgpu-requirements.md (limits table),
  04-graphty-element-usage.md
- /home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/forceatlas2.ts
- /home/apowers/Projects/graphty-monorepo/graphty-element/src/layout/NGraphLayoutEngine.ts
