# PERF review: performance realism of design/webgpu-acceleration-plan.md

Reviewer lens: performance realism (ms/iteration, memory, upload/readback,
hidden O() work, host round trips, the 100k-1M+ node target on the RTX 4070
SUPER and on integrated GPUs, the exact-vs-approximate crossover, algorithm
speedup claims).

Document reviewed: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
(3328 lines, read in full). Notes 03, 04, 05, 06 and the probe scripts under
tmp/webgpu-plan/probe/ were read for the cited numbers; the accepted design's
sections 10.2, 14.3 and 15 were read for the contracts the plan must honour.

Probes written and run for this review (all under
/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/probes/,
Dawn-in-Node webgpu@0.4.0, NVIDIA RTX 4070 SUPER selected with
`create(["adapter=NVIDIA"])` and `LD_LIBRARY_PATH` pointing at the extracted
libEGL per the memory note):

| Probe | What it measures | Result (10 iterations after warm-up, wall time around submit + onSubmittedWorkDone) |
| --- | --- | --- |
| exact-tile.mjs | plan 7.6's K3 body (mass tile, `jj != i`, coincident-kick branch, `d2` floor, law select) vs the dawn-perf.mjs probe body at 4k-100k nodes | probe: 4k 0.22 ms, 8k 0.40-0.43, 16k 0.88-1.04, 20k 1.09-1.13 (reproduces the plan's 1.11), 32k 2.15-2.17, 65k 7.0-7.2, 100k 13.4-13.7. FA2 body: 4k 0.27, 8k 0.51, 16k 1.13, 20k 1.72-1.75, 32k 2.67-3.46, 65k 8.8-9.2, 100k 17.5-17.8 ms. Ratio FA2/probe 1.1-1.6 (mean ~1.3). Throughput is NOT constant: 0.6e11 pairs/s at 4k, 2.4e11 at 16k, 3.1e11 at 32k, 4.9e11 at 65k, 5.7e11 at 100k (FA2 body). |
| near-field-order.mjs | plan 7.7's G7 near field (9 cells, nearMax 64, Horvitz-Thompson offset, exact pair force) with the grid built on the CPU (G = 512 over the bbox); dispatched in node-index order (as 7.7 specifies) vs in cell-sorted order (`i = sortedIdx[t]`), and stride-3 f32 positions + separate mass vs one packed vec4 | 262k uniform: index 0.77 ms, sorted 0.31, sorted+vec4 0.22. 262k clustered (100 Gaussian clusters): 2.01 / 0.41 / 0.31. 262k clustered + 0.2% outliers at 2-4x the core radius: 19.8 / 0.57 / 0.48. 1M uniform: 8.66 / 1.90 / 0.93. 1M clustered: 34.0 / 2.11 / 1.77. 1M clustered + outliers: 125.4 / 2.09 / 2.55. Occupancy: 1M clustered maxOcc 70, 0% of nodes above nearMax; 1M clustered + outliers maxOcc 871, 88.5% of nodes in cells above nearMax, 17,845 of 262,144 cells occupied. |
| far-field-order.mjs | plan 7.7's G6 far field (4^2..512^2 pyramid, exact-once 6x6-minus-3x3 tiling, 196 evaluations per node) in index vs sorted order; and the per-dispatch overhead of tiny dispatches inside one compute pass | 262k: index 0.84 / 0.67 ms, sorted 0.30 / 0.35 (uniform / clustered). 1M: index 2.61 / 2.22 ms, sorted 0.98 / 1.01. Dispatch overhead: 2.1-2.6 us per 1-workgroup dispatch, 4.3-5.9 us per 4096-workgroup dispatch (31-300 dispatches per pass), i.e. the plan's "~5-10 us" is conservative. |
| grid-occupancy.mjs (CPU only) | finest-cell occupancy of the bbox-derived 512^2 grid when ONE node sits at 2x / 10x / 100x / 1000x the core radius | 1M nodes: 0 outliers -> maxOcc 62, 0% above nearMax. ONE node at 2x -> 16% of nodes in cells above nearMax; at 10x -> 95.5% (maxOcc 1,521, 5,329 cells occupied); at 100x -> 100% (129 cells occupied, maxOcc 33,597); at 1000x -> 2 cells occupied, maxOcc 999,999. 100k nodes: one node at 10x -> 37.6%; at 100x -> 99.7%. |

Two WGSL facts surfaced by the probes: Tint rejects
`lowbias32(i * 0x9E3779B9u ^ j)` ("mixing '*' and '^' requires parenthesis"),
so the plan's hash expressions in 7.2 and 7.7 are not valid WGSL as written;
and `layout: "auto"` drops unreferenced bindings, which the plan's explicit
layouts (5.1) already avoid.

## Findings (most severe first)

### PERF-1 (blocker) -- section 7.7 / 7.17 / 11.4, lines 1552-1559, 1571-1577, 1785-1790, 2740-2744

Claim: the grid tier's geometry (finest cell = bbox extent / 512, bbox from
`state.min / max` recomputed every iteration, "never a fixed spaceSize")
collapses as soon as a handful of nodes sit far from the core, and
ForceAtlas2's own laws as adopted in 7.2 guarantee such nodes on any graph
with an isolated node or small component.

Evidence: with the paper's `1/d` repulsion and regular gravity of constant
magnitude `g m_i` (7.2, 7.9), an isolated node at distance `r` from a core of
total mass `M = sum(deg + 1)` reaches equilibrium at `r = scalingRatio * M /
gravity`. At the 1M / 10M tier `M = 21n = 2.1e7`, so `r = 4.2e7` layout units
against a core whose radius is O(1e3-1e4) (Gephi users see exactly this: FA2
throws isolated nodes to the edge of the universe unless strong gravity is
on). grid-occupancy.mjs shows what the bbox-derived grid does then: ONE node
at 10x the core radius puts 95.5% of a 1M-node layout into finest cells above
`nearMax` (max occupancy 1,521); at 100x, 100% of the nodes share 129 cells;
at 1000x the whole core sits in two cells (max occupancy 999,999). At that
point the near field is a 64-sample Horvitz-Thompson estimate of a
1M-body sum for every node (cosmos's documented shimmer failure, note 03
section 1.3, but for the whole graph), the far field sees the core as one
centroid, G4's per-cell centroid loop runs 1M iterations on one thread, and
the `2 * cellSize` clamp is meaningless (cellSize = 2x the core radius).
The 11.4 fixture list (uniform, clumpy, hub, cosmos's two cases, colinear,
coincident) has no "giant component + dust / isolated nodes" fixture for the
layout, so G4 would pass while real data fails. The same root cause breaks
7.17: `radius = max |p - centroid|` is then 4.2e7, `settleThreshold * radius`
is 4.2e4 units per node per iteration, and `settled` fires after the first
window while the core is still moving (see PERF-3).

Fix (text): in 7.7's geometry table replace the "cell size" row with a
robust extent: `extent = c * rmsRadius` with `rmsRadius = sqrt(sum |p - c|^2
/ n)` (one more vec4 partial in K5, folded by K1; `c` ~ 6-8 so ~99% of a
clustered core is inside) or a fixed multiple of the median nearest-neighbour
distance; nodes outside the extent have their CELL KEY clamped to the
boundary cell (cosmos clamps positions; we clamp the key, never the
position) and are excluded from the near field's Horvitz-Thompson count so
they cannot inflate a boundary cell; the far field treats the coarsest
level's boundary cells as usual (they are far, so the centroid approximation
is fine). Add to 11.4 the fixture "giant component + 1% isolated nodes and
100 small components, after 200 exact iterations (so the dust has flown)",
require RMS <= 5% on it, and add the case to R-3. Add a `stats.outsideGrid`
count next to `maxCellOccupancy`.

### PERF-2 (major) -- section 7.7 kernel table, lines 1570-1571 (G6, G7); 7.4 line 1424; 7.21 line 1956

Claim: dispatching the far-field and near-field kernels in node-index order
("per node: recompute its own finest cell from positions and state") makes
the dominant grid-tier kernel 4-60x slower than dispatching over the sorted
order, because the 32 lanes of a subgroup then walk 32 unrelated cells with
different loop lengths and touch 32 unrelated memory regions.

Evidence: near-field-order.mjs at 1M nodes on the 4070 (Dawn): index order
8.7 ms (uniform), 34.0 ms (100 Gaussian clusters), 125.4 ms (clusters + 0.2%
outliers) versus sorted order 1.9 / 2.1 / 2.1 ms with the SAME kernel body
and `i = sortedIdx[t]` as the only change; far-field-order.mjs: 2.2-2.6 ms
index vs 1.0 ms sorted at 1M. Note 03 section 8.3 item 5 explicitly
recommended using the cell-sorted order as Burtscher's "kernel 4" locality
sort and the plan dropped it. With index order the plan's "1M nodes 30-80
ms/iter" (line 1583) is spent almost entirely in G7 on a clustered layout and
T-6's 100 ms is exceeded on the outlier case; with sorted order the whole
grid repulsion is ~3-5 ms plus the sort.

Fix (text): G6 and G7 dispatch over the SORTED order: "thread `t` handles
node `i = sortedIdx[t]`; its cell is `sortedKeys[t]` (no recompute); `force`
is written at `3 * i`". Keep the `sortedKeys` buffer alive after the sort
(4n bytes, already in the 7.3 budget as `cellKey`). Note the follow-up from
note 03 8.3 item 5 (permuting `positions` / `mass` into cell order every k
iterations so the attraction gather is also spatially coherent) as a P4
decision-record item, not v1. Re-bracket line 1583's 1M figure after G4.

### PERF-3 (major) -- section 7.17, lines 1785-1790

Claim: the settle rule normalises mean displacement by `radius = max |p -
centroid|`, which any isolated node makes enormous (PERF-1), so `settled` is
reported while the core is still moving; conversely the near field's
per-iteration Horvitz-Thompson resampling injects noise that keeps
`meanDisplacement` above threshold for nodes in over-capacity cells, so on the
other side of PERF-1 the layout never settles before `maxIter`.

Evidence: `r = 4.2e7` for one isolated node at the 1M tier (PERF-1
arithmetic); `settleThreshold * radius = 4.2e4` units per node per iteration
against core displacements of O(1-100). Note 03 section 1.3 documents the
0.46 units/tick random walk cosmos saw from resampling.

Fix (text): normalise by the RMS radius (or the 90th-percentile radius) that
PERF-1 adds to the partials; state that the settle window compares the
displacement of nodes NOT in over-capacity cells, or that `nearMax` is raised
adaptively (cosmos's "adaptive K") when `maxCellOccupancy > nearMax` persists
for `settleWindow` iterations.

### PERF-4 (major) -- section 8.2, lines 2028-2031; section 6 row 9, line 1257; section 13 P7 gate, line 3166

Claim: the PageRank pull kernel is said to fit "exactly 8" storage buffers,
but the kernel as specified needs 9-10: the in-degree tiers ("tiered by
IN-degree, `degreeOrder({ of: "reverse" })`", 6 row 9; "`degreeOrder({ of:
"reverse" })` tiers" in P7) require the `perm` array as a binding (the
attraction kernel in 7.4 counts it: 7 bindings including `perm|dummy`), and
the "16-byte STORAGE block the next iteration reads" for the dangling mass
and delta is a further binding unless it is a region of `partials`. The plan's
own rule (3.5, "a kernel that needs more than 8 storage buffers in one stage
is SPLIT") and the P7 descriptor test would then fail, or the tiers (the
load-balancing that makes hub rows tractable) would be dropped.

Evidence: 8.2 lists revRowPtr, revColIdx, revWeights, outWeightSum, rankIn,
rankOut, personalization-or-dummy, partials = 8, with no `perm` and no
dangling/delta block; 7.4 K2 lists `perm|dummy` as a binding for the same
tiered segmented-reduce shape.

Fix (text): count `perm` and the dangling/delta block; keep the kernel at 8
by (a) placing the dangling / delta scalars in the first 16 bytes of
`partials` (offset binding), and (b) binding `personalization` only in the
personalized variant, where `outWeightSum[u]` can be pre-divided into a
`rankIn / outWeightSum` scratch by the previous iteration's finalize (one
extra tiny pass) so `outWeightSum` is not bound in the pull; or state that
personalized PageRank is the one kernel that requests `maxStorageBuffersPerShaderStage
>= 10` when the adapter has it and runs the untiered variant otherwise.

### PERF-5 (major) -- section 6 row 7, line 1255; section 8.4, lines 2059-2060; section 10.1, line 2578

Claim: the two-phase BFS materialises an edge frontier into a buffer of
`min(A, 16M)` entries with no stated behaviour when a level's degree sum
exceeds it; at the 1M / 10M tier (A = 20M) the peak level of an R-MAT graph
touches more than 16M arcs, and at 10M / 100M every large level does, so the
workhorse traversal either overflows (wrong result) or must read back per
level (contradicting D17).

Evidence: A = 20M at the tier the plan calls interactive (7.21 row "1M /
10M (20M)"); R-MAT / power-law BFS visits 60-80% of arcs in its two peak
levels (Beamer SC12, the reason direction-optimizing exists), i.e. 12-16M
arcs in one level; the 10.1 table budgets "+ 64 MB edge queue" (16M x 4 B)
with no overflow path.

Fix (text): specify the overflow rule: the expand kernel appends with the
workgroup-granular `atomicAdd` and CLAMPS the queue length; the finalize
kernel compares the unclamped total with the capacity and, when exceeded,
re-dispatches the expand for the remaining source range (a device-side
`chunkStart` counter and one indirect slot per chunk, at most `ceil(A /
16M)` chunks recorded per level), so correctness never depends on the cap;
or size the buffer from `snapshot.degreeOrder()` (the max possible degree
sum of any frontier is A) with `E_TOO_LARGE` when `A * 4 > maxBufferSize`.
Add "a level whose degree sum exceeds the edge-frontier buffer" to the G8
gate.

### PERF-6 (major) -- section 7.7, line 1576; section 14.1 R-3, line 3209

Claim: the documented runtime mitigation for a pathological grid ("raise
`nearMax` or lower `gridMax`") is inverted: lowering `gridMax` makes the
finest cells LARGER and raises occupancy, which is the failure being
mitigated; the text is carried unchanged from draft A line 971.

Evidence: occupancy per finest cell is `n / G^2`; `G = clamp(nextPow2(2
sqrt(n)), 8, gridMax2D)`, so lowering `gridMax2D` from 512 to 256 quadruples
mean occupancy (grid-occupancy.mjs: 1M nodes at G = 512 already has mean
occupancy 3.8 and max 62 on a clustered core).

Fix (text): "raise `nearMax`, or raise `gridMax2D` (the 512 cap is cosmos's
WebGL texture limit, not a compute limit: 2048^2 finest cells cost 67 MB +
22 MB of pyramid and add two levels = 54 far-field evaluations per node)";
and make the 2D default `gridMax2D = min(2048, nextPow2(2 sqrt(n)))` so the
1M tier is not saturated by default. Keep `gridMax3D = 128` (memory).

### PERF-7 (minor) -- section 7.3, line 1365; 7.5 `load_pos`; 7.6 line 1501

Claim: storing the DEVICE positions as stride-3 `array<f32>` with a separate
`mass` buffer costs four scalar loads per gathered node in the hottest loops
(tile fill, near field, attraction) where one 16-byte `vec4` load would do;
the plan already converts on the device (`toScene`, 7.18) so the owner's
stride-3 contract (design 14.3) is not what forces the internal layout.

Evidence: near-field-order.mjs, sorted order, stride-3 + mass vs packed vec4:
1M uniform 1.90 -> 0.93 ms, 262k clustered 0.41 -> 0.31 ms, 262k uniform
0.31 -> 0.22 ms (the 1M + outliers case was within noise, 2.09 vs 2.55).
Design 10.2 forbids `array<vec3<f32>>` for a `components: 3` COLUMN; it says
nothing about the simulation's own scratch, and `components: 4` "may be read
as `array<vec4<f32>>`".

Fix (text): 7.3: `positions` on the device is `array<vec4f>` (xyz + mass,
16n bytes), converted from / to the owner's stride-3 array in `load()` and
`toScene`; `mass` as a separate buffer disappears from K2 / K3 / G4 / G6 /
G7 (one binding fewer each). Note the trade: +4 B/node.

### PERF-8 (minor) -- section 7.6, lines 1528-1530; 7.8, lines 1604-1606; 7.21, lines 1934, 1955; 10.3, lines 2604-2605

Claim: "3.6e11 pair evaluations per second [M]" is quoted as a constant and
used to extrapolate 100k and 1M exact-tier costs, but the measured throughput
is occupancy-dependent: the 20k probe runs 79 workgroups on 56 SMs.

Evidence: exact-tile.mjs, FA2 body: 0.6e11 pairs/s at 4k, 1.3e11 at 8k,
2.4e11 at 16k, 3.1e11 at 32k, 4.9e11 at 65k, 5.7e11 at 100k. Consequences:
100k exact is 17.5-17.8 ms, not "28-36 ms" (line 1955) / "30-38 ms" (line
2604); 1M exact is ~1.8 s, not "2.8-3.6 s" (line 2605); 16k is 1.13 ms,
above the "0.7-1.0 ms" of line 1604 (T-4's 2 ms still holds); 32k is 2.7-3.5
ms, inside the "2.8-3.7 ms" of line 1605. The 1.3x body factor is confirmed
(1.1-1.6x).

Fix (text): replace the constant with the measured curve (n, ms, pairs/s)
and label it "[M], occupancy-limited below ~65k"; correct the 100k and 1M
exact rows of 7.21 and 10.3.

### PERF-9 (minor) -- section 2.2, lines 330-335; 7.8, lines 1607-1612

Claim: `calibrate()` probes only 8k and 16k and picks "the largest probed n
whose exact iteration is under 4 ms", which (a) can never suggest 32k or 65k
even though the 4070 runs 32k exact in 2.7-3.5 ms, (b) never compares with
the grid cost it also measures (`gridMsPerIter16k`), so it is a frame-budget
rule, not a crossover, and (c) its "20-40 ms" cost omits compiling ~20
pipelines (exact, sort, scan, histogram, grid kernels) on first use, which in
Chromium is hundreds of ms, and Chromium's `performance.now()` granularity
(100 us unless cross-origin isolated) on a 0.25 ms probe.

Evidence: exact-tile.mjs sizes 8k / 16k / 32k / 65k: 0.51 / 1.13 / 2.7-3.5 /
8.8-9.2 ms; the 7.8 mechanical rule for the P3 gate uses 1k-65k while
`calibrate()` uses two sizes.

Fix (text): probe 8k / 16k / 32k / 65k with 10 timed iterations after a
warm-up submit (pipelines compiled by `PipelineCache.warm` first, excluded
from the timing); `suggestedExactMaxNodes = largest n with exactMs(n) <=
min(4 ms, gridMs(n))` with `gridMs` measured at the same sizes; document the
first-call cost as "compile + ~50-100 ms".

### PERF-10 (minor) -- section 11.4, lines 2744-2749; 13 P4 gate; 10.4 T-12

Claim: "from the same start, 200 iterations exact vs grid ... sizes 20k,
100k, 262k and 1M (hardware only)" costs 200 x 1.8 s = 6 minutes of exact
iterations at 1M for ONE test inside the GPU lane's 20-minute budget (T-12),
and 200 x 0.12 s = 24 s at 262k.

Evidence: exact-tile.mjs 100k = 17.5 ms scaled by n^2 at the measured
5.7e11 pairs/s: 1M = 1.75 s per iteration.

Fix (text): at 1M run the one-iteration force-field comparison only (RMS /
p99); run the 200-iteration distributional comparison at <= 262k; put the 1M
200-iteration run in the nightly `bench` project.

### PERF-11 (minor) -- section 7.7 G4, line 1568; section 6 row 12

Claim: "(+ a workgroup-per-cell dispatch for cells above 1,024 entries)" has
no dispatch mechanism: the host does not know which cells exceed 1,024
without a readback, so the tier needs either a device-side compaction of
hub cells plus an indirect dispatch (an extra scan + compact per iteration)
or a fixed dispatch over all 262,144 cells where most workgroups exit
immediately; with PERF-1 unfixed a single cell can hold 1M entries and the
thread-per-cell loop then runs 1M serial iterations.

Evidence: 5.4 defines indirect dispatch only for frontier kernels; G4's row
names a second dispatch without an args source.

Fix (text): "G4a: `compact` the cells with `count > 1024` into a hub-cell
list with a device-side count and indirect args (reuses the 5.4 finalize);
G4b: workgroup-per-cell over that list; G4 skips cells above 1,024 entries".
Count the two extra dispatches in 7.4's total.

### PERF-12 (minor) -- section 8.4, lines 2059-2061 and 2066-2077; section 6 row 8, line 1256

Claim: inside a 32-level batch with one readback per 32 levels, the choice
between the fused expand-contract kernel ("frontiers below 4,096 entries")
and the two-phase kernels, and the direction-optimizing top-down /
bottom-up switch, must be taken on the device; the plan states the
thresholds but not the mechanism, and a host decision per level would
reintroduce the per-level `mapAsync` it rules out.

Evidence: lines 2072-2074 ("32 levels per submit ... one 4-byte readback of
the frontier length every 32 levels") versus lines 2060-2061 and 2066-2069
(size- and degree-sum-based variant selection).

Fix (text): "the `finalizeArgs` kernel of 5.4 writes the indirect args of
exactly ONE variant per level (fused / two-phase top-down / bottom-up) and
`(0, 0, 1)` for the others, using the frontier count and the degree sum the
advance's workgroup scans already produce; the `switches` counter is
incremented on the device".

### PERF-13 (minor) -- section 8.4, lines 2115-2118; 10.3, lines 2604-2605; Q-13

Claim: "256 sampled sources at 1M / 10M is 20-60 s" contradicts the plan's
own bandwidth framing: a BFS plus a backward sweep over 20M arcs moves ~2 x
20M x ~16 B = 640 MB per source, i.e. 160 GB for 256 sources = 0.3-0.5 s at
the 4070's ~500 GB/s, and the batched multi-source form (64 sources per
traversal) amortises the per-level dispatch overhead the plan says
dominates; 20-60 s implies 80-230 ms per source, 2-3x the plan's own
whole-run BFS number, times 256, with none of the batching credited.

Evidence: line 2115 "batching amortises dispatch latency, not bandwidth";
line 2605 BFS whole run 30-100 ms; far-field-order.mjs measured 2-6 us per
dispatch, so 20 levels x ~8 dispatches x 2 passes = 0.6-2 ms of overhead per
source before batching.

Fix (text): give the per-source model explicitly (`levels x dispatches x
overhead + 2 x A x bytes / bandwidth`, batched by k), quote 2-10 s at 1M /
10M and 0.2-1 s at 100k / 1M as the [X] range, and let Q-13's product default
(sampled above 50k) rest on the measured T-11, not on the pessimistic figure.

### PERF-14 (minor) -- section 7.19, lines 1869-1873; 10.2, line 2592; 10.4 T-5

Claim: the Chromium readback model (2.65 ms per MiB, so 12 MB ~30 ms) rests
on a single 1 MiB datapoint while note 05's other Chromium datapoint (1M-element
compute + 4 MiB readback: 20.5 ms, line 227) implies ~5 ms per MiB; 12 MB may
be 30-60 ms, and the "raise `iterationsPerStep` above ~250k nodes" threshold
may need to be ~100k.

Evidence: note 05 lines 227 and 517; T-5 measures only 10k and 100k.

Fix (text): cite both datapoints; add "12 MB readback in Chromium" to T-5
(G4) and derive the `iterationsPerStep` auto-raise threshold (P12) from it.

### PERF-15 (minor) -- section 7.2, line 1340; 7.7, line 1571

Claim: the hash expressions `lowbias32(i * 0x9E3779B9u ^ j)` and `lowbias32(cell
^ iteration * 0x9E3779B9u ^ seed)` are not valid WGSL: Tint rejects mixing
`*` and `^` without parentheses, and 7.2 says the constants are "ported to
WGSL verbatim".

Evidence: exact-tile.mjs first run: "Error while parsing WGSL: :10:68 error:
mixing '*' and '^' requires parenthesis".

Fix (text): `lowbias32((i * 0x9E3779B9u) ^ j)` and `lowbias32(cell ^ (iteration
* 0x9E3779B9u) ^ seed)`.

### PERF-16 (minor) -- section 7.21, line 1940; 10.3, lines 2607-2612

Claim: "integrated GPU = 8x slower than the 4070 on compute, 2x slower on
readback" understates the compute gap for the named parts and has the
readback sign wrong for unified-memory GPUs.

Evidence: RTX 4070 SUPER ~35 TFLOPS FP32 / 504 GB/s; Iris Xe 96 EU ~2.1
TFLOPS / ~60 GB/s shared (17x / 8x); Apple M1 base ~2.6 TFLOPS / 68 GB/s
(13x / 7x); the exact tile is compute-bound (probe: 5.7e11 pairs/s is ~9
TFLOPS-equivalent), the gathers are bandwidth- and cache-bound with no large
L2 on integrated parts; a UMA `mapAsync` has no PCIe copy, so readback is
not slower there (Chromium's IPC copy dominates either way).

Fix (text): "10-17x slower on the exact tile, 5-10x on gathers, readback
roughly equal" and derive the integrated rows of 7.21 / 10.3 from that; keep
them [X].

## Checks that passed (no finding)

- Pyramid arithmetic (349,520 cells = 5.59 MB in 2D; 2,396,736 = 38.3 MB in
  3D), far-field evaluation counts (196 / 982) and the exact-once tiling
  logic (a child's 3x3 lies inside its parent's 3x3 block) are correct.
- Byte model (4.7, 10.1) matches design 15.1; no O(n x cells) work; no host
  round trip inside the FA2 iteration loop; no per-frame buffer creation
  (uniform ring, pool leases, generation counter); the single-workgroup K1 /
  K4 finalizers cost tens of microseconds at 1M (3,907 partials) and are not
  a bottleneck; per-dispatch overhead measured 2-6 us, so 30-45 dispatches
  per grid iteration cost < 0.3 ms.
- `foldArcs` / `renumberPartition` are O(A) / O(n) CPU passes per algorithm
  call (packages/graph-format/src/snapshot/derived.ts:1155), acceptable at
  the stated ~1 ms per 1M nodes.
- The Horvitz-Thompson estimator with a uniformly random circular offset is
  unbiased for the cell sum regardless of the within-cell order (every entry
  has inclusion probability `s / count`), so "unbiased" in 7.7 is right; the
  variance, not the bias, is the problem PERF-1 and PERF-3 address.
- PageRank ~24 B/arc, T-8, T-9, T-10 and the 1000 x 1000 grid BFS budget are
  comfortable given the measured dispatch overhead.

## Overall confidence

The layout design is sound in its bones (batched submissions, device-side
speed controller, gather-only forces, tiered rows) and the exact tier's
numbers hold up (the probe reproduces 1.11 ms at 20k and the FA2 body is
1.1-1.6x). I would not bet on the grid tier as written: its cell size comes
from a bounding box that ForceAtlas2's own laws blow up with a single isolated
node (PERF-1), the settle rule inherits the same radius (PERF-3), and the
near-field kernel is dispatched in the one order that makes it 4-60x slower
than necessary (PERF-2). All three are text-level fixes with measured
evidence; until they are in, the 1M-node target and the P4 gate are only
credible on synthetic fixtures that lack dust. The single biggest risk is
PERF-1: the P4 gate as written cannot see it because 11.4 has no
isolated-node fixture for the layout.
