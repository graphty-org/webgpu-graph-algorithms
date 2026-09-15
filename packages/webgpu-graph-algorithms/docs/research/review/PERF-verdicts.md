# PERF verdicts: skeptical verification of the performance-realism review

Verifier for the lens "Performance realism". Document:
/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
(3328 lines). Every finding below was checked against the cited plan lines, the
research notes, the accepted design, and -- where a probe was cheap -- by
re-running the reviewer's probe or a new one on the RTX 4070 SUPER under
Dawn-in-Node (webgpu@0.4.0, `create(["adapter=NVIDIA"])`, `LD_LIBRARY_PATH`
to the extracted libEGL tree per the memory note).

Probes re-run or added by the verifier (all under
/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/probes/):

| Probe | Verifier result |
| --- | --- |
| grid-occupancy.mjs (re-run, CPU) | reproduces: 1M nodes, one node at 10x the core radius -> 95.5% of nodes in cells above nearMax (maxOcc 1,521); 100x -> 129 cells, maxOcc 33,597; 1000x -> 2 cells, maxOcc 999,999 |
| near-field-order.mjs (re-run, NVIDIA) | reproduces: 1M index order 8.43 / 32.58 / 125.26 ms (uniform / clustered / clustered + 0.2% outliers at 2-4x) vs sorted 1.80 / 2.01 / 2.94 ms; sorted + vec4 0.92 / 1.78 / 1.79 ms |
| far-field-order.mjs (re-run, NVIDIA) | reproduces: 1M index 2.83 / 2.26 ms vs sorted 1.01 / 1.00 ms |
| exact-tile.mjs (re-run, NVIDIA) | reproduces: FA2 body 4k 0.26 ms (0.64e11 pairs/s), 8k 0.53 (1.27e11), 16k 1.13 (2.38e11), 20k 1.81 (2.25e11), 32k 3.48 (3.08e11), 65k 8.66 (4.96e11), 100k 17.67 ms (5.67e11); probe body 20k 1.06 ms (the plan's 1.11) |
| verify-wgsl-mixing.mjs (new, llvmpipe) | Tint: `lowbias32(i * 0x9E3779B9u ^ j)` -> "mixing '*' and '^' requires parenthesis"; `lowbias32(cell ^ iteration * 0x9E3779B9u ^ seed)` -> same error; both parenthesised forms compile |
| clamp-expansion.mjs (new, CPU) | exact FA2 with the 7.2 laws, 1,000 nodes, avg degree 10, seeded in [-1, 1): bbox extent WITHOUT the clamp 2.0 -> 554 by iteration 25 (equilibrium ~575); WITH the plan's `2 * cellSize` clamp (cellSize = extent / 512): 2.0 -> 2.4 (it 25) -> 4.4 (it 100) -> 9.6 (it 200) -> 20.9 (it 299), i.e. growth capped at (1 + 4/G) = 0.78% per iteration |

## Verdicts

### PERF-1 -- CONFIRMED (blocker)

Plan lines 1552-1553 (finest grid G and `cell size = max extent / G` from
`state.min / max`, recomputed every iteration, "never a fixed spaceSize --
cosmos clamps positions, we do not"), 1571 (G7 near field bounded by
`nearMax = 64` with Horvitz-Thompson scaling), 1785-1790 (settle rule),
2740-2749 (11.4 fixtures: uniform, clumpy, hub, two cosmos cases, colinear,
coincident -- no isolated-node / small-component fixture).

Physics check: with the 7.2 reference laws (repulsion magnitude `k m_i m_j /
d`, regular gravity of constant magnitude `g m_i`, mass = degree + 1) an
isolated node or a small component sees net repulsion `k m M / r` from a core
of total mass `M = sum(deg + 1)` and constant gravity `g m`, so its
equilibrium is `r = k M / g` (= 2 x 2.1e7 = 4.2e7 layout units at the 1M / 10M
tier with E = 10n), while the core's own radius is set by attraction vs
repulsion at roughly `sqrt(k m M / deg)` ~ 1e4. Gephi users see exactly this
("isolated nodes fly to the edge"), and the CPU rewrite adopts the same laws.
The cosmos reference avoids the problem only because it clamps positions into
a fixed `spaceSize` (repos/cosmos/src/modules/Points/update-position.frag
lines 56-57), which the plan explicitly rejects; cuGraph's Barnes-Hut adapts
because a quadtree deepens under an outlier, a uniform grid does not.

The occupancy probe reproduces the collapse, and near-field-order.mjs shows
the cost on a realistic clustered 1M layout with only 0.2% of nodes at 2-4x
the core radius: 88.5% of nodes in over-capacity cells and 125 ms per
near-field dispatch in index order. Nothing in the plan (grep for isolated /
dust / outlier / extent) handles it; `stats.maxCellOccupancy` only reports it;
option B is reserved for "pathological distributions", but graphs with dust
are the normal case.

Fix (revised): keep the reviewer's direction but make the extent
`min(bboxExtent, c * rmsRadius)` with `c` ~ 6 (the bbox wins on a clean
uniform or Gaussian core, the RMS bound wins under outliers; a plain `c *
rmsRadius` would quadruple core occupancy on a uniform disk because RMS =
R / sqrt(2)); the sum of `|p - c|^2` fits in the unused `.w` lane of the
existing `sum p` vec4 partial (no change to the 64 B partial stride); nodes
outside the extent have their CELL KEY clamped to the boundary cell and are
excluded from that cell's Horvitz-Thompson `count`; add the "giant component
+ 1% isolated nodes + 100 small components after 200 exact iterations"
fixture to 11.4 with RMS <= 5% and p99 <= 25%, add the case to R-3, add
`stats.outsideGrid`. Must be designed together with MISSED-1 below (the
clamp), because the two interact: a robust extent shrinks `cellSize` and
therefore the clamp.

### PERF-2 -- CONFIRMED (major), fix revised

Plan lines 1570-1571 (G6 / G7 dispatch `ceil(n / WG)` with "per node:
recompute its own finest cell from positions and state"), 1424 (dispatch
count), 1583 (1M 30-80 ms/iter [X]). Note 03 lines 660-680 (8.3 item 5)
recommends the cell-sorted order as Burtscher's locality sort; the plan's
G6 / G7 rows walk nodes in index order.

Reproduced on the 4070: at 1M nodes the SAME near-field kernel body costs
8.4 / 32.6 / 125.3 ms in index order vs 1.8 / 2.0 / 2.9 ms with
`i = sortedIdx[t]`; the far field 2.3-2.8 ms vs 1.0 ms. In index order the
plan's T-6 target (1M <= 100 ms) fails on the clustered + outliers case from
the near field alone, and the "30-80 ms" bracket is spent almost entirely in
G7 on a clustered layout.

Fix (revised so it does not break the 8-binding budget): "G6 and G7: thread
`t` handles node `i = sortedIdx[t]`; the cell is recomputed from
`positions[i]` and `state` exactly as now (no `sortedKeys` binding -- G7 is
already at 8 storage buffers and the probe's sorted variant recomputes the
cell too); `force` is written at `3 * i`". Record note 03 8.3 item 5's
position permutation (every k iterations) as a P4 decision item, not v1.
Re-bracket line 1583 and the 7.21 / 10.3 grid rows after G4: with the sorted
order the whole repulsion at 1M is ~3-5 ms plus the sort / build, so the
current 30-80 ms is conservative by an order of magnitude rather than wrong.

### PERF-3 -- CONFIRMED (major)

Plan lines 1785-1790: `settled` when `meanDisplacement <= settleThreshold *
radius`, `radius = max |p - centroid|` from the bounding box. With the
PERF-1 physics one isolated node makes `radius` thousands of core radii, so
the threshold (1e-3 x radius) exceeds the core's per-iteration displacement
and `settledCount` reaches `settleWindow` while the core is still moving;
the element then stops stepping (7.19: `LayoutManager.step()` only while
`!isSettled`). The arithmetic is correct; the first half of the claim is
solid. The second half (Horvitz-Thompson resampling noise keeps
over-capacity cells from settling) is plausible -- FA2's `speedEfficiency`
floors at 0.05 and speed converges to a jitter-bounded value rather than
zero, so a noisy force never yields zero displacement -- but is not
measured; keep it as a stated risk, not a fact.

Fix: normalise by the RMS radius that PERF-1 adds (one word once that
partial exists); state that `meanDisplacement` is over free nodes NOT in
over-capacity cells, or raise `nearMax` adaptively when
`maxCellOccupancy > nearMax` persists for `settleWindow` iterations, and add
the isolated-node fixture to the settle test.

### PERF-4 -- CONFIRMED (major)

Plan lines 2028-2031 list 8 bindings (revRowPtr, revColIdx, revWeights,
outWeightSum, rankIn, rankOut, personalization-or-dummy, partials) and say
"the dangling sum and the delta are reduced by a tiny kernel into a 16-byte
STORAGE block the next iteration reads". Line 1257 (row 9): spmvPull is
"segmentedReduce specialised ... tiered by IN-degree (`degreeOrder({ of:
"reverse" })`)"; row 3 (line 1251): tiered rows are "visited through
`degreeOrder(opts).perm` with `override USE_PERM`"; 3.5 (lines 797-806):
group 0 is `rowPtr, colIdx, weights or dummy, perm or dummy` and "a kernel
that needs more than 8 storage buffers in one stage is SPLIT"; line 3166:
G7 gate "exactly 8 storage bindings in the PageRank kernel (descriptor
test)". A dummy still occupies a layout entry (the plan uses explicit
layouts so the layout is fixed), so the tiered pull is 9 with `perm|dummy`
and 10 if the dangling / delta block is its own buffer. The plan contradicts
itself and the gate would fail or the tiers would be dropped.

Fix (as the reviewer's, made concrete): count `perm|dummy` in group 0; get
back to 8 by (a) storing dangling / delta in the first 16 B of `partials`
(offset binding, already bound) and (b) having the per-iteration finalize
write `xNorm[u] = rankIn[u] / outWeightSum[u]` (one O(n) map, the ping-pong
buffer's spare region) so `outWeightSum` is not bound in the pull: revRowPtr,
revColIdx, revWeights, perm|dummy, xNorm, rankOut, personalization|dummy,
partials = 8. Update 3.5 and the G7 gate text to name these 8.

### PERF-5 -- CONFIRMED (major), claim narrowed

Plan line 1255 ("an edge-frontier buffer of `min(A, 16M)` entries for the
two-phase expansion"), lines 2059-2061 (two-phase as the workhorse), line
2578 ("+ 64 MB edge queue" at 1M / 10M, no overflow path), 10.3 row 10M /
100M BFS "0.3-1 s". Row 8 (line 1256) appends with a workgroup-granular
`atomicAdd` on the queue counter; nothing states what happens when a level's
degree sum exceeds the capacity -- under WGSL's bounds clamping the excess
writes are silently dropped and `depth` is wrong. The reviewer overstates
the 1M / 10M case: with A = 20M and a peak level carrying 50-70% of the
edge examinations (Beamer SC12), the peak is 10-14M, under 16M; but any 1M
graph denser than E = 12n, and every 10M / 100M level, overflows, and
correctness must never depend on the cap. The direction-optimizing variant
would run the peak levels bottom-up, but the plain two-phase BFS is the
default workhorse and closeness / BC ride on the same machinery.

Fix (as the reviewer's): specify the overflow rule -- the expand clamps its
append and the finalize compares the unclamped total with the capacity and
re-dispatches expand for the remaining source range (device-side `chunkStart`,
one indirect slot per chunk, at most `ceil(A / capacity)` chunks per level);
or size the queue at A entries when `A * 4 <= maxBufferSize` (80 MB at 1M /
10M is trivial on the 4070) and chunk only above that. Add "a level whose
degree sum exceeds the edge-frontier capacity (faked capacity of 4,096)" to
the G8 gate.

### PERF-6 -- DOWNGRADED to minor, fix narrowed

Plan line 1575-1577: "`stats.maxCellOccupancy` reports the largest finest
cell so the element can raise `nearMax` or lower `gridMax`". With occupancy
= n / G^2 and G capped by `gridMax2D`, lowering `gridMax` enlarges the
finest cells and RAISES occupancy; the sentence is inverted (real, one
line). R-3 (line 3209) does not repeat the inverted lever, so the damage is
one sentence. The second half of the finding -- make `gridMax2D = 2048` the
default at 1M -- is a tuning proposal, not a defect: 7.7 already says the
cost model is "unverified until P4 measures it", the 512 cap is a memory /
far-field-work trade (2048^2 = 67 MB finest + 22 MB pyramid + 16 MB
`cellStart`, 54 more far-field evaluations per node), and a larger G makes
the clamp problem of MISSED-1 worse (`2 * cellSize` shrinks with G). Leave
the cap to the P4 decision record.

Fix: rewrite the sentence as "raise `nearMax`, or raise `gridMax2D` (finer
cells; costs memory and far-field evaluations)"; add "re-check `gridMax2D`
in {512, 1024, 2048} at 1M" to the P4 gate's decision record.

### PERF-7 -- CONFIRMED (minor)

Plan line 1365 (device positions as stride-3 `array<f32>` "even in 2D, so
one buffer layout serves both dimensions and the element's stride-3 array
copies in with one `set`"), 1501 (`load_pos` three scalar loads), 7.18
(`toScene` already converts on the device). Design 10.2 (graph-format-design
lines 2360-2362) forbids `array<vec3<f32>>` for a `components: 3` COLUMN and
allows `array<vec4<f32>>` for four components; it says nothing about the
simulation's own scratch. Reproduced: sorted-order near field 1M uniform
1.80 -> 0.92 ms, clustered 2.01 -> 1.78, clustered + outliers 2.94 -> 1.79
with one packed vec4 (xyz + mass). The one-`set` argument in 7.3 is weak
because `load()` already inverse-scales every finite row on the CPU (7.18),
so a 12n -> 16n repack is free. Extra benefit: dropping the separate `mass`
binding frees one storage slot in K2 / K3 / G4 / G6 / G7 (G7 is at the
8-slot cap). Cost: +4 B/node.

Fix: as the reviewer's; note it as a P3 decision (the exact tile packs the
tile as vec4 already, so the change is confined to `load()`, `setPosition`,
`toScene` and the gathers).

### PERF-8 -- CONFIRMED (minor)

Plan lines 1528-1530 ("3.6e11 pair evaluations per second [M]"), 1604-1606
(7.8: "exact at 16k nodes at ~0.7-1.0 ms and at 32k at ~2.8-3.7 ms"), 1934
(7.21 basis "E = 3.6e11 pairs/s ... with the 1.3x FA2-body factor"), 1955
(100k exact "28-36 ms"), 2604-2605 (10.3: 100k "30-38 ms", 1M "2.8-3.6 s").
Reproduced: the FA2 body's throughput is 0.64e11 pairs/s at 4k rising to
5.67e11 at 100k (the 20k probe fills 79 workgroups on 56 SMs), so 100k exact
is 17.7 ms and 1M ~1.8 s (the plan is 1.6-2x pessimistic there) while 16k is
1.13 ms (slightly above the plan's 0.7-1.0). The 1.3x body factor holds
(1.14-1.7x, mean ~1.3). Consequences are limited: T-4 (16k <= 2 ms) still
holds and `exactMaxNodes` is re-fixed by measurement at G3.

Fix: replace the constant with the measured (n, ms, pairs/s) curve labelled
"[M], occupancy-limited below ~65k"; correct the 100k / 1M exact rows in
7.21 and 10.3; fix 7.8's 16k figure to ~1.1-1.3 ms.

### PERF-9 -- DOWNGRADED (minor; part of the claim refuted), fix narrowed

Plan lines 328-335 (2.2 `calibrate()`: "runs the exact-tile repulsion kernel
at 8k and 16k nodes and the grid build at 16k once (about 20-40 ms on a
discrete GPU) ... the largest probed n whose exact iteration is under 4 ms")
and 1607-1612 (7.8's P3 rule measures 1k-65k). The claim "can never suggest
32k or 65k" is only true of the DEFAULT sizes: the signature is
`calibrate(options?: { sizes?: number[] })` (line 328), so the app can probe
32k / 65k today. What stands: the default rule is a frame-budget rule that
never compares the exact cost with the grid cost it also measures
(`gridMsPerIter16k` is returned but unused by the suggestion), and the
"20-40 ms" omits first-use pipeline compilation of the grid build's ~15-20
pipelines and Chromium's 100 us timer quantisation (5.5 already documents the
quantisation).

Fix: default `sizes = [8k, 16k, 32k, 65k]`, 10 timed iterations after
`PipelineCache.warm` and one warm-up submit; `suggestedExactMaxNodes` =
largest probed n with `exactMs(n) <= min(4 ms, gridMs(n))` with `gridMs`
measured at the same sizes; document the first-call cost as "pipeline
compilation + 50-100 ms".

### PERF-10 -- CONFIRMED, severity RAISED to major

Plan lines 2740-2749 (11.4 exact-vs-grid: the fixture list, "sizes 20k,
100k, 262k (finest-grid saturation) and 1M (hardware only)", "from the same
start, 200 iterations exact vs grid agree on the distributional metrics
within 15%"), line 3163 (G4 requires "11.4 exact-vs-grid in full"), T-12
(line 2636, GPU lane <= 20 min). The reviewer counted one test; the text
applies the 200-iteration comparison to every scalable fixture at every
size. At 1M, six scalable fixtures x 200 exact iterations x ~1.8 s (measured
5.67e11 pairs/s) is ~36 minutes of oracle time, plus ~2.5 minutes at 262k,
inside a 20-minute lane: the P4 gate is self-contradictory as written, not
merely expensive.

Fix: at 1M run only the one-iteration force-field comparison and the
32-iteration unbiasedness check (grid iterations are cheap; one exact
iteration is 1.8 s); run the 200-iteration distributional comparison at
<= 262k; move the 1M 200-iteration run to the nightly `bench` project and
say so in G4 and T-12.

### PERF-11 -- CONFIRMED (minor)

Plan line 1568 (G4 "`ceil(cells / WG)` (+ a workgroup-per-cell dispatch for
cells above 1,024 entries)") and line 1259 (row 12 "the workgroup-per-cell
tier handles hub cells above 1,024 entries"). 5.4 (lines 1160-1173) defines
indirect dispatch and the `finalize` kernel for frontier kernels only; no
kernel produces a list of over-full cells, and the host cannot know them
without a readback (D15 / D17 forbid one inside the batch). Without a
mechanism the thread-per-cell loop is serial over the largest cell -- up to
1M dependent gathers on one thread in the PERF-1 collapse (hundreds of ms).

Fix: as the reviewer's: G4a compacts cells with `count > 1024` into a
hub-cell list with a device count and indirect args (the 5.4 finalize); G4b
is workgroup-per-cell over that list; G4 skips those cells; count both in
7.4's dispatch total. A fixed dispatch over all cells with an early exit is
the acceptable cheaper alternative at 262,144 cells (2D) but not at 2.1M
(3D); say which.

### PERF-12 -- CONFIRMED (minor)

Plan lines 2059-2061 (fused expand-contract "for frontiers below 4,096
entries"), 2062-2069 (direction-optimizing switch on degree sums), 2070-2073
("Host loop: 32 levels per submit with indirect args (5.4), one 4-byte
readback of the frontier length every 32 levels"). Both selections depend on
per-level device-side quantities, but the plan does not say who selects;
5.4's finalize writes one `(x, y, 1)` per round for one pipeline. A host
choice per 32-level batch would use the wrong kernel for most of a batch on
an RMAT graph (levels 0-1 tiny, 2-3 huge), and a host choice per level would
reintroduce the per-level `mapAsync` the plan rules out.

Fix: as the reviewer's: `finalizeArgs` writes the indirect args of exactly
one variant per level (fused / two-phase top-down / bottom-up) and `(0, 0,
1)` for the others from the frontier count and the degree sum the advance's
workgroup scans already produce; `switches` is a device counter read with the
final result.

### PERF-13 -- DOWNGRADED (minor; the "contradiction" is a pessimism, the Q-13 impact is refuted)

Plan lines 2115-2118 ("batching amortises dispatch latency, not bandwidth, so
256 sampled sources at 100k / 1M is 1-5 s and at 1M / 10M 20-60 s [X]"),
2604-2605 (10.3 BFS whole run 30-100 ms at 1M / 10M), 2635 (T-10 basis "[P]
Merrill 3.3 GTEPS (2011)"), 3246 (Q-13). The BC figure is NOT unsupported:
it is 256 x 2 x the plan's own 30-100 ms BFS figure. What is inconsistent is
that the BFS figure is 5-15x above T-10's own basis (20M arcs / 3.3 GTEPS = 6
ms) and above the bandwidth bound (~1 ms), and BC inherits that. A realistic
range is 2-10 s at 1M / 10M (BFS kernels reach 10-30% of peak bandwidth
under atomics and dedupe). The Q-13 product default (exact <= 10k, sampled
above 50k) is correct at either figure because exact BC is O(n m), so the
"drives Q-13" part is refuted.

Fix: show the per-source model (`levels x dispatches x overhead + 2 x A x
bytes / bandwidth`, batched by k) next to the figure, align the 10.3 BFS row
with T-10's basis, quote BC as 2-20 s [X] and let T-11 settle it.

### PERF-14 -- REFUTED

Plan lines 1869-1873 (12 MB ~30 ms in Chromium from 2.65 ms per MiB), 2592,
2635 (T-5). The reviewer's second datapoint is misread: note 05 line 227
("1M-element compute + 4 MiB readback: submit 0.09 ms, `onSubmittedWorkDone`
20.5 ms") is the Mesa LLVMPIPE column of the Dawn-node adapter table (header
at note 05 lines 209-211: "NVIDIA RTX 4070 SUPER (Dawn-node)" vs "Mesa
llvmpipe 23.2.1 / LLVM 15 (Dawn-node)"), i.e. a software GPU running the
compute, not a Chromium readback. Note 05 section 7.2 (line 517) has exactly
one Chromium copy datapoint (1 MiB: 2.65 ms NVIDIA, 3.83 ms SwiftShader) and
the plan cites it as such with the 12 MB figure labelled by extrapolation.
There is no evidence for "~5 ms/MiB" or "30-60 ms". Adding a 12 MB Chromium
measurement to T-5 would be sensible housekeeping but is not a defect: the
1M-node browser row already says "readback-bound -> `k` >= 4; batch use".

### PERF-15 -- CONFIRMED (minor)

Plan line 1340 (`lowbias32(i * 0x9E3779B9u ^ j)`) and 1571
(`lowbias32(cell ^ iteration * 0x9E3779B9u ^ seed)`), with 7.2 promising the
constants are "ported to WGSL verbatim". Verified with
verify-wgsl-mixing.mjs on Dawn (Tint): both forms fail with "mixing '*' and
'^' requires parenthesis"; the parenthesised forms compile.

Fix: `lowbias32((i * 0x9E3779B9u) ^ j)` and `lowbias32(cell ^ (iteration *
0x9E3779B9u) ^ seed)`.

### PERF-16 -- CONFIRMED (minor), evidence strengthened

Plan line 1940 ("integrated GPU = 8x slower than the 4070 on compute
(GraphWaGu's Iris Xe ratios and note 03's 5-10x), 2x slower on readback
[X]"), 2607-2612. Two problems: (1) the cited basis does not exist -- note 03
line 297's "Integrated Iris Xe speedups 15.0x-35.2x" is GraphWaGu 2025 over
GraphWaGu 2022 on the SAME Iris Xe, not a discrete-vs-integrated ratio, and
note 03's "5-10x" (line 612) is an unsupported estimate; (2) public FP32
peaks (RTX 4070 SUPER ~35 TFLOPS / 504 GB/s; Iris Xe 96 EU ~2.1 TFLOPS /
~60 GB/s shared; Apple M1 ~2.6 TFLOPS / 68 GB/s) give 13-17x on the
compute-bound exact tile (measured compute-bound at 5.7e11 pairs/s) and 7-8x
on bandwidth-bound gathers, and a unified-memory `mapAsync` has no PCIe copy,
so "2x slower on readback" has no basis. All rows are labelled [X], so the
harm is calibration of the integrated-GPU fps expectations (10-15 fps at
100k becomes 5-8 fps).

Fix: as the reviewer's: "10-17x slower on the exact tile, 5-10x on gathers,
readback roughly equal", derive the integrated rows of 7.21 / 10.3 from that,
keep them [X], and drop the GraphWaGu citation for the ratio.

## Defects the reviewer missed

### MISSED-1 -- blocker -- 7.7 / 7.11 / 11.4 / 13 P4, lines 1573-1575, 1684, 1353, 1741, 2749

Claim: the per-iteration clamp `dp = clamp_length(dp, 2 * state.cellSize)`
(7.11 line 1684, described at 7.7 lines 1573-1575 as "cosmos's clamp,
applied where the displacement exists") is applied to the WHOLE displacement
in the grid tier while `cellSize = max extent / G` is recomputed from the
bounding box every iteration (line 1553). The bbox can therefore grow by at
most `4 * cellSize` per iteration, i.e. the layout's global expansion is
capped at `(1 + 4 / G)` = 0.78% per iteration at G = 512 (G is 512 for every
n >= 16.4k, so throughout the grid tier). The simulation seeds positions in
`[-1, 1)` (7.2 line 1353; 7.18) and the FA2 equilibrium extent is
`O(sqrt(k M / deg))` ~ 1e3-1e4 layout units at 100k-1M nodes, so a grid-tier
layout needs ~1,000 iterations just to expand (ln(1e3-1e4) / ln(1.0078) =
900-1,200) while `maxIter` defaults to 100 (line 1741, growth <= 2.2x). The
exact tier has no clamp, so the two tiers produce different pictures at the
same n, and the 11.4 gate "from the same start, 200 iterations exact vs grid
agree on the distributional metrics within 15%" and "no node moves more
than `2 * cellSize` per iteration" (line 2749) cannot both hold on ANY
fixture seeded from the LCG -- including uniform. The misreading of cosmos
is visible in the plan's own words: cosmos clamps only the NEAR-FIELD
velocity contribution inside `force-nearfield.frag` (lines 145-156: "The
far-field grid levels still drive bulk expansion") in a FIXED `spaceSize`
where points are seeded across the whole space, so nothing in cosmos ever
needs to expand.

Evidence: probes/clamp-expansion.mjs (exact FA2 with the 7.2 laws, 1,000
nodes, avg degree 10, seed in [-1, 1)): bbox extent without the clamp 2.0 ->
554 by iteration 25 (equilibrium ~575); with the plan's clamp 2.0 -> 2.4 (it
25) -> 4.4 (it 100) -> 9.6 (it 200) -> 20.9 (it 299), exactly 1.0078^t.
repos/cosmos/src/modules/ForceManyBody/force-nearfield.frag lines 145-156
(clamp on the near-field velocity only, with the comment quoted above);
repos/cosmos/src/modules/ForceManyBody/index.ts line 781 (`cellSize =
adjustedSpaceSize / gridSize`, a fixed space);
repos/cosmos/src/modules/Points/update-position.frag lines 56-57 (positions
clamped into `spaceSize`). Plan line 1553 (bbox-derived `cellSize`), 1684
(clamp on the whole `dp`), 1353 (seed box), 1741 (`maxIter` 100), 2749
(gate assertion).

Fix: remove the whole-displacement clamp from the FA2 integrate kernel (the
paper's stability mechanism is the swing-based local speed `speed / (1 +
sqrt(speed * swing_i))`, which already damps a node whose near-field estimate
flips; the exact tier relies on it alone). If a bound on the Horvitz-Thompson
kick is still wanted, apply it where cosmos does -- to the near-field FORCE
term inside G7 before it is added to `force`, e.g. `|f_near| <= c x` the
far-field magnitude for that node or a `nearMax`-scaled bound -- never to
attraction, far field or gravity. Replace the 11.4 assertion "no node moves
more than `2 * cellSize`" with an expansion fixture: "from the LCG seed, the
grid tier's bbox extent after 50 and 200 iterations is within 25% of the
exact tier's on the 20k and 100k fixtures", and add it to the G4 gate. Decide
this together with PERF-1 (a robust extent shrinks `cellSize` further, which
would make the clamp even tighter) and PERF-6 (a larger `gridMax2D` shrinks it
by another 2-4x).

### MISSED-2 -- minor -- 7.3 `partials` / 7.4 K1, K5, lines 1380, 1410, 1414

Claim: the `partials` stride of 64 B is exactly full (A = sum / min / max as
three vec4 = 48 B, B = 8 B, C = 8 B), so any additional per-workgroup
statistic the PERF-1 / PERF-3 fixes need (sum of squared deviations for an
RMS radius, or a high-percentile proxy) either grows the stride to 80 B or
must be packed into the unused `.w` lanes of the three position vec4s
(positions are xyz). The plan should state which, because the K1 fold and
the 7.3 byte model ("padded to 64 B", "groups x 64") depend on it.

Evidence: plan line 1380 (`partials` "groups x 64 ... A = sum p (vec4), min
(vec4), max (vec4) ... B = swing, traction (vec2) ... C = displacement,
freeCount (vec2); padded to 64 B"); PERF-1's fix as accepted above.

Fix: reserve `sum.w` for `sum |p - c|^2` (about the start-of-iteration
centroid, which K5 already reads from `state`) and say so in 7.3 and 7.4 K1
/ K5; the stride stays 64 B.

## Checks the verifier ran that produced no finding

- The reviewer's "checks that passed" list (pyramid arithmetic, far-field
  counts, byte model, no per-frame allocation, K1 / K4 finalizer cost,
  dispatch overhead) was spot-checked and holds.
- 3D grid tier at 1M: 982 far-field evaluations per node is ~1e9 gathers
  from a cached 38 MB pyramid; at the measured 2e11 evaluations/s (sorted
  order) that is ~5 ms, so the 50-160 ms bracket is conservative, not wrong.
- The 1000 x 1000 grid BFS target (T-10, 1.5 s for ~2,000 levels) is loose
  against the measured 2-6 us per dispatch (~80 ms of dispatch overhead plus
  63 round trips at 0.1 ms).
- PageRank T-8 at 1M / 10M (<= 1 s for 100 iterations) is consistent with
  480 MB of traffic per iteration at a few hundred GB/s.
