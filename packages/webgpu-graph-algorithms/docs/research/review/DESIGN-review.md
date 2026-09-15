# DESIGN review -- design confidence and WGSL feasibility

Document under review: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md` (3,328 lines, read in full).
Lens: kernel-level feasibility and correctness of every mechanism in sections 4-8 (binding budgets, workgroup memory, atomics, uniformity, dispatch limits, the on-device controller, the grid pyramid, drag / fixed nodes, device loss, residency), plus the algorithm kernels and every platform fact stated as certain.

Probes run under Dawn-in-Node (`webgpu@0.4.0`) on the RTX 4070 SUPER and on Mesa llvmpipe, from
`/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/probes/`:

- `design-probe.mjs` -> `design-probe-nvidia.log`, `design-probe-llvmpipe.log`
- `uniformity-probe.mjs` -> `uniformity-probe-nvidia.log`

## Platform facts settled by the probes (both adapters agree unless stated)

| Fact | Result | Plan text affected |
| --- | --- | --- |
| Default device limits with no `requiredLimits` (note 05 unverified #2) | spec defaults: maxBufferSize 268,435,456; binding 134,217,728; 8 storage buffers; 16 KiB workgroup memory; 256 invocations; 65,535 groups; **minStorageBufferOffsetAlignment 256 and minUniformBufferOffsetAlignment 256** (the 16 / 64 values in the plan are ADAPTER limits, and the plan's `RaisableLimit` list never requests them) | 2.2 step 4 (line 295), 2.6 rows (433-436), 5.3 (1150), R-22 |
| `override` in `@workgroup_size(WG)`, bool `override` in `select()`, dummy bindings (`colIdx` bound as `array<f32>` in the weights slot, `rowPtr` in the perm slot, the same buffer bound twice read-only) | all work; results correct | 3.5, 5.1, 7.5 -- NOT exercised by `packages/graph-format/test/audit/gpu-upload.test.ts` (it uses literal `@workgroup_size(${WORKGROUP})` and no `override`), so line 239's "verified against the installed webgpu@0.4.0" covered only `globals` |
| Explicit bind-group layout `storage` vs shader `var<storage, read>` (and the reverse) | pipeline creation FAILS both ways ("buffer type in the shader ... is not compatible with the type in the layout") | confirms 5.1's `read_write`-everywhere rule is necessary |
| `subgroup_size` builtin under `enable subgroups;` | available in Dawn 0.4.0; 32 on NVIDIA, 8 on llvmpipe; `subgroupAdd` works | D16 (see DESIGN-6) |
| `timestamp-query` granularity on Dawn-node (note 05 unverified #3, Q-16) | NOT quantised to 100 us: NVIDIA deltas 26,624 / 32,768 / 54,272 ns ... (1,024 ns ticks); llvmpipe 6.7-31 ms unquantised | 2.6 row "unquantised (assumed; unverified)", R-15, R-22 |
| `dispatchWorkgroupsIndirect` with x = 70,000 (note 05 unverified #7) | dispatches nothing, no validation error (x = 65,535 runs 65,535 groups) | 5.4, R-22 -- matches the spec sentence |
| A dispatch that WRITES an `INDIRECT \| STORAGE` buffer followed in the SAME compute pass by `dispatchWorkgroupsIndirect` on it | works, no error (counter = 7) | 5.4 finalize-kernel design is valid |
| Second `requestDevice()` on the same adapter | throws `OperationError: adapter is "consumed": it has already been used to create a device` | 2.2 (`options.adapter`), 5.7 / 11.3 device-loss recovery (see DESIGN-15) |
| `workgroupBarrier()` / `subgroupAdd()` inside `if (valid)` or after an early `return` keyed on `global_invocation_id` | shader-creation ERROR: "'workgroupBarrier' must only be called from uniform control flow" / "'subgroupAdd' must only be called from subgroup uniform control flow"; the same code with the guard on `workgroup_id` or with the reduction outside the branch compiles | 7.6 sketch, 7.10, every kernel with a per-node guard plus a workgroup reduction (see DESIGN-5) |

## Findings

Severity: blocker = the plan as written would fail or mislead implementation; major = a wrong or unsupported decision that must change; minor = imprecision.

### DESIGN-1 (blocker) -- 7.7 / 7.11: the cosmos per-iteration clamp `2 * cellSize` with an adaptive bounding box starves the layout of expansion

Lines 1573-1577 ("The per-iteration near-field displacement is clamped to `2 * state.cellSize` in the integrate kernel when the tier is `grid`") and 1684 (`dp = clamp_length(dp, 2.0 * S.cellSize)`), with 1553 (`cell size = max extent / G` ... "recomputed every iteration"), 1353 (seed in `[-1, 1)`) and 1801-1812 (7.18: layout units, never rescaled).

Claim: because the grid is rebuilt every iteration from the CURRENT bounding box, the clamp bounds every node's move to `2 * extent / G` per iteration, so the bounding box can grow by at most a factor `1 + 4/G` per iteration (1.0078 at G = 512 in 2D, 1.031 at G = 128 in 3D). From the `[-1, 1)` seed, reaching the FA2 equilibrium size of a 100k-node graph (hundreds to thousands of layout units: equilibrium edge length is about `sqrt(scalingRatio * m_i * m_j / w)`, ~15 units for degree-10 nodes) needs ~600 iterations in 2D; the default `maxIter` is 100, which allows a 2.2x expansion. The exact tier has NO clamp: with the speed controller as specified (7.10) iteration 0 at 100k nodes moves nodes by ~100 layout units (speed 1.5, `factor = speed / (1 + sqrt(speed * m|F|))`, `|F|` ~ 1e5), so the two tiers differ by four orders of magnitude on the first iteration and the G4 test "from the same start, 200 iterations exact vs grid agree on the distributional metrics within 15%" (lines 2748-2750) cannot pass. cosmos can clamp because its space is a FIXED `[0, spaceSize]` box (note 03 section 1.1, line 88-90) into which points are seeded already spread out; the plan transplants the clamp without the fixed space.

Evidence: plan lines 1353, 1553, 1573-1577, 1684, 1801-1812; note 03 lines 86-90 and 179-183 (`docs/many-body-force/README.md`: clamp to `2 * cellSize` in `spaceSize` units); Gephi `ForceAtlas2.java` lines 355-366 (no displacement clamp in the reference).

Fix (text): delete the clamp from 7.11 or make it scale-free: "the grid tier clamps `|dp|` to `max(2 * cellSize, 0.25 * radius)` where `radius` is the K1 layout radius", and add to 11.4 an explicit test "grid tier from the `[-1, 1)` seed reaches the same layout radius as the exact tier within 20% after 100 iterations at 32k nodes". State in 7.7 that the clamp is a shimmer control for the Horvitz-Thompson near field only, and that the speed controller, not the clamp, bounds the step.

### DESIGN-2 (major) -- 7.2 / 7.7: `eps = 0.05` softening is cuGraph's constant at cuGraph's scale, 100x too large for the plan's seed scale

Lines 1339 (table row "Distance floor": "softening `d^2 + eps^2`, `eps = 0.05`, in the grid tier's far field only (cuGraph `epssq`)") and 1570 (G6: `F += d * (k m_i M_cell / (|d|^2 + eps^2))`).

Claim: cuGraph's `epssq = 0.0025` (`repos/cugraph/cpp/src/layout/legacy/barnes_hut.cuh` line 60) is used with positions seeded uniformly in `[-100, 100]` (`barnes_hut.cuh` line 133, `exact_fa2.cuh` line 81); the plan seeds in `[-1, 1)` (line 1353, 7.18). In early iterations the finest cell is `2 / 512 = 0.0039` layout units, so the finest far-field ring (distance 2-3 cells, `d^2` ~ 1e-4) is damped 25x, the next level 7x, the next 2.5x: the four nearest of eight levels -- which for a 1/d law carry about half the far-field magnitude -- are suppressed exactly in the expansion phase, while the exact tier uses the plain floor `max(d^2, 1e-4)` (line 1510). The two tiers therefore disagree systematically in the regime the 11.4 exact-vs-grid test measures.

Evidence: plan lines 1339, 1510, 1553, 1570; `barnes_hut.cuh:60,133`; `exact_fa2.cuh:81`.

Fix (text): "the far field uses `d^2 + eps^2` with `eps = 0.25 * cellSize_finest` (recomputed by K1 with the cell size), never an absolute constant; the exact tier's floor `max(d^2, 1e-4)` is applied in the near field so the two tiers share the same pair law". Add the value to `state` next to `cellSize`.

### DESIGN-3 (major) -- 7.4 / 7.10 / 7.11: swing and traction sums include fixed nodes, contrary to the declared Gephi reference, and the grid near-field kernel has no binding left to fix it

Lines 1412 (K3 binds `positions, mass, state, force, oldForce, partials` = 6; no `fixed`), 1571 (G7 binds 8, no `fixed`), 1633-1640 ("The epilogue of the repulsion / near-field kernel computes per node swing_i ... reduces them over the workgroup"), 1694-1697 (7.11 says fixed nodes are excluded from the displacement mean but says nothing about swing / traction).

Claim: Gephi's `ForceAtlas2.java` lines 283-293 sum `totalSwinging` / `totalEffectiveTraction` only `if (!n.isFixed())`; the plan's 7.2 table (line 1348) declares Gephi's controller "line for line" as the reference. A dragged node is pinned by the bridge (7.12) and moved by the pointer every frame, so its `|F(t) - F(t-1)|` is large and, with mass `degree + 1`, a dragged hub dominates the global swing, halving `speedEfficiency` every iteration (`swing / traction > 2` branch) and freezing the rest of the layout during the drag. K3 could bind the mask (7 bindings), but G7 is already at the 8-binding cap the plan enforces with a test (lines 805-807), so the grid tier cannot honour the reference as designed.

Evidence: Gephi `ForceAtlas2.java:283-293`, plan lines 1348, 1412, 1571, 1633-1640, 1694-1697, 1716-1722.

Fix (text): move the partials-B reduction out of the repulsion / near-field epilogue into its own dispatch `fa2-swing-partials` (`force, oldForce, mass, fixed, partials` = 5 bindings, `ceil(n / WG)` groups, runs after K3 / G7 and before K4) OR pack `sortedIdx` and `cellStart` into one buffer bound once with an offset uniform so G7 can bind `fixed`; in 7.11 add "fixed nodes are excluded from the swing / traction sums (Gephi `ForceAtlas2.java` line 285) and from the displacement mean, included in the centroid and bounding box"; update the 7.4 table and dispatch counts.

### DESIGN-4 (major) -- 7.10: the WGSL speed finalizer is not the "line-for-line port" it claims to be

Lines 1653-1656: `if (swing / tr > 2.0) { eff = max(eff * 0.5, 0.05); ... }` and `if (swing > jitter * traction) { eff = max(eff * 0.7, 0.05); }`.

Claim: the CPU `estimateFactor` (`layout/src/layouts/force-directed/forceatlas2.ts` lines 208-212 and 218-222) and Gephi (`ForceAtlas2.java` lines 307-311, 318-322) do `if (eff > 0.05) { eff *= 0.5 }` -- the multiply is SKIPPED at or below the floor, never clamped to it. The two differ whenever `eff` is in `(0.05, 0.1]` (CPU 0.04, WGSL 0.05) or already below 0.05 (CPU unchanged 0.04, WGSL RAISES it to 0.05). The trace-parity test (lines 2726-2730, relative error `<= 1e-4` for the first 10 iterations) fails on the first iteration in which either branch fires.

Evidence: `forceatlas2.ts:208-222`, Gephi `ForceAtlas2.java:305-324`, plan lines 1645-1660.

Fix (text): replace the two lines with `if (swing / tr > 2.0) { if (eff > 0.05) { eff = eff * 0.5; } jitter = max(jitter, P.jitterTolerance); }` and `if (swing > jitter * traction) { if (eff > 0.05) { eff = eff * 0.7; } } else if (S.speed < 1000.0) { eff = eff * 1.3; }`, and strike "line-for-line" unless the WGSL is literally the same branch structure.

### DESIGN-5 (major) -- 7.6 / 7.10 and every guarded kernel: workgroup reductions inside a per-invocation guard are rejected by WGSL's uniformity analysis, and the plan never states the rule

Lines 1518 (`if (valid) { epilogue(i, pi, mi, f); }` where 7.10 line 1637 says the epilogue "reduces them over the workgroup (256 -> 1, subgroup variant when available)"), 1451 (`if (row >= P.tierEnd) { return; }` in a module whose TIER 2 variant is a workgroup-per-row reduce), 1256 (advance `block_mapped`: per-workgroup scan in workgroup memory after loading frontier vertices).

Claim: the probe (`uniformity-probe-nvidia.log` cases A, C, E) shows Dawn/Tint 0.4.0 rejects `workgroupBarrier()` and `subgroupAdd()` whenever control flow depends on `global_invocation_id` / `local_invocation_id` -- a shader-creation error, not a warning. The 7.6 sketch as written cannot be compiled; so would any straightforward transcription of "guard the row, then reduce". Note 05 section 6 (the plan's WGSL constraint list) omits uniformity entirely. This is the single most common WGSL porting error and it touches K3, G7, K5 (the sketch at 1673-1692 is correct only because the reduction is drawn outside the `if`), `segmentedReduce` tiers, `advance`, `histogram`, `radixSort` local ranking and the BC backward pass.

Evidence: probe output; WGSL spec 15.2 uniformity analysis (`workgroupBarrier` and subgroup built-ins require uniform control flow).

Fix (text): add to 3.5 conventions: "Every kernel that reduces across a workgroup or subgroup computes its per-invocation value under the `i < n` guard into a local, then performs the reduction in unconditional code; early `return` before a barrier is allowed only when the condition is a function of `workgroup_id` and uniforms (tier bounds are per workgroup in the workgroup-per-row tier); `test/kernel/wgsl-compile.test.ts` catches violations". Rewrite the 7.6 sketch's last line as `var sw = 0.0; var tr = 0.0; if (valid) { ... f = f + gravity; store_force(i, f); sw = mi * length(f - old); tr = 0.5 * mi * length(f + old); } workgroup_reduce2(sw, tr) -> partials B` (or as DESIGN-3's separate dispatch).

### DESIGN-6 (major) -- D16 / 2.6 / 6: `SUBGROUP_SIZE` as an override from `adapter.info.subgroupMinSize` is wrong on every device where min != max

Lines 190 (D16), 435 (2.6 row "kernels take `SUBGROUP_SIZE` as an `override` from `adapter.info.subgroupMinSize` and are compiled per size"), 1262-1266, 3214 (R-8 mitigation).

Claim: the WGSL spec (fetched 2026-09-14, "subgroup_size built-in value"): "the value for a shader compiled for a specific device will be within the range [subgroupMinSize, subgroupMaxSize] ... The actual size depends on the shader, device properties, and the device compiler ... The device compiler selects a size from the supported sizes using a variety of heuristics. ... Each subgroup may contain fewer invocations than the reported subgroup size". Intel Xe / Arc report 8-32 and AMD 32-64; those are the "integrated GPU" and "other GPUs" the plan explicitly targets (7.8, R-2, 10.3). A subgroup-per-row tier whose lane-to-row mapping assumes the override value miscomputes silently whenever the compiler picks a larger size; the three-adapter CI spread (4 / 8 / 32, all min == max) cannot catch it. The `subgroup_size` builtin is available in Dawn 0.4.0 (probe [5]: 32 / 8), so no override is needed for lane mapping.

Evidence: WGSL spec text quoted above (https://www.w3.org/TR/WGSL/, section on subgroup built-in values); gpuweb `proposals/subgroups.md` ("no shader will be launched where the subgroup_size built-in value is less than subgroupMinSize or greater than subgroupMaxSize" -- a range, not a value); probe [5].

Fix (text): "Subgroup kernels read `@builtin(subgroup_size)` and `@builtin(subgroup_invocation_id)` at runtime for lane mapping; `SUBGROUP_MAX` (= `adapter.info.subgroupMaxSize`) is the only compile-time constant and sizes workgroup-memory scratch; rows-per-workgroup in the subgroup tier is `WG / subgroup_size` computed in the shader; a test forces `subgroupMinSize != subgroupMaxSize` by faking caps and asserts the kernel does not read the override". Keep the `@subgroup_size(N)` entry-point attribute as a later optimisation only where the WebGPU feature exists.

### DESIGN-7 (major) -- 8.2 / 3.5: the PageRank kernel is not at "exactly 8" storage bindings, and no other algorithm kernel has a binding table at all

Lines 2028-2030 ("Bindings fit the default 8 storage buffers exactly (revRowPtr, revColIdx, revWeights, outWeightSum, rankIn, rankOut, personalization-or-dummy, partials ...); a test inspects the layout descriptor"), 805 ("the PageRank kernel of 8.2 is the one algorithm kernel that uses exactly 8"), 2009 and 2181 (`degreeOrder({ of: "reverse" })` is bound), 3166 (G7 gate: "exactly 8 storage bindings in the PageRank kernel").

Claim: the list omits `perm` (the in-degree tier permutation the same paragraph and 8.1 / 8.8 say is bound, with `override USE_PERM` exactly as 7.5's attraction kernel binds it as its 4th graph array) and the "16-byte STORAGE block the next iteration reads" for the dangling mass unless it is a region of `partials`. That is 9-10, so the descriptor test the gate requires fails. Nothing tabulates SSSP near-far (rowPtr, colIdx, weights, dist, nearIn, nearOut, far, counters, pred = 9), the tagged multi-source BC forward pass, or the Louvain move kernel (rowPtr, colIdx, weights, community, clusterWeight, vertexWeight, newCommunity, hash keys, hash values, gain partials = 10), yet 3.5 promises every kernel fits 8 and that over-budget kernels are "SPLIT".

Evidence: plan lines 805, 1435-1445 (perm binding pattern), 2009, 2026, 2028-2030, 2148-2166, 2181.

Fix (text): in 8.2 bind `rank` as ONE `2n`-element `read_write` buffer with `P.inOffset` / `P.outOffset` swapped per iteration (7 with perm and partials, the dangling / delta scalars living in `partials`), and add a per-kernel binding table to section 8 (the 7.4 format) for PageRank, Afforest, BFS expand / contract / fused, SSSP near-far, BC forward / backward (tagged), k-core, triangles, LPA and Louvain move / contraction, applying the same packing (ping-pong pairs in one buffer, counters in a `state` block) wherever a kernel exceeds 8.

### DESIGN-8 (major) -- 8.4: the direction-optimizing switch cannot happen inside a 32-level submit as designed

Lines 2064-2074: "switch to bottom-up ... when the frontier's degree sum exceeds the unvisited degree estimate / alpha and is growing, back when `next * 24 < unvisited` and shrinking ... Host loop: 32 levels per submit with indirect args (5.4), one 4-byte readback of the frontier length every 32 levels".

Claim: WebGPU cannot select a pipeline on the device; a recorded batch runs a fixed sequence of dispatches. With the decision inputs (frontier degree sum, unvisited estimate, growth) visible to the host only every 32 levels and one direction recorded per level, an RMAT / social graph (diameter ~10, line 2603) completes entirely inside the first submit in top-down mode: the switch that gives Beamer's speedup on exactly those graphs never fires, and the gate "`switches > 0` on an RMAT fixture" (line 3167) cannot pass with 32 levels per submit.

Evidence: plan lines 2064-2074, 1160-1173 (5.4), 3167.

Fix (text): "Every level is recorded as BOTH a top-down and a bottom-up dispatch; the level's finalize kernel computes the Beamer test from device-side counters (frontier degree sum accumulated by the previous advance, unvisited count) and writes `(0, 0, 1)` into the indirect slot of the direction not taken, so the choice is made on the device; `switches` is a device counter read with the done flag". Alternatively record 4 levels per submit while the frontier is growing and 32 once it shrinks; state which and cost the extra tiny dispatches.

### DESIGN-9 (major) -- 8.4: "`bc[w] += delta[w]` is a plain add" is false for the tagged multi-source batches the same paragraph prescribes

Lines 2103-2107 (successor-pull backward pass, "`bc[w] += delta[w]` is a plain add") and 2107-2111 (sources batched with `n x k` sigma / depth arrays and two-word frontier entries).

Claim: with k sources processed in one backward-level dispatch, entries `(w, s1)` and `(w, s2)` at the same level are distinct invocations that both execute `bc[w] += ...` -- a float read-modify-write race with no atomic available (7.15). The plain add is race-free only for a single source per dispatch, which is the McLaughlin-Bader single-source formulation note 04 section 6 describes; the batched form needs a separate accumulation.

Evidence: plan lines 2103-2111, 1760-1770 (no float atomics); note 04 lines 392-470.

Fix (text): "`delta` is an `n x k` array written once per `(w, s)`; after the batch's last backward level one gather kernel does `bc[w] += sum over s of delta[s][w]` (k reads per node, no atomics); edge BC likewise accumulates per arc from the `n x k` deltas".

### DESIGN-10 (major) -- 4.1: "destroyed when the last sibling is released" with `refs: Set<serial>` cannot be implemented because siblings share the serial

Lines 862-870: "A `ResidentBuffer` records `{ buffer, byteLength, refs: Set<serial>, kind }`; a core shared by sibling snapshots is destroyed when the last sibling is released", next to "a strong `Map<number, ResidencyRecord>` keyed by `serial` so `withColumns()` siblings share one core; the strong map is cleared by `release`".

Claim: `withColumns()` siblings share the SAME serial (`packages/graph-format/src/snapshot/graph-snapshot.ts` line 314: "Process-unique identity of the CORE; shared by withColumns() snapshots"; design 5.8 line 1243), so `refs: Set<serial>` holds one entry for all siblings and "last sibling" is undefined. Either `release(sA)` destroys the core under a live `sB` (and tombstones `sB`'s serial, so `sB` gets `E_RELEASED` on its next bind), or the record must count distinct snapshot OBJECTS. The paragraph promises both behaviours; an implementer must pick one and the design 14.4 flow (`release(previous)` from `snapshot-replaced`) depends on which.

Evidence: `graph-snapshot.ts:314`; design lines 1238-1250, 4738 (D-SERIAL); plan lines 862-870, 997-1005.

Fix (text): choose the owner semantics and say so: "`release(s)` releases the CORE of `s` and of every `withColumns()` sibling (they share `s.serial`); a sibling still in use receives `E_RELEASED` on its next bind -- releasing is the job of the snapshot-lifecycle owner, who created the siblings"; drop the `refs: Set<serial>` sentence. (If per-sibling counting is wanted instead, key the refs on the snapshot object via the `WeakMap<GraphSnapshot, ...>` and say the core is freed when every sibling object seen has been released.)

### DESIGN-11 (major) -- 7.5 / 6 row 3: with degree tiers active, degree-0 rows are written by no dispatch, but K2 is "the first writer of `force` each iteration"

Lines 1251 (row 3: "thread-per-row for `[midEnd, lowEnd)` (degree < 32) ... degree-0 rows written as the identity element" -- by which dispatch is not said), 1450 and 1472-1475 ("`P.tierStart / tierEnd` from `degreeOrder().segmentOffsets`"), 1466 ("attraction is the first writer of `force` each iteration"), design 10.1 line 2342 (`low = [midEnd, lowEnd)`).

Claim: `segmentOffsets = [0, hiEnd, midEnd, lowEnd, n]` (`packages/graph-format/src/types/snapshot.ts` lines 313-323) puts degree-0 nodes in `[lowEnd, n)`. Three dispatches over `[0, hiEnd)`, `[hiEnd, midEnd)`, `[midEnd, lowEnd)` never touch them, so their `force` keeps the previous iteration's TOTAL (attraction + repulsion + gravity) and K3 adds onto it: isolated nodes integrate an ever-growing force. The P3 slice (single tier, `USE_PERM = false`, range `[0, n)`) is correct; the bug appears at P4 when tiers are switched on.

Evidence: `snapshot.ts:313-323`; plan lines 1251, 1450, 1466, 1472-1475.

Fix (text): "the thread-per-row tier's range is `[midEnd, n)` for K2 and for every segmented reduce that writes an identity element (degree-0 rows cost one store); the P4 gate adds an isolated-node fixture asserting `|F|` on an isolated node equals gravity only".

### DESIGN-12 (minor) -- 6 row 12 / 7.7 G3-G4: `cellStart` from "mark boundaries + scan" has no entry for empty cells; hub-cell dispatch unspecified

Lines 1260 (row 12: "`cellStart[cells + 1]` from the sorted keys (mark boundaries + scan)"), 1567 (G3), 1568 (G4: "+ a workgroup-per-cell dispatch for cells above 1,024 entries"), 1385 (`count(c) = cellStart[c+1] - cellStart[c]`).

Claim: scanning boundary marks over the SORTED positions yields the rank of each occupied cell, not an array indexed by cell id; empty cells (the majority: mean occupancy 0.4 at 100k nodes) get no start, so `count(c)` and the near field's 9- / 27-cell loops read garbage. Also, the list of cells with more than 1,024 entries for the second G4 dispatch needs a device-side compaction plus indirect args that the text does not mention.

Fix (text): "`cellStart` = exclusive scan (the `scan` primitive) over a per-cell `u32` histogram of length `cells + 1` (atomicAdd is order-independent, so the histogram is deterministic even on the `deterministic: false` path); G4's hub tier is a `compact` of cell ids with `count > 1024` into a list dispatched indirectly".

### DESIGN-13 (minor) -- 7.2 / 7.6 / 7.7: the coincident kick is 100x the floor force, not antisymmetric, and the own-cell Horvitz-Thompson weight counts the node itself

Lines 1340 ("deterministic unit kick from `lowbias32(i * 0x9E3779B9u ^ j)` scaled by `k m_i m_j / 1e-4`"), 1509 (`kick(i, jj, mi * o.w)`), 1571 ("scale the cell's sum by `count / nearMax`").

Claim: under the paper law a pair at the distance floor exerts `k m_i m_j / 0.01 = 100 k m_i m_j`; `k m_i m_j / 1e-4` is the PORT law's value at the floor, 100x larger. The hash is not symmetric in `(i, j)`, so `kick(i, j) != -kick(j, i)` and the 11.4 force-sum invariant (`|sum F| <= 1e-4 sum |F|` "on every fixture", lines 2753-2756) fails on the coincident-points fixture. cosmos weights by "others / sampled" (`force-nearfield.frag` line 139, note 03 line 175), i.e. excludes the node itself from its own cell's count.

Fix (text): "kick magnitude `k m_i m_j / 0.01` (the floor); direction from `lowbias32(min(i,j) * 0x9E3779B9u ^ max(i,j))`, negated when `i > j`; the own cell's weight is `(count - 1) / sampledExcludingSelf`".

### DESIGN-14 (minor) -- 8.4 / 6 row 4: `atomicCompareExchangeWeak` can fail spuriously; Davidson's dedupe as written is a data race

Lines 2062 ("`atomicCompareExchangeWeak(&depth[v], INVALID_INDEX, level)` as the visit claim"), 1252 (dedupe: "write my queue index into `owner[v]`, read it back, keep iff equal; no atomics, last-writer-wins is correct").

Claim: WGSL 17.8.5: "The equality comparison may spuriously fail on some implementations" -- if every contender fails spuriously, `v` is not claimed this level and its depth is wrong; no retry loop is specified. The ownership trick uses plain conflicting stores from different workgroups, which WGSL classes as a data race (a dynamic error whose result is unspecified), and reading the value back in the same dispatch has no cross-workgroup visibility guarantee.

Fix (text): "visit claim = `atomicMin(&depth[v], level)`; the invocation that observes `old == INVALID_INDEX` is the winner and writes `parent[v]` (no CAS, no loop)"; "`owner` is `array<atomic<u32>>` written with `atomicStore` in one dispatch and read with `atomicLoad` in the next".

### DESIGN-15 (minor) -- 2.2 / 2.6 / 5.5 / 14: platform facts now verified or corrected by the probes, including one the plan does not know (adapter consumed)

Lines 295-299 (2.2 step 4), 433-436 (2.6 alignment and timestamp rows), 1150 ("64 / 16 in Dawn-node"), 1175-1184 (5.5), 2710 (11.3: "a new context + `load()` works afterwards"), 3228 (R-22), 3249 (Q-16).

Claim: (a) a default Dawn-node device has the spec-default limits INCLUDING `minStorageBufferOffsetAlignment` 256 and `minUniformBufferOffsetAlignment` 256 -- the 16 / 64 figures are adapter values that the plan's `RaisableLimit` list never requests, so the device layer will never see them; (b) Dawn-node timestamps are unquantised (1,024 ns ticks on NVIDIA), closing the Q-16 item; (c) the over-limit indirect dispatch does nothing (spec confirmed); (d) `override` in `@workgroup_size`, bool overrides and dummy bindings work on 0.4.0 -- none of which graph-format's tests exercise, so line 239's "verified" should be re-scoped to `globals`; (e) NEW: a `GPUAdapter` is consumed by its first `requestDevice()` (probe: `OperationError: adapter is "consumed"`), so `GpuContext.create({ adapter })` on an adapter the caller already used, and every device-loss recovery path (5.7, 9.5 `ctx.lost.then`, 11.3), must call `requestAdapter()` again; `probe()` (no device) is unaffected.

Fix (text): update the 2.6 rows (alignment "256 on a default device; 16 / 64 only on the adapter"; timestamp "unquantised [M]"), 5.4 ("verified: does nothing"), R-22 / Q-16 (strike the three settled items), and add to 2.2: "an adapter can create ONE device; `create({ adapter })` throws `E_NO_DEVICE { reason: "consumed" }` on a used adapter, and recovery after `lost` always starts from `gpu.requestAdapter()`".

### DESIGN-16 (minor) -- 11.3 / 11.5: "identical" results across the subgroup twin and across three vendors are impossible for f32 reductions

Lines 2709 ("results identical with the variant on and off"), 2781 ("the whole file produces identical checksums on lavapipe, SwiftShader and NVIDIA" for a file that includes an FA2 iteration and its swing / traction trace).

Claim: the subgroup tree and the workgroup tree sum f32 in different orders; different vendors contract `fma` differently (7.16 itself says "coordinates differ at f32 noise level"). Bitwise identity holds for u32 kernels (`degree`, scans, sorts) only; note 06's "same checksum (999.712)" was a value printed to three decimals.

Fix (text): "u32 kernels: bitwise identical; f32 reductions and the FA2 trace: relative `<= 1e-6` between twins on one device and `<= 1e-5` across adapters; checksums compared after rounding to 1e-4".

### DESIGN-17 (minor) -- 8.7: the APSP size bound uses the buffer limit where the binding limit applies

Line 2172 ("`n <= 8,192` at the 256 MiB default, ~32k at 4 GiB").

Claim: `dist` is ONE storage binding of `4 n^2` bytes, bounded by `maxStorageBufferBindingSize` (128 MiB default) -> `n <= 5,792` at defaults; 8,192 needs a raised binding limit (2 GiB on the 4070 under Dawn, 4 GiB in Chromium); 32k needs a 4 GiB binding, which Dawn-node's 2 GiB - 4 does not give.

Fix (text): "`n <= floor(sqrt(maxStorageBufferBindingSize / 4))`: 5,792 at defaults, 23,170 at 2 GiB (Dawn), 32,768 at 4 GiB (Chromium); windowed rows otherwise `E_TOO_LARGE`".

### DESIGN-18 (minor) -- 7.7 / 11.4: no floor on `cellSize`, and the exact-vs-grid relative-error metric is ill-conditioned

Lines 1553 (`cell size = max extent / G`), 2745-2747 ("Force-field error `|F_grid - F_exact| / |F_exact|` RMS over nodes `<= 5%`").

Claim: an all-coincident or single-node `load()` gives `extent = 0`, `cellSize = 0`, `(p - min) / 0 = NaN` in G1; on a uniform fixture interior nodes have `|F_exact|` near zero by cancellation, so the per-node relative error is dominated by them and the 5% target measures noise, not the pyramid.

Fix (text): "`cellSize = max(extent, 1e-6) / G`"; use the 11.4 force-parity denominator `max(|F_exact(i)|, 1e-3 * max_j |F_exact(j)|)` or normalise by the RMS of `|F_exact|` over nodes.

### DESIGN-19 (minor) -- 6 row 7 / 8.4: the edge-frontier buffer can overflow at the desktop tier

Line 1255 ("an edge-frontier buffer of `min(A, 16M)` entries for the two-phase expansion").

Claim: at 1M / 10M (A = 20M) a single level of an RMAT BFS can expand more than 16M arcs; the count is only known on the device (the finalize kernel), and nothing says what happens above the cap (silent truncation = wrong depths).

Fix (text): "size the edge frontier to `A` when `4A <= maxBufferSize`; otherwise the finalize kernel clamps the expansion to the buffer and records a `resume` offset so the level is expanded in chunks (extra dispatches, same result)".

### DESIGN-20 (minor) -- 6 row 6: radix-sort stability depends on an unstated histogram layout; 7.4 dispatch count undercounts

Lines 1254 ("per-workgroup 256-bin histogram, scan of `groups x 256` (a `scan` call), stable scatter"), 1420-1424 ("~26-31 dispatches per iteration in 2D").

Claim: the scanned histogram yields stable global offsets only when stored digit-major (`hist[digit * groups + group]`); group-major breaks stability, which the grid's determinism (7.16) relies on. K1 + K2 (1-3) + G1 + G2 (12-15) + G3 (1 + 3-dispatch scan) + G4 (1-2) + G5 (7) + G6 + G7 + K4 + K5 is 35-40, not 26-31.

Fix (text): state the digit-major layout in row 6 and correct the count (the fixed-overhead estimate of 0.5 ms / iteration is then ~0.4 ms at 10 us per dispatch).

### DESIGN-21 (minor) -- 7.1 / 7.13 / 7.18: "z untouched in 2D" contradicts "z uploaded as 0" and `toScene`

Lines 1317 ("leaves `z` untouched in 2D"), 1729 ("2D: z is uploaded as 0, never integrated"), 1805-1806 (`toScene` writes `p * scale + center` for all three components, copied into the owner's array).

Claim: if the owner's array carries a non-zero z (a graph previously laid out in 3D and switched to 2D), `load()` uploads 0 and every readback writes `center.z` -- z IS touched; the 11.3 property "dim === 2 leaves z exactly 0" assumes a zero input. The design (14.3, line 3970) says the position column is read AND written in place.

Fix (text): pick one: "2D uploads the inverse-scaled z and never integrates it, and the readback copy skips z (`set` of x / y only per row)" or "2D writes `center.z`; the property test asserts `z === center.z`".

## Confidence

The plan is unusually thorough and most kernel strategies (gather-only attraction, tiled exact repulsion, tree reductions, on-device speed controller, u32-only atomics, indirect-args finalize kernels, explicit layouts with `read_write` shared blocks) are feasible in WGSL and were partly verified here. The layout slice's exact tier (P3) is sound apart from the uniformity sketch and the two reference-semantics slips. The grid tier (P4) is where I would not bet on the text as written: the transplanted clamp and softening constants make the first 100 iterations of the approximate tier behave nothing like the exact oracle, and the binding budget leaves no room to fix the fixed-node semantics. The algorithm section is a plausible roadmap but is under-specified at exactly the points (bindings, host-vs-device decisions, batched accumulation) where WebGPU differs from CUDA; it should not be read as designs.
