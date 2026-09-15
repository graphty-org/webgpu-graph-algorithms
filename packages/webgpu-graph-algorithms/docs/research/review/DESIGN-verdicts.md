# DESIGN verdicts -- skeptical verification of the "design confidence and WGSL feasibility" review

Document: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md` (3,328 lines).
Reviewer output: `tmp/webgpu-plan/review/DESIGN-review.md` (21 findings).

Method: every cited plan line was re-read; every cited source file was opened (Gephi `ForceAtlas2.java`,
`layout/src/layouts/force-directed/forceatlas2.ts`, cuGraph `barnes_hut.cuh` / `exact_fa2.cuh`, cosmos
`force-nearfield.frag`, graph-format `graph-snapshot.ts` / `types/snapshot.ts` / `snapshot/views.ts`,
graph-format-design.md 10.1); the WGSL and WebGPU specs were downloaded and grepped
(`probes/wgsl-spec.html`, `probes/wgsl-tr.html`, `probes/webgpu-spec.html`); the reviewer's probe logs
(`probes/design-probe-nvidia.log`, `design-probe-llvmpipe.log`, `uniformity-probe-nvidia.log`) were read;
one new CPU probe was written and run (`probes/verify-clamp-throttle.mjs` -> `verify-clamp-throttle.log`).

Tally: 17 confirmed, 2 downgraded (DESIGN-10 to minor, DESIGN-15 narrowed), 0 refuted; 4 missed defects added.

## Verdicts

### DESIGN-1 -- CONFIRMED, blocker (for the P4 grid slice)

Plan 7.7 lines 1553 and 1573-1577 and 7.11 line 1684 clamp the TOTAL displacement `dp` to `2 * S.cellSize`
with `cellSize = 1.01 * extent / G` recomputed every iteration; 11.4 line 2749 asserts it as a test.

New evidence beyond the reviewer's: `probes/verify-clamp-throttle.mjs` (CPU model of the 7.2 laws, 2,048
nodes, G = 512, seed [-1, 1)) -- exact tier: extent 2.0 -> 1,055 after 10 iterations -> 4,616 after 100,
iteration-0 mean step 194 units; clamped tier: every node moves exactly `2 * cellSize` (0.0079 units at
iteration 0), extent 2.0 -> 4.33 after 100 iterations, and the on-device speed controller runs away to
`speed = 1.36e5` because the clamp decouples displacement from force. Four orders of magnitude at
iteration 0, as claimed; the G4 gate's "200 iterations exact vs grid within 15%" cannot pass.

The reviewer also understated the misreading: cosmos clamps ONLY the near-field contribution
(`repos/cosmos/src/modules/ForceManyBody/force-nearfield.frag` lines 147-160: "The far-field grid
levels still drive bulk expansion"; the fragment outputs the near-field velocity alone). The plan
transplants a near-field-only guard onto the whole step.

Fix (revised): delete the `dp` clamp from 7.11 / K5 and the "no node moves more than 2 * cellSize"
assertion from 11.4. If a fling guard is kept, apply it inside G7 to the near-field force sum only
(cosmos's placement), in force units, and say so in 7.7. Add the reviewer's expansion-parity test to
11.4 ("grid tier from the [-1, 1) seed reaches the exact tier's layout radius within 20% after 100
iterations at 32k nodes") and correct the 7.7 sentence "cosmos's clamp, applied where the displacement
exists" to "cosmos clamps the near-field contribution only".

### DESIGN-2 -- CONFIRMED, major

`repos/cugraph/cpp/src/layout/legacy/barnes_hut.cuh` line 60 `epssq = 0.0025`, line 133 and
`exact_fa2.cuh` line 81 seed `[-100, 100]`; plan line 1353 seeds `[-1, 1)`, line 1339 keeps `eps = 0.05`
absolute, line 1570 applies it in every far-field level. With the finest cell 2 / 512 = 0.0039 the
nearest far-field ring (1.5-3 cells) has `d^2` ~ 1e-4 against `eps^2 = 2.5e-3`: 26x damping; the next
three levels 7x / 2.5x / 1.3x. The exact tier (line 1510) has no softening. The 11.4 one-iteration
exact-vs-grid force test runs "from the same start" at exactly this scale. Fix as the reviewer wrote:
`eps = 0.25 * cellSize_finest` written by K1 into `state`, never an absolute constant.

### DESIGN-3 -- CONFIRMED, major

`repos/gephi/ForceAtlas2.java` lines 283-293: `if (!n.isFixed())` guards the `totalSwinging` /
`totalEffectiveTraction` sums. Plan line 1346 names Gephi as the swing / traction reference; 7.11
(1694-1697) says only that fixed nodes are excluded from the displacement mean; K3 (1412) and G7 (1571)
bind no `fixed`; G7 is at the 8-binding cap the 3.5 / 11.3 descriptor test enforces. One correction to
the reviewer's evidence: the "line for line" claim at line 1348 is about `estimateFactor` (Gephi
296-328), not the summation loop, so the plan is silent rather than self-contradictory -- but silent on
a point where the named reference is explicit, and the bridge pins the dragged node (1716-1722), so a
dragged hub's `m |F(t) - F(t-1)|` enters the global swing.

Fix (revised, two options, either keeps every kernel <= 8): (a) the reviewer's separate
`fa2-swing-partials` dispatch (force, oldForce, mass, fixed, partials = 5) between K3 / G7 and K4, or
(b) pack `sortedIdx` and `cellStart` into one simulation-owned buffer (cellStart at a 256-aligned offset
passed as a uniform) so G7 binds it once and gains a slot for `fixed` with no extra dispatch. Add to
7.11: "fixed nodes are excluded from the swing / traction sums (Gephi `ForceAtlas2.java` line 285) and
from the displacement mean, included in the centroid and bounding box"; update the 7.4 table.

### DESIGN-4 -- CONFIRMED, major

`layout/src/layouts/force-directed/forceatlas2.ts` lines 206-211 and 218-222 and Gephi lines 307-311 /
318-322: `if (eff > 0.05) eff *= 0.5` (skip at or below the floor). Plan lines 1653 and 1655:
`max(eff * 0.5, 0.05)` / `max(eff * 0.7, 0.05)` -- differs for `eff` in (0.05, 0.1] (0.04375 vs 0.05) and
RAISES a sub-floor value. From `eff = 1` the branch reaches that range after four firings
(1 -> 0.7 -> 0.35 -> 0.175 -> 0.0875), well inside the 50-iteration trace-parity window (2726-2730).
Fix exactly as the reviewer wrote.

### DESIGN-5 -- CONFIRMED, major

`probes/uniformity-probe-nvidia.log` cases A, C, E: shader-creation errors for `workgroupBarrier` /
`subgroupAdd` under a guard on `global_invocation_id`; B, D, F compile. WGSL 15.2 (`probes/wgsl-spec.html`):
"If a uniformity failure is triggered for a synchronization builtin, an error diagnostic is triggered,
which results in a shader-creation error." Plan line 1518 `if (valid) { epilogue(...) }` with 7.10 line
1637 placing the workgroup reduction inside the epilogue; note 05 section 6 (the constraint list the plan
inherits) has no uniformity item. The 7.11 sketch (1673-1692) is correct only because it draws the
reduction outside the `if`. Fix as the reviewer wrote (3.5 convention + rewritten 7.6 tail); the
workgroup-per-row tier's early return must key on `workgroup_id` (probe case D).

### DESIGN-6 -- CONFIRMED, major

`probes/wgsl-tr.html` (W3C TR, subgroups section): "the value for a shader compiled for a specific
device will be within the range [subgroupMinSize, subgroupMaxSize] ... The device compiler selects a
size from the supported sizes using a variety of heuristics. Each subgroup may contain fewer invocations
than the reported subgroup size". Plan D16 (190), 2.6 (435), 1265 and R-8 (3214) take
`adapter.info.subgroupMinSize` as the compile-time truth; on min != max devices the lane-to-row mapping
is wrong and the 4 / 8 / 32 CI spread (all min == max) cannot detect it. `@builtin(subgroup_size)` is
available in Dawn 0.4.0 (`design-probe-*.log` [5]). Fix as the reviewer wrote; see also MISSED-2 for
the subgroup-index gap the fix inherits.

### DESIGN-7 -- CONFIRMED, major

Plan 6 row 9 (SpMV "tiered by IN-degree") and row 3 ("rows visited through `degreeOrder(opts).perm` with
`override USE_PERM`"), 8.1 / 8.8 (`degreeOrder({ of: "reverse" })` bound), and 5.1's "one layout serves
the USE_PERM / HAS_WEIGHTS dummy-binding variants" mean the `perm` slot is in the PageRank layout whether
or not the permutation is the identity (7.5 line 1441 shows the pattern). The 8.2 list (2028-2030) has
eight entries without it: nine slots, ten if the "16-byte STORAGE block" (2026) is its own buffer.
The G7 gate (3166) requires exactly eight. No other algorithm kernel has a binding table; SSSP near-far
and the Louvain move kernel plausibly exceed eight. Fix as the reviewer wrote (2n ping-pong `rank`
buffer with offsets, scalars in `partials`, per-kernel tables in section 8).

### DESIGN-8 -- CONFIRMED, major

Plan 2064-2074 records 32 levels per submit and reads one counter every 32 levels; 5.4 (1160-1173)
describes one indirect `(x, y, 1)` per level; the switch inputs (frontier degree sum, growth) exist only
on the device. RMAT diameter ~10 (2603) finishes inside the first submit; gate 3167 (`switches > 0`)
cannot pass. Fix as the reviewer wrote (both directions recorded per level, finalize kernel zeroes the
unused slot, `switches` a device counter). The same mechanism must cover the fused-vs-two-phase choice
(MISSED-1).

### DESIGN-9 -- CONFIRMED, major

Plan 2103-2107 "`bc[w] += delta[w]` is a plain add" and 2107-2111 batches k sources per dispatch with
`n x k` arrays; 7.15 (1760-1770) has no float atomics. Two invocations `(w, s1)`, `(w, s2)` at the same
backward level race on `bc[w]`. Note 04 lines 440-446 carries the same sentence in the single-source
context. Fix as the reviewer wrote (`n x k` deltas, one gather `bc[w] = sum_s delta[s][w]` per batch).

### DESIGN-10 -- DOWNGRADED to minor

`packages/graph-format/src/snapshot/graph-snapshot.ts` line 314 and 371: the serial is the CORE's
identity, shared by `withColumns()` siblings; design lines 1243-1245 agree. Plan 862-870 says both
"`refs: Set<serial>` ... destroyed when the last sibling is released" and "strong `Map` keyed by serial
... cleared by `release`" -- a real contradiction. Downgraded because the design 14.4 lifecycle the plan
serves never creates `withColumns()` siblings of a live snapshot (results attach through `nodes.set()`,
design 5.8; `undirected(s)` on an undirected snapshot returns `this`), so the choice has no runtime
consequence for the element and an implementer will pick owner semantics by default. Fix: adopt the
reviewer's owner-semantics wording and delete the `refs: Set<serial>` sentence.

### DESIGN-11 -- CONFIRMED, major

`packages/graph-format/src/snapshot/views.ts` lines 568-604 (`DEGREE_TIER_LOW = 1`): degree-0 nodes fall
outside `[0, lowEnd)`; `types/snapshot.ts` 313-323 and design 10.1 line 2342 confirm the three ranges.
Plan 1251 says "degree-0 rows written as the identity element" by no named dispatch; 1450 / 1472-1475
take `tierStart / tierEnd` from `segmentOffsets`; 1466 makes K2 the first writer of `force`. Stale force
on isolated nodes accumulates once tiers are enabled at P4. Fix as the reviewer wrote (`[midEnd, n)` for
the thread-per-row tier of every identity-writing segmented reduce, plus the isolated-node fixture).

### DESIGN-12 -- CONFIRMED, minor

Plan 1260 / 1567 "mark cell boundaries in the sorted keys, scan" with bindings (sortedKeys, marks,
cellStart) is a scan over sorted POSITIONS; 1385 and G7 index `cellStart` by cell id and need an entry
for every empty cell. G4's hub tier (1568) has no stated compaction / indirect args. Fix as the reviewer
wrote (histogram over `cells + 1` then `scan`; `compact` of hub cells dispatched indirectly).

### DESIGN-13 -- CONFIRMED, minor

Line 1340 kick `k m_i m_j / 1e-4` versus the paper law's floor value `k m_i m_j / 0.01`; line 1509 calls
it from the pair loop; `lowbias32(i * 0x9E3779B9u ^ j)` is not symmetric so the 11.4 force-sum invariant
(2753-2756, "every fixture") fails on the coincident fixture; line 1571 weights the own cell by
`count / nearMax` while cosmos (`force-nearfield.frag` lines 120-121, 139) uses `others / sampled` with
the node itself excluded. Additional note: with `d2 = max(d2, 1e-4)` applied as `d * (k / d2)` (1510-1512)
the force ramps linearly from `100 k m m` at |d| = 0.01 down to `k m m` at |d| = 1e-4, so the kick at
`d2 < 1e-8` is a 1e4x discontinuity against the ramp value just above the threshold. Fix as the reviewer
wrote; the kick magnitude `k m_i m_j / 0.01` is the maximum the floored law produces, which keeps the
law monotone.

### DESIGN-14 -- CONFIRMED, minor

`probes/wgsl-spec.html` 17.8.5 note: "The equality comparison may spuriously fail on some
implementations"; 6.5.7: conflicting unsynchronised accesses are "a data race, and hence a dynamic
error". Plan 2062 (CAS claim, no retry) and 1252 (plain-store ownership trick read back in the same
dispatch). In practice Tint lowers the weak CAS to a strong one on Vulkan / D3D12 and the race works on
hardware, which is why this stays minor. Fix as the reviewer wrote (`atomicMin` claim; `atomicStore` /
`atomicLoad` across two dispatches).

### DESIGN-15 -- DOWNGRADED (still minor; fix narrowed)

Confirmed by `design-probe-nvidia.log` / `design-probe-llvmpipe.log`: [1] a default device reports
`minStorageBufferOffsetAlignment 256` and `minUniformBufferOffsetAlignment 256`; [1b] "adapter is
consumed" on a second `requestDevice` (WebGPU spec 3.5.1, `probes/webgpu-spec.html`: "Each adapter
object can only be used to create one device"); [6] timestamps in 1,024 ns ticks, not 100 us; [7] an
over-limit indirect dispatch runs nothing. Two parts of the finding are narrowed: (a) plan 2.2 step 4
(295-299) ALREADY states that a default device has the spec defaults and that 1 TiB / alignment 16 are
adapter values, so only the 2.6 rows (433-436) and 5.3 line 1150 need their wording changed to "adapter
value; a device is 256 unless requested"; (d) is refuted -- line 239's "verified against the installed
webgpu@0.4.0" is scoped to `dawn.globals` exactly as written and claims nothing about overrides or
dummy bindings. Keep (b), (c), (e) and the E_NO_DEVICE `{ reason: "consumed" }` addition to 2.2 / 5.7.

### DESIGN-16 -- CONFIRMED, minor

Plan 2709 "results identical with the variant on and off" and 2781 "identical checksums on lavapipe,
SwiftShader and NVIDIA" for a file that includes an FA2 iteration with f32 swing / traction; 7.16
(1773-1782) itself says cross-GPU f32 differs at noise level; `subgroupAdd` order is implementation
defined. Fix as the reviewer wrote.

### DESIGN-17 -- CONFIRMED, minor

Line 2172 uses `maxBufferSize`; one `dist` binding of `4 n^2` bytes is bounded by
`maxStorageBufferBindingSize` (128 MiB default, note 05 table line 378): `n <= 5,792`. Small correction
to the fix: Chromium's 4 GiB - 4 binding gives `n <= 32,767`, not 32,768; Dawn-node's 2 GiB - 4 gives
23,170. Otherwise as the reviewer wrote.

### DESIGN-18 -- CONFIRMED, minor

Line 1553 has no floor; the 11.4 fixture list (2741-2744) includes "coincident points" at 20k+ nodes,
which gives `extent = 0` and NaN keys in G1 (1566). The per-node relative RMS metric (2745-2747) is
dominated by interior nodes whose exact force cancels. Fix as the reviewer wrote.

### DESIGN-19 -- CONFIRMED, minor

Line 1255 caps the edge frontier at 16M entries while 4.6 (1017-1024) says A = 20M is the normal desktop
tier and the frontier family chunks by 2D dispatch above 16.7M -- the buffer cap contradicts the
dispatch plan and no overflow behaviour is stated. Silent truncation would be a correctness bug, so the
fix must land with P8. Fix as the reviewer wrote.

### DESIGN-20 -- CONFIRMED, minor

Recount of 7.4 line 1420-1424 from the 7.7 kernel table: K1 + K2 (1-3) + G1 + G2 (12-15) + G3 (1 + a
3-7 dispatch scan) + G4 (1-2) + G5 (7) + G6 + G7 + K4 + K5 = 31-41, not 26-31. The digit-major histogram
layout is the standard precondition for a stable LSD scatter and row 6 does not state it. Fix as the
reviewer wrote (the cost difference is ~0.1 ms per iteration, so no performance claim changes).

### DESIGN-21 -- CONFIRMED, minor

Lines 1317, 1729, 1805-1806 and 2708 disagree; note 01 section 4.6 says the element expects `z = 0` in
2D, so "writes `center.z` (0 by default)" is the behaviour to state, with the property test asserting
`z === center.z`. Fix as the reviewer wrote (pick one and state it).

## Missed defects under this lens

### MISSED-1 -- 8.4 / 6 row 8: the fused expand-contract selection has the DESIGN-8 problem (minor)

Lines 2060-2061 ("a fused expand-contract kernel for frontiers below 4,096 entries") and 1256 ("fused
expand-contract variant for tiny frontiers (< 4,096 entries)"). Inside a 32-level submit the host does
not know the frontier size, so the per-level choice between the fused and the two-phase kernels cannot
be made on the host either. Evidence: plan 2064-2074 (32 levels per submit), 1160-1173 (5.4, one
indirect slot per level). Fix: fold into the DESIGN-8 mechanism -- every level records the fused, the
two-phase and (direction-optimizing) the bottom-up dispatches, and the level's finalize kernel writes
`(0, 0, 1)` into every slot but the chosen one, with the size threshold as a uniform; or restrict the
fused kernel to the first levels of a batch and say so.

### MISSED-2 -- D16 / 6 rows 1-3: subgroup variants need a subgroup index that WGSL does not give on Chromium 139 (minor)

Lines 1249-1250 ("`subgroupAdd` then one cross-subgroup pass", "`subgroupExclusiveAdd` + cross-subgroup
fixup"), 1251 and 1471 (subgroup-per-row tier). Indexing workgroup memory or rows by subgroup needs
`@builtin(subgroup_id)`, which requires the `subgroup_id` WGSL language feature: present in Dawn-node
(note 05 line 226) but ABSENT from Chromium 139's four language features (note 05 section 2.4 item 2,
lines 245-248; table line 418), and the WGSL spec says "There is no defined relationship between
subgroup values (i.e. subgroup_invocation_id and subgroup_id) and local_invocation_index"
(`probes/wgsl-tr.html`, subgroups section). DESIGN-6's fix (runtime `subgroup_size`) inherits the gap.
Fix: state the portable technique -- the elected lane of each subgroup allocates its slot / row with a
workgroup `atomicAdd` on a `var<workgroup> atomic<u32>` counter and `subgroupBroadcast`s it -- or gate
the subgroup variants on `wgslLanguageFeatures.has("subgroup_id")` and use the workgroup twin
otherwise; add both to the 11.3 compile matrix on Chromium.

### MISSED-3 -- 7.12 / 7.17: `reheat()` resets the speed controller on every pointer move (minor)

Line 1795-1796: `reheat()` sets `speed = 1`, `speedEfficiency = 1`; line 1714: `setPosition` reheats,
and the bridge calls it on every pointer move during a drag (1720-1722). Gephi never resets `speed`
mid-run (`ForceAtlas2.java` initialises it once). With `iterationsPerStep = 1` per frame the global
speed is reset to 1 every frame of a drag, so it cannot exceed 1.5 while the pointer moves;
`probes/verify-clamp-throttle.log` shows the settling regime at 2,048 nodes running at `speed` 2-13, so
neighbours respond 2-10x slower during a drag than the controller would otherwise allow (the same
reset also fires on `setParams`). Fix: `reheat()` resets `iterationsDone`, `settledCount` and the
settle window only; `speed` / `speedEfficiency` reset only in `load()`; state this in 7.17 and D8.

### MISSED-4 -- 7.11 / 7.17: `meanDisplacement` is undefined when every node is fixed (minor)

Lines 1380 and 1691 (partials C = `sum |dp|` over free rows, free count) and 1787-1789 (K1:
`meanDisplacement` over free nodes, compared with `settleThreshold * radius`). With every node pinned
(`freeCount == 0`, a legal `setFixed` mask) the division yields NaN, the comparison is false every
iteration, `settledCount` never increments and the simulation submits batches until `maxIter` although
nothing can move. Fix: K1 writes `meanDisplacement = select(sum / f32(freeCount), 0.0, freeCount == 0u)`
(an all-fixed layout is settled immediately) and 11.3 adds the all-fixed mask to the property test.
