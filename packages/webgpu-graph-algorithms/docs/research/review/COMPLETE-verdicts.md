# COMPLETE lens -- verifier verdicts

Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md (3328 lines)
Verifier method: every cited plan line re-read; every cited external file opened; the two
cheap probes (arena-bytes.mjs, dawn-facts.mjs) re-run from tmp/webgpu-plan/review/probes/;
gh API queried for the repository; pnpm version and Playwright lifecycle checked.

Summary: 22 findings -> 15 confirmed, 5 downgraded (4, 6, 7, 18 narrowed; 4 also loses one
sub-claim), 0 refuted. Four majors survive (1, 2, 3, 5). Six missed defects added (all minor).

## Verdicts

### COMPLETE-1 -- CONFIRMED (major)
Plan 2028-2031 lists eight PageRank bindings (revRowPtr, revColIdx, revWeights, outWeightSum,
rankIn, rankOut, personalization-or-dummy, partials). Section 6 row 9 (line 1257) makes spmvPull
"tiered by IN-degree (degreeOrder({ of: 'reverse' }))" and row 3 says tiers visit rows "through
degreeOrder(opts).perm with override USE_PERM"; 3.5 (797-801) reserves the group-0 slot "perm or
dummy" and 5.1 says one explicit layout serves the USE_PERM dummy variants, so the slot counts
even when the permutation is identity. 8.8 row 1 and P7 (3166) both require the in-degree tiers.
That is nine storage bindings; the 16-byte dangling/delta STORAGE block that "the next iteration
reads" (8.2) is a tenth unless it is carved out of `partials`, which the plan does not say. The
note the plan cites (04-gpu-algorithms-prior-art.md lines 365-385) has the same omission: it lists
8 bindings and, three bullets earlier, prescribes degree-tiered dispatch through degreeOrder.
Fix (revised): either pack outWeightSum + personalization + the 16-byte block into one auxiliary
buffer (as the reviewer proposes), or pre-scale `rankIn[u] / outWeightSum[u]` into a scaled vector
in the per-node delta kernel (cuGraph's form) so the pull kernel binds neither outWeightSum nor a
separate block; or dispatch each tier over all rows with an inline degree-band early-exit and no
perm. Whichever is chosen, rewrite 8.2, the Summary (82), 3.5 (800) and the G7 / 11.3 descriptor
test text to the real count.

### COMPLETE-2 -- CONFIRMED (major)
`accelerator(): GpuAccelerator // one per context, cached` (608) takes no options;
`GpuAccelerator.forceAtlas2(o?: ForceAtlas2Options)` (699) "accepts the CPU option type"; 9.3's
`ForceAtlas2Options` (2350-2357) carries settleThreshold / settleWindow / iterationsPerStep but
none of `GpuLayoutTuning` (exactMaxNodes, maxInFlight, nearMax, gridMax*, compat, repulsion,
deterministic). 9.4 item 7 (2458-2460) defines `behavior.layout.maxInFlight` as a product knob
with no transport, 9.5 (2491) says the app "may call ctx.calibrate() once to pass exactMaxNodes"
while its own `attachAccelerator` sketch never uses `prefs.exactMaxNodes`, and R-2's mitigation
"the app calls calibrate()" is therefore unreachable on the element path. Only the direct
`createForceAtlas2(ctx, options & GpuLayoutTuning)` API can receive tuning. Fix as proposed:
`ctx.accelerator({ layout?: GpuLayoutTuning })` (defaults merged by GpuAccelerator.forceAtlas2 /
fruchtermanReingold) or an opaque `tuning?: Record<string, unknown>` field on
`LayoutAccelerator.forceAtlas2` that the element forwards from `behavior.layout.gpu`; name the
choice in 3.3, 9.3, 9.4 item 7, 9.5 and P12.

### COMPLETE-3 -- CONFIRMED (major; fix revised)
`accelerator-changed` occurs once in the plan (its definition, 2389); nothing consumes it.
`gpuMinNodes` occurs at 2459 (definition, default 0), 2534 and 3171 (its default is "measured")
and nowhere is it evaluated; 2.4 (358) only says it is "an app / element product setting". After
device loss 5.7 (1204) puts every simulation in `disposed` and 9.5 (2485) says the CPU path is used
"for NEW runs", but a live `SimulationLayoutEngine` is not a new run: 7.19 (1860-1862) says the
bridge's `.catch` "stops the layout" and nothing re-creates the engine on the CPU. The per-frame
`.catch` (2426) attaches a new handler to the SAME coalesced promise every frame (7.19 item 3
returns the oldest pending promise), so one rejection fires `onError` once per frame it was
returned. Severity stays major because 9.4 is titled "Exact changes" and the `gpuMinNodes`
evaluation point is not a one-liner: `_setLayoutInternal` runs before `load()` in the element
(9.4 item 4 `init()` comment), so a threshold must be re-evaluated on `load` / `reload` and the
engine re-created when the node count crosses it. Fix (revised): in 9.4 state (a) the consumer of
`accelerator-changed` -- LayoutManager re-creates the current SimulationLayoutEngine via
`createSimulation(type, opts, newAccelerator)` and `load(snapshot, positions)` from the element's
own array (positions survive); (b) `gpuMinNodes` is evaluated at engine creation AND on
`load` / `reload` against `snapshot.nodeCount`, re-creating the engine when the side changes;
(c) after device loss the stopped engine is re-created on the next `accelerator-changed` (which
the app's `setAccelerator(null)` fires) -- or state explicitly that the user must re-select the
layout; (d) the bridge keeps the last promise and attaches `.catch` once per distinct promise.

### COMPLETE-4 -- DOWNGRADED to minor
Real: 7.8's rule (1608-1610: "largest measured n with <= 4 ms per iteration, rounded down to a
power of two") applied to 7.8's own expected numbers (1604-1605: 32k at 2.8-3.7 ms) yields
32,768, yet the default is 16,384 in six places (181, 495, 1596, 1755, 3212, 3239) and 7.21
(1952) places 32k on the grid tier; `calibrate()` (329-336) probes only 8k and 16k by default so
its `suggestedExactMaxNodes` can never exceed 16k and cannot reproduce the P3 rule (the
`sizes?: number[]` option mitigates but the default ladder disagrees). Not real: "the 4 ms
criterion ignores whether the grid is faster" -- 7.8 line 1611-1612 already says "the P4 gate
re-checks that the grid is not faster below it" and G4 (3163) repeats it. The plan labels 16,384
provisional ("re-fixed at P3 with the measurement cited", 495 and 1596), so this is an
expected-vs-stated default mismatch, not a design contradiction. Fix (narrowed): give
`calibrate()` the same default ladder as T-4 (1k..65k, capped by a time budget); label the 16,384
default "conservative until G3; the 7.8 numbers predict 32,768"; make the 7.21 32k row say
"exact or grid (crossover re-fixed at G3/G4)".

### COMPLETE-5 -- CONFIRMED (major; fix narrowed)
Rule (b) (3148-3149) says a phase adds only the primitives its slice needs. P2 (3160) delivers
windowed upload EXECUTION, grid-stride dispatch, the indirect finalize kernel, packViews, scan,
compact, segmentedReduce "(all three tiers)" and the node-limits project with 100M-item tests.
Section 6's "pulled in by" column says scan -> P4/P8 (1250), compact -> P4/P7/P8 (1252), and 7.5
(1474-1476) says "the first slice (P3) ships the thread-per-row tier ... and adds the tiers in
P4", which P4's deliverables (3163, "attraction tiers over degreeOrder()") repeat -- three
sections, three stories for the tiers, and G2 (3160) gates "subgroup variant on and off" for
segmentedReduce. Indirect dispatch is first needed by the Frontier (row 7, P8); packViews by pull
kernels (P7); grid-stride by per-arc maps at 100M arcs (P7/P8). P3 (exact tier, K1-K5 + reduce)
needs none of these. The 20-27 ed critical path therefore carries roughly 3-4 ed of P7/P8/P4 work
before the owner's first deliverable, in direct tension with the owner's ordering constraint that
13 itself restates. Fix (narrowed): keep in P2 what P3 needs -- CommandBatch, UniformRing, Lease,
device-loss state, Profiler, pipeline warm-up, residentBytes, caps tables, oracle skeleton, the
compile matrix, thread-per-row segmentedReduce; move windowed EXECUTION + node-limits to P4
(scale), scan/compact/histogram to P4, indirect finalize + grid-stride to P8/P7, packViews to P7;
make 7.5, section 6 row 3 and P4 agree that the mid/high tiers land in P4 (and gate them at G4,
not G2); restate the critical path.

### COMPLETE-6 -- DOWNGRADED to minor
Real: 7.20 (1914-1920) says the plan "reserves" `forceLaw` / `velocityVerlet` so a preset "can
mimic ngraph"; Q-9 (3242) says "No in v1 ... the preset is reserved (P5)"; P5 (3164) BUILDS the
preset with a G5 gate that runs ngraph on the CPU as a test devDependency. The three texts do not
agree on whether the preset is built or reserved. Why minor, not major: Q-9's question is about
ROUTING the element's `ngraph` layout to the preset, and its "No" answers that; 7.1 item 3 (the
owner-facing scope) explicitly wants the GPU simulation able to "replace the DEFAULT engine at
large n", and the default engine IS ngraph's spring-electrical model (note 01 section 3.1), so
building the preset is inside the owner's request, not scope creep; P5 is off the critical path.
Fix (narrowed): rewrite 7.20 and Q-9 to "the preset is BUILT in P5; routing `ngraph` to it is not
v1", and size P5 for it (3-4 ed is tight for FR + preset + ngraph parity + benchmarks).

### COMPLETE-7 -- DOWNGRADED to minor
Verified: `git log` -> "your current branch 'master' does not have any commits yet"; `git remote -v`
is empty; `gh api repos/graphty-org/webgpu-graph-algorithms` -> 404 (the org's other repos exist
and are public); root package.json `repository.url` = https://github.com/graphty/webgpu-graph-
algorithms (wrong org; siblings use graphty-org); P0 (3158) never lists creating / pushing the
repository, and G0's "GPU lane registered and green ... recorded in the PR" and 12.4's
`RUNNER_REPO=graphty-org/webgpu-graph-algorithms` presuppose it. Minor because it is an omission
nobody can miss at the first step of P0 and the fix is four lines; it is not a design defect.
Note that "public" is itself an undeclared decision the CI cost model (12.1) depends on. Fix as
proposed, plus one sentence declaring the repository public.

### COMPLETE-8 -- CONFIRMED (minor)
7.3 (1381) gives `state` the usage "storage read_write, COPY_SRC"; `positions` (1373) explicitly
lists COPY_DST for its host writes. 7.4 (1403-1405) has `load()` compute the initial centroid /
bbox / radius on the CPU "so iteration 0 needs no extra pass" -- those values reach `state` only
through `writeBuffer`, which requires COPY_DST; 7.17 (1794-1797) has `reheat()` set speed /
speedEfficiency / settledCount, which K4 (1648) reads from `state`. Fix as proposed.

### COMPLETE-9 -- CONFIRMED (minor)
2.5 item 1 (388-392) asserts the string "webgpu" does not occur in the root and browser bundles;
3.3 (705) puts `readonly kind: "webgpu"` on the public GpuAccelerator, which lives in the root
entry (3.1 accelerator.ts), and the root bundle's own file name / source-map comment contains
"webgpu". 3.1 (487) and 2.5 also call node/index.ts "the only file that mentions 'webgpu'",
contradicted by the same literal and by the error code E_NO_WEBGPU. Fix as proposed (assert on
import / require specifiers); reword the "only file that mentions" claims to "the only file that
imports the module".

### COMPLETE-10 -- CONFIRMED (minor)
4.7 (1040) "~69 B/node"; 7.3 (1389-1390) "+ 4 + 4 + 8 + 4 = ~73"; the 7.3 table sums the grid
additions to cellKey 4 + cellVal 4 + sort scratch 16 (2 x (4 + 4), pool-leased) + sortedIdx 4 =
28 (81 total) or 12 without the leased scratch (65 total); no term is 8. 10.1 (2578) uses 73.
Fix as proposed.

### COMPLETE-11 -- CONFIRMED (minor; one more location)
Re-ran probes/arena-bytes.mjs against graph-format's real `layoutSegments` (used by
src/builder/arena.ts:98): n=1M hot=164,000,256 / full=284,000,256; n=10M hot=1,640,000,256 /
full=2,840,000,256; 100k matches the plan. rowPtr = 4,000,004 B pads to 4,000,256 because
4,000,000 is itself a multiple of 256. The wrong numbers also appear in DEPARTURE-2 (line 205),
which the reviewer did not list. Fix as proposed, adding line 205.

### COMPLETE-12 -- CONFIRMED (minor; narrowed)
10.4 (2622) `benchmarks/results/<host>-node<version>.json` vs 11.1 (2664) / 11.7 (2805-2808)
`benchmarks/results/<runner-class>.json` are two names for the checked-in BASELINE; 12.3's
`bench/results.json` (2999-3004) is plausibly the raw vitest output that bench-compare.mjs reads,
so that one is not necessarily a third baseline path -- say so. Ladders: 7.8 / T-4 (1608, 2634)
1k..65k; 11.1 (2664) 4k/16k/65k/262k/1M; P4 (3163) 32k/65k/100k/262k/1M -- three ladders. Fix:
one baseline name (10.4 = 11.1 = 11.7), state that bench/results.json is the run output, and one
exact ladder + one grid ladder used by 7.8, 11.1, T-4, P3, P4.

### COMPLETE-13 -- CONFIRMED (minor)
12.6 (3124) "opens / refreshes a tracking issue on failure (actions/github-script)" needs
`issues: write`; 12.2 (2876-2877) mandates read-only workflow permissions and no secrets in the
GPU job; 12.3 (2916) sets `permissions: { contents: read }`. Fix as proposed (separate notify job).

### COMPLETE-14 -- CONFIRMED (minor)
12.3's test-gpu steps (2965-3005) contain no `playwright install chromium`; the default lane
(2952-2959) has the cache + install steps; the 12.4 image comment says only `install-deps`.
packages/package.json pins `packageManager: pnpm@10.0.0`, and pnpm 10 does not run dependency
lifecycle scripts unless allow-listed, so `pnpm install` does not download Chromium either. The
runner container persists `~/.cache` across its ephemeral re-registration loop, so only the first
job on a fresh image fails -- but that first job is G0. Fix as proposed.

### COMPLETE-15 -- CONFIRMED (minor)
1.1 goals G1-G9 (109-120); 13 gates G0-G12 (3158-3172); "Gate G3 (section 13)" at 1359 and
"G1 (P1)" at 2631 are ambiguous against goals G3 / G1. Fix as proposed.

### COMPLETE-16 -- CONFIRMED (minor)
`grep -o 'Q-<n>\b' | wc -l`: Q-6, Q-8, Q-11, Q-15, Q-17, Q-18, Q-20, Q-21, Q-22 each occur exactly
once (14.2 only). Lines 185 and 3162 cite "(graft: C Q-10)"; draft-C.md:2115 Q-10 is the Ubuntu
24.04 question while this plan's Q-10 (3243) is the weight-goes-live question. Fix as proposed.

### COMPLETE-17 -- CONFIRMED (minor)
Plan 1.3 row 14.5 (164) lists "injected as runAlgorithm(snapshot, { accelerator: gpu })" among
the inherited decisions honoured "as written" (216); design line 4239-4240 says exactly that;
9.2 defines `accelerated(acc).pageRank(s)` instead and D3 (179) calls the plug-in contract a
decision "the design left open". Plan 1.3 row 16.2 (168) cites design 4545 "1e-5 for f32 GPU
parity"; 9.7 (2510) uses 1e-4 for betweenness (Q-24 acknowledges it). Under the plan's own rule
these are undeclared departures. Fix: the reviewer's lighter option (reword the two 1.3 rows) is
sufficient; a DEPARTURE entry is warranted only for the injection shape, since the design text is
a one-line sketch and 9.2 is the real contract.

### COMPLETE-18 -- DOWNGRADED (minor; one sub-claim refuted, fix narrowed)
Verified: columns.ts:80 `NumericVector = F32 | F64 | U32 | I32` (D13 at 192 says F32 | F64);
Algorithm.ts:217 is `abstract run(g: Graph): Promise<void>` and :283 is the registry (plan cites
:283 at 2237 and 2404); RenderManager.ts:64-65 already constructs `WebGPUEngine` behind
`config.useWebGPU` (plan 131 calls it "a future WebGPUEngine device"); note 05 lines 946-947 say
0.5.0 was not inspected (plan 2921 states "0.5+ needs glibc 2.38"); "~5-10 us GPU-side overhead
per tiny dispatch" (1424) has no basis label or source in any note; dawn-facts.mjs re-run:
`backend=null` returns an adapter under webgpu@0.4.0, default-device limits equal the spec
defaults (268435456 / 134217728 / 8 / 65535 / min*OffsetAlignment 256) while the adapter reports
16 for the alignments -- so 2.2 step 4, 5.1 and R-22's first item can be promoted to [M].
REFUTED sub-claim: "europe.osm ~19,000 levels" (2074) IS sourced -- the sentence ends "(note 04
section 3)" and note 04 line 284 cites Merrill Table 1; it only lacks a [P] tag. Fix (narrowed):
add [P] at 2074 and a source or [X] at 1424; "0.6.1 verified, 0.5.x assumed" at 2921; correct D13;
cite Algorithm.ts:217; write "the element's existing but unused useWebGPU path"; promote the
three verified facts to [M] and record device-level alignment 256.

### COMPLETE-19 -- CONFIRMED (minor)
678 "resolves when positions holds the result of these iterations" vs 7.19 item 3 (1837) which
returns the OLDEST pending promise and queues nothing; 1758 and 2458 "default = stepMultiplier"
(an element concept) in the package option table; E_IN_FLIGHT occurs once (574, the union) and
is never thrown; 550 "the only error type the package throws" vs 942-943 E_GPU_INELIGIBLE (a
GraphFormatError) deliberately propagated; `SimulationType` (2360) includes "spring" with no
CPU class, no accelerator method and no statement that it aliases FR -- graphty-element's
SpringLayoutEngine.ts:73-76 is "Spring layout engine using Fruchterman-Reingold". Fix as proposed.

### COMPLETE-20 -- CONFIRMED (minor)
grep counts reproduce (83 draft references, 49 "judge"); Q-18 (3251) commits the seven notes,
probes and repos but not the drafts the 83 citations point at. Fix as proposed.

### COMPLETE-21 -- CONFIRMED (minor)
3184 "P6 makes it 'detected' in the app"; 9.4 items 2-4 assume `dm.getSnapshot()`, the
`snapshot-replaced` listener list and the element-owned position column, all of which are the
design's E1 port (design 14.6, line 4256: "DataManager owns the builder; adapters and
SimpleLayoutEngine subclasses use indexed.*; position column; EdgeMap removed ..."), and 9.3's
steppable CPU FA2 is part of the design's L1; P6's 8-10 ed covers only this plan's deltas. 9.8 does
state the preconditions per row, so the omission is confined to the critical-path paragraph.
Fix as proposed.

### COMPLETE-22 -- CONFIRMED (minor)
3160 (G2, in the P2 row) "default lane <= 10 min" vs 2642 (T-12) "<= 15 min" vs 3122 "target <= 15". 10.3 rows:
100k per-iter 0.2-1.4 ms x 100 = 20-140 ms, table 20-70 ms; 1M 2-14 ms x 100 = 0.2-1.4 s, table
0.2-0.7 s; 10M 20-140 ms x 100 = 2-14 s, table 2-7 s (the integrated and lavapipe rows DO
multiply correctly, so only the 4070 rows halve the upper bracket); T-8 (2638) ceilings 100 ms
and 1 s sit below 140 ms and 1.4 s. Fix as proposed.

## Missed defects (same lens)

### COMPLETE-M1 (minor) -- 9.7 / G7 vs 8.2: PageRank `iterations` cannot be "within +-1"
9.7 (2505) and G7 (3166) require `iterations` within +-1 of the CPU's and `converged` identical;
8.2 (2027-2028) runs k = 8 iterations per submit and reads convergence "every k", and T-8 / 10.3
repeat "check every 8". The CPU stops at the first iteration whose delta is below tolerance; the
GPU as specified stops at the next multiple of 8, so the reported count differs by up to 7 unless
the device-side delta block also records the first converged iteration index (and then `scores`
come from up to 7 later iterations than the CPU's). Fix: either state that the finalize kernel
records `firstConvergedIteration` in the 16-byte block and `iterations` reports it (scores are
still compared "after equal iterations"), or relax 9.7 / G7 to "within k".

### COMPLETE-M2 (minor) -- T-5 has no measurement vehicle
T-5 (2635) requires Chromium per-frame cost at 10k nodes (G3) and 100k nodes (G4). 11.6 (2797)
caps the browser project at a 500-node FA2 smoke and says "Nothing larger"; 11.1 (2664) makes
`bench` a Node project; 11.7 (2813) says browser numbers "come from the frame-loop test",
which is the 500-node smoke. No listed test or benchmark can produce the 10k / 100k Chromium
numbers G3 and G4 gate on. Fix: add a `bench`-tagged browser test (skipped unless
GRAPHTY_BROWSER_GPU=nvidia) at 10k and 100k that prints `performance.now()` deltas into the
report artifact, or re-scope T-5 to the sizes the browser smoke runs.

### COMPLETE-M3 (minor) -- G1's coverage gate is never evaluated by CI as written
G1 (3159) requires "coverage >= 80/80/75/80 on the node project" green on the lanes (rule (a),
3146). 11.8 (2818-2819) says thresholds apply "when run whole" and are "skipped when a single
--project is selected" (the cited algorithms/vitest.config.ts:63-76 pattern skips on
`--project=`); 12.3 (2950) and the 12.5 shard (3077) always run `vitest run --project=node
--coverage`, and neither the staging workflow nor the monorepo's tools/merge-coverage.sh enforces
thresholds after merging. So the number G1 gates on is checked only by a local whole-package
run. Fix: make the threshold skip conditional on `--project=browser` / `--project=bench` /
`--project=node-limits` only (keep thresholds for `--project=node`), or add an explicit
`vitest run --coverage.thresholds...` check step to the default lane.

### COMPLETE-M4 (minor) -- the no-subgroups twin is exercised for primitives only
D16 (196) and 11.3 (2709) say `GRAPHTY_GPU_NO_SUBGROUPS=1` forces the workgroup-memory twin for
"every subgroup kernel", and the 12.2 table (2894) shows a `0 / 1` matrix; 12.3 (2951, 2998)
runs the NO_SUBGROUPS pass on `test/primitives` only. The FA2 repulsion / near-field epilogue
reduce (7.10, 1638: "subgroup variant when available") lives under `layouts/`, and the layout
parity tests of 11.4 therefore never run on the twin in CI. Fix: run the NO_SUBGROUPS pass over
`test/primitives test/layouts` (and later `test/algorithms`), or state that only primitives carry
subgroup variants and move the epilogue reduce onto the `reduce` primitive.

### COMPLETE-M5 (minor) -- small arithmetic / number drift not in the reviewer's list
(a) 10.3 (2604) FA2 exact at 100k = "30-38 ms" while 7.21 (1958) says "28-36 ms" for the same
[X]; 10.3's own 1M row "2.8-3.6 s" is 100 x 28-36, not 100 x its 30-38. (b) 13 (3187) "Rough
total: 83-112 ed" -- the fourteen phase sizes sum to 82-112. (c) DEPARTURE-2 (205) carries the
128-vs-256 arena figures (add to COMPLETE-11). Fix: one exact-tier figure in 7.21 and 10.3;
"82-112"; the four arena numbers at line 205.

### COMPLETE-M6 (minor) -- the per-iteration trace slot is under-specified
7.3 (1381) sizes the `state` trace region at "k x 32" bytes "appended by the speed finalizer";
the K4 snippet (1659) writes a 16-byte `vec4f(swing, traction, speed, eff)`; `LayoutStats.trace`
(674) carries five fields including `meanDisplacement`, which only K1 computes (into
`state.meanDisplacement`, K1 row at 1410) and no kernel is stated to write into the trace slot. Fix:
say K1 writes `meanDisplacement` (and `settledCount`) into slot `iterationIndex` of the 32-byte
trace record, or drop `meanDisplacement` from the trace type.
