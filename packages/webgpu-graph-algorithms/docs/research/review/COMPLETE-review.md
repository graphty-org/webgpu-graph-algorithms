# Adversarial review -- lens: Completeness and internal consistency

Document reviewed: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md`
(3,328 lines, read in full). Reviewer: completeness / consistency lens. Date: 2026-09-14.

Probes written and run for this review (under `tmp/webgpu-plan/review/probes/`):

- `arena-bytes.mjs` -- recomputes the arena hot prefix / byteLength at the plan's three tiers
  with graph-format's own `layoutSegments` (dist build). Output:
  `n=100000: hot=16,400,128 total=28,400,128`; `n=1000000: hot=164,000,256 total=284,000,256`;
  `n=10000000: hot=1,640,000,256 total=2,840,000,256`.
- `dawn-facts.mjs` -- against the installed `webgpu@0.4.0` on the RTX 4070 SUPER:
  `backend=null` IS honoured (returns an adapter with empty vendor/architecture); a device
  requested with NO `requiredLimits` reports the spec defaults (maxBufferSize 268,435,456,
  binding 134,217,728, 8 storage buffers/stage, 16 KiB workgroup storage, 256 invocations,
  minStorageBufferOffsetAlignment 256 on the DEVICE although the adapter says 16);
  `timestamp-query` deltas of 26624 / 24576 / 24576 / 24576 / 25600 ns for a ~25 us pass,
  i.e. NOT quantised to 100 us in Dawn-node; a Dawn-node adapter is "consumed" after one
  `requestDevice()` (a second call throws `OperationError`).

## 0. Mechanical checks

- ASCII: `grep -nP '[^\x00-\x7F]'` -> 0 hits. Clean.
- URLs: every URL cited inline in sections 1-14 (5 of them) is in section 15; the only
  "missing" one is the shell variable `https://github.com/$RUNNER_REPO`. Section 15's extra
  URLs are reachable through "note NN" citations as the preamble promises. Clean.
- Q-numbers: all 25 exist in 14.2 and each has a default. NINE of them are referenced ONLY
  from the 14.2 table and never from the section that depends on them: Q-6, Q-8, Q-11, Q-15,
  Q-17, Q-18, Q-20, Q-21, Q-22 (finding COMPLETE-16).
- T-numbers T-1..T-15 all defined in 10.4 and each is bound to a gate; R-1..R-22 all defined.
- Gates G0-G12 and G-ENV and phases P0-P12 / P-ENV all exist in section 13; every phase gate
  references sections that exist. BUT goal ids G1-G9 (section 1.1) collide with gate ids
  (finding COMPLETE-15).
- Error codes: the 3.3 list is used consistently except `E_IN_FLIGHT` (declared, never used)
  and `E_GPU_INELIGIBLE` (a format error the package lets through despite D12).

## 1. Owner request coverage

| Owner constraint | Covered? | Where / gap |
| --- | --- | --- |
| Force-directed layout first, after a walking skeleton | yes | P1 -> P3; but P2 inflates the pre-layout path (COMPLETE-5) and P5 adds an un-requested preset before P6 (COMPLETE-6) |
| Uses graph-format | yes | 1.3, 4, G3 |
| High performance at 100k-1M+ | yes | 7.7, 7.21, 10 |
| Optional / DETECTED acceleration for the existing packages, end to end | partly | probe -> import -> create -> inject is specified (2.3, 9.5); GPU package absent = no injection; device missing = probe false. NOT specified: what the element does with a live LAYOUT engine after device loss / a kernel throw, who consumes `accelerator-changed`, where `gpuMinNodes` is evaluated (COMPLETE-3); how app-side tuning (`calibrate()` -> `exactMaxNodes`, `maxInFlight`) reaches the simulation the element creates (COMPLETE-2) |
| NVIDIA input (Merrill, McLaughlin-Bader, cuGraph, Buffalo, cluster page) | yes | 8.4, 8.9: each used or set aside with a reason; note 04 did extract the Buffalo thesis text (`tmp/webgpu-plan/papers/buffalo-2023-06.txt`), so 1.2's "negative result" claim is sourced |
| Node AND browser | yes | 2 |
| Node-primary tests, light browser tests | yes | 11 (DEPARTURE-1 declared) |
| CI: default runner + GPU runner, monorepo | yes | 12; but the staging repository does not exist on GitHub yet and P0 never creates it (COMPLETE-7); the GPU job lacks a browser-install step (COMPLETE-14); nightly issue creation contradicts the permission rules (COMPLETE-13) |
| GPU package never falls back | yes | 2.4; the CPU short-circuits (n = 0, LCG seeding, renumbering, segmentOffsets) are correctly not fallbacks |

## 2. Design contradictions not declared as DEPARTURE

- 1.3 claims to honour design 14.5's injection form `runAlgorithm(snapshot, { accelerator: gpu })`
  (plan line 164) but 9.2 defines `accelerated(acc).pageRank(s)`; 1.3 claims 16.2's blanket
  `1e-5` f32 parity (line 168) but 9.7 uses `1e-4` for betweenness. Both are defensible; both
  are undeclared (COMPLETE-17).

## 3. Findings

### COMPLETE-1 (major) -- The PageRank kernel needs 9 storage bindings, not "exactly 8"
- Lines: 2028-2031 (8.2), 1257 (section 6 row 9), 797-801 (3.5), 82 (Summary), 2714 (11.3), 3166 (G7).
- Claim: 8.2 lists `revRowPtr, revColIdx, revWeights, outWeightSum, rankIn, rankOut,
  personalization-or-dummy, partials` = 8, but section 6 row 9 and 8.1 / 8.8 / P7 tier the pull by
  in-degree through `degreeOrder({ of: "reverse" }).perm` with `override USE_PERM`, and 3.5 reserves a
  `perm | dummy` slot in group 0. That is a 9th storage binding; the G7 descriptor test ("exactly 8")
  and the Summary's "every kernel fits the core default of 8" fail as written.
- Fix: pack `outWeightSum`, `personalization` and the 16-byte dangling / delta block into ONE
  auxiliary buffer indexed at `[0, n)`, `[n, 2n)`, `[2n, 2n + 4)` (the packing 3.5 already recommends),
  or state that the pull kernel physically permutes rows (no `perm` binding). Update 8.2, the
  Summary and G7 accordingly.

### COMPLETE-2 (major) -- GPU tuning has no path from the app / element to the simulation
- Lines: 608 (`accelerator(): GpuAccelerator // one per context, cached`), 695-703
  (`GpuAccelerator.forceAtlas2(o?: ForceAtlas2Options)`), 2323 (9.2 "GPU-only tuning goes through
  the factory options, never through the dispatcher"), 2358-2363 (9.3 `LayoutAccelerator`),
  2458-2460 (9.4 item 7 `behavior.layout.maxInFlight`), 2489-2491 (9.5 "may call `ctx.calibrate()`
  once to pass `exactMaxNodes`"), 1612-1614 (7.8), 3171 (P12 "`calibrate()` wiring").
- Claim: the element creates simulations with `createSimulation(type, opts, graph.accelerator)`
  -> `accelerator.forceAtlas2(opts)` where `opts` is the CPU `ForceAtlas2Options` (9.3), which has
  no `maxInFlight`, `exactMaxNodes`, `nearMax`, `deterministic`, `gridMax2D/3D` or `compat`, and
  `ctx.accelerator()` takes no arguments. So `calibrate()` -> `exactMaxNodes`, the
  `behavior.layout.maxInFlight` knob and Q-21's `gridMax3D` cannot reach the simulation the
  element steps; "calibrate() wiring" in P12 has no API to wire into.
- Fix: give the accelerator factory defaults -- `ctx.accelerator({ layout?: GpuLayoutTuning })` --
  and let `GpuAccelerator.forceAtlas2` merge them; or add an opaque `tuning?: Record<string,
  unknown>` field to `LayoutAccelerator.forceAtlas2` in 9.3 (owned by `@graphty/layout`) that the
  element forwards from `behavior.layout.gpu`. State which and update 3.3, 9.3, 9.4 item 7, 9.5, P12.

### COMPLETE-3 (major) -- Detection is specified up to injection; the failure half is not
- Lines: 2389-2399 (9.4 item 1: `accelerator-changed` event; "the element never does it by
  itself"), 2429-2431 (bridge `.catch` -> `onError`), 2458-2460 (`gpuMinNodes`), 2485 (9.5
  `ctx.lost.then(... setAccelerator(null) ...)`, "CPU path for NEW runs"), 1204 (5.7: on device loss
  "every simulation enters disposed; the CALLER decides"), 1860-1862 (7.19: the bridge "stops the
  layout"), 361-363 (2.4), 3171 (P12 "device-loss UX").
- Claim: (a) `accelerator-changed` has no consumer anywhere in the plan; after device loss the
  `SimulationLayoutEngine` holds a disposed GPU simulation, the app has set the accelerator to
  `null`, and nothing says whether `LayoutManager` rebuilds the engine with the CPU simulation,
  keeps the frozen positions, or requires the user to re-select a layout. "CPU path for NEW runs"
  covers algorithms, not a persistent layout engine. (b) `gpuMinNodes` (9.4 item 7) is a knob with
  no evaluation point: `createSimulation(type, opts, accelerator)` takes no node count and is called
  before `load(snapshot)`; the algorithm side has no threshold at all, so a 34-node karate PageRank
  goes to the GPU whenever an accelerator is injected, while 8.6 says "the caller chooses the CPU
  package for small graphs". (c) 7.19 item 3 returns the SAME pending promise to every coalesced
  per-frame call, so one rejection triggers `onError` once per frame that coalesced.
- Fix: specify in 9.4: on `accelerator-changed` (including -> `null`) `LayoutManager` re-creates the
  current `SimulationLayoutEngine` via `createSimulation(type, opts, newAccelerator)` and
  `load(snapshot, positions)` (positions survive because the array is element-owned); specify that
  `gpuMinNodes` is applied by the element in `_setLayoutInternal` (`nodeCount >= gpuMinNodes ?
  graph.accelerator : null`) and, if wanted, by the adapters as `accelerated(n >= gpuMinNodes ? acc :
  null)`; specify that the bridge attaches `.catch` once per distinct promise.

### COMPLETE-4 (major) -- `exactMaxNodes` default 16,384 contradicts the plan's own crossover rule
- Lines: 1594-1614 (7.8), 329-336 (2.2 `calibrate`), 1952 (7.21 row "32k ... grid 2D (exact would be
  2.8-3.7 ms)"), 181 (D7), 495 (`constants.ts`), 1755 (7.14), 3212 (R-2), 3239 (Q-6).
- Claim: the mechanical rule is "the largest measured n with <= 4 ms per iteration, rounded down to a
  power of two". By 7.8's own extrapolation 32k costs 2.8-3.7 ms (< 4 ms), so the rule yields
  32,768, not the 16,384 written in D7, constants.ts, 7.14, R-2 and Q-6, and 7.21 places 32k on the
  grid tier in contradiction with the rule. `calibrate()` probes only 8k and 16k, so it can never
  suggest more than 16,384 while the P3 rule measures up to 65k -- the two mechanisms disagree by
  construction. The 4 ms criterion also ignores whether the grid is faster (the criterion the P4
  gate actually uses).
- Fix: make the rule "the largest n at which exact <= grid, measured on the same fixture" (with
  4 ms as a ceiling), apply it at G4 (when the grid exists) rather than G3, make `calibrate()` probe
  the same size ladder (1k..65k) as the rule, and either change the documented default to what the
  rule predicts or label 16,384 as "conservative until G4".

### COMPLETE-5 (major) -- P2 bundles infrastructure the first deliverable does not need
- Lines: 3148-3149 (rule (b): "a phase adds only the primitives its slice needs"), 3160 (P2),
  1250-1252 (section 6 rows 2-4 "pulled in by": scan P4/P8, compact P4/P7/P8, segmentedReduce
  P3/P7), 1471-1476 (7.5: "The first slice (P3) ships the thread-per-row tier ... adds the tiers in
  P4"), 3180-3186 (critical path).
- Claim: P2 (6-8 ed, on the critical path to the layout) delivers windowed upload EXECUTION,
  grid-stride dispatch, the indirect-dispatch finalize kernel, `Profiler`, `packViews`, `scan`,
  `compact`, `segmentedReduce` with ALL THREE tiers, and the `node-limits` project with a 100M-item
  dispatch -- none of which P3 (exact-tier FA2, thread-per-row attraction, wall-clock gates) uses.
  This contradicts rule (b), contradicts section 6's "pulled in by" column (scan / compact first
  needed at P4), and contradicts 7.5 (attraction tiers deferred to P4 although the identical tiered
  primitive is required at G2).
- Fix: move indirect finalize, `packViews`, `compact`, `scan`, grid-stride, windowed EXECUTION and
  the 100M-item `node-limits` tests to the phase that first needs them (P4 / P7 / P8 per section 6);
  keep in P2 only `Lease`, `CommandBatch`, `UniformRing`, device-loss state, `reduce`, thread-per-row
  `segmentedReduce`, caps tables and the compile matrix; either ship the three attraction tiers in P3
  or drop them from G2. Re-state the critical path (should fall to ~16-22 ed).

### COMPLETE-6 (major) -- P5 builds the spring-electrical preset that 7.20 / Q-9 say is reserved
- Lines: 3164 (P5 deliverables and G5: "the `spring-electrical` preset with the velocity integrator
  and ngraph's settle rule ... settles within 1,000 steps on the 150-node story graph ... ngraph run on
  the CPU in the test, devDependency"), 1911-1921 (7.20: "The plan reserves `forceLaw` ... a
  `"spring-electrical"` preset can mimic ngraph's feel ... a product decision (Q-9)"), 3242 (Q-9
  default: "No in v1 ... the preset is reserved (P5)").
- Claim: 7.20 and Q-9 present the preset as a reserved design hook and a non-v1 decision; P5 builds
  it, adds a 12n velocity buffer, a new integrator, ngraph as a test devDependency and a gate metric.
  The owner asked for FA2 (and the plan's own G1 names FR); this is un-requested scope inside the
  layout slice, and the two sections disagree on whether it exists.
- Fix: either remove the preset from P5 / G5 (keep the `forceLaw` override reserved as 7.20 says) or
  change Q-9 and 7.20 to say the preset IS built in P5 and only its ROUTING is deferred. Size P5
  accordingly.

### COMPLETE-7 (major) -- The CI plan presupposes a public GitHub repository that does not exist
- Lines: 2842-2846 (12.1 "free for public repos"), 2864-2879 (12.2 "public repositories"
  mitigations, repository settings), 3025-3031 (12.4 `RUNNER_REPO=graphty-org/webgpu-graph-algorithms`,
  registration token), 3158 (P0 / G0 "GPU lane registered and green", "the deliberate red run
  recorded in the PR").
- Evidence: `git log` -> "your current branch 'master' does not have any commits yet"; `git remote
  -v` -> none; `package.json` `repository.url` = `https://github.com/graphty/webgpu-graph-algorithms`
  (org `graphty`, not `graphty-org` used by every sibling package and by 12.4).
- Claim: G0 cannot be reached as written: there is no repository to run `ubuntu-latest` on, no PR to
  record the red run in, no repo to register the ephemeral runner against, and the "free for public
  repos" cost model requires the repo to be public. P0 lists file deletions and the runner recipe
  but not "create `graphty-org/webgpu-graph-algorithms` (public), push the initial commit, fix
  `repository.url`, protect `master`".
- Fix: add to P0's deliverables: create the public repository under `graphty-org`, push the reset
  scaffold as the first commit, set `repository.url` / `directory`, create the `gpu` label, flip the
  two settings of 12.4, and make G0's first line "the default lane ran on GitHub".

### COMPLETE-8 (minor) -- The `state` buffer is written by the host but has no `COPY_DST`
- Lines: 1381 (7.3 `state` usage "storage read_write, COPY_SRC"), 1794-1797 (7.17 `reheat()` sets
  `speed = 1`, `speedEfficiency = 1`, `settledCount = 0` -- fields that live in `state` per 7.3 and are
  read by K4 as `S.speed`), 1403-1405 (7.4 `load()` seeds the initial centroid / bbox / radius into
  `state`).
- Claim: `queue.writeBuffer` to a buffer without `COPY_DST` is a validation error; `reheat()` (on
  every `setPosition` during a drag) and `load()` both need it.
- Fix: change the usage column to `storage read_write, COPY_SRC, COPY_DST` and say `reheat()` is a
  16-byte `writeBuffer` into the speed block (or a flag in the next batch's first uniform slot that K1
  honours).

### COMPLETE-9 (minor) -- The build-output test is unsatisfiable against the plan's own API
- Lines: 388-392 (2.5 item 1: "asserts the string `"webgpu"` (the module specifier) does not occur"
  in the root and browser bundles), 705 (3.3 `readonly kind: "webgpu"`), 2713 (11.3).
- Claim: the root bundle necessarily contains the string literal `"webgpu"` (`GpuAccelerator.kind`),
  so the test as specified fails on a correct build.
- Fix: assert the absence of an import/require of the specifier (`from "webgpu"`, `import("webgpu")`,
  `require("webgpu")`), e.g. by parsing the bundle's import statements, not a substring match.

### COMPLETE-10 (minor) -- Bytes per node for the grid tier: 69, 73 and neither matches the table
- Lines: 1040 (4.7 "FA2 grid ~69 B/node"), 1389-1391 (7.3 "grid tier + `4 + 4 + 8 + 4` = ~73"),
  2578 (10.1: 73 MB at 1M), 1383-1387 (7.3 table: `cellKey` 4n, `cellVal` 4n, sort scratch
  `2 x (4n + 4n)` = 16n, `sortedIdx` 4n).
- Claim: the 7.3 table adds 28 B/node (81 total) if the leased sort scratch is counted, or 12 B/node
  (65 total) if it is not; the text says 73 and 4.7 says 69. Three numbers, none derivable.
- Fix: pick one accounting (state whether pool-leased scratch counts), recompute, and use the same
  number in 4.7, 7.3 and 10.1.

### COMPLETE-11 (minor) -- Arena byte figures at the 1M and 10M tiers are off by 128 B
- Lines: 908-909 (4.2 table), 1052-1054 (4 review notes), 2582 (10.1).
- Evidence: probe `arena-bytes.mjs` with graph-format's `layoutSegments`: 1M / 10M hot prefix
  164,000,256 B, full 284,000,256 B; 10M / 100M 1,640,000,256 / 2,840,000,256 B (`rowPtr` of
  4,000,004 B pads to 4,000,256, not 4,000,128; 4,000,000 is itself a multiple of 256). The 100k
  figures are right.
- Fix: replace the four numbers; the conclusions (fits / exceeds) are unchanged. A test that pins
  these constants would otherwise be written wrong.

### COMPLETE-12 (minor) -- Benchmark results paths and size ladders disagree
- Lines: 2622 (10.4 `benchmarks/results/<host>-node<version>.json`), 2664 (11.1 bench sizes
  4k / 16k / 65k / 262k / 1M), 2805-2808 (11.7 `benchmarks/results/<runner-class>.json`),
  2999-3004 (12.3 `--outputJson bench/results.json`, artifact path `bench/results.json`), 1604-1610
  (7.8 rule needs 1k / 4k / 8k / 16k / 32k / 65k), 3168 (P4 32k / 65k / 100k / 262k / 1M).
- Claim: three different results-file conventions; the `bench` project's exact-tier ladder omits the
  32k point the 7.8 rule and T-4 require.
- Fix: one path (`benchmarks/results/<runner-class>.json`, with host and node version inside the
  file) used in 10.4, 11.7, 12.3 and `bench-compare.mjs`; one exact ladder (1k .. 65k) in 11.1, 7.8,
  T-4 and P3.

### COMPLETE-13 (minor) -- The nightly tracking issue contradicts the permission and secret rules
- Lines: 3124 (12.6 nightly "opens / refreshes a tracking issue on failure (`actions/github-script`)"),
  2876-2877 (12.2 "workflow permissions read-only; NO secrets in the GPU job"), 2916 (12.3
  `permissions: { contents: read }`).
- Claim: creating an issue needs `issues: write` on the job's token; the plan grants nothing above
  `contents: read` and forbids secrets in the GPU job.
- Fix: add a separate `notify` job on `ubuntu-latest` with `needs: test-gpu`, `if: failure() &&
  github.event_name == 'schedule'` and `permissions: { issues: write }`; keep the GPU job read-only.

### COMPLETE-14 (minor) -- The GPU job runs the browser project without installing Chromium
- Lines: 2965-3005 (12.3 `test-gpu` steps: no Playwright cache / install step), 2952-2959 (the
  default lane's install steps), 3020 (12.4 Dockerfile comment installs only `install-deps`, not the
  browser).
- Claim: `timeout 900 pnpm exec vitest run --project=browser` on a fresh ephemeral runner has no
  Chromium build 1181; the first GPU-lane run fails at P0.
- Fix: add the cache + `playwright install chromium` steps to the GPU job (or bake the browser into
  the runner image and set `PLAYWRIGHT_BROWSERS_PATH`).

### COMPLETE-15 (minor) -- Goal ids collide with gate ids
- Lines: 109-120 (goals G1-G9), 1359 ("Gate G3 (section 13)"), 2631-2645 (T-table gates "G1
  (P1)"), 3154-3172 (gates G0-G12), plus "graft: C G4 / G8" (draft C's gates).
- Claim: "G3" means "consume graph-format" in 1.1 and "FA2 exact gate" in 7.2 / 10.4 / 13; a reader
  of 1.3 or 7.2 cannot tell which.
- Fix: rename goals to `GOAL-1..9` (or gates to `GATE-0..12`).

### COMPLETE-16 (minor) -- Nine open questions are orphaned from the sections that depend on them
- Lines: 3239 (Q-6, should be cited from 7.8), 3241 (Q-8 <- 7.17), 3244 (Q-11 <- 7.1 / 5.7),
  3248 (Q-15 <- 5.7), 3250 (Q-17 <- 7.21 / 10), 3251 (Q-18 <- 3.1 / P0), 3253 (Q-20 <- 12.3 / P0),
  3254 (Q-21 <- 7.7), 3255 (Q-22 <- 11.6). Also 185 and 3162 cite "(graft: C Q-10)" -- draft C's
  question about the Ubuntu move -- while THIS plan's Q-10 is the `weight`-goes-live question.
- Fix: add the Q-id at each decision point; write "(draft C question Q-10, here Q-4)" at 185 / 3162.

### COMPLETE-17 (minor) -- Two undeclared departures in the "inherited decisions" table
- Lines: 164 (1.3: honours 14.5 "injected as `runAlgorithm(snapshot, { accelerator: gpu })`") vs
  2259-2320 (9.2 defines `accelerated(acc).pageRank(s)`; D3 line 177 calls this a decision the design
  "left open"); 168 (1.3: honours 16.2 "`1e-5` relative for f32 GPU parity") vs 2510 (9.7 BC
  `1e-4`) and 3257 (Q-24).
- Claim: the plan's own rule is that any other contradiction is a defect; both are reasonable but
  must be declared (DEPARTURE-5 / -6) or the 1.3 rows reworded ("14.5's sketch; concrete form in 9.2";
  "1e-5 except BC 1e-4, Q-24").

### COMPLETE-18 (minor) -- Unverified assumptions stated as facts, and three now verifiable
- Line 1424: "~5-10 us of GPU-side overhead per tiny dispatch" -- no basis label.
- Line 2921: workflow note "0.5+ needs glibc 2.38" -- note 05 section 12 item 1 says 0.5.0 was NOT
  inspected (assumed); write "0.6.1 verified; 0.5.x assumed".
- Line 2074: "europe.osm ~19,000 levels" -- no [P] label / source.
- Line 187 (D13): `NumericVector` is described as `F32 | F64`; graph-format
  `src/types/columns.ts:80` defines `F32 | F64 | U32 | I32`, so `ScoresResultLike.scores` is looser
  than the plan believes.
- Lines 2237, 2404: `Algorithm.ts:283` is cited as the `async run()` adapter; the file has
  `abstract run(g: Graph): Promise<void>` at line 217 and the registry at 283 (the note carried the
  same error).
- Line 131: "a future `WebGPUEngine` device" -- `RenderManager.ts:64` already builds a
  `WebGPUEngine` when `config.useWebGPU` is set (unused today); say "the element's existing but
  unused `useWebGPU` path".
- Verified by `dawn-facts.mjs` (can be promoted from "assumed" to [M]): 2.2 step 4 (Dawn-node
  default device limits are the spec defaults); 5.1 / 11.3 "`backend=null` where available" (it is
  available in 0.4.0); 2.6 row / R-22 / Q-16 "timestamp-query unquantised (assumed)" (Dawn-node
  reports ~1 us granularity, not 100 us). Also note the 2.6 row "minStorageBufferOffsetAlignment
  16" is the ADAPTER value; the default DEVICE reports 256.

### COMPLETE-19 (minor) -- Public-contract statements that contradict the mechanism
- Line 678: `step()` "resolves when `positions` holds the result of these iterations" vs 1835-1839
  (7.19 item 3: when saturated, return the OLDEST pending promise; the requested iterations are
  dropped, `iterationsSubmitted` does not grow). Say "may coalesce: the promise then belongs to an
  earlier batch and the requested iterations are not queued".
- Line 1758 / 2458: `iterationsPerStep` "default = `stepMultiplier`" -- the package does not know the
  element config; state the package default (1) and that the element passes `stepMultiplier`.
- Line 574: `E_IN_FLIGHT` declared, never thrown anywhere; delete or define.
- Lines 550 / 943: "the only error type the package throws" vs `E_GPU_INELIGIBLE` (a
  `GraphFormatError`) deliberately propagated; wrap it or amend D12.
- Lines 1355-1356 / 2365: `SimulationType` includes `"spring"` with no CPU class and no accelerator
  method named `spring`; say `"spring"` is an alias of `fruchtermanReingold` (the element's
  `SpringLayoutEngine` is FR) and distinct from the 7.20 `"spring-electrical"` preset.

### COMPLETE-20 (minor) -- Review-process residue reduces maintainability
- 83 "draft A/B/C" references and 49 "judge" references, plus a "Review notes" subsection per
  section, embed the three-draft review into the implementation plan; Q-18 (line 3251) commits the
  seven notes under `docs/research/` but says nothing about the drafts the plan cites.
- Fix: either commit the drafts next to the notes (so the citations resolve) or strip the
  "(graft: ...)" / "judge ..." annotations and move the Review notes into a separate
  `review-log.md` once the owner accepts the plan.

### COMPLETE-21 (minor) -- Time-to-"detected" understated
- Lines: 3180-3186 ("Critical path ... P6 makes it 'detected' in the app"), 2531-2533 (9.8 E1
  precondition "A2 first commit + L1"), 2384-2470 (9.4 assumes `DataManager.getSnapshot()`,
  `snapshot-replaced`, the element-owned position column -- all products of the design's E1 port).
- Claim: the 9.4 work is a delta ON TOP OF the design's E1 (and L1's steppable CPU FA2), which this
  plan does not schedule or size; a reader can take "20-27 ed + P6" as time-to-detection.
- Fix: add one sentence to 13's critical path: "detected-in-app additionally waits for F2, A2 (first
  commit), L1 and the design's E1 element port to land in the monorepo; until then the layout is
  usable from Node and from a story with an injected accelerator".

### COMPLETE-22 (minor) -- Small numeric inconsistencies in budgets and tables
- Lines 3161 (G2 "default lane <= 10 min") vs 2642 (T-12 "<= 15 min") vs 3122 (12.6 "target <= 15").
- Lines 2606-2607 (10.3): PageRank x 100 iterations is given as 20-70 ms at 100k and 0.2-0.7 s at
  1M, but 100 x the per-iteration column (0.2-1.4 ms; 2-14 ms) is 20-140 ms and 0.2-1.4 s; T-8's
  ceilings (100 ms, 1 s) sit below the upper brackets.
- Fix: one lane budget; recompute the x100 column or state the per-iteration figure it assumes.

## 4. What was checked and found consistent (no finding)

- `iterationsPerStep`, `maxInFlight = 2`, `settleThreshold = 1e-3`, `settleWindow = 10`, `nearMax =
  64`, `gridMax2D = 512`, `gridMax3D = 128`, `warnUnreleasedSnapshots = 2`, the 16,776,960 rule, the
  8-storage-buffer budget for every FA2 kernel (K2 = 7, K3 = 6, K5 = 7, G7 = 8), pyramid sizes
  (349,520 cells / 5.59 MB; 2,396,736 / 38.3 MB), far-field evaluation counts (196 / 982), key
  widths (18 / 21 bits, 3 radix passes), the 100k-tier arena figures, the FA2 port line citations
  (forceatlas2.ts 266, 316-330, 376-392), the LCG constants and the seed-0 quirk (`random.ts`), the
  monorepo `package.json` pins (lines 130 / 148 / 158 / 160), `ci.yml` shard matrix / Playwright /
  `all-checks` lines, `pool: "forks"` and 80/80/75/80 in graph-format's vitest config, the
  `webgpu@0.4.0` entry (`create`, `globals`), the four DEPARTUREs against the design text, the
  Buffalo / NVIDIA-cluster exclusions, the env-variable names across 2.3 / 11 / 12, the subpath
  export map, the Safari dynamic-import citation, the web-llm isolation citation.

## 5. Overall

No blocker: the plan is unusually complete and its four departures are real and justified. The
seven majors are of two kinds: (1) the plug-in half of "optional / detected" is specified for the
happy path only -- tuning cannot reach the element-created simulation (COMPLETE-2) and the
layout-after-failure path is missing (COMPLETE-3); (2) the plan contradicts itself on things it
also gates on -- the PageRank binding count (COMPLETE-1), the crossover default versus its own rule
(COMPLETE-4), P2's contents versus rule (b) and section 6 (COMPLETE-5), the preset versus Q-9
(COMPLETE-6) -- and it assumes a GitHub repository that does not exist yet (COMPLETE-7). The single
biggest risk to the owner's first need is schedule inflation before P3: P2 as written carries
roughly half a phase of work the exact-tier layout never touches.
