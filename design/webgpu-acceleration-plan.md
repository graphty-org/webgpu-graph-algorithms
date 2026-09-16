# @graphty/webgpu-graph-algorithms -- implementation plan for WebGPU-accelerated algorithms and layouts

Status: Approved by the owner on 2026-09-14 (plan of record; later changes
are appended to the Review log)
Date: 2026-09-14
Author: synthesis of three planning drafts (A performance-first, B integration-first,
C verification-first) after a three-judge review; draft B is the base, every
graftable idea the judges listed is folded in, every defect they listed is fixed
or answered in a "Review notes" subsection. A second, adversarial six-lens
review (performance realism, design confidence, maintainability, integration,
verifiability, completeness) was applied on 2026-09-14; the "Review log" at the
end of the document lists every finding that survived verification and what was
changed.

## The owner's request (verbatim)

"accelerated force directed layout is the first need. create a plan for
implementing webgpu accelerated algorithms and layouts. it should use our
graph-format package and be conscientious about high performance for a large
number of nodes. the plan is to plug it in as an optional / detected
acceleration for our existing algorithm and layout packages. use the information
from this context, especially the nvidia gpu algorithms, as input to the design.
the webgpu algorithms should run under both nodejs and browser, and should be
tested primarily under nodejs with some light browser testing to prove that they
will work. we will also need to configure ci/cd for GPU-based testing --
https://github.com/atoms-org/cuda-ffi may serve as an example or research for a
more recent approach of how to do GPU-based testing on GitHub. when this package
eventually moves under graphty-monorepo we may want one runner for GPU and a
default runner for other tests."

## How to read this document

- Fifteen numbered sections follow the required planning outline. Section 7
  (the force-directed layout) is the first deliverable and the deepest section;
  section 13 gives the phase order and the gate each phase must pass; section 14
  lists every decision left to the owner with a recommended default; section 15
  collects every URL.
- "design S.S, line N" cites the accepted design
  `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
  (section 10 = GPU contract, 14.3-14.6 = package integration and landing order,
  15 = performance model, 16 = testing). "note NN section X" cites the seven
  research notes under
  `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/` (01 layout
  needs, 02 algorithm needs, 03 GPU layout prior art, 04 GPU algorithm prior
  art, 05 WebGPU platform, 06 GPU CI, 07 format API and conventions); each note
  carries the URLs behind its facts and section 15 repeats them. "draft A/B/C
  S.S" cites the three drafts under the same directory.
- Every number carries a basis label: [M] measured on the dev box (probe
  scripts under `tmp/webgpu-plan/probe/` or `tmp/webgpu-plan/review/probes/`,
  or a cited test), [P] published by a
  paper or repository, [D] design arithmetic (design 15.1 byte model), [X]
  extrapolated from [M] or [P] with the scaling stated. Every [X] is replaced by
  a measurement at the phase gate named in section 10.4.
- "(graft: A 7.8)" marks an idea taken from another draft at the judges'
  request; "Review notes" at the end of a section records each judge finding
  about that section and what was done, including the two places where a judge
  finding was only partly right.
- "DEPARTURE" marks a deliberate difference from the accepted graph-format
  design; section 1.5 lists all of them in one place. Goal ids are `GOAL-n`
  (1.1) and phase-gate ids are `Gn` (section 13); the two never overlap.
- The "Review log" after section 15 records the adversarial review of
  2026-09-14 (finding ids `PERF-*`, `DESIGN-*`, `MAINT-*`, `INTEG-*`,
  `VERIFY-*`, `COMPLETE-*` and the verifier's `MISSED-*` / `M-*` additions,
  prefixed by lens where the ids collide). A change made for a finding is not
  annotated inline; the log's table maps each finding to the section changed.
- Vocabulary: n = node count, E = logical edge count, A = arc count
  (`snapshot.arcCount`; `2E - selfLoops` on undirected snapshots), dim = 2 or 3,
  WG = workgroup size (256), "ctx" = a `GpuContext`, "the 4070" = the dev box's
  NVIDIA RTX 4070 SUPER under Dawn-in-Node unless "Chromium" is stated,
  "element" = `@graphty/graphty-element`, "CPU packages" = `@graphty/algorithms`
  and `@graphty/layout`.
- Plain ASCII throughout; nothing in this document is implementation code
  beyond short illustrative snippets.

## Summary

- One TypeScript + WGSL code base, `@graphty/webgpu-graph-algorithms`, runs the
  same kernels under Google Dawn in Node (npm `webgpu@0.4.0`) and in browsers.
  The core never touches `navigator` or the native module; two thin subpath
  entries (`./browser`, `./node`) acquire the device. It consumes
  `@graphty/graph-format` snapshots exactly as implemented (arena hot-prefix,
  per-array and windowed uploads; `override USE_PERM`; `gpuView()`; `foldArcs`;
  `renumberPartition`; `INVALID_INDEX`).
- The first deliverable after a walking skeleton is a GPU ForceAtlas2
  `LayoutSimulation` (design 14.3) that graphty-element steps per frame with
  pins and drag, in 2D and 3D: exact tiled all-pairs repulsion below a measured
  node-count crossover (default 16,384, conservative until measured at G3) and
  a cell-sorted grid pyramid above it whose extent is robust to isolated nodes,
  attraction as a CSR-row gather, and the whole adaptive-speed controller on the
  device so k iterations are one submission with no host round trip. Every
  kernel fits the core default of 8 storage buffers per stage (the PageRank pull
  kernel of 8.2, the grid near field of 7.7 and five frontier / community
  kernels of 8.10 use all 8, each with its bindings named); no kernel
  scatters a float. Expected on the 4070: ~1.1-1.3 ms per iteration at 16k
  nodes [M], 4-10 ms at 100k nodes, 35-95 ms at 1M nodes (2D, conservative:
  the sorted-order near field measured 2-3 ms at 1M), with
  Fruchterman-Reingold on the same kernels.
- Plug-in: `@graphty/algorithms` and `@graphty/layout` OWN structural
  `AlgorithmAccelerator` / `LayoutAccelerator` interfaces plus one async
  dispatcher each; graphty-element exposes an `accelerator` property and a
  bridge from its synchronous frame loop to the asynchronous `step()`; the
  graphty app probes and injects. The GPU package never falls back; the only
  branch that selects the CPU is "no accelerator method", evaluated before any
  GPU work.
- Algorithms follow the layout, grouped by the primitive family they need:
  PageRank family (pull SpMV), connected components (Afforest), the frontier
  family (BFS, direction-optimizing BFS, SSSP near-far, closeness, sampled
  betweenness), then structure and community (k-core, triangles, label
  propagation, Louvain). Every design is a translation of cuGraph / Gunrock /
  GAP / Merrill / Davidson / McLaughlin-Bader into WGSL's constraints (u32 / i32
  atomics only, no cross-workgroup locks, subgroup size not constant).
- Testing is Node-first on Dawn (the full suite; the real GPU locally, Mesa
  lavapipe on hosted CI) with a light Playwright Chromium smoke suite; every
  GPU result is checked against a CPU oracle or an invariant, never "no error".
  CI has a default lane (`ubuntu-latest`, lavapipe + SwiftShader, required) and a
  GPU lane (a GitHub-hosted GPU larger runner, paid, in its OWN
  workflow so it can never gate a release), shaped to join the monorepo at
  move-in: the software shards join `ci.yml`, the GPU job lives in `gpu.yml`.

---------------------------------------------------------------------------

## 1. Goals, non-goals and inherited decisions

### 1.1 Goals

| Id | Goal | Source |
| --- | --- | --- |
| GOAL-1 | GPU-accelerated force-directed layout first: a `LayoutSimulation` for ForceAtlas2 (and Fruchterman-Reingold on the same kernels) that graphty-element can step per frame with pins and drag, and that Node can run in batch. | owner request; design 14.3 lines 3977-3985 |
| GOAL-2 | One code base under Node (Dawn via the `webgpu` npm package) and in browsers; the package never touches `navigator` and never imports the native module from its root entry. | owner request; note 05 section 1; note 02 finding 5 |
| GOAL-3 | Consume `@graphty/graph-format` snapshots exactly as implemented: arena hot prefix, per-array and windowed uploads, `override USE_PERM`, `gpuView()`, `foldArcs`, `renumberPartition`, `INVALID_INDEX`. | design 10; note 07 sections 1-3 |
| GOAL-4 | High performance at 10^5-10^6 nodes is a design driver: batched iterations per submit, no per-iteration readback, approximate repulsion above a measured crossover, degree-tier load balancing, every kernel designed for the core default limits. | owner request; design 15.3 |
| GOAL-5 | Optional / detected acceleration for the EXISTING algorithm and layout packages: the GPU package is injected by the caller, detection is the app's job, and the GPU package itself never falls back. | owner request; design lines 2324-2325 and 4243-4244; `/home/apowers/Projects/webgpu-graph-algorithms/CLAUDE.md` |
| GOAL-6 | Tests primarily in Node (Dawn), a light browser suite (Playwright Chromium) to prove the browser path, both on the real GPU locally and on software adapters on hosted CI. | owner request; note 05 section 9 |
| GOAL-7 | CI with a default lane (no GPU) and a GPU lane on a paid GitHub-hosted GPU runner (NVIDIA T4; the owner declined a self-hosted runner, 2026-09-14), designed to slot into the monorepo's `ci.yml` shard matrix later as "one runner for GPU and a default runner for other tests". | owner request; note 06 |
| GOAL-8 | A GPU algorithm library grouped by primitive family, with the NVIDIA / cuGraph / Gunrock / GAP techniques translated to WGSL's constraints. | owner request; note 04 |
| GOAL-9 | Every performance figure in this plan is replaced by a measured number at a named phase gate; the package README table is regenerated from `benchmarks/results/`. | judges' verifiability lens; graft: C 1.1 G6 (draft C's goal id) |

### 1.2 Non-goals (v1)

- No CPU, WebGL or "software mode" inside this package. A missing adapter is an
  error (`E_NO_ADAPTER`); the CPU packages are the default path, not a fallback
  (project rule; design 10.8 lines 2545-2550). Software adapters (lavapipe,
  SwiftShader) are a TEST environment the package happens to run on, never a
  product path.
- No rendering and no sharing of a `GPUDevice` with Babylon in v1. The element
  renders on WebGL by default; its `useWebGPU` branch
  (`graphty-element/src/managers/RenderManager.ts` lines 63-69 constructs a
  Babylon `WebGPUEngine`, and `Graph.ts` line 94 types `engine` as
  `WebGPUEngine | Engine`) exists but is unwired (no other reference in
  `src/`); positions are read back into the element's array. `GpuContext.from
  (device)` exists so that engine's `_device` (Babylon 8 exposes it) can be
  adopted later; until it is wired, `useWebGPU` plus an injected accelerator
  means two devices on one adapter with no buffer sharing (risk R-23).
- No `SharedArrayBuffer`, no worker orchestration inside the package (the format
  has no SAB in v1; a consumer may run a `GpuContext` in a worker and transfer
  snapshots with `toWire`).
- No `shader-f16` requirement (absent on the NVIDIA Vulkan path under both Dawn
  and Chromium 139, note 05 section 2.4), no compatibility-mode work beyond the
  defaults table (note 05 section 4); optional features are fast paths only.
- Not GPU targets (kept on the CPU path by the element's adapters): DFS,
  topological sort, cycle detection, Prim, Girvan-Newman, hierarchical /
  TeraHAC / GRSBM / SynC, MCL, max-flow / min-cut, bipartite matching,
  isomorphism, A* (note 02 section 7.2 last row).
- No dense-matrix betweenness (the Buffalo 2023-06 thesis is a negative result
  for sparse graphs, note 04 section 0 item 8) and nothing from the GPL
  `jaredmcqueen/analytics` repository (note 03 section 3).
- Kamada-Kawai, ARF and spectral layouts are later slices (note 02 section 7.1
  ranks them L4-L6); `nodeSize` / `adjustSizes` is deferred (section 7.14).

### 1.3 Decisions inherited from the graph-format design (not relitigated)

| Design section | Inherited decision | How this plan honours it |
| --- | --- | --- |
| 10.1 (lines 2327-2350) | The arrays a GPU package binds: `rowPtr`, `colIdx`, `weights`, `arcToEdge`, `edgeToArc`, `reverse()`, `coo().src`, `edgeList()`, degrees, `degreeOrder(opts).perm / segmentOffsets`, `mate()`, `gpuView()` columns; the weighted-degree normaliser is computed on the device; `override USE_PERM` with `colIdx` / `rowPtr` as never-read dummies for identity permutations; `segmentOffsets` read on the CPU. | 4.1 (residency keys), 6 row 3 (segmented reduce), 3.5 (WGSL prelude declares `USE_PERM`). |
| 10.2 (2352-2373) | 4-byte arrays over plain `ArrayBuffer`; `u8` via `paddedU32View()` + `unpack4xU8`; `bool` / masks as u32 bitmaps LSB-first; `components: 3` read as `array<f32>` with `3*i`, never `array<vec3<f32>>`; `f64` via the cached f32 `gpuView()`. | 4.3 (column uploads), 7.3 (positions as `array<f32>`). |
| 10.3 (2375-2441) | Upload plan: whole arena when the arena AND every segment fit the device limits; else per array (also the `arena === null` path); else windows. | 4.2 (planner), with one refinement marked DEPARTURE-2 in 1.5. |
| 10.4 (2443-2465) | `gpuEligibility`; results attach by reference through `nodes.set()`. | 9.7. |
| 10.5 (2467-2491) | Invariants the GPU may assume; never bind a zero-length array; guard division by weight sums; rows are sorted by target (I4). | 5.6 (empty ranges), 8.2 (PageRank normaliser), 8.5 (triangle merge). |
| 10.6 (2493-2520) | 1D dispatch legal iff `ceil(count / WG) <= maxComputeWorkgroupsPerDimension` (65,535 x 256 = 16,776,960, NOT 2^24); 2D grid or grid-stride above; windows start at 64-arc boundaries computed with `%`. | 5.2 (`plan1d` / `plan2d` / `planGridStride`), 4.2 (window planner). |
| 10.7 (2522-2543) | Readback conventions: index-aligned `Uint32Array` / `Float32Array`, `foldArcs` for per-arc results, copy out before `unmap()`, `dest?` on every algorithm. | 3.3 (`dest?` in `GpuRunOptions`), 4.4 (Readback). |
| 10.8 (2545-2550) | Device queries, chunking, 2D dispatch, frontier queues, scans, dense relabelling and "GPU unavailable" behaviour belong to the GPU package; it never falls back. | 2.4, 5, 6. |
| 14.2 (line 3738) | Label results are `{ labels: Uint32Array; count: number; groups(): Uint32Array[] }`; labels dense in first-seen order (`renumberPartition`, line 2978). | 3.3, 8.3, 9.7. |
| 14.3 (3959-4019) | `LayoutSimulation { load, step, settled, setFixed, setPosition, dispose }`; positions are the owner's stride-3 scene-unit array read AND written in place; the GPU buffer is authoritative while stepping; FA2 default mass `outDegree() + 1`; weights via `snapshot.weights` (`weight === true`) or a named edge column; `LayoutSimulation` and the position helpers are OWNED by `@graphty/layout`. | 7.19, 9.3. Two DEPARTURES: the formulas (DEPARTURE-3) and the device position layout under the stride-3 contract (DEPARTURE-7). |
| 14.4 (4048-4211) | `DataManager` owns the builder and the position array; `snapshot-replaced { previous, next, report }` triggers `accelerator.release(previous)`; adapters call `getSnapshot()` once and run `indexed.*` "(or the injected GPU accelerator)"; layouts get `dm.undirected(s).snapshot`; drag writes `simulation.setPosition`; `column.markDirty()` once per frame. | 9.4. |
| 14.5 (4212-4243) | Move-in as `webgpu-graph-algorithms/`; delete `CSRGraph` / `EdgeListGraph`; every entry takes `GraphSnapshot`; `noUncheckedIndexedAccess` OFF; upload cache keyed on the typed-array object, dropped via `column.version`, released by `release(snapshot)`; `parents` are `Uint32Array` with `INVALID_INDEX`; the element injects the accelerator per call (the design's one-line sketch spells it `runAlgorithm(snapshot, { accelerator: gpu })`). | 3, 4, 9. Three DEPARTURES: Node-first tests (DEPARTURE-1), the per-snapshot residency record beside the `WeakMap` (DEPARTURE-4) and the injection spelling `accelerated(acc).x(s, options)` from note 02 section 4.5 (DEPARTURE-5). |
| 14.6 (4245-4275) | Landing order F1 -> A1 -> F2 -> A2 / L1 / E1 -> W1 -> D1 -> IO1 -> 2.0; ownership of shared helpers (`foldArcs`, `renumberPartition` -> graph-format; `LayoutSimulation` -> layout; upload planning, chunking, readback, `release` -> webgpu-graph-algorithms). | 9.8. |
| 13.5 rules 3 and 5 (3638-3680) | Consumers declare `@graphty/graph-format` in BOTH `dependencies` and `peerDependencies`; `isGraphSnapshot()` is a brand check; NO consumer PR that adds the format to `dependencies` merges to master before the format is `>= 1.0.0`. | 3.1, 9.8 (preconditions), 13 (P6). The manifest uses `workspace:^`, not the design's `workspace:*`: pnpm publishes `workspace:*` as an EXACT pin (pnpm.io/workspaces; `npm view @graphty/graphty-element dependencies` shows `@graphty/algorithms: 1.7.2`), so rule 3's "published as a caret range" only holds with `workspace:^` (Q-31 proposes the same correction for graph-io and the design). |
| 15.3 (4349-4357) | Target tiers: 100k / 1M (mobile), 1M / 10M (desktop interactive), 10M / 100M (batch, raised limits and windowed bindings). | 10 uses exactly these tiers. |
| 16.2 (4535-4551) | Differential tests: `1e-5` relative for f32 GPU parity, order-agnostic for component lists; `validate({ checksum: true })` after each call. | 11.3. `1e-5` everywhere except two documented cases at `1e-4` (betweenness over many sources, 9.7; the one-iteration force parity of 11.4 with f32 tile summation) -- DEPARTURE-6, Q-24. |
| 16.6 (4622-4641) | A strict-consumer `tsc` compile of the published d.ts with `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` on. | 3.1, 11.3. |

### 1.4 Decisions this plan makes that the design left open

| Id | Decision | Where |
| --- | --- | --- |
| D1 | Node (Dawn) is the PRIMARY test project; the browser project is a light smoke suite (DEPARTURE-1). | 11, 12 |
| D2 | Runtime model: the core takes a `GPU` / `GPUAdapter` / `GPUDevice` from the caller; `./browser` and `./node` subpath entries do acquisition; `webgpu` (Dawn) is an optional peer loaded only by dynamic import inside `./node`. | 2 |
| D3 | Plug-in contract: `@graphty/algorithms` and `@graphty/layout` OWN structural `AlgorithmAccelerator` / `LayoutAccelerator` interfaces plus one async dispatcher each; the GPU package implements them with no runtime import of the CPU packages; no registry, no global state. | 9 |
| D4 | Detection lives in the graphty APP (probe + create + `element.setAccelerator(gpu)`); graphty-element only exposes the property and the bridges; an element-level `"auto"` loader is a later convenience using the web-llm isolation pattern. | 9.4-9.5 |
| D5 | ForceAtlas2 reference formulas for BOTH the CPU rewrite and the GPU kernel are the published algorithm as Gephi and cuGraph implement it (`1/d` repulsion magnitude, force-based swing / traction, fresh global sums each iteration) -- the DEFAULT, `compat: "paper"`; a `compat: "networkx"` option reproduces NetworkX `forceatlas2_layout` (its position-mixed per-node swing / traction accumulated across iterations from 1) so NetworkX trajectories are an external oracle and a migration target. The current port's own laws (`1/d^2`, reset position-based swing) are NOT preserved (DEPARTURE-3; owner decision Q-1, 2026-09-14). | 7.2 |
| D6 | Async step bridge: fire-and-forget, at most `maxInFlight` (default 2) batches in flight, positions copied into the owner's array when each batch's readback resolves; `settled` reflects the last completed batch. | 7.19 |
| D7 | Two repulsion back-ends selected by node count: exact tiled all-pairs (`n <= exactMaxNodes`, default 16,384 -- conservative until the G3 measurement; the 7.8 numbers predict 32,768 -- re-fixed by a mechanical rule, Q-6) and a cell-sorted grid pyramid (cosmos P3M re-expressed as compute kernels) above; a Hilbert-sorted cluster tree (GraphWaGu style) is the documented second experiment, not v1. `calibrateLayout(ctx)` suggests `exactMaxNodes` on the actual device (graft: A 7.8). | 7.6-7.8 |
| D8 | Settlement: `settled` at `maxIter` OR when the mean per-node displacement over a window is below a threshold relative to the RMS layout radius; `setPosition`, unpin and `load` reheat; adding pins does not. `reheat()` resets the iteration budget and the settle window ONLY; the speed controller (`speed`, `speedEfficiency`) is reset by `load()` alone, because Gephi never resets it mid-run and a drag calls `setPosition` every frame. | 7.17 |
| D9 | WGSL lives in `src/wgsl/*.wgsl.ts` template-string modules composed by string concatenation (tsc-only dist, knip, eslint all work); no `?raw`. | 3.5 |
| D10 | CI: default lane on `ubuntu-latest` (Dawn on Mesa lavapipe + Chromium on SwiftShader) on every PR; GPU lane on a GitHub-hosted GPU larger runner (Linux, 4 vCPU, NVIDIA T4; requires moving `graphty-org` to the Team plan; owner decision Q-3, 2026-09-14: no self-hosted runner, a paid hosted runner is acceptable) in its OWN workflow file (`gpu.yml`, never a job of `CI`: the monorepo's `release.yml` / `coverage.yml` are `workflow_run` on the CI workflow's conclusion, so any job of `CI` gates every release, and a paid lane must not run on every push of every package). The dev box's RTX 4070 SUPER is where the T-table performance targets are MEASURED (locally, by hand); the T4 lane checks correctness on real NVIDIA hardware and guards against regressions with per-runner-class baselines. | 12 |
| D11 | `webgpu` pinned to `0.4.0` until the dev container / runner image move to Ubuntu 24.04 in ONE scheduled environment change between the FA2 exact phase and the grid phase (draft C's question Q-10; here Q-4). | 2.5, 13 (P-ENV) |
| D12 | The package throws its own `WebGpuGraphError { code, details }` for every condition IT detects, with a stable code list including `E_RELEASED` (graft: C 5.6); errors raised by graph-format accessors the package calls on the caller's behalf (`gpuView()` -> `E_GPU_INELIGIBLE`, `ids.requireIndex()` -> `E_UNKNOWN_NODE`) propagate unchanged as `GraphFormatError` and are listed as pass-through codes in 5.7. | 3.3, 5.7 |
| D13 | GPU scores are `Float32Array`; the accelerator interfaces type scores as `NumericVector` (graph-format's `F32 \| F64 \| U32 \| I32`, `src/types/columns.ts` line 80; the GPU always returns the `F32` member); labels / parents are `Uint32Array` with `INVALID_INDEX`; labels are renumbered on the CPU in first-seen order with `renumberPartition`. | 9.7 |
| D14 | Layouts simulate in LAYOUT units on the device and apply `scale` / `center` in a write-back kernel; `setPosition` and `load` apply the inverse; never `rescaleLayout` per step. | 7.18 |
| D15 | FA2's swing / traction reductions and the adaptive-speed controller run on the device (workgroup partials + one-workgroup finalize kernels), so k iterations are one submission. | 7.10 |
| D16 | Every subgroup kernel reads `@builtin(subgroup_size)` and `@builtin(subgroup_invocation_id)` at RUNTIME for its lane-to-row mapping (the WGSL spec lets the compiler pick any size in `[subgroupMinSize, subgroupMaxSize]` per shader, and a subgroup may be partial); the only compile-time constant is `SUBGROUP_MAX = adapter.info.subgroupMaxSize`, used to size workgroup-memory scratch; a subgroup's slot in workgroup memory comes from an elected-lane `atomicAdd` on a `var<workgroup> atomic<u32>` counter broadcast with `subgroupBroadcast` (never `@builtin(subgroup_id)`, which needs the `subgroup_id` language feature Chromium 139 lacks); every such kernel has a workgroup-memory twin selected when `subgroups` is absent from `device.features`; tests force the twin by creating a context whose `optionalFeatures` exclude `subgroups` (the test setup maps `GRAPHTY_GPU_NO_SUBGROUPS=1` to that option). | 5.1, 6, 11.3 |
| D17 | Iterative algorithms run k rounds per `queue.submit` with `dispatchWorkgroupsIndirect` driven by device-side counters; host readback of convergence flags only every k rounds. | 5.4, 8 |
| D18 | Package skeleton mirrors `packages/graph-io` (both deps and peers on graph-format, `@webgpu/types` as a dependency, `./browser` and `./node` subpaths, coverage port 9058). | 3.1 |
| D19 | The test harness reads ONE adapter-policy variable, `GRAPHTY_GPU_REQUIRE` with values `any` / `hardware` / `nvidia` (unset = skip with a printed reason); the drafts' two near-identical variables are merged (graft: C 12.2 policy, judges' naming defect). | 11.2, 12.2 |
| D20 | Uniform structs are generated by one `UniformBlock` descriptor that emits both the padded WGSL struct text and the byte writer, so the two cannot disagree; a negative test proves a misaligned hand-written struct is rejected on Chromium (graft: C 5.3). | 5.3 |
| D21 | Every phase gate is a list of tests plus recorded MEASURED numbers that must be green on BOTH CI lanes; the owner signs off the FA2 formula table (7.2) in the P0 PR, before any WGSL that implements a 7.2 row merges -- P1's K3 / K4 included (graft: C 13). | 10.4, 13 |
| D22 | The exact-vs-approximate parity, the layout parity tolerance schedule and the force-sum invariant of section 11.4 are the acceptance tests of the layout slice (graft: C 11.4-11.6, corrected). | 11.4 |
| D23 | Device positions are `array<vec4f>` (xyz + mass, 16 B per node) in the simulation's own buffer; the owner's stride-3 array is repacked at `load()` / `setPosition` and unpacked by `toScene` (which already converts units), so one aligned load replaces four scalar loads in every gather and no kernel binds a separate `mass` buffer. Design 10.2's `array<vec3>` prohibition applies to `components: 3` COLUMNS, not to scratch. The mechanism differs from design 14.3 lines 3997-3999 (kernels take the STRIDE as a uniform and operate on the stride-3 column directly) and is declared as DEPARTURE-7. | 7.3 |
| D24 | The grid tier's far-field and near-field kernels are dispatched in CELL-SORTED order (`i = sortedIdx[t]`), never in node-index order: measured at 1M nodes (7.7 [M]) the same near-field body costs 8.4 / 32.6 / 125.3 ms in node-index order against 1.8 / 2.0 / 2.9 ms sorted (4.7x / 16x / 43x on the uniform / clustered / clustered + outliers fixtures; 9x / 18x / 70x once the `vec4f` positions of D23 are added, 7.3) and the far field 2.3-2.8 ms against 1.0 ms, because a subgroup then walks one cell instead of 32 unrelated ones. | 7.7 |
| D25 | No per-iteration displacement clamp anywhere in the integrate kernel: FA2's stability mechanism is the swing-based local speed; cosmos's `2 * cellSize` clamp bounds only its near-field VELOCITY term inside a fixed `spaceSize`, and applied to the whole step it caps the layout's expansion at `(1 + 4 / G)` per iteration (measured on the 7.2 laws: 0.78% at G = 512). The grid extent is made robust to isolated nodes instead (7.7). | 7.7, 7.11 |
| D26 | Layering: `src/device/` is acquisition, caps, error scopes and device loss only; `src/context.ts` is the named composition root above `memory/` and `kernel/`; `createAccelerator(ctx, options?)` and `calibrateLayout(ctx, options?)` are functions in `src/accelerator.ts` and `src/layouts/calibrate.ts`, never methods of the device layer, so a lower layer never imports a higher one and import-boundary lint can be enabled. | 3.2 |
| D27 | `@graphty/algorithms` and `@graphty/layout` are OPTIONAL peer dependencies of this package (owner decision Q-26, 2026-09-14) and `src/types/accelerator.ts` uses `import type` of the real `AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation` and option types from them -- types only, erased at build, so no runtime coupling and the bundle tests still assert no specifier. UNTIL those interfaces exist in published versions of the two packages (A2 / L1 land after this package's P0-P5), the file holds structural mirrors and `test/types/conformance.test-d.ts` asserts two-way assignability against the monorepo checkout; at W1 the mirrors are deleted and the `implements` clauses do the checking. Consequence accepted: the published d.ts references the two packages, so a consumer that type-checks against this package installs them (they are optional peers with `^1.0.0` ranges; the README's Node recipe lists them; `skipLibCheck` consumers that never touch the accelerator types are unaffected). | 3.3, 9.8 |
| D28 | Node-level layout inputs resolve by graph-format ROLE, on both the CPU and the GPU path: `mass` (default `outDegree() + 1`), `size`, `fixed`, `position` are the `KnownColumnRole`s graph-format already declares; a simulation reads `nodes.byRole(...)` at `load()`, accepts a `Float32Array` or a column name, and rejects the legacy `Record` form (`E_UNSUPPORTED`), which the caller converts into the role column (graphty-element at engine creation; `@graphty/layout`'s `resolveNodeVector` on the CPU path). Edge weights: `snapshot.weights` (the role-`weight` arcs) or a named edge column through `expandEdges`. Nothing from `@graphty/layout` is duplicated (owner decision Q-30, 2026-09-14). | 7.5, 7.14, 9.3, 9.4 |

### 1.5 Departures from the accepted graph-format design (all of them)

| Id | Design text | This plan | Why |
| --- | --- | --- | --- |
| DEPARTURE-1 | 14.5 line 4215: "a browser-only vitest project (Playwright Chromium on the real GPU)". | The `node` project on Dawn is primary; `browser` is a smoke suite (D1). 14.5, 14.6 (W1 gate: "browser tests green") and 16.7 are amended when this plan is accepted. | The owner's request supersedes it ("tested primarily under nodejs with some light browser testing"). |
| DEPARTURE-2 | 10.3 line 2425: whole-arena path when `arena.byteLength <= maxBufferSize`. | The arena buffer is created at `arena.hotByteLength` (the traversal prefix) unless the caller's first upload explicitly needs a cold segment AND `arena.byteLength` also fits; cold segments requested later are separate buffers sourced zero-copy from the arena bytes (4.2). | At the 1M / 10M undirected weighted tier the hot prefix is 164,000,256 B (fits the 256 MiB default) while the full arena with `arcToEdge` + `edgeToArc` is 284,000,256 B (does not; both figures from graph-format's `layoutSegments`, `tmp/webgpu-plan/review/probes/arena-bytes.mjs`: `rowPtr` = 4,000,004 B pads to 4,000,256); sizing to the hot prefix keeps the desktop tier on the one-buffer path at default limits. |
| DEPARTURE-3 | 14.3 lines 4028-4035 (layout port 2): "swing / traction / adaptive speed as today". | The L1 CPU rewrite and the GPU kernel adopt the published ForceAtlas2 laws (7.2, D5) with a `compat: "networkx"` option; the port's own variants are dropped (Q-1, decided). | Note 01 section 2.1.9 shows the port's `1/d^2` repulsion and position-based swing / traction are transcription deviations from NetworkX, Gephi and cuGraph; one reference for both paths is what makes parity tests meaningful (owner question Q-1). |
| DEPARTURE-4 | 14.5 line 4231: "cached in a `WeakMap` keyed on the typed-array object". | The `WeakMap` is kept and joined by a per-snapshot residency record indexed by `snapshot.serial` (4.1); `withColumns()` siblings, which share the core AND the serial (`graph-snapshot.ts` lines 923-942), are one residency unit. | `dropCaches()` changes view array objects (note 07 section 2 item 3), so `release(snapshot)` must find every buffer through a record, not through the array objects; the serial is the core's identity, so it cannot distinguish siblings and the plan does not pretend it can (Q-27). |
| DEPARTURE-5 | 14.5 lines 4239-4240: the element "injects it as `runAlgorithm(snapshot, { accelerator: gpu })`". | Per-call injection is spelled `accelerated(acc).pageRank(s, options)`: one dispatcher object owned by `@graphty/algorithms` (9.2), no per-function option plumbing. | The design line is a one-line sketch with no signature; note 02 section 4.5 recommends the dispatcher shape because the "no accelerator method -> CPU" decision then lives in one place. Same mechanism, different spelling. |
| DEPARTURE-6 | 16.2 line 4545: "`1e-5` for f32 GPU parity". | `1e-4` relative for betweenness (f32 accumulation over many sources, 9.7) and for the one-iteration force parity of 11.4 (f32 tile summation order against an f64 oracle); `1e-5` everywhere else. | Both are f32 accumulation-length effects the design's blanket number did not anticipate (Q-24); recorded here so 16.2 is amended at W1 rather than silently missed. |
| DEPARTURE-7 | 14.3 lines 3997-3999: `LayoutSimulation` "kernels take the position STRIDE (3) as a uniform and operate on the owner's stride-3 column directly, so no per-frame `withComponents` copy exists in either direction". | The kernels operate on the simulation's own `array<vec4f>` device buffer (xyz + mass; D23, 7.3); the owner's stride-3 array is read once at `load()` / `setPosition` (repacked with the inverse unit scale) and written by every readback through the `toScene` kernel (7.18). No `withComponents` copy exists in either direction, so the sentence's outcome holds; its mechanism (the stride as a uniform, kernels on the stride-3 column) does not. | One aligned 16-byte load per gather instead of three scalar loads plus a separate `mass` fetch: the sorted near field at 1M nodes drops 1.80 -> 0.92 ms uniform and 2.94 -> 1.79 ms with outliers (7.3 [M]), and the repack rides on the unit conversion `toScene` performs anyway. 14.3 is amended in the L1 PR alongside DEPARTURE-3. |

Everything else in sections 10, 14.3-14.6, 15 and 16 is honoured as written.

### Review notes (section 1)

- Judges (all three) flagged the drafts' reference to a `flags.sortedRows`
  flag. `SnapshotFlags` (`packages/graph-format/src/types/snapshot.ts` lines
  161-181) has no such flag; rows are ALWAYS sorted (invariant I4, design 10.5).
  This plan cites I4 wherever sorted rows are relied on.
- Judge integration-feasibility asked for the `groups(): U32[]` shape to be
  cited: design 14.2 line 3738 (table row "components, communities").

---------------------------------------------------------------------------

## 2. Runtime model

### 2.1 One code base, three entry points

The library is split into a runtime-agnostic CORE and two thin ACQUISITION
entries. The core contains every kernel, planner and algorithm; it never
references `navigator`, `window`, `process` or the `webgpu` module. This is the
single property that makes Node and browser share one code base (note 02
finding 5; note 05 section 1 item 1).

```
@graphty/webgpu-graph-algorithms           (".")   core: GpuContext.probe/create/from, createAccelerator, calibrateLayout, algorithms, layouts, errors, types
@graphty/webgpu-graph-algorithms/browser   uses globalThis.navigator.gpu; probeBrowserWebGpu(), requestGpuContext()
@graphty/webgpu-graph-algorithms/node      dynamic import("webgpu") (Dawn); createNodeGpu(), createNodeGpuContext(), probeNodeWebGpu()
```

Two rules keep the core import-order independent under Dawn-node, where
`GPUBufferUsage`, `GPUMapMode` and `GPUShaderStage` do not exist on
`globalThis` until `Object.assign(globalThis, dawn.globals)` has run (note 05
section 2.2, verified against the installed `webgpu@0.4.0`
`packages/graph-format/node_modules/webgpu/index.js`):

1. The core never reads those namespaces at module top level. It keeps its own
   numeric constants in `src/device/webgpu-constants.ts` (`BufferUsage.STORAGE
   = 0x0080`, `COPY_SRC = 0x0004`, `COPY_DST = 0x0008`, `MAP_READ = 0x0001`,
   `UNIFORM = 0x0040`, `INDIRECT = 0x0100`, `MapMode.READ = 0x0001`,
   `ShaderStage.COMPUTE = 0x4`), each with a comment naming the spec constant.
   `test/device/constants.test.ts` compares them with the browser / Dawn globals
   at runtime so a drift is caught.
2. The `./node` entry installs `dawn.globals` anyway (user code and
   `@webgpu/types`-typed helpers expect them), but the core does not depend on
   it having happened.

### 2.2 Device acquisition (core)

```ts
// src/context.ts (public; the composition root of D26 -- src/device/ holds only acquire.ts, caps.ts, webgpu-constants.ts, error-scope.ts, lost.ts)
export type LimitPolicy = "default" | "raise" | Readonly<Partial<Record<RaisableLimit, number>>>;
export type RaisableLimit =
    | "maxBufferSize" | "maxStorageBufferBindingSize" | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupStorageSize" | "maxComputeInvocationsPerWorkgroup" | "maxComputeWorkgroupSizeX";
    // maxComputeWorkgroupsPerDimension is NOT raisable here: the 4070 reports the spec minimum 65,535 and cannot raise it
    // (tmp/webgpu-plan/review/probes/maint-sync-probe.mjs), and the prelude's linear_id bakes the same constant (3.5)

export interface GpuContextOptions {
    readonly gpu?: GPU | undefined;               // browser: navigator.gpu; Node: dawn.create([...])
    readonly adapter?: GPUAdapter | undefined;    // skip requestAdapter(); the adapter must be UNUSED (see step 1)
    readonly device?: GPUDevice | undefined;      // adopt a device the caller owns (e.g. Babylon's WebGPUEngine._device once wired)
    readonly powerPreference?: GPUPowerPreference | undefined;   // default "high-performance" (Chrome 145 needs it, note 05 section 3.2)
    readonly rejectSoftware?: boolean | undefined;   // default false; true -> E_SOFTWARE_ONLY instead of a device on lavapipe / SwiftShader
    readonly limits?: LimitPolicy | undefined;    // default "raise"
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;   // default ["subgroups", "timestamp-query"]; requested only when the adapter has them; tests pass [] to force the subgroup twins (D16)
    readonly requiredFeatures?: readonly GPUFeatureName[] | undefined;   // default []; missing -> E_NO_DEVICE
    readonly label?: string | undefined;
    readonly onError?: ((error: WebGpuGraphError) => void) | undefined;  // uncapturederror sink; default: keep the last error and throw it from the next public call (5.7)
    readonly warnUnreleasedSnapshots?: number | undefined;   // default 2: console.warn once when more snapshots than this are resident (graft: C R-18)
}
```

`GpuContext.create(options)` runs, in order:

1. `options.device` given -> adopt (`ownsDevice = false`); else `options.adapter`
   given -> step 2; else `options.gpu.requestAdapter({ powerPreference })`;
   `null` -> throw `E_NO_ADAPTER`. A `GPUAdapter` is CONSUMED by its first
   `requestDevice` (WebGPU spec 3.5.1 "each adapter object can only be used to
   create one device"; Dawn-node 0.4.0 rejects the second call with
   `OperationError: adapter is "consumed"`, `tmp/webgpu-plan/review/probes/
   uncaptured-order-probe.mjs` [M]), so `create({ adapter })` on an adapter that
   already created a device throws `E_NO_DEVICE { reason: "consumed" }`, and
   every recovery path after device loss starts again from
   `gpu.requestAdapter()`, never from a kept adapter.
2. Read `adapter.info` (vendor, architecture, device, description,
   `subgroupMinSize`, `subgroupMaxSize`); compute `software =
   isSoftwareAdapter(info)` = `architecture === "software"` (Dawn-node
   llvmpipe) OR `architecture === "swiftshader"` OR `info.isFallbackAdapter ===
   true` (Chromium; `undefined` in Dawn-node 0.4.0, note 05 section 2.2).
   `options.rejectSoftware && software` -> throw `E_SOFTWARE_ONLY` before any
   device is created.
3. Build `requiredLimits` from the policy: `"raise"` takes each `RaisableLimit`
   from `adapter.limits` (clamped to what the adapter reports, never above),
   `"default"` requests nothing, an object requests exactly those values
   (rejecting a value above `adapter.limits` with `E_NO_DEVICE` before calling
   `requestDevice`, so the failure is diagnosable). Build `requiredFeatures =
   required + (optional intersect adapter.features)`. `requestDevice({
   requiredLimits, requiredFeatures, label })`; a rejection -> `E_NO_DEVICE`
   with the adapter summary in `details`.
4. Capture `GpuCaps` from `device.limits` (never `adapter.limits`: Dawn-node
   reports 1 TiB / offset alignment 16 on the ADAPTER while a default device
   carries the spec defaults -- `maxBufferSize` 268,435,456, binding
   134,217,728, 8 storage buffers, `minStorageBufferOffsetAlignment` and
   `minUniformBufferOffsetAlignment` 256 -- verified on NVIDIA and llvmpipe
   [M], `tmp/webgpu-plan/review/probes/dawn-facts.mjs`), `device.features`, the
   adapter info, `runtime: "browser" | "node" | "unknown"` (set by the ENTRY
   that created the context, never by sniffing globals; `GpuContext.from
   (device)` sets `"unknown"`).
5. Install `device.addEventListener("uncapturederror", ...)` (works identically
   in Dawn-node and browsers, note 05 section 7.3; under Dawn-node the event is
   delivered SYNCHRONOUSLY before `queue.submit()` returns, [M]
   `uncaptured-order-probe.mjs`, which 5.8 exploits) and chain `device.lost`
   into `ctx.lost`; on loss the context enters `state = "lost"`, every pending
   promise rejects with `E_DEVICE_LOST`, and every residency record is dropped
   (the buffers are gone with the device).
6. Create the singletons the context owns: `PipelineCache`, `BufferPool`,
   `Readback` staging ring, `GraphResidency`, `Profiler` (when
   `timestamp-query` was granted). This is why `GpuContext` lives in
   `src/context.ts` above `memory/` and `kernel/` (D26): the device layer
   itself constructs nothing above it.

`GpuContext.from(device, info?)` is the zero-cost adoption path for callers
that already have a device (tests, a shared Babylon device once wired). `probe`
never creates a device:

```ts
export interface AdapterSummary {
    readonly vendor: string; readonly architecture: string; readonly device: string; readonly description: string;
    readonly software: boolean; readonly subgroupMinSize: number; readonly subgroupMaxSize: number;
    readonly features: readonly string[]; readonly limits: Readonly<Record<string, number>>;
}
export interface ProbeResult {
    readonly ok: boolean;
    readonly code: "OK" | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_SOFTWARE_ONLY";
    readonly reason: string | null;
    readonly adapter: GPUAdapter | null;          // the UNUSED adapter the probe requested: pass it to create({ adapter }) so create() does not request a second, possibly different, adapter
    readonly summary: AdapterSummary | null;      // what the app displays / logs
}
export interface ProbeOptions { readonly gpu: GPU; readonly powerPreference?: GPUPowerPreference; readonly rejectSoftware?: boolean; }  // rejectSoftware default false
GpuContext.probe(options: ProbeOptions): Promise<ProbeResult>
```

`rejectSoftware: true` turns a lavapipe / SwiftShader adapter into
`E_SOFTWARE_ONLY` (in `probe` and in `create`, the same rule) so an APP can
decide "no acceleration on software adapters" without the package deciding for
it (the package itself runs on software adapters -- that is how the default CI
lane works, section 12).

`calibrateLayout(ctx, options?: CalibrateOptions): Promise<GpuCalibration>`
(`src/layouts/calibrate.ts`; graft: A 7.8) is the layout-tier micro-benchmark:

```ts
export interface CalibrateOptions { readonly sizes?: readonly number[] | undefined; }   // default [8_192, 16_384, 32_768, 65_536]
export interface GpuCalibration {
    readonly pairsPerSecond: number;                       // at the largest probed size
    readonly exactMsPerIter: Readonly<Record<number, number>>;
    readonly gridMsPerIter: Readonly<Record<number, number>>;   // the grid build + far + near field at the SAME sizes
    readonly suggestedExactMaxNodes: number;               // largest probed n with exactMs(n) <= min(4 ms, gridMs(n)), rounded down to a power of two
    readonly firstCallMs: number;                          // pipeline compilation included; the app should call it once, off the critical path
}
```

Each size runs 10 timed iterations after `PipelineCache.warm` and one warm-up
submit; the first call costs pipeline compilation of ~20 pipelines plus
50-100 ms of timed work on a discrete GPU, later calls 50-100 ms. The app
passes `suggestedExactMaxNodes` to `createAccelerator(ctx, { layout: {
exactMaxNodes } })` (3.3, 9.5); the package never calls it implicitly and the
`"auto"` crossover never depends on `caps.software` (7.8, Q-6).

### 2.3 The detection helper, per runtime

| Runtime | Entry | Helper | What it does |
| --- | --- | --- | --- |
| Browser | `./browser` | `probeBrowserWebGpu(opts?)` | `navigator.gpu` absent -> `{ ok: false, code: "E_NO_WEBGPU" }`; else `GpuContext.probe({ gpu: navigator.gpu, ...opts })`. Never throws. |
| Browser | `./browser` | `requestGpuContext(opts?)` | `GpuContext.create({ gpu: navigator.gpu, powerPreference: "high-performance", ...opts })`; throws `E_NO_WEBGPU` / `E_NO_ADAPTER` / `E_NO_DEVICE`. |
| Node | `./node` | `createNodeGpu(opts?)` | `await import("webgpu")` (throws `E_NO_WEBGPU` with the loader message when the native module is missing or its glibc is too old), `Object.assign(globalThis, dawn.globals)` unless `installGlobals: false`, `dawn.create(flags)` where flags come from `{ adapter?: string; backend?: string; dawnFeatures?: string[]; software?: boolean }` (`software: true` -> `adapter=llvmpipe`, documented as Linux / Mesa specific; note 05 section 2.5). Returns `{ gpu, dispose() }`; `dispose()` drops the reference so the process can exit (note 05 section 2.2, lifetime). |
| Node | `./node` | `probeNodeWebGpu(opts?)` | `createNodeGpu` + `GpuContext.probe` + `dispose`; never throws. |
| Node | `./node` | `createNodeGpuContext(opts?)` | `createNodeGpu` + `GpuContext.create`; `ctx.dispose()` also disposes the `GPU` handle. |
| Any | `.` | `GpuContext.probe({ gpu })`, `GpuContext.create({ gpu \| adapter \| device })`, `isSoftwareAdapter(info)` | the runtime-agnostic primitives the two entries are built on. |

Environment variables are never read by `src/`. Their three readers are
`test/setup/gpu.ts` (Node projects, section 11.2), `vitest.config.ts` (which
forwards `GRAPHTY_GPU_REQUIRE` / `GRAPHTY_BROWSER_GPU` into the browser project
through `test.env` at config-evaluation time, where `test/setup/browser.ts`
reads them from `import.meta.env` -- Chromium has no `process.env`) and
`scripts/gpu-report.js` (12.3). The adapter-policy rule of 11.2 is parsed and
checked by ONE module, `scripts/gpu-policy.js` (`parseGpuRequire(env) -> {
level, vendor }`, `checkAdapter(info, policy)`), imported by both the test
setup and the report script so the vendor-match rule has one copy.

### 2.4 The no-fallback rule inside the package versus optional / detected acceleration in consumers

The rule is enforced by WHERE decisions are made, not by discipline:

| Question | Who answers | How |
| --- | --- | --- |
| Is WebGPU present, is the adapter hardware, is it worth using? | The APP (graphty) at start-up, or a Node script | `probe*()` then `create*()`; the app sets `element.setAccelerator(gpu)` or leaves it `null`. A node-count threshold for "use the GPU layout only above N nodes" is the element's `behavior.layout.gpuMinNodes` setting, evaluated by `LayoutManager` at engine creation and again on every `load` / `reload` (9.4 item 7), not package logic (note 02 section 7.1, L3). |
| No accelerator injected | The CPU packages' dispatcher (`accelerated(null)`) | runs the CPU implementation. This is the ONLY branch that chooses CPU, evaluated BEFORE any GPU work. |
| Accelerator injected, method missing (not yet implemented on the GPU) | The dispatcher | `acc.pageRank === undefined` -> CPU. Also evaluated before any GPU work; the GPU package declares only the methods it implements. |
| Accelerator injected, method throws (`E_DEVICE_LOST`, `E_OUT_OF_MEMORY`, `E_VALIDATION`, `E_UNSUPPORTED`, `E_TOO_LARGE`) | The caller (graphty-element's operation queue reports a failed run; a Node script sees the rejection) | the error propagates. The element MAY offer "disable GPU acceleration" as a user action; that is a user decision, not a fallback. |
| Graph too large for the device (needs windowing the algorithm does not support yet) | The GPU package | throws `E_TOO_LARGE` with `details { needed, limit, path, algorithm }` BEFORE allocating; never silently degrades. |
| Software adapter (lavapipe / SwiftShader) | The GPU package runs on it (correctness); the APP decides whether to inject it (`rejectSoftware`) | see 2.2. |

Consequences written into the code: `src/` contains no `try { gpu } catch { cpu
}`, no RUNTIME import of `@graphty/algorithms` or `@graphty/layout` (`import
type` from the two is allowed in exactly one file, `src/types/accelerator.ts`,
D27; `@typescript-eslint/no-restricted-imports` with `allowTypeImports` for
that file only), and -- outside `src/browser/**`,
the one directory that must read `globalThis.navigator.gpu` -- no reference to
`navigator`. Lint rules in a package-local `eslint.config.js` that imports and
spreads the root config (the precedent is `graphty-element/eslint.config.js`
lines 1-9; no root config edit is needed) enforce the last two:
`no-restricted-imports` over `src/**`, `no-restricted-globals` over `src/**`
with `src/browser/**` in its `ignores`, and the mirror rule that
`src/browser/**` and `src/node/**` are imported by nothing in `src/` outside
themselves. `caps.software` is read inside `src/` by exactly two functions,
`isSoftwareAdapter` (which computes it) and `planGridStride` (a performance
default for the workgroup count, 5.2, never a behaviour decision); a test greps
for any third reader.

### 2.5 Subpath exports and keeping the native module out of browser bundles

```jsonc
// package.json (excerpt; full manifest in section 3.1)
"exports": {
    ".":         { "types": "./dist/webgpu-graph-algorithms.d.ts", "import": "./dist/webgpu-graph-algorithms.js", "default": "./dist/webgpu-graph-algorithms.js" },
    "./browser": { "types": "./dist/browser.d.ts",                 "import": "./dist/browser.js",                 "default": "./dist/browser.js" },
    "./node":    { "types": "./dist/node.d.ts",                    "import": "./dist/node.js",                    "default": "./dist/node.js" }
},
"sideEffects": false,
"dependencies":     { "@graphty/graph-format": "workspace:^", "@webgpu/types": "^0.1.72" },
"peerDependencies": { "@graphty/graph-format": "^0.1.0", "webgpu": ">=0.4.0 <1.0.0", "@graphty/algorithms": "^1.0.0", "@graphty/layout": "^1.0.0" },
"peerDependenciesMeta": { "webgpu": { "optional": true }, "@graphty/algorithms": { "optional": true }, "@graphty/layout": { "optional": true } },
"devDependencies":  { "@graphty/algorithms": "workspace:^", "@graphty/layout": "workspace:^" }   // from W1 (types only, D27); before W1 the mirrors stand in
```

`workspace:^` (not `workspace:*`) because pnpm rewrites `workspace:*` to an
EXACT version at publish time and `workspace:^` to a caret range
(pnpm.io/workspaces); the design's 13.5 rule 3 expects a caret range (1.3,
Q-31). The `webgpu` peer range is written to admit the 0.6.x that P-ENV moves
to (`^0.4.0` on a 0.x package means `>=0.4.0 <0.5.0`); the devDependency and
the `E_NO_WEBGPU` install hint carry the exact pinned version and are updated
at P-ENV.

Mechanisms, each with the test that guards it:

1. The `webgpu` module is IMPORTED by exactly one source file,
   `src/node/index.ts`, and only inside `await import("webgpu")` within a
   function body (the word itself also appears in the package name, in
   `kind: "webgpu"` on the accelerator and in `E_NO_WEBGPU`, so no test is a
   substring match). `test/build-output.test.ts` parses `dist/webgpu-graph-
   algorithms.js` and `dist/browser.js` for import / require SPECIFIERS
   (`from "webgpu"`, `import("webgpu")`, `require("webgpu")`, both quote
   styles) and asserts none names `webgpu`; it asserts `dist/node.js` names it
   only in a dynamic `import()`. `./browser` and `./node` are bundle-only
   outputs (`dist/browser.js`, `dist/node.js` and their one-line d.ts shims
   exist only after `build:bundle`), so the test HARD-FAILS when the bundle is
   absent and `process.env.CI` is set (graph-io's `it.skipIf(!bundleExists)`
   pattern is kept for local tsc-only runs only); every CI lane builds with
   `pnpm run build` at the workspace root, which is `pnpm -r run build:all`
   (`packages/package.json` line 8), never the tsc-only `pnpm -r run build`.
2. The vite library build (`scripts/build-bundle.js`, copied from
   `packages/graph-io/scripts/build-bundle.js` lines 33-47 which externalise
   every `dependencies` + `peerDependencies` name) marks `webgpu` and
   `@graphty/graph-format` external, so a consumer's bundler resolves them --
   and only resolves `webgpu` if the consumer imported `./node`.
   `scripts/entries.js` maps `{ "webgpu-graph-algorithms": "src/index.ts",
   browser: "src/browser/index.ts", node: "src/node/index.ts" }` exactly as
   graph-io's does; there is NO root `webgpu-graph-algorithms.ts` shim
   (graph-format's convention), because two barrels drift and graph-io, the
   package mirrored file for file, has none.
3. `"sideEffects": false` lets bundlers drop unused algorithms; nothing in the
   package registers itself anywhere (D3 rejects the registry pattern, note 02
   section 4.2).
4. The optional peer means `npm` / `pnpm` do not install `webgpu` for browser
   consumers ("Npm will not automatically install optional peer dependencies",
   note 02 sources). A Node consumer that forgets it gets `E_NO_WEBGPU` with
   the message "install the optional peer dependency webgpu@0.4.0".
5. graphty-element learned that Safari fails on dynamic imports of non-existent
   modules even before the import is called
   (`graphty-element/src/ai/providers/index.ts:9-13`, note 02 finding 3). That
   concerns a browser importing a MISSING module; `./node` is never reachable
   from browser code because no browser-side file imports it. The rule is
   written into the package `CLAUDE.md`: "`./node` is imported only by Node
   entry points and tests".
6. No `"browser"` field and no conditional `"node"` export condition: an
   explicit subpath is unambiguous, while a condition can be flipped by
   bundler configuration and would let a browser bundle pull `node.js`.

Version pins (note 05 sections 2.1 and 9.2; note 07 section 4.2): `webgpu@0.4.0`
(the last Linux binary linking against glibc <= 2.34; 0.6.1 needs `GLIBC_2.38`
and fails to `require()` on the Ubuntu 22.04 / glibc 2.35 dev container --
verified with `strings` and a failed load; the 0.5.x binaries were NOT
inspected, note 05 section 12 item 1, so "0.5+" is assumed and only 0.6.1 is
verified), `@webgpu/types ^0.1.72`, `vitest` / `@vitest/browser ^3.2.4`,
`playwright ^1.54.1` (Chromium build 1181 = Chromium 139, the build verified
on the 4070), `vite ^7`, matching the monorepo root `package.json` lines 130,
148, 158, 160. One `webgpu` version in both CI lanes; bumped once, everywhere,
at P-ENV (section 13), which also re-pins the devDependency and the install
hint.

### 2.6 Runtime differences the device layer absorbs

| Difference (note 05) | Dawn-node 0.4.0 | Browsers | What the device layer does |
| --- | --- | --- | --- |
| `adapter.info.isFallbackAdapter` (`@webgpu/types` 0.1.72 puts it on `GPUAdapterInfo`, `dist/index.d.ts` line 2229) | `undefined` | boolean | `isSoftwareAdapter(info)` tests `architecture` first, `isFallbackAdapter` second. |
| `forceFallbackAdapter`, `powerPreference` | ignored | honoured | Node adapter choice is `adapter=` / `backend=` strings only. |
| One device per adapter | `requestDevice` a second time -> `OperationError: adapter is "consumed"` [M] | same (spec 3.5.1) | `create({ adapter })` on a used adapter -> `E_NO_DEVICE { reason: "consumed" }`; the test harness requests a fresh adapter for every device it creates (11.2); recovery after loss starts from `requestAdapter()`. |
| `minStorageBufferOffsetAlignment` / `minUniformBufferOffsetAlignment` | ADAPTER reports 16 / 64; a default DEVICE is 256 unless a smaller value is requested [M] | 256 | always align to 256 (graph-format arena is 256-aligned); never assert `=== 256`; never request the adapter's smaller value. |
| `maxBufferSize` on the adapter | 1 TiB (driver value) | 4 GiB | plan against `device.limits`; wrap large `createBuffer` in `pushErrorScope("out-of-memory")`. |
| `wgslLanguageFeatures` | 9 incl. `uniform_buffer_standard_layout` and `subgroup_id` | 4 in Chromium 139 (no `subgroup_id`) | uniform structs generated with strict 16-byte layout by `UniformBlock` (5.3); every variant compiled on BOTH runtimes in CI (11.3); subgroup slots allocated without `@builtin(subgroup_id)` (D16). |
| `shader-f16` on NVIDIA | absent | absent (present on lavapipe) | never required; not used in v1. |
| `subgroups` size | 32 (NVIDIA), 8 (lavapipe); min == max on both | 32 (NVIDIA), 4 (SwiftShader); Intel Xe / Arc report 8-32, AMD 32-64 | kernels read `@builtin(subgroup_size)` at runtime (available in Dawn 0.4.0, `tmp/webgpu-plan/review/probes/design-probe-*.log` item 5) and take only `SUBGROUP_MAX = adapter.info.subgroupMaxSize` as a compile-time constant (D16); a faked-caps test with `min != max` asserts no result depends on the override. |
| `timestamp-query` resolution | unquantised: 1,024 ns ticks [M] (`design-probe-nvidia.log` item 6) | 100 us quantised (Chrome 121+) | profiler reports `quantised: true` in browsers; per-kernel profiling is a Node activity. |
| `getMappedRange()` after `device.destroy()` | stays attached in 0.4.0 (0.6.1 adds a shim) | detached | `Readback` always `unmap()`s / `destroy()`s its staging buffers itself. |
| `mapAsync` round trip | 0.03-0.04 ms [M] | 0.10-0.15 ms [M] | both are far below one frame; batching decisions are made for the browser number. |
| Uncaptured validation errors | printed to stderr, no throw | console, no throw | `uncapturederror` listener + error scopes in both. |
| `adapter.requestAdapterInfo()` | absent | removed (Chrome 131) | `adapter.info` only. |
| Process lifetime | process stays alive while a `GPU` reference is reachable | n/a | `createNodeGpu().dispose()` drops the reference; `ctx.dispose()` calls it. |

### 2.7 Threads and workers

`navigator.gpu` exists in dedicated workers, so a `GpuContext` may be created in
a worker and fed snapshots through `toWire({ transfer: true })` (design 16.3).
Nothing in the package assumes the main thread. v1 does not ship a worker
wrapper; graphty-element runs the layout on the main thread because its frame
loop is synchronous (note 01 section 4.1) and the GPU work is asynchronous
anyway.

### 2.8 Lifetime

`ctx.dispose()`: rejects pending work with `E_DISPOSED`, destroys every buffer
in the residency, the pool and the staging ring, destroys the device if
`ownsDevice`, and (Node) disposes the `GPU` handle if the context created it.
`ctx.release(snapshot)` destroys only that snapshot's buffers (section 4.5).
Nothing is freed by garbage collection (design 14.4 line 4123: "which no
`WeakMap` can do for it").

### Review notes (section 2)

- Judge integration-feasibility (drafts A and C): the app detection sketch
  called `GpuContext.probe` before importing the module that exports it. Here
  the app imports `./browser` statically (9.5); a code-split is the app's
  choice and must import first.
- Judge verifiability (draft C): `GpuCaps.runtime: "browser" | "node"` had no
  value for `GpuContext.from(device)`; `"unknown"` is added.
- The env-variable naming defect (A 11.1, C 11.2 / 12.2: `GRAPHTY_REQUIRE_GPU`
  versus `GRAPHTY_GPU_REQUIRE`) is resolved by D19: one variable.

---------------------------------------------------------------------------

## 3. Package architecture

### 3.1 Skeleton and manifest

The package lives at `packages/webgpu-graph-algorithms/` in this staging repo
and moves to `webgpu-graph-algorithms/` in the monorepo at W1 (design 14.5). It
mirrors `packages/graph-io` file for file (note 07 section 4; the
`packages/README.md` move checklist, lines 44-198, then applies verbatim). The
staging repository itself is created in P0 as the PUBLIC repository
`graphty-org/webgpu-graph-algorithms` (the sibling org; the scaffold's
`repository.url` names the wrong org) -- section 12's cost model and the
runner registration of 12.4 presuppose it (Q-28).

```
packages/webgpu-graph-algorithms/
+-- package.json  project.json  tsconfig.json  tsconfig.build.json  tsconfig.strict-consumer.json  eslint.config.js (package-local, extends the root)
+-- vitest.config.ts              # projects: node (primary), node-limits, browser (smoke)   -- section 11; benchmarks are NOT a vitest project
+-- scripts/entries.js  build-bundle.js  bundle-types.js  gpu-report.js  bench-compare.js  gpu-policy.js   # plain .js (package "type": "module"; knip's scripts/**/*.{ts,js} glob)
+-- README.md  CLAUDE.md  LICENSE  docs/HEADLESS_GPU_REPORT.md (moved from the repo root)  docs/research/ (the seven notes and the three drafts, Q-18)
+-- src/
|   +-- index.ts                  # the only public barrel; explicit named exports; /// <reference types="@webgpu/types" />
|   +-- browser/index.ts          # ./browser entry (section 2.3); the only directory allowed to reference navigator
|   +-- node/index.ts             # ./node entry; the only file that IMPORTS the "webgpu" module
|   +-- errors.ts                 # WebGpuGraphError
|   +-- constants.ts              # WORKGROUP_SIZE = 256, MAX_WORKGROUPS_PER_DIM = 65535, ARC_WINDOW_ALIGN = 64, STORAGE_ALIGN = 256, EXACT_MAX_NODES = 16384 (conservative until G3; the 7.8 numbers predict 32,768; re-fixed at P3 with the measurement cited, Q-6); interpolated into the WGSL prelude (3.5), never duplicated as literals
|   +-- types/                    # public option / result / accelerator types (types only); accelerator.ts = the structural mirrors that ARE the published contract (D27)
|   +-- device/                   # acquire.ts (requestAdapter / requestDevice / isSoftwareAdapter), caps.ts, webgpu-constants.ts, error-scope.ts, lost.ts -- acquisition and capability only (D26)
|   +-- context.ts                # GpuContext: the composition root (probe / create / from; owns PipelineCache, BufferPool, Readback, GraphResidency, Profiler)
|   +-- memory/                   # residency.ts, upload-plan.ts (planUpload), buffer-pool.ts, readback.ts, lease.ts
|   +-- kernel/                   # pipeline-cache.ts, kernel.ts, dispatch.ts (plan1d / plan2d / planGridStride / planIndirect), struct-block.ts (UniformBlock + storage mode), wgsl.ts (composeWgsl), batch.ts (CommandBatch), profiler.ts
|   +-- kernels.ts                # THE registry: every WgslModuleSpec with its override axes; consumed by the compile matrix, the bind-group-budget test and PipelineCache.warm()
|   +-- wgsl/                     # *.wgsl.ts template-string modules (prelude.wgsl.ts, reduce.wgsl.ts, scan.wgsl.ts, ..., fa2-*.wgsl.ts, grid-*.wgsl.ts): kernel BODIES only; bindings and overrides are emitted by the composer from the spec (3.5)
|   +-- primitives/               # reduce.ts scan.ts segmented-reduce.ts compact.ts histogram.ts radix-sort.ts frontier.ts advance.ts spmv.ts coo-to-csr.ts grid.ts
|   +-- algorithms/               # spmv/ (pagerank, hits, eigenvector, katz), traversal/ (bfs, sssp, bellman-ford, closeness, betweenness, apsp), components/ (wcc), structure/ (kcore, triangles), community/ (lpa, louvain)
|   +-- layouts/                  # force-simulation.ts (shared core + the ForceModel hook interface, 7.19), forceatlas2.ts, fruchterman-reingold.ts, spring-electrical.ts, repulsion-exact.ts, repulsion-grid.ts, seed.ts (LCG), inputs.ts (role / name / array lookup of nodeMass and weights through the snapshot, D28, 7.14), calibrate.ts (calibrateLayout)
|   +-- accelerator.ts            # createAccelerator(ctx, options?): the object implementing AlgorithmAccelerator & LayoutAccelerator (section 9)
+-- test/
|   +-- setup/gpu.ts              # acquire() per Node project: a FRESH adapter per device; requireGpu(); the GRAPHTY_GPU_REQUIRE policy via scripts/gpu-policy.js (section 11.2)
|   +-- setup/browser.ts          # the browser project's setup: reads the forwarded policy from import.meta.env (2.3)
|   +-- oracle/<name>.ts          # one CPU reference per primitive / algorithm / layout, mirroring src/ (11.3); FA2 / FR oracles are the SPEC of L1
|   +-- helpers/device.ts graphs.ts caps-tables.ts matchers.ts leak-counter.ts frame-loop.ts override-matrix.ts
|   +-- device/ memory/ kernel/ primitives/ algorithms/ layouts/   # node project
|   +-- limits/                   # node-limits project (real large limits and the 262k / 1M layout fixtures; GPU lane only)
|   +-- browser/                  # browser smoke project (*.test.ts)
|   +-- types/*.test-d.ts  conformance.test-d.ts (imports the real CPU packages, D27)  index.test.ts  build-output.test.ts
+-- benchmarks/                   # harness.ts (graph-format's, with an ASYNC run body) datasets.ts run.ts <group>.bench.ts results/<runner-class>.json (checked-in baselines) out/ (run output, gitignored)
```

`package.json` is the note 07 section 4.2 manifest with one addition (the
`./browser` export) and the name `webgpu-graph-algorithms` in
`repository.directory`. `@graphty/graph-format` is in both `dependencies`
(`workspace:^`, 2.5) and `peerDependencies` (`^0.1.0`, bumped to `^1.0.0` at
F2); `@webgpu/types ^0.1.72` is a `dependency` because exported signatures
name `GPUDevice`; `webgpu` is an optional peer (`>=0.4.0 <1.0.0`) and a
devDependency pinned to `0.4.0`; no `browserslist`; not `private`;
`coverage:preview` on port 9058; the `lint` script is `eslint && tsc --noEmit
&& tsc -p tsconfig.strict-consumer.json` so the design-16.6 strict-consumer
compile runs wherever `lint` runs (the monorepo's `nx affected -t lint`
included; graph-format runs it only from `ready:commit`, a gap this package
does not inherit); the scripts gain `bench` (`tsx benchmarks/run.ts`) and
`bench:compare` (`node scripts/bench-compare.js`, which reads the runner class
from `gpu-report.json` and compares `benchmarks/out/<runner-class>.json` with
the checked-in `benchmarks/results/<runner-class>.json`: the 3x-regression
check of 11.7).

`tsconfig.json`: `noUncheckedIndexedAccess: false` (design 14.5 line 4218),
`lib: ["ES2020", "DOM", "DOM.Iterable"]` (the browser entry needs `navigator`
typings; the core and node entries never use them), `types: ["node",
"vitest/globals", "@webgpu/types"]`. `tsconfig.build.json` sets
`stripInternal: true` (graph-format's convention) so members marked
`@internal` (`GpuContext.residency`, 3.3) stay out of the published d.ts.
`tsconfig.strict-consumer.json` compiles `test/types/*.test-d.ts` against
`dist/*.d.ts` with the strict flags ON (design 16.6); it needs the bundle d.ts
shims of `./browser` and `./node`, so it runs after `build:all` (2.5).

The package `CLAUDE.md` (the sibling model is `packages/graph-io/CLAUDE.md`:
Package Structure / Adding a format / House Style / Distribution) carries
these sections: Package Structure (the tree above and the layer rule of 3.2);
WGSL Conventions (3.5, including the uniformity rule and the binding /
override emission); House Style (3.6); Testing (the 11.1 projects, the
`GRAPHTY_GPU_REQUIRE` policy and the 12.2 environment table, the Vitest
per-instance `launch` spelling recorded at P0, the browser-project env
forwarding); Verified Platform Facts (the R-22 answers as they are settled,
with the probe that settled each); Adding an Algorithm / a Kernel / a Layout
Model (the recipe: `types/`, `wgsl/` body, `kernels.ts` entry, driver,
`accelerator.ts` method, `test/oracle/`, differential test, 9.2 dispatcher
method and 9.4 adapter in the CPU packages); Distribution (2.5); and the rule
"`./node` is imported only by Node entry points and tests".

### 3.2 Layers and their one-line responsibilities

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| device | `acquire`, `GpuCaps`, `isSoftwareAdapter`, error scopes, `lost` | acquisition (section 2), capability record, error scopes, uncaptured-error routing, device-loss state. Nothing else: no pipeline cache, no pool, no benchmark (D26). |
| context | `GpuContext` (`src/context.ts`) | the composition root: `probe` / `create` / `from`; constructs and owns `PipelineCache`, `BufferPool`, `Readback`, `GraphResidency`, `Profiler`; `release` / `dispose`. Sits ABOVE memory and kernel. |
| memory | `GraphResidency`, `planUpload`, `BufferPool`, `Readback`, `Lease` | upload cache keyed on typed-array objects + per-snapshot record; arena / per-array / windowed plans (`planUpload` is a pure function); pooled scratch; staging ring readback; `release(snapshot)`; `residentBytes`. |
| kernel | `PipelineCache`, `Kernel`, `plan1d` / `plan2d` / `planGridStride` / `planIndirect`, `UniformBlock` (+ storage mode), `UniformRing`, `composeWgsl`, `CommandBatch`, `Profiler`, `kernels.ts` | compile-once pipelines keyed by module + overrides + features; bind-group layout DERIVED from the module spec's binding list (3.5); 1D / 2D / indirect dispatch math (pure functions of caps); generated 16-byte-safe uniform and storage structs with dynamic offsets; WGSL composition; recording k iterations into one command buffer; optional timestamp profiling; the registry of every module spec. |
| primitives | `reduce`, `scan`, `segmentedReduce`, `compact`, `histogram`, `radixSort`, `Frontier`, `advance`, `spmv`, `cooToCsr`, `grid` | the reusable building blocks (section 6), each with a CPU reference in `test/oracle/<name>.ts`. |
| algorithms | one async function per algorithm | `(ctx, snapshot, ...args, options?) => Promise<Result>`; index-aligned typed arrays; no id mapping (section 8). |
| layouts | `ForceSimulation` (shared) + the `ForceModel` hook interface, `ForceAtlas2Model`, `FruchtermanReingoldModel`, `SpringElectricalModel`, `RepulsionExact`, `RepulsionGrid`, role-based vector lookup (`nodes.byRole` / `gpuView` / `expandEdges`, D28), `calibrateLayout` | `LayoutSimulation` implementations over the owner's stride-3 array (section 7); the crossover micro-benchmark. |
| accelerator | `createAccelerator(ctx, options?)` -> `GpuAccelerator` | the injectable object: implements `AlgorithmAccelerator & LayoutAccelerator` structurally, carries the layout tuning defaults every simulation it creates inherits, plus `release(snapshot)` and `dispose()` (section 9). |

A lower layer never imports a higher one: device < context < memory < kernel
< primitives < algorithms / layouts < accelerator, with `context.ts` the one
file that may import memory and kernel from below the primitives (it
constructs them; memory and kernel receive the device and caps, never the
context). `src/wgsl/**` is imported only by `kernels.ts`; `test/**` may import
any layer. The rule is enforced by `import/no-restricted-paths` zones in the
package `eslint.config.js` (one zone per layer, `from: src/<higher>/`,
`target: src/<lower>/`) and `import/no-cycle`, both enabled from P0 while the
tree is small.

Class list (kept deliberately short; algorithms are plain functions; planners
and the composer are pure functions, not classes):

| Class | One line |
| --- | --- |
| `WebGpuGraphError` | `Error` with stable `code` and frozen `details`; the only error type the package throws for conditions it detects itself (D12: format accessor errors pass through). |
| `GpuContext` | owns one device, its caps, pipeline cache, buffer pool, staging ring and residency; `probe` / `create` / `from` / `release` / `dispose`. `createAccelerator(ctx)` and `calibrateLayout(ctx)` are functions above it, not methods (D26). |
| `GraphResidency` | WeakMap-keyed upload cache for cores, views and columns; per-snapshot record for `release`; `packViews`; `stats()`. Reached as `ctx.residency`, marked `@internal`. |
| `BufferPool` | size-class pool of `GPUBuffer`s by usage; `acquire` / `release` / `trim`; `liveBytes`. |
| `Lease` | scope object that returns every scratch buffer it acquired when an algorithm resolves or rejects. |
| `Readback` | ring of `MAP_READ` staging buffers; `read(src, byteLength, dest?)` copies out before `unmap`; chunked above the staging size; a `CommandBatch` borrows one slot for its lifetime and ALWAYS unmaps and returns it (4.4). |
| `PipelineCache` | `get(spec) -> GPUComputePipeline` with error scopes and compilation-info diagnostics; the explicit layout comes from `spec.bindings`; `warm(specs)`. |
| `Kernel` | a compiled pipeline plus the binding list of its spec; `bind(resources)` creates / caches bind groups keyed by the spec's binding names; `dispatch(pass, plan)`. |
| `UniformBlock` | declares fields with WGSL types; emits the padded struct text AND writes / reads the bytes (D20); `{ layout: "uniform" \| "storage" }` so the FA2 `state` block is generated too (5.3). |
| `UniformRing` | one `UNIFORM` buffer with 256-byte-stride slots for per-iteration params inside a batch. |
| `CommandBatch` | records N dispatches / copies into one command encoder and submits once; borrows the staging slot for the batch's readback and carries its batch id; checks the context's pending-error slot after `submit()` (5.8). |
| `Profiler` | wraps compute passes with `timestampWrites` when the feature was granted; resolves per pass; `quantised` flag (5.5). |
| `Frontier` | vertex / edge queues, length counters, indirect-args buffer with one slot per (level, variant), ping-pong. |
| `ForceSimulation` | the shared layout state machine (buffers, in-flight batches, readback, settle, fixed mask, `setPosition` overrides, trace); consumes a `ForceModel` (7.19) that supplies the per-iteration kernel sequence, its buffers, overrides, per-iteration params and reheat rule. |
| `GpuAccelerator` | thin adapter from the CPU packages' interfaces to the functions above, created by `createAccelerator(ctx, options?)`. |

### 3.3 Public TypeScript API (root entry)

Errors:

```ts
export type WebGpuGraphErrorCode =
    | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_NO_DEVICE" | "E_SOFTWARE_ONLY" | "E_DEVICE_LOST" | "E_DISPOSED"
    | "E_VALIDATION" | "E_SHADER_COMPILE" | "E_OUT_OF_MEMORY" | "E_TOO_LARGE" | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT" | "E_SNAPSHOT" | "E_RELEASED" | "E_NOT_LOADED" | "E_ABORTED";
export class WebGpuGraphError extends Error {
    readonly code: WebGpuGraphErrorCode;
    readonly details: Readonly<Record<string, unknown>>;
    constructor(code: WebGpuGraphErrorCode, message: string, details?: Record<string, unknown>);
}
```

Every code above is thrown by at least one documented path (5.7 lists them);
there is no `E_IN_FLIGHT` because a saturated `step()` coalesces (7.19) rather
than throwing. `GraphFormatError` codes that pass through unchanged
(`E_GPU_INELIGIBLE`, `E_UNKNOWN_NODE`, D12) are listed in 5.7.

Context and capabilities:

```ts
export interface GpuCaps {
    readonly limits: GPUSupportedLimits;            // device.limits
    readonly features: ReadonlySet<string>;         // device.features
    readonly wgslFeatures: ReadonlySet<string>;     // navigator.gpu.wgslLanguageFeatures / dawn equivalent; informational only (never relied on)
    readonly subgroupMinSize: number;               // 0 when the feature is absent
    readonly subgroupMaxSize: number;
    readonly software: boolean;
    readonly runtime: "browser" | "node" | "unknown";
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
}
export declare class GpuContext {                   // src/context.ts (D26)
    static probe(options: ProbeOptions): Promise<ProbeResult>;
    static create(options: GpuContextOptions): Promise<GpuContext>;
    static from(device: GPUDevice, info?: Partial<GpuCaps>): GpuContext;
    readonly device: GPUDevice;
    readonly caps: GpuCaps;
    readonly state: "ready" | "lost" | "disposed";
    readonly lost: Promise<GPUDeviceLostInfo>;
    /** @internal -- the upload cache; stripped from the published d.ts (stripInternal). Tests and the layouts reach it; consumers use release(). */
    readonly residency: GraphResidency;
    readonly profiler: Profiler | null;             // non-null when "timestamp-query" was granted
    release(snapshot: GraphSnapshot): void;
    dispose(): void;
}
export function isSoftwareAdapter(info: GPUAdapterInfo & { isFallbackAdapter?: boolean }): boolean;
export function createAccelerator(ctx: GpuContext, options?: AcceleratorOptions): GpuAccelerator;   // src/accelerator.ts; one per call -- the app creates one and injects it
export function calibrateLayout(ctx: GpuContext, options?: CalibrateOptions): Promise<GpuCalibration>;   // src/layouts/calibrate.ts; section 2.2
export interface AcceleratorOptions {
    readonly layout?: GpuLayoutTuning | undefined;      // defaults inherited by every simulation the accelerator creates (exactMaxNodes from calibrateLayout, nearMax, gridMax2D / 3D, deterministic, compat, repulsion)
    readonly algorithms?: { readonly betweenness?: { readonly k?: number; readonly sources?: readonly number[] } } | undefined;   // defaults for options the CPU option types carry only after A2 (9.2)
}
```

Algorithms (section 8 gives the full list; the signature shape is uniform):

```ts
export interface GpuRunOptions {
    readonly dest?: Float32Array | Uint32Array | undefined;     // design 10.7: preallocated destination; E_INVALID_ARGUMENT when the length does not match
    readonly signal?: AbortSignal | undefined;                  // E_ABORTED between batches
    readonly onProgress?: ((done: number, total: number) => void) | undefined;
}

export function pageRank(ctx: GpuContext, s: GraphSnapshot, options?: PageRankOptions & GpuRunOptions): Promise<GpuPageRankResult>;
export function personalizedPageRank(ctx: GpuContext, s: GraphSnapshot, personalization: F32, options?: PageRankOptions & GpuRunOptions): Promise<GpuPageRankResult>;
export function hits(ctx: GpuContext, s: GraphSnapshot, options?: HitsOptions & GpuRunOptions): Promise<GpuHitsResult>;
export function eigenvectorCentrality(ctx: GpuContext, s: GraphSnapshot, options?: EigenvectorOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function katzCentrality(ctx: GpuContext, s: GraphSnapshot, options?: KatzOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function connectedComponents(ctx: GpuContext, s: GraphSnapshot, options?: ComponentsOptions & GpuRunOptions): Promise<GpuLabelResult>;   // WCC semantics on directed input; renumber: true by default
export function breadthFirstSearch(ctx: GpuContext, s: GraphSnapshot, source: number, options?: BfsOptions & GpuRunOptions): Promise<GpuBfsResult>;
export function sssp(ctx: GpuContext, s: GraphSnapshot, source: number, options?: SsspOptions & GpuRunOptions): Promise<GpuSsspResult>;
export function bellmanFord(ctx: GpuContext, s: GraphSnapshot, source: number, options?: BellmanFordOptions & GpuRunOptions): Promise<GpuBellmanFordResult>;
export function closenessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: ClosenessOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function betweennessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: BetweennessOptions & GpuRunOptions): Promise<GpuBetweennessResult>;
export function edgeBetweennessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: BetweennessOptions & GpuRunOptions): Promise<GpuEdgeScoresResult>;
export function allPairsShortestPath(ctx: GpuContext, s: GraphSnapshot, options?: ApspOptions & GpuRunOptions): Promise<GpuApspResult>;
export function kCoreDecomposition(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuCorenessResult>;
export function triangleCount(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuTriangleResult>;
export function labelPropagation(ctx: GpuContext, s: GraphSnapshot, options?: LabelPropagationOptions & GpuRunOptions): Promise<GpuLabelResult>;
export function minimumSpanningTree(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuMstResult>;
export function louvain(ctx: GpuContext, s: GraphSnapshot, options?: LouvainOptions & GpuRunOptions): Promise<GpuCommunityResult>;
export function degree(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<U32>;   // walking-skeleton kernel; kept as a PUBLIC diagnostic (documented in the README as such: cheaper on the CPU than the upload)
```

Result types (all index-aligned; `F32`, `U32` are the graph-format aliases
`Float32Array<ArrayBuffer>` / `Uint32Array<ArrayBuffer>`, note 07 section 1.2):

```ts
export interface GpuScoresResult      { readonly scores: F32; readonly iterations: number; readonly converged: boolean; readonly precision: "f32"; }   // precision on EVERY score result so the element can label GPU scores (Q-24)
export interface GpuPageRankResult    extends GpuScoresResult { readonly danglingMass: number; }   // iterations = the first iteration whose L1 delta fell below tolerance (8.2), not the batch boundary
export interface GpuHitsResult        { readonly hubs: F32; readonly authorities: F32; readonly iterations: number; readonly converged: boolean; readonly precision: "f32"; }
export interface GpuLabelResult       { readonly labels: U32; readonly count: number; groups(): U32[]; }      // labels dense 0..count-1, first-seen order (renumberPartition); design 14.2 line 3738
export interface GpuBfsResult         { readonly depth: U32; readonly parent: U32; readonly order: U32; readonly visitedCount: number; readonly levels: number; readonly switches: number; }   // INVALID_INDEX = unreached / root; switches = direction changes (a device counter, 8.4)
export interface GpuSsspResult        { readonly dist: F32; readonly predArc: U32; readonly reachedCount: number; }   // +Infinity = unreached; pathTo / pathEdges are attached by the CPU package's dispatcher (9.2), never computed here
export interface GpuBellmanFordResult extends GpuSsspResult { readonly hasNegativeCycle: boolean; }
export interface GpuBetweennessResult extends GpuScoresResult { readonly sourcesUsed: number; readonly sigmaOverflow: boolean; }
export interface GpuEdgeScoresResult  { readonly scores: F32; readonly precision: "f32"; }        // length edgeCount (folded with foldArcs "first")
export interface GpuApspResult        { readonly dist: F32; readonly n: number; }   // n * n row-major
export interface GpuCorenessResult    { readonly coreness: U32; readonly maxCore: number; }
export interface GpuTriangleResult    { readonly perNode: U32; readonly total: number; }
export interface GpuMstResult         { readonly edges: U32; readonly totalWeight: number; }   // logical edge indices
export interface GpuCommunityResult   extends GpuLabelResult { readonly modularity: number; readonly levels: number; }
```

Layouts:

```ts
export interface LayoutStatsBase {
    readonly iteration: number; readonly meanDisplacement: number; readonly rmsRadius: number; readonly layoutRadius: number;   // rmsRadius = sqrt(mean |p - centroid|^2), the settle normaliser (7.17); layoutRadius = max |p - centroid|
    readonly centroid: readonly [number, number, number];
    readonly repulsionTier: "exact" | "grid"; readonly maxCellOccupancy: number | null; readonly outsideGrid: number | null;   // grid tier: the largest finest-cell population (graft: A 7.7 stats) and the number of nodes beyond the grid extent (7.7)
    readonly msPerIteration: number | null;                                               // from the profiler when present, else wall time (labelled)
}
export interface ForceAtlas2Stats extends LayoutStatsBase {
    readonly swing: number; readonly traction: number; readonly speed: number; readonly speedEfficiency: number;
    readonly trace: ReadonlyArray<{ swing: number; traction: number; speed: number; speedEfficiency: number; meanDisplacement: number; settledCount: number }>;   // per iteration of the last completed batch: K4 writes the first four, K1 the last two (7.3)
}
export interface FruchtermanReingoldStats extends LayoutStatsBase { readonly temperature: number; readonly trace: ReadonlyArray<{ temperature: number; meanDisplacement: number; settledCount: number }>; }
export interface SpringElectricalStats  extends LayoutStatsBase { readonly kineticEnergy: number; readonly trace: ReadonlyArray<{ kineticEnergy: number; meanDisplacement: number; settledCount: number }>; }

export interface GpuLayoutSimulation<Options, Stats extends LayoutStatsBase> extends LayoutSimulation {   // LayoutSimulation is the design-14.3 interface, mirrored in src/types (D27)
    load(snapshot: GraphSnapshot, positions: F32): void;
    /** Submits k iterations. Resolves when the batch that carries them has been read back into `positions`. When `maxInFlight` batches are already
     *  in flight the call COALESCES: it queues nothing and returns the promise of the oldest pending batch, so the requested iterations are not run (7.19). */
    step(iterations?: number): Promise<void>;
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
    // GPU-specific additions (not part of LayoutSimulation):
    readonly inFlight: number;                                    // batches submitted but not yet read back
    readonly iterationsDone: number;
    readonly stats: Stats;                                        // last completed batch
    flush(): Promise<void>;                                       // wait for every in-flight batch
    reheat(): void;                                               // reset the settle window and the iteration budget ONLY (D8)
    setParams(patch: Partial<Options>): void;                     // live tuning without reload; dim is rejected; a change to a force LAW resets the speed controller explicitly
    run(options?: { readonly maxIter?: number; readonly batch?: number; readonly signal?: AbortSignal }): Promise<Stats>;   // Node batch driver: step until settled
    inspect?(name: string): Promise<Float32Array | Uint32Array>;     // test builds only (GRAPHTY_GPU_INSPECT=1, 11.9): read back any named buffer after the last submitted kernel
}
export interface GpuLayoutTuning {                                // GPU-only knobs (7.14); settleThreshold / settleWindow / iterationsPerStep / maxInFlight are LAYOUT-OWNED options (9.3) and live in ForceAtlas2Options etc.
    readonly repulsion?: "exact" | "grid" | "auto"; readonly exactMaxNodes?: number; readonly nearMax?: number; readonly deterministic?: boolean;
    readonly gridMax2D?: number; readonly gridMax3D?: number; readonly extentFactor?: number; readonly compat?: "paper" | "networkx";   // 7.2, 7.7; Q-6, Q-21, Q-32
}
export function createForceAtlas2(ctx: GpuContext, options?: ForceAtlas2Options & GpuLayoutTuning): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;
export function createFruchtermanReingold(ctx: GpuContext, options?: FruchtermanReingoldOptions & GpuLayoutTuning): GpuLayoutSimulation<FruchtermanReingoldOptions, FruchtermanReingoldStats>;
export function createSpringElectrical(ctx: GpuContext, options?: SpringElectricalOptions & GpuLayoutTuning): GpuLayoutSimulation<SpringElectricalOptions, SpringElectricalStats>;   // the ngraph-like preset (7.20, P5)
```

The accelerator object (section 9 defines the two interfaces it satisfies):

```ts
export interface GpuAccelerator {
    readonly kind: "webgpu";
    readonly ctx: GpuContext;
    readonly options: Readonly<AcceleratorOptions>;               // the tuning defaults given to createAccelerator; frozen
    // AlgorithmAccelerator members (each delegating to the function above; only implemented ones are present):
    pageRank(s: GraphSnapshot, o?: PageRankOptions): Promise<GpuPageRankResult>;
    connectedComponents(s: GraphSnapshot): Promise<GpuLabelResult>;
    breadthFirstSearch(s: GraphSnapshot, source: number, o?: BfsOptions): Promise<GpuBfsResult>;
    betweennessCentrality(s: GraphSnapshot, o?: BetweennessOptions): Promise<GpuBetweennessResult>;   // o.sources / o.k from the CPU option type once A2 adds them (9.2); options.algorithms.betweenness supplies defaults until then
    // ... one per shipped algorithm; `dest`, `signal` and `onProgress` are reachable ONLY through the package's own functions (GpuRunOptions), never through the accelerator interface
    // LayoutAccelerator members (the CPU option types; GPU tuning comes from `options.layout`, never from the caller):
    forceAtlas2(o?: ForceAtlas2Options): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;            // nodeSize set -> E_UNSUPPORTED at runtime (7.14)
    fruchtermanReingold(o?: FruchtermanReingoldOptions): GpuLayoutSimulation<FruchtermanReingoldOptions, FruchtermanReingoldStats>;
    springElectrical(o?: SpringElectricalOptions): GpuLayoutSimulation<SpringElectricalOptions, SpringElectricalStats>;
    // lifecycle:
    release(s: GraphSnapshot): void;
    dispose(): void;
}
```

Option types re-declare the CPU packages' option shapes STRUCTURALLY in
`src/types/options.ts` (same field names and defaults as
`layout/src/layouts/force-directed/forceatlas2.ts` lines 26-42 and the
`indexed.*` options of design 14.2), and `src/types/accelerator.ts` mirrors
`AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation`. The mirrors
are the published contract for the life of the package (D27): from W1
`test/types/conformance.test-d.ts` imports the real `@graphty/algorithms` /
`@graphty/layout` types (devDependencies, types only, section 9.8) and asserts
mutual assignability with `expectTypeOf` in both directions; `src/` never
imports them, so the published d.ts never references a package that is
neither a dependency nor a peer (Q-26 records the alternative). Conventions:
options use `?: T | undefined`; absent output is `null`; `dest` arrays must be
`<ArrayBuffer>`-backed and of exact length; node arguments are INDICES (ids are
resolved by the caller through `snapshot.ids.requireIndex`, note 07 section
1.8).

### 3.4 Browser and Node entries

```ts
// ./browser
export interface BrowserGpuOptions extends Omit<GpuContextOptions, "gpu" | "device"> {}   // rejectSoftware and adapter come from GpuContextOptions (2.2)
export function probeBrowserWebGpu(options?: BrowserGpuOptions): Promise<ProbeResult>;   // never requests a device; the returned adapter is unused
export function requestGpuContext(options?: BrowserGpuOptions): Promise<GpuContext>;     // pass { adapter: probe.adapter } to reuse the probed adapter; create() honours rejectSoftware (E_SOFTWARE_ONLY)

// ./node
export interface NodeGpuOptions extends Omit<GpuContextOptions, "gpu" | "adapter" | "device"> {
    readonly adapter?: string | undefined;        // Dawn "adapter=<substring>"
    readonly backend?: "vulkan" | "d3d12" | "d3d11" | "metal" | "opengl" | "opengles" | "null" | undefined;
    readonly dawnFeatures?: readonly string[] | undefined;   // "enable-dawn-features=a,b"
    readonly software?: boolean | undefined;      // adapter=llvmpipe (Linux / Mesa)
    readonly installGlobals?: boolean | undefined;   // default true
}
export interface NodeGpuHandle { readonly gpu: GPU; dispose(): void; }
export function createNodeGpu(options?: NodeGpuOptions): Promise<NodeGpuHandle>;
export function createNodeGpuContext(options?: NodeGpuOptions): Promise<GpuContext>;
export function probeNodeWebGpu(options?: NodeGpuOptions): Promise<ProbeResult>;
```

### 3.5 WGSL source organisation and composition

Decision D9: WGSL is TypeScript. Each kernel is `src/wgsl/<name>.wgsl.ts`
exporting `const <name>Wgsl = /* wgsl */ \`...\`` (the `/* wgsl */` tag gives
editor highlighting). Reasons, verified against the toolchain in `packages/`
(note 07 section 4.6): the pre-push hook builds with `tsc -p
tsconfig.build.json` only, which neither copies `.wgsl` files nor understands
`?raw`; knip's project globs are `src/**/*.ts`; eslint lints `.ts` only. If
`.wgsl` files are ever wanted, a generator script emits the `.wgsl.ts` files
with the owner's auto-generated header.

Composition is string concatenation in `src/kernel/wgsl.ts`, driven by ONE
declaration per module from which everything the host and the shader must
agree on is generated:

```ts
export interface BindingDecl { readonly group: 0 | 1 | 2 | 3; readonly binding: number; readonly name: string; readonly kind: "storage" | "storage-ro" | "uniform"; readonly wgslType: string; }
export interface OverrideDecl { readonly name: string; readonly type: "u32" | "bool" | "f32"; readonly default: number | boolean; }
export interface WgslModuleSpec {
    readonly id: string;                                  // "fa2-attraction"
    readonly body: string;                                // the kernel FUNCTION source only: no @group / @binding lines, no `override` lines
    readonly bindings: readonly BindingDecl[];            // the composer emits the `@group(g) @binding(b) var<storage, read> name: T;` block from this list; Kernel derives the explicit GPUBindGroupLayout and the bind() record keys from the same list
    readonly overrideDecls: readonly OverrideDecl[];      // the composer emits `override NAME: T = default;` from this list; spec.overrides may set only these names (an unknown or misspelled key is E_SHADER_COMPILE at compose time, not a GPUPipelineError at pipeline creation)
    readonly overrides: Readonly<Record<string, number | boolean>>;   // the values for this variant: WG, USE_PERM, HAS_WEIGHTS, LINLOG, ...
    readonly needs: readonly ("subgroups")[];             // splice `enable subgroups;` + subgroup helpers only when the device has it
    readonly uniforms: readonly UniformBlock[];           // generated struct text spliced in (5.3)
    readonly snippets?: Readonly<Record<string, string>>; // operator bodies substituted at `//@@NAME@@` markers (Gunrock-style advance / filter functors); a snippet may reference only the parameters named in the primitive's functor signature (6 rows 3 and 8)
}
```

Why one declaration: a storage binding whose access mode or index disagrees
between the WGSL text and the layout, or an override name that the composed
WGSL does not declare, is a `GPUPipelineError` raised only when THAT variant is
created (`tmp/webgpu-plan/review/probes/binding-mismatch-probe.mjs`,
`maint-sync-probe.mjs`, on NVIDIA and llvmpipe); generating both sides from
`bindings` / `overrideDecls` makes the disagreement impossible, exactly as D20
does for uniform structs. A unit test asserts that no `.wgsl.ts` body contains
`@group(` or `override `. `composeWgsl(spec, caps)` throws `E_SHADER_COMPILE {
id, stage: "compose", slot }` when a `//@@` marker remains after substitution
or a snippet key has no marker -- an unfilled marker is otherwise a WGSL line
comment that compiles cleanly and silently does nothing. The compilation-info
formatter subtracts the prelude line count so messages point into the body.
`src/kernels.ts` lists every `WgslModuleSpec` with the override axes the
package uses; the compile matrix (5.1, 11.3), the bind-group-budget test and
`PipelineCache.warm()` iterate that list, so a variant cannot be added without
being enumerated.

The prelude every module receives is itself composed: the constants are
interpolated from `constants.ts` and graph-format (`INVALID_INDEX`), never
retyped as literals, and a unit test greps every `.wgsl.ts` for `65535u`,
`256u` and `0xFFFFFFFFu`:

```wgsl
// prelude.wgsl.ts (excerpt; ${...} are TypeScript interpolations)
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
override WG: u32 = ${WORKGROUP_SIZE}u;
override USE_PERM: bool = false;      // design 10.1: select(a, arcToEdge[a], USE_PERM)
override HAS_WEIGHTS: bool = false;   // weights === null -> 1.0; the weights slot is bound to colIdx and never read
override SUBGROUP_MAX: u32 = 0u;      // 0 = no subgroups; adapter.info.subgroupMaxSize otherwise: sizes workgroup-memory scratch ONLY (D16)
fn linear_id(wid: vec3<u32>, lid: u32) -> u32 { return (wid.x + wid.y * ${MAX_WORKGROUPS_PER_DIM}u) * WG + lid; }   // 2D grid linearisation (5.2)
fn lowbias32(x: u32) -> u32 { ... }   // integer hash (cosmos: sin() hashes diverge across vendors, note 03 section 1.3)
fn mask_bit(w: u32, i: u32) -> bool { return ((w >> (i & 31u)) & 1u) == 1u; }
fn unpack_u8(w: u32, i: u32) -> u32 { return (w >> (8u * (i & 3u))) & 0xFFu; }
```

Pipeline identity is `(id, overrides, needs present on device, snippets)`; the
cache key is the JSON of that tuple (section 5.1). Operator snippets are part
of the key (the same advance kernel with a BFS functor and an SSSP functor are
two pipelines).

Two WGSL rules every kernel body obeys, both enforced by the compile matrix
because Tint makes a violation a shader-creation ERROR:

1. Uniformity. `workgroupBarrier`, `subgroupAdd` and every other
   synchronisation or subgroup builtin must be reached in UNIFORM control
   flow: per-invocation values are computed under the `i < n` guard into
   locals (`var sw = 0.0; if (valid) { ... }`), and every workgroup / subgroup
   reduction runs in unconditional code after the guard; an early `return`
   before a barrier is legal only on a condition that is a function of
   `workgroup_id` and uniforms (the workgroup-per-row tier keys its exit on
   `workgroup_id`). Verified: guarded barriers and `subgroupAdd` under a
   `global_invocation_id` condition fail on Dawn ("must only be called from
   uniform control flow", `tmp/webgpu-plan/review/probes/
   uniformity-probe-nvidia.log` cases A / C / E); the unconditional forms
   compile (B / D / F). The 7.6 and 7.11 sketches are written this way.
2. Operator precedence. WGSL refuses to mix `*` and `^` (or `&`, `|`) without
   parentheses ("mixing '*' and '^' requires parenthesis",
   `verify-wgsl-mixing.mjs`); every hash expression in this plan is written
   fully parenthesised.

Subgroup kernels (D16) read `subgroup_size` and `subgroup_invocation_id` at
runtime; a subgroup obtains its index within the workgroup by having its
elected lane (`subgroupElect()`) do one `atomicAdd` on a `var<workgroup>
atomic<u32>` counter and `subgroupBroadcast` the result, because
`@builtin(subgroup_id)` needs the `subgroup_id` language feature Chromium 139
does not expose and WGSL defines no relationship between subgroup ids and
`local_invocation_index`. Scratch arrays are sized `WG / SUBGROUP_MAX`
(rounded up), which is enough for the smallest size the compiler may pick.

Bind-group conventions (core defaults: 8 storage buffers per stage, 4 bind
groups; note 05 section 4): group 0 = the graph (immutable per snapshot:
`rowPtr`, `colIdx`, `weights` or dummy, `perm` or dummy -- FOUR slots whether
or not the permutation is the identity, because one explicit layout serves
every dummy variant), group 1 = algorithm state (ping-pong vectors, queues,
partials), group 2 = the params uniform (dynamic offset into the
`UniformRing`), group 3 = optional / cold arrays (`arcToEdge`, `edgeToArc`,
columns). A kernel that needs more than 8 storage buffers in one stage is
SPLIT, never given a raised limit as a requirement; section 7.4 tabulates the
budget of every layout kernel, 8.10 the budget of every algorithm kernel; the
PageRank pull kernel of 8.2 is the first kernel that uses exactly 8 (revRowPtr,
revColIdx, revWeights | dummy, perm | dummy, xNorm, rankOut, personalization |
dummy, partials); G7 of 7.7 and the five 8.10 kernels marked (8) are the
others. A test inspects every bind-group-layout descriptor the
package creates (enumerated through `kernels.ts`) and fails above 8 storage
entries per stage.

### 3.6 Naming and conventions carried from the sibling packages

JSDoc on every export, explicit return types, `.js` suffixes on relative
imports, no default exports, no `console.log` in `src/` (`console.warn` only for
the residency warning of 4.1), plain ASCII, prettier 4 / 120 / all, knip clean,
no `eslint-disable`, no `ts-expect-error` outside negative type tests (note 07
section 7). Results are `Uint32Array<ArrayBuffer>` / `Float32Array<ArrayBuffer>`;
`INVALID_INDEX` is the only sentinel; never a bitwise operator on an arc index
or byte offset (I3); never a write into a view; never a zero-length binding.

### Review notes (section 3)

- Judge integration-feasibility (draft A): the public API and the accelerator
  interface disagreed on the closeness / betweenness result shapes. Here every
  score-producing function returns a `GpuScoresResult` (or a subtype) and the
  accelerator's `ScoresResultLike` (9.2) is its supertype.
- Judge integration-feasibility (draft C): `nodeSize?: never` narrows the GPU
  option type below the CPU `ForceAtlas2Options` that `LayoutAccelerator.
  forceAtlas2` must accept, so the element's zod-parsed options would not
  type-check. NOT grafted as a type; the deferred option is rejected at runtime
  with `E_UNSUPPORTED` and named in the README (7.14).
- `precision: "f32"` on every score-carrying result (graft: A 14 item 8,
  widened from `GpuBetweennessResult` alone so 9.4 item 3 and Q-24 hold for
  every result) lets the element label GPU scores; `sigmaOverflow` is the
  saturation flag of 8.4.

---------------------------------------------------------------------------

## 4. Memory and upload

### 4.1 GraphResidency: what is cached and under which key

```ts
/** @internal (3.3): reached as ctx.residency by the layouts and the tests; consumers see only ctx.release(). */
export declare class GraphResidency {
    core(s: GraphSnapshot, need?: readonly CoreArrayName[]): CoreBinding;          // rowPtr, colIdx, weights (+ arcToEdge, edgeToArc on demand)
    view(s: GraphSnapshot, name: Extract<ViewName, "reverse" | "coo" | "edgeList" | "outDegree" | "inDegree" | "degreeOrder" | "reverseDegreeOrder" | "mate">, options?: { packViews?: boolean }): ViewBinding;   // ViewName is graph-format's export (src/types/snapshot.ts lines 226-243)
    column(table: AttributeTable, name: string): ColumnBinding;                   // gpuView(name) + column.version
    array(key: TypedArrayData, label: string, owner?: GraphSnapshot): ArrayBinding;   // any format array (expanded weights, a mask), keyed on the object
    release(s: GraphSnapshot): void;                                             // destroys every buffer recorded for s
    stats(): ResidencyStats;                                                     // { buffers, bytes, snapshots, perSnapshot: { serial, label, bytes }[] }
    readonly residentBytes: number;                                              // graft: C R-18
}
export interface Binding { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number; readonly window: ArcWindow | null; }
```

Keys (design 14.5 lines 4231-4234; note 07 sections 1.7, 2.3):

| Thing | Key | Invalidation |
| --- | --- | --- |
| core arena | `snapshot.rowPtr` (the array object; two snapshots sharing a core via `withColumns()` share `rowPtr` AND `serial`) | `release(snapshot)`; the record is also indexed by `snapshot.serial` so `withColumns()` siblings find it (DEPARTURE-4). Siblings are ONE residency unit: `release(s)` destroys the core for `s` and every sibling, and a sibling still in use throws `E_RELEASED` on its next bind (Q-27) |
| per-array core upload | each of `rowPtr`, `colIdx`, `weights`, `arcToEdge`, `edgeToArc` array objects | `release(snapshot)` |
| view array | the view array object (`reverse().colIdx`, `coo().src`, `edgeList().src`, `degreeOrder().perm`, ...) | `release(snapshot)`; `dropCaches()` changes the object (note 07 section 2.3) -- the OLD buffer stays recorded in the per-snapshot record and is freed by `release`, so no leak and no stale read |
| column | the `gpuView(name)` array object PLUS `column.version` | a version bump (`markDirty()` / `setAll()`) re-uploads into the same buffer when the byte length is unchanged (a `writeBuffer`, no realloc); `release(snapshot)` frees it |
| ad hoc array (a per-arc weight vector the simulation expands at `load()` with graph-format's `expandEdges` from a named edge column, a resolved `nodeMass` vector, a `NodeMask`) | the array object; for an expanded column additionally `(gpuView array, column.version)` so a re-expansion is skipped while the column is unchanged | `release(owner)` when registered against a snapshot, else `binding.destroy()` by the caller |

Two `WeakMap`s: `WeakMap<TypedArrayData, ResidentBuffer>` (fast lookup) and
`WeakMap<GraphSnapshot, ResidencyRecord>` (enumeration for `release`), plus a
strong `Map<number, ResidencyRecord>` keyed by `serial` so `withColumns()`
siblings share one record; the strong map entry is cleared by `release`. A
`ResidentBuffer` records `{ buffer, byteLength, serial, kind }`. Because the
serial IS the core's identity (`packages/graph-format/src/snapshot/
graph-snapshot.ts` lines 314 and 923-942: siblings share "the core, the id map
and the serial"), the residency cannot tell siblings apart and does not try:
`release(s)` releases the record of `s.serial`, i.e. the core, every view and
every column buffer of every sibling, and tombstones the serial; a sibling
still in use gets `E_RELEASED` on its next bind. The design-14.4 lifecycle
never releases a sibling of a live snapshot (results attach through
`nodes.set()`, `undirected(s)` on an undirected snapshot returns `this`), so
the element never meets this rule; a Node script that wants per-sibling
lifetimes keeps distinct snapshots (`Q-27`). `release` is idempotent and safe
on a snapshot that was never uploaded. Because the strong map pins the JS record of a snapshot that is
never released (the intended GPU-memory semantics: nothing is freed by GC),
the residency exposes `residentBytes` and `stats()`, and `console.warn`s ONCE
when the number of resident snapshots exceeds `warnUnreleasedSnapshots`
(default 2; the element holds at most one superseded snapshot per design 14.4
line 4126, so 3 resident snapshots means a missing `release`).

Identity permutations (note 07 section 2.2): `core()` never touches
`snapshot.arcToEdge`, `edgeToArc`, `coo().arcToEdge`, `edgeList().arc` or
`reverse().fwdArc` when `flags.arcToEdgeIsIdentity` (or `!directed` for
`fwdArc`); it returns `{ perm: null }` and the kernel is compiled with
`USE_PERM = false`, binding `colIdx` in the `arcToEdge` slot and `rowPtr` in
the `edgeToArc` slot (design 10.1). A test asserts that after `core()`,
`snapshot.byteLength({ views: true })` is unchanged (no materialisation).

### 4.2 Upload planner: arena hot prefix, per array, windowed

`planUpload(s, caps, need)` is a pure function (unit-tested with faked caps)
returning one of three plans, evaluated in this order (design 10.3 lines
2425-2438 with DEPARTURE-2; note 07 section 3):

| Plan | Condition | Buffers | Bindings |
| --- | --- | --- | --- |
| `arena` | `s.arena !== null` AND `bytes <= limits.maxBufferSize` where `bytes = need includes a cold segment ? arena.byteLength : arena.hotByteLength` AND every non-null needed `segment.byteLength <= limits.maxStorageBufferBindingSize` | ONE `GPUBuffer` of `bytes`; ONE `writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, bytes))` | `{ buffer: gbuf, offset: seg.byteOffset - arena.byteOffset, size: seg.byteLength }` per segment (offsets are 256-multiples relative to `arena.byteOffset`, satisfying any `minStorageBufferOffsetAlignment <= 256`) |
| `perArray` | `arena === null` (every `fromCsr` on separate arrays, every `transpose()`, note 07 section 2.5) OR the arena bytes exceed `maxBufferSize` but each needed array fits its binding limit | one buffer per array, `writeBuffer(bufX, 0, s.colIdx)` etc. (no cast: `Uint32Array<ArrayBuffer>` is a `BufferSource`) | whole-buffer bindings |
| `windowed` | some needed array's byte length exceeds `maxStorageBufferBindingSize` (33,554,432 arcs per binding at the 128 MiB default; 536,870,911 at the 4070's 2 GiB under Dawn) | per-array buffers as above (a buffer may exceed the BINDING limit while staying under `maxBufferSize`; if it exceeds `maxBufferSize` too, the array is split across buffers at the same window boundaries) | a list of `ArcWindow { start, end, rowFirst, rowLast, bufferIndex, offset }` where `start = rowPtr[v0] - (rowPtr[v0] % 64)` (256-byte aligned; `%`, never `& ~63`, design 10.6); kernels get `start` as a rebase uniform and iterate `[max(rowPtr[u], start), min(rowPtr[u+1], end))`; a row longer than one window is split across windows with the row's contribution accumulated across dispatches |

Cold segments (`arcToEdge`, `edgeToArc`) after an arena upload sized to the hot
prefix: when a later kernel needs one, the residency uploads it as its OWN
buffer sourced zero-copy from the arena bytes (`new Uint8Array(arena.buffer,
seg.byteOffset, seg.byteLength)`), keyed on the corresponding array object;
when the arena buffer was created at full `byteLength` (the caller's first
`core()` named a cold segment and it fit), the cold segment is a second
`writeBuffer` into `[hotByteLength, byteLength)` of the same buffer (graft: C
4.1, conditioned). Worked numbers at the tiers (design 15.1 arithmetic):

| Tier (undirected weighted) | `hotByteLength` | `byteLength` (with `arcToEdge` + `edgeToArc`) | Path at default limits (256 MiB buffer, 128 MiB binding) | Path on the 4070 (raised) |
| --- | --- | --- | --- | --- |
| 100k / 1M | 16,400,128 B | 28,400,128 B | arena (either size fits) | arena |
| 1M / 10M | 164,000,256 B | 284,000,256 B | arena for the hot prefix (segments of 80,000,000 B fit the binding limit); cold segments per-array (the full arena exceeds 268,435,456 B) | arena at full size |
| 10M / 100M | 1,640,000,256 B | 2,840,000,256 B | `colIdx` = 800,000,000 B exceeds both limits -> windowed (25 windows of 33,554,432 arcs) | perArray (800 MB < 2 GiB binding); the format freezes with `arena: false` at this tier anyway (design 15.3) |

Which algorithms support `windowed` in v1 is explicit (section 4.6): the
per-row gather family (attraction, SpMV, degree, segmented reduce) does; the
frontier family and sort-based algorithms throw `E_TOO_LARGE` with `{ path:
"windowed", algorithm }` until they are extended. Raised limits are requested
from the adapter first (section 2.2), so on the 4070 SUPER the windowed path is
reached only above ~500M arcs.

Upload cost model [X]: `writeBuffer` copies synchronously from the V8 backing
store (note 05 section 2.3); PCIe 4.0 x16 staging typically lands at 5-12 GB/s
host-to-device: the 100k / 1M hot prefix (16.4 MB) ~2-4 ms; 1M / 10M (164 MB)
~20-40 ms; 10M / 100M (1.64 GB, windowed) ~200-400 ms. The walking skeleton
(section 13, P1) measures these (T-1 in 10.4).

### 4.3 View and column uploads

Views upload through the same planner in `perArray` mode (they are never in the
arena, design 10.3). `reverse()` on an undirected snapshot returns the forward
array objects (note 07 section 1.6), so `residency.view(s, "reverse")` resolves
to the SAME buffers as `core()` with no upload -- the WeakMap key is the array
object. `degreeOrder().segmentOffsets` is read on the CPU (5 numbers) and
passed as uniform scalars (design 10.1 line 2342). `weightedOutDegree()` is
never uploaded (F64); the normaliser is computed on the device (section 6 row
3). `packViews` (graft: A 4.3): `view(s, "reverse", { packViews: true })`
concatenates the four reverse arrays of a DIRECTED snapshot into one buffer with
256-aligned offsets (the arena layout applied to a view) when the total fits
`maxBufferSize` and each array fits the binding limit; it saves three buffer
objects and bind-group churn for pull kernels. `ReverseView.arcToEdge` on a
directed identity snapshot aliases `fwdArc` (note 07 section 2 item 8), so it
binds `fwdArc` with no second upload.

Columns: `column(table, name)` calls `table.gpuView(name)` (throws
`E_GPU_INELIGIBLE` from the format for string / list / json -- a
`GraphFormatError` the GPU package lets propagate, one of the pass-through
codes D12 and 5.7 name) and records `column.version`. `u8` columns bind
`paddedU32View()` and kernels bound-check `i < rows * components`; `bool`
columns and masks bind their `data` words. A mutable column whose `version`
changed since upload is re-uploaded on the next `column()` call (same buffer
when the byte length matches). The position column is NOT uploaded through
this path: `LayoutSimulation.load()` takes the owner's array directly (section
7.3). A `u32` column with `refersTo` is bound-checked in the kernel that
dereferences it (the format does not range-check `refersTo` values, note 07
section 2 item 6).

### 4.4 BufferPool, Lease and Readback

`BufferPool.acquire(byteLength, usage, label)` rounds up to a size class
(powers of two from 4 KiB to 64 MiB, then 16 MiB steps; classes above
`maxBufferSize` are never created) and keeps at most `maxIdlePerClass = 4` idle
buffers per (class, usage). `release(buf)` returns it; `trim()` destroys idle
buffers (called by `ctx.release` and by layouts on `dispose`); `liveBytes` is
asserted 0 after `dispose()` in tests. Large allocations run inside
`pushErrorScope("out-of-memory")`; an OOM error becomes `E_OUT_OF_MEMORY {
requested, resident }` (never a silent smaller buffer). The 1 TiB
`maxBufferSize` Dawn-node reports on the adapter is not physical memory (note
05 section 4); the scope, not the limit, is what catches exhaustion.

`Lease` is the scope object algorithms use: `const lease = pool.lease(); try {
const a = lease.storage(n * 4, "sigma"); ... } finally { lease.release(); }` --
every buffer acquired through the lease is released when the algorithm resolves
or rejects (a `try/finally`, not `Symbol.dispose`, until the monorepo's TS
target supports `using`). Persistent buffers (layout state) are owned by the
simulation object and freed in `dispose()`.

`Readback` owns a ring of `MAP_READ | COPY_DST` staging buffers (default 3; the
browser needs more than one because `mapAsync` on a buffer in use by a queued
copy is a validation error, note 05 section 7.2). `read(src, byteLength,
dest?)`: pick a free staging buffer (`mapState === "unmapped"`) or grow the
ring, `copyBufferToBuffer` inside the batch encoder, `submit`, `await
mapAsync(READ)`, `dest.set(new Float32Array(getMappedRange()))` (copy BEFORE
`unmap`, design 10.7), `unmap()`. Ownership is one-directional: `Readback`
owns the ring; a `CommandBatch` BORROWS one slot for its lifetime and always
unmaps and returns it, whether its readback resolved, was discarded as stale
(7.19 item 6), or was abandoned by an abort (5.7) -- a discarded batch still
awaits its `mapAsync` before returning the slot, so the ring never grows
because of a slot left pending. Requests above the staging size are split
into chunks. A `readU32(src, offset)` helper reads a single counter (BFS frontier
length, convergence flag) through the same ring. Never `onSubmittedWorkDone` as
a poll (an extra ~0.1 ms promise in Chromium, note 05 section 7.2). Measured
latencies to design around [M] (note 05 section 7.2): 4-byte round trip 0.04 ms
Dawn-node NVIDIA / 0.10 ms Chromium NVIDIA / 0.15 ms SwiftShader; 1 MiB copy +
map + slice 2.65 ms in Chromium.

### 4.5 Release lifecycle

```
DataManager.getSnapshot()  --freeze-->  snapshot-replaced { previous, next, report }        (previous is null on the first freeze: the listener guards it)
    graphty-element listener: for each of  previous,
                                           dm.undirected(previous).snapshot   when it is not previous (a directed source: a DISTINCT snapshot with its own serial and core arrays),
                                           the visible(previous) cache's induced and undirected snapshots:
                                  accelerator.release(s)  -> GraphResidency.release(s): destroy core, views, columns, ad hoc arrays recorded for s
    LayoutSimulation engines: engine.reload(undirected(next), report, positions)  -> simulation.load(next, positions) re-uploads next's core; the simulation's OWN scratch (forces, partials, grid) is resized in place when nodeCount changed
    Graph.dispose(): the same list for the CURRENT snapshot
    accelerator.dispose()  -> ctx.dispose(): everything, then device.destroy() if owned
```

Why the derived snapshots are listed: under `data.directed: "auto"` a
record-pushed graph is directed (design 4052-4057), every layout and every
undirected-group adapter uploads `dm.undirected(s).snapshot` (design
4103-4107, 4159-4162, 4189-4197), and that snapshot shares nothing with `s`
(`tmp/webgpu-plan/review/probes/serial-sharing.mjs`: `undirectedIsSameObject:
false, undirectedSerial: 2, undirectedSharesRowPtr: false`). Nothing in the
GPU package can find a derived snapshot from its source, so releasing only
`previous` would leak the undirected copy of every superseded snapshot and
fire the 4.1 warning after three freezes in normal use; an 11.3 test asserts
`residency.stats().snapshots === 1` after freeze + layout + `snapshot-replaced`
on a directed graph.

Rules: `release` is called by the OWNER of the snapshot lifecycle (the element,
a Node script), never by an algorithm; an algorithm that receives a snapshot
with no residency uploads it and leaves it resident (the common case: the next
algorithm on the same snapshot pays nothing); a `LayoutSimulation.load(next)`
on a different snapshot releases nothing by itself (the element releases
`previous`), but the simulation drops its references to the previous core
bindings so a later `release(previous)` finds no live user. A kernel asked to
bind a snapshot whose residency was released throws `E_RELEASED` (the record
is tombstoned by serial; graft: C 5.6); a live simulation whose snapshot is
released rejects its next `step()` with `E_RELEASED`.

### 4.6 Chunking and dispatch limits per family

| Family | > 16,776,960 items per dispatch (2D grid) | Windowed bindings (arc ranges) | Notes |
| --- | --- | --- | --- |
| per-node map / reduce / integrate | yes (n > 16.7M nodes) | n/a | trivially chunked |
| per-row gather (attraction, SpMV, segmented reduce, degree) | yes | yes (v1) | window loop on the host: one dispatch per window, accumulating into the same output |
| per-arc map (coo / edgeList kernels: CC hook, Bellman-Ford relax) | yes (A > 16.7M arcs -- the 1M / 10M tier undirected has 20M arcs, so this is the NORMAL case at the desktop tier) | yes (v1) | the arc range of a window is the dispatch range |
| frontier advance / compaction | yes (edge frontier > 16.7M) | no (v1: `E_TOO_LARGE`) | needs the whole `colIdx` bound; window-aware advance lands with P8 (section 13) |
| radix sort / scan | yes | n/a (scratch is the package's own) | |
| grid pyramid | yes | n/a | fixed-size grid buffers |

### 4.7 Bytes per node and per edge on the device

Core (undirected, doubled arcs, weights present unless noted; design 15.1;
E = 10n at every tier):

| Component | Bytes | 100k / 1M | 1M / 10M | 10M / 100M |
| --- | --- | --- | --- | --- |
| `rowPtr` | 4(n + 1) | 0.4 MB | 4 MB | 40 MB |
| `colIdx` | 4A = 8E | 8 MB | 80 MB | 800 MB |
| `weights` | 4A (0 when null) | 8 MB | 80 MB | 800 MB |
| hot prefix total | | 16.4 MB | 164 MB | 1.64 GB (windowed at defaults; per-array on raised limits; lavapipe cannot raise its 128 MiB binding, so this tier is GPU-lane only) |
| `arcToEdge` (cold; only for edge-column gathers) | 4A | 8 MB | 80 MB | 800 MB |
| `edgeToArc` (cold; per-edge writeback) | 4E | 4 MB | 40 MB | 400 MB |
| `coo().src` | 4A | 8 MB | 80 MB | 800 MB |
| `edgeList().src/.dst` | 8E | 8 MB | 80 MB | 800 MB |
| `degreeOrder().perm` | 4n | 0.4 MB | 4 MB | 40 MB |
| `reverse()` (directed only) | 4(n+1) + 8A (+4A weights) | 12.4 MB | 124 MB | 1.24 GB |

Per-algorithm scratch (per node unless stated): PageRank 2 x 4 (ping-pong) + 4
(out-weight sum) + partials = ~12 B/node; BFS 8 (depth, parent) + 2 x 4
(queues) + 1/8 (bitset) = ~16 B/node + optional 4A edge queue; CC 4 B/node +
readback; betweenness with batch k sources: (4 sigma + 4 depth + 4 delta) x k
per node; FA2 exact ~53 B/node, FA2 grid ~81 B/node at the peak of a batch
(65 B/node resident between batches: the 16 B/node radix-sort scratch is
pool-leased per batch; one accounting, used by 7.3 and 10.1) plus a fixed
pyramid (5.6 MB at 512^2 in 2D, ~38 MB at 128^3 in 3D) (section 7.3, note 03
section 8.2). Memory is never the binding constraint below 10M nodes on a 12 GB card;
time is (section 10).

### Review notes (section 4)

- Judge integration-feasibility (draft B): "a consumer that never calls
  release keeps its record forever ... worth a residentBytes warning as draft C
  proposes". Added (4.1, `warnUnreleasedSnapshots`).
- Judges performance-realism and verifiability (drafts B and C) on the 1M / 10M
  arena versus the limits: the judges' arithmetic is right for the HOT PREFIX
  (164,000,256 B < 268,435,456 B and 80,000,000 B segments < 134,217,728 B;
  the trailing 256 is `rowPtr`'s 4,000,004 B padded to the next 256-multiple,
  `tmp/webgpu-plan/review/probes/arena-bytes.mjs`), and wrong for draft C's
  "fails the buffer limit" claim; but the FULL arena with both cold segments
  is 284,000,256 B, which does exceed the default
  `maxBufferSize`. DEPARTURE-2 sizes the buffer to the hot prefix so the tier
  stays on the one-buffer path; cold segments at that tier are per-array at
  default limits.
- Judge verifiability (draft A): `planCoreUpload` tested only the binding limit
  in its per-array branch, never `maxBufferSize`. The table above tests both
  and splits an array across buffers when it exceeds `maxBufferSize`.

---------------------------------------------------------------------------

## 5. Kernel infrastructure

### 5.1 PipelineCache and Kernel

```ts
export declare class PipelineCache {
    get(spec: WgslModuleSpec): Promise<GPUComputePipeline>;      // createComputePipelineAsync inside pushErrorScope("validation"); the explicit layout is derived from spec.bindings (3.5)
    warm(specs: readonly WgslModuleSpec[]): Promise<void>;        // called by load() so the first step() does not compile; kernels.ts supplies the list
    readonly size: number;
}
```

Key = `spec.id + "|" + stableJson(spec.overrides) + "|" + spec.needs.filter(f
=> caps.features.has(f)).join(",") + "|" + hash(spec.snippets)`. Every
distinct `override` set is a distinct pipeline (WGSL 7.2.2, note 05 section 6
item 3), so overrides are limited to things that genuinely change the code
(`USE_PERM`, `HAS_WEIGHTS`, `LINLOG`, `STRONG_GRAVITY`, `DISTRIBUTED`,
`SWING_MODE`, `GRAVITY_CENTER`, `TIER`, `SUBGROUP_MAX`, `WG`,
`LEVELS`, `LAW`, `FR_APPLY`); everything numeric that varies per iteration or
per load -- including `dim`, which 7.13 makes a uniform -- is a uniform.
Pipelines are created with EXPLICIT `GPUBindGroupLayout`s (never `layout:
"auto"`) derived from `spec.bindings` so one layout serves the `USE_PERM` /
`HAS_WEIGHTS` dummy-binding variants and bind groups can be reused across
variants; a kernel whose variants disagree on read-only versus read-write for
one binding declares that binding `read_write` in every variant (the FA2
`state` block is `read_write` everywhere; judge finding on draft A 7.5 / 5.3;
the probe of 3.5 shows even a "looser" layout is rejected, so this is the only
workable sharing rule). Compilation failure: `getCompilationInfo()` messages
are formatted with line numbers against the kernel BODY (prelude lines
subtracted) and thrown as `E_SHADER_COMPILE { id, messages }`;
`test/kernel/wgsl-compile.test.ts` compiles every module listed in
`kernels.ts` in every override combination of `test/helpers/override-matrix.ts`
(an explicit exported table: the defaults, each override toggled alone, and
the exact combinations the factories emit; a test asserts it covers every
`PipelineCache` key seen during the node suite, so the matrix is bounded and
complete) on Dawn's `backend=null` (returns an adapter under webgpu@0.4.0 [M],
`tmp/webgpu-plan/review/probes/dawn-facts.mjs`) and on the real device, and
the browser project compiles the same table on Chromium (graft: C G8), which
catches uniform-layout bugs that Dawn-node's `uniform_buffer_standard_layout`
masks.

`Kernel` binds a compiled pipeline to the binding list of its spec (3.5),
creates bind groups from a `Record<name, Binding>` keyed by those names and
caches them by the identity of the buffers and offsets (a layout's bind groups
are created once per `load()`, not per iteration). `WG = min(256,
caps.limits.maxComputeInvocationsPerWorkgroup)` (128 in compat mode); tiles in
workgroup memory are sized for the 16 KiB default (a 256 x `vec4<f32>` tile is
4 KiB).

### 5.2 Dispatch planning (`plan1d` / `plan2d` / `planGridStride` / `planIndirect`): the 16,776,960 rule, 2D grids, grid-stride

```ts
export interface DispatchPlan { readonly x: number; readonly y: number; readonly z: 1; readonly items: number; readonly stride: number | null; }
export function plan1d(items: number, wg: number, caps: GpuCaps): DispatchPlan;      // throws E_TOO_LARGE if a 2D grid is also insufficient
export function planGridStride(items: number, wg: number, caps: GpuCaps, maxGroups?: number): DispatchPlan;
```

Rules (design 10.6 lines 2500-2507):

- `groups = ceil(items / wg)`. If `groups <= MAX_WORKGROUPS_PER_DIM` (the
  spec's 65,535, from `constants.ts`; the device limit is asserted equal to it
  at `create()` and is not raisable, 2.2): `{ x: groups, y: 1 }`. The boundary
  test in `test/kernel/dispatch.test.ts` asserts `items = 16,776,960` is 1D
  and `16,776,961` is 2D with `wg = 256` -- NOT the round 2^24.
- Else 2D: `x = MAX_WORKGROUPS_PER_DIM`, `y = ceil(groups / MAX_WORKGROUPS_PER_DIM)`,
  and every kernel computes its item index with `linear_id()` from the prelude
  (the same constant interpolated, 3.5) and returns early when `id >=
  params.items`. `y` above the limit
  -> `E_TOO_LARGE` (that is 65,535^2 x 256 = 1.1e12 items, beyond any
  snapshot). A WGSL-side `linear_id` test at the boundary runs on lavapipe (a
  17M-item elementwise kernel, 68 MB, fits the default lane).
- Grid-stride is used for kernels whose per-item work is tiny and whose item
  count is huge (per-arc maps at 100M arcs): `groups = min(groups, maxGroups)`
  (default `caps.software ? 64 : 4096` -- the one performance default in
  `src/` that reads `caps.software`, 2.4; results never depend on it because
  grid-stride maps are order-independent) and the kernel loops `for (i = id; i
  < items; i += stride)`. The planner is pure so both branches are unit-tested
  with faked caps (spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like
  tables in `test/helpers/caps-tables.ts`, from note 05 section 4).
- Arc counts up to `0xFFFFFFFE` are passed as `u32` uniforms; JS never applies
  bitwise operators to them (`%`, `Math.floor`).

### 5.3 Uniforms

Chromium 139 lacks `uniform_buffer_standard_layout` (note 05 section 2.4), so
every params struct obeys the strict rules: scalars grouped into 16-byte-aligned
members, `vec3` never used in uniforms, arrays of scalars avoided (a 20-byte
`array<u32, 5>` is illegal, design 10.1), `bool` is not host-shareable (WGSL
6.5.2) so flags are `u32`. `UniformBlock.define([["n", "u32"], ["dim", "u32"],
["flags", "u32"], ["iteration", "u32"], ["scalingRatio", "f32"], ["gravity",
"f32"], ["jitterTolerance", "f32"], ["scale", "f32"], ["center", "vec4f"], ...])`
emits the padded WGSL struct text spliced into the module AND `write(view:
DataView, values)` that fills the bytes at the same offsets, so layout and text
cannot disagree (D20). A negative test proves that a deliberately misaligned
hand-written struct is rejected on Chromium and accepted on Dawn-node -- the
reason the generator exists.

`UniformBlock` also has a `{ layout: "storage" }` mode with a `read(view)`
counterpart, so the FA2 `state` block (7.3) -- the one storage struct whose
byte offsets TypeScript reads back for `stats` and the parity tests -- is
generated from the same descriptor as the uniforms and cannot drift from the
WGSL; an 11.3 test round-trips `state` host -> kernel -> host on both runtimes.
`partials` is never read by the host and `GridSpec` travels inside the params
uniform, so those two need no generator.

`UniformRing` writes a params object into slot `k` of one `UNIFORM` buffer with
a 256-byte stride (`minUniformBufferOffsetAlignment` is 256 in browsers and on
a default Dawn-node DEVICE; the Dawn-node adapter advertises 64 / 16, which
the package never requests; 256 satisfies all) and binds it with a dynamic
offset. A batch
of k iterations therefore writes k slots once (`writeBuffer` of `k * 256`
bytes) and records k dispatches with different dynamic offsets -- no
per-iteration `writeBuffer` calls and no host round trip. Values that change on
the device between iterations (FA2 speed, centroid, radius) live in a small
STORAGE block written by the finalize kernels (section 7.10), never in the
uniform (a compute shader cannot write a uniform-bound buffer; judge finding on
draft C 5.4).

### 5.4 Indirect dispatch

Frontier-driven kernels and the grid build's hub-cell tier use
`dispatchWorkgroupsIndirect(argsBuffer, offset)`. A one-workgroup `finalize`
kernel turns a device-side count into `(x, y, 1)` using the same rule as
`plan1d` (clamped to 65,535 with a `y` split; the kernel side uses
`linear_id`) and writes the count into a STORAGE scalar the next round reads,
so an over-limit count never reaches the API (an over-limit indirect count
runs nothing and raises no error under Dawn 0.4.0 [M],
`design-probe-nvidia.log` item 7; the finalize kernel makes it moot either
way). The finalize kernel is also the device-side SELECTOR: when a round has
several candidate pipelines (BFS: fused, two-phase top-down, bottom-up, 8.4)
every candidate is recorded for every round, one `INDIRECT | STORAGE |
COPY_DST` buffer holds one 16-byte slot per (round, candidate), and finalize
writes real args into exactly one slot per round and `(0, 0, 1)` into the
others from the device-side counts it already has. For a batch of k rounds
the buffer holds `k x candidates` slots, so k BFS levels are recorded into one
command buffer (D17); a level whose frontier is empty dispatches `(0, 0, 1)`
everywhere and the following levels are no-ops, which is what makes "record
32 levels, read the done flag once" correct without knowing the diameter.
Same-pass write-then-indirect-read is verified on Dawn 0.4.0 [M]
(`design-probe-*.log` item 8).

### 5.5 Timestamp queries and profiling

When `timestamp-query` was granted, `ctx.profiler` wraps compute passes with
`timestampWrites` (begin / end per pass) into a query set of 256 slots,
resolves into a buffer read back with the batch's staging buffer, and reports
`{ label, ns }` per pass. Chromium quantises to 100 us (note 05 section 3.1)
while Dawn-node does not (1,024 ns ticks [M], `design-probe-nvidia.log` item
6); the profiler reports `quantised: true` in browsers. Per-kernel profiling is a
Node activity (benchmarks, section 11.6); the layout's `stats.msPerIteration`
is derived from it when present, else from `performance.now()` around the
batch (wall time including queue latency, labelled as such). Gate numbers
(10.4) always use wall time around `await step()`.

### 5.6 Empty ranges, zero-length arrays

`colIdx`, `weights`, `arcToEdge` have length 0 when `arcCount === 0`;
`edgeToArc` when `edgeCount === 0` (design 10.5). The residency never creates a
zero-byte buffer and the kernel layer never dispatches a plan with `items ===
0`: `plan1d(0, ...)` returns `{ x: 0 }` and `Kernel.dispatch` skips it. A
snapshot with `nodeCount === 0` short-circuits every algorithm to its empty
result on the CPU (an empty `Uint32Array(0)`), which is not a fallback -- there
is no work. The FA2 attraction dispatch is skipped when `arcCount === 0`.

### 5.7 Error handling: validation, out-of-memory, device loss, cancellation

| Event | Detection | Behaviour |
| --- | --- | --- |
| Validation error during pipeline / bind-group creation | `pushErrorScope("validation")` around creation | thrown from the awaiting call as `E_VALIDATION { label, message }`; the message includes the buffer / pipeline label (every object is labelled) |
| Validation error at submit time (a kernel bug) | `uncapturederror` listener | routed to `options.onError`; otherwise stored in the context's pending-error slot. Under Dawn-node the event is delivered BEFORE `queue.submit()` returns [M], so `CommandBatch` checks the slot right after `submit()` and rejects ITS OWN readback with `E_VALIDATION { batchId, label }` (an invalid command buffer is a no-op submit, so the readback would otherwise resolve with stale staging bytes); in browsers delivery is asynchronous and the stored error is thrown from the NEXT public call, so a silent stderr block cannot pass a test (note 05 section 2.2); the browser test project drains the slot in `afterEach` after `onSubmittedWorkDone()` |
| Out of memory | `pushErrorScope("out-of-memory")` around every `createBuffer` above 16 MiB | `E_OUT_OF_MEMORY { requested, resident }`; the layout / algorithm releases what it allocated in the failing call |
| Device lost | `device.lost` | `ctx.state = "lost"`, all pending promises reject with `E_DEVICE_LOST { reason, message }`, residency cleared, every simulation enters `disposed`; the CALLER decides whether to create a new context -- always from a fresh `gpu.requestAdapter()`, the old adapter being consumed (2.2 step 1) -- and `load()` again (note 05 section 7.4); in the element the app's `setAccelerator(null)` re-creates the running layout on the CPU (9.4 item 1). A test destroys the device mid-batch on both runtimes. |
| An unused adapter passed to `create({ adapter })` has already created a device | `requestDevice` rejects | `E_NO_DEVICE { reason: "consumed" }` (2.2 step 1) |
| A graph-format accessor the package calls on the caller's behalf throws | `gpuView()` on a string / list / json column, `nodes.get(name)` on a column name that does not exist (`nodeMass: "<name>"`, 7.14) | the `GraphFormatError` (`E_GPU_INELIGIBLE`, `E_UNKNOWN_NODE`) propagates UNCHANGED (D12); these are the only non-`WebGpuGraphError` errors a public call can raise, and the JSDoc of every affected entry names them |
| `AbortSignal` (`GpuRunOptions.signal`) | checked between batches | the algorithm stops recording, releases its lease and rejects with `E_ABORTED`; `GpuLayoutSimulation.run()` honours the same signal; a batch already submitted completes on the device and its readback is discarded once its staging slot is returned (4.4; Q-15) |
| Wrong argument (source index >= n, mask too short, `dest` too small, a directed snapshot to an undirected-only kernel or to a layout's `load()` (Q-11: "pass `toUndirected().snapshot`"), `SharedArrayBuffer`-backed input) | argument checks before any GPU work | `E_INVALID_ARGUMENT` / `E_SNAPSHOT` with the offending value in `details`; state unchanged |
| Snapshot too large for the current plan | `planUpload` / planner | `E_TOO_LARGE { needed, limit, path, algorithm }` before allocation |
| Snapshot released while still bound | residency tombstone | `E_RELEASED { serial }` from the next call that binds it |
| Out-of-bounds storage reads (clamped by WebGPU, never an error) | -- | every kernel test compares against a CPU oracle; "no error" is never a pass (note 05 section 6 item 6) |

Aborting never leaves a residency inconsistent because uploads are atomic per
array and scratch is lease-scoped.

### 5.8 Submission model

`CommandBatch` records one `GPUCommandEncoder` with one compute pass per
"phase" (a pass may contain many dispatches; implicit barriers between
dispatches in a pass order storage writes, note 05 section 7.1),
`copyBufferToBuffer` calls for readbacks at the end, and submits ONCE.
`queue.writeBuffer` calls issued before the submit (uniform slots,
`setPosition` writes, mask updates) are ordered before it by the queue
semantics. The batch borrows one staging slot from `Readback` for its
lifetime (4.4), checks the context's pending-error slot immediately after
`submit()` (5.7) and returns `{ id: number, readback: Promise<void> }`.
Layouts and iterative algorithms never call `onSubmittedWorkDone`; `mapAsync`
on the batch's staging buffer is the completion signal. A run-scope helper
that bundles `Lease`, `CommandBatch` and `UniformRing` for an algorithm call
is an implementation convenience, not a plan requirement.

### Review notes (section 5)

- Judge verifiability (draft B 6 row 1): the one-workgroup second reduce pass
  assumed `partials <= 65,535`. Section 6 row 1 now inserts a third level
  above 16,776,960 items.
- Judge integration-feasibility (draft A 7.5 / 5.3): a kernel that binds
  `state` read-only while another writes it needs two layouts under the
  "one explicit layout per kernel" rule; resolved in 5.1 (`read_write`
  everywhere for shared blocks).

---------------------------------------------------------------------------

## 6. Primitives

Every primitive has a TypeScript interface in `src/primitives/`, one or more
WGSL modules, a stated complexity, and a CPU reference implementation in
`test/oracle/<name>.ts` used by its differential test. The order matches note
04 section 15 item 1 (what unblocks the most algorithms), reordered so the
layout slice's needs come first; "pulled in by" names the first phase that
needs the primitive (section 13).

| # | Primitive | Interface (all take a `CommandBatch` and record into it) | WGSL strategy | Complexity | CPU reference | Pulled in by |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `reduce` | `reduce(batch, src: Binding, count, op: "sum" \| "min" \| "max", dtype: "f32" \| "u32" \| "vec4f", out: Binding, outOffset)` | workgroup tree reduce of 256 items into `partials[groups]` (subgroup variant: `subgroupAdd` then one cross-subgroup pass), then ONE workgroup reduces the partials with a grid-stride loop when `groups <= 65,535`; above 16,776,960 items (2D dispatch, `groups > 65,535`) a THIRD level reduces the partials with `ceil(groups / 256)` workgroups first; deterministic order; multi-channel `vec4f` partials so several sums share one pass (graft: A 6.1) | O(count), 2-3 dispatches | `Array.reduce` in f64, tolerance scaled by count | P1 (skeleton), P3 (FA2) |
| 2 | `scan` | `exclusiveScan(batch, src, count, out, totalOut?)` (u32) | reduce-then-scan: (a) workgroup scan of 256 (Hillis-Steele in workgroup memory, or `subgroupExclusiveAdd` + cross-subgroup fixup) writing block sums, (b) scan of block sums (recursive when > 256 blocks: at most 3 levels for 16.7M items, 4 above), (c) add-back; no decoupled look-back in v1 (WGSL atomics are relaxed; note 04 section 2.1) | O(count), 3-7 dispatches | sequential prefix sum | P4 (grid), P8 (frontier) |
| 3 | `segmentedReduce` | `segmentedReduce(batch, graph: CoreBinding, tiers: DegreeTiers, per: "arc" -> value snippet, op, out)` | THREE pipelines from one module with a `TIER` override: thread-per-row for `[midEnd, n)` -- NOT `[midEnd, lowEnd)`: graph-format's `segmentOffsets = [0, hiEnd, midEnd, lowEnd, n]` puts degree-0 rows in `[lowEnd, n)` (`DEGREE_TIER_LOW = 1`, `src/snapshot/views.ts` lines 568-604), and the thread-per-row dispatch is the one that writes their identity element, so its range runs to `n` -- subgroup-per-row for `[hiEnd, midEnd)` when `subgroups` exists (without the feature the mid tier is a 32-invocations-per-row variant of the workgroup kernel, 8 rows per 256-wide workgroup, so no device ever runs a 1,000-arc row on one thread), workgroup-per-row for `[0, hiEnd)` (degree >= 1024) with a 256-wide workgroup reduce whose early exit keys on `workgroup_id` (3.5 rule 1); rows visited through `degreeOrder(opts).perm` with `override USE_PERM` (identity when the caller passes no tiers, then one dispatch over `[0, n)`); the value snippet is Gunrock's `neighborreduce` functor (note 04 section 2.2) and may reference only `(row, arc, target, weight)`; Kahan compensation in the workgroup-per-row loop | O(A), 1-3 dispatches | per-row loop in f64 | P3 (attraction, thread-per-row tier), P4 (the two upper tiers, gated at G4), P7 (PageRank) |
| 4 | `compact` | `compact(batch, flags: Binding, count, out, outCount)` and `dedupe(batch, queue, count, owner: Binding, out, outCount)` | flag + scan + scatter (3-7 dispatches); dedupe by Davidson's ownership trick (note 04 section 2.3) written race-free for WGSL: `owner` is `array<atomic<u32>>`, dispatch A does `atomicStore(&owner[v], myIndex)` for every queue entry, dispatch B reads `atomicLoad(&owner[v])` and keeps the entry iff equal (a plain store read back in the SAME dispatch is a data race and a dynamic error under WGSL 6.5.7; two dispatches with relaxed atomics make last-writer-wins well defined) | O(count) | filter / Set | P4 (grid hub cells), P7 (WCC), P8 |
| 5 | `histogram` / counting sort | `histogram(batch, keys, count, bins, out)`; `countingSortByKey(batch, keys, count, bins, outIndex, outStart)` | `atomicAdd(&count[key], 1u)` (u32); privatised in workgroup memory ONLY for <= 256 bins (1 KiB; radix digits), global atomics above that so a 4,096-bin histogram never consumes the whole 16 KiB budget (judge finding on draft A 6.5); scan; scatter with a per-bin `atomicAdd` cursor; order inside a bin is nondeterministic (documented); used by COO->CSR and the grid build's `deterministic: false` fast path | O(count + bins) | bucket loop | P4 (grid), P11 (COO->CSR) |
| 6 | `radixSort` | `radixSort(batch, keys, values, count, bits: 8 \| 16 \| 24 \| 32)` (LSD, key-value, stable) | 8 bits per pass, per pass: per-workgroup 256-bin histogram stored DIGIT-MAJOR (`hist[digit * groups + group]`, the layout a stable LSD scatter needs so that the scan of `groups x 256` (a `scan` call) yields, for each digit, the offsets of the workgroups in workgroup order), stable scatter using per-workgroup local ranking in workgroup memory (the GraphWaGu / Fuchsia structure, note 03 section 2.2, re-derived, MIT); `bits` limits passes (grid cell keys of 18 bits need 3 passes); scratch `2 x (keys + values)` from the pool; onesweep rejected for the same reason as decoupled look-back | O(count x passes), 4-5 dispatches per pass | `Array.sort` with a stable comparator; stability asserted | P4 (grid), P11 (Louvain) |
| 7 | `Frontier` | `class Frontier { readonly vertices: [Binding, Binding]; readonly count: [Binding, Binding]; readonly args: Binding; swap(); reset(batch, seed: number[]) }` | 2 x n-slot vertex queues, 2 x 4-byte counters (`atomic<u32>`), one `INDIRECT` args buffer with `k x candidates` slots and the `finalizeArgs` selector kernel (5.4); an edge-frontier buffer of `A` entries whenever `4A <= maxBufferSize` (80 MB at the 1M / 10M tier: no overflow is possible), else `capacity = floor(maxBufferSize / 4)` entries with an explicit OVERFLOW RULE: `expand` appends with a workgroup-granular `atomicAdd` and clamps its writes to the capacity, `finalizeArgs` compares the unclamped total with the capacity and, when it is larger, records `chunkStart` on the device and re-dispatches `expand` for the remaining source range (one indirect slot per chunk, at most `ceil(A / capacity)` chunks per level, all recorded in the batch); silent truncation is never possible, and G8 tests a level whose degree sum exceeds a FAKED capacity of 4,096 | O(1) per round + the advance | JS arrays | P8 |
| 8 | `advance` | `advance(batch, graph, frontier, functor: { visit: snippet, filter: snippet }, tiers?)` | Gunrock `block_mapped` (note 04 section 2.5): each workgroup loads 256 frontier vertices, scans their degrees in workgroup memory (the workgroup scan runs in uniform control flow after the guarded loads, 3.5 rule 1), then every invocation strips `[local, aggregate)` with a binary search (`upper_bound`) over the scanned degrees to find its source; a `WORKGROUP_PER_ROW` tier for rows above 1,024 arcs (from `degreeOrder().segmentOffsets`, or a per-frontier degree check when the frontier is small); subgroup tier only with the feature; output appended with one `atomicAdd` per workgroup on the queue counter (workgroup-granular allocation, note 04 section 1 table); the frontier degree sum the workgroup scans produce is also accumulated into a device counter that `finalizeArgs` reads; a fused expand-contract variant for tiny frontiers (< 4,096 entries) is SELECTED PER LEVEL ON THE DEVICE by `finalizeArgs` (5.4: both variants are recorded for every level, the threshold is a uniform, the unselected slot gets `(0, 0, 1)`), never by the host, which sees a frontier size only every 32 levels; functor snippets may reference only `(u, arc, v, weight, level)` | O(frontier degree sum) | edge loop | P8 |
| 9 | `spmv` (pull) | `spmvPull(batch, rev: CoreBinding, xNorm: Binding, y: Binding, personalization?: Binding, alpha, beta, tiers)` | `segmentedReduce` specialised: `y[v] = beta + alpha * sum_{u in in(v)} w * xNorm[u]` where `xNorm[u] = x[u] / norm[u]` is PRE-SCALED by the caller's per-node kernel (8.2), so the pull binds no normaliser; tiered by IN-degree (`degreeOrder({ of: "reverse" })`, whose `perm` occupies group 0 slot 3 even when identity) | O(A) | f64 loop | P7 |
| 10 | `cooToCsr` | `cooToCsr(batch, src, dst, w?, count, n) -> { rowPtr, colIdx, weights }` on the device | histogram by `src`, scan, scatter with cursors (Gunrock `from_coo`, note 04 section 2.7); rows NOT sorted by target (invariant I4 applies to format snapshots, not device-internal graphs; Louvain's contraction sorts arcs by (community(src), community(dst)) with `radixSort` first so rows come out sorted) | O(count + n) | `fromEdgeArrays` then compare | P11 |
| 11 | `bbox` | fused into the FA2 integrate kernel's `vec4f` min / max partials (7.4); standalone `bbox(batch, positions, n, dim, out)` = `reduce` with `op: "min"` / `"max"` | two-level f32 reduce, deterministic; GraphWaGu's i32 fixed-point `atomicMin/Max` (note 03 section 2.2 item 6) is documented as the cheaper alternative and NOT used (a fixed-point scale of 1000 on a unit-scale layout quantises the finest cell to 1e-3, coarse against a 512-cell grid of a [-1, 1] box, judge finding on draft C 7.5) | O(n), 0-4 dispatches | `Math.min/max` loop | P3 / P4 |
| 12 | `grid` | `buildGrid(batch, positions, n, dim, spec: GridSpec) -> GridBinding` (mass is the `.w` lane of `positions`, D23); `farField`, `nearField` | cell id from the ROBUST extent of 7.7 (`floor((p - gridMin) / cellSize)` per axis, clamped to the boundary cell; a node outside the extent gets the boundary cell's key and an `outside` flag), stable `radixSort` by cell id with node index as value (3 passes; bitwise deterministic within a cell) OR `countingSortByKey` (`deterministic: false`, fewer dispatches), `cellStart[cells + 2]` as the EXCLUSIVE SCAN of a per-cell `u32` histogram over the real cells plus the outside pseudo-cell of 7.7 (a scan over marks in the sorted keys would yield ranks of occupied cells only and leave every empty cell -- the majority -- unreadable; `atomicAdd` counts are order-independent, so the histogram is deterministic), per-cell `[sum m*x, sum m*y, sum m*z, sum m]` as `array<vec4f>` by a thread-per-cell segmented reduce over the sorted ranges (no atomics, no fixed point) that SKIPS cells above 1,024 entries and appends their ids to a device hub list (`atomicAdd` on `hubCount`; results are per cell, so the list order does not matter), a one-workgroup `finalize` that turns `hubCount` into indirect args (5.4), and a workgroup-per-cell dispatch over the hub list (indirect); then one `downsample` dispatch per coarser level (each parent sums 4 / 8 children) | O(n x passes + cells x levels), ~24 + levels dispatches | JS grid | P4 |

Subgroup variants (D16): reduce, scan, segmentedReduce, advance and the FA2
repulsion / near-field epilogue (7.10) have a `needs: ["subgroups"]` variant
using `subgroupAdd`, `subgroupExclusiveAdd`, `subgroupBallot`,
`subgroupBroadcast` with the lane mapping taken from `@builtin(subgroup_size)`
/ `@builtin(subgroup_invocation_id)` at runtime and the subgroup's scratch
slot from the elected-lane counter of 3.5 (never `subgroup_id`); rows per
workgroup in the subgroup-per-row tier are computed in-shader as `WG /
subgroup_size`; `SUBGROUP_MAX` only sizes scratch. CI runs them at sizes 4
(SwiftShader), 8 (lavapipe) and 32 (NVIDIA) (note 05 section 3.2), all of
which report `min == max`, so a faked-caps unit test with `min != max` (Intel
Xe: 8-32) asserts that no planner or kernel result depends on the override.
The non-subgroup twin is always compiled and tested too: every kernel that has
a twin is tested against BOTH in the same process through a second context
created with `optionalFeatures: []` (11.3), and the CI lanes additionally
re-run the node project with the test setup's `GRAPHTY_GPU_NO_SUBGROUPS=1`
(the whole project on the GPU lane, `test/primitives test/layouts` on the
default lane, 12.3).

Determinism policy: reduce, scan, segmented reduce, radix sort, grid centroids
and the exact repulsion tile are bitwise reproducible on the same device and
dispatch shape (fixed tree order, no atomics on values). Counting sort,
frontier append order and BFS `order` are set-deterministic only, which is
documented per algorithm (section 8) and asserted by the tests as set /
partition equality.

### Review notes (section 6)

- Judge performance-realism (draft A 7.7 step 4): i32 fixed-point
  mass-weighted centroid accumulation overflows for a hub of degree ~100k
  (mass 1e5 x 2^16 > 2^31). Cell centroids here are segmented reductions over
  the sorted order; no fixed point anywhere.
- Judge verifiability (draft C 6 / 7.10): a "per-bucket index sort" for
  determinism was never specified. Determinism comes from the stable radix
  sort (row 6), with counting sort as the documented non-deterministic fast
  path.
- Judge performance-realism (draft C 8.5): "64-bit packed atomicMin split
  across two u32 words" is not atomic in WGSL; MST uses the two-pass scheme
  (8.5).

---------------------------------------------------------------------------

## 7. Force-directed layouts -- FIRST DELIVERABLE

### 7.1 Scope and contract

The deliverable is `createForceAtlas2(ctx, options)` returning a
`GpuLayoutSimulation` (section 3.3) that:

1. implements design 14.3's `LayoutSimulation` over the element's stride-3
   scene-unit `Float32Array`, with the GPU buffer authoritative while stepping;
2. reproduces the ForceAtlas2 force model and adaptive-speed controller of
   `@graphty/layout`'s `forceatlas2.ts` as it will exist after the L1 rewrite
   (with the reference decisions of 7.2), so the `forceatlas2` layout type
   gives the same family of pictures with or without a GPU;
3. is steppable, settle-reporting, pinnable and draggable the way
   `ngraph.forcelayout` is used by graphty-element today (note 01 section 3),
   i.e. it can replace the DEFAULT engine at large n, not only the one-shot
   `forceatlas2Layout`;
4. scales from 10^2 nodes (parity with the CPU) through 10^4 (exact repulsion,
   interactive) to 10^5-10^6 (approximate repulsion; interactive at 10^5, batch
   at 10^6);
5. runs identically in Node (batch: `run()` until settled) and in the browser
   (per frame: `step(stepMultiplier)`);
6. honours `setFixed` every step and `setPosition` during a drag, un-settles on
   both, survives a topology change through `load` without losing placed
   coordinates (the bridge re-applies the pin mask through `report.nodeRemap`,
   9.4 item 4), in 2D uploads `z = 0`, never integrates it and writes `z =
   center.z` back on every readback (7.13), is deterministic given `seed` and
   the snapshot on the same device, and frees everything on `dispose()`.

Fruchterman-Reingold (`createFruchtermanReingold`) ships in the phase after the
grid tier (P5) because it is the same kernel family with a simpler force law and
a host-side temperature (7.20).

### 7.2 Reference semantics (decision D5) and the override constants that pin them

The CPU port (`layout/src/layouts/force-directed/forceatlas2.ts`) is a
transcription of NetworkX `forceatlas2_layout` with three measurable deviations
from the published algorithm (note 01 section 2.1.9). The GPU kernel and the L1
CPU rewrite adopt ONE reference -- the published ForceAtlas2 (Jacomy et al.
2014) as implemented by Gephi's `ForceAtlas2.java` / `ForceFactory.java` and by
cuGraph's `fa2_kernels.cuh` (note 03 sections 4.3 and 5) -- as the default,
`compat: "paper"`, and offer `compat: "networkx"`, which reproduces NetworkX
`forceatlas2_layout` exactly (owner decision Q-1, 2026-09-14: "default laws
with a networkx option"). NetworkX shares the paper's force laws; it differs
only in the per-node swing / traction form and in ACCUMULATING the global
sums across iterations (note 01 section 2.1.9), so the option is one
`override` plus one branch in the finalize kernel. The current port's own
variants (`1/d^2` repulsion, reset position-based swing) are not preserved:
they were transcription errors, not a mode anyone chose (DEPARTURE-3). The
table's third column is therefore NetworkX, not the port:

| Quantity | Port today (dropped) | Reference adopted (`compat: "paper"`, default) | NetworkX variant (`compat: "networkx"`) | Override | Why |
| --- | --- | --- | --- | --- | --- |
| Repulsion on i from j | magnitude `k m_i m_j / d^2` (lines 322-329: `factor = m_i m_j / d^2 * k`, applied to `diff / d`) | magnitude `k m_i m_j / d` (paper; Gephi `factor = coef * m1 * m2 / d / d` applied to the component vector); component update `(p_i - p_j) * k m_i m_j / d^2` | same as the paper (NetworkX `einsum` form, note 01 section 2.1.9) | -- (one law) | the published law; Gephi / NetworkX / cuGraph users expect it; the port's `1/d^2` decayed one power faster and collapsed hubs |
| Distance floor | `max(d, 0.01)` (line 266) | keep `max(d, 0.01)` (`d^2 >= 1e-4`) in the exact tier AND in the grid tier's near field (one pair law for both tiers); the grid tier's FAR field alone softens `d^2 + eps^2` with `eps = 0.25 * cellSize` written by K1 into `state` every iteration -- never an absolute constant: cuGraph's `epssq = 0.0025` belongs to positions seeded in `[-100, 100]` (`barnes_hut.cuh` lines 60, 133), while this package seeds in `[-1, 1)`, where an absolute 0.05 would damp the four nearest of eight far-field levels 2.5-25x during the expansion phase 11.4 measures | same | -- | scale-relative softening; prevents the near-field blow-up in cells |
| Coincident nodes (`d^2 < 1e-8`) | `max(d, 0.01)` gives a fixed 100x kick along a zero vector (no direction) | deterministic unit kick with magnitude `k m_i m_j / 0.01` -- the maximum the floored law produces, so the law stays monotone in `d` (a `/ 1e-4` kick would be a 1e4x discontinuity against the ramp just above the threshold) -- along the direction `hash_dir(lowbias32((min(i, j) * 0x9E3779B9u) ^ max(i, j)))`, negated when `i > j`, so `kick(i, j) = -kick(j, i)` and the 11.4 force-sum invariant holds on the coincident fixture (parentheses required by WGSL, 3.5 rule 2) | same | -- | cosmos found `sin()` hashes diverge across vendors (note 03 section 1.3) |
| Attraction (linear) | `F_i += (p_j - p_i) * w` (lines 288-297) | same | same | -- | |
| Attraction (linlog) | `F_i += (p_j - p_i) * w * log(1 + d) / d` (lines 271-285) | same | same | `LINLOG: bool` | |
| Distributed action | `F_i /= m_i` after the row sum | same | same | `DISTRIBUTED: bool` | |
| Gravity centre | centroid of positions (lines 335-364) | centroid (the port; drag and pins want a centroid pull) | same | `GRAVITY_CENTER: u32` (0 centroid, 1 origin as Gephi / cuGraph) | documented; Q-1 row |
| Gravity law | regular `-g m_i q / \|q\|` when `\|q\| > 0.01`, strong `-g m_i q` | same | same | `STRONG_GRAVITY: bool` | |
| Swing / traction per node | `swing_i = m_i \|update_i\|`, `traction_i = 0.5 m_i \|2 p_i + update_i\|` from positions (lines 379-390) | `swing_i = m_i \|F_i(t) - F_i(t-1)\|`, `traction_i = 0.5 m_i \|F_i(t) + F_i(t-1)\|` (paper; Gephi `ForceAtlas2.java`; cuGraph `compute_local_speed`) | NetworkX: `swing_i = m_i \|p_i - F_i\|`, `traction_i = 0.5 m_i \|p_i + F_i\|` (positions and forces mixed, as upstream computes them; note 01 section 2.1.9) | `SWING_MODE: u32` (0 paper, 1 networkx) | the force form is what the speed controller was designed for; mode 1 reproduces NetworkX trajectories bit-for-bit up to f32 and needs no `oldForce` buffer |
| Global swing / traction | summed fresh every iteration (port) versus accumulated from 1 across iterations (NetworkX); the port sums over every node | fresh every iteration (Gephi = cuGraph), summed over FREE nodes only (Gephi `ForceAtlas2.java` lines 283-293: `if (!n.isFixed())`), so a dragged or pinned hub cannot halve `speedEfficiency` every iteration | ACCUMULATED across iterations from `swing = traction = 1` over all nodes (NetworkX; K4 adds to `state.swing` / `state.traction` instead of overwriting, `load()` and `reheat()` reset them to 1) | `SWING_MODE` (same override) | NetworkX's accumulation is the outlier but it is what NetworkX users have; the L1 rewrite adopts the fixed-node exclusion in paper mode |
| `estimateFactor` | port lines 184-230 | unchanged (Gephi `ForceAtlas2.java` lines 296-328; the port's `if (eff > 0.05) eff *= 0.5` / `*= 0.7` CONDITIONAL form, which skips at or below the floor and never raises a sub-floor value -- not `max(eff * 0.5, 0.05)`, which differs for `eff` in `(0.05, 0.1]`) | same | -- | ported to WGSL line for line (7.10); the L1 CPU code is the executable spec of the WGSL |
| Local speed / apply | `factor = speed / (1 + sqrt(speed * swing_i))`, `p += F * factor` (lines 403-428) | unchanged, with `swing_i` recomputed inline in the integrate kernel (no per-node swing array) | same | -- | |
| `adjustSizes` correction | `d - (size_i - size_j)` (line 318) | `d - size_i - size_j` (symmetric; paper) -- DEFERRED to a later slice, `E_UNSUPPORTED` when `nodeSize` is set | same (NetworkX `distance += -size_i - size_j`) | `ADJUST_SIZES` (reserved) | rarely used; sign bug in the port |
| Edge weight influence | `w` directly (delta = 1) | same; `edgeWeightInfluence` reserved | same | -- | |
| Mass | `degree + 1` via an O(n m) scan | `outDegree()[i] + 1` (design 14.3 line 4015; cuGraph `barnes_hut.cuh` lines 176-187) | same | -- | |
| Initial positions | LCG in `[-1, 1)` per axis (port lines 57-65; `utils/random.ts` `m = 2^35 - 31, a = 185852, c = 1`) | the same LCG on the CPU in index order for NaN rows (note 01 section 2.5.1), seed 0 = unseeded (quirk preserved); `seedPositions` writes into a `Float32Array`, and every oracle consumes that f32-valued array (f64 arithmetic on f32 inputs), so "iteration 0 is bit-identical" is true by construction rather than off by up to 2^-24 | same | -- | a seed gives the same start on both paths; cross-tested against the real `RandomNumberGenerator` at W1 on the f32-rounded values (graft: C G4) |
| Termination | `totalMovement < 1e-10` (never fires) or `maxIter` | `maxIter` OR the settle rule of 7.17 | same | -- | |

These are documented behaviour changes for the CPU FA2 too; the L1 Chromatic
re-baseline commit already planned by design 14.3 lines 4041-4046 carries
them. The GPU kernel takes them as constants, never rediscovers them (note 01
section 2.1.9 recommendation). Gate G0 (section 13) requires the owner's
sign-off on this table, recorded in the P0 PR, before any WGSL that implements
a row of it merges -- P1's K3 / K4 included (D21; graft: C G3). Because the
WGSL and the f64 oracle would otherwise be two transcriptions of one table by
one author, G3 also cross-checks the oracle in `compat: "networkx"` against
committed multi-iteration TRAJECTORY fixtures generated from NetworkX
`forceatlas2_layout` (pinned version, `test/fixtures/networkx/`, 11.4), which
covers every force row of this table (shared by both modes) and the NetworkX
controller; the paper-mode controller differs from it by the two `SWING_MODE`
lines only, which a unit test pins against hand-computed values. The
published `@graphty/layout@1.6.2` port is NOT an oracle: its laws are the ones
being replaced (Q-1). `compat` is a factory option never exposed in the
element UI; the L1 CPU `ForceAtlas2Simulation` takes the same option.

### 7.3 Buffers

Device positions are `array<vec4f>` -- `xyz` plus the node's mass in `.w`
(D23) -- so every gather (attraction, tile fill, near field, cell centroids)
does one aligned 16-byte load instead of three scalar loads plus a separate
`mass` fetch (measured on the sorted near field at 1M nodes: 1.80 -> 0.92 ms
uniform, 2.01 -> 1.78 ms clustered, 2.94 -> 1.79 ms clustered with outliers,
`tmp/webgpu-plan/review/probes/near-field-order.mjs` [M]). The owner's array
stays stride-3 scene units (design 14.3): `load()` and `setPosition` repack
while they apply the inverse scale (7.18), and `toScene` unpacks while it
applies the scale, so the extra lane costs no extra pass. The kernels
therefore do NOT take the stride as a uniform and operate on the stride-3
column directly as design 14.3 lines 3997-3999 describe; the outcome that
sentence asks for (no per-frame `withComponents` copy in either direction)
holds, the mechanism is declared as DEPARTURE-7 (1.5). Design 10.2's
"never `array<vec3<f32>>`" rule concerns `components: 3` COLUMNS, not scratch.
Layout units on the device (7.18).

| Buffer | Bytes | Usage | Source / owner | Notes |
| --- | --- | --- | --- | --- |
| `rowPtr`, `colIdx`, `weights` (or `colIdx` as the dummy) | `4(n+1) + 4A (+4A)` | storage read | `ctx.residency.core(snapshot)` (undirected snapshot: both arcs present, design 10.5) | shared with every other kernel on the snapshot; freed by `release(snapshot)` |
| `perm` (degree tiers; or `rowPtr` as the dummy) | `4n` | storage read | `residency.view(snapshot, "degreeOrder")` when `segmentOffsets` reports any row of degree >= 32, else `USE_PERM = false` | |
| `positions` (layout units, `vec4f` = xyz + mass) | `16n` | storage read_write, COPY_DST | the simulation; seeded from the owner's array at `load()`; `setPosition` writes 12 bytes at `16 i` (the mass lane is untouched) | authoritative while stepping; mass = `outDegree()[i] + 1` from one O(n) loop over `rowPtr` at `load()`, or `nodeMass` resolved by `src/layouts/resolve.ts` from graph-format primitives only (7.14) |
| `scenePositions` | `12n` | storage write, COPY_SRC | the simulation | `toScene` kernel output = `positions.xyz * scale + center` (z = `center.z` in 2D, 7.13); what the staging ring copies |
| `force` | `12n` | storage read_write | the simulation | attraction writes, repulsion + gravity add |
| `oldForce` | `12n` (0 when `SWING_MODE = 1`, the NetworkX form needs no previous force) | storage read_write | the simulation | `F(t-1)` for swing / traction (7.2) |
| `fixed` mask | `4 ceil(n/32)` | storage read, COPY_DST | `setFixed(mask)` (the `NodeMask` bitmap, LSB-first, `packages/graph-format/src/util/mask.ts`) plus the bridge's temporary drag bits | re-uploaded when dirty; cleared by a `load()` that changes `n` (7.12) |
| `partials` | `groups x 64` | storage read_write | the simulation | per-workgroup regions, exactly 64 B: A = `sum p` (vec4: `.xyz` = sum of positions, `.w` = sum of `\|p - c\|^2` about the start-of-iteration centroid `c` from `state`), `min` (vec4), `max` (vec4) written by integrate; B = `swing, traction` (vec2, over FREE nodes) written by the repulsion / near-field epilogue; C = `displacement, freeCount` (vec2) written by integrate; padded to 64 B. The `.w` lane of `sum p` is what makes the RMS radius free (7.17) |
| `state` | 128 + `k x 32` | storage read_write, COPY_SRC, COPY_DST | the simulation; declared through `UniformBlock` in storage mode (5.3) so the TypeScript reader shares the WGSL offsets | `speed, speedEfficiency, swing, traction, centroid xyz, rmsRadius, radius, min xyz, max xyz, gridMin xyz, cellSize, eps, meanDisplacement, iteration, settledCount, outsideGrid, maxCellOccupancy` written by the finalize kernels; read by every kernel; COPY_DST because `load()` writes the initial centroid / bbox / radius and `reheat()` / `load()` write the controller fields from the host (7.17); a `k`-slot trace region of 32-byte records: K4 writes `(swing, traction, speed, speedEfficiency)` into slot `iterationIndex`, K1 writes `(meanDisplacement, settledCount)` into the same slot; copied to staging each batch |
| `params` uniform ring | `256 x k` per batch | uniform (dynamic offset) | `UniformRing` | `n, dim, flags, tierStart, tierEnd, scalingRatio, gravity, jitterTolerance, scale, center (vec4), iterationIndex, seed, nearMax, extentFactor, gridSpec` |
| grid tier only: `cellKey`, `cellVal` | `4n + 4n` | storage | the simulation | radix-sort keys (cell id; the outside pseudo-cell is `G^dim`, 7.7) and values (node index) |
| grid tier only: sort scratch | `2 x (4n + 4n)` | storage | pool lease per batch | keys + values ping-pong |
| grid tier only: `sortedIdx`, `cellHist`, `cellStart` | `4n + 4(cells+2) + 4(cells+2)` | storage | the simulation | `cellHist` is the per-cell `u32` histogram (real cells + the outside pseudo-cell), `cellStart` its exclusive scan (6 row 12); no separate `cellCount`: `count(c) = cellStart[c+1] - cellStart[c]` (judge finding on draft B 7.7) |
| grid tier only: `hubList`, `hubCounters` (`hubCount`, `maxOccupancy`), hub indirect args | `4 x cells + 8 + 16` | storage (+ INDIRECT for the args) | the simulation | cells with more than 1,024 entries, appended by G4 and dispatched by G4b (7.7); the counters are reset by K1 every iteration |
| grid tier only: pyramid | `16 x sum(cells per level)` | storage | the simulation | `[sum m*x, sum m*y, sum m*z, sum m]` per cell per level; 5.59 MB at 512^2 (2D), 38.3 MB at 128^3 (3D); one buffer with per-level offsets |
| staging ring | `maxInFlight x (12n + state)` | MAP_READ, COPY_DST | `Readback` | one slot per in-flight batch |

Per-node total (one accounting, also used by 4.7 and 10.1): exact tier `16 +
12 + 12 + 12 + 0.125 + 64/256` = ~53 B/node; grid tier adds `4 + 4 + 4` = 12
B/node of simulation-owned buffers (65 B/node resident) plus `16` B/node of
pool-leased sort scratch during a batch (~81 B/node at the peak of a batch),
plus the fixed pyramid, `cellHist` / `cellStart` and the hub list (note 03
section 8.2 estimates 40-64 B; ours carries `scenePositions` and `oldForce`).

### 7.4 Per-iteration kernel sequence and the binding budget

One `CommandBatch` records `k` iterations (the `iterations` argument of `step`,
the element's `stepMultiplier`) as `k` repetitions of the sequence below,
followed by `toScene` + the staging copies. Everything stays on the device;
the host sees one `mapAsync` per batch. The centroid used by gravity in
iteration `it` is the centroid of the positions at the START of `it` (the port
computes it at the start of each iteration from the previous positions, note
01 section 2.1.3 step 5): the integrate kernel of iteration `it - 1` writes the
position partials, and the first kernel of iteration `it` folds them; `load()`
computes the initial centroid, bounding box, RMS radius and radius on the CPU
and writes them into `state` (COPY_DST) so iteration 0 needs no extra pass
(graft: C 7.7 step 5).

Exact tier (n <= `exactMaxNodes`), five dispatches per iteration:

| # | Kernel (`src/wgsl/`) | Dispatch | Storage bindings (count) | Reads | Writes |
| --- | --- | --- | --- | --- | --- |
| K1 | `fa2-stats-finalize` | 1 workgroup | partials, state (2); grid tier + cellHist, hubCounters (4) | partials A and C of the previous iteration (grid-stride over <= 65,535 partials; a third level above 16.7M nodes); grid tier: the previous iteration's `cellHist[G^dim]` and `hubCounters.maxOccupancy` | `state.centroid, min, max, rmsRadius, radius, meanDisplacement` (= `select(sum / f32(freeCount), 0.0, freeCount == 0u)`: an all-fixed layout settles immediately instead of dividing by zero), `settledCount, iteration`, the trace slot's `(meanDisplacement, settledCount)`; grid tier: `gridMin, cellSize, eps` from the robust extent of 7.7, `outsideGrid`, `maxCellOccupancy`, and the reset of `hubCounters` (`hubCount = 0`, `maxOccupancy = 0`) for this iteration's G4 |
| K2 | `fa2-attraction` | 1-3 dispatches (tiers; the thread-per-row dispatch covers `[midEnd, n)` so degree-0 rows are written, 6 row 3) | rowPtr, colIdx, weights\|dummy, perm\|dummy, positions, force (6) | mass from `positions.w` when `DISTRIBUTED` | `force` (overwrites: the first writer each iteration) |
| K3 | `fa2-repulsion-exact` | `ceil(n / WG)` | positions, state, force, oldForce, fixed, partials (6) | `state.centroid` for gravity; `fixed` to exclude pinned nodes from the swing / traction sums (7.2) | `force` (+= repulsion + gravity), partials B (swing, traction over free nodes) |
| K4 | `fa2-speed-finalize` | 1 workgroup | partials, state (2) | partials B | `state.swing, traction, speed, speedEfficiency`, the trace slot's four controller fields |
| K5 | `fa2-integrate` | `ceil(n / WG)` | force, oldForce, fixed, state, positions, partials (6) | `state.speed`, `state.centroid` (for the `.w` lane) | `positions`, `oldForce`, partials A (sum, sum of squared deviation, min, max) and C (displacement, free count) |

At `k = 4` a batch is 20 dispatches + 1 `toScene` (positions, scenePositions =
2) + 2 copies. Every kernel is under the core default of 8 storage buffers per
stage; the uniform is a separate limit.

Grid tier (n > `exactMaxNodes`): K3 is replaced by the sequence of 7.7 (cell
keys; stable sort; cell histogram + scan; centroids with the hub-cell
compaction and its indirect dispatch; downsample x levels; far field; near
field with the fused gravity / swing / traction epilogue), i.e. ~33-42
dispatches per iteration in 2D (up to 8 levels) and ~31-40 in 3D (up to 6
levels): K1 + K2 (1-3) + G1 + G2 (12-15) + G3 (1 + a 3-7 dispatch scan) + G4 +
G4a + G4b + G5 (levels - 1) + G6 + G7 + K4 + K5, all in one command buffer.
The measured per-dispatch overhead is 2-6 us on the 4070 under Dawn
(`tmp/webgpu-plan/review/probes/far-field-order.mjs` [M]), so the fixed cost
is 0.1-0.3 ms per iteration.

### 7.5 Attraction over CSR rows (K2)

Per node gather over the undirected snapshot (both arcs present, so the sum is
symmetric with no atomics; cosmos's two in/out passes and cuGraph's four float
`atomicAdd`s become one loop, note 03 section 1.6 / 4.5). The module's spec
declares its bindings and overrides (3.5) -- `bindings`: group 0 `rowPtr`,
`colIdx`, `weights` (`colIdx` bound when `!HAS_WEIGHTS`, never read), `perm`
(`rowPtr` bound when `!USE_PERM`, never read); group 1 `pos: array<vec4f>`
(read), `force: array<f32>` (read_write); group 2 `P: Fa2Params`;
`overrideDecls`: `LINLOG`, `DISTRIBUTED`, `TIER` -- and the composer emits the
declarations; the `.wgsl.ts` file holds only the body:

```wgsl
// fa2-attraction.wgsl.ts (body of the thread-per-row tier; @group / @binding and override lines are generated from the spec)
@compute @workgroup_size(WG)
fn attraction(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.tierStart;
    if (row >= P.tierEnd) { return; }                          // no barrier follows in this tier, so a per-invocation return is legal (3.5 rule 1)
    let i = select(row, perm[row], USE_PERM);
    let pi = pos[i];                                           // xyz + mass in one load (D23)
    var f = vec3f(0.0);
    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {
        let j = colIdx[a];
        if (j == i) { continue; }                              // a self-loop exerts no force (one arc, design 10.5)
        var w = 1.0;
        if (HAS_WEIGHTS) { w = weights[a]; }
        let d = pos[j].xyz - pi.xyz;                           // toward j
        let len = max(length(d), 0.01);
        let mag = select(w, w * log(1.0 + len) / len, LINLOG); // linear: |F| = w * len; linlog: |F| = w * log(1 + len)
        f = f + d * mag;
    }
    if (DISTRIBUTED) { f = f / pi.w; }
    store_force(i, f);                                         // overwrites: attraction is the first writer of `force` each iteration
}
```

Load balancing: the same module compiles to three tiers by `override TIER`
(thread-per-row, subgroup-per-row, workgroup-per-row) exactly as the
`segmentedReduce` primitive (section 6 row 3), with `P.tierStart / tierEnd`
from `degreeOrder().segmentOffsets` read on the CPU; the thread-per-row tier
always runs to `n` so isolated nodes get their (zero) attraction written and
K3 never accumulates onto a stale force (the P4 gate asserts an isolated
node's total force equals gravity alone). The first slice (P3) ships the
thread-per-row tier with `USE_PERM = false` (one dispatch over `[0, n)`); the
subgroup and workgroup tiers land in P4 together with the primitive's upper
tiers and are gated at G4 with the 10k-degree-hub fixture (note 03 section
8.6); the API takes the permutation from day one so no signature changes.
Weights: `weight === true` binds `snapshot.weights` (`HAS_WEIGHTS =
flags.weighted`); `weight === "<edge column>"` is resolved by the simulation
itself at `load()` with graph-format's exported `expandEdges(snapshot,
column.data)` (`graph-format/src/index.ts` line 27) into a per-arc array
registered in the residency under `(gpuView array, column.version)` (4.1) --
no layout helper is involved: names and roles resolve through the snapshot
itself (D28; Q-30, decided 2026-09-14), so nothing is duplicated and the
`createSimulation -> forceAtlas2(options) -> load(snapshot, positions)` chain
needs no wrapper; `weight === null` -> ones. Parallel arcs sum (design
14.3 line 4037 documents the change). When `arcCount === 0` the dispatch is
skipped and `force` is zeroed by a `fill` kernel instead.

### 7.6 Repulsion, exact tier: tiled all-pairs (K3)

```wgsl
// fa2-repulsion-exact.wgsl.ts (body; bindings positions (vec4f), state, force, oldForce, fixed, partials and the SWING_MODE override come from the spec)
var<workgroup> tile: array<vec4f, WG>;          // xyz + mass, 4 KiB at WG = 256 (16 KiB default limit)

@compute @workgroup_size(WG)
fn repulsion(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    let valid = i < P.n;
    var pi = vec4f(0.0);
    if (valid) { pi = pos[i]; }
    var f = vec3f(0.0);
    let tiles = (P.n + WG - 1u) / WG;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let j = t * WG + lid.x;
        if (j < P.n) { tile[lid.x] = pos[j]; } else { tile[lid.x] = vec4f(0.0); }   // guarded fill, as the probe kernel does; one 16-byte load
        workgroupBarrier();                              // uniform: every invocation reaches it (3.5 rule 1)
        for (var s = 0u; s < WG; s = s + 1u) {
            let o = tile[s];
            let jj = t * WG + s;
            if (o.w > 0.0 && jj != i) {                      // mass is >= 1 for every real node, 0 for the pad
                let d = pi.xyz - o.xyz;
                var d2 = dot(d, d);
                if (d2 < 1.0e-8) { f = f + kick(i, jj, pi.w * o.w); continue; }   // coincident: deterministic antisymmetric kick of magnitude k m_i m_j / 0.01 (7.2)
                d2 = max(d2, 1.0e-4);                         // d >= 0.01, the port's floor
                let k = P.scalingRatio * pi.w * o.w;
                f = f + d * (k / d2);                                          // paper = NetworkX: |F| = k m_i m_j / d, along d / d
            }
        }
        workgroupBarrier();
    }
    // epilogue (7.9, 7.10): gravity and force += under the guard, the swing / traction REDUCTION outside it
    var sw = 0.0; var tr = 0.0;
    if (valid) {
        f = f + gravity(i, pi.xyz);
        let fnew = load_force(i) + f;
        store_force(i, fnew);
        if (!mask_bit(fixedMask[i >> 5u], i)) {          // fixed nodes are excluded from the global sums (7.2)
            let fold = load_old(i);
            sw = pi.w * length(fnew - fold);
            tr = 0.5 * pi.w * length(fnew + fold);
        }
    }
    workgroup_reduce2(sw, tr, lid.x);                    // uniform control flow: 256 -> 1 (subgroup variant when available), writes partials[wid].B
}
```

Properties: no atomics, deterministic (fixed summation order), `O(n^2)` pair
evaluations, memory-free, 2D and 3D identical (z is 0 for 2D and never
integrated). This kernel is also the ORACLE for the approximate tier (7.7) and
the differential target for the CPU FA2 (11.4). Measured on the RTX 4070 SUPER
under Dawn-node with the FULL FA2 body (mass multiply, floor, `select`, kick
branch; `tmp/webgpu-plan/review/probes/exact-tile.mjs` [M]):

| n | ms / iteration | pairs / s |
| --- | --- | --- |
| 4k | 0.26 | 0.64e11 |
| 8k | 0.53 | 1.27e11 |
| 16k | 1.13 | 2.38e11 |
| 20k | 1.81 (probe body without the FA2 terms: 1.06-1.11, `dawn-perf.mjs`) | 2.25e11 |
| 32k | 3.48 | 3.08e11 |
| 65k | 8.66 | 4.96e11 |
| 100k | 17.7 | 5.67e11 |

The throughput is OCCUPANCY-LIMITED below ~65k nodes (79 workgroups on 56 SMs
at 20k) and saturates near 5.7e11 pairs/s; the FA2 body costs 1.1-1.7x the
probe body (the plan's 1.3x factor holds on average). Every extrapolation in
7.8, 7.21 and 10.3 uses this curve, never a constant pairs/s. lavapipe runs
the 20k probe in 388 ms (~350x slower).

### 7.7 Repulsion, approximate tier: cell-sorted grid pyramid (option A) and cluster tree (option B)

Every large-scale GPU force layout approximates repulsion (note 03 section 0):
cuGraph / Brinkmann with Burtscher's locked quadtree (not portable: CAS
spin-locks and cross-block spin-waits, note 03 section 4.5 / 8.1), GraphWaGu
with a Hilbert-sorted 4-ary cluster tree (portable; 95k nodes / 6.6M edges in
5.48 ms/iter, 1.13M nodes in ~160 ms/iter on an RTX 4070 Laptop at theta 2 [P],
note 03 section 2.3), cosmos.gl with a grid pyramid plus a Monte-Carlo near
field (portable; 100k in 6.6 ms/step, 200k in 13.8 ms/step in WebGL [P], note
03 section 1.3). Decision D7 takes the grid pyramid as the primary back-end
(note 03 section 8.3: every build stage is a primitive the package needs
anyway, no locks, no float atomics, fixed traversal loops, production failure
modes documented, extends to 3D) and keeps the cluster tree as the documented
experiment for pathological distributions.

Geometry (`GridSpec`, computed on the CPU from `n`, `dim` and caps; the
per-iteration extent on the device; graft: A 7.7 table):

| Item | 2D | 3D |
| --- | --- | --- |
| finest grid per axis `G` | `clamp(nextPow2(2 sqrt(n)), 8, gridMax2D = 512)` (cosmos: `2 sqrt(n)` capped at 512; the cap is cosmos's WebGL texture limit carried over and is re-checked in {512, 1024, 2048} at 1M in the P4 decision record, Q-32) | `clamp(nextPow2(2 cbrt(n)), 8, gridMax3D = 128)` (Q-21) |
| grid extent (per iteration, by K1) | `extent = min(bboxExtent, extentFactor * rmsRadius)`, `extentFactor = 6` by default (Q-32), `rmsRadius = sqrt(sum \|p - c\|^2 / n)` from the `.w` lane of partials A (7.3), `bboxExtent` = the largest axis of `state.min / max` with a 1% margin; `gridMin = c - extent / 2` per axis. The bbox wins on a clean uniform or Gaussian core; the RMS bound wins under outliers. Why: with the 7.2 laws (`k m_i m_j / d` repulsion against a gravity of CONSTANT magnitude `g m_i`) an isolated node or a small component settles at `r = k M / g` (4.2e7 layout units at the 1M / 10M tier, `M = sum(deg + 1)`) while the core's radius is ~1e4, so a bbox-derived cell size collapses the whole core into a few finest cells: ONE node at 10x the core radius puts 95.5% of 1M nodes into cells above `nearMax` (`tmp/webgpu-plan/review/probes/grid-occupancy.mjs`), 0.2% of nodes at 2-4x already puts 88.5% over capacity (`near-field-order.mjs`). cosmos avoids this only by clamping positions into a fixed `spaceSize` (`update-position.frag` lines 56-57), which this package does not do; cuGraph's quadtree deepens under an outlier, a uniform grid cannot. A plain `extentFactor * rmsRadius` would quadruple core occupancy on a uniform disk (RMS = R / sqrt(2)), hence the `min`. | same rule |
| cell size | `max(extent, 1e-6) / G` (square cells; the floor keeps an all-coincident or single-node load from producing `cellSize = 0` and NaN keys); recomputed every iteration; never a fixed `spaceSize` | `max(extent, 1e-6) / G` (cubic) |
| nodes outside the extent | their CELL KEY is the OUTSIDE PSEUDO-CELL `G^2` (`G^3`), one past the last real cell, so the sort groups them, `cellHist` / `cellStart` have `cells + 2` entries, and G4 / G4b compute the pseudo-cell's mass-weighted centroid like any cell's (no atomics, no special pass). An inside node adds the pseudo-cell as one more far-field term (its true neighbours are all far away, so one centroid is the right approximation); an outside node gets the far field of every real level, NO pseudo-cell term (it would include itself) and a near field over the pseudo-cell alone, Horvitz-Thompson-sampled at `nearMax` like any over-full cell; no real cell's `count` ever includes a stray, so a boundary cell that would otherwise collect a thousand strays does not scale its 64 samples up a thousandfold. `stats.outsideGrid` reports their number | same |
| far-field softening | `eps = 0.25 * cellSize`, written by K1 into `state.eps` (7.2: never an absolute constant) | same |
| levels (coarsest 4 per axis) | `log2(G / 4) + 1`: 8 levels at G = 512 (4^2 .. 512^2) | 6 levels at G = 128 (4^3 .. 128^3) |
| finest cells | 262,144 at the cap | 2,097,152 at the cap (34 MB; 256^3 would be 268 MB and is not offered, note 03 section 8.2) |
| pyramid bytes (16 B per cell, all levels) | 349,520 cells = 5.59 MB | 2,396,736 cells = 38.3 MB |
| far-field evaluations per node | coarsest 16 - 9 = 7, then 7 levels x (36 - 9) = 189 -> 196, + 1 (the pseudo-cell) | coarsest 64 - 27 = 37, then 5 levels x (216 - 27) = 945 -> 982, + 1 |
| near-field cells | 3 x 3 = 9 | 3 x 3 x 3 = 27 |
| finest-level saturation | above n ~ 65k the cap holds and mean occupancy grows as n / 262,144 | above n ~ 262k |

Kernels per iteration (replacing K3), with their storage-binding counts. G6
and G7 are dispatched in CELL-SORTED order (D24): thread `t` handles node `i =
sortedIdx[t]`, recomputes its finest cell from `positions[i]` and `state` (no
`sortedKeys` binding: G7 is at the 8-slot cap) and writes `force` at `3 * i`.
Measured at 1M nodes on the 4070 (`near-field-order.mjs`,
`far-field-order.mjs` [M]): the SAME near-field body costs 8.4 / 32.6 / 125.3
ms in node-index order (uniform / clustered / clustered + 0.2% outliers) and
1.8 / 2.0 / 2.9 ms in sorted order; the far field 2.3-2.8 ms versus 1.0 ms. In
index order a subgroup's 32 lanes walk 32 unrelated cells with divergent loop
lengths and incoherent gathers; in sorted order they walk one. Note 03 8.3
item 5's stronger form (permuting the POSITION buffer itself every k
iterations) is a P4 decision item, not v1.

| # | Kernel | Dispatch | Bindings (count) | What |
| --- | --- | --- | --- | --- |
| G1 | `grid-cell-key` | `ceil(n / WG)` | positions, state, cellKey, cellVal (4) | `key[i] = linearise(floor((p - state.gridMin) / state.cellSize))` when every axis is in `[0, G)`, else the pseudo-cell `G^dim`; `val[i] = i`; 19-bit keys in 2D at 512^2 (2^18 real cells + 1), 22-bit in 3D at 128^3 -- still 3 passes at `bits = 24` |
| G2 | `radixSort` (3 passes, `bits = 24`) | 12-15 | its own (keys, values, scratch, histogram table) | stable sort by cell id -> `sortedIdx` (= sorted values); within a cell nodes are in index order (deterministic); `countingSortByKey` when `deterministic: false` |
| G3 | `grid-cell-hist` + `scan` | `ceil(n / WG)` + 3-7 | cellKey, cellHist (2), then the scan's own | after `encoder.clearBuffer(cellHist)` (no dispatch): `atomicAdd(&cellHist[key], 1u)` over the keys (order-independent, hence deterministic), then `cellStart = exclusiveScan(cellHist)` over `cells + 2` entries (6 row 12: every cell, empty or not, and the pseudo-cell have a start) |
| G4 | `grid-centroid` | `ceil((cells + 1) / WG)` | sortedIdx, cellStart, positions, pyramid, hubList, hubCounters (6) | thread per cell (the pseudo-cell included): `count = cellStart[c+1] - cellStart[c]`; `atomicMax(&hubCounters.maxOccupancy, count)` (a `u32` atomic, read by K1 next iteration); `count <= 1024`: loop `sortedIdx[cellStart[c] .. cellStart[c+1])` summing `p.w * p.xyz` and `p.w` -> `level0[c] = vec4f(sum m*x, sum m*y, sum m*z, sum m)`; `count > 1024`: write nothing, `hubList[atomicAdd(&hubCounters.hubCount, 1u)] = c` |
| G4a | `grid-hub-finalize` | 1 workgroup | hubCounters, hubArgs (2) | the 5.4 finalize: `hubCount` -> indirect `(x, y, 1)`; a 3D grid's 2.1M cells would make a fixed workgroup-per-cell dispatch with early exit 8,192 mostly-empty workgroups per iteration, so the compaction is used in both dimensions |
| G4b | `grid-centroid-hub` | indirect over `hubList` | sortedIdx, cellStart, positions, pyramid, hubList (5) | workgroup per hub cell: a 256-wide strided sum with a workgroup reduce (uniform control flow) -> `level0[c]`; without it the thread-per-cell loop of G4 is serial over the largest cell (up to 1M dependent gathers on one thread under a collapsed extent) |
| G5 | `grid-downsample` x (levels - 1) | `ceil(cells_L / WG)` each | pyramid (1; two offsets) | `parent = sum of 4 (8) children` -- one dispatch per level, no atomics; the pseudo-cell (index `cells` of level 0) is outside the grid and is never downsampled |
| G6 | `grid-far-field` | `ceil(n / WG)`, sorted order | positions, sortedIdx, pyramid, state, force (5) | per node `i = sortedIdx[t]` (its cell recomputed from `positions[i]` and `state`, pseudo-cell included): for an inside node, at the coarsest level sum over every cell except the 3x3 (3x3x3) neighbourhood; at each finer level sum the 6x6 (6x6x6) block aligned to the parent's 3x3 minus this level's own 3x3 (3x3x3): space tiled exactly once, no theta (cosmos `force-level.frag` lines 60-96); plus the outside pseudo-cell's centroid; for an outside node the coarsest level's 16 (64) cells in full -- it is far from all of them -- and NO pseudo-cell term (it would include itself); per cell `F += d * (k m_i M_cell / (\|d\|^2 + state.eps^2))` with `d = p_i - centroid_cell` (mass-weighted centroid, Gephi `Region` semantics); loop bounds are compile-time per `override LEVELS` |
| G7 | `grid-near-field` | `ceil(n / WG)`, sorted order | positions, sortedIdx, cellStart, state, force, oldForce, fixed, partials (8) | per node `i = sortedIdx[t]`: recompute its own finest cell from `positions[i]` and `state`; for each of the 9 (27) finest cells (an outside node: the pseudo-cell alone) iterate `sortedIdx[cellStart[c] .. cellStart[c+1])` with the EXACT pair force of 7.6 (`max(d^2, 1e-4)` floor, no softening), bounded by `nearMax = 64` entries per cell; above the cap take entries `[h, h + nearMax) mod count` with a per-iteration hashed offset `h = lowbias32(cell ^ (iteration * 0x9E3779B9u) ^ seed) % count` and scale the cell's sum by `others / sampled` -- for the node's OWN cell `(count - 1) / (nearMax - 1)` with the node itself excluded from both counts, for the other cells `count / nearMax` (Horvitz-Thompson, cosmos `force-nearfield.frag` lines 120-121 and 139; unbiased, no depth peeling because the sorted range is indexable); coincident points get the 7.2 kick; fused epilogue (gravity, `force +=`, swing / traction partials B over free nodes) exactly as K3's, with the reduction in uniform control flow |

No displacement clamp (D25). The plan's earlier `dp = clamp_length(dp, 2 *
cellSize)` in the integrate kernel was a misreading of cosmos: cosmos clamps
only the NEAR-FIELD velocity term inside `force-nearfield.frag` (lines
145-156, "the far-field grid levels still drive bulk expansion") in a fixed
`spaceSize` where points are seeded across the whole space; applied to the
whole displacement with a bbox-derived `cellSize` recomputed every iteration
it caps the layout's expansion at `(1 + 4 / G)` per iteration -- 0.78% at G =
512, i.e. ~1,000 iterations to grow from the `[-1, 1)` seed to the ~1e3-1e4
unit equilibrium against a `maxIter` of 100, four orders of magnitude of
displacement difference from the exact tier at iteration 0, and a speed
controller that runs away because displacement is decoupled from force
(`tmp/webgpu-plan/review/probes/clamp-expansion.mjs`,
`verify-clamp-throttle.mjs`). FA2's stability mechanism is the swing-based
local speed `speed / (1 + sqrt(speed * swing_i))`, which the exact tier relies
on alone and which already damps a node whose near-field estimate flips. If a
bound on the Horvitz-Thompson kick proves necessary at G4, it is applied where
cosmos applies it: to the near-field FORCE sum inside G7 before it is added to
`force` (`|f_near| <= 4 |f_far|` per node, a P4 decision item), never to
attraction, far field, gravity or the integrated step; 11.4 tests expansion
parity between the tiers instead of a clamp.

`stats.maxCellOccupancy` reports the largest finest cell and
`stats.outsideGrid` the number of nodes beyond the extent, so the element can
raise `nearMax`, or raise `gridMax2D` (FINER cells, hence lower occupancy; it
costs memory and far-field evaluations -- lowering `gridMax` would make cells
larger and occupancy higher), for a pathological graph (graft: A 7.7). The
near-field resampling of over-capacity cells changes each node's force
estimate every iteration by construction; because `speedEfficiency` floors at
0.05, a node in such a cell may never reach zero displacement. This is
recorded as risk R-24 and measured at G4 on the hub-cell fixture; the
mitigation, if needed, is to raise `nearMax` adaptively while
`maxCellOccupancy > nearMax` persists for `settleWindow` iterations.

Cost model (note 03 section 8.2; unverified until P4 measures it): far field is
`O(n x evaluations)` with coherent, branch-free access (`2 x 10^8` at 1M nodes
in 2D, `10^9` in 3D); near field is `O(n x mean occupancy)` with hub cells as
the tail, capped by `nearMax`; build is a 3-pass sort plus `O(cells x
levels)`. Expected on the 4070: 100k nodes 3-8 ms/iter (2D), 1M nodes 30-80
ms/iter (2D), 3D 1.5-2x [X] -- bracketed by cosmos's WebGL numbers below and
GraphWaGu's WebGPU tree numbers above. The sorted-order measurements above put
the whole 1M repulsion (far + near field) at ~3-5 ms plus the sort and build,
so the 30-80 ms bracket is conservative by up to an order of magnitude; it is
re-bracketed from the G4 measurements rather than lowered now.

Option B, Hilbert-sorted cluster tree (GraphWaGu 2025; note 03 section 2.2):
reuses `radixSort` and the level-wise `downsample` shape (`log_4 n` merge
dispatches, one per level), traversal with a private 64-entry stack and
`theta`. Reserved for the case the P4 hub-heavy fixtures show the grid's near
field degrading (very clumpy layouts); shares every primitive, adds ~110
B/node. Not built unless the P4 decision record demands it (Q-5).

### 7.8 Crossover by n

```ts
repulsion: "exact" | "grid" | "auto"      // default "auto": picks by n only (never by caps.software)
exactMaxNodes: number                     // default EXACT_MAX_NODES = 16,384 in constants.ts: CONSERVATIVE until G3 (the measured curve of 7.6 predicts 32,768); re-fixed at P3 with the measurement cited (Q-6)
```

Basis: Burtscher measured O(n^2) fastest below ~10k bodies on a 2009 GPU;
GraphWaGu 2022 found their O(n^2) FR best below ~5k on an RTX 2060; cosmos
switches at 4,096 because each WebGL peel pass costs ~0.1 ms (note 03 section
8.2 [P]). On the 4070 the measured curve of 7.6 puts the exact tile at 16k
nodes at 1.1-1.3 ms and at 32k at ~3.5 ms per iteration [M], both under a
frame, while the grid tier's per-iteration cost is estimated at 2-4 ms at
those sizes (build-dominated) [X]. Mechanical rule at the P3 gate (graft: C
G3): measure the exact tile at 1k / 4k / 8k / 16k / 32k / 65k (the T-4
ladder, the same one `calibrateLayout` walks by default from 8k); `exactMaxNodes`
= the largest measured n with <= 4 ms per iteration AND not slower than the
grid tier at the same n, rounded down to a power of two, written into
`constants.ts` with the benchmark session cited; the P4 gate re-checks that
the grid is not faster below it. Integrated GPUs are 10-17x slower on the
compute-bound exact tile (7.21), so their crossover is ~8k: the app calls
`calibrateLayout(ctx)` (2.2) and passes `exactMaxNodes` through
`createAccelerator(ctx, { layout })`; the package never guesses from adapter
strings (judge finding on draft C 7.5).

The exact kernel is also the ORACLE: `test/layouts/repulsion-grid.test.ts`
asserts the grid forces agree with the exact forces within the tolerances of
11.4 on uniform, clumpy and isolated-node fixtures (cosmos's 163-node
country-graph shape and 1,024 points in one cell as the stress cases) and that
the FA2 pipeline with either tier reaches the same distributional metrics and
the same layout extent.

### 7.9 Gravity and the centroid

`q = p_i - c` where `c = state.centroid` (the unweighted mean of positions as in
the port; `GRAVITY_CENTER = 1` uses the origin); regular gravity `F += -gravity
* m_i * q / |q|` when `|q| > 0.01`, strong gravity `F += -gravity * m_i * q`
(`STRONG_GRAVITY` override). `gravity = 0` is accepted (the element schema
forbids it, the CPU accepts it; note 01 section 7.1; the E1 schema loosens it
to `nonnegative()`). Fused into the repulsion epilogue (exact tier) or the
near-field epilogue (grid tier); the centroid comes from K1, which folds the
position partials the previous integrate wrote.

### 7.10 Swing, traction and the global speed: on the device (epilogue + K4)

The epilogue of the repulsion / near-field kernel computes per FREE node
`swing_i = m_i |F_i - Fold_i|` and `traction_i = 0.5 m_i |F_i + Fold_i|` (7.2;
`SWING_MODE = 1` computes NetworkX's `m_i |p_i - F_i|` / `0.5 m_i |p_i + F_i|` and K4 ACCUMULATES the sums across iterations; in mode 0 fixed nodes contribute 0, Gephi
`ForceAtlas2.java` lines 283-293), then reduces them over the workgroup in
UNIFORM control flow (256 -> 1, subgroup variant when available; 3.5 rule 1)
and writes `partials[group].B = vec2f(swing, traction)`. `fa2-speed-finalize`
(one workgroup) sums the partials and runs `estimateFactor` -- ~20 scalar
operations, a line-for-line port of the CPU `estimateFactor`
(`forceatlas2.ts` lines 184-230, whose L1 form is the executable spec of this
snippet):

```wgsl
// fa2-speed-finalize.wgsl.ts (single workgroup; the partial reduction into `swing` / `traction` is elided)
if (lid.x == 0u) {
    let n = f32(P.n);
    let optJitter = 0.05 * sqrt(n);
    let minJitter = sqrt(optJitter);
    let maxJitter = 10.0;
    let tr = max(traction, 1.0e-30);                                             // guards the division only; the ratio tests below have the same truth value with or without it
    let other = min(maxJitter, optJitter * traction / (n * n));
    var jitter = P.jitterTolerance * max(minJitter, other);
    var eff = S.speedEfficiency;
    if (swing / tr > 2.0) { if (eff > 0.05) { eff = eff * 0.5; } jitter = max(jitter, P.jitterTolerance); }   // the CPU's conditional multiply: skip at or below the floor, never raise (7.2)
    let target = select(jitter * eff * traction / swing, 1.0e30, swing == 0.0);   // +Inf in the port; 1e30 gives the same min() below
    if (swing > jitter * traction) { if (eff > 0.05) { eff = eff * 0.7; } } else if (S.speed < 1000.0) { eff = eff * 1.3; }
    S.speed = S.speed + min(target - S.speed, 0.5 * S.speed);
    S.speedEfficiency = eff; S.swing = swing; S.traction = traction;
    S.trace[P.iterationIndex].controller = vec4f(swing, traction, S.speed, eff);   // the per-batch trace read by stats / tests; K1 fills the record's other half (7.3)
}
```

cuGraph brings these two sums to the host with `thrust::reduce` per iteration
(note 03 section 4.3 step 9); with WebGPU's asynchronous submission that would
cost a `mapAsync` per iteration, so the controller stays on the device (D15).
The reduction order is fixed (partials in workgroup order, tree in lane
order), so the speed trajectory is reproducible on a device; the differential
test compares the per-iteration `(swing, traction, speed, speedEfficiency)`
trace with the CPU reference within the tolerance schedule of 11.4 (the
tight leg against an f32 oracle summing in tile order, the loose leg against
the f64 oracle).

### 7.11 Position update (K5)

```wgsl
// fa2-integrate.wgsl.ts (body)
let i = linear_id(wid, lid.x);
var dp = vec3f(0.0); var p = vec4f(0.0); var free = false; var valid = false;
if (i < P.n) {
    valid = true;
    let f = load_force(i);
    p = pos[i];
    let swing_i = select(p.w * length(p.xyz - f), p.w * length(f - load_old(i)), SWING_MODE == 0u);   // recomputed; no per-node swing array; mode 1 = NetworkX's form
    let factor = S.speed / (1.0 + sqrt(S.speed * swing_i));
    let fixed = mask_bit(fixedMask[i >> 5u], i);
    dp = select(f * factor, vec3f(0.0), fixed);                       // no clamp on dp in either tier (D25)
    if (P.dim == 2u) { dp.z = 0.0; }                                  // 2D never integrates z (7.13)
    p = vec4f(p.xyz + dp, p.w);
    pos[i] = p;
    store_old(i, f);
    free = !fixed;
}
// UNIFORM control flow from here (3.5 rule 1): workgroup reduce -> partials A (sum p.xyz, sum |p.xyz - S.centroid|^2 in the .w lane, min p, max p over valid rows)
// and C (sum |dp| over free rows, free count)
```

Fixed nodes still exert forces (they are in every gather and tile) and are
simply not moved; their `oldForce` is still updated so a later unpin does not
see a stale swing; they are excluded from the swing / traction sums (7.10) and
from the displacement mean, and included in the centroid, the RMS radius and
the bounding box. `adjustSizes`'s `0.1 * speed` factor and the 10-unit cap are
the deferred variant (7.2).

### 7.12 Fixed nodes and drag

`setFixed(mask)`: validates `mask.length >= ceil(n / 32)` (`E_INVALID_
ARGUMENT`), copies into the simulation's own words, marks the buffer dirty
(re-uploaded by `writeBuffer` before the next submit, `4 ceil(n/32)` bytes),
and reheats if any bit went from 1 to 0 (an unpin, like d3's `unpin`, note 01
section 3.2); adding pins does not reheat (d3's comment: reheating on pin
makes the layout never settle). A `load()` that changes `n` or the index
space CLEARS the simulation's fixed words and the override list below (the
mask is keyed by node index and a freeze can remap indices); the element
bridge re-issues `setFixed` from `node.pinned` after every `reload` (9.4 item
4), so pins survive a topology change without the simulation guessing at a
remap.

`setPosition(i, x, y, z)` (scene units): validates `i < n`; writes the three
floats into the OWNER's array immediately (the renderer reads it this frame);
converts to layout units and `queue.writeBuffer(positions, 16 * i, ...)` (12
bytes: the mass lane is untouched) immediately (queue order places it before
the next submit, after every batch already submitted); records `{ i,
afterBatch: lastSubmittedBatchId }` in an override list; reheats (which,
per D8, resets only the settle window and the iteration budget -- the speed
controller keeps its state, because the bridge calls `setPosition` on every
pointer move of a drag and a controller reset per frame would hold the global
speed below 1.5 for the whole drag while a settling layout runs at 2-13). When a batch's readback resolves, rows with an
override whose `afterBatch >= batch.id` are NOT copied from the readback (the
batch was computed before the write and would move the node back); the
override is cleared once a batch with `id > afterBatch` completes. This is the
race a held-mask design (draft A) closes with an extra buffer; the override
list closes it with bookkeeping only. During a drag graphty-element's
`NodeBehavior.onDragUpdate` calls `setNodePosition` on every pointer move (note
01 section 4.5); the element bridge (9.4) additionally marks the dragged node
fixed for the duration of the drag (a temporary bit OR-ed into the mask upload)
so the integrate kernel does not fight the pointer; `pinOnDrag` decides whether
the bit stays at drag end. Neighbours lag by one batch, as they do with
ngraph's one-step-per-frame today.

### 7.13 2D and 3D

`dim` is a uniform (5.1: never an override); positions are `vec4f` in both.
2D: `z` is uploaded as 0 whatever the owner's array holds, never integrated
(7.11), and every readback writes `z = center.z` (0 by default) into the
owner's array through `toScene` -- so a non-zero incoming `z` is REPLACED, not
preserved; the 11.3 property asserts `z === center.z` after any number of
steps. The grid is 2D (`G^2` cells, 3x3 / 6x6 loops). 3D: full vector maths,
`G^3` cells capped at 128 per axis, 27 / 216 loops.
graphty-element re-creates the engine on a view-mode switch
(`LayoutManager.updateLayoutDimension`, note 01 section 4.6), so the
simulation never changes `dim` after `load()`; `setParams({ dim })` is
rejected (`E_INVALID_ARGUMENT`).

### 7.14 Parameter parity with the CPU forceatlas2.ts and the element schema

| Option | CPU default (`forceatlas2.ts` lines 26-42) | Element schema (`ForceAtlas2LayoutEngine.ts` lines 99-115) | GPU v1 | Binding |
| --- | --- | --- | --- | --- |
| `maxIter` | 100 | int > 0, default 100 | honoured: total iteration budget across `step` calls (7.17) | host counter |
| `jitterTolerance` | 1.0 | > 0, default 1.0 | honoured | uniform |
| `scalingRatio` | 2.0 | > 0, default 2.0 | honoured | uniform |
| `gravity` | 1.0 | > 0, default 1.0 | honoured; 0 accepted (schema loosened at E1) | uniform |
| `strongGravity` | false | bool | honoured | `override STRONG_GRAVITY` |
| `distributedAction` | false | bool | honoured | `override DISTRIBUTED` |
| `linlog` | false | bool | honoured | `override LINLOG` |
| `nodeMass` | null -> degree + 1 | `Record \| null` | resolved by graph-format ROLE (D28; Q-30, decided 2026-09-14): `null` -> the node column with role `mass` when the snapshot has one (`nodes.byRole("mass")`, any numeric dtype through `gpuView`), else `outDegree() + 1`; a `Float32Array(n)` -> used as is; a column NAME -> `nodes.get(name)` (E_INVALID_ARGUMENT when absent or non-numeric); a `Record` -> `E_UNSUPPORTED` with the message "write a role 'mass' node column (`nodes.set(name, vec, { role: 'mass', replaceRole: true })`) or pass a Float32Array; `@graphty/layout`'s `resolveNodeVector` does this on the CPU path". graphty-element writes its `nodeMass` config as the role column once at engine creation (9.4 item 10), so both simulations find it the same way and the GPU package parses no ids | the `.w` lane of `positions` |
| `weight` | null | `weightPath: string \| null` (inert today, note 01 section 7.1) | `true` -> `snapshot.weights` (the role-`weight` column's arc array, design 3.7), a node-independent edge column NAME -> `expandEdges(snapshot, nodes.get(name))` at `load()` (7.5; D28), or `null` -> ones; BECOMES LIVE (documented change, Q-10) | `weights` binding |
| `seed` | null | `number \| null` | honoured for NaN rows via the CPU LCG; `0` means unseeded (LCG quirk preserved); also seeds the near-field hash | CPU + uniform |
| `dim` | 2 | 2..3 | honoured; fixed at `load()` | uniform |
| `pos` | null | `Record \| null` | replaced by the owner's array: finite rows are kept (inverse-scaled), NaN rows seeded (design 14.4 lines 4075-4077); the missing-axis fill of note 01 section 2.1.2 applies per row | `positions` |
| `nodeSize` / `adjustSizes` | null | `Record \| null` | DEFERRED: throws `E_UNSUPPORTED { option: "nodeSize" }` when set (sign-suspect on the CPU, 7.2; Q-25) | -- |
| `dissuadeHubs` | ignored (`_dissuadeHubs`) | bool (schema lines 66-73) | accepted and ignored, exactly like the CPU | -- |
| `scale`, `center` (`CommonLayoutOptions`) | n/a (always unit ball) | `scalingFactor` 100 | applied at `load()` (inverse) and in `toScene` (7.18); never per-step rescaling | uniform |
| `settleThreshold`, `settleWindow` (layout-owned options, L1, 9.3; the CPU simulation honours them too) | -- (added by L1) | -- (E1 exposes them) | 7.17; defaults `1e-3`, 10 (Q-8) | host |
| `iterationsPerStep` (layout-owned, L1; package default 1 -- the element passes its `stepMultiplier`), `maxInFlight` (layout-owned, L1; default 2; the CPU simulation ignores it) | -- (added by L1) | `behavior.layout.*` (9.4 item 7) | 7.19 | host |
| GPU-only: `repulsion`, `exactMaxNodes`, `nearMax`, `deterministic`, `gridMax2D`, `gridMax3D`, `extentFactor`, `compat` (`GpuLayoutTuning`, 3.3) | -- | -- (not reachable through the element in v1) | 7.7, 7.2; defaults `"auto"`, 16,384 (Q-6), 64, true, 512, 128 (Q-21), 6 (Q-32), `"paper"`; reach a simulation the element creates ONLY as the defaults given to `createAccelerator(ctx, { layout })` (9.5); a Node caller passes them to `createForceAtlas2` directly | overrides / uniform |

### 7.15 The float-atomics workaround, summarised

WGSL has `atomic<u32>` / `atomic<i32>` only (note 05 section 6 item 1). Every
accumulation in this layout is therefore a GATHER owned by one invocation
(attraction over CSR rows, repulsion over tiles or cells, near-field over
sorted ranges) or a tree REDUCTION (centroid, bbox, swing, traction,
displacement, grid centroids, downsampling). No kernel scatters a float. The
only atomics are `u32` counters in the optional counting-sort path and the
radix sort's histograms. cuGraph's edge-parallel `atomicAdd(float)` attraction
(`fa2_kernels.cuh` lines 77-80), Burtscher's spin-wait summarisation and
d3-force-webgpu's racy link kernel are explicitly NOT ported (note 03 section
8.1, note 04 section 1 table).

### 7.16 Determinism

Given the same snapshot, seed, options, device and dispatch shape, the exact
tier is bitwise reproducible (fixed loop orders, tree reductions). The grid
tier is bitwise reproducible with `deterministic: true` (stable radix sort;
near-field hash seeded from `(seed, iteration)`) and set-deterministic with
`deterministic: false` (counting sort; the `nearMax` subset of an over-full cell
may differ). Across GPUs (different subgroup sizes, different fma contraction)
coordinates differ at f32 noise level; parity tests compare distributions and
the swing / traction / speed trace, never coordinates (note 01 section 8.7).

### 7.17 Settlement

`settled` is `iterationsDone >= maxIter` OR `settledCount >= settleWindow`,
where K1 increments `settledCount` when `meanDisplacement <= settleThreshold *
rmsRadius` and resets it otherwise. The normaliser is the RMS radius
`sqrt(sum |p - centroid|^2 / n)` (the `.w` lane of partials A, 7.3), NOT the
bounding-box radius `max |p - centroid|`: with the 7.2 laws one isolated node
sits thousands of core radii out (7.7), so `settleThreshold * maxRadius` would
exceed the core's own per-iteration displacement and `settled` would fire while
the core still moves; the RMS radius is dominated by the core.
`meanDisplacement` is the mean over free nodes, `0` when every node is fixed
(K1's `select`, 7.4), so an all-fixed layout is settled at once instead of
comparing NaN forever. Defaults: `settleThreshold = 1e-3`, `settleWindow = 10`
(Q-8; ngraph's `0.01` per body and the element's 10-step average of `0.05` in
scene units are the models, note 01 section 3.1, made scale-relative because
layout units are not scene units: at `scalingFactor` 100 and a unit-radius
layout the default is 0.1 scene units per node per iteration). `setPosition`,
an unpin, `setParams` and `load` call `reheat()` = `iterationsDone = 0`,
`settledCount = 0` and nothing else (D8); `load()` additionally resets the
speed controller (`speed = 1`, `speedEfficiency = 1`, a `writeBuffer` into
`state`, which is why `state` has COPY_DST), and `setParams` resets it only
when a force LAW changes (a `compat` / `linlog` / `strongGravity` /
`distributedAction` switch), never for a numeric tweak; matching d3's reheat on
`setNodePosition` / `unpin` and ngraph's counter reset on topology change (note
01 sections 3.1-3.2), and Gephi, which initialises `speed` once. The element
must see `settled === true` within `maxIter` steps regardless of the threshold
(screenshots and label animations wait for it, note 01 section 4.3). The
settle test of 11.4 includes the isolated-node fixture and the all-fixed mask;
the near-field resampling concern of 7.7 is risk R-24.

### 7.18 Units: layout units on the device, scene units in the array

FA2's forces are not scale-invariant (linear attraction vs `1/d` repulsion),
so the simulation runs in LAYOUT units (the CPU's `[-1, 1)` seed scale) in
`positions`, and the `toScene` kernel writes `scene = p.xyz * scale + center`
(`z = center.z` in 2D, 7.13) into `scenePositions` for the staging copy;
`load()` applies the inverse on the CPU to finite rows (`fromPositionColumn`
semantics, design 14.3 line 3972) while it repacks them into `vec4f` (D23),
and `setPosition` inverts its three floats. No `rescaleLayout` per step (it would
move pinned / dragged nodes and change the camera framing every frame, note 01
section 8.4). The CPU FA2's final unit-ball normalisation is therefore NOT
reproduced by the steppable engine; the element's `scalingFactor` (default
100) sets the scene scale as it does for `SimpleLayoutEngine` today.

### 7.19 The LayoutSimulation contract in the element's frame loop

Facts: `UpdateManager.updateLayout()` calls `layoutManager.step()`
`stepMultiplier` times per render frame, synchronously, from Babylon's render
loop (`graphty-element/src/managers/UpdateManager.ts` lines 203-214);
`LayoutManager.step()` calls `engine.step()` only while `running &&
!isSettled` (`LayoutManager.ts` lines 241-245); nothing awaits (note 01 section
4.1). A Promise-returning `step()` therefore needs the fire-and-forget bridge
of decision D6.

State machine of `ForceSimulation` (`src/layouts/force-simulation.ts`):

```
created --load()--> loaded --step(k)--> loaded (inFlight 1..maxInFlight) --readback resolves--> loaded
loaded --load(next)--> loaded (generation + 1; readbacks of the old generation are discarded)
loaded --dispose()--> disposed;   any --device lost--> disposed (pending promises reject E_DEVICE_LOST)
```

`ForceSimulation` owns the state machine, the buffers every model shares
(`positions`, `scenePositions`, `fixed`, `partials`, `state`, the staging
ring), the fixed mask and override list, the settle window, the trace and the
batch driver; the per-model variation goes through ONE named hook interface
that FA2, FR and the spring-electrical preset implement (whether the
simulation consumes it by inheritance or composition is an implementation
choice):

```ts
export interface ForceModel<Options, Stats extends LayoutStatsBase> {
    readonly kind: "forceatlas2" | "fruchtermanReingold" | "springElectrical";
    buffers(n: number, dim: 2 | 3): readonly BufferSpec[];                     // model-owned buffers beyond the shared set (oldForce, velocity, the grid tier's)
    overrides(options: Options): Readonly<Record<string, number | boolean>>;   // the override set this option combination compiles to
    paramsFor(iteration: number, options: Options): UniformValues;             // the per-iteration uniform slot (FR: its temperature)
    recordIteration(batch: CommandBatch, slot: number, tier: "exact" | "grid"): void;   // the kernel sequence of 7.4 / 7.7 / 7.20
    onLoad(state: StateWriter): void;                                          // controller reset (FA2: speed = 1, speedEfficiency = 1)
    onReheat(state: StateWriter): void;                                        // FA2: nothing; FR: iteration = floor(0.7 * iterations) (7.20)
    onSetParams(patch: Partial<Options>, state: StateWriter): void;            // FA2: reset the controller only when a law changed
    readStats(state: DataView, trace: DataView): Stats;                         // model-specific stats from the generated state block
}
```

`step(k = iterationsPerStep)`:

1. `state !== "loaded"` -> reject `E_NOT_LOADED` / `E_DISPOSED` / `E_RELEASED`.
2. `settled` -> resolve immediately (no submission).
3. `inFlight >= maxInFlight` -> return the OLDEST pending promise (the call
   COALESCES: nothing is queued, the requested `k` iterations are not run, and
   the promise belongs to an earlier batch -- the 3.3 JSDoc says so; the
   element's per-frame call is naturally throttled to the GPU's pace and never
   queues unbounded work).
4. Flush host writes: dirty mask -> `writeBuffer`; changed params -> the
   batch's uniform slots; `setPosition` writes already went to the queue.
5. Record `k` iterations + `toScene` + copies into a `CommandBatch`; submit;
   `inFlight++`; `iterationsSubmitted += k`; the batch carries the current
   `generation`.
6. Return a promise that awaits the batch's `mapAsync`, then: if the batch's
   `generation` is stale (a `load()` happened meanwhile) discard it; else copy
   `scenePositions` into the owner's array with `set` (skipping overridden
   rows, 7.12), copy the state trace into `stats`, `iterationsDone += k`,
   update `settled`, `inFlight--`. `column.markDirty()` is the CALLER's job
   (the simulation does not know the column; the element bridge does it once
   per frame, design 14.4 M12).

Why `maxInFlight = 2` (judge question on draft B): with one batch in flight the
GPU idles while the CPU maps and copies the previous readback (~3 ms at 100k
nodes in Chromium, section 10.2); with two, batch `b+1` is already queued when
`b`'s readback lands, so the GPU stays busy and the owner's array lags the GPU
by exactly one in-flight batch either way. A third batch would only add lag.
`stats` and `settled` describe the LAST COMPLETED batch. The element config can
set `maxInFlight: 1` for the strictest freshness.

The element bridge (section 9.4) calls `sim.step(stepMultiplier)` once per
frame and attaches its `.catch` ONCE per distinct promise (it remembers the
last promise it saw; a coalesced call returns the same promise, and attaching
a handler per frame would fire `onError` once per frame for one failure); the
handler routes the error to the element's error channel and stops the layout,
and `LayoutManager` re-creates the engine on the next `accelerator-changed`
(9.4 item 1). Positions the renderer draws lag the simulation by one batch,
invisible for a settling layout (note 01 section 5.3 / 8.5).

Node batch API: `await sim.run({ maxIter, batch: 8 })` loops `step(batch)` until
`settled` (or `maxIter`, or the signal aborts), returning the final
`Stats` (`ForceAtlas2Stats` for FA2). The same kernels, no frame loop.

Play, pause and stop (owner question of 2026-09-14): the run flag lives in
the element, not in the simulation. Today `LayoutManager.running` gates
stepping (`LayoutManager.step()` calls the engine only while `running &&
!isSettled`, `graphty-element/src/managers/LayoutManager.ts` lines 38-47 and
241-245), `Graph.setRunning(bool)` flips it (`Graph.ts` line 2052) and drag
end sets it true (`NodeBehavior.ts` lines 114 and 188); the public element
exposes only `isRunning()` (`graphty-element.ts` line 1741). The GPU
simulation therefore has no `pause()` / `resume()` of its own; the semantics
are:

- Pause = the caller stops calling `step()`. The at most `maxInFlight`
  submitted batches complete on the device and land in the owner's array
  (their `settled` / `stats` updates apply); nothing else is submitted; the
  simulation keeps its buffers, its `speed` and its settle window; the
  owner's array is current the moment `inFlight === 0`, which `flush()`
  awaits (a screenshot after a pause or a serialisation waits on it). A
  paused simulation holds its GPU memory (10.1) until `dispose()`.
- Resume = call `step()` again. No implicit `reheat()`: a simulation paused
  at `settled === true` stays settled and `step()` resolves without a
  submission (item 2 above); the element calls `reheat()` where it sets
  `running = true` today after a drag or a pin change (9.4 items 5 and 9),
  which is the "play again" the user sees.
- Stop = `dispose()` (the simulation) plus `release(snapshot)` (the core)
  when the layout type changes or the element shuts down; a stopped
  simulation cannot be resumed (`E_DISPOSED`), the element creates a new one.
- Rendering is never paused by this package: Babylon's render loop keeps
  drawing the last landed positions, so camera, picking and styling stay
  live while the physics is paused; pausing the render loop itself is the
  element's `shutdown()` and out of scope here.

The frame-loop test (11.4) covers pause with batches in flight: stop calling
`step()` at a random tick while `inFlight === maxInFlight`, assert that
exactly those batches land, that `flush()` resolves, that no submission
happens for the next 100 ticks, and that a later `step()` continues from the
landed state (`iterationsDone` and the speed trace are continuous across the
pause).

Readback cadence: one `12n`-byte copy per batch. At 100k nodes that is 1.2 MB:
~3 ms in Chromium (measured 2.65 ms for 1 MiB copy + map + slice, note 05
section 7.2) and well under 1 ms in Node; at 1M nodes 12 MB is ~30 ms in
Chromium -- longer than a frame -- so above ~250k nodes the element should
raise `iterationsPerStep` (fewer readbacks per iteration) and accept a lower
position refresh rate (Q-7); sharing a device with Babylon's `WebGPUEngine`
(`engine._device`, note 01 section 4.8; the engine branch exists but is
unwired, 1.2) removes the readback entirely and is the documented follow-up
(Q-7, R-23).

Topology change: `reload(undirected, report, positions)` (design 14.4
M3/M4/M5) maps to `load(next, positions)` on the same simulation: the element
has already remapped / grown the array (finite rows kept, new rows NaN); the
simulation bumps its `generation` so in-flight readbacks are discarded rather
than copied into a remapped array (the concern draft A 7.18 solved by
blocking), re-uploads the new core (the old one is released by the element's
`snapshot-replaced` listener), resizes its scratch, seeds NaN rows with the
LCG inside the CURRENT bounding box so new nodes appear near the graph (a
"place at the neighbours' centroid" option is left to the element, Q-23),
clears the fixed words and the override list when `n` or the index space
changed (the bridge re-applies the pins, 7.12), and reheats. `dispose()` destroys every simulation buffer (`buffer.destroy()`
rejects a pending `mapAsync`, whose promise then resolves as discarded) and
leaves the snapshot residency to `release(snapshot)`.

### 7.20 Fruchterman-Reingold, the ngraph-like preset, ARF, Kamada-Kawai

Fruchterman-Reingold (`fruchterman-reingold.ts` lines 24-160, note 01 section
2.2) on the same skeleton:

| Item | CPU (`fruchterman-reingold.ts`) | GPU simulation |
| --- | --- | --- |
| `k` | `1 / sqrt(n)` in `[0, 1]` units | same (layout units) |
| initial positions | uniform `[0, 1)` per axis from ONE LCG (the single-RNG fix of design 14.3 line 4045) | same, on the CPU, for NaN rows |
| repulsion | `k^2 / d` along `delta / d`, `d = \|delta\| \|\| 0.1` (an exact 0 becomes 0.1, otherwise unclamped -- replicated) | the same exact-tile / grid kernels with `override LAW = FR` (component `d * k^2 / len^2`, `mass = 1`) |
| attraction | per edge `d^2 / k` toward the neighbour (unweighted; the CPU ignores weights) | CSR gather `F_i += (p_j - p_i) * len / k` per arc (both endpoints by construction) |
| temperature | `t = 0.1`, `dt = t / (iterations + 1)`, `t -= dt` per iteration (lines 81-83, 154) | the batch's `k` temperatures written into its uniform slots on the host (a batch knows them in advance); `reheat()` sets the iteration to `floor(0.7 * iterations)` so a drag gets a small temperature (graft: A 7.16) |
| apply | move along `disp` by `min(\|disp\|, t)`; skip `fixed` | integrate variant `FR_APPLY` with the fixed mask |
| termination | fixed `iterations` (50) | `settled` at `iterations` or the 7.17 movement rule |
| output | `rescaleLayout` unless `fixed` given | scene units through `toScene`; never rescaled (7.18) |
| singularities | `\|\| 0.1` exact-zero guard | replicated; coincident kick as FA2 |

No swing / traction; kernels K1, K2, K3, K5 of 7.4 with a trivial integrate
and no K4; its `FruchtermanReingoldStats` carry `temperature` instead of the
controller fields (3.3). Parity oracle: the CPU FR one-iteration displacement
(11.4).

Spring-electrical preset (note 02 section 7.1 row L3; BUILT in P5 as
`createSpringElectrical` / `GpuAccelerator.springElectrical`, 3.3):
graphty-element's default engine is `ngraph.forcelayout`
(`config/GraphBehavior.ts` line 13), a spring-electrical simulation with
Coulomb `1/d^2` repulsion, Hooke springs, drag and a velocity integrator (note
01 section 3.1), and 7.1 item 3 wants the GPU able to replace that DEFAULT
engine at large n. The preset uses `forceLaw: "fa2" | "fr" | "coulomb"` on
the repulsion kernels (the grid's centroid approximation is valid for any
radial law) and a `velocityVerlet` integrate variant with a 12n velocity
buffer, with ngraph's option names and defaults (`springLength` 10,
`springCoefficient` 0.8, `gravity` -12, `dragCoefficient` 0.9, `timeStep`
0.5) and ngraph's settle rule (total kinetic energy below a threshold),
reported through `SpringElectricalStats.kineticEnergy`. The element's
`SimulationType` gains `"spring-electrical"` and `LayoutAccelerator` gains
`springElectrical?` (9.3) so the type is selectable; whether the element
ROUTES its `ngraph` layout to the preset above a node-count threshold is a
product decision for the element / app, not a v1 change (Q-9).

ARF (`arf.ts`, 2D only, note 01 section 2.3): the pair sum is an all-pairs term
plus a CSR-row correction, i.e. the exact tile with `LAW = ARF` plus the
attraction kernel with a constant; low priority, not scheduled.

Kamada-Kawai: not a steppable force simulation (L-BFGS line search needs a cost
readback per evaluation); its GPU pieces are the APSP primitive (8.7) and a
dense cost / gradient kernel; scheduled after APSP, bounded to n <= ~10k by the
`n x n` distance matrix (note 01 section 2.4).

### 7.21 Scaling table

Basis: `E` = the measured exact-tile curve of 7.6 (FA2 body, occupancy-limited
below ~65k, 5.7e11 pairs/s at 100k) [M]; `G` = grid-tier extrapolation
bracketed by cosmos's WebGL measurements (100k: 6.6 ms, 200k: 13.8 ms) and
GraphWaGu's WebGPU tree (95k: 5.5 ms, 1.13M: 160 ms on an RTX 4070 Laptop)
[P], known to be conservative after the sorted-order measurements of 7.7; `A`
= measured CSR gather 0.1-0.7 ms per iteration for 1M arcs [M] scaled
linearly in arcs [X]; `R` = measured Chromium readback 2.65 ms per MiB [M],
Node ~0.3 ms per MB [X]; integrated GPU (Iris Xe 96 EU ~2.1 TFLOPS / ~60
GB/s, Apple M1 ~2.6 TFLOPS / 68 GB/s against the 4070 SUPER's ~35 TFLOPS /
504 GB/s, public FP32 peaks) = 10-17x slower on the compute-bound exact tile,
5-10x on the bandwidth-bound gathers and grid, and roughly EQUAL on readback
(a unified-memory `mapAsync` has no PCIe copy) [X] -- the earlier "8x / 2x"
had no basis: note 03's Iris Xe "15-35x" is GraphWaGu 2025 over GraphWaGu
2022 on the same part. Edges = 10n (average degree 10, undirected doubled: A
= 20n), the design 15.3 tiers, so this table and section 10 use one set of
graphs. All numbers are per ITERATION unless stated; unverified until P3 / P4
measure them (T-4 .. T-7; Q-17).

| Nodes / edges (arcs) | Tier | Repulsion 4070 | Attraction (A) | Total / iter 4070 | Total / iter integrated | Readback per batch (Chromium / Node) | Frames at 60 fps, `k = 1` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 10k (20k) | exact | ~0.05 ms (dispatch floor) | < 0.05 ms | ~0.3 ms (5 dispatches) | ~1-2 ms | 12 KB: < 0.2 / < 0.1 ms | trivially 60 fps, `k` up to 30 |
| 10k / 100k (200k) | exact | 0.6-0.7 ms [M] | 0.05-0.15 ms | ~0.8 ms | ~8-12 ms | 120 KB: 0.4 / 0.1 ms | 60 fps with `k` up to 8 |
| 16k / 160k (320k) | exact (default crossover) | 1.1-1.3 ms [M] | 0.1 ms | ~1.3-1.5 ms | ~14-22 ms | 192 KB: 0.6 / 0.1 ms | 60 fps, `k` up to 4 |
| 32k / 320k (640k) | exact or grid (the crossover is re-fixed at G3 / G4; exact measured ~3.5 ms [M]) | 2-4 ms (grid) | 0.15 ms | ~2.5-4.5 ms | ~15-45 ms | 384 KB: 1 / 0.2 ms | 60 fps |
| 65k / 650k (1.3M) | grid 2D | 2-5 ms | 0.15-0.9 ms | ~3-6 ms | ~15-60 ms | 780 KB: 2 / 0.3 ms | 60 fps at `k = 1-2`; integrated 15-30 fps |
| 100k / 1M (2M) | grid 2D | 3-8 ms | 0.2-1.4 ms | ~4-10 ms | ~20-100 ms | 1.2 MB: 3 / 0.4 ms | 60 fps at `k = 1`; integrated 8-30 fps |
| 100k / 1M (2M) | exact (for comparison) | ~18 ms [M] | 0.2-1.4 ms | ~18-20 ms | ~0.2-0.35 s | same | ~45 fps in Node, ~35 in Chromium -- the grid tier exists for 1M, not 100k |
| 1M / 10M (20M) | grid 2D | 30-80 ms (conservative; sorted-order far + near field measured 3-5 ms, 7.7) | 2-14 ms | ~35-95 ms | ~0.2-1 s | 12 MB: 30 / 3 ms | 10-25 fps in Node; browser readback-bound -> `k` >= 4; batch use |
| 1M / 10M (20M) | exact (for comparison) | ~1.8 s [X from the curve] | 2-14 ms | ~1.8 s | ~20-30 s | same | batch only: 100 iterations ~3 min |
| 1M / 10M (20M) | grid 3D | 50-160 ms | 2-14 ms | ~55-175 ms | ~0.3-1.7 s | same | batch |
| 10M / 100M (200M) | grid 2D (512^2 saturated: near field dominates) | 0.3-0.8 s | 20-140 ms | ~0.35-0.95 s | not targeted | 120 MB: 0.3 / 0.04 s | Node batch only; 500 iterations ~3-8 min |

The CPU FA2 for comparison allocates `n x n x dim` per iteration and is
practical only to a few thousand nodes (note 01 section 2.1.6); the CPU ngraph
engine settles a 150-node story graph today (note 01 section 6).

### Review notes (section 7)

- Judge performance-realism (drafts A and C): the attraction kernel bound 9
  and 10 storage buffers respectively. Here the swing / traction reduction is
  in the repulsion / near-field epilogue and the attraction kernel binds 6
  (7.4 table; mass rides in the position `vec4f`, D23). Judge finding on draft
  B 7.7 (near-field kernel at 9 with a redundant `cellCount`): fixed, `count =
  cellStart[c+1] - cellStart[c]`; the near-field kernel binds exactly 8 with
  the freed mass slot taken by `fixed` (7.2's free-node sums).
- Judge verifiability (draft B 7.4 row 1): the stats-reduce kernel's binding
  list contradicted the integrate kernel's displacement partials. The
  standalone stats-reduce dispatch is gone: integrate writes the position and
  displacement partials, K1 folds them at the start of the next iteration, and
  `load()` seeds the first centroid on the CPU.
- Judge performance-realism (draft A 7.6 and draft B 7.6): the tile fill
  evaluated `mass[j]` for `j >= n` inside a `select`. The fill is now an
  `if / else` as in the probe kernel.
- Judge performance-realism (draft A D8): simulating in scene units with
  `scalingFactor` as the seeding radius does not control the layout size
  (the FA2 equilibrium is set by `scalingRatio` and masses). Layout units on
  the device with `toScene` (7.18) as in drafts B and C.
- Judge verifiability (draft B 7.20 vs 10): the tables used different edge
  counts. Both now use E = 10n.
- The `held` bitmap (draft A) and the override list (draft B) solve the same
  drag race; the override list needs no buffer and is kept. Draft A's "`load()`
  waits for the in-flight readback" is replaced by a generation counter that
  discards stale readbacks without blocking a synchronous `load()`.

---------------------------------------------------------------------------

## 8. Algorithms

Every algorithm is an async function `(ctx, snapshot, ...args, options?)`
returning index-aligned typed arrays (section 3.3), grouped by the primitive
family it needs; it records k rounds per `queue.submit()`, reads back only at
the end (or every k rounds for convergence), never keeps buffers across calls
except through the residency, and checks `signal` between batches. References
are to note 04 (which read the cuGraph, Gunrock, GAP sources and the Merrill /
Beamer / Davidson / McLaughlin-Bader papers) unless a URL is given.
Result-shape parity with `indexed.*` is in section 9.7; priority scores are
note 02 section 7.2's `value x speedup / risk`.

### 8.1 Family overview

| Family | Primitives | Algorithms | Views bound | Host loop |
| --- | --- | --- | --- | --- |
| SpMV / power iteration | `spmvPull`, `segmentedReduce`, `reduce` | PageRank, personalized PageRank, HITS, eigenvector, Katz | `reverse()` (aliases forward arrays when undirected), device out-weight sums, `degreeOrder({ of: "reverse" })`, `gpuView` personalization | k iterations per submit; convergence read every k |
| Edge-parallel (each edge once) | per-arc map, `compact`, `reduce`, `histogram` | WCC (Afforest), Bellman-Ford, Boruvka MST | `edgeList().src/.dst/.weights` | fixed rounds + changed-flag every k |
| Frontier | `Frontier`, `advance`, `compact` / `dedupe`, `scan`, bitset | BFS, direction-optimizing BFS, SSSP (near-far), closeness, betweenness (multi-source), k-core peeling | `rowPtr`, `colIdx`, `weights`, `reverse()` (bottom-up on directed), `degreeOrder()` | k levels per submit with indirect dispatch |
| Sort / group-by | `radixSort`, `segmentedReduce`, `cooToCsr` | Louvain, label propagation (hub rows), triangle counting orientation | `rowPtr`, `colIdx`, `weights`, `edgeList()`, `outDegree()` | per level / pass readback of Q and move count |
| Dense | tiled matrix kernels | APSP / Floyd-Warshall (n <= 5,792 at default limits, 8.7) | `rowPtr`, `colIdx`, `weights` | one submit per block sweep |

### 8.2 SpMV family: PageRank first (A1, score 25)

Strategy (cuGraph `pagerank_impl.cuh` lines 222-320, note 04 section 5): pull
over `reverse()`. Once per call: `outWeightSum` by `segmentedReduce` over
forward `rowPtr` / `weights` (the device-side normaliser design 10.1 line 2340
prescribes; guard `sum == 0`: a node with out-arcs and zero weight sum is
dangling). Per iteration, three dispatches: (a) `pr-scale` (`ceil(n / WG)`):
`xNorm[u] = rankIn[u] / outWeightSum[u]` (0 for dangling), and per-workgroup
partials of the dangling mass `sum rankIn[u] where outWeightSum[u] == 0` and,
from the previous iteration's pair, of the L1 delta `|rankIn - rankPrev|`
(cuGraph's pre-scaled form); (b) `pr-finalize` (1 workgroup): folds the
partials into the first 16 B of `partials` (`danglingMass`, `delta`,
`firstConvergedIteration`, `iteration`) -- a STORAGE region, never a uniform,
read by the next kernel -- and records `firstConvergedIteration` the first time
`delta < tol * n`, so the reported `iterations` equals the CPU's first
converged iteration exactly (9.7) although the host reads the block only every
k; (c) `spmvPull`: `rankOut[v] = (1 - alpha) / n + alpha * (danglingMass / n +
sum_{u in in(v)} w * xNorm[u])` (+ personalization vector term when given).
Ping-pong `rankIn` / `rankOut` by swapping bind groups; `k = 8` iterations per
submit, the block read back every k (typical 20-60 iterations); when
`firstConvergedIteration` is set the batch's remaining iterations are
harmless extra work and `scores` are compared with the CPU "after equal
iterations" by re-running to that count in the test. Bindings of the pull
kernel are the FULL default of 8 storage buffers -- group 0 `revRowPtr`,
`revColIdx`, `revWeights | dummy`, `perm | dummy` (the in-degree tiers of 6
row 9; the slot exists whether or not the permutation is the identity, 3.5),
group 1 `xNorm`, `rankOut`, `personalization | dummy`, `partials` -- which is
why `outWeightSum` is not bound in the pull (it is folded into `xNorm` by (a))
and why the dangling / delta scalars live inside `partials` rather than in a
separate block; the descriptor test of 11.3 names these 8. The scale kernel
binds `rankIn`, `rankPrev`, `outWeightSum`, `xNorm`, `partials` (5); the
finalize binds `partials` (1). f32 accumulation with Kahan in the
workgroup-per-row tier; documented tolerance `1e-5` relative (design 16.2). HITS alternates two pulls
(forward and reverse) with sum normalisation; eigenvector adds an L2
normalise and converges on `delta < n * eps`; Katz is `alpha * SpMV + beta`.
WebGPU adjustment: no push / float atomics (Gunrock's `pr.hxx` push form is the
counter-example). The `outWeightSum` buffer is registered against the snapshot
in the residency so a second PageRank call on the same snapshot reuses it.

### 8.3 Connected components (A3, score 8)

Afforest from GAP `gapbs/cc.cc` lines 40-150 (note 04 section 8): `comp[v] = v`;
2 sampled link rounds over the r-th neighbour of every vertex (`colIdx[rowPtr[v]
+ r]` when `r < degree`), compress (pointer jumping; reads through `atomicLoad`
on the same `array<atomic<u32>>` because WGSL forbids mixing atomic and plain
access to one element), a 1,024-entry histogram readback to find the giant
component, then link the remaining edges of vertices not in it (`edgeList()`
each edge once, correct for directed and undirected alike, design 10.1) until a
device-side changed flag stays 0 (checked every 4 rounds). All `u32` CAS
(`atomicCompareExchangeWeak`). Readback labels -> `renumberPartition`
(graph-format `src/snapshot/derived.ts` line 1155) on the CPU in first-seen
order so `groups()` is identical to the CPU's (`renumber: false` returns raw
roots, Q-12). cuGraph's multi-root frontier expansion is more machinery for the
same asymptotics (note 04 section 8) and is not used.

### 8.4 Frontier family: BFS (A4), SSSP (A6), closeness (A5), betweenness (A7)

BFS (Merrill-Garland-Grimshaw 2011, note 04 section 3;
https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal):
two-phase (expand into an edge frontier, contract into the next vertex
frontier) as the workhorse, a fused expand-contract kernel for frontiers below
4,096 entries (Merrill's "fleeting iterations"), `atomicMin(&depth[v], level)`
as the visit claim -- the invocation that observes `old == INVALID_INDEX` is
the winner and writes `parent[v] = u`; no CAS and no retry loop, because WGSL's
`atomicCompareExchangeWeak` "may spuriously fail on some implementations"
(WGSL 17.8.5) and a vertex could then go unclaimed for its level -- and
Davidson's ownership dedupe (6 row 4: `atomicStore` in one dispatch, `atomicLoad`
in the next) as the exact safety net behind any workgroup hash culling.
Direction-optimizing variant (Beamer SC12 via cuGraph `bfs_impl.cuh` lines
291-297, 637-638, 843-846): `alpha = m / n`, `beta = 24` (Beamer's 14 / 24 as
an option), switch to bottom-up (over `reverse()`; the forward arrays when
undirected) when the frontier's degree sum exceeds the unvisited degree
estimate `/ alpha` and is growing, back when `next * 24 < unvisited` and
shrinking; bottom-up iterates the non-zero-degree unvisited list with a bitset
frontier (`atomicOr`, bulk non-atomic path when the frontier is >= 40% of n,
cuGraph `bfs_impl.cuh` lines 729-765). Every per-level choice -- fused versus
two-phase, top-down versus bottom-up -- is made ON THE DEVICE by the level's
`finalizeArgs` kernel (5.4): all three candidate dispatches are recorded for
every level, finalize evaluates the size threshold and Beamer's test from the
device counters the advance already produces (frontier count, frontier degree
sum, unvisited count and the previous level's values for "growing" /
"shrinking"), writes real args into the chosen slot and `(0, 0, 1)` into the
others, and increments a device `switches` counter on a direction change. The
host cannot make these choices: it sees a frontier size only every 32 levels,
an RMAT graph of diameter ~10 finishes inside the first submit, and a per-level
host choice would reintroduce the per-level `mapAsync` ruled out below.
Host loop: 32 levels per submit with indirect args (5.4), one 4-byte readback
of the frontier length every 32 levels -- a per-level `mapAsync` on a road
network (europe.osm ~19,000 levels [P], Merrill Table 1 via note 04 section 3)
would be slower than the CPU. The edge frontier is sized to `A` entries
whenever `4A <= maxBufferSize` (80 MB at 1M / 10M) and chunked with the
overflow rule of 6 row 7 above that; a level's degree sum can exceed any fixed
cap (a peak RMAT level carries 50-70% of A) and silent truncation would give
wrong depths, so G8 tests a level that overflows a faked 4,096-entry queue.
Result parity: `depth` exact; `parent` any valid level-1 predecessor; `order`
grouped by level (note 02 section 5); `switches` (the device counter, read
with the result) reports direction changes so a test can assert the switch
happened on an RMAT fixture.

SSSP (Davidson 2014 near-far, cuGraph `sssp_impl.cuh` lines 189-262, note 04
section 4): `dist` as `array<atomic<u32>>` holding f32 bit patterns
(`atomicMin` is exact for non-negative floats; `0x7F800000` = +Inf = unreached),
`delta = 32 * avgWeight / avgDegree`, near / far piles with the ownership
dedupe, a two-level near queue with 16 subpartitions; requires
`flags.nonNegativeWeights` (else `E_UNSUPPORTED` pointing at `bellmanFord`);
`flags.allWeightsOne` -> runs BFS. `predArc` by a second pass `atomicMin(&pred[v],
arc)` over the settled frontier where `dist[u] + w == dist[v]` (the
@antv/webgpu-graph `updatePred` idea, note 04 section 4) -- ties differ from
the CPU. The near-empty test is a device flag turned into a zero indirect
dispatch so extra queued rounds are no-ops. Bellman-Ford (A11): edge-parallel
relax over `edgeList()` both directions on undirected, `n - 1` rounds with a
changed flag every 8, one more round for the negative-cycle flag; signed floats
need a CAS loop on the bit pattern.

Closeness (A5): batched multi-source BFS (32 sources per `u32` word as a
bit-parallel frontier for unweighted graphs; repeated near-far for weighted),
per-source distance rows reduced on the device (sum, sum of 1/d, max) without
materialising `n x n` (note 04 section 7); batch size from `maxBufferSize`.

Betweenness (A7; McLaughlin-Bader CACM 2018 from the author's mirror
https://davidbader.net/publication/2018-mb/2018-mb.pdf, note 04 section 6):
forward pass = BFS with `sigma` as `array<atomic<u32>>` (`atomicAdd`, exact
until 2^32 paths, with `sigmaOverflow` reported in the result, never silent)
and per-level `S` / `ends` ranges written by the compaction; backward pass =
per level from the deepest, each `(w, s)` PULLS over its successors `v` with
`depth[s][v] == depth[s][w] + 1`: `delta[s][w] = sum sigma[s][w] / sigma[s][v]
* (1 + delta[s][v])` -- "eliminate the use of atomics by checking successors"
-- written once per `(w, s)` into the `n x k` delta array. `bc[w]` is NOT
accumulated inside the backward pass: with k tagged sources per dispatch,
`(w, s1)` and `(w, s2)` at the same level would race on a float with no
atomic available; instead one gather kernel after the batch's last backward
level does `bc[w] += sum over s of delta[s][w]` (k reads per node, no
atomics), and edge BC accumulates per arc from the same `n x k` deltas in the
same way. Sources are batched (cuGraph's tagged multi-source BFS with `n x k`
sigma / depth / delta arrays; two-word frontier entries in general; `k`
planned from `maxBufferSize` and a 25% budget, note 04 section 6);
`options.sources` / `options.k` (sampled) give the approximate variant graphty
needs at 1M nodes (exact BC is `O(n m)`); A2 adds the same two fields to
`indexed.betweennessCentrality`'s options so the shared option type carries
them and 9.7's sampled-vs-sampled comparison exists; the McLaughlin-Bader
online switch to the edge-parallel form (median BFS depth `< gamma log2 n`) is
implemented as a per-batch choice. Edge betweenness writes per arc and folds
with `foldArcs(s, vec, "first")` (design 10.7), halved on undirected snapshots
as both papers do. Cost model per source: `levels x dispatches x
per-dispatch overhead + 2 x A x bytes / bandwidth` (one forward BFS plus one
backward sweep, each streaming the arcs once at ~16 B per arc), the overhead
term divided by the batch size k (batching amortises dispatch latency, not
bandwidth): at 1M / 10M that is `2 x 20M x 16 B` = 640 MB per source, ~1.5-3
ms at 200-400 GB/s achieved, plus ~10 levels x 3 dispatches x 2-6 us; 256
sampled sources at 100k / 1M is therefore 0.5-2 s and at 1M / 10M 2-20 s [X]
(section 10.3, aligned with T-10's basis; T-11 settles it; the element's
adapter defaults to sampled BC above 50k nodes, Q-13).

### 8.5 Structure: k-core (A10), triangle counting / k-truss (A13), MST (A12)

k-core (cuGraph `core_number_impl.cuh` lines 97-230, note 04 section 9): counts
start at degree; rounds of "frontier of vertices with count < k" -> `atomicSub`
neighbour counts -> compaction; `k` increases when the frontier empties; `O(max
core)` host-visible rounds, batched 32 per submit. Triangle counting: orient
edges low-to-high degree (tie by id) as a compaction of arcs, intersect rows by
merge (rows are sorted by target, invariant I4; binary search into the longer
list when degrees differ by > 32x), `u32` atomic per-node counts, workgroup
partial totals (no single global atomic), workgroup-per-arc tier for hub pairs;
the intersection kernel binds rowPtr, colIdx, the oriented arc list (2),
counts, mask = 6. k-truss peels edges with support `< k - 2` over an `EdgeMask`
with per-edge support written through `edgeToArc`. MST (Boruvka): per-component
minimum edge via a TWO-PASS `atomicMin` (weight bits first, then the minimum
edge index among ties -- no 64-bit packed atomic exists in WGSL), union via
Afforest's compress; edge set identical to Kruskal on distinct weights,
`totalWeight` within `1e-5`.

### 8.6 Community: label propagation (A8), Louvain (A14)

Label propagation: per node the (weighted) mode of neighbour labels via a
per-row group-by-key -- rows <= 256 arcs sort keys in workgroup memory (2 KiB
per row: 256 keys + 256 f32), larger rows use a global open-addressing hash
region sized `2 x degree` (nu-Louvain's layout, note 04 section 10) --
synchronous updates with cuGraph's `up_down` swap-avoidance rule, changed-count
reduce every k. Ties are nondeterministic on the CPU too; parity is
planted-partition recovery.

Louvain (cuGraph `louvain_impl.cuh` lines 172-215, `detail/common_methods.cuh`
lines 70-152 and 402-446, note 04 section 10): per level, vertex weights by
`segmentedReduce`; synchronous best-move pass with `delta_Q = 2 * ((new - old)
/ total - resolution * (a_new k - a_old k + k^2) / total^2)` per neighbouring
community, deterministic tie-break, move only if `delta_Q > minGain` AND the
direction matches `up_down` (flips every pass); cluster weights RECOMPUTED by
reduce-by-key after each pass (never adjusted atomically; no float atomics,
deterministic); modularity by an edge reduce; contraction ON THE DEVICE by
`radixSort` of arcs by `(community src, community dst)` + `segmentedReduce` +
`cooToCsr` (the format's `contract()` is the CPU alternative the accelerator
does not use, so the package never depends on the CPU for a level); host reads
Q and the move count per pass. Expectation management: 2-10x over the CPU at
1M edges, not 100x (nu-Louvain finds GPU Louvain only 1.03x faster than a
64-thread CPU because later passes lose parallelism,
https://arxiv.org/html/2501.19004); the package runs the small levels on the
GPU too (no CPU handoff inside the package; the caller chooses the CPU package
for small graphs); partitions are not identical to the CPU's -- parity is a
modularity band. Leiden's refinement (maximal independent moves) follows once
Louvain is stable.

### 8.7 Dense: APSP / Floyd-Warshall (A9)

Blocked Floyd-Warshall with 32 x 32 tiles for weighted graphs, `n` batched BFS
rows for unweighted; result `Float32Array(n * n)` held as ONE storage binding,
so the bound is `maxStorageBufferBindingSize`, not `maxBufferSize`: `n <=
floor(sqrt(maxStorageBufferBindingSize / 4))` = 5,792 at the 128 MiB default,
23,170 at Dawn-node's 2 GiB - 4, 32,767 at Chromium's 4 GiB - 4 (note 05
section 4 table); above that the rows are windowed through the 4.2 planner,
and `E_TOO_LARGE` when even `maxBufferSize` is exceeded;
feeds Kamada-Kawai's `dist` (design 14.3 lines 4013-4014) and
`allPairsShortestPath` / `floydWarshall` adapters.

### 8.8 Priority order, the primitive each rank pulls in, and rationale

| Order | Algorithm | Score (note 02) | New primitive it pulls in | Views | Phase | Why here |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PageRank + personalized | 25 | `spmvPull` (tiered segmented reduce by in-degree), multi-channel reduce | `reverse()` (aliases forward when undirected), device out-weight sums | P7 | one pull kernel, two reductions, no atomics; first parity test against `indexed.pageRank`; the SpMV primitive serves HITS / eigenvector / Katz for free |
| 2 | HITS, eigenvector, Katz | 15 | none (same SpMV, sum / L2 normalise) | forward + `reverse()` | P7 | same kernel, different normalisation |
| 3 | WCC (Afforest) | 8 | edge map with CAS, compress, histogram sample, `compact` | `edgeList()`, `rowPtr` / `colIdx` for the r-th neighbour | P7 | proves the edge-list path and `renumberPartition` |
| 4 | BFS (+ direction-optimizing) | 5.3 | `Frontier`, `advance`, `dedupe`, bitset, indirect dispatch | `rowPtr`, `colIdx`, `reverse()` (directed), `degreeOrder()` | P8 | builds the frontier machinery every later traversal reuses |
| 5 | Closeness / harmonic / eccentricity | 5.3 | batched multi-source BFS (32 sources per word) | as BFS | P8 | same machinery |
| 6 | SSSP near-far, Bellman-Ford | 4 / 6 | `atomicMin` on f32 bits, far-pile histogram, CAS relax | `rowPtr`, `colIdx`, `weights`, flags; `edgeList()` | P8 | predecessor pass; the `flags.allWeightsOne` route to BFS |
| 7 | Betweenness (sampled and exact), edge BC | 6.3 | tagged multi-source BFS, successor-pull dependency, `S / ends` ranges | `rowPtr`, `colIdx`, `edgeList()` (edge-parallel mode), `edgeToArc` for edge BC | P9 | the most expensive thing graphty users run; needs everything above |
| 8 | APSP (then Kamada-Kawai) | 5 | blocked FW | `rowPtr`, `colIdx`, `weights` | P9 | dense; bounded n |
| 9 | k-core, triangles / k-truss, label propagation, Boruvka MST | 4.5 / 4 / 4 / 3 | peeling rounds, oriented intersection, per-row group-by-key, two-pass min | sorted rows (I4), `outDegree()`, `edgeToArc`, `edgeList()` | P11 | as demand appears; each is small once the primitives exist |
| 10 | Louvain, Leiden | 3 | `cooToCsr`, radix sort by community pair, global hash region | symmetric CSR, `edgeList()`, device weighted degree | P11 (last) | highest value, highest risk; after the sort primitive is proven by the layout's grid build |
| -- | SCC, spectral | 1.5 | forward-backward reachability; Laplacian SpMV + inverse iteration | `reverse()` | not scheduled | label order cannot match Tarjan's; spectral is a correctness fix (note 02 L4), low demand |

Not GPU targets (section 1.2) keep their CPU adapters; `degree()` stays as the
walking-skeleton diagnostic only (cheaper on the CPU than the upload). The
force-directed slice needs only `reduce`, `segmentedReduce` and, for the grid
tier, `scan`, `histogram`, `radixSort` and the grid kernels; it does not block
on the frontier machinery (note 04 section 15 item 1), which is why it can be
the first product slice.

### 8.9 Prior-art contribution table (graft: A 8.8)

| Reference | Used for | Not used / excluded |
| --- | --- | --- |
| Merrill, Garland, Grimshaw 2011 (NVIDIA research page + TR) | scan-based frontier expansion, gather tiers, duplicate culling, expand / contract couplings, the fused kernel for fleeting iterations (8.4) | the texture-cached bitmask (no equivalent worth using) |
| Beamer, Asanovic, Patterson SC12 | direction-optimizing switch and constants (8.4) | -- |
| Davidson, Baxter, Garland, Owens IPDPS 2014 | near-far SSSP, ownership dedupe (8.4, 6 row 4) | load-balanced partitioning by sorted search (a v2 advance) |
| McLaughlin, Bader CACM 2018 (author mirror PDF) | atomic-free dependency accumulation, hybrid selection, source sampling (8.4) | multi-GPU source distribution |
| cuGraph source (`bfs_impl.cuh`, `sssp_impl.cuh`, `pagerank_impl.cuh`, `betweenness_centrality_impl.cuh`, `louvain_impl.cuh`, `core_number_impl.cuh`, `triangle_count_impl.cuh`, `fa2_kernels.cuh`, `barnes_hut.cuh`) | verified constants (alpha, beta, delta, batch caps, `up_down`), pull formulations, FA2 speed controller and mass default (7.2) | float `atomicAdd` attraction, Burtscher's locked tree build (7.15) |
| Gunrock (`block_mapped.hxx`, `neighborreduce.hxx`, `csr.hxx from_coo`) | advance load balancing, segmented-reduce functor shape, COO -> CSR (6) | push PageRank / BC with float atomics |
| GAP `cc.cc` (Afforest) | connected components (8.3) | -- |
| cosmos.gl (`ForceManyBody/*`, `docs/many-body-force/README.md`) | grid pyramid, exact-once tiling, Horvitz-Thompson near field, integer hashes, failure modes and per-step numbers (7.7) | its WebGL render-pass aggregation; its two-pass link force |
| GraphWaGu (`sort.ts`, `create_tree.wgsl`, `apply_forces.wgsl`) | WGSL radix sort shape, level-wise builds, published ms/iteration (6 row 6, 7.7 option B) | i32 fixed-point bbox atomics (precision, 6 row 11); the single-thread CSR build |
| ForceAtlas2 paper (Jacomy et al. 2014) + Gephi `ForceAtlas2.java` / `ForceFactory.java` / `Region.java` (formulas only) | the reference laws of 7.2 | Gephi code (GPL / CDDL) |
| NetworkX `layout.py` | the CPU port's origin and its deviations (7.2) | accumulated swing / traction |
| Buffalo CSE 2023-06 (Kumar MS thesis: dense cuBLAS BC) | excluded: dense-only, beats McLaughlin-Bader only at >= 50% density (note 04 finding 8) | everything |
| NVIDIA cluster-analysis page | background for spectral / multilevel partitioning only; no kernel detail | everything else |
| jaredmcqueen/analytics | cautionary example only (O(n^2) WebGL1, GPL) | everything |
| @antv/webgpu-graph | cautionary: dense matrices, per-iteration readback (note 04 finding 10); the `updatePred` two-pass idea | everything else |

### 8.10 Binding budget of the algorithm kernels

Every algorithm kernel fits the core default of 8 storage buffers per stage
(3.5); the table gives the storage bindings of the kernels that come closest,
in the 7.4 format, so the descriptor test of 11.3 has a per-kernel expectation
rather than a promise. Ping-pong pairs are one buffer with two bind groups;
counters and flags share one small `u32` block; the dangling / delta scalars
of PageRank share `partials`.

| Kernel | Storage bindings (count) | Notes |
| --- | --- | --- |
| PageRank `spmvPull` (8.2) | revRowPtr, revColIdx, revWeights \| dummy, perm \| dummy, xNorm, rankOut, personalization \| dummy, partials (8) | at exactly 8, with G7 (7.7) and the five kernels below marked (8) |
| PageRank `pr-scale` / `pr-finalize` | rankIn, rankPrev, outWeightSum, xNorm, partials (5) / partials (1) | |
| Afforest link / compress (8.3) | edgeSrc, edgeDst, comp (atomic), changed (3) / comp (1) | the histogram sample binds comp, hist (2) |
| BFS expand (8.4) | rowPtr, colIdx, frontierIn, frontierCount, edgeQueue, edgeCount, chunk (7) | `chunk` = the overflow bookkeeping of 6 row 7 |
| BFS contract | edgeQueue, edgeCount, depth (atomic), parent, owner (atomic), frontierOut, frontierCount (7) | |
| BFS fused expand-contract | rowPtr, colIdx, frontierIn, frontierCount, depth, parent, frontierOut, frontierCount2 (8) | |
| BFS bottom-up | revRowPtr, revColIdx, unvisitedList, unvisitedCount, frontierBits, depth, parent, nextBits (8) | `nextBits` and `frontierBits` are the ping-pong bitsets |
| BFS `finalizeArgs` | counters block, args (2) | writes one slot per (level, candidate), 5.4 |
| SSSP near-far relax | rowPtr, colIdx, weights, dist (atomic), pred, nearIn, nearOut, farQueue (8) | the queue counters and `delta` share the `nearOut` header block (the first 16 B of the queue buffer, offset binding) so the kernel stays at 8 |
| SSSP predecessor pass | rowPtr, colIdx, weights, dist, pred (atomic), frontier (6) | |
| BC forward (tagged) | rowPtr, colIdx, frontierIn, frontierCount, depthK, sigmaK (atomic), frontierOut, frontierCount2 (8) | `n x k` arrays are single buffers indexed `s * n + v` |
| BC backward (successor pull) | rowPtr, colIdx, depthK, sigmaK, deltaK, levelRange (6) | writes `deltaK[s][w]` once |
| BC gather | deltaK, bc (2) | `bc[w] += sum_s deltaK[s][w]` |
| k-core peel | rowPtr, colIdx, count (atomic), frontierIn, frontierCount, frontierOut, frontierCount2 (7) | |
| Triangle intersection (8.5) | rowPtr, colIdx, orientedArcs, orientedCount, counts (atomic), partials (6) | |
| Label propagation (8.6) | rowPtr, colIdx, weights \| dummy, labelsIn, labelsOut, hashRegion, changed (7) | rows <= 256 arcs use workgroup memory instead of `hashRegion` |
| Louvain move | rowPtr, colIdx, weights, community, vertexWeight, clusterWeight, bestMove, hashRegion (8) | the `up_down` flag and the move counter are uniforms / a header block |
| Louvain contraction (`radixSort` + `segmentedReduce` + `cooToCsr`) | the primitives' own | sorted arcs by (community src, community dst) |

### Review notes (section 8)

- Judge performance-realism (draft B 10.3): 1,000 sampled BC sources at 100k /
  1M in 0.6 s implied 0.6 ms per source including a full BFS; corrected to the
  per-source cost model in 8.4 and the rows of 10.3.
- All drafts: `flags.sortedRows` does not exist; triangle counting relies on
  invariant I4 (design 10.5).

---------------------------------------------------------------------------

## 9. Integration with @graphty/algorithms, @graphty/layout and @graphty/graphty-element

### 9.1 The seam is async, and the dependency direction

Every public CPU algorithm is synchronous (`pageRank` at
`algorithms/src/algorithms/centrality/pagerank.ts:83`); WebGPU results need
`mapAsync`. GPU acceleration can therefore never be spliced into the sync entry
points. The seams are the element's `async run()` adapters (the abstract
`run(g: Graph): Promise<void>` at `graphty-element/src/algorithms/Algorithm.ts:217`;
line 283 is the static registry) and the design's `LayoutSimulation.step():
void | Promise<void>` (note 02 finding 1).

```
@graphty/graph-format  <---- runtime dependency ----  @graphty/webgpu-graph-algorithms   (types only, dev: @graphty/algorithms, @graphty/layout)
        ^                                                        ^
        |                                                        | injected object (no import)
@graphty/algorithms, @graphty/layout  <---- runtime ----  @graphty/graphty-element  <---- runtime ----  @graphty/graphty (the app)
   (own the accelerator interfaces)                          (owns the `accelerator` property, the bridges)      (imports the GPU package, probes, injects)
```

Acyclic: the GPU package imports the CPU packages ONLY as devDependencies,
ONLY in `test/types/conformance.test-d.ts`, for type conformance
(`expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator &
LayoutAccelerator>()` and the reverse direction; D27); the CPU packages import
nothing new; graphty-element imports nothing new AT RUNTIME (it types the
property against interfaces exported by packages it already depends on) and
gains no devDependency either, because the real-GPU stories live in the app
(9.4 item 8); the app is the only importer of the GPU package (note 02 section
4.5, mechanism (a) + (d)). The registry mechanism (self-registration through a side-effect module)
is rejected: inverted dependency, global state that breaks with duplicate
package copies, tree-shaking defeated, and it puts the "GPU threw, what now?"
decision in the CPU package (note 02 section 4.2).

### 9.2 @graphty/algorithms (lands with A2; can land as the FIRST A2 commit)

New file `algorithms/src/indexed/accelerator.ts`, exported from the barrel next
to the `indexed` namespace of design 14.2. It contains no WebGPU types.

```ts
import type { GraphSnapshot, F32, F64, U32, NumericVector } from "@graphty/graph-format";

/** Result shapes the accelerator may return: scores may be f32 (GPU) or f64 (CPU). */
export interface ScoresResultLike       { readonly scores: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface PageRankResultLike     extends ScoresResultLike { readonly danglingMass?: number | undefined; }
export interface HitsResultLike         { readonly hubs: NumericVector; readonly authorities: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface LabelResultLike        { readonly labels: U32; readonly count: number; groups(): U32[]; }     // design 14.2 line 3738
export interface BfsResultLike          { readonly depth: U32; readonly parent: U32; readonly order: U32; readonly visitedCount: number; }
export interface SsspResultLike         { readonly dist: NumericVector; readonly predArc: U32; }
export interface BellmanFordResultLike  extends SsspResultLike { readonly hasNegativeCycle: boolean; }
export interface EdgeScoresResultLike   { readonly scores: NumericVector; }
export interface ApspResultLike         { readonly dist: NumericVector; readonly n: number; }
export interface CorenessResultLike     { readonly coreness: U32; }
export interface MstResultLike          { readonly edges: U32; readonly totalWeight: number; }
export interface CommunityResultLike    extends LabelResultLike { readonly modularity: number; }

/** Structural contract an injected accelerator satisfies. Every member is optional: the accelerator declares only what it implements. */
export interface AlgorithmAccelerator {
    readonly kind: string;                                                     // "webgpu"
    pageRank?(s: GraphSnapshot, options?: PageRankOptions): Promise<PageRankResultLike>;
    personalizedPageRank?(s: GraphSnapshot, personalization: F32 | F64, options?: PageRankOptions): Promise<PageRankResultLike>;
    hits?(s: GraphSnapshot, options?: HitsOptions): Promise<HitsResultLike>;
    eigenvectorCentrality?(s: GraphSnapshot, options?: EigenvectorOptions): Promise<ScoresResultLike>;
    katzCentrality?(s: GraphSnapshot, options?: KatzOptions): Promise<ScoresResultLike>;
    connectedComponents?(s: GraphSnapshot): Promise<LabelResultLike>;
    weaklyConnectedComponents?(s: GraphSnapshot): Promise<LabelResultLike>;
    breadthFirstSearch?(s: GraphSnapshot, source: number, options?: BfsOptions): Promise<BfsResultLike>;
    sssp?(s: GraphSnapshot, source: number, options?: SsspOptions): Promise<SsspResultLike>;
    bellmanFord?(s: GraphSnapshot, source: number, options?: BellmanFordOptions): Promise<BellmanFordResultLike>;
    closenessCentrality?(s: GraphSnapshot, options?: ClosenessOptions): Promise<ScoresResultLike>;
    betweennessCentrality?(s: GraphSnapshot, options?: BetweennessOptions): Promise<ScoresResultLike>;
    edgeBetweennessCentrality?(s: GraphSnapshot, options?: BetweennessOptions): Promise<EdgeScoresResultLike>;
    allPairsShortestPath?(s: GraphSnapshot, options?: ApspOptions): Promise<ApspResultLike>;
    kCoreDecomposition?(s: GraphSnapshot): Promise<CorenessResultLike>;
    triangleCount?(s: GraphSnapshot): Promise<{ readonly perNode: U32; readonly total: number }>;
    labelPropagation?(s: GraphSnapshot, options?: LabelPropagationOptions): Promise<LabelResultLike>;
    minimumSpanningTree?(s: GraphSnapshot): Promise<MstResultLike>;
    louvain?(s: GraphSnapshot, options?: LouvainOptions): Promise<CommunityResultLike>;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}

/** The async dispatcher: one method per accelerable indexed function; delegates to the accelerator when it has the method, else runs indexed.* on the CPU. */
export interface AcceleratedAlgorithms {
    readonly accelerator: AlgorithmAccelerator | null;
    pageRank(s: GraphSnapshot, options?: PageRankOptions): Promise<PageRankResultLike>;
    sssp(s: GraphSnapshot, source: number, options?: SsspOptions): Promise<SsspResult>;   // the DESIGN's SsspResult (pathTo / pathEdges attached, see below)
    // ... the same list, non-optional; the list GROWS with the A2 ports (each port PR adds its method), it is not complete in the first commit
}
export function accelerated(acc: AlgorithmAccelerator | null | undefined): AcceleratedAlgorithms;
```

Implementation of `accelerated()` is ~60 lines: for each method, `acc?.x !==
undefined ? acc.x(s, ...) : Promise.resolve(indexed.x(s, ...))`. The
`Promise.resolve` wrapper makes the CPU path async too, which is what lets
graphty-element's `async run()` adapters treat both alike (note 02 finding 1).
`algorithms/src` has no `indexed/` directory today, so the "first A2 commit"
dispatcher carries only the methods whose `indexed.*` function has landed and
each later port PR adds its method; the dispatcher is additive at every step
and never throws for a method it does not yet have (the method does not exist,
so the adapter falls back to calling `indexed.x` directly until it does). For
`sssp` and `bellmanFord` the dispatcher DECORATES the accelerator's
`SsspResultLike { dist, predArc }` with the design's `pathTo(t)` / `pathEdges(t)`
closures (walking `predArc` through `snapshot.arcSource` and `arcToEdge`,
`graph-snapshot.ts` line 544) so both paths return the design's `SsspResult`
(design line 3735) and the element's adapters keep one result-writing loop
(9.4 item 3); the GPU package cannot attach them itself (D3). The sync
`indexed.*` functions and the legacy facades never change. Options types: the
accelerator methods reuse the `indexed.*` option types (`PageRankOptions`
etc.) so an app cannot pass a GPU option the CPU does not understand; A2 adds
`sources?: readonly number[]` and `k?: number` to `BetweennessCentralityOptions`
(today `normalized`, `endpoints`, `optimized`, `betweenness.ts` lines 15-28)
so sampled betweenness is expressible on both paths. GPU-only tuning
(`GpuLayoutTuning`, `GpuRunOptions`) goes through the GPU package's own
factory options and through the defaults given to `createAccelerator(ctx,
options)` (section 3.3), never through the dispatcher's method signatures;
`dest`, `signal` and `onProgress` are reachable only by calling the GPU
package's functions directly.

Tests in `@graphty/algorithms`: a fake accelerator `{ kind: "fake", pageRank:
async () => fixture }` proves delegation; `{ kind: "fake" }` (no methods) proves
the CPU path; a throwing method proves the throw propagates unchanged (no
fallback in the dispatcher).

### 9.3 @graphty/layout (lands with L1)

`layout/src/simulation/` (new), exported from the barrel:

```ts
export interface LayoutSimulation {                       // design 14.3 lines 3978-3985, verbatim
    load(snapshot: GraphSnapshot, positions: F32): void;
    step(iterations?: number): void | Promise<void>;
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
}
export interface SimulationOptions {                      // shared by every simulation type; the CPU simulations honour the first three and ignore maxInFlight
    readonly settleThreshold?: number; readonly settleWindow?: number;   // 7.17
    readonly iterationsPerStep?: number;                    // default 1; the element passes its stepMultiplier (9.4 item 4)
    readonly maxInFlight?: number;                          // default 2; GPU simulations only (7.19); layout-owned so the element's behavior.layout.maxInFlight has a typed path
}
export interface ForceAtlas2Options extends CommonLayoutOptions, SimulationOptions {
    readonly maxIter?: number; readonly jitterTolerance?: number; readonly scalingRatio?: number; readonly gravity?: number;
    readonly strongGravity?: boolean; readonly distributedAction?: boolean; readonly linlog?: boolean;
    readonly nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null; readonly nodeSize?: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly weight?: boolean | string | null; readonly dissuadeHubs?: boolean;
}
export interface FruchtermanReingoldOptions extends CommonLayoutOptions, SimulationOptions { readonly k?: number | null; readonly iterations?: number; readonly fixed?: NodeMask | string | null; }
export interface SpringElectricalOptions   extends CommonLayoutOptions, SimulationOptions { readonly springLength?: number; readonly springCoefficient?: number; readonly gravity?: number; readonly dragCoefficient?: number; readonly timeStep?: number; }   // ngraph's names and defaults (7.20)

export interface LayoutAccelerator {
    readonly kind: string;
    forceAtlas2?(options?: ForceAtlas2Options): LayoutSimulation;
    fruchtermanReingold?(options?: FruchtermanReingoldOptions): LayoutSimulation;
    springElectrical?(options?: SpringElectricalOptions): LayoutSimulation;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}
export type SimulationType = "forceatlas2" | "fruchtermanReingold" | "spring" | "spring-electrical";
export function createSimulation(type: SimulationType, options?: ForceAtlas2Options | FruchtermanReingoldOptions | SpringElectricalOptions, accelerator?: LayoutAccelerator | null): LayoutSimulation;
export declare class ForceAtlas2Simulation implements LayoutSimulation { constructor(options?: ForceAtlas2Options); step(iterations?: number): void; /* sync, CPU */ }
export declare class FruchtermanReingoldSimulation implements LayoutSimulation { /* sync, CPU */ }
export function resolveNodeVector(spec: F32 | string | Readonly<Record<NodeId, number>> | null | undefined, s: GraphSnapshot, fallback: (i: number) => number): F32;   // design 14.3 line 4029; the CPU path's resolver for the legacy Record form; both the CPU and the GPU simulations read a role-`mass` column first (D28; 7.14, Q-30)
export function resolveWeights(spec: boolean | string | null | undefined, s: GraphSnapshot): F32 | null;                                                      // snapshot.weights | expandEdges(column) | null
export function seedPositions(s: GraphSnapshot, positions: F32, seed: number | null, dim: 2 | 3, scale: number, center: ArrayLike<number> | null, range: "fa2" | "fr"): void;   // LCG in index order for NaN rows, written into the Float32Array (graft: C 9.2)
```

The `SimulationType` -> CPU class -> accelerator method mapping, since the
element's `"spring"` type IS Fruchterman-Reingold today
(`graphty-element/src/layout/SpringLayoutEngine.ts` line 73 "Spring layout
engine using Fruchterman-Reingold"; `layout/src/layouts/force-directed/spring.ts`
line 33 delegates to `fruchtermanReingoldLayout`):

| `SimulationType` | CPU class | Accelerator method | Note |
| --- | --- | --- | --- |
| `"forceatlas2"` | `ForceAtlas2Simulation` | `forceAtlas2` | |
| `"fruchtermanReingold"` | `FruchtermanReingoldSimulation` | `fruchtermanReingold` | |
| `"spring"` | `FruchtermanReingoldSimulation` (alias) | `fruchtermanReingold` | keeps the element's existing type name and zod schema |
| `"spring-electrical"` | none in v1 (the CPU path stays `ngraph.forcelayout` through `NGraphLayoutEngine`) | `springElectrical` | the 7.20 preset; distinct from `"spring"` |

`createSimulation` is the layout-side dispatcher: `accelerator?.forceAtlas2 !==
undefined ? accelerator.forceAtlas2(options) : new
ForceAtlas2Simulation(options)` per the table (a `"spring-electrical"` request
with no accelerator method throws, since there is no CPU simulation of that
type; the element only offers the type when the method exists). The CPU FA2 becomes STEPPABLE in the L1
rewrite (one code path in graphty-element; the GPU is an implementation swap,
note 01 section 8.8 item 3): the one-shot `indexed.forceAtlas2(s, options):
LayoutResult` is `sim.load(s, seed); for (maxIter) sim.step(); return {
positions, dim, n }` over the CSR row loop and the all-pairs loop with no
per-pair allocation (design 14.3 port 2). The L1 rewrite adopts the 7.2
reference formulas; `layout/test/forceatlas2-layout.test.ts` asserts no exact
coordinate (note 01 section 2.1.8) so it keeps passing, and the Chromatic
re-baseline commit documents the change. `seedPositions` is what both the GPU
package and the element call so a seed gives the same start on every path; it
writes f32 values (the array is a `Float32Array`), so an f64 oracle that reads
the same array starts bit-identically (7.2); the GPU package carries its own
copy of the LCG for the life of the package (D27: it cannot import layout) and
cross-tests it against the real `RandomNumberGenerator` at W1 (10,000 draws,
bit-identical after f32 rounding).

### 9.4 @graphty/graphty-element (lands with E1)

Exact changes (on top of the design's E1 port of 14.4 -- `DataManager` owning
the builder, `getSnapshot()`, `snapshot-replaced`, `dm.undirected(s)`, the
element-owned position column and the `load` / `reload` engine shape -- which
is a precondition of this list, 9.8):

1. `Graph` gains `accelerator: GraphAccelerator | null` (default `null`) with
   `setAccelerator(acc)` and an `accelerator-changed` event, where

   ```ts
   export type GraphAccelerator = AlgorithmAccelerator & LayoutAccelerator & { release(s: GraphSnapshot): void; dispose?(): void };
   ```

   (both interface types come from packages the element already depends on:
   `graphty-element/package.json` lists `@graphty/algorithms` and
   `@graphty/layout`). `LayoutManager` is the CONSUMER of `accelerator-changed`:
   when the active engine is a `SimulationLayoutEngine` it disposes the old
   simulation, re-creates one through `createSimulation(type, opts,
   graph.accelerator)`, calls `load(dm.undirected(dm.getSnapshot()).snapshot,
   positions)` on the element's own array (coordinates survive because the
   array is element-owned, design 4067-4077) and re-applies the pin mask; so
   an accelerator injected AFTER the layout was set (the app's
   `attachAccelerator` is async; a declarative layout attribute may run first)
   engages without a layout change, and `setAccelerator(null)` after a GPU
   failure -- the user's "disable acceleration" action, which the element
   never takes by itself -- moves the RUNNING layout onto the CPU simulation
   instead of leaving it stopped. Algorithm runs are unaffected: they read
   `graph.accelerator` per call.
2. `DataManager`'s `snapshot-replaced` listener list (design 14.4 line 4155)
   gets the release list of 4.5: `graph.accelerator?.release(previous)` when
   `previous !== null`, `release(dm.undirected(previous).snapshot)` when that is
   a distinct snapshot, and the `visible(previous)` cache's induced and
   undirected snapshots. `Graph.dispose()` releases the same list for the
   current snapshot but does NOT call `dispose()` (the app owns the
   accelerator's lifetime; two elements may share one).
3. Algorithm adapters (`graphty-element/src/algorithms/*Algorithm.ts`, the
   `async run()` of `Algorithm.ts:217`): the body of design 14.4 M7 becomes
   `const s = dm.getSnapshot(); const r = await accelerated(this.graph.accelerator)
   .pageRank(s, opts); for (i < n) addNodeResult(s.ids.idOf(i), "rank",
   r.scores[i]);` -- ONE result-writing loop for CPU and GPU (the `*Pct`
   normalisation is an O(n) pass over the readback, CPU side, unchanged); the
   SSSP adapters call `r.pathTo(t)` on both paths because the dispatcher
   attaches it (9.2). Adapters whose algorithm is not a GPU target call
   `indexed.x(s)` directly. Undirected adapters pass `dm.undirected(s).snapshot`
   exactly as for the CPU (design 14.4 lines 4189-4197), so the GPU never sees
   a graph the CPU would not. A GPU result carries `precision: "f32"` (3.3);
   the adapter labels it.
4. `LayoutManager._setLayoutInternal` (note 01 section 4.2): for
   `SimulationType` layouts it creates `new SimulationLayoutEngine(type, opts,
   createSimulation(type, opts, graph.accelerator))`, a bridge implementing the
   design-E1 engine shape (`load` / `reload` / `step` / `pin` / `unpin` /
   `setNodePosition` / `getNodePositionInto` / `dispose`; today's abstract
   `LayoutEngine`, `graphty-element/src/layout/LayoutEngine.ts` lines 36-63,
   has `init` / `addNode` / `addEdge` / `getNodePosition` / `getEdgePosition`
   / `nodes` / `edges` and none of these, so the shape lands with E1) over a
   `LayoutSimulation`:

   ```ts
   class SimulationLayoutEngine extends LayoutEngine {
       private pending: Promise<void> | null = null;                // the last promise step() returned, so .catch is attached ONCE per distinct promise
       async init(): Promise<void> { /* nothing: load happens in setLayout after getSnapshot() */ }
       load(snapshot: GraphSnapshot, positions: F32): void { this.sim.load(snapshot, positions); this.applyPins(); }
       reload(snapshot: GraphSnapshot, report: FreezeReport, positions: F32): void {
           this.sim.load(snapshot, positions);                       // clears the simulation's fixed words and override list when n or the index space changed (7.12)
           this.applyPins();                                          // rebuilt from node.pinned over dm.nodes AFTER the remap (node.index = report.nodeRemap[old], design 4088-4090), never from the stale index-keyed mask
       }
       step(): void {                                                 // called ONCE per frame (see below)
           const r = this.sim.step(this.iterationsPerStep);
           if (r !== undefined && r !== this.pending) { this.pending = r; r.catch((err: unknown) => this.onError(err)); }   // fire-and-forget (D6); a coalesced call returns the same promise; sync CPU sims return void
           this.positionColumn.markDirty();                          // design 14.4 M12; once per frame
       }
       get isSettled(): boolean { return this.sim.settled; }
       pin(n: Node): void { maskSet(this.mask, n.index, true); this.sim.setFixed(this.mask); }
       unpin(n: Node): void { maskSet(this.mask, n.index, false); this.sim.setFixed(this.mask); }
       setNodePosition(n: Node, p: Position): void { this.sim.setPosition(n.index, p.x, p.y, p.z ?? 0); }
       beginDrag(n: Node): void { this.dragBit(n.index, true); }      // temporary fixed bit (7.12); called from NodeBehavior.onDragStart
       endDrag(n: Node, pin: boolean): void { this.dragBit(n.index, pin); }
       getNodePositionInto(index: number, out: F32): void { out[0] = this.positions[3 * index]; out[1] = this.positions[3 * index + 1]; out[2] = this.positions[3 * index + 2]; }
       dispose(): void { this.sim.dispose(); }
       private applyPins(): void { this.mask = maskFromNodes(dm.nodes, (n) => n.pinned); this.sim.setFixed(this.mask); }
   }
   ```

   `onError` routes the error to the element's error channel and stops the
   layout; `E_DEVICE_LOST` is then followed by the app's `setAccelerator(null)`
   (9.5), whose `accelerator-changed` re-creates the engine on the CPU (item
   1). `UpdateManager.updateLayout()` (lines 203-214) keeps its
   `stepMultiplier` loop for engines that are not simulations and calls
   `LayoutManager.step()` ONCE for a `SimulationLayoutEngine`, passing
   `stepMultiplier` as `iterationsPerStep` (a one-line branch): a GPU
   simulation's `step()` returns the same in-flight promise when saturated
   (7.19 item 3), so calling it `stepMultiplier` times would coalesce anyway,
   and a CPU simulation runs `stepMultiplier` iterations synchronously in one
   call. `behavior.layout.gpuMinNodes` (item 7) is evaluated here against
   `snapshot.nodeCount` at engine creation AND on every `load` / `reload`
   (`_setLayoutInternal` runs before the first `load`, so creation alone cannot
   decide): when the count crosses the threshold the engine is re-created on
   the other side, exactly as for an accelerator change.
5. `NodeBehavior` (`onDragStart` / `onDragUpdate` / `onDragEnd`, note 01
   section 4.5): calls `engine.beginDrag(node)` when the engine has it, keeps
   calling `setNodePosition` per pointer move, and `endDrag(node, pinOnDrag)`;
   `context.setRunning(true)` on drag end is unchanged (the simulation reheated
   itself on `setPosition`).
6. `ForceAtlas2LayoutEngine` / `SpringLayoutEngine` (the one-shot
   `SimpleLayoutEngine` subclasses) are REPLACED by `SimulationLayoutEngine`
   registrations under the same type names with the same zod schemas
   (Storybook controls keep working, `stories/Layout.stories.ts` lines 87-137;
   `"spring"` maps to the FR simulation per the 9.3 table); a
   `"spring-electrical"` registration is offered only when
   `graph.accelerator?.springElectrical` exists; `gravity` is loosened to
   `nonnegative()`; `weightPath` becomes live; `scalingFactor` becomes the
   simulation's `scale` option.
7. Config: `behavior.layout.iterationsPerStep` (default = `stepMultiplier`),
   `behavior.layout.maxInFlight` (default 2) and `behavior.layout.gpuMinNodes`
   (default 0: whenever an accelerator is injected) are the product knobs; the
   first two travel as the layout-owned `SimulationOptions` of 9.3 (a typed
   path through `LayoutAccelerator.forceAtlas2(options)`), the third is
   evaluated by `LayoutManager` (item 4). GPU-only tuning (`GpuLayoutTuning`)
   has NO element-level knob in v1: it reaches the element's simulations only
   as the defaults the app gave `createAccelerator(ctx, { layout })` (9.5); no
   `"auto"` acquisition in v1. A later `accelerator: "auto"` element option
   would use the `@mlc-ai/web-llm` isolation pattern (`vite.config.ts:39`
   external, a loader module never imported from the barrel,
   `peerDependenciesMeta` optional; note 02 section 4.3) -- not on the critical
   path (Q-19).
8. Stories: "Layout/ForceAtlas2 (GPU)" and "Layout/Spring (GPU)" in
   graphty-element with a FAKE accelerator for Chromatic (a `LayoutSimulation`
   that moves nodes deterministically) -- the element gains NO dependency on
   the GPU package, not even a devDependency (9.1); the stories that use the
   REAL accelerator behind a `navigator.gpu` check live in the graphty app,
   which already imports `@graphty/webgpu-graph-algorithms/browser` (W2, 9.8),
   one story per GPU algorithm under a `gpu` tag.

9. Public play / pause API: the element exposes `setRunning(running:
   boolean)` (delegating to `Graph.setRunning`, which exists at `Graph.ts`
   line 2052 but is not surfaced by `graphty-element.ts`), so an app can
   pause and resume the layout the way drag end does today; `isRunning()`
   stays as it is. `setRunning(true)` on a settled simulation also calls
   `reheat()` so "play" visibly restarts; `setRunning(false)` stops the
   per-frame `step()` calls and nothing else (7.19: in-flight batches land,
   nothing is submitted, no GPU memory is released, rendering continues).
   No new event is added: `graph-settled` and `isRunning()` cover the
   transitions an app needs.
10. Role columns for layout inputs (D28, Q-30): when the layout config carries
    `nodeMass` / `nodeSize` records or paths, `LayoutManager` resolves them
    ONCE at engine creation with `@graphty/layout`'s `resolveNodeVector` and
    writes the result as a node column with role `mass` / `size`
    (`builder`-side so it survives re-freezes, `replaceRole: true`); the
    element passes the simulation `nodeMass: null`, and both the CPU and the
    GPU simulation find the column by role. The GPU package never sees an id.

E1 element tests: a fake accelerator injected AFTER `setLayout` engages the
GPU simulation on the running layout; the accelerator removed mid-run hands
the layout to the CPU simulation with the positions and pins preserved; pin
node A, remove node B < A, freeze, `reload` -> A is still fixed (also an 11.3
property in the GPU package); a `gpuMinNodes` above the
node count keeps the CPU engine until a `reload` crosses it; `setRunning(false)`
with fake batches in flight lands exactly those batches and submits nothing
until `setRunning(true)`, which reheats a settled simulation.

Settlement, screenshots and label animation (note 01 section 4.3) work
unchanged because `isSettled` is truthful and bounded (7.17).

### 9.5 The graphty app: detection

```ts
// graphty/src/gpu/accelerator.ts (sketch)
import { probeBrowserWebGpu, requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser";   // static import: the app owns its bundle; a code-split import() must complete BEFORE probe is called
import { createAccelerator, calibrateLayout } from "@graphty/webgpu-graph-algorithms";
export async function attachAccelerator(element: GraphtyElement, prefs: { gpu: "auto" | "off" | "required"; exactMaxNodes?: number; calibrate?: boolean }): Promise<void> {
    if (prefs.gpu === "off") { return; }
    const probe = await probeBrowserWebGpu({ rejectSoftware: prefs.gpu === "auto" });
    if (!probe.ok) { if (prefs.gpu === "required") { throw new Error(probe.reason ?? probe.code); } return; }   // "auto": stay on the CPU path; nothing was created
    const ctx = await requestGpuContext({ adapter: probe.adapter ?? undefined, limits: "raise" });   // the PROBED adapter (unused so far): no second requestAdapter() that could return a different one; `?? undefined` narrows the probe's `GPUAdapter | null` to the option's `GPUAdapter | undefined` (`ok` is a plain boolean, not a discriminant)
    const exactMaxNodes = prefs.exactMaxNodes ?? (prefs.calibrate ? (await calibrateLayout(ctx)).suggestedExactMaxNodes : undefined);
    element.setAccelerator(createAccelerator(ctx, { layout: { exactMaxNodes } }));   // accelerator-changed: a layout already running moves onto the GPU (9.4 item 1)
    ctx.lost.then((info) => { element.setAccelerator(null); showToast(`GPU device lost: ${info.message}`); });   // accelerator-changed again: the running layout continues on the CPU; nothing in flight is retried; a new context needs a fresh requestAdapter() (2.2)
}
```

The app surfaces "GPU acceleration: on (NVIDIA lovelace) / off" from
`ctx.caps`; `calibrateLayout` is called at most once, off the critical path,
and its suggestion reaches every simulation the element creates through the
accelerator's defaults (3.3). Node consumers (a CLI, a benchmark, a test of
the element) do the same with `createNodeGpuContext()` from `./node`.

### 9.6 Detection in Node

`@graphty/algorithms` users in Node inject exactly the same way: `const ctx =
await createNodeGpuContext(); const r = await accelerated(createAccelerator(ctx))
.pageRank(s);`. The element never obtains a Dawn `GPU` itself (it must not
depend on the native module, note 02 section 4.3).

### 9.7 Result-shape parity

| GPU method | Returns | Parity check against `indexed.*` (design 16.2, note 02 section 5) | Adapter writes |
| --- | --- | --- | --- |
| `pageRank` | `{ scores: F32, iterations, converged, danglingMass, precision }` | relative error `<= 1e-5` per node after equal iterations; identical `converged`; `iterations` within +-1 (the device records `firstConvergedIteration`, 8.2, so the batch size of 8 does not enter the count) ; top-k rank order | `rank`, `rankPct = rank / maxRank`, graph `iterations` / `converged` / `maxRank` (`PageRankAlgorithm.ts:224-240`) |
| `hits`, `eigenvectorCentrality`, `katzCentrality` | `F32` vectors | same | as today |
| `connectedComponents` | `{ labels: U32, count, groups() }` | partition equality; IDENTICAL labels after first-seen renumbering | `component` per node |
| `breadthFirstSearch` | `{ depth, parent, order, visitedCount, levels, switches }` | `depth` exact; `parent[v]` any vertex with `depth[parent] === depth[v] - 1` and an arc to `v`; `order` grouped by level | as today (visited set, order) |
| `sssp` / `bellmanFord` | `{ dist: F32 (+Inf unreached), predArc: U32, hasNegativeCycle? }`; the dispatcher attaches `pathTo` / `pathEdges` (9.2) | `dist` within `1e-5` relative; `predArc` any arc attaining `dist`; negative-cycle flag exact | `distance`, `isInPath` via the attached `pathTo()` -- the same call on both paths |
| `betweennessCentrality` | `F32(n)` raw, same normalisation convention as `indexed`, `sourcesUsed`, `sigmaOverflow` | `1e-4` relative (f32 accumulation over many sources; DEPARTURE-6); top-k order; exact sources only (sampled BC is compared against sampled CPU BC with the same `sources` list, which A2 adds to the shared option type; Spearman >= 0.9 on a 10k subgraph) | `score`, min-max `scorePct` (`BetweennessCentralityAlgorithm.ts:59-78`) |
| `edgeBetweennessCentrality` | `F32(edgeCount)` via `foldArcs` | same | per edge through `edgeRemap` (design 14.4 lines 4110-4113) |
| `closenessCentrality` | `F32(n)` | `1e-5` (integer distances before division) | as today |
| `allPairsShortestPath` | `{ dist: F32(n*n), n }` | exact unweighted, `1e-5` weighted | KK `dist` input |
| `kCoreDecomposition` | `{ coreness: U32, maxCore }` | exact | as today |
| `minimumSpanningTree` | `{ edges: U32, totalWeight }` | `totalWeight` within `1e-5`; identical edge set on distinct weights | edge flag through `edgeRemap` |
| `labelPropagation` | `{ labels: U32, count, groups() }` | planted-partition recovery (ARI >= 0.9; LPA is tie-nondeterministic on the CPU too) | `communityId` |
| `louvain` | `{ labels, count, groups(), modularity, levels }` | modularity within a band (`>= cpu - 0.02` on karate / planted partitions, never below the CPU's by more than 0.05 on random fixtures) | `communityId`, `groupCount`, `modularity` (`LouvainAlgorithm.ts:166-184`) |
| `forceAtlas2` / `fruchtermanReingold` / `springElectrical` | a `LayoutSimulation` writing the owner's array | no coordinate parity (chaotic); the 11.4 schedule: one-iteration forces, the swing / traction / speed trace, distributional metrics, layout extent | positions read per frame |

Rules: results are index-aligned typed arrays attached by reference
(`nodes.set(name, vec)`) when the element adopts Option B of design 14.4 M7;
nothing keyed by id leaves the GPU package; `Float32Array` scores satisfy
`NumericVector` so no conversion happens in the adapters.

### 9.8 Timeline against the landing order (design 14.6)

| Step | Package | Integration content | Precondition |
| --- | --- | --- | --- |
| W0 (now, this repo) | webgpu-graph-algorithms | standalone development against `@graphty/graph-format` only; structural copies of `AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation` in `src/types/accelerator.ts` (the published contract, D27; "verified at W1"); CPU reference implementations in `test/oracle/<name>.ts` written from design Ports 1-6 and the 7.2 table; two CI lanes (section 12) | F1 done (it is) |
| A2 (first commit) | algorithms | `indexed/accelerator.ts` (9.2) + `accelerated()` dispatcher with the methods whose `indexed.*` port exists + fake-accelerator tests; ADDITIVE, so it can be the first A2 PR and does not wait for all 95 ports (each port PR adds its dispatcher method); `sources` / `k` on `BetweennessCentralityOptions` | A1 merged AND F2 cut (design 13.5 rule 5: no consumer PR with the format in `dependencies` merges before 1.0.0); may be PREPARED on a branch after A1 |
| L1 | layout | `LayoutSimulation`, `LayoutAccelerator`, `SimulationOptions`, `createSimulation` and the 9.3 type table, steppable CPU FA2 / FR with the 7.2 formulas (including the free-node swing / traction sums and the conditional `estimateFactor`), `resolveNodeVector` / `resolveWeights` / `seedPositions`, Chromatic re-baseline | A1 merged AND F2 cut |
| E1 | graphty-element | `accelerator` property with its `LayoutManager` consumer, `snapshot-replaced -> release` list, adapter dispatch through `accelerated()`, `SimulationLayoutEngine` bridge, drag hooks, schema changes, config knobs, fake-accelerator stories (9.4) | A2 first commit + L1 + the design's E1 `DataManager` / position-column refactor (14.4: `getSnapshot()`, `snapshot-replaced`, `dm.undirected(s)`, `engine.load` / `reload`) merged or on the same branch -- none of it exists in `graphty-element/src` today |
| W1 | webgpu-graph-algorithms | move-in; the structural mirrors are DELETED and `src/types/accelerator.ts` switches to `import type` from `@graphty/algorithms` / `@graphty/layout` (optional peers, D27; `test/types/conformance.test-d.ts` is retired with them); `indexed.*` is ADDED as a second oracle next to `test/oracle/` (the independent references stay); `seedPositions` cross-test; two software shards join `ci.yml` and the GPU job gets its own `gpu.yml` (section 12.5); design 10.3 / 14.5 / 14.6 / 16.2 / 16.7 amendments (DEPARTURE-1, -2, -4, -5, -6; DEPARTURE-3 and -7 amend 14.3 in the L1 PR) | A2 complete, L1, E1 |
| W2 (new) | graphty (app) | `attachAccelerator` with `createAccelerator` / `calibrateLayout` wiring (9.5), the "GPU: on / off" indicator, the real-GPU stories under a `gpu` tag in the APP (skipped on Chromatic's software renderer), the `gpuMinNodes` default measured from 7.21 | W1 |
| D1 / 2.0 | -- | no GPU-specific content; the dispatcher's CPU branch calls the promoted top-level functions | |

Versioning: the GPU package is an independent nx project
(`projectsRelationship: independent`, `graphty-monorepo/nx.json`), conventional
commits with scope `webgpu-graph-algorithms`, OIDC trusted publishing like its
siblings (`release.yml` lines 36-52, note 02 section 3). Its peer range on
graph-format is `^1.0.0` from F2 (published as a caret range because the
manifest says `workspace:^`, 2.5); it declares NO peer on algorithms / layout
(types only, dev), so a CONSUMER can combine any versions whose structural
interfaces match -- the type conformance test in W1 is what guards drift, and
a breaking change to an accelerator interface is a `feat!:` in the OWNING CPU
package. Inside the monorepo the devDependency is still a release-graph edge:
`nx.json` sets `updateDependents: "auto"` and nx counts devDependencies, so
every `@graphty/algorithms` or `@graphty/layout` release also patch-bumps and
publishes this package (commit `5ef67039` shows `graphty` bumped as a
dependent of `compact-mantine`). This is the monorepo's existing convention
and is accepted (Q-29 records the `release.groups` / `updateDependents:
"never"` alternative).

### Review notes (section 9)

- Judge integration-feasibility (draft B 13 P4 vs 9.8): the phase table said
  integration PRs run once A1 has merged while 9.8 required F2. Both now say
  "prepared after A1, merged after F2" (also section 13, P6).
- Judge integration-feasibility (draft A 7.18 / 9.2): a `LayoutStepper`
  protocol leaked a non-`LayoutSimulation` method into the element. The bridge
  above uses only `step(): void | Promise<void>`; `beginDrag` / `endDrag` are
  element-side bridge methods, not simulation methods.
- Judge integration-feasibility (draft C 7.9): the bridge's readback copy could
  overwrite a `setPosition` on an unpinned node, and the bridge never caught
  the promise. Both handled (7.12 override list; `.catch` in `step()`).

---------------------------------------------------------------------------

## 10. Performance targets and memory model

Tiers from design 15.3; undirected weighted snapshots; E = 10n (A = 20n) at
every tier, the same graphs as 7.21. Basis codes: [M] measured on the dev box
(probes cited in 7.21 and note 06 section 3.5), [D] design 15.1 arithmetic,
[X] extrapolation from [M] scaled linearly in bytes / arcs / pairs, [P]
published numbers (cosmos, GraphWaGu, Brinkmann; note 03). Every [X] is
replaced by a measurement at the gate named in 10.4 and recorded in
`benchmarks/results/`.

### 10.1 Device memory (resident, excluding the staging ring)

| Nodes / edges | Core hot prefix [D] | + views typical (`edgeList` or `coo`, `degreeOrder`) [D] | FA2 exact scratch [D] | FA2 grid scratch at batch peak (65 B/node resident + 16 B/node leased, 7.3) + pyramid (2D / 3D) [D] | PageRank scratch [D] | BFS scratch [D] | Betweenness batch (k = 64 sources) [D] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 1.6 MB | +0.8 MB | 0.5 MB | 0.8 MB + 5.6 MB / 38 MB | 0.1 MB | 0.2 MB + 0.8 MB edge queue (A entries) | 7.7 MB |
| 100k / 1M | 16.4 MB | +8.4 MB | 5.3 MB | 8.1 MB + 5.6 MB / 38 MB | 1.2 MB | 1.6 MB + 8 MB edge queue | 77 MB |
| 1M / 10M | 164 MB | +84 MB | 53 MB | 81 MB + 5.6 MB / 38 MB | 12 MB | 16 MB + 80 MB edge queue (A entries: no overflow) | 768 MB (planner caps k by `maxBufferSize`) |
| 10M / 100M | 1.64 GB (windowed at defaults; per-array on raised limits) | +840 MB | 530 MB | 810 MB + 5.6 MB / 38 MB | 120 MB | 160 MB + a chunked edge queue at `maxBufferSize` (6 row 7) | k capped to ~6 on a 12 GB card |

Limit consequences: the 12 GB RTX 4070 SUPER holds every tier. The 1M / 10M
core FITS lavapipe's limits (hot prefix 164,000,256 B under the 256 MiB
`maxBufferSize`; 80,000,000 B segments under the 128 MiB binding limit that
lavapipe cannot raise, note 06 section 3.5) -- it is excluded from the default
CI lane by TIME (lavapipe is ~350x slower on the exact tile and 10-50x on
gathers), not by limits; the 10M / 100M tier needs raised limits or windows and
is GPU-lane only. The pyramid at 10k nodes in 2D is 4^2 .. 256^2 (1.4 MB), not
the 512^2 cap; the table shows the cap for the row's worst case.

### 10.2 Upload and readback

| Nodes / edges | Upload hot prefix (5-12 GB/s `writeBuffer`) [X] | Freeze on the CPU for comparison [M, design 15.4] | Positions readback per batch (Chromium 2.65 ms/MiB [M]; Node ~10x less [X]) | Result readback (one `F32(n)`) |
| --- | --- | --- | --- | --- |
| 10k / 100k | ~0.3 ms | ~2 ms | 0.3 / 0.05 ms | < 0.2 ms |
| 100k / 1M | 2-4 ms | 22 ms directed / 48 ms undirected | 3 / 0.4 ms | 1 / 0.1 ms |
| 1M / 10M | 20-40 ms | ~250-500 ms [X] | 30 / 3 ms | 10 / 1 ms |
| 10M / 100M | 200-400 ms | seconds | 300 / 30 ms | 100 / 10 ms |

### 10.3 Per-iteration / per-run time on the RTX 4070 SUPER class

| Nodes / edges | FA2 exact / iter | FA2 grid 2D / iter | PageRank / iter (bandwidth-bound: ~24 B/arc) [X from A] | PageRank x 100 iterations (= 100 x the per-iteration column; the convergence block is read every 8) | BFS (whole run, 32 levels/submit; basis: T-10's Merrill 3.3 GTEPS [P] and the bandwidth bound, at 10-30% of peak under atomics and dedupe) [X] | WCC (Afforest, ~6 rounds) [X] | Betweenness, 256 sampled sources (the 8.4 per-source model) [X] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 0.6-0.7 ms [M] | 1-2 ms (build-dominated) | 0.05 ms + ~0.02 ms/dispatch latency | ~10 ms | ~1-2 ms | ~1-2 ms | 0.1-0.5 s |
| 100k / 1M | ~18 ms [M] | 4-10 ms [P-bracketed] | 0.2-1.4 ms [M-scaled] | 20-140 ms | 2-5 ms (diameter ~10-20) | ~5-15 ms | 0.5-2 s |
| 1M / 10M | ~1.8 s [X from the 7.6 curve] | 35-95 ms [P-bracketed; conservative, 7.7] | 2-14 ms | 0.2-1.4 s | 10-30 ms | 30-100 ms | 2-20 s |
| 10M / 100M | n/a | 0.35-0.95 s | 20-140 ms | 2-14 s | 0.1-0.3 s | 0.3-1 s | not offered (sampled k <= 16) |
| integrated GPU, 100k / 1M | 0.2-0.35 s (10-17x) | 20-100 ms (5-10x) | 1-14 ms | 0.1-1.4 s | 10-50 ms | 30-150 ms | 3-20 s |
| lavapipe, 4 threads, 100k / 1M (correctness only) | ~20 s | ~1-2 s | 3.3-5.5 ms [M] | 0.3-0.5 s | 0.3-1 s | 0.3-1 s | not run |

Integrated GPU (Iris Xe / Apple M-class base) expectations, derived from the
public FP32 peaks and bandwidths in 7.21 [X]: 10-17x slower on the
compute-bound exact tile, 5-10x on gathers and the grid, readback roughly
equal (unified memory); the exact-tier crossover halves to ~8k nodes; the 100k
FA2 grid tier lands at 20-100 ms/iter (8-30 fps) and is still usable
interactively with `iterationsPerStep = 1`. Speedup claims versus the CPU packages are NOT made until the harness
produces them (design line 76 measures `toCSRGraph` at 2,376 ms versus freeze
20 ms at 100k / 1M; at that tier the win is dominated by conversion cost
already removed by the format, at 1M+ by the kernels).

### 10.4 Acceptance targets bound to phases (graft: C 10.3)

Each target is MEASURED at the named gate on the 4070 under Dawn-in-Node
unless "Chromium" is stated, recorded in the checked-in baseline
`benchmarks/results/<runner-class>.json` (one file per runner class =
`<vendor>-<architecture>-driver<major>`, e.g. `nvidia-lovelace-driver580`;
the harness writes each run to `benchmarks/out/<runner-class>.json` and
`bench:compare` compares the two, 11.7) with the adapter's vendor /
architecture / device, driver and requested limits, and the README table is
regenerated from the baseline file. A target that is missed does not close
its phase; the owner either re-fixes the target in this table (a recorded
decision in the PR) or the phase continues -- a target is never relaxed
silently. The same rule covers every tolerance and fixture number of 11.4:
an unmeetable parity number is re-fixed by a recorded owner decision, never
loosened in the test file.

| Id | Target | Basis / expectation | Gate |
| --- | --- | --- | --- |
| T-1 | Upload of the 100k / 1M weighted hot prefix (16.4 MB) <= 10 ms; 1M / 10M (164 MB) <= 100 ms | [X] 5-12 GB/s | G1 (P1) |
| T-2 | `degree` kernel + 400 KB readback at 100k <= 2 ms wall in Node | [M] 1M-element kernel submit 0.40 ms + `onSubmittedWorkDone` 0.17 ms | G1 |
| T-3 | Empty submit + 4-byte `mapAsync` round trip <= 0.1 ms Dawn, <= 0.3 ms Chromium | [M] 0.04 / 0.10 ms | G1 |
| T-4 | FA2 exact: 10k nodes <= 1 ms per iteration; 16k <= 2 ms; the curve at the exact ladder 1k / 4k / 8k / 16k / 32k / 65k (the one ladder 7.8, 11.7 and P3 share; `calibrateLayout` walks its 8k-65k subset by default; the grid ladder is 32k / 65k / 100k / 262k / 1M) recorded and `exactMaxNodes` re-fixed by the 7.8 rule | [M] 7.6 curve: 0.53 ms at 8k, 1.13 ms at 16k with the FA2 body | G3 (P3) |
| T-5 | FA2 per-frame cost in Chromium (`step(1)` + `12n` readback): 10k nodes <= 6 ms; 100k nodes (grid) <= 12 ms; measured by the `bench`-tagged browser test of 11.6 (skipped unless `GRAPHTY_BROWSER_GPU=nvidia`) at 10k and 100k, which writes `performance.now()` deltas into the results file through the Vitest `server.commands` bridge -- the 500-node smoke cannot produce these numbers | [M] Chromium 1 MiB readback 2.65 ms | G3 (10k), G4 (100k) |
| T-6 | FA2 grid: 100k / 1M <= 10 ms per iteration (2D); 1M / 10M <= 100 ms (2D); 100k 3D <= 20 ms | [P] cosmos 6.6 ms at 100k; GraphWaGu 160 ms at 1.13M (laptop, tree) | G4 (P4) |
| T-7 | Attraction gather at 1M / 10M (20M arcs) <= 15 ms per iteration | [M] 0.1-0.7 ms at 1M arcs | G4 |
| T-8 | PageRank 100k / 1M, 100 iterations, convergence check every 8: <= 150 ms wall end to end including upload; 1M / 10M <= 1.5 s (the upper brackets of the 10.3 column) | [M] gather 0.1-0.7 ms per iteration at 1M arcs | G7 (P7) |
| T-9 | WCC 1M / 10M <= 100 ms | [X] ~6 rounds of O(m) u32 CAS | G7 |
| T-10 | BFS 1M / 10M RMAT (diameter ~10) <= 100 ms; 1000 x 1000 grid (~2,000 levels) <= 1.5 s with `mapAsync` calls <= levels / 32 + 1 | [P] Merrill 3.3 GTEPS (2011); [M] Chromium round trip 0.10 ms | G8 (P8) |
| T-11 | Betweenness, 100k / 1M, 256 sampled sources <= 5 s; karate exact <= 20 ms | per source = one BFS + one backward sweep (the 8.4 model: 0.5-2 s expected) | G9 (P9) |
| T-12 | Default CI lane (lavapipe + SwiftShader) <= 15 min; GPU lane <= 20 min (the ONE lane budget; G2 and 12.6 cite these numbers); the 1M 200-iteration exact-vs-grid run of 11.4 is excluded from the lane and runs in the nightly benchmark job | wgpu budgets 5-15 min for its lavapipe job (note 06 section 3.5) | G1 onward |
| T-13 | Benchmark regression: any tracked median (of 5 runs) > 3x its checked-in baseline for the runner class fails the GPU lane's `bench:compare` step (design 15.5 policy); `bench:compare` SKIPS (does not fail) when `gpu-report.json` shows GPU utilisation > 10% or memory in use by other processes during its 10-second sample (`nvidia-smi --query-gpu=utilization.gpu,memory.used`), because the runner is the owner's interactive dev GPU, and a nightly tracking issue is opened only after two consecutive failures (12.6) | -- | G3 onward |
| T-14 | FR 10k / 100k nodes per-iteration numbers recorded | -- | G5 (P5) |
| T-15 | Louvain 1M / 10M and 100k / 1M end-to-end recorded with the CPU comparison (expected 2-10x) | [P] nu-Louvain | G11 (P11) |

### Review notes (section 10)

- Judge verifiability (draft B 10.3): "Targets" was prose and the P3 gate had a
  soft "or a written analysis" clause. Replaced by the T-table and the
  owner-decision rule above.
- Judges on the lavapipe / 1M-tier sentence: see section 4 Review notes.

---------------------------------------------------------------------------

## 11. Testing strategy

### 11.1 Projects (vitest 3.2.x, one config, three projects; benchmarks are a tsx harness)

| Project | Environment | Contents | Default lane | GPU lane |
| --- | --- | --- | --- | --- |
| `node` (PRIMARY) | Node 22, `pool: "forks"` (native addon; verified by graph-format's 1309-test suite, `packages/graph-format/vitest.config.ts` line 7), Dawn via `test/setup/gpu.ts` | every unit, kernel, primitive, algorithm, layout and planner test; differential tests; property tests; device-loss; `include` covers `test/{device,memory,kernel,primitives,algorithms,layouts,types}/**` and the root tests and EXCLUDES `test/limits/**` and `test/browser/**`, so `--project=node` IS the whole node suite and carries the 80 / 80 / 75 / 80 thresholds (11.8) | yes (lavapipe, `GRAPHTY_GPU_ADAPTER=llvmpipe`, `GRAPHTY_GPU_REQUIRE=any`) | yes (NVIDIA, `GRAPHTY_GPU_REQUIRE=nvidia`) |
| `node-limits` | same | tests that need limits or time above lavapipe's (bindings > 128 MiB, `maxBufferSize` near 2 GiB, a real 2D dispatch above 16,776,960 items on 100M elements, vendor feature assertions, the OOM scope on a real over-allocation, the 262k and 1M layout fixtures of 11.4) | no (selected out by `--project`) | yes |
| `browser` (LIGHT) | Playwright Chromium, `fileParallelism: false`, flags by `GRAPHTY_BROWSER_GPU` (`swiftshader` / `nvidia`); `test.env` forwards `GRAPHTY_GPU_REQUIRE` and `GRAPHTY_BROWSER_GPU` at config-evaluation time and `test/setup/browser.ts` reads them from `import.meta.env` (2.3); a Vitest `server.commands` entry lets a browser test append a result record to the benchmark output file | 11.6 | yes (SwiftShader) | yes (NVIDIA, four flags + `libEGL.so.1`) |

Benchmarks are NOT a vitest project: `pnpm run bench` runs `tsx
benchmarks/run.ts` (graph-format's harness with an async `run` body, 11.7),
because `vitest bench` emits its own JSON shape while `bench-compare.js`, the
T-table and the README generator all read the harness's `BenchResult`
sessions; one mechanism, one file shape. The config is the note 07 section
4.5 sketch plus the `node-limits` project and the flag switch of note 06
section 5. The per-instance Playwright `launch`
spelling is confirmed against the installed `@vitest/browser` at P0 and
recorded in the package `CLAUDE.md` (note 07 unverified item 1); the flag sets
live in one exported constant so the Vitest 4 provider change is a one-line
move (note 05 section 9.3).

### 11.2 Device policy in tests (`test/setup/gpu.ts`)

Generalises `packages/graph-format/test/audit/gpu-upload.test.ts` `acquire()`
(lines 38-63): dynamic `import("webgpu")`, install globals, `create([...])`
with `adapter=${GRAPHTY_GPU_ADAPTER}` when set and
`enable-dawn-features=${GRAPHTY_DAWN_FEATURES}` when set, then a FRESH
`requestAdapter()` for every device it creates (an adapter is consumed by one
`requestDevice`, 2.2 step 1, so the harness never hands a used adapter to
`GpuContext.create({ adapter })`; graph-format's `acquire()` returns `{ gpu,
adapter, device }` and must not be generalised into a cached adapter),
`adapter.info`, `requestDevice` with raised limits; each failure is an
`E_NO_ADAPTER: ...` reason. `acquire({ subgroups: false })` creates a context
with `optionalFeatures: []` for the twin tests (D16); `GRAPHTY_GPU_NO_SUBGROUPS=1`
makes that the default for a whole run. The ONE policy variable (D19), parsed
by `scripts/gpu-policy.js` (2.3):

| `GRAPHTY_GPU_REQUIRE` | Meaning |
| --- | --- |
| unset | no adapter -> `t.skip(reason)` with the printed `E_NO_ADAPTER` reason (local convenience only) |
| `any` | an adapter must exist (lavapipe / SwiftShader count); none -> hard failure. The DEFAULT lane sets this so a missing Vulkan ICD on the runner is red, never a silent skip (graft: C 12.2) |
| `hardware` | additionally `!isSoftwareAdapter(info)` |
| `nvidia` (or any vendor string) | additionally `adapter.info.vendor === value` and, in the browser, `isFallbackAdapter === false`. The GPU lane sets `nvidia` so a silent lavapipe / SwiftShader run is red (the failure mode `HEADLESS_GPU_REPORT.md` found locally) |

"A wrong result is never a skip." An `uncapturederror` listener fails the
current test. `gpuScale()` returns 1 on hardware and 1/50 on a software adapter
and scales fixture sizes and iteration counts (lavapipe is ~350x slower on the
exact tile, note 05 section 2.5). `XDG_RUNTIME_DIR` is set to silence Mesa's
stderr lines. The Node setup drops the `GPU` reference in `afterAll` so forks
exit. Every fixture is deterministic (seeded LCG / xorshift from the
graph-format harness, `packages/graph-format/benchmarks/datasets.ts`).

### 11.3 Test kinds

| Kind | What | Oracle |
| --- | --- | --- |
| Planner unit tests (no device) | `planUpload` (four caps tables x {arena, no arena} x {fits, exceeds buffer, exceeds binding}; a row longer than a window split on arc ranges; `start % 64 === 0` computed with `%`, including a case above 2^31 arcs as a pure computation), `plan1d` / `planGridStride` (the `(16,776,960, 16,777,216]` boundary), `UniformBlock` offsets, pipeline cache keys, `BufferPool` size classes | hand-computed expectations under faked caps: spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like (`test/helpers/caps-tables.ts`, note 05 section 4) |
| Upload contract | arena hot prefix bindings equal CPU views; per-array path on a `fromCsr` snapshot (`arena === null`) and on `transpose()`; windowed path on a 64-arc boundary; windowed EXECUTION of `degree` with a FAKED `maxStorageBufferBindingSize = 1 MiB` on the 100k / 1M graph (>= 8 windows) and with a hub row longer than a window (graft: C G2); `arena.byteOffset !== 0` from `fromBytes`; packed `u8` / `bool` columns; identity permutations never materialised (`byteLength({ views: true })` unchanged); `release` destroys every buffer (`stats().buffers === 0`, `pool.liveBytes === 0`, no uncaptured error); `E_RELEASED` on the next bind | copied from `gpu-upload.test.ts` lines 254-666 (note 07 section 6) |
| Primitive differential | each primitive vs `test/oracle/<name>.ts` on random and adversarial inputs (sizes 0, 1, 255, 256, 257, 4097; all-equal keys; one giant row; empty; exactly 16,776,960 and +1 items in `node-limits`) | f64 CPU references; exact for u32 |
| Algorithm differential | vs `test/oracle/<name>.ts` (independent index-based references written from design Ports 1-6: FIFO BFS, heap Dijkstra, union-find, Brandes, tens of lines each) always, PLUS `indexed.*` as a second oracle from W1 (the independent references are kept: a GPU tested only against the CPU package shares its design and its bugs); on the empty graph, one node, one self-loop, karate, grids, paths (high diameter), stars (hub), complete graphs, seeded G(n, m) with self-loops and parallels, planted partitions, a 10k-degree hub graph, directed and undirected, weighted (incl. zero weights) and not; `validate({ checksum: true })` after every call (no write into a view) | tolerances of 9.7 |
| Layout parity | 11.4 | CPU FA2 / FR oracle (post-L1 the real `ForceAtlas2Simulation`) |
| Exact-vs-approximate | 11.4 | the exact kernel (7.6) |
| Property / invariant (fast-check, `numRuns` 200 default, 1,000 nightly) | fixed nodes never move (random `setFixed` between steps, INCLUDING the all-fixed mask, which must report `settled` within `settleWindow` steps); `dim === 2` writes `z === center.z` on every readback whatever `z` was uploaded; `setPosition` visible in the next readback and never clobbered by an older batch (a batch submitted before the write completes after it); `settled` within `maxIter`; reheat on unpin / `setPosition` / `load`, not on pin, and `speed` NOT reset by `setPosition`; pin node A, remove node B < A, `load(next)` with the remapped array and a re-issued mask -> A is still fixed; results are `<ArrayBuffer>`-typed of exact length; `dest` returned when supplied; `arcCount === 0` binds nothing; `residency.stats().snapshots === 1` after freeze + layout + `snapshot-replaced` on a DIRECTED source (the undirected copy was released, 4.5); PageRank sums to 1 within 1e-5; BFS `depth[parent[v]] === depth[v] - 1` and `order` non-decreasing in depth; SSSP `dist[v] <= dist[u] + w` over every arc; WCC labels dense and endpoints share a label; BC sums equal the analytic total on a path and a star; FR displacement never exceeds `t`; FA2 per-node displacement never exceeds `speed * \|F\| / (1 + sqrt(speed * swing_i))` | generators sized by `gpuScale()` |
| Subgroup variants (a `variants` axis of every kernel test that has a twin) | every kernel with a twin runs against BOTH variants in the same process -- the feature context and a second context from `acquire({ subgroups: false })` -- on every adapter; u32 results bitwise identical between twins; f32 reductions and the FA2 swing / traction trace within `1e-6` relative between twins on one device (the subgroup tree and the workgroup tree sum in different orders); the sizes 4 / 8 / 32 come from the three adapters across the lanes; a faked-caps `min != max` table asserts no planner output depends on `subgroupMinSize` | the twin |
| Device loss / errors / leaks | `device.destroy()` mid-batch rejects pending promises with `E_DEVICE_LOST` and leaves the context in `lost`; a new context from a FRESH adapter + `load()` works afterwards, and `create({ adapter })` on the consumed adapter throws `E_NO_DEVICE { reason: "consumed" }`; an OOM scope on a deliberately oversized buffer yields `E_OUT_OF_MEMORY` (`node-limits`); a bad binding yields `E_VALIDATION` with the label, and under Dawn-node it rejects the SAME batch's readback (5.7); `E_NOT_LOADED`, `E_INVALID_ARGUMENT`, `E_UNSUPPORTED`, `E_TOO_LARGE`, `E_ABORTED`, `E_SOFTWARE_ONLY`, `E_SHADER_COMPILE { stage: "compose" }` each reachable; the `state` block round-trips host -> kernel -> host through its generated descriptor (5.3); a counting proxy around `createBuffer` / `destroy` / `mapAsync` (`test/helpers/leak-counter.ts`) asserts 0 live buffers after `dispose()` and bounds the `mapAsync` count per batch (graft: C 11.7) | both runtimes |
| WGSL compile matrix | every module of `kernels.ts` in every combination of `test/helpers/override-matrix.ts` (5.1: bounded, and asserted to cover every pipeline key the node suite creates) compiles on Dawn (`backend=null`) AND on Chromium in the browser project; a misaligned hand-written uniform struct is rejected on Chromium (graft: C R-10 / G8); a `.wgsl.ts` body containing `@group(` or `override ` fails the unit test of 3.5 | compilation info |
| Type-level | `expectTypeOf` for the public surface; `tsconfig.strict-consumer.json` compiles a consumer sample with `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` ON (after `build:all`, 3.1); from W1 `test/types/conformance.test-d.ts` asserts mutual assignability with the real `AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation` (D27) | tsc |
| Build output | `dist/webgpu-graph-algorithms.js` and `dist/browser.js` import no `webgpu` specifier and `dist/node.js` imports it only dynamically (the specifier-form test of 2.5); HARD failure when the bundle is absent under `CI`; `package.json` exports match `scripts/entries.js`; `sideEffects: false`; the barrel export list pinned | file reads |
| Bind-group budget | every bind-group-layout descriptor created by the package (enumerated through `kernels.ts`) has <= 8 storage entries per stage; the PageRank pull kernel has exactly the 8 named in 8.2; every kernel of 8.10 has the count 8.10 states | descriptor inspection |

### 11.4 Layout parity, exact-vs-approximate, and invariants (the FA2 acceptance tests; graft: C 11.4-11.6)

- Same start: both sides seed NaN rows with the same LCG in index order
  (`seedPositions` into a `Float32Array`), and the f64 oracle reads that same
  f32-valued array, so iteration 0 positions are bit-identical by
  construction.
- Oracle independence (G3): the FA2 oracle `test/oracle/forceatlas2.ts` in
  `compat: "networkx"` with settling disabled reproduces NetworkX
  `forceatlas2_layout` (pinned NetworkX version; fixtures generated ONCE by
  `test/fixtures/networkx/generate.py` and committed as JSON with the version
  and the command in their header, so the suite needs no Python) on karate, a
  10 x 10 grid, a star of 200 and a seeded G(200, 600), passing the SAME
  initial positions through NetworkX's `pos` argument (bypasses the LCG
  question), with `linlog`, `distributed_action`, `strong_gravity`, weights
  and `gravity = 0` variants, within `1e-9` for `max_iter` 1 and 5 and `1e-6`
  for `max_iter` 50 after rescaling on both sides (NetworkX's `1e-10` early
  exit disabled by construction at those counts; a chaotic controller
  amplifies f64 ordering noise, so `1e-9` at 50 would be over-tight; the
  `max(d, 0.01)` floor is kept in both modes and the fixtures avoid pairs
  under it). Because NetworkX shares every FORCE law with `compat: "paper"`,
  the same fixtures verify the paper mode's forces at iteration 0; the
  paper-mode CONTROLLER (force-based swing / traction, fresh sums over free
  nodes) differs from NetworkX's by the two `SWING_MODE` lines, which a unit
  test pins against hand-computed values on a three-node graph. The WGSL is
  then checked against an oracle that was itself checked against an
  independent, widely used implementation, not against a second
  transcription of the same table by the same author.
- Force parity (one iteration): run K2 + K3 on the GPU, read `force` back,
  compare with the f64 oracle's per-node force on karate, a 10 x 10 grid, a
  star of 200 and random 1,000-node graphs: `|F_gpu(i) - F_cpu(i)| <= 1e-4 *
  max(|F_cpu(i)|, 1e-3 * max_j |F_cpu(j)|)` (f32 tile summation order;
  DEPARTURE-6), with and without weights, linlog, distributedAction,
  strongGravity, `gravity = 0`, `nodeMass` vector, 2D and 3D, `compat:
  "paper"` and `"networkx"`, with a pinned node (its force is computed, its
  swing excluded).
- Trace parity: over 50 iterations compare the per-iteration `{ swing,
  traction, speed, speedEfficiency }` trace (read from `state` after each
  `step(1)`) with the oracle's trace: relative error `<= 1e-4` for the first 10
  iterations against an f32 ORACLE (the same TypeScript run with
  `Float32Array` scratch, summing in tile order: `swing_i` is a difference of
  nearly equal forces, so f32 force noise of ~1e-6 becomes ~1e-4 in swing and
  an f64 reference cannot be held to 1e-4), and `<= 5e-2` through iteration 50
  against the f64 oracle on 10-1,000-node graphs (chaotic divergence beyond is
  expected). An unmeetable number here is re-fixed by the owner-decision rule
  of 10.4, never loosened in the test.
- Distributional parity (same seed, 100 iterations): stress, edge-length
  distribution quantiles, per-node nearest-neighbour distance histogram and
  inter-component separation within 10% between GPU and CPU. Coordinates are
  never compared.
- Existing behaviour pins from `layout/test/forceatlas2-layout.test.ts` (note
  01 section 2.1.8) re-expressed on the GPU: empty graph, single node,
  disconnected components separated by > 0.03 after 100 iterations, `maxIter`
  respected, `completeGraph(6)` spread > 0.3, same seed -> same layout (bitwise
  on the same device), different seeds -> different.
- Exact-vs-grid: fixtures = uniform random, clumpy (Gaussian mixtures with 10 /
  100 / 1,000 clusters), a scale-free graph with a 10k-degree hub, cosmos's two
  documented failure cases (a 163-node country-graph shape and 1,024 points in
  one finest cell), a line (all points colinear), coincident points (which
  exercises the `cellSize` floor of 7.7), and the ISOLATED-NODE fixture: a
  giant component plus 1% isolated nodes plus 100 small components, taken
  after 200 exact iterations so the strays sit at their `k M / g` equilibrium
  (7.7) -- the normal shape of real data, not a pathology. Sizes 20k and 100k
  on both lanes (scaled by `gpuScale()` on software adapters), 262k
  (finest-grid saturation, which a 1/50-scaled fixture cannot reach) and 1M in
  `node-limits` on the GPU lane only, plus a software-adapter saturation case
  with `gridMax2D = 32` so saturation occurs at ~1k nodes on the default lane.
  Force-field error RMS over nodes `<= 5%` and 99th percentile `<= 25%` on
  uniform, clumpy AND isolated-node fixtures, with the denominator of the
  force-parity test above (`max(|F_exact(i)|, 1e-3 * max_j |F_exact(j)|)`,
  never the raw `|F_exact(i)|`, which is dominated by interior nodes whose
  exact force nearly cancels); an isolated node's total force equals gravity
  alone (the degree-0 row of 6 row 3 written); on the hub-cell fixture the
  Horvitz-Thompson path is checked for unbiasedness (mean over 32 seeded
  iterations within 5% of exact); from the same LCG start, the grid tier's
  bbox extent after 50 and 200 iterations is within 25% of the exact tier's
  on the 20k and 100k fixtures (expansion parity: the test that a
  displacement clamp cannot pass, D25); from the same start, 200 iterations
  exact vs grid agree on the distributional metrics within 15% at 20k, 100k
  and 262k; at 1M only the one-iteration force-field comparison and the
  32-iteration unbiasedness check run in the lane (one exact iteration is
  ~1.8 s, so six fixtures x 200 exact iterations would be ~36 minutes inside
  a 20-minute lane), and the 1M 200-iteration comparison runs in the nightly
  benchmark job; two runs are bitwise identical with `deterministic: true`;
  3D at 27 / 216 loops with the pyramid asserted <= 40 MB; the settle test
  runs on the isolated-node fixture and asserts the core is still moving when
  `settled` would have fired under a bbox-radius normaliser (7.17).
- Force-sum invariant (the corrected form of draft C's "momentum" test): with
  `gravity = 0` and `distributedAction = false`, after one iteration
  `|sum_i F_i| <= 1e-4 * sum_i |F_i|` on every fixture (Newton's third law over
  the doubled arcs and the antisymmetric pair force). A centroid-drift test is
  NOT used: per-node speed factors `speed / (1 + sqrt(speed * swing_i))` make
  displacements non-antisymmetric even when forces are, so centroid drift is
  expected.
- Frame-loop test (`test/helpers/frame-loop.ts`): a synchronous 600-tick loop
  calls `step()` through the same bridge logic as the element without awaiting;
  asserts at most `maxInFlight` submissions in flight, monotone iteration
  counts, that a `setPosition` during flight lands in the following batch and
  is never overwritten by an older one, and that `settled` is reported; also
  run in the browser project on SwiftShader and NVIDIA.

### 11.5 What the walking skeleton proves (P1 gate)

One Node test and one browser test that: create a context through `./node` /
`./browser`; upload the hot prefix of a `fromEdgeArrays` snapshot (arena path)
AND a `fromCsr` snapshot (per-array path); run the `degree` kernel with the
`USE_PERM` dummy-binding pattern; read back a `Uint32Array` equal to
`outDegree()` on every fixture including `arcCount === 0` (nothing bound,
dispatch skipped) and a 17M-item synthetic map (2D dispatch, lavapipe-sized);
run one exact-tile FA2 iteration on karate and read the swing / traction trace
from the state block; `release(snapshot)` then `stats().buffers === 0` and the
leak counter at 0; no uncaptured error; the `uncapturederror` hook fails a
deliberately broken bind group; `plan1d(16_776_960)` is 1D and `+1` is 2D;
`device.destroy()` mid-readback gives `E_DEVICE_LOST`; the staging ring reuses
slots without validation errors under 100 back-to-back submissions; the `u32`
results (`degree`, the 17M-item map) are BITWISE identical on lavapipe,
SwiftShader and NVIDIA (graft: A 11.4), while the FA2 iteration's swing /
traction / per-node force agree within `1e-5` relative (`1e-6` absolute
floor) across the three adapters and bitwise only between two runs on the
SAME adapter -- the exact tile is not bit-identical across adapters (7.16;
measured: lavapipe `f0.z` 317.65924072265625 vs NVIDIA 317.6592102050781,
`tmp/webgpu-plan/review/probes/verify-checksum-cross-adapter.mjs`); T-1 /
T-2 / T-3 recorded; the noise-floor file of 11.9 item 3 created from the
`degree` and one-iteration FA2 kernels across the three adapters, and the
first sabotage mutations (skipped last workgroup, swapped `USE_PERM`
select, ignored rebase) shown to fail their tests (11.9 item 1).

### 11.6 Light browser testing

`test/browser/*.test.ts`: (1) `requestGpuContext()` and the typed `E_NO_WEBGPU`
when `navigator.gpu` is absent (simulated by passing `undefined` to `probe`);
`probe()` reports `software` correctly and the vendor assertion holds under
`GRAPHTY_GPU_REQUIRE=nvidia`; (2) the skeleton of 11.5; (3) `createForceAtlas2`
on a 500-node graph: `load`, `step(10)` five times with `maxInFlight = 2`,
positions written back, `setPosition` / `setFixed` honoured, `dispose` clean,
plus the 11.4 frame-loop test; (4) one PageRank, one BFS and one CC on karate
vs the oracle; (5) a subgroup-variant kernel when `features.has("subgroups")`
(SwiftShader size 4); (6) the WGSL compile matrix of 11.3 (the bounded
`override-matrix.ts` table, so SwiftShader's slow JIT pipeline creation stays
inside the "light" budget); (7) `release()` leaves no buffers; (8) a
`bench`-tagged test, skipped unless `GRAPHTY_BROWSER_GPU=nvidia`, that steps a
10k-node exact-tier and a 100k-node grid-tier FA2 with `step(1)` + readback
and appends the `performance.now()` deltas to the benchmark output through
the `server.commands` bridge -- the only vehicle for T-5. Passes on
SwiftShader (default lane) and NVIDIA (GPU lane); Chromium only, no Firefox
or WebKit instance until those ship the features on Linux CI (Q-22). The `browser.close()` hang
after GPU work on the NVIDIA path (note 06 section 7, R-13) is handled by
running the project with `--reporter=default --reporter=json
--outputFile=browser-results.json` under `timeout -k 10 600` and treating exit
124 as a PASS iff the JSON has `numTotalTests > 0 && numFailedTests === 0`
(GNU `timeout` exits 124 whenever the limit hits, so a run whose tests all
passed and whose `browser.close()` hung would otherwise be red; `timeout`
signals the process group and Playwright SIGKILLs the browser's group on
SIGTERM, so nothing is orphaned); both lanes use the same rule (12.3). Nothing
larger: property tests, large fixtures, planners, windowed uploads, indirect
loops and device loss live in `node`.

### 11.7 Benchmarks and baselines

`benchmarks/` copies graph-format's harness (`bench()`, `printTable`,
`appendSession` with a `gpu` field `{ vendor, architecture, device, driver,
limits }`; note 07 section 6) with ONE change: `bench()`'s `run` body is
`async` and the timer brackets `await run(input); await
device.queue.onSubmittedWorkDone()` (graph-format's `run: (input) => unknown`,
`harness.ts` line 52, is synchronous and cannot time GPU work), and
`datasets.ts` (seeded `randomEdges`, an R-MAT-like hub graph, `gridEdges`,
`KARATE_EDGES`). Groups: `upload`, `roundtrip`, `primitives`, `layout-exact`
(the exact ladder of T-4), `layout-grid` (the grid ladder 32k / 65k / 100k /
262k / 1M, 2D / 3D, per-frame cost with readback), `layout-fr`, `pagerank`,
`bfs`, `wcc`, `betweenness`. Each run writes `benchmarks/out/<runner-class>.json`
(gitignored); the checked-in baseline is `benchmarks/results/<runner-class>.json`
with the runner class defined in 10.4; sessions record adapter vendor /
architecture / device, driver and requested limits so 4070, CI-runner and
browser numbers never mix. The GPU lane uploads the output JSON (90-day
retention) and `bench:compare` fails on `> 3x` the checked-in baseline for its
runner class (T-13), with the quiet-GPU skip rule of T-13 (utilisation sampled
by `gpu-report.js` over 10 s; medians of 5 runs). Software adapters never time
anything. Browser numbers (T-3, T-5) come from the browser project's
`bench`-tagged tests appending records to the same output file through the
Vitest `server.commands` bridge (a browser test cannot write files itself).

### 11.8 Coverage

Thresholds 80 / 80 / 75 / 80 apply whenever the selected project set is
EXACTLY `node` (`process.argv` contains `--project=node` and no other
`--project`), which is what every CI invocation runs (12.3, 12.5), and are
disabled only when `COVERAGE_DIR` is set (a merge run) or another project is
selected (`browser`, `node-limits`). This deliberately departs from the
`algorithms/vitest.config.ts` pattern (lines 62-72), which sets `thresholds:
undefined` whenever ANY `--project=` is on the command line: under that
pattern no CI command ever evaluates the number G1 gates on, and the
monorepo's `tools/merge-coverage.sh` enforces no threshold after merging
either (its "checked at CI level after merging" comment is not true today).
`src/wgsl/**` excluded (template strings); device-limit branches (windowed
uploads, 2D dispatch, OOM) are covered by the faked-caps unit tests so the
default lane reaches the thresholds without the GPU lane (note 06 section 9
item 6). Coverage is produced by the default lane only (section 12.4).

### 11.9 Not tricking ourselves: test sensitivity, per-kernel inspection, derived tolerances (owner question of 2026-09-14)

A GPU kernel offers no debugger, no printf and no stack trace; a test that
passes because its tolerance is wide, because a compensating error hides a
real one, or because both sides share a misreading is worse than no test.
Four mechanisms, all mechanical, all part of the gates from P1 on:

1. Sabotage matrix (the tests are tested). Kernels are composed from
   `.wgsl.ts` template strings (3.5), so a test can splice a known defect
   into a kernel's source before compiling it: `test/helpers/sabotage.ts`
   defines, per kernel, a list of named mutations with the magnitude of
   error each must cause -- a flipped sign in the gravity term, `1/d`
   replaced by `1/d^2`, the `i == j` self-interaction guard removed, the
   last workgroup's rows skipped (`n - 1` instead of `n` in the bound), a
   stale `oldForce` (swap the two buffers), `atomicAdd` replaced by a plain
   store in the histogram, the outside pseudo-cell dropped, a mass lane read
   as `.x` instead of `.w`, `select` arguments swapped in the `USE_PERM`
   read, the rebase uniform ignored in a windowed dispatch. `vitest run
   --project=node test/sabotage` compiles every mutation and asserts that the
   SAME parity / invariant / exact-vs-approximate test that passes on the
   real kernel FAILS on the mutant, and by at least 10x the test's tolerance.
   A mutation that survives is a bug in the test suite and blocks the gate
   exactly as a failing test does. The matrix grows with every kernel (a
   kernel lands with at least three mutations) and runs on the default lane
   (lavapipe is enough: sabotage is about the test, not the hardware).
2. Per-kernel inspection, not end-to-end only. `GpuLayoutSimulation.inspect(
   name)` (test builds only, gated behind `GRAPHTY_GPU_INSPECT=1` so it
   costs nothing in production) reads back ANY named buffer of the iteration
   after ANY kernel (`force` after K2, after K3, after the epilogue; the
   `partials`; `cellKey`, `sortedIdx`, `cellStart`, each pyramid level,
   `state` after K1 / K4; `scenePositions` after `toScene`), and the f64
   oracle exposes the same intermediates (`oracle.forceatlas2.stages`). The
   11.4 parity tests compare stage by stage, so a wrong attraction that a
   wrong repulsion happens to cancel is caught at K2, not averaged away at
   K5; the same holds for every algorithm (frontier contents per level,
   partial sums, sort output). The one-iteration force parity of 11.4 is the
   K2 + K3 special case of this rule.
3. Tolerances are derived, not chosen. Every f32 tolerance in 11.4 and 9.7
   is justified by a measured noise floor recorded in
   `benchmarks/results/noise-floor.json` at G1: the same kernel run on its
   subgroup twin, on lavapipe vs NVIDIA, and against the f64 oracle with
   f32-rounded inputs, gives the spread that pure summation-order noise
   produces; a tolerance is at most 10x that floor, and the sabotage
   magnitudes (item 1) must exceed the tolerance by at least 10x, so the
   gap between "noise" and "bug" is two orders of magnitude and both ends
   are measured. A tolerance that has to be loosened to pass is a finding,
   recorded in the phase's decision record with the reason.
4. Run twice, compare bitwise. Every kernel test runs its kernel twice on
   the same adapter with the same inputs (`deterministic: true` where the
   option exists) and asserts bitwise-identical outputs before comparing to
   any oracle; a race, an uninitialised read, an out-of-bounds read that the
   driver clamps, or a missing barrier shows up as run-to-run disagreement
   long before it shows up as a wrong answer. The cross-adapter checks of
   11.5 (bitwise for `u32`, within the derived tolerance for f32) are the
   same idea across implementations: Dawn on lavapipe, SwiftShader, NVIDIA
   and the T4 lane share no driver code, so a result they agree on is not
   an artefact of one of them.

What this does NOT do: prove performance (the T-table gates do, with
measured numbers), or prove the element integration (G6's story on the real
GPU with the Playwright + nanobanana screenshot routine does, per the
owner's rules). Every gate in section 13 lists the sabotage run and the
per-kernel comparisons of the phase's new kernels explicitly; "the suite is
green" never means "no error was thrown" -- 11.3's oracles and invariants
and this section's sensitivity checks are what green means.

### Review notes (section 11)

- Judge verifiability flagged draft C's momentum invariant as false while judge
  performance-realism listed it as graftable; the verifiability judge is right
  (per-node factors break antisymmetry of displacements), so the invariant is
  grafted in its force-sum form (11.4).
- Draft A's "identical checksums on NVIDIA, lavapipe and SwiftShader" skeleton
  criterion is grafted into 11.5 for the `u32` results only; note 06 section
  3.5's identical checksum (999.712, printed to three decimals) on all three
  adapters was a multiply-add gather, and the FA2 tile measurably differs
  across adapters at f32 noise level, so the f32 part of the skeleton is held
  to `1e-5` across adapters and bitwise per adapter.

---------------------------------------------------------------------------

## 12. CI/CD

### 12.1 Lanes and runners

| Lane | Workflow / runner | Adapter | Runs | Required check | Contents |
| --- | --- | --- | --- | --- | --- |
| default | `ci.yml` on `ubuntu-latest` (GitHub-hosted, free for public repos) | Dawn-in-Node on Mesa lavapipe (`apt-get install mesa-vulkan-drivers libvulkan1`; `create(["adapter=llvmpipe"])` -- the `webgpu` npm package's own CI pattern, also wgpu and three.js, note 06 section 3.5) + Chromium on bundled SwiftShader (`--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader`; verified bit-identical `u32` results to NVIDIA on the probe kernel) | every push and PR | YES | build (`build:all`), lint (incl. the strict-consumer compile), knip, the whole `node` project with coverage and thresholds, the no-subgroups pass over `test/primitives test/layouts`, the `browser` smoke project |
| GPU | `gpu.yml` -- its OWN workflow file -- on a GitHub-hosted GPU larger runner (Linux, 4 vCPU, 28 GB, one NVIDIA Tesla T4 16 GB, the NVIDIA partner image; runner group `gpu`, runner name `gpu-linux-t4`, `runs-on: gpu-linux-t4`; 12.4) | NVIDIA T4 via Vulkan, `GRAPHTY_GPU_REQUIRE=nvidia` (the image's llvmpipe = failure) | push to `master`, nightly (skipped when `master` has not moved since the last green run, 12.6), `workflow_dispatch`, and same-repo PRs labelled `gpu` by a maintainer | NO -- and "not required" means NOT A JOB OF THE `CI` WORKFLOW: a job of `CI` cannot be optional in the monorepo, because `release.yml` and `coverage.yml` are `workflow_run` on `CI` gated on `conclusion == 'success'` (`release.yml` lines 3-19, `coverage.yml` lines 3-16 and 51), a failed GPU job would block every package's release, and a job waiting for an offline self-hosted runner is queued for 24 hours before GitHub cancels it (docs.github.com/en/actions/reference/limits), delaying every release by a day; a powered-off box must not block the team | `gpu-report.json`, graph-format's `gpu-upload.test.ts` on NVIDIA as a canary (graft: C Q-14; it proves NVIDIA only because the preceding `gpu-report` step exits non-zero unless the default `create([]) + requestAdapter()` pick -- the same acquisition the audit uses -- is NVIDIA, and because the step greps the audit's printed `vendor=nvidia` line and fails on a skip), `node` + `node-limits` on NVIDIA, the no-subgroups pass over the whole `node` project, the tsx benchmarks with baseline comparison, `browser` smoke on the real GPU, artifacts |

Why a GitHub-hosted GPU runner (owner decision Q-3, 2026-09-14: "I don't want
to self-host the runner, I'm happy to pay for a hosted runner"): it is the only
option that keeps every job on GitHub-managed ephemeral VMs with no
credentials, hosts or containers of the owner's involved, and it slots into
the monorepo the same way as any `runs-on`. Prerequisites and costs (note 06
sections 3.1-3.2): larger runners require GitHub Team or Enterprise Cloud and
`gh api /orgs/graphty-org` reports `plan: free`, so `graphty-org` moves to
Team ($4 per user per month); the T4 runner bills $0.052 per minute even on
public repositories; a 20-minute nightly is ~$1.04, ~$32 per month, plus the
labelled PR runs -- an org-level Actions spending limit (12.2) caps the total.
Capability: the T4 (2018, 8.1 TFLOPS f32, 320 GB/s) runs every correctness
test and the browser smoke on real NVIDIA hardware; it is NOT the machine the
T-table performance targets are measured on -- those numbers come from the dev
box's RTX 4070 SUPER through `pnpm run bench` run by hand and committed to
`benchmarks/results/` (11.7), and the T4 lane compares against ITS OWN
baseline (`benchmarks/baselines/gpu-linux-t4.json`) for regression only.
Alternative if the Team upgrade is unwanted: `machine.dev` (`runs-on:
machine/gpu=t4`, per-minute spot pricing, no plan change; note 06 section
3.4) -- the workflow differs only in the `runs-on` value and the runner
provisioning of 12.4; Cirun / RunsOn need a cloud account the project does
not have. This is the same shape as `atoms-org/cuda-ffi` (`runs-on:
cudaffi-gpu-runner` + `container: { options: --gpus all }`,
`tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`), improved in
two ways: cuda-ffi has no GPU-free correctness lane (only lint runs without a
GPU) and triggers only on push (no PR path); this plan runs the whole
correctness suite on the default lane and gates the paid GPU lane by event +
label. Cost: default lane $0 (public repo); GPU lane as above. The repository
is PUBLIC under `graphty-org` (3.1; Q-28, decided 2026-09-14); npm trusted
publishing for the package is configured later, before its first real
release (the placeholder-package step is not needed for this package until
then).

### 12.2 Cost and security controls for the hosted GPU lane, and the environment table

The GPU lane runs on GitHub-managed ephemeral VMs, so the self-hosted
concerns (fork PRs executing code on the owner's machine, registration
credentials, persistent containers) do not arise. What remains is cost and
blast radius, all applied: the GPU job's `if` requires
`github.event.pull_request.head.repo.full_name == github.repository` AND the
`gpu` label (only triage / write users can apply the label, so a fork PR
never spends GPU minutes; the same-repo clause also keeps the label from
being the only guard); an org-level Actions spending limit of $50 per month
(Settings > Billing > Spending limits; raised deliberately, never silently);
the nightly run is skipped when `master` has not moved since the last green
GPU run (12.6); `concurrency: gpu-lane` with `cancel-in-progress: true` on PR
events so a re-push does not queue a second paid run; `timeout-minutes: 45`;
workflow permissions read-only (the nightly tracking-issue job is a SEPARATE
`ubuntu-latest` job with `issues: write`, 12.3); NO secrets in the GPU job
(artifacts only; the automatic `GITHUB_TOKEN` is read-only); the runner group
`gpu` restricted to this repository and, after W1, the monorepo. Image facts
to VERIFY at G0 and record in the package `CLAUDE.md` (note 06 section 3.2,
practitioner reports): the partner image may need `sudo modprobe nvidia
nvidia_uvm` before the driver answers; headless Chromium on the T4 may need
`xvfb-run -a` around the browser project with the same Vulkan flags as the
dev box (one report found new-headless did not pick the T4 until run headed
under Xvfb); the image ships Mesa llvmpipe as a software fallback, which is
exactly why `GRAPHTY_GPU_REQUIRE=nvidia` fails the job on a software adapter;
the image's Ubuntu release and glibc decide whether the `webgpu@0.4.0` pin can
be lifted on the lane before P-ENV (D11). `libegl1` is present on the partner
image (a full desktop driver install), so the dev box's `LD_LIBRARY_PATH`
workaround is local only.

Environment variables, read by `test/setup/gpu.ts`, `vitest.config.ts` (which
forwards the browser-relevant ones into the browser project through
`test.env`) and `scripts/gpu-report.js`, all through `scripts/gpu-policy.js`
(2.3; graft: C 12.2):

| Variable | Default lane | GPU lane | Local (dev box) |
| --- | --- | --- | --- |
| `GRAPHTY_GPU_ADAPTER` | `llvmpipe` | unset (Dawn picks the discrete GPU) | unset (NVIDIA) or `llvmpipe` to mirror CI |
| `GRAPHTY_GPU_REQUIRE` | `any` | `nvidia` | unset (skip with reason) or `hardware` |
| `GRAPHTY_BROWSER_GPU` | `swiftshader` (flag set) | `nvidia` (flag set) | `nvidia` |
| `GRAPHTY_GPU_NO_SUBGROUPS` | a second pass over `test/primitives test/layouts` with `1` | a second pass over the whole `node` project with `1` | unset (the twins are also tested in-process by the `variants` axis, 11.3) |
| `GRAPHTY_DAWN_FEATURES` | unset | unset | optional Dawn toggles |
| `GRAPHTY_EGL_LIB_DIR` / `LD_LIBRARY_PATH` | -- | unset (the partner image has `libegl1`; verified at G0) | the extracted tree (`HEADLESS_GPU_REPORT.md` appendix D) |
| `VK_DRIVER_FILES` | `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (determinism) | unset | unset |
| `XDG_RUNTIME_DIR` | `/tmp` (silences Mesa) | `/tmp` | `/tmp` |
| `CI` | set by GitHub | set by GitHub | unset: the build-output test's bundle assertions hard-fail only under `CI` (2.5) |

### 12.3 Workflows for this repository now

Replaces the stale scaffold `.github/workflows/test.yml` (Node 18/20, `npm ci`,
Xvfb, SwiftShader-forcing flags; note 06 section 1). Package manager follows
`packages/` (pnpm workspace root `packages/package.json`,
`packages/pnpm-workspace.yaml`, `packages/pnpm-lock.yaml`; the repository root
holds only an npm scaffold and is deleted at P0, Q-20, so every pnpm step
runs with `working-directory: packages` AND `pnpm/action-setup@v4` is pointed
at `packages/package.json`,
whose `packageManager: pnpm@10.0.0` field is what the action reads its version
from -- the root scaffold has no such field and the action fails without
one). Two workflow files: `ci.yml` (the required default lane) and `gpu.yml`
(the never-required, paid GPU lane).

```yaml
# .github/workflows/ci.yml -- default lane, required
name: CI
on:
    push: { branches: [master] }
    pull_request:
    workflow_dispatch:
permissions: { contents: read }
concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: ${{ github.event_name == 'pull_request' }}
env:
    WEBGPU_NPM_PIN_NOTE: "webgpu@0.4.0: 0.6.1 verified to need glibc 2.38 (0.5.x assumed); dev box is Ubuntu 22.04 / 2.35"

jobs:
    test:
        name: Test (software adapters)
        runs-on: ubuntu-latest
        timeout-minutes: 30
        defaults: { run: { working-directory: packages, shell: bash } }   # shell: bash = pipefail, so a piped step cannot mask a non-zero exit
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
              with: { package_json_file: packages/package.json }
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm, cache-dependency-path: packages/pnpm-lock.yaml }
            - run: pnpm install --frozen-lockfile
            - name: Install Mesa lavapipe (software Vulkan ICD for Dawn-in-Node; not preinstalled on ubuntu-24.04)
              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
            - run: pnpm run build                    # = pnpm -r run build:all (tsc + the vite bundle): dist/browser.js, dist/node.js and their d.ts shims exist only after the bundle step (2.5)
            - run: pnpm -r run lint                  # eslint + tsc --noEmit + the strict-consumer compile (3.1)
            - run: pnpm exec knip
            - name: Node suite on lavapipe (coverage with thresholds; subgroup twins in-process, plus the no-subgroups pass)
              working-directory: packages/webgpu-graph-algorithms
              env:
                  GRAPHTY_GPU_ADAPTER: llvmpipe
                  GRAPHTY_GPU_REQUIRE: any
                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
                  XDG_RUNTIME_DIR: /tmp
              run: |
                  pnpm exec vitest run --project=node --coverage
                  GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives test/layouts
            - name: Cache Playwright browsers
              id: pw
              uses: actions/cache@v4
              with: { path: ~/.cache/ms-playwright, key: "playwright-${{ runner.os }}-${{ hashFiles('packages/pnpm-lock.yaml') }}" }
            - if: steps.pw.outputs.cache-hit != 'true'
              run: pnpm exec playwright install chromium --with-deps
            - if: steps.pw.outputs.cache-hit == 'true'
              run: pnpm exec playwright install-deps chromium
            - name: Browser smoke on SwiftShader
              working-directory: packages/webgpu-graph-algorithms
              env: { GRAPHTY_BROWSER_GPU: swiftshader, GRAPHTY_GPU_REQUIRE: any }
              run: node scripts/run-browser-project.js   # timeout -k 10 600 around vitest --project=browser --reporter=json; exit 124 passes iff the JSON has numTotalTests > 0 and numFailedTests === 0 (11.6, the browser.close() hang)
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: coverage-webgpu-graph-algorithms, path: packages/webgpu-graph-algorithms/coverage/lcov.info, retention-days: 1, if-no-files-found: error, overwrite: true }   # overwrite: a re-run of the job would otherwise fail with "artifact already exists" (v4 artifacts are immutable per run)
```

```yaml
# .github/workflows/gpu.yml -- GPU lane, never required, never a job of CI
name: GPU
on:
    push: { branches: [master] }
    pull_request: { types: [labeled, synchronize] }
    schedule: [{ cron: "17 6 * * *" }]          # nightly
    workflow_dispatch:
permissions: { contents: read }
concurrency: { group: "gpu-lane-${{ github.event.pull_request.number || github.ref }}", cancel-in-progress: ${{ github.event_name == 'pull_request' }} }

jobs:
    changed:                                     # nightly cost guard: skip when master has not moved since the last green GPU run (12.6)
        runs-on: ubuntu-latest
        outputs: { run: ${{ steps.check.outputs.run }} }
        steps:
            - id: check
              env: { GH_TOKEN: "${{ github.token }}" }
              run: |
                  if [ "${{ github.event_name }}" != "schedule" ]; then echo run=true >> "$GITHUB_OUTPUT"; exit 0; fi
                  last=$(gh run list --workflow GPU --branch master --status success --limit 1 --json headSha --jq '.[0].headSha')
                  [ "$last" = "${{ github.sha }}" ] && echo run=false >> "$GITHUB_OUTPUT" || echo run=true >> "$GITHUB_OUTPUT"
    test-gpu:
        name: Test (NVIDIA T4, GitHub-hosted)
        needs: changed
        if: >-
            needs.changed.outputs.run == 'true' &&
            (github.event_name != 'pull_request' ||
             (github.event.pull_request.head.repo.full_name == github.repository &&
              contains(github.event.pull_request.labels.*.name, 'gpu')))
        runs-on: gpu-linux-t4                    # the runner created in 12.4 (runner group `gpu`); machine.dev alternative: `machine/gpu=t4`
        timeout-minutes: 45
        env:
            GRAPHTY_GPU_REQUIRE: nvidia                 # a software adapter fails the job
            GRAPHTY_BROWSER_GPU: nvidia                 # --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface
            XDG_RUNTIME_DIR: /tmp
        defaults: { run: { working-directory: packages/webgpu-graph-algorithms, shell: bash } }
        steps:
            - uses: actions/checkout@v4
            - name: Driver up (the partner image may need the modules loaded; G0 records whether this step is a no-op)
              run: sudo modprobe nvidia nvidia_uvm || true; nvidia-smi
            - uses: pnpm/action-setup@v4
              with: { package_json_file: packages/package.json }
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm, cache-dependency-path: packages/pnpm-lock.yaml }
            - run: pnpm install --frozen-lockfile
              working-directory: packages
            - run: pnpm run build
              working-directory: packages
            - name: Adapter report (fails loudly on a software adapter; samples GPU utilisation for 10 s)
              run: node scripts/gpu-report.js > gpu-report.json && cat gpu-report.json   # no pipe into tee: the report's non-zero exit must reach the step
            - name: graph-format GPU audit on NVIDIA (canary)
              working-directory: packages/graph-format
              run: pnpm exec vitest run test/audit/gpu-upload.test.ts 2>&1 | tee canary.log && grep -q "adapter vendor=nvidia" canary.log   # the audit reads no env var (dawn.create([])) and SKIPS on acquisition failure; the grep turns a skip or a lavapipe pick into a red step
            - run: pnpm exec vitest run --project=node --project=node-limits
            - run: GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node        # the whole node project on the twins, layouts included
            - name: Cache Playwright browsers
              id: pw
              uses: actions/cache@v4
              with: { path: ~/.cache/ms-playwright, key: "playwright-${{ runner.os }}-${{ hashFiles('packages/pnpm-lock.yaml') }}" }
            - if: steps.pw.outputs.cache-hit != 'true'
              run: pnpm exec playwright install chromium --with-deps      # a fresh VM each job; pnpm 10 runs no dependency lifecycle scripts
            - name: Browser smoke on NVIDIA
              run: node scripts/run-browser-project.js                     # G0 decides whether this needs `xvfb-run -a` on the T4 image (12.2)
            - run: pnpm run bench                                          # tsx benchmarks/run.ts -> benchmarks/out/gpu-linux-t4.json
            - run: node scripts/bench-compare.js                           # > 3x the checked-in baseline for THIS runner class (benchmarks/baselines/gpu-linux-t4.json) fails; the 4070 numbers are a different class and never compared here
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: "gpu-results-${{ github.run_id }}", path: "packages/webgpu-graph-algorithms/gpu-report.json\npackages/webgpu-graph-algorithms/benchmarks/out/", retention-days: 90, overwrite: true }

    gpu-nightly-report:                          # the only job with a write permission; runs on a standard runner, never on the paid one
        needs: test-gpu
        if: always() && github.event_name == 'schedule' && needs.test-gpu.result != 'success'
        runs-on: ubuntu-latest
        permissions: { issues: write }
        steps:
            - uses: actions/github-script@v7
              with: { script: "/* open or refresh the 'GPU lane nightly' tracking issue; only after two consecutive nightly failures (the previous night's result is read from the issue body), 12.6 */" }
```

`scripts/gpu-report.js` prints adapter info, features, the four limits,
subgroup sizes, the 4-byte round-trip latency (the probe scripts of
`tmp/webgpu-plan/probe/` productised) and a 10-second sample of
`nvidia-smi --query-gpu=utilization.gpu,memory.used` (for the T-13 skip rule),
is the first step of every GPU job so a driver regression is visible in the
log (graft: A 12.2), and exits non-zero when `GRAPHTY_GPU_REQUIRE` names a
vendor that does not match -- through the same `scripts/gpu-policy.js` the
test setup uses (2.3). `scripts/run-browser-project.js` is the ~20-line
wrapper of 11.6 around the browser project. The staging `pnpm-workspace.yaml`
and `knip.config.ts` (both enumerate packages by name) gain the package's
entries in P0.

### 12.4 Provisioning the hosted GPU runner (org side, one-time)

Nothing here is committed to the package; it is the owner's checklist, done
once in P0 (G0 records each item):

1. Move `graphty-org` to the GitHub Team plan (larger runners are gated on
   it, note 06 section 3.2).
2. Organization > Settings > Actions > Runners > New runner > New
   GitHub-hosted runner: Linux x64, size "4-core GPU" (NVIDIA T4), image
   "NVIDIA GPU-Optimized Image for AI and HPC" (the partner image the GPU
   size offers), name `gpu-linux-t4`, runner group `gpu` with repository
   access limited to `webgpu-graph-algorithms` (the monorepo is added at W1)
   and "Allow public repositories" ON (both repositories are public);
   maximum concurrency 1.
3. Organization > Billing: an Actions spending limit ($50 per month to start)
   and the billing e-mail alert at 75%.
4. Repository settings: read-only default workflow permissions; "Require
   approval for all external contributors"; the `gpu` label created.
5. A first `workflow_dispatch` run of `gpu.yml` on the empty skeleton, then
   the deliberate red run: `GRAPHTY_GPU_REQUIRE=nvidia` on the software lane
   must fail, recorded in the P0 PR (graft: C G0) -- possible only because
   the report step never pipes into `tee`. The same run records the image's
   Ubuntu release, glibc, driver, whether `modprobe` was needed and whether
   headless Chromium found the T4 without Xvfb (12.2).
6. `benchmarks/baselines/gpu-linux-t4.json` committed from that first green
   run (11.7); the dev box's 4070 baseline is a separate file and a separate
   runner class.

If the Team upgrade is declined, steps 1-2 become a `machine.dev` account and
`runs-on: machine/gpu=t4`, and step 6's file is named for that class; nothing
else changes. Keep `webgpu@0.4.0` until the dev container moves to Ubuntu
24.04 (glibc 2.39) at P-ENV, then bump once everywhere (D11); the hosted
image's glibc is recorded at G0 and does not by itself unblock the bump,
because the dev box is where the suite is run most.

Coverage comes only from the default lane: `tools/merge-coverage.sh --ci`
fails on any missing package artifact and `coverage.yml` runs only on a
successful CI run (note 06 section 4.4) -- the same `workflow_run` gate that
forbids the GPU job from living in `ci.yml` -- so `gpu.yml` never uploads
`coverage-*`. It uploads `gpu-results-*` (90-day retention, like the
monorepo's `performance-baseline`).

### 12.5 Slotting into the monorepo at W1

`graphty-monorepo/.github/workflows/ci.yml` builds once, then fans out a `Test
(${{ matrix.shard }})` matrix on `ubuntu-latest` (lines 236-351:
`algorithms-default`, `algorithms-browser`, `layout`, five
`graphty-element-browser-*` shards, ...) with `needs-browser` /
`needs-storybook` flags and a cached `playwright install chromium --with-deps`
step (lines 413-427); `all-checks` (lines 706-738) needs only `test` and the
Chromatic jobs. Following `packages/move/root-touch-points.diff` (note 06
section 6), the two SOFTWARE shards join `ci.yml`; the GPU job does NOT.
`ci.yml` must never gain a `schedule` trigger or a `labeled` PR type: its build
steps run only on `pull_request` and on `push` / `workflow_dispatch` to master
(lines 79-90), every one of its 16 test shards downloads all six build
artifacts unconditionally (lines 380-403) and its five Chromatic jobs carry no
event guard, so a scheduled run would fail every shard with "Artifact not
found" (the failure mode `root-touch-points.diff` lines 133-141 documents) and
burn Chromatic snapshots nightly, and a `labeled` type would re-run the whole
workflow on every label of every PR.

```diff
@@ jobs.build
+            - name: Build webgpu-graph-algorithms (PR)
+              if: github.event_name == 'pull_request'
+              run: pnpm exec nx run webgpu-graph-algorithms:build            # nx build = build:all (project.json, as graph-io's)
+            - uses: actions/upload-artifact@v4
+              with: { name: build-webgpu-graph-algorithms, path: webgpu-graph-algorithms/dist/, retention-days: 1 }
@@ jobs.test.strategy.matrix.shard
+                    - webgpu-graph-algorithms-node
+                    - webgpu-graph-algorithms-browser
@@ jobs.test.strategy.matrix.include
+                    - shard: webgpu-graph-algorithms-node        # Dawn on Mesa lavapipe, no GPU; thresholds active (11.8)
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && pnpm exec vitest run --project=node --coverage && GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives test/layouts
+                      needs-browser: false
+                      needs-storybook: false
+                      needs-vulkan: true
+                    - shard: webgpu-graph-algorithms-browser     # Chromium SwiftShader smoke
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && node scripts/run-browser-project.js
+                      needs-browser: true
+                      needs-storybook: false
+                      needs-vulkan: false
@@ jobs.test.steps (before "Run tests")
+            - name: Install Mesa lavapipe
+              if: matrix.needs-vulkan
+              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
+            - uses: actions/download-artifact@v4
+              with: { name: build-webgpu-graph-algorithms, path: webgpu-graph-algorithms/dist/ }
@@ jobs.test.steps "Run tests" env
+                  GRAPHTY_GPU_ADAPTER: llvmpipe
+                  GRAPHTY_GPU_REQUIRE: any
+                  GRAPHTY_BROWSER_GPU: swiftshader
+                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
@@ coverage upload condition
+  || startsWith(matrix.shard, 'webgpu-graph-algorithms-node')      # the browser shard uploads no coverage
```

```yaml
# .github/workflows/gpu.yml (monorepo) -- the 12.3 gpu.yml with three changes: paths without the packages/ prefix, its own
# install + `pnpm exec nx run-many -t build --projects=graph-format,webgpu-graph-algorithms` on the runner (no `needs:` on any ci.yml job and
# no artifact download, so it is independent of CI's conclusion and of CI's build gating), and `pnpm/action-setup@v4` without
# package_json_file (the monorepo root package.json has packageManager)
```

Plus the root touch points: `pnpm-workspace.yaml`, `commitlint.config.js`
scope, `knip.config.ts` workspace (`src/node/index.ts` and
`src/browser/index.ts` as extra entries), `tools/merge-coverage.sh` `PACKAGES`,
`tools/prepush.sh` (`test:node` only; tsc-only, so the bundle assertions of
2.5 run in CI only; the pre-push hook on the dev box uses NVIDIA or
`GRAPHTY_GPU_ADAPTER=llvmpipe` to match CI), root `package.json`
`coverage:preview:webgpu-graph-algorithms` on port 9058, `release.yml` download
step (OIDC trusted publishing, npm 11, gated on CI success -- unchanged for
this package precisely because the GPU job is not in `CI`; the GPU lane never
publishes). No root `eslint.config.js` edit: the package brings its own
package-local config (2.4). The strict-consumer compile needs no new step
because it is part of the package's `lint` script (3.1), which `nx affected
-t lint` runs. The shard matrix has no `nx affected` gating (note 06 section
6), so the two software shards simply join it; the GPU workflow is gated by
event + label, not by paths ("a change to graph-format is exactly when the GPU
lane should run"). This is the owner's "one runner for GPU and a default
runner for other tests": the package-level split into vitest projects is what
makes the routing a `runs-on` per job, and the workflow-level split is what
keeps the paid GPU lane's cost and availability out of every other package's
release.

### 12.6 Budgets and drift

| Job | timeout | Expected duration | Artifacts | Retention |
| --- | --- | --- | --- | --- |
| default `test` (`ci.yml`) | 30 min (target <= 15, T-12) | lavapipe suite 3-10 min (wgpu budgets 5-15 min for its whole lavapipe job; the 100k / 1M gather probe took ~190 ms per 50 iterations at 4 threads, note 06 section 3.5) + 1-2 min browser smoke | `coverage-webgpu-graph-algorithms` | 1 day |
| `test-gpu` (`gpu.yml`) | 45 min (target <= 20, T-12) | ~2 min VM boot + install, 3-8 min tests on the T4, 3-10 min bench; ~$0.50-1.00 per run at $0.052/min | `gpu-results-*` | 90 days |
| nightly `test-gpu` + `gpu-nightly-report` | 45 min | same, and SKIPPED by the `changed` job when `master` has not moved since the last green run (no cost on quiet days); the report job opens / refreshes the tracking issue only after TWO consecutive nightly failures (a single failure can be a shared-tenant slow run, T-13) | same | 90 days |

If the lavapipe `node` project passes ~10 minutes it is split into two shards
with `--shard=1/2` (the monorepo already shards graphty-element five ways).
Drift risks handled: lavapipe version skew (dev box Mesa 23.2 vs runner Mesa
25.2) and the partner image's driver updates are diagnosable because every
job prints the adapter description first;
Playwright bumps change the bundled Chromium (139 today) -- the nightly GPU lane
catches a broken flag set; the `webgpu` pin is re-evaluated at P-ENV.

### Review notes (section 12)

- Judges (drafts A and C): pnpm steps at the repository root, where no pnpm
  workspace exists; draft C's job `container:` on a sibling-container runner
  and its literal image tag. All avoided: `working-directory: packages`, no
  job container, the runner image carries the libraries.
- Judge integration-feasibility (draft B 12.3): the GPU job installed pnpm
  without a cache. `cache: pnpm` + `cache-dependency-path` added.

---------------------------------------------------------------------------

## 13. Phased implementation plan

Rules for every phase (graft: C 13): (a) the gate is a list of tests and
recorded MEASURED numbers that must be GREEN on the default lane AND the GPU
lane before the next phase starts (a browser step that hit the `browser.close()`
timeout with every test passed counts as green, 11.6); (b) a phase adds only
the primitives its slice needs; (c) every [X] number the phase touches is
replaced by a measured one in `benchmarks/results/`; (d) nothing lands with
`eslint-disable`, `@ts-expect-error` (outside negative type tests), non-ASCII,
or a CPU fallback; (f) every phase that adds a kernel adds its sabotage
mutations, its `inspect()` stage comparisons and its noise-floor row (11.9),
and the gate lists them; (e) sizes are engineer-days (ed) for one engineer familiar
with the code base; the WGSL phases carry the most uncertainty. The owner's
ordering constraint -- force-directed layout right after the skeleton -- is
P3 (exact) and P4 (scale), and rule (b) is what keeps P2 from absorbing P4 /
P7 / P8 infrastructure ahead of P3.

| Phase | Scope | Deliverables | Gate (must be green before the next phase) | Size |
| --- | --- | --- | --- | --- |
| P0 Reset + package skeleton | create the PUBLIC repository `graphty-org/webgpu-graph-algorithms`, push the reset scaffold as its first commit, set `repository.url` / `directory`, create the `gpu` label, provision the hosted GPU runner and the spending limit per 12.4 (Q-3, Q-28); scaffold triage per note 07 section 5 (delete `src/types/index.ts` `CSRGraph`, the empty `src/*` dirs, `test/setup/*`, `test/helpers/*`, `vitest.config.ts`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `knip.json`, `.husky`, `package-lock.json`, `.env*`, `.github/workflows/test.yml`, `examples/`, `STRATEGY.md`, `IMPLEMENTATION_CHECKLIST.md`; keep `CLAUDE.md`; move `HEADLESS_GPU_REPORT.md` under the package `docs/`); create `packages/webgpu-graph-algorithms/` mirroring graph-io (3.1) with its entries in `packages/pnpm-workspace.yaml` and `packages/knip.config.ts` (both enumerate packages by name); `src/index.ts` exporting only `WebGpuGraphError` and constants; `test/setup/gpu.ts` (acquisition with a fresh adapter per device, the `GRAPHTY_GPU_REQUIRE` policy through `scripts/gpu-policy.js`, uncapturederror hook), `test/setup/browser.ts`; `scripts/gpu-report.js`, `scripts/run-browser-project.js`; the 12.3 `ci.yml` and `gpu.yml` with one trivial device test; the runner recipe of 12.4 executed by the owner on the host; the owner's sign-off on the 7.2 formula table recorded in this PR (D21) | a buildable, lintable, knip-clean package; both vitest projects run one device test; default lane green; GPU lane registered and green | G0: the default lane ran on GitHub; `pnpm run build` (= `build:all`), `lint` (incl. `typecheck:strict-consumer`), `knip` pass; the `node` project acquires an adapter and prints `adapter.info` on the dev box (NVIDIA), the dev box with `GRAPHTY_GPU_ADAPTER=llvmpipe`, and `ubuntu-latest` (lavapipe); the `browser` project acquires SwiftShader on `ubuntu-latest` and NVIDIA locally with the four flags; the deliberate red run (`GRAPHTY_GPU_REQUIRE=nvidia` on the software lane fails) recorded in the PR; `gpu-report.js` exits non-zero on a software adapter under `nvidia` and the step is red (no `tee`); the 7.2 sign-off recorded; the Vitest 3 per-instance `launch` spelling recorded in `CLAUDE.md` | 2-3 ed |
| P1 Walking skeleton | `GpuContext` in `src/context.ts` over `src/device/` (probe / create / from / caps / error scopes / lost / dispose, D26), `./browser`, `./node`, `webgpu-constants`, `planUpload` (arena / perArray; windowed PLANNING), minimal `GraphResidency` (core + `outDegree()` view + one column, `release`, `stats`), `BufferPool`, `Readback` ring, `PipelineCache`, `Kernel` with layouts derived from `spec.bindings`, `plan1d` / 2D, `UniformBlock` (uniform + storage modes), `composeWgsl` + prelude with interpolated constants, `kernels.ts`, the `degree` kernel, ONE exact-tile FA2 repulsion iteration (K3 with its swing / traction epilogue plus the one-workgroup K4 speed finalize, both written to the 3.5 uniformity rule and the 7.2 table signed off at G0) and the state-block trace readback, `reduce` (section 6 row 1) with its twin, `test/helpers/leak-counter.ts`, the tsx benchmark harness with `upload` and `roundtrip` groups | `degree(ctx, snapshot)` correct in Node and browser; the skeleton test of 11.5 | G1: 11.5 in full on lavapipe, SwiftShader and NVIDIA (`u32` results bitwise identical across adapters, the FA2 iteration within `1e-5` across adapters and bitwise per adapter); upload contract tests ported from `gpu-upload.test.ts`; `build-output.test.ts` proves no `webgpu` import specifier in the root / browser bundles with the bundle present (hard-fail under `CI`); T-1 / T-2 / T-3 recorded; coverage >= 80/80/75/80 on `--project=node` (thresholds active, 11.8) | 4-6 ed |
| P2 Batch + dispatch infrastructure (what P3 needs, and only that) | `Lease`, `CommandBatch` (with the post-submit pending-error check), `UniformRing` dynamic offsets, `Profiler`, pipeline warm-up, device-loss state, `residentBytes` + warning; `segmentedReduce` in its thread-per-row tier (over `[midEnd, n)`) with the twin mechanism; `test/helpers/caps-tables.ts`, `test/helpers/override-matrix.ts`, the `test/oracle/` skeleton; the WGSL compile matrix on both runtimes. NOT in P2 (rule (b)): windowed upload EXECUTION and the `node-limits` project (P4), `scan` / `compact` / `histogram` (P4), the indirect finalize (P4: first needed by the grid's hub-cell tier G4a), grid-stride dispatch and `packViews` (P7), the mid / high tiers of `segmentedReduce` (P4, gated at G4) | the kernel layer of section 5 complete with tests; the primitives P3 needs | G2: `planUpload` unit tests for every path x caps table; the `(16,776,960, 16,777,216]` boundary and a WGSL `linear_id` test at the boundary on lavapipe; `reduce` / thread-per-row `segmentedReduce` equal their oracles at scaled sizes 0..2^24 incl. all-equal keys and one hub row, subgroup variant and twin in-process, bitwise deterministic across two runs; `UniformBlock` negative test rejected on Chromium and the `state` round trip green; `E_DEVICE_LOST` mid-readback on both runtimes with recovery from a fresh adapter; a bad bind group rejects its own batch's readback under Dawn-node; leak counter 0 after `release` and `dispose`; GPU lane green with `GRAPHTY_GPU_REQUIRE=nvidia` and `gpu-report.json` uploaded; default lane <= 15 min (T-12) | 4-6 ed |
| P3 ForceAtlas2, exact tier, as a `LayoutSimulation` (FIRST DELIVERABLE) | `ForceSimulation` state machine with the `ForceModel` hook interface (7.19), `createForceAtlas2` with the exact tier (7.4-7.6, 7.9-7.11, 7.17-7.18; `vec4f` positions D23; no clamp D25), the `compat` overrides (7.2), `setFixed` / `setPosition` override list (7.12), `reheat` (D8), `setParams`, `run()` batch driver, `stats` + trace, LCG seeding (`seed.ts`, f32), `inputs.ts` (role-`mass` column, `Float32Array` or column-name `nodeMass`; `true` / column-name weights; D28), `toScene`, the CPU FA2 oracle in `test/oracle/forceatlas2.ts` (f64 and an f32 variant, index-based, the 7.2 table; the SPEC of the L1 `ForceAtlas2Simulation`, moved -- not copied -- into `layout/` if L1 wants it) cross-checked against committed NetworkX trajectory fixtures in `compat: "networkx"` (11.4), the `frame-loop.ts` helper, browser smoke (3) and the 10k `bench`-tagged browser test of 11.6, `layout-exact` benchmarks; `createAccelerator` with `forceAtlas2` and the layout tuning defaults | a usable GPU FA2 from Node (`benchmarks/layout-run.ts` lays out the 100k / 1M graph end to end on the exact tier: ~18 ms per iteration, correct) and from the browser frame loop | G3: 11.4 layout parity in full (oracle independence vs the NetworkX fixtures (1, 5, 50 iterations in `networkx` mode; iteration-0 forces cover `paper` mode; the two-line `SWING_MODE` unit test); force 1e-4; trace 1e-4 vs the f32 oracle / 5e-2 vs f64; distributional 10%; the behaviour pins; every option combination incl. `compat`, 2D / 3D, seeded / unseeded, `arcCount === 0`, a pinned node); 11.3 properties for layouts (all-fixed mask, `z === center.z`, pin survives a remapped `load`, `speed` not reset by `setPosition`) and the force-sum invariant incl. the coincident fixture; the FA2 kernels' subgroup twins identical (`1e-6`) in-process; lifecycle (`dispose` leak 0, `E_RELEASED` after `release` during a live simulation, device loss); the frame-loop test (600 ticks, at most `maxInFlight` in flight, `setPosition` never clobbered) also in the browser on SwiftShader and NVIDIA; T-4 met and the exact curve recorded; `exactMaxNodes` re-fixed by the 7.8 rule and written into `constants.ts` with the citation; T-5 at 10k in Chromium from the `bench`-tagged test; lavapipe runs the whole FA2 suite at `gpuScale` sizes in <= 3 min; the 11.9 sabotage matrix for K1-K5 (at least three mutations per kernel, each failing its parity test by >= 10x the tolerance), stage-by-stage `inspect()` parity for K2, K3, the epilogue, K4 and K5 against the oracle's stages, and every 11.4 tolerance traced to the noise-floor file | 8-10 ed |
| P-ENV Environment move (one change) | the dev container and the runner image move to Ubuntu 24.04 (glibc 2.39, Mesa 25.x lavapipe, `libegl1` present); `webgpu` bumped from 0.4.0 to the current 0.6.x in both lanes, in graph-format's devDependencies and in this package's devDependency and `E_NO_WEBGPU` install hint (the peer range `>=0.4.0 <1.0.0` already admits it, 2.5); the `LD_LIBRARY_PATH` workaround removed; the 0.6.x unmap-on-destroy shim noted as redundant with `Readback`'s own `unmap` | one environment change between the exact and grid phases (draft C's question Q-10; here Q-4) | G-ENV: G1-G3 re-run green on both lanes with the new image; `gpu-report.json` shows the new driver / Mesa versions; benchmark baselines re-recorded for the new runner class | 1-2 ed (+ owner time on the host) |
| P4 Scale: grid pyramid + degree tiers | `scan`, `compact`, `histogram` / counting sort, `radixSort` (digit-major histograms), the indirect `finalize` kernel and `planIndirect`, windowed upload EXECUTION for row-walking kernels (`ArcWindow`, rebase uniform, row clamping) and the `node-limits` project, the grid kernels G1-G7 with G4a / G4b (7.7) in 2D and 3D with the robust extent, the outside pseudo-cell, `state.eps`, the sorted-order dispatch (D24), `repulsion: "auto"` crossover, `calibrateLayout()`, the mid / high attraction and `segmentedReduce` tiers over `degreeOrder()` (7.5, 6 row 3), `nearMax` / `gridMax` / `extentFactor` options, `stats.maxCellOccupancy` / `outsideGrid`, the exact-vs-approximate fixtures and tests (11.4), `layout-grid` benchmarks on the grid ladder in 2D and 3D, hub-heavy and isolated-node fixtures | FA2 usable at 10^5-10^6 nodes | G4: 11.4 exact-vs-grid in full at 20k / 100k / 262k (RMS <= 5%, p99 <= 25% on uniform, clumpy and isolated-node fixtures with the floored denominator; unbiasedness; distributional 15% over 200 iterations; EXPANSION PARITY within 25% at 50 and 200 iterations; an isolated node's force equals gravity; bitwise determinism with `deterministic: true`; pyramid <= 40 MB in 3D) and the one-iteration + unbiasedness checks at 1M in `node-limits`; the settle test on the isolated-node fixture; `radixSort` equals a stable `Array.sort` on 8 / 16 / 24 / 32-bit keys with values, sizes 0..2^22 (scaled), all-equal keys; `histogram` / counting sort equal their oracles with one hot bucket; `cellStart` correct with empty cells and a 1M-entry hub cell dispatched through G4b; windowed `degree` with a FAKED 1 MiB binding limit (>= 8 windows) and a hub row longer than a window equals `outDegree()`; `node-limits`: a real 2 GiB binding request succeeds on the 4070, a 200 MB per-array upload is bound windowed at defaults, a real 2D dispatch on 100M items; the upper `segmentedReduce` tiers equal their oracles with the 10k-degree hub, twin in-process; T-6 and T-7 met; T-5 at 100k in Chromium; the crossover re-checked and `exactMaxNodes` adjusted if the grid is faster below it; a decision record on: option B (cluster tree) -- default no; `gridMax2D` in {512, 1024, 2048} at 1M and `extentFactor` (Q-32); a near-field force bound (D25) and adaptive `nearMax` (R-24) -- default neither; the position permutation of note 03 8.3 item 5 -- default no; lavapipe runs the grid suite at `gpuScale` sizes in <= 4 min; the 11.9 sabotage matrix for G1-G7 (dropped pseudo-cell, plain store for the histogram atomic, off-by-one cell bound, wrong level offset) and stage-by-stage `inspect()` parity of `cellKey` / `sortedIdx` / `cellStart` / every pyramid level / the far-field and near-field forces against the oracle's stages | 10-14 ed |
| P5 Fruchterman-Reingold + the spring-electrical preset | `createFruchtermanReingold` (7.20: `LAW = FR`, temperature slots, `FR_APPLY`, `fixed`, the `\|\| 0.1` guard, `reheat` at 0.7, `FruchtermanReingoldStats`), `createSpringElectrical` (the preset of 7.20 with the velocity integrator, ngraph's option names and settle rule, `SpringElectricalStats`), FR oracle, browser smoke, `layout-fr` benchmarks; `GpuAccelerator.fruchtermanReingold` / `springElectrical` | second and third layouts; the element can route `spring` (FR) and offer `spring-electrical`; routing `ngraph` to the preset stays a product decision (Q-9) | G5: one-iteration displacement parity with the CPU FR oracle (<= 1e-4), fixed nodes immobile, output NOT rescaled when `fixed` is given; the preset settles within 1,000 steps on the 150-node / 250-edge "Performance/Large Graph" story graph to an edge-length distribution within 25% of ngraph's (ngraph run on the CPU in the test, devDependency of the test only); T-14 recorded | 4-5 ed |
| P6 Integration PRs (monorepo) | `algorithms`: 9.2 interfaces + `accelerated()` with the ported methods + `pathTo` / `pathEdges` decoration + `sources` / `k` + fake tests (first A2 commit); `layout`: 9.3 `LayoutSimulation`, `LayoutAccelerator`, `SimulationOptions`, `createSimulation` with the type table, steppable CPU FA2 / FR with the 7.2 formulas, `resolveNodeVector` / `resolveWeights` / `seedPositions`, Chromatic re-baseline; `graphty-element`: 9.4 items 1-9 incl. the `accelerator-changed` consumer, the release list, the pin re-application and the `gpuMinNodes` evaluation, with the E1 element tests; app: 9.5 `attachAccelerator` | the GPU layout "detected" in the app; a Storybook story the owner can open on the dev box to see the GPU FA2 animate with drag and pins | G6: element tests + stories green with a fake accelerator (incl. late injection, mid-run removal, pin survival across a remap); the story on the real GPU locally settles, drags and pins (screenshot checked with the Playwright + nanobanana routine the owner's rules require for visual work); the GPU package's structural mirrors match the real interfaces (type test run manually against the monorepo checkout until W1); the same story on the CPU simulation looks statistically the same (the 11.4 distributional metrics) | 8-10 ed (across three packages, on top of the design's E1 refactor, which is a precondition and NOT sized here); PREPARED on branches after A1 merges, MERGED only after F2 (design 13.5 rule 5) |
| P7 SpMV family + WCC + the algorithm accelerator surface | grid-stride dispatch, `packViews`, `spmvPull` over pre-scaled `xNorm`, the device out-weight normaliser, PageRank (+ personalized) with `pr-scale` / `pr-finalize` and `firstConvergedIteration`, HITS, eigenvector, Katz, Afforest WCC (with the two-dispatch atomic dedupe), `renumberPartition` on readback, `reverse()` residency (identity when undirected; `fwdArc` never touched), `degreeOrder({ of: "reverse" })` tiers, the `AlgorithmAccelerator` structural interface and `GpuAccelerator` methods, oracles (NetworkX-semantics PageRank matching `pagerank.ts`, union-find), `pagerank` / `wcc` benchmarks | first algorithms through the accelerator interface | G7: 9.7 parity on all fixtures (<= 1e-5, top-k, `iterations` +-1 through `firstConvergedIteration`, `converged` identical; weighted with zero-weight arcs and dangling nodes; directed and undirected; personalization one-hot and uniform); the PageRank pull kernel binds exactly the 8 of 8.2 and every 8.10 kernel matches its count (descriptor test); no host readback inside a batch of 8 iterations (`mapAsync` count through the leak counter); SpMV twin identical in-process; WCC partition equality after renumbering incl. directed inputs treated weakly, singletons, giant component + dust, `arcCount === 0`; T-8 and T-9 recorded; browser smoke (4) green | 8-10 ed (may start after P2 in parallel with P3-P5; coordinate on `segmentedReduce`) |
| P8 Frontier family | `Frontier` with the sized / chunked edge queue, `advance` (block_mapped + workgroup tier + subgroup variant), `dedupe`, bitset, the multi-candidate indirect args and the device-side `finalizeArgs` selector (5.4); BFS (+ direction-optimizing, `switches` as a device counter, `atomicMin` claims), closeness / harmonic / eccentricity, SSSP near-far with the two-pass predecessor, Bellman-Ford, window-aware advance (lifting the `E_TOO_LARGE`), oracles (FIFO BFS, binary-heap Dijkstra, Bellman-Ford), `bfs` benchmarks | BFS, closeness, SSSP, Bellman-Ford | G8: BFS `depth` exact and parent / order level-consistent on all fixtures incl. the 1000 x 1000 grid and a 10k-degree star; the fused and two-phase kernels agree and the device-side selection picks each at least once on an RMAT fixture; the direction-optimizing path agrees with top-down and `switches > 0` on an RMAT fixture; a level whose degree sum exceeds a FAKED 4,096-entry edge-frontier capacity gives exact depths (the chunked overflow rule); SSSP `dist` within 1e-5 incl. zero weights, `predArc` attains `dist`, `E_UNSUPPORTED` on negative weights, `flags.allWeightsOne` routes to BFS; Bellman-Ford detects a planted negative cycle; `mapAsync` count <= levels / 32 + 1 on the grid fixture; the indirect finalize clamps above 65,535 workgroups (a synthetic 17M frontier on lavapipe); subgroup tier identical on / off at sizes 4 / 8 / 32; T-10 recorded | 10-14 ed |
| P9 Betweenness + APSP | McLaughlin-Bader forward pass (u32 sigma with `sigmaOverflow`, `S` / `ends`), successor-pull backward pass writing `n x k` deltas, the per-batch `bc` gather (no float races), tagged multi-source batching planned from `maxBufferSize`, the online work-efficient / edge-parallel switch, sampling (`sources` / `k`), edge BC via `foldArcs(..., "first")` halved on undirected, normalisation identical to the CPU, `onProgress` / `signal`; blocked Floyd-Warshall and BFS-based APSP with the binding-size bound of 8.7; Brandes oracle | BC (sampled and exact for small n), APSP | G9: exact BC on karate, path, star, cycle, grid, random 2k <= 1e-4 relative and top-k; analytic sums on path and star; edge BC folded correctly (both arcs equal before folding, asserted); sampled BC (256 sources) on a 100k-node RMAT has Spearman >= 0.9 with exact BC on a 10k subgraph and equals the CPU's sampled result on the same `sources` list; the overflow flag fires on a constructed small-world graph; batch planning honours a faked `maxBufferSize` (k shrinks; results equal); APSP exact unweighted / 1e-5 weighted in `node-limits`, `E_TOO_LARGE` above the 8.7 bound; T-11 recorded | 5-7 ed |
| P10 Move-in (W1) | `packages/README.md` checklist verbatim; the 12.5 `ci.yml` diff, the monorepo `gpu.yml` and the root touch points; the structural mirrors are deleted for `import type` from the optional peers `@graphty/algorithms` / `@graphty/layout` (D27) and `test/types/conformance.test-d.ts` is retired; `indexed.*` ADDED as a second oracle beside `test/oracle/`; `seedPositions` cross-test against the real `RandomNumberGenerator`; the self-hosted runner registered for the monorepo; design 10.3 / 14.5 / 14.6 / 16.2 / 16.7 amendments (DEPARTURE-1, -2, -4, -5, -6; DEPARTURE-3 and -7 amend 14.3 in the L1 PR); README performance table regenerated from `benchmarks/results/` | the package in the monorepo, both CI lanes live, first release `0.1.0` with provenance | G10: monorepo default shards green on lavapipe + SwiftShader with the strict-consumer compile inside `lint`; the `test-gpu` job of `gpu.yml` green on a labelled PR and on master with `ci.yml`'s `on:` untouched; `all-checks` unchanged; coverage merged by `tools/merge-coverage.sh --ci` with the new package in PACKAGES; `nx release` dry run versions the package independently; `expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>()` and the reverse compile; every differential test passes against BOTH oracles with the 9.7 tolerances (a mismatch is a bug in one of the two packages and blocks W1) | 3-4 ed |
| P11 Structure + community | k-core, triangle counting / k-truss, label propagation, Boruvka MST (two-pass min), `cooToCsr`, per-row group-by-key (workgroup sort / global hash), Louvain (move phase with `up_down`, reduce-by-key cluster weights, device contraction), Leiden refinement if time allows; oracles for each; the 8.10 binding counts asserted | the long tail, each merged separately | G11: `radixSort` on 32-bit keys with values and `cooToCsr` output passing `fromCsr(...).validate({ level: "full" })`; k-core exact; triangles exact per node and total; k-truss support exact; LPA recovers planted partitions (ARI >= 0.9) on 10 seeds; Boruvka `totalWeight` within 1e-5 and edge set identical on distinct weights; Louvain modularity within 0.02 of the CPU on karate / planted partitions and never below the CPU's by more than 0.05 on random fixtures; every level's contraction preserves `totalWeight`; no mixed atomic / non-atomic access (the compile matrix on both runtimes); T-15 recorded | 12-16 ed |
| P12 Element polish (W2) | `gpuMinNodes` default from measurements, `iterationsPerStep` auto-raise above 250k nodes, `calibrateLayout()` + `createAccelerator` defaults wiring in the app, the real-GPU stories under a `gpu` tag in the app, device-loss UX (toast + the CPU simulation taking over the running layout), docs (README with the Node and browser recipes) | the "GPU: on / off" indicator and stories | G12: stories green; nightly GPU lane green for a week; README numbers regenerated from `benchmarks/results/` | 4-6 ed |

Phase order and parallelism:

```
P0 -> P1 -> P2 -> P3 (FA2 exact) -> P-ENV -> P4 (grid) -> P5 (FR / preset)
                    \-> P7 (SpMV + WCC) -> P8 (frontier) -> P9 (BC + APSP) -> P11 (structure, community)
P6 (integration PRs): prepared on branches once A1 has merged and P3's simulation surface is fixed; merged after F2 + L1 / E1 scheduling
P10 (W1): after A2 / L1 / E1 land in the monorepo and at least P3-P5 and P7 are green
P12 (W2): after P10
```

Critical path to the owner's first need: P0 -> P1 -> P2 -> P3 (18-25 ed)
gives an interactive GPU ForceAtlas2 usable from Node and from a
graphty-element story with an injected accelerator; P-ENV + P4 (11-16 ed)
takes it to 10^5-10^6 nodes; P6 makes it "detected" in the app -- but
"detected in graphty" additionally waits for F2, the first A2 commit, L1 and
the design's E1 element port (`DataManager` owning the builder,
`snapshot-replaced`, the position column) to land in the monorepo, none of
which this plan sizes or schedules; until then the layout is usable from Node
and from a story with an injected accelerator. Everything after P6 is
additive. Rough total: 83-113 ed across P0-P12 plus the monorepo-side
coordination at A2 / L1 / E1.

### Review notes (section 13)

- Judge verifiability (draft B P0): one 6-8 day phase bundled the scaffold
  reset, the whole infrastructure and an FA2 kernel behind a single gate. Split
  into P0 (reset, G0), P1 (skeleton, G1) and P2 (infrastructure, G2) with
  draft C's gate discipline.
- Judge integration-feasibility (draft B P4): "parallel with P3 once A1 has
  merged" contradicted the F2 precondition; P6 now states both.

---------------------------------------------------------------------------

## 14. Risks and open questions for the owner

### 14.1 Risk register

| Id | Risk | Likelihood / impact | Mitigation | Default |
| --- | --- | --- | --- | --- |
| R-1 | FA2 semantics: if the CPU rewrite keeps the port's `1/d^2` while the GPU uses `1/d`, the two never agree and every parity test is meaningless; and the WGSL and the f64 oracle are two transcriptions of one table by one author, so a shared misreading would pass every parity test | high / high | ONE table (7.2) for both; G0 requires owner sign-off before any WGSL implementing a 7.2 row merges (D21); G3 cross-checks the oracle against committed NetworkX trajectory fixtures in `compat: "networkx"` (11.4); the layout tests assert no exact coordinates, so adopting the published law does not break them; the old port variants are dropped, not kept behind a switch | published laws by default, `"networkx"` option (Q-1, decided 2026-09-14) |
| R-2 | The exact-tier crossover is wrong for other GPUs (integrated, Apple, T4) | medium / low | `exactMaxNodes` is an option; `calibrateLayout()` measures it on the actual device and the app passes it through `createAccelerator(ctx, { layout })` (the only route to a simulation the element creates, 9.5); the grid tier is correct at any n so a wrong crossover only costs time | 16,384 on discrete GPUs (conservative), re-fixed at G3 (Q-6); the app calls `calibrateLayout()` |
| R-3 | Grid pyramid quality on clumpy layouts (hub cells, empty space) is worse than Barnes-Hut; cosmos reports shimmer on a 163-node graph before its fixes; and the NORMAL case of a giant component with 1% isolated nodes and small components puts the strays at `k M / g` (7.7), which collapses a bbox-derived grid | high / medium | the robust extent (`min(bbox, extentFactor * rmsRadius)`) and the outside pseudo-cell; exact tier below the crossover; `nearMax` cap with Horvitz-Thompson weighting; the hub-cell workgroup tier; P4 fixtures include cosmos's failure cases AND the isolated-node fixture (RMS <= 5%, p99 <= 25%); the Hilbert cluster tree is the documented escape hatch sharing `radixSort` | grid first, tree only if G4 fails (Q-5); `extentFactor` and `gridMax2D` re-checked at G4 (Q-32) |
| R-4 | Chromium per-frame readback (2.65 ms per MiB [M]) caps interactive n well below the compute limit | high / medium | `iterationsPerStep` auto-raise; positions stay GPU-authoritative; at 1M nodes the renderer cannot draw per-node meshes anyway (note 01 section 6); a Babylon `WebGPUEngine` device share removes the copy (`GpuContext.from(device)` exists; R-23) | per-frame readback up to ~250k nodes, then every 2-4 frames (Q-7) |
| R-5 | lavapipe is ~350x slower on O(n^2) kernels; the default lane could exceed its budget as the suite grows | medium / medium | `gpuScale` fixture scaling; per-file budget (~2 min); shard the node project; heavy sizes only in the benchmarks / `node-limits` on the GPU lane; the 1M 200-iteration exact-vs-grid run only in the nightly benchmark job | 15-minute lane target (T-12) |
| R-6 | The hosted GPU lane is unavailable (runner group quota, image change, billing limit hit) or its cost drifts | medium / low | its own workflow (`gpu.yml`), never a job of `CI`, so an unavailable lane cannot delay a release or a coverage publish (12.1); nightly skipped on quiet days; label + same-repo gating; the spending limit; fork PRs never reach it; `machine.dev` as escape hatches; T4 hosted runners only if the org plan changes | dev-box runner now (Q-3) |
| R-7 | `webgpu` cannot be upgraded past 0.4.0 on the 22.04 container | medium / low | `Readback` never relies on `device.destroy()` unmapping; P-ENV moves both images to 24.04 in one change; the peer range already admits 0.6.x | stay on 0.4.0 through P3 (Q-4) |
| R-8 | Subgroup size is not constant (32 NVIDIA, 8 lavapipe, 4 SwiftShader; 8-32 on Intel Xe, 32-64 on AMD, per shader, possibly partial); a kernel assuming a compile-time size silently miscomputes | high / high | every subgroup kernel reads `subgroup_size` / `subgroup_invocation_id` at runtime and has a workgroup-memory twin (D16); the three-adapter spread in CI plus the faked `min != max` table is the test; the twins run in-process and in the no-subgroups CI pass over the whole node project | always ship the non-subgroup variant |
| R-9 | Windowed bindings for the 10M tier are supported only by row-walking kernels until P8; frontier algorithms on > 33M arcs at default limits throw `E_TOO_LARGE` | low / medium | raised limits on discrete GPUs make windows rare; `E_TOO_LARGE` carries the numbers; window-aware advance lands in P8 | accept until P8 |
| R-10 | Dawn-node's `uniform_buffer_standard_layout` masks a layout bug that only Chromium rejects | medium / medium | `UniformBlock` generates padded structs; the compile matrix runs on both runtimes; the negative test | generated layouts only (D20) |
| R-11 | Float accumulation error (f32 tile sums, PageRank pull, BC over many sources) exceeds the tolerances on adversarial graphs | medium / low | tree-shaped partials, Kahan in hub loops, documented tolerances scaled by iteration count; f64 CPU results remain the reference; `precision: "f32"` on results | keep tolerances; document (Q-24) |
| R-12 | Behaviour change: `weight` becomes live for FA2 through the snapshot; parallel arcs sum in the gather | medium / low | documented in the L1 / E1 changelogs and the design (14.3 line 4037, 14.4 `weightFromPath`) | accept (Q-10) |
| R-13 | `browser.close()` hangs after GPU work on the NVIDIA path | high / low | job-level timeouts, `timeout -k 10 600` around the browser step with the JSON reporter, exit 124 accepted iff every test passed (11.6), `fileParallelism: false` | hard kill backstop that cannot turn a green run red |
| R-14 | The element's synchronous frame loop with a coalescing bridge changes perceived layout speed versus ngraph (one physics step per `step()`) | medium / low | `iterationsPerStep = stepMultiplier`; settle semantics unchanged; the frame-loop test; stories compare feel | coalesce, document |
| R-15 | Timestamp-query quantisation (Chromium 100 us; Dawn-node unquantised, 1,024 ns ticks [M]) makes browser per-kernel profiling noisy | low / low | gates use wall-clock around `await step()`; the profiler is a benchmark aid; per-kernel profiling is a Node activity | wall-clock gates |
| R-16 | Louvain on the GPU gives only 2-10x and loses parallelism at coarse levels | high / low | scheduled last; expectation stated up front; no in-package CPU handoff -- the caller chooses the CPU package for small graphs | ship with the measured number (Q-14) |
| R-17 | Design 14.5 says "browser-only vitest project"; leaving it unamended confuses W1 | certain / low | amend 14.5, 14.6 and 16.7 in the W1 PR | amend (Q-2) |
| R-18 | A consumer that never calls `release` leaks GPU memory (nothing is freed by GC) | medium / medium | per-snapshot record; `release` from `snapshot-replaced` is in the design; `residentBytes`, `stats()` and the once-only warning above `warnUnreleasedSnapshots` | counter + warning |
| R-19 | Dawn-node process lifetime: a CLI that keeps the `GPU` object reachable never exits | low / low | `createNodeGpu().dispose()` drops the reference; vitest `forks` kills workers | dispose in `afterAll` |
| R-20 | Vitest 3 -> 4 changes the browser provider API | medium / low | flags in one exported constant; the monorepo already pins overrides for both majors | follow the monorepo |
| R-21 | The 1M-node interactive tier depends on the element rendering 10^6 meshes, which it cannot today (note 01 section 6) | certain / low | 1M is a Node batch tier until instanced rendering exists; the layout itself is measured (T-6) regardless | batch tier now |
| R-22 | Unverified platform facts (Firefox / Safari exposure of `subgroups` / `timestamp-query`; lavapipe on an actual GitHub runner; the exact Vitest 3 launch spelling; `pool: "threads"` with the Dawn addon; mobile / Safari 26 behaviour). SETTLED by the review probes and no longer open: a default Dawn-node device carries the spec-default limits with offset alignments 256 [M]; Dawn-node timestamps are unquantised [M]; an over-limit indirect dispatch runs nothing without an error [M]; same-pass write-then-indirect works [M]; `uncapturederror` fires synchronously under Dawn-node [M]; an adapter is consumed by one `requestDevice` [M]; `backend=null` yields an adapter under webgpu@0.4.0 [M] | medium / low | P0 / P1 verify each remaining item and record the answer in the package `CLAUDE.md` "Verified Platform Facts"; the finalize kernel makes the indirect question moot either way; `forks` is the verified pool | verify at G0 / G1 (Q-16) |
| R-23 | graphty-element's `useWebGPU` render path (present, unwired, 1.2) is switched on while an accelerator is injected: two devices on one adapter, the layout readback still copies through the CPU, and the element's Babylon device is not the accelerator's | low / low | nothing breaks (two devices are legal); `GpuContext.from(engine._device)` is the wiring point when the element exposes its engine's device, and it removes the readback (Q-7) | document; wire at the element's `WebGPUEngine` milestone, not in v1 |
| R-24 | Near-field resampling of an over-capacity cell changes a node's force estimate every iteration; with `speedEfficiency` floored at 0.05 such nodes may never reach zero displacement and delay `settled` (7.7) | medium / low | measured at G4 on the hub-cell fixture; mitigation if it shows: raise `nearMax` adaptively while `maxCellOccupancy > nearMax` persists for `settleWindow` iterations, or bound the near-field force sum inside G7 (cosmos's placement, D25) | measure at G4, decide in the P4 record |
| R-25 | The T4 is a shared-tenant cloud GPU: run-to-run benchmark medians vary, and a nightly that opens tracking issues for noise trains people to ignore it | medium / low | medians of 5 runs; the 3x regression threshold against the T4's OWN baseline; `gpu-report.js` records clocks and utilisation so an outlier is explainable; an issue only after two consecutive nightly failures (T-13, 12.6); the T-table targets are measured on the dev box by hand, never on the T4 | tolerate, never fail on a single slow run |

### 14.2 Open questions (each with the recommended default; silence means the default stands)

| Id | Question | Recommended default |
| --- | --- | --- |
| Q-1 | DECIDED 2026-09-14: the published laws (Jacomy 2014 / Gephi / cuGraph) are the default, with a `compat: "networkx"` option that reproduces NetworkX `forceatlas2_layout`; the port's own variants are dropped (7.2, D5, DEPARTURE-3). Gravity toward the centroid (port, NetworkX) confirmed by the owner; `GRAVITY_CENTER: 1` (origin, Gephi / cuGraph) stays an override for tests. The 7.2 table as amended is the G0 sign-off artefact (D21). | -- |
| Q-2 | Node-first testing departs from design 14.5 "browser-only vitest project". | Amend 14.5, 14.6 and 16.7 when this plan is accepted |
| Q-3 | DECIDED 2026-09-14: no self-hosted runner; a paid hosted GPU runner. The plan uses GitHub's own GPU larger runner (T4; needs the Team plan, ~$4 per user per month plus $0.052 per minute), with `machine.dev` as the drop-in alternative if the plan upgrade is unwanted (12.1, 12.4). Owner: "whatever is easier and more maintainable" -> GitHub Team + T4 (one vendor, one bill, GitHub-managed VMs, a runner group the monorepo joins at W1, no third-party account or runner lifecycle to maintain); `machine.dev` stays documented as the fallback only. | -- |
| Q-4 | `webgpu@0.4.0` pin versus moving the dev container / runner image to Ubuntu 24.04. | Stay on 0.4.0 through P3; move both images at P-ENV in one change |
| Q-5 | Grid pyramid (option A) as the primary approximation versus a GraphWaGu-style cluster tree (option B). | A first; B only if G4's clumpy fixtures fail the error bound |
| Q-6 | `exactMaxNodes` default fixed now at 16,384 or re-fixed by measurement? (cited from D7, 3.1, 7.8, 7.14) | Keep 16,384 as a conservative default until G3 (the measured curve predicts 32,768); re-fixed at G3 by the 7.8 rule; `calibrateLayout()` for other devices |
| Q-7 | Readback at 1M nodes in the browser (~30 ms per batch) is frame-limiting. (cited from 7.19, R-4) | `iterationsPerStep` auto-raise above 250k nodes; the structural fix (sharing the Babylon `WebGPUEngine` device, R-23) is not scheduled; the rendering-side notes are in 14.3 |
| Q-8 | Settlement threshold for FA2: relative `1e-3` of the RMS layout radius over 10 iterations (7.17), with `maxIter` as the hard stop? (cited from 7.14, 7.17) | Yes; tune from stories |
| Q-9 | The `spring-electrical` preset (7.20) is BUILT in P5 as its own `SimulationType`; should the element also ROUTE its default `ngraph` layout to it above a node-count threshold? (cited from 7.20, 9.3) | Build the preset (P5); do not route `ngraph` to it in v1 -- the GPU layout is selected by type (`forceatlas2` / `spring` / `spring-electrical`); the app decides any threshold later |
| Q-10 | `weight` becomes LIVE for FA2 through the snapshot (inert in the element today). | Accept the documented behaviour change at L1 / E1 |
| Q-11 | Directed snapshots passed directly to a GPU layout in Node. (cited from 7.1 through `load()`'s argument checks, 5.7) | `E_SNAPSHOT` ("pass `toUndirected().snapshot`"), matching `toLayoutSnapshot`; graphty-element always passes the undirected copy and releases it (4.5) |
| Q-12 | Labels renumbered on the CPU in first-seen order (identical `groups()`) cost O(n) per CC readback. | Do it (parity is worth ~1 ms at 1M nodes); `renumber: false` returns raw roots |
| Q-13 | Betweenness at 1M nodes is seconds to tens of seconds even sampled (2-20 s for 256 sources by the 8.4 model). | `sources` / `k` become part of the SHARED `BetweennessCentralityOptions` at A2 so both paths accept them; `onProgress` and `signal` are reachable only through the GPU package's own functions; the element's adapter defaults to exact for n <= 10k and 256 sampled sources above 50k (a product decision for E1 / W2), which is the right default at either figure because exact BC is O(n m) |
| Q-14 | Louvain partitions will not match the CPU's; parity is a modularity band. | Accept; the adapter labels the result "GPU Louvain" |
| Q-15 | Cancellation semantics: `E_ABORTED` on `signal.abort()`; a submitted batch still completes on the device and its staging slot is returned when its `mapAsync` resolves (4.4). (cited from 5.7) | Accept |
| Q-16 | Unverified platform facts (R-22; the seven settled by the review probes are struck). (cited from 2.6, R-22) | P0 / P1 verify each remaining item and record the answer in the package `CLAUDE.md` |
| Q-17 | Performance numbers in 7.21 and 10 are extrapolations except the cited probes (the exact-tile curve and the sorted-order grid kernels are now measured). (cited from 7.21, 10.3) | The T-table gates are the real targets; the tables are revised from `benchmarks/results/` |
| Q-18 | Where do the plan and the research live? `tmp/` is not gitignored at the root today (note 07 section 5). (cited from 3.1) | This plan stays at `design/webgpu-acceleration-plan.md` and moves to `graphty-monorepo/design/webgpu/` at W1; the seven notes AND the three drafts (the "(graft: ...)" citations point at them) are committed under `packages/webgpu-graph-algorithms/docs/research/`; the review probes and their logs under `docs/research/review/`; cloned repos stay under a gitignored `tmp/`. On acceptance the "Review notes" subsections, the "(graft: ...)" and "judge" annotations and the Review log move into `docs/research/review-log.md` and the plan keeps only decisions |
| Q-19 | Should graphty-element get an `accelerator: "auto"` convenience (optional peer + isolated dynamic import) in E1, or only the injected property with app-side detection? | Property + app detection first; "auto" later |
| Q-20 | Package manager for the staging repo: pnpm (`packages/` already uses it) or the root scaffold's npm? (cited from 12.3, P0) | pnpm; the root scaffold is deleted at P0; `pnpm/action-setup` reads `packages/package.json` |
| Q-21 | 3D grid cap 128^3 (38 MB pyramid) versus 160^3 (75 MB); lower to 64^3 on integrated GPUs? (cited from 7.7, 7.14, 3.3) | 128^3; `gridMax3D` is an option; `calibrateLayout()` may suggest 64 |
| Q-22 | Should the browser smoke include a Firefox or WebKit instance later? (cited from 11.6, R-22) | Not until those ship the needed features on Linux CI; Chromium only; no claims about mobile / Safari until someone runs the story on a Mac |
| Q-23 | Interactive topology change with a live simulation: where do new nodes appear? | `load(next, positions)` seeds NaN rows randomly inside the current bounding box; "place at the neighbours' centroid" is a later element option |
| Q-24 | f32 versus f64: GPU scores are f32 (PageRank parity 1e-5; BC and the one-iteration force parity 1e-4, DEPARTURE-6); sigma path counts are u32 with an overflow flag. | Document the tolerances in the accelerator interface docs and amend design 16.2 at W1; EVERY score result carries `precision: "f32"` (3.3) so the element can label it; no f64 on the device (WGSL has none) |
| Q-25 | `nodeSize` / `adjustSizes` in v1? | No: `E_UNSUPPORTED` at runtime when set (the type stays the CPU's); add after the CPU rewrite fixes the sign |
| Q-26 | DECIDED 2026-09-14: optional peer dependencies on `@graphty/algorithms` / `@graphty/layout` with `import type` of the real interfaces (D27). Accepted consequence: the published d.ts references the two packages, so a type-checking consumer installs them (optional peers; documented in the README's Node recipe); mirrors only until W1. | -- |
| Q-27 | Residency semantics for `withColumns()` siblings, which share the core and the serial: one residency unit (releasing any sibling releases the core; a live sibling gets `E_RELEASED`), or count distinct snapshot objects per record and destroy at zero? | One unit (4.1): the element never releases a sibling of a live snapshot, and object counting would keep buffers alive for a sibling nobody can enumerate; a Node script that wants per-sibling lifetimes keeps distinct snapshots |
| Q-28 | DECIDED 2026-09-14: public `graphty-org/webgpu-graph-algorithms`, created and pushed in P0 (the owner commits; this plan's author never does); npm trusted publishing: the placeholder `@graphty/webgpu-graph-algorithms@0.0.0` (the same "OIDC trusted publishing setup package" as graph-format's and graph-io's 0.0.0, published from the dev box with `--provenance=false` and an OTP) was published on 2026-09-14 and is committed at `packages/webgpu-graph-algorithms/`; the trusted publisher itself needs an existing package and npm >= 11.15 (`npx -y -p npm@11 npm trust github @graphty/webgpu-graph-algorithms --repo graphty-org/graphty-monorepo --file release.yml --allow-publish`, the monorepo's `release.yml` being what publishes `0.1.0` at P10) and is the owner's to run, together with the same command for graph-format and graph-io. | -- |
| Q-29 | nx release: accept that the devDependency on `@graphty/algorithms` / `@graphty/layout` makes every release of those packages patch-bump and publish this one (`updateDependents: "auto"`), or add a `release.groups` entry with `updateDependents: "never"` for this package? | Accept (9.8): it is how every dependent in the monorepo already behaves and nothing breaks; the `release.groups` entry is a one-line root touch point if the noise is unwanted |
| Q-30 | DECIDED 2026-09-14 (owner: "shouldn't node mass be part of the graph-format package?"): yes -- graph-format already defines the node column roles `mass`, `size`, `fixed` and `position` (`packages/graph-format/src/types/columns.ts` lines 152-190, `KnownColumnRole`), so node vectors resolve by ROLE on both paths (D28): the simulation reads `nodes.byRole("mass")` at `load()`, a `Float32Array` or a column name is accepted, and the legacy `Record` form is converted into the role column by the caller (graphty-element at engine creation, 9.4 item 10; `@graphty/layout`'s `resolveNodeVector` on the CPU path). No layout helper is duplicated and no interface changes. | -- |
| Q-31 | `workspace:^` instead of `workspace:*` for the graph-format dependency, and the same correction for graph-io and design 13.5 rule 3 (pnpm publishes `workspace:*` as an exact pin)? | `workspace:^` here (2.5); propose the correction to graph-io and the design at W1 -- an exact pin beside a `^1.0.0` peer gives an app on a newer format two copies |
| Q-32 | Grid extent constant `extentFactor = 6` and the finest-grid cap `gridMax2D = 512` (cosmos's WebGL texture limit carried into compute): re-check `extentFactor` in {4, 6, 8} and `gridMax2D` in {512, 1024, 2048} at 1M in the P4 decision record? (2048 costs ~105 MB and 54 more far-field evaluations per node; a larger G also shrinks the near-field cells the isolated-node fixture stresses) | Yes, measured at G4 on the uniform, clumpy and isolated-node fixtures; defaults stay 6 / 512 until the record says otherwise |

---------------------------------------------------------------------------

### 14.3 Deferred: rendering at scale (owner notes of 2026-09-14, not scheduled)

Scope decision: the interactive target for v1 is a few hundred thousand
nodes in the browser (7.21: 60 fps at 100k with a per-frame readback;
per-frame readback up to ~250k nodes, then every 2-4 frames, R-4), which this
plan covers as written. The two items below are what "very large" (1M+)
animated layouts need on the RENDERING side; they belong to graphty-element,
are recorded here so the GPU package keeps the seams open, and are on no
phase's critical path.

1. Thin instances for nodes. graphty-element draws nodes as one Babylon
   `InstancedMesh` per node (`meshes/MeshCache.ts` lines 28-47:
   `mesh.createInstance(name)`) and copies `x, y, z` into each mesh's
   `position` every frame (`Node.ts` lines 207-211), so the per-frame cost is
   n JavaScript objects plus n world-matrix updates; that, not the layout,
   is the ceiling at 10^5+ nodes (R-21). There is no `thinInstance*` call in
   `graphty-element/src` today; the one attempt (arrow heads,
   `meshes/EdgeMesh.ts` line 385: "thin instances were causing 1,147 ms
   bottleneck, 35x slower than direct position updates") updated instances
   one at a time through the per-instance API, which is the slow path. The
   scalable path is one mesh per node style with
   `thinInstanceSetBuffer("matrix", buf, 16, false)` (or a custom 3-float
   attribute and a vertex shader that builds the matrix) written ONCE per
   frame from the same stride-3 array the simulation reads back into (7.19);
   the bridge already produces that array, so the GPU package needs no
   change. Picking, per-node styling and metadata move from `InstancedMesh`
   properties to per-instance attribute buffers; that is the element's work,
   and it is what makes the 262k-1M rows of 7.21 drawable.
2. Zero-copy rendering. With thin instances in place the remaining per-frame
   cost is the 12n-byte readback (Q-7, R-4). It disappears when the element
   renders with Babylon's `WebGPUEngine` (the branch exists but is unwired,
   1.2) and shares its device with the accelerator (`GpuContext.from(
   engine._device)`, 2.2; R-23): the simulation's `scenePositions` buffer is
   then bound as the thin-instance attribute buffer directly, with no
   `mapAsync` and no `set`. What the GPU package keeps true so that this is
   a small change later: `scenePositions` stays a plain `STORAGE | COPY_SRC`
   buffer with a stable layout (stride 3 f32, scene units, 7.18) that a
   vertex stage can read (`VERTEX` usage is added by an option when the
   device is shared); `setPosition` / `setFixed` work with no CPU copy of
   the positions; `flush()` remains the synchronisation point for anything
   that must see the array on the CPU (serialisation, screenshots).

Ordering when the element takes this up: thin instances first (they pay off
on the WebGL engine today), device sharing second (it needs the element's
WebGPU render path).

---------------------------------------------------------------------------

## 15. References

Local (read-only unless stated):

- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` -- sections 10 (GPU contract, lines 2320-2553), 13.5 (versioning rules, 3638-3680), 14.2 (result conventions, line 3738), 14.3 (`LayoutSimulation`, layout ports, 3959-4047), 14.4 (graphty-element ownership, `snapshot-replaced`, 4048-4211), 14.5 (WebGPU package move-in, 4212-4243), 14.6 (landing order, 4245-4278), 15 (performance model, 4279-4449), 16 (testing, 4450-4657).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/01-layout-needs.md` .. `07-format-api-and-conventions.md` -- the seven research notes this plan is built on (each carries its own URL list); `draft-A.md`, `draft-B.md`, `draft-C.md` -- the three drafts synthesised here.
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/src/index.ts` (the exact export list), `src/types/columns.ts`, `src/types/snapshot.ts` (`SnapshotFlags` lines 161-181), `src/snapshot/views.ts`, `src/snapshot/derived.ts` (`renumberPartition` line 1155), `src/builder/arena.ts`, `src/util/mask.ts`, `src/columns/column.ts` -- the implemented format API.
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-upload.test.ts` (`acquire()` lines 38-63, the 65,535 dispatch assertion line 186-191, the upload contract lines 254-666), `gpu-contract.test.ts` -- Dawn-in-Node device acquisition and the upload contract audit.
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-io/package.json`, `scripts/*.js`, `tsconfig*.json`, `vitest.config.ts`; `packages/README.md` (move checklist lines 44-198); `packages/pnpm-workspace.yaml`; `packages/move/root-touch-points.diff` -- the package skeleton mirrored and the monorepo touch points.
- `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md` -- the four Chromium flags, the `libEGL.so.1` root cause, Chrome 145 `powerPreference`, appendix D.
- `/home/apowers/Projects/webgpu-graph-algorithms/CLAUDE.md` -- "Never create fallbacks if WebGPU isn't supported".
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/*.mjs` -- the measurements cited (`dawn-perf.mjs` exact tile 1.11 ms at 20k NVIDIA / 388 ms lavapipe; `dawn-latency.mjs`, `chromium-latency.mjs` round trips; `bench-node.mjs`, `bench-browser.mjs` gather kernel and checksums).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/` -- the 2026-09-14 adversarial review: six `*-review.md` / `*-verdicts.md` pairs (PERF, DESIGN, MAINT, INTEG, VERIFY, COMPLETE) and `probes/` with the scripts and logs cited in this plan as [M]: `exact-tile.mjs` (the 7.6 curve), `near-field-order.mjs` / `far-field-order.mjs` (sorted-order dispatch, `vec4f` positions), `grid-occupancy.mjs` (isolated-node collapse), `clamp-expansion.mjs` / `verify-clamp-throttle.mjs` (the displacement clamp), `verify-wgsl-mixing.mjs` (operator precedence), `uniformity-probe-nvidia.log` (uniform control flow), `design-probe-nvidia.log` / `design-probe-llvmpipe.log` and `dawn-facts.mjs` (default-device limits, timestamps, indirect dispatch, adapter consumption, `backend=null`), `uncaptured-order-probe.mjs` (synchronous `uncapturederror`, consumed adapter), `binding-mismatch-probe.mjs` / `maint-sync-probe.mjs` (layout and override mismatches), `serial-sharing.mjs` (sibling serials, undirected copies), `arena-bytes.mjs` (arena sizes), `verify-checksum-cross-adapter.mjs` (f32 cross-adapter noise), `verify-uncapturederror.mjs`; `wgsl-spec.html`, `wgsl-tr.html`, `webgpu-spec.html` (the specs grepped).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/*` and `papers/*` -- cloned sources and extracted paper texts behind the [P] facts (cosmos, GraphWaGu, cugraph, cugraph-algos, gunrock, gapbs, gephi, d3, d3-force-webgpu, GraphGPU, analytics, antv-webgpu-graph, hybrid_BC, cuda-ffi, wgpu and three.js CI files).
- `/home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/forceatlas2.ts` (lines 26-42, 184-230, 233-434), `fruchterman-reingold.ts`, `layout/src/utils/random.ts`, `layout/test/forceatlas2-layout.test.ts` -- the CPU layouts, the LCG and the behaviour pins reproduced.
- `/home/apowers/Projects/graphty-monorepo/graphty-element/src/layout/{LayoutEngine,NGraphLayoutEngine,ForceAtlas2LayoutEngine,D3GraphLayoutEngine}.ts`, `src/managers/{LayoutManager,UpdateManager,DataManager,RenderManager}.ts`, `src/algorithms/{Algorithm,PageRankAlgorithm,BetweennessCentralityAlgorithm,LouvainAlgorithm}.ts`, `src/config/GraphBehavior.ts`, `src/ai/providers/index.ts`, `vite.config.ts`, `package.json` -- the element seams changed in 9.4.
- `/home/apowers/Projects/graphty-monorepo/algorithms/src/algorithms/centrality/pagerank.ts` (line 83), `algorithms/src/optimized/*` -- the sync CPU entry points and the code the design deletes.
- `/home/apowers/Projects/graphty-monorepo/.github/workflows/ci.yml` (shard matrix lines 236-351, Playwright cache 413-427, `all-checks` 705-738), `release.yml` (lines 36-52), `coverage.yml`, `nx.json`, `commitlint.config.js`, `tools/merge-coverage.sh`, `tools/prepush.sh`, `package.json` (lines 130, 148, 158, 160) -- the CI shard matrix, release process and version pins slotted into in 12.5.

External (URLs; what each was used for):

- https://github.com/atoms-org/cuda-ffi -- self-hosted GPU runner pattern (owner-supplied): `runs-on: cudaffi-gpu-runner`, `container: { options: --gpus all }`, push-only triggers.
- https://github.com/cosmosgl/cosmos -- grid-pyramid many-body force, exact path below 4,096 points, Horvitz-Thompson near field, integer hashes, per-step timings and failure modes (owner-supplied; source read, MIT). https://cosmograph.app/examples , https://pypi.org/project/cosmograph/ -- the product and Python bindings over cosmos.gl (owner-supplied; only their concept pages were fetched by the notes; no algorithmic content beyond cosmos.gl).
- https://github.com/harp-lab/GraphWaGu -- WebGPU FR + Barnes-Hut: WGSL radix sort, Hilbert codes, level-wise tree build, i32 fixed-point bbox atomics, CSR gather attraction (owner-supplied; source and papers read, MIT). https://www2.evl.uic.edu/documents/pacificvisgraphwagu.pdf , https://stevepetruzza.io/pubs/graphwagu-2022.pdf -- the GraphWaGu papers (timings on an RTX 4070 Laptop and RTX 2060, crossovers).
- https://github.com/jaredmcqueen/analytics -- WebGL1 O(n^2) FR; cautionary only (GPL-3, "1M nodes at 60 fps" not credible for simulation) (owner-supplied).
- https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal -- Merrill, Garland, Grimshaw 2011: scan-based frontier expansion, gather tiers, duplicate culling, expand / contract couplings (owner-supplied).
- https://dl.acm.org/doi/10.1145/3230485 (403 on fetch) / https://davidbader.net/publication/2018-mb/2018-mb.pdf -- McLaughlin, Bader, "Accelerating GPU Betweenness Centrality": work-efficient vs edge-parallel BC, atomic-free dependency accumulation, sampling (owner-supplied).
- https://cse.buffalo.edu/tech-reports/2023-06.pdf -- Kumar, dense-matrix BC; excluded as a negative result for sparse graphs (owner-supplied).
- https://developer.nvidia.com/discover/cluster-analysis -- nvGRAPH-era spectral / multilevel overview; background only (owner-supplied).
- https://github.com/rapidsai/cugraph -- FA2 (Burtscher BH port, `fa2_kernels.cuh`, `barnes_hut.cuh`), PageRank pull, BFS direction-optimizing constants, SSSP near-far, BC batching, Louvain, k-core, triangle counting (Apache-2.0; read as design input, not copied). https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.force_atlas2/ -- FA2 parameter defaults and the iteration guidance.
- https://github.com/gunrock/gunrock -- `block_mapped` advance, `neighborreduce`, `from_coo`; push PR / BC as float-atomic counter-examples.
- https://raw.githubusercontent.com/sbeamer/gapbs/master/src/cc.cc -- Afforest connected components.
- https://scottbeamer.net/pubs/beamer-sc2012.pdf -- direction-optimizing BFS alpha / beta.
- https://escholarship.org/content/qt8qr166v2/qt8qr166v2.pdf -- Davidson et al. 2014 near-far SSSP and the ownership dedupe.
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679 -- ForceAtlas2 (Jacomy et al. 2014) formulas (CC-BY).
- https://raw.githubusercontent.com/gephi/gephi/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java (and `ForceFactory.java`, `Region.java`) -- Gephi formulas and defaults (GPL / CDDL: formulas only, no code).
- https://raw.githubusercontent.com/networkx/networkx/main/networkx/drawing/layout.py -- NetworkX `forceatlas2_layout` / `estimate_factor` (the CPU port's origin and its deviations).
- https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf , https://liacs.leidenuniv.nl/~takesfw/pdf/exploiting-gpus-fast.pdf -- Burtscher-Pingali Barnes-Hut and Brinkmann et al. 2017 timings; why the locked tree is not portable.
- https://arxiv.org/html/2501.19004 , https://arxiv.org/html/2608.01503 -- nu-Louvain and Gilbert-Madduri: GPU Louvain expectations.
- https://github.com/jamescarruthers/d3-force-webgpu , https://github.com/drkameleon/GraphGPU -- WebGPU layouts with racy edge-parallel scatters; counter-examples; d3-force-webgpu's tiled kernel shape.
- https://raw.githubusercontent.com/d3/d3-force/main/src/manyBody.js -- the d3 1/d law cosmos reproduces.
- https://arxiv.org/abs/2303.03964 -- t-FDP (FFT far field), noted as a non-pursued option.
- https://gpuweb.github.io/gpuweb/ , https://gpuweb.github.io/gpuweb/wgsl/ -- limits, `requestDevice`, `dispatchWorkgroupsIndirect`, error scopes, device loss; WGSL atomics (6.2.8), recursion (11.4), overrides (7.2.2), uniform layout (14.4.5), `bool` not host-shareable (6.5.2), subgroups (17.12).
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status , https://developer.chrome.com/blog/new-in-webgpu-120 , -121 , -128 , -134 , https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/ , https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ -- browser availability and feature timeline (subgroups Chrome 134, timestamp quantisation, `adapter.info`, `maxStorageBuffersPerShaderStage` 10 as an adapter maximum).
- https://registry.npmjs.org/webgpu , https://github.com/dawn-gpu/node-webgpu , https://github.com/dawn-gpu/node-webgpu/issues , https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md , https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/native/Toggles.cpp -- the `webgpu` npm package versions, glibc requirements, `create()` options, `adapter=llvmpipe`, Dawn toggles.
- https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/.github/workflows/build.yml , https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/test/webgpu.js , https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/workflows/ci.yml , https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/actions/install-mesa/action.yml , https://raw.githubusercontent.com/mrdoob/three.js/dev/.github/workflows/ci.yml , https://raw.githubusercontent.com/mrdoob/three.js/dev/test/e2e/puppeteer.js -- lavapipe / SwiftShader CI patterns, budgets, the `browser.close()` hang.
- https://github.com/chromium/chromium/blob/main/docs/gpu/swiftshader.md , https://groups.google.com/a/chromium.org/g/blink-dev/c/yhFguWS_3pM , https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips , https://github.com/jasonmayes/headless-chrome-nvidia-t4-gpu-support -- SwiftShader flags, `--enable-unsafe-swiftshader`, headless Chromium on NVIDIA.
- https://docs.github.com/en/actions/reference/runners/larger-runners , https://docs.github.com/en/enterprise-cloud@latest/actions/concepts/runners/larger-runners , https://docs.github.com/en/billing/reference/actions-runner-pricing , https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/ , https://github.blog/changelog/2025-12-16-coming-soon-simpler-pricing-and-a-better-experience-for-github-actions/ , https://github.blog/changelog/2026-01-01-reduced-pricing-for-github-hosted-runners-usage/ -- GPU runner spec, price, plan gate, and the 2026 self-hosted pricing note.
- https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners , https://docs.github.com/en/actions/reference/runners/self-hosted-runners , https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow , https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-configuration-for-a-just-in-time-runner-for-a-repository , https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#configuring-required-approval-for-workflows-from-public-forks , https://docs.github.com/en/actions/reference/limits -- ephemeral / JIT runners, security guidance, fork approval settings, and the 24-hour queue limit for a job waiting on an offline self-hosted runner (12.1).
- https://raw.githubusercontent.com/actions/runner-images/main/images/ubuntu/Ubuntu2404-Readme.md , https://launchpad.net/ubuntu/noble/+source/mesa , https://launchpad.net/ubuntu/jammy/+source/mesa , https://launchpad.net/ubuntu/noble/+source/glibc -- runner image contents and package versions (Mesa 25.2.8, glibc 2.39).
- https://github.com/orgs/community/discussions/190443 , https://github.com/GDeLaurentis/docker-gpu-runner-for-github-actions -- NVIDIA GPU on self-hosted runners, an example runner container.
- https://cirun.io/ , https://runs-on.com/runners/gpu/ , https://machine.dev/docs/platform-specifications/gpu-runners/ -- third-party GPU runner options (not evaluated beyond their pages).
- https://betatim.github.io/posts/github-action-with-gpu/ , https://davesnider.com/posts/gputests -- practitioner reports (label-gated GPU jobs; headless Chromium on a T4 needing Vulkan flags).
- https://github.com/dorny/paths-filter -- optional path gating for the GPU job (not used in v1).
- https://vite.dev/guide/assets -- `?raw` imports (rejected in favour of `.wgsl.ts`).
- https://docs.npmjs.com/cli/v10/configuring-npm/package-json -- optional peer dependency semantics.
- https://pnpm.io/workspaces -- `workspace:*` is published as an exact version and `workspace:^` as a caret range (2.5, Q-31).
- https://www.npmjs.com/package/@antv/webgpu-graph , https://g.antv.antgroup.com/en/api/gpgpu/webgpu-graph -- @antv/webgpu-graph (cautionary; the `updatePred` two-pass idea).
- https://github.com/michaelsutton/afforest/blob/master/README.md -- the Afforest repository README (secondary source).

Not fetched / not verified (carried over from the notes so the plan does not
overstate): the GraphWaGu README's Google-Drive paper PDF; Cosmograph product
pages in depth; the RAPIDS cuGraph FA2 performance blog (403); Naim et al.
2017 (paywalled); the Gunrock paper; RunsOn pricing; GitHub GPU runner
concurrency limits; Firefox / Safari optional-feature exposure; `pool:
"threads"` with the Dawn addon; lavapipe on a real hosted runner (only the
local container run); whether `webgpu@0.4.0` honours `backend=` other than
`null` (only `adapter=` and `backend=null` were exercised); every [X] number
in sections 7.21 and 10 (Dawn-node timestamp quantisation is now verified:
unquantised, 2.6).

---------------------------------------------------------------------------

## Review log

Date: 2026-09-14. A second review of this document by six adversarial lenses,
each followed by an independent verifier that re-read every cited line,
opened every cited file, re-ran the reviewer's probes and added its own
(`tmp/webgpu-plan/review/`, section 15). Only findings that survived the
verifier were applied; the verifier's downgrades and narrowings were applied
in their narrowed form. The reviewers' own line numbers refer to the document
BEFORE this pass.

| Lens | Reviewer findings | Confirmed | Downgraded (applied as narrowed) | Refuted (not applied) | Verifier additions (applied) |
| --- | --- | --- | --- | --- | --- |
| PERF (performance realism) | 16 | 12 | 3 (PERF-6, -9, -13) | 1 (PERF-14: the 12 MB Chromium readback figure is correctly labelled [X]) | 2 (PERF-MISSED-1, -2) |
| DESIGN (design confidence, WGSL feasibility) | 21 | 19 | 2 (DESIGN-10, -15) | 0 | 4 (DESIGN-MISSED-1..4) |
| MAINT (maintainability) | 20 | 10 | 9 (MAINT-2, -3, -5, -6, -8, -9, -13, -15, -17) | 1 (MAINT-12: dropping the DOM lib would silently weaken `writeBuffer` typing) | 6 (MAINT-M1..M6) |
| INTEG (integration, API conformance) | 23 | 20 | 3 (INTEG-7, -8, -16) | 0 | 5 (INTEG-M-1..M-5) |
| VERIFY (verifiability, testing, CI) | 21 | 15 | 6 (VERIFY-1, -2, -8, -10, -13, -14) | 0 | 5 (VERIFY-MISSED-1..5) |
| COMPLETE (completeness, internal consistency) | 22 | 18 | 4 (COMPLETE-4, -6, -7, -18) | 0 | 6 (COMPLETE-M1..M6) |
| Total | 123 | 94 | 27 | 2 | 28 |

Surviving findings: 149 (94 + 27 + 28). Applied: 149. Rejected: 0. Two
findings whose fixes pulled in opposite directions were reconciled rather
than chosen between: COMPLETE-5 (move the indirect `finalize` kernel out of
P2 to P8 / P7) and PERF-11 / DESIGN-12 (the grid's hub-cell tier needs an
indirect dispatch in P4) -- the finalize kernel lands in P4, out of P2 as
COMPLETE-5 asked and before P8 as the grid needs; MAINT-14 (replace the
`GRAPHTY_GPU_NO_SUBGROUPS` variable with a context option) and D19 / 12.2
(one policy variable read by the test setup) -- the variable stays as a test
setup knob that maps onto the existing `optionalFeatures` option, and the
twins are additionally tested in-process (11.3). Verifier corrections to the
reviewers' evidence (a wrong line number, an overstated sub-claim) were taken
as the verifier wrote them; where a reviewer's fix was impossible under the
plan's own rules (MAINT-18's CPU helper inside a GPU result, D3) the
verifier's alternative was used.

Fix verification and repair (same date). An independent check pass re-read
the document after the 149 fixes: 149 / 149 applied, 0 wrong rejections, 4
regressions introduced by the fixes and 1 pre-existing consistency gap, all
five repaired by the checker (`tmp/webgpu-plan/review/check-report.md`); 6
further consistency observations were handed to a repair pass, which acted
on 4 and recorded the 2 that need no edit. The CHECK-R / CHECK-C / REPAIR
rows at the end of the table below are those edits; the totals above are
unchanged by them (no finding was added or removed) and the departure count
in 1.5 is now seven (DEPARTURE-7, the device position layout under design
14.3's stride-3 sentence, which the check pass found undeclared). Not
edited, by design: the low-degree tier of section 6 row 3 spans `[midEnd,
n)` where design 10.1 writes `[midEnd, lowEnd)` -- the plan says so
explicitly and cites `views.ts` 568-604; and the summary lines of three
verdict files disagree with their own per-finding verdicts (the table above
follows the per-finding verdicts, which are the ones that were applied).

| Finding | Section(s) changed | What changed |
| --- | --- | --- |
| PERF-1 | 7.7, 7.3, 11.4, R-3, 3.3, P4 | grid extent = `min(bbox, extentFactor * rmsRadius)`; RMS radius in the `sum p` `.w` lane; outside pseudo-cell; `stats.outsideGrid`; isolated-node fixture |
| PERF-2 | 7.7, D24, 7.21 | G6 / G7 dispatched in cell-sorted order; measurements recorded; brackets marked conservative |
| PERF-3 | 7.17, D8, R-24 | settle normalised by the RMS radius; resampling noise recorded as a risk with a measured mitigation |
| PERF-4 | 8.2, 3.5, 6 row 9, G7 | PageRank pull rebound to 8 named bindings via pre-scaled `xNorm` and scalars in `partials` |
| PERF-5 | 6 row 7, 8.4, 10.1, G8 | edge frontier sized to `A` or chunked with an explicit overflow rule; faked-capacity test |
| PERF-6 | 7.7, Q-32 | the inverted `gridMax` lever corrected; cap re-check recorded as a P4 decision |
| PERF-7 | D23, 7.3, 7.4-7.7 | device positions as `vec4f` (xyz + mass); the separate mass binding removed everywhere |
| PERF-8 | 7.6, 7.8, 7.21, 10.3, T-4 | the measured occupancy-limited curve replaces the 3.6e11 constant; 16k / 32k / 100k / 1M figures corrected |
| PERF-9 | 2.2, 7.8 | `calibrateLayout` default ladder 8k-65k, grid cost measured at the same sizes, min(4 ms, grid) rule, first-call cost stated |
| PERF-10 | 11.4, G4, T-12 | 1M runs only the one-iteration and unbiasedness checks in the lane; 200-iteration comparison at <= 262k; nightly for 1M |
| PERF-11 | 7.7, 6 row 12, 5.4, 7.4 | G4 appends hub cells to a device list; G4a finalize; G4b indirect workgroup-per-cell |
| PERF-12 | 5.4, 6 row 8, 8.4 | the finalize kernel selects the per-level variant on the device; `switches` is a device counter |
| PERF-13 | 8.4, 10.3, Q-13 | per-source cost model stated; BFS row aligned with T-10's basis; BC 2-20 s |
| PERF-15 | 7.2, 7.7, 3.5 | hash expressions parenthesised; the precedence rule recorded |
| PERF-16 | 7.21, 10.3 | integrated GPU 10-17x / 5-10x / equal readback from public peaks; the GraphWaGu citation dropped |
| PERF-MISSED-1 | D25, 7.7, 7.11, 11.4, G4 | the whole-displacement clamp removed; expansion-parity test added |
| PERF-MISSED-2 | 7.3, 7.4 | the `.w` lane of the `sum p` partial carries the squared deviation; stride stays 64 B |
| DESIGN-1 | D25, 7.7, 7.11, 11.4 | as PERF-MISSED-1 (the same defect found by two lenses) |
| DESIGN-2 | 7.2, 7.7, 7.3 | `eps = 0.25 * cellSize` in `state`; the near field uses the exact floor |
| DESIGN-3 | 7.2, 7.4, 7.6, 7.10, 7.11 | swing / traction over free nodes; K3 and G7 bind `fixed` |
| DESIGN-4 | 7.2, 7.10 | the conditional `estimateFactor` form; the L1 code is the executable spec |
| DESIGN-5 | 3.5, 7.6, 7.11, 6 rows 3 and 8 | the uniformity rule; sketches rewritten with reductions outside the guard |
| DESIGN-6 | D16, 2.6, 3.5, 6, R-8 | runtime `subgroup_size`; `SUBGROUP_MAX` for scratch only; faked `min != max` test |
| DESIGN-7 | 8.2, 8.10, 3.5 | PageRank count corrected; per-kernel binding table for every algorithm family |
| DESIGN-8 | 5.4, 8.4, G8 | all candidates recorded per level; device-side direction switch |
| DESIGN-9 | 8.4, 8.10, P9 | `n x k` deltas and a per-batch `bc` gather; no float race |
| DESIGN-10 | 4.1, DEPARTURE-4, Q-27 | siblings are one residency unit; `refs: Set<serial>` deleted |
| DESIGN-11 | 6 row 3, 7.4, 7.5, G4 | thread-per-row tier runs to `n`; isolated-node force test |
| DESIGN-12 | 6 row 12, 7.7, 7.3 | `cellStart` from a per-cell histogram + scan; hub-cell compaction |
| DESIGN-13 | 7.2, 7.6, 7.7 | kick magnitude `/ 0.01`, antisymmetric hash, own-cell HT weight |
| DESIGN-14 | 8.4, 6 row 4 | `atomicMin` visit claim; two-dispatch atomic ownership dedupe |
| DESIGN-15 | 2.2, 2.6, 5.3, 5.4, 5.5, 5.7, R-22, Q-16 | device-level alignments, unquantised timestamps, over-limit indirect, consumed adapters recorded as [M] |
| DESIGN-16 | 11.3, 11.5 | u32 bitwise, f32 within tolerance between twins and across adapters |
| DESIGN-17 | 8.7, 8.1 | APSP bound by the binding size: 5,792 / 23,170 / 32,767 |
| DESIGN-18 | 7.7, 11.4 | `cellSize` floor; floored parity denominator |
| DESIGN-19 | 6 row 7, 8.4 | as PERF-5 |
| DESIGN-20 | 6 row 6, 7.4 | digit-major radix histograms; dispatch count recounted |
| DESIGN-21 | 7.1, 7.13, 7.18, 11.3 | 2D writes `z = center.z`; property test updated |
| DESIGN-MISSED-1 | 6 row 8, 8.4 | fused-vs-two-phase selected on the device |
| DESIGN-MISSED-2 | D16, 3.5, 6, 2.6 | subgroup slot from an elected-lane counter, never `subgroup_id` |
| DESIGN-MISSED-3 | D8, 7.12, 7.17 | `reheat()` no longer resets the speed controller |
| DESIGN-MISSED-4 | 7.4, 7.17, 11.3 | all-fixed layout settles immediately |
| MAINT-1 | 3.5, 5.1, 7.5, 7.6 | bindings and overrides declared once in the module spec; emitted by the composer |
| MAINT-2 | 3.5 | `composeWgsl` rejects unfilled or unknown snippet markers; functor parameter lists named |
| MAINT-3 | 5.3, 7.3, 11.3 | `UniformBlock` storage mode for `state`; round-trip test |
| MAINT-4 | D26, 2.2, 3.1, 3.2, 3.3, 9.5, 9.6 | `src/context.ts` composition root; `createAccelerator` / `calibrateLayout` functions; import-boundary lint |
| MAINT-5 | 7.19, 3.2, 3.1 | the `ForceModel` hook interface named |
| MAINT-6 | 3.1, 3.5, 5.1, 11.3 | `src/kernels.ts` registry; CLAUDE.md recipes |
| MAINT-7 | D27, 3.3, 9.8, P10 | mirrors are the published contract; conformance test in `test/types` |
| MAINT-8 | 11.8, 11.1, G1 | thresholds active for `--project=node` |
| MAINT-9 | 4.4, 5.8 | staging-slot lifecycle stated |
| MAINT-10 | 3.5, 5.1, 5.2, 2.2 | constants interpolated into the prelude; `DIM` dropped from the override list; `maxComputeWorkgroupsPerDimension` not raisable |
| MAINT-11 | 2.2, 3.2, 3.3, 4.1 | one name per thing; `Profiler` listed; `AdapterSummary` / `GpuCalibration` / `CalibrateOptions` declared; `Extract<ViewName, ...>` |
| MAINT-13 | 2.4, 5.2 | the two readers of `caps.software` named |
| MAINT-14 | 3.1, 10.4, 11.1, 11.3, 11.7, 12.2, 12.3 | one results directory; scripts as `.js`; twins in-process plus the CI pass over layouts |
| MAINT-15 | 3.1, 6, 11.3, 9.8, P3, P10 | `test/oracle/<name>.ts`; `indexed.*` added as a second oracle, not a replacement |
| MAINT-16 | 3.3, 7.14, 9.3 | `SimulationOptions` (layout-owned) carries settle / step / in-flight knobs; `GpuLayoutTuning` GPU-only |
| MAINT-17 | 3.3, 3.1, 4.1, Q-24 | `precision` on every score result; `residency` `@internal` with `stripInternal`; `degree` a public diagnostic |
| MAINT-18 | 1.3, DEPARTURE-5, 9.2 | injection spelling declared; dispatcher attaches `pathTo` / `pathEdges` |
| MAINT-19 | 3.1, Q-18 | package CLAUDE.md contents enumerated; review annotations dropped on acceptance |
| MAINT-20 | 2.2, 2.6, 5.7, 5.8, 11.2 | consumed adapters; fresh adapter per device; batch-local validation rejection under Dawn-node |
| MAINT-M1 | D12, 5.7, 4.3, 3.3 | pass-through `GraphFormatError` codes (`E_GPU_INELIGIBLE`, `E_UNKNOWN_NODE`) |
| MAINT-M2 | 3.3, 7.14, 9.4, 9.5, R-2 | `createAccelerator(ctx, { layout })` carries GPU tuning to element-created simulations |
| MAINT-M3 | 2.3, 12.2, 12.3 | three env readers through one `scripts/gpu-policy.js` |
| MAINT-M4 | 2.4 | lint rule scoped `src/**` except `src/browser/**`; mirror import rule |
| MAINT-M5 | 3.1, 2.5 | root `.ts` shim dropped; `scripts/entries.js` |
| MAINT-M6 | 3.3, 7.20, 7.19 | `GpuLayoutSimulation<Options, Stats>` with per-model stats types |
| INTEG-1 | D10, 12.1, 12.3, 12.4, 12.5, R-6 | the GPU lane is its own workflow (`gpu.yml`), never a job of `CI` |
| INTEG-2 | D27, 2.4, 3.3, 9.8, P10 | as MAINT-7 |
| INTEG-3 | 3.3, 8.4, 9.2, 9.3, 9.4, 9.5, Q-13 | tuning defaults on `createAccelerator`; `sources` / `k` shared at A2; `SimulationOptions`; `ForceAtlas2Params` replaced |
| INTEG-4 | 7.3, 7.5, 7.14, 9.3, Q-30 | the GPU simulation resolves `nodeMass` / weights from graph-format primitives at `load()` |
| INTEG-5 | 4.5, 9.4, 11.3 | the release list includes the undirected and visible copies; residency test |
| INTEG-6 | 4.1, DEPARTURE-4, Q-27 | as DESIGN-10 |
| INTEG-7 | 9.8, Q-29 | nx side-effect patch releases accepted and recorded |
| INTEG-8 | 7.2, 7.10 | as DESIGN-4 |
| INTEG-9 | 2.5, 3.1, 12.3, 12.5, G0, G1 | `pnpm run build` (= `build:all`) on every lane; specifier-form bundle test; bundle-only subpaths |
| INTEG-10 | 1.3, DEPARTURE-5 | injection spelling declared |
| INTEG-11 | 1.3, DEPARTURE-6, Q-24 | the two `1e-4` cases declared |
| INTEG-12 | 3.3, 9.2, 9.7 | `pathTo` / `pathEdges` attached by the dispatcher |
| INTEG-13 | D13, 2.6, 9.1, 9.4 | `NumericVector` union; `info.isFallbackAdapter`; `Algorithm.ts:217`; the design-E1 engine shape |
| INTEG-14 | 9.3, 3.3, 7.20, 9.4 | `SimulationType` table; `spring` = FR; `spring-electrical` with its accelerator method |
| INTEG-15 | 9.2, 9.8 | the dispatcher's method list grows with the ports |
| INTEG-16 | 2.4, 12.5 | package-local `eslint.config.js` extending the root |
| INTEG-17 | 11.8, 12.3, 12.5 | as MAINT-8 |
| INTEG-18 | 1.3, 2.5, 3.1, 9.8, Q-31 | `workspace:^`; the design rule flagged |
| INTEG-19 | 2.2, 3.4, 9.5 | `ProbeResult.adapter` reused; `create()` honours `rejectSoftware` |
| INTEG-20 | 1.2, R-23, 7.19 | the `useWebGPU` branch described as present and unwired; risk row |
| INTEG-21 | 9.1, 9.4, 9.8 | real-GPU stories in the app; the element gains no devDependency |
| INTEG-22 | 9.4, 9.8, P6 | the design-E1 refactor named as a precondition |
| INTEG-23 | 2.5, P-ENV | `webgpu` peer range `>=0.4.0 <1.0.0`; P-ENV re-pins the devDependency and hint |
| INTEG-M-1 | 9.4, 9.5, 7.19, 5.7 | `LayoutManager` consumes `accelerator-changed`; running layouts move between paths |
| INTEG-M-2 | 12.5 | `ci.yml` never gains `schedule` or `labeled` |
| INTEG-M-3 | 12.1, 12.3 | the canary step greps `vendor=nvidia` and fails on a skip |
| INTEG-M-4 | 3.1, 12.5, G10 | strict-consumer compile inside `lint` |
| INTEG-M-5 | 7.1, 7.12, 7.19, 9.4, 11.3 | pins re-applied from `node.pinned` after `reload`; `load()` clears the mask on a resize |
| VERIFY-1 | 2.5, 12.3, G0 | as INTEG-9 |
| VERIFY-2 | 12.5 | separate monorepo `gpu.yml`; no `schedule` on `ci.yml` |
| VERIFY-3 | D10, 12.1 | as INTEG-1 |
| VERIFY-4 | 12.2, 12.4 | host-side JIT loop; per-job container; the PAT never in the container |
| VERIFY-5 | 11.5, G1, 11 review notes | u32 bitwise across adapters; f32 within `1e-5` |
| VERIFY-6 | 11.8, 11.1, G1 | as MAINT-8, with the node project's `include` stated |
| VERIFY-7 | 12.3 | `shell: bash`; no `tee` on the report step |
| VERIFY-8 | 11.6, 12.3, R-13, 13 rule (a) | JSON reporter + `timeout -k`; exit 124 passes iff all tests passed |
| VERIFY-9 | 12.3, 12.4 | Playwright browser installed in the GPU job |
| VERIFY-10 | D21, 7.2, G0 | the 7.2 sign-off moves to G0 |
| VERIFY-11 | 7.2, 11.4, G3, P3 | oracle cross-checked against `@graphty/layout@1.6.2` and a NetworkX iteration-0 fixture |
| VERIFY-12 | 12.2, 12.3, 12.5, G3, G7 | the no-subgroups pass covers layouts (default lane) and the whole node project (GPU lane) |
| VERIFY-13 | 11.1, 11.7, 12.3, 10.4, 3.1 | the tsx harness with an async `run`; no `bench` vitest project; runner class defined |
| VERIFY-14 | 12.3, 12.6 | `gpu-nightly-report` job with `issues: write` on `ubuntu-latest` |
| VERIFY-15 | 12.1, 12.3 | as INTEG-M-3 |
| VERIFY-16 | 2.3, 11.1, 3.1 | browser project env forwarding through `test.env` / `import.meta.env` |
| VERIFY-17 | 11.4, 11.1 | 262k / 1M in `node-limits`; `gridMax2D = 32` saturation case for software adapters |
| VERIFY-18 | 5.1, 11.3, 11.6, 3.1 | `override-matrix.ts`: a bounded, asserted-complete compile matrix |
| VERIFY-19 | 12.3 | `overwrite: true` on artifact uploads |
| VERIFY-20 | T-13, 11.7, 12.3, 12.6, R-25 | utilisation sample; skip on a busy GPU; two nightly failures before an issue |
| VERIFY-21 | 10.4, 11.4 | the owner-decision rule covers 11.4; the `1e-4` leg against an f32 oracle |
| VERIFY-MISSED-1 | 12.3, P0 | `pnpm/action-setup` reads `packages/package.json`; workspace and knip entries in P0 |
| VERIFY-MISSED-2 | 7.2, 9.3, 11.4 | `seedPositions` writes f32; the oracle reads the same array |
| VERIFY-MISSED-3 | 12.5 | as INTEG-M-2 |
| VERIFY-MISSED-4 | 12.2, 12.4 | the runner user's NOPASSWD sudo documented; acceptable only per job |
| VERIFY-MISSED-5 | 11.7 | async `bench()` |
| COMPLETE-1 | 8.2, 3.5, Summary | as PERF-4 |
| COMPLETE-2 | 3.3, 9.4, 9.5, P12 | as MAINT-M2 / INTEG-3 |
| COMPLETE-3 | 9.4, 2.4, 7.19 | `accelerator-changed` consumer; `gpuMinNodes` evaluated on load / reload; `.catch` once per promise; device-loss re-creation |
| COMPLETE-4 | D7, 3.1, 7.8, 7.21, Q-6 | 16,384 labelled conservative; `calibrateLayout` ladder aligned with T-4; 32k row "exact or grid" |
| COMPLETE-5 | 13 (P2, P4, P7, P8), 7.5, 6 row 3 | P2 rescoped to what P3 needs; tiers and primitives moved to the phases that need them |
| COMPLETE-6 | 7.20, Q-9, P5 | the preset is BUILT in P5; routing stays a product decision |
| COMPLETE-7 | 3.1, 12.1, P0, Q-28 | the public repository created in P0 |
| COMPLETE-8 | 7.3, 7.17 | `state` has COPY_DST |
| COMPLETE-9 | 2.5, 3.1, 11.3 | specifier-form test; "the only file that imports the module" |
| COMPLETE-10 | 4.7, 7.3, 10.1 | one bytes-per-node accounting (53 exact; 65 resident / 81 peak grid) |
| COMPLETE-11 | 4.2, 4 review notes, 10.1, DEPARTURE-2 | arena figures corrected to `...,256` |
| COMPLETE-12 | 10.4, 11.1, 11.7, T-4, P4 | one baseline name; one exact ladder and one grid ladder |
| COMPLETE-13 | 12.3, 12.6 | as VERIFY-14 |
| COMPLETE-14 | 12.3 | as VERIFY-9 |
| COMPLETE-15 | 1.1, preamble | goals renamed `GOAL-n` |
| COMPLETE-16 | 7.8, 7.14, 7.17, 5.7, 12.3, 14.2, D11, P-ENV | every Q cited at its decision point; draft C's Q-10 disambiguated |
| COMPLETE-17 | 1.3, DEPARTURE-5, DEPARTURE-6 | as INTEG-10 / INTEG-11 |
| COMPLETE-18 | 2.2, 2.5, 5.1, 7.4, 8.4, D13, 9.1, 1.2, R-22 | basis labels added; facts corrected; verified facts promoted to [M] |
| COMPLETE-19 | 3.3, 7.14, 7.19, D12, 9.3 | `step()` contract; package default `iterationsPerStep = 1`; `E_IN_FLIGHT` removed; `spring` aliased |
| COMPLETE-20 | Q-18, 3.1 | drafts committed beside the notes; annotations dropped on acceptance |
| COMPLETE-21 | 13 | the critical-path paragraph names the monorepo landings "detected" waits for |
| COMPLETE-22 | G2, T-12, 12.6, 10.3, T-8 | one lane budget; the x100 column recomputed; T-8 aligned |
| COMPLETE-M1 | 8.2, 3.3, 9.7 | `firstConvergedIteration` recorded on the device |
| COMPLETE-M2 | T-5, 11.6, 11.7 | the `bench`-tagged browser test at 10k / 100k |
| COMPLETE-M3 | 11.8, G1 | as MAINT-8 |
| COMPLETE-M4 | 12.2, 12.3, 12.5 | as VERIFY-12 |
| COMPLETE-M5 | 7.21, 10.3, 13, DEPARTURE-2 | one exact-tier figure; the total re-summed (83-113 ed after the phase resizes); arena figures |
| COMPLETE-M6 | 7.3, 7.10, 3.3 | the 32-byte trace record's two writers named |
| CHECK-R1 | Summary, 3.5, 8.10 | the PageRank pull is no longer "the one kernel at 8": G7 and five 8.10 kernels are named beside it |
| CHECK-R2 | 1.3 row 14.5 | "Three DEPARTURES (1, 4, 5)" -- DEPARTURE-4 also departs from 14.5 line 4231 |
| CHECK-R3 | 9.8 W1 row, P10 | the design amendments scheduled at W1 cover DEPARTURE-1, -2, -4, -5, -6 (DEPARTURE-3 and -7 amend 14.3 in the L1 PR) |
| CHECK-R4 | 12 table rows (1.4, 2.3, 7.2, 7.3, 7.7, 7.20, 11.3, P4) | 35 unescaped pipe characters inside backtick spans escaped (backslash-pipe) so GFM keeps the column counts |
| CHECK-C1 | 15 | `pnpm.io/workspaces` and `docs.github.com/en/actions/reference/limits` added to the external references (cited at 1.3 / 2.5 and 12.1) |
| REPAIR-1 | 1.5, 1.3 row 14.3, D23, 7.3, 9.8 W1 row, P10 | DEPARTURE-7 declared: `array<vec4f>` device positions with `load()` / `setPosition` repack and `toScene` unpack instead of design 14.3's stride-as-uniform kernels on the owner's stride-3 column (the no-copy outcome holds, the mechanism does not); 14.3 amended in the L1 PR with DEPARTURE-3 |
| REPAIR-2 | D24, 7.3 | the "4-60x" bracket replaced by the 7.7 / 7.3 measurements (4.7x / 16x / 43x sorted; 9x / 18x / 70x with `vec4f`; far field 2.3-2.8x); the clustered `vec4f` figure (2.01 -> 1.78 ms) added to 7.3 from the verifier's re-run |
| REPAIR-3 | 5.2 heading, 1.3 row 10.6 | "DispatchPlanner" label replaced by the pure-function names `plan1d` / `plan2d` / `planGridStride` / `planIndirect` (one name per thing, MAINT-11) |
| REPAIR-4 | 9.5 sketch | `adapter: probe.adapter ?? undefined` -- `ProbeResult.adapter` is `GPUAdapter \| null`, `GpuContextOptions.adapter` is `GPUAdapter \| undefined` and `ok` is not a discriminant |

Owner additions (2026-09-14, after the review, not review findings): the
play / pause / stop semantics paragraph and its frame-loop test case in 7.19;
9.4 item 9 (public `setRunning` on the element) and the matching E1 test;
14.3 (deferred rendering-at-scale notes: thin instances, zero-copy rendering)
with a pointer from Q-7. The scope decision recorded in 14.3 is the owner's:
a few hundred thousand nodes interactively in v1.

Owner decisions applied (2026-09-14, after the review): Q-1 (published laws
by default, `compat: "networkx"`, the port's variants dropped: D5, 7.2, 7.10,
7.11, 11.4, R-1, DEPARTURE-3), Q-3 and Q-28 (paid GitHub-hosted T4 runner in
its own workflow, no self-hosted runner; public repository; trusted publishing
later: D10, GOAL-7, 12.1-12.6, R-6, R-25, P0), Q-26 (optional peer
dependencies with `import type`, mirrors only until W1: D27, 2.4, 3.1, 9.8
W1 row), Q-30 (node vectors by graph-format role: D28, 3.2, 7.5, 7.14, 9.3,
9.4 item 10). Each decided row in 14.2 now starts with "DECIDED".
Q-1's centroid gravity and Q-3's GitHub Team + T4 choice confirmed by the owner the same day.
Also added for the owner's verification question of 2026-09-14: 11.9 (sabotage matrix, per-kernel `inspect()`, derived tolerances, run-twice determinism), rule (f) of section 13, and the G1 / G3 / G4 gate items that cite it.

Approved by the owner on 2026-09-14. The npm placeholder 0.0.0 was published
the same day and its directory is the seed of P0's package (Q-28).

Interface contract (2026-09-15, `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md`,
the normative P0-P3 declarations derived from this plan): four corrections
it makes to statements above, PENDING the owner's confirmation at the G0
sign-off of the 7.2 table (D21); until then the contract's reading is what
P0-P3 implement:

1. 7.2 gravity-centre row and Q-1: NetworkX 3.4.2 `forceatlas2_layout` pulls
   toward the ORIGIN (`layout.py` lines 1466-1471: `-gravity * mass * pos /
   |pos|`), not the centroid; `compat: "networkx"` therefore compiles
   `GRAVITY_CENTER = 1`, and the "centroid (port, NetworkX)" wording is wrong
   about NetworkX (it is right about the port).
2. 7.2 "Local speed / apply" row, 7.10 and 7.11: NetworkX's PER-NODE factor
   uses `swinging = mass * |update|` (`layout.py` line 1497, the force), and
   only the GLOBAL sums mix positions (`swing += (mass * |pos - update|).sum()`,
   line 1483). `SWING_MODE = 1` keeps the position-mixed accumulated global
   sums and uses `m_i |F_i|` in K5's local factor.
3. 3.5 lines 1032-1033 and D16: "scratch arrays are sized `WG / SUBGROUP_MAX`
   (rounded up), which is enough for the smallest size the compiler may
   pick" is inverted -- the subgroup COUNT is `WG / subgroup_size`, largest
   at the SMALLEST size; scratch is sized by `SUBGROUP_MIN`
   (`adapter.info.subgroupMinSize`, a second standard override), and
   `SUBGROUP_MAX` stays declared.
4. 4.1 lines 1130-1136 versus 6 row 3 / 7.3 / 7.5: 4.1 makes `USE_PERM` the
   arcToEdge / edgeToArc identity guard, the other three make it the
   `degreeOrder()` row permutation with `rowPtr` as the dummy. The contract
   follows the majority (`USE_PERM` / `perm` = the row permutation; the arc
   guard of P7+ is the reserved `USE_ARC_PERM`).

P3 gate findings (2026-09-16, `packages/webgpu-graph-algorithms/docs/decisions/G3.md`
section 10), PENDING the owner's confirmation; until then the G3 record's
reading is what the code and tests implement:

5. 7.10 `fa2-speed-finalize` snippet, the halving predicate (G3-F6, CONTRACT
   DECISION K4-1): `swing / tr > 2.0` is written `swing > 2.0 * tr` in the
   kernel AND in the oracle's `estimateFactor`. Right after `load()`
   `oldForce = 0`, so traction is exactly half of swing and the predicate
   sits on its knife edge; NVIDIA's f32 shader division is not correctly
   rounded (`x / (x / 2) != 2` for 15% of inputs, 2.2% above 2; lavapipe is
   exact), so ~2% of paper-mode first iterations halved the efficiency where
   the CPU reference never does and the two subgroup twins disagreed by
   exactly 2x. The multiplication form is exact in every implementation; the
   CPU port's `swing / traction > 2` is unaffected in f64.
6. 11.4 "Trace parity" and 11.3 "Subgroup variants" (G3-F3, G3-F4): the
   free-running 10- and 50-iteration comparisons cannot be held to 1e-4 /
   5e-2 in `compat: "paper"` on ANY pair of implementations -- the per-node
   swing `m |F(t) - F(t-1)|` cancels near equilibrium and the error grows
   x1.1-1.5 per iteration, so the f64 oracle misses both caps against ITSELF
   under a one-ulp start perturbation (0.86-1.71 through 50 iterations), and
   a seed sweep shows the same for `networkx` mode at 50 iterations (7 of 8
   karate seeds over 5e-2). A re-synchronised comparison (a fresh oracle
   seeded with the GPU's iteration-start state for ONE iteration, over 50
   iterations, both modes, both oracles) agrees at 1e-7..1e-5 on every
   fixture with zero controller-branch mismatches, which is what the tests
   now assert (tolerances derived from measured floors: f32 6.8e-5, f64
   1e-4; the twins' re-synchronised trace at 1e-6); the free-running legs
   stay asserted only for `networkx` mode over the first 10 iterations at
   the derived 2.1e-5 (seed 7; seed-bound on NVIDIA -- owner item G3-F3 (2))
   and are printed otherwise. The sabotage rows of K1-K5 fail the
   re-synchronised assertions by >= 1.2e3x. The distributional cases of 11.4
   are re-selected on the same basis (the f64 oracle against itself under
   one-ulp perturbations): grid10 / paper, karate / networkx 2D and the
   isolated-node fixture in paper mode land in different basins (0.16-0.22)
   and are dropped; eight stable cases stay at 10%.
7. 11.3 "Property" row "speed NOT reset by setPosition": restated as D8
   promises it -- `state.speed` / `speedEfficiency` untouched at the moment of
   the call, the next iteration continuing from them (checked through the
   re-synchronised oracle over 200 drags) -- instead of comparing across a
   batch that legitimately ran.
