# WebGPU graph algorithms and layouts: implementation plan (draft C, verification-first)

Status: plan draft, 2026-09-14. Angle: every phase ends in a hard verification
gate; the walking skeleton is phase 1; the ForceAtlas2 layout is the first
product slice; primitives are pulled in by the slice that needs them.

Conventions: n = node count, E = logical edge count, A = arc count
(`snapshot.arcCount`; 2E - selfLoops on undirected snapshots), dim = 2 or 3,
"4070" = the dev box's NVIDIA RTX 4070 SUPER under Dawn-in-Node unless
"Chromium" is stated. "design" = the accepted design document
`/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
(cited by section and line). "note NN" = the research notes in
`/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/NN-*.md`.
Every claim about existing code carries a path; every external claim carries
a note reference (which carries the URL) or a URL in section 15. Numbers are
labelled MEASURED (probe or test on the dev box), REPORTED (a paper or repo
benchmark) or EXTRAPOLATED (a derived estimate the gates must replace).

Contents

1. Goals, non-goals and inherited decisions
2. Runtime model
3. Package architecture
4. Memory and upload
5. Kernel infrastructure
6. Primitives
7. Force-directed layouts -- FIRST DELIVERABLE
8. Algorithms
9. Integration with @graphty/algorithms, @graphty/layout and @graphty/graphty-element
10. Performance targets and memory model
11. Testing strategy
12. CI/CD
13. Phased implementation plan
14. Risks and open questions for the owner
15. References

---------------------------------------------------------------------------

## 1. Goals, non-goals and inherited decisions

### 1.1 Goals

G1. `@graphty/webgpu-graph-algorithms`: one TypeScript + WGSL code base that
    runs the same kernels under Google Dawn in Node (npm `webgpu@0.4.0`) and
    in browsers, consuming `@graphty/graph-format` snapshots as implemented
    in `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format`.
G2. First product deliverable after the walking skeleton: a GPU ForceAtlas2
    `LayoutSimulation` (design 14.3) that graphty-element can step per frame
    with pins and drag, in 2D and 3D, exact below a measured node-count
    crossover and approximate (grid pyramid) above it, targeting 100k nodes
    interactively and 1M nodes in batch on a discrete GPU.
G3. GPU algorithm accelerators for the highest-value `indexed.*` functions
    (PageRank family, connected components, BFS/SSSP/closeness, betweenness
    with sampling, then structure/community), returning index-aligned typed
    arrays that graphty-element's single result-writing loop consumes
    unchanged (design 14.4 line 4180).
G4. Optional / detected acceleration: the CPU packages own structural
    accelerator interfaces and an async dispatcher; graphty-element exposes
    an `accelerator` property; the graphty app probes and injects. The GPU
    package never falls back (root `CLAUDE.md`; design 10.8 lines 2545-2552).
G5. Tests primarily in Node on Dawn (full suite, real GPU locally, Mesa
    lavapipe on hosted CI), a light browser smoke suite on Playwright
    Chromium (real GPU locally, SwiftShader on hosted CI), and two CI lanes:
    a default runner and a GPU runner.
G6. Measured performance: every phase gate records numbers on the 4070 into
    `benchmarks/results/` so the plan's EXTRAPOLATED figures become MEASURED
    before the next phase starts.

### 1.2 Non-goals (v1)

- No rendering; no sharing of a `GPUDevice` with Babylon (graphty-element
  renders on WebGL today: `graphty-element/src/managers/RenderManager.ts`
  lines 63-69 per note 01 section 4.8). Positions are read back per frame.
- No CPU, WebGL or SwiftShader "fallback" inside the package (root
  `CLAUDE.md`). Software adapters are a TEST environment, never a product
  path.
- No WebGPU compatibility-mode target (core defaults only; note 05 section 4).
- No `SharedArrayBuffer`, no f16 storage (the 4070 exposes no `shader-f16`
  under Dawn or Chromium 139: note 05 section 2.4), no multi-GPU.
- Not GPU targets (note 02 section 7.2 last row): DFS, topological sort,
  cycle detection, Prim, Girvan-Newman, hierarchical / TeraHAC / GRSBM /
  SynC, MCL, max-flow / min-cut, bipartite matching, isomorphism, A*.
- No on-device incremental graph mutation: every freeze produces a new
  snapshot; the GPU re-uploads what changed by array identity (design 14.5).
- Kamada-Kawai, ARF and spectral layouts are later slices (note 02 section
  7.1 ranks them L4-L6).

### 1.3 Decisions inherited from the accepted design (not relitigated)

| Id | Design section (lines) | Decision the plan honours |
| --- | --- | --- |
| I-1 | 10.1 (2327-2351) | The bind list: `rowPtr`, `colIdx`, `weights`, `arcToEdge`, `edgeToArc`, `reverse()`, `coo().src`, `edgeList()`, degrees, `degreeOrder(opts)`, `mate()`, `gpuView()` columns; the `override USE_PERM` pattern with `colIdx` / `rowPtr` as never-read dummies; the weighted out-degree normaliser is computed on the device; `segmentOffsets` is read on the CPU; entry points take `GraphSnapshot`; field names `nodeCount / arcCount / weights`; parents are `Uint32Array` with `INVALID_INDEX` |
| I-2 | 10.2 (2352-2374) | 4-byte alignment by construction; `u8` via `paddedU32View()` + `unpack4xU8`; `bool` / masks as u32 words LSB-first; `components: 3` read as `array<f32>` with `3*i` indexing, never `array<vec3<f32>>`; `f64` columns via the cached f32 `gpuView()` copy |
| I-3 | 10.3 (2375-2442) | Upload plan: whole-arena hot prefix when `arena.byteLength <= maxBufferSize` and every segment `<= maxStorageBufferBindingSize`; else per array (also the `arena === null` path); else windowed |
| I-4 | 10.4 (2443-2466) | `gpuEligibility` table; results attach by reference with `nodes.set()` |
| I-5 | 10.5 (2467-2492) | Invariants I1-I10 assumed without checking; zero-length arrays are never bound; a weighted out-degree may be 0 for a node with arcs |
| I-6 | 10.6 (2493-2521) | 1D dispatch legal iff `ceil(count / 256) <= 65,535`, i.e. at most 16,776,960 invocations (not 2^24); windows start at arc indices that are multiples of 64 computed with `%`; rows longer than a window are split on ARC ranges |
| I-7 | 10.7 (2522-2544) | Per-node `Uint32Array(n)` / `Float32Array(n)`; per-arc results folded with `foldArcs`; copy out of `getMappedRange()` before `unmap()`; `dest?` on every algorithm; nothing keyed by id leaves the package; GPU scores are `Float32Array` |
| I-8 | 10.8 (2545-2552) | Device queries, buffers, chunking, dispatch math, frontiers, scans and dense relabelling live in the GPU package; no "GPU unavailable" behaviour inside it |
| I-9 | 14.3 (3959-4047) | `LayoutSimulation { load, step(iterations?): void \| Promise<void>, settled, setFixed(NodeMask), setPosition(i,x,y,z), dispose }`; positions are the owner's stride-3 scene-unit `F32`, read and written in place; the GPU buffer is authoritative while stepping; FA2 default mass is `outDegree()[i] + 1`; weights via `snapshot.weights` when `weight === true` or a named edge column; layouts receive an undirected snapshot |
| I-10 | 14.4 (4048-4211) | `DataManager` owns the builder and the positions array; `NaN` rows are unplaced; `snapshot-replaced { previous, next, report }` triggers `accelerator.release(previous)`; layout engines receive `dm.undirected(s).snapshot`; drag is `simulation.setPosition`; `column.markDirty()` once per frame; adapters run `indexed.*` or the injected accelerator and share one result-writing loop |
| I-11 | 14.5 (4212-4244) | Move-in as `webgpu-graph-algorithms/`; `CSRGraph` / `EdgeListGraph` deleted; `noUncheckedIndexedAccess` OFF; upload cache is a `WeakMap` keyed on the typed-array object, invalidated by `column.version`, released by `release(snapshot)`; never falls back |
| I-12 | 14.6 (4245-4278) | Landing order F1 -> A1 -> F2 -> A2 / L1 / E1 -> W1 -> D1 -> IO1 -> 2.0; the GPU package owns upload planning, chunking, readback and `release` |
| I-13 | 15.1, 15.3 (4281-4364) | Byte model per array; target tiers 100k/1M (mobile), 1M/10M (desktop interactive), 10M/100M (batch, `arena: false`, windowed GPU bindings) |
| I-14 | 16.2 (4535-4552) | `1e-5` relative tolerance for f32 GPU parity against the f64 CPU path; order-agnostic comparison for component lists |
| I-15 | 16.6 (4622-4642) | A strict-consumer `tsc` compile of the published d.ts with `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` on |

### 1.4 What this plan decides (left open or superseded by the design)

| Id | Decision | Where |
| --- | --- | --- |
| P-1 | Node on Dawn is the PRIMARY test project; the browser project is a light smoke suite. This departs from design 14.5 ("a browser-only vitest project", line 4214) at the owner's request; amend 14.5 and 16.7 when this plan is accepted (note 06 section 9 item 5) | sections 11, 12 |
| P-2 | Runtime model: the core never touches `navigator`; `GpuContext.create({ gpu })` takes a `GPU`; two subpaths `.` and `./node`; `webgpu` is an optional peer loaded by dynamic import | section 2 |
| P-3 | Plug-in mechanism: explicit injection as the contract, graphty-element wiring, app-side detection; the CPU packages own `AlgorithmAccelerator` / `LayoutAccelerator` and one async dispatcher; the registry mechanism is rejected (note 02 section 4) | section 9 |
| P-4 | The async `step()` bridge in graphty-element is fire-and-forget with a double-buffered readback (note 01 section 5.3 option A) | section 7.9 |
| P-5 | ForceAtlas2 reference semantics for BOTH the CPU rewrite (L1) and the GPU: the published Jacomy 2014 / Gephi / cuGraph laws (repulsion magnitude `kr m_i m_j / d`, swing `m * norm(F(t) - F(t-1))`, traction `m * norm(F(t) + F(t-1)) / 2`, sums reset per iteration, symmetric size correction), gravity toward the centroid as the current port does. This fixes the three deviations of note 01 section 2.1.9 in one place before any WGSL is written | section 7.3, risk R-1 |
| P-6 | Repulsion back-ends: exact tiled all-pairs up to `exactMaxNodes` (default 16,384, to be re-fixed by measurement), cosmos-style grid pyramid above; GraphWaGu-style Hilbert cluster tree is a documented experiment, not a deliverable (note 03 section 8) | section 7.5 |
| P-7 | The simulation runs in LAYOUT units on the device; scene scale and center are applied by a write-back kernel; `setPosition` converts scene -> layout on the CPU (note 01 section 8.4) | section 7.7 |
| P-8 | `settled` = iteration budget exhausted OR windowed mean displacement below a scene-unit threshold; reheat on `load`, `setPosition`, unpin (note 01 section 8.6) | section 7.8 |
| P-9 | WGSL lives in `src/wgsl/*.wgsl.ts` template-string modules composed by string functions; no `.wgsl?raw` (note 07 section 4.6) | section 3.4 |
| P-10 | Kernel shape rules: gather/pull first; u32/i32 atomics only; fixed-point i32 for min/max reductions; no cross-workgroup spinning; every optional feature (`subgroups`, `timestamp-query`) is a pipeline variant, never a requirement (notes 04 section 1, 05 section 6) | sections 5, 6 |
| P-11 | Two CI lanes: `ubuntu-latest` on lavapipe + SwiftShader (required check), a self-hosted ephemeral runner on the dev box with the 4070 (not required; push / nightly / dispatch / `gpu` label on same-repo PRs) (note 06) | section 12 |
| P-12 | `webgpu@0.4.0` pinned until the dev container and the runner image move to Ubuntu 24.04 (glibc 2.38 requirement of 0.5+/0.6+, note 05 section 2.1) | sections 2, 14 |
| P-13 | Algorithm order by value x speedup / risk: PageRank family, WCC, BFS/SSSP/closeness, betweenness, structure, Louvain last (note 02 section 7.2) | section 8 |
| P-14 | Own error class `WebGpuGraphError { code, details }` with stable codes; never `GraphFormatError` (note 07 section 1.1) | section 5.6 |
| P-15 | Documented behaviour change: `weight` becomes LIVE for FA2 through the snapshot (it is inert in the element today, note 01 section 7.1) | section 9, risk R-12 |

---------------------------------------------------------------------------

## 2. Runtime model

### 2.1 One code base, two runtimes

The core package (`src/**`, entry `.`) never references `navigator`,
`window`, `process` or the `webgpu` module. Every entry point takes a
`GpuContext` created from a caller-supplied `GPU` or `GPUDevice`. This is
what makes the same WGSL and the same TypeScript run under Dawn-in-Node and
in Chromium / Firefox / Safari (note 02 finding 5; note 05 section 1
recommendation 1).

Node specifics the core must tolerate (note 05 section 2.2):

- Dawn-node exposes `GPUBufferUsage`, `GPUMapMode`, `GPUShaderStage` only
  after `Object.assign(globalThis, dawn.globals)`. The core therefore reads
  those namespaces ONLY inside functions (never at module top level) and the
  `./node` entry installs the globals before returning a context. Kernel
  modules use numeric literals with a comment where a constant is needed at
  module scope (e.g. `0x0080 /* GPUBufferUsage.STORAGE */`).
- `adapter.isFallbackAdapter` is `undefined` under Dawn-node; software
  adapters are detected by `adapter.info.architecture === "software"`
  (lavapipe) or `vendor === "google" && architecture === "swiftshader"`
  (Chromium). One helper `isSoftwareAdapter(info)` in `src/device/caps.ts`.
- `requestAdapterInfo()` no longer exists; use `adapter.info` (Chrome 127+;
  the scaffold's `test/setup/webgpu-global.ts` line 58 still calls the
  removed method and is deleted).
- Dawn-node 0.4.0 has no shim that unmaps buffers on `device.destroy()`
  (0.6.1 adds one); the readback helper always `unmap()`s or destroys its
  staging buffer itself.
- Uncaptured validation errors print to stderr and do not throw; the
  context installs an `uncapturederror` listener (works identically in both
  runtimes, note 05 section 2.2) and error scopes around creation calls.

### 2.2 Device acquisition and the detection helper

```ts
// src/device/context.ts (entry ".")
export interface GpuProbe {
    readonly ok: boolean;              // an adapter exists and requestDevice would be attempted
    readonly reason: string | null;    // "E_NO_GPU" | "E_NO_ADAPTER" | null
    readonly vendor: string | null;    // adapter.info.vendor when ok
    readonly architecture: string | null;
    readonly software: boolean;        // isSoftwareAdapter(adapter.info); the caller decides whether that is acceptable
}
export interface GpuContextOptions {
    readonly gpu: GPU;                                       // navigator.gpu in the browser; dawn.create([...]) in Node
    readonly powerPreference?: GPUPowerPreference | undefined;   // default "high-performance" (Chrome 145 needs it to return NVIDIA, note 05 section 3.2)
    readonly raiseLimits?: boolean | undefined;              // default true: request adapter maxima for the six limits in section 5.2
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;   // default ["subgroups", "timestamp-query"]; requested only when the adapter has them
    readonly label?: string | undefined;
}
export class GpuContext {
    static probe(options: { gpu: GPU | null | undefined; powerPreference?: GPUPowerPreference }): Promise<GpuProbe>;   // never throws; no device is created
    static create(options: GpuContextOptions): Promise<GpuContext>;   // throws WebGpuGraphError E_NO_ADAPTER / E_NO_DEVICE; NEVER returns a CPU stand-in
    static from(device: GPUDevice, info?: GPUAdapterInfo | undefined): GpuContext;   // adopt a device the caller owns (a future Babylon WebGPUEngine)
    readonly device: GPUDevice;
    readonly caps: GpuCaps;                       // section 5.2; device.limits, features, subgroup sizes, software flag, runtime
    readonly lost: Promise<GPUDeviceLostInfo>;    // device.lost; the context transitions to "lost" and rejects pending work
    readonly state: "ready" | "lost" | "disposed";
    accelerator(): GpuAccelerator;                // the object graphty-element injects (section 9.2)
    release(snapshot: GraphSnapshot): void;       // destroys every GPUBuffer uploaded for that snapshot (section 4.6)
    dispose(): void;                              // releases every residency, destroys pools, destroys the device if this context created it
}
```

Detection is the INJECTOR's job, not the package's (note 02 section 4.5):
`probe()` answers "is there an adapter, is it software" without side
effects; `create()` either succeeds or throws. There is no code path that
returns something that computes on the CPU.

### 2.3 The no-fallback rule versus optional / detected acceleration

- Inside the package: no adapter -> `E_NO_ADAPTER`; device lost -> pending
  promises reject with `E_DEVICE_LOST`; a kernel validation error -> a
  thrown `E_VALIDATION`. Nothing catches these to run a CPU path.
- In the consumers (section 9): the ONLY branch that selects the CPU is
  `accelerator?.method === undefined`, evaluated BEFORE any GPU work. A throw
  from a GPU method propagates to graphty-element's operation queue and is
  reported as a failed algorithm run; the element may offer "disable
  accelerator" as a user action, which is a user decision, not a fallback
  (note 02 section 4.5).
- Detection in the app: `GpuContext.probe({ gpu: navigator.gpu })`, then
  `await import("@graphty/webgpu-graph-algorithms")` (code-split), then
  `create()`, then `element.accelerator = ctx.accelerator()`.

### 2.4 Subpath exports and keeping `webgpu` out of browser bundles

```jsonc
"exports": {
    ".":      { "types": "./dist/webgpu-graph-algorithms.d.ts", "import": "./dist/webgpu-graph-algorithms.js", "default": "./dist/webgpu-graph-algorithms.js" },
    "./node": { "types": "./dist/node.d.ts",                   "import": "./dist/node.js",                   "default": "./dist/node.js" }
},
"dependencies":         { "@graphty/graph-format": "workspace:*", "@webgpu/types": "^0.1.72" },
"peerDependencies":     { "@graphty/graph-format": "^0.1.0", "webgpu": "^0.4.0" },
"peerDependenciesMeta": { "webgpu": { "optional": true } },
"sideEffects": false
```

- `.` is browser-safe: no `node:` imports, no `webgpu` import, DOM types only
  through `@webgpu/types` (a runtime-free package that MUST be a dependency
  because `GPUDevice` appears in the public d.ts; `src/index.ts` starts with
  `/// <reference types="@webgpu/types" />`, note 07 section 4.2).
- `./node` exports `createNodeGpuContext(options?: { adapter?: string;
  backend?: string; dawnFeatures?: string[]; software?: boolean })`: it does
  `const dawn = await import("webgpu")` (dynamic so a bundler that reaches
  the file cannot statically pull the native addon), installs
  `dawn.globals`, builds the `create([...])` option strings (`adapter=...`,
  `backend=...`, `enable-dawn-features=...`; `software: true` becomes
  `adapter=llvmpipe`, documented as Linux/Mesa-specific), and returns a
  `GpuContext` that holds the `GPU` object so `dispose()` can drop it (the
  process cannot exit while it is reachable from a global, note 05 section
  2.2). Absent module or adapter -> `E_NO_ADAPTER`, never a fallback.
- The vite library build externalises every dependency and peer including
  `webgpu` (the `packages/graph-io/scripts/build-bundle.js` pattern, note 07
  section 4.6), and the tsc build (`tsc -p tsconfig.build.json`) is what the
  pre-push hook runs, so the dist must not depend on vite-only syntax
  (hence P-9).
- Browser usage is simply `GpuContext.create({ gpu:
  navigator.gpu, powerPreference: "high-performance" })`; there is no
  `./browser` subpath because nothing browser-specific exists beyond passing
  `navigator.gpu` (note 07 section 4.2 lists `.` and `./node` only; note 05
  section 8.1 proposed a `./browser` entry, which this plan folds into `.`).

### 2.5 Version pins

`webgpu@0.4.0` (Linux binary needs `GLIBC_2.34`; 0.6.1 needs `GLIBC_2.38`
and fails to load on the Ubuntu 22.04 / glibc 2.35 container: note 05
section 2.1 verified by `strings` and a failed `require()`), `@webgpu/types
^0.1.72`, `vitest ^3.2.4` / `@vitest/browser ^3.2.4` / `playwright ^1.54.1`
/ `vite ^7` to match the monorepo (`/home/apowers/Projects/graphty-monorepo/package.json`
lines 130, 148, 158, 160 per note 05 section 9.2). One `webgpu` version in
both CI lanes; bump once when the container image moves to 24.04 (P-12).

---------------------------------------------------------------------------

## 3. Package architecture

### 3.1 Layers

| Layer | Modules | Depends on |
| --- | --- | --- |
| L0 device | `GpuContext`, `GpuCaps`, `isSoftwareAdapter`, error scopes, device-loss state, `WebGpuGraphError` | `@webgpu/types` only |
| L1 memory | `GraphResidency` (upload cache + plans + release), `BufferPool`, `Readback` (staging ring), `planUpload` (pure) | L0, `@graphty/graph-format` types and helpers |
| L2 kernel | `PipelineCache`, `Kernel` (pipeline + layout + dispatch), `DispatchPlanner` (pure), `UniformBlock`, `composeWgsl`, `Profiler` (timestamp queries) | L0 |
| L3 primitives | reduce, scan, segmented reduce, compaction, histogram / counting sort, radix sort, frontier + advance, SpMV pull, grid build (cell sort + pyramid), COO -> CSR | L1, L2 |
| L4 algorithms | plain async functions per algorithm, grouped by primitive family | L3 |
| L4 layouts | `ForceAtlas2Simulation`, `FruchtermanReingoldSimulation` implementing `LayoutSimulation` | L3 |
| L5 accelerator | `GpuAccelerator` object binding L4 to the structural interfaces; `release` | L4 |

Rule: a lower layer never imports a higher one; `src/wgsl/**` is imported
only by L2-L4; `test/**` may import any layer.

### 3.2 Directory tree (mirrors `packages/graph-io`, note 07 section 4.1)

```
packages/webgpu-graph-algorithms/
+-- package.json  project.json  webgpu-graph-algorithms.ts  tsconfig.json  tsconfig.build.json  tsconfig.strict-consumer.json  vitest.config.ts
+-- scripts/entries.js  scripts/build-bundle.js  scripts/bundle-types.js  scripts/gpu-report.mjs
+-- README.md  CLAUDE.md  LICENSE  docs/HEADLESS_GPU_REPORT.md (moved from the repo root)
+-- src/
|   +-- index.ts                      # explicit named exports only (browser-safe)
|   +-- node/index.ts                 # ./node subpath: createNodeGpuContext()
|   +-- errors.ts                     # WebGpuGraphError, codes
|   +-- constants.ts                  # WORKGROUP_SIZE 256, MAX_1D_INVOCATIONS 16_776_960, ARC_WINDOW_ALIGN 64, STAGING_RING 3
|   +-- types/                        # public option/result types; accelerator.ts (structural copies of the CPU-package interfaces until W1)
|   +-- device/  context.ts  caps.ts  errors-scope.ts
|   +-- memory/  residency.ts  upload-plan.ts  buffer-pool.ts  readback.ts
|   +-- kernel/  pipeline-cache.ts  kernel.ts  dispatch.ts  uniforms.ts  wgsl.ts  profiler.ts
|   +-- wgsl/    prelude.wgsl.ts  reduce.wgsl.ts  scan.wgsl.ts  segmented-reduce.wgsl.ts  compact.wgsl.ts  radix-sort.wgsl.ts
|   |            histogram.wgsl.ts  frontier.wgsl.ts  advance.wgsl.ts  spmv.wgsl.ts  grid.wgsl.ts  fa2.wgsl.ts  fr.wgsl.ts  bfs.wgsl.ts ...
|   +-- primitives/  reduce.ts  scan.ts  segmented-reduce.ts  compact.ts  histogram.ts  radix-sort.ts  frontier.ts  advance.ts  spmv.ts  grid.ts  coo-to-csr.ts
|   +-- algorithms/  degree.ts (skeleton) pagerank.ts hits.ts katz.ts eigenvector.ts wcc.ts bfs.ts sssp.ts closeness.ts betweenness.ts
|   |                bellman-ford.ts k-core.ts triangles.ts label-propagation.ts mst.ts apsp.ts louvain.ts
|   +-- layouts/     force-atlas2.ts  fruchterman-reingold.ts  seed.ts (LCG initial placement)  settle.ts
|   +-- accelerator.ts                # GpuAccelerator factory
+-- test/
|   +-- setup/gpu.ts                  # acquire() Dawn or navigator.gpu; REQUIRE flags; uncapturederror -> fail
|   +-- helpers/  device.ts  graphs.ts  oracle.ts  caps-tables.ts  matchers.ts  leak-counter.ts  frame-loop.ts
|   +-- device/ memory/ kernel/ primitives/ algorithms/ layouts/ *.test.ts     # node project
|   +-- browser/*.test.ts             # browser smoke project
|   +-- limits/*.test.ts              # node-limits project (GPU lane only)
|   +-- types/*.test-d.ts  index.test.ts  build-output.test.ts
+-- benchmarks/  run.ts  harness.ts  datasets.ts  layout.bench.ts  pagerank.bench.ts  bfs.bench.ts  upload.bench.ts  results/
```

### 3.3 Public TypeScript API (entry `.`)

```ts
// re-exported types from @graphty/graph-format used in signatures: GraphSnapshot, U32, F32, NodeMask, NumericVector
export { GpuContext, type GpuContextOptions, type GpuProbe, type GpuCaps } from "./device/context.js";
export { WebGpuGraphError, type WebGpuGraphErrorCode } from "./errors.js";
// createNodeGpuContext is exported from ./node only, never from "."

// ---- layouts (section 7)
export interface ForceAtlas2GpuOptions {
    readonly dim?: 2 | 3 | undefined;                   // default 2 (the element overrides from the view mode)
    readonly maxIter?: number | undefined;              // default 100: total iteration budget across step() calls
    readonly jitterTolerance?: number | undefined;      // 1.0
    readonly scalingRatio?: number | undefined;         // 2.0
    readonly gravity?: number | undefined;              // 1.0; 0 accepted
    readonly strongGravity?: boolean | undefined;       // false
    readonly distributedAction?: boolean | undefined;   // false
    readonly linlog?: boolean | undefined;              // false
    readonly nodeMass?: F32 | string | null | undefined;   // Float32Array(n), a node column name, or null -> outDegree()+1
    readonly weight?: true | string | null | undefined; // true -> snapshot.weights; string -> edge column expanded by the caller (section 9); null -> 1
    readonly seed?: number | null | undefined;          // LCG seed for NaN rows; 0 or null = unseeded (layout package quirk kept)
    readonly scale?: number | undefined;                // scene scale applied at write-back (element scalingFactor, default 100)
    readonly center?: ArrayLike<number> | null | undefined;
    readonly repulsion?: "exact" | "grid" | "auto" | undefined;   // default "auto"
    readonly exactMaxNodes?: number | undefined;        // default 16_384 (re-fixed by phase 3 measurement)
    readonly iterationsPerStep?: number | undefined;    // default 1; step(k) overrides
    readonly settleThreshold?: number | undefined;      // scene units per node, default 0.05 (NGraphLayoutEngine.ts lines 184-190)
    readonly settleWindow?: number | undefined;         // iterations averaged, default 10
    readonly nodeSize?: never;                          // adjustSizes deferred (note 01 section 7.2); typed `never` so a caller cannot pass it silently
    readonly dissuadeHubs?: boolean | undefined;        // accepted and ignored, like the CPU port
}
export function createForceAtlas2(ctx: GpuContext, options?: ForceAtlas2GpuOptions): GpuLayoutSimulation;
export function createFruchtermanReingold(ctx: GpuContext, options?: FruchtermanReingoldGpuOptions): GpuLayoutSimulation;
export interface GpuLayoutSimulation extends LayoutSimulation {   // LayoutSimulation is declared structurally in src/types until L1 exports it
    load(snapshot: GraphSnapshot, positions: F32): void;
    step(iterations?: number): Promise<void>;          // always a Promise here; one submission, one readback per call
    readonly settled: boolean;
    readonly iterationsDone: number;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;   // scene units in; 12-byte writeBuffer of layout units
    dispose(): void;
    readonly trace: ForceAtlas2Trace | null;           // last step's { swing, traction, speed, speedEfficiency, movement } per iteration, for tests
}

// ---- algorithms (section 8); every function: (ctx, snapshot, options?, dest?) -> Promise<result>
export function degree(ctx: GpuContext, s: GraphSnapshot, dest?: U32): Promise<U32>;                       // walking skeleton
export function pageRank(ctx: GpuContext, s: GraphSnapshot, o?: PageRankGpuOptions, dest?: F32): Promise<{ scores: F32; iterations: number; converged: boolean }>;
export function personalizedPageRank(ctx: GpuContext, s: GraphSnapshot, personalization: F32, o?: PageRankGpuOptions, dest?: F32): Promise<...>;
export function hits(ctx, s, o?): Promise<{ hubs: F32; authorities: F32; iterations: number; converged: boolean }>;
export function eigenvectorCentrality(ctx, s, o?, dest?): Promise<{ scores: F32; iterations: number; converged: boolean }>;
export function katzCentrality(ctx, s, o?, dest?): Promise<{ scores: F32; iterations: number; converged: boolean }>;
export function connectedComponents(ctx, s, dest?): Promise<{ labels: U32; count: number }>;              // WCC; labels dense, first-seen order
export function breadthFirstSearch(ctx, s, source: number, o?): Promise<{ depth: U32; parent: U32; order: U32; visitedCount: number }>;
export function sssp(ctx, s, source: number, o?): Promise<{ dist: F32; predArc: U32 }>;                    // non-negative weights; near-far
export function bellmanFord(ctx, s, source: number, o?): Promise<{ dist: F32; predArc: U32; hasNegativeCycle: boolean }>;
export function closenessCentrality(ctx, s, o?, dest?): Promise<F32>;
export function betweennessCentrality(ctx, s, o?: { sources?: U32 | number; normalized?: boolean; edges?: boolean }, dest?): Promise<{ nodes: F32; edges: F32 | null }>;
export function kCore(ctx, s, dest?): Promise<U32>;
export function triangleCount(ctx, s, dest?): Promise<{ perNode: U32; total: number }>;
export function labelPropagation(ctx, s, o?, dest?): Promise<{ labels: U32; iterations: number; converged: boolean }>;
export function minimumSpanningTree(ctx, s): Promise<{ edges: U32; totalWeight: number }>;
export function allPairsShortestPath(ctx, s, dest?): Promise<F32>;                                         // n*n, bounded by maxBufferSize
export function louvain(ctx, s, o?): Promise<{ labels: U32; count: number; modularity: number; iterations: number }>;

// ---- accelerator object (section 9.2)
export interface GpuAccelerator extends AlgorithmAccelerator, LayoutAccelerator { release(s: GraphSnapshot): void; dispose(): void; readonly ctx: GpuContext; }
```

Conventions: options objects use `?: T | undefined` (note 07 section 7);
absent output is `null`; `dest` arrays must be `<ArrayBuffer>`-backed and of
exact length (`E_BAD_ARGUMENT` otherwise); node arguments are INDICES (ids
are resolved by the caller through `snapshot.ids.requireIndex`, note 07
section 1.8).

### 3.4 WGSL source organisation and composition

- Each kernel family is a `src/wgsl/<name>.wgsl.ts` module exporting a
  function `(v: Variant) => string` that returns complete WGSL for that
  variant, built from `/* wgsl */` tagged template pieces. Variants are
  small frozen objects (`{ dim: 2 | 3; useWeights: boolean; usePerm: boolean;
  subgroups: boolean; tier: "thread" | "workgroup" }`) and also the
  PipelineCache key.
- `src/wgsl/prelude.wgsl.ts` supplies: `const INVALID_INDEX: u32 =
  0xFFFFFFFFu;`, `override WG: u32 = 256u;`, `override USE_PERM: bool =
  false;`, `override HAS_WEIGHTS: bool = false;`, `fn lowbias32(x: u32) ->
  u32` (integer hash; cosmos found `sin()` hashes diverge across vendors,
  note 03 section 1.3), `fn unpack_u8(words, i) -> u32`, `fn mask_bit(mask,
  i) -> bool`, `fn pos3(p: ptr<storage, array<f32>>, i: u32) -> vec3<f32>`
  (the `3*i` read), and the 16-byte-aligned uniform struct helpers.
- `composeWgsl({ enables, prelude, structs, body })` concatenates; `enable
  subgroups;` is spliced only when `caps.features.has("subgroups")` (WGSL
  `enable` fails compilation otherwise, note 05 section 6 item 7).
- Gunrock-style operators (`advance`, `filter`, `neighborReduce`) take a WGSL
  SNIPPET string implementing a fixed-signature function (`fn visit(u: u32,
  v: u32, a: u32, w: f32) -> bool`) that the composer inserts into the
  operator template; the composed string is hashed into the cache key.
- Uniform structs are generated from `UniformBlock` descriptors (section
  5.3) so the WGSL text and the byte layout can never disagree.

### 3.5 Class list (few classes, plain functions elsewhere)

| Class | Responsibility (one line) |
| --- | --- |
| `GpuContext` | Owns the device, caps, error listeners, the residency, the pipeline cache, the buffer pool, the staging ring; `accelerator()`, `release()`, `dispose()` |
| `GraphResidency` | Uploads a snapshot's core, views and columns once per array object; hands out `{ buffer, offset, size }` bindings; tracks every buffer per snapshot for `release` |
| `BufferPool` | Size-class recycling of scratch `GPUBuffer`s with usage flags; `acquire(bytes, usage)` / `release(buf)`; destroyed on dispose |
| `Readback` | Ring of N `MAP_READ` staging buffers; `copy(encoder, src, offset, size) -> slot`; `await slot.map() -> typed array copy` |
| `PipelineCache` | `get(kernelId, variant) -> Kernel` compiled once per variant; `pushErrorScope("validation")` around creation; `getCompilationInfo()` errors become `E_VALIDATION` |
| `Kernel` | A compute pipeline plus its bind-group-layout descriptor; `bind(entries) -> GPUBindGroup` (cached per entry identity), `dispatch(pass, count)` via the planner |
| `UniformBlock` | Declares fields with WGSL types, emits the struct text and writes bytes with correct 16-byte alignment |
| `Frontier` | Two vertex queues, an edge queue, a length atomic, an indirect-args buffer, the `count -> workgroups` finalise kernel |
| `ForceAtlas2Simulation`, `FruchtermanReingoldSimulation` | `LayoutSimulation` implementations (section 7) |
| `WebGpuGraphError` | `code`, frozen `details`, `name` |

Everything else (`planUpload`, `planDispatch`, primitives, algorithms,
`createNodeGpuContext`, `isSoftwareAdapter`) is a function.

---------------------------------------------------------------------------

## 4. Memory and upload

### 4.1 GraphResidency: the three upload paths (design 10.3, note 07 section 3)

`planUpload(caps, snapshot): UploadPlan` is a pure function of
`device.limits` (never `adapter.limits`: Dawn-node reports the raw Vulkan
adapter limits, e.g. 1 TiB `maxBufferSize`, note 05 section 2.4) and the
snapshot's byte lengths:

| Path | Condition | Buffers | Binding |
| --- | --- | --- | --- |
| `arena` | `arena !== null` AND `arena.hotByteLength <= limits.maxBufferSize` (or `byteLength` when a cold segment is needed) AND every non-null `segment.byteLength <= limits.maxStorageBufferBindingSize` | ONE `createBuffer` + ONE `writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength))` | `{ buffer: gbuf, offset: seg.byteOffset - arena.byteOffset, size: seg.byteLength }`; offsets are 256-aligned by construction (design 10.3), which satisfies any `minStorageBufferOffsetAlignment <= 256` (16 under Dawn-node); never assert `=== 256` |
| `perArray` | `arena === null` (the NORMAL case for `fromCsr` on separate arrays and `transpose()`, note 07 section 2 item 5) or the arena exceeds a limit but every array fits `maxStorageBufferBindingSize` | one buffer per array, `writeBuffer(buf, 0, snapshot.colIdx)` etc. | whole buffer |
| `windowed` | some array's `byteLength > limits.maxStorageBufferBindingSize` (128 MiB default: 33,554,432 arcs; lavapipe cannot raise it, note 06 section 3.5) | per-array buffers as above (a buffer may be larger than a binding up to `maxBufferSize`), or several buffers when `> maxBufferSize` | ARC windows `[arcStart, arcEnd)` with `arcStart = rowPtr[v0] - (rowPtr[v0] % 64)` (`%`, never `& ~63`, design 10.6); the kernel receives `rebase = arcStart` as a uniform scalar; a row longer than a window is split across windows and the kernel's row loop is clamped to `[max(rowPtr[u], arcStart), min(rowPtr[u+1], arcEnd))` |

Cold segments (`arcToEdge`, `edgeToArc`) are uploaded lazily by the first
kernel that needs them (edge-column gathers, per-edge write-back); in the
arena path a second `writeBuffer` fills `[hotByteLength, byteLength)` of the
same buffer. `weights === null` -> no buffer; the kernel variant has
`HAS_WEIGHTS = false` and the slot is filled with `colIdx` as the never-read
dummy (the `USE_PERM` pattern applied to weights, note 03 section 8.5).
Identity permutations: test `flags.arcToEdgeIsIdentity` and `!directed`
BEFORE touching `arcToEdge`, `edgeToArc`, `coo().arcToEdge`,
`edgeList().arc` or `reverse().fwdArc` (all getters that allocate, note 07
section 2 items 1-2).

Windowed kernels in v1: the row-walking kernels (degree, attraction,
PageRank pull, segmented reduce) support windows from phase 2; frontier
kernels throw `E_UNSUPPORTED` with `details.reason = "windowed"` until phase
9 adds window-aware advance (risk R-9).

### 4.2 Views and columns: keys and invalidation

| Upload | Cache key | Invalidated when |
| --- | --- | --- |
| core (`rowPtr`, `colIdx`, `weights`, cold perms) | the `rowPtr` array object (core identity; `withColumns()` snapshots share it, note 07 section 1.3 `serial`) | `release(snapshot)` only |
| view (`reverse()`, `coo().src`, `edgeList()`, `outDegree()`, `degreeOrder().perm`, `mate()`) | the view's typed-array object | `release(snapshot)`; after `dropCaches()` the view object changes and the old buffer stays until `release` (section 4.6) |
| column (`table.gpuView(name)`) | `(gpuView array object, column.version)` | `column.version` changes (`markDirty()` / `setAll()`), `dropCaches()` (new f64 copy), `release(snapshot)` |
| position column (element-owned, role `position`, mutable) | NOT uploaded by the residency: the layout simulation owns the position buffer (section 7.4) | -- |

Two `WeakMap`s: `arrays: WeakMap<ArrayBufferView, Resident>` (array object
-> `{ buffer, byteLength, version }`) and `records: WeakMap<GraphSnapshot,
ResidencyRecord>` (snapshot -> every buffer created on its behalf, including
buffers whose CPU key was dropped by `dropCaches()`), so `release(snapshot)`
can enumerate and destroy everything (note 07 section 2 item 3).

`ResidentGraph` (what kernels receive):

```ts
interface Binding { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number; }
interface ResidentGraph {
    readonly snapshot: GraphSnapshot; readonly plan: UploadPlan;
    readonly rowPtr: Binding; readonly colIdx: Binding | null;        // null when arcCount === 0 (never bind zero length, design 10.5)
    readonly weights: Binding | null; readonly arcToEdge: Binding | null; readonly edgeToArc: Binding | null;
    readonly windows: readonly ArcWindow[];                              // one window covering everything in the arena / perArray paths
    view(name: "reverse" | "coo" | "edgeList" | "outDegree" | "degreeOrder" | "reverseDegreeOrder" | "mate"): ResidentView;
    column(table: "nodes" | "edges", name: string): Binding;            // gpuView() upload keyed on version
}
```

### 4.3 BufferPool

Scratch buffers (forces, partials, frontier queues, sort scratch, staging)
come from a pool keyed on `(usage, sizeClass)` with size classes at powers of
two from 4 KiB up and exact sizes above 64 MiB. `acquire()` wraps
`createBuffer` in `pushErrorScope("out-of-memory")`; a non-null scope result
becomes `E_OUT_OF_MEMORY` with `details.bytes` (the 1 TiB `maxBufferSize`
Dawn-node reports is not physical memory, note 05 section 4). Buffers are
labelled (`label: "fa2.force"`) so validation messages name them. `dispose()`
destroys everything; tests assert `pool.liveBytes === 0` afterwards.

### 4.4 Readback

- Ring of `STAGING_RING = 3` `MAP_READ | COPY_DST` staging buffers per size
  class, so a `mapAsync` on iteration k's slot overlaps the submission of
  k+1 (`mapAsync` on a buffer still in use by a queued copy is a validation
  error, note 05 section 7.2; `buffer.mapState` guards reuse).
- The idiom: `copyBufferToBuffer(src, off, staging, 0, size)` in the same
  command buffer as the kernels; `submit`; `await staging.mapAsync(READ)`;
  `dest.set(new Float32Array(staging.getMappedRange(0, size)))`; `unmap()`.
  Copy BEFORE unmap (the mapped range detaches, design 10.7).
- Never poll `onSubmittedWorkDone` in addition to `mapAsync` (it costs
  another ~0.1 ms in Chromium, note 05 section 7.2).
- Results are fresh `Uint32Array<ArrayBuffer>` / `Float32Array<ArrayBuffer>`
  or the caller's `dest` (design 10.7; type parameter load-bearing, note 07
  section 1.2).
- Measured latencies to design around (note 05 section 7.2): 4-byte round
  trip 0.04 ms Dawn-node NVIDIA / 0.10 ms Chromium NVIDIA / 0.15 ms
  SwiftShader; 1 MiB copy + map + slice 2.65 ms in Chromium.

### 4.5 Chunking and the 10M tier

At 10M nodes / 100M edges undirected (`colIdx` 800 MB) the arena path is
impossible at defaults and the design already prescribes `arena: false`
(design 15.3). The residency then uploads per array in `maxBufferSize`-sized
pieces and the row-walking kernels iterate windows: `for (w of
resident.windows) kernel.dispatch(rows in w)`. This is a Node batch tier
(section 10); the gate for it is a correctness test on lavapipe with FAKED
limits forcing 3+ windows on a 100k graph, plus one real run on the 4070 with
requested 2 GiB bindings (`node-limits` project).

### 4.6 Release lifecycle

- `ctx.release(snapshot)`: destroy every buffer in `records.get(snapshot)`,
  delete the record and every `arrays` entry pointing at those buffers; a
  running simulation that still binds them is disposed first (it holds a
  reference to the record and is notified). graphty-element calls it from
  `snapshot-replaced` (design 14.4 lines 4121-4126).
- `simulation.dispose()`: destroys the simulation's own buffers (positions,
  forces, grid) and returns pooled scratch; it does NOT release the
  snapshot (an algorithm may still use it).
- `ctx.dispose()`: releases every record, destroys the pool and the staging
  ring, destroys the device if `create()` made it (not for `from()`), drops
  the Dawn `GPU` reference in Node.
- Device loss (`device.lost`): the context marks `state = "lost"`, every
  pending `step()` / algorithm promise rejects with `E_DEVICE_LOST`, buffers
  are treated as gone (no destroy calls), and the caller decides whether to
  create a new context and `load()` again (note 05 section 7.4). Tests
  trigger it deterministically with `device.destroy()` mid-run.

### 4.7 Bytes per node and per edge on the device

Snapshot core (undirected, A = 2E): `rowPtr` 4(n+1) + `colIdx` 4A + `weights`
4A when weighted (design 15.1). Layout and algorithm state:

| Component | Bytes | Notes |
| --- | --- | --- |
| FA2 exact tier | 12 pos + 12 force + 12 oldForce + 4 mass + 12 writeback out + partials 16/256 per node + mask 1/8 = ~52 B/node | old force is needed by the Gephi swing/traction law (P-5) |
| FA2 grid tier | exact + 4 cellId + 4 sortedIdx + 4 cellCursor(per cell) = ~64 B/node + pyramid: 2D 512^2 x 16 B x 1.33 = 5.6 MB fixed; 3D 128^3 x 16 B x 1.14 = 38 MB fixed | note 03 section 8.2 |
| FR | 12 pos + 12 disp + 12 out + mask = ~37 B/node | |
| PageRank | 2 x 4 rank + 4 outWeightSum + partials = ~12 B/node + reverse view (0 when undirected) | |
| BFS / SSSP | 4 depth + 4 parent + 2 x 4 queues + 4 edge queue per arc (two-phase) + bitset = ~16 B/node + 4 B/arc | |
| Betweenness (batch k sources) | 8 x n x k (sigma u32 + dist u32) + 4 x n x k delta + queues | batch size planned from `maxBufferSize` (note 04 section 6) |
| WCC (Afforest) | 4 comp + 4 sample histogram (1024 words) | |

Worked totals at 100k nodes / 1M edges undirected weighted: core 16.4 MB
(hot prefix, design 10.3), FA2 exact 5.2 MB, FA2 grid 6.4 + 5.6 MB; at 1M /
10M: core 164 MB, FA2 grid 64 + 5.6 MB (2D). Section 10 tabulates all tiers.

---------------------------------------------------------------------------

## 5. Kernel infrastructure

### 5.1 PipelineCache

Key = `kernelId + JSON(variant) + hash(composed WGSL)` (override constants
make distinct pipelines, note 05 section 6 item 3). `createShaderModule` and
`createComputePipeline` run inside `pushErrorScope("validation")`;
`module.getCompilationInfo()` messages of type `error` are collected into
`E_VALIDATION.details.messages` with line numbers, so a WGSL typo fails the
first test that uses it instead of printing to stderr (Dawn-node behaviour,
note 05 section 2.2). Pipelines are created with explicit
`GPUBindGroupLayout`s (never `layout: "auto"`) so one layout serves the
`USE_PERM` / `HAS_WEIGHTS` dummy-binding variants. Async
`createComputePipelineAsync` is used at warm-up (`ctx.warm(["fa2", "pagerank"])`)
so the first frame does not stall on shader compilation.

### 5.2 GpuCaps and DispatchPlanner

```ts
export interface GpuCaps {
    readonly limits: GPUSupportedLimits;            // device.limits after raiseLimits
    readonly features: ReadonlySet<string>;
    readonly subgroupMin: number; readonly subgroupMax: number;   // adapter.info.subgroup{Min,Max}Size, 0 when absent
    readonly software: boolean; readonly vendor: string; readonly architecture: string;
    readonly runtime: "browser" | "node";
    readonly wgslFeatures: ReadonlySet<string>;      // uniform_buffer_standard_layout is present in Dawn-node but NOT Chromium 139: never rely on it
}
```

Limits requested at `create()` (each clamped to `adapter.limits`, only when
`raiseLimits`): `maxBufferSize`, `maxStorageBufferBindingSize`,
`maxStorageBuffersPerShaderStage`, `maxComputeWorkgroupStorageSize`,
`maxComputeInvocationsPerWorkgroup`, `maxComputeWorkgroupSizeX`. Kernels are
DESIGNED for the spec defaults (8 storage buffers per stage, 16 KiB workgroup
memory, 256 invocations, 65,535 workgroups per dimension) and use raised
values only through variants selected from `caps` (note 05 section 4).

`planDispatch(caps, count, wg = 256): DispatchShape`:

| Count | Shape | Kernel side |
| --- | --- | --- |
| `ceil(count / wg) <= maxComputeWorkgroupsPerDimension` | `{ kind: "1d", x }` | `gid.x` |
| otherwise | `{ kind: "2d", x: 65535, y: ceil(groups / 65535) }` | `linear = gid.x + gid.y * 65535u * WG` and `if (linear >= n) { return; }` |
| memory-bound elementwise kernels above 2^26 items | `{ kind: "stride", x: min(groups, 4096) }` grid-stride loop | `for (i = gid.x; i < n; i += stride)` |

The boundary `(16,776,960, 16,777,216]` is a unit test (design 10.6:
a planner using 2^24 silently drops 256 items); the 2D shape is exercised for
real on a 17M-element elementwise kernel (68 MB, fits lavapipe) in the node
project, and on 100M elements in `node-limits`.

Bind-group budget rule: group 0 = graph (immutable per snapshot), group 1 =
algorithm state, group 2 = per-dispatch params; at most 8 storage buffers
per stage per kernel (PageRank with personalization uses exactly 8, note 04
section 5; cold arrays go to a second group).

### 5.3 Uniforms packing

`UniformBlock.define([["n", "u32"], ["k", "f32"], ["center", "vec3f"], ...])`
emits `struct Params { n: u32, k: f32, _p0: vec2<u32>, center: vec3<f32>,
_p1: f32 }` with explicit padding so the layout is legal WITHOUT
`uniform_buffer_standard_layout` (absent in Chromium 139, note 05 section
2.4), and `write(view: DataView, values)` fills the bytes. Per-dispatch
params use dynamic offsets at 256-byte stride (`minUniformBufferOffsetAlignment`
is 256 in Chromium, 64 / 16 under Dawn-node; 256 satisfies all) so k
iterations of a layout can each read their own `iter` scalar from one
buffer written once. `bool` is not host-shareable (WGSL 6.5.2): flags are
`u32`.

### 5.4 Indirect dispatch

`dispatchWorkgroupsIndirect(buffer, offset)` with a 12-byte `INDIRECT` buffer
written by a one-workgroup "finalise" kernel: `x = min(ceil(len / WG),
65535)`, `y = ceil(ceil(len / WG) / 65535)`, `z = 1`, and it also copies
`len` into the next round's uniform. This is what lets BFS / SSSP / k-core
record k rounds per `queue.submit()` with no host readback; the spec text
"an over-limit count does nothing" (note 05 section 12 item 7) is NOT relied
on because the finalise kernel clamps.

### 5.5 Timestamp queries and the Profiler

When `caps.features.has("timestamp-query")`, `ctx.profiler` wraps compute
passes with `timestampWrites` into a `GPUQuerySet(2 * passes)`, resolves into
a buffer read back by benchmarks only. Chromium quantises to 100 us (note 05
section 3.1), Dawn-node quantisation is unverified (risk R-15), so per-kernel
timings come from the Node benchmarks and wall-clock `performance.now()`
around `await step()` is the number the gates use.

### 5.6 Error handling

```ts
export type WebGpuGraphErrorCode =
    | "E_NO_ADAPTER" | "E_NO_DEVICE" | "E_DEVICE_LOST" | "E_VALIDATION" | "E_OUT_OF_MEMORY"
    | "E_UNSUPPORTED" | "E_BAD_ARGUMENT" | "E_LIMIT" | "E_RELEASED" | "E_DISPOSED" | "E_NOT_LOADED";
export class WebGpuGraphError extends Error { readonly code: WebGpuGraphErrorCode; readonly details: Readonly<Record<string, unknown>>; }
```

- Creation calls (`createBuffer` above 16 MiB, shader modules, pipelines,
  bind groups) are wrapped in error scopes and converted to typed errors.
- `device.addEventListener("uncapturederror", ...)` records the first error
  on the context (`ctx.lastError`) and, in tests, fails the test (the setup
  file installs the listener). Every buffer, pipeline and pass carries a
  `label`.
- Device loss: section 4.6. `E_RELEASED` when a kernel is asked to bind a
  snapshot whose residency was released; `E_NOT_LOADED` when `step()` runs
  before `load()`; `E_LIMIT` when a planner cannot satisfy the request
  (e.g. APSP `n * n * 4 > maxBufferSize`) with the numbers in `details`.
- Out-of-bounds storage reads are clamped by WebGPU, so a kernel bug returns
  wrong numbers rather than an error (note 05 section 6 item 6): every kernel
  test compares against a CPU oracle, never just "no error".

---------------------------------------------------------------------------

## 6. Primitives

Each primitive is a function over a `GpuContext` and existing bindings,
records into a caller-supplied `GPUComputePassEncoder` (so k iterations of an
algorithm are one pass / one command buffer), and has a CPU reference in
`test/helpers/oracle.ts` (index-based, O(n + m), 10-40 lines each). "Pulled
in by" names the first slice that needs it (section 13).

| Primitive | Interface (TS) | WGSL strategy | Complexity | CPU reference | Pulled in by |
| --- | --- | --- | --- | --- | --- |
| `reduce` | `reduce(pass, { input: Binding, count, op: "sum" \| "min" \| "max" \| "sumVec4", out: Binding })` | workgroup tree reduce of 256 items into `partials[wg]` (vec4 for fused sums), then a one-workgroup second pass; `subgroupAdd` variant when available; i32 fixed-point `atomicMin/Max` variant for bounding boxes (GraphWaGu `apply_forces.wgsl` lines 79-82, note 03 section 2.2) | O(n), 2 dispatches | `Array.reduce` in f64 | walking skeleton (P1), FA2 (P3) |
| `scan` | `exclusiveScan(pass, { input, count, out, blockSums })` | reduce-then-scan: block scan of 256 in workgroup memory (Hillis-Steele or Blelloch), scan of block sums (recursive for > 65k blocks), add-back; NO decoupled look-back in v1 (WGSL atomics are relaxed; note 04 section 2.1) | O(n), 3 dispatches per level | prefix loop | grid build (P5), frontier (P9) |
| `segmentedReduce` | `segmentedReduce(pass, { rowPtr, values: Binding \| snippet, count: n, tiers: DegreeTiers, out })` | per-row gather with three tiers from `degreeOrder().segmentOffsets` (1024 / 32 / 1, `views.ts` lines 41-45): workgroup-per-row (hi), subgroup-per-row (mid, only with `subgroups`), thread-per-row (low); rows permuted through `perm` with `USE_PERM`; Kahan-style compensated add in the hub loop | O(A), 1-3 dispatches | loop over `rowPtr` | FA2 attraction (P3), PageRank (P7) |
| `compact` | `compact(pass, { flags \| predicate snippet, count, out, outCount })` | flag -> scan -> scatter; Davidson "ownership" dedupe for vertex queues (`owner[v] = idx` then keep iff `owner[v] === idx`, no atomics; note 04 section 2.3) | O(n), 4 dispatches | `filter` | frontier (P9), k-core (P11) |
| `histogram` / counting sort | `countingSort(pass, { keys, count, buckets, outIndex, outStart })` | `atomicAdd(&count[key], 1u)`, scan, scatter with `atomicAdd(&cursor[key], 1u)`; order INSIDE a bucket is nondeterministic, optional per-bucket index sort restores determinism | O(n + buckets), 4-6 dispatches | JS counting sort | grid build (P5), COO->CSR (P11) |
| `radixSort` | `radixSort(pass, { keys, values?, count, bits: 32 \| 64 })` | LSD 8-bit x 4 passes (histogram + scan + scatter), the GraphWaGu / Fuchsia shape (`repos/GraphWaGu/src/webgpu/sort.ts`, note 03 section 2.2); 64-bit keys as two 32-bit passes | O(4n), 12 dispatches | `Array.sort` on keys | Louvain contraction (P11), BH experiment |
| `Frontier` + `advance` | `frontier.advance(pass, { visit: snippet, mode: "twoPhase" \| "fused" })` | Gunrock `block_mapped`: workgroup scan of 256 degrees + `upper_bound` per output slot (`repos/gunrock/.../advance/block_mapped.hxx` lines 123-165, note 04 section 2.5), workgroup-per-row tier for degree >= 1024 from `segmentOffsets`, subgroup tier when present; next-frontier length via one `atomicAdd` per workgroup; indirect args by the finalise kernel | O(frontier degrees), 2-4 dispatches per level | level-synchronous BFS loop | BFS (P9) |
| `spmvPull` | `spmvPull(pass, { graph: reverse view, xIn, yOut, combine: snippet })` | `segmentedReduce` over `reverse().rowPtr / colIdx / weights` with `USE_PERM` for `fwdArc` (identity when undirected); tiered by in-degree (`degreeOrder({ of: "reverse" })`) | O(A) | loop | PageRank (P7) |
| `grid` (cell sort + pyramid) | `grid.build(pass, { positions, mass, n, dim, finestCells })`, `grid.downsample`, `grid.farField`, `grid.nearField` | bbox (fixed-point atomics) -> cell id + histogram -> scan -> scatter -> per-cell mass-weighted centroid (segmented reduce over the sorted order, deterministic) -> per-level downsample (one dispatch per level, no atomics) -> far field fixed 3x3/6x6 (27/216 in 3D) loops -> near field over sorted cell ranges with `NEAR_MAX` cap and Horvitz-Thompson weighting (note 03 sections 1.3, 8.4) | O(n + cells x levels) | brute-force O(n^2) forces (the exact kernel is ALSO an oracle) | FA2 large-n (P5) |
| `cooToCsr` | `cooToCsr(pass, { src, dst, weights?, n, m }) -> device CSR` | counting sort by `src`, scan, scatter, optional per-row sort by target | O(m) | `fromCsr` on the CPU for parity | Louvain (P11) |
| `bitset` | `bitset.set / test / clearAll` | `atomicOr` on `array<atomic<u32>>`; bulk non-atomic path when the frontier is >= 40 % of n (cuGraph `bfs_impl.cuh` lines 729-765, note 04 section 2.8) | O(n / 32) | `Uint32Array` ops | BFS bottom-up (P9) |

Determinism policy: `reduce`, `scan`, `segmentedReduce`, `spmvPull`, the
grid pyramid (with per-bucket index sort) and `radixSort` are bitwise
reproducible on the same device; `compact` output order and counting-sort
in-bucket order are not, and tests compare them as sets.

---------------------------------------------------------------------------

## 7. Force-directed layouts -- FIRST DELIVERABLE

### 7.1 What "drop-in" means (from note 01 section 5.1)

The GPU ForceAtlas2 must (1) take the same options as the CPU FA2 of the L1
rewrite so `ForceAtlas2LayoutEngine`'s zod config (`graphty-element/src/layout/ForceAtlas2LayoutEngine.ts`
lines 99-115) maps 1:1; (2) apply the same forces and the same adaptive
speed controller, so N GPU steps and N CPU iterations from the same seeded
start give statistically the same picture; (3) report `settled` truthfully
and in bounded time (screenshots and label animations wait for it,
`graphty-element/src/screenshot/ScreenshotCapture.ts` lines 332-380 per note
01 section 4.3); (4) honour `setFixed` every step and `setPosition` during a
drag, and un-settle on both; (5) `load` again after a topology change
without losing placed coordinates; (6) leave `z` untouched in 2D; (7) be
deterministic given `seed` and the snapshot on the same device; (8) free
everything on `dispose()`.

### 7.2 Buffers

All positions are `array<f32>` read with `3u * i + k` (design C14; stride 3
whatever `dim` is). Layout units on the device (P-7).

| Buffer | Bytes | Usage | Source / update |
| --- | --- | --- | --- |
| `pos` | 12n | STORAGE RW, COPY_DST | `load()`: NaN rows seeded by the CPU LCG, finite rows converted scene -> layout; `setPosition`: 12-byte `writeBuffer` |
| `force` | 12n | STORAGE RW | zeroed by the integrate kernel each iteration |
| `oldForce` | 12n | STORAGE RW | `F(t-1)` for swing / traction (P-5) |
| `mass` | 4n | STORAGE R | `outDegree()[i] + 1` computed on the CPU from `rowPtr` at `load()` (one O(n) loop) or the `nodeMass` vector / column |
| `fixed` | 4 ceil(n/32) | STORAGE R, COPY_DST | `setFixed(mask)`: whole-mask `writeBuffer` (12.5 KB at 100k) |
| `rowPtr`, `colIdx`, `weights` | from the residency | STORAGE R | undirected snapshot; both arcs present so the gather needs no atomics |
| `perm` (degree order) | 4n | STORAGE R | `degreeOrder().perm` for the tiered attraction gather; `USE_PERM = false` variant when every row is below the mid tier |
| `partials` | 16 x ceil(n/256) | STORAGE RW | per-workgroup `vec4f`: (sumSwing, sumTraction, sumMovement, unused) |
| `centroidPartials` | 16 x ceil(n/256) | STORAGE RW | per-workgroup (sum x, sum y, sum z, count) of NEW positions |
| `state` | 128 | STORAGE RW, COPY_SRC | `{ speed, speedEfficiency, totalSwing, totalTraction, centroid: vec3f, movement, iter: u32, settledFlag: u32, movementRing: array<f32, 10> }` (the ring feeds the settle window of section 7.8) |
| `out` | 12n | STORAGE RW, COPY_SRC | write-back kernel output in SCENE units |
| `params` | 256 x k | UNIFORM, dynamic offset | per-iteration scalars (section 7.6) |
| grid tier extras | section 7.5 | | |

Bindings per kernel stay <= 8 storage buffers (the attraction kernel binds
`rowPtr, colIdx, weights|dummy, perm|dummy, pos, mass, force, state` = 8).

### 7.3 Force laws and the reference semantics (P-5)

Symbols: `m_i = mass[i]`, `d = |p_i - p_j|` with softening `d' = max(d,
0.01)` as the CPU port does (`layout/src/layouts/force-directed/forceatlas2.ts`
line 266, note 01 section 2.1.3), `kr = scalingRatio`, `kg = gravity`,
`w_a = weights ? weights[a] : 1`.

| Force | Law (magnitude, direction) | Reference | Kernel |
| --- | --- | --- | --- |
| repulsion | `kr * m_i * m_j / d'` pushing i away from j (component update `delta * kr m_i m_j / d'^2`) | Jacomy 2014 eq. for Fr; Gephi `ForceFactory.java` lines 132-148; cuGraph `bh_kernels.cuh` (note 03 sections 4.3, 5) -- NOT the port's `1/d^2` (note 01 section 2.1.9) | `fa2_repulse_exact` or grid kernels |
| attraction (linear) | `w_a * d` toward j; linlog: `w_a * log(1 + d)`; `distributedAction`: divide by `m_i` | Jacomy 2014; NetworkX; the CPU port lines 271-307 | `fa2_attract` (CSR gather) |
| gravity | regular: `kg * m_i` toward the centroid `c` (unit direction, 0 when `norm(p_i - c) <= 0.01`); strong: `kg * m_i * norm(p_i - c)` | CPU port lines 335-364 (centroid); Gephi uses the origin -- the plan keeps the centroid for parity with the port (open question Q-3) | fused into `fa2_attract` |
| swing / traction | `swg_i = m_i * norm(F_i(t) - F_i(t-1))`, `tra_i = m_i * norm(F_i(t) + F_i(t-1)) / 2`; global sums reset every iteration | Jacomy 2014 section "Adaptive speed"; Gephi `ForceAtlas2.java` lines 296-328; cuGraph `fa2_kernels.cuh` `compute_local_speed` (note 03 section 4.3 item 9) | `fa2_swing` (fused into attract) |
| global speed | `estimateFactor` exactly as the CPU port lines 184-230 (identical to NetworkX and a line-by-line port of Gephi): `optJitter = 0.05 sqrt(n)`, `minJitter = sqrt(optJitter)`, `maxJitter = 10`, `jitter = jitterTolerance * max(minJitter, min(maxJitter, optJitter * traction / n^2))`; `if swing/traction > 2: speedEfficiency = max(0.05, speedEfficiency/2); jitter = max(jitter, jitterTolerance)`; `targetSpeed = swing == 0 ? +inf : jitter * speedEfficiency * traction / swing`; `if swing > jitter * traction: speedEfficiency = max(0.05, 0.7 speedEfficiency) else if speed < 1000: speedEfficiency *= 1.3`; `speed += min(targetSpeed - speed, 0.5 speed)` | note 01 section 2.1.3 item 8 | `fa2_adapt_speed` (one workgroup) |
| local speed / integrate | `factor = speed / (1 + sqrt(speed * swg_i))`; `p_i += F_i * factor` unless fixed; no `adjustSizes` in v1 | CPU port lines 403-428 | `fa2_integrate` |

The CPU rewrite in L1 adopts the same table (Layout port 2, design lines
4016-4029, already describes a CSR-row attraction and an all-pairs
repulsion). The GPU kernel takes each law as a documented constant; nothing
is rediscovered in WGSL. `dissuadeHubs` is accepted and ignored on both
sides (note 01 section 7.2). `nodeSize` / `adjustSizes` are rejected at the
type level in v1 (section 3.3).

### 7.4 Per-iteration kernel sequence (exact tier), k iterations per submission

```
for it in 0..k-1 (all recorded into ONE compute pass, dynamic uniform offset = 256 * it):
  K1 fa2_repulse_exact   (n/256 workgroups)  force[i]  = sum_j kr m_i m_j / d'^2 * (p_i - p_j)      tiled: 256 x vec4f(p, m) per tile in workgroup memory (4 KiB)
  K2 fa2_attract         (tiered: hi rows workgroup-per-row, mid subgroup-per-row if present, low thread-per-row)
                                             force[i] += sum_{a in row i} w_a * f(d) * (p_j - p_i) [/ m_i]  + gravity(p_i, state.centroid)
                                             swg_i, tra_i -> workgroup reduce -> partials[wg].xy
  K3 fa2_adapt_speed     (1 workgroup)       totalSwing, totalTraction = sum partials; estimateFactor -> state.speed, state.speedEfficiency;
                                             state.movement = sum partials.z (from the previous K4); state.centroid = sum centroidPartials / n;
                                             state.iter += 1; settledFlag per section 7.8
  K4 fa2_integrate       (n/256 workgroups)  swg_i recomputed from force/oldForce; factor = speed / (1 + sqrt(speed * swg_i)); if !fixed(i): p_i += F_i * factor (z untouched when dim == 2)
                                             oldForce = force; |dp_i| -> partials[wg].z; new p -> centroidPartials[wg]   (K1 overwrites force next iteration)
after the loop:
  K5 fa2_writeback       (n/256 workgroups)  out[3i+k] = p[3i+k] * scale + center[k]   (scene units)
  copyBufferToBuffer(out -> staging slot);  copyBufferToBuffer(state -> staging slot + 12n)
submit; await mapAsync; positions.set(out); read state (settled, trace); unmap
```

Four dispatches per iteration plus one write-back per `step()`; the swing
reduction is fused into K2 so no separate pass touches `force` again. The
whole `estimateFactor` (about 20 scalar operations) runs in K3 on the device,
so k iterations need NO host round trip (note 01 section 8.1 option 2;
cuGraph's `thrust::reduce` to host is unaffordable with asynchronous
submission, note 03 section 8.1 item 4). The centroid used by gravity in
iteration `it` is the centroid of the positions written by iteration
`it - 1` (one-iteration lag, identical to the CPU port's "center of mass
before the update" semantics since the port computes it at the start of
each iteration from the previous positions).

K1 exact tile kernel (the probe kernel `tmp/webgpu-plan/probe/dawn-perf.mjs`
is its skeleton; MEASURED 1.11 ms per iteration at n = 20,000 on the 4070
under Dawn, 388 ms on lavapipe, note 05 section 2.5): each invocation owns
node i; the workgroup streams the other nodes 256 at a time through
`var<workgroup> tile: array<vec4<f32>, 256>` (position xyz, mass); two
`workgroupBarrier()` per tile; the inner loop is branch-free except the
`j == i` skip; `d' = max(d, 0.01)` matches the port's floor. 3D costs the
same tile (vec4 already carries z).

Float atomics workaround: WGSL has atomics only on `u32` / `i32` (note 05
section 6 item 1), so nothing in the FA2 pipeline scatters. Repulsion is
computed per node by the owning invocation (K1); attraction is a per-node
GATHER over the undirected CSR row (both arcs are present by construction,
design 10.5), which replaces cuGraph's four `atomicAdd(float)` per COO edge
(`fa2_kernels.cuh` lines 77-80, note 04 section 12) and cosmos's two
in/out passes; the global sums are workgroup partials reduced by K3, not
atomics; the bounding box in the grid tier uses i32 fixed-point
`atomicMin/Max` (GraphWaGu's pattern), and the counting sort uses `u32`
`atomicAdd` on histogram and cursor arrays whose in-bucket order is made
deterministic by a per-bucket index sort. No kernel in the package performs
a read-modify-write on an f32 shared between invocations (the racy pattern
of d3-force-webgpu and GraphGPU, note 03 section 6, is explicitly rejected).

### 7.5 Repulsion back-ends and the crossover by n (P-6)

| n (discrete GPU) | Back-end | Why |
| --- | --- | --- |
| `n <= exactMaxNodes` (default 16,384; `repulsion: "auto"`) | exact tiled all-pairs | deterministic, no build, exact hubs, trivially 3D; Burtscher-Pingali: O(n^2) fastest below ~10k bodies (2009 GPU); cosmos: 4,096 (WebGL pass floor); note 03 section 8.2 recommends ~16k on a modern discrete GPU; the phase 3 gate measures ms/iteration at 4k / 8k / 16k / 32k and re-fixes the default so that the exact tier stays under ~4 ms/iteration |
| `n > exactMaxNodes` | grid pyramid (cosmos P3M as compute kernels) | build = counting sort + scan + scatter + segmented reduce + per-level downsample (all primitives the package needs anyway; no locks, no float atomics, deterministic); traversal = fixed loops (coarsest level minus 3x3, then 6x6 child block minus own 3x3 per level; 3x3x3 / 6x6x6 in 3D), no theta, no per-thread stack; production reference with documented failure modes (note 03 sections 1.3, 8.3) |
| integrated GPU | same code, `exactMaxNodes` halved by default when `caps.architecture` is not a discrete vendor string (heuristic, overridable) | note 03 section 8.2: integrated ~5-10x slower |
| software adapter | same code; tests scale fixtures (`gpuScale`) | correctness only |

Grid tier details (2D / 3D):

- finest grid: `2 * sqrt(n)` cells per axis rounded to a power of two,
  floor 8, cap 512 (2D) -> at most 262,144 cells; 3D: `2 * cbrt(n)` capped
  at 128 -> 2,097,152 cells (34 MB at 16 B per cell; 256^3 would be 268 MB
  and is not offered, note 03 section 8.2). Levels: 4^2 (4^3) up to the
  finest, log_2 ratio + 1 levels (up to 8 in 2D, 6 in 3D).
- bounding box per iteration by `reduce` (fixed-point i32 `atomicMin/Max`
  of `floor(x * 1000)` or two-level f32 reduce); NO clamping of scene
  positions (cosmos clamps to a fixed `spaceSize`; the element must not).
- cell centroids are MASS-weighted (`sum m p`, `sum m`) plus a count, so the
  far field applies `kr * m_i * M_cell / d'^2 * delta` -- the same 1/d law
  as the exact kernel with the cell's total mass (Gephi `Region.java`
  semantics).
- near field: for each of the 9 (27) finest cells around i, iterate the
  sorted index range exactly up to `NEAR_MAX = 64` entries per cell; above
  the cap take a hashed window of `NEAR_MAX` entries (lowbias32 of `(cell,
  iter, seed)`) weighted by `count / NEAR_MAX` (Horvitz-Thompson, unbiased;
  cosmos's fix for hub cells, note 03 section 1.3). Coincident points get a
  deterministic hashed kick. Per-iteration near-field step clamped to `2 *
  cellSize`.
- the locality sort is free: `sortedIdx` doubles as a cache-friendly
  traversal order for the attraction gather every k iterations (Burtscher
  kernel 4, note 03 section 8.3 item 5) -- an optimisation to measure, not
  a requirement.
- dispatch count per iteration: ~12 + levels, all in the same pass; still
  zero host round trips.

Parity: the approximate tier is tested against the EXACT GPU tier on the
same start (force-field RMS error and final-layout distributional metrics),
never against coordinates (section 11.5).

Barnes-Hut (GraphWaGu-style Hilbert sort + level-wise 4-ary merge, `theta`)
is kept as a documented experiment behind `repulsion: "tree"` ONLY if the
grid tier fails the phase 5 gate on clumpy fixtures; it shares `radixSort`
and would add ~110 B/node (note 03 section 2.4).

### 7.6 Parameter parity table

| Option | CPU (`forceatlas2.ts` lines 26-42) | GPU binding | Notes |
| --- | --- | --- | --- |
| `maxIter` 100 | iteration budget across `step()` calls | `state.iter >= maxIter` -> settled | reheat resets `state.iter` |
| `jitterTolerance` 1.0 | uniform scalar read by K3 | | |
| `scalingRatio` 2.0 | uniform `kr` | K1 / grid | |
| `gravity` 1.0 | uniform `kg`; 0 accepted (the element schema's `positive()` is loosened at E1) | K2 | |
| `strongGravity` false | uniform flag u32 | K2 | |
| `distributedAction` false | uniform flag | K2 | |
| `linlog` false | pipeline variant (branch-free) | K2 | |
| `nodeMass` null | `mass` buffer | `outDegree()+1` default | design line 4019 |
| `weight` null | `HAS_WEIGHTS` variant + `weights` binding; `true` = `snapshot.weights`; a named column = the element's cached `expandEdges` array passed as `weightsOverride: F32(arcCount)` | K2 | LIVE (P-15) |
| `seed` null | CPU LCG (`layout/src/utils/random.ts` lines 21-36: m = 2^35 - 31, a = 185852, c = 1; seed 0 == unseeded) in node-index order for NaN rows | `load()` | same start as the CPU path; verified by a differential test against `RandomNumberGenerator` at W1 |
| `dim` 2 | uniform; K4 skips z | | element recreates the engine on a mode switch (`LayoutManager.updateLayoutDimension`, note 01 section 4.6) |
| `pos` | replaced by the element's position array (`fromPositionColumn` inverse at `load`) | | missing-axis / missing-node fill rules of note 01 section 2.1.2 preserved for NaN rows |
| `scale` / `center` | applied by K5 only | | never rescaled per step (note 01 section 8.4) |
| `nodeSize` / `adjustSizes` | deferred | typed `never` | note 01 section 7.2 |
| `dissuadeHubs` | ignored | ignored | |

### 7.7 Units, initial placement, `load()` and `reload`

`load(snapshot, positions)`:

1. Validate: `isGraphSnapshot`, `!snapshot.directed` (throw `E_BAD_ARGUMENT`
   otherwise: layouts always receive the undirected snapshot, design 14.3),
   `positions.length >= 3 * nodeCount`, `<ArrayBuffer>`-backed.
2. Acquire the residency for `rowPtr / colIdx / weights` and, when any row
   has degree >= 32, `degreeOrder().perm` + tier bounds from
   `segmentOffsets` (CPU side, design 10.1).
3. Build the layout-unit seed on the CPU: for each node, if `positions[3i]`
   is finite: `q = (p - center) / scale`; else: LCG draws `rand() * 2 - 1`
   per axis in index order (2D: z = 0). Upload `pos` in one `writeBuffer`.
4. `mass` from `rowPtr` differences + 1 (or the provided vector / column).
5. Allocate `force`, `oldForce`, `partials`, `centroidPartials`, `state`,
   `out`; zero them; compute the initial centroid on the CPU for iteration 0.
6. `iterationsDone = 0`, `settled = false`.

`reload(snapshot, report, positions)` (design 14.4 M3/M4/M5) is `load()`
with the same rules: finite rows keep their coordinates, NaN rows are
seeded; the element may pre-place new rows at their neighbours' centroid
before calling (product decision left open by the design). The previous
residency is released by the element's `snapshot-replaced` listener, not by
the simulation.

Why layout units on the device (P-7): FA2's forces are not scale-invariant
(1/d repulsion against linear attraction), so simulating in scene units with
the same constants would give a different equilibrium; keeping layout units
keeps every CPU default valid and `fromPositionColumn` already provides the
inverse mapping (note 01 section 8.4).

### 7.8 Settlement and reheat (P-8)

`settled` becomes true when `state.iter >= maxIter`, OR when the mean
per-node displacement over the last `settleWindow` (10) iterations, in
SCENE units (`movement * scale / n`), is below `settleThreshold` (0.05, the
element's ngraph heuristic, `NGraphLayoutEngine.ts` lines 184-190). K3
evaluates the window on the device from a small ring of movement sums in
`state` and sets `settledFlag`; the CPU reads it with the per-step staging
copy. `load()`, `setPosition()` and `setFixed()` with fewer bits set than
before (an unpin) reset `state.iter` and the window (reheat), matching d3's
reheat on `setNodePosition` / `unpin` and ngraph's counter reset (note 01
section 8.6). `setFixed()` that only ADDS pins does not reheat (d3's comment:
reheating on pin makes the layout never settle, note 01 section 3.2).
`maxIter` is the hard bound regardless of the threshold, so screenshots
always proceed.

### 7.9 The `LayoutSimulation` contract and graphty-element's frame loop (P-4)

Facts: `UpdateManager.updateLayout()` calls `layoutManager.step()`
`stepMultiplier` times per frame synchronously
(`graphty-element/src/managers/UpdateManager.ts` lines 203-214);
`LayoutManager.step()` calls `engine.step()` only while `running &&
!isSettled` (`LayoutManager.ts` lines 241-245); nothing in the frame
awaits (note 01 section 4.1). A Promise-returning `step()` therefore needs a
bridge in the element:

```ts
// graphty-element/src/layout/GpuLayoutEngine.ts (E1; sketch)
class GpuLayoutEngine extends LayoutEngine {
    private inflight: Promise<void> | null = null;
    private pendingSteps = 0;
    step(): void {                                   // called synchronously by LayoutManager
        this.pendingSteps += 1;
        if (this.inflight !== null) { return; }      // at most one submission in flight; extra calls coalesce
        const k = Math.min(this.pendingSteps, this.maxBatch); this.pendingSteps = 0;
        this.inflight = this.sim.step(k).then(() => { this.positionColumn.markDirty(); }).finally(() => { this.inflight = null; });
    }
    get isSettled(): boolean { return this.sim.settled; }            // reflects the last COMPLETED readback
    setNodePosition(n, p): void { this.sim.setPosition(n.index, p.x, p.y, p.z ?? 0); }   // 12-byte writeBuffer, applied before the next submit
    pin(n): void { maskSet(this.mask, n.index, true); this.sim.setFixed(this.mask); }
    unpin(n): void { maskSet(this.mask, n.index, false); this.sim.setFixed(this.mask); }
}
```

- Each frame costs one submission (k iterations) and one `mapAsync`; the
  readback is 12n bytes into the element's array (`positions.set(...)`),
  then `column.markDirty()` once (design 14.4 M12). The renderer draws
  positions at most one submission behind; a dragged node is exempt because
  `NodeBehavior.onDragUpdate` writes `mesh.position` directly and
  `Node.update()` skips the copy while dragging (note 01 section 4.5).
- Batch size: `maxBatch = stepMultiplier` by default; the simulation's own
  `iterationsPerStep` applies when the caller passes no count.
- Node (batch API): `while (!sim.settled) await sim.step(32);` -- the same
  function, no frame loop.
- `setPosition` while a submission is in flight is queued as a
  `writeBuffer` that precedes the NEXT submit (queue order), so the GPU
  buffer stays authoritative (design 14.3).
- `dispose()` is called by `LayoutManager` when the engine is replaced
  (`hasDispose`, note 01 section 4.2).

Readback cadence: every `step()` (per frame) up to ~1M nodes on a discrete
GPU in Node (12 MB in ~4 ms, EXTRAPOLATED from the 0.07 ms / 234 KiB
measurement, note 05 section 7.2); in Chromium 1 MiB costs ~2.65 ms
MEASURED, so per-frame readback is comfortable to ~200k nodes and the bridge
exposes `readbackEvery` (default 1) for larger graphs (the renderer cannot
draw 10^6 per-node meshes today anyway, note 01 section 6).

### 7.10 Fixed nodes, drag, 2D / 3D, determinism

- Fixed: K4 tests `(fixed[i >> 5] >> (i & 31)) & 1`; fixed nodes still exert
  forces (repulsion / attraction on others) and accumulate swing so the
  global speed sees them, exactly like pinned bodies in ngraph.
- Drag: `setPosition(i, x, y, z)` converts scene -> layout on the CPU and
  writes 12 bytes at offset `12 * i` of `pos`; the dragged node is normally
  also pinned during the drag (`pinOnDrag`, `GraphBehavior.ts` line 8), so
  K4 does not move it; after drag end the element sets `running = true` and
  the engine's reheat un-settles the simulation.
- 2D: `dim` uniform; K4 does not integrate z; the seed writes z = 0; the
  grid uses 2D cells. 3D: same kernels with 3D loops; the exact tile is
  identical.
- Determinism: same device + same seed + same options -> bitwise identical
  positions for the exact tier and for the grid tier with per-bucket index
  sort (section 6); across vendors only distributional parity is promised.
  The Monte-Carlo near-field window is seeded from `(seed, iter)` so a
  re-run reproduces it.

### 7.11 Fruchterman-Reingold / spring, and others

`createFruchtermanReingold(ctx, { dim, iterations = 50, k = 1 / sqrt(n),
temperature = 0.1, scale, center, seed })`: two kernels per iteration --
`fr_repulse` (exact tile or the grid tier with the `k^2 / d` law: the same
1/d-family kernel parameterised by `strength = k^2`, `mass = 1`) and
`fr_attract_integrate` (CSR gather `d^2 / k`, then move `min(|disp|, t)`
along `disp`, skip fixed) -- plus the write-back. Temperature is a
host-written uniform (`t -= t / (iterations + 1)` per iteration, the CPU's
linear cooling, `fruchterman-reingold.ts` lines 81-83, 154), so no
reduction is needed except the settle movement sum. The `|| 0.1`
coincident-node guard is replicated (note 01 section 2.2). Parity oracle:
the CPU FR one-iteration displacement.

`ngraph-like` preset (note 02 section 7.1 L3): `createFruchtermanReingold`
gains `preset: "ngraph"` mapping `springLength`, `springCoefficient`,
`gravity`, `dragCoefficient`, `timeStep` onto a spring-electrical variant
(Hooke springs + 1/d^2 repulsion + velocity Verlet with drag, ngraph's
`generateIntegrator.js` semantics per note 01 section 3.1) so the element
can route the default `ngraph` type to the GPU above an app-chosen node
count. It is a variant of the same two kernels plus a velocity buffer
(12n); scheduled in phase 6, product wiring left to the app.

ARF (2D only, all-pairs + CSR correction, note 01 section 2.3) and
Kamada-Kawai (needs APSP and a line-search readback per evaluation, not a
`LayoutSimulation`) are later slices and not planned in detail here.

### 7.12 Scaling table (expected ms per iteration)

Basis: exact kernel MEASURED 1.11 ms at 20k on the 4070 (4e8 pairs, i.e.
~3.6e11 pair evaluations/s for the probe's simpler law); the FA2 tile adds a
mass multiply and the softening max, so the plan budgets 2e11 pairs/s;
attraction MEASURED 0.1-0.7 ms per iteration for a 100k-node / 1M-arc
gather (note 06 section 3.5); grid tier REPORTED: cosmos 6.6 ms at 100k and
13.8 ms at 200k (WebGL, GPU unnamed), GraphWaGu BH 5.5 ms at 95k / 6.6M
edges and ~160 ms at 1.13M nodes on an RTX 4070 Laptop (note 03 sections
1.3, 2.3). Integrated column: EXTRAPOLATED 5-10x slower (note 03 section
8.2). Readback: Chromium 1 MiB 2.65 ms MEASURED; Dawn-node ~0.3 ms/MB
EXTRAPOLATED.

| n / E (undirected) | Exact repulsion 4070 | Grid repulsion 4070 | Attraction + speed + integrate 4070 | Per-frame readback (Chromium / Dawn) | Integrated GPU (grid) | Interactive verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1k / 5k | < 0.1 ms (dispatch bound) | n/a (exact) | < 0.1 ms | 0.1 ms / < 0.1 ms | ~0.5 ms | 60 fps with many iterations per frame |
| 10k / 100k | ~0.5 ms | n/a (exact) | ~0.2 ms | 0.3 ms / 0.05 ms | 3-5 ms | 60 fps, several iterations per frame |
| 16k / 160k | ~1.3 ms | ~2 ms | ~0.2 ms | 0.5 ms / 0.1 ms | 5-10 ms | crossover region (measure) |
| 100k / 1M | ~50 ms | 3-8 ms | 0.5-1 ms | 3 ms / 0.4 ms | 20-60 ms | 60 fps at 1 iteration/frame on discrete; ~15 fps integrated |
| 1M / 10M | ~5 s | 30-80 ms | 5-15 ms | 30 ms / 4 ms | 0.3-0.8 s | batch / Node; ~10 fps stepping without per-frame readback |
| 10M / 100M | not viable | 0.1-0.5 s | 50-150 ms | 300 ms / 40 ms | not targeted | Node batch only; windowed uploads; raised limits |

Gate numbers (section 10) turn the 10k, 16k, 100k and 1M rows into MEASURED
values in phases 3 and 5.

---------------------------------------------------------------------------

## 8. Algorithms

Every algorithm is `(ctx, snapshot, options?, dest?) -> Promise<result>`,
records k rounds per `queue.submit()`, reads back only at the end (or every
k rounds for convergence), and returns index-aligned typed arrays (design
10.7). Views named are graph-format views (note 07 section 1.6).

### 8.1 SpMV / iterative family (pull, no atomics)

| Algorithm | Strategy | Primitives | Views | References | WebGPU adjustments |
| --- | --- | --- | --- | --- | --- |
| PageRank, personalized PageRank | power iteration: `rankOut[v] = (1 - alpha) / n * (or personalization[v]) + alpha * (sum_{u -> v} rankIn[u] / outWeight[u] + danglingSum / n)`; L1 delta reduce; converge when `delta < tol` or `maxIter` | `spmvPull`, `segmentedReduce` (weighted out-degree normaliser on device, design 10.1), `reduce` (dangling sum, L1 delta) | `reverse()` (identity arrays when undirected), forward `rowPtr` / `weights` for the normaliser, `degreeOrder({ of: "reverse" })` | cuGraph `pagerank_impl.cuh` lines 222-320 (note 04 section 5); NetworkX semantics of the CPU port (`algorithms/src/algorithms/centrality/pagerank.ts` line 83) | ping-pong rank buffers; dangling and delta partials reduced by a tiny kernel into a 16-byte block copied into the next iteration's uniform (no host readback inside the loop); host checks convergence every 8 iterations; f32 accumulation with tree-shaped partials; 8 storage bindings exactly, personalization in a second group when absent |
| HITS | alternate pull for authorities (over forward rows) and hubs (over reverse rows), normalise by sum | same | forward + `reverse()` | cuGraph `hits_impl.cuh` (note 04 section 5) | two `spmvPull` variants per iteration |
| Eigenvector | pull + L2 normalise; converge on `delta < n * eps` | same | `reverse()` | cuGraph `eigenvector_centrality_impl.cuh` | as PageRank |
| Katz | `alpha * pull + beta` | same | `reverse()` | cuGraph `katz_centrality_impl.cuh` | as PageRank |
| Label propagation | per-node mode of neighbour labels (workgroup-local sort or hash for degree <= 256, global hash region for hubs), synchronous update with the up/down swap rule; `changed` count reduce | `segmentedReduce` (group-by-key form), `reduce` | `rowPtr`, `colIdx`, `weights` | cuGraph Louvain move-phase machinery (note 04 section 10) | ties are nondeterministic on the CPU too; parity = planted-partition recovery |

### 8.2 Traversal family (frontier)

| Algorithm | Strategy | Primitives | Views | References | WebGPU adjustments |
| --- | --- | --- | --- | --- | --- |
| BFS (depth, parent, level order) | level-synchronous top-down: advance the vertex frontier, claim with `atomicCompareExchangeWeak(&depth[v], INVALID, level)`, compact with ownership dedupe; direction-optimizing switch to bottom-up over `reverse()` when `m_f > m_u / alpha` and growing, back when `n_f < n / beta` and shrinking (alpha = m/n, beta = 24 as cuGraph; Beamer 14 / 24 as an option) | `Frontier` + `advance`, `compact`, `bitset`, indirect dispatch | `rowPtr`, `colIdx`, `reverse()` (free when undirected), `degreeOrder()` | Merrill 2011 (scan + warp + CTA expansion, duplicate culling, expand/contract couplings); Beamer SC12; cuGraph `bfs_impl.cuh` lines 291-297, 637-638, 843-846 (note 04 section 3) | k levels per submit with indirect dispatch; fused expand-contract kernel for tiny frontiers; `parent[v]` written by the winning CAS only (level-consistent, not FIFO-identical to the CPU); `order` grouped by level; 2D dispatch above 16,776,960 frontier entries |
| SSSP (non-negative) | Davidson near-far: relax with `atomicMin` on the u32 bit pattern of non-negative f32 (`+Inf = 0x7F800000` = unreached); near pile `dist < (i+1) * delta`, far pile deferred; `delta = 32 * avgWeight / avgDegree`; predecessor by a second pass `atomicMin(&pred[v], u)` where `dist[u] + w == dist[v]` | frontier, `compact`, `histogram` (far-pile buckets) | `rowPtr`, `colIdx`, `weights`, `flags.nonNegativeWeights` (required, else `E_UNSUPPORTED`), `flags.allWeightsOne` -> run BFS | Davidson 2014; cuGraph `sssp_impl.cuh` lines 189-262 (note 04 section 4) | device-side "near empty" flag turns extra queued rounds into zero dispatches; result buffer reinterpreted as `Float32Array` with no conversion; `predArc` ties differ from Dijkstra's (documented) |
| Bellman-Ford (negative weights) | edge-parallel relax over `edgeList()` (both directions when undirected), CAS loop on the f32 bit pattern, `n - 1` rounds max with a changed flag, one more round for negative-cycle detection | `reduce` (changed flag) | `edgeList()` | design 10.1 row for `edgeList()`; note 04 section 1 table | rounds batched 8 per submit |
| Closeness / harmonic / eccentricity | batched multi-source BFS (bitmask frontier, 32 sources per u32 word) or repeated near-far SSSP for weighted; per-source rows reduced on the device (sum, sum of 1/d, max); never materialises n x n | BFS machinery, `reduce` | as BFS | cuGraph multi-source BFS (note 04 sections 6-7) | batch size from `maxBufferSize` |
| APSP / Floyd-Warshall | unweighted: n batched BFS writing rows of an n x n `F32`; weighted: blocked Floyd-Warshall (tiled kernel) | as above | `rowPtr`, `colIdx`, `weights` | classic blocked FW | `n * n * 4 <= maxBufferSize` else `E_LIMIT` (n <= 8,192 at defaults, ~32k with 4 GiB) |

### 8.3 Centrality: betweenness

Brandes with McLaughlin-Bader structure: forward pass = work-efficient BFS
with `Qcurr / Qnext` queues, `sigma` as `array<atomic<u32>>` (`atomicAdd`;
overflow detected by the return value and reported in
`details.sigmaOverflow` -- documented, note 04 section 6), and an `S / ends`
array recording each level's vertices contiguously; backward pass = per
level from the deepest, each `w` PULLS over its successors `v` with `d[v]
== d[w] + 1`: `delta[w] += sigma[w] / sigma[v] * (1 + delta[v])` (no float
atomics, "checking successors rather than predecessors"); `bc[w] +=
delta[w]` is a plain add. Sources are processed in batches as a TAGGED
multi-source BFS (cuGraph: up to 65,535 sources, 2D `n x k` sigma / distance
arrays capped at 25 % of device memory, `betweenness_centrality_impl.cuh`
lines 660-700, 1380-1400); the batch size is planned from `maxBufferSize`.
The hybrid choice (work-efficient vs edge-parallel) is made online: run the
first batch work-efficiently, read back the median depth, switch to the
all-edges kernel over `coo().src` when `median < gamma * log2(n)` (Algorithm
3 of the CACM paper). `sources: number` selects k sampled sources
(approximate BC, the only interactive option at 1M nodes: exact BC is O(nm));
`edges: true` accumulates per-arc values folded with `foldArcs(...,
"first")` and halved on undirected graphs. Result: `F32(n)` raw or
normalised exactly as `indexed.betweennessCentrality` will. References:
McLaughlin-Bader CACM 2018 (author mirror PDF, note 04 section 6); the
Buffalo 2023-06 thesis is dense-matrix BC and is excluded (note 04 section
0 item 8).

### 8.4 Components and community

| Algorithm | Strategy | Primitives | Views | References | WebGPU adjustments |
| --- | --- | --- | --- | --- | --- |
| WCC | Afforest: `comp[v] = v`; 2 sampled link rounds over the r-th neighbour of every vertex; compress (pointer jumping with `atomicLoad`); sample `comp` (1024-entry histogram) to find the giant component; link the remaining vertices' remaining edges skipping the giant component; compress until a device `changed` flag stays 0 | edge map with `atomicCompareExchangeWeak`, `reduce`, `histogram`, `compact` | `edgeList()` (each edge once; correct for directed WCC without `reverse()`), `rowPtr` / `colIdx` for the r-th neighbour rounds | GAP `gapbs/cc.cc` lines 40-150 (note 04 section 8) | all u32; mixed atomic / non-atomic access to one element is illegal in WGSL, so `comp` is `array<atomic<u32>>` throughout; dense relabel in first-seen order by `renumberPartition` on the CPU (`packages/graph-format/src/snapshot/derived.ts` line 1155) so `groups()` matches the CPU exactly |
| SCC | forward-backward reachability with trimming | BFS x2 | `reverse()` | -- | label order cannot match Tarjan's; set-equality parity only; low priority (note 02 A15) |
| Louvain | per level: vertex weights by `segmentedReduce`; synchronous best-move pass with `delta_Q` per neighbouring community (per-row group-by-key: workgroup sort for degree <= 256, global hash region sized `2 * degree` for hubs), the `up_down` alternating rule to prevent swaps, cluster weights recomputed by reduce-by-key (not atomics); contraction = `radixSort` by `(cluster(src), cluster(dst))` + `segmentedReduce` + `cooToCsr` on the device; modularity by `reduce`; host reads Q and the move count per pass | `segmentedReduce`, `radixSort`, `cooToCsr`, `reduce` | symmetric `rowPtr / colIdx / weights`, `edgeList()`, `weightedDegree` computed on device | cuGraph `louvain_impl.cuh` lines 172-215, `detail/common_methods.cuh` lines 70-152, 402-446; nu-Louvain and Gilbert-Madduri 2026 (note 04 section 10) | fixed-point i32 where a float accumulation is unavoidable; expectation 2-10x over the CPU (later levels lose parallelism); the package runs small levels on the GPU too (no CPU handoff inside the package) |
| Leiden | Louvain + refinement with a maximal-independent-set kernel | as Louvain | as Louvain | cuGraph `leiden_impl.cuh`, `refine_impl.cuh` | after Louvain is stable |

### 8.5 Structure

| Algorithm | Strategy | Primitives | Views | References | WebGPU adjustments |
| --- | --- | --- | --- | --- | --- |
| k-core | rounds: frontier of vertices with `count < k` -> `atomicSub` neighbour counts -> compaction; k increases when the frontier empties; O(max core) rounds | frontier, `compact` | `rowPtr`, `colIdx`, `outDegree()` copy | cuGraph `core_number_impl.cuh` lines 97-230 (note 04 section 9) | indirect dispatch; k rounds per submit |
| Triangle counting / k-truss | orient edges low-to-high degree (tie by id), per oriented arc intersect sorted rows (merge, or binary search when degrees differ > 32x), `atomicAdd(u32)` per-vertex counts, workgroup partial totals | intersection advance, `compact`, masks | sorted `rowPtr / colIdx` (`flags.sortedRows` invariant), `outDegree()`, `edgeToArc` for per-edge support | cuGraph `triangle_count_impl.cuh` lines 344-470, `k_truss_impl.cuh` lines 183-300; Gunrock `tc.hxx` | workgroup-per-arc tier for hub pairs |
| MST (Boruvka) | per-component minimum edge via `atomicMin` on packed `(weightBits << 32 \| edge)` split across two u32 words (two-pass scheme), union via WCC compress | WCC primitives | `edgeList()` | design Port 5 (edge indices) | edge set identical when weights are distinct; ties differ from Kruskal |
| Common neighbours / Adamic-Adar | the triangle intersection kernel with a score accumulator | as triangles | sorted rows | design Port 6 | later |

### 8.6 Priority order with rationale (note 02 section 7.2 scores)

1. PageRank + personalized (25): highest value, one pull kernel, no atomics;
   first algorithm parity tests.
2. HITS / eigenvector / Katz (15): same kernel, different combine snippets.
3. WCC (8): edge-parallel u32 atomics over `edgeList()`, exercises
   `renumberPartition` and the ownership of dense relabelling.
4. BFS (5.3) -> closeness (5.3) -> SSSP (4) -> betweenness (6.3): the
   frontier family in dependency order, ending in the most expensive analytic
   graphty users run; sampling makes it interactive.
5. Bellman-Ford (6), k-core (4.5), label propagation (4), triangles (4),
   MST (3), APSP (5): as demand appears; each is small once the primitives
   exist.
6. Louvain / Leiden (3): highest user value and highest risk; last, after
   the sort, segmented reduce and COO->CSR primitives are proven by the
   layout and SpMV slices.

The force-directed slice needs only `reduce`, `segmentedReduce`, and (for
the grid tier) `histogram`, `scan` and the grid kernels; it does not block on
the frontier machinery (note 04 section 15 item 1), which is why it can be
the first product slice.

---------------------------------------------------------------------------

## 9. Integration with @graphty/algorithms, @graphty/layout and @graphty/graphty-element

### 9.1 Mechanism and dependency direction (P-3; note 02 section 4)

```
graphty app  ----imports---->  @graphty/webgpu-graph-algorithms  ----runtime dep---->  @graphty/graph-format
     |                                   |  (devDependencies only: type conformance)
     | sets element.accelerator          v
     v                          @graphty/algorithms, @graphty/layout  (OWN the accelerator interfaces; no WebGPU imports)
@graphty/graphty-element  ----runtime dep----^
```

- Explicit injection is the contract (design line 70: "an OPTIONAL
  accelerator injected by the caller"; 14.4 line 4239 `runAlgorithm(
  snapshot, { accelerator: gpu })`).
- The CPU packages own STRUCTURAL interfaces next to their `indexed.*`
  types and add one async dispatcher each. The GPU package satisfies them
  by shape and is checked at compile time in its own CI with `expectTypeOf`
  (algorithms / layout as devDependencies at W1), so the runtime dependency
  graph stays acyclic and tree-shaking is untouched.
- The registry mechanism (`registerAccelerator` singleton) is rejected:
  inverted dependency, side-effect module defeats tree-shaking, global
  state breaks with duplicate package copies, and it puts the "GPU threw,
  what now?" decision into the CPU package (note 02 section 4.2).
- Optional-peer + dynamic import inside the ELEMENT ("auto" mode) is a
  later convenience only, behind the web-llm isolation pattern
  (`graphty-element/src/ai/providers/index.ts` lines 9-13: Safari fails on
  dynamic imports of non-existent modules even before the import is called;
  `vite.config.ts` line 39 externalises the specifier); never in algorithms
  or layout, and never for Node (the element must not depend on the Dawn
  addon) (note 02 section 4.3).

### 9.2 Exact API changes per package and when

| Package | Change | Lands at (design 14.6) |
| --- | --- | --- |
| `@graphty/algorithms` | `src/indexed/accelerator.ts`: `export interface AlgorithmAccelerator { pageRank?(s, o?): Promise<PageRankResult>; personalizedPageRank?; hits?; eigenvectorCentrality?; katzCentrality?; connectedComponents?; breadthFirstSearch?(s, source: number, o?); sssp?; bellmanFord?; closenessCentrality?; betweennessCentrality?; kCore?; triangleCount?; labelPropagation?; minimumSpanningTree?; allPairsShortestPath?; louvain?; release?(s): void }` with the SAME option types as `indexed.*` and result score types widened to `NumericVector` (`F32 \| F64`); `export function accelerated(acc?: AlgorithmAccelerator \| null): AcceleratedAlgorithms` whose methods are `pageRank(s, o): Promise<PageRankResult>` delegating to `acc.pageRank` when defined, else `Promise.resolve(indexed.pageRank(s, o))`. Sync `indexed.*` and legacy facades unchanged. Tests: a fake accelerator object exercises the dispatcher without a GPU | A2 |
| `@graphty/layout` | `export interface LayoutAccelerator { forceAtlas2?(o: ForceAtlas2Options): LayoutSimulation; fruchtermanReingold?(o: FruchtermanReingoldOptions): LayoutSimulation }`; `forceAtlas2Simulation(o): LayoutSimulation` (the CPU FA2 becomes steppable so the element has ONE code path and the GPU is an implementation swap); the P-5 formula settlement in the CPU rewrite; `seedPositions(snapshot, positions, seed, dim, scale, center)` (LCG in index order for NaN rows) exported so the element and tests can reproduce starts; `LayoutSimulation` exported (design 14.3) | L1 |
| `@graphty/graphty-element` | `Graph` / element property `accelerator: (AlgorithmAccelerator & LayoutAccelerator & { release?(s): void }) \| null` (typed against the CPU packages' interfaces, no WebGPU types); adapters call `accelerated(this.graph.accelerator).x(s, o)` then the shared result-writing loop; `snapshot-replaced` listener calls `accelerator?.release?.(previous)`; `ForceAtlas2LayoutEngine` / `SpringLayoutEngine` create `accelerator?.forceAtlas2?.(opts) ?? forceAtlas2Simulation(opts)` and drive it through the `GpuLayoutEngine` bridge when `step()` returns a Promise (section 7.9); the FA2 zod schema loosens `gravity` to `nonnegative()`; `weightPath` becomes live; a Storybook story "Layout/ForceAtlas2 (GPU)" with a fake accelerator for Chromatic and the real one behind a flag | E1 |
| `@graphty/graphty` (app) | detection: `const p = await GpuContext.probe({ gpu: navigator.gpu }); if (p.ok && (!p.software \|\| allowSoftware)) { const m = await import("@graphty/webgpu-graph-algorithms"); const ctx = await m.GpuContext.create({ gpu: navigator.gpu }); element.accelerator = ctx.accelerator(); }` plus a "GPU acceleration: on / off" indicator and a node-count threshold for routing the `ngraph` default to the GPU preset | E1 or later |
| `@graphty/webgpu-graph-algorithms` | `src/types/accelerator.ts` declares the two interfaces STRUCTURALLY (copies) until W1; at W1 `test/types/accelerator-conformance.test-d.ts` asserts `expectTypeOf(ctx.accelerator()).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>()` against the real packages; differential tests switch their oracle from `test/helpers/oracle.ts` to `indexed.*` | W1 |

### 9.3 Result-shape parity (note 02 section 5)

| GPU method | Returns | Parity rule against `indexed.*` |
| --- | --- | --- |
| `pageRank`, personalized, HITS, eigenvector, Katz | `F32(n)` scores (+ `iterations`, `converged`) | relative error <= 1e-5 per node (design 16.2); identical top-k order; `iterations` within +-1 |
| `degree` (skeleton) | `U32(n)` | exact |
| `breadthFirstSearch` | `{ depth: U32, parent: U32, order: U32, visitedCount }` | `depth` exact; `parent[v]` any u with `depth[u] === depth[v] - 1` and an arc u -> v; `order` grouped by level |
| `connectedComponents` | `{ labels: U32, count }` | identical after first-seen renumbering (`renumberPartition`) |
| `sssp`, `bellmanFord` | `{ dist: F32 (Infinity = unreached), predArc: U32 }` | `dist` within 1e-5 relative; `predArc` any arc attaining `dist`; negative-cycle flag exact |
| `betweennessCentrality` | `{ nodes: F32, edges: F32 \| null }` | relative error <= 1e-4 (f32 accumulation over many sources); top-k order; edges via `foldArcs` |
| `closenessCentrality` | `F32(n)` | 1e-5 |
| `allPairsShortestPath` | `F32(n * n)` | exact unweighted, 1e-5 weighted |
| `labelPropagation`, `louvain` | `{ labels: U32, ... }` | planted-partition recovery / modularity within a band, not identical partitions |
| `kCore` | `U32(n)` | exact |
| `minimumSpanningTree` | `{ edges: U32, totalWeight }` | `totalWeight` within 1e-5; identical edge set on distinct weights |
| layouts | a `LayoutSimulation` writing the owner's stride-3 `F32` | no coordinate parity; trace and property tests (section 11.4) |

### 9.4 What changes where in the landing order

- Now (this repo, before A2 / L1 exist): the GPU package develops against
  `@graphty/graph-format` alone with its own index-based CPU references in
  `test/helpers/oracle.ts` (written from design Ports 1-6 and the FA2 table
  of section 7.3).
- A2 / L1: the interfaces and dispatchers land in the CPU packages (small,
  fake-tested).
- E1: the element property, the bridge and the story.
- W1: move-in (`packages/README.md` checklist), conformance type tests,
  oracle switch, the monorepo CI shards and the GPU lane (section 12.5).
- 2.0: nothing GPU-specific.

---------------------------------------------------------------------------

## 10. Performance targets and memory model

### 10.1 Assumptions

Graph tiers use average degree 10 (E = 10n), undirected snapshots (A = 2E),
unweighted unless stated. "Device" bytes exclude the CPU snapshot. Bases:
design 15.1 byte model; probe measurements in notes 05 and 06; REPORTED
figures from cosmos and GraphWaGu; EXTRAPOLATED where marked. Upload
bandwidth is EXTRAPOLATED at 5-10 GB/s from the probe's 4 MiB submission in
0.40 ms (note 05 section 2.4) and must be measured in phase 1.

### 10.2 Device memory

| Tier | n / E | Snapshot hot prefix (unweighted / weighted) | FA2 exact state | FA2 grid state (2D / 3D) | PageRank state | BFS state | Betweenness batch (k sources) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | 10k / 100k | 0.84 MB / 1.6 MB | 0.5 MB | n/a (exact) | 0.12 MB | ~1 MB | 12 MB (k = 100) |
| mobile | 100k / 1M | 8.4 MB / 16.4 MB (design 10.3: 16,400,128 B) | 5.2 MB | 6.4 + 5.6 MB / 6.4 + 38 MB | 1.2 MB | ~10 MB | 120 MB (k = 100) |
| desktop | 1M / 10M | 84 MB / 164 MB | 52 MB | 64 + 5.6 MB / 64 + 38 MB | 12 MB | ~96 MB | 120 MB (k = 10) |
| batch | 10M / 100M | 840 MB / 1.64 GB (windowed; `arena: false`) | 520 MB | 640 + 5.6 MB | 120 MB | ~1 GB | not offered (sampled k <= 1) |

Limit consequences: at defaults (256 MiB buffer, 128 MiB binding) the
desktop tier's weighted arena (164 MB) fails the buffer limit -> per-array
path (`colIdx` 80 MB fits a binding); at the batch tier `colIdx` (800 MB)
needs raised limits (2 GiB binding under Dawn NVIDIA, 4 GiB in Chromium;
lavapipe cannot raise beyond 128 MiB) or windows.

### 10.3 Time targets (acceptance criteria on the 4070, Dawn-in-Node unless stated)

| Id | Target | Measured basis / expectation | Gate (phase) |
| --- | --- | --- | --- |
| T-1 | Upload hot prefix 100k / 1M weighted (16.4 MB) <= 10 ms; 1M / 10M (164 MB) <= 100 ms | EXTRAPOLATED 5-10 GB/s | P1 |
| T-2 | `degree` kernel + 400 KB readback at 100k <= 2 ms wall | probe: 1M-element kernel submit 0.40 ms + `onSubmittedWorkDone` 0.17 ms MEASURED | P1 |
| T-3 | Round trip (submit + 4-byte mapAsync) <= 0.1 ms Dawn, <= 0.3 ms Chromium | MEASURED 0.04 / 0.10 ms | P1 |
| T-4 | FA2 exact, 10k nodes: <= 2 ms per iteration; 16k: <= 4 ms; the crossover measured at 4k / 8k / 16k / 32k | MEASURED 1.11 ms at 20k for the simpler probe kernel | P3 |
| T-5 | FA2 per-frame cost in Chromium (step(1) + 12n readback) at 10k nodes <= 6 ms; at 100k nodes (grid) <= 12 ms | Chromium 1 MiB readback 2.65 ms MEASURED | P3 (10k), P5 (100k) |
| T-6 | FA2 grid, 100k / 1M edges: <= 10 ms per iteration; 1M / 10M: <= 100 ms per iteration; 3D at 100k <= 20 ms | REPORTED cosmos 6.6 ms at 100k; GraphWaGu 160 ms at 1.13M (laptop, BH) | P5 |
| T-7 | Attraction gather 1M / 10M (20M arcs) <= 15 ms | MEASURED 0.1-0.7 ms at 1M arcs | P5 |
| T-8 | PageRank 100k / 1M, 100 iterations, convergence check every 8: <= 100 ms wall; 1M / 10M <= 1 s | MEASURED gather 0.1-0.7 ms per iteration at 1M arcs | P7 |
| T-9 | WCC 1M / 10M <= 100 ms | EXTRAPOLATED (~5-10 rounds of O(m) u32 CAS) | P8 |
| T-10 | BFS 1M / 10M RMAT (diameter ~10) <= 100 ms; 1000 x 1000 grid (2,000 levels) <= 1.5 s, i.e. <= 0.75 ms per level with <= 1 readback per 16 levels | Merrill 3.3 GTEPS REPORTED (2011); Chromium round trip 0.10 ms MEASURED | P9 |
| T-11 | Betweenness, 100k / 1M, 256 sampled sources <= 5 s; karate exact <= 20 ms | per source = one BFS + one backward sweep | P10 |
| T-12 | Default CI lane (lavapipe + SwiftShader) <= 15 min; GPU lane <= 20 min | wgpu budgets 5-15 min for its lavapipe job (note 06 section 3.5) | P1 onward |
| T-13 | Benchmark regression: any tracked number > 3x its checked-in baseline for the runner class fails the GPU lane's bench step (design 15.5 policy) | | P3 onward |

### 10.4 Per-iteration / per-run time table (expected; 4070 unless stated)

| n / E | Upload | FA2 exact | FA2 grid | Readback 12n (Dawn / Chromium) | PageRank x 100 | BFS (low diameter) | WCC | BC 256 sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | ~1 ms | 0.5 ms | n/a | 0.05 / 0.3 ms | ~10 ms (dispatch bound) | ~2 ms | ~2 ms | ~0.3 s |
| 100k / 1M | 2-4 ms | 50 ms | 3-8 ms | 0.4 / 3 ms | 20-70 ms | 5-15 ms | 5-15 ms | 1-5 s |
| 1M / 10M | 20-40 ms | 5 s | 30-80 ms | 4 / 30 ms | 0.2-0.7 s | 30-100 ms | 30-100 ms | 20-60 s |
| 10M / 100M | 0.2-0.5 s + windows | -- | 0.1-0.5 s | 40 / 300 ms | 2-7 s | 0.3-1 s | 0.3-1 s | -- |
| integrated GPU (100k / 1M) | 5-10 ms | 0.3-0.5 s | 20-60 ms | 0.4 / 3 ms | 0.1-0.5 s | 30-100 ms | 30-100 ms | 5-30 s |
| lavapipe, 4 threads (100k / 1M; correctness only) | 10-20 ms | ~20 s | ~1-2 s | ~5 ms | 0.3-0.5 s (MEASURED 3.3-5.5 ms per gather iteration) | 0.3-1 s | 0.3-1 s | not run |

Every EXTRAPOLATED cell is replaced by a MEASURED value from
`benchmarks/results/<host>-node<version>.json` (the graph-format harness
shape with an added `gpu` field, note 07 section 6) at the gate of the phase
that delivers the feature; the table in the package README is regenerated
from that file.

---------------------------------------------------------------------------

## 11. Testing strategy

### 11.1 Principles

1. One set of test files, two vitest projects (`node`, `browser`), plus
   `node-limits` and `bench` projects that only the GPU lane runs (note 06
   section 4.1). The node project on Dawn is the full suite: kernels,
   primitives, algorithms, layouts, lifecycle, planners, property tests.
2. Every GPU test compares against a CPU reference or an invariant; "no
   error" is never a pass (out-of-bounds storage access is clamped
   silently, note 05 section 6 item 6).
3. A wrong result is never a skip. Without any adapter a test skips with a
   printed `E_NO_ADAPTER` reason (the graph-format convention,
   `packages/graph-format/test/audit/gpu-upload.test.ts` lines 11-17); under
   `GRAPHTY_REQUIRE_GPU=1` it fails; under `GRAPHTY_GPU_REQUIRE=nvidia` a
   software adapter fails (the silent-lavapipe failure mode of
   `HEADLESS_GPU_REPORT.md`).
4. Fixtures scale with the adapter: `gpuScale()` returns 1 on hardware and
   1/50 on a software adapter (lavapipe is ~350x slower on O(n^2) kernels,
   note 05 section 2.5); iteration counts and sizes multiply by it; every
   test's fixture is deterministic (seeded LCG / xorshift from the
   graph-format harness, `packages/graph-format/benchmarks/datasets.ts`).
5. The `uncapturederror` listener in `test/setup/gpu.ts` fails the current
   test; every buffer and pipeline is labelled so the message names it.
6. Planners are pure functions of `GpuCaps` and are unit-tested with FAKED
   caps tables (spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like
   from note 05 section 4) with no device, so limit branches are covered on
   the default lane.

### 11.2 Test projects and environments

| Project | Runtime | Adapter locally | Adapter on default lane | Adapter on GPU lane | Content |
| --- | --- | --- | --- | --- | --- |
| `node` | vitest, `environment: "node"`, `pool: "forks"` (native addon; graph-format's 1309 tests run this way) | NVIDIA via Dawn with `libEGL.so.1` on `LD_LIBRARY_PATH` (`HEADLESS_GPU_REPORT.md` appendix D) | lavapipe (`GRAPHTY_GPU_ADAPTER=llvmpipe`) | NVIDIA, `GRAPHTY_GPU_REQUIRE=nvidia` | everything below except browser and limits |
| `browser` | vitest browser mode, Playwright Chromium, `fileParallelism: false` | NVIDIA with the four flags `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface` | SwiftShader with `--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader` | NVIDIA flags | smoke (section 11.8) |
| `node-limits` | node | NVIDIA | skipped (project selection, not runtime detection) | NVIDIA | bindings > 128 MiB, `maxBufferSize` near 2 GiB, real 2D dispatch on 100M items, vendor feature assertions |
| `bench` | node, `singleFork` | NVIDIA | not run | NVIDIA | `vitest bench` + `benchmarks/run.ts` writing JSON |

Setup file sketch (`test/setup/gpu.ts`): dynamic `import("webgpu")` in Node
(or `navigator.gpu` in the browser), `Object.assign(globalThis,
dawn.globals)`, `create([adapter=...])` from `GRAPHTY_GPU_ADAPTER`,
`XDG_RUNTIME_DIR` defaulted to silence Mesa's stderr lines, adapter info and
the four limits printed once per worker, the `GRAPHTY_GPU_REQUIRE` vendor
assertion, `afterAll` destroying the device and nulling the `GPU` reference
(the process cannot exit while it is reachable, note 05 section 2.2).

### 11.3 Test categories (each with its oracle)

| Category | What | Oracle | Where |
| --- | --- | --- | --- |
| Planner unit tests | `planUpload` (arena / perArray / windowed incl. the 64-arc `%` rule and a row longer than a window), `planDispatch` (1D / 2D / stride, the `(16,776,960, 16,777,216]` boundary), `UniformBlock` layouts, `PipelineCache` keys, `BufferPool` size classes | hand-computed expectations; faked caps tables | `test/memory`, `test/kernel` (no device) |
| Kernel / primitive tests | each primitive at sizes 0, 1, 255, 256, 257, 4097, 65,535 x 256 +/- 1 (scaled), random and adversarial inputs (all-equal keys, one hub row of 10^5 arcs, empty rows, `arcCount === 0`) | `test/helpers/oracle.ts` CPU references (f64) | `test/primitives` |
| Upload contract tests | ported from `gpu-upload.test.ts` lines 254-666: 256-byte offsets accepted, whole-arena upload equals CPU views per segment, cold segments gather / write back, undirected doubled storage, packed `u8` / `bool` columns, 64-arc windowed binding equals a copied window, `fromBytes` container at a non-zero `arena.byteOffset`, `arena === null` snapshots (`fromCsr`, `transpose()`) | graph-format views | `test/memory` |
| Differential tests | every algorithm vs the CPU oracle on: empty graph, one node, one self-loop, karate (`KARATE_EDGES`), grids (`gridEdges`), paths (high diameter), stars (hub), complete graphs, random G(n, m) with parallels and self-loops, planted partitions, directed and undirected, weighted (incl. zero weights) and unweighted; sizes up to 100k / 1M on hardware | `oracle.ts` now, `indexed.*` at W1; tolerances of section 9.3 | `test/algorithms` |
| Layout parity | one-iteration force parity, trace parity, distributional parity (section 11.4) | CPU FA2 / FR oracle | `test/layouts` |
| Exact vs approximate | grid tier vs exact tier (section 11.5) | the exact GPU kernel | `test/layouts` |
| Property tests (fast-check, `numRuns` 200 default, 1,000 nightly) | section 11.6 | invariants, no oracle | across |
| Contract / lifecycle | `LayoutSimulation` semantics, `release` / `dispose` / device loss / leak counting (section 11.7) | counters and state | `test/device`, `test/layouts` |
| Type tests | public d.ts under strict-consumer flags; accelerator conformance at W1 | `tsc -p tsconfig.strict-consumer.json`, `expectTypeOf` | `test/types` |
| Barrel / build | export list pinned; `package.json` exports map incl. `./node`; dist has no `webgpu` import in the root entry | `test/index.test.ts`, `test/build-output.test.ts` (graph-io pattern) | |
| Browser smoke | section 11.8 | small fixtures | `test/browser` |
| Benchmarks | section 11.9 | baselines | `benchmarks/` |

### 11.4 Layout parity tests (the FA2 slice)

- Same start: both sides seed NaN rows with the same LCG in index order
  (`seedPositions`), so iteration 0 positions are bit-identical.
- Force parity (one iteration): run K1 + K2 on the GPU, read `force` back,
  compare with the f64 oracle's per-node force on karate, a 10 x 10 grid, a
  star of 200 and random 1,000-node graphs: relative L2 error per node <=
  1e-4 (f32 tile summation order), with and without weights, linlog,
  distributedAction, strongGravity, `gravity = 0`, 2D and 3D.
- Trace parity: over 50 iterations compare the per-iteration
  `{ totalSwing, totalTraction, speed, speedEfficiency }` trace (read from
  `state` after each `step(1)`) with the oracle's trace: relative error
  growing at most geometrically with a documented bound (target 1e-3 at
  iteration 50 on 1,000 nodes; chaotic divergence beyond is expected and
  the test asserts the FIRST 10 iterations tightly (1e-4) and the rest
  loosely (5e-2)).
- Distributional parity (same seed, 100 iterations): stress, edge-length
  distribution quantiles, per-node nearest-neighbour distance histogram
  and inter-component separation compared within 10 % between GPU and CPU.
  Coordinates are never compared.
- Existing behaviour pins from `layout/test/forceatlas2-layout.test.ts`
  (note 01 section 2.1.8) re-expressed on the GPU: empty graph, single
  node, disconnected components separated by > 0.03 after 100 iterations,
  `maxIter` respected, `completeGraph(6)` spread > 0.3, same seed -> same
  layout (bitwise on the same device), different seeds -> different.

### 11.5 Exact-vs-approximate tests (the grid tier)

- Fixtures: uniform random, clumpy (Gaussian mixtures with 10 / 100 / 1,000
  clusters), a scale-free graph with a 10k-degree hub, cosmos's two
  documented failure cases (a 163-node country graph shape and 1,024 points
  in one finest cell), a line (all points colinear), and coincident points;
  sizes 20k, 100k, 262k (finest-grid saturation) and 1M (hardware only).
- Force-field error: `|F_grid - F_exact| / |F_exact|` RMS over nodes <= 5 %
  and 99th percentile <= 25 % on uniform and clumpy fixtures; on the
  hub-cell fixture the Horvitz-Thompson path is checked for unbiasedness
  (mean over 32 seeded iterations within 5 % of exact).
- Layout-level: from the same start, 200 iterations exact vs grid, the
  distributional metrics of section 11.4 agree within 15 %; no node moves
  more than `2 * cellSize` per iteration (the clamp holds).
- Determinism: two runs on the same device are bitwise identical (grid with
  per-bucket sort); the Monte-Carlo window is reproducible from `(seed,
  iter)`.
- 3D: the same tests at 27 / 216 loops with the 128^3 cap; memory of the
  pyramid asserted <= 40 MB.

### 11.6 Invariant / property tests (no oracle needed)

- Fixed nodes never move (any mask, any options, random `setFixed` calls
  between steps).
- `dim === 2` leaves every `z` exactly 0 after `load` and after any number
  of steps.
- `setPosition(i, ...)` is visible in the next readback and the node stays
  put when also fixed.
- `settled` becomes true within `maxIter` steps for every fixture; `load`,
  `setPosition` and an unpin reset it; adding pins does not.
- Momentum: with `gravity = 0` the centroid drift per iteration is below
  1e-4 (repulsion and attraction are pairwise antisymmetric).
- Energy-like monotonicity: FR temperature-clamped displacement never
  exceeds `t`; FA2 per-node displacement never exceeds `speed * |F| /
  (1 + sqrt(speed * swg))`.
- Results are index-aligned typed arrays over `ArrayBuffer` of exact
  length; `INVALID_INDEX` never appears where a valid index is required;
  `dest` is returned when supplied.
- BFS: `depth[parent[v]] === depth[v] - 1` and an arc exists; `order` has
  non-decreasing depth; `visitedCount` equals the number of finite depths.
- SSSP: triangle inequality over every arc (`dist[v] <= dist[u] + w`);
  `predArc` attains `dist`.
- WCC: labels dense `0..count-1`, endpoints of every edge share a label,
  `count` equals the oracle's.
- Betweenness: sum over nodes equals the analytic total for a path and a
  star; symmetry on vertex-transitive graphs (cycle, complete).
- PageRank: scores sum to 1 within 1e-5; personalization with a one-hot
  vector concentrates mass.
- Upload: after `release(snapshot)` every buffer of that snapshot is
  destroyed (leak counter) and a kernel that binds it throws `E_RELEASED`.

### 11.7 Lifecycle, error and device-loss tests

- Leak counting: tests wrap `device.createBuffer` / `buffer.destroy` in a
  counting proxy (`test/helpers/leak-counter.ts`); after `dispose()` the
  live count is 0 and `pool.liveBytes === 0`.
- Validation errors are typed: a deliberately broken variant (wrong binding
  count) rejects with `E_VALIDATION` and `details.messages` non-empty;
  `getCompilationInfo` errors surface with line numbers.
- Out-of-memory: with faked caps, a request above `maxBufferSize` throws
  `E_LIMIT` before touching the device; on the real device a huge
  `createBuffer` inside the OOM scope yields `E_OUT_OF_MEMORY`
  (`node-limits`).
- Device loss: `device.destroy()` mid-`step()` rejects the pending promise
  with `E_DEVICE_LOST`, `ctx.state === "lost"`, subsequent calls throw
  `E_DEVICE_LOST`, and creating a new context and `load()`ing again works
  (both runtimes; deterministic).
- `E_NOT_LOADED`, `E_BAD_ARGUMENT` (directed snapshot to a layout, wrong
  `dest` length, `SharedArrayBuffer`-backed input), `E_UNSUPPORTED`
  (windowed frontier before phase 9; SSSP on negative weights).

### 11.8 What "light browser testing" contains (project `browser`)

- Device acquisition through `GpuContext.create({ gpu: navigator.gpu })`
  and the typed error when `navigator.gpu` is absent (simulated by passing
  `undefined`); `probe()` reports the adapter and `software` correctly
  (SwiftShader on the default lane, NVIDIA on the GPU lane with the vendor
  assertion under `GRAPHTY_GPU_REQUIRE=nvidia`).
- Walking skeleton end-to-end: upload the karate snapshot, `degree`,
  readback equals `outDegree()`, `release()` leaves no uncaptured error.
- FA2: `load` a 500-node graph, `step(10)` five times, positions written
  back, `settled` eventually true, `setFixed` / `setPosition` honoured,
  `dispose()` clean.
- A frame-loop test using the same `GpuLayoutEngine` bridge logic as the
  element (a synchronous 60-tick loop calling `step()` without awaiting;
  asserts at most one submission in flight and that positions advance).
- One PageRank and one BFS on karate (parity with the oracle), one
  subgroup-variant primitive when `device.features.has("subgroups")`
  (SwiftShader size 4, NVIDIA 32 -- the spread that catches width
  assumptions).
- Nothing larger: property tests, large fixtures, planners, windowed
  uploads, indirect loops and device loss live in `node`.
- The runner wraps Chromium in a hard timeout (the `browser.close()` hang
  after GPU work reproduced locally; three.js SIGKILLs for the same reason,
  note 06 section 3.5).

### 11.9 Benchmarks and baselines

- `benchmarks/harness.ts` copied from graph-format (`bench()`, `printTable`,
  `appendSession`, note 07 section 6) with an async `run` and a `gpu`
  session field `{ vendor, architecture, device, limits }`; datasets from
  `datasets.ts` (seeded, host-independent) at 4k / 16k / 65k / 262k / 1M
  nodes, average degree 2-10, plus the hub fixture.
- Groups: `upload` (hot prefix per tier), `layout` (exact and grid ms per
  iteration, per-frame cost with readback, 2D / 3D), `pagerank`, `bfs`,
  `wcc`, `betweenness`.
- Baselines checked into `benchmarks/results/` per runner class (dev box
  4070; later the GPU runner's card); the GPU lane compares medians and
  fails on > 3x (T-13); the default lane never times anything.
- Browser numbers come from the `browser` project's frame-loop test
  printing `performance.now()` deltas into the report artifact (Chromium
  timestamp queries are quantised to 100 us, note 05 section 3.1).

### 11.10 Coverage

Thresholds 80 / 80 / 75 / 80 (lines / functions / branches / statements)
measured by the `node` project on the default lane (lavapipe), `src/wgsl/**`
excluded (template strings), thresholds skipped when a single `--project`
is selected (the `algorithms/vitest.config.ts` pattern, note 07 section
4.5). Limit branches are covered by the faked-caps planner tests; the
browser project carries no coverage. Coverage artifacts come only from the
default lane so `tools/merge-coverage.sh --ci` never sees a missing package
when the GPU lane is skipped (note 06 section 4.4).

### 11.11 What the walking skeleton proves (phase 1 gate, restated as tests)

1. The same `src/` runs under Dawn (Node) and Chromium (browser) with the
   device passed in; no `navigator` reference in `src/`.
2. `planUpload` picks the arena path for the 100k / 1M benchmark graph at
   raised limits and the per-array path for `fromCsr` / `transpose()`
   snapshots; both upload paths produce bindings whose contents equal the
   CPU views (segment by segment).
3. A kernel bound through the residency (`degree`: `rowPtr[i+1] -
   rowPtr[i]`) equals `outDegree()` on every fixture, including
   `arcCount === 0` (no zero-length binding, dispatch skipped) and a 17M-node
   synthetic `rowPtr` (2D dispatch; lavapipe-sized).
4. Readback copies out before unmap, returns `Uint32Array<ArrayBuffer>`,
   honours `dest`, and the staging ring reuses slots without validation
   errors under 100 back-to-back submissions.
5. `release(snapshot)` destroys every buffer (leak counter 0) and
   `dispose()` leaves the pool empty; `device.destroy()` produces
   `E_DEVICE_LOST` on a pending readback.
6. Validation errors and compilation errors are thrown as typed errors,
   not printed.
7. The default CI lane runs 1-6 on lavapipe and SwiftShader; the GPU lane
   runs them on the 4070 with the vendor assertion; T-1 / T-2 / T-3 numbers
   are recorded.

---------------------------------------------------------------------------

## 12. CI/CD

### 12.1 Lanes and runners (P-11; note 06)

| Lane | Runner | Adapter | Trigger | Required check | Runs |
| --- | --- | --- | --- | --- | --- |
| default | `ubuntu-latest` (GitHub-hosted, free for public repos) | Dawn-in-Node on Mesa lavapipe (`apt-get install mesa-vulkan-drivers libvulkan1`; `create(["adapter=llvmpipe"])`, the `webgpu` package's own CI pattern); Chromium on bundled SwiftShader | every push and PR | YES | build, lint, typecheck, strict-consumer tsc, knip, `node` project with coverage, `browser` smoke, coverage upload |
| gpu | self-hosted ephemeral runner on the dev box (RTX 4070 SUPER), labels `[self-hosted, linux, x64, gpu, nvidia]`, job `container` with `--gpus all`, `NVIDIA_DRIVER_CAPABILITIES=all`, image with `libegl1 libvulkan1 mesa-vulkan-drivers` | NVIDIA via Dawn and via headless Chromium (four flags); `GRAPHTY_GPU_REQUIRE=nvidia` so a software adapter FAILS | push to master, nightly `schedule`, `workflow_dispatch`, `pull_request` only when `head.repo.full_name == github.repository` AND label `gpu` | NO (a powered-off dev box must not block merges) | `gpu-report.mjs`, `node` + `node-limits`, `bench` with baseline comparison, `browser` on the real GPU; uploads `gpu-report.json` and `bench/results.json` (90-day retention), never lcov |

Why not GitHub's T4 runners: larger runners require GitHub Team or
Enterprise Cloud and `gh api /orgs/graphty-org` reports `plan: free`; they
are also billed on public repos ($0.052/min Linux) (note 06 sections 0, 3.2).
Revisit if the org plan changes. Third-party fallback if the dev box proves
unreliable: Cirun.io (free for public repos, runners in your own cloud,
`gpu: nvidia-tesla-t4`), RunsOn (own AWS account), machine.dev; all need a
cloud account (note 06 section 3.4).

Cost: default lane $0 (public repo); GPU lane $0 in GitHub charges
("Runner usage in public repositories will remain free" for self-hosted,
note 06 section 3.1) plus electricity; a hosted T4 would be ~$0.52 per
10-minute job if the plan ever allows it.

Security of the self-hosted lane (public repo): ephemeral / JIT
registration (`config.sh --ephemeral --disableupdate` loop or
`generate-jitconfig`), never triggered by fork PRs (the `if` guard), repo
setting "Require approval for all external contributors", read-only
workflow permissions, no secrets in the GPU job, the runner in its own
container (never the dev workspace), the registration token held only by
the host-side loop (note 06 sections 3.3, 5.1).

Precedent: `atoms-org/cuda-ffi` uses `runs-on: cudaffi-gpu-runner` with
`container: { image: ghcr.io/apowers313/roc-dev:1.5.2, options: "--gpus all
--user root" }` and triggers on push / dispatch only
(`tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`); it never
solved "what runs without a GPU" (only lint), which the default lane here
does (note 06 section 2).

### 12.2 Environment variables (one place: `test/setup/gpu.ts`)

| Variable | Default lane | GPU lane | Local |
| --- | --- | --- | --- |
| `GRAPHTY_GPU_ADAPTER` | `llvmpipe` | unset (Dawn picks the discrete GPU) | unset |
| `GRAPHTY_GPU_REQUIRE` | unset | `nvidia` | optional |
| `GRAPHTY_REQUIRE_GPU` | `1` (an adapter must exist; lavapipe counts) | `1` | unset (skip with reason) |
| `GRAPHTY_BROWSER_GPU` | `swiftshader` (flag set) | `nvidia` (flag set) | `nvidia` |
| `GRAPHTY_EGL_LIB_DIR` / `LD_LIBRARY_PATH` | -- | until the image has `libegl1`: the extracted tree (`HEADLESS_GPU_REPORT.md` appendix D) | same |
| `VK_DRIVER_FILES` | `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (determinism) | unset | unset |
| `XDG_RUNTIME_DIR` | `/tmp` (silences Mesa) | same | same |

### 12.3 Workflow sketch for this repo now

```yaml
name: CI
on:
    push: { branches: [master] }
    pull_request: { types: [opened, synchronize, reopened, labeled] }
    schedule: [{ cron: "17 6 * * *" }]
    workflow_dispatch:
permissions: { contents: read }
concurrency: { group: "${{ github.workflow }}-${{ github.ref }}", cancel-in-progress: "${{ github.event_name == 'pull_request' }}" }

jobs:
    test:                                  # ---- default lane (required)
        name: Test (software adapters)
        if: github.event_name != 'schedule'
        runs-on: ubuntu-latest
        timeout-minutes: 30
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm }
            - run: pnpm install --frozen-lockfile
            - name: Install Mesa lavapipe (software Vulkan for Dawn-in-Node)
              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
            - run: pnpm -r run build
            - run: pnpm -r run lint
            - run: pnpm --filter @graphty/webgpu-graph-algorithms run typecheck:strict-consumer
            - name: Node suite on lavapipe (coverage)
              env: { GRAPHTY_GPU_ADAPTER: llvmpipe, GRAPHTY_REQUIRE_GPU: "1", VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json, XDG_RUNTIME_DIR: /tmp }
              run: pnpm --filter @graphty/webgpu-graph-algorithms exec vitest run --project=node --coverage
            - uses: actions/cache@v4
              id: pw
              with: { path: ~/.cache/ms-playwright, key: "playwright-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}" }
            - if: steps.pw.outputs.cache-hit != 'true'
              run: pnpm exec playwright install chromium --with-deps
            - if: steps.pw.outputs.cache-hit == 'true'
              run: pnpm exec playwright install-deps chromium
            - name: Browser smoke on SwiftShader
              env: { GRAPHTY_BROWSER_GPU: swiftshader }
              run: timeout 600 pnpm --filter @graphty/webgpu-graph-algorithms exec vitest run --project=browser
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: coverage-webgpu-graph-algorithms, path: packages/webgpu-graph-algorithms/coverage/lcov.info, retention-days: 1, if-no-files-found: error }

    test-gpu:                              # ---- GPU lane (not required)
        name: Test (NVIDIA, self-hosted)
        if: >-
            github.event_name == 'push' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' ||
            (github.event_name == 'pull_request' &&
             github.event.pull_request.head.repo.full_name == github.repository &&
             contains(github.event.pull_request.labels.*.name, 'gpu'))
        runs-on: [self-hosted, linux, x64, gpu, nvidia]
        timeout-minutes: 45
        concurrency: { group: gpu-runner, cancel-in-progress: false }
        container:
            image: ghcr.io/graphty-org/dev:<tag>       # dev image + libegl1 libvulkan1 mesa-vulkan-drivers + Playwright deps
            options: "--gpus all"
            env: { NVIDIA_DRIVER_CAPABILITIES: all }
        env: { GRAPHTY_GPU_REQUIRE: nvidia, GRAPHTY_REQUIRE_GPU: "1", GRAPHTY_BROWSER_GPU: nvidia, XDG_RUNTIME_DIR: /tmp }
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x }
            - run: pnpm install --frozen-lockfile
            - run: pnpm -r run build
            - name: Report adapter (fails on software)
              run: node packages/webgpu-graph-algorithms/scripts/gpu-report.mjs | tee gpu-report.json
            - run: pnpm --filter @graphty/webgpu-graph-algorithms exec vitest run --project=node --project=node-limits
            - run: pnpm --filter @graphty/webgpu-graph-algorithms exec vitest bench --project=bench --outputJson bench/results.json
            - run: pnpm --filter @graphty/webgpu-graph-algorithms run bench:compare -- bench/results.json   # > 3x baseline fails
            - run: timeout 900 pnpm --filter @graphty/webgpu-graph-algorithms exec vitest run --project=browser
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: "gpu-results-${{ github.run_id }}", path: "gpu-report.json\nbench/results.json", retention-days: 90 }
```

### 12.4 Self-hosted runner recipe (dev box; not committed to the package)

Sibling container from `ghcr.io/actions/actions-runner` with `libegl1
libvulkan1 mesa-vulkan-drivers` and Playwright's Chromium deps installed;
started on the host with `docker run --gpus all -e
NVIDIA_DRIVER_CAPABILITIES=all ...`; an entry loop that fetches a
registration token, runs `config.sh --unattended --ephemeral
--disableupdate --labels gpu,nvidia,rtx4070`, `run.sh`, wipes `_work`, and
repeats (or `generate-jitconfig` + `run.sh --jitconfig`). The dev container
itself has no Docker socket (note 06 section 1), so the runner must be a
sibling started by the owner on the host. Two repo settings flipped
beforehand: "Require approval for all external contributors" and read-only
workflow permissions; a `gpu` label created once (note 06 section 5.1).

### 12.5 Slotting into the monorepo at W1 (`.github/workflows/ci.yml`)

Mirrors `packages/move/root-touch-points.diff`: a `Build
webgpu-graph-algorithms (PR)` step and `build-webgpu-graph-algorithms`
artifact in `build`; two new `test` matrix shards
`webgpu-graph-algorithms-node` (`needs-vulkan: true` -> the apt step,
`GRAPHTY_GPU_ADAPTER=llvmpipe`) and `webgpu-graph-algorithms-browser`
(`needs-browser: true`, SwiftShader flags); the coverage upload condition
extended with `startsWith(matrix.shard, 'webgpu-graph-algorithms-')`;
`tools/merge-coverage.sh` PACKAGES, `tools/prepush.sh` (node project only),
`release.yml` download step, `pnpm-workspace.yaml`, `commitlint` scope,
`knip.config.ts`; and a separate `test-gpu` job (the sketch above with
`needs: build`) that is NOT in the `all-checks` gate (`ci.yml` lines
705-738 per note 06 section 4.3). The existing matrix has no `nx affected`
gating, so the shards simply join it; the GPU job is gated by event + label,
not by paths (a graph-format change is exactly when it should run). Design
14.5 / 16.7 are amended in the same PR (P-1). Release: `nx release` with
independent versioning and OIDC trusted publishing gated on CI success
(`release.yml` lines 19, 36-52 per note 02 section 3) needs no GPU; the
GPU lane never publishes.

### 12.6 Budgets, artifacts, retention

| Job | timeout | artifacts | retention |
| --- | --- | --- | --- |
| default `test` | 30 min (target <= 15) | `coverage-webgpu-graph-algorithms` (lcov) | 1 day |
| `test-gpu` | 45 min (target <= 20) | `gpu-results-<run>` (`gpu-report.json`, `bench/results.json`) | 90 days |
| nightly `test-gpu` | 45 min | same; opens / refreshes a tracking issue on failure | 90 days |

If the lavapipe `node` project passes ~10 minutes it is split into two
shards with `--shard=1/2` (the monorepo already shards graphty-element five
ways, note 06 section 8).

---------------------------------------------------------------------------

## 13. Phased implementation plan

Rules for every phase: (a) the gate is a list of tests and recorded numbers
that must be GREEN on the default lane and the GPU lane before the next
phase starts; (b) a phase adds only the primitives its slice needs; (c)
every EXTRAPOLATED number the phase touches is replaced by a MEASURED one
in `benchmarks/results/`; (d) nothing lands with `eslint-disable`,
`@ts-expect-error` (outside negative type tests), non-ASCII, or a CPU
fallback; (e) sizes are estimates in engineer-days (ed) for one engineer
plus rough source line counts, excluding tests unless stated.

### Phase 0 -- Repository reset and package skeleton (2-3 ed, ~400 lines config)

Scope: apply the scaffold triage of note 07 section 5 (delete
`src/types/index.ts` with `CSRGraph`, the empty `src/*` dirs, `test/setup/*`,
`test/helpers/*`, `vitest.config.ts`, `vite.config.ts`, `tsconfig.json`,
`eslint.config.js`, `knip.json`, `.husky`, `package-lock.json`, `.env*`,
`.github/workflows/test.yml`, `examples/`, `STRATEGY.md`,
`IMPLEMENTATION_CHECKLIST.md`; keep `CLAUDE.md`, move
`HEADLESS_GPU_REPORT.md` under the package `docs/`); create
`packages/webgpu-graph-algorithms/` mirroring graph-io (package.json,
project.json, tsconfig trio, vitest two projects, scripts, knip workspace
entry, pnpm workspace entry); `src/index.ts` exporting only
`WebGpuGraphError` and constants; `test/setup/gpu.ts` (acquisition, REQUIRE
flags, uncapturederror hook); `scripts/gpu-report.mjs`; the CI workflow of
section 12.3 with one trivial device test.

Deliverables: buildable, lintable, knip-clean package; both vitest projects
run one trivial device test; default lane green on lavapipe + SwiftShader;
GPU lane registered and green on the 4070 (the runner recipe of 12.4
executed by the owner).

Gate G0:
- `pnpm -r run build`, `lint`, `typecheck:strict-consumer`, `knip` pass.
- `vitest run --project=node` acquires an adapter and prints
  `adapter.info` under Dawn on: the dev box (NVIDIA), the dev box with
  `GRAPHTY_GPU_ADAPTER=llvmpipe`, and `ubuntu-latest` (lavapipe).
- `vitest run --project=browser` acquires SwiftShader on `ubuntu-latest`
  and NVIDIA locally with the four flags; the vendor assertion fails when
  `GRAPHTY_GPU_REQUIRE=nvidia` is set on the software lane (a deliberate
  red run recorded in the PR).
- `gpu-report.mjs` exits non-zero on a software adapter when
  `GRAPHTY_GPU_REQUIRE=nvidia`.
- Verified spelling of the Vitest 3 per-instance Playwright `launch` args
  recorded in the package CLAUDE.md (open item in note 07).

### Phase 1 -- Walking skeleton (4-6 ed, ~900 lines src, ~700 lines test)

Scope: L0 + L1 + minimal L2: `GpuContext` (`probe`, `create`, `from`, caps,
error scopes, `lost`), `createNodeGpuContext`, `WebGpuGraphError`,
`planUpload` (arena / perArray; windowed planning only, not execution),
`GraphResidency` for the core + `outDegree()` view + one column,
`BufferPool`, `Readback` ring, `PipelineCache`, `Kernel`, `planDispatch`
(1D / 2D), `UniformBlock`, `composeWgsl` + prelude, the `degree` kernel and
`degree()` function, `release()` / `dispose()`, `ctx.accelerator()` with
only `release`.

Deliverables: `degree(ctx, snapshot)` correct in Node and browser; the
benchmark harness with `upload` and `roundtrip` groups; README section
"Runtime model".

Gate G1 (section 11.11 in full), plus:
- Upload contract tests ported from `gpu-upload.test.ts` (arena at
  `byteOffset` 0 and 8, per-array, `arcCount === 0`, packed columns) pass on
  lavapipe and NVIDIA.
- 2D dispatch on 17M items equals the CPU (lavapipe, 68 MB).
- Leak counter 0 after `release` and `dispose`; `E_DEVICE_LOST` on
  `device.destroy()` mid-readback in both runtimes.
- MEASURED and recorded: T-1 (upload 16.4 MB and 164 MB), T-2, T-3 on the
  4070 under Dawn and Chromium.
- Coverage >= 80/80/75/80 on the node project (lavapipe).

### Phase 2 -- Kernel and memory infrastructure hardened with faked limits (3-4 ed, ~600 lines src, ~600 lines test)

Scope: windowed upload EXECUTION for row-walking kernels (`ArcWindow`,
`rebase` uniform, row clamping), grid-stride dispatch, dynamic uniform
offsets, `Profiler` (timestamp queries when present), pipeline warm-up,
`reduce` primitive (sum / min / max / vec4, fixed-point min/max variant,
subgroup variant), `test/helpers/caps-tables.ts` (spec default,
SwiftShader, lavapipe, NVIDIA), `oracle.ts` skeleton.

Gate G2:
- `planUpload` unit tests: every path chosen correctly for the four caps
  tables x {arena, no arena} x {fits, exceeds buffer, exceeds binding}; a
  row longer than a window is split on arc ranges; `arcStart % 64 === 0`
  with `%` (a case above 2^31 arcs is included as a pure computation).
- `planDispatch`: the `(16,776,960, 16,777,216]` boundary, 2D and stride
  shapes; a WGSL-side linear-id test at the boundary on lavapipe.
- Windowed `degree` with FAKED `maxStorageBufferBindingSize = 1 MiB` on the
  100k / 1M graph (>= 8 windows) equals `outDegree()`; the same with a
  hub row longer than a window.
- `reduce` equals the f64 oracle within 1e-6 relative at sizes 0..2^24
  (scaled), subgroup variant only where `subgroups` exists, bitwise
  deterministic across two runs.
- `UniformBlock` layouts compile on Chromium (no
  `uniform_buffer_standard_layout`) and Dawn; a negative test proves a
  misaligned hand-written struct is rejected.
- `node-limits`: a real 2 GiB binding request succeeds on the 4070 under
  Dawn and a 200 MiB per-array upload is bound windowed at defaults.
- Coverage holds; default lane <= 10 min.

### Phase 3 -- ForceAtlas2, exact tier, as a `LayoutSimulation` (6-8 ed, ~1,200 lines src incl. WGSL, ~1,000 lines test)

Scope: `segmentedReduce` (tiered gather with `degreeOrder`), the FA2
kernels K1-K5 of section 7.4 in 2D and 3D, `ForceAtlas2Simulation`
(`load` / `reload`, `step(k)`, `settled`, `setFixed`, `setPosition`,
`dispose`, `trace`), LCG seeding, layout-unit / scene-unit write-back,
settle logic, `seed.ts`, `settle.ts`, the FA2 oracle in `oracle.ts` (f64,
index-based, the section 7.3 laws), the `frame-loop.ts` helper simulating
the element's synchronous loop with the `GpuLayoutEngine` bridge logic,
benchmarks `layout` group (exact tier), browser smoke for FA2.

Deliverables: `createForceAtlas2(ctx, options)` usable from Node (batch loop)
and the browser (frame loop); README "Layouts".

Gate G3:
- Section 11.4 layout parity: force parity <= 1e-4, trace parity (1e-4
  first 10 iterations, 5e-2 to 50), distributional parity within 10 %, the
  existing FA2 behaviour pins, all option combinations (weights on/off,
  linlog, distributedAction, strongGravity, gravity 0, nodeMass vector,
  2D / 3D, seed / unseeded, `arcCount === 0`).
- Section 11.6 properties for layouts (fixed, z, setPosition, settle /
  reheat, momentum, displacement bound, index alignment).
- Section 11.7 lifecycle: `dispose` leak 0; `release(snapshot)` during a
  live simulation throws `E_RELEASED` on the next `step`; device loss.
- Frame-loop test: 600 synchronous ticks with at most one submission in
  flight, positions advance, `settled` reported, no uncaptured errors; also
  in the browser project on SwiftShader and NVIDIA.
- MEASURED on the 4070 (Dawn) and recorded: ms per iteration at 1k, 4k, 8k,
  10k, 16k, 32k, 65k nodes (exact), and per-frame `step(1)` + readback in
  Chromium at 10k; T-4 met (10k <= 2 ms, 16k <= 4 ms); `exactMaxNodes`
  default re-fixed from the curve (largest n with <= 4 ms per iteration,
  rounded down to a power of two) and written into `constants.ts` with the
  measurement cited.
- lavapipe runs the whole FA2 suite at `gpuScale` sizes in <= 3 min.
- Owner sign-off on the P-5 formula table (open question Q-1) recorded in
  the PR before the WGSL is merged.

### Phase 4 -- Element bridge prototype and the layout accelerator surface (2-3 ed in this repo; E1 work in the monorepo later)

Scope (this repo): `LayoutAccelerator` structural interface in
`src/types/accelerator.ts`; `ctx.accelerator().forceAtlas2(options)`; a
documented `GpuLayoutEngine` bridge reference implementation under
`test/helpers/frame-loop.ts` (the code that graphty-element's E1 bridge
copies); `readbackEvery`; `reload` with a `FreezeReport`-shaped remap
(finite rows kept, NaN rows seeded). Scope (monorepo, at E1/L1): the API
changes of section 9.2 for `@graphty/layout` (steppable CPU FA2 with the
P-5 laws, `seedPositions`, `LayoutAccelerator`) and graphty-element (the
`accelerator` property, the bridge, the story).

Gate G4 (this repo):
- The bridge test drives `reload` after a synthetic topology change:
  existing rows keep coordinates bitwise, new rows are seeded, the
  simulation reheats.
- A differential test of the package's LCG against a copy of
  `layout/src/utils/random.ts` semantics (m = 2^35 - 31, a = 185852, c =
  1) for 10,000 draws; at W1 this test imports the real
  `RandomNumberGenerator`.
- Type test: `ctx.accelerator()` matches the structural
  `LayoutAccelerator`.
Gate G4 (monorepo, at E1): element tests green with a fake accelerator; the
"Layout/ForceAtlas2 (GPU)" story renders and settles on the real GPU
locally (screenshot compared by eye and via the Playwright + nanobanana
check the owner's rules require for visual work); Chromatic uses the fake.

### Phase 5 -- Grid-pyramid repulsion for large n, 2D and 3D (7-9 ed, ~1,100 lines src incl. WGSL, ~800 lines test)

Scope: `scan` (reduce-then-scan), `histogram` / counting sort, the grid
kernels (bbox, cellId + histogram, scan, scatter, per-cell mass-weighted
centroid by segmented reduce with per-bucket index sort, downsample per
level, far field, near field with `NEAR_MAX` and Horvitz-Thompson),
`repulsion: "auto" | "exact" | "grid"`, the locality-sorted attraction
gather (measured optimisation), fixtures of section 11.5, benchmarks
`layout` group (grid tier, 100k / 262k / 1M, 2D / 3D).

Gate G5:
- Section 11.5 in full: force-field RMS <= 5 % and p99 <= 25 % on uniform
  and clumpy fixtures; unbiasedness on the hub-cell fixture; distributional
  parity within 15 % over 200 iterations; per-iteration clamp holds;
  bitwise determinism; 3D pyramid <= 40 MB.
- `scan` and `histogram` primitives equal their oracles at sizes 0..2^24
  (scaled) including all-equal keys and one hot bucket.
- MEASURED and recorded on the 4070: T-6 (100k / 1M <= 10 ms; 1M / 10M <=
  100 ms; 3D at 100k <= 20 ms), T-7, T-5 at 100k in Chromium (<= 12 ms per
  frame); the exact-vs-grid crossover re-checked and `exactMaxNodes`
  adjusted if the grid is faster below it.
- lavapipe: the grid suite at `gpuScale` sizes in <= 4 min.
- Decision record: whether the Hilbert cluster-tree experiment is needed
  (only if a clumpy fixture fails the error bound) -- default no.

### Phase 6 -- Fruchterman-Reingold / spring and the ngraph-like preset (3-4 ed, ~500 lines src, ~400 lines test)

Scope: `createFruchtermanReingold` (two kernels + write-back; exact and
grid tiers through the shared 1/d kernel family), `fixed` mask, linear
cooling, the `preset: "ngraph"` spring-electrical variant with a velocity
buffer, FR oracle, browser smoke.

Gate G6:
- One-iteration displacement parity with the CPU FR oracle (<= 1e-4), the
  `|| 0.1` coincident guard, fixed nodes immobile, output NOT rescaled when
  `fixed` is given (the CPU rule, note 01 section 2.2).
- Preset: on the 150-node / 250-edge "Performance/Large Graph" story graph
  the preset settles within 1,000 steps to a layout whose edge-length
  distribution is within 25 % of ngraph's (ngraph run on the CPU in the
  test as the reference; ngraph is a devDependency of the test only).
- MEASURED: FR ms per iteration at 10k / 100k; recorded.

### Phase 7 -- PageRank family and the algorithm accelerator surface (4-5 ed, ~700 lines src, ~600 lines test)

Scope: `spmvPull`, the weighted out-degree normaliser (segmented reduce on
device with the zero-sum guard), device-side dangling and L1 reductions
into a 16-byte block copied into the next iteration's uniform, `pageRank`,
`personalizedPageRank`, `hits`, `eigenvectorCentrality`, `katzCentrality`,
the `AlgorithmAccelerator` structural interface and `ctx.accelerator()`
methods, `reverse()` residency (identity when undirected; `fwdArc` never
touched when `!directed`), `degreeOrder({ of: "reverse" })` tiers, PageRank
oracle (NetworkX semantics matching `pagerank.ts`), benchmarks `pagerank`.

Gate G7:
- Parity of section 9.3 on all fixtures: <= 1e-5 relative, top-k order,
  `iterations` within +-1, `converged` identical; weighted with zero-weight
  arcs and dangling nodes (the `outDegree > 0` trap of design 10.5); directed
  and undirected; personalization one-hot and uniform.
- Exactly 8 storage bindings in the PageRank kernel (a test inspects the
  layout descriptor); personalization moves to group 1 when absent.
- No host readback inside a batch of 8 iterations (a test counts `mapAsync`
  calls through the leak-counter proxy).
- MEASURED T-8 (100k / 1M x 100 <= 100 ms; 1M / 10M <= 1 s) recorded.

### Phase 8 -- Connected components (2-3 ed, ~400 lines src, ~400 lines test)

Scope: Afforest over `edgeList()` (`array<atomic<u32>>` comp, sampled
rounds, compress with `atomicLoad`, giant-component histogram sample, link
remaining, changed-flag loop), `renumberPartition` on readback (first-seen
order), `compact` primitive (flag + scan + scatter; ownership dedupe),
`histogram` reuse, WCC oracle (union-find).

Gate G8:
- Partition equality after renumbering on all fixtures incl. directed
  inputs treated weakly, disconnected singletons, one giant component +
  dust, `arcCount === 0`; property: endpoints share labels, labels dense.
- No mixed atomic / non-atomic access (a WGSL review checklist item is
  enforced by a test that compiles every variant on Dawn AND Chromium).
- MEASURED T-9 recorded.

### Phase 9 -- Frontier machinery: BFS, direction-optimizing BFS, SSSP, closeness, APSP (8-10 ed, ~1,400 lines src, ~1,000 lines test)

Scope: `Frontier` (queues, length atomic, indirect args, finalise kernel),
`advance` (block_mapped + workgroup-per-row tier + subgroup tier variant),
fused expand-contract kernel for tiny frontiers, `bitset`, BFS with the
Beamer / cuGraph switch (alpha = m/n, beta = 24 defaults; 14 / 24 option),
near-far SSSP with `atomicMin` on f32 bit patterns and the two-pass
predecessor, Bellman-Ford, closeness by batched multi-source BFS, APSP
(BFS rows / blocked FW with the `E_LIMIT` check), window-aware advance
(lifting the phase-2 `E_UNSUPPORTED`), oracles (FIFO BFS, Dijkstra with a
binary heap, Bellman-Ford), benchmarks `bfs`.

Gate G9:
- BFS `depth` exact and parent / order level-consistent on all fixtures
  incl. the 1000 x 1000 grid (2,000 levels) and a 10k-degree star; the
  fused small-frontier kernel and the two-phase kernel agree; the
  direction-optimizing path agrees with top-down and is exercised on an
  RMAT fixture (a test asserts the switch happened via `details.switches`).
- SSSP `dist` within 1e-5 on non-negative weights incl. zero weights;
  `predArc` attains `dist`; `E_UNSUPPORTED` on negative weights;
  `flags.allWeightsOne` routes to BFS; Bellman-Ford detects a planted
  negative cycle.
- `mapAsync` count <= levels / 16 + 1 on the grid fixture (the no-per-level-
  readback rule); indirect dispatch clamps above 65,535 workgroups (a
  synthetic frontier of 17M on lavapipe).
- Subgroup tier: identical results with the variant on (NVIDIA 32, lavapipe
  8, SwiftShader 4) and off.
- MEASURED T-10 recorded (RMAT 1M / 10M <= 100 ms; grid <= 1.5 s).

### Phase 10 -- Betweenness centrality (4-6 ed, ~700 lines src, ~500 lines test)

Scope: McLaughlin-Bader forward pass (u32 sigma with overflow flag, `S` /
`ends`), successor-pull backward pass, tagged multi-source batching planned
from `maxBufferSize`, the online work-efficient / edge-parallel switch,
sampling (`sources: number | U32`), edge betweenness via per-arc
accumulation + `foldArcs(..., "first")` (halved when undirected),
normalisation identical to the CPU, Brandes oracle.

Gate G10:
- Exact BC on karate, path, star, cycle, grid, random 2k: <= 1e-4 relative,
  top-k order; analytic sums on path and star; edge BC folded correctly
  (both arcs equal before folding, asserted).
- Sampled BC (256 sources) on a 100k-node RMAT: Spearman rank correlation
  >= 0.9 with exact BC computed by the CPU oracle on a 10k subgraph (the
  full exact is too slow); the overflow flag fires on a constructed
  small-world graph and is reported, never silent.
- Batch planning honours faked `maxBufferSize` (k shrinks; results equal).
- MEASURED T-11 recorded.

### Phase 11 -- Structure and community: k-core, triangles / k-truss, label propagation, Boruvka MST, Louvain (10-14 ed, ~1,800 lines src, ~1,200 lines test)

Scope: `radixSort` (LSD 8-bit x 4), `cooToCsr`, per-row group-by-key
(workgroup sort / global hash), k-core peeling, oriented intersection
kernel (triangles, k-truss, common neighbours), label propagation, Boruvka,
Louvain (move phase with the up/down rule, reduce-by-key cluster weights,
device contraction), Leiden refinement if time allows; oracles for each.

Gate G11 (per algorithm, each merged separately):
- `radixSort` equals `Array.sort` on 32- and 64-bit keys with values, sizes
  0..2^22 (scaled), all-equal keys; `cooToCsr` output passes
  `fromCsr(...).validate({ level: "full" })` on the CPU.
- k-core exact; triangles exact on every fixture (per-node and total);
  k-truss support exact; LPA recovers planted partitions (ARI >= 0.9) on 10
  seeds; Boruvka `totalWeight` within 1e-5 and edge set identical on
  distinct weights; Louvain modularity within 0.02 of the CPU on karate /
  planted partitions and never below the CPU's by more than 0.05 on random
  fixtures; every level's contraction preserves `totalWeight`.
- MEASURED per-algorithm numbers at 100k / 1M and 1M / 10M recorded;
  Louvain expectation 2-10x over the CPU stated with the number.

### Phase 12 -- Move-in (W1), monorepo CI, design amendments (3-4 ed)

Scope: `packages/README.md` checklist verbatim; the section 12.5 `ci.yml`
diff; conformance type tests against the real `AlgorithmAccelerator` /
`LayoutAccelerator`; differential tests switched to `indexed.*`;
`seedPositions` cross-test against the real LCG; design 14.5 / 16.7
amendments; README performance table regenerated from
`benchmarks/results/`; `HEADLESS_GPU_REPORT.md` referenced from the package
CLAUDE.md.

Gate G12:
- Monorepo default shards green on lavapipe + SwiftShader; the `test-gpu`
  job green on the self-hosted runner; `all-checks` unchanged (GPU job not
  required); coverage merged by `tools/merge-coverage.sh --ci` with the new
  package in PACKAGES; `nx release` dry run versions the package
  independently.
- `expectTypeOf(ctx.accelerator()).toMatchTypeOf<AlgorithmAccelerator &
  LayoutAccelerator>()` compiles; the strict-consumer compile passes.
- Every differential test passes against `indexed.*` with the tolerances of
  section 9.3 (a mismatch here is a bug in one of the two packages and
  blocks W1).

### Phase order and parallelism

```
P0 -> P1 -> P2 -> P3 (FA2 exact) -> P4 (bridge) -> P5 (grid) -> P6 (FR / preset)
                               \-> P7 (PageRank family) -> P8 (WCC) -> P9 (frontier) -> P10 (BC) -> P11 (structure, community)
P12 (W1) after the monorepo's A2 / L1 / E1 land and at least P3-P5 and P7-P8 are green.
```

P7 can start after P2 in parallel with P3-P6 (it needs only `reduce` and
`segmentedReduce`, which P3 also builds -- coordinate on `segmentedReduce`
first). The owner's priority makes P3 the first product slice; P5 is the
second because 100k-node interactivity is the format's mobile tier target
(design 15.3).

Rough total: 58-79 engineer-days of implementation across P0-P12 plus
monorepo-side work at A2 / L1 / E1 (~5-8 ed, not counted). The first
usable layout (P3 + P4) is reachable in 15-20 ed from the reset.

---------------------------------------------------------------------------

## 14. Risks and open questions for the owner

### 14.1 Risk register

| Id | Risk | Likelihood / impact | Mitigation | Owner default |
| --- | --- | --- | --- | --- |
| R-1 | FA2 semantics: the CPU port deviates from the published law in three places (note 01 section 2.1.9); if the CPU rewrite keeps the port's `1/d^2` while the GPU uses `1/d`, the two never agree and every parity test is meaningless | high / high | P-5 fixes ONE table for both; the P3 gate requires owner sign-off before WGSL merges; the layout tests assert no exact coordinates (note 01 section 2.1.8), so adopting the published law does not break them; Chromatic re-baselines once in L1 | adopt Jacomy / Gephi / cuGraph laws (Q-1) |
| R-2 | The exact-tier crossover is wrong for other GPUs (integrated, Apple, the T4 class) | medium / low | `exactMaxNodes` is an option; the default halves on non-discrete architectures; measured on the 4070 in P3; the grid tier is correct at any n so a wrong crossover only costs time | 16,384 discrete / 8,192 otherwise, re-fixed in P3 |
| R-3 | Grid pyramid quality on clumpy layouts (hub cells, empty space) is worse than Barnes-Hut; cosmos reports shimmer on a 163-node graph before its fixes | medium / medium | exact tier below the crossover; `NEAR_MAX` cap with Horvitz-Thompson weighting; per-iteration step clamp; P5 fixtures include cosmos's failure cases; the Hilbert cluster tree is the documented escape hatch sharing `radixSort` | grid first, tree only if G5 fails |
| R-4 | Chromium per-frame readback (2.65 ms per MiB MEASURED) caps interactive n well below the compute limit | high / medium | `readbackEvery` on the bridge; positions stay GPU-authoritative; at 1M nodes the renderer cannot draw per-node meshes anyway (note 01 section 6); a future Babylon `WebGPUEngine` device share removes the copy (`GpuContext.from(device)` is already in the API) | per-frame readback up to ~200k, then every 2-4 frames |
| R-5 | lavapipe is ~350x slower on O(n^2) kernels; the default lane could exceed its budget as the suite grows | medium / medium | `gpuScale` fixture scaling; per-file budget (~2 min); shard the node project; heavy sizes only in `bench` / `node-limits` on the GPU lane | 15-minute lane target (T-12) |
| R-6 | The self-hosted GPU runner is offline or compromised (public repo) | medium / medium | not a required check; nightly schedule; fork PRs never reach it; ephemeral registration, approval for external contributors, no secrets; Cirun / RunsOn / machine.dev as paid escape hatches; T4 hosted runners only if the org plan changes | dev-box runner now |
| R-7 | `webgpu` npm cannot be upgraded past 0.4.0 on the 22.04 container; a future Dawn fix (e.g. the 0.6.1 unmap shim) is unavailable | medium / low | the readback helper never relies on `device.destroy()` unmapping; plan the container / runner image move to 24.04 (glibc 2.39) and bump once everywhere (P-12) | stay on 0.4.0 until the image moves |
| R-8 | Subgroup size is not constant (32 NVIDIA, 8 lavapipe, 4 SwiftShader); a kernel assuming 32 silently miscomputes | high / high | every subgroup kernel reads `subgroup_size`; the workgroup-memory fallback is the default variant; the three-adapter spread in CI is the test | always ship the non-subgroup variant |
| R-9 | Windowed bindings for the 10M tier are only supported by row-walking kernels until P9; frontier algorithms on > 33M arcs at default limits throw `E_UNSUPPORTED` | low / medium | raised limits on discrete GPUs make windows rare; `E_UNSUPPORTED` with the numbers in `details`; window-aware advance lands in P9 | accept until P9 |
| R-10 | Dawn-node's `uniform_buffer_standard_layout` masks a layout bug that only Chromium rejects | medium / medium | `UniformBlock` generates padded structs; the browser smoke compiles every kernel variant; the P8 "compile every variant on both" test | generated layouts only |
| R-11 | Float accumulation error (f32 tile sums, PageRank pull, BC over many sources) exceeds the 1e-5 / 1e-4 tolerances on adversarial graphs | medium / low | tree-shaped partials, Kahan add in hub loops, documented tolerances scaled by iteration count; f64 CPU results remain the reference | keep tolerances; document |
| R-12 | Behaviour change: `weight` becomes live for FA2 through the snapshot; parallel arcs sum in the gather (design 14.3 already documents the multigraph change) | medium / low | documented in the L1 / E1 changelogs; the element's `weightPath` default `"weight"` (design 14.4) | accept (P-15) |
| R-13 | `browser.close()` hangs after GPU work on the NVIDIA path | high / low | job-level timeouts, `timeout` wrapper on the browser step, `fileParallelism: false` | hard kill backstop |
| R-14 | The element's frame loop is synchronous; a bridge that coalesces steps changes perceived layout speed versus ngraph (one physics step per `step()`) | medium / low | `maxBatch = stepMultiplier`; settle semantics unchanged; the bridge test asserts one submission in flight; stories compare feel | coalesce, document |
| R-15 | Timestamp-query quantisation (Chromium 100 us; Dawn-node unverified) makes per-kernel profiling noisy | low / low | gates use wall-clock around `await step()`; the profiler is a benchmark aid only | wall-clock gates |
| R-16 | Louvain on the GPU gives only 2-10x and loses parallelism at coarse levels (nu-Louvain finding) | high / low | scheduled last; expectation stated up front; no in-package CPU handoff (rule) -- the caller chooses the CPU package for small graphs | ship with the measured number |
| R-17 | Design 14.5 says "browser-only vitest project"; leaving the design unamended causes confusion at W1 | certain / low | amend 14.5 and 16.7 in the W1 PR (P-1) | amend |
| R-18 | `dropCaches()` on a snapshot leaves GPU buffers keyed by dropped arrays until `release`; an element that never calls `release` leaks | medium / medium | per-snapshot residency record; `release` from `snapshot-replaced` is in the design; a `ctx.residentBytes` counter and a warning when a released-less snapshot count exceeds 2 | counter + warning |
| R-19 | Dawn-node process lifetime: a CLI benchmark that keeps the `GPU` object reachable never exits | low / low | `createNodeGpuContext().dispose()` drops the reference; vitest `forks` kills workers | dispose in `afterAll` |
| R-20 | Vitest 3 -> 4 migration changes the browser provider API (`@vitest/browser-playwright`, non-merging per-instance provider) | medium / low | flags in one exported constant; the monorepo already pins overrides for both majors | follow the monorepo |

### 14.2 Open questions (each with the recommended default)

| Id | Question | Recommended default |
| --- | --- | --- |
| Q-1 | Which ForceAtlas2 laws do CPU (L1) and GPU implement: the current port's (`1/d^2` repulsion, `swing = m\|update\|`, asymmetric size correction) or the published Jacomy 2014 / Gephi / cuGraph laws? | Published laws (P-5); per-iteration reset of the global sums (as the port and Gephi do, not NetworkX's accumulation) |
| Q-2 | Should `exactMaxNodes` default be re-fixed by measurement in P3 (proposed) or fixed now at 16,384? | Measure; ship the measured value |
| Q-3 | Gravity toward the centroid (current port, NetworkX) or the origin (Gephi, cuGraph)? | Centroid (parity with the port); expose `gravityCenter: "centroid" \| "origin"` only if a story needs it |
| Q-4 | Is the settle threshold in scene units (0.05 per node, ngraph-like) acceptable for FA2, given FA2 has no usable convergence test of its own? | Yes, with `maxIter` as the hard bound |
| Q-5 | Should graphty-element get an `accelerator: "auto"` convenience (optional peer + isolated dynamic import) in E1, or only the injected property with app-side detection? | Property + app detection first; "auto" later |
| Q-6 | Route the element's default `ngraph` layout to the GPU FR preset above a node-count threshold? | Reserve the preset (P6); the app decides the threshold |
| Q-7 | Where does the self-hosted GPU runner live: a sibling container on the dev box (needs the owner to run it on the host) or a cloud runner via Cirun? | Dev box now; Cirun if availability bites |
| Q-8 | Should `tmp/webgpu-plan/` (notes, probes, cloned repos) be committed, moved to `docs/research/` (notes only), or stay gitignored? | Commit the seven notes and this plan under `packages/webgpu-graph-algorithms/docs/research/`; keep repos and probes gitignored |
| Q-9 | Package manager for the staging repo: pnpm (packages/ already uses it) or the scaffold's npm? | pnpm |
| Q-10 | Upgrade the dev container to Ubuntu 24.04 now (unblocks `webgpu@0.6.x`, newer Mesa lavapipe, removes the `libEGL` workaround) or after W1? | After P1 is green, before P5 (one environment change at a time) |
| Q-11 | Is `nodeSize` / `adjustSizes` needed in v1? | No (typed `never`); add after the CPU rewrite fixes the sign |
| Q-12 | 3D grid cap 128^3 (34 MB) versus 160^3 (65 MB)? | 128^3 |
| Q-13 | Should the browser smoke include a Firefox or WebKit instance later? | Not until those ship the needed features on Linux CI; Chromium only |
| Q-14 | Should the GPU lane also run the graph-format GPU audit tests (`gpu-upload.test.ts`) on NVIDIA as a canary? | Yes, cheap, in `test-gpu` |
| Q-15 | Betweenness default: exact for n <= 10k, sampled (256 sources) above? | Yes; the adapter surfaces the mode |

---------------------------------------------------------------------------

## 15. References

Research notes (this repository, `tmp/webgpu-plan/`; each carries its own
URL list):

- `01-layout-needs.md` -- CPU force layouts, graphty-element frame loop,
  `LayoutSimulation` contract, FA2 option parity, settle semantics.
- `02-algorithm-needs.md` -- algorithm inventory, plug-in mechanism
  comparison, result-shape contract, priority scores.
- `03-gpu-layout-prior-art.md` -- cosmos.gl, GraphWaGu, cuGraph FA2 /
  Burtscher, ForceAtlas2 paper and Gephi, crossover analysis, grid design.
- `04-gpu-algorithms-prior-art.md` -- Merrill, Beamer, Davidson,
  McLaughlin-Bader, cuGraph, Gunrock, GAP Afforest, Louvain, CUDA -> WGSL
  translation table.
- `05-webgpu-platform.md` -- Dawn-node facts, limits, features, latencies,
  WGSL rules, vitest configs.
- `06-gpu-ci.md` -- runner options, lavapipe / SwiftShader verification,
  cuda-ffi precedent, workflow sketches, security.
- `07-format-api-and-conventions.md` -- graph-format API as implemented,
  package skeleton, scaffold triage, conventions.

Project files:

- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
  sections 10 (lines 2320-2553), 14.3 (3959-4047), 14.4 (4048-4211), 14.5
  (4212-4244), 14.6 (4245-4278), 15 (4279-4449), 16 (4450-4657) -- the
  accepted design this plan implements against.
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-upload.test.ts`
  -- device acquisition and upload-contract helpers to port.
- `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md`
  -- headless Chromium on the NVIDIA GPU (flags, `libEGL.so.1`).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/dawn-perf.mjs`
  -- the exact-tile repulsion probe (MEASURED 1.11 ms at 20k on the 4070).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`
  -- the self-hosted GPU runner precedent.
- `/home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/forceatlas2.ts`,
  `graphty-element/src/managers/{LayoutManager,UpdateManager}.ts`,
  `graphty-element/src/layout/{LayoutEngine,NGraphLayoutEngine,ForceAtlas2LayoutEngine}.ts`,
  `graphty-element/src/config/GraphBehavior.ts` -- integration facts.

External (URL -- what it was used for):

- https://github.com/cosmosgl/cosmos -- cosmos.gl source: grid pyramid,
  Monte-Carlo near field, exact path below 4,096, link force, failure
  modes and benchmarks (`docs/many-body-force/README.md`, `history/2026/*`).
- https://cosmograph.app/examples , https://pypi.org/project/cosmograph/ --
  owner-supplied product links; no algorithmic content beyond cosmos.gl
  (not fetched in depth; note 03 section 1).
- https://github.com/harp-lab/GraphWaGu -- WebGPU FR + Barnes-Hut: WGSL
  radix sort, Hilbert codes, level-wise tree build, i32 fixed-point bbox
  atomics, CSR gather attraction; PacificVis / EGPGV timings.
- https://github.com/jaredmcqueen/analytics -- WebGL1 O(n^2) FR; cautionary
  only (GPL-3; "1M nodes at 60 fps" not credible for simulation).
- https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal
  -- Merrill, Garland, Grimshaw 2011: scan-based frontier expansion, gather
  tiers, duplicate culling, expand / contract couplings.
- https://dl.acm.org/doi/10.1145/3230485 (403 on fetch) and
  https://davidbader.net/publication/2018-mb/2018-mb.pdf -- McLaughlin and
  Bader, "Accelerating GPU Betweenness Centrality": work-efficient vs
  edge-parallel, successor-pull dependency accumulation, sampling switch.
- https://cse.buffalo.edu/tech-reports/2023-06.pdf -- Buffalo MS thesis on
  dense-matrix BC; excluded from the design (note 04 section 0 item 8).
- https://developer.nvidia.com/discover/cluster-analysis -- nvGRAPH-era
  overview of spectral and multilevel partitioning; background only.
- https://github.com/rapidsai/cugraph -- cuGraph source: FA2 (`cpp/src/layout/legacy/`),
  BFS / SSSP / PageRank / betweenness / Louvain / WCC / core number /
  triangle implementations and their verified constants.
- https://github.com/gunrock/gunrock -- `block_mapped` advance,
  neighbour-reduce, BFS / BC / PR / TC operators.
- https://raw.githubusercontent.com/sbeamer/gapbs/master/src/cc.cc -- GAP
  Afforest connected components.
- https://scottbeamer.net/pubs/beamer-sc2012.pdf -- direction-optimizing
  BFS (alpha 14, beta 24).
- https://escholarship.org/content/qt8qr166v2/qt8qr166v2.pdf -- Davidson et
  al. 2014, near-far SSSP and the ownership dedupe.
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679
  -- ForceAtlas2 paper (Jacomy et al. 2014): force laws, adaptive speed.
- https://raw.githubusercontent.com/gephi/gephi/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java
  (and `ForceFactory.java`, `Region.java`) -- Gephi FA2 formulas (GPL /
  CDDL: formulas only, no code).
- https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf -- Burtscher
  and Pingali 2011 Barnes-Hut on CUDA (O(n^2) fastest below ~10k bodies).
- https://liacs.leidenuniv.nl/~takesfw/pdf/exploiting-gpus-fast.pdf --
  Brinkmann, Rietveld, Takes 2017: FA2 on GPU timings (231 ms/iter at 1.13M
  on a Titan X; repulsion ~80 % of time).
- https://www2.evl.uic.edu/documents/pacificvisgraphwagu.pdf ,
  https://stevepetruzza.io/pubs/graphwagu-2022.pdf -- GraphWaGu papers.
- https://arxiv.org/html/2501.19004 , https://arxiv.org/html/2608.01503 --
  nu-Louvain and Gilbert-Madduri GPU Louvain findings.
- https://gpuweb.github.io/gpuweb/ and https://gpuweb.github.io/gpuweb/wgsl/
  -- WebGPU limits, `dispatchWorkgroupsIndirect`, device loss, error
  scopes; WGSL atomics (u32 / i32 only), recursion ban, `override`, uniform
  layout, `bool` not host-shareable, subgroup builtins.
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status -- browser
  availability matrix (2026-08-13).
- https://developer.chrome.com/blog/new-in-webgpu-120 , -121 , -128 , -134
  -- shader-f16, timestamp-query quantisation, `adapter.info`, subgroups
  shipping versions.
- https://registry.npmjs.org/webgpu , https://github.com/dawn-gpu/node-webgpu
  , https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md
  -- Dawn-in-Node versions, `create()` options, glibc requirements, CI
  pattern (`adapter=llvmpipe`).
- https://github.com/atoms-org/cuda-ffi -- owner-supplied self-hosted GPU
  runner precedent.
- https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/
  , https://docs.github.com/en/actions/reference/runners/larger-runners ,
  https://docs.github.com/en/billing/reference/actions-runner-pricing ,
  https://docs.github.com/en/enterprise-cloud@latest/actions/concepts/runners/larger-runners
  -- GitHub GPU runner spec, price and plan gate.
- https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners
  , https://docs.github.com/en/actions/reference/runners/self-hosted-runners
  , https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-configuration-for-a-just-in-time-runner-for-a-repository
  -- ephemeral / JIT runners and hardening.
- https://cirun.io/ , https://runs-on.com/runners/gpu/ ,
  https://machine.dev/docs/platform-specifications/gpu-runners/ --
  third-party GPU runner fallbacks (not evaluated in depth).
- https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/workflows/ci.yml
  , https://raw.githubusercontent.com/mrdoob/three.js/dev/.github/workflows/ci.yml
  -- lavapipe / SwiftShader CI patterns and budgets of other GPU projects.
- https://github.com/chromium/chromium/blob/main/docs/gpu/swiftshader.md --
  SwiftShader flags.
- https://vite.dev/guide/assets -- `?raw` imports (rejected in favour of
  template modules, P-9).
- https://github.com/jamescarruthers/d3-force-webgpu ,
  https://github.com/drkameleon/GraphGPU -- WebGPU layouts with racy
  edge-parallel scatters; counter-examples only.
- https://arxiv.org/abs/2303.03964 -- t-FDP (FFT far field); not pursued.

Not fetched or not verified (carried over from the notes so the plan does
not overstate): the GraphWaGu paper's Google Drive PDF; Cosmograph product
pages in depth; the RAPIDS cuGraph FA2 performance blog (403); Naim et al.
2017 (paywalled); the Gunrock paper; RunsOn pricing page; GitHub GPU runner
concurrency limits; Firefox / Safari optional feature exposure; Dawn-node
timestamp quantisation; whether vitest `pool: "threads"` works with the
Dawn addon; lavapipe on a real hosted runner (only the local container
run); every EXTRAPOLATED number in sections 7.12 and 10.
