# @graphty/webgpu-graph-algorithms -- implementation plan, draft A (performance-first)

Date: 2026-09-14. Status: PLAN DRAFT for the owner's review; no implementation code
exists beyond the illustrative snippets in this document.

Inputs (all read in full): research notes 01-07 under
`/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/` and the accepted
design `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
sections 10, 14.3-14.6, 15 and 16. Facts about existing code cite a path; facts about
external projects cite a note section or a URL (section 15). Numbers are labelled
MEASURED (probe scripts under `tmp/webgpu-plan/probe/`, note 05/06), PUBLISHED (a
paper or repository) or ESTIMATE (an extrapolation with its basis stated).

Vocabulary: n = nodeCount, E = edgeCount (logical), A = arcCount (`2E - selfLoops`
undirected, `E` directed), dim = 2 or 3, WG = workgroup size (256 unless stated).
"Element" = `@graphty/graphty-element`. "Format" = `@graphty/graph-format` as staged
at `packages/graph-format/`. "CPU packages" = `@graphty/algorithms` and
`@graphty/layout`.

Angle of this draft: performance first. Sections 4 (memory), 6 (primitives), 7 (force
layouts) and 8 (algorithm kernels) are the deepest; every other section is complete but
shorter.

---------------------------------------------------------------------------

## 1. Goals, non-goals and inherited decisions

### 1.1 Goals

| # | Goal | Source |
| --- | --- | --- |
| G1 | GPU ForceAtlas2 (then Fruchterman-Reingold) as steppable `LayoutSimulation` implementations that graphty-element can drive per frame, honouring pins, drag, 2D/3D, and reporting settlement | owner: "accelerated force directed layout is the first need"; design 14.3 |
| G2 | High performance at 10^5-10^6 nodes: exact O(n^2) repulsion only where it is the fastest choice, an approximate repulsion for everything larger, attraction as a CSR gather, no host round trips inside an iteration batch | owner: "conscientious about high performance for a large number of nodes"; design 15.3 tiers |
| G3 | Consume `GraphSnapshot` exactly as implemented in `packages/graph-format` (arena hot prefix, per-array, windowed uploads; views; `gpuView()` columns; `INVALID_INDEX`) | design 10, 14.5; note 07 |
| G4 | One code base for Node (Dawn via `webgpu@0.4.0`) and browsers; the core never touches `navigator` | owner; note 05 section 1 |
| G5 | Tests primarily in Node on Dawn (real NVIDIA locally, Mesa lavapipe on hosted CI), light browser smoke on Playwright Chromium | owner; note 05 section 9, note 06 |
| G6 | CI with a default lane (no GPU) and a GPU lane (self-hosted runner), ready to slot into the monorepo's shard matrix | owner: "one runner for GPU and a default runner"; note 06 |
| G7 | Optional / detected acceleration for the existing algorithm and layout packages through injected accelerator objects; the GPU package itself never falls back | owner; design line 2324-2325, 4243-4244; root `CLAUDE.md` |
| G8 | Algorithms grouped by primitive family, in a priority order driven by value x speedup / risk, using the cuGraph / Gunrock / Merrill / McLaughlin-Bader / Afforest designs translated to WGSL | owner: "use the nvidia gpu algorithms as input"; notes 02, 04 |

### 1.2 Non-goals (v1)

- No CPU, WebGL or SwiftShader "acceptance" path inside `src/` (project rule; the
  test layer may skip with `E_NO_ADAPTER` and must fail under `GRAPHTY_REQUIRE_GPU=1`).
- No rendering. Positions are read back into the element's `Float32Array`; sharing a
  device with a future Babylon `WebGPUEngine` is an optimisation left for later
  (note 01 section 4.8: the element renders on WebGL today).
- No on-device graph construction from records; the format ships CSR. On-device
  COO -> CSR exists only for derived graphs (Louvain contraction).
- No Kamada-Kawai, ARF, spectral, Louvain/Leiden, SCC, MST, k-truss in the first
  three phases (scheduled in section 8/13; KK/ARF/spectral only if demanded).
- No f16 kernels (the NVIDIA driver under both Dawn-node and Chromium 139 does not
  expose `shader-f16`; note 05 section 5).
- No multi-device, no SharedArrayBuffer, no workers inside the package.

### 1.3 Decisions inherited from the graph-format design (not relitigated)

| Design section | Decision this plan honours |
| --- | --- |
| 10.1 | Fields the GPU binds (`rowPtr`, `colIdx`, `weights`, `arcToEdge`/`edgeToArc` with `override USE_PERM`, `reverse()`, `coo().src`, `edgeList()`, `degreeOrder().perm`, `mate()`, `gpuView()` columns); entry points take `GraphSnapshot`; `parents` are `Uint32Array` + `INVALID_INDEX`; weighted normaliser computed on device |
| 10.2 | 4-byte arrays over plain `ArrayBuffer`; `u8` via `paddedU32View()` + `unpack4xU8`; `bool` as bit words; stride-3 columns read as `array<f32>` with `3*i` indexing, never `array<vec3f>` |
| 10.3 | Upload plan: whole arena hot prefix when it and every segment fit `device.limits`, else per array, else windows on arc ranges at 64-arc boundaries; views/columns each their own buffer |
| 10.4 | `gpuView()` eligibility table; results attached as columns by reference |
| 10.5 | Invariants assumed without checking (I1-I10, flags); never bind a zero-length array; guard the zero weight sum |
| 10.6 | 1D dispatch legal iff `ceil(count / WG) <= maxComputeWorkgroupsPerDimension` (16,776,960 at 65,535 x 256, NOT 2^24); 2D grid or grid-stride above; window start `rowPtr[v0] - (rowPtr[v0] % 64)` with `%` not `& ~63` |
| 10.7 | Readback conventions: per-node `U32`/`F32`, per-arc folded with `foldArcs`, copy out of `getMappedRange()` before `unmap()`, caller-supplied `dest?` |
| 10.8 | Device/limit queries, buffer creation, chunk planning, 2D dispatch, frontier queues, scans, dense relabel and the no-fallback rule live in the GPU package |
| 14.3 | `LayoutSimulation { load, step(iterations?), settled, setFixed(mask), setPosition(i,x,y,z), dispose }`; positions are the owner's stride-3 scene-unit `F32`, read and written in place; the GPU buffer is authoritative while stepping; FA2 default mass = `outDegree()[i] + 1`; weights via `snapshot.weights` (`weight === true`) or a named edge column expanded per column version |
| 14.4 | The element owns the builder and the position array; `snapshot-replaced` triggers `gpu.release(previous)`; adapters call `getSnapshot()` once then `indexed.*` or the injected accelerator, then one shared result-writing loop; layouts receive `dm.undirected(s).snapshot` |
| 14.5 | Move-in as `webgpu-graph-algorithms/`; `@webgpu/types`; `noUncheckedIndexedAccess` OFF; delete `CSRGraph` / `EdgeListGraph`; upload cache is a `WeakMap` keyed on the typed-array object plus `column.version`; `release(snapshot)` explicit |
| 14.6 | Landing order F1 -> A1 -> F2 -> A2 / L1 / E1 -> W1 -> D1 -> IO1 -> 2.0; the GPU package is a consumer of `indexed.*` types from W1 |
| 15.3 | Target tiers: 100k / 1M (mobile), 1M / 10M (desktop interactive), 10M / 100M (batch, `arena: false`, windowed bindings) |
| 16.2 | GPU parity tolerance `1e-5` for f32 scores; order-agnostic for component lists |

One explicit departure: design 14.5 says "a browser-only vitest project (Playwright
Chromium on the real GPU)". The owner's request supersedes it: the `node` project on Dawn
is primary and `browser` is a smoke suite (note 06 section 9 item 5). Section 14 asks the
owner to amend 14.5 and 16.7 when this plan is accepted.

### 1.4 What this plan decides that the design left open

| # | Decision | Section |
| --- | --- | --- |
| D1 | Runtime shape: `GpuContext` wrapping a caller-provided `GPUDevice`; `./browser` and `./node` subpaths do acquisition; `webgpu` is an optional peer loaded by dynamic import | 2 |
| D2 | Public API: algorithms are plain async functions `(ctx, snapshot, options?, dest?)`; layouts are factories returning `LayoutSimulation`; `ctx.accelerator()` bundles both behind the CPU packages' structural interfaces | 3, 9 |
| D3 | WGSL lives in `src/wgsl/*.wgsl.ts` template-string modules composed by string concatenation (prelude + overrides + operator snippets); no `?raw` | 3.4 |
| D4 | `GraphResidency` = `WeakMap<object, Resident>` keyed on the typed-array object PLUS `WeakMap<GraphSnapshot, ResidencyRecord>` so `release(snapshot)` frees buffers whose CPU keys were dropped by `dropCaches()` | 4 |
| D5 | Force accumulation is gather-only (attraction over the undirected CSR row, repulsion per node); integer atomics only for counting sort, bounding box (i32 fixed point) and sigma path counts; no fixed-point float scatter in v1 | 6, 7 |
| D6 | Two repulsion back-ends: exact tiled all-pairs (also the oracle) for `n <= exactMaxNodes` (default 16,384, configurable, `calibrate()` may raise it) and a cell-sorted uniform-grid pyramid (cosmos P3M re-expressed as compute kernels) above; GraphWaGu-style Hilbert cluster tree is a documented later experiment, Burtscher-style locked trees are rejected | 7.7-7.8 |
| D7 | FA2 swing/traction and the adaptive global speed stay on the device: workgroup partials + a single-workgroup `adaptSpeed` kernel; k iterations per submit with zero host round trips | 7.9 |
| D8 | Steppable FA2 (CPU and GPU) simulates in SCENE units with Gephi/cuGraph semantics; `scalingFactor` becomes the seeding radius; the one-shot `indexed.forceAtlas2` keeps the unit-ball rescale | 7.11, 14 |
| D9 | Reference formulas for parity: adopt the paper / Gephi / cuGraph definitions (repulsion `kr m_i m_j / d`, `swing = m |F(t) - F(t-1)|`, `traction = m |F(t) + F(t-1)| / 2`, symmetric size correction) in the L1 CPU rewrite; the GPU kernel exposes them as `override` constants so the old port's variants remain buildable for A/B tests | 7.2, 14 |
| D10 | `settled` = iteration budget exhausted OR 10-iteration mean displacement per free node below a threshold (default 0.05 scene units, the element's ngraph heuristic); `setPosition`, `setFixed` (clearing bits) and `load` reheat | 7.12 |
| D11 | Frame-loop bridge: fire-and-forget `step(k)` with at most one submission in flight, a ring of 3 `MAP_READ` staging buffers, positions copied into the element's array when the map resolves, `column.markDirty()` once per copy | 7.18 |
| D12 | Grid tier sorts nodes by cell with the stable LSD radix sort primitive (deterministic on a device) rather than an atomic-cursor counting sort; near-field capped at 64 samples per cell with Horvitz-Thompson weighting above the cap | 7.7, 7.13 |
| D13 | Algorithm priority: PageRank family -> WCC (Afforest) -> BFS / closeness / SSSP (near-far) -> betweenness (McLaughlin-Bader, sampled sources) -> LPA, k-core, Bellman-Ford -> Louvain last | 8 |
| D14 | Tests: `node` (full, Dawn), `node-limits` (real limits only), `bench`, `browser-smoke`; env `GRAPHTY_GPU_ADAPTER`, `GRAPHTY_GPU_REQUIRE`, `GRAPHTY_BROWSER_GPU`; fixtures scale by `caps.software` | 11 |
| D15 | CI: default lane on `ubuntu-latest` (Dawn on lavapipe + Chromium on SwiftShader, required check); GPU lane on an ephemeral self-hosted runner on the dev box, label-gated for same-repo PRs, never required | 12 |
| D16 | Phasing: P0 scaffold reset -> P1 walking skeleton -> P2 memory/dispatch infrastructure with faked limits -> P3 FA2 exact tier as `LayoutSimulation` -> P4 element bridge -> P5 grid repulsion + degree tiers -> P6 FR + presets -> P7.. algorithms -> W1 move-in | 13 |

---------------------------------------------------------------------------

## 2. Runtime model

### 2.1 One code base, two acquisition adapters

The core package (`.` entry) exports functions and classes that take a `GPUDevice`
(wrapped in `GpuContext`) and never reference `navigator`, `window` or `process`. This
is what makes the same WGSL and the same TypeScript run under Dawn-in-Node and under a
browser (note 05 section 1 item 1; note 02 finding 5). Acquisition is split into two
tiny subpath modules:

| Entry | Does | Runtime |
| --- | --- | --- |
| `@graphty/webgpu-graph-algorithms` | `GpuContext.probe({ gpu })`, `GpuContext.create({ gpu, ... })`, `GpuContext.from(device)`, every algorithm and layout | both |
| `@graphty/webgpu-graph-algorithms/browser` | `requestGpuContext({ powerPreference: "high-performance", raiseLimits: true })` over `navigator.gpu`; throws `WebGpuGraphError("E_NO_WEBGPU")` when `navigator.gpu` is absent, `E_NO_ADAPTER` when `requestAdapter()` is null | browser |
| `@graphty/webgpu-graph-algorithms/node` | `createNodeGpuContext({ adapter?, backend?, dawnFeatures?, software? })`: `const dawn = await import("webgpu")`, `Object.assign(globalThis, dawn.globals)`, `dawn.create([...])`, then the same `GpuContext.create`; holds the `GPU` object so `dispose()` can drop it (the process cannot exit while it is reachable; note 05 section 2.2) | Node |

`powerPreference: "high-performance"` is always passed in the browser adapter because
Chrome 145 only returned the NVIDIA adapter with an explicit preference
(`HEADLESS_GPU_REPORT.md` lines 177, 233-235 via note 05 section 3.2). In Dawn-node
`powerPreference` and `forceFallbackAdapter` are ignored; software selection is
`adapter=llvmpipe` (note 05 section 2.2), which `software: true` maps to (documented as
Linux / Mesa specific).

### 2.2 Detection helper (consumer side) versus no-fallback (package side)

- `GpuContext.probe({ gpu })` returns `{ ok: true, info, software, features, limits }`
  or `{ ok: false, reason }` WITHOUT creating a device (it requests an adapter and reads
  `adapter.info`, `adapter.features`, `adapter.limits`). It never throws for "no GPU";
  it throws only for a programming error (no `gpu` argument).
- `GpuContext.create(...)` throws `WebGpuGraphError` with code `E_NO_ADAPTER` or
  `E_NO_DEVICE`; it never returns a CPU stand-in (root `CLAUDE.md`; design 10.8).
- Every algorithm and layout throws when the context is disposed or lost
  (`E_DISPOSED`, `E_DEVICE_LOST`). Nothing in `src/` catches a GPU error to run a CPU
  path.
- "Detected" acceleration is the CONSUMER's job (note 02 section 4.5): the graphty app
  (or a test) calls `probe`, then `create`, then injects `ctx.accelerator()` into the
  element (`element.accelerator = gpu`) or into `indexed.accelerated(acc)` of the CPU
  packages (section 9). The only branch that chooses the CPU is
  `acc?.method === undefined`, evaluated BEFORE any GPU work.
- `isSoftwareAdapter(info)` is exported so consumers can decide to skip acceleration on
  SwiftShader / lavapipe (`architecture === "software"` for Mesa, `"swiftshader"` for
  Chromium; `isFallbackAdapter` is `undefined` in Dawn-node -- note 05 section 2.2).

### 2.3 Keeping `webgpu` (Dawn) out of browser bundles

- `package.json`: `webgpu` in `peerDependencies` (`^0.4.0`) with
  `peerDependenciesMeta.webgpu.optional = true`, and in `devDependencies` (`0.4.0`) for
  the Node test project (note 07 section 4.2). `@graphty/graph-format` in both
  `dependencies` (`workspace:*`) and `peerDependencies` (`^0.1.0`, `^1.0.0` after F2).
- `src/node/index.ts` is the ONLY module that mentions `webgpu`, and only inside
  `await import("webgpu")`. The root barrel never re-exports it. The vite library build
  externalises `webgpu` and `@graphty/graph-format` (`scripts/build-bundle.js`,
  graph-io pattern, note 07 section 4.6). `sideEffects: false`.
- The core never reads `GPUBufferUsage.*` / `GPUMapMode.*` / `GPUShaderStage.*` at
  module top level: usage flags are numeric constants in `src/constants.ts`
  (`BUF_STORAGE = 0x80`, `BUF_COPY_DST = 0x08`, `BUF_COPY_SRC = 0x04`, `BUF_MAP_READ =
  0x01`, `BUF_UNIFORM = 0x40`, `BUF_INDIRECT = 0x100`, `MAP_READ = 0x01`) with a comment
  naming the spec enum, so import order relative to `Object.assign(globalThis,
  dawn.globals)` cannot break the package (note 05 section 2.2). Tests assert the
  constants equal the runtime enums on both runtimes.

### 2.4 Pin and platform facts the runtime layer encodes

| Fact | Consequence |
| --- | --- |
| `webgpu@0.4.0` is the last version that loads on glibc 2.35 (0.6.1 needs GLIBC_2.38; note 05 section 2.1, MEASURED by `strings`) | devDependency pinned `0.4.0`; peer range `^0.4.0`; bump once when the dev container and runner image move to Ubuntu 24.04 |
| 0.4.0 lacks the 0.6.1 shim that unmaps buffers on `device.destroy()` | `Readback` always `unmap()`s or `destroy()`s its staging buffers itself |
| Dawn-node reports adapter raw limits (`maxBufferSize` 1 TiB, offset alignment 16); Chromium reports 4 GiB / 256 | planners read `device.limits`, never `adapter.limits`; never assert alignment `=== 256`; the format's 256-byte arena satisfies any alignment <= 256 |
| `uniform_buffer_standard_layout` is in Dawn-node but not Chromium 139 | every uniform struct uses explicit 16-byte layout (section 5.3) |
| `subgroups` present on NVIDIA (32), lavapipe (8), SwiftShader (4); `shader-f16` absent on NVIDIA | subgroup size is an `override`; subgroup kernels are a second module variant selected by `device.features.has("subgroups")`; no f16 |
| `timestamp-query` present everywhere probed; Chromium quantises to 100 us | timing is a Node-side profiling tool, never a runtime decision input (except the optional `calibrate()`) |
| Uncaptured validation errors print to stderr in Dawn-node and do not throw | `GpuContext` installs an `uncapturederror` listener and error scopes around creation calls (section 5.6) |

### 2.5 Device loss

`ctx.lost: Promise<GPUDeviceLostInfo>` resolves from `device.lost`. On loss the context
marks itself lost, rejects in-flight readbacks with `E_DEVICE_LOST`, drops residency
records (the buffers are gone), and never recreates a device. The consumer
(LayoutManager / DataManager) decides whether to build a new context and `load()` again
(note 05 section 7.4).

---------------------------------------------------------------------------

## 3. Package architecture

### 3.1 Layers

| Layer | Modules | Responsibility | Depends on |
| --- | --- | --- | --- |
| L0 device | `device/context.ts`, `device/caps.ts`, `device/errors.ts` | `GpuContext`, `GpuCaps` (limits, features, subgroup sizes, software flag, runtime), error class, uncaptured-error hook, device loss | `@webgpu/types` |
| L1 memory | `memory/residency.ts`, `memory/upload-plan.ts`, `memory/buffer-pool.ts`, `memory/readback.ts` | snapshot/view/column uploads and their cache, arena/per-array/windowed plans (pure functions of `GpuCaps` + byte lengths), size-class buffer pool, staging ring readback | L0, graph-format |
| L2 kernel | `kernel/pipeline-cache.ts`, `kernel/dispatch.ts`, `kernel/uniforms.ts`, `kernel/wgsl.ts`, `kernel/kernel.ts`, `kernel/timing.ts` | pipeline cache keyed by (source id, overrides, layout), dispatch planner (1D / 2D / grid-stride / indirect), 16-byte uniform packing, WGSL composition, `Kernel` = pipeline + bind-group-layout + dispatch helper, optional timestamp queries | L0, L1 |
| L3 primitives | `primitives/reduce.ts`, `scan.ts`, `segmented-reduce.ts`, `compact.ts`, `histogram.ts`, `radix-sort.ts`, `bitset.ts`, `frontier.ts`, `advance.ts`, `spmv.ts`, `coo-to-csr.ts`, `bbox.ts` | reusable device algorithms with persistent scratch, each with a CPU reference in `test/helpers/oracle.ts` | L2 |
| L4 algorithms | `algorithms/<family>/<name>.ts` | plain async functions `(ctx, snapshot, options?, dest?)` | L3 |
| L4 layouts | `layouts/force-atlas2.ts`, `layouts/fruchterman-reingold.ts`, `layouts/repulsion-exact.ts`, `layouts/repulsion-grid.ts`, `layouts/simulation-base.ts`, `layouts/stepper.ts` | `LayoutSimulation` implementations and the async frame-loop bridge | L3 |
| L5 facade | `accelerator.ts`, `index.ts`, `browser/index.ts`, `node/index.ts` | `ctx.accelerator()` object implementing the CPU packages' interfaces; barrels; acquisition adapters | L4 |

### 3.2 Directory tree (mirrors `packages/graph-io/`; note 07 section 4.1)

```
packages/webgpu-graph-algorithms/
+-- package.json  project.json  webgpu-graph-algorithms.ts  README.md  CLAUDE.md  LICENSE
+-- tsconfig.json  tsconfig.build.json  tsconfig.strict-consumer.json  vitest.config.ts
+-- scripts/entries.js  build-bundle.js  bundle-types.js  gpu-report.mjs
+-- src/
|   +-- index.ts                 # public barrel (explicit named exports; browser-safe)
|   +-- browser/index.ts         # requestGpuContext() over navigator.gpu
|   +-- node/index.ts            # createNodeGpuContext() over dynamic import("webgpu")
|   +-- constants.ts             # WORKGROUP_SIZE, MAX_1D_INVOCATIONS = 65535 * 256, ARC_WINDOW_ALIGN = 64, usage flags
|   +-- errors.ts                # WebGpuGraphError { code, details }
|   +-- types/                   # options, results, LayoutSimulation re-declaration, accelerator interfaces (structural)
|   +-- device/                  # context.ts caps.ts
|   +-- memory/                  # residency.ts upload-plan.ts buffer-pool.ts readback.ts
|   +-- kernel/                  # pipeline-cache.ts dispatch.ts uniforms.ts wgsl.ts kernel.ts timing.ts
|   +-- wgsl/                    # prelude.wgsl.ts reduce.wgsl.ts scan.wgsl.ts ... fa2-*.wgsl.ts (template strings)
|   +-- primitives/              # one file per primitive
|   +-- algorithms/              # spmv/ traversal/ components/ centrality/ community/ structure/
|   +-- layouts/                 # force-atlas2.ts fruchterman-reingold.ts repulsion-*.ts stepper.ts
|   +-- accelerator.ts
+-- test/
|   +-- setup/gpu.ts             # acquire(): Dawn or navigator.gpu; env policy; uncapturederror -> fail
|   +-- helpers/{device,graphs,oracle,fixtures,limits}.ts
|   +-- device/ memory/ kernel/ primitives/ algorithms/ layouts/   # node project
|   +-- limits/                  # node-limits project (real large limits only)
|   +-- browser/                 # browser-smoke project
|   +-- types/*.test-d.ts  index.test.ts  build-output.test.ts
+-- benchmarks/                  # run.ts harness.ts datasets.ts layouts.bench.ts algorithms.bench.ts primitives.bench.ts results/
```

### 3.3 Public TypeScript API (signatures)

```ts
/// <reference types="@webgpu/types" />
import type { GraphSnapshot, F32, U32, NodeMask, NumericVector } from "@graphty/graph-format";

export class WebGpuGraphError extends Error {
    readonly code: WebGpuGraphErrorCode;      // "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_NO_DEVICE" | "E_DEVICE_LOST" | "E_DISPOSED"
                                              // | "E_VALIDATION" | "E_OUT_OF_MEMORY" | "E_TOO_LARGE" | "E_UNSUPPORTED" | "E_BAD_OPTION" | "E_EMPTY"
    readonly details: Readonly<Record<string, unknown>>;
}

export interface GpuCaps {                     // captured once at creation (note 05 section 8.3)
    readonly limits: GPUSupportedLimits;       // device.limits
    readonly features: ReadonlySet<string>;
    readonly subgroupMin: number; readonly subgroupMax: number;   // 0 when absent
    readonly software: boolean; readonly vendor: string; readonly architecture: string;
    readonly runtime: "browser" | "node" | "unknown";
}
export function isSoftwareAdapter(info: GPUAdapterInfo): boolean;

export interface GpuProbeResult { ok: boolean; reason?: string; info?: GPUAdapterInfo; software?: boolean; features?: string[]; }
export interface GpuCreateOptions { gpu: GPU; powerPreference?: GPUPowerPreference; raiseLimits?: boolean /* default true */;
                                    requiredFeatures?: GPUFeatureName[]; label?: string; }

export class GpuContext {
    static probe(o: { gpu: GPU; powerPreference?: GPUPowerPreference }): Promise<GpuProbeResult>;
    static create(o: GpuCreateOptions): Promise<GpuContext>;          // throws E_NO_ADAPTER / E_NO_DEVICE
    static from(device: GPUDevice, info?: Partial<GPUAdapterInfo>): GpuContext;   // wrap an existing device
    readonly device: GPUDevice; readonly caps: GpuCaps; readonly lost: Promise<GPUDeviceLostInfo>;
    readonly isLost: boolean; readonly isDisposed: boolean;
    residency: GraphResidency;                                         // upload cache (section 4)
    accelerator(): GpuAccelerator;                                     // section 9
    release(snapshot: GraphSnapshot): void;                            // frees every buffer uploaded for this snapshot
    calibrate(): Promise<GpuCalibration>;                              // optional micro-benchmark: exactMaxNodes suggestion, pairs/s
    dispose(): void;                                                   // destroys buffers, pipelines, device (if owned)
}

// Algorithms: plain async functions. `dest` is written when given and its length matches (design 10.7).
export function pageRank(ctx: GpuContext, s: GraphSnapshot, o?: PageRankOptions, dest?: F32): Promise<PageRankResult>;
export function personalizedPageRank(ctx: GpuContext, s: GraphSnapshot, personalization: F32, o?: PageRankOptions, dest?: F32): Promise<PageRankResult>;
export function hits(ctx, s, o?): Promise<{ hubs: F32; authorities: F32; iterations: number; converged: boolean }>;
export function eigenvectorCentrality(ctx, s, o?, dest?): Promise<{ scores: F32; iterations; converged }>;
export function katzCentrality(ctx, s, o?, dest?): Promise<{ scores: F32; iterations; converged }>;
export function connectedComponents(ctx, s, o?, dest?: U32): Promise<{ labels: U32; count: number }>;
export function breadthFirstSearch(ctx, s, source: number, o?: BfsOptions): Promise<{ depth: U32; parent: U32; order: U32; visitedCount: number }>;
export function sssp(ctx, s, source: number, o?: SsspOptions): Promise<{ dist: F32; predArc: U32 }>;
export function bellmanFord(ctx, s, source: number, o?): Promise<{ dist: F32; predArc: U32; hasNegativeCycle: boolean }>;
export function closenessCentrality(ctx, s, o?, dest?): Promise<F32>;
export function betweennessCentrality(ctx, s, o?: { sources?: U32 | number; normalized?; directed? }, dest?): Promise<F32>;
export function edgeBetweennessCentrality(ctx, s, o?): Promise<F32 /* edgeCount, via foldArcs */>;
export function labelPropagation(ctx, s, o?, dest?): Promise<{ labels: U32; iterations; converged }>;
export function kCoreDecomposition(ctx, s, dest?): Promise<U32>;
export function allPairsShortestPath(ctx, s, o?): Promise<{ dist: F32 /* n*n */ }>;
export function degree(ctx, s, dest?): Promise<U32>;                   // walking-skeleton kernel, kept as a test surface

// Layouts
export function createForceAtlas2(ctx: GpuContext, o?: ForceAtlas2SimulationOptions): GpuLayoutSimulation;
export function createFruchtermanReingold(ctx: GpuContext, o?: FrSimulationOptions): GpuLayoutSimulation;
export interface GpuLayoutSimulation extends LayoutSimulation {       // LayoutSimulation re-declared structurally (design 14.3)
    step(iterations?: number): Promise<void>;                          // always async on the GPU
    readonly settled: boolean; readonly iterations: number;
    readonly inFlight: boolean;                                        // a submission is pending (frame-loop bridge, section 7.18)
    requestStep(iterations?: number): void;                            // fire-and-forget form used by the element
    stats(): LayoutStats;                                              // last speed, swing, traction, movement, repulsion tier
    reheat(): void;
}
```

The `GpuAccelerator` type (section 9) is `AlgorithmAccelerator & LayoutAccelerator & {
release(s): void; dispose(): void }`, declared STRUCTURALLY in `src/types/accelerator.ts`
until A2 / L1 land the canonical interfaces, and checked with `expectTypeOf` against the
CPU packages' exported interfaces at W1.

### 3.4 WGSL organisation and composition

- Every kernel is an exported template string in `src/wgsl/<name>.wgsl.ts`
  (`export const fa2Attraction = /* wgsl */ \`...\`;`). Reasons verified in note 07
  section 4.6: the pre-push build is tsc-only, knip globs `src/**/*.ts`, eslint lints
  `.ts`. `?raw` is not used.
- `kernel/wgsl.ts` composes `[enableDirectives, prelude, overridesDecl, snippets...,
  body].join("\n")`. The prelude carries `const INVALID_INDEX: u32 = 0xFFFFFFFFu;`, the
  `Params` layout helpers, `lowbias32(x)` integer hash, and `fixedToF32` / `f32ToFixed`.
- Optional-feature variants: a kernel that has a subgroup fast path is composed twice
  (`enable subgroups;` spliced in) and the variant is chosen at pipeline creation by
  `caps.features.has("subgroups")`; the workgroup-memory fallback is always present
  (note 05 section 6 item 7).
- Operator snippets (Gunrock-style `advance` / `filter`) are WGSL function bodies
  passed as strings into a kernel template: `advanceKernel({ visit: "fn visit(u, v, a)
  -> bool { ... }" })`. The composed source is hashed (FNV-1a over the string) for the
  pipeline cache key.
- Every kernel declares `override WG: u32 = 256;` and `@workgroup_size(WG)`; planners
  pass `WG = min(256, caps.limits.maxComputeInvocationsPerWorkgroup)` (128 in compat
  mode, note 05 section 4).

### 3.5 Class list (few classes; algorithms are functions)

| Class | One-line responsibility |
| --- | --- |
| `GpuContext` | owns the device, caps, residency, pipeline cache, buffer pool, error hooks, lifecycle |
| `GraphResidency` | upload cache for cores, views, columns and derived scratch; plan selection; `release(snapshot)` |
| `BufferPool` | size-class recycling of `GPUBuffer`s by usage; explicit `acquire` / `release`; destroyed on `dispose` |
| `StagingRing` | N `MAP_READ` staging buffers; `copy(src, byteLength) -> Promise<ArrayBuffer>` overlapping map with the next submit |
| `PipelineCache` | `get(sourceId, overrides, layoutKey) -> GPUComputePipeline`, compiled once, `getCompilationInfo()` errors surfaced |
| `Kernel` | a pipeline + bind-group-layout + typed uniform block + `dispatch(pass, count | indirect)` helper using the planner |
| `Frontier` | vertex queue pair + length atomics + indirect-args buffer + bitset; `swap()`, `lengthAsync()` |
| `RadixSort` | persistent histogram / scratch buffers for u32 keys with optional values |
| `Scan` | persistent block-sum buffers for reduce-then-scan |
| `ForceAtlas2Simulation`, `FruchtermanReingoldSimulation` | `GpuLayoutSimulation` implementations (share `SimulationBase`) |
| `LayoutStepper` | the fire-and-forget bridge (section 7.18): request coalescing, in-flight tracking, readback copy, `settled` |

---------------------------------------------------------------------------

## 4. Memory and upload

### 4.1 GraphResidency

Two weak maps, both required (note 07 section 2 item 3):

```ts
type ResidentKey = ArrayBufferView;                       // the typed-array OBJECT (rowPtr, colIdx, a view array, a gpuView() array)
interface Resident { buffer: GPUBuffer; byteLength: number; version: number; kind: "arena" | "array" | "window"; refs: number }
class GraphResidency {
    private byArray = new WeakMap<ResidentKey, Resident>();
    private bySnapshot = new WeakMap<GraphSnapshot, Set<Resident>>();  // so release() can enumerate buffers whose CPU key was dropped
    core(snapshot): CoreBinding;            // rowPtr/colIdx/weights(+arcToEdge/edgeToArc on demand) as { buffer, offset, size } triples
    view(snapshot, name): ViewBinding;      // reverse() arrays, coo().src, edgeList().src/dst/weights, degreeOrder().perm, mate()
    column(snapshot, table, name): ColumnBinding;   // gpuView(name) with column.version invalidation
    scratch(snapshot, tag, byteLength): GPUBuffer;  // per-snapshot algorithm scratch (e.g. out-weight sums) that survives across calls
    release(snapshot): void;
}
```

Rules encoded:

1. Cache hit = same array object AND (for columns) same `column.version`. A `markDirty()`
   on a mutable column bumps `version` (note 07 section 1.7); an f64 column's
   `gpuView()` returns a NEW `Float32Array` after `markDirty()`, so the object key
   changes as well.
2. Identity permutations are never read: `flags.arcToEdgeIsIdentity` (and `!directed`
   for `reverse().fwdArc`) selects `USE_PERM = false` and binds `colIdx` / `rowPtr` as
   the dummy in the permutation slot (design 10.1; note 07 section 2 items 1-2).
3. `dropCaches()` on the snapshot orphans array keys; the `bySnapshot` set still holds
   the `Resident`, so `release(snapshot)` destroys it. A re-upload after `dropCaches()`
   creates a new `Resident` for the new array object (documented cost; the element calls
   `dropCaches()` only on the PREVIOUS snapshot, design 14.4).
4. A `Resident` is destroyed only by `release(snapshot)` or `ctx.dispose()`; never by GC.

### 4.2 Upload plan selection (pure function; unit-tested with faked caps)

```ts
export type UploadPlan =
    | { kind: "arena"; bytes: number; hotOnly: boolean; segments: Record<CoreArrayName, { offset: number; size: number } | null> }
    | { kind: "per-array"; arrays: CoreArrayName[] }
    | { kind: "windowed"; windows: { arcStart: number; arcEnd: number; rowStart: number; rowEnd: number }[] };

export function planCoreUpload(snapshot: GraphSnapshot, limits: { maxBufferSize: number; maxStorageBufferBindingSize: number }, need: { cold: boolean }): UploadPlan {
    const a = snapshot.arena;
    const bytes = a === null ? 0 : (need.cold ? a.byteLength : a.hotByteLength);
    if (a !== null && bytes <= limits.maxBufferSize && everySegmentFits(a, limits.maxStorageBufferBindingSize)) return { kind: "arena", ... };
    if (4 * snapshot.arcCount <= limits.maxStorageBufferBindingSize) return { kind: "per-array", ... };
    return { kind: "windowed", windows: planArcWindows(snapshot.rowPtr, limits.maxStorageBufferBindingSize / 4) };
}
```

| Path | When | Upload calls | Binding |
| --- | --- | --- | --- |
| arena (hot prefix) | `arena !== null`, `hotByteLength <= maxBufferSize`, every non-null segment `<= maxStorageBufferBindingSize` | one `createBuffer(hotByteLength)` + one `writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength))` | `{ buffer: gbuf, offset: seg.byteOffset - arena.byteOffset, size: seg.byteLength }` per segment (absolute-to-relative conversion, note 07 section 1.5) |
| arena (full) | a kernel needs `arcToEdge` / `edgeToArc` and the full `byteLength` fits | as above with `byteLength` | same |
| per array | `arena === null` (the NORMAL case for `fromCsr` on separate arrays and `transpose()`, note 07 section 2 item 5) or the arena does not fit | one buffer per array, `writeBuffer(buf, 0, snapshot.colIdx)` | whole buffer |
| windowed | a single array exceeds `maxStorageBufferBindingSize` (33,554,432 arcs at the 128 MiB default; 536M arcs at 2 GiB on the 4070 under Dawn) | windows over ARC ranges: `start = rowPtr[v0] - (rowPtr[v0] % 64)`, `colIdx.subarray(start, end)` (zero-copy, `%` not `& ~63`), each window its own buffer | kernel gets `arcBase = start` and `rowBase = v0` as uniforms; row loops clamp to the window; a row longer than a window is split across windows (design 10.6) |

Raised limits are requested at `create()` (`raiseLimits: true` default): `maxBufferSize`,
`maxStorageBufferBindingSize`, `maxComputeWorkgroupsPerDimension`,
`maxStorageBuffersPerShaderStage`, `maxComputeInvocationsPerWorkgroup`,
`maxComputeWorkgroupStorageSize` are set to `adapter.limits` values (spec: requests must
lie between default and adapter value). On lavapipe `maxStorageBufferBindingSize` stays
at 128 MiB (note 06 section 3.5), so the windowed path is exercised on the default CI
lane with a 40M-arc synthetic graph only in `node-limits` (memory) and otherwise by
FAKED limits in unit tests (planner functions take `limits` as data).

### 4.3 Views and columns

| Source | Key object | Notes |
| --- | --- | --- |
| `reverse()` (directed) | `reverse().rowPtr`, `.colIdx`, `.weights`, `.fwdArc` | four buffers (or one arena-like packed buffer created by the GPU package: `packViews` concatenates them 256-aligned into one `createBuffer` when total <= `maxBufferSize`, to save bind-group churn); undirected: the forward objects hit the cache, `fwdArc` untouched |
| `coo().src` | `coo().src` | `dst` aliases `colIdx` (cache hit) |
| `edgeList()` | `.src`, `.dst`, `.weights` | `arc` getter avoided unless `!arcToEdgeIsIdentity` |
| `outDegree()` etc. | the `U32` view | rarely uploaded (kernels read `rowPtr`) |
| `degreeOrder({ of }).perm` | the `U32` | `segmentOffsets` read on the CPU (design 10.1) |
| node/edge columns | `table.gpuView(name)` object + `column.version` | `E_GPU_INELIGIBLE` propagates; `u8` bound as padded words; `bool` as bit words; results attach back with `table.set(name, readback)` |
| position column (layout) | element-owned `positions.subarray(0, 3n)` | uploaded ONCE at `load()`, then GPU-authoritative; the readback writes into it; `column.markDirty()` per copy |

### 4.4 BufferPool

Size classes are powers of two from 4 KiB to 256 MiB plus "exact" above; keyed by
`(sizeClass, usageBits)`. `acquire(byteLength, usage, label)` returns a buffer of at
least `byteLength`; `release(buffer)` returns it to the pool; `trim()` destroys idle
buffers above a byte budget (default 256 MiB). The pool is used for per-call scratch
(partials, frontiers, sort scratch, staging) so an algorithm called every frame does
not `createBuffer` each time. Large allocations are wrapped in
`pushErrorScope("out-of-memory")`; a non-null result throws `E_OUT_OF_MEMORY` with the
requested size in `details` (note 05 section 4 note).

### 4.5 Readback (StagingRing)

- `StagingRing(ctx, byteLength, depth = 3)`: `depth` `MAP_READ | COPY_DST` buffers.
  `copy(encoder, src, srcOffset, size)` records a `copyBufferToBuffer` into the next
  free ring slot (`mapState === "unmapped"`), and after `submit` returns a promise
  that resolves to a fresh `ArrayBuffer` (`getMappedRange().slice(0)` then `unmap()`),
  or copies into a caller-supplied typed array via `dest.set(new Float32Array(range))`
  before `unmap()` (the mapped range is detached at unmap; design 10.7).
- Never `onSubmittedWorkDone` as a poll; `mapAsync` already waits (note 05 section 7.2).
- MEASURED costs (note 05 section 7.2): 4-byte round trip 0.04 ms Dawn-node / 0.10 ms
  Chromium; 1 MiB copy + map + slice 2.65 ms Chromium; +0.07 ms for 234 KiB on
  Dawn-node. ESTIMATE: 12 MB (1M-node positions) ~30 ms in Chromium, a few ms in Node.
  Hence the layout stepper batches iterations per readback (section 7.18).

### 4.6 Release lifecycle

```
DataManager.getSnapshot()  -> freeze -> emit snapshot-replaced { previous, next }
    listener: gpu.release(previous)        // destroys every Resident in bySnapshot(previous), removes byArray keys
              previous.dropCaches()        // element; harmless order either way
LayoutSimulation.load(next, positions)    // re-uploads core through residency (cache miss: new arrays)
ctx.dispose()                              // destroys everything, then device.destroy() when owned
```

`release(snapshot)` also releases per-snapshot `scratch()` buffers (e.g. the PageRank
out-weight sums) and any windowed buffers. Algorithms never keep buffers across calls
except through `scratch()`.

### 4.7 Bytes on the device

Snapshot residency (undirected weighted; design 15.1 numbers):

| Item | Bytes | 100k / 1M edges | 1M / 10M | 10M / 100M |
| --- | --- | --- | --- | --- |
| `rowPtr` | 4(n+1) | 0.4 MB | 4 MB | 40 MB |
| `colIdx` | 4A (A = 2E) | 8 MB | 80 MB | 800 MB |
| `weights` | 4A or 0 | 8 MB | 80 MB | 800 MB |
| hot prefix total | | 16.4 MB | 164 MB | 1.64 GB (windowed at defaults: 128 MiB per binding) |
| `arcToEdge` (cold, only for edge-column gathers) | 4A | 8 MB | 80 MB | 800 MB |
| `edgeToArc` (cold, per-edge writeback) | 4E | 4 MB | 40 MB | 400 MB |
| directed `reverse()` (pull kernels) | 4(n+1) + 12A | 12.4 MB | 124 MB | 1.24 GB |

Per-algorithm scratch (per node unless stated; the layout tables are in section 7.3):

| Algorithm | Bytes/node | Extra | 100k | 1M |
| --- | --- | --- | --- | --- |
| PageRank | 8 (two rank buffers) + 4 (out-weight sum) + 4 (personalization, optional) | partials 16 B per 256 nodes | 1.6 MB | 16 MB |
| WCC (Afforest) | 4 (comp) | histogram 4 KiB; remaining-vertex list 4n | 0.8 MB | 8 MB |
| BFS | 4 depth + 4 parent + 8 (two queues) + bitset 1/8 | indirect args 16 B | 1.6 MB | 16 MB |
| SSSP near-far | 4 dist + 4 pred + 12 (near/far queues) | histogram per subpartition | 2 MB | 20 MB |
| Betweenness (batch k sources) | k x (4 sigma + 4 depth) + 4 delta + 4 bc + queues | batch sized from `maxBufferSize` | k=64: 52 MB | k=8: 68 MB |
| Louvain | 4 cluster + 4 weight + hash region 8 x 2 x degree | coarse graph CSR per level | ~24 MB | ~240 MB |

Peak device memory for "FA2 at 1M nodes / 10M edges, grid tier, 3D" is ~164 MB (hot
prefix) + ~64 MB (simulation) + ~48 MB (pyramid) = ~280 MB: inside a 4070 SUPER's 12 GB
and inside the raised limits; at the 256 MiB default `maxBufferSize` the hot prefix
still fits as one buffer (164 MB) but each array is bound separately (colIdx 80 MB <
128 MiB). 10M / 100M needs windowing (section 4.2) and is a Node batch case.

---------------------------------------------------------------------------

## 5. Kernel infrastructure

### 5.1 PipelineCache

- Key: `sourceId` (module name + FNV-1a hash of the composed WGSL) + sorted `overrides`
  (`{ WG: 256, USE_PERM: 0, HAS_WEIGHTS: 1, DIM: 3, ... }`) + bind-group-layout key.
  Distinct override sets are distinct pipelines (WGSL 7.2.2; note 05 section 6 item 3),
  so the cache MUST include them.
- Compilation: `createShaderModule` inside `pushErrorScope("validation")`; on a popped
  error or any `getCompilationInfo().messages` of type `"error"`, throw `E_VALIDATION`
  with the message list and a line-numbered excerpt in `details` (the
  `gpu-upload.test.ts` `dispatch()` helper does this today, note 07 section 6).
- Explicit `GPUBindGroupLayout`s (never `layout: "auto"`) so one layout serves every
  variant of a kernel and bind groups can be reused across pipelines; layouts are cached
  by a string key of `(binding, type, readonly)` tuples.
- Warm-up: `ctx.warm(["fa2"])` compiles a family's pipelines eagerly (the first frame of
  a layout should not pay ~10-50 ms of shader compilation); `createComputePipelineAsync`
  is used so warm-up does not block the main thread.

### 5.2 DispatchPlanner

Pure functions of `caps.limits` (unit-tested with faked limits):

```ts
export function plan1D(count: number, wg: number, limits): { x: number; y: number; linear: boolean } {
    const groups = Math.ceil(count / wg);
    if (groups <= limits.maxComputeWorkgroupsPerDimension) return { x: groups, y: 1, linear: true };
    const x = limits.maxComputeWorkgroupsPerDimension;                   // 65,535
    return { x, y: Math.ceil(groups / x), linear: false };                // kernel: id = gid.x + gid.y * (x * WG)
}
```

| Rule | Detail |
| --- | --- |
| 1D limit | `count <= 65,535 x 256 = 16,776,960` at defaults (design 10.6; asserted by `gpu-upload.test.ts` line 189); the planner never uses 2^24 |
| 2D grid | above the limit: `dispatchWorkgroups(65535, ceil(groups / 65535))`; the prelude's `linearId(gid, numWorkgroupsX)` computes `gid.x + gid.y * numWorkgroupsX * WG` with `num_workgroups` builtin; every kernel bound-checks `id < count` from the uniform |
| grid-stride | kernels that keep per-invocation state across many items (segmented reductions, histograms) use `for (i = id; i < count; i += stride)` with `stride = numWorkgroups.x * numWorkgroups.y * WG`; the planner caps the grid at a multiple of the device's occupancy (`caps.software ? 64 : 4096` workgroups) |
| indirect | `dispatchWorkgroupsIndirect(args, offset)`: `args` is `INDIRECT | STORAGE | COPY_DST`, 16-byte aligned entries `[x, y, 1, pad]`; a 1-invocation `finalizeDispatch` kernel converts a device-side count into `(min(ceil(c/WG), 65535), ceil(ceil(c/WG) / 65535), 1)`; over-limit counts are clamped BEFORE they reach the API, so the "does nothing" spec behaviour (note 05 unverified item 7) is never relied on |
| WG | `override WG` = 256 (128 when `maxComputeInvocationsPerWorkgroup < 256`); tile sizes for workgroup memory are sized to 16 KiB (a 256 x vec4f tile is 4 KiB) unless `caps.limits.maxComputeWorkgroupStorageSize >= 32 KiB` selects a larger variant |
| arc indices | arc counts up to `0xFFFFFFFE` are passed as `u32` uniforms; JS never applies bitwise operators to them (`%`, `Math.floor`) |

### 5.3 Uniforms

`kernel/uniforms.ts` packs a declared layout into an `ArrayBuffer` with explicit
16-byte struct alignment (no reliance on `uniform_buffer_standard_layout`, absent in
Chromium 139; note 05 section 2.4):

```ts
const fa2Params = uniformLayout({ n: "u32", arcCount: "u32", dim: "u32", iteration: "u32",        // 16 B
                                  scalingRatio: "f32", gravity: "f32", jitterTolerance: "f32", flags: "u32",  // 16 B
                                  seedRadius: "f32", settleThreshold: "f32", arcBase: "u32", rowBase: "u32" }); // 16 B
```

Per-dispatch params use a single 64 KiB uniform buffer with dynamic offsets at a
256-byte stride (`minUniformBufferOffsetAlignment` is 256 in Chromium, 64 / 16 in
Dawn-node; 256 satisfies all), written once per submit with `writeBuffer` for all k
iterations (iteration-varying scalars such as `iteration` live in the device-side STATE
block instead, so most kernels bind the same uniform block for every iteration).

Bind group conventions (design 10, note 04 section 1 binding budget): group 0 = graph
(immutable per snapshot: rowPtr, colIdx, weights, perm slots), group 1 = algorithm
state (ping-pong buffers, frontiers, partials), group 2 = params (uniform, dynamic
offset). Every kernel is designed for the DEFAULT 8 storage buffers per stage; cold /
optional arrays go to group 1 and are bound to a 16-byte dummy buffer when absent.

### 5.4 Command recording pattern

An iterative algorithm records `k` iterations into ONE command encoder (one compute
pass per iteration or one pass with implicit barriers between dispatches) and submits
once; readback of a convergence scalar happens every `k` iterations (PageRank k = 8,
BFS k = 16 levels with indirect dispatch, FA2 k = `iterationsPerStep`). This is the
single most important WebGPU-specific rule (note 04 section 15 item 4; note 05 section
7.2): per-level `mapAsync` makes BFS on a 19,000-level road graph slower than the CPU.

### 5.5 Timestamp queries

When `caps.features.has("timestamp-query")`, `Kernel.dispatch` can wrap dispatches in
`timestampWrites` of a 2-entry query set per kernel and `ctx.profile()` resolves them
into `{ kernel, us }[]` on the next readback. Chromium quantises to 100 us (note 05
section 3.1), so the profiler is a Node / Dawn tool; the benchmark harness records
per-kernel breakdowns from it. Never used for runtime decisions except `calibrate()`.

### 5.6 Error handling

| Event | Handling |
| --- | --- |
| Validation error during creation (`createShaderModule`, `createComputePipeline`, `createBindGroup`, `createBuffer`) | error scopes around each; popped error -> throw `E_VALIDATION` with label and message |
| Validation error during a submitted pass (bad offset, OOB indirect) | `device.addEventListener("uncapturederror")` in `GpuContext`: records the first error, rejects the pending readback promises with `E_VALIDATION`; tests fail on any uncaptured error (note 05 section 7.3) |
| Out of memory | `pushErrorScope("out-of-memory")` around large `createBuffer`; throw `E_OUT_OF_MEMORY { requested }`; the residency planner may retry with a smaller window only when the caller passed `allowWindowing` (never silently) |
| Device lost | `ctx.isLost = true`, everything rejects with `E_DEVICE_LOST { reason, message }`; no recreation (section 2.5) |
| Unsupported input | `E_UNSUPPORTED` (e.g. `sssp` on `!flags.nonNegativeWeights`, `mate()` on directed), `E_EMPTY` handled as a valid zero-work result (empty arrays), `E_TOO_LARGE` when a plan cannot fit even windowed |
| Labels | every buffer, pipeline, bind group and pass carries `label` (`"fa2/positions"`, `"pagerank/rankA"`) so Dawn's messages name the object |

---------------------------------------------------------------------------

## 6. Primitives

Each primitive has: a TypeScript interface, its WGSL strategy, complexity, and the CPU
reference used in tests (`test/helpers/oracle.ts`, plain loops over typed arrays; the
oracles are also what the algorithm differential tests use until W1 switches them to
`indexed.*`). All primitives work on `u32` / `i32` / `f32` storage buffers, respect the
8-binding budget and use only `u32` / `i32` atomics.

### 6.1 Reduce (`primitives/reduce.ts`)

- Interface: `reduce(pass, src, count, op: "sum" | "min" | "max" | "sumAbs" | "sumSq", dtype: "f32" | "u32" | "i32") -> partials buffer`; `reduceFinal(pass, partials, groups) -> 16-byte result`.
- Strategy: pass 1 grid-stride accumulation into workgroup memory (256 lanes), tree
  reduce (or `subgroupAdd` + one cross-subgroup step when available), one partial per
  workgroup; pass 2 a single workgroup reduces `groups` partials. Two dispatches, no
  atomics, deterministic order. Multi-channel variants (`vec4` partials) serve FA2's
  swing + traction + movement + centroid in one pass.
- Complexity O(count), 2 dispatches.
- Oracle: sequential f64 sum; tolerance `1e-6 * count` relative for f32.

### 6.2 Exclusive scan (`primitives/scan.ts`)

- Interface: `scan(pass, src, dst, count) -> void` (u32), persistent block-sum buffers
  sized for `count` allocated on first use (`Scan` class).
- Strategy: reduce-then-scan, 3 dispatches: block sums (256 items per workgroup via
  workgroup-memory Blelloch scan or `subgroupExclusiveAdd`), scan of block sums
  (recursive when `groups > 256`, at most 3 levels for 2^24 items), add-back. Decoupled
  look-back is rejected for v1 (WGSL atomics are relaxed; note 04 section 2.1).
- Complexity O(count), 3-5 dispatches.
- Oracle: sequential prefix sum.

### 6.3 Segmented reduce over `rowPtr` (`primitives/segmented-reduce.ts`)

- Interface: `segmentedReduce(pass, rows: { rowPtr, colIdx?, values, perm?, segmentOffsets }, mapExpr: WGSL, op) -> per-row f32/u32` with the `override USE_PERM` and `HAS_WEIGHTS` pattern.
- Strategy (the design-10 tiered gather): three dispatches from
  `degreeOrder().segmentOffsets` (note 07 section 1.6: `[0, hiEnd, midEnd, lowEnd, n]`,
  tiers 1024 / 32 / 1):
  - hi (`deg >= 1024`): one WORKGROUP per row, 256 lanes stride the row, workgroup tree reduce;
  - mid (`32 <= deg < 1024`): one SUBGROUP per row when `subgroups` exists (`subgroup_size` lanes stride the row, `subgroupAdd`), else 32 lanes of a workgroup with a workgroup-memory reduce;
  - low (`1 <= deg < 32`): one THREAD per row, serial loop;
  - degree-0 rows (`[lowEnd, n)`) are written as the identity element by the low kernel.
  The row index is `perm[segmentStart + i]`. For v1 of the FA2 attraction the single
  thread-per-row kernel is used (element graphs are small); the tiers arrive in P5.
- Complexity O(A), 3 dispatches.
- Oracle: per-row loop. This primitive is the pull SpMV of section 8 and the attraction gather of section 7.5.

### 6.4 Stream compaction and dedupe (`primitives/compact.ts`)

- Interface: `compact(pass, flags | predicateExpr, src, dst, count) -> lengthAtomic`; `dedupeOwnership(pass, queue, owner: U32(n))`.
- Strategy: flag + scan + scatter (3-5 dispatches, deterministic order) for bulk
  compaction; a `compactAtomic` variant (workgroup-local scan, one `atomicAdd` per
  workgroup on the global length to claim a base offset, Gunrock `block_mapped` style)
  for frontier construction where output order does not matter. Dedupe uses the
  Davidson "ownership" trick: each queue entry writes its queue index into `owner[v]`,
  reads it back, keeps the entry only if it owns `v` (one store + one load, no atomics;
  note 04 section 2.3).
- Complexity O(count).
- Oracle: filter + Set.

### 6.5 Histogram / count-by-key (`primitives/histogram.ts`)

- Interface: `histogram(pass, keys, count, bins, out: atomic u32[bins])`; `histogramWorkgroup` variant with per-workgroup privatised bins in workgroup memory (bins <= 4096) flushed with one atomicAdd per bin.
- Strategy: `atomicAdd(&out[key], 1u)`; privatised when bins are small (radix digits, degree buckets, SSSP subpartitions); global when bins are large (grid cells).
- Complexity O(count).
- Oracle: counting loop.

### 6.6 Radix sort (`primitives/radix-sort.ts`)

- Interface: `RadixSort.sort(pass, keys, values?, count, bits = 32)`; stable, LSD, 8 bits per pass (4 passes for 32-bit keys, fewer when `bits` is smaller: grid cell ids of 2^18 cells need 3 passes).
- Strategy (GraphWaGu's WGSL port of the Fuchsia sort validates the shape; note 03
  section 2.2, MIT): per pass histogram (privatised 256 bins per workgroup) -> exclusive
  scan of the 256 x groups digit table -> stable scatter (each workgroup ranks its
  items via a workgroup-local scan per digit). 3 dispatches per pass, 12 for 32 bits.
  Onesweep is rejected for the same reason as decoupled look-back.
- Complexity O(passes x count).
- Oracle: `Array.prototype.sort` on (key, index) pairs; stability asserted.

### 6.7 Bitset (`primitives/bitset.ts`)

- Interface: `bitsetSet(pass, indices, count, words)` (`atomicOr`), `bitsetClear`, `bitsetFromMask(NodeMask)` (upload), `bitsetToIndices` (compaction).
- Strategy: `array<atomic<u32>>` for concurrent set; bulk set without atomics when the
  frontier is >= 40% of n (cuGraph's rule, note 04 section 2.8).
- Oracle: `maskSet` / `maskTest` from the format (`src/util/mask.ts`).

### 6.8 Frontier (`primitives/frontier.ts`)

- Interface: `class Frontier { current, next: GPUBuffer(4n); length: atomic u32 x 2; indirect: GPUBuffer(16 x 2); bitset; swap(); finalizeDispatch(pass) }`.
- Strategy: two n-slot vertex queues (ping-pong) plus an edge-frontier buffer of
  `min(A, 16M)` entries for the two-phase expansion; `finalizeDispatch` (1 invocation)
  turns the device-side length into indirect args for the next level (section 5.2). The
  host reads `length` back only every k levels (or on a timestamp budget).
- Oracle: CPU BFS queues.

### 6.9 Advance (`primitives/advance.ts`)

- Interface: `advance(pass, frontier, graph, op: { visit: WGSL "fn visit(u: u32, v: u32, a: u32) -> bool" }, mode: "expand-contract" | "two-phase")`.
- Strategy: Gunrock `block_mapped` (note 04 section 2.5): each workgroup loads 256
  frontier vertices, scans their degrees in workgroup memory, then every lane loops
  `for (i = lid; i < aggregate; i += WG)` and `upper_bound`s the scanned degrees to
  find its source vertex and arc; calls `visit`, which returns whether `v` joins the next
  frontier (compaction via `compactAtomic`). Rows with degree >= 1024 (hi tier of the
  frontier's vertices, found by a per-frontier partition step) go to a workgroup-per-row
  kernel; a subgroup-per-row tier only when `subgroups` exists. Fused expand-contract
  kernel for frontiers smaller than `4 x WG x occupancy` (Merrill's fleeting iterations).
- Complexity O(sum of frontier degrees) per level, balanced within a workgroup.
- Oracle: CPU level-synchronous BFS.

### 6.10 Pull SpMV (`primitives/spmv.ts`)

- Interface: `spmvPull(pass, rev: AdjacencyBinding, x, y, options: { alpha, beta, normaliser?, personalization? })`.
- Strategy: `segmentedReduce` with `mapExpr = w(a) * x[colIdx[a]] / norm[colIdx[a]]`, tiered by IN-degree (`degreeOrder({ of: "reverse" })`); no atomics; f32 with a Kahan accumulator in the hi-tier loop.
- Oracle: CPU f64 SpMV.

### 6.11 COO -> CSR on device (`primitives/coo-to-csr.ts`)

- Interface: `cooToCsr(pass, src, dst, weights?, count, n) -> { rowPtr, colIdx, weights }` (rows NOT sorted by target unless a `sortRows` radix pass is requested).
- Strategy: histogram of `src` -> scan -> scatter with per-row atomic cursor (Gunrock `from_coo`, note 04 section 2.7). Used only for Louvain contraction and residual graphs; the initial graph is CSR already.
- Oracle: the format's `fromCsr` + `equalsTopology` after readback.

### 6.12 Bounding box (`primitives/bbox.ts`)

- Interface: `bbox(pass, positions, n, dim) -> { min: vec3, max: vec3 }` in a 32-byte state block.
- Strategy: workgroup min/max reduce, then `atomicMin` / `atomicMax` on `i32` fixed
  point (positions x 1024, clamped to +/- 2^20) as GraphWaGu does (note 04 section 1
  table), or two-level reduce for exact f32 (default: two-level reduce, fused into the
  layout integrate kernel's partials).
- Oracle: loop.

### 6.13 Summary and order of implementation

| Primitive | Needed first by | Dispatches | Atomics | Phase |
| --- | --- | --- | --- | --- |
| reduce (multi-channel) | FA2 swing/traction/centroid/movement | 2 | none | P3 |
| segmented reduce (thread tier) | FA2 attraction | 1 | none | P3 |
| segmented reduce (tiers) | FA2 hubs, PageRank | 3 | none | P5 |
| bbox | grid tier | fused | i32 min/max optional | P5 |
| histogram, scan, radix sort | grid tier (cell sort) | 3 / 3-5 / 12 | u32 add | P5 |
| pull SpMV | PageRank | 3 | none | P7 |
| compaction + dedupe, bitset, frontier, advance | WCC remaining list, BFS | 3-5 | u32 add/or/CAS | P8-P9 |
| COO -> CSR | Louvain | 4 | u32 add | P11 |

---------------------------------------------------------------------------

## 7. Force-directed layouts -- FIRST DELIVERABLE

### 7.1 Contract and what "drop-in" means

The GPU layouts implement design 14.3's `LayoutSimulation` over the element's stride-3
scene-unit `Float32Array` (note 01 section 5): `load(snapshot, positions)` where
`snapshot` is the element's cached UNDIRECTED snapshot (`dm.undirected(s).snapshot`,
rows hold both arcs, `outDegree()` is the degree with a self-loop counted once);
`step(iterations?)` is async; `settled` must become true in bounded time; `setFixed`
takes the `NodeMask` bitmap (`ceil(n/32)` words LSB-first, `src/util/mask.ts`);
`setPosition` is a 12-byte `writeBuffer`; `dispose()` frees everything. Drop-in
requirements in order of visibility (note 01 section 5.1): same options object as the
L1 CPU FA2, same forces and speed controller, compatible settlement, pins and drag
honoured every step, topology change via `load`, `dim === 2` leaves z untouched, seeded
starts identical to the CPU path, `release(snapshot)` on `snapshot-replaced`.

### 7.2 Reference formulas and the override constants that pin them

The CPU port deviates from NetworkX / the FA2 paper in three places (note 01 section
2.1.9). This plan adopts the PUBLISHED definitions (paper CC-BY, Gephi `ForceFactory`,
cuGraph `fa2_kernels.cuh`; note 03 sections 4.3 and 5) as the parity target for the L1
rewrite and the GPU kernel, and keeps the port's variants buildable behind overrides so
an A/B story can be shown to the owner before the Chromatic re-baseline (decision D9,
risk R1):

| Quantity | Adopted (paper / Gephi / cuGraph) | Port's variant (kept behind an override) | Override |
| --- | --- | --- | --- |
| repulsion on i from j | `F = kr * m_i * m_j / d` along `(p_i - p_j)/d`, i.e. component `diff * kr * m_i * m_j / d^2`; `d^2 = max(dot(diff, diff), 1e-4)` | magnitude `kr m_i m_j / d^2` (component `diff * kr m_i m_j / d^3`) | `REPULSION_LAW: u32` (0 = paper, 1 = port) |
| attraction (linear) | `F_i += (p_j - p_i) * w_a` per arc `a` in row i | same | -- |
| attraction (linlog) | `F_i += (p_j - p_i) * w_a * log(1 + d) / d` | same | `LINLOG: bool` |
| distributed action | `F_i /= m_i` after the row sum (Gephi outbound attraction distribution) | same | `DISTRIBUTED: bool` |
| gravity | centroid-relative as the port: `q = p_i - c`; regular `-g * m_i * q / |q|` when `|q| > 0.01`; strong `-g * m_i * q` | same (Gephi uses the origin; `GRAVITY_CENTER` override 0 = centroid, 1 = origin) | `STRONG_GRAVITY: bool`, `GRAVITY_CENTER: u32` |
| swing / traction per node | `swing_i = m_i * |F_i(t) - F_i(t-1)|`, `traction_i = 0.5 * m_i * |F_i(t) + F_i(t-1)|` (Gephi, cuGraph `compute_local_speed`) | `swing_i = m_i * |F_i|`, `traction_i = 0.5 m_i |2 p_i + F_i|`, reset each iteration | `SWING_MODE: u32` (0 = paper, 1 = port) |
| global speed | `estimateFactor` exactly as the port / NetworkX (note 01 section 2.1.3 item 8; a port of Gephi lines 296-328 per note 03 section 4.3): `optJitter = 0.05 sqrt(n)`, `minJitter = sqrt(optJitter)`, `maxJitter = 10`, `jitter = jitterTolerance * max(minJitter, min(maxJitter, optJitter * traction / n^2))`; if `swing / traction > 2`: `speedEfficiency = max(0.05, 0.5 * speedEfficiency)`, `jitter = max(jitter, jitterTolerance)`; `targetSpeed = swing == 0 ? inf : jitter * speedEfficiency * traction / swing`; if `swing > jitter * traction`: `speedEfficiency = max(0.05, 0.7 speedEfficiency)` else if `speed < 1000`: `speedEfficiency *= 1.3`; `speed += min(targetSpeed - speed, 0.5 * speed)` | identical | -- |
| local speed / apply | `factor = speed / (1 + sqrt(speed * swing_i))`; `p_i += F_i * factor` | identical; `adjustSizes`: `0.1 * speed` and cap 10 (deferred) | `ADJUST_SIZES: bool` (deferred) |
| size correction (deferred) | `d' = d - size_i - size_j` (symmetric) | `d - (size_i - size_j)` (sign-suspect) | `ADJUST_SIZES` |
| edge weight influence | `w_a` directly (delta = 1, the port) | -- | (`edgeWeightInfluence` option reserved, default 1) |
| distance floor / coincident nodes | `d^2 >= 1e-4`; when `d^2 < 1e-8` (coincident) add a deterministic kick `lowbias32(i * 0x9E3779B9 ^ j) -> unit vector * 1e-2` | port: `max(d, 0.01)` | -- |

`SWING_MODE = 0` needs an `oldForce` buffer (12 B/node); `SWING_MODE = 1` does not.
All overrides are pipeline-cache keys; the option object exposes only the adopted
semantics plus `compat: "port"` to select the port variants (for the A/B story and for
migration tests), never in the element UI.

### 7.3 Buffers (FA2)

| Buffer | Bytes | Source / lifetime | Notes |
| --- | --- | --- | --- |
| `graph/rowPtr`, `graph/colIdx`, `graph/weights` | 4(n+1), 4A, 4A or 0 | `GraphResidency.core(snapshot)` (arena hot prefix or per array); shared with algorithms | `HAS_WEIGHTS` override; `colIdx` bound as the dummy in the weights slot when null |
| `sim/positions` | 12n `array<f32>` | uploaded at `load()` from the element array (scene units, NaN rows seeded); GPU-authoritative after | `3*i + k` indexing; z untouched when `DIM == 2` |
| `sim/force` | 12n | zeroed at load; rewritten every iteration | repulsion writes, attraction+gravity accumulate |
| `sim/oldForce` | 12n (0 when `SWING_MODE = 1`) | previous iteration's force | integrate copies force -> oldForce |
| `sim/mass` | 4n | CPU: `outDegree()[i] + 1` or `nodeMass` (Float32Array / column via `gpuView` / Record) | uploaded at load and on option change |
| `sim/size` | 4n | only with `ADJUST_SIZES` (deferred) | |
| `sim/fixed` | 4 ceil(n/32) | `setFixed(mask)` -> `writeBuffer` of the whole bitmap | integrate skips set bits (they still exert forces) |
| `sim/held` | 4 ceil(n/32) | set by `setPosition(i)`, cleared by the next `setFixed` | dragged rows; readback skips them |
| `sim/partials` | 64 B per 256 nodes (`vec4` swing/traction/movement/mass-sum, `vec4` centroid sum, `vec4` bbox min, `vec4` bbox max) | one workgroup partial per 256 nodes | consumed by the finalize / adaptSpeed kernel |
| `sim/state` | 256 B storage block | `{ iteration: u32, settled: u32, speed: f32, speedEfficiency: f32, swing: f32, traction: f32, movement: f32, centroid: vec3, bboxMin: vec3, bboxMax: vec3, movementRing: array<f32, 16>, ringHead: u32, tier: u32 }` | read back with every position readback (one extra 256-byte copy) |
| `sim/params` | 64 B uniform (section 5.3) | per submit | constants and flags |
| grid tier (section 7.7) | 16 B/node + 8 B/cell + pyramid | allocated when `n > exactMaxNodes` | |

Exact tier: 12 + 12 + 12 + 4 = 40 B/node plus 1/4 B masks (+4 with sizes): 4 MB at
100k, 40 MB at 1M. Grid tier: +16 B/node + cells (section 7.7): 56 B/node plus the fixed
pyramid (~7 MB at 512^2 in 2D, ~48 MB at 128^3 in 3D). Note 03 section 8.2 gives the
same order (exact 40-48, grid 56-64, GraphWaGu tree ~110, Burtscher tree 80-170).

### 7.4 Per-iteration kernel sequence (exact tier)

Five dispatches per iteration, all recorded into one command buffer for `k`
iterations; no host round trip inside the batch (decision D7). Ordering of forces
follows cuGraph (all forces, then integrate once; note 03 section 1.6 last bullet).

| # | Kernel | Workgroups | Reads | Writes | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `fa2Repulsion` (exact tiled, 7.6) or `fa2RepulsionGrid` (7.7) | `ceil(n/256)` | positions, mass | force (=) | 80-95% of the time (note 03 section 0) |
| 2 | `fa2AttractGravity` | `ceil(n/256)` (P5: three tier dispatches) | rowPtr, colIdx, weights, positions, mass, force, oldForce, state.centroid | force (+=), partials[swing, traction] | CSR gather; gravity uses the centroid from the previous finalize |
| 3 | `fa2AdaptSpeed` | 1 (256 lanes) | partials | state.swing, state.traction, state.speed, state.speedEfficiency | reduces `ceil(n/256)` partials with a grid-stride loop, then `estimateFactor` in lane 0 |
| 4 | `fa2Integrate` | `ceil(n/256)` | force, mass, fixed, held, state.speed, state.swingPerNode? (recomputed inline) | positions, oldForce, partials[movement, centroid, bbox] | `factor = speed / (1 + sqrt(speed * swing_i))`; skips fixed/held; `DIM == 2` leaves z |
| 5 | `fa2Finalize` | 1 | partials | state.movement, state.centroid, state.bbox, state.iteration, state.movementRing, state.settled | settlement test (7.12) |

Kernel 2 recomputes `swing_i` from `force` and `oldForce` and kernel 4 recomputes it
again from the same buffers (12 flops) instead of storing a per-node swing array -- one
fewer buffer, no extra pass. Kernel 3 and 5 are one-workgroup kernels; their latency
(~5-10 us each on a discrete GPU) is negligible against kernel 1.

### 7.5 Attraction over CSR rows (kernel 2)

```wgsl
override HAS_WEIGHTS: bool; override LINLOG: bool; override DISTRIBUTED: bool; override DIM: u32;
override STRONG_GRAVITY: bool; override GRAVITY_CENTER: u32; override SWING_MODE: u32; override WG: u32 = 256;
@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;      // colIdx bound here when !HAS_WEIGHTS (never read)
@group(1) @binding(0) var<storage, read> pos: array<f32>;
@group(1) @binding(1) var<storage, read> mass: array<f32>;
@group(1) @binding(2) var<storage, read_write> force: array<f32>;
@group(1) @binding(3) var<storage, read> oldForce: array<f32>;
@group(1) @binding(4) var<storage, read_write> partials: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> state: State;
@group(2) @binding(0) var<uniform> p: Params;
var<workgroup> red: array<vec2<f32>, WG>;
@compute @workgroup_size(WG) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x; var st = vec2<f32>(0.0);
    if (i < p.n) {
        let pi = readPos(i); var acc = vec3<f32>(0.0);
        for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a++) {              // both arcs present on the undirected snapshot: no atomics
            let j = colIdx[a]; let w = select(1.0, weights[a], HAS_WEIGHTS);
            var d = readPos(j) - pi;                                         // toward j
            if (LINLOG) { let l = length(d); d = d * (log(1.0 + l) / max(l, 1e-4)); }
            acc += d * w;
        }
        if (DISTRIBUTED) { acc /= mass[i]; }
        acc += gravity(pi, mass[i]);                                          // centroid or origin per GRAVITY_CENTER
        let f = readForce(i) + acc;                                           // repulsion from kernel 1
        writeForce(i, f);
        let fo = readOld(i);
        st = select(vec2(mass[i] * length(f), 0.5 * mass[i] * length(2.0 * pi + f)),   // SWING_MODE 1 (port)
                    vec2(mass[i] * length(f - fo), 0.5 * mass[i] * length(f + fo)),      // SWING_MODE 0 (paper)
                    SWING_MODE == 0u);
    }
    red[lid.x] = st; workgroupBarrier();                                      // tree reduce to partials[wid.x].xy
    ...
}
```

`readPos(i)` returns `vec3(pos[3i], pos[3i+1], select(0.0, pos[3i+2], DIM == 3u))`.
Load balance: a hub row of 10^5 arcs on one lane stalls its workgroup. P3 ships the
thread-per-row kernel (element graphs are ~10^2-10^4 nodes today, note 01 section 6);
P5 splits it into the three tier dispatches of section 6.3 over
`degreeOrder().perm` with `segmentOffsets` read on the CPU (design 10.1), the hub tier
being one workgroup per row with the reduction in workgroup memory. Parallel arcs sum
(documented behaviour change of design 14.3). Weights: `snapshot.weights` when `weight
=== true`; a named edge column is expanded by the element with `expandEdges` (design
14.4) and uploaded as a per-arc `F32` keyed on its array object; `null` -> `HAS_WEIGHTS
= false`.

### 7.6 Exact tiled repulsion (kernel 1, `n <= exactMaxNodes`)

```wgsl
override REPULSION_LAW: u32; override DIM: u32; override WG: u32 = 256;
var<workgroup> tile: array<vec4<f32>, WG>;                                    // xyz + mass: 4 KiB at WG 256 (16 KiB default limit)
@compute @workgroup_size(WG) fn main(...) {
    let i = gid.x; var me = vec4<f32>(0.0); if (i < p.n) { me = vec4(readPos(i), mass[i]); }
    var acc = vec3<f32>(0.0);
    for (var base = 0u; base < p.n; base += WG) {
        let j = base + lid.x;
        tile[lid.x] = select(vec4(0.0), vec4(readPos(j), mass[j]), j < p.n);
        workgroupBarrier();
        for (var t = 0u; t < WG; t++) {                                       // unrolled x4 by the compiler; skip self by index
            let o = tile[t]; let jj = base + t;
            if (o.w > 0.0 && jj != i) {
                let d = me.xyz - o.xyz; var d2 = dot(d, d);
                if (d2 < 1e-8) { acc += kick(i, jj); continue; }              // coincident: deterministic unit kick
                d2 = max(d2, 1e-4);
                let f = p.scalingRatio * me.w * o.w;
                acc += select(d * (f / d2), d * (f / (d2 * sqrt(d2))), REPULSION_LAW == 1u);
            }
        }
        workgroupBarrier();
    }
    if (i < p.n) { writeForce(i, acc); }
}
```

Cost: `n^2` pair evaluations per iteration, zero memory beyond the tile. MEASURED: the
probe `tmp/webgpu-plan/probe/dawn-perf.mjs` (same tile shape, 3-component positions,
no mass) runs n = 20,000 in 1.11 ms per iteration on the RTX 4070 SUPER under Dawn-node
and 388 ms on lavapipe (note 05 section 2.5): 4 x 10^8 pairs / 1.11 ms = 3.6 x 10^11
pairs/s. The FA2 body has one more multiply and a `select`, so the plan budgets 1.3x
that (ESTIMATE). The mass in the tile costs nothing (the tile is `vec4` either way).
2D and 3D use the same kernel: `readPos` zeroes z for `DIM == 2`.

### 7.7 Approximate repulsion for large n: cell-sorted grid pyramid (P3M as compute)

Chosen over a Barnes-Hut tree for the reasons of note 03 section 8.3: the build is a
counting/radix sort plus scans and segmented reductions (primitives the package needs
anyway), no locks, no spin-waits, no float atomics, a FIXED traversal loop (no
per-thread stack, no divergence beyond the near field), a production reference with
documented failure modes (cosmos.gl, MIT; `docs/many-body-force/README.md`), and a
direct 3D extension (27 / 216 loops) that neither cosmos's grid nor GraphWaGu's tree
has. WebGPU forbids the Burtscher-Pingali build (CAS-locked insertion, cross-workgroup
spin-wait summarisation, warp-vote traversal; note 03 section 4.5, note 04 section 1).

Geometry (per iteration, recomputed from the bounding box written by the previous
finalize):

| Item | 2D | 3D |
| --- | --- | --- |
| finest grid per axis `G` | `clamp(nextPow2(2 sqrt(n)), 8, 512)` (cosmos) | `clamp(nextPow2(2 cbrt(n)), 8, gridMax3D = 128)` |
| cell size | `max(extent.x, extent.y) / G` (square cells; bbox padded by 1 cell) | `maxExtent / G` (cubic) |
| levels | `G = 512 -> 4^2 .. 512^2`: 8 levels | `128^3`: 6 levels (`4^3 .. 128^3`) |
| cells (finest) | 262,144 at cap | 2,097,152 at cap |
| per-cell record | `vec4<f32>` (`sum m*x, sum m*y, sum m*z, sum m`) + `u32` count | same |
| pyramid bytes | 20 B x 262k x 1.33 = ~7 MB | 20 B x 2.1M x 1.14 = ~48 MB |
| far-field evaluations per node | coarsest 16 - 9 = 7, then 7 levels x (36 - 9) = 189 -> 196 | coarsest 64 - 27 = 37, then 5 x (216 - 27) = 945 -> 982 |
| near-field cells | 3 x 3 = 9 | 3 x 3 x 3 = 27 |

Build kernels (per iteration; dispatch counts in parentheses):

1. `gridCellKey` (1): `key[i] = linearise(floor((p_i - bboxMin) / cellSize))`, `idx[i] = i`.
2. `RadixSort.sort(key, idx, n, bits = log2(cells))` (3 per 8-bit digit: 3 digits at
   2^18 cells, 3 at 2^21): stable -> deterministic order inside a cell (decision D12).
   Alternative measured in P5: atomic-cursor counting sort (histogram + scan + scatter,
   5 dispatches) which is nondeterministic within a cell; kept behind
   `deterministic: false` only if it is measurably faster.
3. `gridRanges` (1): `cellStart[c]` / `cellCount[c]` from adjacent key compares in the
   sorted order (a lane whose key differs from its left neighbour writes the start).
4. `gridCentroids` (1): segmented reduce over `[cellStart, cellStart + count)` writing
   the `vec4` record and count of the finest level (one workgroup per 256 sorted
   entries with a segmented scan; hub cells longer than 256 are combined across
   workgroups with `atomicAdd` on `i32` FIXED-POINT copies of the sums (cell-relative
   coordinates x 2^16, up to 2^15 points per cell fit 32 bits; note 03 section 8.1 (c))
   -- exact enough for centroids and deterministic because integer addition commutes).
5. `gridDownsample` (levels - 1): each parent sums its 4 (8) children; one dispatch per
   level, no atomics.

Force kernel (`fa2RepulsionGrid`, 1 dispatch, replaces kernel 1):

```
per node i (thread): c = cell of i at the finest level
  far field:
    for every cell of the coarsest level not in the 3x3 (3x3x3) block around i's ancestor: F += rep(centroid, mass)
    for level L = coarsest+1 .. finest:
        for the 6x6 (6x6x6) child block of the parent's 3x3 (3x3x3), minus this level's own 3x3 (3x3x3): F += rep(centroid, mass)
  near field: for each of the 9 (27) finest cells around c:
        m = count[cell]; s = min(m, NEAR_MAX)            // NEAR_MAX = 64 (override)
        off = select(0, lowbias32(seed ^ iteration ^ cell) % m, m > NEAR_MAX)
        for t in 0..s: j = sorted[cellStart + (off + t) % m]; if (j != i) F += rep_exact(j) * (f32(m) / f32(s))   // Horvitz-Thompson weight, exact when m <= NEAR_MAX
```

`rep(centroid, M)` is `diff * kr * m_i * M / d^2` with `diff = p_i - centroid / M`
(mass-weighted centroid, Gephi `Region` semantics; note 03 section 5). Cells with
`count == 0` are skipped by the `M > 0` test. The far-field loop bounds are compile-time
constants per level count, so the kernel is one `override LEVELS` variant per grid
size. Hub cells are the known risk (cosmos's 163-node country graph shimmered, note 03
section 1.3): the cap plus the per-iteration hashed offset makes the estimate unbiased
and the cost bounded (`9 x 64` or `27 x 64` exact pairs per node worst case); the
`stats()` surface reports `maxCellOccupancy` so the element can raise `NEAR_MAX` or
lower `gridMax` when a graph is pathological. A per-iteration clamp of the near-field
displacement to `2 * cellSize` (cosmos) is applied in the integrate kernel when the
tier is `grid`.

Bytes per node in the grid tier: `key` 4 + `idx` 4 + sort ping-pong 8 = 16 B, plus
8 B per finest cell (start, count) and the pyramid above.

3D at 1M nodes: 48 MB pyramid + 16 MB sort scratch is acceptable on discrete GPUs; the
option `gridMax3D` (default 128) can be lowered to 64 (6 MB) on integrated GPUs at the
cost of near-field occupancy (8x more nodes per cell). This is the one place where a
tree would use less memory; note 03 section 8.3 keeps the Hilbert cluster tree
(GraphWaGu 2025 build: Hilbert code -> radix sort -> `log_4 n` merge dispatches,
private-stack DFS; note 03 section 2) as the documented second experiment for
clumpy distributions, sharing the same radix sort. It is NOT scheduled unless P5's
measurements on the hub-heavy fixtures show the grid losing.

Dispatch count per iteration in the grid tier: 1 (keys) + 9 (sort) + 1 (ranges) + 1
(centroids) + 7 (downsample, 2D) + 1 (force) + 4 (attract, adapt, integrate, finalize)
= 24 in 2D, ~23 in 3D; all in one command buffer. At ~5-10 us of GPU-side overhead per
tiny dispatch this is < 0.3 ms of fixed cost.

### 7.8 Crossover by n

```ts
repulsion: "exact" | "grid" | "auto"      // default "auto"
exactMaxNodes: number                     // default 16,384 (auto picks exact when n <= exactMaxNodes)
```

Basis: Burtscher measured O(n^2) fastest below ~10k bodies on a 2009 GPU; GraphWaGu 2022
found their O(n^2) FR best below ~5k on an RTX 2060; cosmos switches at 4,096 because
each WebGL peel pass costs ~0.1 ms (note 03 section 8.2, PUBLISHED). On the 4070 SUPER
the MEASURED 3.6 x 10^11 pairs/s puts exact at 16k nodes at ~0.75-1 ms and at 32k at
~3-4 ms per iteration, both under a frame, while the grid tier's per-iteration cost is
ESTIMATED at 2-4 ms at those sizes (build-dominated). The default 16,384 is therefore
conservative for a discrete GPU and right for an integrated GPU (5-15x slower on this
compute-bound kernel, ESTIMATE). `ctx.calibrate()` runs the exact kernel at 8k and 16k
and the grid path at 16k once (~20 ms) and returns a suggested `exactMaxNodes`; the
graphty app may call it and pass the result. P5's benchmark fixes the shipped default
from measurements at 4k / 8k / 16k / 32k / 65k on the 4070 SUPER and on lavapipe.

The exact kernel is also the ORACLE: `test/layouts/repulsion-grid.test.ts` asserts the
grid forces agree with the exact forces within a relative tolerance (5% RMS on uniform
random and clumpy fixtures; cosmos's 1,024-points-in-one-cell case as the stress case)
and that the FA2 pipeline with either tier reaches the same distributional metrics
(edge-length distribution, stress, per-node neighbour-distance histogram).

### 7.9 Swing, traction and the global speed on the device (kernel 3)

cuGraph brings `swing` and `traction` to the host with two `thrust::reduce` calls per
iteration (note 03 section 4.3 item 9); with WebGPU's async submission that would cost a
`mapAsync` round trip per iteration (0.04 ms Node, 0.10 ms Chromium, MEASURED) and, worse,
serialise the frame loop. Instead kernel 3 is a single 256-lane workgroup:

```wgsl
@compute @workgroup_size(256) fn adaptSpeed(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = vec2<f32>(0.0);
    for (var g = lid.x; g < p.numPartials; g += 256u) { acc += partials[g].xy; }   // grid-stride over ceil(n/256) partials
    red[lid.x] = acc; workgroupBarrier(); /* tree reduce */ ...
    if (lid.x == 0u) {
        let swing = red[0].x; let traction = red[0].y; let n = f32(p.n);
        var speed = state.speed; var eff = state.speedEfficiency;
        let optJitter = 0.05 * sqrt(n); let minJitter = sqrt(optJitter); let maxJitter = 10.0;
        var jitter = p.jitterTolerance * max(minJitter, min(maxJitter, optJitter * traction / (n * n)));
        if (swing / max(traction, 1e-30) > 2.0) { eff = max(0.05, eff * 0.5); jitter = max(jitter, p.jitterTolerance); }
        let target = select(jitter * eff * traction / swing, 1e30, swing == 0.0);
        if (swing > jitter * traction) { eff = max(0.05, eff * 0.7); } else if (speed < 1000.0) { eff = eff * 1.3; }
        speed = speed + min(target - speed, 0.5 * speed);
        state.swing = swing; state.traction = traction; state.speed = speed; state.speedEfficiency = eff;
    }
}
```

The reduction order is fixed (partials in workgroup order, tree in lane order), so the
speed trajectory is reproducible on a device; the differential test compares the
per-iteration `(swing, traction, speed)` trace with the CPU reference (f64) within a
tolerance on 10-1,000-node graphs (note 01 section 8.7).

### 7.10 Integrate, fixed nodes, 2D / 3D (kernel 4)

```wgsl
let f = readForce(i); let fo = readOld(i);
let swing = select(mass[i] * length(f), mass[i] * length(f - fo), SWING_MODE == 0u);
let factor = state.speed / (1.0 + sqrt(state.speed * swing));
var dp = f * factor;
if (p.tier == GRID) { dp = clampLength(dp, 2.0 * state.cellSize); }
let locked = maskBit(fixed, i) | maskBit(held, i);
if (!locked) { pos[3i] += dp.x; pos[3i+1] += dp.y; if (DIM == 3u) { pos[3i+2] += dp.z; } }
writeOld(i, f);
movement = select(0.0, length(dp), !locked);            // partials: sum |dp| over FREE nodes, plus position sums and bbox
```

Fixed nodes exert forces but never move (`setFixed`); held nodes are those written by
`setPosition` since the last `setFixed` (dragging). The centroid partial sums the NEW
positions so gravity in the next iteration uses an up-to-date centroid. `DIM == 2`:
z is neither read nor written (positions keep whatever z the element holds, normally 0).

### 7.11 Units, initial placement and `load()`

Decision D8: steppable FA2 simulates in SCENE units with Gephi / cuGraph semantics
(`scalingRatio` controls spread), because a stepping engine cannot re-normalise to the
unit ball per step (it would rescale pinned and dragged nodes and change the camera
framing every frame; note 01 section 8.4) and because FA2's forces are not
scale-invariant. `scalingFactor` (element `SimpleLayoutConfig`, default 100) becomes the
SEEDING RADIUS: NaN rows are seeded uniformly in `[-scalingFactor, scalingFactor)` per
axis with the layout package's LCG (`m = 2^35 - 31`, `a = 185852`, `c = 1`; `seed`
semantics identical, seed 0 = unseeded; note 01 section 2.5.1) on the CPU in node-index
order, so a seeded GPU start equals a seeded CPU-simulation start. Finite rows are kept
verbatim (that is how a re-`load` after a topology change preserves the user's layout,
design 14.4). `dim === 2` seeds `z = 0`. The one-shot `indexed.forceAtlas2` of the layout
package keeps its unit-ball rescale for backward compatibility; the L1 rewrite makes
its steppable CPU twin follow this plan's convention so the element has ONE code path
and the GPU is an implementation swap (note 01 section 8.8 item 3).

`load(snapshot, positions)`:

1. `E_BAD_OPTION` unless `!snapshot.directed` (layouts receive `dm.undirected(s).snapshot`).
2. `residency.core(snapshot)` (hot prefix); `arcCount === 0` -> attraction dispatch skipped, no zero-length binding.
3. mass: `nodeMass` as `Float32Array` (length n) / column name (`nodes.gpuView`) / `Record` (converted through `snapshot.ids`) else `outDegree()[i] + 1` (CPU loop into a fresh `F32`, uploaded).
4. seed NaN rows (above); `writeBuffer(positions)` (12n bytes; 1.2 MB at 100k, ~12 MB at 1M).
5. zero `force`, `oldForce`; state block `{ speed: 1, speedEfficiency: 1, iteration: 0, settled: 0 }`; `fixed` from the `fixed` option mask or zeros; `held` zeros.
6. tier selection (7.8) and grid allocation when needed; pipelines warmed for the chosen overrides.
7. `settled = false`, `iterations = 0`.

`load` on a second snapshot (topology change) keeps step 4's semantics: the element has
already remapped and NaN-filled the array (design 14.4).

### 7.12 Settlement and reheat

FA2 has no usable convergence test (the port's `1e-10` movement test never fires; note
01 section 2.1.3 item 10). `settled` is true when EITHER `state.iteration >= maxIter`
(default 100; the element may pass 500-1000 for an interactive session, cuGraph docs
say 50-100 give good short-term quality and > 1000 is discouraged, note 03 section 4.1)
OR the mean of the last `settleWindow = 10` per-iteration values of `movement /
freeNodes` is below `settleThreshold = 0.05` scene units (the element's loosened ngraph
heuristic, `NGraphLayoutEngine.ts` lines 179-191 via note 01 section 3.1). The finalize
kernel keeps the 16-entry ring and writes `state.settled`; the host reads it with every
readback. `setPosition`, `setFixed` with fewer bits than before (an unpin), `load` and
`reheat()` reset `iteration = 0` and clear the ring (d3 / ngraph behaviour, note 01
section 8.6). `settled` is only reported from a COMPLETED readback (the frame loop is
one submission behind).

### 7.13 Determinism

| Tier | Same device, same dispatch shape | Across devices / runtimes |
| --- | --- | --- |
| exact | bitwise reproducible: fixed tile order, fixed partial and tree order, no atomics | not bitwise (fma contraction, subgroup width differences in the reduce variant); tests use traces with tolerance and distributional metrics |
| grid | bitwise reproducible with the radix sort (stable) and the fixed-point centroid atomics (integer addition commutes); the near-field offset is a pure hash of `(seed, iteration, cell)` | same as above |

The differential test runs the CPU reference (L1 rewrite, f64) and the GPU for the same
seed and iteration count on graphs of 10-1,000 nodes and compares the per-iteration
`(swing, traction, speed)` trace within `1e-3` relative (growing with iteration) and
final distributional metrics; coordinate equality is not a goal (note 01 section 8.7).

### 7.14 Option parity with the CPU `forceatlas2.ts` and the element schema

From `layout/src/layouts/force-directed/forceatlas2.ts` lines 26-42 and
`graphty-element/src/layout/ForceAtlas2LayoutEngine.ts` (note 01 section 7):

| Option | CPU default | GPU v1 | Binding |
| --- | --- | --- | --- |
| `maxIter` | 100 | honoured as the total iteration budget across `step` calls | settlement |
| `jitterTolerance` | 1.0 | honoured | uniform |
| `scalingRatio` | 2.0 | honoured | uniform |
| `gravity` | 1.0 (element schema forbids 0; CPU accepts 0) | accepts 0 | uniform |
| `strongGravity` | false | honoured | override |
| `distributedAction` | false | honoured | override |
| `linlog` | false | honoured | override |
| `nodeMass` | null -> degree + 1 | `Float32Array` / column name / Record / null | mass buffer |
| `nodeSize` / `adjustSizes` | null | DEFERRED (`E_UNSUPPORTED` when given, until the L1 rewrite fixes the sign) | -- |
| `weight` | null | `true` (snapshot weights) or `null`; a named column arrives pre-expanded from the element (`F32(arcCount)`) | `HAS_WEIGHTS` |
| `dissuadeHubs` | ignored by the CPU | accepted and ignored | -- |
| `seed` | null | honoured for seeding (LCG) and the near-field hash | CPU + uniform |
| `dim` | 2 | 2 or 3; the element overrides from the view mode and RE-CREATES the engine on a switch (note 01 section 4.6) | override `DIM` |
| `pos` | null | replaced by the `positions` array of `load()` (finite rows kept, NaN seeded) | -- |
| `scale` / `center` (`CommonLayoutOptions`) | -- | honoured at `load` only as the seeding radius / offset | -- |
| new: `repulsion`, `exactMaxNodes`, `gridMax2D`, `gridMax3D`, `nearMax`, `deterministic` | -- | section 7.7-7.8 defaults `"auto"`, 16384, 512, 128, 64, true | overrides / uniforms |
| new: `settleThreshold`, `settleWindow`, `iterationsPerStep` | -- | 0.05, 10, 1 (the element passes `stepMultiplier`) | state |
| new: `compat` | -- | `"paper"` (default) or `"port"` (section 7.2) | overrides |

### 7.15 The float-atomic workaround, summarised

WGSL has `atomic<u32>` / `atomic<i32>` only (note 05 section 6 item 1). Every force
term is therefore GATHERED by the node that owns it: attraction sums over the node's own
CSR row (both arcs of an undirected edge are stored, design 10.5), repulsion sums over
the tile or the grid, gravity is per node. The only atomics in the layout are `u32`
histogram counts in the radix sort, optional `i32` fixed-point cell sums for hub cells
and, in the alternative counting sort, `u32` cursors. cuGraph's four `atomicAdd(float)`
per edge (`fa2_kernels.cuh` lines 77-80, note 04 section 12) and d3-force-webgpu's racy
`+=` are both avoided by construction.

### 7.16 Fruchterman-Reingold / spring (second layout)

Same skeleton, two force kernels and a simpler controller (note 01 section 2.2; note 02
L2 score 7.5):

| Item | CPU (`fruchterman-reingold.ts`) | GPU simulation |
| --- | --- | --- |
| `k` | `1 / sqrt(n)` in `[0, 1]` units | `k = 2 * seedRadius / sqrt(n)` (same formula on the seeding box `area = (2 seedRadius)^2`) |
| repulsion | `k^2 / d` along `delta / d`, `d = |delta| || 0.1` | `diff * k^2 / d^2` with the exact tile or grid tier (`mass = 1`, `REPULSION_LAW = 0`, strength `k^2`): the same kernel family as FA2 with different `override` strength semantics |
| attraction | per edge `d^2 / k` toward the neighbour | CSR gather: `F_i += (p_j - p_i) * d / k` per arc (edge counted from both endpoints by construction) |
| temperature | `t = 0.1`, `dt = t / (iterations + 1)`, `t -= dt` per iteration | `t0 = 0.1 * 2 * seedRadius`; `t = t0 * (1 - iteration / (iterations + 1))` computed in the finalize kernel; `reheat()` sets `iteration = floor(0.7 * iterations)` so a drag gets a small temperature |
| apply | move along `disp` by `min(|disp|, t)`; skip `fixed` | integrate variant `FR_APPLY` with `fixed` / `held` masks |
| termination | fixed `iterations` (50) | `settled` when `iteration >= iterations` or the movement threshold |
| output | `rescaleLayout` unless `fixed` given | scene units, no rescale (same D8 convention) |
| singularities | `|| 0.1` exact-zero guard | coincident kick as FA2 |

Four dispatches per iteration (repulsion, attraction, integrate, finalize; no
adaptSpeed). The single-RNG fix of design 14.3 applies to seeding.

### 7.17 Other layouts

- "ngraph-like" preset (note 02 L3): the element's DEFAULT layout is ngraph
  (`config/GraphBehavior.ts` line 13). Rather than port ngraph, the FR simulation gets a
  preset `spring-electrical` with a velocity buffer (12n), Hooke springs
  (`(d - springLength) * springCoefficient`), `1/d` repulsion with `gravity` strength,
  drag and `timeStep` (ngraph's own defaults 10 / 0.8 / -12 / 0.9 / 0.5; note 01
  section 3.1), and ngraph's settle rule (`lastMove / n <= 0.01` or the element's 10-step
  average 0.05 or 1000 steps). It is the same kernel family with an `INTEGRATOR` override.
  Reserved for P6; the element decides whether to route `ngraph` to it above a node
  count (product decision, risk R12).
- ARF: `K_ij = 1 + (a - 1) * isEdge` decomposes into an all-pairs term plus a CSR-row
  correction (note 01 section 2.3); one override on the FR kernels; not scheduled.
- Kamada-Kawai: needs APSP (section 8.5) and a per-line-search cost readback; not a
  `LayoutSimulation`; scheduled only after the APSP kernel exists and only for
  `n <= ~10k` (note 02 L5).
- Spectral: Laplacian SpMV + shifted inverse iteration (fixes the CPU's wrong
  eigenvectors, note 02 L4); after the SpMV primitive; low priority.

### 7.18 `LayoutSimulation` implementation and the element frame loop

The element's `UpdateManager.update()` is synchronous and calls `engine.step()` up to
`stepMultiplier` times per frame from Babylon's render loop; nothing awaits
(`managers/UpdateManager.ts` lines 146-214 via note 01 section 4.1). The GPU engine
therefore uses fire-and-forget double buffering (note 01 section 5.3 option A,
decision D11):

```
LayoutStepper (one per simulation)
  requestStep(k):  if (inFlight) { pending = max(pending, k); return; }  submit(k)
  step(k): Promise -> requestStep(k) and return the promise of THAT submission's readback (Node batch API: `while (!sim.settled) await sim.step(10)`)
  submit(k):
      encoder = createCommandEncoder()
      pass = beginComputePass(); for (it in 0..k) recordIteration(pass); pass.end()
      ring.copy(encoder, positions, 12n); ring.copy(encoder, state, 256)       // one staging slot for both, laid out back to back
      queue.submit([encoder.finish()]); inFlight = true
      ring.map().then(onReadback, onError)
  onReadback(buf):
      for free rows (not held): elementPositions.set(bufView)                 // held rows keep the CPU value (drag)
      column.markDirty()                                                       // design 14.4 M12
      iterations = state.iteration; settled = state.settled != 0; stats = ...
      inFlight = false; if (pending > 0) { k = pending; pending = 0; submit(k); }
  setPosition(i, x, y, z): queue.writeBuffer(positions, 12 i, [x, y, z]); elementPositions[3i..] = x,y,z; held.set(i); queue.writeBuffer(heldWord); reheat()
  setFixed(mask): queue.writeBuffer(fixed, 0, mask); held.clear(); queue.writeBuffer(held); if (fewer bits than before) reheat()
```

Properties:

- The frame loop never blocks; the renderer is at most one submission behind the
  simulation. `settled` flips only after a completed readback, so the element's
  `graph-settled` / zoom-to-fit / screenshot polling (note 01 section 4.3) sees a truthful
  value.
- `queue.writeBuffer` calls issued between submissions are ordered before the next
  submit (WebGPU queue ordering), so a drag position is honoured by the next batch.
  The dragged node's mesh is written directly by `NodeBehavior` (note 01 section 4.5),
  so the one-frame readback lag affects only neighbours, as with ngraph today.
- Readback cost per frame: 12n bytes + 256 (MEASURED 2.65 ms per MiB copy+map+slice in
  Chromium, note 05 section 7.2): ~0.3 ms at 10k nodes, ~3 ms at 100k, ~30 ms at 1M
  (ESTIMATE, linear). At 1M nodes in a browser the element cannot render per-node
  meshes anyway (note 01 section 6); `readbackEvery: k` (default 1) lets a batch caller
  read back every k submissions, and Node's in-process map is ~10x cheaper.
- Node.js: identical code path (Dawn), no frame loop: `await sim.step(k)` in a loop.
- Topology change: `load(next, positions)` while in flight waits for the in-flight
  readback (it would otherwise copy stale rows into a remapped array), then re-uploads.
- `dispose()`: waits for / cancels the in-flight map (`buffer.destroy()` rejects the
  pending `mapAsync`), destroys every simulation buffer, leaves the snapshot residency
  to `release(snapshot)`.

The `LayoutManager` bridge (E1 + this plan's P4): `ForceAtlas2LayoutEngine` becomes a
thin `LayoutEngine` whose `step()` calls `sim.requestStep(stepMultiplier)`, `isSettled`
returns `sim.settled`, `pin/unpin` maintain the pinned `NodeMask` and call
`sim.setFixed`, `setNodePosition` calls `sim.setPosition`, `getNodePositionInto` reads
the element array; the simulation object is `accelerator?.forceAtlas2?.(opts) ??
cpuSimulation(opts)` (section 9).

### 7.19 Scaling table (per iteration; 4070 SUPER class discrete GPU and an integrated GPU)

Basis: exact tier = MEASURED 3.6 x 10^11 pairs/s x 1.3 FA2 factor (ESTIMATE);
attraction = MEASURED 0.1-0.7 ms per iteration for a 100k-node / 1M-arc gather on the
4070 under Dawn (note 06 section 3.5), scaled linearly in arcs (ESTIMATE); grid tier =
ESTIMATE from the operation counts of section 7.7 (build ~1-3 ms at 1M nodes for the
3-digit radix sort + reductions, far field ~2 x 10^8 (2D) / ~10^9 (3D) centroid
evaluations per 1M nodes at 1-3 x 10^11 /s, near field ~10^7-10^8 pairs) cross-checked
against PUBLISHED numbers: cosmos 100k 6.6 ms and 200k 13.8 ms per step (WebGL, GPU
unnamed), GraphWaGu 95k nodes / 6.6M edges 5.48 ms and 1.13M nodes ~160 ms per iteration
(RTX 4070 Laptop, theta 2), Brinkmann 1.13M 231 ms (Titan X, 2017). Integrated GPU
column: 8-15x slower than the 4070 SUPER for the compute-bound exact tier, 4-8x for the
bandwidth-bound grid tier (ESTIMATE; GraphWaGu's Iris Xe numbers are speedups over its
2022 version, not a ratio to the 4070, so no direct measurement exists). The tiny
dispatches (adaptSpeed, integrate, finalize) add ~0.1-0.2 ms per iteration of fixed
cost on a discrete GPU.

| n / arcs (undirected, avg degree 10) | Tier (auto) | Repulsion 4070 | Attraction 4070 | Iteration 4070 | Iteration integrated | Readback per frame (Chromium / Node) | Interactive verdict (1 iteration per 16.7 ms frame) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 10k | exact | 0.05 ms (dispatch floor) | 0.05 ms | ~0.3 ms | ~1 ms | 0.1 / 0.03 ms | yes, 10+ iterations per frame |
| 10k / 100k | exact | 0.3-0.4 ms | 0.05 ms | ~0.6 ms | ~4 ms | 0.3 / 0.1 ms | yes, 5+ per frame |
| 16k / 160k | exact (ceiling) | 0.9-1.0 ms | 0.1 ms | ~1.3 ms | ~10 ms | 0.5 / 0.1 ms | yes |
| 32k / 320k | grid (exact would be 3-4 ms) | 2-3 ms | 0.15 ms | ~3 ms | ~12 ms | 1 / 0.2 ms | yes |
| 100k / 1M | grid | 3-5 ms (2D), 5-8 ms (3D) | 0.1-0.7 ms (MEASURED) | 4-9 ms | 20-50 ms | 3 / 0.4 ms | yes on discrete (1 iteration per frame); batch on integrated |
| 1M / 10M | grid | 5-10 ms (2D), 8-15 ms (3D) | 1-7 ms | 8-25 ms | 50-150 ms | 30 / 4 ms | batch / Node; browser needs `readbackEvery` and instanced rendering |
| 1M / 10M | exact (forced) | ~3.6 s | -- | -- | -- | -- | never interactive; Node batch only, correctness oracle |
| lavapipe (CI), 20k | exact | 388 ms (MEASURED) | 3-5 ms (MEASURED at 4 threads, 1M arcs) | ~0.4 s | -- | -- | correctness only; fixtures scaled by `gpuScale()` |

Every row of this table is re-measured by `benchmarks/layouts.bench.ts` in P3 (exact)
and P5 (grid) on the 4070 SUPER and on lavapipe, and the GPU-lane CI job records them
as JSON (section 12) so the numbers in the shipped README come from the harness, not
from this plan.

### 7.20 Layout test plan (details in section 11)

- Kernel unit tests on Dawn: attraction equals a CPU row loop (`1e-5`), exact repulsion
  equals a CPU double loop (`1e-4` relative), adaptSpeed equals the CPU `estimateFactor`
  for random `(swing, traction)` pairs, integrate honours `fixed` / `held` and `DIM`.
- Simulation properties (no reference needed): fixed nodes never move; `dim === 2`
  leaves z exactly unchanged; disconnected components separate; gravity pulls the
  centroid toward the centre; `setPosition` is visible in the next readback; `settled`
  is reached within `maxIter`; `dispose` leaves no buffers (Dawn leak check via a
  wrapped `createBuffer` counter) and no uncaptured errors; `arcCount === 0` binds no
  zero-length buffer.
- Differential vs the CPU reference: `(swing, traction, speed)` trace and distributional
  metrics (section 7.13).
- Exact-vs-grid: force agreement and end-state metrics (section 7.8).
- Frame-loop bridge: a fake render loop calls `requestStep(1)` 600 times without
  awaiting; asserts exactly one submission in flight at any time, monotone iteration
  counts, and that a `setPosition` during flight lands in the following batch.
- Browser smoke: `load / requestStep x 10 / positions written / settled / dispose` on
  the karate club in Playwright Chromium (SwiftShader on the default lane, NVIDIA on the
  GPU lane).

---------------------------------------------------------------------------

## 8. Algorithms

Conventions shared by every algorithm below: entry `(ctx, snapshot, options?, dest?)`;
index-only sources (`snapshot.ids.requireIndex` is the caller's job, note 07 section
1.8); results index-aligned `F32` / `U32` with `INVALID_INDEX`; per-arc results folded
with `foldArcs`; labels renumbered dense in first-seen order with `renumberPartition`
(`derived.ts` line 1155) so `groups()` matches the CPU; `k` rounds per submit with
indirect dispatch; pull / gather first, `u32` atomics only where the value is an
integer; every kernel designed for 8 storage bindings. The CPU references until W1 are
small index-based implementations in `test/helpers/oracle.ts` written from the design's
Port 1-6 code; W1 switches the differential tests to `indexed.*` (note 02 section 4.5).

### 8.1 Priority order and rationale

Score = value x speedup / risk (note 02 section 6-7), re-ordered by primitive
dependencies so each slice pulls in at most one new primitive:

| Rank | Algorithm | Score | New primitive | Views | Phase |
| --- | --- | --- | --- | --- | --- |
| A1 | PageRank (+ personalized) | 25 | pull SpMV (tiered segmented reduce), multi-channel reduce | `reverse()` (aliases forward when undirected), device out-weight sums | P7 |
| A2 | HITS, eigenvector, Katz | 15 | none (same SpMV, L2 / sum normalise) | forward + `reverse()` | P7 |
| A3 | Weakly connected components (Afforest) | 8 | edge map with CAS, compress, histogram sample | `edgeList().src/.dst`, `rowPtr/colIdx` for r-th neighbour | P8 |
| A4 | BFS (depth + parent), direction-optimizing | 5.3 | frontier, advance, compaction/dedupe, bitset | `rowPtr`, `colIdx`, `reverse()` (directed), `degreeOrder()` | P9 |
| A5 | Closeness / harmonic / eccentricity | 5.3 | batched multi-source BFS (bitmask frontier, 32 sources per word) | as A4 | P9 |
| A6 | SSSP (near-far) | 4 | `atomicMin` on f32 bit patterns, histogram by subpartition | `rowPtr`, `colIdx`, `weights`, flags | P9 |
| A11 | Bellman-Ford (negative weights) | 6 | edge-parallel relax with CAS loop | `edgeList()` | P9 |
| A7 | Betweenness (node + edge), sampled sources | 6.3 | tagged multi-source BFS, successor-pull dependency | `rowPtr`, `colIdx`, `edgeList()` for the edge-parallel mode, `edgeToArc` for edge BC | P10 |
| A9 | APSP / Floyd-Warshall | 5 | blocked FW (weighted) or n batched BFS (unweighted), `n <= 8192` at the default `maxBufferSize` | `rowPtr`, `colIdx`, `weights` | P10 |
| A8 | Label propagation | 4 | per-row group-by-key (workgroup sort / hash) | `rowPtr`, `colIdx`, `weights` | P11 |
| A10 | k-core | 4.5 | peeling rounds with `atomicSub` and compaction | `rowPtr`, `colIdx` | P11 |
| A12 | MST (Boruvka) | 3 | per-component min edge via two-pass `atomicMin` | `edgeList()`, A3's compress | P11 |
| A13 | Triangle count / common neighbours / Adamic-Adar / k-truss | 4 | oriented-edge sorted-row intersection | sorted `rowPtr/colIdx`, `outDegree()`, `edgeToArc` | P11 |
| A14 | Louvain / Leiden | 3 | per-row group-by-key, radix sort by (cluster src, cluster dst), segmented reduce, COO -> CSR | symmetric CSR, `edgeList()`, weighted degree on device | P11 (last) |
| -- | SCC, spectral | 1.5 | forward-backward reachability; Lanczos | `reverse()` | not scheduled |
| -- | DFS, topological sort, cycle detection, Prim, Girvan-Newman, hierarchical, MCL, max-flow / min-cut, bipartite matching, isomorphism, A* | -- | sequential or small-graph by nature: CPU path stays in the adapters | -- | never |

### 8.2 Family: SpMV / iterative (PageRank, personalized PageRank, Katz, eigenvector, HITS)

Strategy (cuGraph `pagerank_impl.cuh` lines 222-320, `katz_centrality_impl.cuh`,
`eigenvector_centrality_impl.cuh`, `hits_impl.cuh`; note 04 section 5): power iteration
as a PULL over in-neighbours (`per_v_transform_reduce_incoming_e` + plus reduce); no
atomics (Gunrock's push `atomic::add` on f32, `pr.hxx` line 145, is the form WebGPU
cannot use).

Per-iteration kernels (PageRank):

1. `prDangling` (reduce): `sum of rank[v] where outWeight[v] == 0` -> partials.
2. `prPull` (segmented reduce tiered by IN-degree over `reverse()`): `rankOut[v] = (1 -
   alpha) / n + alpha * (danglingSum / n + sum_a w[a] * rankIn[colIdx[a]] /
   outWeight[colIdx[a]])`; personalization replaces the `1/n` terms by
   `personalization[v]` (normalised on upload).
3. `prDelta` (reduce): `sum |rankOut - rankIn|` -> partials; a 1-workgroup finalize
   writes `delta` and the dangling sum into the state block for the next iteration
   (the dangling reduce of step 1 folds into this finalize: three dispatches per
   iteration, 2 of them tiny).

Setup: `outWeight` = segmented reduce over forward `rowPtr` / `weights` (design 10.1: NOT
a format upload; the `weightedOutDegree()` view is f64), guarded division (`outWeight ==
0` is the dangling test, not `outDegree == 0`); cached via `residency.scratch(snapshot,
"outWeight")`. Convergence: readback of `delta` every `k = 8` iterations; stop at `delta
< tol` (default `1e-6`, f32-meaningful, note 04 section 5) or `maxIterations` (100).
Katz: `alpha * sum + beta`; eigenvector: L2 normalise per iteration (a `sumSq` reduce +
a scale map), convergence `delta < n * tol`; HITS: authorities pull over `reverse()`,
hubs pull over forward, sum-normalise. WebGPU adjustments: Kahan accumulation in the
hi-tier row loop; bindings = revRowPtr, revColIdx, revWeights, outWeight, rankIn,
rankOut, personalization, partials = exactly 8 (note 04 section 5); personalization
goes to a 16-byte dummy when absent. Cost: O(A) per iteration, bandwidth-trivial (the
100k / 1M graph is 8-16 MB), dispatch-latency bound: MEASURED 0.1-0.7 ms per gather
iteration (note 06 section 3.5). Result: `{ scores: F32(n), iterations, converged }`;
parity `1e-5` relative and top-k order versus the CPU f64.

### 8.3 Family: components (WCC via Afforest)

Strategy (GAP `gapbs/cc.cc` lines 40-150; Sutton-Ben-Nun-Barak 2018; note 04 section 8):

1. `comp[v] = v`.
2. Two sampled link rounds: each vertex links to its r-th neighbour (`colIdx[rowPtr[v]
   + r]` if `r < degree`): `link(u, v)` hooks the higher root to the lower with
   `atomicCompareExchangeWeak`, retrying up the trees.
3. `compress`: pointer-jump every vertex to its root; reads are `atomicLoad` on the same
   `array<atomic<u32>>` (WGSL forbids mixing atomic and non-atomic access to one
   element; note 04 section 8).
4. Frequent-element sample: histogram 1,024 random `comp` values (one tiny kernel +
   one 4 KiB readback) -> the giant component's root `c`.
5. Link the remaining edges of vertices not in `c` via `edgeList().src/.dst` (each edge
   once; correct for directed and undirected without a reverse view, design 10.1),
   compress, repeat until a device-side `changed` flag stays 0 (checked every 4
   rounds).
6. Readback `comp`, `renumberPartition(comp)` on the CPU (first-seen order) -> `{ labels,
   count }`, identical to the CPU `groups()` output.

All `u32`; expected 5-10 rounds; result set-equal to the CPU union-find; cuGraph's
multi-root frontier expansion (`weakly_connected_components_impl.cuh`) is more machinery
for the same asymptotics (note 04 section 8).

### 8.4 Family: traversal (BFS, closeness, SSSP, Bellman-Ford)

BFS (Merrill-Garland-Grimshaw 2011; Beamer 2012; cuGraph `bfs_impl.cuh`; note 04
section 3): level-synchronous two-phase frontier (expand kernel = advance with
`visit(u, v, a) = atomicCompareExchangeWeak(&depth[v], INVALID, level).exchanged`,
writing `parent[v] = u` only from the winning invocation; contract = compaction with
ownership dedupe), a fused expand-contract kernel for frontiers smaller than `4 x WG x
occupancy` (Merrill's "fleeting iterations"), degree tiers from `degreeOrder()`, and
`dispatchWorkgroupsIndirect` driven by the device-side frontier length so `k = 16`
levels run per submit; the host reads the length every k levels. Direction-optimizing
(Beamer): switch to bottom-up when `m_f > m_u / alpha` and the frontier grows, back when
`n_f < n / beta` and it shrinks; constants from cuGraph verified in source: `alpha = m /
n` (average degree), `beta = 24` (`bfs_impl.cuh` lines 291-297, 637-638, 843-846; note 04
finding 4); bottom-up iterates the non-zero-degree unvisited list over `reverse()`
(the forward CSR when undirected: free), so the option `directionOptimizing: "auto"`
enables it whenever `!directed || reverse` is already resident. Result `{ depth: U32,
parent: U32, order: U32, visitedCount }`: `depth` exact; `parent[v]` is any vertex at
`depth[v] - 1` with an arc to `v` (NOT the CPU FIFO parent); `order` grouped by level
(design 14.2 parity rules, note 02 section 5). Cost O(n + A) work, O(diameter) dispatch
rounds; Merrill reports 3.3 GTEPS on one 2011 GPU (PUBLISHED); the plan expects 10-30x
over the Map-of-Maps CPU BFS at 1M edges once the graph is resident (ESTIMATE), and
NOTHING if a level pays a `mapAsync` (europe.osm ~19,000 levels).

Closeness / harmonic / eccentricity: batched multi-source BFS with a 32-source bitmask
frontier per `u32` word (each word holds "is v in the frontier of source s"), per-source
distance rows reduced ON THE DEVICE (sum, sum of `1/d`, max) so `n x n` is never
materialised; weighted closeness = batched near-far SSSP (note 04 section 7). Result
`F32(n)`; exact in integers before the division.

SSSP (Davidson-Baxter-Garland-Owens 2014 near-far; cuGraph `sssp_impl.cuh` lines
189-262; note 04 section 4): requires `flags.nonNegativeWeights` (`E_UNSUPPORTED`
otherwise; `flags.allWeightsOne` degrades to BFS). `dist` is `array<atomic<u32>>` of f32
bit patterns; `atomicMin` is exact for non-negative floats; `+Inf` (`0x7F800000`) is
"unreached". Near pile = queue entries with `dist < (i + 1) * delta`, far pile deferred;
`delta = 32 * avgWeight / avgDegree` (Davidson's `c = warp width`), cuGraph's two-level
near queue with 16 subpartitions capped at `occupancy x 2048 / avgDegree` entries;
predecessors by a second pass over the settled frontier writing `atomicMin(&predArc[v],
a)` where `dist[u] + w == dist[v]` (no 64-bit packed atomic exists; note 04 section 4).
The near-empty test is a device flag turned into a zero indirect dispatch so extra
queued iterations are no-ops. Result `{ dist: F32, predArc: U32 }`; `dist` within `1e-5`
relative; `predArc` any arc attaining `dist` (ties differ from Dijkstra's).

Bellman-Ford: edge-parallel relax over `edgeList()` (both directions on undirected
snapshots inside the kernel), signed floats via a CAS loop on the bit pattern, `n - 1`
rounds max with early exit on a `changed` flag checked every 8 rounds, one more round
for the negative-cycle flag. Result `{ dist, predArc, hasNegativeCycle }`.

### 8.5 Family: centrality (betweenness, APSP)

Betweenness (McLaughlin-Bader 2014/2018, read from the author's mirror PDF; cuGraph
`betweenness_centrality_impl.cuh`; note 04 section 6): Brandes with a work-efficient
forward pass (queues, `atomicCompareExchangeWeak` on depth, `atomicAdd(u32)` on sigma
with a saturation flag -- overflow past 2^32 paths on small-world graphs is detected and
reported in `details`), an `S / ends` per-level layout produced by the compaction
itself, and a backward dependency pass that PULLS from successors (`delta[w] +=
sigma[w] / sigma[v] * (1 + delta[v])` for `v` at `depth[w] + 1`, "to eliminate the use of
atomics") so `bc[w] += delta[w]` is a plain add; no float atomics anywhere. Sources are
batched cuGraph-style into a tagged multi-source BFS (two-word `(vertex, source)`
frontier entries; `n x k` sigma / depth arrays sized from `device.limits.maxBufferSize`
and a 25% budget of a caller-supplied `memoryBudget`) so k tiny per-level dispatches
become one big one. The McLaughlin-Bader hybrid: run the first batch work-efficiently,
read back the max depth, switch to the edge-parallel kernel (`edgeList()`, every arc
relaxed if its source is at the current level) when the median depth `< gamma *
log2(n)`. Exact BC is O(n x A): at 100k nodes it is ~100k traversals (minutes on the
GPU), so the API exposes `sources: number | U32` (k sampled sources, "trivially
adjusted for approximation" per both papers) and reports `sourcesUsed`. Edge BC writes
per arc and folds with `foldArcs(.., "first")`, halved for undirected as the CPU does.
Result `F32(n)`, normalised in the same convention as `indexed`; parity `1e-4` relative
(f32 accumulation over many sources) and top-k order.

APSP / Floyd-Warshall: unweighted = the batched BFS of A5 writing rows into an `n x n`
`F32`; weighted = blocked Floyd-Warshall (tiled 32 x 32 phases, the classic GPU
kernel); bounded by `maxBufferSize` (`n <= 8,192` at 256 MiB, `n <= 32k` at 4 GiB).
Feeds Kamada-Kawai's `dist: Float32Array(n * n)` (design 14.3).

### 8.6 Family: community (label propagation, Louvain / Leiden)

Label propagation: per node the most frequent (weighted) neighbour label = a per-row
group-by-key (workgroup-memory sort of `<= 256` keys for the thread / subgroup tiers,
an open-addressing hash in workgroup memory for `<= 4,096`, a global hash region of
`2 x degree` slots for hubs -- nu-Louvain's layout, arXiv 2501.19004), Jacobi-style
synchronous update with cuGraph's `up_down` swap-avoidance rule, `changed` reduce.
Result `{ labels: U32, iterations, converged }`; set-equality on planted partitions.

Louvain (cuGraph `louvain_impl.cuh` lines 60-300, `detail/common_methods.cuh` lines
70-152, 402-446; note 04 section 10): per level, vertex weights by segmented reduce;
synchronous best-move pass: per-row group-by-key on `cluster[dst]` accumulating edge
weight per neighbouring cluster, `delta_Q = 2 * ((new - old) / total - resolution *
(a_new k - a_old k + k^2) / total^2)`, deterministic tie-break, move only if `delta_Q >
minGain` AND the direction matches `up_down` (flips every pass); cluster weights
RECOMPUTED by reduce-by-key after each pass (no float atomics, deterministic);
contraction = radix sort of arcs by `(cluster src, cluster dst)` + segmented reduce +
COO -> CSR on the device; loop while modularity improves; flatten the dendrogram at the
end. Host readback per level: `Q` and the move count (two floats). Expectation
management (nu-Louvain: GPU Louvain only 1.03x faster than a 64-thread CPU because later
passes lose parallelism): 2-10x over the CPU package at 1M edges, not 100x; the GPU
runs the small levels too (no fallback). Leiden adds the refinement phase with the cut
condition and a maximal-independent-set kernel; scheduled after Louvain. Result `{
labels, count, modularity, iterations }`; parity = modularity within a band.

### 8.7 Family: structure (k-core, triangles, k-truss, MST)

k-core (cuGraph `core_number_impl.cuh`): core numbers initialised to degrees, rounds of
"frontier of vertices with count < k" -> `atomicSub` on neighbour counts -> compaction;
k increases when the frontier empties; O(max core) host-visible rounds with indirect
dispatch. Result `U32(n)` exact.

Triangle counting / common neighbours / Adamic-Adar / k-truss (cuGraph
`triangle_count_impl.cuh` lines 344-470; Gunrock `tc.hxx`): orient each edge from lower
to higher degree (ties by id) via a compaction of arcs where `(deg[u], u) < (deg[v],
v)`; per oriented arc intersect the sorted rows (`flags.sortedRows` makes it a merge;
binary search of the shorter into the longer when degrees differ by > 32x);
`atomicAdd(u32)` per-vertex counts; workgroup-per-arc tier for hub pairs; per-edge
support through `edgeToArc`; k-truss peels edges with support `< k - 2` and recounts
affected edges.

MST (Boruvka): per-component minimum edge via two-pass `atomicMin` (weight bits, then
edge index among ties), union via A3's compress; edge set identical to Kruskal on
distinct weights, `totalWeight` within `1e-5`.

### 8.8 Prior-art references and what each contributes

| Reference | Used for |
| --- | --- |
| Merrill, Garland, Grimshaw 2011 (NVIDIA research page + TR) | scan-based frontier expansion, gather tiers, duplicate culling, expand / contract couplings (8.4) |
| Beamer, Asanovic, Patterson SC12 | direction-optimizing switch (8.4) |
| Davidson, Baxter, Garland, Owens IPDPS 2014 | near-far SSSP, ownership dedupe (8.4, 6.4) |
| McLaughlin, Bader CACM 2018 (mirror PDF) | atomic-free dependency accumulation, hybrid selection, source sampling (8.5) |
| cuGraph source (`bfs_impl.cuh`, `sssp_impl.cuh`, `pagerank_impl.cuh`, `betweenness_centrality_impl.cuh`, `louvain_impl.cuh`, `core_number_impl.cuh`, `triangle_count_impl.cuh`) | verified constants (alpha, beta, delta, batch caps, up_down), pull formulations |
| Gunrock (`block_mapped.hxx`, `neighborreduce.hxx`, `csr.hxx from_coo`) | advance load balancing, COO -> CSR |
| GAP `cc.cc` (Afforest) | connected components (8.3) |
| GraphWaGu (`sort.ts`, `create_tree.wgsl`, `apply_forces.wgsl`) | WGSL radix sort shape, level-wise builds, i32 fixed-point min/max (6.6, 6.12) |
| Buffalo CSE 2023-06 (Kumar MS thesis: dense cuBLAS BC) | excluded: dense-only, beats McLaughlin-Bader only at >= 50% density (note 04 finding 8) |
| NVIDIA cluster-analysis page | background for spectral / multilevel partitioning only; no kernel detail |
| @antv/webgpu-graph | cautionary: dense matrices, per-iteration readback (note 04 finding 10) |

---------------------------------------------------------------------------

## 9. Integration with @graphty/algorithms, @graphty/layout and @graphty/graphty-element

### 9.1 The seam is async

Every public CPU algorithm is synchronous (`pageRank` at
`algorithms/src/algorithms/centrality/pagerank.ts:83`); WebGPU results need `mapAsync`.
GPU acceleration can therefore never be spliced into the sync entry points. The seams
are the element's `async run()` adapters (`graphty-element/src/algorithms/Algorithm.ts:283`)
and the design's `LayoutSimulation.step(): void | Promise<void>` (note 02 finding 1).

### 9.2 Interfaces owned by the CPU packages; dispatcher; wiring (note 02 section 4.5)

```ts
// @graphty/algorithms (A2), next to the indexed.* result types; no WebGPU types anywhere
export interface AlgorithmAccelerator {
    pageRank?(s: GraphSnapshot, o?: PageRankOptions): Promise<PageRankResult>;               // scores: NumericVector (F32 on the GPU)
    personalizedPageRank?(s, personalization: F32, o?): Promise<PageRankResult>;
    hits?(s, o?): Promise<HitsResult>; eigenvectorCentrality?(s, o?): Promise<ScoreResult>; katzCentrality?(s, o?): Promise<ScoreResult>;
    connectedComponents?(s): Promise<LabelResult>;
    breadthFirstSearch?(s, source: number, o?): Promise<BfsResult>;
    sssp?(s, source: number, o?): Promise<SsspResult>; bellmanFord?(s, source, o?): Promise<BellmanFordResult>;
    closenessCentrality?(s, o?): Promise<ScoreResult>; betweennessCentrality?(s, o?): Promise<ScoreResult>; edgeBetweennessCentrality?(s, o?): Promise<EdgeScoreResult>;
    labelPropagation?(s, o?): Promise<LabelResult>; kCoreDecomposition?(s): Promise<U32>; allPairsShortestPath?(s, o?): Promise<ApspResult>;
    louvain?(s, o?): Promise<CommunityResult>;
    release?(s: GraphSnapshot): void;
}
export function accelerated(acc: AlgorithmAccelerator | null | undefined): AcceleratedIndexed;
// accelerated(acc).pageRank(s, o) === acc?.pageRank ? acc.pageRank(s, o) : Promise.resolve(indexed.pageRank(s, o))

// @graphty/layout (L1)
export interface LayoutAccelerator {
    forceAtlas2?(o: ForceAtlas2SimulationOptions): LayoutSimulation;
    fruchtermanReingold?(o: FrSimulationOptions): LayoutSimulation;
}
```

| Package | Change | Landing |
| --- | --- | --- |
| `@graphty/algorithms` | export `AlgorithmAccelerator`, `accelerated(acc)`; result types use `NumericVector` for scores so `F32` and `F64` satisfy one interface; sync `indexed.*` and legacy facades untouched | A2 |
| `@graphty/layout` | export `LayoutAccelerator`, `LayoutSimulation` (design 14.3), the steppable CPU FA2 / FR (`createForceAtlas2Simulation`) following section 7.2 / 7.11 conventions; `ForceAtlas2SimulationOptions` superset of the legacy signature | L1 |
| `@graphty/graphty-element` | `accelerator: (AlgorithmAccelerator & LayoutAccelerator & { release(s): void }) \| null` property + `GraphBehavior` config key + `Graph` constructor option; adapters call `accelerated(this.graph.accelerator).x(s, o)` then the shared result-writing loop (design 14.4 M7); `snapshot-replaced` listener calls `accelerator?.release(previous)`; `ForceAtlas2LayoutEngine` / `SpringLayoutEngine` create `accelerator?.forceAtlas2?.(opts) ?? cpuSimulation(opts)` and drive it through the `LayoutStepper` protocol (7.18); a "GPU: on/off" indicator reads `accelerator !== null` | E1 (+ this plan's P4 for the stepper bridge) |
| graphty app | detection: `const p = await GpuContext.probe({ gpu: navigator.gpu }); if (p.ok && !p.software) { const { GpuContext } = await import("@graphty/webgpu-graph-algorithms"); element.accelerator = (await GpuContext.create({ gpu: navigator.gpu })).accelerator(); }` -- the app owns the bundle, the code-split and the user toggle | after E1 + W1 |
| `@graphty/webgpu-graph-algorithms` | `ctx.accelerator()` implements both interfaces; `@graphty/algorithms` and `@graphty/layout` are `devDependencies` only, for the type-conformance test `expectTypeOf(ctx.accelerator()).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>()`; no runtime import of either (acyclic) | W1 |

Rejected: a registry (`registerAccelerator`) -- inverted dependency, side-effect module
defeats tree-shaking, global state breaks with duplicate copies, and it puts the "GPU
threw, what now?" decision in the CPU package (the forbidden fallback temptation);
optional-peer dynamic import inside algorithms / layout -- WebGPU knowledge in
packages the design says have none, and Node detection would need Dawn (note 02
sections 4.2-4.3). An element-level `accelerator: "auto"` convenience (the web-llm
isolation pattern: separate module, specifier externalised in vite, Safari-safe) is a
later option, never on the critical path.

### 9.3 Result-shape parity

| Concern | Rule |
| --- | --- |
| scores | GPU `F32`, CPU `F64`; interface type `NumericVector`; the element's `*Pct` normalisation is an O(n) CPU pass over either |
| labels / parents / predArc | `U32` with `INVALID_INDEX`; labels dense 0..count-1 via `renumberPartition` in first-seen order (identical `groups()`) |
| BFS | `depth` exact; `parent` level-consistent, not FIFO-identical; `order` grouped by level |
| SSSP | `dist` within `1e-5`; `predArc` ties differ |
| SCC (if ever) | set equality only; Tarjan's label order is not reproducible |
| per-edge results | `F32(edgeCount)` via `foldArcs`; the element writes through `edgeRemap` as for `indexed.*` |
| errors | a throw from an accelerator method propagates to `runAlgorithm`'s operation queue and is reported as a failed run; the element may offer "disable accelerator" as a USER action (not a fallback) |

### 9.4 Timing against the landing order

Until A2 / L1 exist, the GPU package develops against `@graphty/graph-format` alone with
its own CPU references (`test/helpers/oracle.ts`, from the design's Port 1-6 code) and
declares the accelerator interfaces STRUCTURALLY in `src/types/accelerator.ts`; W1
replaces the structural copies with `import type` from the CPU packages (devDependency)
and switches the differential suites to `indexed.*`. The element's stepper bridge (7.18)
can be prototyped against the CPU steppable FA2 in L1 before any GPU code lands, which
de-risks the frame-loop semantics early (P4 has a CPU-only fallback for its OWN tests;
that is a test of the element, not a fallback in the GPU package).

---------------------------------------------------------------------------

## 10. Performance targets and memory model

All numbers for an UNDIRECTED weighted snapshot with average degree 10 (`A = 2E`);
MEASURED / PUBLISHED / ESTIMATE as labelled; the benchmark harness (section 11.6)
replaces every ESTIMATE before the package is published.

### 10.1 Device memory

| n / E | Hot prefix (rowPtr + colIdx + weights) | FA2 exact scratch (40 B/node) | FA2 grid scratch (56 B/node + pyramid) | PageRank scratch | BFS scratch | BC batch (k sources) | Fits default 256 MiB `maxBufferSize`? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 1.6 MB | 0.4 MB | -- (exact) | 0.16 MB | 0.16 MB | k = 1024: 82 MB | yes |
| 100k / 1M | 16.4 MB | 4 MB | 5.6 MB + 7 MB (2D) / 48 MB (3D) | 1.6 MB | 1.6 MB | k = 256: 205 MB | yes (single arena buffer) |
| 1M / 10M | 164 MB | 40 MB | 56 MB + 7 MB / 48 MB | 16 MB | 16 MB | k = 32: 256 MB (over budget: k = 24) | arena yes; per-segment bindings 80 MB < 128 MiB yes |
| 10M / 100M | 1.64 GB | 400 MB | 560 MB + 7 / 48 MB | 160 MB | 160 MB | k = 2 | no: raised limits (4070: 2-4 GiB) or windowed at 64-arc boundaries; Node batch only (`arena: false` on the CPU side, design 15.3) |

### 10.2 Upload

| n / E | Bytes | Time (ESTIMATE: `writeBuffer` is a CPU memcpy into a staging ring plus a PCIe copy; budget 3-6 GB/s effective) | Notes |
| --- | --- | --- | --- |
| 10k / 100k | 1.6 MB | < 1 ms | one `writeBuffer` |
| 100k / 1M | 16.4 MB | 3-6 ms | one `writeBuffer` of the hot prefix; the format's freeze is 22 / 48 ms (MEASURED, `packages/STATUS.md`), so the upload is a fraction of the re-freeze |
| 1M / 10M | 164 MB | 30-60 ms | once per snapshot; cached by `GraphResidency` |
| 10M / 100M | 1.64 GB | 0.3-0.6 s | windowed; batch |

MEASURED anchor: 1M-element compute + 4 MiB readback = submit 0.40 ms + `onSubmittedWorkDone`
0.17 ms on the 4070 under Dawn-node (note 05 section 2.4); the graph-format audit uploads
the 100k / 1M arena on every run in well under its 30 s test budget (not timed
separately -- listed as an ESTIMATE target, measured in P1).

### 10.3 Per-iteration / per-run time on the 4070 SUPER (Dawn-node; Chromium adds ~0.1 ms per submit)

| n / E | FA2 iteration (7.19) | PageRank iteration | PageRank run (60 iterations, readback every 8) | BFS (avg degree 10, diameter ~10) | WCC (Afforest) | BC (k = 64 sources) |
| --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 0.6 ms (exact) | ~0.05 ms (dispatch-bound) | ~5 ms incl. 8 readbacks | ~1 ms (10 levels x 2 dispatches + 1 readback) | ~2 ms | ~50 ms |
| 100k / 1M | 4-9 ms (grid) | 0.1-0.7 ms (MEASURED gather) | 10-45 ms | 2-5 ms | 5-10 ms | 0.3-0.6 s |
| 1M / 10M | 8-25 ms (grid) | 1-7 ms | 0.1-0.5 s | 20-60 ms | 30-100 ms | 3-6 s (sampled; exact BC at 1M is hours on any device) |
| 10M / 100M | 100-300 ms (grid, windowed) | 10-70 ms | 1-5 s | 0.3-1 s | 0.5-2 s | batch only |

CPU references for the speedup claims (MEASURED in the design, line 76 / section 15.4):
`toCSRGraph` 2,376 ms vs freeze 20 ms; the legacy `Graph` build 650 ms / 243 MB at
100k / 1M. The GPU wins are therefore dominated at 100k by removing conversion cost
(already done by the format) and at 1M+ by the kernels themselves; the plan does NOT
claim a per-algorithm speedup number until the harness produces one.

### 10.4 Readback

| n | Positions (12n) Chromium / Node | Scores (4n) Chromium / Node | Basis |
| --- | --- | --- | --- |
| 10k | 0.3 / 0.05 ms | 0.1 / 0.04 ms | MEASURED 4-byte round trip 0.10 / 0.04 ms; 1 MiB 2.65 ms Chromium |
| 100k | 3 / 0.4 ms | 1 / 0.15 ms | linear ESTIMATE |
| 1M | 30 / 4 ms | 10 / 1.3 ms | linear ESTIMATE; browser frame loop must batch (`readbackEvery`) |
| 10M | 300 / 40 ms | 100 / 13 ms | batch only |

### 10.5 Targets the phases must meet (gates in section 13)

| Target | Value | Where measured |
| --- | --- | --- |
| T1 walking skeleton: upload 100k / 1M hot prefix + degree kernel + readback + release | < 20 ms end to end on the 4070; correct vs `outDegree()` | P1 |
| T2 FA2 exact, 10k nodes / 100k arcs, 3D | <= 1 ms per iteration on the 4070; 60 fps with `stepMultiplier = 5` through the element | P3 / P4 |
| T3 FA2 exact, 16k nodes | <= 2 ms per iteration on the 4070 | P3 |
| T4 FA2 grid, 100k / 1M, 2D and 3D | <= 10 ms per iteration on the 4070; forces within 5% RMS of exact on the fixtures | P5 |
| T5 FA2 grid, 1M / 10M, 2D | <= 30 ms per iteration on the 4070 in Node | P5 |
| T6 PageRank 100k / 1M, 60 iterations | <= 50 ms on the 4070 incl. readbacks; `1e-5` parity | P7 |
| T7 BFS 1M / 10M | <= 100 ms on the 4070; exact depth parity | P9 |
| T8 lavapipe CI lane | full `node` project < 10 min on 4 vCPUs | every phase |

---------------------------------------------------------------------------

## 11. Testing strategy

### 11.1 Principles

- One set of test files; vitest projects select subsets (note 05 section 9.1; note 06
  section 4.1). `node` on Dawn is PRIMARY and carries coverage; `browser-smoke` proves
  the same code runs in a real browser; `node-limits` and `bench` run only where real
  limits / real hardware exist.
- A wrong result is never a skip (graph-format's rule, `gpu-upload.test.ts` line 16).
  Tests skip with a printed `E_NO_ADAPTER` only when no adapter exists at all and
  `GRAPHTY_REQUIRE_GPU` is unset; under `GRAPHTY_REQUIRE_GPU=1` that is a failure; under
  `GRAPHTY_GPU_REQUIRE=nvidia` a software adapter is a failure (so a broken driver mount
  can never turn into a silently green lavapipe run -- the exact local failure mode of
  `HEADLESS_GPU_REPORT.md`).
- `test/setup/gpu.ts` acquires once per worker (`pool: "forks"`, verified by
  graph-format's 1,309-test suite; `threads` is untested with the Dawn addon), prints
  `adapter.info` and the four limits, installs the `uncapturederror` -> test failure
  hook, sets `XDG_RUNTIME_DIR`, honours `GRAPHTY_GPU_ADAPTER` (`adapter=<v>`) and
  `GRAPHTY_DAWN_FEATURES`, drops the `GPU` reference in a global `afterAll` (note 05
  section 9.4).
- Every fixture size and iteration count is multiplied by `gpuScale()` = 1 on hardware,
  1/50 on a software adapter (lavapipe is ~350x slower on O(n^2) and ~10-50x on gathers;
  MEASURED, note 05 section 2.5 / note 06 section 3.5).
- Every planner (upload, dispatch, tile size, batch size) is a pure function of
  `GpuCaps` + byte lengths and is unit-tested WITHOUT a device against faked caps
  (spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-Dawn-like, NVIDIA-Chromium-like
  tables from note 05 section 4), so the windowed and 2D branches are covered on the
  default lane.

### 11.2 Test layers

| Layer | What | Reference | Project |
| --- | --- | --- | --- |
| device | probe / create / from; error codes; `isSoftwareAdapter`; uncaptured error hook; device loss (`device.destroy()` mid-run rejects with `E_DEVICE_LOST`, deterministic in both runtimes); globals-independent constants | -- | node, browser-smoke |
| memory | arena hot-prefix upload with per-segment bindings equals CPU views; per-array path (`fromCsr` on separate arrays, `transpose()`); windowed path at 64-arc boundaries equals a copied window; `arena.byteOffset !== 0` (`fromBytes` at offset 8); column uploads (u8 packed, bool bits, f64 convert + `markDirty` invalidation); `release` destroys every buffer incl. after `dropCaches()`; BufferPool reuse; StagingRing overlap | the graph-format audit assertions (note 07 section 6) copied, plus new ones | node (+ one in browser-smoke) |
| kernel | pipeline cache keys include overrides; `plan1D` boundary at 16,776,960 (both sides); 2D dispatch on a 20M-item map equals a CPU map (`node-limits` for the real thing, faked caps in `node`); indirect finalize clamps; uniform packing round-trips; compilation errors surface with line numbers | CPU maps | node |
| primitives | each primitive vs its oracle on random sizes incl. 0, 1, 255, 256, 257, 65,535 x 256 boundaries, and on all three adapters' subgroup sizes (4 / 8 / 32: SwiftShader, lavapipe, NVIDIA) -- the "never assume the subgroup size" bug class | `test/helpers/oracle.ts` | node, browser-smoke (subgroup variant only if the feature exists) |
| layouts | section 7.20: kernel unit tests, properties, differential traces, exact-vs-grid, frame-loop bridge | CPU FA2 reference (oracle now, `@graphty/layout` steppable FA2 at W1) | node, browser-smoke (one end-to-end) |
| algorithms | differential vs oracle / `indexed.*` on karate, grids, seeded G(n, m) with self-loops and parallels, star graphs (hub tiers), path graphs (high diameter: exercises k-levels-per-submit), disconnected graphs, empty graph, single node, `arcCount === 0`; property tests (fast-check, `numRuns` 100 default / 1,000 nightly): PageRank sums to 1, BFS depth triangle inequality, CC labels form a partition equal to union-find, SSSP `dist[v] <= dist[u] + w` for every arc, BC symmetric on symmetric graphs | oracle / `indexed.*` | node |
| invariants | after every algorithm call the snapshot's views are unchanged (`validate({ checksum: true })`, design 16.2) and no uncaptured error occurred | -- | node |
| exact vs approximate | grid repulsion vs exact forces (5% RMS), distributional end-state metrics, hub-cell stress fixtures | exact kernel | node |
| types | `expectTypeOf` conformance to `AlgorithmAccelerator & LayoutAccelerator` (W1), result arrays are `Float32Array<ArrayBuffer>` / `Uint32Array<ArrayBuffer>` accepted by `writeBuffer` without a cast; strict-consumer compile of the public d.ts | -- | typecheck |
| packaging | barrel export list pinned; `build-output.test.ts` (exports map incl. `./node` and `./browser`, no `webgpu` import reachable from the root bundle -- grep the built `dist/webgpu-graph-algorithms.js`) | -- | node |

### 11.3 Browser smoke (light, by design)

`test/browser/*.test.ts` on Playwright Chromium (`browser.instances[0].launch.args`
carrying the flag set chosen by `GRAPHTY_BROWSER_GPU`: `swiftshader` =
`--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader`; `nvidia`
= `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan
--disable-vulkan-surface`; the exact Vitest 3.2 spelling is confirmed when the config
is written, note 07 unverified item 1):

1. `requestGpuContext()` returns a context; `adapter.info.vendor === "nvidia"` and
   `isFallbackAdapter === false` under `GRAPHTY_GPU_REQUIRE=nvidia`.
2. upload + degree kernel + readback + release on the karate club.
3. one PageRank (P7+), one BFS (P9+), one connected components (P8+) on a 1k-node
   fixture, compared with the oracle.
4. FA2 `load / requestStep(10) x 30 / settled / dispose` on karate with a fake render
   loop; `setPosition` visible.
5. subgroup-variant primitive test when `features.has("subgroups")` (true on
   SwiftShader, size 4).

`fileParallelism: false` (one GPU-bearing browser at a time); job-level timeout as the
backstop for the known `browser.close()` hang after GPU work on the NVIDIA path
(reproduced; three.js SIGKILLs for the same reason; note 06 section 3.5).

### 11.4 What the walking skeleton proves (P1 gate)

On BOTH runtimes: a context is created from a caller-provided `GPU`; the 100k / 1M
undirected weighted snapshot's hot prefix uploads through the arena path with
per-segment bindings; a `degree` kernel (`rowPtr[v + 1] - rowPtr[v]`) runs with the
planner (including a forced 2D dispatch on a synthetic 17M-item map); the result reads
back into a caller-supplied `Uint32Array` equal to `outDegree()`; `release(snapshot)`
destroys the buffers (a wrapped `createBuffer` / `destroy` counter returns to zero);
`dispose()` leaves no uncaptured errors and lets the Node process exit; the same test
file passes on NVIDIA, lavapipe and SwiftShader with identical checksums.

### 11.5 Coverage

Thresholds 80 / 80 / 75 / 80 measured by the `node` project on the default lane
(lavapipe), `src/wgsl/**` excluded; the browser project carries no coverage; the GPU
lane uploads no lcov (note 06 section 4.4). Device-limit branches are covered by the
faked-caps unit tests.

### 11.6 Benchmarks

`benchmarks/` copies graph-format's harness (`bench`, `printTable`, `appendSession`,
seeded `datasets.ts`; note 07 section 6) with an async `run` and a `gpu` field (vendor,
architecture, requested limits) in the session record. Groups: `primitives`
(reduce, scan, sort at 2^16-2^22), `layouts` (FA2 exact 4k / 8k / 16k / 32k; grid 32k /
100k / 300k / 1M in 2D and 3D; FR; per-kernel breakdown via timestamp queries),
`algorithms` (PageRank, BFS, WCC, BC-sampled at 100k / 1M and 1M / 10M). The GPU lane
runs `vitest bench --project=bench` and uploads JSON with 90-day retention; a checked-in
baseline per runner class gates regressions at 3x (the design's rule for freeze,
section 15.5) in an opt-in `perf` run.

---------------------------------------------------------------------------

## 12. CI/CD

### 12.1 Lanes

| Lane | Runner | Adapter | Runs | Trigger | Required check |
| --- | --- | --- | --- | --- | --- |
| default | `ubuntu-latest` (GitHub-hosted, 4 vCPU) | Dawn-node on Mesa lavapipe (`apt-get install mesa-vulkan-drivers libvulkan1`, `GRAPHTY_GPU_ADAPTER=llvmpipe`); Chromium on SwiftShader | build, lint, typecheck, strict-consumer compile, `node` project with coverage, `browser-smoke` | every push / PR | YES |
| gpu | self-hosted ephemeral runner on the dev box (RTX 4070 SUPER, driver 580.173.02), labels `[self-hosted, linux, x64, gpu, nvidia]` | NVIDIA via Vulkan (`GRAPHTY_GPU_REQUIRE=nvidia`; `libegl1` in the runner image, else `LD_LIBRARY_PATH` to the extracted tree per `HEADLESS_GPU_REPORT.md` appendix D) | `node` + `node-limits`, `bench` (JSON artifact), `browser-smoke` with the NVIDIA flags, `gpu-report.json` | push to master, nightly `schedule`, `workflow_dispatch`, same-repo PRs labelled `gpu` | NO (a powered-off dev box must never block merges) |

Why not GitHub's T4 runners: larger runners require GitHub Team / Enterprise Cloud and
`gh api /orgs/graphty-org` reports `plan: free`; they are also not free for public repos
($0.052/min Linux; note 06 sections 1 and 3.2, verified 2026-09-14). Third-party
fallbacks if the dev box proves unreliable: Cirun.io (free for public repos, runners in
your own cloud, `gpu: nvidia-tesla-t4`), RunsOn (own AWS account, `g4dn.xlarge`,
`ubuntu24-gpu-x64`), machine.dev (`runs-on: machine/gpu=t4`); all need a cloud account
(note 06 section 3.4).

Cost: default lane $0 (public repo); GPU lane $0 (self-hosted usage in public repos
stays free after the March 2026 pricing change; note 06 section 3.1) plus electricity.

Security (public repos + self-hosted runner; GitHub's guidance "should almost never be
used for public repositories" applies): ephemeral registration (`config.sh --ephemeral
--disableupdate` loop or `generate-jitconfig`), the `if:` guard `head.repo.full_name ==
github.repository && contains(labels, 'gpu')` so fork PRs can never reach it, repository
setting "Require approval for all external contributors", `permissions: contents:
read`, no secrets in the GPU job, the runner in its own sibling container (not the dev
workspace) started on the host with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`
(the toolkit injects the Vulkan ICD only with that capability; `HEADLESS_GPU_REPORT.md`
line 32), `concurrency: { group: gpu-runner }` so one job holds the GPU at a time
(note 06 sections 3.3, 4.3, 5.1). cuda-ffi's precedent: `runs-on: cudaffi-gpu-runner` +
`container: { image, options: "--gpus all --user root" }` triggered on push /
`workflow_dispatch` only (`tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`);
this plan adds the label-gated same-repo PR path and a real GPU-free lane, which
cuda-ffi never had (note 06 section 2).

### 12.2 Workflow sketch (this repo, standalone; pnpm as in the staging root)

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
    test:                                   # default lane: required
        if: github.event_name != 'schedule'
        runs-on: ubuntu-latest
        timeout-minutes: 30
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm }
            - run: pnpm install --frozen-lockfile
            - name: Mesa lavapipe (software Vulkan ICD for Dawn-in-Node)
              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
            - run: pnpm -r run build && pnpm -r run lint && pnpm -r run typecheck:strict-consumer
            - name: Node suite on lavapipe (+ coverage)
              env: { GRAPHTY_GPU_ADAPTER: llvmpipe, VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json }
              run: cd packages/webgpu-graph-algorithms && pnpm exec vitest run --project=node --coverage
            - uses: actions/cache@v4
              id: pw
              with: { path: ~/.cache/ms-playwright, key: "playwright-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}" }
            - run: pnpm exec playwright install chromium --with-deps
              if: steps.pw.outputs.cache-hit != 'true'
            - name: Browser smoke on SwiftShader
              env: { GRAPHTY_BROWSER_GPU: swiftshader }
              run: cd packages/webgpu-graph-algorithms && pnpm exec vitest run --project=browser-smoke
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: coverage-webgpu-graph-algorithms, path: packages/webgpu-graph-algorithms/coverage/lcov.info, retention-days: 1, if-no-files-found: error }

    test-gpu:                               # GPU lane: never required
        if: >-
            github.event_name == 'push' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' ||
            (github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository &&
             contains(github.event.pull_request.labels.*.name, 'gpu'))
        runs-on: [self-hosted, linux, x64, gpu, nvidia]
        timeout-minutes: 45
        concurrency: { group: gpu-runner, cancel-in-progress: false }
        env: { GRAPHTY_GPU_REQUIRE: nvidia, GRAPHTY_BROWSER_GPU: nvidia }
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x }
            - run: pnpm install --frozen-lockfile && pnpm -r run build
            - run: cd packages/webgpu-graph-algorithms && node scripts/gpu-report.mjs | tee gpu-report.json
            - run: cd packages/webgpu-graph-algorithms && pnpm exec vitest run --project=node --project=node-limits
            - run: cd packages/webgpu-graph-algorithms && pnpm exec vitest bench --project=bench --outputJson bench/results.json
            - run: cd packages/webgpu-graph-algorithms && pnpm exec vitest run --project=browser-smoke
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: "gpu-results-${{ github.run_id }}", path: "packages/webgpu-graph-algorithms/{gpu-report.json,bench/results.json}", retention-days: 90 }
```

`scripts/gpu-report.mjs` prints adapter info, features, limits, subgroup sizes and the
4-byte round-trip latency (the probe scripts of `tmp/webgpu-plan/probe/` consolidated)
and is the first step of every GPU job so a driver regression is visible in the log.

### 12.3 Self-hosted runner recipe (host side, not committed to the package)

Sibling container from `ghcr.io/actions/actions-runner` with `libegl1 libvulkan1
mesa-vulkan-drivers` and Playwright's Chromium deps; started with `docker run --gpus
all -e NVIDIA_DRIVER_CAPABILITIES=all`; entry loop: fetch a registration token (`POST
/repos/{owner}/{repo}/actions/runners/registration-token`), `./config.sh --unattended
--ephemeral --disableupdate --labels gpu,nvidia --name devbox-$RANDOM`, `./run.sh`,
wipe `_work`, repeat (or `generate-jitconfig` + `./run.sh --jitconfig`). The
registration credential (fine-grained PAT with `Administration: write`, or a GitHub
App) lives only on the host loop (note 06 sections 3.3 and 5.1). Until the image has
`libegl1` the job exports `LD_LIBRARY_PATH` to the extracted tree exactly as the local
recipe does.

### 12.4 Monorepo integration at W1

Mirrors `packages/move/root-touch-points.diff` (note 06 section 6; note 07 section 4.7):
build-once upload of `build-webgpu-graph-algorithms`; two new shards in the `test`
matrix (`webgpu-graph-algorithms-node` with a `needs-vulkan: true` apt step and the
`GRAPHTY_GPU_ADAPTER=llvmpipe` env, `webgpu-graph-algorithms-browser` on SwiftShader
with `needs-browser: true`); the coverage-upload condition extended with
`startsWith(matrix.shard, 'webgpu-graph-algorithms-')`; `tools/merge-coverage.sh`
PACKAGES, `tools/prepush.sh` (node project only; `GRAPHTY_GPU_ADAPTER=llvmpipe` there
to match CI, or the local NVIDIA adapter), `pnpm-workspace.yaml`, `commitlint` scope,
`knip.config.ts`, root `package.json` `coverage:preview` on port 9058, `release.yml`
download step; plus the separate `test-gpu` job above with `needs: build`, downloading
`build-graph-format` and `build-webgpu-graph-algorithms`. The existing matrix has no
`nx affected` gating, so the two shards simply join it; the GPU job is gated by event +
label, not by paths (a graph-format change is exactly when it should run). It is
excluded from the `all-checks` gate (`ci.yml` lines 705-738). `release.yml` (OIDC
trusted publishing, gated on CI success) needs only the download step; the GPU package
publishes with provenance like its siblings; the `webgpu` optional peer keeps npm from
installing Dawn for browser consumers.

### 12.5 Version pins to keep straight

`webgpu@0.4.0` in both lanes (glibc 2.34 binary; 0.6.1 needs 2.38; the dev box is
22.04 / 2.35, `ubuntu-24.04` is 2.39): bump once, everywhere, when the dev container
and the runner image move to 24.04. `@webgpu/types ^0.1.72`. Playwright `^1.54.1`
(Chromium 139 build 1181, verified on the NVIDIA path). Vitest / `@vitest/browser`
`^3.2.4` with the monorepo's overrides; the GPU flag sets live in one exported constant
so the Vitest 4 provider change (`@vitest/browser-playwright`, `launchOptions`) is a
one-line move (note 05 section 9.3).

---------------------------------------------------------------------------

## 13. Phased implementation plan

Sizes: S = 1-2 days, M = 3-5 days, L = 1-2 weeks of focused work (single engineer),
with the caveat that the plan's ESTIMATE numbers are measured, not assumed, at each gate.
The force-directed layout is the first product deliverable after the walking skeleton
(P3), as the owner asked; algorithms follow.

| Phase | Scope | Deliverables | Gate (must be green before the next phase) | Size |
| --- | --- | --- | --- | --- |
| P0 scaffold reset | apply note 07 section 5: delete the July-2025 scaffold (`src/types/index.ts` `CSRGraph`, `src/index.ts`, `test/setup/*`, `test/helpers/*`, `vitest.config.ts`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `knip.json`, `.husky`, `package-lock.json`, `.env*`, `.github/workflows/test.yml`, `.vscode`, `examples`, `STRATEGY.md`, `IMPLEMENTATION_CHECKLIST.md`); create `packages/webgpu-graph-algorithms/` mirroring graph-io (package.json of note 07 section 4.2, project.json, tsconfig trio, vitest config with `node` / `node-limits` / `bench` / `browser-smoke` projects, scripts, CLAUDE.md, README); keep `HEADLESS_GPU_REPORT.md` under `docs/`; `.github/workflows/ci.yml` of section 12.2 (GPU lane job present but the runner not yet registered) | an empty package that builds, lints, typechecks and runs a trivial test on all three adapters; the default lane green on GitHub | `pnpm -r run build && lint && test` green; CI default lane green | S |
| P1 walking skeleton | `GpuContext` (probe / create / from / dispose / lost), `GpuCaps`, error class, usage constants, `test/setup/gpu.ts`, `GraphResidency` arena hot-prefix path only, `StagingRing`, `PipelineCache` + `Kernel`, `plan1D` incl. 2D, the `degree` kernel, `release`, `./node` and `./browser` entries, `scripts/gpu-report.mjs`, `benchmarks/` harness copy | section 11.4 proof on NVIDIA, lavapipe, SwiftShader; upload + kernel + readback of 100k / 1M measured (T1) | skeleton test green in `node` (NVIDIA + lavapipe) and `browser-smoke` (SwiftShader + NVIDIA); T1 <= 20 ms; no leaked buffers; coverage wiring works | M |
| P2 memory + dispatch infrastructure | per-array and windowed upload paths, `packViews`, column uploads with `column.version`, `BufferPool`, uniform packer, indirect-dispatch finalize, timestamp profiler, faked-caps planner tests (spec / SwiftShader / lavapipe / NVIDIA tables), `node-limits` tests (2D dispatch on 17M items, > 128 MiB binding), primitives `reduce` (multi-channel) and `segmentedReduce` (thread tier) | the memory layer of section 4 complete with tests; primitives that P3 needs | all section 4 / 5 tests green on the default lane; `node-limits` green on the GPU lane; coverage >= thresholds | M |
| P3 FA2 exact tier as `LayoutSimulation` | `ForceAtlas2Simulation` (exact repulsion, attraction thread-per-row, adaptSpeed, integrate, finalize), `load` semantics (seeding LCG, mass, masks), `setFixed` / `setPosition` / `reheat`, `LayoutStepper` (`step` + `requestStep`), `stats()`, options of 7.14 with `compat`, kernel unit tests, properties, differential trace vs `test/helpers/oracle-fa2.ts` (a faithful CPU port of section 7.2's adopted formulas, f64), frame-loop bridge test, FA2 benchmark at 1k / 4k / 8k / 16k / 32k on NVIDIA and lavapipe | a usable GPU FA2 in Node; the browser end-to-end smoke | T2 and T3 met; differential traces within tolerance; properties green on all three adapters; benchmark JSON committed under `benchmarks/results/` | L |
| P4 element bridge (in the monorepo, alongside L1 / E1) | `LayoutSimulation` + steppable CPU FA2 in `@graphty/layout` with the section 7.2 / 7.11 conventions (the parity oracle, replacing `oracle-fa2.ts` at W1); `LayoutAccelerator`; element `accelerator` property; `ForceAtlas2LayoutEngine` driving a simulation through the stepper protocol; a Storybook story "ForceAtlas2 (GPU)" gated on `navigator.gpu`; Chromatic re-baseline commit for the documented reasons | the owner can open the story on the dev box and see the GPU layout animate with drag and pins | element tests + story green; drag / pin / settle / zoom-to-fit behave; the same story on the CPU simulation looks statistically the same | M (element side) |
| P5 scale: grid repulsion + degree tiers | histogram, scan, radix sort primitives; `bbox`; grid pyramid build and force kernels (2D + 3D), `repulsion: "auto"`, `calibrate()`; tiered attraction from `degreeOrder()`; exact-vs-grid tests; hub-cell stress fixtures; benchmarks 32k / 100k / 300k / 1M in 2D and 3D; the `exactMaxNodes` default fixed from measurements | FA2 usable at 10^5-10^6 nodes | T4, T5 met on the 4070; grid forces within 5% RMS of exact; deterministic across two runs on one device; lavapipe correctness at scaled sizes | L |
| P6 FR + presets | `FruchtermanReingoldSimulation` on the same skeleton (temperature, `fixed`), the `spring-electrical` preset with a velocity integrator and ngraph-like settle rule, `LayoutAccelerator.fruchtermanReingold` | second layout; the element can route `spring` (and optionally `ngraph` above a threshold) to the GPU | FR tests and benchmarks green; element story | M |
| P7 PageRank family | pull SpMV with the three tiers by in-degree, device out-weight sums (`scratch`), dangling / delta reduces, k = 8 batching, personalized, HITS, eigenvector, Katz; `ctx.accelerator()` object with `release`; differential vs oracle (`indexed.pageRank` at W1) | first algorithms through the accelerator interface | T6; `1e-5` parity; top-k order; browser smoke PageRank | M |
| P8 connected components | Afforest (link / compress / sample / remaining), bitset + compaction, `renumberPartition` on readback, directed and undirected fixtures | WCC | set-equality parity; identical `groups()` after renumbering; browser smoke CC | S-M |
| P9 traversal family | `Frontier`, `advance` (block_mapped + hub tier + fused small-frontier), dedupe, indirect dispatch loop with k = 16 levels, direction-optimizing switch (cuGraph constants), BFS; batched multi-source BFS -> closeness / harmonic / eccentricity; near-far SSSP with `atomicMin` on f32 bits and the two-pass predecessor; Bellman-Ford CAS relax | BFS, closeness, SSSP, Bellman-Ford | T7; parity rules of 9.3; a 10k-level path graph runs faster than the oracle (the k-levels-per-submit proof) | L |
| P10 betweenness + APSP | tagged multi-source forward pass, successor-pull backward pass, sigma overflow flag, hybrid switch, sampled sources, edge BC via `foldArcs`; blocked Floyd-Warshall and BFS-based APSP with the `maxBufferSize` bound | BC (sampled and exact for small n), APSP | `1e-4` parity on karate / grids / small random graphs; sampled BC top-k agreement; APSP exact unweighted | L |
| P11 remaining kernels | label propagation, k-core, Boruvka MST, triangle / common neighbours / Adamic-Adar / k-truss, COO -> CSR, Louvain (then Leiden) | the long tail, in demand order | per-algorithm parity rules; Louvain modularity within a band | L (Louvain alone is M-L) |
| W1 move-in | `git mv` into `graphty-monorepo/webgpu-graph-algorithms/`; root touch points (section 12.4); replace the structural accelerator interfaces with `import type` from A2 / L1; switch differential tests to `indexed.*`; register the self-hosted runner for the monorepo; amend design 14.5 / 16.7 | the package in the monorepo, both CI lanes live | monorepo default lane green with the two new shards; GPU lane green once on master; `nx release` dry run | M |

Ordering notes: P4 depends on L1 / E1 progress in the monorepo and can start as soon as
P3's `LayoutStepper` protocol is fixed (the element side is testable against the CPU
steppable FA2 first). P5 is the phase where the "high performance for a large number of
nodes" driver is met; it is scheduled before any algorithm because the owner's first
need is the layout at scale. P7 onward can interleave with element work; each phase
ends with a benchmark JSON commit and a README table update from the harness.

---------------------------------------------------------------------------

## 14. Risks and open questions for the owner

Each item has a recommended default the plan assumes unless the owner says otherwise.

1. FA2 reference formulas (section 7.2, decision D9). The CPU port differs from the
   published algorithm in the repulsion law, swing / traction definition and size
   correction. Default: adopt the paper / Gephi / cuGraph definitions in the L1 rewrite
   and the GPU kernel, keep the port's variants behind `compat: "port"`, and re-baseline
   Chromatic in the L1 commit that already re-baselines for other reasons. Alternative:
   keep the port's formulas as the reference (no visual change on the CPU path; the GPU
   still matches it), at the cost of diverging from every other FA2 implementation and
   losing cuGraph as a cross-check.
2. Units of the steppable FA2 (decision D8). Default: scene units, Gephi semantics,
   `scalingFactor` = seeding radius, no per-step rescale; documented behaviour change for
   the "forceatlas2" element type (which today is one-shot and normalised). Alternative:
   layout units on the device with `scalingFactor` applied in the write-back kernel;
   keeps the one-shot constants but makes drag / pin coordinates round-trip through a
   scale and does not solve the equilibrium-size question.
3. `exactMaxNodes` default (section 7.8). Default 16,384, `calibrate()` optional, fixed
   in P5 from measurements; the MEASURED 3.6 x 10^11 pairs/s suggests 32k is viable on
   the 4070 but integrated GPUs argue for the lower default.
4. Deterministic grid tier (decision D12). Default: stable radix sort (bitwise
   reproducible per device, ~1-2 ms at 1M nodes, ESTIMATE). Alternative: atomic-cursor
   counting sort, possibly faster, nondeterministic within cells; keep behind
   `deterministic: false` only if P5 measures a real gain.
5. `webgpu@0.4.0` pin versus the container image. Default: stay on 0.4.0 in both lanes
   until the dev container and the runner image move to Ubuntu 24.04, then bump once.
   The 0.6.x line has the `device.destroy()` unmap shim we otherwise re-implement.
6. Self-hosted GPU runner on a public repo. Default: ephemeral runner in a sibling
   container, label-gated same-repo PRs, "require approval for all external
   contributors", no secrets, never a required check. Alternative: Cirun / RunsOn with a
   cloud account; GitHub T4 runners need a Team plan ($4/user/month) and are billed for
   public repos.
7. Design 14.5 / 16.7 amendment. Default: amend to "Node-first (`node` project on Dawn,
   lavapipe on hosted CI) with a browser smoke project", and add the CI budget (default
   lane < 10 min on lavapipe, GPU lane < 45 min).
8. f32 versus f64. GPU scores are f32; PageRank parity `1e-5`, BC `1e-4`; sigma path
   counts are `u32` with a saturation flag. Default: document the tolerances in the
   accelerator interface docs; expose `precision: "f32"` in results so the element can
   label GPU results. No f64 on the device (WGSL has none).
9. Per-frame readback at 1M nodes in the browser (~30 ms ESTIMATE). Default:
   `readbackEvery` option and the note that the element cannot render 1M per-node meshes
   today anyway; 1M is a Node batch tier until instanced rendering exists. Sharing the
   device with a Babylon `WebGPUEngine` (positions never leave the GPU) is the eventual
   fix and is why `GpuContext.from(device)` exists.
10. `subgroups` availability on Firefox / Safari is unverified (note 05 section 12).
    Default: every subgroup kernel has a workgroup-memory twin; the feature is only a
    fast path. No behaviour depends on it.
11. 3D grid memory (48 MB at 128^3). Default `gridMax3D = 128`; lower to 64 on
    integrated GPUs via the option or `calibrate()`. Alternative: the Hilbert cluster
    tree (GraphWaGu 2025) for 3D if the hub-cell measurements are bad -- not scheduled.
12. Routing the element's default `ngraph` layout to the GPU above a node count
    (product decision). Default: reserve the `spring-electrical` preset (P6) and let the
    app decide the threshold; do not change the element default.
13. Louvain expectations: 2-10x over the CPU package, not 100x (nu-Louvain's finding);
    the GPU runs the small levels too. Default: schedule last (P11) and ship it only if
    the measured gain justifies the code.
14. Betweenness at scale is inherently O(n x A). Default: `sources` sampling as the
    primary API, exact BC only for small n, with a progress / cancel hook (an
    `AbortSignal` option on long-running algorithms) -- add `signal?: AbortSignal` to
    every algorithm's options from P7 so the element can cancel.
15. Landing-order dependency: the accelerator interfaces live in A2 / L1. Default: the
    GPU package ships structural copies and its own oracles until W1; P4 (element bridge)
    is scheduled against the CPU steppable FA2 first so the frame-loop semantics are
    settled before the GPU version is wired in.
16. `nodeSize` / `adjustSizes` deferred (sign-suspect on the CPU). Default:
    `E_UNSUPPORTED` when given, until the L1 rewrite settles the formula; then a
    `size` buffer and the `0.1 * speed` / cap-10 apply rule (both already sketched).
17. Named-weight columns for layouts. Default: v1 accepts `weight: true | null`; the
    element passes a pre-expanded per-arc `F32` for a named column (design 14.4 cache);
    documented behaviour change: weights become LIVE for the element's FA2 (inert today).
18. Where the plan directory lives. `tmp/` is not gitignored at the root today (note 07
    section 5). Default: keep `tmp/webgpu-plan/` untracked (add `tmp/` to `.gitignore`)
    and copy the accepted plan to `docs/plan.md` in the package (or to
    `graphty-monorepo/design/webgpu/`) when accepted.
19. Interactive topology change with a live simulation (design 14.4 leaves new-node
    placement open). Default: `load(next, positions)` keeps finite rows and seeds NaN
    rows randomly inside the current bounding box (not the seeding radius) so new nodes
    appear near the graph; a "place at neighbours' centroid" option is a later product
    choice for the element.
20. Mobile / Safari 26 behaviour is unverified for this package (only the platform
    matrix of note 05 section 3.1). Default: no claims; the browser smoke on Chromium
    is the only browser evidence until someone runs the story on a Mac.

---------------------------------------------------------------------------

## 15. References

Local (read-only unless stated):

- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` sections 10 (GPU contract), 14.3-14.6 (layout, element, WebGPU move-in, landing order), 15 (performance and memory), 16 (testing) -- the accepted design every section above cites.
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/01-layout-needs.md` -- CPU force layouts, element frame loop, `LayoutSimulation` gap analysis, FA2 option parity.
- `.../02-algorithm-needs.md` -- algorithm inventory, plug-in mechanisms (a)-(d), result-shape contract, priority scores.
- `.../03-gpu-layout-prior-art.md` -- cosmos.gl, GraphWaGu, cuGraph / Burtscher, d3-force-webgpu, GraphGPU, analytics; force laws, crossovers, grid recommendation.
- `.../04-gpu-algorithms-prior-art.md` -- CUDA -> WebGPU translation table, primitive catalogue, per-algorithm strategies from cuGraph / Gunrock / GAP / papers.
- `.../05-webgpu-platform.md` -- Dawn-node facts and pins, browser status, limits, WGSL constraints, latency measurements, test architecture.
- `.../06-gpu-ci.md` -- runner options, lavapipe / SwiftShader lanes, cuda-ffi analysis, workflow sketches, monorepo diff.
- `.../07-format-api-and-conventions.md` -- the format API as implemented, package skeleton, scaffold triage, conventions.
- `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md` -- headless Chromium on the NVIDIA GPU (flags, `libegl1` root cause).
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/` (`src/snapshot/views.ts` lines 41-45, 568-609, 852, 922; `src/snapshot/derived.ts` line 1155; `src/constants.ts` lines 16, 22, 45; `test/audit/gpu-upload.test.ts`; `package.json`) -- format facts cited.
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/{dawn-perf,dawn-probe,dawn-latency,chromium-latency,bench-node,bench-browser}.mjs` -- the MEASURED numbers (20k-node O(n^2) 1.11 ms NVIDIA / 388 ms lavapipe; CSR gather 0.1-0.7 ms; round trips 0.04 / 0.10 ms; 1 MiB readback 2.65 ms).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/{cosmos,GraphWaGu,cugraph,cugraph-algos,gunrock,gapbs,d3-force-webgpu,GraphGPU,analytics,antv-webgpu-graph,cuda-ffi}` and `papers/` -- cloned sources and extracted paper texts behind the PUBLISHED facts.
- `/home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/forceatlas2.ts` (lines 26-42, 184-230, 233-434), `fruchterman-reingold.ts`; `graphty-element/src/layout/{LayoutEngine,ForceAtlas2LayoutEngine,NGraphLayoutEngine,D3GraphLayoutEngine}.ts`; `graphty-element/src/managers/{LayoutManager,UpdateManager,DataManager}.ts`; `graphty-element/src/algorithms/Algorithm.ts`; `algorithms/src/algorithms/centrality/pagerank.ts`; `.github/workflows/{ci,release}.yml`; `nx.json` -- integration facts cited.

External (URL -- what it was used for):

- https://github.com/cosmosgl/cosmos -- grid pyramid + Monte-Carlo near field, exact path below 4,096 points, per-step timings, failure modes (MIT).
- https://cosmograph.app/examples , https://pypi.org/project/cosmograph/ -- owner-supplied; product pages over cosmos.gl, no additional algorithmic content (not fetched by this plan; note 03).
- https://github.com/harp-lab/GraphWaGu -- WebGPU FR + Barnes-Hut: WGSL radix sort, Hilbert codes, level-wise tree build, i32 fixed-point bounding box, CSR gather attraction (MIT); PacificVis / EGPGV timings.
- https://github.com/jaredmcqueen/analytics -- O(n^2) WebGL1 FR; cautionary only (GPL-3, not copied).
- https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal -- Merrill, Garland, Grimshaw: scan-based frontier BFS, gather tiers, dedupe, couplings.
- https://cse.buffalo.edu/tech-reports/2023-06.pdf -- Kumar MS thesis, dense cuBLAS betweenness; excluded (sparse graphs).
- https://dl.acm.org/doi/10.1145/3230485 (403) and https://davidbader.net/publication/2018-mb/2018-mb.pdf -- McLaughlin, Bader: work-efficient / edge-parallel BC, atomic-free dependency accumulation, sampling.
- https://developer.nvidia.com/discover/cluster-analysis -- nvGRAPH-era spectral / multilevel clustering overview; background only.
- https://github.com/atoms-org/cuda-ffi -- self-hosted GPU runner precedent (`runs-on: cudaffi-gpu-runner`, `--gpus all`).
- https://github.com/rapidsai/cugraph -- FA2 (`cpp/src/layout/legacy/*`), BFS / SSSP / PageRank / BC / Louvain / core number / triangle count implementations and constants (Apache-2.0).
- https://github.com/gunrock/gunrock -- `block_mapped` advance, `neighborreduce`, `from_coo`.
- https://raw.githubusercontent.com/sbeamer/gapbs/master/src/cc.cc -- Afforest connected components.
- https://scottbeamer.net/pubs/beamer-sc2012.pdf -- direction-optimizing BFS constants.
- https://escholarship.org/content/qt8qr166v2/qt8qr166v2.pdf -- Davidson et al. near-far SSSP and ownership dedupe.
- https://arxiv.org/html/2501.19004 , https://arxiv.org/html/2608.01503 -- nu-Louvain / Gilbert-Madduri GPU Louvain findings (secondary sources for Naim et al. 2017, which could not be fetched).
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679 -- ForceAtlas2 paper (CC-BY): force and speed formulas.
- https://raw.githubusercontent.com/gephi/gephi/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/{ForceAtlas2,ForceFactory,Region}.java -- Gephi formulas and defaults (GPL/CDDL; formulas only).
- https://raw.githubusercontent.com/networkx/networkx/main/networkx/drawing/layout.py -- NetworkX `forceatlas2_layout` / `estimate_factor` (the CPU port's origin).
- https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf , https://liacs.leidenuniv.nl/~takesfw/pdf/exploiting-gpus-fast.pdf -- Burtscher-Pingali Barnes-Hut and Brinkmann et al. timings / kernel shares.
- https://www2.evl.uic.edu/documents/pacificvisgraphwagu.pdf , https://stevepetruzza.io/pubs/graphwagu-2022.pdf -- GraphWaGu papers (timings, crossover).
- https://gpuweb.github.io/gpuweb/ , https://gpuweb.github.io/gpuweb/wgsl/ -- WebGPU / WGSL spec: limits, `requestDevice`, indirect dispatch, device lost, atomics (6.2.8), overrides, uniform layout, subgroups.
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status , https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ , https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/ , https://developer.chrome.com/blog/new-in-webgpu-{120,121,128,134} -- browser availability and feature timelines.
- https://registry.npmjs.org/webgpu , https://github.com/dawn-gpu/node-webgpu , https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md -- Dawn-node versions, `create()` options, lifetime.
- https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/.github/workflows/build.yml , https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/workflows/ci.yml , https://raw.githubusercontent.com/mrdoob/three.js/dev/.github/workflows/ci.yml -- lavapipe / SwiftShader CI precedents.
- https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/ , https://docs.github.com/en/actions/reference/runners/larger-runners , https://docs.github.com/en/billing/reference/actions-runner-pricing , https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions -- GPU runner availability, price, security guidance.
- https://cirun.io/ , https://runs-on.com/runners/gpu/ , https://machine.dev/docs/platform-specifications/gpu-runners/ -- third-party GPU runner fallbacks.
- https://vite.dev/guide/assets -- `?raw` (considered and rejected for WGSL).
- https://docs.npmjs.com/cli/v10/configuring-npm/package-json -- optional peer dependency semantics.

Not verified / not fetched in this plan (carried over from the notes): the GraphWaGu
Google-Drive paper PDF; cosmograph product pages; Naim et al. 2017 (paywalled); Firefox /
Safari exposure of `subgroups` / `timestamp-query`; lavapipe on an actual GitHub
runner; Dawn-node timestamp quantisation; the exact Vitest 3.2 per-instance launch
spelling; whether `webgpu@0.4.0` honours `backend=` (only `adapter=` was exercised);
RunsOn pricing. Every ESTIMATE in sections 7.19 and 10 is replaced by harness output
at the phase gates.
